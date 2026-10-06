#!/usr/bin/env python3
"""Stratified subset sampler for COCO-Search18 (build-plan step 1).

Reads config.yaml, writes a frozen, balanced image-id list per condition plus a
manifest documenting how it was produced. Deterministic given the seed: re-running
yields byte-identical id files. All downstream code reads these lists, never the
raw fixation dump.

Strata (CLAUDE.md):
  TP: category x eccentricity{near,far} x clutter{low,high}  -> 4 cells/category
  TA: category x clutter{low,high}                           -> 2 cells/category
Thresholds are per-category medians. Clutter = #COCO object instances in the image.
"""
import os
import json
import math
import statistics
import collections

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TP_CELLS = ["near-low", "near-high", "far-low", "far-high"]
TA_CELLS = ["low", "high"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def coco_id(name):
    return int(os.path.splitext(name)[0])


def eccentricity(bbox, W, H):
    """Distance of target-bbox centre from image centre in canonical normalized coords."""
    x, y, w, h = bbox
    return math.hypot((x + w / 2) / W - 0.5, (y + h / 2) / H - 0.5)


def build_object_counts(ann_dir):
    """coco image_id -> number of annotated object instances (iscrowd counts as 1)."""
    count = collections.Counter()
    for fn in ("instances_train2017.json", "instances_val2017.json"):
        d = load_json(os.path.join(ann_dir, fn))
        for a in d["annotations"]:
            count[a["image_id"]] += 1
    return count


def load_records(fix_dir, condition, pool):
    """Return the raw human-fixation records for the chosen pool."""
    if condition == "TP":
        files = ["coco_search18_fixations_TP_validation_split1.json"]
        if pool == "train+val":
            files.append("coco_search18_fixations_TP_train_split1.json")
        recs = []
        for fn in files:
            recs += load_json(os.path.join(fix_dir, fn))
        return recs
    recs = load_json(os.path.join(fix_dir, "coco_search18_fixations_TA_trainval.json"))
    if pool == "validation":
        recs = [r for r in recs if r["split"] == "valid"]
    return recs


def build_metas(records, condition, cfg):
    """Aggregate per-subject records to one meta dict per (category, image).

    Keeps only images with exactly require_subjects scanpaths; asserts the image
    file exists on disk. Returns (metas, n_dropped_for_subject_count)."""
    W, H = cfg["canonical_size"]["w"], cfg["canonical_size"]["h"]
    counts = build_object_counts(os.path.join(ROOT, cfg["paths"]["coco_annotations_dir"]))
    images_dir = os.path.join(ROOT, cfg["paths"]["images_dir"])
    require = cfg["subset"]["require_subjects"]

    grouped = collections.defaultdict(list)
    for r in records:
        grouped[(r["task"], r["name"])].append(r)

    metas, dropped = [], 0
    for (task, name), recs in grouped.items():
        if len(recs) != require:
            dropped += 1
            continue
        img_path = os.path.join(images_dir, condition, task, name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"image referenced by fixations but missing: {img_path}")
        m = {
            "image_id": name,
            "coco_id": coco_id(name),
            "target_category": task,
            "condition": condition,
            "split": recs[0].get("split", ""),
            "n_subjects": len(recs),
            "object_count": int(counts[coco_id(name)]),
        }
        if condition == "TP":
            m["bbox"] = list(recs[0]["bbox"])
            m["eccentricity"] = round(eccentricity(m["bbox"], W, H), 6)
        metas.append(m)
    return metas, dropped


def stratify_and_sample(metas, condition, cfg):
    """Per-category median binning + seeded balanced per-cell sampling."""
    per_cat = cfg["subset"]["per_category_tp" if condition == "TP" else "per_category_ta"]
    backfill = cfg["subset"]["backfill"]
    cells = TP_CELLS if condition == "TP" else TA_CELLS
    per_cell = per_cat // len(cells)
    rng = np.random.default_rng(cfg["seed"])

    by_cat = collections.defaultdict(list)
    for m in metas:
        by_cat[m["target_category"]].append(m)

    selected, shortfalls, thresholds = [], [], {}
    for cat in sorted(by_cat):
        items = by_cat[cat]
        med_clutter = statistics.median(m["object_count"] for m in items)
        thr = {"n_pool": len(items), "clutter_median": med_clutter}
        if condition == "TP":
            med_ecc = statistics.median(m["eccentricity"] for m in items)
            thr["ecc_median"] = round(med_ecc, 6)
        thresholds[cat] = thr

        # assign bins -> stratum label
        for m in items:
            m["clutter_bin"] = "low" if m["object_count"] <= med_clutter else "high"
            if condition == "TP":
                m["ecc_bin"] = "near" if m["eccentricity"] <= med_ecc else "far"
                m["stratum"] = f"{m['ecc_bin']}-{m['clutter_bin']}"
            else:
                m["stratum"] = m["clutter_bin"]

        cell_map = collections.defaultdict(list)
        for m in items:
            cell_map[m["stratum"]].append(m)

        chosen_names = set()
        for ck in cells:
            pool = sorted(cell_map.get(ck, []), key=lambda m: m["image_id"])
            take = min(per_cell, len(pool))
            if len(pool) < per_cell:
                shortfalls.append({"condition": condition, "category": cat, "cell": ck,
                                   "wanted": per_cell, "available": len(pool)})
            if take:
                idx = sorted(rng.choice(len(pool), size=take, replace=False).tolist())
                for i in idx:
                    selected.append(pool[i])
                    chosen_names.add(pool[i]["image_id"])

        if backfill and len(chosen_names) < per_cat:
            extra_pool = sorted((m for m in items if m["image_id"] not in chosen_names),
                                key=lambda m: m["image_id"])
            need = min(per_cat - len(chosen_names), len(extra_pool))
            if need:
                idx = sorted(rng.choice(len(extra_pool), size=need, replace=False).tolist())
                for i in idx:
                    selected.append(extra_pool[i])

    # deterministic on-disk ordering
    selected.sort(key=lambda m: (m["target_category"], m["stratum"], m["image_id"]))
    return selected, shortfalls, thresholds


def print_table(selected, condition, thresholds):
    cells = TP_CELLS if condition == "TP" else TA_CELLS
    counts = collections.Counter((m["target_category"], m["stratum"]) for m in selected)
    header = f"{'category':14s} " + " ".join(f"{c:>9s}" for c in cells) + f" {'tot':>4s}"
    print(f"\n=== {condition} selection ({len(selected)} images) ===")
    print(header)
    for cat in sorted(thresholds):
        row = [counts[(cat, c)] for c in cells]
        print(f"{cat:14s} " + " ".join(f"{n:9d}" for n in row) + f" {sum(row):4d}")


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    fix_dir = os.path.join(ROOT, cfg["paths"]["fixations_dir"])
    pool = cfg["subset"]["pool"]

    out = {}
    manifest = {"seed": cfg["seed"], "config": cfg, "pool": pool,
                "thresholds": {}, "shortfalls": [], "dropped_wrong_subject_count": {},
                "counts": {}}

    for condition, ids_key in (("TP", "ids_tp"), ("TA", "ids_ta")):
        records = load_records(fix_dir, condition, pool)
        metas, dropped = build_metas(records, condition, cfg)
        selected, shortfalls, thresholds = stratify_and_sample(metas, condition, cfg)

        # --- assertions (step-1 gate) ---
        require = cfg["subset"]["require_subjects"]
        assert all(m["n_subjects"] == require for m in selected), "subject-count invariant broken"
        assert len({(m["target_category"], m["image_id"]) for m in selected}) == len(selected), \
            "duplicate (category, image) in selection"

        out_path = os.path.join(ROOT, cfg["paths"][ids_key])
        with open(out_path, "w") as f:
            json.dump(selected, f, indent=2, sort_keys=True)

        print_table(selected, condition, thresholds)
        print(f"  pool={pool}  candidate_images={len(metas)}  dropped(!=10 subj)={dropped}"
              f"  shortfall_cells={len(shortfalls)}")

        manifest["thresholds"][condition] = thresholds
        manifest["shortfalls"] += shortfalls
        manifest["dropped_wrong_subject_count"][condition] = dropped
        manifest["counts"][condition] = {"candidate_images": len(metas), "selected": len(selected)}
        out[condition] = (out_path, len(selected))

    with open(os.path.join(ROOT, cfg["paths"]["manifest"]), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print("\nWrote:")
    for cond, (p, n) in out.items():
        print(f"  {p}  ({n} images)")
    print(f"  {os.path.join(ROOT, cfg['paths']['manifest'])}")
    if manifest["shortfalls"]:
        print(f"\nNOTE: {len(manifest['shortfalls'])} cell(s) under-filled (logged in manifest). "
              f"Set subset.pool: train+val for fuller strata.")


if __name__ == "__main__":
    main()
