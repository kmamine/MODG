#!/usr/bin/env python3
"""Materialize the frozen subset into a standalone, exportable folder.

Copies the selected stimulus images and the matching human fixation records (raw
COCO-Search18 schema, lossless) plus provenance files into data/subset/. The result
is self-contained: it can be zipped and used with no access to the full COCO-Search18
download or the COCO instance annotations. Reads only the frozen id lists + config.
"""
import os
import sys
import json
import shutil
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_subset import load_records, load_json, ROOT  # noqa: E402

import yaml  # noqa: E402

OUT = os.path.join(ROOT, "data", "subset")


def write_readme(cfg, summary):
    s = summary
    txt = f"""# COCO-Search18 — Stratified Working Subset (standalone)

Self-contained subset for the GazeAgent human-vs-MLLM visual-exploration study.
Everything needed to run the experiment on this subset is in this folder — no access
to the full COCO-Search18 download or the COCO annotations is required.

## Contents
```
images/TP/<category>/<id>.jpg          {s['TP']['images']} target-present stimulus images
images/TA/<category>/<id>.jpg          {s['TA']['images']} target-absent stimulus images
human_fixations/fixations_tp.json      human scanpaths for the TP images (all subjects)
human_fixations/fixations_ta.json      human scanpaths for the TA images (all subjects)
image_ids_tp.json / image_ids_ta.json  frozen id lists + per-image strata (ecc / clutter)
subset_manifest.json                   seed, thresholds, shortfalls — how it was drawn
config.yaml                            the knobs used to build it
```

## Counts
- TP: {s['TP']['images']} images, {s['TP']['fixation_records']} human scanpaths
- TA: {s['TA']['images']} images, {s['TA']['fixation_records']} human scanpaths
- {cfg['subset']['require_subjects']} human subjects per image.

## Coordinate space (IMPORTANT)
All gaze was recorded on images resized to {cfg['canonical_size']['w']}x{cfg['canonical_size']['h']}
(zero-padded, aspect ratio preserved). Fixation `X`/`Y` are pixels in that frame.
Normalize to [0,1] as `x / {cfg['canonical_size']['w']}`, `y / {cfg['canonical_size']['h']}`.

## human_fixations schema (raw COCO-Search18, lossless; one record per image x subject)
`name`, `subject`, `task` (target category), `condition` (present|absent),
`bbox` [x,y,w,h] (TP only — the target), `X`[], `Y`[], `T`[] (fixation x / y / duration
sequences, incl. the initial centre fixation), `length`, `correct`, `RT` (TP) /
`fixOnTarget` (TA), `split`.

## image_ids_* schema (one record per selected image)
`image_id`, `coco_id`, `target_category`, `condition`, `split`, `n_subjects`,
`object_count`, `clutter_bin` (low|high), `stratum`; TP additionally has `bbox`,
`eccentricity`, `ecc_bin`.

## Provenance
- Pool: **{cfg['subset']['pool']}** split of COCO-Search18 (test-split fixations are
  withheld upstream, so validation is the held-out evaluation set).
- Seed: {cfg['seed']}. Clutter = number of COCO-2017 instance annotations per image.
- Stratification: TP = category x eccentricity x clutter; TA = category x clutter;
  thresholds are per-category medians.
- Regenerate: `python scripts/build_subset.py && python scripts/export_subset.py`.

## Cite the source dataset
Yang, Z., Huang, L., Chen, Y., Wei, Z., Ahn, S., Zelinsky, G., Samaras, D., & Hoai, M.
(2020). Predicting Goal-directed Human Attention Using Inverse Reinforcement Learning. CVPR.
"""
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write(txt)


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    pool = cfg["subset"]["pool"]
    require = cfg["subset"]["require_subjects"]
    images_dir = os.path.join(ROOT, cfg["paths"]["images_dir"])
    fix_dir = os.path.join(ROOT, cfg["paths"]["fixations_dir"])

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "human_fixations"))

    summary = {}
    for cond, ids_key in (("TP", "ids_tp"), ("TA", "ids_ta")):
        sel = load_json(os.path.join(ROOT, cfg["paths"][ids_key]))
        sel_keys = {(m["target_category"], m["image_id"]) for m in sel}

        # copy stimulus images
        for m in sel:
            src = os.path.join(images_dir, cond, m["target_category"], m["image_id"])
            dst = os.path.join(OUT, "images", cond, m["target_category"], m["image_id"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

        # filter raw fixations to the selected images (all subjects), lossless schema
        recs = load_records(fix_dir, cond, pool)
        kept = [r for r in recs if (r["task"], r["name"]) in sel_keys]
        kept.sort(key=lambda r: (r["task"], r["name"], r["subject"]))
        with open(os.path.join(OUT, "human_fixations", f"fixations_{cond.lower()}.json"), "w") as f:
            json.dump(kept, f, indent=1)

        # standalone-integrity asserts
        cnt = collections.Counter((r["task"], r["name"]) for r in kept)
        assert set(cnt) == sel_keys, f"{cond}: fixation/image key mismatch"
        assert all(v == require for v in cnt.values()), f"{cond}: not all images have {require} subjects"
        copied = sum(len(files) for _, _, files in os.walk(os.path.join(OUT, "images", cond)))
        assert copied == len(sel), f"{cond}: copied {copied} images, expected {len(sel)}"
        summary[cond] = {"images": len(sel), "fixation_records": len(kept), "subjects_per_image": require}

    # provenance copies
    for k in ("ids_tp", "ids_ta", "manifest"):
        shutil.copy2(os.path.join(ROOT, cfg["paths"][k]),
                     os.path.join(OUT, os.path.basename(cfg["paths"][k])))
    shutil.copy2(os.path.join(ROOT, "config.yaml"), os.path.join(OUT, "config.yaml"))

    write_readme(cfg, summary)

    print("Standalone subset written to", OUT)
    for c, s in summary.items():
        print(f"  {c}: {s}")


if __name__ == "__main__":
    main()
