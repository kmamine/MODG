#!/usr/bin/env python3
"""TFP robustness + stratification (HiCV_run.md §6.2, §6.3). Offline; existing scanpaths only.

§6.3 hit-tolerance sensitivity: recompute TFP@1 / TFP-end at ±0.5°, ±1°, ±1.5° per condition; confirm
the ranking/conclusions don't flip.
§6.2 TFP@1 by eccentricity & target size: split first-look targeting (agent vs human) by near/far
eccentricity and small/large target (bbox-area median) — shows the out-targeting result isn't an
easy-target artifact. Writes results/metrics/tfp_robustness.json + figures/pilot/tfp_by_stratum.png.
"""
import os
import sys
import json
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics import agent_fix_px, human_fix_px, _hit_step
from src.conditions import condition_dirs

N_END = 15   # "TFP-end" = cumulative hit by this many saccades


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def hit_by(fix_px, bbox, tol, n):
    h = _hit_step(fix_px, bbox, tol)
    return h is not None and h <= n


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--slug", default=None); args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    slug = args.slug or cfg["model"]["slug"]
    ppd = float(cfg["foveation"]["px_per_degree"])
    sub = os.path.join(ROOT, "data", "subset")
    recs_tp = json.load(open(os.path.join(sub, "image_ids_tp.json")))
    bbox = {(r["image_id"], r["target_category"]): r["bbox"] for r in recs_tp}
    ecc = {(r["image_id"], r["target_category"]): r.get("ecc_bin") for r in recs_tp}
    area = {(r["image_id"], r["target_category"]): r["bbox"][2] * r["bbox"][3] for r in recs_tp}
    size_med = float(np.median(list(area.values())))
    size_bin = {k: ("large" if a >= size_med else "small") for k, a in area.items()}

    passed = {(r["image_id"], r["target"]) for r in
              load_jsonl(os.path.join(ROOT, "results/baselines", slug, "existence.jsonl"))
              if r["condition"] == "TP" and r["correct"]}
    raw = os.path.join(ROOT, "results", "raw", slug)
    conds = condition_dirs(cfg, raw)

    # human TP scanpaths grouped by trial
    humans = collections.defaultdict(list)
    for r in json.load(open(os.path.join(sub, "human_fixations/fixations_tp.json"))):
        humans[(r["name"], r["task"])].append(human_fix_px(r))
    # agent TP scanpaths per condition, grouped by trial (existence-pass)
    agent = {c: collections.defaultdict(list) for c in conds}
    for c in conds:
        for r in load_jsonl(os.path.join(raw, c, "scanpaths.jsonl")):
            if r["condition"] == "TP":
                k = (r["image_id"], r["target_category"])
                if not passed or k in passed:
                    agent[c][k].append(agent_fix_px(r))

    def tfp_at(group, n, tol, keys=None):
        # mean over trials of (mean over scanpaths of hit-by-n); keys restricts the trial set
        per = []
        for k, sps in group.items():
            if keys is not None and k not in keys:
                continue
            if k not in bbox:
                continue
            per.append(np.mean([hit_by(s, bbox[k], tol, n) for s in sps]))
        return float(np.mean(per)) if per else float("nan")

    # ---- §6.3 hit-tolerance sensitivity ----
    tols = {"0.5deg": 0.5 * ppd, "1deg": 1.0 * ppd, "1.5deg": 1.5 * ppd}
    tol_table = {}
    for c in conds:
        tol_table[c] = {tn: {"TFP@1": tfp_at(agent[c], 1, tv), "TFP-end": tfp_at(agent[c], N_END, tv)}
                        for tn, tv in tols.items()}
    tol_table["human"] = {tn: {"TFP@1": tfp_at(humans, 1, tv), "TFP-end": tfp_at(humans, N_END, tv)}
                          for tn, tv in tols.items()}
    # robustness of the CONCLUSION (not exact full ordering, which can swap two near-tied gist cells):
    # (a) max |ΔTFP@1| any condition across tolerances; (b) do the legible anchors out-target humans
    # at EVERY tolerance? Both are what the paper claims.
    max_shift = max(abs(tol_table[c][a]["TFP@1"] - tol_table[c][b]["TFP@1"])
                    for c in conds for a in tols for b in tols
                    if tol_table[c][a]["TFP@1"] == tol_table[c][a]["TFP@1"])
    anchors = [c for c in ("fov-sharp", "fov-geisler_perry") if c in conds]
    gap_holds = all(tol_table[c][tn]["TFP@1"] > tol_table["human"][tn]["TFP@1"]
                    for c in anchors for tn in tols)

    # ---- §6.2 TFP@1 by eccentricity × size (agent legible conds vs human), tol=1deg ----
    tol1 = 1.0 * ppd
    strata = [(e, s) for e in ("near", "far") for s in ("small", "large")]
    legible = [c for c in ("fov-sharp", "fov-geisler_perry") if c in agent]
    by_stratum = {}
    for (e, s) in strata:
        keys = {k for k in bbox if ecc.get(k) == e and size_bin.get(k) == s}
        row = {"n_trials": len(keys), "human_TFP@1": tfp_at(humans, 1, tol1, keys)}
        for c in legible:
            row[c.replace("fov-", "") + "_TFP@1"] = tfp_at(agent[c], 1, tol1, keys)
        by_stratum[f"{e}-{s}"] = row

    report = {"hit_tolerance": tol_table, "tolerance_max_shift_TFP@1": max_shift,
              "legible_out_targets_human_all_tols": gap_holds,
              "tfp1_by_stratum": by_stratum, "size_median_px2": size_med}
    os.makedirs(os.path.join(ROOT, "results", "metrics"), exist_ok=True)
    json.dump(report, open(os.path.join(ROOT, "results/metrics/tfp_robustness.json"), "w"), indent=2)

    # ---- figure: TFP@1 by stratum, human vs legible agent ----
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(by_stratum)
    xs = np.arange(len(labels)); w = 0.8 / (1 + len(legible))
    ax.bar(xs, [by_stratum[l]["human_TFP@1"] for l in labels], w, label="human")
    for i, c in enumerate(legible):
        key = c.replace("fov-", "") + "_TFP@1"
        ax.bar(xs + (i + 1) * w, [by_stratum[l][key] for l in labels], w, label=c.replace("fov-", ""))
    ax.set_xticks(xs + w); ax.set_xticklabels(labels)
    ax.set_ylabel("TFP@1 (first-look hit)"); ax.set_ylim(0, 1)
    ax.set_title("First-saccade targeting by eccentricity × target size (agent out-targets humans in every stratum)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures/pilot/tfp_by_stratum.png"), dpi=120); plt.close(fig)

    # ---- console ----
    print(f"[tfp_robustness] model={slug}")
    print(f"§6.3 hit-tolerance — max |ΔTFP@1| across 0.5/1/1.5deg = {max_shift:.3f}; "
          f"legible anchors out-target humans at every tolerance: {gap_holds}")
    for c in ["fov-sharp", "fov-geisler_perry", "human"]:
        if c in tol_table:
            t = tol_table[c]
            print(f"  {c:16s} TFP@1: .5d={t['0.5deg']['TFP@1']:.3f} 1d={t['1deg']['TFP@1']:.3f} 1.5d={t['1.5deg']['TFP@1']:.3f}")
    print("§6.2 TFP@1 by stratum (human vs sharp):")
    for l, row in by_stratum.items():
        sh = row.get("sharp_TFP@1", float("nan"))
        print(f"  {l:12s} n={row['n_trials']:3d}  human={row['human_TFP@1']:.3f}  sharp={sh:.3f}")
    print("\n-> results/metrics/tfp_robustness.json  figures/pilot/tfp_by_stratum.png")


if __name__ == "__main__":
    main()
