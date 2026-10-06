#!/usr/bin/env python3
"""Temperature-sweep self-consistency (Task 4). Offline; reads fov-<core>-t<T> scanpath dirs.

The headline self-consistency finding ("the model agrees with itself across seeds far more than
humans agree with each other") currently rests on one temperature. This recomputes agent↔agent
scanpath similarity (ScanMatch / DTW) per temperature on the core conditions, against the fixed
human↔human ceiling, and asks whether consistency erodes as temperature rises. Writes
results/metrics/temp_sweep.json + figures/pilot/selfconsistency_temp.png.
"""
import os
import sys
import re
import json
import collections
from itertools import combinations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics import agent_fix_px
from compare import pair_sims, nanmean_over, group_humans, load_jsonl  # reuse the comparison core

TEMP_RE = re.compile(r"^(fov-.+)-t([0-9.]+)$")


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    slug = cfg["model"]["slug"]
    raw = os.path.join(ROOT, "results", "raw", slug)
    figdir = os.path.join(ROOT, "figures", "pilot"); os.makedirs(figdir, exist_ok=True)

    temp_dirs = sorted(d for d in os.listdir(raw) if TEMP_RE.match(d)) if os.path.isdir(raw) else []
    if not temp_dirs:
        print(f"[temp_sweep] no fov-*-t* dirs under {raw}; run the temperature sweep first.")
        return

    # human↔human ceiling (same construction as compare.py)
    humans = group_humans(os.path.join(ROOT, "data", "subset", "human_fixations/fixations_tp.json"))
    passed = {r["image_id"] for r in load_jsonl(os.path.join(ROOT, "results/baselines", slug, "existence.jsonl"))
              if r["condition"] == "TP" and r["correct"]}
    hh = [pair_sims(a, b) for hs in humans.values() for a, b in combinations(hs, 2)]
    ceiling = nanmean_over(hh)

    # agent↔agent (across seeds) per (core condition, temperature)
    rows = {}                                            # (core, temp) -> {scanmatch, dtw, ...} + n
    for d in temp_dirs:
        core, temp = TEMP_RE.match(d).groups()
        recs = [r for r in load_jsonl(os.path.join(raw, d, "scanpaths.jsonl"))
                if r["condition"] == "TP" and (not passed or r["image_id"] in passed)]
        by_img = collections.defaultdict(list)
        for r in recs:
            by_img[r["image_id"]].append(agent_fix_px(r))
        aa = [pair_sims(a, b) for aps in by_img.values() for a, b in combinations(aps, 2)]
        rows[(core, float(temp))] = {**nanmean_over(aa), "n_pairs": len(aa),
                                     "n_scanpaths": sum(len(v) for v in by_img.values())}

    cores = sorted({c for c, _ in rows})
    temps = sorted({t for _, t in rows})
    report = {"ceiling_human_human": ceiling,
              "agent_agent": {f"{c}@t{t}": rows[(c, t)] for (c, t) in rows}}
    os.makedirs(os.path.join(ROOT, "results", "metrics"), exist_ok=True)
    json.dump(report, open(os.path.join(ROOT, "results/metrics/temp_sweep.json"), "w"),
              indent=2, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else float(o))

    # figure: ScanMatch + DTW self-consistency vs temperature, with the human↔human ceiling
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, pol in [(axes[0], "scanmatch", "higher = more self-consistent"),
                         (axes[1], "dtw", "lower = more self-consistent")]:
        for c in cores:
            ys = [rows[(c, t)][key] for t in temps]
            ax.plot(temps, ys, "o-", label=c.replace("fov-", ""))
        ax.axhline(ceiling[key], color="k", ls="--", lw=1.5, label="human↔human ceiling")
        ax.set_xlabel("temperature"); ax.set_ylabel(f"agent↔agent {key}")
        ax.set_title(f"Self-consistency vs temperature — {key}\n({pol})", fontsize=9)
        ax.set_xticks(temps)
    axes[0].legend(fontsize=8)
    fig.suptitle("Self-consistency across seeds vs temperature (target-present)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "selfconsistency_temp.png"), dpi=120); plt.close(fig)

    # console table
    print(f"[temp_sweep] human↔human ceiling: ScanMatch={ceiling['scanmatch']:.3f}  DTW={ceiling['dtw']:.1f}")
    print(f"{'condition':18s} {'temp':>5s} {'ScanMatch(a-a)':>15s} {'DTW(a-a)':>10s} {'n_pairs':>8s}")
    for c in cores:
        for t in temps:
            r = rows[(c, t)]
            print(f"{c.replace('fov-',''):18s} {t:5.1f} {r['scanmatch']:15.3f} {r['dtw']:10.1f} {r['n_pairs']:8d}")
    print(f"\n-> figures/pilot/selfconsistency_temp.png  results/metrics/temp_sweep.json")


if __name__ == "__main__":
    main()
