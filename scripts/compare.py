#!/usr/bin/env python3
"""Detailed agent↔human scanpath comparison (offline, on existing pilot scanpaths).

For each foveation condition, on target-present images the model passed the existence baseline:
  - intrinsic spatial signature: per-scanpath distributions (saccade amplitude, turning angle,
    scanpath length, gaze entropy, refixation rate, center bias) — agent vs the human distribution
    (effect size Cliff's delta + medians + bootstrap CIs);
  - scanpath similarity: ScanMatch / DTW / Hausdorff / Fréchet / MultiMatch, computed three ways —
    agent↔human, human↔human (the ceiling), agent↔agent (self-consistency);
  - a headline PCA embedding of {human, each condition} on a shared metric vector (interpretable axes).
Writes results/metrics/comparison.json + figures/pilot/{intrinsic_signature,scanpath_similarity,
pca}.png. No model calls. Cross-MODEL comparison is in scripts/compare_models.py.
"""
import os
import sys
import json
import collections
from itertools import combinations, product

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")

from src.metrics import (agent_fix_px, human_fix_px, numfix, saccade_amplitudes, turning_angles,
                         scanpath_length, gaze_entropy, refixation_rate, center_bias)
from src import scanpath_metrics as spm
from src.stats import cluster_bootstrap_ci, cliffs_delta_ci, pca_embed, pc_axis_label
from src.conditions import condition_dirs

PPD = 30.0   # reassigned from config in main(); the SIG lambdas read this global at call time
# per-scanpath signature metrics (scalar per scanpath) for distributions + the MDS vector
SIG = {"scanpath_length": lambda f: scanpath_length(f, PPD),
       "median_saccade_amp": lambda f: float(np.median(saccade_amplitudes(f, PPD))) if len(f) > 1 else np.nan,
       "gaze_entropy": gaze_entropy, "refixation_rate": refixation_rate,
       "center_bias": lambda f: center_bias(f, PPD),
       "median_turn_angle": lambda f: float(np.median(np.abs(turning_angles(f)))) if len(f) > 2 else np.nan,
       "numfix": lambda f: float(len(f))}
# pairwise similarity (higher=more alike for scanmatch/mm; lower=more alike for dtw/hausdorff/frechet)
SIM = {"scanmatch": spm.scanmatch_score, "dtw": spm.dtw_distance, "hausdorff": spm.hausdorff_distance,
       "frechet": spm.frechet_distance, "mm_position": lambda a, b: spm.multimatch(a, b)["mm_position"]}


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def group_humans(p):
    g = collections.defaultdict(list)
    for r in json.load(open(p)):
        g[(r["name"], r["task"])].append(human_fix_px(r))   # trial = (image, target)
    return g


def pair_sims(a, b):
    out = {}
    for k, fn in SIM.items():
        try:
            out[k] = float(fn(a, b))
        except Exception:
            out[k] = np.nan
    return out


def nanmean_over(dicts):
    keys = SIM.keys()
    return {k: float(np.nanmean([d[k] for d in dicts])) if dicts else np.nan for k in keys}


def main():
    global PPD
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default=None, help="analyse this model slug instead of config.model.slug "
                                                 "(regenerate a figure without touching the live config)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    PPD = float(cfg["foveation"]["px_per_degree"])       # SIG lambdas read this global at call time
    slug = args.slug or cfg["model"]["slug"]
    sub = os.path.join(ROOT, "data", "subset")
    humans = group_humans(os.path.join(sub, "human_fixations/fixations_tp.json"))
    passed = {(r["image_id"], r["target"]) for r in load_jsonl(os.path.join(ROOT, "results/baselines", slug, "existence.jsonl"))
              if r["condition"] == "TP" and r["correct"]}
    raw = os.path.join(ROOT, "results", "raw", slug)
    conds = condition_dirs(cfg, raw)                     # config-driven bracket (excludes temp-sweep dirs)
    figdir = os.path.join(ROOT, "figures", "pilot"); os.makedirs(figdir, exist_ok=True)

    # ---- human reference: signature distribution (+ image cluster ids) + human↔human ceiling ----
    human_sig = {m: [] for m in SIG}
    human_clusters = []                                  # image id per human scanpath (for clustered CIs)
    hh = []                                              # human↔human similarity, pooled over images
    for iid, hs in humans.items():
        for f in hs:
            for m, fn in SIG.items():
                human_sig[m].append(fn(f))
            human_clusters.append(iid)
        for a, b in combinations(hs, 2):
            hh.append(pair_sims(a, b))
    ceiling = nanmean_over(hh)
    print(f"[compare] model={slug}  conditions={conds}  TP existence-pass images={len(passed)}")

    report = {"_human": {"signature_median": {m: float(np.nanmedian(human_sig[m])) for m in SIG},
                         "similarity_ceiling(human-human)": ceiling}}
    agent_sig_by_cond, sim_ah_by_cond, sim_aa_by_cond = {}, {}, {}

    for cond in conds:
        recs = [r for r in load_jsonl(os.path.join(raw, cond, "scanpaths.jsonl"))
                if r["condition"] == "TP" and (not passed or (r["image_id"], r["target_category"]) in passed)]
        by_img = collections.defaultdict(list)
        for r in recs:
            by_img[(r["image_id"], r["target_category"])].append(agent_fix_px(r))

        sig = {m: [] for m in SIG}
        clusters = []                                    # image id per agent scanpath
        ah, aa = [], []
        for iid, aps in by_img.items():
            for f in aps:
                for m, fn in SIG.items():
                    sig[m].append(fn(f))
                clusters.append(iid)
            if iid in humans:                            # agent↔human
                for a, h in product(aps, humans[iid]):
                    ah.append(pair_sims(a, h))
            for a, b in combinations(aps, 2):            # agent↔agent (across seeds)
                aa.append(pair_sims(a, b))

        agent_sig_by_cond[cond] = sig
        sim_ah_by_cond[cond] = nanmean_over(ah)
        sim_aa_by_cond[cond] = nanmean_over(aa)
        # effect size vs human with an IMAGE-clustered bootstrap CI (Task 5): resamples images, so the
        # CI respects seed×subject dependence rather than treating every scanpath pair as independent.
        report[cond] = {
            "n_scanpaths": int(sum(len(v) for v in by_img.values())),
            "signature": {m: {"median": float(np.nanmedian(sig[m])),
                              "ci": list(cluster_bootstrap_ci(sig[m], clusters)),
                              "delta_vs_human": cliffs_delta_ci(sig[m], clusters, human_sig[m], human_clusters)}
                          for m in SIG},
            "similarity": {"agent_human": sim_ah_by_cond[cond], "agent_agent": sim_aa_by_cond[cond]},
        }

    json.dump(report, open(os.path.join(ROOT, "results/metrics/comparison.json"), "w"),
              indent=2, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else float(o))

    # ================= FIGURES =================
    labels = [c.replace("fov-", "") for c in conds]
    # 1. intrinsic signature: box per condition vs human band, one panel per metric
    panels = ["scanpath_length", "median_saccade_amp", "gaze_entropy", "refixation_rate",
              "center_bias", "median_turn_angle"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, m in zip(axes.ravel(), panels):
        data = [[v for v in agent_sig_by_cond[c][m] if not np.isnan(v)] for c in conds]
        hv = [v for v in human_sig[m] if not np.isnan(v)]
        ax.boxplot(data + [hv], labels=labels + ["HUMAN"], showfliers=False)
        ax.axhline(np.nanmedian(human_sig[m]), color="k", ls="--", lw=1, alpha=.6)
        ax.set_title(m); ax.tick_params(axis="x", rotation=35, labelsize=8)
    fig.suptitle("Intrinsic scanpath signature — agent conditions vs human (dashed = human median)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "intrinsic_signature.png"), dpi=120); plt.close(fig)

    # 2. scanpath similarity: agent↔human vs human↔human ceiling vs agent↔agent
    sims_to_plot = ["scanmatch", "dtw", "hausdorff", "mm_position"]
    fig, axes = plt.subplots(1, len(sims_to_plot), figsize=(16, 4.5))
    xs = np.arange(len(conds)); w = 0.27
    for ax, k in zip(axes, sims_to_plot):
        ax.bar(xs - w, [sim_ah_by_cond[c][k] for c in conds], w, label="agent↔human")
        ax.bar(xs, [sim_aa_by_cond[c][k] for c in conds], w, label="agent↔agent")
        ax.axhline(ceiling[k], color="k", ls="--", lw=1.5, label="human↔human (ceiling)")
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=35, fontsize=7)
        pol = "higher=more alike" if k in ("scanmatch", "mm_position") else "lower=more alike"
        ax.set_title(f"{k}\n({pol})", fontsize=9)
    axes[0].legend(fontsize=7)
    fig.suptitle("Scanpath similarity (target-present)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "scanpath_similarity.png"), dpi=120); plt.close(fig)

    # 3. PCA: {human, each condition} on the shared 5-metric signature. PCA (not MDS) so the axes
    #    are interpretable -- each is a named linear combo of the metrics (shown in the axis labels).
    feats = ["scanpath_length", "median_saccade_amp", "gaze_entropy", "refixation_rate", "center_bias"]
    groups = ["human"] + conds
    vec = [[np.nanmedian(human_sig[m]) for m in feats]]
    for c in conds:
        vec.append([np.nanmedian(agent_sig_by_cond[c][m]) for m in feats])
    coords, loadings, var = pca_embed(np.array(vec))
    dfp = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "group": groups})
    hdf, cdf = dfp[dfp.group == "human"], dfp[dfp.group != "human"]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.scatterplot(data=cdf, x="x", y="y", s=170, color="#4C72B0", edgecolor="white",
                    linewidth=1.2, alpha=.92, zorder=3, legend=False, ax=ax)
    ax.scatter(hdf.x, hdf.y, marker="*", s=900, color="#333333", edgecolor="white", linewidth=1.5, zorder=6)
    for _, r in dfp.iterrows():
        is_h = r["group"] == "human"
        ax.annotate("HUMAN" if is_h else r["group"].replace("fov-", ""), (r.x, r.y),
                    fontsize=13 if is_h else 9.5, fontweight="bold" if is_h else "normal",
                    alpha=1 if is_h else .65, xytext=(6, 5), textcoords="offset points")
    ax.set_title("Per-group scanpath-signature space (PCA)", fontsize=14, fontweight="bold")
    ax.set_xlabel(pc_axis_label(1, loadings[0], feats, var[0]), fontsize=11)
    ax.set_ylabel(pc_axis_label(2, loadings[1], feats, var[1]), fontsize=11)
    ax.margins(0.18); sns.despine(ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "pca.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

    # ---- console summary ----
    print("\n=== SCANPATH SIMILARITY (ScanMatch, higher=more alike) ===")
    print(f"  human↔human ceiling: {ceiling['scanmatch']:.3f}")
    for c in conds:
        print(f"  {c:18s} agent↔human={sim_ah_by_cond[c]['scanmatch']:.3f}  agent↔agent={sim_aa_by_cond[c]['scanmatch']:.3f}")
    print("\n=== INTRINSIC SIGNATURE (Cliff's delta vs human [image-clustered CI]; |d|>0.33 = non-trivial) ===")
    for c in conds:
        ds = {m: report[c]['signature'][m]['delta_vs_human'] for m in SIG}
        print(f"  {c:18s} " + "  ".join(f"{m.split('_')[0]}:{ds[m]['delta']:+.2f}[{ds[m]['lo']:+.2f},{ds[m]['hi']:+.2f}]" for m in SIG))
    print(f"\nfigures -> {figdir}/  json -> results/metrics/comparison.json")


if __name__ == "__main__":
    main()
