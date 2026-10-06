#!/usr/bin/env python3
"""Cross-model comparison (offline, no model calls).

Headline question (CLAUDE.md): do the MLLMs cluster apart from the human reference -- and from each
other? Places {human, each model x each foveation condition} in ONE embedding of the 5-metric
scanpath signature. Uses PCA (not MDS) so the axes are interpretable: each principal component is a
fixed, named linear combination of the metrics (reported as the axis label + in the JSON loadings).

Reads results/raw/<slug>/ for every slug in config analysis.compare_slugs (or auto-discovers every
model dir under results/raw/); partial runs contribute whatever conditions exist. Writes
figures/pilot/{pca_models,similarity_models}.png + results/metrics/model_comparison.json.
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
PAL = {"Qwen3.5-35B": "#4C72B0", "GLM-4.6V-Flash": "#DD8452", "Gemma-4-E4B": "#55A868"}
HUMANC = "#333333"

from src.metrics import (agent_fix_px, human_fix_px, saccade_amplitudes, scanpath_length,
                         gaze_entropy, refixation_rate, center_bias)
from src import scanpath_metrics as spm
from src.stats import pca_embed, pc_axis_label
from src.conditions import condition_dirs

PPD = 30.0                                              # reassigned from config in main()
FEATS = ["scanpath_length", "median_saccade_amp", "gaze_entropy", "refixation_rate", "center_bias"]
SIG = {"scanpath_length": lambda f: scanpath_length(f, PPD),
       "median_saccade_amp": lambda f: float(np.median(saccade_amplitudes(f, PPD))) if len(f) > 1 else np.nan,
       "gaze_entropy": gaze_entropy, "refixation_rate": refixation_rate,
       "center_bias": lambda f: center_bias(f, PPD)}
LABELS = {"qwen3.5-35b-a3b": "Qwen3.5-35B", "glm-4.6v-flash": "GLM-4.6V-Flash",
          "gemma-4-e4b-it": "Gemma-4-E4B"}
SAMPLE_IMAGES = 40                                      # cap images/condition for the ScanMatch sample
SAMPLE_HUMANS = 4                                       # cap human scanpaths/image in the sample


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def label(slug):
    return LABELS.get(slug, slug)


def group_humans(p):
    g = collections.defaultdict(list)
    for r in json.load(open(p)):
        g[(r["name"], r["task"])].append(human_fix_px(r))   # trial = (image, target)
    return g


def med_sig(scanpaths):
    """Median signature vector (in FEATS order) over a list of fixation arrays."""
    return [float(np.nanmedian([SIG[m](f) for f in scanpaths])) for m in FEATS]


def main():
    global PPD
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    PPD = float(cfg["foveation"]["px_per_degree"])
    sub = os.path.join(ROOT, "data", "subset")
    humans = group_humans(os.path.join(sub, "human_fixations/fixations_tp.json"))
    rawroot = os.path.join(ROOT, "results", "raw")
    figdir = os.path.join(ROOT, "figures", "pilot"); os.makedirs(figdir, exist_ok=True)

    slugs = (cfg.get("analysis") or {}).get("compare_slugs") or \
            sorted(d for d in os.listdir(rawroot) if os.path.isdir(os.path.join(rawroot, d)))
    slugs = [s for s in slugs if os.path.isdir(os.path.join(rawroot, s))
             and condition_dirs(cfg, os.path.join(rawroot, s))]
    if not slugs:
        print("[compare_models] no model dirs with bracket conditions found; nothing to do."); return

    # ---- human reference: signature over all human TP scanpaths + human<->human ScanMatch ceiling ----
    hflat = [f for hs in humans.values() for f in hs]
    human_vec = med_sig(hflat)
    hh = [spm.scanmatch_score(a, b) for hs in humans.values() for a, b in combinations(hs[:SAMPLE_HUMANS], 2)]
    ceiling = float(np.nanmean(hh)) if hh else np.nan

    groups = [("human", None, human_vec)]               # (slug, cond, signature-vector)
    sims = {}                                           # slug -> mean agent<->human ScanMatch
    report = {"_human": {"signature_median": dict(zip(FEATS, human_vec)),
                         "scanmatch_ceiling_human_human": ceiling}}

    for slug in slugs:
        raw = os.path.join(rawroot, slug)
        passed = {(r["image_id"], r["target"]) for r in
                  load_jsonl(os.path.join(ROOT, "results/baselines", slug, "existence.jsonl"))
                  if r["condition"] == "TP" and r["correct"]}
        report[slug] = {"label": label(slug), "by_condition": {}}
        ah_per_cond = []
        for cond in condition_dirs(cfg, raw):
            recs = [r for r in load_jsonl(os.path.join(raw, cond, "scanpaths.jsonl"))
                    if r["condition"] == "TP" and (not passed or (r["image_id"], r["target_category"]) in passed)]
            by_img = collections.defaultdict(list)
            for r in recs:
                by_img[(r["image_id"], r["target_category"])].append(agent_fix_px(r))
            aps = [f for v in by_img.values() for f in v]
            if not aps:
                continue
            groups.append((slug, cond, med_sig(aps)))
            # agent<->human ScanMatch, sampled (bounded) for speed
            ah = []
            for i, (iid, ap_list) in enumerate(by_img.items()):
                if i >= SAMPLE_IMAGES:
                    break
                if iid in humans:
                    for a, h in product(ap_list[:2], humans[iid][:SAMPLE_HUMANS]):
                        ah.append(spm.scanmatch_score(a, h))
            m = float(np.nanmean(ah)) if ah else np.nan
            ah_per_cond.append(m)
            report[slug]["by_condition"][cond] = {"signature_median": dict(zip(FEATS, groups[-1][2])),
                                                  "scanmatch_agent_human": m}
        sims[slug] = float(np.nanmean(ah_per_cond)) if ah_per_cond else np.nan
        report[slug]["scanmatch_agent_human_mean"] = sims[slug]

    model_slugs = [s for s in slugs if any(g[0] == s for g in groups)]

    # ---- PCA of all group signature vectors (interpretable axes) ----
    vecs = np.array([g[2] for g in groups], float)
    # guard: replace any all-NaN column entry with the column mean so PCA never sees NaN
    col_mean = np.nanmean(vecs, axis=0)
    vecs = np.where(np.isnan(vecs), col_mean, vecs)
    coords, loadings, var = pca_embed(vecs)
    report["_pca"] = {"feature_order": FEATS,
                      "explained_variance_ratio": [float(v) for v in var],
                      "loadings": {f"PC{k+1}": dict(zip(FEATS, [float(x) for x in loadings[k]]))
                                   for k in range(len(loadings))}}
    json.dump(report, open(os.path.join(ROOT, "results/metrics/model_comparison.json"), "w"),
              indent=2, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else float(o))

    # ---- figure 1: PCA scatter (seaborn), colored by model, human = star ----
    hue_order = [label(s) for s in model_slugs]
    dfp = pd.DataFrame([{"model": label(slug) if slug != "human" else "human",
                         "cond": (cond or "").replace("fov-", ""), "x": float(x), "y": float(y)}
                        for (slug, cond, _), (x, y) in zip(groups, coords)])
    mdf, hdf = dfp[dfp.model != "human"], dfp[dfp.model == "human"]
    fig, ax = plt.subplots(figsize=(10.5, 8))
    sns.scatterplot(data=mdf, x="x", y="y", hue="model", hue_order=hue_order, palette=PAL,
                    s=210, edgecolor="white", linewidth=1.2, alpha=.92, zorder=3, ax=ax)
    ax.scatter(hdf.x, hdf.y, marker="*", s=950, color=HUMANC, edgecolor="white", linewidth=1.5,
               zorder=6, label="human (reference)")
    ax.annotate("HUMAN", (float(hdf.x.iloc[0]), float(hdf.y.iloc[0])), fontsize=14, fontweight="bold",
                xytext=(10, 7), textcoords="offset points", zorder=7)
    for _, r in mdf.iterrows():
        ax.annotate(r["cond"], (r.x, r.y), fontsize=9.5, alpha=.6, xytext=(5, 4), textcoords="offset points")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, fontsize=12, framealpha=.92, loc="best")
    ax.set_xlabel(pc_axis_label(1, loadings[0], FEATS, var[0]), fontsize=12)
    ax.set_ylabel(pc_axis_label(2, loadings[1], FEATS, var[1]), fontsize=12)
    ax.set_title("Scanpath-signature space across models (PCA)\n"
                 "each dot = one model × one foveation condition; ★ = human", fontsize=15, fontweight="bold")
    ax.margins(.16); sns.despine(ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "pca_models.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

    # ---- figure 2: agent<->human ScanMatch per model vs the human<->human ceiling (seaborn) ----
    dsim = pd.DataFrame([{"model": label(s), "scanmatch": sims[s]} for s in model_slugs])
    fig, ax = plt.subplots(figsize=(8.5, 6))
    sns.barplot(data=dsim, x="model", y="scanmatch", hue="model", hue_order=hue_order, palette=PAL,
                legend=False, edgecolor="white", linewidth=1.2, ax=ax)
    if not np.isnan(ceiling):
        ax.axhline(ceiling, color=HUMANC, ls="--", lw=2.4, label=f"human↔human ceiling ({ceiling:.2f})")
        ax.legend(fontsize=12, loc="lower right")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=13)
    ax.set_xlabel(""); ax.set_ylabel("agent↔human ScanMatch\n(higher = more human-like)", fontsize=12)
    ax.set_ylim(0, max(0.62, (ceiling or 0) + .1))
    ax.set_title("Scanpath similarity to humans, per model\n(mean over conditions; sampled)",
                 fontsize=14, fontweight="bold")
    sns.despine(ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "similarity_models.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

    # ---- console summary ----
    print(f"[compare_models] models = {[label(s) for s in model_slugs]}")
    print(f"  PCA explained variance: PC1={var[0]*100:.0f}%  PC2={var[1]*100:.0f}%  (axes labelled by loadings)")
    print(f"  human↔human ScanMatch ceiling = {ceiling:.3f}")
    for s in model_slugs:
        nconds = len(report[s]["by_condition"])
        print(f"  {label(s):16s} agent↔human ScanMatch = {sims[s]:.3f}   ({nconds} conditions)")
    print(f"figures -> {figdir}/pca_models.png, similarity_models.png  |  json -> results/metrics/model_comparison.json")


if __name__ == "__main__":
    main()
