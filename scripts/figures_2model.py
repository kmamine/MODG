#!/usr/bin/env python3
"""Two-model comparison figures (offline, no model calls) for the cross-architecture paper.

For every slug in config analysis.compare_slugs, recompute per-condition TFP (first-look + final) and
target-absent stopping from results/raw/<slug>/, reusing the metric functions in src/metrics.py, then
render two LARGE, publication-quality SEABORN figures used in the main paper:
  - gist_sweep_2model.png : TFP@1 and TFP-end vs gist-k, one model per colour + human reference lines
                            ("no human-like band" holds for BOTH models).
  - stopping_2model.png   : target-absent search extent (median NumFix) and declared-absent fraction
                            by condition, grouped by model + human reference (Qwen one-glances; GLM
                            searches).
Qwen is existence-filtered (intact baseline); GLM is unfiltered (baseline archived/pending) — handled
automatically by the per-slug existence file being present or absent.
"""
import os
import sys
import json
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.metrics import agent_fix_px, human_fix_px, tfp, stopping
from src.conditions import condition_dirs

N_TFP = 15
HUMAN_TFP1, HUMAN_TFPEND, HUMAN_TA_NUMFIX = 0.49, 0.93, 5      # COCO-Search18 reference
LABELS = {"qwen3.5-35b-a3b": "Qwen3.5-35B", "glm-4.6v-flash": "GLM-4.6V-Flash",
          "gemma-4-e4b-it": "Gemma-4-E4B"}
PAL = {"Qwen3.5-35B": "#4C72B0", "GLM-4.6V-Flash": "#DD8452", "Gemma-4-E4B": "#55A868"}
HUMANC = "#333333"

sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def group_humans(path):
    g = collections.defaultdict(list)
    for r in json.load(open(path)):
        g[(r["name"], r["task"])].append(r)
    return g


def per_model(slug, cfg, bbox, tol):
    """Per-condition {tfp1, tfpend, ta_numfix_med, ta_declared} for one model (mirrors compute_metrics)."""
    raw = os.path.join(ROOT, "results", "raw", slug)
    conds = condition_dirs(cfg, raw)
    passed = {(r["image_id"], r["target"], r["condition"]) for r in
              load_jsonl(os.path.join(ROOT, "results/baselines", slug, "existence.jsonl")) if r["correct"]}
    have = bool(passed)
    out = {}
    for cond in conds:
        recs = load_jsonl(os.path.join(raw, cond, "scanpaths.jsonl"))
        tp = collections.defaultdict(list); ta = []
        for r in recs:
            if have and (r["image_id"], r["target_category"], r["condition"]) not in passed:
                continue
            (tp[(r["image_id"], r["target_category"])].append(r) if r["condition"] == "TP" else ta.append(r))
        curves = [tfp([agent_fix_px(r) for r in rs], bbox[iid], tol, N_TFP)
                  for iid, rs in tp.items() if iid in bbox]
        d = {}
        if curves:
            m = np.mean(curves, axis=0); d["tfp1"], d["tfpend"] = float(m[1]), float(m[-1])
        if ta:
            st = stopping(ta, "TA")
            d["ta_numfix_med"] = float(st.get("numfix_median", np.nan))
            d["ta_declared"] = float(st.get("frac_declared_absent", np.nan))
        out[cond] = d
    return out, have


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    ppd = float(cfg["foveation"]["px_per_degree"]); tol = ppd
    sub = os.path.join(ROOT, "data", "subset")
    bbox = {(r["image_id"], r["target_category"]): r["bbox"]
            for r in json.load(open(os.path.join(sub, "image_ids_tp.json")))}
    slugs = [s for s in ((cfg.get("analysis") or {}).get("compare_slugs") or [])
             if os.path.isdir(os.path.join(ROOT, "results/raw", s))]
    figdir = os.path.join(ROOT, "figures", "pilot"); os.makedirs(figdir, exist_ok=True)
    data = {s: per_model(s, cfg, bbox, tol)[0] for s in slugs}
    bracket = cfg["analysis"]["bracket"]
    ks = sorted(int(d.rsplit("-k", 1)[1]) for d in bracket if d.startswith("fov-gaussian-k"))
    order_lbl = [s for s in (LABELS[x] for x in slugs)]

    # ---------- Figure 1: gist-k sweep (seaborn lineplot) ----------
    rows = []
    for s in slugs:
        for k in ks:
            d = data[s].get(f"fov-gaussian-k{k}", {})
            rows.append({"model": LABELS[s], "k": k, "metric": "TFP@1 (first look)", "value": d.get("tfp1", np.nan)})
            rows.append({"model": LABELS[s], "k": k, "metric": "TFP-end (final)", "value": d.get("tfpend", np.nan)})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11.5, 7))
    sns.lineplot(data=df, x="k", y="value", hue="model", style="metric", markers=True,
                 dashes={"TFP@1 (first look)": "", "TFP-end (final)": (4, 2)}, palette=PAL,
                 markersize=13, linewidth=3, errorbar=None, ax=ax)
    ax.axhline(HUMAN_TFP1, color=HUMANC, ls=":", lw=2.2)
    ax.axhline(HUMAN_TFPEND, color="#999999", ls=":", lw=2.2)
    ax.text(ks[-1], HUMAN_TFP1 + .015, f"human TFP@1 ≈ {HUMAN_TFP1}", ha="right", va="bottom",
            fontsize=13, color=HUMANC)
    ax.text(ks[-1], HUMAN_TFPEND + .015, f"human TFP-end ≈ {HUMAN_TFPEND}", ha="right", va="bottom",
            fontsize=13, color="#777777")
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel("gaussian gist-$k$  (peripheral acuity ÷ $k$)"); ax.set_ylabel("target-fixation probability")
    ax.set_ylim(0, 1.04)
    ax.set_title("No human-like band in any model:\nTFP@1 and TFP-end fall together as the periphery degrades",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, ncol=1, framealpha=.92, loc="upper right")
    sns.despine(ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "gist_sweep_2model.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 2: target-absent stopping (seaborn grouped bars, 2 panels) ----------
    rows = []
    for s in slugs:
        for c in bracket:
            d = data[s].get(c, {})
            rows.append({"model": LABELS[s], "condition": c.replace("fov-", ""),
                         "ta_numfix": d.get("ta_numfix_med", np.nan), "ta_declared": d.get("ta_declared", np.nan)})
    df = pd.DataFrame(rows)
    cond_order = [c.replace("fov-", "") for c in bracket]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 7))
    sns.barplot(data=df, x="condition", y="ta_numfix", hue="model", order=cond_order, palette=PAL,
                hue_order=order_lbl, edgecolor="white", linewidth=1, ax=axL)
    axL.axhline(HUMAN_TA_NUMFIX, color=HUMANC, ls="--", lw=2.4, label=f"human ≈ {HUMAN_TA_NUMFIX} fixations")
    axL.set_xlabel(""); axL.set_ylabel("median fixations before declaring absent")
    axL.set_title("Target-absent search extent\n(scale gradient: the small model searches at human length)", fontsize=15, fontweight="bold")
    axL.tick_params(axis="x", rotation=35); axL.legend(fontsize=12, loc="upper left")
    for lb in axL.get_xticklabels(): lb.set_ha("right")
    sns.barplot(data=df, x="condition", y="ta_declared", hue="model", order=cond_order, palette=PAL,
                hue_order=order_lbl, edgecolor="white", linewidth=1, ax=axR, legend=False)
    axR.set_xlabel(""); axR.set_ylabel("fraction of TA trials declared absent"); axR.set_ylim(0, 1.04)
    axR.set_title("Declared-absent rate", fontsize=15, fontweight="bold")
    axR.tick_params(axis="x", rotation=35)
    for lb in axR.get_xticklabels(): lb.set_ha("right")
    fig.suptitle("Target-absent stopping — a graded inter-model difference", fontsize=18, fontweight="bold")
    sns.despine(fig=fig)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "stopping_2model.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    for s in slugs:
        print(f"[figures_2model] {LABELS.get(s, s)}:")
        for d in ["fov-sharp", "fov-geisler_perry", "fov-gaussian-k32", "fov-crop"]:
            x = data[s].get(d, {})
            print(f"    {d:20s} TFP@1={x.get('tfp1','-')}  TFP-end={x.get('tfpend','-')}  "
                  f"TA-numfix-med={x.get('ta_numfix_med','-')}  decl-abs={x.get('ta_declared','-')}")
    print(f"figures -> {figdir}/gist_sweep_2model.png, stopping_2model.png  (seaborn)")


if __name__ == "__main__":
    main()
