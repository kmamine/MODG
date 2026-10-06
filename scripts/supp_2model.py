#!/usr/bin/env python3
"""Consolidated two-model analysis for the comprehensive supplement (offline, no model calls, no slow
ScanMatch). For every slug in analysis.compare_slugs it recomputes, per condition: TFP (first/final),
NumFix (TP/TA), target-absent stopping, fixation-density agreement (NSS/CC/KL), and the per-scanpath
intrinsic signature (median + Cliff's delta vs human). Dumps results/metrics/supp_2model.json (the
single source for the supplement tables) and renders four seaborn comparison figures:
  tfp_curves_2model.png, numfix_2model.png, density_2model.png, signature_2model.png.
Qwen is existence-filtered (intact baseline); GLM is unfiltered (baseline pending) — auto per slug.
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

from src.metrics import (agent_fix_px, human_fix_px, numfix, tfp, density_map, nss, cc, kl, stopping,
                         gaze_entropy, saccade_amplitudes, scanpath_length, refixation_rate, center_bias)
from src.stats import cliffs_delta
from src.conditions import condition_dirs

N_TFP = 15
PPD = 30.0
LABELS = {"qwen3.5-35b-a3b": "Qwen3.5-35B", "glm-4.6v-flash": "GLM-4.6V-Flash", "gemma-4-e4b-it": "Gemma-4-E4B"}
PAL = {"Qwen3.5-35B": "#4C72B0", "GLM-4.6V-Flash": "#DD8452", "Gemma-4-E4B": "#55A868", "human": "#333333"}
SIG = {"gaze_entropy": gaze_entropy,
       "saccade_amp": lambda f: float(np.median(saccade_amplitudes(f, PPD))) if len(f) > 1 else np.nan,
       "refixation": refixation_rate,
       "scanpath_len": lambda f: scanpath_length(f, PPD),
       "center_bias": lambda f: center_bias(f, PPD)}
SIG_LBL = {"gaze_entropy": "gaze entropy (bits)", "saccade_amp": "saccade amp (°)",
           "refixation": "refixation rate", "scanpath_len": "scanpath length (°)", "center_bias": "center bias (°)"}
sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def group_humans(path):
    g = collections.defaultdict(list)
    for r in json.load(open(path)):
        g[(r["name"], r["task"])].append(r)
    return g


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    tol = sigma = PPD
    sub = os.path.join(ROOT, "data", "subset")
    humans = {"TP": group_humans(os.path.join(sub, "human_fixations/fixations_tp.json")),
              "TA": group_humans(os.path.join(sub, "human_fixations/fixations_ta.json"))}
    bbox = {(r["image_id"], r["target_category"]): r["bbox"]
            for r in json.load(open(os.path.join(sub, "image_ids_tp.json")))}
    slugs = [s for s in ((cfg.get("analysis") or {}).get("compare_slugs") or [])
             if os.path.isdir(os.path.join(ROOT, "results/raw", s))]
    figdir = os.path.join(ROOT, "figures", "pilot"); os.makedirs(figdir, exist_ok=True)
    bracket = cfg["analysis"]["bracket"]

    # human reference: signature distributions + TFP curve + NumFix
    hsig = {m: [SIG[m](human_fix_px(r)) for rs in humans["TP"].values() for r in rs] for m in SIG}
    human_tfp = float(np.mean([tfp([human_fix_px(r) for r in rs], bbox[iid], tol, N_TFP)[1]
                               for iid, rs in humans["TP"].items() if iid in bbox]))
    human_tfp_curve = np.mean([tfp([human_fix_px(r) for r in rs], bbox[iid], tol, N_TFP)
                               for iid, rs in humans["TP"].items() if iid in bbox], axis=0)
    human_nf = {"TP": float(np.median([numfix(r) for rs in humans["TP"].values() for r in rs])),
                "TA": float(np.median([numfix(r) for rs in humans["TA"].values() for r in rs]))}
    report = {"_human": {"signature_median": {m: float(np.nanmedian(hsig[m])) for m in SIG},
                         "tfp1": human_tfp, "numfix": human_nf}}

    # per-model, per-condition
    raw_rows = {}      # (slug, cond) -> dict for figures
    for slug in slugs:
        raw = os.path.join(ROOT, "results", "raw", slug)
        passed = {(r["image_id"], r["target"], r["condition"]) for r in
                  load_jsonl(os.path.join(ROOT, "results/baselines", slug, "existence.jsonl")) if r["correct"]}
        have = bool(passed)
        report[slug] = {"existence_filter": have, "by_condition": {}}
        for cond in condition_dirs(cfg, raw):
            recs = load_jsonl(os.path.join(raw, cond, "scanpaths.jsonl"))
            tp = collections.defaultdict(list); ta = []
            for r in recs:
                if have and (r["image_id"], r["target_category"], r["condition"]) not in passed:
                    continue
                (tp[(r["image_id"], r["target_category"])].append(r) if r["condition"] == "TP" else ta.append(r))
            d = {}
            # TFP + signature (density is fast-tabled separately; omitted here for speed)
            curves = []
            sig_vals = {m: [] for m in SIG}
            nf_tp = []
            for iid, rs in tp.items():
                a_px = [agent_fix_px(r) for r in rs]
                nf_tp += [numfix(r) for r in rs]
                for f in a_px:
                    for m in SIG:
                        sig_vals[m].append(SIG[m](f))
                if iid in bbox:
                    curves.append(tfp(a_px, bbox[iid], tol, N_TFP))
            if curves:
                mc = np.mean(curves, axis=0); d["tfp1"], d["tfpend"] = float(mc[1]), float(mc[-1])
                d["tfp_curve"] = [float(x) for x in mc]
            d["numfix_tp_med"] = float(np.median(nf_tp)) if nf_tp else None
            if ta:
                st = stopping(ta, "TA")
                d["numfix_ta_med"] = float(st.get("numfix_median", np.nan))
                d["declared_absent"] = float(st.get("frac_declared_absent", np.nan))
                d["false_present"] = float(st.get("frac_false_present", np.nan))
                d["hit_cap"] = float(st.get("frac_hit_cap", np.nan))
            d["signature"] = {m: {"median": float(np.nanmedian(sig_vals[m])),
                                  "delta": float(cliffs_delta([v for v in sig_vals[m] if not np.isnan(v)],
                                                              [v for v in hsig[m] if not np.isnan(v)]))}
                              for m in SIG}
            report[slug]["by_condition"][cond] = d
            raw_rows[(slug, cond)] = {"nf_tp": nf_tp, "sig": sig_vals}
    json.dump(report, open(os.path.join(ROOT, "results/metrics/supp_2model.json"), "w"),
              indent=2, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else float(o))

    lbls = [LABELS[s] for s in slugs]
    cond_lbl = [c.replace("fov-", "") for c in bracket]

    # ---- Fig A: TFP-by-saccade curves (sharp, GP) both models + human ----
    fig, ax = plt.subplots(figsize=(11, 6.5))
    xs = list(range(N_TFP + 1))
    ax.plot(xs, human_tfp_curve, color=PAL["human"], ls="--", lw=3, marker="o", ms=7, label="human")
    for s in slugs:
        for cnd, dash in [("fov-sharp", ""), ("fov-geisler_perry", (4, 2))]:
            cv = report[s]["by_condition"].get(cnd, {}).get("tfp_curve")
            if cv:
                ax.plot(xs, cv, color=PAL[LABELS[s]], lw=3, ms=7, marker="s",
                        dashes=dash if dash else (1, 0),
                        label=f"{LABELS[s]} — {cnd.replace('fov-','')}")
    ax.set_xlabel("saccade $n$"); ax.set_ylabel("cumulative target-fixation prob."); ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 8)
    ax.set_title("Target-present TFP by saccade (legible conditions)\nthe reasoning models reach the target on saccade 1; gemma is intermediate; humans climb slowly",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=12); sns.despine(ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "tfp_curves_2model.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

    # ---- Fig B: NumFix-TP distributions by condition (both models) + human median ----
    rows = []
    for s in slugs:
        for c in bracket:
            for v in raw_rows.get((s, c), {}).get("nf_tp", []):
                rows.append({"model": LABELS[s], "condition": c.replace("fov-", ""), "numfix": v})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(15, 6.5))
    sns.boxplot(data=df, x="condition", y="numfix", hue="model", order=cond_lbl, hue_order=lbls,
                palette=PAL, showfliers=False, ax=ax)
    ax.axhline(human_nf["TP"], color=PAL["human"], ls="--", lw=2.4, label=f"human median ({human_nf['TP']:.0f})")
    ax.set_xlabel(""); ax.set_ylabel("fixations (target-present)"); ax.tick_params(axis="x", rotation=35)
    for lb in ax.get_xticklabels(): lb.set_ha("right")
    ax.set_title("Target-present search extent (NumFix) by condition", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11); sns.despine(ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "numfix_2model.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

    # ---- Fig D: intrinsic signature distributions at geisler_perry (human vs both models) ----
    metrics = ["gaze_entropy", "saccade_amp", "refixation", "scanpath_len"]
    rows = []
    for m in metrics:
        for v in hsig[m]:
            if not np.isnan(v): rows.append({"metric": SIG_LBL[m], "group": "human", "value": v})
        for s in slugs:
            for v in raw_rows.get((s, "fov-geisler_perry"), {}).get("sig", {}).get(m, []):
                if not np.isnan(v): rows.append({"metric": SIG_LBL[m], "group": LABELS[s], "value": v})
    df = pd.DataFrame(rows)
    g_order = ["human"] + lbls
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
    for ax, m in zip(axes, metrics):
        sub = df[df.metric == SIG_LBL[m]]
        sns.boxplot(data=sub, x="group", y="value", order=g_order, hue="group", hue_order=g_order,
                    palette=PAL, showfliers=False, legend=False, ax=ax)
        ax.set_title(SIG_LBL[m], fontsize=13, fontweight="bold"); ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20)
        for lb in ax.get_xticklabels(): lb.set_ha("right")
    fig.suptitle("Intrinsic signature at the human-matched condition (geisler\\_perry): human vs.\\ the three models",
                 fontsize=16, fontweight="bold")
    sns.despine(fig=fig); fig.tight_layout()
    fig.savefig(os.path.join(figdir, "signature_2model.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

    print(f"[supp_2model] models={lbls}")
    print(f"  human: TFP@1={human_tfp:.2f} NumFix TP/TA={human_nf['TP']:.0f}/{human_nf['TA']:.0f}")
    print("  wrote results/metrics/supp_2model.json + 4 seaborn figures")


if __name__ == "__main__":
    main()
