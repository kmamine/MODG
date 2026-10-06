#!/usr/bin/env python3
"""Mixed-effects models for the agent-vs-human comparison (HiCV_run.md §6.1). Offline.

Per headline metric (gaze entropy, median saccade amplitude, NumFix, first-look hit TFP@1), pool the
human and agent per-scanpath values and fit a linear mixed model with **crossed** random intercepts
for image AND rater (human subject / agent seed). Fixed effect = condition, with `human` as the
reference level, so each agent-condition coefficient is "agent(cond) − human" on that metric,
controlling for image and rater. Crossed REs are done via statsmodels `vc_formula` over a single
global group (MixedLM's `groups=` alone gives only ONE grouping; do NOT fit single-grouping and call
it crossed). TFP@1 is a 0/1 hit → fit as a linear probability model (coefficient = Δ hit-probability).

Writes results/metrics/mixed_effects.json. Reads only saved scanpaths + human fixations.
"""
import os
import sys
import json
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import yaml
import pandas as pd
import statsmodels.formula.api as smf

from src.metrics import (agent_fix_px, human_fix_px, numfix, saccade_amplitudes,
                         gaze_entropy, _hit_step)
from src.conditions import condition_dirs


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


# per-scanpath metric extractors (fix_px = N×2 px incl. centre); bbox/tol only used by tfp1
def _sigs(fix_px, bbox, tol):
    amp = saccade_amplitudes(fix_px, 30.0)
    hit = _hit_step(fix_px, bbox, tol) if bbox is not None else None
    return {"gaze_entropy": gaze_entropy(fix_px),
            "median_saccade_amp": float(np.median(amp)) if len(amp) else np.nan,
            "numfix": float(len(fix_px)),
            "tfp1": (1.0 if (hit is not None and hit <= 1) else 0.0) if bbox is not None else np.nan}

METRICS = ["gaze_entropy", "median_saccade_amp", "numfix", "tfp1"]


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--slug", default=None); args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    slug = args.slug or cfg["model"]["slug"]
    ppd = float(cfg["foveation"]["px_per_degree"])
    tol = ppd
    sub = os.path.join(ROOT, "data", "subset")
    bbox = {(r["image_id"], r["target_category"]): r["bbox"]
            for r in json.load(open(os.path.join(sub, "image_ids_tp.json")))}
    passed = {(r["image_id"], r["target"]) for r in
              load_jsonl(os.path.join(ROOT, "results/baselines", slug, "existence.jsonl"))
              if r["condition"] == "TP" and r["correct"]}
    raw = os.path.join(ROOT, "results", "raw", slug)
    conds = condition_dirs(cfg, raw)

    rows = []   # long format: one row per (scanpath, metric-bundle)
    # ---- humans (TP reference) ----
    for r in json.load(open(os.path.join(sub, "human_fixations/fixations_tp.json"))):
        key = (r["name"], r["task"])
        s = _sigs(human_fix_px(r), bbox.get(key), tol)
        rows.append(dict(image=f"{key[0]}|{key[1]}", rater=f"H{r['subject']}", cond="human", **s))
    # ---- agents (TP, existence-pass) per condition ----
    for cond in conds:
        for r in load_jsonl(os.path.join(raw, cond, "scanpaths.jsonl")):
            if r["condition"] != "TP":
                continue
            key = (r["image_id"], r["target_category"])
            if passed and key not in passed:
                continue
            s = _sigs(agent_fix_px(r), bbox.get(key), tol)
            rows.append(dict(image=f"{key[0]}|{key[1]}", rater=f"S{r['seed']}",
                             cond=cond.replace("fov-", ""), **s))

    df = pd.DataFrame(rows)
    if df.empty or df["cond"].nunique() < 2:
        print("[mixed] not enough data / conditions; run the pilot or full subset first.")
        return
    levels = ["human"] + [c for c in df["cond"].unique() if c != "human"]
    print(f"[mixed] model={slug}  scanpaths={len(df)}  conditions={levels}  "
          f"images={df['image'].nunique()}  raters={df['rater'].nunique()}")

    report = {}
    for m in METRICS:
        d = df[["image", "rater", "cond", m]].dropna().copy()
        d = d.rename(columns={m: "y"})
        d["grp"] = 1                                    # single global group -> crossed REs via vc
        try:
            md = smf.mixedlm("y ~ C(cond, Treatment(reference='human'))", d, groups="grp",
                             vc_formula={"image": "0 + C(image)", "rater": "0 + C(rater)"})
            r = md.fit(reml=True, method="lbfgs", maxiter=200, disp=False)
            ci = r.conf_int()
            fe = {}
            for name in r.fe_params.index:
                if name == "Intercept":
                    continue
                lvl = name.split("[T.")[-1].rstrip("]") if "[T." in name else name
                fe[lvl] = {"delta_vs_human": float(r.fe_params[name]),
                           "lo": float(ci.loc[name, 0]), "hi": float(ci.loc[name, 1]),
                           "p": float(r.pvalues[name])}
            report[m] = {"fixed_vs_human": fe,
                         "var_image": float(r.vcomp[0]), "var_rater": float(r.vcomp[1]),
                         "scale": float(r.scale), "converged": bool(r.converged), "n": int(len(d))}
            print(f"  {m:20s} converged={r.converged}  var(image)={r.vcomp[0]:.3f} "
                  f"var(rater)={r.vcomp[1]:.3f}  (crossed REs both present)")
        except Exception as e:
            report[m] = {"error": str(e)[:200]}
            print(f"  {m:20s} FAILED: {str(e)[:120]}")

    os.makedirs(os.path.join(ROOT, "results", "metrics"), exist_ok=True)
    json.dump(report, open(os.path.join(ROOT, "results/metrics/mixed_effects.json"), "w"), indent=2)
    print("\n=== fixed effect vs human (Δ; CI) for the legible anchors ===")
    for m in METRICS:
        fe = report.get(m, {}).get("fixed_vs_human", {})
        for c in ("sharp", "geisler_perry"):
            if c in fe:
                v = fe[c]; print(f"  {m:20s} {c:14s} Δ={v['delta_vs_human']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}]")
    print("\n-> results/metrics/mixed_effects.json")


if __name__ == "__main__":
    main()
