#!/usr/bin/env python3
"""SINGLE SOURCE OF TRUTH for the question-driven paper (offline; NO model calls; NO --slug).

Computes EVERY number the main paper and supplement cite, for all three models + the human reference,
with exactly ONE canonical definition per quantity, into ONE file: results/metrics/paper_data.json.
It is the only writer of paper numbers; it never touches the shared per-slug JSONs (comparison.json,
mixed_effects.json, ...), so the cross-model clobbering that produced main<->supp contradictions cannot
recur. emit_paper_tex.py turns this JSON into LaTeX macros + table fragments; verify_numbers.py asserts
the .tex never disagrees with it.

Axes: DECISION (existence accuracy, yes-bias, in-harness present/absent d'/criterion, TA stopping),
FINDING (TFP@1/TFP-end/curve, NumFix, NumFix2T, density), GAZE (per-scanpath signature Cliff's delta +
CIs, scanpath similarity AH/AA vs the human ceiling). Plus strata, hit-tolerance, mixed-effects, the
cross-model PCA, and the Qwen temperature pilot. Reuses src/ functions throughout.

Run:  python3 scripts/build_paper_data.py            (full; slow: all-pairs ScanMatch + density KDE)
      python3 scripts/build_paper_data.py --fast      (skip similarity + density for a quick check)
"""
import os, sys, json, collections, argparse, subprocess, hashlib, math
from itertools import combinations, product

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import yaml

from src.metrics import (agent_fix_px, human_fix_px, numfix, numfix2t, tfp, stopping,
                         density_map, nss, cc, kl, gaze_entropy, saccade_amplitudes, scanpath_length,
                         refixation_rate, center_bias, turning_angles, hull_area, saccade_directions,
                         _hit_step)
from src.scanpath_metrics import scanmatch_score
from src.stats import cliffs_delta, cliffs_delta_ci, pca_embed
from src.conditions import condition_dirs

PPD = 30.0                                  # set from config in main()
GP = "fov-geisler_perry"                    # the human-matched reference condition
N_TFP = 15                                  # TFP curve length (saccades); TFP-end = curve[N_TFP]
LABELS = {"qwen3.5-35b-a3b": "Qwen3.5-35B", "glm-4.6v-flash": "GLM-4.6V-Flash",
          "gemma-4-e4b-it": "Gemma-4-E4B"}

# ---- canonical per-scanpath GAZE signature (ONE key set, shared by human + all models) ----
def _sig_funcs():
    return {
        "gaze_entropy":   gaze_entropy,
        "saccade_amp":    lambda f: float(np.median(saccade_amplitudes(f, PPD))) if len(f) > 1 else np.nan,
        "scanpath_length": lambda f: scanpath_length(f, PPD),
        "refixation":     refixation_rate,
        "center_bias":    lambda f: center_bias(f, PPD),
        "turn_angle":     lambda f: float(np.median(np.abs(turning_angles(f)))) if len(f) > 2 else np.nan,
        "hull_area":      lambda f: hull_area(f, PPD),
    }
PCA_FEATS = ["scanpath_length", "saccade_amp", "gaze_entropy", "refixation", "center_bias"]


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else None


def zinv(p):
    """Inverse normal CDF (probit) via erfinv; for d'/criterion."""
    return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


def _erfinv(y):
    # Winitzki approximation; adequate for SDT rates after log-linear correction
    a = 0.147
    ln = math.log(1 - y * y)
    t = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), y)


def sdt(n_hit, n_tp, n_fa, n_ta):
    """Signal-detection d' and criterion from present-decision counts, log-linear corrected."""
    H = (n_hit + 0.5) / (n_tp + 1.0)
    F = (n_fa + 0.5) / (n_ta + 1.0)
    zH, zF = zinv(H), zinv(F)
    return {"hit_rate": float(n_hit / n_tp) if n_tp else float("nan"),
            "fa_rate": float(n_fa / n_ta) if n_ta else float("nan"),
            "dprime": float(zH - zF), "criterion": float(-0.5 * (zH + zF))}


# ============================== human reference ==============================
def build_human(cfg, bbox, sub):
    tp = load_json(os.path.join(sub, "human_fixations/fixations_tp.json"))
    ta = load_json(os.path.join(sub, "human_fixations/fixations_ta.json"))
    tol = PPD
    SIG = _sig_funcs()
    by_img = collections.defaultdict(list)
    for r in tp:
        by_img[(r["name"], r["task"])].append(human_fix_px(r))
    # TFP curve: mean over (image,target) of metrics.tfp on that trial's pooled human scanpaths
    curves = [tfp(sps, bbox[k], tol, N_TFP) for k, sps in by_img.items() if k in bbox]
    curve = list(np.mean(curves, axis=0))
    # signature medians + raw per-scanpath values (for clustered deltas + PCA)
    sig_vals = {m: [] for m in SIG}; clusters = []
    for k, sps in by_img.items():
        for f in sps:
            for m, fn in SIG.items():
                sig_vals[m].append(fn(f))
            clusters.append(k)
    numfix_tp = [len(f) for sps in by_img.values() for f in sps]
    numfix_ta = [len(human_fix_px(r)) for r in ta]
    # SDT from gamepad correctness: hit = correct present on TP; FA = wrongly-present on TA
    n_tp, n_hit = len(tp), sum(int(r.get("correct", 0)) for r in tp)
    n_ta, n_fa = len(ta), sum(1 - int(r.get("correct", 0)) for r in ta)
    H = {
        "label": "human",
        "tfp1": float(curve[1]), "tfpend": float(curve[N_TFP]), "tfp_curve": [float(x) for x in curve],
        "numfix": {"TP": float(np.median(numfix_tp)), "TA": float(np.median(numfix_ta)),
                   "TA_max": int(np.max(numfix_ta))},
        "signature_median": {m: float(np.nanmedian(sig_vals[m])) for m in SIG},
        "decision_sdt": sdt(n_hit, n_tp, n_fa, n_ta),
    }
    return H, sig_vals, clusters, by_img


# ============================== per model ==============================
def build_model(slug, cfg, bbox, strata_meta, human_sig, human_clusters, human_by_img, fast):
    tol = PPD
    SIG = _sig_funcs()
    raw = os.path.join(ROOT, "results", "raw", slug)
    conds = condition_dirs(cfg, raw)
    ex = load_jsonl(os.path.join(ROOT, "results/baselines", slug, "existence.jsonl"))
    loc = load_jsonl(os.path.join(ROOT, "results/baselines", slug, "localization.jsonl"))
    pass_tp = {(r["image_id"], r["target"]) for r in ex if r["condition"] == "TP" and r["correct"]}
    pass_ta = {(r["image_id"], r["target"]) for r in ex if r["condition"] == "TA" and r["correct"]}

    # ---- detection baseline (condition-free) ----
    tp_ex = [r for r in ex if r["condition"] == "TP"]; ta_ex = [r for r in ex if r["condition"] == "TA"]
    det = {"tp_acc": float(np.mean([r["correct"] for r in tp_ex])),
           "ta_acc": float(np.mean([r["correct"] for r in ta_ex])),
           "yes_bias": float(np.mean([r["answer"] == "yes" for r in ta_ex])),
           "n_tp": len(tp_ex), "n_ta": len(ta_ex)}
    loc_hits = [r for r in loc if r.get("hit") is not None]
    det["localization_inbox"] = (float(np.mean([r["hit"] for r in loc_hits])) if loc_hits else None)
    det["localization_n"] = len(loc_hits)

    out = {"label": LABELS.get(slug, slug), "detection": det,
           "axes": {"decision": {}, "finding": {}, "dynamics": {}, "similarity": {}},
           "pca_vec": {}}
    sig_raw_by_cond = {}                 # for PCA (per-condition signature medians)

    for cond in conds:
        recs = load_jsonl(os.path.join(raw, cond, "scanpaths.jsonl"))
        tp_all = [r for r in recs if r["condition"] == "TP"]
        ta_all = [r for r in recs if r["condition"] == "TA"]
        tp_f = [r for r in tp_all if (r["image_id"], r["target_category"]) in pass_tp]
        ta_f = [r for r in ta_all if (r["image_id"], r["target_category"]) in pass_ta]

        # ---------- FINDING ----------
        by_img = collections.defaultdict(list)
        for r in tp_f:
            by_img[(r["image_id"], r["target_category"])].append(agent_fix_px(r))
        curves = [tfp(sps, bbox[k], tol, N_TFP) for k, sps in by_img.items() if k in bbox]
        curve = list(np.mean(curves, axis=0)) if curves else [float("nan")] * (N_TFP + 1)
        nf_tp = [numfix(r) for r in tp_f]
        n2t = [v for r in tp_f for v in [numfix2t(agent_fix_px(r), bbox.get((r["image_id"], r["target_category"])), tol)]
               if not np.isnan(v)]
        find = {"tfp1": float(curve[1]), "tfpend": float(curve[N_TFP]),
                "tfp_curve": [float(x) for x in curve],
                "numfix_tp_med": float(np.median(nf_tp)) if nf_tp else None,
                "numfix2t_med": float(np.median(n2t)) if n2t else None}

        # ---------- DECISION ----------
        st = stopping(ta_f, "TA") if ta_f else {}
        # in-harness present/absent SDT: hit = FOUND on TP (filtered), FA = FOUND on TA (filtered)
        n_hit = sum(1 for r in tp_f if r["stop_reason"] == "found"); n_tp = len(tp_f)
        n_fa = sum(1 for r in ta_f if r["stop_reason"] == "found"); n_ta = len(ta_f)
        dec = {"declared_absent": float(st.get("frac_declared_absent", float("nan"))),
               "false_present": float(st.get("frac_false_present", float("nan"))),
               "hit_cap": float(st.get("frac_hit_cap", float("nan"))),
               "numfix_ta_med": float(st.get("numfix_median", float("nan")))}
        dec.update(sdt(n_hit, n_tp, n_fa, n_ta))

        # ---------- GAZE: signature ----------
        sig_vals = {m: [] for m in SIG}; clusters = []
        dirs, turns = [], []
        for r in tp_f:
            f = agent_fix_px(r)
            for m, fn in SIG.items():
                sig_vals[m].append(fn(f))
            clusters.append((r["image_id"], r["target_category"]))
            dirs.extend(list(saccade_directions(f)))
            if len(f) > 2:
                turns.extend(list(turning_angles(f)))
        # point Cliff's delta for every cell (fast); bootstrap CI only for the cited GP gaze metrics
        dyn = {"signature": {}}
        for m in SIG:
            av = [v for v in sig_vals[m] if v == v]; hv = [v for v in human_sig[m] if v == v]
            entry = {"median": float(np.nanmedian(sig_vals[m])),
                     "delta": float(cliffs_delta(av, hv)) if (av and hv) else float("nan")}
            if cond == GP and m in ("gaze_entropy", "saccade_amp", "center_bias", "refixation"):
                entry["ci"] = cliffs_delta_ci(sig_vals[m], clusters, human_sig[m], human_clusters, n=1000)
            dyn["signature"][m] = entry
        # histograms (for figures): saccade direction (16 bins, deg) + turning angle (18 bins)
        if dirs:
            h, e = np.histogram(dirs, bins=16, range=(-180, 180)); dyn["sacc_dir_hist"] = [int(x) for x in h]
        if turns:
            h, e = np.histogram(turns, bins=18, range=(-180, 180)); dyn["turn_angle_hist"] = [int(x) for x in h]
        out["pca_vec"][cond] = [float(np.nanmedian(sig_vals[m])) for m in PCA_FEATS]

        # ---------- density (NSS/CC/KL vs human), drop step-0 ----------
        if not fast:
            a_pts = [agent_fix_px(r)[1:] for r in tp_f if len(agent_fix_px(r)) > 1]
            h_pts = [f[1:] for sps in human_by_img.values() for f in sps if len(f) > 1]
            if a_pts and h_pts:
                am = density_map(a_pts, PPD); hm = density_map(h_pts, PPD)
                hp = np.vstack(h_pts)
                find["density"] = {"nss": float(nss(am, hp)), "cc": float(cc(am, hm)), "kl": float(kl(hm, am))}

        # ---------- similarity (all-pairs ScanMatch + MultiMatch-position) ----------
        if not fast:
            ah_sm, aa_sm = [], []
            for k, aps in by_img.items():
                if k in human_by_img:
                    for a, h in product(aps, human_by_img[k]):
                        ah_sm.append(scanmatch_score(a, h))
                for a, b in combinations(aps, 2):
                    aa_sm.append(scanmatch_score(a, b))
            out["axes"]["similarity"][cond] = {
                "scanmatch_ah": float(np.nanmean(ah_sm)) if ah_sm else None,
                "scanmatch_aa": float(np.nanmean(aa_sm)) if aa_sm else None,
                "n_scanpaths": int(sum(len(v) for v in by_img.values()))}

        out["axes"]["finding"][cond] = find
        out["axes"]["decision"][cond] = dec
        out["axes"]["dynamics"][cond] = dyn

    # ---------- strata (sharp) + hit-tolerance ----------
    out["strata"] = strata_tfp1(slug, cfg, bbox, strata_meta, pass_tp)
    out["tolerance"] = tolerance_tfp(slug, cfg, bbox, pass_tp)
    # ---------- mixed-effects ----------
    out["mixed_effects"] = mixed_effects(slug, cfg, bbox, pass_tp)
    # ---------- temperature pilot (only where -t dirs exist) ----------
    out["temp_sweep"] = temp_sweep(slug, raw)
    return out


def strata_tfp1(slug, cfg, bbox, meta, pass_tp):
    tol = PPD
    raw = os.path.join(ROOT, "results", "raw", slug)
    recs = load_jsonl(os.path.join(raw, "fov-sharp", "scanpaths.jsonl"))
    by_img = collections.defaultdict(list)
    for r in recs:
        if r["condition"] == "TP" and (r["image_id"], r["target_category"]) in pass_tp:
            by_img[(r["image_id"], r["target_category"])].append(agent_fix_px(r))
    out = {}
    for stratum, keys in meta["strata_keys"].items():
        per = [np.mean([(_hit_step(s, bbox[k], tol) or 1e9) <= 1 for s in by_img[k]])
               for k in keys if k in by_img and k in bbox]
        out[stratum] = {"sharp_tfp1": float(np.mean(per)) if per else None,
                        "human_tfp1": meta["human_tfp1"][stratum], "n": len(keys)}
    return out


def tolerance_tfp(slug, cfg, bbox, pass_tp):
    raw = os.path.join(ROOT, "results", "raw", slug)
    conds = condition_dirs(cfg, raw)
    tols = {"0.5deg": 0.5 * PPD, "1deg": 1.0 * PPD, "1.5deg": 1.5 * PPD}
    tab, vals = {}, []
    for c in conds:
        recs = load_jsonl(os.path.join(raw, c, "scanpaths.jsonl"))
        by_img = collections.defaultdict(list)
        for r in recs:
            if r["condition"] == "TP" and (r["image_id"], r["target_category"]) in pass_tp:
                by_img[(r["image_id"], r["target_category"])].append(agent_fix_px(r))
        row = {}
        for tn, tv in tols.items():
            per = [np.mean([(_hit_step(s, bbox[k], tv) or 1e9) <= 1 for s in sps])
                   for k, sps in by_img.items() if k in bbox]
            row[tn] = float(np.mean(per)) if per else float("nan")
        tab[c] = row; vals.append(row)
    max_shift = max((abs(r[a] - r[b]) for r in vals for a in tols for b in tols
                     if not (np.isnan(r[a]) or np.isnan(r[b]))), default=float("nan"))
    return {"tfp1_by_tol": tab, "max_shift_tfp1": float(max_shift)}


def mixed_effects(slug, cfg, bbox):
    pass  # placeholder; replaced below (kept name stable)


def mixed_effects(slug, cfg, bbox, pass_tp):  # noqa: F811  (single canonical definition)
    import pandas as pd, statsmodels.formula.api as smf
    tol = PPD
    sub = os.path.join(ROOT, "data", "subset")
    raw = os.path.join(ROOT, "results", "raw", slug)
    conds = [c for c in condition_dirs(cfg, raw) if c in ("fov-sharp", "fov-geisler_perry")]

    def sigs(f, b):
        amp = saccade_amplitudes(f, PPD); h = _hit_step(f, b, tol) if b is not None else None
        return {"gaze_entropy": gaze_entropy(f),
                "saccade_amp": float(np.median(amp)) if len(amp) else np.nan,
                "numfix": float(len(f)),
                "tfp1": (1.0 if (h is not None and h <= 1) else 0.0) if b is not None else np.nan}
    rows = []
    for r in load_json(os.path.join(sub, "human_fixations/fixations_tp.json")):
        k = (r["name"], r["task"]); s = sigs(human_fix_px(r), bbox.get(k))
        rows.append(dict(image=f"{k[0]}|{k[1]}", rater=f"H{r['subject']}", cond="human", **s))
    for c in conds:
        for r in load_jsonl(os.path.join(raw, c, "scanpaths.jsonl")):
            if r["condition"] != "TP": continue
            k = (r["image_id"], r["target_category"])
            if k not in pass_tp: continue
            s = sigs(agent_fix_px(r), bbox.get(k))
            rows.append(dict(image=f"{k[0]}|{k[1]}", rater=f"S{r['seed']}", cond=c.replace("fov-", ""), **s))
    df = pd.DataFrame(rows)
    rep = {}
    for m in ["gaze_entropy", "saccade_amp", "numfix", "tfp1"]:
        d = df[["image", "rater", "cond", m]].dropna().rename(columns={m: "y"}); d["grp"] = 1
        try:
            r = smf.mixedlm("y ~ C(cond, Treatment(reference='human'))", d, groups="grp",
                            vc_formula={"image": "0 + C(image)", "rater": "0 + C(rater)"}).fit(
                            reml=True, method="lbfgs", maxiter=200, disp=False)
            ci = r.conf_int(); fe = {}
            for name in r.fe_params.index:
                if name == "Intercept": continue
                lvl = name.split("[T.")[-1].rstrip("]")
                fe[lvl] = {"delta": float(r.fe_params[name]), "lo": float(ci.loc[name, 0]),
                           "hi": float(ci.loc[name, 1]), "p": float(r.pvalues[name])}
            rep[m] = {"fixed_vs_human": fe, "converged": bool(r.converged)}
        except Exception as e:
            rep[m] = {"error": str(e)[:160]}
    return rep


def temp_sweep(slug, raw):
    """AA ScanMatch per (core condition @ temperature) where -t dirs exist (Qwen pilot)."""
    import glob, re
    out = {}
    for d in sorted(glob.glob(os.path.join(raw, "fov-*-t*"))):
        cond = os.path.basename(d)
        recs = [r for r in load_jsonl(os.path.join(d, "scanpaths.jsonl")) if r["condition"] == "TP"]
        by_img = collections.defaultdict(list)
        for r in recs:
            by_img[(r["image_id"], r["target_category"])].append(agent_fix_px(r))
        aa = [scanmatch_score(a, b) for sps in by_img.values() for a, b in combinations(sps, 2)]
        if aa:
            out[cond] = {"scanmatch_aa": float(np.nanmean(aa)), "n_pairs": len(aa)}
    return out


def main():
    global PPD
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip similarity + density (quick validation)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    PPD = float(cfg["foveation"]["px_per_degree"])
    slugs = [s for s in cfg["analysis"]["compare_slugs"]
             if os.path.isdir(os.path.join(ROOT, "results/raw", s))]
    sub = os.path.join(ROOT, "data", "subset")
    recs_tp = load_json(os.path.join(sub, "image_ids_tp.json"))
    bbox = {(r["image_id"], r["target_category"]): r["bbox"] for r in recs_tp}

    # strata meta (ecc_bin x size median split) — computed once, shared by all models
    ecc = {(r["image_id"], r["target_category"]): r.get("ecc_bin") for r in recs_tp}
    area = {(r["image_id"], r["target_category"]): r["bbox"][2] * r["bbox"][3] for r in recs_tp}
    smed = float(np.median(list(area.values())))
    sizeb = {k: ("large" if a >= smed else "small") for k, a in area.items()}
    strata_keys = {f"{e}-{s}": {k for k in bbox if ecc.get(k) == e and sizeb.get(k) == s}
                   for e in ("near", "far") for s in ("small", "large")}
    # human TFP@1 per stratum (tol 1deg)
    humans_tp = collections.defaultdict(list)
    for r in load_json(os.path.join(sub, "human_fixations/fixations_tp.json")):
        humans_tp[(r["name"], r["task"])].append(human_fix_px(r))
    human_tfp1_stratum = {}
    for st, keys in strata_keys.items():
        per = [np.mean([(_hit_step(s, bbox[k], PPD) or 1e9) <= 1 for s in humans_tp[k]])
               for k in keys if k in humans_tp and k in bbox]
        human_tfp1_stratum[st] = float(np.mean(per)) if per else None
    strata_meta = {"strata_keys": strata_keys, "human_tfp1": human_tfp1_stratum, "size_median_px2": smed}

    H, human_sig, human_clusters, human_by_img = build_human(cfg, bbox, sub)
    # human-human ScanMatch ceiling (all 45 pairs/image, uncapped) — the ONE ceiling value
    if not args.fast:
        hh = [scanmatch_score(a, b) for sps in human_by_img.values() for a, b in combinations(sps, 2)]
        H["scanmatch_ceiling"] = float(np.nanmean(hh)) if hh else None

    data = {"_human": H}
    pca_vecs = [[H["signature_median"][m] for m in PCA_FEATS]]; pca_groups = [("human", None)]
    for slug in slugs:
        print(f"[build] {slug} ...", flush=True)
        m = build_model(slug, cfg, bbox, strata_meta, human_sig, human_clusters, human_by_img, args.fast)
        if not args.fast:
            m["scanmatch_ceiling"] = H.get("scanmatch_ceiling")
        for cond, vec in m.pop("pca_vec").items():
            pca_vecs.append(vec); pca_groups.append((slug, cond))
        data[slug] = m

    # ---- cross-model PCA on the 5-metric signature (interpretable axes) ----
    vecs = np.array(pca_vecs, float)
    vecs = np.where(np.isnan(vecs), np.nanmean(vecs, axis=0), vecs)
    coords, loadings, var = pca_embed(vecs)
    data["_pca"] = {"feature_order": PCA_FEATS,
                    "explained_variance_ratio": [float(v) for v in var],
                    "loadings": {f"PC{i+1}": dict(zip(PCA_FEATS, [float(x) for x in loadings[i]]))
                                 for i in range(len(loadings))},
                    "coords": [{"slug": g[0], "cond": g[1], "x": float(c[0]), "y": float(c[1])}
                               for g, c in zip(pca_groups, coords)]}

    # ---- meta / provenance + canonical definitions ----
    try:
        commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        commit = "unknown"
    n_ep = {}
    for slug in slugs:
        raw = os.path.join(ROOT, "results/raw", slug)
        n_ep[slug] = sum(len(load_jsonl(os.path.join(raw, c, "scanpaths.jsonl")))
                         for c in condition_dirs(cfg, raw))
    data["_meta"] = {
        "generated_by": "scripts/build_paper_data.py", "git_commit": commit,
        "config_sha256": hashlib.sha256(open(os.path.join(ROOT, "config.yaml"), "rb").read()).hexdigest()[:12],
        "ppd": PPD, "tol_deg": 1.0, "glimpse_cap": cfg["harness"]["max_fixations_tp"],
        "n_seeds": len(cfg["run"]["seeds"]), "bracket": cfg["analysis"]["bracket"],
        "slugs": slugs, "labels": {s: LABELS.get(s, s) for s in slugs},
        "n_images": {"TP": 141, "TA": 144}, "n_episodes": n_ep,
        "human_ceiling_scanmatch": H.get("scanmatch_ceiling"),
        "fast_mode": args.fast,
        "definitions": {
            "existence_filter": "TP analyses keep (image_id,target) with TP existence correct; TA analyses keep TA-existence-correct; identical for all models.",
            "tfp": "per (image,target) trial = fraction of that trial's existence-pass scanpaths whose first in-box (<=1deg) fixation is by saccade n; TFP@1=n=1, TFP-end=n=15; reported as mean over trials. ONE definition for headline, strata, tolerance.",
            "declared_absent": "fraction of TA existence-pass episodes ending stop_reason=not_present.",
            "false_present": "fraction of TA existence-pass episodes ending stop_reason=found.",
            "yes_bias": "fraction of one-shot TA existence answers == 'yes' (TA false-yes rate).",
            "dprime": "in-harness present decision: hit=FOUND on TP, FA=FOUND on TA; log-linear corrected.",
            "self_consistency": "agent<->agent ScanMatch over cross-seed pairs per trial, pooled; ceiling = human<->human ScanMatch over all 45 pairs/image, uncapped.",
            "signature_delta": "Cliff's delta agent vs human (sign=agent-human) with image-clustered bootstrap 95% CI.",
        }}

    os.makedirs(os.path.join(ROOT, "results", "metrics"), exist_ok=True)
    outp = os.path.join(ROOT, "results", "metrics", "paper_data.json")
    json.dump(data, open(outp, "w"), indent=1,
              default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else float(o))
    print(f"[build] wrote {outp}  (slugs={slugs}, fast={args.fast})")
    # quick headline echo
    for s in slugs:
        gp = data[s]["axes"]
        print(f"  {LABELS.get(s,s):14s} GP TFP@1={gp['finding']['fov-geisler_perry']['tfp1']:.2f} "
              f"declAbs={gp['decision']['fov-geisler_perry']['declared_absent']:.2f} "
              f"entropy-d={gp['dynamics']['fov-geisler_perry']['signature']['gaze_entropy']['delta']:+.2f}")
    print(f"  human TFP@1={H['tfp1']:.3f} ceiling={H.get('scanmatch_ceiling')}")


if __name__ == "__main__":
    main()
