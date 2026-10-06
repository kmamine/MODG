#!/usr/bin/env python3
"""All figures for the question-driven paper, from results/metrics/paper_data.json (offline).

Number-bearing figures read the JSON so they cannot disagree with the tables/macros; distribution
figures (boxplots/polar) recompute raw per-scanpath values for visualization only (no scalar claim).
Writes seaborn PNGs into the paper figures dir. Reuses src/ for the few raw recomputations.
"""
import os, sys, json, collections, argparse
from itertools import combinations
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.metrics import (agent_fix_px, human_fix_px, gaze_entropy, saccade_amplitudes, scanpath_length,
                         refixation_rate, saccade_directions, turning_angles)
from src.conditions import condition_dirs

sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")
PAL = {"Qwen3.5-35B-A3B": "#4C72B0", "GLM-4.6V-Flash": "#DD8452", "Gemma-4-E4B": "#55A868"}
HUMANC = "#333333"; PPD = 30.0
GP, SHARP = "fov-geisler_perry", "fov-sharp"
CORDER = ["fov-sharp", "fov-geisler_perry", "fov-gaussian-k8", "fov-gaussian-k16", "fov-gaussian-k24",
          "fov-gaussian-k32", "fov-gaussian-k48", "fov-gaussian-k128", "fov-crop"]
KVALS = [8, 16, 24, 32, 48, 128]
D = json.load(open(os.path.join(ROOT, "results/metrics/paper_data.json")))
SLUGS = D["_meta"]["slugs"]; LAB = {s: D[s]["label"] for s in SLUGS}; H = D["_human"]
LAB = {s: ("Qwen3.5-35B-A3B" if v == "Qwen3.5-35B" else v) for s, v in LAB.items()}


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=160, bbox_inches="tight"); plt.close(fig)
    print("  ", name)


def fig_dissociation(out):
    """Fig 1: three-axis dissociation — DECISION match, FINDING beat, GAZE alien+shared."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    labs = [LAB[s] for s in SLUGS]; cols = [PAL[l] for l in labs]; x = np.arange(len(SLUGS))
    # A DECISION: d' per model vs human d'
    dpr = [D[s]["axes"]["decision"][GP]["dprime"] for s in SLUGS]
    ax[0].bar(x, dpr, color=cols, edgecolor="white")
    ax[0].axhline(H["decision_sdt"]["dprime"], color=HUMANC, ls="--", lw=2.4, label=f"human $d'$={H['decision_sdt']['dprime']:.2f}")
    ax[0].set_title("DECISION\n(present/absent $d'$): matches human", fontsize=14, fontweight="bold")
    ax[0].set_ylabel("$d'$ (higher = better)"); ax[0].legend(fontsize=11, loc="lower right")
    # B FINDING: TFP@1 per model vs human 0.49
    tf = [D[s]["axes"]["finding"][GP]["tfp1"] for s in SLUGS]
    ax[1].bar(x, tf, color=cols, edgecolor="white")
    ax[1].axhline(H["tfp1"], color=HUMANC, ls="--", lw=2.4, label=f"human {H['tfp1']:.2f}")
    ax[1].set_ylim(0, 1.05); ax[1].set_title("FINDING\n(first-saccade TFP@1): beats human", fontsize=14, fontweight="bold")
    ax[1].set_ylabel("TFP@1"); ax[1].legend(fontsize=11, loc="lower right")
    for i, v in enumerate(tf): ax[1].text(i, v + .02, f"{v:.2f}", ha="center", fontsize=11)
    # C GAZE: self-consistency vs human ceiling (alien + shared); annotate entropy delta
    aa = [D[s]["axes"]["similarity"][GP]["scanmatch_aa"] for s in SLUGS]
    ax[2].bar(x, aa, color=cols, edgecolor="white")
    ax[2].axhline(H["scanmatch_ceiling"], color=HUMANC, ls="--", lw=2.4,
                  label=f"human$\\leftrightarrow$human ceiling {H['scanmatch_ceiling']:.2f}")
    ax[2].set_ylim(0, 1.0); ax[2].set_title("GAZE\n(self-consistency): non-human $\\gg$ ceiling, shared", fontsize=14, fontweight="bold")
    ax[2].set_ylabel("agent$\\leftrightarrow$agent ScanMatch"); ax[2].legend(fontsize=10, loc="lower left")
    for i, s in enumerate(SLUGS):
        de = D[s]["axes"]["dynamics"][GP]["signature"]["gaze_entropy"]["delta"]
        ax[2].text(i, aa[i] + .02, f"entropy $\\delta${de:+.2f}", ha="center", fontsize=9)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(labs, rotation=18, ha="right", fontsize=10); sns.despine(ax=a)
    fig.suptitle("MLLMs match/beat humans on decision and finding, diverge on gaze",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(); save(fig, out, "fig_dissociation.png")


def fig_gist(out):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for s in SLUGS:
        t1 = [D[s]["axes"]["finding"][f"fov-gaussian-k{k}"]["tfp1"] for k in KVALS]
        te = [D[s]["axes"]["finding"][f"fov-gaussian-k{k}"]["tfpend"] for k in KVALS]
        ax.plot(KVALS, t1, "-o", color=PAL[LAB[s]], lw=2.6, ms=8, label=f"{LAB[s]} TFP@1")
        ax.plot(KVALS, te, "--s", color=PAL[LAB[s]], lw=2.2, ms=7, alpha=.8)
    ax.axhline(H["tfp1"], color=HUMANC, ls=":", lw=2); ax.axhline(H["tfpend"], color="#999", ls=":", lw=2)
    ax.text(KVALS[-1], H["tfp1"]+.01, f"human TFP@1 {H['tfp1']:.2f}", ha="right", fontsize=11, color=HUMANC)
    ax.text(KVALS[-1], H["tfpend"]+.01, f"human TFP-end {H['tfpend']:.2f}", ha="right", fontsize=11, color="#777")
    ax.set_xscale("log", base=2); ax.set_xticks(KVALS); ax.set_xticklabels(KVALS)
    ax.set_xlabel("gaussian gist-$k$"); ax.set_ylabel("target-fixation probability"); ax.set_ylim(0, 1.04)
    ax.set_title("TFP@1 and TFP-end fall together for every model",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, ncol=1); sns.despine(ax=ax)
    fig.tight_layout(); save(fig, out, "fig_gist.png")


def fig_pca(out):
    pc = D["_pca"]; co = pc["coords"]
    fig, ax = plt.subplots(figsize=(9, 7))
    for s in SLUGS:
        pts = [(c["x"], c["y"]) for c in co if c["slug"] == s]
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=150, color=PAL[LAB[s]], edgecolor="white", lw=1.1, alpha=.9, label=LAB[s], zorder=3)
    hp = [c for c in co if c["slug"] == "human"][0]
    ax.scatter([hp["x"]], [hp["y"]], marker="*", s=900, color=HUMANC, edgecolor="white", lw=1.5, zorder=6)
    ax.annotate("HUMAN", (hp["x"], hp["y"]), fontsize=13, fontweight="bold", xytext=(8, 6), textcoords="offset points")
    v = pc["explained_variance_ratio"]
    ax.set_xlabel(f"PC1 ({v[0]*100:.0f}%): scanpath length + gaze entropy", fontsize=11)
    ax.set_ylabel(f"PC2 ({v[1]*100:.0f}%): saccade amp $-$ refixation", fontsize=11)
    ax.set_title("Gaze-signature space (PCA): human vs. model groups by condition", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11); ax.margins(.15); sns.despine(ax=ax)
    fig.tight_layout(); save(fig, out, "fig_pca.png")


def fig_tfp_curves(out):
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = list(range(len(H["tfp_curve"])))
    ax.plot(xs, H["tfp_curve"], "--o", color=HUMANC, lw=3, ms=6, label="human")
    for s in SLUGS:
        for cond, dash in [(SHARP, "-"), (GP, ":")]:
            cv = D[s]["axes"]["finding"][cond]["tfp_curve"]
            ax.plot(xs, cv, dash, color=PAL[LAB[s]], lw=2.4, label=f"{LAB[s]} {cond.replace('fov-','').replace('geisler_perry','GP')}")
    ax.set_xlim(0, 8); ax.set_ylim(0, 1.02); ax.set_xlabel("saccade $n$"); ax.set_ylabel("cumulative TFP")
    ax.set_title("Target-fixation probability by saccade\n(models reach target on saccade 1; humans climb)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=2); sns.despine(ax=ax)
    fig.tight_layout(); save(fig, out, "fig_tfp_curves.png")


def _raw_sig_gp(cfg, bbox):
    """Recompute raw per-scanpath signature at GP for boxplots (visual only)."""
    sub = os.path.join(ROOT, "data", "subset")
    funcs = {"gaze entropy (bits)": gaze_entropy,
             "saccade amp (°)": lambda f: float(np.median(saccade_amplitudes(f, PPD))) if len(f) > 1 else np.nan,
             "refixation": refixation_rate,
             "scanpath len (°)": lambda f: scanpath_length(f, PPD)}
    rows = []
    for r in json.load(open(os.path.join(sub, "human_fixations/fixations_tp.json"))):
        f = human_fix_px(r)
        for nm, fn in funcs.items(): rows.append({"metric": nm, "group": "human", "value": fn(f)})
    for s in SLUGS:
        ex = [json.loads(l) for l in open(os.path.join(ROOT, "results/baselines", s, "existence.jsonl"))]
        pass_tp = {(r["image_id"], r["target"]) for r in ex if r["condition"] == "TP" and r["correct"]}
        p = os.path.join(ROOT, "results/raw", s, GP, "scanpaths.jsonl")
        for l in open(p):
            r = json.loads(l)
            if r["condition"] != "TP" or (r["image_id"], r["target_category"]) not in pass_tp: continue
            f = agent_fix_px(r)
            for nm, fn in funcs.items(): rows.append({"metric": nm, "group": LAB[s], "value": fn(f)})
    return rows, list(funcs)


def fig_signature(out, cfg, bbox):
    import pandas as pd
    rows, metrics = _raw_sig_gp(cfg, bbox)
    df = pd.DataFrame(rows); order = ["human"] + [LAB[s] for s in SLUGS]
    pal = {"human": HUMANC, **PAL}
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
    for ax, m in zip(axes, metrics):
        sub = df[(df.metric == m) & df.value.notna()]
        sns.boxplot(data=sub, x="group", y="value", order=order, hue="group", hue_order=order,
                    palette=pal, showfliers=False, legend=False, ax=ax)
        ax.set_title(m, fontsize=13); ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25); [t.set_ha("right") for t in ax.get_xticklabels()]
    fig.suptitle("Gaze signature at the human-matched condition (GP): human vs. the three models",
                 fontsize=15, fontweight="bold")
    sns.despine(fig=fig); fig.tight_layout(); save(fig, out, "fig_signature.png")


def fig_decision(out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5)); x = np.arange(len(SLUGS)); labs = [LAB[s] for s in SLUGS]
    cols = [PAL[l] for l in labs]
    dp = [D[s]["axes"]["decision"][GP]["dprime"] for s in SLUGS]
    cr = [D[s]["axes"]["decision"][GP]["criterion"] for s in SLUGS]
    ax[0].bar(x, dp, color=cols, edgecolor="white"); ax[0].axhline(H["decision_sdt"]["dprime"], color=HUMANC, ls="--", lw=2, label="human")
    ax[0].set_title("sensitivity $d'$"); ax[0].legend(fontsize=10)
    ax[1].bar(x, cr, color=cols, edgecolor="white"); ax[1].axhline(H["decision_sdt"]["criterion"], color=HUMANC, ls="--", lw=2, label="human")
    ax[1].set_title("criterion $c$"); ax[1].legend(fontsize=10)
    for a in ax: a.set_xticks(x); a.set_xticklabels(labs, rotation=18, ha="right", fontsize=10); sns.despine(ax=a)
    fig.suptitle("DECISION axis: signal-detection $d'$ and criterion (in-harness present/absent)", fontsize=14, fontweight="bold")
    fig.tight_layout(); save(fig, out, "fig_decision.png")


def fig_polar(out):
    edges = np.linspace(-180, 180, 17); ctr = np.deg2rad((edges[:-1]+edges[1:])/2); w = np.deg2rad(360/16)
    # human hist
    hh = np.zeros(16)
    for r in json.load(open(os.path.join(ROOT, "data/subset/human_fixations/fixations_tp.json"))):
        d = saccade_directions(human_fix_px(r))
        if len(d): hh += np.histogram(d, bins=16, range=(-180, 180))[0]
    fig, axes = plt.subplots(1, len(SLUGS)+1, figsize=(4.5*(len(SLUGS)+1), 4.6), subplot_kw={"projection": "polar"})
    def plot(ax, h, title, col):
        h = np.array(h, float); h = h/h.sum() if h.sum() else h
        ax.bar(ctr, h, width=w, color=col, edgecolor="white", alpha=.85); ax.set_title(title, fontsize=12); ax.set_yticklabels([])
    plot(axes[0], hh, "human", HUMANC)
    for ax, s in zip(axes[1:], SLUGS):
        plot(ax, D[s]["axes"]["dynamics"][GP].get("sacc_dir_hist", np.ones(16)), LAB[s], PAL[LAB[s]])
    fig.suptitle("Saccade-direction distribution at GP (y grows downward)", fontsize=14, fontweight="bold")
    fig.tight_layout(); save(fig, out, "fig_polar.png")


def fig_turn(out):
    ctr = (np.linspace(-180, 180, 19)[:-1] + np.linspace(-180, 180, 19)[1:]) / 2
    ht = np.zeros(18)
    for r in json.load(open(os.path.join(ROOT, "data/subset/human_fixations/fixations_tp.json"))):
        f = human_fix_px(r)
        if len(f) > 2: ht += np.histogram(turning_angles(f), bins=18, range=(-180, 180))[0]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ctr, ht/ht.sum(), "--o", color=HUMANC, lw=2.5, label="human")
    for s in SLUGS:
        h = np.array(D[s]["axes"]["dynamics"][GP].get("turn_angle_hist", np.ones(18)), float)
        ax.plot(ctr, h/h.sum(), "-", color=PAL[LAB[s]], lw=2.4, label=LAB[s])
    ax.set_xlabel("turning angle (°)  [0=straight, ±180=reversal]"); ax.set_ylabel("fraction")
    ax.set_title("Turning-angle distribution at GP", fontsize=14, fontweight="bold"); ax.legend(fontsize=10); sns.despine(ax=ax)
    fig.tight_layout(); save(fig, out, "fig_turn.png")


def fig_gp_sharp(out):
    """GP vs sharp near-identity (Q8): each point a (sharp, GP) value; on the diagonal = no effect."""
    fig, ax = plt.subplots(figsize=(7, 7))
    for s in SLUGS:
        xs, ys = [], []
        for met, axis, key in [("tfp1", "finding", "tfp1"), ("tfpend", "finding", "tfpend"),
                               ("declared_absent", "decision", "declared_absent")]:
            xs.append(D[s]["axes"][axis][SHARP][key]); ys.append(D[s]["axes"][axis][GP][key])
        # entropy/saccade deltas (already vs human; compare sharp vs GP delta)
        for met in ["gaze_entropy", "saccade_amp"]:
            xs.append(abs(D[s]["axes"]["dynamics"][SHARP]["signature"][met]["delta"]))
            ys.append(abs(D[s]["axes"]["dynamics"][GP]["signature"][met]["delta"]))
        ax.scatter(xs, ys, s=110, color=PAL[LAB[s]], edgecolor="white", lw=1, alpha=.9, label=LAB[s], zorder=3)
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=2)
    ax.set_xlabel("metric under sharp"); ax.set_ylabel("metric under GP")
    ax.set_title("GP $\\approx$ sharp: the human-matched foveation has\nnegligible effect (points on the diagonal)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); sns.despine(ax=ax)
    fig.tight_layout(); save(fig, out, "fig_gp_sharp.png")


def fig_spatial_temporal(out):
    """Q9: spatial prior matched (density CC, |center-bias delta| small) while temporal alien (entropy |delta|, self-consistency)."""
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for s in SLUGS:
        spatial = D[s]["axes"]["finding"][GP].get("density", {}).get("cc")     # high = matched
        temporal = abs(D[s]["axes"]["dynamics"][GP]["signature"]["gaze_entropy"]["delta"])  # high = alien
        if spatial is None: continue
        ax.scatter([spatial], [temporal], s=240, color=PAL[LAB[s]], edgecolor="white", lw=1.4, zorder=3, label=LAB[s])
        ax.annotate(LAB[s], (spatial, temporal), fontsize=10, xytext=(8, 4), textcoords="offset points")
    ax.set_xlabel("SPATIAL: fixation-density CC with human (higher = matched)")
    ax.set_ylabel("TEMPORAL: |gaze-entropy $\\delta$| (higher = non-human)")
    ax.set_title("Spatial prior matched, temporal structure non-human", fontsize=14, fontweight="bold")
    sns.despine(ax=ax); fig.tight_layout(); save(fig, out, "fig_spatial_temporal.png")


def fig_thrash(out):
    """Q11: refixation delta rises through mid-gist (thrash), declared-absent reverts up at k128 (prior reversion)."""
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()
    for s in SLUGS:
        refix = [D[s]["axes"]["dynamics"][f"fov-gaussian-k{k}"]["signature"]["refixation"]["delta"] for k in KVALS]
        decl = [D[s]["axes"]["decision"][f"fov-gaussian-k{k}"]["declared_absent"] for k in KVALS]
        ax1.plot(KVALS, refix, "-o", color=PAL[LAB[s]], lw=2.4, label=LAB[s])
        ax2.plot(KVALS, decl, "--s", color=PAL[LAB[s]], lw=1.8, alpha=.6)
    ax1.set_xscale("log", base=2); ax1.set_xticks(KVALS); ax1.set_xticklabels(KVALS)
    ax1.set_xlabel("gaussian gist-$k$"); ax1.set_ylabel("refixation $\\delta$ vs human (solid)")
    ax2.set_ylabel("TA declared-absent (dashed)"); ax2.set_ylim(0, 1)
    ax1.set_title("Refixation rises with degradation, then declared-absent rebounds at $k$=128",
                  fontsize=12.5, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper left"); sns.despine(ax=ax1, right=False)
    fig.tight_layout(); save(fig, out, "fig_thrash.png")


def fig_stratum(out):
    sts = ["near-small", "near-large", "far-small", "far-large"]
    x = np.arange(len(sts)); w = 0.8/(len(SLUGS)+1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x, [D[SLUGS[0]]["strata"][st]["human_tfp1"] for st in sts], w, color=HUMANC, label="human")
    for i, s in enumerate(SLUGS):
        ax.bar(x+(i+1)*w, [D[s]["strata"][st]["sharp_tfp1"] for st in sts], w, color=PAL[LAB[s]], label=LAB[s])
    ax.set_xticks(x+0.4); ax.set_xticklabels(sts); ax.set_ylabel("TFP@1 (sharp)"); ax.set_ylim(0, 1.05)
    ax.set_title("First-saccade targeting by eccentricity × size\n(models out-target humans in every cell)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, ncol=2); sns.despine(ax=ax)
    fig.tight_layout(); save(fig, out, "fig_stratum.png")


def fig_temp(out):
    ts = D.get("qwen3.5-35b-a3b", {}).get("temp_sweep", {})
    if not ts: return
    import re
    rows = collections.defaultdict(dict)
    for k, v in ts.items():
        m = re.match(r"fov-(.+)-t([0-9.]+)$", k)
        if m: rows[m.group(1)][float(m.group(2))] = v["scanmatch_aa"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cond, d in sorted(rows.items()):
        ts_, vs = zip(*sorted(d.items()))
        ax.plot(ts_, vs, "-o", lw=2.4, ms=8, label=cond)
    ax.axhline(H["scanmatch_ceiling"], color=HUMANC, ls="--", lw=2.2, label=f"human ceiling {H['scanmatch_ceiling']:.2f}")
    ax.set_xlabel("sampling temperature"); ax.set_ylabel("agent$\\leftrightarrow$agent ScanMatch")
    ax.set_title("Self-consistency vs temperature (anchor model, Qwen):\nstays far above ceiling even at $T$=1.0", fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=10); sns.despine(ax=ax); fig.tight_layout(); save(fig, out, "fig_temp.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/eccv_template/ECCV_2026_GazeAgent_qpaper/figures"))
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    recs_tp = json.load(open(os.path.join(ROOT, "data/subset/image_ids_tp.json")))
    bbox = {(r["image_id"], r["target_category"]): r["bbox"] for r in recs_tp}
    print("[figures] writing ->", args.out)
    fig_dissociation(args.out); fig_gist(args.out); fig_pca(args.out); fig_tfp_curves(args.out)
    fig_signature(args.out, cfg, bbox); fig_decision(args.out); fig_polar(args.out); fig_turn(args.out)
    fig_gp_sharp(args.out); fig_spatial_temporal(args.out); fig_thrash(args.out); fig_stratum(args.out)
    fig_temp(args.out)
    # bracket sample (reuse existing if present)
    import shutil
    for cand in ["docs/eccv_template/ECCV_2026_GazeAgent_3model/figures/bracket_all.png",
                 "figures/pilot/bracket_all.png"]:
        p = os.path.join(ROOT, cand)
        if os.path.exists(p): shutil.copy(p, os.path.join(args.out, "bracket_all.png")); break
    print("[figures] done")


if __name__ == "__main__":
    main()
