#!/usr/bin/env python3
"""Descriptive analysis + figures for the constructed COCO-Search18 subset.

Reads ONLY data/subset/ and data/subset_manifest.json (offline, no model calls,
free to re-run). Writes figures/*.png and figures/analysis_stats.json — the single
source of truth every number in docs/ is quoted from.
"""
import os
import json
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "data", "subset")
FIG = os.path.join(ROOT, "figures")
W, H = 1680, 1050
DIAG = math.hypot(W, H)
CELLS_TP = ["near-low", "near-high", "far-low", "far-high"]
CELLS_TA = ["low", "high"]
CATS = ["bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", "knife",
        "laptop", "microwave", "mouse", "oven", "potted plant", "sink", "stop sign",
        "toilet", "tv"]


def L(path):
    with open(path) as f:
        return json.load(f)


def summary(xs):
    a = np.asarray(xs, dtype=float)
    p = np.percentile(a, [25, 50, 75, 90, 95, 99])
    return {"n": int(a.size), "min": float(a.min()), "p25": float(p[0]), "median": float(p[1]),
            "mean": float(a.mean()), "p75": float(p[2]), "p90": float(p[3]),
            "p95": float(p[4]), "p99": float(p[5]), "max": float(a.max())}


def saccade_amps(fix):
    out = []
    for r in fix:
        X, Y = r["X"], r["Y"]
        for i in range(1, len(X)):
            out.append(math.hypot(X[i] - X[i - 1], Y[i] - Y[i - 1]) / DIAG)
    return out


def compute():
    man = L(os.path.join(ROOT, "data", "subset_manifest.json"))
    stats = {"seed": man["seed"], "pool": man["pool"], "canonical_size": [W, H],
             "image_center": [W // 2, H // 2], "shortfalls": man["shortfalls"],
             "thresholds": man["thresholds"], "tp": {}, "ta": {}}

    for cond in ("tp", "ta"):
        ids = L(os.path.join(SUB, f"image_ids_{cond}.json"))
        fix = L(os.path.join(SUB, "human_fixations", f"fixations_{cond}.json"))
        d = stats[cond]
        d["n_images"] = len(ids)
        d["n_scanpaths"] = len(fix)
        d["subjects_per_image"] = len({(r["task"], r["name"]) for r in fix}) and \
            len(fix) // len({(r["task"], r["name"]) for r in fix})
        d["accuracy_pct"] = round(100 * sum(r["correct"] for r in fix) / len(fix), 1)
        d["scanpath_length"] = summary([r["length"] for r in fix])
        d["max_fixation"] = int(max(r["length"] for r in fix))
        d["saccade_amplitude_diag"] = summary(saccade_amps(fix))
        d["clutter_object_count"] = summary([m["object_count"] for m in ids])
        d["initial_fixation_mean"] = [round(float(np.mean([r["X"][0] for r in fix])), 1),
                                      round(float(np.mean([r["Y"][0] for r in fix])), 1)]
        d["per_category_selected"] = {c: sum(m["target_category"] == c for m in ids) for c in CATS}
        cells = CELLS_TP if cond == "tp" else CELLS_TA
        d["per_stratum"] = {
            c: {cell: sum(m["target_category"] == c and m["stratum"] == cell for m in ids)
                for cell in cells} for c in CATS}
        if cond == "tp":
            d["rt_ms"] = summary([r["RT"] for r in fix])
            d["target_area_pct"] = summary([(m["bbox"][2] * m["bbox"][3]) / (W * H) * 100 for m in ids])
            d["eccentricity"] = summary([m["eccentricity"] for m in ids])
        else:
            fot = [r.get("fixOnTarget") for r in fix if r.get("fixOnTarget") is not None]
            d["fix_on_target_rate_pct"] = round(100 * sum(bool(x) for x in fot) / len(fot), 1) if fot else 0.0
    return stats


# ----------------------------- figures -----------------------------
def fig_category_balance(stats):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    y = np.arange(len(CATS))
    for ax, cond, cells, cmap in [(axes[0], "tp", CELLS_TP, plt.cm.Blues),
                                  (axes[1], "ta", CELLS_TA, plt.cm.Greens)]:
        ps = stats[cond]["per_stratum"]
        left = np.zeros(len(CATS))
        colors = cmap(np.linspace(0.45, 0.9, len(cells)))
        for j, cell in enumerate(cells):
            vals = np.array([ps[c][cell] for c in CATS])
            ax.barh(y, vals, left=left, color=colors[j], label=cell, edgecolor="white")
            left += vals
        ax.set_yticks(y); ax.set_yticklabels(CATS)
        ax.set_xlabel("images selected")
        ax.set_title(f"{cond.upper()}  (n={stats[cond]['n_images']})")
        ax.legend(fontsize=8, title="stratum"); ax.invert_yaxis()
    fig.suptitle("Per-category subset balance by stratum cell")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_category_balance.png"), dpi=120); plt.close(fig)


def fig_stratum_heatmap(stats):
    ps = stats["tp"]["per_stratum"]
    M = np.array([[ps[c][cell] for cell in CELLS_TP] for c in CATS])
    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(M, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(CELLS_TP))); ax.set_xticklabels(CELLS_TP, rotation=30, ha="right")
    ax.set_yticks(range(len(CATS))); ax.set_yticklabels(CATS)
    for i in range(len(CATS)):
        for j in range(len(CELLS_TP)):
            ax.text(j, i, int(M[i, j]), ha="center", va="center",
                    color="white" if M[i, j] > M.max() / 2 else "black", fontsize=9)
    ax.set_title("TP stratum occupancy (target = 2/cell)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="images")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_stratum_heatmap.png"), dpi=120); plt.close(fig)


def _overlaid_hist(fname, tp_vals, ta_vals, xlabel, title, bins, vlines=None, logx=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tp_vals, bins=bins, alpha=0.6, label="TP", color="#3b6fb5", density=True)
    ax.hist(ta_vals, bins=bins, alpha=0.6, label="TA", color="#3f9b54", density=True)
    if vlines:
        for x, txt, col in vlines:
            ax.axvline(x, color=col, ls="--", lw=1.3)
            ax.text(x, ax.get_ylim()[1] * 0.92, txt, rotation=90, va="top", ha="right", fontsize=8, color=col)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel("density"); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=120); plt.close(fig)


def fig_scanpath_length(stats, fixtp, fixta):
    _overlaid_hist("fig_scanpath_length.png",
                   [r["length"] for r in fixtp], [r["length"] for r in fixta],
                   "scanpath length (number of fixations)", "Human scanpath length",
                   bins=np.arange(1, 47, 1),
                   vlines=[(stats["tp"]["max_fixation"], f"TP cap={stats['tp']['max_fixation']}", "#3b6fb5"),
                           (stats["ta"]["max_fixation"], f"TA cap={stats['ta']['max_fixation']}", "#3f9b54")])


def fig_clutter(fixids_tp, fixids_ta):
    _overlaid_hist("fig_clutter.png",
                   [m["object_count"] for m in fixids_tp], [m["object_count"] for m in fixids_ta],
                   "COCO object instances per image (clutter)", "Clutter (object count)",
                   bins=np.arange(0, 46, 2))


def fig_saccade(fixtp, fixta):
    _overlaid_hist("fig_saccade_amplitude.png", saccade_amps(fixtp), saccade_amps(fixta),
                   "saccade amplitude (fraction of image diagonal)", "Human saccade amplitude",
                   bins=np.linspace(0, 0.6, 40))


def fig_eccentricity(ids_tp):
    vals = [m["eccentricity"] for m in ids_tp]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=np.linspace(0.15, 0.55, 30), color="#3b6fb5", alpha=0.8)
    ax.axvline(float(np.median(vals)), color="k", ls="--", lw=1.3, label=f"overall median={np.median(vals):.2f}")
    ax.set_xlabel("target eccentricity (normalized dist. from image centre)")
    ax.set_ylabel("images"); ax.set_title("TP target eccentricity (bins split per-category)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_eccentricity.png"), dpi=120); plt.close(fig)


def fig_target_size(ids_tp):
    vals = [(m["bbox"][2] * m["bbox"][3]) / (W * H) * 100 for m in ids_tp]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=np.linspace(0, 10, 30), color="#b5563b", alpha=0.85)
    ax.axvline(float(np.median(vals)), color="k", ls="--", lw=1.3, label=f"median={np.median(vals):.1f}%")
    ax.set_xlabel("target area (% of image)"); ax.set_ylabel("images")
    ax.set_title("TP target size"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_target_size.png"), dpi=120); plt.close(fig)


def fig_initial_fixation(fixtp, fixta):
    xs = [r["X"][0] for r in fixtp + fixta]
    ys = [r["Y"][0] for r in fixtp + fixta]
    fig, ax = plt.subplots(figsize=(7, 5))
    h = ax.hist2d(xs, ys, bins=[np.linspace(0, W, 48), np.linspace(0, H, 30)], cmap="magma")
    ax.plot(W / 2, H / 2, "c+", ms=14, mew=2, label="image centre")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_xlabel("x (px, 1680 wide)"); ax.set_ylabel("y (px, 1050 tall)")
    ax.set_title("Initial fixation density (centre bias)"); ax.legend(loc="upper right")
    fig.colorbar(h[3], ax=ax, label="scanpaths"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_initial_fixation.png"), dpi=120); plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    stats = compute()
    with open(os.path.join(FIG, "analysis_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    ids_tp = L(os.path.join(SUB, "image_ids_tp.json"))
    ids_ta = L(os.path.join(SUB, "image_ids_ta.json"))
    fixtp = L(os.path.join(SUB, "human_fixations", "fixations_tp.json"))
    fixta = L(os.path.join(SUB, "human_fixations", "fixations_ta.json"))

    fig_category_balance(stats)
    fig_stratum_heatmap(stats)
    fig_scanpath_length(stats, fixtp, fixta)
    fig_clutter(ids_tp, ids_ta)
    fig_saccade(fixtp, fixta)
    fig_eccentricity(ids_tp)
    fig_target_size(ids_tp)
    fig_initial_fixation(fixtp, fixta)

    pngs = sorted(f for f in os.listdir(FIG) if f.endswith(".png"))
    print(f"Wrote figures/analysis_stats.json and {len(pngs)} figures:")
    for p in pngs:
        print("  figures/" + p)


if __name__ == "__main__":
    main()
