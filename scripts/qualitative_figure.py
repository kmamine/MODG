#!/usr/bin/env python3
"""Qualitative scanpath comparison grid (offline, deterministic, no model calls).

Rows = example (scene, target) trials; columns = human + the agent under four foveation
conditions. Paths are drawn over the sharp 1680x1050 scene (dimmed) so only the behaviour
differs between cells; the target box is outlined in every cell.

Scene selection is a rule, not a hand-pick: candidates are trials whose chosen human subject
answered correctly and that have a seed-0 episode in all four conditions; one trial is taken
from each of three eccentricity-clutter strata spanning the difficulty range, and within a
stratum the pick is the trial whose human scanpath length is closest to that stratum's human
median (a typical human trial, not a flattering one). The human subject shown is likewise the
median-length correct subject. Ties break on image_id / subject order, so the output is
byte-reproducible. Usage:
  python3 scripts/qualitative_figure.py [--scenes id.jpg:target,...] [--model SLUG] [--seed N]
"""
import os
import sys
import json
import argparse
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

W, H = 1680, 1050                      # canonical coordinate space (src/metrics.py)

# columns: (directory under results/raw/<model>/, display label)
CONDS = [("fov-sharp", "sharp"),
         ("fov-geisler_perry", "geisler–perry\n(human-matched)"),
         ("fov-gaussian-k32", "gaussian gist-k32"),
         ("fov-crop", "crop (fovea-only)")]
# three strata spanning the difficulty range: near target/low clutter -> far/high clutter
STRATA = ["near-low", "far-low", "far-high"]

HUMAN_C, AGENT_C, BOX_C = "#1f77b4", "#d62728", "#00a152"


def load_agent(model, cond_dir, seed):
    """{(image_id, target_category): record} for seed-0 target-present episodes."""
    path = os.path.join(ROOT, "results/raw", model, cond_dir, "scanpaths.jsonl")
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["condition"] == "TP" and r["seed"] == seed:
            out[(r["image_id"], r["target_category"])] = r
    return out


def pick_human(records):
    """Median-length correct subject; None if the trial has no correct human."""
    ok = sorted((r for r in records if r.get("correct") == 1), key=lambda r: (len(r["X"]), r["subject"]))
    return ok[len(ok) // 2] if ok else None


def select_scenes(trials, humans, agents):
    """One trial per stratum: the one whose human path length is nearest the stratum median."""
    by_stratum = collections.defaultdict(list)
    for t in trials:
        key = (t["image_id"], t["target_category"])
        if not all(key in a for a in agents.values()):
            continue
        h = pick_human(humans.get(key, []))
        if h is not None:
            by_stratum[t["stratum"]].append((t, h))

    chosen = []
    for s in STRATA:
        cands = sorted(by_stratum[s], key=lambda th: th[0]["image_id"])
        if not cands:
            continue
        median = float(np.median([len(h["X"]) for _, h in cands]))
        chosen.append(min(cands, key=lambda th: (abs(len(th[1]["X"]) - median), th[0]["image_id"])))
    return chosen


def draw_path(ax, xy, color, found, tag):
    """Numbered fixation sequence: square = first fixation, filled last marker = target found.

    A one-fixation episode (the model stopped without moving) draws only the square, so the
    outcome tag is what carries it."""
    ms, fs = (11, 6.5) if len(xy) <= 10 else (7.5, 4.5)   # shrink markers on long, thrashing paths
    ax.plot(xy[:, 0], xy[:, 1], "-", color=color, lw=1.8, alpha=0.75, zorder=3)
    ax.plot(xy[0, 0], xy[0, 1], "s", color=color, ms=ms * 0.73, mec="white", mew=1.4, zorder=4)
    for i, (x, y) in enumerate(xy[1:], start=1):
        last = i == len(xy) - 1
        fill = color if (last and found) else "white"
        ax.plot(x, y, "o", ms=ms, zorder=4, mec="white", mew=1.4, color=fill, markerfacecolor=fill)
        if last and not found:
            ax.plot(x, y, "o", ms=ms, zorder=4, mfc="none", mec=color, mew=2.4)
        ax.text(x, y, str(i), color="white" if (last and found) else color, zorder=5,
                fontsize=fs, fontweight="bold", ha="center", va="center")
    ax.text(0.015, 0.03, tag, transform=ax.transAxes, fontsize=7.5, fontweight="bold",
            color=color, ha="left", va="bottom", zorder=6,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=color, lw=0.8, alpha=0.9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-35b-a3b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenes", default=None,
                    help="override: 'id.jpg:target[:subject],...' — naming a subject shows that "
                         "observer even if they failed the trial (default: median-length correct)")
    ap.add_argument("--out", default="figures/qualitative/qualitative_scanpaths.png")
    args = ap.parse_args()

    trials = json.load(open(os.path.join(ROOT, "data/image_ids_tp.json")))
    humans = collections.defaultdict(list)
    for r in json.load(open(os.path.join(ROOT, "data/subset/human_fixations/fixations_tp.json"))):
        humans[(r["name"], r["task"])].append(r)
    agents = {c: load_agent(args.model, c, args.seed) for c, _ in CONDS}

    if args.scenes:
        by_key = {(t["image_id"], t["target_category"]): t for t in trials}
        chosen = []
        for spec in args.scenes.split(","):
            parts = spec.split(":")
            k = (parts[0], parts[1])
            h = (next(r for r in humans[k] if r["subject"] == int(parts[2])) if len(parts) > 2
                 else pick_human(humans[k]))
            chosen.append((by_key[k], h))
    else:
        chosen = select_scenes(trials, humans, agents)

    nrow, ncol = len(chosen), len(CONDS) + 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.15 * nrow))
    axes = np.atleast_2d(axes)

    for r, (trial, human) in enumerate(chosen):
        key = (trial["image_id"], trial["target_category"])
        img = np.asarray(Image.open(os.path.join(
            ROOT, "data/images/TP", trial["target_category"], trial["image_id"])).convert("RGB"))
        dim = (img * 0.62 + 255 * 0.38).astype(np.uint8)          # wash out so paths read
        bx, by, bw, bh = trial["bbox"]

        med_h = np.median([len(r["X"]) for r in humans[key] if r.get("correct") == 1])
        hxy = np.array(list(zip(human["X"], human["Y"])), float)
        h_found = human.get("correct") == 1          # gamepad response, the human's own verdict
        cells = [("human", hxy, HUMAN_C, h_found,
                  f"{'found' if h_found else 'MISSED'} · {len(hxy)} fix")]
        for cond_dir, label in CONDS:
            rec = agents[cond_dir][key]
            xy = np.array([[f["x"] * W, f["y"] * H] for f in rec["fixations"]], float)
            outcome = {"found": "found", "not_present": "declared absent",
                       "max_fixations": "hit cap"}.get(rec["stop_reason"], rec["stop_reason"])
            cells.append((label, xy, AGENT_C, bool(rec["found"]), f"{outcome} · {len(xy)} fix"))

        for c, (label, xy, color, found, tag) in enumerate(cells):
            ax = axes[r, c]
            ax.imshow(dim)
            ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec=BOX_C, lw=2.2, zorder=2))
            draw_path(ax, xy, color, found, tag)
            ax.set_xlim(0, W); ax.set_ylim(H, 0)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(label, fontsize=9.5,
                             fontweight="bold" if "human-matched" in label else "normal")
            if c == 0:
                ax.set_ylabel(f"{trial['target_category']}\n({trial['stratum']})\n"
                              f"human med. {med_h:.0f} fix", fontsize=8.5)

    fig.suptitle(
        f"Human vs. {args.model} scanpaths on the same scenes  —  green box = search target;  square = first "
        f"fixation (the agent's is the forced centre);\nnumbered circles = fixation order;  filled last marker = "
        f"target found, open = stopped without it", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    outp = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    fig.savefig(outp, dpi=300); plt.close(fig)

    print(f"wrote {args.out}")
    for trial, human in chosen:
        print(f"  {trial['image_id']}  {trial['target_category']:<12s} {trial['stratum']:<10s} "
              f"human subj {human['subject']} ({len(human['X'])} fix)")


if __name__ == "__main__":
    main()
