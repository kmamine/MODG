#!/usr/bin/env python3
"""Comparative figure of the full foveation bracket on one example image (offline, deterministic).

Renders the same image+gaze under every condition — sharp, geisler_perry, the gaussian gist ladder
(k4..k128), and crop — exactly as the harness would (stored images are the canonical 1680x1050 at
ppd=30, so no resize). Lays them in a labelled grid. Usage:
  python3 scripts/foveation_figure.py [IMAGE_PATH] [--gaze X Y] [--gist 4,8,16,24,32,48,64,128]
"""
import os
import sys
import copy
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.foveation import render_foveated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default="data/images/TP/bottle/000000332968.jpg",
                    help="path to a (canonical 1680x1050) image")
    ap.add_argument("--gaze", nargs=2, type=float, default=[0.5, 0.5], help="normalized gaze x y")
    ap.add_argument("--gist", default="4,8,16,24,32,48,64,128", help="comma-separated gist-k ladder")
    ap.add_argument("--out", default="figures/foveation_samples/bracket_grid.png")
    ap.add_argument("--ncol", type=int, default=4, help="columns in the grid")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    cfg["ior"]["enabled"] = False                      # no IoR mask — show the pure foveation
    img = Image.open(os.path.join(ROOT, args.image)).convert("RGB")
    gx, gy = args.gaze
    ks = [int(k) for k in args.gist.split(",")]

    # condition list: (label, mode, gist_k)
    conds = [("sharp (no foveation)", "sharp", None),
             ("geisler_perry (human-matched)", "geisler_perry", None)]
    conds += [(f"gaussian gist-k{k}", "gaussian", k) for k in ks]
    conds += [("crop (fovea-only)", "crop", None)]

    n = len(conds)
    ncol = args.ncol
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow))
    axes = np.array(axes).ravel()
    for ax, (label, mode, k) in zip(axes, conds):
        c = copy.deepcopy(cfg)
        c["foveation"]["mode"] = mode
        if k is not None:
            c["foveation"]["gist_k"] = k
        out = render_foveated(img, (gx, gy), c)
        ax.imshow(np.asarray(out))
        w, h = out.size
        ax.plot(gx * w, gy * h, "+", color="red", ms=12, mew=2)   # gaze point
        emph = " ←" if mode == "geisler_perry" else ""
        ax.set_title(label + emph, fontsize=10,
                     fontweight="bold" if mode == "geisler_perry" else "normal")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Foveation constraint bracket — only geisler_perry is human-matched (gaze = red +)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    outp = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    fig.savefig(outp, dpi=110); plt.close(fig)
    print(f"wrote {args.out}  ({n} conditions: {[c[1]+('-k'+str(c[2]) if c[2] else '') for c in conds]})")


if __name__ == "__main__":
    main()
