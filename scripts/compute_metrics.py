#!/usr/bin/env python3
"""Compute the pilot readouts from saved scanpaths + human data (offline, no model calls).

Reads results/raw/<model>/<fov-*>/scanpaths.jsonl per foveation condition, the human reference
scanpaths, target bboxes, and the existence baseline; emits NumFix distributions, TFP curves,
fixation-density agreement (NSS/CC/KL vs human), and existence accuracy. Filtering rule: search
metrics only on images the model PASSED the existence baseline.
"""
import os
import sys
import json
import argparse
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics import (agent_fix_px, human_fix_px, numfix, tfp, density_map,
                         nss, cc, kl, stopping, gaze_entropy)
from src.conditions import condition_dirs

N_TFP = 15  # saccades shown on the TFP curve


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def group_humans(path):
    g = collections.defaultdict(list)
    for r in json.load(open(path)):     # human fixations are a JSON array, not JSONL
        g[(r["name"], r["task"])].append(r)   # trial = (image, target); some images recur per target
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="fov-", help="condition-dir prefix to include")
    ap.add_argument("--slug", default=None, help="override config.model.slug (regenerate without touching live config)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    slug, ppd = (args.slug or cfg["model"]["slug"]), cfg["foveation"]["px_per_degree"]
    tol, sigma = float(ppd), float(ppd)                       # 1 dva tolerance / density sigma

    sub = os.path.join(ROOT, "data", "subset")
    humans = {"TP": group_humans(os.path.join(sub, "human_fixations/fixations_tp.json")),
              "TA": group_humans(os.path.join(sub, "human_fixations/fixations_ta.json"))}
    bbox = {(r["image_id"], r["target_category"]): r["bbox"]
            for r in json.load(open(os.path.join(sub, "image_ids_tp.json")))}
    passed = {(r["image_id"], r["target"], r["condition"]) for r in
              load_jsonl(os.path.join(ROOT, "results/baselines", slug, "existence.jsonl")) if r["correct"]}
    have_baseline = bool(passed)

    raw = os.path.join(ROOT, "results", "raw", slug)
    conds = condition_dirs(cfg, raw, args.prefix)
    if not conds:
        print(f"[metrics] no condition dirs under {raw} (prefix {args.prefix!r}); run the pilot first.")
        return
    print(f"[metrics] model={slug}  conditions={conds}  existence-filter={'on' if have_baseline else 'off'}")

    report, figdir = {}, os.path.join(ROOT, "figures", "pilot")
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results", "metrics"), exist_ok=True)

    # human TFP reference (TP)
    human_tfp = np.mean([tfp([human_fix_px(r) for r in recs], bbox[iid], tol, N_TFP)
                         for iid, recs in humans["TP"].items() if iid in bbox], axis=0)

    for cond_dir in conds:
        recs = load_jsonl(os.path.join(raw, cond_dir, "scanpaths.jsonl"))
        by = {"TP": collections.defaultdict(list), "TA": collections.defaultdict(list)}
        for r in recs:
            if not have_baseline or (r["image_id"], r["target_category"], r["condition"]) in passed:
                by[r["condition"]][(r["image_id"], r["target_category"])].append(r)
        c = report.setdefault(cond_dir, {})
        for condition in ("TP", "TA"):
            imgs = by[condition]
            flat = [r for rs in imgs.values() for r in rs]
            if not flat:
                continue
            c[condition] = {"numfix": [numfix(r) for r in flat], **stopping(flat, condition)}
            c[condition]["gaze_entropy_median"] = float(np.median([gaze_entropy(agent_fix_px(r)) for r in flat]))
            if condition == "TP":
                # TFP (agent, averaged over images) + density agreement vs human
                curves, nsss, ccs, kls = [], [], [], []
                for iid, rs in imgs.items():
                    if iid not in bbox or iid not in humans["TP"]:
                        continue
                    a_px = [agent_fix_px(r) for r in rs]
                    curves.append(tfp(a_px, bbox[iid], tol, N_TFP))
                    h_recs = humans["TP"][iid]
                    mmap = density_map([p[1:] for p in a_px], sigma)
                    hmap = density_map([human_fix_px(r)[1:] for r in h_recs], sigma)
                    hpts = np.concatenate([human_fix_px(r)[1:] for r in h_recs]) if h_recs else np.empty((0, 2))
                    nsss.append(nss(mmap, hpts)); ccs.append(cc(mmap, hmap)); kls.append(kl(hmap, mmap))
                c["TP"]["tfp"] = np.mean(curves, axis=0).tolist() if curves else None
                c["TP"]["density"] = {"NSS": float(np.mean(nsss)), "CC": float(np.mean(ccs)),
                                      "KL": float(np.mean(kls))} if nsss else None

    # ---- existence accuracy ----
    ex = load_jsonl(os.path.join(ROOT, "results/baselines", slug, "existence.jsonl"))
    exist = {}
    for c in ("TP", "TA"):
        s = [r for r in ex if r["condition"] == c]
        exist[c] = {"acc": float(np.mean([r["correct"] for r in s])) if s else None, "n": len(s)}
    report["_existence"] = exist

    def _ser(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        return str(o)
    json.dump(report, open(os.path.join(ROOT, "results/metrics/pilot_metrics.json"), "w"),
              indent=2, default=_ser)

    # ---- figures ----
    order = [d for d in conds]
    # 1. NumFix box per condition x {TP,TA}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, condition in zip(axes, ("TP", "TA")):
        data = [report[d].get(condition, {}).get("numfix", []) for d in order]
        ax.boxplot([x or [0] for x in data], labels=[d.replace("fov-", "") for d in order])
        ax.set_title(f"NumFix — {condition}"); ax.set_ylabel("fixations"); ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "numfix.png"), dpi=120); plt.close(fig)

    # 2. TFP curves (TP) + human reference
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(N_TFP + 1), human_tfp, "k--", lw=2, label="human")
    for d in order:
        t = report[d].get("TP", {}).get("tfp")
        if t:
            ax.plot(range(N_TFP + 1), t, marker="o", ms=3, label=d.replace("fov-", ""))
    ax.set_xlabel("saccade n"); ax.set_ylabel("cumulative target-fixation prob."); ax.set_ylim(0, 1)
    ax.set_title("TFP (target-present)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "tfp.png"), dpi=120); plt.close(fig)

    # 3. density agreement (NSS/CC/KL) per condition
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics3 = ["NSS", "CC", "KL"]
    xs = np.arange(len(order)); w = 0.25
    for j, mname in enumerate(metrics3):
        vals = [(report[d].get("TP", {}).get("density") or {}).get(mname, np.nan) for d in order]
        ax.bar(xs + j * w, vals, w, label=mname)
    ax.set_xticks(xs + w); ax.set_xticklabels([d.replace("fov-", "") for d in order], rotation=30)
    ax.set_title("Fixation-density agreement vs human (TP)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "density_agreement.png"), dpi=120); plt.close(fig)

    # 4. gist-k sweep: TFP@1 and TFP-end vs k (the "is there a human-like band?" figure)
    HUMAN_TFP1, HUMAN_TFPEND = 0.49, 0.93                  # COCO-Search18 TP reference
    gk = []
    for d in order:
        if d.startswith("fov-gaussian-k"):
            t = report[d].get("TP", {}).get("tfp")
            if t:
                gk.append((int(d.rsplit("-k", 1)[1]), t[1], t[-1]))
    if len(gk) >= 2:
        gk.sort()
        ks = [k for k, _, _ in gk]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ks, [a for _, a, _ in gk], "o-", color="C0", label="TFP@1 (first-look hit)")
        ax.plot(ks, [b for _, _, b in gk], "s-", color="C1", label="TFP-end (final hit)")
        ax.axhline(HUMAN_TFP1, color="C0", ls=":", lw=1.5, label=f"human TFP@1 ≈ {HUMAN_TFP1}")
        ax.axhline(HUMAN_TFPEND, color="C1", ls=":", lw=1.5, label=f"human TFP-end ≈ {HUMAN_TFPEND}")
        # anchors: sharp / GP first-look for context
        for d, col, mk in [("fov-sharp", "gray", "*"), ("fov-geisler_perry", "k", "x")]:
            t = report.get(d, {}).get("TP", {}).get("tfp")
            if t:
                ax.scatter([ks[0]], [t[1]], color=col, marker=mk, s=80, zorder=5,
                           label=f"{d.replace('fov-','')} TFP@1")
        ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
        ax.set_xlabel("gaussian gist-k (peripheral acuity ÷ k)"); ax.set_ylabel("target-fixation prob.")
        ax.set_ylim(0, 1.02); ax.set_title("Gist-k sweep: is there a human-like band?\n"
                                           "(does TFP-end collapse before TFP@1 falls to ~human?)")
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout(); fig.savefig(os.path.join(figdir, "gist_sweep.png"), dpi=120); plt.close(fig)
        print(f"[metrics] gist-k sweep figure: k={ks}")

    # 5. target-absent stopping: declared-absent / false-present / hit-cap fractions per condition
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    declared = [report[d].get("TA", {}).get("frac_declared_absent", 0) or 0 for d in order]
    falsep = [report[d].get("TA", {}).get("frac_false_present", 0) or 0 for d in order]
    capf = [report[d].get("TA", {}).get("frac_hit_cap", 0) or 0 for d in order]
    xs = np.arange(len(order)); labels = [d.replace("fov-", "") for d in order]
    axL.bar(xs, declared, label="declared absent")
    axL.bar(xs, falsep, bottom=declared, label="false 'present'")
    axL.bar(xs, capf, bottom=[a + b for a, b in zip(declared, falsep)], label="hit cap")
    axL.set_xticks(xs); axL.set_xticklabels(labels, rotation=35, fontsize=8)
    axL.set_ylabel("fraction of TA episodes"); axL.set_title("Target-absent stopping decision"); axL.legend(fontsize=8)
    ta_med = [report[d].get("TA", {}).get("numfix_median", np.nan) for d in order]
    axR.bar(xs, ta_med, color="C3")
    axR.axhline(5, color="k", ls="--", lw=1.5, label="human ≈ 5 fixations")
    axR.set_xticks(xs); axR.set_xticklabels(labels, rotation=35, fontsize=8)
    axR.set_ylabel("median fixations before stopping"); axR.set_title("TA search extent (vs human)"); axR.legend(fontsize=8)
    fig.suptitle("Target-absent stopping — the sharpest human divergence")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "stopping_ta.png"), dpi=120); plt.close(fig)

    # 6. existence baseline: TP / TA accuracy + TA false-'yes' (yes-bias)
    ta_rows = [r for r in ex if r["condition"] == "TA"]
    ta_yes = (sum(r["answer"] == "yes" for r in ta_rows) / len(ta_rows)) if ta_rows else 0.0
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = {"TP acc": exist["TP"]["acc"] or 0, "TA acc": exist["TA"]["acc"] or 0, "TA false-'yes'": ta_yes}
    ax.bar(list(bars), list(bars.values()), color=["C0", "C0", "C3"])
    ax.set_ylim(0, 1); ax.set_ylabel("rate"); ax.set_title("One-shot existence baseline (detection ceiling + yes-bias)")
    for i, v in enumerate(bars.values()):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "existence.png"), dpi=120); plt.close(fig)

    # ---- console summary ----
    print("\n=== PILOT SUMMARY ===")
    print(f"existence: TP acc={exist['TP']['acc']}  TA acc={exist['TA']['acc']}")
    for d in order:
        tp, ta = report[d].get("TP", {}), report[d].get("TA", {})
        dens = tp.get("density") or {}
        print(f"  {d:22s} NumFix TP med={tp.get('numfix_median','-')} TA med={ta.get('numfix_median','-')} "
              f"| TFP@1={ (tp.get('tfp') or [None,None])[1]} | NSS={dens.get('NSS','-')} CC={dens.get('CC','-')} "
              f"| TA absent={ta.get('frac_declared_absent','-')} cap={ta.get('frac_hit_cap','-')}")
    print(f"\nfigures -> {figdir}/  metrics -> results/metrics/pilot_metrics.json")


if __name__ == "__main__":
    main()
