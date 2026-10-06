#!/usr/bin/env python3
"""Per-episode dropout accounting (Task 6). Offline; reads only saved logs.

For every (image_id, condition, seed) episode in each foveation condition, classify the outcome:
  - completed        : a schema-valid scanpath was written
  - completed_retry  : completed, but a nudge retry fired on some step (recovered malformed output)
  - dropped_parse    : attempted but never completed; terminal step was unparseable
  - dropped_api      : attempted but never completed; terminal step was an API failure
Aggregates by condition x {TP,TA} x target category. After the 10k-token fix this should be ~0;
whatever remains is characterized here. Writes results/metrics/dropout.json + prints a table.
"""
import os
import sys
import json
import argparse
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import yaml
from src.conditions import condition_dirs


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="fov-", help="condition-dir prefix to include")
    ap.add_argument("--slug", default=None, help="override config.model.slug")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    slug = args.slug or cfg["model"]["slug"]

    # image_id -> target_category (for the by-category breakdown)
    cat = {}
    for c in ("TP", "TA"):
        for r in json.load(open(os.path.join(ROOT, cfg["paths"][f"ids_{c.lower()}"]))):
            cat[r["image_id"]] = r["target_category"]

    raw = os.path.join(ROOT, "results", "raw", slug)
    conds = condition_dirs(cfg, raw, args.prefix)        # config bracket; excludes temp-sweep dirs
    if not conds:
        print(f"[dropout] no condition dirs under {raw} (prefix {args.prefix!r}).")
        return

    report = {}
    OUTCOMES = ["completed", "completed_retry", "dropped_parse", "dropped_api"]
    for cond in conds:
        completed = {(r["image_id"], r["condition"], r["seed"])
                     for r in load_jsonl(os.path.join(raw, cond, "scanpaths.jsonl"))}
        steps = load_jsonl(os.path.join(raw, cond, "responses.jsonl"))
        # group steps by episode; keep terminal step + whether any step retried
        ep_terminal, ep_retried, attempted = {}, collections.defaultdict(bool), set()
        for s in steps:
            key = (s["image_id"], s["condition"], s["seed"])
            attempted.add(key)
            ep_retried[key] |= bool(s.get("retried"))
            if key not in ep_terminal or s["step"] >= ep_terminal[key]["step"]:
                ep_terminal[key] = s

        rows = collections.Counter()                 # (TPTA, category, outcome) -> n
        for key in attempted:
            iid, c, _ = key
            category = cat.get(iid, "?")
            if key in completed:
                outcome = "completed_retry" if ep_retried[key] else "completed"
            else:
                t = ep_terminal[key]
                outcome = "dropped_api" if not t.get("api_ok", True) else "dropped_parse"
            rows[(c, category, outcome)] += 1

        n_att = len(attempted)
        n_drop = sum(v for (c, ct, o), v in rows.items() if o.startswith("dropped"))
        report[cond] = {
            "attempted": n_att,
            "completed": sum(v for (c, ct, o), v in rows.items() if o == "completed"),
            "completed_retry": sum(v for (c, ct, o), v in rows.items() if o == "completed_retry"),
            "dropped_parse": sum(v for (c, ct, o), v in rows.items() if o == "dropped_parse"),
            "dropped_api": sum(v for (c, ct, o), v in rows.items() if o == "dropped_api"),
            "dropout_rate": (n_drop / n_att) if n_att else 0.0,
            "by_cond": {c: {o: sum(v for (cc, ct, oo), v in rows.items() if cc == c and oo == o)
                            for o in OUTCOMES} for c in ("TP", "TA")},
            "dropped_by_category": {f"{c}/{ct}": v for (c, ct, o), v in sorted(rows.items())
                                    if o.startswith("dropped")},
        }

    os.makedirs(os.path.join(ROOT, "results", "metrics"), exist_ok=True)
    json.dump(report, open(os.path.join(ROOT, "results/metrics/dropout.json"), "w"), indent=2)

    # ---- console table ----
    print(f"[dropout] model={slug}")
    print(f"{'condition':20s} {'att':>5s} {'done':>5s} {'retry':>6s} {'drop_p':>7s} {'drop_api':>9s} {'rate':>7s}")
    tot = collections.Counter()
    for cond in conds:
        r = report[cond]
        for k in ("attempted", "completed", "completed_retry", "dropped_parse", "dropped_api"):
            tot[k] += r[k]
        print(f"{cond:20s} {r['attempted']:5d} {r['completed']:5d} {r['completed_retry']:6d} "
              f"{r['dropped_parse']:7d} {r['dropped_api']:9d} {r['dropout_rate']*100:6.1f}%")
    rate = (tot['dropped_parse'] + tot['dropped_api']) / max(tot['attempted'], 1)
    print(f"{'TOTAL':20s} {tot['attempted']:5d} {tot['completed']:5d} {tot['completed_retry']:6d} "
          f"{tot['dropped_parse']:7d} {tot['dropped_api']:9d} {rate*100:6.1f}%")
    drops = {k: v for cond in conds for k, v in report[cond]["dropped_by_category"].items()}
    if drops:
        print("\ndropped by condition/category:")
        for k, v in sorted(drops.items(), key=lambda x: -x[1]):
            print(f"  {k:24s} {v}")
    print("\n-> results/metrics/dropout.json")


if __name__ == "__main__":
    main()
