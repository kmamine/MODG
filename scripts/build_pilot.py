#!/usr/bin/env python3
"""Freeze a small pilot subset: deterministic, category-stratified 20 TP + 20 TA images
drawn from the frozen subset (round-robin over categories so all 18 are represented)."""
import os
import json
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 20


def pick(records, n):
    by = collections.defaultdict(list)
    for r in sorted(records, key=lambda r: (r["target_category"], r["image_id"])):
        by[r["target_category"]].append(r)
    cats = sorted(by)
    out, idx = [], collections.defaultdict(int)
    while len(out) < n:
        progressed = False
        for c in cats:
            if idx[c] < len(by[c]):
                out.append(by[c][idx[c]]); idx[c] += 1; progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
    return sorted(out, key=lambda r: (r["target_category"], r["image_id"]))


for cond, src in (("tp", "image_ids_tp.json"), ("ta", "image_ids_ta.json")):
    recs = json.load(open(os.path.join(ROOT, "data", src)))
    sel = pick(recs, N)
    json.dump(sel, open(os.path.join(ROOT, "data", f"pilot_ids_{cond}.json"), "w"), indent=2, sort_keys=True)
    cats = collections.Counter(r["target_category"] for r in sel)
    print(f"pilot_ids_{cond}: {len(sel)} images across {len(cats)} categories -> {dict(sorted(cats.items()))}")
