#!/usr/bin/env python3
"""One-shot baselines (no harness loop) — the model's "normal-conditions" reference.

EXISTENCE ceiling: full sharp image + "Is there a {target}? yes/no" on BOTH target-present and
target-absent images (TA catches the MLLM yes-bias). The detection ceiling; search behaviour is
only interpreted on cases a model passes here (filtering rule, applied in compute_metrics).

LOCALIZATION (optional, --localize, TP only): full image + "point to the {target}" -> hit if
inside the target bbox. A diagnostic separating "searches differently" from "can't localize".
"""
import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import yaml
from PIL import Image

from src.models.client import OpenAIVisionClient, encode_image, image_block, text_block
from src.models.parse import parse_yesno, parse_directive
from run_agent import preflight, load_jsonl, append_jsonl  # reuse

EXIST_PROMPT = "Is there a {target} in this image? Answer with one word: yes or no."
LOC_PROMPT = ("Point to the {target} in this image. Give its location as normalized coordinates "
              "x=<0..1>, y=<0..1> (end with that line).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--localize", action="store_true", help="also run the TP localization diagnostic")
    ap.add_argument("--ids-tp", default=None, help="override TP id list (e.g. a pilot subset)")
    ap.add_argument("--ids-ta", default=None, help="override TA id list")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--base-url", default=None, help="override config.model.base_url (serve another model elsewhere)")
    ap.add_argument("--served-id", default=None, help="override config.model.served_id")
    ap.add_argument("--slug", default=None, help="override config.model.slug (results/baselines/<slug>/)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    m, paths = cfg["model"], cfg["paths"]
    # CLI overrides let us baseline a model served on a different port/slug WITHOUT editing the live
    # config (so a concurrent run on the configured endpoint is never disturbed).
    if args.base_url:   m = {**m, "base_url": args.base_url}
    if args.served_id:  m = {**m, "served_id": args.served_id}
    if args.slug:       m = {**m, "slug": args.slug}
    images_dir = os.path.join(ROOT, paths["images_dir"])
    api_key = os.environ.get("VLLM_API_KEY") or m["api_key"]
    client = OpenAIVisionClient(m["base_url"], api_key, m["served_id"], m["request_timeout_s"], m["retries"])
    if not args.skip_preflight:
        preflight(client, cfg)

    outdir = os.path.join(ROOT, "results", "baselines", m["slug"])
    os.makedirs(outdir, exist_ok=True)
    ex_path = os.path.join(outdir, "existence.jsonl")
    loc_path = os.path.join(outdir, "localization.jsonl")
    extra = {k: m[k] for k in ("top_k", "min_p") if k in m}

    def ask(image, prompt):                                   # single free-form call, full (10k) budget
        # use the model's recommended sampling (temp 0.6 / top_p / top_k / min_p), NOT greedy temp=0:
        # greedy decoding makes this reasoning model loop/over-deliberate and exhaust the budget empty.
        url = encode_image(image, fmt=cfg["harness"]["render_format"],
                           quality=cfg["harness"]["jpeg_quality"], max_side=m["image_max_side"])
        msgs = [{"role": "user", "content": [image_block(url), text_block(prompt)]}]
        return client.chat(msgs, seed=0, temperature=m["temperature"], top_p=m["top_p"],
                           max_tokens=m["max_tokens"], extra_body=extra)

    id_files = {"TP": args.ids_tp or os.path.join(ROOT, paths["ids_tp"]),
                "TA": args.ids_ta or os.path.join(ROOT, paths["ids_ta"])}

    # key by the (image, TARGET, condition) trial — 8 images recur under two target categories, so
    # keying existence by image alone would skip the 2nd target's trial (the n=142-vs-144 gap).
    ex_done = {(r["image_id"], r.get("target"), r["condition"]) for r in load_jsonl(ex_path)}
    loc_done = {(r["image_id"], r.get("target")) for r in load_jsonl(loc_path)}

    for cond in ("TP", "TA"):
        ids = json.load(open(id_files[cond]))
        if args.limit:
            ids = ids[:args.limit]
        for rec in ids:
            iid, cat = rec["image_id"], rec["target_category"]
            path = os.path.join(images_dir, cond, cat, iid)
            if (iid, cat, cond) not in ex_done:
                r = ask(Image.open(path).convert("RGB"), EXIST_PROMPT.format(target=cat))
                ans = parse_yesno(r.content)
                truth = "yes" if cond == "TP" else "no"
                append_jsonl(ex_path, {"image_id": iid, "condition": cond, "target": cat,
                                       "answer": ans, "truth": truth, "correct": ans == truth,
                                       "raw": r.content if cfg["run"].get("log_raw_responses") else None})
            if args.localize and cond == "TP" and (iid, cat) not in loc_done:
                r = ask(Image.open(path).convert("RGB"), LOC_PROMPT.format(target=cat))
                d = parse_directive(r.content)
                hit = None
                if d.x is not None:
                    px, py = d.x * 1680, d.y * 1050
                    bx, by, bw, bh = rec["bbox"]
                    hit = bool(bx <= px <= bx + bw and by <= py <= by + bh)
                append_jsonl(loc_path, {"image_id": iid, "target": cat, "x": d.x, "y": d.y,
                                        "hit": hit, "parse_status": d.parse_status})

    # summary
    rows = load_jsonl(ex_path)
    by_cond = {}
    for c in ("TP", "TA"):
        sub = [r for r in rows if r["condition"] == c]
        by_cond[c] = (sum(r["correct"] for r in sub), len(sub))
    print(f"[baselines] existence accuracy -> "
          f"TP {by_cond['TP'][0]}/{by_cond['TP'][1]}  TA {by_cond['TA'][0]}/{by_cond['TA'][1]}")
    cats = sorted({r["target"] for r in rows})
    for cat in cats:
        s = [r for r in rows if r["target"] == cat]
        tp = [r for r in s if r["condition"] == "TP"]; ta = [r for r in s if r["condition"] == "TA"]
        print(f"  {cat:14s} TP {sum(r['correct'] for r in tp)}/{len(tp)}  "
              f"TA {sum(r['correct'] for r in ta)}/{len(ta)}  (yes-rate TA={sum(r['answer']=='yes' for r in ta)}/{len(ta)})")
    print(f"-> {ex_path}")


if __name__ == "__main__":
    main()
