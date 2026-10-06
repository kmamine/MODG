#!/usr/bin/env python3
"""Resumable batch runner for the agentic search harness (build-plan step 6).

Iterates the frozen subset x conditions x seeds, runs each episode through the harness,
and writes schema-valid scanpaths + a raw-response audit log. Resumable: skips any
(image_id, condition, seed) already present in scanpaths.jsonl. Failed/malformed
episodes are written ONLY to responses.jsonl and re-attempted on the next run, so
scanpaths.jsonl stays schema-pure. Adapts src_legacy_project_for_ref/generation/
description_pipeline.py (cache-skip + per-item append + run_config snapshot).
"""
import os
import sys
import json
import time
import hashlib
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import yaml
from PIL import Image

from src.harness import run_episode, validate_record, SYSTEM_PROMPT, USER_TURN
from src.models.client import OpenAIVisionClient, encode_image, image_block, text_block

try:
    from tqdm import tqdm
except ImportError:  # tqdm optional
    def tqdm(it, **k):
        return it


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def code_hash():
    h = hashlib.sha256()
    for rel in ("src/harness.py", "src/foveation.py", "src/models/parse.py", "src/models/client.py"):
        h.update(open(os.path.join(ROOT, rel), "rb").read())
    return h.hexdigest()[:16]


def preflight(client, cfg):
    served = cfg["model"]["served_id"]
    models = client.list_models()
    if served not in models:
        raise SystemExit(f"[preflight] '{served}' not served (have: {models}). Start scripts/serve_model.sh.")
    probe = Image.new("RGB", (16, 16), (40, 160, 40))  # solid green
    msgs = [{"role": "user", "content": [image_block(encode_image(probe, max_side=16)),
                                         text_block("What color is this image? Answer in one word.")]}]
    # Reasoning model: needs room to finish thinking before any answer reaches `content`.
    r = client.chat(msgs, seed=0, temperature=0.0, top_p=1.0, max_tokens=512,
                    extra_body={k: cfg["model"][k] for k in ("top_k", "min_p") if k in cfg["model"]})
    if not r.ok or not r.content.strip():
        raise SystemExit("[preflight] image probe failed — is the served model vision-capable?")
    print(f"[preflight] OK: {served} accepts images | probe reply: {r.content[:40]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap #episodes")
    ap.add_argument("--smoke", action="store_true", help="1 TP + 1 TA at the first seed only")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds, e.g. 0,1,2 (overrides config)")
    ap.add_argument("--memory-k", dest="memory_k", default=None,
                    help="IoR memory window: int (e.g. 3) or none/full; auto-sets the result slug")
    ap.add_argument("--slug", default=None, help="explicit result-dir slug (overrides auto)")
    ap.add_argument("--foveation", default=None,
                    help="condition: sharp|geisler_perry|gaussian|crop|periphery (auto-slugs fov-<mode>)")
    ap.add_argument("--gist-k", dest="gist_k", type=int, default=None, help="gaussian gist factor k")
    ap.add_argument("--temperature", type=float, default=None,
                    help="override sampling temperature; tags the slug -t<T> (temperature sweep)")
    ap.add_argument("--ids-tp", default=None, help="override TP id list (e.g. a pilot subset)")
    ap.add_argument("--ids-ta", default=None, help="override TA id list")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    m, run, paths = cfg["model"], cfg["run"], cfg["paths"]
    images_dir = os.path.join(ROOT, paths["images_dir"])

    # CLI overrides — mutate cfg so run_config.json records the actual params used.
    if args.foveation:
        cfg["foveation"]["mode"] = args.foveation
        if args.gist_k is not None:
            cfg["foveation"]["gist_k"] = args.gist_k
    if args.temperature is not None:
        cfg["model"]["temperature"] = args.temperature
    if args.seeds:
        run["seeds"] = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    if args.memory_k is not None:
        k = None if args.memory_k.lower() in ("none", "null", "full", "inf") else int(args.memory_k)
        cfg["ior"]["memory_window_k"] = k
    # result slug: explicit > foveation condition > memory window > config default
    if args.slug:
        cfg["ior"]["slug"] = args.slug
    elif args.foveation:
        s = f"fov-{args.foveation}"
        if args.foveation == "gaussian":
            s += f"-k{int(cfg['foveation']['gist_k'])}"
        if args.temperature is not None:           # temperature sweep -> separate dir (excluded from bracket)
            s += f"-t{args.temperature}"
        cfg["ior"]["slug"] = s
    elif args.memory_k is not None:
        k = cfg["ior"]["memory_window_k"]
        cfg["ior"]["slug"] = "fullmem_nomask" if k is None else f"mem{k}"

    if args.smoke:  # fast + isolated: short episodes, separate results dir
        cfg["harness"]["max_fixations_tp"] = cfg["harness"]["max_fixations_ta"] = 6
        cfg["ior"]["slug"] = cfg["ior"]["slug"] + "_smoke"

    api_key = os.environ.get("VLLM_API_KEY") or m["api_key"]  # token from env, never from file
    client = OpenAIVisionClient(m["base_url"], api_key, m["served_id"],
                                m["request_timeout_s"], m["retries"])
    if not args.skip_preflight:
        preflight(client, cfg)

    outdir = os.path.join(ROOT, run["results_dir"], m["slug"], cfg["ior"]["slug"])
    os.makedirs(outdir, exist_ok=True)
    sp_path = os.path.join(outdir, "scanpaths.jsonl")
    resp_path = os.path.join(outdir, "responses.jsonl")

    json.dump({"config": cfg, "served_id": m["served_id"],
               "prompts": {"system": SYSTEM_PROMPT, "user": USER_TURN},
               "code_sha256": code_hash(),
               "cap_convention": "cap counts total fixations incl. the step-0 center start",
               "note": "vLLM sampling is not bit-reproducible even with a seed; report across seeds."},
              open(os.path.join(outdir, "run_config.json"), "w"), indent=2)

    # episode identity = (image, TARGET, condition, seed): some images recur under 2 target
    # categories, so keying on image alone would silently skip the 2nd target's trial.
    done = {(r["image_id"], r["target_category"], r["condition"], r["seed"]) for r in load_jsonl(sp_path)}

    id_files = {"TP": args.ids_tp or paths["ids_tp"], "TA": args.ids_ta or paths["ids_ta"]}
    seeds = run["seeds"][:1] if args.smoke else run["seeds"]
    tasks = []
    for cond in run["conditions"]:
        ids = json.load(open(os.path.join(ROOT, id_files[cond])))
        if args.smoke:
            ids = ids[:1]
        for rec in ids:
            for seed in seeds:
                if (rec["image_id"], rec["target_category"], cond, seed) not in done:
                    tasks.append((cond, rec, seed))
    if args.limit is not None:
        tasks = tasks[:args.limit]
    print(f"[run] {len(tasks)} episodes to run -> {outdir}")

    t0 = time.perf_counter()
    n_ok = n_fail = 0
    for cond, idrec, seed in tqdm(tasks, desc="episodes"):
        path = os.path.join(images_dir, cond, idrec["target_category"], idrec["image_id"])
        image = Image.open(path).convert("RGB")
        record, steps, failed = run_episode(client, image, idrec["target_category"],
                                            cond, seed, cfg, idrec["image_id"])
        for s in steps:
            append_jsonl(resp_path, s)
        if failed:
            n_fail += 1
            continue
        validate_record(record)
        append_jsonl(sp_path, record)
        done.add((idrec["image_id"], cond, seed))
        n_ok += 1

    print(f"[run] done: {n_ok} written, {n_fail} failed (will retry next run), "
          f"{time.perf_counter() - t0:.0f}s. -> {sp_path}")


if __name__ == "__main__":
    main()
