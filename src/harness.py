"""Agentic visual-search loop (build-plan step 4).

Renders a foveated glimpse at the current fixation, prompts the model in a multi-turn
chat that keeps prior glimpse images (bounded by the IoR memory window), parses the
next coordinate / stop decision, appends, repeats. The harness injects NO search
policy: where-to-look and when-to-stop are entirely the model's. It NEVER auto-stops on
a target hit; the fixation cap is a safety bound only.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.foveation import render_foveated  # noqa: E402
from src.models.parse import parse_directive  # noqa: E402
from src.models.client import encode_image, image_block, text_block  # noqa: E402

SYSTEM_PROMPT = (
    "You are controlling a single eye that searches a photograph for a specific object.\n"
    "You can only see clearly at the point you are currently looking; everything else is\n"
    "blurred, with sharpness falling off the farther it is from your gaze, like human\n"
    "peripheral vision. To inspect another region you must move your gaze there.\n\n"
    "Each turn you receive the image as it currently looks from your gaze point. Earlier\n"
    "turns show where you looked before and what you saw; use that history to decide where\n"
    "to look next and to avoid re-checking the same spots.\n\n"
    "Coordinates are normalized: x and y are each between 0.0 and 1.0. (0,0) is the\n"
    "top-left corner, (1,1) the bottom-right; x grows rightward, y grows downward.\n\n"
    "Your job is a PRESENT/ABSENT decision: is the target object in this image? On every turn\n"
    "do exactly one of:\n"
    "  - move your gaze to a new point to keep searching;\n"
    "  - decide the target is PRESENT (FOUND): you can clearly see it at your gaze point;\n"
    "  - decide the target is ABSENT: you are confident it is nowhere in the image.\n\n"
    "Think briefly if you want, then end your reply with ONE directive line in EXACTLY one\n"
    "of these forms and nothing after it:\n"
    "  LOOK: x=<0..1>, y=<0..1>\n"
    "  FOUND: x=<0..1>, y=<0..1>\n"
    "  ABSENT"
)
USER_TURN = (
    "Search for: {target}.\nThis is your current foveated view. Where do you look next, or "
    "is the target found or absent? End with one directive line: "
    "LOOK: x=.., y=.. / FOUND: x=.., y=.. / ABSENT."
)
NUDGE = ("You have analysed this view enough — stop reasoning and give your final answer NOW as one "
         "directive line: LOOK: x=.., y=.. / FOUND: x=.., y=.. / ABSENT.")

STOP_ENUM = {"found", "not_present", "max_fixations"}  # locked schema enum


def _build_messages(turns, K, current_url, target):
    """K = total glimpses the model sees, INCLUDING the current one.

    Keep the last K-1 prior turns (with their glimpse images + the model's own replies);
    K=1 => no prior turns (memoryless, current glimpse only); K=None => full history.
    No coordinate trail and no stated gaze coordinate — memory is exactly the kept images."""
    if K is None:
        kept = turns
    else:
        n_prior = max(int(K) - 1, 0)
        kept = turns[-n_prior:] if n_prior > 0 else []
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for t in kept:
        msgs.append({"role": "user", "content": t["blocks"]})
        msgs.append({"role": "assistant", "content": t["assistant"]})
    cur = [image_block(current_url), text_block(USER_TURN.format(target=target))]
    msgs.append({"role": "user", "content": cur})
    return msgs, cur


def _call(client, messages, seed, cfg, max_tokens=None, temperature=None):
    m = cfg["model"]
    extra = {k: m[k] for k in ("top_k", "min_p") if k in m}
    return client.chat(messages, seed=seed,
                       temperature=m["temperature"] if temperature is None else temperature,
                       top_p=m["top_p"],
                       max_tokens=m["max_tokens"] if max_tokens is None else max_tokens,
                       extra_body=extra)


def run_episode(client, image, target, condition, seed, cfg, image_id):
    """Run one search episode. Returns (record, steps_log, failed)."""
    h, iorcfg = cfg["harness"], cfg.get("ior", {}) or {}
    cap = h["max_fixations_tp"] if condition == "TP" else h["max_fixations_ta"]
    K = iorcfg.get("memory_window_k")
    mask_on = bool(iorcfg.get("enabled"))
    start = tuple(h["initial_fixation"])

    fixations = [{"x": float(start[0]), "y": float(start[1]), "step": 0}]
    visited = [start]
    turns, steps_log = [], []
    stopped, stop_reason, found, failed = False, "max_fixations", False, False

    while len(fixations) < cap:
        fx, fy = fixations[-1]["x"], fixations[-1]["y"]
        glimpse = render_foveated(image, (fx, fy), cfg, visited if mask_on else None)
        url = encode_image(glimpse, fmt=h["render_format"], quality=h["jpeg_quality"],
                           max_side=cfg["model"]["image_max_side"])
        messages, cur_blocks = _build_messages(turns, K, url, target)

        resp = _call(client, messages, seed, cfg)              # single free-form call at max_tokens
        d = parse_directive(resp.content)
        retried = False
        # Failure recovery: empty/unparseable + api_ok means the reasoning did not terminate within
        # budget (over-reasoning / loop signature). Redo the step ONCE with a much larger budget and
        # a perturbed temperature + a "conclude now" nudge — to give long-but-terminating reasoning
        # room and to break a degenerate loop. (Only the failing ~1% take this path.)
        if d.action == "error" and resp.ok and h.get("on_malformed") == "retry_then_stop":
            retried = True
            mm = cfg["model"]
            messages[-1] = {"role": "user", "content": messages[-1]["content"] + [text_block(NUDGE)]}
            resp = _call(client, messages, seed, cfg,
                         max_tokens=mm.get("retry_max_tokens", mm["max_tokens"]),
                         temperature=mm.get("retry_temperature", mm["temperature"]))
            d = parse_directive(resp.content)

        steps_log.append({
            "image_id": image_id, "condition": condition, "seed": seed,
            "step": len(fixations) - 1, "retried": retried, "api_ok": resp.ok,
            "action": d.action, "x": d.x, "y": d.y, "clamped": d.clamped,
            "parse_status": d.parse_status, "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens, "latency_s": resp.latency_s,
            "raw_text": resp.content if cfg["run"].get("log_raw_responses") else None,
        })

        if not resp.ok:
            stopped, stop_reason, failed = True, "api_error", True
            break
        if d.action == "error":
            stopped, stop_reason, failed = True, "parse_error", True
            break

        turns.append({"blocks": cur_blocks, "assistant": resp.content, "coord": (fx, fy)})

        if d.action == "absent":
            stopped, stop_reason, found = True, "not_present", False
            break
        # look or found -> record the chosen coordinate as the next fixation
        fixations.append({"x": d.x, "y": d.y, "step": len(fixations)})
        visited.append((d.x, d.y))
        if d.action == "found":
            stopped, stop_reason, found = True, "found", True
            break

    if not stopped:                       # fell through -> cap
        stopped, stop_reason = True, "max_fixations"

    record = {
        "image_id": image_id, "condition": condition, "target_category": target,
        "agent": cfg["model"]["served_id"], "seed": seed, "fixations": fixations,
        "stopped": stopped, "stop_reason": stop_reason, "found": found,
    }
    return record, steps_log, failed


def validate_record(rec):
    """Assert a scanpath record satisfies the locked schema (used by tests/runner)."""
    assert rec["condition"] in {"TP", "TA"}
    assert rec["stop_reason"] in STOP_ENUM, rec["stop_reason"]
    assert isinstance(rec["found"], bool) and isinstance(rec["stopped"], bool)
    assert rec["fixations"][0] == {"x": 0.5, "y": 0.5, "step": 0}
    for i, f in enumerate(rec["fixations"]):
        assert f["step"] == i
        assert 0.0 <= f["x"] <= 1.0 and 0.0 <= f["y"] <= 1.0
    return True
