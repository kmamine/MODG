"""Stub-client smoke tests for src/harness.py (build-plan step-4 gate). No GPU/network."""
import os
import sys
import glob
import copy

import pytest
import yaml
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.harness import run_episode, validate_record, _build_messages  # noqa: E402
from src.models.client import ChatResponse, image_block, text_block  # noqa: E402


class StubClient:
    """Returns scripted replies; repeats the last once exhausted."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, seed, temperature, top_p, max_tokens, extra_body=None):
        text = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return ChatResponse(text, prompt_tokens=1, completion_tokens=1, latency_s=0.0, ok=True)


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))


@pytest.fixture(scope="module")
def img():
    p = sorted(glob.glob(os.path.join(ROOT, "data/subset/images/TP/*/*.jpg")))[0]
    return Image.open(p).convert("RGB")


def test_tp_found(cfg, img):
    stub = StubClient(["LOOK: x=0.3, y=0.4", "LOOK: x=0.6, y=0.5", "FOUND: x=0.62, y=0.55"])
    rec, steps, failed = run_episode(stub, img, "bottle", "TP", 0, cfg, "000000.jpg")
    assert not failed and validate_record(rec)
    assert rec["stop_reason"] == "found" and rec["found"] is True
    assert [f["step"] for f in rec["fixations"]] == list(range(len(rec["fixations"])))
    assert rec["fixations"][-1]["x"] == pytest.approx(0.62)


def test_ta_absent(cfg, img):
    stub = StubClient(["LOOK: x=0.2, y=0.2", "ABSENT"])
    rec, steps, failed = run_episode(stub, img, "bottle", "TA", 1, cfg, "000000.jpg")
    assert not failed and validate_record(rec)
    assert rec["stop_reason"] == "not_present" and rec["found"] is False


def test_cap_when_never_stops(cfg, img):
    c = copy.deepcopy(cfg); c["harness"]["max_fixations_tp"] = 5
    stub = StubClient(["LOOK: x=0.4, y=0.6"])
    rec, steps, failed = run_episode(stub, img, "bottle", "TP", 0, c, "000000.jpg")
    assert not failed
    assert rec["stop_reason"] == "max_fixations" and rec["found"] is False
    assert len(rec["fixations"]) == 5 and stub.calls == 4


def test_malformed_retries_once_then_drops(cfg, img):
    # one nudge retry, then drop on still-unparseable output (rare now that max_tokens=10k).
    stub = StubClient(["I am not sure.", "still not sure"])
    rec, steps, failed = run_episode(stub, img, "bottle", "TP", 0, cfg, "000000.jpg")
    assert failed and rec["stop_reason"] == "parse_error"
    assert stub.calls == 2 and steps[-1]["retried"] is True


# --- memory-window K semantics: K = total glimpses incl. current, no trail ---
def _fake_turns(n):
    return [{"blocks": [image_block(f"img{i}"), text_block("q")],
             "assistant": f"LOOK: x=0.{i}, y=0.{i}", "coord": (0.1 * i, 0.1 * i)}
            for i in range(n)]


def _count_images(msgs):
    return sum(1 for m in msgs if isinstance(m["content"], list)
               for b in m["content"] if isinstance(b, dict) and b.get("type") == "image_url")


def _has_trail(msgs):
    txt = " ".join(b.get("text", "") for m in msgs if isinstance(m["content"], list)
                   for b in m["content"] if isinstance(b, dict) and b.get("type") == "text")
    return ("Earlier" in txt) or ("looked at" in txt)


@pytest.mark.parametrize("K,n_turns,expected_imgs", [
    (1, 5, 1),       # memoryless: current glimpse only
    (1, 0, 1),       # first step at K=1
    (3, 5, 3),       # current + 2 prior
    (3, 1, 2),       # only 1 prior available -> current + 1
    (5, 5, 5),
    (None, 5, 6),    # full memory: all 5 prior + current
])
def test_memory_window_image_count(K, n_turns, expected_imgs):
    msgs, cur = _build_messages(_fake_turns(n_turns), K, "current_url", "bottle")
    assert _count_images(msgs) == expected_imgs
    assert not _has_trail(msgs)            # coordinate trail removed
    assert cur[0]["type"] == "image_url"   # current glimpse always present
