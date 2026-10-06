"""Unit + fuzz tests for src/models/parse.py (build-plan step-3 gate)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.models.parse import parse_directive  # noqa: E402

VALID = [
    ("LOOK: x=0.31, y=0.72", "look", 0.31, 0.72, "ok"),
    ("FOUND: x=0.1, y=0.9", "found", 0.1, 0.9, "ok"),
    ("ABSENT", "absent", None, None, "absent"),
    ('{"x":0.4,"y":0.6}', "look", 0.4, 0.6, "ok"),
    ('{"found":true,"x":0.4,"y":0.6}', "found", 0.4, 0.6, "ok"),
    ("I scan the shelf, nothing yet.\nLOOK: x=0.5, y=0.5", "look", 0.5, 0.5, "ok"),
    ('<point x="34.0" y="88.0">mug</point>', "look", 0.34, 0.88, "ok"),
    ("0.5 0.5", "look", 0.5, 0.5, "ok"),
    # last directive wins
    ("LOOK: x=0.1, y=0.1\nFOUND: x=0.9, y=0.8", "found", 0.9, 0.8, "ok"),
    ("It might be ABSENT, but let me check.\nLOOK: x=0.2, y=0.3", "look", 0.2, 0.3, "ok"),
    # BARE coords after the keyword (no x=/y=) — some models emit "FOUND: 0.80, 0.15"
    ("FOUND: 0.80, 0.15", "found", 0.80, 0.15, "ok"),
    ("LOOK: 0.5, 0.5", "look", 0.5, 0.5, "ok"),
    ("I see it near the cake stand.\nFOUND: 0.80, 0.15", "found", 0.80, 0.15, "ok"),
    # reasoning mentions "absent" but the bare directive is a LOOK -> keep searching (not a stop)
    ("the target is likely absent here, but I should check the rest.\nLOOK: 0.8, 0.8",
     "look", 0.8, 0.8, "ok"),
    # a genuine ABSENT (no coords) still stops
    ("Nothing matches anywhere.\nABSENT", "absent", None, None, "absent"),
]


@pytest.mark.parametrize("text,action,x,y,status", VALID)
def test_valid(text, action, x, y, status):
    d = parse_directive(text)
    assert d.action == action
    assert d.parse_status == status
    if x is not None:
        assert d.x == pytest.approx(x, abs=1e-6) and d.y == pytest.approx(y, abs=1e-6)


def test_out_of_range_is_clamped():
    d = parse_directive("x=1280, y=900")
    assert d.action == "look" and d.clamped and d.parse_status == "clamped"
    assert d.x == 1.0 and d.y == 1.0
    d2 = parse_directive("x=-0.2, y=1.5")
    assert d2.x == 0.0 and d2.y == 1.0 and d2.clamped


@pytest.mark.parametrize("text", ["", "   ", "I cannot tell where to look here.", "[VLM_CALL_FAILED]"])
def test_malformed_is_error(text):
    d = parse_directive(text)
    assert d.action == "error" and d.parse_status == "error"
    assert d.x is None and d.y is None


def test_never_raises_on_garbage():
    rng = np.random.default_rng(0)
    alphabet = list("xy=0123456789.,{}<>\"point FOUNDABSENTLOOK \n\t-+/")
    for _ in range(200):
        s = "".join(rng.choice(alphabet, size=int(rng.integers(0, 60))))
        parse_directive(s)  # must not raise


from src.models.parse import parse_yesno  # noqa: E402


@pytest.mark.parametrize("text,expect", [
    ("yes", "yes"), ("No.", "no"),
    ("I scan the shelf... I can see a bottle. Yes.", "yes"),
    ("There is no bottle anywhere here. No.", "no"),
    ("The target is absent.", "no"),
    ("It is present in the lower left.", "yes"),
    ("I think it's not present.", "no"),
    ("", "unknown"),
])
def test_parse_yesno(text, expect):
    assert parse_yesno(text) == expect
