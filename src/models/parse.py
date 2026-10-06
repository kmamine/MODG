"""Parse a model reply into a search directive (build-plan step 3).

Extracts the next-fixation coordinate (normalized [0,1]) and the stop decision from
free-form model text. Tolerant of several output styles; tries them in priority order
and prefers the LAST directive in the text (the contract asks the model to end with one
directive line). NEVER raises — unparseable text yields action="error".
"""
from __future__ import annotations

import re
import json
from dataclasses import dataclass

_C = r"[-+]?\d*\.?\d+"  # a float
_KEYED = re.compile(r"(?i)\b(FOUND|LOOK)\b[^\n]*?x\s*=\s*(" + _C + r")\s*,?\s*y\s*=\s*(" + _C + r")")
# Some models follow the FOUND/LOOK keyword with a BARE coordinate pair ("FOUND: 0.80, 0.15")
# instead of the requested x=..,y=.. form. Accept that too: keyword directly followed (optionally
# ": " / "-") by two same-line floats. Strict superset of _KEYED — never fires when x=/y= is present
# (the float branch can't begin at the "x"), so models that comply are unaffected.
_KEYED_BARE = re.compile(r"(?i)\b(FOUND|LOOK)\b\s*[:\-]?\s*(" + _C + r")\s*[, ]\s*(" + _C + r")")
_ABSENT = re.compile(r"(?i)\bABSENT\b")
_JSON = re.compile(r"\{[^{}]*\}")
_MOLMO = re.compile(r'(?i)<points?\b[^>]*?\bx\s*=\s*"?(' + _C + r')"?[^>]*?\by\s*=\s*"?(' + _C + r')"?')
_BARE_KV = re.compile(r"(?i)\bx\s*=\s*(" + _C + r")\s*,?\s*y\s*=\s*(" + _C + r")")
_BARE_PAIR = re.compile(r"(?<![\w.])(" + _C + r")\s*[, ]\s*(" + _C + r")(?![\w.])")


@dataclass
class Directive:
    action: str            # "look" | "found" | "absent" | "error"
    x: float | None
    y: float | None
    clamped: bool
    parse_status: str      # "ok" | "clamped" | "absent" | "error"


def _finalize(action, x, y, molmo):
    if molmo and (x > 1.5 or y > 1.5):   # Molmo points are 0-100 percentages
        x, y = x / 100.0, y / 100.0
    cx, cy = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
    clamped = (cx != x) or (cy != y)
    return Directive(action, cx, cy, clamped, "clamped" if clamped else "ok")


_YESNO = re.compile(r"(?i)\b(yes|no)\b")


def parse_yesno(text: str) -> str:
    """Existence-baseline parser: 'yes' | 'no' | 'unknown'. Last decisive token wins
    (reasoning models end on the answer); falls back to present/absent synonyms."""
    if not text:
        return "unknown"
    ms = list(_YESNO.finditer(text))
    if ms:
        return ms[-1].group(1).lower()
    if re.search(r"(?i)\b(absent|not present|isn't|is not)\b", text):
        return "no"
    if re.search(r"(?i)\bpresent\b", text):
        return "yes"
    return "unknown"


def parse_directive(text: str) -> Directive:
    if not text or not text.strip():
        return Directive("error", None, None, False, "error")

    # 1. Explicit directives (FOUND/LOOK + coords, or ABSENT). Last one in the text wins.
    cands = [(m.start(), m.group(1).lower(), float(m.group(2)), float(m.group(3)))
             for m in _KEYED.finditer(text)]
    cands += [(m.start(), m.group(1).lower(), float(m.group(2)), float(m.group(3)))
              for m in _KEYED_BARE.finditer(text)]
    cands += [(m.start(), "absent", None, None) for m in _ABSENT.finditer(text)]
    if cands:
        cands.sort(key=lambda c: c[0])
        _, action, x, y = cands[-1]
        if action == "absent":
            return Directive("absent", None, None, False, "absent")
        return _finalize(action, x, y, molmo=False)

    # 2. JSON object with x,y (+ optional found/action/absent)
    for m in _JSON.finditer(text):
        try:
            d = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(d, dict) and "x" in d and "y" in d:
            found = d.get("found") is True or str(d.get("action", "")).lower() == "found"
            try:
                return _finalize("found" if found else "look", float(d["x"]), float(d["y"]), molmo=False)
            except (TypeError, ValueError):
                continue
        if isinstance(d, dict) and (d.get("absent") is True or str(d.get("action", "")).lower() == "absent"):
            return Directive("absent", None, None, False, "absent")

    # 3. Molmo <point x=".." y=".."> (0-100 scale)
    mol = list(_MOLMO.finditer(text))
    if mol:
        m = mol[-1]
        return _finalize("look", float(m.group(1)), float(m.group(2)), molmo=True)

    # 4. Bare "x=.., y=.."
    kv = list(_BARE_KV.finditer(text))
    if kv:
        m = kv[-1]
        return _finalize("look", float(m.group(1)), float(m.group(2)), molmo=False)

    # 5. Bare float pair
    fp = list(_BARE_PAIR.finditer(text))
    if fp:
        m = fp[-1]
        return _finalize("look", float(m.group(1)), float(m.group(2)), molmo=False)

    return Directive("error", None, None, False, "error")
