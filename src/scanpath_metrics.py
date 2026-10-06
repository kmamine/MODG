"""Sequence-level scanpath-similarity metrics.

Lifted from src_legacy_project_for_ref/metrics/spatial_metrics.py + multimatch.py and adapted to
N×2 pixel arrays in the canonical 1680×1050 space (ScanMatch grid 14×9 — the dataset's config).
Two polarities: ScanMatch / MultiMatch return SIMILARITY in [0,1] (higher = more alike); DTW /
Hausdorff / Fréchet / TDE / Levenshtein return DISTANCE (lower = more alike). All take N×2 px
arrays. Used for agent↔human, human↔human (the ceiling), and agent↔agent comparisons.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import euclidean, directed_hausdorff

W, H = 1680, 1050


# ---------------- distance metrics (lower = more similar) ----------------
def dtw_distance(a, b):
    from fastdtw import fastdtw
    d, _ = fastdtw(np.asarray(a, float), np.asarray(b, float), dist=euclidean)
    return float(d)


def hausdorff_distance(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]))


def frechet_distance(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n, m = len(a), len(b)
    ca = np.full((n, m), -1.0)

    def _c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        if i == 0 and j == 0:
            ca[i, j] = euclidean(a[0], b[0])
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(i - 1, 0), euclidean(a[i], b[0]))
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(0, j - 1), euclidean(a[0], b[j]))
        else:
            ca[i, j] = max(min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), euclidean(a[i], b[j]))
        return ca[i, j]
    return float(_c(n - 1, m - 1))


def tde_distance(a, b, k=3, mode="mean"):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < k or len(b) < k:
        return float("nan")
    va = [a[i:i + k].flatten() for i in range(len(a) - k + 1)]
    vb = [b[i:i + k].flatten() for i in range(len(b) - k + 1)]
    d = [min(np.linalg.norm(x - y) for y in va) / k for x in vb]
    return float(np.mean(d) if mode == "mean" else np.max(d))


def _to_string(scanpath, xbins=12, ybins=8):
    c = np.asarray(scanpath, float)[:, :2]
    xs = np.clip((c[:, 0] / (W / xbins)).astype(int), 0, xbins - 1)
    ys = np.clip((c[:, 1] / (H / ybins)).astype(int), 0, ybins - 1)
    return "".join(chr(65 + int(b % 26)) + chr(65 + int(b // 26)) for b in ys * xbins + xs)


def levenshtein_distance(a, b, xbins=12, ybins=8):
    import editdistance
    return int(editdistance.eval(_to_string(a, xbins, ybins), _to_string(b, xbins, ybins)))


# ---------------- ScanMatch (similarity in [0,1]) ----------------
class ScanMatch:
    """Needleman–Wunsch alignment of spatially-binned fixation sequences (Cristino et al. 2010)."""

    def __init__(self, xres=W, yres=H, xbin=14, ybin=9, threshold=3.5, gap_value=0.0):
        self.xres, self.yres, self.xbin, self.ybin = xres, yres, xbin, ybin
        self.gap_value = gap_value
        n = xbin * ybin
        sm = np.zeros((n, n))
        for i in range(ybin):
            for j in range(xbin):
                for ii in range(ybin):
                    for jj in range(xbin):
                        sm[i * xbin + j, ii * xbin + jj] = np.hypot(j - jj, i - ii)
        self.sub = np.abs(sm - sm.max()) - (sm.max() - threshold)
        # vectorized per-pixel bin id (legacy built this with a 1.76M-iteration python loop)
        xs = np.clip((np.arange(xres) / (xres / xbin)).astype(int), 0, xbin - 1)
        ys = np.clip((np.arange(yres) / (yres / ybin)).astype(int), 0, ybin - 1)
        self.grid = ys[:, None] * xbin + xs[None, :]

    def _seq(self, sp):
        c = np.clip(np.asarray(sp, float)[:, :2], 0, [self.xres - 1, self.yres - 1]).astype(int)
        return self.grid[c[:, 1], c[:, 0]]

    def score(self, a, b):
        sa, sb = self._seq(a), self._seq(b)
        if len(sa) == 0 or len(sb) == 0:
            return 0.0
        n, m = len(sa), len(sb)
        F = np.zeros((n + 1, m + 1))
        F[:, 0] = self.gap_value * np.arange(n + 1)
        F[0, :] = self.gap_value * np.arange(m + 1)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                F[i, j] = max(F[i - 1, j - 1] + self.sub[sa[i - 1], sb[j - 1]],
                              F[i - 1, j] + self.gap_value, F[i, j - 1] + self.gap_value)
        scale = self.sub.max() * max(n, m)
        return float(F.max() / scale) if scale > 0 else 0.0


_SM = None


def scanmatch_score(a, b):
    """Sequence Score in [0,1] (cached default 1680×1050 / 14×9 matcher)."""
    global _SM
    if _SM is None:
        _SM = ScanMatch()
    return _SM.score(a, b)


# ---------------- MultiMatch (4 spatial dims; duration excluded — agents have none) ----------------
def _rec(xy, dur=1.0):
    a = np.asarray(xy, float)
    r = np.recarray((len(a),), dtype=[("start_x", "<f8"), ("start_y", "<f8"), ("duration", "<f8")])
    r.start_x, r.start_y, r.duration = a[:, 0], a[:, 1], dur
    return r


def multimatch(a, b, screensize=(W, H)):
    """MultiMatch (Dewhurst et al. 2012): vector/direction/length/position similarity in [0,1].
    Duration dimension is omitted (agents have no fixation durations). NaN if too short."""
    keys = ["mm_vector", "mm_direction", "mm_length", "mm_position"]
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return {k: float("nan") for k in keys}
    try:
        import multimatch_gaze as mm
        r = mm.docomparison(_rec(a), _rec(b), screensize=list(screensize))
        v = list(r[0]) if isinstance(r, tuple) else list(r)
        return {k: float(v[i]) for i, k in enumerate(keys)}   # [vector, direction, length, position, (duration)]
    except Exception:
        return {k: float("nan") for k in keys}
