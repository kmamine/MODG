"""Behavioural metrics — agent vs human scanpaths (build-plan step 5, reframed).

Pure functions over fixation arrays in the canonical 1680x1050 px space. Agent fixations are
normalized [0,1] (→ ×W/×H here); human X/Y are already px. The initial central fixation
(step 0) is the forced start shared by humans and agents; density/agreement metrics drop it,
TFP counts from it (n=0). Implements TFP, NumFix, fixation-density agreement (NSS/CC/KL), and
stopping. NSS/CC/KL follow the standard saliency conventions; CC is Pearson, KL is
KL(human || model).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

W, H = 1680, 1050


def agent_fix_px(record):
    """Agent scanpath record {fixations:[{x,y,step}]} (normalized) -> N×2 px array."""
    return np.array([[f["x"] * W, f["y"] * H] for f in record["fixations"]], dtype=float)


def human_fix_px(record):
    """Human raw COCO-Search18 record {X[],Y[]} (already px) -> N×2 px array."""
    return np.array(list(zip(record["X"], record["Y"])), dtype=float)


def numfix(record):
    """Number of fixations at termination (incl. the step-0 centre)."""
    return len(record["fixations"]) if "fixations" in record else int(record["length"])


def _hit_step(fix_px, bbox, tol):
    """Index of the first fixation inside bbox (expanded by tol px), or None."""
    x, y, w, h = bbox
    for i, (px, py) in enumerate(fix_px):
        if (x - tol) <= px <= (x + w + tol) and (y - tol) <= py <= (y + h + tol):
            return i
    return None


def tfp(scanpaths_px, bbox, tol, n_max):
    """Cumulative target-fixation probability by saccade n (n=0 = initial centre).
    scanpaths_px: list of N×2 px arrays. Returns array length n_max+1."""
    hits = [_hit_step(fp, bbox, tol) for fp in scanpaths_px]
    return np.array([np.mean([h is not None and h <= n for h in hits]) for n in range(n_max + 1)])


def density_map(scanpaths_px, sigma_px):
    """Gaussian fixation-density map over the canonical frame from a list of N×2 px arrays."""
    g = np.zeros((H, W), dtype=float)
    for fp in scanpaths_px:
        for px, py in fp:
            xi = min(max(int(round(px)), 0), W - 1)
            yi = min(max(int(round(py)), 0), H - 1)
            g[yi, xi] += 1.0
    return gaussian_filter(g, sigma_px) if g.any() else g


def nss(model_map, human_points_px):
    """Normalized Scanpath Saliency: mean of the z-scored model map at human fixation points."""
    s = (model_map - model_map.mean()) / (model_map.std() + 1e-12)
    vals = []
    for px, py in human_points_px:
        xi = min(max(int(round(px)), 0), W - 1)
        yi = min(max(int(round(py)), 0), H - 1)
        vals.append(s[yi, xi])
    return float(np.mean(vals)) if vals else float("nan")


def cc(map_a, map_b):
    """Pearson correlation between two density maps."""
    a = map_a.ravel() - map_a.mean()
    b = map_b.ravel() - map_b.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()) + 1e-12
    return float((a * b).sum() / denom)


def kl(human_map, model_map, eps=1e-12):
    """KL(human || model) over the two maps normalized to probability distributions."""
    p = human_map.ravel().astype(float); p = p / (p.sum() + eps) + eps
    q = model_map.ravel().astype(float); q = q / (q.sum() + eps) + eps
    return float(np.sum(p * np.log(p / q)))


def stopping(records, condition):
    """NumFix summary + (TA) the stopping breakdown that is the sharpest divergence axis."""
    n = [numfix(r) for r in records]
    out = {"n_episodes": len(n), "numfix_mean": float(np.mean(n)),
           "numfix_median": float(np.median(n))}
    if condition == "TA":
        reasons = [r.get("stop_reason") for r in records]
        out["frac_declared_absent"] = float(np.mean([x == "not_present" for x in reasons]))
        out["frac_hit_cap"] = float(np.mean([x == "max_fixations" for x in reasons]))
        out["frac_false_present"] = float(np.mean([x == "found" for x in reasons]))
    return out


# ===================== group A extras: NumFix2T, TFP-AUC =====================
def numfix2t(fix_px, bbox, tol):
    """Fixations to first target fixation (TP); NaN if the target is never fixated."""
    h = _hit_step(np.asarray(fix_px, float), bbox, tol)
    return float(h) if h is not None else float("nan")


def tfp_auc(curve):
    """Scalar search efficiency = mean of the TFP curve (normalized area)."""
    return float(np.mean(curve)) if len(curve) else float("nan")


# ===================== group B: intrinsic spatial signature =====================
# Per-scanpath properties (N×2 px); distributions are compared agent-vs-human in compare.py.
def saccade_amplitudes(fix_px, ppd=30.0):
    f = np.asarray(fix_px, float)
    return np.linalg.norm(np.diff(f, axis=0), axis=1) / ppd if len(f) > 1 else np.array([])


def saccade_directions(fix_px):
    """Saccade direction in degrees (atan2(dy,dx); y grows downward). One per saccade."""
    f = np.asarray(fix_px, float)
    if len(f) < 2:
        return np.array([])
    d = np.diff(f, axis=0)
    return np.degrees(np.arctan2(d[:, 1], d[:, 0]))


def turning_angles(fix_px):
    """Change of direction between successive saccades, wrapped to [-180,180] (meander vs directed)."""
    dirs = saccade_directions(fix_px)
    if len(dirs) < 2:
        return np.array([])
    return (np.diff(dirs) + 180) % 360 - 180


def scanpath_length(fix_px, ppd=30.0):
    """Total gaze path length in degrees of visual angle."""
    return float(saccade_amplitudes(fix_px, ppd).sum())


def bbox_area(fix_px, ppd=30.0):
    f = np.asarray(fix_px, float)
    return float(np.ptp(f[:, 0]) * np.ptp(f[:, 1]) / (ppd * ppd)) if len(f) > 1 else 0.0


def hull_area(fix_px, ppd=30.0):
    """Convex-hull area of the fixations (dva^2) — spatial coverage/exploration extent."""
    from scipy.spatial import ConvexHull
    f = np.unique(np.asarray(fix_px, float), axis=0)
    if len(f) < 3:
        return 0.0
    try:
        return float(ConvexHull(f).volume / (ppd * ppd))   # 2-D ConvexHull.volume == area
    except Exception:                                       # collinear points etc.
        return 0.0


def spread_2d(fix_px, ppd=30.0):
    f = np.asarray(fix_px, float)
    return float(np.hypot(f[:, 0].std(), f[:, 1].std()) / ppd) if len(f) > 1 else 0.0


def gaze_entropy(fix_px, xbin=14, ybin=9):
    """Shannon entropy (bits) of the fixation distribution over an xbin×ybin grid."""
    f = np.asarray(fix_px, float)
    if len(f) == 0:
        return 0.0
    xs = np.clip((f[:, 0] / (W / xbin)).astype(int), 0, xbin - 1)
    ys = np.clip((f[:, 1] / (H / ybin)).astype(int), 0, ybin - 1)
    counts = np.bincount(ys * xbin + xs, minlength=xbin * ybin).astype(float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def center_bias(fix_px, ppd=30.0):
    """Mean fixation eccentricity from image center (dva)."""
    f = np.asarray(fix_px, float)
    if len(f) == 0:
        return float("nan")
    return float(np.hypot(f[:, 0] - W / 2, f[:, 1] - H / 2).mean() / ppd)


def refixation_rate(fix_px, xbin=14, ybin=9):
    """Fraction of fixations (after the first) landing in an already-visited grid cell.
    Intrinsic inhibition-of-return signature: humans rarely re-visit, runaway agents may."""
    f = np.asarray(fix_px, float)
    if len(f) < 2:
        return 0.0
    xs = np.clip((f[:, 0] / (W / xbin)).astype(int), 0, xbin - 1)
    ys = np.clip((f[:, 1] / (H / ybin)).astype(int), 0, ybin - 1)
    cells = (ys * xbin + xs).tolist()
    seen, revisits = set(), 0
    for c in cells:
        if c in seen:
            revisits += 1
        seen.add(c)
    return float(revisits / (len(cells) - 1))
