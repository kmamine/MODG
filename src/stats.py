"""Statistics for the agent↔human comparison: effect sizes, bootstrap CIs, paired tests, and a
multivariate embedding. We lead with effect sizes + CIs (not p-values); NaNs are dropped.
"""
from __future__ import annotations

import numpy as np


def _clean(x):
    x = np.asarray(x, float)
    return x[~np.isnan(x)]


def cliffs_delta(a, b):
    """Nonparametric effect size in [-1,1]: P(a>b) − P(a<b). 0 = no difference."""
    a, b = _clean(a), _clean(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(np.sign(a[:, None] - b[None, :]).sum() / (len(a) * len(b)))


def cohens_d(a, b):
    """Standardized mean difference with pooled SD."""
    a, b = _clean(a), _clean(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def bootstrap_ci(x, stat=np.median, n=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for a statistic (default median)."""
    x = _clean(x)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = [stat(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return (float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2))))


def cluster_bootstrap_ci(values, clusters, stat=np.median, n=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI that resamples CLUSTERS with replacement, not individual rows.

    Dependency structure: our observations are NOT independent — each image yields many
    scanpaths (seeds × the 10 human subjects), so they share image-level variance. A flat
    bootstrap (resampling rows) treats them as independent and understates the CI. Here the
    image is the resampling unit: we draw images with replacement and pool every observation in
    each drawn image, so a metric's spread reflects between-image variability (the level at which
    the per-image paired design varies). `clusters` is a same-length id array aligning to `values`."""
    v = np.asarray(values, float)
    m = ~np.isnan(v)
    v = v[m]
    if len(v) == 0:
        return (float("nan"), float("nan"))
    # encode cluster ids to 1-D int codes — ids may be tuples (image,target); np.asarray on a list
    # of equal-length tuples would make a 2-D array and break the boolean masking below.
    enc = {}
    c = np.array([enc.setdefault(cl, len(enc)) for cl, keep in zip(clusters, m) if keep], dtype=int)
    uniq = np.unique(c)
    groups = {u: v[c == u] for u in uniq}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        bs.append(stat(np.concatenate([groups[u] for u in pick])))
    return (float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2))))


def cliffs_delta_ci(a_values, a_clusters, b_values, b_clusters, n=2000, alpha=0.05, seed=0):
    """Cliff's delta (a vs b) with an image-clustered bootstrap CI.

    The point estimate and CI use only clusters (images) present in BOTH groups — the honest
    per-image paired comparison. The bootstrap resamples those shared images with replacement and
    pools each group's observations within the drawn images, so the CI does NOT treat the seed×
    subject cells (or the 45 human↔human pairs) as independent. Returns {delta, lo, hi, n_clusters}."""
    av, bv = np.asarray(a_values, float), np.asarray(b_values, float)
    ma, mb = ~np.isnan(av), ~np.isnan(bv)
    av, bv = av[ma], bv[mb]
    # encode a's and b's cluster ids into ONE shared code space, so the same (image,target) maps to
    # the same code in both groups; handles tuple ids (would otherwise become a 2-D array).
    enc = {}
    ac = np.array([enc.setdefault(cl, len(enc)) for cl, keep in zip(a_clusters, ma) if keep], dtype=int)
    bc = np.array([enc.setdefault(cl, len(enc)) for cl, keep in zip(b_clusters, mb) if keep], dtype=int)
    shared = np.intersect1d(np.unique(ac), np.unique(bc))
    if len(shared) == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_clusters": 0}
    ag = {u: av[ac == u] for u in shared}
    bg = {u: bv[bc == u] for u in shared}
    point = cliffs_delta(np.concatenate([ag[u] for u in shared]),
                         np.concatenate([bg[u] for u in shared]))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        pick = rng.choice(shared, len(shared), replace=True)
        bs.append(cliffs_delta(np.concatenate([ag[u] for u in pick]),
                               np.concatenate([bg[u] for u in pick])))
    return {"delta": float(point),
            "lo": float(np.percentile(bs, 100 * alpha / 2)),
            "hi": float(np.percentile(bs, 100 * (1 - alpha / 2))),
            "n_clusters": int(len(shared))}


def paired_wilcoxon(a, b):
    """Paired Wilcoxon signed-rank on matched samples (e.g. per-image medians) -> (stat, p)."""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 1 or np.allclose(a, b):
        return (float("nan"), float("nan"))
    try:
        s, p = wilcoxon(a, b)
        return (float(s), float(p))
    except Exception:
        return (float("nan"), float("nan"))


def holm(pvals):
    """Holm–Bonferroni step-down adjusted p-values (preserves order, monotone)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    adj = np.full(m, np.nan)
    prev = 0.0
    for rank, idx in enumerate(order):
        if np.isnan(p[idx]):
            continue
        prev = max(prev, min(1.0, (m - rank) * p[idx]))
        adj[idx] = prev
    return adj


def mds_embed(vectors, seed=0):
    """Standardize per-group metric vectors and embed to 2-D via MDS (headline cluster view)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import MDS
    X = StandardScaler().fit_transform(np.asarray(vectors, float))
    try:
        return MDS(n_components=2, random_state=seed, normalized_stress="auto").fit_transform(X)
    except TypeError:                                  # older sklearn without normalized_stress
        return MDS(n_components=2, random_state=seed).fit_transform(X)


def pca_embed(vectors, n_components=2):
    """Standardize per-group metric vectors and project to n_components via PCA.

    Unlike MDS, the axes are interpretable: each PC is a fixed linear combo of the (standardized)
    input metrics. Returns (coords, loadings, explained_variance_ratio):
      coords[i]      -> 2-D position of group i
      loadings[k]    -> PC k as weights over the input features (use to name the axis)
      explained_variance_ratio[k] -> fraction of variance captured by PC k
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    X = StandardScaler().fit_transform(np.asarray(vectors, float))
    p = PCA(n_components=n_components)
    coords = p.fit_transform(X)
    return coords, p.components_, p.explained_variance_ratio_


def pc_axis_label(pc_index, loadings_k, feature_names, var_ratio_k, top=2):
    """Human-readable axis label: 'PC1 (62% var): +0.71 scanpath_length, -0.55 gaze_entropy'."""
    order = np.argsort(-np.abs(loadings_k))[:top]
    terms = ", ".join(f"{loadings_k[i]:+.2f} {feature_names[i]}" for i in order)
    return f"PC{pc_index} ({var_ratio_k*100:.0f}% var): {terms}"
