"""Unit tests for src/stats.py."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.stats import (cliffs_delta, cohens_d, bootstrap_ci, paired_wilcoxon, holm, mds_embed,  # noqa: E402
                       cluster_bootstrap_ci, cliffs_delta_ci)


def test_cliffs_delta_separation():
    assert cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0      # fully above
    assert cliffs_delta([1, 2, 3], [5, 6, 7]) == -1.0     # fully below
    assert abs(cliffs_delta([1, 2, 3], [1, 2, 3])) < 1e-9  # identical -> 0


def test_cohens_d_sign():
    assert cohens_d([10, 11, 12], [1, 2, 3]) > 2          # large positive
    assert np.isnan(cohens_d([1], [2]))                   # too few


def test_bootstrap_ci_brackets_median():
    lo, hi = bootstrap_ci(list(range(101)), stat=np.median, n=500, seed=0)
    assert lo <= 50 <= hi


def test_paired_wilcoxon_and_holm():
    s, p = paired_wilcoxon([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    assert not np.isnan(p)
    adj = holm([0.01, 0.04, 0.03])
    assert np.all(np.diff(np.sort(adj)) >= -1e-12)        # monotone non-decreasing
    assert np.all(adj <= 1.0)


def test_mds_shape():
    rng = np.random.default_rng(0)
    coords = mds_embed(rng.normal(size=(7, 10)), seed=0)
    assert coords.shape == (7, 2)


def test_cluster_bootstrap_brackets_median():
    vals = list(range(101))
    clusters = list(range(101))                      # one obs per cluster
    lo, hi = cluster_bootstrap_ci(vals, clusters, stat=np.median, n=500, seed=0)
    assert lo <= 50 <= hi


def test_cluster_bootstrap_wider_than_flat_under_dependence():
    # 10 images, 20 identical obs each -> within-image variance is 0, all spread is between images.
    # Flat bootstrap sees 200 "independent" rows and underestimates; cluster bootstrap must be wider.
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 5, size=10)
    vals, clusters = [], []
    for i, c in enumerate(centers):
        vals += [c] * 20
        clusters += [i] * 20
    flo, fhi = bootstrap_ci(vals, stat=np.mean, n=1000, seed=1)
    clo, chi = cluster_bootstrap_ci(vals, clusters, stat=np.mean, n=1000, seed=1)
    assert (chi - clo) > (fhi - flo)                 # cluster CI respects dependence -> wider


def test_cliffs_delta_ci_separation_and_bracket():
    # agent fully above human, 8 images, 3 seeds each; humans 10 subjects each.
    a_vals, a_cl, b_vals, b_cl = [], [], [], []
    for img in range(8):
        a_vals += [10, 11, 12]; a_cl += [img] * 3
        b_vals += [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; b_cl += [img] * 10
    r = cliffs_delta_ci(a_vals, a_cl, b_vals, b_cl, n=500, seed=0)
    assert r["n_clusters"] == 8
    assert r["delta"] > 0.9                          # agent ~ entirely above human
    assert r["lo"] <= r["delta"] <= r["hi"]


def test_cliffs_delta_ci_no_shared_clusters():
    r = cliffs_delta_ci([1, 2], [0, 0], [3, 4], [9, 9], n=100, seed=0)
    assert r["n_clusters"] == 0 and np.isnan(r["delta"])


def test_clustered_ci_accepts_tuple_cluster_ids():
    # cluster ids are (image, target) tuples (the trial key) — must NOT become a 2-D array.
    a_vals, a_cl, b_vals, b_cl = [], [], [], []
    for img in range(6):
        a_vals += [10, 11]; a_cl += [("im%d" % img, "cat")] * 2
        b_vals += [1, 2, 3]; b_cl += [("im%d" % img, "cat")] * 3
    lo, hi = cluster_bootstrap_ci(a_vals, a_cl, stat=np.median, n=200, seed=0)
    assert lo <= 10.5 <= hi
    r = cliffs_delta_ci(a_vals, a_cl, b_vals, b_cl, n=200, seed=0)
    assert r["n_clusters"] == 6 and r["delta"] > 0.9 and r["lo"] <= r["delta"] <= r["hi"]
