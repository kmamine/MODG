"""Unit tests for src/scanpath_metrics.py (sanity: self-similarity maximal, distance self=0)."""
import os
import sys
import math

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.scanpath_metrics import (scanmatch_score, dtw_distance, hausdorff_distance,  # noqa: E402
                                  frechet_distance, tde_distance, levenshtein_distance, multimatch)

A = np.array([[840, 525], [400, 300], [1200, 700], [200, 800], [1500, 200]], float)
B = np.array([[840, 525], [420, 310], [1180, 690]], float)


def test_scanmatch_self_is_one():
    assert abs(scanmatch_score(A, A) - 1.0) < 1e-9


def test_scanmatch_diff_in_range():
    s = scanmatch_score(A, B)
    assert 0.0 <= s < 1.0


def test_distance_metrics_self_zero():
    assert dtw_distance(A, A) == 0.0
    assert hausdorff_distance(A, A) == 0.0
    assert frechet_distance(A, A) == 0.0
    assert tde_distance(A, A) < 1e-6
    assert levenshtein_distance(A, A) == 0


def test_distance_metrics_positive_for_diff():
    assert dtw_distance(A, B) > 0
    assert hausdorff_distance(A, B) > 0
    assert frechet_distance(A, B) > 0
    assert levenshtein_distance(A, B) > 0


def test_multimatch_self_high():
    r = multimatch(A, A)
    assert set(r) == {"mm_vector", "mm_direction", "mm_length", "mm_position"}
    assert all(v > 0.99 for v in r.values())


def test_multimatch_too_short_is_nan():
    r = multimatch(A[:2], A[:2])
    assert all(math.isnan(v) for v in r.values())
