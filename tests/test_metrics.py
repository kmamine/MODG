"""Unit tests for src/metrics.py on tiny fixtures."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.metrics import (agent_fix_px, human_fix_px, numfix, tfp,  # noqa: E402
                         density_map, nss, cc, kl, stopping)


def test_numfix():
    assert numfix({"fixations": [{"x": .5, "y": .5, "step": 0}, {"x": .1, "y": .2, "step": 1}]}) == 2
    assert numfix({"X": [1, 2, 3], "Y": [1, 2, 3], "length": 3}) == 3


def test_coordinate_accessors():
    assert np.allclose(agent_fix_px({"fixations": [{"x": 0.5, "y": 0.5, "step": 0}]}), [[840, 525]])
    assert np.allclose(human_fix_px({"X": [840.0], "Y": [525.0]}), [[840, 525]])


def test_tfp_hits_at_known_step():
    bbox = [800, 500, 80, 60]
    sp = np.array([[840, 300], [100, 100], [820, 520]], float)  # inside bbox first at index 2
    t = tfp([sp], bbox, tol=0, n_max=3)
    assert t[0] == 0 and t[1] == 0 and t[2] == 1.0 and t[3] == 1.0


def test_tfp_self_reaches_one():
    sp = np.array([[840, 525], [820, 520]], float)
    assert tfp([sp], [800, 500, 80, 60], tol=0, n_max=2)[-1] == 1.0


def test_density_self_agreement():
    pts = [np.array([[840, 525], [400, 300], [1200, 700]], float)]
    m = density_map(pts, 30.0)
    assert cc(m, m) > 0.999          # self-correlation = 1
    assert kl(m, m) < 1e-6           # self-KL = 0
    assert nss(m, pts[0]) > 0        # density high where the points are


def test_stopping_ta_breakdown():
    recs = [{"stop_reason": "not_present", "fixations": [0, 1, 2]},
            {"stop_reason": "max_fixations", "fixations": list(range(50))},
            {"stop_reason": "found", "fixations": [0, 1]}]
    s = stopping(recs, "TA")
    assert abs(s["frac_declared_absent"] - 1 / 3) < 1e-9
    assert abs(s["frac_hit_cap"] - 1 / 3) < 1e-9
    assert abs(s["frac_false_present"] - 1 / 3) < 1e-9


from src.metrics import (numfix2t, tfp_auc, saccade_amplitudes, turning_angles,  # noqa: E402
                         scanpath_length, hull_area, gaze_entropy, center_bias, refixation_rate)


def test_numfix2t_and_auc():
    sp = np.array([[840, 300], [100, 100], [820, 520]], float)  # in bbox at idx 2
    assert numfix2t(sp, [800, 500, 80, 60], tol=0) == 2.0
    assert np.isnan(numfix2t(np.array([[0, 0]], float), [800, 500, 80, 60], 0))
    assert abs(tfp_auc([0, 0, 1, 1]) - 0.5) < 1e-9


def test_intrinsic_signature():
    # 3 fixations forming a right angle at the middle point
    sp = np.array([[300, 300], [300 + 30, 300], [300 + 30, 300 + 30]], float)  # 1 dva saccades at ppd=30
    amps = saccade_amplitudes(sp, ppd=30.0)
    assert np.allclose(amps, [1.0, 1.0])
    assert abs(scanpath_length(sp, 30.0) - 2.0) < 1e-9
    assert abs(abs(turning_angles(sp)[0]) - 90.0) < 1e-6     # 90-degree turn
    # gaze entropy: all in one cell -> 0; spread -> > 0
    assert gaze_entropy(np.array([[10, 10], [12, 12]], float)) == 0.0
    assert gaze_entropy(np.array([[10, 10], [1600, 1000]], float)) > 0
    # refixation: revisit the same cell -> rate > 0
    assert refixation_rate(np.array([[10, 10], [1600, 1000], [11, 11]], float)) > 0
    assert hull_area(np.array([[0, 0], [300, 0], [0, 300]], float), 30.0) > 0  # a triangle
    assert center_bias(np.array([[840, 525]], float), 30.0) < 0.1               # at center


