"""Golden-image tests for src/foveation.py (build-plan step-2 gate)."""
import os
import sys
import copy

import numpy as np
import pytest
import yaml
from PIL import Image
from scipy.ndimage import laplace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.foveation import render_foveated, _to_px, _radius_px  # noqa: E402

W, H = 1680, 1050


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))


@pytest.fixture(scope="module")
def img():
    rng = np.random.default_rng(0)  # high-frequency noise so blur is measurable
    return Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8), "RGB")


def hf(region):
    """High-frequency energy: mean squared Laplacian of the grayscale region."""
    g = region.astype(np.float64).mean(axis=2)
    return float((laplace(g) ** 2).mean())


def patch(im, cx, cy, half):
    a = np.asarray(im)
    return a[cy - half:cy + half, cx - half:cx + half]


def _mode(cfg, mode, **over):
    c = copy.deepcopy(cfg); c["foveation"]["mode"] = mode; c["foveation"].update(over)
    return c


def test_foveal_disk_is_pixel_identical(cfg, img):
    out = render_foveated(img, (0.5, 0.5), _mode(cfg, "periphery"))
    cx, cy = _to_px((0.5, 0.5), (W, H))
    half = 30  # well inside the foveal radius (~105 px) -> mask is 255 -> exact source
    assert np.array_equal(patch(out, cx, cy, half), patch(img, cx, cy, half))


def test_periphery_is_degraded(cfg, img):
    out = render_foveated(img, (0.5, 0.5), _mode(cfg, "periphery"))
    cx, cy = _to_px((0.5, 0.5), (W, H))
    center = hf(patch(out, cx, cy, 30))
    corner = hf(np.asarray(out)[0:100, 0:100])
    assert center > 5 * corner                      # fovea far sharper than periphery
    assert corner < 0.5 * hf(np.asarray(img)[0:100, 0:100])  # corner actually blurred


def test_gaussian_falloff_monotonic(cfg, img):
    c = copy.deepcopy(cfg); c["foveation"]["mode"] = "gaussian"
    out = np.asarray(render_foveated(img, (0.5, 0.5), c))
    cx, cy = _to_px((0.5, 0.5), (W, H))
    energies = [hf(out[cy - rad:cy - rad + 40, cx - 20:cx + 20])
                for rad in (20, 140, 260, 380, 500)]  # increasing eccentricity bands
    for a, b in zip(energies, energies[1:]):
        assert b <= a + 1.0                          # non-increasing (small tolerance)


def test_crop_hides_periphery(cfg, img):
    c = copy.deepcopy(cfg)
    c["foveation"]["mode"] = "crop"; c["foveation"]["crop_shape"] = "disk"
    fill = tuple(c["foveation"]["crop_fill_rgb"])
    out = np.asarray(render_foveated(img, (0.5, 0.5), c))
    assert tuple(out[0, 0]) == fill                  # far corner == gray fill exactly
    cx, cy = _to_px((0.5, 0.5), (W, H))
    assert np.array_equal(patch(out, cx, cy, 30), patch(img, cx, cy, 30))  # fovea == source


def test_ior_degrades_only_visited_region(cfg, img):
    c = copy.deepcopy(cfg)
    c["ior"]["enabled"] = True; c["ior"]["method"] = "blur"; c["ior"]["recent_k"] = 3
    visited = [(0.25, 0.25)]
    off = np.asarray(render_foveated(img, (0.5, 0.5), cfg, visited=visited))
    on = np.asarray(render_foveated(img, (0.5, 0.5), c, visited=visited))
    vx, vy = _to_px((0.25, 0.25), (W, H))
    assert hf(on[vy - 20:vy + 20, vx - 20:vx + 20]) < 0.7 * hf(off[vy - 20:vy + 20, vx - 20:vx + 20])
    mx, my = _to_px((0.75, 0.75), (W, H))            # mirror, not visited -> unchanged
    assert np.array_equal(on[my - 20:my + 20, mx - 20:mx + 20],
                          off[my - 20:my + 20, mx - 20:mx + 20])


def test_ior_off_by_default_ignores_visited(cfg, img):
    assert cfg["ior"]["enabled"] is False
    a = render_foveated(img, (0.5, 0.5), cfg, visited=[(0.2, 0.2), (0.8, 0.8)])
    b = render_foveated(img, (0.5, 0.5), cfg, visited=None)
    assert a.tobytes() == b.tobytes()


def test_deterministic_and_nonmutating(cfg, img):
    before = np.array(img)
    a = render_foveated(img, (0.37, 0.62), cfg)
    b = render_foveated(img, (0.37, 0.62), cfg)
    assert a.tobytes() == b.tobytes()                # deterministic
    assert np.array_equal(np.array(img), before)     # input not mutated


def test_border_fixations_safe(cfg, img):
    for fx, fy in [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]:
        out = render_foveated(img, (fx, fy), cfg)
        assert out.size == (W, H)


# --- Geisler-Perry (primary) golden tests ---
def test_gp_is_default_primary(cfg):
    assert cfg["foveation"]["mode"] == "geisler_perry"


def test_gp_monotonic_falloff(cfg, img):
    out = np.asarray(render_foveated(img, (0.5, 0.5), _mode(cfg, "geisler_perry")))
    cx, cy = _to_px((0.5, 0.5), (W, H))
    energies = [hf(out[cy - rad:cy - rad + 40, cx - 20:cx + 20]) for rad in (20, 140, 260, 380, 500)]
    for a, b in zip(energies, energies[1:]):
        assert b <= a + 1.0                       # HF non-increasing with eccentricity


def test_gp_fovea_sharper_than_periphery(cfg, img):
    out = render_foveated(img, (0.5, 0.5), _mode(cfg, "geisler_perry"))
    cx, cy = _to_px((0.5, 0.5), (W, H))
    center = hf(patch(out, cx, cy, 25))
    corner = hf(np.asarray(out)[0:80, 0:80])
    assert center > 1.5 * corner                  # foveation: fovea much sharper than periphery
    # canonical (log2) GP keeps the fovea at pyramid level 0 -> near-sharp:
    assert center > 0.8 * hf(patch(img, cx, cy, 25))


def test_gp_ppd_parameterized_and_deterministic(cfg, img):
    a = render_foveated(img, (0.5, 0.5), _mode(cfg, "geisler_perry", px_per_degree=42))
    b = render_foveated(img, (0.5, 0.5), _mode(cfg, "geisler_perry", px_per_degree=126))
    assert a.tobytes() != b.tobytes()             # px_per_degree changes the falloff
    a2 = render_foveated(img, (0.5, 0.5), _mode(cfg, "geisler_perry", px_per_degree=42))
    assert a.tobytes() == a2.tobytes()            # deterministic for fixed params


def test_sharp_mode_returns_input_unmodified(cfg, img):
    out = render_foveated(img, (0.3, 0.6), _mode(cfg, "sharp"))
    assert np.array_equal(np.asarray(out), np.asarray(img))   # no foveation -- unconstrained anchor


def test_gaussian_gist_more_blurred_than_gp(cfg, img):
    cx, cy = _to_px((0.5, 0.5), (W, H))
    gp = render_foveated(img, (0.5, 0.5), _mode(cfg, "geisler_perry"))
    gist = render_foveated(img, (0.5, 0.5), _mode(cfg, "gaussian", gist_k=8))
    # at a mid-periphery patch, the k=8 gist condition is blurrier (lower HF) than faithful GP:
    p = (cx + 300, cy - 200)
    assert hf(patch(gist, *p, 25)) < hf(patch(gp, *p, 25))
