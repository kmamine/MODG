"""Foveated-image renderer (build-plan step 2).

Pure, deterministic function of (image, fixation, cfg, visited): identical inputs
produce identical output pixels, and the same process renders for every model, so
the stimulus is byte-identical across models. The harness is the only caller.

Conditions in the constraint bracket (cfg['foveation']['mode']):
  sharp         : no foveation (image returned unmodified) -- unconstrained anchor.
  geisler_perry : faithful Geisler & Perry (1998) acuity falloff -- mild human-matched
                  reference (the "input != constraint" bound).
  gaussian      : GP-shaped, peripheral acuity reduced by factor gist_k (>1) -- the
                  "usable gist" synthetic manipulation.
  crop          : fovea-only disk/box, rest gray -- no peripheral gist, over-constrained bound.
  periphery     : two-level sharp-disk + single blur (legacy control, out of the pilot bracket).
Only `geisler_perry` is human-matched; the others are synthetic manipulations/anchors.
Inhibition-of-Return visual mask (cfg['ior']['enabled'], default off, a manipulation
only): suppress (blur/dim) the most-recent-K visited disks. Never on in the primary run.

Reuses the crop/clamp arithmetic of src_legacy_project_for_ref/data/patch_extraction.py
and the copy-then-draw pattern of data/image_marking.py (lifted, not imported).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter, ImageChops


def _to_px(fixation_xy, size):
    """Normalized (x,y) in [0,1] -> integer pixel, clamped inside the frame."""
    w, h = size
    x, y = fixation_xy
    cx = min(max(int(round(x * (w - 1))), 0), w - 1)
    cy = min(max(int(round(y * (h - 1))), 0), h - 1)
    return cx, cy


def _radial_mask(size, cx, cy, r_in, r_out):
    """'L' mask: 255 inside r_in, 0 beyond r_out, linear ramp in the annulus."""
    w, h = size
    yy, xx = np.ogrid[0:h, 0:w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = np.clip((r_out - d) / max(r_out - r_in, 1), 0.0, 1.0)
    return Image.fromarray((alpha * 255 + 0.5).astype(np.uint8), mode="L")


def _radius_px(fcfg):
    """Foveal radius in px, anchored in degrees so it scales with px_per_degree."""
    return int(round(fcfg["px_per_degree"] * fcfg["foveal_radius_deg"]))


def _clamped_box(cx, cy, r, size):
    """Square box of half-width r centered at (cx,cy), clamped to the frame.

    Mirrors extract_patch (patch_extraction.py:33-41)."""
    w, h = size
    left, upper = max(0, cx - r), max(0, cy - r)
    right, lower = min(w, cx + r), min(h, cy + r)
    return left, upper, right, lower


def _render_periphery(img, cx, cy, r, fcfg):
    edge = fcfg["edge_softness_px"]
    blurred = img.filter(ImageFilter.GaussianBlur(radius=fcfg["surround_blur_px"]))
    mask = _radial_mask(img.size, cx, cy, r, r + edge)
    return Image.composite(img, blurred, mask)  # mask=255 -> sharp, 0 -> blurred


def _render_geisler_perry(img, cx, cy, fcfg, acuity_scale=1.0):
    """Geisler & Perry (1998) foveation -- continuous, eccentricity-scaled acuity falloff.

    Perspective eccentricity via atan over a flat screen at `view_dist_m`; eye cutoff
    fc(ec) = e2*ln(1/CT0)/(alpha*(ec+e2)) [cyc/deg]; per-pixel mip level
    `L = log2(acuity_scale * local_Nyquist / fc)` -- the canonical (log2) Geisler & Perry form,
    which renders the local image cutoff equal to the eye's cutoff (faithful, mild; the centre is
    sharp because the ratio < 1 -> log2 < 0 -> clips to level 0). Blended over an octave pyramid
    (level k matches a 2^k mip: Gaussian sigma = 0.3748*2^k px, cutoff 0.5/2^k cyc/px).
    `acuity_scale = k > 1` divides the eye cutoff by k -- the "usable gist" gaussian condition
    (periphery at 1/k of human acuity); k = 1 is the faithful GP reference. Deterministic;
    parameterized by px_per_degree (pix2deg) + view_dist_m."""
    a = np.asarray(img, dtype=np.float32)
    H, W = a.shape[:2]
    e2, alpha, ct0 = float(fcfg["gp_e2"]), float(fcfg["gp_alpha"]), float(fcfg["gp_ct0"])
    pix2deg, vd = float(fcfg["px_per_degree"]), float(fcfg["view_dist_m"])
    dp = (2.0 * vd * math.tan(math.radians(W / pix2deg) / 2.0)) / W   # dotPitch (m/px), per the ref
    nlev = 1 + int(math.floor(math.log2(max(W, H))))                  # full octave chain
    K = math.log(1.0 / ct0)
    yy, xx = np.ogrid[0:H, 0:W]
    eradius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) * dp           # metres on the screen
    ec = np.degrees(np.arctan(eradius / vd))                         # eccentricity (deg), perspective
    eyefreq = e2 / (alpha * (ec + e2)) * K                           # eye cutoff (cyc/deg)
    maxfreq = math.pi / ((np.arctan((eradius + dp) / vd) - np.arctan((eradius - dp) / vd)) * 180.0)
    L = np.clip(np.log2(acuity_scale * maxfreq / eyefreq), 0.0, float(nlev))  # canonical GP (log2)
    out = np.zeros_like(a)
    for k in range(int(math.ceil(float(L.max()))) + 1):
        w = np.clip(1.0 - np.abs(L - k), 0.0, 1.0)                   # tent: floor/ceil only, sums to 1
        if not w.any():
            continue
        layer = a if k == 0 else np.asarray(
            img.filter(ImageFilter.GaussianBlur(radius=0.3748 * (2.0 ** k))), dtype=np.float32)
        out += layer * w[..., None]
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGB")


def _render_crop(img, cx, cy, r, fcfg):
    gray = Image.new("RGB", img.size, tuple(fcfg["crop_fill_rgb"]))
    if fcfg.get("crop_shape", "disk") == "box":
        out = gray.copy()
        box = _clamped_box(cx, cy, r, img.size)
        out.paste(img.crop(box), (box[0], box[1]))
        return out
    mask = _radial_mask(img.size, cx, cy, r, r + fcfg["edge_softness_px"])
    return Image.composite(img, gray, mask)


def _apply_ior(out, visited_px, iorcfg):
    edge = iorcfg.get("edge_softness_px", 16)
    rad = iorcfg["radius_px"]
    recent = visited_px[-int(iorcfg["recent_k"]):]
    mask = Image.new("L", out.size, 0)
    for vx, vy in recent:
        mask = ImageChops.lighter(mask, _radial_mask(out.size, vx, vy, rad, rad + edge))
    if iorcfg.get("method", "blur") == "dim":
        overlay = Image.new("RGB", out.size, tuple(iorcfg["dim_rgb"]))
        a = int(iorcfg["dim_alpha"])
        mask = mask.point(lambda v: v * a // 255)
        return Image.composite(overlay, out, mask)
    blurred = out.filter(ImageFilter.GaussianBlur(radius=iorcfg["blur_px"]))
    return Image.composite(blurred, out, mask)


def render_foveated(image, fixation_xy, cfg, visited=None):
    """Return a foveated RGB copy of `image` for `fixation_xy` (normalized).

    Selects mode + IoR from cfg['foveation'] / cfg['ior']. Never mutates `image`."""
    img = image.convert("RGB") if image.mode != "RGB" else image.copy()
    fcfg = cfg["foveation"]
    iorcfg = cfg.get("ior", {}) or {}
    cx, cy = _to_px(fixation_xy, img.size)
    r = _radius_px(fcfg)
    mode = fcfg.get("mode", "geisler_perry")
    if mode == "sharp":
        out = img                                    # no foveation -- unconstrained anchor
    elif mode == "geisler_perry":
        out = _render_geisler_perry(img, cx, cy, fcfg)
    elif mode == "gaussian":
        out = _render_geisler_perry(img, cx, cy, fcfg, acuity_scale=float(fcfg["gist_k"]))
    elif mode == "crop":
        out = _render_crop(img, cx, cy, r, fcfg)
    elif mode == "periphery":
        out = _render_periphery(img, cx, cy, r, fcfg)
    else:
        raise ValueError(f"unknown foveation mode: {mode!r}")
    if iorcfg.get("enabled") and visited:
        out = _apply_ior(out, [_to_px(v, img.size) for v in visited], iorcfg)
    return out
