"""A transparent region-supported ion localization baseline.

This module has no learned parameters and no dark-site inference. A segmented
region may contain multiple peaks. The reported ``snr`` is a heuristic local
contrast/noise score, NOT a calibrated detection probability or quantum fidelity.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from ..data import _check_image


DEFAULT_DETECTION = {
    "sigma": 1.0,
    "background_sigma": 10.0,
    "threshold_sigma": 4.0,
    "threshold_abs": 0.0,
    "min_area": 3,
    "min_distance": 4.0,
    "radius": 3.0,
    "max_ions": 5000,
    "saturation_value": None,
}


def _gaussian_noise_factor(sigma: float) -> Tuple[float, float]:
    """White-noise gains of Gaussian smoothing and its high-pass residual."""
    if sigma <= 0:
        return 1.0, 0.0
    radius = max(1, int(4.0 * sigma + 0.5))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    energy2d = float(np.sum(kernel * kernel) ** 2)
    center2d = float(kernel[radius] ** 2)
    return np.sqrt(energy2d), np.sqrt(max(1.0 - 2.0 * center2d + energy2d, 1e-12))


def estimate_detection_noise(image: np.ndarray, sigma: float = 1.0) -> float:
    """Raw-frame noise projected to the smoothed response (heuristic, ADU).

    Supply the acquisition image even when localizing a restored image. Neural
    smoothing makes the old white-noise estimate on its output too permissive.
    """
    raw = _check_image(image, "noise reference")
    if np.any(raw < 0) or not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Noise reference must be nonnegative and sigma finite/nonnegative")
    highpass = raw - ndi.gaussian_filter(raw, 1.0, mode="reflect")
    mad = float(np.median(np.abs(highpass - np.median(highpass))))
    _, highpass_factor = _gaussian_noise_factor(1.0)
    smoothing_factor, _ = _gaussian_noise_factor(float(sigma))
    return max(1.4826 * mad / highpass_factor * smoothing_factor,
               float(raw.max() - raw.min()) * 1e-6, 1e-12)


def detect_ions(image: np.ndarray, config: Optional[Dict] = None,
                region_mask: Optional[np.ndarray] = None,
                noise_reference: Optional[np.ndarray] = None) -> Tuple[List[Dict], np.ndarray]:
    """Return observed peak records and a bool foreground-support mask.

    All lengths are pixels and thresholds are in input-image intensity units.
    The centroid is calculated from nonnegative, background-subtracted ORIGINAL
    pixels near a peak; local pixels are assigned to their nearest surviving
    peak to reduce neighbor attraction. This is a baseline, not a joint PSF fit.
    region_mask limits both peak selection and final centers; no lattice sites
    are invented. noise_reference can anchor thresholds to raw acquisition
    evidence instead of the artificially low residual noise after restoration.
    """
    c = dict(DEFAULT_DETECTION)
    if config:
        unknown = set(config) - set(c)
        if unknown:
            raise ValueError("Unknown detection configuration keys: {}".format(sorted(unknown)))
        c.update(config)
    for key in ("sigma", "background_sigma", "threshold_sigma", "threshold_abs", "min_distance", "radius"):
        if not np.isfinite(c[key]):
            raise ValueError("Detection setting {} must be finite".format(key))
    if c["sigma"] < 0 or c["background_sigma"] <= c["sigma"]:
        raise ValueError("Require 0 <= sigma < background_sigma")
    if c["threshold_sigma"] <= 0 or c["threshold_abs"] < 0 or c["min_distance"] <= 0 or c["radius"] <= 0:
        raise ValueError("Require positive threshold_sigma/min_distance/radius and nonnegative threshold_abs")
    for key in ("min_area", "max_ions"):
        if not isinstance(c[key], (int, np.integer)) or c[key] < 1:
            raise ValueError("{} must be a positive integer".format(key))
    if c["saturation_value"] is not None and (not np.isfinite(c["saturation_value"]) or c["saturation_value"] <= 0):
        raise ValueError("saturation_value must be null or positive and finite")

    raw = _check_image(image, "detector input")
    if np.any(raw < 0):
        raise ValueError("Detector expects nonnegative raw-scale intensity; clip restoration undershoot before detection")
    h, w = raw.shape
    empty = np.zeros(raw.shape, dtype=bool)
    allowed = np.ones(raw.shape, dtype=bool)
    if region_mask is not None:
        region_mask = np.asarray(region_mask)
        if region_mask.shape != raw.shape or region_mask.dtype != np.bool_:
            raise ValueError("region_mask must be a boolean array matching the image")
        allowed = region_mask
    reference = raw if noise_reference is None else _check_image(noise_reference, "noise reference")
    if reference.shape != raw.shape:
        raise ValueError("Noise reference must have the same shape as the detector input")
    response_noise = estimate_detection_noise(reference, float(c["sigma"]))
    if not allowed.any():
        return [], empty
    dynamic_range = float(np.max(raw) - np.min(raw))
    if dynamic_range <= 0:
        return [], empty
    smooth = ndi.gaussian_filter(raw, float(c["sigma"]), mode="reflect") if c["sigma"] else raw.copy()
    background = ndi.gaussian_filter(raw, float(c["background_sigma"]), mode="reflect")
    response = smooth - background

    threshold = max(float(c["threshold_abs"]), float(c["threshold_sigma"]) * response_noise)

    components, count = ndi.label((response > threshold) & allowed, structure=np.ones((3, 3), dtype=np.uint8))
    if not count:
        return [], empty
    sizes = np.bincount(components.ravel())
    keep = sizes >= int(c["min_area"])
    keep[0] = False
    mask = keep[components]
    if not mask.any():
        return [], empty

    maxima = mask & (response == ndi.maximum_filter(response, size=3, mode="nearest"))
    plateaus, num_plateaus = ndi.label(maxima, structure=np.ones((3, 3), dtype=np.uint8))
    candidates = []
    for plateau_id, region in enumerate(ndi.find_objects(plateaus), start=1):
        if region is None:
            continue
        ys, xs = region
        local = plateaus[region] == plateau_id
        # Each connected maximum plateau is represented once, near its center.
        yy, xx = np.nonzero(local)
        if len(yy) == 0:
            continue
        score = response[ys.start + yy, xs.start + xx]
        best = np.flatnonzero(score == score.max())
        center_y, center_x = yy[best].mean(), xx[best].mean()
        near = best[np.argmin((yy[best] - center_y) ** 2 + (xx[best] - center_x) ** 2)]
        y, x = int(ys.start + yy[near]), int(xs.start + xx[near])
        candidates.append((y, x, float(response[y, x])))
    if not candidates:
        return [], mask
    candidates.sort(key=lambda p: (-p[2], p[0], p[1]))
    candidate_yx = np.asarray([(p[0], p[1]) for p in candidates], dtype=np.float64)
    tree = cKDTree(candidate_yx)
    suppressed = np.zeros(len(candidates), dtype=bool)
    selected = []
    for index, point in enumerate(candidates):
        if suppressed[index]:
            continue
        selected.append(point)
        suppressed[tree.query_ball_point(candidate_yx[index], float(c["min_distance"]))] = True
        if len(selected) >= c["max_ions"]:
            break
    selected_tree = cKDTree(np.asarray([(p[0], p[1]) for p in selected], dtype=np.float64))
    ions = []
    radius = float(c["radius"])
    half_width = int(np.ceil(radius))
    for index, (py, px, score) in enumerate(selected):
        y0, y1 = max(0, py - half_width), min(h, py + half_width + 1)
        x0, x1 = max(0, px - half_width), min(w, px + half_width + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        _, nearest = selected_tree.query(np.column_stack([yy.ravel(), xx.ravel()]))
        support = ((yy - py) ** 2 + (xx - px) ** 2 <= radius ** 2)
        support &= nearest.reshape(yy.shape) == index
        support &= mask[y0:y1, x0:x1]
        weights = np.maximum(raw[y0:y1, x0:x1] - background[y0:y1, x0:x1], 0.0) * support
        flux = float(weights.sum(dtype=np.float64))
        if flux > 0:
            x = float(np.sum(weights * xx, dtype=np.float64) / flux)
            y = float(np.sum(weights * yy, dtype=np.float64) / flux)
        else:
            x, y = float(px), float(py)
        # A weighted centroid can fall into a concave mask's excluded pixels.
        # Fall back to its known in-region peak; never export an outside center.
        ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
        if not (0 <= ix < w and 0 <= iy < h and allowed[iy, ix]):
            x, y = float(px), float(py)
        saturation = c["saturation_value"]
        saturated = bool(saturation is not None and np.any(raw[y0:y1, x0:x1][support] >= saturation))
        ions.append({
            "id": 0, "x": x, "y": y,
            "peak": float(raw[py, px]), "flux": flux,
            "radius": radius, "snr": float(score / response_noise),
            "status": "observed", "saturated": saturated,
            "edge": bool(px - radius < 0 or py - radius < 0 or px + radius >= w or py + radius >= h),
        })
    ions.sort(key=lambda ion: (ion["y"], ion["x"]))
    for index, ion in enumerate(ions, start=1):
        ion["id"] = index
    return ions, mask


def save_overlay(raw: np.ndarray, ions: List[Dict], path: Union[str, Path],
                 mask: Optional[np.ndarray] = None) -> None:
    """Save an 8-bit RGB visualization; never use it as quantitative image data."""
    raw = _check_image(raw, "overlay input")
    lo, hi = np.percentile(raw, [0.5, 99.9])
    if hi <= lo:
        hi = float(raw.max())
        lo = float(raw.min())
    if hi <= lo:
        gray = np.zeros(raw.shape, dtype=np.uint8)
    else:
        gray = np.rint(np.clip((raw - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != raw.shape:
            raise ValueError("Overlay mask shape must match image")
        boundary = mask & ~ndi.binary_erosion(mask)
        rgb[boundary] = (40, 120, 230)
    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)
    for ion in ions:
        x, y, r = float(ion["x"]), float(ion["y"]), float(ion["radius"])
        color = (255, 140, 0) if ion.get("saturated", False) else (70, 255, 100)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=1)
        draw.point((round(x), round(y)), fill=(255, 70, 70))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(str(path), format="PNG")
