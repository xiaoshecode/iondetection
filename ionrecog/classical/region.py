"""Conservative crystal-envelope segmentation from ORIGINAL camera pixels.

The optional search rectangle is an optical prior, not the accepted region.
Inside it, every frame must supply enough raw signal for an adaptive envelope.
No region means no detections; restoration output is never used to invent one.
This image-processing prior does not establish quantum state or occupancy.
"""
import numbers

import numpy as np
from scipy import ndimage as ndi

from ..data import _check_image


DEFAULT_REGION = {
    "mode": "off",
    "search_roi": None,
    "expected_shape": None,
    "envelope_sigma": 2.5,
    "background_sigma": 20.0,
    "support_sigma": 3.0,
    "seed_sigma": 6.0,
    "support_fraction": 0.06,
    "min_contrast": 80.0,
    "min_area": 40,
    "min_seed_area": 12,
    "closing_radius": 3,
    "padding": 3,
    "reject_border": True,
    "max_extent_fraction": 0.85,
}


def _integer_list(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError("region.%s must contain %d integers" % (name, length))
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Integral) for v in value):
        raise ValueError("region.%s must contain integers" % name)
    return [int(v) for v in value]


def _validated_config(config, shape):
    options = dict(DEFAULT_REGION)
    if config is not None:
        if not isinstance(config, dict):
            raise ValueError("region must be a configuration mapping")
        unknown = set(config) - set(options)
        if unknown:
            raise ValueError("Unknown region settings: %s" % sorted(unknown))
        options.update(config)
    if options["mode"] not in ("off", "auto"):
        raise ValueError("region.mode must be off or auto")
    if options["expected_shape"] is not None:
        expected = _integer_list(options["expected_shape"], 2, "expected_shape")
        if min(expected) <= 0 or tuple(expected) != tuple(shape):
            raise ValueError("Image shape %s does not match region.expected_shape %s; use the matching camera configuration" % (shape, expected))
    height, width = shape
    roi = [0, 0, width, height] if options["search_roi"] is None else _integer_list(options["search_roi"], 4, "search_roi")
    x0, y0, x1, y1 = roi
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError("region.search_roi must be an in-bounds [x0,y0,x1,y1] rectangle with exclusive right/bottom edges")
    options["search_roi"] = roi
    for name in ("envelope_sigma", "background_sigma", "support_sigma", "seed_sigma", "support_fraction", "min_contrast", "max_extent_fraction"):
        value = options[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not np.isfinite(value):
            raise ValueError("region.%s must be finite and numeric" % name)
    if not 0 < options["envelope_sigma"] < options["background_sigma"]:
        raise ValueError("Require 0 < region.envelope_sigma < region.background_sigma")
    if not 0 < options["support_sigma"] <= options["seed_sigma"]:
        raise ValueError("Require 0 < region.support_sigma <= region.seed_sigma")
    if not 0 <= options["support_fraction"] < 1 or options["min_contrast"] < 0:
        raise ValueError("Require 0 <= region.support_fraction < 1 and nonnegative region.min_contrast")
    if not 0 < options["max_extent_fraction"] <= 1:
        raise ValueError("region.max_extent_fraction must be in (0,1]")
    for name in ("min_area", "min_seed_area", "closing_radius", "padding"):
        value = options[name]
        minimum = 1 if name in ("min_area", "min_seed_area") else 0
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
            raise ValueError("region.%s must be an integer >= %d" % (name, minimum))
    if not isinstance(options["reject_border"], (bool, np.bool_)):
        raise ValueError("region.reject_border must be boolean")
    return options


def _noise_gains(sigma):
    radius = max(1, int(4 * sigma + 0.5))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coordinates / sigma) ** 2)
    kernel /= kernel.sum()
    energy = float(np.sum(kernel * kernel) ** 2)
    return np.sqrt(energy), np.sqrt(max(1 - 2 * kernel[radius] ** 2 + energy, 1e-12))


def _disk(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def segment_crystal(raw, config=None):
    """Return an adaptive bool mask and JSON-safe diagnostics.

    In ``auto`` mode a low envelope threshold provides support, while an
    independent high threshold requires enough raw bright pixels. A coarse
    search ROI excludes known optical artifacts but is never returned as a
    fallback mask. The strongest qualified connected component is selected;
    this assumes one visible crystal in the search field. Large boundary
    artifacts are rejected conservatively. A small dilation preserves outer
    ions; it does not force points into the region or complete a lattice.
    """
    raw = _check_image(raw, "crystal region input")
    if np.any(raw < 0):
        raise ValueError("Crystal segmentation requires nonnegative raw camera pixels")
    options = _validated_config(config, raw.shape)
    height, width = raw.shape
    x0, y0, x1, y1 = options["search_roi"]
    info = {"mode": options["mode"], "status": "off" if options["mode"] == "off" else "no_crystal",
            "source": "original_raw_image", "shape": [height, width],
            "search_roi": [x0, y0, x1, y1], "area": 0, "bbox": None,
            "candidate_components": 0, "qualified_components": 0,
            "note": "Image-based crystal envelope, not an independently verified ground-truth segmentation."}
    if options["mode"] == "off":
        info.update({"area": int(raw.size), "bbox": [0, 0, width, height]})
        return np.ones(raw.shape, dtype=bool), info
    empty = np.zeros(raw.shape, dtype=bool)
    roi = np.zeros(raw.shape, dtype=bool)
    roi[y0:y1, x0:x1] = True
    if np.ptp(raw[y0:y1, x0:x1]) == 0:
        info["reason"] = "constant_search_region"
        return empty, info

    # Full-frame filtering avoids sharp artificial edges at the search rectangle.
    envelope = ndi.gaussian_filter(raw, float(options["envelope_sigma"]), mode="reflect")
    envelope -= ndi.gaussian_filter(raw, float(options["background_sigma"]), mode="reflect")
    highpass = raw - ndi.gaussian_filter(raw, 1.0, mode="reflect")
    roi_residual = highpass[roi]
    mad = float(np.median(np.abs(roi_residual - np.median(roi_residual))))
    _, highpass_gain = _noise_gains(1.0)
    smooth_gain, _ = _noise_gains(float(options["envelope_sigma"]))
    envelope_noise = max(1.4826 * mad / highpass_gain * smooth_gain, float(np.ptp(raw)) * 1e-7, 1e-12)
    robust_peak = max(0.0, float(np.percentile(envelope[roi], 99.5)))
    support_threshold = max(options["support_sigma"] * envelope_noise, options["support_fraction"] * robust_peak)
    seed_threshold = max(options["seed_sigma"] * envelope_noise, options["min_contrast"], support_threshold * 1.5)
    info.update({"estimated_envelope_noise_adu": float(envelope_noise),
                 "support_threshold_adu": float(support_threshold), "seed_threshold_adu": float(seed_threshold)})
    seeds = (envelope > seed_threshold) & roi
    if np.count_nonzero(seeds) < options["min_seed_area"]:
        info["reason"] = "insufficient_raw_signal"
        return empty, info
    support = (envelope > support_threshold) & roi
    radius = int(options["closing_radius"])
    if radius:
        support = ndi.binary_closing(support, structure=_disk(radius)) & roi
    support = ndi.binary_fill_holes(support) & roi
    labels, count = ndi.label(support, structure=np.ones((3, 3), dtype=bool))
    info["candidate_components"] = int(count)
    candidates = []
    # Closing can shrink components away from the physical edge by radius pixels;
    # include that margin when rejecting boundary-connected optical artifacts.
    border_margin = radius + 1
    for label, slices in enumerate(ndi.find_objects(labels), start=1):
        if slices is None:
            continue
        ys, xs = slices
        component = labels[slices] == label
        area = int(np.count_nonzero(component))
        seed_count = int(np.count_nonzero(seeds[slices] & component))
        if area < options["min_area"] or seed_count < options["min_seed_area"]:
            continue
        if (xs.stop - xs.start) > options["max_extent_fraction"] * (x1 - x0) or (ys.stop - ys.start) > options["max_extent_fraction"] * (y1 - y0):
            continue
        if options["reject_border"] and (xs.start < border_margin or ys.start < border_margin or xs.stop > width - border_margin or ys.stop > height - border_margin):
            continue
        score = float(np.maximum(envelope[slices][component], 0).sum(dtype=np.float64))
        candidates.append((score, label, area, seed_count))
    info["qualified_components"] = len(candidates)
    if not candidates:
        info["reason"] = "no_component_meets_signal_and_geometry_requirements"
        return empty, info
    _, selected, core_area, seed_count = max(candidates)
    result = labels == selected
    padding = int(options["padding"])
    if padding:
        result = ndi.binary_dilation(result, structure=_disk(padding))
    result &= roi
    y, x = np.nonzero(result)
    info.update({"status": "detected", "area": int(result.sum()), "core_area": core_area,
                 "seed_pixels": seed_count, "bbox": [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1],
                 "selection": "highest_integrated_positive_raw_envelope",
                 "touches_search_boundary": bool(result[y0:y1, x0].any() or result[y0:y1, x1 - 1].any() or result[y0, x0:x1].any() or result[y1 - 1, x0:x1].any())})
    return result, info
