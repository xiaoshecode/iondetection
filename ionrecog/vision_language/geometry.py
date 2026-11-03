"""Auditable image geometry and raw-signal checks for VLM point proposals.

The VLM receives a display rendering, never calibrated photon measurements.
Coordinates here are zero-based original-image pixel centers. A supported point
is still a candidate, not proof of an ion or a calibrated quantum-state label.
"""
import copy
import json
import re
from numbers import Integral, Real

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from ..data import _check_image
from ..classical.reference import measure_sites


COORDINATE_CONVENTIONS = {
    "normalized_1000": "u,v in [0,1000]; x=x0+u*(width-1)/1000, y=y0+v*(height-1)/1000; endpoints are first/last raw pixel centers",
    "pixels": "u,v are zero-based DISPLAY pixel centers; local x=(u+0.5)/display_scale-0.5, local y=(v+0.5)/display_scale-0.5; outer subpixels are clamped to first/last raw centers and marked uncertain",
}


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return int(value)


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError("{} must be a finite number, not a string or boolean".format(name))
    return float(value)


def _tile_bounds(tile):
    if not isinstance(tile, dict):
        raise ValueError("Tile must be a mapping")
    try:
        x0, y0, x1, y1 = [_integer(tile[k], "tile." + k) for k in ("x0", "y0", "x1", "y1")]
    except KeyError as exc:
        raise ValueError("Tile requires x0,y0,x1,y1") from exc
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Tile must be a nonempty half-open rectangle")
    return x0, y0, x1, y1


def make_tiles(shape, tile_size=128, overlap=32, roi=None):
    """Tile a full image or explicit half-open ROI, covering its far edges.

    Last windows are shifted back to avoid tiny trailing tiles. ROI selection is
    an experimental input and must be shared by comparison methods.
    """
    if len(shape) != 2:
        raise ValueError("shape must contain image height and width")
    h, w = [_integer(v, "shape dimension", 1) for v in shape]
    tile_size = _integer(tile_size, "tile_size", 1)
    overlap = _integer(overlap, "overlap")
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size")
    if roi is None:
        roi = [0, 0, w, h]
    if not isinstance(roi, (list, tuple, np.ndarray)) or len(roi) != 4:
        raise ValueError("roi must be [x0,y0,x1,y1]")
    x0, y0, x1, y1 = [_integer(v, "ROI coordinate") for v in roi]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("ROI must be nonempty and inside the image")

    def starts(lo, hi):
        last = max(lo, hi - tile_size)
        values = list(range(lo, last + 1, tile_size - overlap))
        if values[-1] != last:
            values.append(last)
        return values

    tiles = []
    for top in starts(y0, y1):
        for left in starts(x0, x1):
            tiles.append({"id": len(tiles), "x0": left, "y0": top,
                          "x1": min(left + tile_size, x1), "y1": min(top + tile_size, y1)})
    return tiles


def render_tile(raw, tile, lo, hi, display_scale=3):
    """Render RGB without editing raw data; all tiles use supplied lo/hi.

    Nearest-neighbor integer enlargement introduces no synthetic small features.
    It repeats raw pixels, it does not increase physical resolving power.
    """
    raw = _check_image(raw, "VLM rendering input")
    x0, y0, x1, y1 = _tile_bounds(tile)
    if x1 > raw.shape[1] or y1 > raw.shape[0]:
        raise ValueError("Tile lies outside the raw image")
    lo, hi = _number(lo, "lo"), _number(hi, "hi")
    if hi <= lo:
        raise ValueError("Display range requires hi > lo")
    scale = _integer(display_scale, "display_scale", 1)
    gray = np.rint(255 * np.clip((raw[y0:y1, x0:x1] - lo) / (hi - lo), 0, 1)).astype(np.uint8)
    return Image.fromarray(gray).convert("RGB").resize(
        ((x1 - x0) * scale, (y1 - y0) * scale), resample=Image.Resampling.NEAREST)


def _reject_constant(value):
    raise ValueError("Non-finite JSON constant: " + value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def parse_points(text, tile, coordinate_system="normalized_1000", display_scale=3):
    """Strictly parse JSON points; never extract guessed numbers from prose.

    Accepted envelope: {"points": [[u,v], ...], "uncertain": false}.
    A bare list is also accepted; entries may be [u,v] or {"x":u,"y":v}.
    Per-point boolean ``uncertain`` is supported. Malformed output rejects the
    entire response, so partial responses cannot masquerade as complete counts.
    """
    x0, y0, x1, y1 = _tile_bounds(tile)
    scale = _integer(display_scale, "display_scale", 1)
    if coordinate_system not in COORDINATE_CONVENTIONS:
        raise ValueError("Unknown coordinate system: " + str(coordinate_system))
    if not isinstance(text, str):
        raise ValueError("VLM response must be JSON text")
    body = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", body, flags=re.S | re.I)
    if fenced:
        body = fenced.group(1).strip()
    try:
        parsed = json.loads(body, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("VLM response is not complete valid JSON") from exc
    whole_uncertain = False
    if isinstance(parsed, dict):
        if "points" not in parsed:
            raise ValueError("VLM JSON object requires points")
        whole_uncertain = parsed.get("uncertain", False)
        if not isinstance(whole_uncertain, bool):
            raise ValueError("uncertain must be a JSON boolean")
        parsed = parsed["points"]
    if not isinstance(parsed, list):
        raise ValueError("points must be a JSON list")
    result = []
    for index, item in enumerate(parsed):
        local_uncertain = False
        if isinstance(item, dict):
            if "x" not in item or "y" not in item:
                raise ValueError("Each point mapping requires x and y")
            u, v = item["x"], item["y"]
            local_uncertain = item.get("uncertain", False)
            if not isinstance(local_uncertain, bool):
                raise ValueError("Point uncertain must be a JSON boolean")
        elif isinstance(item, list) and len(item) == 2:
            u, v = item
        else:
            raise ValueError("Each point must be [u,v] or an x/y mapping")
        u, v = _number(u, "point x"), _number(v, "point y")
        width, height = x1 - x0, y1 - y0
        clamped = False
        if coordinate_system == "normalized_1000":
            if not (0 <= u <= 1000 and 0 <= v <= 1000):
                raise ValueError("Normalized coordinates must be in [0,1000]")
            x, y = x0 + u * (width - 1) / 1000, y0 + v * (height - 1) / 1000
        else:
            if not (0 <= u <= width * scale - 1 and 0 <= v <= height * scale - 1):
                raise ValueError("Pixel coordinates must be inside the displayed tile")
            lx, ly = (u + 0.5) / scale - 0.5, (v + 0.5) / scale - 0.5
            clamped = not (0 <= lx <= width - 1 and 0 <= ly <= height - 1)
            x, y = x0 + float(np.clip(lx, 0, width - 1)), y0 + float(np.clip(ly, 0, height - 1))
        uncertain = whole_uncertain or local_uncertain or clamped
        result.append({"x": float(x), "y": float(y), "status": "uncertain" if uncertain else "proposed",
                       "model_uncertain": bool(whole_uncertain or local_uncertain),
                       "coordinate_edge_clamped": bool(clamped), "tile_id": tile.get("id"),
                       "tile_point_index": index, "input_coordinates": [u, v],
                       "coordinate_system": coordinate_system,
                       "tile_margin": float(min(x - x0, y - y0, x1 - 1 - x, y1 - 1 - y))})
    return result


def _point_xy(point):
    if not isinstance(point, dict) or "x" not in point or "y" not in point:
        raise ValueError("Point requires numeric x and y")
    return _number(point["x"], "point x"), _number(point["y"], "point y")


def deduplicate(points, min_distance=2.0):
    """Greedy spatial deduplication with every merged proposal retained.

    Prefer non-uncertain predictions, then larger crop margins, then original
    order. Do not average coordinates or use model confidence as probability.
    Every output has merged_proposals and source_tile_ids for the audit trail.
    """
    distance = _number(min_distance, "min_distance")
    if distance <= 0:
        raise ValueError("min_distance must be positive")
    proposals = [copy.deepcopy(p) for p in points]
    coordinates = [_point_xy(p) for p in proposals]
    order = sorted(range(len(proposals)), key=lambda i: (
        proposals[i].get("status") == "uncertain", -_number(proposals[i].get("tile_margin", 0), "tile_margin"), i))
    kept = []
    for index in order:
        point, (x, y) = proposals[index], coordinates[index]
        nearby = [(np.hypot(x - q["x"], y - q["y"]), i) for i, q in enumerate(kept)
                  if np.hypot(x - q["x"], y - q["y"]) < distance]
        if nearby:
            representative = kept[min(nearby)[1]]
            representative["merged_proposals"].append(point)
        else:
            representative = copy.deepcopy(point)
            representative["merged_proposals"] = [point]
            representative["source_tile_ids"] = []
            kept.append(representative)
        tile_id = point.get("tile_id")
        if tile_id is not None and tile_id not in representative["source_tile_ids"]:
            representative["source_tile_ids"].append(tile_id)
        representative["proposal_count"] = len(representative["merged_proposals"])
    return kept


def refine_points(raw, points, radius=3, max_shift=2, min_snr=5):
    """Refine proposed centers with exclusive raw-pixel support, never add points.

    This is a local background-subtracted centroid, NOT PSF or micromotion
    deconvolution. Nearest-proposal ownership limits neighbor attraction. The
    support SNR is a heuristic using a local annular MAD with a 1 ADU floor;
    neither it nor repeated VLM agreement is a calibrated probability.

    Return (supported_candidates, uncertain_candidates). Failed checks retain
    the original position plus attempted refined coordinates and explicit
    reasons; large shifts are NOT silently clipped to the allowed radius.
    """
    raw = _check_image(raw, "VLM refinement input")
    radius, max_shift, min_snr = (_number(v, n) for v, n in
                                  ((radius, "radius"), (max_shift, "max_shift"), (min_snr, "min_snr")))
    if radius < 1 or max_shift < 0 or min_snr <= 0:
        raise ValueError("Require radius >= 1, max_shift >= 0 and min_snr > 0")
    points = [copy.deepcopy(p) for p in points]
    h, w = raw.shape
    xy = [_point_xy(p) for p in points]
    if any(not (0 <= x <= w - 1 and 0 <= y <= h - 1) for x, y in xy):
        raise ValueError("Proposal outside raw image pixel-center bounds")
    if not points:
        return [], []
    tree = cKDTree(xy)
    rows = []
    reach = int(np.ceil(min(3 * radius, max(h, w))))
    for index, (point, (px, py)) in enumerate(zip(points, xy)):
        reasons = []
        if point.get("status") == "uncertain" or point.get("model_uncertain", False):
            reasons.append("model_or_coordinate_uncertain")
        if px - radius < 0 or py - radius < 0 or px + radius > w - 1 or py + radius > h - 1:
            reasons.append("edge_clipped")
        x0, x1 = max(0, int(px) - reach), min(w, int(px) + reach + 2)
        y0, y1 = max(0, int(py) - reach), min(h, int(py) + reach + 2)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        nearest_dist, nearest = tree.query(np.column_stack([xx.ravel(), yy.ravel()]))
        nearest, nearest_dist = nearest.reshape(xx.shape), nearest_dist.reshape(xx.shape)
        distances = np.hypot(xx - px, yy - py)
        # Text coordinates can round an integer center to 15.999999999998.
        # A numerical tolerance avoids including a circle-boundary pixel on
        # only one side and introducing a spurious centroid shift.
        boundary_epsilon = 1e-9
        support = (distances <= radius + boundary_epsilon) & (nearest == index)
        annulus = ((distances >= 2 * radius - boundary_epsilon)
                   & (distances <= 3 * radius + boundary_epsilon)
                   & (nearest_dist > radius + boundary_epsilon))
        patch = raw[y0:y1, x0:x1].astype(np.float64)
        if np.count_nonzero(annulus) < 6:
            reasons.append("insufficient_background")
        background = float(np.median(patch[annulus])) if annulus.any() else float(np.median(patch))
        residual = patch[annulus] - background
        noise = max(float(1.4826 * np.median(np.abs(residual))) if residual.size else 0., 1.)
        flux = float(np.sum((patch - background)[support]))
        snr = flux / (noise * np.sqrt(max(1, int(support.sum()))))
        weights = np.maximum(patch - background, 0) * support
        positive_flux = float(weights.sum())
        if positive_flux > 0:
            cx, cy = float(np.sum(xx * weights) / positive_flux), float(np.sum(yy * weights) / positive_flux)
        else:
            cx, cy = px, py
            reasons.append("no_positive_support")
        shift = float(np.hypot(cx - px, cy - py))
        if snr < min_snr or flux <= 0:
            reasons.append("insufficient_raw_signal")
        if shift > max_shift:
            reasons.append("excessive_shift")
        _, nearest_new = tree.query([cx, cy])
        if int(nearest_new) != index:
            reasons.append("neighbor_ambiguity")
        point.update({"proposal_x": px, "proposal_y": py, "refined_x": cx, "refined_y": cy,
                      "shift": shift, "radius": radius, "support_flux": flux, "support_snr": float(snr),
                      "support_background": background, "support_noise": noise,
                      "support_pixels": int(support.sum()), "background_pixels": int(annulus.sum()),
                      "reasons": reasons})
        # In doubt preserve the proposal rather than report an untrusted move.
        point["x"], point["y"] = (px, py) if reasons else (cx, cy)
        rows.append(point)
    # Guard close convergence; keep both records as uncertain rather than merge
    # an ambiguous region into an apparently precise single measurement.
    refined_tree = cKDTree([[p["refined_x"], p["refined_y"]] for p in rows])
    for i, j in refined_tree.query_pairs(min(1.0, radius / 2)):
        for k in (i, j):
            if "refined_collision" not in rows[k]["reasons"]:
                rows[k]["reasons"].append("refined_collision")
            rows[k]["x"], rows[k]["y"] = xy[k]
    measurements = measure_sites(raw, rows, radius)
    accepted, uncertain = [], []
    for index, (point, measurement) in enumerate(zip(rows, measurements), 1):
        point.update({k: v for k, v in measurement.items() if k.startswith("raw_") or k == "roi_clipped"})
        point["id"] = index
        point["status"] = "uncertain" if point["reasons"] else "supported_candidate"
        point["reason"] = ";".join(point["reasons"]) if point["reasons"] else "raw_signal_supported"
        (uncertain if point["reasons"] else accepted).append(point)
    return accepted, uncertain
