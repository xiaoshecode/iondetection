"""Reference-site photometry. Labels are intensity heuristics, NOT quantum states."""
import numpy as np

from .data import _check_image


def measure_sites(image, sites, radius=3.0):
    """Measure fixed circular sites in raw pixels; never infer missing sites."""
    image = _check_image(image, "reference photometry input")
    try:
        radius = float(radius)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Reference radius must be a positive finite number.") from exc
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Reference radius must be a positive finite number.")
    rows = []
    h, w = image.shape
    # A larger reach cannot include additional pixels and may overflow int.
    reach = int(np.ceil(min(3.0 * radius, max(h, w))))
    for i, site in enumerate(sites):
        try:
            x, y = float(site["x"]), float(site["y"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Reference site {} requires numeric x and y coordinates.".format(i)) from exc
        if not np.isfinite([x, y]).all():
            raise ValueError("Reference site {} coordinates must be finite.".format(i))
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("Reference site {} lies outside the current image.".format(i))
        x0, x1 = max(0, int(x) - reach), min(w, int(x) + reach + 1)
        y0, y1 = max(0, int(y) - reach), min(h, int(y) + reach + 1)
        if x0 >= x1 or y0 >= y1:
            raise ValueError("Reference site lies outside the current image.")
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distance = np.hypot(xx - x, yy - y)
        roi = distance <= radius
        if not roi.any():
            raise ValueError(
                "Reference site {} has an empty pixel ROI; increase radius or check coordinates.".format(i)
            )
        annulus = (distance >= radius * 2) & (distance <= radius * 3)
        patch = image[y0:y1, x0:x1]
        bg = float(np.median(patch[annulus])) if annulus.any() else 0.0
        residual = patch[annulus] - bg
        noise = max(float(1.4826 * np.median(np.abs(residual))) if residual.size else 0.0, 1.0)
        flux = float((patch[roi] - bg).sum())
        rows.append({"id": site.get("id", i), "x": x, "y": y, "radius": float(radius),
                     "raw_flux": flux, "raw_peak": float(patch[roi].max()), "raw_background": bg,
                     "raw_snr": flux / (noise * np.sqrt(max(1, int(roi.sum())))),
                     "roi_clipped": x < radius or y < radius or x + radius >= w or y + radius >= h})
    return rows


def compare_reference(image, reference, bright_ratio=0.5, dim_ratio=0.15):
    image = _check_image(image, "reference comparison input")
    if tuple(image.shape) != tuple(reference["shape"]):
        raise ValueError("Reference/query shape mismatch. No automatic geometric registration is implemented.")
    if not np.isfinite([dim_ratio, bright_ratio]).all() or not 0 <= dim_ratio < bright_ratio:
        raise ValueError("Require finite 0 <= dim_ratio < bright_ratio.")
    rows = measure_sites(image, reference["sites"], reference["radius"])
    for row, site in zip(rows, reference["sites"]):
        base = float(site["raw_flux"])
        if not np.isfinite(base):
            raise ValueError("Reference raw_flux must be finite.")
        ratio = row["raw_flux"] / base if base > 0 else None
        row["reference_flux"] = base
        row["flux_ratio"] = ratio
        row["status"] = "uncertain"
        if ratio is not None and not row["roi_clipped"]:
            if ratio >= bright_ratio:
                row["status"] = "bright_like"
            elif ratio <= dim_ratio:
                row["status"] = "dim_candidate"
        row["interpretation"] = "intensity heuristic; dim is not proof of dark quantum state or occupancy"
    return rows
