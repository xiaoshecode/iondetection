"""Joint point-source fitting; a research baseline, not calibrated EMCCD MLE.

The PSF parameters are FIXED across each run. Positions, nonnegative amplitudes
and a nonnegative constant background are fitted together. Model selection uses
RSS/noise_std**2 + k*log(n), a Gaussian-noise BIC-style heuristic. In particular,
neither the score nor score differences are calibrated ion probabilities.
"""
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from ..classical.detection import detect_ions, estimate_detection_noise
from ..data import _check_image


@dataclass(frozen=True)
class PSFConfig:
    sigma_x: float = 1.4
    sigma_y: float = 1.4
    angle_deg: float = 0.0
    motion_amplitude: float = 0.0
    motion_angle_deg: float = 0.0
    motion_samples: int = 16
    noise_std: Optional[float] = None
    group_distance: float = 6.0
    padding: int = 6
    max_emitters: int = 5
    max_roi_pixels: int = 4096
    max_fit_groups: int = 256
    max_nfev: int = 120
    max_models: int = 12
    selection_rounds: int = 3
    min_delta_score: float = 6.0
    residual_sigma: float = 4.0
    min_separation: float = 0.75
    radius: float = 3.0

    def validate(self):
        for name in ("sigma_x", "sigma_y", "group_distance", "radius", "residual_sigma", "min_separation"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + " must be positive and finite")
        if min(self.sigma_x, self.sigma_y) < 0.5:
            raise ValueError("sigma_x/y must be >=0.5 px for this pixel-center PSF approximation")
        for name in ("angle_deg", "motion_angle_deg", "motion_amplitude", "min_delta_score"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(name + " must be finite")
        if self.motion_amplitude < 0 or self.min_delta_score < 0:
            raise ValueError("motion_amplitude and min_delta_score must be nonnegative")
        for name in ("motion_samples", "padding", "max_emitters", "max_roi_pixels", "max_fit_groups", "max_nfev", "max_models", "selection_rounds"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
                raise ValueError(name + " must be a positive integer")
        if self.motion_samples < 4 or self.max_models < 2:
            raise ValueError("motion_samples must be >=4 and max_models >=2")
        if self.noise_std is not None and (not np.isfinite(self.noise_std) or self.noise_std <= 0):
            raise ValueError("noise_std must be null or positive and finite, in raw ADU")
        return self


def _config(config):
    if config is None:
        return PSFConfig().validate()
    if isinstance(config, dict):
        try:
            config = PSFConfig(**config)
        except TypeError as exc:
            raise ValueError("Unknown PSF configuration field") from exc
    if not isinstance(config, PSFConfig):
        raise ValueError("config must be a PSFConfig or a mapping")
    return config.validate()


def _spot(xx, yy, x, y, config):
    angle = np.deg2rad(config.angle_deg)
    ct, st = np.cos(angle), np.sin(angle)
    if config.motion_amplitude:
        phase = np.arange(config.motion_samples) * (2 * np.pi / config.motion_samples)
        displacement = config.motion_amplitude * np.cos(phase)
        motion_angle = np.deg2rad(config.motion_angle_deg)
        dx = xx[None, :, :] - x - displacement[:, None, None] * np.cos(motion_angle)
        dy = yy[None, :, :] - y - displacement[:, None, None] * np.sin(motion_angle)
        return np.exp(-0.5 * (((ct * dx + st * dy) / config.sigma_x) ** 2 +
                              ((-st * dx + ct * dy) / config.sigma_y) ** 2)).mean(axis=0)
    dx, dy = xx - x, yy - y
    return np.exp(-0.5 * (((ct * dx + st * dy) / config.sigma_x) ** 2 +
                          ((-st * dx + ct * dy) / config.sigma_y) ** 2))


def render_expected(shape, ions, background=0.0, config=None):
    """Render a noiseless image; ions require x/y/amplitude_adu.

    amplitude_adu is the underlying unblurred Gaussian peak, not the observed
    motion-blurred peak or a calibrated photon count. Pixel-center sampling is
    used (not exact pixel integration); all distances are in pixels.
    """
    cfg = _config(config)
    if len(shape) != 2 or any(not isinstance(v, (int, np.integer)) or v < 1 for v in shape):
        raise ValueError("shape must contain two positive integer dimensions")
    if not np.isfinite(background) or background < 0:
        raise ValueError("background must be finite and nonnegative")
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    rendered = np.full(shape, float(background), dtype=np.float64)
    for ion in ions:
        x, y, amplitude = (float(ion[k]) for k in ("x", "y", "amplitude_adu"))
        if not np.isfinite([x, y, amplitude]).all() or amplitude < 0:
            raise ValueError("PSF coordinates/amplitude must be finite; amplitude >=0")
        rendered += amplitude * _spot(xx, yy, x, y, cfg)
    return rendered


def _seeds(seeds, shape):
    if isinstance(seeds, np.ndarray):
        values = np.asarray(seeds, dtype=float)
    else:
        values = np.asarray([[s["x"], s["y"]] if isinstance(s, dict) else s for s in seeds], dtype=float)
    if values.size == 0:
        return np.empty((0, 2), dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("seeds must be finite [x,y] rows or x/y dictionaries")
    if np.any(values < 0) or np.any(values[:, 0] > shape[1] - 1) or np.any(values[:, 1] > shape[0] - 1):
        raise ValueError("Seed centers must lie within pixel-center image bounds")
    return values


def _optimize(raw, seeds, cfg, noise):
    height, width = raw.shape
    count = len(seeds)
    scale = max(float(np.ptp(raw)), noise, 1e-12)
    observed = raw.astype(np.float64) / scale
    yy, xx = np.mgrid[:height, :width]
    bg = max(0.0, float(np.percentile(observed, 15)))
    if count == 0:
        background = max(float(observed.mean()), 0.0)
        expected = np.full(raw.shape, background * scale)
        rss = float(np.square(raw - expected).sum())
        return {"ions": [], "background_adu": background * scale, "score": rss / noise ** 2 + np.log(raw.size),
                "rss": rss, "expected": expected, "success": True, "nfev": 0, "message": "background_only"}
    initial = [bg]
    lower, upper = [0.0], [max(2.0, float(observed.max()) * 2)]
    for x, y in seeds:
        spot = _spot(xx, yy, x, y, cfg)
        amplitude = max(float(np.sum((observed - bg) * spot) / max(np.sum(spot ** 2), 1e-12)), 1e-6)
        initial.extend((float(x), float(y), amplitude))
        lower.extend((0.0, 0.0, 0.0))
        upper.extend((width - 1.0, height - 1.0, max(10.0, amplitude * 10.0)))

    def model(parameters):
        prediction = np.full(raw.shape, parameters[0], dtype=float)
        for i in range(count):
            x, y, amplitude = parameters[1 + 3*i:4 + 3*i]
            prediction += amplitude * _spot(xx, yy, x, y, cfg)
        return prediction

    opt = least_squares(lambda p: (model(p) - observed).ravel(), initial,
                        bounds=(lower, upper), x_scale="jac", max_nfev=cfg.max_nfev,
                        ftol=1e-7, xtol=1e-7, gtol=1e-7)
    expected = model(opt.x) * scale
    rss = float(np.square(raw - expected).sum())
    ions = []
    for i in range(count):
        x, y, amplitude = opt.x[1 + 3*i:4 + 3*i]
        ions.append({"x": float(x), "y": float(y), "amplitude_adu": float(amplitude * scale),
                     "radius": float(cfg.radius), "status": "psf_fitted"})
    coords = np.asarray([[ion["x"], ion["y"]] for ion in ions])
    separated = not any(np.linalg.norm(coords[i] - coords[j]) < cfg.min_separation
                        for i in range(count) for j in range(i))
    return {"ions": ions, "background_adu": float(opt.x[0] * scale),
            "score": rss / noise ** 2 + (3 * count + 1) * np.log(raw.size),
            "rss": rss, "expected": expected, "success": bool(opt.success and separated),
            "nfev": int(opt.nfev), "message": str(opt.message) if separated else "coincident_fitted_centers"}


def fit_patch(image, seeds, config=None, region_mask=None):
    """Jointly fit a small ROI; coordinates returned are ROI-relative.

    Start from supplied candidates; try background-only, merge/drop proposals,
    and positive-residual additions under a strict optimization budget. This is
    a local heuristic search, not an exhaustive comparison of all ion counts.
    An empty seed list stays empty: the public full-frame locator supplies
    classical candidates rather than hallucinating a crystal from background.
    """
    cfg = _config(config)
    raw = _check_image(image, "PSF patch").astype(np.float64)
    if min(raw.shape) < 2 or np.any(raw < 0):
        raise ValueError("PSF patch must be at least 2x2 and nonnegative")
    positions = _seeds(seeds, raw.shape)
    if len(positions) > cfg.max_emitters or raw.size > cfg.max_roi_pixels:
        raise ValueError("Patch exceeds max_emitters or max_roi_pixels; use locate_ions for explicit fallback")
    allowed = np.ones(raw.shape, dtype=bool)
    if region_mask is not None:
        if np.asarray(region_mask).shape != raw.shape or np.asarray(region_mask).dtype != np.bool_:
            raise ValueError("region_mask must be boolean with the same image shape")
        allowed = np.asarray(region_mask)
    if any(not allowed[min(raw.shape[0]-1, int(y+.5)), min(raw.shape[1]-1, int(x+.5))] for x, y in positions):
        raise ValueError("Seeds must be inside region_mask")
    noise = float(cfg.noise_std or estimate_detection_noise(raw, sigma=0.0))
    # Deterministic arrays with exactly zero residual need a meaningful floor.
    noise = max(noise, float(np.ptp(raw)) * 1e-6, 1e-9)
    trace = []

    def evaluate(candidate, label):
        result = _optimize(raw, candidate, cfg, noise)
        valid_region = all(allowed[min(raw.shape[0]-1, int(ion["y"]+.5)),
                                   min(raw.shape[1]-1, int(ion["x"]+.5))] for ion in result["ions"])
        if not valid_region:
            result["success"] = False
            result["message"] = "fitted_center_outside_region"
        trace.append({"proposal": label, "count": len(candidate), "score": float(result["score"]),
                      "success": result["success"], "nfev": result["nfev"], "message": result["message"]})
        return result

    if not len(positions):
        return {"ions": [], "diagnostics": {"status": "no_candidates", "models": [], "noise_std_adu": noise}}
    best = evaluate(positions, "initial")
    # A split peak may drive a two-emitter initial model into coincident
    # centers. Try a genuinely smaller nonempty model before falling back.
    # A converged empty model must not hide general optimization failure.
    if not best["success"] and len(positions) > 1:
        pairs = sorted((float(np.linalg.norm(positions[i] - positions[j])), i, j)
                       for i in range(len(positions)) for j in range(i))
        for _, i, j in pairs:
            if len(trace) >= cfg.max_models - 1:
                break
            merged = np.vstack((np.delete(positions, [i, j], axis=0), (positions[i] + positions[j]) / 2))
            candidate = evaluate(merged, "recover_by_merging")
            if candidate["success"]:
                best = candidate
                break
    if not best["success"]:
        fallback = [{"x": float(x), "y": float(y), "radius": float(cfg.radius),
                     "status": "classical_fallback"} for x, y in positions]
        return {"ions": fallback, "diagnostics": {"status": "optimization_failed", "models": trace,
                                                    "noise_std_adu": noise, "fallback": True}}
    null = evaluate(np.empty((0, 2)), "background_only")
    if null["score"] + cfg.min_delta_score < best["score"]:
        best = null
    for _ in range(cfg.selection_rounds):
        if len(trace) >= cfg.max_models or not best["ions"]:
            break
        centers = np.asarray([[ion["x"], ion["y"]] for ion in best["ions"]])
        proposals = []
        if len(centers) > 1:
            # Merge nearest pair first; its midpoint helps a double-peaked
            # motion PSF converge to the physical mean instead of one lobe.
            pairs = [(float(np.linalg.norm(centers[i] - centers[j])), i, j)
                     for i in range(len(centers)) for j in range(i)]
            _, i, j = min(pairs)
            merged = np.delete(centers, [i, j], axis=0)
            merged = np.vstack((merged, (centers[i] + centers[j]) / 2))
            proposals.append((merged, "merge_nearest"))
            for i in np.argsort([ion["amplitude_adu"] for ion in best["ions"]]):
                proposals.append((np.delete(centers, i, axis=0), "drop_one"))
        if len(centers) < cfg.max_emitters:
            residual = raw - best["expected"]
            yy, xx = np.mgrid[:raw.shape[0], :raw.shape[1]]
            eligible = allowed.copy()
            for x, y in centers:
                eligible &= (xx - x)**2 + (yy - y)**2 >= cfg.min_separation ** 2
            response = np.where(eligible, residual, -np.inf)
            iy, ix = np.unravel_index(np.argmax(response), response.shape)
            if response[iy, ix] > cfg.residual_sigma * noise:
                proposals.append((np.vstack((centers, (ix, iy))), "positive_residual"))
        improved = best
        for candidate, label in proposals:
            if len(trace) >= cfg.max_models:
                break
            result = evaluate(candidate, label)
            if result["success"] and result["score"] + cfg.min_delta_score < improved["score"]:
                improved = result
        if improved is best:
            break
        best = improved
    ions = sorted(best["ions"], key=lambda ion: (ion["y"], ion["x"]))
    for index, ion in enumerate(ions, 1):
        ion.update(id=index, fit_background_adu=best["background_adu"])
    return {"ions": ions, "diagnostics": {"status": "fitted", "initial_count": len(positions),
            "selected_count": len(ions), "score": float(best["score"]), "rss": float(best["rss"]),
            "noise_std_adu": noise, "models": trace, "model_budget_reached": len(trace) >= cfg.max_models,
            "fallback": False}}


def _groups(positions, distance):
    tree = cKDTree(positions)
    unseen = set(range(len(positions)))
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        group, pending = [first], [first]
        while pending:
            current = pending.pop()
            for index in tree.query_ball_point(positions[current], distance):
                if index in unseen:
                    unseen.remove(index)
                    group.append(index)
                    pending.append(index)
        yield sorted(group)


def _deduplicate_groups(ions, minimum_distance):
    """One-to-one cross-group duplicate reconciliation, without averaging.

    Separate fit halos may independently discover the same missed emitter.
    Prefer fitted results, then a center with more ROI support, then lower
    normalized residual. A representative accepts at most one point per source
    group; this prevents a middle hypothesis from swallowing two already
    distinct points from another joint fit. Matching uses a STRICT distance
    threshold to the retained coordinate, never a transitive cluster diameter.
    """
    if not ions:
        return [], []
    positions = np.asarray([[ion["x"], ion["y"]] for ion in ions], dtype=float)
    tree = cKDTree(positions)

    def rank(index):
        ion = ions[index]
        return (ion.get("status") != "psf_fitted",
                -float(ion.get("fit_support_margin_px", 0.0)),
                float(ion.get("fit_residual_noise_ratio", 1e30)),
                ion["y"], ion["x"], ion["fit_group"], index)

    accepted = set()
    represented_groups = {}
    removed = []
    for index in sorted(range(len(ions)), key=rank):
        group = ions[index]["fit_group"]
        matches = []
        for other in tree.query_ball_point(positions[index], minimum_distance):
            if other not in accepted or group in represented_groups[other]:
                continue
            distance = float(np.linalg.norm(positions[index] - positions[other]))
            if distance < minimum_distance:
                matches.append((distance, rank(other), other))
        if not matches:
            accepted.add(index)
            represented_groups[index] = {group}
            continue
        distance, _, keeper = min(matches)
        represented_groups[keeper].add(group)
        removed.append({"removed_group": group, "removed_x": float(ions[index]["x"]),
                        "removed_y": float(ions[index]["y"]), "removed_status": ions[index]["status"],
                        "kept_group": ions[keeper]["fit_group"], "kept_x": float(ions[keeper]["x"]),
                        "kept_y": float(ions[keeper]["y"]), "distance_px": distance})
    kept = []
    for index in sorted(accepted):
        row = dict(ions[index])
        if len(represented_groups[index]) > 1:
            row["merged_fit_groups"] = sorted(represented_groups[index])
        kept.append(row)
    return kept, removed


def locate_ions(image, config=None, detection_config=None, region_mask=None):
    """Classical proposals -> connected local groups -> joint PSF refinement.

    Oversized groups and groups exceeding the per-frame budget retain explicitly
    marked classical coordinates. Large connected lattices are never subjected
    to an unbounded all-ion nonlinear optimization. No incomplete fitting is
    represented as a successful physical inference.
    """
    cfg = _config(config)
    raw = _check_image(image, "PSF image")
    candidates, _ = detect_ions(raw, detection_config, region_mask=region_mask)
    if not candidates:
        return {"ions": [], "diagnostics": {"status": "no_candidates", "groups": [], "config": asdict(cfg)}}
    positions = np.asarray([[ion["x"], ion["y"]] for ion in candidates])
    ions, reports = [], []
    fitted_count = 0
    # The requested halo is automatically enlarged to include the prescribed
    # motion support plus four optical sigmas, unless limited by image edges.
    padding = max(cfg.padding, int(np.ceil(cfg.motion_amplitude + 4 * max(cfg.sigma_x, cfg.sigma_y))))
    for group_id, indices in enumerate(_groups(positions, cfg.group_distance)):
        local = positions[indices]
        x0 = max(0, int(np.floor(local[:, 0].min())) - padding)
        x1 = min(raw.shape[1], int(np.ceil(local[:, 0].max())) + padding + 1)
        y0 = max(0, int(np.floor(local[:, 1].min())) - padding)
        y1 = min(raw.shape[0], int(np.ceil(local[:, 1].max())) + padding + 1)
        reason = None
        if len(indices) > cfg.max_emitters:
            reason = "group_exceeds_max_emitters"
        elif (x1-x0)*(y1-y0) > cfg.max_roi_pixels:
            reason = "roi_exceeds_max_pixels"
        elif fitted_count >= cfg.max_fit_groups:
            reason = "frame_fit_budget_reached"
        # Other classical centers in this halo are nuisance emitters. Jointly
        # fitting only part of them would pull positions toward omitted light.
        neighbors = [i for i, (x, y) in enumerate(positions)
                     if x0 <= x <= x1-1 and y0 <= y <= y1-1 and i not in indices]
        if neighbors and reason is None:
            reason = "neighbor_candidates_in_fit_halo"
        if reason:
            for index in indices:
                ion = dict(candidates[index])
                ion.update(status="classical_fallback", fit_group=group_id, fallback_reason=reason)
                ions.append(ion)
            reports.append({"group": group_id, "status": reason, "fallback": True, "initial_count": len(indices)})
            continue
        fitted_count += 1
        result = fit_patch(raw[y0:y1, x0:x1], local - (x0, y0), cfg,
                           region_mask[y0:y1, x0:x1] if region_mask is not None else None)
        for ion in result["ions"]:
            if ion["status"] == "psf_fitted":
                ion["fit_support_margin_px"] = float(min(ion["x"], ion["y"],
                                                         x1-x0-1-ion["x"], y1-y0-1-ion["y"]))
                ion["fit_residual_noise_ratio"] = float(np.sqrt(result["diagnostics"]["rss"] /
                                                                ((x1-x0)*(y1-y0))) /
                                                        result["diagnostics"]["noise_std_adu"])
            ion.update(x=float(ion["x"] + x0), y=float(ion["y"] + y0), fit_group=group_id)
            ions.append(ion)
        reports.append(dict(result["diagnostics"], group=group_id, roi=[x0, y0, x1, y1]))
    ions, duplicates = _deduplicate_groups(ions, cfg.min_separation)
    ions.sort(key=lambda ion: (ion["y"], ion["x"]))
    for index, ion in enumerate(ions, 1):
        ion["id"] = index
    fallback_count = sum(ion["status"] == "classical_fallback" for ion in ions)
    return {"ions": ions, "coverage_complete": True, "has_abstentions": False,
            "diagnostics": {"status": "mixed_fallback" if fallback_count else "fitted",
                            "initial_count": len(candidates), "fitted_groups_attempted": fitted_count,
                            "fallback_ions": fallback_count, "groups": reports, "config": asdict(cfg),
                            "cross_group_duplicates_removed": len(duplicates),
                            "cross_group_duplicate_details": duplicates,
                            "cross_group_merge_distance_px": cfg.min_separation,
                            "noise_model": "constant Gaussian residual, BIC-style heuristic; not calibrated EMCCD",
                            "center_definition": "center of fixed PSF; symmetric motion uses temporal mean"}}
