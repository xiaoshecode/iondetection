"""Point matching requires explicit ground truth; restoration loss is not accuracy."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def _point_array(points, name):
    """Accept explicit [x, y] rows, including an empty JSON list."""
    message = name + " must be a finite Nx2 array of [x, y] positions."
    try:
        result = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(message) from error
    if result.ndim == 1 and result.size == 0:
        result = result.reshape(0, 2)
    if result.ndim != 2 or result.shape[1] != 2 or not np.isfinite(result).all():
        raise ValueError(message)
    return result


def match_points(predicted, truth, tolerance=3.0):
    try:
        if np.ndim(tolerance) != 0:
            raise ValueError("tolerance must be a finite positive scalar")
        tolerance = float(tolerance)
    except (TypeError, ValueError) as error:
        raise ValueError("tolerance must be a finite positive scalar") from error
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a finite positive scalar")
    p = _point_array(predicted, "predicted")
    t = _point_array(truth, "truth")
    errors = []
    if len(p) and len(t):
        distances = cdist(p, t)
        # Large inadmissible cost prioritizes maximum valid matching cardinality.
        penalty = (min(len(p), len(t)) + 1) * (tolerance + 1)
        pi, ti = linear_sum_assignment(np.where(distances <= tolerance, distances, penalty))
        errors = distances[pi, ti][distances[pi, ti] <= tolerance].tolist()
    tp, fp, fn = len(errors), len(p) - len(errors), len(t) - len(errors)
    precision = tp / len(p) if len(p) else (1.0 if not len(t) else 0.0)
    recall = tp / len(t) if len(t) else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "mean_error_px": float(np.mean(errors)) if errors else None,
            "rmse_px": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            "all_correct": fp == 0 and fn == 0, "tolerance_px": tolerance}
