"""Tiled point prediction in raw-image coordinates; no classical detector labels."""
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from ..data import _check_image


def choose_device(device="cpu"):
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu, cuda or auto")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this Python environment")
    return torch.device(device)


def validate_options(threshold, min_distance, max_ions, tile_size, halo, support_snr):
    if not np.isfinite([threshold, min_distance, support_snr]).all():
        raise ValueError("Inference thresholds must be finite")
    if not 0 < threshold < 1 or min_distance < 0 or support_snr < 0:
        raise ValueError("Require 0 < threshold < 1; min_distance/support_snr >= 0")
    if any(not isinstance(v, int) or isinstance(v, bool) for v in (max_ions, tile_size, halo)):
        raise ValueError("max_ions, tile_size and halo must be integers")
    if max_ions < 1 or tile_size < 32 or halo < 16 or tile_size <= 2 * halo:
        raise ValueError("Require max_ions >=1, tile_size >=32, halo >=16, tile_size >2*halo")


def decode_maps(probability, offsets, threshold=.3, min_distance=1.5, max_ions=5000, region_mask=None):
    """Decode in zero-based x=column/y=row coordinates, retain learned offsets.

    The 3x3 peak suppression is a baseline limitation in extremely dense images.
    It does not claim SHOT's NMS-free set prediction.
    """
    probability = np.asarray(probability)
    offsets = np.asarray(offsets)
    if probability.ndim != 2 or offsets.shape != (2,) + probability.shape:
        raise ValueError("Expected HxW probability and 2xHxW offsets")
    if not np.isfinite(probability).all() or not np.isfinite(offsets).all():
        raise ValueError("Non-finite network output")
    if not np.isfinite([threshold, min_distance]).all() or not 0 < threshold < 1 or min_distance < 0:
        raise ValueError("Invalid decode threshold/min_distance")
    if not isinstance(max_ions, int) or max_ions < 1:
        raise ValueError("max_ions must be a positive integer")
    h, w = probability.shape
    if region_mask is not None:
        region_mask = np.asarray(region_mask)
        if region_mask.shape != probability.shape or region_mask.dtype != np.bool_:
            raise ValueError("region_mask must be a matching boolean array")
    candidate = (probability >= threshold) & (probability == ndi.maximum_filter(probability, 3, mode="constant", cval=-np.inf))
    yy, xx = np.nonzero(candidate)
    order = np.argsort(-probability[yy, xx], kind="stable")
    rows = []
    # Spatial buckets avoid quadratic postprocessing for hundreds of ions.
    cell = max(float(min_distance), 1.)
    buckets = {}
    capped = False
    for index in order:
        iy, ix = yy[index], xx[index]
        x, y = float(ix + offsets[0, iy, ix]), float(iy + offsets[1, iy, ix])
        if not (0 <= x < w and 0 <= y < h):
            continue
        mx, my = min(w - 1, int(np.floor(x + .5))), min(h - 1, int(np.floor(y + .5)))
        if region_mask is not None and not region_mask[my, mx]:
            continue
        bx, by = int(x // cell), int(y // cell)
        close = any((x - px) ** 2 + (y - py) ** 2 < min_distance ** 2
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    for px, py in buckets.get((bx + dx, by + dy), ()))
        if close:
            continue
        if len(rows) >= max_ions:
            capped = True
            break
        # Display/photometry radius belongs to the shared detection config.
        # The localization head estimates centers, not physical spot radii.
        rows.append({"x": x, "y": y, "score": float(probability[iy, ix]),
                     "status": "candidate"})
        buckets.setdefault((bx, by), []).append((x, y))
    rows.sort(key=lambda row: (row["y"], row["x"]))
    return rows, capped


def predict_maps(model, raw, input_offset, input_scale, device, tile_size=256, halo=32):
    """Stitch core pixel predictions before global decoding, avoiding tile duplicates.

    Each tile keeps its halo only for convolutional context. Pixel cores partition
    the image exactly, then one global peak/offset operation handles boundaries.
    """
    import torch
    raw = _check_image(raw, "point inference")
    if min(raw.shape) < 2:
        raise ValueError("Point inference requires both dimensions >= 2")
    if not np.isfinite([input_offset, input_scale]).all() or input_scale <= 0:
        raise ValueError("Invalid checkpoint normalization")
    if not isinstance(tile_size, int) or not isinstance(halo, int) or halo < 16 or tile_size <= 2 * halo:
        raise ValueError("tile_size must exceed 2*halo, with halo >=16")
    h, w = raw.shape
    output = np.empty((3, h, w), dtype=np.float32)
    core = tile_size - 2 * halo
    model.eval()
    with torch.no_grad():
        for y in range(0, h, core):
            for x in range(0, w, core):
                ye, xe = min(h, y + core), min(w, x + core)
                y0, x0 = max(0, y - halo), max(0, x - halo)
                y1, x1 = min(h, ye + halo), min(w, xe + halo)
                # Keep pooling alignment fixed relative to the original image.
                y0 -= y0 % 2
                x0 -= x0 % 2
                patch = (raw[y0:y1, x0:x1] - input_offset) / input_scale
                # Even spatial sizes keep downsampling and 2x interpolation
                # on the same lattice across tiles. Pad only bottom/right, so
                # the owned core keeps its original absolute coordinates.
                patch = np.pad(patch, ((0, patch.shape[0] % 2), (0, patch.shape[1] % 2)), mode="edge")
                tensor = torch.from_numpy(np.ascontiguousarray(patch))[None, None].to(device)
                logits, offset = model(tensor)
                maps = torch.cat((torch.sigmoid(logits), offset), 1)[0].cpu().numpy()
                output[:, y:ye, x:xe] = maps[:, y - y0:ye - y0, x - x0:xe - x0]
    return output[0], output[1:]


class PointPredictor:
    def __init__(self, checkpoint, device="cpu", threshold=None, min_distance=1.5,
                 max_ions=5000, tile_size=256, halo=32, support_snr=0., num_threads=2):
        import torch
        from .models import PointNet
        if not isinstance(num_threads, int) or num_threads < 1:
            raise ValueError("num_threads must be positive")
        torch.set_num_threads(num_threads)
        self.device = choose_device(device)
        self.checkpoint = Path(checkpoint)
        # Project-generated trusted checkpoints only: torch 1.13 uses pickle.
        state = torch.load(str(self.checkpoint), map_location="cpu")
        if not isinstance(state, dict) or state.get("method") != "point_localization" or state.get("schema_version") != 1:
            raise ValueError("Not a supported point_localization checkpoint")
        if not isinstance(state.get("completed_epochs"), int) or state["completed_epochs"] < 1:
            raise ValueError("Checkpoint has no completed training epoch")
        try:
            sim = state["simulation"]
            self.input_offset = float(sim["input_offset"])
            self.input_scale = float(sim["input_scale"])
            self.threshold = float(state["selected_threshold"] if threshold is None else threshold)
            self.model = PointNet(state["channels"]).to(self.device)
            self.model.load_state_dict(state["model_state"])
        except (KeyError, TypeError, RuntimeError) as exc:
            raise ValueError("Malformed or incompatible point checkpoint: %s" % exc) from exc
        if not np.isfinite([self.input_offset, self.input_scale]).all() or self.input_scale <= 0:
            raise ValueError("Checkpoint normalization must be finite, scale >0")
        validate_options(self.threshold, min_distance, max_ions, tile_size, halo, support_snr)
        self.min_distance, self.max_ions = min_distance, max_ions
        self.tile_size, self.halo, self.support_snr = tile_size, halo, support_snr
        self.model.eval()

    def __call__(self, raw, region_mask=None):
        raw = _check_image(raw, "point inference")
        if np.any(raw < 0):
            raise ValueError("Point inference expects nonnegative raw-scale pixels")
        if raw.max() == raw.min():
            return {"ions": [], "diagnostics": {"constant_image": True, "threshold": self.threshold}}
        probability, offsets = predict_maps(self.model, raw, self.input_offset, self.input_scale,
                                            self.device, self.tile_size, self.halo)
        ions, capped = decode_maps(probability, offsets, self.threshold, self.min_distance,
                                  self.max_ions, region_mask)
        rejected = 0
        if self.support_snr > 0 and ions:
            from ..classical.detection import estimate_detection_noise
            signal = ndi.gaussian_filter(raw, 1.) - ndi.gaussian_filter(raw, 10.)
            noise = estimate_detection_noise(raw, 1.)
            accepted = []
            for ion in ions:
                x, y = int(np.floor(ion["x"] + .5)), int(np.floor(ion["y"] + .5))
                x, y = min(x, raw.shape[1] - 1), min(y, raw.shape[0] - 1)
                # Filter evidence only; do not move learned centers to raw peaks.
                if signal[y, x] >= noise * self.support_snr:
                    accepted.append(ion)
                else:
                    rejected += 1
            ions = accepted
        return {"ions": ions, "coverage_complete": not capped, "has_abstentions": capped,
                "diagnostics": {"threshold": self.threshold, "score_is_calibrated_probability": False,
                                "max_ions_reached": capped, "raw_support_rejected": rejected,
                                "support_snr": self.support_snr, "tile_size": self.tile_size,
                                "halo": self.halo, "input_offset": self.input_offset,
                                "input_scale": self.input_scale,
                                "training_domain": "synthetic_not_experimental_ground_truth"}}
