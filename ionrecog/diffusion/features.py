"""Frozen FLUX.1-Kontext spatial features and offline cosine correspondence.

This probes the reference/target image streams *during an editing trajectory*;
it is not DIFT input-noising, a DiTF reproduction, or an ion detector. Hooks use
the public PyTorch module interface and the FluxKontextPipeline implementation
verified against diffusers 0.35.1. Text tokens are never saved as image tokens.
"""

from __future__ import annotations

import inspect
import json
import math
from functools import wraps
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_PROMPT = "Preserve the image exactly."
SCHEMA = "ionrecog.flux-kontext-features.v1"


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy().copy()
    return np.asarray(value)


def _grid_from_ids(ids, original_hw):
    """Return row-major order and pixel-center coordinates (x, y), not edges."""
    positions = np.asarray(ids)[:, 1:3]
    if not np.isfinite(positions).all() or not np.allclose(positions, np.round(positions)):
        raise ValueError("Flux image IDs must contain finite integer grid positions")
    positions = positions.astype(np.int64)
    if len(positions) == 0 or positions.min() < 0:
        raise ValueError("Missing or negative image grid positions")
    gh, gw = (positions.max(axis=0) + 1).tolist()
    linear = positions[:, 0] * gw + positions[:, 1]
    if len(linear) != gh * gw or not np.array_equal(np.sort(linear), np.arange(gh * gw)):
        raise ValueError("Image token IDs do not form one complete rectangular grid")
    order = np.argsort(linear)
    oh, ow = original_hw
    yy, xx = np.indices((gh, gw), dtype=np.float32)
    xy = np.stack(((xx + 0.5) * ow / gw - 0.5, (yy + 0.5) * oh / gh - 0.5), axis=-1)
    return order, (gh, gw), xy.reshape(-1, 2)


def split_spatial_tokens(image_tokens, image_ids, original_hw):
    """Split actual Flux image IDs (0=target, 1=reference), excluding text.

    Returns dictionaries with features [grid_h, grid_w, channels] and xy [N,2].
    Only one input image and batch size one are supported deliberately.
    """
    tokens, ids = _numpy(image_tokens), _numpy(image_ids)
    if tokens.ndim == 3 and tokens.shape[0] == 1:
        tokens = tokens[0]
    if ids.ndim == 3 and ids.shape[0] == 1:
        ids = ids[0]
    if tokens.ndim != 2 or ids.ndim != 2 or ids.shape[1] != 3 or len(tokens) != len(ids):
        raise ValueError("Expected one batch of spatial tokens [N,C] and matching image IDs [N,3]")
    if not np.isfinite(tokens).all() or not np.isfinite(ids).all():
        raise ValueError("Non-finite feature or image ID")
    if not np.array_equal(np.unique(ids[:, 0]), np.array([0, 1])):
        raise ValueError("Expected exactly Flux target IDs=0 and one reference image IDs=1")
    result = {}
    for name, identifier in (("target", 0), ("reference", 1)):
        mask = ids[:, 0] == identifier
        order, shape, xy = _grid_from_ids(ids[mask], original_hw)
        result[name] = {"features": tokens[mask][order].reshape(*shape, tokens.shape[-1]), "xy": xy}
    return result


class FluxFeatureCapture:
    """Context manager capturing chosen dual-stream block outputs once.

    FluxTransformerBlock returns (text_tokens, image_tokens). The transformer's
    own img_ids explicitly distinguish reference tokens from noisy target ones.
    We fail on incompatible signatures/shapes instead of guessing offsets.
    """

    def __init__(self, transformer, layers=(0,), capture_step=0):
        self.transformer = transformer
        self.layers = tuple(int(x) for x in layers)
        if not self.layers or len(set(self.layers)) != len(self.layers) or min(self.layers) < 0:
            raise ValueError("layers must be unique non-negative dual-stream block indices")
        if capture_step < 0:
            raise ValueError("capture_step must be non-negative")
        blocks = getattr(transformer, "transformer_blocks", ())
        if max(self.layers) >= len(blocks):
            raise ValueError("Requested layer exceeds Flux dual-stream transformer_blocks")
        self.capture_step = int(capture_step)
        self.call_index = -1
        self.image_ids = None
        self.timestep = None
        self.features = {}
        self.handles = []
        self._original_forward = None

    def _before_transformer(self, module, args, kwargs):
        self.call_index += 1
        if self.call_index != self.capture_step:
            return
        values = inspect.signature(module.forward).bind_partial(*args, **kwargs).arguments
        if "img_ids" not in values:
            raise RuntimeError("Flux transformer call did not expose img_ids; unsupported diffusers API")
        self.image_ids = _numpy(values["img_ids"])
        if "timestep" in values:
            self.timestep = _numpy(values["timestep"]).reshape(-1).tolist()

    def _after_block(self, index):
        def capture(module, args, output):
            if self.call_index != self.capture_step:
                return
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError("Expected Flux block output (text_tokens, image_tokens)")
            image_tokens = output[1]
            if image_tokens.ndim != 3 or image_tokens.shape[0] != 1:
                raise RuntimeError("Feature probe requires batch=1 spatial image tokens")
            self.features[index] = _numpy(image_tokens)
        return capture

    def __enter__(self):
        try:
            if "with_kwargs" in inspect.signature(self.transformer.register_forward_pre_hook).parameters:
                self.handles.append(self.transformer.register_forward_pre_hook(self._before_transformer, with_kwargs=True))
            else:
                # Allows CPU protocol tests with the project's older torch 1.13.
                # Actual diffusers/FLUX inference still requires torch >= 2.
                original = self.transformer.forward
                self._original_forward = original
                @wraps(original)
                def observed_forward(*args, **kwargs):
                    self._before_transformer(self.transformer, args, kwargs)
                    return original(*args, **kwargs)
                self.transformer.forward = observed_forward
            for index in self.layers:
                self.handles.append(self.transformer.transformer_blocks[index].register_forward_hook(self._after_block(index)))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self._original_forward is not None:
            self.transformer.forward = self._original_forward
            self._original_forward = None


def _read_probe_image(path, width, height, intensity_offset, intensity_scale):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile
        array = tifffile.imread(path)
    else:
        with Image.open(path) as image:
            array = np.array(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (3, 4) or min(array.shape[:2]) < 1:
        raise ValueError("Expected a single grayscale or RGB image, not a stack")
    array = array[..., :3]
    if not np.isfinite(array).all():
        raise ValueError("Input image has non-finite values")
    if not np.isfinite(intensity_offset):
        raise ValueError("intensity_offset must be finite")
    if intensity_scale is None:
        if np.issubdtype(array.dtype, np.unsignedinteger):
            intensity_scale = float(np.iinfo(array.dtype).max)
        elif np.issubdtype(array.dtype, np.signedinteger):
            raise ValueError("Signed-integer images require intensity_scale (some Pillow versions expose 16-bit PNG as int32)")
        elif array.min() >= 0 and array.max() <= 1:
            intensity_scale = 1.0
        else:
            raise ValueError("Float images outside [0,1] require an explicit intensity_scale")
    if not np.isfinite(intensity_scale) or intensity_scale <= 0:
        raise ValueError("intensity_scale must be finite and positive")
    original_hw = array.shape[:2]
    oh, ow = original_hw
    if width is None and height is None:
        width, height = math.ceil(ow / 16) * 16, math.ceil(oh / 16) * 16
    elif width is None:
        width = max(16, round(height * ow / oh / 16) * 16)
    elif height is None:
        height = max(16, round(width * oh / ow / 16) * 16)
    width, height = int(width), int(height)
    if min(width, height) < 16 or width % 16 or height % 16:
        raise ValueError("Probe width and height must be positive multiples of 16")
    scaled = (array.astype(np.float32) - intensity_offset) / intensity_scale
    clipping_fraction = float(np.mean((scaled < 0) | (scaled > 1)))
    # A three-channel float tensor preserves raw-camera precision through the VAE.
    # PIL float per-channel resizing avoids an accidental uint8 conversion.
    channels = [np.asarray(Image.fromarray(np.clip(scaled[..., c], 0, 1)).resize(
        (width, height), Image.Resampling.BILINEAR), dtype=np.float32) for c in range(3)]
    rgb = np.stack(channels, axis=0)
    info = {"original_hw": list(original_hw), "processed_hw": [height, width],
            "intensity_offset": float(intensity_offset), "intensity_scale": float(intensity_scale),
            "clipping_fraction": clipping_fraction, "resize": "bilinear_exact_no_crop",
            "processed_to_original_scale_xy": [ow / width, oh / height]}
    return rgb, info


def _load_pipeline(model_path, dtype, local_files_only):
    try:
        import diffusers
        from diffusers import FluxKontextPipeline
    except ImportError as exc:
        raise ImportError("Feature extraction needs the optional diffusers/transformers/accelerate dependencies; use diffusers==0.35.1") from exc
    pipe = FluxKontextPipeline.from_pretrained(str(model_path), torch_dtype=dtype, local_files_only=local_files_only)
    pipe._ion_probe_diffusers_version = diffusers.__version__
    return pipe


def extract_features(image_path, output, model_path, *, device="cuda", layers=(0,), seed=42,
                     steps=1, precision="bf16", local_files_only=True, prompt=DEFAULT_PROMPT,
                     width=None, height=None, capture_step=0, cpu_offload=False,
                     intensity_offset=0.0, intensity_scale=None, max_sequence_length=128):
    """Save frozen editing-trajectory features.npz + metadata.json; return JSON path.

    ``output`` is a new/empty artifact directory. At ``steps=1`` the target is
    initially noise; the reference stream represents the actual input. No image
    reconstruction quality is implied. ``local_files_only=False`` explicitly
    opts into model download. Coordinates describe token centers, not ions.
    """
    import torch
    if steps < 1 or capture_step < 0 or capture_step >= steps:
        raise ValueError("Require steps >= 1 and 0 <= capture_step < steps")
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("precision must be bf16, fp16, or fp32")
    if not 1 <= max_sequence_length <= 512:
        raise ValueError("max_sequence_length must be between 1 and 512")
    if cpu_offload and not str(device).startswith("cuda"):
        raise ValueError("cpu_offload requires a CUDA execution device")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use a CUDA server or device='cpu', precision='fp32'")
    target_dir = Path(output)
    if (target_dir / "features.npz").exists() or (target_dir / "metadata.json").exists():
        raise FileExistsError("Feature output already exists; choose a new output directory")
    rgb, image_info = _read_probe_image(image_path, width, height, intensity_offset, intensity_scale)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    pipe = _load_pipeline(model_path, dtype, local_files_only)
    parameters = inspect.signature(pipe.__call__).parameters
    if "_auto_resize" not in parameters or "max_area" not in parameters:
        raise RuntimeError("Unsupported FluxKontextPipeline API: require _auto_resize/max_area (verified on diffusers 0.35.1)")
    for component in ("transformer", "vae", "text_encoder", "text_encoder_2"):
        module = getattr(pipe, component, None)
        if module is not None:
            module.eval()
            module.requires_grad_(False)
    if cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    ph, pw = image_info["processed_hw"]
    # Respect the observed pixels: disable Kontext's automatic 1-megapixel resize.
    capture = FluxFeatureCapture(pipe.transformer, layers=layers, capture_step=capture_step)
    with capture, torch.inference_mode():
        pipe(image=torch.from_numpy(rgb).unsqueeze(0), prompt=prompt, height=ph, width=pw,
             max_area=ph * pw, _auto_resize=False, num_inference_steps=int(steps),
             guidance_scale=2.5, true_cfg_scale=1.0, num_images_per_prompt=1,
             max_sequence_length=max_sequence_length, output_type="latent",
             generator=torch.Generator("cpu").manual_seed(int(seed)))
    if capture.image_ids is None or set(capture.features) != set(capture.layers):
        raise RuntimeError("Requested features were not captured; verify diffusers model/hook compatibility")
    arrays, grids = {}, {}
    for index in capture.layers:
        streams = split_spatial_tokens(capture.features[index], capture.image_ids, image_info["original_hw"])
        for name, stream in streams.items():
            arrays[f"{name}_layer_{index}"] = stream["features"].astype(np.float32)
            arrays[f"{name}_xy"] = stream["xy"]
            grids[name] = list(stream["features"].shape[:2])
    expected_grid = [ph // 16, pw // 16]
    if grids["reference"] != expected_grid or grids["target"] != expected_grid:
        raise RuntimeError("Pipeline changed image geometry unexpectedly; refusing inaccurate coordinate mapping")
    metadata = {"schema": SCHEMA, "backend": "flux_kontext", "model_path": str(model_path),
                "image_path": str(Path(image_path).resolve()), "diffusers_version": getattr(pipe, "_ion_probe_diffusers_version", "unknown"),
                "torch_version": torch.__version__, "layers": list(capture.layers), "steps": int(steps),
                "capture_step": int(capture_step), "transformer_timestep": capture.timestep,
                "seed": int(seed), "precision": precision, "device": str(device),
                "prompt": prompt, "guidance_scale": 2.5, "max_sequence_length": max_sequence_length,
                "local_files_only": bool(local_files_only), "cpu_offload": bool(cpu_offload),
                "grid_hw": grids, "image": image_info,
                "feature_kind": "frozen_editing_trajectory_dual_stream_block_output",
                "coordinate_convention": "zero-based pixel centers; xy is flattened row-major",
                "target_coordinate_caveat": "target_xy is nominal output-to-input resize geometry, not evidence of content correspondence",
                "limitations": ["Not input-noising DIFT or DiTF", "Token matches are not ion detections", "All model weights frozen; no feature quality guarantee"],
                "artifacts": {"features": "features.npz", "metadata": "metadata.json"}}
    arrays["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False))
    target_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target_dir / "features.npz", **arrays)
    path = target_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _load_feature(path, layer, stream):
    with np.load(path, allow_pickle=False) as data:
        key = f"{stream}_layer_{layer}"
        if key not in data or f"{stream}_xy" not in data or "metadata_json" not in data:
            raise ValueError(f"Missing {key}, coordinates or metadata in {path}")
        features = np.asarray(data[key], dtype=np.float32)
        xy = np.asarray(data[f"{stream}_xy"], dtype=np.float32)
        metadata = json.loads(str(data["metadata_json"].item()))
    if metadata.get("schema") != SCHEMA:
        raise ValueError("Unsupported feature artifact schema")
    if features.ndim != 3 or min(features.shape) < 1 or xy.shape != (features.shape[0] * features.shape[1], 2):
        raise ValueError("Feature grid and coordinate shape mismatch")
    if not np.isfinite(features).all() or not np.isfinite(xy).all():
        raise ValueError("Feature artifact contains non-finite values")
    return features.reshape(-1, features.shape[-1]), xy, metadata, list(features.shape[:2])


def compare_features(source_npz, target_npz, output, *, layer=0, stream="reference",
                     normalization="l2", one_to_one=False, mutual=False, min_similarity=-1.0,
                     max_matrix_entries=16777216):
    """Save cosine.npy and matches.json (token correspondences, not ion centers).

    normalization='channel_standardize' uses pooled per-channel mean/std from
    both images followed by L2 normalization. It is explicitly NOT DiTF.
    one_to_one uses a maximum-total-cosine Hungarian assignment before score
    filtering; mutual then optionally filters to bidirectional nearest matches.
    Full cosine matrices are capped to keep memory and disk use explicit.
    """
    if stream not in {"reference", "target"} or normalization not in {"l2", "channel_standardize"}:
        raise ValueError("Unsupported stream or normalization")
    if not -1 <= min_similarity <= 1 or max_matrix_entries < 1:
        raise ValueError("Invalid min_similarity or max_matrix_entries")
    a, axy, ameta, ashape = _load_feature(source_npz, layer, stream)
    b, bxy, bmeta, bshape = _load_feature(target_npz, layer, stream)
    if a.shape[1] != b.shape[1]:
        raise ValueError("Feature channel dimensions differ")
    for field in ("backend", "model_path", "capture_step", "steps", "prompt", "guidance_scale", "precision"):
        if ameta.get(field) != bmeta.get(field):
            raise ValueError(f"Feature extraction settings differ: {field}")
    entries = len(a) * len(b)
    if entries > max_matrix_entries:
        raise ValueError(f"Cosine matrix requires {entries} entries (limit {max_matrix_entries}); use smaller crops/grids")
    if normalization == "channel_standardize":
        pooled = np.concatenate((a, b), axis=0)
        mean, std = pooled.mean(axis=0), pooled.std(axis=0)
        a, b = (a - mean) / np.maximum(std, 1e-6), (b - mean) / np.maximum(std, 1e-6)
    anorm, bnorm = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    av, bv = anorm > 1e-8, bnorm > 1e-8
    a = a / np.maximum(anorm[:, None], 1e-8)
    b = b / np.maximum(bnorm[:, None], 1e-8)
    cosine = np.clip(a @ b.T, -1, 1).astype(np.float32)
    # Invalid zero vectors have undefined cosine and must never become matches.
    valid_scores = cosine.copy()
    valid_scores[~av, :] = -2
    valid_scores[:, ~bv] = -2
    if one_to_one:
        from scipy.optimize import linear_sum_assignment
        source_indices = np.flatnonzero(av)
        target_indices = np.flatnonzero(bv)
        if len(source_indices) and len(target_indices):
            rows, cols = linear_sum_assignment(-cosine[np.ix_(source_indices, target_indices)])
            rows, cols = source_indices[rows], target_indices[cols]
        else:
            rows, cols = np.array([], dtype=int), np.array([], dtype=int)
    else:
        rows, cols = np.arange(len(a)), valid_scores.argmax(axis=1)
    keep = av[rows] & bv[cols] & (cosine[rows, cols] >= min_similarity)
    if mutual:
        keep &= (valid_scores.argmax(axis=0)[cols] == rows) & (valid_scores.argmax(axis=1)[rows] == cols)
    rows, cols = rows[keep], cols[keep]
    target_dir = Path(output)
    if (target_dir / "cosine.npy").exists() or (target_dir / "matches.json").exists():
        raise FileExistsError("Comparison output exists; choose a new directory")
    result = {"schema": "ionrecog.feature-correspondence.v1", "source": str(Path(source_npz).resolve()),
              "target": str(Path(target_npz).resolve()), "layer": int(layer), "stream": stream,
              "normalization": normalization, "normalization_note": "Channel standardization is not a DiTF reproduction",
              "one_to_one": bool(one_to_one), "mutual": bool(mutual), "min_similarity": float(min_similarity),
              "assignment": "maximum-total-cosine then threshold" if one_to_one else "nearest-cosine then threshold",
              "source_grid_hw": ashape, "target_grid_hw": bshape, "matrix_shape": list(cosine.shape),
              "invalid_zero_source_tokens": int((~av).sum()), "invalid_zero_target_tokens": int((~bv).sum()),
              "coordinate_convention": ameta.get("coordinate_convention"),
              "caveat": "Spatial token correspondences, NOT ion detections or validated physical ion identity; no confidence calibration",
              "matches": [{"source_index": int(i), "target_index": int(j), "source_xy": axy[i].tolist(),
                           "target_xy": bxy[j].tolist(), "cosine": float(cosine[i, j])} for i, j in zip(rows, cols)]}
    target_dir.mkdir(parents=True, exist_ok=True)
    np.save(target_dir / "cosine.npy", cosine, allow_pickle=False)
    path = target_dir / "matches.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
