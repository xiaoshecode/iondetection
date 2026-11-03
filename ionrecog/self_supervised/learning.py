"""Self-supervised restoration baselines, not validated ion-state classifiers.

Masked training assumes conditionally independent, suitably zero-mean pixel
noise. Diffusion learns the distribution of independent repeat-frame averages:
those are low-noise proxy targets, NOT clean, deblurred ground truth. No manual
ion annotations are used. Repeats must share geometry, brightness, exposure and
gain; this module does not register motion or correct state changes.
"""
import copy
import json
import math
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..data import load_image, list_images, find_groups
from .models import SmallUNet, diffusion_schedule, predict_tiled


DEFAULTS = {
    "data": {"offset": 0.0, "scale": 65535.0, "patch_size": 64, "max_target_frames": 8},
    "model": {"base_channels": 16},
    "train": {"epochs": 5, "steps_per_epoch": 100, "batch_size": 4, "lr": 0.001,
              "seed": 42, "mask_fraction": 0.02, "val_steps": 10, "num_threads": 4},
    "diffusion": {"timesteps": 100, "sampling_steps": 20, "schedule": "cosine",
                  "beta_start": 0.0001, "beta_end": 0.02},
    "inference": {"tile_size": 256, "overlap": 32},
}


def effective_config(config):
    result = copy.deepcopy(DEFAULTS)
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key].update(copy.deepcopy(value))
        else:
            result[key] = copy.deepcopy(value)
    if not isinstance(result["data"]["scale"], (int, float)):
        raise ValueError("Resolve data.scale: auto using the CLI before training, or provide a positive numeric scale")
    if not math.isfinite(result["data"]["scale"]) or result["data"]["scale"] <= 0:
        raise ValueError("data.scale must be finite and positive")
    if not math.isfinite(result["data"]["offset"]) or result["data"]["offset"] < 0:
        raise ValueError("data.offset must be finite and nonnegative; this workflow supports raw camera ADU, not negative background-subtracted intensities")
    if int(result["data"]["patch_size"]) < 8:
        raise ValueError("data.patch_size must be at least 8")
    for key in ("epochs", "steps_per_epoch", "batch_size", "val_steps", "num_threads"):
        if int(result["train"][key]) < 1:
            raise ValueError("train.%s must be positive" % key)
    if not 0 < float(result["train"]["mask_fraction"]) < 1:
        raise ValueError("train.mask_fraction must lie strictly between 0 and 1")
    return result


def select_device(device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or install a suitable PyTorch build")
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are cpu, cuda and auto")
    return selected


def mask_pixels(batch, fraction=0.02, generator=None):
    """Hide pixels using NON-masked neighbors and return a loss mask.

    Source values are always read from unmasked locations, so an original
    masked target is never copied back into another location in the input.
    """
    if batch.ndim != 4 or batch.shape[1] != 1 or not 0 < fraction < 1:
        raise ValueError("Expected Bx1xHxW tensor and 0 < fraction < 1")
    b, _, h, w = batch.shape
    count = min(h * w - 1, max(1, int(round(h * w * fraction))))
    if count < 1:
        raise ValueError("At least two pixels are required")
    output = batch.clone()
    masks = torch.zeros_like(batch, dtype=torch.bool)
    for index in range(b):
        chosen = torch.randperm(h * w, device=batch.device, generator=generator)[:count]
        mask = masks[index, 0].view(-1)
        mask[chosen] = True
        y, x = chosen // w, chosen % w
        sources = torch.empty_like(chosen)
        pending = torch.ones(count, dtype=torch.bool, device=batch.device)
        for _ in range(16):
            offsets = torch.randint(-2, 3, (count, 2), device=batch.device, generator=generator)
            sy, sx = y + offsets[:, 0], x + offsets[:, 1]
            valid = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
            flat = sy.clamp(0, h - 1) * w + sx.clamp(0, w - 1)
            valid &= ~mask[flat]
            take = pending & valid
            sources[take] = flat[take]
            pending &= ~take
            if not pending.any():
                break
        if pending.any():
            available = torch.nonzero(~mask, as_tuple=False).flatten()
            fallback = torch.randint(len(available), (int(pending.sum().item()),), device=batch.device, generator=generator)
            sources[pending] = available[fallback]
        output[index, 0].view(-1)[chosen] = batch[index, 0].reshape(-1)[sources]
    return output, masks


def choose_repeat_paths(paths, rng, max_target_frames=8):
    """Select one input and >=2 OTHER frames; no input/target frame overlap."""
    paths = [Path(p) for p in paths]
    if len(paths) < 3:
        raise ValueError("Diffusion requires at least 3 static repeated frames per group")
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Repeat group contains a duplicate path")
    if int(max_target_frames) < 2:
        raise ValueError("data.max_target_frames must be at least 2")
    order = rng.permutation(len(paths))
    target_count = min(int(max_target_frames), len(paths) - 1)
    return paths[int(order[0])], [paths[int(i)] for i in order[1:target_count + 1]]


def make_repeat_pair(paths, rng=None, max_target_frames=8):
    rng = np.random.default_rng() if rng is None else rng
    input_path, target_paths = choose_repeat_paths(paths, rng, max_target_frames)
    source = load_image(input_path)
    target = np.zeros_like(source, dtype=np.float32)
    for path in target_paths:
        frame = load_image(path)
        if frame.shape != source.shape:
            raise ValueError("All repeat frames must have identical dimensions")
        target += frame / len(target_paths)
    return source, target, {"input_path": str(input_path), "target_paths": [str(p) for p in target_paths]}


class PatchSampler:
    """Sample on demand with a bounded cache, never load the whole dataset."""
    def __init__(self, root, method, config):
        self.root, self.method, self.config = Path(root), method, config
        self.paths = list_images(root)
        if not self.paths:
            raise ValueError("No training images found in %s" % root)
        self.groups = list(find_groups(root).values())
        if method == "diffusion" and any(len(group) < 3 for group in self.groups):
            raise ValueError("Every diffusion group must contain at least 3 static repeated frames")
        self.cache = OrderedDict()

    def read(self, path):
        key = str(path)
        if key not in self.cache:
            self.cache[key] = load_image(path)
            if len(self.cache) > 16:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def batch(self, rng, count):
        inputs, targets = [], []
        data = self.config["data"]
        size, offset, scale = int(data["patch_size"]), data["offset"], data["scale"]
        for _ in range(count):
            if self.method == "diffusion":
                group = self.groups[int(rng.integers(len(self.groups)))]
                source_path, target_paths = choose_repeat_paths(group, rng, data["max_target_frames"])
            else:
                source_path = self.paths[int(rng.integers(len(self.paths)))]
                target_paths = []
            source = self.read(source_path)
            if min(source.shape) < size:
                raise ValueError("Image %s is smaller than patch_size=%d; reduce patch_size (no reflection padding during training)" % (source_path, size))
            y = int(rng.integers(source.shape[0] - size + 1))
            x = int(rng.integers(source.shape[1] - size + 1))
            crop = source[y:y + size, x:x + size]
            if target_paths:
                target = np.zeros((size, size), dtype=np.float32)
                for path in target_paths:
                    frame = self.read(path)
                    if frame.shape != source.shape:
                        raise ValueError("Repeated frames differ in shape: %s" % path)
                    target += frame[y:y + size, x:x + size] / len(target_paths)
            else:
                target = crop.copy()
            # Same normalization and geometric transform for input and target.
            source_n = np.clip((crop - offset) / scale, 0, 1)
            target_n = np.clip((target - offset) / scale, 0, 1)
            turns = int(rng.integers(4))
            source_n, target_n = np.rot90(source_n, turns), np.rot90(target_n, turns)
            if rng.random() < 0.5:
                source_n, target_n = source_n[:, ::-1], target_n[:, ::-1]
            inputs.append(np.ascontiguousarray(source_n, dtype=np.float32))
            targets.append(np.ascontiguousarray(target_n, dtype=np.float32))
        return torch.from_numpy(np.stack(inputs)[:, None]), torch.from_numpy(np.stack(targets)[:, None])


def _generator(device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _batch_loss(model, source, target, method, config, abar, generator):
    if method == "masked":
        hidden, mask = mask_pixels(source, config["train"]["mask_fraction"], generator)
        return F.mse_loss(model(hidden)[mask], target[mask])
    target = target * 2 - 1
    condition = source * 2 - 1
    t = torch.randint(len(abar), (source.shape[0],), device=source.device, generator=generator)
    noise = torch.randn(target.shape, device=target.device, generator=generator)
    alpha = abar[t, None, None, None]
    noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
    prediction = model(torch.cat((noisy, condition), dim=1), t)
    return F.mse_loss(prediction, noise)


def _load_checkpoint(path):
    # Checkpoints are executable pickle objects on old PyTorch: load trusted files only.
    return torch.load(str(path), map_location="cpu")


def _atomic_save(payload, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def train_model(method, train_root, val_root, config, output_dir, device="auto", resume=None):
    """Train and return best.pt. Validation metrics are proxy losses, not accuracy."""
    if method not in ("masked", "diffusion"):
        raise ValueError("method must be masked or diffusion")
    from ..validation import validate_splits
    validation_report = validate_splits(train_root, val_root, method=method)
    config = copy.deepcopy(config)
    if config.get("data", {}).get("scale") == "auto":
        offset = float(config["data"].get("offset", 0.0))
        config["data"]["scale"] = max(float(validation_report["train"]["max"]) - offset, 1.0)
    config = effective_config(config)
    for split in ("train", "val"):
        if any(min(shape) < int(config["data"]["patch_size"]) for shape in validation_report[split]["shapes"]):
            raise ValueError("%s images are smaller than data.patch_size; reduce patch_size (minimum 8)" % split)
    output_dir = Path(output_dir)
    if resume is None and any((output_dir / name).exists() for name in ("last.pt", "best.pt", "history.json")):
        raise FileExistsError("Training output already exists; choose a new output directory or --resume")
    selected = select_device(device)
    torch.set_num_threads(int(config["train"]["num_threads"]))
    seed = int(config["train"]["seed"])
    torch.manual_seed(seed)
    if selected.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = SmallUNet(config["model"]["base_channels"], diffusion=method == "diffusion").to(selected)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))
    abar = diffusion_schedule(**{key: config["diffusion"][key] for key in ("timesteps", "schedule", "beta_start", "beta_end")})[2].to(selected) if method == "diffusion" else None
    history, start, best = [], 0, float("inf")
    if resume is not None:
        if Path(resume).resolve().parent != output_dir.resolve():
            raise ValueError("Resume in the original output directory so its best checkpoint and history are preserved")
        saved = _load_checkpoint(resume)
        if saved["method"] != method:
            raise ValueError("Resume checkpoint method does not match")
        for section in ("data", "model", "diffusion"):
            if saved["config"][section] != config[section]:
                raise ValueError("Resume config differs in %s; retain original settings" % section)
        if any(saved["config"]["train"][key] != config["train"][key] for key in saved["config"]["train"] if key != "epochs"):
            raise ValueError("Only train.epochs may change when resuming a run")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        history, start = saved["history"], int(saved["epoch"])
        best = min((row["val_proxy_loss"] for row in history), default=float("inf"))
    epochs = int(config["train"]["epochs"])
    if epochs <= start:
        raise ValueError("train.epochs must exceed the resumed checkpoint epoch")
    train_data = PatchSampler(train_root, method, config)
    val_data = PatchSampler(val_root, method, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "data_validation.json").write_text(json.dumps(validation_report, indent=2, ensure_ascii=False), encoding="utf-8")
    batch_size = int(config["train"]["batch_size"])
    for epoch in range(start, epochs):
        rng = np.random.default_rng(seed + epoch * 1009)
        generator = _generator(selected, seed + epoch * 1013)
        model.train()
        losses = []
        for _ in range(int(config["train"]["steps_per_epoch"])):
            source, target = train_data.batch(rng, batch_size)
            source, target = source.to(selected), target.to(selected)
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(model, source, target, method, config, abar, generator)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; inspect normalization and learning rate")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        # Fixed validation samples/noise each epoch make the proxy curve comparable.
        rng = np.random.default_rng(seed + 1000003)
        generator = _generator(selected, seed + 1000033)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(int(config["train"]["val_steps"])):
                source, target = val_data.batch(rng, batch_size)
                loss = _batch_loss(model, source.to(selected), target.to(selected), method, config, abar, generator)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite validation loss; inspect normalization and learning rate")
                val_losses.append(float(loss.cpu()))
        row = {"epoch": epoch + 1, "train_proxy_loss": float(np.mean(losses)), "val_proxy_loss": float(np.mean(val_losses))}
        history.append(row)
        improved = row["val_proxy_loss"] < best
        best = min(best, row["val_proxy_loss"])
        payload = {"format_version": 1, "method": method, "config": config,
                   "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                   "epoch": epoch + 1, "history": history,
                   "target_description": "masked noisy pixels" if method == "masked" else "independent repeat-frame mean; residual-noise proxy, not clean ground truth"}
        _atomic_save(payload, output_dir / "last.pt")
        if improved or not (output_dir / "best.pt").exists():
            _atomic_save(payload, output_dir / "best.pt")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print("[%s] epoch %d/%d train_proxy_loss=%.6g val_proxy_loss=%.6g" % (method, epoch + 1, epochs, row["train_proxy_loss"], row["val_proxy_loss"]), flush=True)
    return output_dir / "best.pt"


class Restorer:
    def __init__(self, checkpoint_path, device="auto"):
        saved = _load_checkpoint(checkpoint_path)
        if saved.get("format_version") != 1 or saved.get("method") not in ("masked", "diffusion"):
            raise ValueError("Unsupported checkpoint")
        self.config, self.method = effective_config(saved["config"]), saved["method"]
        self.device = select_device(device)
        torch.set_num_threads(int(self.config["train"]["num_threads"]))
        self.model = SmallUNet(self.config["model"]["base_channels"], self.method == "diffusion").to(self.device)
        self.model.load_state_dict(saved["model_state"])
        self.model.eval()
        self.abar = diffusion_schedule(**{key: self.config["diffusion"][key] for key in ("timesteps", "schedule", "beta_start", "beta_end")})[2].to(self.device) if self.method == "diffusion" else None

    @torch.no_grad()
    def restore(self, image, seed=0, sample_steps=None):
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 2 or min(image.shape) < 4 or not np.isfinite(image).all():
            raise ValueError("Expected a finite 2D image of at least 4 x 4 pixels")
        offset, scale = self.config["data"]["offset"], self.config["data"]["scale"]
        normalized = np.clip((image - offset) / scale, 0, 1)
        source = torch.from_numpy(np.ascontiguousarray(normalized))[None, None].to(self.device)
        tiling = self.config["inference"]
        predict = lambda x, t=None: predict_tiled(self.model, x, t, tiling["tile_size"], tiling["overlap"])
        if self.method == "masked":
            output = predict(source).clamp(0, 1)
        else:
            steps = int(self.config["diffusion"]["sampling_steps"] if sample_steps is None else sample_steps)
            if steps < 1:
                raise ValueError("sampling_steps must be positive")
            steps = min(steps, len(self.abar))
            times = np.linspace(len(self.abar) - 1, 0, steps, dtype=int).tolist()
            generator = _generator(self.device, seed)
            sample = torch.randn(source.shape, device=self.device, generator=generator)
            condition = source * 2 - 1
            for index, time in enumerate(times):
                t = torch.full((1,), time, device=self.device, dtype=torch.long)
                epsilon = predict(torch.cat((sample, condition), dim=1), t)
                alpha = self.abar[time]
                x0 = ((sample - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()).clamp(-1, 1)
                previous = times[index + 1] if index + 1 < len(times) else -1
                if previous < 0:
                    sample = x0
                else:
                    # Recompute epsilon after clipping x0 so the DDIM equation stays consistent.
                    epsilon = (sample - alpha.sqrt() * x0) / (1 - alpha).sqrt()
                    previous_alpha = self.abar[previous]
                    sample = previous_alpha.sqrt() * x0 + (1 - previous_alpha).sqrt() * epsilon
            output = ((sample + 1) / 2).clamp(0, 1)
        result = output[0, 0].cpu().numpy() * scale + offset
        if not np.isfinite(result).all():
            raise RuntimeError("Restoration produced non-finite values")
        return result.astype(np.float32)


def load_restorer(checkpoint_path, device="auto"):
    return Restorer(checkpoint_path, device)


def restore_image(image, checkpoint_path, device="auto", seed=0, sample_steps=None):
    return load_restorer(checkpoint_path, device).restore(image, seed, sample_steps)
