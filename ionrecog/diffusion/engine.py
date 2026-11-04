"""Server training and DDIM inference for the pixel-space diffusion baseline.

The loss is noise-prediction MSE, not localization accuracy. Repeated-frame
targets retain motion blur. Trusted local checkpoints only: torch.load can
deserialize executable pickle data. Resume is exact on the tested CPU setup;
CUDA numerics can vary between devices, drivers and PyTorch versions.
"""
import contextlib
import copy
import hashlib
import inspect
import json
import math
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .model import build_model, alpha_bar, predict_tiled


FORMAT = "ionrecog.diffusion.v1"
DEFAULTS = {
    "data": {"patch_size": 64, "offset": 0.0, "scale": 65535.0, "max_target_frames": 8},
    "model": {"base_channels": 32},
    "diffusion": {"timesteps": 100, "sampling_steps": 20},
    "train": {"steps": 20000, "batch_size": 4, "grad_accum": 1, "lr": 0.0002,
              "seed": 42, "validate_every": 500, "val_batches": 10,
              "save_every": 500, "ema_decay": 0.999, "precision": "fp32",
              "num_threads": 4, "grad_clip": 1.0},
    "inference": {"tile_size": 256, "overlap": 32},
}


def effective_config(config):
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    result = copy.deepcopy(DEFAULTS)
    # Extra top-level settings (for example detection) are preserved in checkpoints.
    for key, value in config.items():
        if key in DEFAULTS:
            if not isinstance(value, dict):
                raise ValueError("config.%s must be a mapping" % key)
            unknown = set(value) - set(DEFAULTS[key])
            if unknown:
                raise ValueError("Unknown %s settings: %s" % (key, sorted(unknown)))
            result[key].update(value)
        else:
            result[key] = copy.deepcopy(value)
    for section, keys in {
        "data": ["patch_size", "max_target_frames"], "model": ["base_channels"],
        "diffusion": ["timesteps", "sampling_steps"],
        "train": ["steps", "batch_size", "grad_accum", "validate_every", "val_batches", "save_every", "num_threads"],
    }.items():
        for key in keys:
            value = result[section][key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("%s.%s must be a positive integer" % (section, key))
    if result["data"]["patch_size"] < 8 or result["data"]["max_target_frames"] < 2:
        raise ValueError("patch_size >= 8 and max_target_frames >= 2 required")
    if result["model"]["base_channels"] < 4 or result["diffusion"]["timesteps"] < 2:
        raise ValueError("base_channels >= 4 and timesteps >= 2 required")
    if result["diffusion"]["sampling_steps"] > result["diffusion"]["timesteps"]:
        raise ValueError("sampling_steps must not exceed timesteps")
    for section, key in [("data", "offset"), ("data", "scale"), ("train", "lr"), ("train", "grad_clip"), ("train", "ema_decay")]:
        try:
            value = float(result[section][key])
        except (ValueError, TypeError):
            raise ValueError("%s.%s must be finite numeric" % (section, key))
        if not math.isfinite(value):
            raise ValueError("%s.%s must be finite numeric" % (section, key))
        result[section][key] = value
    if result["data"]["offset"] < 0 or result["data"]["scale"] <= 0 or result["train"]["lr"] <= 0 or result["train"]["grad_clip"] <= 0:
        raise ValueError("offset must be nonnegative; scale, lr and grad_clip must be positive")
    if not 0 <= result["train"]["ema_decay"] < 1:
        raise ValueError("ema_decay must be in [0, 1)")
    if result["train"]["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16 or bf16")
    seed = result["train"]["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("train.seed must be an integer in [0, 2**31)")
    _validate_tiling(result["inference"]["tile_size"], result["inference"]["overlap"])
    return result


def _validate_tiling(tile_size, overlap):
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (tile_size, overlap)):
        raise ValueError("tile_size and overlap must be integers")
    if tile_size < 0 or overlap < 0 or (tile_size > 0 and (tile_size < 16 or overlap >= tile_size // 2)):
        raise ValueError("tile_size=0 disables tiling; otherwise tile_size>=16 and 0<=overlap<tile_size/2")


def _json_write(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(str(temporary), str(path))


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def load_checkpoint(path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    saved = torch.load(str(path), **kwargs)
    if not isinstance(saved, dict) or saved.get("format") != FORMAT:
        raise ValueError("Not an ionrecog.diffusion.v1 checkpoint")
    return saved


def _save_checkpoint(saved, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(_cpu_tree(saved), str(temporary))
    os.replace(str(temporary), str(path))


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint(manifest, records):
    h = hashlib.sha256(Path(manifest).read_bytes())
    cache = {}
    for record in records:
        for key in ("input", "target", "frames"):
            paths = record.get(key)
            if not paths:
                continue
            if not isinstance(paths, list):
                paths = [paths]
            for path in paths:
                path = str(Path(path).resolve())
                if path not in cache:
                    cache[path] = _sha(path)
                h.update(cache[path].encode("ascii"))
    return h.hexdigest()


def _validate_output(output, manifest, records):
    images = []
    for record in records:
        images.extend(record.get("frames", []) or [record["input"], record["target"]])
    source_files = [Path(manifest).resolve()] + [Path(p).resolve() for p in images]
    if any(output == path or output in path.parents for path in source_files):
        raise ValueError("Output must not be an input file or an ancestor of source data")
    parents = {p.parent for p in source_files[1:]}
    try:
        common = Path(os.path.commonpath([str(p) for p in parents]))
        if common != Path(common.anchor):
            parents.add(common)
    except ValueError:  # Source files can live on different Windows drives.
        pass
    if any(output == parent or parent in output.parents for parent in parents):
        raise ValueError("Output must be outside source image directories")
    if output.exists() and not output.is_dir():
        raise ValueError("Output must be a directory")


def _device(name, local_rank=0, distributed=False):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    selected = torch.device(name)
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA devices are supported")
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; use --device cpu or install a CUDA PyTorch build")
        selected = torch.device("cuda", local_rank if distributed else (selected.index or 0))
        torch.cuda.set_device(selected)
    return selected


def _autocast(device, precision):
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _generator(device, seed):
    return torch.Generator(device=device).manual_seed(int(seed) % (2**63 - 1))


def _loss(model, batch, abar, device, seed, precision):
    source = batch["observed"].to(device=device, dtype=torch.float32) * 2 - 1
    target = batch["target"].to(device=device, dtype=torch.float32) * 2 - 1
    if source.shape != target.shape or source.ndim != 4 or source.shape[1] != 1:
        raise ValueError("Expected matching Bx1xHxW observed and target")
    generator = _generator(device, seed)
    t = torch.randint(len(abar), (len(source),), device=device, generator=generator)
    noise = torch.randn(target.shape, device=device, generator=generator)
    a = abar[t, None, None, None]
    noisy = a.sqrt() * target + (1 - a).sqrt() * noise
    with _autocast(device, precision):
        epsilon = model(torch.cat((noisy, source), dim=1), t)
    return (epsilon.float() - noise).square().mean()


def _finite_all(tensor, distributed):
    flag = torch.isfinite(tensor).all().to(dtype=torch.int32)
    if distributed:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _rank_zero(action, rank, distributed):
    """Propagate rank-zero I/O errors before any worker enters the next step."""
    if not distributed:
        return action()
    error = [None]
    if rank == 0:
        try:
            action()
        except Exception as exc:
            error[0] = "%s: %s" % (type(exc).__name__, exc)
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError("Rank-zero operation failed: " + error[0])


def train(config, manifest, output, device="auto", resume=None, init_checkpoint=None):
    """Train epsilon predictor; one step includes grad_accum microbatches per rank.

    All ranks must see the same manifest, image files and shared output directory.
    Resume from last.pt, with the same world size and only train.steps increased.
    Per-step seeds remove dependence on restored worker/global RNG state.
    """
    from .data import audit_manifest, load_manifest, PatchSampler

    config = effective_config(config)
    if resume is not None and init_checkpoint is not None:
        raise ValueError("resume and init_checkpoint are mutually exclusive")
    report = audit_manifest(manifest)
    records = load_manifest(manifest)
    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "val"]
    if not train_records or not val_records:
        raise ValueError("Training needs nonempty train and val splits")
    fingerprint = _fingerprint(manifest, records)
    output = Path(output).resolve()
    _validate_output(output, manifest, records)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    selected = _device(device, local_rank, distributed)
    precision = config["train"]["precision"]
    if selected.type == "cpu" and precision == "fp16":
        raise ValueError("CPU fp16 training is unsupported; use fp32 or bf16")
    if selected.type == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Selected CUDA device does not support bf16")
    torch.set_num_threads(config["train"]["num_threads"])
    torch.manual_seed(config["train"]["seed"])
    if selected.type == "cuda":
        torch.cuda.manual_seed_all(config["train"]["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    saved = None
    if resume is not None:
        if Path(resume).resolve() != output / "last.pt":
            raise ValueError("Resume must use last.pt in the original output directory")
        saved = load_checkpoint(resume)
        previous = copy.deepcopy(saved["config"])
        requested = copy.deepcopy(config)
        previous["train"].pop("steps")
        requested["train"].pop("steps")
        if requested != previous or saved["world_size"] != world_size:
            raise ValueError("Resume requires identical configuration/world_size except train.steps")
        if saved["data_fingerprint"] != fingerprint:
            raise ValueError("Manifest or source image content changed since checkpoint")
        if config["train"]["steps"] <= saved["step"]:
            raise ValueError("train.steps must exceed the saved step")
        if config["train"]["steps"] < saved["config"]["train"]["steps"]:
            raise ValueError("Resume may extend, not decrease, the planned train.steps")
        if math.isfinite(saved["best_val_loss"]) and not (output / "best.pt").is_file():
            raise FileNotFoundError("Original best.pt is missing; cannot resume preserving best model")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new run or resume last.pt")
    # All preflight checks occur before creating a process group, so every worker
    # sees a useful error without waiting at a barrier for a failed rank zero.
    owns_group = distributed and not dist.is_initialized()
    if owns_group:
        dist.init_process_group(backend="nccl" if selected.type == "cuda" else "gloo")
    try:
        _rank_zero(lambda: output.mkdir(parents=True, exist_ok=True), rank, distributed)
        data = config["data"]
        kwargs = {key: data[key] for key in ("patch_size", "offset", "scale", "max_target_frames")}
        train_data = PatchSampler(train_records, augment=True, **kwargs)
        val_data = PatchSampler(val_records, augment=False, **kwargs)
        model = build_model(config).to(selected)
        if init_checkpoint is not None:
            initial = load_checkpoint(init_checkpoint)
            for section in ("model", "diffusion"):
                if initial["config"][section] != config[section]:
                    raise ValueError("init_checkpoint incompatible %s configuration" % section)
            for key in ("scale", "offset"):
                if initial["config"]["data"][key] != data[key]:
                    raise ValueError("init_checkpoint must retain scale and offset")
            model.load_state_dict(initial["ema_state"])
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"], weight_decay=0.0)
        scaler = torch.cuda.amp.GradScaler(enabled=selected.type == "cuda" and precision == "fp16")
        history, start, best, optimizer_updates = [], 0, float("inf"), 0
        if saved is not None:
            model.load_state_dict(saved["model_state"])
            ema.load_state_dict(saved["ema_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            scaler.load_state_dict(saved["scaler_state"])
            history, start, best = saved["history"], saved["step"], saved["best_val_loss"]
            optimizer_updates = saved.get("optimizer_updates", start)
        network = DistributedDataParallel(model, device_ids=[selected.index] if selected.type == "cuda" else None) if distributed else model
        abar = alpha_bar(config).to(selected)
        def write_provenance():
            _json_write(config, output / "config.json")
            _json_write(report, output / "data_audit.json")
            _json_write({"format": FORMAT, "manifest_sha256": _sha(manifest), "data_fingerprint": fingerprint,
                         "world_size": world_size, "device": str(selected), "torch": torch.__version__,
                         "python": platform.python_version(), "platform": platform.platform(),
                         "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
                         "resume": str(resume) if resume else None,
                         "target_warning": "Repeated-frame targets are low-noise proxies, not deblurred truth",
                         "seed_strategy": "seed + step/rank/microbatch arithmetic; validation seeds fixed",
                         "effective_batch_size": world_size * config["train"]["batch_size"] * config["train"]["grad_accum"]}, output / "provenance.json")
        _rank_zero(write_provenance, rank, distributed)
        settings = config["train"]
        for step in range(start + 1, settings["steps"] + 1):
            begin = time.perf_counter()
            network.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=selected)
            train_clipped = torch.zeros(2, device=selected)
            for micro in range(settings["grad_accum"]):
                key = (step - 1) * world_size * settings["grad_accum"] + rank * settings["grad_accum"] + micro
                seed = settings["seed"] + key * 1009
                batch = train_data.batch(seed=seed, batch_size=settings["batch_size"])
                fractions = batch.get("clipped_fraction", {})
                train_clipped += torch.tensor([fractions.get("observed", 0.0), fractions.get("target", 0.0)], device=selected) / settings["grad_accum"]
                sync_context = network.no_sync() if distributed and micro + 1 < settings["grad_accum"] else contextlib.nullcontext()
                with sync_context:
                    loss = _loss(network, batch, abar, selected, seed + 503, precision)
                    if not _finite_all(loss, distributed):
                        raise RuntimeError("Non-finite training loss; inspect input scaling")
                    scaler.scale(loss / settings["grad_accum"]).backward()
                loss_sum += loss.detach() / settings["grad_accum"]
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
            finite_gradient = _finite_all(grad_norm, distributed)
            if not finite_gradient and not scaler.is_enabled():
                raise RuntimeError("Non-finite gradient; reduce learning rate or use bf16/fp32")
            # FP16 overflow is handled by GradScaler: skip this optimizer update
            # and reduce scale. Do not move EMA towards unchanged weights.
            scaler.step(optimizer)
            scaler.update()
            if finite_gradient:
                optimizer_updates += 1
                with torch.no_grad():
                    for averaged, parameter in zip(ema.parameters(), model.parameters()):
                        averaged.lerp_(parameter.detach(), 1.0 - settings["ema_decay"])
                    for averaged, buffer in zip(ema.buffers(), model.buffers()):
                        averaged.copy_(buffer)
            if distributed:
                dist.all_reduce(loss_sum)
                loss_sum /= world_size
                dist.all_reduce(train_clipped)
                train_clipped /= world_size
            row = {"step": step, "train_noise_mse": float(loss_sum.cpu()),
                   "optimizer_updates": optimizer_updates, "fp16_update_skipped": not finite_gradient,
                   "grad_scale": scaler.get_scale(),
                   "train_clipped_fraction": dict(zip(("observed", "target"), train_clipped.cpu().tolist())),
                   "seconds": time.perf_counter() - begin}
            improved = False
            if step % settings["validate_every"] == 0 or step == settings["steps"]:
                val_loss = torch.zeros((), device=selected)
                val_clipped = torch.zeros(2, device=selected)
                with torch.no_grad():
                    for index in range(settings["val_batches"]):
                        seed = settings["seed"] + 1000000007 + rank * settings["val_batches"] + index
                        batch = val_data.batch(seed=seed, batch_size=settings["batch_size"])
                        fractions = batch.get("clipped_fraction", {})
                        val_clipped += torch.tensor([fractions.get("observed", 0.0), fractions.get("target", 0.0)], device=selected) / settings["val_batches"]
                        loss = _loss(ema, batch, abar, selected, seed + 200003, precision)
                        if not _finite_all(loss, distributed):
                            raise RuntimeError("Non-finite validation loss")
                        val_loss += loss / settings["val_batches"]
                if distributed:
                    dist.all_reduce(val_loss)
                    val_loss /= world_size
                    dist.all_reduce(val_clipped)
                    val_clipped /= world_size
                row["val_ema_noise_mse"] = float(val_loss.cpu())
                row["val_clipped_fraction"] = dict(zip(("observed", "target"), val_clipped.cpu().tolist()))
                improved = row["val_ema_noise_mse"] < best
                best = min(best, row["val_ema_noise_mse"])
            history.append(row)
            def write_step():
                if step == 1 or step % min(settings["validate_every"], 50) == 0 or step == settings["steps"]:
                    print(json.dumps(row), flush=True)
                save = improved or step % settings["save_every"] == 0 or step == settings["steps"]
                if save:
                    # Keep a reserve and enough temporary space for model + EMA +
                    # optimizer states. Files are replaced atomically, never deleted.
                    estimate = sum(p.numel() for p in model.parameters()) * 40
                    if shutil.disk_usage(output).free < estimate + 2 * 1024**3:
                        raise OSError("Insufficient disk space for atomic checkpoint and reserve")
                    payload = {"format": FORMAT, "config": config, "step": step, "world_size": world_size,
                               "optimizer_updates": optimizer_updates,
                               "model_state": model.state_dict(), "ema_state": ema.state_dict(),
                               "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(),
                               "history": history, "best_val_loss": best, "data_fingerprint": fingerprint}
                    _save_checkpoint(payload, output / "last.pt")
                    if improved:
                        _save_checkpoint(payload, output / "best.pt")
                    _json_write(history, output / "history.json")
            _rank_zero(write_step, rank, distributed)
        return output / "best.pt"
    finally:
        if owns_group:
            dist.destroy_process_group()


class Restorer:
    """EMA or raw predictor with globally consistent tiled deterministic DDIM."""
    def __init__(self, checkpoint, device="auto", weights="ema"):
        if weights not in {"ema", "raw"}:
            raise ValueError("weights must be ema or raw")
        saved = load_checkpoint(checkpoint)
        self.config = effective_config(saved["config"])
        self.device = _device(device)
        torch.set_num_threads(self.config["train"]["num_threads"])
        self.model = build_model(self.config).to(self.device).eval()
        self.model.load_state_dict(saved["ema_state" if weights == "ema" else "model_state"])
        self.abar = alpha_bar(self.config).to(self.device)
        self.step = saved["step"]
        self.weights = weights

    @torch.no_grad()
    def restore(self, image, seed=0, sampling_steps=None, tile_size=None, overlap=None):
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 2 or min(image.shape) < 4 or not np.isfinite(image).all():
            raise ValueError("Expected a finite 2D image at least 4x4")
        sampling_steps = self.config["diffusion"]["sampling_steps"] if sampling_steps is None else sampling_steps
        if isinstance(sampling_steps, bool) or not isinstance(sampling_steps, int) or not 1 <= sampling_steps <= len(self.abar):
            raise ValueError("sampling_steps must be an integer between 1 and timesteps")
        tile_size = self.config["inference"]["tile_size"] if tile_size is None else tile_size
        overlap = self.config["inference"]["overlap"] if overlap is None else overlap
        _validate_tiling(tile_size, overlap)
        offset, scale = self.config["data"]["offset"], self.config["data"]["scale"]
        normalized = np.clip((image - offset) / scale, 0, 1)
        source = torch.from_numpy(np.ascontiguousarray(normalized))[None, None].to(self.device)
        sample = torch.randn(source.shape, device=self.device, generator=_generator(self.device, seed))
        condition = source * 2 - 1
        times = np.linspace(len(self.abar) - 1, 0, sampling_steps, dtype=int).tolist()
        for index, time_index in enumerate(times):
            t = torch.full((1,), time_index, device=self.device, dtype=torch.long)
            epsilon = predict_tiled(self.model, torch.cat((sample, condition), dim=1), t, tile_size, overlap)
            a = self.abar[time_index]
            x0 = ((sample - (1 - a).sqrt() * epsilon) / a.sqrt()).clamp(-1, 1)
            if index + 1 == len(times):
                sample = x0
            else:
                epsilon = (sample - a.sqrt() * x0) / (1 - a).sqrt()
                a_previous = self.abar[times[index + 1]]
                sample = a_previous.sqrt() * x0 + (1 - a_previous).sqrt() * epsilon
        result = ((sample[0, 0] + 1) / 2).clamp(0, 1).cpu().numpy() * scale + offset
        if not np.isfinite(result).all():
            raise RuntimeError("Non-finite restoration output")
        return result.astype(np.float32)
