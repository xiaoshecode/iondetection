"""Strict, versioned experiment settings; fixed camera intensity scale."""
import copy
import math
from pathlib import Path

import yaml

from ..classical.detection import DEFAULT_DETECTION
from ..classical.region import DEFAULT_REGION


DEFAULTS = {
    "data": {"patch_size": 64, "offset": 0.0, "scale": 65535.0, "max_target_frames": 8},
    "model": {"base_channels": 32},
    "diffusion": {"timesteps": 100, "sampling_steps": 20},
    "train": {"steps": 20000, "batch_size": 4, "grad_accum": 1, "lr": 0.0002,
              "seed": 42, "validate_every": 500, "val_batches": 10, "save_every": 500,
              "ema_decay": 0.999, "precision": "fp32", "num_threads": 4, "grad_clip": 1.0},
    "inference": {"tile_size": 256, "overlap": 32},
    "detection": dict(DEFAULT_DETECTION),
    "region": dict(DEFAULT_REGION),
}


def load_config(path=None):
    config = copy.deepcopy(DEFAULTS)
    supplied = {} if path is None else yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(supplied, dict):
        raise ValueError("Configuration must be a mapping")
    for section, values in supplied.items():
        if section not in config or not isinstance(values, dict):
            raise ValueError("Unknown/invalid configuration section: " + section)
        unknown = set(values) - set(config[section])
        if unknown:
            raise ValueError("Unknown settings in %s: %s" % (section, sorted(unknown)))
        config[section].update(values)
    for section, keys in {
        "data": ("patch_size", "max_target_frames"), "model": ("base_channels",),
        "diffusion": ("timesteps", "sampling_steps"),
        "train": ("steps", "batch_size", "grad_accum", "validate_every", "val_batches", "save_every", "num_threads"),
    }.items():
        for key in keys:
            value = config[section][key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("%s.%s must be a positive integer" % (section, key))
    for section, key in (("data", "scale"), ("train", "lr"), ("train", "grad_clip")):
        if not math.isfinite(config[section][key]) or config[section][key] <= 0:
            raise ValueError("%s.%s must be finite and positive" % (section, key))
    if not math.isfinite(config["data"]["offset"]) or config["data"]["offset"] < 0:
        raise ValueError("data.offset must be finite and nonnegative")
    if config["data"]["patch_size"] < 8 or config["data"]["max_target_frames"] < 2:
        raise ValueError("patch_size >= 8 and max_target_frames >= 2 required")
    if config["model"]["base_channels"] < 4 or config["diffusion"]["timesteps"] < 2:
        raise ValueError("base_channels >= 4 and timesteps >= 2 required")
    if config["diffusion"]["sampling_steps"] > config["diffusion"]["timesteps"]:
        raise ValueError("sampling_steps cannot exceed timesteps")
    if config["train"]["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16 or bf16")
    if not 0 <= config["train"]["ema_decay"] < 1:
        raise ValueError("ema_decay must be in [0, 1)")
    if isinstance(config["train"]["seed"], bool) or not isinstance(config["train"]["seed"], int) or config["train"]["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    tile, overlap = config["inference"]["tile_size"], config["inference"]["overlap"]
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (tile, overlap)):
        raise ValueError("tile_size and overlap must be integers")
    if tile < 0 or overlap < 0 or (tile and (tile < 16 or overlap >= tile // 2)):
        raise ValueError("Use tile_size=0 (whole image) or >=16, with 0<=overlap<tile_size/2")
    if config["region"]["mode"] not in ("off", "auto"):
        raise ValueError('region.mode must be "off" (quoted in YAML) or auto')
    return config
