"""Small, explicit physics-approximation fixtures for training and evaluation.

Stationary target = optical PSF at known centers, without motion or detector
noise. Observation = exposure-averaged sinusoidal motion + Poisson electrons +
optional Gamma EM multiplication + Gaussian read noise. This is NOT calibrated
experimental EMCCD data, an exact ion dynamics simulation, or a detector model
including CIC, saturation, charge transfer, aberrations and laser modulation.
"""

import json
import shutil
from pathlib import Path

import numpy as np


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return int(value)


def sample_camera(expected, rng, em_gain=1.0, read_noise=1.5, offset=10.0):
    """Draw effective input-referred ADU, including stochastic EM when gain>1.

    For ideal high-gain multiplication conditioned on n input electrons,
    output charge is Gamma(shape=n, scale=G). Divide by G to keep the mean scale
    invariant across modes; read_noise is specified in output electrons before
    this conversion. n=0 produces exactly zero multiplied signal.
    """
    expected = np.asarray(expected, dtype=np.float64)
    if expected.ndim != 2 or not np.isfinite(expected).all() or np.any(expected < 0):
        raise ValueError("expected must be a finite nonnegative 2D electron-rate image")
    if not np.isfinite(em_gain) or em_gain < 1 or not np.isfinite(read_noise) or read_noise < 0:
        raise ValueError("em_gain>=1 and read_noise>=0 must be finite")
    if not np.isfinite(offset):
        raise ValueError("offset must be finite")
    counts = rng.poisson(expected)
    if em_gain > 1:
        multiplied = np.zeros_like(expected)
        mask = counts > 0
        multiplied[mask] = rng.gamma(shape=counts[mask], scale=em_gain)
        signal = multiplied / em_gain
    else:
        signal = counts.astype(np.float64)
    return (signal + rng.normal(0, read_noise / em_gain, size=expected.shape) + offset).astype(np.float32)


def _scene(size, rng, empty=False):
    yy, xx = np.mgrid[:size, :size].astype(np.float64)
    # Random background is retained in both targets and observations. Even
    # empty scenes in different splits are not pixel-identical clean targets.
    background = rng.uniform(0.2, 2.0) + rng.uniform(0.0, 0.7) * xx / size + rng.uniform(0, 0.7) * yy / size
    sharp, blurred = background.copy(), background.copy()
    spacing = float(rng.uniform(6.5, 11.5))
    radius = size * rng.uniform(0.31, 0.43)
    center = np.array([size / 2, size / 2]) + rng.uniform(-1.5, 1.5, 2)
    angle = float(rng.uniform(-np.pi, np.pi))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    phase_cos = np.cos(np.linspace(0, 2 * np.pi, 32, endpoint=False))
    points, ions = [], []
    candidates, missing = 0, 0
    for row, dy in enumerate(np.arange(-radius, radius + spacing, spacing * np.sqrt(3) / 2)):
        for dx in np.arange(-radius, radius + spacing, spacing):
            uv = np.array([dx + (row % 2) * spacing / 2, dy])
            if np.dot(uv, uv) > radius ** 2:
                continue
            position = center + rotation @ uv + rng.normal(0, 0.18, 2)
            x, y = map(float, position)
            if not (5 < x < size - 6 and 5 < y < size - 6):
                continue
            candidates += 1
            if empty or rng.random() < 0.1:
                missing += 1
                continue
            photons = float(rng.uniform(180, 1400))
            weak = bool(rng.random() < 0.2)
            if weak:
                photons *= float(rng.uniform(0.04, 0.18))
            sigma_x, sigma_y = rng.uniform(0.8, 1.6, 2)
            motion = float(rng.uniform(0, 0.8) + np.linalg.norm(uv) / max(radius, 1) * rng.uniform(0.5, 3.5))
            direction = float(rng.uniform(0, 2 * np.pi))
            normalization = photons / (2 * np.pi * sigma_x * sigma_y)
            sharp += normalization * np.exp(-0.5 * (((xx - x) / sigma_x) ** 2 + ((yy - y) / sigma_y) ** 2))
            for phase in phase_cos:
                mx, my = motion * phase * np.cos(direction), motion * phase * np.sin(direction)
                blurred += normalization / len(phase_cos) * np.exp(
                    -0.5 * (((xx - x - mx) / sigma_x) ** 2 + ((yy - y - my) / sigma_y) ** 2))
            points.append([x, y])
            ions.append({"x": x, "y": y, "photons": photons, "weak": weak,
                         "sigma_x_px": float(sigma_x), "sigma_y_px": float(sigma_y),
                         "motion_amplitude_px": motion, "motion_direction_rad": direction})
    return sharp, blurred, points, {"ions": ions, "missing_sites": missing,
                                    "candidate_sites": candidates, "empty_scene": not bool(points)}


def make_synthetic_dataset(output, train_groups=8, val_groups=2, test_groups=2,
                           size=64, seed=42, em_gain=1.0):
    """Write one independent paired frame per scene and return manifest.jsonl.

    Small defaults are installation fixtures. Increase groups for experiments;
    keep the unobserved simulator-to-real gap explicit. No existing directory
    containing files is overwritten. A random local generator is used.
    """
    counts = {"train": _integer(train_groups, "train_groups", 1),
              "val": _integer(val_groups, "val_groups", 1),
              "test": _integer(test_groups, "test_groups", 1)}
    size = _integer(size, "size", 24)
    if not np.isfinite(em_gain) or em_gain < 1:
        raise ValueError("em_gain must be finite and >=1")
    output = Path(output).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Use a new or empty synthetic output directory: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)
    # Two float arrays, modest metadata and temporary I/O margin. No compression
    # assumption is needed, and storage is checked before generating frames.
    required = sum(counts.values()) * (size * size * 8 + 32 * 1024) + 64 * 1024 ** 2
    if shutil.disk_usage(output).free < required:
        raise OSError("Insufficient disk space for synthetic fixtures")
    rng = np.random.default_rng(seed)
    rows, metadata = [], []
    max_value, sequence = 0.0, 0
    for split, count in counts.items():
        for index in range(count):
            rid = "{}_{:05d}".format(split, index)
            group = "synthetic_{:06d}".format(sequence)
            # Guarantee a blank within a reasonably sized split, while the
            # smallest smoke configuration still contains useful emitters.
            empty = index == count - 1 and count >= 4
            sharp, blurred, points, info = _scene(size, rng, empty=empty)
            offset = 10.0
            observed = sample_camera(blurred, rng, em_gain=em_gain, read_noise=1.5 * em_gain, offset=offset)
            target = (sharp + offset).astype(np.float32)
            relative_input, relative_target = "images/{}_input.npy".format(rid), "images/{}_target.npy".format(rid)
            (output / "images").mkdir(exist_ok=True)
            np.save(output / relative_input, observed, allow_pickle=False)
            np.save(output / relative_target, target, allow_pickle=False)
            rows.append({"id": rid, "split": split, "group_id": group,
                         "input": relative_input, "target": relative_target,
                         "target_kind": "synthetic_clean", "points": points,
                         "points_kind": "synthetic_ground_truth"})
            metadata.append({"id": rid, "group_id": group, **info})
            # Fit scale to training data only; val/test overflow is reported by
            # the sampler rather than silently refitting normalization.
            if split == "train":
                max_value = max(max_value, float(observed.max()), float(target.max()))
            sequence += 1
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    (output / "metadata.json").write_text(json.dumps({
        "schema_version": 1, "source": "synthetic_physics_approximation_not_experimental_validation",
        "seed": int(seed), "size": size, "split_groups": counts,
        "target_definition": "stationary optical Gaussian PSF at equilibrium center plus expected background and offset",
        "observation_definition": "32 phase samples of sinusoidal motion; Poisson electrons; optional ideal Gamma EM; Gaussian read noise",
        "limitations": ["not calibrated", "constant fluorescence rate", "no CIC or saturation", "point-sampled optical PSF; no pixel area integration"],
        "coordinates": "zero-based [x=column,y=row], continuous equilibrium centers",
        "em_gain": float(em_gain), "read_noise_output_electrons": float(1.5 * em_gain),
        "effective_read_noise_input_electrons": 1.5, "offset": 10.0,
        "normalization_suggestion": {"offset": 0.0, "scale": max(1.0, max_value * 1.05)},
        "scenes": metadata,
    }, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return manifest
