"""Online point-labelled synthetic data with explicit, configurable assumptions.

Counts are Poisson plus Gaussian read noise and a fixed ADU bias. EM gain,
optical aberration phase retrieval and physical RF phase inference are absent.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class SimulationConfig:
    size: int = 64
    min_ions: int = 1
    max_ions: int = 20
    empty_fraction: float = .15
    min_separation: float = 2.5
    lattice_fraction: float = .5
    sigma_min: float = 1.0
    sigma_max: float = 1.8
    motion_max: float = 1.5
    phase_samples: int = 12
    amplitude_min: float = 500.0
    amplitude_max: float = 4500.0
    background_min: float = 20.0
    background_max: float = 200.0
    gradient_max: float = 60.0
    read_noise: float = 8.0
    camera_bias: float = 128.0
    input_offset: float = 128.0
    input_scale: float = 4096.0
    heatmap_sigma: float = 1.0

    def validate(self):
        for key, value in asdict(self).items():
            if not np.isscalar(value) or not np.isfinite(value):
                raise ValueError("Simulation parameter %s must be finite" % key)
        for key in ("size", "min_ions", "max_ions", "phase_samples"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("%s must be an integer" % key)
        if not 16 <= self.size <= 512 or not 0 <= self.min_ions <= self.max_ions <= self.size ** 2:
            raise ValueError("Require size in [16,512] and 0 <= min_ions <= max_ions <= size**2")
        if not 4 <= self.phase_samples <= 128:
            raise ValueError("phase_samples must be in [4,128]")
        for key in ("empty_fraction", "lattice_fraction"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError("%s must be in [0,1]" % key)
        if not 0 < self.sigma_min <= self.sigma_max or self.motion_max < 0:
            raise ValueError("PSF widths must be positive ordered bounds; motion_max >= 0")
        if not 0 < self.amplitude_min <= self.amplitude_max:
            raise ValueError("Amplitude bounds must be positive and ordered")
        if not 0 <= self.background_min <= self.background_max:
            raise ValueError("Background bounds must be nonnegative and ordered")
        if min(self.gradient_max, self.read_noise, self.camera_bias) < 0:
            raise ValueError("Noise, bias and gradient parameters must be nonnegative")
        if self.input_scale <= 0 or self.heatmap_sigma <= 0 or self.min_separation < 1.5:
            raise ValueError("Positive scale/heatmap_sigma and min_separation >= 1.5 required")
        if self.margin * 2 >= self.size:
            raise ValueError("PSF/motion too wide for simulation size; increase size")
        return self

    @property
    def margin(self):
        return max(3, int(np.ceil(3 * self.sigma_max + self.motion_max)))

    def to_dict(self):
        return asdict(self)


def read_simulation_config(path=None):
    import json
    from pathlib import Path
    values = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    if not isinstance(values, dict):
        raise ValueError("Simulation JSON must be an object")
    unknown = set(values) - set(SimulationConfig.__dataclass_fields__)
    if unknown:
        raise ValueError("Unknown simulation keys: %s" % sorted(unknown))
    return SimulationConfig(**values).validate()


def sample_seed(seed, split, index):
    """Domain-separated deterministic streams: 0=train, 1=validation, 2=test."""
    if any(not isinstance(x, (int, np.integer)) or x < 0 for x in (seed, split, index)):
        raise ValueError("seed/split/index must be nonnegative integers")
    return np.random.SeedSequence([int(seed), int(split), int(index)])


def simulate(config, seed):
    config.validate()
    rng = np.random.default_rng(seed)
    size, margin = config.size, config.margin
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    background = rng.uniform(config.background_min, config.background_max)
    gx, gy = rng.uniform(-config.gradient_max, config.gradient_max, size=2)
    expected = np.maximum(0, background + gx * (xx / size - .5) + gy * (yy / size - .5))
    count = 0 if rng.random() < config.empty_fraction else int(rng.integers(config.min_ions, config.max_ions + 1))
    points = []
    lattice = rng.random() < config.lattice_fraction
    if lattice:
        spacing = rng.uniform(max(3.5, config.min_separation), max(4.0, config.min_separation + 6))
        candidates = [(x + (row % 2) * spacing / 2, y)
                      for row, y in enumerate(np.arange(margin, size - margin, spacing * np.sqrt(3) / 2))
                      for x in np.arange(margin, size - margin, spacing)]
        rng.shuffle(candidates)
        proposals = [(x + rng.uniform(-.4, .4), y + rng.uniform(-.4, .4)) for x, y in candidates]
    else:
        proposals = rng.uniform(margin, size - 1 - margin, size=(max(100, count * 100), 2))
    occupied = set()
    for x, y in proposals:
        if len(points) >= count:
            break
        if not margin <= x <= size - 1 - margin or not margin <= y <= size - 1 - margin:
            continue
        key = (int(np.floor(x + .5)), int(np.floor(y + .5)))
        if key in occupied or any(np.hypot(x - px, y - py) < config.min_separation for px, py in points):
            continue
        points.append((float(x), float(y)))
        occupied.add(key)
    # Dense requested scenes can contain fewer points if rejection sampling or
    # lattice capacity is exhausted. Labels always describe actual emitters.
    for x, y in points:
        sx, sy = rng.uniform(config.sigma_min, config.sigma_max, size=2)
        angle = rng.uniform(-np.pi, np.pi)
        motion = rng.uniform(0, config.motion_max)
        motion_angle = rng.uniform(-np.pi, np.pi)
        spot = np.zeros((size, size), np.float32)
        for phase in np.arange(config.phase_samples) * (2 * np.pi / config.phase_samples):
            dx = xx - x - motion * np.cos(phase) * np.cos(motion_angle)
            dy = yy - y - motion * np.cos(phase) * np.sin(motion_angle)
            u = dx * np.cos(angle) + dy * np.sin(angle)
            v = -dx * np.sin(angle) + dy * np.cos(angle)
            spot += np.exp(-.5 * ((u / sx) ** 2 + (v / sy) ** 2))
        expected += rng.uniform(config.amplitude_min, config.amplitude_max) * spot / config.phase_samples
    raw = rng.poisson(expected).astype(np.float32)
    raw += rng.normal(0, config.read_noise, size=(size, size)).astype(np.float32) + config.camera_bias
    return np.maximum(raw, 0), np.asarray(points, dtype=np.float32).reshape(-1, 2)


def make_targets(points, size, heatmap_sigma=1.0):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if not np.isfinite(points).all() or np.any(points < 0) or np.any(points > size - 1):
        raise ValueError("Target points must be finite coordinates inside the image pixel-center bounds")
    heatmap = np.zeros((1, size, size), np.float32)
    offsets = np.zeros((2, size, size), np.float32)
    centers = np.zeros((1, size, size), np.float32)
    yy, xx = np.mgrid[:size, :size]
    for x, y in points:
        ix, iy = int(np.floor(x + .5)), int(np.floor(y + .5))
        if centers[0, iy, ix]:
            raise ValueError("This baseline supports at most one emitter center per pixel")
        heatmap[0] = np.maximum(heatmap[0], np.exp(-.5 * ((xx - ix) ** 2 + (yy - iy) ** 2) / heatmap_sigma ** 2))
        offsets[:, iy, ix] = [x - ix, y - iy]
        centers[0, iy, ix] = 1
    return heatmap, offsets, centers
