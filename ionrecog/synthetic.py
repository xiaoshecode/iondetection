"""Small synthetic repeated-frame datasets for installation and pipeline checks.

These images are not measurements and do not establish real-ion performance.
The generator models Poisson counts plus Gaussian read noise, NOT a complete
EMCCD electron-multiplication process. Each group's integrated motion blur,
positions and expected brightness remain fixed across independent noise draws.
"""

import json
from pathlib import Path
from typing import Dict, Union

import numpy as np
from PIL import Image


def generate_demo(root: Union[str, Path], groups: int = 4, frames: int = 6,
                  size: int = 128, seed: int = 0) -> Dict:
    """Write uint16 train/val groups plus synthetic-only ``truth.json``.

    The last max(1, groups//4) groups are validation scenes. New layouts are
    sampled for every group; consecutive frames are NEVER split across subsets.
    Existing planned files cause an error rather than being overwritten.
    """
    for name, value, lower in (("groups", groups, 2), ("frames", frames, 1), ("size", size, 32)):
        if not isinstance(value, (int, np.integer)) or value < lower:
            raise ValueError("{} must be an integer >= {}".format(name, lower))
    root = Path(root)
    val_groups = max(1, groups // 4)
    train_groups = groups - val_groups
    outputs = []
    for g in range(groups):
        split = "train" if g < train_groups else "val"
        for f in range(frames):
            outputs.append(root / split / "group_{:03d}".format(g) / "frame_{:03d}.png".format(f))
    existing = [p for p in outputs + [root / "truth.json"] if p.exists()]
    if existing:
        raise FileExistsError("Demo output already exists; choose a fresh directory: {}".format(existing[0]))
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float64)
    truth = {
        "schema_version": 1,
        "source": "synthetic_demo_not_experimental_ground_truth",
        "coordinate_system": "zero-based x=column, y=row; integrated symmetric motion centers",
        "noise_model": "Poisson photon counts + Gaussian read noise; no EM multiplication",
        "seed": int(seed), "groups": {}, "images": [],
    }
    total_images = 0
    for g in range(groups):
        split = "train" if g < train_groups else "val"
        group = "{}/group_{:03d}".format(split, g)
        directory = root / group
        directory.mkdir(parents=True, exist_ok=True)
        spacing = float(rng.uniform(10.0, 13.0))
        cx, cy = size / 2.0 + rng.uniform(-1, 1, size=2)
        theta = float(rng.uniform(-0.12, 0.12))
        sx, sy = float(rng.uniform(1.15, 1.55)), float(rng.uniform(1.15, 1.55))
        positions = []
        for row, gy in enumerate(np.arange(7, size - 7, spacing * np.sqrt(3) / 2)):
            for gx in np.arange(7, size - 7, spacing):
                dx = gx + (row % 2) * spacing / 2 - cx
                dy = gy - cy
                if (dx / (size * 0.39)) ** 2 + (dy / (size * 0.42)) ** 2 > 1:
                    continue
                x = cx + dx * np.cos(theta) - dy * np.sin(theta) + rng.normal(0, 0.1)
                y = cy + dx * np.sin(theta) + dy * np.cos(theta) + rng.normal(0, 0.1)
                if 4 < x < size - 5 and 4 < y < size - 5:
                    positions.append((float(x), float(y)))
        expected = 100.0 + 35.0 * xx / size + 20.0 * yy / size
        ions = []
        for idx, (x, y) in enumerate(positions, start=1):
            amplitude = float(rng.uniform(2200, 4500))
            motion = float(0.2 + 1.0 * abs(x - cx) / (size * 0.4))
            spot = np.zeros_like(expected)
            for phase in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                dx = motion * np.cos(phase)
                spot += np.exp(-0.5 * (((xx - x - dx) / sx) ** 2 + ((yy - y) / sy) ** 2))
            expected += amplitude * spot / 12.0
            ions.append({"id": idx, "x": x, "y": y, "motion_amplitude_px": motion, "status": "observed"})
        truth["groups"][group] = {"count": len(ions), "ions": ions, "static_scene": True}
        for f in range(frames):
            observed = rng.poisson(expected).astype(np.float64) + rng.normal(0, 8.0, size=expected.shape) + 128.0
            observed = np.clip(np.rint(observed), 0, 65535).astype(np.uint16)
            relative = "{}/frame_{:03d}.png".format(group, f)
            Image.fromarray(observed).save(str(root / relative))
            truth["images"].append({"file": relative, "group": group, "ions": ions})
            total_images += 1
    truth_path = root / "truth.json"
    truth_path.write_text(json.dumps(truth, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"root": str(root.resolve()), "train_groups": train_groups, "val_groups": val_groups,
            "images": total_images, "truth": str(truth_path.resolve())}
