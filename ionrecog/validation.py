"""Dataset checks; no inference of physical ground truth from directory names."""
import hashlib
from pathlib import Path

import numpy as np

from .data import find_groups, list_images, load_image


def validate_splits(train_root, val_root, method="masked"):
    train_root, val_root = Path(train_root).resolve(), Path(val_root).resolve()
    if train_root == val_root or train_root in val_root.parents or val_root in train_root.parents:
        raise ValueError("Train and validation roots must be separate, non-nested directories.")
    roots = {"train": train_root, "val": val_root}
    report, seen = {"method": method, "warnings": []}, {}
    group_sets = {}
    for split, root in roots.items():
        paths = list_images(root)
        if not paths:
            raise ValueError("No images in %s: %s. Populate data folders or run the demo." % (split, root))
        groups = find_groups(root)
        group_sets[split] = set(groups) - {"__root__"}
        if "__root__" in groups:
            report["warnings"].append(split + ": loose files treated as one group; use acquisition folders.")
        shapes, minimum, maximum = set(), float("inf"), float("-inf")
        metadata = {}
        for path in paths:
            image = load_image(path)
            shapes.add(tuple(image.shape))
            image_min, image_max = float(image.min()), float(image.max())
            if image_min < 0:
                raise ValueError("Negative pixels are unsupported; provide nonnegative raw camera ADU images, not background-subtracted frames: %s" % path)
            minimum, maximum = min(minimum, image_min), max(maximum, image_max)
            # Decode before hashing: catches the same pixels saved as PNG and NPY.
            digest = hashlib.sha256(str(image.shape).encode() + np.ascontiguousarray(image, dtype="<f4").tobytes()).hexdigest()
            metadata[path] = (tuple(image.shape), digest)
            if digest in seen and seen[digest][0] != split:
                raise ValueError("Duplicate image across train/val: %s and %s" % (seen[digest][1], path))
            if digest in seen:
                report["warnings"].append("Repeated identical pixels within %s: %s" % (split, path.name))
            seen[digest] = (split, str(path))
        if method == "diffusion":
            for name, files in groups.items():
                if len(files) < 3:
                    raise ValueError("Diffusion group '%s/%s' needs >=3 independent static repeat frames; found %d." % (split, name, len(files)))
                if len({metadata[p][0] for p in files}) != 1:
                    raise ValueError("Diffusion repeats must have identical image dimensions: " + name)
                group_hashes = {}
                for path in files:
                    digest = metadata[path][1]
                    if digest in group_hashes:
                        raise ValueError("Duplicate decoded frame in diffusion group '%s/%s': %s and %s. Independent repeat exposures are required; copied frames leak input pixels into the target." % (split, name, group_hashes[digest], path))
                    group_hashes[digest] = path
            report["warnings"].append("Static positions, states, exposure and gain CANNOT be verified from images alone; confirm acquisition metadata.")
        report[split] = {"root": str(root), "images": len(paths), "groups": {k: len(v) for k, v in groups.items()},
                         "shapes": sorted(shapes), "min": minimum, "max": maximum}
    shared = group_sets["train"] & group_sets["val"]
    if shared:
        raise ValueError("Same acquisition group names in train/val: %s. Split entire independent sessions, not adjacent frames." % sorted(shared))
    return report
