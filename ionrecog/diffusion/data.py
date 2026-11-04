"""Explicit, group-safe data manifests for conditional diffusion.

Images retain their raw ADU scale until PatchSampler applies one fixed mapping.
A repeat group is a USER ASSERTION of stable, aligned independent exposures,
not an automatic motion-registration or micromotion-removal procedure.
"""

import hashlib
import json
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch

from ..data import load_image


SPLITS = ("train", "val", "test")
TARGET_KINDS = ("clean", "synthetic_clean", "low_noise")


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return int(value)


def _points(value, shape):
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("points must contain finite [x, y] coordinates") from exc
    if result.shape == (0,):
        result = result.reshape(0, 2)
    if result.ndim != 2 or result.shape[1] != 2 or not np.isfinite(result).all():
        raise ValueError("points must be finite Nx2 coordinates; [] explicitly means an empty scene")
    h, w = shape
    if len(result) and (np.any(result < 0) or np.any(result[:, 0] >= w) or np.any(result[:, 1] >= h)):
        raise ValueError("points lie outside the zero-based image coordinates")
    if len(result) != len(np.unique(result, axis=0)):
        raise ValueError("points contains duplicate centers")
    return result


def _paths(record):
    return record["frames"] if "frames" in record else [record["input"], record["target"]]


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_manifest(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Manifest does not exist: {}".format(path))
    records, ids, group_splits = [], set(), {}
    file_cache, path_splits, pixel_splits, byte_splits = {}, {}, {}, {}
    fingerprint = hashlib.sha256()
    with path.open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSON at {} line {}".format(path, number)) from exc
            if not isinstance(record, dict):
                raise ValueError("Each manifest line must be a JSON object")
            record = dict(record)
            for field in ("id", "split", "group_id"):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    raise ValueError("Record line {} requires nonempty {}".format(number, field))
            rid, split, group = record["id"], record["split"], record["group_id"]
            if rid in ids:
                raise ValueError("Duplicate record id: {}".format(rid))
            ids.add(rid)
            if split not in SPLITS:
                raise ValueError("split must be train, val, or test: {}".format(rid))
            if group in group_splits and group_splits[group] != split:
                raise ValueError("Cross-split group leakage: {}".format(group))
            group_splits[group] = split
            if "frames" in record:
                if "input" in record or "target" in record:
                    raise ValueError("Record must use either frames or input/target, not both")
                if not isinstance(record["frames"], list) or len(record["frames"]) < 3:
                    raise ValueError("Repeat records require >=3 independent frames")
                if record.get("stable") is not True:
                    raise ValueError("Repeat records require stable:true after checking alignment and state stability")
                record["mode"] = "repeat"
                record["target_kind"] = "repeat_mean"
                fields = [("frames", i, value) for i, value in enumerate(record["frames"])]
                record["frames"] = list(record["frames"])
            else:
                if record.get("target_kind") not in TARGET_KINDS:
                    raise ValueError("Paired records require target_kind in {}".format(TARGET_KINDS))
                record["mode"] = "paired"
                fields = [(field, None, record.get(field)) for field in ("input", "target")]
            record_files, shapes = [], []
            for field, index, value in fields:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("Image paths must be nonempty strings: {}".format(rid))
                image_path = Path(value).expanduser()
                if not image_path.is_absolute():
                    image_path = path.parent / image_path
                resolved = str(image_path.resolve())
                if not Path(resolved).is_file():
                    raise FileNotFoundError("Image missing in {}: {}".format(rid, resolved))
                if index is None:
                    record[field] = resolved
                else:
                    record[field][index] = resolved
                if resolved in record_files:
                    raise ValueError("Input/target or repeat frames reuse the same path: {}".format(rid))
                record_files.append(resolved)
                if resolved not in file_cache:
                    image = load_image(resolved)
                    pixels = hashlib.sha256()
                    pixels.update(str(tuple(image.shape)).encode("ascii"))
                    pixels.update(np.ascontiguousarray(image, dtype="<f4").tobytes())
                    file_cache[resolved] = {
                        "shape": tuple(image.shape), "min": float(image.min()), "max": float(image.max()),
                        "pixels": pixels.hexdigest(), "bytes": _sha256_file(resolved),
                    }
                info = file_cache[resolved]
                shapes.append(info["shape"])
                for seen, key, kind in ((path_splits, resolved, "path"),
                                        (pixel_splits, info["pixels"], "decoded content"),
                                        (byte_splits, info["bytes"], "file content")):
                    if key in seen and seen[key] != split:
                        raise ValueError("Cross-split {} leakage at {}".format(kind, resolved))
                    seen[key] = split
                fingerprint.update((rid + "\0" + split + "\0" + group + "\0" + info["pixels"]).encode("utf-8"))
            if any(shape != shapes[0] for shape in shapes):
                raise ValueError("All images in record {} must have equal shapes".format(rid))
            if "frames" in record:
                pixel_ids = [file_cache[p]["pixels"] for p in record_files]
                if len(set(pixel_ids)) != len(pixel_ids):
                    raise ValueError("Repeat group contains pixel-identical frames; check independent acquisition: {}".format(rid))
            record["shape"] = list(shapes[0])
            if "points" in record:
                pts = _points(record["points"], shapes[0])
                record["points"] = None if pts is None else pts.tolist()
            # Labels and target semantics are part of the reproducibility fingerprint.
            fingerprint.update(json.dumps({"points": record.get("points"), "target_kind": record["target_kind"]},
                                          sort_keys=True, allow_nan=False).encode("utf-8"))
            records.append(record)
    if not records:
        raise ValueError("Manifest contains no records")
    splits = {}
    for split in SPLITS:
        subset = [r for r in records if r["split"] == split]
        splits[split] = {"records": len(subset), "groups": len({r["group_id"] for r in subset}),
                         "unique_images": len({p for r in subset for p in _paths(r)}),
                         "with_points": sum(r.get("points") is not None for r in subset)}
    report = {
        "schema_version": 1, "manifest": str(path), "manifest_sha256": _sha256_file(path),
        "content_fingerprint": fingerprint.hexdigest(), "records": len(records), "splits": splits,
        "unique_images": len(file_cache),
        "target_kinds": dict(Counter(r["target_kind"] for r in records)),
        "normalization_range": {"min": min(v["min"] for v in file_cache.values()),
                                "max": max(v["max"] for v in file_cache.values())},
        "leakage_checks": ["group_id", "resolved_path", "sha256_file", "sha256_decoded_float32"],
        "repeat_note": "stable:true is an assertion; no registration or physical stability test is performed",
    }
    return records, report


def load_manifest(path):
    """Read JSONL, resolve paths and verify shapes, values, labels and split isolation.

    Content hashes catch copied/re-encoded frames across splits. Identity of
    physical crystal configurations must additionally be recorded in group_id.
    This reads all unique images once; it does not retain a full image cache.
    """
    return _inspect_manifest(path)[0]


def audit_manifest(path):
    """Return a JSON-serializable audit; raise on corruption or split leakage."""
    return _inspect_manifest(path)[1]


def _read_case(record, rng, max_target_frames, reader):
    if "frames" in record:
        frames = record["frames"]
        if len(frames) < 3 or record.get("stable") is not True:
            raise ValueError("Repeat cases need >=3 stable, independent, aligned frames")
        if len({str(Path(p).resolve()) for p in frames}) != len(frames):
            raise ValueError("Repeat case contains duplicate paths")
        order = rng.permutation(len(frames))
        input_path = frames[int(order[0])]
        target_paths = [frames[int(i)] for i in order[1:1 + min(max_target_frames, len(frames) - 1)]]
        observed = reader(input_path)
        target = np.zeros_like(observed, dtype=np.float32)
        for item in target_paths:
            image = reader(item)
            if image.shape != observed.shape:
                raise ValueError("Repeat frames have different shapes")
            target += image / len(target_paths)
        kind = "repeat_mean"
    else:
        input_path, target_paths = record["input"], [record["target"]]
        if Path(input_path).resolve() == Path(target_paths[0]).resolve():
            raise ValueError("Input and target must not reference the same file")
        observed, target = reader(input_path), reader(target_paths[0])
        if observed.shape != target.shape:
            raise ValueError("Input and target have different shapes")
        kind = record["target_kind"]
    points = _points(record.get("points"), observed.shape)
    return {"observed": observed, "target": target, "points": points, "target_kind": kind,
            "input_path": str(input_path), "target_paths": [str(p) for p in target_paths]}


def read_case(record, seed=0, max_target_frames=8):
    """Read an evaluation pair in raw units using deterministic local sampling.

    Missing points -> None (unlabelled); points:[] -> known empty image. A repeat
    target excludes its input and keeps the same integrated motion blur.
    """
    max_target_frames = _integer(max_target_frames, "max_target_frames", 2)
    return _read_case(record, np.random.default_rng(seed), max_target_frames, load_image)


class PatchSampler:
    """Deterministic on-demand sampling; bounded cache and paired augmentation."""

    def __init__(self, records, patch_size, offset, scale, augment=True, max_target_frames=8):
        self.records = list(records)
        if not self.records:
            raise ValueError("PatchSampler needs nonempty records")
        self.patch_size = _integer(patch_size, "patch_size", 4)
        if not np.isfinite(offset) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("offset must be finite; scale must be finite and positive")
        self.offset, self.scale, self.augment = float(offset), float(scale), bool(augment)
        self.max_target_frames = _integer(max_target_frames, "max_target_frames", 2)
        self.cache = OrderedDict()
        for record in self.records:
            shape = record.get("shape")
            if shape is not None and min(shape) < self.patch_size:
                raise ValueError("Record {} is smaller than patch_size".format(record["id"]))

    def _read(self, path):
        key = str(path)
        if key not in self.cache:
            self.cache[key] = load_image(path)
            if len(self.cache) > 16:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def batch(self, seed, batch_size):
        batch_size = _integer(batch_size, "batch_size")
        rng = np.random.default_rng(seed)
        observed_batch, target_batch = [], []
        clipped = {"observed": 0, "target": 0}
        for _ in range(batch_size):
            record = self.records[int(rng.integers(len(self.records)))]
            case = _read_case(record, rng, self.max_target_frames, self._read)
            h, w = case["observed"].shape
            size = self.patch_size
            if min(h, w) < size:
                raise ValueError("Image is smaller than patch_size; reduce it rather than padding training targets")
            y, x = int(rng.integers(h - size + 1)), int(rng.integers(w - size + 1))
            pair = []
            for key in ("observed", "target"):
                patch = (case[key][y:y + size, x:x + size] - self.offset) / self.scale
                clipped[key] += int(np.count_nonzero((patch < 0) | (patch > 1)))
                pair.append(np.clip(patch, 0, 1))
            if self.augment:
                turns, flip = int(rng.integers(4)), bool(rng.integers(2))
                pair = [np.rot90(p, turns) for p in pair]
                if flip:
                    pair = [p[:, ::-1] for p in pair]
            observed_batch.append(np.ascontiguousarray(pair[0], dtype=np.float32))
            target_batch.append(np.ascontiguousarray(pair[1], dtype=np.float32))
        pixels = batch_size * self.patch_size ** 2
        return {"observed": torch.from_numpy(np.stack(observed_batch)[:, None]),
                "target": torch.from_numpy(np.stack(target_batch)[:, None]),
                "clipped_fraction": {k: v / pixels for k, v in clipped.items()}}
