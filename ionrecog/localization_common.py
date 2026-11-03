"""Shared I/O for experimental localization methods, without changing raw data.

Coordinates are zero-based x=column, y=row. All model scores are uncalibrated
unless a separate experiment establishes calibration. No quantum-state labels
are inferred here.
"""
import csv
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np


def ensure_disk_space(output, reserve_gib=2.0, estimated_bytes=0):
    if not np.isfinite(reserve_gib) or reserve_gib < 0:
        raise ValueError("reserve_gib must be finite and nonnegative")
    if not np.isfinite(estimated_bytes) or estimated_bytes < 0:
        raise ValueError("estimated_bytes must be finite and nonnegative")
    location = Path(output).resolve()
    while not location.exists():
        location = location.parent
    free = shutil.disk_usage(location).free
    reserve = int(float(reserve_gib) * 1024 ** 3)
    if free < reserve + int(estimated_bytes):
        raise ValueError("Insufficient disk space: %.2f GiB free; %.2f GiB reserve plus %.2f GiB estimated output required" %
                         (free / 1024 ** 3, reserve_gib, estimated_bytes / 1024 ** 3))
    return {"free_bytes": free, "reserve_bytes": reserve, "estimated_bytes": int(estimated_bytes)}


def prepare_output(output, sources=(), reserve_gib=2.0, estimated_bytes=0):
    from .cli import check_destination
    destination = check_destination(output, sources)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Output must be a new or empty directory: " + str(destination))
    ensure_disk_space(destination, reserve_gib, estimated_bytes)
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def save_checkpoint_atomic(path, state):
    """Write a complete local checkpoint before replacing a previous snapshot."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        with temporary.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    # Serialize first, so a serialization failure cannot replace valid output.
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def add_infer_arguments(parser):
    parser.add_argument("--input", required=True, help="Raw grayscale image or dataset directory")
    parser.add_argument("--output", required=True, help="Separate results directory")
    parser.add_argument("--config", help="Existing region/detection YAML (e.g. configs/existing_crystal.yaml)")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true", help="Replace matching result files only")


def _sanitize_ions(rows, shape, allowed, radius):
    height, width = shape
    accepted = []
    rejected = {"outside_image": 0, "outside_region": 0}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Predictor must return a list of coordinate dictionaries")
        row = dict(row)
        try:
            x, y = float(row["x"]), float(row["y"])
            r = float(row.get("radius", radius))
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("Every predicted point requires numeric x, y and radius") from exc
        if not np.isfinite([x, y, r]).all() or r <= 0:
            raise ValueError("Predicted coordinates/radius must be finite; radius must be positive")
        if not (0 <= x < width and 0 <= y < height):
            rejected["outside_image"] += 1
            continue
        ix, iy = min(width - 1, int(np.floor(x + .5))), min(height - 1, int(np.floor(y + .5)))
        if allowed is not None and not allowed[iy, ix]:
            rejected["outside_region"] += 1
            continue
        row.update(x=x, y=y, radius=r)
        row.setdefault("status", "candidate")
        accepted.append(row)
    accepted.sort(key=lambda row: (row["y"], row["x"]))
    for index, row in enumerate(accepted, 1):
        row["id"] = index
    return accepted, rejected


def run_prediction(args, predictor, method, metadata=None):
    """Run ``predictor(raw, region_mask)`` and preserve method diagnostics.

    A predictor returns {ions: [...], diagnostics: {...}}. Optional
    coverage_complete/has_abstentions flags propagate to JSON evaluation guards.
    No-crystal is an abstention, never proof that a physical image is empty.
    """
    from .cli import check_destination, check_disk_budget, load_config, write_csv, write_json
    from .data import list_images, load_image
    from .classical.region import segment_crystal
    from .classical.reference import measure_sites
    from .classical.detection import save_overlay

    config = load_config(args.config)
    source = Path(args.input)
    files = list_images(source)
    if args.max_images is not None:
        if args.max_images < 1:
            raise ValueError("--max-images must be positive")
        files = files[:args.max_images]
    if not files:
        raise ValueError("No supported raw images found: " + str(source))
    protected = [source]
    for name in ("checkpoint", "config"):
        value = getattr(args, name, None)
        if value:
            protected.append(Path(value))
    output = check_destination(args.output, protected)
    if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not args.overwrite)):
        raise ValueError("Inference output is not empty; choose another output or --overwrite")
    budget = check_disk_budget(output, files, False, args.reserve_gib)
    output.mkdir(parents=True, exist_ok=True)
    fields = ("image", "method", "id", "x", "y", "radius", "status", "score", "raw_flux",
              "raw_peak", "raw_background", "raw_snr", "roi_clipped")
    aggregate = output / "detections.csv"
    frames = []
    note = "Experimental candidate localization; scores are not calibrated confidence or quantum-state fidelity."
    def write_summary():
        _write_json_atomic(output / "summary.json", {"method": method, "metadata": metadata or {},
                   "config": config, "images_requested": len(files), "images_completed": len(frames),
                   "frames": frames, "disk_budget": budget, "metrics_are_ground_truth_accuracy": False,
                   "complete": len(frames) == len(files), "note": note})
    # Invalidate the previous run before overwriting anything. If a later read
    # or predictor fails, an old complete=true report cannot describe this run.
    write_summary()
    write_csv(aggregate, [], fields)
    write_csv(output / "summary.csv", [], ("image", "count", "seconds", "region_status", "coverage_complete", "has_abstentions"))
    for index, path in enumerate(files):
        ensure_disk_space(output, args.reserve_gib)
        raw = load_image(path)
        if np.any(raw < 0):
            raise ValueError("Raw localization input must be nonnegative: " + str(path))
        mask, region = segment_crystal(raw, config["region"])
        allowed = mask if region["mode"] != "off" else None
        start = time.perf_counter()
        no_region = allowed is not None and not allowed.any()
        result = ({"ions": [], "diagnostics": {"reason": "no_reliable_crystal_region"},
                   "has_abstentions": True} if no_region else predictor(raw, allowed))
        if not isinstance(result, dict) or not isinstance(result.get("ions"), list):
            raise ValueError("Predictor must return a dictionary with an ions list")
        rows, rejected = _sanitize_ions(result["ions"], raw.shape, allowed, config["detection"]["radius"])
        for row in rows:
            measurement = measure_sites(raw, [row], row["radius"])[0]
            row.update({k: v for k, v in measurement.items() if k.startswith("raw_") or k == "roi_clipped"})
        rel = path.relative_to(source).as_posix() if source.is_dir() else path.name
        stem = output / "images" / rel
        stem.parent.mkdir(parents=True, exist_ok=True)
        complete = bool(result.get("coverage_complete", True))
        abstentions = bool(result.get("has_abstentions", False))
        payload = {"image": rel, "shape": list(raw.shape), "method": method, "ions": rows,
                   "region": region, "diagnostics": result.get("diagnostics", {}),
                   "rejected": rejected, "coverage_complete": complete, "has_abstentions": abstentions,
                   "metadata": metadata or {}, "note": note}
        write_json(str(stem) + "_ions.json", payload)
        write_csv(str(stem) + "_ions.csv", rows)
        save_overlay(raw, rows, str(stem) + "_overlay.png", mask=allowed)
        with aggregate.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            for row in rows:
                writer.writerow({key: ({"image": rel, "method": method}.get(key, row.get(key, ""))) for key in fields})
        frames.append({"image": rel, "count": len(rows), "seconds": time.perf_counter() - start,
                       "region_status": region["status"], "coverage_complete": complete,
                       "has_abstentions": abstentions, "rejected": rejected})
        print("[%d/%d] %s: %d candidates (%s)" % (index + 1, len(files), rel, len(rows), region["status"]))
        # A valid partial summary survives a later failed image or disk limit.
        write_summary()
    write_csv(output / "summary.csv", [{k: v for k, v in frame.items() if k != "rejected"} for frame in frames])
    print("Results:", output)
    return output
