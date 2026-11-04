"""Adapt signed NPY frames and call the existing, unmodified inference CLI.

From Net Recog: .\.venv\Scripts\python.exe CCD_2000ion/script/run_recognition.py
All new artifacts stay under CCD_2000ion. No training or algorithm edits.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

DATA = Path(__file__).resolve().parents[1]
PROJECT = DATA.parent
CHECKPOINT = PROJECT / "runs/existing_20261103/masked_training/best.pt"
CONFIG = PROJECT / "configs/default.yaml"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def snapshot():
    # Includes all algorithm modules, CLI, existing configurations and checkpoint.
    files = sorted((PROJECT / "ionrecog").rglob("*.py"))
    files += sorted((PROJECT / "configs").glob("*.yaml")) + [CHECKPOINT]
    return {p.relative_to(PROJECT).as_posix(): digest(p) for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA / "20260403_2000pure")
    parser.add_argument("--output", type=Path, default=DATA / "recognition_20261104_20260403")
    args = parser.parse_args()
    source, out = args.input.resolve(), args.output.resolve()
    if DATA not in out.parents or DATA not in source.parents:
        parser.error("Input and output must be inside CCD_2000ion")
    if source == out or source in out.parents or out in source.parents:
        parser.error("Input and output must not overlap")
    if out.exists():
        parser.error("Output already exists; choose a new --output")
    files = sorted(source.glob("*.npy"))
    if not files:
        parser.error("No NPY input frames")
    if shutil.disk_usage(DATA).free < 5 * 1024 ** 3:
        parser.error("At least 5 GiB free space is required")
    protected_before = snapshot()
    out.mkdir()
    write_json(out / "protected_before.json", protected_before)
    records = []
    global_min, global_max = float("inf"), float("-inf")
    for path in files:
        raw = np.load(str(path), allow_pickle=False)
        if raw.ndim != 2 or not raw.size or raw.dtype.kind not in "buif" or not np.isfinite(raw).all():
            raise ValueError("Expected finite real 2-D image: {}".format(path))
        low, high = float(raw.min()), float(raw.max())
        global_min, global_max = min(global_min, low), max(global_max, high)
        records.append({"source": path.name, "source_sha256": digest(path), "shape": list(raw.shape),
                        "dtype": str(raw.dtype), "raw_min": low, "raw_max": high,
                        "negative_pixels": int(np.count_nonzero(raw < 0))})
    offset = int(max(0, np.ceil(-global_min)))
    if global_max + offset > 65535:
        raise ValueError("Unit-gain uint16 conversion would overflow; no automatic rescaling")
    converted = out / "input_png16"
    converted.mkdir()
    max_error = 0.0
    for path, record in zip(files, records):
        raw = np.load(str(path), allow_pickle=False)
        pixels = np.rint(raw.astype(np.float64) + offset).astype(np.uint16)
        destination = converted / (path.stem + ".png")
        Image.fromarray(pixels).save(str(destination))
        with Image.open(destination) as image:
            decoded = np.asarray(image)
        if not np.array_equal(decoded, pixels):
            raise RuntimeError("PNG round-trip failed: {}".format(destination))
        error = float(np.max(np.abs(decoded.astype(np.float64) - offset - raw)))
        if error > 0.500001:
            raise RuntimeError("Unexpected conversion error")
        max_error = max(max_error, error)
        record.update({"png": destination.relative_to(out).as_posix(), "png_sha256": digest(destination),
                       "png_min": int(pixels.min()), "png_max": int(pixels.max()),
                       "max_abs_rounding_error": error})
    # PyYAML treats an unquoted YAML 1.1 'off' as boolean False. Quote it in
    # this run-local config only, preserving the existing full-frame setting.
    config_text = CONFIG.read_text(encoding="utf-8")
    config_text = config_text.replace("  mode: off", '  mode: "off"')
    run_config = out / "config_used.yaml"
    run_config.write_text(config_text, encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "source": str(source), "frames": len(files),
        "mapping": "png_uint16 = round(original_float + offset)", "offset": offset, "gain": 1.0,
        "raw_min": global_min, "raw_max": global_max, "max_abs_rounding_error": max_error,
        "clipped_pixels": 0, "resized": False, "transposed": False,
        "reason": "Existing segmentation/detection reject negative pixels. One dataset-wide offset preserves signed variation without clipping; PNG introduces <=0.5 stored-unit rounding.",
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": digest(CHECKPOINT),
        "source_config": str(CONFIG), "source_config_sha256": digest(CONFIG),
        "config": str(run_config), "config_sha256": digest(run_config),
        "config_adaptation": "Quote region.mode off so YAML reads the intended string instead of boolean False. No numeric parameter changes.",
        "model_input_note": "Unmodified checkpoint offset=0, scale=8426; global input offset is an adaptation, not camera calibration. No retraining. No per-image scaling.",
        "photometry_note": "Algorithm CSV raw_* fields and restored NPY use shifted integer-input units. Subtract offset for absolute levels in source units; background-subtracted quantities cancel constant offset but retain rounding error.",
        "records": records,
    }
    write_json(out / "conversion_manifest.json", manifest)
    with (out / "conversion_manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("Converted {} frames; common offset={}; maximum error={}".format(len(files), offset, max_error), flush=True)
    commands = []
    try:
        for method in ("classical", "masked"):
            command = [sys.executable, "-B", "-u", "-m", "ionrecog", "infer", "--method", method,
                       "--input", str(converted), "--config", str(run_config), "--output", str(out / method),
                       "--device", "cpu"]
            if method == "masked":
                command += ["--checkpoint", str(CHECKPOINT)]
            commands.append(command)
            write_json(out / "commands.json", commands)
            started = time.perf_counter()
            with (out / (method + ".log")).open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=str(PROJECT), stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                code = process.wait()
            if code:
                raise RuntimeError("{} failed with exit code {}".format(method, code))
            print("{} wall seconds: {:.3f}".format(method, time.perf_counter() - started), flush=True)
    finally:
        protected_after = snapshot()
        changed = sorted(k for k in set(protected_before) | set(protected_after)
                         if protected_before.get(k) != protected_after.get(k))
        source_changes = [r["source"] for r in records if digest(source / r["source"]) != r["source_sha256"]]
        audit = {"protected_files_checked": len(protected_before), "changed_protected_files": changed,
                 "original_frames_checked": len(records), "changed_original_frames": source_changes,
                 "algorithms_configs_checkpoint_unchanged": not changed, "original_data_unchanged": not source_changes}
        write_json(out / "integrity_audit.json", audit)
        if changed or source_changes:
            raise RuntimeError("Integrity differences found; inspect integrity_audit.json")
    print("Complete: {}".format(out), flush=True)


if __name__ == "__main__":
    main()
