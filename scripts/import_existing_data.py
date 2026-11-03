"""Copy the audited local acquisitions without changing their raw pixels.

This is an explicit import recipe for the existing October 2026 captures,
not an automatic arbitrary-dataset splitter. Adjacent sessions are a temporal
holdout from the same experiment, NOT independent-crystal validation.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from ionrecog.data import load_image


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_verified(source, target, digest):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_hash(target) != digest:
            raise ValueError("Existing destination differs; refusing to overwrite: %s" % target)
    else:
        shutil.copy2(source, target)
    if file_hash(target) != digest:
        raise ValueError("Copy hash mismatch: %s" % target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=PROJECT.parent / "Camera Link/PCIE-Update/captures")
    parser.add_argument("--legacy-root", type=Path, default=PROJECT.parent / "Recog/CCD-Recognition/single_scan_images")
    parser.add_argument("--destination", type=Path, default=PROJECT / "data")
    args = parser.parse_args()
    destination = args.destination.resolve()
    for source_root in (args.capture_root.resolve(), args.legacy_root.resolve()):
        if destination == source_root or source_root in destination.parents or destination in source_root.parents:
            raise ValueError("Destination and original data roots must be separate")
    sessions = sorted(p for p in args.capture_root.glob("ion5_20261027_*") if p.is_dir())
    if len(sessions) < 3:
        raise ValueError("Expected at least three audited October 27 acquisition groups")
    plan = []
    metadata = []
    group_assignments = []
    diagnostics = sorted(p for p in args.capture_root.glob("ion5_20261023_*") if p.is_dir())
    for session in sessions + diagnostics:
        split = "diagnostic" if session in diagnostics else "val" if session == sessions[-2] else "test" if session == sessions[-1] else "train"
        experiment = session / (session.name + "_experiment.json")
        info = json.loads(experiment.read_text(encoding="utf-8"))
        acquisition = info.get("acquisition", {})
        if acquisition.get("demo") is not False or acquisition.get("source") != "pcie":
            raise ValueError("Source provenance differs from audited real acquisition: %s" % session)
        settings = acquisition.get("requested_camera_settings", {})
        if split != "diagnostic" and (settings.get("exposure_seconds") != 0.1 or settings.get("em_gain") != 300 or settings.get("em_enabled") is not True):
            raise ValueError("Camera settings differ from the selected training regime: %s" % session)
        images = sorted(session.glob("*.png"))
        group_assignments.append({"group": session.name, "split": split, "images": len(images), "requested_camera_settings": settings})
        for source in images:
            plan.append((source, destination / split / session.name / source.name, split, session.name))
        for source in sorted(session.glob("*.json*")):
            metadata.append((source, destination / split / session.name / source.name))
    legacy_pattern = re.compile(r"single_scan_frame\d+_\d{8}_\d{6}_\d{3}\.npy$")
    for source in sorted(args.legacy_root.glob("*.npy")):
        if legacy_pattern.fullmatch(source.name):
            plan.append((source, destination / "test/legacy_20260116" / source.name, "test", "legacy_20260116"))
    records, seen, duplicates = [], {}, []
    # Preflight every candidate before copying. Byte/pixel hashes preserve provenance.
    for source, target, split, group in plan:
        raw = load_image(source)
        if raw.min() < 0:
            raise ValueError("Expected nonnegative raw camera values: %s" % source)
        pixels = hashlib.sha256(str(raw.shape).encode() + np.ascontiguousarray(raw, dtype="<f4").tobytes()).hexdigest()
        if pixels in seen:
            duplicates.append({"source": str(source.resolve()), "identical_to": seen[pixels]})
            continue
        seen[pixels] = str(source.resolve())
        digest = file_hash(source)
        if target.exists() and file_hash(target) != digest:
            raise ValueError("Preflight: destination differs from source: %s" % target)
        records.append({"source": str(source.resolve()), "destination": str(target.relative_to(destination)),
                        "split": split, "group": group, "shape": list(raw.shape), "min": float(raw.min()), "max": float(raw.max()),
                        "bytes": source.stat().st_size, "file_sha256": digest, "pixels_sha256": pixels})
    for n, row in enumerate(records, 1):
        copy_verified(Path(row["source"]), destination / row["destination"], row["file_sha256"])
        if n % 100 == 0:
            print("Verified copies: %d/%d" % (n, len(records)), flush=True)
    for source, target in metadata:
        copy_verified(source, target, file_hash(source))
    counts = dict(collections.Counter(row["split"] for row in records))
    manifest = {"schema_version": 1, "counts": counts, "image_bytes": sum(row["bytes"] for row in records),
                "split_policy": "October27 whole chronological sessions: last=testing, penultimate=validation, previous=training; October23 different exposure kept diagnostic; January original NPY external test",
                "limitation": "Adjacent sessions from the same experiment; no claim of independent-crystal generalization. No manual labels or derived/noise-added arrays imported.",
                "groups": group_assignments, "excluded_pixel_duplicates": duplicates, "copied_metadata_files": len(metadata), "images": records}
    manifest_path = destination / "existing_import_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": counts, "duplicates_excluded": len(duplicates), "manifest": str(manifest_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
