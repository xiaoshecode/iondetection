"""Select a reproducible, explicitly unlabeled pilot without modifying raw data.

The October 27 validation/test acquisitions are nearby temporal holdouts, not
independent dates or crystals. The extra failure image is a diagnostic extreme,
not an unbiased sample. Display mappings are recorded but never used as raw data.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from ionrecog.data import IMAGE_SUFFIXES, load_image


def uniform_indices(length, count):
    """Deterministic evenly spaced indices, including endpoints when count > 1."""
    if length < 0 or count < 0:
        raise ValueError("length and count must be nonnegative")
    if length == 0 or count == 0:
        return []
    return np.linspace(0, length - 1, min(length, count), dtype=int).tolist()


def image_files(root, suffix=None):
    if not root.exists():
        return []
    return sorted((p for p in root.rglob("*") if p.is_file()
                   and p.suffix.lower() in (IMAGE_SUFFIXES if suffix is None else {suffix})),
                  key=lambda p: p.relative_to(root).as_posix())


def select_images(data_root, max_images=32, failure_report=None):
    """Return balanced deterministic selections with disjoint roles and paths."""
    if isinstance(max_images, bool) or not isinstance(max_images, int) or max_images < 1:
        raise ValueError("max_images must be a positive integer")
    data_root = Path(data_root).resolve()
    if not data_root.is_dir():
        raise ValueError("Data root must be an existing directory")
    categories = []
    plans = [
        ("calibration", image_files(data_root / "val"), 3, "validation_acquisition_uniform"),
        ("pilot_test", image_files(data_root / "test", ".png"), 16, "test_png_uniform"),
        ("pilot_test", image_files(data_root / "test/legacy_20260116", ".npy"), 8,
         "legacy_npy_uniform"),
        ("diagnostic", image_files(data_root / "diagnostic"), 4, "different_settings_uniform"),
    ]
    for role, paths, count, reason in plans:
        categories.append([{"path": p.resolve(), "role": role, "selection_reason": reason,
                            "selection_index": index, "selection_pool_size": len(paths)}
                           for index in uniform_indices(len(paths), count)
                           for p in [paths[index]]])
    if failure_report is not None:
        report_path = Path(failure_report)
        info = json.loads(report_path.read_text(encoding="utf-8"))
        rel = info.get("relative")
        if not isinstance(rel, str) or not rel:
            raise ValueError("Failure report requires a relative image path")
        test_root = (data_root / "test").resolve()
        failure = (test_root / rel).resolve()
        if test_root not in failure.parents or not failure.is_file() or failure.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError("Failure report must name an existing image inside data/test")
        # Never list the same image as both a test and deliberately chosen failure.
        for category in categories:
            category[:] = [row for row in category if row["path"] != failure]
        categories.append([{"path": failure, "role": "diagnostic",
                            "selection_reason": "previous_failure_extreme_not_representative",
                            "selection_detail": info.get("selection_rule", "Previously identified failure"),
                            "selection_report": str(report_path.resolve())}])
    # Round robin preserves category coverage even when --max-images is smaller.
    chosen, seen = [], set()
    for slot in range(max((len(c) for c in categories), default=0)):
        for category in categories:
            if slot >= len(category):
                continue
            row = category[slot]
            if row["path"] in seen:
                continue
            chosen.append(row)
            seen.add(row["path"])
            if len(chosen) == max_images:
                return chosen
    return chosen


def display_mapping(raw):
    lo, hi = (float(v) for v in np.percentile(raw, [0.5, 99.9]))
    if hi <= lo:
        lo, hi = float(raw.min()), float(raw.max())
    return {"lo": lo, "hi": hi, "percentiles": [0.5, 99.9],
            "scope": "per_full_image_preview_only", "constant_image_is_black": hi <= lo}


def preview(raw, mapping):
    if mapping["hi"] <= mapping["lo"]:
        return Image.fromarray(np.zeros(raw.shape, dtype=np.uint8))
    gray = np.rint(np.clip((raw - mapping["lo"]) / (mapping["hi"] - mapping["lo"]), 0, 1) * 255)
    return Image.fromarray(gray.astype(np.uint8))


def study_region(shape):
    """Predefined manual regions for this acquisition, never VLM-inferred ROIs."""
    if tuple(shape) == (512, 512):
        return [180, 220, 300, 430]
    if tuple(shape) == (300, 800):
        return [0, 86, 800, 214]
    raise ValueError("--study-regions only defines regions for audited 512x512 and 300x800 images")


def prepare(data_root, output, max_images=32, failure_report=None, study_regions=False):
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if output == data_root or data_root in output.parents or output in data_root.parents:
        raise ValueError("Output must be separate from the raw data tree")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output is not an empty directory; choose a new path")
    selections = select_images(data_root, max_images, failure_report)
    if not selections:
        raise ValueError("No images selected from the expected data directories")
    records, panels = [], []
    for selection in selections:
        path = selection["path"]
        raw = load_image(path)
        relative = path.relative_to(data_root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        mapping = display_mapping(raw)
        row = dict(selection)
        row.update({"id": "image_" + hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16],
                    "path": str(path), "relative_path": relative, "sha256": digest,
                    "shape": list(raw.shape), "group": path.parent.name,
                    "roi": study_region(raw.shape) if study_regions else None,
                    "display": mapping, "ground_truth_available": False,
                    "full_frame_evaluation": not study_regions})
        if study_regions:
            row["roi_origin"] = "predefined_manual_study_region_shared_by_all_methods"
            row["roi_note"] = ("512x512 uses the historical manual ROI, including different-settings diagnostic images where it may miss the crystal. 300x800 uses a predefined central 128-pixel horizontal band. These are partial-image experiments, not automatic crystal-region detection or full-image evaluation.")
        if raw.shape == (512, 512) and selection["selection_reason"] in {
                "test_png_uniform", "validation_acquisition_uniform", "previous_failure_extreme_not_representative"}:
            row["selection_roi"] = [180, 220, 300, 430]
            row["selection_roi_note"] = "Historical manual display ROI only; not a VLM result, not a ground-truth crystal boundary, not applied to selection or inference."
        records.append(row)
        panel = preview(raw, mapping).convert("RGB")
        panel.thumbnail((256, 256), Image.Resampling.LANCZOS)
        if study_regions:
            box = row["roi"]
            sx, sy = panel.width / raw.shape[1], panel.height / raw.shape[0]
            # Half-open input ROI: show the first/last included pixel positions.
            ImageDraw.Draw(panel).rectangle((box[0] * sx, box[1] * sy,
                (box[2] - 1) * sx, (box[3] - 1) * sy), outline="#4fc3ff", width=2)
        panels.append(panel)
    columns, panel_width, panel_height = 4, 300, 310
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * panel_width, rows * panel_height + 48), "#141b23")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), "VLM pilot: UNLABELED raw-image previews; no detection or accuracy claims", fill="white")
    subtitle = ("BLUE BOX = predefined MANUAL study region, shared by all methods. PARTIAL images; not full-frame evaluation."
                if study_regions else "Per-image display ranges are saved in manifest.json. Input is the full original image.")
    draw.text((12, 27), subtitle, fill="#b8c6d4")
    colors = {"calibration": "#8ec5ff", "pilot_test": "#a5e2aa", "diagnostic": "#ffd194"}
    for i, (record, panel) in enumerate(zip(records, panels)):
        x, y = (i % columns) * panel_width, (i // columns) * panel_height + 48
        sheet.paste(panel, (x + (panel_width - panel.width) // 2, y + (256 - panel.height) // 2))
        draw.text((x + 8, y + 260), "%02d %s" % (i + 1, record["role"]), fill=colors[record["role"]])
        draw.text((x + 8, y + 276), record["id"], fill="white")
        draw.text((x + 8, y + 291), record["group"][:39], fill="#b8c6d4")
    manifest = {"version": 1, "data_root": str(data_root), "max_images": max_images,
                "images": records, "counts_by_role": dict(Counter(r["role"] for r in records)),
                "study_regions": bool(study_regions), "full_frame_evaluation": not study_regions,
                "study_region_policy": ("Predefined manual rectangles: 512x512 [180,220,300,430); 300x800 [0,86,800,214). Shared by all comparison methods; no VLM-generated ROI. Diagnostic image regions may not cover a crystal."
                                        if study_regions else None),
                "selection_policy": "Up to 3 validation, 16 test PNG, 8 legacy NPY, 4 different-settings diagnostic, and 1 prior failure; equal-interval indices within sorted pools, round-robin cap.",
                "note": "No ground-truth labels. Calibration uses val only. Existing October 27 test/val are nearby temporal holds, not independent crystal/date generalization. Failure example is a deliberately selected diagnostic extreme. Raw files are referenced without copying or alteration. Do not report candidate counts as accuracy.",
                "contact_sheet": "contact_sheet.png"}
    output.mkdir(parents=True, exist_ok=True)
    sheet.save(str(output / "contact_sheet.png"))
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--failure-report", type=Path)
    parser.add_argument("--study-regions", action="store_true",
                        help="Use fixed manual partial-image study regions for the audited 512x512 and 300x800 images; identical selections, not automatic ROI detection")
    args = parser.parse_args(argv)
    failure_report = args.failure_report
    default_report = PROJECT / "runs/existing_20261103/additional_failure_case.json"
    if failure_report is None and args.data_root.resolve() == (PROJECT / "data").resolve() and default_report.is_file():
        failure_report = default_report
    try:
        report = prepare(args.data_root, args.output, args.max_images, failure_report, args.study_regions)
    except (ValueError, OSError) as exc:
        parser.exit(2, "Error: %s\n" % exc)
    print("Selected %d unlabeled images: %s" % (len(report["images"]), report["counts_by_role"]))
    print("Manifest:", args.output.resolve() / "manifest.json")
    print("Contact sheet:", args.output.resolve() / "contact_sheet.png")


if __name__ == "__main__":
    main()
