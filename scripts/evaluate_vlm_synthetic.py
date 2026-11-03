"""Evaluate a saved VLM run against explicitly recorded synthetic generator truth.

Only synthetic_validation entries with hash-verified raw images and independent
generator coordinates are eligible. Incomplete or abstained predictions remain
unavailable and are excluded from metrics, with missing coverage reported. This
does not estimate accuracy on experimental ion images or perform model inference.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from ionrecog.evaluation import match_points

METHODS = ("classical", "classical_tiled", "vlm_direct", "vlm_checked")
TOLERANCES = (1.0, 3.0)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _reject_nonfinite(value):
    raise ValueError("Non-finite JSON constant: " + value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)


def _resolve_input(value, base, name):
    if not isinstance(value, str) or not value:
        raise ValueError("{} must be an explicit file path".format(name))
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _coordinates(value, shape, name):
    if not isinstance(value, list):
        raise ValueError(name + " must be a list of [x,y] coordinates")
    h, w = shape
    result = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point
        ):
            raise ValueError(name + " requires finite numeric [x,y] coordinates")
        x, y = point
        if not (0 <= x <= w - 1 and 0 <= y <= h - 1):
            raise ValueError(name + " contains coordinates outside image pixel centers")
        result.append([float(x), float(y)])
    return result


def validate_truth(entry, base):
    if entry.get("ground_truth_available") is not True or not isinstance(entry.get("generator"), dict):
        raise ValueError("Synthetic entry requires explicit generator truth: " + entry["id"])
    shape = entry.get("shape")
    if (not isinstance(shape, list) or len(shape) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in shape)):
        raise ValueError("Synthetic entry requires a positive integer [height,width] shape")
    truth_path = _resolve_input(entry.get("truth_path"), base, "truth_path")
    raw_path = _resolve_input(entry.get("path"), base, "raw path")
    for path, expected, name in ((truth_path, entry.get("truth_sha256"), "truth"),
                                 (raw_path, entry.get("sha256"), "raw image")):
        if not path.is_file() or digest(path) != expected:
            raise ValueError("Missing or modified {}: {}".format(name, path))
    truth = _coordinates(read_json(truth_path), shape, "truth")
    generated = _coordinates(entry["generator"].get("centers"), shape, "generator.centers")
    if truth != generated:
        raise ValueError("Truth must match independent generator.centers: " + entry["id"])
    # Evaluating a cropped run against every full-image label would be invalid.
    active_roi = entry.get("active_roi", [0, 0, shape[1], shape[0]])
    if active_roi != [0, 0, shape[1], shape[0]]:
        raise ValueError("Synthetic evaluation requires full-image coverage, not a cropped ROI")
    return truth, truth_path, raw_path


def prediction_state(path, entry):
    if not path.is_file():
        return {"reasons": ["missing_prediction"], "points": None, "sha256": None,
                "coverage_complete": None, "has_abstentions": None}
    state = {"reasons": [], "points": None, "sha256": digest(path),
             "coverage_complete": None, "has_abstentions": None}
    try:
        value = read_json(path)
        if (not isinstance(value, dict) or value.get("image") != entry["id"]
                or value.get("shape") != entry["shape"]
                or value.get("method") != path.name.removesuffix("_ions.json")):
            raise ValueError("Prediction image identity/shape/method does not match the plan and filename")
        complete, abstained = value.get("coverage_complete"), value.get("has_abstentions", False)
        if not isinstance(complete, bool) or not isinstance(abstained, bool):
            raise ValueError("Prediction coverage/abstention flags must be booleans")
        state.update(coverage_complete=complete, has_abstentions=abstained)
        if not complete:
            state["reasons"].append("incomplete_coverage")
        if abstained:
            state["reasons"].append("model_abstained")
        ions = value.get("ions")
        if not isinstance(ions, list) or any(not isinstance(p, dict) or "x" not in p or "y" not in p for p in ions):
            raise ValueError("Prediction requires ions with x/y coordinates")
        state["points"] = _coordinates([[p["x"], p["y"]] for p in ions], entry["shape"], "prediction")
    except (ValueError, TypeError, KeyError) as exc:
        state["reasons"].append("invalid_prediction")
        state["error"] = str(exc)
    return state


def aggregate_rows(rows, method, tolerance):
    subset = [row for row in rows if row["method"] == method and row["tolerance_px"] == tolerance]
    available = [row for row in subset if row["status"] == "available"]
    result = {"method": method, "tolerance_px": tolerance,
              "expected_images": len(subset), "evaluated_images": len(available),
              "unavailable_images": len(subset) - len(available),
              "missing_prediction_images": sum("missing_prediction" in row["reasons"] for row in subset),
              "incomplete_images": sum("incomplete_coverage" in row["reasons"] for row in subset),
              "abstained_images": sum("model_abstained" in row["reasons"] for row in subset),
              "invalid_prediction_images": sum("invalid_prediction" in row["reasons"] for row in subset),
              "complete_evaluation": len(available) == len(subset),
              "negative_images_evaluated": sum(row["truth_count"] == 0 for row in available),
              "negative_images_unavailable": sum(row["truth_count"] == 0 and row["status"] != "available" for row in subset)}
    if not available:
        result.update(status="unavailable", tp=None, fp=None, fn=None, precision=None, recall=None,
                      micro_f1=None, matched_points=0, mean_error_px=None, rmse_px=None,
                      all_correct_images=None, all_correct_fraction_evaluated_only=None, negative_fp_total=None)
        return result
    tp, fp, fn = (sum(row[key] for row in available) for key in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else (1. if tp + fn == 0 else 0.)
    recall = tp / (tp + fn) if tp + fn else 1.
    mean_error = sum(row["mean_error_px"] * row["tp"] for row in available if row["tp"]) / tp if tp else None
    rmse = math.sqrt(sum(row["rmse_px"] ** 2 * row["tp"] for row in available if row["tp"]) / tp) if tp else None
    correct = sum(row["all_correct"] for row in available)
    result.update(status="available" if result["complete_evaluation"] else "partial_evaluation",
                  tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                  micro_f1=2 * precision * recall / (precision + recall) if precision + recall else 0.,
                  matched_points=tp, mean_error_px=mean_error, rmse_px=rmse,
                  all_correct_images=correct, all_correct_fraction_evaluated_only=correct / len(available),
                  negative_fp_total=sum(row["fp"] for row in available if row["truth_count"] == 0))
    return result


def evaluate(run, output=None):
    run = Path(run).resolve()
    plan_path = run / "plan.json"
    plan = read_json(plan_path)
    if not isinstance(plan, dict) or not isinstance(plan.get("images"), list):
        raise ValueError("Run requires a plan.json images list")
    entries, skipped, ids = [], [], set()
    for entry in plan["images"]:
        if not isinstance(entry, dict):
            raise ValueError("Malformed plan image entry")
        if entry.get("role") != "synthetic_validation":
            skipped.append(entry.get("id"))
            continue
        ident = entry.get("id")
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", ident) or ident in ids:
            raise ValueError("Synthetic image IDs must be unique safe file names")
        ids.add(ident)
        truth, truth_path, raw_path = validate_truth(entry, run)
        entries.append((entry, truth, truth_path, raw_path))
    if not entries:
        raise ValueError("No explicit synthetic_validation truth is available; experimental images cannot be scored")
    output = Path(output).resolve() if output is not None else run / "synthetic_evaluation.json"
    if output.suffix.lower() != ".json":
        raise ValueError("Evaluation output must have a .json extension")
    csv_path = output.with_suffix(".csv")
    protected_files = [plan_path]
    protected_dirs = []
    for entry, _, truth_path, raw_path in entries:
        prediction_dir = run / "images" / entry["id"]
        protected_dirs.extend((truth_path.parent, raw_path.parent, prediction_dir))
        protected_files.extend((truth_path, raw_path))
        protected_files.extend(prediction_dir / (method + "_ions.json") for method in METHODS)
    for target in (output, csv_path):
        if target.exists():
            raise ValueError("Evaluation output already exists: " + str(target))
        if target in protected_files or any(folder == target.parent or folder in target.parents for folder in protected_dirs):
            raise ValueError("Evaluation output must be outside raw, truth, and prediction directories")
    rows, provenance = [], []
    for entry, truth, truth_path, raw_path in entries:
        image_provenance = {"id": entry["id"], "raw_path": str(raw_path), "raw_sha256": entry["sha256"],
                            "truth_path": str(truth_path), "truth_sha256": entry["truth_sha256"], "predictions": {}}
        for method in METHODS:
            path = run / "images" / entry["id"] / (method + "_ions.json")
            state = prediction_state(path, entry)
            image_provenance["predictions"][method] = {"path": str(path), "sha256": state["sha256"]}
            for tolerance in TOLERANCES:
                row = {"id": entry["id"], "method": method, "tolerance_px": tolerance,
                       "status": "unavailable" if state["reasons"] else "available",
                       "reasons": list(state["reasons"]), "truth_count": len(truth),
                       "predicted_count": len(state["points"]) if state["points"] is not None else None,
                       "coverage_complete": state["coverage_complete"], "has_abstentions": state["has_abstentions"],
                       "error": state.get("error"), "tp": None, "fp": None, "fn": None,
                       "precision": None, "recall": None, "f1": None,
                       "mean_error_px": None, "rmse_px": None, "all_correct": None}
                if not state["reasons"]:
                    row.update(match_points(state["points"], truth, tolerance))
                rows.append(row)
        provenance.append(image_provenance)
    report = {"version": 1, "scope": "synthetic_generator_truth_only", "run": str(run),
              "run_backend": plan.get("settings", {}).get("backend"),
              "model": plan.get("settings", {}).get("model"), "plan_sha256": digest(plan_path),
              "evaluated_role": "synthetic_validation", "skipped_non_synthetic_ids": skipped,
              "tolerances_px": list(TOLERANCES), "provenance": provenance, "images": rows,
              "summary": [aggregate_rows(rows, method, tolerance) for method in METHODS for tolerance in TOLERANCES],
              "note": "Independent synthetic generator truth only; not real-image accuracy. Missing, incomplete, invalid, or model-abstained predictions are unavailable, never empty detections. Partial aggregate scores cover only available images and must be read with missing/abstained counts. Error means and RMSE are weighted by matched point counts. This script performs no model inference; replay backend remains replay."}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental overwrite even if another run
    # creates either destination after the earlier path checks.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with csv_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(dict(row, reasons=";".join(row["reasons"])) for row in rows)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.run, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, "Error: %s\n" % exc)
    output = args.output.resolve() if args.output else args.run.resolve() / "synthetic_evaluation.json"
    for row in report["summary"]:
        print("%s @ %.0fpx: %s; evaluated %d/%d; TP=%s FP=%s FN=%s; micro-F1=%s" %
              (row["method"], row["tolerance_px"], row["status"], row["evaluated_images"], row["expected_images"],
               row["tp"], row["fp"], row["fn"], row["micro_f1"]))
    print("Evaluation:", output)
    print("CSV:", output.with_suffix(".csv"))


if __name__ == "__main__":
    main()
