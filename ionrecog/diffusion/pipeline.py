"""Batch restoration, raw-supported localization and explicitly labelled evaluation."""
import csv
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np

from ..classical.detection import detect_ions, save_overlay
from ..classical.region import segment_crystal
from ..data import list_images, load_image
from ..evaluation import match_points


def write_json(path, value):
    path = Path(path)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(str(temporary), str(path))


def check_space(path, estimated=0, reserve_gib=2.0):
    if not math.isfinite(reserve_gib) or reserve_gib < 0:
        raise ValueError("reserve_gib must be finite and nonnegative")
    location = Path(path).resolve()
    while not location.exists():
        location = location.parent
    if shutil.disk_usage(location).free < estimated + reserve_gib * 1024**3:
        raise ValueError("Insufficient disk space for output and reserve")


def new_output(path, sources=(), reserve_gib=2.0):
    output = Path(path).resolve()
    for source in sources:
        source = Path(source).resolve()
        if source == output or output in source.parents or (source.is_dir() and source in output.parents):
            raise ValueError("Output must be separate from input files/directories")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Choose a new or empty output directory: " + str(output))
    check_space(output, reserve_gib=reserve_gib)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _coordinates(ions):
    return [[float(i["x"]), float(i["y"])] for i in ions]


def predict_frame(observed, restorer, config, seed=42, samples=1, mode="restored", raw_weight=0.5):
    """Keep stochastic samples separate. Fused mode is an intensity blend, not a trained head."""
    if samples < 1 or mode not in ("raw", "restored", "fused"):
        raise ValueError("samples >= 1 and mode raw/restored/fused required")
    if not math.isfinite(raw_weight) or not 0 <= raw_weight <= 1:
        raise ValueError("raw_weight must be in [0, 1]")
    if mode != "raw" and restorer is None:
        raise ValueError("A diffusion checkpoint is required")
    allowed, region_info = segment_crystal(observed, config.get("region"))
    raw_ions, _ = detect_ions(observed, config.get("detection"), allowed, observed)
    outputs = []
    for index in range(1 if mode == "raw" else samples):
        start = time.perf_counter()
        restored = observed if mode == "raw" else restorer.restore(
            observed, seed=seed + index, sampling_steps=config["diffusion"]["sampling_steps"],
            tile_size=config["inference"]["tile_size"], overlap=config["inference"]["overlap"])
        image = raw_weight * observed + (1 - raw_weight) * restored if mode == "fused" else restored
        ions, _ = detect_ions(image, config.get("detection"), allowed, observed)
        outputs.append({"ions": ions, "restored": restored, "seed": seed + index,
                        "seconds": time.perf_counter() - start})
    return {"samples": outputs, "raw_ions": raw_ions, "region": region_info, "mask": allowed}


def _save_case(output, name, observed, prediction, save_arrays=True):
    destination = output / name
    destination.mkdir()
    for index, sample in enumerate(prediction["samples"]):
        prefix = "sample_%03d" % index
        if save_arrays:
            np.save(str(destination / (prefix + "_restored.npy")), sample["restored"].astype(np.float32))
        save_overlay(observed, sample["ions"], destination / (prefix + "_overlay.png"), prediction["mask"])
        with (destination / (prefix + "_points.csv")).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["x", "y", "snr", "flux"])
            writer.writeheader()
            writer.writerows({k: ion[k] for k in writer.fieldnames} for ion in sample["ions"])
    record = {"region": prediction["region"], "raw_points": _coordinates(prediction["raw_ions"]),
              "samples": [{k: v for k, v in sample.items() if k != "restored"}
                          for sample in prediction["samples"]]}
    write_json(destination / "prediction.json", record)


def infer(input_path, output, config, checkpoint=None, device="auto", samples=1,
          seed=42, mode="restored", raw_weight=0.5, reserve_gib=2.0):
    from .engine import Restorer
    paths = list_images(input_path)
    if not paths:
        raise ValueError("No images found")
    sources = [input_path] + ([checkpoint] if checkpoint else [])
    destination = new_output(output, sources, reserve_gib)
    restorer = None if mode == "raw" else Restorer(checkpoint, device=device)
    if restorer is not None and any(config["data"][k] != restorer.config["data"][k] for k in ("offset", "scale")):
        raise ValueError("Inference offset/scale must match checkpoint")
    rows = []
    write_json(destination / "settings.json", {"config": config, "checkpoint": str(checkpoint),
               "seed": seed, "samples": samples, "mode": mode, "raw_weight": raw_weight})
    for index, path in enumerate(paths):
        raw = load_image(path)
        check_space(destination, raw.nbytes * max(1, samples) * 3, reserve_gib)
        prediction = predict_frame(raw, restorer, config, seed + index * 1009, samples, mode, raw_weight)
        name = "%06d_%s" % (index, path.stem)
        _save_case(destination, name, raw, prediction)
        rows.append({"input": str(path.resolve()), "result": name + "/prediction.json",
                     "counts": [len(s["ions"]) for s in prediction["samples"]]})
        write_json(destination / "manifest.json", {"complete": index == len(paths)-1, "images": rows})
        print("infer %d/%d %s" % (index+1, len(paths), path.name), flush=True)
    return destination / "manifest.json"


def restoration_metrics(image, target, data_range):
    mse = float(np.mean((image.astype(np.float64) - target) ** 2))
    return {"mse_adu2": mse, "mae_adu": float(np.mean(np.abs(image.astype(np.float64)-target))),
            "psnr_db": 10 * math.log10(data_range**2 / mse) if mse > 0 else None,
            "identical": mse == 0}


def summarize_localization(rows):
    """No score is invented for unlabelled images. Every stochastic sample has equal weight."""
    if not rows:
        return {"labelled_predictions": 0, "precision": None, "recall": None, "f1": None}
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    precision = tp/(tp+fp) if tp+fp else (1.0 if not fn else 0.0)
    recall = tp/(tp+fn) if tp+fn else 1.0
    return {"labelled_predictions": len(rows), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall,
            "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
            "all_correct_fraction": sum(r["all_correct"] for r in rows)/len(rows),
            "mean_matched_error_px": sum(r["mean_error_px"]*r["tp"] for r in rows if r["tp"])/tp if tp else None}


def evaluate(manifest, output, config, checkpoint=None, split="test", device="auto",
             samples=1, seed=42, mode="restored", raw_weight=0.5, tolerance=2.0,
             reserve_gib=2.0, save_images=False):
    from .data import audit_manifest, load_manifest, read_case
    from .engine import Restorer
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    audit = audit_manifest(manifest)
    records = [r for r in load_manifest(manifest) if r["split"] == split]
    if not records:
        raise ValueError("Requested split is empty")
    inputs = [manifest] + [p for r in records for p in
                           ([r["input"], r["target"]] if "input" in r else r["frames"])]
    destination = new_output(output, inputs, reserve_gib)
    restorer = None if mode == "raw" else Restorer(checkpoint, device=device)
    if restorer is not None and any(config["data"][k] != restorer.config["data"][k] for k in ("offset", "scale")):
        raise ValueError("Evaluation offset/scale must match checkpoint")
    rows, localization, raw_localization = [], [], []
    for index, record in enumerate(records):
        check_space(destination, reserve_gib=reserve_gib)
        case = read_case(record, seed=seed + index * 1009, max_target_frames=config["data"]["max_target_frames"])
        raw, target = case["observed"], case["target"]
        if save_images:
            check_space(destination, raw.nbytes * max(1, samples) * 3, reserve_gib)
        prediction = predict_frame(raw, restorer, config, seed + index*1009, samples, mode, raw_weight)
        truth = case["points"]
        baseline = match_points(_coordinates(prediction["raw_ions"]), truth, tolerance) if truth is not None else None
        if baseline is not None:
            raw_localization.append(baseline)
        results = []
        for sample in prediction["samples"]:
            metrics = match_points(_coordinates(sample["ions"]), truth, tolerance) if truth is not None else None
            if metrics is not None:
                localization.append(metrics)
            results.append({"seed": sample["seed"], "seconds": sample["seconds"],
                            "points": _coordinates(sample["ions"]), "localization": metrics,
                            "restoration": restoration_metrics(sample["restored"], target, config["data"]["scale"])})
        rows.append({"id": record["id"], "group_id": record["group_id"],
                     "input_path": case["input_path"], "target_paths": case["target_paths"],
                     "target_kind": case["target_kind"], "raw_localization": baseline,
                     "raw_restoration": restoration_metrics(raw, target, config["data"]["scale"]),
                     "region": prediction["region"], "samples": results})
        if save_images:
            _save_case(destination, "%06d" % index, raw, prediction)
        write_json(destination / "per_image.json", rows)
        print("evaluate %d/%d" % (index+1, len(records)), flush=True)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["group_id"], []).extend(s["localization"] for s in row["samples"] if s["localization"] is not None)
    report = {"complete": True, "split": split, "config": config, "data_audit": audit,
              "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
              "checkpoint": str(checkpoint), "mode": mode, "raw_weight": raw_weight,
              "sampling_seeds_base": seed, "samples": samples, "tolerance_px": tolerance,
              "images": len(rows), "labelled_images": len(raw_localization),
              "restored_localization": summarize_localization(localization),
              "raw_localization": summarize_localization(raw_localization),
              "per_group": {k: summarize_localization(v) for k, v in grouped.items()},
              "note": "Repeat means are noisy proxies, not deblurred truth. Multiple samples/groups are correlated; no best-of-K selection or independent-sample confidence interval is reported."}
    write_json(destination / "report.json", report)
    return destination / "report.json"
