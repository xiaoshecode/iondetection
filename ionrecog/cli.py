"""Command line entry points. Image outputs never replace source acquisitions."""
import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .data import find_groups, list_images, load_image
from .detection import detect_ions, save_overlay
from .reference import compare_reference, measure_sites
from .validation import validate_splits

DEFAULTS = {
    "data": {"offset": 0., "scale": "auto", "patch_size": 64, "max_target_frames": 8},
    "model": {"base_channels": 16},
    "train": {"epochs": 10, "steps_per_epoch": 100, "val_steps": 10, "batch_size": 4,
              "lr": .001, "seed": 42, "mask_fraction": .02, "num_threads": 4},
    "diffusion": {"timesteps": 100, "sampling_steps": 20, "schedule": "cosine"},
    "detection": {"sigma": 1., "background_sigma": 10., "threshold_sigma": 4.,
                  "min_area": 3, "min_distance": 4., "radius": 3., "max_ions": 5000},
    "reference": {"bright_ratio": .5, "dim_ratio": .15},
    "inference": {"tile_size": 256, "overlap": 32},
}


def load_config(path=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Configuration must be a YAML mapping.")
        for section, options in value.items():
            if section not in cfg or not isinstance(options, dict):
                raise ValueError("Unknown or invalid configuration section: " + section)
            cfg[section].update(options)
    return cfg


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, empty_fields=("id", "x", "y", "status")):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else list(empty_fields)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def image_paths(source):
    source = Path(source)
    files = [source] if source.is_file() else list_images(source)
    if not files:
        raise ValueError("No input images: " + str(source))
    return files


def check_destination(output, sources):
    output = Path(output).resolve()
    for source in sources:
        source = Path(source).resolve()
        if source.is_dir() and (source == output or source in output.parents or output in source.parents):
            raise ValueError("Output must be outside the input dataset: " + str(output))
        if source.is_file() and (source == output or output in source.parents):
            raise ValueError("Output must not replace or contain an input file: " + str(output))
    return output


def check_report_path(output, sources):
    target = check_destination(output, sources)
    if target.suffix.lower() != ".json":
        raise ValueError("Report output must have a .json extension.")
    return target


def run_train(args):
    if args.method == "diffusion" and not args.confirm_static:
        raise ValueError("Diffusion requires independent STATIC repeat frames per group. Verify fixed positions/states/exposure/gain, then add --confirm-static. For unrelated single images use --method masked.")
    from .learning import train_model
    cfg = load_config(args.config)
    for key, val in (("epochs", args.epochs), ("steps_per_epoch", args.steps), ("batch_size", args.batch_size)):
        if val is not None:
            cfg["train"][key] = val
    if args.patch_size is not None:
        cfg["data"]["patch_size"] = args.patch_size
    output = check_destination(args.output, (args.train, args.val))
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Training output already contains files. Choose a new --output, or use --resume with its checkpoint.")
    checkpoint = train_model(args.method, Path(args.train), Path(args.val), cfg, output,
                             device=args.device, resume=Path(args.resume) if args.resume else None)
    print("Checkpoint:", checkpoint)
    print("Training loss validates the restoration objective, NOT ion detection accuracy.")


def run_infer(args):
    cfg = load_config(args.config)
    files = image_paths(args.input)
    if args.max_images is not None:
        if args.max_images < 1:
            raise ValueError("--max-images must be >=1")
        files = files[:args.max_images]
    if args.samples < 1:
        raise ValueError("--samples must be >=1")
    if args.method != "diffusion" and args.samples != 1:
        raise ValueError("Multiple samples are only meaningful for diffusion.")
    if args.sample_steps is not None and (args.method != "diffusion" or args.sample_steps < 1):
        raise ValueError("--sample-steps must be positive and is only supported for diffusion.")
    output = check_destination(args.output, (args.input,))
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise ValueError("Inference output is not empty. Choose a new path or --overwrite to replace matching output files.")
    output.mkdir(parents=True, exist_ok=True)
    restorer = None
    if args.method != "classical":
        if not args.checkpoint:
            raise ValueError("Learned methods require --checkpoint. Train first; no pretrained experimental weights are supplied.")
        from .learning import load_restorer
        restorer = load_restorer(Path(args.checkpoint), device=args.device)
        if getattr(restorer, "method", args.method) != args.method:
            raise ValueError("Checkpoint method does not match --method.")
        if args.config:
            explicit = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
            restorer.config["inference"].update(explicit.get("inference", {}))
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8")) if args.reference else None
    summaries = []
    aggregate = output / "detections.csv"
    aggregate_fields = ("id", "x", "y", "peak", "flux", "radius", "snr", "status", "saturated", "edge",
                        "raw_flux", "raw_peak", "raw_background", "raw_snr", "roi_clipped", "image", "sample", "source")
    write_csv(aggregate, [], aggregate_fields)
    source = Path(args.input)
    for index, path in enumerate(files):
        raw = load_image(path)
        rel = path.relative_to(source) if source.is_dir() else Path(path.name)
        for sample in range(args.samples):
            start = time.perf_counter()
            seed = args.seed + index * args.samples + sample
            restored = restorer.restore(raw, seed=seed, sample_steps=args.sample_steps) if restorer else raw
            ions, mask = detect_ions(restored, cfg["detection"])
            measurements = measure_sites(raw, ions, cfg["detection"]["radius"])
            for ion, raw_info in zip(ions, measurements):
                ion.update({k: v for k, v in raw_info.items() if k.startswith("raw_") or k == "roi_clipped"})
                ion.update({"image": rel.as_posix(), "sample": sample, "source": args.method})
            stem = output / "images" / rel.parent / (rel.name + ("_sample%02d" % sample if args.samples > 1 else ""))
            stem.parent.mkdir(parents=True, exist_ok=True)
            write_json(str(stem) + "_ions.json", {"image": rel.as_posix(), "shape": list(raw.shape),
                "method": args.method, "seed": seed, "ions": ions,
                "note": "Candidate detections. SNR and sampling variability are not calibrated confidence or quantum-state fidelity."})
            write_csv(str(stem) + "_ions.csv", ions)
            save_overlay(raw, ions, str(stem) + "_overlay.png")
            Image.fromarray(mask.astype(np.uint8) * 255).save(str(stem) + "_foreground.png")
            if restorer:
                np.save(str(stem) + "_restored.npy", restored.astype(np.float32), allow_pickle=False)
            # Stream millions of detections instead of retaining every row in RAM.
            with aggregate.open("a", encoding="utf-8", newline="") as stream:
                csv.DictWriter(stream, fieldnames=aggregate_fields).writerows(ions)
            summaries.append({"image": rel.as_posix(), "sample": sample, "seed": seed,
                              "count": len(ions), "seconds": time.perf_counter() - start})
        if reference:
            state_rows = compare_reference(raw, reference, **cfg["reference"])
            ref_stem = output / "images" / rel.parent / rel.name
            write_csv(str(ref_stem) + "_reference_photometry.csv", state_rows)
            save_overlay(raw, state_rows, str(ref_stem) + "_reference_overlay.png")
        print("[%d/%d] %s: %d candidates" % (index + 1, len(files), rel, len(ions)))
    write_csv(output / "summary.csv", summaries, ("image", "sample", "seed", "count", "seconds"))
    write_json(output / "summary.json", {"method": args.method, "checkpoint": args.checkpoint,
        "config": cfg, "restoration_config": restorer.config if restorer else None,
        "images": len(files), "samples": args.samples, "frames": summaries,
        "reference": args.reference, "metrics_are_ground_truth_accuracy": False})
    print("Results:", output)


def run_reference(args):
    if not args.confirm_static:
        raise ValueError("Reference requires one aligned, static all-bright acquisition with matching camera settings. Add --confirm-static after checking this condition.")
    cfg = load_config(args.config)
    files = image_paths(args.input)
    if len(find_groups(args.input)) != 1:
        raise ValueError("Reference input must contain exactly one acquisition group; point --input at its leaf directory.")
    target = check_report_path(args.output, (args.input,))
    if target.exists():
        raise ValueError("Reference output already exists; choose a new file.")
    mean = None
    for n, path in enumerate(files, 1):
        raw = load_image(path)
        if mean is None:
            mean = raw.astype(np.float64)
        elif raw.shape != mean.shape:
            raise ValueError("Reference frames have different shapes.")
        else:
            mean += (raw - mean) / n
    sites, _ = detect_ions(mean.astype(np.float32), cfg["detection"])
    sites = measure_sites(mean.astype(np.float32), sites, cfg["detection"]["radius"])
    if not sites:
        raise ValueError("No sites detected in reference; inspect thresholds/exposure first.")
    write_json(target, {"shape": list(mean.shape), "radius": cfg["detection"]["radius"], "frames": len(files),
                        "sites": sites, "note": "Requires stable all-bright reference and identical query coordinates/settings; no registration or state calibration."})
    print("Reference:", target, "sites:", len(sites))


def build_parser():
    parser = argparse.ArgumentParser(description="Ion recognition: unlabeled restoration + region-based point localization")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate small synthetic repeated frames, without copying real acquisitions")
    demo.add_argument("--output", default="data/demo")
    demo.add_argument("--groups", type=int, default=4)
    demo.add_argument("--frames", type=int, default=6)
    demo.add_argument("--size", type=int, default=128)
    demo.add_argument("--seed", type=int, default=42)
    inspect = sub.add_parser("inspect", help="Check data formats, groups and train/validation leakage")
    inspect.add_argument("--train", default="data/train")
    inspect.add_argument("--val", default="data/val")
    inspect.add_argument("--method", choices=("masked", "diffusion"), default="masked")
    inspect.add_argument("--output", default="runs/dataset_report.json")
    train = sub.add_parser("train", help="Train masked-pixel restoration or conditional diffusion")
    train.add_argument("--method", choices=("masked", "diffusion"), required=True)
    train.add_argument("--train", default="data/train")
    train.add_argument("--val", default="data/val")
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="auto")
    train.add_argument("--config")
    train.add_argument("--epochs", type=int)
    train.add_argument("--steps", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--patch-size", type=int)
    train.add_argument("--resume")
    train.add_argument("--confirm-static", action="store_true", help="Confirm diffusion groups contain aligned static independent repeated exposures")
    infer = sub.add_parser("infer", help="Write candidate centers, overlays, masks and raw photometry")
    infer.add_argument("--input", default="data/test")
    infer.add_argument("--output", required=True)
    infer.add_argument("--method", choices=("classical", "masked", "diffusion"), default="classical")
    infer.add_argument("--checkpoint")
    infer.add_argument("--device", default="auto")
    infer.add_argument("--config")
    infer.add_argument("--reference")
    infer.add_argument("--samples", type=int, default=1)
    infer.add_argument("--sample-steps", type=int)
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--max-images", type=int)
    infer.add_argument("--overwrite", action="store_true")
    ref = sub.add_parser("reference", help="Build reference sites from same-scene all-bright frames")
    ref.add_argument("--input", required=True)
    ref.add_argument("--output", default="runs/reference_sites.json")
    ref.add_argument("--config")
    ref.add_argument("--confirm-static", action="store_true", help="Confirm one aligned all-bright reference acquisition with stable settings")
    evaluate = sub.add_parser("evaluate", help="Evaluate centers against an explicit JSON list of true [x,y] positions")
    evaluate.add_argument("--prediction", required=True)
    evaluate.add_argument("--truth", required=True)
    evaluate.add_argument("--tolerance", type=float, default=3.)
    evaluate.add_argument("--output", default="runs/evaluation.json")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from .synthetic import generate_demo
            dest = Path(args.output)
            if dest.exists() and list_images(dest):
                raise ValueError("Demo output already contains images; choose a new output.")
            generate_demo(dest, args.groups, args.frames, args.size, args.seed)
            print("Synthetic demo:", dest.resolve())
        elif args.command == "inspect":
            target = check_report_path(args.output, (args.train, args.val))
            report = validate_splits(args.train, args.val, args.method)
            write_json(target, report)
            print("Dataset valid: train=%d, val=%d. Report: %s" % (report["train"]["images"], report["val"]["images"], args.output))
        elif args.command == "train":
            run_train(args)
        elif args.command == "infer":
            run_infer(args)
        elif args.command == "reference":
            run_reference(args)
        else:
            from .evaluation import match_points
            target = check_report_path(args.output, (args.prediction, args.truth))
            pred = json.loads(Path(args.prediction).read_text(encoding="utf-8"))
            truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))
            if not isinstance(truth, list):
                raise ValueError("--truth must be JSON [[x,y],...], not an algorithm's pseudo-labels or an entire demo manifest.")
            result = match_points([[p["x"], p["y"]] for p in pred["ions"]], truth, args.tolerance)
            write_json(target, result)
            print(json.dumps(result, ensure_ascii=False))
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as error:
        parser.exit(2, "Error: %s\n" % error)


if __name__ == "__main__":
    main()
