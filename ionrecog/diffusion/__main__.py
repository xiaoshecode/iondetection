"""Standalone CLI: python -m ionrecog.diffusion COMMAND --help."""
import argparse
import copy
import json
import os
from pathlib import Path

from .config import load_config


def inference_config(path, checkpoint):
    if path or not checkpoint:
        return load_config(path)
    import torch
    # Only load your own/trusted training checkpoints (pickle on old torch).
    try:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        saved = torch.load(checkpoint, map_location="cpu")
    return copy.deepcopy(saved["config"])


def prepare_repeats(root, output, confirm_static=False):
    from ..data import list_images
    from .data import audit_manifest
    if not confirm_static:
        raise ValueError("Pass --confirm-static only after checking repeated frames have stable geometry, state, exposure and gain")
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise ValueError("root must contain train/val/test group directories")
    if output.exists():
        raise FileExistsError(output)
    records = []
    for split in ("train", "val", "test"):
        folder = root / split
        if not folder.exists():
            continue
        files = list_images(folder)
        groups = {}
        for image in files:
            if image.parent == folder:
                raise ValueError("Place repeat frames in named group subdirectories")
            group = image.parent.relative_to(folder).as_posix()
            groups.setdefault(group, []).append(image)
        for group, images in sorted(groups.items()):
            records.append({"id": split + "/" + group, "split": split, "group_id": group,
                            "stable": True,
                            "frames": [os.path.relpath(p, output.parent).replace(os.sep, "/") for p in images]})
    if not records:
        raise ValueError("No groups found")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".partial")
    if temp.exists():
        raise FileExistsError("Temporary manifest already exists: " + str(temp))
    try:
        temp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        audit_manifest(temp)
        os.replace(str(temp), str(output))
    finally:
        if temp.exists():
            temp.unlink()
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Conditional diffusion restoration and frozen FLUX spatial-feature experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("synthetic", help="Generate physically approximated paired data and exact point labels")
    p.add_argument("--output", required=True)
    p.add_argument("--train-groups", type=int, default=8)
    p.add_argument("--val-groups", type=int, default=2)
    p.add_argument("--test-groups", type=int, default=2)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("prepare-repeats", help="Index already split, stable repeat groups; no random frame splitting")
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--confirm-static", action="store_true")
    p = sub.add_parser("validate", help="Validate manifest, image shapes and split/hash leakage")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output")
    p = sub.add_parser("train", help="Train or resume; also supports torchrun DDP")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config")
    p.add_argument("--device", default="auto")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--resume")
    group.add_argument("--init-checkpoint", help="Initialize EMA weights only for a new fine-tuning run")
    for command in ("infer", "evaluate"):
        p = sub.add_parser(command, help="Batch prediction" if command == "infer" else "Held-out restoration and point evaluation")
        p.add_argument("--input" if command == "infer" else "--manifest", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--checkpoint")
        p.add_argument("--config", help="Omit to use checkpoint settings; raw baseline requires its own config")
        p.add_argument("--device", default="auto")
        p.add_argument("--mode", choices=("raw", "restored", "fused"), default="restored")
        p.add_argument("--raw-weight", type=float, default=0.5)
        p.add_argument("--samples", type=int, default=1)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--reserve-gib", type=float, default=2.0)
        if command == "evaluate":
            p.add_argument("--split", choices=("train", "val", "test"), default="test")
            p.add_argument("--tolerance", type=float, default=2.0)
            p.add_argument("--save-images", action="store_true")
    p = sub.add_parser("extract-features", help="Frozen FLUX.1-Kontext-dev editing-trajectory spatial features")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True, help="Local diffusers checkpoint directory or authorized HF model id")
    p.add_argument("--device", default="cuda")
    p.add_argument("--layers", type=int, nargs="+", default=[0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--capture-step", type=int, default=0)
    p.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    p.add_argument("--prompt", default="Preserve the image exactly.")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--cpu-offload", action="store_true")
    p.add_argument("--allow-download", action="store_true", help="Explicitly permit model downloads")
    p.add_argument("--intensity-offset", type=float, default=0.0)
    p.add_argument("--intensity-scale", type=float, required=True, help="Fixed ADU scale shared across compared images")
    p = sub.add_parser("compare-features", help="Offline token cosine maps and correspondences; not ion identities")
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--stream", choices=("reference", "target"), default="reference")
    p.add_argument("--normalization", choices=("l2", "channel_standardize"), default="l2")
    p.add_argument("--one-to-one", action="store_true")
    p.add_argument("--mutual", action="store_true")
    p.add_argument("--min-similarity", type=float, default=-1.0)
    args = parser.parse_args(argv)
    try:
        if args.command == "synthetic":
            from .synthetic import make_synthetic_dataset
            result = make_synthetic_dataset(args.output, args.train_groups, args.val_groups, args.test_groups, args.size, args.seed)
        elif args.command == "prepare-repeats":
            result = prepare_repeats(args.root, args.output, args.confirm_static)
        elif args.command == "validate":
            from .data import audit_manifest
            from .pipeline import write_json
            result = audit_manifest(args.manifest)
            if args.output:
                path = Path(args.output)
                if path.exists():
                    raise FileExistsError(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json(path, result)
            result = json.dumps(result, indent=2, ensure_ascii=False)
        elif args.command == "train":
            from .engine import train
            result = train(load_config(args.config), args.manifest, args.output, device=args.device,
                           resume=args.resume, init_checkpoint=args.init_checkpoint)
        elif args.command in ("infer", "evaluate"):
            from .pipeline import infer, evaluate
            if args.mode != "raw" and not args.checkpoint:
                raise ValueError("--checkpoint is required for restored/fused modes")
            config = inference_config(args.config, args.checkpoint)
            kwargs = {"config": config, "checkpoint": args.checkpoint, "device": args.device,
                      "samples": args.samples, "seed": args.seed, "mode": args.mode,
                      "raw_weight": args.raw_weight, "reserve_gib": args.reserve_gib}
            if args.command == "infer":
                result = infer(args.input, args.output, **kwargs)
            else:
                result = evaluate(args.manifest, args.output, split=args.split, tolerance=args.tolerance,
                                  save_images=args.save_images, **kwargs)
        elif args.command == "extract-features":
            from .features import extract_features
            result = extract_features(args.input, args.output, args.model, device=args.device,
                    layers=tuple(args.layers), seed=args.seed, steps=args.steps, precision=args.precision,
                    local_files_only=not args.allow_download, prompt=args.prompt, width=args.width,
                    height=args.height, capture_step=args.capture_step, cpu_offload=args.cpu_offload,
                    intensity_offset=args.intensity_offset, intensity_scale=args.intensity_scale)
        else:
            from .features import compare_features
            result = compare_features(args.source, args.target, args.output, layer=args.layer,
                    stream=args.stream, normalization=args.normalization, one_to_one=args.one_to_one,
                    mutual=args.mutual, min_similarity=args.min_similarity)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, "error: %s\n" % error)
    if int(os.environ.get("RANK", "0")) == 0:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
