"""Create a tiny independently labelled VLM coordinate-check dataset.

Four 96x96 scenes cover nine separated spots, a near pair, unequal-brightness
spots, and empty background. Labels come from the generating coordinates, not
from a detector or a VLM. This is an idealized Gaussian/Poisson/Gaussian-noise
test, not a full EMCCD, trapped-ion dynamics, or micromotion simulation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

DEFAULT_OUTPUT = PROJECT / "runs/vlm_synthetic_20261103/preparation"


def scene_specs():
    """Return fresh deterministic specifications; units are original pixels."""
    return [
        {"id": "synthetic_separated_nine", "description": "Nine clear separated Gaussian emitters",
         "centers": [[20.25, 20.65], [47.4, 21.15], [74.1, 20.35],
                     [21.3, 47.75], [48.25, 48.1], [73.6, 47.2],
                     [20.8, 74.4], [47.6, 73.7], [74.65, 74.2]],
         "amplitudes": [900.] * 9, "sigma": 2.0},
        {"id": "synthetic_near_pair", "description": "Two nearby partly overlapping Gaussian emitters",
         "centers": [[44.25, 47.5], [51.1, 48.2]], "amplitudes": [900., 750.], "sigma": 2.0},
        {"id": "synthetic_weak_spots", "description": "Four emitters with two weak relative to the bright pair",
         "centers": [[24.5, 23.2], [70.2, 24.1], [25.8, 71.6], [69.4, 69.8]],
         "amplitudes": [900., 900., 90., 220.], "sigma": 2.0},
        {"id": "synthetic_background", "description": "Background with Poisson and Gaussian read noise, no emitters",
         "centers": [], "amplitudes": [], "sigma": 2.0},
    ]


def generate_scene(spec, seed, shape=(96, 96)):
    """Ideal emitter expectation sampled at pixel centers, plus independent noise.

    We sample the Gaussian at centers rather than integrate over pixel areas.
    Output units are idealized ADU at one ADU per Poisson count, plus black level.
    The center labels are exact generator parameters, not estimated peak pixels.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    yy, xx = np.indices(shape, dtype=np.float64)
    expectation = 40. + 4. * xx / (shape[1] - 1)
    for (x, y), amplitude in zip(spec["centers"], spec["amplitudes"]):
        expectation += amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * spec["sigma"] ** 2))
    raw_float = 100. + rng.poisson(expectation) + rng.normal(0., 3., size=shape)
    return np.rint(np.clip(raw_float, 0, 65535)).astype(np.uint16)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def prepare(output=DEFAULT_OUTPUT, seed=20261103):
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output is not an empty directory; choose a new path")
    raw_dir, truth_dir = output / "raw", output / "truth"
    raw_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    records, previews = [], []
    display = {"lo": 100., "hi": 1100., "scope": "fixed_for_all_synthetic_scenes_preview_only"}
    for index, spec in enumerate(scene_specs()):
        raw = generate_scene(spec, int(seed) + index)
        raw_path = raw_dir / (spec["id"] + ".png")
        truth_path = truth_dir / (spec["id"] + ".json")
        Image.fromarray(raw).save(raw_path)
        _write_json(truth_path, spec["centers"])
        records.append({"id": spec["id"], "path": str(raw_path),
            "relative_path": "raw/" + raw_path.name,
            "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "shape": list(raw.shape), "group": spec["id"], "role": "synthetic_validation",
            "selection_reason": "predefined_synthetic_coordinate_check",
            "selection_roi": None, "roi": None, "display": display,
            "ground_truth_available": True, "truth_path": str(truth_path),
            "truth_sha256": hashlib.sha256(truth_path.read_bytes()).hexdigest(),
            "ground_truth_origin": "Independent generating center coordinates; not algorithm predictions",
            "generator": dict(spec, scene_seed=int(seed) + index)})
        gray = np.rint(np.clip((raw.astype(float) - display["lo"]) / (display["hi"] - display["lo"]), 0, 1) * 255).astype(np.uint8)
        previews.append(Image.fromarray(gray).convert("RGB").resize((288, 288), Image.Resampling.NEAREST))
    sheet = Image.new("RGB", (4 * 304, 352), "#141b23")
    draw = ImageDraw.Draw(sheet)
    for index, (record, preview) in enumerate(zip(records, previews)):
        sheet.paste(preview, (index * 304 + 8, 8))
        draw.text((index * 304 + 8, 303), record["id"].replace("synthetic_", ""), fill="white")
        draw.text((index * 304 + 8, 323), "Generator truth: %d emitters" % len(record["generator"]["centers"]), fill="#a5e2aa")
    sheet.save(output / "contact_sheet.png")
    manifest = {"version": 1, "data_root": str(raw_dir), "images": records,
        "counts_by_role": {"synthetic_validation": 4}, "contact_sheet": "contact_sheet.png",
        "recommended_inference_display": {"display_low": display["lo"], "display_high": display["hi"]},
        "generator_parameters": {"base_seed": int(seed), "rng": "numpy.random.Generator(PCG64)",
            "numpy_version": np.__version__, "shape": [96, 96], "black_level_adu": 100.,
            "background_expected_counts": "40 + 4*x/(width-1)", "adu_per_count": 1.,
            "gaussian_read_noise_sd_adu": 3., "quantization": "round and clip to uint16",
            "pixel_response": "sample Gaussian at pixel center, no pixel area integration"},
        "note": "Tiny synthetic coordinate/empty-field diagnostic only, not experimental accuracy. All truth comes from generator parameters. Not a full EMCCD or micromotion simulation. No model inference occurs in this script. Use separate output for VLM predictions and compare only complete coverage."}
    _write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args(argv)
    try:
        manifest = prepare(args.output, args.seed)
    except (ValueError, OSError) as exc:
        parser.exit(2, "Error: %s\n" % exc)
    print("Generated %d independent-truth synthetic scenes; no inference performed." % len(manifest["images"]))
    print("Manifest:", args.output.resolve() / "manifest.json")
    print("Use --display-low 100 --display-high 1100 for a shared display across all four scenes.")


if __name__ == "__main__":
    main()
