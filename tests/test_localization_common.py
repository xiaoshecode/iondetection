"""Prepared checks for the new localizers; running these writes only temp data."""
from argparse import Namespace
import json

import numpy as np
import pytest

from ionrecog.localization_common import ensure_disk_space, prepare_output, run_prediction


def _args(tmp_path, image):
    return Namespace(input=str(image), output=str(tmp_path / "results"), config=None,
                     max_images=None, reserve_gib=0, overwrite=False)


def test_prepare_output_preserves_input_and_existing_results(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    for target in (source, source / "results", tmp_path):
        with pytest.raises(ValueError):
            prepare_output(target, [source], reserve_gib=0)
    result = prepare_output(tmp_path / "new", [source], reserve_gib=0)
    (result / "checkpoint.txt").write_text("keep")
    with pytest.raises(ValueError):
        prepare_output(result, reserve_gib=0)
    assert (result / "checkpoint.txt").read_text() == "keep"


@pytest.mark.parametrize("reserve", [-1, float("nan"), float("inf")])
def test_invalid_disk_reserve(tmp_path, reserve):
    with pytest.raises(ValueError):
        ensure_disk_space(tmp_path, reserve)


def test_export_retains_subpixels_diagnostics_and_raw_photometry(tmp_path):
    source = tmp_path / "input.npy"
    raw = np.full((32, 32), 30, dtype=np.uint16)
    raw[13, 12] = 1000
    np.save(source, raw)
    args = _args(tmp_path, source)

    def predictor(image, mask):
        assert image[13, 12] == 1000
        assert mask is None
        return {"ions": [{"x": 12.25, "y": 13.5, "score": .9},
                         {"x": -1., "y": 5.}],
                "diagnostics": {"optimizer_converged": False}}

    out = run_prediction(args, predictor, "test_method", {"supervision": "synthetic"})
    data = json.loads((out / "images" / "input.npy_ions.json").read_text())
    assert len(data["ions"]) == 1
    assert data["ions"][0]["x"] == 12.25
    assert data["ions"][0]["y"] == 13.5
    assert data["ions"][0]["raw_peak"] == 1000
    assert data["diagnostics"]["optimizer_converged"] is False
    assert data["rejected"]["outside_image"] == 1
    assert np.array_equal(np.load(source), raw)
    assert (out / "images" / "input.npy_overlay.png").exists()


def test_no_reliable_region_is_abstention_without_model_call(tmp_path):
    source = tmp_path / "input.npy"
    np.save(source, np.zeros((32, 32), dtype=np.uint16))
    config = tmp_path / "config.yaml"
    config.write_text("region:\n  mode: auto\n  min_area: 8\n")
    args = _args(tmp_path, source)
    args.config = str(config)

    def predictor(image, mask):
        raise AssertionError("No reliable crystal: do not call predictor")

    out = run_prediction(args, predictor, "test_method")
    data = json.loads((out / "images" / "input.npy_ions.json").read_text())
    assert data["ions"] == []
    assert data["has_abstentions"] is True
    assert data["diagnostics"]["reason"] == "no_reliable_crystal_region"


def test_inference_cannot_overwrite_its_checkpoint(tmp_path):
    source = tmp_path / "input.npy"
    np.save(source, np.ones((12, 12), dtype=np.uint16))
    checkpoint_dir = tmp_path / "model"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "best.pt"
    checkpoint.write_bytes(b"untouched")
    args = _args(tmp_path, source)
    args.output = str(checkpoint_dir)
    args.checkpoint = str(checkpoint)
    args.overwrite = True
    with pytest.raises(ValueError):
        run_prediction(args, lambda raw, mask: {"ions": []}, "test_method")
    assert checkpoint.read_bytes() == b"untouched"


def test_nonfinite_predictions_fail_instead_of_exporting_invalid_json(tmp_path):
    source = tmp_path / "input.npy"
    np.save(source, np.ones((12, 12), dtype=np.uint16))
    with pytest.raises(ValueError, match="finite"):
        run_prediction(_args(tmp_path, source),
                       lambda raw, mask: {"ions": [{"x": float("nan"), "y": 2}]}, "test_method")


def test_failed_overwrite_invalidates_previous_complete_summary(tmp_path):
    source = tmp_path / "input.npy"
    np.save(source, np.ones((12, 12), dtype=np.uint16))
    args = _args(tmp_path, source)
    out = tmp_path / "results"
    out.mkdir()
    (out / "summary.json").write_text('{"complete": true, "images_completed": 20}')
    args.overwrite = True

    def fails(raw, mask):
        raise ValueError("Deliberate predictor failure")

    with pytest.raises(ValueError, match="predictor failure"):
        run_prediction(args, fails, "new_method")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["complete"] is False
    assert summary["images_completed"] == 0
    assert summary["method"] == "new_method"
