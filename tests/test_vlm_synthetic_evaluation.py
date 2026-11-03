"""Use explicit fixture predictions to verify evaluation gates, not model quality."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_script("prepare_vlm_synthetic")
evaluation = load_script("evaluate_vlm_synthetic")


def build_fixture(tmp_path):
    manifest = prepare.prepare(tmp_path / "preparation")
    run = tmp_path / "run"
    run.mkdir()
    plan = {"images": manifest["images"], "settings": {"backend": "replay", "model": "unit_test_fixture"}}
    (run / "plan.json").write_text(json.dumps(plan))
    for entry in manifest["images"]:
        folder = run / "images" / entry["id"]
        folder.mkdir(parents=True)
        for method in evaluation.METHODS:
            value = {"image": entry["id"], "shape": entry["shape"], "method": method,
                     "coverage_complete": True, "has_abstentions": False,
                     "ions": [{"x": x, "y": y} for x, y in entry["generator"]["centers"]]}
            (folder / (method + "_ions.json")).write_text(json.dumps(value))
    return run, manifest


def change_prediction(run, ident, method, **changes):
    path = run / "images" / ident / (method + "_ions.json")
    value = json.loads(path.read_text())
    value.update(changes)
    path.write_text(json.dumps(value))


def test_partial_and_abstained_excluded_while_negative_fp_preserved(tmp_path):
    run, _ = build_fixture(tmp_path)
    change_prediction(run, "synthetic_separated_nine", "vlm_direct", coverage_complete=False, ions=[])
    change_prediction(run, "synthetic_near_pair", "vlm_direct", has_abstentions=True, ions=[])
    change_prediction(run, "synthetic_background", "vlm_direct", ions=[{"x": 30., "y": 30.}])
    (run / "images/synthetic_weak_spots/vlm_direct_ions.json").unlink()
    report = evaluation.evaluate(run)
    summary = next(row for row in report["summary"] if row["method"] == "vlm_direct" and row["tolerance_px"] == 1)
    assert summary["evaluated_images"] == 1 and summary["unavailable_images"] == 3
    assert summary["missing_prediction_images"] == summary["incomplete_images"] == summary["abstained_images"] == 1
    assert summary["tp"] == 0 and summary["fp"] == summary["negative_fp_total"] == 1
    assert summary["fn"] == 0 and summary["micro_f1"] == 0
    assert summary["status"] == "partial_evaluation"
    assert report["run_backend"] == "replay"
    unavailable = [row for row in report["images"] if row["method"] == "vlm_direct" and row["status"] == "unavailable"]
    assert len(unavailable) == 6 and all(row["tp"] is None and row["fn"] is None for row in unavailable)
    assert (run / "synthetic_evaluation.csv").exists()


def test_inputs_hash_verified_and_output_cannot_overwrite_sources(tmp_path):
    run, manifest = build_fixture(tmp_path)
    truth_path = Path(manifest["images"][0]["truth_path"])
    original = truth_path.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        evaluation.evaluate(run, truth_path)
    with pytest.raises(ValueError, match="outside raw, truth, and prediction"):
        evaluation.evaluate(run, truth_path.parent / "new_evaluation.json")
    truth_path.write_text("[]")
    with pytest.raises(ValueError, match="modified truth"):
        evaluation.evaluate(run)
    assert not (run / "synthetic_evaluation.json").exists()
    truth_path.write_bytes(original)
    report = evaluation.evaluate(run)
    assert all(row["tp"] == 15 and row["fp"] == row["fn"] == 0 for row in report["summary"])
    with pytest.raises(ValueError, match="already exists"):
        evaluation.evaluate(run)
    assert hashlib.sha256(truth_path.read_bytes()).hexdigest() == manifest["images"][0]["truth_sha256"]
