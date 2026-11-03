"""Failure-inclusive metrics and audit artifacts; no model weights required."""
import copy
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from ionrecog.vision_language.training_evaluation import evaluate_predictions


def fixture_records(tmp_path, points=None):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (64, 32), color=(20, 20, 20)).save(dataset / "tile.png")
    records = [{"id": "tile_001", "image": "tile.png", "points": [[31.5, 15.5]] if points is None else points,
                "tile": [100, 200, 164, 232], "split": "validation", "label_source": "classical_pseudo",
                "classical_pseudo": True}]
    return dataset, records


def reply(text='{"points":[[500,500]],"uncertain":false}', finish="stop"):
    return {"text": text, "finish_reason": finish, "seconds": 0.25}


def strict(report, tolerance=1):
    return next(row for row in report["summary"] if row["scope"] == "strict" and row["tolerance_px"] == tolerance)


def test_original_local_coordinates_and_one_to_one_matching(tmp_path):
    dataset, records = fixture_records(tmp_path)
    response = reply('{"points":[[500,500],[500,500]],"uncertain":false}')
    output = tmp_path / "pre"
    report = evaluate_predictions(records, {"tile_001": response}, output, dataset)
    assert strict(report)["tp"] == 1
    assert strict(report)["fp"] == 1
    assert strict(report)["fn"] == 0
    assert not strict(report)["all_correct"]
    parsed = json.loads((output / "images/tile_001/parsed_points.json").read_text(encoding="utf-8"))
    assert parsed["strict_points"] == [[31.5, 15.5], [31.5, 15.5]]
    assert (output / "images/tile_001/overlay.png").is_file()
    assert (output / "summary.csv").is_file() and (output / "metrics.csv").is_file()
    assert report["classical_pseudo_images"] == 1
    assert report["metric_interpretation"] == "reference_agreement_not_experimental_accuracy"


@pytest.mark.parametrize("response,status", [
    (reply("not json"), "malformed"),
    (reply(finish="length"), "truncated"),
    (reply('{"points":[[500,500]],"uncertain":true}'), "abstained"),
    (reply(finish=None), "malformed"),
    (reply(finish="backend_error"), "malformed"),
])
def test_failures_never_disappear_from_strict_denominator(tmp_path, response, status):
    dataset, records = fixture_records(tmp_path)
    second = copy.deepcopy(records[0])
    second["id"] = "tile_002"
    records.append(second)
    report = evaluate_predictions(records, {"tile_001": response, "tile_002": reply()}, tmp_path / "result", dataset)
    assert report["total_images"] == 2 and report["failure_images"] == 1
    assert report["status_counts"][status] == 1
    summary = strict(report)
    assert summary["evaluated_images"] == 2 and summary["excluded_images"] == 0
    assert summary["tp"] == summary["fn"] == 1 and summary["fp"] == 0
    assert summary["all_correct_fraction"] == 0.5
    assert summary["micro_f1"] == pytest.approx(2 / 3)
    valid = next(row for row in report["summary"] if row["scope"] == "valid_only" and row["tolerance_px"] == 1)
    assert valid["evaluated_images"] == valid["excluded_images"] == 1
    assert valid["selection_bias"] and valid["micro_f1"] == 1


@pytest.mark.parametrize("response,status", [
    (reply("bad json"), "malformed"),
    (reply('{"points":[],"uncertain":true}'), "abstained"),
    (reply('{"points":[]}', "max_new_tokens"), "truncated"),
])
def test_empty_reference_failure_is_not_all_correct(tmp_path, response, status):
    dataset, records = fixture_records(tmp_path, [])
    report = evaluate_predictions(records, {"tile_001": response}, tmp_path / "result", dataset)
    assert report["images"][0]["status"] == status
    assert not strict(report)["all_correct"]
    assert strict(report)["all_correct_fraction"] == 0
    assert strict(report)["micro_f1"] == 0
    assert strict(report)["tp"] == strict(report)["fp"] == strict(report)["fn"] == 0
    assert all(not row["all_correct"] for row in report["metrics"])
    valid = next(row for row in report["summary"] if row["scope"] == "valid_only")
    assert valid["micro_f1"] is None and valid["evaluated_images"] == 0


def test_valid_negative_and_normalized_endpoints(tmp_path):
    dataset, records = fixture_records(tmp_path, [[0, 0], [63, 31]])
    response = reply('{"points":[[0,0],[1000,1000]]}')
    report = evaluate_predictions(records, {"tile_001": response}, tmp_path / "positive", dataset)
    assert strict(report)["all_correct"] and strict(report)["tp"] == 2
    records[0]["points"] = []
    report = evaluate_predictions(records, {"tile_001": reply('{"points":[]}')}, tmp_path / "negative", dataset)
    assert strict(report)["all_correct"] and strict(report)["micro_f1"] == 1


def test_tolerances_use_original_pixels(tmp_path):
    dataset, records = fixture_records(tmp_path, [[29.5, 15.5]])
    report = evaluate_predictions(records, {"tile_001": reply()}, tmp_path / "result", dataset)
    assert strict(report, 1)["tp"] == 0
    assert strict(report, 1)["fn"] == strict(report, 1)["fp"] == 1
    assert strict(report, 3)["tp"] == 1 and strict(report, 3)["all_correct"]
    assert strict(report, 3)["rmse_px"] == 2


def test_output_protection_and_complete_id_validation(tmp_path):
    dataset, records = fixture_records(tmp_path)
    output = tmp_path / "result"
    responses = {"tile_001": reply()}
    source_bytes = (dataset / "tile.png").read_bytes()
    with pytest.raises(ValueError, match="exactly match"):
        evaluate_predictions(records, {}, output, dataset)
    with pytest.raises(ValueError, match="exactly match"):
        evaluate_predictions(records, dict(responses, extra=reply()), output, dataset)
    assert not output.exists()
    with pytest.raises(ValueError, match="outside dataset_root"):
        evaluate_predictions(records, responses, dataset / "result", dataset)
    output.mkdir()
    (output / "keep.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        evaluate_predictions(records, responses, output, dataset)
    assert (output / "keep.txt").read_text() == "preserve"
    assert (dataset / "tile.png").read_bytes() == source_bytes


@pytest.mark.parametrize("change,match", [
    ({"id": "../escape"}, "safe file names"),
    ({"id": "CON"}, "safe file names"),
    ({"points": [[100, 0]]}, "outside local"),
    ({"points": [[True, 0]]}, "finite local"),
    ({"tile": [100, 200, 165, 232]}, "Image dimensions"),
    ({"image": "../outside.png"}, "inside dataset_root"),
])
def test_invalid_record_cannot_write_partial_report(tmp_path, change, match):
    dataset, records = fixture_records(tmp_path)
    records[0].update(change)
    output = tmp_path / "result"
    with pytest.raises(ValueError, match=match):
        evaluate_predictions(records, {"tile_001": reply()}, output, dataset)
    assert not output.exists()


def test_report_escapes_user_metadata_and_retains_exact_reply(tmp_path):
    dataset, records = fixture_records(tmp_path)
    records[0]["label_source"] = "<script>alert('x')</script>"
    response = reply("<script>bad output</script>")
    output = tmp_path / "result"
    evaluate_predictions(records, {"tile_001": response}, output, dataset)
    report_html = (output / "index.html").read_text(encoding="utf-8")
    assert "<script>" not in report_html and "&lt;script&gt;" in report_html
    saved = json.loads((output / "images/tile_001/reply.json").read_text(encoding="utf-8"))
    assert saved["text"] == response["text"]


def test_module_does_not_import_transformers():
    subprocess.run([sys.executable, "-c", "import sys; import ionrecog.vision_language.training_evaluation; "
                    "assert 'transformers' not in sys.modules; assert 'torch' not in sys.modules"],
                   cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)
