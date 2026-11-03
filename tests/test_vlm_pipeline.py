"""Contract/guardrail tests use explicit synthetic replay, never fake a model run."""
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ionrecog.cli import main
from ionrecog.vlm import sha256_file, read_manifest, request_points, validate_url


def fixture(tmp_path, response=None, size=32):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    yy, xx = np.mgrid[:size, :size]
    raw_path = raw_dir / "source.npy"
    np.save(raw_path, (100 + 300 * np.exp(-((xx-16)**2 + (yy-16)**2)/4)).astype("float32"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "images": [{"id": "one", "path": str(raw_path),
        "shape": [size, size], "sha256": sha256_file(raw_path), "role": "synthetic_validation", "group": "one"}]}))
    replay = tmp_path / "responses.json"
    replay.write_text(json.dumps({"one/0": response or {"text": '{"points":[[516.129032258,516.129032258]],"uncertain":false}', "finish_reason": "stop"}}))
    output = tmp_path / "run"
    args = ["vlm", "--manifest", str(manifest), "--output", str(output), "--backend", "replay", "--responses", str(replay), "--min-free-gib", "0"]
    return args, output, manifest, raw_path


def test_real_pipeline_replay_preserves_source_and_measures_raw(tmp_path):
    args, output, _, raw_path = fixture(tmp_path)
    before = sha256_file(raw_path)
    main(args)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["settings"]["backend"] == "replay"
    assert summary["metrics_are_ground_truth_accuracy"] is False
    assert summary["images"][0]["vlm_direct_count"] == 1
    assert summary["images"][0]["vlm_checked_count"] == 1
    assert sha256_file(raw_path) == before
    points = json.loads((output / "images/one/vlm_checked_ions.json").read_text())["ions"]
    assert points[0]["x"] == pytest.approx(16, abs=.1)
    assert points[0]["raw_flux"] > 0
    assert (output / "index.html").exists()


@pytest.mark.parametrize("response", [
    {"text": '{"points": [[100,200]', "finish_reason": "length"},
    {"text": "there are no points", "finish_reason": "stop"},
    {"text": '{"points":[[1001,5]]}', "finish_reason": "stop"},
])
def test_invalid_or_truncated_tiles_are_failure_not_empty_prediction(tmp_path, response):
    args, output, _, _ = fixture(tmp_path, response)
    main(args)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "incomplete"
    assert summary["images"][0]["tiles_failed"] == 1
    assert not summary["images"][0]["coverage_complete"]
    raw_response = json.loads((output / "images/one/tiles/0.json").read_text())
    assert raw_response["response"] == response


def test_empty_success_is_distinct_from_failure(tmp_path):
    args, output, _, _ = fixture(tmp_path, {"text": '{"points":[],"uncertain":false}', "finish_reason": "stop"})
    main(args)
    row = json.loads((output / "summary.json").read_text())["images"][0]
    assert row["coverage_complete"] and row["vlm_direct_count"] == 0


def test_uncertain_empty_response_is_visible_abstention(tmp_path):
    args, output, _, _ = fixture(tmp_path, {"text": '{"points":[],"uncertain":true}', "finish_reason": "stop"})
    main(args)
    row = json.loads((output / "summary.json").read_text())["images"][0]
    assert row["status"] == "model_abstained"
    assert row["abstained_tiles"] == 1
    assert json.loads((output / "images/one/vlm_direct_ions.json").read_text())["has_abstentions"] is True


def test_resume_rejects_model_change_and_source_change(tmp_path):
    args, output, manifest, raw_path = fixture(tmp_path)
    main(args)
    main(args + ["--resume"])
    with pytest.raises(SystemExit) as exc:
        main(args + ["--resume", "--model", "different"])
    assert exc.value.code == 2
    np.save(raw_path, np.zeros((32, 32)))
    with pytest.raises(ValueError, match="modified"):
        read_manifest(manifest)


def test_no_writes_in_raw_directory(tmp_path):
    args, _, _, raw_path = fixture(tmp_path)
    args[args.index("--output") + 1] = str(raw_path.parent / "results")
    with pytest.raises(SystemExit):
        main(args)
    assert not (raw_path.parent / "results").exists()


def test_disk_reserve_stops_before_output(tmp_path, monkeypatch):
    args, output, _, _ = fixture(tmp_path)
    monkeypatch.setattr("ionrecog.vlm.shutil.disk_usage", lambda _: SimpleNamespace(free=100))
    args[args.index("--min-free-gib") + 1] = "5"
    with pytest.raises(SystemExit):
        main(args)
    assert not output.exists()


def test_api_prompt_and_coordinate_system_must_agree(tmp_path):
    args, output, _, _ = fixture(tmp_path)
    args[args.index("--backend") + 1] = "api"
    with pytest.raises(SystemExit):
        main(args + ["--coordinate-system", "pixels"])
    assert not output.exists()


def test_evaluation_rejects_incomplete_coverage(tmp_path):
    prediction = tmp_path / "partial.json"
    truth = tmp_path / "truth.json"
    prediction.write_text(json.dumps({"ions": [], "coverage_complete": False}))
    truth.write_text("[]")
    with pytest.raises(SystemExit) as error:
        main(["evaluate", "--prediction", str(prediction), "--truth", str(truth), "--output", str(tmp_path / "score.json")])
    assert error.value.code == 2
    assert not (tmp_path / "score.json").exists()


def test_local_api_serializes_image_and_keeps_response_provenance(monkeypatch):
    captured = {}
    class Opener:
        def open(self, request, timeout):
            captured["body"] = json.loads(request.data)
            captured["headers"] = request.headers
            return io.BytesIO(json.dumps({"model": "test-fixture", "choices": [{"finish_reason": "stop", "message": {"content": '{"points":[]}'}}], "usage": {"total_tokens": 12}}).encode())
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    result = request_points(Image.new("RGB", (16,16)), "http://127.0.0.1:8087/v1", "requested")
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "Authorization" not in captured["headers"]
    assert result["response_model"] == "test-fixture"
    assert result["usage"]["total_tokens"] == 12


@pytest.mark.parametrize("url", ["http://example.com/v1", "https://user:secret@example.com/v1", "https://example.com/v1?key=secret", "file:///tmp"])
def test_urls_cannot_embed_secrets_or_unencrypted_remote_credentials(url):
    with pytest.raises(ValueError):
        validate_url(url)
