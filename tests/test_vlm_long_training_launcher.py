"""Long-job lifecycle guards without loading a model or starting GPU work."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_vlm_long_training.py"
SPEC = importlib.util.spec_from_file_location("vlm_long_training_launcher", SOURCE)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def fixture_args(tmp_path):
    dataset, model = tmp_path / "dataset", tmp_path / "model"
    dataset.mkdir()
    model.mkdir()
    row = {"id": "test_1", "image_sha256": "same-image", "tile": [0, 0, 64, 64], "points": [[1.5, 2.5]]}
    launcher.write_json(dataset / "manifest.json", {"splits": {"test": "test.jsonl"}})
    (dataset / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    reports = []
    for name in ("base", "previous"):
        report = tmp_path / name
        report.mkdir()
        launcher.write_json(report / "points.json", {"reference_points": row["points"]})
        launcher.write_json(report / "summary.json", {
            "metric_interpretation": "reference_agreement_not_experimental_accuracy",
            "images": [dict(row, reference_count=1, parsed_points="points.json")]})
        reports.append(report)
    return launcher.build_parser().parse_args([
        "--dataset", str(dataset), "--model", str(model), "--output", str(tmp_path / "experiment"),
        "--python", sys.executable, "--baseline-report", str(reports[0]), "--previous-report", str(reports[1]),
        "--reserve-gib", "0"])


@pytest.mark.parametrize("outcome, expected, phases", [
    ("complete", "complete", ["training", "comparison_base", "comparison_previous"]),
    ("stopped", "stopped", ["training"]),
    ("error", "failed", ["training"]),
    ("missing_summary", "failed", ["training"]),
])
def test_only_complete_training_runs_comparisons(tmp_path, monkeypatch, outcome, expected, phases):
    args = fixture_args(tmp_path)
    seen = []
    monkeypatch.setattr(launcher, "temporary_system_awake", lambda: nullcontext(False))

    def child(command, phase, root, status):
        seen.append(phase)
        if phase == "training":
            training = root / "training"
            training.mkdir()
            launcher.write_json(training / "progress.json", {"stage": "stopped" if outcome == "stopped" else "complete"})
            if outcome == "complete":
                launcher.write_json(training / "training_summary.json", {"status": "complete", "optimizer_steps": 6144})
            return 9 if outcome == "error" else 0
        destination = root / phase
        destination.mkdir()
        (destination / "index.html").write_text("comparison", encoding="utf-8")
        return 0

    monkeypatch.setattr(launcher, "run_child", child)
    if expected == "failed":
        with pytest.raises((RuntimeError, FileNotFoundError)):
            launcher.run(args)
    else:
        assert launcher.run(args) == 0
    root = Path(args.output)
    assert seen == phases
    assert launcher.read_json(root / "launcher_status.json")["state"] == expected
    page = (root / "index.html").read_text(encoding="utf-8")
    assert ("comparison_previous/index.html" in page) == (expected == "complete")


def test_changed_test_labels_rejected_before_output_creation(tmp_path):
    args = fixture_args(tmp_path)
    launcher.write_json(Path(args.previous_report) / "points.json", {"reference_points": [[5., 6.]]})
    with pytest.raises(ValueError, match="reference coordinates"):
        launcher.run(args)
    assert not Path(args.output).exists()


def test_existing_output_never_overwritten(tmp_path):
    args = fixture_args(tmp_path)
    Path(args.output).mkdir()
    with pytest.raises(ValueError, match="new directory"):
        launcher.run(args)


def test_command_is_argument_list_with_resume_and_training_contract(tmp_path):
    args = fixture_args(tmp_path)
    args.resume_from = str(tmp_path / "path with spaces/latest.json")
    command = launcher.training_command(args, Path(args.output))
    assert isinstance(command, list)
    assert command[command.index("--steps") + 1] == "6144"
    assert command[command.index("--recovery-every") + 1] == "128"
    assert command[command.index("--selection-metric") + 1] == "f1_3px"
    assert command[command.index("--resume-from") + 1] == str(Path(args.resume_from).resolve())
    assert "--skip-baseline-generation" in command


def test_real_cpu_child_has_utf8_logs_and_recorded_lifecycle(tmp_path):
    root = tmp_path / "job"
    root.mkdir()
    (root / "logs").mkdir()
    status = {"state": "starting", "launcher_pid": 1, "processes": []}
    command = [sys.executable, "-c", "import sys; print('子进程输出'); sys.stderr.write('stderr marker\\n')"]
    assert launcher.run_child(command, "training", root, status) == 0
    assert "子进程输出" in (root / "logs/training.stdout.log").read_text(encoding="utf-8")
    assert "stderr marker" in (root / "logs/training.stderr.log").read_text(encoding="utf-8")
    assert status["child_pid"] is None
    assert status["processes"][0]["pid"] > 0
    assert status["processes"][0]["exit_code"] == 0
