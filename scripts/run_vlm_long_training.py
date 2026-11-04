"""Run a detached-friendly VLM training job and its fixed historical comparisons.

Launch this wrapper with Start-Process -WindowStyle Hidden. It owns child logs,
status, a temporary Windows idle-sleep request, and success-gated comparisons.
It never modifies system power settings or retries failed training automatically.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def replace_with_retry(source, destination):
    """Retry only brief Windows sharing/access locks, for at most one second."""
    for attempt in range(21):
        try:
            source.replace(destination)
            return
        except OSError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 20:
                raise
            time.sleep(0.05)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    replace_with_retry(temporary, path)


def contained_file(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("Missing or escaping input file: " + str(relative))
    return path


def check_fixed_test(dataset, reports):
    """Fail before a long run if historical comparisons would use other targets."""
    manifest = read_json(dataset / "manifest.json")
    location = manifest["splits"]["test"]
    if isinstance(location, dict):
        location = location.get("path", location.get("file"))
    rows = [json.loads(line) for line in contained_file(dataset, location).read_text(encoding="utf-8").splitlines() if line.strip()]
    records = {row["id"]: row for row in rows}
    if not records or len(records) != len(rows):
        raise ValueError("Test records are empty or have duplicate IDs")
    for directory in reports:
        report = read_json(directory / "summary.json")
        if report.get("metric_interpretation") != "reference_agreement_not_experimental_accuracy":
            raise ValueError("Expected a teacher-agreement historical report")
        previous = {entry["id"]: entry for entry in report["images"]}
        if set(previous) != set(records) or len(previous) != len(report["images"]):
            raise ValueError("Test IDs differ from historical report: " + str(directory))
        for ident, row in records.items():
            entry = previous[ident]
            if (entry["image_sha256"] != row["image_sha256"] or entry["tile"] != row["tile"]
                    or entry["reference_count"] != len(row["points"])):
                raise ValueError("Historical test image/geometry differs: " + ident)
            points = read_json(contained_file(directory, entry["parsed_points"]))["reference_points"]
            if points != row["points"]:
                raise ValueError("Historical test reference coordinates differ: " + ident)
    return len(rows)


@contextmanager
def temporary_system_awake():
    """Keep this Windows thread's execution request until all children finish."""
    function = None
    if os.name == "nt":
        import ctypes
        function = ctypes.windll.kernel32.SetThreadExecutionState
        function.argtypes = [ctypes.c_uint]
        function.restype = ctypes.c_uint
        if not function(0x80000000 | 0x00000001):
            raise OSError("SetThreadExecutionState failed; no global power settings were changed")
    try:
        yield bool(function)
    finally:
        if function is not None:
            function(0x80000000)


def publish(root, status):
    status["updated_utc"] = utc_now()
    progress_path = root / "training" / "progress.json"
    if progress_path.is_file():
        status["training_progress"] = read_json(progress_path)
    write_json(root / "launcher_status.json", status)
    summary_path = root / "training" / "training_summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else None
    labels = {"starting": "正在启动", "training": "训练中", "comparison_base": "生成原始模型对照",
              "comparison_previous": "生成上一轮微调对照", "complete": "已完成", "stopped": "已停止", "failed": "失败"}
    state = status["state"]
    refresh = '<meta http-equiv="refresh" content="30">' if state not in ("complete", "stopped", "failed") else ""
    links = []
    for relative, title in (("launcher_status.json", "运行状态 JSON"), ("training/progress.json", "训练进度"),
                            ("logs/training.stdout.log", "训练日志"), ("logs/training.stderr.log", "错误与警告日志"),
                            ("training/training_summary.json", "训练汇总"), ("training/validation_history.json", "验证历史"),
                            ("comparison_base/index.html", "原始模型 → 本轮微调"),
                            ("comparison_previous/index.html", "上一轮微调 → 本轮微调")):
        if (root / relative).is_file():
            links.append('<li><a href="%s">%s</a></li>' % (relative, title))
    detail = {key: status.get(key) for key in ("launcher_pid", "child_pid", "started_utc", "updated_utc", "training_progress", "error") if status.get(key) is not None}
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8">' + refresh
    page += '<meta name="viewport" content="width=device-width,initial-scale=1"><title>VLM 长训练记录</title>'
    page += '<style>body{max-width:1000px;margin:32px auto;padding:0 20px;font:16px/1.65 system-ui;color:#182332;background:#f7f9fc}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:20px;border:1px solid #dce2eb}a{color:#1459a5}</style>'
    page += '<h1>VLM 长训练：%s</h1><p>指标表示与 classical 伪标签的一致性，不是真实识别准确率。历史测试集用于回归比较。</p>' % html.escape(labels.get(state, state))
    page += '<ul>' + ''.join(links) + '</ul><h2>运行状态</h2><pre>' + html.escape(json.dumps(detail, ensure_ascii=False, indent=2)) + '</pre>'
    if summary is not None:
        page += '<h2>训练汇总</h2><pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre>'
    page += '</html>\n'
    temporary = root / "index.html.tmp"
    temporary.write_text(page, encoding="utf-8")
    replace_with_retry(temporary, root / "index.html")


def training_command(args, root):
    command = [str(Path(args.python).resolve()), "-u", "-m", "ionrecog.vision_language.finetune",
               "--dataset", str(Path(args.dataset).resolve()), "--model", str(Path(args.model).resolve()),
               "--output", str(root / "training"), "--skip-baseline-generation"]
    for name in ("steps", "accumulate", "eval_every", "eval_samples", "save_every", "recovery_every", "selection_metric",
                 "reserve_gib", "rank", "lr", "image_tokens", "max_new_tokens", "seed"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    if args.resume_from:
        command.extend(["--resume-from", str(Path(args.resume_from).resolve())])
    return command


def run_child(command, phase, root, status):
    environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    record = {"phase": phase, "command": command, "started_utc": utc_now()}
    status["state"], status["command"] = phase, command
    status["processes"].append(record)
    stdout = root / "logs" / (phase + ".stdout.log")
    stderr = root / "logs" / (phase + ".stderr.log")
    process = None
    try:
        with stdout.open("xb") as out, stderr.open("xb") as err:
            process = subprocess.Popen(command, cwd=str(PROJECT), env=environment, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            status["child_pid"] = record["pid"] = process.pid
            publish(root, status)
            while True:
                try:
                    code = process.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    publish(root, status)
            record.update(exit_code=code, finished_utc=utc_now())
            status["child_pid"] = None
            publish(root, status)
            return code
    except BaseException:
        if process is not None and process.poll() is None:
            # Only this wrapper's direct child is terminated on wrapper failure.
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


def run(args):
    root, dataset, model = (Path(path).resolve() for path in (args.output, args.dataset, args.model))
    reports = [Path(args.baseline_report).resolve(), Path(args.previous_report).resolve()]
    if root.exists():
        raise ValueError("Experiment output must be a new directory")
    if not Path(args.python).resolve().is_file() or not model.is_dir():
        raise ValueError("Python executable or local model directory is missing")
    if args.resume_from and not Path(args.resume_from).resolve().exists():
        raise ValueError("Resume checkpoint does not exist")
    for source in [dataset, model] + reports:
        if root == source or source in root.parents or root in source.parents:
            raise ValueError("Output overlaps an input directory")
    for name in ("steps", "accumulate", "eval_every", "eval_samples", "save_every", "recovery_every", "rank", "image_tokens", "max_new_tokens"):
        if getattr(args, name) < 1:
            raise ValueError(name + " must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.reserve_gib) or args.reserve_gib < 0:
        raise ValueError("Invalid learning rate or disk reserve")
    existing = root.parent
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free / 1024**3
    if free < args.reserve_gib:
        raise ValueError("Insufficient disk reserve")
    test_count = check_fixed_test(dataset, reports)
    if args.eval_samples < test_count:
        raise ValueError("eval_samples must include the complete fixed historical test split")
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    status = {"state": "starting", "launcher_pid": os.getpid(), "child_pid": None,
              "started_utc": utc_now(), "processes": [], "arguments": vars(args),
              "historical_test_images": test_count, "free_gib_before": free,
              "note": "No automatic retry; resume uses a new output directory and a verified recovery checkpoint."}
    try:
        publish(root, status)
        with temporary_system_awake() as active:
            status["temporary_system_awake"] = active
            code = run_child(training_command(args, root), "training", root, status)
            if code:
                raise RuntimeError("Training exited with code %d; see logs/training.stderr.log" % code)
            progress_file = root / "training" / "progress.json"
            progress = read_json(progress_file) if progress_file.is_file() else {}
            if progress.get("stage") == "stopped":
                status["state"] = "stopped"
                return 0
            summary = read_json(root / "training" / "training_summary.json")
            if summary.get("status") != "complete":
                raise RuntimeError("Training did not report complete; comparisons were not run")
            for before, phase in zip(reports, ("comparison_base", "comparison_previous")):
                command = [str(Path(args.python).resolve()), "-u", str(PROJECT / "scripts" / "compare_vlm_training.py"),
                           "--before", str(before), "--after", str(root / "training" / "after"), "--output", str(root / phase)]
                code = run_child(command, phase, root, status)
                if code:
                    raise RuntimeError("%s exited with code %d" % (phase, code))
            status["state"] = "complete"
        return 0
    except BaseException as exc:
        status.update(state="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        status["temporary_system_awake"] = False
        status["child_pid"] = None
        status["finished_utc"] = utc_now()
        status["free_gib_after"] = shutil.disk_usage(root).free / 1024**3
        publish(root, status)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True, help="New experiment root; trainer writes its training/ subdirectory")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model", default=str(PROJECT / "checkpoints/Qwen3-VL-2B-Instruct-bf16"))
    parser.add_argument("--baseline-report", default=str(PROJECT / "runs/vlm_train_20261103/base_test24/report"))
    parser.add_argument("--previous-report", default=str(PROJECT / "runs/vlm_train_20261103/adapter_test24/report"))
    parser.add_argument("--steps", type=int, default=6144)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=1024)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=1024)
    parser.add_argument("--recovery-every", type=int, default=128)
    parser.add_argument("--selection-metric", choices=("loss", "f1_3px"), default="f1_3px")
    parser.add_argument("--reserve-gib", type=float, default=8.)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--resume-from")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print("Launcher failed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
