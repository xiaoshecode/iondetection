"""CPU-only, failure-inclusive evaluation of VLM fine-tuning responses.

Classical coordinates are pseudo-labels: matching them measures agreement, not
experimental detection accuracy. No model inference or coordinate refinement is
performed here. Failed responses are retained in the primary denominator.
"""
import csv
import hashlib
import html
import json
import math
from collections import Counter
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
import re

from PIL import Image, ImageDraw

from ..evaluation import match_points
from .geometry import COORDINATE_CONVENTIONS, parse_points


TOLERANCES = (1.0, 3.0)
STATUSES = ("valid", "malformed", "truncated", "abstained")
_STOP = {"stop", "eos", "eos_token", "end_turn"}
_TRUNCATED = {"length", "max_tokens", "max_new_tokens", "max_output_tokens"}
_ABSTAINED = {"abstained", "content_filter", "refusal"}
_NOTE = (
    "与 classical 伪标签比较得到的是一致性（agreement），不是实验图像的真实识别准确率。"
    "严格统计包括全部样本；格式错误、截断和弃答按无预测计漏检，整图全对强制为否。"
    "仅有效输出的统计有选择偏差，不能替代严格统计。空标签失败不会增加虚构的漏检点，"
    "仍计入失败图像并降低整图全对比例。"
)


def _within(path, directory):
    return path == directory or directory in path.parents


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _validate(records, responses, output, dataset_root):
    root, destination = Path(dataset_root).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise ValueError("dataset_root must be an existing directory")
    if destination.exists():
        raise ValueError("Evaluation output already exists: " + str(destination))
    if _within(destination, root) or _within(root, destination):
        raise ValueError("Evaluation output must be outside dataset_root")
    if not isinstance(responses, Mapping):
        raise ValueError("responses must map each record ID to a response")
    records = list(records)
    if not records:
        raise ValueError("At least one evaluation record is required")
    validated, ids, filename_ids = [], set(), set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Each evaluation record must be a mapping")
        ident = record.get("id")
        if (not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", ident)
                or ident.endswith(".") or ident in ids or ident.casefold() in filename_ids
                or re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", ident.split(".")[0], re.I)):
            raise ValueError("Record IDs must be unique safe file names (case insensitive)")
        ids.add(ident)
        filename_ids.add(ident.casefold())
        tile = record.get("tile")
        if (not isinstance(tile, (list, tuple)) or len(tile) != 4
                or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in tile)):
            raise ValueError("tile must contain four nonnegative integer coordinates")
        x0, y0, x1, y1 = map(int, tile)
        width, height = x1 - x0, y1 - y0
        if width <= 0 or height <= 0:
            raise ValueError("tile must be a nonempty half-open rectangle")
        points = record.get("points")
        if not isinstance(points, (list, tuple)):
            raise ValueError("points must be explicit local [x,y] coordinates")
        truth = []
        for point in points:
            if (not isinstance(point, (list, tuple)) or len(point) != 2
                    or not all(_finite(v) for v in point)):
                raise ValueError("points must be finite local [x,y] coordinates")
            x, y = map(float, point)
            if not (0 <= x <= width - 1 and 0 <= y <= height - 1):
                raise ValueError("Reference points are outside local tile pixel centers")
            truth.append([x, y])
        for key in ("split", "label_source"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(key + " must be a nonempty string")
        pseudo = record.get("classical_pseudo", record["label_source"] == "classical_pseudo")
        if not isinstance(pseudo, bool):
            raise ValueError("classical_pseudo must be boolean")
        if "classical" in record["label_source"].lower():
            pseudo = True
        image_value = record.get("image")
        if not isinstance(image_value, str) or not image_value:
            raise ValueError("image must be an explicit file path")
        image_path = Path(image_value)
        image_path = image_path.resolve() if image_path.is_absolute() else (root / image_path).resolve()
        if not _within(image_path, root):
            raise ValueError("Input images must remain inside dataset_root")
        with Image.open(image_path) as source:
            if source.size != (width, height):
                raise ValueError("Image dimensions must equal the original tile dimensions: " + ident)
            picture = source.convert("RGB")
            picture.load()
        validated.append({"id": ident, "image": image_path.relative_to(root).as_posix(),
                          "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                          "tile": [x0, y0, x1, y1], "width": width, "height": height,
                          "points": truth, "split": record["split"],
                          "label_source": record["label_source"], "classical_pseudo": pseudo,
                          "picture": picture})
    if set(responses) != ids:
        missing, extra = ids - set(responses), set(responses) - ids
        raise ValueError("Response IDs must exactly match record IDs; missing={}, extra={}".format(
            sorted(missing), sorted(map(str, extra))))
    for ident, response in responses.items():
        if not isinstance(response, Mapping) or not {"text", "finish_reason", "seconds"} <= set(response):
            raise ValueError("Response {} requires text, finish_reason, and seconds".format(ident))
        if not isinstance(response["text"], str):
            raise ValueError("Response text must be a string: " + ident)
        if response["finish_reason"] is not None and not isinstance(response["finish_reason"], str):
            raise ValueError("finish_reason must be a string or None: " + ident)
        if not _finite(response["seconds"]) or response["seconds"] < 0:
            raise ValueError("Response seconds must be finite and nonnegative: " + ident)
    return validated, root, destination


def _response_state(response, width, height):
    points, parse_error, uncertain = [], None, False
    try:
        points = parse_points(response["text"], {"x0": 0, "y0": 0, "x1": width, "y1": height},
                              coordinate_system="normalized_1000")
        # parse_points validates this envelope but an uncertain empty list has
        # no per-point flag, so retain the whole-response abstention explicitly.
        body = response["text"].strip()
        fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", body, flags=re.S | re.I)
        envelope = json.loads(fenced.group(1).strip() if fenced else body)
        uncertain = (isinstance(envelope, dict) and envelope.get("uncertain", False)) or any(
            point["model_uncertain"] for point in points)
    except (ValueError, RecursionError) as exc:
        parse_error = str(exc)
    finish = response["finish_reason"]
    if finish in _TRUNCATED:
        return "truncated", points, parse_error or "Generation reached the token limit"
    if finish in _ABSTAINED:
        return "abstained", points, parse_error or "Generation refused or abstained"
    if parse_error is not None:
        return "malformed", [], parse_error
    if finish not in _STOP:
        return "malformed", points, "Unknown or missing generation finish_reason: " + str(finish)
    if uncertain:
        return "abstained", points, "Model marked the response or a point uncertain"
    return "valid", points, None


def _aggregate(rows, tolerance, scope):
    candidates = [row for row in rows if row["tolerance_px"] == tolerance]
    selected = candidates if scope == "strict" else [row for row in candidates if row["status"] == "valid"]
    result = {"scope": scope, "tolerance_px": tolerance, "total_images": len(candidates),
              "evaluated_images": len(selected), "excluded_images": len(candidates) - len(selected),
              "failure_images": sum(row["status"] != "valid" for row in candidates),
              "selection_bias": scope == "valid_only"}
    if not selected:
        result.update(tp=None, fp=None, fn=None, precision=None, recall=None, micro_f1=None,
                      mean_error_px=None, rmse_px=None, all_correct=False,
                      all_correct_images=0, all_correct_fraction=None, negative_fp_total=None)
        return result
    tp, fp, fn = (sum(row[key] for row in selected) for key in ("tp", "fp", "fn"))
    failures = any(row["status"] != "valid" for row in selected)
    if tp + fp + fn:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * tp / (2 * tp + fp + fn)
    else:
        precision = recall = f1 = 0.0 if failures else 1.0
    correct = sum(row["all_correct"] for row in selected)
    result.update(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, micro_f1=f1,
                  mean_error_px=sum(row["mean_error_px"] * row["tp"] for row in selected if row["tp"]) / tp if tp else None,
                  rmse_px=math.sqrt(sum(row["rmse_px"] ** 2 * row["tp"] for row in selected if row["tp"]) / tp) if tp else None,
                  all_correct=correct == len(selected), all_correct_images=correct,
                  all_correct_fraction=correct / len(selected),
                  negative_fp_total=sum(row["fp"] for row in selected if row["reference_count"] == 0))
    return result


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _overlay(picture, truth, proposals, path):
    scale = max(1, min(8, 512 // max(picture.size)))
    canvas = picture.resize((picture.width * scale, picture.height * scale), Image.Resampling.NEAREST)
    drawing = ImageDraw.Draw(canvas)
    radius = max(3, scale)
    for x, y in truth:
        x, y = (x + 0.5) * scale - 0.5, (y + 0.5) * scale - 0.5
        drawing.ellipse((x - radius, y - radius, x + radius, y + radius), outline="#00d9ff", width=2)
    for point in proposals:
        x, y = (point["x"] + 0.5) * scale - 0.5, (point["y"] + 0.5) * scale - 0.5
        drawing.line((x - radius, y - radius, x + radius, y + radius), fill="#ff9300", width=2)
        drawing.line((x - radius, y + radius, x + radius, y - radius), fill="#ff9300", width=2)
    with path.open("xb") as stream:
        canvas.save(stream, format="PNG")


def _write_html(path, report):
    escape = lambda value: html.escape(str(value), quote=True)
    sections = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'>",
                "<title>VLM 微调评估</title><style>body{font:16px sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.6}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px 12px}img{max-width:100%;image-rendering:pixelated}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f3f3;padding:12px}.failure{color:#a20}</style>",
                "<h1>VLM 微调评估</h1><p>" + escape(_NOTE) + "</p>",
                "<p>全部样本：{}；有效：{}；失败：{}。状态计数：{}</p>".format(
                    report["total_images"], report["valid_images"], report["failure_images"], escape(report["status_counts"])),
                "<p>坐标为原始图块的零起点像素中心；青色圆为参考坐标，橙色叉为模型可解析坐标。"
                "失败输出的橙色叉仅供审计，不进入严格匹配。</p>",
                "<table><tr><th>统计范围</th><th>容差 px</th><th>纳入/全部</th><th>TP</th><th>FP</th><th>FN</th><th>一致性 F1</th><th>整图全对</th></tr>"]
    for row in report["summary"]:
        sections.append("<tr>" + "".join("<td>" + escape(value) + "</td>" for value in (
            "严格：全部样本" if row["scope"] == "strict" else "仅有效输出（有选择偏差）",
            row["tolerance_px"], "{}/{}".format(row["evaluated_images"], row["total_images"]),
            row["tp"], row["fp"], row["fn"], row["micro_f1"], row["all_correct_fraction"])) + "</tr>")
    sections.append("</table>")
    for record in report["images"]:
        sections.extend(["<h2>" + escape(record["id"]) + "</h2>",
                         "<p class='{}'>状态：{}；参考点：{}；可解析点：{}；来源：{}；用时：{:.3f}s</p>".format(
                             "failure" if record["status"] != "valid" else "", escape(record["status"]),
                             record["reference_count"], record["parsed_count"], escape(record["label_source"]), record["seconds"]),
                         "<p>" + escape(record["error"] or "") + "</p>",
                         "<img alt='参考与模型坐标叠图' src='" + escape(record["overlay"]) + "'>",
                         "<p><a href='" + escape(record["reply"]) + "'>原始回复</a> · <a href='" + escape(record["parsed_points"]) + "'>解析坐标</a></p>"])
    sections.append("</html>")
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(sections))


def evaluate_predictions(records, responses, output, dataset_root):
    """Write a new evaluation directory and return its JSON-serializable report.

    Records require id/image/points/tile/split/label_source, with an optional
    boolean classical_pseudo. Image dimensions must equal tile dimensions;
    labels are local original-pixel coordinates, not full-image coordinates.
    Responses must match IDs exactly and contain text/finish_reason/seconds.
    No existing output or input file is overwritten. Validation precedes writes.
    """
    entries, root, destination = _validate(records, responses, output, dataset_root)
    rows, image_results, pending = [], [], []
    for entry in entries:
        ident, response = entry["id"], responses[entry["id"]]
        status, proposals, error = _response_state(response, entry["width"], entry["height"])
        predicted = [[point["x"], point["y"]] for point in proposals] if status == "valid" else []
        image_result = {key: value for key, value in entry.items() if key not in {"picture", "points"}}
        image_result.update(status=status, error=error, reference_count=len(entry["points"]),
                            parsed_count=len(proposals), strict_prediction_count=len(predicted),
                            finish_reason=response["finish_reason"], seconds=float(response["seconds"]),
                            reply="images/{}/reply.json".format(ident),
                            parsed_points="images/{}/parsed_points.json".format(ident),
                            overlay="images/{}/overlay.png".format(ident))
        for tolerance in TOLERANCES:
            metrics = match_points(predicted, entry["points"], tolerance)
            if status != "valid":
                metrics["all_correct"] = False
                if not entry["points"]:
                    metrics.update(precision=0.0, recall=0.0, f1=0.0)
            rows.append(dict(id=ident, status=status, split=entry["split"], label_source=entry["label_source"],
                             classical_pseudo=entry["classical_pseudo"], reference_count=len(entry["points"]),
                             parsed_count=len(proposals), strict_prediction_count=len(predicted), **metrics))
        image_results.append(image_result)
        pending.append((entry, response, status, proposals, error))
    status_counts = Counter(record["status"] for record in image_results)
    report = {"version": 1, "metric_interpretation": "reference_agreement_not_experimental_accuracy",
              "note": _NOTE, "dataset_root": str(root), "output": str(destination),
              "coordinate_convention": COORDINATE_CONVENTIONS["normalized_1000"],
              "tolerances_px": list(TOLERANCES), "total_images": len(entries),
              "valid_images": status_counts["valid"], "failure_images": len(entries) - status_counts["valid"],
              "status_counts": {status: status_counts[status] for status in STATUSES},
              "classical_pseudo_images": sum(entry["classical_pseudo"] for entry in entries),
              "label_sources": dict(Counter(entry["label_source"] for entry in entries)),
              "splits": dict(Counter(entry["split"] for entry in entries)),
              "summary": [_aggregate(rows, tolerance, scope) for scope in ("strict", "valid_only") for tolerance in TOLERANCES],
              "images": image_results, "metrics": rows}
    # Exclusive creation prevents overlapping runs from reusing one destination.
    destination.mkdir(parents=True, exist_ok=False)
    for entry, response, status, proposals, error in pending:
        folder = destination / "images" / entry["id"]
        folder.mkdir(parents=True, exist_ok=False)
        _write_json(folder / "reply.json", {"id": entry["id"], "text": response["text"],
                                           "finish_reason": response["finish_reason"], "seconds": float(response["seconds"])})
        _write_json(folder / "parsed_points.json", {"id": entry["id"], "status": status, "error": error,
                                                   "reference_points": entry["points"], "points": proposals,
                                                   "strict_points": [[p["x"], p["y"]] for p in proposals] if status == "valid" else []})
        _overlay(entry["picture"], entry["points"], proposals, folder / "overlay.png")
    _write_csv(destination / "metrics.csv", rows)
    _write_csv(destination / "summary.csv", report["summary"])
    _write_html(destination / "index.html", report)
    _write_json(destination / "summary.json", report)
    return report
