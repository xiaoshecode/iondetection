"""Compare paired VLM evaluation reports without inference or image copying."""
import argparse
from collections import Counter
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import quote


TOLERANCES = (1.0, 3.0)
STATUSES = ("valid", "malformed", "truncated", "abstained")
NOTE = ("这是与 classical 伪标签的教师一致性（teacher agreement），不是实验图像的真实识别准确率。"
        "以下均为包含全部图块的严格统计；格式错误、截断和拒答按无预测计漏检，整块全对为否。"
        "多个图块可来自同一原帧，不能视为独立实验重复。")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError("Non-finite JSON value: " + value)
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def contained_file(root, value):
    """Resolve report-controlled filenames without permitting directory escape."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Expected a nonempty relative file path")
    relative = Path(value)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError("Input path must stay inside its declared root: " + value)
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("Missing or escaping input file: " + value)
    return path


def integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(name + " must be a nonnegative integer")
    return value


def aggregate(rows):
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    failures = sum(row["status"] != "valid" for row in rows)
    denominator = 2 * tp + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 1.
    if not denominator:
        precision = recall = 0. if failures else 1.
    return {"images": len(rows), "failures": failures, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall,
            "micro_f1": 2 * tp / denominator if denominator else (0. if failures else 1.),
            "all_correct_images": sum(row["all_correct"] for row in rows)}


def load_report(directory):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError("Report directory does not exist: " + str(root))
    source = contained_file(root, "summary.json")
    report = read_json(source)
    if report.get("metric_interpretation") != "reference_agreement_not_experimental_accuracy":
        raise ValueError("Expected a teacher-agreement evaluation report")
    dataset = Path(report["dataset_root"]).resolve()
    if not dataset.is_dir():
        raise ValueError("Dataset root does not exist")
    records = report.get("images")
    if not isinstance(records, list) or not records:
        raise ValueError("Report must contain image records")
    images, paths, references = {}, {}, {}
    for entry in records:
        ident = entry.get("id")
        if (not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", ident)
                or ident.casefold() in {key.casefold() for key in images}):
            raise ValueError("Image IDs must be unique safe names")
        if entry.get("status") not in STATUSES:
            raise ValueError("Unknown response status: " + ident)
        if entry.get("classical_pseudo") is not True or entry.get("label_source") != "classical_pseudo":
            raise ValueError("This comparison expects explicit classical pseudo-labels")
        tile = entry.get("tile")
        if not isinstance(tile, list) or len(tile) != 4:
            raise ValueError("Invalid tile: " + ident)
        for value in tile:
            integer(value, "tile coordinate")
        if (tile[2] <= tile[0] or tile[3] <= tile[1]
                or entry.get("width") != tile[2] - tile[0] or entry.get("height") != tile[3] - tile[1]):
            raise ValueError("Tile dimensions do not match: " + ident)
        count = integer(entry.get("reference_count"), "reference_count")
        integer(entry.get("parsed_count"), "parsed_count")
        integer(entry.get("strict_prediction_count"), "strict_prediction_count")
        expected = entry.get("image_sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid image SHA256: " + ident)
        if digest(contained_file(dataset, entry["image"])) != expected:
            raise ValueError("Image SHA256 differs from report: " + ident)
        paths[ident] = {key: contained_file(root, entry[key]) for key in ("overlay", "reply", "parsed_points")}
        parsed = read_json(paths[ident]["parsed_points"])
        points = parsed.get("reference_points")
        if parsed.get("id") != ident or not isinstance(points, list) or len(points) != count:
            raise ValueError("Reference coordinates/count mismatch: " + ident)
        for point in points:
            if (not isinstance(point, list) or len(point) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point)
                    or not 0 <= point[0] <= entry["width"] - 1
                    or not 0 <= point[1] <= entry["height"] - 1):
                raise ValueError("Invalid reference center: " + ident)
        images[ident], references[ident] = entry, points
    status_counts = Counter(entry["status"] for entry in records)
    if (report.get("total_images") != len(images)
            or report.get("valid_images") != status_counts["valid"]
            or report.get("failure_images") != len(images) - status_counts["valid"]
            or any(report.get("status_counts", {}).get(key) != status_counts[key] for key in STATUSES)):
        raise ValueError("Report status counts are inconsistent")
    metrics = {}
    for row in report.get("metrics", []):
        ident, tolerance = row.get("id"), row.get("tolerance_px")
        if ident not in images or tolerance not in TOLERANCES or (ident, tolerance) in metrics:
            raise ValueError("Unexpected, duplicate or missing per-image metric keys")
        entry = images[ident]
        if row.get("status") != entry["status"] or row.get("reference_count") != entry["reference_count"]:
            raise ValueError("Per-image metric metadata differs: " + ident)
        for key in ("tp", "fp", "fn"):
            integer(row.get(key), key)
        if (row["tp"] + row["fn"] != entry["reference_count"]
                or row["tp"] + row["fp"] != entry["strict_prediction_count"]
                or not isinstance(row.get("all_correct"), bool)):
            raise ValueError("Per-image strict matching counts are inconsistent: " + ident)
        if entry["status"] != "valid" and (row["tp"] or row["fp"] or row["all_correct"]):
            raise ValueError("Failed responses must remain failures in strict metrics")
        metrics[ident, tolerance] = row
    if set(metrics) != {(ident, tolerance) for ident in images for tolerance in TOLERANCES}:
        raise ValueError("Every image must have strict metrics at 1 and 3 pixels")
    summaries = {}
    for tolerance in TOLERANCES:
        selected = [row for row in report["summary"] if row.get("scope") == "strict" and row.get("tolerance_px") == tolerance]
        if len(selected) != 1:
            raise ValueError("Expected exactly one strict summary per tolerance")
        saved = selected[0]
        computed = aggregate([metrics[ident, tolerance] for ident in images])
        if (saved.get("total_images") != len(images) or saved.get("evaluated_images") != len(images)
                or saved.get("excluded_images") != 0 or saved.get("failure_images") != computed["failures"]
                or any(saved.get(key) != computed[key] for key in ("tp", "fp", "fn", "all_correct_images"))
                or not isinstance(saved.get("micro_f1"), (int, float))
                or not math.isclose(saved["micro_f1"], computed["micro_f1"], abs_tol=1e-12)):
            raise ValueError("Strict summary disagrees with per-image matching")
        summaries[str(int(tolerance))] = computed
    return {"root": root, "dataset": dataset, "source_sha256": digest(source), "report": report,
            "images": images, "paths": paths, "references": references, "metrics": metrics,
            "summary": summaries, "status_counts": {key: status_counts[key] for key in STATUSES}}


def validate_pair(before, after):
    if set(before["images"]) != set(after["images"]):
        raise ValueError("Before/after image ID sets differ")
    if before["report"].get("coordinate_convention") != after["report"].get("coordinate_convention"):
        raise ValueError("Coordinate conventions differ")
    for ident, first in before["images"].items():
        second = after["images"][ident]
        for key in ("image_sha256", "tile", "reference_count", "width", "height", "split", "label_source"):
            if first[key] != second[key]:
                raise ValueError("Paired image differs in %s: %s" % (key, ident))
        if before["references"][ident] != after["references"][ident]:
            raise ValueError("Reference center coordinates differ: " + ident)


def linked_path(path, output):
    try:
        return quote(Path(os.path.relpath(path, output)).as_posix(), safe="/")
    except ValueError:  # Different Windows volumes cannot have a relative path.
        return path.as_uri()


def render_html(comparison):
    escape = lambda value: html.escape(str(value), quote=True)
    pieces = ["<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>",
              "<meta name='viewport' content='width=device-width,initial-scale=1'>",
              "<title>VLM 微调前后比较</title><style>body{font:16px system-ui,sans-serif;max-width:1120px;margin:32px auto;padding:0 18px;line-height:1.6;color:#17212c}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{border:1px solid #d7dce2;padding:7px;text-align:left}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{border:1px solid #d7dce2;padding:12px;min-width:0}.card img{display:block;width:100%;max-width:512px;image-rendering:pixelated}.bad{color:#a32922}.note{background:#fff5da;padding:12px}code{overflow-wrap:anywhere}.scroll{overflow-x:auto}@media(max-width:640px){.grid{grid-template-columns:1fr}}</style></head><body>",
              "<h1>VLM 微调前后比较</h1><p class='note'>" + escape(NOTE) + "</p>",
              "<p>相同图块：%d；已核对图块 ID、实际图像 SHA256、裁剪框、参考点数及完整参考坐标。"
              "青色圆为教师中心，橙色叉为模型输出；失败输出的橙色叉仅供审计，不进入严格匹配。</p>" % comparison["total_images"],
              "<div class='scroll'><table><tr><th>全部图块严格指标</th><th>微调前</th><th>微调后</th></tr>"]
    for status in STATUSES:
        pieces.append("<tr><td>%s</td><td>%d</td><td>%d</td></tr>" %
                      (escape(status), comparison["before"]["status_counts"][status], comparison["after"]["status_counts"][status]))
    for tolerance in ("1", "3"):
        first, second = (comparison[side]["summary"][tolerance] for side in ("before", "after"))
        for title, values in ((tolerance + " px 一致性 F1", ["%.4f" % row["micro_f1"] for row in (first, second)]),
                              (tolerance + " px TP / FP / FN", ["%d / %d / %d" % (row["tp"], row["fp"], row["fn"]) for row in (first, second)]),
                              (tolerance + " px 整块全对", ["%d/%d" % (row["all_correct_images"], row["images"]) for row in (first, second)])):
            pieces.append("<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % (escape(title), escape(values[0]), escape(values[1])))
    pieces.append("</table></div><h2>按教师点数分层</h2><p>桶内保留全部失败图块；F1 按桶内 TP/FP/FN 汇总计算，不是逐图 F1 的平均。</p><div class='scroll'><table><tr><th>教师点数</th><th>图块数</th><th>1 px 前 → 后</th><th>3 px 前 → 后</th><th>有效图块 前 → 后</th></tr>")
    for row in comparison["density"]:
        values = ["%.4f → %.4f" % (row["before"][t]["micro_f1"], row["after"][t]["micro_f1"]) for t in ("1", "3")]
        pieces.append("<tr><td>%s</td><td>%d</td><td>%s</td><td>%s</td><td>%d → %d</td></tr>" %
                      (escape(row["bucket"]), row["images"], values[0], values[1],
                       row["images"] - row["before"]["1"]["failures"], row["images"] - row["after"]["1"]["failures"]))
    pieces.append("</table></div><h2>逐图核查</h2><p>下列叠图链接到原报告文件，没有复制图像。请保留原评估目录及其相对位置。</p>")
    for pair in comparison["images"]:
        pieces.append("<h3>%s · 教师 %d 点</h3><p>原帧裁剪框：%s</p><div class='grid'>" %
                      (escape(pair["id"]), pair["reference_count"], escape(pair["tile"])))
        for side, title in (("before", "微调前"), ("after", "微调后")):
            value = pair[side]
            pieces.append("<section class='card'><strong>%s</strong><p class='%s'>状态：%s；可解析点：%d；严格计入点：%d</p><a href='%s'><img loading='lazy' alt='%s 教师与模型中心叠图' src='%s'></a><p><a href='%s'>原始回复</a> · <a href='%s'>解析坐标</a></p><p>%s</p></section>" %
                          (title, "bad" if value["status"] != "valid" else "", escape(value["status"]),
                           value["parsed_count"], value["strict_prediction_count"], escape(value["overlay"]), title,
                           escape(value["overlay"]), escape(value["reply"]), escape(value["parsed_points"]), escape(value["error"] or "")))
        pieces.append("</div>")
    pieces.append("<p><a href='comparison.json'>比较数据与来源哈希</a></p></body></html>")
    return "\n".join(pieces)


def compare(before_path, after_path, output):
    destination = Path(output).resolve()
    if destination.exists():
        raise ValueError("Output already exists; choose a fresh directory")
    before, after = load_report(before_path), load_report(after_path)
    for source in (before["root"], after["root"], before["dataset"], after["dataset"]):
        if destination == source or source in destination.parents or destination in source.parents:
            raise ValueError("Output must not overlap input reports or datasets")
    validate_pair(before, after)
    comparison = {"version": 1, "note": NOTE, "metric_interpretation": "teacher_agreement_not_true_accuracy",
                  "total_images": len(before["images"]), "images": [], "density": []}
    for side, data in (("before", before), ("after", after)):
        comparison[side] = {"report": str(data["root"]), "summary_sha256": data["source_sha256"],
                            "status_counts": data["status_counts"], "summary": data["summary"]}
    for low, high, label in ((1, 4, "1–4"), (5, 12, "5–12"), (13, 24, "13–24"), (0, 0, "0"), (25, math.inf, "25+")):
        ids = [ident for ident, entry in before["images"].items() if low <= entry["reference_count"] <= high]
        if not ids:
            continue
        row = {"bucket": label, "images": len(ids)}
        for side, data in (("before", before), ("after", after)):
            row[side] = {str(int(tolerance)): aggregate([data["metrics"][ident, tolerance] for ident in ids]) for tolerance in TOLERANCES}
        comparison["density"].append(row)
    for ident, entry in before["images"].items():
        pair = {key: entry[key] for key in ("id", "image_sha256", "tile", "reference_count", "split", "label_source")}
        for side, data in (("before", before), ("after", after)):
            pair[side] = {key: data["images"][ident][key] for key in ("status", "parsed_count", "strict_prediction_count", "error")}
            pair[side].update({key: linked_path(path, destination) for key, path in data["paths"][ident].items()})
        comparison["images"].append(pair)
    # Validate and render everything before creating a fresh output directory.
    page = render_html(comparison)
    payload = json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "comparison.json").open("x", encoding="utf-8") as stream:
        stream.write(payload)
    with (destination / "index.html").open("x", encoding="utf-8") as stream:
        stream.write(page)
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, help="Before evaluation directory containing summary.json")
    parser.add_argument("--after", required=True, help="After evaluation directory containing summary.json")
    parser.add_argument("--output", required=True, help="New comparison directory; existing paths are rejected")
    args = parser.parse_args(argv)
    try:
        result = compare(args.before, args.after, args.output)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, "Comparison rejected: %s\n" % exc)
    print("Compared %d paired crops: %s" % (result["total_images"], Path(args.output).resolve() / "index.html"))
    print(NOTE)


if __name__ == "__main__":
    main()
