"""Audit and compare two existing inference runs without changing their outputs.

Example (project .venv, from Net Recog):
  python CCD_2000ion/script/summarize_recognition.py \
    --run-dir CCD_2000ion/recognition_20261104_20260403 \
    --input CCD_2000ion/20260403_2000pure --offset 1161

Matching measures agreement between algorithms, never ground-truth accuracy.
"""

import argparse
import csv
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


DATASET = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def link(path, output):
    return quote(os.path.relpath(Path(path).resolve(), output).replace("\\", "/"), safe="/")


def points(ions, shape):
    xy = np.asarray([[p["x"], p["y"]] for p in ions], dtype=np.float64).reshape(-1, 2)
    h, w = shape
    if not np.isfinite(xy).all() or (len(xy) and not (
        (xy[:, 0] >= 0).all() and (xy[:, 0] < w).all()
        and (xy[:, 1] >= 0).all() and (xy[:, 1] < h).all()
    )):
        raise ValueError("Invalid candidate coordinates")
    return xy


def compare_points(a, b, tolerance=2.0):
    """Maximum valid match cardinality, then minimum summed Euclidean distance."""
    if not len(a) or not len(b):
        return np.empty(0, dtype=float)
    distance = cdist(a, b)
    # One fewer invalid assignment outweighs every possible valid-distance change.
    penalty = (min(len(a), len(b)) + 1) * (tolerance + 1)
    rows, cols = linear_sum_assignment(np.where(distance <= tolerance, distance, penalty))
    chosen = distance[rows, cols]
    return chosen[chosen <= tolerance]


def number_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "min": float(values.min()), "max": float(values.max()),
            "std_population": float(values.std(ddof=0)), "sum": float(values.sum())}


def font(size):
    for path in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def display(array, low, high):
    gray = np.rint(np.clip((array.astype(np.float64) - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(gray).convert("RGB")


def overlay(base, ions, color):
    result = base.copy()
    draw = ImageDraw.Draw(result)
    for ion in ions:
        x, y, r = float(ion["x"]), float(ion["y"]), float(ion["radius"])
        draw.ellipse((x-r, y-r, x+r, y+r), outline=color, width=1)
    return result


def make_visuals(record, output, offset):
    raw = np.load(record["source"], allow_pickle=False)
    restored = np.load(record["restored"], allow_pickle=False) - offset
    low, high = map(float, np.percentile(raw, [0.5, 99.9]))
    if high <= low:
        high = low + 1.0
    base = display(raw, low, high)
    c = read_json(record["classical_json"])["ions"]
    m = read_json(record["masked_json"])["ions"]
    images = [base, overlay(base, c, (70, 255, 100)),
              overlay(base, m, (255, 180, 65)), display(restored, low, high)]
    names = ["Original float32 image", "Classical: {} candidates (green)".format(len(c)),
             "Masked: {} candidates (orange), over original".format(len(m)),
             "Masked restoration, offset subtracted"]
    w, h = base.size
    header, label, gap, pad = 64, 28, 8, 12
    canvas = Image.new("RGB", (w+2*pad, header+4*(h+label+gap)+pad), "#111827")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 9), record["image"], fill="white", font=font(18))
    draw.text((pad, 36), "Shared display range: [{:.2f}, {:.2f}] original units; x=column, y=row; no resizing".format(low, high),
              fill="#cbd5e1", font=font(14))
    for i, (im, name) in enumerate(zip(images, names)):
        y = header + i*(h+label+gap)
        draw.text((pad, y+3), name, fill="white", font=font(16))
        canvas.paste(im, (pad, y+label))
    name = Path(record["image"]).stem
    overview = output / "examples" / (name + "_overview.png")
    overview.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(overview)

    # Fixed display crops show center, upper/lower edge and both ends. They do
    # not affect candidate selection or define a true crystal boundary.
    boxes = [("center", (384, 65, 640, 145)), ("upper edge", (384, 0, 640, 80)),
             ("lower edge", (384, 121, 640, 201)), ("left end", (0, 55, 256, 135)),
             ("right end", (768, 55, 1024, 135))]
    scale, cw, ch, title_h, col_gap = 2, 512, 160, 29, 10
    sheet = Image.new("RGB", (3*cw+4*col_gap, header+len(boxes)*(ch+title_h+gap)), "#111827")
    draw = ImageDraw.Draw(sheet)
    draw.text((col_gap, 8), name + " | fixed crops, nearest-neighbor x2", fill="white", font=font(18))
    draw.text((col_gap, 36), "Original", fill="white", font=font(16))
    draw.text((cw+2*col_gap, 36), "Classical (green)", fill="#46ff64", font=font(16))
    draw.text((2*cw+3*col_gap, 36), "Masked (orange)", fill="#ffb441", font=font(16))
    for i, (label_name, box) in enumerate(boxes):
        x0, y0, x1, y1 = box
        if x1 > w or y1 > h:
            raise ValueError("Fixed diagnostic crops require at least 201 x 1024 images")
        y = header + i*(ch+title_h+gap)
        draw.text((col_gap, y+3), "{}: x=[{},{}), y=[{},{})".format(label_name, x0, x1, y0, y1),
                  fill="#cbd5e1", font=font(15))
        for j, im in enumerate(images[:3]):
            crop = im.crop(box).resize((cw, ch), Image.Resampling.NEAREST)
            sheet.paste(crop, (col_gap+j*(cw+col_gap), y+title_h))
    crop_path = output / "examples" / (name + "_crops.png")
    sheet.save(crop_path)
    return {"image": record["image"], "overview": link(overview, output), "crops": link(crop_path, output),
            "display_limits_original_units": [low, high],
            "crops_xyxy_exclusive": {name: list(box) for name, box in boxes}}


def audit_records(run, source, expected, tolerance):
    files = sorted(source.glob("*.npy"))
    if len(files) != expected:
        raise ValueError("Expected {} original images; found {}".format(expected, len(files)))
    summaries = {method: read_json(run/method/"summary.json") for method in ("classical", "masked")}
    by_name = {}
    for method, summary in summaries.items():
        if summary["method"] != method or summary["images"] != expected or summary["samples"] != 1:
            raise ValueError("Incomplete or incompatible {} run".format(method))
        by_name[method] = {row["image"]: row for row in summary["frames"]}
        if len(by_name[method]) != expected or len(summary["frames"]) != expected:
            raise ValueError("Duplicate/missing {} frame summaries".format(method))
        actual = list((run/method/"images").glob("*_ions.json"))
        if len(actual) != expected:
            raise ValueError("Wrong {} result file count".format(method))
    records, distances = [], []
    for i, path in enumerate(files):
        raw = np.load(path, allow_pickle=False)
        if raw.shape != (201, 1024) or not np.isfinite(raw).all():
            raise ValueError("Unexpected original array: {}".format(path))
        png_name = path.stem + ".png"
        record = {"image": path.name, "source": path, "png_image": png_name}
        ions = {}
        for method in ("classical", "masked"):
            prefix = run/method/"images"/png_name
            for suffix in ("_ions.json", "_ions.csv", "_overlay.png", "_foreground.png"):
                if not Path(str(prefix)+suffix).is_file():
                    raise ValueError("Missing output: " + str(prefix)+suffix)
            json_path = Path(str(prefix)+"_ions.json")
            item = read_json(json_path)
            frame = by_name[method][png_name]
            if item["shape"] != list(raw.shape) or len(item["ions"]) != frame["count"]:
                raise ValueError("Shape/count mismatch: " + str(json_path))
            if item["method"] != method or item["image"] != png_name:
                raise ValueError("Method/image mismatch: " + str(json_path))
            ions[method] = points(item["ions"], raw.shape)
            xy = ions[method]
            border = ((xy[:, 0] < 8) | (xy[:, 0] >= raw.shape[1]-8)
                      | (xy[:, 1] < 8) | (xy[:, 1] >= raw.shape[0]-8))
            record[method+"_json"] = json_path
            record[method+"_count"] = len(ions[method])
            record[method+"_outer8px_count"] = int(border.sum())
            record[method+"_seconds"] = frame["seconds"]
            record[method+"_threshold"] = frame["detection_threshold_adu"]
        if record["classical_threshold"] != record["masked_threshold"]:
            raise ValueError("Methods used different detection thresholds: " + path.name)
        restored_path = run/"masked"/"images"/(png_name+"_restored.npy")
        restored = np.load(restored_path, allow_pickle=False)
        if restored.shape != raw.shape or not np.isfinite(restored).all():
            raise ValueError("Invalid restored image: " + str(restored_path))
        record["restored"] = restored_path
        matched = compare_points(ions["classical"], ions["masked"], tolerance)
        distances.extend(matched.tolist())
        record.update({"matched_pairs": len(matched),
                       "classical_unmatched": len(ions["classical"])-len(matched),
                       "masked_unmatched": len(ions["masked"])-len(matched),
                       "count_difference_masked_minus_classical": len(ions["masked"])-len(ions["classical"]),
                       "mean_matched_distance_px": float(matched.mean()) if len(matched) else None})
        records.append(record)
        if (i+1) % 25 == 0 or i+1 == len(files):
            print("Compared {}/{} frames".format(i+1, len(files)), flush=True)
    return records, summaries, np.asarray(distances)


def build_report(args):
    run, source = args.run_dir.resolve(), args.input.resolve()
    if DATASET not in run.parents or DATASET not in source.parents:
        raise ValueError("Both input and run directory must be inside CCD_2000ion")
    output = run/"comparison"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise ValueError("Comparison already exists; pass --overwrite to regenerate this report")
    conversion = read_json(run/"conversion_manifest.json")
    if (conversion["frames"] != args.expected_images or conversion["offset"] != args.offset
            or conversion["gain"] != 1 or conversion["clipped_pixels"] != 0
            or conversion["resized"] or conversion["transposed"]
            or conversion["max_abs_rounding_error"] > 0.5
            or Path(conversion["source"]).resolve() != source):
        raise ValueError("Conversion manifest does not match this report's input assumptions")
    records, summaries, distances = audit_records(run, source, args.expected_images, args.tolerance)
    output.mkdir(parents=True, exist_ok=True)
    numeric_fields = [key for key in records[0] if key not in {"source", "classical_json", "masked_json", "restored"}]
    with (output/"per_frame.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, numeric_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in numeric_fields} for row in records)
    methods = {method: {"count": number_stats([r[method+"_count"] for r in records]),
                        "outer8px_border_count": number_stats([r[method+"_outer8px_count"] for r in records]),
                        "seconds_including_exports": number_stats([r[method+"_seconds"] for r in records])}
               for method in ("classical", "masked")}
    totals = {method: int(methods[method]["count"]["sum"]) for method in methods}
    matches = sum(row["matched_pairs"] for row in records)
    summary = {"images": len(records), "shape_yx": [201, 1024], "methods": methods,
               "count_difference_masked_minus_classical": number_stats([r["count_difference_masked_minus_classical"] for r in records]),
               "agreement": {"tolerance_px": args.tolerance, "one_to_one_matched_pairs": matches,
                   "classical_unmatched": totals["classical"]-matches, "masked_unmatched": totals["masked"]-matches,
                   "matched_fraction_of_classical": matches/totals["classical"] if totals["classical"] else None,
                   "matched_fraction_of_masked": matches/totals["masked"] if totals["masked"] else None,
                   "matched_distance_px": number_stats(distances) if len(distances) else None},
               "input_conversion": {"formula": "png_uint16 = round(original_float32 + offset)", "offset": args.offset,
                   "gain": 1, "max_quantization_error": 0.5, "resized": False, "clipped": False},
               "border_diagnostic": {"width_px": 8, "coordinate_rule": "x < 8 or x >= 1016 or y < 8 or y >= 193",
                   "used_to_filter_detections": False, "ground_truth_region": False},
               "accuracy_evaluated": False, "ground_truth_available": False,
               "no_accuracy_claim": True, "no_count_target_assumption": True,
               "masked_checkpoint": summaries["masked"].get("checkpoint"),
               "restoration_config": summaries["masked"].get("restoration_config"),
               "all_required_outputs_present_and_coordinates_valid": True}
    indices = np.linspace(0, len(records)-1, min(6, len(records)), dtype=int).tolist()
    example_rows = [(i, "按文件名时间顺序等间隔抽样") for i in indices]
    extreme = max(range(len(records)), key=lambda i: abs(records[i]["count_difference_masked_minus_classical"]))
    if extreme not in indices:
        example_rows.append((extreme, "补充诊断：两方法候选数量差绝对值最大；不是代表性抽样"))
    examples = []
    for i, reason in example_rows:
        example = make_visuals(records[i], output, args.offset)
        example.update({"index_zero_based": i, "selection": reason})
        examples.append(example)
    summary["examples"] = examples
    write_json(output/"summary.json", summary)
    agreement = summary["agreement"]
    text = ["20260403_2000pure：classical 与 masked 识别结果", "",
            "处理完成：两方法各 {} 张，总计 {} 次单帧识别，尺寸 201×1024。".format(len(records), 2*len(records)),
            "输入与算法：", "  保留原始 NPY；统一加 {} 后四舍五入到 uint16 PNG。无缩放、无裁零、无逐图归一化，量化误差至多 0.5 原始单位。".format(args.offset),
            "  复用现有 classical 和 masked 算法、已有 best.pt；未重新训练。",
            "  检测参数：sigma=1, background_sigma=10, threshold_sigma=4, min_distance=4, radius=3, max_ions=5000。",
            '  使用全图模式 region.mode="off"，noise_reference="raw"。原 512×512 专用 ROI 不适用于本数据。',
            '  config_used.yaml 仅将原配置中被 YAML 1.1 解析成布尔值的 off 写成字符串 "off"；算法参数数值未调整。', "",
            "候选数统计（不是独立确认的真实离子数）："]
    for method in methods:
        cs, ts = methods[method]["count"], methods[method]["seconds_including_exports"]
        text.append("  {}：均值 {:.2f}，中位数 {:.0f}，范围 {:.0f}–{:.0f}，总体标准差 {:.2f}，总候选 {:.0f}。".format(
            method, cs["mean"], cs["median"], cs["min"], cs["max"], cs["std_population"], cs["sum"]))
        text.append("    单帧平均 {:.3f} 秒，总计 {:.2f} 秒（原 CLI 计时含定位/恢复与导出，不含初次模型加载）。".format(ts["mean"], ts["sum"]))
    diff = summary["count_difference_masked_minus_classical"]
    text.extend(["", "逐帧 masked − classical 候选数差：平均 {:.2f}，范围 {:.0f}–{:.0f}。".format(diff["mean"], diff["min"], diff["max"]),
                 "两方法在 {:.1f} 像素内一对一配对：{} 对。".format(args.tolerance, matches)])
    if totals["classical"] and totals["masked"]:
        text.append("  分别覆盖 classical 候选的 {:.2%} 和 masked 候选的 {:.2%}；未配对分别为 {}、{}。".format(
            agreement["matched_fraction_of_classical"], agreement["matched_fraction_of_masked"],
            agreement["classical_unmatched"], agreement["masked_unmatched"]))
    if len(distances):
        text.append("  已配对中心距离：均值 {:.3f} 像素，中位数 {:.3f} 像素。".format(distances.mean(), np.median(distances)))
    text.extend(["  配对优先最大化有效配对数，再最小化距离。配对只表示两算法一致性，未配对不等于误检或漏检。", "",
                 "图像外侧 8 像素边带诊断（x<8 或 x≥1016 或 y<8 或 y≥193）："])
    for method in methods:
        bs = methods[method]["outer8px_border_count"]
        text.append("  {}：每张均值 {:.2f}，范围 {:.0f}–{:.0f}，总计 {:.0f} 个边带候选。".format(
            method, bs["mean"], bs["min"], bs["max"], bs["sum"]))
    text.extend(["  边带仅用于描述候选在图像边界的聚集，没有用于剔除候选，也不是晶格真值区域；边带候选不能自动等同于误检。", "",
        "视觉检查（本次运行）：",
        "  已查看按文件名时间顺序等间隔抽取的 6 张示例和候选数差最大的补充示例。两种方法均跟随了中央多数可见亮峰。",
        "  抽查的所有 masked 示例均在原图暗背景的上下图像边界出现候选行，提示边界伪候选问题。恢复图更平滑，尚不能证明定位性能提高；外围拉长光斑的中心及数量仍存在判读歧义。", "",
        "结果边界：", "  没有逐离子独立真值；文件夹名称中的 2000 不能当作每张精确真值，因此不报告准确率、召回率、F1 或离子量子态保真度。",
        "  masked 使用旧数据训练的权重，checkpoint offset=0、scale=8426 保持不变。新晶格、噪声、背景及 +1161 偏置均可能影响迁移表现。",
        "  两方法均为全图检测；没有人工晶格边界或背景排除。候选数增加不能单独证明更准确。",
        "  原算法 CSV/JSON 的 raw_* 字段来自加偏置并量化的 PNG。原始 float32 强度仍以源 NPY 为准；峰值/背景还原需减去偏置。",
        "  半径 3 像素的局部测光与背景环带可能受到密集相邻离子信号影响；raw_flux、raw_snr 不代表独立校准的单离子亮度或置信度。",
        "  图示在原始 NPY 上标圈；恢复图减去偏置后，所有同帧面板共用原图 0.5%–99.9% 显示范围。局部图为最近邻放大，仅用于查看。", "",
        "产物：index.html 为可离线浏览的完整对照；per_frame.csv 为逐帧统计；summary.json 为机器可读汇总。",
        "       ../classical 和 ../masked 中保留全部候选坐标、原算法标圈与掩码；masked 另有恢复数组。",
        "       ../conversion_manifest.json、../config_used.yaml、../commands.json、../integrity_audit.json 记录转换、配置、命令与完整性。"])
    (output/"识别结果总结.txt").write_text("\n".join(text)+"\n", encoding="utf-8-sig")
    sections = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>CCD 2000ion — classical / masked</title><style>body{font:16px/1.65 system-ui,'Microsoft YaHei',sans-serif;max-width:1280px;margin:32px auto;padding:0 20px;color:#18212f;background:#f5f7fa}h1{font-size:28px}section{background:white;border:1px solid #d8dee7;border-radius:10px;padding:22px;margin:20px 0}pre{white-space:pre-wrap;font:inherit}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:7px;border-bottom:1px solid #e4e7ed;text-align:right}th:first-child,td:first-child{text-align:left}thead{background:#e8eef5}a{color:#0758aa}img{display:block;max-width:100%;height:auto;margin:10px auto}.scroll{overflow:auto}small{color:#526072}</style>",
        "<h1>CCD_2000ion / 20260403_2000pure</h1><p>classical 与 masked：完整批次识别与逐帧对照</p><section><pre>"+html.escape("\n".join(text))+"</pre>",
        "<p><a href='summary.json'>汇总 JSON</a> · <a href='per_frame.csv'>逐帧 CSV</a> · <a href='"+quote("识别结果总结.txt")+"'>中文总结 TXT</a></p>",
        "<p>运行记录：<a href='../conversion_manifest.json'>转换清单</a> · <a href='../config_used.yaml'>配置</a> · <a href='../commands.json'>命令</a> · <a href='../integrity_audit.json'>源码/原始数据完整性核验</a></p></section>"]
    for ex in examples:
        sections.append("<section><h2>"+html.escape(ex["image"])+"</h2><p>"+html.escape(ex["selection"])+"</p><a href='"+ex["overview"]+"'><img loading='lazy' src='"+ex["overview"]+"' alt='同帧原图、两方法定位与恢复图'></a><details><summary>查看中心、上下边缘和两端的固定局部放大图</summary><a href='"+ex["crops"]+"'><img loading='lazy' src='"+ex["crops"]+"' alt='固定坐标局部放大对照'></a></details></section>")
    sections.append("<section><h2>全部 {} 张结果</h2><p><small>点击文件名查看原始 NPY；C/M 为两算法。未配对只表示方法间差异。边带统计指图像外侧 8 像素。</small></p><div class='scroll'><table><thead><tr><th>图像</th><th>C 候选</th><th>M 候选</th><th>M−C</th><th>配对</th><th>C 未配对</th><th>M 未配对</th><th>C 边带</th><th>M 边带</th><th>结果文件</th></tr></thead><tbody>".format(len(records)))
    for row in records:
        links = []
        for method, short in (("classical", "C"), ("masked", "M")):
            prefix = run/method/"images"/row["png_image"]
            for suffix, label in (("_overlay.png", "标圈"), ("_ions.csv", "坐标")):
                links.append("<a href='{}'>{} {}</a>".format(link(str(prefix)+suffix, output), short, label))
        links.append("<a href='{}'>恢复 NPY</a>".format(link(row["restored"], output)))
        sections.append("<tr><td><a href='{}'>{}</a></td><td>{}</td><td>{}</td><td>{:+d}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            link(row["source"], output), html.escape(row["image"]), row["classical_count"], row["masked_count"],
            row["count_difference_masked_minus_classical"], row["matched_pairs"], row["classical_unmatched"],
            row["masked_unmatched"], row["classical_outer8px_count"], row["masked_outer8px_count"], " · ".join(links)))
    sections.append("</tbody></table></div></section></html>")
    (output/"index.html").write_text("\n".join(sections), encoding="utf-8")
    print(json.dumps({"output": str(output), "images": len(records), "methods": methods,
                      "agreement": agreement}, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--offset", type=float, required=True)
    parser.add_argument("--expected-images", type=int, default=267)
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.offset) or args.offset < 0 or not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("Require finite nonnegative offset and finite positive tolerance")
    build_report(args)


if __name__ == "__main__":
    main()
