"""Build an offline classical/masked comparison without inventing accuracy.

Run from the project root, for example:
  python scripts/compare_runs.py --input data/test --classical runs/classical \
      --masked runs/masked_predictions --output runs/comparison

All display panels for one frame use the SAME percentiles measured on the raw
image. Display ROI counts are descriptive, never ground truth. Source data and
existing inference results are read-only. No third-party plotting dependency.
"""
import argparse
import base64
import csv
from collections import defaultdict
import html
import io
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from ionrecog.data import list_images, load_image


DISPLAY_ROI = (180, 220, 300, 430)  # x0,y0,x1,y1, right/bottom exclusive


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative_link(path, parent):
    """Escape a local relative URL, including spaces and Windows separators."""
    try:
        value = os.path.relpath(str(Path(path).resolve()), str(parent.resolve())).replace("\\", "/")
        return quote(value, safe="/")
    except ValueError:  # Different Windows drives.
        return Path(path).resolve().as_uri()


def protected_output(output, inputs):
    output = Path(output).resolve()
    for source in inputs:
        source = Path(source).resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Comparison output must not overlap an input path: " + str(output))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new comparison directory.")
    return output


def uniform_indices(length, count):
    return np.linspace(0, length - 1, min(length, count), dtype=int).tolist()


def select_examples(records, count):
    """Equal group allocation then evenly spaced time order; never rank results."""
    grouped = defaultdict(list)
    for record in records:
        grouped[record["group"]].append(record)
    keys = sorted(grouped)
    if len(keys) > count:
        keys = [keys[i] for i in uniform_indices(len(keys), count)]
    allocations = {key: 0 for key in keys}
    remaining = min(count, sum(len(grouped[k]) for k in keys))
    while remaining:
        for key in keys:
            if allocations[key] < len(grouped[key]) and remaining:
                allocations[key] += 1
                remaining -= 1
    selected = []
    for key in keys:
        selected.extend(grouped[key][i] for i in uniform_indices(len(grouped[key]), allocations[key]))
    return selected


def png_data_uri(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def raw_limits(raw):
    low, high = map(float, np.percentile(raw, [0.5, 99.9]))
    if high <= low:
        low, high = float(raw.min()), float(raw.max())
    return low, max(high, low + 1.0)


def display_image(array, limits):
    low, high = limits
    gray = np.rint(np.clip((array - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(gray).convert("RGB")


def overlay(image, ions, color):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for ion in ions:
        x, y, radius = float(ion["x"]), float(ion["y"]), float(ion.get("radius", 3))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=1)
    return result


def roi_count(ions, roi):
    x0, y0, x1, y1 = roi
    return sum(x0 <= p["x"] < x1 and y0 <= p["y"] < y1 for p in ions)


def make_panels(record):
    raw = load_image(record["source"])
    restored = np.load(record["restored"], allow_pickle=False)
    if restored.shape != raw.shape or not np.isfinite(restored).all():
        raise ValueError("Invalid restored image: " + str(record["restored"]))
    limits = raw_limits(raw)
    original = display_image(raw, limits)
    classical_ions = read_json(record["classical_json"])["ions"]
    masked_ions = read_json(record["masked_json"])["ions"]
    panels = [original, overlay(original, classical_ions, (70, 255, 100)),
              overlay(original, masked_ions, (255, 180, 65)), display_image(restored, limits)]
    return panels, limits


def font(size=14):
    for filename in ("C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(filename, size)
        except OSError:
            pass
    return ImageFont.load_default()


def thumbnail_panel(image, size=(145, 245)):
    result = Image.new("RGB", size, "#0b1017")
    image = image.copy()
    image.thumbnail(size, Image.Resampling.LANCZOS)
    result.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return result


def contact_sheet(examples, panel_cache):
    """At most six examples, evenly selected from the report examples."""
    samples = [examples[i] for i in uniform_indices(len(examples), 6)]
    width, cell_height = 620, 310
    sheet = Image.new("RGB", (width * 2, ((len(samples) + 1) // 2) * cell_height), "#101923")
    draw = ImageDraw.Draw(sheet)
    small, title_font = font(12), font(14)
    labels = ("Raw", "Classical on raw", "Masked on raw", "Restored")
    for index, record in enumerate(samples):
        panels = panel_cache[record["relative"]]
        roi = DISPLAY_ROI if record["shape"] == [512, 512] else None
        if roi:
            panels = [image.crop(roi) for image in panels]
        x, y = (index % 2) * width + 12, (index // 2) * cell_height
        draw.text((x, y + 8), "Example %02d | %s" % (record["example_id"], "display ROI" if roi else "full frame"), font=title_font, fill="#f2f5fa")
        for column, panel in enumerate(panels):
            px = x + column * 150
            draw.text((px, y + 32), labels[column], font=small, fill="#c3d0df")
            sheet.paste(thumbnail_panel(panel), (px, y + 52))
    return sheet


def path_cell(label, path, output):
    return '<a href="%s">%s</a>' % (html.escape(relative_link(path, output), quote=True), html.escape(label))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--classical", required=True)
    parser.add_argument("--masked", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args(argv)
    if not 1 <= args.examples <= 12:
        parser.error("--examples must lie between 1 and 12")
    source, classical, masked = (Path(v).resolve() for v in (args.input, args.classical, args.masked))
    output = protected_output(args.output, (source, classical, masked))
    files = list_images(source)
    if not files:
        raise ValueError("No source images found.")
    summaries = {name: read_json(path / "summary.json") for name, path in (("classical", classical), ("masked", masked))}
    if summaries["classical"]["method"] != "classical" or summaries["masked"]["method"] != "masked":
        raise ValueError("Expected a classical run and a masked run.")
    if any(s.get("samples", 1) != 1 for s in summaries.values()):
        raise ValueError("This comparison requires single-sample inference outputs.")
    records = []
    for file in files:
        relative = file.relative_to(source) if source.is_dir() else Path(file.name)
        stems = {name: directory / "images" / relative for name, directory in (("classical", classical), ("masked", masked))}
        payloads = {name: read_json(str(stem) + "_ions.json") for name, stem in stems.items()}
        if any(payload.get("image") != relative.as_posix() for payload in payloads.values()):
            raise ValueError("Run image identifiers disagree for " + str(relative))
        if payloads["classical"]["shape"] != payloads["masked"]["shape"]:
            raise ValueError("Run shapes disagree for " + str(relative))
        restored = Path(str(stems["masked"]) + "_restored.npy")
        if not restored.is_file():
            raise FileNotFoundError(str(restored))
        shape = payloads["classical"]["shape"]
        row = {"relative": relative.as_posix(), "group": relative.parent.as_posix(), "source": file,
               "shape": shape,
               "restored": restored, "classical_json": Path(str(stems["classical"]) + "_ions.json"),
               "masked_json": Path(str(stems["masked"]) + "_ions.json"),
               "classical_overlay": Path(str(stems["classical"]) + "_overlay.png"),
               "masked_overlay": Path(str(stems["masked"]) + "_overlay.png")}
        for method in ("classical", "masked"):
            row[method + "_count"] = len(payloads[method]["ions"])
            row[method + "_display_roi_count"] = roi_count(payloads[method]["ions"], DISPLAY_ROI) if shape == [512, 512] else None
        records.append(row)
    examples = select_examples(records, args.examples)
    output.mkdir(parents=True, exist_ok=False) if not output.exists() else None
    fields = ("relative", "group", "classical_count", "masked_count", "classical_display_roi_count", "masked_display_roi_count")
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in records)
    grouped = defaultdict(list)
    for record in records:
        grouped[record["group"]].append(record)
    group_rows = []
    for group, rows in sorted(grouped.items()):
        entry = {"group": group, "images": len(rows)}
        for key in fields[2:]:
            values = [r[key] for r in rows if r[key] is not None]
            for name, function in (("mean", np.mean), ("min", np.min), ("max", np.max)):
                entry[key + "_" + name] = float(function(values)) if values else None
        group_rows.append(entry)
    with (output / "group_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)
    equal_detection = summaries["classical"].get("config", {}).get("detection") == summaries["masked"].get("config", {}).get("detection")
    report = {"source": str(source), "classical_run": str(classical), "masked_run": str(masked),
              "images": len(records), "groups": group_rows, "examples": [r["relative"] for r in examples],
              "example_selection": "Deterministic equal allocation across parent groups, then equally spaced file indices; no result-based selection.",
              "same_detection_configuration": equal_detection,
              "display_roi_xyxy": list(DISPLAY_ROI), "display_roi_applies_to_shape": [512, 512],
              "ground_truth_accuracy": None,
              "note": "Counts are algorithm outputs, not true ion counts. Display ROI neither restricts inference nor establishes labels. No denoising or detection superiority is inferred."}
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    example_sections, panel_cache = [], {}
    titles = ("原始图", "Classical：原图标圈", "Masked：原图标圈", "Masked 恢复图")
    for index, record in enumerate(examples, start=1):
        record["example_id"] = index
        panels, limits = make_panels(record)
        panel_cache[record["relative"]] = panels
        figures = []
        for title, panel in zip(titles, panels):
            figures.append('<figure><figcaption>%s</figcaption><img loading="lazy" alt="%s" src="%s"></figure>' % (title, title, png_data_uri(panel)))
        content = '<section class="example"><h3>%02d · %s</h3><p>全图候选：classical %d，masked %d。显示范围：%.2f–%.2f ADU。</p><div class="panels">%s</div>' % (index, html.escape(record["relative"]), record["classical_count"], record["masked_count"], limits[0], limits[1], "".join(figures))
        if record["shape"] == [512, 512]:
            crops = ['<figure><figcaption>%s · 局部放大</figcaption><img loading="lazy" alt="%s局部放大" src="%s"></figure>' % (title, title, png_data_uri(panel.crop(DISPLAY_ROI))) for title, panel in zip(titles, panels)]
            content += '<p>显示 ROI 候选：classical %d，masked %d。矩形仅用于查看及统计，不限制检测。</p><div class="panels crops">%s</div>' % (record["classical_display_roi_count"], record["masked_display_roi_count"], "".join(crops))
        content += "</section>"
        example_sections.append(content)
    contact_sheet(examples, panel_cache).save(output / "contact_sheet.png")
    group_html = []
    def number(value):
        return "—" if value is None else "%.1f" % value
    for row in group_rows:
        group_html.append('<tr><td>%s</td><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (html.escape(row["group"]), row["images"], number(row["classical_count_mean"]), number(row["masked_count_mean"]), number(row["classical_display_roi_count_mean"]), number(row["masked_display_roi_count_mean"])))
    frame_html = []
    for row in records:
        links = " · ".join((path_cell("原文件", row["source"], output), path_cell("classical 标圈", row["classical_overlay"], output), path_cell("masked 标圈", row["masked_overlay"], output), path_cell("classical JSON", row["classical_json"], output), path_cell("masked JSON", row["masked_json"], output), path_cell("恢复 NPY", row["restored"], output)))
        frame_html.append('<tr><td>%s</td><td>%d</td><td>%d</td><td>%s / %s</td><td>%s</td></tr>' % (html.escape(row["relative"]), row["classical_count"], row["masked_count"], number(row["classical_display_roi_count"]), number(row["masked_display_roi_count"]), links))
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Classical 与 Masked 实验对照</title>
<style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#111923;color:#edf2f8;line-height:1.7}main{max-width:1500px;margin:auto;padding:28px}h1{font-size:28px}h2{margin-top:35px}p{max-width:1200px}.note{padding:18px;background:#243343;border-left:4px solid #f0bd6c}a{color:#9fd2ff}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:9px;border-bottom:1px solid #364456;vertical-align:top;overflow-wrap:anywhere}th{background:#223044}h3{font-size:15px;overflow-wrap:anywhere}.example{padding:18px;background:#172330;margin:24px 0}.panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}figure{margin:0;background:#0b1017}figcaption{padding:8px;font-size:13px}img{display:block;width:100%;height:auto;image-rendering:auto}.crops img{max-height:460px;object-fit:contain}.muted{color:#bac7d9}.tag{background:#304156;padding:4px 10px;border-radius:4px}@media(max-width:850px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}main{padding:14px}}</style><main>
<h1>Classical 与 Masked：同图实验对照</h1>
<p class="note">本报告展示算法输出，不提供识别准确率。没有人工或独立物理真值，圈数减少不等于去噪更好，圈数增加也不等于召回更高。绿色圈为 classical 候选，橙色圈为 masked 候选；两者都画在原始图上。恢复图可能过度平滑或产生伪结构，必须结合原始像素检查。</p>
<p><span class="tag">{{IMAGE_COUNT}} 张图</span> <span class="tag">{{GROUP_COUNT}} 个目录组</span> <span class="tag">检测参数{{MATCHED_CONFIG}}</span></p>
<p>每张示例的四个面板共用由原始图 0.5% / 99.9% 分位数确定的显示范围；恢复图不单独拉伸。不同示例可能使用不同显示范围，不能凭跨图亮度比较绝对荧光强度。全图 512×512 的数据另展示固定矩形 [180,220,300,430) 的局部放大和候选数；它只是显示 ROI，不是检测约束、实例标签或真值。其他尺寸显示全图。</p>
<p>示例按目录组均衡分配，再按文件顺序等间隔选择，最多 12 张；不按结果好坏挑选。页面示例已嵌入，可离线打开；逐帧链接需要保留原始数据和运行结果的相对目录。<a href="comparison.csv">逐帧 CSV</a> · <a href="group_summary.csv">分组 CSV</a> · <a href="summary.json">报告元数据</a> · <a href="contact_sheet.png">紧凑预览</a></p>
<h2>分组候选数汇总</h2><div class="tablewrap"><table><thead><tr><th>组</th><th>图片数</th><th>Classical 全图均值</th><th>Masked 全图均值</th><th>Classical 显示 ROI 均值</th><th>Masked 显示 ROI 均值</th></tr></thead><tbody>{{GROUP_ROWS}}</tbody></table></div>
<h2>固定规则选取的示例</h2>{{EXAMPLES}}
<h2>全部图片</h2><div class="tablewrap"><table><thead><tr><th>图片</th><th>Classical</th><th>Masked</th><th>显示 ROI：C / M</th><th>原始数据与输出</th></tr></thead><tbody>{{FRAME_ROWS}}</tbody></table></div>
<p class="muted">该报告不修改原始图、训练权重或检测结果。CSV 中空的显示 ROI 值表示不适用，不能按 0 个离子解释。</p></main></html>'''
    replacements = {"IMAGE_COUNT": str(len(records)), "GROUP_COUNT": str(len(group_rows)),
                    "MATCHED_CONFIG": "一致" if equal_detection else "不同：不构成受控对照",
                    "GROUP_ROWS": "".join(group_html), "EXAMPLES": "".join(example_sections), "FRAME_ROWS": "".join(frame_html)}
    for key, value in replacements.items():
        document = document.replace("{{" + key + "}}", value)
    (output / "index.html").write_text(document, encoding="utf-8")
    print("Compared %d frames in %d groups. Report: %s" % (len(records), len(group_rows), output / "index.html"))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit("Error: " + str(error))
