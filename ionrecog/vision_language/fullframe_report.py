"""CPU-only, incremental galleries for region-assisted VLM original-frame inference."""
import csv
import html
import io
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import time
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw


_COLORS = {"vlm": "#5bff9e", "classical": "#ffd66b", "roi": "#61b9ff", "failed": "#ff615e"}
_LABELS = {
    "prepared": "已准备", "loading_model": "正在加载模型", "running": "正在识别",
    "complete": "完成", "complete_with_issues": "处理结束，部分结果需检查",
    "failed": "已停止：发生错误", "pending": "等待识别", "incomplete": "结果不完整",
    "failed_tiles": "部分图块失败", "abstained": "模型弃答", "valid": "有效输出",
    "model_abstained": "模型弃答", "no_crystal": "未找到可靠晶体区域",
}
_CSS = """
:root{color-scheme:dark;font-family:system-ui,'Microsoft YaHei',sans-serif;background:#101622;color:#e6edf7}
body{max-width:1480px;margin:0 auto;padding:28px 24px 60px;line-height:1.6}h1{font-size:28px;line-height:1.3;margin:4px 0 10px}
h2{font-size:19px;margin:22px 0 12px}p{margin:10px 0}a{color:#89c8ff;text-decoration:none}a:hover{text-decoration:underline}
.muted,small{color:#aab8ce}.notice{background:#1d293c;border-left:3px solid #61b9ff;padding:12px 16px;border-radius:4px}
.warning{background:#3b2429;border-color:#ff615e}.stats{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}.stat{background:#192335;border:1px solid #2b3c54;border-radius:10px;padding:13px 20px;min-width:130px}
.stat strong{font-size:26px;display:block}.stat span{color:#aab8ce;font-size:13px}.controls{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}
input,select{font:inherit;background:#1b2639;color:#eef4fc;border:1px solid #405471;border-radius:6px;padding:8px 12px}input{flex:1;min-width:240px}
.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #2b384c;text-align:left;vertical-align:top}th{color:#aab8ce;background:#182234;position:sticky;top:0}td.path{max-width:420px;overflow-wrap:anywhere}
.good{color:#5bff9e}.bad{color:#ff9a97}.tag{display:inline-block;border:1px solid #405471;border-radius:5px;padding:0 6px;font-size:12px;white-space:nowrap}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,350px),1fr));gap:16px}
figure{margin:0;background:#172132;border:1px solid #293a50;border-radius:8px;padding:12px;overflow:auto}figure img{display:block;max-width:100%;height:auto;margin:10px auto;background:#080b10}figcaption{font-size:15px;font-weight:600}.legend{display:flex;gap:20px;flex-wrap:wrap}.legend span:before{content:'●';margin-right:7px}
details{margin:12px 0}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#172132;padding:12px;border-radius:6px;font-size:12px}code{overflow-wrap:anywhere}progress{width:100%;height:10px;accent-color:#5bff9e}
@media(max-width:600px){body{padding:20px 12px}.stat{min-width:110px;padding:10px}h1{font-size:23px}}
"""


def _escape(value):
    return html.escape(str(value), quote=True)


def _label(value):
    return _LABELS.get(str(value), str(value))


def _atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 19:
                raise
            time.sleep(.05)


def _text(path, text):
    _atomic(path, text.encode("utf-8"))


def _save_image(path, image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    _atomic(path, stream.getvalue())


def _document(title, body, refresh=False, script=""):
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            + ('<meta http-equiv="refresh" content="30">' if refresh else '')
            + '<title>' + _escape(title) + '</title><style>' + _CSS + '</style></head><body>'
            + body + script + '</body></html>')


def _metric(metric):
    value = metric.get("f1") if isinstance(metric, dict) else None
    return "—" if value is None else "%.4f" % value


def _duration(seconds):
    seconds = max(0, int(seconds or 0))
    if seconds >= 3600:
        return "%d h %02d min" % (seconds // 3600, seconds % 3600 // 60)
    return "%d min %02d s" % (seconds // 60, seconds % 60)


def _stats(items):
    return '<div class="stats">' + ''.join(
        '<div class="stat"><strong>%s</strong><span>%s</span></div>' % (_escape(value), _escape(label))
        for label, value in items) + '</div>'


def _figure(filename, title, note=""):
    return ('<figure><figcaption>%s</figcaption><a href="%s" target="_blank" rel="noopener">'
            '<img src="%s" loading="lazy" alt="%s"></a><small>%s</small></figure>') % (
                _escape(title), quote(filename), quote(filename), _escape(title), _escape(note))


def _overlay(base, points, roi, tile_records, color, scale=1, offset=(0, 0), numbers=False):
    image = base.copy()
    draw = ImageDraw.Draw(image)
    ox, oy = offset
    def xy(x, y):
        return (float(x) - ox) * scale, (float(y) - oy) * scale
    if roi:
        draw.rectangle((*xy(roi[0], roi[1]), *xy(roi[2] - 1, roi[3] - 1)),
                       outline=_COLORS["roi"], width=max(1, scale))
    for record in tile_records:
        if record.get("status") not in ("complete", "valid", "success"):
            tile = record["tile"]
            draw.rectangle((*xy(tile["x0"], tile["y0"]), *xy(tile["x1"] - 1, tile["y1"] - 1)),
                           outline=_COLORS["failed"], width=max(1, scale))
    for index, point in enumerate(points, 1):
        x, y = xy(point["x"], point["y"])
        radius = 2.8 * scale
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=max(1, scale))
        if numbers:
            draw.text((x + radius + 1, y - radius), str(index), fill=color)
    return image


def render_frame(folder, raw, entry, result):
    """Write full-size previews, direct VLM coordinates and an auditable frame page.

    Previews use the inference display limits and never replace raw camera data.
    No coordinate refinement, filtering or fallback detections occur here.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    lo, hi = entry["display_limits"]
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("Invalid display limits")
    gray = np.rint(255 * np.clip((raw - lo) / (hi - lo), 0, 1)).astype(np.uint8)
    base = Image.fromarray(gray).convert("RGB")
    roi, records = entry["roi"], result.get("tiles", [])
    points, classical = result.get("points", []), result.get("classical_points", [])
    _save_image(folder / "raw.png", base)
    _save_image(folder / "vlm_overlay.png", _overlay(base, points, roi, records, _COLORS["vlm"]))
    _save_image(folder / "classical_overlay.png", _overlay(base, classical, roi, [], _COLORS["classical"]))
    x0, y0, x1, y1 = roi
    crop = base.crop((x0, y0, x1, y1))
    scale = 3
    crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.NEAREST)
    _save_image(folder / "vlm_roi.png", _overlay(crop, points, roi, records, _COLORS["vlm"], scale, (x0, y0), True))
    _save_image(folder / "classical_roi.png", _overlay(crop, classical, roi, [], _COLORS["classical"], scale, (x0, y0), True))
    csv_stream = io.StringIO(newline="")
    writer = csv.writer(csv_stream)
    writer.writerow(["index", "x_raw_px", "y_raw_px", "status"])
    writer.writerows((index, point["x"], point["y"], point.get("status", "proposed"))
                    for index, point in enumerate(points, 1))
    _atomic(folder / "vlm_coordinates.csv", csv_stream.getvalue().encode("utf-8-sig"))
    _text(folder / "display_metadata.json", json.dumps(dict(
        raw_path=entry["raw_path"], shape=entry["shape"], roi=roi,
        display_limits=entry["display_limits"], preview="8-bit display mapping, not raw ADU",
        coordinate_convention="Zero-based original-image pixel centers; x is column, y is row",
        coverage_complete=result.get("coverage_complete", False),
        full_frame_covered=result.get("full_frame_covered", False)), ensure_ascii=False, indent=2))
    try:
        raw_url = Path(entry["raw_path"]).resolve().as_uri()
    except ValueError:
        raw_url = ""
    status = result.get("status", "incomplete")
    complete = bool(result.get("coverage_complete"))
    body = '<p><a href="../../index.html">← 全部原图</a></p><h1>%s</h1><p class="muted">%s · %s</p>' % (
        _escape(entry["id"]), _escape(entry.get("split", "")), _escape(entry.get("group", "")))
    reference_available = result.get("classical_reference_available", True)
    body += _stats([("处理状态", _label(status)), ("VLM 候选点", len(points)),
                    ("Classical 对照点", len(classical) if reference_available else "不可用"),
                    ("VLM 生成耗时", _duration(result.get("inference_seconds")))])
    body += '<div class="notice%s">%s<br>这是原图坐标上的识别结果。模型内部使用 64×64 图块；蓝框是本张识别范围，框外未识别。绿色为 VLM 直接预测，黄色为 Classical 对照。红框代表未获得可靠输出的图块，不能将其视为无离子。<br>预览为 8 位显示映射，不是原始 ADU；未修改原始文件。</div>' % (
        '' if complete else ' warning', '计划识别范围内的图块均已正常返回。' if complete else '结果不完整：点数是当前得到的候选数，不是整幅图的离子总数。')
    body += '<p><strong>范围说明：</strong>%s · ROI %s · 原图 %s</p>' % (
        _escape(entry.get("domain_note", "")), _escape(roi), _escape(entry["shape"]))
    if not reference_available:
        body += '<p class="notice warning">本图未找到可靠晶体区域，Classical 参考不可用；没有对照点不表示没有离子，也不计算与空标签的一致性。VLM 已在后备搜索范围中运行，结果需人工检查。</p>'
    body += '<p><a href="vlm_coordinates.csv">下载 VLM 原图坐标 CSV</a> · <a href="display_metadata.json">显示与坐标元数据</a>'
    if raw_url:
        body += ' · <a href="%s">打开原始文件</a>' % _escape(raw_url)
    body += '</p><p class="muted">%s</p>' % _escape(entry["raw_path"])
    body += '<h2>完整原图</h2><div class="gallery">' + _figure("raw.png", "原始画面预览", "原尺寸像素；点击在新页面查看。")
    body += _figure("vlm_overlay.png", "VLM 原图识别", "绿色圆圈：预测中心；蓝框：识别范围。")
    body += _figure("classical_overlay.png", "Classical 原图对照", "黄色圆圈仅作对照，未用于替换 VLM 输出。") + '</div>'
    body += '<h2>识别范围放大</h2><div class="gallery">' + _figure("vlm_roi.png", "VLM · 3 倍像素放大", "编号与 VLM 坐标 CSV 一致。")
    body += _figure("classical_roi.png", "Classical · 3 倍像素放大", "最近邻放大，不增加物理分辨率。") + '</div>'
    body += '<p>与 Classical 的一致性 F1：1 px = <strong>%s</strong>；3 px = <strong>%s</strong>。这是与伪标签的一致性，不是真实识别准确率。</p>' % (
        _metric(result.get("agreement_1px")), _metric(result.get("agreement_3px")))
    audit = {"region": entry.get("region"), "domain_note": entry.get("domain_note"),
             "status": status, "coverage_scope": result.get("coverage_scope"),
             "failed_tiles": result.get("failed_tiles"), "abstained_tiles": result.get("abstained_tiles"),
             "tiles": [{k: r[k] for k in ("tile", "status", "error") if k in r} for r in records]}
    body += '<details><summary>区域与图块状态审计</summary><pre>%s</pre></details>' % _escape(json.dumps(audit, ensure_ascii=False, indent=2))
    _text(folder / "index.html", _document("原图识别 · " + entry["id"], body))


def write_report(output, plan, rows, state):
    """Refresh an index of every planned original, retaining pending/failed rows."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lookup = {row["id"]: row for row in rows}
    entries = plan["images"]
    entries_by_id = {entry["id"]: entry for entry in entries}
    completed = sum(bool(row.get("coverage_complete")) for row in rows)
    issues = sum(not bool(row.get("coverage_complete")) for row in rows)
    total_tiles = int(plan.get("planned_tiles", sum(len(entry.get("tiles", [])) for entry in entries)))
    processed_tiles = sum(int(row.get("tiles_processed", len(entries_by_id[row["id"]].get("tiles", [])))) for row in rows)
    seconds = sum(float(row.get("inference_seconds", 0)) for row in rows)
    summary = {"state": state, "updated_local": datetime.now().isoformat(timespec="seconds"),
               "total_images": len(entries), "processed_images": len(rows), "complete_images": completed,
               "incomplete_images": issues, "pending_images": len(entries) - len(rows),
               "planned_tiles": total_tiles, "processed_tiles": processed_tiles,
               "classical_reference_unavailable_images": sum(not row.get("classical_reference_available", True) for row in rows),
               "inference_seconds": seconds, "split_counts": dict(Counter(e.get("split", "") for e in entries)),
               "note": "Region-assisted VLM; outside ROI is not inferred. Classical agreement is not true accuracy.",
               "rows": rows}
    _text(output / "report_summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    body = '<p class="muted">微调 VLM · 原图识别</p><h1>全部原图识别结果</h1><p>%s · 更新于 %s</p>' % (
        _escape(_label(state)), _escape(summary["updated_local"]))
    body += _stats([("已处理 / 原图总数", "%d / %d" % (len(rows), len(entries))),
                    ("范围内完整结果", completed), ("需检查结果", issues),
                    ("已处理 / 计划图块", "%d / %d" % (processed_tiles, total_tiles))])
    body += '<progress value="%d" max="%d"></progress>' % (len(rows), max(1, len(entries)))
    body += '<div class="notice">逐张保存完整原图上的识别圈与全局坐标。内部使用 64×64 重叠图块，并映射、合并回原图；蓝框显示选定识别范围，框外未识别。绿色为 VLM 直接预测，Classical 单独作为对照。<br>表中 F1 是与 Classical 伪标签的一致性，不是真实识别准确率。训练集原图的结果不能作为独立泛化评估。预览采用 8 位显示映射，原始 ADU 文件保持不变。</div>'
    body += '<p><a href="report_summary.json">进度和逐图统计 JSON</a> · <a href="plan.json">原图及图块计划</a> · VLM 累计生成时间：%s</p>' % _escape(_duration(seconds))
    if state in ("prepared", "loading_model", "running"):
        body += '<p class="muted">此页每 30 秒自动刷新；每张原图处理完成后可点击查看。</p>'
    if issues:
        body += '<p class="notice warning">有 %d 张已处理图像存在失败或弃答。候选点数不代表这些图像的完整离子总数，请查看红框和图块审计。</p>' % issues
    splits = sorted({e.get("split", "") for e in entries})
    body += '<div class="controls"><input id="search" placeholder="搜索文件名、采集组、图像 ID" aria-label="搜索图像"><select id="split" aria-label="筛选集合"><option value="">全部集合</option>'
    body += ''.join('<option value="%s">%s</option>' % (_escape(s), _escape(s)) for s in splits)
    body += '</select><select id="status" aria-label="筛选状态"><option value="">全部状态</option><option value="complete">范围内完整</option><option value="issue">需检查</option><option value="pending">等待识别</option></select><span id="visible"></span></div>'
    body += '<div class="scroll"><table><thead><tr><th>原图 / 采集组</th><th>集合</th><th>状态</th><th>VLM 点数</th><th>Classical 点数</th><th>F1 · 1 px</th><th>F1 · 3 px</th><th>范围 / 提示</th></tr></thead><tbody id="frames">'
    for entry in entries:
        row = lookup.get(entry["id"])
        category = "pending" if row is None else "complete" if row.get("coverage_complete") else "issue"
        title = Path(entry["raw_path"]).name
        link = ('<a href="images/%s/index.html">%s</a>' % (quote(str(entry["id"]), safe=""), _escape(title))) if row else _escape(title)
        status_label = _label(row.get("status", "incomplete")) if row else _label("pending")
        values = [row.get("vlm_count", 0), row.get("classical_count", 0) if row.get("classical_reference_available", True) else "不可用", _metric(row.get("agreement_1px")), _metric(row.get("agreement_3px"))] if row else ["—"] * 4
        body += '<tr data-split="%s" data-status="%s"><td class="path">%s<br><small>%s<br>%s</small></td><td>%s</td><td class="%s">%s</td>%s<td class="path"><small>%s<br>%s</small></td></tr>' % (
            _escape(entry.get("split", "")), category, link, _escape(entry.get("group", "")), _escape(entry["id"]),
            _escape(entry.get("split", "")), "good" if category == "complete" else "bad" if category == "issue" else "muted", _escape(status_label),
            ''.join('<td>%s</td>' % _escape(v) for v in values), _escape(entry.get("roi")), _escape(entry.get("domain_note", "")))
    body += '</tbody></table></div><details><summary>推理设置</summary><pre>%s</pre></details>' % _escape(json.dumps(plan.get("settings", {}), ensure_ascii=False, indent=2))
    script = """<script>
const controls=['search','split','status'].map(id=>document.getElementById(id));
function filter(){const [q,split,status]=controls.map(el=>el.value.toLowerCase());let visible=0;
for(const row of document.querySelectorAll('#frames tr')){const show=(!q||row.textContent.toLowerCase().includes(q))&&(!split||row.dataset.split.toLowerCase()===split)&&(!status||row.dataset.status===status);row.hidden=!show;if(show)visible++;}
document.getElementById('visible').textContent=visible+' 张';}
controls.forEach(el=>el.addEventListener('input',filter));filter();
</script>"""
    _text(output / "index.html", _document("VLM 全部原图识别", body, state in ("prepared", "loading_model", "running"), script))
