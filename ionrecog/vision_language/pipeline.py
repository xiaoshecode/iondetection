"""Auditable tiled VLM proposals, raw-image checks, and matched classical controls.

No model is downloaded here. API mode uses a user-selected chat-completions
server; replay imports saved responses and is never presented as new inference.
"""
import base64
import hashlib
import html
import io
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..data import load_image
from ..classical.detection import detect_ions
from ..classical.reference import measure_sites
from .geometry import make_tiles, render_tile, parse_points, deduplicate, refine_points


PROMPT = """Locate visible fluorescent ion spots in this grayscale experimental image crop.
Return only JSON: {\"points\": [[x,y], ...], \"uncertain\": false}.
Coordinates use a 0 to 1000 normalized grid: x=0 is the center of the leftmost
original pixel, x=1000 the center of the rightmost original pixel; likewise y
is top to bottom. The displayed crop is an enlarged view of those original pixels.
Mark each individually supported ion spot once, near its fluorescence center.
Do not complete a regular lattice or add points because a lattice site is expected.
Do not mark diffuse background, streaks, isolated noise pixels or the image border.
For an empty/background-only crop return {\"points\": [], \"uncertain\": false}.
If blur/overlap prevents a reliable separation, set uncertain=true and include
only centers that have visible evidence. Do not infer quantum states or invisible ions.
Do not explain, draw, use bounding boxes, or return a count instead of coordinates."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_manifest(path, roles=None, max_images=None):
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != 1 or not isinstance(value.get("images"), list) or not value["images"]:
        raise ValueError("Manifest requires version=1 and a nonempty images list.")
    ids, entries = set(), []
    for entry in value["images"]:
        if not isinstance(entry, dict):
            raise ValueError("Each manifest image must be an object.")
        ident = entry.get("id", "")
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", ident) or ident in ids:
            raise ValueError("Image IDs must be unique safe file names of at most 101 characters.")
        ids.add(ident)
        if not isinstance(entry.get("path"), str) or not isinstance(entry.get("shape"), list):
            raise ValueError("Manifest image requires path and shape.")
        raw_path = Path(entry["path"])
        raw_path = raw_path.resolve() if raw_path.is_absolute() else (source.parent / raw_path).resolve()
        if not raw_path.is_file() or sha256_file(raw_path) != entry.get("sha256"):
            raise ValueError("Missing or modified raw image: " + str(raw_path))
        if roles and entry.get("role") not in roles:
            continue
        entries.append(dict(entry, path=str(raw_path)))
    if max_images is not None:
        if max_images < 1:
            raise ValueError("--max-images must be positive.")
        entries = entries[:max_images]
    if not entries:
        raise ValueError("No images match the selected roles.")
    return entries


def validate_url(base_url):
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Use an http(s) base URL without credentials, query, or fragment.")
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Remote API servers require HTTPS; HTTP is supported for loopback servers.")
    return base_url.rstrip("/") + "/chat/completions"


def request_points(image, base_url, model, max_tokens=2048, timeout=180, api_key_env=None):
    endpoint = validate_url(base_url)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    body = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
    ]}], "temperature": 0, "max_tokens": max_tokens, "stream": False,
        "seed": 42}
    headers = {"Content-Type": "application/json"}
    if api_key_env:
        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError("API credential environment variable is not set: " + api_key_env)
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(endpoint, data=json.dumps(body).encode("utf-8"), headers=headers)
    # A local model server must not traverse a machine-wide HTTP proxy.
    local = urllib.parse.urlsplit(endpoint).hostname in ("127.0.0.1", "localhost", "::1")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if local else urllib.request.build_opener()
    start = time.perf_counter()
    try:
        with opener.open(request, timeout=timeout) as response:
            data = response.read(4 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as exc:
        raise ValueError("VLM server returned HTTP %d; no detections accepted for this tile." % exc.code) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ValueError("VLM server could not complete the request (%s)." % type(exc).__name__) from exc
    if len(data) > 4 * 1024 * 1024:
        raise ValueError("VLM response exceeded 4 MiB.")
    value = json.loads(data)
    try:
        choice = value["choices"][0]
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("VLM response content must be text.")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("VLM server returned no usable message.") from exc
    return {"text": content, "finish_reason": choice.get("finish_reason"),
            "response_model": value.get("model"), "usage": value.get("usage"),
            "seconds": time.perf_counter() - start}


def classical_controls(raw, tiles, roi, detection, merge_distance):
    x0, y0, x1, y1 = roi
    region, _ = detect_ions(raw[y0:y1, x0:x1], detection)
    for point in region:
        point.update(x=point["x"] + x0, y=point["y"] + y0)
    tiled = []
    for tile in tiles:
        points, _ = detect_ions(raw[tile["y0"]:tile["y1"], tile["x0"]:tile["x1"]], detection)
        for point in points:
            point["tile_margin"] = min(point["x"], point["y"], tile["x1"] - tile["x0"] - 1 - point["x"], tile["y1"] - tile["y0"] - 1 - point["y"])
            point.update(x=point["x"] + tile["x0"], y=point["y"] + tile["y0"], tile_id=tile["id"])
        tiled.extend(points)
    return region, deduplicate(tiled, min_distance=merge_distance)


def write_points(path, image_id, shape, method, points, complete, note, abstained_tiles=0):
    from ..cli import write_csv
    rows = [dict(point, id=i + 1) for i, point in enumerate(points)]
    save_json(path, {"image": image_id, "shape": list(shape), "method": method,
                     "coverage_complete": complete, "ions": rows, "note": note,
                     "has_abstentions": abstained_tiles > 0, "abstained_tiles": abstained_tiles})
    write_csv(Path(path).with_suffix(".csv"), rows)


def comparison_image(raw, roi, limits, sets):
    """Shared raw display limits and identical field of view in every panel."""
    x0, y0, x1, y1 = roi
    lo, hi = limits
    gray = (np.clip((raw[y0:y1, x0:x1] - lo) / max(hi - lo, 1e-12), 0, 1) * 255).round().astype("uint8")
    base = Image.fromarray(gray).convert("RGB")
    scale = min(3., 420. / max(base.height, base.width))
    width, height = max(1, round(base.width * scale)), max(1, round(base.height * scale))
    canvas = Image.new("RGB", (len(sets) * width, height + 44), "#111923")
    draw = ImageDraw.Draw(canvas)
    for column, (label, points, color) in enumerate(sets):
        panel = base.resize((width, height), resample=Image.Resampling.NEAREST)
        pdraw = ImageDraw.Draw(panel)
        for point in points:
            x = (point["x"] - x0 + .5) * width / base.width - .5
            y = (point["y"] - y0 + .5) * height / base.height - .5
            radius = max(2, float(point.get("radius", 3)) * scale)
            pdraw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=1)
        canvas.paste(panel, (column * width, 44))
        draw.text((column * width + 4, 4), label, fill="white")
        draw.text((column * width + 4, 20), "n=%d" % len(points) if column else "same raw display", fill="white")
    return canvas


def write_report(output, summaries, settings, status):
    from ..cli import write_csv
    save_json(output / "summary.json", {"status": status, "settings": settings,
        "images": summaries, "metrics_are_ground_truth_accuracy": False,
        "note": "Candidate counts only. No independent real-image truth. Partial/failed coverage must not be treated as empty predictions. Raw support is heuristic, not calibrated confidence or occupancy."})
    write_csv(output / "summary.csv", summaries)
    rows = []
    for row in summaries:
        ident = html.escape(row["id"])
        rows.append('<article><h2>%s — %s</h2><p>role=%s; tiles=%d/%d; classical=%d; tiled classical=%d; VLM=%d; raw-supported=%d; uncertain=%d; abstained tiles=%d</p>'
                    '<img src="images/%s/comparison.png"><p><a href="images/%s/vlm_direct_ions.json">VLM coordinates</a> · '
                    '<a href="images/%s/vlm_checked_ions.json">raw-supported coordinates</a> · '
                    '<a href="images/%s/uncertain.json">uncertain proposals</a></p></article>' %
                    (ident, html.escape(row["status"]), html.escape(row["role"]), row["tiles_complete"], row["tiles_planned"],
                     row["classical_count"], row["classical_tiled_count"], row["vlm_direct_count"], row["vlm_checked_count"],
                     row["uncertain_count"], row["abstained_tiles"], ident, ident, ident, ident))
    document = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Ion VLM pilot</title>
<style>body{font:16px system-ui;background:#101720;color:#eef3fa;max-width:1500px;margin:30px auto;padding:20px}a{color:#80caff}img{max-width:100%%;image-rendering:auto}article{border-top:1px solid #465260;margin-top:30px}p{line-height:1.6}</style>
<h1>离子 VLM 小规模对照</h1><p>状态：%s。模型：%s。候选数不代表真实离子数或准确率；尚无独立真值。</p>
<p>从左到右：原图、相同区域 classical、相同分块 classical、VLM 原始候选、原图支持候选、不确定候选。所有面板显示同一原图、使用相同强度映射。</p>
<p>selection 模式使用历史显示 ROI（人工先验），不是 VLM 自动找出的区域；legacy 图无该 ROI 时使用全图。失败或未完成的分块会明确保留，不能解释成没有离子。支持度阈值、质心精修及去重都可能误伤弱点或近邻点，需独立验证。</p>
<p><a href="summary.csv">逐帧统计 CSV</a> · <a href="summary.json">参数和状态</a> · <a href="plan.json">原图来源与完整请求计划</a></p>%s</html>''' % (html.escape(status), html.escape(settings["model"]), "\n".join(rows))
    (output / "index.html").write_text(document, encoding="utf-8")


def run_vlm(args):
    from ..cli import check_destination, load_config
    for name in ("max_tokens", "timeout", "display_scale"):
        if getattr(args, name) <= 0:
            raise ValueError("--%s must be positive." % name.replace("_", "-"))
    if args.max_requests is not None and args.max_requests < 1:
        raise ValueError("--max-requests must be positive.")
    for name in ("merge_distance", "radius", "max_shift", "min_snr"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError("--%s must be positive and finite." % name.replace("_", "-"))
    if (args.display_low is None) != (args.display_high is None):
        raise ValueError("Supply both --display-low and --display-high.")
    if args.display_low is not None and (not np.isfinite([args.display_low, args.display_high]).all() or args.display_high <= args.display_low):
        raise ValueError("Require finite display-high > display-low.")
    if args.backend == "api":
        validate_url(args.base_url)
        if args.coordinate_system != "normalized_1000":
            raise ValueError("API prompt requires normalized_1000; pixels is only supported for replay imports.")
        if args.api_key_env and not os.environ.get(args.api_key_env):
            raise ValueError("API credential environment variable is not set: " + args.api_key_env)
    if not np.isfinite(args.min_free_gib) or args.min_free_gib < 0:
        raise ValueError("--min-free-gib must be finite and nonnegative.")
    entries = read_manifest(args.manifest, args.roles, args.max_images)
    sources = [args.manifest] + [entry["path"] for entry in entries]
    if args.responses:
        sources.append(args.responses)
    output = check_destination(args.output, sources)
    def check_disk():
        parent = output
        while not parent.exists():
            parent = parent.parent
        free = shutil.disk_usage(parent).free / 1024 ** 3
        if free < args.min_free_gib:
            raise ValueError("Disk free space %.2f GiB is below reserve %.2f GiB; stopping without deleting results." % (free, args.min_free_gib))
        return free
    free_before = check_disk()
    for entry in entries:
        raw_parent = Path(entry["path"]).parent
        if output == raw_parent or raw_parent in output.parents:
            raise ValueError("VLM output must not be inside a raw-image directory.")
    config = load_config(args.config)
    runtime_path = Path(args.runtime_metadata).resolve() if args.runtime_metadata else None
    if args.backend == "api" and runtime_path is None and urllib.parse.urlsplit(args.base_url).hostname in ("localhost", "127.0.0.1", "::1"):
        default_runtime = Path(__file__).resolve().parents[2] / ".vlm-runtime" / "active_server.json"
        if default_runtime.is_file():
            runtime_path = default_runtime
    runtime_metadata = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path else None
    settings = {key: getattr(args, key) for key in ("backend", "base_url", "model", "tile_size", "overlap", "display_scale",
        "roi_mode", "coordinate_system", "display_low", "display_high", "merge_distance", "radius", "max_shift", "min_snr", "max_tokens", "roles", "max_images")}
    settings.update(prompt=PROMPT, detection=config["detection"], manifest_sha256=sha256_file(args.manifest),
                    responses_sha256=sha256_file(args.responses) if args.responses else None,
                    runtime_metadata=runtime_metadata,
                    code_sha256={name: sha256_file(Path(__file__).resolve().parents[1] / name) for name in
                                 ("vision_language/pipeline.py", "vision_language/geometry.py",
                                  "classical/detection.py", "classical/reference.py", "data.py")},
                    protocol_version=1)
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            raise ValueError("VLM output is not empty. Use a new directory or --resume with identical settings.")
        saved = json.loads((output / "plan.json").read_text(encoding="utf-8"))
        if saved.get("fingerprint") != fingerprint:
            raise ValueError("Resume settings/source manifest differ from this run; choose a new output.")
    elif args.resume:
        raise ValueError("--resume requires an existing run with plan.json.")
    if args.backend == "replay" and not args.responses:
        raise ValueError("Replay requires --responses JSON keyed by image_id/tile_id, containing text and finish_reason.")
    replay = json.loads(Path(args.responses).read_text(encoding="utf-8")) if args.responses else {}
    plans = []
    for entry in entries:
        raw = load_image(entry["path"])
        if list(raw.shape) != entry["shape"]:
            raise ValueError("Manifest shape differs from raw image: " + entry["id"])
        roi = entry.get("selection_roi") if args.roi_mode == "selection" else entry.get("roi")
        roi = roi if roi is not None else [0, 0, raw.shape[1], raw.shape[0]]
        tiles = make_tiles(raw.shape, args.tile_size, args.overlap, roi=roi)
        lo, hi = (args.display_low, args.display_high) if args.display_low is not None else map(float, np.percentile(raw, [.5, 99.9]))
        if hi <= lo:
            lo, hi = float(raw.min()), float(raw.max()) + 1.
        plans.append(dict(entry, active_roi=roi, display_limits=[lo, hi], tiles=tiles))
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "plan.json", {"fingerprint": fingerprint, "settings": settings, "images": plans,
        "planned_requests": sum(len(p["tiles"]) for p in plans),
        "roi_note": "Historical selection_roi is a manual prior, not VLM predicted. Full mode honors explicit manifest roi when provided."})
    save_json(output / "disk_space.json", {"free_gib_before": free_before, "reserve_gib": args.min_free_gib,
        "note": "Checked before preparation and each tile. Raw acquisitions/model weights are not copied into this run."})
    summaries, requests = [], 0
    for entry in plans:
        raw = load_image(entry["path"])
        folder = output / "images" / entry["id"]
        folder.mkdir(parents=True, exist_ok=True)
        tile_records, proposed = [], []
        start = time.perf_counter()
        for tile in entry["tiles"]:
            check_disk()
            tile_id = str(tile["id"])
            response_path = folder / "tiles" / (tile_id + ".json")
            rendered = render_tile(raw, tile, *entry["display_limits"], display_scale=args.display_scale)
            response_path.parent.mkdir(parents=True, exist_ok=True)
            rendered.save(response_path.with_suffix(".png"))
            record = {"tile": tile, "status": "pending", "backend": args.backend}
            cached = json.loads(response_path.read_text(encoding="utf-8")) if args.resume and response_path.exists() else None
            if cached and cached.get("status") == "complete":
                record = cached
            elif args.backend != "dry-run" and (args.max_requests is None or requests < args.max_requests):
                requests += 1
                print("[%s tile %s] %s request %d" % (entry["id"], tile_id, args.backend, requests), flush=True)
                try:
                    response = (request_points(rendered, args.base_url, args.model, args.max_tokens, args.timeout, args.api_key_env)
                                if args.backend == "api" else replay[entry["id"] + "/" + tile_id])
                    record["response"] = response
                    if response.get("finish_reason") != "stop":
                        raise ValueError("Response did not finish normally (possibly truncated); tile not accepted.")
                    points = parse_points(response["text"], tile, args.coordinate_system, args.display_scale)
                    envelope_text = response["text"].strip()
                    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", envelope_text, flags=re.S | re.I)
                    envelope = json.loads(fenced.group(1).strip() if fenced else envelope_text)
                    response_uncertain = envelope.get("uncertain", False) if isinstance(envelope, dict) else False
                    for point in points:
                        point["tile_id"] = tile_id
                    record.update(status="complete", points=points, response_uncertain=response_uncertain)
                except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
                    record.update(status="failed", error=str(exc))
            save_json(response_path, record)
            tile_records.append({"id": tile_id, "status": record["status"], "error": record.get("error"),
                                 "response_uncertain": record.get("response_uncertain", False)})
            if record["status"] == "complete":
                proposed.extend(record["points"])
        complete_tiles = sum(r["status"] == "complete" for r in tile_records)
        failed_tiles = sum(r["status"] == "failed" for r in tile_records)
        abstained_tiles = sum(bool(r["response_uncertain"]) for r in tile_records)
        coverage_complete = complete_tiles == len(tile_records)
        direct = deduplicate(proposed, min_distance=args.merge_distance)
        measurements = measure_sites(raw, direct, radius=args.radius)
        for point, measured in zip(direct, measurements):
            point.update(measured)
        accepted, uncertain = refine_points(raw, direct, radius=args.radius, max_shift=args.max_shift, min_snr=args.min_snr)
        classical, classical_tiled = classical_controls(raw, entry["tiles"], entry["active_roi"], config["detection"], args.merge_distance)
        note = "VLM candidates; raw SNR is heuristic. No calibrated accuracy, confidence, quantum-state or occupancy labels."
        for method, points in (("classical", classical), ("classical_tiled", classical_tiled), ("vlm_direct", direct), ("vlm_checked", accepted)):
            write_points(folder / (method + "_ions.json"), entry["id"], raw.shape, method, points,
                         True if method.startswith("classical") else coverage_complete, note,
                         0 if method.startswith("classical") else abstained_tiles)
        save_json(folder / "uncertain.json", {"ions": uncertain, "note": "Excluded from raw-supported list, not proven absent; retain for review."})
        save_json(folder / "tile_status.json", tile_records)
        panel = comparison_image(raw, entry["active_roi"], entry["display_limits"], [
            ("Raw", [], "white"), ("Classical ROI", classical, "#55ff88"), ("Classical tiled", classical_tiled, "#55bbff"),
            ("VLM direct", direct, "#ffcc55"), ("Raw supported", accepted, "#55ff88"), ("Uncertain", uncertain, "#ff8866")])
        panel.save(folder / "comparison.png")
        row = {"id": entry["id"], "role": entry.get("role", "unspecified"), "group": entry.get("group", ""),
               "status": ("model_abstained" if abstained_tiles else "complete") if coverage_complete else ("failed_tiles" if failed_tiles else "pending_tiles"),
               "tiles_planned": len(tile_records), "tiles_complete": complete_tiles, "tiles_failed": failed_tiles,
               "abstained_tiles": abstained_tiles,
               "classical_count": len(classical), "classical_tiled_count": len(classical_tiled),
               "vlm_direct_count": len(direct), "vlm_checked_count": len(accepted), "uncertain_count": len(uncertain),
               "processing_seconds_this_invocation": time.perf_counter() - start,
               "inference_seconds_cached_total": sum(json.loads((folder / "tiles" / (str(t["id"]) + ".json")).read_text(encoding="utf-8")).get("response", {}).get("seconds", 0.) for t in entry["tiles"]),
               "raw_path": entry["path"], "coverage_complete": coverage_complete}
        summaries.append(row)
        write_report(output, summaries, settings, "running")
        print("%s: %s, VLM %d, raw-supported %d, uncertain %d" %
              (row["id"], row["status"], len(direct), len(accepted), len(uncertain)), flush=True)
    status = "complete" if all(row["coverage_complete"] for row in summaries) else "incomplete"
    if status == "complete" and any(row["abstained_tiles"] for row in summaries):
        status = "complete_with_abstentions"
    if args.backend == "dry-run":
        status = "prepared_not_inferred"
    write_report(output, summaries, settings, status)
    save_json(output / "disk_space.json", {"free_gib_before": free_before, "free_gib_after": check_disk(),
        "reserve_gib": args.min_free_gib,
        "output_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file())})
    print("VLM pilot %s: %s" % (status, output / "index.html"), flush=True)


def add_vlm_parser(sub):
    parser = sub.add_parser("vlm", help="Tiled VLM proposals, raw checks, and classical controls (separate from restoration)")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("api", "replay", "dry-run"), default="api")
    parser.add_argument("--base-url", default="http://127.0.0.1:8087/v1")
    parser.add_argument("--model", default="local-vlm")
    parser.add_argument("--api-key-env", help="Name of credential environment variable; never stored in outputs")
    parser.add_argument("--runtime-metadata", help="JSON with model hashes and server launch configuration; included in resume fingerprint")
    parser.add_argument("--responses", help="Replay JSON keyed by image_id/tile_id")
    parser.add_argument("--config")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--display-scale", type=int, default=3)
    parser.add_argument("--display-low", type=float)
    parser.add_argument("--display-high", type=float)
    parser.add_argument("--roi-mode", choices=("full", "selection"), default="full")
    parser.add_argument("--coordinate-system", choices=("normalized_1000", "pixels"), default="normalized_1000")
    parser.add_argument("--merge-distance", type=float, default=2.)
    parser.add_argument("--radius", type=float, default=3.)
    parser.add_argument("--max-shift", type=float, default=2.)
    parser.add_argument("--min-snr", type=float, default=5.)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-requests", type=int, help="Maximum NEW tile requests this invocation; others remain pending")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--roles", nargs="+")
    parser.add_argument("--resume", action="store_true", help="Reuse successful responses with identical source hashes/model/prompt/settings")
    parser.add_argument("--min-free-gib", type=float, default=5., help="Stop before writing tiles below this free-space reserve; never delete prior data")
