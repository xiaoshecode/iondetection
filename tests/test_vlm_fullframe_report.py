"""Full-frame coordinate display, incomplete coverage and unprocessed-row checks."""
import csv
import json

import numpy as np
from PIL import Image

from ionrecog.vision_language.fullframe_report import render_frame, write_report


def _entry(image_id="frame-a"):
    return {"id": image_id, "raw_path": "data/test/<untrusted>.png", "split": "test",
            "group": "<script>alert(1)</script>", "shape": [96, 128], "roi": [16, 24, 80, 88],
            "display_limits": [0., 15000.], "region": {"status": "no_crystal"},
            "domain_note": "fallback <search>", "tiles": [{"id": 0, "x0": 16, "y0": 24, "x1": 80, "y1": 88}]}


def _result():
    return {"id": "frame-a", "status": "incomplete", "coverage_complete": False,
            "full_frame_covered": False, "vlm_count": 1, "classical_count": 0,
            "failed_tiles": 1, "abstained_tiles": 0, "inference_seconds": 12.3,
            "tiles_processed": 1, "classical_reference_available": False,
            "points": [{"x": 32., "y": 40., "status": "proposed"}], "classical_points": [],
            "tiles": [{"tile": {"x0": 16, "y0": 24, "x1": 80, "y1": 88}, "status": "truncated"}],
            "agreement_1px": {"f1": None}, "agreement_3px": {"f1": None}}


def test_original_coordinates_full_size_and_failed_tile_are_preserved(tmp_path):
    raw = np.arange(96 * 128, dtype=np.float32).reshape(96, 128)
    before = raw.copy()
    render_frame(tmp_path, raw, _entry(), _result())
    assert np.array_equal(raw, before)
    with Image.open(tmp_path / "raw.png") as preview:
        assert preview.size == (128, 96)
    with Image.open(tmp_path / "vlm_overlay.png") as overlay:
        assert overlay.size == (128, 96)
        assert overlay.getpixel((29, 40)) == (91, 255, 158)  # raw x/y, not crop-local
        assert overlay.getpixel((16, 30)) == (255, 97, 94)  # failed tile retained
    with Image.open(tmp_path / "vlm_roi.png") as crop:
        assert crop.size == (192, 192)
    with (tmp_path / "vlm_coordinates.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["x_raw_px"] == "32.0"
    assert rows[0]["y_raw_px"] == "40.0"
    document = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document
    assert "<script>alert(1)</script>" not in document
    assert "结果不完整" in document and "Classical 参考不可用" in document
    assert "0.0000" not in document
    assert "http-equiv=\"refresh\"" not in document


def test_progress_lists_pending_frames_without_zero_predictions(tmp_path):
    entry, pending = _entry(), _entry("frame-b")
    result = _result()
    row = {k: v for k, v in result.items() if k not in ("points", "classical_points", "tiles")}
    plan = {"images": [entry, pending], "planned_tiles": 2, "settings": {"image_tokens": 1024}}
    write_report(tmp_path, plan, [row], "running")
    summary = json.loads((tmp_path / "report_summary.json").read_text(encoding="utf-8"))
    assert summary["processed_images"] == 1 and summary["pending_images"] == 1
    assert summary["complete_images"] == 0 and summary["incomplete_images"] == 1
    assert summary["classical_reference_unavailable_images"] == 1
    document = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert 'href="images/frame-a/index.html"' in document
    assert 'href="images/frame-b/index.html"' not in document
    assert '<td>—</td><td>—</td><td>—</td><td>—</td>' in document
    assert 'http-equiv="refresh"' in document
    assert "frame-b" in document
    write_report(tmp_path, plan, [row], "failed")
    assert 'http-equiv="refresh"' not in (tmp_path / "index.html").read_text(encoding="utf-8")
