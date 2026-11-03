import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


spec = importlib.util.spec_from_file_location("prepare_vlm_pilot", Path(__file__).parents[1] / "scripts/prepare_vlm_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def write_images(root, subdir, count, suffix=".png"):
    target = root / subdir
    target.mkdir(parents=True)
    files = []
    for i in range(count):
        path = target / ("frame_%03d" % i + suffix)
        raw = np.arange(600, dtype=np.uint16).reshape(20, 30) + i
        if suffix == ".npy":
            np.save(str(path), raw)
        else:
            Image.fromarray(raw).save(str(path))
        files.append(path)
    return files


def test_balanced_reproducible_selection_and_explicit_failure_role(tmp_path):
    root = tmp_path / "data"
    write_images(root, "val/session_v", 10)
    test = write_images(root, "test/session_t", 50)
    write_images(root, "test/legacy_20260116", 13, ".npy")
    write_images(root, "diagnostic/session_d", 5)
    report = tmp_path / "failure.json"
    report.write_text(json.dumps({"relative": "session_t/" + test[1].name,
                                  "selection_rule": "previous failure"}))
    selected = pilot.select_images(root, 32, report)
    assert selected == pilot.select_images(root, 32, report)
    assert len(selected) == 32
    assert len({r["path"] for r in selected}) == 32
    assert {r["role"] for r in selected[:5]} == {"calibration", "pilot_test", "diagnostic"}
    assert all("val" in r["path"].parts for r in selected if r["role"] == "calibration")
    failure = next(r for r in selected if r["path"] == test[1].resolve())
    assert failure["role"] == "diagnostic"
    assert "not_representative" in failure["selection_reason"]
    indices = [r["selection_index"] for r in selected if r["selection_reason"] == "test_png_uniform"]
    assert indices[0] == 0 and indices[-1] == 49
    assert len(pilot.select_images(root, 3, report)) == 3


def test_manifest_hashes_no_source_changes_and_output_guards(tmp_path):
    root = tmp_path / "data"
    sources = write_images(root, "test/session_t", 3)
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    output = tmp_path / "out"
    result = pilot.prepare(root, output, 2)
    assert (output / "contact_sheet.png").is_file()
    assert json.loads((output / "manifest.json").read_text()) == result
    assert len(result["images"]) == 2
    for row in result["images"]:
        assert row["roi"] is None
        assert row["ground_truth_available"] is False
        assert row["sha256"] == hashes[Path(row["path"])]
        assert row["shape"] == [20, 30]
        assert row["display"]["hi"] > row["display"]["lo"]
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources} == hashes
    assert set(p.name for p in output.iterdir()) == {"manifest.json", "contact_sheet.png"}
    with pytest.raises(ValueError, match="empty"):
        pilot.prepare(root, output)
    with pytest.raises(ValueError, match="separate"):
        pilot.prepare(root, root / "output")
    with pytest.raises(ValueError, match="separate"):
        pilot.prepare(root, tmp_path)


def test_failure_cannot_escape_test_root_or_duplicate_test_role(tmp_path):
    root = tmp_path / "data"
    test = write_images(root, "test/session_t", 3)
    report = tmp_path / "failure.json"
    report.write_text(json.dumps({"relative": "session_t/" + test[0].name}))
    rows = pilot.select_images(root, 32, report)
    assert len(rows) == 3
    assert sum(r["path"] == test[0].resolve() for r in rows) == 1
    assert next(r for r in rows if r["path"] == test[0].resolve())["role"] == "diagnostic"
    report.write_text(json.dumps({"relative": "../../outside.png"}))
    with pytest.raises(ValueError, match="inside"):
        pilot.select_images(root, 32, report)


def test_constant_preview_and_invalid_limits():
    raw = np.full((8, 9), 1234, dtype=np.float32)
    mapping = pilot.display_mapping(raw)
    assert mapping["constant_image_is_black"]
    assert not np.asarray(pilot.preview(raw, mapping)).any()
    assert pilot.uniform_indices(5, 10) == [0, 1, 2, 3, 4]
    assert pilot.uniform_indices(0, 3) == []
    with pytest.raises(ValueError):
        pilot.select_images("missing", 0)


def test_study_regions_preserve_selection_and_record_partial_evaluation(tmp_path):
    root = tmp_path / "data"
    pngs = root / "test/session_t"
    npys = root / "test/legacy_20260116"
    pngs.mkdir(parents=True)
    npys.mkdir(parents=True)
    Image.fromarray(np.zeros((512, 512), dtype=np.uint16)).save(str(pngs / "frame.png"))
    np.save(str(npys / "frame.npy"), np.zeros((300, 800), dtype=np.uint16))
    full = pilot.prepare(root, tmp_path / "full")
    partial = pilot.prepare(root, tmp_path / "partial", study_regions=True)
    assert partial["study_regions"] is True
    assert partial["full_frame_evaluation"] is False
    assert [r["id"] for r in full["images"]] == [r["id"] for r in partial["images"]]
    assert [r["sha256"] for r in full["images"]] == [r["sha256"] for r in partial["images"]]
    for row in partial["images"]:
        assert row["roi"] == ([180, 220, 300, 430] if row["shape"] == [512, 512] else [0, 86, 800, 214])
        assert row["full_frame_evaluation"] is False
        assert row["roi_origin"] == "predefined_manual_study_region_shared_by_all_methods"
        assert row["ground_truth_available"] is False
    with pytest.raises(ValueError, match="only defines"):
        pilot.study_region([128, 128])
