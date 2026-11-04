"""Scientific data failure modes for the server diffusion manifest."""

import json

import numpy as np
import pytest
from PIL import Image

from ionrecog.diffusion.data import PatchSampler, audit_manifest, load_manifest, read_case


def _save(tmp_path, name, value):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(value, dtype=np.float32))
    return name


def _manifest(tmp_path, rows):
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _pair(tmp_path, rid="a", split="train", group="g", base=0):
    values = np.arange(256, dtype=np.float32).reshape(16, 16) + base
    return {"id": rid, "split": split, "group_id": group,
            "input": _save(tmp_path, rid + "_in.npy", values),
            "target": _save(tmp_path, rid + "_target.npy", values + 3),
            "target_kind": "clean"}


def test_audit_resolves_relative_paths_without_changing_raw_scale(tmp_path):
    pair = _pair(tmp_path, base=1000)
    manifest = _manifest(tmp_path, [pair])
    records = load_manifest(manifest)
    case = read_case(records[0])
    assert case["observed"].dtype == np.float32
    assert case["observed"][0, 0] == 1000
    assert case["observed"].shape == (16, 16)
    assert records[0]["input"] == str((tmp_path / pair["input"]).resolve())
    report = audit_manifest(manifest)
    assert report["splits"]["train"] == {"records": 1, "groups": 1, "unique_images": 2, "with_points": 0}
    assert report["normalization_range"] == {"min": 1000.0, "max": 1258.0}
    json.dumps(report, allow_nan=False)


def test_reject_same_crystal_group_across_splits(tmp_path):
    first = _pair(tmp_path)
    second = _pair(tmp_path, "b", "val", "g", 400)
    with pytest.raises(ValueError, match="Cross-split group leakage"):
        load_manifest(_manifest(tmp_path, [first, second]))


def test_detect_copied_pixels_even_after_png_reencoding(tmp_path):
    first = _pair(tmp_path)
    second = _pair(tmp_path, "b", "val", "different", 500)
    Image.fromarray(np.arange(256, dtype=np.uint16).reshape(16, 16)).save(tmp_path / "copy.png")
    second["input"] = "copy.png"
    with pytest.raises(ValueError, match="Cross-split decoded content leakage"):
        load_manifest(_manifest(tmp_path, [first, second]))


def test_duplicate_ids_and_self_target_are_rejected(tmp_path):
    first = _pair(tmp_path)
    with pytest.raises(ValueError, match="Duplicate record id"):
        load_manifest(_manifest(tmp_path, [first, first]))
    first["target"] = first["input"]
    with pytest.raises(ValueError, match="same path"):
        load_manifest(_manifest(tmp_path, [first]))


@pytest.mark.parametrize("value", [None, [], [[1.25, 2.75]]])
def test_missing_points_are_not_mislabelled_as_empty_scene(tmp_path, value):
    pair = _pair(tmp_path)
    if value is not None:
        pair["points"] = value
    record = load_manifest(_manifest(tmp_path, [pair]))[0]
    points = read_case(record)["points"]
    if value is None:
        assert points is None
    else:
        assert points.shape == (len(value), 2)


@pytest.mark.parametrize("points", [[[16, 4]], [[-0.1, 3]], [[1, float("nan")]], [[2, 3], [2, 3]], [1, 2]])
def test_invalid_or_duplicate_point_labels_rejected(tmp_path, points):
    pair = _pair(tmp_path)
    pair["points"] = points
    with pytest.raises(ValueError):
        load_manifest(_manifest(tmp_path, [pair]))


@pytest.mark.parametrize("target", [np.zeros((8, 8)), np.full((16, 16), np.nan), np.zeros((16, 16, 3))])
def test_shape_and_finite_image_checks(tmp_path, target):
    pair = _pair(tmp_path)
    _save(tmp_path, pair["target"], target)
    with pytest.raises(ValueError):
        load_manifest(_manifest(tmp_path, [pair]))


def test_repeat_reference_never_contains_its_input(tmp_path):
    frames = [_save(tmp_path, "f{}.npy".format(i), np.full((8, 8), i + 1)) for i in range(5)]
    row = {"id": "repeats", "group_id": "crystal", "split": "train", "frames": frames, "stable": True}
    record = load_manifest(_manifest(tmp_path, [row]))[0]
    for seed in range(10):
        case = read_case(record, seed, max_target_frames=2)
        assert case["input_path"] not in case["target_paths"]
        assert len(case["target_paths"]) == 2
        expected = np.mean([np.load(path) for path in case["target_paths"]], axis=0)
        np.testing.assert_array_equal(case["target"], expected)
        assert case["target_kind"] == "repeat_mean"
        np.testing.assert_array_equal(case["target"], read_case(record, seed, 2)["target"])


def test_repeat_requires_stability_assertion_and_distinct_frames(tmp_path):
    frames = [_save(tmp_path, "f{}.npy".format(i), np.full((8, 8), i)) for i in range(3)]
    row = {"id": "repeats", "group_id": "crystal", "split": "train", "frames": frames}
    with pytest.raises(ValueError, match="stable:true"):
        load_manifest(_manifest(tmp_path, [row]))
    row["stable"] = True
    np.save(tmp_path / frames[1], np.load(tmp_path / frames[0]))
    with pytest.raises(ValueError, match="pixel-identical"):
        load_manifest(_manifest(tmp_path, [row]))


def test_patch_pairs_transform_together_and_use_one_scale(tmp_path):
    pair = _pair(tmp_path)
    records = load_manifest(_manifest(tmp_path, [pair]))
    sampler = PatchSampler(records, patch_size=8, offset=0, scale=1000, augment=True)
    batch = sampler.batch(seed=12, batch_size=4)
    assert batch["observed"].shape == (4, 1, 8, 8)
    np.testing.assert_allclose(batch["target"].numpy() - batch["observed"].numpy(), 0.003, atol=3e-8)
    np.testing.assert_array_equal(batch["observed"].numpy(), sampler.batch(12, 4)["observed"].numpy())
    assert not np.array_equal(batch["observed"].numpy(), sampler.batch(13, 4)["observed"].numpy())
    assert batch["clipped_fraction"] == {"observed": 0, "target": 0}


def test_sampler_reports_clipping_and_does_not_touch_global_rng(tmp_path):
    records = load_manifest(_manifest(tmp_path, [_pair(tmp_path)]))
    sampler = PatchSampler(records, patch_size=16, offset=0, scale=100)
    np.random.seed(123)
    expected = np.random.random(3)
    np.random.seed(123)
    batch = sampler.batch(99, 1)
    np.testing.assert_array_equal(np.random.random(3), expected)
    assert batch["clipped_fraction"]["observed"] == 155 / 256
    assert float(batch["observed"].max()) == 1.0
    with pytest.raises(ValueError, match="smaller than patch_size"):
        PatchSampler(records, patch_size=32, offset=0, scale=100)


def test_audit_fingerprint_detects_changed_target_and_labels(tmp_path):
    pair = _pair(tmp_path)
    path = _manifest(tmp_path, [pair])
    original = audit_manifest(path)["content_fingerprint"]
    np.save(tmp_path / pair["target"], np.ones((16, 16), dtype=np.float32))
    changed = audit_manifest(path)["content_fingerprint"]
    assert changed != original
    pair["points"] = [[3, 4]]
    assert audit_manifest(_manifest(tmp_path, [pair]))["content_fingerprint"] != changed
