"""Physical and reproducibility checks for the small synthetic fixtures."""

import json

import numpy as np
import pytest

from ionrecog.diffusion.data import audit_manifest, load_manifest, read_case
from ionrecog.diffusion.synthetic import make_synthetic_dataset, sample_camera


def test_synthetic_dataset_has_independent_groups_labels_and_distinct_targets(tmp_path):
    manifest = make_synthetic_dataset(tmp_path / "simulation", 4, 2, 2, size=32, seed=9)
    records = load_manifest(manifest)
    report = audit_manifest(manifest)
    assert report["records"] == 8
    assert len({r["group_id"] for r in records}) == 8
    assert report["splits"]["train"]["records"] == 4
    assert any(r["points"] == [] for r in records)
    assert any(len(r["points"]) > 0 for r in records)
    for record in records:
        case = read_case(record)
        assert case["observed"].dtype == np.float32
        assert case["observed"].shape == (32, 32)
        assert not np.array_equal(case["observed"], case["target"])
        assert case["points"] is not None
        assert record["target_kind"] == "synthetic_clean"
    metadata = json.loads((manifest.parent / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["normalization_suggestion"]["scale"] > 0
    assert "no pixel area integration" in metadata["limitations"][-1]
    assert any(ion["motion_amplitude_px"] > 0 for scene in metadata["scenes"] for ion in scene["ions"])


def test_same_seed_reproduces_images_without_global_rng_changes(tmp_path):
    np.random.seed(42)
    next_random = np.random.random(3)
    np.random.seed(42)
    first = make_synthetic_dataset(tmp_path / "first", 1, 1, 1, size=32, seed=33)
    np.testing.assert_array_equal(np.random.random(3), next_random)
    second = make_synthetic_dataset(tmp_path / "second", 1, 1, 1, size=32, seed=33)
    assert first.read_text() == second.read_text()
    for r1, r2 in zip(load_manifest(first), load_manifest(second)):
        np.testing.assert_array_equal(read_case(r1)["observed"], read_case(r2)["observed"])


def test_simulator_refuses_overwrite(tmp_path):
    out = tmp_path / "data"
    make_synthetic_dataset(out, 1, 1, 1, size=24)
    with pytest.raises(FileExistsError):
        make_synthetic_dataset(out, 1, 1, 1, size=24)


def test_em_multiplication_preserves_mean_and_increases_variance():
    # At high gain ideal compound Poisson-Gamma counts have twice the Poisson
    # variance in input-electron units, rather than deterministic gain scaling.
    expected = np.full((256, 256), 20.0)
    direct = sample_camera(expected, np.random.default_rng(1), em_gain=1, read_noise=0, offset=0)
    multiplied = sample_camera(expected, np.random.default_rng(2), em_gain=100, read_noise=0, offset=0)
    assert abs(float(direct.mean()) - 20) < 0.2
    assert abs(float(multiplied.mean()) - 20) < 0.2
    assert 18 < float(direct.var()) < 22
    assert 37 < float(multiplied.var()) < 43
    zero = sample_camera(np.zeros((8, 8)), np.random.default_rng(3), em_gain=100, read_noise=0, offset=10)
    np.testing.assert_array_equal(zero, 10)


@pytest.mark.parametrize("kwargs", [{"size": 8}, {"train_groups": 0}, {"val_groups": 0}, {"em_gain": 0.5}])
def test_invalid_generation_arguments_fail_before_writing(tmp_path, kwargs):
    out = tmp_path / "bad"
    with pytest.raises(ValueError):
        make_synthetic_dataset(out, **kwargs)
    assert not out.exists()
