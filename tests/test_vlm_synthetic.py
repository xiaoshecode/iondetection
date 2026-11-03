"""The synthetic pilot is generated from independent known coordinates."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ionrecog.data import load_image
from ionrecog.vlm import read_manifest

spec = importlib.util.spec_from_file_location("prepare_vlm_synthetic", Path(__file__).resolve().parents[1] / "scripts/prepare_vlm_synthetic.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_manifest_hashes_independent_truth_and_negative_scene(tmp_path):
    manifest = module.prepare(tmp_path / "synthetic")
    entries = read_manifest(tmp_path / "synthetic/manifest.json", ["synthetic_validation"])
    assert len(entries) == 4
    assert [len(p["generator"]["centers"]) for p in entries] == [9, 2, 4, 0]
    for row in entries:
        raw = load_image(row["path"])
        assert raw.shape == (96, 96)
        truth_path = Path(row["truth_path"])
        assert hashlib.sha256(truth_path.read_bytes()).hexdigest() == row["truth_sha256"]
        assert json.loads(truth_path.read_text()) == row["generator"]["centers"]
        assert row["selection_roi"] is None and row["ground_truth_available"]
    assert manifest["generator_parameters"]["base_seed"] == 20261103
    negative = load_image(entries[-1]["path"])
    assert negative.std() > 0  # Empty truth means no emitters, not a constant-black shortcut.
    assert negative.max() < 220


def test_generation_is_deterministic_and_refuses_overwrite(tmp_path):
    first = module.prepare(tmp_path / "one", seed=19)
    second = module.prepare(tmp_path / "two", seed=19)
    assert [p["sha256"] for p in first["images"]] == [p["sha256"] for p in second["images"]]
    with pytest.raises(ValueError, match="not an empty directory"):
        module.prepare(tmp_path / "one", seed=20)
    after = json.loads((tmp_path / "one/manifest.json").read_text())
    assert after == first


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_invalid_seed_writes_nothing(tmp_path, seed):
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="nonnegative integer"):
        module.prepare(output, seed)
    assert not output.exists()
