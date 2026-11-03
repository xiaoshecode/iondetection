"""Independent repeats and raw-intensity preconditions for restoration."""
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from ionrecog import validation


def make_repeats(tmp_path):
    for split, group, base in (("train", "crystal_a", 10), ("val", "crystal_b", 20)):
        folder = tmp_path / split / group
        folder.mkdir(parents=True)
        for index in range(3):
            np.save(folder / ("frame_%d.npy" % index), np.full((8, 9), base + index, dtype=np.uint16))
    return tmp_path / "train", tmp_path / "val"


def test_diffusion_rejects_copied_frame_even_in_another_format(tmp_path):
    train, val = make_repeats(tmp_path)
    group = train / "crystal_a"
    Image.fromarray(np.load(group / "frame_0.npy")).save(group / "copied.png")
    with pytest.raises(ValueError, match="Duplicate decoded frame in diffusion group"):
        validation.validate_splits(train, val, method="diffusion")


def test_diffusion_decodes_each_image_only_once(tmp_path, monkeypatch):
    train, val = make_repeats(tmp_path)
    original = validation.load_image
    calls = Counter()

    def counted_load(path):
        calls[path] += 1
        return original(path)

    monkeypatch.setattr(validation, "load_image", counted_load)
    report = validation.validate_splits(train, val, method="diffusion")
    assert report["train"]["images"] == report["val"]["images"] == 3
    assert len(calls) == 6
    assert set(calls.values()) == {1}


@pytest.mark.parametrize("method", ["masked", "diffusion"])
def test_training_data_rejects_negative_raw_intensities(tmp_path, method):
    train, val = make_repeats(tmp_path)
    np.save(train / "crystal_a" / "frame_0.npy", np.full((8, 9), -1, dtype=np.float32))
    with pytest.raises(ValueError, match="nonnegative raw camera ADU"):
        validation.validate_splits(train, val, method=method)


@pytest.mark.parametrize("offset", [-1.0, float("nan"), float("inf")])
def test_learning_config_rejects_unsupported_offset(offset):
    from ionrecog.learning import effective_config

    with pytest.raises(ValueError, match="data.offset must be finite and nonnegative"):
        effective_config({"data": {"offset": offset}})


def test_nonfinite_validation_loss_does_not_save_checkpoint(tmp_path, monkeypatch):
    import torch
    from ionrecog import learning

    train, val = make_repeats(tmp_path / "data")
    original = learning._batch_loss

    def validation_nan(model, *args, **kwargs):
        if not model.training:
            return torch.tensor(float("nan"))
        return original(model, *args, **kwargs)

    monkeypatch.setattr(learning, "_batch_loss", validation_nan)
    config = {
        "data": {"patch_size": 8},
        "model": {"base_channels": 4},
        "train": {"epochs": 1, "steps_per_epoch": 1, "batch_size": 1, "val_steps": 1, "num_threads": 1},
    }
    output = tmp_path / "run"
    previous_threads = torch.get_num_threads()
    try:
        with pytest.raises(RuntimeError, match="Non-finite validation loss"):
            learning.train_model("masked", train, val, config, output, device="cpu")
    finally:
        torch.set_num_threads(previous_threads)
    assert not (output / "best.pt").exists()
    assert not (output / "last.pt").exists()
