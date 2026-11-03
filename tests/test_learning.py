"""Check supervision separation and end-to-end execution on tiny CPU examples."""

from pathlib import Path

import numpy as np
import pytest
import torch

from ionrecog.learning import (
    choose_repeat_paths,
    make_repeat_pair,
    mask_pixels,
    restore_image,
    train_model,
)


@pytest.fixture(autouse=True)
def one_torch_thread():
    # Tiny smoke tests run faster and more consistently without thread oversubscription.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config():
    return {
        "data": {"offset": 0.0, "scale": 65535.0, "patch_size": 16, "max_target_frames": 2},
        "model": {"base_channels": 4},
        "train": {
            "epochs": 1,
            "steps_per_epoch": 1,
            "batch_size": 2,
            "lr": 0.001,
            "seed": 17,
            "mask_fraction": 0.1,
            "val_steps": 1,
        },
        "diffusion": {"timesteps": 8, "sampling_steps": 3},
    }


def make_training_roots(root):
    rng = np.random.default_rng(8)
    yy, xx = np.indices((24, 32))
    for split, center in [("train", (10, 11)), ("val", (21, 12))]:
        group = root / split / (split + "_crystal")
        group.mkdir(parents=True)
        signal = 1200 + 22000 * np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / 8)
        for frame in range(4):
            noisy = np.clip(signal + rng.normal(0, 150, signal.shape), 0, 65535).astype(np.uint16)
            np.save(group / ("frame_%02d.npy" % frame), noisy)
    return root / "train", root / "val"


def test_masking_never_reintroduces_hidden_target_values():
    image = torch.arange(1, 257, dtype=torch.float32).reshape(1, 1, 16, 16)
    corrupted, mask = mask_pixels(image, fraction=0.15, generator=torch.Generator().manual_seed(5))
    assert mask.dtype == torch.bool
    assert mask.shape == image.shape
    assert 0 < mask.sum() < image.numel()
    torch.testing.assert_close(corrupted[~mask], image[~mask])
    hidden_values = set(image[mask].tolist())
    assert not hidden_values.intersection(corrupted[mask].tolist())
    torch.testing.assert_close(image, torch.arange(1, 257, dtype=torch.float32).reshape_as(image))


def test_repeat_targets_are_independent_of_selected_input(tmp_path):
    paths = []
    for index, value in enumerate([1000, 2000, 4000, 8000]):
        path = tmp_path / ("frame_%d.npy" % index)
        np.save(path, np.full((16, 16), value, dtype=np.uint16))
        paths.append(path)
    input_path, targets = choose_repeat_paths(paths, np.random.default_rng(17), max_target_frames=2)
    assert Path(input_path) not in {Path(p) for p in targets}
    assert len(targets) == len(set(targets)) == 2

    observed, target_mean, metadata = make_repeat_pair(paths, np.random.default_rng(17), max_target_frames=2)
    selected_input = Path(metadata["input_path"])
    selected_targets = [Path(p) for p in metadata["target_paths"]]
    assert selected_input not in selected_targets
    expected = np.mean([np.load(p).astype(np.float32) for p in selected_targets], axis=0)
    np.testing.assert_array_equal(observed, np.load(selected_input))
    np.testing.assert_allclose(target_mean, expected)


def test_diffusion_pair_requires_repeated_frames(tmp_path):
    path = tmp_path / "only_frame.npy"
    np.save(path, np.zeros((16, 16), dtype=np.uint16))
    with pytest.raises(ValueError):
        choose_repeat_paths([path], np.random.default_rng(0))


@pytest.mark.parametrize("method", ["masked", "diffusion"])
def test_tiny_training_checkpoint_and_odd_shape_inference(tmp_path, method):
    train_root, val_root = make_training_roots(tmp_path / "data")
    checkpoint = train_model(
        method, train_root, val_root, tiny_config(), tmp_path / "run", device="cpu"
    )
    assert Path(checkpoint).is_file()
    stored = torch.load(checkpoint, map_location="cpu")
    assert stored["method"] == method
    assert stored["model_state"]
    assert stored["history"]
    rng = np.random.default_rng(2)
    raw = rng.uniform(0, 30000, size=(19, 23)).astype(np.float32)
    first = restore_image(raw, checkpoint, device="cpu", seed=19, sample_steps=3)
    second = restore_image(raw, checkpoint, device="cpu", seed=19, sample_steps=3)
    assert first.shape == raw.shape
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-5)


def test_training_rejects_identical_train_and_validation_root(tmp_path):
    train_root, _ = make_training_roots(tmp_path / "data")
    with pytest.raises(ValueError):
        train_model("masked", train_root, train_root, tiny_config(), tmp_path / "run", device="cpu")
