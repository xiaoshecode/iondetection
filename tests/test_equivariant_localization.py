"""Prepared checks; execution includes tiny CPU training and is explicitly deferred."""
import numpy as np
import pytest
import torch

from ionrecog.equivariant_localization.model import (
    SignalCentroidNet, equivariance_loss, invert_points, transform_image,
)
from ionrecog.equivariant_localization.pipeline import (
    METHOD, EquivariantRefiner, load_crop_splits, load_model, pad_crop, train_model, validate_config,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def gaussian(shape=(11, 11), x=5.2, y=4.7, sigma=1.1):
    yy, xx = np.indices(shape)
    return (40 + 2000 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))).astype(np.float32)


@pytest.mark.parametrize("k", range(4))
@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("dx,dy", [(0, 0), (-2, 1), (1, -2)])
def test_exact_transform_matches_image_and_inverse(k, flip, dx, dy):
    image = np.zeros((17, 17), dtype=np.float32)
    image[6, 4] = 10
    transformed, matrix = transform_image(image, k, flip, dx, dy)
    expected = matrix @ np.array([4., 6., 1.])
    ys, xs = np.nonzero(transformed)
    assert xs.tolist() == [int(expected[0])]
    assert ys.tolist() == [int(expected[1])]
    restored = invert_points(torch.tensor(expected[None, :2], dtype=torch.float32), torch.from_numpy(matrix[None]))
    torch.testing.assert_close(restored, torch.tensor([[4., 6.]]))


def test_constant_patch_center_does_not_satisfy_translations():
    base = np.zeros((17, 17), np.float32)
    _, first = transform_image(base, dx=-2)
    _, second = transform_image(base, dx=2)
    matrices = torch.from_numpy(np.stack((first, second)))
    constant = torch.tensor([[8., 8.], [8., 8.]])
    assert float(equivariance_loss(constant, matrices, 1, 2)) > 1
    correct = torch.tensor([[6., 8.], [10., 8.]])
    assert float(equivariance_loss(correct, matrices, 1, 2)) == 0


def test_signal_supported_weights_follow_translated_object():
    net = SignalCentroidNet(channels=4)
    image = np.zeros((17, 17), np.float32)
    image[7, 5] = 1
    shifted, _ = transform_image(image, dx=3, dy=-1)
    with torch.no_grad():
        predicted = net(torch.from_numpy(np.stack((image, shifted))[:, None]))
    torch.testing.assert_close(predicted, torch.tensor([[5., 7.], [8., 6.]]))


def test_training_rejects_dense_sized_or_constant_crops_and_no_confirmation(tmp_path):
    with pytest.raises(ValueError, match="<=13"):
        pad_crop(np.ones((20, 20)), 17, 2)
    with pytest.raises(ValueError, match="nonconstant"):
        pad_crop(np.ones((9, 9)), 17, 2)
    with pytest.raises(ValueError, match="single-ion"):
        train_model(tmp_path / "train", tmp_path / "val", tmp_path / "out")
    with pytest.raises(ValueError, match="max_shift"):
        validate_config({"max_shift": 0})


def make_splits(tmp_path):
    train = tmp_path / "train"
    val = tmp_path / "val"
    train.mkdir()
    val.mkdir()
    np.save(train / "ion.npy", gaussian(x=5.2))
    np.save(val / "ion.npy", gaussian(x=4.8, y=5.3))
    return train, val


def test_split_pixel_hashes_detect_same_crop_under_new_filename(tmp_path):
    train, val = make_splits(tmp_path)
    np.save(val / "copy.npy", gaussian(x=5.2))
    with pytest.raises(ValueError, match="pixel-duplicate"):
        load_crop_splits(train, val, validate_config())


def test_cpu_training_checkpoint_and_masked_inference(tmp_path):
    train, val = make_splits(tmp_path)
    config = dict(channels=4, epochs=1, steps_per_epoch=2, val_steps=1,
                  batch_size=2, views=2, num_threads=1, noise_std=.001)
    checkpoint = train_model(train, val, tmp_path / "model", config,
                             confirm_single_ion=True, reserve_gib=0)
    model, saved, device = load_model(checkpoint)
    assert device.type == "cpu"
    assert saved["scale"] == pytest.approx(float(np.max(gaussian(x=5.2))))
    assert (tmp_path / "model" / "summary.json").is_file()
    predictor = EquivariantRefiner(checkpoint, dict(threshold_sigma=3., min_area=2, min_distance=3.))
    raw = gaussian((41, 45), x=22.2, y=20.4)
    empty = predictor(np.zeros_like(raw))
    assert empty["ions"] == []
    blocked = predictor(raw, np.zeros_like(raw, dtype=bool))
    assert blocked["ions"] == []
    result = predictor(raw)
    assert len(result["ions"]) == 1
    assert result["diagnostics"]["refined"] == 1
    assert np.hypot(result["ions"][0]["x"] - 22.2, result["ions"][0]["y"] - 20.4) < 0.75
    assert result["ions"][0]["refined"]


def test_checkpoint_identity_and_dimensions(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"method": "masked"}, bad)
    with pytest.raises(ValueError, match="not an equivariant"):
        load_model(bad)
    cfg = validate_config({"channels": 4, "scale": 2000.})
    torch.save(dict(method=METHOD, schema_version=1, config=cfg,
                    state_dict=SignalCentroidNet(channels=8).state_dict()), bad)
    with pytest.raises(ValueError, match="shapes"):
        load_model(bad)


def test_ambiguous_neighbours_retain_classical_positions(tmp_path):
    cfg = validate_config({"channels": 4, "scale": 2000.})
    path = tmp_path / "model.pt"
    torch.save(dict(method=METHOD, schema_version=1, config=cfg,
                    state_dict=SignalCentroidNet(channels=4).state_dict()), path)
    predictor = EquivariantRefiner(path, dict(min_distance=3., min_area=2), isolation_distance=15.)
    raw = gaussian((41, 45), x=18., y=20.) + gaussian((41, 45), x=26., y=20.)
    result = predictor(raw)
    assert len(result["ions"]) == 2
    assert result["diagnostics"]["unrefined_neighbour"] == 2
    for ion in result["ions"]:
        assert not ion["refined"]
        assert ion["x"] == ion["proposal_x"]
        assert ion["y"] == ion["proposal_y"]
