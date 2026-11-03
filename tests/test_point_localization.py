"""Prepared algorithm tests. Running this file performs short training/inference."""
import json

import numpy as np
import pytest

from ionrecog.point_localization.simulation import SimulationConfig, make_targets, sample_seed, simulate
from ionrecog.point_localization.inference import decode_maps, predict_maps


def test_simulation_is_reproducible_and_split_streams_are_independent():
    config = SimulationConfig(size=32, max_ions=3, empty_fraction=0., motion_max=.2)
    a, points = simulate(config, sample_seed(42, 0, 0))
    b, labels = simulate(config, sample_seed(42, 0, 0))
    val, _ = simulate(config, sample_seed(42, 1, 0))
    assert np.array_equal(a, b) and np.array_equal(points, labels)
    assert not np.array_equal(a, val)
    assert len(points) > 0 and np.all(points >= config.margin)
    heatmap, offsets, centers = make_targets(points, config.size)
    assert centers.sum() == len(points)
    iy, ix = np.nonzero(centers[0])
    reconstructed = np.stack((ix + offsets[0, iy, ix], iy + offsets[1, iy, ix]), 1)
    from scipy.spatial.distance import cdist
    assert cdist(reconstructed, points).min(axis=1).max() < 1e-5
    assert np.all(heatmap[0, iy, ix] == 1)


def test_empty_scene_and_invalid_simulation():
    raw, points = simulate(SimulationConfig(size=32, empty_fraction=1.), 2)
    assert raw.shape == (32, 32) and points.shape == (0, 2)
    assert make_targets(points, 32)[2].sum() == 0
    with pytest.raises(ValueError):
        SimulationConfig(input_scale=0).validate()
    with pytest.raises(ValueError):
        SimulationConfig(size=16, motion_max=10).validate()
    with pytest.raises(ValueError, match="one emitter"):
        make_targets([[10., 10.], [10.1, 10.1]], 32)


def test_decode_preserves_subpixel_offsets_and_region_and_reports_cap():
    probability = np.zeros((21, 25), np.float32)
    offsets = np.zeros((2, 21, 25), np.float32)
    probability[4, 6], probability[17, 20] = .9, .8
    offsets[:, 4, 6] = [.3, -.2]
    rows, capped = decode_maps(probability, offsets, .5, max_ions=1)
    assert capped and len(rows) == 1
    assert rows[0]["x"] == pytest.approx(6.3) and rows[0]["y"] == pytest.approx(3.8)
    mask = np.zeros_like(probability, dtype=bool)
    mask[17, 20] = True
    rows, capped = decode_maps(probability, offsets, .5, region_mask=mask)
    assert not capped and len(rows) == 1 and rows[0]["x"] == 20
    assert decode_maps(np.zeros_like(probability), offsets)[0] == []


def test_tiling_stitches_coordinates_at_core_boundaries_once():
    import torch
    class PixelModel(torch.nn.Module):
        def forward(self, image):
            return (image - .5) * 20, torch.zeros((image.shape[0], 2) + image.shape[-2:]) + .2
    raw = np.zeros((59, 71), np.float32)
    expected = [(15, 15), (32, 16), (46, 48), (68, 56)]
    for x, y in expected:
        raw[y, x] = 1.
    p, offset = predict_maps(PixelModel(), raw, 0., 1., torch.device("cpu"), tile_size=48, halo=16)
    rows, _ = decode_maps(p, offset, .5)
    assert len(rows) == len(expected)
    assert sorted((round(r["x"], 2), round(r["y"], 2)) for r in rows) == sorted((x + .2, y + .2) for x, y in expected)


def test_odd_image_tiling_matches_full_network_interior():
    import torch
    from ionrecog.point_localization.models import PointNet
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = PointNet(4).eval()
    raw = np.random.default_rng(3).uniform(0, 1, (79, 91)).astype(np.float32)
    tiled_p, tiled_o = predict_maps(model, raw, 0., 1., torch.device("cpu"), tile_size=48, halo=16)
    full_p, full_o = predict_maps(model, raw, 0., 1., torch.device("cpu"), tile_size=256, halo=32)
    assert tiled_p.shape == raw.shape and tiled_o.shape == (2,) + raw.shape
    assert np.allclose(tiled_p, full_p, atol=1e-6)
    assert np.allclose(tiled_o, full_o, atol=1e-6)
    with pytest.raises(ValueError, match="halo"):
        predict_maps(model, raw, 0., 1., torch.device("cpu"), tile_size=32, halo=8)


def test_network_shapes_loss_and_fixed_scale_input():
    import torch
    from ionrecog.point_localization.models import PointNet, point_loss
    torch.set_num_threads(1)
    model = PointNet(4)
    logits, offsets = model(torch.ones(2, 1, 25, 31))
    assert logits.shape == (2, 1, 25, 31) and offsets.shape == (2, 2, 25, 31)
    assert offsets.abs().max() <= .5
    target = torch.zeros_like(logits)
    center = torch.zeros_like(logits)
    target[:, :, 10, 12] = center[:, :, 10, 12] = 1
    loss, _, _ = point_loss(logits, offsets, target, torch.zeros_like(offsets), center)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.heatmap.weight.grad is not None


def test_short_train_checkpoint_infer_and_resume(tmp_path):
    from ionrecog.point_localization.training import train_model
    from ionrecog.point_localization.inference import PointPredictor
    config = SimulationConfig(size=32, max_ions=2, motion_max=.2, empty_fraction=0.)
    output = tmp_path / "run"
    report = train_model(output, config, epochs=1, steps=1, batch_size=1, val_samples=2,
                         channels=4, num_threads=1, reserve_gib=0.)
    assert report["experimental_accuracy_verified"] is False
    assert report["best_epoch"] == 1
    saved = json.loads((output / "configuration.json").read_text())
    assert saved["simulation"]["input_scale"] == config.input_scale
    predictor = PointPredictor(output / "best.pt", num_threads=1)
    blank = np.full((40, 41), 200, dtype=np.float32)
    assert predictor(blank)["ions"] == []
    raw, _ = simulate(config, 3)
    prediction = predictor(raw)
    assert prediction["diagnostics"]["training_domain"] == "synthetic_not_experimental_ground_truth"
    with pytest.raises(ValueError, match="new or empty"):
        train_model(output, config, epochs=1, steps=1, batch_size=1, val_samples=2, channels=4, reserve_gib=0.)
    continued = train_model(tmp_path / "continued", config, epochs=2, steps=1, batch_size=1,
                            val_samples=2, channels=4, num_threads=1, reserve_gib=0., resume=output / "last.pt")
    assert continued["completed_epochs"] == 2


def test_invalid_checkpoint_and_parser_help(tmp_path):
    import torch
    from ionrecog.point_localization.inference import PointPredictor
    from ionrecog.point_localization.cli import main
    path = tmp_path / "bad.pt"
    torch.save({"method": "masked"}, path)
    with pytest.raises(ValueError, match="checkpoint"):
        PointPredictor(path, num_threads=1)
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
