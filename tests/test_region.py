"""Crystal support and raw-noise constraints; synthetic geometry is known.

These checks do not establish real-image ion detection accuracy.
"""
import numpy as np
import pytest

from ionrecog.detection import detect_ions


def gaussian_scene(shape, centers, amplitudes=None, sigma=1.4):
    yy, xx = np.indices(shape, dtype=np.float32)
    image = np.full(shape, 500.0, dtype=np.float32)
    amplitudes = [800.0] * len(centers) if amplitudes is None else amplitudes
    for (x, y), amplitude in zip(centers, amplitudes):
        image += amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    return image


def crystal_scene(include_crystal=True):
    centers = [(float(x), float(y)) for y in range(40, 89, 8) for x in range(52, 77, 8)]
    image = gaussian_scene((128, 128), centers if include_crystal else [])
    yy, xx = np.indices(image.shape, dtype=np.float32)
    # Scattering beams are brighter than the crystal but outside its search area.
    image += 1800 * np.exp(-((xx - 12) ** 2) / 32)
    image += 2200 * np.exp(-((xx - 115) ** 2) / 32)
    image += np.random.default_rng(123).normal(0, 8, image.shape).astype(np.float32)
    return image, centers


def region_config():
    return {"mode": "auto", "search_roi": [36, 28, 94, 104], "expected_shape": [128, 128]}


def test_segment_keeps_lattice_edges_and_excludes_brighter_scattering_beams():
    from ionrecog.region import segment_crystal

    image, truth = crystal_scene()
    original = image.copy()
    mask, diagnostics = segment_crystal(image, region_config())

    np.testing.assert_array_equal(image, original)
    assert mask.dtype == np.bool_ and mask.shape == image.shape
    assert diagnostics["status"] == "detected"
    assert all(mask[round(y), round(x)] for x, y in truth), "Support must include the peripheral ions"
    assert not mask[:, :36].any() and not mask[:, 94:].any()
    assert not mask[:28].any() and not mask[104:].any()
    assert 0 < mask.sum() < (94 - 36) * (104 - 28), "A search rectangle is not itself crystal evidence"


def test_no_crystal_does_not_fall_back_to_search_rectangle():
    from ionrecog.region import segment_crystal

    image, _ = crystal_scene(include_crystal=False)
    mask, diagnostics = segment_crystal(image, region_config())
    assert diagnostics["status"] == "no_crystal"
    assert not mask.any()


def test_camera_shape_mismatch_is_rejected_before_using_pixel_coordinates():
    from ionrecog.region import segment_crystal

    with pytest.raises(ValueError):
        segment_crystal(np.full((160, 200), 500, dtype=np.float32), region_config())


def test_disabled_region_preserves_full_frame_mode():
    from ionrecog.region import segment_crystal

    image, _ = crystal_scene()
    mask, diagnostics = segment_crystal(image, {"mode": "off"})
    assert mask.all() and diagnostics["status"] == "off"


def test_detection_ignores_stronger_outside_peak_and_limits_foreground():
    image = gaussian_scene((80, 96), [(48.25, 41.3), (12, 15)], [800, 8000])
    allowed = np.zeros(image.shape, dtype=bool)
    allowed[25:60, 30:70] = True
    ions, foreground = detect_ions(image, {"background_sigma": 12}, region_mask=allowed)
    assert len(ions) == 1
    assert abs(ions[0]["x"] - 48.25) < 0.7 and abs(ions[0]["y"] - 41.3) < 0.7
    assert not np.any(foreground & ~allowed)
    assert all(allowed[round(ion["y"]), round(ion["x"])] for ion in ions)


def test_empty_crystal_mask_suppresses_even_restored_bright_peaks():
    restored = gaussian_scene((64, 64), [(30, 30)])
    ions, foreground = detect_ions(restored, region_mask=np.zeros(restored.shape, dtype=bool))
    assert ions == [] and not foreground.any()


def test_centroid_cannot_move_into_a_gap_in_the_allowed_mask():
    image = gaussian_scene((96, 96), [(44.8, 50), (52.1, 50)], [1000, 900])
    allowed = np.zeros(image.shape, dtype=bool)
    allowed[40:61, 42:48] = True
    allowed[40:61, 51:57] = True
    ions, foreground = detect_ions(
        image, {"min_distance": 10, "radius": 10, "sigma": 0.5, "background_sigma": 15},
        region_mask=allowed,
    )
    assert len(ions) == 1, "The close double peak is intentionally one unresolved candidate"
    assert allowed[round(ions[0]["y"]), round(ions[0]["x"])]
    assert not np.any(foreground & ~allowed)


def test_raw_noise_reference_rejects_weak_restoration_artifact():
    # A denoised image can contain a very small bump in a nearly flat background.
    # Its own tiny residual noise must not make that bump a convincing ion.
    restored = gaussian_scene((128, 128), [(40, 60), (90, 60)], [400, 10])
    raw = restored + np.random.default_rng(9).normal(0, 30, restored.shape).astype(np.float32)
    own_noise, _ = detect_ions(restored)
    anchored, _ = detect_ions(restored, noise_reference=raw)
    assert len(own_noise) == 2
    assert len(anchored) == 1
    assert abs(anchored[0]["x"] - 40) < 0.7 and abs(anchored[0]["y"] - 60) < 0.7


def test_incompatible_region_and_noise_reference_are_rejected():
    image = gaussian_scene((64, 64), [(30, 30)])
    with pytest.raises(ValueError):
        detect_ions(image, region_mask=np.ones((32, 32), dtype=bool))
    with pytest.raises(ValueError):
        detect_ions(image, noise_reference=np.ones((32, 32), dtype=np.float32))


def test_cli_exports_the_actual_region_and_keeps_all_centers_inside(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from PIL import Image
    from ionrecog import cli

    monkeypatch.setattr(cli.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    source = tmp_path / "acquisition"
    source.mkdir()
    image, _ = crystal_scene()
    np.save(source / "frame.npy", image)
    configuration = tmp_path / "region.yaml"
    configuration.write_text(json.dumps({"region": region_config()}), encoding="utf-8")
    output = tmp_path / "results"

    cli.main(["infer", "--method", "classical", "--input", str(source),
              "--output", str(output), "--config", str(configuration)])

    payload = json.loads((output / "images/frame.npy_ions.json").read_text(encoding="utf-8"))
    with Image.open(output / "images/frame.npy_crystal_mask.png") as file:
        mask = np.asarray(file) > 0
    with Image.open(output / "images/frame.npy_foreground.png") as file:
        foreground = np.asarray(file) > 0
    assert payload["region"]["status"] == "detected"
    assert payload["region"]["area"] == int(mask.sum())
    assert payload["ions"]
    assert all(mask[round(ion["y"]), round(ion["x"])] for ion in payload["ions"])
    assert not np.any(foreground & ~mask)
    assert payload["noise_reference"] == "raw"
