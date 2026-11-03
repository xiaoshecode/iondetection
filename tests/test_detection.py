"""Geometric checks, not claims of performance on unlabelled real data."""

from itertools import permutations

import numpy as np
import pytest
from PIL import Image

from ionrecog.detection import detect_ions, save_overlay


def gaussian_scene(shape, centers, sigma=2.0, amplitude=5000.0, background=100.0):
    yy, xx = np.indices(shape, dtype=np.float32)
    image = np.full(shape, background, dtype=np.float32)
    for x, y in centers:
        image += amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    return image


@pytest.mark.parametrize("level", [0, 100, 65535])
def test_constant_image_does_not_invent_ions(level):
    image = np.full((32, 40), level, dtype=np.float32)
    ions, support = detect_ions(image, {})
    assert ions == []
    assert support.shape == image.shape
    assert not np.any(support)


def test_two_nearby_ions_have_distinct_subpixel_centers():
    truth = np.array([[24.25, 30.35], [33.5, 30.1]])
    image = gaussian_scene((64, 64), truth)
    ions, support = detect_ions(
        image,
        {
            "sigma": 0.7,
            "background_sigma": 15,
            "threshold_sigma": 2,
            "min_area": 3,
            "min_distance": 4,
            "radius": 3,
        },
    )
    assert len(ions) == 2, "A connected fluorescence region may contain multiple ions"
    coordinates = np.array([[ion["x"], ion["y"]] for ion in ions])
    distances = np.linalg.norm(coordinates[:, None] - truth[None, :], axis=-1)
    best_error = min(max(distances[i, j] for i, j in enumerate(order)) for order in permutations(range(2)))
    assert best_error < 0.65
    assert support.shape == image.shape
    assert all(ion["status"] == "observed" for ion in ions)
    assert all(ion["flux"] > 0 for ion in ions)


def test_border_ion_has_finite_in_bounds_coordinates():
    image = gaussian_scene((32, 40), [(1.5, 1.75)])
    ions, _ = detect_ions(image, {"background_sigma": 8, "threshold_sigma": 2})
    assert len(ions) == 1
    assert np.isfinite([ions[0]["x"], ions[0]["y"], ions[0]["flux"]]).all()
    assert 0 <= ions[0]["x"] < 40
    assert 0 <= ions[0]["y"] < 32


def test_overlay_does_not_modify_raw_measurement(tmp_path):
    image = gaussian_scene((32, 40), [(20, 16)])
    original = image.copy()
    ions, _ = detect_ions(image, {})
    destination = tmp_path / "overlay.png"
    save_overlay(image, ions, destination)
    np.testing.assert_array_equal(image, original)
    with Image.open(destination) as output:
        assert output.size == (40, 32)
