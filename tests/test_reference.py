"""Fixed-site photometry validates geometry and reports intensity heuristics."""

import numpy as np
import pytest

from ionrecog.reference import compare_reference, measure_sites


@pytest.mark.parametrize("radius", [0, -1, np.nan, np.inf, None, "invalid"])
def test_invalid_radius_is_rejected_even_with_no_sites(radius):
    with pytest.raises(ValueError, match="positive finite"):
        measure_sites(np.zeros((16, 16)), [], radius)


@pytest.mark.parametrize("site", [{"x": 4}, {"x": "bad", "y": 4}, {"x": None, "y": 4}])
def test_malformed_site_coordinates_are_rejected(site):
    with pytest.raises(ValueError, match="numeric x and y"):
        measure_sites(np.zeros((16, 16)), [site])


@pytest.mark.parametrize("coordinate", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_nonfinite_coordinates_are_rejected(coordinate, axis):
    site = {"x": 4, "y": 4, axis: coordinate}
    with pytest.raises(ValueError, match="coordinates must be finite"):
        measure_sites(np.zeros((16, 16)), [site])


@pytest.mark.parametrize("x,y", [(-0.1, 4), (16, 4), (4, -0.1), (4, 16)])
def test_center_outside_image_is_rejected_even_if_roi_overlaps(x, y):
    with pytest.raises(ValueError, match="outside the current image"):
        measure_sites(np.zeros((16, 16)), [{"x": x, "y": y}], radius=3)


@pytest.mark.parametrize("image", [np.zeros((16, 16, 3)), np.zeros((0, 16)), np.full((16, 16), np.nan)])
@pytest.mark.parametrize("operation", ["measure", "compare"])
def test_invalid_image_is_rejected_before_photometry(image, operation):
    with pytest.raises(ValueError, match="2D grayscale|Non-finite"):
        if operation == "measure":
            measure_sites(image, [{"x": 4, "y": 4}])
        else:
            compare_reference(image, {"shape": [16, 16], "radius": 3, "sites": []})


def test_subpixel_site_with_no_pixel_in_circle_has_clear_error():
    with pytest.raises(ValueError, match="empty pixel ROI"):
        measure_sites(np.zeros((16, 16)), [{"x": 4.5, "y": 4.5}], radius=0.1)


def reference_scene():
    image = np.full((32, 32), 100.0, dtype=np.float32)
    image[16, 16] += 5000
    sites = measure_sites(image, [{"id": 7, "x": 16, "y": 16}], radius=2)
    return image, {"shape": [32, 32], "radius": 2, "sites": sites}


def test_same_frame_reference_has_unit_flux_ratio_and_heuristic_label():
    image, reference = reference_scene()
    row, = compare_reference(image, reference)
    assert row["id"] == 7
    assert row["raw_background"] == pytest.approx(100)
    assert row["raw_flux"] == pytest.approx(5000)
    assert row["flux_ratio"] == pytest.approx(1)
    assert row["status"] == "bright_like"
    assert "not proof of dark quantum state or occupancy" in row["interpretation"]


@pytest.mark.parametrize("fraction,status", [(0.1, "dim_candidate"), (0.3, "uncertain")])
def test_reduced_flux_remains_an_intensity_heuristic(fraction, status):
    image, reference = reference_scene()
    query = 100 + (image - 100) * fraction
    row, = compare_reference(query, reference)
    assert row["flux_ratio"] == pytest.approx(fraction)
    assert row["status"] == status


def test_clipped_reference_site_is_uncertain_even_with_matching_flux():
    image = np.full((16, 16), 100.0, dtype=np.float32)
    image[1, 1] = 5000
    reference = {"shape": [16, 16], "radius": 3,
                 "sites": measure_sites(image, [{"x": 1, "y": 1}], radius=3)}
    row, = compare_reference(image, reference)
    assert row["roi_clipped"]
    assert row["flux_ratio"] == pytest.approx(1)
    assert row["status"] == "uncertain"


@pytest.mark.parametrize("reference_flux", [0, -1])
def test_nonpositive_reference_flux_does_not_assign_state(reference_flux):
    image, reference = reference_scene()
    reference["sites"][0]["raw_flux"] = reference_flux
    row, = compare_reference(image, reference)
    assert row["flux_ratio"] is None
    assert row["status"] == "uncertain"


def test_nonfinite_reference_flux_is_rejected():
    image, reference = reference_scene()
    reference["sites"][0]["raw_flux"] = np.nan
    with pytest.raises(ValueError, match="raw_flux must be finite"):
        compare_reference(image, reference)


@pytest.mark.parametrize("bright_ratio,dim_ratio", [(np.inf, 0.1), (0.5, np.nan), (0.1, 0.5)])
def test_invalid_ratio_thresholds_are_rejected(bright_ratio, dim_ratio):
    image, reference = reference_scene()
    with pytest.raises(ValueError, match="Require finite"):
        compare_reference(image, reference, bright_ratio=bright_ratio, dim_ratio=dim_ratio)
