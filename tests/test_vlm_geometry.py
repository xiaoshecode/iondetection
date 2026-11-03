"""Coordinate and raw-signal safeguards, not claims of VLM accuracy."""
import json

import numpy as np
import pytest

from ionrecog.vlm_geometry import deduplicate, make_tiles, parse_points, refine_points, render_tile


def scene(centers, shape=(64, 64), amplitudes=None, sigma=1.2):
    yy, xx = np.indices(shape)
    raw = np.full(shape, 100., dtype=np.float32)
    for (x, y), amplitude in zip(centers, amplitudes or [1000.] * len(centers)):
        raw += amplitude * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    return raw


def test_tiles_cover_roi_and_shift_last_window_without_tiny_tail():
    tiles = make_tiles((155, 181), 64, 16, [13, 7, 178, 149])
    coverage = np.zeros((155, 181), dtype=int)
    for tile in tiles:
        assert tile["x1"] - tile["x0"] == tile["y1"] - tile["y0"] == 64
        coverage[tile["y0"]:tile["y1"], tile["x0"]:tile["x1"]] += 1
    assert np.all(coverage[7:149, 13:178] >= 1)
    assert np.all(coverage[:, :13] == 0)
    assert np.all(coverage[:7] == 0)
    assert tiles[-1]["x1"] == 178 and tiles[-1]["y1"] == 149
    assert make_tiles((17, 23), 128, 32) == [{"id": 0, "x0": 0, "y0": 0, "x1": 23, "y1": 17}]


@pytest.mark.parametrize("kwargs", [{"overlap": 128}, {"overlap": -1}, {"tile_size": True},
                                     {"roi": [0, 0, 200, 20]}, {"roi": [3, 4, 3, 10]},
                                     {"roi": [0, 0, 8.5, 8]}])
def test_invalid_tiling_rejected(kwargs):
    with pytest.raises(ValueError):
        make_tiles((32, 32), **kwargs)


def test_display_uses_one_mapping_and_never_changes_raw():
    raw = np.array([[0., 50., 100., 50.], [25., 75., 25., 75.]])
    copy = raw.copy()
    left, right = make_tiles(raw.shape, 2, 0)
    a = np.array(render_tile(raw, left, 0, 100, 3))
    b = np.array(render_tile(raw, right, 0, 100, 3))
    np.testing.assert_array_equal(a[:3, 3:6], b[:3, 3:6])
    np.testing.assert_array_equal(raw, copy)
    assert a.shape == (6, 6, 3)
    assert np.all(a[:3, :3] == 0)


def test_normalized_coordinates_map_first_and_last_centers_and_not_whole_image():
    tile = {"id": 4, "x0": 20, "y0": 30, "x1": 121, "y1": 81}
    points = parse_points('```json\n{"points":[[0,0],[1000,1000],[500,500]],"uncertain":false}\n```', tile)
    assert [[p["x"], p["y"]] for p in points] == [[20, 30], [120, 80], [70, 55]]
    assert all(p["tile_id"] == 4 for p in points)
    assert parse_points('{"points":[]}', tile) == []


def test_pixel_center_mapping_and_audited_outer_subpixels():
    tile = {"id": 1, "x0": 10, "y0": 20, "x1": 20, "y1": 30}
    points = parse_points('[[1,1],[4,7],[0,29]]', tile, "pixels", 3)
    assert (points[0]["x"], points[0]["y"]) == (10, 20)
    assert (points[1]["x"], points[1]["y"]) == (11, 22)
    assert (points[2]["x"], points[2]["y"]) == (10, 29)
    assert points[2]["coordinate_edge_clamped"] and points[2]["status"] == "uncertain"


@pytest.mark.parametrize("text", ['{"points":[[1,2]]', 'Here are 2 points: [[1,2]]',
    '{"points":[[true,2]]}', '{"points":[["1",2]]}', '{"points":[[NaN,2]]}',
    '{"points":[[1e999,2]]}', '{"points":[[-1,0]]}', '{"points":[[1001,2]]}',
    '{"points":[[1,2]],"uncertain":"false"}', '{"points":[],"points":[[1,2]]}',
    '{"points":[{"x":1}]}', '{"points":[[1,2,3]]}', 'null'])
def test_malformed_response_rejects_entire_tile(text):
    with pytest.raises(ValueError):
        parse_points(text, make_tiles((32, 32))[0])


def test_uncertainty_is_preserved_at_envelope_and_point_levels():
    tile = make_tiles((32, 32))[0]
    assert parse_points('{"points":[[100,100]],"uncertain":true}', tile)[0]["status"] == "uncertain"
    p = parse_points('[{"x":100,"y":100,"uncertain":true}]', tile)[0]
    assert p["model_uncertain"]


def test_overlap_merge_retains_every_proposal_and_prefers_interior_view():
    proposals = [{"x": 10., "y": 12., "tile_id": 1, "tile_margin": 1},
                 {"x": 10.4, "y": 12.1, "tile_id": 2, "tile_margin": 8},
                 {"x": 13., "y": 12., "tile_id": 1, "tile_margin": 4}]
    result = deduplicate(proposals, 2)
    assert len(result) == 2
    merged = next(p for p in result if p["proposal_count"] == 2)
    assert merged["x"] == 10.4
    assert set(merged["source_tile_ids"]) == {1, 2}
    assert sum(len(p["merged_proposals"]) for p in result) == len(proposals)
    assert "merged_proposals" not in proposals[0]


def test_flat_background_does_not_validate_hallucinated_grid():
    raw = np.full((64, 64), 100.)
    proposals = [{"x": x, "y": 30.} for x in (15., 25., 35., 45.)]
    accepted, uncertain = refine_points(raw, proposals)
    assert accepted == [] and len(uncertain) == 4
    assert all("insufficient_raw_signal" in p["reasons"] for p in uncertain)
    assert all(p["shift"] == 0 and p["raw_flux"] == 0 for p in uncertain)


def test_refinement_improves_isolated_candidate_and_measures_raw():
    raw = scene([(30.3, 29.8)])
    accepted, uncertain = refine_points(raw, [{"x": 29.8, "y": 30.2}], radius=4)
    assert uncertain == [] and len(accepted) == 1
    point = accepted[0]
    assert np.hypot(point["x"] - 30.3, point["y"] - 29.8) < .08
    assert point["status"] == "supported_candidate" and point["raw_snr"] > 5
    assert (point["proposal_x"], point["proposal_y"]) == (29.8, 30.2)


def test_rounded_normalized_integer_center_keeps_circular_support_symmetric():
    raw = scene([(16., 16.)], shape=(32, 32))
    tile = make_tiles(raw.shape)[0]
    proposals = parse_points('{"points":[[516.129032258,516.129032258]]}', tile)
    assert proposals[0]["x"] < 16.  # Decimal JSON rounding, not a real displacement.
    accepted, uncertain = refine_points(raw, proposals, radius=3)
    assert not uncertain
    assert accepted[0]["x"] == pytest.approx(16., abs=1e-10)
    assert accepted[0]["y"] == pytest.approx(16., abs=1e-10)
    assert accepted[0]["support_pixels"] == 29


def test_large_shift_is_not_silently_clipped_or_accepted():
    raw = scene([(33., 30.)], sigma=.7)
    accepted, uncertain = refine_points(raw, [{"x": 30., "y": 30.}], radius=4, max_shift=1)
    assert not accepted and uncertain[0]["shift"] > 1
    assert "excessive_shift" in uncertain[0]["reasons"]
    assert uncertain[0]["x"] == 30. and uncertain[0]["refined_x"] > 32


def test_neighbor_ownership_keeps_close_unequal_brightness_ions_distinct():
    raw = scene([(26., 30.), (32., 30.)], amplitudes=[1000., 6000.])
    proposals = [{"x": 25.7, "y": 30.2}, {"x": 32.2, "y": 29.8}]
    accepted, uncertain = refine_points(raw, proposals, radius=4, max_shift=2)
    assert len(accepted) == 2 and not uncertain
    assert abs(accepted[0]["x"] - 26.) < .4
    assert abs(accepted[1]["x"] - 32.) < .4
    assert accepted[1]["x"] - accepted[0]["x"] > 5


def test_edge_and_vlm_uncertainty_are_not_erased_by_bright_signal():
    raw = scene([(1., 1.), (30., 30.)])
    accepted, uncertain = refine_points(raw, [{"x": 1., "y": 1.},
                                             {"x": 30., "y": 30., "status": "uncertain"}])
    assert not accepted and len(uncertain) == 2
    assert "edge_clipped" in uncertain[0]["reasons"]
    assert "model_or_coordinate_uncertain" in uncertain[1]["reasons"]
    json.dumps(uncertain, allow_nan=False)


@pytest.mark.parametrize("kwargs", [{"radius": .1}, {"radius": True}, {"max_shift": -1},
                                     {"min_snr": 0}, {"min_snr": np.nan}])
def test_refinement_parameters_checked_even_for_empty_list(kwargs):
    with pytest.raises(ValueError):
        refine_points(np.zeros((16, 16)), [], **kwargs)
