"""Full-original coordinate, coverage and failure semantics (CPU only)."""
import json

import numpy as np
import pytest

from ionrecog.vision_language.fullframe import (
    aggregate, decode_response, frame_geometry, padded_roi,
)
from ionrecog.vision_language.geometry import make_tiles, render_tile


def response(points=None, uncertain=False, finish='stop'):
    return {'text': json.dumps({'points': points or [], 'uncertain': uncertain}),
            'finish_reason': finish, 'seconds': 1.25}


def entry(roi, shape=(512, 512), status='detected'):
    return {'id': 'test_frame', 'roi': roi, 'shape': list(shape),
            'tiles': make_tiles(shape, 64, 16, roi), 'region': {'status': status}}


@pytest.mark.parametrize('bbox,shape', [
    ([0, 0, 3, 4], (512, 512)),
    ([508, 509, 512, 512], (512, 512)),
    ([10, 15, 30, 35], (64, 64)),
    ([185, 280, 275, 410], (512, 512)),
    ([0, 0, 800, 300], (300, 800)),
])
def test_padded_roi_contains_source_region_and_tiles_cover_every_selected_pixel(bbox, shape):
    roi = padded_roi(bbox, shape)
    x0, y0, x1, y1 = roi
    assert 0 <= x0 <= bbox[0] < bbox[2] <= x1 <= shape[1]
    assert 0 <= y0 <= bbox[1] < bbox[3] <= y1 <= shape[0]
    assert x1 - x0 >= 64 and y1 - y0 >= 64
    coverage = np.zeros(shape, dtype=np.uint8)
    for tile in make_tiles(shape, 64, 16, roi):
        assert tile['x1'] - tile['x0'] == tile['y1'] - tile['y0'] == 64
        coverage[tile['y0']:tile['y1'], tile['x0']:tile['x1']] += 1
    assert np.all(coverage[y0:y1, x0:x1] > 0)
    coverage[y0:y1, x0:x1] = 0
    assert not coverage.any()


def test_small_original_is_rejected_without_resizing_physical_pixel_context():
    with pytest.raises(ValueError, match='smaller'):
        padded_roi([0, 0, 30, 40], (40, 30))


def test_no_crystal_is_inferred_in_search_box_not_labeled_empty():
    raw = np.ones((512, 512), dtype=np.float32) * 100
    roi, region, mask, limits, note = frame_geometry(raw, 'diagnostic', {
        'mode': 'auto', 'expected_shape': [512, 512], 'search_roi': [180, 220, 300, 430]})
    assert region['status'] == 'no_crystal'
    assert roi == [180, 220, 300, 430]
    assert make_tiles(raw.shape, 64, 16, roi)
    assert not mask.any()
    assert limits == [100., 101.]
    assert 'not_confirmed_empty' in note and 'diagnostic' in note
    frame = entry(roi, status='no_crystal')
    records = [decode_response(response(), tile) for tile in frame['tiles']]
    result = aggregate(frame, records, [], 2.)
    assert result['coverage_complete']  # The declared rectangle was attempted successfully.
    assert not result['full_frame_covered']
    assert not result['classical_reference_available']
    assert result['agreement_3px']['f1'] is None
    assert not result['agreement_3px']['all_correct']


def test_other_camera_does_not_reuse_known_camera_search_box():
    raw = np.arange(300 * 800, dtype=np.float32).reshape(300, 800)
    roi, region, mask, limits, note = frame_geometry(raw, 'test', {
        'mode': 'auto', 'expected_shape': [512, 512], 'search_roi': [180, 220, 300, 430]})
    assert roi == [0, 0, 800, 300]
    assert region['status'] == 'different_camera_full_frame'
    assert mask.all()
    np.testing.assert_allclose(limits, np.percentile(raw, [.5, 99.9]))
    assert 'out_of_training_distribution' in note


def test_display_limits_are_fixed_over_full_frame_not_recomputed_per_crop():
    raw = np.arange(512 * 512, dtype=np.float32).reshape(512, 512)
    before = raw.copy()
    _, _, _, limits, _ = frame_geometry(raw, 'test', {
        'mode': 'auto', 'expected_shape': [100, 100]})
    tile = {'id': 0, 'x0': 180, 'y0': 220, 'x1': 244, 'y1': 284}
    rendered = np.asarray(render_tile(raw, tile, *limits, display_scale=1))
    expected = np.rint(255 * np.clip((raw[220:284, 180:244] - limits[0]) /
                                   (limits[1] - limits[0]), 0, 1)).astype(np.uint8)
    assert rendered.shape == (64, 64, 3)
    for channel in range(3):
        np.testing.assert_array_equal(rendered[:, :, channel], expected)
    np.testing.assert_array_equal(raw, before)


def test_normalized_tile_endpoints_map_to_original_global_centers():
    tile = {'id': 7, 'x0': 180, 'y0': 220, 'x1': 244, 'y1': 284}
    record = decode_response(response([[0, 0], [1000, 1000], [500, 500]]), tile)
    assert record['status'] == 'complete'
    assert [(p['x'], p['y']) for p in record['points']] == [
        (180., 220.), (243., 283.), (211.5, 251.5)]


def test_overlap_merges_repeat_observations_without_snapping_to_classical_teacher():
    frame = entry([100, 200, 212, 264])
    a, b = frame['tiles']
    # Same ion is near the right edge of a but better inside b. A second ion
    # four pixels away must survive, even with an unrelated classical result.
    first = decode_response(response([[(158 - a['x0']) * 1000 / 63, 25 * 1000 / 63]]), a)
    second = decode_response(response([[(158.25 - b['x0']) * 1000 / 63, 25 * 1000 / 63],
                                      [(162.25 - b['x0']) * 1000 / 63, 25 * 1000 / 63]]), b)
    result = aggregate(frame, [first, second], [{'x': 120., 'y': 210.}], 2.)
    assert result['coverage_complete'] and result['vlm_count'] == 2
    assert not result['full_frame_covered']
    np.testing.assert_allclose([(p['x'], p['y']) for p in result['points']],
                               [(158.25, 225.), (162.25, 225.)])
    merged = result['points'][0]
    assert merged['proposal_count'] == 2
    assert set(merged['source_tile_ids']) == {0, 1}
    assert result['classical_points'] == [{'x': 120., 'y': 210.}]
    assert result['agreement_3px']['tp'] == 0


@pytest.mark.parametrize('bad_response,status', [
    (response([[500, 500]], finish='length'), 'failed'),
    ({'text': '{"points": [[1,2]', 'finish_reason': 'stop', 'seconds': .5}, 'failed'),
    (response(uncertain=True), 'abstained'),
    (response([[500, 500]], uncertain=True), 'abstained'),
    ({'text': '{"points": [{"x":500,"y":500,"uncertain":true}]}',
      'finish_reason': 'stop', 'seconds': .5}, 'abstained'),
])
def test_failed_or_abstained_tiles_cannot_become_successful_empty_frames(bad_response, status):
    frame = entry([100, 200, 164, 264])
    record = decode_response(bad_response, frame['tiles'][0])
    assert record['status'] == status
    result = aggregate(frame, [record], [], 2.)
    assert result['status'] == 'incomplete'
    assert not result['coverage_complete']
    assert result['vlm_count'] == 0
    assert not result['agreement_3px']['all_correct']
    assert result['failed_tiles'] + result['abstained_tiles'] == 1


@pytest.mark.parametrize('text', [
    '```json\n{"points": [], "uncertain": true}\n```',
    '```json {"points": [], "uncertain": true}```',
])
def test_empty_fenced_abstention_is_retained_without_crashing_batch(text):
    record = decode_response({'text': text, 'finish_reason': 'stop'},
                             {'id': 0, 'x0': 0, 'y0': 0, 'x1': 64, 'y1': 64})
    assert record['status'] == 'abstained'


def test_missing_tile_still_incomplete_even_when_remaining_ions_match():
    frame = entry([100, 200, 212, 264])
    record = decode_response(response([[500, 500]]), frame['tiles'][0])
    teacher = [{'x': 131.5, 'y': 231.5}]
    result = aggregate(frame, [record], teacher, 2.)
    assert result['agreement_3px']['f1'] == 1.
    assert not result['agreement_3px']['all_correct']
    assert not result['coverage_complete']
    assert result['tiles_processed'] == 1 and result['tiles_planned'] == 2


def test_full_frame_coverage_requires_every_planned_tile_to_succeed():
    frame = entry([0, 0, 112, 64], shape=(64, 112), status='different_camera_full_frame')
    records = [decode_response(response(), tile) for tile in frame['tiles']]
    complete = aggregate(frame, records, [], 2.)
    assert complete['coverage_complete'] and complete['full_frame_covered']
    records[-1] = decode_response(response(finish='length'), frame['tiles'][-1])
    incomplete = aggregate(frame, records, [], 2.)
    assert not incomplete['coverage_complete']
    assert not incomplete['full_frame_covered']
