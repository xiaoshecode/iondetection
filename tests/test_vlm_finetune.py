"""Data provenance and split guards used before any GPU allocation."""
import json

import pytest

from ionrecog.vision_language.finetune import checkpoint_score, evaluation_subset, load_records, sha256, uniform_subset, validate_split_separation


def dataset_fixture(tmp_path):
    image = tmp_path / 'tile.png'
    image.write_bytes(b'image-content-for-hash-check')
    row = {'id': 'train_001', 'split': 'train', 'image': image.name,
           'image_sha256': sha256(image), 'group': 'group_a', 'raw_sha256': 'raw_a',
           'prompt': 'Locate ions', 'response': '{"points":[[500,500]],"uncertain":false}'}
    (tmp_path / 'train.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
    (tmp_path / 'manifest.json').write_text(json.dumps({'splits': {'train': 'train.jsonl'}}), encoding='utf-8')
    return row


def test_changed_image_is_rejected_before_training(tmp_path):
    row = dataset_fixture(tmp_path)
    assert load_records(tmp_path, 'train') == [row]
    (tmp_path / 'tile.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Changed/escaping'):
        load_records(tmp_path, 'train')


def test_split_path_must_remain_within_dataset(tmp_path):
    dataset_fixture(tmp_path)
    (tmp_path / 'manifest.json').write_text(json.dumps({'splits': {'train': '../outside.jsonl'}}), encoding='utf-8')
    with pytest.raises(ValueError, match='escapes'):
        load_records(tmp_path, 'train')


def test_duplicate_record_is_not_silently_oversampled(tmp_path):
    row = dataset_fixture(tmp_path)
    (tmp_path / 'train.jsonl').write_text((json.dumps(row) + '\n') * 2, encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate'):
        load_records(tmp_path, 'train')


@pytest.mark.parametrize('key', ['group', 'raw_sha256', 'image_sha256'])
def test_each_form_of_split_leakage_is_rejected(key):
    rows = [{'group': 'group_' + str(i), 'raw_sha256': 'raw_' + str(i),
             'image_sha256': 'image_' + str(i)} for i in range(3)]
    validate_split_separation(*[[row] for row in rows])
    rows[2][key] = rows[0][key]
    with pytest.raises(ValueError, match='share'):
        validate_split_separation(*[[row] for row in rows])


def test_fixed_subset_spans_records_without_duplicates():
    rows = list(range(24))
    selected = uniform_subset(rows, 8)
    assert selected[0] == 0 and selected[-1] == 23
    assert len(set(selected)) == 8
    assert uniform_subset(rows, 25) == rows


def test_density_subset_avoids_round_robin_index_aliasing():
    rows = [{'id': str(i), 'points': [[0, 0]] * [2, 8, 20][i % 3]} for i in range(32)]
    selected = evaluation_subset(rows, 8)
    assert [sum(len(r['points']) == n for r in selected) for n in (2, 8, 20)] == [3, 3, 2]
    assert len({r['id'] for r in selected}) == 8


def generation_report(f1, failures):
    return {'total_images': 32, 'failure_images': failures,
            'summary': [{'scope': 'valid_only', 'tolerance_px': 3., 'micro_f1': 1.},
                        {'scope': 'strict', 'tolerance_px': 3., 'micro_f1': f1,
                         'evaluated_images': 32, 'excluded_images': 0}]}


def test_checkpoint_selection_uses_strict_localization_before_loss():
    # Lower token loss must not overrule worse actual generation/localization.
    good = checkpoint_score(.8, 'f1_3px', generation_report(.7, 3))
    bad = checkpoint_score(.5, 'f1_3px', generation_report(.6, 2))
    assert good > bad
    assert checkpoint_score(.9, 'f1_3px', generation_report(.7, 2)) > good
    assert checkpoint_score(.7, 'f1_3px', generation_report(.7, 3)) > good
    assert checkpoint_score(.5, 'loss') > checkpoint_score(.8, 'loss')


def test_selection_rejects_excluded_failures_and_nonfinite_loss():
    report = generation_report(.7, 3)
    report['summary'][1]['excluded_images'] = 3
    with pytest.raises(ValueError, match='include all'):
        checkpoint_score(.8, 'f1_3px', report)
    with pytest.raises(ValueError, match='finite'):
        checkpoint_score(float('nan'), 'loss')
    with pytest.raises(ValueError, match='report'):
        checkpoint_score(.8, 'f1_3px')
