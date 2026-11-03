"""Pseudo-label crop selection must preserve labels and split independence."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ionrecog.vision_language.training_data import (
    balanced_take, density_balanced_take, check_output, inspect_tile, response_for_points, verify_split_independence,
)


def test_crop_rejected_if_spot_cut_from_either_side_of_boundary():
    for point in ([1.,20.],[-1.,20.],[62.,20.],[65.,20.]):
        points=np.asarray([[20.,20.],point])
        accepted,reason=inspect_tile([points]*3,[0,0,64,64])
        assert accepted is None and reason=='crop_cuts_spot'


def test_threshold_disagreement_rejects_whole_crop_instead_of_deleting_point():
    normal=np.asarray([[20.,20.],[40.,40.]])
    missing=np.asarray([[20.,20.]])
    accepted,reason=inspect_tile([normal,normal,missing],[0,0,64,64])
    assert accepted is None and reason=='threshold_count_disagreement'


def test_stable_crop_preserves_float_centers_and_normalizes_endpoints():
    points=np.asarray([[40.2,40.4],[20.25,20.5]])
    accepted,reason=inspect_tile([points+.1,points,points-.1],[0,0,64,64])
    assert reason=='accepted' and accepted==[[20.25,20.5],[40.2,40.4]]
    assert json.loads(response_for_points([[0,0],[63,63]]))['points']==[[0,0],[1000,1000]]


def test_empty_pseudo_detection_is_not_a_trusted_negative():
    accepted,reason=inspect_tile([np.empty((0,2))]*3,[0,0,64,64])
    assert accepted is None and reason=='empty_pseudo_tile_not_trusted'


def test_cross_split_group_and_source_hash_leakage_rejected():
    a={'group':'a','raw_sha256':'x'}
    with pytest.raises(ValueError,match='group'):
        verify_split_independence({'train':[a],'val':[dict(a,raw_sha256='y')]})
    with pytest.raises(ValueError,match='hash'):
        verify_split_independence({'train':[a],'test':[dict(a,group='b')]})


def test_balancing_does_not_let_one_long_acquisition_dominate():
    records=[{'group':'a','raw_path':'one','n':i} for i in range(10)]
    records += [{'group':'a','raw_path':'two','n':11},{'group':'b','raw_path':'three','n':12}]
    selected=balanced_take(records,3)
    assert [r['raw_path'] for r in selected]==['one','three','two']


def test_density_strata_prevent_sparse_crops_domination():
    records=[{'group':'a','raw_path':'one','points':[[10.,10.]]*n} for n in [1]*20+[8]*4+[16]*4]
    chosen=density_balanced_take(records,9)
    assert [r['density_bucket'] for r in chosen]==['1-4','5-12','13-24']*3


def test_reduced_guard_still_rejects_centers_just_outside_crop():
    normal=np.asarray([[20.,20.],[-1.,30.]])
    accepted,reason=inspect_tile([normal]*3,[0,0,64,64],edge_margin=1.5)
    assert accepted is None and reason=='crop_cuts_spot'


def test_output_protection_and_disk_reserve(tmp_path,monkeypatch):
    raw=tmp_path/'raw'
    raw.mkdir()
    with pytest.raises(ValueError,match='overlap'):
        check_output(raw/'labels',[raw],0)
    monkeypatch.setattr('ionrecog.vision_language.training_data.shutil.disk_usage',lambda _:SimpleNamespace(free=1))
    with pytest.raises(ValueError,match='disk'):
        check_output(tmp_path/'labels',[raw],8)
