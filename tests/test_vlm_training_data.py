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


def test_expansion_balances_density_and_prefers_unused_source_frames():
    from ionrecog.vision_language.training_expansion import select_additions
    def record(name,raw,points,left=0):
        return {'id':name,'group':'g','raw_sha256':raw,'points':[[1,1]]*points,'tile':[left,0,left+64,64]}
    parent=[record('a','old',2),record('b','old',8,80),record('c','old',20,160)]
    candidates=[record('d','old',2,240),record('e','old',8,320),record('f','old',20,400)]
    candidates += [record('new'+str(i),'new'+str(i),n) for i,n in enumerate((2,8,20))]
    selected,_=select_additions(parent,candidates,6)
    assert len(selected)==3 and all(r['raw_sha256'].startswith('new') for r in selected)
    assert [len(r['points']) for r in selected]==[2,8,20]


def test_expansion_rejects_duplicate_and_nearly_identical_same_frame_crops():
    from ionrecog.vision_language.training_expansion import select_additions
    parent=[{'id':'old','group':'g','raw_sha256':'frame','points':[[1,1]],'tile':[0,0,64,64]}]
    same=dict(parent[0],id='near',tile=[1,0,65,64])
    candidates=[parent[0],same,
                dict(parent[0],id='medium',raw_sha256='f2',points=[[1,1]]*8),
                dict(parent[0],id='dense',raw_sha256='f3',points=[[1,1]]*20)]
    with pytest.raises(ValueError,match='Not enough'):
        select_additions(parent,candidates,4)


def test_expansion_preserves_parent_bytes_and_frozen_splits(tmp_path,monkeypatch):
    from pathlib import Path
    from PIL import Image
    from ionrecog.vision_language import training_expansion as expansion
    from ionrecog.vision_language.training_data import file_hash
    parent=tmp_path/'parent'; parent.mkdir()
    data=tmp_path/'raw'; data.mkdir()
    manifest={'splits':{s:s+'.jsonl' for s in ('train','val','test')},'tile_size':64,'edge_margin_px':1.5,
              'max_points':24,'teacher_config':{'region':{'expected_shape':[128,128]}},'limitations':[],
              'teacher_code_sha256':{}}
    package=Path(expansion.__file__).parents[1]
    for name,path in [('training_data.py',Path(expansion.__file__).with_name('training_data.py')),
                      ('detection.py',package/'classical/detection.py'),('region.py',package/'classical/region.py')]:
        manifest['teacher_code_sha256'][name]=file_hash(path)
    for i,split in enumerate(('train','val','test')):
        rows=[]
        for j,n in enumerate((2,8,20) if split=='train' else (2,)):
            Image.new('RGB',(64,64),(10*i+j+1,0,0)).save(parent/(split+str(j)+'.png'))
            rows.append({'id':split+str(j),'split':split,'image':split+str(j)+'.png',
                         'image_sha256':file_hash(parent/(split+str(j)+'.png')),'raw_sha256':split+str(j),
                         'raw_path':split+str(j),'group':split,'points':[[20,20]]*n,'tile':[0,0,64,64],
                         'prompt':'same prompt','response':'{"points":[],"uncertain":false}'})
        # Deliberate extra JSON whitespace proves these lines are not re-serialized.
        (parent/(split+'.jsonl')).write_text(''.join(json.dumps(row,indent=None,separators=(', ', ': '))+'\n' for row in rows),encoding='utf-8')
    (parent/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    frames=[{'id':'add'+str(i),'split':'train','group':'train','raw_path':str(i),'raw_sha256':'new'+str(i)} for i in range(3)]
    monkeypatch.setattr(expansion,'select_frames',lambda *args:(frames,[]))
    def candidate(frame,*args,**kwargs):
        i=int(frame['raw_path'])
        return [dict(frame,points=[[20,20]]*(2,8,20)[i],tile=[0,0,64,64],prompt='same prompt',
                     response='{"points":[],"uncertain":false}',display={'lo':0,'hi':65535})], {'id':frame['id']}
    monkeypatch.setattr(expansion,'frame_candidates',candidate)
    monkeypatch.setattr(expansion,'load_image',lambda name:np.arange(128*128,dtype=np.float32).reshape(128,128)+int(name)*4096)
    output=tmp_path/'expanded'
    result=expansion.expand_training_dataset(parent,data,output,train_crops=6,reserve_gib=0,workers=1)
    assert result['counts']=={'train':6,'val':1,'test':1}
    assert (output/'train.jsonl').read_bytes().startswith((parent/'train.jsonl').read_bytes())
    for split in ('val','test'):
        assert (output/(split+'.jsonl')).read_bytes()==(parent/(split+'.jsonl')).read_bytes()
        assert (output/(split+'0.png')).read_bytes()==(parent/(split+'0.png')).read_bytes()
    assert result['expansion']['all_crop_image_hashes_unique']
