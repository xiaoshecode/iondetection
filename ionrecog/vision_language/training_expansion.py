"""Expand training crops while preserving a frozen evaluation contract byte-for-byte."""
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from ..data import load_image
from .finetune import load_records, validate_split_separation
from .training_data import (check_output, contact_sheet, file_hash, frame_candidates,
                            save_json, select_frames, verify_split_independence)


BUCKETS = ('1-4', '5-12', '13-24')


def bucket(row):
    number = len(row['points'])
    return '1-4' if number <= 4 else ('5-12' if number <= 12 else '13-24')


def tile_iou(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return area / ((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


def select_additions(original, candidates, total, max_same_frame_iou=.9):
    """Balance density and groups, favoring source frames with fewer selected crops."""
    quotas = dict(zip(BUCKETS, [total//3 + (i < total%3) for i in range(3)]))
    existing = Counter(bucket(row) for row in original)
    needed = {key: quotas[key]-existing[key] for key in BUCKETS}
    if any(value < 0 for value in needed.values()):
        raise ValueError('Requested density quota cannot preserve all parent crops')
    frame_counts = Counter(row['raw_sha256'] for row in original)
    group_counts = Counter(row['group'] for row in original)
    frame_tiles = defaultdict(list)
    for row in original:
        frame_tiles[row['raw_sha256']].append(row['tile'])
    seen = {row['id'] for row in original}
    grouped = {key: defaultdict(lambda: defaultdict(deque)) for key in BUCKETS}
    rejections = Counter()
    for row in candidates:
        if row['id'] in seen:
            rejections['duplicate_crop_id'] += 1
            continue
        seen.add(row['id'])
        grouped[bucket(row)][row['group']][row['raw_sha256']].append(row)
    chosen = []
    while any(needed.values()):
        for key in BUCKETS:
            if not needed[key]:
                continue
            groups = grouped[key]
            while groups:
                group = min(groups, key=lambda name: (min(frame_counts[frame] for frame in groups[name]),
                                                       group_counts[name], name))
                frames = groups[group]
                raw_hash = min(frames, key=lambda name: (frame_counts[name], name))
                queue = frames[raw_hash]
                row = queue.popleft()
                if not queue:
                    del frames[raw_hash]
                if not frames:
                    del groups[group]
                if any(tile_iou(row['tile'], old) > max_same_frame_iou for old in frame_tiles[raw_hash]):
                    rejections['same_frame_iou_above_limit'] += 1
                    continue
                row = dict(row, density_bucket=key)
                chosen.append(row)
                frame_tiles[raw_hash].append(row['tile'])
                frame_counts[raw_hash] += 1
                group_counts[group] += 1
                needed[key] -= 1
                break
            else:
                raise ValueError('Not enough unique crops for density bucket ' + key)
    return chosen, dict(rejections)


def overlap_audit(rows):
    frames = defaultdict(list)
    for row in rows:
        frames[row['raw_sha256']].append(row['tile'])
    values = [tile_iou(a, b) for tiles in frames.values() for i, a in enumerate(tiles) for b in tiles[i+1:]]
    return {'same_frame_pairs': len(values), 'intersecting_pairs': sum(v > 0 for v in values),
            'pairs_iou_ge_0_5': sum(v >= .5 for v in values),
            'pairs_iou_gt_0_9': sum(v > .9 for v in values), 'max_iou': max(values, default=0.),
            'warning': 'Overlapping crops and similar frames are not independent experimental replicates.'}


def expand_training_dataset(parent, data_root, output, train_crops=1024, seed=20261103,
                            reserve_gib=8., workers=4, candidates_per_frame=48):
    parent, data_root = Path(parent).resolve(), Path(data_root).resolve()
    if train_crops < 1 or not 1 <= workers <= 8 or not np.isfinite(reserve_gib) or reserve_gib < 0 or candidates_per_frame < 3:
        raise ValueError('Invalid crop target, worker count, or disk reserve')
    output = check_output(output, [parent, data_root], reserve_gib)
    previous = json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
    original = {split: load_records(parent, split) for split in ('train','val','test')}
    validate_split_separation(*(original[s] for s in ('train','val','test')))
    if train_crops <= len(original['train']):
        raise ValueError('Expansion must add training crops')
    if previous['tile_size'] != 64 or previous['edge_margin_px'] != 1.5 or previous['max_points'] != 24:
        raise ValueError('Parent dataset does not match the frozen 64px crop contract')
    config = previous['teacher_config']
    # Detect code drift before creating labels with a different teacher.
    if file_hash(Path(__file__).with_name('training_data.py')) != previous['teacher_code_sha256']['training_data.py']:
        raise ValueError('Crop teacher/generator code differs from parent')
    package = Path(__file__).parents[1]
    for name in ('detection.py', 'region.py'):
        if file_hash(package/'classical'/name) != previous['teacher_code_sha256'][name]:
            raise ValueError('Classical teacher code differs from parent: ' + name)
    frames, exclusions = select_frames(data_root/'train', 'train', 10**9, config['region']['expected_shape'])
    verify_split_independence(dict(original, train=frames))
    unique_frames, seen_raw = [], set()
    for frame in frames:
        if frame['raw_sha256'] in seen_raw:
            exclusions.append(dict(frame, reason='duplicate_raw_hash'))
            continue
        seen_raw.add(frame['raw_sha256'])
        unique_frames.append(dict(frame, selection='all_eligible_train_frames_for_expansion'))

    cache = output.parent/('candidate_pool_%d' % candidates_per_frame)
    if any(cache == source or source in cache.parents or cache in source.parents for source in (parent,data_root)):
        raise ValueError('Candidate cache overlaps source data')
    contract = {'teacher_config':config,'training_data_sha256':file_hash(Path(__file__).with_name('training_data.py')),
                'seed':seed,'candidates_per_frame':candidates_per_frame,'attempts':12000,'edge_margin':1.5}
    if cache.exists() and any(cache.iterdir()):
        if not (cache/'contract.json').is_file() or json.loads((cache/'contract.json').read_text(encoding='utf-8')) != contract:
            raise ValueError('Candidate cache has a different sampling contract')
    else:
        cache.mkdir(parents=True,exist_ok=True)
        save_json(cache/'contract.json',contract)

    def candidates_for_frame(frame):
        if shutil.disk_usage(output.parent if output.parent.exists() else parent).free < reserve_gib*2**30:
            raise ValueError('Disk reserve reached during candidate generation')
        cached = cache/(frame['raw_sha256']+'.json')
        if cached.is_file():
            entry=json.loads(cached.read_text(encoding='utf-8'))
            if entry['frame'] != frame:
                raise ValueError('Cached raw frame identity changed')
            return entry['candidates'],entry['audit']
        raw_seed = int(hashlib.sha256((str(seed)+':'+frame['raw_sha256']).encode()).hexdigest()[:16], 16)
        found,audit=frame_candidates(frame, config, np.random.default_rng(raw_seed), edge_margin=1.5,
                                    max_candidates=candidates_per_frame,attempts=12000)
        temporary=cached.with_suffix('.tmp')
        save_json(temporary,{'frame':frame,'candidates':found,'audit':audit})
        temporary.replace(cached)
        return found,audit

    candidates, audits = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (found, audit) in enumerate(pool.map(candidates_for_frame, unique_frames), 1):
            candidates.extend(found)
            audits.append(audit)
            print('Expansion frame %d/%d: %d eligible crops' % (index, len(unique_frames), len(found)), flush=True)
    additions, rejection_counts = select_additions(original['train'], candidates, train_crops)
    selected = dict(original, train=original['train']+additions)
    verify_split_independence(selected)
    output.mkdir(parents=True, exist_ok=True)
    free_before = shutil.disk_usage(output).free
    # Preserve original PNG bytes and JSONL lines. No re-rendering or re-serialization.
    for split, rows in original.items():
        for row in rows:
            target = output/row['image']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(parent/row['image'], target)
        shutil.copyfile(parent/previous['splits'][split], output/previous['splits'][split])
    raw_cache = {}
    for row in additions:
        if shutil.disk_usage(output).free < reserve_gib*2**30:
            raise ValueError('Disk reserve reached while saving expanded crops')
        source = row['raw_path']
        # One frame cache at a time bounds RAM despite group-balanced output order.
        if source not in raw_cache:
            raw_cache = {source: load_image(source)}
        raw = raw_cache[source]
        x0,y0,x1,y1 = row['tile']
        lo,hi = row['display']['lo'],row['display']['hi']
        gray = np.rint(255*np.clip((raw[y0:y1,x0:x1]-lo)/(hi-lo),0,1)).astype('uint8')
        row['image'] = 'images/train/'+row['id']+'.png'
        target = output/row['image']
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(gray).convert('RGB').save(target)
        row['image_sha256'] = file_hash(target)
    all_hashes = [row['image_sha256'] for rows in selected.values() for row in rows]
    if len(set(all_hashes)) != len(all_hashes):
        raise ValueError('Duplicate rendered crop hash; expanded data must not train')
    path = output/previous['splits']['train']
    if path.read_bytes() and not path.read_bytes().endswith(b'\n'):
        raise ValueError('Parent training JSONL lacks final newline')
    with path.open('ab') as stream:
        for row in additions:
            stream.write((json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))
    frozen = {s: {'parent_jsonl_sha256': file_hash(parent/previous['splits'][s]),
                  'expanded_jsonl_sha256': file_hash(output/previous['splits'][s]),
                  'all_png_bytes_identical': all((parent/r['image']).read_bytes()==(output/r['image']).read_bytes() for r in original[s])}
              for s in ('val','test')}
    if any(v['parent_jsonl_sha256'] != v['expanded_jsonl_sha256'] or not v['all_png_bytes_identical'] for v in frozen.values()):
        raise ValueError('Frozen evaluation split changed')
    manifest = copy.deepcopy(previous)
    manifest.update(counts={s:len(v) for s,v in selected.items()},
        density_counts={s:dict(Counter(bucket(r) for r in v)) for s,v in selected.items()},
        source_frame_counts={s:len({r['raw_sha256'] for r in v}) for s,v in selected.items()},
        split_groups={s:sorted({r['group'] for r in v}) for s,v in selected.items()},
        seed=seed, revision_note='Expanded parent train crops; frozen validation/test and all parent train records retained.',
        frame_selection={'train':'all eligible raw frames considered; prioritize least-used source frames globally, then acquisition groups',
                         'val':'byte-identical parent split','test':'byte-identical parent split'},
        selection_exclusions=exclusions,
        disk={'reserve_gib':reserve_gib,'free_before_bytes':free_before,'free_after_bytes':shutil.disk_usage(output).free})
    old_raw = {r['raw_sha256'] for r in original['train']}
    manifest['expansion'] = {'parent_dataset':str(parent),'parent_manifest_sha256':file_hash(parent/'manifest.json'),
        'parent_train_rows_retained':len(original['train']),'additional_train_rows':len(additions),
        'parent_train_jsonl_is_exact_prefix':path.read_bytes().startswith((parent/previous['splits']['train']).read_bytes()),
        'eligible_train_source_frames_considered':len(unique_frames),
        'previously_unused_train_source_frames_selected':len({r['raw_sha256'] for r in additions}-old_raw),
        'all_crop_ids_unique':len({r['id'] for rows in selected.values() for r in rows}) == sum(len(v) for v in selected.values()),
        'all_crop_image_hashes_unique':True,'frozen_splits':frozen,'selection_rejections':rejection_counts,
        'new_crop_max_iou_with_previous_crops_same_frame':.9,
        'candidate_cache':str(cache),'candidate_contract_sha256':file_hash(cache/'contract.json'),
        'candidates_per_frame_limit':candidates_per_frame,
        'overlap_audit':overlap_audit(selected['train']),
        'generator_code_sha256':file_hash(Path(__file__))}
    manifest['limitations'].append('Expansion preserves the previously inspected test split as a diagnostic comparison, not a new blind test.')
    save_json(output/'frame_audit.json',audits)
    save_json(output/'manifest.json',manifest)
    shutil.copyfile(Path(__file__),output/'expansion_generator_snapshot.py')
    shutil.copyfile(Path(__file__).with_name('training_data.py'),output/'generator_snapshot.py')
    validate_split_separation(*(load_records(output,s) for s in ('train','val','test')))
    contact_sheet(output,selected)
    return manifest
