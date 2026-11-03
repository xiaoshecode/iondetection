"""Conservative classical pseudo-labels for a small VLM adaptation experiment.

Only the train split supplies optimization targets. Teacher agreement is not
independent localization accuracy. No labels are inferred from no_crystal.
"""
import hashlib
import json
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from ..data import list_images, load_image
from ..classical.detection import DEFAULT_DETECTION, detect_ions
from ..classical.region import segment_crystal


TRAINING_PROMPT = (
    'Locate the center of every individually visible fluorescent ion spot. '
    'Return only JSON: {"points": [[x,y], ...], "uncertain": false}. '
    'Coordinates are normalized to 0-1000: 0 is the center of the first image pixel '
    'and 1000 is the center of the last image pixel on each axis. '
    'List points from top to bottom, then left to right. '
    'Do not complete a lattice or infer invisible spots. Return an empty points list if none.'
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def balanced_take(records, count):
    """Round-robin acquisition groups and source frames, not candidate abundance."""
    grouped = defaultdict(lambda: defaultdict(deque))
    for record in records:
        grouped[record['group']][record['raw_path']].append(record)
    frames = {group: deque(sorted(paths)) for group, paths in grouped.items()}
    groups = deque(sorted(grouped))
    result = []
    while groups and len(result) < count:
        group = groups.popleft()
        frame = frames[group].popleft()
        result.append(grouped[group][frame].popleft())
        if grouped[group][frame]:
            frames[group].append(frame)
        if frames[group]:
            groups.append(group)
    return result


def density_balanced_take(records, count):
    """Predefined point-count strata; only teacher labels determine allocation."""
    buckets = defaultdict(list)
    for record in records:
        number = len(record['points'])
        bucket = '1-4' if number <= 4 else ('5-12' if number <= 12 else '13-24')
        buckets[bucket].append(record)
    queues = {key: deque(balanced_take(values, len(values))) for key, values in buckets.items()}
    order = deque(key for key in ('1-4','5-12','13-24') if key in queues)
    selected = []
    while order and len(selected) < count:
        key = order.popleft()
        record = queues[key].popleft()
        selected.append(dict(record, density_bucket=key))
        if queues[key]:
            order.append(key)
    return selected


def verify_split_independence(records_by_split):
    group_owner, hash_owner = {}, {}
    for split, records in records_by_split.items():
        for record in records:
            for value, owners, label in ((record['group'], group_owner, 'acquisition group'),
                                         (record['raw_sha256'], hash_owner, 'source hash')):
                if value in owners and owners[value] != split:
                    raise ValueError('Cross-split %s leakage between %s and %s' % (label, owners[value], split))
                owners[value] = split


def _inside(points, tile):
    x0, y0, x1, y1 = tile
    return points[(points[:, 0] >= x0) & (points[:, 0] < x1) &
                  (points[:, 1] >= y0) & (points[:, 1] < y1)]


def inspect_tile(teacher_points, tile, edge_margin=3., max_points=24, tolerance=1.):
    """Accept or reject the WHOLE tile; never erase a doubtful individual label.

    Reject centers near either side of a crop boundary, including centers just
    outside the image that may still contribute a partially visible PSF.
    """
    x0, y0, x1, y1 = tile
    inside = []
    for points in teacher_points:
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        expanded = _inside(points, (x0-edge_margin, y0-edge_margin, x1+edge_margin, y1+edge_margin))
        if len(expanded):
            near_vertical = np.minimum(np.abs(expanded[:, 0] - x0), np.abs(expanded[:, 0] - (x1-1))) <= edge_margin
            near_horizontal = np.minimum(np.abs(expanded[:, 1] - y0), np.abs(expanded[:, 1] - (y1-1))) <= edge_margin
            if np.any(near_vertical | near_horizontal):
                return None, 'crop_cuts_spot'
        inside.append(_inside(points, tile))
    target = inside[1]
    if len(target) == 0:
        # Avoid using mere detection absence as true background supervision.
        return None, 'empty_pseudo_tile_not_trusted'
    if len(target) > max_points:
        return None, 'too_many_points'
    for other in inside:
        if len(other) != len(target):
            return None, 'threshold_count_disagreement'
        rows, cols = linear_sum_assignment(cdist(target, other))
        if np.any(np.linalg.norm(target[rows] - other[cols], axis=1) > tolerance):
            return None, 'threshold_position_disagreement'
    local = target - [x0, y0]
    local = local[np.lexsort((local[:, 0], local[:, 1]))]
    return local.tolist(), 'accepted'


def response_for_points(points, tile_size=64):
    normalized = [[int(np.rint(float(x) / (tile_size - 1) * 1000)),
                   int(np.rint(float(y) / (tile_size - 1) * 1000))] for x, y in points]
    if any(not (0 <= x <= 1000 and 0 <= y <= 1000) for x, y in normalized):
        raise ValueError('Coordinates lie outside the crop')
    return json.dumps({'points': normalized, 'uncertain': False}, separators=(',', ':'))


def select_frames(root, split, per_group, expected_shape):
    groups = defaultdict(list)
    exclusions = []
    for path in list_images(root):
        # Reading raw-scale images here verifies dimensions before group quotas.
        shape = list(load_image(path).shape)
        if expected_shape and shape != list(expected_shape):
            exclusions.append({'raw_path': str(path.resolve()), 'split': split, 'reason': 'different_camera_shape', 'shape': shape})
            continue
        group = path.parent.relative_to(Path(root)).as_posix()
        groups[group].append(path)
    selected = []
    for group, files in sorted(groups.items()):
        indices = np.linspace(0, len(files)-1, min(per_group, len(files)), dtype=int)
        for index in indices:
            path = files[int(index)].resolve()
            selected.append({'split': split, 'raw_path': str(path), 'raw_sha256': file_hash(path),
                             'group': group, 'selection': 'uniform_within_whole_acquisition_group'})
    return selected, exclusions


def frame_candidates(frame, config, rng, tile_size=64, max_points=24, max_candidates=24,
                     attempts=6000, edge_margin=1.5):
    raw = load_image(frame['raw_path'])
    mask, region = segment_crystal(raw, config['region'])
    audit = dict(frame, region=region, status='rejected', reason=None, crop_rejections={})
    if region['status'] != 'detected':
        audit['reason'] = 'no_crystal_is_not_a_negative_label'
        return [], audit
    if region.get('touches_search_boundary'):
        audit['reason'] = 'crystal_touches_search_boundary'
        return [], audit
    thresholds = [3.5, 4., 4.5]
    teacher_points = []
    for threshold in thresholds:
        detection = dict(config['detection'], threshold_sigma=threshold)
        ions, _ = detect_ions(raw, detection, region_mask=mask, noise_reference=raw)
        teacher_points.append(np.asarray([[p['x'], p['y']] for p in ions], dtype=float).reshape(-1,2))
    audit['teacher_counts'] = [len(p) for p in teacher_points]
    if not len(teacher_points[1]):
        audit['reason'] = 'no_teacher_points'
        return [], audit
    x0, y0, x1, y1 = region['bbox']
    left_min, left_max = max(0, x0-tile_size+8), min(raw.shape[1]-tile_size, x1-8)
    top_min, top_max = max(0, y0-tile_size+8), min(raw.shape[0]-tile_size, y1-8)
    if left_max < left_min or top_max < top_min:
        audit['reason'] = 'invalid_crop_range'
        return [], audit
    lo, hi = map(float, np.percentile(raw, [.5, 99.9]))
    if hi <= lo:
        audit['reason'] = 'constant_display_range'
        return [], audit
    seen, rejections, candidates = set(), Counter(), []
    density_counts = Counter()
    for _ in range(attempts):
        left = int(rng.integers(left_min, left_max+1))
        top = int(rng.integers(top_min, top_max+1))
        tile = [left, top, left+tile_size, top+tile_size]
        if (left, top) in seen:
            continue
        seen.add((left, top))
        points, reason = inspect_tile(teacher_points, tile, edge_margin, max_points)
        if points is None:
            rejections[reason] += 1
            continue
        bucket = '1-4' if len(points) <= 4 else ('5-12' if len(points) <= 12 else '13-24')
        if density_counts[bucket] >= max_candidates // 3:
            rejections['density_bucket_quota'] += 1
            continue
        density_counts[bucket] += 1
        key = hashlib.sha256((frame['raw_sha256'] + ':' + ','.join(map(str,tile))).encode()).hexdigest()[:16]
        candidates.append(dict(frame, id=frame['split']+'_'+key, tile=tile, points=points,
                               response=response_for_points(points, tile_size), prompt=TRAINING_PROMPT,
                               label_source='classical_pseudo', label_is_ground_truth=False,
                               display={'lo': lo, 'hi': hi, 'scope': 'fixed_per_raw_frame'},
                               teacher_thresholds=thresholds, teacher_tolerance_px=1.,
                               label_note='Whole-crop agreement under threshold perturbation, not independent truth.'))
        if len(candidates) >= max_candidates:
            break
    audit.update(status='available' if candidates else 'rejected', reason=None if candidates else 'no_unambiguous_crop',
                 available_crops=len(candidates), density_buckets=dict(density_counts), crop_rejections=dict(rejections))
    return candidates, audit


def check_output(output, roots, reserve_gib):
    output = Path(output).resolve()
    for root in roots:
        root = Path(root).resolve()
        if output == root or output in root.parents or root in output.parents:
            raise ValueError('Output must not overlap raw dataset paths')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output is not empty; choose a new dataset directory')
    parent = output
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < reserve_gib * 1024**3:
        raise ValueError('Insufficient disk space; reserve %.1f GiB' % reserve_gib)
    return output


def contact_sheet(output, records_by_split):
    chosen = []
    for split, records in records_by_split.items():
        for index in np.linspace(0, len(records)-1, min(len(records), 8), dtype=int) if records else []:
            chosen.append(records[int(index)])
    columns, cell_w, cell_h = 6, 200, 224
    canvas = Image.new('RGB', (columns*cell_w, max(1,(len(chosen)+columns-1)//columns)*cell_h), '#111923')
    draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(chosen):
        left, top = (index%columns)*cell_w, (index//columns)*cell_h
        with Image.open(output / record['image']) as source:
            display = source.convert('RGB').resize((192,192), Image.Resampling.NEAREST)
        marks = ImageDraw.Draw(display)
        for x,y in record['points']:
            px,py = (x+.5)*3-.5,(y+.5)*3-.5
            marks.ellipse((px-5,py-5,px+5,py+5),outline='#66ff99',width=1)
        canvas.paste(display,(left,top))
        draw.text((left+2,top+195),'%s: %d pseudo points' % (record['split'],len(record['points'])),fill='white')
        draw.text((left+2,top+209),record['id'][-12:],fill='#bbccdd')
    canvas.save(output/'contact_sheet.png')


def prepare_training_dataset(data_root, output, config, targets=None, seed=20261103,
                             tile_size=64, reserve_gib=8., edge_margin=1.5):
    data_root = Path(data_root).resolve()
    roots = [data_root/split for split in ('train','val','test')]
    if not np.isfinite([reserve_gib,edge_margin]).all() or reserve_gib < 0 or edge_margin < 1.5:
        raise ValueError('Require finite reserve_gib>=0 and edge_margin>=1.5')
    output = check_output(output, [data_root], reserve_gib)
    if tile_size != 64:
        raise ValueError('This initial data contract uses 64x64 raw-pixel crops')
    if config.get('region',{}).get('mode') != 'auto':
        raise ValueError('Pseudo-label training requires the explicit adaptive crystal region')
    targets = targets or {'train':256, 'val':32, 'test':24}
    frames, selection_exclusions = {}, []
    for split, quota in (('train',3),('val',8),('test',8)):
        frames[split], exclusions = select_frames(data_root/split, split, quota, config['region'].get('expected_shape'))
        selection_exclusions.extend(exclusions)
    verify_split_independence(frames)
    rng = np.random.default_rng(seed)
    selected, audits = {}, []
    for split in ('train','val','test'):
        candidates = []
        for index, frame in enumerate(frames[split],1):
            found, audit = frame_candidates(frame, config, rng, tile_size=tile_size, edge_margin=edge_margin)
            candidates.extend(found)
            audits.append(audit)
            print('%s frame %d/%d: %d eligible crops' % (split,index,len(frames[split]),len(found)),flush=True)
        selected[split] = density_balanced_take(candidates, targets[split])
    verify_split_independence(selected)
    output.mkdir(parents=True,exist_ok=True)
    free_before = shutil.disk_usage(output).free
    for split, records in selected.items():
        image_dir = output/'images'/split
        image_dir.mkdir(parents=True,exist_ok=True)
        cached_path, raw = None, None
        for record in records:
            if shutil.disk_usage(output).free < reserve_gib*1024**3:
                raise ValueError('Disk reserve reached; stopped without deleting saved outputs')
            if record['raw_path'] != cached_path:
                cached_path, raw = record['raw_path'], load_image(record['raw_path'])
            x0,y0,x1,y1 = record['tile']
            lo,hi = record['display']['lo'],record['display']['hi']
            gray = np.rint(255*np.clip((raw[y0:y1,x0:x1]-lo)/(hi-lo),0,1)).astype('uint8')
            target = image_dir/(record['id']+'.png')
            Image.fromarray(gray).convert('RGB').save(target)
            record['image'] = target.relative_to(output).as_posix()
            record['image_sha256'] = file_hash(target)
        with (output/(split+'.jsonl')).open('w',encoding='utf-8') as stream:
            for record in records:
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    code_paths = [Path(__file__),Path(__file__).parents[1]/'classical'/'detection.py',Path(__file__).parents[1]/'classical'/'region.py']
    manifest = {'version':1,'splits':{s:s+'.jsonl' for s in selected},'counts':{s:len(v) for s,v in selected.items()},
                'alltruth_is_pseudo':True,'label_source':'classical_pseudo','prompt':TRAINING_PROMPT,
                'teacher_config':config,'teacher_code_sha256':{p.name:file_hash(p) for p in code_paths},
                'tile_size':tile_size,'seed':seed,'max_points':24,'edge_margin_px':edge_margin,
                'density_selection':'Round-robin fixed teacher-count buckets 1-4, 5-12, 13-24; group/frame balanced within each bucket.',
                'density_counts':{s:dict(Counter(r['density_bucket'] for r in v)) for s,v in selected.items()},
                'revision_note':'v1 with 3px boundary guards selected almost only outer 1-2 point crops. v2 uses predeclared 1.5px guard and density strata, before any model training/evaluation.',
                'threshold_sigma_checks':[3.5,4.,4.5],'threshold_match_tolerance_px':1.,
                'frame_selection':{'train_per_group':3,'val_per_group':8,'test_per_group':8},
                'split_groups':{s:sorted({r['group'] for r in v}) for s,v in selected.items()},
                'source_frame_counts':{s:len({r['raw_path'] for r in v}) for s,v in selected.items()},
                'split_group_and_hash_disjoint':True,'selection_exclusions':selection_exclusions,
                'full_frame_evaluation':False,'test_usage':'Held-out optimization split; previously viewed in diagnostics, not a blind publication test. Teacher agreement only.',
                'empty_tile_policy':'No no_crystal images or zero-detection crops used as trusted negatives.',
                'limitations':['Teacher errors and missing weak spots can remain despite threshold stability.',
                              'Same-day adjacent sessions do not establish cross-crystal/day generalization.',
                              'Selecting clear threshold-stable crops biases data toward easier examples.',
                              '1.5px crop guard can retain PSF tails cut by the boundary; review border cases and do not treat center labels as complete PSF support.',
                              'No human independent centers, dark-state labels, or quantum-state fidelity.'],
                'disk':{'reserve_gib':reserve_gib,'free_before_bytes':free_before,'free_after_bytes':shutil.disk_usage(output).free}}
    save_json(output/'frame_audit.json',audits)
    save_json(output/'manifest.json',manifest)
    (output/'generator_snapshot.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    contact_sheet(output,selected)
    if not selected['train'] or not selected['val']:
        raise ValueError('No suitable train/validation crops; audit saved, do not start training')
    return manifest
