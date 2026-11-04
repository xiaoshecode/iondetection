"""Resumable best-adapter inference with original-image coordinates and previews.

The known camera uses the existing raw-signal crystal envelope to select a
rectangle. Other camera geometries are tiled in full. No classical coordinates
are supplied to, substituted for, or used to filter VLM predictions.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from ..classical.detection import detect_ions
from ..classical.region import segment_crystal
from ..data import list_images, load_image
from ..evaluation import match_points
from .finetune import (check_space, efficient_attention_setup, encode_record,
                       load_model, restore_adapter, sha256, to_device, write_json)
from .geometry import deduplicate, make_tiles, parse_points, render_tile
from .training_data import TRAINING_PROMPT


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def padded_roi(bbox, shape, margin=8, size=64):
    """Preserve at least 64 raw pixels per axis without leaving the source."""
    h, w = shape
    if min(h, w) < size:
        raise ValueError('Original image is smaller than the trained 64-pixel context')
    bounds = []
    for lo, hi, length in ((bbox[0], bbox[2], w), (bbox[1], bbox[3], h)):
        left, right = max(0, lo - margin), min(length, hi + margin)
        width = max(size, right - left)
        left = max(0, min(length - width, (left + right - width) // 2))
        bounds.append((left, left + width))
    return [bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]]


def frame_geometry(raw, split, region_config):
    """A missing envelope triggers explicit search-box inference, never zero ions."""
    expected = region_config.get('expected_shape')
    known = expected is None or list(raw.shape) == expected
    h, w = raw.shape
    if known:
        mask, region = segment_crystal(raw, region_config)
        if region['status'] == 'detected':
            roi = padded_roi(region['bbox'], raw.shape)
            note = 'known_camera_region_assisted'
        else:
            roi = padded_roi(region['search_roi'], raw.shape, margin=0)
            note = 'no_crystal_envelope_search_box_fallback_not_confirmed_empty'
    else:
        mask = np.ones(raw.shape, dtype=bool)
        roi = [0, 0, w, h]
        region = {'status': 'different_camera_full_frame', 'bbox': None,
                  'search_roi': roi, 'note': 'No matched camera envelope configuration'}
        note = 'different_camera_out_of_training_distribution_full_frame'
    if split == 'diagnostic':
        note += ';diagnostic_exposure_gain_out_of_training_distribution'
    lo, hi = map(float, np.percentile(raw, [.5, 99.9]))
    constant = hi <= lo
    if constant:
        lo, hi = float(raw.min()), float(raw.max()) + 1.
        note += ';constant_percentile_display_range'
    return roi, region, mask, [lo, hi], note


def decode_response(response, tile):
    """Reject incomplete replies in their entirety; retain raw replies on disk."""
    record = {'tile': tile, 'response': response, 'points': [], 'status': 'failed'}
    try:
        if response.get('finish_reason') != 'stop':
            raise ValueError('Generation did not finish normally; partial JSON is not accepted')
        points = parse_points(response['text'], tile, 'normalized_1000', 1)
        body = response['text'].strip()
        # Empty whole-image abstention has no point-level flag to inspect.
        fenced = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', body, flags=re.S | re.I)
        if fenced:
            body = fenced.group(1).strip()
        envelope = json.loads(body)
        abstained = (isinstance(envelope, dict) and envelope.get('uncertain', False))
        abstained = abstained or any(p['model_uncertain'] for p in points)
        record.update(points=points, status='abstained' if abstained else 'complete')
    except (ValueError, KeyError, TypeError) as error:
        record['error'] = str(error)
    return record


def aggregate(entry, records, classical, merge_distance):
    proposed = [p for row in records if row['status'] == 'complete' for p in row['points']]
    points = sorted(deduplicate(proposed, merge_distance), key=lambda p: (p['y'], p['x']))
    failed = sum(row['status'] == 'failed' for row in records)
    abstained = sum(row['status'] == 'abstained' for row in records)
    complete = len(records) == len(entry['tiles']) and not failed and not abstained
    result = {'id': entry['id'], 'status': 'complete' if complete else 'incomplete',
              'coverage_complete': complete, 'coverage_scope': 'declared_roi_only',
              'roi': entry['roi'], 'full_frame_covered': complete and entry['roi'] == [0, 0, entry['shape'][1], entry['shape'][0]],
              'vlm_count': len(points), 'classical_count': len(classical),
              'failed_tiles': failed, 'abstained_tiles': abstained,
              'tiles_planned': len(entry['tiles']), 'tiles_processed': len(records),
              'inference_seconds': sum(r['response'].get('seconds', 0.) for r in records),
              'points': points, 'classical_points': classical, 'tiles': records,
              'coordinate_convention': 'zero_based_original_pixel_centers_x_column_y_row',
              'metric_interpretation': 'classical_teacher_agreement_not_independent_accuracy',
              'classical_reference_scope': 'original_adaptive_mask_for_known_camera;full_frame_for_other_camera',
              'classical_reference_available': entry['region']['status'] != 'no_crystal',
              'note': 'Incomplete counts are partial, not confirmed absence. Outside ROI was not inferred.'}
    for tolerance in (1, 3):
        if not result['classical_reference_available']:
            result['agreement_%dpx' % tolerance] = {'f1': None, 'tp': None, 'fp': None, 'fn': None,
                'all_correct': False, 'tolerance_px': tolerance,
                'reason': 'No crystal envelope is not a trusted negative reference'}
            continue
        metric = match_points([[p['x'], p['y']] for p in points],
                              [[p['x'], p['y']] for p in classical], tolerance)
        if not complete:
            metric['all_correct'] = False
        result['agreement_%dpx' % tolerance] = metric
    return result


def settings_for(args):
    import yaml
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    adapter = Path(args.adapter).resolve()
    adapter_config = json.loads((adapter / 'adapter_config.json').read_text(encoding='utf-8'))
    if adapter_config['lora_alpha'] != 2 * adapter_config['r']:
        raise ValueError('Adapter scale is incompatible with the trained loader')
    files = ['fullframe.py', 'fullframe_report.py', 'finetune.py', 'geometry.py', 'training_data.py']
    support_files = ['classical/region.py', 'classical/detection.py', 'data.py', 'evaluation.py']
    here = Path(__file__).parent
    return {'data': str(Path(args.data).resolve()), 'model': str(Path(args.model).resolve()),
            'adapter': str(adapter), 'adapter_sha256': sha256(adapter / 'adapter_model.safetensors'),
            'adapter_config_sha256': sha256(adapter / 'adapter_config.json'), 'rank': adapter_config['r'],
            'model_config_sha256': sha256(Path(args.model) / 'config.json'),
            'config': config, 'config_sha256': sha256(args.config), 'prompt': TRAINING_PROMPT,
            'image_tokens': 1024, 'tile_size': 64, 'overlap': 16, 'roi_padding': 8,
            'merge_distance': 2., 'max_new_tokens': args.max_new_tokens,
            'reserve_gib': args.reserve_gib, 'splits': ['test', 'val', 'train', 'diagnostic'],
            'input_manifest': str(Path(args.input_manifest).resolve()) if args.input_manifest else None,
            'input_manifest_sha256': sha256(args.input_manifest) if args.input_manifest else None,
            'code_sha256': {name: sha256(here / name) for name in files},
            'support_code_sha256': {name: sha256(here.parent / name) for name in support_files},
            'interpretation': 'Region-assisted VLM; original-image outputs, not full-frame coverage outside ROI; train frames are in-sample'}


def prepare(output, settings):
    root = Path(settings['data'])
    sources = [(split, p) for split in settings['splits'] for p in list_images(root / split)]
    if settings['input_manifest']:
        manifest = json.loads(Path(settings['input_manifest']).read_text(encoding='utf-8'))
        requested = [Path(p).resolve() for p in manifest['paths']]
        available = {p.resolve(): (split, p) for split, p in sources}
        if not requested or len(requested) != len(set(requested)) or any(p not in available for p in requested):
            raise ValueError('Selected originals must be distinct files from the declared data splits')
        sources = [available[p] for p in requested]
    # Show one frame from each current-camera split early; keep every source.
    first = [next(((s, p) for s, p in sources if s == split and p.suffix.lower() != '.npy'), None)
             for split in settings['splits']]
    first = [p for p in first if p is not None]
    sources = first + [p for p in sources if p not in first]
    images = []
    for split, path in sources:
        check_space(output, settings['reserve_gib'])
        raw = load_image(path)
        relative = path.relative_to(root).as_posix()
        roi, region, _, limits, note = frame_geometry(raw, split, settings['config']['region'])
        images.append({'id': split + '_' + hashlib.sha256(relative.encode()).hexdigest()[:16],
            'raw_path': str(path.resolve()), 'relative_path': relative, 'raw_sha256': sha256(path),
            'split': split, 'group': path.parent.relative_to(root).as_posix(), 'shape': list(raw.shape),
            'roi': roi, 'region': region, 'display_limits': limits, 'domain_note': note,
            'tiles': make_tiles(raw.shape, 64, 16, roi)})
        if len(images) % 100 == 0:
            print('Prepared %d / %d originals' % (len(images), len(sources)), flush=True)
    payload = {'settings': settings, 'images': images,
               'planned_tiles': sum(len(e['tiles']) for e in images)}
    payload['fingerprint'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    payload['created_utc'] = now()
    write_json(output / 'plan.json', payload)
    snapshot = output / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in settings['code_sha256']:
        (snapshot / name).write_bytes((Path(__file__).parent / name).read_bytes())
    for name in settings['support_code_sha256']:
        destination = snapshot / 'support' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((Path(__file__).parent.parent / name).read_bytes())
    return payload


@contextmanager
def keep_awake():
    function = None
    if os.name == 'nt':
        import ctypes
        function = ctypes.windll.kernel32.SetThreadExecutionState
        function.argtypes = [ctypes.c_uint]
        function.restype = ctypes.c_uint
        if not function(0x80000001):
            raise OSError('Could not keep this inference process awake')
    try:
        yield
    finally:
        if function is not None:
            function(0x80000000)


def run(args, output, plan):
    from .fullframe_report import render_frame, write_report
    settings = plan['settings']
    rows, completed_tiles, elapsed_generation = [], 0, 0.
    status = {'state': 'loading_model', 'started_utc': now(), 'pid': os.getpid(),
              'images_total': len(plan['images']), 'tiles_total': plan['planned_tiles'],
              'adapter_sha256': settings['adapter_sha256'], 'images_done': 0, 'tiles_done': 0}
    write_json(output / 'status.json', status)
    write_report(output, plan, rows, 'loading_model')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import torch
    from safetensors.torch import load_file
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('Requires the isolated CUDA training environment')
    torch.set_num_threads(4)
    attention, patched = efficient_attention_setup(torch)
    model, processor = load_model(Path(settings['model']), settings['rank'], torch)
    restore_adapter(model, load_file(str(Path(settings['adapter']) / 'adapter_model.safetensors')))
    model.eval()
    model.gradient_checkpointing_disable()
    status.update(state='running', windows_sdpa_gqa_materialization=patched)
    with keep_awake():
        for entry in plan['images']:
            free = check_space(output, settings['reserve_gib'])
            if sha256(entry['raw_path']) != entry['raw_sha256']:
                raise ValueError('Original changed since planning: ' + entry['raw_path'])
            raw = load_image(entry['raw_path'])
            folder = output / 'images' / entry['id']
            (folder / 'tiles').mkdir(parents=True, exist_ok=True)
            records = []
            for tile in entry['tiles']:
                free = check_space(output, settings['reserve_gib'])
                name = '%04d' % tile['id']
                response_path = folder / 'tiles' / (name + '.json')
                image_path = response_path.with_suffix('.png')
                if response_path.exists():
                    cached = json.loads(response_path.read_text(encoding='utf-8'))
                    if cached['fingerprint'] != plan['fingerprint'] or cached['tile'] != tile:
                        raise ValueError('Cached response is from a different plan')
                    response = cached['response']
                else:
                    render_tile(raw, tile, *entry['display_limits'], display_scale=1).save(image_path)
                    encoded, prefix = encode_record({'image': str(image_path), 'prompt': TRAINING_PROMPT},
                        processor, output, 1024, torch, include_answer=False)
                    inputs = to_device(encoded, torch)
                    start = time.perf_counter()
                    with torch.inference_mode(), attention():
                        generated = model.generate(**inputs, max_new_tokens=settings['max_new_tokens'],
                            do_sample=False, use_cache=True, pad_token_id=processor.tokenizer.pad_token_id)
                    torch.cuda.synchronize()
                    ids = generated[0, prefix:]
                    eos = model.generation_config.eos_token_id
                    eos = eos if isinstance(eos, list) else [eos]
                    ended = any(int(token) in eos for token in ids)
                    response = {'text': processor.tokenizer.decode(ids, skip_special_tokens=True).strip(),
                        'finish_reason': 'stop' if ended or len(ids) < settings['max_new_tokens'] else 'length',
                        'seconds': time.perf_counter() - start, 'generated_tokens': len(ids)}
                    write_json(response_path, {'fingerprint': plan['fingerprint'], 'tile': tile,
                                              'render_sha256': sha256(image_path), 'response': response})
                    del inputs, generated, encoded
                record = decode_response(response, tile)
                records.append(record)
                completed_tiles += 1
                elapsed_generation += response['seconds']
                status.update(current_image=entry['id'], current_path=entry['relative_path'],
                    current_tile=tile['id'], tiles_done=completed_tiles, images_done=len(rows),
                    updated_utc=now(), free_gib=free,
                    estimated_remaining_hours=(plan['planned_tiles']-completed_tiles) * elapsed_generation / completed_tiles / 3600)
                write_json(output / 'status.json', status)
                print('%s tile %d/%d: %s (%d points); total %d/%d' % (entry['id'],
                    tile['id']+1, len(entry['tiles']), record['status'], len(record['points']),
                    completed_tiles, plan['planned_tiles']), flush=True)
            _, _, mask, _, _ = frame_geometry(raw, entry['split'], settings['config']['region'])
            classical, _ = detect_ions(raw, settings['config']['detection'], region_mask=mask, noise_reference=raw)
            result = aggregate(entry, records, classical, settings['merge_distance'])
            write_json(folder / 'result.json', result)
            write_json(folder / 'vlm_ions.json', {key: result[key] for key in ('id', 'status', 'coverage_complete',
                'coverage_scope', 'roi', 'full_frame_covered', 'coordinate_convention', 'points', 'note')})
            render_frame(folder, raw, entry, result)
            row = {key: value for key, value in result.items() if key not in ('points', 'classical_points', 'tiles')}
            row.update({key: entry[key] for key in ('split', 'group', 'raw_path', 'roi', 'domain_note')})
            rows.append(row)
            status.update(images_done=len(rows), updated_utc=now())
            write_json(output / 'status.json', status)
            write_report(output, plan, rows, 'running')
            print('FRAME %d/%d %s: VLM=%d classical=%d %s' % (len(rows), len(plan['images']),
                entry['id'], result['vlm_count'], result['classical_count'], result['status']), flush=True)
        state = 'complete' if all(row['coverage_complete'] for row in rows) else 'complete_with_issues'
        status.update(state=state, finished_utc=now(), images_done=len(rows), free_gib=check_space(output, settings['reserve_gib']))
        write_json(output / 'status.json', status)
        write_report(output, plan, rows, state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data')
    parser.add_argument('--config', default='configs/existing_crystal.yaml')
    parser.add_argument('--model', default='checkpoints/Qwen3-VL-2B-Instruct-bf16')
    parser.add_argument('--adapter', default='runs/vlm_expand_20261103/experiment_4096_6epochs/training/best_adapter')
    parser.add_argument('--output', required=True)
    parser.add_argument('--input-manifest', help='Optional JSON {paths:[absolute raw paths]} to process selected originals first')
    parser.add_argument('--reserve-gib', type=float, default=8.)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not np.isfinite(args.reserve_gib) or args.reserve_gib < 0 or args.max_new_tokens < 1:
        raise ValueError('Invalid disk reserve or token limit')
    from ..cli import check_destination
    output = check_destination(args.output, [args.data, args.model, args.adapter, args.config])
    check_space(output, args.reserve_gib)
    settings = settings_for(args)
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            raise ValueError('Output is not empty; use a new directory or --resume')
        plan = json.loads((output / 'plan.json').read_text(encoding='utf-8'))
        if plan['settings'] != settings:
            raise ValueError('Resume code/configuration/model differs from saved plan')
    else:
        if args.resume:
            raise ValueError('Resume requires an existing plan')
        output.mkdir(parents=True, exist_ok=True)
        plan = prepare(output, settings)
    from .fullframe_report import write_report
    if args.prepare_only:
        write_report(output, plan, [], 'prepared')
        print(json.dumps({'images': len(plan['images']), 'tiles': plan['planned_tiles'], 'output': str(output)}), flush=True)
        return
    try:
        run(args, output, plan)
    except BaseException as error:
        status_path = output / 'status.json'
        status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
        status.update(state='failed', error=repr(error), updated_utc=now())
        write_json(status_path, status)
        rows = []
        for entry in plan['images']:
            result_path = output / 'images' / entry['id'] / 'result.json'
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding='utf-8'))
                row = {k: v for k, v in result.items() if k not in ('points', 'classical_points', 'tiles')}
                row.update({k: entry[k] for k in ('split', 'group', 'raw_path', 'roi', 'domain_note')})
                rows.append(row)
        write_report(output, plan, rows, 'failed')
        raise


if __name__ == '__main__':
    main()
