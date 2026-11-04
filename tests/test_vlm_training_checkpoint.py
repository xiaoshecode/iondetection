"""Safe persistence and exact CPU optimizer/RNG continuation, without a VLM."""
import copy
import json
import random
import re

import pytest
import torch

from ionrecog.vision_language.training_checkpoint import load_resume_checkpoint, save_resume_checkpoint


CONFIG = {'dataset': 'fixed_hash', 'model': 'fixed_revision', 'rank': 4, 'lr': .001}
_VERSION = tuple(map(int, re.match(r'(\d+)\.(\d+)', str(torch.__version__)).groups()))
pytestmark = pytest.mark.skipif(_VERSION < (2, 5), reason='Requires isolated torch2.5+ weights_only recovery runtime')


def minimal_state(step):
    return {'version': 1, 'completed_step': step, 'immutable_config': dict(CONFIG),
            'adapter_state': {'weight': torch.tensor([step], dtype=torch.float32)}}


def test_older_runtime_refused_without_unsafe_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(torch, '__version__', '1.13.0+cpu')
    with pytest.raises(ValueError, match='PyTorch 2.5'):
        save_resume_checkpoint(tmp_path, minimal_state(1), torch)


def test_hash_check_and_configuration_mismatch(tmp_path):
    target = save_resume_checkpoint(tmp_path, minimal_state(1), torch)
    restored = load_resume_checkpoint(tmp_path/'recovery/latest.json', CONFIG, torch)
    assert restored['completed_step'] == 1
    with pytest.raises(ValueError, match='configuration mismatch: rank'):
        load_resume_checkpoint(target, dict(CONFIG, rank=8), torch)
    with target.open('r+b') as stream:
        stream.seek(10)
        stream.write(b'corrupted')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        load_resume_checkpoint(target, CONFIG, torch)


def test_keep_three_and_duplicate_step_refusal(tmp_path):
    for step in range(1, 6):
        save_resume_checkpoint(tmp_path, minimal_state(step), torch)
    assert [p.name for p in sorted((tmp_path/'recovery').glob('step_*.pt'))] == [
        'step_000003.pt', 'step_000004.pt', 'step_000005.pt']
    assert len(list((tmp_path/'recovery').glob('*.sha256.json'))) == 3
    with pytest.raises(ValueError, match='already exists'):
        save_resume_checkpoint(tmp_path, minimal_state(5), torch)
    assert load_resume_checkpoint(tmp_path/'recovery/latest.json', CONFIG, torch)['completed_step'] == 5


def test_failed_write_does_not_destroy_previous_pointer(tmp_path, monkeypatch):
    save_resume_checkpoint(tmp_path, minimal_state(1), torch)
    original = (tmp_path/'recovery/latest.json').read_bytes()
    def fail_save(payload, stream):
        stream.write(b'partial')
        raise OSError('simulated write failure')
    with monkeypatch.context() as context:
        context.setattr(torch, 'save', fail_save)
        with pytest.raises(OSError, match='simulated'):
            save_resume_checkpoint(tmp_path, minimal_state(2), torch)
    assert (tmp_path/'recovery/latest.json').read_bytes() == original
    assert not (tmp_path/'recovery/writer.lock').exists()
    assert load_resume_checkpoint(tmp_path/'recovery/latest.json', CONFIG, torch)['completed_step'] == 1
    save_resume_checkpoint(tmp_path, minimal_state(2), torch)


def test_latest_pointer_cannot_escape_directory(tmp_path):
    save_resume_checkpoint(tmp_path, minimal_state(1), torch)
    pointer = tmp_path/'recovery/latest.json'
    value = json.loads(pointer.read_text())
    value['file'] = '../step_000001.pt'
    pointer.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='escapes'):
        load_resume_checkpoint(pointer, CONFIG, torch)


def test_unknown_objects_and_cycles_never_reach_serializer(tmp_path, monkeypatch):
    class Unknown:
        pass
    def unsafe_save(*args):
        pytest.fail('Unsafe state reached torch.save')
    monkeypatch.setattr(torch, 'save', unsafe_save)
    value = minimal_state(1)
    value['bad'] = Unknown()
    with pytest.raises(ValueError, match='unsupported object'):
        save_resume_checkpoint(tmp_path, value, torch)
    value['bad'] = value
    with pytest.raises(ValueError, match='cyclic'):
        save_resume_checkpoint(tmp_path, value, torch)


def test_optimizer_rng_and_shuffle_resume_matches_uninterrupted(tmp_path):
    torch.manual_seed(149)
    x, y = torch.randn(7, 3), torch.randn(7, 1)
    def build():
        model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.ReLU(),
                                    torch.nn.Dropout(.2), torch.nn.Linear(8, 1))
        return model, torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    model, optimizer = build()
    initial = copy.deepcopy(dict(model.state_dict()))
    rng, order, position = random.Random(211), list(range(len(x))), 0
    rng.shuffle(order)
    def steps(model, optimizer, rng, order, position, count):
        for _ in range(count):
            if position == len(order):
                rng.shuffle(order)
                position = 0
            index = order[position]
            position += 1
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x[index:index+1])-y[index:index+1]).square().mean()
            loss.backward()
            optimizer.step()
        return position
    position = steps(model, optimizer, rng, order, position, 3)
    state = {'version': 1, 'completed_step': 3, 'immutable_config': dict(CONFIG),
             'adapter_state': dict(model.state_dict()), 'best_adapter_state': dict(model.state_dict()),
             'optimizer_state': optimizer.state_dict(), 'initial_adapter': initial,
             'order': order, 'position': position, 'shuffle_rng_state': rng.getstate(),
             'torch_rng_state': torch.get_rng_state(), 'cuda_rng_states': [],
             'history': [{'step': i} for i in range(1, 4)], 'validation_history': [],
             'best_score': (.2, -1, -.7), 'best_step': 3, 'best_loss': .7,
             'initial_val': 1.2, 'elapsed_seconds': 2.3}
    target = save_resume_checkpoint(tmp_path, state, torch)
    final_position = steps(model, optimizer, rng, order, position, 8)
    expected_weights = copy.deepcopy(dict(model.state_dict()))
    expected_rng, expected_order = torch.get_rng_state().clone(), list(order)

    resumed, resumed_optimizer = build()  # Deliberately consumes RNG before restoration.
    recovered = load_resume_checkpoint(target, CONFIG, torch)
    resumed.load_state_dict(recovered['adapter_state'])
    resumed_optimizer.load_state_dict(recovered['optimizer_state'])
    resumed_rng = random.Random()
    resumed_rng.setstate(recovered['shuffle_rng_state'])
    torch.set_rng_state(recovered['torch_rng_state'])
    resumed_position = steps(resumed, resumed_optimizer, resumed_rng, recovered['order'], recovered['position'], 8)
    assert resumed_position == final_position
    assert recovered['order'] == expected_order
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for key, value in resumed.state_dict().items():
        assert torch.equal(value, expected_weights[key]), key
    assert all(int(value['step']) == 11 for value in resumed_optimizer.state.values())
