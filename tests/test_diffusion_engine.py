"""CPU integration checks: optimizer resume, data identity and DDIM sampling."""
import copy
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from ionrecog.diffusion.engine import Restorer, effective_config, load_checkpoint, train


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(steps=2):
    return {"data": {"patch_size": 16, "scale": 1000.0}, "model": {"base_channels": 4},
            "diffusion": {"timesteps": 8, "sampling_steps": 3},
            "train": {"steps": steps, "batch_size": 1, "grad_accum": 2, "num_threads": 1,
                      "validate_every": 1, "save_every": 1, "val_batches": 1, "ema_decay": 0.8},
            "inference": {"tile_size": 16, "overlap": 3}}


def manifest(tmp_path, seed=3):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records = []
    for index, split in enumerate(("train", "val", "test")):
        y, x = np.mgrid[:19, :23]
        target = 20 + index * 2 + 500 * np.exp(-((x - (8 + index))**2 + (y - 10)**2) / 7)
        observed = target + rng.normal(0, 5, size=target.shape)
        np.save(tmp_path / (split + "_input.npy"), observed.astype(np.float32))
        np.save(tmp_path / (split + "_target.npy"), target.astype(np.float32))
        records.append({"id": split, "split": split, "group_id": split + "_crystal",
                        "input": split + "_input.npy", "target": split + "_target.npy",
                        "target_kind": "synthetic_clean", "points": [[8 + index, 10]]})
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path


def test_resume_matches_uninterrupted_model_optimizer_and_ema(tmp_path):
    data = manifest(tmp_path / "data")
    train(config(2), data, tmp_path / "full", device="cpu")
    train(config(1), data, tmp_path / "resumed", device="cpu")
    train(config(2), data, tmp_path / "resumed", device="cpu", resume=tmp_path / "resumed" / "last.pt")
    full = load_checkpoint(tmp_path / "full" / "last.pt")
    resumed = load_checkpoint(tmp_path / "resumed" / "last.pt")
    assert full["step"] == resumed["step"] == 2
    for state in ("model_state", "ema_state"):
        for key in full[state]:
            assert torch.equal(full[state][key], resumed[state][key]), (state, key)
    for key in full["optimizer_state"]["state"]:
        for name, value in full["optimizer_state"]["state"][key].items():
            assert torch.equal(value, resumed["optimizer_state"]["state"][key][name])
    assert all(t.device.type == "cpu" for t in resumed["model_state"].values())


def test_restore_finite_seeded_tiled_output_and_invalid_steps(tmp_path):
    data = manifest(tmp_path / "data")
    checkpoint = train(config(1), data, tmp_path / "run", device="cpu")
    restorer = Restorer(checkpoint, device="cpu")
    image = np.load(tmp_path / "data" / "test_input.npy")
    first = restorer.restore(image, seed=92)
    second = restorer.restore(image, seed=92)
    assert first.shape == image.shape
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    assert first.min() >= 0 and first.max() <= 1000
    assert not np.array_equal(first, restorer.restore(image, seed=93))
    for bad in (0, 9, 2.5, True):
        with pytest.raises(ValueError, match="sampling_steps"):
            restorer.restore(image, sampling_steps=bad)


def test_resume_rejects_changed_source_and_configuration(tmp_path):
    data = manifest(tmp_path / "data")
    output = tmp_path / "run"
    train(config(1), data, output, device="cpu")
    changed = config(2)
    changed["train"]["lr"] = 0.1
    with pytest.raises(ValueError, match="identical configuration"):
        train(changed, data, output, device="cpu", resume=output / "last.pt")
    original = np.load(tmp_path / "data" / "train_input.npy")
    np.save(tmp_path / "data" / "train_input.npy", original + 1)
    with pytest.raises(ValueError, match="content changed"):
        train(config(2), data, output, device="cpu", resume=output / "last.pt")


def test_init_weights_on_new_data_and_refuse_overwriting_run(tmp_path):
    first = manifest(tmp_path / "first")
    second = manifest(tmp_path / "second", seed=7)
    checkpoint = train(config(1), first, tmp_path / "pretrained", device="cpu")
    output = tmp_path / "finetuned"
    result = train(config(1), second, output, device="cpu", init_checkpoint=checkpoint)
    assert result.is_file()
    assert load_checkpoint(result)["step"] == 1
    with pytest.raises(FileExistsError):
        train(config(1), second, output, device="cpu")
    bad = config(1)
    bad["data"]["scale"] = 2000
    with pytest.raises(ValueError, match="scale and offset"):
        train(bad, second, tmp_path / "bad_init", device="cpu", init_checkpoint=checkpoint)


@pytest.mark.parametrize("section,key,value", [("data", "scale", 0), ("train", "steps", True),
                                               ("train", "lr", float("nan")), ("train", "precision", "fp8"),
                                               ("inference", "overlap", 12)])
def test_config_rejects_bad_training_arguments(section, key, value):
    settings = copy.deepcopy(config())
    settings.setdefault(section, {})[key] = value
    with pytest.raises(ValueError):
        effective_config(settings)


def test_training_refuses_output_inside_source_tree(tmp_path):
    data = manifest(tmp_path / "data")
    for output in (tmp_path / "data" / "train_input.npy", tmp_path, tmp_path / "data" / "new_run"):
        with pytest.raises(ValueError, match="[Oo]utput"):
            train(config(1), data, output, device="cpu")


def _ddp_worker(rank, data, output, rendezvous):
    import torch.distributed as dist
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        train(config(1), data, output, device="cpu")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="PyTorch distributed unavailable")
def test_two_cpu_ranks_train_one_checkpoint(tmp_path):
    """Exercise real DDP reductions and rank-zero writing without CUDA."""
    data = manifest(tmp_path / "data")
    context = torch.multiprocessing.spawn(_ddp_worker,
        args=(str(data), str(tmp_path / "ddp"), str(tmp_path / "rendezvous")),
        nprocs=2, join=False)
    deadline = time.monotonic() + 50
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("CPU DDP test exceeded 50 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    checkpoint = load_checkpoint(tmp_path / "ddp" / "last.pt")
    assert checkpoint["world_size"] == 2 and checkpoint["step"] == 1
    assert np.isfinite(checkpoint["history"][0]["val_ema_noise_mse"])
