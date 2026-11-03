"""Online simulation-supervised training with a separate fixed validation stream."""
import json
import time
from pathlib import Path

import numpy as np

from .simulation import SimulationConfig, make_targets, sample_seed, simulate


def _aggregate(metrics):
    tp, fp, fn = (sum(m[k] for m in metrics) for k in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else (1. if not fn else 0.)
    recall = tp / (tp + fn) if tp + fn else 1.
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
            "all_correct_fraction": float(np.mean([m["all_correct"] for m in metrics])),
            "ground_truth_points": tp + fn}


def evaluate_model(model, simulation, seed, samples, device,
                   thresholds=(.1, .2, .3, .4, .5, .6, .7), min_distance=1.5):
    import torch
    from ..evaluation import match_points
    from .inference import decode_maps
    model.eval()
    predictions = []
    with torch.no_grad():
        for index in range(samples):
            raw, truth = simulate(simulation, sample_seed(seed, 1, index))
            image = ((raw - simulation.input_offset) / simulation.input_scale)[None, None]
            logits, offsets = model(torch.from_numpy(image).to(device))
            predictions.append((torch.sigmoid(logits)[0, 0].cpu().numpy(), offsets[0].cpu().numpy(), truth))
    curve = []
    for threshold in thresholds:
        metrics1, metrics3 = [], []
        for probability, offsets, truth in predictions:
            rows, _ = decode_maps(probability, offsets, threshold, min_distance)
            points = [[r["x"], r["y"]] for r in rows]
            metrics1.append(match_points(points, truth, 1.))
            metrics3.append(match_points(points, truth, 3.))
        curve.append({"threshold": float(threshold), "at_1px": _aggregate(metrics1),
                      "at_3px": _aggregate(metrics3)})
    # Deterministic tie-break: favor a higher precision, then a higher threshold.
    selected = max(curve, key=lambda row: (row["at_1px"]["f1"], row["at_1px"]["precision"], row["threshold"]))
    return {"selected": selected, "curve": curve, "samples": samples, "split": "synthetic_validation",
            "held_out_test_score": False,
            "note": "Threshold selected here; validation is not an untouched test set or experimental accuracy."}


def train_model(output, simulation=None, epochs=10, steps=100, batch_size=4,
                val_samples=16, channels=16, lr=.001, seed=42, device="cpu",
                num_threads=2, reserve_gib=2., resume=None, protected_sources=()):
    import torch
    from ..cli import write_json
    from ..localization_common import ensure_disk_space, prepare_output, save_checkpoint_atomic
    from .inference import choose_device
    from .models import PointNet, point_loss
    simulation = (simulation or SimulationConfig()).validate()
    for key, value in (("epochs", epochs), ("steps", steps), ("batch_size", batch_size),
                       ("val_samples", val_samples), ("num_threads", num_threads)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("%s must be a positive integer" % key)
    if not isinstance(seed, int) or seed < 0 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("seed must be nonnegative integer and lr finite and positive")
    if simulation.max_ions < 1 or simulation.empty_fraction >= 1:
        raise ValueError("Training must include positive scenes; max_ions >=1 and empty_fraction <1")
    torch.set_num_threads(num_threads)
    target_device = choose_device(device)
    torch.manual_seed(seed)
    model = PointNet(channels).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    settings = {"epochs": epochs, "steps": steps, "batch_size": batch_size, "val_samples": val_samples,
                "channels": channels, "lr": lr, "seed": seed, "device": str(target_device),
                "num_threads": num_threads, "simulation": simulation.to_dict()}
    start_epoch, best_f1, history = 0, -1., []
    if resume:
        state = torch.load(str(resume), map_location="cpu")
        if not isinstance(state, dict) or state.get("method") != "point_localization" or state.get("schema_version") != 1:
            raise ValueError("Resume requires a point_localization schema-1 checkpoint")
        if state.get("simulation") != simulation.to_dict() or state.get("channels") != channels:
            raise ValueError("Resume simulation/architecture must match; supply the original settings")
        saved_settings = state.get("train_settings", {})
        for key in ("steps", "batch_size", "seed", "val_samples", "lr"):
            if saved_settings.get(key) != settings[key]:
                raise ValueError("Resume must retain %s to preserve the training/validation streams" % key)
        start_epoch = int(state.get("completed_epochs", 0))
        if start_epoch < 1 or start_epoch >= epochs:
            raise ValueError("Resume requires --epochs greater than the completed epoch count")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        # torch 1.13 optimizer state must follow the resumed model device.
        for opt_state in optimizer.state.values():
            for key, value in opt_state.items():
                if torch.is_tensor(value):
                    opt_state[key] = value.to(target_device)
        history = list(state.get("history", []))
        # A fresh output's best.pt means best of this continuation. The previous
        # experiment's best checkpoint remains independently available.
    sources = list(protected_sources) + ([Path(resume)] if resume else [])
    output = prepare_output(output, sources, reserve_gib, estimated_bytes=64 * 1024 ** 2)
    write_json(output / "configuration.json", settings)
    started = time.monotonic()
    selected_threshold = .3
    best_epoch = None
    for epoch in range(start_epoch, epochs):
        ensure_disk_space(output, reserve_gib, estimated_bytes=32 * 1024 ** 2)
        model.train()
        losses = []
        for step in range(steps):
            images, heatmaps, offsets, centers = [], [], [], []
            for item in range(batch_size):
                index = (epoch * steps + step) * batch_size + item
                raw, truth = simulate(simulation, sample_seed(seed, 0, index))
                heatmap, offset, center = make_targets(truth, simulation.size, simulation.heatmap_sigma)
                images.append(((raw - simulation.input_offset) / simulation.input_scale)[None])
                heatmaps.append(heatmap)
                offsets.append(offset)
                centers.append(center)
            batch = [torch.from_numpy(np.stack(x)).to(target_device) for x in (images, heatmaps, offsets, centers)]
            optimizer.zero_grad(set_to_none=True)
            logits, predicted_offsets = model(batch[0])
            loss, detection_loss, offset_loss = point_loss(logits, predicted_offsets, *batch[1:])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; check simulation scale/brightness")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if (step + 1) % 20 == 0 or step + 1 == steps:
                write_json(output / "progress.json", {"status": "training", "epoch": epoch + 1, "epochs": epochs,
                    "step": step + 1, "steps_per_epoch": steps, "loss": losses[-1],
                    "seconds": time.monotonic() - started})
                print("points epoch %d/%d step %d/%d loss %.5f" % (epoch + 1, epochs, step + 1, steps, losses[-1]))
        validation = evaluate_model(model, simulation, seed, val_samples, target_device)
        selected_threshold = validation["selected"]["threshold"]
        score = validation["selected"]["at_1px"]["f1"]
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation": validation}
        history.append(row)
        checkpoint = {"schema_version": 1, "method": "point_localization", "channels": channels,
                      "simulation": simulation.to_dict(), "train_settings": settings,
                      "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                      "completed_epochs": epoch + 1, "selected_threshold": selected_threshold,
                      "history": history, "training_source": "simulation_supervised",
                      "experimental_accuracy_verified": False}
        ensure_disk_space(output, reserve_gib, estimated_bytes=32 * 1024 ** 2)
        save_checkpoint_atomic(output / "last.pt", checkpoint)
        if score > best_f1:
            best_f1, best_epoch = score, epoch + 1
            save_checkpoint_atomic(output / "best.pt", checkpoint)
        write_json(output / "history.json", history)
    summary = {"method": "point_localization", "training_source": "simulation_supervised",
               "best_checkpoint": str((output / "best.pt").resolve()), "last_checkpoint": str((output / "last.pt").resolve()),
               "best_epoch": best_epoch, "best_validation_f1_1px": best_f1, "completed_epochs": epochs,
               "seconds_this_run": time.monotonic() - started, "resumed_from": str(resume) if resume else None,
               "validation_is_threshold_selection_set": True, "experimental_accuracy_verified": False,
               "note": "Synthetic validation only. A short smoke run does not establish a usable detector; review recall, false positives and independent real references."}
    write_json(output / "summary.json", summary)
    write_json(output / "progress.json", dict(summary, status="completed"))
    return summary
