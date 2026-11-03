"""Isolated-crop self-supervision and conservative classical-candidate refinement."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..data import _check_image, list_images, load_image
from ..localization_common import ensure_disk_space, prepare_output, save_checkpoint_atomic
from .model import SignalCentroidNet, equivariance_loss, transform_image


METHOD = "equivariant_centroid_refinement_v1"
DEFAULTS = dict(patch_size=17, max_shift=2, channels=16, epochs=20,
                steps_per_epoch=50, batch_size=8, views=3, val_steps=10,
                learning_rate=0.001, seed=42, noise_std=0.005, num_threads=2,
                offset=0.0, scale=None, weight_floor=0.1)


def validate_config(config=None):
    c = dict(DEFAULTS)
    if config:
        unknown = set(config) - set(c)
        if unknown:
            raise ValueError("Unknown equivariant config keys: " + str(sorted(unknown)))
        c.update(config)
    for key in ("patch_size", "channels", "epochs", "steps_per_epoch", "batch_size", "views", "val_steps", "num_threads"):
        if not isinstance(c[key], (int, np.integer)) or isinstance(c[key], bool) or c[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if not isinstance(c["max_shift"], (int, np.integer)) or c["max_shift"] < 1:
        raise ValueError("max_shift must be a positive integer to constrain fixed-center collapse")
    if not isinstance(c["seed"], (int, np.integer)) or c["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if c["patch_size"] % 2 != 1 or c["patch_size"] < 9 or c["patch_size"] > 129:
        raise ValueError("patch_size must be odd and between 9 and 129")
    if c["patch_size"] - 2 * c["max_shift"] < 5:
        raise ValueError("Translation margin leaves less than a 5-pixel isolated crop")
    if not 4 <= c["channels"] <= 128 or c["views"] < 2:
        raise ValueError("Require channels in [4,128] and views >=2")
    for key in ("learning_rate", "weight_floor"):
        if not np.isfinite(c[key]) or not 0 < c[key] <= 1:
            raise ValueError(key + " must be finite and in (0,1]")
    if not np.isfinite(c["offset"]) or not np.isfinite(c["noise_std"]) or c["noise_std"] < 0:
        raise ValueError("Require finite offset and finite nonnegative noise_std")
    if c["scale"] is not None and (not np.isfinite(c["scale"]) or c["scale"] <= 0):
        raise ValueError("scale must be null or finite and positive")
    return c


def _device(name):
    if name == "auto":
        name = "cpu"  # Explicit opt-in required to share the currently busy GPU.
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is not available in this Python environment")
    try:
        return torch.device(name)
    except (TypeError, RuntimeError) as error:
        raise ValueError("Invalid PyTorch device: " + str(name)) from error


def _hash_image(image):
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(image, dtype="<f4").tobytes())
    return digest.hexdigest()


def pad_crop(image, size, margin):
    image = _check_image(image, "isolated-ion crop")
    if np.any(image < 0):
        raise ValueError("Isolated-ion crops must contain nonnegative raw pixels")
    if min(image.shape) < 5 or max(image.shape) > size - 2 * margin:
        raise ValueError("Isolated crops must be >=5 pixels and <=%d per side; do not resize or auto-crop dense images" % (size - 2 * margin))
    if float(np.max(image) - np.min(image)) <= 0:
        raise ValueError("An isolated-ion crop must contain nonconstant signal")
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    out = np.full((size, size), float(np.median(border)), dtype=np.float32)
    y0, x0 = (size - image.shape[0]) // 2, (size - image.shape[1]) // 2
    out[y0:y0 + image.shape[0], x0:x0 + image.shape[1]] = image
    return out


def load_crop_splits(train_root, val_root, config):
    """Keep source crops isolated; pixel hashes reject copied validation images."""
    train_root, val_root = Path(train_root).resolve(), Path(val_root).resolve()
    if train_root == val_root or train_root in val_root.parents or val_root in train_root.parents:
        raise ValueError("Train and validation inputs must be separate, nonnested paths")
    datasets, records, seen = [], [], set()
    for split, root in (("train", train_root), ("val", val_root)):
        paths = list_images(root)
        if not paths:
            raise ValueError("No isolated-ion crops in " + str(root))
        canvases, entries = [], []
        for path in paths:
            image = load_image(path)
            digest = _hash_image(image)
            if split == "val" and digest in seen:
                raise ValueError("Train/validation pixel-duplicate crop: " + str(path))
            if split == "train":
                seen.add(digest)
            canvases.append(pad_crop(image, config["patch_size"], config["max_shift"]))
            entries.append(dict(path=str(path.resolve()), sha256=digest, shape=list(image.shape)))
        datasets.append(np.stack(canvases))
        records.append({"split": split, "crops": entries})
    return datasets[0], datasets[1], records


def make_batch(canvases, config, rng, device):
    images, matrices = [], []
    for _ in range(config["batch_size"]):
        canvas = canvases[int(rng.integers(len(canvases)))]
        for _ in range(config["views"]):
            transformed, matrix = transform_image(canvas, int(rng.integers(4)), bool(rng.integers(2)),
                int(rng.integers(-config["max_shift"], config["max_shift"] + 1)),
                int(rng.integers(-config["max_shift"], config["max_shift"] + 1)))
            normalized = (transformed - config["offset"]) / config["scale"]
            if config["noise_std"]:
                # Extra independent perturbation, not a calibrated EMCCD noise model.
                normalized = normalized + rng.normal(0, config["noise_std"], normalized.shape)
            images.append(normalized.astype(np.float32))
            matrices.append(matrix)
    return (torch.from_numpy(np.stack(images)[:, None]).to(device),
            torch.from_numpy(np.stack(matrices)).to(device))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _validation_loss(model, val, config, target):
    model.eval()
    losses = []
    # Fixed augmentations on independent held-out crops for every checkpoint.
    rng = np.random.default_rng(config["seed"] + 1000003)
    with torch.no_grad():
        for _ in range(config["val_steps"]):
            image, matrices = make_batch(val, config, rng, target)
            loss = equivariance_loss(model(image), matrices, config["batch_size"], config["views"])
            losses.append(float(loss.cpu()))
    score = float(np.mean(losses))
    if not np.isfinite(score):
        raise ValueError("Non-finite validation consistency loss")
    return score


def train_model(train_root, val_root, output, config=None, device="cpu", confirm_single_ion=False, reserve_gib=2.0):
    if not confirm_single_ion:
        raise ValueError("Training requires verified single-ion crops: pass --confirm-single-ion after checking the data")
    c = validate_config(config)
    train, val, records = load_crop_splits(train_root, val_root, c)
    if c["scale"] is None:
        c["scale"] = float(np.max(train) - c["offset"])
        if c["scale"] <= 0:
            raise ValueError("Training data maximum must exceed offset to estimate scale")
    output = prepare_output(output, sources=(train_root, val_root), reserve_gib=reserve_gib, estimated_bytes=16 * 1024 ** 2)
    target = _device(device)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(c["num_threads"])
    torch.manual_seed(c["seed"])
    model = SignalCentroidNet(c["channels"], c["weight_floor"]).to(target)
    optimizer = torch.optim.Adam(model.parameters(), lr=c["learning_rate"])
    history, best = [], float("inf")
    rng = np.random.default_rng(c["seed"])
    _write_json(output / "dataset.json", {"splits": records, "confirm_single_ion": True,
        "note": "Coordinate labels are not used. Pixel hashes cannot establish acquisition provenance."})
    _write_json(output / "config.json", c)
    start = time.perf_counter()
    try:
        baseline = _validation_loss(model, val, c, target)
        best = baseline
        # A learned checkpoint need not beat the initial image-dependent centroid.
        # Keep epoch zero eligible rather than force a worse trained model.
        save_checkpoint_atomic(output / "best.pt", dict(method=METHOD, schema_version=1, config=c,
            state_dict=model.state_dict(), epoch=0, validation_consistency_px2=baseline,
            note="Initial foreground-centroid baseline; geometric consistency is not accuracy."))
        history.append(dict(epoch=0, train_consistency_px2=None, val_consistency_px2=baseline))
        for epoch in range(1, c["epochs"] + 1):
            ensure_disk_space(output, reserve_gib, 8 * 1024 ** 2)
            model.train()
            train_losses = []
            for _ in range(c["steps_per_epoch"]):
                image, matrices = make_batch(train, c, rng, target)
                optimizer.zero_grad()
                loss = equivariance_loss(model(image), matrices, c["batch_size"], c["views"])
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite geometric consistency loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            score = _validation_loss(model, val, c, target)
            row = dict(epoch=epoch, train_consistency_px2=float(np.mean(train_losses)), val_consistency_px2=score)
            history.append(row)
            state = dict(method=METHOD, schema_version=1, config=c, state_dict=model.state_dict(),
                         epoch=epoch, validation_consistency_px2=score,
                         note="Consistency is not independent localization accuracy; candidate refinement only.")
            if score < best:
                best = score
                save_checkpoint_atomic(output / "best.pt", state)
            save_checkpoint_atomic(output / "last.pt", state)
            _write_json(output / "history.json", history)
            _write_json(output / "progress.json", dict(epoch=epoch, epochs=c["epochs"], complete=False,
                        seconds=time.perf_counter() - start, **{k: v for k, v in row.items() if k != "epoch"}))
            print("Equivariant epoch %d/%d: train %.6f, val %.6f px^2" % (epoch, c["epochs"], row["train_consistency_px2"], score))
        _write_json(output / "summary.json", dict(method=METHOD, complete=True, epochs=c["epochs"],
            best_validation_consistency_px2=best, seconds=time.perf_counter() - start,
            initial_centroid_consistency_px2=baseline,
            improved_over_initial_centroid=bool(best < baseline),
            metrics_are_ground_truth_accuracy=False, checkpoint=str(output / "best.pt")))
        _write_json(output / "progress.json", dict(epoch=c["epochs"], epochs=c["epochs"], complete=True,
                    seconds=time.perf_counter() - start))
    finally:
        torch.set_num_threads(previous_threads)
    return output / "best.pt"


def load_model(checkpoint, device="cpu"):
    # Load only trusted local research checkpoints (torch serialization uses pickle).
    target = _device(device)
    state = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(state, dict) or state.get("method") != METHOD or state.get("schema_version") != 1:
        raise ValueError("Checkpoint is not an equivariant centroid refinement v1 model")
    if not isinstance(state.get("config"), dict) or not isinstance(state.get("state_dict"), dict):
        raise ValueError("Checkpoint lacks valid config or state_dict")
    if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
           for value in state["state_dict"].values()):
        raise ValueError("Checkpoint parameters must be finite tensors")
    c = validate_config(state["config"])
    if c["scale"] is None:
        raise ValueError("Checkpoint must preserve a fixed fitted intensity scale")
    model = SignalCentroidNet(c["channels"], c["weight_floor"])
    try:
        model.load_state_dict(state["state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError("Checkpoint parameter shapes do not match its architecture") from error
    model.to(target).eval()
    return model, c, target


class EquivariantRefiner:
    """Classical proposals plus geometric learned refinement of isolated sites.

    Neighbor-contaminated, edge or excessive-shift crops retain their classical
    coordinates, with explicit counters. This method cannot recover a site
    missed by the classical detector and does not change the candidate count.
    """
    def __init__(self, checkpoint, detection_config=None, device="cpu", max_refine_shift=2.0,
                 isolation_distance=None, num_threads=2):
        if not np.isfinite(max_refine_shift) or max_refine_shift <= 0:
            raise ValueError("max_refine_shift must be finite and positive")
        if not isinstance(num_threads, int) or num_threads < 1:
            raise ValueError("num_threads must be a positive integer")
        self.model, self.config, self.device = load_model(checkpoint, device)
        self.detection_config = detection_config
        self.max_refine_shift = float(max_refine_shift)
        self.num_threads = num_threads
        inner = self.config["patch_size"] - 2 * self.config["max_shift"]
        default_distance = np.sqrt(2.) * (inner // 2) + max_refine_shift
        self.isolation_distance = default_distance if isolation_distance is None else float(isolation_distance)
        if not np.isfinite(self.isolation_distance) or self.isolation_distance <= 0:
            raise ValueError("isolation_distance must be finite and positive")

    def __call__(self, raw, region_mask=None):
        from scipy.spatial import cKDTree
        from ..classical.detection import detect_ions
        raw = _check_image(raw, "equivariant inference input")
        ions, _ = detect_ions(raw, self.detection_config, region_mask=region_mask)
        diag = dict(initial_candidates=len(ions), refined=0, unrefined_neighbour=0,
                    unrefined_edge=0, unrefined_shift=0, unrefined_region=0,
                    mode="classical_candidates_with_learned_centroid_refinement",
                    cannot_recover_classical_misses=True,
                    isolation_distance_px=float(self.isolation_distance))
        if not ions:
            return {"ions": [], "diagnostics": diag}
        h, w = raw.shape
        points = np.array([[ion["x"], ion["y"]] for ion in ions])
        distances = cKDTree(points).query(points, k=2)[0][:, 1] if len(points) > 1 else np.array([np.inf])
        c = self.config
        inner = c["patch_size"] - 2 * c["max_shift"]
        half = inner // 2
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(self.num_threads)
        try:
            with torch.no_grad():
                for i, ion in enumerate(ions):
                    ion["refined"] = False
                    ion["proposal_x"], ion["proposal_y"] = ion["x"], ion["y"]
                    if distances[i] < self.isolation_distance:
                        diag["unrefined_neighbour"] += 1
                        continue
                    px, py = int(np.floor(ion["x"] + .5)), int(np.floor(ion["y"] + .5))
                    if px - half < 0 or py - half < 0 or px + half >= w or py + half >= h:
                        diag["unrefined_edge"] += 1
                        continue
                    patch = raw[py - half:py + half + 1, px - half:px + half + 1]
                    canvas = pad_crop(patch, c["patch_size"], c["max_shift"])
                    image = torch.from_numpy(((canvas - c["offset"]) / c["scale"])[None, None]).to(self.device)
                    xy = self.model(image)[0].cpu().numpy()
                    x = float(xy[0]) + px - half - c["max_shift"]
                    y = float(xy[1]) + py - half - c["max_shift"]
                    if not np.isfinite([x, y]).all() or np.hypot(x - ion["x"], y - ion["y"]) > self.max_refine_shift:
                        diag["unrefined_shift"] += 1
                        continue
                    ix, iy = int(np.floor(x + .5)), int(np.floor(y + .5))
                    if not (0 <= ix < w and 0 <= iy < h) or (region_mask is not None and not region_mask[iy, ix]):
                        diag["unrefined_region"] += 1
                        continue
                    ion.update(x=x, y=y, refined=True)
                    diag["refined"] += 1
        finally:
            torch.set_num_threads(previous_threads)
        return {"ions": ions, "diagnostics": diag}
