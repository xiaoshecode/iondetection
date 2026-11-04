"""Atomic, hash-checked local recovery checkpoints; never use unsafe pickle loading.

Requires the isolated PyTorch 2.5+ training runtime. The caller owns experiment-specific state and supplies the immutable configuration
fields that must match on restore. Only basic containers, primitives and tensors
are serialized. A writer lock prevents two jobs publishing into the same folder.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid
import warnings


_NAME = re.compile(r"step_(\d{6,})\.pt\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _runtime(torch):
    match = re.match(r"(\d+)\.(\d+)", str(torch.__version__))
    if not match or tuple(map(int, match.groups())) < (2, 5):
        raise ValueError("Recovery checkpoints require PyTorch 2.5+ weights_only loading; use the isolated training environment")


def _digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def _plain(value, torch, copy_tensors=False, stack=None):
    """Reject arbitrary classes before torch.save can invoke custom reducers."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Recovery state contains a nonfinite number")
        return value
    if type(value) in (torch.Tensor, torch.nn.Parameter):
        return value.detach().cpu().clone() if copy_tensors else value
    if type(value) not in (dict, list, tuple):
        raise ValueError("Recovery state contains an unsupported object: " + type(value).__name__)
    stack = set() if stack is None else stack
    if id(value) in stack:
        raise ValueError("Recovery state contains a cyclic container")
    stack.add(id(value))
    try:
        if type(value) is dict:
            if any(type(key) not in (str, int) for key in value):
                raise ValueError("Recovery dictionaries require string or integer keys")
            return {key: _plain(item, torch, copy_tensors, stack) for key, item in value.items()}
        items = [_plain(item, torch, copy_tensors, stack) for item in value]
        return tuple(items) if type(value) is tuple else items
    finally:
        stack.remove(id(value))


def _configuration(value):
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError("immutable_config must be a plain dictionary with string keys")
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Immutable configuration must be finite JSON data") from exc


def _state(value, torch, copy_tensors=False):
    if type(value) is not dict or value.get("version") != 1 or type(value.get("version")) is not int:
        raise ValueError("Unsupported recovery state version")
    if type(value.get("completed_step")) is not int or value["completed_step"] < 0:
        raise ValueError("completed_step must be a nonnegative integer")
    _configuration(value.get("immutable_config"))
    return _plain(value, torch, copy_tensors)


def _inside_file(root, name):
    relative = Path(name)
    if not name or relative.name != name or relative.is_absolute() or relative.drive:
        raise ValueError("Checkpoint filename escapes recovery directory")
    path = root / name
    if path.is_symlink() or path.resolve().parent != root:
        raise ValueError("Checkpoint path escapes recovery directory")
    return path


def _read_json(path):
    def reject(value):
        raise ValueError("Nonfinite checkpoint metadata")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def _metadata(root, value):
    if type(value) is not dict or type(value.get("version")) is not int or value["version"] != 1:
        raise ValueError("Unsupported recovery metadata")
    name = value.get("file")
    if type(name) is not str:
        raise ValueError("Invalid checkpoint filename")
    path = _inside_file(root, name)
    match = _NAME.fullmatch(name)
    if not match or type(value.get("completed_step")) is not int or int(match.group(1)) != value["completed_step"]:
        raise ValueError("Checkpoint filename and completed step differ")
    if type(value.get("sha256")) is not str or not _SHA.fullmatch(value["sha256"]):
        raise ValueError("Invalid checkpoint SHA256")
    if type(value.get("bytes")) is not int or value["bytes"] < 1:
        raise ValueError("Invalid checkpoint byte count")
    return path


def _verified(root, meta):
    path = _metadata(root, meta)
    if not path.is_file() or path.stat().st_size != meta["bytes"] or _digest(path) != meta["sha256"]:
        raise ValueError("Recovery checkpoint size or SHA256 mismatch")
    return path


def _atomic_json(path, value):
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _prune(root, keep):
    candidates = []
    for path in root.glob("step_*.pt"):
        if not _NAME.fullmatch(path.name):
            continue
        try:
            sidecar = _inside_file(root, path.name + ".sha256.json")
            meta = _read_json(sidecar)
            verified = _verified(root, meta)
            if verified != path:
                continue
            candidates.append((meta["completed_step"], path, sidecar))
        except (OSError, ValueError, TypeError):
            # Preserve unknown or corrupted artifacts for diagnosis, not deletion.
            continue
    for _, path, sidecar in sorted(candidates, reverse=True)[keep:]:
        try:
            # Only individual, previously verified files within this recovery root.
            if path.resolve().parent != root or sidecar.resolve().parent != root:
                raise ValueError("Refusing checkpoint cleanup outside recovery directory")
            path.unlink()
            sidecar.unlink()
        except OSError as exc:
            warnings.warn("Could not prune an older recovery checkpoint: " + str(exc), RuntimeWarning)


def save_resume_checkpoint(output, state, torch, keep=3):
    """Publish a complete local snapshot, then atomically update latest.json.

    The same step is never overwritten. At least two valid generations are kept.
    Interrupted writes leave the previous pointer intact. An unclean writer.lock
    does not prevent reading; recover into a new output directory after a crash.
    """
    _runtime(torch)
    if type(keep) is not int or keep < 2:
        raise ValueError("Keep at least two recovery checkpoints")
    payload = _state(state, torch, copy_tensors=True)
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    root = destination / "recovery"
    if root.is_symlink() or root.resolve().parent != destination:
        raise ValueError("Recovery directory escapes output")
    root.mkdir(exist_ok=True)
    root = root.resolve()
    step = payload["completed_step"]
    target = _inside_file(root, "step_%06d.pt" % step)
    sidecar = _inside_file(root, target.name + ".sha256.json")
    latest = _inside_file(root, "latest.json")
    lock = _inside_file(root, "writer.lock")
    token = "%d:%s" % (os.getpid(), uuid.uuid4().hex)
    temporary = _inside_file(root, "." + target.name + "." + uuid.uuid4().hex + ".tmp")
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(token)
    try:
        if target.exists() or sidecar.exists():
            raise ValueError("Recovery checkpoint already exists; never overwrite a saved step")
        if latest.exists():
            previous = _read_json(latest)
            _verified(root, previous)
            if previous["completed_step"] >= step:
                raise ValueError("Recovery checkpoints must advance completed_step")
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        meta = {"version": 1, "file": target.name, "completed_step": step,
                "sha256": _digest(temporary), "bytes": temporary.stat().st_size}
        temporary.replace(target)
        _atomic_json(sidecar, meta)
        _atomic_json(latest, meta)
        _prune(root, keep)
        return target
    finally:
        if temporary.exists():
            temporary.unlink()
        if lock.is_file() and lock.read_text(encoding="utf-8") == token:
            lock.unlink()


def load_resume_checkpoint(path, expected, torch):
    """Verify hashes and every expected immutable field before returning state.

    expected may omit the total step budget so a caller can extend a run. Include
    all model/data/code/training settings that must remain identical there.
    """
    _runtime(torch)
    _configuration(expected)
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Recovery input must not be a symbolic link")
    path = path.resolve()
    root = path.parent
    if path.name == "latest.json":
        meta = _read_json(path)
        target = _metadata(root, meta)
    elif _NAME.fullmatch(path.name):
        target = path
        meta = _read_json(_inside_file(root, target.name + ".sha256.json"))
    else:
        raise ValueError("Expected latest.json or step_XXXXXX.pt")
    sidecar = _inside_file(root, target.name + ".sha256.json")
    stored_meta = _read_json(sidecar)
    if stored_meta != meta or _verified(root, meta) != target:
        raise ValueError("Recovery pointer and sidecar disagree")
    # No weights_only=False fallback, including on older PyTorch versions.
    state = _state(torch.load(target, map_location="cpu", weights_only=True), torch)
    if state["completed_step"] != meta["completed_step"]:
        raise ValueError("Recovery state and metadata step differ")
    for key, value in expected.items():
        actual = state["immutable_config"].get(key)
        if key not in state["immutable_config"] or _configuration({key: actual}) != _configuration({key: value}):
            raise ValueError("Recovery immutable configuration mismatch: " + key)
    return state
