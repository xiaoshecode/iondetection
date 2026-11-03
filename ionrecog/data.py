"""Grayscale image I/O with explicit, fixed intensity normalization.

No per-image contrast stretching is performed on training data. Coordinates used
throughout the project are zero-based: x is a column and y is a row.
"""

from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".npy"}
PathLike = Union[str, Path]


def list_images(root: PathLike) -> List[Path]:
    """Return supported images in deterministic relative-path order.

    A single supported file is also accepted. Missing inputs and unsupported
    single files raise an error; an existing empty directory returns an empty
    list so that callers can issue task-specific instructions.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError("Image input does not exist: {}".format(root))
    if root.is_file():
        if root.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError("Unsupported image extension: {}".format(root))
        return [root]
    return sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: (p.relative_to(root).as_posix().casefold(), p.relative_to(root).as_posix()),
    )


def _check_image(array: np.ndarray, source: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2 or min(array.shape, default=0) == 0:
        raise ValueError("Expected one nonempty 2D grayscale image, got {}: {}".format(array.shape, source))
    if array.dtype.kind not in "buif":
        raise ValueError("Expected real numeric pixels, got {}: {}".format(array.dtype, source))
    # uint16 pixel values are exactly representable in float32. The conversion
    # deliberately retains their original numeric scale, not an 8-bit preview.
    with np.errstate(over="ignore", invalid="ignore"):
        result = array.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite or float32-overflow pixel values: {}".format(source))
    return result


def load_image(path: PathLike) -> np.ndarray:
    """Load a single grayscale PNG/TIFF/NPY as raw-scale float32.

    RGB/paletted images and frame stacks are rejected rather than silently
    taking a channel, averaging channels, or choosing the first frame.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(str(path), allow_pickle=False)
    elif suffix in {".tif", ".tiff"}:
        import tifffile

        with tifffile.TiffFile(str(path)) as tif:
            if len(tif.pages) != 1 or len(tif.series) != 1:
                raise ValueError("Multi-frame/multi-series TIFF is not supported; export each frame separately: {}".format(path))
            page = tif.pages[0]
            if page.samplesperpixel != 1 or int(page.photometric) not in {0, 1}:
                raise ValueError("Expected grayscale TIFF, not RGB/paletted/multichannel data: {}".format(path))
            array = tif.asarray()
    elif suffix == ".png":
        with Image.open(str(path)) as im:
            if getattr(im, "n_frames", 1) != 1:
                raise ValueError("Multi-frame image is not supported: {}".format(path))
            if im.mode not in {"1", "L", "I", "I;16", "I;16B", "I;16L", "F"}:
                raise ValueError("Expected grayscale PNG, got mode {}: {}".format(im.mode, path))
            array = np.asarray(im)
    else:
        raise ValueError("Unsupported image extension: {}".format(path))
    return _check_image(array, str(path))


def normalize_image(image: np.ndarray, offset: float = 0.0, scale: float = 65535.0,
                    clip: bool = False) -> np.ndarray:
    """Apply the same ``(image - offset) / scale`` mapping to every frame.

    ``clip=False`` preserves out-of-range values so data mistakes are visible.
    Choose the camera scale in the config; never estimate it independently for
    two frames in a self-supervised pair.
    """
    if not np.isfinite(offset) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Normalization requires a finite offset and a positive finite scale")
    result = (_check_image(image, "normalization input") - float(offset)) / float(scale)
    if clip:
        result = np.clip(result, 0.0, 1.0)
    return result.astype(np.float32, copy=False)


def denormalize_image(image: np.ndarray, offset: float = 0.0, scale: float = 65535.0) -> np.ndarray:
    """Invert fixed normalization without clipping, quantizing, or uint wrapping."""
    if not np.isfinite(offset) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Normalization requires a finite offset and a positive finite scale")
    return (_check_image(image, "denormalization input") * float(scale) + float(offset)).astype(np.float32)


def find_groups(root: PathLike) -> Dict[str, List[Path]]:
    """Group images by their relative parent directory, without guessing scenes.

    ``group_001/frame_000.png`` and ``group_001/frame_001.png`` are one group.
    Files directly in ``root`` belong to ``__root__``. Directory grouping is only
    an organization convention: it does NOT establish that frames are aligned,
    independent-noise observations of the same scene. The user must ensure this
    for repeated-frame diffusion training.
    """
    root = Path(root)
    files = list_images(root)
    if root.is_file():
        return {"__root__": files}
    groups = {}  # type: Dict[str, List[Path]]
    for path in files:
        parent = path.parent.relative_to(root).as_posix()
        key = "__root__" if parent == "." else parent
        groups.setdefault(key, []).append(path)
    return groups
