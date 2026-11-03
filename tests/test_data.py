"""Image integrity checks: raw scientific measurements must survive loading."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from ionrecog.data import (
    denormalize_image,
    find_groups,
    list_images,
    load_image,
    normalize_image,
)


@pytest.mark.parametrize("extension", ["png", "tif", "npy"])
def test_uint16_values_survive_loading(tmp_path, extension):
    values = np.array([[0, 1, 255], [256, 32768, 65535]], dtype=np.uint16)
    path = tmp_path / ("raw." + extension)
    if extension == "npy":
        np.save(path, values)
    elif extension == "tif":
        tifffile.imwrite(path, values)
    else:
        Image.fromarray(values).save(path)

    image = load_image(path)

    assert image.dtype == np.float32
    np.testing.assert_array_equal(image, values)


def test_fixed_normalization_preserves_brightness_between_frames():
    dim = np.full((4, 4), 1500, dtype=np.float32)
    bright = np.full((4, 4), 2500, dtype=np.float32)
    normalized_dim = normalize_image(dim, offset=500, scale=4000)
    normalized_bright = normalize_image(bright, offset=500, scale=4000)
    np.testing.assert_allclose(normalized_dim, 0.25)
    np.testing.assert_allclose(normalized_bright, 0.5)
    np.testing.assert_array_equal(
        denormalize_image(normalized_dim, offset=500, scale=4000), dim
    )


def test_normalization_does_not_silently_clip_out_of_range_measurements():
    image = np.array([[-10, 100]], dtype=np.float32)
    result = normalize_image(image, offset=0, scale=50)
    np.testing.assert_allclose(result, [[-0.2, 2.0]])


def test_reject_rgb_instead_of_silently_selecting_channel(tmp_path):
    path = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((8, 9, 3), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError):
        load_image(path)


def test_reject_tiff_stack_instead_of_silently_selecting_frame(tmp_path):
    path = tmp_path / "stack.tif"
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.zeros((8, 9), dtype=np.uint16))
        writer.write(np.ones((8, 9), dtype=np.uint16))
    with pytest.raises(ValueError):
        load_image(path)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_reject_nonfinite_data(tmp_path, bad):
    values = np.zeros((8, 9), dtype=np.float32)
    values[2, 3] = bad
    path = tmp_path / "invalid.npy"
    np.save(path, values)
    with pytest.raises(ValueError):
        load_image(path)


def test_discovery_is_recursive_sorted_and_groups_by_parent(tmp_path):
    for filename in ["b/02.npy", "a/02.npy", "a/01.npy"]:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.zeros((4, 5), dtype=np.uint16))
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    files = list_images(tmp_path)
    relative = [Path(p).relative_to(tmp_path).as_posix() for p in files]
    assert relative == ["a/01.npy", "a/02.npy", "b/02.npy"]
    groups = find_groups(tmp_path)
    assert set(groups) == {"a", "b"}
    assert len(groups["a"]) == 2
    assert len(groups["b"]) == 1
