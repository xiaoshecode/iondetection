"""Prevent train/test leakage and incorrectly optimistic localisation metrics."""

import numpy as np
import pytest
from PIL import Image

from ionrecog.evaluation import match_points
from ionrecog.validation import validate_splits


def write_npy(path, value, shape=(8, 9)):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.full(shape, value, dtype=np.uint16))


def test_validation_rejects_nested_directories(tmp_path):
    with pytest.raises(ValueError, match="non-nested"):
        validate_splits(tmp_path / "data", tmp_path / "data" / "val")


def test_same_pixels_in_different_formats_are_detected_as_leakage(tmp_path):
    pixels = np.arange(72, dtype=np.uint16).reshape(8, 9)
    train = tmp_path / "train" / "session_a"
    val = tmp_path / "val" / "session_b"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    Image.fromarray(pixels).save(train / "frame.png")
    np.save(val / "renamed.npy", pixels)
    with pytest.raises(ValueError, match="Duplicate image across"):
        validate_splits(train.parent, val.parent)


def test_same_acquisition_cannot_be_split_across_train_and_validation(tmp_path):
    write_npy(tmp_path / "train" / "session_01" / "a.npy", 100)
    write_npy(tmp_path / "val" / "session_01" / "b.npy", 101)
    with pytest.raises(ValueError, match="Same acquisition group"):
        validate_splits(tmp_path / "train", tmp_path / "val")


def test_diffusion_rejects_groups_without_enough_repeats(tmp_path):
    for split, group, offset in [("train", "a", 100), ("val", "b", 200)]:
        for i in range(2):
            write_npy(tmp_path / split / group / (str(i) + ".npy"), offset + i)
    with pytest.raises(ValueError, match=">=3"):
        validate_splits(tmp_path / "train", tmp_path / "val", method="diffusion")


def test_valid_groups_keep_entire_acquisitions_separate(tmp_path):
    for split, group, offset in [("train", "a", 100), ("val", "b", 200)]:
        for i in range(3):
            write_npy(tmp_path / split / group / (str(i) + ".npy"), offset + i)
    report = validate_splits(tmp_path / "train", tmp_path / "val", method="diffusion")
    assert report["train"]["images"] == report["val"]["images"] == 3
    assert set(report["train"]["groups"]).isdisjoint(report["val"]["groups"])


def test_matching_prioritizes_number_of_valid_matches_over_total_distance():
    # Minimum total unconstrained distance chooses one invalid long diagonal.
    # Threshold-aware matching must instead choose the two valid cross-matches.
    predicted = [[0, 0], [0, 2]]
    truth = [[0, 0.2], [1.8, 0]]
    result = match_points(predicted, truth, tolerance=2)
    assert (result["tp"], result["fp"], result["fn"]) == (2, 0, 0)
    assert result["mean_error_px"] == pytest.approx(1.8)
    assert result["all_correct"]


def test_duplicate_predictions_cannot_match_one_ion_twice():
    result = match_points([[5, 5], [5.1, 5]], [[5, 5]], tolerance=1)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)
    assert result["precision"] == 0.5
    assert not result["all_correct"]


def test_empty_predictions_are_reported_as_missed_ions():
    result = match_points([], [[4, 5]], tolerance=1)
    assert result["fn"] == 1
    assert result["recall"] == 0
    assert result["mean_error_px"] is None


def test_empty_ground_truth_still_penalizes_false_detections():
    result = match_points([[4, 5]], [], tolerance=1)
    assert result["fp"] == 1
    assert result["precision"] == 0
    assert not result["all_correct"]
