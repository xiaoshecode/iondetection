"""CLI protections for experimental input data and acquisition groups."""
import json

import numpy as np
import pytest

from ionrecog.cli import check_destination, main


def test_output_cannot_overlap_dataset_in_either_direction(tmp_path):
    source = tmp_path / "dataset" / "session"
    source.mkdir(parents=True)
    for output in (source, source / "results", source.parent):
        with pytest.raises(ValueError):
            check_destination(output, (source,))
    assert check_destination(tmp_path / "results", (source,)) == tmp_path / "results"


def test_evaluate_refuses_overwriting_ground_truth(tmp_path):
    truth = tmp_path / "truth.json"
    prediction = tmp_path / "prediction.json"
    original = "[[3, 4]]"
    truth.write_text(original)
    prediction.write_text(json.dumps({"ions": [{"x": 3, "y": 4}]}))
    with pytest.raises(SystemExit) as error:
        main(["evaluate", "--prediction", str(prediction), "--truth", str(truth), "--output", str(truth)])
    assert error.value.code == 2
    assert truth.read_text() == original


def test_reference_refuses_averaging_different_acquisitions(tmp_path):
    source = tmp_path / "reference"
    for name in ("session_1", "session_2"):
        group = source / name
        group.mkdir(parents=True)
        np.save(group / "frame.npy", np.ones((16, 16), dtype=np.uint16))
    output = tmp_path / "reference.json"
    with pytest.raises(SystemExit) as error:
        main(["reference", "--input", str(source), "--output", str(output), "--confirm-static"])
    assert error.value.code == 2
    assert not output.exists()
