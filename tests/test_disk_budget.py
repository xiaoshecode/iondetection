"""Output-space preflight protects a full scientific acquisition run."""
from types import SimpleNamespace

import numpy as np
import pytest

from ionrecog import cli


def test_low_disk_cli_fails_before_creating_result_directory(tmp_path, monkeypatch):
    source = tmp_path / "frame.npy"
    np.save(source, np.full((32, 32), 500, dtype=np.uint16))
    output = tmp_path / "new_results"
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda _: SimpleNamespace(free=1024**3))

    with pytest.raises(SystemExit) as error:
        cli.main(["infer", "--method", "classical", "--input", str(source), "--output", str(output)])

    assert error.value.code == 2
    assert not output.exists()
    np.testing.assert_array_equal(np.load(source), np.full((32, 32), 500, dtype=np.uint16))


def test_sufficient_space_produces_a_reviewable_budget_without_writes(tmp_path, monkeypatch):
    source = tmp_path / "frame.npy"
    np.save(source, np.full((32, 32), 500, dtype=np.uint16))
    output = tmp_path / "new_results"
    available = 20 * 1024**3
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda _: SimpleNamespace(free=available))

    budget = cli.check_disk_budget(output, [source], learned=True, reserve_gib=2)

    assert budget["free_before_bytes"] == available
    assert budget["reserve_bytes"] == 2 * 1024**3
    assert source.stat().st_size < budget["estimated_output_bytes"] < available - budget["reserve_bytes"]
    assert not output.exists()
