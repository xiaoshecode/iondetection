import json

import numpy as np
import pytest

from ionrecog.diffusion.config import load_config
from ionrecog.diffusion.pipeline import evaluate, infer, predict_frame, restoration_metrics, summarize_localization
from ionrecog.diffusion.__main__ import prepare_repeats


def test_metrics_do_not_treat_unlabelled_as_empty():
    assert summarize_localization([])["precision"] is None
    m = restoration_metrics(np.zeros((4, 4)), np.zeros((4, 4)), 100)
    assert m["identical"] and m["psnr_db"] is None
    with pytest.raises(ValueError):
        predict_frame(np.zeros((16, 16), dtype=np.float32), None, load_config(), samples=0)


def test_raw_evaluation_and_unlabelled_records(tmp_path):
    lines = []
    for i, points in enumerate(([], None)):
        image = np.full((16, 16), 10+i, dtype=np.float32)
        np.save(tmp_path / ("input%d.npy" % i), image)
        np.save(tmp_path / ("target%d.npy" % i), image+0.25)
        record = {"id": str(i), "split": "test", "group_id": str(i),
                  "input": "input%d.npy" % i, "target": "target%d.npy" % i,
                  "target_kind": "clean"}
        if points is not None:
            record["points"] = points
        lines.append(json.dumps(record))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(lines), encoding="utf-8")
    result = evaluate(manifest, tmp_path / "out", load_config(), mode="raw", reserve_gib=0)
    report = json.loads(result.read_text())
    assert report["images"] == 2 and report["labelled_images"] == 1
    assert report["raw_localization"]["all_correct_fraction"] == 1
    assert report["restored_localization"]["labelled_predictions"] == 1
    with pytest.raises(FileExistsError):
        evaluate(manifest, tmp_path / "out", load_config(), mode="raw", reserve_gib=0)


def test_repeat_preparation_and_relocatable_paths(tmp_path):
    source = tmp_path / "frames"
    for split in ("train", "val"):
        group = source / split / (split + "_crystal")
        group.mkdir(parents=True)
        for i in range(3):
            rng = np.random.default_rng(i + (100 if split == "val" else 0))
            np.save(group / (str(i) + ".npy"), rng.uniform(10, 11, (16, 16)).astype(np.float32))
    out = tmp_path / "manifest.jsonl"
    with pytest.raises(ValueError, match="confirm-static"):
        prepare_repeats(source, out)
    prepare_repeats(source, out, True)
    row = json.loads(out.read_text().splitlines()[0])
    assert not row["frames"][0].startswith(str(tmp_path))
    assert row["stable"] is True


def test_inference_preserves_distinct_same_stem_images(tmp_path):
    source = tmp_path / "input"
    for name in ("a", "b"):
        folder = source / name
        folder.mkdir(parents=True)
        np.save(folder / "frame.npy", np.ones((16, 16), dtype=np.float32))
    result = infer(source, tmp_path / "out", load_config(), mode="raw", reserve_gib=0)
    rows = json.loads(result.read_text())["images"]
    assert len(rows) == 2 and rows[0]["result"] != rows[1]["result"]


def test_config_rejects_unknown_settings(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("train:\n  epohcs: 10\n")
    with pytest.raises(ValueError, match="Unknown settings"):
        load_config(p)


def test_shipped_config_and_cli_end_to_end(tmp_path):
    from pathlib import Path
    from ionrecog.diffusion.__main__ import main
    project = Path(__file__).resolve().parents[1]
    config = project / "ionrecog/diffusion/configs/smoke.yaml"
    assert load_config(config)["region"]["mode"] == "off"
    assert load_config(project / "ionrecog/diffusion/configs/server.yaml")["region"]["mode"] == "off"
    data, run, result = tmp_path / "data", tmp_path / "run", tmp_path / "result"
    main(["synthetic", "--output", str(data), "--train-groups", "1", "--val-groups", "1",
          "--test-groups", "1", "--size", "32"])
    main(["train", "--manifest", str(data / "manifest.jsonl"), "--config", str(config),
          "--output", str(run), "--device", "cpu"])
    main(["evaluate", "--manifest", str(data / "manifest.jsonl"), "--checkpoint", str(run / "best.pt"),
          "--output", str(result), "--device", "cpu", "--samples", "2"])
    report = json.loads((result / "report.json").read_text())
    assert report["complete"] and report["labelled_images"] == 1
    assert report["restored_localization"]["labelled_predictions"] == 2
