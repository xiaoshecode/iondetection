"""CPU tests for real hook protocol, image geometry, and offline correspondence.

No model download, mock inference only. These tests do not establish FLUX model
quality, CUDA memory requirements or compatibility beyond the checked protocol.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ionrecog.diffusion import features


def _ids(h, w):
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    ids = torch.stack((torch.zeros_like(yy), yy, xx), dim=-1).reshape(-1, 3)
    reference = ids.clone()
    reference[:, 0] = 1
    return torch.cat((ids, reference)).float()


class _Block(torch.nn.Module):
    def forward(self, hidden_states, encoder_hidden_states):
        return encoder_hidden_states + 5000, hidden_states + 1


class _Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.transformer_blocks = torch.nn.ModuleList([_Block(), _Block()])

    def forward(self, hidden_states, img_ids, timestep, encoder_hidden_states):
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(hidden_states, encoder_hidden_states)
        return hidden_states


class _Pipeline:
    def __init__(self):
        self.transformer = _Transformer()
        self.vae = torch.nn.Linear(1, 1)
        self.text_encoder = torch.nn.Linear(1, 1)
        self.text_encoder_2 = torch.nn.Linear(1, 1)
        self._ion_probe_diffusers_version = "mock-0.35.1"
        self.call_kwargs = None

    def to(self, device):
        self.device = device
        return self

    def __call__(self, image, prompt, height, width, max_area, _auto_resize,
                 num_inference_steps, guidance_scale, true_cfg_scale,
                 num_images_per_prompt, max_sequence_length, output_type, generator):
        assert not _auto_resize and output_type == "latent"
        assert max_area == height * width
        assert image.dtype == torch.float32 and image.shape == (1, 3, height, width)
        assert 0 <= image.min() <= image.max() <= 1
        self.call_kwargs = dict(seed=generator.initial_seed(), output_type=output_type)
        ids = _ids(height // 16, width // 16)
        for i in range(num_inference_steps):
            hidden = ids[:, 0:1].unsqueeze(0).repeat(1, 1, 3) * 100 + i * 10
            self.transformer(hidden_states=hidden, img_ids=ids,
                             timestep=torch.tensor([1.0 - i * 0.1]),
                             encoder_hidden_states=torch.ones(1, 7, 3) * -999)


def test_hooks_capture_selected_step_and_remove_even_after_exception():
    model = _Transformer()
    image_ids = _ids(2, 3)
    with features.FluxFeatureCapture(model, layers=(0, 1), capture_step=1) as capture:
        for i in range(3):
            model(torch.ones(1, 12, 4) * i, image_ids, torch.tensor([i / 3]), torch.zeros(1, 9, 4))
    np.testing.assert_array_equal(capture.features[0], np.ones((1, 12, 4)) * 2)
    np.testing.assert_array_equal(capture.features[1], np.ones((1, 12, 4)) * 3)
    assert not model._forward_pre_hooks and not model.transformer_blocks[0]._forward_hooks
    with pytest.raises(RuntimeError, match="intentional"):
        with features.FluxFeatureCapture(model):
            raise RuntimeError("intentional")
    assert not model._forward_pre_hooks and not model.transformer_blocks[0]._forward_hooks


def test_spatial_split_uses_ids_and_preserves_pixel_center_mapping():
    ids = _ids(2, 3).numpy()
    values = np.arange(12, dtype=np.float32).reshape(1, 12, 1)
    # Nontrivial ordering proves we do not assume a flat half/half grid reshape.
    order = np.array([11, 0, 6, 2, 7, 5, 10, 1, 4, 3, 8, 9])
    split = features.split_spatial_tokens(values[:, order], ids[order], (20, 60))
    np.testing.assert_array_equal(split["reference"]["features"].reshape(-1), np.arange(6, 12))
    np.testing.assert_array_equal(split["target"]["features"].reshape(-1), np.arange(6))
    np.testing.assert_allclose(split["reference"]["xy"][[0, -1]], [[9.5, 4.5], [49.5, 14.5]])


def test_spatial_split_rejects_text_or_incomplete_grid():
    ids = _ids(2, 2)
    with pytest.raises(ValueError, match="matching image IDs"):
        features.split_spatial_tokens(np.zeros((1, 10, 3)), ids, (32, 32))
    ids[1] = ids[0]
    with pytest.raises(ValueError, match="complete rectangular"):
        features.split_spatial_tokens(np.zeros((1, 8, 3)), ids, (32, 32))


def test_mock_extraction_end_to_end_is_local_frozen_and_geometric(tmp_path, monkeypatch):
    image_path = tmp_path / "input.npy"
    np.save(image_path, np.arange(20 * 40, dtype=np.uint16).reshape(20, 40))
    pipe = _Pipeline()
    loaded = {}
    def loader(model_path, dtype, local_files_only):
        loaded.update(model_path=str(model_path), dtype=dtype, local=local_files_only)
        return pipe
    monkeypatch.setattr(features, "_load_pipeline", loader)
    result = features.extract_features(image_path, tmp_path / "probe", "/local/FLUX", device="cpu",
                                       precision="fp32", layers=(0, 1), steps=3, capture_step=1)
    metadata = json.loads(result.read_text(encoding="utf-8"))
    assert loaded["local"] is True
    assert metadata["image"]["original_hw"] == [20, 40]
    assert metadata["image"]["processed_hw"] == [32, 48]
    assert metadata["image"]["intensity_scale"] == 65535
    assert metadata["grid_hw"]["reference"] == [2, 3]
    assert pipe.call_kwargs["seed"] == 42
    assert not any(p.requires_grad for p in pipe.transformer.parameters())
    assert not any(p.requires_grad for p in pipe.vae.parameters())
    with np.load(result.parent / "features.npz", allow_pickle=False) as data:
        assert data["reference_layer_0"].shape == (2, 3, 3)
        np.testing.assert_array_equal(data["reference_layer_0"], np.full((2, 3, 3), 111))
        np.testing.assert_array_equal(data["target_layer_0"], np.full((2, 3, 3), 11))
        assert "frozen_editing_trajectory" in str(data["metadata_json"])
    with pytest.raises(FileExistsError):
        features.extract_features(image_path, result.parent, "/local/FLUX", device="cpu", precision="fp32")


def _save_artifact(path, vectors):
    arr = np.asarray(vectors, dtype=np.float32)
    n = len(arr)
    metadata = {"schema": features.SCHEMA, "model_path": "same-model", "backend": "flux_kontext"}
    np.savez(path, reference_layer_0=arr.reshape(1, n, -1),
             reference_xy=np.stack((np.arange(n) * 10, np.zeros(n)), axis=1),
             metadata_json=np.array(json.dumps(metadata)))


def test_cosine_matching_and_one_to_one_are_token_coordinates(tmp_path):
    source, target = tmp_path / "a.npz", tmp_path / "b.npz"
    _save_artifact(source, [[1, 0], [0, 1], [0, 0]])
    _save_artifact(target, [[0, 1], [1, 0]])
    result = features.compare_features(source, target, tmp_path / "matches", one_to_one=True, mutual=True)
    metadata = json.loads(result.read_text())
    assert [(x["source_index"], x["target_index"]) for x in metadata["matches"]] == [(0, 1), (1, 0)]
    assert metadata["matches"][0]["target_xy"] == [10.0, 0.0]
    assert metadata["invalid_zero_source_tokens"] == 1
    np.testing.assert_allclose(np.load(result.parent / "cosine.npy"), [[0, 1], [1, 0], [0, 0]])
    assert "NOT ion detections" in metadata["caveat"]


def test_matching_empty_zero_vectors_and_matrix_guard(tmp_path):
    source, target = tmp_path / "a.npz", tmp_path / "b.npz"
    _save_artifact(source, [[0, 0], [0, 0]])
    _save_artifact(target, [[0, 0], [0, 0]])
    for one_to_one in (False, True):
        result = features.compare_features(source, target, tmp_path / str(one_to_one), one_to_one=one_to_one)
        assert not json.loads(result.read_text())["matches"]
    with pytest.raises(ValueError, match="limit"):
        features.compare_features(source, target, tmp_path / "guard", max_matrix_entries=3)


def test_channel_standardization_is_explicit_and_float_scaling_required(tmp_path):
    source, target = tmp_path / "a.npz", tmp_path / "b.npz"
    _save_artifact(source, [[1000, 2], [1000, -2]])
    _save_artifact(target, [[1000, -2], [1000, 2]])
    result = features.compare_features(source, target, tmp_path / "normalized", normalization="channel_standardize")
    metadata = json.loads(result.read_text())
    assert "not a DiTF" in metadata["normalization_note"]
    np.testing.assert_allclose(np.load(result.parent / "cosine.npy"), [[-1, 1], [1, -1]])
    image = tmp_path / "float.npy"
    np.save(image, np.ones((16, 16), dtype=np.float32) * 500)
    with pytest.raises(ValueError, match="intensity_scale"):
        features._read_probe_image(image, None, None, 0, None)
