"""
tests/test_dataset.py
======================
Unit tests for common/dataset.py (Grocery3DDataset and make_dataloader).

These tests require the processed data to already exist:
  - data/processed/labels.json
  - data/processed/splits/train.json
  - data/processed/splits/test.json
  - Raw .dat files at configs/data_config.yaml path

Run with:
  cd <project_root>
  pytest tests/test_dataset.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.dataset import Grocery3DDataset, make_dataloader, build_id_name_maps
from common.preprocessing import N_POINTS

LABELS_JSON = "data/processed/labels.json"
SPLITS_DIR = "data/processed/splits"


def labels_json_exists():
    return Path(LABELS_JSON).exists()


def splits_exist():
    return (
        Path(SPLITS_DIR, "train.json").exists()
        and Path(SPLITS_DIR, "test.json").exists()
    )


# Skip all tests if processed data not available yet
pytestmark = pytest.mark.skipif(
    not (labels_json_exists() and splits_exist()),
    reason="Run prepare_dataset.py first to generate processed data",
)


# ---------------------------------------------------------------------------
# Dataset output shape tests
# ---------------------------------------------------------------------------

class TestDatasetShape:
    @pytest.fixture(scope="class")
    def train_xyz(self):
        return Grocery3DDataset(split="train", mode="xyz")

    @pytest.fixture(scope="class")
    def train_rgb(self):
        return Grocery3DDataset(split="train", mode="xyzrgb")

    @pytest.fixture(scope="class")
    def test_xyz(self):
        return Grocery3DDataset(split="test", mode="xyz")

    def test_train_xyz_len(self, train_xyz):
        assert len(train_xyz) == 66032

    def test_test_xyz_len(self, test_xyz):
        assert len(test_xyz) == 21866

    def test_xyz_point_shape(self, train_xyz):
        pts, lbl = train_xyz[0]
        assert pts.shape == torch.Size([N_POINTS, 3])

    def test_xyzrgb_point_shape(self, train_rgb):
        pts, lbl = train_rgb[0]
        assert pts.shape == torch.Size([N_POINTS, 6])

    def test_label_type(self, train_xyz):
        pts, lbl = train_xyz[0]
        assert isinstance(lbl, int)

    def test_label_range(self, train_xyz):
        for i in range(0, len(train_xyz), 5000):
            _, lbl = train_xyz[i]
            assert 0 <= lbl <= 99

    def test_tensor_dtype(self, train_xyz):
        pts, _ = train_xyz[0]
        assert pts.dtype == torch.float32

    def test_xyz_normalization(self, train_xyz):
        pts, _ = train_xyz[0]
        xyz = pts.numpy()
        centroid = xyz.mean(axis=0)
        import numpy as np
        np.testing.assert_allclose(centroid, [0, 0, 0], atol=1e-4)

    def test_invalid_split_raises(self):
        with pytest.raises(ValueError, match="split"):
            Grocery3DDataset(split="val")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            Grocery3DDataset(split="train", mode="normal")


# ---------------------------------------------------------------------------
# Metadata mode
# ---------------------------------------------------------------------------

class TestMetadataMode:
    def test_metadata_returns_three_values(self):
        ds = Grocery3DDataset(split="train", mode="xyz", metadata=True)
        result = ds[0]
        assert len(result) == 3
        pts, lbl, sample_id = result
        assert isinstance(sample_id, str)
        assert "=" in sample_id

    def test_no_metadata_returns_two_values(self):
        ds = Grocery3DDataset(split="train", mode="xyz", metadata=False)
        result = ds[0]
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Label map helpers
# ---------------------------------------------------------------------------

class TestLabelMap:
    def test_id_to_label_length(self):
        id_to_label, label_to_id = build_id_name_maps(LABELS_JSON)
        assert len(id_to_label) == 100

    def test_label_to_id_length(self):
        id_to_label, label_to_id = build_id_name_maps(LABELS_JSON)
        assert len(label_to_id) == 100

    def test_bijection(self):
        id_to_label, label_to_id = build_id_name_maps(LABELS_JSON)
        for cls_id, name in id_to_label.items():
            assert label_to_id[name] == cls_id

    def test_all_ids_in_range(self):
        id_to_label, _ = build_id_name_maps(LABELS_JSON)
        for cls_id in id_to_label:
            assert 0 <= cls_id <= 99

    def test_get_class_name(self):
        ds = Grocery3DDataset(split="train", mode="xyz")
        name = ds.get_class_name(4)
        assert isinstance(name, str)
        assert len(name) > 0

    def test_get_class_id_roundtrip(self):
        ds = Grocery3DDataset(split="train", mode="xyz")
        for cls_id in range(100):
            name = ds.get_class_name(cls_id)
            recovered_id = ds.get_class_id(name)
            assert recovered_id == cls_id


# ---------------------------------------------------------------------------
# DataLoader batch shape
# ---------------------------------------------------------------------------

class TestDataLoader:
    def test_train_batch_xyz_shape(self):
        ds = Grocery3DDataset(split="train", mode="xyz")
        loader = make_dataloader(ds, batch_size=8, num_workers=0, shuffle=False)
        batch_pts, batch_lbl = next(iter(loader))
        assert batch_pts.shape == torch.Size([8, N_POINTS, 3])
        assert batch_lbl.shape == torch.Size([8])

    def test_train_batch_xyzrgb_shape(self):
        ds = Grocery3DDataset(split="train", mode="xyzrgb")
        loader = make_dataloader(ds, batch_size=4, num_workers=0, shuffle=False)
        batch_pts, batch_lbl = next(iter(loader))
        assert batch_pts.shape == torch.Size([4, N_POINTS, 6])

    def test_batch_label_range(self):
        ds = Grocery3DDataset(split="train", mode="xyz")
        loader = make_dataloader(ds, batch_size=32, num_workers=0, shuffle=False)
        _, batch_lbl = next(iter(loader))
        assert batch_lbl.min() >= 0
        assert batch_lbl.max() <= 99
