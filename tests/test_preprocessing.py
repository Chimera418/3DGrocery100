"""
tests/test_preprocessing.py
============================
Unit tests for common/preprocessing.py

Run with:
  cd <project_root>
  pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow import from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.preprocessing import (
    N_POINTS,
    clean_points,
    normalize_xyz,
    sample_points,
    select_channels,
    validate_points,
    process_sample,
    set_global_seed,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cloud_1024_xyzrgb() -> np.ndarray:
    """Valid (1024, 6) float32 array."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.0, 1.0, (1024, 6)).astype(np.float32)
    pts[:, 3:6] = rng.uniform(0.0, 1.0, (1024, 3)).astype(np.float32)
    return pts


@pytest.fixture
def cloud_2000_xyzrgb() -> np.ndarray:
    """Oversized (2000, 6) cloud."""
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1.0, 1.0, (2000, 6)).astype(np.float32)
    pts[:, 3:6] = np.clip(pts[:, 3:6], 0, 1)
    return pts


@pytest.fixture
def cloud_500_xyzrgb() -> np.ndarray:
    """Undersized (500, 6) cloud."""
    rng = np.random.default_rng(2)
    pts = rng.uniform(-1.0, 1.0, (500, 6)).astype(np.float32)
    pts[:, 3:6] = np.clip(pts[:, 3:6], 0, 1)
    return pts


# ---------------------------------------------------------------------------
# 1. Label mapping (validate_points)
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_sample(self, cloud_1024_xyzrgb):
        result = validate_points(cloud_1024_xyzrgb, label=5)
        assert result["status"] == "VALID"
        assert result["issues"] == []

    def test_label_out_of_range_rejects(self, cloud_1024_xyzrgb):
        result = validate_points(cloud_1024_xyzrgb, label=100)
        assert result["status"] == "REJECTED"
        assert any("LABEL_OUT_OF_RANGE" in i for i in result["issues"])

    def test_negative_label_rejects(self, cloud_1024_xyzrgb):
        result = validate_points(cloud_1024_xyzrgb, label=-1)
        assert result["status"] == "REJECTED"

    def test_nan_xyz_rejects(self):
        pts = np.ones((1024, 6), dtype=np.float32)
        pts[0, 0] = float("nan")
        result = validate_points(pts, label=0)
        assert result["status"] == "REJECTED"
        assert any("NAN_XYZ" in i for i in result["issues"])

    def test_inf_xyz_rejects(self):
        pts = np.ones((1024, 6), dtype=np.float32)
        pts[5, 1] = float("inf")
        result = validate_points(pts, label=0)
        assert result["status"] == "REJECTED"

    def test_empty_array_rejects(self):
        pts = np.zeros((0, 6), dtype=np.float32)
        result = validate_points(pts, label=0)
        assert result["status"] == "REJECTED"

    def test_too_few_channels_rejects(self):
        pts = np.ones((1024, 2), dtype=np.float32)
        result = validate_points(pts, label=0)
        assert result["status"] == "REJECTED"

    def test_wrong_point_count_warns(self):
        pts = np.ones((512, 6), dtype=np.float32)
        result = validate_points(pts, label=0)
        assert result["status"] == "WARNING"
        assert any("POINT_COUNT_MISMATCH" in i for i in result["issues"])

    def test_require_rgb_missing_warns(self, cloud_1024_xyzrgb):
        pts = cloud_1024_xyzrgb[:, :3]  # only XYZ
        result = validate_points(pts, label=0, require_rgb=True)
        assert result["status"] == "REJECTED"

    def test_boundary_labels(self, cloud_1024_xyzrgb):
        for lbl in (0, 99):
            result = validate_points(cloud_1024_xyzrgb, label=lbl)
            assert result["status"] == "VALID"


# ---------------------------------------------------------------------------
# 2. Sampling output shape
# ---------------------------------------------------------------------------

class TestSampling:
    def test_exact_count_returns_same(self, cloud_1024_xyzrgb):
        sampled = sample_points(cloud_1024_xyzrgb, n=1024)
        assert sampled.shape == (1024, 6)

    def test_downsample_shape(self, cloud_2000_xyzrgb):
        sampled = sample_points(cloud_2000_xyzrgb, n=1024, seed=42)
        assert sampled.shape == (1024, 6)

    def test_upsample_shape(self, cloud_500_xyzrgb):
        sampled = sample_points(cloud_500_xyzrgb, n=1024, seed=42)
        assert sampled.shape == (1024, 6)

    def test_downsample_deterministic(self, cloud_2000_xyzrgb):
        a = sample_points(cloud_2000_xyzrgb, n=1024, seed=42)
        b = sample_points(cloud_2000_xyzrgb, n=1024, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_upsample_deterministic(self, cloud_500_xyzrgb):
        a = sample_points(cloud_500_xyzrgb, n=1024, seed=7)
        b = sample_points(cloud_500_xyzrgb, n=1024, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_output_dtype_float32(self, cloud_2000_xyzrgb):
        sampled = sample_points(cloud_2000_xyzrgb, n=1024, seed=0)
        assert sampled.dtype == np.float32


# ---------------------------------------------------------------------------
# 3. Normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_output_shape_preserved(self, cloud_1024_xyzrgb):
        normed = normalize_xyz(cloud_1024_xyzrgb)
        assert normed.shape == cloud_1024_xyzrgb.shape

    def test_centroid_near_zero(self, cloud_1024_xyzrgb):
        normed = normalize_xyz(cloud_1024_xyzrgb)
        centroid = normed[:, :3].mean(axis=0)
        np.testing.assert_allclose(centroid, np.zeros(3), atol=1e-5)

    def test_max_dist_equals_one(self, cloud_1024_xyzrgb):
        normed = normalize_xyz(cloud_1024_xyzrgb)
        xyz = normed[:, :3]
        max_dist = np.sqrt((xyz ** 2).sum(axis=1)).max()
        assert abs(max_dist - 1.0) < 1e-5

    def test_rgb_unchanged(self, cloud_1024_xyzrgb):
        original_rgb = cloud_1024_xyzrgb[:, 3:].copy()
        normed = normalize_xyz(cloud_1024_xyzrgb)
        np.testing.assert_array_equal(normed[:, 3:], original_rgb)

    def test_idempotent_on_unit_sphere(self, cloud_1024_xyzrgb):
        normed1 = normalize_xyz(cloud_1024_xyzrgb)
        normed2 = normalize_xyz(normed1)
        np.testing.assert_allclose(normed1[:, :3], normed2[:, :3], atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Channel selection (XYZ vs XYZRGB mode)
# ---------------------------------------------------------------------------

class TestChannelSelection:
    def test_xyz_mode_returns_3_channels(self, cloud_1024_xyzrgb):
        out = select_channels(cloud_1024_xyzrgb, mode="xyz")
        assert out.shape == (1024, 3)

    def test_xyzrgb_mode_returns_6_channels(self, cloud_1024_xyzrgb):
        out = select_channels(cloud_1024_xyzrgb, mode="xyzrgb")
        assert out.shape == (1024, 6)

    def test_xyz_values_match_original(self, cloud_1024_xyzrgb):
        out = select_channels(cloud_1024_xyzrgb, mode="xyz")
        np.testing.assert_array_equal(out, cloud_1024_xyzrgb[:, :3])

    def test_xyzrgb_values_match_original(self, cloud_1024_xyzrgb):
        out = select_channels(cloud_1024_xyzrgb, mode="xyzrgb")
        np.testing.assert_array_equal(out, cloud_1024_xyzrgb[:, :6])

    def test_xyzrgb_on_3channel_raises(self):
        pts = np.ones((1024, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="xyzrgb"):
            select_channels(pts, mode="xyzrgb")

    def test_unknown_mode_raises(self, cloud_1024_xyzrgb):
        with pytest.raises(ValueError, match="Unknown mode"):
            select_channels(cloud_1024_xyzrgb, mode="xyznormal")


# ---------------------------------------------------------------------------
# 5. Cleaning
# ---------------------------------------------------------------------------

class TestCleaning:
    def test_clean_valid_unchanged(self, cloud_1024_xyzrgb):
        cleaned = clean_points(cloud_1024_xyzrgb)
        assert cleaned.shape == cloud_1024_xyzrgb.shape

    def test_clean_removes_nan_rows(self):
        pts = np.ones((100, 6), dtype=np.float32)
        pts[10, 0] = float("nan")
        pts[50, 2] = float("nan")
        cleaned = clean_points(pts)
        assert cleaned.shape[0] == 98

    def test_clean_clamps_rgb(self):
        pts = np.ones((10, 6), dtype=np.float32)
        pts[:, 3] = 1.5   # out of range
        pts[:, 4] = -0.3  # out of range
        cleaned = clean_points(pts)
        assert cleaned[:, 3].max() <= 1.0
        assert cleaned[:, 4].min() >= 0.0

    def test_clean_all_bad_raises(self):
        pts = np.full((10, 6), float("nan"), dtype=np.float32)
        with pytest.raises(ValueError, match="All points removed"):
            clean_points(pts)


# ---------------------------------------------------------------------------
# 6. End-to-end process_sample
# ---------------------------------------------------------------------------

class TestProcessSample:
    def test_valid_xyz_output_shape(self, cloud_1024_xyzrgb):
        pts, lbl, rpt = process_sample(
            cloud_1024_xyzrgb, label=7, mode="xyz", n_points=1024
        )
        assert pts is not None
        assert pts.shape == (1024, 3)
        assert lbl == 7
        assert rpt["status"] in ("VALID", "WARNING")

    def test_valid_xyzrgb_output_shape(self, cloud_1024_xyzrgb):
        pts, lbl, rpt = process_sample(
            cloud_1024_xyzrgb, label=7, mode="xyzrgb", n_points=1024
        )
        assert pts is not None
        assert pts.shape == (1024, 6)

    def test_rejected_returns_none(self):
        bad_pts = np.full((1024, 6), float("nan"), dtype=np.float32)
        pts, lbl, rpt = process_sample(bad_pts, label=0)
        assert pts is None
        assert lbl is None
        assert rpt["status"] == "REJECTED"

    def test_normalization_applied(self, cloud_1024_xyzrgb):
        pts, _, _ = process_sample(
            cloud_1024_xyzrgb, label=0, mode="xyz",
            apply_normalization=True
        )
        centroid = pts[:, :3].mean(axis=0)
        np.testing.assert_allclose(centroid, np.zeros(3), atol=1e-4)


# ---------------------------------------------------------------------------
# 7. Seed reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_set_global_seed_idempotent(self, cloud_2000_xyzrgb):
        set_global_seed(42)
        a = sample_points(cloud_2000_xyzrgb, n=1024, seed=42)
        set_global_seed(99)
        b = sample_points(cloud_2000_xyzrgb, n=1024, seed=42)
        np.testing.assert_array_equal(a, b)
