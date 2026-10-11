"""Fast unit tests for ``worker.exominer_native.preprocess.normalize``.

Covers the dict-based replacement for the oracle's TFRecord normalizers:
load_norm_stats, normalize_example, and the inner helpers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from worker.exominer_native.preprocess.normalize import (
    DEFAULT_DIFF_IMG_DATA_SHAPE,
    NORM_SIGMA_EPS,
    centering_and_normalization,
    load_norm_stats,
    normalize_centroid,
    normalize_diff_img,
    normalize_example,
    normalize_scalar_parameters,
    perimage_min_max_normalize_diff_img,
    perimage_std_normalize_diff_img,
)


NORM_STATS_DIR = (
    Path(__file__).resolve().parents[4]
    / "worker" / "exominer_native" / "data" / "norm_stats"
)


# ---------------------------------------------------------------------------
# load_norm_stats
# ---------------------------------------------------------------------------


def test_load_norm_stats_returns_expected_keys():
    stats = load_norm_stats(NORM_STATS_DIR)
    assert "scalar_params" in stats
    assert "centroid" in stats
    assert "diff_img" in stats
    # Spot-check known inner keys
    assert "global_centr_view" in stats["centroid"]
    assert "diff_imgs" in stats["diff_img"]
    assert "boot_fap" in stats["scalar_params"]


def test_load_norm_stats_rejects_missing_dir():
    with pytest.raises(FileNotFoundError):
        load_norm_stats("/tmp/definitely-not-a-dir-2026-04-18")


# ---------------------------------------------------------------------------
# centering_and_normalization  (tiny, but load-bearing — re-used by centroid)
# ---------------------------------------------------------------------------


def test_centering_and_normalization_matches_formula():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    out = centering_and_normalization(v, val_centr=2.0, val_norm=0.5)
    expected = (v - 2.0) / (0.5 + NORM_SIGMA_EPS)
    np.testing.assert_allclose(out, expected, rtol=0, atol=0)


def test_centering_and_normalization_zero_norm_returns_finite():
    # When val_norm == 0 the oracle still returns finite numbers thanks
    # to NORM_SIGMA_EPS; the log-only warning is non-blocking.
    v = np.array([1.0, 2.0])
    out = centering_and_normalization(v, val_centr=0.0, val_norm=0.0)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Scalar parameter normalization
# ---------------------------------------------------------------------------


def _scalar_stats():
    return {
        "boot_fap": {
            "median": -10.0,
            "mad_std": 2.0,
            "info": {
                "missing_value": None,
                "replace_value": None,
                "log_transform": True,
                "log_transform_eps": 1e-32,
                "clip_factor": 20,
                "standardize": True,
                "dtype": "float",
            },
        },
        "tce_depth": {
            "median": 100.0,
            "mad_std": 10.0,
            "info": {
                "missing_value": -1,
                "replace_value": 100.0,
                "log_transform": False,
                "log_transform_eps": float("nan"),
                "clip_factor": 5,
                "standardize": True,
                "dtype": "float",
            },
        },
    }


def test_normalize_scalar_parameters_returns_norm_keys():
    stats = _scalar_stats()
    example = {"boot_fap": 1e-5, "tce_depth": 120.0}
    out = normalize_scalar_parameters(example, stats)
    assert set(out.keys()) == {"boot_fap_norm", "tce_depth_norm"}
    # Values are wrapped in single-element lists
    assert isinstance(out["boot_fap_norm"], list)
    assert np.isfinite(out["boot_fap_norm"][0])
    # Clipped + standardized: should be small magnitude
    assert abs(out["tce_depth_norm"][0] - (120.0 - 100.0) / (10.0 + 1e-32)) < 1e-6


def test_normalize_scalar_parameters_replaces_missing_sentinel():
    stats = _scalar_stats()
    # missing_value = -1 → replace with replace_value (100.0) → standardize → 0
    out = normalize_scalar_parameters({"tce_depth": -1}, stats)
    assert abs(out["tce_depth_norm"][0]) < 1e-6


# ---------------------------------------------------------------------------
# Centroid normalization
# ---------------------------------------------------------------------------


def test_normalize_centroid_shape_and_keys():
    rng = np.random.default_rng(0)
    example = {
        "global_centr_view": rng.uniform(0, 50, size=301),
        "local_centr_view": rng.uniform(0, 50, size=31),
        "local_centr_view_var": rng.uniform(0, 5, size=31),
    }
    centroid_stats = {
        "global_centr_view": {
            "std": 10.0, "std_clip": 10.0, "clip_value": 30,
            "median": 5.0, "median_clip": 5.0,
        },
        "local_centr_view": {
            "std": 12.0, "std_clip": 12.0, "clip_value": 30,
            "median": 5.0, "median_clip": 5.0,
        },
    }
    out = normalize_centroid(example, centroid_stats)
    assert set(out.keys()) == {
        "global_centr_view_std_clip", "local_centr_view_std_clip",
        "global_centr_view_std_noclip", "local_centr_view_std_noclip",
        "local_centr_view_std_noclip_var",
    }
    assert out["global_centr_view_std_clip"].shape == (301,)
    assert out["local_centr_view_std_noclip_var"].shape == (31,)


# ---------------------------------------------------------------------------
# Difference-image normalization (training-set stats)
# ---------------------------------------------------------------------------


def test_normalize_diff_img_produces_std_trainset():
    stats = load_norm_stats(NORM_STATS_DIR)["diff_img"]
    rng = np.random.default_rng(1)
    example = {
        "diff_imgs": rng.uniform(-100, 100, size=DEFAULT_DIFF_IMG_DATA_SHAPE).astype(np.float32),
        "oot_imgs": rng.uniform(0, 1000, size=DEFAULT_DIFF_IMG_DATA_SHAPE).astype(np.float32),
    }
    out = normalize_diff_img(example, stats, imgs_dims=DEFAULT_DIFF_IMG_DATA_SHAPE)
    # Two images → 4 outputs (minmax + std for each)
    assert "diff_imgs_std_trainset" in out
    assert "diff_imgs_minmaxnorm_trainset" in out
    assert "oot_imgs_std_trainset" in out
    assert out["diff_imgs_std_trainset"].shape == DEFAULT_DIFF_IMG_DATA_SHAPE


def test_normalize_diff_img_handles_neighbors_fixed_min_max():
    stats = load_norm_stats(NORM_STATS_DIR)["diff_img"]
    rng = np.random.default_rng(2)
    example = {
        "neighbors_imgs": rng.uniform(0, 5, size=DEFAULT_DIFF_IMG_DATA_SHAPE).astype(np.float32),
    }
    out = normalize_diff_img(example, stats, imgs_dims=DEFAULT_DIFF_IMG_DATA_SHAPE)
    # neighbors_imgs takes the fixed-divisor branch, not min-max
    assert "neighbors_imgs_fixed_min_max_norm" in out
    assert "neighbors_imgs_minmaxnorm_trainset" not in out


# ---------------------------------------------------------------------------
# Per-image diff-img helpers (diagnostic path, not used by ONNX graph)
# ---------------------------------------------------------------------------


def test_perimage_min_max_normalize_diff_img_bounds():
    rng = np.random.default_rng(3)
    imgs = rng.uniform(-10, 10, size=(3, 5, 5)).astype(np.float32)
    out = perimage_min_max_normalize_diff_img(imgs)
    # Each image normalized to [0, ~1]
    for i in range(3):
        assert out[i].min() >= 0.0
        # allow for the tiny zero_division_eps stabiliser
        assert out[i].max() <= 1.0 + 1e-6


def test_perimage_std_normalize_diff_img_shape_preserved():
    rng = np.random.default_rng(4)
    imgs = rng.uniform(-5, 5, size=(2, 4, 4))
    out = perimage_std_normalize_diff_img(imgs)
    assert out.shape == imgs.shape


# ---------------------------------------------------------------------------
# Public entry point: normalize_example
# ---------------------------------------------------------------------------


def test_normalize_example_end_to_end_on_synthetic_features():
    stats = load_norm_stats(NORM_STATS_DIR)
    rng = np.random.default_rng(5)
    features = {
        # Scalar (one known to exist in stats)
        "boot_fap": 1e-6,
        # Centroid views
        "global_centr_view": rng.uniform(0, 50, size=301),
        "local_centr_view": rng.uniform(0, 50, size=31),
        "local_centr_view_var": rng.uniform(0, 5, size=31),
        # Diff images
        "diff_imgs": rng.uniform(-100, 100, size=DEFAULT_DIFF_IMG_DATA_SHAPE).astype(np.float32),
        "oot_imgs": rng.uniform(0, 1000, size=DEFAULT_DIFF_IMG_DATA_SHAPE).astype(np.float32),
    }
    out = normalize_example(features, stats)
    assert isinstance(out, dict)
    # Scalar branch
    assert "boot_fap_norm" in out
    # Centroid branch
    assert "global_centr_view_std_clip" in out
    # Diff-img branch
    assert "diff_imgs_std_trainset" in out
    assert out["diff_imgs_std_trainset"].shape == DEFAULT_DIFF_IMG_DATA_SHAPE


def test_normalize_example_skips_absent_stat_groups():
    # Only scalar stats present → centroid/diff branches skipped
    stats = {"scalar_params": {
        "boot_fap": {
            "median": -10.0, "mad_std": 2.0,
            "info": {
                "missing_value": None, "replace_value": None,
                "log_transform": True, "log_transform_eps": 1e-32,
                "clip_factor": 20, "standardize": True, "dtype": "float",
            },
        },
    }}
    out = normalize_example({"boot_fap": 1e-5}, stats)
    assert list(out.keys()) == ["boot_fap_norm"]
