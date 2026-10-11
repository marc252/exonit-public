"""Regression tests: triceratops_fpp/nfpp == 0.0 must NOT collapse into the
-1 "missing" sentinel at serving time.

Root cause: `float(candidate.get("triceratops_fpp", -1) or -1)` relies on
Python truthiness, and `0.0 or -1` evaluates to `-1` — so a legitimate,
frequent (~149k rows in production) FPP/NFPP reading of exactly 0.0 was
silently treated as missing, producing NaN features at inference time even
though the TRAINING-side preprocessing (`preprocess_v2`/`preprocess_v2_1`,
which use `tri >= 0`) always treated 0.0 as valid. This is a train/serve
parity bug. See exonit/verdict_classifier.py::_float_or_sentinel for the fix.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from exonit.verdict_classifier import _float_or_sentinel, _predict_single_numpy
from exonit.verdict_classifier_v2 import _predict_single_v2, preprocess_v2
from exonit.verdict_classifier_v2_1 import (
    _base_feature_names_v2_1,
    _predict_single_v2_1,
    preprocess_v2_1,
)

_TOP_SECTORS = [
    "s01", "s02", "s03", "s04", "s05",
    "s06", "s07", "s08", "s09", "s10",
    "s11", "s12", "s13", "s14", "s15",
]


# ---------------------------------------------------------------------------
# 1. Unit tests for the helper itself
# ---------------------------------------------------------------------------

def test_float_or_sentinel_none_becomes_sentinel():
    assert _float_or_sentinel(None) == -1.0


def test_float_or_sentinel_zero_stays_zero():
    """The core regression: 0.0 must NOT collapse to the sentinel."""
    assert _float_or_sentinel(0.0) == 0.0


def test_float_or_sentinel_stored_sentinel_passthrough():
    assert _float_or_sentinel(-1.0) == -1.0


def test_float_or_sentinel_regular_value_passthrough():
    assert _float_or_sentinel(0.37) == 0.37


def test_float_or_sentinel_nan_passthrough():
    result = _float_or_sentinel(float("nan"))
    assert math.isnan(result)


def test_float_or_sentinel_custom_sentinel():
    assert _float_or_sentinel(None, sentinel=-2.0) == -2.0


# ---------------------------------------------------------------------------
# 2. v1 serve path (exonit/verdict_classifier.py::_predict_single_numpy)
# ---------------------------------------------------------------------------

_BASE_CAND = {
    "period_days": 3.5, "depth_ppm": 500.0, "duration_days": 0.12,
    "snr": 12.0, "cnn_score": 0.7,
    "exominer_score": 0.6, "exominer_tce_period": 3.5,
}
_BASE_STAR = {"tmag": 10.0, "ra": 120.0, "dec": -30.0,
              "n_sectors": 3, "time_span_days": 100.0}


@pytest.mark.parametrize("fpp_value,expected_valid,expected_feature", [
    (0.0, 1, 0.0),
    (None, 0, None),   # None -> NaN feature
    (-1.0, 0, None),   # stored sentinel -> NaN feature
    (0.42, 1, 0.42),
])
def test_v1_serve_path_tri_valid_and_value(fpp_value, expected_valid, expected_feature):
    cand = {**_BASE_CAND, "triceratops_fpp": fpp_value}
    row_df = _predict_single_numpy(None, cand, _BASE_STAR)
    assert row_df["tri_valid"].iloc[0] == expected_valid
    if expected_feature is None:
        assert np.isnan(row_df["triceratops_fpp"].iloc[0])
    else:
        assert row_df["triceratops_fpp"].iloc[0] == expected_feature


def test_v1_serve_path_missing_key_treated_as_none():
    cand = {k: v for k, v in _BASE_CAND.items()}  # no triceratops_fpp key at all
    row_df = _predict_single_numpy(None, cand, _BASE_STAR)
    assert row_df["tri_valid"].iloc[0] == 0
    assert np.isnan(row_df["triceratops_fpp"].iloc[0])


# ---------------------------------------------------------------------------
# 3. v2 serve path vs preprocess_v2 (train/serve parity)
# ---------------------------------------------------------------------------

class _SpyStage:
    """Minimal stand-in for a fitted LGBMClassifier stage: records the last
    feature DataFrame it was asked to score and returns a fixed 2-class
    probability row, so `_predict_single_v2[_1]` can run end-to-end without
    training a real model."""

    def __init__(self):
        self.last_X = None

    def predict_proba(self, X):
        self.last_X = X.copy()
        return np.array([[0.5, 0.5]])


_V2_CAND_BASE = {
    "period_days": 3.5, "depth_ppm": 500.0, "duration_days": 0.12,
    "snr": 12.0, "cnn_score": 0.7,
    "exominer_score": 0.6, "exominer_tce_period": 3.5,
    "exominer_sector_run": "s01", "n_datapoints": 5000,
}
_V2_STAR_BASE = {"n_sectors": 3, "time_span_days": 100.0, "n_datapoints": 5000}


@pytest.mark.parametrize("fpp_value,expected_valid,expected_nan", [
    (0.0, 1, False),
    (None, 0, True),
    (-1.0, 0, True),
    (0.42, 1, False),
])
def test_v2_serve_path_tri_valid_and_value(fpp_value, expected_valid, expected_nan):
    # Build the same feature-name list preprocess_v2 would produce for a
    # 1-row frame with this fpp value, so the spy stage can accept the row.
    df = pd.DataFrame([{**_V2_CAND_BASE, "triceratops_fpp": fpp_value,
                        **_V2_STAR_BASE}])
    feature_names = list(preprocess_v2(df, top_sectors=_TOP_SECTORS).columns)

    stage1, stage2 = _SpyStage(), _SpyStage()
    cand = {**_V2_CAND_BASE, "triceratops_fpp": fpp_value}
    _predict_single_v2(stage1, stage2, cand, _V2_STAR_BASE,
                       feature_names, _TOP_SECTORS)

    row = stage1.last_X.iloc[0]
    assert row["tri_valid"] == expected_valid
    if expected_nan:
        assert np.isnan(row["triceratops_fpp"])
    else:
        assert row["triceratops_fpp"] == fpp_value


def test_v2_serve_path_matches_preprocess_v2_for_zero_fpp():
    """Train/serve parity: the serve-path row for fpp=0.0 must carry the
    exact same tri_valid/triceratops_fpp values as preprocess_v2 (the
    TRAINING-side feature builder) produces for the same input row."""
    df = pd.DataFrame([{**_V2_CAND_BASE, "triceratops_fpp": 0.0,
                        **_V2_STAR_BASE}])
    train_row = preprocess_v2(df, top_sectors=_TOP_SECTORS).iloc[0]
    feature_names = list(preprocess_v2(df, top_sectors=_TOP_SECTORS).columns)

    stage1, stage2 = _SpyStage(), _SpyStage()
    cand = {**_V2_CAND_BASE, "triceratops_fpp": 0.0}
    _predict_single_v2(stage1, stage2, cand, _V2_STAR_BASE,
                       feature_names, _TOP_SECTORS)
    serve_row = stage1.last_X.iloc[0]

    assert serve_row["tri_valid"] == train_row["tri_valid"] == 1
    assert serve_row["triceratops_fpp"] == train_row["triceratops_fpp"] == 0.0


# ---------------------------------------------------------------------------
# 4. v2.1 serve path vs preprocess_v2_1 (train/serve parity, FPP + NFPP)
# ---------------------------------------------------------------------------

_V2_1_CAND_BASE = {
    "period_days": 3.5, "depth_ppm": 500.0, "duration_days": 0.12,
    "snr": 12.0, "cnn_score": 0.7,
    "exominer_score": 0.6, "exominer_tce_period": 3.5,
    "exominer_sector_run": "s01", "n_datapoints": 5000,
    "triceratops_prob_planet": 0.5,
}
_V2_1_STAR_BASE = {"n_sectors": 3, "time_span_days": 100.0, "n_datapoints": 5000,
                   "berger_mass": 1.0, "berger_rad": 1.0, "berger_teff": 5778.0,
                   "berger_logg": 4.4, "berger_feh": 0.0}


@pytest.mark.parametrize("value,expected_valid,expected_nan", [
    (0.0, 1, False),
    (None, 0, True),
    (-1.0, 0, True),
    (0.42, 1, False),
])
def test_v2_1_serve_path_tri_valid_and_value(value, expected_valid, expected_nan):
    df = pd.DataFrame([{**_V2_1_CAND_BASE, "triceratops_fpp": value,
                        "triceratops_nfpp": 0.5, **_V2_1_STAR_BASE}])
    feature_names = list(preprocess_v2_1(df, top_sectors=_TOP_SECTORS).columns)

    stage1, stage2 = _SpyStage(), _SpyStage()
    cand = {**_V2_1_CAND_BASE, "triceratops_fpp": value, "triceratops_nfpp": 0.5}
    _predict_single_v2_1(stage1, stage2, cand, _V2_1_STAR_BASE,
                         feature_names, _TOP_SECTORS)

    row = stage1.last_X.iloc[0]
    assert row["tri_valid"] == expected_valid
    if expected_nan:
        assert np.isnan(row["triceratops_fpp"])
    else:
        assert row["triceratops_fpp"] == value


@pytest.mark.parametrize("value,expected_valid,expected_nan", [
    (0.0, 1, False),
    (None, 0, True),
    (-1.0, 0, True),
    (0.42, 1, False),
])
def test_v2_1_serve_path_nfpp_valid_and_value(value, expected_valid, expected_nan):
    df = pd.DataFrame([{**_V2_1_CAND_BASE, "triceratops_fpp": 0.5,
                        "triceratops_nfpp": value, **_V2_1_STAR_BASE}])
    feature_names = list(preprocess_v2_1(df, top_sectors=_TOP_SECTORS).columns)

    stage1, stage2 = _SpyStage(), _SpyStage()
    cand = {**_V2_1_CAND_BASE, "triceratops_fpp": 0.5, "triceratops_nfpp": value}
    _predict_single_v2_1(stage1, stage2, cand, _V2_1_STAR_BASE,
                         feature_names, _TOP_SECTORS)

    row = stage1.last_X.iloc[0]
    assert row["nfpp_valid"] == expected_valid
    if expected_nan:
        assert np.isnan(row["triceratops_nfpp"])
    else:
        assert row["triceratops_nfpp"] == value


def test_v2_1_serve_path_matches_preprocess_v2_1_for_zero_fpp_and_nfpp():
    """Train/serve parity for v2.1 with BOTH fpp=0.0 and nfpp=0.0."""
    df = pd.DataFrame([{**_V2_1_CAND_BASE, "triceratops_fpp": 0.0,
                        "triceratops_nfpp": 0.0, **_V2_1_STAR_BASE}])
    train_row = preprocess_v2_1(df, top_sectors=_TOP_SECTORS).iloc[0]
    feature_names = list(preprocess_v2_1(df, top_sectors=_TOP_SECTORS).columns)

    stage1, stage2 = _SpyStage(), _SpyStage()
    cand = {**_V2_1_CAND_BASE, "triceratops_fpp": 0.0, "triceratops_nfpp": 0.0}
    _predict_single_v2_1(stage1, stage2, cand, _V2_1_STAR_BASE,
                         feature_names, _TOP_SECTORS)
    serve_row = stage1.last_X.iloc[0]

    assert serve_row["tri_valid"] == train_row["tri_valid"] == 1
    assert serve_row["triceratops_fpp"] == train_row["triceratops_fpp"] == 0.0
    assert serve_row["nfpp_valid"] == train_row["nfpp_valid"] == 1
    assert serve_row["triceratops_nfpp"] == train_row["triceratops_nfpp"] == 0.0
