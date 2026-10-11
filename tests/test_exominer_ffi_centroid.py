"""Unit tests for TGLC centroid extraction."""
from __future__ import annotations

import numpy as np
import pytest


def test_extract_centroid_returns_xy():
    from exonit.exominer_ffi.centroid import extract_centroid_xy

    t = np.linspace(2000.0, 2027.0, 100)
    x = np.full_like(t, 256.5) + np.random.RandomState(0).normal(0, 0.05, 100)
    y = np.full_like(t, 512.5) + np.random.RandomState(1).normal(0, 0.05, 100)

    out = extract_centroid_xy(t, x, y, quality=np.zeros_like(t, dtype=int))
    assert out["time"].shape == (100,)
    assert out["x"].shape == (100,)
    assert out["y"].shape == (100,)
    # After median subtraction, the median should be ~0
    assert abs(np.median(out["x"])) < 1e-6
    assert abs(np.median(out["y"])) < 1e-6


def test_extract_centroid_drops_bad_quality():
    from exonit.exominer_ffi.centroid import extract_centroid_xy

    t = np.arange(10, dtype=float)
    x = np.zeros(10) + 100.0
    y = np.zeros(10) + 200.0
    q = np.zeros(10, dtype=int)
    q[3] = 1  # one bad sample
    out = extract_centroid_xy(t, x, y, quality=q)
    assert out["time"].shape == (9,)
    assert 3 not in out["time"]


def test_centroid_columns_returns_none_if_missing():
    """QLP files lack centroid columns → helper returns None."""
    from exonit.exominer_ffi.centroid import find_centroid_columns

    cols_qlp = ["TIME", "SAP_FLUX", "QUALITY"]
    assert find_centroid_columns(cols_qlp) is None

    cols_tglc = ["TIME", "TESS_FLUX", "X_CENT", "Y_CENT", "QUALITY"]
    assert find_centroid_columns(cols_tglc) == ("X_CENT", "Y_CENT")


def test_centroid_columns_recognises_psf_variants():
    """Some TGLC versions use X_PSF / Y_PSF instead of X_CENT / Y_CENT."""
    from exonit.exominer_ffi.centroid import find_centroid_columns

    cols = ["TIME", "TESS_FLUX", "X_PSF", "Y_PSF", "QUALITY"]
    assert find_centroid_columns(cols) == ("X_PSF", "Y_PSF")


def test_extract_centroid_handles_empty_after_filter():
    """If all samples are bad, return empty arrays cleanly."""
    from exonit.exominer_ffi.centroid import extract_centroid_xy

    t = np.arange(5, dtype=float)
    out = extract_centroid_xy(t, np.zeros(5), np.zeros(5),
                              quality=np.ones(5, dtype=int))
    assert out["time"].shape == (0,)
    assert out["x"].shape == (0,)
    assert out["y"].shape == (0,)
