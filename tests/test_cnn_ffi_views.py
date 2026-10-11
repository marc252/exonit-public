"""Pure-function tests for exonit/cnn_ffi_views.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

from exonit.cnn_ffi_views import (
    GLOBAL_BINS, LOCAL_BINS, global_view, local_view, ViewError,
    phase_fold_sort,
)


def test_phase_fold_sort_centers_transit_at_zero():
    """Synthetic transit at t0=10, period=2.0, duration=0.1.
    After phase-folding the minimum flux must sit near t=0."""
    rng = np.random.default_rng(42)
    period = 2.0
    duration = 0.1
    t0 = 10.0
    time = np.linspace(0, 30, 3000)
    flux = np.ones_like(time) + 0.001 * rng.standard_normal(time.size)
    in_transit = np.abs(((time - t0) % period)) < duration / 2
    flux[in_transit] -= 0.01
    t_sorted, f_sorted = phase_fold_sort(time, flux, period, t0, duration)
    min_idx = int(np.argmin(f_sorted))
    assert abs(t_sorted[min_idx]) < 0.05, f"Min at t={t_sorted[min_idx]}, expected near 0"


def test_phase_fold_sort_handles_t0_at_window_edge():
    """Edge case: t0 == start of timeseries."""
    time = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    flux = np.array([1.0, 0.99, 1.0, 0.99, 1.0])
    period = 1.0
    t_sorted, f_sorted = phase_fold_sort(time, flux, period, 0.0, 0.1)
    assert len(t_sorted) == len(time)
    assert (t_sorted >= -period / 2).all()
    assert (t_sorted < period / 2).all()


def _synthetic_transit(period=2.0, t0=1.0, duration=0.1, depth=0.01, n=3000, seed=0):
    """Helper: synthetic light curve with a box transit."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 30.0, n)
    flux = np.ones_like(time) + 0.0005 * rng.standard_normal(time.size)
    phase = (time - t0) % period
    in_transit = np.abs(phase) < duration / 2
    in_transit |= np.abs(phase - period) < duration / 2
    flux[in_transit] -= depth
    return time, flux


def test_global_view_shape():
    """global_view returns GLOBAL_BINS array of float32."""
    time, flux = _synthetic_transit()
    view = global_view(time, flux, period=2.0, t0=1.0, duration=0.1)
    assert view.shape == (GLOBAL_BINS,)
    assert view.dtype == np.float32


def test_global_view_normalization_min_neg_one():
    """AstroNet convention: median→0, |min|→1."""
    time, flux = _synthetic_transit(depth=0.05)
    view = global_view(time, flux, period=2.0, t0=1.0, duration=0.1)
    assert abs(np.median(view)) < 0.05
    assert abs(view.min() + 1.0) < 0.05


def test_local_view_shape():
    """local_view returns LOCAL_BINS array."""
    time, flux = _synthetic_transit()
    view = local_view(time, flux, period=2.0, t0=1.0, duration=0.1)
    assert view.shape == (LOCAL_BINS,)


def test_views_raise_on_sparse_window():
    """If <5 points fall inside the view window, raise ViewError."""
    time = np.array([0.0, 1.0, 2.0])
    flux = np.array([1.0, 0.99, 1.0])
    with pytest.raises(ViewError):
        global_view(time, flux, period=10.0, t0=5.0, duration=0.5)


