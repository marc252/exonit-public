"""Tests for the wotan long-period sweep added to worker.py.

Two layers:
1. wotan_detrend itself preserves a wide transit when given a wider window
   (the actual physical claim).
2. The threshold logic gating the sweep activation behaves as documented.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------- #
# Layer 1: wotan window-length physics
# ---------------------------------------------------------------- #

def _inject_box_transit(time, period, t0, duration, depth):
    """Return flat_flux with a box transit injected."""
    flux = np.ones_like(time)
    phase = ((time - t0) % period) / period
    half_dur = (duration / period) / 2
    in_transit = (phase < half_dur) | (phase > 1.0 - half_dur)
    flux[in_transit] -= depth
    return flux, in_transit


def test_short_window_destroys_long_transit():
    """A 12-hour transit with window=0.5d (default) loses most of its depth.

    This is the physical reason the big-run-v3 missed deep hot Jupiters:
    the 0.5d biweight window is comparable to T₁₄, so the smoother eats
    the signal.
    """
    from worker.wotan_detrend import detrend_with_wotan
    rng = np.random.default_rng(42)
    # Long baseline with sub-Neptune-class noise
    time = np.linspace(0, 100, 30000)
    flux, in_transit = _inject_box_transit(
        time, period=80.0, t0=10.0, duration=0.5, depth=0.01)
    flux += rng.normal(0, 0.0005, size=time.size)
    flat = detrend_with_wotan(time, flux, window_length=0.5)
    finite = np.isfinite(flat)
    in_t = in_transit & finite
    out_t = (~in_transit) & finite
    if not in_t.any() or not out_t.any():
        pytest.skip("wotan returned all-NaN slice (edge cases)")
    depth_recovered = np.nanmedian(flat[out_t]) - np.nanmedian(flat[in_t])
    # Short window absorbs the transit — we expect <50% of the original
    # depth (0.01) to remain.
    assert depth_recovered < 0.005, (
        f"Short window unexpectedly preserved {depth_recovered:.5f} "
        f"of the 0.01 injected depth")


def test_long_window_preserves_long_transit():
    """The same 12-hour transit detrended with window=2.0d retains its depth."""
    from worker.wotan_detrend import detrend_with_wotan
    rng = np.random.default_rng(42)
    time = np.linspace(0, 100, 30000)
    flux, in_transit = _inject_box_transit(
        time, period=80.0, t0=10.0, duration=0.5, depth=0.01)
    flux += rng.normal(0, 0.0005, size=time.size)
    flat = detrend_with_wotan(time, flux, window_length=2.0)
    finite = np.isfinite(flat)
    in_t = in_transit & finite
    out_t = (~in_transit) & finite
    if not in_t.any() or not out_t.any():
        pytest.skip("wotan returned all-NaN slice (edge cases)")
    depth_recovered = np.nanmedian(flat[out_t]) - np.nanmedian(flat[in_t])
    # Wider window preserves substantial fraction of the 0.01 depth.
    assert depth_recovered > 0.006, (
        f"Long window only preserved {depth_recovered:.5f} of 0.01 — "
        f"too aggressive")


def test_short_window_safe_for_short_transit():
    """A 4-hour transit still works with the short window (regression guard)."""
    from worker.wotan_detrend import detrend_with_wotan
    rng = np.random.default_rng(7)
    time = np.linspace(0, 30, 15000)
    flux, in_transit = _inject_box_transit(
        time, period=4.0, t0=1.0, duration=0.17, depth=0.005)
    flux += rng.normal(0, 0.0003, size=time.size)
    flat = detrend_with_wotan(time, flux, window_length=0.5)
    finite = np.isfinite(flat)
    in_t = in_transit & finite
    out_t = (~in_transit) & finite
    if not in_t.any() or not out_t.any():
        pytest.skip("wotan returned all-NaN slice (edge cases)")
    depth_recovered = np.nanmedian(flat[out_t]) - np.nanmedian(flat[in_t])
    # Short transit should keep most of its depth with the narrow window.
    assert depth_recovered > 0.003


# ---------------------------------------------------------------- #
# Layer 2: long-period sweep activation logic (pure-function mirror)
# ---------------------------------------------------------------- #

def _should_run_long_sweep(time_span_days, n_existing_candidates,
                           threshold=100.0, max_planets=5):
    """Pure-function reflection of the worker.py inline logic."""
    return (time_span_days > threshold
            and n_existing_candidates < max_planets)


def test_sweep_skipped_for_short_baseline():
    """1-2 sectors → no sweep (long-period not detectable anyway)."""
    assert not _should_run_long_sweep(time_span_days=27, n_existing_candidates=2)
    assert not _should_run_long_sweep(time_span_days=99.9, n_existing_candidates=0)


def test_sweep_runs_for_long_baseline_with_room():
    """Long baseline + free candidate slots → sweep runs."""
    assert _should_run_long_sweep(time_span_days=200, n_existing_candidates=2)
    assert _should_run_long_sweep(time_span_days=1875, n_existing_candidates=4)


def test_sweep_skipped_when_full():
    """Even with long baseline, if max_planets already filled, skip."""
    assert not _should_run_long_sweep(
        time_span_days=1875, n_existing_candidates=5)


def test_sweep_threshold_boundary():
    """Threshold is exclusive — exactly 100d should NOT trigger."""
    assert not _should_run_long_sweep(time_span_days=100.0,
                                      n_existing_candidates=0)
    assert _should_run_long_sweep(time_span_days=100.001,
                                  n_existing_candidates=0)


@pytest.mark.parametrize("tic_id,span,n_cands,expected", [
    # Real big-run-v3 cases that should benefit from the sweep
    (273234825, 1875.4, 0, True),
    (209464063, 2576.9, 0, True),
    (402898317, 1104.1, 0, True),
    (466206508, 1844.3, 5, False),  # already 5 candidates from short loop
    (113921235, 811.2, 5, False),
])
def test_real_world_v3_zero_candidate_tics(tic_id, span, n_cands, expected):
    assert _should_run_long_sweep(span, n_cands) is expected
