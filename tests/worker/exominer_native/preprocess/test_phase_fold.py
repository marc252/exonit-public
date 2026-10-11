"""Fast unit tests for ``worker.exominer_native.preprocess.phase_fold``."""
from __future__ import annotations

import numpy as np

from worker.exominer_native.preprocess.phase_fold import (
    generate_view,
    global_view,
    local_view,
    normalize_view,
    phase_fold_and_sort_light_curve,
)


def _make_synthetic_transit_series(n=2000):
    rng = np.random.default_rng(7)
    time = np.linspace(0.0, 20.0, n)
    period = 3.0
    t0 = 0.5
    duration = 0.2
    # flat + Gaussian noise + boxy transits
    flux = 1.0 + rng.normal(0.0, 0.0002, n)
    phase = np.mod(time + period / 2 - t0, period) - period / 2
    in_transit = np.abs(phase) < duration / 2
    flux[in_transit] -= 0.01
    return time, flux, period, t0, duration


def test_phase_fold_and_sort_produces_monotonic_phase():
    time, flux, period, t0, _ = _make_synthetic_transit_series()
    t_f, f_f, num_transits = phase_fold_and_sort_light_curve(time, flux, period, t0)
    assert t_f.shape == time.shape
    assert f_f.shape == flux.shape
    # After sorting, the phase values are monotonically non-decreasing
    assert np.all(np.diff(t_f) >= -1e-12)
    assert num_transits >= 6  # ~20 days / 3 d period


def test_global_view_returns_default_2001_bins_centered_around_zero():
    time, flux, period, t0, duration = _make_synthetic_transit_series()
    t_f, f_f, _ = phase_fold_and_sort_light_curve(time, flux, period, t0)
    tce = {"tce_period": period, "tce_duration": duration}
    view, time_bins, view_var, inds_nan, bin_counts = global_view(
        t_f, f_f, period, tce, tce_duration=duration
    )
    assert view.shape == (2001,)
    assert time_bins.shape == (2001,)
    # centering=True by default: median of the view ~ 0
    assert abs(np.nanmedian(view)) < 0.5


def test_local_view_returns_default_201_bins():
    time, flux, period, t0, duration = _make_synthetic_transit_series()
    t_f, f_f, _ = phase_fold_and_sort_light_curve(time, flux, period, t0)
    tce = {"tce_period": period, "tce_duration": duration}
    view, time_bins, view_var, inds_nan, bin_counts = local_view(
        t_f, f_f, period, duration, tce, tce_duration=duration
    )
    assert view.shape == (201,)
    assert time_bins.shape == (201,)


def test_normalize_view_divides_by_extreme_value():
    view = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    # flux branch: divide by |min| = 4
    out = normalize_view(view.copy(), val=None, centroid=False)
    np.testing.assert_allclose(out, view / (4.0 + 1e-12))


def test_normalize_view_divide_by_zero_returns_view_via_eps():
    view = np.zeros(5)
    out = normalize_view(view.copy(), val=None, centroid=False)
    # val = |min| = 0 → divide by eps, all zeros remain
    assert np.all(out == 0.0)


def test_generate_view_bin_count_matches_num_bins():
    time, flux, period, t0, duration = _make_synthetic_transit_series()
    t_f, f_f, _ = phase_fold_and_sort_light_curve(time, flux, period, t0)
    tce = {"tce_period": period, "tce_duration": duration}
    view, time_bins, view_var, inds_nan, bin_counts = generate_view(
        t_f, f_f, num_bins=301, bin_width=0.01, t_min=-period / 2, t_max=period / 2, tce=tce,
    )
    assert view.shape == (301,)
    assert bin_counts.shape == (301,)
