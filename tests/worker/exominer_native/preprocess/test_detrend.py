"""Fast unit tests for ``worker.exominer_native.preprocess.detrend``."""
from __future__ import annotations

import numpy as np

from worker.exominer_native.preprocess.detrend import (
    compute_bic,
    detrend_flux_using_spline,
)


DEFAULT_CONFIG = {
    "gapWidth": 0.75,
    "random_seed": 42,
    "spline_bkspace_min": 0.5,
    "spline_bkspace_max": 20.0,
    "spline_bkspace_num": 20,
    "spline_maxiter": 5,
    "spline_penalty_coeff": 1.0,
    "spline_outlier_cut": 3.0,
}


def _make_synthetic_sector(n_points=800):
    """Build a synthetic two-segment light curve with a slow trend + gap."""
    rng = np.random.default_rng(7)
    t1 = np.linspace(1000.0, 1013.0, n_points // 2)
    t2 = np.linspace(1015.0, 1027.0, n_points // 2)
    # slow cosine trend + small noise
    flux1 = 1.0 + 0.02 * np.cos(0.4 * (t1 - 1000.0)) + rng.normal(0.0, 0.0005, t1.size)
    flux2 = 1.0 + 0.02 * np.cos(0.4 * (t2 - 1000.0)) + rng.normal(0.0, 0.0005, t2.size)
    intransit1 = np.zeros(t1.size, dtype=bool)
    intransit2 = np.zeros(t2.size, dtype=bool)
    # mark a narrow in-transit window
    intransit1[50:56] = True
    intransit2[120:126] = True
    return [t1, t2], [flux1, flux2], [intransit1, intransit2]


def test_detrend_flux_using_spline_returns_same_length_arrays():
    time_arrs, flux_arrs, intransit = _make_synthetic_sector()
    t, detrended, spline, interp = detrend_flux_using_spline(
        flux_arrs, time_arrs, intransit, DEFAULT_CONFIG
    )

    assert t.shape == detrended.shape == spline.shape == interp.shape
    assert t.size == sum(a.size for a in time_arrs)


def test_detrended_flux_is_close_to_unity_on_a_slow_trend():
    time_arrs, flux_arrs, intransit = _make_synthetic_sector()
    _, detrended, _, _ = detrend_flux_using_spline(
        flux_arrs, time_arrs, intransit, DEFAULT_CONFIG
    )
    # median of a well-detrended series sits near 1.0
    assert np.isclose(np.nanmedian(detrended), 1.0, atol=0.005)


def test_compute_bic_penalises_higher_order():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    x_fit_good = x.copy()
    x_fit_bad = x + 1.0
    bic_good_k1 = compute_bic(1, len(x), x, x_fit_good)
    bic_bad_k1 = compute_bic(1, len(x), x, x_fit_bad)
    # Perfect fit BIC < bad fit BIC
    assert bic_good_k1 < bic_bad_k1
    # Higher k increases penalty even with identical fit
    bic_good_k3 = compute_bic(3, len(x), x, x_fit_good)
    assert bic_good_k3 > bic_good_k1
