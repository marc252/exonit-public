"""Fast unit tests for ``worker.exominer_native.preprocess.imputing``."""
from __future__ import annotations

import numpy as np

from worker.exominer_native.preprocess.imputing import (
    impute_binned_ts,
    impute_transits,
    imputing_gaps,
)


def test_impute_binned_ts_fills_oot_nans_with_stats():
    phase = np.linspace(-5.0, 5.0, 21)
    binned = np.ones(21, dtype=float)
    binned[0] = np.nan  # out-of-transit NaN far from centre
    binned[-1] = np.nan
    var_in = np.full(21, 0.1)
    result, var, inds = impute_binned_ts(
        phase, binned.copy(), period=10.0, duration=0.5, binned_ts_var=var_in.copy()
    )
    # the nan mask is reported back
    assert inds["oot"].sum() >= 2
    # no NaNs remain in the returned series after imputation
    assert not np.isnan(result[~inds["it"]]).any()
    assert not np.isnan(var).any()


def test_impute_binned_ts_interpolates_in_transit_nans_linearly():
    phase = np.linspace(-5.0, 5.0, 21)
    # a ramp 0..20 with an in-transit NaN near the centre
    series = np.arange(21, dtype=float)
    series[10] = np.nan
    out, _, inds = impute_binned_ts(phase, series, period=10.0, duration=1.0)
    # centre bin is in-transit (|phase| <= 1.5*duration = 1.5)
    assert inds["it"][10]
    # linear interpolation of the ramp restores the 10.0 value
    np.testing.assert_allclose(out[10], 10.0)


def test_imputing_gaps_replaces_flagged_indices_with_gaussian_noise():
    rng = np.random.default_rng(0)
    ts = rng.normal(size=200)
    mask = np.zeros(200, dtype=bool)
    mask[50:60] = True
    ts_before = ts.copy()
    ts_after = imputing_gaps(ts.copy(), mask, seed=42)
    # values inside the mask changed; values outside did not
    assert not np.array_equal(ts_after[50:60], ts_before[50:60])
    np.testing.assert_array_equal(ts_after[~mask], ts_before[~mask])


def test_imputing_gaps_returns_untouched_when_all_nan():
    ts = np.full(50, np.nan)
    mask = np.zeros(50, dtype=bool)
    mask[10:20] = True
    out = imputing_gaps(ts.copy(), mask, seed=1)
    # all-NaN → skipped → NaNs remain
    assert np.isnan(out).all()


def test_impute_transits_linear_between_boundaries():
    ts = [np.arange(11, dtype=float)]
    # transit at indices 4, 5, 6
    pulses = [np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0])]
    out = impute_transits(ts, pulses)
    # indices 4-6: window = 3, left median = median([0,1,2,3]) = 1.5,
    # right median = median([7,8,9,10]) = 8.5, then np.interp(x=[4,5,6], xp=[3,7], fp=[1.5, 8.5])
    expected = np.arange(11, dtype=float)
    expected[4:7] = np.interp([4, 5, 6], [3, 7], [1.5, 8.5])
    np.testing.assert_allclose(out[0], expected)
