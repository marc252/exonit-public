"""Fast unit tests for ``worker.exominer_native.preprocess.ephemeris``."""
from __future__ import annotations

import numpy as np
import pytest

from worker.exominer_native.preprocess.ephemeris import (
    create_binary_time_series,
    find_first_epoch_after_this_time,
    get_startend_idxs_inter,
    lininterp_transits,
)


def test_find_first_epoch_before_reference_adds_multiples_of_period():
    # epoch=10, period=3, reference_time=25 → ceil((25-10)/3)=5 → 10+15=25
    assert find_first_epoch_after_this_time(10.0, 3.0, 25.0) == 25.0


def test_find_first_epoch_after_reference_subtracts_multiples_of_period():
    # epoch=30, period=3, reference_time=25 → floor((30-25)/3)=1 → 30-3=27
    assert find_first_epoch_after_this_time(30.0, 3.0, 25.0) == 27.0


def test_create_binary_time_series_marks_in_transit_cadences():
    # transits at t=0.5, 2.5, 4.5, each 0.2 days wide, period=2
    time = np.linspace(0.0, 5.0, 101)
    bts = create_binary_time_series(time, epoch=0.5, duration=0.2, period=2.0)
    assert bts.dtype == bool
    assert bts.shape == time.shape
    # exactly at t=0.5 we expect a 1
    assert bts[np.argmin(np.abs(time - 0.5))]
    # far from any transit center we expect a 0
    assert not bts[np.argmin(np.abs(time - 1.5))]


def test_get_startend_idxs_inter_finds_contiguous_runs():
    pulses = [np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0])]
    idxs_it, start_idxs, end_idxs = get_startend_idxs_inter(pulses)
    assert np.array_equal(idxs_it[0], np.array([1, 2, 5, 7, 8, 9]))
    # 3 transits: indices 1-2, 5, 7-9 inside idxs_it
    assert np.array_equal(start_idxs[0], np.array([0, 2, 3]))
    assert np.array_equal(end_idxs[0], np.array([2, 3, 6]))


def test_lininterp_transits_interpolates_middle_transit_linearly():
    # ramp 0..10 with a transit at indices 4-6 to be overwritten
    ts = [np.arange(11, dtype=float)]
    pulses = [np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0])]
    idxs_it, start, end = get_startend_idxs_inter(pulses)
    result = lininterp_transits(ts, pulses, idxs_it, start, end)
    # the 1's were at values 4, 5, 6 before; linear interp between index 3 (=3)
    # and index 7 (=7) gives 4, 5, 6 — numerically the same ramp
    np.testing.assert_allclose(result[0], np.arange(11, dtype=float))


@pytest.mark.parametrize("epoch,period,ref", [
    (0.0, 1.0, 100.5),     # epoch < ref, equal splits
    (500.0, 7.3, 499.9),   # epoch > ref by tiny margin
    (-1.0, 3.14, 10.0),    # negative epoch
])
def test_find_first_epoch_result_is_within_one_period_of_ref(epoch, period, ref):
    t = find_first_epoch_after_this_time(epoch, period, ref)
    assert t >= ref - 1e-9
    assert t - ref < period + 1e-9
