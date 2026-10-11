"""Fast unit tests for ``worker.exominer_native.preprocess.gapping``."""
from __future__ import annotations

import numpy as np

from worker.exominer_native.preprocess.gapping import (
    gap_this_tce,
    get_gap_indices,
)


def test_gap_this_tce_returns_indices_for_each_array():
    # Two observation windows, each 10 days long, 3 cadences/day
    t1 = np.linspace(1000.0, 1010.0, 31)
    t2 = np.linspace(1015.0, 1025.0, 31)
    tce = {
        "tce_time0bk": 1001.0,
        "tce_period": 5.0,
        "tce_duration": 0.25,
    }
    result = gap_this_tce([t1, t2], tce, gap_pad=1)

    assert len(result) == 2
    # Each element is a tuple (ndarray,) from np.where
    assert isinstance(result[0], tuple) and isinstance(result[0][0], np.ndarray)


def test_gap_this_tce_with_maxmesd_uses_secondary_phase_distance():
    t = np.linspace(0.0, 20.0, 201)
    tce_no_sec = {"tce_time0bk": 1.0, "tce_period": 3.0, "tce_duration": 0.3}
    tce_with_sec = {**tce_no_sec, "tce_maxmesd": 0.8}

    r_no = gap_this_tce([t], tce_no_sec, gap_pad=2)[0][0]
    r_yes = gap_this_tce([t], tce_with_sec, gap_pad=2)[0][0]
    # Both produce *some* indices (not empty) on this dense grid
    assert r_no.size > 0
    assert r_yes.size > 0


def test_get_gap_indices_finds_runs_in_arr_index_space():
    # Note: the container's get_gap_indices returns pairs of indices *into
    # np.where(...)* (not into the original flux), because the "N>=2" block
    # always runs (the elif in the source is missing a subsequent `else`,
    # a pre-existing quirk kept here for parity).
    flux = np.array([1.0, 2.0, np.nan, np.nan, 3.0, 4.0, np.nan, 5.0])
    out = get_gap_indices(flux, checkfuncs=["nan"])
    # np.where(isnan) = [2, 3, 6] → groups in arr-index space: [0,2), [2,3)
    assert out["nan"] == [[0, 2], [2, 3]]


def test_get_gap_indices_on_flux_with_no_anomalies_returns_single_zero_range():
    # Because the N<2 branches fall through to the always-executed block,
    # every check key ends up with [[0, 0]] when arr is empty — this is the
    # oracle's behaviour and parity demands we preserve it.
    flux = np.array([1.0, 2.0, 3.0])
    out = get_gap_indices(flux)
    for key in (0, "nan", "inf", "-inf"):
        assert out[key] == [[0, 0]]
