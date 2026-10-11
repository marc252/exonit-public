"""Fast unit tests for ``worker.exominer_native.preprocess.utils_common``."""
from __future__ import annotations

import numpy as np

from worker.exominer_native.preprocess.utils_common import (
    check_inputs,
    count_transits,
    get_out_of_transit_idxs_glob,
    get_out_of_transit_idxs_loc,
    min_max_normalization,
    remove_non_finite_values,
    remove_outliers,
)


def test_get_out_of_transit_idxs_loc_marks_edges():
    (idxs,) = get_out_of_transit_idxs_loc(num_bins_loc=31, num_transit_durations=5)
    # centre bins (|i-15| < 31/5/2 ~ 3) are in-transit, the rest are oot
    assert 0 in idxs
    assert 30 in idxs
    assert 15 not in idxs


def test_get_out_of_transit_idxs_glob_marks_edges():
    (idxs,) = get_out_of_transit_idxs_glob(
        num_bins_glob=301, transit_duration=0.1, orbital_period=10.0
    )
    # transit is ~1% of period → centre bin marked in-transit
    assert 150 not in idxs
    assert 0 in idxs


def test_count_transits_counts_in_range():
    time = np.linspace(0.0, 30.0, 1500)
    n = count_transits(time, period=3.0, epoch=1.0, duration=0.2)
    assert n >= 8  # 10 transits inside 30 days, allow boundary slop


def test_min_max_normalization_maps_to_0_1():
    arr = np.linspace(5.0, 15.0, 11)
    out = min_max_normalization(arr, max_val=15.0, min_val=5.0)
    assert out[0] == 0.0
    assert out[-1] == 1.0


def test_remove_non_finite_values_drops_rows_with_nan_or_inf():
    t = [np.array([1.0, 2.0, 3.0, 4.0])]
    f = [np.array([1.0, np.nan, 3.0, np.inf])]
    out = remove_non_finite_values([t, f])
    np.testing.assert_array_equal(out[0][0], np.array([1.0, 3.0]))
    np.testing.assert_array_equal(out[1][0], np.array([1.0, 3.0]))


def test_remove_outliers_sets_nan_when_fill_false():
    ts = np.array([1.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0])
    out, idxs = remove_outliers(ts.copy(), sigma=3, fill=False, outlier_type="upper")
    assert np.isnan(out[4])
    assert idxs[4]


def test_check_inputs_flags_sparse_centroid_coverage():
    data = {
        "all_centroids": {
            "x": [np.concatenate([np.full(999, np.nan), np.array([1.0])])],
            "y": [np.zeros(1000)],
        }
    }
    errs = check_inputs(data)
    assert "Less than 0.5% cadences are valid for centroid data." in errs
