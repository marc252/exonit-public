"""Fast unit tests for ``worker.exominer_native.preprocess.odd_even``."""
from __future__ import annotations

import numpy as np

from worker.exominer_native.preprocess.odd_even import (
    check_odd_even_phasefolded,
    check_odd_even_views,
    compute_oot_it_var_oddeven,
    computed_oot_it_var_oddeven,
    generate_view,
    impute_odd_even_views,
    odd_even_binning,
    perform_bin_replacement_oddeven,
    phase_fold_and_sort_light_curve_odd_even,
)


def _synthetic_series(n=2000):
    rng = np.random.default_rng(7)
    time = np.linspace(0.0, 20.0, n)
    period = 3.0
    t0 = 0.5
    duration = 0.2
    flux = 1.0 + rng.normal(0.0, 0.0002, n)
    phase = np.mod(time + period / 2 - t0, period) - period / 2
    in_transit = np.abs(phase) < duration / 2
    flux[in_transit] -= 0.01
    return time, flux, period, t0, duration


def test_phase_fold_and_sort_odd_even_produces_two_tuples_with_expected_contents():
    time, flux, period, t0, _ = _synthetic_series()
    odd, even = phase_fold_and_sort_light_curve_odd_even(time, flux, period, t0)

    odd_time, odd_flux, num_tr_odd, odd_nophase = odd
    even_time, even_flux, num_tr_even, even_nophase = even

    # combined length should equal the input, and each side should have a sensible count.
    assert odd_time.shape[0] + even_time.shape[0] == time.shape[0]
    assert odd_flux.shape == odd_time.shape
    assert even_flux.shape == even_time.shape
    # phase values live in [-period/2, period/2) after sorting and should be monotonic.
    assert np.all(np.diff(odd_time) >= -1e-12)
    assert np.all(np.diff(even_time) >= -1e-12)
    # across ~20 days with 3-day period → roughly 6 transits split between odd/even.
    assert num_tr_odd + num_tr_even >= 6


def test_check_odd_even_views_replaces_flat_odd_with_even():
    odd = np.ones(10)
    even = np.linspace(-0.5, 0.5, 10)
    out_odd, out_even, flag = check_odd_even_views(odd.copy(), even.copy())
    assert flag == "odd_flat"
    np.testing.assert_array_equal(out_odd, even)
    np.testing.assert_array_equal(out_even, even)


def test_check_odd_even_views_ok_when_neither_flat():
    odd = np.linspace(-0.5, 0.5, 10)
    even = np.linspace(-0.4, 0.4, 10)
    out_odd, out_even, flag = check_odd_even_views(odd.copy(), even.copy())
    assert flag == "ok"
    np.testing.assert_array_equal(out_odd, odd)
    np.testing.assert_array_equal(out_even, even)


def test_check_odd_even_phasefolded_replaces_odd_when_missing():
    odd_time = np.array([])
    odd_flux = np.array([])
    even_time = np.linspace(-0.5, 0.5, 10)
    even_flux = np.linspace(0.9, 1.0, 10)

    (ot, of, no), (et, ef, ne), flag = check_odd_even_phasefolded(
        odd_time, odd_flux, 0, even_time, even_flux, 5,
    )
    assert flag == "no_odd_time"
    # odd side got replaced with even
    np.testing.assert_array_equal(ot, even_time)
    np.testing.assert_array_equal(of, even_flux)
    # even side is untouched
    np.testing.assert_array_equal(et, even_time)
    np.testing.assert_array_equal(ef, even_flux)


def test_odd_even_binning_and_generate_view_equivalence_and_shape():
    time, flux, period, t0, _ = _synthetic_series()
    odd, _ = phase_fold_and_sort_light_curve_odd_even(time, flux, period, t0)
    odd_time, odd_flux, _, _ = odd

    num_bins = 101
    bin_width = 0.05
    t_min, t_max = -period / 2, period / 2

    res_a = odd_even_binning(odd_time, odd_flux, num_bins, bin_width, t_min, t_max)
    res_b = generate_view(odd_time, odd_flux, num_bins, bin_width, t_min, t_max)

    # generate_view is a thin wrapper — same five outputs
    np.testing.assert_array_equal(res_a[0], res_b[0])
    np.testing.assert_array_equal(res_a[1], res_b[1])
    np.testing.assert_array_equal(res_a[2], res_b[2])
    np.testing.assert_array_equal(res_a[3], res_b[3])

    view, time_bins, view_var, bin_counts, bin_values = res_a
    assert view.shape == (num_bins,)
    assert time_bins.shape == (num_bins,)
    assert view_var.shape == (num_bins,)
    assert bin_counts.shape == (num_bins,)
    assert len(bin_values) == num_bins
    # in-transit bins (phase ~0) should contain the negative dip → median below 1
    center = num_bins // 2
    assert view[center] < 1.0


def test_impute_odd_even_views_replaces_missing_bins():
    n = 10
    odd_view = np.array([1.0, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0, np.nan, 1.0, 1.0])
    odd_var = np.full(n, 0.01)
    even_view = np.array([1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 0.95, 1.0, 1.0])
    even_var = np.full(n, 0.02)
    # non-zero bin counts for even for the bins we want to copy; 0 for odd at those indices.
    bin_counts_odd = np.array([2, 2, 0, 2, 2, 2, 2, 0, 2, 2], dtype="uint64")
    bin_counts_even = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype="uint64")
    # declare only bins [0, 1, 8, 9] as oot (outside transit)
    inds_oot = np.array([True, True, False, False, False, False, False, False, True, True])

    o_view, o_var, e_view, e_var, flag, bins_repl = impute_odd_even_views(
        odd_view.copy(), odd_var.copy(), bin_counts_odd.copy(),
        even_view.copy(), even_var.copy(), bin_counts_even.copy(), inds_oot,
    )

    # in-transit bins missing from odd (indices 2, 7) should now match even's values
    assert o_view[2] == 0.9
    assert o_view[7] == 0.95
    assert o_var[2] == 0.02
    assert "replaced bins" in flag
    # bookkeeping dict has the three expected keys
    assert set(bins_repl.keys()) == {"odd_to_even", "even_to_odd", "odd_even_miss"}


def test_computed_oot_it_var_oddeven_returns_four_scalars():
    # simple synthetic case: 5 bins, oot at ends, it in the middle
    bin_values_odd = [
        np.array([1.0, 1.01]),
        np.array([1.0, 0.99]),
        np.array([0.9, 0.91]),
        np.array([1.0, 1.01]),
        np.array([0.99, 1.0]),
    ]
    bin_values_even = [np.array([1.0, 1.0])] * 5
    inds_oot = np.array([0, 1, 3, 4])

    sot_o, sit_o, sot_e, sit_e = computed_oot_it_var_oddeven(bin_values_odd, bin_values_even, inds_oot)
    assert np.all(np.isfinite([sot_o, sit_o, sot_e, sit_e]))
    # even bins are constant so mad_std is 0 for both it and oot
    assert sot_e == 0.0
    assert sit_e == 0.0


def test_compute_oot_it_var_oddeven_splits_on_phase_window():
    # constructed arrays where in/out of transit have distinct noise scales
    rng = np.random.default_rng(42)
    phase = np.linspace(-0.5, 0.5, 200)
    odd_flux = np.where(np.abs(phase) < 0.1, 0.99 + rng.normal(0, 0.001, 200),
                        1.0 + rng.normal(0, 0.01, 200))
    even_flux = np.where(np.abs(phase) < 0.1, 0.99 + rng.normal(0, 0.001, 200),
                         1.0 + rng.normal(0, 0.01, 200))
    s_oot_o, s_it_o, s_oot_e, s_it_e = compute_oot_it_var_oddeven(
        phase, odd_flux, phase, even_flux, -0.1, 0.1,
    )
    # oot should be noisier than it for both
    assert s_oot_o > s_it_o
    assert s_oot_e > s_it_e


def test_perform_bin_replacement_oddeven_copies_bin_values_and_counts():
    bin_values_odd = [np.array([1.0, 1.1]), np.array([]), np.array([1.0])]
    bin_values_even = [np.array([1.0]), np.array([0.9, 0.95]), np.array([1.0])]
    bin_counts_odd = np.array([2, 0, 1], dtype="uint64")
    bin_counts_even = np.array([1, 2, 1], dtype="uint64")

    bins_repl = {
        "odd_to_even": np.array([False, True, False]),
        "even_to_odd": np.array([True, False, False]),
        "odd_even_miss": np.array([False, False, False]),
    }
    out_vo, out_co, out_ve, out_ce = perform_bin_replacement_oddeven(
        list(bin_values_odd), bin_counts_odd.copy(),
        list(bin_values_even), bin_counts_even.copy(),
        bins_repl,
    )
    # odd bin 1 got even's values
    np.testing.assert_array_equal(out_vo[1], bin_values_even[1])
    assert out_co[1] == 2
    # even bin 0 got odd's values
    np.testing.assert_array_equal(out_ve[0], bin_values_odd[0])
    assert out_ce[0] == 2
