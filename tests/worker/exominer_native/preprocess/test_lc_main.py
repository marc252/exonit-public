"""Fast unit tests for ``worker.exominer_native.preprocess.lc_main``.

The module's entry point :func:`preprocess_tce` is a thick orchestrator
that consumes already-phase-folded timeseries and emits a flat feature
dict ready for the ExoMiner ONNX model.  End-to-end parity against the
oracle's TFRecord writer is deferred to Phase D — here we exercise the
import surface, the ported pure-compute helpers, and a minimum viable
invocation of the top-level entry point on a synthetic transit.
"""
from __future__ import annotations

import numpy as np
import pytest

from worker.exominer_native.preprocess import lc_main
from worker.exominer_native.preprocess.lc_main import (
    MAP_VIEWS_TO_OLD_NAMES,
    NORM_SIGMA_EPS,
    create_example_stats,
    find_intransit_cadences,
    gap_intransit_cadences_other_tces,
    generate_centroid_binned_views,
    generate_flux_binned_views,
    generate_flux_trend_binned_views,
    generate_momentum_dump_views,
    generate_odd_even_binned_views,
    generate_weak_secondary_binned_views,
    phase_fold_timeseries,
    preprocess_tce,
    set_duration_gap_for_tce,
)


# ---------------------------------------------------------------------------
# Smoke: module loads, exposes the documented public names
# ---------------------------------------------------------------------------


def test_module_imports_and_exposes_core_names():
    for name in [
        "preprocess_tce",
        "phase_fold_timeseries",
        "generate_flux_binned_views",
        "generate_flux_trend_binned_views",
        "generate_odd_even_binned_views",
        "generate_weak_secondary_binned_views",
        "generate_centroid_binned_views",
        "generate_momentum_dump_views",
        "generate_view_momentum_dump",
        "create_example_stats",
        "find_intransit_cadences",
        "gap_intransit_cadences_other_tces",
        "set_duration_gap_for_tce",
        "MAP_VIEWS_TO_OLD_NAMES",
        "NORM_SIGMA_EPS",
    ]:
        assert hasattr(lc_main, name), name


def test_norm_sigma_eps_matches_oracle_constant():
    assert NORM_SIGMA_EPS == 1e-12


def test_map_views_to_old_names_has_expected_onnx_keys():
    # Sanity: a handful of the ExoMiner ONNX feature names that the
    # downstream normalizer (C.14) expects must be emitted.
    values = set(MAP_VIEWS_TO_OLD_NAMES.values())
    for expected in [
        "global_flux_view",
        "local_flux_view",
        "global_flux_view_fluxnorm",
        "local_flux_view_fluxnorm",
        "local_flux_odd_view_fluxnorm",
        "local_flux_even_view_fluxnorm",
        "local_weak_secondary_view",
        "local_weak_secondary_view_selfnorm",
        "global_centr_view",
        "local_centr_view",
        "local_momentum_dump_view",
    ]:
        assert expected in values


# ---------------------------------------------------------------------------
# set_duration_gap_for_tce — direct pure-compute helper
# ---------------------------------------------------------------------------


def test_set_duration_gap_without_secondary():
    primary, secondary = set_duration_gap_for_tce(0.2, 3.0, 2.0)
    assert primary == pytest.approx(0.4)  # 2*0.2 < 3.0 → transit_factor*duration
    assert secondary is None


def test_set_duration_gap_caps_by_period():
    primary, secondary = set_duration_gap_for_tce(0.2, 0.1, 5.0)
    # transit_factor*duration = 1.0 > period 0.1 → min() clamps to period
    assert primary == pytest.approx(0.1)
    assert secondary is None


def test_set_duration_gap_with_secondary_equals_primary():
    primary, secondary = set_duration_gap_for_tce(0.2, 3.0, 2.0, tce_secondary_offset=1.0)
    # max( min(0.4, 2*1-0.2, 3.0), 0.2 ) = max( min(0.4, 1.8, 3.0), 0.2 ) = 0.4
    assert primary == pytest.approx(0.4)
    assert secondary == primary


# ---------------------------------------------------------------------------
# find_intransit_cadences + gap_intransit_cadences_other_tces on tiny inputs
# ---------------------------------------------------------------------------


class _FakeTCEs:
    """Minimal pandas.DataFrame-like shim that supports ``iterrows``."""

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def iterrows(self):
        for i, r in enumerate(self._rows):
            yield i, r


def test_find_intransit_cadences_flags_transit_points():
    time_arrs = [np.linspace(0.0, 10.0, 1001)]
    tces = _FakeTCEs(
        [
            {"tce_duration": 0.2, "tce_period": 2.0, "tce_time0bk": 1.0,
             "tce_maxmesd": None, "uid": "A"},
        ]
    )
    out = find_intransit_cadences(tces, time_arrs, transit_duration_factor=2.0)
    assert len(out) == 1
    assert out[0].shape == (1, 1001)
    # Epoch 1.0 with period 2.0 → transits at ~1, 3, 5, 7, 9
    # Duration 0.4 (2 × 0.2) → ~40 in-transit samples per transit; 5 transits × 40 = 200
    assert out[0][0].sum() > 100


def test_gap_intransit_cadences_other_tces_sets_other_transits_to_nan():
    # Two TCEs in the target; gap the second one → first TCE's transits stay,
    # but values at the second TCE's cadences become NaN in the flux timeseries.
    target_it_arr = [np.array([[True, False, False, True], [False, True, True, False]])]
    data = {"all_flux": [np.array([10.0, 20.0, 30.0, 40.0], dtype=float)]}
    out = gap_intransit_cadences_other_tces(
        target_it_arr, idx_tce=0, data=data, gap_keep_overlap=True, impute_gaps=False,
    )
    # Cadences [1,2] belonged to the other TCE → NaN
    assert np.isnan(out["all_flux"][0][1])
    assert np.isnan(out["all_flux"][0][2])
    assert out["all_flux"][0][0] == 10.0
    assert out["all_flux"][0][3] == 40.0


# ---------------------------------------------------------------------------
# preprocess_tce — minimal viable synthetic invocation
# ---------------------------------------------------------------------------


def _make_synthetic_phase_folded(period=3.0, duration=0.2, depth=0.01, n_bins=5000):
    """Build a phase-folded data dict matching ``phase_fold_timeseries`` output.

    All timeseries share the same phase grid so ``generate_*_binned_views``
    can run to completion.  The flux carries a boxy transit at phase 0.
    """
    rng = np.random.default_rng(0)
    phase = np.linspace(-period / 2, period / 2, n_bins)

    # Boxy transit
    flux = np.ones_like(phase) + rng.normal(0.0, 1e-4, n_bins)
    flux[np.abs(phase) < duration / 2] -= depth

    trend = np.ones_like(phase) + rng.normal(0.0, 1e-4, n_bins)

    # Weak secondary around -period/4 (arbitrary offset)
    flux_wks = np.ones_like(phase) + rng.normal(0.0, 1e-4, n_bins)

    centroid = np.full_like(phase, 0.01)  # tiny constant offset
    centroid += rng.normal(0.0, 1e-5, n_bins)

    # Synthesize odd/even by splitting the main flux in two interleaved slices
    odd_time = phase[0::2]
    even_time = phase[1::2]
    odd_flux = flux[0::2]
    even_flux = flux[1::2]

    n_phases = 3
    flux_unfolded_values = np.tile(flux, (n_phases, 1))
    flux_trend_unfolded_values = np.tile(trend, (n_phases, 1))
    phase_stack = np.tile(phase, (n_phases, 1))

    return {
        "flux": (phase, flux, 5),
        "flux_unfolded": (phase_stack, flux_unfolded_values, n_phases),
        "flux_trend": (phase, trend, 5),
        "flux_trend_unfolded": (phase_stack, flux_trend_unfolded_values, n_phases),
        "flux_odd": (odd_time, odd_flux, 3),
        "flux_even": (even_time, even_flux, 2),
        "flux_weak_secondary": (phase, flux_wks, 5),
        "centroid_offset_distance_to_target": (phase, centroid, 5),
        "momentum_dump": (phase, np.zeros_like(phase), 0),
    }


def _make_synthetic_tce(period=3.0, duration=0.2):
    return {
        "tce_period": period,
        "tce_duration": duration,
        "tce_time0bk": 0.0,
        "tce_maxmesd": -period / 4,  # negative offset → primary gap in wks
        "uid": "T-smoke-1",
        "label": "PC",
    }


def _make_synthetic_config(period=3.0, get_momentum_dump=False):
    return {
        "num_bins_glob": 301,
        "num_bins_loc": 31,
        "num_durations": 5,
        "bin_width_factor_glob": 1 / 301,
        "bin_width_factor_loc": 0.16,
        "get_momentum_dump": get_momentum_dump,
        "random_seed": 42,
    }


def test_preprocess_tce_returns_expected_schema_and_feature_names():
    data = _make_synthetic_phase_folded()
    tce = _make_synthetic_tce()
    config = _make_synthetic_config()

    out = preprocess_tce(data, tce, config)

    # Top-level keys
    assert set(out.keys()) >= {
        "views", "views_var", "num_transits", "stats",
        "example_stats", "odd_even_flag", "tce_table",
    }
    assert isinstance(out["views"], dict)
    assert isinstance(out["views_var"], dict)

    # Must include the canonical ExoMiner ONNX feature names
    for expected in [
        "global_flux_view",
        "local_flux_view",
        "global_flux_view_fluxnorm",
        "local_flux_view_fluxnorm",
        "local_flux_odd_view",
        "local_flux_even_view",
        "local_weak_secondary_view",
        "local_weak_secondary_view_selfnorm",
        "global_centr_view",
        "local_centr_view",
    ]:
        assert expected in out["views"], expected
        assert f"{expected}_var" in out["views_var"], expected

    # Global flux view shape matches num_bins_glob
    assert out["views"]["global_flux_view"].shape == (config["num_bins_glob"],)
    assert out["views"]["local_flux_view"].shape == (config["num_bins_loc"],)

    # Stat scalars are finite floats
    for stat_name, stat_val in out["stats"].items():
        assert np.isfinite(stat_val), stat_name

    # odd_even_flag is a status string; the oracle emits the literal
    # ``'ok'`` when happy and a variety of descriptive strings otherwise
    # (e.g. ``'replaced bins N (odd) M (even) '``).  We just assert the
    # field is a non-empty string.
    assert isinstance(out["odd_even_flag"], str)
    assert out["odd_even_flag"]

    # tce_table carries scalar fields straight through
    assert out["tce_table"]["tce_period"] == pytest.approx(3.0)
    assert out["tce_table"]["uid"] == "T-smoke-1"


def test_preprocess_tce_imputes_non_finite_view_values():
    """All-NaN flux must NOT raise: lc_main mirrors the NASA exclusion logic
    and imputes the view from phase-folded statistics (median/mad_std, or
    zeros when those are not finite). This test used to expect a raise that
    the port never implemented — stale expectation fixed 2026-06-09."""
    data = _make_synthetic_phase_folded()
    tce = _make_synthetic_tce()
    config = _make_synthetic_config()

    phase = data["flux"][0]
    data["flux"] = (phase, np.full_like(phase, np.nan), 0)
    data["flux_unfolded"] = (
        data["flux_unfolded"][0],
        np.full_like(data["flux_unfolded"][1], np.nan),
        data["flux_unfolded"][2],
    )

    out = preprocess_tce(data, tce, config)
    for name, view in out["views"].items():
        view = np.asarray(view)
        assert np.all(np.isfinite(view)), f"non-finite values in view {name}"


def test_preprocess_tce_skips_momentum_dump_when_disabled():
    data = _make_synthetic_phase_folded()
    tce = _make_synthetic_tce()
    config = _make_synthetic_config(get_momentum_dump=False)

    out = preprocess_tce(data, tce, config)

    # momentum_dump_local is only populated when get_momentum_dump=True
    assert "local_momentum_dump_view" not in out["views"]


# ---------------------------------------------------------------------------
# Individual view generators — smoke-level shape checks
# ---------------------------------------------------------------------------


def test_generate_flux_binned_views_produces_expected_keys():
    data = _make_synthetic_phase_folded()
    tce = _make_synthetic_tce()
    config = _make_synthetic_config()

    binned, glob_stats, loc_stats = generate_flux_binned_views(data, tce, config)

    for key in [
        "flux_global", "flux_local",
        "flux_global_norm", "flux_local_norm",
        "flux_global_stat_abs_min", "flux_local_stat_abs_min",
    ]:
        assert key in binned

    assert glob_stats["sigma"] > 0
    assert loc_stats["sigma"] > 0


def test_generate_centroid_binned_views_produces_global_and_local():
    data = _make_synthetic_phase_folded()
    tce = _make_synthetic_tce()
    config = _make_synthetic_config()

    binned = generate_centroid_binned_views(data, tce, config)

    assert "centroid_offset_distance_to_target_global" in binned
    assert "centroid_offset_distance_to_target_local" in binned
    assert "centroid_offset_distance_to_target_global_stat_abs_max" in binned
    assert "centroid_offset_distance_to_target_local_stat_abs_max" in binned

    gbt, gvals, gvars, gn = binned["centroid_offset_distance_to_target_global"]
    assert gvals.shape == (config["num_bins_glob"],)
