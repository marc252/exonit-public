"""Tests for cross-source photometric confirmation (worker/cross_source.py)."""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))

from cross_source import (
    fixed_period_box_fit,
    _classify_agreement,
    run_cross_source_check,
)

DT = 30.0 / 1440.0  # 30-min FFI cadence in days


def _synthetic_lc(period, t0, duration, depth_frac, span=80.0, noise=2e-4,
                  seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(t0, t0 + span, DT)
    f = 1.0 + rng.normal(0, noise, t.size)
    phase = ((t - t0) / period + 0.5) % 1.0 - 0.5
    in_tr = np.abs(phase) <= (duration / period) / 2.0
    f[in_tr] -= depth_frac
    return t, f


def test_box_fit_recovers_injected_transit():
    t, f = _synthetic_lc(period=6.0, t0=1500.0, duration=0.12,
                         depth_frac=2e-3, noise=2e-4)
    fit = fixed_period_box_fit(t, f, 6.0, 1500.0, 0.12)
    assert fit["is_dip"]
    assert 1500 < fit["depth_ppm"] < 2500   # ~2000 ppm injected
    assert fit["snr"] > 10
    assert fit["n_in_transit"] > 5


def test_box_fit_flat_lc_no_signal():
    rng = np.random.default_rng(1)
    t = np.arange(1500.0, 1580.0, DT)
    f = 1.0 + rng.normal(0, 2e-4, t.size)
    fit = fixed_period_box_fit(t, f, 6.0, 1500.0, 0.12)
    assert fit["snr"] < 5            # no transit at this ephemeris


def test_box_fit_wrong_ephemeris_misses():
    # Inject at t0=1500 but query a half-period-shifted epoch -> out of transit.
    t, f = _synthetic_lc(period=6.0, t0=1500.0, duration=0.12, depth_frac=2e-3)
    fit = fixed_period_box_fit(t, f, 6.0, 1503.0, 0.12)  # shifted by P/2
    assert fit["snr"] < 5


def test_box_fit_too_few_points_returns_null():
    t = np.array([1500.0, 1500.1, 1500.2])
    f = np.array([1.0, 0.99, 1.0])
    fit = fixed_period_box_fit(t, f, 6.0, 1500.0, 0.12)
    assert fit["snr"] == 0.0 and fit["n_in_transit"] == 0


def test_classify_agreement():
    assert _classify_agreement(8.0, 1.0, True) is True
    assert _classify_agreement(3.0, 1.0, True) is False     # snr too low
    assert _classify_agreement(8.0, 1.0, False) is False    # not a dip
    assert _classify_agreement(8.0, 10.0, True) is False     # depth mismatch
    assert _classify_agreement(8.0, None, True) is True      # no primary depth


def test_run_cross_source_mutates_candidates_via_injected_fetch():
    t, f = _synthetic_lc(period=6.0, t0=1500.0, duration=0.12, depth_frac=2e-3)
    cands = [{"period_days": 6.0, "t0": 1500.0, "duration_days": 0.12,
              "depth_ppm": 2000.0}]

    def fake_fetch(server_url, tic_id, *, token, timeout):
        return t, f, np.zeros_like(t)

    # identity detrend (already normalized)
    run_cross_source_check(
        tic_id="123", candidates=cands, data_source="spoc",
        config={"server_url": "http://x"}, logger=_Logger(),
        fetch_fn=fake_fetch, detrend_fn=lambda tt, ff, **k: ff)
    c = cands[0]
    assert c["cross_source_name"] == "ffi"
    assert c["cross_source_agreement"] is True
    assert c["cross_source_snr"] > 10


def test_run_cross_source_skips_ffi_primary():
    cands = [{"period_days": 6.0, "t0": 1500.0, "duration_days": 0.12,
              "depth_ppm": 2000.0}]
    run_cross_source_check(
        tic_id="123", candidates=cands, data_source="ffi",
        config={"server_url": "http://x"}, logger=_Logger(),
        fetch_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    assert "cross_source_name" not in cands[0]


def test_run_cross_source_no_ffi_available():
    cands = [{"period_days": 6.0, "t0": 1500.0, "duration_days": 0.12,
              "depth_ppm": 2000.0}]
    run_cross_source_check(
        tic_id="123", candidates=cands, data_source="spoc",
        config={"server_url": "http://x"}, logger=_Logger(),
        fetch_fn=lambda *a, **k: None)
    assert "cross_source_name" not in cands[0]


class _Logger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
