"""Tests for exonit/cnn_ffi_periods.resolve_period."""
import numpy as np
import pandas as pd
import pytest

from exonit.cnn_ffi_periods import resolve_period, PeriodResult, PeriodSource, PeriodNotFound


@pytest.fixture
def fake_toi_catalog(tmp_path):
    df = pd.DataFrame({
        "tid": [100, 200],
        "pl_orbper": [3.5, 12.1],
        "pl_tranmid": [1356.0, 1358.5],
        "pl_trandurh": [2.4, 5.0],
    })
    p = tmp_path / "toi.csv"
    df.to_csv(p, index=False)
    return p


def test_resolve_period_real_positive_uses_toi(fake_toi_catalog):
    row = pd.Series({"tic_id": 100, "label": "PC", "is_synthetic_row": False, "is_negative_row": False})
    res = resolve_period(row, time=None, flux=None, toi_catalog_path=fake_toi_catalog)
    assert res.source == PeriodSource.TOI
    assert res.period == pytest.approx(3.5)
    assert res.t0 == pytest.approx(1356.0)
    assert res.duration_days == pytest.approx(2.4 / 24)


def test_resolve_period_synthetic_uses_inj_columns():
    row = pd.Series({
        "tic_id": 999, "label": "synthetic_pos", "is_synthetic_row": True,
        "is_negative_row": False, "inj_period_days": 4.2, "inj_t0_btjd": 1400.5,
        "inj_duration_hours": 3.0,
    })
    res = resolve_period(row, time=None, flux=None, toi_catalog_path=None)
    assert res.source == PeriodSource.SYNTHETIC
    assert res.period == 4.2
    assert res.t0 == 1400.5
    assert res.duration_days == pytest.approx(3.0 / 24)


def test_resolve_period_negative_uses_bls():
    """Negative row with synthetic transit injected → BLS detects it."""
    rng = np.random.default_rng(42)
    n = 5000
    time = np.linspace(0, 27, n)
    flux = np.ones_like(time) + 0.0005 * rng.standard_normal(n)
    period = 3.7
    depth = 0.03
    t0 = 2.1
    duration = 0.15
    # Single transit injection
    phase = np.mod(time - t0, period) / period
    in_transit = (phase < duration / period)
    flux[in_transit] -= depth
    row = pd.Series({"tic_id": 1, "label": "FP", "is_synthetic_row": False, "is_negative_row": True})
    res = resolve_period(row, time=time, flux=flux, toi_catalog_path=None)
    assert res.source == PeriodSource.BLS
    # Check it found a plausible signal (period within 10% or 1/2 harmonic within 5%)
    period_frac_err = abs(res.period - period) / period
    harmonic_frac_err = abs(res.period - period / 2) / (period / 2)
    is_fundamental_or_harmonic = period_frac_err < 0.10 or harmonic_frac_err < 0.05
    assert is_fundamental_or_harmonic, f"BLS period={res.period}, expected≈{period} or {period/2}"
    assert res.snr is not None and res.snr > 5.0


def test_resolve_period_negative_no_bls_signal_raises():
    """Insufficient data points → PeriodNotFound."""
    time = np.linspace(0, 27, 500)  # Only 500 points, below 1000 threshold
    flux = np.ones_like(time)
    row = pd.Series({"tic_id": 2, "label": "FP", "is_synthetic_row": False, "is_negative_row": True})
    with pytest.raises(PeriodNotFound):
        resolve_period(row, time=time, flux=flux, toi_catalog_path=None)
