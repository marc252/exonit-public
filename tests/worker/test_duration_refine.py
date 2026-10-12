"""Unit tests for worker/duration_refine.refine_duration (P2)."""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "worker", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from duration_refine import refine_duration  # noqa: E402


def _synthetic(n_days=27.0, cadence_d=2.0 / 1440, period=3.1737, t0=1.0,
               duration=1.87 / 24, depth=0.004, noise=5e-4, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, n_days, cadence_d)
    f = 1.0 + rng.normal(0.0, noise, t.size)
    x = (t - t0 + 0.5 * period) % period - 0.5 * period
    f[np.abs(x) < duration / 2] -= depth
    return t, f


def test_recovers_duration_and_t0_from_coarse_start():
    P, t0, dur = 3.1737, 1.0, 1.87 / 24
    t, f = _synthetic(period=P, t0=t0, duration=dur)
    out = refine_duration(t, f, P, t0 + 0.6 * dur, 2 * dur)
    assert abs(out["duration"] - dur) / dur < 0.15
    assert abs(out["t0"] - t0) < 0.1 * out["duration"]
    assert abs(out["depth_ppm"] - 4000) < 600
    assert out["chi2_improvement"] > 0


def test_flat_noise_is_stable():
    rel = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        t = np.arange(0.0, 27.0, 2.0 / 1440)
        f = 1.0 + rng.normal(0.0, 5e-4, t.size)
        out = refine_duration(t, f, 3.17, 1.0, 0.08)
        # No real dip: improvement over the input box must stay at noise level.
        assert out["chi2_improvement"] < 60.0
        rel.append((out["duration"] - 0.08) / 0.08)
    # Signed bias: noise fits may shrink the box but must not systematically
    # inflate it.
    assert float(np.mean(rel)) < 0.25


def test_all_too_few_points_returns_unchanged():
    t = np.linspace(0, 1, 5)
    out = refine_duration(t, np.ones(5), 3.0, 0.5, 0.1)
    assert out["duration"] == 0.1 and out["t0"] == 0.5
    assert out["chi2_improvement"] == 0.0


def test_nan_handling_matches_clean_data():
    P, t0, dur = 3.1737, 1.0, 1.87 / 24
    t, f = _synthetic(period=P, t0=t0, duration=dur)
    f2 = f.copy()
    f2[::50] = np.nan
    t2 = t.copy()
    t2[7::97] = np.nan
    out = refine_duration(t2, f2, P, t0 + 0.3 * dur, 2 * dur)
    assert np.isfinite(out["duration"]) and np.isfinite(out["t0"])
    assert abs(out["duration"] - dur) / dur < 0.2


def test_deterministic():
    t, f = _synthetic()
    a = refine_duration(t, f, 3.1737, 1.05, 0.16)
    b = refine_duration(t, f, 3.1737, 1.05, 0.16)
    assert a == b


def test_tic4206066_like_30min_cadence():
    P, t0, dur = 3.1737, 1.2, 1.87 / 24
    t, f = _synthetic(n_days=21.25, cadence_d=30.0 / 1440, period=P, t0=t0,
                      duration=dur, depth=0.006, noise=8e-4, seed=11)
    assert 1000 <= t.size <= 1100
    out = refine_duration(t, f, P, t0 - 0.07, 3.81 / 24, cadence=30.0 / 1440)
    assert out["duration"] * 24 <= 2.2


def test_runtime_under_50ms():
    t, f = _synthetic(n_days=27.0, cadence_d=2.0 / 1440 * 0.8)
    assert t.size > 15000
    refine_duration(t, f, 3.1737, 1.0, 0.16)
    t0_ = _time.perf_counter()
    refine_duration(t, f, 3.1737, 1.0, 0.16)
    assert _time.perf_counter() - t0_ < 0.05


def test_real_shape_tic4206066_box_2p4h_early_ensemble():
    """3.81 h box, t0 2.4 h early; 40 seeds (observed: median |err| ~6 %,
    mean signed ~-1 %, median |t0 err| ~5 min)."""
    P, t0, dur = 3.1737, 1.3, 1.87 / 24
    errs, terrs = [], []
    for seed in range(40):
        t, f = _synthetic(n_days=21.7, cadence_d=30.0 / 1440, period=P, t0=t0,
                          duration=dur, depth=600e-6, noise=400e-6, seed=seed)
        assert 950 <= t.size <= 1100
        out = refine_duration(t, f, P, t0 - 2.4 / 24, 3.81 / 24,
                              cadence=30.0 / 1440)
        errs.append((out["duration"] - dur) / dur)
        terrs.append(abs(out["t0"] - t0) * 1440)
    errs = np.array(errs)
    assert np.median(np.abs(errs)) < 0.15
    assert abs(np.mean(errs)) < 0.10          # no longer-box / shorter-box bias
    assert np.median(terrs) < 15.0


def test_long_transit_12h_at_40d_from_20h_start():
    P, t0, dur = 40.0, 5.0, 12.0 / 24
    rng = np.random.default_rng(1)
    t = np.arange(0.0, 120.0, 30.0 / 1440)
    f = 1.0 + rng.normal(0.0, 5e-4, t.size)
    x = (t - t0 + 0.5 * P) % P - 0.5 * P
    f[np.abs(x) < dur / 2] -= 0.004
    out = refine_duration(t, f, P, t0 - 0.3, 20.0 / 24)
    assert abs(out["duration"] - dur) / dur < 0.15
    assert abs(out["t0"] - t0) < 0.1 * dur
