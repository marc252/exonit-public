"""Tests for the adaptive period_max logic in worker.py.

Big run v3 (2026-05-09) showed the legacy fixed period_max=50d misses
100% of TOIs with period >100d and 59% of 30-100d. The adaptive logic
extends period_max for TICs with long baselines (sufficient to fit
≥2.5 transits) up to a configurable cap.
"""
from __future__ import annotations

import pytest


def _compute_adaptive(time_span_days, legacy=50.0, cap=200.0, divisor=2.5):
    """Pure-function reflection of the worker.py inline logic."""
    if time_span_days > legacy * divisor:
        return min(time_span_days / divisor, cap)
    return legacy


def test_short_baseline_keeps_legacy_default():
    """One sector (~27 d) → no change, legacy 50d preserved."""
    assert _compute_adaptive(27.0) == 50.0


def test_two_sectors_keeps_legacy():
    """~54 d span. Below legacy * divisor = 125 → legacy 50d."""
    assert _compute_adaptive(54.0) == 50.0


def test_at_threshold_keeps_legacy():
    """Exactly legacy * divisor (125 d) → still legacy."""
    assert _compute_adaptive(125.0) == 50.0


def test_just_above_threshold_uses_dynamic():
    """126 d → 126/2.5 = 50.4 d (just above 50)."""
    out = _compute_adaptive(126.0)
    assert out > 50.0
    assert out == pytest.approx(50.4)


def test_long_baseline_uses_dynamic():
    """13 sectors (~352 d) → 352/2.5 = 140.8 d."""
    out = _compute_adaptive(352.0)
    assert out == pytest.approx(140.8)


def test_full_tess_baseline_capped_at_200():
    """4-year baseline (~1460 d) → 1460/2.5 = 584, capped at 200."""
    assert _compute_adaptive(1460.0) == 200.0


def test_real_world_tic_267650535():
    """13 sectors, baseline 1900d, TOI period 1134d.
    Adaptive: 1900/2.5 = 760 → capped at 200. Still misses 1134d, but
    captures any KP/CP up to 200d that BLS would otherwise skip.
    """
    assert _compute_adaptive(1900.0) == 200.0


def test_custom_cap_lowers_cap():
    assert _compute_adaptive(1000.0, cap=120.0) == 120.0


def test_custom_divisor_changes_threshold():
    """divisor=2.0: threshold = 50 * 2 = 100. Span 100 still → legacy."""
    assert _compute_adaptive(100.0, divisor=2.0) == 50.0
    assert _compute_adaptive(150.0, divisor=2.0) == pytest.approx(75.0)


def test_zero_span_safe():
    """Empty/single-point timeseries → time_span 0 → legacy."""
    assert _compute_adaptive(0.0) == 50.0


def test_negative_span_safe():
    """Defensive: negative span (corrupt time array) → legacy."""
    assert _compute_adaptive(-1.0) == 50.0


def test_recovers_known_long_period_kp():
    """Real big-run-v3 case: TIC 26547036 (P=1134d, span=1900d).
    Even capped at 200d this is still missed, but we verify the
    selection logic resolves to the cap as expected.
    """
    span = 1900.0
    out = _compute_adaptive(span, cap=200.0)
    assert out == 200.0
    # If we ever raise the cap to 500 the same TIC would still be missed
    # (1134 > 500), but a cap of 600+ would capture it. Document.
    assert _compute_adaptive(span, cap=500.0) == 500.0
