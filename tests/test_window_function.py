"""Tests for window-function-aware detectability (exonit/scorecard.py).

Pure functions exercised with synthetic cadence arrays — no LC loading.
"""
import numpy as np

from exonit.scorecard import (
    compute_window_function,
    classify_detectability,
)

# 2-minute SPOC cadence in days
DT_2MIN = 2.0 / 1440.0
# A TESS BTJD epoch and its full-BJD equivalent (offset 2457000).
T0_BTJD = 1500.0
T0_BJD = T0_BTJD + 2457000.0


def _continuous(t_start, t_end, dt=DT_2MIN):
    return np.arange(t_start, t_end, dt)


def test_continuous_coverage_is_detectable():
    # 100-day continuous baseline, P=5d -> ~20 transits, all observed.
    t = _continuous(T0_BTJD, T0_BTJD + 100.0)
    wf = compute_window_function(t, period_days=5.0, epoch_bjd=T0_BJD,
                                 duration_hours=3.0)
    assert wf["observed_transits"] >= 19
    assert wf["in_transit_cadences"] > 100
    assert wf["detectability_class"] == "detectable"
    assert wf["duty_cycle"] > 0.9


def test_single_transit_is_impossible():
    # Baseline shorter than the period -> at most one transit folds.
    t = _continuous(T0_BTJD, T0_BTJD + 3.0)
    wf = compute_window_function(t, period_days=10.0, epoch_bjd=T0_BJD,
                                 duration_hours=3.0)
    assert wf["observed_transits"] <= 1
    assert wf["detectability_class"] == "impossible"


def test_two_observed_transits_is_marginal():
    # Two short observing windows that each catch exactly one transit of a
    # P=10d planet (transits at t0, t0+10, ...). Windows miss the others.
    win_a = _continuous(T0_BTJD - 0.1, T0_BTJD + 0.1)        # catches cycle 0
    win_b = _continuous(T0_BTJD + 29.9, T0_BTJD + 30.1)      # catches cycle 3
    t = np.concatenate([win_a, win_b])
    wf = compute_window_function(t, period_days=10.0, epoch_bjd=T0_BJD,
                                 duration_hours=3.0)
    assert wf["observed_transits"] == 2
    assert wf["detectability_class"] == "marginal"


def test_transits_falling_in_gaps_are_impossible():
    # P=10d, but the only observing window sits between transits (off by ~P/2),
    # so no transit is ever in-cadence.
    t = _continuous(T0_BTJD + 4.0, T0_BTJD + 6.0)  # around phase 0.4-0.6
    wf = compute_window_function(t, period_days=10.0, epoch_bjd=T0_BJD,
                                 duration_hours=3.0)
    assert wf["observed_transits"] == 0
    assert wf["detectability_class"] == "impossible"


def test_epoch_frame_autodetect_btjd_vs_bjd():
    # Passing the epoch in BTJD (already in LC frame) would misalign by
    # 2457000 days; the helper expects full BJD and converts. Verify a full-BJD
    # epoch folds correctly on a BTJD time array.
    t = _continuous(T0_BTJD, T0_BTJD + 50.0)
    wf = compute_window_function(t, period_days=5.0, epoch_bjd=T0_BJD,
                                 duration_hours=2.0)
    assert wf["observed_transits"] >= 9  # ~10 transits in 50d


def test_classify_detectability_buckets():
    assert classify_detectability(0, 0) == "impossible"
    assert classify_detectability(1, 50) == "impossible"
    assert classify_detectability(2, 50) == "marginal"
    assert classify_detectability(5, 2) == "marginal"   # too few in-transit pts
    assert classify_detectability(5, 50) == "detectable"


def test_invalid_inputs_return_unknown():
    t = _continuous(T0_BTJD, T0_BTJD + 10.0)
    wf = compute_window_function(t, period_days=np.nan, epoch_bjd=T0_BJD,
                                 duration_hours=3.0)
    assert wf["detectability_class"] == "unknown"
    wf2 = compute_window_function(np.array([]), period_days=5.0,
                                  epoch_bjd=T0_BJD, duration_hours=3.0)
    assert wf2["detectability_class"] == "unknown"
