"""Layer 2 functional tests — real BLS/vetting pipeline on cached lightcurves.

These are SLOW tests that run the actual ExoNit worker processing pipeline
(BLS search, vetting) on pre-cached .npy lightcurve data. They skip
automatically if test data is not available.

To get test data:
    python3 scripts/download_test_data.py

Data format per TIC:
    tests/data/{tic_id}/time.npy   — BTJD times (float64)
    tests/data/{tic_id}/flux.npy   — normalized flux (float64, ~1.0)
    tests/data/{tic_id}/meta.json  — metadata dict
"""

import json
import numpy as np
import pytest

from pathlib import Path

from bls import run_bls
from vetting import vet_candidate

DATA_DIR = Path(__file__).parent / "data"


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _load_test_data(tic_id):
    """Load cached lightcurve. Returns (time, flux, meta) or skips test."""
    tic_dir = DATA_DIR / tic_id
    if not tic_dir.exists():
        pytest.skip(
            f"Test data not available for TIC {tic_id}. "
            "Run: python3 scripts/download_test_data.py"
        )
    time_arr = np.load(tic_dir / "time.npy")
    flux_arr = np.load(tic_dir / "flux.npy")
    meta = json.loads((tic_dir / "meta.json").read_text())
    return time_arr, flux_arr, meta


def _is_period_match(detected, expected, tolerance=0.10):
    """Check if detected period matches expected within tolerance,
    including harmonic multiples up to ±5× (same set as the production
    scorecard in exonit/scorecard.py: HARMONIC_FACTORS_SUB/SUPER)."""
    for ratio in [1.0, 0.5, 1.0 / 3, 0.25, 0.2, 2.0, 3.0, 4.0, 5.0]:
        if abs(detected - expected * ratio) / (expected * ratio) <= tolerance:
            return True
    return False


# ------------------------------------------------------------------ #
# Test 2.1: Pipeline real E2E (TIC 150430484)
# ------------------------------------------------------------------ #

@pytest.mark.slow
def test_pipeline_real_e2e():
    """Full BLS pipeline on real TIC 150430484 data — must detect TOI-157b."""
    tic_id = "150430484"
    expected_period = 5.154

    time_arr, flux_arr, meta = _load_test_data(tic_id)

    result = run_bls(time_arr, flux_arr,
                     period_min=0.5, period_max=50.0, n_periods=50000)

    # Must detect at least one significant signal
    assert result["snr"] >= 7, (
        f"Best BLS candidate SNR={result['snr']:.1f}, expected >= 7"
    )

    # Best period must match known planet (or harmonic)
    assert _is_period_match(result["period"], expected_period), (
        f"Best period {result['period']:.3f}d does not match "
        f"expected {expected_period}d (or 1:2/2:1 harmonic)"
    )


# ------------------------------------------------------------------ #
# Test 2.2: Anti-fill_gaps real (TIC 254113311)
# ------------------------------------------------------------------ #

@pytest.mark.slow
def test_anti_fill_gaps_real():
    """Cached data for TIC 254113311 must not have inflated point count."""
    tic_id = "254113311"

    time_arr, flux_arr, meta = _load_test_data(tic_id)

    n_points = len(time_arr)
    n_sectors = meta["n_sectors"]
    # 65000 per sector covers both 120s (~13K/sector) and 20s (~65K/sector) cadence
    max_allowed = n_sectors * 65000

    assert n_points <= max_allowed, (
        f"n_points={n_points} exceeds n_sectors*30000={max_allowed} — "
        "fill_gaps may not have been applied correctly"
    )


# ------------------------------------------------------------------ #
# Test 2.3: Recall batch (parametrized over 3 stars)
# ------------------------------------------------------------------ #

@pytest.mark.slow
@pytest.mark.parametrize("tic_id, expected_period", [
    ("150430484", 5.154),
    ("140691463", 2.084),
    ("142105158", 2.936),
])
def test_recall_batch(tic_id, expected_period):
    """BLS must recover known planet period for reference stars."""
    time_arr, flux_arr, meta = _load_test_data(tic_id)

    result = run_bls(time_arr, flux_arr,
                     period_min=0.5, period_max=50.0, n_periods=50000)

    # At least one candidate must match the expected period
    assert _is_period_match(result["period"], expected_period), (
        f"TIC {tic_id}: best period {result['period']:.3f}d does not match "
        f"expected {expected_period}d (or 1:2/2:1 harmonic). "
        f"SNR={result['snr']:.1f}"
    )
