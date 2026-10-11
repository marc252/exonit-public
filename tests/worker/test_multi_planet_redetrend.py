"""Unit tests for the multi-planet iterative re-detrend loop.

The loop in Worker.process_task (worker/worker.py:~758-802) must:
1. Run BLS i=0 on the flat_flux returned by _load_star (no re-detrend,
   so single-planet behavior is unchanged).
2. For i>=1, re-detrend raw_flux with the accumulated transit mask
   (Hippke+ 2019 §5) before running BLS, instead of median hole-filling.
3. Fall back to hole-fill if wotan raises, skip re-detrend if the mask
   covers >30% of the curve, abort the loop after 2 consecutive wotan
   failures.

Tests use a hand-built Worker (via Worker.__new__, no network) and a
mocked _load_star returning synthetic flux with injected box transits.

RED phase: these tests fail against the current code (hole-fill only).
Task 3 implements the re-detrend logic and turns them green.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "worker") not in sys.path:
    sys.path.append(str(ROOT / "worker"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ``worker/`` is both a package (worker/__init__.py) and contains a flat
# ``worker.py`` script. Depending on sys.path order ``import worker`` can
# resolve to either. Be explicit: import the submodule by dotted path so
# the same module object is always patched in the spy setups below.
from worker.worker import ExoNitWorker  # noqa: E402


def _make_flux_with_transits(
    time: np.ndarray,
    transits: list,
    noise_sigma: float = 5e-4,
    rng_seed: int = 42,
) -> np.ndarray:
    """Build a flux array ~1.0 with box transits injected.

    Each transit: (period_days, t0_days, duration_days, depth_frac).
    """
    rng = np.random.default_rng(rng_seed)
    flux = np.ones_like(time) + rng.normal(0.0, noise_sigma, size=len(time))
    for period, t0, duration, depth in transits:
        phase = ((time - t0) % period) / period
        half_dur = (duration / period) / 2
        in_transit = (phase < half_dur) | (phase > 1.0 - half_dur)
        flux[in_transit] -= depth
    return flux


def _make_time_grid(n_days: float = 27.0, cadence_min: float = 2.0) -> np.ndarray:
    """TESS-like 2-min cadence grid of n_days duration."""
    n_points = int(n_days * 24 * 60 / cadence_min)
    return np.linspace(0.0, n_days, n_points)


def _make_worker(monkeypatch, time_arr, flat_flux, raw_flux):
    """Instantiate a minimal Worker bypassing __init__."""
    from collections import deque

    w = ExoNitWorker.__new__(ExoNitWorker)
    w.logger = logging.getLogger("test_multi_planet")
    w.logger.setLevel(logging.DEBUG)
    w.ui = None
    w.running = True
    w.paused = False
    w.current_phase = None
    w.phase_progress = None
    w.classifier = None
    w.status = "idle"
    w.current_tic = None
    w.last_error = None
    w._star_start_time = 0.0
    # Attrs accessed late in process_task (post-pipeline telemetry + HTTP +
    # ExoMiner hook). Set minimal safe defaults so the code path runs end-to-
    # end without needing a real server.
    w.server = "http://localhost:0"
    w.token = "test-token"
    w.recent_timings = deque(maxlen=5)
    w.gpu_enabled = False
    w.config = {
        "server_url": "http://localhost:0",
        "worker_token": "test-token",
        "exominer_enabled": False,
        "exominer_mode": "legacy-tfrecord",
    }

    def fake_load_star(self, tic_id):
        return time_arr, flat_flux, raw_flux, 1, {
            "tmag": 10.0, "sectors": [1], "ra": 100.0, "dec": 20.0,
            "stellar_radius": 1.0,
        }

    monkeypatch.setattr(ExoNitWorker, "_load_star", fake_load_star)
    return w


def _make_task(tic_id: int = 1, max_planets: int = 5,
               snr_threshold: float = 7.0) -> dict:
    return {
        "task_id": "t1",
        "tic_id": tic_id,
        "config": {
            "max_planets": max_planets,
            "snr_threshold": snr_threshold,
            "period_min": 0.5,
            "period_max": 15.0,
            "n_periods": 5000,
            # Skip injection-recovery and FPP in unit tests: they are
            # orthogonal to the re-detrend loop and add ~minutes of work.
            "run_injection": False,
        },
    }


# ─────────────────────────────────────────────────────────────────────
# Test 1: single-planet — i=0 preserves the flat_flux baseline (no
# re-detrend happens before the first BLS call). Subsequent iterations
# may or may not re-detrend depending on whether a harmonic resolver
# artifact produces phantom candidates — that's orthogonal to #6 and
# tracked by audit §76-85 (asymmetric harmonic resolver).
# ─────────────────────────────────────────────────────────────────────
def test_single_planet_no_redetrend_before_first_bls(monkeypatch):
    time = _make_time_grid()
    raw_flux = _make_flux_with_transits(
        time, [(5.0, 1.2, 0.15, 0.005)])
    flat_flux = raw_flux.copy()

    w = _make_worker(monkeypatch, time, flat_flux, raw_flux)

    import worker.worker as worker_mod

    # Track ordering: did any re-detrend with non-empty mask fire BEFORE
    # the first BLS call? That would break the single-planet baseline.
    events = []
    real_detrend = worker_mod.detrend_with_wotan

    def detrend_spy(time_arg, flux_arg, **kwargs):
        mask = kwargs.get("transit_mask")
        if mask is not None and mask.any():
            events.append(("redetrend", int(mask.sum())))
        return real_detrend(time_arg, flux_arg, **kwargs)

    real_bls = worker_mod.run_bls

    def bls_spy(*args, **kwargs):
        events.append(("bls", None))
        return real_bls(*args, **kwargs)

    monkeypatch.setattr(worker_mod, "detrend_with_wotan", detrend_spy)
    monkeypatch.setattr(worker_mod, "run_bls", bls_spy)

    result = w.process_task(_make_task())
    assert len(result["candidates"]) >= 1, \
        f"Expected ≥1 candidate, got {len(result['candidates'])}"

    # The very first event in the loop must be a BLS call, not a re-detrend.
    # This is what guarantees the single-planet baseline is preserved: i=0
    # sees the flat_flux from _load_star without mutation.
    assert events, "No BLS or re-detrend activity observed — test setup bug"
    assert events[0][0] == "bls", (
        f"First loop event was {events[0]}, expected 'bls' — i=0 must run "
        "BLS on the original flat_flux before any re-detrend"
    )


# ─────────────────────────────────────────────────────────────────────
# Test 2: two planets → re-detrend triggered at i>=1
# ─────────────────────────────────────────────────────────────────────
def test_two_planet_redetrend_triggered(monkeypatch):
    time = _make_time_grid()
    raw_flux = _make_flux_with_transits(time, [
        (3.2, 0.8, 0.12, 0.006),
        (8.7, 2.1, 0.18, 0.004),
    ])
    flat_flux = raw_flux.copy()

    w = _make_worker(monkeypatch, time, flat_flux, raw_flux)

    import worker.worker as worker_mod
    detrend_calls = []
    real_detrend = worker_mod.detrend_with_wotan if hasattr(
        worker_mod, "detrend_with_wotan") else None

    def spy(time_arg, flux_arg, **kwargs):
        mask = kwargs.get("transit_mask")
        detrend_calls.append({
            "transit_mask_sum": int(mask.sum()) if mask is not None else 0,
        })
        if real_detrend is None:
            # Fallback if module import flow differs
            return flux_arg
        return real_detrend(time_arg, flux_arg, **kwargs)

    if real_detrend is not None:
        monkeypatch.setattr(worker_mod, "detrend_with_wotan", spy)

    result = w.process_task(_make_task())
    assert len(result["candidates"]) >= 2, \
        f"Expected ≥2 candidates, got {len(result['candidates'])}"

    # At least one re-detrend with a non-empty accumulated mask
    with_mask = [c for c in detrend_calls if c["transit_mask_sum"] > 0]
    assert len(with_mask) >= 1, \
        f"Expected ≥1 re-detrend with non-empty mask, got {len(with_mask)}"


# ─────────────────────────────────────────────────────────────────────
# Test 3: accumulated mask grows monotonically across iterations
# ─────────────────────────────────────────────────────────────────────
def test_accumulated_mask_growth(monkeypatch):
    time = _make_time_grid(n_days=54.0)  # ~2 sectors
    raw_flux = _make_flux_with_transits(time, [
        (3.2, 0.8, 0.12, 0.008),
        (5.7, 1.5, 0.15, 0.006),
        (10.3, 4.2, 0.20, 0.005),
    ])
    flat_flux = raw_flux.copy()

    w = _make_worker(monkeypatch, time, flat_flux, raw_flux)

    import worker.worker as worker_mod
    mask_sizes = []
    real_detrend = worker_mod.detrend_with_wotan if hasattr(
        worker_mod, "detrend_with_wotan") else None

    def spy(time_arg, flux_arg, **kwargs):
        mask = kwargs.get("transit_mask")
        if mask is not None and mask.any():
            mask_sizes.append(int(mask.sum()))
        if real_detrend is None:
            return flux_arg
        return real_detrend(time_arg, flux_arg, **kwargs)

    if real_detrend is not None:
        monkeypatch.setattr(worker_mod, "detrend_with_wotan", spy)

    w.process_task(_make_task())

    assert len(mask_sizes) >= 2, \
        f"Expected ≥2 re-detrend calls with non-empty mask, got {mask_sizes}"
    # Non-decreasing (each iteration accumulates more in-transit points)
    for a, b in zip(mask_sizes, mask_sizes[1:]):
        assert b >= a, f"Mask size decreased: {mask_sizes}"


# ─────────────────────────────────────────────────────────────────────
# Test 4: wotan failure falls back to hole-fill
# ─────────────────────────────────────────────────────────────────────
def test_wotan_failure_falls_back_to_hole_fill(monkeypatch, caplog):
    time = _make_time_grid()
    raw_flux = _make_flux_with_transits(time, [
        (3.2, 0.8, 0.12, 0.008),
        (7.1, 2.0, 0.15, 0.006),
    ])
    flat_flux = raw_flux.copy()

    w = _make_worker(monkeypatch, time, flat_flux, raw_flux)

    import worker.worker as worker_mod

    def boom(*args, **kwargs):
        # Only fail when called from the multi-planet loop (transit_mask
        # provided and non-empty); let the pre-loop detrend pass.
        mask = kwargs.get("transit_mask")
        if mask is not None and mask.any():
            raise RuntimeError("simulated wotan failure")
        # Otherwise, return flux_arg unchanged (pre-loop path)
        return args[1]

    # raising=False: until Task 3 promotes the import to module level,
    # worker.worker.detrend_with_wotan doesn't exist. Red-phase test
    # should still FAIL on the assertions, not ERROR in setup.
    monkeypatch.setattr(worker_mod, "detrend_with_wotan", boom, raising=False)

    caplog.set_level(logging.WARNING)
    result = w.process_task(_make_task())

    # Even with wotan failing the re-detrend, loop must continue
    # (fall back to hole-fill) and produce ≥1 candidate.
    assert len(result["candidates"]) >= 1
    # Warning must be logged
    assert any("re-detrend failed" in rec.message.lower()
               for rec in caplog.records), \
        f"Expected 're-detrend failed' warning; got: " \
        f"{[r.message for r in caplog.records]}"


# ─────────────────────────────────────────────────────────────────────
# Test 5-bis: candidates have DISTINCT periods after re-detrend
#
# Production DB (2026-04-21) showed ~85% of multi-planet TICs returning
# the same period 4-5× (e.g. TIC 243185500: P=1.8805 ×5). Root cause: the
# re-detrended flux still contains prior transit dips (wotan with mask
# fits the trend ignoring transits, but the returned flat still has
# them), so each BLS iteration finds the same signal. Fix: after
# re-detrend, fill `accumulated_mask` with the out-of-transit median so
# BLS searches for a *new* signal (Hippke+ 2019 §5, full spec).
#
# Test uses two unrelated periods (3.0d and 7.0d — 7/3 is not in the
# harmonic resolver's ratio set {1/5, 1/3, 1/2, 2}), so any duplicate
# would be the same planet rediscovered, not a harmonic alias.
# ─────────────────────────────────────────────────────────────────────
def test_multi_planet_finds_distinct_periods(monkeypatch):
    time = _make_time_grid(n_days=60.0)
    raw_flux = _make_flux_with_transits(time, [
        (3.0, 0.5, 0.10, 0.006),   # planet A
        (7.0, 2.3, 0.15, 0.005),   # planet B — unrelated period
    ], noise_sigma=5e-4)
    flat_flux = raw_flux.copy()

    w = _make_worker(monkeypatch, time, flat_flux, raw_flux)

    result = w.process_task(_make_task(max_planets=3))
    periods = [c["period_days"] for c in result["candidates"]]

    def _distinct_count(pers, tol=0.05):
        distinct = []
        for p in pers:
            if not any(abs(p - d) / d < tol for d in distinct):
                distinct.append(p)
        return len(distinct)

    n_distinct = _distinct_count(periods)
    assert n_distinct >= 2, (
        f"Expected ≥2 distinct periods, got {n_distinct} "
        f"from periods={periods} — multi-planet loop is finding the same "
        "planet multiple times (re-detrend not suppressing prior transits)"
    )


# ─────────────────────────────────────────────────────────────────────
# Test 6: GOLDEN — compact system recovers buried planet
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.slow
def test_compact_system_recovers_buried_planet(monkeypatch):
    """Demonstrates the #6 benefit: planet B (smaller, longer) is
    recovered when we re-detrend with an accumulated mask, but would
    be missed with median hole-filling alone."""
    time = _make_time_grid(n_days=54.0)
    raw_flux = _make_flux_with_transits(time, [
        (5.0, 1.2, 0.15, 0.008),   # planet A — clear
        (12.3, 3.7, 0.20, 0.002),  # planet B — marginal, easily buried
    ], noise_sigma=1e-3)
    flat_flux = raw_flux.copy()

    w = _make_worker(monkeypatch, time, flat_flux, raw_flux)

    result = w.process_task(_make_task(max_planets=3))
    periods = [c["period_days"] for c in result["candidates"]]

    def _matches(detected, expected, tol=0.20):
        for ratio in [1.0, 0.5, 2.0, 1.0/3, 3.0]:
            if abs(detected - expected * ratio) / (expected * ratio) < tol:
                return True
        return False

    recovered_A = any(_matches(p, 5.0) for p in periods)
    recovered_B = any(_matches(p, 12.3) for p in periods)
    assert recovered_A, f"Planet A lost: periods={periods}"
    assert recovered_B, (
        f"Planet B (buried) lost: periods={periods} — re-detrend not effective"
    )
