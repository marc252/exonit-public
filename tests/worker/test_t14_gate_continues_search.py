"""Regression test for the T14-gate multi-planet abort bug.

Baseline-100 2026-04-21 C showed 8 TICs (7 CP + 1 KP) losing *all*
candidates versus baseline B. Investigation:
``data/scorecards/baseline-100-post-t14-harmonic-2026-04-21`` vs
``...-post-harmonic-...`` showed the T14 gate committed at 5be73ae was
correctly rejecting the unphysical long-period first peak (e.g. TIC
116483514 P=35.8d dur=5.4d), but the ``break`` at worker.py:1060 then
aborted the whole multi-planet search — so the legit planet at P=1.45d
(ExoMiner score 0.9993) was never found.

The fix: mask the rejected candidate's transits into ``accumulated_mask``
(so BLS on the next iteration cannot re-find the same peak) and
``continue`` instead of breaking. The gate still rejects unphysical
durations; it just doesn't kill the search for legit shorter-period
planets.
"""
from __future__ import annotations

import logging
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "worker") not in sys.path:
    sys.path.append(str(ROOT / "worker"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.worker import ExoNitWorker  # noqa: E402


def _make_time_grid(n_days: float = 30.0, cadence_min: float = 2.0) -> np.ndarray:
    n_points = int(n_days * 24 * 60 / cadence_min)
    return np.linspace(0.0, n_days, n_points)


def _make_worker(monkeypatch, time_arr, flat_flux, raw_flux):
    w = ExoNitWorker.__new__(ExoNitWorker)
    w.logger = logging.getLogger("test_t14_continue")
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
            "stellar_radius": 1.0, "stellar_mass": 1.0,
        }

    monkeypatch.setattr(ExoNitWorker, "_load_star", fake_load_star)
    return w


def _make_task():
    return {
        "task_id": "t1",
        "tic_id": 1,
        "config": {
            "max_planets": 5,
            "snr_threshold": 7.0,
            "period_min": 0.5,
            "period_max": 50.0,
            "n_periods": 5000,
            "run_injection": False,
        },
    }


def test_unphysical_first_peak_does_not_abort_multi_planet_search(monkeypatch):
    """T14 gate rejection must mask + continue, not break.

    Simulates baseline-100 TIC 116483514: first BLS peak is an unphysical
    long-period systematic (P=35d dur=5d, T14 > 3×T14_max). The real
    planet sits at P≈3d. With the break-on-reject bug, candidates=[]. With
    the mask+continue fix, the real planet is found on iteration 2.
    """
    time = _make_time_grid(n_days=30.0)
    # Content of flat_flux doesn't matter — we mock run_bls entirely.
    flat_flux = np.ones_like(time)
    raw_flux = np.ones_like(time)

    w = _make_worker(monkeypatch, time, flat_flux, raw_flux)

    import worker.worker as worker_mod

    # Distinguish main BLS calls (n_periods large, broad range) from
    # harmonic resolver probes (n_periods=500, narrow range).
    main_calls = []

    def fake_bls(time_arg, flux_arg, period_min, period_max, n_periods):
        # Harmonic probes: narrow band (period_max/period_min ~ 1.1). Return
        # a weak result so the resolver never promotes over the seed.
        if n_periods <= 600:
            return {
                "period": (period_min + period_max) / 2,
                "t0": 0.0,
                "duration": 0.05,
                "depth_ppm": 100.0,
                "depth": 1e-4,
                "snr": 1.0,
                "power": 1.0,
            }

        main_calls.append((period_min, period_max, n_periods))
        if len(main_calls) == 1:
            # Unphysical long-period peak: T14_max(P=35) ≈ 0.33d; 3× = 1.0d.
            # dur=5.0d is well above and must be rejected.
            return {
                "period": 35.0,
                "t0": 3.2,
                "duration": 5.0,
                "depth_ppm": 20000.0,
                "depth": 0.02,
                "snr": 200.0,
                "power": 500.0,
            }
        # Second main BLS call: real planet at P=3d with healthy SNR.
        return {
            "period": 3.0,
            "t0": 0.8,
            "duration": 0.12,
            "depth_ppm": 5000.0,
            "depth": 5e-3,
            "snr": 30.0,
            "power": 100.0,
        }

    monkeypatch.setattr(worker_mod, "run_bls", fake_bls)

    result = w.process_task(_make_task())

    # --- Primary assertion: multi-planet loop kept going after rejection ---
    assert len(main_calls) >= 2, (
        f"Only {len(main_calls)} main BLS call(s) — T14 gate rejection "
        "aborted the loop. Expected ≥2 (1 unphysical + 1 real planet)."
    )

    # --- Secondary: the real planet at P=3d made it into candidates ---
    periods = [c["period_days"] for c in result["candidates"]]
    assert any(2.9 < p < 3.1 for p in periods), (
        f"Real P=3d planet not recovered. Candidates: {periods}. "
        "T14 gate must mask the rejected peak, not break the loop."
    )

    # --- Tertiary: the rejected unphysical peak is NOT in candidates ---
    assert not any(34.0 < p < 36.0 for p in periods), (
        f"Unphysical P=35d peak was accepted. Candidates: {periods}. "
        "T14 gate failed to reject it."
    )
