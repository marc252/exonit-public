"""Worker-level tests: duration_refine / merge_same_period flags (P2).

Covers the three call sites (main loop, protected slot, long-period sweep),
that refined duration/t0 reach the candidate dict, and that the physical-T14
gate sees the REFINED duration. Flags default to off (opt-in).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "worker", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tests.worker.test_multi_planet_redetrend import (  # noqa: E402
    _make_flux_with_transits, _make_time_grid, _make_worker, _make_task)

P, T0, TRUE_DUR = 3.17, 1.0, 0.078


def _run(monkeypatch, time, raw, fake_bls, **flags):
    w = _make_worker(monkeypatch, time, raw.copy(), raw)
    import worker.worker as worker_mod
    monkeypatch.setattr(worker_mod, "run_bls", fake_bls)
    task = _make_task()
    task["config"].update(flags)
    return w.process_task(task)["candidates"]


def _main_loop_run(monkeypatch, **flags):
    time = _make_time_grid()
    raw = _make_flux_with_transits(time, [(P, T0, TRUE_DUR, 0.004)])

    def fake_bls(t, f, **kw):
        # Coarse BLS-like answer: 2x too long, shifted, same period every call.
        return {"period": P, "t0": T0 + 0.05, "duration": 0.159,
                "depth": 0.004, "depth_ppm": 4000.0, "snr": 12.0,
                "power": 0.5}

    return _run(monkeypatch, time, raw, fake_bls, **flags), time


def test_flags_off_full_candidate_equality(monkeypatch):
    # Pre-change behaviour: every BLS answer becomes a candidate verbatim.
    for flags in ({}, {"duration_refine": False, "merge_same_period": False}):
        cands, time = _main_loop_run(monkeypatch, **flags)
        n_tr = int((time[-1] - time[0]) / P)
        assert len(cands) == 5
        for i, c in enumerate(cands):
            assert c == {
                "candidate_num": i + 1, "period_days": 3.17, "t0": 1.05,
                "duration_days": 0.159, "depth_ppm": 4000.0, "snr": 12.0,
                "bls_power": 0.5, "n_transits_expected": n_tr}


def test_main_loop_flags_on_no_shared_period_and_refined(monkeypatch):
    cands, _ = _main_loop_run(monkeypatch, duration_refine=True,
                              merge_same_period=True)
    assert len(cands) == 1 and cands[0]["candidate_num"] == 1
    assert abs(cands[0]["duration_days"] - TRUE_DUR) / TRUE_DUR < 0.25
    assert abs(cands[0]["t0"] - T0) < 0.1 * TRUE_DUR * 2


def test_protected_slot_refined_and_t14_sees_refined(monkeypatch):
    # P=5 d: 3*T14_max = 0.4286 d. Coarse box 0.5 d would be rejected as
    # unphysical; the refined ~0.1 d box passes the gate.
    time = _make_time_grid()
    raw = _make_flux_with_transits(time, [(5.0, 1.0, 0.10, 0.006)])

    def fake_bls(t, f, **kw):
        if kw.get("period_min", 0.0) == 3.0:
            return {"period": 5.0, "t0": 1.05, "duration": 0.5,
                    "depth": 0.002, "depth_ppm": 2000.0, "snr": 15.0,
                    "power": 0.5}
        return {"period": 5.0, "t0": 1.0, "duration": 0.1, "depth": 0.0,
                "depth_ppm": 0.0, "snr": 3.0, "power": 0.1}

    off = _run(monkeypatch, time, raw, fake_bls)
    assert off == []
    on = _run(monkeypatch, time, raw, fake_bls, duration_refine=True)
    assert len(on) == 1
    assert abs(on[0]["duration_days"] - 0.10) / 0.10 < 0.25
    assert abs(on[0]["t0"] - 1.0) < 0.05
    assert on[0]["depth_ppm"] > 4000.0   # refined (narrow) box is deeper
    assert on[0]["snr"] == 15.0          # BLS SNR untouched


def test_long_period_sweep_refined_and_t14_sees_refined(monkeypatch):
    # P=40 d: 3*T14_max = 0.856 d. Coarse 1.0 d box rejected; refined ~0.3 d ok.
    time = np.arange(0.0, 120.0, 30.0 / 1440)
    raw = _make_flux_with_transits(time, [(40.0, 5.0, 0.30, 0.006)])

    def fake_bls(t, f, **kw):
        if kw.get("period_min", 0.0) == 30.0:
            return {"period": 40.0, "t0": 5.1, "duration": 1.0,
                    "depth": 0.002, "depth_ppm": 2000.0, "snr": 15.0,
                    "power": 0.5}
        return {"period": 40.0, "t0": 5.0, "duration": 0.3, "depth": 0.0,
                "depth_ppm": 0.0, "snr": 3.0, "power": 0.1}

    off = _run(monkeypatch, time, raw, fake_bls)
    assert off == []
    on = _run(monkeypatch, time, raw, fake_bls, duration_refine=True)
    assert len(on) == 1
    assert abs(on[0]["duration_days"] - 0.30) / 0.30 < 0.3
    assert abs(on[0]["t0"] - 5.0) < 0.1
    assert on[0]["period_days"] == 40.0
