"""Behaviour tests for the injection-recovery wall-clock timeout.

Context: on Apple-Silicon workers injection-recovery runs on CPU by design
(Metal is reserved for FPP). For a giant multi-sector FFI target (e.g. TIC
243278113: 173,105 points, 43 sectors) the CPU injection-recovery can take
30-45 min on a single thread, monopolising the worker for the whole run
(observed live: worker-<host>-3 at 42 min CPU on one target).

`run_injection_recovery(..., max_seconds=...)` adds a configurable
wall-clock deadline: once exceeded, no new period-cells (or radius-cells,
in the last-resort per-call fallback) are launched, the partial
completeness computed so far is returned, and the result is flagged
``truncated=True``. The default (``max_seconds=None``) must remain
byte-identical to the pre-timeout behaviour -- these tests prove both
sides: truncation fires under a tiny deadline, and does NOT fire (and
does not change the returned shape) when the deadline is absent.
"""
from __future__ import annotations

import sys
import time as _real_time
from pathlib import Path

import numpy as np
import pytest

# Ensure the worker package (and its flat worker/ siblings, e.g. `injection`,
# `bls_fast`) are importable -- same pattern as the other tests/worker/*.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_WORKER_PKG = str(_REPO_ROOT / "worker")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if _WORKER_PKG not in sys.path:
    sys.path.append(_WORKER_PKG)

import injection  # noqa: E402


def _make_lc(n_points=500, span_days=30.0):
    rng = np.random.RandomState(0)
    time = np.linspace(0, span_days, n_points)
    flux = 1.0 + rng.normal(0, 1e-4, n_points)
    return time, flux


def _slow_batch_fn(sleep_s):
    """Stub for ``_bls_gpu_batch`` / ``bls_fast_batch``: sleeps ``sleep_s``
    per call (i.e. per period-cell) and reports zero recoveries."""
    def _fn(time, flux_list, period_min=0.5, period_max=50.0,
            n_periods=5000, n_bins=150, n_dur_steps=10):
        _real_time.sleep(sleep_s)
        return [{"period": period_min, "snr": 0.0} for _ in flux_list]
    return _fn


def test_max_seconds_truncates_pathological_target(monkeypatch):
    """A slow batch_fn + a tiny max_seconds must truncate the batched path:
    fewer period-cells evaluated than the full grid, truncated=True, and
    it returns in well under the time an untruncated run would take."""
    time, flux = _make_lc()
    period_grid = np.array([1.0, 2.0, 4.0, 8.0])
    radius_grid = np.array([1.0, 2.0])

    # Each period-cell takes 0.2s; the deadline (0.05s) is exceeded before
    # even the first cell finishes, so truncation after cell 1 is
    # deterministic regardless of test-host jitter.
    monkeypatch.setattr(injection, "_bls_gpu_batch", _slow_batch_fn(0.2))
    monkeypatch.setattr(injection, "_gpu_backend", "TestStub")

    t0 = _real_time.monotonic()
    result = injection.run_injection_recovery(
        time, flux, radius_grid=radius_grid, period_grid=period_grid,
        n_injections_per_cell=2, max_seconds=0.05)
    elapsed = _real_time.monotonic() - t0

    assert result["truncated"] is True
    assert result["periods_evaluated"] < len(period_grid)
    # An untruncated run would take >= 0.2s * len(period_grid) = 0.8s.
    assert elapsed < 0.2 * len(period_grid)
    completeness = np.array(result["completeness_map"])
    assert completeness.shape == (len(radius_grid), len(period_grid))


def test_max_seconds_none_default_is_unaffected(monkeypatch):
    """No max_seconds (or explicit None) -> old behaviour: every cell
    evaluated, no truncation, same shape as before the feature existed."""
    time, flux = _make_lc()
    period_grid = np.array([1.0, 2.0])
    radius_grid = np.array([1.0, 2.0])

    # Deliberately slow, to prove the *absence* of a deadline is what lets
    # it run to completion (not just "it happened to be fast enough").
    monkeypatch.setattr(injection, "_bls_gpu_batch", _slow_batch_fn(0.01))
    monkeypatch.setattr(injection, "_gpu_backend", "TestStub")

    result_default = injection.run_injection_recovery(
        time, flux, radius_grid=radius_grid, period_grid=period_grid,
        n_injections_per_cell=2)  # max_seconds not passed -> None
    result_explicit_none = injection.run_injection_recovery(
        time, flux, radius_grid=radius_grid, period_grid=period_grid,
        n_injections_per_cell=2, max_seconds=None)

    for result in (result_default, result_explicit_none):
        assert result["truncated"] is False
        assert result["periods_evaluated"] == len(period_grid)
        completeness = np.array(result["completeness_map"])
        assert completeness.shape == (len(radius_grid), len(period_grid))
        assert len(result["details"]) == len(radius_grid) * len(period_grid)


def test_per_call_fallback_loop_respects_max_seconds(monkeypatch):
    """Same guard on the last-resort per-call BLS loop (only reached in prod
    when numba/bls_fast_batch is unavailable, but must not regress either)."""
    time, flux = _make_lc()
    period_grid = np.array([1.0, 2.0])
    radius_grid = np.array([1.0, 2.0, 4.0])

    def _slow_bls_fn(time, flux, period_min=0.5, period_max=50.0,
                      n_periods=5000, n_bins=150, n_dur_steps=10):
        _real_time.sleep(0.2)
        return {"period": period_min, "snr": 0.0}

    # Force the per-call fallback path: no GPU/CPU batch fn available, so
    # run_injection_recovery falls through to the last-resort loop.
    monkeypatch.setattr(injection, "_bls_gpu_batch", None)
    monkeypatch.setattr(injection, "_bls_gpu", _slow_bls_fn)
    import bls_fast as bls_fast_module
    monkeypatch.delattr(bls_fast_module, "bls_fast_batch", raising=False)

    t0 = _real_time.monotonic()
    result = injection.run_injection_recovery(
        time, flux, radius_grid=radius_grid, period_grid=period_grid,
        n_injections_per_cell=1, max_seconds=0.05)
    elapsed = _real_time.monotonic() - t0

    assert result["truncated"] is True
    assert result["radii_evaluated"] < len(radius_grid)
    # An untruncated run would take >= 0.2s * len(period_grid) * len(radius_grid).
    assert elapsed < 0.2 * len(period_grid) * len(radius_grid)
    completeness = np.array(result["completeness_map"])
    assert completeness.shape == (len(radius_grid), len(period_grid))


def test_per_call_fallback_max_seconds_none_runs_to_completion(monkeypatch):
    """Regression guard: the per-call fallback with max_seconds=None (the
    only value any pre-existing caller passes) must still evaluate every
    radius cell -- no accidental truncation from the new guard."""
    time, flux = _make_lc()
    period_grid = np.array([1.0])
    radius_grid = np.array([1.0, 2.0])

    def _fast_bls_fn(time, flux, period_min=0.5, period_max=50.0,
                      n_periods=5000, n_bins=150, n_dur_steps=10):
        return {"period": period_min, "snr": 0.0}

    monkeypatch.setattr(injection, "_bls_gpu_batch", None)
    monkeypatch.setattr(injection, "_bls_gpu", _fast_bls_fn)
    import bls_fast as bls_fast_module
    monkeypatch.delattr(bls_fast_module, "bls_fast_batch", raising=False)

    result = injection.run_injection_recovery(
        time, flux, radius_grid=radius_grid, period_grid=period_grid,
        n_injections_per_cell=1)  # max_seconds defaults to None

    assert result["truncated"] is False
    assert result["radii_evaluated"] == len(radius_grid)
