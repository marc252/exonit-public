"""GPU->CPU fallback must be loud, counted and bounded (2026-07-14).

Two failures came together here:

1. A CPU fallback that logs at DEBUG is a SILENT fallback. A worker could run a
   whole job on the CPU (~5x slower for FPP) without one line saying so — the
   exact state project rule 2 forbids ("CPU fallback is NOT an
   acceptable option"). The rule only stays enforceable if the fallback is visible.

2. Retrying the GPU forever. On the mac mini M1 (2026-07-13), 5 concurrent
   FPP-Metal instances made macOS's GPU watchdog kill the command buffers
   ("Impacting Interactivity"). Retrying straight back into a saturated GPU
   just feeds the watchdog: the failure belongs to the machine's GPU pressure,
   not to the draw. After K strikes this process finishes on the CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))

import fpp_native  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_stats():
    fpp_native._GPU_FALLBACK_STATS.update(
        consecutive=0, total=0, disabled=False, last_error=None)
    saved = (fpp_native._gpu_batch_transit, fpp_native._gpu_batch_eb_transit)
    yield
    fpp_native._gpu_batch_transit, fpp_native._gpu_batch_eb_transit = saved
    fpp_native._GPU_FALLBACK_STATS.update(
        consecutive=0, total=0, disabled=False, last_error=None)


def _boom(*_args):
    raise RuntimeError(
        "[METAL] Command buffer execution failed: Impacting Interactivity")


def _cpu(*_args):
    return "cpu-result"


def test_gpu_failure_falls_back_to_cpu_and_says_so(caplog):
    caplog.set_level("WARNING", logger="exonit")

    out = fpp_native._dispatch_lnL(_boom, _cpu, (), "transit")

    assert out == "cpu-result"                       # the task still completes
    assert any("FAILED" in r.message and "CPU" in r.message
               for r in caplog.records), "the fallback was silent"
    stats = fpp_native.gpu_fallback_stats()
    assert stats["total"] == 1
    assert "Impacting Interactivity" in stats["last_error"]


def test_a_success_resets_the_strike_count():
    """Isolated hiccups must not accumulate into a breaker trip over hours."""
    fpp_native._dispatch_lnL(_boom, _cpu, (), "transit")
    assert fpp_native.gpu_fallback_stats()["consecutive"] == 1

    fpp_native._dispatch_lnL(lambda *_a: "gpu-result", _cpu, (), "transit")
    assert fpp_native.gpu_fallback_stats()["consecutive"] == 0
    assert fpp_native.gpu_fallback_stats()["total"] == 1   # history is kept


def test_breaker_retires_the_gpu_after_k_consecutive_failures(caplog):
    caplog.set_level("WARNING", logger="exonit")
    fpp_native._gpu_batch_transit = _boom
    fpp_native._gpu_batch_eb_transit = _boom

    for _ in range(fpp_native._GPU_CIRCUIT_BREAKER_K):
        assert fpp_native._dispatch_lnL(_boom, _cpu, (), "transit") == "cpu-result"

    stats = fpp_native.gpu_fallback_stats()
    assert stats["disabled"] is True
    # The GPU entry points are retired, so later batches never touch the GPU
    # again in this process (the callers read these globals at call time).
    assert fpp_native._gpu_batch_transit is None
    assert fpp_native._gpu_batch_eb_transit is None
    assert sum("DISABLED" in r.message for r in caplog.records) == 1


def test_breaker_warns_exactly_once_not_on_every_later_batch(caplog):
    caplog.set_level("WARNING", logger="exonit")
    for _ in range(fpp_native._GPU_CIRCUIT_BREAKER_K + 4):
        fpp_native._dispatch_lnL(_boom, _cpu, (), "transit")

    assert sum("DISABLED" in r.message for r in caplog.records) == 1


def test_no_gpu_backend_uses_the_guarded_cpu_path_without_warning(caplog):
    """A worker with no GPU at all is not a 'fallback' — don't cry wolf."""
    caplog.set_level("WARNING", logger="exonit")
    assert fpp_native._dispatch_lnL(None, _cpu, (), "transit") == "cpu-result"
    assert fpp_native.gpu_fallback_stats()["total"] == 0
    assert not caplog.records
