"""Tests for the CUDA context-poison auto-restart feature in worker.py.

Covers:
1. _is_cuda_context_error classifier:
   - matches every listed signature (case-insensitive)
   - does NOT match OOM / transient errors
   - does NOT match generic errors

2. _note_cuda_failure streak logic:
   - 2 CUDA errors + 1 success + 2 CUDA errors → no restart (streak resets)
   - 3 consecutive CUDA errors → os.execv called with bumped restart count

3. Guard against restart loops:
   - EXONIT_CUDA_RESTART_COUNT=3 → os.execv NOT called, CRITICAL logged

4. Success resets streak AND lineage counter (env var cleared).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — mirror the pattern from test_worker_bootstrap.py
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WORKER_PKG = str(_REPO_ROOT / "worker")
if _WORKER_PKG not in sys.path:
    sys.path.insert(0, _WORKER_PKG)

from worker.worker import _is_cuda_context_error, ExoNitWorker  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Classifier tests
# ---------------------------------------------------------------------------

class TestIsCudaContextError:
    """Unit tests for the _is_cuda_context_error() module-level helper."""

    # --- Signatures that MUST match ---

    def test_matches_cudaErrorLaunchFailure(self):
        assert _is_cuda_context_error(RuntimeError("cudaErrorLaunchFailure"))

    def test_matches_unspecified_launch_failure(self):
        assert _is_cuda_context_error(
            RuntimeError("CUDA error: unspecified launch failure")
        )

    def test_matches_cudaErrorIllegalAddress(self):
        assert _is_cuda_context_error(RuntimeError("cudaErrorIllegalAddress"))

    def test_matches_an_illegal_memory_access(self):
        assert _is_cuda_context_error(
            RuntimeError("CUDA error: an illegal memory access was encountered")
        )

    def test_matches_CUDA_ERROR_LAUNCH_FAILED(self):
        assert _is_cuda_context_error(RuntimeError("CUDA_ERROR_LAUNCH_FAILED"))

    def test_matches_CUDA_ERROR_ILLEGAL_ADDRESS(self):
        assert _is_cuda_context_error(RuntimeError("CUDA_ERROR_ILLEGAL_ADDRESS"))

    def test_matches_cudaErrorECCUncorrectable(self):
        assert _is_cuda_context_error(RuntimeError("cudaErrorECCUncorrectable"))

    def test_case_insensitive_upper(self):
        assert _is_cuda_context_error(
            RuntimeError("CUDAERRORLAUNCHFAILURE")
        )

    def test_case_insensitive_mixed(self):
        assert _is_cuda_context_error(
            RuntimeError("CudaErrorLaunchFailure in kernel")
        )

    # --- Errors that must NOT match ---

    def test_no_match_cudaErrorMemoryAllocation(self):
        """OOM is transient — must NOT trigger a restart."""
        assert not _is_cuda_context_error(
            RuntimeError("cudaErrorMemoryAllocation")
        )

    def test_no_match_out_of_memory(self):
        """Generic 'out of memory' is transient."""
        assert not _is_cuda_context_error(
            RuntimeError("CUDA out of memory: tried to allocate 2.00 GiB")
        )

    def test_no_match_generic_exception(self):
        assert not _is_cuda_context_error(ValueError("something went wrong"))

    def test_no_match_connection_error(self):
        assert not _is_cuda_context_error(
            ConnectionError("Connection refused")
        )

    def test_no_match_empty_string(self):
        assert not _is_cuda_context_error(RuntimeError(""))

    def test_no_match_timeout(self):
        assert not _is_cuda_context_error(TimeoutError("request timed out"))


# ---------------------------------------------------------------------------
# Helpers to build a minimal ExoNitWorker via __new__
# ---------------------------------------------------------------------------

def _make_worker(monkeypatch=None, cuda_restart_count: int = 0) -> ExoNitWorker:
    """Construct a bare ExoNitWorker via __new__ (skips GPU probes/__init__).

    Sets only the attributes accessed by _note_cuda_failure and
    _reset_cuda_streak; enough to exercise the CUDA-restart logic without
    a real server, config file, or GPU.
    """
    w = ExoNitWorker.__new__(ExoNitWorker)
    w.logger = logging.getLogger("test_cuda_restart")
    w._cuda_error_streak = 0
    w._cuda_restart_count = cuda_restart_count
    return w


# ---------------------------------------------------------------------------
# 2. Streak + restart logic
# ---------------------------------------------------------------------------

class TestNoteCudaFailureStreak:
    """Tests for the streak counter and restart trigger in _note_cuda_failure."""

    def test_two_cuda_errors_no_restart(self, monkeypatch):
        """Two consecutive CUDA errors must not trigger a restart."""
        execv_calls = []
        monkeypatch.setattr(os, "execv", lambda *a: execv_calls.append(a))

        w = _make_worker()
        exc = RuntimeError("cudaErrorLaunchFailure")
        w._note_cuda_failure(exc)
        w._note_cuda_failure(exc)

        assert w._cuda_error_streak == 2
        assert execv_calls == []

    def test_non_cuda_error_does_not_change_streak(self, monkeypatch):
        """Non-CUDA errors must leave streak unchanged."""
        execv_calls = []
        monkeypatch.setattr(os, "execv", lambda *a: execv_calls.append(a))

        w = _make_worker()
        cuda_exc = RuntimeError("cudaErrorLaunchFailure")
        other_exc = ValueError("unrelated error")

        w._note_cuda_failure(cuda_exc)  # streak → 1
        w._note_cuda_failure(other_exc)  # streak unchanged → 1
        w._note_cuda_failure(other_exc)  # streak unchanged → 1

        assert w._cuda_error_streak == 1
        assert execv_calls == []

    def test_success_resets_streak(self, monkeypatch):
        """After two CUDA errors a successful task (via _reset_cuda_streak)
        must reset the streak so 2 more errors don't trigger a restart."""
        execv_calls = []
        monkeypatch.setattr(os, "execv", lambda *a: execv_calls.append(a))
        monkeypatch.delenv("EXONIT_CUDA_RESTART_COUNT", raising=False)

        w = _make_worker()
        exc = RuntimeError("cudaErrorLaunchFailure")

        w._note_cuda_failure(exc)  # streak 1
        w._note_cuda_failure(exc)  # streak 2
        w._reset_cuda_streak()     # streak → 0
        w._note_cuda_failure(exc)  # streak 1
        w._note_cuda_failure(exc)  # streak 2

        assert w._cuda_error_streak == 2
        assert execv_calls == []

    def test_three_consecutive_triggers_restart(self, monkeypatch):
        """Three consecutive CUDA errors must call os.execv."""
        execv_calls = []
        monkeypatch.setattr(os, "execv", lambda *a: execv_calls.append(a))
        monkeypatch.delenv("EXONIT_CUDA_RESTART_COUNT", raising=False)

        w = _make_worker(cuda_restart_count=0)
        exc = RuntimeError("unspecified launch failure")

        w._note_cuda_failure(exc)
        w._note_cuda_failure(exc)
        assert execv_calls == [], "premature restart after 2 errors"
        w._note_cuda_failure(exc)

        assert len(execv_calls) == 1, "os.execv must be called exactly once"
        # Verify the bumped restart count was placed in the environment
        assert os.environ.get("EXONIT_CUDA_RESTART_COUNT") == "1"

    def test_restart_bumps_count(self, monkeypatch):
        """Each restart increments EXONIT_CUDA_RESTART_COUNT by 1."""
        execv_calls = []
        monkeypatch.setattr(os, "execv", lambda *a: execv_calls.append(a))
        # Simulate: worker already restarted twice
        monkeypatch.setenv("EXONIT_CUDA_RESTART_COUNT", "2")

        w = _make_worker(cuda_restart_count=2)
        exc = RuntimeError("CUDA_ERROR_LAUNCH_FAILED")

        w._note_cuda_failure(exc)
        w._note_cuda_failure(exc)
        w._note_cuda_failure(exc)

        assert len(execv_calls) == 1
        assert os.environ.get("EXONIT_CUDA_RESTART_COUNT") == "3"

    def test_restart_passes_sys_argv(self, monkeypatch):
        """os.execv must be called with sys.executable and sys.argv."""
        captured = {}

        def fake_execv(executable, args):
            captured["executable"] = executable
            captured["args"] = args

        monkeypatch.setattr(os, "execv", fake_execv)
        monkeypatch.delenv("EXONIT_CUDA_RESTART_COUNT", raising=False)

        w = _make_worker(cuda_restart_count=0)
        exc = RuntimeError("cudaErrorIllegalAddress")

        w._note_cuda_failure(exc)
        w._note_cuda_failure(exc)
        w._note_cuda_failure(exc)

        assert captured.get("executable") == sys.executable
        assert captured.get("args") == [sys.executable] + sys.argv


# ---------------------------------------------------------------------------
# 3. Guard against restart loops
# ---------------------------------------------------------------------------

class TestRestartGuard:
    """When EXONIT_CUDA_RESTART_COUNT reaches 3, do NOT restart."""

    def test_guard_no_execv_when_count_3(self, monkeypatch):
        execv_calls = []
        monkeypatch.setattr(os, "execv", lambda *a: execv_calls.append(a))
        monkeypatch.setenv("EXONIT_CUDA_RESTART_COUNT", "3")

        w = _make_worker(cuda_restart_count=3)
        exc = RuntimeError("cudaErrorLaunchFailure")

        w._note_cuda_failure(exc)
        w._note_cuda_failure(exc)
        w._note_cuda_failure(exc)

        assert execv_calls == [], (
            "os.execv must NOT be called when restart_count >= 3"
        )

    def test_guard_logs_critical(self, monkeypatch):
        """When the guard fires a CRITICAL must be emitted."""
        monkeypatch.setattr(os, "execv", lambda *a: None)
        monkeypatch.setenv("EXONIT_CUDA_RESTART_COUNT", "3")

        w = _make_worker(cuda_restart_count=3)
        exc = RuntimeError("cudaErrorLaunchFailure")

        with patch.object(w.logger, "critical") as mock_critical:
            w._note_cuda_failure(exc)
            w._note_cuda_failure(exc)
            w._note_cuda_failure(exc)

        assert mock_critical.called, "logger.critical must be called when guard fires"

    def test_guard_resets_streak_to_avoid_spam(self, monkeypatch):
        """After the guard fires, streak must reset to 0 so CRITICAL isn't logged
        on every subsequent task failure."""
        monkeypatch.setattr(os, "execv", lambda *a: None)
        monkeypatch.setenv("EXONIT_CUDA_RESTART_COUNT", "3")

        w = _make_worker(cuda_restart_count=3)
        exc = RuntimeError("cudaErrorLaunchFailure")

        w._note_cuda_failure(exc)
        w._note_cuda_failure(exc)
        w._note_cuda_failure(exc)  # guard fires, streak resets to 0

        assert w._cuda_error_streak == 0, (
            "streak must be reset to 0 after the guard fires to avoid CRITICAL spam"
        )

    def test_guard_count_exactly_3(self, monkeypatch):
        """Count of exactly 3 must also be blocked (>= comparison)."""
        execv_calls = []
        monkeypatch.setattr(os, "execv", lambda *a: execv_calls.append(a))
        monkeypatch.setenv("EXONIT_CUDA_RESTART_COUNT", "3")

        w = _make_worker(cuda_restart_count=3)
        exc = RuntimeError("an illegal memory access was encountered")

        for _ in range(3):
            w._note_cuda_failure(exc)

        assert execv_calls == []

    def test_count_2_still_restarts(self, monkeypatch):
        """Count of 2 is below the limit and must still restart."""
        execv_calls = []
        monkeypatch.setattr(os, "execv", lambda *a: execv_calls.append(a))
        monkeypatch.delenv("EXONIT_CUDA_RESTART_COUNT", raising=False)

        w = _make_worker(cuda_restart_count=2)
        exc = RuntimeError("CUDA_ERROR_ILLEGAL_ADDRESS")

        for _ in range(3):
            w._note_cuda_failure(exc)

        assert len(execv_calls) == 1


# ---------------------------------------------------------------------------
# 4. Success resets streak and lineage counter
# ---------------------------------------------------------------------------

class TestResetCudaStreak:
    """Tests for _reset_cuda_streak() clearing both streak and lineage."""

    def test_reset_clears_streak(self, monkeypatch):
        monkeypatch.delenv("EXONIT_CUDA_RESTART_COUNT", raising=False)
        w = _make_worker()
        w._cuda_error_streak = 2
        w._cuda_restart_count = 1
        w._reset_cuda_streak()
        assert w._cuda_error_streak == 0

    def test_reset_clears_restart_count(self, monkeypatch):
        monkeypatch.delenv("EXONIT_CUDA_RESTART_COUNT", raising=False)
        w = _make_worker(cuda_restart_count=2)
        w._cuda_error_streak = 1
        w._reset_cuda_streak()
        assert w._cuda_restart_count == 0

    def test_reset_removes_env_var(self, monkeypatch):
        monkeypatch.setenv("EXONIT_CUDA_RESTART_COUNT", "2")
        w = _make_worker(cuda_restart_count=2)
        w._cuda_error_streak = 1
        w._reset_cuda_streak()
        assert "EXONIT_CUDA_RESTART_COUNT" not in os.environ

    def test_reset_idempotent_when_already_zero(self, monkeypatch):
        """Calling _reset_cuda_streak when streak is already 0 must not raise."""
        monkeypatch.delenv("EXONIT_CUDA_RESTART_COUNT", raising=False)
        w = _make_worker()
        w._cuda_error_streak = 0
        w._cuda_restart_count = 0
        w._reset_cuda_streak()  # must not raise
        assert w._cuda_error_streak == 0
        assert w._cuda_restart_count == 0
