"""Regression test for the startup telemetry-purge deadlock (2026-07-10).

With the A3 multi-process server (``uvicorn --workers N``), every process
runs ``create_app()`` at boot. Before this fix, each of the N processes
called ``db.purge_telemetry(5)`` directly, firing N concurrent
``DELETE FROM worker_telemetry ...`` against PostgreSQL — typically N-1 of
them died with ``psycopg.errors.DeadlockDetected`` (noisy but harmless).

``exonit.server._purge_telemetry_once`` guards the purge behind the
shared-state NX primitive (``set_if_absent``) so only the first process
(per boot burst) performs the DELETE. This test exercises that helper
directly against a ``MemoryBackend`` (works identically to Redis for the
NX semantics that matter here), simulating N=4 "processes" racing at boot.
"""

from unittest.mock import MagicMock

from exonit.server import _purge_telemetry_once
from exonit.shared_state import MemoryBackend


def test_purge_runs_once_across_simulated_processes():
    """Two (or more) concurrent 'boots' sharing state → exactly one DELETE."""
    shared_state = MemoryBackend()
    db = MagicMock()

    won_first = _purge_telemetry_once(db, shared_state, retention_days=5)
    won_second = _purge_telemetry_once(db, shared_state, retention_days=5)

    assert won_first is True
    assert won_second is False
    db.purge_telemetry.assert_called_once_with(5)


def test_purge_runs_again_after_lock_expires():
    """Once the startup lock TTL elapses, a later restart purges again."""
    # Fake clock injected into MemoryBackend so TTL expiry is deterministic
    # (see MemoryBackend(clock=...) docstring) rather than sleeping 300s+.
    fake_now = [0.0]
    shared_state = MemoryBackend(clock=lambda: fake_now[0])
    db = MagicMock()

    assert _purge_telemetry_once(db, shared_state, retention_days=5) is True

    # Advance the fake clock past the 300s startup-lock TTL (see
    # exonit.server._STARTUP_TELEMETRY_PURGE_TTL_S).
    fake_now[0] += 301.0

    assert _purge_telemetry_once(db, shared_state, retention_days=5) is True
    assert db.purge_telemetry.call_count == 2


def test_purge_single_process_still_runs():
    """Single-process (A0-A2) deployments must still purge on the only boot."""
    shared_state = MemoryBackend()
    db = MagicMock()

    won = _purge_telemetry_once(db, shared_state)

    assert won is True
    db.purge_telemetry.assert_called_once_with(5)
