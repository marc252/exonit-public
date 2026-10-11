"""Unit tests for the worker PID lock.

The lock prevents multiple worker processes from sharing the same
worker_name on one host (observed <host> + worker-default 2026-04-21:
zombies from repeated restarts without killing the prior process caused
3× slowdown per TIC). Tests cover:
1. Fresh start writes the lockfile
2. Stale lock (process gone) is silently replaced
3. Live peer with different PID raises SystemExit
4. os.execv self-restart (same PID) is a no-op, never raises
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "worker") not in sys.path:
    sys.path.append(str(ROOT / "worker"))

from worker.worker import _pid_alive, ExoNitWorker  # noqa: E402


def _write_config(tmp_path: Path, worker_name: str) -> Path:
    cfg = {
        "server_url": "http://localhost:0",
        "token": "test-token",
        "worker_name": worker_name,
        "cpu_threads": 1,
        "gpu_enabled": False,
        "poll_interval_seconds": 5,
        "heartbeat_interval_seconds": 30,
        "fits_cache_dir": str(tmp_path / "fits_cache"),
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path


def _make_worker_from_config(cfg_path: Path) -> ExoNitWorker:
    """Instantiate via the full __init__ so the PID lock runs."""
    return ExoNitWorker(config_path=str(cfg_path))


# ─────────────────────────────────────────────────────────────────────
# Test 1: fresh start writes the lockfile
# ─────────────────────────────────────────────────────────────────────
def test_fresh_start_writes_lock(tmp_path):
    cfg = _write_config(tmp_path, "worker-test-fresh")
    w = _make_worker_from_config(cfg)
    lockfile = tmp_path / ".worker_worker-test-fresh.pid"
    assert lockfile.exists(), f"lockfile missing: {lockfile}"
    assert int(lockfile.read_text().strip()) == os.getpid()
    w._release_pid_lock()


# ─────────────────────────────────────────────────────────────────────
# Test 2: stale lockfile (process gone) is replaced
# ─────────────────────────────────────────────────────────────────────
def test_stale_lock_replaced(tmp_path):
    cfg = _write_config(tmp_path, "worker-test-stale")
    lockfile = tmp_path / ".worker_worker-test-stale.pid"
    lockfile.write_text("999999")  # PID very unlikely to exist
    w = _make_worker_from_config(cfg)
    assert int(lockfile.read_text().strip()) == os.getpid()
    w._release_pid_lock()


# ─────────────────────────────────────────────────────────────────────
# Test 3: live peer with different PID → SystemExit
# ─────────────────────────────────────────────────────────────────────
def test_live_peer_refuses_startup(tmp_path):
    cfg = _write_config(tmp_path, "worker-test-live")
    lockfile = tmp_path / ".worker_worker-test-live.pid"
    # PID of the parent process (pytest runner) — alive but not ours
    # (distinct from os.getpid()). os.getppid() satisfies both constraints.
    parent_pid = os.getppid()
    lockfile.write_text(str(parent_pid))
    with pytest.raises(SystemExit):
        _make_worker_from_config(cfg)
    # Cleanup (we don't own the lock, so leave it)
    lockfile.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────
# Test 4: lockfile already holds our PID (os.execv self-restart) → no-op
# ─────────────────────────────────────────────────────────────────────
def test_self_restart_is_noop(tmp_path):
    cfg = _write_config(tmp_path, "worker-test-execv")
    lockfile = tmp_path / ".worker_worker-test-execv.pid"
    lockfile.write_text(str(os.getpid()))
    w = _make_worker_from_config(cfg)
    # Still holds our PID (never raised, didn't replace)
    assert int(lockfile.read_text().strip()) == os.getpid()
    w._release_pid_lock()


# ─────────────────────────────────────────────────────────────────────
# Test 5: _pid_alive sanity checks
# ─────────────────────────────────────────────────────────────────────
def test_pid_alive_helper():
    assert _pid_alive(os.getpid()) is True
    # PID 1 is init on Unix, usually alive; skip the positive case
    # to avoid platform-dependent assumptions.
    assert _pid_alive(999999) is False  # huge gap — unlikely to exist


# ─────────────────────────────────────────────────────────────────────
# Test 6: Windows CPython quirk — os.kill raises SystemError on dead PID
# ─────────────────────────────────────────────────────────────────────
def test_pid_alive_returns_false_on_system_error(monkeypatch):
    """os.kill raising SystemError (Windows OpenProcess quirk after os.execv)
    must be treated as 'not alive', not propagated."""
    def _fake_kill(pid, sig):
        raise SystemError("<class 'OSError'> returned a result with an exception set")

    monkeypatch.setattr(os, "kill", _fake_kill)
    assert _pid_alive(12345) is False


def test_pid_alive_returns_true_when_kill_succeeds(monkeypatch):
    """os.kill returning normally means the process is alive — no regression."""
    def _fake_kill(pid, sig):
        return  # success, process alive

    monkeypatch.setattr(os, "kill", _fake_kill)
    assert _pid_alive(12345) is True
