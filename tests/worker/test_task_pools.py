"""Task-pool whitelist: a worker with `task_pools` set should only poll
the listed queues.

Why this matters: when we mix expensive GPU g4dn workers (primary run)
with cheap CPU c6g.medium ARM workers (backfill only), the CPU workers
must NOT poll /api/worker/task. BLS+TRICERATOPS on 1 vCPU is 10-20×
slower than on a T4 — a CPU worker that grabs a primary task starves
the GPU pool of work without compensating throughput.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "worker") not in sys.path:
    sys.path.append(str(ROOT / "worker"))

from worker.worker import ExoNitWorker  # noqa: E402


def _write_config(tmp_path: Path, name: str, **overrides) -> Path:
    cfg = {
        "server_url": "http://localhost:0",
        "token": "test-token",
        "worker_name": name,
        "cpu_threads": 1,
        "gpu_enabled": False,
        "poll_interval_seconds": 5,
        "heartbeat_interval_seconds": 30,
        "fits_cache_dir": str(tmp_path / "fits_cache"),
    }
    cfg.update(overrides)
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


def _make(tmp_path, name, **overrides) -> ExoNitWorker:
    return ExoNitWorker(config_path=str(_write_config(tmp_path, name, **overrides)))


def test_no_task_pools_key_means_all_pools_allowed(tmp_path):
    """Backwards compat: existing configs that don't set task_pools
    must keep polling every queue."""
    w = _make(tmp_path, "worker-test-pools-default")
    try:
        assert w.task_pools is None
        assert w._pool_enabled("primary")
        assert w._pool_enabled("fpp_backfill")
        assert w._pool_enabled("exominer_backfill")
    finally:
        w._release_pid_lock()


def test_backfill_only_config_disables_primary(tmp_path):
    w = _make(
        tmp_path, "worker-test-pools-backfill",
        task_pools=["fpp_backfill", "exominer_backfill"],
    )
    try:
        assert w._pool_enabled("primary") is False
        assert w._pool_enabled("fpp_backfill") is True
        assert w._pool_enabled("exominer_backfill") is True
    finally:
        w._release_pid_lock()


def test_unknown_pool_is_ignored_with_warning(tmp_path, caplog):
    """Typo-tolerance: an unknown pool name shouldn't crash the worker —
    we just warn and drop it, so a config typo never silently disables
    the entire fleet."""
    w = _make(tmp_path, "worker-test-pools-typo",
              task_pools=["primary", "exominer_bckfill"])
    try:
        assert w.task_pools == {"primary"}
        assert w._pool_enabled("primary")
        assert w._pool_enabled("fpp_backfill") is False
    finally:
        w._release_pid_lock()


def test_run_loop_skips_primary_when_disabled(tmp_path):
    """Mechanism test: a worker with primary disabled must not call
    ``get_task()`` on its main poll iteration. Verifies the gate in
    ``run()`` rather than just the helper."""
    import threading
    w = _make(tmp_path, "worker-test-pools-skip-primary",
              task_pools=["exominer_backfill"])
    try:
        # Exit the loop after one iteration so the test doesn't hang.
        w.running = True

        def stop_after_one(*args, **kwargs):
            w.running = False
        with patch.object(w, "register"), \
             patch.object(w, "get_task") as get_task, \
             patch.object(w, "_try_fpp_backfill") as fpp, \
             patch.object(w, "_try_exominer_backfill",
                          side_effect=stop_after_one) as em, \
             patch.object(threading, "Thread"), \
             patch("time.sleep"):
            w.run()
        get_task.assert_not_called()
        fpp.assert_not_called()
        em.assert_called_once()
    finally:
        w._release_pid_lock()


def test_run_loop_skips_backfill_when_only_primary(tmp_path):
    import threading
    w = _make(tmp_path, "worker-test-pools-only-primary",
              task_pools=["primary"])
    try:
        w.running = True

        def stop_after_one(*args, **kwargs):
            w.running = False
            return None  # no task → loop sleeps then exits
        with patch.object(w, "register"), \
             patch.object(w, "get_task",
                          side_effect=stop_after_one) as get_task, \
             patch.object(w, "_try_fpp_backfill") as fpp, \
             patch.object(w, "_try_exominer_backfill") as em, \
             patch.object(threading, "Thread"), \
             patch("time.sleep"):
            w.run()
        get_task.assert_called_once()
        fpp.assert_not_called()
        em.assert_not_called()
    finally:
        w._release_pid_lock()
