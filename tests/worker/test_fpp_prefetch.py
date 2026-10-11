"""FPP backfill prefetch pipeline tests.

The pipeline overlaps the FITS download for task N+1 with TRICERATOPS
scoring of task N — measured at <host> 2026-05-16, median gap
between tasks was 12s vs ~5s actual compute (so the worker spent more
time on serial setup than on real work). These tests pin down the
mechanics so a future refactor can't silently regress to the synchronous
behaviour.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "worker") not in sys.path:
    sys.path.append(str(ROOT / "worker"))

from worker.worker import ExoNitWorker  # noqa: E402


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
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


def _make_worker(tmp_path: Path, name: str) -> ExoNitWorker:
    return ExoNitWorker(config_path=str(_write_config(tmp_path, name)))


def _fake_prepared(tic: int = 1):
    return {
        "task": {
            "candidate_id": 100 + tic,
            "tic_id": tic,
            "candidate_num": 1,
            "period_days": 3.14,
            "t0": 1325.0,
            "duration_days": 0.1,
            "depth_ppm": 500,
            "snr": 12.5,
            "n_sectors": 1,
        },
        "time_arr": [0.0, 1.0, 2.0],
        "flux_arr": [1.0, 0.999, 1.0],
        "n_sectors": 1,
    }


def test_prefetched_task_is_consumed_without_sync_fetch(tmp_path):
    """When the queue is non-empty, _try_fpp_backfill must use the
    prefetched item and skip the synchronous fetch path entirely."""
    w = _make_worker(tmp_path, "worker-test-prefetch-consume")
    try:
        prepared = _fake_prepared(tic=42)
        w._fpp_prefetch_slot._q.put_nowait(prepared)

        with patch.object(w, "_fetch_and_load_fpp") as fetch, \
             patch.object(w, "_process_prepared_fpp") as proc, \
             patch.object(threading, "Thread"):
            w._try_fpp_backfill()

        fetch.assert_not_called()
        proc.assert_called_once_with(prepared)
    finally:
        w._release_pid_lock()


def test_empty_queue_falls_back_to_sync_fetch(tmp_path):
    """If nothing has been prefetched yet (first iteration, or main loop
    drained faster than the producer), _try_fpp_backfill must still
    return a task by fetching synchronously — otherwise a worker that
    just started would idle for the whole poll_interval."""
    w = _make_worker(tmp_path, "worker-test-prefetch-sync")
    try:
        prepared = _fake_prepared(tic=7)
        with patch.object(w, "_fetch_and_load_fpp", return_value=prepared) as fetch, \
             patch.object(w, "_process_prepared_fpp") as proc, \
             patch.object(threading, "Thread"):
            w._try_fpp_backfill()

        fetch.assert_called_once()
        proc.assert_called_once_with(prepared)
    finally:
        w._release_pid_lock()


def test_no_task_available_returns_quietly(tmp_path):
    """Queue empty AND server has nothing → _try_fpp_backfill must
    return without invoking _process_prepared_fpp and without spawning
    a prefetch thread (the next loop iter will retry on its own)."""
    w = _make_worker(tmp_path, "worker-test-prefetch-empty")
    try:
        with patch.object(w, "_fetch_and_load_fpp", return_value=None) as fetch, \
             patch.object(w, "_process_prepared_fpp") as proc, \
             patch.object(threading, "Thread") as Thread:
            w._try_fpp_backfill()

        fetch.assert_called_once()
        proc.assert_not_called()
        Thread.assert_not_called()
    finally:
        w._release_pid_lock()


def test_enqueue_next_fpp_puts_when_fetch_succeeds(tmp_path):
    w = _make_worker(tmp_path, "worker-test-prefetch-put")
    try:
        prepared = _fake_prepared(tic=11)
        with patch.object(w, "_fetch_and_load_fpp", return_value=prepared):
            w._fpp_prefetch_slot._enqueue_next()
        got = w._fpp_prefetch_slot._q.get_nowait()
        assert got is prepared
    finally:
        w._release_pid_lock()


def test_enqueue_next_fpp_noop_when_queue_full(tmp_path):
    """Producer must not block the main thread if the consumer is slow;
    skipping a put when the queue is already full is the correct back-
    pressure behaviour with maxsize=1."""
    w = _make_worker(tmp_path, "worker-test-prefetch-full")
    try:
        w._fpp_prefetch_slot._q.put_nowait(_fake_prepared(tic=1))
        with patch.object(w, "_fetch_and_load_fpp") as fetch:
            w._fpp_prefetch_slot._enqueue_next()
        fetch.assert_not_called()
    finally:
        w._release_pid_lock()


def test_enqueue_single_flight_lock_blocks_concurrent_prefetch(tmp_path):
    """Two _fpp_prefetch_slot._enqueue_next() calls racing must not
    produce two reservations: the second one bails on the tryacquire and
    exits without calling _fetch_and_load_fpp. Before this guard,
    `_fpp_assignments` server-side drifted to ~2.7× the worker count
    because every main-loop iteration spawned a fresh prefetch thread
    even when the previous one was still in flight."""
    w = _make_worker(tmp_path, "worker-test-prefetch-singleflight")
    try:
        call_count = {"n": 0}
        gate = threading.Event()
        release = threading.Event()

        def slow_fetch():
            call_count["n"] += 1
            gate.set()        # signal first call has started
            release.wait(2)   # hold the lock until the second call has tried
            return None       # producer returns nothing so the queue stays empty

        with patch.object(w, "_fetch_and_load_fpp", side_effect=slow_fetch):
            t1 = threading.Thread(target=w._fpp_prefetch_slot._enqueue_next)
            t1.start()
            assert gate.wait(2), "first thread never started fetch"
            # Second call should bail immediately on the lock and NOT
            # invoke _fetch_and_load_fpp.
            w._fpp_prefetch_slot._enqueue_next()
            assert call_count["n"] == 1, (
                f"expected single-flight lock to block second call, "
                f"got {call_count['n']} fetches")
            release.set()
            t1.join(2)
    finally:
        w._release_pid_lock()


def test_enqueue_releases_reservation_on_race(tmp_path):
    """If the queue went from empty (when we checked) to full (after the
    fetch+load), we must release the reservation server-side — sitting
    on it would starve a sibling worker. Race window is small but real
    when multiple prefetch threads are spawned in quick succession."""
    w = _make_worker(tmp_path, "worker-test-prefetch-race")
    try:
        prepared = _fake_prepared(tic=99)

        def fake_fetch():
            # Simulate another iteration filling the slot while we were
            # downloading FITS.
            w._fpp_prefetch_slot._q.put_nowait(_fake_prepared(tic=88))
            return prepared

        with patch.object(w, "_fetch_and_load_fpp", side_effect=fake_fetch), \
             patch.object(w, "_report_fpp_failed") as release:
            w._fpp_prefetch_slot._enqueue_next()

        release.assert_called_once_with(prepared["task"])
    finally:
        w._release_pid_lock()
