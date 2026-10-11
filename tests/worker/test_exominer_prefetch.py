"""ExoMiner backfill prefetch pipeline tests.

Same producer/consumer shape as test_fpp_prefetch but for the ExoMiner
path, where the prefetched payload includes pre-downloaded DV XML +
TICv8 + Gaia + FITS files (the slow HTTP-bound portion) and the main
loop runs preprocess + ONNX inference on top.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "worker") not in sys.path:
    sys.path.append(str(ROOT / "worker"))

from worker.worker import ExoNitWorker  # noqa: E402


def _write_config(tmp_path: Path, name: str) -> Path:
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
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


def _make_worker(tmp_path: Path, name: str) -> ExoNitWorker:
    return ExoNitWorker(config_path=str(_write_config(tmp_path, name)))


def _fake_prepared_full(tic: int = 1):
    return {
        "task": {
            "candidate_id": 1000 + tic,
            "tic_id": tic,
            "candidate_num": 1,
            "period_days": 3.14,
            "t0": 1325.0,
            "duration_days": 0.1,
            "depth_ppm": 500,
            "snr": 12.5,
        },
        "cand": {
            "candidate_num": 1,
            "period_days": 3.14,
            "t0": 1325.0,
            "duration_days": 0.1,
            "depth_ppm": 500,
        },
        "prepared_inputs": {
            "tic_id": str(tic),
            "sector_run": "1-36",
            "dv_xml": b"<DV/>",
            "ticv8": {},
            "gaia": {},
            "lc_paths": [],
            "tmp_root": Path("/tmp/fake-exominer"),
        },
    }


def test_prefetched_consumed_no_sync_fetch(tmp_path):
    """Queue has a prepared payload → consume it, skip sync fetch."""
    w = _make_worker(tmp_path, "worker-test-em-prefetch-consume")
    try:
        prepared = _fake_prepared_full(tic=42)
        w._em_prefetch_slot._q.put_nowait(prepared)

        with patch.object(w, "_exominer_backfill_available", return_value=True), \
             patch.object(w, "_fetch_and_prepare_exominer") as fetch, \
             patch.object(w, "_process_prepared_exominer") as proc, \
             patch.object(threading, "Thread"):
            w._try_exominer_backfill()

        fetch.assert_not_called()
        proc.assert_called_once_with(prepared)
    finally:
        w._release_pid_lock()


def test_empty_queue_falls_back_to_sync(tmp_path):
    """First iteration / drained queue → synchronous prepare + process."""
    w = _make_worker(tmp_path, "worker-test-em-prefetch-sync")
    try:
        prepared = _fake_prepared_full(tic=7)
        with patch.object(w, "_exominer_backfill_available", return_value=True), \
             patch.object(w, "_fetch_and_prepare_exominer",
                          return_value=prepared) as fetch, \
             patch.object(w, "_process_prepared_exominer") as proc, \
             patch.object(threading, "Thread"):
            w._try_exominer_backfill()

        fetch.assert_called_once()
        proc.assert_called_once_with(prepared)
    finally:
        w._release_pid_lock()


def test_no_task_quiet_return(tmp_path):
    """No task available anywhere → no consumer call, no prefetch thread."""
    w = _make_worker(tmp_path, "worker-test-em-prefetch-noop")
    try:
        with patch.object(w, "_exominer_backfill_available", return_value=True), \
             patch.object(w, "_fetch_and_prepare_exominer",
                          return_value=None) as fetch, \
             patch.object(w, "_process_prepared_exominer") as proc, \
             patch.object(threading, "Thread") as Thread:
            w._try_exominer_backfill()

        fetch.assert_called_once()
        proc.assert_not_called()
        Thread.assert_not_called()
    finally:
        w._release_pid_lock()


def test_disabled_when_capability_missing(tmp_path):
    """If the worker lacks the ExoMiner model/hook, never poll the queue."""
    w = _make_worker(tmp_path, "worker-test-em-prefetch-disabled")
    try:
        with patch.object(w, "_exominer_backfill_available", return_value=False), \
             patch.object(w, "_fetch_and_prepare_exominer") as fetch:
            w._try_exominer_backfill()
        fetch.assert_not_called()
    finally:
        w._release_pid_lock()


def test_enqueue_single_flight_lock_blocks_concurrent_prefetch(tmp_path):
    """Concurrent _em_prefetch_slot._enqueue_next() calls must
    single-flight: second call exits without fetching while the first
    holds the lock."""
    w = _make_worker(tmp_path, "worker-test-em-prefetch-singleflight")
    try:
        call_count = {"n": 0}
        gate = threading.Event()
        release = threading.Event()

        def slow_fetch():
            call_count["n"] += 1
            gate.set()
            release.wait(2)
            return None

        with patch.object(w, "_fetch_and_prepare_exominer",
                          side_effect=slow_fetch):
            t1 = threading.Thread(target=w._em_prefetch_slot._enqueue_next)
            t1.start()
            assert gate.wait(2), "first thread never started fetch"
            w._em_prefetch_slot._enqueue_next()
            assert call_count["n"] == 1, (
                f"single-flight broken: got {call_count['n']} fetches")
            release.set()
            t1.join(2)
    finally:
        w._release_pid_lock()


def test_enqueue_release_on_race(tmp_path):
    """Producer thread loses race with a sibling: must release the slot
    server-side AND rmtree the FITS scratch dir so we don't leak MB on
    every loop."""
    w = _make_worker(tmp_path, "worker-test-em-prefetch-race")
    try:
        prepared = _fake_prepared_full(tic=99)

        def fake_fetch():
            # Sibling prefetch landed in the queue before we could
            w._em_prefetch_slot._q.put_nowait(_fake_prepared_full(tic=88))
            return prepared

        with patch.object(w, "_fetch_and_prepare_exominer",
                          side_effect=fake_fetch), \
             patch.object(w, "_report_exominer_skipped") as release, \
             patch("shutil.rmtree") as rmtree:
            w._em_prefetch_slot._enqueue_next()

        release.assert_called_once_with(
            prepared["task"]["candidate_id"], prepared["task"]["tic_id"])
        rmtree.assert_called()
    finally:
        w._release_pid_lock()
