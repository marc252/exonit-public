"""Item 7 (2026-07-02 cleanup review): the success-path result (end of
process_task), the exception-path fallback (main run loop), and
_aborted_result used to each build a bespoke dict with a slightly
different key set. All three now start from ExoNitWorker._base_result()
and dict.update() their specifics on top.

This test pins down the two invariants that matter:
  1. _base_result's key set is a superset of what the three call sites
     needed (nothing server.py's /api/worker/result handler reads got
     dropped) — additive only, no renames.
  2. _aborted_result's actual output still carries every key it did
     before the refactor, with the same values for the fields it
     explicitly sets.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "worker") not in sys.path:
    sys.path.append(str(ROOT / "worker"))

from worker.worker import ExoNitWorker  # noqa: E402

# Every key exonit/server.py's worker_result handler reads via body.get(...)
# (task_id/tic_id/candidates/cnn_scores/status/ra/dec/tmag/n_sectors/
# n_datapoints|n_points/time_span_days/completeness/lc_load_recovered/
# timings/vetting_results is not read server-side but is worker-internal
# telemetry retained for parity) — see exonit/server.py:2197-2360.
_SERVER_READ_KEYS = {
    "task_id", "tic_id", "candidates", "cnn_scores", "status",
    "ra", "dec", "tmag", "n_sectors", "n_points", "time_span_days",
    "completeness", "lc_load_recovered", "timings", "processing_time_seconds",
    "gpu_used",
}


def _make_worker() -> ExoNitWorker:
    import logging
    w = ExoNitWorker.__new__(ExoNitWorker)
    w.logger = logging.getLogger("test_result_shape")
    return w


def test_base_result_covers_every_server_read_key():
    w = _make_worker()
    task = {"task_id": "abc123", "tic_id": "TIC1"}
    base = w._base_result(task)
    missing = _SERVER_READ_KEYS - set(base.keys())
    assert not missing, f"_base_result is missing keys server.py reads: {missing}"


def test_base_result_absent_values_are_explicit_none_not_missing():
    w = _make_worker()
    base = w._base_result({"task_id": "t1", "tic_id": "TIC1"})
    for key in ("time_span_days", "ra", "dec", "tmag", "timings"):
        assert key in base
        assert base[key] is None


def test_base_result_uses_get_and_never_raises_on_missing_task_keys():
    """task dicts without task_id/tic_id (shouldn't normally happen, but
    _aborted_result/exception-path build these from task.get(...) too)
    must not raise — a strict improvement over the pre-refactor code,
    which indexed task["task_id"]/task["tic_id"] directly."""
    w = _make_worker()
    base = w._base_result({})
    assert base["task_id"] is None
    assert base["tic_id"] is None


def test_aborted_result_carries_full_key_set_and_expected_values():
    w = _make_worker()
    task = {"task_id": "abc123", "tic_id": "TIC42"}
    t_start = time.time() - 5.0
    result = w._aborted_result(task, t_start, timings={"load": 1.0},
                                lc_load_recovered=True)

    missing = _SERVER_READ_KEYS - set(result.keys())
    assert not missing, f"_aborted_result is missing keys server.py reads: {missing}"

    assert result["task_id"] == "abc123"
    assert result["tic_id"] == "TIC42"
    assert result["status"] == "aborted"
    assert result["candidates"] == []
    assert result["timings"] == {"load": 1.0}
    assert result["lc_load_recovered"] is True
    assert result["gpu_used"] is False
    assert result["n_sectors"] == 0
    assert result["n_points"] == 0
    assert result["processing_time_seconds"] >= 4.9
