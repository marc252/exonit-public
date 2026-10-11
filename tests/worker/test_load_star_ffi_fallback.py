"""Unit tests for ExoNitWorker._try_ffi_fallback (worker/worker.py).

The helper is the new last-resort step inside _load_star. It must:
  - Be a no-op (return None) when the config flag is off.
  - Fetch FFI from the server when the flag is on, return a 5-tuple
    matching _load_star's contract.
  - Reject responses with insufficient data (<100 finite points or
    <1 sector).
  - Tag meta with data_source='ffi' and lc_load_recovered=True on success.

Tests use a minimal Worker stub so we don't need an actual server.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class _FakeWorker:
    """Minimal stand-in: only the attributes _try_ffi_fallback reads."""
    def __init__(self, *, flag: bool, server="http://srv:8400",
                 token="test-token"):
        self.config = {"ffi_fallback_on_load_fail": flag}
        self.token = token  # mirrors ExoNitWorker.__init__
        self.server = server
        self.logger = logging.getLogger("test._try_ffi_fallback")
        self._headers = lambda: {"Authorization": f"Bearer {token}"}


def _import_helper():
    from worker.worker import ExoNitWorker
    return ExoNitWorker._try_ffi_fallback


def test_flag_off_returns_none_without_calling_fetch():
    helper = _import_helper()
    fake = _FakeWorker(flag=False)
    with patch("worker.worker.fetch_ffi") as mock_fetch:
        result = helper(fake, "12345", reason="metadata_404")
    assert result is None
    mock_fetch.assert_not_called()


def test_flag_on_server_200_returns_ffi_tuple():
    helper = _import_helper()
    fake = _FakeWorker(flag=True)
    n = 200
    fake_t = np.linspace(0.0, 27.0, n)
    fake_f = np.ones(n) + np.random.normal(0, 0.001, n)
    fake_q = np.zeros(n, dtype=int)

    metadata_resp = MagicMock(status_code=200)
    metadata_resp.json.return_value = {
        "ra": 100.5, "dec": -25.0, "tmag": 11.2, "sectors": [1, 2]}

    with patch("worker.worker.fetch_ffi",
               return_value=(fake_t, fake_f, fake_q)) as mock_fetch, \
         patch("worker.worker.requests.get",
               return_value=metadata_resp):
        result = helper(fake, "12345", reason="metadata_404")

    assert result is not None
    time_arr, flux_arr, raw_flux_arr, n_sectors, meta = result
    assert len(time_arr) == n
    assert len(flux_arr) == n
    assert len(raw_flux_arr) == n
    np.testing.assert_array_equal(raw_flux_arr, flux_arr)  # FFI: raw=flux copy
    assert n_sectors >= 1
    assert meta["data_source"] == "ffi"
    assert meta["lc_load_recovered"] is True
    assert meta["tic_id"] == "12345"
    mock_fetch.assert_called_once()


def test_flag_on_server_404_returns_none(caplog):
    helper = _import_helper()
    fake = _FakeWorker(flag=True)
    with patch("worker.worker.fetch_ffi", return_value=None), \
         caplog.at_level(logging.INFO, logger="test._try_ffi_fallback"):
        result = helper(fake, "12345", reason="fits_download_fail")
    assert result is None
    assert any("FFI fallback also failed" in rec.message
               for rec in caplog.records)


def test_flag_on_too_few_points_rejected(caplog):
    helper = _import_helper()
    fake = _FakeWorker(flag=True)
    short_t = np.linspace(0.0, 1.0, 50)  # only 50 points (<100 threshold)
    short_f = np.ones(50)
    short_q = np.zeros(50, dtype=int)
    with patch("worker.worker.fetch_ffi",
               return_value=(short_t, short_f, short_q)), \
         caplog.at_level(logging.INFO, logger="test._try_ffi_fallback"):
        result = helper(fake, "12345", reason="lightkurve_parse")
    assert result is None
    assert any("rejected" in rec.message.lower() for rec in caplog.records)


def test_flag_on_fetch_raises_returns_none(caplog):
    helper = _import_helper()
    fake = _FakeWorker(flag=True)
    with patch("worker.worker.fetch_ffi",
               side_effect=RuntimeError("boom")), \
         caplog.at_level(logging.WARNING, logger="test._try_ffi_fallback"):
        result = helper(fake, "12345", reason="metadata_404")
    assert result is None
    assert any("raised" in rec.message.lower() for rec in caplog.records)


def test_load_star_uses_fallback_when_metadata_404(monkeypatch, tmp_path):
    """Drive ExoNitWorker._load_star end-to-end: metadata 404 -> FFI fallback
    -> 5-tuple with data_source='ffi'."""
    from worker.worker import ExoNitWorker
    import worker.worker as worker_mod

    # Build a worker with the flag on; bypass the heavy __init__ via __new__
    w = ExoNitWorker.__new__(ExoNitWorker)
    w.config = {"ffi_fallback_on_load_fail": True,
                "worker_token": "tok"}
    w.token = "tok"
    w.server = "http://srv:8400"
    w.logger = logging.getLogger("test_load_star")
    w.fits_cache_dir = str(tmp_path)
    w._headers = lambda: {"Authorization": "Bearer tok"}

    # First /api/metadata/ call (in _load_star) returns 404; later best-effort
    # call inside _try_ffi_fallback also returns 404. Both forced via patch.
    not_found = MagicMock(status_code=404)
    monkeypatch.setattr(worker_mod.requests, "get",
                        MagicMock(return_value=not_found))

    n = 150
    monkeypatch.setattr(worker_mod, "fetch_ffi",
                        MagicMock(return_value=(
                            np.linspace(0.0, 27.0, n),
                            np.ones(n),
                            np.zeros(n, dtype=int))))

    t, f, raw, n_sec, meta = w._load_star("87654")
    assert t is not None
    assert len(t) == n
    assert meta["data_source"] == "ffi"
    assert meta["lc_load_recovered"] is True


def test_process_task_includes_lc_load_recovered_in_result(monkeypatch):
    """When _load_star returns meta['lc_load_recovered']=True, the result
    dict that process_task returns must contain the same flag."""
    from worker.worker import ExoNitWorker
    import worker.worker as worker_mod

    w = ExoNitWorker.__new__(ExoNitWorker)
    w.config = {"ffi_fallback_on_load_fail": True,
                "worker_token": "tok",
                "max_planets": 0,  # short-circuit BLS to keep test fast
                "run_injection": False, "run_fpp": False}
    w.server = "http://srv:8400"
    w.token = "tok"
    w.logger = logging.getLogger("test_process_task")
    w.status = "idle"
    w.current_tic = None
    w.current_phase = None
    w.phase_progress = None
    w.last_error = ""
    w.ui = None
    w._star_start_time = 0.0
    w._stop_requested = False
    w.running = True
    w.gpu_enabled = False
    w.stars_completed = 0
    w.total_processing_time = 0.0
    w.name = "test-worker"
    w.worker_id = None
    from collections import deque
    w.recent_timings = deque(maxlen=5)
    w._headers = lambda: {"Authorization": "Bearer tok"}
    w.classifier = None
    w.classifier_ffi = None

    # Have _load_star return a recovered-via-FFI 5-tuple directly.
    n = 200
    monkeypatch.setattr(
        ExoNitWorker, "_load_star",
        lambda self, tic: (
            np.linspace(0.0, 27.0, n),
            np.ones(n),
            np.ones(n),
            1,
            {"data_source": "ffi", "lc_load_recovered": True,
             "tic_id": tic, "ra": None, "dec": None, "tmag": None,
             "sectors": [1]}))

    task = {"task_id": "t-x", "tic_id": "87654", "config": {}}
    result = w.process_task(task)
    assert result["lc_load_recovered"] is True
    assert result["task_id"] == "t-x"
