"""Tests for the cross-process download-state fix (fix/download-state-shared).

Bug: with serve --workers N, POST /api/discovery/download stores state in a
per-process dict; GET /api/discovery/download/status in a sibling process
returns never_run=True because its dict is empty.

Fix: cross-process metadata (pid, log_path, started_at) is stored in
shared_state (Redis in prod, MemoryBackend in tests).  pid-liveness is
derived via psutil instead of proc.poll() (which only works in the parent).

These tests reproduce the multi-process scenario by directly writing to
shared_state from one "process" (simulated by the test) and reading via
the status endpoint from another (the TestClient), with the local
_discovery_download dict deliberately empty.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixture: isolated app with forced MemoryBackend (no Redis needed)
# ---------------------------------------------------------------------------

@pytest.fixture
def dl_client(tmp_path, monkeypatch):
    """Isolated FastAPI TestClient with MemoryBackend shared_state."""
    monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv(
        "EXONIT_PIPELINE_STATE_FILE",
        str(tmp_path / "pipeline_state.json"),
    )
    (tmp_path / "reports").mkdir(parents=True)

    from exonit.server import create_app
    app = create_app()

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        yield client, tmp_path, app


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_DOWNLOAD_STATE_KEY = "discovery:download:state"
_DL_FIELD = "current"


def _write_dl_meta(pid: int, log_path: str, started_at: float | None = None):
    """Write download metadata directly to shared_state (simulates the
    worker process that handled the POST).
    """
    from exonit.shared_state import get_shared_state
    st = get_shared_state()
    st.hset_json(_DOWNLOAD_STATE_KEY, _DL_FIELD, {
        "pid": pid,
        "log_path": log_path,
        "started_at": started_at or time.time(),
    })


def _clear_dl_meta():
    """Remove download metadata from shared_state."""
    from exonit.shared_state import get_shared_state
    st = get_shared_state()
    st.hdel(_DOWNLOAD_STATE_KEY, _DL_FIELD)


# ---------------------------------------------------------------------------
# (a) status returns progress parsing a log when pid is "alive"
# ---------------------------------------------------------------------------

class TestStatusFromSharedState:

    def test_status_parses_log_when_pid_alive(self, dl_client, tmp_path):
        """GET /status reads log_path from shared_state and parses it."""
        client, _, _ = dl_client

        log_path = tmp_path / "fake.log"
        log_path.write_text(
            "[100/5000] done=100 skip=2 fail=1 4.5/s eta=10.2min\n"
            "[500/5000] done=500 skip=4 fail=2 5.1/s eta=8.7min\n",
            encoding="utf-8",
        )
        fake_pid = 99999
        _write_dl_meta(pid=fake_pid, log_path=str(log_path))

        mock_psutil_proc = MagicMock()
        mock_psutil_proc.cmdline.return_value = ["python", "discovery_download.py"]
        with patch("psutil.pid_exists", return_value=True), \
             patch("psutil.Process", return_value=mock_psutil_proc):
            data = client.get("/api/discovery/download/status").json()

        assert data["running"] is True
        assert data["done"] == 500
        assert data["total"] == 5000
        assert data["skipped"] == 4
        assert data["failed"] == 2

    # -----------------------------------------------------------------------
    # (b) THE BUG TEST: a different "process" (empty local dict) sees the state
    # -----------------------------------------------------------------------

    def test_sibling_process_sees_state_without_local_proc(self, dl_client, tmp_path):
        """Reproduces the multi-process bug.

        Scenario:
          - Process A (POST worker): stores pid=12345 + log_path in shared_state.
            The local _discovery_download dict is NOT set (simulates a sibling
            worker whose per-process dict is empty).
          - Process B (GET worker): calls GET /status.
            Before the fix: proc=None → returned never_run=True.
            After the fix: reads from shared_state → returns running state.
        """
        client, tmp_path2, app = dl_client

        log_path = tmp_path2 / "run.log"
        log_path.write_text(
            "[200/1000] done=200 skip=1 fail=0 3.0/s eta=2.0min\n",
            encoding="utf-8",
        )
        fake_pid = 88888
        # Simulate "Process A" writing to shared_state directly (bypassing POST).
        _write_dl_meta(pid=fake_pid, log_path=str(log_path))

        # Confirm the local per-process dict is empty (proc=None).
        # This simulates being in a sibling process that never ran the POST.
        local_dict = app.state.discovery_download
        assert local_dict.get("proc") is None, (
            "Test setup error: local proc should be None to simulate sibling process"
        )

        mock_psutil_proc = MagicMock()
        mock_psutil_proc.cmdline.return_value = ["python", "discovery_download.py"]
        with patch("psutil.pid_exists", return_value=True), \
             patch("psutil.Process", return_value=mock_psutil_proc):
            data = client.get("/api/discovery/download/status").json()

        # After the fix: must NOT be never_run; must show running state + log data.
        assert not data.get("never_run"), (
            f"Bug still present: sibling process returned never_run=True. "
            f"Full response: {data}"
        )
        assert data["running"] is True
        assert data["done"] == 200
        assert data["total"] == 1000

    # -----------------------------------------------------------------------
    # (c) never_run when no state in shared_state
    # -----------------------------------------------------------------------

    def test_never_run_when_no_shared_state(self, dl_client):
        """GET /status with no state in shared_state → never_run=True."""
        client, _, _ = dl_client
        _clear_dl_meta()

        data = client.get("/api/discovery/download/status").json()
        assert data["running"] is False
        assert data.get("never_run") is True

    # -----------------------------------------------------------------------
    # (d) guard anti-double returns 409 when pid alive in shared_state
    # -----------------------------------------------------------------------

    def test_guard_returns_409_when_pid_alive_in_shared_state(self, dl_client, tmp_path):
        """POST /download returns 409 if shared_state has a live pid.

        Simulates a sibling worker trying to start a new download while
        Process A already registered a running pid in shared_state.
        """
        client, _, _ = dl_client

        fake_pid = 77777
        _write_dl_meta(pid=fake_pid, log_path="/tmp/fake.log")

        mock_psutil_proc = MagicMock()
        mock_psutil_proc.cmdline.return_value = ["python", "discovery_download.py"]
        with patch("psutil.pid_exists", return_value=True), \
             patch("psutil.Process", return_value=mock_psutil_proc):
            r = client.post(
                "/api/discovery/download", json={"sector": 1, "count": 100}
            )

        assert r.status_code == 409
        assert r.json()["status"] == "already_running"

    # -----------------------------------------------------------------------
    # (e) finished derived from [finish] in log
    # -----------------------------------------------------------------------

    def test_finished_derived_from_finish_line_in_log(self, dl_client, tmp_path):
        """returncode=0 when pid is dead and log has [finish] line."""
        client, tmp_path2, _ = dl_client

        log_path = tmp_path2 / "done.log"
        log_path.write_text(
            "[100/200] done=100 skip=0 fail=0 5.0/s eta=0.5min\n"
            "[finish] done=200 skip=0 fail=0 in 0.3min\n",
            encoding="utf-8",
        )
        fake_pid = 66666
        _write_dl_meta(pid=fake_pid, log_path=str(log_path))

        # Pid is dead (process finished).
        with patch("psutil.pid_exists", return_value=False):
            data = client.get("/api/discovery/download/status").json()

        assert data["running"] is False
        assert data["done"] == 200
        # Cross-process returncode derived from [finish] line → 0.
        assert data["returncode"] == 0

    def test_returncode_none_when_died_without_finish(self, dl_client, tmp_path):
        """returncode=None when pid is dead but log has no [finish] line."""
        client, tmp_path2, _ = dl_client

        log_path = tmp_path2 / "crashed.log"
        log_path.write_text(
            "[50/200] done=50 skip=0 fail=0 5.0/s eta=1.0min\n"
            "Traceback (most recent call last):\n"
            "  ...\n",
            encoding="utf-8",
        )
        fake_pid = 55555
        _write_dl_meta(pid=fake_pid, log_path=str(log_path))

        with patch("psutil.pid_exists", return_value=False):
            data = client.get("/api/discovery/download/status").json()

        assert data["running"] is False
        assert data["returncode"] is None  # can't know cross-process without [finish]

    # -----------------------------------------------------------------------
    # (f) new POST overwrites shared_state (no stale state after re-start)
    # -----------------------------------------------------------------------

    def test_new_post_overwrites_shared_state(self, dl_client, tmp_path):
        """A new POST replaces the old shared_state entry cleanly."""
        client, _, _ = dl_client

        # Simulate an old finished download in shared_state.
        _write_dl_meta(pid=44444, log_path="/tmp/old.log", started_at=1.0)

        mock_proc = MagicMock()
        mock_proc.pid = 33333
        mock_proc.returncode = None
        mock_proc.poll.return_value = None  # still running

        mock_psutil_proc = MagicMock()
        mock_psutil_proc.cmdline.return_value = ["python", "discovery_download.py"]
        # First psutil.pid_exists call is for the guard check (old pid=44444 dead).
        # Second would be from the status endpoint.
        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("psutil.pid_exists", return_value=False), \
             patch("psutil.Process", return_value=mock_psutil_proc):
            r = client.post("/api/discovery/download", json={"sector": 1, "count": 50})

        assert r.status_code == 200
        assert r.json()["pid"] == 33333

        # Shared state must have the new pid.
        from exonit.shared_state import get_shared_state
        meta = get_shared_state().hget_json(_DOWNLOAD_STATE_KEY, _DL_FIELD)
        assert meta is not None
        assert meta["pid"] == 33333
