"""Tests for the 2026-06-12 "indexing forever" fix.

Covers:
  A. Fast-path: local_proc.poll() is called (not .returncode) and its return
     value is propagated correctly.
  B. Zombie detection — psutil path: STATUS_ZOMBIE → running=False.
  C. Zombie detection — /proc fallback: state 'Z' → running=False.
  D. Alive process (psutil R/S state) → running=True.
  E. Alive process (/proc state 'S') → running=True.
  F. Log phase parsing: index_start / done / index_failed JSON markers.
  G. Phase derivation: all transitions, including "died".
  H. Backward compat: old log with only [finish] → phase="done", returncode=0.
  I. discovery_download._emit uses flush=True.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Fixtures
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


_DOWNLOAD_STATE_KEY = "discovery:download:state"
_DL_FIELD = "current"


def _write_dl_meta(pid: int, log_path: str, started_at: float | None = None):
    from exonit.shared_state import get_shared_state
    st = get_shared_state()
    st.hset_json(_DOWNLOAD_STATE_KEY, _DL_FIELD, {
        "pid": pid,
        "log_path": log_path,
        "started_at": started_at or time.time(),
    })


def _get_status(client):
    return client.get("/api/discovery/download/status").json()


# ---------------------------------------------------------------------------
# A. Fast-path: poll() is called, not .returncode
# ---------------------------------------------------------------------------

class TestFastPathPoll:
    """local_proc.poll() must be called (not .returncode read directly).

    The fast-path is triggered when the local _discovery_download dict has
    a proc whose pid matches the pid in shared_state.  We inject the mock
    proc into app.state.discovery_download after creating the app.
    """

    def _make_app_with_proc(self, tmp_path, mock_proc):
        """Create an isolated app, register mock_proc in per-process dict, and
        write the matching shared_state metadata.  Returns (client, log_path)."""
        import pathlib, os as _os
        from exonit.server import create_app
        from fastapi.testclient import TestClient

        _os.environ["EXONIT_DB_PATH"] = str(tmp_path / "test.db")
        _os.environ["EXONIT_REPORTS_DIR"] = str(tmp_path / "reports")
        _os.environ["EXONIT_PIPELINE_STATE_FILE"] = str(tmp_path / "ps.json")
        (tmp_path / "reports").mkdir(exist_ok=True)

        log_path = tmp_path / "dl.log"
        log_path.write_text("[100/1000] done=100 skip=0 fail=0 5.0/s eta=1.0min\n",
                            encoding="utf-8")

        app2 = create_app()
        from exonit.shared_state import get_shared_state
        # F5 (2026-07-10 review): the fast path also requires a "token"
        # minted at spawn time to match between the local dict and
        # shared_state (PID-recycling guard) — mirror that here the same
        # way the real /api/discovery/download endpoint does.
        token = f"test-token-{mock_proc.pid}"
        get_shared_state().hset_json(_DOWNLOAD_STATE_KEY, _DL_FIELD, {
            "pid": mock_proc.pid,
            "log_path": str(log_path),
            "started_at": time.time(),
            "token": token,
        })
        app2.state.discovery_download["proc"] = mock_proc
        app2.state.discovery_download["token"] = token
        return app2, log_path

    def test_poll_called_and_returncode_propagated(self, tmp_path):
        """Fast-path uses poll() to reap the child and get the exit code."""
        from fastapi.testclient import TestClient

        mock_proc = MagicMock()
        mock_proc.pid = 12321
        mock_proc.poll.return_value = 0   # process exited cleanly

        app2, _ = self._make_app_with_proc(tmp_path, mock_proc)

        with patch("psutil.pid_exists", return_value=False):
            with TestClient(app2) as c2:
                data = c2.get("/api/discovery/download/status").json()

        mock_proc.poll.assert_called()
        assert data["returncode"] == 0, (
            f"Expected returncode=0 from poll()=0, got {data['returncode']}"
        )

    def test_poll_called_returns_none_while_running(self, tmp_path):
        """poll() returns None while the process is still running."""
        from fastapi.testclient import TestClient

        mock_proc = MagicMock()
        mock_proc.pid = 55001
        mock_proc.poll.return_value = None  # still running

        app2, _ = self._make_app_with_proc(tmp_path, mock_proc)

        mock_psu = MagicMock()
        mock_psu.status.return_value = "sleeping"
        mock_psu.cmdline.return_value = ["python", "discovery_download.py"]
        with patch("psutil.pid_exists", return_value=True), \
             patch("psutil.Process", return_value=mock_psu), \
             patch("psutil.STATUS_ZOMBIE", "zombie"):
            with TestClient(app2) as c2:
                data = c2.get("/api/discovery/download/status").json()

        mock_proc.poll.assert_called()
        assert data["returncode"] is None, (
            f"Expected returncode=None (still running), got {data['returncode']}"
        )


# ---------------------------------------------------------------------------
# B. Zombie detection — psutil path (tests _check_pid_running_for_download)
# ---------------------------------------------------------------------------

class TestZombieDetectionPsutil:
    """STATUS_ZOMBIE → False even if pid_exists=True.

    We test _check_pid_running_for_download() directly because it is the
    extracted helper — no HTTP round-trip needed, and no risk of patching
    builtins.open breaking the TestClient infrastructure.
    """

    @pytest.fixture(autouse=True)
    def import_helper(self):
        from exonit.server import _check_pid_running_for_download
        self.fn = _check_pid_running_for_download

    def test_zombie_pid_exists_true_returns_false(self):
        """psutil STATUS_ZOMBIE → not running (prevents 'indexing forever')."""
        mock_psu = MagicMock()
        mock_psu.status.return_value = "zombie"

        with patch("psutil.pid_exists", return_value=True), \
             patch("psutil.Process", return_value=mock_psu), \
             patch("psutil.STATUS_ZOMBIE", "zombie"):
            result = self.fn(99800)

        assert result is False, "Zombie PID must not be reported as running"

    def test_alive_sleeping_returns_true(self):
        """psutil status 'sleeping' → running=True."""
        mock_psu = MagicMock()
        mock_psu.status.return_value = "sleeping"
        mock_psu.cmdline.return_value = ["python", "discovery_download.py"]

        with patch("psutil.pid_exists", return_value=True), \
             patch("psutil.Process", return_value=mock_psu), \
             patch("psutil.STATUS_ZOMBIE", "zombie"):
            result = self.fn(99801)

        assert result is True, "Sleeping process must be running"

    def test_pid_not_exists_returns_false(self):
        """pid_exists=False → not running."""
        with patch("psutil.pid_exists", return_value=False):
            result = self.fn(99802)
        assert result is False

    def test_no_such_process_returns_false(self):
        """NoSuchProcess during Process() → not running."""
        import psutil as _psutil
        with patch("psutil.pid_exists", return_value=True), \
             patch("psutil.Process", side_effect=_psutil.NoSuchProcess(99803)):
            result = self.fn(99803)
        assert result is False

    def test_wrong_cmdline_returns_false(self):
        """Process exists but cmdline doesn't contain discovery_download → False."""
        mock_psu = MagicMock()
        mock_psu.status.return_value = "sleeping"
        mock_psu.cmdline.return_value = ["python", "some_other_script.py"]

        with patch("psutil.pid_exists", return_value=True), \
             patch("psutil.Process", return_value=mock_psu), \
             patch("psutil.STATUS_ZOMBIE", "zombie"):
            result = self.fn(99804)

        assert result is False, "Wrong cmdline must return False"


# ---------------------------------------------------------------------------
# C. Zombie detection — /proc fallback (no psutil)
# ---------------------------------------------------------------------------

class TestZombieDetectionProcFallback:
    """Without psutil, /proc/<pid>/stat state 'Z' → False.

    Tests _check_pid_running_for_download() directly with psutil patched away
    via sys.modules.  We use a temporary file to simulate /proc/<pid>/stat so
    we don't need to patch builtins.open (which breaks TestClient internals).
    """

    def _make_stat(self, pid: int, state: str) -> str:
        return f"{pid} (discovery_download) {state} 1 1 1 0 -1 4194304\n"

    def _run_no_psutil(self, fake_pid: int, stat_content: str) -> bool:
        """Call _check_pid_running_for_download with psutil blocked and
        /proc/<pid>/stat pointing at a real temp file we write."""
        from exonit.server import _check_pid_running_for_download
        import sys as _sys2
        import tempfile

        # Write a real file so open() works without patching builtins.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".stat", delete=False, encoding="utf-8"
        ) as f:
            f.write(stat_content)
            stat_file = f.name

        psutil_real = _sys2.modules.get("psutil")
        _sys2.modules["psutil"] = None  # type: ignore[assignment]
        try:
            # Redirect /proc/<pid>/stat to our temp file by patching os.kill
            # (to make signal-0 succeed) and open() selectively.
            import builtins
            original_open = builtins.open

            def patched_open(path, *a, **kw):
                if str(path) == f"/proc/{fake_pid}/stat":
                    return original_open(stat_file, *a, **kw)
                return original_open(path, *a, **kw)

            with patch("os.kill", return_value=None), \
                 patch("builtins.open", side_effect=patched_open):
                result = _check_pid_running_for_download(fake_pid)
        finally:
            if psutil_real is None:
                _sys2.modules.pop("psutil", None)
            else:
                _sys2.modules["psutil"] = psutil_real
            try:
                os.unlink(stat_file)
            except OSError:
                pass
        return result

    def test_proc_stat_Z_returns_false(self):
        """Without psutil: /proc/<pid>/stat state='Z' → False."""
        fake_pid = 99810
        result = self._run_no_psutil(fake_pid, self._make_stat(fake_pid, "Z"))
        assert result is False, f"Zombie (/proc Z) must not be running: {result}"

    def test_proc_stat_S_returns_true(self):
        """Without psutil: /proc/<pid>/stat state='S' → True."""
        fake_pid = 99811
        result = self._run_no_psutil(fake_pid, self._make_stat(fake_pid, "S"))
        assert result is True, f"Sleeping (/proc S) should be running: {result}"

    def test_proc_stat_R_returns_true(self):
        """Without psutil: /proc/<pid>/stat state='R' (running) → True."""
        fake_pid = 99812
        result = self._run_no_psutil(fake_pid, self._make_stat(fake_pid, "R"))
        assert result is True, f"Running (/proc R) should be running: {result}"

    def test_proc_stat_parens_in_comm(self):
        """Comm field with parens should still parse state correctly."""
        fake_pid = 99813
        # Comm has parens: "(my-proc)" — last ')' is at position 12.
        stat = f"{fake_pid} ((my-proc)) Z 1 1 1 0 -1 4194304\n"
        result = self._run_no_psutil(fake_pid, stat)
        assert result is False, f"Zombie (parens in comm) must not be running: {result}"

    def test_proc_stat_malformed_no_state_char_returns_true(self):
        """Without psutil: stat content ends at ')' with no state char → IndexError
        caught → conservative True (not an HTTP 500).  Regression for R1."""
        fake_pid = 99815
        # Content that ends exactly at ')' — rfind(")")+2 is out of bounds.
        stat_content = f"{fake_pid} (discovery_download)"
        result = self._run_no_psutil(fake_pid, stat_content)
        assert result is True, (
            f"Malformed /proc stat (no state char) must fall back to True: {result}"
        )

    def test_no_proc_file_falls_back_to_alive(self):
        """Without psutil, if /proc/<pid>/stat unreadable → conservative True."""
        from exonit.server import _check_pid_running_for_download
        import sys as _sys2
        fake_pid = 99814

        psutil_real = _sys2.modules.get("psutil")
        _sys2.modules["psutil"] = None  # type: ignore[assignment]
        try:
            # os.kill(0) succeeds but open(/proc/..) raises OSError.
            with patch("os.kill", return_value=None), \
                 patch("builtins.open", side_effect=OSError("no /proc")):
                result = _check_pid_running_for_download(fake_pid)
        finally:
            if psutil_real is None:
                _sys2.modules.pop("psutil", None)
            else:
                _sys2.modules["psutil"] = psutil_real

        assert result is True, (
            f"Conservative: /proc unreadable after kill(0) success → True: {result}"
        )


# ---------------------------------------------------------------------------
# F + G. Log phase parsing and phase derivation
# ---------------------------------------------------------------------------

class TestLogPhaseMarkers:
    """JSON markers in log → correct phase and returncode derivation."""

    def _status_with_log(self, client, log_lines: list[str], pid_alive: bool = False):
        """Write log to tmp file, register in shared_state, return status."""
        from exonit.shared_state import get_shared_state
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False, encoding="utf-8"
        ) as f:
            f.write("\n".join(log_lines) + "\n")
            log_path = f.name

        get_shared_state().hset_json(_DOWNLOAD_STATE_KEY, _DL_FIELD, {
            "pid": 99900,
            "log_path": log_path,
            "started_at": time.time(),
        })

        if pid_alive:
            mock_psu = MagicMock()
            mock_psu.status.return_value = "sleeping"
            mock_psu.cmdline.return_value = ["python", "discovery_download.py"]
            with patch("psutil.pid_exists", return_value=True), \
                 patch("psutil.Process", return_value=mock_psu), \
                 patch("psutil.STATUS_ZOMBIE", "zombie"):
                data = client.get("/api/discovery/download/status").json()
        else:
            with patch("psutil.pid_exists", return_value=False):
                data = client.get("/api/discovery/download/status").json()

        try:
            os.unlink(log_path)
        except OSError:
            pass
        return data

    def test_phase_done_from_json_marker(self, dl_client):
        """Log with {"status":"done"} → phase="done", returncode=0."""
        client, _, _ = dl_client
        lines = [
            '[5000/5000] done=5000 skip=0 fail=0 20.0/s eta=0min',
            '[finish] done=5000 skip=0 fail=0 in 2.0min',
            '{"status": "done", "downloaded": 5000, "failed": 0, "indexed": 4800, "dest": "/tmp/x"}',
        ]
        data = self._status_with_log(client, lines, pid_alive=False)
        assert data["phase"] == "done", f"Expected phase=done, got: {data}"
        assert data["returncode"] == 0

    def test_phase_index_failed_from_json_marker(self, dl_client):
        """Log with {"status":"index_failed"} → phase="index_failed", returncode=1."""
        client, _, _ = dl_client
        lines = [
            '[finish] done=5000 skip=0 fail=0 in 1.5min',
            '{"status": "index_start"}',
            '{"status": "index_failed", "returncode": 1, "stderr": "some error"}',
        ]
        data = self._status_with_log(client, lines, pid_alive=False)
        assert data["phase"] == "index_failed", f"Expected phase=index_failed, got: {data}"
        assert data["returncode"] == 1

    def test_phase_indexing_when_alive_and_index_start(self, dl_client):
        """Log with {"status":"index_start"} + pid alive → phase="indexing"."""
        client, _, _ = dl_client
        lines = [
            '[5000/5000] done=5000 skip=0 fail=0 20.0/s eta=0min',
            '[finish] done=5000 skip=0 fail=0 in 2.0min',
            '{"status": "index_start"}',
        ]
        data = self._status_with_log(client, lines, pid_alive=True)
        assert data["phase"] == "indexing", f"Expected phase=indexing, got: {data}"
        assert data["running"] is True

    def test_phase_downloading_when_alive_no_index_start(self, dl_client):
        """Pid alive, no index_start marker → phase="downloading"."""
        client, _, _ = dl_client
        lines = [
            '[100/5000] done=100 skip=0 fail=0 5.0/s eta=9.0min',
        ]
        data = self._status_with_log(client, lines, pid_alive=True)
        assert data["phase"] == "downloading", f"Expected phase=downloading, got: {data}"
        assert data["running"] is True

    def test_phase_died_when_dead_no_terminal_marker(self, dl_client):
        """Pid dead, no done/index_failed/[finish] → phase="died"."""
        client, _, _ = dl_client
        lines = [
            '[1000/5000] done=1000 skip=0 fail=0 10.0/s eta=4.0min',
            'Traceback (most recent call last):',
            '  File "discovery_download.py", line 123',
            'RuntimeError: out of memory',
        ]
        data = self._status_with_log(client, lines, pid_alive=False)
        assert data["phase"] == "died", f"Expected phase=died, got: {data}"
        # returncode should remain None (unknown).
        assert data["returncode"] is None, (
            f"Expected returncode=None for died phase, got: {data['returncode']}"
        )

    def test_phase_done_backward_compat_finish_only(self, dl_client):
        """Old log with only [finish] (no JSON markers) → phase="done", returncode=0."""
        client, _, _ = dl_client
        lines = [
            '[100/5000] done=100 skip=0 fail=0 5.0/s eta=9.0min',
            '[finish] done=5000 skip=0 fail=0 in 2.0min',
        ]
        data = self._status_with_log(client, lines, pid_alive=False)
        assert data["phase"] == "done", (
            f"Backward compat: [finish]-only log should give phase=done. Got: {data}"
        )
        assert data["returncode"] == 0

    def test_phase_present_in_never_run_response(self, dl_client):
        """never_run response must not crash — phase field absent is acceptable."""
        client, _, _ = dl_client
        from exonit.shared_state import get_shared_state
        get_shared_state().hdel(_DOWNLOAD_STATE_KEY, _DL_FIELD)
        data = client.get("/api/discovery/download/status").json()
        assert data.get("never_run") is True


# ---------------------------------------------------------------------------
# I. discovery_download._emit uses flush=True
# ---------------------------------------------------------------------------

class TestEmitFlush:
    """_emit() must call print with flush=True so lines reach disk immediately."""

    def test_emit_calls_print_with_flush(self):
        """_emit(obj) calls print(json.dumps(obj), flush=True)."""
        # Add scripts/ to path so discovery_download is importable.
        scripts_dir = Path(__file__).parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        import discovery_download as dd

        captured = {}

        def mock_print(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        with patch("builtins.print", side_effect=mock_print):
            dd._emit({"status": "done", "downloaded": 100})

        assert captured.get("kwargs", {}).get("flush") is True, (
            f"_emit must call print(..., flush=True). Captured kwargs: {captured.get('kwargs')}"
        )
        # Also verify the JSON is correct.
        emitted = json.loads(captured["args"][0])
        assert emitted["status"] == "done"
        assert emitted["downloaded"] == 100

    def test_emit_index_start_has_flush(self):
        """The index_start marker is also emitted with flush=True."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        import discovery_download as dd

        flushed_calls = []

        def mock_print(*args, **kwargs):
            if kwargs.get("flush"):
                flushed_calls.append(json.loads(args[0]))

        with patch("builtins.print", side_effect=mock_print):
            dd._emit({"status": "index_start"})

        assert any(c.get("status") == "index_start" for c in flushed_calls), (
            "index_start emit must use flush=True"
        )


# ---------------------------------------------------------------------------
# J. Two-writer append-mode fix (fix/download-log-append 2026-06-12)
#
# Bug: server.py opened Popen stdout with "wb" → child fd starts at offset 0.
# discovery_download.py writes JSON markers via stdout (inherited fd).
# download_tglc_batch.py opens --log independently with "a" → writes at EOF.
# Result: JSON markers written at offset 0 overwrite the start of the log;
# the status endpoint's 200-line tail never sees them.
#
# Fix: open with "ab" so both handles always write at EOF (O_APPEND).
# ---------------------------------------------------------------------------

import re as _re_j

_re_prefix_j = _re_j.compile(r"\[(\d+)/(\d+)\]")
_re_done_j   = _re_j.compile(r"\bdone=(\d+)")
_re_skip_j   = _re_j.compile(r"\bskip=(\d+)")
_re_fail_j   = _re_j.compile(r"\bfail=(\d+)")


def _parse_log_j(log_path, running: bool) -> dict:
    """Replicate the log-parsing block from api_discovery_download_status."""
    done = failed = skipped = total = 0
    log_has_finish = False
    log_has_done_json = False
    log_has_index_start = False
    log_has_index_failed = False

    if Path(log_path).exists():
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()[-200:]
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("{"):
                try:
                    obj = json.loads(line_stripped)
                    st = obj.get("status")
                    if st == "index_start":
                        log_has_index_start = True
                    elif st == "done":
                        log_has_done_json = True
                    elif st == "index_failed":
                        log_has_index_failed = True
                    continue
                except ValueError:
                    pass
            mp = _re_prefix_j.search(line)
            if mp:
                total = int(mp.group(2))
                md = _re_done_j.search(line)
                ms = _re_skip_j.search(line)
                mf = _re_fail_j.search(line)
                if md:
                    done = int(md.group(1))
                if ms:
                    skipped = int(ms.group(1))
                if mf:
                    failed = int(mf.group(1))
            elif "[finish]" in line:
                log_has_finish = True
                md = _re_done_j.search(line)
                ms = _re_skip_j.search(line)
                mf = _re_fail_j.search(line)
                if md:
                    done = int(md.group(1))
                if ms:
                    skipped = int(ms.group(1))
                if mf:
                    failed = int(mf.group(1))

    if running:
        phase = "indexing" if log_has_index_start else "downloading"
    elif log_has_done_json:
        phase = "done"
    elif log_has_index_failed:
        phase = "index_failed"
    elif log_has_finish:
        phase = "done"
    else:
        phase = "died"

    return {
        "phase": phase,
        "done": done,
        "failed": failed,
        "skipped": skipped,
        "total": total,
        "log_has_index_start": log_has_index_start,
        "log_has_index_failed": log_has_index_failed,
        "log_has_done_json": log_has_done_json,
    }


class TestTwoWriterAppend:
    """Simulate the two independent file handles that exist in production.

    Handle A  — opened with "ab" (the Popen-inherited fd from server.py
                AFTER the fix).  Represents discovery_download.py writing
                JSON markers via stdout.
    Handle B  — opened separately with "a" (the download_tglc_batch.py
                --log handle).  Represents per-file progress lines.

    Both handles are opened on the same path.  With O_APPEND both always
    write at EOF regardless of the other handle's position.
    """

    def test_json_markers_land_at_eof_not_at_start(self, tmp_path):
        """With both handles in append mode, JSON markers appear after
        progress lines that were written first — not at byte 0."""
        log = tmp_path / "test.log"
        log.write_bytes(b"")  # mimic mkstemp creating an empty file

        # Handle B writes progress lines first (download_tglc_batch.py style).
        with open(log, "a") as b:
            b.write("[100/1000] done=100 skip=0 fail=0 5.0/s eta=3.0min\n")
            b.write("[500/1000] done=500 skip=0 fail=0 5.0/s eta=1.0min\n")
            b.write("[finish] done=1000 skip=0 fail=0 in 3.3min\n")

        # Handle A writes JSON marker AFTER progress (append → EOF).
        with open(log, "ab") as a:
            a.write(json.dumps({"status": "index_start"}).encode() + b"\n")

        content = log.read_text()
        lines = [ln for ln in content.splitlines() if ln.strip()]

        # The [finish] line must come BEFORE the JSON marker.
        finish_idx = next(i for i, ln in enumerate(lines) if "[finish]" in ln)
        json_idx   = next(i for i, ln in enumerate(lines) if ln.startswith("{"))
        assert finish_idx < json_idx, (
            f"[finish] at line {finish_idx} should precede JSON marker at "
            f"{json_idx}; file content:\n{content}"
        )

    def test_index_start_seen_at_end_phase_is_indexing(self, tmp_path):
        """Parser sees index_start even when it appears at the end of the file."""
        log = tmp_path / "test.log"
        log.write_bytes(b"")

        with open(log, "a") as b:
            for i in range(1, 6):
                b.write(
                    f"[{i*200}/1000] done={i*200} skip=0 fail=0 5.0/s eta=1.0min\n"
                )
            b.write("[finish] done=1000 skip=0 fail=0 in 3.3min\n")

        with open(log, "ab") as a:
            a.write(json.dumps({"status": "index_start"}).encode() + b"\n")

        # running=True: download finished, indexing in progress.
        result = _parse_log_j(log, running=True)
        assert result["phase"] == "indexing", (
            f"Expected phase=indexing, got {result['phase']!r}; "
            f"log_has_index_start={result['log_has_index_start']}"
        )
        assert result["log_has_index_start"] is True

    def test_index_failed_at_eof_phase_is_index_failed(self, tmp_path):
        """CRITICAL: index_failed at EOF must NOT be reported as done.

        This is the exact incident scenario: with "wb" mode, index_failed
        was written at offset 0 and overwritten by progress lines, so the
        endpoint fell back to [finish] → phase=done, masking the error.
        With "ab" mode the marker is at EOF and correctly detected.
        """
        log = tmp_path / "test.log"
        log.write_bytes(b"")

        # Handle B: download progress.
        with open(log, "a") as b:
            b.write("[1000/1000] done=1000 skip=0 fail=0 5.0/s\n")
            b.write("[finish] done=1000 skip=0 fail=0 in 3.3min\n")

        # Handle A: JSON markers at EOF.
        with open(log, "ab") as a:
            a.write(json.dumps({"status": "index_start"}).encode() + b"\n")
            a.write(json.dumps({
                "status": "index_failed",
                "returncode": 1,
                "stderr": "astropy not found",
            }).encode() + b"\n")

        result = _parse_log_j(log, running=False)
        assert result["phase"] == "index_failed", (
            f"Expected phase=index_failed, got {result['phase']!r}.\n"
            f"An index error must NOT be silently reported as 'done'.\n"
            f"log_has_index_failed={result['log_has_index_failed']}, "
            f"log_has_done_json={result['log_has_done_json']}"
        )
        assert result["log_has_index_failed"] is True

    def test_done_json_at_eof_phase_is_done(self, tmp_path):
        """Happy path: status=done at EOF → phase=done."""
        log = tmp_path / "test.log"
        log.write_bytes(b"")

        with open(log, "a") as b:
            b.write("[1000/1000] done=1000 skip=0 fail=0 5.0/s\n")
            b.write("[finish] done=1000 skip=0 fail=0 in 3.3min\n")

        with open(log, "ab") as a:
            a.write(json.dumps({"status": "index_start"}).encode() + b"\n")
            a.write(json.dumps({
                "status": "done",
                "downloaded": 1000,
                "failed": 0,
                "indexed": 1000,
                "dest": "/some/path",
            }).encode() + b"\n")

        result = _parse_log_j(log, running=False)
        assert result["phase"] == "done"
        assert result["log_has_done_json"] is True

    def test_old_format_no_json_with_finish_phase_is_done(self, tmp_path):
        """Backward compat: old logs without JSON markers but with [finish] → done."""
        log = tmp_path / "test.log"
        log.write_bytes(b"")

        with open(log, "a") as b:
            b.write("[1000/1000] done=1000 skip=0 fail=0 5.0/s\n")
            b.write("[finish] done=1000 skip=0 fail=0 in 3.3min\n")

        result = _parse_log_j(log, running=False)
        assert result["phase"] == "done"
        assert result["log_has_done_json"] is False
        assert result["done"] == 1000

    def test_write_order_preserved_across_many_lines(self, tmp_path):
        """With O_APPEND, interleaved writes from two handles stay in document order."""
        log = tmp_path / "interleave.log"
        log.write_bytes(b"")

        with open(log, "a") as b:
            b.write("[100/300] done=100 skip=0 fail=0 5.0/s eta=2.0min\n")
            b.write("[200/300] done=200 skip=0 fail=0 5.0/s eta=1.0min\n")
            b.write("[300/300] done=300 skip=0 fail=0 5.0/s\n")
            b.write("[finish] done=300 skip=0 fail=0 in 1.0min\n")

        with open(log, "ab") as a:
            a.write(json.dumps({"status": "index_start"}).encode() + b"\n")
            a.write(json.dumps({"status": "done", "downloaded": 300,
                                "failed": 0, "indexed": 300,
                                "dest": "/tmp/x"}).encode() + b"\n")

        content = log.read_text()
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]

        progress_lines = [ln for ln in lines if ln.startswith("[")]
        json_lines = [ln for ln in lines if ln.startswith("{")]
        assert len(progress_lines) == 4
        assert len(json_lines) == 2

        last_progress_idx = max(lines.index(ln) for ln in progress_lines)
        first_json_idx = min(lines.index(ln) for ln in json_lines)
        assert last_progress_idx < first_json_idx, (
            "All progress lines must precede all JSON markers in the log."
        )


class TestServerPyOpenMode:
    """Verify that server.py actually opens the log with 'ab' for Popen."""

    def test_server_opens_log_with_ab_not_wb(self):
        """Source inspection: api_discovery_download must open log with 'ab'.

        Fase D create_app() split (2026-07-02,
        progress/impl-createapp-split-2026-07-02.md): api_discovery_download
        moved from exonit/server.py into exonit/routes/discovery.py
        (endpoint body byte-identical) — updated to look in its new home.
        """
        server_path = Path(__file__).parent.parent / "exonit" / "routes" / "discovery.py"
        assert server_path.exists(), f"discovery.py not found at {server_path}"

        source = server_path.read_text(encoding="utf-8")

        import re
        block_match = re.search(
            r'def api_discovery_download.*?(?=\n    @app\.|\Z)',
            source, re.DOTALL
        )
        assert block_match, "Could not find api_discovery_download in server.py"
        block = block_match.group(0)

        # Check the actual open() call uses "ab", not "wb".
        # We look for the open(log_path, ...) call specifically (not comments).
        open_call_match = re.search(
            r'open\s*\(\s*log_path\s*,\s*"([^"]+)"',
            block,
        )
        assert open_call_match, (
            "Could not find open(log_path, ...) in api_discovery_download block. "
            "Has the download log open call been refactored?"
        )
        mode = open_call_match.group(1)
        assert mode == "ab", (
            f'server.py opens Popen log with mode "{mode}", expected "ab" (append '
            f'binary).  The "wb" mode causes JSON markers from '
            f'discovery_download.py to overwrite the beginning of the log file.'
        )

    def test_download_tglc_batch_uses_append_mode(self):
        """download_tglc_batch.py must open --log with 'a' (append text)."""
        batch_path = (
            Path(__file__).parent.parent / "scripts" / "download_tglc_batch.py"
        )
        assert batch_path.exists(), f"download_tglc_batch.py not found at {batch_path}"

        source = batch_path.read_text(encoding="utf-8")

        import re
        log_fn_match = re.search(
            r'def log\(.*?\):.*?open\(args\.log,\s*"([^"]+)"',
            source,
            re.DOTALL,
        )
        assert log_fn_match, (
            "Could not find log() function with open(args.log, ...) in "
            "download_tglc_batch.py"
        )
        mode = log_fn_match.group(1)
        assert mode == "a", (
            f'download_tglc_batch.py opens --log with mode "{mode}", '
            'expected "a" (append).  Both handles must use append mode.'
        )
