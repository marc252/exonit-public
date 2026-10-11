"""Tests for ``scripts/exonit_compactd.py`` — the stdlib-only HTTP sidecar
that supervises ``compact_completeness.py`` subprocess runs (Component 2 of
``docs/superpowers/specs/2026-09-09-compactd-service-design.md``).

The real compaction subprocess is replaced by a small fake script (written
to ``tmp_path`` by the ``fake_compact_script`` fixture below) — the cleanest
injection seam is ``JobManager.compact_script``, a plain constructor
parameter, so no env-var indirection or monkeypatching of the real script is
needed. The fake's behaviour (success/failure/how long it sleeps) is itself
controlled via env vars read by the fake script AT SUBPROCESS LAUNCH TIME
(``FAKE_COMPACT_RC``, ``FAKE_COMPACT_SLEEP``) — set with ``monkeypatch`` in
the parent test process, which ``subprocess.Popen`` (via ``os.environ.copy()``
in ``exonit_compactd.py``) inherits. The fake also echoes whatever
``EXONIT_PROGRESS_*`` env vars it received (Component 2 fix 2 — progress
passthrough from ``POST /compact``'s optional ``run_id``/``phase``/
``progress_url`` body fields) both into its stderr log line and into its
JSON summary, so tests can assert on either.

The service is started on an ephemeral port (``port=0``) in a background
thread per test, mirroring ``tests/test_kill_and_resume.py``'s
``stub_server`` fixture pattern.
"""

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import exonit_compactd as compactd  # noqa: E402


TOKEN = "test-secret-token"

FAKE_SCRIPT_SOURCE = '''\
import argparse, json, os, sys, time, logging

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("fake_compact")

p = argparse.ArgumentParser()
p.add_argument("--pg", required=True)
p.add_argument("--batch-rows", type=int, default=None)
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()

progress_run_id = os.environ.get("EXONIT_PROGRESS_RUN_ID")
progress_phase = os.environ.get("EXONIT_PROGRESS_PHASE")
progress_url = os.environ.get("EXONIT_PROGRESS_URL")

log.info("fake compact starting pg=%s batch_rows=%s dry_run=%s "
         "progress_run_id=%s progress_phase=%s progress_url=%s",
         args.pg, args.batch_rows, args.dry_run,
         progress_run_id, progress_phase, progress_url)

sleep_s = float(os.environ.get("FAKE_COMPACT_SLEEP", "0"))
if sleep_s:
    time.sleep(sleep_s)

rc = int(os.environ.get("FAKE_COMPACT_RC", "0"))
if rc != 0:
    print("fake compact: simulated failure", file=sys.stderr)
    sys.exit(rc)

summary = {
    "ok": True,
    "dry_run": args.dry_run,
    "rows_merged": 42,
    "batch_rows": args.batch_rows,
    "pg": args.pg,
    "progress_run_id": progress_run_id,
    "progress_phase": progress_phase,
    "progress_url": progress_url,
}
print(json.dumps(summary, indent=2))
sys.exit(0)
'''


@pytest.fixture()
def fake_compact_script(tmp_path) -> Path:
    path = tmp_path / "fake_compact.py"
    path.write_text(FAKE_SCRIPT_SOURCE)
    return path


@pytest.fixture()
def manager(tmp_path, fake_compact_script) -> compactd.JobManager:
    return compactd.JobManager(
        compact_script=fake_compact_script,
        pg_dsn="postgresql://fake-dsn/ignored",
        batch_rows=123,
        log_dir=tmp_path / "compactd-logs",
        python_exe=sys.executable,
    )


def _start_server(manager: compactd.JobManager):
    """Shared server-lifecycle bit behind both ``running_server`` and
    ``short_timeout_running_server`` — the only thing that differs between
    them is the ``JobManager`` passed in."""
    httpd = compactd.build_server(manager, TOKEN, host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield base_url
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture()
def running_server(manager):
    yield from _start_server(manager)


@pytest.fixture()
def short_timeout_running_server(tmp_path, fake_compact_script):
    """Same as ``running_server`` but with a sub-second
    ``job_timeout_seconds`` — dedicated to the watchdog test, which needs
    the fake subprocess to still be sleeping when the watchdog fires without
    the test itself waiting anywhere near the real 6h default."""
    manager = compactd.JobManager(
        compact_script=fake_compact_script,
        pg_dsn="postgresql://fake-dsn/ignored",
        batch_rows=123,
        log_dir=tmp_path / "compactd-logs-short-timeout",
        python_exe=sys.executable,
        job_timeout_seconds=0.5,
    )
    yield from _start_server(manager)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _request(base_url, method, path, *, token=None, body=None, timeout=5):
    url = base_url + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _poll_until_finished(base_url, job_id, *, timeout=10.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, body = _request(base_url, "GET", f"/jobs/{job_id}", token=TOKEN)
        assert status == 200
        if body["state"] != "running":
            return body
        time.sleep(interval)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

def test_healthz_no_auth(running_server):
    status, body = _request(running_server, "GET", "/healthz")
    assert status == 200
    assert body["ok"] is True
    assert body["running"] is False
    assert isinstance(body["version"], str) and body["version"]


def test_compact_requires_auth(running_server):
    status, body = _request(running_server, "POST", "/compact")
    assert status == 401
    assert body["error"] == "unauthorized"

    status, body = _request(running_server, "POST", "/compact", token="wrong-token")
    assert status == 401
    assert body["error"] == "unauthorized"


def test_jobs_requires_auth(running_server):
    status, body = _request(running_server, "GET", "/jobs/whatever")
    assert status == 401
    assert body["error"] == "unauthorized"

    status, body = _request(running_server, "GET", "/jobs/whatever", token="wrong-token")
    assert status == 401
    assert body["error"] == "unauthorized"


def test_compact_flow_done_with_summary(running_server, monkeypatch):
    monkeypatch.delenv("FAKE_COMPACT_RC", raising=False)
    monkeypatch.delenv("FAKE_COMPACT_SLEEP", raising=False)

    status, body = _request(running_server, "POST", "/compact", token=TOKEN,
                             body={"dry_run": True})
    assert status == 202
    job_id = body["job_id"]
    assert job_id

    final = _poll_until_finished(running_server, job_id)
    assert final["state"] == "done"
    assert final["rc"] == 0
    assert final["reason"] is None
    assert final["finished_at"] is not None
    assert final["summary"] == {
        "ok": True, "dry_run": True, "rows_merged": 42,
        "batch_rows": 123, "pg": "postgresql://fake-dsn/ignored",
        "progress_run_id": None, "progress_phase": None, "progress_url": None,
    }
    assert "fake compact starting" in final["log_tail"]

    # single-flight released once the job finished
    status, health = _request(running_server, "GET", "/healthz")
    assert health["running"] is False


def test_compact_busy_while_job_running(running_server, monkeypatch):
    monkeypatch.setenv("FAKE_COMPACT_SLEEP", "1.0")
    monkeypatch.delenv("FAKE_COMPACT_RC", raising=False)

    status, body = _request(running_server, "POST", "/compact", token=TOKEN)
    assert status == 202
    first_job_id = body["job_id"]

    status, health = _request(running_server, "GET", "/healthz")
    assert health["running"] is True

    status, body = _request(running_server, "POST", "/compact", token=TOKEN)
    assert status == 409
    assert body["error"] == "busy"
    assert body["job_id"] == first_job_id

    final = _poll_until_finished(running_server, first_job_id, timeout=10.0)
    assert final["state"] == "done"


def test_job_not_found(running_server):
    status, body = _request(running_server, "GET", "/jobs/does-not-exist", token=TOKEN)
    assert status == 404
    assert body["error"] == "not_found"


def test_compact_error_state_on_nonzero_rc(running_server, monkeypatch):
    monkeypatch.setenv("FAKE_COMPACT_RC", "1")
    monkeypatch.delenv("FAKE_COMPACT_SLEEP", raising=False)

    status, body = _request(running_server, "POST", "/compact", token=TOKEN)
    assert status == 202
    job_id = body["job_id"]

    final = _poll_until_finished(running_server, job_id)
    assert final["state"] == "error"
    assert final["rc"] == 1
    assert final["reason"] is None  # "reason" is timeout-specific, not general failure
    assert final["summary"] is None
    assert "simulated failure" in final["log_tail"]


def test_job_timeout_kills_subprocess_and_frees_single_flight(
        short_timeout_running_server, monkeypatch):
    # Far longer than short_timeout_running_server's 0.5s job_timeout_seconds
    # — the fake is still sleeping when the watchdog fires.
    monkeypatch.setenv("FAKE_COMPACT_SLEEP", "30")
    monkeypatch.delenv("FAKE_COMPACT_RC", raising=False)

    status, body = _request(short_timeout_running_server, "POST", "/compact", token=TOKEN)
    assert status == 202
    job_id = body["job_id"]

    final = _poll_until_finished(short_timeout_running_server, job_id, timeout=10.0)
    assert final["state"] == "error"
    assert final["reason"] == "timeout"
    assert final["summary"] is None
    assert final["finished_at"] is not None

    # Single-flight is released — a second POST is accepted right away
    # instead of 409ing against the dead job forever.
    status, body2 = _request(short_timeout_running_server, "POST", "/compact", token=TOKEN)
    assert status == 202
    assert body2["job_id"] != job_id

    # Let the second (also-doomed-to-sleep) job's watchdog clean it up too,
    # so the fixture's shutdown doesn't race a still-running subprocess.
    _poll_until_finished(short_timeout_running_server, body2["job_id"], timeout=10.0)


def test_compact_progress_fields_reach_subprocess_env(running_server, monkeypatch):
    monkeypatch.delenv("FAKE_COMPACT_RC", raising=False)
    monkeypatch.delenv("FAKE_COMPACT_SLEEP", raising=False)

    status, body = _request(running_server, "POST", "/compact", token=TOKEN, body={
        "run_id": "run-42", "phase": "FINALIZE",
        "progress_url": "http://example.invalid/progress",
    })
    assert status == 202
    job_id = body["job_id"]

    final = _poll_until_finished(running_server, job_id)
    assert final["state"] == "done"
    assert final["summary"]["progress_run_id"] == "run-42"
    assert final["summary"]["progress_phase"] == "FINALIZE"
    assert final["summary"]["progress_url"] == "http://example.invalid/progress"
    assert "progress_run_id=run-42" in final["log_tail"]
    assert "progress_phase=FINALIZE" in final["log_tail"]
    assert "progress_url=http://example.invalid/progress" in final["log_tail"]


def test_compact_without_progress_fields_leaves_env_unset(running_server, monkeypatch):
    monkeypatch.delenv("FAKE_COMPACT_RC", raising=False)
    monkeypatch.delenv("FAKE_COMPACT_SLEEP", raising=False)

    status, body = _request(running_server, "POST", "/compact", token=TOKEN)
    assert status == 202
    job_id = body["job_id"]

    final = _poll_until_finished(running_server, job_id)
    assert final["summary"]["progress_run_id"] is None
    assert final["summary"]["progress_phase"] is None
    assert final["summary"]["progress_url"] is None


def test_extract_last_json_ignores_leading_noise():
    text = (
        "2026-09-09 INFO some log line\n"
        "another log line with a { brace in it\n"
        '{\n  "ok": true,\n  "n": 1\n}\n'
    )
    assert compactd._extract_last_json(text) == {"ok": True, "n": 1}


def test_extract_last_json_no_json_returns_none():
    assert compactd._extract_last_json("just some log noise, no JSON here") is None


def test_progress_env_from_payload_maps_present_string_fields():
    env = compactd._progress_env_from_payload({
        "run_id": "run-1", "phase": "FINALIZE",
        "progress_url": "http://x", "dry_run": True, "other": 1,
    })
    assert env == {
        "EXONIT_PROGRESS_RUN_ID": "run-1",
        "EXONIT_PROGRESS_PHASE": "FINALIZE",
        "EXONIT_PROGRESS_URL": "http://x",
    }


def test_progress_env_from_payload_ignores_missing_and_non_string():
    assert compactd._progress_env_from_payload({}) == {}
    assert compactd._progress_env_from_payload(
        {"run_id": 123, "phase": "", "progress_url": None}) == {}
