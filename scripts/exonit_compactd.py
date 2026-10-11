#!/usr/bin/env python3
"""``exonit-compactd`` — HTTP sidecar at <host> that turns a completeness
compaction into a short POST + polling instead of a long-lived SSH session.

Why this exists (2026-09-09 incident, see
``docs/superpowers/specs/2026-09-09-compactd-service-design.md`` and memory
``gotcha_compact_remote_ssh_dies_silently_staging_snowball``): the previous
remote-compaction path ran ``compact_completeness.py`` over an SSH session
kept open for the whole compaction (up to ~45 min); a dropped session without
keepalive silently orphaned the process and let ``completeness_staging``
balloon unbounded. This service instead ALWAYS runs on the box that owns PG
and MinIO, supervises the subprocess with its own thread, and exposes its
state over a handful of short-lived HTTP requests — the orchestrator client
(``discovery_orchestrator.py``, Component 3 of the same spec) never holds a
connection open longer than one poll.

stdlib only (``http.server.ThreadingHTTPServer`` + ``json`` + ``threading`` +
``subprocess``) — <host>'s venv has no FastAPI/uvicorn and this is not
worth adding a dependency for.

Single-flight by construction: ``JobManager`` allows exactly one subprocess
in flight; the advisory PG lock inside ``compact_completeness.py`` itself is
the second, cross-process line of defence (this service could be restarted
mid-job, or a stray SSH-path invocation could race it).
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from exonit import __version__

log = logging.getLogger("exonit_compactd")

DEFAULT_PORT = 8646
DEFAULT_PG_DSN = "postgresql://exonit@127.0.0.1:5432/exonit"
DEFAULT_LOG_DIR = Path.home() / "exonit-compactd"
LOG_TAIL_CHARS = 2000
# 6h — sized for large backlog drains (batch mode's whole point is bounding
# MEMORY, not wall-clock; a genuinely huge completeness_staging can
# legitimately take hours). Past this, the job is presumed hung rather than
# slow (review 2026-09-09 finding 1: an unbounded wait here wedges
# single-flight open-ended on any hang — a stalled MinIO/PG connection that
# with_s3_retries's own backoff doesn't cover, for instance).
DEFAULT_JOB_TIMEOUT_SECONDS = 21600
# Grace period given to the killed process group to actually exit and close
# its stdout/stderr pipes (so the reader threads below can join) before this
# gives up on reaping it and reports rc=null instead of hanging the watchdog
# itself.
_KILL_REAP_GRACE_SECONDS = 10
_PUMP_JOIN_GRACE_SECONDS = 30

PORT_ENV = "EXONIT_COMPACTD_PORT"
TOKEN_ENV = "EXONIT_COMPACTD_TOKEN"
PG_DSN_ENV = "EXONIT_COMPACTD_PG_DSN"
BATCH_ROWS_ENV = "EXONIT_COMPACT_BATCH_ROWS"
JOB_TIMEOUT_ENV = "EXONIT_COMPACTD_JOB_TIMEOUT_SECONDS"

# POST /compact body fields -> subprocess env var names (parity with the
# SSH path, which already forwards these three — _compact_remote_cmd in
# discovery_orchestrator.py; review 2026-09-09 finding 2).
_PROGRESS_FIELD_TO_ENV = {
    "run_id": "EXONIT_PROGRESS_RUN_ID",
    "phase": "EXONIT_PROGRESS_PHASE",
    "progress_url": "EXONIT_PROGRESS_URL",
}

# Deployed alongside compact_completeness.py in the same repo checkout
# (the service runs the whole repo checkout) — resolved relative
# to THIS file, not the process cwd, so the service can be launched from
# anywhere (systemd sets WorkingDirectory but this must not depend on it).
_COMPACT_SCRIPT_DEFAULT = Path(__file__).resolve().parent / "compact_completeness.py"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_job_id() -> str:
    """``<unix-ts>-<8 hex chars>`` — monotonic-ish (sortable, human-legible
    in logs/URLs) plus enough entropy that two jobs started in the same
    second (should be impossible under single-flight, but defence in depth)
    never collide."""
    return f"{int(time.time())}-{uuid.uuid4().hex[:8]}"


def _extract_last_json(text: str) -> Optional[dict]:
    """The LAST complete JSON object in *text*, or ``None``.

    ``compact_completeness.py --pg ... `` prints exactly one pretty-printed
    JSON summary (``json.dumps(result, indent=2, ...)``) as the final thing
    on stdout — everything else (its own ``logging.basicConfig`` output)
    goes to stderr. Scanning backward for a ``{`` that both parses AND
    consumes the string to its end (ignoring trailing whitespace) finds
    that outermost object even if stray text preceded it, and specifically
    will NOT stop early on an inner ``{`` that happens to also be valid JSON
    on its own (raw_decode from an inner brace leaves trailing content
    after it — the closing brace(s) of the enclosing object — so the
    "consumes to end" check rejects it and the scan keeps moving left).
    """
    decoder = json.JSONDecoder()
    idx = text.rfind("{")
    while idx != -1:
        try:
            obj, end = decoder.raw_decode(text, idx)
            if isinstance(obj, dict) and text[end:].strip() == "":
                return obj
        except ValueError:
            pass
        idx = text.rfind("{", 0, idx)
    return None


def _progress_env_from_payload(payload: dict) -> dict:
    """``{"run_id": ..., "phase": ..., "progress_url": ...}`` (any subset,
    non-string/empty values ignored) -> the matching
    ``EXONIT_PROGRESS_*`` env var overrides for the subprocess. Empty dict
    if none of the three fields are present/valid — callers treat that as
    "no progress reporting requested", the previous (silent) behaviour."""
    env: dict = {}
    for field, env_name in _PROGRESS_FIELD_TO_ENV.items():
        value = payload.get(field)
        if isinstance(value, str) and value:
            env[env_name] = value
    return env


class JobManager:
    """Supervises at most one ``compact_completeness.py`` subprocess at a
    time and keeps an in-memory record of every job started since this
    process booted (plus a best-effort on-disk snapshot of the latest one).

    *compact_script*/*python_exe*/*pg_dsn*/*batch_rows*/*log_dir* are all
    constructor parameters (not module globals read at call time) so tests
    can point *compact_script* at a small fake script instead of the real
    one — the cleanest injection seam without env-var indirection inside
    the class itself.
    """

    def __init__(self, *, compact_script: Path = _COMPACT_SCRIPT_DEFAULT,
                 pg_dsn: str = DEFAULT_PG_DSN,
                 batch_rows: Optional[int] = None,
                 log_dir: Path = DEFAULT_LOG_DIR,
                 python_exe: str = sys.executable,
                 job_timeout_seconds: int = DEFAULT_JOB_TIMEOUT_SECONDS):
        self.compact_script = compact_script
        self.pg_dsn = pg_dsn
        self.batch_rows = batch_rows
        self.log_dir = log_dir
        self.python_exe = python_exe
        self.job_timeout_seconds = job_timeout_seconds
        self._lock = threading.Lock()
        self._jobs: dict = {}
        self._active_job_id: Optional[str] = None

    def is_running(self) -> bool:
        with self._lock:
            return self._active_job_id is not None

    def start(self, *, dry_run: bool = False, progress_env: Optional[dict] = None):
        """Starts a new job unless one is already in flight.

        Returns ``(job_id, started)``. ``started=False`` means single-flight
        was already held — *job_id* is that OTHER job's id, exactly the
        value the ``409 {"error": "busy", "job_id": ...}`` response needs.

        *progress_env* (already translated from the request body's
        ``run_id``/``phase``/``progress_url`` fields to
        ``EXONIT_PROGRESS_RUN_ID``/``_PHASE``/``_URL`` by the caller — see
        ``_progress_env_from_payload``) is merged into the subprocess's
        environment so ``compact_completeness.py``'s ``ProgressReporter``
        picks it up, exactly like the SSH path it replaces already does.
        """
        with self._lock:
            if self._active_job_id is not None:
                return self._active_job_id, False
            job_id = _new_job_id()
            self._jobs[job_id] = {
                "state": "running", "rc": None, "reason": None,
                "started_at": _now_iso(), "finished_at": None, "summary": None,
            }
            self._active_job_id = job_id
        self._persist_last_job(job_id)
        threading.Thread(target=self._run, args=(job_id, dry_run, progress_env),
                          daemon=True).start()
        return job_id, True

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def log_tail(self, job_id: str, n: int = LOG_TAIL_CHARS) -> str:
        """Last *n* characters of the job's combined stdout+stderr log —
        read fresh off disk every call so it reflects a still-running job,
        not a snapshot taken at start time. Missing file (job never wrote
        anything yet, or was lost across a service restart) -> ``""``."""
        try:
            data = self._log_path(job_id).read_bytes()
        except OSError:
            return ""
        return data[-n:].decode("utf-8", errors="replace")

    def _log_path(self, job_id: str) -> Path:
        return self.log_dir / f"job_{job_id}.log"

    def _run(self, job_id: str, dry_run: bool,
             progress_env: Optional[dict] = None) -> None:
        """Runs in its own daemon thread (started by ``start()``) — the
        request handler that called ``start()`` has already responded 202
        by the time this does any real work."""
        cmd = [self.python_exe, str(self.compact_script), "--pg", self.pg_dsn]
        if self.batch_rows is not None:
            cmd += ["--batch-rows", str(self.batch_rows)]
        if dry_run:
            cmd.append("--dry-run")
        log.info("job %s: starting: %s", job_id, " ".join(cmd))

        env = os.environ.copy()
        if progress_env:
            env.update(progress_env)

        rc = -1
        timed_out = False
        stdout_chunks: list = []
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with open(self._log_path(job_id), "wb") as logf:
                log_lock = threading.Lock()

                def _pump(stream, keep: Optional[list]) -> None:
                    # stdout and stderr are piped SEPARATELY (not merged via
                    # STDOUT) so the JSON summary parse below only ever sees
                    # real stdout bytes, never a logging line interleaved
                    # mid-object — but both streams still land in the SAME
                    # log file (guarded by log_lock) for a human `tail -f`.
                    for chunk in iter(stream.readline, b""):
                        with log_lock:
                            logf.write(chunk)
                            logf.flush()
                        if keep is not None:
                            keep.append(chunk)
                    stream.close()

                # start_new_session=True makes this subprocess the leader of
                # its OWN process group (distinct from exonit-compactd's) —
                # required so the watchdog below can kill the whole tree
                # (compact_completeness.py plus anything it shells out to)
                # via os.killpg without also taking down the daemon itself.
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env=env, start_new_session=True)
                t_out = threading.Thread(target=_pump, args=(proc.stdout, stdout_chunks))
                t_err = threading.Thread(target=_pump, args=(proc.stderr, None))
                t_out.start()
                t_err.start()

                try:
                    rc = proc.wait(timeout=self.job_timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    log.error("job %s: exceeded EXONIT_COMPACTD_JOB_TIMEOUT_SECONDS "
                              "(%ss); killing its process group",
                              job_id, self.job_timeout_seconds)
                    self._kill_process_group(job_id, proc)
                    try:
                        rc = proc.wait(timeout=_KILL_REAP_GRACE_SECONDS)
                    except subprocess.TimeoutExpired:
                        # Killed but couldn't be reaped (defunct/zombie edge
                        # case) — report rc as unknown rather than hang the
                        # watchdog itself waiting for it, per the spec ("rc
                        # del wait posterior o null").
                        log.error("job %s: process did not exit within %ss of "
                                  "SIGKILL; reporting rc=null", job_id,
                                  _KILL_REAP_GRACE_SECONDS)
                        rc = None

                # The process (or its whole group, if killed) is dead by now
                # in every reachable branch above, so stdout/stderr EOF and
                # these joins return promptly — bounded anyway, a stuck
                # reader thread is daemon and can never wedge the service.
                t_out.join(timeout=_PUMP_JOIN_GRACE_SECONDS)
                t_err.join(timeout=_PUMP_JOIN_GRACE_SECONDS)
        except Exception:
            # A launch failure (bad interpreter path, missing script, ...)
            # must still resolve the job to "error" rather than leaving it
            # "running" forever and single-flight wedged.
            log.exception("job %s: failed to launch/supervise subprocess", job_id)

        if timed_out:
            summary = None
            state = "error"
            reason = "timeout"
        else:
            summary = _extract_last_json(
                b"".join(stdout_chunks).decode("utf-8", errors="replace"))
            state = "done" if (rc == 0 and summary is not None) else "error"
            reason = None
        log.info("job %s: finished rc=%s state=%s reason=%s", job_id, rc, state, reason)

        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.update(state=state, rc=rc, reason=reason,
                           finished_at=_now_iso(), summary=summary)
            if self._active_job_id == job_id:
                self._active_job_id = None
        self._persist_last_job(job_id)

    @staticmethod
    def _kill_process_group(job_id: str, proc: subprocess.Popen) -> None:
        """SIGKILL the whole process group *proc* leads (see the
        ``start_new_session=True`` note above). Best-effort: a group that
        has already exited on its own between the timeout firing and this
        call (``ProcessLookupError``) is not an error — the outcome (job
        dead) is the same either way."""
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            log.info("job %s: process group already gone by the time of kill", job_id)
        except OSError:
            log.warning("job %s: failed to kill process group for pid %s",
                        job_id, proc.pid, exc_info=True)

    def _persist_last_job(self, job_id: str) -> None:
        """Best-effort snapshot of the single most recent job to
        ``<log_dir>/last_job.json`` — an operator convenience (readable
        after a service restart, per the spec's accepted "job lost on
        restart, client falls back" tradeoff) that must NEVER be able to
        take down a job thread or the request handler that called
        ``start()``."""
        try:
            job = self.get(job_id)
            if job is None:
                return
            self.log_dir.mkdir(parents=True, exist_ok=True)
            payload = {"job_id": job_id, **job}
            (self.log_dir / "last_job.json").write_text(
                json.dumps(payload, indent=2, default=str))
        except Exception:
            log.warning("failed to persist last_job.json (non-fatal)", exc_info=True)


def _make_handler(manager: JobManager, token: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # route stdlib access log through logging
            log.info("%s - %s", self.address_string(), fmt % args)

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            """``Authorization: Bearer <token>``, constant-time compare
            (``hmac.compare_digest``) — a timing side-channel on a bearer
            token would defeat the whole point of having one."""
            header = self.headers.get("Authorization", "")
            prefix = "Bearer "
            if not header.startswith(prefix):
                return False
            provided = header[len(prefix):]
            return hmac.compare_digest(provided, token)

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            if path == "/healthz":
                # No auth: a load balancer / systemd probe / curl-from-
                # anywhere health check must not need the token.
                self._json(200, {"ok": True, "running": manager.is_running(),
                                  "version": __version__})
                return
            if not self._authorized():
                self._json(401, {"error": "unauthorized"})
                return
            if path.startswith("/jobs/"):
                job_id = path[len("/jobs/"):]
                job = manager.get(job_id)
                if job is None:
                    self._json(404, {"error": "not_found"})
                    return
                body = dict(job)
                body["log_tail"] = manager.log_tail(job_id)
                self._json(200, body)
                return
            self._json(404, {"error": "not_found"})

        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path
            if path != "/compact":
                self._json(404, {"error": "not_found"})
                return
            if not self._authorized():
                self._json(401, {"error": "unauthorized"})
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            dry_run = False
            progress_env: dict = {}
            if raw.strip():
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    self._json(400, {"error": "invalid_json_body"})
                    return
                if isinstance(payload, dict):
                    dry_run = bool(payload.get("dry_run", False))
                    progress_env = _progress_env_from_payload(payload)
            job_id, started = manager.start(dry_run=dry_run, progress_env=progress_env)
            if not started:
                self._json(409, {"error": "busy", "job_id": job_id})
                return
            self._json(202, {"job_id": job_id})

    return Handler


def build_server(manager: JobManager, token: str, *, host: str = "0.0.0.0",
                  port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    handler_cls = _make_handler(manager, token)
    return ThreadingHTTPServer((host, port), handler_cls)


def _int_env(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("%s=%r is not an int; using default %r", name, raw, default)
        return default


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="ExoNit compaction service (stdlib-only HTTP daemon). "
                    "Runs compact_completeness.py as a supervised subprocess "
                    "on request instead of over a long-lived SSH session."
    )
    parser.add_argument("--port", type=int, default=None,
                         help=f"Bind port (default: env {PORT_ENV} or {DEFAULT_PORT})")
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO,
        format="%(asctime)s %(levelname)s exonit_compactd: %(message)s")

    # No default token: an unauthenticated compaction trigger reachable from
    # the whole LAN is not an acceptable "works out of the box" default —
    # the operator MUST set one (systemd's EnvironmentFile, per the runbook).
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        log.error("%s is not set; refusing to start (see docs/runbooks/compactd.md)",
                   TOKEN_ENV)
        return 1

    port = args.port if args.port is not None else _int_env(PORT_ENV, DEFAULT_PORT)
    pg_dsn = os.environ.get(PG_DSN_ENV, DEFAULT_PG_DSN)
    batch_rows = _int_env(BATCH_ROWS_ENV, None)
    job_timeout_seconds = _int_env(JOB_TIMEOUT_ENV, DEFAULT_JOB_TIMEOUT_SECONDS)

    manager = JobManager(pg_dsn=pg_dsn, batch_rows=batch_rows,
                          job_timeout_seconds=job_timeout_seconds)
    httpd = build_server(manager, token, port=port)
    log.info("listening on 0.0.0.0:%d, pg_dsn=%s, batch_rows=%s, "
              "job_timeout_seconds=%s, compact_script=%s",
              port, pg_dsn, batch_rows, job_timeout_seconds, manager.compact_script)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down (KeyboardInterrupt)")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
