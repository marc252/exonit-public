"""Kill-and-resume integration gate (Fase 3, Task 6 — durable orchestration).

THE GATE for the whole durable-orchestration phase: a REAL subprocess running
scripts/discovery_orchestrator.py is SIGKILLed at each of the 5 parametrized
points in HOME_SEQUENCE, then relaunched with ``--resume <run_id>``, and must
reach DONE with the run_id unchanged and every already-checkpointed step
(postprocess/lora) NOT re-executed.

Why a real subprocess (unlike every other orchestrator test in this repo,
which drives ``orch.run(args)`` in-process with ``unittest.mock.patch``): a
SIGKILL can only be observed truthfully against a real OS process — an
in-process mock can't simulate "the interpreter is gone mid-statement, no
except/finally ran". That is exactly the failure mode Fase 3 exists to
survive (the pre-Task-1 problem: OOM/host-reboot/manual kill mid-run lost
everything because state lived only in memory + a JSON file the crashed
process never got to write).

Harness design
---------------
1. Stub HTTP server (stdlib ``http.server`` in a background thread, real
   TCP on 127.0.0.1) answers the two endpoints the orchestrator's steps
   actually poll/call: ``GET /api/health/work`` (SCANNING / FPP_BACKFILL /
   EXOMINER_BACKFILL drain checks) and ``POST /api/admin/mark_unconfirmed_ffi``
   (fired once on entering EXOMINER_BACKFILL). A mutable ``_StubState``
   (thread-locked dict) lets the SAME test toggle "drained" per-phase between
   the killed launch and the resumed launch — no restart needed.
   ``GET /mark?step=X`` additionally lets stubbed *subprocess* scripts (below)
   record a hit, giving a step-execution counter without touching the
   orchestrator's own code.

2. Fake ROOT: ``scripts/discovery_orchestrator.py`` resolves its own
   ``ROOT = Path(__file__).resolve().parent.parent`` at import time, and
   STOP_WORKERS/STOP_NVIDIARACK (``farmctl``) and POSTPROCESS
   (``discovery_postprocess.py`` + ``scorecard.py``) are invoked at
   ROOT-relative paths. So the harness copies the REAL orchestrator file
   VERBATIM (``shutil.copy2`` — not modified one byte) into a throwaway
   ``<tmp>/fakeroot/scripts/discovery_orchestrator.py``, and plants fast stub
   scripts at the sibling paths the real code already looks for
   (``<tmp>/fakeroot/bin/farmctl``, ``.../scripts/discovery_postprocess.py``,
   ``.../scripts/scorecard.py``). This is a pure invocation-location trick —
   the code under test is byte-identical to the real file — NOT a
   modification of orchestrator logic.

3. LORA is the one step whose command is NOT ROOT-relative (it defaults to
   scripts/lora_macs_driver.py, the real 5 Mac mini M4 fleet driver — Fase 3,
   2026-09-26 — or, via the ``<host>`` override, an ssh+gpuq call to the
   real <host> host), so the ROOT trick above can't reach it, and it must
   never be allowed to actually fire (no network beyond localhost; it's also
   real production hardware). The harness pins it explicitly via
   ``env["EXONIT_LORA_CMD"]`` (which step_lora always resolves ahead of its
   own built-in default/override-file lookup — see
   ``discovery_orchestrator._resolve_lora_cmd``) to a local stub script,
   independent of whatever the orchestrator's own default happens to be. A
   PATH-shadowed fake ``ssh`` in ``<tmp>/fakebin/`` is kept as an additional,
   now-redundant safety net (see ``_STUB_LORA``'s docstring for the incident
   this replaced). Zero orchestrator code touched.

4. Determinism, not sleeps-and-hope:
   - SCANNING / FPP_BACKFILL / EXOMINER_BACKFILL: the stub server simply never
     reports "drained" for whichever phase is the test's kill point, so the
     orchestrator sits in a genuine (bounded, ``--poll-interval 1``) polling
     loop until killed — no race, it structurally cannot advance on its own.
   - POSTPROCESS / LORA: the stub scripts read ``FAKE_POSTPROCESS_HANG`` /
     ``FAKE_LORA_HANG`` env vars and, if set, sleep several seconds BEFORE
     recording their execution-count hit — so a step being tested as the
     kill point cannot possibly complete (and therefore cannot get
     checkpointed) before the test observes the state transition and sends
     SIGKILL, which happens near-instantly after the state write (state is
     persisted to the DB row BEFORE the corresponding step function is even
     called — see ``run()``'s loop).
   - ``_wait_for_state`` polls the ``discovery_runs`` DB row (the durable
     source of truth per Fase 3 Task 1/2) on a tight interval with a
     generous timeout — no fixed sleeps.

5. Hermetic: every DB/state-file path is under ``tmp_path``; the stub HTTP
   server only binds 127.0.0.1; the fake ``ssh``/``farmctl`` never touch the
   network or real hardware; ``EXONIT_DATABASE_URL`` is stripped from the
   child env so no ambient shell config can redirect this at production
   Postgres. Every spawned process is killed in a ``finally`` block via
   ``os.killpg`` (the orchestrator is launched with ``start_new_session=True``
   so it is its own process-group leader — this reaches the subprocess
   children IT spawns too, e.g. the hung ssh/postprocess shim, avoiding
   leaked orphans that a bare ``kill -9 <pid>`` would leave behind).

Two independent, race-free signals prove the checkpoint-skip contract
(deliberately NOT one fuzzy "execution count" that has to guess whether a
doomed, about-to-be-killed attempt got far enough to tick a counter — that
would be exactly the sleeps-and-hope non-determinism this harness is meant
to avoid):

  1. Right after the SIGKILL (before resuming), the DB row's own
     ``checkpoints`` dict is inspected directly: the step being killed AT
     must NOT be checkpointed done (it was genuinely interrupted, not
     skipped), while any EARLIER checkpointed step (e.g. postprocess, when
     killed at LORA) must already show done — proving persistence order.
  2. After resume reaches DONE, the stub scripts' completion counters
     (recorded AFTER their optional hang, not before — see their docstrings)
     must show postprocess_script == lora_ssh == 1 in EVERY case: neither
     step is ever allowed to run to completion twice, which is exactly the
     regression checkpoint-skip exists to prevent. This holds uniformly
     whether the step completed in the killed run (then must be skipped on
     resume) or was the one interrupted (then it completes exactly once, on
     resume, having never completed in the killed attempt).
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_ORCHESTRATOR = REPO_ROOT / "scripts" / "discovery_orchestrator.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from exonit.db import ExoNitDB  # noqa: E402

KILL_POINTS = ["SCANNING", "FPP_BACKFILL", "EXOMINER_BACKFILL", "POSTPROCESS", "LORA"]

# Generous but bounded: only matters if the test fails to kill in time (the
# happy path kills within milliseconds of observing the state, long before
# this elapses) or if a run gets orphaned despite the killpg cleanup.
FAKE_HANG_SECONDS = "6"


# ---------------------------------------------------------------------------
# Stub HTTP server — answers the orchestrator's poll/admin endpoints and
# records step-execution hits reported by the stub subprocess scripts below.
# ---------------------------------------------------------------------------

class _StubState:
    def __init__(self):
        self.lock = threading.Lock()
        # "drained" == True means the corresponding poll metric reports 0
        # (the phase is done); False means it reports a positive count
        # forever (the phase never completes on its own).
        self.drained = {"primary": True, "fpp": True, "exominer": True}
        self.hits = {}

    def set_drained(self, key, value):
        with self.lock:
            self.drained[key] = value

    def set_all_drained(self):
        with self.lock:
            for k in self.drained:
                self.drained[k] = True

    def mark(self, step):
        with self.lock:
            self.hits[step] = self.hits.get(step, 0) + 1

    def hit_count(self, step):
        with self.lock:
            return self.hits.get(step, 0)


def _make_handler(state: "_StubState"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence — keep test output clean
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/health/work":
                with state.lock:
                    d = dict(state.drained)
                self._json({
                    "primary_tasks": {"pending": 0 if d["primary"] else 3,
                                      "assigned": 0},
                    "fpp_backfill": {"actionable": 0 if d["fpp"] else 2},
                    "exominer_backfill": {"pending": 0 if d["exominer"] else 4,
                                          "currently_assigned": 0},
                })
            elif parsed.path == "/mark":
                qs = urllib.parse.parse_qs(parsed.query)
                state.mark(qs.get("step", ["?"])[0])
                self._json({"ok": True})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length:
                self.rfile.read(length)
            # /api/admin/mark_unconfirmed_ffi — content doesn't matter, just
            # must return valid JSON so the orchestrator's log line is happy.
            self._json({"updated": 0})

    return Handler


@pytest.fixture
def stub_server():
    state = _StubState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    state.port = server.server_address[1]
    state.url = f"http://127.0.0.1:{state.port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Fake ROOT: a byte-identical copy of the real orchestrator + fast stub
# scripts at every ROOT-relative path it invokes (farmctl, postprocess,
# scorecard). LORA's ssh is handled separately (PATH shadow, not ROOT-relative).
# ---------------------------------------------------------------------------

_STUB_POSTPROCESS = '''#!/usr/bin/env python3
"""Fake discovery_postprocess.py — Task 6 kill-and-resume harness stub.

The execution-count mark is recorded AFTER the (optional) hang, i.e. only
for an invocation that actually ran to completion. This is deliberate, not
an oversight: marking on START instead would race the test's own SIGKILL —
whether the hung, doomed-to-be-killed attempt gets far enough to tick the
counter before the process group dies depends on OS scheduling, not on
anything the orchestrator does. Counting COMPLETIONS is what is actually
race-free and what the checkpoint-skip contract cares about anyway: "does a
step that already finished and got checkpointed run to completion AGAIN on
resume". Whether the interrupted attempt itself got far enough to start is
checked separately in the test via the DB checkpoint's absence right after
the kill, not via this counter.
"""
import argparse, os, sys, time, urllib.request

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    args, _ = p.parse_known_args()

    if os.environ.get("FAKE_POSTPROCESS_HANG") == "1":
        time.sleep(float(os.environ.get("FAKE_HANG_SECONDS", "6")))

    port = os.environ.get("FAKE_STUB_PORT")
    if port:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/mark?step=postprocess_script",
                timeout=5)
        except Exception:
            pass

    if args.out:
        with open(args.out, "w") as f:
            f.write("# stub dossier (test_kill_and_resume.py)\\n")
    print("STUB_POSTPROCESS_OK")
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''

_STUB_SCORECARD = '''#!/usr/bin/env python3
"""Fake scorecard.py — Task 6 kill-and-resume harness stub."""
import argparse, json, os, sys, urllib.request

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    args, _ = p.parse_known_args()

    port = os.environ.get("FAKE_STUB_PORT")
    if port:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/mark?step=scorecard_script",
                timeout=5)
        except Exception:
            pass

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"stub": True}, f)
    print("STUB_SCORECARD_OK")
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''

_STUB_FARMCTL = '''#!/usr/bin/env python3
"""Fake bin/farmctl — Task 6 kill-and-resume harness stub. Always succeeds."""
import sys
sys.exit(0)
'''

_STUB_LORA = '''#!/usr/bin/env python3
"""Fake LoRA command — Task 6 kill-and-resume harness stub.

2026-09-26 incident fix: this used to be planted ONLY as a PATH-shadowed
`ssh` shim, on the assumption that step_lora's command was always
LORA_SSH_CMD (an `ssh ...` call to <host>). Fase 3 (2026-09-26) changed
the orchestrator's DEFAULT LORA_CMD to scripts/lora_macs_driver.py, invoked
directly (no `ssh` in its argv) — so the PATH shadow silently stopped
intercepting anything, and running this suite for real ran the REAL driver:
it stopped the production Swift fleet via `farmctl stop` and generated 1,650
real LoRA explanations against prod Postgres. This stub is now wired in
directly via `EXONIT_LORA_CMD` (see _base_env below), which step_lora always
resolves and runs verbatim — independent of whatever the orchestrator's own
default happens to be. The `ssh` PATH shadow below is kept as an additional,
now-redundant safety net in case any code path ever shells out to a literal
`ssh` again.

All args are ignored.

The mark is recorded AFTER the (optional) hang — see the discovery_postprocess
stub's docstring for why: it makes the completion counter race-free against
the test's SIGKILL, which always lands well before the hang elapses.
"""
import os, sys, time, urllib.request

def main():
    if os.environ.get("FAKE_LORA_HANG") == "1":
        time.sleep(float(os.environ.get("FAKE_HANG_SECONDS", "6")))

    port = os.environ.get("FAKE_STUB_PORT")
    if port:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/mark?step=lora_ssh", timeout=5)
        except Exception:
            pass

    print("STUB_LORA_OK")
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''


@pytest.fixture
def fake_root(tmp_path):
    root = tmp_path / "fakeroot"
    scripts_dir = root / "scripts"
    bin_dir = root / "bin"
    fakebin_dir = tmp_path / "fakebin"
    scripts_dir.mkdir(parents=True)
    bin_dir.mkdir(parents=True)
    fakebin_dir.mkdir(parents=True)

    # The code under test: byte-identical copy, not a rewrite.
    shutil.copy2(REAL_ORCHESTRATOR, scripts_dir / "discovery_orchestrator.py")

    (scripts_dir / "discovery_postprocess.py").write_text(_STUB_POSTPROCESS)
    (scripts_dir / "scorecard.py").write_text(_STUB_SCORECARD)

    farmctl_path = bin_dir / "farmctl"
    farmctl_path.write_text(_STUB_FARMCTL)
    farmctl_path.chmod(0o755)

    # Kept as an additional (now-redundant) safety net — see _STUB_LORA's
    # docstring — in case anything ever shells out to a literal `ssh` again.
    ssh_path = fakebin_dir / "ssh"
    ssh_path.write_text(_STUB_LORA)
    ssh_path.chmod(0o755)

    # The actual interception point (2026-09-26 fix): a directly-invocable
    # stub, wired in via EXONIT_LORA_CMD in _base_env below — independent of
    # whatever command shape the orchestrator's own LORA_CMD default takes.
    lora_stub_path = fakebin_dir / "lora_stub.py"
    lora_stub_path.write_text(_STUB_LORA)
    lora_stub_path.chmod(0o755)

    return root, fakebin_dir


# ---------------------------------------------------------------------------
# Subprocess launch / kill / poll helpers
# ---------------------------------------------------------------------------

def _base_env(fakebin_dir, stub_port):
    env = dict(os.environ)
    # Hermetic: never let an ambient shell's real .env/exports redirect this
    # at production Postgres — --db always wins in the orchestrator anyway,
    # but strip it so nothing else in the child process tree could pick it up.
    env.pop("EXONIT_DATABASE_URL", None)
    env["EXONIT_DEPLOYMENT"] = "home"
    env["PATH"] = f"{fakebin_dir}{os.pathsep}{env.get('PATH', '')}"
    # 2026-09-26 incident fix: pin the LORA command to our local stub
    # explicitly rather than relying on the orchestrator's own default (see
    # _STUB_LORA's docstring) — this is EXONIT_LORA_CMD, which step_lora
    # always resolves ahead of its built-in default/override-file lookup.
    env["EXONIT_LORA_CMD"] = f"{sys.executable} {fakebin_dir / 'lora_stub.py'}"
    env["FAKE_STUB_PORT"] = str(stub_port)
    env["FAKE_HANG_SECONDS"] = FAKE_HANG_SECONDS
    # The orchestrator script is executed from a copy under a throwaway
    # fakeroot (see fake_root fixture) so its OWN sys.path[0] (the directory
    # containing the executed .py file) is that fakeroot's scripts/ dir, not
    # this worktree — Python sets sys.path[0] from the script's location, not
    # cwd, for a plain `python3 script.py` invocation. Without this, `import
    # exonit` falls through to whatever the machine's pip-editable install
    # points at, which is pinned to the MAIN checkout (~/exonit),
    # NOT this git worktree — silently running a stale exonit.db (missing
    # the very discovery_runs methods Fase 3 added). Prepending the real
    # worktree root here is what actually makes this a test of the code on
    # THIS branch.
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (str(REPO_ROOT) + (os.pathsep + existing_pp if existing_pp else ""))
    return env


def _fresh_argv(script_path, *, run_id, lot_csv, sector, server_url,
                state_file, db_path):
    return [
        sys.executable, str(script_path),
        "--lot-csv", str(lot_csv),
        "--sector", str(sector),
        "--run-id", run_id,
        "--server-url", server_url,
        "--state-file", state_file,
        "--db", db_path,
        "--poll-interval", "1",
    ]


def _resume_argv(script_path, *, run_id, server_url, state_file, db_path,
                 force=False):
    argv = [
        sys.executable, str(script_path),
        "--resume", run_id,
        "--server-url", server_url,
        "--state-file", state_file,
        "--db", db_path,
        "--poll-interval", "1",
    ]
    if force:
        argv.append("--force")
    return argv


def _kill_and_reap(proc, timeout=10):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass


def _pid_is_dead(pid, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.05)
    return False


def _wait_for_state(db_path, run_id, target, timeout=20, poll=0.05):
    """Poll the discovery_runs DB row (the durable source of truth) until
    its state column equals *target*. No fixed sleeps — pure condition wait."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        db = ExoNitDB(db_path)
        try:
            row = db.get_discovery_run(run_id)
        finally:
            db.close()
        if row is not None:
            last = row["state"]
            if last == target:
                return row
        time.sleep(poll)
    raise AssertionError(
        f"timed out waiting for discovery_runs(run_id={run_id!r}).state == "
        f"{target!r}; last observed state={last!r}")


def _read_row(db_path, run_id):
    db = ExoNitDB(db_path)
    try:
        return db.get_discovery_run(run_id)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# THE GATE: SIGKILL at each HOME_SEQUENCE point, then --resume to DONE.
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("kill_at_state", KILL_POINTS)
def test_sigkill_then_resume_reaches_done(kill_at_state, tmp_path, stub_server,
                                          fake_root):
    root, fakebin_dir = fake_root
    script_path = root / "scripts" / "discovery_orchestrator.py"
    db_path = str(tmp_path / "exonit.db")
    sf = str(tmp_path / "pipeline_state.json")
    lot_csv = tmp_path / "s0099_lot20260705T000000.csv"
    lot_csv.write_text("tic_id,label\n424242,candidate\n")
    run_id = f"discovery_s0099_lot20260705T000000_2026-07-05_{kill_at_state.lower()}"

    db = ExoNitDB(db_path)
    db.create_discovery_run(run_id, sector=99, lot_csv=str(lot_csv),
                            dry_run=False, chained=False,
                            server_url=stub_server.url)
    db.close()

    # Only the phase being killed at is left un-drained; everything strictly
    # before it must drain immediately so the killed run reaches the target
    # state quickly instead of getting stuck at an earlier one.
    stub_server.set_drained("primary", kill_at_state != "SCANNING")
    stub_server.set_drained("fpp", kill_at_state != "FPP_BACKFILL")
    stub_server.set_drained("exominer", kill_at_state != "EXOMINER_BACKFILL")

    env = _base_env(fakebin_dir, stub_server.port)
    if kill_at_state == "EXOMINER_BACKFILL":
        # Without this the step short-circuits "skipped" and never actually
        # polls — there would be nothing to get stuck at.
        env["EXONIT_EXOMINER_BACKFILL"] = "1"
    if kill_at_state == "POSTPROCESS":
        env["FAKE_POSTPROCESS_HANG"] = "1"
    if kill_at_state == "LORA":
        env["FAKE_LORA_HANG"] = "1"

    argv = _fresh_argv(script_path, run_id=run_id, lot_csv=lot_csv, sector=99,
                       server_url=stub_server.url, state_file=sf, db_path=db_path)

    log1 = open(tmp_path / "orch_kill.log", "wb")
    log2 = open(tmp_path / "orch_resume.log", "wb")
    proc = None
    try:
        proc = subprocess.Popen(argv, cwd=str(root), env=env,
                                stdout=log1, stderr=subprocess.STDOUT,
                                start_new_session=True)

        # Mimic what the real spawn endpoint (exonit/routes/discovery.py)
        # does right after Popen — we bypass the server entirely here, so we
        # must record the pid ourselves for our own dead-pid assertion below.
        db = ExoNitDB(db_path)
        db.update_discovery_run(run_id, pid=proc.pid)
        db.close()

        row = _wait_for_state(db_path, run_id, kill_at_state, timeout=20)
        assert row["orchestrator_pid"] == proc.pid

        _kill_and_reap(proc)
        assert _pid_is_dead(proc.pid), "orchestrator pid must be dead after SIGKILL"

        row = _read_row(db_path, run_id)
        assert row["state"] not in ("DONE", "FAILED"), (
            f"expected a non-terminal state after SIGKILL (the process died "
            f"mid-step, never reaching FAILED), got {row['state']!r}")
        assert row["state"] == kill_at_state
        assert row["orchestrator_pid"] == proc.pid

        # Checkpoint evidence AT THE MOMENT of the kill (before any resume):
        # the step being killed at was genuinely interrupted (never
        # checkpointed done), while a step that legitimately finished BEFORE
        # it (postprocess, when killed at LORA) must already be persisted.
        cps_at_kill = row["checkpoints"]
        if kill_at_state == "POSTPROCESS":
            assert not cps_at_kill.get("postprocess", {}).get("done"), (
                "postprocess must NOT be checkpointed done — it was killed "
                "mid-step, not completed")
        elif kill_at_state == "LORA":
            assert cps_at_kill.get("postprocess", {}).get("done") is True, (
                "postprocess must already be checkpointed done by the time "
                "LORA starts (it runs earlier in HOME_SEQUENCE)")
            assert not cps_at_kill.get("lora", {}).get("done"), (
                "lora must NOT be checkpointed done — it was killed "
                "mid-step, not completed")

        # --- RESUME: everything now drains immediately, no hangs. ---
        stub_server.set_all_drained()
        resume_env = dict(env)
        resume_env.pop("FAKE_POSTPROCESS_HANG", None)
        resume_env.pop("FAKE_LORA_HANG", None)

        resume_argv = _resume_argv(script_path, run_id=run_id,
                                   server_url=stub_server.url,
                                   state_file=sf, db_path=db_path)
        proc = subprocess.Popen(resume_argv, cwd=str(root), env=resume_env,
                                stdout=log2, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            _kill_and_reap(proc)
            pytest.fail(
                f"resumed orchestrator did not exit within 45s "
                f"(kill_at_state={kill_at_state}); see {tmp_path}/orch_resume.log")

        assert rc == 0, (
            f"resumed orchestrator exited with code {rc} "
            f"(kill_at_state={kill_at_state}); see {tmp_path}/orch_resume.log")

        final_row = _read_row(db_path, run_id)
        assert final_row["state"] == "DONE"
        # Identity: run_id is never re-derived across the kill/resume cycle.
        # (Checking final_row["run_id"] == run_id would be vacuous — we
        # looked the row up BY run_id — so instead assert there is still
        # exactly ONE discovery_runs row in the whole DB: a re-derivation
        # bug, e.g. the Task-2 midnight-crossing bug this schema exists to
        # prevent, would show up as a SECOND row under a different run_id
        # rather than as a changed value in the row we already have.)
        db = ExoNitDB(db_path)
        try:
            count_row = db._query_one("SELECT COUNT(*) AS n FROM discovery_runs")
        finally:
            db.close()
        assert count_row["n"] == 1, (
            f"expected exactly 1 discovery_runs row, found {count_row['n']} — "
            f"a re-derived run_id would show up as an extra row")

        cps = final_row["checkpoints"]
        assert cps.get("postprocess", {}).get("done") is True
        assert cps.get("lora", {}).get("done") is True
        assert cps.get("finalize", {}).get("done") is True
        assert not cps.get("_failed_state"), (
            "run never reached FAILED (SIGKILL, not a graceful failure) — "
            "_failed_state must never be set for this path"
        )

        # THE checkpoint-skip invariant: regardless of WHERE the kill
        # happened, neither checkpointed step is ever allowed to run to
        # completion twice. (The step genuinely interrupted by the kill, if
        # any, completes exactly once — on resume, having never completed
        # in the killed attempt — see the checkpoint-at-kill-time assertions
        # above for proof that it really was interrupted, not skipped.)
        assert stub_server.hit_count("postprocess_script") == 1, (
            f"postprocess_script completed {stub_server.hit_count('postprocess_script')} "
            f"times, expected exactly 1 for kill_at_state={kill_at_state}")
        assert stub_server.hit_count("lora_ssh") == 1, (
            f"lora_ssh completed {stub_server.hit_count('lora_ssh')} times, "
            f"expected exactly 1 for kill_at_state={kill_at_state}")
    finally:
        if proc is not None:
            _kill_and_reap(proc)
        log1.close()
        log2.close()


# ---------------------------------------------------------------------------
# Watchdog path (in-process, per the brief): a run with a dead pid + stale
# heartbeat is marked FAILED by app.state.discovery_watchdog_check_once, then
# --resume --force completes it. Distinct from the SIGKILL tests above (which
# resume a run that died mid-step WITHOUT ever reaching FAILED) — this is the
# OTHER dead-run path, where nobody killed the orchestrator "just now" but a
# stale heartbeat proves it's gone, and the watchdog itself flips the state.
# ---------------------------------------------------------------------------

def test_watchdog_marks_failed_then_forced_resume_completes(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    monkeypatch.setenv("EXONIT_DB_PATH", str(db_file))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(reports_dir))
    monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE",
                       str(tmp_path / "pipeline_state.json"))

    from exonit.server import create_app
    app = create_app()

    run_id = "discovery_s0001_lotWD_2026-07-05"
    lot_csv = tmp_path / "lotWD.csv"
    lot_csv.write_text("tic_id,label\n555,candidate\n")

    db = ExoNitDB(str(db_file))
    db.create_discovery_run(run_id, sector=1, lot_csv=str(lot_csv),
                            dry_run=False, chained=False,
                            server_url="http://localhost:8400")
    db.update_discovery_run(run_id, state="SCANNING", pid=2**30 - 1)  # bogus/dead pid
    stale_iso = (datetime.now(timezone.utc) - timedelta(seconds=999)).isoformat()
    db._exec(
        "UPDATE discovery_runs SET heartbeat_at = ?, updated_at = ? WHERE run_id = ?",
        (stale_iso, stale_iso, run_id))
    db.close()

    with patch("os.kill", side_effect=ProcessLookupError):
        app.state.discovery_watchdog_check_once()

    row = _read_row(str(db_file), run_id)
    assert row["state"] == "FAILED"
    assert "orchestrator_died" in row["detail"]
    # F6 (final-review fix): the watchdog's own FAILED write must stamp
    # finished_at too, same as every other terminal transition.
    assert row["finished_at"] is not None

    import discovery_orchestrator as orch
    orch._db_path = None
    orch._pg_dual_write_enabled = False
    orch._db_instance = None
    try:
        sf = str(tmp_path / "pipeline_state_resume.json")
        args = orch._parse_args([
            "--resume", run_id, "--force",
            "--state-file", sf,
            "--db", str(db_file),
            "--poll-interval", "0",
        ])
        with patch.object(orch, "step_scanning", return_value=(True, "")) as mock_scan, \
             patch.object(orch, "step_fpp_backfill", return_value=(True, "")), \
             patch.object(orch, "step_exominer_backfill", return_value=(True, "")), \
             patch.object(orch, "step_stop_workers", return_value=(True, "")), \
             patch.object(orch, "step_postprocess", return_value=(True, "")), \
             patch.object(orch, "step_stop_nvidiarack", return_value=(True, "")), \
             patch.object(orch, "step_lora", return_value=(True, "")), \
             patch.object(orch, "step_finalize", return_value=(True, "")), \
             patch.object(orch, "step_wal_checkpoint", return_value=None), \
             patch("urllib.request.urlopen", side_effect=Exception("no server")):
            orch.run(args)
        # No _failed_state checkpoint was ever written by the watchdog (it
        # only flips the state column) — run() must fall back to SCANNING
        # rather than refuse to resume (see run()'s own warning-logged path).
        mock_scan.assert_called_once()
    finally:
        orch._db_path = None
        orch._pg_dual_write_enabled = False
        orch._db_instance = None

    final = orch._load_state(sf)
    assert final["state"] == "DONE"

    row = _read_row(str(db_file), run_id)
    assert row["state"] == "DONE"
    assert row["run_id"] == run_id
    assert row["finished_at"] is not None
