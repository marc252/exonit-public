"""Tests for the Phase-A orchestrator/postprocess fixes from
docs/reviews/2026-07-02-opus48-cleanup-review.md (B3 timeouts, D4 polling
dedup wiring, D2 select_in_chunks delegation).

Constraints (matching tests/test_discovery_orchestrator.py's header):
- NEVER real SSH, farmctl, server poll, or subprocess network/GPU calls.
- NEVER touch the live DB data/exonit.db or production PostgreSQL.
- All external I/O is mocked/monkeypatched.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import discovery_orchestrator as orch  # noqa: E402
import discovery_postprocess as pp  # noqa: E402


@pytest.fixture(autouse=True)
def mock_scale_fleet():
    """Neuter the fleet-autoscale hook for every test in this module.

    Incident 2026-08-01: TestPollUntilDrainedSharedHelper::
    test_all_three_steps_call_the_same_helper calls the REAL
    step_scanning/step_fpp_backfill/step_exominer_backfill, which each call
    `_scale_fleet(phase, dry_run)` — this shells out to
    `bin/farmctl start --all --phase <phase>`, SSHing into every real worker
    host and starting the fleet for real. tests/test_discovery_orchestrator.py
    has an identical module-scoped autouse fixture that does NOT cover this
    file, which is how that pytest run started the real fleet. Copied here
    (not shared/imported) so this file is safe standalone even if the
    tests/conftest.py EXONIT_FLEET_AUTOSCALE=0 global kill-switch is ever
    removed or bypassed — see tests/conftest.py for that layer.
    """
    with patch.object(orch, "_scale_fleet", return_value=(True, "")) as m:
        yield m


@pytest.fixture(autouse=True)
def _reset_orchestrator_db_globals():
    """Isolate module-level PG dual-write state between tests (Fase 3 Task 2).

    See the identical fixture in tests/test_discovery_orchestrator.py for the
    full rationale — this file shares the same cached ``discovery_orchestrator``
    module across the whole pytest session.
    """
    orch._db_path = None
    orch._pg_dual_write_enabled = False
    orch._db_instance = None
    yield
    orch._db_path = None
    orch._pg_dual_write_enabled = False
    orch._db_instance = None


# ---------------------------------------------------------------------------
# B3 — discovery_postprocess._run_fpp_with_timeout
# ---------------------------------------------------------------------------

class TestRunFppWithTimeout:
    def test_fast_call_returns_result_not_timed_out(self):
        """A calc_fpp call that finishes well within the timeout returns
        (result, False)."""
        def fake_calc_fpp(tic_id, period, t0, dur, depth_ppm, secs, t, f,
                          N=100000, workdir="."):
            assert N == 100000
            assert workdir == "/tmp/tri_work"
            return {"fpp": 0.01, "nfpp": 0.001, "verdict": "validated"}

        result, timed_out = pp._run_fpp_with_timeout(
            fake_calc_fpp, "123", 3.0, 1300.0, 0.1, 1000.0,
            [1], [1300.0], [1.0], timeout_seconds=5.0,
        )
        assert timed_out is False
        assert result == {"fpp": 0.01, "nfpp": 0.001, "verdict": "validated"}

    def test_slow_call_times_out(self):
        """A calc_fpp call that exceeds timeout_seconds returns (None, True)
        instead of blocking the caller — the B3 fix's whole point."""
        def slow_calc_fpp(*a, **kw):
            time.sleep(2.0)
            return {"fpp": 0.5}

        t0 = time.time()
        result, timed_out = pp._run_fpp_with_timeout(
            slow_calc_fpp, "123", 3.0, 1300.0, 0.1, 1000.0,
            [1], [1300.0], [1.0], timeout_seconds=0.1,
        )
        elapsed = time.time() - t0
        assert timed_out is True
        assert result is None
        # Must return promptly (bounded by timeout_seconds), not wait for
        # the full 2s the fake function sleeps.
        assert elapsed < 1.5, f"took {elapsed:.2f}s, expected ~0.1s"

    def test_exception_is_reraised(self):
        """An exception inside calc_fpp_fn propagates to the caller (the
        Step-3 loop's own except Exception around this call handles it,
        matching pre-change behaviour)."""
        def raising_calc_fpp(*a, **kw):
            raise RuntimeError("TRICERATOPS blew up")

        with pytest.raises(RuntimeError, match="TRICERATOPS blew up"):
            pp._run_fpp_with_timeout(
                raising_calc_fpp, "123", 3.0, 1300.0, 0.1, 1000.0,
                [1], [1300.0], [1.0], timeout_seconds=5.0,
            )

    def test_no_result_returns_none_not_timed_out(self):
        """calc_fpp legitimately returning None (its own internal failure
        path) is NOT a timeout."""
        def none_calc_fpp(*a, **kw):
            return None

        result, timed_out = pp._run_fpp_with_timeout(
            none_calc_fpp, "123", 3.0, 1300.0, 0.1, 1000.0,
            [1], [1300.0], [1.0], timeout_seconds=5.0,
        )
        assert timed_out is False
        assert result is None


# ---------------------------------------------------------------------------
# Review follow-up #2 (2026-07-02) — fpp-timeout visibility: a deliberately
# greppable stdout WARNING line, surfaced by the orchestrator on success too.
# ---------------------------------------------------------------------------

class TestFppTimeoutSummaryLine:
    def test_format_single_tic(self):
        line = pp._fpp_timeout_summary_line(["123456789"], 1, 900.0)
        assert line.startswith("WARNING fpp-timeout:")
        assert "1 candidate(s)" in line
        assert "900s" in line
        assert "123456789" in line

    def test_format_truncates_long_list(self):
        tics = [str(i) for i in range(25)]
        line = pp._fpp_timeout_summary_line(tics, 25, 900.0)
        assert "25 candidate(s)" in line
        # Only the first 20 TICs are listed verbatim, plus a "N more" tail.
        for t in tics[:20]:
            assert t in line
        assert "5 more" in line


class TestStepPostprocessSurfacesFppTimeoutWarning:
    def test_success_with_warning_logs_and_returns_nonempty_detail(self, tmp_path, caplog):
        """A successful (rc=0) postprocess subprocess that printed the
        WARNING fpp-timeout line must have it logged AND returned in detail
        — previously only failures (rc!=0) surfaced stdout at all."""
        import logging as _logging

        lot_csv = str(tmp_path / "s0001_lot4.csv")
        Path(lot_csv).write_text("tic_id,label\n12345,candidate\n")
        sf = str(tmp_path / "state.json")

        warn_line = ("WARNING fpp-timeout: 2 candidate(s) skipped FPP "
                     "(exceeded 900s): TIC 111, 222")

        def fake_run(cmd, *args, **kwargs):
            r = MagicMock()
            r.returncode = 0
            r.stdout = f"[fpp] 1 from DB, 3 recomputed, 2 timeout (0 failed)\n{warn_line}\n"
            r.stderr = ""
            return r

        with patch("subprocess.run", side_effect=fake_run), \
             caplog.at_level(_logging.WARNING, logger="discovery_orchestrator"):
            ok, detail = orch.step_postprocess(
                lot_csv=lot_csv, sector=1, db=str(tmp_path / "exonit.db"),
                label="candidate", state_file=sf, dry_run=False,
            )

        assert ok is True
        assert "WARNING fpp-timeout" in detail, (
            "the warning line must be surfaced in the returned detail even "
            "on a successful subprocess run"
        )
        assert warn_line in caplog.text, (
            "the warning line must reach the orchestrator's own log stream"
        )

        state = orch._load_state(sf)
        assert "fpp_timeout_warning" in state
        assert "WARNING fpp-timeout" in state["fpp_timeout_warning"]

    def test_success_without_warning_detail_stays_empty(self, tmp_path):
        """No behaviour change for the common case: no timeout line in
        stdout -> detail stays "" (matches the pre-existing test's
        assumption in test_discovery_orchestrator.py)."""
        lot_csv = str(tmp_path / "s0001_lot4.csv")
        Path(lot_csv).write_text("tic_id,label\n12345,candidate\n")
        sf = str(tmp_path / "state.json")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "[fpp] 1 from DB, 3 recomputed, 0 timeout (0 failed)\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            ok, detail = orch.step_postprocess(
                lot_csv=lot_csv, sector=1, db=str(tmp_path / "exonit.db"),
                label="candidate", state_file=sf, dry_run=False,
            )

        assert ok is True
        assert detail == ""
        state = orch._load_state(sf)
        assert "fpp_timeout_warning" not in state


# ---------------------------------------------------------------------------
# D2 — discovery_postprocess._select_in_chunks delegates to
# exonit.db_backend.select_in_chunks
# ---------------------------------------------------------------------------

class TestSelectInChunksDelegation:
    def test_delegates_to_db_backend_select_in_chunks(self):
        """_select_in_chunks must call exonit.db_backend.select_in_chunks
        (not a re-implementation) with con.execute as the runner."""
        con = MagicMock()
        with patch.object(pp, "select_in_chunks") as mock_select:
            mock_select.return_value = [{"id": 1}]
            out = pp._select_in_chunks(
                con, "SELECT id FROM t WHERE id IN", [1, 2, 3], " AND x=1"
            )
        assert out == [{"id": 1}]
        mock_select.assert_called_once_with(
            con.execute, "SELECT id FROM t WHERE id IN", [1, 2, 3], " AND x=1",
            chunk_size=30000,
        )

    def test_end_to_end_against_sqlite(self, tmp_path):
        """Real end-to-end: a chunked IN query against a throwaway SQLite DB
        returns the same rows an un-chunked query would."""
        from exonit.db_backend import op_session

        db_path = str(tmp_path / "t.db")
        con = op_session(db_path)
        try:
            con.execute("CREATE TABLE t (id INTEGER, val TEXT)")
            ids = list(range(1, 101))
            for i in ids:
                con.execute("INSERT INTO t (id, val) VALUES (?, ?)", (i, f"v{i}"))
            con.commit()

            # chunk_size smaller than the id count forces >1 chunk
            rows = pp._select_in_chunks(
                con, "SELECT id, val FROM t WHERE id IN", ids, chunk_size=10,
            )
        finally:
            con.close()

        assert len(rows) == 100
        assert {r["id"] for r in rows} == set(ids)


# ---------------------------------------------------------------------------
# B3 — discovery_orchestrator subprocess timeouts
# ---------------------------------------------------------------------------

class TestOrchestratorSubprocessTimeouts:
    def test_step_postprocess_timeout_expired_fails_with_clear_detail(self, tmp_path):
        """A TimeoutExpired on the discovery_postprocess.py subprocess must
        return (False, detail) with a clear timeout message — not hang, and
        not just the generic 'exception: ...' text."""
        lot_csv = str(tmp_path / "s0001_lot4.csv")
        Path(lot_csv).write_text("tic_id,label\n12345,candidate\n")
        sf = str(tmp_path / "state.json")

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["x"], timeout=21600),
        ):
            ok, detail = orch.step_postprocess(
                lot_csv=lot_csv, sector=1, db=str(tmp_path / "exonit.db"),
                label="candidate", state_file=sf, dry_run=False,
            )
        assert ok is False
        assert "timeout" in detail.lower()
        assert "postprocess" in detail.lower()

    def test_step_lora_timeout_expired_fails_with_clear_detail(self, monkeypatch):
        # 2026-09-26 incident guard: Fase 3 changed the DEFAULT LORA_CMD to
        # the Macs driver (subprocess.Popen branch, NOT subprocess.run) —
        # this test is specifically about the <host>/gpuq manual-fallback
        # behaviour (asserts "<host>"/"gpuq" in the detail below), so
        # force that path explicitly instead of relying on it being the
        # default. Without this, patching only `subprocess.run` no longer
        # intercepts anything and the REAL lora_macs_driver.py runs
        # (see tests/test_kill_and_resume.py's fix + progress/handoff for
        # the incident writeup).
        monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=orch.LORA_SSH_CMD, timeout=86400),
        ):
            ok, detail = orch.step_lora(state_file="unused", dry_run=False)
        assert ok is False
        assert "timeout" in detail.lower()
        assert "lora" in detail.lower()
        # Review follow-up (2026-07-02): the FAILED detail must tell the
        # operator the remote <host> gpuq job may still be running
        # (shell=True only kills the local ssh shell on TimeoutExpired).
        assert "gpu host" in detail.lower()
        assert "gpuq" in detail.lower()

    def test_step_lora_timeout_triggers_remote_gpuq_cancel_with_expected_argv(self, monkeypatch):
        """Fase 3 Task 5: on TimeoutExpired, step_lora must best-effort cancel
        the remote 'exonit-lora' gpuq job so it doesn't wedge the <host>
        GPU queue for every other consumer (HARD RULE).

        Verified signature (2026-07-05, `gpuq cancel --help` /
        `gpuq status --help` over ssh, read-only): `gpuq cancel <id> [--kill]`
        takes a numeric job id (NOT --name); the id must be discovered from
        `gpuq status` output, matching a line for name=exonit-lora with
        status running/queued. `--kill` is required or cancel only flips the
        DB row without touching the still-running process.
        """
        # See test_step_lora_timeout_expired_fails_with_clear_detail above:
        # force the <host> path so `subprocess.run` patching below is
        # actually hit (Fase 3's default LORA_CMD uses subprocess.Popen).
        monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
        status_result = MagicMock()
        status_result.returncode = 0
        status_result.stdout = (
            "GPU: 2/16311 MiB\n"
            "GPU apps: none\n"
            "Jobs:\n"
            "  #233 done exit=0 name=thumb-seedtest created=2026-07-04T15:31:51+00:00 started=2026-07-04T15:31:51+00:00 note=x\n"
            "  #231 running exit=None name=exonit-lora created=2026-07-04T14:20:25+00:00 started=2026-07-04T14:20:25+00:00 note=x\n"
        )
        status_result.stderr = ""

        cancel_result = MagicMock()
        cancel_result.returncode = 0
        cancel_result.stdout = ""
        cancel_result.stderr = ""

        calls = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired(cmd=orch.LORA_SSH_CMD, timeout=86400)
            if len(calls) == 2:
                return status_result
            return cancel_result

        with patch("subprocess.run", side_effect=fake_run):
            ok, detail = orch.step_lora(state_file="unused", dry_run=False)

        assert ok is False
        assert len(calls) == 3, f"expected main + status + cancel calls, got {calls}"
        assert "gpuq status" in calls[1]
        assert "gpuq cancel 231" in calls[2]
        assert "--kill" in calls[2]
        # Original timeout detail text is unchanged by the cancel attempt.
        assert "gpu host" in detail.lower()
        assert "gpuq" in detail.lower()

    def test_step_lora_cancel_prefers_running_job_over_newer_queued_one(self, monkeypatch):
        """Review fix (Task 5): with TWO matching 'exonit-lora' jobs — the
        stuck RUNNING one (lower id, since gpuq is FIFO and it was enqueued
        first) and a newer QUEUED one behind it (e.g. a resume/retry after
        the stuck run) — the cancel must target the RUNNING job's id.
        Cancelling the queued one would leave the wedging job alive and log
        a misleading success."""
        monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
        status_result = MagicMock()
        status_result.returncode = 0
        status_result.stdout = (
            "GPU: 2/16311 MiB\n"
            "GPU apps: none\n"
            "Jobs:\n"
            "  #240 queued exit=None name=exonit-lora created=2026-07-05T02:00:00+00:00 started=None note=x\n"
            "  #231 running exit=None name=exonit-lora created=2026-07-04T14:20:25+00:00 started=2026-07-04T14:20:25+00:00 note=x\n"
        )
        status_result.stderr = ""

        cancel_result = MagicMock()
        cancel_result.returncode = 0
        cancel_result.stdout = ""
        cancel_result.stderr = ""

        calls = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired(cmd=orch.LORA_SSH_CMD, timeout=86400)
            if len(calls) == 2:
                return status_result
            return cancel_result

        with patch("subprocess.run", side_effect=fake_run):
            ok, detail = orch.step_lora(state_file="unused", dry_run=False)

        assert ok is False
        assert len(calls) == 3
        assert "gpuq cancel 231" in calls[2], (
            f"must cancel the RUNNING job (#231), not the newer queued one "
            f"(#240); got: {calls[2]}"
        )
        assert "--kill" in calls[2]

    def test_step_lora_cancel_falls_back_to_newest_queued_when_none_running(self, monkeypatch):
        """No running match → cancel the highest-id QUEUED 'exonit-lora'."""
        monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
        status_result = MagicMock()
        status_result.returncode = 0
        status_result.stdout = (
            "Jobs:\n"
            "  #240 queued exit=None name=exonit-lora created=2026-07-05T02:00:00+00:00 started=None note=x\n"
            "  #238 queued exit=None name=exonit-lora created=2026-07-05T01:00:00+00:00 started=None note=x\n"
            "  #231 done exit=0 name=exonit-lora created=2026-07-04T14:20:25+00:00 started=2026-07-04T14:20:25+00:00 note=x\n"
        )
        status_result.stderr = ""

        cancel_result = MagicMock()
        cancel_result.returncode = 0
        cancel_result.stdout = ""
        cancel_result.stderr = ""

        calls = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired(cmd=orch.LORA_SSH_CMD, timeout=86400)
            if len(calls) == 2:
                return status_result
            return cancel_result

        with patch("subprocess.run", side_effect=fake_run):
            ok, detail = orch.step_lora(state_file="unused", dry_run=False)

        assert ok is False
        assert len(calls) == 3
        assert "gpuq cancel 240" in calls[2]

    def test_step_lora_remote_cancel_failure_does_not_change_step_outcome(self, monkeypatch):
        """The remote cancel is best-effort: if the status/cancel ssh calls
        themselves fail (here: every subprocess.run call raises, including
        the cancel attempts), step_lora's outcome must be byte-identical to
        the pre-Task-5 timeout outcome — no exception escapes, no different
        return value."""
        monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=orch.LORA_SSH_CMD, timeout=86400),
        ):
            ok, detail = orch.step_lora(state_file="unused", dry_run=False)
        assert ok is False
        assert "timeout" in detail.lower()
        assert "gpu host" in detail.lower()
        assert "gpuq" in detail.lower()

    def test_step_lora_happy_path_never_calls_remote_cancel(self, monkeypatch):
        """No cancel call must happen when the main ssh+gpuq call succeeds."""
        monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
        ok_result = MagicMock()
        ok_result.returncode = 0
        ok_result.stdout = ""
        ok_result.stderr = ""

        calls = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            return ok_result

        with patch("subprocess.run", side_effect=fake_run):
            ok, detail = orch.step_lora(state_file="unused", dry_run=False)

        assert ok is True
        assert len(calls) == 1, f"happy path must not invoke any cancel call, got {calls}"

    def test_step_aws_finalize_export_timeout_is_fatal(self, tmp_path, monkeypatch):
        """AWS export timing out must be FATAL (False, detail) — the bundle
        was never confirmed uploaded."""
        sf = str(tmp_path / "pipeline_state.json")
        import json
        Path(sf).write_text(json.dumps({
            "state": "POSTPROCESS",
            "run_id": "discovery_s0002_lot20260609T120000_2026-06-09",
            "scorecard_json": "data/scorecards/x.json",
            "dossier": "data/scorecards/x_dossier.md",
        }))
        lot_csv = str(tmp_path / "s0002_lot20260609T120000.csv")
        Path(lot_csv).write_text("tic_id,label\n99999,candidate\n")
        monkeypatch.setenv("EXONIT_AWS_S3_RUNS_PREFIX", "s3://exonit-sector2/runs")

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["export_run_bundle.py"], timeout=7200),
        ):
            ok, detail = orch.step_aws_finalize(sf, lot_csv, sector=2, db=None, dry_run=False)

        assert ok is False
        assert "timeout" in detail.lower()
        assert "export" in detail.lower()

    def test_step_aws_finalize_sync_timeout_is_non_fatal(self, tmp_path, monkeypatch):
        """The latest/ mirror sync timing out must stay non-fatal — the
        run_prefix export is already confirmed and is the source of truth."""
        sf = str(tmp_path / "pipeline_state.json")
        import json
        Path(sf).write_text(json.dumps({
            "state": "POSTPROCESS",
            "run_id": "discovery_s0002_lot20260609T120000_2026-06-09",
            "scorecard_json": "data/scorecards/x.json",
            "dossier": "data/scorecards/x_dossier.md",
        }))
        lot_csv = str(tmp_path / "s0002_lot20260609T120000.csv")
        Path(lot_csv).write_text("tic_id,label\n99999,candidate\n")
        monkeypatch.setenv("EXONIT_AWS_S3_RUNS_PREFIX", "s3://exonit-sector2/runs")

        def fake_run(cmd, *args, **kwargs):
            flat = " ".join(str(x) for x in cmd)
            if "sync" in flat:
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=7200)
            r = MagicMock()
            r.returncode = 0
            r.stdout = "{}"
            r.stderr = ""
            return r

        with patch("subprocess.run", side_effect=fake_run), \
             patch.object(orch, "step_finalize", return_value=(True, "ok")), \
             patch("urllib.request.urlopen", side_effect=Exception("no IMDS")):
            ok, detail = orch.step_aws_finalize(sf, lot_csv, sector=2, db=None, dry_run=False)

        assert ok is True, "sync timeout must NOT fail the whole step"


# ---------------------------------------------------------------------------
# D4 — _poll_until_drained is genuinely shared (not 3 copies)
# ---------------------------------------------------------------------------

class TestPollUntilDrainedSharedHelper:
    def test_all_three_steps_call_the_same_helper(self):
        """Regression guard for the dedup itself: step_scanning,
        step_fpp_backfill and step_exominer_backfill must all route through
        the single _poll_until_drained helper (not three separate loops)."""
        calls = []
        original = orch._poll_until_drained

        def spy(*a, **kw):
            calls.append(kw.get("label"))
            return True, ""

        with patch.object(orch, "_poll_until_drained", side_effect=spy):
            orch.step_scanning("http://localhost:8400", "unused", 1, dry_run=False)
            orch.step_fpp_backfill("http://localhost:8400", "unused", 1, dry_run=False)
            import os as _os
            _os.environ["EXONIT_EXOMINER_BACKFILL"] = "1"
            try:
                orch.step_exominer_backfill("http://localhost:8400", "unused", 1, dry_run=False)
            finally:
                _os.environ.pop("EXONIT_EXOMINER_BACKFILL", None)

        assert calls == ["SCANNING", "FPP_BACKFILL", "EXOMINER_BACKFILL"]
