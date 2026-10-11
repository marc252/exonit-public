"""Tests for exonit/done_state.py — the (TIC, sector) done-state design
(docs/superpowers/specs/2026-09-26-tic-sector-done-state-design.md).

Mirrors tests/test_deep_rescan.py's fixtures/patterns (SQLite tmp_path DB +
FastAPI TestClient). Never touches the live DB.
"""
import os

import pytest

from exonit import done_state
from exonit.db import ExoNitDB


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def db(tmp_path):
    return ExoNitDB(str(tmp_path / "test.db"))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE",
                       str(tmp_path / "pipeline_state.json"))
    from exonit.server import create_app
    from fastapi.testclient import TestClient
    app = create_app()
    with TestClient(app) as c:
        yield c


AUTH = {"Authorization": "Bearer exonit-default-token",
        "x-worker-id": "test-worker"}


def _cand(num, period):
    return {"candidate_num": num, "period_days": period, "t0": 0.0,
            "duration_days": 0.1, "depth_ppm": 500.0, "snr": 12.0}


def _post_result(client, tic_id, status="completed", n_sectors=1,
                 candidates=None, task_id=None, error=None):
    body = {
        "task_id": task_id or ("t-" + tic_id), "tic_id": tic_id,
        "status": status, "n_sectors": n_sectors, "n_points": 1000,
        "time_span_days": 26.0, "ra": 1.0, "dec": 2.0, "tmag": 10.0,
        "candidates": candidates or [],
        "cnn_scores": [0.8] * len(candidates or []),
        "processing_time_seconds": 1.0,
    }
    if error is not None:
        body["error"] = error
    return client.post("/api/worker/result", json=body, headers=AUTH)


def _tss_row(db, tic_id, sector):
    return db._query_one(
        "SELECT * FROM tic_sector_state WHERE tic_id = ? AND sector = ?",
        (tic_id, sector))


# --------------------------------------------------------------------------- #
# Write path: mark_tic_done / mark_tic_failed
# --------------------------------------------------------------------------- #

class TestMarkTicDoneFailed:
    def test_mark_tic_done_creates_completed_row(self, db):
        db.add_task("tk1", "100", sector=5)
        done_state.mark_tic_done(db.backend, "100", "tk1")
        row = _tss_row(db, "100", 5)
        assert row["status"] == "completed"
        assert row["last_task_id"] == "tk1"

    def test_mark_tic_done_defaults_legacy_sector_when_task_missing(self, db):
        # No tasks row at all for this task_id -> sector 0 (LEGACY_SECTOR).
        done_state.mark_tic_done(db.backend, "101", "ghost-task")
        row = _tss_row(db, "101", 0)
        assert row is not None and row["status"] == "completed"

    def test_mark_tic_done_defaults_legacy_sector_when_sector_null(self, db):
        db.add_task("tk2", "102")   # sector omitted -> NULL
        done_state.mark_tic_done(db.backend, "102", "tk2")
        row = _tss_row(db, "102", 0)
        assert row is not None

    def test_mark_tic_failed_first_attempt_not_exhausted(self, db):
        db.add_task("tk3", "200", sector=7)
        done_state.mark_tic_failed(db.backend, "200", "tk3", error="boom")
        row = _tss_row(db, "200", 7)
        assert row["status"] == "failed"
        assert row["attempts"] == 1
        assert row["last_error"] == "boom"
        assert "200" not in done_state.exhausted_tic_ids(db.backend.path, 7)

    def test_mark_tic_failed_second_attempt_is_exhausted(self, db):
        db.add_task("tk4", "201", sector=7)
        done_state.mark_tic_failed(db.backend, "201", "tk4", error="e1")
        db.add_task("tk4b", "201", sector=7)
        done_state.mark_tic_failed(db.backend, "201", "tk4b", error="e2")
        row = _tss_row(db, "201", 7)
        assert row["status"] == "failed"
        assert row["attempts"] == 2
        assert row["last_error"] == "e2"
        assert "201" in done_state.exhausted_tic_ids(db.backend.path, 7)

    def test_completed_status_never_downgraded_by_later_failure(self, db):
        """§1's monotonicity requirement (a): done(t) never goes back to
        false, even after MAX_LOTS_PER_SECTOR+ subsequent terminal failures
        for the SAME (tic, sector)."""
        db.add_task("tk5", "202", sector=3)
        done_state.mark_tic_done(db.backend, "202", "tk5")
        db.add_task("tk5b", "202", sector=3)
        done_state.mark_tic_failed(db.backend, "202", "tk5b", error="e1")
        db.add_task("tk5c", "202", sector=3)
        done_state.mark_tic_failed(db.backend, "202", "tk5c", error="e2")
        row = _tss_row(db, "202", 3)
        # attempts kept climbing (diagnostic value) but status stays completed
        assert row["status"] == "completed"
        assert row["attempts"] == 2
        assert "202" in done_state.load_done_tic_set(db.backend.path)

    def test_failed_then_completed_upgrades_to_completed(self, db):
        db.add_task("tk6", "203", sector=3)
        done_state.mark_tic_failed(db.backend, "203", "tk6", error="e1")
        db.add_task("tk6b", "203", sector=3)
        done_state.mark_tic_done(db.backend, "203", "tk6b")
        row = _tss_row(db, "203", 3)
        assert row["status"] == "completed"

    def test_different_sectors_are_independent_rows(self, db):
        db.add_task("tk7", "204", sector=1)
        done_state.mark_tic_done(db.backend, "204", "tk7")
        db.add_task("tk7b", "204", sector=2)
        done_state.mark_tic_failed(db.backend, "204", "tk7b", error="e")
        assert _tss_row(db, "204", 1)["status"] == "completed"
        assert _tss_row(db, "204", 2)["status"] == "failed"
        # done(t) is sector-agnostic: true because of the sector=1 row.
        assert "204" in done_state.load_done_tic_set(db.backend.path)


# --------------------------------------------------------------------------- #
# Worker-result route: dual-write always on, §2.2 star-overwrite guard
# --------------------------------------------------------------------------- #

class TestWorkerResultDualWrite:
    def test_completed_result_writes_tic_sector_state(self, client, tmp_path):
        db = ExoNitDB(str(tmp_path / "test.db"))
        db.add_task("t-300", "300", sector=9)
        r = _post_result(client, "300", status="completed", task_id="t-300")
        assert r.status_code == 200
        row = _tss_row(db, "300", 9)
        assert row is not None and row["status"] == "completed"

    def test_terminal_failure_writes_tic_sector_state(self, client, tmp_path):
        db = ExoNitDB(str(tmp_path / "test.db"))
        db.add_task("t-301", "301", sector=9)
        # MAX_TASK_ATTEMPTS default 3: assign_task bumps attempts each claim.
        db.assign_task("t-301", "w1")
        db.assign_task("t-301", "w1")
        db.assign_task("t-301", "w1")   # attempts=3 -> next failure terminal
        r = _post_result(client, "301", status="failed", task_id="t-301",
                         error="sparseLightCurve")
        assert r.status_code == 200
        row = _tss_row(db, "301", 9)
        assert row is not None and row["status"] == "failed"
        assert row["attempts"] == 1
        assert row["last_error"] == "sparseLightCurve"

    def test_internal_requeue_does_not_write_tic_sector_state(
            self, client, tmp_path):
        """A non-terminal failure (attempts still < max) requeues internally
        — must NOT count toward exhausted(t, S)."""
        db = ExoNitDB(str(tmp_path / "test.db"))
        db.add_task("t-302", "302", sector=9)
        db.assign_task("t-302", "w1")   # attempts=1 (< MAX_TASK_ATTEMPTS)
        r = _post_result(client, "302", status="failed", task_id="t-302",
                         error="transient")
        assert r.status_code == 200
        assert _tss_row(db, "302", 9) is None

    def test_dual_write_survives_missing_backend_method(
            self, client, tmp_path, monkeypatch):
        """Writes must never break the result path (spec: wrap in
        try/except + logger.warning)."""
        import exonit.routes.workers as workers_mod
        monkeypatch.setattr(
            workers_mod.done_state, "mark_tic_done",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        r = _post_result(client, "303", status="completed", task_id="t-303")
        assert r.status_code == 200

    def test_failed_after_completed_star_not_overwritten(
            self, client, tmp_path):
        """§2.2 (operator decision #2): a failed/aborted result must not
        clobber an already-completed star's descriptive fields."""
        db = ExoNitDB(str(tmp_path / "test.db"))
        _post_result(client, "304", status="completed",
                     candidates=[_cand(1, 3.0)], task_id="t-304a")
        star_before = db.get_star("304")
        assert star_before["n_sectors"] == 1

        db.add_task("t-304b", "304", sector=1)
        r = _post_result(client, "304", status="failed", task_id="t-304b",
                         n_sectors=0, error="load_failed")
        assert r.status_code == 200
        star_after = db.get_star("304")
        assert star_after["status"] == "completed"
        assert star_after["n_sectors"] == 1
        # Candidates from the earlier completed pass are untouched.
        assert len(db.get_candidates("304")) == 1

    def test_failed_on_never_completed_star_still_writes_star(
            self, client, tmp_path):
        """The §2.2 guard only protects a PREVIOUSLY completed star — a
        first-ever failed result still records the star (unchanged
        behaviour: e.g. no_fits / load_failed bookkeeping)."""
        db = ExoNitDB(str(tmp_path / "test.db"))
        db.add_task("t-305", "305", sector=1)
        r = _post_result(client, "305", status="failed", task_id="t-305",
                         n_sectors=0, error="load_failed")
        assert r.status_code == 200
        star = db.get_star("305")
        assert star is not None and star["status"] == "failed"


# --------------------------------------------------------------------------- #
# is_enabled()
# --------------------------------------------------------------------------- #

class TestIsEnabled:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("EXONIT_DONE_STATE", raising=False)
        assert done_state.is_enabled() is False

    def test_on_when_set_to_1(self, monkeypatch):
        monkeypatch.setenv("EXONIT_DONE_STATE", "1")
        assert done_state.is_enabled() is True

    def test_off_for_any_other_value(self, monkeypatch):
        monkeypatch.setenv("EXONIT_DONE_STATE", "true")
        assert done_state.is_enabled() is False


# --------------------------------------------------------------------------- #
# Read predicates: done_and_exhausted / blocked_for_enqueue
# --------------------------------------------------------------------------- #

class TestDoneAndExhausted:
    def _seed(self, db):
        # 100: done (sector 1) — must stay excluded from ANY sector's
        # selection (§1(a), monotonic).
        db.add_task("a1", "100", sector=1)
        done_state.mark_tic_done(db.backend, "100", "a1")
        # 200: failed once for sector 2 — still selectable (§1(b)).
        db.add_task("a2", "200", sector=2)
        done_state.mark_tic_failed(db.backend, "200", "a2", error="e")
        # 300: failed twice for sector 2 — exhausted, excluded (§1(b)).
        db.add_task("a3", "300", sector=2)
        done_state.mark_tic_failed(db.backend, "300", "a3", error="e1")
        db.add_task("a3b", "300", sector=2)
        done_state.mark_tic_failed(db.backend, "300", "a3b", error="e2")
        # 400: failed twice for sector 5 (a DIFFERENT sector) — must NOT
        # block sector 2's selection.
        db.add_task("a4", "400", sector=5)
        done_state.mark_tic_failed(db.backend, "400", "a4", error="e1")
        db.add_task("a4b", "400", sector=5)
        done_state.mark_tic_failed(db.backend, "400", "a4b", error="e2")
        # 500: never touched — plainly selectable.

    def test_done_and_exhausted_for_sector_2(self, db):
        self._seed(db)
        ids = ["100", "200", "300", "400", "500"]
        blocked = done_state.done_and_exhausted(db.backend.path, 2, ids)
        assert blocked == {"100", "300"}

    def test_exhausted_is_per_sector(self, db):
        self._seed(db)
        ids = ["100", "200", "300", "400", "500"]
        blocked_s5 = done_state.done_and_exhausted(db.backend.path, 5, ids)
        assert blocked_s5 == {"100", "400"}

    def test_preloaded_done_param_used_instead_of_reload(self, db):
        self._seed(db)
        # Pass a deliberately WRONG preloaded `done` set (empty) — 100 must
        # then NOT be excluded via the done(t) path (only exhausted(2) is
        # still computed live), proving `done` short-circuits the DB scan.
        blocked = done_state.done_and_exhausted(
            db.backend.path, 2, ["100", "300"], done=set())
        assert blocked == {"300"}

    def test_blocked_for_enqueue_matches_done_and_exhausted(self, db):
        self._seed(db)
        ids = ["100", "200", "300", "400", "500"]
        via_path = done_state.done_and_exhausted(db.backend.path, 2, ids)
        via_backend = done_state.blocked_for_enqueue(db.backend, 2, ids)
        assert via_path == via_backend == {"100", "300"}

    def test_empty_ids_returns_empty(self, db):
        assert done_state.done_and_exhausted(db.backend.path, 1, []) == set()
        assert done_state.blocked_for_enqueue(db.backend, 1, []) == set()


# --------------------------------------------------------------------------- #
# /api/worker/enqueue_bulk dedup — EXONIT_DONE_STATE=1
# --------------------------------------------------------------------------- #

class TestEnqueueBulkDoneStateDedup:
    def _enqueue_bulk(self, client, tics, sector=1, force=False):
        body = {"targets": [{"tic_id": t} for t in tics], "sector": sector}
        if force:
            body["force_rescan"] = True
        return client.post("/api/worker/enqueue_bulk", json=body, headers=AUTH)

    def test_flag_off_ignores_tic_sector_state(self, client, tmp_path,
                                               monkeypatch):
        """Default (flag off): a done(t)/exhausted TIC with NO task row at
        all is NOT blocked — pre-existing 'any task row' dedup only."""
        monkeypatch.delenv("EXONIT_DONE_STATE", raising=False)
        db = ExoNitDB(str(tmp_path / "test.db"))
        done_state.mark_tic_done(db.backend, "900", None)
        r = self._enqueue_bulk(client, ["900"])
        assert r.json() == {"enqueued": 1, "skipped": 0, "failed": 0,
                            "adopted": 0,
                            "elapsed_seconds": r.json()["elapsed_seconds"]}

    def test_flag_on_blocks_done_tic(self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("EXONIT_DONE_STATE", "1")
        db = ExoNitDB(str(tmp_path / "test.db"))
        done_state.mark_tic_done(db.backend, "901", None)   # sector 0
        r = self._enqueue_bulk(client, ["901"], sector=0)
        assert r.json()["enqueued"] == 0 and r.json()["skipped"] == 1

    def test_flag_on_single_failure_still_selectable(
            self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("EXONIT_DONE_STATE", "1")
        db = ExoNitDB(str(tmp_path / "test.db"))
        db.add_task("f1", "902", sector=3)
        db.fail_task("f1")   # the task row itself must not stay 'pending'
        done_state.mark_tic_failed(db.backend, "902", "f1", error="e")
        r = self._enqueue_bulk(client, ["902"], sector=3)
        assert r.json()["enqueued"] == 1 and r.json()["skipped"] == 0

    def test_flag_on_exhausted_tic_blocked(self, client, tmp_path,
                                           monkeypatch):
        monkeypatch.setenv("EXONIT_DONE_STATE", "1")
        db = ExoNitDB(str(tmp_path / "test.db"))
        db.add_task("f2", "903", sector=3)
        db.fail_task("f2")
        done_state.mark_tic_failed(db.backend, "903", "f2", error="e1")
        db.add_task("f2b", "903", sector=3)
        db.fail_task("f2b")
        done_state.mark_tic_failed(db.backend, "903", "f2b", error="e2")
        r = self._enqueue_bulk(client, ["903"], sector=3)
        assert r.json()["enqueued"] == 0 and r.json()["skipped"] == 1

    def test_flag_on_exhausted_in_other_sector_not_blocked(
            self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("EXONIT_DONE_STATE", "1")
        db = ExoNitDB(str(tmp_path / "test.db"))
        db.add_task("f3", "904", sector=3)
        db.fail_task("f3")
        done_state.mark_tic_failed(db.backend, "904", "f3", error="e1")
        db.add_task("f3b", "904", sector=3)
        db.fail_task("f3b")
        done_state.mark_tic_failed(db.backend, "904", "f3b", error="e2")
        r = self._enqueue_bulk(client, ["904"], sector=99)
        assert r.json()["enqueued"] == 1 and r.json()["skipped"] == 0

    def test_flag_on_force_rescan_bypasses_done_and_exhausted(
            self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("EXONIT_DONE_STATE", "1")
        db = ExoNitDB(str(tmp_path / "test.db"))
        done_state.mark_tic_done(db.backend, "905", None)
        r = self._enqueue_bulk(client, ["905"], sector=0, force=True)
        assert r.json()["enqueued"] == 1 and r.json()["skipped"] == 0

    def test_flag_on_active_task_still_blocks(self, client, tmp_path,
                                              monkeypatch):
        """Active work always blocks, in both modes — unchanged."""
        monkeypatch.setenv("EXONIT_DONE_STATE", "1")
        db = ExoNitDB(str(tmp_path / "test.db"))
        db.add_task("f4", "906", sector=3)   # pending
        r = self._enqueue_bulk(client, ["906"], sector=3, force=True)
        assert r.json()["enqueued"] == 0 and r.json()["skipped"] == 1


# --------------------------------------------------------------------------- #
# Regression 08-10 (empty-lot loop): a sector whose remaining TICs are all
# exhausted must reach gaps_processable == 0 and discovery_run.py must exit
# cleanly (no_undone_tics), 3x in a row (idempotent), never looping.
# --------------------------------------------------------------------------- #

class TestRegression0810EmptyLotLoop:
    def _make_canonical_csv(self, path, tic_ids):
        import csv
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["tic_id", "label"])
            for t in tic_ids:
                w.writerow([t, "candidate"])

    def test_all_exhausted_sector_reports_no_undone_tics_idempotently(
            self, tmp_path, monkeypatch):
        monkeypatch.setenv("EXONIT_DONE_STATE", "1")
        db_path = str(tmp_path / "test.db")
        db = ExoNitDB(db_path)

        tics = [1001, 1002, 1003]
        for t in tics:
            db.add_task(f"tk-{t}", str(t), sector=42)
            done_state.mark_tic_failed(
                db.backend, str(t), f"tk-{t}", error="e1")
            db.add_task(f"tk-{t}-b", str(t), sector=42)
            done_state.mark_tic_failed(
                db.backend, str(t), f"tk-{t}-b", error="e2")

        canonical_csv = tmp_path / "canonical.csv"
        self._make_canonical_csv(canonical_csv, tics)

        registry_path = tmp_path / "sectors.json"
        import json as _json
        registry_path.write_text(_json.dumps({
            "active_sector": 42,
            "sectors": {"42": {
                "canonical_list": str(canonical_csv),
                "total_tics": len(tics), "downloaded": 0, "cursor": 0,
                "enabled": True,
            }},
        }))

        from scripts import discovery_run

        def _run_once():
            return discovery_run.main([
                "--sector", "42", "--count", "10",
                "--db", db_path, "--registry", str(registry_path),
                "--canonical", str(canonical_csv),
                "--tglc-index", str(tmp_path / "no_such_tglc.db"),
                "--toi-catalog", str(tmp_path / "no_such_toi.csv"),
                "--dry-run",
            ])

        # Idempotent: 3 identical runs, never a crash/loop, always the same
        # "nothing left to do" outcome.
        for _ in range(3):
            rc = _run_once()
            assert rc == 2   # no_undone_tics / canonical_list_empty-shaped error


# --------------------------------------------------------------------------- #
# sector_gap_stats: failed_terminal (shape gated by the flag)
# --------------------------------------------------------------------------- #

class TestSectorGapStatsFailedTerminal:
    def _fixture(self, tmp_path, db):
        import csv
        # discovery_no_data is a PG-only table in the live schema (see
        # scripts/migration/schema_pg.sql) — exonit/db.py's SQLite bootstrap
        # never creates it (verify_sector_complete.py explicitly tolerates
        # "no such table" on SQLite). sector_gap_stats queries it
        # unconditionally, so create it here, same as
        # tests/test_discovery_gap_stats.py's PG DDL fixture does for PG.
        db.backend.execute("""
            CREATE TABLE IF NOT EXISTS discovery_no_data (
                tic_id TEXT NOT NULL,
                sector INTEGER NOT NULL,
                reason TEXT NOT NULL DEFAULT 'no_fits',
                checked_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (tic_id, sector)
            )""")
        tics = [2001, 2002, 2003]
        canonical_csv = tmp_path / "canonical.csv"
        with open(canonical_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["tic_id"])
            for t in tics:
                w.writerow([t])
        # 2001: exhausted for sector 7
        db.add_task("g1", "2001", sector=7)
        db.fail_task("g1")
        done_state.mark_tic_failed(db.backend, "2001", "g1", error="e1")
        db.add_task("g1b", "2001", sector=7)
        db.fail_task("g1b")
        done_state.mark_tic_failed(db.backend, "2001", "g1b", error="e2")
        return str(canonical_csv)

    def test_flag_off_no_failed_terminal_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("EXONIT_DONE_STATE", raising=False)
        db = ExoNitDB(str(tmp_path / "test.db"))
        canonical_csv = self._fixture(tmp_path, db)
        from exonit.discovery_state import sector_gap_stats
        gs = sector_gap_stats(
            7, db_path=str(tmp_path / "test.db"), canonical_csv=canonical_csv,
            tglc_index=str(tmp_path / "no_tglc.db"),
            toi_catalog=str(tmp_path / "no_toi.csv"))
        assert "failed_terminal" not in gs

    def test_flag_on_counts_failed_terminal(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EXONIT_DONE_STATE", "1")
        db = ExoNitDB(str(tmp_path / "test.db"))
        canonical_csv = self._fixture(tmp_path, db)
        from exonit.discovery_state import sector_gap_stats
        gs = sector_gap_stats(
            7, db_path=str(tmp_path / "test.db"), canonical_csv=canonical_csv,
            tglc_index=str(tmp_path / "no_tglc.db"),
            toi_catalog=str(tmp_path / "no_toi.csv"))
        assert gs["failed_terminal"] == 1
        assert gs["gaps_processable"] == 2   # 3 total - 1 exhausted


# --------------------------------------------------------------------------- #
# Review fix #1 (2026-09-26, progress/review-done-state.md): the DLQ
# reconciliation path (routes/workers.py's _reconcile_primary_dlq ->
# db.reconcile_dlq_task) is a SECOND, independent way a task can transition
# to terminal 'failed' — it must also dual-write to tic_sector_state, or a
# TIC that fails via THIS path never accumulates attempts and can never
# reach exhausted(t, S) (same unbounded-reintent shape as the 08-10
# regression, via a different trigger). Mirrors
# tests/test_primary_stream_handler.py's TestPrimaryDlqReconciliation
# fixtures/patterns (MemoryWorkStream injected via monkeypatching
# exonit.work_streams.get_stream).
# --------------------------------------------------------------------------- #

class TestDlqTerminalFailureDualWrite:
    def _build_stream_client(self, tmp_path, monkeypatch,
                             visibility_timeout_s=0.2):
        from exonit import work_streams
        db_path = str(tmp_path / "test.db")
        monkeypatch.setenv("EXONIT_DB_PATH", db_path)
        monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE",
                           str(tmp_path / "pipeline_state.json"))
        monkeypatch.setenv("EXONIT_WORK_STREAMS", "1")
        memory_ws = work_streams.MemoryWorkStream(
            "primary", visibility_timeout_s=visibility_timeout_s)
        monkeypatch.setattr(work_streams, "get_stream", lambda name: memory_ws)
        from exonit.server import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        with TestClient(app) as c:
            yield c, db_path, memory_ws

    @pytest.fixture
    def stream_client(self, tmp_path, monkeypatch):
        yield from self._build_stream_client(tmp_path, monkeypatch)

    def _register(self, client, name):
        r = client.post("/api/worker/register",
                        json={"name": name, "capabilities": {}}, headers=AUTH)
        assert r.status_code == 200
        return r.json()["worker_id"]

    def _get_task(self, client, wid):
        return client.get(
            "/api/worker/task",
            headers={"Authorization": "Bearer exonit-default-token",
                    "X-Worker-ID": wid}).json()

    def _drive_to_dlq(self, memory_ws):
        import time as _time
        for _ in range(memory_ws.max_deliveries):
            _time.sleep(0.25)
            memory_ws.claim("ghost", keep_if=lambda k, c: False)
        assert memory_ws.stats()["dlq"] == 1

    def test_dlq_terminal_failure_writes_tic_sector_state(
            self, stream_client):
        from exonit.db import MAX_TASK_ATTEMPTS
        client, db_path, memory_ws = stream_client
        r = client.post(
            "/api/worker/enqueue_bulk",
            json={"targets": [{"tic_id": "800001"}], "sector": 11},
            headers=AUTH)
        assert r.status_code == 200
        wid = self._register(client, "dlq-w1")
        task_id = self._get_task(client, wid)["task_id"]

        db = ExoNitDB(db_path)
        db._exec("UPDATE tasks SET attempts=? WHERE task_id=?",
                 (MAX_TASK_ATTEMPTS, task_id))
        db.close()

        self._drive_to_dlq(memory_ws)

        poller = self._register(client, "poller-w1")
        resp = self._get_task(client, poller)
        assert resp == {"task_id": None, "wait_seconds": 10}

        db = ExoNitDB(db_path)
        task_row = db.get_task(task_id)
        assert task_row["status"] == "failed"
        tss = db._query_one(
            "SELECT * FROM tic_sector_state WHERE tic_id = ? AND sector = ?",
            ("800001", 11))
        db.close()
        assert tss is not None, (
            "DLQ terminal failure must dual-write tic_sector_state, same "
            "as requeue_or_fail_task's terminal path")
        assert tss["status"] == "failed"
        assert tss["attempts"] == 1
        assert tss["last_error"] == "dlq_exhausted_max_deliveries"

    def test_two_dlq_terminal_failures_reach_exhausted(self, stream_client):
        """Two separate DLQ terminal failures for the SAME (tic, sector)
        must accumulate attempts to done_state.MAX_LOTS_PER_SECTOR (2) and
        become exhausted(t, S) — the whole point of this fix."""
        from exonit.db import MAX_TASK_ATTEMPTS
        client, db_path, memory_ws = stream_client

        def _one_dlq_failure(n):
            r = client.post(
                "/api/worker/enqueue_bulk",
                json={"targets": [{"tic_id": "800002"}], "sector": 12,
                     "force_rescan": True},
                headers=AUTH)
            assert r.status_code == 200
            wid = self._register(client, f"dlq-w2-{n}")
            task_id = self._get_task(client, wid)["task_id"]
            db = ExoNitDB(db_path)
            db._exec("UPDATE tasks SET attempts=? WHERE task_id=?",
                     (MAX_TASK_ATTEMPTS, task_id))
            db.close()
            self._drive_to_dlq(memory_ws)
            poller = self._register(client, f"poller-w2-{n}")
            self._get_task(client, poller)

        _one_dlq_failure(1)
        _one_dlq_failure(2)

        db = ExoNitDB(db_path)
        tss = db._query_one(
            "SELECT * FROM tic_sector_state WHERE tic_id = ? AND sector = ?",
            ("800002", 12))
        db.close()
        assert tss["attempts"] == 2
        assert tss["status"] == "failed"
        assert "800002" in done_state.exhausted_tic_ids(db_path, 12)

    def test_dlq_never_downgrades_a_completed_tic(self, stream_client):
        """A DLQ terminal failure for a (tic, sector) already marked
        'completed' (e.g. a stale DLQ entry reconciled after a real
        success landed) must NOT downgrade it — same monotonicity guard as
        mark_tic_failed's CASE clause."""
        from exonit.db import MAX_TASK_ATTEMPTS
        client, db_path, memory_ws = stream_client

        # Seed tic_sector_state directly (NOT via db.add_task): a stray
        # 'pending' tasks row for this TIC would itself block the
        # force_rescan enqueue below via the unrelated active-task check,
        # which is not what this test exercises.
        db = ExoNitDB(db_path)
        db.backend.execute(
            "INSERT INTO tic_sector_state (tic_id, sector, status, attempts) "
            "VALUES (?, ?, 'completed', 0)",
            ("800003", 13))
        db.close()

        r = client.post(
            "/api/worker/enqueue_bulk",
            json={"targets": [{"tic_id": "800003"}], "sector": 13,
                 "force_rescan": True},
            headers=AUTH)
        assert r.status_code == 200
        wid = self._register(client, "dlq-w3")
        task_id = self._get_task(client, wid)["task_id"]

        db = ExoNitDB(db_path)
        db._exec("UPDATE tasks SET attempts=? WHERE task_id=?",
                 (MAX_TASK_ATTEMPTS, task_id))
        db.close()

        self._drive_to_dlq(memory_ws)
        poller = self._register(client, "poller-w3")
        self._get_task(client, poller)

        db = ExoNitDB(db_path)
        tss = db._query_one(
            "SELECT * FROM tic_sector_state WHERE tic_id = ? AND sector = ?",
            ("800003", 13))
        db.close()
        assert tss["status"] == "completed"
        assert tss["attempts"] == 1   # counter still climbs (diagnostic)


# --------------------------------------------------------------------------- #
# Review fix #2 (2026-09-26): enqueue_bulk fails CLOSED when
# done_state.blocked_for_enqueue raises under EXONIT_DONE_STATE=1 — rejects
# the whole batch (503) instead of silently falling back to a widened,
# unprotected dedup. Flag off: unaffected (blocked_for_enqueue is never
# called at all).
# --------------------------------------------------------------------------- #

class TestEnqueueBulkFailsClosedOnDoneStateError:
    def test_flag_on_blocked_for_enqueue_raises_rejects_batch(
            self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("EXONIT_DONE_STATE", "1")
        db = ExoNitDB(str(tmp_path / "test.db"))
        # Sanity: without the injected failure, this sector/tic combo would
        # otherwise succeed.
        import exonit.routes.workers as workers_mod

        def _boom(*a, **k):
            raise RuntimeError("tic_sector_state missing (simulated)")

        monkeypatch.setattr(workers_mod.done_state, "blocked_for_enqueue",
                           _boom)
        r = client.post(
            "/api/worker/enqueue_bulk",
            json={"targets": [{"tic_id": "910001"}], "sector": 20},
            headers=AUTH)
        assert r.status_code == 503
        assert r.json()["error"] == "done_state_unavailable"
        # Fail CLOSED: nothing inserted.
        assert db._query_scalar(
            "SELECT COUNT(*) FROM tasks WHERE tic_id = ?", ("910001",)) == 0

    def test_flag_off_blocked_for_enqueue_never_called(
            self, client, tmp_path, monkeypatch):
        """With the flag off, blocked_for_enqueue must not even be
        consulted — a raising stub must have zero effect."""
        monkeypatch.delenv("EXONIT_DONE_STATE", raising=False)
        db = ExoNitDB(str(tmp_path / "test.db"))
        import exonit.routes.workers as workers_mod

        def _boom(*a, **k):
            raise RuntimeError("must never be called with the flag off")

        monkeypatch.setattr(workers_mod.done_state, "blocked_for_enqueue",
                           _boom)
        r = client.post(
            "/api/worker/enqueue_bulk",
            json={"targets": [{"tic_id": "910002"}], "sector": 20},
            headers=AUTH)
        assert r.status_code == 200
        assert r.json()["enqueued"] == 1
