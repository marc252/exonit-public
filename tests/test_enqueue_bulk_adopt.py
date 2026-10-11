"""Tests for S4 — re-run idempotent after a timeout (lot adoption).

docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md §2. Mirrors
tests/test_done_state.py's fixtures/patterns (SQLite tmp_path DB + FastAPI
TestClient). Never touches the live DB.
"""
import pytest

from exonit.db import ExoNitDB

AUTH = {"Authorization": "Bearer exonit-default-token",
        "x-worker-id": "test-worker"}


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


def _enqueue_bulk(client, tics, *, sector=1, force_rescan=False, run_id=None):
    body = {"targets": [{"tic_id": t} for t in tics], "sector": sector}
    if force_rescan:
        body["force_rescan"] = True
    if run_id is not None:
        body["run_id"] = run_id
    return client.post("/api/worker/enqueue_bulk", json=body, headers=AUTH)


def _task_run_id(db, tic_id):
    row = db._query_one(
        "SELECT run_id FROM tasks WHERE tic_id = ? AND status IN "
        "('pending','assigned')", (tic_id,))
    return row["run_id"] if row else None


class TestAdoptedCounting:
    """adopted is only ever non-zero when the dedup used the live-rows
    predicate (force_rescan or the done-state dedup) — §2.1."""

    def test_no_live_rows_predicate_adopted_always_zero(self, client, db):
        """Default dedup ('any row blocks'): a pre-existing task blocks the
        TIC (skipped), but we cannot tell "blocked by a live task" from
        "blocked by a completed one" without the live-rows predicate — so
        adopted stays 0 even though the task IS live."""
        db.add_task("t1", "100", sector=1)  # pending
        r = _enqueue_bulk(client, ["100"], sector=1, run_id="run_new")
        assert r.status_code == 200
        data = r.json()
        assert data["enqueued"] == 0 and data["skipped"] == 1
        assert data["adopted"] == 0

    def test_force_rescan_counts_adopted_without_run_id(self, client, db):
        """(d) 'no run_id column' (the caller never passes one): adopted
        is still the TIC-count estimate — no UPDATE is even attempted."""
        db.add_task("t1", "101", sector=1)  # pending, no run_id
        r = _enqueue_bulk(client, ["101"], sector=1, force_rescan=True)
        assert r.status_code == 200
        data = r.json()
        assert data["enqueued"] == 0 and data["skipped"] == 1
        assert data["adopted"] == 1
        # No run_id was given -> no UPDATE ever attempted, task untouched.
        assert _task_run_id(db, "101") is None

    def test_done_state_dedup_counts_adopted(self, client, db, monkeypatch):
        """The realistic S4 trigger: EXONIT_DONE_STATE=1, sector given, NOT
        force_rescan (the same predicate a normal re-run of a lot uses)."""
        monkeypatch.setenv("EXONIT_DONE_STATE", "1")
        db.add_task("t1", "102", sector=1)  # pending
        r = _enqueue_bulk(client, ["102", "103"], sector=1)
        assert r.status_code == 200
        data = r.json()
        # 102 blocked by its live task (adopted-candidate); 103 is new.
        assert data["enqueued"] == 1 and data["skipped"] == 1
        assert data["adopted"] == 1


class TestAdoptionUpdate:
    """EXONIT_LOT_ADOPT_ACTIVE=1 + run_id: live tasks are re-tagged, and
    'adopted' becomes the real UPDATE rowcount (§2.1)."""

    def test_flag_off_no_update_even_with_run_id(self, client, db,
                                                  monkeypatch):
        """(e) flag OFF: response identical to today except the extra
        'adopted' field — no UPDATE ever runs."""
        monkeypatch.delenv("EXONIT_LOT_ADOPT_ACTIVE", raising=False)
        db.add_task("t1", "200", sector=1)  # pending, no run_id
        r = _enqueue_bulk(client, ["200"], sector=1, force_rescan=True,
                          run_id="run_new")
        data = r.json()
        assert data["skipped"] == 1
        assert data["adopted"] == 1  # still computed/returned (harmless)
        assert _task_run_id(db, "200") is None  # but NOT re-tagged

    def test_flag_on_retags_orphaned_task_no_discovery_runs_row(
            self, client, db, monkeypatch):
        """(c) the timeout case: the dead run's task has run_id set (by a
        PRIOR enqueue_bulk call for the same lot) but NO discovery_runs row
        was ever created for it (the endpoint creates that row AFTER the
        subprocess returns — S3 §3.1 evidence) -> adoptable."""
        monkeypatch.setenv("EXONIT_LOT_ADOPT_ACTIVE", "1")
        db.add_task("t1", "300", sector=1)
        db._exec("UPDATE tasks SET run_id = ? WHERE task_id = ?",
                 ("run_dead", "t1"))
        r = _enqueue_bulk(client, ["300"], sector=1, force_rescan=True,
                          run_id="run_new")
        data = r.json()
        assert data["skipped"] == 1
        assert data["adopted"] == 1
        assert _task_run_id(db, "300") == "run_new"

    def test_flag_on_retags_task_from_run_marked_failed(
            self, client, db, monkeypatch):
        """(c) alternate shape: a discovery_runs row DOES exist for the old
        run, but its state is terminal (FAILED) -> still adoptable."""
        monkeypatch.setenv("EXONIT_LOT_ADOPT_ACTIVE", "1")
        db.create_discovery_run("run_dead2", sector=1, lot_csv="x.csv",
                                dry_run=False, chained=False,
                                server_url="http://x")
        db.update_discovery_run("run_dead2", state="FAILED")
        db.add_task("t1", "301", sector=1)
        db._exec("UPDATE tasks SET run_id = ? WHERE task_id = ?",
                 ("run_dead2", "t1"))
        r = _enqueue_bulk(client, ["301"], sector=1, force_rescan=True,
                          run_id="run_new")
        data = r.json()
        assert data["adopted"] == 1
        assert _task_run_id(db, "301") == "run_new"

    def test_flag_on_does_not_steal_task_from_a_live_foreign_run(
            self, client, db, monkeypatch):
        """(b) a task owned by a run that is still ACTIVE (state not in
        FAILED/DONE) must NOT be stolen — 'skipped_active_foreign'."""
        monkeypatch.setenv("EXONIT_LOT_ADOPT_ACTIVE", "1")
        db.create_discovery_run("run_live", sector=1, lot_csv="x.csv",
                                dry_run=False, chained=False,
                                server_url="http://x")
        db.update_discovery_run("run_live", state="SCANNING")
        db.add_task("t1", "302", sector=1)
        db._exec("UPDATE tasks SET run_id = ? WHERE task_id = ?",
                 ("run_live", "t1"))
        r = _enqueue_bulk(client, ["302"], sector=1, force_rescan=True,
                          run_id="run_new")
        data = r.json()
        assert data["skipped"] == 1
        assert data["adopted"] == 0
        assert _task_run_id(db, "302") == "run_live"  # untouched

    def test_flag_on_retagging_same_run_id_is_idempotent(
            self, client, db, monkeypatch):
        """A Q5 chunk retry within the SAME lot (run_id already == run_id):
        the UPDATE still matches (harmless 0-effective-change) and still
        counts toward 'adopted', so the caller's total stays correct."""
        monkeypatch.setenv("EXONIT_LOT_ADOPT_ACTIVE", "1")
        db.add_task("t1", "303", sector=1)
        db._exec("UPDATE tasks SET run_id = ? WHERE task_id = ?",
                 ("run_new", "t1"))
        r = _enqueue_bulk(client, ["303"], sector=1, force_rescan=True,
                          run_id="run_new")
        data = r.json()
        assert data["adopted"] == 1
        assert _task_run_id(db, "303") == "run_new"


class TestNoTicLost:
    """discovery_run.py's own 'no TIC lost' rule is a separate unit
    (tests/test_discovery_run.py) — these confirm the SERVER-side counting
    this rule relies on is correct when a real insert failure happens."""

    def test_insert_failure_still_reports_adopted_for_the_live_ones(
            self, client, db, monkeypatch):
        monkeypatch.setenv("EXONIT_LOT_ADOPT_ACTIVE", "1")
        db.add_task("t1", "400", sector=1)  # live -> adoptable
        r = _enqueue_bulk(client, ["400", "401"], sector=1,
                          force_rescan=True, run_id="run_new")
        data = r.json()
        assert data["adopted"] == 1       # 400
        assert data["enqueued"] == 1      # 401 (new)
        assert data["failed"] == 0


def _drop_column(db, table, column):
    """Simulate a code-before-DDL deploy: the column exists on a fresh
    ExoNitDB (SQLite auto-migrates it) — drop it back out to reproduce the
    state a not-yet-migrated production Postgres would be in. Also drops
    any partial index on that column (SQLite refuses DROP COLUMN
    otherwise, same as tests/test_check_run_scope_parity.py's pattern)."""
    db.conn.execute(f"DROP INDEX IF EXISTS idx_tasks_{column}")
    db.conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
    db.conn.commit()
    from exonit.db_backend import invalidate_column_exists
    invalidate_column_exists(db.backend, table, column)


class TestColumnGuardDeployOrderSafety:
    """Review fix (2026-09-26): discovery_run.py (S4) unconditionally mints
    and forwards run_id/sector on every real lot — the code-before-DDL
    deploy order the spec's own §6 window table schedules (S4 code at W1,
    S3's tasks.run_id DDL at W2) must not turn every INSERT in the first
    real lot into a failure. See exonit.db_backend.column_exists /
    exonit.routes.workers._insert_tasks_bulk_sqlite /
    exonit.routes.workers._adopt_live_tasks."""

    def test_insert_succeeds_without_run_id_column(self, client, db):
        _drop_column(db, "tasks", "run_id")

        r = _enqueue_bulk(client, ["500"], sector=1, run_id="run_missing_col")

        assert r.status_code == 200
        data = r.json()
        assert data["enqueued"] == 1
        assert data["failed"] == 0
        # The row exists (the column-guard skipped ONLY the run_id column,
        # not the whole insert) — no run_id column to check, but the task
        # itself must be there.
        row = db._query_one("SELECT tic_id, sector FROM tasks WHERE tic_id = ?",
                            ("500",))
        assert row is not None
        assert row["sector"] == 1

    def test_insert_succeeds_without_sector_column_either(self, client, db):
        _drop_column(db, "tasks", "sector")

        r = _enqueue_bulk(client, ["501"], sector=1, run_id="run_x")

        assert r.status_code == 200
        data = r.json()
        assert data["enqueued"] == 1
        row = db._query_one("SELECT tic_id, run_id FROM tasks WHERE tic_id = ?",
                            ("501",))
        assert row is not None
        assert row["run_id"] == "run_x"  # run_id column IS present, still used

    def test_run_id_is_written_once_column_exists(self, client, db):
        """Sanity/contrast case: with the column present (the normal,
        post-DDL state), run_id IS written — the guard only skips it when
        genuinely absent, never unconditionally."""
        r = _enqueue_bulk(client, ["502"], sector=1, run_id="run_present")
        assert r.status_code == 200
        row = db._query_one("SELECT run_id FROM tasks WHERE tic_id = ?",
                            ("502",))
        assert row["run_id"] == "run_present"

    def test_adoption_update_degrades_to_count_only_without_run_id_column(
            self, client, db, monkeypatch):
        """Spec §2.1 'No run_id column: adopted = |live TICs| (only
        count)' — the S4 adoption UPDATE must skip cleanly (not crash,
        not fail the request) and the caller still gets a usable count."""
        monkeypatch.setenv("EXONIT_LOT_ADOPT_ACTIVE", "1")
        db.add_task("t1", "503", sector=1)  # a live task, pre-existing
        _drop_column(db, "tasks", "run_id")

        r = _enqueue_bulk(client, ["503", "504"], sector=1,
                          force_rescan=True, run_id="run_new")

        assert r.status_code == 200
        data = r.json()
        assert data["adopted"] == 1     # count-only estimate for 503
        assert data["enqueued"] == 1    # 504 is new

    def test_column_reappearing_mid_process_is_picked_up_without_restart(
            self, client, db):
        """invalidate_column_exists (used internally by _drop_column here,
        and available to callers after DDL is applied live) makes the next
        call re-probe immediately rather than waiting out the 5-minute TTL."""
        from exonit.db_backend import column_exists, invalidate_column_exists

        _drop_column(db, "tasks", "run_id")
        assert column_exists(db.backend, "tasks", "run_id") is False

        # DDL applied live, mid-process (what W2 of the deploy window does).
        db.conn.execute("ALTER TABLE tasks ADD COLUMN run_id TEXT")
        db.conn.commit()
        invalidate_column_exists(db.backend, "tasks", "run_id")

        assert column_exists(db.backend, "tasks", "run_id") is True
        r = _enqueue_bulk(client, ["505"], sector=1, run_id="run_after_ddl")
        assert r.status_code == 200
        row = db._query_one("SELECT run_id FROM tasks WHERE tic_id = ?",
                            ("505",))
        assert row["run_id"] == "run_after_ddl"

    def test_missing_column_logged_once_not_per_call(self, client, db, caplog):
        import logging
        _drop_column(db, "tasks", "run_id")

        with caplog.at_level(logging.INFO, logger="exonit.db_backend"):
            _enqueue_bulk(client, ["506"], sector=1, run_id="run_a")
            _enqueue_bulk(client, ["507"], sector=1, run_id="run_b")

        missing_msgs = [r for r in caplog.records
                       if "not present yet" in r.getMessage()
                       and "run_id" in r.getMessage()]
        assert len(missing_msgs) == 1, (
            f"expected exactly one 'not present yet' log, got "
            f"{len(missing_msgs)}: {[m.getMessage() for m in missing_msgs]}")
