"""Real-Postgres tests for the column-existence guard (review fix,
2026-09-26): docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md's own
§6 deploy-window table schedules S4's code (which unconditionally mints and
forwards ``run_id``/``sector``) BEFORE S3's ``tasks.run_id`` DDL. Without a
guard, the first real lot after an S4-only deploy would fail every INSERT on
a real PostgreSQL backend the moment ``discovery_run.py`` starts sending
``run_id`` — the SQLite-backed suite never caught this because
``ExoNitDB.__init__`` self-migrates the column on every process start
(exonit/db.py), so only the PG path was ever exposed.

Runs ONLY against the throwaway ``exonit_test`` DB (never ``exonit``/
``exonit_meta`` — see tests/_pg_test_db.py's hard guard), inside a FRESH,
uniquely-named schema PER TEST (created and dropped by the fixture below) —
never touches the real (shared) ``public.tasks`` table, so it is safe to run
even while other tests/agents are using ``exonit_test`` concurrently.

Skips cleanly if ``exonit_test`` is unreachable from this host (see
``tests/_pg_test_db.py::pg_test_available``).
"""
import uuid

import pytest

from tests._pg_test_db import pg_test_available, resolve_test_dsn

_BASE_DSN = resolve_test_dsn()


@pytest.fixture
def pg_schema():
    """A backend whose unqualified ``tasks``/``discovery_runs`` references
    resolve ONLY inside a fresh, per-test schema (``SET search_path``,
    NOT ``public``), so ``to_regclass('tasks')`` can never accidentally see
    the real (shared) ``public.tasks`` some OTHER test/agent has loaded into
    this same throwaway ``exonit_test`` DB.

    Deliberately does NOT pass ``search_path`` via the DSN's ``?options=``
    query string (the established pattern in tests/test_db_backend.py):
    ``exonit.db_backend._pg_connect_kwargs()`` unconditionally sets its own
    ``options="-c statement_timeout=..."`` connect kwarg, and psycopg gives
    an explicit keyword argument precedence over the same libpq parameter
    embedded in the DSN string — so that query-string trick is silently
    dropped by ``PgBackend.__init__`` today (discovered while writing this
    fixture: the ``?options=`` DSN alone resolved to the default
    ``public`` schema and collided with the real ``public.tasks``). Instead:
    a dedicated ``PgBackend`` with ``max_size=1`` (a single physical
    connection, guaranteed never handed off mid-test) + an explicit
    ``SET search_path TO <schema>`` issued once right after construction —
    session state (search_path) persists across that one connection's
    checkouts for the rest of this fixture's lifetime.
    """
    if not pg_test_available(_BASE_DSN):
        pytest.skip("EXONIT_TEST_DATABASE_URL (throwaway PG) unset or not reachable")
    from exonit.db_backend import PgBackend, make_backend

    schema = f"col_guard_{uuid.uuid4().hex[:12]}"
    admin_backend = make_backend(_BASE_DSN)
    admin_backend.execute(f"CREATE SCHEMA {schema}")
    backend = PgBackend(_BASE_DSN, min_size=1, max_size=1)
    try:
        backend.execute(f"SET search_path TO {schema}")
        backend.execute(
            "CREATE TABLE tasks (task_id TEXT PRIMARY KEY, tic_id TEXT, "
            "status TEXT NOT NULL DEFAULT 'pending', "
            "priority BIGINT NOT NULL DEFAULT 2, "
            "attempts BIGINT NOT NULL DEFAULT 0, sector BIGINT)"
        )
        backend.execute(
            "CREATE TABLE discovery_runs (run_id TEXT PRIMARY KEY, "
            "state TEXT NOT NULL DEFAULT 'ENQUEUEING')"
        )
        yield backend
    finally:
        backend.close()
        admin_backend.execute(f"DROP SCHEMA {schema} CASCADE")
        admin_backend.close()


class TestColumnExistsRealPg:
    def test_column_exists_false_before_ddl(self, pg_schema):
        from exonit.db_backend import column_exists
        assert column_exists(pg_schema, "tasks", "run_id") is False
        # sector WAS created above -- confirms this isn't just "always False".
        assert column_exists(pg_schema, "tasks", "sector") is True

    def test_column_exists_true_after_ddl_once_invalidated(self, pg_schema):
        from exonit.db_backend import column_exists, invalidate_column_exists
        assert column_exists(pg_schema, "tasks", "run_id") is False

        pg_schema.execute("ALTER TABLE tasks ADD COLUMN run_id TEXT")
        invalidate_column_exists(pg_schema, "tasks", "run_id")

        assert column_exists(pg_schema, "tasks", "run_id") is True

    def test_dropped_column_reported_absent_via_to_regclass(self, pg_schema):
        """attisdropped filtering: PG keeps a dropped column's catalog
        entry around (marked dropped) rather than removing it outright —
        column_exists must not be fooled by that."""
        from exonit.db_backend import column_exists, invalidate_column_exists
        pg_schema.execute("ALTER TABLE tasks ADD COLUMN run_id TEXT")
        invalidate_column_exists(pg_schema, "tasks", "run_id")
        assert column_exists(pg_schema, "tasks", "run_id") is True

        pg_schema.execute("ALTER TABLE tasks DROP COLUMN run_id")
        invalidate_column_exists(pg_schema, "tasks", "run_id")
        assert column_exists(pg_schema, "tasks", "run_id") is False


class TestInsertTasksBulkPgColumnGuard:
    """The exact code path the review finding is about."""

    def test_insert_succeeds_without_run_id_column(self, pg_schema):
        from exonit.routes.workers import _insert_tasks_bulk_pg

        task_rows = [("t1", "600001", 2)]
        _insert_tasks_bulk_pg(pg_schema, task_rows, sector=6, run_id="run_x")

        row = pg_schema.query_one(
            "SELECT tic_id, sector FROM tasks WHERE tic_id = ?", ("600001",))
        assert row is not None
        assert row["sector"] == 6
        # No run_id column exists in this schema -- nothing to assert it
        # was written, but the INSERT above must not have raised.

    def test_insert_writes_run_id_once_column_present(self, pg_schema):
        from exonit.db_backend import invalidate_column_exists
        from exonit.routes.workers import _insert_tasks_bulk_pg

        pg_schema.execute("ALTER TABLE tasks ADD COLUMN run_id TEXT")
        invalidate_column_exists(pg_schema, "tasks", "run_id")

        task_rows = [("t2", "600002", 2)]
        _insert_tasks_bulk_pg(pg_schema, task_rows, sector=6, run_id="run_y")

        row = pg_schema.query_one(
            "SELECT tic_id, run_id FROM tasks WHERE tic_id = ?", ("600002",))
        assert row["run_id"] == "run_y"

    def test_insert_without_sector_column_either(self, pg_schema):
        """Symmetry check: sector gets the same guard (review's 'likewise
        tasks.sector if not already guarded')."""
        from exonit.db_backend import invalidate_column_exists
        from exonit.routes.workers import _insert_tasks_bulk_pg

        pg_schema.execute("ALTER TABLE tasks DROP COLUMN sector")
        invalidate_column_exists(pg_schema, "tasks", "sector")

        task_rows = [("t3", "600003", 2)]
        _insert_tasks_bulk_pg(pg_schema, task_rows, sector=6, run_id=None)

        row = pg_schema.query_one(
            "SELECT tic_id FROM tasks WHERE tic_id = ?", ("600003",))
        assert row is not None  # did not raise; sector silently dropped


class TestAdoptLiveTasksColumnGuard:
    def test_returns_sentinel_without_run_id_column(self, pg_schema):
        from exonit.routes.workers import _adopt_live_tasks

        pg_schema.execute(
            "INSERT INTO tasks (task_id, tic_id, status) VALUES "
            "('t4', '600004', 'pending')")
        result = _adopt_live_tasks(pg_schema, ["600004"], "run_new")
        assert result == -1  # sentinel: caller keeps its TIC-count estimate

    def test_retags_successfully_once_column_present(self, pg_schema):
        from exonit.db_backend import invalidate_column_exists
        from exonit.routes.workers import _adopt_live_tasks

        pg_schema.execute("ALTER TABLE tasks ADD COLUMN run_id TEXT")
        invalidate_column_exists(pg_schema, "tasks", "run_id")
        pg_schema.execute(
            "INSERT INTO tasks (task_id, tic_id, status) VALUES "
            "('t5', '600005', 'pending')")

        result = _adopt_live_tasks(pg_schema, ["600005"], "run_new")

        assert result == 1
        row = pg_schema.query_one(
            "SELECT run_id FROM tasks WHERE tic_id = ?", ("600005",))
        assert row["run_id"] == "run_new"
