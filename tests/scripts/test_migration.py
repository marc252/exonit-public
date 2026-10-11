"""Tests for scripts/migration/migrate_sqlite_to_pg.py and schema_pg.sql.

Scope:
  - Unit tests for the type-coercion helpers (parse_dt, coerce_row, pg_columns).
  - Integration test: dry-run against a minimal in-memory SQLite with all
    operational tables; verifies counts, FK pre-check output, and no PG
    connection is needed.
  - Schema test: verifies that schema_pg.sql exists, is readable, and
    contains the expected CREATE TABLE statements for all operational tables.

These tests do NOT connect to PostgreSQL and do NOT mutate any file on disk.
"""

import importlib.util
import sqlite3
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ── load the migration module without running main() ─────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
MIGRATION_MODULE = ROOT / "scripts" / "migration" / "migrate_sqlite_to_pg.py"
SCHEMA_SQL = ROOT / "scripts" / "migration" / "schema_pg.sql"

spec = importlib.util.spec_from_file_location("migrate_sqlite_to_pg", MIGRATION_MODULE)
mig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mig)


# ─────────────────────────────────────────────────────────────────────────
# 1. parse_dt
# ─────────────────────────────────────────────────────────────────────────

class TestParseDt:
    def test_none_returns_none(self):
        assert mig.parse_dt(None) is None

    def test_empty_string_returns_none(self):
        assert mig.parse_dt("") is None
        assert mig.parse_dt("  ") is None

    def test_space_separator(self):
        result = mig.parse_dt("2026-05-26 20:47:19")
        assert isinstance(result, datetime)
        assert result.tzinfo == timezone.utc
        assert result.year == 2026
        assert result.month == 5
        assert result.day == 26
        assert result.hour == 20

    def test_T_separator_with_microseconds(self):
        result = mig.parse_dt("2026-05-26T22:53:18.856613")
        assert isinstance(result, datetime)
        assert result.microsecond == 856613
        assert result.tzinfo == timezone.utc

    def test_T_separator_without_microseconds(self):
        result = mig.parse_dt("2026-06-15T11:26:27")
        assert isinstance(result, datetime)
        assert result.hour == 11
        assert result.minute == 26

    def test_date_only(self):
        result = mig.parse_dt("2026-06-20")
        assert isinstance(result, datetime)
        assert result.year == 2026
        assert result.day == 20

    def test_unparseable_returns_none(self, capsys):
        result = mig.parse_dt("not-a-date")
        assert result is None
        captured = capsys.readouterr()
        assert "WARN" in captured.out

    # ---------------------------------------------------------------- #
    # Regression (2026-07-02 review, blocking item 1): db.py's
    # naive-localtime timestamp fix (progress/impl-tzfix-2026-07-02.md)
    # made utc_now_iso() write "...+00:00"-suffixed tz-aware strings for
    # stars.analyzed_at / tasks.assigned_at / tasks.completed_at (three of
    # the DATETIME_COLS this migration tool coerces). Before this parse_dt()
    # fix, none of the four strptime formats include a %z/offset, so every
    # such row would fall through to the [WARN]/None path and silently
    # migrate as NULL on a disaster-recovery re-run — a real data-quality
    # regression. datetime.fromisoformat() must be tried first and must
    # both (a) preserve a tz-aware value round-trip-correctly and (b)
    # still tag naive legacy values as UTC exactly as before.
    # ---------------------------------------------------------------- #

    def test_tz_aware_utc_suffix_parses_and_keeps_the_instant(self):
        """New-format row (post-fix utc_now_iso() output): "+00:00" suffix
        must parse (not fall through to NULL) and preserve the exact instant."""
        result = mig.parse_dt("2026-07-02T13:48:00.123456+00:00")
        assert isinstance(result, datetime)
        assert result.tzinfo is not None
        assert result.utcoffset().total_seconds() == 0
        assert result.hour == 13 and result.minute == 48

    def test_tz_aware_non_utc_offset_normalises_correctly(self):
        """A non-UTC offset (e.g. from a CEST-local export elsewhere) must be
        converted to the correct absolute instant, not misread as UTC."""
        result = mig.parse_dt("2026-07-02T15:48:00+02:00")
        assert isinstance(result, datetime)
        # 15:48 CEST (+02:00) == 13:48 UTC — the true instant.
        assert result.astimezone(timezone.utc).hour == 13
        assert result.astimezone(timezone.utc).minute == 48

    def test_naive_legacy_format_still_assumed_utc(self):
        """Old-format row (pre-fix datetime.now().isoformat(), no offset)
        must still be parsed via the fromisoformat() fast path and tagged
        UTC exactly like the historical strptime loop did — no behaviour
        change for legacy data."""
        result = mig.parse_dt("2026-05-26T22:53:18.856613")
        assert isinstance(result, datetime)
        assert result.tzinfo == timezone.utc
        assert result.hour == 22 and result.microsecond == 856613

    def test_garbage_still_falls_through_to_warn_none(self, capsys):
        """Garbage must still hit the [WARN]/None path, not raise — the
        fromisoformat() fast path must not turn a parse failure into an
        uncaught exception."""
        result = mig.parse_dt("garbage-not-a-timestamp")
        assert result is None
        captured = capsys.readouterr()
        assert "WARN" in captured.out


# ─────────────────────────────────────────────────────────────────────────
# 2. pg_columns
# ─────────────────────────────────────────────────────────────────────────

class TestPgColumns:
    def test_no_renames(self):
        cols = mig.pg_columns("stars", ["tic_id", "ra", "dec", "has_spoc"])
        assert cols == ["tic_id", "ra", "dec", "has_spoc"]

    def test_telemetry_rename(self):
        # worker_telemetry: "timestamp" → "ts"
        sqlite_cols = ["id", "worker_id", "timestamp", "status"]
        pg_cols = mig.pg_columns("worker_telemetry", sqlite_cols)
        assert pg_cols == ["id", "worker_id", "ts", "status"]

    def test_rename_does_not_affect_other_tables(self):
        cols = mig.pg_columns("stars", ["timestamp"])
        assert cols == ["timestamp"]   # stars has no rename


# ─────────────────────────────────────────────────────────────────────────
# 3. coerce_row
# ─────────────────────────────────────────────────────────────────────────

class TestCoerceRow:
    def test_bool_coercion_candidates(self):
        # ctoi_submitted is a BOOL_COL for candidates
        cols = ["id", "tic_id", "candidate_num", "ctoi_submitted", "cross_source_agreement"]
        row = (1, "12345", 1, 1, 0)
        result = mig.coerce_row("candidates", cols, row)
        assert result[3] is True    # ctoi_submitted = 1 → True
        assert result[4] is False   # cross_source_agreement = 0 → False

    def test_bool_null_stays_none(self):
        cols = ["id", "tic_id", "ctoi_submitted"]
        row = (1, "12345", None)
        result = mig.coerce_row("candidates", cols, row)
        assert result[2] is None

    def test_datetime_coercion_stars(self):
        cols = ["tic_id", "analyzed_at"]
        row = ("99999", "2026-05-26 20:47:19")
        result = mig.coerce_row("stars", cols, row)
        assert isinstance(result[1], datetime)
        assert result[1].tzinfo == timezone.utc

    def test_non_coerced_columns_pass_through(self):
        cols = ["tic_id", "ra", "dec", "tmag"]
        row = ("99999", 123.45, -30.1, 10.5)
        result = mig.coerce_row("stars", cols, row)
        assert result[1] == 123.45
        assert result[2] == -30.1

    def test_tasks_bool_columns(self):
        cols = ["task_id", "tic_id", "gpu_used", "lc_load_recovered"]
        row = ("abc123", "99999", 0, 1)
        result = mig.coerce_row("tasks", cols, row)
        assert result[2] is False
        assert result[3] is True

    def test_has_spoc_bool(self):
        cols = ["tic_id", "has_spoc"]
        result = mig.coerce_row("stars", cols, ("99999", 1))
        assert result[1] is True
        result2 = mig.coerce_row("stars", cols, ("99999", 0))
        assert result2[1] is False


# ─────────────────────────────────────────────────────────────────────────
# 4. dry_run integration (in-memory SQLite fixture)
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def minimal_sqlite(tmp_path) -> Path:
    """Create a minimal SQLite database with all operational tables and a
    small number of rows to exercise the dry_run() function."""
    db_path = tmp_path / "test_exonit.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(textwrap.dedent("""
        CREATE TABLE stars (
            tic_id TEXT PRIMARY KEY, ra REAL, dec REAL, tmag REAL,
            analyzed_at TEXT, n_sectors INTEGER, n_datapoints INTEGER,
            time_span_days REAL, status TEXT DEFAULT 'completed',
            sectors_observed TEXT, has_spoc INTEGER
        );
        CREATE TABLE candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tic_id TEXT, candidate_num INTEGER,
            period_days REAL, t0 REAL, duration_days REAL,
            depth_ppm REAL, snr REAL, cnn_score REAL,
            triceratops_fpp REAL, claude_verdict TEXT, claude_analysis TEXT,
            ctoi_submitted INTEGER DEFAULT 0, exominer_score REAL,
            exominer_tce_period REAL, exominer_sector_run TEXT,
            triceratops_nfpp REAL, triceratops_prob_planet REAL,
            triceratops_verdict TEXT, exominer_ffi_score REAL,
            exominer_ffi_model_version TEXT, cross_source_name TEXT,
            cross_source_snr REAL, cross_source_depth_ppm REAL,
            cross_source_depth_ratio REAL, cross_source_agreement INTEGER,
            sector INTEGER DEFAULT 1
        );
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY, tic_id TEXT,
            status TEXT DEFAULT 'pending', priority INTEGER DEFAULT 2,
            assigned_to TEXT, assigned_at TEXT, completed_at TEXT,
            processing_time REAL, gpu_used INTEGER DEFAULT 0,
            attempts INTEGER DEFAULT 0, last_error TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            lc_load_recovered INTEGER DEFAULT 0
        );
        CREATE TABLE workers_registry (
            worker_id TEXT PRIMARY KEY, name TEXT UNIQUE NOT NULL,
            first_seen TEXT, last_seen TEXT,
            tasks_completed INTEGER DEFAULT 0, tasks_failed INTEGER DEFAULT 0,
            total_processing_time REAL DEFAULT 0
        );
        CREATE TABLE worker_telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id TEXT NOT NULL, timestamp TEXT NOT NULL,
            status TEXT, current_tic INTEGER, phase TEXT,
            phase_progress REAL, star_elapsed_s REAL, stars_done INTEGER,
            avg_time_per_star REAL, last_error TEXT, cpu_percent REAL,
            ram_used_gb REAL, ram_total_gb REAL, cpu_temp_c REAL,
            pid_rss_mb REAL, recent_timings TEXT, swap_used_gb REAL
        );
        CREATE TABLE worker_debug_mode (
            worker_id TEXT PRIMARY KEY, debug_enabled INTEGER DEFAULT 0,
            enabled_at TEXT, enabled_by TEXT
        );
        CREATE TABLE completeness_agg (
            radius_earth REAL NOT NULL, period_days REAL NOT NULL,
            sum_prob REAL NOT NULL, n INTEGER NOT NULL,
            PRIMARY KEY (radius_earth, period_days)
        );
        CREATE TABLE metadata (
            tic_id TEXT PRIMARY KEY, sectors TEXT NOT NULL,
            exptime REAL NOT NULL, fits_files TEXT NOT NULL,
            stellar_radius REAL, teff REAL, logg REAL, tmag REAL,
            ra REAL, dec REAL, updated_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE fits_paths (
            filename TEXT PRIMARY KEY, full_path TEXT NOT NULL
        );
        CREATE TABLE nearby_stars (
            tic_id TEXT NOT NULL, neighbor_id TEXT NOT NULL,
            ra REAL, dec REAL, mass REAL, rad REAL, teff REAL,
            tmag REAL, plx REAL, gal_b REAL,
            updated_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (tic_id, neighbor_id)
        );

        -- Seed rows
        INSERT INTO stars VALUES
            ('111111','10.0','-20.0','9.5','2026-05-26 20:47:19',2,50000,27.3,'completed','1,2',1),
            ('222222','11.0','-21.0','10.0','2026-05-26T22:53:24.506432',1,25000,13.7,'completed','1',0);

        INSERT INTO workers_registry VALUES
            ('wk001','worker-test','2026-05-26 19:54:22','2026-06-03 13:42:39',10,1,45.5);

        INSERT INTO candidates
            (tic_id,candidate_num,period_days,t0,duration_days,depth_ppm,snr,
             cnn_score,ctoi_submitted,sector)
            VALUES
            ('111111',1,3.5,2458700.5,0.1,500.0,15.0,0.85,0,1),
            ('222222',1,7.2,2458701.0,0.08,300.0,11.0,0.75,1,1);

        INSERT INTO tasks VALUES
            ('task001','111111','completed',2,'wk001',
             '2026-05-26T22:53:18.856613','2026-05-26T22:53:24.510217',
             5.63,0,1,NULL,'2026-05-26 20:47:19',0);

        INSERT INTO worker_telemetry
            (worker_id,timestamp,status,current_tic,phase,cpu_percent)
            VALUES ('wk001','2026-06-15 11:26:27','working',111111,'bls',45.5);

        INSERT INTO completeness_agg VALUES (1.5,3.0,0.85,10),(2.0,7.0,0.72,8);

        INSERT INTO fits_paths VALUES
            ('tess_111111_s001.fits','/mnt/data/tess_111111_s001.fits');
    """))
    conn.close()
    return db_path


class TestDryRun:
    def test_dry_run_reports_counts(self, minimal_sqlite, capsys):
        mig.dry_run(minimal_sqlite)
        out = capsys.readouterr().out
        # Should report counts for every operational table
        assert "stars" in out
        assert "2" in out              # 2 star rows
        assert "candidates" in out
        assert "completeness_agg" in out
        assert "Dry-run complete" in out

    def test_dry_run_no_errors_on_clean_db(self, minimal_sqlite, capsys):
        mig.dry_run(minimal_sqlite)
        out = capsys.readouterr().out
        assert "ERROR" not in out
        assert "WARN" not in out       # no orphans in clean fixture

    def test_dry_run_detects_missing_file(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            mig.dry_run(tmp_path / "nonexistent.db")

    def test_dry_run_detects_task_orphans(self, minimal_sqlite, capsys):
        """Inject a task with a tic_id that has no star entry."""
        conn = sqlite3.connect(str(minimal_sqlite))
        conn.execute(
            "INSERT INTO tasks VALUES "
            "('taskX','999999','pending',2,NULL,NULL,NULL,NULL,0,0,NULL,datetime('now'),0)"
        )
        conn.commit()
        conn.close()

        mig.dry_run(minimal_sqlite)
        out = capsys.readouterr().out
        assert "tasks with no matching star" in out
        assert "WARN" in out


# ─────────────────────────────────────────────────────────────────────────
# 5. schema_pg.sql content checks
# ─────────────────────────────────────────────────────────────────────────

class TestSchemaSql:
    def test_schema_file_exists(self):
        assert SCHEMA_SQL.exists(), f"Expected {SCHEMA_SQL} to exist"

    def test_schema_is_readable(self):
        content = SCHEMA_SQL.read_text()
        assert len(content) > 100

    @pytest.mark.parametrize("table", [
        "stars", "candidates", "tasks", "workers_registry",
        "worker_telemetry", "worker_debug_mode", "completeness_agg",
        "metadata", "fits_paths", "nearby_stars",
    ])
    def test_schema_contains_table(self, table):
        content = SCHEMA_SQL.read_text()
        assert f"CREATE TABLE {table}" in content or \
               f"TABLE {table} " in content, \
               f"schema_pg.sql missing CREATE TABLE for '{table}'"

    def test_schema_has_partitioned_candidates(self):
        content = SCHEMA_SQL.read_text()
        assert "PARTITION BY LIST (sector)" in content
        assert "candidates_sector1" in content
        assert "candidates_sector2" in content
        assert "candidates_default" in content

    def test_schema_tasks_not_partitioned(self):
        # tasks must be unpartitioned so PRIMARY KEY (task_id) is a single-column
        # global constraint; partitioning forced (task_id, status) as PK, which
        # would allow duplicate task_ids across status values (Change 1, 2026-06-20).
        content = SCHEMA_SQL.read_text()
        # No partition child tables for tasks
        assert "CREATE TABLE tasks_active" not in content
        assert "PARTITION OF tasks" not in content
        # tasks table must NOT be declared with PARTITION BY
        # (check that the CREATE TABLE tasks block does not contain PARTITION BY)
        import re
        tasks_block = re.search(
            r"CREATE TABLE tasks\b.*?(?=CREATE TABLE|\Z)", content, re.DOTALL
        )
        assert tasks_block is not None, "CREATE TABLE tasks not found in schema"
        assert "PARTITION BY" not in tasks_block.group()
        # Single-column PK appears in the tasks table DDL
        assert "PRIMARY KEY (task_id)" in tasks_block.group()

    def test_schema_tasks_has_status_index(self):
        # The (status, priority) index replaces the partition for operational queries
        content = SCHEMA_SQL.read_text()
        assert "idx_tasks_status_priority" in content

    def test_schema_completeness_agg_trigger_gap_documented(self):
        # The trigger maintenance gap must be explicitly documented as a blocking
        # open decision (Change 2, 2026-06-20).
        content = SCHEMA_SQL.read_text()
        assert "TRIGGER MAINTENANCE GAP" in content
        assert "OPEN DECISION REQUIRED" in content
        assert "Option A" in content
        assert "Option B" in content

    def test_schema_candidates_tic_id_not_null_comment(self):
        # The intentional NOT NULL tightening on candidates.tic_id must be
        # documented in the schema (Change 3, 2026-06-20).
        content = SCHEMA_SQL.read_text()
        assert "intentional tightening" in content

    def test_schema_has_partitioned_telemetry(self):
        content = SCHEMA_SQL.read_text()
        assert "PARTITION BY RANGE (ts)" in content
        assert "worker_telemetry_2026" in content

    def test_schema_uses_timestamptz(self):
        content = SCHEMA_SQL.read_text()
        assert "TIMESTAMPTZ" in content

    def test_schema_uses_double_precision(self):
        content = SCHEMA_SQL.read_text()
        assert "DOUBLE PRECISION" in content

    def test_schema_uses_boolean(self):
        content = SCHEMA_SQL.read_text()
        assert "BOOLEAN" in content

    def test_schema_excludes_completeness_table(self):
        # The raw completeness table (Parquet path) must NOT be in the schema
        content = SCHEMA_SQL.read_text()
        lines = [l.strip() for l in content.splitlines()]
        create_lines = [l for l in lines if l.startswith("CREATE TABLE completeness")]
        # Only completeness_agg should exist, not the raw completeness table
        assert all("_agg" in l or "completeness_agg" in l
                   for l in create_lines), \
               f"Raw 'completeness' table found in schema: {create_lines}"

    def test_schema_has_sequences(self):
        content = SCHEMA_SQL.read_text()
        assert "candidates_id_seq" in content
        assert "worker_telemetry_id_seq" in content

    def test_schema_grants_to_exonit_role(self):
        content = SCHEMA_SQL.read_text()
        assert "GRANT" in content
        assert "exonit" in content

    def test_schema_has_partial_indexes(self):
        content = SCHEMA_SQL.read_text()
        assert "idx_cand_fpp_todo" in content
        assert "idx_cand_strong" in content
        assert "WHERE" in content   # partial index marker


# ─────────────────────────────────────────────────────────────────────────
# 6. TABLES_ORDERED completeness
# ─────────────────────────────────────────────────────────────────────────

class TestTablesOrdered:
    def test_all_operational_tables_present(self):
        expected = {
            "stars", "candidates", "tasks", "workers_registry",
            "worker_telemetry", "worker_debug_mode", "completeness_agg",
            "metadata", "fits_paths", "nearby_stars",
        }
        actual = set(mig.TABLES_ORDERED)
        assert expected == actual

    def test_stars_before_candidates(self):
        order = mig.TABLES_ORDERED
        assert order.index("stars") < order.index("candidates")

    def test_workers_registry_before_telemetry(self):
        order = mig.TABLES_ORDERED
        assert order.index("workers_registry") < order.index("worker_telemetry")
