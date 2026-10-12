"""Database for tracking analyzed stars and candidates.

ExoNitDB talks to the DB through the backend abstraction in
``exonit/db_backend.py``: SQLite by default (bit-identical to the historical
single-connection behaviour) or PostgreSQL when given a ``postgresql://`` DSN
(via ``EXONIT_DATABASE_URL`` / the path argument). See
docs/superpowers/specs/2026-06-24-postgres-backend-design.md.

The ``completeness`` / ``completeness_agg`` subsystem (+ its 3 triggers) always
stays on SQLite — even in PG mode — on a dedicated ``sqlite3`` connection
(``self._cmpl_conn``), because the raw completeness table is huge and isolated.

Phase 4a (2026-07-05, ``EXONIT_COMPLETENESS_PG=1``): that sidecar is being
retired in favour of a PG staging + per-bucket-aggregate write path (see
docs/superpowers/plans/2026-07-05-phase4-completeness-parquet.md). With the
flag ON *and* the main backend on PostgreSQL, ``self._cmpl_conn`` is never
opened at all — every completeness read/write goes through ``self.backend``
instead. With the flag OFF (the default) behaviour is unchanged.
"""

import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone

from exonit.completeness_store import completeness_pg_enabled
from exonit.db_backend import is_pg_url, make_backend, select_in_chunks

logger = logging.getLogger(__name__)

# Sentinel: "caller did not pass verdict_proba" (distinct from an explicit None).
_UNSET = object()

MAX_TASK_ATTEMPTS = 3
FPP_VALIDATED = 0.015
FPP_CANDIDATE = 0.5
NFPP_VALIDATED = 0.001


def fpp_verdict(fpp, nfpp=None):
    """Conservative TRICERATOPS verdict stored with server results."""
    if fpp is None:
        return None
    try:
        fpp = float(fpp)
    except (TypeError, ValueError):
        return None
    if fpp < 0:
        return "failed"
    if fpp >= FPP_CANDIDATE:
        return "likely_fp"
    if nfpp is not None:
        try:
            nfpp = float(nfpp)
        except (TypeError, ValueError):
            nfpp = None
    if fpp < FPP_VALIDATED and nfpp is not None and 0 <= nfpp < NFPP_VALIDATED:
        return "validated"
    return "candidate"


def derive_exominer_status(score, has_spoc=None):
    """Derive ``candidates.exominer_status`` from the current
    ``(exominer_score, stars.has_spoc)`` pair.

    Single source of truth for the Python-side write points in this module
    (docs/superpowers/specs/exominer-status-column.md — "Step 2: separate
    status from score"). ``scripts/migrate_exominer_status.py`` mirrors
    this exact rule in SQL for the bulk backfill (server-side, for
    performance at 13M+ rows) — see that script's ``_STATUS_CASE_SQL`` and
    the cross-check test that keeps the two in lockstep.

    Rule (corrected 2026-07-14 after auditing production: 51 candidates carry
    ``exominer_score = -5.960464477539063e-08``, a float32 rounding artifact
    of a genuine ~0.0 ExoMiner-SPOC probability, landing just below zero. A
    naive ``score >= 0`` check for "scored" misclassifies these as neither
    scored nor sentinel. The -1 sentinel is therefore the ONLY special-cased
    score value; every other non-NULL value — regardless of sign — is a real
    score):

        score IS NULL                    -> 'pending'
        score == -1 AND has_spoc == 1    -> 'no_dv'   (has SPOC, no DV report)
        score == -1 (has_spoc 0 or None) -> 'no_spoc'
        anything else non-NULL           -> 'scored'

    ``has_spoc`` is only consulted when score == -1 (the only ambiguous
    case); callers that don't know it (e.g. ``update_candidate_exominer``,
    which only ever receives a genuine ExoMiner probability in practice, never
    -1) may omit it and get the conservative 'no_spoc' default for that
    theoretical edge case.
    """
    if score is None:
        return "pending"
    if score == -1:
        return "no_dv" if has_spoc == 1 else "no_spoc"
    return "scored"


def _is_unique_violation(exc) -> bool:
    """True if *exc* is a unique/primary-key constraint violation.

    Backend-agnostic: matches SQLite's ``IntegrityError`` and psycopg's
    ``UniqueViolation`` (matched by class name to avoid importing psycopg in
    SQLite-only deployments).
    """
    if isinstance(exc, sqlite3.IntegrityError):
        return True
    # psycopg.errors.UniqueViolation — match by name without importing psycopg.
    for cls in type(exc).__mro__:
        if cls.__name__ in ("UniqueViolation", "IntegrityError"):
            # IntegrityError here would be psycopg's (sqlite's is caught above).
            return cls.__name__ == "UniqueViolation" or (
                cls.__module__ or "").startswith("psycopg")
    return False


# PG error codes a concurrent "CREATE ... IF NOT EXISTS" race can raise (Phase
# 4a Task 4 follow-up e — see _ensure_completeness_pg_schema below).
_CONCURRENT_DDL_RACE_SQLSTATES = frozenset({
    "42P07",  # duplicate_table
    "42710",  # duplicate_object (e.g. a racing CREATE INDEX)
    "23505",  # unique_violation (a racing catalog row insert)
})


def _is_concurrent_ddl_race(exc) -> bool:
    """True if *exc* is a PG error from a concurrent CREATE-IF-NOT-EXISTS race.

    Matched by ``sqlstate`` — psycopg3 exposes it directly on every database
    error, and this helper is only ever called on the PG-main-backend path
    (where psycopg3 is already a hard dependency), so there is no need for
    the class-name-matching trick ``_is_unique_violation`` uses above to stay
    import-safe on SQLite-only deployments.
    """
    return getattr(exc, "sqlstate", None) in _CONCURRENT_DDL_RACE_SQLSTATES


def utc_now_iso() -> str:
    """Timezone-aware UTC "now" as an ISO-8601 string.

    Every Python-side DB timestamp write MUST go through this helper — never
    ``datetime.now().isoformat()``, which is NAIVE LOCAL TIME (Europe/Madrid,
    UTC+1/+2). On PostgreSQL (``timestamptz`` columns) a naive string is
    interpreted per the *session* timezone; when that session is UTC, the
    naive local wall-clock value gets stored as if it already were UTC — i.e.
    ~1-2h in the future. Diagnosed 2026-07-02: this silently defeated the
    orphan-task auto-release (``age = now() - assigned_at`` came out negative
    and never crossed the 1800s threshold). SQLite (TEXT columns) does not
    have this failure mode, but a single tz-aware writer keeps both backends'
    on-disk representation identical and makes the bug structurally
    impossible to reintroduce site-by-site. SQL-native ``datetime('now')`` /
    ``NOW()`` calls (translated per-backend by db_backend.py) are unaffected
    by this bug and do not need to route through here.
    """
    return datetime.now(timezone.utc).isoformat()


def _to_epoch(value):
    """Convert an ``assigned_at`` value to a POSIX timestamp, backend-agnostic.

    SQLite stores it as an ISO-8601 ``str``; PG returns a ``datetime``.  Returns
    None when the value cannot be parsed (mirrors the old ``except ValueError``).
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    try:
        return datetime.fromisoformat(value).timestamp()
    except (ValueError, TypeError):
        return None


def _tic_str(tic):
    """Normalise a TIC identifier to str for comparison, or None if absent.

    Heartbeat-delivered current_tic values can be int (JSON round-trip may
    drop the string type) while the DB always stores tic_id as TEXT.
    Normalising both sides to str prevents a false-mismatch that would
    reassign a task that is still being processed.
    """
    return str(tic) if tic is not None else None


class ExoNitDB:
    def __init__(self, db_path="data/exonit.db", completeness_path=None):
        self.backend = make_backend(db_path)
        self.dialect = self.backend.dialect
        self._is_pg = self.dialect.name == "postgres"
        # Phase 4a cutover flag: only meaningful combined with a PG main
        # backend — completeness-on-PG with a SQLite main backend has no
        # defined behaviour (the flag exists to retire the PG-mode sidecar).
        self._cmpl_pg = self._is_pg and completeness_pg_enabled()

        if self._cmpl_pg:
            # Phase 4a, flag ON: completeness moves to PG staging + per-bucket
            # aggregate tables under self.backend. No SQLite sidecar at all —
            # self._cmpl_conn/_cmpl_lock stay None/unused, self.conn is None
            # (nothing in the current codebase reaches into db.conn on the PG
            # path; see add_completeness_many/query_completeness_agg/
            # count_completeness_tics/wal_checkpoint below for the PG branch).
            self._cmpl_conn = None
            self._cmpl_lock = None
            self.conn = None
            self._ensure_completeness_pg_schema()
        elif self._is_pg:
            # Completeness stays on SQLite even in PG mode (hybrid backend,
            # flag OFF — the historical/default behaviour, unchanged).
            # If db_path is a pure DSN there is no SQLite file to reuse, so fall
            # back to the canonical production file (overridable for tests).
            cmpl = completeness_path or "data/exonit.db"
            self._cmpl_conn = sqlite3.connect(
                cmpl, check_same_thread=False, timeout=30.0)
            self._cmpl_conn.row_factory = sqlite3.Row
            self._cmpl_conn.execute("PRAGMA journal_mode=WAL")
            self._cmpl_conn.execute("PRAGMA busy_timeout=10000")
            self._cmpl_conn.execute("PRAGMA synchronous=NORMAL")
            self._cmpl_lock = threading.Lock()
            # In SQLite mode self.conn is the operational connection; in PG mode
            # there is no raw operational sqlite connection, but a handful of
            # legacy callers reach into db.conn for completeness/raw SQL — point
            # it at the completeness connection so those keep working.
            self.conn = self._cmpl_conn
        else:
            # SQLite mode: operational + completeness share one connection,
            # exactly as before.  self.conn / self._lock are kept as aliases of
            # the backend's own connection and lock so the historical behaviour
            # (and external code that touches db.conn) is bit-identical.
            self.conn = self.backend.conn
            self._lock = self.backend._lock
            self._cmpl_conn = self.conn
            self._cmpl_lock = self._lock

        self._create_tables()

    def _exec(self, sql, params=()):
        """Execute + commit via the backend."""
        self.backend.execute(sql, params)

    def _query(self, sql, params=()):
        """Query via the backend, returns list of dicts."""
        return self.backend.query(sql, params)

    def _query_one(self, sql, params=()):
        """Query via the backend, returns single dict or None."""
        return self.backend.query_one(sql, params)

    def _query_scalar(self, sql, params=()):
        """Query via the backend, returns single value."""
        return self.backend.query_scalar(sql, params)

    def _create_tables(self):
        # Completeness schema (+ triggers/backfill) lives on SQLite — even in
        # PG mode, on the dedicated self._cmpl_conn — UNLESS the Phase 4a flag
        # is on, in which case __init__ already called
        # _ensure_completeness_pg_schema() and there is no SQLite side at all.
        if not self._cmpl_pg:
            self._create_completeness_schema()
        if self._is_pg:
            # Operational tables live in PostgreSQL; the schema is created out
            # of band (schema_pg.sql on the <host>), so this is a no-op.
            return
        with self._lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS stars (
                    tic_id TEXT PRIMARY KEY,
                    ra REAL, dec REAL, tmag REAL,
                    analyzed_at TEXT,
                    n_sectors INTEGER, n_datapoints INTEGER,
                    time_span_days REAL,
                    status TEXT DEFAULT 'completed'
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tic_id TEXT REFERENCES stars(tic_id),
                    candidate_num INTEGER,
                    period_days REAL, t0 REAL,
                    duration_days REAL, depth_ppm REAL, snr REAL,
                    cnn_score REAL, triceratops_fpp REAL,
                    triceratops_nfpp REAL, triceratops_prob_planet REAL,
                    triceratops_verdict TEXT,
                    claude_verdict TEXT, claude_analysis TEXT,
                    ctoi_submitted INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    tic_id TEXT,
                    status TEXT DEFAULT 'pending',
                    priority INTEGER DEFAULT 2,
                    assigned_to TEXT,
                    assigned_at TEXT,
                    completed_at TEXT,
                    processing_time REAL,
                    gpu_used INTEGER DEFAULT 0,
                    attempts INTEGER DEFAULT 0,
                    last_error TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS worker_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    worker_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT,
                    current_tic INTEGER,
                    phase TEXT,
                    phase_progress REAL,
                    star_elapsed_s REAL,
                    stars_done INTEGER,
                    avg_time_per_star REAL,
                    last_error TEXT,
                    cpu_percent REAL,
                    ram_used_gb REAL,
                    ram_total_gb REAL,
                    cpu_temp_c REAL,
                    pid_rss_mb REAL,
                    recent_timings TEXT
                );
                CREATE TABLE IF NOT EXISTS worker_debug_mode (
                    worker_id TEXT PRIMARY KEY,
                    debug_enabled INTEGER DEFAULT 0,
                    enabled_at TEXT,
                    enabled_by TEXT
                );
                CREATE TABLE IF NOT EXISTS workers_registry (
                    worker_id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    first_seen TEXT,
                    last_seen TEXT
                );
                CREATE TABLE IF NOT EXISTS discovery_runs (
                    run_id       TEXT PRIMARY KEY,
                    sector       INTEGER NOT NULL,
                    lot_csv      TEXT NOT NULL,
                    state        TEXT NOT NULL DEFAULT 'ENQUEUEING',
                    detail       TEXT,
                    dry_run      INTEGER NOT NULL DEFAULT 0,
                    chained      INTEGER NOT NULL DEFAULT 0,
                    server_url   TEXT,
                    orchestrator_pid INTEGER,
                    checkpoints  TEXT NOT NULL DEFAULT '{}',
                    started_at   TEXT NOT NULL,
                    updated_at   TEXT NOT NULL,
                    heartbeat_at TEXT,
                    finished_at  TEXT
                );
                CREATE TABLE IF NOT EXISTS discovery_run_tics (
                    run_id TEXT NOT NULL,
                    tic_id TEXT NOT NULL,
                    PRIMARY KEY (run_id, tic_id)
                );
                CREATE INDEX IF NOT EXISTS idx_run_tics_tic
                    ON discovery_run_tics (tic_id);
            """)
            # Migrate: add sectors_observed column if missing
            try:
                self.conn.execute(
                    "ALTER TABLE stars ADD COLUMN sectors_observed TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists
            # Migrate: add priority column to tasks if missing
            try:
                self.conn.execute(
                    "ALTER TABLE tasks ADD COLUMN priority INTEGER DEFAULT 2")
            except sqlite3.OperationalError:
                pass  # column already exists
            # Migrate: candidates.sector (DEFAULT 1) + composite unique index.
            # Sector lets sector-2 (AWS) candidates merge into the home DB
            # without colliding with sector-1 rows. DEFAULT 1 backfills existing
            # rows and keeps home inserts (which omit sector) on sector 1, so the
            # uniqueness guarantee — now (sector, tic_id, candidate_num) — is
            # preserved. The same TIC may have candidates in two sectors.
            try:
                self.conn.execute(
                    "ALTER TABLE candidates ADD COLUMN sector INTEGER DEFAULT 1")
            except sqlite3.OperationalError:
                pass  # column already exists
            try:
                self.conn.execute("DROP INDEX IF EXISTS idx_candidates_unique")
                self.conn.execute(
                    """CREATE UNIQUE INDEX IF NOT EXISTS idx_candidates_unique
                       ON candidates(sector, tic_id, candidate_num)""")
            except sqlite3.OperationalError:
                pass  # index already exists or duplicates prevent creation
            # Migrate: add ram_total_gb column to worker_telemetry if missing
            try:
                self.conn.execute(
                    "ALTER TABLE worker_telemetry ADD COLUMN ram_total_gb REAL")
            except sqlite3.OperationalError:
                pass  # column already exists
            # Migrate: swap_used_gb on worker_telemetry (Phase 3 #8). Emitted on
            # every heartbeat (not just debug) so host swap pressure is visible
            # during a run — the signal that drives the worker's inline-injection
            # memory guard.
            try:
                self.conn.execute(
                    "ALTER TABLE worker_telemetry ADD COLUMN swap_used_gb REAL")
            except sqlite3.OperationalError:
                pass  # column already exists
            # Migrate: gpu telemetry columns (Status fleet map). Nullable REALs
            # for per-heartbeat GPU metrics — never raise if they already exist.
            try:
                self.conn.execute(
                    "ALTER TABLE worker_telemetry ADD COLUMN gpu_util_pct REAL")
            except sqlite3.OperationalError:
                pass  # column already exists
            try:
                self.conn.execute(
                    "ALTER TABLE worker_telemetry ADD COLUMN gpu_mem_used_gb REAL")
            except sqlite3.OperationalError:
                pass  # column already exists
            try:
                self.conn.execute(
                    "ALTER TABLE worker_telemetry ADD COLUMN gpu_mem_total_gb REAL")
            except sqlite3.OperationalError:
                pass  # column already exists
            try:
                self.conn.execute(
                    "ALTER TABLE worker_telemetry ADD COLUMN gpu_temp_c REAL")
            except sqlite3.OperationalError:
                pass  # column already exists
            try:
                self.conn.execute(
                    "ALTER TABLE worker_telemetry ADD COLUMN gpu_power_w REAL")
            except sqlite3.OperationalError:
                pass  # column already exists
            # Migrate: add attempts + last_error columns to tasks if missing
            try:
                self.conn.execute(
                    "ALTER TABLE tasks ADD COLUMN attempts INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                self.conn.execute(
                    "ALTER TABLE tasks ADD COLUMN last_error TEXT")
            except sqlite3.OperationalError:
                pass
            # Migrate: lc_load_recovered flag — set to 1 when SPOC LC load
            # failed but FFI fallback (worker.py::_try_ffi_fallback) succeeded
            # and the pipeline ran on FFI data instead. See
            # docs/superpowers/specs/2026-05-08-ffi-fallback-on-lc-load-failure-design.md
            try:
                self.conn.execute(
                    "ALTER TABLE tasks ADD COLUMN lc_load_recovered "
                    "INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            # Migrate: persistent worker counters so they survive server restarts
            for col, spec in [
                ("tasks_completed", "INTEGER DEFAULT 0"),
                ("tasks_failed", "INTEGER DEFAULT 0"),
                ("total_processing_time", "REAL DEFAULT 0"),
            ]:
                try:
                    self.conn.execute(
                        f"ALTER TABLE workers_registry ADD COLUMN {col} {spec}")
                except sqlite3.OperationalError:
                    pass
            # Migrate: TRICERATOPS companion columns. NFPP is required for
            # a scientifically valid Giacalone-style validation decision.
            for col, spec in [
                ("triceratops_nfpp", "REAL"),
                ("triceratops_prob_planet", "REAL"),
                ("triceratops_verdict", "TEXT"),
            ]:
                try:
                    self.conn.execute(
                        f"ALTER TABLE candidates ADD COLUMN {col} {spec}")
                except sqlite3.OperationalError:
                    pass
            # Existing FPP-only rows cannot be validated without NFPP. Keep a
            # conservative verdict so old low-FPP rows do not masquerade as
            # validated until workers recompute NFPP.
            self.conn.execute(
                """UPDATE candidates SET triceratops_verdict = CASE
                       WHEN triceratops_fpp < 0 THEN 'failed'
                       WHEN triceratops_fpp >= 0.5 THEN 'likely_fp'
                       WHEN triceratops_nfpp IS NOT NULL
                            AND triceratops_fpp >= 0
                            AND triceratops_fpp < 0.015
                            AND triceratops_nfpp >= 0
                            AND triceratops_nfpp < 0.001 THEN 'validated'
                       ELSE 'candidate'
                   END
                   WHERE triceratops_fpp IS NOT NULL
                     AND triceratops_verdict IS NULL""")
            # Migrate: ExoMiner columns on candidates (Phase 5 integration)
            for col, spec in [
                ("exominer_score", "REAL"),
                ("exominer_tce_period", "REAL"),
                ("exominer_sector_run", "TEXT"),
            ]:
                try:
                    self.conn.execute(
                        f"ALTER TABLE candidates ADD COLUMN {col} {spec}")
                except sqlite3.OperationalError:
                    pass

            # Migrate: tic_sector_state (done-state design,
            # docs/superpowers/specs/2026-09-26-tic-sector-done-state-design.md
            # §2.2) + tasks.sector. See exonit/done_state.py for the read-side
            # predicates (done/exhausted) this table backs; the write side
            # (mark_tic_done/mark_tic_failed) is called unconditionally from
            # routes/workers.py's worker_result, gated by nothing — only the
            # READERS are gated behind EXONIT_DONE_STATE so this dual-write is
            # harmless (and useful for the eventual cutover) while the flag
            # stays off.
            try:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS tic_sector_state (
                        tic_id       TEXT    NOT NULL,
                        sector       INTEGER NOT NULL,
                        status       TEXT    NOT NULL,
                        attempts     INTEGER NOT NULL DEFAULT 0,
                        last_task_id TEXT,
                        last_error   TEXT,
                        updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
                        PRIMARY KEY (tic_id, sector)
                    )""")
            except sqlite3.OperationalError:
                pass  # table already exists
            try:
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_tss_done "
                    "ON tic_sector_state (tic_id) WHERE status = 'completed'")
            except sqlite3.OperationalError:
                pass
            try:
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_tss_sector_failed "
                    "ON tic_sector_state (sector) WHERE status = 'failed'")
            except sqlite3.OperationalError:
                pass
            # tasks.sector: the LOT's sector at enqueue time (NULL = unknown /
            # enqueued outside a --sector-aware path); 0 is reserved for
            # legacy/backfilled rows. See done_state.py's LEGACY_SECTOR.
            try:
                self.conn.execute(
                    "ALTER TABLE tasks ADD COLUMN sector INTEGER")
            except sqlite3.OperationalError:
                pass  # column already exists

            # tasks.run_id (S3, docs/superpowers/specs/2026-09-26-enqueue-
            # s1-s4-design.md §3.2): the LOT's run identity at enqueue time
            # (NULL = enqueued outside a --run-id-aware path, e.g. a manual
            # enqueue_from_csv.py without --run-id — adoptable by S4, out of
            # scope for any run). Replaces discovery_run_tics as the run's
            # TIC-scope source, without dropping that table (operator
            # decision §9.2: drop only after >=2 lots with
            # EXONIT_RUN_SCOPE=tasks + EXONIT_RUN_TICS_WRITE=0 and an export).
            try:
                self.conn.execute(
                    "ALTER TABLE tasks ADD COLUMN run_id TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists
            try:
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_tasks_run_id "
                    "ON tasks (run_id, status, tic_id) "
                    "WHERE run_id IS NOT NULL")
            except sqlite3.OperationalError:
                pass

            try:
                self.conn.execute(
                    "ALTER TABLE candidates ADD COLUMN exominer_ffi_score REAL")
            except sqlite3.OperationalError:
                pass

            try:
                self.conn.execute(
                    "ALTER TABLE candidates ADD COLUMN exominer_ffi_model_version TEXT")
            except sqlite3.OperationalError:
                pass

            # Migrate: exominer_status (docs/superpowers/specs/exominer-
            # status-column.md — "Pas 2"). ADDITIVE, no DEFAULT: existing rows
            # stay NULL here (SQLite dev/test DBs are small — no batching
            # needed; the 13M-row production PG table is backfilled by
            # scripts/migrate_exominer_status.py instead, since PG schema
            # changes don't run through this SQLite-only auto-migration
            # block — see the `if self._is_pg: return` guard above). See
            # derive_exominer_status() for the derivation rule and the write
            # points (mark_ffi_only_candidates, mark_spoc_no_dv_candidates,
            # clear_spurious_minus_one_candidates, mark_unconfirmed_ffi_
            # candidates, update_candidate_exominer) that keep it in sync
            # with exominer_score going forward. exominer_score itself and
            # its -1 sentinel are UNCHANGED — this column only adds an
            # unambiguous state alongside it.
            try:
                self.conn.execute(
                    "ALTER TABLE candidates ADD COLUMN exominer_status TEXT")
            except sqlite3.OperationalError:
                pass

            try:
                # Shadow score: ExoMiner-FFI v8 ensemble scored alongside the
                # production v6 (hold-out validation; non-authoritative).
                # RETIRED 2026-06-27 (commit b60570bf; the v8 ONNX no longer
                # ships — hold-out FAILED, v6 stays production). The write
                # path was removed from add_candidate() in the 2026-07-02
                # cleanup (docs/reviews/2026-07-02-opus48-cleanup-review.md);
                # the column itself is kept (always NULL going forward) for
                # historical rows — no ALTER/DROP.
                self.conn.execute(
                    "ALTER TABLE candidates ADD COLUMN exominer_ffi_score_v8 REAL")
            except sqlite3.OperationalError:
                pass

            # Migrate: WHY a candidate was pre-marked fpp=-1 (2026-07-14).
            # -1 has always meant "TRICERATOPS was never run on this row", and
            # its verdict is 'failed' — which reads as "the system broke". For
            # most rows that is wrong and misleading: they are not failures,
            # they are RESULTS. Two distinct reasons now write it:
            #   'no_stellar_mass'    — the catalogue has no mass for the host,
            #                          so TRICERATOPS cannot run at all. (These
            #                          are overwhelmingly red giants: TIC v8
            #                          only derives masses on the main sequence.)
            #   'stellar_companion'  — depth + R* imply a companion far too
            #                          large to be a planet, i.e. an eclipsing
            #                          binary or a blend (stellar_gates.py).
            # NULL = a real TRICERATOPS result, or not gated. Additive column:
            # nothing reads it yet, so no existing behaviour changes.
            try:
                self.conn.execute(
                    "ALTER TABLE candidates ADD COLUMN fpp_skip_reason TEXT")
            except sqlite3.OperationalError:
                pass

            # Migrate: verdict_proba (P3) — the verdict classifier's raw
            # P(planet), written by update_candidate_claude. NULL = not
            # computed (legacy rows, v3 "(no_fpp)", Claude fallback). Kept
            # untouched by postprocess demotions on purpose (near-miss ranking).
            try:
                self.conn.execute(
                    "ALTER TABLE candidates ADD COLUMN verdict_proba REAL")
            except sqlite3.OperationalError:
                pass

            # Migrate: cross-source photometric confirmation (Codex #3)
            for col, spec in [
                ("cross_source_name", "TEXT"),
                ("cross_source_snr", "REAL"),
                ("cross_source_depth_ppm", "REAL"),
                ("cross_source_depth_ratio", "REAL"),
                ("cross_source_agreement", "INTEGER"),
            ]:
                try:
                    self.conn.execute(
                        f"ALTER TABLE candidates ADD COLUMN {col} {spec}")
                except sqlite3.OperationalError:
                    pass

            # Migrate: SPOC availability flag on stars.
            # NULL = not yet checked; 1 = has 2-min SPOC at MAST; 0 = FFI-only.
            # Populated server-side by _run_spoc_check_background (gated by
            # EXONIT_SPOC_GATE=1).  Used to pre-mark FFI-only candidates with
            # exominer_score=-1 and skip the DV-fetch round-trip.
            try:
                self.conn.execute(
                    "ALTER TABLE stars ADD COLUMN has_spoc INTEGER")
            except sqlite3.OperationalError:
                pass  # column already exists

            # Performance indexes on candidates. The table grows to hundreds of
            # thousands of rows across a discovery campaign (665k+ at sector
            # scale), and the only prior index was the (sector, tic_id,
            # candidate_num) UNIQUE one. Without these, the Results page
            # (ORDER BY snr DESC LIMIT/OFFSET), the verdict filters, and the
            # postprocess "WHERE tic_id IN (...)" / "WHERE claude_verdict=..."
            # queries all did full table scans + sorts (170-500 ms each at 665k;
            # multiplied across COUNT + SELECT per page). With these the same
            # queries are sub-millisecond. The write cost (4 indexes updated per
            # INSERT) is negligible at the few-hundred-inserts/min worker rate.
            #   - snr: default sort column of the Results page.
            #   - (claude_verdict, snr): verdict filter + sort in one shot
            #     (filter=planet, and the dominant postprocess WHERE).
            #   - tic_id: JOINs to stars + "WHERE tic_id IN (...)" (postprocess
            #     steps 1b/1c) — the prefix of the UNIQUE index is `sector`, so
            #     a bare tic_id lookup could not use it.
            #   - cnn_score: strong filter + the LoRA "missing" workload query.
            #
            # Partial "todo" indexes for the worker backfill poll endpoints.
            # /api/worker/fpp_task and /api/worker/exominer_task ran a JOIN +
            # filter + ORDER BY over the ~355k candidates with snr>=10 & cnn>=0.7
            # on EVERY poll — ~1.7 s each even when there was no work to hand out.
            # Idle workers poll continuously, so with several workers x N server
            # processes the server's cores saturated doing useless scans. These
            # partial indexes encode the FULL "needs work" predicate, so they are
            # EMPTY when there is no backfill work → a cheap EXISTS guard in the
            # endpoint returns instantly (1.7 s → 0.1 ms). When work exists they
            # also locate the top-SNR candidate fast. They cover only the few
            # rows that actually need work, so the write cost is tiny.
            #   - fpp_todo:      candidates awaiting TRICERATOPS FPP
            #   - nfpp_todo:     low-FPP candidates awaiting NFPP
            #   - exominer_todo: candidates awaiting ExoMiner (no snr gate —
            #                    matches the exominer_task query exactly)
            # And tasks(status,...) so the primary /api/worker/task poll finds
            # pending / assigned tasks via an index instead of scanning the
            # 233k-row tasks table twice per idle poll (~124 ms → instant).
            for idx_sql in (
                "CREATE INDEX IF NOT EXISTS idx_candidates_snr "
                "ON candidates(snr DESC)",
                "CREATE INDEX IF NOT EXISTS idx_candidates_verdict_snr "
                "ON candidates(claude_verdict, snr DESC)",
                "CREATE INDEX IF NOT EXISTS idx_candidates_tic "
                "ON candidates(tic_id)",
                "CREATE INDEX IF NOT EXISTS idx_candidates_cnn "
                "ON candidates(cnn_score)",
                # idx_cand_fpp_todo / idx_cand_nfpp_todo: OR-gate predicate
                # change (2026-07-11, docs/reviews/injection_e2e_results_
                # 2026-07-11.md + docs/reviews/ffi_cnn_investigation_2026-07-
                # 11.md §5) — snr>=10 AND cnn_score>=0.7 alone kills ~67% of
                # genuine transits in FFI-weak hosts; em_ffi>=0.5 rescues a
                # meaningful slice, so eligibility is now
                # "cnn_score>=0.7 OR exominer_ffi_score>=0.5". DROP+CREATE
                # (not just IF NOT EXISTS), same pattern as idx_cand_strong's
                # 2026-07-02 NULLIF migration just below and
                # idx_candidates_unique above — a plain IF NOT EXISTS would
                # leave an already-created index on the OLD predicate
                # forever, invisible to the new OR-gate queries (predicate
                # mismatch -> silent fallback to a full scan; see
                # worker_fpp_task / /api/health/work / /api/backfill/fpp in
                # exonit/routes/{workers,admin}.py, which all use the
                # literal ``(cnn_score >= 0.7 OR exominer_ffi_score >= 0.5)``
                # text so INDEXED BY keeps matching).
                "DROP INDEX IF EXISTS idx_cand_fpp_todo",
                "CREATE INDEX IF NOT EXISTS idx_cand_fpp_todo "
                "ON candidates(snr DESC) "
                "WHERE triceratops_fpp IS NULL AND snr >= 10.0 "
                "AND (cnn_score >= 0.7 OR exominer_ffi_score >= 0.5)",
                "DROP INDEX IF EXISTS idx_cand_nfpp_todo",
                "CREATE INDEX IF NOT EXISTS idx_cand_nfpp_todo "
                "ON candidates(snr DESC) "
                "WHERE triceratops_nfpp IS NULL AND triceratops_fpp >= 0 "
                "AND triceratops_fpp < 0.5 AND snr >= 10.0 "
                "AND (cnn_score >= 0.7 OR exominer_ffi_score >= 0.5)",
                # idx_cand_exominer_todo: Docker-SPOC ExoMiner backfill
                # (has_spoc=1 path, EXONIT_ENABLE_EXOMINER_AUTO=0 in prod as
                # of 2026-07-11) — deliberately UNCHANGED. It is outside the
                # discovery chain this OR-gate change targets (item 6 of
                # progress/impl-orgate.md); see docs/reviews/ffi_cnn_
                # investigation_2026-07-11.md §5.
                "CREATE INDEX IF NOT EXISTS idx_cand_exominer_todo "
                "ON candidates(snr DESC) "
                "WHERE exominer_score IS NULL AND cnn_score >= 0.7",
                # idx_cand_exominer_status_todo: same "queue" shape as
                # idx_cand_exominer_todo above, keyed on the new status
                # column instead of the ambiguous exominer_score IS NULL
                # (docs/superpowers/specs/exominer-status-column.md — "Pas
                # 2", migration step 3). Not read by any production query
                # yet in this iteration (the spec scopes the READ side to
                # exominer_pending's admin.py counter + the health check
                # only, deliberately left untouched here — see
                # progress/impl-exominer-status.md); this index exists so a
                # future cutover of that counter/selector is a query-text
                # change only, with the fast path already in place.
                "CREATE INDEX IF NOT EXISTS idx_cand_exominer_status_todo "
                "ON candidates(snr DESC) "
                "WHERE exominer_status = 'pending' AND cnn_score >= 0.7",
                "CREATE INDEX IF NOT EXISTS idx_tasks_status_priority "
                "ON tasks(status, priority, created_at)",
                # Strong-tier display count (/api/stats). Partial index on the
                # selective part of the strong predicate ((cnn>=0.7 OR
                # em_ffi>=0.5) AND the exominer gate) so COUNT(*) is ~0.1 ms
                # instead of a full scan.
                # DROP+CREATE (not just IF NOT EXISTS) because the predicate
                # changed twice now: 2026-07-02 (the exominer_score=-1 read-
                # boundary fix, see _strong_where's docstring +
                # progress/impl-minus1-fix-2026-07-02.md) and 2026-07-11 (the
                # OR-gate, see _strong_where's docstring above) — a plain IF
                # NOT EXISTS would leave any already-created index on the OLD
                # predicate forever, which the planner can't match against
                # the new query (predicate mismatch -> silent fallback to a
                # full scan). Same DROP+CREATE-on-definition-change pattern
                # as idx_candidates_unique above.
                "DROP INDEX IF EXISTS idx_cand_strong",
                "CREATE INDEX IF NOT EXISTS idx_cand_strong "
                "ON candidates(snr DESC) "
                "WHERE (cnn_score >= 0.7 OR exominer_ffi_score >= 0.5) "
                "AND COALESCE(NULLIF(exominer_score, -1), exominer_ffi_score) >= 0.7",
                # Latest-telemetry lookup (/api/worker/list does this per worker):
                # "WHERE worker_id=? ORDER BY id DESC LIMIT 1" over the growing
                # worker_telemetry table (131k+ rows) was a ~27 ms scan each;
                # this makes it an instant index seek.
                "CREATE INDEX IF NOT EXISTS idx_telemetry_worker "
                "ON worker_telemetry(worker_id, id DESC)",
                # /api/backfill/nfpp — three queries scanned the full 668k-row
                # candidates table on every call (py-spy confirmed):
                #   1. pending + pending_tics: COUNT(*) / COUNT(DISTINCT tic_id)
                #      WHERE fpp>=0 AND fpp<0.5 AND nfpp IS NULL.
                #      Note: fpp>=0 already implies fpp IS NOT NULL in SQL (NULL
                #      comparisons are always false), so the explicit IS NOT NULL
                #      that the endpoint had is redundant. The index predicate
                #      omits it; the query is simplified to match (same result,
                #      cleaner predicate for the planner to recognise).
                #      Index on tic_id lets it serve both the COUNT(*) and the
                #      COUNT(DISTINCT tic_id) from the same index.
                #   2. verdict GROUP BY: SELECT triceratops_verdict, COUNT(*)
                #      WHERE triceratops_verdict IS NOT NULL GROUP BY ...
                #      The partial index encodes the IS NOT NULL predicate so
                #      the scan covers only the ~357k non-null rows instead of
                #      all 668k; the GROUP BY can be resolved from the index.
                # Both indexes are PARTIAL so they are small (they only cover
                # the rows matching the predicate). The pending index is
                # typically empty between runs (no backfill work outstanding),
                # making the index essentially zero-cost at that time.
                "CREATE INDEX IF NOT EXISTS idx_cand_nfpp_pending "
                "ON candidates(tic_id) "
                "WHERE triceratops_fpp >= 0 AND triceratops_fpp < 0.5 "
                "AND triceratops_nfpp IS NULL",
                "CREATE INDEX IF NOT EXISTS idx_cand_verdict_notnull "
                "ON candidates(triceratops_verdict) "
                "WHERE triceratops_verdict IS NOT NULL",
                # Partial index for /api/stars: limits the completed-stars scan
                # to only the rows the endpoint cares about, ordered so the LIMIT
                # can be satisfied by an index range scan rather than a full sort.
                "CREATE INDEX IF NOT EXISTS idx_stars_completed_analyzed "
                "ON stars(analyzed_at DESC) "
                "WHERE status = 'completed'",
                # P3 (docs/reviews/2026-07-02-opus48-cleanup-review.md):
                # stars.has_spoc had NO index anywhere, despite being the hot
                # filter behind the 2026-06-20 PG meltdown (see
                # gotcha_pg_stale_stats_fpp_backfill_meltdown /
                # gotcha_exominer_backfill_spoc_gate_hole in memory) — the
                # current mitigation is a 20s TTL cache band-aid, not an
                # index. Partial on has_spoc=1 (the selective, "has 2-min
                # SPOC" side of the flag) so it stays cheap: most rows are
                # FFI-only (0) or unchecked (NULL) and never enter the index.
                "CREATE INDEX IF NOT EXISTS idx_stars_has_spoc "
                "ON stars(tic_id) "
                "WHERE has_spoc = 1",
            ):
                try:
                    self.conn.execute(idx_sql)
                except sqlite3.OperationalError:
                    pass  # index already exists

            self.conn.commit()

    def _create_completeness_schema(self):
        """Create the completeness / completeness_agg tables, their triggers and
        the one-time backfill on the dedicated SQLite connection.

        This subsystem stays on SQLite in BOTH backends (in PG mode it lives on
        self._cmpl_conn, a separate sqlite3 file).  In SQLite mode self._cmpl_conn
        is the same object as self.conn, so this runs against the one connection
        exactly as before.
        """
        conn = self._cmpl_conn
        with self._cmpl_lock:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS completeness (
                    tic_id TEXT,
                    radius_earth REAL, period_days REAL,
                    detection_prob REAL,
                    PRIMARY KEY (tic_id, radius_earth, period_days)
                );
                -- Incremental aggregate: sum_prob/n gives AVG(detection_prob)
                -- per (radius_earth, period_days) cell without scanning the
                -- full completeness table (25M+ rows → 34 s → < 1 ms).
                -- Maintained in sync by triggers on completeness + the
                -- ON CONFLICT DO UPDATE writers.
                CREATE TABLE IF NOT EXISTS completeness_agg (
                    radius_earth REAL NOT NULL,
                    period_days REAL NOT NULL,
                    sum_prob REAL NOT NULL,
                    n INTEGER NOT NULL,
                    PRIMARY KEY (radius_earth, period_days)
                );
            """)
            # --- completeness_agg triggers ---
            # Three triggers keep completeness_agg in exact sync with completeness
            # for every row written via the ON CONFLICT DO UPDATE writers below.
            # INSERT trigger: add or accumulate in the aggregate row.
            # UPDATE trigger: adjust the running sum by the delta (new - old).
            # DELETE trigger: subtract the deleted row from the aggregate; remove
            #   the aggregate row when n reaches 0 to avoid a stale zero-row.
            for trig_sql in (
                """CREATE TRIGGER IF NOT EXISTS trg_completeness_agg_insert
                   AFTER INSERT ON completeness
                   BEGIN
                       INSERT INTO completeness_agg (radius_earth, period_days, sum_prob, n)
                           VALUES (NEW.radius_earth, NEW.period_days, NEW.detection_prob, 1)
                       ON CONFLICT(radius_earth, period_days) DO UPDATE SET
                           sum_prob = sum_prob + NEW.detection_prob,
                           n        = n + 1;
                   END""",
                """CREATE TRIGGER IF NOT EXISTS trg_completeness_agg_update
                   AFTER UPDATE OF detection_prob ON completeness
                   BEGIN
                       UPDATE completeness_agg
                          SET sum_prob = sum_prob + (NEW.detection_prob - OLD.detection_prob)
                        WHERE radius_earth = NEW.radius_earth
                          AND period_days  = NEW.period_days;
                   END""",
                """CREATE TRIGGER IF NOT EXISTS trg_completeness_agg_delete
                   AFTER DELETE ON completeness
                   BEGIN
                       UPDATE completeness_agg
                          SET sum_prob = sum_prob - OLD.detection_prob,
                              n        = n - 1
                        WHERE radius_earth = OLD.radius_earth
                          AND period_days  = OLD.period_days;
                       DELETE FROM completeness_agg
                        WHERE radius_earth = OLD.radius_earth
                          AND period_days  = OLD.period_days
                          AND n <= 0;
                   END""",
            ):
                try:
                    conn.execute(trig_sql)
                except sqlite3.OperationalError:
                    pass  # trigger already exists

            # --- completeness_agg backfill (one-time migration) ---
            # If the DB has completeness rows but completeness_agg is empty (e.g.
            # existing deployment upgrading to this version), populate the agg
            # table with a single GROUP BY pass.  The writers going forward keep
            # it in sync via the triggers above, so this path is only taken once.
            agg_count = conn.execute(
                "SELECT COUNT(*) FROM completeness_agg").fetchone()[0]
            if agg_count == 0:
                src_count = conn.execute(
                    "SELECT COUNT(*) FROM completeness").fetchone()[0]
                if src_count > 0:
                    import time as _time
                    _t0 = _time.monotonic()
                    conn.execute("""
                        INSERT INTO completeness_agg (radius_earth, period_days, sum_prob, n)
                        SELECT radius_earth, period_days,
                               SUM(detection_prob), COUNT(*)
                          FROM completeness
                         GROUP BY radius_earth, period_days
                    """)
                    import logging as _logging
                    _logging.getLogger(__name__).info(
                        "completeness_agg backfill: %d source rows → done in %.1f s",
                        src_count, _time.monotonic() - _t0)
            conn.commit()

    # --- Phase 4a: completeness-on-PG schema (EXONIT_COMPLETENESS_PG=1) --- #

    def _ensure_completeness_pg_schema(self):
        """Create the PG completeness_staging / completeness_tics /
        completeness_agg_bucket tables (idempotent ``CREATE TABLE IF NOT
        EXISTS``), mirroring the DDL hand-authored in
        ``scripts/migration/schema_pg.sql``.

        Called from ``__init__`` only when the Phase 4a cutover flag
        (``EXONIT_COMPLETENESS_PG=1``) is on and the main backend is
        PostgreSQL. In-code creation (rather than relying solely on the SQL
        file being applied out of band) means a fresh PG database — e.g. the
        throwaway ``exonit_test`` used by the PG-parity test suite — gets the
        schema automatically the first time a flag-on ``ExoNitDB`` connects,
        with no separate provisioning step.

        Task 4 follow-up: ``CREATE ... IF NOT EXISTS`` is not actually
        race-free in PostgreSQL — two A3 worker processes racing to connect
        for the very first time (fresh DB, nothing created yet) can both pass
        the catalog's "does not exist" check before either commits, so one of
        them still raises a duplicate_table/duplicate_object/unique_violation
        error despite the IF NOT EXISTS clause. Each statement is executed
        individually and that specific class of error is swallowed (DEBUG
        log) — it means the other racing process's CREATE already won, which
        is exactly the end state this method is trying to guarantee. Any
        other error still propagates.
        """
        for stmt in (
            """CREATE TABLE IF NOT EXISTS completeness_staging (
                id BIGSERIAL PRIMARY KEY,
                tic_id TEXT NOT NULL,
                radius_earth DOUBLE PRECISION NOT NULL,
                period_days DOUBLE PRECISION NOT NULL,
                detection_prob DOUBLE PRECISION NOT NULL,
                written_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )""",
            "CREATE INDEX IF NOT EXISTS idx_cmpl_staging_tic "
            "ON completeness_staging(tic_id)",
            """CREATE TABLE IF NOT EXISTS completeness_tics (
                tic_id TEXT PRIMARY KEY,
                bucket SMALLINT NOT NULL
            )""",
            "CREATE INDEX IF NOT EXISTS idx_cmpl_tics_bucket "
            "ON completeness_tics(bucket)",
            """CREATE TABLE IF NOT EXISTS completeness_agg_bucket (
                bucket SMALLINT NOT NULL,
                radius_earth DOUBLE PRECISION NOT NULL,
                period_days DOUBLE PRECISION NOT NULL,
                sum_prob DOUBLE PRECISION NOT NULL,
                n BIGINT NOT NULL,
                PRIMARY KEY (bucket, radius_earth, period_days)
            )""",
        ):
            try:
                self.backend.execute(stmt)
            except Exception as exc:
                if not _is_concurrent_ddl_race(exc):
                    raise
                logger.debug(
                    "_ensure_completeness_pg_schema: concurrent CREATE race "
                    "on %r ignored (sqlstate=%s) — another process already "
                    "created it", stmt.strip().splitlines()[0][:60],
                    getattr(exc, "sqlstate", None))

    _STAR_COLS = ("tic_id", "ra", "dec", "tmag", "analyzed_at", "n_sectors",
                  "n_datapoints", "time_span_days", "status", "sectors_observed")

    def add_star(self, tic_id, ra, dec, tmag, n_sectors, n_datapoints,
                 time_span_days, status="completed", sectors_observed=None):
        sectors_json = json.dumps(sectors_observed) if sectors_observed else None
        sql = self.dialect.upsert("stars", self._STAR_COLS,
                                  conflict_cols=("tic_id",))
        self._exec(
            sql,
            (tic_id, ra, dec, tmag, utc_now_iso(),
             n_sectors, n_datapoints, time_span_days, status, sectors_json))

    def add_stars_bulk(self, tic_ids, ra, dec, tmag, n_sectors, n_datapoints,
                      time_span_days, status="completed", sectors_observed=None):
        """Batch upsert of IDENTICAL star rows for many TIC ids (Q9, perf
        review 2026-09-26, docs/reviews/2026-09-26-enqueue-path-performance-
        review.md).

        exonit.enqueue.enqueue_targets marked no-FITS TICs one at a time via
        add_star() in a Python for-loop — one execute() + one commit (on
        SQLite) per TIC, for rows that are otherwise identical (ra=dec=
        tmag=n_sectors=n_datapoints=time_span_days=0, status='no_fits'; only
        tic_id and the timestamp differ). This does the exact same upsert in
        ONE executemany() round trip (one transaction/commit on SQLite; one
        pipelined call on PG) instead of N.

        *sectors_observed* is shared across every row (same JSON-encoded
        value or NULL for all of them) — matches the only real caller's
        actual usage (enqueue_targets never passes it here); a caller
        needing per-row sectors_observed should call add_star() per row
        instead.
        """
        if not tic_ids:
            return
        sectors_json = json.dumps(sectors_observed) if sectors_observed else None
        sql = self.dialect.upsert("stars", self._STAR_COLS,
                                  conflict_cols=("tic_id",))
        now = utc_now_iso()
        rows = [
            (tic_id, ra, dec, tmag, now, n_sectors, n_datapoints,
             time_span_days, status, sectors_json)
            for tic_id in tic_ids
        ]
        self.backend.executemany(sql, rows)

    def update_star_sectors(self, tic_id, sectors_observed):
        """Update observed sectors for an existing star."""
        self._exec(
            "UPDATE stars SET sectors_observed = ? WHERE tic_id = ?",
            (json.dumps(sectors_observed), tic_id))

    def get_star_sectors(self, tic_id):
        """Get observed sectors as a list of ints, or None."""
        row = self._query_one(
            "SELECT sectors_observed FROM stars WHERE tic_id = ?", (tic_id,))
        if row and row["sectors_observed"]:
            return json.loads(row["sectors_observed"])
        return None

    def get_star(self, tic_id):
        return self._query_one("SELECT * FROM stars WHERE tic_id = ?", (tic_id,))

    def is_analyzed(self, tic_id):
        return self._query_one(
            "SELECT 1 FROM stars WHERE tic_id = ? AND status = 'completed'",
            (tic_id,)) is not None

    def completed_tic_ids(self) -> set:
        """Return the set of TIC IDs (as str) that are already completed.

        Used by enqueue_targets to build a pre-loaded set for O(1) per-target
        lookup instead of one query per target.
        """
        rows = self._query("SELECT tic_id FROM stars WHERE status = 'completed'")
        return {str(r["tic_id"]) for r in rows}

    def completed_tic_ids_in(self, tic_ids) -> set:
        """Return the subset of *tic_ids* (as str) that are already completed.

        Scoped counterpart of :meth:`completed_tic_ids` for
        ``enqueue_targets``'s ``EXONIT_ENQUEUE_METADATA_V2`` path (docs/
        superpowers/specs/2026-09-26-metadata-refresh-and-bulk-enqueue-design.md
        §2.3): a lot is at most a few hundred thousand TICs, so a chunked
        ``WHERE tic_id IN (...)`` probe against ``stars`` is far cheaper than
        materialising the whole ~27M-row ``completed`` set (the RSS spike this
        method exists to avoid — see that method's caller history). Chunked
        via :func:`exonit.db_backend.select_in_chunks` (30000/chunk, the same
        helper/threshold every other bulk IN-probe in this module uses).
        """
        ids = list({str(t) for t in tic_ids})
        if not ids:
            return set()
        rows = select_in_chunks(
            self._query,
            "SELECT tic_id FROM stars WHERE status = 'completed' AND tic_id IN",
            ids)
        return {str(r["tic_id"]) for r in rows}

    def add_candidate(self, tic_id, candidate_num, period_days, t0,
                      duration_days, depth_ppm, snr, cnn_score=None,
                      triceratops_fpp=None, triceratops_nfpp=None,
                      triceratops_prob_planet=None, triceratops_verdict=None,
                      claude_verdict=None, claude_analysis=None,
                      exominer_ffi_score=None,
                      exominer_ffi_model_version=None,
                      exominer_ffi_score_v8=None,
                      # ^ v8 retired 2026-06-27 (commit b60570bf; the v8 ONNX
                      # no longer ships). Accepted-but-ignored so the live
                      # server.py:2244 call site (out of scope for this
                      # change; a follow-up removes it there) keeps working —
                      # it is never written to the DB below. See
                      # docs/reviews/2026-07-02-opus48-cleanup-review.md.
                      cross_source_name=None, cross_source_snr=None,
                      cross_source_depth_ppm=None,
                      cross_source_depth_ratio=None,
                      cross_source_agreement=None, sector=None):
        """Insert or update a candidate without erasing later analyses.

        Reprocessing the same TIC should refresh measured transit parameters but
        must preserve FPP/NFPP, ExoMiner, and verdict fields unless fresh values
        are explicitly supplied.
        """
        tic_id = str(tic_id)
        # sector: the discovery run's TESS sector when an active run tags it (the
        # server passes the active-run sector); otherwise default 1 — preserving
        # historical behaviour for home/non-discovery inserts and the AWS
        # sector-2 dedup. The candidates table is PARTITION BY LIST (sector) on PG.
        _sector = 1 if sector is None else int(sector)
        if triceratops_verdict is None and triceratops_fpp is not None:
            triceratops_verdict = fpp_verdict(triceratops_fpp, triceratops_nfpp)
        # cross_source_agreement is a BOOLEAN column on PG — coerce at the edge.
        if cross_source_agreement is not None:
            cross_source_agreement = self.dialect.bool_param(cross_source_agreement)

        # EXPLICIT sector: strict (tic_id, candidate_num, sector) lookup — the
        # unique index is (sector, tic_id, candidate_num), so a TIC processed in
        # a NEW sector INSERTs its own row instead of overwriting another
        # sector's candidate (prerequisite for cross-sector recurrence).
        # sector=None (untagged caller: straggler results outside SCANNING,
        # non-discovery reprocess): legacy sector-agnostic lookup + UPDATE of
        # the found row, so an untagged caller never creates a duplicate.
        if sector is None:
            sel_sql = ("""SELECT id, sector FROM candidates
                          WHERE tic_id = ? AND candidate_num = ?
                          ORDER BY id ASC LIMIT 1""")
            sel_params = (tic_id, candidate_num)
        else:
            sel_sql = ("""SELECT id, sector FROM candidates
                          WHERE tic_id = ? AND candidate_num = ? AND sector = ?
                          ORDER BY id ASC LIMIT 1""")
            sel_params = (tic_id, candidate_num, _sector)
        row = self.backend.query_one(sel_sql, sel_params)
        if row is None:
            # No existing candidate — insert. Under uvicorn --workers N the
            # per-process lock cannot serialize across processes, so two workers
            # reprocessing the same TIC can race here; the loser hits the unique
            # index (sector, tic_id, candidate_num). On that conflict the row now
            # exists, so re-fetch and fall through to the UPDATE below instead of
            # failing the worker's result POST (500).
            try:
                # exominer_status='pending' is NOT optional here. A fresh
                # candidate has exominer_score IS NULL, which the old
                # score-based queue selector reads as "queue-eligible". If we
                # left status NULL, the status-based selector would never see
                # it — and it would STAY NULL forever: the SPOC gate's success
                # path (has_spoc=1 + DV present) calls
                # clear_spurious_minus_one_candidates, whose `WHERE
                # exominer_score = -1` is a no-op on a never-sentinelled row.
                # So the row would be permanently invisible to the new selector
                # while the old one still queues it — the exact parity break the
                # migration exists to avoid. 'pending' == "score IS NULL" by
                # construction (derive_exominer_status(None) -> 'pending').
                return self.backend.execute_returning_id(
                    """INSERT INTO candidates
                       (tic_id, candidate_num, period_days, t0, duration_days,
                        depth_ppm, snr, cnn_score, triceratops_fpp,
                        triceratops_nfpp, triceratops_prob_planet,
                        triceratops_verdict, claude_verdict, claude_analysis,
                        exominer_ffi_score, exominer_ffi_model_version,
                        cross_source_name, cross_source_snr,
                        cross_source_depth_ppm, cross_source_depth_ratio,
                        cross_source_agreement, sector, exominer_status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, ?, ?, ?, ?, 'pending')""",
                    (tic_id, candidate_num, period_days, t0, duration_days,
                     depth_ppm, snr, cnn_score, triceratops_fpp,
                     triceratops_nfpp, triceratops_prob_planet,
                     triceratops_verdict, claude_verdict, claude_analysis,
                     exominer_ffi_score, exominer_ffi_model_version,
                     cross_source_name, cross_source_snr,
                     cross_source_depth_ppm, cross_source_depth_ratio,
                     cross_source_agreement, _sector))
            except Exception as exc:
                # Unique-constraint race: another writer inserted the same
                # (sector, tic_id, candidate_num) first. Re-fetch and update.
                if not _is_unique_violation(exc):
                    raise
                row = self.backend.query_one(sel_sql, sel_params)
                if row is None:
                    raise

        candidate_id = row["id"]
        _row_sector = row["sector"] if row["sector"] is not None else _sector
        self.backend.execute(
            """UPDATE candidates SET
                   period_days = ?, t0 = ?, duration_days = ?,
                   depth_ppm = ?, snr = ?,
                   cnn_score = COALESCE(?, cnn_score),
                   triceratops_fpp = COALESCE(?, triceratops_fpp),
                   triceratops_nfpp = COALESCE(?, triceratops_nfpp),
                   triceratops_prob_planet = COALESCE(?, triceratops_prob_planet),
                   triceratops_verdict = COALESCE(?, triceratops_verdict),
                   claude_verdict = COALESCE(?, claude_verdict),
                   claude_analysis = COALESCE(?, claude_analysis),
                   exominer_ffi_score = COALESCE(?, exominer_ffi_score),
                   exominer_ffi_model_version = COALESCE(?, exominer_ffi_model_version),
                   cross_source_name = COALESCE(?, cross_source_name),
                   cross_source_snr = COALESCE(?, cross_source_snr),
                   cross_source_depth_ppm = COALESCE(?, cross_source_depth_ppm),
                   cross_source_depth_ratio = COALESCE(?, cross_source_depth_ratio),
                   cross_source_agreement = COALESCE(?, cross_source_agreement)
               WHERE id = ? AND sector = ?""",
            (period_days, t0, duration_days, depth_ppm, snr,
             cnn_score, triceratops_fpp, triceratops_nfpp,
             triceratops_prob_planet, triceratops_verdict,
             claude_verdict, claude_analysis,
             exominer_ffi_score, exominer_ffi_model_version,
             cross_source_name, cross_source_snr,
             cross_source_depth_ppm, cross_source_depth_ratio,
             cross_source_agreement,
             candidate_id, _row_sector))
        return candidate_id

    def delete_candidates_for_tic(self, tic_id, sector=None) -> int:
        """Delete candidates of *tic_id*. Returns rows deleted.

        ``sector=None`` (legacy): ALL sectors. An int scopes the wipe to that
        sector so a cross-sector re-observation cannot destroy the other
        sectors' evidence.

        Support for the deep-rescan replace-when-deeper rule
        (docs/superpowers/specs/2026-07-23-deep-pass-multisector-design.md §5):
        a result computed on a strictly longer baseline supersedes the whole
        old candidate set — per-candidate upsert would glue old FPP/verdicts
        onto signals whose period/rank changed with the longer baseline.
        """
        if sector is not None:
            return int(self.backend.execute_rowcount(
                "DELETE FROM candidates WHERE tic_id = ? AND sector = ?",
                (str(tic_id), int(sector))))
        return int(self.backend.execute_rowcount(
            "DELETE FROM candidates WHERE tic_id = ?", (str(tic_id),)))

    def get_candidates(self, tic_id):
        return self._query("SELECT * FROM candidates WHERE tic_id = ?", (tic_id,))

    def get_all_candidates(self, min_snr=None):
        # snr is a nullable column (schema has no NOT NULL) — B5 dialect-drift
        # fix, see exonit.db_backend.Dialect.nulls_last.
        order = self.dialect.nulls_last("snr")
        if min_snr is not None:
            return self._query(
                f"SELECT * FROM candidates WHERE snr >= ? ORDER BY {order}",
                (min_snr,))
        return self._query(f"SELECT * FROM candidates ORDER BY {order}")

    def update_candidate_fpp(self, candidate_id, fpp, nfpp=None,
                             prob_planet=None, verdict=None):
        """Update TRICERATOPS FPP/NFPP for a candidate.

        A failed NFPP backfill must not erase a previous valid FPP. In that
        case, keep the old FPP and mark NFPP as failed so the task does not
        loop forever and the candidate cannot be validated from FPP alone.

        After the write, best-effort re-enqueues the candidate on the "fpp"
        work stream (Phase 1, Task 2) if arm B (nfpp still pending, see
        exonit.work_producers) applies to the just-written values — if the
        candidate is fully done, the hook is simply a no-op. A hook failure
        must never break the write above, hence the outer try/except.
        """
        try:
            fpp_value = float(fpp) if fpp is not None else None
        except (TypeError, ValueError):
            fpp_value = None

        final_fpp = fpp_value
        with self.backend.transaction() as conn:
            # Widened vs. the original triceratops_fpp-only SELECT to also
            # carry snr/cnn_score/exominer_ffi_score/tic_id for the producer
            # hook below — same WHERE (id = ?), so cardinality/behaviour is
            # unchanged. exominer_ffi_score feeds the OR-gate in
            # work_producers.verdict_eligible (2026-07-11).
            row = self.backend.tx_query_one(
                conn,
                "SELECT triceratops_fpp, snr, cnn_score, exominer_ffi_score, "
                "tic_id FROM candidates WHERE id = ?",
                (candidate_id,))
            old_fpp = row["triceratops_fpp"] if row else None
            if fpp_value is not None and fpp_value < 0 and old_fpp is not None and old_fpp >= 0:
                verdict = fpp_verdict(old_fpp, nfpp)
                final_fpp = old_fpp
                self.backend.tx_execute(
                    conn,
                    """UPDATE candidates SET triceratops_nfpp = ?,
                              triceratops_prob_planet = ?,
                              triceratops_verdict = ?
                       WHERE id = ?""",
                    (nfpp, prob_planet, verdict, candidate_id))
            else:
                if verdict is None:
                    verdict = fpp_verdict(fpp, nfpp)
                self.backend.tx_execute(
                    conn,
                    """UPDATE candidates SET triceratops_fpp = ?,
                              triceratops_nfpp = ?,
                              triceratops_prob_planet = ?,
                              triceratops_verdict = ?
                       WHERE id = ?""",
                    (fpp, nfpp, prob_planet, verdict, candidate_id))

        if row is not None:
            try:
                from exonit import work_producers
                if work_producers.streams_enabled():
                    star = self.get_star(row["tic_id"])
                    work_producers.maybe_enqueue_fpp({
                        "candidate_id": candidate_id,
                        "tic_id": row["tic_id"],
                        "snr": row["snr"],
                        "cnn_score": row["cnn_score"],
                        "exominer_ffi_score": row["exominer_ffi_score"],
                        "triceratops_fpp": final_fpp,
                        "nfpp": nfpp,
                        "star_status": star["status"] if star else None,
                    })
            except Exception:
                logger.warning(
                    "work_producers hook failed for candidate_id=%s",
                    candidate_id, exc_info=True)

    def update_candidate_claude(self, candidate_id, verdict, analysis,
                                verdict_proba=_UNSET):
        """Update verdict + analysis for a candidate.

        ``verdict_proba`` (raw classifier P(planet)): left as the sentinel
        the column is not touched (legacy callers); when passed — even as
        None — it is written, so a re-run never leaves a stale p behind.

        Preserves an existing `[lora-v2]`-tagged analysis if the incoming
        analysis would degrade it (empty, shorter, or untagged). The verdict
        is always updated. This guards against M4 explainer timeouts on
        re-runs silently overwriting good explanations with classifier
        fallback text.
        """
        row = self._query(
            "SELECT claude_analysis FROM candidates WHERE id = ?",
            (candidate_id,))
        existing = row[0]["claude_analysis"] if row else None
        existing_is_lora = (existing or "").startswith("[lora-v2]")
        incoming_is_lora = (analysis or "").startswith("[lora-v2]")
        keep_existing = (
            existing_is_lora
            and not incoming_is_lora
            and len(existing or "") >= len(analysis or "")
        )
        write_proba = verdict_proba is not _UNSET
        if keep_existing:
            if write_proba:
                self._exec(
                    "UPDATE candidates SET claude_verdict = ?, "
                    "verdict_proba = ? WHERE id = ?",
                    (verdict, verdict_proba, candidate_id))
            else:
                self._exec(
                    "UPDATE candidates SET claude_verdict = ? WHERE id = ?",
                    (verdict, candidate_id))
        else:
            if write_proba:
                self._exec(
                    """UPDATE candidates SET claude_verdict = ?,
                       claude_analysis = ?, verdict_proba = ?
                       WHERE id = ?""",
                    (verdict, analysis, verdict_proba, candidate_id))
            else:
                self._exec(
                    """UPDATE candidates SET claude_verdict = ?, claude_analysis = ?
                       WHERE id = ?""",
                    (verdict, analysis, candidate_id))

    def update_candidate_claude_batch(self, updates):
        """Update claude verdict + analysis for a batch of candidates.

        ``updates`` is a list of dicts with mandatory key:
            candidate_id (int)

        And optional keys:
            verdict  (str|None) — ONLY include when you want to write this
                                   value.  **Absent key = do NOT touch
                                   claude_verdict** (preserves any existing
                                   value).  Present key (even None) = write.
            analysis (str|None) — new explanation text.

        Convention rationale: callers that regenerate only the analysis for a
        candidate that already has a verdict (e.g. the LoRA backfill script)
        omit "verdict" so the existing claude_verdict is never clobbered.
        Callers that genuinely assign a new verdict include the key.

        All writes are executed inside a single transaction (one commit).
        The same [lora-v2] preservation logic as ``update_candidate_claude``
        is applied per row: if the existing analysis is already tagged
        [lora-v2] and the incoming one is not (or is shorter), the analysis
        column is left intact.  If the "verdict" key is present the verdict
        column is still written even when the analysis is preserved.

        Returns the number of rows actually updated (rows where at least one
        column was written).
        """
        if not updates:
            return 0

        candidate_ids = [u["candidate_id"] for u in updates]
        with self.backend.transaction() as conn:
            # Fetch existing analyses, chunked (D2: a batch can carry more ids
            # than PG's 65535-param-per-statement limit; see
            # exonit.db_backend.select_in_chunks).
            existing_rows = select_in_chunks(
                lambda sql, params: self.backend.tx_query(conn, sql, params),
                "SELECT id, claude_analysis FROM candidates WHERE id IN",
                candidate_ids,
            )
            existing_by_id = {r["id"]: r["claude_analysis"] for r in existing_rows}

            n_updated = 0
            for u in updates:
                cid = u["candidate_id"]
                # "verdict" key absent → do NOT touch claude_verdict.
                has_verdict = "verdict" in u
                verdict = u.get("verdict")
                analysis = u.get("analysis")
                existing = existing_by_id.get(cid)
                existing_is_lora = (existing or "").startswith("[lora-v2]")
                incoming_is_lora = (analysis or "").startswith("[lora-v2]")
                keep_existing_analysis = (
                    existing_is_lora
                    and not incoming_is_lora
                    and len(existing or "") >= len(analysis or "")
                )
                if keep_existing_analysis:
                    # Preserve analysis.  Update verdict only if provided.
                    if has_verdict:
                        self.backend.tx_execute(
                            conn,
                            "UPDATE candidates SET claude_verdict = ? WHERE id = ?",
                            (verdict, cid),
                        )
                        n_updated += 1
                    # else: nothing to write — skip entirely.
                elif has_verdict:
                    # Write both columns.
                    self.backend.tx_execute(
                        conn,
                        """UPDATE candidates SET claude_verdict = ?,
                                  claude_analysis = ?
                           WHERE id = ?""",
                        (verdict, analysis, cid),
                    )
                    n_updated += 1
                else:
                    # Only analysis, no verdict touch.
                    self.backend.tx_execute(
                        conn,
                        "UPDATE candidates SET claude_analysis = ? WHERE id = ?",
                        (analysis, cid),
                    )
                    n_updated += 1
        return n_updated

    def set_star_has_spoc(self, tic_id, has_spoc):
        """Persist SPOC availability for a star.

        has_spoc: 1 = has 2-min SPOC; 0 = FFI-only confirmed; None clears it.

        ExoMiner producer hook (Phase 1, Task 4): a has_spoc=1 write is
        exactly the transition the legacy ``worker_exominer_task`` queue
        predicate keys off (``s.has_spoc = 1``), so after persisting it we
        best-effort re-enqueue this TIC's currently queue-eligible
        candidates on the "exominer" work stream. Centralised HERE (the
        single write path for stars.has_spoc) rather than duplicated at
        each caller, so both the live SPOC-gate thread
        (``_run_spoc_check_background``) and the offline recovery script
        (``scripts/regate_missed_spoc.py``, which calls this same method)
        get the same producer coverage for free. Queries fresh
        cnn_score/exominer_score/star-status from PG (not values passed
        in here — this method only receives has_spoc) rather than assume
        the star is 'completed', because every caller of this method sets
        has_spoc only for TICs it already knows to be, but a re-check here
        costs one indexed JOIN per gated TIC and removes that assumption.
        A hook failure must never break the has_spoc write above, hence
        the outer try/except (mirrors the same pattern in
        update_candidate_fpp below).
        """
        # BOOLEAN column on PG; keep None (clear) untouched.
        if has_spoc is not None:
            has_spoc = self.dialect.bool_param(has_spoc)
        self._exec(
            "UPDATE stars SET has_spoc = ? WHERE tic_id = ?",
            (has_spoc, str(tic_id)))

        if has_spoc:  # only the has_spoc=1 transition creates new queue work
            try:
                from exonit import work_producers
                if work_producers.streams_enabled():
                    rows = self._query(
                        """SELECT c.id, c.cnn_score, c.exominer_score,
                                  s.status AS star_status
                           FROM candidates c JOIN stars s ON c.tic_id = s.tic_id
                           WHERE c.tic_id = ?""",
                        (str(tic_id),))
                    for row in rows:
                        work_producers.maybe_enqueue_exominer({
                            "candidate_id": row["id"],
                            "tic_id": tic_id,
                            "cnn_score": row["cnn_score"],
                            "exominer_score": row["exominer_score"],
                            "has_spoc": 1,
                            "star_status": row["star_status"],
                        })
            except Exception:
                logger.warning(
                    "ExoMiner producer hook failed for TIC %s", tic_id,
                    exc_info=True)

    def mark_fpp_uncomputable(self, candidate_id: int,
                              reason: str = "no_stellar_mass") -> bool:
        """Mark a single candidate as FPP-uncomputable (fpp=-1, nfpp=-1).

        Called by the FPP-gate (EXONIT_FPP_GATE=1) server-side when the
        target star has no usable stellar mass — the worker would return
        fpp=-1 after a full round-trip anyway.  Pre-marking saves that
        work.

        *reason* records WHY (see the fpp_skip_reason migration): the -1
        sentinel and its 'failed' verdict cannot tell "the catalogue has no
        mass for this star" apart from "this depth implies a stellar
        companion", and those are very different statements about a candidate.

        Safety guards:
          - WHERE triceratops_fpp IS NULL ensures we never overwrite a
            real (≥0) FPP result or a previously gate-marked -1 row.
          - fpp_verdict(-1, -1) = 'failed', consistent with what the
            worker's own -1 path stores.

        Returns True when the row was updated, False otherwise (already
        had a value, or candidate_id not found).
        """
        verdict = fpp_verdict(-1, -1)  # → 'failed'
        rowcount = self.backend.execute_rowcount(
            """UPDATE candidates
               SET triceratops_fpp = -1,
                   triceratops_nfpp = -1,
                   triceratops_verdict = ?,
                   fpp_skip_reason = ?
               WHERE id = ?
                 AND triceratops_fpp IS NULL""",
            (verdict, reason, candidate_id))
        return rowcount > 0

    def mark_fpp_stellar_companion_for_tic(self, tic_id,
                                           max_depth_ppm: float) -> int:
        """Batch-mark this TIC's queue-eligible candidates whose transit is too
        DEEP, for this star, to be a planet.

        *max_depth_ppm* comes from stellar_gates.max_planetary_depth_ppm(R*):
        the deepest transit this particular star could show and still be
        planetary. Passing the bound in (rather than the radius) is what keeps
        this a single UPDATE — the physics is resolved once, per star, in
        Python, and SQL only applies it.

        Same queue predicate as mark_fpp_uncomputable_for_tic (SNR ≥ 10,
        cnn ≥ 0.7 OR ffi ≥ 0.5, fpp IS NULL), so this can never overwrite a
        real FPP score, re-mark an already-gated row, or touch a candidate the
        queue would not have served anyway.

        Deliberately NOT applied to rows with a NULL depth: an unknown depth
        proves nothing, and gating on it would silently convert missing data
        into "eclipsing binary". Fail-open, like every gate here.

        Returns the number of rows marked.
        """
        verdict = fpp_verdict(-1, -1)  # → 'failed' (same sentinel as the rest)
        return self.backend.execute_rowcount(
            """UPDATE candidates
               SET triceratops_fpp = -1,
                   triceratops_nfpp = -1,
                   triceratops_verdict = ?,
                   fpp_skip_reason = 'stellar_companion'
               WHERE tic_id = ?
                 AND triceratops_fpp IS NULL
                 AND depth_ppm IS NOT NULL
                 AND depth_ppm > ?
                 AND (cnn_score >= 0.7 OR exominer_ffi_score >= 0.5)
                 AND snr >= 10.0""",
            (verdict, str(tic_id), float(max_depth_ppm)))

    def mark_fpp_uncomputable_for_tic(self, tic_id) -> int:
        """Batch-mark all FPP-queue-eligible candidates for a TIC as uncomputable.

        Eager FPP-gate companion to mark_fpp_uncomputable (which operates on a
        single candidate_id).  Called from worker_task_result at TIC completion
        when EXONIT_FPP_GATE=1, mirroring the SPOC-gate pattern.

        The sentinel values written (fpp=-1, nfpp=-1, verdict='failed') are
        bit-identical to what the worker's own -1 path stores and to what
        mark_fpp_uncomputable writes, so the lazy gate (in worker_fpp_task) and
        this eager path are always consistent.

        Only touches rows that match the FPP backfill queue predicate
        (snr>=10, (cnn_score>=0.7 OR exominer_ffi_score>=0.5), triceratops_fpp
        IS NULL — OR-gate 2026-07-11, see work_producers.verdict_eligible) so
        we never:
          - overwrite a real (>=0) FPP score,
          - overwrite a previously gate-marked -1 row,
          - or mark candidates that the queue would not have served anyway.

        Returns the number of rows updated (int).
        """
        verdict = fpp_verdict(-1, -1)  # → 'failed'
        return self.backend.execute_rowcount(
            """UPDATE candidates
               SET triceratops_fpp = -1,
                   triceratops_nfpp = -1,
                   triceratops_verdict = ?,
                   fpp_skip_reason = 'no_stellar_mass'
               WHERE tic_id = ?
                 AND triceratops_fpp IS NULL
                 AND (cnn_score >= 0.7 OR exominer_ffi_score >= 0.5)
                 AND snr >= 10.0""",
            (verdict, str(tic_id)))

    def mark_ffi_only_candidates(self, tic_id):
        """Mark cnn>=0.7 candidates for an FFI-only TIC with exominer_score=-1.

        Only updates rows where exominer_score IS NULL (avoids overwriting a
        real score that was written before the gate was active).  Returns the
        number of rows updated.

        Also sets exominer_status='no_spoc' (docs/superpowers/specs/
        exominer-status-column.md — this TIC is confirmed FFI-only, so
        ExoMiner-SPOC does not apply), kept in sync with the -1 write in the
        same statement so the two columns can never drift apart.
        """
        return self.backend.execute_rowcount(
            """UPDATE candidates
               SET exominer_score = -1,
                   exominer_status = 'no_spoc'
               WHERE tic_id = ?
                 AND cnn_score >= 0.7
                 AND exominer_score IS NULL""",
            (str(tic_id),))

    def mark_spoc_no_dv_candidates(self, tic_id):
        """Mark cnn>=0.7 candidates for a has-SPOC-but-no-DV TIC with exominer_score=-1.

        Sibling of ``mark_ffi_only_candidates`` for the OTHER "has_spoc=1"
        case (docs/superpowers/specs/exominer-dv-gate.md): the TIC has 2-min
        SPOC data, but SPOC never published a DV report for it (SPOC only
        runs DV where its own pipeline found a TCE, so most has-SPOC TICs
        don't have one). ExoMiner-SPOC cannot score anything without a DV
        report, so leaving these candidates NULL just means the ExoMiner
        backfill queue tries them, gets a 404 from MAST, and burns the
        round-trip for nothing (see gotcha_exominer_backfill_spoc_gate_hole).
        Marking -1 here, once the gate has confirmed there is no DV, avoids
        that for good.

        Same guard as mark_ffi_only_candidates: only rows where
        exominer_score IS NULL are touched, so a real score is never
        clobbered. Returns the number of rows updated.

        Also sets exominer_status='no_dv' (docs/superpowers/specs/exominer-
        status-column.md), the OTHER -1 bucket: unlike no_spoc, has_spoc IS
        true here — the TIC has 2-min SPOC data, it just never got a DV run.
        """
        return self.backend.execute_rowcount(
            """UPDATE candidates
               SET exominer_score = -1,
                   exominer_status = 'no_dv'
               WHERE tic_id = ?
                 AND cnn_score >= 0.7
                 AND exominer_score IS NULL""",
            (str(tic_id),))

    def clear_spurious_minus_one_candidates(self, tic_id):
        """Retract a spurious exominer_score=-1 sentinel once DV is confirmed.

        Closes the race in docs/superpowers/specs/exominer-dv-gate.md:
        ``mark_unconfirmed_ffi_candidates`` (the bulk sweep) marks -1 for
        any TIC whose ``stars.has_spoc`` was still NULL at
        EXOMINER_BACKFILL entry — a provisional guess, not a fact. If the
        per-TIC gate (``_run_spoc_check_background``) later resolves that
        TIC to has_spoc=1 AND confirms a real DV report, the earlier -1 is
        wrong and must be undone so the candidate re-enters the backfill
        queue (``exominer_score IS NULL AND has_spoc = 1``).

        Only clears rows that are CURRENTLY -1 (idempotent: calling this on
        an already-NULL or already-scored candidate is a no-op), and only
        cnn>=0.7 rows — mirrors the only write paths that ever set -1, so
        this can never touch a real ExoMiner score. Returns the number of
        rows updated (0 in the common case where no spurious -1 was ever
        written for this TIC).

        Also resets exominer_status back to 'pending' (docs/superpowers/
        specs/exominer-status-column.md): the candidate re-enters the
        backfill queue, exactly mirroring exominer_score going back to NULL.
        """
        return self.backend.execute_rowcount(
            """UPDATE candidates
               SET exominer_score = NULL,
                   exominer_status = 'pending'
               WHERE tic_id = ?
                 AND cnn_score >= 0.7
                 AND exominer_score = -1""",
            (str(tic_id),))

    def mark_spoc_no_dv_candidates_for_ids(self, candidate_ids):
        """Sibling of ``mark_spoc_no_dv_candidates``, scoped to explicit
        candidate ids instead of a whole ``tic_id``.

        Fix (docs/superpowers/specs/exominer-dv-gate-sector.md, 2026-07-14):
        a DV report covers a specific sector RANGE (see
        ``exonit.mast_dv.parse_dv_filename`` / ``has_dv_for_sector``) — a TIC
        can have DV for sector 12 and none for sector 14, so the "has SPOC,
        HAS some DV" branch of the SPOC-gate can no longer decide at the
        tic_id granularity: it must resolve, per candidate, whether THAT
        candidate's own ``sector`` is covered. The tic_id-scoped
        ``mark_spoc_no_dv_candidates`` above stays correct and unchanged for
        the OTHER branch (``fetch_dv_runs`` returned ``[]`` — no DV
        anywhere, so no candidate of this TIC can ever be scored regardless
        of sector); this sibling exists for the branch where some DV exists
        but the caller has already worked out, candidate by candidate, which
        ones it does NOT cover.

        Same guard/effect as ``mark_spoc_no_dv_candidates`` (only rows where
        ``exominer_score IS NULL`` are touched, so a real score is never
        clobbered) restricted to ``id IN (...)`` instead of ``tic_id = ?``.
        ``cnn_score >= 0.7`` is re-checked here too (defence in depth — the
        caller is expected to have already filtered on it, but this method
        must never depend on that to stay safe standalone).

        No chunking: called once per completed TIC from the per-TIC
        SPOC-gate, so ``candidate_ids`` is always a handful of rows — never
        the tens-of-thousands-scale backlog ``scripts/fix_exominer_
        spurious_sentinel.py`` chunks at 500. Returns 0 without touching the
        DB for an empty list.
        """
        ids = [int(i) for i in candidate_ids]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        return self.backend.execute_rowcount(
            f"""UPDATE candidates
                SET exominer_score = -1
                WHERE id IN ({placeholders})
                  AND cnn_score >= 0.7
                  AND exominer_score IS NULL""",
            tuple(ids))

    def clear_spurious_minus_one_candidates_for_ids(self, candidate_ids):
        """Sibling of ``clear_spurious_minus_one_candidates``, scoped to
        explicit candidate ids instead of a whole ``tic_id`` — see
        ``mark_spoc_no_dv_candidates_for_ids`` for why the DV-gate decision
        must be per-candidate-sector rather than per-TIC.

        Only clears rows that are CURRENTLY -1 and cnn>=0.7 (same guard as
        the tic_id-scoped sibling), restricted to ``id IN (...)``. No
        chunking, same rationale as the sibling above. Returns 0 without
        touching the DB for an empty list.
        """
        ids = [int(i) for i in candidate_ids]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        return self.backend.execute_rowcount(
            f"""UPDATE candidates
                SET exominer_score = NULL
                WHERE id IN ({placeholders})
                  AND cnn_score >= 0.7
                  AND exominer_score = -1""",
            tuple(ids))

    def mark_unconfirmed_ffi_candidates(self):
        """Bulk-mark exominer_score=-1 for candidates whose star has has_spoc=NULL.

        This is the SPOC-gate bulk clean-up path.  Background:
        - _run_spoc_check_background fires per-TIC during the scan phase
          (post worker_task_result) and marks FFI-only candidates with -1.
        - Under load the SPOC background threads can crash or simply never
          complete, leaving stars with has_spoc=NULL.
        - Those candidates then fall into the ExoMiner backfill selection
          (exominer_score IS NULL AND cnn>=0.7), causing thousands of DV-fetch
          404s that drain the backfill extremely slowly.
        - The ExoMiner SPOC path is useless for FFI-only / unconfirmed TICs
          (they have no DV data product); the result is always -1.0.

        Predicate mirrors mark_ffi_only_candidates but uses a sub-select on
        stars instead of a tic_id argument:
          exominer_score IS NULL          — never overwrite a real score
          AND cnn_score >= 0.7            — only backfill-eligible rows
          AND tic_id IN (stars: has_spoc IS NULL)

        Safety / timing: this method is safe to call once the scan phase has
        fully drained (no more worker_task_result POSTs arriving).  At that
        point the per-TIC SPOC gate has had its chance; any remaining
        has_spoc=NULL rows are gate-failures, not pending checks.  Calling
        it during SCANNING would race with a still-running gate thread that
        might later resolve has_spoc=1 — which is why the caller MUST NOT
        invoke it during SCANNING.  The natural safe hook is the entry to
        EXOMINER_BACKFILL (post-scan, post-FPP_BACKFILL).

        NOTE (docs/superpowers/specs/exominer-dv-gate.md, 2026-07-13): this
        sweep is still needed as a backstop for gate-failures, and is NOT
        changed by that spec. What changed is that a -1 this sweep writes
        on a TIC whose has_spoc later resolves to 1 is no longer permanent:
        if the per-TIC gate then confirms the TIC has a DV report, it calls
        clear_spurious_minus_one_candidates to retract this sweep's -1 and
        put the candidate back in the backfill queue. This sweep itself
        does not need to know that — it keeps writing the same provisional
        -1 it always has.

        Returns the number of rows updated.

        Also sets exominer_status='no_spoc' (docs/superpowers/specs/
        exominer-status-column.md): the derivation rule buckets
        ``has_spoc IS NULL`` together with ``has_spoc = 0`` under 'no_spoc'
        (both mean "not confirmed has-SPOC", the same as this sweep's own
        provisional-guess semantics) — this sweep does not need to
        distinguish them any more than it already didn't for the -1 write.
        """
        return self.backend.execute_rowcount(
            """UPDATE candidates
               SET exominer_score = -1,
                   exominer_status = 'no_spoc'
               WHERE exominer_score IS NULL
                 AND cnn_score >= 0.7
                 AND tic_id IN (
                     SELECT tic_id FROM stars WHERE has_spoc IS NULL
                 )""")

    def update_candidate_exominer(self, candidate_id, score, tce_period=None,
                                  sector_run=None, has_spoc=None):
        """Update ExoMiner score + matched TCE metadata for a candidate.

        This is the "a real ExoMiner score arrives" write point (docs/
        superpowers/specs/exominer-status-column.md): the server-side
        results handler (``exonit.server._run_exominer_background_inner``)
        calls this with a genuine ExoMiner-SPOC probability, never the -1
        sentinel or None in practice — so exominer_status is derived via
        ``derive_exominer_status`` and will be 'scored' on the real path.
        ``has_spoc`` is accepted for completeness (defensive: if a caller
        ever does pass score=-1, the right no_spoc/no_dv bucket needs it),
        defaulting to None like ``derive_exominer_status`` itself.
        """
        status = derive_exominer_status(score, has_spoc=has_spoc)
        self._exec(
            """UPDATE candidates SET exominer_score = ?, exominer_tce_period = ?,
                      exominer_sector_run = ?, exominer_status = ?
               WHERE id = ?""",
            (score, tce_period, sector_run, status, candidate_id))

    def get_candidates_needing_analysis(self, min_snr=10.0, min_cnn=0.7,
                                        claude_only=False):
        """Get candidates that pass criteria but lack analysis.

        If claude_only=True, only requires missing Claude verdict (ignores FPP).
        """
        if claude_only:
            fpp_clause = ""
        else:
            fpp_clause = """AND (
                    c.triceratops_fpp IS NULL
                    OR (
                        c.triceratops_fpp >= 0
                        AND c.triceratops_fpp < 0.5
                        AND c.triceratops_nfpp IS NULL
                    )
                 )"""
        # c.snr is nullable — B5 dialect-drift fix, see
        # exonit.db_backend.Dialect.nulls_last.
        order = self.dialect.nulls_last("c.snr")
        return self._query(
            f"""SELECT c.*, s.ra, s.dec, s.tmag, s.n_sectors, s.sectors_observed
               FROM candidates c
               JOIN stars s ON c.tic_id = s.tic_id
               WHERE c.snr >= ? AND c.cnn_score >= ?
                 {fpp_clause}
                 AND c.claude_verdict IS NULL
                 AND s.status = 'completed'
               ORDER BY {order}""",
            (min_snr, min_cnn))

    def _strong_where(self, threshold: float, min_snr: float = None):
        """Build the "strong tier" WHERE predicate + its bind params.

        D1 (docs/reviews/2026-07-02-opus48-cleanup-review.md): single source
        of truth for the strong-tier predicate, shared by ``strong_candidates``
        and ``count_strong_candidates`` below (previously two hand-built
        copies — the family that shipped the LEAST-vs-min dialect bug). A
        third copy lives in server.py's ``/api/candidates?filter=strong``
        (server.py:~1832); that one is out of scope for this change (a
        follow-up will repoint it at this builder) but is designed to be
        callable the same way: ``where, params = db._strong_where(threshold,
        min_snr)``.

        A candidate passes if:
          1. cnn_score >= 0.7 OR exominer_ffi_score >= 0.5 (OR-gate,
             2026-07-11: docs/reviews/injection_e2e_results_2026-07-11.md +
             docs/reviews/ffi_cnn_investigation_2026-07-11.md §5 — mirrors
             the verdict/FPP eligibility gate, see
             work_producers.verdict_eligible, so em_ffi-rescued candidates
             that reach ``claude_verdict='planet_candidate'`` are visible
             in this display tier too, not silently hidden behind a
             stricter predicate than the one that let them get a verdict)
          2. triceratops_fpp <= 0.5 OR NULL  (NULL = pending — same
             convention as the strong-tier filters in server.py; line-number
             references removed, they rot on every edit)
          3. At least one of exominer_score / exominer_ffi_score is >= 0.7
          4. If BOTH scores are present, the smaller of the two must be >=
             *threshold* (agreement gate). Rows where one score passes 0.7
             but the other dips below that threshold are excluded
             (disagreement → not trustworthy enough for display).

        ``exominer_score`` carries a ``-1`` sentinel (no-DV cache-negative,
        written by the has_spoc / no-metadata / bulk-FFI-only gates — see
        progress/investigation-exominer-minus1-2026-07-02.md) for ~61% of
        rows. It is NOT a real score, so every reference below wraps the
        column in ``NULLIF(exominer_score, -1)`` — turns it back into SQL
        NULL, which "at least one present" (3) and "both present" (4)
        already treat as absent, so a -1 row with a real
        ``exominer_ffi_score`` now falls back to the FFI score instead of
        hard-vetoing the candidate out of the strong tier. Reads only — the
        writers keep storing the raw -1 (queue-selection ``exominer_score IS
        NULL`` checks elsewhere are load-bearing for backfill drain and must
        keep seeing the literal -1 as "not NULL", i.e. "already handled").

        Optional min_snr filter for callers that need to enforce SNR (e.g.
        the display endpoint uses min_snr=10).

        Returns ``(sql_fragment, params)`` — sql_fragment is a WHERE-clause
        body (no leading ``WHERE``), params is the bind-param tuple in the
        order its ``?`` placeholders appear.
        """
        exom = "NULLIF(exominer_score, -1)"
        least = self.dialect.least(exom, "exominer_ffi_score")
        snr_clause = "AND snr >= ?" if min_snr is not None else ""
        sql = f"""(cnn_score >= 0.7 OR exominer_ffi_score >= 0.5)
              AND (triceratops_fpp IS NULL OR triceratops_fpp <= 0.5)
              AND COALESCE({exom}, exominer_ffi_score) >= 0.7
              AND (
                {exom} IS NULL
                OR exominer_ffi_score IS NULL
                OR {least} >= ?
              )
              {snr_clause}"""
        params = (threshold, min_snr) if min_snr is not None else (threshold,)
        return sql, params

    def strong_candidates(self, min_snr: float = None):
        """Return candidates in the strong-tier display with the agreement gate.

        See ``_strong_where`` for the predicate. The threshold is re-read
        from the environment on each call so that live server restarts
        without a code deploy can update the gate.
        """
        threshold = float(os.environ.get(
            "EXONIT_PHASE3_AGREEMENT_THRESHOLD", "0.6"))
        where, params = self._strong_where(threshold, min_snr)
        return self._query(f"SELECT * FROM candidates WHERE {where}", params)

    def count_strong_candidates(self, min_snr: float = None) -> int:
        """COUNT of strong-tier candidates — same predicate as strong_candidates.

        /api/stats only needs the count, but used to call strong_candidates()
        (``SELECT *``) and ``len()`` the result, which scanned the ~355k
        cnn>=0.7-or-em_ffi>=0.5 rows (~0.9 s). This COUNT(*) is planned
        through the partial index ``idx_cand_strong`` (which encodes
        ``(cnn_score>=0.7 OR exominer_ffi_score>=0.5) AND
        COALESCE(NULLIF(exominer_score,-1),exominer_ffi_score)>=0.7`` — the
        selective part, kept in lockstep with ``_strong_where`` above; see
        the NULLIF note there), so it returns in ~0.1 ms. The remaining
        clauses (FPP, agreement gate, snr) are applied on the tiny indexed
        set. Result is identical.

        CAUTION (deploy note): the index predicate must textually match the
        query's WHERE clause for a planner to use it (neither SQLite nor PG
        proves arbitrary predicate implication across a NULLIF rewrite). The
        SQLite migration below DROPs+recreates ``idx_cand_strong`` with the
        OR-gate form on next startup (self-healing; 2026-07-11, same pattern
        as the NULLIF rewrite of 2026-07-02). PostgreSQL is schema-managed
        out of band (``scripts/migration/schema_pg.sql``, applied once, not
        idempotent) — production's live index still has the OLD predicate
        until someone runs the equivalent ``DROP INDEX`` + ``CREATE INDEX``
        by hand (see the updated statements in schema_pg.sql and the
        implementer handoff progress/impl-orgate.md). Until then this query
        silently falls back to a full scan of the ``candidates`` table
        (millions of rows) on every ``/api/stats`` poll instead of the
        ~0.1 ms index seek — see the rollout notes in
        progress/impl-minus1-fix-2026-07-02.md for the shape of that
        historical rollout.
        """
        threshold = float(os.environ.get(
            "EXONIT_PHASE3_AGREEMENT_THRESHOLD", "0.6"))
        where, params = self._strong_where(threshold, min_snr)
        return self._query_scalar(
            f"SELECT COUNT(*) FROM candidates WHERE {where}", params)

    # --- completeness writers ---
    # Both methods use INSERT ... ON CONFLICT DO UPDATE (upsert) instead of
    # INSERT OR REPLACE.  The difference matters for completeness_agg accuracy:
    # INSERT OR REPLACE internally deletes the conflicting row then re-inserts,
    # but SQLite only fires the DELETE trigger for the displaced row when
    # PRAGMA recursive_triggers=ON (off by default).  With ON CONFLICT DO UPDATE
    # SQLite fires only the AFTER INSERT trigger on a new row or the AFTER UPDATE
    # trigger on an existing one — both are exactly the triggers that maintain
    # completeness_agg correctly.

    _INSERT_COMPLETENESS_SQL = """
        INSERT INTO completeness
               (tic_id, radius_earth, period_days, detection_prob)
               VALUES (?, ?, ?, ?)
        ON CONFLICT(tic_id, radius_earth, period_days)
        DO UPDATE SET detection_prob = excluded.detection_prob"""

    def add_completeness(self, tic_id, radius_earth, period_days, detection_prob):
        """Insert (flag OFF: upsert) a single completeness cell.

        Flag ON (Phase 4a): delegates to ``add_completeness_many``'s PG
        staging append (as a one-cell list) — ``scripts/
        injection_probe_driver.py`` is the one production caller of this
        singular form, and it must not crash under
        ``EXONIT_COMPLETENESS_PG=1``.
        """
        if self._cmpl_pg:
            self.add_completeness_many(
                tic_id, [(radius_earth, period_days, detection_prob)])
            return
        # Completeness always lives on the dedicated SQLite connection (the agg
        # triggers are SQLite triggers), even in PG mode (flag OFF).
        with self._cmpl_lock:
            self._cmpl_conn.execute(
                self._INSERT_COMPLETENESS_SQL,
                (tic_id, radius_earth, period_days, detection_prob))
            self._cmpl_conn.commit()

    def add_completeness_many(self, tic_id, cells):
        """Insert many completeness cells in ONE transaction (one commit).

        ``cells``: iterable of ``(radius_earth, period_days, detection_prob)``.

        Flag OFF (default): identical to calling ``add_completeness`` per cell
        (upsert on the ``(tic_id, radius_earth, period_days)`` primary key),
        but the worker's ~72-cell injection-recovery grid is persisted with a
        single commit instead of ~72 — the dominant source of WAL churn /
        write-lock contention per worker result during a run.

        Flag ON (``EXONIT_COMPLETENESS_PG=1``, Phase 4a): pure append into PG
        ``completeness_staging`` — no ON CONFLICT/upsert; the same
        (tic_id, radius_earth, period_days) key can appear more than once
        across calls, and last-write-wins dedup is the compaction job's job
        (Task 3), not this write path's. Still ONE transaction (one pooled
        connection checkout via ``self.backend.executemany``).

        Empty input is a no-op in both modes.
        """
        rows = [(tic_id, r, p, d) for (r, p, d) in cells]
        if not rows:
            return
        if self._cmpl_pg:
            self.backend.executemany(
                "INSERT INTO completeness_staging "
                "(tic_id, radius_earth, period_days, detection_prob) "
                "VALUES (?, ?, ?, ?)", rows)
            return
        with self._cmpl_lock:
            self._cmpl_conn.executemany(self._INSERT_COMPLETENESS_SQL, rows)
            self._cmpl_conn.commit()

    def query_completeness_agg(self):
        """Read the completeness heatmap aggregate (one row per (radius, period)).

        Flag OFF (default): ALWAYS reads from the dedicated SQLite connection
        (``self._cmpl_conn``), in both SQLite and PG *main-backend* mode: the
        ``completeness``/``completeness_agg`` subsystem and its triggers never
        move to PostgreSQL in this mode (hybrid backend, see the module
        docstring / the design spec). Routing this through the generic backend
        would, in PG mode, hit the empty PG ``completeness_agg`` table instead
        of the real SQLite aggregate — hence this explicit method.

        Flag ON (Phase 4a): reads the PG ``completeness_agg_bucket`` table,
        summed across buckets per (radius, period) cell — the per-bucket
        split is a write-side sharding detail invisible to readers.

        Returns a list of dicts with keys ``radius_earth``, ``period_days``,
        ``avg_prob`` (= sum_prob / n) and ``n_stars`` (= n), ordered by
        (radius_earth, period_days) — the exact shape the dashboard expects,
        identical in both modes (keys, key ORDER and value TYPES — Task 5's
        ``tests/test_pg_parity_completeness.py`` pins this contract byte for
        byte against the flag-OFF/SQLite response). ``n_stars`` is explicitly
        cast to ``BIGINT``: PostgreSQL's ``SUM()`` over a ``bigint`` column
        promotes to ``numeric`` (overflow-safety, not a bug), which psycopg3
        maps to ``decimal.Decimal`` — the flag-OFF/SQLite side returns a
        plain ``int`` for the same field, so left uncast this silently broke
        the cross-backend API-parity contract the UI heatmap relies on.
        """
        if self._cmpl_pg:
            return self.backend.query(
                """SELECT radius_earth, period_days,
                          CAST(SUM(sum_prob) AS DOUBLE PRECISION) / SUM(n) AS avg_prob,
                          CAST(SUM(n) AS BIGINT) AS n_stars
                   FROM completeness_agg_bucket
                   GROUP BY radius_earth, period_days
                   ORDER BY radius_earth, period_days""")
        with self._cmpl_lock:
            cur = self._cmpl_conn.execute(
                """SELECT radius_earth, period_days,
                          CAST(sum_prob AS REAL) / n AS avg_prob,
                          n AS n_stars
                   FROM completeness_agg
                   ORDER BY radius_earth, period_days""")
            return [dict(r) for r in cur.fetchall()]

    def count_completeness_tics(self):
        """Exact count of distinct TICs with completeness data (Phase 4a, for
        the scorecard's ``distinct_completeness_tics`` — wired in Task 4).

        Flag ON: combines the compacted ``completeness_tics`` index with any
        ``tic_id`` present in ``completeness_staging`` that hasn't been
        compacted yet, without double-counting the overlap — exact even
        mid-compaction (Task 3 owns compaction, not db.py).

        Flag OFF: the legacy SQLite ``completeness`` table (same concept,
        pre-Phase-4a shape) — kept symmetric so a caller (e.g. the scorecard)
        can call this unconditionally regardless of the flag.
        """
        if self._cmpl_pg:
            return self.backend.query_scalar(
                """SELECT (SELECT COUNT(*) FROM completeness_tics)
                        + (SELECT COUNT(DISTINCT s.tic_id)
                             FROM completeness_staging s
                            WHERE NOT EXISTS (
                                SELECT 1 FROM completeness_tics t
                                 WHERE t.tic_id = s.tic_id))""")
        with self._cmpl_lock:
            return self._cmpl_conn.execute(
                "SELECT COUNT(DISTINCT tic_id) FROM completeness").fetchone()[0]

    def wal_checkpoint(self, mode="TRUNCATE"):
        """Run a WAL checkpoint and return ``(busy, log_frames, checkpointed)``.

        WAL auto-checkpoint is PASSIVE (recycles frames but never shrinks the
        ``-wal`` file), so across runs the file creeps up to its high-water mark
        (observed: 156 MB). A TRUNCATE checkpoint shrinks it back to 0 once no
        other connection still needs the old frames — best run when the DB is
        idle (end of run / after STOP_WORKERS), never mid-run on a hot path.
        ``busy`` is 1 if a reader blocked a full truncate (it still reclaims what
        it can — safe, just not complete). Safe to call anytime; SQLite degrades
        gracefully.

        Flag ON (Phase 4a): there is no SQLite sidecar left to checkpoint —
        this becomes a no-op (DEBUG log), returning ``None``.
        """
        if self._cmpl_pg:
            logger.debug(
                "wal_checkpoint: no-op (EXONIT_COMPLETENESS_PG=1, "
                "completeness SQLite sidecar retired)")
            return None
        mode = (mode or "TRUNCATE").upper()
        if mode not in ("PASSIVE", "FULL", "RESTART", "TRUNCATE"):
            mode = "TRUNCATE"
        # The WAL belongs to the SQLite connection (the operational one in SQLite
        # mode, the completeness one in PG mode — self.conn aliases whichever is
        # the SQLite connection here).  PG manages its own WAL: the operational
        # backend checkpoint is a no-op.
        with self._cmpl_lock:
            row = self._cmpl_conn.execute(
                f"PRAGMA wal_checkpoint({mode})").fetchone()
            return tuple(row) if row is not None else None

    # --- Task queue methods ---

    def add_task(self, task_id, tic_id, priority=2, sector=None):
        # sector is an ADDITIVE column (done-state design, 2026-09-26): the
        # INSERT only references it when a caller actually passes one, so
        # every pre-existing call site (sector never given) keeps emitting
        # the EXACT pre-existing 3-column SQL — no dependency on the
        # tasks.sector migration having been applied yet (deploy order:
        # spec §5 commits this code BEFORE the DDL step). A caller that DOES
        # pass sector= is opting into the new column and requires the
        # migration to be live.
        if sector is None:
            self._exec(
                "INSERT INTO tasks (task_id, tic_id, priority) "
                "VALUES (?, ?, ?)",
                (task_id, tic_id, priority))
        else:
            self._exec(
                "INSERT INTO tasks (task_id, tic_id, priority, sector) "
                "VALUES (?, ?, ?, ?)",
                (task_id, tic_id, priority, sector))

    def get_task(self, task_id):
        return self._query_one("SELECT * FROM tasks WHERE task_id = ?", (task_id,))

    def get_task_sector(self, task_id):
        """tasks.sector for *task_id* (None if unknown/NULL/column absent)."""
        if not task_id:
            return None
        try:
            row = self._query_one("SELECT sector FROM tasks WHERE task_id = ?",
                                  (task_id,))
        except Exception:  # legacy DB without the tasks.sector migration
            return None
        return row["sector"] if row and row["sector"] is not None else None

    def assign_task(self, task_id, worker_id):
        """Mark task as assigned and increment attempts counter."""
        self._exec(
            "UPDATE tasks SET status='assigned', assigned_to=?, assigned_at=?, "
            "attempts=COALESCE(attempts,0)+1 WHERE task_id=?",
            (worker_id, utc_now_iso(), task_id))

    def complete_task(self, task_id, processing_time, gpu_used=0,
                      lc_load_recovered=False):
        """Mark a task as completed.

        ``lc_load_recovered``: True when the worker recovered from a SPOC LC
        load failure by falling back to FFI from the server's Phase 2.5
        cache (worker.py::_try_ffi_fallback). Persisted as 0/1 in the
        ``tasks.lc_load_recovered`` column for analytics. Defaults to False
        so existing callers and old worker payloads remain compatible.
        """
        self._exec(
            """UPDATE tasks SET status='completed', completed_at=?,
               processing_time=?, gpu_used=?, lc_load_recovered=?
               WHERE task_id=?""",
            (utc_now_iso(), processing_time,
             self.dialect.bool_param(gpu_used),
             self.dialect.bool_param(lc_load_recovered), task_id))

    def requeue_or_fail_task(self, task_id, error=None, max_attempts=MAX_TASK_ATTEMPTS):
        """Failure handler: requeue the task if it still has attempts left,
        otherwise mark it as definitively failed so no worker picks it up again.

        Returns the final status ('pending' or 'failed')."""
        with self.backend.transaction() as conn:
            row = self.backend.tx_query_one(
                conn, "SELECT attempts FROM tasks WHERE task_id=?", (task_id,))
            if row is None:
                return None
            attempts = row["attempts"] or 0
            if attempts >= max_attempts:
                self.backend.tx_execute(
                    conn,
                    "UPDATE tasks SET status='failed', last_error=?, "
                    "completed_at=? WHERE task_id=?",
                    (error, utc_now_iso(), task_id))
                final = "failed"
            else:
                self.backend.tx_execute(
                    conn,
                    "UPDATE tasks SET status='pending', assigned_to=NULL, "
                    "assigned_at=NULL, last_error=? WHERE task_id=?",
                    (error, task_id))
                final = "pending"
            return final

    def fail_task(self, task_id):
        self._exec("UPDATE tasks SET status='failed' WHERE task_id=?", (task_id,))

    def get_pending_task(self):
        return self._query_one(
            "SELECT * FROM tasks WHERE status='pending' ORDER BY priority ASC, created_at ASC LIMIT 1")

    def get_and_assign_task(self, worker_id):
        # Atomic SELECT+UPDATE: splitting them released the lock between
        # statements and let two workers pick up the same pending task
        # (observed: TIC 441739020 claimed by 3 workers at once).  SQLite gets
        # atomicity from the global lock; PG needs FOR UPDATE SKIP LOCKED so two
        # uvicorn processes never claim the same row.
        lock_clause = self.dialect.for_update_skip_locked()
        with self.backend.transaction() as conn:
            task = self.backend.tx_query_one(
                conn,
                "SELECT * FROM tasks WHERE status='pending' "
                "ORDER BY priority ASC, created_at ASC LIMIT 1"
                + lock_clause)
            if task is None:
                return None
            now = utc_now_iso()
            self.backend.tx_execute(
                conn,
                "UPDATE tasks SET status='assigned', assigned_to=?, "
                "assigned_at=?, attempts=COALESCE(attempts,0)+1 "
                "WHERE task_id=?",
                (worker_id, now, task["task_id"]))
            task["status"] = "assigned"
            task["assigned_to"] = worker_id
            task["assigned_at"] = now
            task["attempts"] = (task.get("attempts") or 0) + 1
            return task

    def get_task_row_and_assign(self, task_id, worker_id):
        """Claim-time PK lookup + assign for the primary work-stream path
        (Phase 1, Task 5): given a task_id claimed from the "primary" work
        stream, verify it is still assignable and, if so, perform the SAME
        assigned_to/assigned_at/attempts+1 UPDATE get_and_assign_task does —
        so the PG mirror this path writes is byte-identical to the legacy
        scan's, for downstream UI/stats/orphan-visibility consumers that
        read the tasks table directly.

        Accepts status IN ('pending', 'assigned') — NOT 'pending' only.
        This is not just a fresh-dequeue path: the stream's own claim()
        also STEALS stale PEL entries from a dead/moved-on worker (its
        keep_if callback, _worker_alive_and_on_current_tic, already vetted
        that steal), and a stolen item's PG row is still 'assigned' (to the
        PREVIOUS owner) at this point — that is exactly the orphan-recovery
        case this method must re-assign, not reject. Rejecting 'assigned'
        here would ack-and-drop every legitimately-stolen task instead of
        reassigning it, defeating the whole replacement for
        reassign_stale_tasks. Only the two TERMINAL statuses
        ('completed'/'failed' — set by a real result POST that raced this
        steal, or a row that was already finished by the time it was
        claimed) must be rejected; 'pending'/'assigned' are the only two
        non-terminal statuses this column ever takes (see the CREATE TABLE
        default and every UPDATE ... SET status= site in this module).

        Returns None when the row is missing OR in a terminal status
        (completed/failed) — the stream caller should ack the message and
        move on rather than retry (a genuine late race: the presumed-
        stealable worker actually posted a real result between the steal
        decision and this SELECT). Returns the updated row dict on success.

        task_id is the tasks table's TEXT primary key (secrets.token_hex,
        assigned by the enqueue endpoint in Python — never a DB-generated
        integer id), so no int() cast is needed before binding, unlike the
        FPP/ExoMiner streams' candidate_id (an INTEGER PK).
        """
        lock_clause = self.dialect.for_update_skip_locked()
        with self.backend.transaction() as conn:
            row = self.backend.tx_query_one(
                conn, "SELECT * FROM tasks WHERE task_id=?" + lock_clause,
                (task_id,))
            if row is None or row["status"] not in ("pending", "assigned"):
                return None
            now = utc_now_iso()
            self.backend.tx_execute(
                conn,
                "UPDATE tasks SET status='assigned', assigned_to=?, "
                "assigned_at=?, attempts=COALESCE(attempts,0)+1 "
                "WHERE task_id=?",
                (worker_id, now, task_id))
            row["status"] = "assigned"
            row["assigned_to"] = worker_id
            row["assigned_at"] = now
            row["attempts"] = (row.get("attempts") or 0) + 1
            return row

    def reconcile_dlq_task(self, task_id, max_attempts=MAX_TASK_ATTEMPTS):
        """Status-guarded counterpart to requeue_or_fail_task, used ONLY by
        the primary work-stream's DLQ reconciliation (Phase 1, Task 5 — see
        routes/workers.py's _reconcile_primary_dlq).

        A task whose stream message exhausted max_deliveries is moved to
        the DLQ and its message removed — NO future claim() call will ever
        touch that task_id again, so its PG row (left status='assigned' by
        the last successful get_task_row_and_assign UPDATE) would stay
        assigned forever without an explicit reconciliation. Reconciliation
        runs opportunistically against a PEEKED (never popped) DLQ list, so
        the SAME entry can be re-scanned indefinitely: this method
        re-checks status='assigned' INSIDE the same transaction as the
        write (not just before calling it), so a legitimate late completion
        that landed between scans is never stomped back to pending/failed
        — whichever writer's transaction commits second simply wins, and a
        real completion always wins because this method becomes a no-op
        the instant status stops being 'assigned'.

        Returns 'pending'/'failed' (mirrors requeue_or_fail_task's return
        contract), or None if the row is missing or already not 'assigned'
        (nothing to do — safe, idempotent no-op).

        Honest note on which branch actually fires (G4, review package
        2026-07-05): the two branches read as symmetric, but under the
        primary stream's DEFAULT tuning
        (work_streams.STAGE_SETTINGS["primary"].max_deliveries == 3 ==
        MAX_TASK_ATTEMPTS), the 'pending' branch is structurally
        unreachable in production. Every successful claim of a primary
        task (fresh dequeue or a legitimate steal) runs
        get_task_row_and_assign, which bumps this row's `attempts` in
        lockstep with the stream's own delivery count — so by the time a
        message has been delivered `max_deliveries` times (the DLQ-move
        trigger), `attempts` has, by construction, also reached
        `max_deliveries` (== `max_attempts` here by default). 'failed' is
        the branch that actually fires in practice; 'pending' is only
        reachable if an operator tunes max_deliveries below
        MAX_TASK_ATTEMPTS, or via a test that inflates stream deliveries
        directly through ws.claim() without a matching
        get_task_row_and_assign call (see
        tests/test_primary_stream_handler.py's DLQ suite, which exercises
        exactly that scenario to reach this branch at all).
        """
        with self.backend.transaction() as conn:
            row = self.backend.tx_query_one(
                conn,
                "SELECT status, attempts FROM tasks WHERE task_id=?"
                + self.dialect.for_update_skip_locked(),
                (task_id,))
            if row is None or row["status"] != "assigned":
                return None
            attempts = row["attempts"] or 0
            error = "dlq_exhausted_max_deliveries"
            if attempts >= max_attempts:
                self.backend.tx_execute(
                    conn,
                    "UPDATE tasks SET status='failed', last_error=?, "
                    "completed_at=? WHERE task_id=?",
                    (error, utc_now_iso(), task_id))
                return "failed"
            self.backend.tx_execute(
                conn,
                "UPDATE tasks SET status='pending', assigned_to=NULL, "
                "assigned_at=NULL, last_error=? WHERE task_id=?",
                (error, task_id))
            return "pending"

    def reassign_stale_tasks(self, timeout_seconds, alive_worker_ids=None,
                             worker_current_tic=None, abandon_timeout_seconds=None,
                             collect_ids=False):
        """Reassign tasks that have been assigned for longer than timeout_seconds.

        ``collect_ids`` (G1, review package 2026-07-05): when True, return
        the list of ``(task_id, tic_id)`` tuples for every row this flipped
        'assigned' -> 'pending', instead of the bare count. Added for the
        primary work stream's defense-in-depth sweep
        (routes/workers.py's ``_primary_sweep_stale_assignments``), which
        needs to know WHICH rows were reset so it can enqueue a fresh
        stream message for each one — a bare count cannot drive that.
        Default False preserves the original int-count contract byte-for-
        byte for the pre-existing legacy call site (routes/workers.py's
        ``worker_get_task``) and every existing caller/test.

        Decision logic (applied in order for each task older than timeout_seconds):

        1. age > abandon_timeout_seconds (if provided) → REASSIGN unconditionally.
           Hard ceiling: no legitimate single TIC takes this long; catches a
           worker that is genuinely stuck on the same TIC indefinitely.

        2. Worker not in alive_worker_ids (if provided) → REASSIGN.
           Classic case: worker went offline.

        3. worker_current_tic provided AND current_tic == this task's tic_id → SKIP.
           Worker is actively processing this task right now.

           GIL-safety note: the heartbeat is GIL-starved during numba/injection
           blocks that can last 5-7 min, so current_tic may briefly lag.  We only
           reach this function for tasks older than timeout_seconds (default 1800s
           = 30 min).  Any block-GIL event lasting 5-7 min would have resolved
           within 23+ min of a 30-min-old task; at that age current_tic is
           reliable.  This is the same reasoning that justifies WORKER_ALIVE_WINDOW
           = 600s: we trust heartbeats up to 10 min old, and at 30 min they are
           definitively stale or actively updated.

        4. worker_current_tic provided AND current_tic != this task's tic_id
           (includes current_tic == None, i.e. worker is idle) → REASSIGN.
           Worker is alive but has moved on; the task is an orphan.  This is the
           primary bug this function was extended to fix: a worker that lost a
           result-POST and continued to the next TIC left the old task as
           'assigned' forever because the old guard (step 2 above) only checked
           liveness, not whether the worker was still on *this* TIC.

        5. worker_current_tic is None (caller did not provide the map) → fall back
           to legacy behaviour: skip if alive, reassign if dead.  This preserves
           backward-compatibility for callers that do not pass the new param and
           for existing tests that rely on the old semantics.

        tic_id comparison normalises both sides to str() so that an int stored
        in workers_state (e.g. from a JSON-decoded heartbeat that lost the string
        type) never causes a spurious false-mismatch that would reassign a task
        that is genuinely in-flight.
        """
        with self.backend.transaction() as conn:
            # tz-aware "now" (not just system-local .timestamp()) so this cutoff
            # stays correct independent of the host's configured timezone —
            # matches the tz-aware UTC values utc_now_iso() now writes to
            # assigned_at, which _to_epoch() below parses back correctly either
            # way (a tz-aware datetime's .timestamp() is TZ-independent).
            now_epoch = datetime.now(timezone.utc).timestamp()
            cutoff = now_epoch - timeout_seconds
            abandon_cutoff = (
                now_epoch - abandon_timeout_seconds
                if abandon_timeout_seconds is not None else None
            )
            rows = self.backend.tx_query(
                conn,
                "SELECT task_id, tic_id, assigned_at, assigned_to"
                " FROM tasks WHERE status='assigned'")
            reset = 0
            reset_rows = []
            for row in rows:
                if not row["assigned_at"]:
                    continue
                assigned_ts = _to_epoch(row["assigned_at"])
                if assigned_ts is None:
                    continue
                if assigned_ts >= cutoff:
                    # Below the stale threshold — never reassign regardless.
                    continue

                assigned_to = row["assigned_to"]
                task_tic = str(row["tic_id"]) if row["tic_id"] is not None else None

                # --- Branch 1: absolute time ceiling (worker stuck) ---
                if abandon_cutoff is not None and assigned_ts < abandon_cutoff:
                    pass  # fall through to reassign below

                # --- Branch 2: worker offline ---
                elif alive_worker_ids is not None and assigned_to not in alive_worker_ids:
                    pass  # fall through to reassign below

                # --- Branch 3: worker alive AND on this exact TIC → keep ---
                elif (worker_current_tic is not None
                      and _tic_str(worker_current_tic.get(assigned_to)) == task_tic):
                    continue

                # --- Branch 4: worker alive but on a different TIC → reassign ---
                elif worker_current_tic is not None:
                    pass  # fall through to reassign below

                # --- Branch 5: legacy / no current-tic map provided ---
                else:
                    if alive_worker_ids is not None and assigned_to in alive_worker_ids:
                        # Worker still alive — trust it's making progress (old behaviour).
                        continue
                    # Worker dead (or alive_worker_ids not provided) → reassign.

                self.backend.tx_execute(
                    conn,
                    "UPDATE tasks SET status='pending', assigned_to=NULL, assigned_at=NULL"
                    " WHERE task_id=?", (row["task_id"],))
                reset += 1
                if collect_ids:
                    reset_rows.append((row["task_id"], row["tic_id"]))
            return reset_rows if collect_ids else reset

    def get_task_stats(self):
        rows = self._query("SELECT status, COUNT(*) as cnt FROM tasks GROUP BY status")
        return {r["status"]: r["cnt"] for r in rows}

    #: Shared-state keys for get_task_stats_cached() below.
    _TASK_STATS_CACHE_KEY = "cache:task_stats"
    _TASK_STATS_LOCK_KEY = "cache:task_stats:refresh"
    #: Generous vs. the ~2 s query it guards — only ever expires early if the
    #: process holding it dies mid-query, in which case the next stale poll
    #: just retries the lock.
    _TASK_STATS_LOCK_TTL_S = 120.0

    def get_task_stats_cached(self, ttl_s: float | None = None) -> dict:
        """Cross-process cached ``get_task_stats()``, for UI/status display
        callers ONLY (2026-09-25, fix A).

        ``SELECT status, COUNT(*) FROM tasks GROUP BY status`` is a parallel
        seq scan of the whole ``tasks`` table — 34.2 M rows / 8 GB measured
        on production, 1.84 s and 3 cores per call. Before this method
        existed, every UI-facing caller ran that query independently on its
        OWN short cache: ``/api/status`` (admin.py, 3 s outer TTL),
        ``/api/worker/list`` (workers.py, 5 s outer TTL), and ``/api/stats``
        + ``/api/downloads/status`` (both via ``get_stats()`` below) — four
        call sites, each recomputing the SAME aggregate on its own cadence,
        on a 4-process (A3) server. This wraps the raw query in ONE
        shared-state entry so all of them share a single result and a single
        recompute cadence, independent of how tight any individual
        endpoint's own outer cache is.

        TTL defaults to 60 s (``EXONIT_TASK_STATS_CACHE_TTL_S``) — these are
        cumulative lifetime counters in the millions; a minute of staleness
        is invisible on a dashboard, unlike the multi-second stall the query
        itself causes when run on every poll.

        Single-flight: on a stale entry, only the process that wins the
        ``set_if_absent`` lock recomputes (synchronously — the query is what
        the caller was already going to pay for on a cache miss, no
        background thread needed); every other process serves the stale
        value. A COLD cache (nothing stored yet, e.g. right after a server
        restart) always computes inline — there is nothing to serve while
        the first-ever compute is in flight, thundering-herd on that single
        occasion is an accepted cost (matches ``exonit/server.py``'s
        ``_cached_json`` cold-cache behaviour).

        No caller of ``get_task_stats()`` needs a guaranteed-fresh value as
        of this writing (grep confirms admin.py/workers.py's two UI routes
        and ``get_stats()`` below are the only callers) — if one is ever
        added (e.g. an orchestrator phase-exit check), it MUST call
        ``get_task_stats()`` directly; this method is an opt-in display
        cache, not a drop-in replacement.

        Degrades to the direct (uncached) query whenever shared-state is
        unavailable (no Redis configured, or a broken Redis at runtime) —
        matching every other shared-state usage in this codebase (e.g.
        ``exonit/routes/discovery.py``'s ``_get_downloaded_counts``).
        """
        if ttl_s is None:
            try:
                ttl_s = float(os.environ.get("EXONIT_TASK_STATS_CACHE_TTL_S", "60"))
            except ValueError:
                ttl_s = 60.0

        from exonit.shared_state import get_shared_state
        try:
            st = get_shared_state()
            raw = st.get(self._TASK_STATS_CACHE_KEY)
        except Exception:
            return self.get_task_stats()

        entry = None
        if raw:
            try:
                entry = json.loads(raw)
            except (TypeError, ValueError):
                entry = None

        now = time.time()
        if entry is not None and (now - entry.get("ts", 0.0)) <= ttl_s:
            return entry["stats"]

        if entry is None:
            # Nothing to serve — pay for it now and seed the cache.
            stats = self.get_task_stats()
            self._write_task_stats_cache(stats)
            return stats

        # Stale but present: single-flight the recompute.
        try:
            got_lock = st.set_if_absent(
                self._TASK_STATS_LOCK_KEY, "1", ttl=self._TASK_STATS_LOCK_TTL_S)
        except Exception:
            got_lock = False
        if not got_lock:
            return entry["stats"]

        try:
            stats = self.get_task_stats()
            self._write_task_stats_cache(stats)
            return stats
        finally:
            try:
                st.delete(self._TASK_STATS_LOCK_KEY)
            except Exception:
                pass

    def _write_task_stats_cache(self, stats: dict) -> None:
        try:
            from exonit.shared_state import get_shared_state
            get_shared_state().set(
                self._TASK_STATS_CACHE_KEY,
                json.dumps({"ts": time.time(), "stats": stats}))
        except Exception:
            pass  # best-effort — a failed write just means the next call recomputes.

    def get_stats(self):
        # UI-only aggregate (see /api/stats, /api/downloads/status) — the
        # cached variant is safe here for the same reason described in
        # get_task_stats_cached()'s docstring.
        task_counts = self.get_task_stats_cached()
        return {
            "total_stars": self._query_scalar(
                "SELECT COUNT(*) FROM stars WHERE status = 'completed'"),
            "total_candidates": self._query_scalar("SELECT COUNT(*) FROM candidates"),
            "pending_tasks": task_counts.get("pending", 0),
            "assigned_tasks": task_counts.get("assigned", 0),
            "completed_tasks": task_counts.get("completed", 0),
            "failed_tasks": task_counts.get("failed", 0),
        }

    # --- Telemetry methods ---

    def insert_telemetry(self, worker_id, data):
        """Insert a heartbeat telemetry record and refresh the worker's
        ``last_seen`` timestamp in ``workers_registry``.

        Both happen under the same lock+commit so the registry can't drift
        out of sync with the telemetry stream. Before this refresh was
        added, ``last_seen`` only moved on re-register — a worker running
        for hours would show a stale value from its start time, which made
        the dashboard and ``workers_registry`` queries actively misleading
        for uptime and "last alive" diagnostics.
        """
        system = data.get("system", {}) or {}
        gpu    = data.get("gpu", {}) or {}
        ts_col = self.dialect.telemetry_ts()  # 'timestamp' (SQLite) / 'ts' (PG)
        with self.backend.transaction() as conn:
            self.backend.tx_execute(conn, f"""
                INSERT INTO worker_telemetry
                (worker_id, {ts_col}, status, current_tic, phase, phase_progress,
                 star_elapsed_s, stars_done, avg_time_per_star, last_error,
                 cpu_percent, ram_used_gb, ram_total_gb, cpu_temp_c, pid_rss_mb,
                 swap_used_gb, recent_timings,
                 gpu_util_pct, gpu_mem_used_gb, gpu_mem_total_gb, gpu_temp_c, gpu_power_w)
                VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?)
            """, (
                worker_id,
                data.get("status"),
                data.get("current_tic"),
                data.get("phase"),
                data.get("phase_progress"),
                data.get("star_elapsed_s"),
                data.get("stars_done"),
                data.get("avg_time_per_star"),
                json.dumps(data.get("last_error")) if data.get("last_error") else None,
                system.get("cpu_percent"),
                system.get("ram_used_gb"),
                system.get("ram_total_gb"),
                system.get("cpu_temp_c"),
                system.get("pid_rss_mb"),
                system.get("swap_used_gb"),
                json.dumps(data.get("recent_timings")) if data.get("recent_timings") else None,
                gpu.get("util_pct"),
                gpu.get("mem_used_gb"),
                gpu.get("mem_total_gb"),
                gpu.get("temp_c"),
                gpu.get("power_w"),
            ))
            self.backend.tx_execute(
                conn,
                "UPDATE workers_registry SET last_seen=datetime('now') "
                "WHERE worker_id=?",
                (worker_id,),
            )

    def get_telemetry(self, worker_id, limit=100):
        """Get recent telemetry for a worker."""
        return self._query(
            "SELECT * FROM worker_telemetry WHERE worker_id=? ORDER BY id DESC LIMIT ?",
            (worker_id, limit))

    def get_latest_telemetry(self, worker_id):
        """Get the most recent telemetry entry for a worker."""
        rows = self._query(
            "SELECT * FROM worker_telemetry WHERE worker_id=? ORDER BY id DESC LIMIT 1",
            (worker_id,))
        return rows[0] if rows else None

    def set_debug_mode(self, worker_id, enabled, enabled_by="dashboard"):
        """Set debug mode for a worker."""
        self._exec("""
            INSERT INTO worker_debug_mode (worker_id, debug_enabled, enabled_at, enabled_by)
            VALUES (?, ?, datetime('now'), ?)
            ON CONFLICT(worker_id) DO UPDATE SET
                debug_enabled=excluded.debug_enabled,
                enabled_at=excluded.enabled_at,
                enabled_by=excluded.enabled_by
        """, (worker_id, self.dialect.bool_param(enabled), enabled_by))

    def get_debug_mode(self, worker_id):
        """Check if debug mode is enabled for a worker."""
        rows = self._query(
            "SELECT debug_enabled FROM worker_debug_mode WHERE worker_id=?",
            (worker_id,))
        return bool(rows[0]["debug_enabled"]) if rows else False

    def register_worker(self, worker_id, name):
        """Record a worker registration; keeps worker_id stable across restarts.

        Idempotent by ``name``, not just ``worker_id``: ``name`` is the
        worker's stable identity (the flota picks it per-machine), while a
        fresh ``worker_id`` can legitimately show up for an existing name —
        e.g. a Redis ``workers:registry`` entry keyed by the wrong thing, or
        a worker restarted after ``workers_registry`` was reset. If a row
        already exists for this ``name``, we must NOT insert a second row
        (the ``name`` column is UNIQUE) and must NOT touch its ``worker_id``:
        ``worker_telemetry`` has a foreign key to ``workers_registry.worker_id``,
        so repointing it here would either violate the FK or orphan history.
        The existing worker_id stays authoritative; only ``last_seen`` moves.
        Callers (``worker_register`` in routes/workers.py) reconcile any
        caller-side cache (Redis ``workers:registry``) against the DB's
        authoritative worker_id afterwards via ``get_worker_id_by_name``.
        """
        existing = self.get_worker_id_by_name(name)
        if existing is not None:
            self._exec(
                "UPDATE workers_registry SET last_seen=datetime('now') "
                "WHERE worker_id=?", (existing,))
            return
        try:
            self._exec("""
                INSERT INTO workers_registry (worker_id, name, first_seen, last_seen)
                VALUES (?, ?, datetime('now'), datetime('now'))
                ON CONFLICT(worker_id) DO UPDATE SET last_seen=datetime('now')
            """, (worker_id, name))
        except Exception as exc:
            # Race: another writer (or this same worker retrying under A3's
            # multi-process server) inserted a row for this name between our
            # SELECT and INSERT, or the name UNIQUE constraint fired for some
            # other reason. A worker that momentarily fails to persist to
            # workers_registry is far less harmful than one stuck in a 500
            # retry loop that never runs any task (see BUG 1 in the 2026-07-10
            # register()-500-loop incident) — log and move on instead of
            # propagating the failure up to the HTTP layer.
            if _is_unique_violation(exc):
                logger.warning(
                    "register_worker: unique-violation racing name=%r "
                    "worker_id=%r, retrying as update", name, worker_id)
                retry = self.get_worker_id_by_name(name)
                if retry is not None:
                    self._exec(
                        "UPDATE workers_registry SET last_seen=datetime('now') "
                        "WHERE worker_id=?", (retry,))
                    return
            logger.warning(
                "register_worker: failed to persist worker_id=%r name=%r: %s",
                worker_id, name, exc)

    def get_worker_id_by_name(self, name):
        """Return the stored worker_id for a given name, or None."""
        rows = self._query(
            "SELECT worker_id FROM workers_registry WHERE name=?", (name,))
        return rows[0]["worker_id"] if rows else None

    def get_worker_stats(self, worker_id):
        """Return persisted counters {tasks_completed, tasks_failed, total_processing_time}
        for a worker_id, or zeros if no row exists."""
        row = self._query_one(
            "SELECT tasks_completed, tasks_failed, total_processing_time "
            "FROM workers_registry WHERE worker_id=?", (worker_id,))
        if row is None:
            return {"tasks_completed": 0, "tasks_failed": 0, "total_processing_time": 0.0}
        return {
            "tasks_completed": row["tasks_completed"] or 0,
            "tasks_failed": row["tasks_failed"] or 0,
            "total_processing_time": row["total_processing_time"] or 0.0,
        }

    def update_worker_stats(self, worker_id, tasks_completed=None,
                            tasks_failed=None, total_processing_time=None):
        """Persist worker counters. Pass None to leave a field unchanged."""
        sets = []
        vals = []
        if tasks_completed is not None:
            sets.append("tasks_completed=?"); vals.append(int(tasks_completed))
        if tasks_failed is not None:
            sets.append("tasks_failed=?"); vals.append(int(tasks_failed))
        if total_processing_time is not None:
            sets.append("total_processing_time=?"); vals.append(float(total_processing_time))
        if not sets:
            return
        vals.append(worker_id)
        self._exec(
            f"UPDATE workers_registry SET {', '.join(sets)} WHERE worker_id=?",
            tuple(vals))

    def reset_all_worker_stats(self):
        """Zero the persisted counters (tasks_completed, tasks_failed,
        total_processing_time) for every worker. Used when launching a new run
        so the dashboard per-worker averages restart at 0."""
        self._exec(
            "UPDATE workers_registry SET tasks_completed=0, tasks_failed=0, "
            "total_processing_time=0")

    def purge_telemetry(self, retention_days=5):
        """Delete telemetry older than retention_days."""
        ts_col = self.dialect.telemetry_ts()        # 'timestamp' / 'ts'
        now_minus = self.dialect.now_minus()        # datetime('now', ?) / NOW()+interval
        self._exec(
            f"DELETE FROM worker_telemetry WHERE {ts_col} < {now_minus}",
            (f'-{retention_days} days',))

    # --- discovery_runs: durable orchestrator run record (Phase 3, Task 1) -- #
    # Foundation of the durable-orchestration plan: the discovery orchestrator's
    # state lives in the DB instead of only pipeline_state.json, so a restart
    # can resume from the last checkpoint instead of losing run state.

    def create_discovery_run(self, run_id, sector, lot_csv, dry_run, chained,
                             server_url):
        """Insert a new discovery_runs record in the initial ENQUEUEING state.

        run_id is the primary key; a duplicate raises the backend's integrity
        error (callers generate a fresh run_id per lot).
        """
        now = utc_now_iso()
        self._exec(
            """INSERT INTO discovery_runs
                   (run_id, sector, lot_csv, dry_run, chained, server_url,
                    started_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, sector, lot_csv,
             self.dialect.bool_param(dry_run), self.dialect.bool_param(chained),
             server_url, now, now))

    #: Rows per executemany batch when materialising a lot's TIC set. A lot is
    #: ~50k TICs = 50k single-column rows; psycopg3 binds one parameter per
    #: value, and the wire protocol caps a statement at 65535 parameters, so a
    #: single executemany of a whole lot would fail on PG (the same limit that
    #: bit the postprocess IN-list). 10k keeps every batch an order of
    #: magnitude clear of that ceiling on both backends.
    _RUN_TICS_CHUNK = 10_000

    def add_discovery_run_tics(self, run_id, tic_ids) -> int:
        """Materialise the TIC set of *run_id*'s lot into discovery_run_tics.

        This is what lets /api/health/work scope a backfill counter to the run
        that is waiting on it, instead of counting the whole (multi-sector,
        multi-lot) candidates table — see the table's comment in
        scripts/migration/schema_pg.sql for the 2026-07-14 hang this prevents.

        Idempotent: INSERT-ignore on the (run_id, tic_id) PK, so re-running the
        ENQUEUEING step of a resumed run re-inserts nothing. Returns the number
        of rows submitted (not the number actually inserted — the backends do
        not agree on what an ignored conflict reports).

        Empty input is a no-op.
        """
        rows = [(str(run_id), str(t)) for t in tic_ids]
        if not rows:
            return 0
        # NOT dialect.translate()'d here: backend.executemany translates the
        # SQL itself. Translating first would double-apply it — the PG dialect
        # rewrites '?' -> '%s' AND escapes literal '%' as '%%', so a
        # pre-translated '%s' comes back out as '%%s' and psycopg sees a query
        # with ZERO placeholders ("0 placeholders but 2 parameters were
        # passed"). SQLite's translate is the identity, so this bug is
        # invisible on SQLite and only fires against PG — i.e. only in
        # production. (scorecard.py DOES translate, because it binds against a
        # raw connection cursor that does no translation of its own.)
        sql = self.dialect.insert_ignore("discovery_run_tics",
                                         ["run_id", "tic_id"])
        for i in range(0, len(rows), self._RUN_TICS_CHUNK):
            self.backend.executemany(sql, rows[i:i + self._RUN_TICS_CHUNK])
        return len(rows)

    def count_discovery_run_tics(self, run_id) -> int:
        """Number of TICs materialised for *run_id* (0 if the run has none).

        The health/work scoping treats 0 as "this run has no materialised
        scope" and falls back to the global counters, so a run enqueued by an
        older orchestrator keeps working exactly as before.
        """
        return self._query_scalar(
            "SELECT COUNT(*) FROM discovery_run_tics WHERE run_id = ?",
            (str(run_id),)) or 0

    def run_scope_kind(self, run_id) -> str | None:
        """Which scope mechanism has TICs materialised for *run_id*: 'tasks'
        (S3 — ``tasks.run_id``), 'run_tics' (legacy — ``discovery_run_tics``,
        see :func:`add_discovery_run_tics`), or ``None`` (no scope at all —
        callers fall back to the GLOBAL counters, same as an unmaterialised
        run has always done).

        ``EXONIT_RUN_SCOPE`` (default ``"run_tics"``) picks which mechanism
        is tried FIRST:
          - ``"tasks"``: probe ``tasks.run_id`` first; only fall back to
            ``discovery_run_tics`` if that probe finds nothing (a run
            enqueued before the ``RUN_SCOPE`` flip, or the ``tasks.run_id``
            column itself predates the S3 DDL — see below).
          - anything else (the default): go straight to
            ``discovery_run_tics``, byte-for-byte the pre-S3 behaviour.

        A missing ``tasks.run_id`` column (S3 DDL not yet applied — the
        deploy order in the spec's §3.5 requires DDL BEFORE this flag is
        ever set to "tasks", but env-var typos happen, and the review fix
        of 2026-09-26 means the code itself may now ship before the DDL
        too) degrades to "treat the tasks probe as empty" rather than
        raising — this is a routing decision for read-only counters, never
        worth failing a request over. Checked via the same cached
        ``exonit.db_backend.column_exists`` guard the write path
        (``exonit/routes/workers.py``) uses, so the two agree on whether
        the column is live without each re-deriving it independently.
        """
        run_scope = os.environ.get("EXONIT_RUN_SCOPE", "run_tics")
        if run_scope == "tasks":
            from exonit.db_backend import column_exists
            if column_exists(self.backend, "tasks", "run_id"):
                try:
                    found = self._query_scalar(
                        "SELECT 1 FROM tasks WHERE run_id = ? LIMIT 1",
                        (str(run_id),))
                except Exception:
                    found = None
                if found:
                    return "tasks"
        if self.count_discovery_run_tics(run_id) > 0:
            return "run_tics"
        return None

    def get_discovery_run(self, run_id):
        """Return the discovery_runs row for run_id as a dict, or None.

        checkpoints is already json.loads-ed into a dict.
        """
        row = self._query_one(
            "SELECT * FROM discovery_runs WHERE run_id = ?", (run_id,))
        if row is None:
            return None
        row["checkpoints"] = json.loads(row["checkpoints"]) if row["checkpoints"] else {}
        return row

    def get_active_discovery_run(self):
        """Return the single non-terminal discovery_runs row, or None.

        At most one run is active at a time by application-level invariant
        (not a DB constraint); ORDER BY + LIMIT 1 is a defensive tie-breaker
        if that invariant were ever violated.
        """
        row = self._query_one(
            """SELECT * FROM discovery_runs
               WHERE state NOT IN ('DONE', 'FAILED')
               ORDER BY started_at DESC LIMIT 1""")
        if row is None:
            return None
        row["checkpoints"] = json.loads(row["checkpoints"]) if row["checkpoints"] else {}
        return row

    def get_latest_failed_discovery_run(self):
        """Return the most recently updated FAILED discovery_runs row, or None.

        Phase 3 Task 4: fallback resume candidate for POST
        /api/discovery/resume when there is no active (non-terminal) run —
        i.e. the orchestrator already reached FAILED (via its own step
        failure, the scan endpoint's spawn confirmation, or the dead-run
        watchdog) rather than dying silently mid-state.
        """
        row = self._query_one(
            """SELECT * FROM discovery_runs
               WHERE state = 'FAILED'
               ORDER BY updated_at DESC LIMIT 1""")
        if row is None:
            return None
        row["checkpoints"] = json.loads(row["checkpoints"]) if row["checkpoints"] else {}
        return row

    def update_discovery_run(self, run_id, *, state=None, detail=None,
                             pid=None, heartbeat=False, finished=False):
        """Update a discovery_runs row. Pass None to leave a field unchanged.

        updated_at is always refreshed. heartbeat=True also stamps
        heartbeat_at; finished=True also stamps finished_at (run reached a
        terminal state: DONE or FAILED).
        """
        sets = ["updated_at = ?"]
        vals = [utc_now_iso()]
        if state is not None:
            sets.append("state = ?"); vals.append(state)
        if detail is not None:
            sets.append("detail = ?"); vals.append(detail)
        if pid is not None:
            sets.append("orchestrator_pid = ?"); vals.append(pid)
        if heartbeat:
            sets.append("heartbeat_at = ?"); vals.append(utc_now_iso())
        if finished:
            sets.append("finished_at = ?"); vals.append(utc_now_iso())
        vals.append(run_id)
        self._exec(
            f"UPDATE discovery_runs SET {', '.join(sets)} WHERE run_id = ?",
            tuple(vals))

    def set_discovery_run_checkpoint(self, run_id, step, data):
        """Merge ``checkpoints[step] = data`` into a run's checkpoints dict.

        Read-modify-write inside a transaction so concurrent checkpoint writes
        for different steps do not clobber each other.
        """
        with self.backend.transaction() as conn:
            row = self.backend.tx_query_one(
                conn, "SELECT checkpoints FROM discovery_runs WHERE run_id = ?",
                (run_id,))
            if row is None:
                return
            checkpoints = json.loads(row["checkpoints"]) if row["checkpoints"] else {}
            checkpoints[step] = data
            self.backend.tx_execute(
                conn,
                "UPDATE discovery_runs SET checkpoints = ?, updated_at = ? "
                "WHERE run_id = ?",
                (json.dumps(checkpoints), utc_now_iso(), run_id))

    def close(self):
        self.backend.close()
        if self._is_pg and self._cmpl_conn is not None:
            self._cmpl_conn.close()
