"""Database backend abstraction for ExoNit: SQLite (default) or PostgreSQL.

The whole application talks to the DB through ``ExoNitDB`` (exonit/db.py), which
historically opened ``sqlite3`` directly. This module introduces a thin seam so
the same query code can run on either backend, selected at construction time:

    make_backend("data/exonit.db")                  -> SqliteBackend
    make_backend("postgresql://u:p@host/exonit")    -> PgBackend
    make_backend(None)  # reads EXONIT_DATABASE_URL  -> Pg if set, else default file

Design goals
------------
* **SQLite stays bit-identical.** SqliteBackend reproduces the previous
  connection (WAL, busy_timeout, synchronous, ``sqlite3.Row``) and the single
  ``threading.Lock`` that guarded every call.
* **Minimal churn in db.py.** The backend *auto-translates* the regular
  SQLite-isms that appear verbatim in the SQL strings — ``?`` placeholders,
  ``datetime('now')``, ``INDEXED BY`` hints, ``PRAGMA`` — so db.py keeps its SQL
  almost unchanged. The handful of constructs that depend on the schema/intent
  (upsert / ``ON CONFLICT``, scalar ``LEAST``, ``RETURNING id``, the relative
  ``datetime('now', ?)`` interval, the ``timestamp``->``ts`` column rename) are
  exposed as explicit ``Dialect`` helpers that db.py calls deliberately.
* **Same return shapes.** ``query`` returns ``list[dict]``, ``query_one`` a
  ``dict`` or ``None``, ``query_scalar`` a single value — exactly what db.py's
  old ``_query``/``_query_one``/``_query_scalar`` returned.

Only the operational tables move to PG in the first cut; the write-back caches
and the ``completeness``/``completeness_agg`` subsystem stay on SQLite (see
docs/superpowers/specs/2026-06-24-postgres-backend-design.md).
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time as _time
from contextlib import contextmanager
from uuid import uuid4

logger = logging.getLogger(__name__)


def is_pg_url(url: str | None) -> bool:
    """True if *url* is a PostgreSQL DSN (postgres:// or postgresql://)."""
    return bool(url) and str(url).startswith(("postgres://", "postgresql://"))


# --------------------------------------------------------------------------- #
# SQL translation: convert ? placeholders to %s without touching the contents
# of string literals. The rest of the SQLite-isms are simple, safe substring
# rewrites (the affected tokens never appear inside user data).
# --------------------------------------------------------------------------- #

def qmark_to_pyformat(sql: str) -> str:
    """Convert ``?`` placeholders to ``%s``, respecting single-quoted literals.

    psycopg uses pyformat (``%s``); SQLite uses qmark (``?``). A naive replace
    would corrupt a ``?`` inside a string literal, and would also turn a bare
    ``%`` (e.g. ``LIKE '%foo%'``) into a broken format spec — so we also escape
    literal ``%`` to ``%%`` for psycopg.
    """
    out = []
    in_str = False
    i = 0
    n = len(sql)
    while i < n:
        c = sql[i]
        if c == "'":
            # Handle '' escape inside a string literal.
            if in_str and i + 1 < n and sql[i + 1] == "'":
                out.append("''")
                i += 2
                continue
            in_str = not in_str
            out.append(c)
        elif in_str:
            # Inside a literal: escape % for psycopg, keep ? verbatim.
            out.append("%%" if c == "%" else c)
        elif c == "?":
            out.append("%s")
        elif c == "%":
            out.append("%%")
        else:
            out.append(c)
        i += 1
    return "".join(out)


_DATETIME_NOW_RE = re.compile(r"datetime\(\s*'now'\s*\)", re.IGNORECASE)
_INDEXED_BY_RE = re.compile(r"\s+INDEXED\s+BY\s+[A-Za-z_][A-Za-z0-9_]*", re.IGNORECASE)

# Columns declared BOOLEAN in schema_pg.sql. SQLite stores them as 0/1 ints, so
# the codebase compares them with integer literals (``has_spoc = 1``). PG rejects
# ``boolean = integer``, so on PG we rewrite a literal 0/1 comparison to TRUE/FALSE.
# Only literal-comparison sites need this; parameter writes go through bool_param.
_BOOL_COLS = (
    "has_spoc", "ctoi_submitted", "cross_source_agreement",
    "gpu_used", "lc_load_recovered", "debug_enabled",
)
_BOOL_LITERAL_RE = re.compile(
    r"\b((?:[A-Za-z_][A-Za-z0-9_]*\.)?(?:" + "|".join(_BOOL_COLS) + r"))\s*=\s*([01])\b",
    re.IGNORECASE,
)


def _bool_literal_sub(m: "re.Match") -> str:
    return f"{m.group(1)} = {'TRUE' if m.group(2) == '1' else 'FALSE'}"


class Dialect:
    """SQLite dialect (the base/default).  Methods are overridden for PG."""

    name = "sqlite"
    placeholder = "?"
    # Boolean columns are stored as INTEGER 0/1 in SQLite.
    bool_out = staticmethod(lambda v: 1 if v else 0)

    def translate(self, sql: str) -> str:
        """Backend-specific rewrite of a SQL string. Identity for SQLite."""
        return sql

    # --- explicit helpers db.py calls where intent matters --------------- #
    def now(self) -> str:
        return "datetime('now')"

    def now_minus(self) -> str:
        """SQL fragment for "now minus an interval given as a parameter".

        SQLite: ``datetime('now', ?)`` with param like ``'-5 days'``.
        """
        return "datetime('now', ?)"

    def least(self, a: str, b: str) -> str:
        """Scalar minimum of two columns (SQLite ``min`` is scalar here)."""
        return f"min({a}, {b})"

    def nulls_last(self, col: str, direction: str = "DESC") -> str:
        """ORDER BY fragment for *col* sorted *direction*, NULLs always last.

        B5 (docs/reviews/2026-07-02-opus48-cleanup-review.md), same dialect-
        drift family as the ``LEAST`` bug (c9d7ed4b): PostgreSQL's default
        NULL placement is NULLS FIRST on DESC / NULLS LAST on ASC (nulls sort
        as "larger" than any value); SQLite's default is the opposite — NULLS
        FIRST on ASC / NULLS LAST on DESC (nulls sort as "smaller"). A plain
        ``ORDER BY col DESC`` therefore silently reorders a nullable column's
        NULL rows to the *front* on PG while they stayed at the *back* on
        SQLite (e.g. ``/api/candidates?sort=cnn`` — unscored, cnn_score IS
        NULL candidates jumping to the top of a "highest CNN score" sort).

        Both PostgreSQL (always) and the project's bundled SQLite (>=3.30,
        verified: 3.45.1 — see progress/impl-dblayer-2026-07-02.md) support
        the explicit ``NULLS LAST`` syntax, so this pins the SAME placement —
        NULLS LAST, matching the historical (SQLite-only) DESC behaviour — on
        both backends regardless of dialect defaults. Identical on both
        dialects, so no PgDialect override is needed.
        """
        return f"{col} {direction} NULLS LAST"

    def telemetry_ts(self) -> str:
        """Name of the worker_telemetry timestamp column for this backend."""
        return "timestamp"

    def bool_param(self, v) -> object:
        """Coerce a Python truthy value to this backend's boolean storage."""
        return 1 if v else 0

    def for_update_skip_locked(self) -> str:
        """Row-lock clause for the claim-one-task SELECT.

        SQLite serialises writes globally (the ExoNitDB lock + WAL), so no
        per-row lock clause is needed — empty here.  PG needs
        ``FOR UPDATE SKIP LOCKED`` so two uvicorn processes never claim the same
        pending task (the cross-process race the single-process lock cannot
        guard against).
        """
        return ""

    def upsert(self, table, columns, conflict_cols, update_cols=None,
              raw_values=None) -> str:
        """Build an upsert INSERT for this backend.

        * ``columns``        — every column written by the INSERT, in order;
                                the VALUES list has one ``?`` per column
                                (unless overridden by ``raw_values``).
        * ``conflict_cols``  — columns of the unique/PK constraint to upsert on.
        * ``update_cols``    — columns refreshed on conflict (defaults to every
                                inserted column that is not in ``conflict_cols``).
        * ``raw_values``     — optional ``{column: raw_sql_expr}`` map for
                                columns whose VALUES entry is a raw SQL
                                expression (e.g. ``"datetime('now')"``) instead
                                of a bound ``?`` placeholder — for DB-clock
                                timestamp columns (``updated_at``) that must
                                reflect the database's clock, not a Python-side
                                value. The caller's ``params`` tuple must then
                                OMIT the raw-value columns (no placeholder is
                                emitted for them).

        SQLite returns ``INSERT OR REPLACE`` — bit-identical to the historical
        ExoNitDB behaviour, which replaces the whole row on a PK collision
        (``add_star``'s contract). ``update_cols`` is irrelevant for SQLite
        (REPLACE rewrites the WHOLE row — every column not in ``columns``
        reverts to its default/NULL), it only shapes the PG ``DO UPDATE``
        clause. Callers that need a genuine PARTIAL update on conflict — only
        the given columns change, everything else is left untouched, on
        EITHER backend — must use ``upsert_on_conflict()`` instead (see D13
        below); ``upsert()`` here is intentionally REPLACE-semantics-only.

        Placeholders are always ``?``: the returned SQL is handed to the backend
        ``execute``, which runs it through ``translate()`` (``?`` -> ``%s`` for
        PG).  Emitting ``%s`` here would be double-escaped to ``%%s``.
        """
        raw_values = raw_values or {}
        cols = ", ".join(columns)
        ph = ", ".join(raw_values.get(c, "?") for c in columns)
        return f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({ph})"

    def _on_conflict_upsert_sql(self, table, columns, conflict_cols,
                                update_cols=None, raw_values=None) -> str:
        """``INSERT ... ON CONFLICT (...) DO UPDATE SET ...`` builder.

        Identical SQL syntax on SQLite (>=3.24 — the project's bundled SQLite
        is 3.45.1, see progress/impl-dblayer-2026-07-02.md) and PostgreSQL, so
        this single implementation backs BOTH ``PgDialect.upsert()`` (PG has
        no other upsert form) and the base ``Dialect.upsert_on_conflict()``
        below (an explicit partial-update upsert for SQLite callers, since
        ``Dialect.upsert()`` on SQLite is REPLACE-only). See ``upsert()`` for
        the parameter docs (identical here).
        """
        raw_values = raw_values or {}
        if update_cols is None:
            update_cols = [c for c in columns if c not in conflict_cols]
        cols = ", ".join(columns)
        ph = ", ".join(raw_values.get(c, "?") for c in columns)
        conflict = ", ".join(conflict_cols)
        if update_cols:
            sets = ", ".join(
                f"{c} = {raw_values[c]}" if c in raw_values else f"{c} = EXCLUDED.{c}"
                for c in update_cols)
            action = f"DO UPDATE SET {sets}"
        else:
            action = "DO NOTHING"
        return (f"INSERT INTO {table} ({cols}) VALUES ({ph}) "
                f"ON CONFLICT ({conflict}) {action}")

    def upsert_on_conflict(self, table, columns, conflict_cols,
                           update_cols=None, raw_values=None) -> str:
        """Partial-column upsert via ``INSERT ... ON CONFLICT DO UPDATE``.

        D13 (docs/reviews/2026-07-02-opus48-cleanup-review.md): the seam
        ``exonit/discovery_state.py``'s ``set_sector_state``/``set_autofill``
        route through, instead of hand-building the same SQL. UNLIKE
        ``upsert()`` (which is ``INSERT OR REPLACE`` on SQLite — a full-row
        clobber that resets every unlisted column to its default), this
        preserves every column NOT in ``columns``/``update_cols`` on
        conflict — the semantics both callers need (e.g. ``set_autofill``'s
        "only the passed fields are refreshed, everything else keeps its
        stored value" contract, exercised by
        tests/test_discovery_autofill_manager.py against a plain SQLite DB).
        Same SQL text on SQLite and PostgreSQL (see ``_on_conflict_upsert_sql``),
        so no PgDialect override is needed.
        """
        return self._on_conflict_upsert_sql(
            table, columns, conflict_cols, update_cols, raw_values)

    def insert_ignore(self, table, columns) -> str:
        """Insert a row, ignoring conflicts on the PK/unique constraint.

        SQLite: ``INSERT OR IGNORE``. PG: ``INSERT ... ON CONFLICT DO NOTHING``.
        Used by the scorecard's TEMP-TABLE TIC-filter population. Placeholders
        stay ``?`` (translated to ``%s`` by the backend for PG).
        """
        cols = ", ".join(columns)
        ph = ", ".join("?" for _ in columns)
        return f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({ph})"


class PgDialect(Dialect):
    """PostgreSQL dialect."""

    name = "postgres"
    placeholder = "%s"
    bool_out = staticmethod(bool)

    def translate(self, sql: str) -> str:
        sql = _INDEXED_BY_RE.sub("", sql)          # PG: no index hints
        sql = _DATETIME_NOW_RE.sub("NOW()", sql)   # datetime('now') -> NOW()
        sql = _BOOL_LITERAL_RE.sub(_bool_literal_sub, sql)  # boolcol = 0/1 -> FALSE/TRUE
        sql = qmark_to_pyformat(sql)               # ? -> %s, escape %
        return sql

    def now(self) -> str:
        return "NOW()"

    def now_minus(self) -> str:
        # param is an interval string like '-5 days'
        return "NOW() + (?)::interval"

    def least(self, a: str, b: str) -> str:
        return f"LEAST({a}, {b})"

    def telemetry_ts(self) -> str:
        return "ts"

    def bool_param(self, v) -> object:
        return bool(v)

    def for_update_skip_locked(self) -> str:
        return " FOR UPDATE SKIP LOCKED"

    def upsert(self, table, columns, conflict_cols, update_cols=None,
              raw_values=None) -> str:
        # PG has only one upsert form (ON CONFLICT DO UPDATE) — delegate to
        # the shared builder (also used by the base Dialect.upsert_on_conflict
        # for SQLite callers that need the same partial-update semantics).
        # Placeholders stay ``?``; the backend ``execute`` translate() converts
        # them to ``%s`` for PG (emitting ``%s`` here would be double-escaped).
        return self._on_conflict_upsert_sql(
            table, columns, conflict_cols, update_cols, raw_values)

    def insert_ignore(self, table, columns) -> str:
        cols = ", ".join(columns)
        ph = ", ".join("?" for _ in columns)
        return (f"INSERT INTO {table} ({cols}) VALUES ({ph}) "
                f"ON CONFLICT DO NOTHING")


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

class Backend:
    """Common backend interface. ``dialect`` exposes the SQL differences."""

    dialect: Dialect

    def execute(self, sql, params=()):
        raise NotImplementedError

    def executemany(self, sql, seq_of_params):
        raise NotImplementedError

    def query(self, sql, params=()):
        raise NotImplementedError

    def query_one(self, sql, params=()):
        raise NotImplementedError

    def query_scalar(self, sql, params=()):
        raise NotImplementedError

    def execute_returning_id(self, sql, params=()):
        """Run an INSERT and return the generated primary-key id."""
        raise NotImplementedError

    def execute_rowcount(self, sql, params=()):
        """Run a write and return the number of rows affected."""
        raise NotImplementedError

    def iter_id_column(self, sql, params=(), *, batch_rows=50_000):
        """Yield column-0 values of a single-column SELECT, one at a time.

        Q3 (perf review 2026-09-26, docs/reviews/2026-09-26-enqueue-path-
        performance-review.md): id-only queries at campaign scale
        (``load_completed_tic_set``, ``done_state._done_ids``,
        ``MetadataCache.tglc_tic_ids``/``all_tic_ids``) used
        ``{row["col"] for row in backend.query(...)}`` — ``query()`` always
        builds one dict_row (PG) / ``Row`` (SQLite) object per matched row
        AND materializes the whole result set as a Python list before the
        comprehension can even start discarding everything but column 0.
        For the 27.6 M-row completed-set scan that measured a ~7 GB peak.

        This streams the SAME rows as plain tuples via a server-side cursor
        (PG, ``itersize=batch_rows``) or repeated ``fetchmany`` (SQLite),
        yielding ``row[0]`` directly — no dict/Row wrapper ever built, and
        only one batch buffered client-side at a time. Order is preserved
        (whatever the query's own ordering is, or none); callers needing a
        dedup'd set wrap this in ``set(...)``, callers needing the query's
        order (e.g. ``tglc_tic_ids``' rowid/seq ASC contract) use
        ``list(...)`` — both drop-in replacements for the old
        comprehension's result.

        Base-class fallback (used by any ``Backend`` subclass — e.g. a test
        double — that doesn't override this): correctness over memory
        optimality, via the existing ``query()`` surface.
        """
        for r in self.query(sql, params):
            yield r[0] if not isinstance(r, dict) else next(iter(r.values()))

    @contextmanager
    def transaction(self):
        raise NotImplementedError

    # --- in-transaction helpers ------------------------------------------- #
    # These run a statement on the connection yielded by ``transaction()`` so a
    # multi-statement SELECT+UPDATE stays atomic.  They translate the SQL the
    # same way the standalone methods do, so db.py can keep its SQLite-flavoured
    # SQL inside a ``with backend.transaction() as conn:`` block on either
    # backend.  The default impls assume a DB-API connection with ``execute``.
    def tx_execute(self, conn, sql, params=()):
        conn.execute(sql, params)

    def tx_query(self, conn, sql, params=()):
        cur = conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    def tx_query_one(self, conn, sql, params=()):
        cur = conn.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None

    def checkpoint(self):
        """WAL checkpoint (SQLite); no-op on PG."""
        return None

    def op_session(self):
        """A sqlite3-``con``-shaped single-connection session over this backend.

        See :class:`OperationalSession`. The caller closes the session; the
        backend itself is NOT closed (it may be shared).
        """
        return OperationalSession(self)

    def close(self):
        raise NotImplementedError


class SqliteBackend(Backend):
    """SQLite backend — reproduces the previous ExoNitDB connection exactly."""

    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=10000")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.dialect = Dialect()
        self._lock = threading.Lock()

    def execute(self, sql, params=()):
        with self._lock:
            self.conn.execute(sql, params)
            self.conn.commit()

    def executemany(self, sql, seq_of_params):
        with self._lock:
            self.conn.executemany(sql, seq_of_params)
            self.conn.commit()

    def query(self, sql, params=()):
        with self._lock:
            cur = self.conn.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def query_one(self, sql, params=()):
        with self._lock:
            cur = self.conn.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None

    def query_scalar(self, sql, params=()):
        with self._lock:
            return self.conn.execute(sql, params).fetchone()[0]

    def execute_returning_id(self, sql, params=()):
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur.lastrowid

    def execute_rowcount(self, sql, params=()):
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur.rowcount

    def iter_id_column(self, sql, params=(), *, batch_rows=50_000):
        # A dedicated cursor with row_factory=None (plain tuples), not the
        # connection's own sqlite3.Row default (self.conn.row_factory, set
        # in __init__) — the whole point of this method is to skip building
        # a Row wrapper per matched row. Held under the same backend-wide
        # lock as query()/execute() for the whole scan (see execute_stream's
        # docstring on OperationalSession for the equivalent SQLite
        # trade-off): spread over more wall-clock time than an eager
        # fetchall(), same total lock-held time.
        with self._lock:
            cur = self.conn.cursor()
            cur.row_factory = None
            cur.execute(sql, params)
            while True:
                rows = cur.fetchmany(batch_rows)
                if not rows:
                    break
                for r in rows:
                    yield r[0]

    @contextmanager
    def transaction(self):
        with self._lock:
            try:
                yield self.conn
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    @contextmanager
    def raw_connection(self):
        """Yield a raw DB-API connection (for pandas / multi-statement work).

        SQLite: the single shared connection, guarded by the backend lock so a
        whole ``CREATE TEMP TABLE`` + ``SELECT`` sequence stays consistent.
        Caller does NOT commit/close — the backend owns the connection.
        """
        with self._lock:
            yield self.conn

    def checkpoint(self):
        with self._lock:
            return self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()

    def close(self):
        self.conn.close()


def _int_env(name: str, default: int) -> int:
    """Read a non-negative int from an env var; fall back to *default* on
    anything malformed (missing, non-numeric, negative). Never raises — these
    backends are built at server startup, so a bad env var must not block it.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _pg_connect_kwargs() -> dict:
    """libpq connection kwargs for timeouts + TCP keepalives (incident
    2026-08-15): a brief <host><-><host> network blip left psycopg
    connections in a "black hole" (reads blocked forever, no socket error),
    piling up ASGI threadpool threads (130-155/worker stuck in
    futex_do_wait) until the accept queue filled and the server hung.
    ``ConnectionPool`` sets none of this by default.

    ``keepalives_*`` are unconditional, not env-tunable: probe every 10s
    after 30s idle, 3 missed probes -> dead peer detected in ~60s. That is
    the actual fix for the incident. ``connect_timeout`` and
    ``statement_timeout`` ARE env-tunable because these pools are also built
    by batch scripts (postprocess, rebuild_work_streams, ...) that may need
    different bounds than the live server.
    """
    connect_timeout = _int_env("EXONIT_PG_CONNECT_TIMEOUT_S", 10)
    # 600000ms (10min) default: the slowest known query served by these pools
    # is indexed_basename_count for sector 11 (13.2M tglc files + a LIKE seq
    # scan over 76.5M fits_blobs rows), measured at 127s under normal load --
    # 600s gives ~4.7x headroom while still bounding a runaway query. 0
    # disables the timeout (for scripts that legitimately need to).
    statement_timeout_ms = _int_env("EXONIT_PG_STATEMENT_TIMEOUT_MS", 600000)

    kwargs = {
        "connect_timeout": connect_timeout,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 3,
    }
    if statement_timeout_ms != 0:
        kwargs["options"] = f"-c statement_timeout={statement_timeout_ms}"
    return kwargs


class PgBackend(Backend):
    """PostgreSQL backend (psycopg v3 + a per-process connection pool).

    Auto-translates each SQL string via ``PgDialect.translate`` (placeholders,
    ``datetime('now')``, ``INDEXED BY``) so db.py's SQLite-flavoured SQL runs
    unchanged for the regular cases. Rows are returned as plain dicts to match
    the SQLite ``Row``->dict shape.
    """

    def __init__(self, url: str, min_size: int = 1, max_size: int = 8):
        import psycopg  # noqa: F401  (import-time check)
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        self.url = url
        self.dialect = PgDialect()
        self._dict_row = dict_row
        # open=True so a bad DSN fails fast at construction, like sqlite.connect.
        self.pool = ConnectionPool(
            url, min_size=min_size, max_size=max_size,
            kwargs={"row_factory": dict_row, **_pg_connect_kwargs()}, open=True,
        )

    def _t(self, sql: str) -> str:
        return self.dialect.translate(sql)

    def execute(self, sql, params=()):
        with self.pool.connection() as conn:
            conn.execute(self._t(sql), params)

    def executemany(self, sql, seq_of_params):
        with self.pool.connection() as conn:
            conn.cursor().executemany(self._t(sql), list(seq_of_params))

    def query(self, sql, params=()):
        with self.pool.connection() as conn:
            cur = conn.execute(self._t(sql), params)
            # No dict(r) re-wrap needed: every pooled connection is opened
            # with row_factory=dict_row (see __init__), so cur.fetchall()
            # already yields plain dicts (item 8, 2026-07-02 cleanup review).
            return cur.fetchall()

    def query_one(self, sql, params=()):
        with self.pool.connection() as conn:
            cur = conn.execute(self._t(sql), params)
            return cur.fetchone()  # already a dict (dict_row) or None

    def query_scalar(self, sql, params=()):
        with self.pool.connection() as conn:
            cur = conn.execute(self._t(sql), params)
            row = cur.fetchone()
            if row is None:
                return None
            # dict_row: return the first value regardless of column name.
            return next(iter(row.values()))

    def execute_returning_id(self, sql, params=()):
        sql_t = self._t(sql)
        if "returning" not in sql_t.lower():
            sql_t = sql_t.rstrip().rstrip(";") + " RETURNING id"
        with self.pool.connection() as conn:
            cur = conn.execute(sql_t, params)
            row = cur.fetchone()
            return next(iter(row.values())) if row else None

    def execute_rowcount(self, sql, params=()):
        with self.pool.connection() as conn:
            cur = conn.execute(self._t(sql), params)
            return cur.rowcount

    def iter_id_column(self, sql, params=(), *, batch_rows=50_000):
        # A fresh, uniquely-named server-side cursor — same construct as
        # OperationalSession.execute_stream — but with row_factory=tuple_row
        # instead of the connection's pooled default (dict_row, see
        # __init__): a dict per matched row is exactly the cost this method
        # exists to avoid for an id-only SELECT. `conn.cursor()` inherits the
        # CONNECTION's row_factory unless one is passed explicitly here, so
        # this must be spelled out, not just omitted.
        from psycopg.rows import tuple_row
        with self.pool.connection() as conn:
            cur = conn.cursor(name=f"exonit_idcol_{uuid4().hex}",
                              row_factory=tuple_row)
            cur.itersize = batch_rows
            try:
                cur.execute(self._t(sql), params)
                while True:
                    rows = cur.fetchmany(batch_rows)
                    if not rows:
                        break
                    for r in rows:
                        yield r[0]
            finally:
                cur.close()

    @contextmanager
    def transaction(self):
        with self.pool.connection() as conn:
            with conn.transaction():
                yield conn

    @contextmanager
    def raw_connection(self):
        """Yield a raw psycopg connection borrowed from the pool.

        For pandas reads and multi-statement work that must run on ONE
        connection (e.g. a session-scoped ``CREATE TEMP TABLE`` followed by
        ``SELECT``s).  Rolls back on exit so a temp table / uncommitted work
        never leaks back into the pooled connection.
        """
        with self.pool.connection() as conn:
            try:
                yield conn
            finally:
                conn.rollback()

    def tx_execute(self, conn, sql, params=()):
        conn.execute(self._t(sql), params)

    def tx_query(self, conn, sql, params=()):
        # conn is always pool-sourced (transaction() / raw_connection()), so
        # row_factory=dict_row already applies — no dict(r) re-wrap needed.
        cur = conn.execute(self._t(sql), params)
        return cur.fetchall()

    def tx_query_one(self, conn, sql, params=()):
        cur = conn.execute(self._t(sql), params)
        return cur.fetchone()  # already a dict (dict_row) or None

    def checkpoint(self):
        return None  # PG manages its own WAL

    def close(self):
        self.pool.close()


def make_backend(url_or_path: str | None) -> Backend:
    """Build the right backend.

    * a PostgreSQL DSN (or ``EXONIT_DATABASE_URL`` when *url_or_path* is None)
      -> :class:`PgBackend`
    * anything else (a filesystem path, or the default) -> :class:`SqliteBackend`
    """
    if url_or_path is None:
        url_or_path = os.environ.get("EXONIT_DATABASE_URL") or "data/exonit.db"
    if is_pg_url(url_or_path):
        return PgBackend(url_or_path)
    return SqliteBackend(url_or_path)


def connect_operational(db_path: str | None) -> Backend:
    """Open a lightweight backend for the *operational* DB (no completeness).

    The discovery pipeline scripts (scorecard, postprocess, finalize,
    orchestrator) historically did ``sqlite3.connect(config.db_path)`` to read
    and update the operational tables. With PG that breaks. This is the seam
    they call instead: it returns the right :class:`Backend` for the given path
    or DSN — a plain SQLite file when ``db_path`` is a path, a pooled PgBackend
    when it is a ``postgresql://`` DSN.

    Unlike :class:`~exonit.db.ExoNitDB`, this opens *only* the operational
    connection: no completeness subsystem, no schema/trigger init (the scripts
    must not re-run schema init — the PG schema is created out of band, and the
    SQLite schema already exists). The caller is responsible for ``close()``.
    """
    return make_backend(db_path)


# --------------------------------------------------------------------------- #
# Column-existence guard: safe against a code-before-DDL deploy order
# --------------------------------------------------------------------------- #
#
# Review finding (2026-09-26, docs/superpowers/specs/2026-09-26-enqueue-s1-
# s4-design.md's own §6 deploy-window table schedules code BEFORE DDL for
# both S3's tasks.run_id and, historically, tasks.sector): a caller that
# unconditionally references an ADDITIVE column in an INSERT/UPDATE (e.g.
# exonit/routes/workers.py's _insert_tasks_bulk_pg/_insert_tasks_bulk_sqlite/
# _adopt_live_tasks) must not assume the column exists just because the code
# that references it has been deployed — SQLite self-migrates on every
# process start (exonit/db.py's ExoNitDB.__init__), but PostgreSQL's schema
# is applied out of band, on its own deploy window. Reduced to two behaviours,
# gated by this guard: (missing) silently ignore the column (log ONCE per
# (table, column) at INFO, not a warning/error and not on every call), (5+
# minutes later, or after invalidate_column_exists) start using it once the
# DDL lands, with NO restart required.

#: (backend_identity, table, column) -> (exists: bool, checked_at: float).
#: Keyed by backend identity (NOT just (table, column)) even though a real
#: deployed process only ever talks to ONE operational DB — a test session
#: imports this module once but constructs many different SQLite files (or,
#: in principle, several PG DSNs) in the SAME process, and a bare
#: (table, column) key would leak a "column absent" (or "present") result
#: from one test's DB into a completely different test's DB. See
#: ``_backend_identity``.
_column_exists_cache: dict[tuple, tuple[bool, float]] = {}
#: Same keying as above — logged once per (backend, table, column), not on
#: every cache-expiry recheck.
_column_missing_logged: set[tuple] = set()

#: Re-check interval. Column existence changes at most once per DDL deploy
#: window (hours-to-days apart in practice); 5 minutes bounds how long a
#: mid-process DDL apply takes to be picked up without hammering the
#: catalog on every INSERT chunk. Callers that just applied DDL themselves
#: (or a test) can bypass this via invalidate_column_exists().
_COLUMN_EXISTS_TTL_S = 300.0


def _backend_identity(backend):
    """A stable identifier for *backend*'s target database — ``path`` for
    SQLite, ``url`` (the DSN) for PG, falling back to ``id(backend)`` for
    anything else (e.g. a test double) so the cache still degrades to
    "per-backend-instance" rather than crashing."""
    return getattr(backend, "path", None) or getattr(backend, "url", None) or id(backend)


def column_exists(backend, table: str, column: str,
                  *, ttl_s: float = _COLUMN_EXISTS_TTL_S) -> bool:
    """Whether *table* has *column*, on whichever DB *backend* points at.

    PostgreSQL: resolves *table* via ``to_regclass`` — i.e. through the
    CURRENT connection's search_path, exactly like an unqualified ``INSERT
    INTO <table>`` would — then checks ``pg_attribute`` for a live (not
    dropped) column. Deliberately NOT ``information_schema.columns``
    filtered by a hardcoded schema name: that would either need to guess
    the right schema or match a same-named table in an unrelated schema,
    both wrong for a caller relying on the same name resolution the actual
    DML uses (e.g. a test's isolated ``search_path``).
    SQLite: ``PRAGMA table_info(<table>)``.

    Cached per (backend, table, column) for *ttl_s* seconds (default 5
    min) — see the module-level cache's docstring above. A query failure
    (e.g. the table doesn't exist at all, or a transient connection error)
    degrades to ``False`` (column absent) WITHOUT being cached, so a
    transient failure is retried on the next call rather than sticking for
    the whole TTL window; a table that genuinely doesn't exist yet reads
    the same as "column absent" to every caller of this function, which is
    the correct degrade-to-legacy-behaviour reading either way.
    """
    key = (_backend_identity(backend), table, column)
    now = _time.monotonic()
    cached = _column_exists_cache.get(key)
    if cached is not None and (now - cached[1]) < ttl_s:
        return cached[0]

    try:
        if backend.dialect.name == "postgres":
            val = backend.query_scalar(
                "SELECT 1 FROM pg_attribute "
                "WHERE attrelid = to_regclass(?) AND attname = ? "
                "AND attnum > 0 AND NOT attisdropped",
                (table, column))
        else:
            rows = backend.query(f"PRAGMA table_info({table})")
            val = 1 if any(r["name"] == column for r in rows) else None
        exists = bool(val)
    except Exception:
        logger.warning(
            "column_exists probe failed for %s.%s (treating as absent this "
            "call; not cached, will retry next call)",
            table, column, exc_info=True)
        return False

    _column_exists_cache[key] = (exists, now)
    was_logged_missing = key in _column_missing_logged
    if not exists and not was_logged_missing:
        logger.info(
            "%s.%s not present yet (pre-DDL) -- callers will ignore/degrade "
            "wherever it's referenced until the DDL lands (re-checked every "
            "%.0fs, or immediately after invalidate_column_exists())",
            table, column, ttl_s)
        _column_missing_logged.add(key)
    elif exists and was_logged_missing:
        logger.info(
            "%s.%s is now present -- callers will start using it",
            table, column)
        _column_missing_logged.discard(key)
    return exists


def invalidate_column_exists(backend, table: str, column: str | None = None) -> None:
    """Force the next :func:`column_exists` call (against THIS *backend*'s
    target DB) to re-probe *table* (all columns if *column* is None)
    instead of trusting a cached value.

    Call after DDL you just applied yourself (tests, or an operator who
    just ran the migration against a live process), or defensively after
    an INSERT/UPDATE that referenced the column fails unexpectedly despite
    a cached "exists" — so the failure doesn't recur for the rest of the
    TTL window.
    """
    ident = _backend_identity(backend)
    if column is not None:
        key = (ident, table, column)
        _column_exists_cache.pop(key, None)
        _column_missing_logged.discard(key)
        return
    for key in [k for k in _column_exists_cache
               if k[0] == ident and k[1] == table]:
        del _column_exists_cache[key]
        _column_missing_logged.discard(key)


# --------------------------------------------------------------------------- #
# Chunked IN(...) queries: PG's 65535-param-per-statement limit
# --------------------------------------------------------------------------- #

def select_in_chunks(runner, sql_before, ids, sql_after="", params_extra=(),
                     chunk_size=30000):
    """Run ``{sql_before} (<placeholders>){sql_after}`` over *ids* in chunks.

    D2 (docs/reviews/2026-07-02-opus48-cleanup-review.md): promoted from
    ``scripts/discovery_postprocess.py``'s ``_select_in_chunks`` — the
    correctly-sized (30k) version — into a reusable helper so other callers
    with an un-chunked ``WHERE col IN (...)`` (e.g. ``exonit/db.py``'s
    ``update_candidate_claude_batch``, which built the IN list with no
    chunking at all) get the same fix instead of a fresh copy.

    The PostgreSQL backend (psycopg3) binds parameters server-side and rejects
    any statement with more than 65535 parameters, so a single
    ``... WHERE tic_id IN (?, ?, ...)`` over a large id list (>65k) raises
    ``OperationalError: number of parameters must be between 0 and 65535``.
    Splitting the id list into chunks keeps every statement well under the
    limit; the IN subsets are disjoint, so concatenating the per-chunk rows is
    behaviour-identical to one un-chunked query. Works unchanged on the SQLite
    backend (its own per-statement variable limit, ~32766 by default, is also
    comfortably above ``chunk_size``).

    Parameters
    ----------
    runner:
        Callable ``(sql, params) -> Iterable[Mapping]`` that executes *sql*
        and returns its rows — e.g. ``backend.query``, ``backend.tx_query``
        bound to an open transaction connection via a lambda (``lambda sql,
        params: backend.tx_query(conn, sql, params)``), or an
        ``OperationalSession.execute`` (whose ``_Result`` is iterable).
        Rows are coerced with ``dict()`` so both list-of-dict and
        list-of-Row/psycopg-row returns work.
    sql_before:
        SQL text up to (and NOT including) the opening ``(`` of the IN list,
        e.g. ``"SELECT id, claude_analysis FROM candidates WHERE id IN"``.
    ids:
        The full id collection to chunk (any iterable; consumed once).
    sql_after:
        Optional SQL text appended right after the IN list's closing ``)`` —
        e.g. an additional ``AND ...`` clause.
    params_extra:
        Extra bind params for any ``?`` placeholders inside *sql_after*;
        appended after each chunk's id placeholders, on every chunk.
    chunk_size:
        Max ids per statement. Default 30000 — comfortably under both
        backends' per-statement parameter limits.

    Returns a list of dict rows, concatenated across chunks. Implemented as
    a thin (flattening) consumer of :func:`select_in_chunks_iter` — see that
    function if a caller wants the per-chunk rows without ever holding the
    whole concatenated list in memory.
    """
    out = []
    for chunk in select_in_chunks_iter(
            runner, sql_before, ids, sql_after=sql_after,
            params_extra=params_extra, chunk_size=chunk_size):
        out.extend(chunk)
    return out


def select_in_chunks_iter(runner, sql_before, ids, sql_after="", params_extra=(),
                          chunk_size=30000):
    """Like :func:`select_in_chunks`, but a GENERATOR that yields each
    chunk's rows separately instead of concatenating them all into one list.

    Same contract otherwise: same signature, same per-chunk SQL/params
    construction, same dict-coercion of rows, same empty-``ids`` -> no calls
    behaviour (an empty *ids* simply yields nothing).

    Invariant callers rely on: the chunking is **by id**, so every row
    belonging to a given id lands in exactly one chunk — never split across
    two. That makes any per-id grouping/aggregation done *within* a single
    chunk (e.g. "best row per tic_id in this batch") safe to compute
    chunk-locally, without needing to see the whole result set at once.

    Parameters and their meaning are identical to :func:`select_in_chunks`.
    """
    ids = list(ids)
    for i in range(0, len(ids), chunk_size):
        batch = ids[i:i + chunk_size]
        placeholders = ",".join("?" * len(batch))
        sql = f"{sql_before} ({placeholders}){sql_after}"
        params = tuple(batch) + tuple(params_extra)
        yield [dict(r) for r in runner(sql, params)]


# --------------------------------------------------------------------------- #
# Operational session: a sqlite3-``con``-shaped adapter for the pipeline scripts
# --------------------------------------------------------------------------- #

class _Result:
    """A fetched result that is iterable AND exposes fetchall/fetchone.

    Rows are plain ``dict``s on both backends, so the scripts' ``dict(r)`` and
    ``r["col"]`` access work unchanged. Writes return an empty result.
    """

    __slots__ = ("_rows", "_i")

    def __init__(self, rows):
        self._rows = rows
        self._i = 0

    def __iter__(self):
        return iter(self._rows)

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        if self._i >= len(self._rows):
            return None
        row = self._rows[self._i]
        self._i += 1
        return row


class OperationalSession:
    """A single-connection, sqlite3-``con``-flavoured view over a backend.

    The discovery pipeline scripts (notably ``discovery_postprocess.py``) were
    written against a raw ``sqlite3`` connection: ``con.execute(sql, params)``
    returning a cursor they iterate / ``fetchall`` / ``fetchone``, then
    ``con.commit()`` after a batch of UPDATEs. This adapter reproduces that
    surface on top of either backend, translating the SQLite-flavoured SQL
    (``?`` placeholders, ``datetime('now')``, ``INDEXED BY``) for PG and
    returning dict rows.

    It holds ONE connection for its lifetime (SQLite: the shared connection
    under the lock; PG: a borrowed pooled connection), so a SELECT-then-UPDATE
    batch is consistent and ``commit()`` is meaningful, matching the original
    single-connection model. Use as a context manager (``with backend.op_session()
    as con:``) so the connection is released and the lock/pool freed.
    """

    def __init__(self, backend):
        self._backend = backend
        self._dialect = backend.dialect
        self._cm = backend.raw_connection()
        self._conn = self._cm.__enter__()
        self._is_pg = self._dialect.name == "postgres"

    def execute(self, sql, params=()):
        sql_t = self._dialect.translate(sql)
        if self._is_pg:
            from psycopg.rows import dict_row
            # cursor explicitly opened with row_factory=dict_row, so
            # cur.fetchall() already yields plain dicts — no dict(r) re-wrap
            # needed (item 8, 2026-07-02 cleanup review).
            cur = self._conn.cursor(row_factory=dict_row)
            cur.execute(sql_t, params)
            if cur.description is None:      # write (no result set)
                return _Result([])
            return _Result(cur.fetchall())
        # SQLite: row_factory=sqlite3.Row is NOT a dict — this dict(r) wrap
        # stays (the only backend where it's actually needed).
        cur = self._conn.execute(sql_t, params)
        try:
            rows = cur.fetchall()
        except Exception:
            rows = []
        return _Result([dict(r) for r in rows])

    def execute_stream(self, sql, params=(), batch_rows=50_000):
        """Iterate a SELECT's result in batches of dict-rows, without ever
        materializing the whole result set in memory.

        SELECT-only (this is a read seam for large scans — see
        docs/superpowers/specs/2026-08-01-postprocess-streaming.md §3.A).
        There is no active guard rejecting writes beyond what the backend
        already does for a statement with no result set; callers must not
        route INSERT/UPDATE/DELETE through this method.

        Yields lists of up to *batch_rows* dict rows, in result order,
        covering the whole result set with no gaps or overlaps (an empty
        result set simply yields nothing).

        PostgreSQL: opens a server-side (named) cursor — a fresh, uniquely
        named one per call — with ``itersize`` set to *batch_rows*, so only
        one batch is buffered client-side at a time instead of the whole
        result set. The named cursor lives INSIDE this session's open
        transaction: do **not** call :meth:`commit` on this session while a
        stream from it is still being consumed — that would tear down the
        transaction (and the cursor with it) mid-scan. The cursor is always
        closed in a ``finally`` (covers both normal exhaustion and the
        caller abandoning the generator early, which raises
        ``GeneratorExit`` at the suspended ``yield``).

        SQLite: uses the shared connection's own cursor with repeated
        ``fetchmany`` calls — there is no separate server-side-cursor
        concept. Because :class:`SqliteBackend`'s ``raw_connection()``
        holds the backend-wide lock for this whole session's lifetime (see
        the class docstring), iterating this generator to completion holds
        that lock for as long as the scan takes — the same lock an eager
        ``fetchall()`` (as in :meth:`execute`) already held, just spread
        over more wall-clock time rather than paid upfront in one pause.
        """
        sql_t = self._dialect.translate(sql)
        if self._is_pg:
            from psycopg.rows import dict_row
            cur = self._conn.cursor(name=f"exonit_stream_{uuid4().hex}",
                                     row_factory=dict_row)
            cur.itersize = batch_rows
            try:
                cur.execute(sql_t, params)
                while True:
                    rows = cur.fetchmany(batch_rows)
                    if not rows:
                        break
                    yield rows  # already plain dicts (dict_row)
            finally:
                cur.close()
            return
        # SQLite: row_factory=sqlite3.Row is NOT a dict — dict(r) wrap stays,
        # matching execute()'s per-row coercion.
        cur = self._conn.execute(sql_t, params)
        try:
            while True:
                rows = cur.fetchmany(batch_rows)
                if not rows:
                    break
                yield [dict(r) for r in rows]
        finally:
            cur.close()

    def commit(self):
        # SQLite: commit the shared connection. PG: commit the pooled
        # connection so the UPDATE batch is durable before the session closes
        # (raw_connection's exit would otherwise roll it back).
        self._conn.commit()

    def close(self):
        # Release the connection back to the backend (lock release / pool return).
        self._cm.__exit__(None, None, None)


def op_session(db_path: str | None) -> OperationalSession:
    """Open an :class:`OperationalSession` on the operational DB at *db_path*.

    Convenience wrapper that builds the backend and the session in one call.
    The session owns the backend it created and closes it on ``close()``.
    """
    backend = make_backend(db_path)
    sess = OperationalSession(backend)
    # Make close() also close the underlying backend we created here.
    _orig_close = sess.close

    def _close():
        _orig_close()
        backend.close()

    sess.close = _close
    return sess
