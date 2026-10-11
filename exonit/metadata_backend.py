"""Database backend abstraction for the TGLC metadata index: SQLite (default)
or PostgreSQL.

``exonit/metadata_cache.py``'s ``MetadataCache`` (the ``metadata`` /
``fits_paths`` / ``fits_blobs`` / ``nearby_stars`` / ``tglc`` tables — see
``docs/superpowers/specs/2026-07-20-tglc-index-to-pg.md``) historically opened
``sqlite3.connect()`` directly, and ~8-10 other call sites across ``exonit/``
opened a SECOND, independent raw connection to the same file to query
``tglc``. This module introduces the seam both routes go through:

    make_metadata_backend("data/metadata_cache.db")            -> SqliteBackend
    make_metadata_backend("postgresql://u:p@host/exonit_meta")  -> PgBackend

Mirrors ``exonit/db_backend.py``'s proven pattern (psycopg3 +
``psycopg_pool.ConnectionPool``, ``dict_row``, fail-fast on a bad DSN,
``?``-placeholder SQL auto-translated to ``%s`` for PG) but is a **dedicated,
independent module** — it does not import or share state with
``db_backend.py``, because the metadata index is (by design, D1 of the spec)
a completely separate PostgreSQL database (``exonit_meta``) from the
operational DB (``exonit``), with its own connection pool and its own
lifecycle.

Design goals
------------
* **SQLite stays bit-identical.** ``SqliteBackend`` reproduces
  ``MetadataCache``'s previous connection (WAL, ``busy_timeout=30000``,
  ``synchronous=NORMAL``, one shared ``threading.Lock``) exactly.
* **``EXONIT_METADATA_DB_URL`` unset -> zero behaviour change.** The default
  (``None``) always resolves to a ``SqliteBackend`` over whatever path the
  caller already passes (``ExoNitConfig.metadata_cache_path`` in production,
  a ``tmp_path`` file in tests) — see D2 of the spec.
* **``INSERT OR REPLACE`` semantics preserved on PG.** SQLite's REPLACE
  clobbers the whole row on a primary-key collision; the PG equivalent is
  ``INSERT ... ON CONFLICT (pk) DO UPDATE SET <every non-pk column> =
  EXCLUDED.<column>`` (NOT ``DO NOTHING`` — see ``Dialect.upsert``).
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from uuid import uuid4


def is_pg_url(url: str | None) -> bool:
    """True if *url* is a PostgreSQL DSN (postgres:// or postgresql://)."""
    return bool(url) and str(url).startswith(("postgres://", "postgresql://"))


# --------------------------------------------------------------------------- #
# SQL translation: ? placeholders -> %s, without touching string literals.
# Duplicated from db_backend.py's qmark_to_pyformat (deliberately — this
# module must not import db_backend, see the module docstring).
# --------------------------------------------------------------------------- #

def qmark_to_pyformat(sql: str) -> str:
    """Convert ``?`` placeholders to ``%s``, respecting single-quoted literals.

    psycopg uses pyformat (``%s``); SQLite uses qmark (``?``). A naive replace
    would corrupt a ``?`` inside a string literal, and would also turn a bare
    ``%`` (e.g. ``LIKE '%foo%'``) into a broken format spec — so literal ``%``
    is escaped to ``%%`` for psycopg, both inside and outside string literals.
    """
    out = []
    in_str = False
    i = 0
    n = len(sql)
    while i < n:
        c = sql[i]
        if c == "'":
            if in_str and i + 1 < n and sql[i + 1] == "'":
                out.append("''")
                i += 2
                continue
            in_str = not in_str
            out.append(c)
        elif in_str:
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


class Dialect:
    """SQLite dialect (the base/default). Overridden for PG below."""

    name = "sqlite"
    placeholder = "?"

    def translate(self, sql: str) -> str:
        """Backend-specific rewrite of a SQL string. Identity for SQLite."""
        return sql

    def now(self) -> str:
        return "datetime('now')"

    def upsert(self, table, columns, conflict_cols, raw_values=None) -> str:
        """Build an ``INSERT OR REPLACE``-equivalent for this backend.

        * ``columns``       — every column written by the INSERT, in order;
                               the VALUES list has one ``?`` per column
                               (unless overridden by ``raw_values``).
        * ``conflict_cols``  — columns of the PK/unique constraint to upsert on.
        * ``raw_values``     — optional ``{column: raw_sql_expr}`` map for
                               columns whose VALUES entry is a raw SQL
                               expression (e.g. ``"datetime('now')"``) rather
                               than a bound ``?`` — for DB-clock timestamp
                               columns (``updated_at``). The caller's params
                               tuple must OMIT those columns (no placeholder
                               emitted for them).

        SQLite: ``INSERT OR REPLACE`` — bit-identical to ``MetadataCache``'s
        historical behaviour (clobbers the WHOLE row on a PK collision, every
        column not in ``columns`` reverts to its default/NULL — none of
        MetadataCache's REPLACE call sites relied on partial-column
        preservation, so this matches every existing call site verbatim).

        Placeholders are always ``?`` here; the backend's ``execute``/
        ``executemany`` runs the returned SQL through ``translate()`` (``?``
        -> ``%s`` for PG). Emitting ``%s`` directly would be double-escaped.
        """
        raw_values = raw_values or {}
        cols = ", ".join(columns)
        ph = ", ".join(raw_values.get(c, "?") for c in columns)
        return f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({ph})"


class PgDialect(Dialect):
    """PostgreSQL dialect."""

    name = "postgres"
    placeholder = "%s"

    def translate(self, sql: str) -> str:
        sql = _DATETIME_NOW_RE.sub("NOW()", sql)
        sql = qmark_to_pyformat(sql)
        return sql

    def now(self) -> str:
        return "NOW()"

    def upsert(self, table, columns, conflict_cols, raw_values=None) -> str:
        """``INSERT ... ON CONFLICT (pk) DO UPDATE SET ...`` — REPLACE semantics.

        Every column not in ``conflict_cols`` is refreshed from ``EXCLUDED``,
        matching SQLite's ``INSERT OR REPLACE`` full-row-clobber behaviour
        (D4 of the spec: "NOT DO NOTHING — REPLACE overwrites; preserve that").
        """
        raw_values = raw_values or {}
        update_cols = [c for c in columns if c not in conflict_cols]
        cols = ", ".join(columns)
        ph = ", ".join(raw_values.get(c, "?") for c in columns)
        conflict = ", ".join(conflict_cols)
        if update_cols:
            sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
            action = f"DO UPDATE SET {sets}"
        else:
            action = "DO NOTHING"
        return (f"INSERT INTO {table} ({cols}) VALUES ({ph}) "
                f"ON CONFLICT ({conflict}) {action}")


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

    def iter_id_column(self, sql, params=(), *, batch_rows=50_000):
        """Yield column-0 values of a single-column SELECT, one at a time.

        Q3 (perf review 2026-09-26, docs/reviews/2026-09-26-enqueue-path-
        performance-review.md): mirrors ``db_backend.Backend.iter_id_column``
        (deliberately duplicated, not imported — this module does not depend
        on ``db_backend.py``, see the module docstring) for
        ``MetadataCache.tglc_tic_ids``/``all_tic_ids``, which read millions
        of ``tglc``/``metadata`` rows for their id column alone but paid for
        one dict_row (PG) / ``Row`` (SQLite) object per row via ``query()``.
        Base-class fallback: correctness over memory optimality.
        """
        for r in self.query(sql, params):
            yield r[0] if not isinstance(r, dict) else next(iter(r.values()))

    @contextmanager
    def transaction(self):
        """Yield an object with ``.execute(sql, params)`` and
        ``.executemany(sql, seq_of_params)``; commits on exit, rolls back on
        exception. Used where several statements must be atomic (e.g.
        ``add_nearby_stars``'s DELETE + per-star INSERTs, or
        ``exonit/spoc_bulk.py``'s per-sector DELETE + bulk INSERT)."""
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class _SqliteTx:
    """``.execute()`` adapter over the shared SQLite connection, used inside
    ``SqliteBackend.transaction()`` (the lock is already held by the caller)."""

    __slots__ = ("_conn",)

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=()):
        self._conn.execute(sql, params)

    def executemany(self, sql, seq_of_params):
        self._conn.executemany(sql, seq_of_params)


class SqliteBackend(Backend):
    """SQLite backend — reproduces MetadataCache's previous connection exactly."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                    exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # Matches MetadataCache's historical pragmas exactly (WAL for
        # concurrent readers during writes; synchronous=NORMAL halves fsync
        # cost while keeping durability for full commits; busy_timeout=30s —
        # the unified value per D3 of the spec — lets a request wait for a
        # brief writer instead of failing immediately with SQLITE_BUSY).
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=30000")
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
            row = self.conn.execute(sql, params).fetchone()
            return row[0] if row is not None else None

    def iter_id_column(self, sql, params=(), *, batch_rows=50_000):
        # Dedicated cursor with row_factory=None (plain tuples), not the
        # connection's own sqlite3.Row default set in __init__ — see
        # db_backend.SqliteBackend.iter_id_column's identical rationale.
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
                yield _SqliteTx(self.conn)
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    @contextmanager
    def raw_connection(self):
        """Yield the shared connection for construction-time DDL (schema
        self-provisioning) that doesn't fit the query/execute shape — e.g.
        MetadataCache.__init__'s ``CREATE TABLE IF NOT EXISTS`` +
        ``ALTER TABLE ADD COLUMN`` migration block. Caller commits explicitly.
        """
        with self._lock:
            yield self.conn

    def close(self):
        with self._lock:
            self.conn.close()


class _PgTx:
    """``.execute()`` adapter over a psycopg connection borrowed for the
    lifetime of a ``PgBackend.transaction()`` block."""

    __slots__ = ("_conn", "_dialect")

    def __init__(self, conn, dialect):
        self._conn = conn
        self._dialect = dialect

    def execute(self, sql, params=()):
        self._conn.execute(self._dialect.translate(sql), params)

    def executemany(self, sql, seq_of_params):
        self._conn.cursor().executemany(
            self._dialect.translate(sql), list(seq_of_params))


def _int_env(name: str, default: int) -> int:
    """Read a non-negative int from an env var; fall back to *default* on
    anything malformed (missing, non-numeric, negative). Never raises — these
    backends are built at server startup, so a bad env var must not block it.
    Duplicated from db_backend.py's ``_int_env`` (deliberately — see the
    module docstring: this module does not import db_backend).
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
    ``ConnectionPool`` sets none of this by default. Duplicated from
    db_backend.py's ``_pg_connect_kwargs`` (deliberately, same reason as
    above — this is a completely separate pool onto ``exonit_meta``).

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

    Auto-translates each SQL string via ``PgDialect.translate`` (``?`` ->
    ``%s``, ``datetime('now')`` -> ``NOW()``) so the SQLite-flavoured SQL in
    ``metadata_cache.py`` runs unchanged. Rows are returned as plain dicts.
    """

    def __init__(self, url: str, min_size: int = 1, max_size: int = 8):
        import psycopg  # noqa: F401  (import-time check — fail fast if missing)
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        self.url = url
        self.dialect = PgDialect()
        # open=True starts the pool's background connection attempts
        # immediately (matching db_backend.py's PgBackend). Note this does
        # NOT raise at construction for an unreachable host — psycopg_pool
        # retries in a background thread; the first real failure surfaces on
        # the first actual query (pool.connection() times out). A malformed
        # DSN string itself (unparseable) does raise here, synchronously.
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
            return cur.fetchall()  # dict_row -> already plain dicts

    def query_one(self, sql, params=()):
        with self.pool.connection() as conn:
            cur = conn.execute(self._t(sql), params)
            return cur.fetchone()  # already a dict, or None

    def query_scalar(self, sql, params=()):
        with self.pool.connection() as conn:
            cur = conn.execute(self._t(sql), params)
            row = cur.fetchone()
            if row is None:
                return None
            return next(iter(row.values()))

    def iter_id_column(self, sql, params=(), *, batch_rows=50_000):
        # Server-side (named) cursor with row_factory=tuple_row, overriding
        # the pooled connection's default dict_row (see __init__) — same
        # rationale as db_backend.PgBackend.iter_id_column.
        from psycopg.rows import tuple_row
        with self.pool.connection() as conn:
            cur = conn.cursor(name=f"exonit_meta_idcol_{uuid4().hex}",
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
                yield _PgTx(conn, self.dialect)

    def close(self):
        self.pool.close()


def make_metadata_backend(url_or_path: str) -> Backend:
    """Build the right backend for *url_or_path*.

    A PostgreSQL DSN (``postgres://``/``postgresql://``) -> :class:`PgBackend`;
    anything else (a filesystem path) -> :class:`SqliteBackend`. Unlike
    ``db_backend.make_backend``, this never reads an environment variable
    itself — ``MetadataCache.__init__`` resolves ``EXONIT_METADATA_DB_URL``
    (see its docstring) and passes the resolved target here.
    """
    if is_pg_url(url_or_path):
        return PgBackend(url_or_path)
    return SqliteBackend(url_or_path)
