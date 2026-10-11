#!/usr/bin/env python3
"""F2 — TGLC metadata index migration: SQLite (data/metadata_cache.db) ->
PostgreSQL (``exonit_meta`` on <host>).

Spec: docs/superpowers/specs/2026-07-20-tglc-index-to-pg.md (D6/D8).
F1 (seam + schema): exonit/metadata_backend.py, exonit/metadata_cache.py,
scripts/migration/schema_pg_meta.sql (committed 61d530c9).

Tables migrated (streaming, never materialized in full): ``tglc``,
``fits_blobs``, ``metadata``, ``fits_paths``, ``nearby_stars``. Measured row
counts (2026-07-20, read-only): tglc ~27.7M, fits_blobs ~30M, metadata
9,018,117, fits_paths 9,164,693, nearby_stars 330,042,471 (the dominant
table).

Per D6 (UPDATED 2026-07-20): the writer (s7 download) is STOPPED before this
tool runs — a single clean snapshot + verification is sufficient; no
delta/catch-up pass is needed (superseded the original snapshot+delta design).

Usage
-----
  # Migrate everything (fresh start; TRUNCATEs each PG table first):
  python3 scripts/migration/migrate_metadata_to_pg.py \\
      --sqlite data/metadata_cache.db \\
      --pg-dsn postgresql://exonit@localhost:5432/exonit_meta

  # Migrate one table only:
  python3 scripts/migration/migrate_metadata_to_pg.py --table fits_paths ...

  # Resume an interrupted run (continues from the last committed checkpoint,
  # does NOT truncate):
  python3 scripts/migration/migrate_metadata_to_pg.py --resume ...

  # Verification gate only (no writes):
  python3 scripts/migration/migrate_metadata_to_pg.py --verify-only ...

Design decisions (per the F2 task's explicit "decide and document" asks)
--------------------------------------------------------------------------
1. **Resumability: checkpoint-per-batch, not truncate-and-restart — with the
   checkpoint committed IN THE SAME PG TRANSACTION as the data it describes.**
   ``nearby_stars`` alone is 330M rows and the spec estimates its load at
   1-2h; truncate-and-restart would throw away all progress on any failure
   near the end of that load, which is not acceptable at this scale for a
   quality-over-speed project. Every table's COPY is broken into batches of
   ``--batch-rows`` (default 200,000) rows: each batch is one ``COPY ...
   FROM STDIN`` statement (COPY cannot be interleaved with a mid-stream
   commit — a single COPY is one atomic SQL command) followed by an
   ``INSERT ... ON CONFLICT DO UPDATE`` into a bookkeeping table in
   ``exonit_meta`` itself (``_migration_checkpoint`` — created lazily by this
   script, NOT part of ``schema_pg_meta.sql``) — **and only THEN one single
   ``pg_conn.commit()`` for both**. ``--resume`` reads its starting position
   from ``_migration_checkpoint`` (authoritative — see "Review fixes" below),
   not from a local file, so it is mathematically impossible for the
   checkpoint to describe a batch PG doesn't actually have (or vice versa):
   they are the same transaction. A local JSON mirror
   (``data/migration_checkpoints/metadata_to_pg/<table>.json``) is still
   written, AFTER the atomic PG commit succeeds, purely as an
   operator-facing progress display — it is NEVER consulted to decide where
   ``--resume`` continues from, so a crash between the atomic PG commit and
   that local write is harmless by construction. A fresh (non-``--resume``)
   run always TRUNCATEs and clears any ``_migration_checkpoint`` row for
   that table, ignoring whatever state either side was in.

   **Review fixes (2026-07-20, addressing `progress/f2-migration-tool-review.md`,
   CHANGES_REQUESTED — 1 blocker):** the ORIGINAL version of this decision
   wrote the checkpoint to the local JSON file only, AFTER a separate
   ``pg_conn.commit()`` for the COPY batch — a real crash window between
   those two steps (reproduced by the reviewer against real PG) left the
   on-disk checkpoint pointing at the *previous* batch while PG already had
   the *next* one durably committed; ``--resume`` then blindly trusted the
   stale ``last_rowid``, re-selected rows PG already had, and hit an
   uncaught ``UniqueViolation`` — non-self-healing, since the checkpoint
   file was never corrected and every subsequent ``--resume`` repeated the
   exact same failure. Moving the checkpoint into the SAME PG transaction as
   the COPY (rather than reconciling a separate checkpoint against PG's
   state after the fact, the other option the review offered) closes the
   window entirely rather than shrinking or gracefully handling it — this is
   "mathematically window-free," not "handles the window better."
2. **Ordering invariant (D6, "added post-F1-review"): a single continuous
   pass, not a per-sector loop.** ``tglc``'s ``seq`` MUST reproduce the
   source SQLite ``rowid`` order exactly, both within each sector (the gate
   checks ``tglc_tic_ids(sector)`` per sector) and globally (the campaign
   cursor semantics ``discovery_state.py`` depends on, per D6-original).
   Grouping the COPY by sector (i.e. "copy all of sector 1, then all of
   sector 2, ...") would still satisfy the per-sector check but would NOT
   preserve the true global insertion order across sectors if the source
   data was ever touched out of strict sector order (re-runs, corrections).
   This migrator instead does ONE ``SELECT ... WHERE rowid > ? ORDER BY
   rowid ASC`` pass over the *whole* ``tglc`` table (batched for
   checkpointing) and COPYs straight into the partitioned parent table (PG
   routes each row to the right partition automatically) — ``seq`` is a
   plain per-row counter over that single global stream, so both per-sector
   and global order are exact mirrors of the source ``rowid`` order by
   construction, not by a "it happens to work out" side effect. Partitions
   for every distinct sector present in the source are created up front
   (``ensure_tglc_partitions``, before the COPY starts).
3. **Index strategy: drop ``tglc``'s 3 secondary indexes before the COPY,
   recreate + ANALYZE after; every other table's PRIMARY KEY (the only
   index it has) is kept live throughout.** Per the schema (F1's
   ``schema_pg_meta.sql``), ``tglc`` is the ONLY table with secondary
   indexes beyond its PK (``ix_tglc_tic``, ``ix_tglc_sector``,
   ``ix_tglc_path``) — ``fits_blobs``/``metadata``/``fits_paths``/
   ``nearby_stars`` only have their PRIMARY KEY, so "COPY with only
   PK/minimal structure, secondary indexes after" is already the shape of
   those 4 tables' DDL; nothing extra to defer for them. The PK itself is
   kept live on every table (not dropped/recreated) because: (a) it is the
   sole integrity guard against an unexpected duplicate-key row surfacing
   from the SQLite snapshot — this is a one-time snapshot load into a
   TRUNCATEd table, so no legitimate duplicates are expected (the source
   SQLite side already enforced its own PK via ``INSERT OR REPLACE`` at
   write time), and a COPY-time PK violation is exactly the kind of
   data-quality surprise this tool should hard-fail on, not silently work
   around by dropping the constraint; (b) at ``tglc``'s ~27.7M-row scale
   (not the dominant table — that's ``nearby_stars`` at 330M, which has NO
   secondary indexes to defer in the first place) the PK-maintenance cost
   during COPY is bounded and acceptable.
4. **Verification sampling: PK-based reservoir sampling, not
   position/offset sampling.** A byte-for-byte 5,000-row sample must compare
   the SAME logical rows in both backends. SQLite's ``rowid`` has no PG
   analogue and PG's physical row order (``ctid``) has no relationship to
   SQLite's ``rowid`` order, so "every Nth physical row" would silently
   compare unrelated rows in the two systems. Instead: a single streaming
   pass over the SQLite source's PK column(s) only (bounded memory — never
   the full table), using textbook Algorithm-R reservoir sampling with a
   FIXED seed (``SAMPLE_SEED``) for reproducibility, to pick exactly 5,000
   PK tuples; then both backends are queried by those SAME PK values and
   compared field by field.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_SQLITE = ROOT / "data" / "metadata_cache.db"
DEFAULT_CHECKPOINT_DIR = ROOT / "data" / "migration_checkpoints" / "metadata_to_pg"

# Tables migrated, in a deliberate order: cheap/simple tables first (fast
# feedback that the pipe end-to-end works), the ordering-sensitive `tglc`
# next while it's front of mind, and `nearby_stars` (the dominant table,
# 330M rows, the long pole) last.
TABLES = ["fits_paths", "metadata", "fits_blobs", "tglc", "nearby_stars"]

PK_COLUMNS = {
    "tglc": ["tic_id", "sector"],
    "fits_blobs": ["filename"],
    "metadata": ["tic_id"],
    "fits_paths": ["filename"],
    "nearby_stars": ["tic_id", "neighbor_id"],
}

# Source (SQLite) column order == destination (PG) column order, except
# `tglc` which gains a `seq` column the SQLite source has no equivalent of
# (see design decision 2 above).
SOURCE_COLUMNS = {
    "tglc": ["tic_id", "sector", "path", "exptime", "tmag", "ra", "dec"],
    "fits_blobs": ["filename", "container_key", "data_offset", "data_length"],
    "metadata": ["tic_id", "sectors", "exptime", "fits_files", "stellar_radius",
                 "teff", "logg", "tmag", "ra", "dec", "updated_at"],
    "fits_paths": ["filename", "full_path"],
    "nearby_stars": ["tic_id", "neighbor_id", "ra", "dec", "mass", "rad",
                      "teff", "tmag", "plx", "gal_b", "updated_at"],
}

PG_COLUMNS = dict(SOURCE_COLUMNS)
PG_COLUMNS["tglc"] = SOURCE_COLUMNS["tglc"] + ["seq"]

DATETIME_COLUMNS = {"metadata": {"updated_at"}, "nearby_stars": {"updated_at"}}

# tglc's 3 secondary indexes (schema_pg_meta.sql) — dropped before the COPY,
# recreated after (design decision 3). No other migrated table has a
# secondary index (only their PRIMARY KEY), so this dict has one entry.
SECONDARY_INDEXES = {
    "tglc": [
        ("ix_tglc_tic", "CREATE INDEX ix_tglc_tic ON tglc (tic_id)"),
        ("ix_tglc_sector", "CREATE INDEX ix_tglc_sector ON tglc (sector)"),
        ("ix_tglc_path", "CREATE INDEX ix_tglc_path ON tglc (path)"),
    ],
}

DEFAULT_BATCH_ROWS = 200_000
PROGRESS_EVERY = 1_000_000
SAMPLE_SEED = 20260720
SAMPLE_SIZE = 5000
_FLOAT_TOL = 1e-9

_FITS_BLOBS_SECTOR_RE = re.compile(r"-s(\d{4})-")


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fmt_eta(seconds: float) -> str:
    if seconds == float("inf") or seconds != seconds:  # inf or NaN
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


# --------------------------------------------------------------------------- #
# Connections
# --------------------------------------------------------------------------- #

def connect_sqlite_ro(path) -> sqlite3.Connection:
    """Open *path* STRICTLY read-only via a ``file:...?mode=ro`` URI.

    Never writes to the live index — the download/indexer's own writer is
    the only process allowed to touch this file (and per D6, it is stopped
    for the duration of this migration anyway).
    """
    uri = f"file:{Path(path).resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=300.0)
    conn.row_factory = sqlite3.Row
    return conn


def connect_pg(dsn: str):
    import psycopg

    conn = psycopg.connect(dsn, autocommit=False, connect_timeout=10)
    # metadata.updated_at / nearby_stars.updated_at are copied as raw
    # SQLite-formatted ISO-ish text (no offset); PG's COPY TEXT format
    # parses that natively as a naive timestamp interpreted in the session
    # timezone. Force UTC so it matches migrate_sqlite_to_pg.py's
    # "naive == UTC" convention exactly (see that script's parse_dt).
    conn.execute("SET TIME ZONE 'UTC'")
    return conn


def _db_name(dsn: str) -> str:
    tail = dsn.rsplit("/", 1)[-1]
    return tail.split("?", 1)[0]


def assert_not_operational_db(dsn: str) -> None:
    """Hard guard: refuse a DSN pointing at the operational ``exonit`` DB.

    Per the spec's D1/hard constraints: this tool's target is the SEPARATE
    ``exonit_meta`` database; ``exonit`` (candidates/tasks/stars/...) must
    never be written to by this script.
    """
    name = _db_name(dsn)
    if name == "exonit":
        raise SystemExit(
            "REFUSING: --pg-dsn points at the operational 'exonit' database. "
            "This tool migrates the TGLC metadata index into the SEPARATE "
            "'exonit_meta' database (D1 of the spec) — pass a DSN ending in "
            "/exonit_meta (or an exonit_meta* throwaway for tests)."
        )


# --------------------------------------------------------------------------- #
# Checkpointing — PG (`_migration_checkpoint`, AUTHORITATIVE for --resume,
# committed atomically with each COPY batch) + a local JSON mirror
# (informational only, never consulted for a resume decision).
# See design decision 1 / "Review fixes" in the module docstring.
# --------------------------------------------------------------------------- #

_CHECKPOINT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS _migration_checkpoint (
    table_name  TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    last_rowid  BIGINT NOT NULL,
    rows_copied BIGINT NOT NULL,
    seq_next    BIGINT,
    elapsed_s   DOUBLE PRECISION NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_CHECKPOINT_UPSERT_SQL = """
INSERT INTO _migration_checkpoint
    (table_name, status, last_rowid, rows_copied, seq_next, elapsed_s, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, NOW())
ON CONFLICT (table_name) DO UPDATE SET
    status = EXCLUDED.status,
    last_rowid = EXCLUDED.last_rowid,
    rows_copied = EXCLUDED.rows_copied,
    seq_next = EXCLUDED.seq_next,
    elapsed_s = EXCLUDED.elapsed_s,
    updated_at = NOW()
"""


def ensure_checkpoint_table(pg_conn) -> None:
    """Idempotent — safe to call at the start of every ``migrate_table()``
    invocation (fresh DB, or one that already has the table from a prior
    run)."""
    pg_conn.execute(_CHECKPOINT_TABLE_SQL)
    pg_conn.commit()


def pg_load_checkpoint(pg_conn, table: str) -> Optional[dict]:
    """Read *table*'s checkpoint row from ``_migration_checkpoint`` —
    AUTHORITATIVE for ``--resume`` (see module docstring). ``None`` if no
    prior attempt touched this table."""
    row = pg_conn.execute(
        "SELECT status, last_rowid, rows_copied, seq_next, elapsed_s "
        "FROM _migration_checkpoint WHERE table_name = %s", (table,)
    ).fetchone()
    if row is None:
        return None
    status, last_rowid, rows_copied, seq_next, elapsed_s = row
    return {
        "table": table, "status": status, "last_rowid": last_rowid,
        "rows_copied": rows_copied, "seq_next": seq_next,
        "elapsed_s": elapsed_s,
    }


def pg_write_checkpoint(pg_conn, table: str, *, status: str, last_rowid: int,
                         rows_copied: int, seq_next: Optional[int],
                         elapsed_s: float) -> None:
    """Upsert *table*'s checkpoint row — does NOT commit. The caller MUST
    commit this together with the COPY batch it describes (same
    transaction) for the atomicity guarantee to hold; see
    ``migrate_table``."""
    pg_conn.execute(_CHECKPOINT_UPSERT_SQL, (
        table, status, last_rowid, rows_copied, seq_next, elapsed_s))


def pg_clear_checkpoint(pg_conn, table: str) -> None:
    pg_conn.execute("DELETE FROM _migration_checkpoint WHERE table_name = %s", (table,))


# --- Local JSON mirror (informational only — see module docstring) --------

def checkpoint_path(checkpoint_dir: Path, table: str) -> Path:
    return Path(checkpoint_dir) / f"{table}.json"


def load_checkpoint(checkpoint_dir: Path, table: str) -> Optional[dict]:
    p = checkpoint_path(checkpoint_dir, table)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def save_checkpoint(checkpoint_dir: Path, table: str, state: dict) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    p = checkpoint_path(checkpoint_dir, table)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(p)  # atomic on POSIX — no torn/partial checkpoint on crash


def clear_checkpoint(checkpoint_dir: Path, table: str) -> None:
    p = checkpoint_path(checkpoint_dir, table)
    if p.exists():
        p.unlink()


# --------------------------------------------------------------------------- #
# SQLite-side helpers
# --------------------------------------------------------------------------- #

def sqlite_total_count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def sqlite_tglc_sector_counts(conn: sqlite3.Connection) -> dict:
    rows = conn.execute("SELECT sector, COUNT(*) FROM tglc GROUP BY sector").fetchall()
    return {int(r[0]): int(r[1]) for r in rows}


def sqlite_distinct_tglc_sectors(conn: sqlite3.Connection) -> list:
    rows = conn.execute("SELECT DISTINCT sector FROM tglc ORDER BY sector").fetchall()
    return [int(r[0]) for r in rows]


def distinct_fits_blobs_sectors(conn: sqlite3.Connection, batch_rows: int) -> set:
    """Sectors present in ``fits_blobs``, parsed from the ``-sNNNN-``
    filename pattern (mirrors ``MetadataCache.indexed_basenames``'s LIKE
    pattern). Streamed in rowid batches — never materializes the full
    filename list."""
    sectors: set = set()
    after_rowid = 0
    while True:
        rows = conn.execute(
            "SELECT rowid AS __rowid, filename FROM fits_blobs "
            "WHERE rowid > ? ORDER BY rowid ASC LIMIT ?",
            (after_rowid, batch_rows),
        ).fetchall()
        if not rows:
            break
        for r in rows:
            m = _FITS_BLOBS_SECTOR_RE.search(r["filename"] or "")
            if m:
                sectors.add(int(m.group(1)))
        after_rowid = rows[-1]["__rowid"]
        if len(rows) < batch_rows:
            break
    return sectors


# --------------------------------------------------------------------------- #
# PG-side DDL helpers
# --------------------------------------------------------------------------- #

def ensure_tglc_partitions(pg_conn, sectors) -> None:
    """Create every ``tglc_s<NNNN>`` LIST partition for *sectors* that
    doesn't already exist. Deliberately duplicated (not imported) from
    ``exonit.metadata_cache.MetadataCache.ensure_tglc_partition`` — this
    script manages its own single ``pg_conn``/transaction and shouldn't take
    a dependency on that class's separate pooled-connection lifecycle for a
    one-line DDL statement (same rationale metadata_backend.py's module
    docstring gives for duplicating ``qmark_to_pyformat``: script
    independence over avoiding a few duplicated lines).

    2026-07-20 review fix (progress/f3-indexer-pg-review.md, point 5,
    BLOCKER — fixed "for consistency" here too, per the review's explicit
    ask, even though this script has no concurrent-caller pattern today):
    ``CREATE TABLE IF NOT EXISTS ... PARTITION OF`` is not atomic against a
    concurrent identical creator in PostgreSQL (reproduced empirically
    against real PG — see ``exonit.metadata_cache.MetadataCache.
    _partition_race_excs``'s docstring for the full explanation). Unlike
    that method (which borrows a fresh pooled connection per call, so a
    failed statement's transaction is isolated automatically), THIS script
    keeps one single long-lived ``pg_conn`` across the whole per-sector loop
    — an uncaught error here would leave the *shared* transaction aborted,
    poisoning every subsequent statement (including the next sector's
    CREATE) until a ROLLBACK. Each sector's CREATE therefore runs inside its
    own ``pg_conn.transaction()`` block — a SAVEPOINT when nested inside an
    already-open transaction (psycopg3's automatic behaviour) — so a lost
    race only unwinds to that savepoint, not the whole outer transaction."""
    import psycopg

    for s in sectors:
        table = f"tglc_s{int(s):04d}"
        try:
            with pg_conn.transaction():
                pg_conn.execute(
                    f"CREATE TABLE IF NOT EXISTS {table} PARTITION OF tglc "
                    f"FOR VALUES IN ({int(s)})")
        except (psycopg.errors.DuplicateTable, psycopg.errors.UniqueViolation):
            # Lost a race against a concurrent creator — the partition
            # exists (created by the winner), exactly the outcome
            # "IF NOT EXISTS" was meant to guarantee. Benign.
            pass


def drop_secondary_indexes(pg_conn, table: str) -> None:
    for name, _sql in SECONDARY_INDEXES.get(table, []):
        pg_conn.execute(f"DROP INDEX IF EXISTS {name}")


def recreate_secondary_indexes(pg_conn, table: str) -> None:
    for name, sql in SECONDARY_INDEXES.get(table, []):
        pg_conn.execute(sql)


def bump_tglc_sequence(pg_conn) -> None:
    """Advance ``tglc_seq_seq`` past the highest ``seq`` this migration just
    wrote, so the FIRST ``nextval()`` after cutover (F3/F4's
    ``tglc_add_rows_bulk``, via the DEFAULT-bound sequence) continues
    strictly after the migrated data instead of colliding with it. Needed
    because this migrator assigns ``seq`` explicitly per row (bypassing the
    column DEFAULT) — see design decision 2."""
    pg_conn.execute(
        "SELECT setval('tglc_seq_seq', COALESCE((SELECT MAX(seq) FROM tglc), 1))"
    )


# --------------------------------------------------------------------------- #
# Migration (COPY)
# --------------------------------------------------------------------------- #

def migrate_table(
    sq_conn: sqlite3.Connection,
    pg_conn,
    table: str,
    *,
    resume: bool,
    checkpoint_dir: Path,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> int:
    """Migrate one table, batched + checkpointed. Returns rows copied
    (cumulative, including rows copied by a prior interrupted attempt when
    resuming)."""
    columns = SOURCE_COLUMNS[table]
    pg_cols = PG_COLUMNS[table]
    select_cols = ", ".join(columns)

    ensure_checkpoint_table(pg_conn)

    # AUTHORITATIVE source for --resume is PG's own _migration_checkpoint
    # table (committed atomically with the data — see design decision 1 /
    # "Review fixes" in the module docstring), NOT the local JSON mirror.
    ckpt = pg_load_checkpoint(pg_conn, table) if resume else None
    if resume and ckpt is None:
        log(f"[{table}] --resume given but no PG checkpoint found — starting fresh")
    if ckpt is not None and ckpt.get("status") == "completed":
        log(f"[{table}] PG checkpoint says already completed "
            f"({ckpt['rows_copied']:,} rows) — skipping. DELETE FROM "
            f"_migration_checkpoint WHERE table_name = '{table}' to force a redo.")
        return ckpt["rows_copied"]

    is_tglc = table == "tglc"
    if ckpt is None:
        pg_conn.execute(f"TRUNCATE TABLE {table}")
        pg_clear_checkpoint(pg_conn, table)
        pg_conn.commit()
        if is_tglc:
            sectors = sqlite_distinct_tglc_sectors(sq_conn)
            log(f"[{table}] {len(sectors)} distinct sectors in source — "
                f"ensuring partitions before COPY")
            ensure_tglc_partitions(pg_conn, sectors)
            drop_secondary_indexes(pg_conn, table)
            pg_conn.commit()
        after_rowid = 0
        rows_copied = 0
        seq_next = 1
        elapsed_before = 0.0
    else:
        if is_tglc:
            # Idempotent: a sector introduced since the interrupted attempt
            # (shouldn't happen mid-migration with the writer stopped, but
            # cheap to guard) gets its partition too; the secondary indexes
            # were already dropped by the interrupted attempt (DROP INDEX
            # IF EXISTS makes re-dropping here a safe no-op).
            sectors = sqlite_distinct_tglc_sectors(sq_conn)
            ensure_tglc_partitions(pg_conn, sectors)
            drop_secondary_indexes(pg_conn, table)
            pg_conn.commit()
        after_rowid = ckpt["last_rowid"]
        rows_copied = ckpt["rows_copied"]
        seq_next = ckpt.get("seq_next") or 1
        elapsed_before = ckpt.get("elapsed_s") or 0.0
        log(f"[{table}] resuming from PG-authoritative checkpoint: "
            f"rowid={after_rowid} ({rows_copied:,} rows already committed)")

    total_rows = sqlite_total_count(sq_conn, table)
    col_list = ", ".join(f'"{c}"' for c in pg_cols)
    copy_sql = (f'COPY {table} ({col_list}) FROM STDIN '
                f"(FORMAT TEXT, DELIMITER E'\\t', NULL '\\N')")

    t_start = time.monotonic() - elapsed_before
    last_progress_rows = rows_copied - (rows_copied % PROGRESS_EVERY)

    while True:
        rows = sq_conn.execute(
            f"SELECT rowid AS __rowid, {select_cols} FROM {table} "
            f"WHERE rowid > ? ORDER BY rowid ASC LIMIT ?",
            (after_rowid, batch_rows),
        ).fetchall()
        if not rows:
            break

        with pg_conn.cursor().copy(copy_sql) as copy:
            for raw in rows:
                values = [raw[c] for c in columns]
                if is_tglc:
                    values.append(seq_next)
                    seq_next += 1
                copy.write_row(values)

        after_rowid = rows[-1]["__rowid"]
        rows_copied += len(rows)
        elapsed_s = time.monotonic() - t_start
        # ATOMIC with the COPY above: same open transaction, ONE commit for
        # both. This is what closes the crash window the review reproduced
        # — the checkpoint literally cannot be behind PG's true state.
        pg_write_checkpoint(pg_conn, table, status="in_progress",
                            last_rowid=after_rowid, rows_copied=rows_copied,
                            seq_next=seq_next if is_tglc else None,
                            elapsed_s=elapsed_s)
        pg_conn.commit()
        # Local JSON mirror — informational only, written AFTER the atomic
        # PG commit above; never consulted for a --resume decision, so a
        # crash right here (between the PG commit and this write) is
        # harmless (see module docstring "Review fixes" and
        # tests/test_migrate_metadata_to_pg.py's
        # TestCrashBetweenPgCommitAndLocalMirror).
        save_checkpoint(checkpoint_dir, table, {
            "table": table, "status": "in_progress",
            "last_rowid": after_rowid, "rows_copied": rows_copied,
            "seq_next": seq_next if is_tglc else None,
            "elapsed_s": elapsed_s,
            "updated_at": now_iso(),
        })

        if rows_copied - last_progress_rows >= PROGRESS_EVERY or len(rows) < batch_rows:
            elapsed = time.monotonic() - t_start
            rate = rows_copied / elapsed if elapsed > 0 else 0.0
            remaining = max(total_rows - rows_copied, 0)
            eta = remaining / rate if rate > 0 else float("inf")
            log(f"[{table}] {rows_copied:,}/{total_rows:,} rows "
                f"({rate:,.0f} rows/s, ETA {fmt_eta(eta)})")
            last_progress_rows = rows_copied

        if len(rows) < batch_rows:
            break  # short read => source exhausted

    if is_tglc:
        log(f"[{table}] recreating secondary indexes...")
        recreate_secondary_indexes(pg_conn, table)
        bump_tglc_sequence(pg_conn)
    log(f"[{table}] ANALYZE...")
    pg_conn.execute(f"ANALYZE {table}")
    elapsed_s = time.monotonic() - t_start
    # Same atomicity discipline for the final "completed" marker: committed
    # together with ANALYZE/index-recreate, not as a separate step.
    pg_write_checkpoint(pg_conn, table, status="completed",
                        last_rowid=after_rowid, rows_copied=rows_copied,
                        seq_next=seq_next if is_tglc else None,
                        elapsed_s=elapsed_s)
    pg_conn.commit()

    save_checkpoint(checkpoint_dir, table, {
        "table": table, "status": "completed",
        "last_rowid": after_rowid, "rows_copied": rows_copied,
        "seq_next": seq_next if is_tglc else None,
        "elapsed_s": elapsed_s,
        "updated_at": now_iso(),
    })
    log(f"[{table}] done: {rows_copied:,} rows.")
    return rows_copied


# --------------------------------------------------------------------------- #
# Verification gate
# --------------------------------------------------------------------------- #

def reservoir_sample_pks(
    conn: sqlite3.Connection, table: str, pk_cols, sample_size: int,
    seed: int, batch_rows: int,
) -> list:
    """Textbook Algorithm-R reservoir sampling over the source PK column(s),
    streamed in rowid batches. Returns exactly ``min(sample_size, N)`` PK
    tuples, reproducible for a fixed *seed* regardless of *batch_rows* (each
    item's accept/reject decision depends only on how many items have been
    seen so far, not on batch boundaries)."""
    rng = random.Random(seed)
    col_list = ", ".join(pk_cols)
    reservoir: list = []
    seen = 0
    after_rowid = 0
    while True:
        rows = conn.execute(
            f"SELECT rowid AS __rowid, {col_list} FROM {table} "
            f"WHERE rowid > ? ORDER BY rowid ASC LIMIT ?",
            (after_rowid, batch_rows),
        ).fetchall()
        if not rows:
            break
        for r in rows:
            seen += 1
            pk = tuple(r[c] for c in pk_cols)
            if len(reservoir) < sample_size:
                reservoir.append(pk)
            else:
                j = rng.randint(0, seen - 1)
                if j < sample_size:
                    reservoir[j] = pk
        after_rowid = rows[-1]["__rowid"]
        if len(rows) < batch_rows:
            break
    return reservoir


def sqlite_field_row(conn, table, columns, pk_cols, pk_values) -> Optional[dict]:
    where = " AND ".join(f"{c} = ?" for c in pk_cols)
    col_list = ", ".join(columns)
    row = conn.execute(
        f"SELECT {col_list} FROM {table} WHERE {where}", pk_values).fetchone()
    return dict(row) if row is not None else None


def pg_lookup_row(pg_conn, table, columns, pk_cols, pk_values) -> Optional[dict]:
    where = " AND ".join(f"{c} = %s" for c in pk_cols)
    col_list = ", ".join(columns)
    cur = pg_conn.execute(
        f"SELECT {col_list} FROM {table} WHERE {where}", pk_values)
    row = cur.fetchone()
    return dict(zip(columns, row)) if row is not None else None


def _normalize_dt(value):
    """Parse a SQLite ISO-ish TEXT timestamp OR pass through a PG-returned
    ``datetime`` unchanged, so both sides can be diffed as aware datetimes."""
    if value is None or not isinstance(value, str):
        return value
    v = value.strip().replace(" ", "T")
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        return value  # unparsable — compare as a raw string (will surface as a mismatch)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def rows_equal(table: str, sq_row: Optional[dict], pg_row: Optional[dict]) -> bool:
    if sq_row is None or pg_row is None:
        return sq_row == pg_row
    dt_cols = DATETIME_COLUMNS.get(table, set())
    for col in SOURCE_COLUMNS[table]:
        a, b = sq_row.get(col), pg_row.get(col)
        if col in dt_cols:
            a2, b2 = _normalize_dt(a), _normalize_dt(b)
            if a2 is None or b2 is None:
                if a2 != b2:
                    return False
                continue
            if not isinstance(a2, datetime) or not isinstance(b2, datetime):
                if a2 != b2:
                    return False
                continue
            if abs((a2 - b2).total_seconds()) > 1.0:
                return False
            continue
        if isinstance(a, float) or isinstance(b, float):
            if a is None or b is None:
                if a != b:
                    return False
                continue
            if abs(float(a) - float(b)) > _FLOAT_TOL:
                return False
            continue
        if a != b:
            return False
    return True


def verify_counts(sq_conn, pg_conn, table: str):
    if table == "tglc":
        sq_counts = sqlite_tglc_sector_counts(sq_conn)
        pg_rows = pg_conn.execute(
            "SELECT sector, COUNT(*) FROM tglc GROUP BY sector").fetchall()
        pg_counts = {int(r[0]): int(r[1]) for r in pg_rows}
        return sq_counts == pg_counts, {"sqlite": sq_counts, "pg": pg_counts}
    sq_n = sqlite_total_count(sq_conn, table)
    pg_n = pg_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return sq_n == pg_n, {"sqlite": sq_n, "pg": pg_n}


def verify_sample(sq_conn, pg_conn, table: str, *, batch_rows: int,
                   sample_size: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED):
    pk_cols = PK_COLUMNS[table]
    columns = SOURCE_COLUMNS[table]
    pks = reservoir_sample_pks(sq_conn, table, pk_cols, sample_size, seed, batch_rows)
    mismatches = []
    for pk in pks:
        sq_row = sqlite_field_row(sq_conn, table, columns, pk_cols, pk)
        pg_row = pg_lookup_row(pg_conn, table, columns, pk_cols, pk)
        if not rows_equal(table, sq_row, pg_row):
            mismatches.append({"pk": list(pk), "sqlite": sq_row, "pg": pg_row})
    return not mismatches, {
        "sampled": len(pks), "n_mismatches": len(mismatches),
        "mismatches": mismatches[:20],
    }


def _make_cache(target):
    """Build a ``MetadataCache`` pointed at *target* (a SQLite path or a
    ``postgresql://`` DSN), without leaking ``EXONIT_METADATA_DB_URL`` into
    the rest of this process afterward (the class reads that env var at
    construction time only — see its docstring)."""
    from exonit.metadata_cache import MetadataCache, is_pg_url

    prev = os.environ.get("EXONIT_METADATA_DB_URL")
    try:
        if is_pg_url(target):
            os.environ["EXONIT_METADATA_DB_URL"] = target
            return MetadataCache("unused-pg-wins")
        os.environ.pop("EXONIT_METADATA_DB_URL", None)
        return MetadataCache(str(target))
    finally:
        if prev is None:
            os.environ.pop("EXONIT_METADATA_DB_URL", None)
        else:
            os.environ["EXONIT_METADATA_DB_URL"] = prev


def verify_indexed_basenames(sqlite_path, pg_dsn, sectors):
    sq_cache = _make_cache(str(sqlite_path))
    pg_cache = _make_cache(pg_dsn)
    try:
        mismatches = []
        for s in sectors:
            a = sq_cache.indexed_basenames(s)
            b = pg_cache.indexed_basenames(s)
            if a != b:
                mismatches.append({
                    "sector": s, "n_sqlite": len(a), "n_pg": len(b),
                    "only_sqlite": sorted(a - b)[:5], "only_pg": sorted(b - a)[:5],
                })
        return not mismatches, {"sectors_checked": len(sectors), "mismatches": mismatches}
    finally:
        sq_cache.close()
        pg_cache.close()


def verify_tic_ids_order(sqlite_path, pg_dsn, sectors):
    sq_cache = _make_cache(str(sqlite_path))
    pg_cache = _make_cache(pg_dsn)
    try:
        mismatches = []
        for s in sectors:
            a = sq_cache.tglc_tic_ids(s)
            b = pg_cache.tglc_tic_ids(s)
            if a != b:
                first_diff = next(
                    (i for i in range(min(len(a), len(b))) if a[i] != b[i]),
                    min(len(a), len(b)))
                mismatches.append({
                    "sector": s, "len_sqlite": len(a), "len_pg": len(b),
                    "first_diff_index": first_diff,
                })
        return not mismatches, {"sectors_checked": len(sectors), "mismatches": mismatches}
    finally:
        sq_cache.close()
        pg_cache.close()


def run_verify(sqlite_path, pg_dsn: str, tables, *, batch_rows: int = DEFAULT_BATCH_ROWS) -> bool:
    sq_conn = connect_sqlite_ro(sqlite_path)
    pg_conn = connect_pg(pg_dsn)
    all_ok = True
    try:
        for t in tables:
            ok, detail = verify_counts(sq_conn, pg_conn, t)
            all_ok &= ok
            log(f"[{'PASS' if ok else 'FAIL'}] {t}: row counts "
                f"{'match' if ok else 'MISMATCH: ' + json.dumps(detail, default=str)}")

            ok2, detail2 = verify_sample(sq_conn, pg_conn, t, batch_rows=batch_rows)
            all_ok &= ok2
            log(f"[{'PASS' if ok2 else 'FAIL'}] {t}: {detail2['sampled']}-row sample "
                f"({detail2['n_mismatches']} mismatches"
                f"{': ' + json.dumps(detail2['mismatches'], default=str) if detail2['mismatches'] else ''})")

        if "tglc" in tables:
            sectors = set(sqlite_distinct_tglc_sectors(sq_conn))
            if "fits_blobs" in tables:
                sectors |= distinct_fits_blobs_sectors(sq_conn, batch_rows)
            sectors = sorted(sectors)

            ok3, detail3 = verify_indexed_basenames(sqlite_path, pg_dsn, sectors)
            all_ok &= ok3
            log(f"[{'PASS' if ok3 else 'FAIL'}] indexed_basenames union invariant "
                f"({detail3['sectors_checked']} sectors"
                f"{', mismatches: ' + json.dumps(detail3['mismatches'], default=str) if detail3['mismatches'] else ''})")

            ok4, detail4 = verify_tic_ids_order(sqlite_path, pg_dsn, sectors)
            all_ok &= ok4
            log(f"[{'PASS' if ok4 else 'FAIL'}] tglc_tic_ids order invariant "
                f"({detail4['sectors_checked']} sectors"
                f"{', mismatches: ' + json.dumps(detail4['mismatches'], default=str) if detail4['mismatches'] else ''})")
    finally:
        sq_conn.close()
        pg_conn.close()

    log(f"{'PASS' if all_ok else 'FAIL'}: overall verification "
        f"{'succeeded' if all_ok else 'FAILED'}")
    return all_ok


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sqlite", type=Path,
                   default=Path(os.environ.get("EXONIT_METADATA_CACHE", str(DEFAULT_SQLITE))),
                   help=f"Path to the metadata_cache.db SQLite file (default: {DEFAULT_SQLITE})")
    p.add_argument("--pg-dsn", default=os.environ.get("EXONIT_METADATA_DB_URL"),
                   help="Target PostgreSQL DSN (default: $EXONIT_METADATA_DB_URL). "
                        "Must NOT be the operational 'exonit' database.")
    p.add_argument("--table", default=None,
                   help=f"Comma-separated subset of tables to migrate/verify "
                        f"(default: all — {','.join(TABLES)})")
    p.add_argument("--resume", action="store_true",
                   help="Continue from the last checkpoint instead of TRUNCATE + fresh start.")
    p.add_argument("--verify-only", action="store_true",
                   help="Run only the verification gate (no writes).")
    p.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS,
                   help=f"Rows per COPY/checkpoint batch (default {DEFAULT_BATCH_ROWS:,}).")
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR,
                   help=f"Checkpoint directory (default: {DEFAULT_CHECKPOINT_DIR}).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.pg_dsn:
        print("ERROR: --pg-dsn or $EXONIT_METADATA_DB_URL required.", file=sys.stderr)
        return 2
    try:
        assert_not_operational_db(args.pg_dsn)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2

    if not args.sqlite.exists():
        print(f"ERROR: --sqlite file not found: {args.sqlite}", file=sys.stderr)
        return 2

    tables = [t.strip() for t in args.table.split(",")] if args.table else list(TABLES)
    bad = [t for t in tables if t not in TABLES]
    if bad:
        print(f"ERROR: unknown table(s) {bad} — valid: {TABLES}", file=sys.stderr)
        return 2

    if args.verify_only:
        ok = run_verify(args.sqlite, args.pg_dsn, tables, batch_rows=args.batch_rows)
        return 0 if ok else 1

    sq_conn = connect_sqlite_ro(args.sqlite)
    pg_conn = connect_pg(args.pg_dsn)
    try:
        for t in tables:
            log(f"=== {t} ===")
            migrate_table(sq_conn, pg_conn, t, resume=args.resume,
                           checkpoint_dir=args.checkpoint_dir, batch_rows=args.batch_rows)
    finally:
        sq_conn.close()
        pg_conn.close()
    log("Migration run complete. Run --verify-only to confirm before cutover.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
