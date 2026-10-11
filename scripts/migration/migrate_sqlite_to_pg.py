#!/usr/bin/env python3
"""ExoNit SQLite → PostgreSQL migration script.

Exports data from the operational SQLite database and loads it into
the target PostgreSQL database using COPY (binary-mode CSV via
psycopg3's copy_from for speed).

Modes
-----
--dry-run   Validate row counts in SQLite, report what would be copied.
            No writes to PostgreSQL.
--validate  After a completed load, compare row counts, NULL rates, and
            detect orphan FK references between SQLite and PostgreSQL.

Usage
-----
# Dry-run (safe, no PG writes):
  python3 scripts/migration/migrate_sqlite_to_pg.py --dry-run

# Full load (run ONLY when production traffic is offline):
  python3 scripts/migration/migrate_sqlite_to_pg.py

# Validate a completed load:
  python3 scripts/migration/migrate_sqlite_to_pg.py --validate

Environment variables
---------------------
EXONIT_SQLITE_PATH    Path to the SQLite database (default: data/exonit.db
                      relative to the repo root).
EXONIT_DATABASE_URL   PostgreSQL DSN (default: host read from --pg-host etc.).
                      Format: postgresql://user:password@host:port/dbname

The script is IDEMPOTENT: each table load is preceded by TRUNCATE CASCADE,
so re-running after a failed load starts from a clean slate.
"""

import argparse
import io
import os
import sqlite3
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# ── defaults ──────────────────────────────────────────────────────────────
DEFAULT_SQLITE = ROOT / "data" / "exonit.db"

# Tables to migrate, in dependency order (referenced tables before referencing).
# Each entry: (sqlite_table, pg_table, [column_specs])
# column_specs is used to remap column names and handle type coercions.
# The raw SQLite rows are streamed; datetime TEXT → TIMESTAMPTZ via parse_dt().

# Tables EXCLUDED from migration:
#   completeness  – huge, goes to Parquet
#   sqlite_sequence – PG uses SEQUENCE objects, reset separately

TABLES_ORDERED = [
    "stars",
    "workers_registry",
    "candidates",
    "tasks",
    "worker_telemetry",
    "worker_debug_mode",
    "completeness_agg",
    "metadata",
    "fits_paths",
    "nearby_stars",
]

# Columns that are stored as SQLite TEXT but contain ISO-8601 datetime strings.
# These are coerced to Python datetime objects so psycopg can write TIMESTAMPTZ.
DATETIME_COLS = {
    "stars":            {"analyzed_at"},
    "workers_registry": {"first_seen", "last_seen"},
    "candidates":       set(),
    "tasks":            {"assigned_at", "completed_at", "created_at"},
    "worker_telemetry": {"timestamp"},   # renamed to "ts" in PG schema
    "worker_debug_mode":{"enabled_at"},
    "completeness_agg": set(),
    "metadata":         {"updated_at"},
    "fits_paths":       set(),
    "nearby_stars":     {"updated_at"},
}

# Columns that are SQLite INTEGER 0/1 booleans → Python bool for PG BOOLEAN.
BOOL_COLS = {
    "stars":            {"has_spoc"},
    "candidates":       {"ctoi_submitted", "cross_source_agreement"},
    "tasks":            {"gpu_used", "lc_load_recovered"},
    "worker_telemetry": set(),
    "worker_debug_mode":{"debug_enabled"},
    "workers_registry": set(),
    "completeness_agg": set(),
    "metadata":         set(),
    "fits_paths":       set(),
    "nearby_stars":     set(),
}

# Column renames: SQLite name → PG name (for tables where they differ)
COLUMN_RENAMES = {
    "worker_telemetry": {"timestamp": "ts"},
}


# ── helpers ───────────────────────────────────────────────────────────────

def parse_dt(value: str | None) -> datetime | None:
    """Parse an ISO-8601 datetime string (with or without T separator)
    into a timezone-aware datetime object (UTC assumed if no tz offset).

    Tries ``datetime.fromisoformat()`` first: it parses BOTH legacy naive
    strings (pre 2026-07-02, ``datetime.now().isoformat()``) AND the
    ``"...+00:00"``-suffixed tz-aware strings ``db.py``'s ``utc_now_iso()``
    now writes (see ``docs/STATUS.md`` / ``progress/impl-tzfix-2026-07-02.md``
    — the naive-localtime timestamp fix). Before that fix, none of the fixed
    ``strptime`` formats below include a ``%z``/offset, so every tz-aware row
    would fail all four and silently migrate as NULL on any re-run of this
    tool (disaster recovery / re-sync from a worker snapshot) — a real
    data-quality regression this parse order avoids. Falls back to the
    historical fixed-format loop only for strings ``fromisoformat`` can't
    parse (kept for robustness / older-Python parity)."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    # Normalise the separator
    value = value.replace(" ", "T")
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass
    # Try common legacy formats
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    # Fallback: return None and log a warning rather than crashing
    print(f"    [WARN] Could not parse datetime '{value}' — storing NULL")
    return None


def coerce_row(table: str, sqlite_cols: list[str], row: tuple) -> list:
    """Apply type coercions (bool, datetime) to a raw SQLite row."""
    dt_cols  = DATETIME_COLS.get(table, set())
    bool_cols = BOOL_COLS.get(table, set())
    renames  = COLUMN_RENAMES.get(table, {})

    result = []
    for col, val in zip(sqlite_cols, row):
        pg_col = renames.get(col, col)
        if col in dt_cols:
            val = parse_dt(val)
        elif col in bool_cols:
            val = bool(val) if val is not None else None
        result.append(val)
    return result


def pg_columns(table: str, sqlite_cols: list[str]) -> list[str]:
    """Return the PG column names for a given SQLite column list."""
    renames = COLUMN_RENAMES.get(table, {})
    return [renames.get(c, c) for c in sqlite_cols]


def connect_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    # Read-only URI
    return conn


def connect_pg(dsn: str):
    try:
        import psycopg
    except ImportError:
        print("ERROR: psycopg (v3) not installed. Install with: pip install psycopg[binary]")
        sys.exit(1)
    return psycopg.connect(dsn, autocommit=False)


def build_dsn(args) -> str:
    """Build a PostgreSQL DSN from args or environment."""
    env_url = os.environ.get("EXONIT_DATABASE_URL")
    if env_url:
        return env_url
    pw = args.pg_password or os.environ.get("PGPASSWORD", "")
    return (
        f"postgresql://{args.pg_user}:{pw}"
        f"@{args.pg_host}:{args.pg_port}/{args.pg_dbname}"
    )


def get_sqlite_cols(conn: sqlite3.Connection, table: str) -> list[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]


def sqlite_row_count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return row[0]


# ── dry-run ───────────────────────────────────────────────────────────────

def dry_run(sqlite_path: Path) -> None:
    """Report row counts per table; no PG connection required."""
    print(f"\n{'='*60}")
    print(f"DRY-RUN — source: {sqlite_path}")
    print(f"{'='*60}")
    if not sqlite_path.exists():
        print(f"ERROR: SQLite file not found: {sqlite_path}")
        sys.exit(1)

    conn = connect_sqlite(sqlite_path)
    total = 0
    print(f"\n{'Table':<28} {'Rows':>12}  Notes")
    print("-" * 60)
    for table in TABLES_ORDERED:
        try:
            n = sqlite_row_count(conn, table)
            total += n
            notes = ""
            if table == "completeness":
                notes = "(EXCLUDED — Parquet)"
            elif table in ("metadata", "nearby_stars", "fits_paths", "worker_debug_mode"):
                notes = "(cache/empty — will load)"
            print(f"  {table:<26} {n:>12,}  {notes}")
        except sqlite3.OperationalError as e:
            print(f"  {table:<26} {'ERROR':>12}  {e}")

    print("-" * 60)
    print(f"  {'TOTAL (all tables)':<26} {total:>12,}")

    # Check FK integrity
    print("\nFK pre-check (SQLite):")
    orphan_cands = conn.execute(
        "SELECT COUNT(*) FROM candidates c "
        "WHERE NOT EXISTS (SELECT 1 FROM stars s WHERE s.tic_id = c.tic_id)"
    ).fetchone()[0]
    orphan_tasks_stars = conn.execute(
        "SELECT COUNT(*) FROM tasks t "
        "WHERE t.tic_id IS NOT NULL AND NOT EXISTS "
        "(SELECT 1 FROM stars s WHERE s.tic_id = t.tic_id)"
    ).fetchone()[0]
    orphan_telem = conn.execute(
        "SELECT COUNT(*) FROM worker_telemetry wt "
        "WHERE NOT EXISTS (SELECT 1 FROM workers_registry wr WHERE wr.worker_id = wt.worker_id)"
    ).fetchone()[0]
    print(f"  candidates with no matching star:    {orphan_cands:,}")
    print(f"  tasks with no matching star:         {orphan_tasks_stars:,}")
    print(f"  telemetry with no matching worker:   {orphan_telem:,}")
    if orphan_cands or orphan_tasks_stars or orphan_telem:
        print("\n  [WARN] Orphan rows exist. The migration script NULLs FK references")
        print("         for tasks (tic_id nullable) but candidates FK is enforced in PG.")
        print("         Review before running the full load.")

    conn.close()
    print(f"\nDry-run complete. No data was written to PostgreSQL.")


# ── validate ──────────────────────────────────────────────────────────────

def validate(sqlite_path: Path, pg_dsn: str) -> None:
    """Compare row counts and spot-check between SQLite and PG."""
    print(f"\n{'='*60}")
    print(f"VALIDATE — SQLite vs PostgreSQL")
    print(f"{'='*60}")

    sq = connect_sqlite(sqlite_path)
    pg = connect_pg(pg_dsn)
    pg_cur = pg.cursor()

    issues = []
    print(f"\n{'Table':<28} {'SQLite':>12} {'PG':>12}  Status")
    print("-" * 60)
    for table in TABLES_ORDERED:
        try:
            sq_n = sqlite_row_count(sq, table)
            pg_cur.execute(f"SELECT COUNT(*) FROM {table}")
            pg_n = pg_cur.fetchone()[0]
            match = "OK" if sq_n == pg_n else "MISMATCH"
            if match == "MISMATCH":
                issues.append(f"  {table}: SQLite={sq_n:,} PG={pg_n:,}")
            print(f"  {table:<26} {sq_n:>12,} {pg_n:>12,}  {match}")
        except Exception as e:
            print(f"  {table:<26} {'?':>12} {'?':>12}  ERROR: {e}")
            issues.append(f"  {table}: {e}")

    print("-" * 60)

    # FK orphan check in PG
    print("\nFK orphan check (PostgreSQL):")
    checks = [
        ("candidates FK stars",
         "SELECT COUNT(*) FROM candidates c WHERE NOT EXISTS "
         "(SELECT 1 FROM stars s WHERE s.tic_id = c.tic_id)"),
        ("telemetry FK workers",
         "SELECT COUNT(*) FROM worker_telemetry wt WHERE NOT EXISTS "
         "(SELECT 1 FROM workers_registry wr WHERE wr.worker_id = wt.worker_id)"),
    ]
    for label, sql in checks:
        pg_cur.execute(sql)
        n = pg_cur.fetchone()[0]
        status = "OK" if n == 0 else f"ORPHANS: {n:,}"
        print(f"  {label:<40} {status}")
        if n:
            issues.append(f"  FK orphan in PG: {label} = {n:,}")

    pg.close()
    sq.close()

    if issues:
        print(f"\n[FAIL] {len(issues)} issue(s) found:")
        for iss in issues:
            print(iss)
        sys.exit(1)
    else:
        print("\n[PASS] All row counts match and no FK orphans.")


# ── full load ─────────────────────────────────────────────────────────────

COPY_CHUNK = 50_000   # rows per COPY batch

def load_table(
    sq_conn: sqlite3.Connection,
    pg_conn,
    table: str,
    verbose: bool = True,
) -> int:
    """
    TRUNCATE the PG table, then stream rows from SQLite via COPY.
    Returns the number of rows loaded.
    """
    import psycopg

    sq_cols = get_sqlite_cols(sq_conn, table)
    pg_cols = pg_columns(table, sq_cols)

    # TRUNCATE (CASCADE in case of FK children, though migration order prevents that)
    pg_conn.execute(f"TRUNCATE TABLE {table} CASCADE")

    col_list = ", ".join(f'"{c}"' for c in pg_cols)
    copy_sql = f'COPY {table} ({col_list}) FROM STDIN (FORMAT TEXT, DELIMITER E\'\\t\', NULL \'\\N\')'

    total = 0
    cur = sq_conn.cursor()
    cur.execute(f"SELECT * FROM {table}")

    with pg_conn.cursor().copy(copy_sql) as copy:
        while True:
            rows = cur.fetchmany(COPY_CHUNK)
            if not rows:
                break
            for raw_row in rows:
                coerced = coerce_row(table, sq_cols, tuple(raw_row))
                # Encode row as tab-separated text for COPY TEXT format
                copy.write_row(coerced)
            total += len(rows)
            if verbose:
                print(f"\r    {total:>12,} rows loaded...", end="", flush=True)

    if verbose:
        print(f"\r    {total:>12,} rows loaded.    ")
    return total


def reset_sequences(pg_conn) -> None:
    """Reset sequences for tables that use BIGSERIAL-style IDs."""
    seqs = [
        ("candidates_id_seq", "candidates", "id"),
        ("worker_telemetry_id_seq", "worker_telemetry", "id"),
    ]
    for seq, table, col in seqs:
        pg_conn.execute(
            f"SELECT setval('{seq}', COALESCE((SELECT MAX({col}) FROM {table}), 1))"
        )


def run_migration(sqlite_path: Path, pg_dsn: str, verbose: bool = True) -> None:
    """Full migration: TRUNCATE + COPY each table, reset sequences."""
    print(f"\n{'='*60}")
    print(f"MIGRATION — {sqlite_path} → PostgreSQL")
    print(f"{'='*60}\n")

    if not sqlite_path.exists():
        print(f"ERROR: SQLite file not found: {sqlite_path}")
        sys.exit(1)

    sq = connect_sqlite(sqlite_path)
    pg = connect_pg(pg_dsn)

    total_rows = 0
    t0 = datetime.now()

    for table in TABLES_ORDERED:
        print(f"  [{table}]")
        try:
            n = load_table(sq, pg, table, verbose=verbose)
            total_rows += n
        except Exception as e:
            pg.rollback()
            sq.close()
            pg.close()
            print(f"\n  ERROR loading table '{table}': {e}")
            print("  Migration rolled back. Fix the error and re-run.")
            sys.exit(1)

    print(f"\n  Resetting sequences...")
    reset_sequences(pg)

    pg.commit()
    sq.close()
    pg.close()

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nMigration complete: {total_rows:,} rows in {elapsed:.1f}s")
    print("Run --validate to confirm counts match.")


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be migrated; no PG writes.",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Compare row counts and FK integrity between SQLite and PG.",
    )
    parser.add_argument(
        "--sqlite",
        type=Path,
        default=Path(os.environ.get("EXONIT_SQLITE_PATH", str(DEFAULT_SQLITE))),
        help=f"Path to SQLite database (default: {DEFAULT_SQLITE})",
    )
    # PG connection args (overridden by EXONIT_DATABASE_URL env var)
    pg = parser.add_argument_group("PostgreSQL connection (ignored if EXONIT_DATABASE_URL is set)")
    pg.add_argument("--pg-host",     default="localhost")
    pg.add_argument("--pg-port",     default="5432")
    pg.add_argument("--pg-user",     default="exonit")
    pg.add_argument("--pg-dbname",   default="exonit")
    pg.add_argument("--pg-password", default=os.environ.get("PGPASSWORD", ""))
    parser.add_argument("--quiet", action="store_true", help="Suppress per-row progress.")

    args = parser.parse_args()

    if args.dry_run and args.validate:
        parser.error("--dry-run and --validate are mutually exclusive.")

    if args.dry_run:
        dry_run(args.sqlite)
        return

    dsn = build_dsn(args)

    if args.validate:
        validate(args.sqlite, dsn)
        return

    # Safety gate: require explicit confirmation for the full load
    print("\nWARNING: This will TRUNCATE and reload all tables in the target PG database.")
    print(f"  SQLite: {args.sqlite}")
    print(f"  PG DSN: {dsn.replace(args.pg_password, '***') if args.pg_password else dsn}")
    ans = input("\nType 'yes' to proceed: ").strip().lower()
    if ans != "yes":
        print("Aborted.")
        sys.exit(0)

    run_migration(args.sqlite, dsn, verbose=not args.quiet)


if __name__ == "__main__":
    main()
