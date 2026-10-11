"""Tests for scripts/migration/migrate_metadata_to_pg.py (F2 of
docs/superpowers/specs/2026-07-20-tglc-index-to-pg.md).

Scope:
  - Unit tests (no PG needed): rowid->seq ordering math, reservoir sampling
    reproducibility, row-comparison tolerance, checkpoint save/load,
    fits_blobs sector parsing, the operational-DB guard, CLI arg validation.
  - PG-touching integration tests: opt-in via EXONIT_TEST_META_PG_URL (same
    resolver + "database name must contain 'test'" hard guard as
    tests/test_metadata_backend_pg_parity.py — see tests/_metadata_pg_test_db.py).
    Exercise the real partitioned schema_pg_meta.sql schema: batching +
    checkpointing, --resume continuing (not truncating) from a checkpoint,
    tglc seq/rowid ordering end to end, index drop-before/recreate-after,
    and the verification gate catching an injected discrepancy.

No test ever opens data/metadata_cache.db (the live index) or writes to the
operational 'exonit' PG database.
"""

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from tests._metadata_pg_test_db import pg_meta_test_available, resolve_test_meta_dsn

ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts" / "migration" / "migrate_metadata_to_pg.py"
SCHEMA_SQL_PATH = ROOT / "scripts" / "migration" / "schema_pg_meta.sql"

spec = importlib.util.spec_from_file_location("migrate_metadata_to_pg", MODULE_PATH)
mig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mig)

PG_DSN = resolve_test_meta_dsn()
pg_available = pg_meta_test_available(PG_DSN)


# --------------------------------------------------------------------------- #
# Fixtures: a small SQLite metadata_cache.db lookalike
# --------------------------------------------------------------------------- #

def _make_sqlite(tmp_path) -> Path:
    """A tiny SQLite DB with all 5 tables, populated in a KNOWN, deliberately
    cross-sector-interleaved rowid order for tglc (to make the "global order,
    not per-sector-grouped order" design decision testable)."""
    db_path = tmp_path / "metadata_cache.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE tglc (
            tic_id TEXT, sector INTEGER, path TEXT,
            exptime REAL, tmag REAL, ra REAL, dec REAL
        )
    """)
    conn.execute("""
        CREATE TABLE fits_blobs (
            filename TEXT PRIMARY KEY, container_key TEXT,
            data_offset INTEGER, data_length INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE metadata (
            tic_id TEXT PRIMARY KEY, sectors TEXT, exptime REAL, fits_files TEXT,
            stellar_radius REAL, teff REAL, logg REAL, tmag REAL, ra REAL, dec REAL,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE fits_paths (filename TEXT PRIMARY KEY, full_path TEXT)
    """)
    conn.execute("""
        CREATE TABLE nearby_stars (
            tic_id TEXT, neighbor_id TEXT, ra REAL, dec REAL, mass REAL,
            rad REAL, teff REAL, tmag REAL, plx REAL, gal_b REAL, updated_at TEXT
        )
    """)

    # tglc rows: rowid 1..9, sectors interleaved 1,2,1,2,1,2,1,2,1 — a strict
    # per-sector grouping strategy would NOT reproduce this global order.
    tglc_rows = [
        ("100", 1, "/a/100.fits", 1800.0, 10.0, 1.0, 2.0),
        ("200", 2, "/a/200.fits", 1800.0, 11.0, 3.0, 4.0),
        ("101", 1, "/a/101.fits", 1800.0, 12.0, 5.0, 6.0),
        ("201", 2, "/a/201.fits", 1800.0, 13.0, 7.0, 8.0),
        ("102", 1, "/a/102.fits", 1800.0, 14.0, 9.0, 10.0),
        ("202", 2, "/a/202.fits", 1800.0, 15.0, 11.0, 12.0),
        ("103", 1, "/a/103.fits", 1800.0, 16.0, 13.0, 14.0),
        ("203", 2, "/a/203.fits", 1800.0, 17.0, 15.0, 16.0),
        ("104", 1, "/a/104.fits", 1800.0, 18.0, 17.0, 18.0),
    ]
    conn.executemany(
        "INSERT INTO tglc (tic_id, sector, path, exptime, tmag, ra, dec) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)", tglc_rows)

    conn.executemany(
        "INSERT INTO fits_blobs VALUES (?, ?, ?, ?)",
        [
            ("hlsp_tglc_tess_ffi_gaiaid-1-s0001-cam1-ccd1_tess_v1_llc.fits",
             "containers/s0001/dl_1.tar", 0, 1000),
            ("hlsp_tglc_tess_ffi_gaiaid-2-s0002-cam1-ccd1_tess_v1_llc.fits",
             "containers/s0002/dl_1.tar", 1000, 2000),
        ],
    )

    conn.executemany(
        "INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (f"{i}", json.dumps([1]), 1800.0, json.dumps([f"{i}.fits"]),
             1.0, 5000.0, 4.5, 10.0, float(i), float(i) + 1,
             "2026-07-14 10:00:00")
            for i in range(100, 112)
        ],
    )

    conn.executemany(
        "INSERT INTO fits_paths VALUES (?, ?)",
        [(f"{i}.fits", f"/cache/{i}.fits") for i in range(100, 112)],
    )

    conn.executemany(
        "INSERT INTO nearby_stars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("100", f"n{i}", 1.0, 2.0, 0.9, 1.1, 5000.0, 12.0, 3.3, 5.0,
             "2026-07-14 10:00:00")
            for i in range(5)
        ],
    )
    conn.commit()
    conn.close()
    return db_path


# --------------------------------------------------------------------------- #
# Unit tests — no PG required
# --------------------------------------------------------------------------- #

class TestReservoirSampling:
    def test_sample_size_capped_at_row_count(self, tmp_path):
        db = _make_sqlite(tmp_path)
        conn = mig.connect_sqlite_ro(db)
        pks = mig.reservoir_sample_pks(conn, "tglc", ["tic_id", "sector"],
                                        sample_size=100, seed=1, batch_rows=3)
        assert len(pks) == 9  # only 9 tglc rows exist
        conn.close()

    def test_reproducible_with_fixed_seed(self, tmp_path):
        db = _make_sqlite(tmp_path)
        conn = mig.connect_sqlite_ro(db)
        a = mig.reservoir_sample_pks(conn, "nearby_stars", ["tic_id", "neighbor_id"],
                                      sample_size=3, seed=42, batch_rows=2)
        b = mig.reservoir_sample_pks(conn, "nearby_stars", ["tic_id", "neighbor_id"],
                                      sample_size=3, seed=42, batch_rows=2)
        assert a == b
        conn.close()

    def test_batch_boundary_does_not_change_result(self, tmp_path):
        """Same seed, different batch_rows -> identical sample (Algorithm R's
        accept/reject only depends on the running count, not chunk size)."""
        db = _make_sqlite(tmp_path)
        conn = mig.connect_sqlite_ro(db)
        a = mig.reservoir_sample_pks(conn, "metadata", ["tic_id"],
                                      sample_size=4, seed=7, batch_rows=1)
        b = mig.reservoir_sample_pks(conn, "metadata", ["tic_id"],
                                      sample_size=4, seed=7, batch_rows=100)
        assert a == b
        conn.close()


class TestDistinctFitsBlobsSectors:
    def test_parses_sector_from_filename_pattern(self, tmp_path):
        db = _make_sqlite(tmp_path)
        conn = mig.connect_sqlite_ro(db)
        sectors = mig.distinct_fits_blobs_sectors(conn, batch_rows=1)
        assert sectors == {1, 2}
        conn.close()


class TestRowsEqual:
    def test_float_within_tolerance_matches(self):
        assert mig.rows_equal(
            "tglc",
            {"tic_id": "1", "sector": 1, "path": "/a", "exptime": 1800.0,
             "tmag": 10.0, "ra": 1.0, "dec": 2.0},
            {"tic_id": "1", "sector": 1, "path": "/a", "exptime": 1800.0,
             "tmag": 10.0 + 1e-12, "ra": 1.0, "dec": 2.0},
        )

    def test_float_outside_tolerance_mismatches(self):
        assert not mig.rows_equal(
            "tglc",
            {"tic_id": "1", "sector": 1, "path": "/a", "exptime": 1800.0,
             "tmag": 10.0, "ra": 1.0, "dec": 2.0},
            {"tic_id": "1", "sector": 1, "path": "/a", "exptime": 1800.0,
             "tmag": 10.5, "ra": 1.0, "dec": 2.0},
        )

    def test_text_mismatch_detected(self):
        assert not mig.rows_equal(
            "fits_paths",
            {"filename": "a.fits", "full_path": "/x/a.fits"},
            {"filename": "a.fits", "full_path": "/y/a.fits"},
        )

    def test_datetime_naive_string_vs_pg_datetime_match(self):
        from datetime import datetime, timezone
        sq = {"tic_id": "1", "sectors": "[1]", "exptime": 1.0, "fits_files": "[]",
              "stellar_radius": None, "teff": None, "logg": None, "tmag": None,
              "ra": None, "dec": None, "updated_at": "2026-07-14 10:00:00"}
        pg = dict(sq, updated_at=datetime(2026, 7, 14, 10, 0, 0, tzinfo=timezone.utc))
        assert mig.rows_equal("metadata", sq, pg)

    def test_both_none_matches(self):
        assert mig.rows_equal("fits_paths", None, None)

    def test_one_none_mismatches(self):
        assert not mig.rows_equal(
            "fits_paths", None, {"filename": "a", "full_path": "/a"})


class TestCheckpoint:
    def test_roundtrip(self, tmp_path):
        state = {"table": "tglc", "status": "in_progress", "last_rowid": 42,
                 "rows_copied": 42, "seq_next": 43, "elapsed_s": 1.5,
                 "updated_at": "2026-07-20T00:00:00+00:00"}
        mig.save_checkpoint(tmp_path, "tglc", state)
        loaded = mig.load_checkpoint(tmp_path, "tglc")
        assert loaded == state

    def test_load_missing_returns_none(self, tmp_path):
        assert mig.load_checkpoint(tmp_path, "nope") is None

    def test_clear_removes_file(self, tmp_path):
        mig.save_checkpoint(tmp_path, "t", {"status": "in_progress"})
        mig.clear_checkpoint(tmp_path, "t")
        assert mig.load_checkpoint(tmp_path, "t") is None

    def test_save_is_atomic_no_partial_file_left(self, tmp_path):
        mig.save_checkpoint(tmp_path, "t", {"a": 1})
        assert not (tmp_path / "t.tmp").exists()
        assert (tmp_path / "t.json").exists()


class TestOperationalDbGuard:
    def test_refuses_exonit(self):
        with pytest.raises(SystemExit):
            mig.assert_not_operational_db("postgresql://exonit@host/exonit")

    def test_allows_exonit_meta(self):
        mig.assert_not_operational_db("postgresql://exonit@host/exonit_meta")  # no raise

    def test_allows_exonit_meta_test(self):
        mig.assert_not_operational_db("postgresql://exonit@host/exonit_meta_test")  # no raise


class TestCliValidation:
    def test_unknown_table_rejected(self, tmp_path, capsys):
        db = _make_sqlite(tmp_path)
        rc = mig.main([
            "--sqlite", str(db), "--pg-dsn", "postgresql://exonit@host/exonit_meta",
            "--table", "not_a_real_table",
        ])
        assert rc == 2
        assert "unknown table" in capsys.readouterr().err

    def test_missing_pg_dsn_rejected(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("EXONIT_METADATA_DB_URL", raising=False)
        db = _make_sqlite(tmp_path)
        rc = mig.main(["--sqlite", str(db)])
        assert rc == 2

    def test_operational_db_rejected_at_cli_level(self, tmp_path, capsys):
        db = _make_sqlite(tmp_path)
        rc = mig.main([
            "--sqlite", str(db), "--pg-dsn", "postgresql://exonit@host/exonit",
        ])
        assert rc == 2
        assert "REFUSING" in capsys.readouterr().err

    def test_missing_sqlite_file_rejected(self, tmp_path, capsys):
        rc = mig.main([
            "--sqlite", str(tmp_path / "nope.db"),
            "--pg-dsn", "postgresql://exonit@host/exonit_meta",
        ])
        assert rc == 2
        assert "not found" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# PG-touching integration tests — opt-in (EXONIT_TEST_META_PG_URL)
# --------------------------------------------------------------------------- #

pytestmark_pg = pytest.mark.skipif(
    not pg_available, reason="EXONIT_TEST_META_PG_URL (throwaway PG) unset or not reachable")


def _reset_pg_schema(dsn: str) -> None:
    """DROP + reapply the REAL partitioned schema_pg_meta.sql (not a
    simplified stand-in) — this module specifically needs to exercise
    partitioning + index drop/recreate, unlike test_metadata_backend_pg_parity.py's
    plain-table fixture. Also drops ``_migration_checkpoint`` (created
    out-of-band by the migrator itself, not part of schema_pg_meta.sql) so
    each test starts with a fully clean checkpoint table too — otherwise a
    checkpoint row planted/left by one test could leak into the next."""
    import psycopg

    with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
        for t in ("tglc", "fits_blobs", "metadata", "fits_paths", "nearby_stars"):
            c.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
        c.execute("DROP TABLE IF EXISTS _migration_checkpoint CASCADE")
        c.execute("DROP SEQUENCE IF EXISTS tglc_seq_seq")
        c.execute(SCHEMA_SQL_PATH.read_text())


@pytest.fixture
def pg_dsn():
    if not pg_available:
        pytest.skip("EXONIT_TEST_META_PG_URL unset or unreachable")
    _reset_pg_schema(PG_DSN)
    yield PG_DSN
    _reset_pg_schema(PG_DSN)


@pytestmark_pg
class TestMigrateTableAgainstRealPg:
    def test_fits_paths_fresh_migration_matches_source(self, tmp_path, pg_dsn):
        db = _make_sqlite(tmp_path)
        sq = mig.connect_sqlite_ro(db)
        pg = mig.connect_pg(pg_dsn)
        try:
            n = mig.migrate_table(sq, pg, "fits_paths", resume=False,
                                   checkpoint_dir=tmp_path / "ckpt", batch_rows=3)
            assert n == 12
            pg_n = pg.execute("SELECT COUNT(*) FROM fits_paths").fetchone()[0]
            assert pg_n == 12
            row = pg.execute(
                "SELECT full_path FROM fits_paths WHERE filename = %s",
                ("100.fits",)).fetchone()
            assert row[0] == "/cache/100.fits"
        finally:
            sq.close()
            pg.close()

    def test_checkpoint_marked_completed_after_full_migration(self, tmp_path, pg_dsn):
        db = _make_sqlite(tmp_path)
        sq = mig.connect_sqlite_ro(db)
        pg = mig.connect_pg(pg_dsn)
        ckpt_dir = tmp_path / "ckpt"
        try:
            mig.migrate_table(sq, pg, "fits_paths", resume=False,
                               checkpoint_dir=ckpt_dir, batch_rows=4)
        finally:
            sq.close()
            pg.close()
        ckpt = mig.load_checkpoint(ckpt_dir, "fits_paths")
        assert ckpt["status"] == "completed"
        assert ckpt["rows_copied"] == 12

    def test_resume_continues_without_truncating(self, tmp_path, pg_dsn):
        """Manually plant an in-progress checkpoint IN PG (the authoritative
        source since the review fix — planting it in the local JSON mirror
        only, as this test used to do, would no longer influence
        --resume's decision at all), atomically with a sentinel row that a
        TRUNCATE would destroy; --resume must pick up from the PG
        checkpoint and leave the sentinel (and every row after it) intact."""
        db = _make_sqlite(tmp_path)
        sq = mig.connect_sqlite_ro(db)
        pg = mig.connect_pg(pg_dsn)
        ckpt_dir = tmp_path / "ckpt"
        try:
            mig.ensure_checkpoint_table(pg)
            # rows with rowid 1..6 already "migrated" (fits_paths has 12 rows)
            first_six = sq.execute(
                "SELECT rowid, filename, full_path FROM fits_paths "
                "ORDER BY rowid ASC LIMIT 6").fetchall()
            with pg.cursor().copy(
                "COPY fits_paths (filename, full_path) FROM STDIN "
                "(FORMAT TEXT, DELIMITER E'\\t', NULL '\\N')"
            ) as copy:
                for r in first_six:
                    copy.write_row((r["filename"], r["full_path"]))
            last_rowid = first_six[-1]["rowid"]
            # Planted atomically with the sentinel COPY above — same
            # transaction, one commit — mirroring exactly how migrate_table
            # itself keeps data and checkpoint in lockstep.
            mig.pg_write_checkpoint(pg, "fits_paths", status="in_progress",
                                     last_rowid=last_rowid, rows_copied=6,
                                     seq_next=None, elapsed_s=0.0)
            pg.commit()

            n = mig.migrate_table(sq, pg, "fits_paths", resume=True,
                                   checkpoint_dir=ckpt_dir, batch_rows=2)
            assert n == 12  # 6 pre-planted + 6 more from resume
            pg_n = pg.execute("SELECT COUNT(*) FROM fits_paths").fetchone()[0]
            assert pg_n == 12
            # the pre-planted sentinel row must still be there (no truncate happened)
            row = pg.execute(
                "SELECT full_path FROM fits_paths WHERE filename = %s",
                (first_six[0]["filename"],)).fetchone()
            assert row is not None
        finally:
            sq.close()
            pg.close()

    def test_resume_ignores_local_json_and_trusts_pg_even_if_local_missing(
            self, tmp_path, pg_dsn):
        """The local JSON mirror is informational only (see design decision
        1 / Review fixes): --resume must work correctly purely from PG's
        _migration_checkpoint even when the local checkpoint DIRECTORY
        doesn't exist at all."""
        db = _make_sqlite(tmp_path)
        sq = mig.connect_sqlite_ro(db)
        pg = mig.connect_pg(pg_dsn)
        ckpt_dir = tmp_path / "ckpt-never-created"
        try:
            mig.ensure_checkpoint_table(pg)
            first_six = sq.execute(
                "SELECT rowid, filename, full_path FROM fits_paths "
                "ORDER BY rowid ASC LIMIT 6").fetchall()
            with pg.cursor().copy(
                "COPY fits_paths (filename, full_path) FROM STDIN "
                "(FORMAT TEXT, DELIMITER E'\\t', NULL '\\N')"
            ) as copy:
                for r in first_six:
                    copy.write_row((r["filename"], r["full_path"]))
            mig.pg_write_checkpoint(pg, "fits_paths", status="in_progress",
                                     last_rowid=first_six[-1]["rowid"],
                                     rows_copied=6, seq_next=None, elapsed_s=0.0)
            pg.commit()

            assert not ckpt_dir.exists()
            n = mig.migrate_table(sq, pg, "fits_paths", resume=True,
                                   checkpoint_dir=ckpt_dir, batch_rows=2)
            assert n == 12
            pg_n = pg.execute("SELECT COUNT(*) FROM fits_paths").fetchone()[0]
            assert pg_n == 12
        finally:
            sq.close()
            pg.close()

    def test_fresh_run_truncates_ignoring_stale_checkpoint(self, tmp_path, pg_dsn):
        """Without --resume, a stale 'completed' checkpoint from an older
        (smaller) source — planted in BOTH the PG-authoritative table and
        the local JSON mirror — is ignored: the table is truncated and
        reloaded from the CURRENT source, not skipped."""
        db = _make_sqlite(tmp_path)
        sq = mig.connect_sqlite_ro(db)
        pg = mig.connect_pg(pg_dsn)
        ckpt_dir = tmp_path / "ckpt"
        try:
            mig.ensure_checkpoint_table(pg)
            mig.pg_write_checkpoint(pg, "fits_paths", status="completed",
                                     last_rowid=999, rows_copied=999,
                                     seq_next=None, elapsed_s=0.0)
            pg.commit()
            mig.save_checkpoint(ckpt_dir, "fits_paths", {
                "table": "fits_paths", "status": "completed",
                "last_rowid": 999, "rows_copied": 999, "seq_next": None,
                "elapsed_s": 0.0, "updated_at": "2026-07-01T00:00:00+00:00",
            })
            n = mig.migrate_table(sq, pg, "fits_paths", resume=False,
                                   checkpoint_dir=ckpt_dir, batch_rows=5)
            assert n == 12
            ckpt = mig.load_checkpoint(ckpt_dir, "fits_paths")
            assert ckpt["rows_copied"] == 12
            pg_ckpt = mig.pg_load_checkpoint(pg, "fits_paths")
            assert pg_ckpt["rows_copied"] == 12
        finally:
            sq.close()
            pg.close()

    def test_tglc_seq_matches_global_rowid_order_per_sector(self, tmp_path, pg_dsn):
        db = _make_sqlite(tmp_path)
        sq = mig.connect_sqlite_ro(db)
        pg = mig.connect_pg(pg_dsn)
        try:
            mig.migrate_table(sq, pg, "tglc", resume=False,
                               checkpoint_dir=tmp_path / "ckpt", batch_rows=2)

            for sector in (1, 2):
                sq_order = [
                    r[0] for r in sq.execute(
                        "SELECT tic_id FROM tglc WHERE sector = ? ORDER BY rowid ASC",
                        (sector,)).fetchall()
                ]
                pg_order = [
                    r[0] for r in pg.execute(
                        "SELECT tic_id FROM tglc WHERE sector = %s ORDER BY seq ASC",
                        (sector,)).fetchall()
                ]
                assert sq_order == pg_order

            # global order too (not just per-sector) — the whole point of
            # streaming a single rowid-ordered pass instead of grouping by
            # sector (design decision 2 in the module docstring).
            sq_global = [r[0] for r in sq.execute(
                "SELECT tic_id FROM tglc ORDER BY rowid ASC").fetchall()]
            pg_global = [r[0] for r in pg.execute(
                "SELECT tic_id FROM tglc ORDER BY seq ASC").fetchall()]
            assert sq_global == pg_global
        finally:
            sq.close()
            pg.close()

    def test_tglc_partitions_created_for_every_source_sector(self, tmp_path, pg_dsn):
        db = _make_sqlite(tmp_path)
        sq = mig.connect_sqlite_ro(db)
        pg = mig.connect_pg(pg_dsn)
        try:
            mig.migrate_table(sq, pg, "tglc", resume=False,
                               checkpoint_dir=tmp_path / "ckpt", batch_rows=4)
            tables = {r[0] for r in pg.execute(
                "SELECT tablename FROM pg_tables WHERE tablename LIKE 'tglc_s%'"
            ).fetchall()}
            assert "tglc_s0001" in tables
            assert "tglc_s0002" in tables
        finally:
            sq.close()
            pg.close()

    def test_tglc_secondary_indexes_dropped_then_recreated(self, tmp_path, pg_dsn):
        db = _make_sqlite(tmp_path)
        sq = mig.connect_sqlite_ro(db)
        pg = mig.connect_pg(pg_dsn)
        try:
            mig.migrate_table(sq, pg, "tglc", resume=False,
                               checkpoint_dir=tmp_path / "ckpt", batch_rows=4)
            names = {r[0] for r in pg.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'tglc'"
            ).fetchall()}
            assert {"ix_tglc_tic", "ix_tglc_sector", "ix_tglc_path"} <= names
        finally:
            sq.close()
            pg.close()

    def test_tglc_sequence_bumped_past_migrated_max(self, tmp_path, pg_dsn):
        db = _make_sqlite(tmp_path)
        sq = mig.connect_sqlite_ro(db)
        pg = mig.connect_pg(pg_dsn)
        try:
            mig.migrate_table(sq, pg, "tglc", resume=False,
                               checkpoint_dir=tmp_path / "ckpt", batch_rows=4)
            max_seq = pg.execute("SELECT MAX(seq) FROM tglc").fetchone()[0]
            next_val = pg.execute("SELECT nextval('tglc_seq_seq')").fetchone()[0]
            pg.rollback()
            assert next_val > max_seq
        finally:
            sq.close()
            pg.close()

    def test_nearby_stars_and_metadata_datetime_survive_roundtrip(self, tmp_path, pg_dsn):
        db = _make_sqlite(tmp_path)
        sq = mig.connect_sqlite_ro(db)
        pg = mig.connect_pg(pg_dsn)
        try:
            mig.migrate_table(sq, pg, "metadata", resume=False,
                               checkpoint_dir=tmp_path / "ckpt", batch_rows=5)
            mig.migrate_table(sq, pg, "nearby_stars", resume=False,
                               checkpoint_dir=tmp_path / "ckpt", batch_rows=2)
            row = pg.execute(
                "SELECT updated_at FROM metadata WHERE tic_id = %s", ("100",)
            ).fetchone()
            assert row[0].year == 2026 and row[0].month == 7 and row[0].day == 14
            n = pg.execute("SELECT COUNT(*) FROM nearby_stars").fetchone()[0]
            assert n == 5
        finally:
            sq.close()
            pg.close()


@pytestmark_pg
class TestEnsureTglcPartitionsConcurrency:
    """2026-07-20 review fix (progress/f3-indexer-pg-review.md, point 5,
    BLOCKER — fixed "for consistency" in this script too, per the review's
    explicit ask). `ensure_tglc_partitions` shares ONE long-lived `pg_conn`
    across its whole per-sector loop (unlike `MetadataCache.
    ensure_tglc_partition`, which borrows a fresh pooled connection per
    call) — a lost DuplicateTable/UniqueViolation race must not poison the
    REST of that shared transaction (every subsequent statement, including
    the next sector's CREATE, would otherwise fail with "current transaction
    is aborted")."""

    def test_repeated_calls_for_the_same_sector_stay_idempotent(self, pg_dsn):
        """Regression guard for the savepoint fix: calling
        ensure_tglc_partitions twice for the SAME sector, on the SAME
        long-lived connection, must not raise and must not poison the
        connection's transaction for whatever runs after it."""
        pg = mig.connect_pg(pg_dsn)
        try:
            mig.ensure_tglc_partitions(pg, [777])
            mig.ensure_tglc_partitions(pg, [777])  # must be a no-op, not a crash
            pg.commit()
            # The connection's transaction must still be usable afterward —
            # this is exactly what the savepoint fix protects: an uncaught
            # DuplicateTable on a SHARED pg_conn (no savepoint) would have
            # left the transaction aborted, and this INSERT would fail with
            # "current transaction is aborted, commands ignored until end of
            # transaction block" instead of succeeding.
            pg.execute(
                "INSERT INTO tglc (tic_id, sector, path) VALUES (%s, %s, %s)",
                ("1", 777, "/x.fits"))
            pg.commit()
            row = pg.execute(
                "SELECT tic_id FROM tglc WHERE sector = %s", (777,)).fetchone()
            assert row[0] == "1"
        finally:
            pg.close()

    def test_concurrent_connections_racing_the_same_sector_both_succeed(self, pg_dsn):
        """Two SEPARATE connections (mirrors two processes/threads, e.g. two
        overlapping migration attempts or a migration run alongside F3's
        indexer both touching the same brand-new sector for the first time)
        racing ensure_tglc_partitions for the SAME sector must both
        complete without raising."""
        import threading

        sector = 778
        errors = []
        errors_lock = threading.Lock()

        def worker(i):
            pg = mig.connect_pg(pg_dsn)
            try:
                mig.ensure_tglc_partitions(pg, [sector])
                pg.commit()
            except Exception as exc:  # noqa: BLE001 - captured for the assertion below
                with errors_lock:
                    errors.append((i, type(exc).__name__, str(exc)))
            finally:
                pg.close()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], (
            f"ensure_tglc_partitions raised under concurrency: {errors}"
        )

        pg = mig.connect_pg(pg_dsn)
        try:
            pg.execute(
                "INSERT INTO tglc (tic_id, sector, path) VALUES (%s, %s, %s)",
                ("2", sector, "/y.fits"))
            pg.commit()
            row = pg.execute(
                "SELECT tic_id FROM tglc WHERE sector = %s", (sector,)).fetchone()
            assert row[0] == "2"
        finally:
            pg.close()


@pytestmark_pg
class TestCrashBetweenPgCommitAndLocalMirror:
    """Reproduces exactly the scenario from
    progress/f2-migration-tool-review.md's blocker: a crash in the window
    between a batch's PG commit (data + checkpoint, atomic — see design
    decision 1) and the local JSON mirror write. Before the fix this
    crashed with an uncaught ``UniqueViolation`` on --resume (the checkpoint
    trusted was the LOCAL file, which lagged PG's true state) and a second
    --resume attempt failed identically (non-self-healing). After the fix
    the local write is purely informational, so this crash window no longer
    affects --resume at all."""

    def test_resume_after_crash_completes_without_unique_violation_and_verify_passes(
            self, tmp_path, pg_dsn, monkeypatch):
        db = _make_sqlite(tmp_path)
        sq = mig.connect_sqlite_ro(db)
        pg = mig.connect_pg(pg_dsn)
        ckpt_dir = tmp_path / "ckpt"

        call_count = {"n": 0}
        real_save_checkpoint = mig.save_checkpoint

        def flaky_local_write(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                # Simulate the process dying right here: batch 2's PG COPY +
                # checkpoint UPSERT have ALREADY committed (that happened
                # before this function is even called — see migrate_table)
                # but the local JSON mirror never gets written for it.
                raise RuntimeError(
                    "simulated crash between PG commit and local checkpoint write")
            return real_save_checkpoint(*args, **kwargs)

        monkeypatch.setattr(mig, "save_checkpoint", flaky_local_write)

        with pytest.raises(RuntimeError, match="simulated crash"):
            mig.migrate_table(sq, pg, "fits_paths", resume=False,
                               checkpoint_dir=ckpt_dir, batch_rows=2)

        # PG already has batches 1+2 (4 rows) committed atomically with their
        # checkpoint — verify that directly, independent of the local file.
        pg_ckpt = mig.pg_load_checkpoint(pg, "fits_paths")
        assert pg_ckpt is not None
        assert pg_ckpt["status"] == "in_progress"
        assert pg_ckpt["rows_copied"] == 4
        pg_n = pg.execute("SELECT COUNT(*) FROM fits_paths").fetchone()[0]
        assert pg_n == 4  # matches the checkpoint exactly — no drift

        sq.close()
        pg.close()

        # The "natural operator reaction": --resume. Must NOT raise
        # UniqueViolation, must NOT need a second attempt.
        sq2 = mig.connect_sqlite_ro(db)
        pg2 = mig.connect_pg(pg_dsn)
        n = mig.migrate_table(sq2, pg2, "fits_paths", resume=True,
                               checkpoint_dir=ckpt_dir, batch_rows=2)
        assert n == 12
        pg_n2 = pg2.execute("SELECT COUNT(*) FROM fits_paths").fetchone()[0]
        assert pg_n2 == 12
        sq2.close()
        pg2.close()

        ok = mig.run_verify(db, pg_dsn, ["fits_paths"], batch_rows=4)
        assert ok is True

    def test_crash_on_the_very_last_batch_before_completed_marker_is_also_safe(
            self, tmp_path, pg_dsn, monkeypatch):
        """Same crash window, but on the FINAL batch (the one whose commit
        is immediately followed by index/ANALYZE finalization for a
        non-tglc table there's no extra finalize step, so this exercises
        the loop's last iteration specifically)."""
        db = _make_sqlite(tmp_path)
        sq = mig.connect_sqlite_ro(db)
        pg = mig.connect_pg(pg_dsn)
        ckpt_dir = tmp_path / "ckpt"

        call_count = {"n": 0}
        real_save_checkpoint = mig.save_checkpoint

        def flaky_local_write(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 6:  # 12 rows / batch_rows=2 = 6 batches
                raise RuntimeError("simulated crash on the last batch")
            return real_save_checkpoint(*args, **kwargs)

        monkeypatch.setattr(mig, "save_checkpoint", flaky_local_write)

        with pytest.raises(RuntimeError, match="simulated crash"):
            mig.migrate_table(sq, pg, "fits_paths", resume=False,
                               checkpoint_dir=ckpt_dir, batch_rows=2)
        sq.close()
        pg.close()

        # PG already has ALL 12 rows + an "in_progress" checkpoint (the
        # "completed" marker + local mirror never got written) — resume
        # must recognize there's nothing left to copy and just finalize.
        sq2 = mig.connect_sqlite_ro(db)
        pg2 = mig.connect_pg(pg_dsn)
        n = mig.migrate_table(sq2, pg2, "fits_paths", resume=True,
                               checkpoint_dir=ckpt_dir, batch_rows=2)
        assert n == 12
        sq2.close()
        pg2.close()

        ok = mig.run_verify(db, pg_dsn, ["fits_paths"], batch_rows=4)
        assert ok is True


@pytestmark_pg
class TestVerifyGate:
    def _migrate_all(self, sq, pg, tmp_path, batch_rows=4):
        for t in ("fits_paths", "metadata", "fits_blobs", "tglc", "nearby_stars"):
            mig.migrate_table(sq, pg, t, resume=False,
                               checkpoint_dir=tmp_path / "ckpt", batch_rows=batch_rows)

    def test_clean_migration_passes_verification(self, tmp_path, pg_dsn):
        db = _make_sqlite(tmp_path)
        sq = mig.connect_sqlite_ro(db)
        pg = mig.connect_pg(pg_dsn)
        self._migrate_all(sq, pg, tmp_path)
        sq.close()
        pg.close()

        ok = mig.run_verify(db, pg_dsn, mig.TABLES, batch_rows=4)
        assert ok is True

    def test_injected_row_discrepancy_fails_sample_check(self, tmp_path, pg_dsn):
        db = _make_sqlite(tmp_path)
        sq = mig.connect_sqlite_ro(db)
        pg = mig.connect_pg(pg_dsn)
        self._migrate_all(sq, pg, tmp_path)
        sq.close()

        # Corrupt one row's full_path in PG — a discrepancy from the source.
        pg.execute("UPDATE fits_paths SET full_path = %s WHERE filename = %s",
                   ("/CORRUPTED/path.fits", "100.fits"))
        pg.commit()
        pg.close()

        sq3 = mig.connect_sqlite_ro(db)
        pg3 = mig.connect_pg(pg_dsn)
        try:
            ok, detail = mig.verify_sample(
                sq3, pg3, "fits_paths",
                batch_rows=4, sample_size=12, seed=mig.SAMPLE_SEED)
        finally:
            sq3.close()
            pg3.close()
        assert ok is False
        assert detail["n_mismatches"] >= 1

    def test_injected_missing_row_fails_count_check(self, tmp_path, pg_dsn):
        db = _make_sqlite(tmp_path)
        sq = mig.connect_sqlite_ro(db)
        pg = mig.connect_pg(pg_dsn)
        self._migrate_all(sq, pg, tmp_path)
        sq.close()

        pg.execute("DELETE FROM fits_paths WHERE filename = %s", ("100.fits",))
        pg.commit()

        sq2 = mig.connect_sqlite_ro(db)
        ok, detail = mig.verify_counts(sq2, pg, "fits_paths")
        assert ok is False
        assert detail["sqlite"] != detail["pg"]
        sq2.close()
        pg.close()

    def test_indexed_basenames_union_invariant_passes_on_clean_migration(self, tmp_path, pg_dsn):
        db = _make_sqlite(tmp_path)
        sq = mig.connect_sqlite_ro(db)
        pg = mig.connect_pg(pg_dsn)
        self._migrate_all(sq, pg, tmp_path)
        sq.close()
        pg.close()

        ok, detail = mig.verify_indexed_basenames(db, pg_dsn, [1, 2])
        assert ok is True
        assert detail["sectors_checked"] == 2

    def test_tic_ids_order_invariant_passes_on_clean_migration(self, tmp_path, pg_dsn):
        db = _make_sqlite(tmp_path)
        sq = mig.connect_sqlite_ro(db)
        pg = mig.connect_pg(pg_dsn)
        self._migrate_all(sq, pg, tmp_path)
        sq.close()
        pg.close()

        ok, detail = mig.verify_tic_ids_order(db, pg_dsn, [1, 2])
        assert ok is True

    def test_tic_ids_order_invariant_detects_injected_reorder(self, tmp_path, pg_dsn):
        """Directly break the seq ordering for sector 1 in PG and confirm the
        order-invariant check catches it."""
        db = _make_sqlite(tmp_path)
        sq = mig.connect_sqlite_ro(db)
        pg = mig.connect_pg(pg_dsn)
        self._migrate_all(sq, pg, tmp_path)
        sq.close()

        # Swap the seq values of the first two sector-1 rows (reorders them).
        rows = pg.execute(
            "SELECT tic_id, seq FROM tglc WHERE sector = 1 ORDER BY seq ASC LIMIT 2"
        ).fetchall()
        (tic_a, seq_a), (tic_b, seq_b) = rows
        pg.execute("UPDATE tglc SET seq = %s WHERE tic_id = %s AND sector = 1",
                   (seq_b, tic_a))
        pg.execute("UPDATE tglc SET seq = %s WHERE tic_id = %s AND sector = 1",
                   (seq_a, tic_b))
        pg.commit()
        pg.close()

        ok, detail = mig.verify_tic_ids_order(db, pg_dsn, [1])
        assert ok is False
        assert detail["mismatches"]
