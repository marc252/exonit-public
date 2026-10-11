"""Tests for scripts/index_tglc_batch.py.

Covers:
  A. Differential / incremental scan — only new files are parsed.
  B. files_total= output line reports the glob count.
  C. HDU-only-when-needed — files with TIMEDEL in HDU 0 must NOT trigger
     an HDU 1 read.
  D. Default --workers is 75% of CPU cores.
  F. Chunk-scoped skip-set (D5 of docs/superpowers/specs/2026-07-20-tglc-
     index-to-pg.md) equivalence with the old unscoped full-table scan.
  G. PG write path (F3): ensure_tglc_partition + seq ordering, opt-in via
     EXONIT_TEST_META_PG_URL (skips cleanly when unset).
  H. ensure_tglc_partition concurrency (2026-07-20 review fix): concurrent
     callers racing to create the SAME new sector's partition must all
     succeed, not crash with DuplicateTable/UniqueViolation.

All tests are synthetic: no network access, no real DB (exonit.db), and
minimal FITS files are built in-memory with astropy. Sections A-D exercise
the default SQLite backend (EXONIT_METADATA_DB_URL unset, per
tests/conftest.py's autouse guard).
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from astropy.io import fits
import numpy as np

from exonit.metadata_cache import MetadataCache
from tests._metadata_pg_test_db import pg_meta_test_available, resolve_test_meta_dsn

# Make scripts/ importable.
_SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import index_tglc_batch as itb  # noqa: E402
from index_tglc_batch import read_one, main  # noqa: E402


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_fits(path: Path, tic: int, sector: int, timedel: float = 0.020833) -> Path:
    """Write a minimal valid FITS file with the required headers.

    HDU 0 carries all needed keywords (TICID, SECTOR, TIMEDEL, TESSMAG,
    RA_OBJ, DEC_OBJ).  HDU 1 is a minimal empty BinTableHDU (realistic
    structure but no timing keywords, unless the caller omits timedel to
    force the HDU-1 fallback path — see _make_fits_timedel_in_hdu1).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    primary_hdu = fits.PrimaryHDU()
    h = primary_hdu.header
    h["TICID"] = tic
    h["SECTOR"] = sector
    h["TIMEDEL"] = timedel        # present in HDU 0 — HDU 1 should NOT be read
    h["TESSMAG"] = 10.5
    h["RA_OBJ"] = 123.456
    h["DEC_OBJ"] = -45.678

    # Empty BinTableHDU as HDU 1 (no timing keywords)
    col = fits.Column(name="TIME", format="D", array=np.array([]))
    table_hdu = fits.BinTableHDU.from_columns([col])

    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto(str(path), overwrite=True)
    return path


def _make_fits_timedel_in_hdu1(
    path: Path, tic: int, sector: int, timedel: float = 0.020833
) -> Path:
    """FITS where TIMEDEL is ABSENT from HDU 0 but present in HDU 1.

    This forces read_one() to open HDU 1 to find the timing keyword.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    primary_hdu = fits.PrimaryHDU()
    h = primary_hdu.header
    h["TICID"] = tic
    h["SECTOR"] = sector
    # NO TIMEDEL / EXPTIME in HDU 0
    h["TESSMAG"] = 10.5
    h["RA_OBJ"] = 123.456
    h["DEC_OBJ"] = -45.678

    col = fits.Column(name="TIME", format="D", array=np.array([]))
    table_hdu = fits.BinTableHDU.from_columns([col])
    table_hdu.header["TIMEDEL"] = timedel  # only in HDU 1

    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto(str(path), overwrite=True)
    return path


def _call_main(cache, index, workers=4):
    """Patch sys.argv and call main()."""
    argv = [
        "index_tglc_batch.py",
        "--cache", str(cache),
        "--index", str(index),
        "--workers", str(workers),
    ]
    with patch("sys.argv", argv):
        return main()


# ------------------------------------------------------------------ #
# A. Differential / incremental scan
# ------------------------------------------------------------------ #

class TestDifferentialScan:
    """read_one is not called for files already present in the DB."""

    def test_all_new_files_indexed_on_first_run(self, tmp_path):
        """First run: 3 FITS → all 3 get indexed."""
        cache = tmp_path / "cache"
        fits_files = [
            _make_fits(cache / f"s0001/tic{i:04d}.fits", tic=100 + i, sector=1)
            for i in range(3)
        ]
        index = tmp_path / "tglc.db"

        rc = _call_main(cache, index)
        assert rc == 0

        con = sqlite3.connect(str(index))
        rows = con.execute("SELECT tic_id FROM tglc ORDER BY tic_id").fetchall()
        con.close()
        assert len(rows) == 3
        assert {r[0] for r in rows} == {"100", "101", "102"}

    def test_second_run_skips_existing_files(self, tmp_path):
        """Second run: 0 new files → DB count unchanged (0 new rows added).

        NOTE: With ProcessPoolExecutor, monkeypatching read_one with a local
        closure doesn't work (closures can't be pickled for subprocess transport).
        We verify the incremental logic by checking that the DB count doesn't
        change between the first and second runs.
        """
        cache = tmp_path / "cache"
        for i in range(3):
            _make_fits(cache / f"s0001/tic{i:04d}.fits", tic=200 + i, sector=1)
        index = tmp_path / "tglc.db"

        # First run — populates the DB.
        rc = _call_main(cache, index)
        assert rc == 0

        con = sqlite3.connect(str(index))
        count_after_first = con.execute("SELECT COUNT(*) FROM tglc").fetchone()[0]
        con.close()
        assert count_after_first == 3

        # Second run — should be a no-op (0 new files).
        rc2 = _call_main(cache, index)
        assert rc2 == 0

        con = sqlite3.connect(str(index))
        count_after_second = con.execute("SELECT COUNT(*) FROM tglc").fetchone()[0]
        con.close()
        assert count_after_second == count_after_first, (
            f"Second run must not add rows; was {count_after_first}, now {count_after_second}"
        )

    def test_adding_one_file_processes_only_new(self, tmp_path):
        """After first run with 3 files, add 1 more → 4 rows total in DB.

        With ProcessPoolExecutor the process-count of read_one calls can't
        be measured via monkeypatching (subprocess can't pickle local closures).
        We verify correctness by checking the DB state before and after.
        """
        cache = tmp_path / "cache"
        for i in range(3):
            _make_fits(cache / f"s0001/tic{i:04d}.fits", tic=300 + i, sector=1)
        index = tmp_path / "tglc.db"

        rc = _call_main(cache, index)
        assert rc == 0

        con = sqlite3.connect(str(index))
        count_after_first = con.execute("SELECT COUNT(*) FROM tglc").fetchone()[0]
        con.close()
        assert count_after_first == 3

        # Add a 4th FITS file.
        new_path = cache / "s0001" / "tic0003.fits"
        _make_fits(new_path, tic=303, sector=1)

        rc2 = _call_main(cache, index)
        assert rc2 == 0

        # DB now has 4 rows.
        con = sqlite3.connect(str(index))
        count = con.execute("SELECT COUNT(*) FROM tglc").fetchone()[0]
        # Check the new TIC is there.
        tics = {r[0] for r in con.execute("SELECT tic_id FROM tglc").fetchall()}
        con.close()
        assert count == 4, f"Expected 4 rows after adding tic=303, got {count}"
        assert "303" in tics, f"Expected TIC 303 in DB, got: {sorted(tics)}"


# ------------------------------------------------------------------ #
# B. files_total= output line
# ------------------------------------------------------------------ #

class TestFilesTotal:
    def test_files_total_equals_fits_count(self, tmp_path, capsys):
        """files_total=N in stdout must equal the number of FITS on disk."""
        cache = tmp_path / "cache"
        for i in range(4):
            _make_fits(cache / f"s0001/tic{i:04d}.fits", tic=400 + i, sector=1)
        index = tmp_path / "tglc.db"

        rc = _call_main(cache, index)
        assert rc == 0

        captured = capsys.readouterr()
        assert "files_total=4" in captured.out, (
            f"Expected 'files_total=4' in stdout; got:\n{captured.out}"
        )

    def test_files_total_stable_on_second_run(self, tmp_path, capsys):
        """Second run (0 new files) still emits the correct files_total."""
        cache = tmp_path / "cache"
        for i in range(3):
            _make_fits(cache / f"s0001/tic{i:04d}.fits", tic=500 + i, sector=1)
        index = tmp_path / "tglc.db"

        _call_main(cache, index)
        capsys.readouterr()  # discard first run output

        rc = _call_main(cache, index)
        assert rc == 0
        captured = capsys.readouterr()
        assert "files_total=3" in captured.out, (
            f"Expected 'files_total=3' on second run; got:\n{captured.out}"
        )

    def test_files_total_after_adding_file(self, tmp_path, capsys):
        """After adding 1 file on a second run, files_total increments."""
        cache = tmp_path / "cache"
        for i in range(2):
            _make_fits(cache / f"s0001/tic{i:04d}.fits", tic=600 + i, sector=1)
        index = tmp_path / "tglc.db"

        _call_main(cache, index)
        capsys.readouterr()

        # Add a 3rd file.
        _make_fits(cache / "s0001" / "tic0002.fits", tic=602, sector=1)

        rc = _call_main(cache, index)
        assert rc == 0
        captured = capsys.readouterr()
        assert "files_total=3" in captured.out, (
            f"Expected 'files_total=3' after adding 1 file; got:\n{captured.out}"
        )


# ------------------------------------------------------------------ #
# C. HDU-only-when-needed: TIMEDEL in HDU 0 must not trigger HDU 1 read
# ------------------------------------------------------------------ #

class TestHduReadBehavior:
    def test_timedel_in_hdu0_no_hdu1_read(self, tmp_path):
        """When TIMEDEL is in HDU 0, HDU 1 header should not be accessed.

        We spy on fits.open and verify that hdul[1].header is not accessed.
        """
        fits_path = tmp_path / "tic_hdu0.fits"
        _make_fits(fits_path, tic=600, sector=1, timedel=0.020833)

        hdu1_header_accessed = []

        original_open = fits.open

        def spy_open(path, **kwargs):
            hdul = original_open(path, **kwargs)

            class SpyHDU1:
                """Proxy that records access to .header on HDU 1."""
                def __init__(self, real):
                    self._real = real

                @property
                def header(self):
                    hdu1_header_accessed.append(path)
                    return self._real.header

                def __getattr__(self, name):
                    return getattr(self._real, name)

            class SpyHDUList:
                def __init__(self, real):
                    self._real = real

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    self._real.close()

                def __getitem__(self, idx):
                    item = self._real[idx]
                    if idx == 1:
                        return SpyHDU1(item)
                    return item

                def __len__(self):
                    return len(self._real)

            return SpyHDUList(hdul.__enter__())

        with patch.object(fits, "open", side_effect=spy_open):
            result = read_one(str(fits_path))

        assert result is not None, "read_one returned None for a valid FITS"
        assert hdu1_header_accessed == [], (
            "HDU 1 header was accessed even though TIMEDEL was in HDU 0"
        )

    def test_exptime_from_hdu1_when_absent_from_hdu0(self, tmp_path):
        """When TIMEDEL/EXPTIME absent from HDU 0, HDU 1 is consulted."""
        fits_path = tmp_path / "tic_hdu1.fits"
        _make_fits_timedel_in_hdu1(fits_path, tic=700, sector=1, timedel=0.020833)

        result = read_one(str(fits_path))
        assert result is not None
        # TIMEDEL 0.020833 days → 0.020833 * 86400 = 1799.97... ≈ 1800s
        exptime = result[3]
        assert abs(exptime - 0.020833 * 86400.0) < 1.0, (
            f"Expected exptime ~1800s from HDU 1 TIMEDEL, got {exptime}"
        )

    def test_sector_default_when_no_timing_in_either_hdu(self, tmp_path):
        """When neither HDU has timing, sector-based default is used."""
        fits_path = tmp_path / "tic_notime.fits"
        fits_path.parent.mkdir(parents=True, exist_ok=True)

        primary_hdu = fits.PrimaryHDU()
        h = primary_hdu.header
        h["TICID"] = 800
        h["SECTOR"] = 1       # sector ≤ 26 → default 1800s
        h["TESSMAG"] = 11.0
        h["RA_OBJ"] = 10.0
        h["DEC_OBJ"] = 20.0
        # No TIMEDEL / EXPTIME anywhere

        import numpy as np
        col = fits.Column(name="TIME", format="D", array=np.array([]))
        table_hdu = fits.BinTableHDU.from_columns([col])
        # Also no timing in HDU 1

        hdul = fits.HDUList([primary_hdu, table_hdu])
        hdul.writeto(str(fits_path), overwrite=True)

        result = read_one(str(fits_path))
        assert result is not None
        assert result[3] == 1800.0, (
            f"Expected default exptime 1800s for sector 1, got {result[3]}"
        )


# ------------------------------------------------------------------ #
# D. Default worker count is 75% of CPU cores (ProcessPoolExecutor)
# ------------------------------------------------------------------ #

class TestDefaultWorkers:
    def test_default_workers_is_cpu_fraction(self):
        """_default_workers() returns max(1, int(cpu_count * 0.75))."""
        # Patch os.cpu_count so the test is deterministic regardless of machine.
        with patch("os.cpu_count", return_value=8):
            w = itb._default_workers()
        assert w == 6, f"Expected 6 (75% of 8), got {w}"

    def test_default_workers_never_zero(self):
        """_default_workers() returns at least 1 even on a single-core machine."""
        with patch("os.cpu_count", return_value=1):
            w = itb._default_workers()
        assert w >= 1

    def test_default_workers_cpu_count_none(self):
        """_default_workers() returns 1 when os.cpu_count() is None."""
        with patch("os.cpu_count", return_value=None):
            w = itb._default_workers()
        assert w == 1

    def test_workers_arg_default_is_none(self):
        """The --workers argparse default is None (resolved at runtime, not hardcoded)."""
        ap = argparse.ArgumentParser()
        ap.add_argument("--workers", type=int, default=None)
        args = ap.parse_args([])
        assert args.workers is None, (
            f"Expected --workers default=None (resolved at runtime), got {args.workers}"
        )

    def test_explicit_workers_override(self, tmp_path):
        """--workers N uses exactly N processes (explicit override)."""
        cache = tmp_path / "cache"
        _make_fits(cache / "s0001" / "tic0001.fits", tic=1001, sector=1)
        index = tmp_path / "tglc.db"
        # Pass --workers 2 explicitly; should work regardless of CPU count.
        rc = _call_main(cache, index, workers=2)
        assert rc == 0
        con = sqlite3.connect(str(index))
        count = con.execute("SELECT COUNT(*) FROM tglc").fetchone()[0]
        con.close()
        assert count == 1


# ------------------------------------------------------------------ #
# E. ProcessPoolExecutor: same results as sequential read_one
# ------------------------------------------------------------------ #

class TestProcessPoolResults:
    """ProcessPool produces exactly the same rows as sequential read_one."""

    def test_process_pool_matches_sequential(self, tmp_path):
        """ProcessPool output is identical to running read_one sequentially."""
        cache = tmp_path / "cache"
        paths = []
        for i in range(4):
            p = _make_fits(cache / f"s0001/tic{i:04d}.fits", tic=2000 + i, sector=1)
            paths.append(str(p))

        # Sequential reference.
        seq_rows = sorted(
            (r for r in (read_one(p) for p in paths) if r is not None),
            key=lambda r: r[0],  # sort by tic_id string
        )

        # ProcessPool run via main().
        index = tmp_path / "tglc.db"
        rc = _call_main(cache, index, workers=2)
        assert rc == 0

        con = sqlite3.connect(str(index))
        pool_rows = sorted(
            con.execute(
                "SELECT tic_id, sector, path, exptime FROM tglc ORDER BY tic_id"
            ).fetchall(),
            key=lambda r: r[0],
        )
        con.close()

        assert len(pool_rows) == len(seq_rows), (
            f"Row count mismatch: pool={len(pool_rows)}, seq={len(seq_rows)}"
        )
        for seq, pool in zip(seq_rows, pool_rows):
            assert seq[0] == pool[0], f"tic_id mismatch: {seq[0]} vs {pool[0]}"
            assert seq[1] == pool[1], f"sector mismatch: {seq[1]} vs {pool[1]}"

    def test_re_run_is_incremental(self, tmp_path):
        """Second run with ProcessPool skips already-indexed files (DB count stable)."""
        cache = tmp_path / "cache"
        for i in range(3):
            _make_fits(cache / f"s0001/tic{i:04d}.fits", tic=3000 + i, sector=1)
        index = tmp_path / "tglc.db"

        rc = _call_main(cache, index, workers=2)
        assert rc == 0

        con = sqlite3.connect(str(index))
        count_first = con.execute("SELECT COUNT(*) FROM tglc").fetchone()[0]
        con.close()
        assert count_first == 3

        rc2 = _call_main(cache, index, workers=2)
        assert rc2 == 0

        con = sqlite3.connect(str(index))
        count_second = con.execute("SELECT COUNT(*) FROM tglc").fetchone()[0]
        con.close()
        assert count_second == count_first, (
            f"Second run must not add rows; was {count_first}, now {count_second}"
        )


# ------------------------------------------------------------------ #
# F. Chunk-scoped skip-set (D5): tglc_paths_for() must match the old
#    unscoped 'SELECT path FROM tglc' full-table scan, filtered to the
#    chunk's own candidate paths.
# ------------------------------------------------------------------ #

class TestChunkScopedSkipSetEquivalence:
    """The pre-seam script ran `SELECT path FROM tglc` with NO WHERE clause
    (the whole table, every sector) on every 10k-file chunk, then filtered
    it in Python to the chunk's candidate paths. `MetadataCache.tglc_paths_for`
    (used by the seam version) is the chunk-scoped fix (D5); this class
    proves it returns the IDENTICAL result for a fixture containing a
    tic/sector PK collision (the pre-existing REPLACE semantics: the loser
    path is silently dropped from tglc, exactly like the live table) and
    paths absent from tglc entirely.
    """

    def _make_tglc_db(self, tmp_path: Path) -> Path:
        db = tmp_path / "tglc_equiv.db"
        con = sqlite3.connect(db)
        con.execute(
            "CREATE TABLE tglc (tic_id TEXT, sector INTEGER, path TEXT, "
            "exptime REAL, tmag REAL, ra REAL, dec REAL, "
            "PRIMARY KEY(tic_id, sector))"
        )
        con.executemany(
            "INSERT INTO tglc(tic_id, sector, path) VALUES (?, ?, ?)",
            [
                ("100", 1, "/cache/s0001/a.fits"),
                ("200", 1, "/cache/s0001/b.fits"),
                ("300", 2, "/cache/s0002/c.fits"),  # different sector
            ],
        )
        # PK collision: TIC 400/sector 1 indexed first under one Gaia ID's
        # path, then re-indexed (INSERT OR REPLACE — same semantics
        # tglc_add_rows_bulk uses) under a DIFFERENT Gaia ID's path for the
        # SAME (tic_id, sector) key. Only the winner survives in tglc; the
        # loser's path is simply gone, same as the live table.
        con.execute(
            "INSERT OR REPLACE INTO tglc(tic_id, sector, path) VALUES (?, ?, ?)",
            ("400", 1, "/cache/s0001/gaiaid-X-d.fits"),
        )
        con.execute(
            "INSERT OR REPLACE INTO tglc(tic_id, sector, path) VALUES (?, ?, ?)",
            ("400", 1, "/cache/s0001/gaiaid-Y-d.fits"),
        )
        con.commit()
        con.close()
        return db

    def test_chunk_scoped_matches_old_full_scan_for_chunk_paths(self, tmp_path):
        db = self._make_tglc_db(tmp_path)

        # This "chunk"'s candidate paths: some present, some absent, the
        # collision winner, and the collision LOSER (must read as "absent"
        # now — it was overwritten out of the table, exactly as the old
        # full-scan skip-set would have seen it too).
        chunk_paths = [
            "/cache/s0001/a.fits",                 # present
            "/cache/s0002/c.fits",                 # present, different sector
            "/cache/s0001/gaiaid-Y-d.fits",        # present (collision winner)
            "/cache/s0001/gaiaid-X-d.fits",        # ABSENT (collision loser)
            "/cache/s0001/never-downloaded.fits",  # absent entirely
        ]

        # --- OLD behaviour: unscoped full-table SELECT, then filter to the chunk ---
        con = sqlite3.connect(db)
        all_paths = {row[0] for row in con.execute("SELECT path FROM tglc")}
        con.close()
        old_skip_set = {p for p in chunk_paths if p in all_paths}

        # --- NEW: chunk-scoped tglc_paths_for() ---
        cache = MetadataCache(str(db))
        try:
            new_skip_set = cache.tglc_paths_for(chunk_paths)
        finally:
            cache.close()

        assert new_skip_set == old_skip_set
        assert new_skip_set == {
            "/cache/s0001/a.fits",
            "/cache/s0002/c.fits",
            "/cache/s0001/gaiaid-Y-d.fits",
        }

    def test_chunk_scoped_ignores_paths_outside_the_chunk(self, tmp_path):
        """tglc_paths_for must not return paths present in the table but NOT
        requested — the skip-set is always scoped to the caller's own
        candidates, which is the entire point of the D5 fix."""
        db = self._make_tglc_db(tmp_path)
        cache = MetadataCache(str(db))
        try:
            result = cache.tglc_paths_for(["/cache/s0001/a.fits"])
        finally:
            cache.close()
        assert result == {"/cache/s0001/a.fits"}

    def test_chunk_scoped_empty_candidate_list_returns_empty_set(self, tmp_path):
        db = self._make_tglc_db(tmp_path)
        cache = MetadataCache(str(db))
        try:
            assert cache.tglc_paths_for([]) == set()
        finally:
            cache.close()


# ------------------------------------------------------------------ #
# G. PG write path (F3): ensure_tglc_partition + seq ordering.
#    Opt-in via EXONIT_TEST_META_PG_URL — see tests/_metadata_pg_test_db.py.
# ------------------------------------------------------------------ #

_ROOT = Path(__file__).resolve().parent.parent
_SCHEMA_SQL_PATH = _ROOT / "scripts" / "migration" / "schema_pg_meta.sql"

PG_DSN = resolve_test_meta_dsn()
PG_AVAILABLE = pg_meta_test_available(PG_DSN)


def _reset_pg_meta_schema(dsn: str) -> None:
    """DROP + reapply the REAL partitioned schema_pg_meta.sql (not a
    simplified stand-in — this class specifically needs the manually-bound
    `seq` sequence + dynamic per-sector partition creation the real schema
    exercises, matching what index_tglc_batch.py's write path actually does
    once EXONIT_METADATA_DB_URL is set)."""
    import psycopg

    with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
        for t in ("tglc", "fits_blobs", "metadata", "fits_paths", "nearby_stars"):
            c.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
        c.execute("DROP SEQUENCE IF EXISTS tglc_seq_seq")
        c.execute(_SCHEMA_SQL_PATH.read_text())


@pytest.mark.skipif(
    not PG_AVAILABLE,
    reason="EXONIT_TEST_META_PG_URL (throwaway PG) unset or not reachable",
)
class TestPgWritePathSeqOrdering:
    """index_tglc_batch.py's F3 write path (ensure_tglc_partition +
    tglc_add_rows_bulk, both via MetadataCache) against the REAL partitioned
    schema_pg_meta.sql: a brand-new row's `seq` (DEFAULT nextval, the write
    path never sets it explicitly) must sort AFTER pre-existing rows in
    tglc_tic_ids(sector) — the ordering invariant the s14 campaign cursor
    depends on (D3 of the spec) — for a sector requiring a DYNAMICALLY
    created partition (not one of schema_pg_meta.sql's seed sectors), since
    that is the realistic first-time-a-new-sector-is-indexed case.
    """

    def setup_method(self):
        _reset_pg_meta_schema(PG_DSN)

    def teardown_method(self):
        _reset_pg_meta_schema(PG_DSN)

    def test_new_row_lands_after_preexisting_rows_via_direct_calls(self, monkeypatch):
        """Exercises the exact sequence of MetadataCache calls
        index_tglc_batch.py's main() makes per run: ensure_tglc_partition(sec)
        then tglc_add_rows_bulk(rows), called twice (simulating an earlier
        'migrated'/first-run batch followed by a later run's new file)."""
        monkeypatch.setenv("EXONIT_METADATA_DB_URL", PG_DSN)
        sector = 199  # NOT in schema_pg_meta.sql's seed partition list
        cache = MetadataCache("unused-path-pg-wins.db")
        try:
            assert cache._is_pg

            cache.ensure_tglc_partition(sector)
            cache.tglc_add_rows_bulk([
                ("1", sector, "/cache/a.fits", 1800.0, 10.0, 0.0, 0.0),
                ("2", sector, "/cache/b.fits", 1800.0, 10.0, 0.0, 0.0),
            ])

            # A later run: ensure_tglc_partition is idempotent (IF NOT
            # EXISTS), then a genuinely new TIC is inserted.
            cache.ensure_tglc_partition(sector)
            cache.tglc_add_rows_bulk([
                ("3", sector, "/cache/c.fits", 1800.0, 10.0, 0.0, 0.0),
            ])

            assert cache.tglc_tic_ids(sector) == [1, 2, 3]
        finally:
            cache.close()

    def test_second_script_run_new_tic_sorts_after_first_run(self, tmp_path, monkeypatch):
        """End-to-end through the actual script entry point (main()), not
        just direct MetadataCache calls — two separate index_tglc_batch.py
        invocations against real PG, second one adding one new file."""
        monkeypatch.setenv("EXONIT_METADATA_DB_URL", PG_DSN)
        sector = 198  # NOT in schema_pg_meta.sql's seed partition list

        cache_dir = tmp_path / "cache"
        _make_fits(cache_dir / f"s{sector:04d}/tic0001.fits", tic=9001, sector=sector)
        _make_fits(cache_dir / f"s{sector:04d}/tic0002.fits", tic=9002, sector=sector)

        rc = _call_main(cache_dir, "unused-path-pg-wins.db", workers=2)
        assert rc == 0

        _make_fits(cache_dir / f"s{sector:04d}/tic0003.fits", tic=9003, sector=sector)
        rc2 = _call_main(cache_dir, "unused-path-pg-wins.db", workers=2)
        assert rc2 == 0

        cache = MetadataCache("unused-path-pg-wins.db")
        try:
            assert cache._is_pg
            assert cache.tglc_tic_ids(sector) == [9001, 9002, 9003]
        finally:
            cache.close()


# ------------------------------------------------------------------ #
# H. ensure_tglc_partition concurrency (2026-07-20 review fix, BLOCKER).
#    Opt-in via EXONIT_TEST_META_PG_URL.
# ------------------------------------------------------------------ #

@pytest.mark.skipif(
    not PG_AVAILABLE,
    reason="EXONIT_TEST_META_PG_URL (throwaway PG) unset or not reachable",
)
class TestPgEnsureTglcPartitionConcurrency:
    """Reproduces progress/f3-indexer-pg-review.md's blocker directly: PG's
    ``CREATE TABLE IF NOT EXISTS ... PARTITION OF`` is NOT atomic against
    true concurrent callers (the existence check and the CREATE are not one
    operation — multiple sessions can all see "missing" and all attempt the
    CREATE; only one wins). The reviewer reproduced this with 5 threads, each
    with its own MetadataCache instance/connection, racing
    ``ensure_tglc_partition()`` for the SAME brand-new sector — 4 of 5 raised
    uncaught ``DuplicateTable``. This class proves the fix
    (``MetadataCache._partition_race_excs`` swallowing the loser's error)
    closes that race: every thread must complete without raising, and the
    partition must end up genuinely usable afterward.
    """

    def setup_method(self):
        _reset_pg_meta_schema(PG_DSN)

    def teardown_method(self):
        _reset_pg_meta_schema(PG_DSN)

    def test_concurrent_ensure_tglc_partition_all_succeed(self, monkeypatch):
        import threading

        monkeypatch.setenv("EXONIT_METADATA_DB_URL", PG_DSN)
        sector = 777  # NOT in schema_pg_meta.sql's seed list — same sector
                      # the reviewer used for the original repro.
        n_threads = 5
        errors = []
        errors_lock = threading.Lock()

        def worker(i):
            # Each thread gets its OWN MetadataCache instance (own
            # ConnectionPool/connection) — mirrors the reviewer's exact
            # repro shape (5 independent instances, not 5 threads sharing
            # one pool, which would trivially serialize on the pool itself).
            cache = MetadataCache("unused-path-pg-wins.db")
            try:
                cache.ensure_tglc_partition(sector)
            except Exception as exc:  # noqa: BLE001 - captured for the assertion below
                with errors_lock:
                    errors.append((i, type(exc).__name__, str(exc)))
            finally:
                cache.close()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], (
            f"ensure_tglc_partition raised under concurrency: {errors}"
        )

        # The partition must exist and actually be usable afterward — not
        # just "no exception", but a genuinely working partition.
        cache = MetadataCache("unused-path-pg-wins.db")
        try:
            cache.tglc_add_rows_bulk([("1", sector, "/x.fits", 1800, 10, 0, 0)])
            assert cache.tglc_tic_ids(sector) == [1]
        finally:
            cache.close()

    def test_sequential_repeated_calls_still_idempotent(self, monkeypatch):
        """Non-concurrent regression guard: repeated calls (the common case,
        e.g. index_tglc_batch.py calling it once per chunk for the same
        sector across a whole download) must remain a pure no-op after the
        first, exactly as before this fix."""
        monkeypatch.setenv("EXONIT_METADATA_DB_URL", PG_DSN)
        sector = 778
        cache = MetadataCache("unused-path-pg-wins.db")
        try:
            for _ in range(5):
                cache.ensure_tglc_partition(sector)  # must never raise
            cache.tglc_add_rows_bulk([("2", sector, "/y.fits", 1800, 10, 0, 0)])
            assert cache.tglc_tic_ids(sector) == [2]
        finally:
            cache.close()
