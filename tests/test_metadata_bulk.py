"""Tests for exonit.metadata_index.ensure_metadata_bulk and the
MetadataCache bulk helpers it uses (get_entries_bulk / tglc_lookup_bulk /
write_metadata_bulk).

Implements docs/superpowers/specs/2026-09-26-metadata-refresh-and-bulk-
enqueue-design.md §3. All DBs are created under tmp_path (SQLite) or a
throwaway PostgreSQL test database (EXONIT_TEST_META_PG_URL) — never
data/metadata_cache.db or the production exonit_meta database.
"""
import random

import pytest

from exonit.metadata_cache import MetadataCache
from exonit.metadata_index import ensure_metadata, ensure_metadata_bulk


def _seed_tglc(cache, rows):
    """rows: list of (tic_id, sector, path, exptime, tmag, ra, dec) tuples."""
    cache.ensure_tglc_table()
    cache.tglc_add_rows_bulk(rows)


@pytest.fixture
def cache(tmp_path):
    c = MetadataCache(str(tmp_path / "meta.db"))
    yield c
    c.close()


# --------------------------------------------------------------------------- #
# Create — no existing row
# --------------------------------------------------------------------------- #

class TestCreate:
    def test_creates_from_tglc_multi_sector(self, cache):
        _seed_tglc(cache, [
            ("100", 1, "/x/hlsp_tglc-a-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0),
            ("100", 2, "/x/hlsp_tglc-a-s0002-1.fits", 1800.0, 9.0, 1.0, 2.0),
        ])
        res = ensure_metadata_bulk(["100"], cache)
        assert res.ready == ["100"]
        assert res.created == 1
        assert res.refreshed == res.unchanged == res.preserved_non_tglc == 0
        entry = cache.get_entry("100")
        assert entry["sectors"] == [1, 2]
        assert entry["fits_files"] == [
            "hlsp_tglc-a-s0001-1.fits", "hlsp_tglc-a-s0002-1.fits"]
        assert entry["tmag"] == pytest.approx(9.0)
        assert entry["ra"] == pytest.approx(1.0)
        assert entry["dec"] == pytest.approx(2.0)
        assert entry["stellar_radius"] is None

    def test_qlp_fallback_creates_entry_when_no_tglc(self, cache, tmp_path):
        """No TGLC row -> falls back to the QLP (FFI cache) glob."""
        ffi_root = tmp_path / "ffi_cache"
        sector_dir = ffi_root / "s0005"
        sector_dir.mkdir(parents=True)
        tic = "700000001"
        pad = f"{int(tic):016d}"
        fname = f"hlsp_qlp_tess_ffi_s0005-{pad}_tess_v01_llc.fits"
        (sector_dir / fname).write_bytes(b"0" * 40_000)  # > 30_000 size gate

        res = ensure_metadata_bulk(
            [tic], cache, ffi_cache=str(ffi_root),
            spoc_root=str(tmp_path / "no_spoc"))
        assert res.created == 1
        assert res.ready == [tic]
        entry = cache.get_entry(tic)
        assert entry["sectors"] == [5]
        assert entry["fits_files"] == [fname]

    def test_spoc_fallback_when_no_tglc_no_qlp(self, cache, tmp_path):
        tic = "800000002"
        pad = f"{int(tic):016d}"
        spoc_dir = tmp_path / "spoc_root" / f"tess2020-s0007-{pad}-0123-s"
        spoc_dir.mkdir(parents=True)
        fname = f"tess2020-s0007-{pad}-0123-s_lc.fits"
        (spoc_dir / fname).write_bytes(b"0" * 100)

        res = ensure_metadata_bulk(
            [tic], cache, ffi_cache=str(tmp_path / "empty_ffi"),
            spoc_root=str(tmp_path / "spoc_root"))
        assert res.created == 1
        entry = cache.get_entry(tic)
        assert entry["sectors"] == [7]
        assert entry["fits_files"] == [fname]


# --------------------------------------------------------------------------- #
# no_fits
# --------------------------------------------------------------------------- #

class TestNoFits:
    def test_no_tglc_no_glob_is_no_fits(self, cache, tmp_path):
        res = ensure_metadata_bulk(
            ["999"], cache,
            ffi_cache=str(tmp_path / "empty_ffi"),
            spoc_root=str(tmp_path / "empty_spoc"))
        assert res.no_fits == ["999"]
        assert res.ready == []
        assert res.created == 0
        assert cache.get_entry("999") is None

    def test_row_with_empty_path_skipped(self, cache, tmp_path):
        """A tglc row with an empty path is not usable data — the TIC ends up
        no_fits (same contract as _tglc_lookup's null-path skip)."""
        _seed_tglc(cache, [("500", 1, "", 1800.0, 9.0, 1.0, 2.0)])
        res = ensure_metadata_bulk(
            ["500"], cache,
            ffi_cache=str(tmp_path / "empty_ffi"),
            spoc_root=str(tmp_path / "empty_spoc"))
        assert res.no_fits == ["500"]
        assert cache.get_entry("500") is None


# --------------------------------------------------------------------------- #
# Refresh — existing TGLC-sourced row, missing sector(s)
# --------------------------------------------------------------------------- #

class TestRefresh:
    def test_refresh_adds_new_sector_preserves_stellar_params(self, cache):
        # Old row: created previously (e.g. by a first-lot enqueue) with
        # FITS-header-derived stellar params tglc never populates.
        cache.add_entry(tic_id="200", sectors=[1], exptime=1800.0,
                        fits_files=["hlsp_tglc-a-s0001-1.fits"],
                        stellar_radius=1.5, teff=5700.0, logg=4.4,
                        tmag=9.5, ra=10.0, dec=-5.0)
        # tglc now also has sector 2 (a later download indexed it), with
        # tmag/ra/dec all NULL — a real case for some historical tglc rows.
        _seed_tglc(cache, [
            ("200", 1, "/x/hlsp_tglc-a-s0001-1.fits", 1800.0, None, None, None),
            ("200", 2, "/x/hlsp_tglc-a-s0002-1.fits", 1800.0, None, None, None),
        ])
        res = ensure_metadata_bulk(["200"], cache, refresh=True)
        assert res.refreshed == 1
        assert res.ready == ["200"]
        entry = cache.get_entry("200")
        assert entry["sectors"] == [1, 2]
        assert entry["fits_files"] == [
            "hlsp_tglc-a-s0001-1.fits", "hlsp_tglc-a-s0002-1.fits"]
        # stellar_radius/teff/logg carried over from the old row (tglc never
        # has them).
        assert entry["stellar_radius"] == pytest.approx(1.5)
        assert entry["teff"] == pytest.approx(5700.0)
        assert entry["logg"] == pytest.approx(4.4)
        # tmag/ra/dec: tglc's current rows are all NULL -> fall back to the
        # OLD row's values, not None.
        assert entry["tmag"] == pytest.approx(9.5)
        assert entry["ra"] == pytest.approx(10.0)
        assert entry["dec"] == pytest.approx(-5.0)

    def test_refresh_false_leaves_stale_row_untouched(self, cache):
        _seed_tglc(cache, [
            ("400", 1, "/x/hlsp_tglc-a-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0),
        ])
        ensure_metadata_bulk(["400"], cache)  # creates with sector [1]
        _seed_tglc(cache, [
            ("400", 2, "/x/hlsp_tglc-a-s0002-1.fits", 1800.0, 9.0, 1.0, 2.0),
        ])
        res = ensure_metadata_bulk(["400"], cache, refresh=False)
        assert res.unchanged == 1
        assert res.refreshed == 0
        entry = cache.get_entry("400")
        assert entry["sectors"] == [1]  # NOT refreshed

    def test_up_to_date_row_is_unchanged_even_with_refresh_true(self, cache):
        _seed_tglc(cache, [
            ("450", 1, "/x/hlsp_tglc-a-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0),
        ])
        ensure_metadata_bulk(["450"], cache, refresh=True)
        res = ensure_metadata_bulk(["450"], cache, refresh=True)
        assert res.unchanged == 1
        assert res.refreshed == 0
        assert res.created == 0


# --------------------------------------------------------------------------- #
# TOI (SPOC/QLP-source) short-circuit — never refreshed
# --------------------------------------------------------------------------- #

class TestToiPreserved:
    def test_preserves_non_tglc_row_byte_identical(self, cache):
        cache.add_entry(
            tic_id="300", sectors=[9], exptime=120.0,
            fits_files=["tess2020000000300-s0009-0000000000000300-0123-s_lc.fits"],
            stellar_radius=1.1, teff=5800.0, logg=4.5,
            tmag=8.0, ra=5.0, dec=6.0)
        before = cache.get_entry("300")
        # tglc ALSO has data for this TIC -- must not matter, the TOI
        # short-circuit is unconditional.
        _seed_tglc(cache, [
            ("300", 9, "/x/hlsp_tglc-a-s0009-1.fits", 1800.0, 1.0, 2.0, 3.0)])
        res = ensure_metadata_bulk(["300"], cache, refresh=True)
        assert res.preserved_non_tglc == 1
        assert res.ready == ["300"]
        assert res.refreshed == 0
        after = cache.get_entry("300")
        assert after == before

    def test_empty_fits_files_counts_as_tglc_source(self, cache):
        """An entry with fits_files=[] (edge case, e.g. a partial write) is
        treated as TGLC-eligible, not preserved -- matches _is_tglc_source's
        'empty is TGLC-eligible' rule."""
        cache.add_entry(tic_id="350", sectors=[], exptime=1800.0,
                        fits_files=[])
        _seed_tglc(cache, [
            ("350", 1, "/x/hlsp_tglc-a-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0)])
        res = ensure_metadata_bulk(["350"], cache, refresh=True)
        assert res.refreshed == 1
        assert res.preserved_non_tglc == 0


# --------------------------------------------------------------------------- #
# Chunk boundary + idempotence
# --------------------------------------------------------------------------- #

class TestChunkingAndIdempotence:
    def test_chunk_boundary_5001_tics_two_chunks(self, cache):
        n = 5001
        rows = [
            (str(i), 1, f"/x/hlsp_tglc-{i}-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0)
            for i in range(1, n + 1)
        ]
        _seed_tglc(cache, rows)
        tic_ids = [str(i) for i in range(1, n + 1)]
        res = ensure_metadata_bulk(tic_ids, cache, chunk_size=5000)
        assert res.created == n
        assert len(res.ready) == n
        assert cache.get_entry("1") is not None
        assert cache.get_entry(str(n)) is not None
        assert cache.get_entry("1")["fits_files"] == ["hlsp_tglc-1-s0001-1.fits"]

    def test_idempotent_second_run_zero_writes(self, cache, monkeypatch):
        _seed_tglc(cache, [
            ("600", 1, "/x/hlsp_tglc-a-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0),
            ("600", 2, "/x/hlsp_tglc-a-s0002-1.fits", 1800.0, 9.0, 1.0, 2.0),
        ])
        res1 = ensure_metadata_bulk(["600"], cache, refresh=True)
        assert res1.created == 1
        entry_after_1 = cache.get_entry("600")

        write_calls = []
        orig_write = cache.write_metadata_bulk

        def _spy(meta_rows, fits_rows):
            write_calls.append((list(meta_rows), list(fits_rows)))
            return orig_write(meta_rows, fits_rows)

        monkeypatch.setattr(cache, "write_metadata_bulk", _spy)

        res2 = ensure_metadata_bulk(["600"], cache, refresh=True)
        assert res2.created == 0
        assert res2.refreshed == 0
        assert res2.unchanged == 1
        assert write_calls == []  # 0 escriptures — the idempotence contract
        assert cache.get_entry("600") == entry_after_1

    def test_refresh_false_no_writes_at_all(self, cache, monkeypatch):
        """refresh=False no toca res: not even a single write for a stale row."""
        _seed_tglc(cache, [
            ("650", 1, "/x/hlsp_tglc-a-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0),
        ])
        ensure_metadata_bulk(["650"], cache)
        _seed_tglc(cache, [
            ("650", 2, "/x/hlsp_tglc-a-s0002-1.fits", 1800.0, 9.0, 1.0, 2.0),
        ])
        write_calls = []
        orig_write = cache.write_metadata_bulk

        def _spy(meta_rows, fits_rows):
            write_calls.append(1)
            return orig_write(meta_rows, fits_rows)

        monkeypatch.setattr(cache, "write_metadata_bulk", _spy)
        res = ensure_metadata_bulk(["650"], cache, refresh=False)
        assert res.unchanged == 1
        assert write_calls == []


# --------------------------------------------------------------------------- #
# Parity: ensure_metadata(t) vs ensure_metadata_bulk([t])
# --------------------------------------------------------------------------- #

def test_parity_single_vs_bulk_50_random_tics(tmp_path):
    """ensure_metadata(t) delegates to ensure_metadata_bulk([t], refresh=False)
    (§2.2), so this parity holds by construction — this test pins that the
    delegation is wired correctly for a spread of multi-sector TICs."""
    random.seed(42)
    cache_a = MetadataCache(str(tmp_path / "a.db"))
    cache_b = MetadataCache(str(tmp_path / "b.db"))

    tic_ids = [str(1_000_000 + i) for i in range(50)]
    rows = []
    for i, tic in enumerate(tic_ids):
        n_sectors = random.choice([1, 2, 3])
        for s in range(1, n_sectors + 1):
            rows.append((
                tic, s, f"/x/hlsp_tglc-{tic}-s{s:04d}-1.fits",
                1800.0, 9.0 + i, 10.0 + i, -10.0 - i,
            ))
    _seed_tglc(cache_a, rows)
    _seed_tglc(cache_b, rows)

    for tic in tic_ids:
        assert ensure_metadata(tic, mc=cache_a, tglc_index="unused-mc-wins")

    bulk_res = ensure_metadata_bulk(tic_ids, cache_b, refresh=False)
    assert set(bulk_res.ready) == set(tic_ids)

    for tic in tic_ids:
        assert cache_a.get_entry(tic) == cache_b.get_entry(tic), tic

    cache_a.close()
    cache_b.close()


# --------------------------------------------------------------------------- #
# Fault isolation (2026-09-26 review fix): one bad TIC, or one transient
# read/write blip, must never abort the rest of the chunk/lot.
# --------------------------------------------------------------------------- #

class TestFaultIsolation:
    def test_glob_fallback_exception_isolated_to_one_tic(self, cache, monkeypatch):
        """A TIC whose QLP/SPOC glob fallback raises is marked no_fits; every
        OTHER TIC in the same chunk (including another glob-fallback TIC)
        still gets indexed normally — no chunk-level fallback needed for
        this case (caught inline, per TIC)."""
        from exonit import metadata_index as mi

        _seed_tglc(cache, [
            ("100", 1, "/x/hlsp_tglc-a-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0),
        ])
        orig_fallback = mi._glob_fallback

        def _flaky_fallback(tic, ffi_cache, spoc_root):
            if tic == "999":
                raise RuntimeError("simulated glob failure")
            if tic == "200":
                return (
                    [5], 200.0,
                    ["hlsp_qlp_tess_ffi_s0005-1_tess_v01_llc.fits"],
                    [("hlsp_qlp_tess_ffi_s0005-1_tess_v01_llc.fits", "/x/qlp.fits")],
                )
            return orig_fallback(tic, ffi_cache, spoc_root)

        monkeypatch.setattr(mi, "_glob_fallback", _flaky_fallback)

        res = ensure_metadata_bulk(["100", "999", "200"], cache)

        assert set(res.ready) == {"100", "200"}
        assert res.no_fits == ["999"]
        assert res.created == 2  # 100 from tglc, 200 from the glob fallback
        assert cache.get_entry("999") is None
        assert cache.get_entry("200")["sectors"] == [5]

    def test_transient_bulk_write_failure_falls_back_per_tic(
            self, cache, monkeypatch):
        """The FIRST (whole-chunk) write_metadata_bulk call raises (a
        simulated transient PG blip); ensure_metadata_bulk must fall back to
        writing each TIC individually and still complete the chunk — no TIC
        lost, no exception propagated to the caller."""
        _seed_tglc(cache, [
            ("300", 1, "/x/hlsp_tglc-a-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0),
            ("400", 1, "/x/hlsp_tglc-b-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0),
            ("500", 1, "/x/hlsp_tglc-c-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0),
        ])
        orig_write = cache.write_metadata_bulk
        calls = {"n": 0}

        def _flaky_write(metadata_rows, fits_rows):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated transient PG write failure")
            return orig_write(metadata_rows, fits_rows)

        monkeypatch.setattr(cache, "write_metadata_bulk", _flaky_write)

        res = ensure_metadata_bulk(["300", "400", "500"], cache, chunk_size=5000)

        assert set(res.ready) == {"300", "400", "500"}
        assert res.created == 3
        assert res.no_fits == []
        for tic in ("300", "400", "500"):
            assert cache.get_entry(tic) is not None
        # 1 failed bulk attempt + 3 successful per-TIC retries.
        assert calls["n"] == 4

    def test_read_failure_falls_back_per_tic(self, cache, monkeypatch):
        """A bulk tglc_lookup_bulk failure (read side, not write) on the
        first call also triggers the per-TIC chunk fallback."""
        _seed_tglc(cache, [
            ("700", 1, "/x/hlsp_tglc-a-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0),
            ("701", 1, "/x/hlsp_tglc-b-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0),
        ])
        orig_lookup = cache.tglc_lookup_bulk
        calls = {"n": 0}

        def _flaky_lookup(tic_ids, chunk_size=5000):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated transient PG read failure")
            return orig_lookup(tic_ids, chunk_size=chunk_size)

        monkeypatch.setattr(cache, "tglc_lookup_bulk", _flaky_lookup)

        res = ensure_metadata_bulk(["700", "701"], cache)

        assert set(res.ready) == {"700", "701"}
        assert res.created == 2

    def test_persistent_write_failure_isolated_to_one_tic(
            self, cache, monkeypatch):
        """A write failure that persists even for the per-TIC retry (a
        genuinely bad row, not a transient blip) demotes ONLY that TIC to
        no_fits; the rest of the chunk still completes, and the counts stay
        consistent (created == len(ready), no double-counting)."""
        _seed_tglc(cache, [
            ("600", 1, "/x/hlsp_tglc-a-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0),
            ("601", 1, "/x/hlsp_tglc-b-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0),
        ])
        orig_write = cache.write_metadata_bulk

        def _flaky_write(metadata_rows, fits_rows):
            # Fail the whole-chunk attempt (>1 row) AND any single-row
            # attempt whose only row is TIC 601 — i.e. 601 is persistently
            # broken, not just unlucky on the first try.
            if len(metadata_rows) > 1 or any(r[0] == "601" for r in metadata_rows):
                raise RuntimeError("simulated persistent write failure for 601")
            return orig_write(metadata_rows, fits_rows)

        monkeypatch.setattr(cache, "write_metadata_bulk", _flaky_write)

        res = ensure_metadata_bulk(["600", "601"], cache)

        assert res.ready == ["600"]
        assert res.no_fits == ["601"]
        assert res.created == 1
        assert cache.get_entry("600") is not None
        assert cache.get_entry("601") is None
        # Consistency: every input TIC accounted for exactly once.
        assert set(res.ready) | set(res.no_fits) == {"600", "601"}
        assert len(res.ready) + len(res.no_fits) == 2


# --------------------------------------------------------------------------- #
# PG real: one chunk = one transaction (EXONIT_TEST_META_PG_URL only)
# --------------------------------------------------------------------------- #

class TestEnsureMetadataBulkRealPg:
    """End-to-end against a real throwaway PG (exonit_meta_test) — skipped
    when EXONIT_TEST_META_PG_URL is unset/unreachable, per
    tests/_metadata_pg_test_db.py's guard. NEVER touches production
    exonit_meta."""

    def _setup(self, dsn, partitioned: bool):
        import psycopg
        with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
            c.execute("DROP TABLE IF EXISTS tglc CASCADE")
            c.execute("DROP TABLE IF EXISTS metadata CASCADE")
            c.execute("DROP TABLE IF EXISTS fits_paths CASCADE")
            if partitioned:
                c.execute(
                    "CREATE TABLE tglc (tic_id TEXT NOT NULL, sector BIGINT "
                    "NOT NULL, path TEXT NOT NULL, exptime DOUBLE PRECISION, "
                    "tmag DOUBLE PRECISION, ra DOUBLE PRECISION, "
                    "dec DOUBLE PRECISION, PRIMARY KEY (tic_id, sector)) "
                    "PARTITION BY LIST (sector)")
                c.execute("CREATE TABLE tglc_s0001 PARTITION OF tglc "
                         "FOR VALUES IN (1)")
            else:
                c.execute(
                    "CREATE TABLE tglc (tic_id TEXT NOT NULL, sector BIGINT "
                    "NOT NULL, path TEXT NOT NULL, exptime DOUBLE PRECISION, "
                    "tmag DOUBLE PRECISION, ra DOUBLE PRECISION, "
                    "dec DOUBLE PRECISION, PRIMARY KEY (tic_id, sector))")
            c.execute(
                "CREATE TABLE metadata (tic_id TEXT PRIMARY KEY, "
                "sectors TEXT NOT NULL, exptime DOUBLE PRECISION NOT NULL, "
                "fits_files TEXT NOT NULL, stellar_radius DOUBLE PRECISION, "
                "teff DOUBLE PRECISION, logg DOUBLE PRECISION, "
                "tmag DOUBLE PRECISION, ra DOUBLE PRECISION, "
                "dec DOUBLE PRECISION, "
                "updated_at TIMESTAMPTZ DEFAULT NOW())")
            c.execute(
                "CREATE TABLE fits_paths (filename TEXT PRIMARY KEY, "
                "full_path TEXT NOT NULL)")

    def _teardown(self, dsn):
        import psycopg
        with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
            c.execute("DROP TABLE IF EXISTS tglc CASCADE")
            c.execute("DROP TABLE IF EXISTS metadata CASCADE")
            c.execute("DROP TABLE IF EXISTS fits_paths CASCADE")

    def _run(self, monkeypatch, partitioned: bool):
        from exonit.metadata_backend import PgBackend
        from tests._metadata_pg_test_db import (
            pg_meta_test_available, resolve_test_meta_dsn,
        )

        dsn = resolve_test_meta_dsn()
        if not pg_meta_test_available(dsn):
            pytest.skip("EXONIT_TEST_META_PG_URL (throwaway PG) unset or not reachable")

        self._setup(dsn, partitioned)
        monkeypatch.setenv("EXONIT_METADATA_DB_URL", dsn)
        try:
            cache = MetadataCache("unused-path-pg-wins.db")
            assert cache._is_pg
            cache.tglc_add_rows_bulk([
                ("900", 1, "/x/hlsp_tglc-a-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0),
            ])

            # Spy on PgBackend.transaction itself (NOT the Backend base
            # class): PgBackend overrides transaction with its own
            # @contextmanager method, so patching the base would never be
            # reached via normal method resolution.
            tx_calls = []
            orig_transaction = PgBackend.transaction

            def _spy_transaction(self):
                tx_calls.append(1)
                return orig_transaction(self)

            monkeypatch.setattr(PgBackend, "transaction", _spy_transaction)

            res = ensure_metadata_bulk(["900"], cache, chunk_size=5000)
            assert res.created == 1
            assert len(tx_calls) == 1  # ONE transaction for this one chunk

            entry = cache.get_entry("900")
            assert entry["sectors"] == [1]
            assert entry["fits_files"] == ["hlsp_tglc-a-s0001-1.fits"]
            cache.close()
        finally:
            self._teardown(dsn)

    def test_one_chunk_one_transaction_flat_tglc(self, monkeypatch):
        self._run(monkeypatch, partitioned=False)

    def test_one_chunk_one_transaction_partitioned_tglc(self, monkeypatch):
        self._run(monkeypatch, partitioned=True)
