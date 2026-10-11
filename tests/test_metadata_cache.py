"""Tests for MetadataCache — MAST metadata local SQLite cache."""
import os
import pytest
from exonit.metadata_cache import MetadataCache


def test_create_cache(tmp_path):
    db_file = tmp_path / "cache.db"
    cache = MetadataCache(db_path=str(db_file))
    cache.close()
    assert db_file.exists()


def test_add_and_get_entry(tmp_path):
    db_file = tmp_path / "cache.db"
    cache = MetadataCache(db_path=str(db_file))

    cache.add_entry(
        tic_id="TIC123456",
        sectors=[1, 2, 13],
        exptime=120.0,
        fits_files=["file_s1.fits", "file_s2.fits"],
    )
    entry = cache.get_entry("TIC123456")
    cache.close()

    assert entry is not None
    assert entry["tic_id"] == "TIC123456"
    assert entry["sectors"] == [1, 2, 13]
    assert entry["exptime"] == pytest.approx(120.0)
    assert entry["fits_files"] == ["file_s1.fits", "file_s2.fits"]


def test_get_missing_entry(tmp_path):
    db_file = tmp_path / "cache.db"
    cache = MetadataCache(db_path=str(db_file))
    result = cache.get_entry("TIC_NONEXISTENT")
    cache.close()
    assert result is None


def test_get_all_tic_ids(tmp_path):
    db_file = tmp_path / "cache.db"
    cache = MetadataCache(db_path=str(db_file))

    cache.add_entry("TIC_A", sectors=[1], exptime=20.0, fits_files=["a.fits"])
    cache.add_entry("TIC_B", sectors=[2, 3], exptime=120.0, fits_files=["b1.fits", "b2.fits"])

    ids = cache.get_all_tic_ids()
    cache.close()

    assert set(ids) == {"TIC_A", "TIC_B"}


def test_stats(tmp_path):
    db_file = tmp_path / "cache.db"
    cache = MetadataCache(db_path=str(db_file))

    cache.add_entry("TIC_X", sectors=[1], exptime=20.0, fits_files=["x1.fits", "x2.fits"])
    cache.add_entry("TIC_Y", sectors=[4], exptime=60.0, fits_files=["y1.fits"])

    s = cache.stats()
    cache.close()

    assert s["total_stars"] == 2
    assert s["total_fits_files"] == 3


def test_build_from_nfs(tmp_path):
    from exonit.metadata_cache import MetadataCache, build_from_fits_listing

    fits_listing = [
        "tess2018206045859-s0001-0000000261136679-0120-s_lc.fits",
        "tess2018234235059-s0002-0000000261136679-0120-s_lc.fits",
        "tess2020186164531-s0027-0000000261136679-0120-s_lc.fits",
        "tess2018206045859-s0001-0000000149603524-0120-s_lc.fits",
    ]

    db_path = str(tmp_path / "meta.db")
    cache = MetadataCache(db_path)
    stats = build_from_fits_listing(cache, fits_listing)

    assert stats["stars_added"] == 2
    assert stats["fits_parsed"] == 4

    entry = cache.get_entry("261136679")
    assert entry is not None
    assert sorted(entry["sectors"]) == [1, 2, 27]
    assert len(entry["fits_files"]) == 3
    assert entry["exptime"] == 120.0

    entry2 = cache.get_entry("149603524")
    assert entry2 is not None
    assert entry2["sectors"] == [1]
    cache.close()


def test_fits_path_index(tmp_path):
    from exonit.metadata_cache import MetadataCache
    cache = MetadataCache(str(tmp_path / "meta.db"))

    cache.add_fits_path(
        "tess2018206045859-s0001-0000000261136679-0120-s_lc.fits",
        "/data/tess_cache/sector01/tess2018206045859-s0001-0000000261136679-0120-s_lc.fits",
    )
    path = cache.get_fits_path("tess2018206045859-s0001-0000000261136679-0120-s_lc.fits")
    assert path == "/data/tess_cache/sector01/tess2018206045859-s0001-0000000261136679-0120-s_lc.fits"
    assert cache.get_fits_path("nonexistent.fits") is None
    cache.close()


def test_cached_tic_ids_basic(tmp_path):
    """cached_tic_ids returns only the ids that are actually cached."""
    cache = MetadataCache(str(tmp_path / "cache.db"))
    cache.add_entry("TIC_A", sectors=[1], exptime=120.0, fits_files=["a.fits"])
    cache.add_entry("TIC_B", sectors=[2], exptime=120.0, fits_files=["b.fits"])

    result = cache.cached_tic_ids(["TIC_A", "TIC_B", "TIC_C"])
    cache.close()

    assert result == {"TIC_A", "TIC_B"}


def test_cached_tic_ids_none_match(tmp_path):
    """None of the queried ids are cached -> empty set, not an error."""
    cache = MetadataCache(str(tmp_path / "cache.db"))
    cache.add_entry("TIC_A", sectors=[1], exptime=120.0, fits_files=["a.fits"])

    result = cache.cached_tic_ids(["TIC_X", "TIC_Y"])
    cache.close()

    assert result == set()


def test_cached_tic_ids_empty_input(tmp_path):
    """Empty input -> empty output, no query issued against an empty IN()."""
    cache = MetadataCache(str(tmp_path / "cache.db"))
    cache.add_entry("TIC_A", sectors=[1], exptime=120.0, fits_files=["a.fits"])

    assert cache.cached_tic_ids([]) == set()
    assert cache.cached_tic_ids(set()) == set()
    cache.close()


def test_cached_tic_ids_chunk_boundary(tmp_path):
    """Ids on either side of a chunk boundary are both found correctly."""
    cache = MetadataCache(str(tmp_path / "cache.db"))
    cache.add_entry("A", sectors=[1], exptime=120.0, fits_files=[])
    cache.add_entry("C", sectors=[1], exptime=120.0, fits_files=[])
    cache.add_entry("E", sectors=[1], exptime=120.0, fits_files=[])

    # chunk_size=2 over 5 ids -> chunks ["A","B"], ["C","D"], ["E"];
    # exercises the IN(...) query landing in a chunk of size 1 too.
    result = cache.cached_tic_ids(["A", "B", "C", "D", "E"], chunk_size=2)
    cache.close()

    assert result == {"A", "C", "E"}


def test_cached_tic_ids_over_5000_forces_multiple_chunks(tmp_path):
    """>5000 ids (the default chunk_size) must still return every cached hit,
    regardless of which chunk it lands in — this is the case that used to be
    one locked SELECT * + json.loads per id in /api/health/work (18-30 s at
    ~300k+ rows)."""
    cache = MetadataCache(str(tmp_path / "cache.db"))
    cached_ids = {"CACHED_1", "CACHED_2", "CACHED_3"}
    for tid in cached_ids:
        cache.add_entry(tid, sectors=[1], exptime=120.0, fits_files=[])

    query_ids = {f"NOPE_{i}" for i in range(6000)} | cached_ids
    assert len(query_ids) > 5000  # sanity: forces >1 chunk at the default size

    result = cache.cached_tic_ids(query_ids)
    cache.close()

    assert result == cached_ids


def test_fits_path_bulk(tmp_path):
    from exonit.metadata_cache import MetadataCache
    cache = MetadataCache(str(tmp_path / "meta.db"))
    entries = [
        ("a.fits", "/path/a.fits"),
        ("b.fits", "/path/b.fits"),
        ("c.fits", "/path/c.fits"),
    ]
    cache.add_fits_paths_bulk(entries)
    assert cache.get_fits_path("a.fits") == "/path/a.fits"
    assert cache.get_fits_path("b.fits") == "/path/b.fits"
    assert cache.get_fits_path("c.fits") == "/path/c.fits"
    cache.close()


# --------------------------------------------------------------------------- #
# Nearby stars / fits_blobs — regression tests for the metadata_backend.py
# seam refactor (2026-07-20, docs/superpowers/specs/2026-07-20-tglc-index-
# to-pg.md F1): these methods now go through Backend.transaction()/upsert()
# instead of a raw sqlite3 connection.
# --------------------------------------------------------------------------- #

def test_fits_blobs_bulk_roundtrip(tmp_path):
    from exonit.metadata_cache import MetadataCache
    cache = MetadataCache(str(tmp_path / "meta.db"))
    cache.add_fits_blobs_bulk([
        ("a.fits", "containers/run1.tar", 0, 1000),
        ("b.fits", "containers/run1.tar", 1000, 2000),
    ])
    assert cache.get_fits_blob("a.fits") == ("containers/run1.tar", 0, 1000)
    assert cache.get_fits_blob("b.fits") == ("containers/run1.tar", 1000, 2000)
    assert cache.get_fits_blob("nonexistent.fits") is None
    cache.close()


def test_fits_blobs_bulk_replace_semantics(tmp_path):
    """A second bulk insert for the same filename REPLACES the row (INSERT
    OR REPLACE), matching historical MetadataCache behaviour."""
    from exonit.metadata_cache import MetadataCache
    cache = MetadataCache(str(tmp_path / "meta.db"))
    cache.add_fits_blobs_bulk([("a.fits", "containers/run1.tar", 0, 1000)])
    cache.add_fits_blobs_bulk([("a.fits", "containers/run2.tar", 500, 999)])
    assert cache.get_fits_blob("a.fits") == ("containers/run2.tar", 500, 999)
    cache.close()


def test_add_nearby_stars_roundtrip(tmp_path):
    from exonit.metadata_cache import MetadataCache
    cache = MetadataCache(str(tmp_path / "meta.db"))
    stars = [
        {"ID": "200", "ra": 10.0, "dec": 20.0, "mass": 1.0, "rad": 1.1,
         "Teff": 5800.0, "Tmag": 9.0, "plx": 5.0},
        {"ID": "201", "ra": 11.0, "dec": 21.0, "mass": 0.9, "rad": 0.8,
         "Teff": 5000.0, "Tmag": 10.0, "plx": 4.5},
    ]
    cache.add_nearby_stars("100", stars, gal_b=12.5)
    result = cache.get_nearby_stars("100")
    assert result is not None
    got_stars, gal_b = result
    assert gal_b == pytest.approx(12.5)
    assert {s["ID"] for s in got_stars} == {"200", "201"}
    assert cache.count_nearby_cached() == 1
    cache.close()


def test_add_nearby_stars_replaces_old_set(tmp_path):
    """A second add_nearby_stars call for the same TIC DELETEs the old set
    first (atomic DELETE + INSERT via Backend.transaction())."""
    from exonit.metadata_cache import MetadataCache
    cache = MetadataCache(str(tmp_path / "meta.db"))
    cache.add_nearby_stars("100", [{"ID": "1"}], gal_b=1.0)
    cache.add_nearby_stars("100", [{"ID": "2"}, {"ID": "3"}], gal_b=2.0)
    got_stars, gal_b = cache.get_nearby_stars("100")
    assert {s["ID"] for s in got_stars} == {"2", "3"}
    assert gal_b == pytest.approx(2.0)
    cache.close()


def test_get_nearby_stars_missing_returns_none(tmp_path):
    from exonit.metadata_cache import MetadataCache
    cache = MetadataCache(str(tmp_path / "meta.db"))
    assert cache.get_nearby_stars("NOPE") is None
    cache.close()


# --------------------------------------------------------------------------- #
# TGLC helpers (D3 of docs/superpowers/specs/2026-07-20-tglc-index-to-pg.md)
# --------------------------------------------------------------------------- #

def _make_tglc_table(db_path, rows):
    """rows: list of (tic_id, sector, path, exptime, tmag, ra, dec) tuples,
    inserted in list order (SQLite rowid order == list order)."""
    import sqlite3
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE tglc (tic_id TEXT, sector INTEGER, path TEXT, "
        "exptime REAL, tmag REAL, ra REAL, dec REAL, "
        "PRIMARY KEY(tic_id, sector))"
    )
    con.executemany(
        "INSERT INTO tglc VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


def test_tglc_lookup_returns_rows_for_tic(tmp_path):
    from exonit.metadata_cache import MetadataCache
    db = tmp_path / "meta.db"
    _make_tglc_table(str(db), [
        ("100", 1, "/cache/a.fits", 1800.0, 10.0, 1.0, 2.0),
        ("100", 2, "/cache/b.fits", 1800.0, 10.0, 1.0, 2.0),
        ("200", 1, "/cache/c.fits", 1800.0, 11.0, 3.0, 4.0),
    ])
    cache = MetadataCache(str(db))
    rows = cache.tglc_lookup("100")
    assert len(rows) == 2
    assert {r["sector"] for r in rows} == {1, 2}
    assert rows[0]["path"] in {"/cache/a.fits", "/cache/b.fits"}
    cache.close()


def test_tglc_lookup_missing_tic_returns_empty_list(tmp_path):
    from exonit.metadata_cache import MetadataCache
    db = tmp_path / "meta.db"
    _make_tglc_table(str(db), [("100", 1, "/cache/a.fits", 1800.0, 10.0, 1.0, 2.0)])
    cache = MetadataCache(str(db))
    assert cache.tglc_lookup("999") == []
    cache.close()


def test_tglc_lookup_missing_table_returns_empty_list(tmp_path):
    from exonit.metadata_cache import MetadataCache
    cache = MetadataCache(str(tmp_path / "meta.db"))  # no tglc table created
    assert cache.tglc_lookup("100") == []
    cache.close()


def test_tglc_tic_ids_ordered_by_rowid(tmp_path):
    """Extension order must match SQLite rowid ASC (insertion order) — the
    same contract exonit/discovery_state.py::load_extended_canonical relies
    on."""
    from exonit.metadata_cache import MetadataCache
    db = tmp_path / "meta.db"
    _make_tglc_table(str(db), [
        ("1", 1, "/x.fits", 1800, 10, 0, 0),
        ("99", 1, "/x.fits", 1800, 10, 0, 0),
        ("2", 1, "/x.fits", 1800, 10, 0, 0),
        ("50", 1, "/x.fits", 1800, 10, 0, 0),
    ])
    cache = MetadataCache(str(db))
    assert cache.tglc_tic_ids() == [1, 99, 2, 50]
    cache.close()


def test_tglc_tic_ids_sector_filter(tmp_path):
    from exonit.metadata_cache import MetadataCache
    db = tmp_path / "meta.db"
    _make_tglc_table(str(db), [
        ("10", 1, "/x.fits", 1800, 10, 0, 0),
        ("20", 2, "/x.fits", 1800, 10, 0, 0),
        ("30", 1, "/x.fits", 1800, 10, 0, 0),
        ("40", 2, "/x.fits", 1800, 10, 0, 0),
    ])
    cache = MetadataCache(str(db))
    assert cache.tglc_tic_ids(sector=2) == [20, 40]
    assert cache.tglc_tic_ids(sector=1) == [10, 30]
    cache.close()


def test_tglc_tic_ids_missing_table_returns_empty_list(tmp_path):
    from exonit.metadata_cache import MetadataCache
    cache = MetadataCache(str(tmp_path / "meta.db"))
    assert cache.tglc_tic_ids() == []
    cache.close()


def test_tglc_tic_ids_page_ordered_and_bounded(tmp_path):
    """S1 (docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md §4.1.4):
    tglc_tic_ids_page returns (tic_id, seq) pairs strictly after after_seq,
    ascending, capped at limit — SQLite's rowid stands in for PG's seq."""
    from exonit.metadata_cache import MetadataCache
    db = tmp_path / "meta.db"
    _make_tglc_table(str(db), [
        ("1", 1, "/x.fits", 1800, 10, 0, 0),
        ("2", 1, "/x.fits", 1800, 10, 0, 0),
        ("3", 1, "/x.fits", 1800, 10, 0, 0),
        ("4", 1, "/x.fits", 1800, 10, 0, 0),
        ("5", 2, "/x.fits", 1800, 10, 0, 0),  # different sector, must not leak
    ])
    cache = MetadataCache(str(db))
    page1 = cache.tglc_tic_ids_page(1, after_seq=0, limit=2)
    assert [tic for tic, _seq in page1] == [1, 2]
    last_seq = page1[-1][1]
    page2 = cache.tglc_tic_ids_page(1, after_seq=last_seq, limit=2)
    assert [tic for tic, _seq in page2] == [3, 4]
    page3 = cache.tglc_tic_ids_page(1, after_seq=page2[-1][1], limit=2)
    assert page3 == []  # end of sector 1's rows
    cache.close()


def test_tglc_tic_ids_page_missing_table_returns_empty_list(tmp_path):
    from exonit.metadata_cache import MetadataCache
    cache = MetadataCache(str(tmp_path / "meta.db"))
    assert cache.tglc_tic_ids_page(1, after_seq=0, limit=10) == []
    cache.close()


def test_tglc_seq_at_offset(tmp_path):
    from exonit.metadata_cache import MetadataCache
    db = tmp_path / "meta.db"
    _make_tglc_table(str(db), [
        ("10", 1, "/x.fits", 1800, 10, 0, 0),
        ("20", 1, "/x.fits", 1800, 10, 0, 0),
        ("30", 1, "/x.fits", 1800, 10, 0, 0),
    ])
    cache = MetadataCache(str(db))
    seq0 = cache.tglc_seq_at_offset(1, 0)
    seq1 = cache.tglc_seq_at_offset(1, 1)
    seq2 = cache.tglc_seq_at_offset(1, 2)
    assert seq0 < seq1 < seq2
    # A page starting right after seq1 should yield exactly the 3rd row.
    page = cache.tglc_tic_ids_page(1, after_seq=seq1, limit=10)
    assert [tic for tic, _seq in page] == [30]
    assert cache.tglc_seq_at_offset(1, 99) is None  # out of range
    cache.close()


def test_tglc_seq_for_tics(tmp_path):
    from exonit.metadata_cache import MetadataCache
    db = tmp_path / "meta.db"
    _make_tglc_table(str(db), [
        ("10", 1, "/x.fits", 1800, 10, 0, 0),
        ("20", 1, "/x.fits", 1800, 10, 0, 0),
        ("30", 2, "/x.fits", 1800, 10, 0, 0),  # different sector
    ])
    cache = MetadataCache(str(db))
    out = cache.tglc_seq_for_tics(1, [10, 20, 30, 999])
    assert set(out.keys()) == {10, 20}  # 30 is sector 2, 999 absent
    assert out[10] < out[20]
    cache.close()


def test_tglc_seq_for_tics_missing_table_returns_empty_dict(tmp_path):
    from exonit.metadata_cache import MetadataCache
    cache = MetadataCache(str(tmp_path / "meta.db"))
    assert cache.tglc_seq_for_tics(1, [10, 20]) == {}
    cache.close()


def test_tglc_sector_count(tmp_path):
    from exonit.metadata_cache import MetadataCache
    db = tmp_path / "meta.db"
    _make_tglc_table(str(db), [
        ("10", 1, "/x.fits", 1800, 10, 0, 0),
        ("20", 2, "/x.fits", 1800, 10, 0, 0),
        ("30", 1, "/x.fits", 1800, 10, 0, 0),
    ])
    cache = MetadataCache(str(db))
    assert cache.tglc_sector_count(1) == 2
    assert cache.tglc_sector_count(2) == 1
    assert cache.tglc_sector_count(3) == 0
    assert cache.tglc_sector_count() == 3  # None -> every row
    cache.close()


def test_tglc_sector_count_missing_table_returns_none(tmp_path):
    """None (not 0) on error, so callers can distinguish "0 rows" from
    "could not determine" — server.py's cache-invalidation check depends on
    this exact contract (see MetadataCache.tglc_sector_count's docstring)."""
    from exonit.metadata_cache import MetadataCache
    cache = MetadataCache(str(tmp_path / "meta.db"))  # no tglc table
    assert cache.tglc_sector_count(1) is None
    assert cache.tglc_sector_count() is None
    cache.close()


def test_tglc_distinct_tic_count(tmp_path):
    """A TIC observed under two Gaia IDs in the same sector collapses to one
    row via the tglc PK — distinct count should reflect the actual number of
    stars, which here just equals row count since ids are pre-deduplicated."""
    from exonit.metadata_cache import MetadataCache
    db = tmp_path / "meta.db"
    _make_tglc_table(str(db), [
        ("10", 1, "/x.fits", 1800, 10, 0, 0),
        ("10", 2, "/x.fits", 1800, 10, 0, 0),  # same TIC, different sector
        ("20", 1, "/x.fits", 1800, 10, 0, 0),
    ])
    cache = MetadataCache(str(db))
    assert cache.tglc_distinct_tic_count() == 2  # TICs 10 and 20
    assert cache.tglc_distinct_tic_count(sector=1) == 2
    assert cache.tglc_distinct_tic_count(sector=2) == 1
    cache.close()


def test_tglc_paths_for_returns_only_matching_paths(tmp_path):
    from exonit.metadata_cache import MetadataCache
    db = tmp_path / "meta.db"
    _make_tglc_table(str(db), [
        ("10", 1, "/cache/a.fits", 1800, 10, 0, 0),
        ("20", 1, "/cache/b.fits", 1800, 10, 0, 0),
    ])
    cache = MetadataCache(str(db))
    found = cache.tglc_paths_for(["/cache/a.fits", "/cache/nope.fits"])
    assert found == {"/cache/a.fits"}
    cache.close()


def test_tglc_paths_for_empty_input_no_query(tmp_path):
    from exonit.metadata_cache import MetadataCache
    cache = MetadataCache(str(tmp_path / "meta.db"))
    assert cache.tglc_paths_for([]) == set()
    cache.close()


def test_tglc_paths_for_chunk_boundary(tmp_path):
    """Paths on either side of a chunk boundary are both found."""
    from exonit.metadata_cache import MetadataCache
    db = tmp_path / "meta.db"
    rows = [(str(i), 1, f"/cache/{i}.fits", 1800, 10, 0, 0) for i in range(5)]
    _make_tglc_table(str(db), rows)
    cache = MetadataCache(str(db))
    paths = [f"/cache/{i}.fits" for i in range(5)]
    found = cache.tglc_paths_for(paths, chunk_size=2)
    assert found == set(paths)
    cache.close()


def test_tglc_add_rows_bulk_insert_and_replace(tmp_path):
    from exonit.metadata_cache import MetadataCache
    db = tmp_path / "meta.db"
    _make_tglc_table(str(db), [])
    cache = MetadataCache(str(db))
    cache.tglc_add_rows_bulk([
        ("100", 1, "/cache/a.fits", 1800.0, 10.0, 1.0, 2.0),
        ("200", 1, "/cache/b.fits", 1800.0, 11.0, 3.0, 4.0),
    ])
    assert cache.tglc_sector_count(1) == 2
    rows = cache.tglc_lookup("100")
    assert rows[0]["path"] == "/cache/a.fits"

    # REPLACE semantics: re-inserting the same PK overwrites the row.
    cache.tglc_add_rows_bulk([
        ("100", 1, "/cache/a_v2.fits", 1800.0, 10.0, 1.0, 2.0),
    ])
    rows = cache.tglc_lookup("100")
    assert len(rows) == 1
    assert rows[0]["path"] == "/cache/a_v2.fits"
    cache.close()


def test_ensure_tglc_partition_is_noop_on_sqlite(tmp_path):
    from exonit.metadata_cache import MetadataCache
    cache = MetadataCache(str(tmp_path / "meta.db"))
    cache.ensure_tglc_partition(23)  # must not raise
    cache.close()


# --------------------------------------------------------------------------- #
# EXONIT_METADATA_DB_URL env-var override (D2 of the spec)
# --------------------------------------------------------------------------- #

def test_env_var_unset_stays_on_sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("EXONIT_METADATA_DB_URL", raising=False)
    cache = MetadataCache(str(tmp_path / "meta.db"))
    assert cache._backend.dialect.name == "sqlite"
    assert not cache._is_pg
    cache.close()


def test_env_var_set_to_pg_dsn_overrides_sqlite_path_argument(tmp_path, monkeypatch):
    """EXONIT_METADATA_DB_URL, when set, wins over whatever db_path string
    the caller passes — every existing MetadataCache(cfg.metadata_cache_path)
    call site across the codebase flips to PG together without changing
    (D2/D3 of the spec). Proven here by asserting the PG dialect/pool was
    actually selected (and no sqlite fallback file was created) — NOT by
    forcing an actual network round-trip (psycopg_pool's ``open=True``
    starts connecting in a background thread and does not raise
    synchronously for an unreachable host; only a real query times out)."""
    monkeypatch.setenv(
        "EXONIT_METADATA_DB_URL",
        "postgresql://nobody@127.0.0.1:1/does_not_exist_test")
    sqlite_path = str(tmp_path / "meta.db")
    cache = MetadataCache(sqlite_path)
    try:
        assert cache._is_pg
        assert cache._backend.dialect.name == "postgres"
        # The sqlite fallback file must NOT have been created — proof the PG
        # path was actually taken instead of a silent sqlite fallback.
        assert not os.path.exists(sqlite_path)
    finally:
        cache.close()


def test_env_var_non_pg_value_falls_back_to_sqlite_path(tmp_path, monkeypatch):
    """A garbage (non-postgresql://) EXONIT_METADATA_DB_URL is treated the
    same as unset — is_pg_url() gates the override, so this can't
    accidentally shadow the sqlite path with an invalid value."""
    monkeypatch.setenv("EXONIT_METADATA_DB_URL", "not-a-dsn")
    cache = MetadataCache(str(tmp_path / "meta.db"))
    assert cache._backend.dialect.name == "sqlite"
    cache.close()


# --------------------------------------------------------------------------- #
# indexed_basenames — exact port of scripts/discovery_download.py's
# _indexed_basenames union semantics (tglc ⨆ fits_blobs). Mirrors the
# dedicated test class in tests/test_discovery_download.py.
# --------------------------------------------------------------------------- #

def test_indexed_basenames_tglc_only(tmp_path):
    from exonit.metadata_cache import MetadataCache
    db = tmp_path / "meta.db"
    _make_tglc_table(str(db), [
        ("1", 23, "/cache/s0023/a/hlsp_tglc_..._gaiaid-100-s0023-cam1-ccd1_tess_v1_llc.fits",
         1800, 10, 0, 0),
        ("2", 23, "/cache/s0023/b/hlsp_tglc_..._gaiaid-200-s0023-cam1-ccd1_tess_v1_llc.fits",
         1800, 10, 0, 0),
    ])
    cache = MetadataCache(str(db))
    names = cache.indexed_basenames(23)
    assert len(names) == 2
    cache.close()


def test_indexed_basenames_union_includes_fits_blobs_not_in_tglc(tmp_path):
    """Basenames in fits_blobs but absent from tglc (the duplicate-Gaia-ID
    collapse case) must be included in the union — see the PK collision note
    in scripts/discovery_download.py::_indexed_basenames."""
    from exonit.metadata_cache import MetadataCache
    db = tmp_path / "meta.db"
    winner = "hlsp_tglc_tess_ffi_gaiaid-winner-s0023-cam1-ccd1_tess_v1_llc.fits"
    loser = "hlsp_tglc_tess_ffi_gaiaid-loser-s0023-cam1-ccd1_tess_v1_llc.fits"
    _make_tglc_table(str(db), [
        ("42", 23, f"/cache/s0023/a/{winner}", 1800, 10, 0, 0),
    ])
    cache = MetadataCache(str(db))
    cache.add_fits_blobs_bulk([
        (winner, "containers/s0023/dl_1.tar", 0, 10000),
        (loser, "containers/s0023/dl_1.tar", 10000, 20000),
    ])
    names = cache.indexed_basenames(23)
    assert winner in names
    assert loser in names
    cache.close()


def test_indexed_basenames_no_fits_blobs_table_is_a_guarded_fallback(tmp_path):
    """fits_blobs may be absent entirely (older deployments) — falls back to
    the tglc-only set rather than raising."""
    from exonit.metadata_cache import MetadataCache
    db = tmp_path / "meta.db"
    _make_tglc_table(str(db), [
        ("1", 23, "/cache/a-s0023-x.fits", 1800, 10, 0, 0),
    ])
    # Drop the fits_blobs table MetadataCache's own __init__ would have
    # created, to simulate a deployment that predates it.
    import sqlite3
    con = sqlite3.connect(str(db))
    con.execute("DROP TABLE IF EXISTS fits_blobs")
    con.commit()
    con.close()
    cache = MetadataCache(str(db))
    # __init__ recreates fits_blobs (CREATE TABLE IF NOT EXISTS) — drop it
    # again post-construction to actually exercise the guard.
    con = sqlite3.connect(str(db))
    con.execute("DROP TABLE fits_blobs")
    con.commit()
    con.close()
    names = cache.indexed_basenames(23)
    assert names == {"a-s0023-x.fits"}
    cache.close()


# ------------------------------------------------------------------ #
# tglc_basenames_missing_blobs (2026-07-21 pack-gap repair helper —
# scripts/repair_missing_blobs.py's read side)
# ------------------------------------------------------------------ #

def test_tglc_basenames_missing_blobs_returns_only_the_gap(tmp_path):
    """Basenames present in tglc AND fits_blobs are excluded; only the ones
    with NO fits_blobs row come back, mapped to their recorded tglc.path."""
    from exonit.metadata_cache import MetadataCache

    db = tmp_path / "meta.db"
    packed = "hlsp_tglc_tess_ffi_gaiaid-packed-s0007-cam1-ccd1_tess_v1_llc.fits"
    gap = "hlsp_tglc_tess_ffi_gaiaid-gap-s0007-cam1-ccd1_tess_v1_llc.fits"
    _make_tglc_table(str(db), [
        ("1", 7, f"/data/tglc_cache/s0007/a/{packed}", 1800, 10, 0, 0),
        ("2", 7, f"/data/tglc_cache/s0007/a/{gap}", 1800, 10, 0, 0),
    ])
    cache = MetadataCache(str(db))
    cache.add_fits_blobs_bulk([
        (packed, "containers/s0007/dl_1.tar", 0, 10000),
    ])
    missing = cache.tglc_basenames_missing_blobs(7)
    assert missing == {gap: f"/data/tglc_cache/s0007/a/{gap}"}
    cache.close()


def test_tglc_basenames_missing_blobs_no_gap_returns_empty(tmp_path):
    """Every tglc row for the sector has a matching fits_blobs row -> no gap."""
    from exonit.metadata_cache import MetadataCache

    db = tmp_path / "meta.db"
    name = "hlsp_tglc_tess_ffi_gaiaid-1-s0007-cam1-ccd1_tess_v1_llc.fits"
    _make_tglc_table(str(db), [("1", 7, f"/x/{name}", 1800, 10, 0, 0)])
    cache = MetadataCache(str(db))
    cache.add_fits_blobs_bulk([(name, "containers/s0007/dl_1.tar", 0, 10000)])
    assert cache.tglc_basenames_missing_blobs(7) == {}
    cache.close()


def test_tglc_basenames_missing_blobs_sector_scoped(tmp_path):
    """A gap in a DIFFERENT sector must not leak into this sector's result."""
    from exonit.metadata_cache import MetadataCache

    db = tmp_path / "meta.db"
    other_gap = "hlsp_tglc_tess_ffi_gaiaid-1-s0001-cam1-ccd1_tess_v1_llc.fits"
    _make_tglc_table(str(db), [("1", 1, f"/x/{other_gap}", 1800, 10, 0, 0)])
    cache = MetadataCache(str(db))
    assert cache.tglc_basenames_missing_blobs(7) == {}
    cache.close()


def test_tglc_basenames_missing_blobs_no_fits_blobs_table_all_missing(tmp_path):
    """fits_blobs absent entirely (older deployment / nothing ever packed) ->
    every tglc basename for the sector is reported as missing."""
    from exonit.metadata_cache import MetadataCache
    import sqlite3

    db = tmp_path / "meta.db"
    name = "hlsp_tglc_tess_ffi_gaiaid-1-s0007-cam1-ccd1_tess_v1_llc.fits"
    _make_tglc_table(str(db), [("1", 7, f"/x/{name}", 1800, 10, 0, 0)])
    cache = MetadataCache(str(db))  # __init__ creates fits_blobs
    con = sqlite3.connect(str(db))
    con.execute("DROP TABLE fits_blobs")
    con.commit()
    con.close()
    assert cache.tglc_basenames_missing_blobs(7) == {f"/x/{name}".split("/")[-1]: f"/x/{name}"}
    cache.close()


def test_tglc_basenames_missing_blobs_no_tglc_table_returns_empty(tmp_path):
    """No tglc table at all (nothing indexed yet for this deployment) ->
    empty dict, not an error — nothing to repair."""
    from exonit.metadata_cache import MetadataCache

    db = tmp_path / "meta.db"
    cache = MetadataCache(str(db))  # no tglc table created
    assert cache.tglc_basenames_missing_blobs(7) == {}
    cache.close()


# ------------------------------------------------------------------ #
# indexed_basename_count — SQL COUNT parity with len(indexed_basenames)
#
# The contract these pin is "same number, never materialised". The server's
# downloaded-counts refresh thread calls this once per sector on tables of
# ~76 M rows; taking len() of the set instead peaked at ~8 GB RSS per sector
# (measured 2026-08-02 on <host>). Every case below asserts the count
# against len(indexed_basenames(...)) rather than a hard-coded integer, so
# the two implementations can never drift apart silently.
# ------------------------------------------------------------------ #

def _count_parity(cache, sector):
    """Assert SQL count == len(set) and return the agreed value."""
    expected = len(cache.indexed_basenames(sector))
    actual = cache.indexed_basename_count(sector)
    assert actual == expected, f"SQL count {actual} != len(set) {expected}"
    return actual


def test_indexed_basename_count_matches_len_of_union(tmp_path):
    """The union dedups: a basename in BOTH tglc and fits_blobs counts once."""
    from exonit.metadata_cache import MetadataCache

    db = tmp_path / "meta.db"
    shared = "hlsp_tglc_tess_ffi_gaiaid-1-s0023-cam1-ccd1_tess_v1_llc.fits"
    only_tglc = "hlsp_tglc_tess_ffi_gaiaid-2-s0023-cam1-ccd1_tess_v1_llc.fits"
    only_blob = "hlsp_tglc_tess_ffi_gaiaid-3-s0023-cam1-ccd1_tess_v1_llc.fits"
    _make_tglc_table(str(db), [
        ("1", 23, f"/cache/s0023/{shared}", 1800.0, 10.0, 1.0, 2.0),
        ("2", 23, f"/cache/s0023/{only_tglc}", 1800.0, 10.0, 1.0, 2.0),
    ])
    cache = MetadataCache(str(db))  # __init__ creates fits_blobs
    cache.add_fits_blobs_bulk([
        (shared, "containers/run1.tar", 0, 100),
        (only_blob, "containers/run1.tar", 100, 100),
    ])
    assert _count_parity(cache, 23) == 3  # shared counted once, not twice
    cache.close()


def test_indexed_basename_count_ignores_other_sectors(tmp_path):
    """Rows of a different sector must not leak into the count."""
    from exonit.metadata_cache import MetadataCache

    db = tmp_path / "meta.db"
    _make_tglc_table(str(db), [
        ("1", 23, "/c/hlsp_tglc_tess_ffi_gaiaid-1-s0023-cam1-ccd1_tess_v1_llc.fits",
         1800.0, 10.0, 1.0, 2.0),
        ("2", 1, "/c/hlsp_tglc_tess_ffi_gaiaid-2-s0001-cam1-ccd1_tess_v1_llc.fits",
         1800.0, 10.0, 1.0, 2.0),
    ])
    cache = MetadataCache(str(db))
    cache.add_fits_blobs_bulk([
        ("hlsp_tglc_tess_ffi_gaiaid-9-s0001-cam1-ccd1_tess_v1_llc.fits",
         "containers/run1.tar", 0, 100),
    ])
    assert _count_parity(cache, 23) == 1
    assert _count_parity(cache, 1) == 2
    cache.close()


def test_indexed_basename_count_duplicate_basenames_collapse(tmp_path):
    """Two tglc rows whose paths share a basename count once (it is a SET)."""
    from exonit.metadata_cache import MetadataCache

    db = tmp_path / "meta.db"
    same = "hlsp_tglc_tess_ffi_gaiaid-1-s0023-cam1-ccd1_tess_v1_llc.fits"
    _make_tglc_table(str(db), [
        ("1", 23, f"/cache/a/{same}", 1800.0, 10.0, 1.0, 2.0),
        ("2", 23, f"/cache/b/{same}", 1800.0, 10.0, 1.0, 2.0),
    ])
    cache = MetadataCache(str(db))
    assert _count_parity(cache, 23) == 1
    cache.close()


def test_indexed_basename_count_path_edge_cases(tmp_path):
    """NULL/empty paths are skipped; a slash-less path is its own basename;
    a path ending in '/' yields the empty basename — all exactly as the set
    comprehension `{basename(p) for r in rows if r["path"]}` behaves."""
    from exonit.metadata_cache import MetadataCache

    db = tmp_path / "meta.db"
    _make_tglc_table(str(db), [
        ("1", 23, None, 1800.0, 10.0, 1.0, 2.0),          # skipped
        ("2", 23, "", 1800.0, 10.0, 1.0, 2.0),            # skipped
        ("3", 23, "bare_no_slash.fits", 1800.0, 10.0, 1.0, 2.0),   # -> itself
        ("4", 23, "/cache/dir/", 1800.0, 10.0, 1.0, 2.0),          # -> ''
        ("5", 23, "/cache/deep/a/b/c.fits", 1800.0, 10.0, 1.0, 2.0),
    ])
    cache = MetadataCache(str(db))
    names = cache.indexed_basenames(23)
    assert names == {"bare_no_slash.fits", "", "c.fits"}
    assert _count_parity(cache, 23) == 3
    cache.close()


def test_indexed_basename_count_without_fits_blobs_table(tmp_path):
    """fits_blobs absent -> tglc-only count, same fallback as the set."""
    import sqlite3
    from exonit.metadata_cache import MetadataCache

    db = tmp_path / "meta.db"
    _make_tglc_table(str(db), [
        ("1", 7, "/x/hlsp_tglc_tess_ffi_gaiaid-1-s0007-cam1-ccd1_tess_v1_llc.fits",
         1800.0, 10.0, 1.0, 2.0),
    ])
    cache = MetadataCache(str(db))  # __init__ creates fits_blobs
    con = sqlite3.connect(str(db))
    con.execute("DROP TABLE fits_blobs")
    con.commit()
    con.close()
    assert _count_parity(cache, 7) == 1
    cache.close()


def test_indexed_basename_count_without_tglc_table_is_zero(tmp_path):
    """No tglc -> 0, and fits_blobs is NOT consulted even though it exists.

    This is the asymmetry indexed_basenames has (its first `except` returns
    an empty set before the fits_blobs query runs); the count must reproduce
    it rather than 'improve' on it.
    """
    from exonit.metadata_cache import MetadataCache

    db = tmp_path / "meta.db"
    cache = MetadataCache(str(db))  # no tglc table created
    cache.add_fits_blobs_bulk([
        ("hlsp_tglc_tess_ffi_gaiaid-1-s0007-cam1-ccd1_tess_v1_llc.fits",
         "containers/run1.tar", 0, 100),
    ])
    assert cache.indexed_basenames(7) == set()
    assert _count_parity(cache, 7) == 0
    cache.close()


# ------------------------------------------------------------------ #
# download_registry_fingerprint — change-detection gate for the "downloaded
# counts" scan (2026-09-25, fix C). Callers (exonit/routes/discovery.py's
# background refresh; scripts/discovery_download.py's --recompute-registry)
# skip the expensive indexed_basename_count()/indexed_basenames() scan when
# this fingerprint is unchanged since the last real recompute.
# ------------------------------------------------------------------ #

def test_download_registry_fingerprint_none_on_sqlite(tmp_path):
    """SQLite has no pg_stat_user_tables — must return None, the documented
    'no signal, always recompute' fallback."""
    db = tmp_path / "meta.db"
    cache = MetadataCache(str(db))
    assert cache.download_registry_fingerprint() is None
    cache.close()


def test_download_registry_fingerprint_none_on_query_failure(tmp_path):
    """A PG-backend query failure (stats collector down, connection lost,
    ...) must degrade to None, not raise — a fingerprint check must never be
    able to break the endpoint/script it is only meant to speed up."""
    db = tmp_path / "meta.db"
    cache = MetadataCache(str(db))
    # Force the PG branch without a real PG server: flip the flag and make
    # the query raise, exactly what a broken connection would do.
    cache._is_pg = True

    def _boom(sql, params=()):
        raise RuntimeError("simulated PG failure")

    cache._backend.query = _boom
    assert cache.download_registry_fingerprint() is None
    cache.close()


class TestDownloadRegistryFingerprintRealPg:
    """End-to-end against a real throwaway PG (exonit_meta_test) — skipped
    when EXONIT_TEST_META_PG_URL is unset/unreachable, per
    tests/_metadata_pg_test_db.py's guard."""

    def test_fingerprint_changes_after_insert_flat_non_partitioned_tglc(
        self, monkeypatch
    ):
        import psycopg
        from tests._metadata_pg_test_db import (
            pg_meta_test_available, resolve_test_meta_dsn,
        )

        dsn = resolve_test_meta_dsn()
        if not pg_meta_test_available(dsn):
            pytest.skip("EXONIT_TEST_META_PG_URL (throwaway PG) unset or not reachable")

        with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
            c.execute("DROP TABLE IF EXISTS tglc CASCADE")
            c.execute("DROP TABLE IF EXISTS fits_blobs CASCADE")
            c.execute(
                "CREATE TABLE tglc (tic_id TEXT NOT NULL, sector BIGINT NOT NULL, "
                "path TEXT NOT NULL, exptime DOUBLE PRECISION, tmag DOUBLE PRECISION, "
                "ra DOUBLE PRECISION, dec DOUBLE PRECISION, "
                "PRIMARY KEY (tic_id, sector))"
            )
            c.execute(
                "CREATE TABLE fits_blobs (filename TEXT PRIMARY KEY, "
                "container_key TEXT NOT NULL, data_offset BIGINT NOT NULL, "
                "data_length BIGINT NOT NULL)"
            )

        monkeypatch.setenv("EXONIT_METADATA_DB_URL", dsn)
        try:
            cache = MetadataCache("unused-path-pg-wins.db")
            try:
                assert cache._is_pg
                before = cache.download_registry_fingerprint()
                assert before is not None
                assert set(before.keys()) >= {"tglc", "fits_blobs"}

                # Same fingerprint on an immediate re-read (nothing changed).
                assert cache.download_registry_fingerprint() == before

                cache.add_fits_blobs_bulk([
                    ("hlsp_tglc_tess_ffi_gaiaid-1-s0023-cam1-ccd1_tess_v1_llc.fits",
                     "containers/s0023/dl_1.tar", 0, 10000),
                ])
                # ANALYZE forces the stats collector to reflect the write
                # immediately rather than waiting for autovacuum's next pass.
                with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
                    c.execute("ANALYZE fits_blobs")

                after = cache.download_registry_fingerprint()
                assert after != before
            finally:
                cache.close()
        finally:
            with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
                c.execute("DROP TABLE IF EXISTS tglc CASCADE")
                c.execute("DROP TABLE IF EXISTS fits_blobs CASCADE")

    def test_fingerprint_changes_after_insert_into_partitioned_tglc_child(
        self, monkeypatch
    ):
        """2026-09-25 review finding: production's ``tglc`` is a
        declaratively PARTITIONED table (``PARTITION BY LIST (sector)``, one
        ``tglc_sNNNN`` child per sector — mirrors
        ``scripts/migration/schema_pg_meta.sql``'s real DDL, unlike the flat
        ``tglc`` in the test above). PostgreSQL routes ``INSERT INTO tglc``
        to the matching CHILD partition and only the child's
        ``pg_stat_user_tables`` row moves — the parent's row stays
        permanently zero. A naive ``WHERE relname = 'tglc'`` read (the
        pre-fix bug) is therefore blind to every real TGLC download; this
        proves the fingerprint reacts correctly to a write landing in a
        child partition."""
        import psycopg
        from tests._metadata_pg_test_db import (
            pg_meta_test_available, resolve_test_meta_dsn,
        )

        dsn = resolve_test_meta_dsn()
        if not pg_meta_test_available(dsn):
            pytest.skip("EXONIT_TEST_META_PG_URL (throwaway PG) unset or not reachable")

        with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
            c.execute("DROP TABLE IF EXISTS tglc CASCADE")
            c.execute("DROP TABLE IF EXISTS fits_blobs CASCADE")
            c.execute(
                "CREATE TABLE tglc (tic_id TEXT NOT NULL, sector BIGINT NOT NULL, "
                "path TEXT NOT NULL, exptime DOUBLE PRECISION, tmag DOUBLE PRECISION, "
                "ra DOUBLE PRECISION, dec DOUBLE PRECISION, "
                "PRIMARY KEY (tic_id, sector)) PARTITION BY LIST (sector)"
            )
            c.execute("CREATE TABLE tglc_s0001 PARTITION OF tglc FOR VALUES IN (1)")
            c.execute("CREATE TABLE tglc_s0002 PARTITION OF tglc FOR VALUES IN (2)")
            c.execute(
                "CREATE TABLE fits_blobs (filename TEXT PRIMARY KEY, "
                "container_key TEXT NOT NULL, data_offset BIGINT NOT NULL, "
                "data_length BIGINT NOT NULL)"
            )

        monkeypatch.setenv("EXONIT_METADATA_DB_URL", dsn)
        try:
            cache = MetadataCache("unused-path-pg-wins.db")
            try:
                assert cache._is_pg
                before = cache.download_registry_fingerprint()
                assert before is not None
                # Parent + 2 empty child partitions; a flat (non-partitioned)
                # fits_blobs is exactly 1 relation.
                assert before["tglc"]["n_relations"] == 3
                assert before["fits_blobs"]["n_relations"] == 1
                # The pre-fix bug's exact failure mode: reading only the
                # (permanently zero) parent would report this baseline as
                # {n_tup_ins: 0, n_live_tup: 0, ...} even after the insert
                # below — asserted explicitly, not just "before != after".
                assert before["tglc"]["n_tup_ins"] == 0
                assert before["tglc"]["n_live_tup"] == 0

                # Write lands in the CHILD partition tglc_s0001, never the
                # parent directly (PostgreSQL routes it there automatically).
                with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
                    c.execute(
                        "INSERT INTO tglc (tic_id, sector, path) "
                        "VALUES ('1', 1, '/x/hlsp_tglc_tess_ffi_gaiaid-1-"
                        "s0001-cam1-ccd1_tess_v1_llc.fits')")
                    c.execute("ANALYZE tglc")

                after = cache.download_registry_fingerprint()
                assert after["tglc"] != before["tglc"], (
                    "fingerprint must change after a write to a tglc CHILD "
                    "partition -- this is the exact bug being fixed"
                )
                assert after["tglc"]["n_tup_ins"] >= 1
                assert after["tglc"]["n_live_tup"] >= 1
                # fits_blobs was never touched -> must be byte-identical.
                assert after["fits_blobs"] == before["fits_blobs"]

                # Attaching a brand-new (still EMPTY) partition must also
                # register as a change via n_relations, even though its
                # stats are all-zero (same as tglc_s0002's, already present)
                # -- proves the fingerprint distinguishes relations by
                # identity, not just by their numeric counters.
                with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
                    c.execute(
                        "CREATE TABLE tglc_s0003 PARTITION OF tglc FOR VALUES IN (3)")
                attached = cache.download_registry_fingerprint()
                assert attached["tglc"]["n_relations"] == after["tglc"]["n_relations"] + 1
                assert attached["tglc"] != after["tglc"]
            finally:
                cache.close()
        finally:
            with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
                c.execute("DROP TABLE IF EXISTS tglc CASCADE")
                c.execute("DROP TABLE IF EXISTS fits_blobs CASCADE")
