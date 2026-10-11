"""Direct unit tests for exonit/metadata_index.py::_tglc_lookup.

2026-07-20 review fix (progress/f1-metadata-backend-review.md item 2): the
F1 seam refactor changed `_tglc_lookup`'s row-access pattern (tuple-unpack ->
dict-key access via MetadataCache.tglc_lookup) and added a process-wide
MetadataCache cache, but `tests/test_enqueue.py` always mocks
`ensure_metadata`/`_tglc_lookup` away, so the function itself had no direct
test coverage. This file exercises the real function against a real (tmp
SQLite) tglc index, per the review's ask: row shape, missing-TIC, and the
PG-DSN-set file-exists-check bypass.
"""
import os
import sqlite3

import pytest

from exonit import metadata_index
from exonit.metadata_index import _tglc_lookup


def _make_tglc_db(db_path, rows):
    """rows: list of (tic_id, sector, path, exptime, tmag, ra, dec) tuples."""
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE tglc (tic_id TEXT, sector INTEGER, path TEXT, "
        "exptime REAL, tmag REAL, ra REAL, dec REAL, "
        "PRIMARY KEY(tic_id, sector))"
    )
    con.executemany("INSERT INTO tglc VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


@pytest.fixture(autouse=True)
def _clear_shared_cache():
    """_tglc_lookup caches MetadataCache instances process-wide, keyed by the
    tglc_index string. Different tests reuse tmp_path-derived paths that are
    never equal across tests (pytest gives each test its own tmp_path), so
    this isn't strictly required for isolation — but clearing it keeps each
    test's assertions about cache construction independent of prior tests'
    side effects (e.g. a stale entry surviving with a closed connection)."""
    metadata_index._shared_tglc_caches.clear()
    yield
    metadata_index._shared_tglc_caches.clear()


# --------------------------------------------------------------------------- #
# Row shape: sectors/exptime/files/tmag/ra/dec extraction
# --------------------------------------------------------------------------- #

def test_tglc_lookup_returns_expected_shape_for_single_sector(tmp_path):
    db = str(tmp_path / "meta.db")
    _make_tglc_db(db, [
        ("100", 23, "/cache/s0023/hlsp_tglc_gaiaid-1-s0023-cam1-ccd1_v1_llc.fits",
         1800.0, 9.5, 12.34, -5.67),
    ])
    result = _tglc_lookup(100, db)
    assert result is not None
    sectors, exptime, files, tmag, ra, dec = result
    assert sectors == [23]
    assert exptime == pytest.approx(1800.0)
    assert len(files) == 1
    basename, abspath = files[0]
    assert basename == "hlsp_tglc_gaiaid-1-s0023-cam1-ccd1_v1_llc.fits"
    assert abspath.endswith(basename)
    assert tmag == pytest.approx(9.5)
    assert ra == pytest.approx(12.34)
    assert dec == pytest.approx(-5.67)


def test_tglc_lookup_multi_sector_dedupes_and_sorts_sectors(tmp_path):
    db = str(tmp_path / "meta.db")
    _make_tglc_db(db, [
        ("200", 5, "/cache/s0005/a-s0005-x.fits", 1800.0, 10.0, 1.0, 2.0),
        ("200", 2, "/cache/s0002/b-s0002-x.fits", 1800.0, 10.0, 1.0, 2.0),
    ])
    result = _tglc_lookup(200, db)
    assert result is not None
    sectors, exptime, files, tmag, ra, dec = result
    assert sectors == [2, 5]  # sorted ascending
    assert len(files) == 2


def test_tglc_lookup_skips_rows_with_null_path(tmp_path):
    """A row with a NULL/empty path (deleted-after-upload placeholder) must
    be skipped when building the files list, per the existing MinIO-era
    contract (see the function's own inline comment about not gating on
    local existence, but still requiring a non-empty path string)."""
    db = str(tmp_path / "meta.db")
    _make_tglc_db(db, [("300", 1, "", 1800.0, 10.0, 1.0, 2.0)])
    assert _tglc_lookup(300, db) is None


# --------------------------------------------------------------------------- #
# Missing cases
# --------------------------------------------------------------------------- #

def test_tglc_lookup_tic_not_found_returns_none(tmp_path):
    db = str(tmp_path / "meta.db")
    _make_tglc_db(db, [("100", 1, "/cache/a.fits", 1800.0, 10.0, 1.0, 2.0)])
    assert _tglc_lookup(999, db) is None


def test_tglc_lookup_missing_index_file_returns_none(tmp_path):
    missing = str(tmp_path / "does_not_exist.db")
    assert not os.path.exists(missing)
    assert _tglc_lookup(100, missing) is None
    # Must not have created the file as a side effect (the pre-change
    # contract: a raw sqlite3.connect() would silently create an empty file;
    # the os.path.exists() guard here exists specifically to avoid that).
    assert not os.path.exists(missing)


def test_tglc_lookup_empty_tglc_index_arg_returns_none():
    assert _tglc_lookup(100, "") is None
    assert _tglc_lookup(100, None) is None


def test_tglc_lookup_table_missing_returns_none(tmp_path):
    """The index file exists but has no tglc table yet (e.g. a fresh
    MetadataCache-provisioned file with only metadata/fits_paths/etc)."""
    db = str(tmp_path / "meta.db")
    from exonit.metadata_cache import MetadataCache
    MetadataCache(db).close()  # provisions metadata/fits_paths/... but not tglc
    assert _tglc_lookup(100, db) is None


# --------------------------------------------------------------------------- #
# PG-DSN-set: the file-exists() guard on tglc_index must be bypassed
# --------------------------------------------------------------------------- #

def test_tglc_lookup_pg_dsn_set_bypasses_file_exists_check(tmp_path, monkeypatch):
    """When EXONIT_METADATA_DB_URL is a valid postgresql:// DSN,
    MetadataCache uses PG regardless of the tglc_index string passed — so
    the local os.path.exists(tglc_index) guard must not short-circuit just
    because that (now-irrelevant) sqlite fallback path doesn't exist
    locally. Verified via a stub cache (no real network connection needed —
    see the docstring in exonit/metadata_index.py's guard)."""
    monkeypatch.setenv(
        "EXONIT_METADATA_DB_URL", "postgresql://u@h:5432/exonit_meta_test")
    nonexistent = str(tmp_path / "does_not_exist.db")
    assert not os.path.exists(nonexistent)

    class _StubCache:
        def tglc_lookup(self, tic_id):
            assert tic_id == "100"
            return [{"sector": 7, "path": "/cache/a-s0007-x.fits",
                      "exptime": 1800.0, "tmag": 8.0, "ra": 3.0, "dec": 4.0}]

    monkeypatch.setattr(metadata_index, "_get_tglc_cache", lambda idx: _StubCache())
    result = _tglc_lookup(100, nonexistent)
    assert result is not None
    sectors, exptime, files, tmag, ra, dec = result
    assert sectors == [7]
    assert files[0][0] == "a-s0007-x.fits"


def test_tglc_lookup_pg_dsn_unset_still_gates_on_missing_file(tmp_path, monkeypatch):
    """Sanity counterpart to the bypass test above: with the env var unset
    (the default), a missing sqlite file still short-circuits to None
    without ever reaching the (stubbed, would-assert-if-called) cache."""
    monkeypatch.delenv("EXONIT_METADATA_DB_URL", raising=False)
    nonexistent = str(tmp_path / "does_not_exist.db")

    def _boom(idx):
        raise AssertionError("must not be reached when the file is missing "
                              "and no PG override is set")

    monkeypatch.setattr(metadata_index, "_get_tglc_cache", _boom)
    assert _tglc_lookup(100, nonexistent) is None


# --------------------------------------------------------------------------- #
# 2026-07-21 perf fix: abspath (pure string op), not resolve() (per-component
# lstat syscalls). See the inline comment on the files.append() line in
# exonit/metadata_index.py::_tglc_lookup for the incident this fixes (300k-TIC
# enqueue 3min->40min against a 15M-entry tglc_cache with a cold dentry cache).
# --------------------------------------------------------------------------- #

def test_tglc_lookup_files_use_abspath_existing_and_missing_path(tmp_path):
    """The second element of each files[] entry must equal
    os.path.abspath(<path stored in the tglc row>), identically whether that
    path exists on disk or not. Since the 2026-06-24 MinIO migration the
    local file is *always* gone by the time discovery enqueue runs (deleted
    right after upload — see the "FITS live in MinIO" comment a few lines
    above the call site), so the not-exists case is the real production
    path, not an edge case."""
    db = str(tmp_path / "meta.db")

    existing_dir = tmp_path / "exists_on_disk"
    existing_dir.mkdir()
    existing_file = existing_dir / "hlsp_tglc-exists-s0001-x.fits"
    existing_file.write_bytes(b"x")

    # Relative path with a ".." component, deliberately NOT created on disk
    # (mirrors the real post-MinIO state: row present, file gone).
    missing_relpath = "../nowhere/hlsp_tglc-gone-s0002-x.fits"
    assert not os.path.exists(missing_relpath)

    _make_tglc_db(db, [
        ("400", 1, str(existing_file), 1800.0, 10.0, 1.0, 2.0),
        ("400", 2, missing_relpath, 1800.0, 10.0, 1.0, 2.0),
    ])
    result = _tglc_lookup(400, db)
    assert result is not None
    _sectors, _exptime, files, _tmag, _ra, _dec = result
    by_basename = dict(files)

    existing_bn = os.path.basename(str(existing_file))
    assert by_basename[existing_bn] == os.path.abspath(str(existing_file))

    missing_bn = os.path.basename(missing_relpath)
    assert by_basename[missing_bn] == os.path.abspath(missing_relpath)
    # Confirms this really is the missing-file case, not a fluke.
    assert not os.path.exists(by_basename[missing_bn])


def test_tglc_lookup_does_not_call_path_resolve(tmp_path, monkeypatch):
    """Contract: _tglc_lookup must not invoke pathlib.Path.resolve() (the
    syscall-heavy, per-component realpath walk this fix removed) anywhere in
    its per-row path handling. Patch Path.resolve to blow up if reached."""
    from pathlib import Path as _Path

    def _must_not_be_called(self, *a, **kw):
        raise AssertionError(
            "_tglc_lookup must not call Path.resolve() - abspath() only")

    monkeypatch.setattr(_Path, "resolve", _must_not_be_called)

    db = str(tmp_path / "meta.db")
    _make_tglc_db(db, [
        ("500", 1, "/cache/s0001/hlsp_tglc-a-s0001-x.fits",
         1800.0, 10.0, 1.0, 2.0),
    ])
    result = _tglc_lookup(500, db)
    assert result is not None
    _sectors, _exptime, files, _tmag, _ra, _dec = result
    assert files[0] == (
        "hlsp_tglc-a-s0001-x.fits",
        os.path.abspath("/cache/s0001/hlsp_tglc-a-s0001-x.fits"),
    )


def test_ensure_metadata_bulk_warns_when_spoc_root_missing(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="exonit.metadata_index"):
        metadata_index.ensure_metadata_bulk([], None, spoc_root=str(tmp_path / "nope"))
    assert any("not a directory" in r.message for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="exonit.metadata_index"):
        metadata_index.ensure_metadata_bulk([], None, spoc_root=str(tmp_path))
    assert not any("not a directory" in r.message for r in caplog.records)
