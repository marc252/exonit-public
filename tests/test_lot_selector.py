"""Tests for exonit/lot_selector.py — S1's O(lot) paged selector.

docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md §4. Compares the
paged selector against the legacy next_block/load_extended_canonical path
on random fixtures (no canonical CSV — the case paged mode actually
handles), and pins the 08-10 empty-lot-loop regression + the 09-26
exhausted-plan incident guard in this new code path too.
"""
import random
import sqlite3

import pytest

from exonit.db import ExoNitDB
from exonit.discovery_state import next_block
from exonit.lot_selector import select_lot_paged

SECTOR = 6


def _make_tglc(path, rows):
    """rows: (tic_id str, sector, path, exptime, tmag, ra, dec) tuples,
    inserted in list order (SQLite rowid order == list order == the paged
    selector's/legacy's seq order for this test)."""
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE tglc (tic_id TEXT, sector INTEGER, path TEXT, "
        "exptime REAL, tmag REAL, ra REAL, dec REAL, "
        "PRIMARY KEY(tic_id, sector))"
    )
    con.executemany("INSERT INTO tglc VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


def _tss(db, tic_id, sector, status, attempts=0):
    db._exec(
        "INSERT INTO tic_sector_state (tic_id, sector, status, attempts) "
        "VALUES (?, ?, ?, ?)",
        (str(tic_id), sector, status, attempts))


def _build_fixture(seed, n=200, n_done=20, n_exhausted=15, n_toi=10):
    """Deterministic-per-seed random fixture: n TICs in a shuffled tglc
    insertion order, split into done/exhausted/TOI/plain categories (no
    retry-eligible TICs — see test_paged_matches_legacy_no_retry, kept
    separate from the retry-priority tests below on purpose)."""
    rng = random.Random(seed)
    tic_ids = list(range(900000, 900000 + n))
    rng.shuffle(tic_ids)

    pool = list(tic_ids)
    rng.shuffle(pool)
    done_tics = set(pool[:n_done])
    exhausted_tics = set(pool[n_done:n_done + n_exhausted])
    toi_tics = set(pool[n_done + n_exhausted:n_done + n_exhausted + n_toi])

    tglc_rows = [(str(t), SECTOR, f"/{t}.fits", 1800.0, 10.0, 0.0, 0.0)
                for t in tic_ids]
    return tic_ids, tglc_rows, done_tics, exhausted_tics, toi_tics


def _setup_dbs(tmp_path, tglc_rows, done_tics, exhausted_tics, seed):
    db_path = str(tmp_path / f"ex_{seed}.db")
    tglc_path = str(tmp_path / f"tglc_{seed}.db")
    _make_tglc(tglc_path, tglc_rows)
    db = ExoNitDB(db_path)
    for t in done_tics:
        _tss(db, t, SECTOR, "completed", attempts=0)
    for t in exhausted_tics:
        _tss(db, t, SECTOR, "failed", attempts=2)  # MAX_LOTS_PER_SECTOR
    return db_path, tglc_path


@pytest.mark.parametrize("seed", range(10))
def test_paged_matches_legacy_no_retry(tmp_path, seed):
    """Paged vs legacy parity (§4.4): with no retry TIC, the selection
    must be IDENTICAL (same order) to next_block over the same list
    (tglc in insertion order = "canonical_tics" without a canonical CSV), and
    new_cursor must match the legacy index formula."""
    tic_ids, tglc_rows, done_tics, exhausted_tics, toi_tics = _build_fixture(seed)
    db_path, tglc_path = _setup_dbs(
        tmp_path, tglc_rows, done_tics, exhausted_tics, seed)

    count = 47  # 3 pages of 20 -> a non-round count exercises a partial page
    legacy_done_set = done_tics | exhausted_tics | toi_tics
    legacy = next_block(tic_ids, count, set(tic_ids), legacy_done_set)

    paged = select_lot_paged(
        SECTOR, count,
        db_path=db_path, tglc_index_path=tglc_path,
        cursor=0, cursor_seq=None, toi_tic_set=toi_tics,
        page_size=20,
    )

    assert paged["retry_selected"] == 0
    assert paged["selected"] == legacy["selected"], seed
    assert paged["ready"] == legacy["ready"]
    assert paged["missing_download"] == []
    expected_new_cursor = tic_ids.index(legacy["selected"][-1]) + 1
    assert paged["new_cursor"] == expected_new_cursor, seed
    # remaining_undone (§4.1.6) is an upper bound over what's left AFTER
    # this lot's selection (spec's own formula: len(retry)-retry_selected +
    # (sector_total - new_cursor)) — NOT legacy's own "whole-list, before
    # this lot" number (which counts this lot's own selection too), so the
    # two are different quantities by design; compare against the TRUE
    # remaining-after-this-lot count instead.
    true_remaining_after_lot = len([
        t for t in tic_ids
        if t not in legacy_done_set and t not in set(legacy["selected"])
    ])
    assert paged["remaining_undone"] >= true_remaining_after_lot, seed
    assert paged["remaining_undone_is_upper_bound"] is True


def test_paged_dedupes_retry_against_page_scan(tmp_path):
    """A retry-eligible TIC sitting in the MIDDLE of tglc order is placed
    FIRST in the output (priority) and never re-selected a second time
    when the page scan later reaches its natural position (§4.4 dedupe)."""
    n = 100
    tic_ids = list(range(800000, 800000 + n))
    tglc_rows = [(str(t), SECTOR, f"/{t}.fits", 1800.0, 10.0, 0.0, 0.0)
                for t in tic_ids]
    db_path = str(tmp_path / "ex.db")
    tglc_path = str(tmp_path / "tglc.db")
    _make_tglc(tglc_path, tglc_rows)
    db = ExoNitDB(db_path)

    retry_tic = tic_ids[50]  # sits at position 50, not the front
    _tss(db, retry_tic, SECTOR, "failed", attempts=1)  # retry-eligible

    block = select_lot_paged(
        SECTOR, 10,
        db_path=db_path, tglc_index_path=tglc_path,
        cursor=0, cursor_seq=None, toi_tic_set=set(),
        page_size=20,
    )

    assert block["retry_selected"] == 1
    assert block["selected"][0] == retry_tic          # placed FIRST
    assert block["selected"].count(retry_tic) == 1    # never duplicated
    assert len(block["selected"]) == 10


def test_paged_retry_pool_includes_legacy_sector_zero_failures(tmp_path):
    """retryable_failed_ids covers sector IN (S, 0) — the legacy/backfill
    sentinel sector (done_state.LEGACY_SECTOR) — per the spec's own note
    ("the 0 covers the legacy/backfill failures")."""
    n = 30
    tic_ids = list(range(700000, 700000 + n))
    tglc_rows = [(str(t), SECTOR, f"/{t}.fits", 1800.0, 10.0, 0.0, 0.0)
                for t in tic_ids]
    db_path = str(tmp_path / "ex.db")
    tglc_path = str(tmp_path / "tglc.db")
    _make_tglc(tglc_path, tglc_rows)
    db = ExoNitDB(db_path)

    legacy_failed_tic = tic_ids[10]
    _tss(db, legacy_failed_tic, 0, "failed", attempts=1)  # sector=0 (legacy)

    block = select_lot_paged(
        SECTOR, 5,
        db_path=db_path, tglc_index_path=tglc_path,
        cursor=0, cursor_seq=None, toi_tic_set=set(),
    )
    assert block["retry_selected"] == 1
    assert block["selected"][0] == legacy_failed_tic


def test_paged_excludes_exhausted_and_done_from_retry_pool(tmp_path):
    """A TIC failed twice (exhausted) or already completed must NOT be
    treated as retry-eligible even if it also has a stray 'failed' row for
    a DIFFERENT attempt count is not the case here — this pins the filter
    at the retry-set-construction stage, not just the page-scan stage."""
    n = 20
    tic_ids = list(range(600000, 600000 + n))
    tglc_rows = [(str(t), SECTOR, f"/{t}.fits", 1800.0, 10.0, 0.0, 0.0)
                for t in tic_ids]
    db_path = str(tmp_path / "ex.db")
    tglc_path = str(tmp_path / "tglc.db")
    _make_tglc(tglc_path, tglc_rows)
    db = ExoNitDB(db_path)

    exhausted_tic = tic_ids[3]
    _tss(db, exhausted_tic, SECTOR, "failed", attempts=2)  # exhausted, not retry

    block = select_lot_paged(
        SECTOR, 20,
        db_path=db_path, tglc_index_path=tglc_path,
        cursor=0, cursor_seq=None, toi_tic_set=set(),
    )
    assert exhausted_tic not in block["selected"]
    assert block["retry_selected"] == 0


def test_cursor_seq_bootstrap_from_legacy_cursor(tmp_path):
    """First paged call for a sector with an existing LEGACY (row-count)
    cursor: cursor_seq=None triggers a bootstrap via tglc_seq_at_offset, and
    the page scan resumes AFTER that offset (§4.1.3) — rows before the
    cursor are never re-examined in paged mode (even if eligible; this is
    the "deriva accepted", §4.1.8)."""
    n = 50
    tic_ids = list(range(500000, 500000 + n))
    tglc_rows = [(str(t), SECTOR, f"/{t}.fits", 1800.0, 10.0, 0.0, 0.0)
                for t in tic_ids]
    db_path = str(tmp_path / "ex.db")
    tglc_path = str(tmp_path / "tglc.db")
    _make_tglc(tglc_path, tglc_rows)
    ExoNitDB(db_path)  # provision tic_sector_state, empty

    legacy_cursor = 20  # rows 0..19 already "processed" per the registry
    block = select_lot_paged(
        SECTOR, 5,
        db_path=db_path, tglc_index_path=tglc_path,
        cursor=legacy_cursor, cursor_seq=None, toi_tic_set=set(),
    )
    # The first NEW candidate must be tic_ids[20] (0-indexed) -- rows
    # 0..19 are skipped by the bootstrap without being re-examined.
    assert block["selected"][0] == tic_ids[20]
    assert block["new_cursor"] == legacy_cursor + 5


def test_cursor_seq_reused_when_already_persisted(tmp_path):
    """A subsequent paged call with a persisted cursor_seq skips the
    bootstrap entirely and resumes from EXACTLY that seq."""
    n = 50
    tic_ids = list(range(400000, 400000 + n))
    tglc_rows = [(str(t), SECTOR, f"/{t}.fits", 1800.0, 10.0, 0.0, 0.0)
                for t in tic_ids]
    db_path = str(tmp_path / "ex.db")
    tglc_path = str(tmp_path / "tglc.db")
    _make_tglc(tglc_path, tglc_rows)
    ExoNitDB(db_path)

    from exonit.metadata_cache import MetadataCache
    mc = MetadataCache(tglc_path)
    seq_at_10 = mc.tglc_seq_at_offset(SECTOR, 9)  # seq of the 10th row (0-idx 9)
    mc.close()

    block = select_lot_paged(
        SECTOR, 3,
        db_path=db_path, tglc_index_path=tglc_path,
        cursor=999,  # deliberately WRONG/stale legacy cursor
        cursor_seq=seq_at_10,  # but a VALID persisted cursor_seq wins
        toi_tic_set=set(),
    )
    assert block["selected"][0] == tic_ids[10]


def test_only_retry_selected_does_not_move_cursor(tmp_path):
    """§4.1.5: if the retry pool alone satisfies `count`, no page is ever
    scanned and the cursor does not move at all."""
    n = 30
    tic_ids = list(range(300000, 300000 + n))
    tglc_rows = [(str(t), SECTOR, f"/{t}.fits", 1800.0, 10.0, 0.0, 0.0)
                for t in tic_ids]
    db_path = str(tmp_path / "ex.db")
    tglc_path = str(tmp_path / "tglc.db")
    _make_tglc(tglc_path, tglc_rows)
    db = ExoNitDB(db_path)
    _tss(db, tic_ids[5], SECTOR, "failed", attempts=1)
    _tss(db, tic_ids[6], SECTOR, "failed", attempts=1)

    block = select_lot_paged(
        SECTOR, 2,  # exactly the retry pool's size
        db_path=db_path, tglc_index_path=tglc_path,
        cursor=7, cursor_seq=100, toi_tic_set=set(),
    )
    assert block["retry_selected"] == 2
    assert block["new_cursor"] == 7          # unchanged
    assert block["new_cursor_seq"] == 100    # unchanged


class TestRegression0810EmptyLotLoopPaged:
    """08-10 regression + 09-26 exhausted-plan incident, in the paged path:
    an all-exhausted sector must report remaining_undone == 0 EXACTLY and
    be idempotent across repeated calls with the persisted cursor_seq."""

    def test_all_exhausted_reports_zero_remaining_idempotently(self, tmp_path):
        n = 15
        tic_ids = list(range(200000, 200000 + n))
        tglc_rows = [(str(t), SECTOR, f"/{t}.fits", 1800.0, 10.0, 0.0, 0.0)
                    for t in tic_ids]
        db_path = str(tmp_path / "ex.db")
        tglc_path = str(tmp_path / "tglc.db")
        _make_tglc(tglc_path, tglc_rows)
        db = ExoNitDB(db_path)
        for t in tic_ids:
            _tss(db, t, SECTOR, "failed", attempts=2)  # every TIC exhausted

        block1 = select_lot_paged(
            SECTOR, 100,
            db_path=db_path, tglc_index_path=tglc_path,
            cursor=0, cursor_seq=None, toi_tic_set=set(),
        )
        assert block1["selected"] == []
        assert block1["remaining_undone"] == 0
        assert block1["end_reached"] is True

        # Second call, same persisted cursor: idempotent, still empty/zero.
        block2 = select_lot_paged(
            SECTOR, 100,
            db_path=db_path, tglc_index_path=tglc_path,
            cursor=block1["new_cursor"], cursor_seq=block1["new_cursor_seq"],
            toi_tic_set=set(),
        )
        assert block2["selected"] == []
        assert block2["remaining_undone"] == 0
        assert block2["new_cursor"] == block1["new_cursor"]  # unchanged
