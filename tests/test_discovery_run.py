"""Tests for exonit/discovery_state.py and scripts/discovery_run.py.

All tests use tmp_path — never touch the live DB or the real registry.
The subprocess tests run scripts/discovery_run.py with --dry-run so no
enqueue or registry write happens against production data.
"""

import csv
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_canonical_csv(path: Path, tics: list) -> None:
    """Write a minimal tic_id,label CSV."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tic_id", "label"])
        for t in tics:
            w.writerow([t, "candidate"])


def _make_tasks_db(path: Path, done_tics: list, pending_tics: list) -> None:
    """Create a minimal tasks table with some completed and some pending rows.

    Uses status='completed' to match the production value written by
    db.py::mark_task_completed (UPDATE tasks SET status='completed' ...).
    Also inserts an 'assigned' sentinel to prove the filter discriminates.
    """
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE tasks ("
        "task_id TEXT PRIMARY KEY, tic_id TEXT, status TEXT DEFAULT 'pending')"
    )
    for i, t in enumerate(done_tics):
        con.execute(
            "INSERT INTO tasks VALUES (?, ?, ?)",
            (f"comp-{i}", str(t), "completed"),
        )
    for i, t in enumerate(pending_tics):
        con.execute(
            "INSERT INTO tasks VALUES (?, ?, ?)",
            (f"pend-{i}", str(t), "pending"),
        )
    # Sentinel: 'assigned' must never be treated as done
    if done_tics:
        sentinel_tic = done_tics[0] + 900000  # guaranteed not in test TIC lists
        con.execute(
            "INSERT INTO tasks VALUES (?, ?, ?)",
            ("sentinel-assigned", str(sentinel_tic), "assigned"),
        )
    con.commit()
    con.close()


def _make_tglc_db(path: Path, tic_ids: list) -> None:
    """Create a minimal tglc index DB."""
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE tglc ("
        "tic_id TEXT, sector INTEGER, path TEXT, exptime REAL, "
        "tmag REAL, ra REAL, dec REAL, PRIMARY KEY(tic_id, sector))"
    )
    for t in tic_ids:
        con.execute(
            "INSERT INTO tglc VALUES (?, 1, '/tmp/fake.fits', 1800, 10, 0, 0)",
            (str(t),),
        )
    con.commit()
    con.close()


def _make_registry(path: Path, canonical: str, cursor: int = 0) -> None:
    data = {
        "active_sector": 1,
        "sectors": {
            "1": {
                "target_list": "data/discovery/target_lists/s0001.csv",
                "canonical_list": canonical,
                "total_tics": 4973,
                "downloaded": 5000,
                "cursor": cursor,
                "enabled": True,
            }
        },
    }
    path.write_text(json.dumps(data, indent=2))


def _make_deep_registry(path: Path, sector: int, cursor: int, deep_cursor: int) -> None:
    """Registry entry with BOTH cursor (campaign) and deep_cursor (deep-rescan)."""
    data = {
        "active_sector": sector,
        "sectors": {
            str(sector): {
                "target_list": f"data/discovery/target_lists/s{sector:04d}.csv",
                "canonical_list": "unused-in-deep-mode.csv",
                "total_tics": 0,
                "downloaded": 0,
                "cursor": cursor,
                "deep_cursor": deep_cursor,
                "enabled": True,
            }
        },
    }
    path.write_text(json.dumps(data, indent=2))


# ------------------------------------------------------------------ #
# discovery_state unit tests
# ------------------------------------------------------------------ #

class TestLoadRegistry:
    def test_returns_default_when_file_missing(self, tmp_path):
        from exonit.discovery_state import load_registry
        reg = load_registry(str(tmp_path / "nonexistent.json"))
        assert reg["active_sector"] == 1
        assert "1" in reg["sectors"]
        assert reg["sectors"]["1"]["enabled"] is True

    def test_roundtrip_save_load(self, tmp_path):
        from exonit.discovery_state import load_registry, save_registry
        data = {"active_sector": 1, "sectors": {"1": {"cursor": 42, "enabled": True}}}
        path = str(tmp_path / "sectors.json")
        save_registry(path, data)
        loaded = load_registry(path)
        assert loaded["sectors"]["1"]["cursor"] == 42

    def test_save_is_atomic(self, tmp_path):
        """save_registry must not leave a .tmp file behind."""
        from exonit.discovery_state import save_registry
        path = tmp_path / "sectors.json"
        save_registry(str(path), {"active_sector": 1, "sectors": {}})
        assert path.exists()
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []


class TestGetSector:
    def test_returns_entry(self, tmp_path):
        from exonit.discovery_state import get_sector
        data = {"sectors": {"1": {"enabled": True, "cursor": 100}}}
        entry = get_sector(data, 1)
        assert entry["cursor"] == 100

    def test_returns_none_for_missing_sector(self):
        from exonit.discovery_state import get_sector
        data = {"sectors": {}}
        assert get_sector(data, 99) is None


class TestSetCursor:
    def test_updates_cursor(self, tmp_path):
        from exonit.discovery_state import set_cursor, load_registry
        path = tmp_path / "sectors.json"
        path.write_text(json.dumps({
            "active_sector": 1,
            "sectors": {"1": {"cursor": 0, "enabled": True}},
        }))
        set_cursor(str(path), 1, 1500)
        reg = load_registry(str(path))
        assert reg["sectors"]["1"]["cursor"] == 1500

    def test_creates_sector_if_absent(self, tmp_path):
        from exonit.discovery_state import set_cursor, load_registry
        path = tmp_path / "sectors.json"
        path.write_text(json.dumps({"active_sector": 1, "sectors": {}}))
        set_cursor(str(path), 2, 999)
        reg = load_registry(str(path))
        assert reg["sectors"]["2"]["cursor"] == 999


def test_set_deep_cursor_independent_of_cursor(tmp_path):
    from exonit.discovery_state import (load_registry, set_cursor,
                                        set_deep_cursor)
    reg = tmp_path / "sectors.json"
    set_cursor(str(reg), 14, 500)
    set_deep_cursor(str(reg), 14, 42)
    data = load_registry(str(reg))
    assert data["sectors"]["14"]["cursor"] == 500
    assert data["sectors"]["14"]["deep_cursor"] == 42


class TestDeriveDoneSet:
    def test_returns_only_done(self, tmp_path):
        from exonit.discovery_state import derive_done_set
        db = tmp_path / "test.db"
        _make_tasks_db(db, done_tics=[10, 20, 30], pending_tics=[40, 50])
        result = derive_done_set(str(db), [10, 20, 30, 40, 50, 99])
        assert result == {10, 20, 30}

    def test_empty_input(self, tmp_path):
        from exonit.discovery_state import derive_done_set
        db = tmp_path / "test.db"
        _make_tasks_db(db, done_tics=[10], pending_tics=[])
        result = derive_done_set(str(db), [])
        assert result == set()

    def test_no_done_tasks(self, tmp_path):
        from exonit.discovery_state import derive_done_set
        db = tmp_path / "test.db"
        _make_tasks_db(db, done_tics=[], pending_tics=[10, 20])
        result = derive_done_set(str(db), [10, 20])
        assert result == set()

    def test_only_completed_status_counts(self, tmp_path):
        """status='assigned'/'pending'/'failed' must NOT be treated as done.

        This test verifies the exact status string used by the production
        db.py::mark_task_completed path (status='completed').
        """
        from exonit.discovery_state import derive_done_set
        db = tmp_path / "status_test.db"
        con = sqlite3.connect(str(db))
        con.execute(
            "CREATE TABLE tasks ("
            "task_id TEXT PRIMARY KEY, tic_id TEXT, status TEXT)"
        )
        # One of each non-'completed' status
        con.executemany(
            "INSERT INTO tasks VALUES (?, ?, ?)",
            [
                ("t1", "1", "pending"),
                ("t2", "2", "assigned"),
                ("t3", "3", "failed"),
                ("t4", "4", "completed"),   # only this one should match
            ],
        )
        con.commit()
        con.close()
        result = derive_done_set(str(db), [1, 2, 3, 4])
        assert result == {4}, (
            "Only status='completed' should be returned; got: %r" % result
        )

    def test_db_missing_returns_empty(self, tmp_path):
        """If DB doesn't exist, derive_done_set should fail — but in
        discovery_run.py we guard with Path.exists(), so here we just confirm
        the function raises rather than silently returns empty when called
        directly on a missing file."""
        from exonit.discovery_state import derive_done_set
        # sqlite3.connect creates the file; derive_done_set with no table
        # will raise OperationalError — that's the expected behaviour.
        # We test with an empty (schema-less) db.
        db = tmp_path / "empty.db"
        db.touch()
        with pytest.raises(Exception):
            derive_done_set(str(db), [1, 2, 3])


class TestNextBlock:
    def test_basic_ordering(self):
        from exonit.discovery_state import next_block
        tics = [1, 2, 3, 4, 5, 6, 7, 8]
        block = next_block(tics, count=3, downloaded_set={1, 2, 3, 4}, done_set=set())
        assert block["selected"] == [1, 2, 3]
        assert block["ready"] == [1, 2, 3]
        assert block["missing_download"] == []
        assert block["remaining_undone"] == 8

    def test_skips_done(self):
        from exonit.discovery_state import next_block
        tics = [1, 2, 3, 4, 5]
        block = next_block(tics, count=3, downloaded_set={1, 2, 3, 4, 5}, done_set={1, 3})
        assert block["selected"] == [2, 4, 5]
        assert block["remaining_undone"] == 3  # 2, 4, 5

    def test_ready_missing_partition(self):
        from exonit.discovery_state import next_block
        tics = [10, 20, 30, 40, 50]
        block = next_block(
            tics, count=4,
            downloaded_set={10, 30},  # 20, 40 not downloaded
            done_set=set(),
        )
        assert set(block["ready"]) == {10, 30}
        assert set(block["missing_download"]) == {20, 40}
        assert block["remaining_undone"] == 5

    def test_count_larger_than_remaining(self):
        from exonit.discovery_state import next_block
        tics = [1, 2, 3]
        block = next_block(tics, count=10, downloaded_set={1, 2, 3}, done_set=set())
        assert block["selected"] == [1, 2, 3]
        assert block["remaining_undone"] == 3

    def test_all_done(self):
        from exonit.discovery_state import next_block
        tics = [1, 2, 3]
        block = next_block(tics, count=5, downloaded_set={1, 2, 3}, done_set={1, 2, 3})
        assert block["selected"] == []
        assert block["remaining_undone"] == 0


# ------------------------------------------------------------------ #
# load_extended_canonical unit tests
# ------------------------------------------------------------------ #

class TestLoadExtendedCanonical:
    """Unit tests for discovery_state.load_extended_canonical."""

    def test_both_present_overlapping(self, tmp_path):
        """Frozen order preserved; extension is non-overlapping TICs in rowid order."""
        from exonit.discovery_state import load_extended_canonical

        canon = tmp_path / "canonical.csv"
        _make_canonical_csv(canon, [10, 20, 30])  # frozen: 10, 20, 30

        tglc_db = tmp_path / "tglc.db"
        # rowid 1→10, 2→20, 3→30, 4→40, 5→50  (10/20/30 overlap; 40/50 are new)
        _make_tglc_db(tglc_db, [10, 20, 30, 40, 50])

        result = load_extended_canonical(str(canon), str(tglc_db))

        assert result[:3] == [10, 20, 30], "Canonical order must be preserved"
        assert result[3:] == [40, 50], "Extension must append non-overlapping in rowid order"
        assert len(result) == 5

    def test_csv_missing_returns_index_only(self, tmp_path):
        """When canonical CSV is absent, result = all TGLC TICs in rowid order."""
        from exonit.discovery_state import load_extended_canonical

        tglc_db = tmp_path / "tglc.db"
        _make_tglc_db(tglc_db, [7, 8, 9])

        result = load_extended_canonical(
            str(tmp_path / "no_such_canonical.csv"),
            str(tglc_db),
        )
        assert result == [7, 8, 9]

    def test_tglc_db_missing_returns_canonical_only(self, tmp_path):
        """When TGLC index is absent, result = canonical rows only."""
        from exonit.discovery_state import load_extended_canonical

        canon = tmp_path / "canonical.csv"
        _make_canonical_csv(canon, [1, 2, 3])

        result = load_extended_canonical(
            str(canon),
            str(tmp_path / "no_such_tglc.db"),
        )
        assert result == [1, 2, 3]

    def test_sector_filter_restricts_extension(self, tmp_path):
        """sector=N restricts the TGLC extension to that sector's interleaved rows.

        The tglc rowid order mixes sectors; without the filter a sector-2 run
        would pick sector-1 TICs first. With it, only sector-2 TICs come back.
        """
        import sqlite3
        from exonit.discovery_state import load_extended_canonical

        tglc_db = tmp_path / "tglc.db"
        con = sqlite3.connect(str(tglc_db))
        con.execute(
            "CREATE TABLE tglc (tic_id TEXT, sector INTEGER, path TEXT, "
            "exptime REAL, tmag REAL, ra REAL, dec REAL, "
            "PRIMARY KEY(tic_id, sector))")
        # Interleaved rowids: s1, s2, s1, s2, s2
        for tic, sec in [(10, 1), (20, 2), (30, 1), (40, 2), (50, 2)]:
            con.execute("INSERT INTO tglc VALUES (?,?,?,?,?,?,?)",
                        (str(tic), sec, "/x.fits", 1800, 10, 0, 0))
        con.commit()
        con.close()

        no_canon = str(tmp_path / "none.csv")
        assert load_extended_canonical(no_canon, str(tglc_db), sector=2) == [20, 40, 50]
        assert load_extended_canonical(no_canon, str(tglc_db), sector=1) == [10, 30]
        # No sector → all rows in rowid order (back-compat).
        assert load_extended_canonical(no_canon, str(tglc_db)) == [10, 20, 30, 40, 50]

    def test_both_missing_returns_empty(self, tmp_path):
        """When both inputs are absent, result is empty list."""
        from exonit.discovery_state import load_extended_canonical

        result = load_extended_canonical(
            str(tmp_path / "no_canon.csv"),
            str(tmp_path / "no_tglc.db"),
        )
        assert result == []

    def test_no_new_tics_extension_empty(self, tmp_path):
        """When TGLC has no TICs beyond the canonical, extension is empty."""
        from exonit.discovery_state import load_extended_canonical

        canon = tmp_path / "canonical.csv"
        _make_canonical_csv(canon, [100, 200, 300])

        tglc_db = tmp_path / "tglc.db"
        _make_tglc_db(tglc_db, [100, 200, 300])  # same TICs, no extras

        result = load_extended_canonical(str(canon), str(tglc_db))
        assert result == [100, 200, 300]

    # ------------------------------------------------------------------ #
    # Q2 (perf review 2026-09-26): return_indexed_set reuses the same
    # tglc_tic_ids() dump instead of a second, separate query.
    # ------------------------------------------------------------------ #

    def test_return_indexed_set_false_is_unchanged_default(self, tmp_path):
        """Default (no kwarg) still returns a plain list — every existing
        caller (verify_sector_complete.py, tests) must be unaffected."""
        from exonit.discovery_state import load_extended_canonical

        canon = tmp_path / "canonical.csv"
        _make_canonical_csv(canon, [10, 20])
        tglc_db = tmp_path / "tglc.db"
        _make_tglc_db(tglc_db, [10, 20, 30])

        result = load_extended_canonical(str(canon), str(tglc_db))
        assert result == [10, 20, 30]
        assert isinstance(result, list)

    def test_return_indexed_set_matches_load_downloaded_set(self, tmp_path):
        """indexed_set must be byte-for-byte what
        scripts.discovery_run._load_downloaded_set (the old, separate query
        this replaces) returns for the same tglc_index/sector."""
        import sys as _sys
        from pathlib import Path as _Path
        from exonit.discovery_state import load_extended_canonical

        canon = tmp_path / "canonical.csv"
        _make_canonical_csv(canon, [10, 20])  # 10, 20 frozen; 30, 40 in tglc only
        tglc_db = tmp_path / "tglc.db"
        _make_tglc_db(tglc_db, [10, 20, 30, 40])

        result, indexed_set = load_extended_canonical(
            str(canon), str(tglc_db), return_indexed_set=True)

        assert result == [10, 20, 30, 40]
        assert indexed_set == {10, 20, 30, 40}

        # Cross-check against the standalone helper it replaces at the
        # discovery_run.py call site.
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "scripts"))
        import discovery_run as _dr
        assert indexed_set == _dr._load_downloaded_set(str(tglc_db))

    def test_return_indexed_set_sector_scoped(self, tmp_path):
        """With a sector filter, indexed_set only contains that sector's
        TICs — same contract as _load_downloaded_set(path, sector)."""
        import sqlite3
        from exonit.discovery_state import load_extended_canonical

        tglc_db = tmp_path / "tglc.db"
        con = sqlite3.connect(str(tglc_db))
        con.execute(
            "CREATE TABLE tglc (tic_id TEXT, sector INTEGER, path TEXT, "
            "exptime REAL, tmag REAL, ra REAL, dec REAL, "
            "PRIMARY KEY(tic_id, sector))")
        for tic, sec in [(10, 1), (20, 2), (30, 1), (40, 2)]:
            con.execute("INSERT INTO tglc VALUES (?,?,?,?,?,?,?)",
                        (str(tic), sec, "/x.fits", 1800, 10, 0, 0))
        con.commit()
        con.close()

        no_canon = str(tmp_path / "none.csv")
        _, indexed_s2 = load_extended_canonical(
            no_canon, str(tglc_db), sector=2, return_indexed_set=True)
        _, indexed_s1 = load_extended_canonical(
            no_canon, str(tglc_db), sector=1, return_indexed_set=True)

        assert indexed_s2 == {20, 40}
        assert indexed_s1 == {10, 30}

    def test_return_indexed_set_empty_when_tglc_missing(self, tmp_path):
        """A missing/unreadable TGLC index yields an empty indexed_set, same
        contract as _load_downloaded_set (never raises)."""
        from exonit.discovery_state import load_extended_canonical

        canon = tmp_path / "canonical.csv"
        _make_canonical_csv(canon, [1, 2, 3])

        result, indexed_set = load_extended_canonical(
            str(canon), str(tmp_path / "no_such_tglc.db"),
            return_indexed_set=True)

        assert result == [1, 2, 3]
        assert indexed_set == set()

    def test_duplicate_handling_in_canonical(self, tmp_path):
        """Duplicate TIC ids in the canonical CSV: only first occurrence is kept."""
        from exonit.discovery_state import load_extended_canonical
        import csv as _csv

        canon = tmp_path / "canonical.csv"
        # Write duplicates manually
        with open(canon, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["tic_id", "label"])
            for t in [5, 5, 6, 5, 7]:   # 5 appears 3 times
                w.writerow([t, "candidate"])

        tglc_db = tmp_path / "tglc.db"
        _make_tglc_db(tglc_db, [5, 6, 7, 8])  # 8 is new

        result = load_extended_canonical(str(canon), str(tglc_db))
        # 5 must appear exactly once; order: 5, 6, 7 (canonical) then 8 (extension)
        assert result.count(5) == 1
        assert result == [5, 6, 7, 8]

    def test_extension_order_matches_tglc_rowid(self, tmp_path):
        """Extension TICs are appended in tglc rowid ASC order (insertion order)."""
        from exonit.discovery_state import load_extended_canonical

        canon = tmp_path / "canonical.csv"
        _make_canonical_csv(canon, [1])  # only TIC 1 in canonical

        tglc_db = tmp_path / "tglc.db"
        # Insert in order 1 (existing), then new TICs 99, 2, 50
        # rowid 1→1, 2→99, 3→2, 4→50
        con = sqlite3.connect(str(tglc_db))
        con.execute(
            "CREATE TABLE tglc ("
            "tic_id TEXT, sector INTEGER, path TEXT, exptime REAL, "
            "tmag REAL, ra REAL, dec REAL, PRIMARY KEY(tic_id, sector))"
        )
        for t in [1, 99, 2, 50]:
            con.execute(
                "INSERT INTO tglc VALUES (?, 1, '/tmp/fake.fits', 1800, 10, 0, 0)",
                (str(t),),
            )
        con.commit()
        con.close()

        result = load_extended_canonical(str(canon), str(tglc_db))
        assert result == [1, 99, 2, 50], (
            "Extension must follow rowid ASC order: 99 before 2 before 50"
        )


# ------------------------------------------------------------------ #
# count_extended_canonical unit tests
# ------------------------------------------------------------------ #

class TestCountExtendedCanonical:
    """``count_extended_canonical`` must equal ``len(load_extended_canonical)``.

    The whole point of the function is to answer "how big is the work list?"
    without materialising it (the list version allocated four ~8M-element
    objects for sector 14 and produced multi-GB RSS spikes on the server,
    2026-07-27). A count that is FASTER but disagrees with the list would be
    worse than the problem it solves — the work-list size drives the UI's
    remaining-TICs figure — so every case here asserts the two against each
    other rather than against a hardcoded number.
    """

    def _both(self, tmp_path, canon_tics, tglc_rows, sector=None):
        """Return (count, len(list)) for the same inputs."""
        from exonit.discovery_state import (count_extended_canonical,
                                            load_extended_canonical)
        canon = tmp_path / "canonical.csv"
        if canon_tics is not None:
            _make_canonical_csv(canon, canon_tics)
        tglc_db = tmp_path / "tglc.db"
        if tglc_rows is not None:
            con = sqlite3.connect(str(tglc_db))
            con.execute(
                "CREATE TABLE tglc (tic_id TEXT, sector INTEGER, path TEXT, "
                "exptime REAL, tmag REAL, ra REAL, dec REAL)")
            for tic, sec in tglc_rows:
                con.execute("INSERT INTO tglc VALUES (?,?,?,?,?,?,?)",
                            (str(tic), sec, "/x.fits", 1800, 10, 0, 0))
            con.commit()
            con.close()
        return (count_extended_canonical(str(canon), str(tglc_db), sector),
                len(load_extended_canonical(str(canon), str(tglc_db), sector)))

    def test_overlapping_canonical_and_index(self, tmp_path):
        """Overlap counted once: 10/20 are in both, 40/50 extend."""
        count, length = self._both(
            tmp_path, [10, 20, 30],
            [(10, 1), (20, 1), (40, 1), (50, 1)])
        assert count == length == 5

    def test_no_canonical_is_pure_rowcount(self, tmp_path):
        """The sector-14/15/17 shape: no canonical list, so count == COUNT(*)."""
        count, length = self._both(
            tmp_path, None, [(7, 1), (8, 1), (9, 1)])
        assert count == length == 3

    def test_sector_filter_applies_to_both(self, tmp_path):
        count, length = self._both(
            tmp_path, None,
            [(10, 1), (20, 2), (30, 1), (40, 2), (50, 2)], sector=2)
        assert count == length == 3

    def test_duplicate_tglc_rows_counted_per_row(self, tmp_path):
        """A TIC indexed twice in the SAME sector is two work-list entries.

        The list version's extension comprehension does not dedupe among
        indexed TICs, so a COUNT(DISTINCT tic_id) here would silently
        under-count. Pinned so nobody "optimises" it into DISTINCT.
        """
        count, length = self._both(
            tmp_path, None, [(7, 1), (7, 1), (8, 1)])
        assert count == length == 3

    def test_duplicate_overlap_subtracted_per_row(self, tmp_path):
        """A canonical TIC indexed twice must subtract BOTH of its rows."""
        count, length = self._both(
            tmp_path, [7], [(7, 1), (7, 1), (8, 1)])
        assert count == length == 2  # frozen [7] + extension [8]

    def test_tglc_missing_returns_canonical_count(self, tmp_path):
        from exonit.discovery_state import (count_extended_canonical,
                                            load_extended_canonical)
        canon = tmp_path / "canonical.csv"
        _make_canonical_csv(canon, [1, 2, 3])
        missing = str(tmp_path / "no_such_tglc.db")
        assert count_extended_canonical(str(canon), missing) == 3
        assert len(load_extended_canonical(str(canon), missing)) == 3

    def test_both_missing_is_zero(self, tmp_path):
        from exonit.discovery_state import count_extended_canonical
        assert count_extended_canonical(
            str(tmp_path / "no_canon.csv"), str(tmp_path / "no_tglc.db")) == 0

    def test_canonical_duplicates_counted_once(self, tmp_path):
        count, length = self._both(
            tmp_path, [5, 5, 6], [(5, 1), (7, 1)])
        assert count == length == 3  # frozen {5,6} + extension [7]

    def test_chunked_in_list_matches(self, tmp_path):
        """Overlap chunking must not double-count or drop rows.

        Uses a chunk_size small enough to force several round trips without
        writing 20k rows in a unit test.
        """
        from exonit import discovery_state
        from exonit.metadata_cache import MetadataCache
        real = MetadataCache.tglc_rowcount_for_tics

        def chunked_by_3(self, tic_ids, sector=None):
            tics = list(tic_ids)
            total = 0
            for i in range(0, len(tics), 3):
                total += real(self, tics[i:i + 3], sector)
            return total

        MetadataCache.tglc_rowcount_for_tics = chunked_by_3
        try:
            canon_tics = list(range(100, 110))          # 10 canonical TICs
            rows = [(t, 1) for t in range(100, 115)]    # 10 overlap + 5 new
            count, length = self._both(tmp_path, canon_tics, rows)
            assert count == length == 15
        finally:
            MetadataCache.tglc_rowcount_for_tics = real


# ------------------------------------------------------------------ #
# discovery_run.py subprocess tests (--dry-run only)
# ------------------------------------------------------------------ #

SCRIPT = str(ROOT / "scripts" / "discovery_run.py")


def _run(args: list) -> subprocess.CompletedProcess:
    cmd = [sys.executable, SCRIPT] + args
    # This test file lives in a git worktree (deep-pass-multisector). The
    # `exonit` package is also pip-installed editable pointing at the main
    # repo (~/exonit) — without PYTHONPATH, a bare subprocess spawn
    # resolves `import exonit` to THAT install, not this worktree's copy,
    # silently exercising the wrong code. Force this worktree's ROOT first.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


class TestDiscoveryRunSectorNotEnabled:
    def test_sector_2_exits_2(self, tmp_path):
        """Sector 2 is not in the registry → exit 2, JSON error."""
        registry = tmp_path / "sectors.json"
        canonical = tmp_path / "canonical.csv"
        _make_canonical_csv(canonical, [1, 2, 3])
        _make_registry(registry, str(canonical))

        result = _run([
            "--sector", "2",
            "--count", "2",
            "--registry", str(registry),
            "--canonical", str(canonical),
            "--tglc-index", str(tmp_path / "tglc.db"),
            "--db", str(tmp_path / "ex.db"),
            "--dry-run",
        ])
        assert result.returncode == 2
        out = json.loads(result.stdout)
        assert out["status"] == "error"
        assert out["reason"] == "sector_not_found"
        assert out["sector"] == 2

    def test_sector_disabled_exits_2(self, tmp_path):
        """Sector with enabled=false → exit 2."""
        registry = tmp_path / "sectors.json"
        canonical = tmp_path / "canonical.csv"
        _make_canonical_csv(canonical, [1, 2, 3])
        data = {
            "active_sector": 1,
            "sectors": {
                "1": {
                    "target_list": "x",
                    "canonical_list": str(canonical),
                    "total_tics": 3,
                    "downloaded": 3,
                    "cursor": 0,
                    "enabled": False,  # <-- disabled
                }
            },
        }
        registry.write_text(json.dumps(data))

        result = _run([
            "--sector", "1",
            "--count", "2",
            "--registry", str(registry),
            "--canonical", str(canonical),
            "--tglc-index", str(tmp_path / "tglc.db"),
            "--db", str(tmp_path / "ex.db"),
            "--dry-run",
        ])
        assert result.returncode == 2
        out = json.loads(result.stdout)
        assert out["reason"] == "sector_not_enabled"


class TestDiscoveryRunNeedDownload:
    def test_exits_3_when_tglc_missing(self, tmp_path):
        """When some of the next-N TICs are not in the TGLC index → exit 3."""
        registry = tmp_path / "sectors.json"
        canonical = tmp_path / "canonical.csv"
        tglc_db = tmp_path / "tglc.db"
        tasks_db = tmp_path / "ex.db"

        tics = list(range(100, 110))  # 10 TICs
        _make_canonical_csv(canonical, tics)
        _make_registry(registry, str(canonical))
        # Only TICs 100, 101 are downloaded; rest are not
        _make_tglc_db(tglc_db, [100, 101])
        _make_tasks_db(tasks_db, done_tics=[], pending_tics=[])

        result = _run([
            "--sector", "1",
            "--count", "5",  # want 5, but only 2 downloaded
            "--registry", str(registry),
            "--canonical", str(canonical),
            "--tglc-index", str(tglc_db),
            "--db", str(tasks_db),
            "--dry-run",
        ])
        assert result.returncode == 3
        out = json.loads(result.stdout)
        assert out["status"] == "need_download"
        assert out["ready"] == 2
        assert out["missing"] == 3


class TestDiscoveryRunHappyPath:
    def test_dry_run_returns_plan(self, tmp_path):
        """Happy path: all TICs downloaded, --dry-run returns plan JSON.

        canonical = [200..214], done = {200, 201}, count = 5
        selected  = [202, 203, 204, 205, 206]
        last selected TIC = 206, canonical index (0-based) = 6 → cursor_after = 7
        """
        registry = tmp_path / "sectors.json"
        canonical = tmp_path / "canonical.csv"
        tglc_db = tmp_path / "tglc.db"
        tasks_db = tmp_path / "ex.db"

        tics = list(range(200, 215))  # 15 TICs
        _make_canonical_csv(canonical, tics)
        _make_registry(registry, str(canonical), cursor=0)
        # All downloaded
        _make_tglc_db(tglc_db, tics)
        # TICs 200, 201 already done
        _make_tasks_db(tasks_db, done_tics=[200, 201], pending_tics=[])

        result = _run([
            "--sector", "1",
            "--count", "5",
            "--registry", str(registry),
            "--canonical", str(canonical),
            "--tglc-index", str(tglc_db),
            "--db", str(tasks_db),
            "--dry-run",
        ])
        assert result.returncode == 0, result.stderr
        out = json.loads(result.stdout)
        assert out["status"] == "dry_run"
        assert out["enqueued"] == 5  # 5 not-done TICs selected, all ready
        # Should skip 200, 201 (done); selected starts at 202
        assert out["tics_selected"][0] == 202
        assert out["tics_selected"][1] == 203
        # Cursor must be the canonical row index (1-based) of the LAST selected TIC.
        # Last selected = 206, canonical position = index 6 + 1 = 7.
        # Must NOT be old_cursor + len(selected) = 0 + 5 = 5.
        assert out["cursor_after"] == 7, (
            "cursor_after must be canonical row index of last selected TIC "
            f"(expected 7, got {out['cursor_after']})"
        )

    def test_cursor_canonical_index_interleaved_done(self, tmp_path):
        """Cursor must track canonical row position, not count of selected TICs.

        canonical = [1, 2, 3, 4, 5], done = {2}, count = 3
        next_block selects [1, 3, 4]  (skips 2)
        last selected TIC = 4 → canonical index 3 (0-based) → cursor_after = 4
        Wrong arithmetic would give: 0 + 3 = 3  (off by one, misses the gap)
        """
        registry = tmp_path / "sectors.json"
        canonical = tmp_path / "canonical.csv"
        tglc_db = tmp_path / "tglc.db"
        tasks_db = tmp_path / "ex.db"

        tics = [1, 2, 3, 4, 5]
        _make_canonical_csv(canonical, tics)
        _make_registry(registry, str(canonical), cursor=0)
        _make_tglc_db(tglc_db, tics)
        # TIC 2 is interleaved as done
        _make_tasks_db(tasks_db, done_tics=[2], pending_tics=[])

        result = _run([
            "--sector", "1",
            "--count", "3",
            "--registry", str(registry),
            "--canonical", str(canonical),
            "--tglc-index", str(tglc_db),
            "--db", str(tasks_db),
            "--dry-run",
        ])
        assert result.returncode == 0, result.stderr
        out = json.loads(result.stdout)
        assert out["status"] == "dry_run"
        # selected = [1, 3, 4]; last = 4; canonical row 4 (1-based) → cursor_after=4
        assert out["cursor_after"] == 4, (
            "cursor_after must be canonical row index (4), "
            f"not count-based (3). Got: {out['cursor_after']}"
        )
        assert out["tics_selected"] == [1, 3, 4]

    def test_dry_run_no_registry_write(self, tmp_path):
        """--dry-run must not mutate the registry."""
        registry = tmp_path / "sectors.json"
        canonical = tmp_path / "canonical.csv"
        tglc_db = tmp_path / "tglc.db"
        tasks_db = tmp_path / "ex.db"

        tics = list(range(300, 310))
        _make_canonical_csv(canonical, tics)
        _make_registry(registry, str(canonical), cursor=5)
        _make_tglc_db(tglc_db, tics)
        _make_tasks_db(tasks_db, done_tics=[], pending_tics=[])

        before = json.loads(registry.read_text())

        _run([
            "--sector", "1",
            "--count", "3",
            "--registry", str(registry),
            "--canonical", str(canonical),
            "--tglc-index", str(tglc_db),
            "--db", str(tasks_db),
            "--dry-run",
        ])

        after = json.loads(registry.read_text())
        assert before == after, "registry should not change under --dry-run"


class TestDiscoveryRunExtendedWorkList:
    """Integration test: canonical=4 TICs + tglc has those 4 plus 6 more.

    discovery_run.py --count 10 --dry-run must produce a lot that contains
    all 10 TICs in the correct order (canonical first, then extended in
    tglc rowid ASC order).
    """

    def test_extended_list_all_selected(self, tmp_path):
        """Extended work list: canonical 4 + 6 tglc extras → count=10 selects all."""
        registry = tmp_path / "sectors.json"
        canonical = tmp_path / "canonical.csv"
        tglc_db = tmp_path / "tglc.db"
        tasks_db = tmp_path / "ex.db"

        canonical_tics = [1001, 1002, 1003, 1004]
        extra_tics = [2001, 2002, 2003, 2004, 2005, 2006]
        all_tics = canonical_tics + extra_tics  # expected order

        _make_canonical_csv(canonical, canonical_tics)
        _make_registry(registry, str(canonical), cursor=0)
        # tglc DB contains both canonical and extra TICs, in rowid order
        _make_tglc_db(tglc_db, all_tics)
        # No TICs done yet
        _make_tasks_db(tasks_db, done_tics=[], pending_tics=[])

        result = _run([
            "--sector", "1",
            "--count", "10",
            "--registry", str(registry),
            "--canonical", str(canonical),
            "--tglc-index", str(tglc_db),
            "--db", str(tasks_db),
            "--dry-run",
        ])
        assert result.returncode == 0, result.stderr
        out = json.loads(result.stdout)
        assert out["status"] == "dry_run"
        assert out["enqueued"] == 10, (
            f"Expected 10 TICs enqueued (4 canonical + 6 extended), got {out['enqueued']}"
        )
        # First TICs in output must be the canonical ones, then the extras
        selected = out["tics_selected"]  # first 10 are shown
        assert selected[:4] == [1001, 1002, 1003, 1004], (
            "First 4 must be canonical TICs in canonical order"
        )
        assert selected[4:] == [2001, 2002, 2003, 2004, 2005, 2006], (
            "Last 6 must be extended TICs in tglc rowid order"
        )

    def test_extended_list_canonical_done_skipped(self, tmp_path):
        """When canonical TICs are done, extended TICs fill in."""
        registry = tmp_path / "sectors.json"
        canonical = tmp_path / "canonical.csv"
        tglc_db = tmp_path / "tglc.db"
        tasks_db = tmp_path / "ex.db"

        canonical_tics = [1001, 1002, 1003, 1004]
        extra_tics = [2001, 2002]

        _make_canonical_csv(canonical, canonical_tics)
        _make_registry(registry, str(canonical), cursor=0)
        _make_tglc_db(tglc_db, canonical_tics + extra_tics)
        # All canonical TICs already done
        _make_tasks_db(tasks_db, done_tics=canonical_tics, pending_tics=[])

        result = _run([
            "--sector", "1",
            "--count", "5",
            "--registry", str(registry),
            "--canonical", str(canonical),
            "--tglc-index", str(tglc_db),
            "--db", str(tasks_db),
            "--dry-run",
        ])
        assert result.returncode == 0, result.stderr
        out = json.loads(result.stdout)
        assert out["status"] == "dry_run"
        # Only 2 not-done TICs remain (the extras)
        assert out["enqueued"] == 2
        assert out["tics_selected"] == [2001, 2002]

    def test_extended_work_list_info_in_stderr(self, tmp_path):
        """discovery_run.py logs the work list breakdown on stderr."""
        registry = tmp_path / "sectors.json"
        canonical = tmp_path / "canonical.csv"
        tglc_db = tmp_path / "tglc.db"
        tasks_db = tmp_path / "ex.db"

        _make_canonical_csv(canonical, [1, 2])
        _make_registry(registry, str(canonical), cursor=0)
        _make_tglc_db(tglc_db, [1, 2, 3, 4])  # 2 canonical + 2 extra
        _make_tasks_db(tasks_db, done_tics=[], pending_tics=[])

        result = _run([
            "--sector", "1",
            "--count", "4",
            "--registry", str(registry),
            "--canonical", str(canonical),
            "--tglc-index", str(tglc_db),
            "--db", str(tasks_db),
            "--dry-run",
        ])
        assert result.returncode == 0, result.stderr
        assert "work list" in result.stderr.lower() or "INFO" in result.stderr, (
            "Expected INFO log about work list breakdown on stderr"
        )


# ------------------------------------------------------------------ #
# Deep-rescan mode (Task 6, spec 2026-07-23 §4)
#
# --tics-csv + --force-rescan slice a pre-cleaned worklist CSV by the
# sector's OWN deep_cursor (independent of the campaign cursor). These
# tests call scripts.discovery_run.main() IN-PROCESS (not via the _run()
# subprocess wrapper used above) because we need to:
#   - monkeypatch subprocess.run to intercept the enqueue_from_csv.py call
#     (capture argv) without actually shelling out or touching a real DB —
#     the same pattern tests/test_discovery_orchestrator.py uses for
#     scripts that shell out (patch("subprocess.run", ...)).
#   - monkeypatch _append_campaign_log so no test ever appends to the real
#     data/discovery/CAMPAIGN.md.
#   - monkeypatch ROOT so the default (non---out) lot-csv path lands under
#     tmp_path instead of the real repo's data/discovery/.
# ------------------------------------------------------------------ #

class TestDeepMode:
    def _worklist(self, tmp_path, tics):
        path = tmp_path / "deep_worklist.csv"
        _make_canonical_csv(path, tics)
        return path

    def test_deep_mode_slices_by_deep_cursor_and_advances_it(self, tmp_path, monkeypatch):
        """worklist of 10 TICs, deep_cursor=4, --count 3
        -> selects rows 5,6,7 (1-based); deep_cursor_after == 7;
           sector_entry["cursor"] INTACT; enqueue_from_csv called with
           --force-rescan (args captured); lot csv starts with
           s0014_deep_lot_.
        """
        from scripts import discovery_run

        tics = list(range(9000, 9010))  # 10 TICs
        worklist = self._worklist(tmp_path, tics)
        registry = tmp_path / "sectors.json"
        _make_deep_registry(registry, sector=14, cursor=500, deep_cursor=4)

        monkeypatch.setattr(discovery_run, "ROOT", tmp_path)

        captured_subprocess = {}

        def fake_run(cmd, **kwargs):
            captured_subprocess["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(discovery_run.subprocess, "run", fake_run)

        captured_log = {}

        def fake_log(sector, lot_csv, first_row, last_row, tag="lot-auto"):
            captured_log.update(sector=sector, lot_csv=lot_csv,
                                 first_row=first_row, last_row=last_row, tag=tag)

        monkeypatch.setattr(discovery_run, "_append_campaign_log", fake_log)

        plan = {}
        monkeypatch.setattr(discovery_run, "_emit", lambda obj: plan.update(obj))

        rc = discovery_run.main([
            "--sector", "14",
            "--count", "3",
            "--registry", str(registry),
            "--tics-csv", str(worklist),
            "--force-rescan",
            "--db", str(tmp_path / "ex.db"),
        ])

        assert rc == 0, plan
        assert plan["mode"] == "deep_rescan"
        assert plan["sector"] == 14
        assert plan["deep_cursor_before"] == 4
        assert plan["deep_cursor_after"] == 7
        assert plan["tics_selected"] == [9004, 9005, 9006]
        assert Path(plan["lot_csv"]).name.startswith("s0014_deep_lot_"), (
            f"lot_csv basename must start with s0014_deep_lot_, got {plan['lot_csv']!r}"
        )

        # enqueue_from_csv.py must have been called WITH --force-rescan
        cmd = captured_subprocess["cmd"]
        assert "--force-rescan" in cmd
        assert "--csv" in cmd
        assert cmd[cmd.index("--csv") + 1] == plan["lot_csv"]

        # CAMPAIGN.md row tagged lot-deep, rows 5-7 (1-based, deep_cursor+1..new)
        assert captured_log["tag"] == "lot-deep"
        assert captured_log["first_row"] == 5
        assert captured_log["last_row"] == 7
        assert captured_log["sector"] == 14

        # Registry: campaign cursor untouched, deep_cursor advanced
        reg_after = json.loads(registry.read_text())
        assert reg_after["sectors"]["14"]["cursor"] == 500, (
            "campaign cursor must never be touched by deep mode"
        )
        assert reg_after["sectors"]["14"]["deep_cursor"] == 7

    def test_deep_mode_does_not_skip_done(self, tmp_path, monkeypatch):
        """done_set contains ALL TICs (deep-rescan targets are completed by
        definition) -> --count TICs are STILL selected; done/TOI/downloaded
        checks are not applied in deep mode (the worklist arrives pre-cleaned).
        """
        from scripts import discovery_run

        tics = list(range(9100, 9105))  # 5 TICs
        worklist = self._worklist(tmp_path, tics)
        registry = tmp_path / "sectors.json"
        _make_deep_registry(registry, sector=14, cursor=0, deep_cursor=0)

        tasks_db = tmp_path / "ex.db"
        _make_tasks_db(tasks_db, done_tics=tics, pending_tics=[])  # ALL done

        monkeypatch.setattr(discovery_run, "ROOT", tmp_path)
        monkeypatch.setattr(
            discovery_run.subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        )
        monkeypatch.setattr(discovery_run, "_append_campaign_log", lambda *a, **k: None)

        plan = {}
        monkeypatch.setattr(discovery_run, "_emit", lambda obj: plan.update(obj))

        rc = discovery_run.main([
            "--sector", "14",
            "--count", "5",
            "--registry", str(registry),
            "--tics-csv", str(worklist),
            "--force-rescan",
            "--db", str(tasks_db),
        ])

        assert rc == 0, plan
        assert plan["mode"] == "deep_rescan"
        assert plan["enqueued"] == 5
        assert plan["tics_selected"] == tics
        assert plan["deep_cursor_after"] == 5

    def test_deep_mode_exhausted_worklist(self, tmp_path, monkeypatch):
        """deep_cursor already at the end of the worklist
        -> {"status": "error", "reason": "deep_worklist_exhausted"}, rc=2;
        no enqueue attempt; registry (deep_cursor) untouched.
        """
        from scripts import discovery_run

        tics = list(range(9200, 9205))  # 5 TICs
        worklist = self._worklist(tmp_path, tics)
        registry = tmp_path / "sectors.json"
        _make_deep_registry(registry, sector=14, cursor=0, deep_cursor=5)  # exhausted

        monkeypatch.setattr(discovery_run, "ROOT", tmp_path)
        called = {"subprocess": False}
        monkeypatch.setattr(
            discovery_run.subprocess, "run",
            lambda *a, **k: called.__setitem__("subprocess", True),
        )

        plan = {}
        monkeypatch.setattr(discovery_run, "_emit", lambda obj: plan.update(obj))

        rc = discovery_run.main([
            "--sector", "14",
            "--count", "3",
            "--registry", str(registry),
            "--tics-csv", str(worklist),
            "--force-rescan",
            "--db", str(tmp_path / "ex.db"),
        ])

        assert rc == 2
        assert plan == {
            "status": "error",
            "reason": "deep_worklist_exhausted",
            "deep_cursor": 5,
            "worklist_size": 5,
        }
        assert called["subprocess"] is False

        reg_after = json.loads(registry.read_text())
        assert reg_after["sectors"]["14"]["deep_cursor"] == 5

    def test_normal_mode_unchanged(self, tmp_path):
        """Without --tics-csv: JSON identical to the pre-Task-6 contract —
        no deep_* keys, no "mode" key. Exact key-set assertion locks the
        normal-mode dry-run plan shape.
        """
        registry = tmp_path / "sectors.json"
        canonical = tmp_path / "canonical.csv"
        tglc_db = tmp_path / "tglc.db"
        tasks_db = tmp_path / "ex.db"

        tics = list(range(400, 410))
        _make_canonical_csv(canonical, tics)
        _make_registry(registry, str(canonical), cursor=0)
        _make_tglc_db(tglc_db, tics)
        _make_tasks_db(tasks_db, done_tics=[], pending_tics=[])

        result = _run([
            "--sector", "1",
            "--count", "3",
            "--registry", str(registry),
            "--canonical", str(canonical),
            "--tglc-index", str(tglc_db),
            "--db", str(tasks_db),
            "--dry-run",
        ])
        assert result.returncode == 0, result.stderr
        out = json.loads(result.stdout)
        assert out["status"] == "dry_run"
        assert "mode" not in out
        for key in ("deep_cursor_before", "deep_cursor_after", "remaining_deep"):
            assert key not in out
        assert set(out.keys()) == {
            "status", "sector", "requested", "enqueued", "lot_csv", "run_id",
            "cursor_before", "cursor_after", "remaining_undone",
            "tics_selected", "tois_skipped",
        }


# ------------------------------------------------------------------ #
# ENQUEUE_SUMMARY_JSON wiring (2026-08 empty-lot-loop fix)
#
# scripts/enqueue_from_csv.py now emits a machine-parseable
# "ENQUEUE_SUMMARY_JSON: {...}" line on stdout with the tasks ACTUALLY
# created by exonit.enqueue.enqueue_targets — which additionally skips
# TICs already in `stars` (analyzed_set) or without metadata (no_fits),
# filters that derive_done_set/next_block above do not apply. Without
# this, `plan["enqueued"]` reported len(ready)/len(selected) even when
# the real enqueue created ZERO task rows, and the caller (autonomy
# supervisor / UI) had no way to tell an empty lot from a real one —
# the root cause of the sector-2 empty-lot loop (8 identical lots,
# 0 tasks created, run "started" and DONE in 13 min each time).
#
# These tests call scripts.discovery_run.main() IN-PROCESS (same pattern
# as TestDeepMode above) so subprocess.run can be monkeypatched to return
# a fake enqueue_from_csv.py stdout without shelling out or touching the
# live DB/server.
# ------------------------------------------------------------------ #

def _summary_stdout(enqueued, skipped=0, failed=0, extra_lines=()):
    """Build a fake enqueue_from_csv.py stdout: human print(summary) line
    (dict repr, NOT valid JSON — matches production) plus the parseable
    ENQUEUE_SUMMARY_JSON line, exactly as scripts/enqueue_from_csv.py emits
    it (see the print(summary) immediately followed by the JSON line)."""
    import json as _json
    summary = {"enqueued": enqueued, "skipped": skipped, "failed": failed,
               "elapsed_seconds": 0.1}
    lines = list(extra_lines) + [
        f"Loaded {enqueued + skipped + failed} targets from lot.csv",
        str(summary),  # print(summary) — dict repr, unparseable as JSON
        "ENQUEUE_SUMMARY_JSON: " + _json.dumps(summary),
    ]
    return "\n".join(lines)


class TestSurveyEnqueueSummaryWiring:
    """Survey (normal) mode: plan["enqueued"] must reflect the real
    enqueue_from_csv.py summary, not len(ready)."""

    def _setup(self, tmp_path, cursor=0):
        registry = tmp_path / "sectors.json"
        canonical = tmp_path / "canonical.csv"
        tglc_db = tmp_path / "tglc.db"
        tasks_db = tmp_path / "ex.db"

        tics = list(range(200, 215))  # 15 TICs, none done -> next_block(count=5)
        _make_canonical_csv(canonical, tics)
        _make_registry(registry, str(canonical), cursor=cursor)
        _make_tglc_db(tglc_db, tics)
        _make_tasks_db(tasks_db, done_tics=[], pending_tics=[])
        return registry, canonical, tglc_db, tasks_db

    def _run_inprocess(self, tmp_path, monkeypatch, registry, canonical,
                        tglc_db, tasks_db, fake_stdout, count=5):
        from scripts import discovery_run

        monkeypatch.setattr(discovery_run, "ROOT", tmp_path)
        monkeypatch.setattr(
            discovery_run.subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(
                cmd, 0, stdout=fake_stdout, stderr=""),
        )
        log_calls = []
        monkeypatch.setattr(
            discovery_run, "_append_campaign_log",
            lambda *a, **k: log_calls.append((a, k)),
        )
        plan = {}
        monkeypatch.setattr(discovery_run, "_emit", lambda obj: plan.update(obj))

        rc = discovery_run.main([
            "--sector", "1",
            "--count", str(count),
            "--registry", str(registry),
            "--canonical", str(canonical),
            "--tglc-index", str(tglc_db),
            "--db", str(tasks_db),
            "--out", str(tmp_path / "lot.csv"),
        ])
        return rc, plan, log_calls

    def test_zero_real_enqueued_emits_no_undone_tics_and_cursor_untouched(
        self, tmp_path, monkeypatch
    ):
        """enqueue_from_csv.py reports enqueued=0 (all 5 selected TICs were
        already in `stars` or had no metadata) -> status error /
        reason no_undone_tics, rc=2, and the registry cursor must NOT move
        (a lot that created zero tasks never really happened)."""
        registry, canonical, tglc_db, tasks_db = self._setup(tmp_path, cursor=0)
        fake_stdout = _summary_stdout(enqueued=0, skipped=5, failed=0)

        rc, plan, log_calls = self._run_inprocess(
            tmp_path, monkeypatch, registry, canonical, tglc_db, tasks_db,
            fake_stdout,
        )

        assert rc == 2
        assert plan["status"] == "error"
        assert plan["reason"] == "no_undone_tics"
        assert plan["enqueue_skipped"] == 5
        assert plan["enqueue_failed"] == 0
        assert log_calls == [], "CAMPAIGN.md must not be appended for a 0-task lot"

        reg_after = json.loads(registry.read_text())
        assert reg_after["sectors"]["1"]["cursor"] == 0, (
            "cursor must not advance when the lot created zero real tasks"
        )

    def test_partial_real_enqueued_reported_and_cursor_advances(
        self, tmp_path, monkeypatch
    ):
        """enqueue_from_csv.py reports enqueued=3 (< len(ready)==5, some
        skipped) -> status started, plan["enqueued"] == 3 (the REAL count,
        not len(ready)), cursor advances exactly as in the pre-fix
        contract (based on canonical row position, unaffected by the real
        enqueue count)."""
        registry, canonical, tglc_db, tasks_db = self._setup(tmp_path, cursor=0)
        fake_stdout = _summary_stdout(enqueued=3, skipped=2, failed=0)

        rc, plan, log_calls = self._run_inprocess(
            tmp_path, monkeypatch, registry, canonical, tglc_db, tasks_db,
            fake_stdout,
        )

        assert rc == 0, plan
        assert plan["status"] == "started"
        assert plan["enqueued"] == 3
        assert plan["enqueue_skipped"] == 2
        assert plan["enqueue_failed"] == 0
        assert len(log_calls) == 1, "CAMPAIGN.md must be appended for a real lot"

        reg_after = json.loads(registry.read_text())
        # selected = [200..204] (5 TICs, none done); last=204 -> canonical
        # row index (0-based 4) + 1 = 5.
        assert reg_after["sectors"]["1"]["cursor"] == 5

    def test_backward_compat_no_summary_line_falls_back_to_len_ready(
        self, tmp_path, monkeypatch
    ):
        """An older enqueue_from_csv.py that doesn't emit
        ENQUEUE_SUMMARY_JSON must not break discovery_run.py: falls back
        to the pre-fix len(ready) behaviour, status started, cursor
        advances as before."""
        registry, canonical, tglc_db, tasks_db = self._setup(tmp_path, cursor=0)
        # No ENQUEUE_SUMMARY_JSON line anywhere in stdout.
        fake_stdout = "Loaded 5 targets from lot.csv\n{'enqueued': 5, 'skipped': 0, 'failed': 0}\n"

        rc, plan, log_calls = self._run_inprocess(
            tmp_path, monkeypatch, registry, canonical, tglc_db, tasks_db,
            fake_stdout,
        )

        assert rc == 0, plan
        assert plan["status"] == "started"
        assert plan["enqueued"] == 5, "must fall back to len(ready) == 5"
        assert "enqueue_skipped" not in plan
        assert "enqueue_failed" not in plan
        assert len(log_calls) == 1

        reg_after = json.loads(registry.read_text())
        assert reg_after["sectors"]["1"]["cursor"] == 5


class TestSurveyProgressSubsteps:
    """Q9 (perf review 2026-09-26, docs/reviews/2026-09-26-enqueue-path-
    performance-review.md): the survey path must report progress substeps
    (build_work_list, done_set) BEFORE the ~90s work list/done-set build
    completes, not sit at a flat 0 until step 7's enqueue_from_csv mark."""

    def _setup(self, tmp_path, cursor=0):
        registry = tmp_path / "sectors.json"
        canonical = tmp_path / "canonical.csv"
        tglc_db = tmp_path / "tglc.db"
        tasks_db = tmp_path / "ex.db"

        tics = list(range(200, 215))
        _make_canonical_csv(canonical, tics)
        _make_registry(registry, str(canonical), cursor=cursor)
        _make_tglc_db(tglc_db, tics)
        _make_tasks_db(tasks_db, done_tics=[], pending_tics=[])
        return registry, canonical, tglc_db, tasks_db

    def test_substeps_reported_in_order_before_enqueue(self, tmp_path, monkeypatch):
        from scripts import discovery_run

        registry, canonical, tglc_db, tasks_db = self._setup(tmp_path)

        monkeypatch.setattr(discovery_run, "ROOT", tmp_path)
        monkeypatch.setattr(
            discovery_run.subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(
                cmd, 0, stdout=_summary_stdout(enqueued=5, skipped=0, failed=0),
                stderr=""),
        )
        monkeypatch.setattr(discovery_run, "_append_campaign_log", lambda *a, **k: None)
        monkeypatch.setattr(discovery_run, "_emit", lambda obj: None)

        substeps_seen: list = []
        real_report = discovery_run.ProgressReporter.report

        def _spy_report(self, done, *, total=None, message=None,
                        substep=None, force=False):
            effective = substep if substep is not None else self.substep
            substeps_seen.append(effective)
            return real_report(self, done, total=total, message=message,
                               substep=substep, force=force)

        monkeypatch.setattr(discovery_run.ProgressReporter, "report", _spy_report)
        # ProgressReporter is a no-op unless EXONIT_PROGRESS_RUN_ID/PHASE are
        # set (see its own docstring) -- set them so .report() actually runs
        # its body (still enabled=False re: the HTTP POST itself, since no
        # EXONIT_PROGRESS_URL is set here -- _send is never reached, only
        # the rate-limit/substep-tracking logic in report() itself).
        monkeypatch.setenv("EXONIT_PROGRESS_RUN_ID", "test-run")
        monkeypatch.setenv("EXONIT_PROGRESS_PHASE", "ENQUEUEING")
        monkeypatch.delenv("EXONIT_PROGRESS_URL", raising=False)
        monkeypatch.delenv("EXONIT_SERVER_URL", raising=False)

        rc = discovery_run.main([
            "--sector", "1",
            "--count", "5",
            "--registry", str(registry),
            "--canonical", str(canonical),
            "--tglc-index", str(tglc_db),
            "--db", str(tasks_db),
        ])

        assert rc == 0
        # build_work_list and done_set must both appear, in that order,
        # before enqueue_from_csv -- the whole point of Q9 (a substep
        # change is always sent immediately, per ProgressReporter.report's
        # own docstring, so ordering here is exactly what the dashboard
        # would see).
        assert "build_work_list" in substeps_seen
        assert "done_set" in substeps_seen
        assert "enqueue_from_csv" in substeps_seen
        assert (substeps_seen.index("build_work_list")
                < substeps_seen.index("done_set")
                < substeps_seen.index("enqueue_from_csv"))

    def test_progress_reporter_disabled_when_not_launched_as_part_of_a_run(
        self, tmp_path, monkeypatch
    ):
        """No EXONIT_PROGRESS_RUN_ID/PHASE (interactive CLI use, or a test
        that doesn't set them) -- ProgressReporter.enabled is False and the
        substep calls are pure no-ops, exactly like every existing
        discovery_run.py test (none of which set these env vars) already
        implicitly relies on."""
        from scripts import discovery_run

        registry, canonical, tglc_db, tasks_db = self._setup(tmp_path)
        monkeypatch.delenv("EXONIT_PROGRESS_RUN_ID", raising=False)
        monkeypatch.delenv("EXONIT_PROGRESS_PHASE", raising=False)

        monkeypatch.setattr(discovery_run, "ROOT", tmp_path)
        monkeypatch.setattr(
            discovery_run.subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(
                cmd, 0, stdout=_summary_stdout(enqueued=5, skipped=0, failed=0),
                stderr=""),
        )
        monkeypatch.setattr(discovery_run, "_append_campaign_log", lambda *a, **k: None)
        plan = {}
        monkeypatch.setattr(discovery_run, "_emit", lambda obj: plan.update(obj))

        rc = discovery_run.main([
            "--sector", "1",
            "--count", "5",
            "--registry", str(registry),
            "--canonical", str(canonical),
            "--tglc-index", str(tglc_db),
            "--db", str(tasks_db),
        ])

        assert rc == 0, plan
        assert plan["status"] == "started"


class TestDeepEnqueueSummaryWiring:
    """Deep-rescan mode: zero real creates must end in a DIFFERENT reason
    (no_tasks_created, not no_undone_tics) -- build_deep_pass_worklist.py
    pre-cleans the worklist, so 0 real creates here is an anomaly the
    supervisor must pause+alert on, not a naturally exhausted plan."""

    def test_zero_real_enqueued_emits_no_tasks_created_and_deep_cursor_untouched(
        self, tmp_path, monkeypatch
    ):
        from scripts import discovery_run

        tics = list(range(9300, 9310))  # 10 TICs
        worklist = tmp_path / "deep_worklist.csv"
        _make_canonical_csv(worklist, tics)
        registry = tmp_path / "sectors.json"
        _make_deep_registry(registry, sector=14, cursor=500, deep_cursor=0)

        fake_stdout = _summary_stdout(enqueued=0, skipped=3, failed=0)

        monkeypatch.setattr(discovery_run, "ROOT", tmp_path)
        monkeypatch.setattr(
            discovery_run.subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(
                cmd, 0, stdout=fake_stdout, stderr=""),
        )
        log_calls = []
        monkeypatch.setattr(
            discovery_run, "_append_campaign_log",
            lambda *a, **k: log_calls.append((a, k)),
        )
        plan = {}
        monkeypatch.setattr(discovery_run, "_emit", lambda obj: plan.update(obj))

        rc = discovery_run.main([
            "--sector", "14",
            "--count", "3",
            "--registry", str(registry),
            "--tics-csv", str(worklist),
            "--force-rescan",
            "--db", str(tmp_path / "ex.db"),
        ])

        assert rc == 2
        assert plan["status"] == "error"
        assert plan["reason"] == "no_tasks_created"
        assert plan["mode"] == "deep_rescan"
        assert plan["enqueue_skipped"] == 3
        assert log_calls == [], "CAMPAIGN.md must not be appended for a 0-task deep lot"

        reg_after = json.loads(registry.read_text())
        assert reg_after["sectors"]["14"]["deep_cursor"] == 0, (
            "deep_cursor must not advance when the deep lot created zero real tasks"
        )



# ------------------------------------------------------------------ #
# Fase 2 regression (2026-08-27): _load_downloaded_set via MetadataCache
# ------------------------------------------------------------------ #

def test_load_downloaded_set_sector_scoped_via_metadata_cache(tmp_path):
    """downloaded_set must come from the SAME MetadataCache seam as the work
    list, sector-scoped. The old raw sqlite3 read against the frozen
    data/metadata_cache.db returned an empty set for any sector indexed only
    in PG after the 2026-07-21 migration (s18, s26, ...), so every survey of
    such a sector failed with a spurious need_download (ready=0)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("discovery_run_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    db = tmp_path / "meta.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE tglc (tic_id TEXT, sector INTEGER, path TEXT, "
        "exptime REAL, tmag REAL, ra REAL, dec REAL, "
        "PRIMARY KEY(tic_id, sector))")
    con.executemany(
        "INSERT INTO tglc (tic_id, sector, path) VALUES (?, ?, ?)",
        [("111", 18, "a"), ("222", 18, "b"), ("333", 15, "c")])
    con.commit()
    con.close()

    assert mod._load_downloaded_set(str(db), 18) == {111, 222}
    assert mod._load_downloaded_set(str(db), 15) == {333}
    assert mod._load_downloaded_set(str(db)) == {111, 222, 333}


# ------------------------------------------------------------------ #
# S1 (docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md §4):
# paged-vs-legacy selector gating at the discovery_run.py CLI level.
# ------------------------------------------------------------------ #

def _make_registry_no_canonical(path: Path, sector: int = 6, cursor: int = 0) -> None:
    """A sector with NO canonical_list — the exact case S1's paged
    selector targets (the six open sectors as of 2026-09-26)."""
    data = {
        "active_sector": sector,
        "sectors": {
            str(sector): {
                "canonical_list": None,
                "total_tics": 0,
                "downloaded": 0,
                "cursor": cursor,
                "enabled": True,
            }
        },
    }
    path.write_text(json.dumps(data, indent=2))


class TestPagedSelectorGating:
    """--dry-run only (matches the module's own convention above): asserts
    WHICH selector ran (plan["selector"] == "paged" is the paged path's own
    tell — see discovery_run.py's plan dict; the legacy path never sets
    this key) without touching a real DB/registry."""

    def _setup(self, tmp_path, sector=6, n=30, cursor=0):
        tics = list(range(950000, 950000 + n))
        registry = tmp_path / "sectors.json"
        tglc_db = tmp_path / "tglc.db"
        tasks_db = tmp_path / "ex.db"
        _make_registry_no_canonical(registry, sector=sector, cursor=cursor)
        _make_tglc_db(tglc_db, tics)  # writes sector=1 rows -- patch below
        # _make_tglc_db hardcodes sector=1; rewrite for our test sector.
        con = sqlite3.connect(str(tglc_db))
        con.execute(f"UPDATE tglc SET sector = {sector}")
        con.commit()
        con.close()
        # Provision tic_sector_state (done-state's own table) BEFORE the
        # subprocess runs -- real deploy order guarantees this exists
        # whenever EXONIT_DONE_STATE=1 is live (see done-state design §5);
        # select_lot_paged has no defensive fallback for a missing table,
        # by design (matches every other done_state.* reader).
        from exonit.db import ExoNitDB
        ExoNitDB(str(tasks_db))
        return registry, tglc_db, tasks_db, tics

    def test_paged_used_when_flag_set_no_canonical_done_state_on(self, tmp_path, monkeypatch):
        registry, tglc_db, tasks_db, tics = self._setup(tmp_path)
        monkeypatch.setenv("EXONIT_SELECTOR_PAGED", "1")
        monkeypatch.setenv("EXONIT_DONE_STATE", "1")

        result = _run([
            "--sector", "6", "--count", "5",
            "--registry", str(registry),
            "--tglc-index", str(tglc_db),
            "--db", str(tasks_db),
            "--dry-run",
        ])
        assert result.returncode == 0, result.stderr
        plan = json.loads(result.stdout)
        assert plan["selector"] == "paged"
        assert plan["remaining_undone_is_upper_bound"] is True
        assert len(plan["tics_selected"]) == 5

    def test_legacy_used_when_canonical_list_present(self, tmp_path, monkeypatch):
        """A canonical_list sector ALWAYS uses the legacy path, flags or not
        (§0 summary: the paged path is scoped to the no-canonical case)."""
        tics = list(range(960000, 960030))
        registry = tmp_path / "sectors.json"
        canonical = tmp_path / "canonical.csv"
        tglc_db = tmp_path / "tglc.db"
        tasks_db = tmp_path / "ex.db"
        _make_canonical_csv(canonical, tics)
        _make_registry(registry, str(canonical))
        _make_tglc_db(tglc_db, tics)
        from exonit.db import ExoNitDB
        ExoNitDB(str(tasks_db))
        monkeypatch.setenv("EXONIT_SELECTOR_PAGED", "1")
        monkeypatch.setenv("EXONIT_DONE_STATE", "1")

        result = _run([
            "--sector", "1", "--count", "5",
            "--registry", str(registry),
            "--canonical", str(canonical),
            "--tglc-index", str(tglc_db),
            "--db", str(tasks_db),
            "--dry-run",
        ])
        assert result.returncode == 0, result.stderr
        plan = json.loads(result.stdout)
        assert "selector" not in plan

    def test_legacy_used_when_done_state_off_even_if_paged_requested(self, tmp_path, monkeypatch):
        registry, tglc_db, tasks_db, tics = self._setup(tmp_path)
        monkeypatch.setenv("EXONIT_SELECTOR_PAGED", "1")
        monkeypatch.delenv("EXONIT_DONE_STATE", raising=False)

        result = _run([
            "--sector", "6", "--count", "5",
            "--registry", str(registry),
            "--tglc-index", str(tglc_db),
            "--db", str(tasks_db),
            "--dry-run",
        ])
        assert result.returncode == 0, result.stderr
        assert "EXONIT_DONE_STATE=0" in result.stderr
        plan = json.loads(result.stdout)
        assert "selector" not in plan

    def test_full_scan_forces_legacy_even_with_paged_flag(self, tmp_path, monkeypatch):
        registry, tglc_db, tasks_db, tics = self._setup(tmp_path)
        monkeypatch.setenv("EXONIT_DONE_STATE", "1")

        result = _run([
            "--sector", "6", "--count", "5",
            "--registry", str(registry),
            "--tglc-index", str(tglc_db),
            "--db", str(tasks_db),
            "--paged", "--full-scan",
            "--dry-run",
        ])
        assert result.returncode == 0, result.stderr
        plan = json.loads(result.stdout)
        assert "selector" not in plan

    def test_paged_cli_flag_forces_paged_even_without_env_var(self, tmp_path, monkeypatch):
        registry, tglc_db, tasks_db, tics = self._setup(tmp_path)
        monkeypatch.delenv("EXONIT_SELECTOR_PAGED", raising=False)
        monkeypatch.setenv("EXONIT_DONE_STATE", "1")

        result = _run([
            "--sector", "6", "--count", "5",
            "--registry", str(registry),
            "--tglc-index", str(tglc_db),
            "--db", str(tasks_db),
            "--paged",
            "--dry-run",
        ])
        assert result.returncode == 0, result.stderr
        plan = json.loads(result.stdout)
        assert plan["selector"] == "paged"

    def test_dump_selection_writes_full_paged_list(self, tmp_path, monkeypatch):
        registry, tglc_db, tasks_db, tics = self._setup(tmp_path, n=30)
        monkeypatch.setenv("EXONIT_DONE_STATE", "1")
        dump_path = tmp_path / "dump.csv"

        result = _run([
            "--sector", "6", "--count", "25",
            "--registry", str(registry),
            "--tglc-index", str(tglc_db),
            "--db", str(tasks_db),
            "--paged", "--dry-run",
            "--dump-selection", str(dump_path),
        ])
        assert result.returncode == 0, result.stderr
        plan = json.loads(result.stdout)
        assert len(plan["tics_selected"]) == 10  # preview capped at 10
        with open(dump_path) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 25  # full selection, not just the preview

    def test_paged_no_undone_tics_when_all_exhausted(self, tmp_path, monkeypatch):
        registry, tglc_db, tasks_db, tics = self._setup(tmp_path, n=10)
        monkeypatch.setenv("EXONIT_DONE_STATE", "1")
        from exonit.db import ExoNitDB
        db = ExoNitDB(str(tasks_db))
        for t in tics:
            db._exec(
                "INSERT INTO tic_sector_state (tic_id, sector, status, attempts) "
                "VALUES (?, 6, 'failed', 2)", (str(t),))

        result = _run([
            "--sector", "6", "--count", "100",
            "--registry", str(registry),
            "--tglc-index", str(tglc_db),
            "--db", str(tasks_db),
            "--paged", "--dry-run",
        ])
        assert result.returncode == 2
        plan = json.loads(result.stdout)
        assert plan["reason"] == "no_undone_tics"
        assert plan["remaining_undone"] == 0
