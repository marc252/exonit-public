"""add_candidate must survive the insert race introduced by uvicorn --workers N.

The per-process self._lock cannot serialize inserts across processes, so two
workers reprocessing the same TIC can both pass the existence SELECT and both
INSERT — the loser hits the unique index (sector, tic_id, candidate_num). That
used to raise IntegrityError and 500 the worker's /api/worker/result POST.
add_candidate must instead fall back to UPDATE.
"""
import sqlite3
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exonit.db import ExoNitDB


def _add(db, tic, num=1, snr=10.0):
    db.add_candidate(tic, num, period_days=1.0, t0=0.0, duration_days=0.1,
                     depth_ppm=100.0, snr=snr)


def test_add_candidate_idempotent(tmp_path):
    path = str(tmp_path / "t.db")
    db = ExoNitDB(path)
    _add(db, "111", snr=10.0)
    _add(db, "111", snr=20.0)  # reprocess same (tic, candidate_num)
    con = sqlite3.connect(path)
    n = con.execute("SELECT COUNT(*) FROM candidates WHERE tic_id='111'").fetchone()[0]
    snr = con.execute("SELECT snr FROM candidates WHERE tic_id='111'").fetchone()[0]
    con.close()
    assert n == 1            # no duplicate
    assert snr == 20.0       # refreshed


def test_add_candidate_concurrent_no_integrity_error(tmp_path):
    # Two separate connections (simulating two uvicorn worker processes) insert
    # the same brand-new candidate at the same time. Neither may raise; exactly
    # one row must result. Looped to make the race reliable.
    path = str(tmp_path / "t.db")
    ExoNitDB(path)  # init schema once
    errors = []
    for i in range(25):
        tic = str(1000 + i)
        dbs = [ExoNitDB(path), ExoNitDB(path)]
        barrier = threading.Barrier(2)

        def worker(db):
            try:
                barrier.wait()
                _add(db, tic)
            except Exception as e:  # noqa: BLE001 — we want to catch IntegrityError
                errors.append(repr(e))

        threads = [threading.Thread(target=worker, args=(d,)) for d in dbs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert not errors, f"add_candidate raised under concurrency: {errors[:5]}"
    con = sqlite3.connect(path)
    dupes = con.execute(
        "SELECT tic_id, COUNT(*) c FROM candidates GROUP BY tic_id HAVING c > 1"
    ).fetchall()
    con.close()
    assert not dupes, f"duplicate candidates: {dupes}"


def _periods(path, tic="222"):
    con = sqlite3.connect(path)
    rows = con.execute("SELECT sector, period_days, claude_verdict FROM candidates "
                       "WHERE tic_id=? ORDER BY sector", (tic,)).fetchall()
    con.close()
    return rows


def _addp(db, period, sector, verdict=None, tic="222"):
    return db.add_candidate(tic, 1, period, 0.0, 0.1, 100.0, 10.0,
                            claude_verdict=verdict, sector=sector)


def test_add_candidate_new_sector_inserts_new_row(tmp_path):
    path = str(tmp_path / "t.db")
    db = ExoNitDB(path)
    id6 = _addp(db, 3.00, 6, "planet_candidate")
    id7 = _addp(db, 6.01, 7)
    assert id6 != id7
    assert _periods(path) == [(6, 3.00, "planet_candidate"), (7, 6.01, None)]


def test_add_candidate_same_sector_twice_updates_keeping_verdict(tmp_path):
    path = str(tmp_path / "t.db")
    db = ExoNitDB(path)
    a = _addp(db, 3.00, 6, "planet_candidate")
    b = _addp(db, 3.10, 6)  # force-rescan: COALESCE keeps verdict
    assert a == b
    assert _periods(path) == [(6, 3.10, "planet_candidate")]


def test_add_candidate_sector_none_twice_one_row(tmp_path):
    path = str(tmp_path / "t.db")
    db = ExoNitDB(path)
    _addp(db, 3.00, None)
    _addp(db, 3.20, None)
    assert _periods(path) == [(1, 3.20, None)]


def test_add_candidate_explicit_then_untagged_updates_same_row(tmp_path):
    path = str(tmp_path / "t.db")
    db = ExoNitDB(path)
    a = _addp(db, 3.00, 6, "planet_candidate")
    b = _addp(db, 3.30, None)  # straggler / untagged caller
    assert a == b
    assert _periods(path) == [(6, 3.30, "planet_candidate")]


def test_delete_candidates_for_tic_scoped_by_sector(tmp_path):
    path = str(tmp_path / "t.db")
    db = ExoNitDB(path)
    _addp(db, 3.0, 6)
    _addp(db, 6.0, 7)
    assert db.delete_candidates_for_tic("222", sector=7) == 1
    assert [r[0] for r in _periods(path)] == [6]
    _addp(db, 6.0, 7)
    assert db.delete_candidates_for_tic("222") == 2  # legacy: all sectors
