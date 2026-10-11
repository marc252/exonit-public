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
