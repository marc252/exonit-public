"""The queue and the DB must be reconcilable (2026-07-14).

The producer hooks enqueue at WRITE time, which leaves two drifts that never
heal on their own — and both stranded a run on 2026-07-14:

  1. Eligible in PG, in NO queue. A writer that bypasses the hooks (raw-SQL
     rescore scripts, a late has_spoc flip) makes a row eligible without
     enqueueing it. health/work counts it as pending; no worker can ever be
     served it; a phase waiting for that counter waits forever. This is why the
     ExoMiner phase of the 50k run sat at "pending=4" with an empty stream.

  2. Dead-lettered, still eligible in PG. An item whose FITS metadata is not
     cached is held back at claim time (no serve, no ack) and, after
     max_deliveries, moves to the DLQ. It leaves the queue — but nothing writes
     its PG row, so it keeps counting as pending FOREVER. 56 FPP candidates had
     been in that state since 2026-07-12, invisible and uncountable.

reconcile_stream() closes both, non-destructively (in-flight claims survive),
and the orchestrator runs it at the start of every backfill phase.
"""

from __future__ import annotations

import pytest

from exonit import work_producers, work_streams
from exonit.db import ExoNitDB


@pytest.fixture
def db(tmp_path):
    d = ExoNitDB(str(tmp_path / "r.db"))
    yield d
    d.close()


@pytest.fixture
def streams(monkeypatch):
    made = {n: work_streams.MemoryWorkStream(n)
            for n in ("fpp", "exominer", "primary")}
    monkeypatch.setenv("EXONIT_WORK_STREAMS", "1")
    monkeypatch.setattr(work_producers, "_get_stream", lambda n: made[n])
    return made


def _star(db, tic, has_spoc=True):
    db.add_star(tic, ra=1.0, dec=2.0, tmag=11.0, n_sectors=1,
                n_datapoints=1000, time_span_days=27.0, status="completed")
    db.set_star_has_spoc(tic, has_spoc)


def _cand(db, tic, num=1, snr=20.0, cnn=0.9):
    return db.add_candidate(tic, candidate_num=num, period_days=3.0, t0=0.0,
                            duration_days=0.1, depth_ppm=1000, snr=snr,
                            cnn_score=cnn)


def test_eligible_in_db_but_never_enqueued_gets_enqueued(db, streams):
    """Drift 1: the row a hook never saw."""
    _star(db, "100")
    cid = _cand(db, "100")
    streams["fpp"].clear()                      # simulate the hook never firing
    assert streams["fpp"].stats()["queued"] == 0

    out = work_producers.reconcile_stream(db, "fpp")

    assert out["enqueued"] == 1
    assert streams["fpp"].stats()["queued"] == 1
    assert streams["fpp"].claim("w").item_key == str(cid)


def test_dead_lettered_but_still_eligible_comes_back(db, streams):
    """Drift 2: the 56 FPP candidates that sat in the DLQ since 2026-07-12.

    They left the queue (max_deliveries exhausted while their metadata was
    missing) but their PG row never changed, so they counted as pending
    forever with nothing able to serve them.
    """
    _star(db, "200")
    cid = _cand(db, "200")
    fpp = streams["fpp"]
    fpp.clear()

    # Drive it into the DLQ the way production did: claimed, never acked, until
    # max_deliveries is exhausted.
    fpp.enqueue(str(cid), {"candidate_id": cid, "tic_id": "200"})
    fpp.visibility_timeout_s = 0                # every claim is a re-delivery
    for _ in range(fpp.max_deliveries + 1):
        fpp.claim("stuck-worker")
    assert fpp.stats()["dlq"] == 1
    assert fpp.stats()["queued"] == 0 and fpp.stats()["pending"] == 0

    out = work_producers.reconcile_stream(db, "fpp")

    assert out["enqueued"] == 1
    assert out["dlq_cleared"] == 1              # the ghost copy is dropped
    assert fpp.stats()["queued"] == 1
    assert fpp.stats()["dlq"] == 0


def test_reconcile_never_duplicates_or_steals_in_flight_work(db, streams):
    """Non-destructive: an item a worker is COMPUTING right now must not be
    re-enqueued (that is a duplicate computation) nor dropped."""
    _star(db, "300")
    cid = _cand(db, "300")
    fpp = streams["fpp"]
    fpp.clear()
    fpp.enqueue(str(cid), {"candidate_id": cid, "tic_id": "300"})
    claimed = fpp.claim("busy-worker")           # in flight, not acked
    assert claimed.item_key == str(cid)

    out = work_producers.reconcile_stream(db, "fpp")

    assert out["enqueued"] == 0
    assert out["already_queued"] == 1
    assert fpp.stats() == {"queued": 0, "pending": 1, "dlq": 0}


def test_reconcile_is_idempotent(db, streams):
    _star(db, "400")
    _cand(db, "400")
    streams["fpp"].clear()

    first = work_producers.reconcile_stream(db, "fpp")
    second = work_producers.reconcile_stream(db, "fpp")

    assert first["enqueued"] == 1
    assert second["enqueued"] == 0
    assert streams["fpp"].stats()["queued"] == 1


def test_reconcile_exominer_stage(db, streams):
    """The ExoMiner stream drifts the same way and reconciles the same way."""
    _star(db, "500", has_spoc=True)
    cid = _cand(db, "500", cnn=0.95)
    streams["exominer"].clear()

    out = work_producers.reconcile_stream(db, "exominer")

    assert out["enqueued"] == 1
    assert streams["exominer"].claim("w").item_key == str(cid)


def test_reconcile_does_not_enqueue_ineligible_rows(db, streams):
    """A weak candidate (cnn below the gate) must stay out of the queue."""
    _star(db, "600")
    _cand(db, "600", snr=20.0, cnn=0.1)     # fails the OR-gate (no em_ffi)
    streams["fpp"].clear()

    out = work_producers.reconcile_stream(db, "fpp")

    assert out["enqueued"] == 0
    assert streams["fpp"].stats()["queued"] == 0


def test_noop_when_streams_are_disabled(db, streams, monkeypatch):
    monkeypatch.delenv("EXONIT_WORK_STREAMS", raising=False)
    out = work_producers.reconcile_stream(db, "fpp")
    assert out == {"stage": "fpp", "streams_enabled": False}


def test_dlq_is_purged_of_resolved_entries(db, streams):
    """A DLQ full of already-resolved items is noise, not a signal.

    Production had 43 ExoMiner DLQ entries for candidates that had since been
    scored, and 2 for candidates that had been deleted. The `dlq` figure
    health/work reports is only actionable if it means "still owed".
    """
    _star(db, "700")
    cid = _cand(db, "700")
    em = streams["exominer"]
    em.clear()

    # Dead-letter it...
    em.enqueue(str(cid), {"candidate_id": cid, "tic_id": "700"})
    em.visibility_timeout_s = 0
    for _ in range(em.max_deliveries + 1):
        em.claim("stuck")
    assert em.stats()["dlq"] == 1

    # ...and then it gets scored by some other path, so it is no longer owed.
    db.update_candidate_exominer(cid, 0.42)

    out = work_producers.reconcile_stream(db, "exominer")

    assert out["eligible"] == 0          # nothing left to do
    assert out["enqueued"] == 0
    assert out["dlq_cleared"] == 1       # the resolved ghost is gone
    assert em.stats()["dlq"] == 0
