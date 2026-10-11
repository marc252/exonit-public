"""A restarted worker must give its claimed work back immediately (2026-07-14).

The incident
------------
The fleet autoscaler restarts workers at the SCAN->FPP phase transition. On
2026-07-14 it killed 7 workers ~40 s AFTER they had claimed FPP items from the
work stream. The consumer name is derived from the worker NAME, so each new
process re-registered under the SAME worker_id: to Redis the consumer looked
alive, while the new process knew nothing about those items. Nobody served
them, nobody acked them, and the FPP queue had nothing else — the run sat idle
at FPP_BACKFILL for the FULL 30-minute visibility timeout before the steal path
recovered them.

Note what did NOT happen, because the original spec for this fix got it wrong:
`keep_if` never vetoed anything. `_worker_alive_and_on` reads the
`worker:backfill_tic` hash, which was EMPTY (the items were claimed but never
served), so the steal was allowed the moment the visibility timeout elapsed.
There was no forever-veto to break — the wait WAS the bug. Hence the fix is to
requeue on register, not to add another ceiling.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from exonit import work_streams


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EXONIT_METADATA_CACHE", str(tmp_path / "mc.db"))
    monkeypatch.setenv("EXONIT_WORK_STREAMS", "1")
    os.makedirs(tmp_path / "reports", exist_ok=True)

    streams = {n: work_streams.MemoryWorkStream(n)
               for n in ("primary", "fpp", "exominer")}
    monkeypatch.setattr(work_streams, "get_stream", lambda n: streams[n])

    from exonit.server import create_app
    with TestClient(create_app()) as c:
        c.streams = streams
        yield c


AUTH = {"Authorization": "Bearer exonit-default-token"}


def _register(client, name="worker-<host>-1"):
    r = client.post("/api/worker/register", json={"name": name},
                    headers=AUTH)
    assert r.status_code == 200
    return r.json()["worker_id"]


def test_restart_requeues_the_claims_the_dead_process_held(client):
    """The 7-orphaned-FPP-items hang, in miniature."""
    fpp = client.streams["fpp"]
    worker_id = _register(client)

    # The (about to die) process claims an item.
    fpp.enqueue("42", {"candidate_id": 42})
    claimed = fpp.claim(worker_id)
    assert claimed.item_key == "42"
    assert fpp.stats() == {"queued": 0, "pending": 1, "dlq": 0}

    # The autoscaler kills it; the new process re-registers under the same name
    # (=> same worker_id => same stream consumer).
    assert _register(client) == worker_id

    # The item is back on the queue RIGHT NOW — not in 30 minutes.
    assert fpp.stats() == {"queued": 1, "pending": 0, "dlq": 0}
    assert fpp.claim("worker-someone-else").item_key == "42"


def test_restart_releases_every_stream_not_just_fpp(client):
    """ExoMiner and the primary BLS queue orphan claims exactly the same way."""
    worker_id = _register(client)
    for stage in ("primary", "fpp", "exominer"):
        st = client.streams[stage]
        st.enqueue(f"{stage}-1", {"x": 1})
        st.claim(worker_id)
        assert st.stats()["pending"] == 1

    _register(client)

    for stage in ("primary", "fpp", "exominer"):
        st = client.streams[stage]
        assert st.stats()["pending"] == 0, f"{stage} still holds an orphan"
        assert st.stats()["queued"] == 1


def test_another_workers_claims_are_untouched(client):
    """Releasing must be surgical: only the restarting worker's own items."""
    fpp = client.streams["fpp"]
    wid_a = _register(client, "worker-A")
    wid_b = _register(client, "worker-B")

    fpp.enqueue("a", {"candidate_id": 1})
    fpp.enqueue("b", {"candidate_id": 2})
    assert fpp.claim(wid_a).item_key == "a"
    assert fpp.claim(wid_b).item_key == "b"

    _register(client, "worker-A")            # only A restarts

    # A's item came back; B is still legitimately working on its own.
    assert fpp.stats() == {"queued": 1, "pending": 1, "dlq": 0}
    assert fpp.claim("worker-C").item_key == "a"


def test_register_still_succeeds_when_the_release_blows_up(client, monkeypatch):
    """A Redis hiccup must degrade to the old timeout recovery, never to a
    worker that cannot register (that would take the whole fleet down)."""
    def boom(_name):
        raise RuntimeError("redis is having a day")
    monkeypatch.setattr(work_streams, "get_stream", boom)

    r = client.post("/api/worker/register", json={"name": "worker-X"},
                    headers=AUTH)
    assert r.status_code == 200
    assert r.json()["worker_id"]


def test_restart_clears_the_stale_backfill_tic_record(client):
    """A dead process's backfill_tic would otherwise make _worker_alive_and_on
    veto steals of that TIC — on behalf of a process that no longer exists."""
    from exonit.server import (_get_worker_backfill_tic,
                               _set_worker_backfill_tic)

    worker_id = _register(client)
    _set_worker_backfill_tic(worker_id, "12345", "FPP")
    assert _get_worker_backfill_tic(worker_id) is not None

    _register(client)
    assert _get_worker_backfill_tic(worker_id) is None
