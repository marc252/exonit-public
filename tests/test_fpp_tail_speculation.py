"""The FPP straggler tail (2026-07-14) — the three mechanisms that fix it.

The failure this file pins down, from run s0014/lot20260714T175216: the FPP
queue drained 7488 -> 31 items in 17 minutes, then spent 16 MORE minutes on
those last 31 — half the phase for 0.4% of the work, in flat steps of zero
progress, while ~80 worker instances sat idle. A claimed item is invisible to
every other worker, and `keep_if` (_worker_alive_and_on) vetoed stealing it
back for as long as its owner *claimed* to be alive and on that TIC — with no
ceiling. A worker that is frozen but still heartbeating (<host>'s WDDM VRAM
freeze; a sleeping Mac) therefore held its item forever. The operator unstuck
the run by killing workers by hand, which is exactly that veto being lifted.

Three mechanisms, one test class each:
  1. BACKFILL_HOLD_CEILING_SECONDS — the veto expires; a liar loses its item.
  2. STAGE_SETTINGS["fpp"] visibility 300s — an orphaned item returns fast.
  3. claim_speculative — an idle worker races a live-but-slow owner's item,
     WITHOUT taking it away. First result wins; the loser's is idempotent.
"""
from __future__ import annotations

import os
import time

import pytest

from exonit import work_streams
from exonit.work_streams import MemoryWorkStream, RedisWorkStream

AUTH = {"Authorization": "Bearer exonit-default-token"}


@pytest.fixture(params=["memory", "redis"])
def stream(request):
    """Both backends, same contract (mirrors tests/test_work_streams.py)."""
    if request.param == "memory":
        ws = MemoryWorkStream("spec", visibility_timeout_s=1, max_deliveries=3)
    else:
        redis_lib = pytest.importorskip("redis")
        client = redis_lib.Redis.from_url("redis://localhost:6379/9",
                                          decode_responses=True)
        try:
            client.ping()
        except Exception:
            pytest.skip("no local redis")
        client.flushdb()
        ws = RedisWorkStream(client, "spec", visibility_timeout_s=1,
                             max_deliveries=3)
    ws.clear()
    yield ws
    ws.clear()


# ---------------------------------------------------------------- mechanism 3
class TestSpeculativeTailExecution:

    def test_no_speculation_while_unclaimed_work_remains(self, stream):
        """The whole point is to spend IDLE capacity. If anything is still
        unclaimed, an idle worker must take THAT — duplicating an item while
        untouched ones wait would be strictly worse than doing nothing."""
        stream.enqueue("c1", {"candidate_id": 1})
        stream.enqueue("c2", {"candidate_id": 2})
        stream.claim("worker-A")                    # c1 in flight, c2 queued
        time.sleep(1.1)                             # past visibility timeout

        assert stream.claim_speculative("worker-B") is None

    def test_idle_tail_item_is_served_as_a_copy_without_taking_it(self, stream):
        """The straggler case: queue empty, one item in flight, one idle
        worker. B gets a COPY — and A does NOT lose its claim (no PEL
        mutation), so a slow-but-healthy A can still win the race."""
        stream.enqueue("c1", {"candidate_id": 1})
        owned = stream.claim("worker-A")
        time.sleep(1.1)

        spec = stream.claim_speculative("worker-B")
        assert spec is not None
        assert spec.item_key == "c1"
        assert spec.payload == {"candidate_id": 1}
        assert spec.speculative is True
        assert owned.speculative is False
        # A still owns the PEL entry: this is a race, not a repossession.
        assert stream.stats() == {"queued": 0, "pending": 1, "dlq": 0}

    def test_not_offered_before_the_item_has_gone_idle(self, stream):
        """An item handed out seconds ago is probably about to finish.
        Duplicating it would burn a worker for nothing."""
        stream.enqueue("c1", {"candidate_id": 1})
        stream.claim("worker-A")

        assert stream.claim_speculative("worker-B") is None

    def test_worker_never_races_itself(self, stream):
        stream.enqueue("c1", {"candidate_id": 1})
        stream.claim("worker-A")
        time.sleep(1.1)

        assert stream.claim_speculative("worker-A") is None

    def test_at_most_one_live_replica_per_item(self, stream):
        """Two idle workers, one straggler: exactly one copy goes out. Without
        this gate an idle fleet would pile every spare instance onto the same
        item."""
        stream.enqueue("c1", {"candidate_id": 1})
        stream.claim("worker-A")
        time.sleep(1.1)

        assert stream.claim_speculative("worker-B") is not None
        assert stream.claim_speculative("worker-C") is None

    def test_released_replica_slot_is_reusable(self, stream):
        """A copy handed out but never actually run (ineligible, no metadata)
        must not lock the item out of speculation for the whole TTL."""
        stream.enqueue("c1", {"candidate_id": 1})
        stream.claim("worker-A")
        time.sleep(1.1)

        stream.claim_speculative("worker-B")
        stream.release_replica("c1")

        assert stream.claim_speculative("worker-C") is not None

    def test_ack_by_the_speculative_winner_retires_the_item(self, stream):
        """B wins the race. ack() is keyed by ITEM, not by owner, so B's ack
        XACKs the entry A still holds in the PEL — the item is done and
        disappears for both."""
        stream.enqueue("c1", {"candidate_id": 1})
        stream.claim("worker-A")
        time.sleep(1.1)
        stream.claim_speculative("worker-B")

        assert stream.ack("c1") is True                  # B finishes first
        assert stream.stats() == {"queued": 0, "pending": 0, "dlq": 0}
        assert stream.ack("c1") is False                 # A's late ack: no-op
        assert stream.claim("worker-C") is None          # nothing left at all


class TestSettleDecidesTheRace:
    """settle() is what stops speculation from corrupting the DB.

    TRICERATOPS is a Monte Carlo drawing from np.random with NO seed
    (worker/fpp_native.py), so the two racers compute DIFFERENT FPP values for
    the same candidate. Writing both would leave the row holding whichever POST
    happened to win the network race — nondeterministic, and not even the value
    of the worker that won. Duplicate CPU is the price of speculation; a
    nondeterministic number in the database is not (CLAUDE.md: reproducibility
    is non-negotiable)."""

    def test_first_result_wins_and_the_loser_is_told_to_drop_its_own(self, stream):
        stream.enqueue("c1", {"candidate_id": 1})
        stream.claim("worker-A")
        time.sleep(1.1)
        stream.claim_speculative("worker-B")

        assert stream.settle("c1") == "winner"    # B gets there first: persist
        assert stream.settle("c1") == "loser"     # A arrives later: DISCARD
        assert stream.stats() == {"queued": 0, "pending": 0, "dlq": 0}

    def test_an_unraced_item_is_never_a_loser(self, stream):
        """No speculation, no race: the ordinary FPP result path must be
        completely unaffected."""
        stream.enqueue("c1", {"candidate_id": 1})
        stream.claim("worker-A")

        assert stream.settle("c1") == "winner"
        # A second, unexpected result for the same candidate (a manual re-run, a
        # result landing after reconcile_stream rebuilt the queue) is "unknown",
        # NOT "loser" — it must still be persisted, exactly as before.
        assert stream.settle("c1") == "unknown"

    def test_result_for_an_item_the_stream_never_knew_is_unknown(self, stream):
        """The legacy path and manual re-runs POST results for candidates that
        were never on the stream at all. Those must always be persisted."""
        assert stream.settle("c999") == "unknown"

    def test_rebuild_does_not_leave_a_marker_that_rejects_future_results(
            self, stream):
        """reconcile_stream() clears and re-derives the queue. A settled marker
        surviving that would silently discard a legitimate later result for the
        same candidate."""
        stream.enqueue("c1", {"candidate_id": 1})
        stream.claim("worker-A")
        time.sleep(1.1)
        stream.claim_speculative("worker-B")
        stream.settle("c1")                        # marker now exists

        stream.clear()

        stream.enqueue("c1", {"candidate_id": 1})
        stream.claim("worker-C")
        assert stream.settle("c1") == "winner", (
            "a stale settled marker must not survive a queue rebuild")

    def test_no_speculation_when_the_tail_is_not_a_tail(self, stream):
        """`tail_max` is the "is this actually the tail?" test. Mid-run, with
        many items in flight, slow items are normal and duplicating them would
        just halve throughput."""
        for i in range(5):
            stream.enqueue(f"c{i}", {"candidate_id": i})
            stream.claim(f"worker-{i}")
        time.sleep(1.1)

        assert stream.claim_speculative("worker-idle", tail_max=3) is None
        assert stream.claim_speculative("worker-idle", tail_max=5) is not None


# ---------------------------------------------------------------- mechanism 2
def test_fpp_visibility_timeout_is_five_minutes():
    """An ORPHANED fpp item (worker process died between claim and ack) must
    come back in 5 min, not 15. primary keeps 1800s on purpose: its keep_if
    trusts a heartbeat known to stall for minutes under the GIL during BLS."""
    assert work_streams.STAGE_SETTINGS["fpp"]["visibility_timeout_s"] == 300
    assert work_streams.STAGE_SETTINGS["primary"]["visibility_timeout_s"] == 1800


# ---------------------------------------------------------------- mechanism 1
class TestBackfillHoldCeiling:
    """_worker_alive_and_on is a backfill stream's keep_if. It must stop
    believing a worker that has held one item past the ceiling."""

    def _state(self, fresh=True):
        return {"w1": {"last_heartbeat": time.time() - (1 if fresh else 10_000)}}

    def test_alive_and_on_tic_within_ceiling_keeps_the_item(self):
        from exonit.server import _set_worker_backfill_tic, _worker_alive_and_on

        _set_worker_backfill_tic("w1", "12345", "FPP")
        assert _worker_alive_and_on(self._state(), "w1", "12345") is True

    def test_hold_past_the_ceiling_releases_the_item(self, monkeypatch):
        """The frozen-but-heartbeating worker (WDDM VRAM freeze, sleeping Mac):
        it says it is alive and on the TIC, and it will say so forever. Past the
        ceiling that claim stops being enough to veto a steal."""
        import exonit.server as server

        server._set_worker_backfill_tic("w1", "12345", "FPP")
        monkeypatch.setattr(server, "BACKFILL_HOLD_CEILING_SECONDS", 0)

        assert server._worker_alive_and_on(self._state(), "w1", "12345") is False

    def test_ceiling_does_not_delete_the_dashboard_record(self, monkeypatch):
        """The ceiling is a pure READ: the worker may well still be grinding on
        that TIC, and /api/worker/list must keep showing it."""
        import exonit.server as server

        server._set_worker_backfill_tic("w1", "12345", "FPP")
        monkeypatch.setattr(server, "BACKFILL_HOLD_CEILING_SECONDS", 0)

        server._worker_alive_and_on(self._state(), "w1", "12345")
        assert server._get_worker_backfill_tic("w1") == ("12345", "FPP")

    def test_dead_worker_still_loses_its_item_immediately(self):
        """Pre-existing behaviour, unchanged: a stale heartbeat alone is
        enough to release the item."""
        from exonit.server import _set_worker_backfill_tic, _worker_alive_and_on

        _set_worker_backfill_tic("w1", "12345", "FPP")
        assert _worker_alive_and_on(self._state(fresh=False), "w1", "12345") is False


# ------------------------------------------------------- handler integration
def test_fpp_task_serves_a_speculative_copy_when_the_queue_is_a_tail(
        tmp_path, monkeypatch):
    """End to end through /api/worker/fpp_task: with the queue empty and one
    idle in-flight item, a second worker asking for work gets that item's
    candidate — instead of the {"task": None} that left 79 instances idle
    while the run waited 16 minutes on 31 candidates."""
    from exonit.db import ExoNitDB
    from exonit.metadata_cache import MetadataCache

    db_path = str(tmp_path / "test.db")
    meta_path = str(tmp_path / "meta.db")
    monkeypatch.setenv("EXONIT_DB_PATH", db_path)
    monkeypatch.setenv("EXONIT_METADATA_CACHE", meta_path)
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE", str(tmp_path / "st.json"))
    monkeypatch.setenv("EXONIT_WORK_STREAMS", "1")
    os.makedirs(str(tmp_path / "reports"), exist_ok=True)

    db = ExoNitDB(db_path)
    db.add_star("777", 0, 0, 10, 1, 1000, 30)
    candidate_id = db.add_candidate("777", 1, 2.0, 0, 0.1, 500, 50.0,
                                    cnn_score=0.9)
    db.close()
    cache = MetadataCache(meta_path)
    cache.add_entry("777", sectors=[1], exptime=120.0, fits_files=["a.fits"])
    cache.close()

    ws = work_streams.MemoryWorkStream("fpp", visibility_timeout_s=0.2)
    monkeypatch.setattr(work_streams, "get_stream", lambda name: ws)

    from fastapi.testclient import TestClient

    from exonit.server import create_app

    with TestClient(create_app()) as client:
        # The straggler must be ALIVE and ON-TIC, or this test proves nothing:
        # a dead owner's item is simply STOLEN by the ordinary claim path (its
        # keep_if returns False), and speculation never comes into it. The whole
        # point of speculation is the case the steal cannot touch — an owner
        # that looks perfectly healthy and just will not finish.
        reg = client.post("/api/worker/register", json={"name": "worker-slow"},
                          headers=AUTH)
        wid_slow = reg.json()["worker_id"]

        ws.enqueue(str(candidate_id), {"candidate_id": candidate_id})
        served = client.get("/api/worker/fpp_task",
                            headers={**AUTH, "X-Worker-ID": wid_slow})
        assert served.json()["task"]["candidate_id"] == candidate_id
        time.sleep(0.3)                     # ...and then it just sits there

        resp = client.get("/api/worker/fpp_task",
                          headers={**AUTH, "X-Worker-ID": "worker-idle"})
        assert resp.status_code == 200
        task = resp.json()["task"]
        assert task is not None, (
            "an idle worker must be given the stalled tail item to race, "
            "not None")
        assert task["candidate_id"] == candidate_id
        # The straggler keeps its claim — we duplicated the work, we did not
        # confiscate it.
        assert ws.stats()["pending"] == 1

        # ── the race resolves: first result in wins, the other is dropped ──
        # Both workers finish and POST. Their FPP values DIFFER (TRICERATOPS is
        # an unseeded Monte Carlo), so persisting both would leave the row
        # holding whichever POST arrived last over the network.
        def _post(worker, fpp_value):
            return client.post(
                "/api/worker/fpp_result",
                headers={**AUTH, "X-Worker-ID": worker},
                json={"candidate_id": candidate_id, "fpp": fpp_value,
                      "nfpp": 0.01, "tic_id": "777"})

        assert _post("worker-idle", 0.11).status_code == 200      # winner
        late = _post(wid_slow, 0.99)                               # loser
        assert late.status_code == 200
        assert late.json().get("outcome") == "speculative_loser"

        db = ExoNitDB(db_path)
        stored = db.get_candidates("777")[0]["triceratops_fpp"]
        db.close()
        assert stored == pytest.approx(0.11), (
            "the winner's value must survive: the loser's late POST must not "
            f"overwrite the row (got {stored})")
