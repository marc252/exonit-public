"""Work-stream queue tests, parametrised over memory and redis backends
(mirrors tests/test_shared_state.py's fixture pattern at :70)."""
import json
import time
import pytest

from exonit import work_streams
from exonit.work_streams import MemoryWorkStream, RedisWorkStream


@pytest.fixture(params=["memory", "redis"])
def stream(request):
    if request.param == "memory":
        ws = MemoryWorkStream("t1", visibility_timeout_s=1, max_deliveries=2)
    else:
        redis_lib = pytest.importorskip("redis")
        client = redis_lib.Redis.from_url("redis://localhost:6379/9",
                                          decode_responses=True)
        try:
            client.ping()
        except Exception:
            pytest.skip("no local redis")
        client.flushdb()
        ws = RedisWorkStream(client, "t1", visibility_timeout_s=1,
                             max_deliveries=2)
    ws.clear()
    yield ws
    ws.clear()


def test_enqueue_claim_ack_roundtrip(stream):
    assert stream.enqueue("c1", {"candidate_id": 1}) is True
    item = stream.claim("worker-A")
    assert item is not None
    assert item.item_key == "c1"
    assert item.payload == {"candidate_id": 1}
    assert stream.claim("worker-B") is None          # cua buida (c1 al PEL)
    assert stream.ack("c1") is True
    assert stream.stats() == {"queued": 0, "pending": 0, "dlq": 0}


def test_enqueue_dedup(stream):
    assert stream.enqueue("c1", {"x": 1}) is True
    assert stream.enqueue("c1", {"x": 1}) is False   # ja encuat
    stream.claim("worker-A")
    assert stream.enqueue("c1", {"x": 1}) is False   # inflight also deduplicated
    stream.ack("c1")
    assert stream.enqueue("c1", {"x": 1}) is True    # after ack it can be re-enqueued


def test_stale_claim_is_stolen_after_visibility_timeout(stream):
    stream.enqueue("c1", {"x": 1})
    a = stream.claim("worker-A")
    assert a.item_key == "c1"
    time.sleep(1.2)                                   # visibility_timeout_s=1
    b = stream.claim("worker-B")                      # XAUTOCLAIM el roba
    assert b is not None and b.item_key == "c1"
    assert b.deliveries >= 2
    assert stream.ack("c1") is True


def test_keep_if_prevents_steal(stream):
    stream.enqueue("c1", {"x": 1})
    stream.claim("worker-A")
    time.sleep(1.2)
    b = stream.claim("worker-B", keep_if=lambda key, consumer: True)
    assert b is None                                  # protected, not stolen


def test_keep_if_receives_original_owner(stream):
    # Regression: keep_if must see the TRUE stale owner (worker-A), not
    # the caller.  XAUTOCLAIM — even with JUSTID — reassigns PEL
    # ownership to the caller as a side effect before keep_if could run,
    # so the redis backend must enumerate via XPENDING instead.
    stream.enqueue("c1", {"x": 1})
    stream.claim("worker-A")
    time.sleep(1.2)
    seen = []
    b = stream.claim(
        "worker-B",
        keep_if=lambda key, consumer: seen.append((key, consumer)) or True)
    assert b is None
    assert seen == [("c1", "worker-A")]


def test_steal_with_keep_if_false_then_ack_consistent(stream):
    # Race-safety companion: when keep_if does NOT veto, the steal must
    # complete and leave ack()/stats() fully consistent.
    stream.enqueue("c1", {"x": 1})
    stream.claim("worker-A")
    time.sleep(1.2)
    b = stream.claim("worker-B", keep_if=lambda key, consumer: False)
    assert b is not None and b.item_key == "c1"
    assert b.deliveries >= 2
    assert stream.ack("c1") is True
    assert stream.stats() == {"queued": 0, "pending": 0, "dlq": 0}


def test_max_deliveries_moves_to_dlq(stream):
    stream.enqueue("c1", {"x": 1})
    stream.claim("worker-A")
    time.sleep(1.2)
    stream.claim("worker-B")                          # delivery 2 (= max)
    time.sleep(1.2)
    c = stream.claim("worker-C")                      # superaria max → DLQ
    assert c is None
    s = stream.stats()
    assert s["dlq"] == 1 and s["pending"] == 0
    assert stream.enqueue("c1", {"x": 1}) is True     # DLQ neteja el dedup


def test_peek_dlq_unified_shape_across_backends(stream):
    """Phase 1, Task 5: peek_dlq() must produce the SAME {"k","p","msg_id",
    "ts"} shape (p as a JSON STRING) regardless of backend — Redis's native
    DLQ entry shape and MemoryWorkStream's internal item_key/payload shape
    must be normalised to the one uniform contract callers rely on."""
    stream.enqueue("c1", {"tic_id": "999"})
    stream.claim("worker-A")
    time.sleep(1.2)
    stream.claim("worker-B")                          # delivery 2 (= max)
    time.sleep(1.2)
    assert stream.claim("worker-C") is None            # would be delivery 3 -> DLQ
    entries = stream.peek_dlq()
    assert len(entries) == 1
    e = entries[0]
    assert e["k"] == "c1"
    assert json.loads(e["p"]) == {"tic_id": "999"}
    assert "msg_id" in e and "ts" in e
    # peek must NOT remove it — safe to call again.
    assert stream.peek_dlq() == entries


def test_enqueue_many_basic_dedup_within_call_and_stats(stream):
    """G3 (review package, 2026-07-05)."""
    items = [("c1", {"x": 1}), ("c2", {"x": 2}), ("c1", {"x": 1})]
    n = stream.enqueue_many(items)
    assert n == 2, "c1 duplicated WITHIN the same call must only count once"
    assert stream.stats() == {"queued": 2, "pending": 0, "dlq": 0}

    seen = {}
    for _ in range(2):
        item = stream.claim("worker-A")
        assert item is not None
        seen[item.item_key] = item.payload
    assert seen == {"c1": {"x": 1}, "c2": {"x": 2}}


def test_enqueue_many_respects_pre_existing_dedup(stream):
    stream.enqueue("c1", {"x": 1})
    n = stream.enqueue_many([("c1", {"x": 1}), ("c2", {"x": 2})])
    assert n == 1, "c1 already queued -> only c2 is newly enqueued"
    assert stream.stats()["queued"] == 2


def test_enqueue_many_empty_list_is_noop(stream):
    assert stream.enqueue_many([]) == 0
    assert stream.stats() == {"queued": 0, "pending": 0, "dlq": 0}


def test_enqueue_many_n1000_matches_stats_and_dedups_on_repeat(stream):
    """G3 contract: N=1000 items via enqueue_many -> stats queued == 1000,
    and a repeat call dedups entirely (0 new) — both backends agree
    (parametrised ``stream`` fixture)."""
    items = [(f"item-{i}", {"i": i}) for i in range(1000)]
    n = stream.enqueue_many(items)
    assert n == 1000
    assert stream.stats()["queued"] == 1000
    n2 = stream.enqueue_many(items)
    assert n2 == 0, "a repeat call with the same keys must dedup entirely"
    assert stream.stats()["queued"] == 1000


def test_remove_dlq_removes_matching_entry(stream):
    """G4 (review package, 2026-07-05): DLQ hygiene — a resolved entry can
    be pruned by msg_id, shrinking the list instead of growing forever."""
    stream.enqueue("c1", {"x": 1})
    stream.claim("worker-A")
    time.sleep(1.2)
    stream.claim("worker-B")                          # delivery 2 (= max)
    time.sleep(1.2)
    assert stream.claim("worker-C") is None            # -> DLQ
    entries = stream.peek_dlq()
    assert len(entries) == 1
    msg_id = entries[0]["msg_id"]

    assert stream.remove_dlq(msg_id) is True
    assert stream.peek_dlq() == []
    assert stream.stats()["dlq"] == 0


def test_remove_dlq_unknown_msg_id_returns_false(stream):
    assert stream.remove_dlq("no-such-id") is False


def test_remove_dlq_does_not_affect_other_entries(stream):
    def _drive_to_dlq(key):
        # Each item is driven through its FULL queue -> PEL -> DLQ cycle
        # before the next is enqueued, so the queue is guaranteed empty
        # when max_deliveries triggers (matches the existing single-item
        # DLQ tests' shape above) — a second item still queued at that
        # point would be popped fresh instead of the third claim()
        # correctly returning None.
        stream.enqueue(key, {"k": key})
        stream.claim("worker-A")
        time.sleep(1.2)
        stream.claim("worker-B")
        time.sleep(1.2)
        assert stream.claim("worker-C") is None

    _drive_to_dlq("c1")
    _drive_to_dlq("c2")
    entries = stream.peek_dlq()
    assert len(entries) == 2

    target = entries[0]
    assert stream.remove_dlq(target["msg_id"]) is True
    remaining = stream.peek_dlq()
    assert len(remaining) == 1
    assert remaining[0]["k"] != target["k"]


def test_ack_unknown_key_returns_false(stream):
    assert stream.ack("mai-vist") is False


def test_clear_resets_everything(stream):
    stream.enqueue("c1", {"x": 1})
    stream.enqueue("c2", {"x": 2})
    stream.claim("worker-A")
    stream.clear()
    assert stream.stats() == {"queued": 0, "pending": 0, "dlq": 0}
    assert stream.claim("worker-A") is None


def test_get_stream_factory_singleton():
    s1 = work_streams.get_stream("factory-test")
    s2 = work_streams.get_stream("factory-test")
    assert s1 is s2
