"""Server-side work queues over Redis Streams (Fase 1, plan 2026-07-04).

One WorkStream per stage (fpp / exominer / primary).  Work is discovered
ONCE (producer XADDs on the eligibility transition, or the rebuild script
re-derives the queue from PG) and consumed by the existing HTTP handlers
via claim()/ack() — workers themselves are untouched and keep polling
HTTP.  PG remains the source of record: the queue is always rebuildable,
so losing Redis never loses work.

Key layout (all prefixed ws:{name}):
  ws:{name}:stream    the stream itself (group "workers")
  ws:{name}:queued    SET of item_keys currently queued or inflight (dedup)
  ws:{name}:inflight  HASH item_key -> msg_id (results echo item_key, not
                      msg_id, so ack() needs this mapping to XACK)
  ws:{name}:dlq       LIST of JSON payloads that exceeded max_deliveries
  ws:{name}:spec      HASH item_key -> {"consumer","ts"} — the speculative
                      replica currently racing the PEL owner (claim_speculative)
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass

_GROUP = "workers"


@dataclass
class ClaimedItem:
    item_key: str
    payload: dict
    msg_id: str
    deliveries: int
    #: True when this is a speculative COPY of an item another worker still
    #: owns in the PEL (claim_speculative), rather than an owned claim. The
    #: caller must not treat it as an assignment it can drop: releasing it
    #: (release_replica) is what frees the item for the next idle worker.
    speculative: bool = False


#: Speculative tail execution (2026-07-14). See RedisWorkStream.claim_speculative.
#:
#: TAIL_MAX: only speculate once the queue has shrunk to a TAIL — at most this
#: many items still in flight and NOTHING left unclaimed. Duplicating work is
#: only free when the fleet is otherwise idle; mid-queue it would steal capacity
#: from items nobody has started yet.
SPEC_TAIL_MAX = 32
#: How long a speculative replica is remembered before another idle worker may
#: take over the race (a replica whose worker itself died must not lock the item
#: out of further speculation forever).
SPEC_REPLICA_TTL_S = 1800


class RedisWorkStream:
    def __init__(self, client, name, *, visibility_timeout_s=1800,
                 max_deliveries=3):
        self._c = client
        self.name = name
        self.visibility_timeout_s = visibility_timeout_s
        self.max_deliveries = max_deliveries
        self._k_stream = f"ws:{name}:stream"
        self._k_queued = f"ws:{name}:queued"
        self._k_inflight = f"ws:{name}:inflight"
        self._k_dlq = f"ws:{name}:dlq"
        self._k_spec = f"ws:{name}:spec"
        self._ensure_group()

    def _ensure_group(self):
        try:
            self._c.xgroup_create(self._k_stream, _GROUP, id="0",
                                  mkstream=True)
        except Exception as exc:          # BUSYGROUP = already exists
            if "BUSYGROUP" not in str(exc):
                raise

    def enqueue(self, item_key, payload):
        # SADD is the atomic dedup gate: only the first caller enqueues.
        # SADD→XADD is not atomic: a crash in between leaves a dedup
        # entry with no message (known small window; the PG rebuild is
        # the recovery path).
        if not self._c.sadd(self._k_queued, item_key):
            return False
        self._c.xadd(self._k_stream,
                     {"k": item_key, "p": json.dumps(payload)})
        return True

    def enqueue_many(self, items):
        """Batch variant of enqueue() (G3, review package 2026-07-05).

        A single enqueue() is one SADD + one XADD round trip; calling it
        once per item in a Python loop (e.g. from enqueue_bulk's ~50k-100k
        item producer hook) is 2*N synchronous Redis round trips inline in
        an async request handler, which can block the event loop for tens
        of seconds. This batches both round trips via pipelines, in chunks
        of a few hundred items to bound pipeline size on very large calls
        (mirrors the routes/workers.py enqueue_bulk endpoint's own
        _CHUNK=10000 IN-list chunking, scaled down since each item here
        costs two pipelined commands instead of one).

        Preserves per-item SADD-gates-XADD dedup semantics exactly:
        round 1 pipelines a SADD per item and collects which ones
        actually claimed the dedup slot (were not already queued/
        inflight); round 2 pipelines an XADD ONLY for those — an item
        that loses its SADD is silently skipped, precisely as a lone
        enqueue() call for it would have been (returns False, no XADD).
        A duplicate item_key appearing twice within ``items`` is likewise
        safe: pipelined commands still execute server-side in submission
        order, so the second SADD for the same key correctly sees it as
        already a member and only the first survives — matching two
        sequential enqueue() calls for the same key.

        Returns the number of items actually enqueued (dedup-skipped
        items don't count) — same per-item truthy contract as summing
        ``enqueue()``'s bool return over every item, but in
        2*ceil(N/chunk) round trips instead of 2*N.
        """
        items = list(items)
        if not items:
            return 0
        _CHUNK = 500
        enqueued = 0
        for start in range(0, len(items), _CHUNK):
            chunk = items[start:start + _CHUNK]
            pipe = self._c.pipeline(transaction=False)
            for item_key, _payload in chunk:
                pipe.sadd(self._k_queued, item_key)
            sadd_results = pipe.execute()
            to_add = [(item_key, payload) for (item_key, payload), added
                     in zip(chunk, sadd_results) if added]
            if not to_add:
                continue
            pipe = self._c.pipeline(transaction=False)
            for item_key, payload in to_add:
                pipe.xadd(self._k_stream,
                         {"k": item_key, "p": json.dumps(payload)})
            pipe.execute()
            enqueued += len(to_add)
        return enqueued

    def claim(self, consumer, keep_if=None):
        # 1) Steal stale PEL entries first (dead/stuck consumers).
        item = self._autoclaim(consumer, keep_if)
        if item is not None:
            return item
        # 2) Fresh messages.
        resp = self._c.xreadgroup(_GROUP, consumer,
                                  {self._k_stream: ">"}, count=1)
        if not resp:
            return None
        msg_id, fields = resp[0][1][0]
        return self._admit(consumer, msg_id, fields, deliveries=1)

    def _autoclaim(self, consumer, keep_if):
        # Enumerate stale PEL entries WITHOUT claiming: XPENDING with an
        # IDLE filter reports the true current owner and delivery count
        # without mutating ownership, so keep_if can veto the steal.
        # (XAUTOCLAIM — even with JUSTID — reassigns the entry to the
        # caller as a side effect, which would make keep_if always see
        # the caller as owner instead of the stale worker.)
        idle_ms = self.visibility_timeout_s * 1000
        try:
            pend = self._c.xpending_range(self._k_stream, _GROUP,
                                          min="-", max="+", count=8,
                                          idle=idle_ms)
        except Exception:
            return None
        for entry in pend:
            msg_id = entry["message_id"]
            deliveries = int(entry.get("times_delivered", 1))
            owner = entry.get("consumer", "")
            if keep_if is not None and keep_if(self._key_of(msg_id), owner):
                continue                       # legitimately long-running
            # min_idle_time=idle_ms keeps the steal race-safe: if another
            # process claimed (and reset the idle clock) since the
            # XPENDING read above, XCLAIM returns empty and we skip.
            claimed = self._c.xclaim(self._k_stream, _GROUP, consumer,
                                     min_idle_time=idle_ms,
                                     message_ids=[msg_id])
            if not claimed:
                continue
            _mid, fields = claimed[0]
            if deliveries >= self.max_deliveries:
                self._to_dlq(msg_id, fields)
                continue
            return self._admit(consumer, msg_id, fields,
                               deliveries=deliveries + 1)
        return None

    def claim_speculative(self, consumer, *, tail_max=SPEC_TAIL_MAX,
                          min_idle_s=None):
        """Hand an idle worker a COPY of an in-flight tail item. Returns a
        ClaimedItem, or None when speculation does not apply.

        THE PROBLEM (run s0014/lot20260714T175216): the FPP queue drained
        7488 -> 31 items in 17 minutes, then took 16 MORE minutes to finish
        those last 31 — half the phase for 0.4% of the work, in flat steps of
        zero progress. Once an item is claimed it is invisible to everyone
        else, so a slow (or frozen) worker holding one stalls the whole run
        while 79 other instances sit idle with nothing to do. TRICERATOPS cost
        varies by orders of magnitude between candidates, so a tail of
        stragglers is not an anomaly — it is the expected shape of every run.

        THE FIX (the classic straggler mitigation — Hadoop/Spark speculative
        execution): when the queue is empty and only a tail is left, give an
        idle worker the same item its current owner is still chewing on and let
        them race. Whoever finishes first wins. This does NOT take the item
        away from the owner (no PEL mutation, no XCLAIM): a slow-but-healthy
        worker keeps its progress and may still win — we are spending idle
        capacity, not confiscating work. That is what makes it safe to trigger
        at a much shorter idle threshold than an outright steal ever could be.

        Correctness rests on ack() being keyed by ITEM, not by owner: whichever
        worker POSTs its result first, routes/workers.py calls
        ws.ack(candidate_id), which XACKs+XDELs the entry out of the PEL — the
        loser's entry vanishes with it. When the loser eventually POSTs too, its
        write is an idempotent re-write of the same candidate row (the exact
        same thing that already happens today whenever a stolen item's original
        owner comes back), and its ack() is a no-op returning False.

        Guard rails:
          • queue must be EMPTY and in-flight <= *tail_max* — never speculate
            while unclaimed work exists; that would burn the fleet on copies
            instead of on items nobody has touched.
          • the item must have been idle >= *min_idle_s* (default: this
            stream's visibility timeout) — never duplicate an item that was
            just handed out and is very likely about to complete.
          • at most ONE live replica per item (HSETNX is the atomic gate, so
            the A3 uvicorn processes cannot each add their own), and a worker
            is never given a copy of an item it already owns.
        """
        if min_idle_s is None:
            min_idle_s = self.visibility_timeout_s
        try:
            st = self.stats()
            if st["pending"] == 0 or st["pending"] > tail_max or st["queued"] > 0:
                return None
            pend = self._c.xpending_range(
                self._k_stream, _GROUP, min="-", max="+",
                count=tail_max, idle=int(min_idle_s * 1000))
        except Exception:
            return None
        now = time.time()
        for entry in pend:
            msg_id = entry["message_id"]
            if str(entry.get("consumer", "")) == str(consumer):
                continue                       # never race yourself
            rng = self._c.xrange(self._k_stream, min=msg_id, max=msg_id)
            if not rng:
                continue                       # acked out from under us
            fields = rng[0][1]
            item_key = fields.get("k")
            if not item_key:
                continue
            if not self._take_replica_slot(item_key, consumer, now):
                continue                       # someone is already racing it
            return ClaimedItem(item_key, json.loads(fields.get("p", "{}")),
                               msg_id, int(entry.get("times_delivered", 1)),
                               speculative=True)
        return None

    def release_replica(self, item_key) -> None:
        """Give back a speculative slot taken by claim_speculative but never
        actually served (the caller found the item ineligible, or could not run
        it). Without this the item would stay flagged as "already being raced"
        for SPEC_REPLICA_TTL_S while nobody is in fact racing it."""
        try:
            self._c.hdel(self._k_spec, item_key)
        except Exception:
            pass

    def settle(self, item_key) -> str:
        """Adjudicate a result arriving for *item_key*. One of:

          "winner"  — this result retired the item (it was in flight; we acked).
          "loser"   — a SPECULATIVE race for this item was already won by
                      someone else. The caller MUST NOT persist this result.
          "unknown" — the stream knows nothing about this item: not in flight,
                      no race on record. The caller persists it as it always
                      has (legacy path, a manual re-run, or a result landing
                      after reconcile_stream() rebuilt the queue).

        Why a "loser" verdict has to exist at all: TRICERATOPS is a Monte Carlo
        and worker/fpp_native.py draws from np.random WITHOUT a seed, so two
        workers racing the same candidate produce two DIFFERENT FPP values. If
        both were written, the row would end up holding whichever POST arrived
        last over the network — not the one that won the race, and not
        reproducibly either. Speculation must cost duplicated CPU, never a
        nondeterministic value in the database (CLAUDE.md: reproducibility is
        non-negotiable). So the first result to arrive wins, and the loser's is
        dropped on the floor. Both are equally valid draws; what matters is that
        exactly one of them lands.

        The "loser" verdict is only ever returned for an item that was actually
        SPECULATED (the winner leaves a marker keyed on the replica having
        existed). An ordinary un-raced item that is no longer in flight yields
        "unknown" and is persisted exactly as before — this must stay true, or a
        result computed across a reconcile_stream() rebuild would be silently
        discarded.
        """
        try:
            had_replica = bool(self._c.hexists(self._k_spec, item_key))
            msg_id = self._c.hget(self._k_inflight, item_key)
            if msg_id is not None:
                if had_replica:
                    # Tell the losing racer, when it eventually POSTs, that this
                    # item is already decided. TTL: the loser is still computing
                    # right now, so the marker only has to outlive one in-flight
                    # TRICERATOPS run.
                    self._c.set(self._k_settled_of(item_key), "1",
                                ex=SPEC_REPLICA_TTL_S)
                self.ack(item_key)
                return "winner"
            if self._c.get(self._k_settled_of(item_key)):
                self._c.delete(self._k_settled_of(item_key))
                return "loser"
        except Exception:
            return "unknown"      # a Redis hiccup degrades to the old behaviour
        return "unknown"

    def _k_settled_of(self, item_key):
        return f"ws:{self.name}:settled:{item_key}"

    def _take_replica_slot(self, item_key, consumer, now) -> bool:
        """Claim THE one speculative-replica slot for *item_key*. Atomic.

        HSETNX is the gate, so two uvicorn processes racing here produce one
        replica, not two. An existing slot older than SPEC_REPLICA_TTL_S is
        stale (its replica worker died too) and is dropped so a live worker can
        take over the race; the HSETNX retry after that HDEL keeps even THAT
        handover single-winner.
        """
        mine = json.dumps({"consumer": str(consumer), "ts": now})
        if self._c.hsetnx(self._k_spec, item_key, mine):
            return True
        try:
            cur = json.loads(self._c.hget(self._k_spec, item_key) or "{}")
        except Exception:
            cur = {}
        if now - float(cur.get("ts", 0)) < SPEC_REPLICA_TTL_S:
            return False                       # a live replica already races it
        self._c.hdel(self._k_spec, item_key)
        return bool(self._c.hsetnx(self._k_spec, item_key, mine))

    def _key_of(self, msg_id):
        rng = self._c.xrange(self._k_stream, min=msg_id, max=msg_id)
        return rng[0][1].get("k", "") if rng else ""

    def _admit(self, consumer, msg_id, fields, deliveries):
        item_key = fields["k"]
        payload = json.loads(fields.get("p", "{}"))
        self._c.hset(self._k_inflight, item_key, msg_id)
        return ClaimedItem(item_key, payload, msg_id, deliveries)

    def _to_dlq(self, msg_id, fields):
        # RPUSH→XACK→XDEL→SREM/HDEL is not atomic: a crash mid-sequence
        # can leave a partially-moved entry (known small window; the PG
        # rebuild is the recovery path).
        self._c.rpush(self._k_dlq, json.dumps(
            {"k": fields.get("k"), "p": fields.get("p"),
             "msg_id": msg_id, "ts": time.time()}))
        self._c.xack(self._k_stream, _GROUP, msg_id)
        self._c.xdel(self._k_stream, msg_id)
        if fields.get("k"):
            self._c.srem(self._k_queued, fields["k"])
            self._c.hdel(self._k_inflight, fields["k"])
            self._c.hdel(self._k_spec, fields["k"])

    def ack(self, item_key):
        # The speculative-replica slot dies with the item, whichever of the two
        # racers acked it: the item is done, so nothing is left to race.
        self._c.hdel(self._k_spec, item_key)
        msg_id = self._c.hget(self._k_inflight, item_key)
        if msg_id is None:
            return False
        self._c.xack(self._k_stream, _GROUP, msg_id)
        self._c.xdel(self._k_stream, msg_id)
        self._c.hdel(self._k_inflight, item_key)
        self._c.srem(self._k_queued, item_key)
        return True

    def queued_keys(self) -> set:
        """Every item_key the stream is currently tracking — queued OR in-flight.

        ``ws:<name>:queued`` is added to on enqueue and removed from only on ack
        (or on the DLQ move), so it is exactly "what this stream still owes",
        which is what reconciliation must diff the DB against. Diffing against
        the queued-but-unclaimed part alone would re-enqueue every item a worker
        is busy computing right now.
        """
        return set(self._c.smembers(self._k_queued))

    def release_consumer(self, consumer) -> int:
        """Requeue every PEL entry still owned by *consumer*. Returns the count.

        Called when a worker PROCESS RESTARTS (routes/workers.py's
        /api/worker/register — a fresh process by definition holds no work).
        Without this, a claim made moments before the process died stays
        pending, unserved, until the visibility timeout: the fleet autoscaler
        restarts workers at the SCAN->FPP transition, and on 2026-07-14 it
        killed 7 workers ~40 s AFTER they had claimed FPP items. The new
        processes re-registered under the SAME worker_id, so to Redis the
        consumer looked alive while knowing nothing about those items — the
        run sat idle at FPP_BACKFILL for the whole timeout with an empty queue.
        The visibility timeout is the backstop; this is the fix.

        Implemented as ack + re-enqueue (Redis has no "un-claim"): the entry
        leaves the PEL and comes back as a fresh, immediately-deliverable
        message. Delivery count resets, which is correct — the item was never
        actually served to anyone.

        TOCTOU: between reading the PEL and acking, another worker could steal
        the same entry (its idle already exceeded the visibility timeout), and
        we would then re-enqueue an item that worker is now computing — a
        duplicate computation, never a lost item or a corrupt write (both
        backfill result paths are idempotent upserts on the candidate row).
        The window is milliseconds and the alternative (a Lua CAS on ownership)
        buys nothing that matters here.
        """
        try:
            pend = self._c.xpending_range(self._k_stream, _GROUP,
                                          min="-", max="+", count=1000,
                                          consumername=str(consumer))
        except Exception:
            return 0
        released = 0
        for entry in pend:
            msg_id = entry["message_id"]
            rng = self._c.xrange(self._k_stream, min=msg_id, max=msg_id)
            if not rng:
                # Already gone (acked/xdel'd by a racing path) — nothing to do.
                self._c.xack(self._k_stream, _GROUP, msg_id)
                continue
            fields = rng[0][1]
            item_key = fields.get("k")
            # Drop the old message, then re-add the SAME (key, payload) as a
            # new entry. srem/hdel first so enqueue()'s sadd de-dup gate (an
            # item already in `queued` is not re-added) cannot swallow it.
            self._c.xack(self._k_stream, _GROUP, msg_id)
            self._c.xdel(self._k_stream, msg_id)
            if item_key:
                self._c.hdel(self._k_inflight, item_key)
                self._c.hdel(self._k_spec, item_key)   # back in the queue for
                                                       # real: no race to run
                self._c.srem(self._k_queued, item_key)
                self.enqueue(item_key, json.loads(fields.get("p", "{}")))
                released += 1
        return released

    def stats(self):
        try:
            pending = self._c.xpending(self._k_stream, _GROUP)["pending"]
        except Exception:
            pending = 0
        return {
            "queued": self._c.xlen(self._k_stream) - pending,
            "pending": pending,
            "dlq": self._c.llen(self._k_dlq),
        }

    def peek_dlq(self):
        """Read every DLQ entry WITHOUT removing it — the DLQ stays a
        visible, inspectable audit trail (same guarantee as the plain
        FPP/ExoMiner DLQ). Callers that need to reconcile a mirror outside
        this module (e.g. the primary stream's PG-row reconciliation,
        Fase 1 Task 5) read via this method instead of popping, so the SAME
        entries can be safely re-scanned indefinitely — reconciliation is
        expected to be idempotent on the caller's side. Unbounded on
        purpose: a fixed window (e.g. "peek the oldest 50") would silently
        stop covering NEW arrivals forever once the DLQ grew past the
        window, since new entries are always appended past it. In practice
        this list is expected to stay small (max_deliveries exhaustion
        should be rare).
        """
        raw = self._c.lrange(self._k_dlq, 0, -1)
        return [json.loads(r) for r in raw]

    def remove_dlq(self, msg_id) -> bool:
        """Remove the ONE DLQ entry whose ``msg_id`` matches (G4, review
        package 2026-07-05).

        Called by a caller (routes/workers.py's _reconcile_primary_dlq)
        once it has ACTUALLY resolved the entry's PG row to a real
        transition — so the append-only DLQ list shrinks as rows are
        reconciled instead of growing forever. Matches by VALUE (the
        exact raw JSON string), not by index, so this stays correct even
        if the list mutated concurrently (another process appended or
        pruned entries) between the peek_dlq() call that found this
        entry and this call. ``count=1`` on LREM removes at most one
        occurrence, tolerating an — extremely unlikely — exact-duplicate
        payload without over-deleting a different entry that happens to
        share the same msg_id. Returns True if an entry was removed,
        False if none matched (already pruned by a concurrent caller, or
        the DLQ was cleared).
        """
        for raw in self._c.lrange(self._k_dlq, 0, -1):
            try:
                entry = json.loads(raw)
            except Exception:
                continue
            if entry.get("msg_id") == msg_id:
                return bool(self._c.lrem(self._k_dlq, 1, raw))
        return False

    def clear(self):
        self._c.delete(self._k_stream, self._k_queued,
                       self._k_inflight, self._k_dlq, self._k_spec)
        # The settled markers are one key per item (they carry a real Redis TTL,
        # unlike the hashes above), so they need a scan to go with the rest —
        # a rebuild must not leave a stale "this race is already decided" marker
        # behind to reject a legitimate future result for the same candidate.
        try:
            for k in self._c.scan_iter(match=f"ws:{self.name}:settled:*",
                                       count=500):
                self._c.delete(k)
        except Exception:
            pass
        self._ensure_group()


class MemoryWorkStream:
    """In-process stand-in for tests and single-process dev mode."""

    def __init__(self, name, *, visibility_timeout_s=1800, max_deliveries=3):
        self.name = name
        self.visibility_timeout_s = visibility_timeout_s
        self.max_deliveries = max_deliveries
        self._lock = threading.Lock()
        self._seq = 0
        self.clear()

    def clear(self):
        with getattr(self, "_lock", threading.Lock()):
            self._queue = []          # [(msg_id, item_key, payload)]
            self._pel = {}            # msg_id -> dict(item_key, payload,
                                      #   consumer, claimed_at, deliveries)
            self._queued = set()
            self._inflight = {}       # item_key -> msg_id
            self._dlq = []
            self._spec = {}           # item_key -> dict(consumer, ts)
            self._settled = {}        # item_key -> ts a speculative race was won
                                      #   (so the loser's late result is dropped)

    def enqueue(self, item_key, payload):
        with self._lock:
            if item_key in self._queued:
                return False
            self._queued.add(item_key)
            self._seq += 1
            self._queue.append((str(self._seq), item_key, dict(payload)))
            return True

    def enqueue_many(self, items):
        """Batch variant of enqueue() (G3, review package 2026-07-05).

        Single-process backend — there is no round-trip cost to batch
        away, so this is a trivial loop under ONE lock acquisition
        (instead of one lock acquisition per item), preserving the exact
        same per-item dedup semantics as calling enqueue() N times.
        Returns the count of items actually enqueued.
        """
        enqueued = 0
        with self._lock:
            for item_key, payload in items:
                if item_key in self._queued:
                    continue
                self._queued.add(item_key)
                self._seq += 1
                self._queue.append((str(self._seq), item_key, dict(payload)))
                enqueued += 1
        return enqueued

    def claim(self, consumer, keep_if=None):
        with self._lock:
            now = time.time()
            for msg_id, ent in list(self._pel.items()):
                if now - ent["claimed_at"] < self.visibility_timeout_s:
                    continue
                if keep_if is not None and keep_if(ent["item_key"],
                                                   ent["consumer"]):
                    continue
                if ent["deliveries"] >= self.max_deliveries:
                    self._dlq.append(dict(ent, msg_id=msg_id, ts=now))
                    del self._pel[msg_id]
                    self._queued.discard(ent["item_key"])
                    self._inflight.pop(ent["item_key"], None)
                    self._spec.pop(ent["item_key"], None)
                    continue
                ent.update(consumer=consumer, claimed_at=now,
                           deliveries=ent["deliveries"] + 1)
                self._inflight[ent["item_key"]] = msg_id
                return ClaimedItem(ent["item_key"], dict(ent["payload"]),
                                   msg_id, ent["deliveries"])
            if not self._queue:
                return None
            msg_id, item_key, payload = self._queue.pop(0)
            self._pel[msg_id] = {"item_key": item_key, "payload": payload,
                                 "consumer": consumer, "claimed_at": now,
                                 "deliveries": 1}
            self._inflight[item_key] = msg_id
            return ClaimedItem(item_key, dict(payload), msg_id, 1)

    def claim_speculative(self, consumer, *, tail_max=SPEC_TAIL_MAX,
                          min_idle_s=None):
        """Same contract as RedisWorkStream.claim_speculative (read it there for
        the straggler problem this solves and the race semantics): serve a COPY
        of a tail item to an idle worker, WITHOUT touching the PEL entry, and
        let the two race to ack it."""
        if min_idle_s is None:
            min_idle_s = self.visibility_timeout_s
        with self._lock:
            now = time.time()
            if (not self._pel or len(self._pel) > tail_max or self._queue):
                return None
            for msg_id, ent in self._pel.items():
                if now - ent["claimed_at"] < min_idle_s:
                    continue
                if str(ent["consumer"]) == str(consumer):
                    continue           # never race yourself
                rep = self._spec.get(ent["item_key"])
                if rep and now - rep["ts"] < SPEC_REPLICA_TTL_S:
                    continue           # a live replica already races it
                self._spec[ent["item_key"]] = {"consumer": str(consumer),
                                               "ts": now}
                return ClaimedItem(ent["item_key"], dict(ent["payload"]),
                                   msg_id, ent["deliveries"],
                                   speculative=True)
            return None

    def release_replica(self, item_key) -> None:
        """Same contract as RedisWorkStream.release_replica."""
        with self._lock:
            self._spec.pop(item_key, None)

    def settle(self, item_key) -> str:
        """Same contract as RedisWorkStream.settle — read it there for WHY a
        speculative loser's result must be dropped rather than written."""
        with self._lock:
            now = time.time()
            had_replica = item_key in self._spec
            msg_id = self._inflight.get(item_key)
            if msg_id is not None:
                if had_replica:
                    self._settled[item_key] = now
                self._spec.pop(item_key, None)
                self._inflight.pop(item_key, None)
                self._pel.pop(msg_id, None)
                self._queued.discard(item_key)
                return "winner"
            ts = self._settled.get(item_key)
            if ts is not None and now - ts < SPEC_REPLICA_TTL_S:
                del self._settled[item_key]
                return "loser"
        return "unknown"

    def ack(self, item_key):
        with self._lock:
            self._spec.pop(item_key, None)
            msg_id = self._inflight.pop(item_key, None)
            if msg_id is None:
                return False
            self._pel.pop(msg_id, None)
            self._queued.discard(item_key)
            return True

    def queued_keys(self) -> set:
        """Same contract as RedisWorkStream.queued_keys: queued OR in-flight."""
        with self._lock:
            return set(self._queued)

    def release_consumer(self, consumer) -> int:
        """Same contract as RedisWorkStream.release_consumer (see it for the
        2026-07-14 incident this exists for): move every PEL entry owned by
        *consumer* back to the head of the queue, immediately deliverable.

        Front of the queue, not the back: these items were claimed BEFORE
        anything still queued, and re-queueing them behind a large backlog
        would starve exactly the items a restart already delayed.
        """
        with self._lock:
            released = 0
            for msg_id, ent in list(self._pel.items()):
                if str(ent["consumer"]) != str(consumer):
                    continue
                del self._pel[msg_id]
                self._inflight.pop(ent["item_key"], None)
                self._spec.pop(ent["item_key"], None)   # back in the queue for
                                                        # real: no race to run
                self._queue.insert(0, (msg_id, ent["item_key"],
                                       dict(ent["payload"])))
                self._queued.add(ent["item_key"])
                released += 1
            return released

    def stats(self):
        with self._lock:
            return {"queued": len(self._queue), "pending": len(self._pel),
                    "dlq": len(self._dlq)}

    def peek_dlq(self):
        """Same contract as RedisWorkStream.peek_dlq() — normalises this
        backend's internal DLQ entry shape (keys item_key/payload, payload
        as a live dict) to the SAME {"k","p","msg_id","ts"} shape Redis
        produces (p as a JSON STRING in both cases), so callers get one
        uniform contract regardless of backend."""
        with self._lock:
            return [{"k": e.get("item_key"), "p": json.dumps(e.get("payload", {})),
                     "msg_id": e.get("msg_id"), "ts": e.get("ts")}
                    for e in self._dlq]

    def remove_dlq(self, msg_id) -> bool:
        """Memory-backend equivalent of RedisWorkStream.remove_dlq (G4,
        review package 2026-07-05) — same contract: remove the one entry
        matching ``msg_id``, return whether one was found."""
        with self._lock:
            for i, e in enumerate(self._dlq):
                if e.get("msg_id") == msg_id:
                    del self._dlq[i]
                    return True
            return False


_streams = {}
_streams_lock = threading.Lock()

#: per-stage tuning — max_deliveries matches MAX_TASK_ATTEMPTS-style retry caps.
#:
#: visibility_timeout_s is the ONLY clock that recovers an orphaned PEL entry:
#: an item claimed by a worker process that dies before acking stays pending,
#: unserved, until this many seconds of idle elapse (2026-07-14: the fleet
#: autoscaler restarts the fleet at the SCAN→FPP transition, and it killed 7
#: workers ~40s after they had claimed FPP items — the run then sat idle at
#: FPP_BACKFILL for the full timeout with nothing else in the queue).
#:
#: fpp is cut to 300s (2026-07-14, the FPP tail): a legitimately long-running
#: item is NOT at risk of being stolen at that mark, because its keep_if
#: (server._worker_alive_and_on) vetoes the steal while the owning worker is
#: alive and still on that TIC — up to BACKFILL_HOLD_CEILING_SECONDS. So this
#: timeout governs exactly two things, both of which we want fast:
#:   • how quickly a genuinely ORPHANED item (worker process died between
#:     claim and ack) returns to the queue — was 15 min, now 5;
#:   • when the tail becomes eligible for SPECULATIVE re-serving to an idle
#:     worker (claim_speculative uses this as its min-idle threshold).
#: exominer stays at 900s: its tail has not been measured, and one change per
#: iteration.
#:
#: primary: 1800s -> 900s (2026-08-07, operator decision). The 1800 guarded against
#: stealing from Python workers whose heartbeat stalls under the GIL during
#: heavy BLS/injection (legit 5-7 min silences). The primary fleet is now
#: ~96% native (Rust/Swift; heartbeat on its own thread), keep_if still
#: vetoes any steal while the owner is alive and on the TIC, and 900s keeps
#: 2x margin over the documented worst legit silence of the remaining
#: Python scanners (laptop/M1). Worst downside: duplicate compute.
STAGE_SETTINGS = {
    "fpp":      {"visibility_timeout_s": 300, "max_deliveries": 3},
    "exominer": {"visibility_timeout_s": 900, "max_deliveries": 3},
    "primary":  {"visibility_timeout_s": 900, "max_deliveries": 3},
}


def get_stream(name):
    """Per-process singleton factory. Redis-backed when shared_state is
    Redis (A3 multi-process prod), in-memory otherwise (tests, dev)."""
    with _streams_lock:
        ws = _streams.get(name)
        if ws is not None:
            return ws
        from exonit.shared_state import RedisBackend, get_shared_state
        backend = get_shared_state()
        kw = STAGE_SETTINGS.get(name, {})
        if isinstance(backend, RedisBackend):
            ws = RedisWorkStream(backend.raw_client, name, **kw)
        else:
            ws = MemoryWorkStream(name, **kw)
        _streams[name] = ws
        return ws
