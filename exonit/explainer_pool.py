"""Shared-queue work pool for distributing LoRA explanation calls across
several explainer endpoints (design: docs/superpowers/plans — "fase LoRA
als 5 Mac mini M4", part 1 "Work distribution").

Why a shared queue and not a partition by ``id % N`` (the <host> precedent,
memory ``reference_lora_turbo_distributed_pcmarc``): a static partition
leaves a dead host's share undone and turns the tail of the run into a
straggler queue.  A shared queue lets every live endpoint keep pulling work
until the backlog is exhausted, and self-balances when one host is slower.

Design
------
* One shared work list guarded by a ``threading.Condition``; ``concurrency``
  consumer threads per endpoint pull from it.  Order is not preserved (the
  caller's flush-by-batch loop does not depend on it).  A condition variable
  (rather than a plain ``queue.Queue``) is what makes the "avoid the endpoint
  that just failed this item" rule deterministic: a thread that finds only
  items excluded for itself waits (releasing the lock) instead of looping
  back onto the same item — giving another, non-excluded endpoint's thread
  the lock and the item first. If nobody else claims it within one poll
  window, it is taken anyway (never hangs on a single straggler item).
* A failed call increments the item's ``attempts`` and the endpoint's
  ``consecutive_failures``; the item is requeued (up to ``max_item_attempts``)
  and — when other endpoints are alive — a consumer avoids immediately
  retrying the item on the endpoint that just failed it.
* An endpoint with ``consecutive_failures >= dead_after`` is marked dead: its
  threads stop pulling new work.  A background monitor probes dead endpoints
  every ``probe_every`` seconds (``GET {base}/healthz``, ``upstream=="ok"``)
  and revives them.
* If *all* endpoints are dead, a grace timer (``all_dead_grace``) gives the
  prober a last chance to revive one; if it expires, every item still queued
  is marked failed (``ok=False``) — the pool always terminates, it never
  hangs.
* An optional ``max_seconds`` deadline: once elapsed, no new item is
  dispatched (items still queued are finalized as *skipped* — not counted as
  failures, they simply were never attempted and are left for the next run);
  in-flight calls are allowed to finish normally.
* ``call_fn`` and the healthz prober are both injectable so unit tests never
  touch the network.
* The "is dispatch closed?" check and the "requeue this failed item" mutation
  in the retry path are one atomic critical section under ``self._work_cv``
  — the same lock the deadline/all-dead-grace drains use to close dispatch
  and empty the work list. This closes a TOCTOU race found in review
  (2026-09-26): checking the closed-flag outside the lock, followed by a
  separate lock acquisition to append, could let a drain run to completion
  in between, silently orphaning the requeued item forever and hanging
  ``run()``. An ``_in_flight`` counter (mutated under the same lock as
  ``_work_list``) and a watchdog timeout in ``run()`` (``_is_quiescent()``)
  are defence in depth on top of that fix, in case of a future bug of the
  same shape: rather than blocking on ``self._output.get()`` forever, a
  quiescent pool with results still missing logs loudly and synthesizes
  failures for the missing payloads.

Usage
-----
    pool = EndpointPool(endpoints, call_fn=my_call_fn, timeout=90.0)
    for result in pool.run(payloads):
        ...  # result.ok / result.skipped / result.value / result.error
    pool.stats()  # {endpoint_name: {"ok":.., "fail":.., "retries":.., ...}}
    pool.deadline_hit  # True if max_seconds cut the run short
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

logger = logging.getLogger(__name__)

__all__ = ["Endpoint", "PoolResult", "EndpointPool"]

# Poll granularity for queue.get()/Event.wait() calls used purely so a
# thread can notice shutdown/topology-change flags — not busy-waiting
# (Queue.get(timeout=...) blocks and releases the GIL between checks).
_QUEUE_POLL_INTERVAL = 0.05
_MONITOR_MAX_TICK = 1.0

# run()'s own watchdog poll interval on self._output.get() — defence in
# depth (2026-09-26 review): if some future bug ever loses an item despite
# the atomic dispatch-closed/requeue section below, run() must not block
# forever waiting for a result that will never arrive.
_RUN_WATCHDOG_INTERVAL = 2.0

_COUNTER_KEYS = ("ok", "fail", "retries", "deaths", "revivals")


@dataclass(frozen=True)
class Endpoint:
    """Public, read-only view of an endpoint passed to ``call_fn``."""

    url: str
    name: str
    concurrency: int


@dataclass
class PoolResult:
    """One outcome per submitted payload (exactly one, always)."""

    payload: Any
    ok: bool
    value: Any = None
    error: str | None = None
    endpoint: str | None = None
    attempts: int = 0
    skipped: bool = False
    index: int = -1


@dataclass
class _WorkItem:
    payload: Any
    index: int = -1
    attempts: int = 0
    last_endpoint: str | None = None
    skip_count: int = 0


class _EndpointState:
    """Internal mutable state for one endpoint (locked, never exposed)."""

    def __init__(self, cfg: dict, index: int):
        self.url: str = cfg["url"]
        self.concurrency: int = max(1, int(cfg.get("concurrency", 1)))
        self.name: str = cfg.get("name") or f"ep{index}:{self.url}"
        self.alive: bool = True
        self.consecutive_failures: int = 0
        self.next_probe_time: float = 0.0
        self.lock = threading.Lock()
        # Set while alive; cleared on death so parked threads can Event.wait()
        # instead of polling in a tight loop.
        self.revived_event = threading.Event()
        self.revived_event.set()
        self.counters = {k: 0 for k in _COUNTER_KEYS}

    def public(self) -> Endpoint:
        return Endpoint(url=self.url, name=self.name, concurrency=self.concurrency)


def _default_probe(url: str, timeout: float = 5.0) -> bool:
    """Default healthz prober: derive ``/healthz`` from an ``/api/explain``
    URL exactly like the generator's own startup health check does, and
    require ``upstream == "ok"``. Only imports ``requests`` when actually
    used (no network in unit tests, which inject a fake ``probe_fn``)."""
    import requests

    health_url = url.replace("/api/explain", "/healthz")
    r = requests.get(health_url, timeout=timeout)
    if r.status_code != 200:
        return False
    return r.json().get("upstream") == "ok"


class EndpointPool:
    """Distribute a batch of payloads across several endpoints via a shared
    queue.  See module docstring for the full design."""

    def __init__(
        self,
        endpoints: Iterable[dict],
        call_fn: Callable[[Endpoint, Any], Any],
        timeout: float = 90.0,
        *,
        max_item_attempts: int = 3,
        dead_after: int = 3,
        probe_every: float = 60.0,
        all_dead_grace: float = 600.0,
        max_seconds: float | None = None,
        probe_fn: Callable[[str], bool] | None = None,
        now_fn: Callable[[], float] = time.monotonic,
    ):
        endpoints = list(endpoints)
        if not endpoints:
            raise ValueError("EndpointPool requires at least one endpoint")
        self._endpoints = [_EndpointState(e, i) for i, e in enumerate(endpoints)]
        self._call_fn = call_fn
        self._timeout = timeout
        self._max_item_attempts = max(1, int(max_item_attempts))
        self._dead_after = max(1, int(dead_after))
        self._probe_every = probe_every
        self._all_dead_grace = all_dead_grace
        self._max_seconds = max_seconds
        self._probe_fn = probe_fn or _default_probe
        self._now = now_fn

        self._work_cv = threading.Condition()
        self._work_list: list[_WorkItem] = []
        # Items popped from _work_list by _acquire_item but not yet finalized
        # or re-queued — i.e. genuinely mid-call_fn. Always mutated under
        # self._work_cv, together with _work_list, so a single lock acquisition
        # gives an atomic, consistent snapshot of "how many items are neither
        # finalized nor sitting in the list" (used by the dispatch-closed
        # race fix below and by run()'s quiescence watchdog).
        self._in_flight = 0
        self._output: "queue.Queue[PoolResult]" = queue.Queue()

        self._stop_event = threading.Event()
        self._dispatch_closed_event = threading.Event()
        self._topology_changed_event = threading.Event()
        # Which closer tripped _dispatch_closed_event: all-dead-grace expiry
        # produces real failures (ok=False, skipped=False); the max_seconds
        # deadline produces benign skips (not counted as failures). Set
        # strictly before _dispatch_closed_event in _force_fail_remaining so
        # any consumer thread that observes the event set — under the work_cv
        # lock — also observes the correct flavour.
        self._all_dead_triggered = False

        self._all_dead_since: float | None = None
        self._deadline_hit = False
        self._start_time: float | None = None

        self._threads: list[threading.Thread] = []
        self._monitor_thread: threading.Thread | None = None

        # TESTING ONLY. When set, called synchronously in a consumer thread
        # right before the atomic dispatch-closed-check/requeue section in
        # _process(), so tests can force the exact adversarial interleaving
        # a dispatch-closing drain races against a retry — deterministically,
        # with no sleeps. Must never be set outside tests.
        self._test_pre_requeue_hook: Callable[[], None] | None = None
        # TESTING ONLY. When set, called synchronously right after a drain
        # (_close_dispatch_and_drain_skipped/_force_fail_remaining) reserves
        # the drained items against _in_flight but before finalizing any of
        # them — lets tests deterministically observe/pause that exact
        # window (e.g. to prove run()'s watchdog cannot see a false
        # "quiescent" state while items are merely still being finalized).
        # Must never be set outside tests.
        self._test_mid_drain_hook: Callable[[], None] | None = None

    # -- introspection --------------------------------------------------

    @property
    def deadline_hit(self) -> bool:
        return self._deadline_hit

    def stats(self) -> dict[str, dict[str, int]]:
        return {e.name: dict(e.counters) for e in self._endpoints}

    def _alive_count(self) -> int:
        return sum(1 for e in self._endpoints if e.alive)

    # -- public API -------------------------------------------------------

    def run(self, payloads: Iterable[Any]) -> Iterator[PoolResult]:
        """Submit every payload and yield exactly one :class:`PoolResult`
        per payload (order not guaranteed). Always terminates: bounded by
        ``max_item_attempts`` retries, ``all_dead_grace`` and, if set,
        ``max_seconds``.

        Defence in depth: ``self._output.get()`` is called with a timeout in
        a loop, not blocked on forever. If it ever times out while the pool
        is quiescent (dispatch closed, nothing queued, nothing in flight)
        and results are still missing, that is a bug (every item should
        already have been finalized) — it is logged loudly and the missing
        payloads are synthesized as failures instead of hanging the caller.
        """
        payloads = list(payloads)
        n = len(payloads)
        self._start_time = self._now()
        with self._work_cv:
            self._work_list.extend(_WorkItem(payload=p, index=i)
                                   for i, p in enumerate(payloads))
            self._work_cv.notify_all()

        for ep in self._endpoints:
            for _ in range(ep.concurrency):
                t = threading.Thread(target=self._consumer_loop, args=(ep,),
                                     daemon=True)
                t.start()
                self._threads.append(t)
        self._monitor_thread = threading.Thread(target=self._monitor_loop,
                                                 daemon=True)
        self._monitor_thread.start()

        pending = set(range(n))
        try:
            while pending:
                try:
                    result = self._output.get(timeout=_RUN_WATCHDOG_INTERVAL)
                except queue.Empty:
                    if self._is_quiescent():
                        logger.error(
                            "EndpointPool.run(): %d item(s) never finalized "
                            "(%s) while the pool is quiescent (dispatch "
                            "closed, nothing queued, nothing in flight) — "
                            "this indicates a bug in EndpointPool, not in "
                            "call_fn. Synthesizing failures instead of "
                            "hanging.", len(pending), sorted(pending),
                        )
                        for idx in sorted(pending):
                            yield PoolResult(
                                payload=payloads[idx], ok=False, skipped=False,
                                endpoint=None, attempts=0, index=idx,
                                error="internal: item lost by EndpointPool "
                                      "(synthesized by run()'s watchdog)",
                            )
                        pending.clear()
                    continue
                pending.discard(result.index)
                yield result
        finally:
            self.shutdown()

    def _is_quiescent(self) -> bool:
        """True if dispatch is closed and there is nothing left anywhere:
        not queued, not mid-call_fn. Used only by run()'s watchdog."""
        if not self._dispatch_closed_event.is_set():
            return False
        with self._work_cv:
            return not self._work_list and self._in_flight == 0

    def shutdown(self, join_timeout: float = 5.0) -> None:
        """Signal every thread to stop and join them. Idempotent."""
        self._stop_event.set()
        self._dispatch_closed_event.set()
        self._topology_changed_event.set()
        for e in self._endpoints:
            e.revived_event.set()
        with self._work_cv:
            self._work_cv.notify_all()
        for t in self._threads:
            t.join(timeout=join_timeout)
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=join_timeout)

    # -- finalization -------------------------------------------------------

    def _finalize(self, item: _WorkItem, *, ok: bool, skipped: bool,
                  endpoint_name: str | None, value: Any, error: str | None) -> None:
        self._output.put(PoolResult(
            payload=item.payload, ok=ok, value=value, error=error,
            endpoint=endpoint_name, attempts=item.attempts, skipped=skipped,
            index=item.index,
        ))

    def _dec_in_flight(self) -> None:
        with self._work_cv:
            self._in_flight -= 1

    # -- consumer threads ---------------------------------------------------

    def _consumer_loop(self, ep: _EndpointState) -> None:
        while not self._stop_event.is_set():
            if not ep.alive:
                # Parked: wait for revival or shutdown, no polling loop.
                ep.revived_event.wait(timeout=_MONITOR_MAX_TICK)
                continue
            if self._dispatch_closed_event.is_set():
                return
            item = self._acquire_item(ep)
            if item is None:
                continue
            self._process(ep, item)

    def _acquire_item(self, ep: _EndpointState) -> "_WorkItem | None":
        """Pop the next item this endpoint may work on, or ``None`` on a
        (bounded) empty/deferred wait — the caller just loops and re-checks
        stop/alive/dispatch flags.

        If every remaining item was *just* failed by this same endpoint and
        another endpoint is alive, this thread waits (releasing the lock) for
        up to one poll window so that other endpoint's consumer gets first
        dibs — deterministically, since ``Condition.wait`` hands the lock to
        another waiter rather than racing a plain queue.  If nobody claims it
        within the window, it is taken anyway (never hangs on a straggler).
        """
        deadline = self._now() + _QUEUE_POLL_INTERVAL
        with self._work_cv:
            while True:
                if self._stop_event.is_set() or self._dispatch_closed_event.is_set():
                    return None
                alive_count = self._alive_count()
                chosen_idx = None
                only_excluded = bool(self._work_list)
                for i, item in enumerate(self._work_list):
                    excluded = (item.last_endpoint == ep.name and alive_count > 1
                               and item.skip_count < alive_count - 1)
                    if not excluded:
                        chosen_idx = i
                        only_excluded = False
                        break
                if chosen_idx is not None:
                    item = self._work_list.pop(chosen_idx)
                    item.skip_count = 0
                    self._in_flight += 1
                    return item
                remaining = deadline - self._now()
                if remaining <= 0:
                    if only_excluded:
                        # Fallback: take the excluded item anyway rather
                        # than wait forever for an endpoint that never comes.
                        item = self._work_list.pop(0)
                        item.skip_count += 1
                        self._in_flight += 1
                        return item
                    return None
                self._work_cv.wait(timeout=remaining)

    def _process(self, ep: _EndpointState, item: _WorkItem) -> None:
        public_ep = ep.public()
        try:
            value = self._call_fn(public_ep, item.payload)
            ok, error = True, None
        except Exception as exc:  # noqa: BLE001 - call_fn's failure mode is opaque
            value, ok, error = None, False, repr(exc)

        if ok:
            with ep.lock:
                ep.consecutive_failures = 0
                ep.counters["ok"] += 1
            self._finalize(item, ok=True, skipped=False, endpoint_name=ep.name,
                           value=value, error=None)
            self._dec_in_flight()
            return

        item.attempts += 1
        item.last_endpoint = ep.name
        died = False
        with ep.lock:
            ep.counters["fail"] += 1
            ep.consecutive_failures += 1
            if ep.alive and ep.consecutive_failures >= self._dead_after:
                ep.alive = False
                ep.counters["deaths"] += 1
                ep.revived_event.clear()
                ep.next_probe_time = self._now() + self._probe_every
                died = True
        if died:
            self._topology_changed_event.set()

        if item.attempts >= self._max_item_attempts:
            self._finalize(item, ok=False, skipped=False, endpoint_name=ep.name,
                           value=None, error=error)
            self._dec_in_flight()
            return

        # TESTING ONLY seam (see __init__): lets a test force the exact
        # adversarial interleaving below deterministically.
        if self._test_pre_requeue_hook is not None:
            self._test_pre_requeue_hook()

        # The "is dispatch closed?" check and the "put the item back in
        # _work_list" mutation MUST be one atomic critical section under
        # self._work_cv — the same lock _close_dispatch_and_drain_skipped()/
        # _force_fail_remaining() use to drain the list. A check-then-act
        # split here (event checked outside the lock, append happening
        # after further lock acquisitions) is a TOCTOU race: a drain could
        # run to completion in between, permanently emptying the list and
        # closing dispatch, and then this append would silently resurrect
        # an item nothing will ever pick up again — orphaning it forever and
        # hanging run()'s `self._output.get()` (2026-09-26 review finding,
        # reproduced deterministically and fixed here).
        with self._work_cv:
            if self._dispatch_closed_event.is_set():
                dispatch_closed = True
                all_dead = self._all_dead_triggered
            else:
                self._work_list.append(item)
                self._in_flight -= 1
                self._work_cv.notify_all()
                dispatch_closed = False
                all_dead = False

        if dispatch_closed:
            if all_dead:
                # All endpoints dead and the grace period already expired
                # (or expired in the instant between this call's failure and
                # this check): a real failure, not a benign skip.
                self._finalize(item, ok=False, skipped=False, endpoint_name=ep.name,
                               value=None, error="all_endpoints_dead")
            else:
                # Deadline hit while this call was in flight: don't dispatch
                # a retry, the item was still "attempted" but is left for
                # next run.
                self._finalize(item, ok=False, skipped=True, endpoint_name=ep.name,
                               value=None, error="deadline")
            # finalize-then-decrement (same order as the ok-path and the
            # max-attempts path above) so _in_flight only reaches 0 once the
            # result is actually visible on the output queue.
            self._dec_in_flight()
            return

        with ep.lock:
            ep.counters["retries"] += 1

    # -- monitor thread: probes, deadline, all-dead grace --------------------

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            now = self._now()
            candidates: list[float] = []
            if (self._max_seconds is not None
                    and not self._dispatch_closed_event.is_set()):
                candidates.append(self._start_time + self._max_seconds - now)
            for e in self._endpoints:
                if not e.alive:
                    candidates.append(e.next_probe_time - now)
            if self._all_dead_since is not None:
                candidates.append(self._all_dead_since + self._all_dead_grace - now)
            wait_for = min(candidates) if candidates else _MONITOR_MAX_TICK
            wait_for = max(0.0, min(wait_for, _MONITOR_MAX_TICK))

            if self._topology_changed_event.wait(timeout=wait_for):
                self._topology_changed_event.clear()
            if self._stop_event.is_set():
                return

            now = self._now()

            if (self._max_seconds is not None
                    and not self._dispatch_closed_event.is_set()
                    and now >= self._start_time + self._max_seconds):
                self._deadline_hit = True
                self._close_dispatch_and_drain_skipped()

            for e in self._endpoints:
                if not e.alive and now >= e.next_probe_time:
                    e.next_probe_time = now + self._probe_every
                    try:
                        revived = self._probe_fn(e.url)
                    except Exception:
                        revived = False
                    if revived:
                        with e.lock:
                            e.alive = True
                            e.consecutive_failures = 0
                            e.counters["revivals"] += 1
                        e.revived_event.set()
                        self._topology_changed_event.set()

            if self._alive_count() == 0:
                if self._all_dead_since is None:
                    self._all_dead_since = now
                elif now - self._all_dead_since >= self._all_dead_grace:
                    self._force_fail_remaining()
                    return
            else:
                self._all_dead_since = None

    def _close_dispatch_and_drain_skipped(self) -> None:
        self._dispatch_closed_event.set()
        with self._work_cv:
            drained, self._work_list[:] = self._work_list[:], []
            # Reserve the drained items against _in_flight *inside* the same
            # lock acquisition that empties _work_list — otherwise there is a
            # window, between here and the finalize loop below actually
            # putting each result, where _is_quiescent() sees an empty list
            # AND _in_flight == 0 (nothing reserved yet) even though these
            # items are not finalized yet. run()'s watchdog polling in that
            # window would wrongly conclude they were lost and synthesize
            # "internal: item lost" failures instead of the correct
            # skipped=True/error="deadline" (2026-09-26 review delta finding
            # on 6bbdbed). Released one by one below as each is finalized.
            self._in_flight += len(drained)
            self._work_cv.notify_all()
        if self._test_mid_drain_hook is not None:
            self._test_mid_drain_hook()
        for item in drained:
            self._finalize(item, ok=False, skipped=True, endpoint_name=None,
                           value=None, error="deadline")
            self._dec_in_flight()

    def _force_fail_remaining(self) -> None:
        # Order matters: set the flavour flag strictly before the event, so
        # any consumer thread that observes _dispatch_closed_event set (under
        # self._work_cv, in _process's atomic section) also observes the
        # correct flavour — same happens-before reasoning as the event/drain
        # ordering below.
        self._all_dead_triggered = True
        self._dispatch_closed_event.set()
        with self._work_cv:
            drained, self._work_list[:] = self._work_list[:], []
            # See _close_dispatch_and_drain_skipped() above for why this
            # reservation is needed.
            self._in_flight += len(drained)
            self._work_cv.notify_all()
        if self._test_mid_drain_hook is not None:
            self._test_mid_drain_hook()
        for item in drained:
            self._finalize(item, ok=False, skipped=False, endpoint_name=None,
                           value=None, error="all_endpoints_dead")
            self._dec_in_flight()
