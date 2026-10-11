"""Tests for exonit/explainer_pool.py — EndpointPool.

All tests use a fake ``call_fn`` (and, where relevant, a fake ``probe_fn``):
no network, no real sleeping beyond tiny thresholds chosen so the whole
suite runs in well under a second per test.
"""
from __future__ import annotations

import threading
import time

import pytest

import exonit.explainer_pool as explainer_pool_module
from exonit.explainer_pool import Endpoint, EndpointPool, PoolResult


def _endpoints(*specs):
    """specs: list of (url, concurrency, name)."""
    return [{"url": u, "concurrency": c, "name": n} for u, c, n in specs]


# ---------------------------------------------------------------------------
# Basic distribution: one endpoint, all succeed.
# ---------------------------------------------------------------------------

class TestSingleEndpointEquivalence:
    """A single endpoint with concurrency=N must behave like the previous
    ThreadPoolExecutor(max_workers=N).map(...) path: every payload is
    processed exactly once, all succeed, no retries, no deaths."""

    def test_all_payloads_processed_exactly_once(self):
        seen = []
        lock = threading.Lock()

        def call_fn(ep: Endpoint, payload):
            with lock:
                seen.append(payload)
            return f"explained-{payload}"

        pool = EndpointPool(_endpoints(("http://h1/api/explain", 4, "h1")),
                            call_fn=call_fn)
        results = list(pool.run(range(50)))

        assert len(results) == 50
        assert sorted(seen) == list(range(50))
        assert all(r.ok and not r.skipped for r in results)
        assert {r.value for r in results} == {f"explained-{i}" for i in range(50)}
        stats = pool.stats()
        assert stats["h1"]["ok"] == 50
        assert stats["h1"]["fail"] == 0
        assert stats["h1"]["deaths"] == 0

    def test_concurrency_is_respected_upper_bound(self):
        """With concurrency=1 only one call_fn invocation is in flight at a time."""
        in_flight = {"n": 0, "max": 0}
        lock = threading.Lock()

        def call_fn(ep, payload):
            with lock:
                in_flight["n"] += 1
                in_flight["max"] = max(in_flight["max"], in_flight["n"])
            time.sleep(0.01)
            with lock:
                in_flight["n"] -= 1
            return payload

        pool = EndpointPool(_endpoints(("http://h1/api/explain", 1, "h1")),
                            call_fn=call_fn)
        list(pool.run(range(10)))
        assert in_flight["max"] == 1


class TestDistributionByConcurrency:

    def test_two_endpoints_both_get_work(self):
        counts = {"a": 0, "b": 0}
        lock = threading.Lock()

        def call_fn(ep: Endpoint, payload):
            with lock:
                counts[ep.name] += 1
            time.sleep(0.005)
            return payload

        pool = EndpointPool(
            _endpoints(("http://a/api/explain", 4, "a"),
                      ("http://b/api/explain", 2, "b")),
            call_fn=call_fn,
        )
        results = list(pool.run(range(60)))
        assert len(results) == 60
        assert all(r.ok for r in results)
        # Both endpoints must have done real work (shared queue self-balances,
        # but a 4-vs-2 split with a slow-ish call should give both a share).
        assert counts["a"] > 0
        assert counts["b"] > 0
        assert counts["a"] + counts["b"] == 60


# ---------------------------------------------------------------------------
# Retry / re-queue to another endpoint.
# ---------------------------------------------------------------------------

class TestRetryAndRequeue:

    def test_failure_retried_on_another_endpoint_and_succeeds(self):
        """endpoint 'bad' always fails; endpoint 'good' always succeeds.
        Every item must eventually succeed via 'good' (never marked dead
        for 'bad' matters here, only that a failed item is retried and can
        land on a different, live endpoint)."""
        calls = {"bad": 0, "good": 0}
        lock = threading.Lock()

        def call_fn(ep: Endpoint, payload):
            with lock:
                calls[ep.name] += 1
            if ep.name == "bad":
                raise RuntimeError("simulated failure")
            return payload

        pool = EndpointPool(
            _endpoints(("http://bad/api/explain", 1, "bad"),
                      ("http://good/api/explain", 1, "good")),
            call_fn=call_fn,
            dead_after=1000,  # disable death for this test: pure retry test
        )
        results = list(pool.run(range(20)))
        assert len(results) == 20
        assert all(r.ok for r in results), [r.error for r in results if not r.ok]
        assert calls["good"] >= 20

    def test_exhausting_max_item_attempts_finalizes_as_failed(self):
        """Both endpoints always fail -> item exhausts max_item_attempts and
        is finalized ok=False (not skipped)."""
        def call_fn(ep, payload):
            raise RuntimeError("always fails")

        pool = EndpointPool(
            _endpoints(("http://a/api/explain", 1, "a"),
                      ("http://b/api/explain", 1, "b")),
            call_fn=call_fn,
            max_item_attempts=3,
            dead_after=1000,  # keep both endpoints alive throughout
            probe_every=1000,
            all_dead_grace=1000,
        )
        results = list(pool.run([42]))
        assert len(results) == 1
        r = results[0]
        assert r.ok is False
        assert r.skipped is False
        assert r.attempts == 3

    def test_item_avoids_endpoint_that_just_failed_it_when_other_alive(self):
        """A single item: 'flaky' fails it once, then the retry must land on
        'reliable' (not bounce back to 'flaky') while both are alive."""
        endpoint_seen_order = []
        lock = threading.Lock()

        def call_fn(ep: Endpoint, payload):
            with lock:
                endpoint_seen_order.append(ep.name)
            if ep.name == "flaky" and len(endpoint_seen_order) == 1:
                raise RuntimeError("first attempt fails")
            return payload

        pool = EndpointPool(
            _endpoints(("http://flaky/api/explain", 1, "flaky"),
                      ("http://reliable/api/explain", 1, "reliable")),
            call_fn=call_fn,
            dead_after=1000,
        )
        results = list(pool.run([1]))
        assert results[0].ok is True
        # The second attempt (if any) must not be on "flaky" again while
        # "reliable" was alive and free.
        assert "reliable" in endpoint_seen_order


# ---------------------------------------------------------------------------
# Endpoint death / revival.
# ---------------------------------------------------------------------------

class TestDeathAndRevival:

    def test_endpoint_marked_dead_after_consecutive_failures(self):
        def call_fn(ep, payload):
            if ep.name == "dying":
                raise RuntimeError("boom")
            return payload

        pool = EndpointPool(
            _endpoints(("http://dying/api/explain", 1, "dying"),
                      ("http://ok/api/explain", 1, "ok")),
            call_fn=call_fn,
            dead_after=3,
            max_item_attempts=1,  # each failure is final for its own item
            probe_every=1000,     # don't revive during this test
        )
        results = list(pool.run(range(30)))
        assert len(results) == 30
        stats = pool.stats()
        assert stats["dying"]["deaths"] >= 1
        # Once dead, "dying" should stop accumulating failures beyond a
        # handful right at the death boundary (a few in-flight calls may
        # have started concurrently, but it must not be anywhere near 30).
        assert stats["dying"]["fail"] < 30

    def test_dead_endpoint_is_revived_by_prober_and_gets_work_again(self):
        state = {"dead": True}
        probed = threading.Event()

        def call_fn(ep, payload):
            # A small delay keeps the whole run (40 items, mostly serialized
            # through "steady" at concurrency=1 once "flappy" dies) running
            # long enough for the monitor to actually get a chance to probe
            # and revive "flappy" before the queue drains.
            time.sleep(0.01)
            if ep.name == "flappy" and state["dead"]:
                raise RuntimeError("still down")
            return payload

        def probe_fn(url):
            probed.set()
            return True  # instantly healthy again once probed

        pool = EndpointPool(
            _endpoints(("http://flappy/api/explain", 1, "flappy"),
                      ("http://steady/api/explain", 1, "steady")),
            call_fn=call_fn,
            dead_after=2,
            max_item_attempts=5,
            probe_every=0.02,
            probe_fn=probe_fn,
        )

        # Flip "dead" to False shortly after the pool has had a chance to
        # mark "flappy" dead and probe it once.
        def flipper():
            probed.wait(timeout=2.0)
            state["dead"] = False

        threading.Thread(target=flipper, daemon=True).start()
        results = list(pool.run(range(40)))
        assert len(results) == 40
        assert all(r.ok for r in results)
        stats = pool.stats()
        assert stats["flappy"]["revivals"] >= 1


class TestAllDeadGrace:

    def test_all_dead_grace_expires_marks_remaining_items_failed_no_hang(self):
        """Both endpoints always fail and never revive -> after the (tiny)
        grace period every remaining item is finalized ok=False. Must
        complete promptly, never hang."""
        def call_fn(ep, payload):
            raise RuntimeError("down for good")

        def probe_fn(url):
            return False  # never revives

        pool = EndpointPool(
            _endpoints(("http://a/api/explain", 2, "a"),
                      ("http://b/api/explain", 2, "b")),
            call_fn=call_fn,
            dead_after=1,
            max_item_attempts=100,  # would retry "forever" without the grace cutoff
            probe_every=0.02,
            all_dead_grace=0.2,
            probe_fn=probe_fn,
        )
        t0 = time.monotonic()
        results = list(pool.run(range(10)))
        elapsed = time.monotonic() - t0

        assert len(results) == 10
        assert all(not r.ok and not r.skipped for r in results)
        # Bounded: grace (0.2s) plus normal scheduling slack, well under a
        # test timeout — this is the "never hang" guarantee.
        assert elapsed < 5.0


# ---------------------------------------------------------------------------
# Deadline (--max-seconds).
# ---------------------------------------------------------------------------

class TestDeadline:

    def test_deadline_stops_dispatch_and_marks_rest_skipped_not_failed(self):
        """A slow endpoint plus a short deadline: items still queued when the
        deadline hits must come back skipped=True, ok=False, and must NOT
        count as failures (n_fail must stay 0 for the caller)."""
        def call_fn(ep, payload):
            time.sleep(0.05)
            return payload

        pool = EndpointPool(
            _endpoints(("http://slow/api/explain", 1, "slow")),
            call_fn=call_fn,
            max_seconds=0.08,
        )
        results = list(pool.run(range(30)))
        assert len(results) == 30
        assert pool.deadline_hit is True
        failed = [r for r in results if not r.ok and not r.skipped]
        skipped = [r for r in results if r.skipped]
        ok = [r for r in results if r.ok]
        assert not failed, "deadline must never produce a real failure"
        assert skipped, "some items must be left un-dispatched by the deadline"
        assert ok, "some items must have completed before the deadline"
        assert len(ok) + len(skipped) == 30

    def test_no_deadline_means_deadline_hit_stays_false(self):
        def call_fn(ep, payload):
            return payload

        pool = EndpointPool(_endpoints(("http://a/api/explain", 2, "a")),
                            call_fn=call_fn)
        list(pool.run(range(5)))
        assert pool.deadline_hit is False


# ---------------------------------------------------------------------------
# TOCTOU race regression (2026-09-26 review): the "is dispatch closed?"
# check and the "requeue this failed item" append used to be two separate
# lock acquisitions. If a deadline/all-dead-grace drain ran to completion in
# between, the item was silently orphaned forever and run() hung on
# self._output.get(). These tests force that exact interleaving
# deterministically via EndpointPool._test_pre_requeue_hook — no sleeps used
# as synchronisation — and run the pool in a background thread with a hard
# join timeout, so a regression shows up as a test failure, not a wedged
# test run.
# ---------------------------------------------------------------------------

def _run_with_hard_timeout(pool, payloads, timeout=5.0):
    """Run pool.run(payloads) to completion in a background thread. Returns
    the results list, or raises AssertionError if it didn't finish in time
    (which is itself the regression this helper exists to catch)."""
    box: dict = {}

    def target():
        box["results"] = list(pool.run(payloads))

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    assert not t.is_alive(), (
        "EndpointPool.run() hung — TOCTOU race between dispatch-close and "
        "item-requeue reproduced (or the watchdog failed to unstick it)"
    )
    return box["results"]


class TestDispatchCloseRequeueRace:

    def test_deadline_close_racing_a_retry_does_not_orphan_the_item(self):
        """Force the exact interleaving from the review finding: the failed
        call's retry path is about to check dispatch-closed/requeue when a
        deadline-driven close-and-drain runs to completion first (here,
        triggered synchronously from the hook, i.e. guaranteed to complete
        *before* the atomic check below it runs — the worst case). The item
        must still be accounted for exactly once — skipped, not lost."""
        call_count = {"n": 0}

        def call_fn(ep, payload):
            call_count["n"] += 1
            raise RuntimeError("always fails")

        pool = EndpointPool(
            _endpoints(("http://a/api/explain", 1, "a")),
            call_fn=call_fn,
            max_item_attempts=2,   # first failure takes the retry path
            dead_after=1000,       # keep the endpoint alive; isolate this race
        )
        pool._test_pre_requeue_hook = pool._close_dispatch_and_drain_skipped

        results = _run_with_hard_timeout(pool, [1])

        assert len(results) == 1
        r = results[0]
        assert r.ok is False
        assert r.skipped is True, "deadline-flavoured close must skip, not fail"
        assert r.error == "deadline"
        assert pool.deadline_hit is False  # this pool never had max_seconds set

    def test_all_dead_grace_close_racing_a_retry_produces_a_real_failure(self):
        """Same interleaving, but via the all-dead-grace closer: unlike the
        deadline case, this must land as a real failure (ok=False,
        skipped=False), matching _force_fail_remaining()'s existing
        semantics for items still sitting in the queue."""
        def call_fn(ep, payload):
            raise RuntimeError("always fails")

        pool = EndpointPool(
            _endpoints(("http://a/api/explain", 1, "a")),
            call_fn=call_fn,
            max_item_attempts=2,
            dead_after=1000,
        )
        pool._test_pre_requeue_hook = pool._force_fail_remaining

        results = _run_with_hard_timeout(pool, [1])

        assert len(results) == 1
        r = results[0]
        assert r.ok is False
        assert r.skipped is False, "all-dead-grace closure must be a real failure"
        assert r.error == "all_endpoints_dead"

    def test_race_does_not_leak_in_flight_counter(self):
        """After the race resolves, _in_flight must be back to 0 — otherwise
        a later run() on the same pool could wrongly think something is
        still in flight forever (_is_quiescent() would never return True)."""
        def call_fn(ep, payload):
            raise RuntimeError("always fails")

        pool = EndpointPool(
            _endpoints(("http://a/api/explain", 1, "a")),
            call_fn=call_fn,
            max_item_attempts=2,
            dead_after=1000,
        )
        pool._test_pre_requeue_hook = pool._close_dispatch_and_drain_skipped
        _run_with_hard_timeout(pool, [1, 2, 3])
        assert pool._in_flight == 0

    def test_many_items_racing_close_all_accounted_for_exactly_once(self):
        """With several items in flight when the hook fires the close on the
        very first failure, every item must still resolve to exactly one
        result — none lost, none duplicated."""
        def call_fn(ep, payload):
            raise RuntimeError("always fails")

        pool = EndpointPool(
            _endpoints(("http://a/api/explain", 4, "a")),
            call_fn=call_fn,
            max_item_attempts=2,
            dead_after=1000,
        )
        fired = threading.Event()

        def hook_once():
            if not fired.is_set():
                fired.set()
                pool._close_dispatch_and_drain_skipped()

        pool._test_pre_requeue_hook = hook_once

        results = _run_with_hard_timeout(pool, list(range(20)), timeout=8.0)
        assert len(results) == 20
        assert {r.index for r in results} == set(range(20))
        assert all(not r.ok for r in results)


# ---------------------------------------------------------------------------
# run()'s own watchdog — defence in depth for a *different*, hypothetical
# future bug that truly drops an item (bypassing _process entirely), i.e.
# independent of the TOCTOU race fixed above.
# ---------------------------------------------------------------------------

class TestRunWatchdogDefenseInDepth:

    def test_watchdog_synthesizes_failure_for_a_genuinely_lost_item(self, monkeypatch):
        monkeypatch.setattr("exonit.explainer_pool._RUN_WATCHDOG_INTERVAL", 0.05)

        def call_fn(ep, payload):
            return payload

        pool = EndpointPool(_endpoints(("http://a/api/explain", 1, "a")),
                            call_fn=call_fn)

        # Simulate a hypothetical, *different* bug: the item is picked up
        # (goes in-flight) and then vanishes without ever being finalized or
        # requeued through the normal (now-fixed) path, while dispatch
        # closes around it — exactly the state run()'s quiescence watchdog
        # exists to detect and recover from.
        def vanishing_process(ep, item):
            pool._dec_in_flight()
            pool._close_dispatch_and_drain_skipped()

        pool._process = vanishing_process

        results = _run_with_hard_timeout(pool, [1], timeout=5.0)

        assert len(results) == 1
        assert results[0].ok is False
        assert results[0].skipped is False
        assert "lost" in (results[0].error or "").lower()
        assert results[0].payload == 1


# ---------------------------------------------------------------------------
# Drain reservation vs. watchdog false-fire (2026-09-26 review delta on
# 6bbdbed): _close_dispatch_and_drain_skipped()/_force_fail_remaining() used
# to empty _work_list under the lock and only *then* finalize each drained
# item outside the lock, without reserving them against _in_flight. A
# watchdog poll landing in that window would see "_work_list empty AND
# _in_flight == 0 AND dispatch closed" and wrongly conclude the pool was
# quiescent with results missing, synthesizing "internal: item lost"
# failures for items that were merely still being finalized (wrong flavour:
# should have been skipped=True/error="deadline", or a real
# "all_endpoints_dead" failure, depending on which drain fired). Fixed by
# reserving len(drained) against _in_flight atomically with the same lock
# acquisition that empties the list, releasing one by one as each item is
# actually finalized. These tests use the same deterministic-hook style as
# TestDispatchCloseRequeueRace (no sleeps as synchronisation).
# ---------------------------------------------------------------------------

class TestDrainReservationAgainstWatchdogFalseFire:

    def test_in_flight_reserved_atomically_with_the_drain(self):
        """At the exact moment _test_mid_drain_hook fires (_work_list just
        emptied, no item finalized yet), _in_flight must already reflect the
        drained items — so _is_quiescent() must be False right there. The
        pre-fix code would have _in_flight == 0 at this point and would
        (wrongly) report quiescent."""
        pool = EndpointPool(_endpoints(("http://a/api/explain", 1, "a")),
                            call_fn=lambda ep, p: p)
        with pool._work_cv:
            pool._work_list.extend(
                explainer_pool_module._WorkItem(payload=i, index=i)
                for i in range(5)
            )

        seen = {}

        def mid_drain_hook():
            with pool._work_cv:
                seen["work_list_empty"] = not pool._work_list
                seen["in_flight_at_hook"] = pool._in_flight
            seen["quiescent_at_hook"] = pool._is_quiescent()

        pool._test_mid_drain_hook = mid_drain_hook
        pool._close_dispatch_and_drain_skipped()

        assert seen["work_list_empty"] is True
        assert seen["in_flight_at_hook"] == 5
        assert seen["quiescent_at_hook"] is False, (
            "watchdog would have falsely seen a quiescent pool mid-drain"
        )
        assert pool._in_flight == 0, "reservation must be fully released after the drain"

    def test_in_flight_reserved_atomically_with_all_dead_grace_drain(self):
        """Same invariant for the other drain (_force_fail_remaining)."""
        pool = EndpointPool(_endpoints(("http://a/api/explain", 1, "a")),
                            call_fn=lambda ep, p: p)
        with pool._work_cv:
            pool._work_list.extend(
                explainer_pool_module._WorkItem(payload=i, index=i)
                for i in range(3)
            )

        seen = {}

        def mid_drain_hook():
            seen["quiescent_at_hook"] = pool._is_quiescent()

        pool._test_mid_drain_hook = mid_drain_hook
        pool._force_fail_remaining()

        assert seen["quiescent_at_hook"] is False
        assert pool._in_flight == 0

    def test_cross_thread_quiescence_poll_during_a_paused_drain_sees_in_flight(self):
        """Same invariant as the two tests above, but with a genuinely
        different OS thread running the drain (not a synchronous same-thread
        call) while the main thread polls `_is_quiescent()` — proving the
        reservation is visible across threads, not just within one call
        stack. No call_fn/run() involved (so there is no confound from an
        unrelated, separately-tracked in-flight network call): the only
        in-flight accounting in play is the drain's own reservation, which
        is exactly what the review finding was about."""
        pool = EndpointPool(_endpoints(("http://a/api/explain", 1, "a")),
                            call_fn=lambda ep, p: p)
        with pool._work_cv:
            pool._work_list.extend(
                explainer_pool_module._WorkItem(payload=i, index=i)
                for i in range(5)
            )

        hook_entered = threading.Event()
        release_hook = threading.Event()

        def mid_drain_hook():
            hook_entered.set()
            release_hook.wait(timeout=3.0)

        pool._test_mid_drain_hook = mid_drain_hook

        drain_thread = threading.Thread(
            target=pool._close_dispatch_and_drain_skipped, daemon=True)
        drain_thread.start()

        assert hook_entered.wait(timeout=3.0), "drain never reached the hook"
        # The drain (running on a *different* thread) is now paused right
        # after reserving _in_flight and before finalizing anything. A
        # watchdog poll landing right now, from this thread, must not see
        # quiescence — the pre-fix code would have _in_flight == 0 here.
        assert pool._is_quiescent() is False

        release_hook.set()
        drain_thread.join(timeout=5.0)
        assert not drain_thread.is_alive()
        assert pool._in_flight == 0

        results = [pool._output.get_nowait() for _ in range(5)]
        assert all(r.skipped and not r.ok and r.error == "deadline" for r in results)


# ---------------------------------------------------------------------------
# Construction guards.
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_requires_at_least_one_endpoint(self):
        with pytest.raises(ValueError):
            EndpointPool([], call_fn=lambda ep, p: p)

    def test_stats_keys_present_for_every_endpoint(self):
        pool = EndpointPool(
            _endpoints(("http://a/api/explain", 1, "a"),
                      ("http://b/api/explain", 1, "b")),
            call_fn=lambda ep, p: p,
        )
        list(pool.run(range(3)))
        stats = pool.stats()
        assert set(stats) == {"a", "b"}
        for name in ("a", "b"):
            assert set(stats[name]) == {"ok", "fail", "retries", "deaths", "revivals"}
