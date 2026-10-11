"""Tests for exonit/dist_semaphore.py.

Coverage (required per A2b spec):
1. Limit respected: at most N concurrent holders; (N+1)th blocks until release.
2. Context manager: ``with sem:`` acquires and releases; re-acquisition works.
3. CROSS-PROCESS (anti-OOM key test): two DistributedSemaphore instances over
   the SAME Redis DB with limit=1 → mutual exclusion across instances.
4. Crash recovery: a slot held without renewal expires after lease_ttl.
5. Renewal keeps slot alive past lease_ttl.
6. Release is a no-op when the slot was re-acquired by another holder (token
   check prevents evicting a foreign slot).

Tests run against both MemoryBackend and RedisBackend (db 15).
Redis tests are skipped when Redis is unreachable.

Run:
    /usr/bin/python3 -m pytest tests/test_dist_semaphore.py -q
"""

from __future__ import annotations

import threading
import time

import pytest

from exonit.dist_semaphore import DistributedSemaphore
from exonit.shared_state import MemoryBackend, RedisBackend

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_REDIS_TEST_URL = "redis://localhost:6379/15"


def _redis_available() -> bool:
    try:
        b = RedisBackend(_REDIS_TEST_URL)
        return b.ping()
    except Exception:
        return False


@pytest.fixture
def mem() -> MemoryBackend:
    """Fresh MemoryBackend per test."""
    return MemoryBackend()


@pytest.fixture
def redis_db():
    """RedisBackend on DB 15, flushed before and after each test."""
    if not _redis_available():
        pytest.skip("Redis not reachable at " + _REDIS_TEST_URL)
    b = RedisBackend(_REDIS_TEST_URL)
    b._client.flushdb()
    yield b
    b._client.flushdb()


def make_sem(
    backend,
    name: str = "test_sem",
    limit: int = 1,
    *,
    lease_ttl: float = 0.3,
    poll_interval: float = 0.05,
    renew: bool = True,
) -> DistributedSemaphore:
    """Convenience factory with short TTLs for fast, deterministic tests."""
    return DistributedSemaphore(
        backend,
        name,
        limit,
        lease_ttl=lease_ttl,
        poll_interval=poll_interval,
        renew=renew,
    )


# ---------------------------------------------------------------------------
# Test 1 — limit respected (threading concurrency)
# ---------------------------------------------------------------------------


class TestLimitRespected:
    """With limit=N, at most N threads hold the semaphore simultaneously."""

    def _run_limit_test(self, backend, limit: int = 2) -> None:
        sem = make_sem(backend, name="limit_test", limit=limit, lease_ttl=5.0)

        active = {"count": 0}
        max_observed = {"count": 0}
        violation = {"seen": False}
        lock = threading.Lock()

        inside_event = threading.Event()
        can_proceed = threading.Event()
        WORKERS = limit + 1  # one extra that must block

        def worker(idx):
            sem.acquire(blocking=True)
            try:
                with lock:
                    active["count"] += 1
                    if active["count"] > limit:
                        violation["seen"] = True
                    max_observed["count"] = max(max_observed["count"], active["count"])
                inside_event.set()
                can_proceed.wait(timeout=3.0)
            finally:
                with lock:
                    active["count"] -= 1
                sem.release()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(WORKERS)]
        for t in threads:
            t.start()

        # Wait until at least one thread is inside.
        inside_event.wait(timeout=3.0)
        time.sleep(0.1)

        # At most `limit` threads should be inside.
        with lock:
            current = active["count"]
        assert current <= limit, f"observed {current} concurrent holders > limit {limit}"
        assert not violation["seen"], "limit was exceeded at some point"

        can_proceed.set()
        for t in threads:
            t.join(timeout=5.0)

        assert max_observed["count"] <= limit

    def test_limit_1_memory(self, mem):
        self._run_limit_test(mem, limit=1)

    def test_limit_3_memory(self, mem):
        self._run_limit_test(mem, limit=3)

    def test_limit_1_redis(self, redis_db):
        self._run_limit_test(redis_db, limit=1)

    def test_limit_3_redis(self, redis_db):
        self._run_limit_test(redis_db, limit=3)

    def test_nth_plus_1_blocks_until_release_memory(self, mem):
        """(limit+1)th acquire must block until a holder releases."""
        sem = make_sem(mem, limit=1, lease_ttl=5.0)
        released = threading.Event()
        acquired_second = threading.Event()

        def holder():
            sem.acquire(blocking=True)
            released.wait(timeout=3.0)
            sem.release()

        def waiter():
            sem.acquire(blocking=True)  # must block
            acquired_second.set()
            sem.release()

        t_holder = threading.Thread(target=holder)
        t_waiter = threading.Thread(target=waiter)
        t_holder.start()
        time.sleep(0.05)
        t_waiter.start()
        time.sleep(0.1)

        # Waiter should not have acquired yet.
        assert not acquired_second.is_set(), "waiter acquired before holder released"

        released.set()
        acquired_second.wait(timeout=3.0)
        assert acquired_second.is_set(), "waiter never acquired after holder released"

        t_holder.join(timeout=3.0)
        t_waiter.join(timeout=3.0)

    def test_nth_plus_1_blocks_until_release_redis(self, redis_db):
        """Same as above for Redis backend."""
        sem = make_sem(redis_db, limit=1, lease_ttl=5.0)
        released = threading.Event()
        acquired_second = threading.Event()

        def holder():
            sem.acquire(blocking=True)
            released.wait(timeout=3.0)
            sem.release()

        def waiter():
            sem.acquire(blocking=True)
            acquired_second.set()
            sem.release()

        t_holder = threading.Thread(target=holder)
        t_waiter = threading.Thread(target=waiter)
        t_holder.start()
        time.sleep(0.05)
        t_waiter.start()
        time.sleep(0.1)

        assert not acquired_second.is_set()

        released.set()
        acquired_second.wait(timeout=3.0)
        assert acquired_second.is_set()

        t_holder.join(timeout=3.0)
        t_waiter.join(timeout=3.0)


# ---------------------------------------------------------------------------
# Test 2 — context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    """``with sem:`` acquires and releases; slot is re-acquirable afterwards."""

    def _run_context_manager_test(self, backend) -> None:
        sem = make_sem(backend, limit=1, lease_ttl=5.0)

        # First use via context manager.
        with sem:
            # Inside: slot is held; non-blocking acquire on a second sem (same
            # backend, same name) must fail.
            sem2 = make_sem(backend, name=sem._name, limit=1, lease_ttl=5.0)
            assert sem2.acquire(blocking=False) is False, \
                "second sem acquired while first is inside with-block"

        # After exiting the context, the slot is free.
        # Re-acquiring on the original instance must succeed.
        assert sem.acquire(blocking=False) is True, \
            "could not re-acquire after context manager exit"
        sem.release()

    def test_context_manager_memory(self, mem):
        self._run_context_manager_test(mem)

    def test_context_manager_redis(self, redis_db):
        self._run_context_manager_test(redis_db)


# ---------------------------------------------------------------------------
# Test 3 — CROSS-PROCESS: two separate instances, same backend, mutual exclusion
# This is the key anti-OOM guarantee for A3 multi-worker uvicorn.
# ---------------------------------------------------------------------------


class TestCrossProcessExclusion:
    """Two DistributedSemaphore instances on the same backend with limit=1 are
    mutually exclusive — simulates two uvicorn worker processes.

    This test is the primary safety guarantee: N processes × limit=1 must NOT
    result in N concurrent Docker containers.
    """

    def _run_cross_instance_test(self, backend) -> None:
        sem_a = make_sem(backend, name="docker_sem", limit=1, lease_ttl=5.0)
        sem_b = make_sem(backend, name="docker_sem", limit=1, lease_ttl=5.0)

        # A acquires the slot.
        assert sem_a.acquire(blocking=False) is True, "A could not acquire slot"

        # B must fail immediately (non-blocking).
        assert sem_b.acquire(blocking=False) is False, \
            "B acquired while A holds the slot — cross-instance exclusion broken (OOM risk!)"

        # Release A; now B should succeed.
        sem_a.release()
        assert sem_b.acquire(blocking=False) is True, \
            "B could not acquire after A released"
        sem_b.release()

    def test_cross_instance_memory(self, mem):
        self._run_cross_instance_test(mem)

    def test_cross_instance_redis(self, redis_db):
        self._run_cross_instance_test(redis_db)

    def test_cross_instance_blocking_memory(self, mem):
        """A holds limit=1 slot; B.acquire(blocking=True) must wait until A releases."""
        sem_a = make_sem(mem, name="blocking_cross", limit=1, lease_ttl=5.0)
        sem_b = make_sem(mem, name="blocking_cross", limit=1, lease_ttl=5.0)

        b_acquired = threading.Event()
        release_a = threading.Event()

        def thread_a():
            sem_a.acquire(blocking=True)
            release_a.wait(timeout=3.0)
            sem_a.release()

        def thread_b():
            sem_b.acquire(blocking=True)  # must block until A releases
            b_acquired.set()
            sem_b.release()

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        time.sleep(0.05)
        tb.start()
        time.sleep(0.1)

        assert not b_acquired.is_set(), "B acquired before A released"

        release_a.set()
        b_acquired.wait(timeout=3.0)
        assert b_acquired.is_set(), "B never acquired after A released"

        ta.join(timeout=3.0)
        tb.join(timeout=3.0)

    def test_cross_instance_blocking_redis(self, redis_db):
        """Same cross-instance blocking test for Redis backend."""
        sem_a = make_sem(redis_db, name="blocking_cross_r", limit=1, lease_ttl=5.0)
        sem_b = make_sem(redis_db, name="blocking_cross_r", limit=1, lease_ttl=5.0)

        b_acquired = threading.Event()
        release_a = threading.Event()

        def thread_a():
            sem_a.acquire(blocking=True)
            release_a.wait(timeout=3.0)
            sem_a.release()

        def thread_b():
            sem_b.acquire(blocking=True)
            b_acquired.set()
            sem_b.release()

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        time.sleep(0.05)
        tb.start()
        time.sleep(0.1)

        assert not b_acquired.is_set()

        release_a.set()
        b_acquired.wait(timeout=3.0)
        assert b_acquired.is_set()

        ta.join(timeout=3.0)
        tb.join(timeout=3.0)


# ---------------------------------------------------------------------------
# Test 4 — crash recovery: slot expires when renewer is stopped
# ---------------------------------------------------------------------------


class TestCrashRecovery:
    """Simulates a dead process: ``renew=False`` → slot expires after lease_ttl."""

    def _run_recovery_test(self, backend) -> None:
        # lease_ttl=0.2s, no renewer (simulates crashed/killed process).
        sem = make_sem(
            backend,
            name="crash_recovery",
            limit=1,
            lease_ttl=0.2,
            renew=False,
        )
        assert sem.acquire(blocking=False) is True

        # Do NOT release — simulate crash.  The slot should expire.
        # Give it 2× lease_ttl to account for timing jitter.
        time.sleep(0.5)

        # A fresh semaphore instance must now be able to acquire the slot.
        sem2 = make_sem(
            backend,
            name="crash_recovery",
            limit=1,
            lease_ttl=0.2,
            renew=False,
        )
        result = sem2.acquire(blocking=False)
        if result:
            sem2.release()
        assert result, "slot did not expire after crashed holder (recovery broken)"

    def test_crash_recovery_memory(self, mem):
        self._run_recovery_test(mem)

    def test_crash_recovery_redis(self, redis_db):
        self._run_recovery_test(redis_db)


# ---------------------------------------------------------------------------
# Test 5 — renewal keeps slot alive past lease_ttl
# ---------------------------------------------------------------------------


class TestRenewalKeepsAlive:
    """With ``renew=True`` (default), the holder retains the slot even after
    the initial lease_ttl passes."""

    def _run_renewal_test(self, backend) -> None:
        lease_ttl = 0.2
        sem = make_sem(
            backend,
            name="renewal_test",
            limit=1,
            lease_ttl=lease_ttl,
            poll_interval=0.02,
            renew=True,
        )
        assert sem.acquire(blocking=False) is True

        # Wait well past the initial lease_ttl (3× to be safe).
        time.sleep(lease_ttl * 3.5)

        # The slot should still be ours: a competitor must fail.
        sem2 = make_sem(
            backend,
            name="renewal_test",
            limit=1,
            lease_ttl=lease_ttl,
            renew=False,
        )
        still_blocked = not sem2.acquire(blocking=False)
        if not still_blocked:
            sem2.release()

        sem.release()

        assert still_blocked, (
            "slot was not renewed: a competitor acquired it while the "
            "original holder's renewer should have been running"
        )

    def test_renewal_memory(self, mem):
        self._run_renewal_test(mem)

    def test_renewal_redis(self, redis_db):
        self._run_renewal_test(redis_db)


# ---------------------------------------------------------------------------
# Test 6 — release only deletes the caller's own slot (token check)
# ---------------------------------------------------------------------------


class TestTokenCheckOnRelease:
    """A late release must not evict a slot re-acquired by another holder.

    Scenario:
    1. A acquires slot with renew=False and a very short TTL.
    2. Wait for A's slot to expire (A is a "zombie" — still alive but missed renewal).
    3. B acquires the same slot.
    4. A.release() is called late — must be a no-op (token mismatch).
    5. B still holds the slot (B's token is intact).
    """

    def _run_token_check_test(self, backend) -> None:
        lease_ttl = 0.2
        sem_a = make_sem(
            backend,
            name="token_check",
            limit=1,
            lease_ttl=lease_ttl,
            renew=False,  # no renewal → lease expires
        )
        sem_b = make_sem(
            backend,
            name="token_check",
            limit=1,
            lease_ttl=5.0,  # long lease so B doesn't expire during the test
            renew=False,
        )

        # Step 1: A acquires.
        assert sem_a.acquire(blocking=False) is True

        # Step 2: Wait for A's lease to expire.
        time.sleep(lease_ttl * 3)

        # Step 3: B acquires (A's slot has expired).
        assert sem_b.acquire(blocking=False) is True, \
            "B could not acquire after A's lease expired"

        # Step 4: A calls release() late.
        sem_a.release()  # must be a no-op (token mismatch)

        # Step 5: B's slot must still be intact (not deleted by A's release).
        # Verify by trying a third instance — must fail.
        sem_c = make_sem(
            backend,
            name="token_check",
            limit=1,
            lease_ttl=0.2,
            renew=False,
        )
        c_result = sem_c.acquire(blocking=False)
        if c_result:
            sem_c.release()
            sem_b.release()
            assert not c_result, (
                "C acquired: A's late release evicted B's slot (token check broken)"
            )
        else:
            sem_b.release()

    def test_token_check_memory(self, mem):
        self._run_token_check_test(mem)

    def test_token_check_redis(self, redis_db):
        self._run_token_check_test(redis_db)


# ---------------------------------------------------------------------------
# Test — non-blocking acquire returns False immediately when no slot free
# ---------------------------------------------------------------------------


class TestNonBlocking:
    def _run_non_blocking_test(self, backend) -> None:
        sem = make_sem(backend, limit=1, lease_ttl=5.0)
        assert sem.acquire(blocking=False) is True
        assert sem.acquire(blocking=False) is False
        sem.release()
        assert sem.acquire(blocking=False) is True
        sem.release()

    def test_non_blocking_memory(self, mem):
        self._run_non_blocking_test(mem)

    def test_non_blocking_redis(self, redis_db):
        self._run_non_blocking_test(redis_db)


# ---------------------------------------------------------------------------
# Test — timeout acquire returns False after timeout expires
# ---------------------------------------------------------------------------


class TestTimeoutAcquire:
    def _run_timeout_test(self, backend) -> None:
        sem = make_sem(backend, limit=1, lease_ttl=5.0, poll_interval=0.05)
        assert sem.acquire(blocking=True) is True

        start = time.monotonic()
        result = sem.acquire(blocking=True, timeout=0.2)
        elapsed = time.monotonic() - start

        sem.release()

        assert result is False, "timed-out acquire should return False"
        # Should have respected the timeout (within a generous margin).
        assert elapsed < 1.0, f"timeout took too long: {elapsed:.2f}s"

    def test_timeout_memory(self, mem):
        self._run_timeout_test(mem)

    def test_timeout_redis(self, redis_db):
        self._run_timeout_test(redis_db)


# ---------------------------------------------------------------------------
# Test — release without prior acquire is a no-op (idempotent)
# ---------------------------------------------------------------------------


class TestIdempotentRelease:
    def _run_idempotent_test(self, backend) -> None:
        sem = make_sem(backend, limit=1, lease_ttl=5.0)
        # Release without acquire — must not raise.
        sem.release()
        # Double release — must not raise.
        sem.acquire(blocking=False)
        sem.release()
        sem.release()

    def test_idempotent_memory(self, mem):
        self._run_idempotent_test(mem)

    def test_idempotent_redis(self, redis_db):
        self._run_idempotent_test(redis_db)


# ---------------------------------------------------------------------------
# Test — concurrent threads share ONE instance (thread-local state bug)
#
# This test specifically reproduces the concurrency bug where N threads that
# share a SINGLE DistributedSemaphore instance (the server singleton pattern)
# would clobber each other's _held_slot/_held_token/renewer state when the
# state was stored as instance attributes instead of threading.local().
#
# With the buggy code:
#   - Thread B's acquire() overwrites self._held_slot/token set by thread A.
#   - Thread A's release() then reads B's slot/token → stops B's renewer,
#     deletes B's slot, and leaves A's slot + renewer dangling (leaked).
#   - Result: barrier deadlock (N threads can't all enter) or slot leak.
#
# With the fix (threading.local()):
#   - Each thread manages its own slot/token/renewer independently.
#   - All N threads enter simultaneously; barrier passes without deadlock.
#   - (N+1)th acquire fails while the barrier holds all N slots.
#   - After all N exit, every slot key is deleted → no leaked slots.
# ---------------------------------------------------------------------------


class TestConcurrentThreadsSingleInstance:
    """N threads share ONE DistributedSemaphore instance (server singleton
    pattern).  Each thread must track its own acquisition state independently.
    """

    N = 4  # matches spoc_check default limit

    def _run_concurrent_test(self, backend) -> None:
        sem = DistributedSemaphore(
            backend,
            "concurrent",
            limit=self.N,
            lease_ttl=5.0,
            poll_interval=0.05,
            renew=True,
        )

        # Barrier: all N threads must arrive before any proceeds past it.
        barrier = threading.Barrier(self.N, timeout=5.0)

        errors: list[str] = []
        threads_entered: list[bool] = []
        threads_entered_lock = threading.Lock()

        def worker() -> None:
            try:
                with sem:
                    # All N threads must be inside simultaneously.
                    barrier.wait()
                    with threads_entered_lock:
                        threads_entered.append(True)
            except threading.BrokenBarrierError:
                with threads_entered_lock:
                    errors.append("barrier broken — not all N threads acquired simultaneously")
            except Exception as exc:
                with threads_entered_lock:
                    errors.append(f"unexpected exception: {exc}")

        # --- Part 1: all N threads acquire simultaneously ---
        workers = [threading.Thread(target=worker, daemon=True) for _ in range(self.N)]
        for t in workers:
            t.start()
        for t in workers:
            t.join(timeout=8.0)
            if t.is_alive():
                errors.append("worker thread did not finish — possible deadlock")

        assert not errors, f"concurrent acquisition errors: {errors}"
        assert len(threads_entered) == self.N, (
            f"only {len(threads_entered)}/{self.N} threads entered "
            "— limit was incorrectly blocking concurrent holders on same instance"
        )

        # --- Part 2: (N+1)th acquire must fail while N slots held ---
        # Re-run with an extra thread gated by a barrier so we can test
        # blocking=False from the main thread while the N are inside.
        holding = threading.Barrier(self.N + 1, timeout=5.0)  # N workers + main
        release_gate = threading.Event()

        def gated_worker() -> None:
            with sem:
                holding.wait()       # signal "I'm inside"
                release_gate.wait(timeout=5.0)  # wait for main's OK to exit

        gated_workers = [
            threading.Thread(target=gated_worker, daemon=True) for _ in range(self.N)
        ]
        for t in gated_workers:
            t.start()

        # Wait until all N are inside the with-block.
        holding.wait()

        # Now the semaphore is saturated: blocking=False must return False.
        extra_result = sem.acquire(blocking=False)
        if extra_result:
            sem.release()  # clean up if it somehow succeeded

        release_gate.set()
        for t in gated_workers:
            t.join(timeout=5.0)

        assert extra_result is False, (
            "(N+1)th acquire succeeded while N slots are held "
            "— slot limit not respected"
        )

        # --- Part 3: after all N exit, every slot key is deleted ---
        for i in range(self.N):
            key = f"sem:concurrent:slot:{i}"
            val = backend.get(key)
            assert val is None, (
                f"slot key {key!r} still set after all threads exited "
                "— slot leaked (threading.local() fix broken or renewer still running)"
            )

        # Part 3b: a new acquire must succeed (slots are truly free).
        reacquire = sem.acquire(blocking=False)
        if reacquire:
            sem.release()
        assert reacquire is True, (
            "could not re-acquire after all N threads released — slot was not cleaned up"
        )

    def test_concurrent_threads_single_instance_memory(self, mem):
        self._run_concurrent_test(mem)

    def test_concurrent_threads_single_instance_redis(self, redis_db):
        self._run_concurrent_test(redis_db)
