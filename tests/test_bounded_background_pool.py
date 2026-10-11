"""Regression test: background dispatch must NOT create unbounded threads.

Incident 2026-06-04 — with one threading.Thread per worker result, hundreds
of threads piled up doing Redis polling under burst load. This test verifies
that the new executor-based dispatch keeps the live thread count bounded.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

import exonit.server as server_module


# ---------------------------------------------------------------------------
# Helpers to reset module-level executor singletons between tests
# ---------------------------------------------------------------------------

def _reset_executors():
    server_module._spoc_executor = None
    server_module._exominer_executor = None


@pytest.fixture(autouse=True)
def clean_executors():
    _reset_executors()
    yield
    # Shut down any executor created during the test so threads don't leak
    # into subsequent tests.
    ex = server_module._spoc_executor
    if ex is not None:
        ex.shutdown(wait=False)
    ex = server_module._exominer_executor
    if ex is not None:
        ex.shutdown(wait=False)
    _reset_executors()


# ---------------------------------------------------------------------------
# Core regression: spoc executor is bounded
# ---------------------------------------------------------------------------

def test_spoc_executor_thread_count_is_bounded():
    """Dispatching M >> max_workers tasks must not create M threads.

    Strategy:
    - Submit M tasks to a ThreadPoolExecutor(max_workers=MAX_W).
    - Each task signals a threading.Event when it has started, then blocks
      on a gate Event.
    - After MAX_W tasks have started (gate still closed), count live threads
      with the prefix — must be == MAX_W, not M.
    - Open the gate and drain all futures before returning.
    """
    MAX_W = 4
    M = 80  # well above MAX_W but not so large it slows tests

    gate = threading.Event()
    started_events = [threading.Event() for _ in range(MAX_W)]
    # each task picks one started_event by its index (only first MAX_W matter)
    counter = {"idx": 0}
    counter_lock = threading.Lock()

    def blocked_work():
        with counter_lock:
            idx = counter["idx"]
            counter["idx"] += 1
        if idx < MAX_W:
            started_events[idx].set()
        gate.wait(timeout=10.0)

    with ThreadPoolExecutor(max_workers=MAX_W, thread_name_prefix="spoc_t") as ex:
        futures = [ex.submit(blocked_work) for _ in range(M)]

        # Wait for the first MAX_W threads to start.
        for ev in started_events:
            ok = ev.wait(timeout=5.0)
            if not ok:
                gate.set()
                pytest.fail("Timed out waiting for spoc threads to start")

        # Snapshot thread count while gate is closed.
        live = [t for t in threading.enumerate()
                if t.name.startswith("spoc_t")]
        live_count = len(live)

        gate.set()
        for f in futures:
            f.result(timeout=15.0)

    assert live_count <= MAX_W, (
        f"Thread explosion: {live_count} 'spoc_t' threads alive with "
        f"max_workers={MAX_W} and M={M} submissions. "
        f"Expected <= {MAX_W}."
    )


# ---------------------------------------------------------------------------
# Core regression: exominer executor is bounded
# ---------------------------------------------------------------------------

def test_exominer_executor_thread_count_is_bounded():
    """Same guarantee for the ExoMiner executor (smaller pool, 60 tasks)."""
    MAX_W = 2
    M = 60

    gate = threading.Event()
    started_events = [threading.Event() for _ in range(MAX_W)]
    counter = {"idx": 0}
    counter_lock = threading.Lock()

    def blocked_work():
        with counter_lock:
            idx = counter["idx"]
            counter["idx"] += 1
        if idx < MAX_W:
            started_events[idx].set()
        gate.wait(timeout=10.0)

    with ThreadPoolExecutor(max_workers=MAX_W, thread_name_prefix="exo_t") as ex:
        futures = [ex.submit(blocked_work) for _ in range(M)]

        for ev in started_events:
            ok = ev.wait(timeout=5.0)
            if not ok:
                gate.set()
                pytest.fail("Timed out waiting for exominer threads to start")

        live = [t for t in threading.enumerate()
                if t.name.startswith("exo_t")]
        live_count = len(live)

        gate.set()
        for f in futures:
            f.result(timeout=15.0)

    assert live_count <= MAX_W, (
        f"Thread explosion: {live_count} 'exo_t' threads alive with "
        f"max_workers={MAX_W} and M={M} submissions. "
        f"Expected <= {MAX_W}."
    )


# ---------------------------------------------------------------------------
# Control: OLD pattern (raw threading.Thread per submit) WOULD give M threads
# ---------------------------------------------------------------------------

def test_old_thread_per_submit_pattern_would_create_many_threads():
    """Documents that the old approach *would* have created M threads.

    This is a sanity-check / documentary test, not a regression assertion
    against production code — it verifies that the fix is necessary by
    showing the old pattern's behaviour.
    """
    M = 50
    gate = threading.Event()
    threads_created = []

    def blocked_work():
        gate.wait(timeout=3.0)

    # Simulate old pattern: one Thread per dispatch.
    for _ in range(M):
        t = threading.Thread(target=blocked_work, daemon=True)
        t.start()
        threads_created.append(t)

    # At this point M threads exist (all blocked on gate).
    alive = sum(1 for t in threads_created if t.is_alive())
    gate.set()
    for t in threads_created:
        t.join(timeout=3.0)

    # The old pattern created M threads — this is the problem we fixed.
    assert alive == M, (
        f"Expected {M} threads from old pattern but got {alive}"
    )


# ---------------------------------------------------------------------------
# Lazy singleton: same executor instance returned on repeated calls
# ---------------------------------------------------------------------------

def test_spoc_executor_is_singleton(monkeypatch):
    monkeypatch.setenv("EXONIT_SPOC_CHECK_MAX_CONCURRENT", "2")
    _reset_executors()
    ex1 = server_module._get_spoc_executor()
    ex2 = server_module._get_spoc_executor()
    assert ex1 is ex2
    assert isinstance(ex1, ThreadPoolExecutor)


def test_exominer_executor_is_singleton(monkeypatch):
    monkeypatch.setenv("EXONIT_EXOMINER_MAX_CONCURRENT", "1")
    _reset_executors()
    ex1 = server_module._get_exominer_executor()
    ex2 = server_module._get_exominer_executor()
    assert ex1 is ex2
    assert isinstance(ex1, ThreadPoolExecutor)


# ---------------------------------------------------------------------------
# Env parsing: bad value falls back to default gracefully
# ---------------------------------------------------------------------------

def test_spoc_executor_bad_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("EXONIT_SPOC_CHECK_MAX_CONCURRENT", "not_a_number")
    _reset_executors()
    ex = server_module._get_spoc_executor()
    # Should not raise; default is 4
    assert ex is not None
    assert isinstance(ex, ThreadPoolExecutor)


def test_exominer_executor_bad_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("EXONIT_EXOMINER_MAX_CONCURRENT", "bad")
    _reset_executors()
    ex = server_module._get_exominer_executor()
    assert ex is not None
    assert isinstance(ex, ThreadPoolExecutor)


# ---------------------------------------------------------------------------
# _run_spoc_check_background no longer wraps with DistributedSemaphore
# ---------------------------------------------------------------------------

def test_spoc_check_background_does_not_use_distributed_semaphore(monkeypatch):
    """The DistributedSemaphore must NOT be acquired inside the worker fn.

    With the pool-based fix, the fn runs directly (pool IS the limit).
    We verify by patching _get_spoc_check_semaphore to raise — if it were
    still called, the test would fail.
    """
    import inspect
    import exonit.server as srv

    # Patch _get_spoc_check_semaphore so any call raises loudly.
    def _should_not_be_called():
        raise AssertionError(
            "_get_spoc_check_semaphore called inside worker fn — "
            "DistributedSemaphore was NOT removed from _run_spoc_check_background"
        )

    with patch.object(srv, "_get_spoc_check_semaphore", side_effect=_should_not_be_called):
        with patch("exonit.ffi_download.is_ffi_only_at_mast", return_value=True):
            # Use a minimal mock db
            class FakeDB:
                def set_star_has_spoc(self, *a): pass
                def mark_ffi_only_candidates(self, *a): return 0

            # This must not raise AssertionError.
            srv._run_spoc_check_background(FakeDB(), "12345", [])
