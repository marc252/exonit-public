"""Tests for exonit/worker_recycle.py — post-run staggered worker recycling.

See docs/superpowers/specs/2026-07-17-worker-recycle-postrun.md.

conftest.py's autouse ``_isolate_shared_state`` fixture forces MemoryBackend
and deletes EXONIT_SERVER_WORKERS before every test in this file, so
``get_shared_state()`` here is always a fresh, isolated MemoryBackend —
exactly the "InMemory parity" surface the spec's validation section asks
for (no separate Redis-backend test needed: the coordination primitive used
throughout, ``set_if_absent(..., ttl=...)``, already has its own Redis/
MemoryBackend parity tests in tests/test_shared_state.py; this file only
exercises the recycling logic built on top of it).
"""

from __future__ import annotations

import asyncio

import pytest

from exonit import worker_recycle as wr
from exonit.shared_state import get_shared_state


# ---------------------------------------------------------------------------
# server_workers_enabled
# ---------------------------------------------------------------------------

class TestServerWorkersEnabled:
    def test_absent_is_disabled(self):
        assert wr.server_workers_enabled({}) is False

    def test_one_is_disabled(self):
        assert wr.server_workers_enabled({"EXONIT_SERVER_WORKERS": "1"}) is False

    def test_two_is_enabled(self):
        assert wr.server_workers_enabled({"EXONIT_SERVER_WORKERS": "2"}) is True

    def test_four_is_enabled(self):
        assert wr.server_workers_enabled({"EXONIT_SERVER_WORKERS": "4"}) is True

    def test_zero_is_disabled(self):
        assert wr.server_workers_enabled({"EXONIT_SERVER_WORKERS": "0"}) is False

    def test_malformed_value_is_disabled(self):
        """A parse error must fail toward NOT arming the self-kill loop."""
        assert wr.server_workers_enabled({"EXONIT_SERVER_WORKERS": "banana"}) is False


# ---------------------------------------------------------------------------
# download_active
# ---------------------------------------------------------------------------

class TestDownloadActive:
    def test_no_meta_is_not_active(self):
        st = get_shared_state()
        assert wr.download_active(st) is False

    def test_meta_present_pid_alive_is_active(self):
        st = get_shared_state()
        st.hset_json(wr.DOWNLOAD_STATE_KEY, wr.DOWNLOAD_STATE_FIELD,
                     {"pid": 12345, "log_path": "/tmp/x.log"})
        assert wr.download_active(st, pid_checker=lambda pid: True) is True

    def test_meta_present_pid_dead_is_not_active(self):
        st = get_shared_state()
        st.hset_json(wr.DOWNLOAD_STATE_KEY, wr.DOWNLOAD_STATE_FIELD,
                     {"pid": 12345, "log_path": "/tmp/x.log"})
        assert wr.download_active(st, pid_checker=lambda pid: False) is False

    def test_meta_without_pid_is_not_active(self):
        st = get_shared_state()
        st.hset_json(wr.DOWNLOAD_STATE_KEY, wr.DOWNLOAD_STATE_FIELD, {})
        assert wr.download_active(st) is False

    def test_shared_state_read_error_defers_as_active(self):
        class _Boom:
            def hget_json(self, *a, **k):
                raise RuntimeError("redis hiccup")
        assert wr.download_active(_Boom()) is True

    def test_pid_checker_error_defers_as_active(self):
        st = get_shared_state()
        st.hset_json(wr.DOWNLOAD_STATE_KEY, wr.DOWNLOAD_STATE_FIELD,
                     {"pid": 12345, "log_path": "/tmp/x.log"})

        def _boom(pid):
            raise RuntimeError("psutil hiccup")

        assert wr.download_active(st, pid_checker=_boom) is True


# ---------------------------------------------------------------------------
# maybe_recycle_once
# ---------------------------------------------------------------------------

class TestMaybeRecycleOnce:
    def test_generation_not_bumped_is_noop(self):
        st = get_shared_state()
        calls = []
        won = wr.maybe_recycle_once(st, birth_generation=0,
                                    exit_callback=lambda: calls.append(1))
        assert won is False
        assert calls == []

    def test_generation_bumped_recycles(self):
        st = get_shared_state()
        st.incr(wr.RECYCLE_GENERATION_KEY)  # 0 -> 1
        calls = []
        won = wr.maybe_recycle_once(st, birth_generation=0,
                                    exit_callback=lambda: calls.append(1))
        assert won is True
        assert calls == [1]

    def test_only_one_of_two_workers_recycles_slot_serialises(self):
        """Two processes (same shared_state) both see generation > birth;
        only the one that wins the recycle:slot NX lock actually recycles —
        the spec's "staggered, never more than one at a time" guarantee."""
        st = get_shared_state()
        st.incr(wr.RECYCLE_GENERATION_KEY)  # 0 -> 1

        calls_a: list = []
        calls_b: list = []
        won_a = wr.maybe_recycle_once(st, birth_generation=0,
                                      exit_callback=lambda: calls_a.append(1),
                                      worker_pid=111)
        won_b = wr.maybe_recycle_once(st, birth_generation=0,
                                      exit_callback=lambda: calls_b.append(1),
                                      worker_pid=222)

        assert (won_a, won_b) in [(True, False), (False, True)]
        assert len(calls_a) + len(calls_b) == 1

    def test_slot_still_held_defers_subsequent_ticks(self):
        st = get_shared_state()
        st.incr(wr.RECYCLE_GENERATION_KEY)
        calls = []
        won1 = wr.maybe_recycle_once(st, birth_generation=0,
                                     exit_callback=lambda: calls.append(1))
        assert won1 is True
        won2 = wr.maybe_recycle_once(st, birth_generation=0,
                                     exit_callback=lambda: calls.append(1))
        assert won2 is False
        assert calls == [1]

    def test_active_download_defers_even_with_free_slot(self):
        st = get_shared_state()
        st.incr(wr.RECYCLE_GENERATION_KEY)
        st.hset_json(wr.DOWNLOAD_STATE_KEY, wr.DOWNLOAD_STATE_FIELD,
                     {"pid": 999, "log_path": "/tmp/x.log"})
        calls = []
        won = wr.maybe_recycle_once(
            st, birth_generation=0, exit_callback=lambda: calls.append(1),
            pid_checker=lambda pid: True)
        assert won is False
        assert calls == []
        # The slot itself must NOT have been claimed either — a deferred
        # tick must not block a sibling worker from claiming it once the
        # download finishes.
        assert st.set_if_absent(wr.RECYCLE_SLOT_KEY, "probe") is True

    def test_download_finishing_lets_a_later_tick_recycle(self):
        st = get_shared_state()
        st.incr(wr.RECYCLE_GENERATION_KEY)
        st.hset_json(wr.DOWNLOAD_STATE_KEY, wr.DOWNLOAD_STATE_FIELD,
                     {"pid": 999, "log_path": "/tmp/x.log"})
        calls = []
        won1 = wr.maybe_recycle_once(
            st, birth_generation=0, exit_callback=lambda: calls.append(1),
            pid_checker=lambda pid: True)
        assert won1 is False

        # Download finished by the next tick.
        won2 = wr.maybe_recycle_once(
            st, birth_generation=0, exit_callback=lambda: calls.append(1),
            pid_checker=lambda pid: False)
        assert won2 is True
        assert calls == [1]

    def test_never_raises_on_broken_shared_state(self):
        class _Boom:
            def get_int(self, *a, **k):
                raise RuntimeError("redis down")

        calls = []
        won = wr.maybe_recycle_once(_Boom(), birth_generation=0,
                                    exit_callback=lambda: calls.append(1))
        assert won is False
        assert calls == []


# ---------------------------------------------------------------------------
# recycle_loop (async)
# ---------------------------------------------------------------------------

class TestRecycleLoop:
    def test_loop_recycles_then_returns_without_further_ticks(self):
        st = get_shared_state()
        st.incr(wr.RECYCLE_GENERATION_KEY)
        calls = []

        async def _run():
            await wr.recycle_loop(
                st, birth_generation=0,
                exit_callback=lambda: calls.append(1),
                check_interval_s=0.01)

        asyncio.run(_run())
        assert calls == [1]

    def test_loop_cancellation_is_clean(self):
        st = get_shared_state()  # generation stays 0 -> never recycles

        async def _run():
            task = asyncio.create_task(
                wr.recycle_loop(st, birth_generation=0,
                                exit_callback=lambda: None,
                                check_interval_s=0.01))
            await asyncio.sleep(0.03)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# install()
# ---------------------------------------------------------------------------

class _FakeRouter:
    def __init__(self):
        self.handlers: dict = {}

    def add_event_handler(self, event, fn):
        self.handlers.setdefault(event, []).append(fn)


class _FakeApp:
    def __init__(self):
        self.router = _FakeRouter()


class TestInstall:
    def test_single_process_does_not_register_handlers(self):
        app = _FakeApp()
        st = get_shared_state()
        wr.install(app, st, env={})
        assert app.router.handlers == {}

    def test_multi_process_registers_startup_and_shutdown(self):
        app = _FakeApp()
        st = get_shared_state()
        wr.install(app, st, env={"EXONIT_SERVER_WORKERS": "4"})
        assert len(app.router.handlers.get("startup", [])) == 1
        assert len(app.router.handlers.get("shutdown", [])) == 1

    def test_startup_creates_task_shutdown_cancels_it_cleanly(self):
        app = _FakeApp()
        st = get_shared_state()
        wr.install(app, st, env={"EXONIT_SERVER_WORKERS": "4"})

        async def _run():
            await app.router.handlers["startup"][0]()
            await asyncio.sleep(0.01)
            await app.router.handlers["shutdown"][0]()

        asyncio.run(_run())  # must not raise / hang
