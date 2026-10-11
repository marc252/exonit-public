"""Tests for exonit/shared_state.py.

Coverage:
- set/get/delete
- set_if_absent (NX semantics, including re-use after TTL expiry)
- incr / get_int / set_int (including default when key absent)
- hset_json / hget_json round-trip with nested dict, hgetall_json, hdel, hkeys
- TTL determinism via injected clock (MemoryBackend only)
- Redis backend parametrised — skipped if Redis is unreachable
- BlockingConnectionPool: pool type, env config, concurrency regression

Run with:
    /usr/bin/python3 -m pytest tests/test_shared_state.py -q
"""

from __future__ import annotations

import threading
import time

import pytest

from exonit.shared_state import (
    MemoryBackend,
    RedisBackend,
    reset_shared_state,
    get_shared_state,
    DEFAULT_REDIS_URL,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_REDIS_TEST_URL = "redis://localhost:6379/15"


def _redis_available() -> bool:
    """Return True if the Redis test DB (db=15) is reachable."""
    try:
        b = RedisBackend(_REDIS_TEST_URL)
        return b.ping()
    except Exception:
        return False


_skip_no_redis = pytest.mark.skipif(
    not _redis_available(),
    reason="Redis not reachable at " + _REDIS_TEST_URL,
)


@pytest.fixture
def memory_backend():
    """A fresh MemoryBackend with the real clock."""
    return MemoryBackend()


@pytest.fixture
def redis_backend():
    """A RedisBackend connected to DB 15, flushed before and after the test."""
    if not _redis_available():
        pytest.skip("Redis not reachable at " + _REDIS_TEST_URL)
    b = RedisBackend(_REDIS_TEST_URL)
    b._client.flushdb()
    yield b
    b._client.flushdb()


@pytest.fixture(params=["memory", "redis"])
def backend(request, memory_backend, redis_backend):
    """Parametrised fixture: runs each test against both backends."""
    if request.param == "memory":
        return memory_backend
    return redis_backend


# ---------------------------------------------------------------------------
# Basic scalar ops: set / get / delete
# ---------------------------------------------------------------------------


class TestSetGetDelete:
    def test_get_missing_returns_none(self, backend):
        assert backend.get("no_such_key") is None

    def test_set_then_get(self, backend):
        backend.set("k", "hello")
        assert backend.get("k") == "hello"

    def test_overwrite(self, backend):
        backend.set("k", "first")
        backend.set("k", "second")
        assert backend.get("k") == "second"

    def test_delete_existing(self, backend):
        backend.set("k", "v")
        backend.delete("k")
        assert backend.get("k") is None

    def test_delete_absent_is_noop(self, backend):
        backend.delete("nonexistent")  # must not raise


# ---------------------------------------------------------------------------
# set_if_absent (NX semantics)
# ---------------------------------------------------------------------------


class TestSetIfAbsent:
    def test_first_call_returns_true(self, backend):
        assert backend.set_if_absent("lock", "owner1") is True

    def test_second_call_returns_false(self, backend):
        backend.set_if_absent("lock", "owner1")
        assert backend.set_if_absent("lock", "owner2") is False

    def test_value_unchanged_after_failed_nx(self, backend):
        backend.set_if_absent("lock", "owner1")
        backend.set_if_absent("lock", "owner2")
        assert backend.get("lock") == "owner1"

    def test_succeeds_after_delete(self, backend):
        backend.set_if_absent("lock", "owner1")
        backend.delete("lock")
        assert backend.set_if_absent("lock", "owner2") is True
        assert backend.get("lock") == "owner2"


# ---------------------------------------------------------------------------
# TTL — MemoryBackend with fake clock
# ---------------------------------------------------------------------------


class FakeClock:
    """Simple mutable clock for TTL tests."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class TestMemoryTTL:
    def test_key_exists_before_expiry(self):
        clock = FakeClock(0.0)
        b = MemoryBackend(clock=clock)
        b.set("k", "v", ttl=10.0)
        clock.now = 9.9
        assert b.get("k") == "v"

    def test_key_absent_after_expiry(self):
        clock = FakeClock(0.0)
        b = MemoryBackend(clock=clock)
        b.set("k", "v", ttl=10.0)
        clock.now = 10.0  # exactly at boundary → expired
        assert b.get("k") is None

    def test_set_if_absent_succeeds_after_ttl_expiry(self):
        clock = FakeClock(0.0)
        b = MemoryBackend(clock=clock)
        assert b.set_if_absent("lock", "a", ttl=5.0) is True
        clock.now = 4.9
        assert b.set_if_absent("lock", "b") is False  # still alive
        clock.now = 5.0
        assert b.set_if_absent("lock", "c") is True   # expired → accepted
        assert b.get("lock") == "c"

    def test_key_without_ttl_never_expires(self):
        clock = FakeClock(0.0)
        b = MemoryBackend(clock=clock)
        b.set("eternal", "yes")
        clock.now = 1_000_000.0
        assert b.get("eternal") == "yes"


# ---------------------------------------------------------------------------
# TTL — RedisBackend (only immediate existence; no real-time wait)
# ---------------------------------------------------------------------------


class TestRedisTTL:
    def test_key_exists_immediately_after_set_with_ttl(self, redis_backend):
        redis_backend.set("k", "v", ttl=60.0)
        assert redis_backend.get("k") == "v"

    def test_set_if_absent_with_ttl_exists_immediately(self, redis_backend):
        result = redis_backend.set_if_absent("lock", "x", ttl=60.0)
        assert result is True
        assert redis_backend.get("lock") == "x"


# ---------------------------------------------------------------------------
# Integer counters
# ---------------------------------------------------------------------------


class TestIntCounters:
    def test_get_int_default_when_absent(self, backend):
        assert backend.get_int("counter") == 0

    def test_get_int_custom_default(self, backend):
        assert backend.get_int("counter", 42) == 42

    def test_set_int_then_get_int(self, backend):
        backend.set_int("counter", 7)
        assert backend.get_int("counter") == 7

    def test_incr_from_zero(self, backend):
        result = backend.incr("counter")
        assert result == 1

    def test_incr_accumulates(self, backend):
        backend.incr("counter")
        backend.incr("counter")
        result = backend.incr("counter", amount=3)
        assert result == 5

    def test_incr_uses_amount(self, backend):
        result = backend.incr("counter", amount=10)
        assert result == 10

    def test_incr_after_set_int(self, backend):
        backend.set_int("counter", 100)
        result = backend.incr("counter", 5)
        assert result == 105

    def test_get_int_after_incr(self, backend):
        backend.incr("counter", 3)
        assert backend.get_int("counter") == 3


# ---------------------------------------------------------------------------
# Hash JSON ops
# ---------------------------------------------------------------------------


class TestHashJSON:
    _NESTED = {"id": 42, "meta": {"tags": ["a", "b"], "active": True}}

    def test_hset_hget_round_trip(self, backend):
        backend.hset_json("workers", "w1", self._NESTED)
        result = backend.hget_json("workers", "w1")
        assert result == self._NESTED

    def test_hget_absent_field_returns_none(self, backend):
        assert backend.hget_json("workers", "no_such_field") is None

    def test_hget_absent_key_returns_none(self, backend):
        assert backend.hget_json("no_such_hash", "field") is None

    def test_hgetall_returns_all_fields(self, backend):
        backend.hset_json("workers", "w1", {"a": 1})
        backend.hset_json("workers", "w2", {"b": 2})
        result = backend.hgetall_json("workers")
        assert result == {"w1": {"a": 1}, "w2": {"b": 2}}

    def test_hgetall_empty_key_returns_empty_dict(self, backend):
        assert backend.hgetall_json("no_such_hash") == {}

    def test_hdel_removes_field(self, backend):
        backend.hset_json("workers", "w1", {"x": 1})
        backend.hset_json("workers", "w2", {"y": 2})
        backend.hdel("workers", "w1")
        assert backend.hget_json("workers", "w1") is None
        assert backend.hget_json("workers", "w2") == {"y": 2}

    def test_hdel_absent_field_is_noop(self, backend):
        backend.hset_json("workers", "w1", {"x": 1})
        backend.hdel("workers", "nonexistent")  # must not raise
        assert backend.hget_json("workers", "w1") == {"x": 1}

    def test_hkeys_returns_field_names(self, backend):
        backend.hset_json("workers", "w1", {})
        backend.hset_json("workers", "w2", {})
        keys = backend.hkeys("workers")
        assert sorted(keys) == ["w1", "w2"]

    def test_hkeys_absent_key_returns_empty_list(self, backend):
        assert backend.hkeys("no_such_hash") == []

    def test_overwrite_field(self, backend):
        backend.hset_json("workers", "w1", {"v": 1})
        backend.hset_json("workers", "w1", {"v": 99})
        assert backend.hget_json("workers", "w1") == {"v": 99}


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------


class TestGetSharedState:
    def setup_method(self):
        reset_shared_state()

    def teardown_method(self):
        reset_shared_state()

    def test_force_memory_returns_memory_backend(self):
        b = get_shared_state(force_memory=True)
        assert isinstance(b, MemoryBackend)

    def test_env_memory_returns_memory_backend(self, monkeypatch):
        monkeypatch.setenv("EXONIT_SHARED_STATE_BACKEND", "memory")
        b = get_shared_state()
        assert isinstance(b, MemoryBackend)

    def test_singleton_same_instance(self):
        b1 = get_shared_state(force_memory=True)
        b2 = get_shared_state(force_memory=True)
        assert b1 is b2

    def test_reset_clears_singleton(self):
        b1 = get_shared_state(force_memory=True)
        reset_shared_state()
        b2 = get_shared_state(force_memory=True)
        assert b1 is not b2

    def test_unreachable_redis_falls_back_to_memory(self, monkeypatch):
        monkeypatch.delenv("EXONIT_SHARED_STATE_BACKEND", raising=False)
        b = get_shared_state(url="redis://127.0.0.1:19999/0")  # nothing there
        assert isinstance(b, MemoryBackend)


# ---------------------------------------------------------------------------
# hsetnx_json / hlen  (A2a additions)
# ---------------------------------------------------------------------------


class TestHsetnxJson:
    """hsetnx_json: atomic set-if-absent for hash fields."""

    def test_first_call_returns_true(self, backend):
        assert backend.hsetnx_json("h", "f1", {"v": 1}) is True

    def test_field_written_on_first_call(self, backend):
        backend.hsetnx_json("h", "f1", {"v": 1})
        assert backend.hget_json("h", "f1") == {"v": 1}

    def test_second_call_same_field_returns_false(self, backend):
        backend.hsetnx_json("h", "f1", {"v": 1})
        result = backend.hsetnx_json("h", "f1", {"v": 99})
        assert result is False

    def test_value_not_overwritten_when_field_exists(self, backend):
        backend.hsetnx_json("h", "f1", {"v": 1})
        backend.hsetnx_json("h", "f1", {"v": 99})
        assert backend.hget_json("h", "f1") == {"v": 1}

    def test_different_fields_in_same_hash_succeed(self, backend):
        assert backend.hsetnx_json("h", "f1", {"a": 1}) is True
        assert backend.hsetnx_json("h", "f2", {"b": 2}) is True
        assert backend.hget_json("h", "f1") == {"a": 1}
        assert backend.hget_json("h", "f2") == {"b": 2}

    def test_succeeds_after_hdel(self, backend):
        backend.hsetnx_json("h", "f1", {"v": 1})
        backend.hdel("h", "f1")
        assert backend.hsetnx_json("h", "f1", {"v": 2}) is True
        assert backend.hget_json("h", "f1") == {"v": 2}

    def test_two_concurrent_claims_one_winner(self, backend):
        """Simulates two workers racing for the same candidate_id.

        The first hsetnx_json wins (returns True); the second returns False.
        This is the cross-process assignment-claim pattern.
        """
        winner = backend.hsetnx_json("assignments", "cand-42", {"ts": 100.0})
        loser = backend.hsetnx_json("assignments", "cand-42", {"ts": 200.0})
        assert winner is True
        assert loser is False
        # Value is the winner's
        assert backend.hget_json("assignments", "cand-42") == {"ts": 100.0}


class TestHlen:
    """hlen: number of fields in a hash."""

    def test_empty_key_returns_zero(self, backend):
        assert backend.hlen("no_such_hash") == 0

    def test_one_field(self, backend):
        backend.hset_json("h", "f1", {})
        assert backend.hlen("h") == 1

    def test_multiple_fields(self, backend):
        backend.hset_json("h", "f1", {})
        backend.hset_json("h", "f2", {})
        backend.hset_json("h", "f3", {})
        assert backend.hlen("h") == 3

    def test_overwrite_does_not_increase_count(self, backend):
        backend.hset_json("h", "f1", {"v": 1})
        backend.hset_json("h", "f1", {"v": 2})
        assert backend.hlen("h") == 1

    def test_decreases_after_hdel(self, backend):
        backend.hset_json("h", "f1", {})
        backend.hset_json("h", "f2", {})
        backend.hdel("h", "f1")
        assert backend.hlen("h") == 1


# ---------------------------------------------------------------------------
# Atomic sub-document mutation (B6, 2026-07-02): hupdate_json /
# hincr_json_field / hdelete_json_field / hsetdefault_json_field.
# ---------------------------------------------------------------------------


class TestHupdateJson:
    def test_missing_field_treated_as_empty_dict(self, backend):
        result = backend.hupdate_json("h", "f1", {"a": 1})
        assert result == {"a": 1}
        assert backend.hget_json("h", "f1") == {"a": 1}

    def test_merges_into_existing_untouched_fields_survive(self, backend):
        backend.hset_json("h", "f1", {"a": 1, "b": "keep"})
        result = backend.hupdate_json("h", "f1", {"a": 2})
        assert result == {"a": 2, "b": "keep"}
        assert backend.hget_json("h", "f1") == {"a": 2, "b": "keep"}

    def test_adds_new_key_alongside_existing(self, backend):
        backend.hset_json("h", "f1", {"a": 1})
        backend.hupdate_json("h", "f1", {"c": 3})
        assert backend.hget_json("h", "f1") == {"a": 1, "c": 3}

    def test_overwrites_multiple_keys_at_once(self, backend):
        backend.hset_json("h", "f1", {"a": 1, "b": 2, "c": 3})
        backend.hupdate_json("h", "f1", {"a": 10, "b": 20})
        assert backend.hget_json("h", "f1") == {"a": 10, "b": 20, "c": 3}

    def test_float_value_survives(self, backend):
        backend.hupdate_json("h", "f1", {"total_processing_time": 12.5})
        assert backend.hget_json("h", "f1")["total_processing_time"] == 12.5

    def test_none_value_sets_present_null_not_delete(self, backend):
        backend.hset_json("h", "f1", {"current_tic": "TIC1"})
        backend.hupdate_json("h", "f1", {"current_tic": None})
        entry = backend.hget_json("h", "f1")
        assert "current_tic" in entry
        assert entry["current_tic"] is None

    def test_nested_dict_value_survives(self, backend):
        caps = {"gpu": "RTX 5060 Ti", "cpu_cores": 24, "hardware": {"vendor": "nvidia"}}
        backend.hupdate_json("h", "f1", {"capabilities": caps})
        assert backend.hget_json("h", "f1")["capabilities"] == caps

    def test_unicode_worker_name_survives(self, backend):
        backend.hupdate_json("h", "f1", {"name": "wörker-é-日本語"})
        assert backend.hget_json("h", "f1")["name"] == "wörker-é-日本語"

    def test_bool_and_int_values_survive(self, backend):
        backend.hupdate_json("h", "f1", {"debug_mode": True, "tasks_completed": 7})
        entry = backend.hget_json("h", "f1")
        assert entry["debug_mode"] is True
        assert entry["tasks_completed"] == 7

    def test_returns_merged_entry(self, backend):
        backend.hset_json("h", "f1", {"a": 1})
        result = backend.hupdate_json("h", "f1", {"b": 2})
        assert result == backend.hget_json("h", "f1")


class TestHincrJsonField:
    def test_missing_field_starts_at_zero(self, backend):
        new_val = backend.hincr_json_field("h", "f1", "tasks_completed", 1)
        assert new_val == 1
        assert backend.hget_json("h", "f1")["tasks_completed"] == 1

    def test_missing_subfield_starts_at_zero(self, backend):
        backend.hset_json("h", "f1", {"name": "w"})
        new_val = backend.hincr_json_field("h", "f1", "tasks_completed", 1)
        assert new_val == 1

    def test_accumulates_across_calls(self, backend):
        backend.hincr_json_field("h", "f1", "tasks_completed", 1)
        backend.hincr_json_field("h", "f1", "tasks_completed", 1)
        new_val = backend.hincr_json_field("h", "f1", "tasks_completed", 3)
        assert new_val == 5

    def test_float_delta(self, backend):
        backend.hincr_json_field("h", "f1", "total_processing_time", 12.5)
        new_val = backend.hincr_json_field("h", "f1", "total_processing_time", 7.5)
        assert new_val == 20.0

    def test_non_numeric_existing_value_treated_as_zero(self, backend):
        backend.hset_json("h", "f1", {"tasks_completed": "not_a_number"})
        new_val = backend.hincr_json_field("h", "f1", "tasks_completed", 1)
        assert new_val == 1

    def test_untouched_fields_survive(self, backend):
        backend.hset_json("h", "f1", {"name": "w", "tasks_completed": 0})
        backend.hincr_json_field("h", "f1", "tasks_completed", 1)
        assert backend.hget_json("h", "f1")["name"] == "w"


class TestHdeleteJsonField:
    def test_removes_field(self, backend):
        backend.hset_json("h", "f1", {"debug_mode": True, "name": "w"})
        backend.hdelete_json_field("h", "f1", "debug_mode")
        entry = backend.hget_json("h", "f1")
        assert "debug_mode" not in entry
        assert entry["name"] == "w"

    def test_returns_previous_entry_with_value_present(self, backend):
        backend.hset_json("h", "f1", {"debug_mode": True})
        previous = backend.hdelete_json_field("h", "f1", "debug_mode")
        assert previous["debug_mode"] is True

    def test_absent_subfield_is_noop_previous_lacks_key(self, backend):
        backend.hset_json("h", "f1", {"name": "w"})
        previous = backend.hdelete_json_field("h", "f1", "debug_mode")
        assert "debug_mode" not in previous
        assert backend.hget_json("h", "f1") == {"name": "w"}

    def test_absent_hash_field_is_noop(self, backend):
        previous = backend.hdelete_json_field("h", "no_such_field", "x")
        assert previous == {}


class TestHsetdefaultJsonField:
    def test_absent_sets_and_returns_default(self, backend):
        result = backend.hsetdefault_json_field("h", "f1", "debug_mode", False)
        assert result is False
        assert backend.hget_json("h", "f1")["debug_mode"] is False

    def test_present_does_not_overwrite(self, backend):
        backend.hset_json("h", "f1", {"debug_mode": True})
        result = backend.hsetdefault_json_field("h", "f1", "debug_mode", False)
        assert result is True
        assert backend.hget_json("h", "f1")["debug_mode"] is True

    def test_untouched_fields_survive(self, backend):
        backend.hset_json("h", "f1", {"name": "w"})
        backend.hsetdefault_json_field("h", "f1", "debug_mode", False)
        assert backend.hget_json("h", "f1")["name"] == "w"


class TestRedisCjsonCaveats:
    """Direct, empirically-verified documentation of the two Redis-backend-
    only cjson round-trip quirks the atomic sub-document methods carry (see
    the caveat on ``SharedStateBackend.hupdate_json`` et al. in
    ``exonit/shared_state.py``). MemoryBackend does not go through cjson
    (plain ``json.dumps``/``json.loads``) so it does not exhibit either
    quirk — covered by the negative-control tests below.
    """

    @_skip_no_redis
    def test_whole_number_float_becomes_int_on_redis(self, redis_backend):
        """Numerically harmless (0 == 0.0) but a real type change: a
        whole-number float loses its decimal point through cjson and comes
        back as a Python int, not a float."""
        redis_backend.hupdate_json("h", "f1", {"total_processing_time": 0.0})
        value = redis_backend.hget_json("h", "f1")["total_processing_time"]
        assert value == 0.0  # value-equal
        assert isinstance(value, int)  # but type-changed — documented, not a bug

    def test_whole_number_float_stays_float_on_memory(self, memory_backend):
        """Negative control: MemoryBackend preserves the float/int distinction
        exactly (no cjson involved)."""
        memory_backend.hupdate_json("h", "f1", {"total_processing_time": 0.0})
        value = memory_backend.hget_json("h", "f1")["total_processing_time"]
        assert isinstance(value, float)

    @_skip_no_redis
    def test_fractional_float_survives_redis(self, redis_backend):
        """Floats WITH a fractional part are unaffected (only whole numbers
        lose the type marker)."""
        redis_backend.hupdate_json("h", "f1", {"total_processing_time": 12.5})
        value = redis_backend.hget_json("h", "f1")["total_processing_time"]
        assert value == 12.5
        assert isinstance(value, float)

    @_skip_no_redis
    def test_empty_list_value_rejected_on_redis(self, redis_backend):
        """The real corruption mode (empty list -> empty dict) is guarded
        against by raising instead of silently corrupting the registry."""
        with pytest.raises(ValueError, match="empty list"):
            redis_backend.hupdate_json("h", "f1", {"tags": []})
        # Nothing was written — no partial/corrupted state left behind.
        assert redis_backend.hget_json("h", "f1") is None

    @_skip_no_redis
    def test_empty_list_nested_in_dict_also_rejected_on_redis(self, redis_backend):
        with pytest.raises(ValueError, match="empty list"):
            redis_backend.hupdate_json("h", "f1", {"meta": {"tags": []}})

    @_skip_no_redis
    def test_nonempty_list_survives_redis(self, redis_backend):
        """Non-empty arrays are unambiguous and round-trip correctly — only
        the empty case is corrupted/rejected."""
        redis_backend.hupdate_json("h", "f1", {"tags": ["a", "b"]})
        assert redis_backend.hget_json("h", "f1")["tags"] == ["a", "b"]

    def test_empty_list_value_allowed_on_memory(self, memory_backend):
        """Negative control: MemoryBackend has no cjson round trip, so the
        guard is Redis-only and empty lists work fine there."""
        memory_backend.hupdate_json("h", "f1", {"tags": []})
        assert memory_backend.hget_json("h", "f1")["tags"] == []

    @_skip_no_redis
    def test_hincr_and_hsetdefault_also_guard_empty_list(self, redis_backend):
        """The guard applies to every atomic mutation entry point, not just
        hupdate_json — hincr_json_field's delta is never a list, but
        hsetdefault_json_field's default can be."""
        with pytest.raises(ValueError, match="empty list"):
            redis_backend.hsetdefault_json_field("h", "f1", "tags", [])

    @_skip_no_redis
    def test_hset_json_itself_is_intentionally_ungated(self, redis_backend):
        """Review finding (B6 follow-up, 2026-07-02): the empty-list guard
        must NOT be a blanket check inside hset_json — that method is shared
        by unrelated hash keys that legitimately store empty lists (e.g. the
        pipeline state's ``log: {"lines": []}``). This test documents that
        boundary deliberately: hset_json stays permissive (this is correct,
        not a bug), and the *reviewer's exact reproduced corruption* is still
        directly reachable by going around the guarded layer (WorkerRegistry)
        and calling hset_json/hupdate_json directly. The actual fix for
        real callers lives one layer up, in
        ``worker_registry.WorkerRegistry.__setitem__`` (see
        ``tests/test_worker_registry.py::TestEmptyListGuard`` for proof that
        NO path through the registry API can reach this)."""
        redis_backend.hset_json("h", "f1", {"name": "worker-1", "recent_errors": []})
        assert redis_backend.hget_json("h", "f1")["recent_errors"] == []
        # An unrelated field touch via the atomic primitive corrupts it —
        # exactly the reviewer's repro. This is expected/documented at THIS
        # layer; real callers never reach it because WorkerRegistry guards
        # the only path that can seed the bad value in the first place.
        redis_backend.hupdate_json("h", "f1", {"status": "idle"})
        assert redis_backend.hget_json("h", "f1")["recent_errors"] == {}


# ---------------------------------------------------------------------------
# Concurrency: hammer hupdate_json / hincr_json_field with real threads and
# assert no lost updates (the actual B6 bug this whole primitive exists to
# fix). MemoryBackend's lock makes this deterministic (100% pass rate, not
# a "usually passes" race detector) — Redis's Lua-script atomicity gives the
# same guarantee across real network round trips.
# ---------------------------------------------------------------------------


class TestConcurrentMutation:
    _N_THREADS = 20
    _N_PER_THREAD = 50

    def test_concurrent_increment_no_lost_updates(self, backend):
        def worker():
            for _ in range(self._N_PER_THREAD):
                backend.hincr_json_field("h", "f1", "tasks_completed", 1)

        threads = [threading.Thread(target=worker) for _ in range(self._N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        final = backend.hget_json("h", "f1")["tasks_completed"]
        assert final == self._N_THREADS * self._N_PER_THREAD

    def test_concurrent_update_distinct_fields_no_lost_updates(self, backend):
        """Each thread repeatedly writes its OWN field via hupdate_json.
        Without atomicity, a thread's read-before-write can miss another
        thread's just-applied field and clobber it back out on save — the
        exact B6 bug. With atomicity, all N fields must survive."""

        def worker(idx):
            for _ in range(self._N_PER_THREAD):
                backend.hupdate_json("h", "f1", {f"field_{idx}": idx})

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(self._N_THREADS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        entry = backend.hget_json("h", "f1")
        for i in range(self._N_THREADS):
            assert entry.get(f"field_{i}") == i, f"field_{i} lost or wrong: {entry}"

    def test_concurrent_mixed_increment_and_update_no_lost_updates(self, backend):
        """Half the threads increment a shared counter, half set their own
        distinct field — both kinds of write must survive interleaved."""
        n_each = self._N_THREADS // 2

        def incrementer():
            for _ in range(self._N_PER_THREAD):
                backend.hincr_json_field("h", "f1", "counter", 1)

        def setter(idx):
            for _ in range(self._N_PER_THREAD):
                backend.hupdate_json("h", "f1", {f"owner_{idx}": idx})

        threads = (
            [threading.Thread(target=incrementer) for _ in range(n_each)]
            + [threading.Thread(target=setter, args=(i,)) for i in range(n_each)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        entry = backend.hget_json("h", "f1")
        assert entry["counter"] == n_each * self._N_PER_THREAD
        for i in range(n_each):
            assert entry.get(f"owner_{i}") == i


# ---------------------------------------------------------------------------
# BlockingConnectionPool — pool type, env config, concurrency regression
# ---------------------------------------------------------------------------

# _redis_available() and _skip_no_redis are defined at the top of this module.


class TestBlockingPool:
    """Verify that RedisBackend uses BlockingConnectionPool and is concurrency-safe."""

    @_skip_no_redis
    def test_uses_blocking_connection_pool(self):
        """RedisBackend must use BlockingConnectionPool, not the default pool."""
        import redis as redis_lib

        b = RedisBackend(_REDIS_TEST_URL)
        pool = b._client.connection_pool
        assert isinstance(pool, redis_lib.BlockingConnectionPool), (
            f"Expected BlockingConnectionPool, got {type(pool).__name__}"
        )

    @_skip_no_redis
    def test_env_max_connections_respected(self, monkeypatch):
        """EXONIT_REDIS_MAX_CONNECTIONS env var is parsed and applied to the pool."""
        monkeypatch.setenv("EXONIT_REDIS_MAX_CONNECTIONS", "7")
        monkeypatch.delenv("EXONIT_REDIS_POOL_TIMEOUT", raising=False)
        b = RedisBackend(_REDIS_TEST_URL)
        assert b._client.connection_pool.max_connections == 7

    @_skip_no_redis
    def test_env_max_connections_invalid_uses_default(self, monkeypatch):
        """Invalid EXONIT_REDIS_MAX_CONNECTIONS falls back to the default (256)."""
        monkeypatch.setenv("EXONIT_REDIS_MAX_CONNECTIONS", "not_a_number")
        b = RedisBackend(_REDIS_TEST_URL)
        assert b._client.connection_pool.max_connections == RedisBackend._DEFAULT_MAX_CONNECTIONS

    @_skip_no_redis
    def test_env_max_connections_zero_uses_default(self, monkeypatch):
        """Zero/negative EXONIT_REDIS_MAX_CONNECTIONS falls back to the default."""
        monkeypatch.setenv("EXONIT_REDIS_MAX_CONNECTIONS", "0")
        b = RedisBackend(_REDIS_TEST_URL)
        assert b._client.connection_pool.max_connections == RedisBackend._DEFAULT_MAX_CONNECTIONS

    @_skip_no_redis
    def test_control_negative_normal_pool_raises_under_load(self):
        """Regression control: a normal ConnectionPool with max_connections=5 raises
        MaxConnectionsError when >5 threads hold a connection simultaneously.

        This documents the original production bug and confirms our test harness
        can reproduce it reliably.
        """
        import redis as redis_lib

        # Build a tiny non-blocking pool (the bug scenario).
        pool = redis_lib.ConnectionPool.from_url(
            _REDIS_TEST_URL,
            decode_responses=True,
            max_connections=5,
        )
        client = redis_lib.Redis(connection_pool=pool)
        client.flushdb()

        errors: list[Exception] = []
        threads_ready = threading.Barrier(40)

        def worker(idx: int) -> None:
            threads_ready.wait()  # all threads start simultaneously
            try:
                # Execute multiple ops in a tight loop to saturate the pool.
                for _ in range(20):
                    client.set(f"key:{idx}", str(idx))
                    client.get(f"key:{idx}")
            except redis_lib.exceptions.MaxConnectionsError as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(40)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        client.flushdb()
        assert errors, (
            "Expected MaxConnectionsError from a non-blocking pool under load, "
            "but no errors were raised.  The test harness may not be generating "
            "enough concurrency on this system.  The positive test below is still "
            "the authoritative check."
        )

    @_skip_no_redis
    def test_blocking_pool_no_errors_under_load(self):
        """BlockingConnectionPool with max_connections=5 does NOT raise under the
        same load that causes MaxConnectionsError with a normal pool.

        This is the primary regression test for the production bug fix.
        """
        b = RedisBackend.__new__(RedisBackend)
        # Inject a tiny BlockingConnectionPool directly to make the test deterministic
        # regardless of the EXONIT_REDIS_MAX_CONNECTIONS env var.
        import redis as redis_lib

        pool = redis_lib.BlockingConnectionPool.from_url(
            _REDIS_TEST_URL,
            decode_responses=True,
            max_connections=5,
            timeout=30,
        )
        b._client = redis_lib.Redis(connection_pool=pool)
        b._client.flushdb()

        errors: list[Exception] = []
        threads_ready = threading.Barrier(40)

        def worker(idx: int) -> None:
            threads_ready.wait()
            try:
                for _ in range(20):
                    b._client.set(f"key:{idx}", str(idx))
                    b._client.get(f"key:{idx}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(40)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        b._client.flushdb()
        assert not errors, (
            f"BlockingConnectionPool raised unexpected errors: {errors[:5]}"
        )
