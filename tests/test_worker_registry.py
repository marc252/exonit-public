"""Tests for exonit/worker_registry.py.

Coverage:
- WorkerRegistry over MemoryBackend: CRUD, contains, keys/items/values
  snapshots, get with default, pop of field, update of field, clear,
  KeyError on absent.
- _WorkerEntryView: write-back (assignment persists), nested read, pop,
  update, setdefault.
- WorkerRegistry over RedisBackend (db=15): same suite, skipped if Redis
  is not reachable.
- Integration: create_app() + TestClient, register→heartbeat→/api/worker/list
  round-trip via Redis registry (skipped if Redis absent).

Run with:
    /usr/bin/python3 -m pytest tests/test_worker_registry.py -q
"""

from __future__ import annotations

import os
import threading

import pytest

from exonit.shared_state import MemoryBackend, RedisBackend, reset_shared_state
from exonit.worker_registry import WorkerRegistry, _WorkerEntryView

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
def mem_reg():
    """A WorkerRegistry backed by a fresh MemoryBackend."""
    return WorkerRegistry(MemoryBackend())


@pytest.fixture
def redis_reg():
    """A WorkerRegistry backed by Redis db=15, flushed before and after."""
    if not _redis_available():
        pytest.skip("Redis not reachable at " + _REDIS_TEST_URL)
    backend = RedisBackend(_REDIS_TEST_URL)
    backend._client.flushdb()
    yield WorkerRegistry(backend)
    backend._client.flushdb()


@pytest.fixture(params=["memory", "redis"])
def reg(request, mem_reg, redis_reg):
    """Parametrised fixture: runs each test against both backends."""
    if request.param == "memory":
        return mem_reg
    return redis_reg


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------

class TestBasicCRUD:
    def test_setitem_and_contains(self, reg):
        reg["w1"] = {"name": "worker-1", "status": "idle"}
        assert "w1" in reg

    def test_setitem_absent_means_not_in(self, reg):
        assert "w99" not in reg

    def test_getitem_returns_view(self, reg):
        reg["w1"] = {"name": "worker-1"}
        view = reg["w1"]
        assert isinstance(view, _WorkerEntryView)

    def test_getitem_absent_raises_key_error(self, reg):
        with pytest.raises(KeyError):
            _ = reg["no_such_worker"]

    def test_delitem_removes_worker(self, reg):
        reg["w1"] = {"name": "worker-1"}
        del reg["w1"]
        assert "w1" not in reg

    def test_delitem_absent_raises_key_error(self, reg):
        with pytest.raises(KeyError):
            del reg["no_such_worker"]

    def test_len(self, reg):
        assert len(reg) == 0
        reg["w1"] = {}
        reg["w2"] = {}
        assert len(reg) == 2

    def test_iter_yields_worker_ids(self, reg):
        reg["w1"] = {}
        reg["w2"] = {}
        assert sorted(reg) == ["w1", "w2"]


# ---------------------------------------------------------------------------
# Write-back via _WorkerEntryView
# ---------------------------------------------------------------------------

class TestWriteBack:
    def test_field_assignment_persists(self, reg):
        reg["w1"] = {"status": "working", "name": "foo"}
        reg["w1"]["status"] = "idle"
        # Reload via the registry to confirm persistence
        assert reg["w1"]["status"] == "idle"

    def test_get_after_write_back(self, reg):
        reg["w1"] = {"status": "working", "name": "bar"}
        reg["w1"]["status"] = "idle"
        snapshot = reg.get("w1")
        assert snapshot["status"] == "idle"

    def test_nested_write_back_chain(self, reg):
        reg["w1"] = {"tasks_completed": 0, "name": "baz"}
        reg["w1"]["tasks_completed"] = reg["w1"]["tasks_completed"] + 5
        assert reg["w1"]["tasks_completed"] == 5

    def test_update_persists(self, reg):
        reg["w1"] = {"a": 1, "b": 2}
        reg["w1"].update({"b": 99, "c": 3})
        snap = reg.get("w1")
        assert snap["b"] == 99
        assert snap["c"] == 3
        assert snap["a"] == 1

    def test_pop_field_persists(self, reg):
        reg["w1"] = {"debug_mode": True, "name": "w"}
        val = reg["w1"].pop("debug_mode", None)
        assert val is True
        assert "debug_mode" not in reg["w1"]

    def test_pop_field_absent_returns_default(self, reg):
        reg["w1"] = {"name": "w"}
        val = reg["w1"].pop("debug_mode", None)
        assert val is None

    def test_pop_field_absent_no_default_raises(self, reg):
        reg["w1"] = {"name": "w"}
        with pytest.raises(KeyError):
            reg["w1"].pop("no_such_field")

    def test_setdefault_absent_sets_and_returns(self, reg):
        reg["w1"] = {"name": "w"}
        val = reg["w1"].setdefault("debug_mode", False)
        assert val is False
        assert reg["w1"]["debug_mode"] is False

    def test_setdefault_present_does_not_overwrite(self, reg):
        reg["w1"] = {"debug_mode": True}
        val = reg["w1"].setdefault("debug_mode", False)
        assert val is True
        assert reg["w1"]["debug_mode"] is True


# ---------------------------------------------------------------------------
# increment() — atomic counter primitive (B6, 2026-07-02)
# ---------------------------------------------------------------------------

class TestIncrement:
    def test_missing_field_starts_at_zero(self, reg):
        reg["w1"] = {"name": "w"}
        new_val = reg["w1"].increment("tasks_completed", 1)
        assert new_val == 1
        assert reg["w1"]["tasks_completed"] == 1

    def test_accumulates(self, reg):
        reg["w1"] = {"tasks_completed": 3}
        reg["w1"].increment("tasks_completed", 1)
        new_val = reg["w1"].increment("tasks_completed", 1)
        assert new_val == 5

    def test_float_delta_preserves_untouched_fields(self, reg):
        reg["w1"] = {"total_processing_time": 10.0, "name": "keep-me"}
        new_val = reg["w1"].increment("total_processing_time", 2.5)
        assert new_val == 12.5
        assert reg["w1"]["name"] == "keep-me"


# ---------------------------------------------------------------------------
# Empty-list guard (B6 follow-up, review CHANGES_REQUESTED, 2026-07-02)
#
# Finding: WorkerRegistry.__setitem__ wrote full worker entries via
# hset_json UNGUARDED, so an entry seeded with an empty list survived that
# write untouched (hset_json doesn't go through the Lua script) and then got
# silently corrupted into {} by the very next unrelated hupdate_json/
# increment/pop/setdefault call for that worker (the Lua round-trips the
# WHOLE resident document through cjson, reproduced by the reviewer on
# Redis db=15). Fixed by applying the same _reject_empty_list_values check
# inside WorkerRegistry.__setitem__. These tests prove the invariant holds
# at every entry point of the registry API — no code path through
# WorkerRegistry/_WorkerEntryView can ever seed an empty list, so the
# corruption trigger (an unrelated atomic touch) can never find one to
# corrupt in the first place.
# ---------------------------------------------------------------------------

class TestEmptyListGuard:
    def test_setitem_full_entry_rejects_top_level_empty_list(self, reg):
        """The reviewer's exact repro, but through the registry API instead
        of a raw backend.hset_json call: __setitem__ must raise, not write."""
        with pytest.raises(ValueError, match="empty list"):
            reg["w1"] = {"name": "worker-1", "recent_errors": []}
        assert "w1" not in reg

    def test_setitem_full_entry_rejects_nested_empty_list(self, reg):
        with pytest.raises(ValueError, match="empty list"):
            reg["w1"] = {"name": "worker-1", "capabilities": {"tags": []}}
        assert "w1" not in reg

    def test_setitem_full_entry_allows_nonempty_list(self, reg):
        """Non-empty lists are unambiguous and must not be rejected."""
        reg["w1"] = {"name": "worker-1", "tags": ["a", "b"]}
        assert reg["w1"]["tags"] == ["a", "b"]

    def test_setitem_rejecting_leaves_no_partial_write(self, reg):
        """A rejected __setitem__ must not leave the worker_id registered at
        all — not even with the OTHER (valid) fields from the same call."""
        with pytest.raises(ValueError):
            reg["w1"] = {"name": "worker-1", "status": "idle", "bad": []}
        assert "w1" not in reg
        assert len(reg) == 0

    def test_entry_view_setitem_rejects_empty_list(self, reg):
        """Per-field assignment on an EXISTING entry is guarded too (this
        path was already covered by the atomic primitives' own guard, prior
        to this follow-up — asserted here for completeness of the "no path
        through the registry API can seed an empty list" invariant)."""
        reg["w1"] = {"name": "w"}
        with pytest.raises(ValueError, match="empty list"):
            reg["w1"]["recent_errors"] = []
        assert "recent_errors" not in reg["w1"]

    def test_entry_view_update_rejects_empty_list(self, reg):
        reg["w1"] = {"name": "w"}
        with pytest.raises(ValueError, match="empty list"):
            reg["w1"].update({"recent_errors": []})
        assert "recent_errors" not in reg["w1"]

    def test_entry_view_setdefault_rejects_empty_list(self, reg):
        reg["w1"] = {"name": "w"}
        with pytest.raises(ValueError, match="empty list"):
            reg["w1"].setdefault("recent_errors", [])
        assert "recent_errors" not in reg["w1"]

    def test_reviewer_repro_blocked_end_to_end(self, reg):
        """The exact scenario from progress/review-b6-registry-2026-07-02.md
        (seed an empty-list field, then perform an unrelated atomic touch)
        blocked at the registry-API layer: the bad write never lands, so
        there is nothing left for the follow-up touch to corrupt. Also
        confirms normal registration + heartbeat-shaped usage is unaffected
        by the guard."""
        with pytest.raises(ValueError):
            reg["w1"] = {"name": "worker-1", "recent_errors": []}
        assert "w1" not in reg

        # Register properly (no empty list) — normal flow unaffected.
        reg["w1"] = {"name": "worker-1"}
        # Unrelated field touch (the reviewer's corruption trigger) — must
        # behave normally since there was never a bad field to corrupt.
        reg["w1"].update({"status": "idle"})
        assert reg["w1"]["status"] == "idle"
        assert reg["w1"]["name"] == "worker-1"


# ---------------------------------------------------------------------------
# Concurrency: hammer .update()/.increment() on the SAME worker entry with
# real threads and assert no lost updates — the actual B6 bug this class was
# rewritten to fix. Deterministic on both backends (MemoryBackend: real
# threading.Lock; RedisBackend: atomic Lua script), not a "usually passes"
# race detector.
# ---------------------------------------------------------------------------

class TestConcurrentEntryMutation:
    _N_THREADS = 20
    _N_PER_THREAD = 50

    def test_concurrent_increment_no_lost_updates(self, reg):
        reg["w1"] = {"tasks_completed": 0}

        def worker():
            for _ in range(self._N_PER_THREAD):
                reg["w1"].increment("tasks_completed", 1)

        threads = [threading.Thread(target=worker) for _ in range(self._N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert reg["w1"]["tasks_completed"] == self._N_THREADS * self._N_PER_THREAD

    def test_concurrent_update_distinct_fields_no_lost_updates(self, reg):
        """Mirrors the heartbeat-vs-task-result race: many callers each
        .update() the same registry entry with their OWN field. Without B6's
        atomic hupdate_json, a caller's read-before-write can miss another
        caller's just-applied field and silently drop it on save."""
        reg["w1"] = {"name": "w1"}

        def worker(idx):
            for _ in range(self._N_PER_THREAD):
                reg["w1"].update({f"field_{idx}": idx})

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(self._N_THREADS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        entry = reg.get("w1")
        assert entry["name"] == "w1"  # original field survives too
        for i in range(self._N_THREADS):
            assert entry.get(f"field_{i}") == i, f"field_{i} lost or wrong: {entry}"

    def test_concurrent_update_and_increment_no_lost_updates(self, reg):
        """The exact shape of worker_task_result: some callers .update()
        status-like fields, others .increment() counters, all against the
        same worker entry concurrently."""
        reg["w1"] = {"tasks_completed": 0, "tasks_failed": 0}
        n_each = self._N_THREADS // 2

        def completer():
            for _ in range(self._N_PER_THREAD):
                reg["w1"].increment("tasks_completed", 1)
                reg["w1"]["status"] = "idle"

        def failer():
            for _ in range(self._N_PER_THREAD):
                reg["w1"].increment("tasks_failed", 1)

        threads = (
            [threading.Thread(target=completer) for _ in range(n_each)]
            + [threading.Thread(target=failer) for _ in range(n_each)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        entry = reg.get("w1")
        assert entry["tasks_completed"] == n_each * self._N_PER_THREAD
        assert entry["tasks_failed"] == n_each * self._N_PER_THREAD
        assert entry["status"] == "idle"


# ---------------------------------------------------------------------------
# Snapshot methods: keys / items / values
# ---------------------------------------------------------------------------

class TestSnapshots:
    def test_keys_returns_list(self, reg):
        reg["w1"] = {}
        reg["w2"] = {}
        assert sorted(reg.keys()) == ["w1", "w2"]

    def test_values_snapshot_does_not_persist(self, reg):
        """Mutating a snapshot dict from values() must NOT persist."""
        reg["w1"] = {"status": "idle"}
        for entry in reg.values():
            entry["status"] = "MUTATED"
        # The registry entry should be unchanged
        assert reg["w1"]["status"] == "idle"

    def test_items_snapshot_is_plain_dicts(self, reg):
        reg["w1"] = {"name": "foo"}
        items = reg.items()
        assert len(items) == 1
        wid, entry = items[0]
        assert wid == "w1"
        assert entry == {"name": "foo"}
        assert isinstance(entry, dict)

    def test_values_are_plain_dicts(self, reg):
        reg["w1"] = {"x": 1}
        vals = reg.values()
        assert all(isinstance(v, dict) for v in vals)


# ---------------------------------------------------------------------------
# get with default
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_existing_returns_dict(self, reg):
        reg["w1"] = {"name": "foo"}
        result = reg.get("w1")
        assert isinstance(result, dict)
        assert result["name"] == "foo"

    def test_get_absent_returns_none(self, reg):
        assert reg.get("absent") is None

    def test_get_absent_returns_custom_default(self, reg):
        result = reg.get("absent", {})
        assert result == {}

    def test_get_absent_get_on_default_dict(self, reg):
        """Mirrors server.py pattern: workers_state.get(wid, {}).get("name", "")."""
        name = reg.get("absent", {}).get("name", "fallback")
        assert name == "fallback"


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_empties_registry(self, reg):
        reg["w1"] = {}
        reg["w2"] = {}
        reg.clear()
        assert len(reg) == 0

    def test_clear_empty_is_noop(self, reg):
        reg.clear()  # must not raise
        assert len(reg) == 0


# ---------------------------------------------------------------------------
# contains / __iter__ / __len__ consistency
# ---------------------------------------------------------------------------

class TestContainsIterLen:
    def test_contains_after_set(self, reg):
        reg["w1"] = {"name": "w"}
        assert "w1" in reg

    def test_not_contains_after_del(self, reg):
        reg["w1"] = {}
        del reg["w1"]
        assert "w1" not in reg

    def test_overwrite_does_not_add_duplicate(self, reg):
        reg["w1"] = {"a": 1}
        reg["w1"] = {"a": 2}
        assert len(reg) == 1
        assert reg["w1"]["a"] == 2


# ---------------------------------------------------------------------------
# Integration: create_app + TestClient + Redis round-trip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _redis_available(), reason="Redis not reachable")
def test_registry_redis_roundtrip_via_app(tmp_path, monkeypatch):
    """Register a worker, send a heartbeat, call /api/worker/list.

    Verifies that the entry made a full round-trip through the Redis-backed
    WorkerRegistry (we read the hash directly from Redis to confirm).
    """
    import time

    # Point at Redis db=15 so we don't pollute db=0 (production).
    monkeypatch.setenv("EXONIT_REDIS_URL", _REDIS_TEST_URL)
    monkeypatch.delenv("EXONIT_SHARED_STATE_BACKEND", raising=False)

    # Ensure a fresh singleton backed by Redis db=15.
    reset_shared_state()

    # Flush the test DB before we start.
    backend = RedisBackend(_REDIS_TEST_URL)
    backend._client.flushdb()

    try:
        os.environ["EXONIT_DB_PATH"] = str(tmp_path / "test_redis_rt.db")
        os.environ["EXONIT_REPORTS_DIR"] = str(tmp_path / "reports_redis_rt")
        os.makedirs(tmp_path / "reports_redis_rt", exist_ok=True)

        from fastapi.testclient import TestClient
        from exonit.server import create_app

        app = create_app()
        client = TestClient(app)
        auth = {"Authorization": "Bearer exonit-default-token"}

        # Register
        resp = client.post(
            "/api/worker/register",
            json={"name": "test-redis-worker", "capabilities": {"gpu": "none"}},
            headers=auth,
        )
        assert resp.status_code == 200
        worker_id = resp.json()["worker_id"]

        # Heartbeat
        resp = client.post(
            "/api/worker/heartbeat",
            json={"worker_id": worker_id, "status": "idle", "current_tic": None},
            headers=auth,
        )
        assert resp.status_code == 200

        # /api/worker/list must contain our worker
        resp = client.get("/api/worker/list")
        assert resp.status_code == 200
        workers = resp.json()["workers"]
        ids = [w["worker_id"] for w in workers]
        assert worker_id in ids

        # Confirm the entry is actually in Redis (not just in-memory).
        raw = backend.hget_json("workers:registry", worker_id)
        assert raw is not None, "Entry not found in Redis hash"
        assert raw.get("name") == "test-redis-worker"

    finally:
        backend._client.flushdb()
        reset_shared_state()
        os.environ.pop("EXONIT_DB_PATH", None)
        os.environ.pop("EXONIT_REPORTS_DIR", None)
