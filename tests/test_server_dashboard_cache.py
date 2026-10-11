"""Tests for the _cached_json shared-state response cache in server.py.

Verifies three properties of the caching helper used by the six expensive
dashboard aggregate endpoints (/api/stats, /api/status, /api/worker/list,
/api/health/work, /api/backfill/fpp, /api/backfill/nfpp):

1. Within TTL: compute() is called exactly once; subsequent calls return
   the cached value without invoking compute() again.
2. After TTL expiry: compute() is called again (cache miss).
3. Shared-state failure: if get_shared_state() raises or the backend is
   broken, _cached_json falls back to calling compute() directly with no
   exception propagated.

The helper is a closure inside create_app() so we test it indirectly via
HTTP calls to the app, using a call counter injected through a monkeypatched
DB method or directly through the MemoryBackend's fake-clock TTL.
"""

from __future__ import annotations

import os
import json
import time
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app_and_state(tmp_path, monkeypatch):
    """Create a TestClient backed by a fresh MemoryBackend shared state.

    Returns (client, shared_state_instance) so tests can inspect / corrupt
    the cache entries directly.
    """
    monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    os.makedirs(tmp_path / "reports", exist_ok=True)

    # _isolate_shared_state in conftest.py already sets EXONIT_SHARED_STATE_BACKEND=memory
    # and resets the singleton, so we get a fresh MemoryBackend here.
    from exonit.shared_state import get_shared_state, reset_shared_state
    reset_shared_state()

    from exonit.server import create_app
    app = create_app()
    client = TestClient(app)
    state = get_shared_state()
    yield client, state


# ---------------------------------------------------------------------------
# 1. Within TTL: compute runs only once
# ---------------------------------------------------------------------------


class TestCachedJsonHitRate:
    """Verify that repeated calls within the TTL hit the cache, not the DB."""

    def test_stats_compute_called_once_within_ttl(self, app_and_state, monkeypatch):
        """Within the TTL window /api/stats should compute only once.

        Strategy: patch ExoNitDB.get_stats to a counter; call the endpoint
        twice; verify the counter is 1.
        """
        client, state = app_and_state
        call_count = {"n": 0}

        from exonit import db as db_module
        original_get_stats = db_module.ExoNitDB.get_stats

        def counting_get_stats(self_db):
            call_count["n"] += 1
            return original_get_stats(self_db)

        monkeypatch.setattr(db_module.ExoNitDB, "get_stats", counting_get_stats)

        # First call — cache miss, DB queried.
        r1 = client.get("/api/stats")
        assert r1.status_code == 200
        assert call_count["n"] == 1

        # Second call within TTL — should be a cache hit.
        r2 = client.get("/api/stats")
        assert r2.status_code == 200
        assert call_count["n"] == 1, (
            "compute() should not have been called again within TTL"
        )

        # Response schema must be identical on both calls.
        assert r1.json() == r2.json()

    def test_backfill_fpp_hit_within_ttl(self, app_and_state):
        """Two consecutive /api/backfill/fpp calls within TTL return same payload."""
        client, state = app_and_state
        r1 = client.get("/api/backfill/fpp")
        r2 = client.get("/api/backfill/fpp")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json() == r2.json()

    def test_backfill_nfpp_hit_within_ttl(self, app_and_state):
        """Two consecutive /api/backfill/nfpp calls within TTL return same payload."""
        client, state = app_and_state
        r1 = client.get("/api/backfill/nfpp")
        r2 = client.get("/api/backfill/nfpp")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json() == r2.json()

    def test_worker_list_hit_within_ttl(self, app_and_state, monkeypatch):
        """Within TTL /api/worker/list should not re-query the DB task stats."""
        client, state = app_and_state
        call_count = {"n": 0}

        from exonit import db as db_module
        original = db_module.ExoNitDB.get_task_stats

        def counting(self_db):
            call_count["n"] += 1
            return original(self_db)

        monkeypatch.setattr(db_module.ExoNitDB, "get_task_stats", counting)

        r1 = client.get("/api/worker/list")
        n_after_first = call_count["n"]
        r2 = client.get("/api/worker/list")
        assert r2.status_code == 200
        assert call_count["n"] == n_after_first, (
            "get_task_stats() should not be called again within TTL"
        )

    def test_status_hit_within_ttl(self, app_and_state, monkeypatch):
        """Within TTL /api/status should not re-query the DB."""
        client, state = app_and_state
        call_count = {"n": 0}

        from exonit import db as db_module
        original = db_module.ExoNitDB.get_task_stats

        def counting(self_db):
            call_count["n"] += 1
            return original(self_db)

        monkeypatch.setattr(db_module.ExoNitDB, "get_task_stats", counting)

        r1 = client.get("/api/status")
        n_after_first = call_count["n"]
        r2 = client.get("/api/status")
        assert r2.status_code == 200
        assert call_count["n"] == n_after_first, (
            "get_task_stats() should not be called again within TTL"
        )

    def test_health_work_hit_within_ttl(self, app_and_state):
        """Two consecutive /api/health/work calls return same payload within TTL."""
        client, state = app_and_state
        r1 = client.get("/api/health/work")
        r2 = client.get("/api/health/work")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json() == r2.json()


# ---------------------------------------------------------------------------
# 2. After TTL expiry: compute runs again
# ---------------------------------------------------------------------------


class TestCachedJsonTTLExpiry:
    """Verify that the cache is bypassed once the TTL has elapsed.

    We write a stale value directly into the MemoryBackend with a tiny TTL,
    wait for it to expire (using the fake clock already available on
    MemoryBackend, or by deleting the key), then confirm a fresh call
    recomputes.
    """

    def test_cache_miss_after_manual_deletion(self, app_and_state, monkeypatch):
        """After manually deleting the cache key the endpoint recomputes."""
        client, state = app_and_state
        call_count = {"n": 0}

        from exonit import db as db_module
        original = db_module.ExoNitDB.get_stats

        def counting(self_db):
            call_count["n"] += 1
            return original(self_db)

        monkeypatch.setattr(db_module.ExoNitDB, "get_stats", counting)

        # Prime the cache.
        client.get("/api/stats")
        assert call_count["n"] == 1

        # Simulate TTL expiry by removing the key from the backend.
        state.delete("cache:stats")

        # Next call must recompute.
        client.get("/api/stats")
        assert call_count["n"] == 2, (
            "compute() should have been called again after cache expiry"
        )

    def test_cache_miss_after_ttl_on_memory_backend(self, monkeypatch, tmp_path):
        """MemoryBackend with injected fake clock expires the cache key correctly.

        This tests the TTL mechanism itself at the shared-state layer without
        going through the full HTTP stack.
        """
        from exonit.shared_state import MemoryBackend

        class FakeClock:
            def __init__(self):
                self.now = 0.0
            def __call__(self):
                return self.now

        clock = FakeClock()
        backend = MemoryBackend(clock=clock)

        # Store a cache entry with TTL=5s.
        backend.set("cache:test_key", json.dumps({"v": 42}), ttl=5.0)

        # Before expiry: value is readable.
        clock.now = 4.9
        raw = backend.get("cache:test_key")
        assert raw is not None
        assert json.loads(raw)["v"] == 42

        # At expiry boundary: key is gone.
        clock.now = 5.0
        assert backend.get("cache:test_key") is None


# ---------------------------------------------------------------------------
# 2b. Stale-while-revalidate: a stale entry is served immediately and
#     refreshed in a background thread, single-flighted via a Redis lock.
# ---------------------------------------------------------------------------


class TestCachedJsonStaleWhileRevalidate:
    """_cached_json (2026-07-31 SWR rework) never makes a caller pay for a
    cold aggregate: a stale wrapped entry (``{"_swr_ts": ..., "payload": ...}``)
    is returned as-is, and a background thread refreshes the cache under a
    single-flight lock so only one of the N uvicorn processes recomputes.
    """

    def test_stale_entry_served_immediately_without_inline_compute(
        self, app_and_state, monkeypatch
    ):
        client, state = app_and_state
        call_count = {"n": 0}

        from exonit import db as db_module
        original_get_stats = db_module.ExoNitDB.get_stats

        def counting_get_stats(self_db):
            # Small delay so the background thread's increment can never race
            # ahead of the main thread's assertion right below the request —
            # the request/response round-trip itself takes far less than
            # this, so a 0 here is a real "no inline compute", not a fluke
            # of scheduling.
            time.sleep(0.3)
            call_count["n"] += 1
            return original_get_stats(self_db)

        monkeypatch.setattr(db_module.ExoNitDB, "get_stats", counting_get_stats)

        marker_payload = {
            "total_stars": -1,
            "total_candidates": -1,
            "strong_candidates": -1,
            "_marker": "stale-seeded-by-test",
        }
        # Stale (past /api/stats' 60 s freshness window) but inside the
        # max-staleness ceiling added 2026-08-01, which is where "serve stale,
        # refresh behind" still applies. Past the ceiling the contract is the
        # opposite — see test_past_max_stale_entry_is_recomputed_inline.
        state.set(
            "cache:stats",
            json.dumps({"_swr_ts": time.time() - 120.0, "payload": marker_payload}),
        )

        r = client.get("/api/stats")
        assert r.status_code == 200
        assert r.json() == marker_payload, (
            "a stale entry must be served as-is, not recomputed inline"
        )
        assert call_count["n"] == 0, (
            "compute() must NOT run inline on the request that finds a stale "
            "entry — only the background refresh thread may call it"
        )

        # The stale hit above should have kicked off a background refresh.
        # Poll the cache entry until it flips to the fresh payload (or time out).
        deadline = time.time() + 5.0
        refreshed = False
        while time.time() < deadline:
            raw = state.get("cache:stats")
            if raw is not None:
                entry = json.loads(raw)
                if entry.get("payload") != marker_payload:
                    refreshed = True
                    break
            time.sleep(0.05)

        assert refreshed, "background refresh did not update cache:stats in time"
        assert call_count["n"] == 1, "background refresh should call compute() exactly once"

        r2 = client.get("/api/stats")
        assert r2.status_code == 200
        assert r2.json() != marker_payload
        assert call_count["n"] == 1, "the second request should hit the now-fresh cache"

    def test_past_max_stale_entry_is_recomputed_inline(self, app_and_state):
        """An entry far past its freshness window must NOT be served.

        Regression, 2026-08-01: the persist TTL (24 h) was the only bound on
        how stale a served payload could be, so the first request after a quiet
        stretch got hours-old numbers presented as current, and only the
        request AFTER that saw reality. On /api/health/work that fed the
        discovery progress ring a ~50-minute-old FPP reading dated "now", which
        is what made the dashboard's rate gauge read ~7,000 TIC/min.
        """
        client, state = app_and_state

        marker_payload = {
            "total_stars": -1,
            "total_candidates": -1,
            "strong_candidates": -1,
            "_marker": "ancient-seeded-by-test",
        }
        # /api/stats has a 60 s window => a 600 s ceiling. 100_000 s is far
        # past it; before this fix it was served verbatim.
        state.set(
            "cache:stats",
            json.dumps({"_swr_ts": time.time() - 100_000.0, "payload": marker_payload}),
        )

        r = client.get("/api/stats")
        assert r.status_code == 200
        assert r.json() != marker_payload, (
            "an entry past the max-staleness ceiling must be recomputed "
            "inline, not served"
        )

        # And the recomputed value must have replaced the ancient entry, so the
        # next caller hits a fresh cache rather than paying compute() again.
        entry = json.loads(state.get("cache:stats"))
        assert entry["payload"] != marker_payload
        assert time.time() - entry["_swr_ts"] < 60.0

    def test_refresh_lock_prevents_duplicate_refresh(self, app_and_state, monkeypatch):
        client, state = app_and_state
        call_count = {"n": 0}

        from exonit import db as db_module
        original_get_stats = db_module.ExoNitDB.get_stats

        def counting_get_stats(self_db):
            call_count["n"] += 1
            return original_get_stats(self_db)

        monkeypatch.setattr(db_module.ExoNitDB, "get_stats", counting_get_stats)

        marker_payload = {
            "total_stars": -2,
            "total_candidates": -2,
            "strong_candidates": -2,
            "_marker": "stale-seeded-by-test-2",
        }
        state.set(
            "cache:stats",
            json.dumps({"_swr_ts": time.time() - 100_000.0, "payload": marker_payload}),
        )
        # Simulate another uvicorn process already holding the single-flight lock.
        state.set("cache:refresh:stats", "1", ttl=120.0)

        r = client.get("/api/stats")
        assert r.status_code == 200
        assert r.json() == marker_payload

        time.sleep(0.5)
        assert call_count["n"] == 0, (
            "with the lock already held, this process must not also refresh"
        )

    def test_legacy_unwrapped_entry_served_as_fresh(self, app_and_state, monkeypatch):
        """A payload written by the pre-SWR code (bare dict, no _swr_ts
        wrapper) must be served as-is and treated as fresh."""
        client, state = app_and_state
        call_count = {"n": 0}

        from exonit import db as db_module
        original_get_stats = db_module.ExoNitDB.get_stats

        def counting_get_stats(self_db):
            call_count["n"] += 1
            return original_get_stats(self_db)

        monkeypatch.setattr(db_module.ExoNitDB, "get_stats", counting_get_stats)

        legacy_payload = {
            "total_stars": -3,
            "total_candidates": -3,
            "strong_candidates": -3,
        }
        state.set("cache:stats", json.dumps(legacy_payload), ttl=60.0)

        r = client.get("/api/stats")
        assert r.status_code == 200
        assert r.json() == legacy_payload
        assert call_count["n"] == 0, (
            "a legacy unwrapped entry is fresh — no compute() call expected"
        )


# ---------------------------------------------------------------------------
# 3. Shared-state failure: fallback to compute with no exception
# ---------------------------------------------------------------------------


class TestCachedJsonFallback:
    """Verify that a broken shared_state does not break an endpoint."""

    def test_broken_shared_state_does_not_raise(self, app_and_state, monkeypatch):
        """If get_shared_state().get() raises, the endpoint still returns 200.

        We patch the MemoryBackend.get method to always raise RuntimeError
        and confirm that /api/stats still returns a valid response.
        """
        client, state = app_and_state

        original_get = state.get

        def exploding_get(key):
            raise RuntimeError("simulated Redis failure")

        # Patch 'get' on the existing singleton instance.
        monkeypatch.setattr(state, "get", exploding_get)

        # Also patch 'set' so writing the result doesn't fail silently and
        # hide a bug where the result is never returned.
        original_set = state.set

        def exploding_set(key, value, **kw):
            raise RuntimeError("simulated Redis write failure")

        monkeypatch.setattr(state, "set", exploding_set)

        r = client.get("/api/stats")
        assert r.status_code == 200, (
            "Endpoint must return 200 even when shared_state raises"
        )
        data = r.json()
        assert "total_stars" in data
        assert "total_candidates" in data

    def test_broken_shared_state_returns_correct_data(self, app_and_state, monkeypatch):
        """With a broken cache the endpoint still returns correct (live) data."""
        client, state = app_and_state

        monkeypatch.setattr(state, "get", lambda k: (_ for _ in ()).throw(RuntimeError("boom")))
        monkeypatch.setattr(state, "set", lambda k, v, **kw: (_ for _ in ()).throw(RuntimeError("boom")))

        r = client.get("/api/backfill/nfpp")
        assert r.status_code == 200
        data = r.json()
        # Schema check: these keys must always be present.
        for key in ("total_candidates", "with_nfpp", "pending_candidates",
                    "pending_tics", "validated", "candidate", "likely_fp", "failed"):
            assert key in data, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 4. Response schema unchanged (no fields added / removed by caching)
# ---------------------------------------------------------------------------


class TestResponseSchema:
    """Cached responses must be byte-identical in schema to the live response."""

    def test_stats_schema(self, app_and_state):
        r = client_get(app_and_state, "/api/stats")
        data = r.json()
        for key in ("total_stars", "total_candidates", "strong_candidates"):
            assert key in data, f"Missing key in /api/stats: {key}"

    def test_backfill_fpp_schema(self, app_and_state):
        r = client_get(app_and_state, "/api/backfill/fpp")
        data = r.json()
        for key in ("baseline", "pending", "done", "elapsed_seconds",
                    "rate_per_min", "active_workers_assigned"):
            assert key in data, f"Missing key in /api/backfill/fpp: {key}"

    def test_backfill_nfpp_schema(self, app_and_state):
        r = client_get(app_and_state, "/api/backfill/nfpp")
        data = r.json()
        for key in ("total_candidates", "with_nfpp", "pending_candidates",
                    "pending_tics", "validated", "candidate", "likely_fp", "failed"):
            assert key in data, f"Missing key in /api/backfill/nfpp: {key}"

    def test_worker_list_schema(self, app_and_state):
        r = client_get(app_and_state, "/api/worker/list")
        data = r.json()
        assert "workers" in data
        assert "tasks" in data

    def test_status_schema(self, app_and_state):
        r = client_get(app_and_state, "/api/status")
        data = r.json()
        assert "running" in data
        assert "progress" in data
        for key in ("totalStars", "currentStar", "phase"):
            assert key in data["progress"], f"Missing key in /api/status progress: {key}"

    def test_health_work_schema(self, app_and_state):
        r = client_get(app_and_state, "/api/health/work")
        data = r.json()
        for section in ("primary_tasks", "fpp_backfill", "exominer_backfill",
                        "lora_explanation", "workers", "disk", "status",
                        "warnings", "critical"):
            assert section in data, f"Missing section in /api/health/work: {section}"


# ---------------------------------------------------------------------------
# 5. work_streams stats in /api/health/work (Fase 1 streams, I2 final-review
#    fix): per-stage queue depths under the flag; byte-for-byte unchanged
#    payload when the flag is off.
# ---------------------------------------------------------------------------


def _build_streams_app(tmp_path, monkeypatch):
    monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    os.makedirs(tmp_path / "reports", exist_ok=True)
    monkeypatch.setenv("EXONIT_WORK_STREAMS", "1")

    from exonit.shared_state import reset_shared_state
    reset_shared_state()

    from exonit.server import create_app
    app = create_app()
    return app


class TestHealthWorkStreamsStats:
    """EXONIT_WORK_STREAMS=1 — the legacy currently_assigned counters read a
    claims hash the streams path never writes, so with the flag on they
    always report 0 inflight and the DLQ is invisible outside redis-cli.
    /api/health/work must ADD a work_streams block (and a
    currently_assigned_streams fallback key alongside the untouched legacy
    key) instead of repurposing anything the exporter already parses."""

    def test_flag_on_includes_per_stage_stream_stats(self, tmp_path, monkeypatch):
        from exonit import work_streams

        fake_streams = {
            name: work_streams.MemoryWorkStream(name)
            for name in ("fpp", "exominer", "primary")
        }
        fake_streams["fpp"].enqueue("1", {"candidate_id": 1})
        fake_streams["fpp"].claim("worker-a")        # -> 1 pending, 0 queued
        fake_streams["exominer"].enqueue("2", {"candidate_id": 2})
        fake_streams["exominer"].claim("worker-b")   # -> 1 pending, 0 queued
        fake_streams["primary"].enqueue("3", {"task_id": 3})  # -> 1 queued, unclaimed
        monkeypatch.setattr(work_streams, "get_stream",
                            lambda name: fake_streams[name])

        app = _build_streams_app(tmp_path, monkeypatch)
        with TestClient(app) as client:
            r = client.get("/api/health/work")
        assert r.status_code == 200
        data = r.json()

        assert "work_streams" in data
        for stage in ("fpp", "exominer", "primary"):
            for key in ("queued", "pending", "dlq"):
                assert key in data["work_streams"][stage], (
                    f"missing {key} in work_streams.{stage}")
        assert data["work_streams"]["fpp"]["pending"] == 1
        assert data["work_streams"]["exominer"]["pending"] == 1
        assert data["work_streams"]["primary"]["queued"] == 1

        # Existing keys untouched; new fallback key added alongside.
        assert data["fpp_backfill"]["currently_assigned"] == 0
        assert data["fpp_backfill"]["currently_assigned_streams"] == 1
        assert data["exominer_backfill"]["currently_assigned"] == 0
        assert data["exominer_backfill"]["currently_assigned_streams"] == 1

    def test_flag_on_stats_failure_does_not_500(self, tmp_path, monkeypatch):
        """A stats() blow-up (e.g. Redis hiccup) must be swallowed, not
        500 the health endpoint."""
        from exonit import work_streams

        class _BrokenStream:
            def stats(self):
                raise RuntimeError("redis down")

        monkeypatch.setattr(work_streams, "get_stream",
                            lambda name: _BrokenStream())

        app = _build_streams_app(tmp_path, monkeypatch)
        with TestClient(app) as client:
            r = client.get("/api/health/work")
        assert r.status_code == 200
        assert "work_streams" not in r.json()

    def test_flag_off_payload_unchanged_byte_for_byte(self, app_and_state):
        """EXONIT_WORK_STREAMS unset — the flag adds NOTHING to the payload.

        The key sets below are the endpoint's contract with the Prometheus
        exporter and the UI. exominer_backfill grew 'actionable'/
        'unrecoverable' on 2026-07-14 — mirroring the shape fpp_backfill has
        always had — because 'pending' includes candidates the serve loop
        deliberately holds back (uncached FITS metadata), which can therefore
        never be served and made the ExoMiner phase-exit condition unreachable.
        Both legacy keys ('pending', 'currently_assigned', ...) stay exactly
        where they were: this is an addition, never a repurposing.
        """
        client, state = app_and_state
        r = client.get("/api/health/work")
        assert r.status_code == 200
        data = r.json()
        assert "work_streams" not in data
        assert set(data["fpp_backfill"].keys()) == {
            "actionable", "unrecoverable", "currently_assigned"}
        assert set(data["exominer_backfill"].keys()) == {
            "pending", "actionable", "unrecoverable",
            "currently_assigned", "baseline", "done"}


def client_get(app_and_state, path):
    """Helper: call client.get and assert 200."""
    client, state = app_and_state
    r = client.get(path)
    assert r.status_code == 200
    return r


class TestHealthWorkCacheIsScoped:
    """The run-scoped payload must not share a cache entry with the global one.

    Regression guard for the 2026-07-25 dashboard bug: /api/health/work is
    polled BOTH by the dashboard (with ?run_id=<live run>, wanting that run's
    counters) and by the Prometheus exporter / any other client (no run_id,
    wanting the global counters). Both were writing to the single
    "health_work" cache key, so for up to 10 s each audience could be served
    the other's payload — the run's progress bar jumped backwards or froze.
    The docstring on the endpoint already claimed a scoped request "can never
    be served from — or written into—" the shared entry; only the key was
    missing the scope.
    """

    def test_scoped_and_global_do_not_share_a_cache_entry(self, app_and_state,
                                                          monkeypatch):
        client, state = app_and_state

        # Make the run_id look materialised so the endpoint takes the scoped
        # branch, and make the scoped SQL distinguishable from the global one.
        from exonit import db as db_module
        monkeypatch.setattr(db_module.ExoNitDB, "count_discovery_run_tics",
                            lambda self_db, rid: 7)

        r_scoped = client.get("/api/health/work?run_id=run-A")
        r_global = client.get("/api/health/work")
        assert r_scoped.status_code == 200 and r_global.status_code == 200

        # The scope block is the endpoint's own statement of which payload it
        # served — it must survive the cache round-trip intact for BOTH.
        assert r_scoped.json()["scope"] == {"run_id": "run-A", "scoped": True, "kind": "run_tics"}
        assert r_global.json()["scope"] == {"run_id": None, "scoped": False, "kind": None}

        # And the reverse order (global cached first) must not poison the
        # scoped read either.
        r_global2 = client.get("/api/health/work")
        r_scoped2 = client.get("/api/health/work?run_id=run-A")
        assert r_global2.json()["scope"] == {"run_id": None, "scoped": False, "kind": None}
        assert r_scoped2.json()["scope"] == {"run_id": "run-A", "scoped": True, "kind": "run_tics"}

    def test_two_different_runs_do_not_share_a_cache_entry(self, app_and_state,
                                                           monkeypatch):
        """Chained lots run back to back — run B must not read run A's entry."""
        client, state = app_and_state
        from exonit import db as db_module
        monkeypatch.setattr(db_module.ExoNitDB, "count_discovery_run_tics",
                            lambda self_db, rid: 7)

        a = client.get("/api/health/work?run_id=run-A").json()
        b = client.get("/api/health/work?run_id=run-B").json()
        assert a["scope"]["run_id"] == "run-A"
        assert b["scope"]["run_id"] == "run-B"
