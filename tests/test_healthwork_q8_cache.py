"""Q8 (perf review 2026-09-26): /api/health/work's unscoped, cacheable path
routes through db.get_task_stats_cached() instead of a raw
SELECT status, COUNT(*) FROM tasks GROUP BY status.

docs/reviews/2026-09-26-enqueue-path-performance-review.md measured this
query at 603 calls/7h, ~2.2s each (a parallel seq scan of the whole
34.5M-row tasks table) for the display-only (UI dashboard / health polling)
caller. The orchestrator's own phase-exit polls (scoped by run_id, or
unscoped-but-nocache=1 when a run has no materialised TIC set — see
scripts/discovery_orchestrator.py's _health_work_url) must keep reading
ground truth, per get_task_stats_cached()'s own documented caveat.
"""

import os
from unittest.mock import patch

from fastapi.testclient import TestClient


def _build_client(tmp_path, monkeypatch):
    monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "hw.db"))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    os.makedirs(tmp_path / "reports", exist_ok=True)

    from exonit.server import create_app
    return TestClient(create_app())


class TestHealthWorkQ8CachedPath:
    def test_unscoped_no_nocache_uses_cached_task_stats(self, tmp_path, monkeypatch):
        """The display path (no run_id, no nocache) must call
        get_task_stats_cached(), not the raw GROUP BY."""
        client = _build_client(tmp_path, monkeypatch)

        from exonit.db import ExoNitDB
        with patch.object(
            ExoNitDB, "get_task_stats_cached",
            return_value={"pending": 3, "completed": 7},
        ) as m_cached:
            resp = client.get("/api/health/work")

        assert resp.status_code == 200
        m_cached.assert_called_once()
        body = resp.json()
        assert body["primary_tasks"]["pending"] == 3
        assert body["primary_tasks"]["completed"] == 7

    def test_unscoped_with_nocache_bypasses_cached_task_stats(self, tmp_path, monkeypatch):
        """nocache=1 (the orchestrator's own ground-truth poll, used when a
        run has no materialised TIC set) must NOT go through
        get_task_stats_cached() -- it needs a fresh count every time."""
        client = _build_client(tmp_path, monkeypatch)

        from exonit.db import ExoNitDB
        with patch.object(ExoNitDB, "get_task_stats_cached") as m_cached:
            resp = client.get("/api/health/work?nocache=1")

        assert resp.status_code == 200
        m_cached.assert_not_called()

    def test_scoped_bypasses_cached_task_stats_even_without_nocache(
        self, tmp_path, monkeypatch
    ):
        """A scoped (run_id) request -- the orchestrator's normal phase-exit
        poll -- must always use the fresh per-run JOIN query, never the
        global cached counter, regardless of the nocache flag."""
        client = _build_client(tmp_path, monkeypatch)

        from exonit.db import ExoNitDB
        with patch.object(ExoNitDB, "count_discovery_run_tics",
                          return_value=1), \
             patch.object(ExoNitDB, "get_task_stats_cached") as m_cached:
            resp = client.get("/api/health/work?run_id=some-run")

        assert resp.status_code == 200
        m_cached.assert_not_called()

    def test_scoped_with_nocache_bypasses_cached_task_stats(self, tmp_path, monkeypatch):
        client = _build_client(tmp_path, monkeypatch)

        from exonit.db import ExoNitDB
        with patch.object(ExoNitDB, "count_discovery_run_tics",
                          return_value=1), \
             patch.object(ExoNitDB, "get_task_stats_cached") as m_cached:
            resp = client.get("/api/health/work?run_id=some-run&nocache=1")

        assert resp.status_code == 200
        m_cached.assert_not_called()

    def test_cached_task_stats_shape_matches_raw_query_shape(self, tmp_path, monkeypatch):
        """get_task_stats_cached()'s {status: count} dict must feed
        primary_tasks the same way the raw GROUP BY result did -- a real
        end-to-end check (no mocking of get_task_stats_cached itself),
        exercising its real (uncached-fallback, since shared_state's
        MemoryBackend is fresh per test) code path."""
        client = _build_client(tmp_path, monkeypatch)

        resp = client.get("/api/health/work")
        assert resp.status_code == 200
        body = resp.json()
        assert "primary_tasks" in body
        assert isinstance(body["primary_tasks"], dict)
        # ExoNitDB imported at module level would be unused here -- this
        # test deliberately exercises the real (unmocked) get_task_stats_cached
        # against a fresh (empty) shared_state, i.e. its own inline-compute
        # fallback path.
