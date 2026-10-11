"""Tests for POST /api/admin/recycle-workers (exonit/routes/admin.py).

See docs/superpowers/specs/2026-07-17-worker-recycle-postrun.md. The
recycle LOOP itself (single-process no-arm, generation/slot semantics) is
covered by tests/test_worker_recycle.py; this file only covers the HTTP
surface: auth + the generation increment.

conftest.py's autouse fixture forces MemoryBackend and deletes
EXONIT_SERVER_WORKERS, so create_app() here always takes the
single-process path — the background recycle loop is never armed for these
tests (verified separately in test_worker_recycle.py::TestInstall), so the
only thing under test is the endpoint handler itself.
"""

from __future__ import annotations

import pytest

AUTH = {"Authorization": "Bearer exonit-default-token"}


@pytest.fixture
def recycle_client(tmp_path):
    import os
    os.environ["EXONIT_DB_PATH"] = str(tmp_path / "test.db")
    os.environ["EXONIT_REPORTS_DIR"] = str(tmp_path / "reports")
    os.makedirs(tmp_path / "reports", exist_ok=True)

    from exonit.server import create_app
    app = create_app()

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        yield client

    del os.environ["EXONIT_DB_PATH"]
    del os.environ["EXONIT_REPORTS_DIR"]


class TestRecycleWorkersEndpoint:
    def test_missing_token_returns_401(self, recycle_client):
        resp = recycle_client.post("/api/admin/recycle-workers")
        assert resp.status_code == 401

    def test_wrong_token_returns_401(self, recycle_client):
        resp = recycle_client.post(
            "/api/admin/recycle-workers",
            headers={"Authorization": "Bearer wrong-token"})
        assert resp.status_code == 401

    def test_valid_token_bumps_generation_from_zero(self, recycle_client):
        resp = recycle_client.post("/api/admin/recycle-workers", headers=AUTH)
        assert resp.status_code == 200
        assert resp.json() == {"generation": 1}

    def test_repeated_calls_keep_incrementing(self, recycle_client):
        """Not idempotent in the strict HTTP sense (each call bumps the
        counter by one) — "idempotent" per the spec means safe/side-effect-
        free to invoke repeatedly by hand, not that the response is
        unchanged. Verifies the counter never gets stuck or double-bumps."""
        r1 = recycle_client.post("/api/admin/recycle-workers", headers=AUTH)
        r2 = recycle_client.post("/api/admin/recycle-workers", headers=AUTH)
        r3 = recycle_client.post("/api/admin/recycle-workers", headers=AUTH)
        assert [r1.json()["generation"], r2.json()["generation"],
                r3.json()["generation"]] == [1, 2, 3]

    def test_generation_visible_via_shared_state(self, recycle_client):
        from exonit import worker_recycle as wr
        from exonit.shared_state import get_shared_state
        recycle_client.post("/api/admin/recycle-workers", headers=AUTH)
        assert get_shared_state().get_int(wr.RECYCLE_GENERATION_KEY) == 1
