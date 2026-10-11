"""Tests for /api/backfill/fpp rate_per_min accuracy fix.

Before the fix, api_backfill_fpp called set_if_absent on the started_at key,
which planted the clock on the first UI poll — potentially hours before any
FPP work was done — diluting rate_per_min to near-zero.

After the fix, the GET only reads the key (never seeds it), so:
- When no done has been recorded, elapsed_seconds == 0.0 and
  rate_per_min == 0.0, and the started_at key is absent.
- Once the increment path seeds started_at, the GET uses the real start
  time and reports an accurate rate.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixture: isolated TestClient with temp DB (same pattern as test_discovery_api)
# ---------------------------------------------------------------------------

@pytest.fixture
def fpp_client(tmp_path, monkeypatch):
    """FastAPI TestClient with isolated DB and temp reports dir."""
    db_file = tmp_path / "test.db"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)

    monkeypatch.setenv("EXONIT_DB_PATH", str(db_file))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(reports_dir))
    monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE",
                       str(tmp_path / "pipeline_state.json"))

    from exonit.server import create_app
    app = create_app()

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        yield client, tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFppBackfillRate:
    """GET /api/backfill/fpp — rate_per_min semantics."""

    def test_no_done_returns_zero_rate_and_elapsed(self, fpp_client):
        """Before any FPP work is done, rate_per_min and elapsed_seconds
        must both be 0.0.  The GET itself must NOT seed started_at."""
        client, _ = fpp_client

        resp = client.get("/api/backfill/fpp")
        assert resp.status_code == 200
        data = resp.json()
        assert data["rate_per_min"] == 0.0
        assert data["elapsed_seconds"] == 0.0
        assert data["done"] == 0

    def test_get_does_not_seed_started_at(self, fpp_client):
        """Multiple GETs with done=0 must NOT create the started_at key.

        If they did, the next GET after the first FPP write-back would
        compute elapsed from the very first poll instead of from the
        first write-back, diluting rate_per_min.
        """
        client, _ = fpp_client

        # Call the endpoint several times simulating repeated UI polls
        for _ in range(3):
            client.get("/api/backfill/fpp")

        from exonit.shared_state import get_shared_state
        from exonit import server as srv
        store = get_shared_state()
        assert store.get(srv._FPP_BACKFILL_STARTED_AT_KEY) is None, (
            "started_at key must not exist before any FPP write-back"
        )

    def test_after_increment_rate_is_accurate(self, fpp_client):
        """After simulating a FPP write-back (increment path), elapsed and
        rate must reflect the time since first write-back, not since server
        start or first GET.

        We inject a backdated started_at (simulating the increment path
        having run 60 seconds ago) and set done=10, then verify the
        GET returns rate == 10.0 (10 done / 60 s * 60 s/min).
        """
        client, _ = fpp_client

        from exonit.shared_state import get_shared_state
        from exonit import server as srv
        store = get_shared_state()

        # Simulate: 60 s ago the first FPP result was written back
        simulated_start = time.time() - 60.0
        store.set_if_absent(srv._FPP_BACKFILL_STARTED_AT_KEY,
                            str(simulated_start))
        # Simulate: 10 FPP results have been completed
        store.set_int(srv._FPP_BACKFILL_DONE_KEY, 10)

        resp = client.get("/api/backfill/fpp")
        assert resp.status_code == 200
        data = resp.json()

        assert data["done"] == 10
        # elapsed must be close to 60 s (allow ±2 s for test execution time)
        assert 58.0 <= data["elapsed_seconds"] <= 62.0, (
            f"elapsed_seconds={data['elapsed_seconds']} expected ~60"
        )
        # rate = 10 done / ~60 s * 60 s/min ≈ 10.0 /min
        assert 9.0 <= data["rate_per_min"] <= 11.0, (
            f"rate_per_min={data['rate_per_min']} expected ~10.0"
        )

    def test_zero_elapsed_guard(self, fpp_client):
        """If elapsed rounds to 0.0 (started_at is present but just set),
        rate_per_min must be 0.0 (no division by zero)."""
        client, _ = fpp_client

        from exonit.shared_state import get_shared_state
        from exonit import server as srv
        store = get_shared_state()

        # Plant started_at as right now — elapsed will be near 0
        store.set_if_absent(srv._FPP_BACKFILL_STARTED_AT_KEY,
                            str(time.time()))
        store.set_int(srv._FPP_BACKFILL_DONE_KEY, 5)

        resp = client.get("/api/backfill/fpp")
        assert resp.status_code == 200
        data = resp.json()
        # Should not raise; rate might be very high or 0 depending on
        # sub-ms elapsed, but must be a finite number.
        assert isinstance(data["rate_per_min"], (int, float))
        assert data["rate_per_min"] >= 0.0
