"""Tests for the extended /api/discovery/sectors endpoint (Task 4).

Verifies that each sector entry carries ``state``, ``gaps_processable``,
``no_data``, ``pct_complete`` and that the top-level ``active_sector`` is
derived from the sector whose ``state=='active'`` (with fallback to the
registry value).

Also covers Task 10: ``POST /api/discovery/sector/{n}/autofill`` (start/stop),
idempotency guard, sectors response autofill field, and startup resume.

Pattern: SQLite-backed TestClient with mocked ``sector_gap_stats`` and
``get_sector_state`` (the DB-heavy functions are independently tested in
``test_discovery_gap_stats.py`` and ``test_discovery_sector_state.py``).
Follows the test conventions in ``test_discovery_api.py``.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Fixture: isolated TestClient with temp SQLite DB
# ---------------------------------------------------------------------------

@pytest.fixture
def sectors_client(tmp_path, monkeypatch):
    """FastAPI TestClient with isolated SQLite DB and temp pipeline state."""
    monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv(
        "EXONIT_PIPELINE_STATE_FILE",
        str(tmp_path / "pipeline_state.json"),
    )
    # Never use the real production database URL.
    monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)

    from exonit.server import create_app
    app = create_app()

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        yield client, tmp_path


# ---------------------------------------------------------------------------
# Helper: controlled return values for the expensive DB functions
# ---------------------------------------------------------------------------

_FAKE_GAP_STATS = {
    "work_list_size": 1000,
    "completed": 300,
    "no_data": 50,
    "gaps_processable": 650,
    "pct_complete": 0.35,
}


# ---------------------------------------------------------------------------
# Task 4 tests
# ---------------------------------------------------------------------------

class TestDiscoverySectorsExtended:
    """Endpoint must expose state + gap metrics + derived active_sector."""

    def test_sector_entry_has_state_field(self, sectors_client):
        """Each sector entry must include 'state'."""
        client, _ = sectors_client
        reg = {
            "active_sector": 1,
            "sectors": {
                "1": {
                    "enabled": True,
                    "total_tics": 1000,
                    "downloaded": 5,
                    "cursor": 3,
                    "canonical_list": "",
                }
            },
        }
        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "sector_gap_stats",
                          return_value=_FAKE_GAP_STATS), \
             patch.object(discovery_state, "get_sector_state",
                          return_value="active"):
            resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        s = resp.json()["sectors"][0]
        assert "state" in s, "sector entry must have 'state' field"
        assert s["state"] == "active"

    def test_sector_entry_has_gap_fields(self, sectors_client):
        """Each sector entry must include gaps_processable, no_data, pct_complete."""
        client, _ = sectors_client
        reg = {
            "active_sector": 1,
            "sectors": {
                "1": {
                    "enabled": True,
                    "total_tics": 1000,
                    "downloaded": 5,
                    "cursor": 3,
                    "canonical_list": "",
                }
            },
        }
        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "sector_gap_stats",
                          return_value=_FAKE_GAP_STATS), \
             patch.object(discovery_state, "get_sector_state",
                          return_value="coming_soon"):
            resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        s = resp.json()["sectors"][0]
        assert "gaps_processable" in s, "sector entry must have 'gaps_processable'"
        assert "no_data" in s, "sector entry must have 'no_data'"
        assert "pct_complete" in s, "sector entry must have 'pct_complete'"
        assert s["gaps_processable"] == 650
        assert s["no_data"] == 50
        assert abs(s["pct_complete"] - 0.35) < 1e-6

    def test_active_sector_derived_from_state(self, sectors_client):
        """active_sector must equal the sector whose state=='active',
        overriding the registry value."""
        client, _ = sectors_client
        reg = {
            "active_sector": 1,   # registry says 1
            "sectors": {
                "1": {
                    "enabled": False,
                    "total_tics": 500,
                    "downloaded": 500,
                    "cursor": 500,
                    "canonical_list": "",
                },
                "23": {
                    "enabled": True,
                    "total_tics": 500000,
                    "downloaded": 10000,
                    "cursor": 1000,
                    "canonical_list": "",
                },
            },
        }

        def _state(sector, *, db_path):
            return "active" if sector == 23 else "complete"

        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "sector_gap_stats",
                          return_value=_FAKE_GAP_STATS), \
             patch.object(discovery_state, "get_sector_state",
                          side_effect=_state):
            resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        data = resp.json()
        # active_sector must be 23 (from state), NOT 1 (from registry)
        assert data["active_sector"] == 23, (
            f"expected active_sector=23 (state-derived), got {data['active_sector']}"
        )

    def test_active_sector_fallback_when_no_active_state(self, sectors_client):
        """When no sector has state=='active', active_sector falls back to
        the registry value (back-compat)."""
        client, _ = sectors_client
        reg = {
            "active_sector": 7,   # registry fallback
            "sectors": {
                "1": {
                    "enabled": False,
                    "total_tics": 100,
                    "downloaded": 100,
                    "cursor": 100,
                    "canonical_list": "",
                },
            },
        }
        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "sector_gap_stats",
                          return_value=_FAKE_GAP_STATS), \
             patch.object(discovery_state, "get_sector_state",
                          return_value="complete"):   # no sector is 'active'
            resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        data = resp.json()
        # No sector has state=='active' → fallback to registry value 7
        assert data["active_sector"] == 7, (
            f"expected registry fallback active_sector=7, got {data['active_sector']}"
        )

    def test_existing_fields_still_present(self, sectors_client):
        """The new fields must not displace the existing ones (back-compat)."""
        client, _ = sectors_client
        reg = {
            "active_sector": 1,
            "sectors": {
                "1": {
                    "enabled": True,
                    "total_tics": 999,
                    "downloaded": 10,
                    "cursor": 42,
                    "canonical_list": "",
                }
            },
        }
        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "sector_gap_stats",
                          return_value=_FAKE_GAP_STATS), \
             patch.object(discovery_state, "get_sector_state",
                          return_value="coming_soon"):
            resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        s = resp.json()["sectors"][0]
        # Pre-existing fields must still be present
        for key in ("number", "enabled", "total_tics", "downloaded",
                    "cursor", "examined", "pending"):
            assert key in s, f"existing field '{key}' must still be present"
        assert s["total_tics"] == 999
        assert s["cursor"] == 42
        assert s["examined"] == 42    # examined = cursor (unchanged)
        # pending now mirrors gaps_processable (the honest remaining-work
        # metric) whenever gap stats are available, NOT total_tics - examined
        # (phantom remaining fix — see progress/impl-pending-phantom-fix-2026-07-04.md).
        assert s["pending"] == _FAKE_GAP_STATS["gaps_processable"]

    def test_endpoint_robust_when_gap_stats_raises(self, sectors_client):
        """If sector_gap_stats / get_sector_state raise, endpoint must still
        return 200 with safe defaults (never a 500)."""
        client, _ = sectors_client
        reg = {
            "active_sector": 1,
            "sectors": {
                "1": {
                    "enabled": True,
                    "total_tics": 100,
                    "downloaded": 5,
                    "cursor": 0,
                    "canonical_list": "",
                }
            },
        }
        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "sector_gap_stats",
                          side_effect=RuntimeError("PG unavailable")), \
             patch.object(discovery_state, "get_sector_state",
                          side_effect=RuntimeError("PG unavailable")):
            resp = client.get("/api/discovery/sectors")

        # Must not be 500
        assert resp.status_code == 200
        s = resp.json()["sectors"][0]
        # Robust: gap metrics are null when the completed-set scan can't run (the
        # UI falls back to examined/total_tics rather than showing a fake 0).
        assert s.get("gaps_processable") is None
        assert s.get("no_data") is None
        # State table unreachable → no explicit row → derived from the registry's
        # enabled flag, which IS available: an enabled sector is 'paused', not a
        # misleading 'coming soon'.
        assert s.get("state") == "paused"

    def test_pending_uses_gaps_processable_when_available(self, sectors_client):
        """'pending' must equal gaps_processable, not total_tics - examined.

        Regression for the "sector 1 always showed more remaining than
        existed" bug: total_tics is the full MAST sector total while examined
        (=cursor) only advances through the downloaded work list — a smaller,
        different population — so total_tics - examined never reaches 0 at
        sector close. gaps_processable (work_list - completed - no_data, TOIs
        excluded) is the honest remaining-work metric and must win whenever
        gap stats are available.
        """
        client, _ = sectors_client
        reg = {
            "active_sector": 1,
            "sectors": {
                "1": {
                    "enabled": True,
                    "total_tics": 5000,   # full MAST sector total
                    "downloaded": 900,
                    "cursor": 100,        # examined = 100 -> phantom formula = 4900
                    "canonical_list": "",
                }
            },
        }
        gap = {"work_list_size": 1000, "completed": 950, "no_data": 50,
               "gaps_processable": 0, "pct_complete": 1.0}
        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "sector_gap_stats", return_value=gap), \
             patch.object(discovery_state, "get_sector_state",
                          return_value="active"):
            resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        s = resp.json()["sectors"][0]
        assert s["gaps_processable"] == 0
        # pending must reflect the honest 0-remaining metric, NOT the phantom
        # total_tics - examined = 5000 - 100 = 4900.
        assert s["pending"] == 0, (
            f"expected pending=gaps_processable=0 (sector actually complete), "
            f"got {s['pending']} (phantom formula would give "
            f"{max(0, 5000 - 100)})"
        )

    def test_pending_falls_back_to_total_minus_examined_when_gap_stats_absent(
        self, sectors_client
    ):
        """When gap stats haven't computed yet (g is None), 'pending' falls
        back to max(0, total_tics - examined) — the pre-fix formula, kept
        only as a transitional estimate until the background compute lands.
        """
        client, _ = sectors_client
        reg = {
            "active_sector": 1,
            "sectors": {
                "1": {
                    "enabled": True,
                    "total_tics": 1000,
                    "downloaded": 10,
                    "cursor": 42,
                    "canonical_list": "",
                }
            },
        }
        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "sector_gap_stats",
                          side_effect=RuntimeError("gap stats not computed yet")), \
             patch.object(discovery_state, "get_sector_state",
                          return_value="active"):
            resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        s = resp.json()["sectors"][0]
        assert s["gaps_processable"] is None
        assert s["pending"] == max(0, 1000 - 42)

    def test_pct_complete_is_relative_to_total_tics_while_mid_download(
            self, sectors_client):
        """pct_complete must use max(total_tics, work_list_size) — NOT
        work_list_size alone — while a sector's download is NOT yet
        verifiably complete (``downloaded < total_tics``).

        Regression for the original "sector 23 shows 54%" bug: work_list_size
        only counts downloaded TICs, so completed/work_list overstates
        progress while a sector is mid-download (an enabled/active sector can
        legitimately still be receiving bulk downloads while under active
        scan — discovery_run.py's need_download gate only requires the
        CURRENT LOT's TICs, not the whole sector). The 2026-10-05 fix (making
        pct_complete ignore total_tics for CLOSED, fully-downloaded sectors)
        must not regress this case for an open, mid-download sector: here
        downloaded(500) < total_tics(2000), so download_complete is False and
        the old safety-net formula still applies.
        """
        client, _ = sectors_client
        reg = {
            "active_sector": 1,
            "sectors": {
                "1": {
                    "enabled": True,
                    "total_tics": 2000,     # full sector
                    "downloaded": 500,      # mid-download: 500 < 2000
                    "cursor": 0,
                    "canonical_list": "",
                }
            },
        }
        gap = {"work_list_size": 500, "completed": 300, "no_data": 50,
               "gaps_processable": 150, "pct_complete": 0.70}
        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "sector_gap_stats", return_value=gap), \
             patch.object(discovery_state, "get_sector_state",
                          return_value="active"):
            resp = client.get("/api/discovery/sectors")
        s = resp.json()["sectors"][0]
        # (completed + no_data) / total_tics(2000) = (300 + 50) / 2000 = 0.175
        # NOT (300 + 50) / work_list_size(500) = 0.70 — the sector is still
        # mid-download, so work_list_size alone would overstate progress.
        assert abs(s["pct_complete"] - 0.175) < 1e-6

    def test_pct_complete_uses_work_list_when_download_complete(
            self, sectors_client):
        """Once a sector's download is verifiably complete (downloaded >=
        total_tics, both known and positive), pct_complete uses work_list_size
        alone — total_tics is dropped from the denominator.

        s5-like fixture (2026-10-05 UI audit): total_tics slightly larger
        than the real work list (a stale/inflated file count), downloaded has
        caught up to (or passed) it, and only 8 real gaps remain out of
        2,564,123 — the sector should read as ~100.0% complete, not the
        99.3% the old max(total_tics, work_list_size) formula gave.

        Uses an open (enabled/active) sector deliberately — download_complete
        is driven purely by the registry's own downloaded/total_tics numbers,
        independent of open/closed state (s5 itself is open), AND the LIVE
        gap-stats store computes inline on a cold cache (unlike the closed
        store, which never computes synchronously) so a single GET exercises
        the fix without a background-thread race.
        """
        client, _ = sectors_client
        reg = {
            "active_sector": 1,
            "sectors": {
                "1": {
                    "enabled": True,
                    "total_tics": 2564125,   # slightly > the real work list
                    "downloaded": 2564125,   # download_complete: downloaded >= total_tics
                    "cursor": 2564111,
                    "canonical_list": "",
                }
            },
        }
        gap = {"work_list_size": 2564123, "completed": 2564113, "no_data": 2,
               "gaps_processable": 8, "pct_complete": 0.9999968797}
        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "sector_gap_stats", return_value=gap), \
             patch.object(discovery_state, "get_sector_state",
                          return_value="active"):
            resp = client.get("/api/discovery/sectors")
        s = resp.json()["sectors"][0]
        # (2564113 + 2) / work_list_size(2564123) ≈ 0.999997 —
        # NOT / max(total_tics, work_list_size) = 2564125, which would give
        # the same ~0.999997 here too (total_tics barely exceeds work_list)
        # but for a MUCH larger gap between the two (see s5's real registry
        # drift) would read noticeably low; the point is the denominator
        # must be work_list_size, not total_tics, once download_complete.
        expected = (2564113 + 2) / 2564123
        assert abs(s["pct_complete"] - expected) < 1e-6
        assert abs(s["pct_complete"] - 0.999997) < 1e-5

    def test_pct_complete_never_exceeds_one_when_index_outgrows_total(
            self, sectors_client):
        """Regression for "sector 18 shows 100% with 340k TICs pending".

        The TGLC index (work_list_size) can grow beyond the registry's
        total_tics (s18: 5.55M indexed vs 5.07M target list). completed is
        counted over the index; here downloaded(1200) >= total_tics(1000) so
        download_complete is True and the denominator is work_list_size
        alone — which already dominates total_tics in this fixture, so the
        result is identical to the pre-2026-10-05 max()-based formula.
        """
        client, _ = sectors_client
        reg = {
            "active_sector": 1,
            "sectors": {
                "1": {
                    "enabled": True,
                    "total_tics": 1000,     # stale/smaller target list
                    "downloaded": 1200,
                    "cursor": 900,
                    "canonical_list": "",
                }
            },
        }
        gap = {"work_list_size": 1200, "completed": 1050, "no_data": 0,
               "gaps_processable": 150, "pct_complete": 0.875}
        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "sector_gap_stats", return_value=gap), \
             patch.object(discovery_state, "get_sector_state",
                          return_value="active"):
            resp = client.get("/api/discovery/sectors")
        s = resp.json()["sectors"][0]
        # 1050 / work_list_size(1200) = 0.875 — NOT 1050 / total_tics(1000) = 1.05
        assert abs(s["pct_complete"] - 0.875) < 1e-6
        assert s["pct_complete"] <= 1.0
        assert s["gaps_processable"] == 150

    def test_state_derived_when_no_explicit_row(self, sectors_client):
        """No explicit state row: enabled/started → 'paused'; untouched →
        'coming_soon'. An explicit row always wins (tested elsewhere)."""
        client, _ = sectors_client
        reg = {
            "active_sector": 1,
            "sectors": {
                "1": {"enabled": True, "total_tics": 100, "downloaded": 0,
                      "cursor": 0, "canonical_list": ""},     # enabled → paused
                "2": {"enabled": False, "total_tics": 100, "downloaded": 0,
                      "cursor": 0, "canonical_list": ""},     # untouched → coming_soon
                "3": {"enabled": False, "total_tics": 100, "downloaded": 0,
                      "cursor": 7, "canonical_list": ""},     # has progress → paused
            },
        }
        gap = {"work_list_size": 100, "completed": 0, "no_data": 0,
               "gaps_processable": 100, "pct_complete": 0.0}
        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "sector_gap_stats", return_value=gap), \
             patch.object(discovery_state, "get_sector_state",
                          return_value="coming_soon"):   # simulate no row
            resp = client.get("/api/discovery/sectors")
        by_num = {s["number"]: s for s in resp.json()["sectors"]}
        assert by_num[1]["state"] == "paused", "enabled sector must not be coming_soon"
        assert by_num[2]["state"] == "coming_soon", "untouched sector stays coming_soon"
        assert by_num[3]["state"] == "paused", "sector with progress must not be coming_soon"


# ---------------------------------------------------------------------------
# Task 10: POST /api/discovery/sector/{n}/autofill
# ---------------------------------------------------------------------------

def _make_autofill_sqlite_db(tmp_path: "Path") -> str:
    """Create a minimal SQLite DB with discovery_autofill + state tables."""
    db_path = str(tmp_path / "test.db")
    con = sqlite3.connect(db_path)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            tic_id TEXT,
            status TEXT DEFAULT 'pending',
            priority INTEGER DEFAULT 2,
            assigned_to TEXT,
            assigned_at TEXT,
            completed_at TEXT,
            processing_time REAL,
            gpu_used INTEGER DEFAULT 0,
            attempts INTEGER DEFAULT 0,
            last_error TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS discovery_autofill (
          sector       INTEGER PRIMARY KEY,
          status       TEXT,
          current_lot  TEXT,
          lots_done    INTEGER DEFAULT 0,
          anomalies    INTEGER DEFAULT 0,
          last_alert   TEXT,
          updated_at   TEXT
        );
        CREATE TABLE IF NOT EXISTS discovery_sector_state (
          sector     INTEGER PRIMARY KEY,
          state      TEXT NOT NULL DEFAULT 'coming_soon',
          detail     TEXT,
          updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS discovery_no_data (
          tic_id     TEXT NOT NULL,
          sector     INTEGER NOT NULL,
          reason     TEXT NOT NULL DEFAULT 'no_fits',
          checked_at TEXT NOT NULL DEFAULT (datetime('now')),
          PRIMARY KEY (tic_id, sector)
        );
    """)
    con.commit()
    con.close()
    return db_path


@pytest.fixture
def autofill_client(tmp_path, monkeypatch):
    """TestClient with SQLite DB that has discovery_autofill table.

    Creates the full schema so the autofill helpers work without PG.
    Mocks run_autofill so no real loop runs during tests.

    ``load_registry`` is monkeypatched to return an empty sectors dict for the
    ENTIRE fixture scope.  This prevents the startup-resume daemon thread from
    racing with test setup: the real sectors.json contains sector 23, and
    without the patch the thread might see a 'running' row after the test
    inserts it, dispatch an extra loop, and corrupt the dispatch-count assertions.
    Tests that need a specific registry value can use ``patch.object`` inside
    their body (which stacks on top of the monkeypatch correctly).
    """
    db_file = _make_autofill_sqlite_db(tmp_path)
    monkeypatch.setenv("EXONIT_DB_PATH", db_file)
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv(
        "EXONIT_PIPELINE_STATE_FILE",
        str(tmp_path / "pipeline_state.json"),
    )
    monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)

    # Prevent the startup-resume thread from loading the real sectors.json
    # (which has sector 23) while the test DB is being set up.
    from exonit import discovery_state as _ds
    monkeypatch.setattr(
        _ds, "load_registry",
        lambda path: {"active_sector": 1, "sectors": {}},
    )

    from exonit.server import create_app
    app = create_app()

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        yield client, tmp_path, db_file


class TestAutofillEndpoint:
    """POST /api/discovery/sector/{n}/autofill endpoint tests (Task 10)."""

    def test_post_autofill_start_returns_running(self, autofill_client):
        """POST start dispatches manager (mocked) and returns status 'running'."""
        client, tmp_path, db_file = autofill_client

        dispatched = []

        # Mock the dispatch function so no real run_autofill loop starts.
        # The patch target is the closure helper inside the server module.
        from exonit.discovery_autofill import run_autofill
        with patch("exonit.discovery_autofill.run_autofill",
                   side_effect=lambda sector, deps: dispatched.append(sector)):
            resp = client.post(
                "/api/discovery/sector/23/autofill",
                json={"action": "start"},
            )

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data.get("status") == "running", f"expected 'running', got: {data}"

        # DB row must have been set to 'running'
        from exonit.discovery_state import get_autofill
        af = get_autofill(23, db_path=db_file)
        assert af is not None, "autofill row must be created"
        assert af["status"] == "running"

    def test_post_autofill_stop_sets_stopped(self, autofill_client):
        """POST stop updates autofill row to 'stopped'."""
        client, tmp_path, db_file = autofill_client

        # First set the row to running via the DB helper
        from exonit.discovery_state import set_autofill, get_autofill
        set_autofill(23, db_path=db_file, status="running", lots_done=2)

        resp = client.post(
            "/api/discovery/sector/23/autofill",
            json={"action": "stop"},
        )

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data.get("status") == "stopped", f"expected 'stopped', got: {data}"

        # DB row must reflect the stopped status
        af = get_autofill(23, db_path=db_file)
        assert af is not None
        assert af["status"] == "stopped"

    def test_post_autofill_start_idempotent_when_already_running(self, autofill_client):
        """POST start with db_running=True and no live thread calls _dispatch_autofill_loop.

        This is the "crashed-loop" recovery case: run_autofill crashed before any
        DB write, so the PG advisory lock was released but the DB row kept
        status='running'.  The next POST start must attempt to restart the loop
        instead of silently returning "already running".

        _dispatch_autofill_loop is safe to call here: on PG it tries the advisory
        lock first (no-op if another process holds it); on SQLite (test mode) it
        proceeds and starts the thread.
        """
        client, tmp_path, db_file = autofill_client

        # Pre-seed the DB row as 'running' with no live thread (crashed-loop scenario)
        from exonit.discovery_state import set_autofill
        set_autofill(23, db_path=db_file, status="running", lots_done=5)

        dispatched = []
        dispatch_done = threading.Event()

        def _capture(sector, deps=None):
            dispatched.append(sector)
            dispatch_done.set()

        with patch("exonit.discovery_autofill.run_autofill", side_effect=_capture):
            resp = client.post(
                "/api/discovery/sector/23/autofill",
                json={"action": "start"},
            )
            # Wait up to 2 s for the background daemon thread to call run_autofill.
            dispatch_done.wait(timeout=2.0)

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data.get("status") == "running", f"expected 'running', got: {data}"

        # In the crashed-loop case (db_running=True, thread_alive=False),
        # _dispatch_autofill_loop MUST have been called.  In SQLite test mode
        # the advisory lock is skipped, so run_autofill is actually invoked.
        assert 23 in dispatched, (
            "run_autofill must be dispatched in the crashed-loop case "
            f"(db_running=True, thread_alive=False); dispatched={dispatched}"
        )

    def test_post_autofill_start_no_second_dispatch_when_thread_alive(self, autofill_client):
        """POST start when a thread is genuinely alive does NOT spawn a second loop.

        The in-process guard inside _dispatch_autofill_loop short-circuits if a
        thread for the sector already exists and is_alive().  This test simulates
        a live thread by blocking the first run_autofill call until after the
        second POST start has been issued.
        """
        client, tmp_path, db_file = autofill_client

        dispatched = []
        first_started = threading.Event()
        allow_finish = threading.Event()

        def _blocking_run(sector, deps=None):
            dispatched.append(sector)
            first_started.set()
            allow_finish.wait(timeout=5.0)

        with patch("exonit.discovery_autofill.run_autofill",
                   side_effect=_blocking_run):
            # First POST: DB has no row → goes to the normal "not running" path,
            # sets DB to 'running', dispatches _dispatch_autofill_loop.
            resp1 = client.post(
                "/api/discovery/sector/23/autofill",
                json={"action": "start"},
            )
            # Wait until the background thread is confirmed inside run_autofill
            # so is_alive() returns True for the registered thread.
            first_started.wait(timeout=2.0)

            # Second POST: DB='running' AND thread is alive in this process →
            # must return current status WITHOUT dispatching a second loop.
            resp2 = client.post(
                "/api/discovery/sector/23/autofill",
                json={"action": "start"},
            )

            # Release the blocking thread so the TestClient can shut down cleanly.
            allow_finish.set()

        assert resp1.status_code == 200, resp1.text
        assert resp2.status_code == 200, resp2.text
        assert resp2.json().get("status") == "running"

        # Only one run_autofill invocation must have occurred across both POSTs.
        assert dispatched.count(23) == 1, (
            "run_autofill must be dispatched exactly once when a live thread "
            f"already owns the loop; dispatched={dispatched}"
        )

    def test_post_autofill_invalid_action_returns_400(self, autofill_client):
        """POST with an unrecognised action must return 400, never 500."""
        client, tmp_path, db_file = autofill_client

        resp = client.post(
            "/api/discovery/sector/23/autofill",
            json={"action": "explode"},
        )
        assert resp.status_code == 400, (
            f"bad action must return 400, got {resp.status_code}: {resp.text}"
        )

    def test_post_autofill_missing_action_returns_400(self, autofill_client):
        """POST with no 'action' key must return 400."""
        client, tmp_path, db_file = autofill_client

        resp = client.post(
            "/api/discovery/sector/23/autofill",
            json={},
        )
        assert resp.status_code == 400, (
            f"missing action must return 400, got {resp.status_code}: {resp.text}"
        )

    def test_sectors_response_includes_autofill_field(self, autofill_client):
        """GET /api/discovery/sectors includes an 'autofill' field per sector."""
        client, tmp_path, db_file = autofill_client

        # Seed an autofill row for sector 1
        from exonit.discovery_state import set_autofill
        set_autofill(1, db_path=db_file, status="running", lots_done=3)

        reg = {
            "active_sector": 1,
            "sectors": {
                "1": {
                    "enabled": True,
                    "total_tics": 1000,
                    "downloaded": 100,
                    "cursor": 50,
                    "canonical_list": "",
                }
            },
        }
        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "sector_gap_stats",
                          return_value=_FAKE_GAP_STATS), \
             patch.object(discovery_state, "get_sector_state",
                          return_value="active"), \
             patch.object(discovery_state, "get_autofill",
                          return_value={"status": "running", "lots_done": 3,
                                        "current_lot": "lot.csv", "last_alert": None}):
            resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        s = resp.json()["sectors"][0]
        assert "autofill" in s, "sector entry must have 'autofill' field"
        af = s["autofill"]
        assert af.get("status") == "running"
        assert af.get("lots_done") == 3

    def test_sectors_autofill_field_safe_when_no_row(self, autofill_client):
        """When no autofill row exists, 'autofill' field has None status (never 500)."""
        client, tmp_path, db_file = autofill_client

        reg = {
            "active_sector": 1,
            "sectors": {
                "1": {
                    "enabled": True,
                    "total_tics": 100,
                    "downloaded": 5,
                    "cursor": 0,
                    "canonical_list": "",
                }
            },
        }
        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "sector_gap_stats",
                          return_value=_FAKE_GAP_STATS), \
             patch.object(discovery_state, "get_sector_state",
                          return_value="coming_soon"), \
             patch.object(discovery_state, "get_autofill", return_value=None):
            resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        s = resp.json()["sectors"][0]
        assert "autofill" in s
        assert s["autofill"].get("status") is None


class TestAutofillStartupResume:
    """Verify that create_app() resumes autofill loops for 'running' sectors."""

    def test_startup_resumes_running_sectors(self, tmp_path, monkeypatch):
        """Sectors with discovery_autofill.status='running' are dispatched on startup."""
        # Create a DB with an autofill row already set to 'running' for sector 23.
        db_file = _make_autofill_sqlite_db(tmp_path)
        con = sqlite3.connect(db_file)
        con.execute(
            "INSERT INTO discovery_autofill(sector, status, lots_done) VALUES (23, 'running', 2)"
        )
        con.commit()
        con.close()

        monkeypatch.setenv("EXONIT_DB_PATH", db_file)
        monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setenv(
            "EXONIT_PIPELINE_STATE_FILE",
            str(tmp_path / "pipeline_state.json"),
        )
        monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)

        resumed = []
        # Use a threading event to wait for the background startup thread to complete.
        startup_done = threading.Event()

        def _fake_run_autofill(sector, *, deps):
            resumed.append(sector)
            startup_done.set()

        with patch("exonit.discovery_autofill.run_autofill",
                   side_effect=_fake_run_autofill):
            from exonit.server import create_app
            app = create_app()

            # Wait up to 2 s for the background resume thread to fire.
            startup_done.wait(timeout=2.0)

        assert 23 in resumed, (
            f"expected sector 23 to be resumed on startup, got: {resumed}"
        )


class TestAutofillAdvisoryLockGuard:
    """When pg_try_advisory_lock returns False, dispatch must be skipped.

    This covers the A3 multi-process startup race: if 4 uvicorn workers all
    call _dispatch_autofill_loop for the same sector, only the one that wins
    the PG advisory lock should start a run_autofill thread.

    Tests use SQLite + mocks so no real PG is required.
    """

    def test_dispatch_skipped_when_pg_lock_not_acquired(self, tmp_path, monkeypatch):
        """_try_pg_advisory_lock returning None must prevent run_autofill dispatch.

        Patches the module-level ``exonit.server._try_pg_advisory_lock`` to
        return ``None`` (lock already held by another process) without touching
        the broader ``is_pg_url`` / ``connect_operational`` paths, so the
        SQLite test DB still works normally for ``set_autofill`` / ``get_autofill``.
        """
        db_file = _make_autofill_sqlite_db(tmp_path)
        monkeypatch.setenv("EXONIT_DB_PATH", db_file)
        monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setenv(
            "EXONIT_PIPELINE_STATE_FILE",
            str(tmp_path / "pipeline_state.json"),
        )
        monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)

        # Prevent startup resume from dispatching any loops.
        from exonit import discovery_state as _ds
        monkeypatch.setattr(
            _ds, "load_registry",
            lambda path: {"active_sector": 1, "sectors": {}},
        )

        from exonit.server import create_app
        app = create_app()

        dispatched = []

        from fastapi.testclient import TestClient
        import time as _time

        with TestClient(app) as client:
            # Patch only _try_pg_advisory_lock at the server module level.
            # Returning None simulates "lock already held by another process".
            # This does NOT affect is_pg_url / connect_operational, so the
            # SQLite test DB continues to work for all other DB calls.
            with patch("exonit.server._try_pg_advisory_lock", return_value=None), \
                 patch("exonit.discovery_autofill.run_autofill",
                       side_effect=lambda sector, deps: dispatched.append(sector)):
                resp = client.post(
                    "/api/discovery/sector/23/autofill",
                    json={"action": "start"},
                )
                # Allow enough time for a (mistakenly) dispatched daemon thread to run.
                _time.sleep(0.05)

        assert resp.status_code == 200, resp.text
        assert len(dispatched) == 0, (
            "run_autofill must NOT be dispatched when _try_pg_advisory_lock "
            f"returns None; was called for sectors: {dispatched}"
        )


# ---------------------------------------------------------------------------
# Q1 (perf review 2026-09-26): gap-stats TTL raised + disabled/complete
# sectors skipped, and force-refresh hook at lot boundaries.
#
# Exercises app.state.gap_stats_compute_all / gap_stats_force_refresh_now /
# gap_stats_ttl_s directly (see the app.state exposure added next to
# _force_gap_stats_refresh_now) rather than through the /api/discovery/sectors
# route, which wraps gap-stats in a SECOND layer of stale-while-revalidate
# HTTP response caching (_cached_json, 30 s TTL) — going through both layers
# to prove a skip/refresh happened would be flaky (thread timing) and would
# really be testing _cached_json, not this change.
# ---------------------------------------------------------------------------

class TestGapStatsQ1SkipAndRefresh:
    def _make_app(self, tmp_path, monkeypatch, *, ttl_env=None):
        monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setenv(
            "EXONIT_PIPELINE_STATE_FILE",
            str(tmp_path / "pipeline_state.json"),
        )
        monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)
        if ttl_env is not None:
            monkeypatch.setenv("EXONIT_GAP_STATS_TTL_S", ttl_env)
        else:
            monkeypatch.delenv("EXONIT_GAP_STATS_TTL_S", raising=False)

        from exonit.server import create_app
        return create_app()

    def test_ttl_defaults_to_3600_and_is_env_overridable(self, tmp_path, monkeypatch):
        app = self._make_app(tmp_path, monkeypatch)
        assert app.state.gap_stats_ttl_s == 3600.0

        app2 = self._make_app(tmp_path, monkeypatch, ttl_env="120")
        assert app2.state.gap_stats_ttl_s == 120.0

    def test_disabled_sector_without_autofill_is_skipped(self, tmp_path, monkeypatch):
        """A disabled sector with no running auto-fill must not pay for the
        expensive per-sector tglc scan (sector_gap_stats)."""
        app = self._make_app(tmp_path, monkeypatch)
        sectors_raw = {
            "1": {"enabled": False, "canonical_list": ""},
        }
        from exonit import discovery_state
        with patch.object(discovery_state, "get_sector_state",
                           return_value="coming_soon"), \
             patch.object(discovery_state, "get_autofill", return_value=None), \
             patch.object(discovery_state, "sector_gap_stats") as m_gap:
            out = app.state.gap_stats_compute_all(sectors_raw, "unused.db")

        m_gap.assert_not_called()
        assert 1 not in out

    def test_complete_sector_without_autofill_is_skipped(self, tmp_path, monkeypatch):
        app = self._make_app(tmp_path, monkeypatch)
        sectors_raw = {
            "1": {"enabled": True, "canonical_list": ""},
        }
        from exonit import discovery_state
        with patch.object(discovery_state, "get_sector_state",
                           return_value="complete"), \
             patch.object(discovery_state, "get_autofill", return_value=None), \
             patch.object(discovery_state, "sector_gap_stats") as m_gap:
            out = app.state.gap_stats_compute_all(sectors_raw, "unused.db")

        m_gap.assert_not_called()
        assert 1 not in out

    def test_complete_sector_with_running_autofill_still_computes(
        self, tmp_path, monkeypatch
    ):
        """A 'complete' sector whose auto-fill loop is actively running
        (re-checking for newly-downloaded gaps) must NOT be skipped."""
        app = self._make_app(tmp_path, monkeypatch)
        sectors_raw = {
            "1": {"enabled": True, "canonical_list": ""},
        }
        fake_gap = {"gaps_processable": 0, "no_data": 0, "completed": 5,
                    "work_list_size": 5}
        from exonit import discovery_state
        with patch.object(discovery_state, "get_sector_state",
                           return_value="complete"), \
             patch.object(discovery_state, "get_autofill",
                          return_value={"status": "running"}), \
             patch.object(discovery_state, "sector_gap_stats",
                          return_value=fake_gap) as m_gap:
            out = app.state.gap_stats_compute_all(sectors_raw, "unused.db")

        m_gap.assert_called_once()
        assert 1 in out

    def test_enabled_active_sector_is_never_skipped(self, tmp_path, monkeypatch):
        app = self._make_app(tmp_path, monkeypatch)
        sectors_raw = {
            "23": {"enabled": True, "canonical_list": ""},
        }
        fake_gap = {"gaps_processable": 10, "no_data": 0, "completed": 5,
                    "work_list_size": 15}
        from exonit import discovery_state
        with patch.object(discovery_state, "get_sector_state",
                           return_value="active"), \
             patch.object(discovery_state, "sector_gap_stats",
                          return_value=fake_gap) as m_gap:
            out = app.state.gap_stats_compute_all(sectors_raw, "unused.db")

        m_gap.assert_called_once()
        assert 23 in out

    def test_state_lookup_failure_fails_open(self, tmp_path, monkeypatch):
        """A transient error reading discovery_sector_state must never hide a
        sector's gap stats -- fail open (still compute), not silently skip."""
        app = self._make_app(tmp_path, monkeypatch)
        sectors_raw = {
            "1": {"enabled": False, "canonical_list": ""},
        }
        fake_gap = {"gaps_processable": 1, "no_data": 0, "completed": 0,
                    "work_list_size": 1}
        from exonit import discovery_state
        with patch.object(discovery_state, "get_sector_state",
                           side_effect=RuntimeError("PG down")), \
             patch.object(discovery_state, "get_autofill",
                          side_effect=RuntimeError("PG down")), \
             patch.object(discovery_state, "sector_gap_stats",
                          return_value=fake_gap) as m_gap:
            out = app.state.gap_stats_compute_all(sectors_raw, "unused.db")

        m_gap.assert_called_once()
        assert 1 in out

    def test_force_refresh_kicks_background_compute(self, tmp_path, monkeypatch):
        """gap_stats_force_refresh_now must schedule a recompute (via the
        same lock the TTL-driven background refresh uses) rather than being
        a no-op, so scan-start/sector-DONE lot boundaries get fresh numbers
        without waiting out the (now much longer) TTL."""
        import time as _time_mod

        app = self._make_app(tmp_path, monkeypatch)
        sectors_raw = {"1": {"enabled": True, "canonical_list": ""}}
        fake_gap = {"gaps_processable": 3, "no_data": 0, "completed": 0,
                    "work_list_size": 3}

        from exonit import discovery_state
        with patch.object(discovery_state, "get_sector_state",
                           return_value="active"), \
             patch.object(discovery_state, "sector_gap_stats",
                          return_value=fake_gap) as m_gap:
            app.state.gap_stats_force_refresh_now(sectors_raw, "unused.db")
            # Background daemon thread; give it a moment to land.
            for _ in range(100):
                if m_gap.called:
                    break
                _time_mod.sleep(0.02)

        assert m_gap.called, "force refresh must trigger a recompute"

    def test_force_refresh_is_noop_when_refresh_already_in_flight(
        self, tmp_path, monkeypatch
    ):
        """Calling force-refresh while a refresh already holds the lock must
        not start a second, overlapping recompute."""
        app = self._make_app(tmp_path, monkeypatch)
        sectors_raw = {"1": {"enabled": True, "canonical_list": ""}}

        from exonit.shared_state import get_shared_state
        st = get_shared_state()
        # Simulate an in-flight refresh by pre-holding the lock (same key the
        # TTL path and the force-refresh path both use).
        got = st.set_if_absent("gap_stats:computing", "1", ttl=60.0)
        assert got, "test setup: expected to win the lock first"

        from exonit import discovery_state
        with patch.object(discovery_state, "sector_gap_stats") as m_gap:
            app.state.gap_stats_force_refresh_now(sectors_raw, "unused.db")

        m_gap.assert_not_called()


# ---------------------------------------------------------------------------
# B (2026-10-05, UI completion audit): closed (disabled/complete, no running
# autofill) sectors get their OWN background-refreshed gap-stats store
# (separate shared_state key, separate lock, 24 h TTL) instead of NEVER being
# computed at all -- the normal gap_stats store (above) deliberately skips
# them forever (Q1, 2026-09-26). Exercises app.state.closed_gap_stats_*
# directly, same rationale as TestGapStatsQ1SkipAndRefresh above (going
# through the HTTP route would also cross _cached_json's 30 s SWR layer).
# ---------------------------------------------------------------------------

class TestClosedGapStatsStore:
    def _make_app(self, tmp_path, monkeypatch, *, ttl_env=None):
        monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setenv(
            "EXONIT_PIPELINE_STATE_FILE",
            str(tmp_path / "pipeline_state.json"),
        )
        monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)
        if ttl_env is not None:
            monkeypatch.setenv("EXONIT_CLOSED_GAP_STATS_TTL_S", ttl_env)
        else:
            monkeypatch.delenv("EXONIT_CLOSED_GAP_STATS_TTL_S", raising=False)
        from exonit.server import create_app
        return create_app()

    def test_ttl_defaults_to_24h_and_is_env_overridable(self, tmp_path, monkeypatch):
        app = self._make_app(tmp_path, monkeypatch)
        assert app.state.closed_gap_stats_ttl_s == 86400.0

        app2 = self._make_app(tmp_path, monkeypatch, ttl_env="60")
        assert app2.state.closed_gap_stats_ttl_s == 60.0

    def test_closed_sector_is_computed(self, tmp_path, monkeypatch):
        """A disabled sector with no running auto-fill -- NEVER computed by
        the normal store -- must be computed by the closed-sector store."""
        app = self._make_app(tmp_path, monkeypatch)
        sectors_raw = {"15": {"enabled": False, "canonical_list": ""}}
        fake_gap = {"gaps_processable": 2500, "no_data": 10, "completed": 997490,
                    "work_list_size": 1000000}
        from exonit import discovery_state
        with patch.object(discovery_state, "get_sector_state",
                           return_value="complete"), \
             patch.object(discovery_state, "get_autofill", return_value=None), \
             patch.object(discovery_state, "sector_gap_stats",
                          return_value=fake_gap) as m_gap:
            out = app.state.closed_gap_stats_compute_all(sectors_raw, "unused.db")

        m_gap.assert_called_once()
        assert 15 in out
        assert out[15]["gaps_processable"] == 2500

    def test_open_sector_excluded(self, tmp_path, monkeypatch):
        """An active/moving sector must never be computed by the closed
        store -- it belongs exclusively to the normal gap_stats store."""
        app = self._make_app(tmp_path, monkeypatch)
        sectors_raw = {"6": {"enabled": True, "canonical_list": ""}}
        from exonit import discovery_state
        with patch.object(discovery_state, "get_sector_state",
                           return_value="active"), \
             patch.object(discovery_state, "sector_gap_stats") as m_gap:
            out = app.state.closed_gap_stats_compute_all(sectors_raw, "unused.db")

        m_gap.assert_not_called()
        assert 6 not in out

    def test_closed_sector_with_running_autofill_excluded(self, tmp_path, monkeypatch):
        """A closed sector whose auto-fill loop is actively running already
        gets fresh numbers from the normal store (idle TTL) -- it must not
        ALSO be picked up here (the two stores' "closed" definition must be
        identical and non-overlapping by construction)."""
        app = self._make_app(tmp_path, monkeypatch)
        sectors_raw = {"17": {"enabled": True, "canonical_list": ""}}
        from exonit import discovery_state
        with patch.object(discovery_state, "get_sector_state",
                           return_value="complete"), \
             patch.object(discovery_state, "get_autofill",
                          return_value={"status": "running"}), \
             patch.object(discovery_state, "sector_gap_stats") as m_gap:
            out = app.state.closed_gap_stats_compute_all(sectors_raw, "unused.db")

        m_gap.assert_not_called()
        assert 17 not in out

    def test_per_sector_failure_isolated(self, tmp_path, monkeypatch):
        """A compute failure for one closed sector must not prevent another
        closed sector from being computed (mirrors _compute_all_gap_stats'
        own per-sector try/except)."""
        app = self._make_app(tmp_path, monkeypatch)
        sectors_raw = {
            "2": {"enabled": False, "canonical_list": ""},
            "3": {"enabled": False, "canonical_list": ""},
        }
        fake_gap = {"gaps_processable": 3, "no_data": 0, "completed": 0,
                    "work_list_size": 3}

        def _gap(num, **kw):
            if num == 2:
                raise RuntimeError("PG down for sector 2")
            return fake_gap

        from exonit import discovery_state
        with patch.object(discovery_state, "get_sector_state",
                           return_value="complete"), \
             patch.object(discovery_state, "get_autofill", return_value=None), \
             patch.object(discovery_state, "sector_gap_stats", side_effect=_gap):
            out = app.state.closed_gap_stats_compute_all(sectors_raw, "unused.db")

        assert 2 not in out
        assert 3 in out

    def test_force_refresh_kicks_background_compute(self, tmp_path, monkeypatch):
        import time as _time_mod

        app = self._make_app(tmp_path, monkeypatch)
        sectors_raw = {"15": {"enabled": False, "canonical_list": ""}}
        fake_gap = {"gaps_processable": 8, "no_data": 0, "completed": 2564115,
                    "work_list_size": 2564123}

        from exonit import discovery_state
        with patch.object(discovery_state, "get_sector_state",
                           return_value="complete"), \
             patch.object(discovery_state, "get_autofill", return_value=None), \
             patch.object(discovery_state, "sector_gap_stats",
                          return_value=fake_gap) as m_gap:
            app.state.closed_gap_stats_force_refresh_now(sectors_raw, "unused.db",
                                                          sector=15)
            for _ in range(100):
                if m_gap.called:
                    break
                _time_mod.sleep(0.02)

        assert m_gap.called, "force refresh must trigger a recompute"

    def test_force_refresh_is_noop_when_already_in_flight(self, tmp_path, monkeypatch):
        app = self._make_app(tmp_path, monkeypatch)
        sectors_raw = {"15": {"enabled": False, "canonical_list": ""}}

        from exonit.shared_state import get_shared_state
        st = get_shared_state()
        got = st.set_if_absent("closed_gap_stats:computing", "1", ttl=60.0)
        assert got, "test setup: expected to win the lock first"

        from exonit import discovery_state
        with patch.object(discovery_state, "sector_gap_stats") as m_gap:
            app.state.closed_gap_stats_force_refresh_now(sectors_raw, "unused.db")

        m_gap.assert_not_called()

    def test_sectors_endpoint_serves_closed_sector_stats_from_warm_cache(
        self, tmp_path, monkeypatch,
    ):
        """End-to-end: a warm closed_gap_stats:data entry must reach a
        disabled sector's gaps_processable/pct_complete in the
        /api/discovery/sectors payload -- exactly like the live store does
        for open sectors (TestSectorsGapStatsWarmCache)."""
        import time as _time_mod
        monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setenv(
            "EXONIT_PIPELINE_STATE_FILE",
            str(tmp_path / "pipeline_state.json"),
        )
        monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)
        from exonit.server import create_app
        app = create_app()

        reg = {"active_sector": 1, "sectors": {"15": {
            "enabled": False, "total_tics": 2564123, "downloaded": 2564123,
            "cursor": 2564111, "canonical_list": str(tmp_path / "no_canon.csv"),
        }}}

        from exonit.shared_state import get_shared_state
        get_shared_state().set("closed_gap_stats:data", json.dumps({
            "data": {"15": {
                "gaps_processable": 8, "no_data": 2, "completed": 2564113,
                "work_list_size": 2564123,
                "computed_at": _time_mod.time(),
            }},
        }))

        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "get_sector_state",
                          return_value="complete"), \
             patch.object(discovery_state, "get_autofill", return_value=None):
            from fastapi.testclient import TestClient
            with TestClient(app) as client:
                resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        s = resp.json()["sectors"][0]
        assert s["number"] == 15
        assert s["gaps_processable"] == 8, (
            "closed-sector cache must reach the sector entry, same as the "
            "live gap_stats cache does for open sectors")
        assert s["no_data"] == 2
        assert abs(s["pct_complete"] - (2564113 + 2) / 2564123) < 1e-6
        assert s["pending"] == 8

    # -----------------------------------------------------------------
    # Review fix (2026-10-05, item 1 of
    # progress/review-discovery-closed-sector-gap-stats-2026-10-05.md):
    # the scan-START refresh trigger does nothing for a mop-up lot's
    # RESULT -- a lot-END trigger (pipeline_state.json reaching DONE/
    # FAILED for that exact sector) must also invalidate the closed
    # snapshot, not just the 24h TTL.
    # -----------------------------------------------------------------

    def _wait_for(self, predicate, timeout=2.0, interval=0.02):
        import time as _time_mod
        deadline = _time_mod.time() + timeout
        while _time_mod.time() < deadline:
            if predicate():
                return True
            _time_mod.sleep(interval)
        return predicate()

    def test_lot_end_triggers_refresh_for_stale_closed_snapshot(
        self, tmp_path, monkeypatch,
    ):
        """pipeline_state.json reaching DONE for a CLOSED sector, with a
        closed-store snapshot older than the lot's finish time, must kick a
        scoped background refresh for that sector -- without waiting for the
        24h TTL."""
        import time as _time_mod
        monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
        state_file = tmp_path / "pipeline_state.json"
        monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE", str(state_file))
        monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)
        from exonit.server import create_app
        app = create_app()

        reg = {"active_sector": 1, "sectors": {"15": {
            "enabled": False, "total_tics": 2564123, "downloaded": 2564123,
            "cursor": 2564119, "canonical_list": str(tmp_path / "no_canon.csv"),
        }}}

        # Snapshot from BEFORE the mop-up lot ran.
        from exonit.shared_state import get_shared_state
        st = get_shared_state()
        stale_computed_at = _time_mod.time() - 300.0
        st.set("closed_gap_stats:data", json.dumps({"data": {"15": {
            "gaps_processable": 2523, "no_data": 10, "completed": 2561590,
            "work_list_size": 2564123, "computed_at": stale_computed_at,
        }}}))

        # Lot against sector 15 just finished (DONE), AFTER the snapshot above.
        state_file.write_text(json.dumps({
            "state": "DONE", "sector": 15,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }))

        fresh_gap = {"gaps_processable": 8, "no_data": 10, "completed": 2564105,
                    "work_list_size": 2564123}

        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "get_sector_state",
                          return_value="complete"), \
             patch.object(discovery_state, "get_autofill", return_value=None), \
             patch.object(discovery_state, "sector_gap_stats",
                          return_value=fresh_gap) as m_gap:
            from fastapi.testclient import TestClient
            with TestClient(app) as client:
                resp = client.get("/api/discovery/sectors")
                assert resp.status_code == 200
                assert self._wait_for(lambda: m_gap.called)
                assert self._wait_for(
                    lambda: json.loads(st.get("closed_gap_stats:data"))
                    ["data"]["15"].get("gaps_processable") == 8)

        cached = json.loads(st.get("closed_gap_stats:data"))["data"]["15"]
        assert cached["gaps_processable"] == 8

    def test_lot_end_does_not_refresh_already_fresh_snapshot(
        self, tmp_path, monkeypatch,
    ):
        """A closed-store snapshot computed AFTER the lot's finish time must
        NOT be recomputed again -- no spurious refresh."""
        import time as _time_mod
        monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
        state_file = tmp_path / "pipeline_state.json"
        monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE", str(state_file))
        monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)
        from exonit.server import create_app
        app = create_app()

        reg = {"active_sector": 1, "sectors": {"15": {
            "enabled": False, "total_tics": 2564123, "downloaded": 2564123,
            "cursor": 2564119, "canonical_list": str(tmp_path / "no_canon.csv"),
        }}}

        state_file.write_text(json.dumps({
            "state": "DONE", "sector": 15,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }))

        from exonit.shared_state import get_shared_state
        st = get_shared_state()
        # Snapshot computed AFTER the lot's finish time (set second, so its
        # wall-clock computed_at is strictly later than updated_at above).
        fresh_computed_at = _time_mod.time()
        st.set("closed_gap_stats:data", json.dumps({"data": {"15": {
            "gaps_processable": 8, "no_data": 10, "completed": 2564105,
            "work_list_size": 2564123, "computed_at": fresh_computed_at,
        }}}))

        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "get_sector_state",
                          return_value="complete"), \
             patch.object(discovery_state, "get_autofill", return_value=None), \
             patch.object(discovery_state, "sector_gap_stats") as m_gap:
            from fastapi.testclient import TestClient
            with TestClient(app) as client:
                resp = client.get("/api/discovery/sectors")
                assert resp.status_code == 200
                _time_mod.sleep(0.2)  # give any stray background thread a chance

        m_gap.assert_not_called()

    def test_lot_end_for_different_sector_does_not_refresh(
        self, tmp_path, monkeypatch,
    ):
        """A finished lot against a DIFFERENT sector must not trigger a
        refresh for an unrelated closed sector's stale snapshot."""
        import time as _time_mod
        monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
        state_file = tmp_path / "pipeline_state.json"
        monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE", str(state_file))
        monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)
        from exonit.server import create_app
        app = create_app()

        reg = {"active_sector": 1, "sectors": {"15": {
            "enabled": False, "total_tics": 2564123, "downloaded": 2564123,
            "cursor": 2564119, "canonical_list": str(tmp_path / "no_canon.csv"),
        }}}

        # A DONE lot, but for sector 6 (not 15).
        state_file.write_text(json.dumps({
            "state": "DONE", "sector": 6,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }))

        from exonit.shared_state import get_shared_state
        st = get_shared_state()
        stale_computed_at = _time_mod.time() - 300.0
        st.set("closed_gap_stats:data", json.dumps({"data": {"15": {
            "gaps_processable": 2523, "no_data": 10, "completed": 2561590,
            "work_list_size": 2564123, "computed_at": stale_computed_at,
        }}}))

        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "get_sector_state",
                          return_value="complete"), \
             patch.object(discovery_state, "get_autofill", return_value=None), \
             patch.object(discovery_state, "sector_gap_stats") as m_gap:
            from fastapi.testclient import TestClient
            with TestClient(app) as client:
                resp = client.get("/api/discovery/sectors")
                assert resp.status_code == 200
                _time_mod.sleep(0.2)

        m_gap.assert_not_called()


# ---------------------------------------------------------------------------
# S2 (2026-09-26, docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md
# §5.2/§5.3): per-sector force-refresh must never touch OTHER sectors'
# cached entries, and a per-sector compute failure (the fdw-down case in
# production) must never drop that sector's last-known-good value.
# ---------------------------------------------------------------------------

class TestGapStatsS2PartialRefreshAndStaleWhileError:
    def _make_app(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setenv(
            "EXONIT_PIPELINE_STATE_FILE",
            str(tmp_path / "pipeline_state.json"),
        )
        monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)
        from exonit.server import create_app
        return create_app()

    def _seed_cache(self, sectors: dict) -> None:
        """Write a gap_stats:data payload with the CURRENT (per-sector
        computed_at) shape directly, bypassing the app's own write path —
        the fixture for "what was already cached before this refresh"."""
        import json as _json
        import time as _time_mod
        from exonit.shared_state import get_shared_state
        data = {}
        for num, stats in sectors.items():
            entry = dict(stats)
            entry.setdefault("computed_at", _time_mod.time() - 100000)  # stale
            data[str(num)] = entry
        get_shared_state().set("gap_stats:data", _json.dumps({"data": data}))

    def _wait_for(self, predicate, timeout=2.0, interval=0.02):
        import time as _time_mod
        deadline = _time_mod.time() + timeout
        while _time_mod.time() < deadline:
            if predicate():
                return True
            _time_mod.sleep(interval)
        return predicate()

    def test_per_sector_force_refresh_leaves_other_sectors_untouched(
        self, tmp_path, monkeypatch,
    ):
        app = self._make_app(tmp_path, monkeypatch)
        sectors_raw = {
            "1": {"enabled": True, "canonical_list": ""},
            "2": {"enabled": True, "canonical_list": ""},
        }
        self._seed_cache({
            1: {"gaps_processable": 10, "no_data": 0, "completed": 0,
                "work_list_size": 10},
            2: {"gaps_processable": 20, "no_data": 0, "completed": 0,
                "work_list_size": 20},
        })

        import json as _json
        from exonit.shared_state import get_shared_state
        st = get_shared_state()
        before_sector_2 = _json.loads(st.get("gap_stats:data"))["data"]["2"]

        fresh_sector_1 = {"gaps_processable": 1, "no_data": 0, "completed": 9,
                          "work_list_size": 10}
        from exonit import discovery_state
        with patch.object(discovery_state, "get_sector_state",
                          return_value="active"), \
             patch.object(discovery_state, "sector_gap_stats",
                          return_value=fresh_sector_1) as m_gap:
            app.state.gap_stats_force_refresh_now(sectors_raw, "unused.db", sector=1)
            assert self._wait_for(lambda: m_gap.called)
            # Let the background thread's write land.
            assert self._wait_for(
                lambda: _json.loads(st.get("gap_stats:data"))["data"]["1"]
                .get("gaps_processable") == 1)

        m_gap.assert_called_once()  # sector 2 was never recomputed
        after = _json.loads(st.get("gap_stats:data"))["data"]
        assert after["1"]["gaps_processable"] == 1, "sector 1 was refreshed"
        assert after["2"]["gaps_processable"] == before_sector_2["gaps_processable"], (
            "sector 2's cached value must be untouched by a sector-1-only refresh")
        assert after["2"] == before_sector_2

    def test_compute_failure_keeps_old_cached_value_for_that_sector(
        self, tmp_path, monkeypatch,
    ):
        """The stale-while-error contract (spec §5.2): a genuine compute
        failure (fdw down, in production) for a sector must never blank out
        or drop that sector's last-known-good cached value."""
        app = self._make_app(tmp_path, monkeypatch)
        sectors_raw = {"1": {"enabled": True, "canonical_list": ""}}
        self._seed_cache({
            1: {"gaps_processable": 5, "no_data": 1, "completed": 4,
                "work_list_size": 10},
        })

        import json as _json
        from exonit.shared_state import get_shared_state
        st = get_shared_state()

        from exonit import discovery_state
        with patch.object(discovery_state, "get_sector_state",
                          return_value="active"), \
             patch.object(discovery_state, "sector_gap_stats",
                          side_effect=RuntimeError("fdw down")) as m_gap:
            app.state.gap_stats_force_refresh_now(sectors_raw, "unused.db", sector=1)
            assert self._wait_for(lambda: m_gap.called)
            # Give the background thread's (failed) compute + merge-write a
            # moment to run to completion before asserting nothing changed.
            import time as _time_mod
            _time_mod.sleep(0.2)

        after = _json.loads(st.get("gap_stats:data"))["data"]["1"]
        assert after["gaps_processable"] == 5, (
            "a failed recompute must leave the previous value in place")
        assert after["completed"] == 4
        st.delete("gap_stats:computing")
