"""Tests for the discovery-campaign API endpoints added to exonit/server.py.

Constraints:
- NEVER touches the live DB (data/exonit.db).
- Uses FastAPI TestClient with a temp DB.
- No real SSH, no real subprocess network calls, no real workers spawned.
- All external calls (subprocess, Popen, farmctl) are monkeypatched.
"""

import json
import os
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixture: isolated TestClient with temp DB and temp scorecards dir
# ---------------------------------------------------------------------------

@pytest.fixture
def discovery_client(tmp_path, monkeypatch):
    """FastAPI TestClient with isolated DB, scorecards dir, and registry."""
    db_file = tmp_path / "test.db"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)

    monkeypatch.setenv("EXONIT_DB_PATH", str(db_file))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(reports_dir))
    # Isolate the pipeline_state file from the live one so the double-launch
    # guard and FAILED-state writes can be exercised without touching prod.
    monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE",
                       str(tmp_path / "pipeline_state.json"))
    # Isolate the metadata/TGLC index too. Without this the fixture falls back
    # to the repo default data/metadata_cache.db — which on <host> is the
    # FROZEN 61 GiB production index: /api/discovery/sectors then counts the
    # real work list and the class below went from 0,6 s to 48 s, with
    # order-dependent failures once the queries got slow enough. Tests must
    # never read a production artefact to decide a work-list size.
    monkeypatch.setenv("EXONIT_METADATA_CACHE",
                       str(tmp_path / "metadata_cache.db"))
    monkeypatch.delenv("EXONIT_METADATA_DB_URL", raising=False)

    from exonit.server import create_app
    app = create_app()

    # Expose the scorecards dir as a fixture attribute via app.state
    app.state._test_tmp = tmp_path

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        yield client, tmp_path


# ---------------------------------------------------------------------------
# Helper: build a minimal SQLite tasks DB
# ---------------------------------------------------------------------------

def _make_tasks_db(path: Path, done_tics: list, pending_tics: list = None):
    """Insert rows into an existing ExoNitDB tasks table.

    ExoNitDB._create_tables() has already created the tasks table (12 cols)
    when create_app() ran. We just INSERT with the columns we care about.
    """
    con = sqlite3.connect(str(path))
    # Ensure table exists (for standalone use outside the fixture)
    con.execute(
        "CREATE TABLE IF NOT EXISTS tasks "
        "(task_id TEXT PRIMARY KEY, tic_id TEXT, status TEXT DEFAULT 'pending',"
        " priority INTEGER DEFAULT 2, assigned_to TEXT, assigned_at TEXT,"
        " completed_at TEXT, processing_time REAL, gpu_used INTEGER DEFAULT 0,"
        " attempts INTEGER DEFAULT 0, last_error TEXT,"
        " created_at TEXT DEFAULT (datetime('now')))"
    )
    for i, t in enumerate(done_tics):
        con.execute(
            "INSERT OR IGNORE INTO tasks (task_id, tic_id, status) VALUES (?,?,?)",
            (f"done-{i}", str(t), "completed"),
        )
    for i, t in enumerate(pending_tics or []):
        con.execute(
            "INSERT OR IGNORE INTO tasks (task_id, tic_id, status) VALUES (?,?,?)",
            (f"pend-{i}", str(t), "pending"),
        )
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# Helper: write a minimal canonical CSV
# ---------------------------------------------------------------------------

def _write_canonical(path: Path, tic_ids: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("tic_id,label\n")
        for t in tic_ids:
            fh.write(f"{t},candidate\n")


# ---------------------------------------------------------------------------
# /api/discovery/sectors — read endpoint
# ---------------------------------------------------------------------------

class TestDiscoverySectors:

    def test_sectors_shape_no_registry(self, discovery_client):
        """When registry is missing, returns sane defaults without 500."""
        client, tmp_path = discovery_client

        # Patch the registry path to a nonexistent file
        from exonit import server as srv
        with patch.object(srv, "_DISCOVERY_REGISTRY" if hasattr(srv, "_DISCOVERY_REGISTRY") else "__name__",
                          str(tmp_path / "no_registry.json"), create=True):
            resp = client.get("/api/discovery/sectors")
        # Should be 200 regardless
        assert resp.status_code == 200
        data = resp.json()
        assert "active_sector" in data
        assert "sectors" in data
        assert isinstance(data["sectors"], list)

    def test_sectors_shape_with_registry(self, discovery_client, tmp_path):
        """With a real registry file, returns correct shape."""
        client, tmp_path = discovery_client

        # Write a fake registry
        reg_path = tmp_path / "sectors.json"
        canonical_csv = tmp_path / "canonical.csv"
        _write_canonical(canonical_csv, [101, 102, 103])

        reg = {
            "active_sector": 1,
            "sectors": {
                "1": {
                    "enabled": True,
                    "total_tics": 100,
                    "downloaded": 5,
                    "cursor": 3,
                    "canonical_list": str(canonical_csv),
                }
            }
        }
        reg_path.write_text(json.dumps(reg))

        # Patch the module-level registry path used inside create_app closure.
        # The endpoint references _DISCOVERY_REGISTRY which is a local string
        # inside create_app(); we patch via load_registry to return our reg.
        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg):
            resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        data = resp.json()
        assert data["active_sector"] == 1
        sectors = data["sectors"]
        assert len(sectors) == 1
        s = sectors[0]
        assert s["number"] == 1
        assert s["enabled"] is True
        assert s["total_tics"] == 100
        assert s["downloaded"] == 5
        assert "examined" in s
        assert "pending" in s
        # pending = total_tics - examined; examined <= total_tics
        assert s["pending"] >= 0

    def test_sectors_missing_canonical_and_tglc_falls_back_to_cursor(self, discovery_client):
        """When both canonical CSV and TGLC index yield no TICs, examined falls
        back to the registry cursor value."""
        client, tmp_path = discovery_client

        reg = {
            "active_sector": 1,
            "sectors": {
                "1": {
                    "enabled": True,
                    "total_tics": 1000,
                    "downloaded": 200,
                    "cursor": 42,
                    "canonical_list": str(tmp_path / "no_such_canonical.csv"),
                }
            }
        }
        from exonit import discovery_state
        # Patch the work-list COUNT to 0 (simulates both canonical and tglc
        # index missing — no TICs in the work list). The endpoint counts in SQL
        # via count_extended_canonical since 2026-07-27; patching the old
        # list-building load_extended_canonical here would no longer intercept
        # anything and the test would silently read the REAL metadata index.
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "count_extended_canonical", return_value=0):
            resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        s = resp.json()["sectors"][0]
        assert s["examined"] == 42   # fallback to cursor
        assert s["pending"] == 1000 - 42

    def test_sectors_includes_work_list_size(self, discovery_client):
        """Each sector entry includes work_list_size = len(extended work list)."""
        client, tmp_path = discovery_client

        reg = {
            "active_sector": 1,
            "sectors": {
                "1": {
                    "enabled": True,
                    "total_tics": 9999,
                    "downloaded": 10,
                    "cursor": 0,
                    "canonical_list": str(tmp_path / "no_canon.csv"),
                }
            }
        }
        from exonit import discovery_state
        # Synthetic work-list size of 42 (see count_extended_canonical note above)
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "count_extended_canonical",
                          return_value=42):
            resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        s = resp.json()["sectors"][0]
        assert "work_list_size" in s, "work_list_size must be present in sector entry"
        assert s["work_list_size"] == 42

    def test_sectors_examined_equals_cursor_not_db_count(self, discovery_client):
        """examined must always equal the persistent registry cursor, NOT the
        COUNT of completed tasks in the DB.

        This is the key correctness property: the cursor survives a DB reset
        (AWS use-case) while the DB count resets to 0.  Both values can differ;
        examined must follow the cursor.
        """
        client, tmp_path = discovery_client

        # Registry cursor = 100 (persistent progress from past runs).
        # DB has only 5 completed tasks (simulates a fresh DB after a reset).
        canonical_csv = tmp_path / "canonical.csv"
        _write_canonical(canonical_csv, list(range(1, 201)))

        db_path = tmp_path / "test.db"
        _make_tasks_db(db_path, done_tics=list(range(1, 6)), pending_tics=[])

        reg = {
            "active_sector": 1,
            "sectors": {
                "1": {
                    "enabled": True,
                    "total_tics": 200,
                    "downloaded": 150,
                    "cursor": 100,          # persistent cursor from registry
                    "canonical_list": str(canonical_csv),
                }
            }
        }
        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "count_extended_canonical",
                          return_value=200):
            resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        s = resp.json()["sectors"][0]
        # examined must be the cursor (100), NOT the DB count (5).
        assert s["examined"] == 100, (
            f"examined should be cursor=100, got {s['examined']} "
            f"(DB has 5 completed tasks; cursor must win)"
        )
        assert s["pending"] == 100   # total_tics(200) - examined(100)


# ---------------------------------------------------------------------------
# /api/discovery/sectors — 'downloaded' derived from the shared metadata
# index (2026-07-25 fix), not the per-host registry file alone.
#
# Root cause fixed: discovery_download.py writes the registry's 'downloaded'
# field exactly once, at the very end of a whole (possibly multi-day) run,
# AND on a host whose local sectors.json the server may not even be the one
# reading (registry writes are per-host; the server reads only its own
# copy). The index (PG/SQLite metadata_cache.db) is shared and current, so
# the endpoint now prefers it, via the background-refreshed
# _DOWNLOADED_COUNTS_* store in exonit/routes/discovery.py (same
# stale-while-revalidate shape as _get_gap_stats, but never computes
# inline — see that block's docstring).
# ---------------------------------------------------------------------------

class TestDiscoverySectorsDownloadedCounts:

    _KEY = "downloaded_counts:data"

    def _reg(self, tmp_path, downloaded=5, sector=1, extra_sectors=None):
        sectors = {
            str(sector): {
                "enabled": True,
                "total_tics": 100,
                "downloaded": downloaded,
                "cursor": 3,
                "canonical_list": str(tmp_path / "no_canon.csv"),
            }
        }
        if extra_sectors:
            sectors.update(extra_sectors)
        return {"active_sector": sector, "sectors": sectors}

    def test_uses_index_count_when_cache_warm(self, discovery_client):
        """A warm downloaded-counts cache wins over the registry's own
        (possibly stale, possibly wrong-host) stored value."""
        client, tmp_path = discovery_client
        reg = self._reg(tmp_path, downloaded=5, sector=1)

        from exonit.shared_state import get_shared_state
        st = get_shared_state()
        st.set(self._KEY, json.dumps(
            {"data": {"1": 5249999}, "computed_at": time.time()}))

        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg):
            resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        s = resp.json()["sectors"][0]
        assert s["downloaded"] == 5249999   # index count, not the stale 5

    def test_falls_back_to_registry_when_cache_cold(self, discovery_client):
        """Cold cache (no background compute has landed yet, e.g. right
        after a server restart): falls back to the registry's stored value
        -- never a false zero, never a 500. The endpoint also kicks off a
        background refresh (fire-and-forget; not asserted here)."""
        client, tmp_path = discovery_client
        reg = self._reg(tmp_path, downloaded=5, sector=1)

        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg):
            resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        s = resp.json()["sectors"][0]
        assert s["downloaded"] == 5   # registry fallback, not 0

    def test_falls_back_to_registry_when_shared_state_unreachable(self, discovery_client):
        """A broken shared-state backend must never raise or show 0 --
        falls back to the registry's stored value, same as the cold-cache
        path."""
        client, tmp_path = discovery_client
        reg = self._reg(tmp_path, downloaded=7, sector=1)

        from exonit.shared_state import get_shared_state
        st = get_shared_state()
        with patch.object(st, "get", side_effect=RuntimeError("shared-state down")):
            from exonit import discovery_state
            with patch.object(discovery_state, "load_registry", return_value=reg):
                resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        s = resp.json()["sectors"][0]
        assert s["downloaded"] == 7

    def test_no_cross_sector_bleed(self, discovery_client):
        """Two sectors resolved from ONE cached payload must not clobber
        each other -- the exact class of bug the health/work
        scoped-vs-global cache-key incident was (gotcha_health_work_cache_
        key_ignores_run_id): a shared key must disambiguate every
        parameterised value it holds."""
        client, tmp_path = discovery_client
        reg = self._reg(
            tmp_path, downloaded=1, sector=1,
            extra_sectors={"2": {
                "enabled": True, "total_tics": 200, "downloaded": 2,
                "cursor": 0, "canonical_list": str(tmp_path / "no_canon2.csv"),
            }},
        )

        from exonit.shared_state import get_shared_state
        st = get_shared_state()
        st.set(self._KEY, json.dumps(
            {"data": {"1": 111, "2": 222}, "computed_at": time.time()}))

        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg):
            resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        by_number = {s["number"]: s["downloaded"] for s in resp.json()["sectors"]}
        assert by_number[1] == 111
        assert by_number[2] == 222

    def test_downloaded_counts_key_distinct_from_gap_stats_key(self, discovery_client):
        """The downloaded-counts store and the gap-stats store must use
        distinct shared-state keys -- a key collision would silently make
        one clobber the other (same incident class as the health/work
        scoped-vs-global cache-key bug). Verified two ways: (1) the
        endpoint reads 'downloaded' from ITS OWN key, not the gap_stats
        sentinel: (2) the gap_stats key is provably untouched by the
        downloaded-counts write path."""
        client, tmp_path = discovery_client
        reg = self._reg(tmp_path, downloaded=9, sector=1)

        from exonit.shared_state import get_shared_state
        st = get_shared_state()
        # A gap_stats payload that, if downloaded_counts wrongly read from
        # (or wrote to) the same key, would look plausible but be
        # completely wrong for a downloaded count.
        gap_sentinel = json.dumps({
            "data": {"1": {"gaps_processable": 999, "no_data": 1, "completed": 2}},
            "computed_at": time.time(),
        })
        st.set("gap_stats:data", gap_sentinel)
        st.set(self._KEY, json.dumps(
            {"data": {"1": 42}, "computed_at": time.time()}))

        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg):
            resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        s = resp.json()["sectors"][0]
        assert s["downloaded"] == 42          # from downloaded_counts:data
        # gap_stats' own key must be untouched by the downloaded-counts
        # read/write path (distinct keys, no collision).
        assert st.get("gap_stats:data") == gap_sentinel


# ---------------------------------------------------------------------------
# downloaded_counts change-detection gate (2026-09-25, fix C): the
# background refresh must skip the expensive per-sector scan when the
# tglc/fits_blobs fingerprint is unchanged, and must still recompute when it
# changed, or when the safety-net max-staleness window has elapsed. The
# fingerprint and the per-sector scan are both monkeypatched at the
# MetadataCache class level — real for every instance the background thread
# constructs, regardless of which SQLite file _TGLC_INDEX happens to point at.
# ---------------------------------------------------------------------------

class TestDownloadedCountsFingerprintGate:

    _KEY = "downloaded_counts:data"

    def _reg(self, tmp_path, sector=1):
        return {"active_sector": sector, "sectors": {
            str(sector): {
                "enabled": True, "total_tics": 100, "downloaded": 5,
                "cursor": 3, "canonical_list": str(tmp_path / "no_canon.csv"),
            }
        }}

    def _wait_for(self, predicate, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.05)
        return False

    def test_skips_scan_when_fingerprint_unchanged(self, discovery_client, monkeypatch):
        client, tmp_path = discovery_client
        reg = self._reg(tmp_path)

        from exonit.shared_state import get_shared_state
        from exonit.metadata_cache import MetadataCache
        st = get_shared_state()

        fp = {"tglc": {"n_live_tup": 10}, "fits_blobs": {"n_live_tup": 20}}
        now = time.time()
        # Stale entry (past the 1800 s TTL) but fingerprint matches and the
        # last real recompute is recent -> must skip the scan.
        st.set(self._KEY, json.dumps({
            "data": {"1": 111}, "computed_at": now - 2000.0,
            "fingerprint": fp, "last_full_recompute": now - 60.0,
        }))

        monkeypatch.setattr(
            MetadataCache, "download_registry_fingerprint", lambda self: fp)
        calls = {"n": 0}

        def _counting_count(self, sector):
            calls["n"] += 1
            return 999

        monkeypatch.setattr(MetadataCache, "indexed_basename_count", _counting_count)

        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg):
            resp = client.get("/api/discovery/sectors")
        assert resp.status_code == 200

        # Wait for the background thread's "renew" write (computed_at bumped
        # past the seeded stale one) rather than asserting immediately.
        assert self._wait_for(
            lambda: json.loads(st.get(self._KEY) or "{}").get("computed_at", 0) > now - 10)

        entry = json.loads(st.get(self._KEY))
        assert entry["data"] == {"1": 111}, "skip must preserve the existing data verbatim"
        assert entry["fingerprint"] == fp
        assert entry["last_full_recompute"] == now - 60.0, (
            "a skip must NOT bump last_full_recompute — only a real scan does"
        )
        assert calls["n"] == 0, "indexed_basename_count must not run on a skip"

    def test_recomputes_when_fingerprint_changed(self, discovery_client, monkeypatch):
        client, tmp_path = discovery_client
        reg = self._reg(tmp_path)

        from exonit.shared_state import get_shared_state
        from exonit.metadata_cache import MetadataCache
        st = get_shared_state()

        now = time.time()
        st.set(self._KEY, json.dumps({
            "data": {"1": 111}, "computed_at": now - 2000.0,
            "fingerprint": {"tglc": {"n_live_tup": 10}},
            "last_full_recompute": now - 60.0,
        }))

        monkeypatch.setattr(
            MetadataCache, "download_registry_fingerprint",
            lambda self: {"tglc": {"n_live_tup": 999}})  # changed
        calls = {"n": 0}

        def _counting_count(self, sector):
            calls["n"] += 1
            return 777

        monkeypatch.setattr(MetadataCache, "indexed_basename_count", _counting_count)

        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg):
            resp = client.get("/api/discovery/sectors")
        assert resp.status_code == 200

        assert self._wait_for(lambda: calls["n"] > 0)
        entry = json.loads(st.get(self._KEY))
        assert entry["data"] == {"1": 777}
        assert entry["fingerprint"] == {"tglc": {"n_live_tup": 999}}
        assert entry["last_full_recompute"] > now - 10

    def test_forces_recompute_past_max_stale_hours_despite_matching_fingerprint(
        self, discovery_client, monkeypatch
    ):
        """Safety net: EXONIT_DOWNLOAD_REGISTRY_MAX_STALE_HOURS bounds how
        long a matching fingerprint can keep skipping the real scan."""
        client, tmp_path = discovery_client
        reg = self._reg(tmp_path)
        monkeypatch.setenv("EXONIT_DOWNLOAD_REGISTRY_MAX_STALE_HOURS", "1")

        from exonit.shared_state import get_shared_state
        from exonit.metadata_cache import MetadataCache
        st = get_shared_state()

        fp = {"tglc": {"n_live_tup": 10}}
        now = time.time()
        st.set(self._KEY, json.dumps({
            "data": {"1": 111}, "computed_at": now - 2000.0,
            "fingerprint": fp,
            "last_full_recompute": now - 7200.0,  # 2h ago, past the 1h ceiling
        }))

        monkeypatch.setattr(
            MetadataCache, "download_registry_fingerprint", lambda self: fp)
        calls = {"n": 0}

        def _counting_count(self, sector):
            calls["n"] += 1
            return 555

        monkeypatch.setattr(MetadataCache, "indexed_basename_count", _counting_count)

        # discovery.py reads EXONIT_DOWNLOAD_REGISTRY_MAX_STALE_HOURS once at
        # register()-time (module-level closure constant, mirroring every
        # other _TTL constant in this block) — build the app AFTER setting
        # the env var so this test's override actually takes effect.
        from exonit.server import create_app
        app = create_app()
        from fastapi.testclient import TestClient
        from exonit import discovery_state
        with TestClient(app) as c2:
            with patch.object(discovery_state, "load_registry", return_value=reg):
                resp = c2.get("/api/discovery/sectors")
        assert resp.status_code == 200

        assert self._wait_for(lambda: calls["n"] > 0)
        entry = json.loads(st.get(self._KEY))
        assert entry["data"] == {"1": 555}
        assert entry["last_full_recompute"] > now - 10

    def test_non_pg_backend_always_recomputes(self, discovery_client):
        """Cold cache, real SQLite metadata index (no monkeypatch): the
        fingerprint is None (no pg_stat_user_tables), so the gate must never
        claim 'unchanged' — the background refresh must still populate the
        store, exactly like the pre-fix behaviour."""
        client, tmp_path = discovery_client
        reg = self._reg(tmp_path)

        from exonit.shared_state import get_shared_state
        st = get_shared_state()
        assert st.get(self._KEY) is None  # cold

        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg):
            resp = client.get("/api/discovery/sectors")
        assert resp.status_code == 200

        assert self._wait_for(lambda: st.get(self._KEY) is not None)
        entry = json.loads(st.get(self._KEY))
        assert entry["fingerprint"] is None
        assert entry["data"] == {"1": 0}, "empty SQLite index -> 0 files indexed"


# ---------------------------------------------------------------------------
# gap_stats warm-cache read — /api/discovery/sectors
# ---------------------------------------------------------------------------

class TestSectorsGapStatsWarmCache:
    """A cached gap_stats payload must actually reach the sector entry.

    Regression for 2026-07-27: the warm-cache path returned the JSON payload's
    dict verbatim, so its keys were STRINGS while the per-sector lookup uses
    an int sector number — a silent always-miss. Sector 14 had
    gaps_processable=1478302 sitting in the cache while the API answered null
    on every request, the UI fell back to the phantom `total_tics - examined`
    (~460k too high), and the multi-million-row completed-set scan was re-run
    every TTL for a value nobody could read.
    """

    _KEY = "gap_stats:data"

    def _reg(self, tmp_path, sector=14, total_tics=100, cursor=30):
        return {"active_sector": sector, "sectors": {str(sector): {
            "enabled": True, "total_tics": total_tics, "downloaded": total_tics,
            "cursor": cursor, "canonical_list": str(tmp_path / "no_canon.csv"),
        }}}

    def test_warm_cache_string_keys_reach_the_sector_entry(self, discovery_client):
        client, tmp_path = discovery_client
        reg = self._reg(tmp_path)

        from exonit.shared_state import get_shared_state
        # Exactly what a real cache write looks like after the JSON round trip
        # that _write_gap_stats_cache performs: int keys become strings.
        get_shared_state().set(self._KEY, json.dumps({
            "data": {"14": {"gaps_processable": 7, "no_data": 2,
                            "completed": 91}},
            "computed_at": time.time(),
        }))

        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg):
            resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        s = resp.json()["sectors"][0]
        assert s["gaps_processable"] == 7, (
            "cached gap stats must reach the sector entry; None here means the "
            "string/int key mismatch is back")
        assert s["no_data"] == 2
        # pending must follow gaps_processable, NOT total_tics - examined (70).
        assert s["pending"] == 7

    def test_pending_falls_back_only_when_cache_is_empty(self, discovery_client):
        """With no cached gap stats the fallback still applies (unchanged)."""
        client, tmp_path = discovery_client
        reg = self._reg(tmp_path)

        from exonit import discovery_state
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "load_completed_tic_set",
                          side_effect=RuntimeError("no completed set in test")):
            resp = client.get("/api/discovery/sectors")

        assert resp.status_code == 200
        s = resp.json()["sectors"][0]
        assert s["gaps_processable"] is None
        assert s["pending"] == 70          # total_tics(100) - cursor(30)


# ---------------------------------------------------------------------------
# /api/scorecards — list endpoint
# ---------------------------------------------------------------------------

class TestScorecardsEndpoints:

    def _make_sc_dir(self, tmp_path: Path) -> Path:
        """Create a temp scorecards directory with 2 fake scorecard JSONs."""
        sc_dir = tmp_path / "fake_scorecards"
        sc_dir.mkdir()

        sc1 = {
            "campaign": "discovery-tglc",
            "sector": 1,
            "lot": 1,
            "date": "2026-05-26",
            "n_stars": 500,
            "clean_planets": 0,
        }
        sc2 = {
            "generated_at": "2026-04-21T21:16:54Z",
            "sample": "data/samples/stratified_100_seed42.csv",
            "n_tics": 98,
        }
        (sc_dir / "discovery_s0001_lot1_2026-05-26.json").write_text(json.dumps(sc1))
        (sc_dir / "baseline-100-5workers-2026-04-21.json").write_text(json.dumps(sc2))
        (sc_dir / "broken.json").write_text("NOT JSON {{{")
        return sc_dir

    def test_scorecards_list(self, discovery_client):
        """List endpoint returns known fields and skips broken files."""
        client, tmp_path = discovery_client
        sc_dir = self._make_sc_dir(tmp_path)

        from exonit import server as srv
        with patch.object(srv, "_scorecards_dir", return_value=sc_dir):
            resp = client.get("/api/scorecards")

        assert resp.status_code == 200
        items = resp.json()
        # broken.json is skipped; 2 valid ones should appear
        names = [i["name"] for i in items]
        assert "broken.json" not in names
        assert any("discovery_s0001_lot1" in n for n in names)
        assert any("baseline-100-5workers" in n for n in names)
        for item in items:
            assert "name" in item
            assert "date" in item
            assert "summary" in item

    def test_scorecards_list_newest_first(self, discovery_client):
        """Scorecards are ordered by date descending (newest first), even when
        filename-alphabetical order disagrees with date order."""
        client, tmp_path = discovery_client
        sc_dir = tmp_path / "ordered_scorecards"
        sc_dir.mkdir()
        # Filenames chosen so alphabetical order != date order: by name,
        # 'lot20260529T…' sorts before 'lot4', but it is the NEWEST by date.
        files = {
            "discovery_s0001_lot4_2026-05-28.json": "2026-05-28",
            "discovery_s0001_lot20260529T233851_2026-05-30.json": "2026-05-30",
            "discovery_s0001_lot1_2026-05-26.json": "2026-05-26",
        }
        for fname, d in files.items():
            (sc_dir / fname).write_text(json.dumps({"date": d, "clean_planets": 0}))

        from exonit import server as srv
        with patch.object(srv, "_scorecards_dir", return_value=sc_dir):
            resp = client.get("/api/scorecards")

        assert resp.status_code == 200
        dates = [i["date"] for i in resp.json()]
        assert dates == ["2026-05-30", "2026-05-28", "2026-05-26"]

    def test_scorecards_summary_contains_real_keys(self, discovery_client):
        """Summary contains only real keys found in that JSON."""
        client, tmp_path = discovery_client
        sc_dir = self._make_sc_dir(tmp_path)

        from exonit import server as srv
        with patch.object(srv, "_scorecards_dir", return_value=sc_dir):
            resp = client.get("/api/scorecards")

        items = {i["name"]: i for i in resp.json()}
        lot1 = items.get("discovery_s0001_lot1_2026-05-26.json", {})
        assert lot1.get("summary", {}).get("campaign") == "discovery-tglc"
        assert lot1.get("summary", {}).get("n_stars") == 500

    def test_scorecards_empty_dir(self, discovery_client):
        """Returns empty list when dir is empty."""
        client, tmp_path = discovery_client
        sc_dir = tmp_path / "empty_sc"
        sc_dir.mkdir()

        from exonit import server as srv
        with patch.object(srv, "_scorecards_dir", return_value=sc_dir):
            resp = client.get("/api/scorecards")

        assert resp.status_code == 200
        assert resp.json() == []

    def test_scorecards_missing_dir(self, discovery_client):
        """Returns empty list when scorecards dir doesn't exist."""
        client, tmp_path = discovery_client
        sc_dir = tmp_path / "nonexistent_sc"

        from exonit import server as srv
        with patch.object(srv, "_scorecards_dir", return_value=sc_dir):
            resp = client.get("/api/scorecards")

        assert resp.status_code == 200
        assert resp.json() == []

    def test_scorecard_get_by_name(self, discovery_client):
        """GET /api/scorecards/{name} returns the raw JSON."""
        client, tmp_path = discovery_client
        sc_dir = self._make_sc_dir(tmp_path)

        from exonit import server as srv
        with patch.object(srv, "_scorecards_dir", return_value=sc_dir):
            resp = client.get("/api/scorecards/discovery_s0001_lot1_2026-05-26.json")

        assert resp.status_code == 200
        data = resp.json()
        assert data["campaign"] == "discovery-tglc"
        assert data["n_stars"] == 500

    def test_scorecard_get_not_found(self, discovery_client):
        """Missing scorecard returns 404."""
        client, tmp_path = discovery_client
        sc_dir = self._make_sc_dir(tmp_path)

        from exonit import server as srv
        with patch.object(srv, "_scorecards_dir", return_value=sc_dir):
            resp = client.get("/api/scorecards/nonexistent-2099-01-01.json")

        assert resp.status_code == 404

    def test_scorecard_path_traversal_rejected(self, discovery_client):
        """Path traversal in name returns 400, not 500 or a file leak."""
        client, tmp_path = discovery_client

        resp = client.get("/api/scorecards/../etc/passwd")
        # FastAPI will route /api/scorecards/ and treat ../etc/passwd as name;
        # the endpoint validates and returns 400.
        assert resp.status_code in (400, 404)

    def test_scorecard_dotdot_name_rejected(self, discovery_client):
        """Filename containing '..' returns 400."""
        client, tmp_path = discovery_client

        from exonit import server as srv
        sc_dir = self._make_sc_dir(tmp_path)
        with patch.object(srv, "_scorecards_dir", return_value=sc_dir):
            # URL-encode the '..' so FastAPI passes it as the name param
            resp = client.get("/api/scorecards/..%2Fetc%2Fpasswd.json")
        assert resp.status_code in (400, 404)


# ---------------------------------------------------------------------------
# /api/discovery/scan — action endpoint
# ---------------------------------------------------------------------------

def _make_scan_popen(status: str, extra: dict = None):
    """Return a mock mimicking asyncio.create_subprocess_exec's Process.

    The /api/discovery/scan endpoint now runs discovery_run.py via
    asyncio.create_subprocess_exec + await proc.communicate() so it NEVER
    blocks the single uvicorn event loop (the old Popen+communicate() froze
    the loop for the whole enqueue, forcing the slow per-target fallback and
    the 600 s timeout). The async Process mock exposes:
      - .pid
      - .communicate() -> awaitable (stdout_bytes, stderr_bytes)
      - .wait() -> awaitable returncode  (used after a timeout kill)
      - .returncode
    """
    payload = {"status": status}
    if extra:
        payload.update(extra)
    stdout_bytes = (json.dumps(payload) + "\n").encode("utf-8")

    exit_codes = {
        "started": 0,
        "dry_run": 0,
        "need_download": 3,
        "sector_not_enabled": 2,
        "error": 2,
        "enqueue_failed": 4,
    }
    rc = exit_codes.get(status, 1)

    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_proc.returncode = rc
    mock_proc.communicate = AsyncMock(return_value=(stdout_bytes, b""))
    mock_proc.wait = AsyncMock(return_value=rc)
    return mock_proc


def test_lot_stamp_from_csv_matches_orchestrator_convention():
    """The scan endpoint's run_id-minting helper MUST use the exact same lot
    identifier convention as scripts/discovery_orchestrator.py's
    _lot_number_from_csv (Fase 3 Task 2 hard requirement: naming compat with
    existing scorecards/dossiers)."""
    import sys as _sys
    from pathlib import Path as _Path

    from exonit.routes.discovery import _lot_stamp_from_csv

    scripts_dir = _Path(__file__).resolve().parent.parent / "scripts"
    _sys.path.insert(0, str(scripts_dir))
    import discovery_orchestrator as orch  # noqa: E402

    for name in ["data/discovery/s0001_lot4.csv",
                 "data/discovery/s0001_lot_20260531T082820.csv",
                 "weird.csv",
                 "data/discovery/nolotnumber.csv"]:
        assert _lot_stamp_from_csv(name) == orch._lot_number_from_csv(name), name


def test_lot_stamp_from_csv_unchanged_for_deep_lots():
    """FIX-1 (final-review, spec §4): _lot_stamp_from_csv's bare ts/number
    extraction must stay byte-identical even for deep-rescan csv names --
    the deep_lot/lot distinction lives in the separate _lot_prefix_from_csv
    helper, not here (this regex already matches the embedded 'lot_<ts>'
    substring inside 's0014_deep_lot_<ts>.csv', so no change was needed)."""
    from exonit.routes.discovery import _lot_stamp_from_csv
    assert (_lot_stamp_from_csv("data/discovery/s0014_deep_lot_20260723T101010.csv")
            == "20260723T101010")


def test_lot_prefix_from_csv_deep_vs_normal():
    """Spec §4: deep-rescan csv (s0014_deep_lot_<ts>.csv) -> 'deep_lot'
    prefix; normal lots (timestamp or numbered) -> unchanged 'lot' prefix."""
    from exonit.routes.discovery import _lot_prefix_from_csv
    assert _lot_prefix_from_csv(
        "data/discovery/s0014_deep_lot_20260723T101010.csv") == "deep_lot"
    assert _lot_prefix_from_csv(
        "data/discovery/s0014_lot_20260723T101010.csv") == "lot"
    assert _lot_prefix_from_csv("data/discovery/s0001_lot4.csv") == "lot"


def test_mint_sites_agree_on_deep_run_id():
    """Both-sites-must-agree invariant (FIX-1): exonit/routes/discovery.py's
    run_id mint expression and scripts/discovery_orchestrator.py's
    _derive_run_id must produce the IDENTICAL run_id for the same lot_csv,
    for both a deep-rescan lot and a normal lot."""
    import sys as _sys
    from pathlib import Path as _Path

    from exonit.routes.discovery import _lot_prefix_from_csv, _lot_stamp_from_csv

    scripts_dir = _Path(__file__).resolve().parent.parent / "scripts"
    _sys.path.insert(0, str(scripts_dir))
    import discovery_orchestrator as orch  # noqa: E402

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sector = 14
    cases = {
        "data/discovery/s0014_deep_lot_20260723T101010.csv":
            f"discovery_s0014_deep_lot20260723T101010_{today}",
        "data/discovery/s0014_lot_20260723T101010.csv":
            f"discovery_s0014_lot20260723T101010_{today}",
    }
    for csv, expected in cases.items():
        route_run_id = (f"discovery_s{sector:04d}_"
                         f"{_lot_prefix_from_csv(csv)}{_lot_stamp_from_csv(csv)}_{today}")
        orch_run_id = orch._derive_run_id(sector, csv)
        assert route_run_id == expected, csv
        assert orch_run_id == expected, csv


def test_derive_run_id_unified_across_the_three_minting_sites():
    """S3 (docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md §3.1):
    exonit.discovery_state.derive_run_id is now the single canonical
    implementation — scripts/discovery_orchestrator.py's _derive_run_id is a
    thin wrapper over it (asserted directly here, not just via a
    byte-identical re-implementation as the older tests above do), and
    scripts/discovery_run.py's own mint (exercised end-to-end in
    tests/test_discovery_run.py) must produce the identical string for the
    same (sector, lot_csv) — including across a UTC midnight straddle
    (the exact bug FIX-1/F5 already guard against at the route-mint site)."""
    import sys as _sys
    from datetime import datetime as _dt, timezone as _tz
    from pathlib import Path as _Path

    from exonit.discovery_state import derive_run_id

    scripts_dir = _Path(__file__).resolve().parent.parent / "scripts"
    _sys.path.insert(0, str(scripts_dir))
    import discovery_orchestrator as orch  # noqa: E402

    cases = [
        (1, "data/discovery/s0001_lot_20260531T082820.csv"),
        (6, "data/discovery/s0006_lot4.csv"),
        (14, "data/discovery/s0014_deep_lot_20260723T101010.csv"),
        (45, "weird.csv"),
    ]
    for sector, csv in cases:
        assert orch._derive_run_id(sector, csv) == derive_run_id(sector, csv), csv

    # UTC midnight straddle: an explicit now_utc must be honoured exactly
    # (this is what lets discovery_run.py mint BEFORE spawning the
    # orchestrator and the route's "child wins" preference — S3 §3.1 —
    # never have to re-derive a DIFFERENT date for the same lot_csv).
    before = _dt(2026, 12, 31, 23, 59, 59, tzinfo=_tz.utc)
    after = _dt(2027, 1, 1, 0, 0, 1, tzinfo=_tz.utc)
    csv = "data/discovery/s0006_lot_20261231T235959.csv"
    assert derive_run_id(6, csv, now_utc=before) == (
        "discovery_s0006_lot20261231T235959_2026-12-31")
    assert derive_run_id(6, csv, now_utc=after) == (
        "discovery_s0006_lot20261231T235959_2027-01-01")


def test_reap_in_background_reaps_real_child():
    """Zombie-leak fix (2026-07-07): _reap_in_background's daemon thread must
    actually call proc.wait() on a REAL child process. Without it, every
    orchestrator Popen'd by the long-lived uvicorn worker (scan/resume/
    autofill) finishes as a <defunct> zombie that only the next server
    restart collects. Uses a real short-lived subprocess (no mocking).

    F1/F2 (2026-07-10 review, tautological-test fix): this test must NEVER
    call proc.poll()/proc.wait() itself -- either one reaps the zombie on
    its own, which would make the test pass even if _reap_in_background is
    a complete no-op (the original version of this test called proc.poll()
    in its own wait loop and was tautological for exactly this reason).
    Instead it polls the plain ``proc.returncode`` ATTRIBUTE -- a plain
    attribute read never reaps anything, it is only ever set as a
    side-effect of a REAL poll()/wait() call, which here can only be
    _reap_in_background's own background thread.
    """
    from exonit.routes.discovery import _reap_in_background

    proc = subprocess.Popen(["sleep", "0.05"])
    pid = proc.pid
    _reap_in_background(proc)

    deadline = time.monotonic() + 2.0
    while proc.returncode is None and time.monotonic() < deadline:
        time.sleep(0.02)

    assert proc.returncode == 0, (
        f"child pid={pid} was not reaped within 2s -- _reap_in_background's "
        "background thread is not wait()ing on it")

    # Reinforcement: waitpid() (what Popen.wait() calls) removes the kernel's
    # process-table entry the instant it collects the exit status, so
    # /proc/<pid> should already be gone (or, if the pid got recycled in the
    # meantime, at least not still parked in zombie state 'Z').
    #
    # F2: the /proc entry can vanish between the existence check and the
    # read (a short-lived unrelated process reusing the same pid, or the
    # kernel finishing cleanup) -- treat that TOCTOU race as "reaped", not
    # a test failure.
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        state = stat_path.read_text().split(")")[-1].split()[0]
        assert state != "Z", f"pid={pid} is still a zombie in /proc"
    except FileNotFoundError:
        pass  # already gone -- also proof it was reaped, not left a zombie


class TestDiscoveryScan:

    def _post_scan(self, client, monkeypatch, status: str, extra=None):
        scan_proc = _make_scan_popen(status, extra)

        async def _fake_create(*args, **kwargs):
            return scan_proc

        # The scan goes through asyncio.create_subprocess_exec; the post-run
        # orchestrator (status=started) still spawns via subprocess.Popen. Mock
        # both so no real process is launched.
        with patch("asyncio.create_subprocess_exec", side_effect=_fake_create), \
                patch("subprocess.Popen", return_value=MagicMock(pid=99999)):
            resp = client.post("/api/discovery/scan",
                               json={"sector": 1, "count": 100})
        return resp

    def test_scan_started_returns_200(self, discovery_client, monkeypatch):
        client, _ = discovery_client
        resp = self._post_scan(client, monkeypatch, "started",
                               {"enqueued": 100, "cursor_after": 100})
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"

    def test_scan_enqueueing_drops_previous_lot_fields(self, discovery_client,
                                                      monkeypatch, tmp_path):
        """A new launch must not carry the PREVIOUS lot's identity into the
        ENQUEUEING mirror (incident 2026-09-24: the UI showed the closed
        lot's run_id/lot_csv/dead orchestrator_pid for the whole enqueue)."""
        client, _ = discovery_client
        sf = tmp_path / "pipeline_state.json"
        sf.write_text(json.dumps({
            "state": "DONE", "run_id": "discovery_s0005_lot_OLD",
            "lot_csv": "data/discovery/old.csv", "orchestrator_pid": 424242,
            "scorecard_json": "data/scorecards/old.json",
            "fleet_autoscale_warning": "old warning",
            "finished_at": "2026-09-24T15:49:50+00:00"}))
        seen = {}
        real_write = Path.write_text

        def _spy(self, data, *a, **kw):
            if self.name.startswith("pipeline_state") and '"ENQUEUEING"' in data:
                seen.update(json.loads(data))
            return real_write(self, data, *a, **kw)

        with patch.object(Path, "write_text", _spy):
            resp = self._post_scan(client, monkeypatch, "need_download",
                                   {"ready": 5, "missing": 95})
        assert resp.status_code == 409
        assert seen["state"] == "ENQUEUEING"
        assert seen["sector"] == 1
        for stale in ("run_id", "lot_csv", "orchestrator_pid", "scorecard_json",
                      "fleet_autoscale_warning", "finished_at"):
            assert stale not in seen, stale

    def test_scan_dry_run_returns_200(self, discovery_client, monkeypatch):
        client, _ = discovery_client
        resp = self._post_scan(client, monkeypatch, "dry_run")
        assert resp.status_code == 200
        assert resp.json()["status"] == "dry_run"

    def test_scan_need_download_returns_409(self, discovery_client, monkeypatch):
        client, _ = discovery_client
        resp = self._post_scan(client, monkeypatch, "need_download",
                               {"ready": 5, "missing": 95})
        assert resp.status_code == 409
        data = resp.json()
        assert data["status"] == "need_download"
        assert "ready" in data

    def test_scan_sector_not_enabled_returns_400(self, discovery_client, monkeypatch):
        client, _ = discovery_client
        resp = self._post_scan(client, monkeypatch, "sector_not_enabled")
        assert resp.status_code == 400

    def test_scan_error_returns_400(self, discovery_client, monkeypatch):
        client, _ = discovery_client
        resp = self._post_scan(client, monkeypatch, "error")
        assert resp.status_code == 400

    def test_scan_enqueue_failed_returns_500(self, discovery_client, monkeypatch):
        client, _ = discovery_client
        resp = self._post_scan(client, monkeypatch, "enqueue_failed",
                               {"detail": "enqueue_from_csv.py exited 1: error"})
        assert resp.status_code == 500
        assert resp.json()["status"] == "enqueue_failed"

    def test_scan_non_json_body_returns_400(self, discovery_client):
        """Non-JSON body must return 400, not 500 with traceback."""
        client, _ = discovery_client
        resp = client.post(
            "/api/discovery/scan",
            content=b"not json at all",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data.get("status") == "error"
        assert data.get("reason") == "bad_request"

    def test_scan_non_numeric_sector_returns_400(self, discovery_client, monkeypatch):
        """sector='abc' must return 400 before any subprocess is spawned."""
        client, _ = discovery_client
        created = []

        async def _fake_create(*args, **kwargs):
            created.append(args)
            return _make_scan_popen("started")

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_create):
            resp = client.post(
                "/api/discovery/scan",
                json={"sector": "abc", "count": 10},
            )
        assert resp.status_code == 400
        data = resp.json()
        assert data.get("status") == "error"
        assert data.get("reason") == "bad_request"
        assert created == [], "scan subprocess must not spawn on bad request"

    def test_scan_started_resets_progress_baselines(self, discovery_client,
                                                    monkeypatch):
        """A new lot must NOT inherit the previous run's progress counters.

        The baselines are set-once in the shared store and `done = baseline -
        pending`, so a leftover baseline makes the new run open already
        "done": on 2026-07-26 a 100,000-TIC lot was born reading "200,049 of
        299,944" — the numbers of the 300,000-TIC lot before it — and its
        header, bar, ETA and time-remaining stayed wrong for the whole run.
        Pressing "Reset stats" by hand fixed it, which is exactly what this
        endpoint must now do by itself.
        """
        from exonit.server import (
            _EXOMINER_BASELINE_PENDING_KEY,
            _FPP_BACKFILL_DONE_KEY,
            _LORA_BASELINE_PENDING_KEY,
            _PRIMARY_BASELINE_PENDING_KEY,
            _PRIMARY_BASELINE_STARTED_AT_KEY,
        )
        from exonit.shared_state import get_shared_state

        client, _ = discovery_client
        st = get_shared_state()
        st.set_int(_PRIMARY_BASELINE_PENDING_KEY, 299944)
        st.set_int(_PRIMARY_BASELINE_STARTED_AT_KEY, 1753000000)
        st.set_int(_EXOMINER_BASELINE_PENDING_KEY, 5000)
        st.set_int(_LORA_BASELINE_PENDING_KEY, 700)
        st.set_int(_FPP_BACKFILL_DONE_KEY, 4242)

        resp = self._post_scan(client, monkeypatch, "started",
                               {"enqueued": 100, "cursor_after": 100})

        assert resp.status_code == 200
        assert st.get_int(_PRIMARY_BASELINE_PENDING_KEY, -1) in (0, -1), \
            "the previous run's primary baseline survived the new scan"
        assert st.get_int(_EXOMINER_BASELINE_PENDING_KEY, -1) in (0, -1)
        assert st.get_int(_LORA_BASELINE_PENDING_KEY, -1) in (0, -1)
        # This one is a cumulative counter, not a snapshot: zeroed, not deleted.
        assert st.get_int(_FPP_BACKFILL_DONE_KEY, -1) == 0

    def test_scan_dry_run_leaves_baselines_alone(self, discovery_client,
                                                 monkeypatch):
        """A dry run enqueues nothing, so it must not disturb a live run's
        counters — someone checking what a lot WOULD do should not blank the
        dashboard of the run currently in flight."""
        from exonit.server import _PRIMARY_BASELINE_PENDING_KEY
        from exonit.shared_state import get_shared_state

        client, _ = discovery_client
        st = get_shared_state()
        st.set_int(_PRIMARY_BASELINE_PENDING_KEY, 299944)

        resp = self._post_scan(client, monkeypatch, "dry_run")

        assert resp.status_code == 200
        assert st.get_int(_PRIMARY_BASELINE_PENDING_KEY, -1) == 299944

    def test_scan_started_spawns_orchestrator(self, discovery_client, monkeypatch):
        """started + lot_csv → orchestrator spawned via subprocess.Popen with
        discovery_orchestrator.py and --lot-csv <path> in its argv.

        The scan itself goes through asyncio.create_subprocess_exec; only the
        orchestrator still uses subprocess.Popen. The pipeline_state file is
        isolated to tmp_path by the fixture (no worktree skip needed).

        Fase 3 Task 2: also asserts the discovery_runs PG row is created
        up-front (state=ENQUEUEING default) and --run-id is passed to the
        orchestrator's argv with the SAME minted run_id.

        Fase 3 Task 4: the endpoint now polls discovery_runs for the
        orchestrator's first SCANNING transition before returning 200 (spawn
        confirmation) — the fake Popen dispatch below writes that transition
        itself via a real ExoNitDB handle against the fixture's test.db,
        exactly mirroring what the real subprocess's first _set_state() call
        does. Contract change from before this task: the row is observed as
        SCANNING here, not ENQUEUEING (the endpoint would no longer return
        200 for a row stuck at ENQUEUEING — see
        test_scan_orchestrator_spawn_timeout_returns_502_and_marks_failed)."""
        client, tmp_path = discovery_client

        lot_csv = "/tmp/s0001_lot5.csv"
        scan_proc = _make_scan_popen("started", {"lot_csv": lot_csv})

        async def _fake_create(*args, **kwargs):
            return scan_proc

        # Ensure no active state lingers so _already_running stays False.
        (tmp_path / "pipeline_state.json").unlink(missing_ok=True)

        orchestrator_calls = []

        from exonit.db import ExoNitDB
        test_db_path = str(tmp_path / "test.db")

        def _popen_dispatch(cmd, **kwargs):
            orchestrator_calls.append(cmd)
            # Fase 3 Task 4: simulate the orchestrator's first act (moving
            # the discovery_runs row off ENQUEUEING) so the endpoint's spawn
            # confirmation poll succeeds instead of timing out into a 502.
            run_id_idx = cmd.index("--run-id") + 1
            fake_db = ExoNitDB(test_db_path)
            try:
                fake_db.update_discovery_run(cmd[run_id_idx], state="SCANNING")
            finally:
                fake_db.close()
            m = MagicMock()
            m.pid = 77777
            return m

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_create), \
                patch("subprocess.Popen", side_effect=_popen_dispatch):
            resp = client.post("/api/discovery/scan",
                               json={"sector": 1, "count": 100})

        assert resp.status_code == 200
        assert resp.json()["status"] == "started"
        assert len(orchestrator_calls) == 1, f"Expected 1 orchestrator Popen, got {orchestrator_calls}"
        argv = orchestrator_calls[0]
        assert any("discovery_orchestrator.py" in str(a) for a in argv), (
            f"discovery_orchestrator.py not found in Popen argv: {argv}"
        )
        assert "--lot-csv" in argv, f"--lot-csv not found in Popen argv: {argv}"
        lot_csv_idx = argv.index("--lot-csv")
        assert argv[lot_csv_idx + 1] == lot_csv, (
            f"Expected lot_csv={lot_csv!r}, got {argv[lot_csv_idx + 1]!r}"
        )

        # Fase 3 Task 2: run_id minted once at launch, passed via --run-id,
        # and its discovery_runs PG row created before the Popen spawn.
        assert "--run-id" in argv, f"--run-id not found in Popen argv: {argv}"
        run_id_idx = argv.index("--run-id")
        run_id = argv[run_id_idx + 1]
        assert run_id.startswith("discovery_s0001_lot5_"), run_id
        assert resp.json().get("run_id") == run_id

        db = ExoNitDB(test_db_path)
        try:
            row = db.get_discovery_run(run_id)
            assert row is not None
            # Fase 3 Task 4: confirmed past ENQUEUEING (the fake wrote
            # SCANNING) — this is the new spawn-confirmation contract.
            assert row["state"] == "SCANNING"
            assert row["sector"] == 1
            assert row["lot_csv"] == lot_csv
            # Fase 3 Task 4: closes a real gap — before this task nothing
            # wrote orchestrator_pid into the PG row (only the JSON mirror
            # got it, from the orchestrator's own first _set_state call),
            # which would have silently defeated both the dead-run watchdog
            # and the /api/discovery/resume alive-guard (both key off this
            # column).
            assert row["orchestrator_pid"] == 77777
        finally:
            db.close()

    def test_scan_orchestrator_popen_crash_returns_502_and_marks_failed(
            self, discovery_client, monkeypatch):
        """Fase 3 Task 4: a Popen() exception spawning the orchestrator now
        returns 502 (not the pre-Task-4 silent 200) and marks the
        discovery_runs row FAILED. This closes the 2026-06-30 orphan-run
        incident: previously the code only logged a warning and fell through
        to 200, leaving the row stuck ENQUEUEING forever with nothing ever
        going to work it."""
        client, tmp_path = discovery_client

        lot_csv = "/tmp/s0001_lot9.csv"
        scan_proc = _make_scan_popen("started", {"lot_csv": lot_csv})

        async def _fake_create(*args, **kwargs):
            return scan_proc

        (tmp_path / "pipeline_state.json").unlink(missing_ok=True)

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_create), \
                patch("subprocess.Popen", side_effect=OSError("exec failed")):
            resp = client.post("/api/discovery/scan",
                               json={"sector": 1, "count": 100})

        assert resp.status_code == 502
        assert resp.json() == {"status": "error",
                               "reason": "orchestrator_spawn_failed"}

        state = json.loads((tmp_path / "pipeline_state.json").read_text())
        assert state["state"] == "FAILED"

        from exonit.db import ExoNitDB
        db = ExoNitDB(str(tmp_path / "test.db"))
        try:
            # The Popen exception happens AFTER the discovery_runs row is
            # created (run_id never makes it into the response body since
            # the orchestrator never got a pid), so recover it via the new
            # get_latest_failed_discovery_run() accessor instead.
            row = db.get_latest_failed_discovery_run()
            assert row is not None
            assert row["run_id"].startswith("discovery_s0001_lot9_")
            assert row["detail"] == "orchestrator_spawn_failed"
        finally:
            db.close()

    def test_scan_orchestrator_spawn_timeout_returns_502_and_marks_failed(
            self, discovery_client, monkeypatch):
        """Fase 3 Task 4: if the orchestrator process never moves the
        discovery_runs row off ENQUEUEING within the confirmation window,
        the scan endpoint reports 502 instead of a false-positive 200 — a
        Popen 'succeeding' only means the OS could exec the interpreter, not
        that the script went on to do anything (broken import, PG
        unreachable at ITS OWN startup, etc.)."""
        client, tmp_path = discovery_client
        # Fase 3 Task 4: these two envs are read fresh on every call (not
        # cached at register()/create_app() time), so setting them here,
        # after the fixture already built the app, still takes effect —
        # keeps this test from actually waiting out the 15s/0.5s defaults.
        monkeypatch.setenv("EXONIT_ORCH_SPAWN_CONFIRM_TIMEOUT_SECONDS", "0.2")
        monkeypatch.setenv("EXONIT_ORCH_SPAWN_CONFIRM_POLL_SECONDS", "0.05")

        lot_csv = "/tmp/s0001_lot13.csv"
        scan_proc = _make_scan_popen("started", {"lot_csv": lot_csv})

        async def _fake_create(*args, **kwargs):
            return scan_proc

        (tmp_path / "pipeline_state.json").unlink(missing_ok=True)

        def _popen_dispatch(cmd, **kwargs):
            # Never writes a state transition — the row stays ENQUEUEING,
            # simulating a dead-on-arrival orchestrator process.
            m = MagicMock()
            m.pid = 55555
            return m

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_create), \
                patch("subprocess.Popen", side_effect=_popen_dispatch):
            resp = client.post("/api/discovery/scan",
                               json={"sector": 1, "count": 100})

        assert resp.status_code == 502
        assert resp.json() == {"status": "error",
                               "reason": "orchestrator_spawn_failed"}

        state = json.loads((tmp_path / "pipeline_state.json").read_text())
        assert state["state"] == "FAILED"

        from exonit.db import ExoNitDB
        db = ExoNitDB(str(tmp_path / "test.db"))
        try:
            row = db.get_latest_failed_discovery_run()
            assert row is not None
            assert row["run_id"].startswith("discovery_s0001_lot13_")
            assert row["detail"] == "orchestrator_spawn_failed"
        finally:
            db.close()

    def test_scan_create_discovery_run_failure_returns_502_before_popen(
            self, discovery_client, monkeypatch):
        """F4 (final-review fix, deploy-order hazard): if
        db.create_discovery_run(...) raises (e.g. the discovery_runs table
        doesn't exist yet because schema_pg.sql wasn't applied), the endpoint
        must fail FAST with 502 BEFORE ever spawning the orchestrator —
        not warn-and-continue-to-Popen (the old behavior left a healthy
        orchestrator running with no row for the spawn-confirmation poll to
        ever see, so it timed out and marked the MIRROR failed underneath a
        live process — reopening the anti-double-launch guard)."""
        client, tmp_path = discovery_client

        lot_csv = "/tmp/s0001_lot20.csv"
        scan_proc = _make_scan_popen("started", {"lot_csv": lot_csv})

        async def _fake_create(*args, **kwargs):
            return scan_proc

        (tmp_path / "pipeline_state.json").unlink(missing_ok=True)

        popen_calls = []

        def _popen_dispatch(cmd, **kwargs):
            popen_calls.append(cmd)
            m = MagicMock()
            m.pid = 66666
            return m

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_create), \
                patch("subprocess.Popen", side_effect=_popen_dispatch), \
                patch("exonit.db.ExoNitDB.create_discovery_run",
                      side_effect=Exception("no such table: discovery_runs")):
            resp = client.post("/api/discovery/scan",
                               json={"sector": 1, "count": 100})

        assert resp.status_code == 502
        assert resp.json() == {"status": "error",
                               "reason": "run_record_create_failed"}
        assert popen_calls == [], (
            f"orchestrator must NOT be spawned when the discovery_runs row "
            f"could not be created, got Popen calls: {popen_calls}")

        state = json.loads((tmp_path / "pipeline_state.json").read_text())
        assert state["state"] == "FAILED"

    def test_scan_mints_run_id_from_utc_date(self, discovery_client, monkeypatch):
        """F5 (final-review fix): the scan endpoint's run_id mint must use the
        SAME UTC date source as discovery_orchestrator.py's _derive_run_id
        (datetime.now(timezone.utc)), not local date.today() — a local-vs-UTC
        midnight straddle used to be able to mint a run_id the orchestrator's
        own convention would never independently derive."""
        client, tmp_path = discovery_client

        lot_csv = "/tmp/s0001_lot21.csv"
        scan_proc = _make_scan_popen("started", {"lot_csv": lot_csv})

        async def _fake_create(*args, **kwargs):
            return scan_proc

        (tmp_path / "pipeline_state.json").unlink(missing_ok=True)

        def _popen_dispatch(cmd, **kwargs):
            run_id_idx = cmd.index("--run-id") + 1
            from exonit.db import ExoNitDB
            fake_db = ExoNitDB(str(tmp_path / "test.db"))
            try:
                fake_db.update_discovery_run(cmd[run_id_idx], state="SCANNING")
            finally:
                fake_db.close()
            m = MagicMock()
            m.pid = 88888
            return m

        # A local date far from the UTC date pinned below — if the endpoint
        # ever regresses to date.today() (local), the run_id would carry the
        # WRONG (local) date instead of the UTC one asserted here.
        fake_utc_now = datetime(2026, 7, 6, 0, 30, tzinfo=timezone.utc)
        with patch("asyncio.create_subprocess_exec", side_effect=_fake_create), \
                patch("subprocess.Popen", side_effect=_popen_dispatch), \
                patch("exonit.routes.discovery.datetime") as mock_dt:
            mock_dt.now.return_value = fake_utc_now
            resp = client.post("/api/discovery/scan",
                               json={"sector": 1, "count": 100})

        assert resp.status_code == 200
        run_id = resp.json()["run_id"]
        assert run_id == "discovery_s0001_lot21_2026-07-06", run_id

    def test_scan_prefers_child_minted_run_id_over_rederiving(
            self, discovery_client, monkeypatch):
        """S3 (docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md
        §3.1): once discovery_run.py itself mints run_id (S4/S3) and reports
        it in its JSON, the scan endpoint must use THAT value verbatim
        ("the child wins") rather than re-deriving one — the single source
        of a possible midnight-UTC mismatch this design explicitly removes.
        Proven by making the route's own UTC clock disagree with the
        child's minted run_id's date: if the route ever re-derived instead
        of trusting the child, the assertion below would fail."""
        client, tmp_path = discovery_client

        lot_csv = "/tmp/s0001_lot22.csv"
        child_run_id = "discovery_s0001_lot22_2020-01-01"  # deliberately "wrong" date
        scan_proc = _make_scan_popen(
            "started", {"lot_csv": lot_csv, "run_id": child_run_id})

        async def _fake_create(*args, **kwargs):
            return scan_proc

        (tmp_path / "pipeline_state.json").unlink(missing_ok=True)

        def _popen_dispatch(cmd, **kwargs):
            run_id_idx = cmd.index("--run-id") + 1
            assert cmd[run_id_idx] == child_run_id
            from exonit.db import ExoNitDB
            fake_db = ExoNitDB(str(tmp_path / "test.db"))
            try:
                fake_db.update_discovery_run(cmd[run_id_idx], state="SCANNING")
            finally:
                fake_db.close()
            m = MagicMock()
            m.pid = 99999
            return m

        # The route's OWN clock says a totally different date -- if it ever
        # fell back to re-deriving, the run_id would carry THIS date instead.
        fake_utc_now = datetime(2026, 12, 31, 23, 59, tzinfo=timezone.utc)
        with patch("asyncio.create_subprocess_exec", side_effect=_fake_create), \
                patch("subprocess.Popen", side_effect=_popen_dispatch), \
                patch("exonit.routes.discovery.datetime") as mock_dt:
            mock_dt.now.return_value = fake_utc_now
            resp = client.post("/api/discovery/scan",
                               json={"sector": 1, "count": 100})

        assert resp.status_code == 200
        assert resp.json()["run_id"] == child_run_id

    def test_scan_double_launch_guard_returns_409(self, discovery_client, monkeypatch):
        """When pipeline_state shows an active state, a new scan is refused with
        409 and NEITHER the scan subprocess NOR the orchestrator is spawned.

        (Old behaviour re-enqueued and only skipped the orchestrator — that is
        exactly what let repeated 'Scan' clicks stack concurrent enqueues.)"""
        client, tmp_path = discovery_client

        # The fixture points EXONIT_PIPELINE_STATE_FILE at tmp_path.
        (tmp_path / "pipeline_state.json").write_text(json.dumps({"state": "SCANNING"}))

        created = []

        async def _fake_create(*args, **kwargs):
            created.append(args)
            return _make_scan_popen("started", {"lot_csv": "/tmp/x.csv"})

        orchestrator_calls = []

        def _popen_dispatch(cmd, **kwargs):
            orchestrator_calls.append(cmd)
            return MagicMock(pid=88888)

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_create), \
                patch("subprocess.Popen", side_effect=_popen_dispatch):
            resp = client.post("/api/discovery/scan",
                               json={"sector": 1, "count": 100})

        assert resp.status_code == 409
        assert resp.json()["status"] == "already_running"
        assert created == [], "scan subprocess must NOT run when guard fires"
        assert orchestrator_calls == [], "orchestrator must NOT spawn when guard fires"

    def test_scan_error_sets_pipeline_failed(self, discovery_client, monkeypatch):
        """A failed scan writes FAILED to pipeline_state so the UI bar never
        shows a stale 'DONE' from a previous run (the 'all green' symptom)."""
        client, tmp_path = discovery_client
        state_path = tmp_path / "pipeline_state.json"
        # Pre-existing stale DONE state from a previous run.
        state_path.write_text(json.dumps({"state": "DONE"}))

        self._post_scan(client, monkeypatch, "error")

        new_state = json.loads(state_path.read_text())
        assert new_state["state"] == "FAILED", (
            f"expected FAILED after a failed scan, got {new_state}"
        )


# ---------------------------------------------------------------------------
# /api/discovery/scan — mode=deep_rescan (Task 7)
# ---------------------------------------------------------------------------

class TestScanDeepMode:
    """body.mode == 'deep_rescan' + tics_csv appends --tics-csv <path>
    --force-rescan to the discovery_run.py cmd (Task 6 flags). Unknown mode
    or deep_rescan without tics_csv -> 400, BEFORE the anti-double-launch
    guard flips pipeline state to ENQUEUEING (i.e. no subprocess spawned).
    No mode key -> cmd byte-identical to today.
    """

    def test_deep_mode_passes_flags(self, discovery_client, monkeypatch, tmp_path):
        client, _ = discovery_client
        captured = {}

        async def _fake_create(*args, **kwargs):
            captured["cmd"] = list(args)
            return _make_scan_popen("dry_run")

        wl = tmp_path / "wl.csv"
        wl.write_text("tic_id\n1\n")

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_create):
            resp = client.post("/api/discovery/scan", json={
                "sector": 14, "count": 5000,
                "mode": "deep_rescan", "tics_csv": str(wl)})

        assert resp.status_code == 200
        assert "--tics-csv" in captured["cmd"]
        assert str(wl) in captured["cmd"]
        assert "--force-rescan" in captured["cmd"]

    def test_unknown_mode_400(self, discovery_client):
        client, _ = discovery_client
        created = []

        async def _fake_create(*args, **kwargs):
            created.append(args)
            return _make_scan_popen("dry_run")

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_create):
            resp = client.post("/api/discovery/scan",
                               json={"mode": "banana", "count": 5})

        assert resp.status_code == 400
        data = resp.json()
        assert data.get("status") == "error"
        assert data.get("reason") == "unknown_mode"
        assert created == [], "scan subprocess must not spawn on unknown mode"

    def test_deep_mode_without_csv_400(self, discovery_client):
        client, _ = discovery_client
        created = []

        async def _fake_create(*args, **kwargs):
            created.append(args)
            return _make_scan_popen("dry_run")

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_create):
            resp = client.post("/api/discovery/scan",
                               json={"mode": "deep_rescan", "count": 5})

        assert resp.status_code == 400
        data = resp.json()
        assert data.get("status") == "error"
        assert data.get("reason") == "deep_rescan_requires_tics_csv"
        assert created == [], (
            "scan subprocess must not spawn when tics_csv is missing")

    def test_normal_mode_untouched(self, discovery_client, monkeypatch):
        """No mode key -> cmd carries neither --tics-csv nor --force-rescan
        (byte-identical to the pre-Task-7 cmd)."""
        client, _ = discovery_client
        captured = {}

        async def _fake_create(*args, **kwargs):
            captured["cmd"] = list(args)
            return _make_scan_popen("dry_run")

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_create):
            resp = client.post("/api/discovery/scan",
                               json={"sector": 1, "count": 100})

        assert resp.status_code == 200
        assert "--tics-csv" not in captured["cmd"]
        assert "--force-rescan" not in captured["cmd"]


# ---------------------------------------------------------------------------
# D (2026-10-05): POST /api/discovery/scan — allow_disabled / full_scan body
# fields map onto scripts/discovery_run.py's pre-existing --allow-disabled /
# --full-scan flags (used today by the autofill manager / verify_sector_
# complete.py), so the UI's one-shot Scan button can launch a mop-up lot
# against a CLOSED sector without flipping it back to enabled in the
# registry.
# ---------------------------------------------------------------------------

class TestScanAllowDisabledFullScan:

    def test_allow_disabled_and_full_scan_passed_through(self, discovery_client):
        client, _ = discovery_client
        captured = {}

        async def _fake_create(*args, **kwargs):
            captured["cmd"] = list(args)
            return _make_scan_popen("dry_run")

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_create):
            resp = client.post("/api/discovery/scan", json={
                "sector": 15, "count": 2500,
                "allow_disabled": True, "full_scan": True})

        assert resp.status_code == 200, resp.text
        assert "--allow-disabled" in captured["cmd"]
        assert "--full-scan" in captured["cmd"]

    def test_absent_allow_disabled_full_scan_omits_flags(self, discovery_client):
        """Default behaviour (both fields absent) must be byte-identical to
        before this change — no new flags on the cmd line."""
        client, _ = discovery_client
        captured = {}

        async def _fake_create(*args, **kwargs):
            captured["cmd"] = list(args)
            return _make_scan_popen("dry_run")

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_create):
            resp = client.post("/api/discovery/scan",
                               json={"sector": 1, "count": 100})

        assert resp.status_code == 200
        assert "--allow-disabled" not in captured["cmd"]
        assert "--full-scan" not in captured["cmd"]

    def test_explicit_false_omits_flags(self, discovery_client):
        client, _ = discovery_client
        captured = {}

        async def _fake_create(*args, **kwargs):
            captured["cmd"] = list(args)
            return _make_scan_popen("dry_run")

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_create):
            resp = client.post("/api/discovery/scan", json={
                "sector": 1, "count": 100,
                "allow_disabled": False, "full_scan": False})

        assert resp.status_code == 200
        assert "--allow-disabled" not in captured["cmd"]
        assert "--full-scan" not in captured["cmd"]

    def test_non_bool_allow_disabled_returns_400(self, discovery_client):
        client, _ = discovery_client
        created = []

        async def _fake_create(*args, **kwargs):
            created.append(args)
            return _make_scan_popen("dry_run")

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_create):
            resp = client.post("/api/discovery/scan", json={
                "sector": 1, "count": 100, "allow_disabled": "yes"})

        assert resp.status_code == 400
        data = resp.json()
        assert data.get("status") == "error"
        assert data.get("reason") == "allow_disabled_and_full_scan_must_be_bool"
        assert created == [], "scan subprocess must not spawn on a bad type"

    def test_non_bool_full_scan_returns_400(self, discovery_client):
        client, _ = discovery_client
        created = []

        async def _fake_create(*args, **kwargs):
            created.append(args)
            return _make_scan_popen("dry_run")

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_create):
            # 1 is an int, not a bool -- isinstance(1, bool) is False even
            # though JSON/Python both treat it as truthy.
            resp = client.post("/api/discovery/scan", json={
                "sector": 1, "count": 100, "full_scan": 1})

        assert resp.status_code == 400
        assert resp.json().get("reason") == "allow_disabled_and_full_scan_must_be_bool"
        assert created == []

    def test_allow_disabled_with_deep_rescan_returns_400(
            self, discovery_client, tmp_path):
        """deep_rescan already has its own enable/force semantics
        (--force-rescan); combining it with allow_disabled/full_scan is
        rejected outright rather than guessing at an untested interaction."""
        client, _ = discovery_client
        created = []

        async def _fake_create(*args, **kwargs):
            created.append(args)
            return _make_scan_popen("dry_run")

        wl = tmp_path / "wl.csv"
        wl.write_text("tic_id\n1\n")

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_create):
            resp = client.post("/api/discovery/scan", json={
                "sector": 14, "count": 5000,
                "mode": "deep_rescan", "tics_csv": str(wl),
                "allow_disabled": True})

        assert resp.status_code == 400
        data = resp.json()
        assert data.get("status") == "error"
        assert data.get("reason") == (
            "allow_disabled_full_scan_incompatible_with_deep_rescan")
        assert created == []


# ---------------------------------------------------------------------------
# B (2026-10-05): a mop-up lot started against a CLOSED sector must also
# invalidate/recompute THAT sector's snapshot in the SEPARATE closed-sector
# gap-stats store (exonit/routes/discovery.py's _CLOSED_GAP_STATS_* block) —
# the live gap_data store's force-refresh is a no-op for a closed sector by
# design, so without this the mop-up lot's progress would sit invisible for
# up to the closed store's own 24 h TTL.
# ---------------------------------------------------------------------------

class TestScanTriggersClosedGapStatsRefresh:
    def _wait_for(self, predicate, timeout=2.0, interval=0.02):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(interval)
        return predicate()

    def test_scan_on_closed_sector_refreshes_closed_store(
        self, discovery_client, tmp_path,
    ):
        client, _ = discovery_client
        reg = {"active_sector": 1, "sectors": {"15": {
            "enabled": False, "total_tics": 2564123, "downloaded": 2564123,
            "cursor": 2564111, "canonical_list": str(tmp_path / "no_canon.csv"),
        }}}
        fake_gap = {"gaps_processable": 8, "no_data": 2, "completed": 2564113,
                    "work_list_size": 2564123}

        async def _fake_create(*args, **kwargs):
            return _make_scan_popen("dry_run")

        from exonit import discovery_state
        from exonit.shared_state import get_shared_state
        st = get_shared_state()

        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "get_sector_state",
                          return_value="complete"), \
             patch.object(discovery_state, "get_autofill", return_value=None), \
             patch.object(discovery_state, "sector_gap_stats",
                          return_value=fake_gap) as m_gap, \
             patch("asyncio.create_subprocess_exec", side_effect=_fake_create):
            resp = client.post("/api/discovery/scan", json={
                "sector": 15, "count": 2500, "allow_disabled": True})

            assert resp.status_code == 200, resp.text
            assert self._wait_for(lambda: m_gap.called)
            assert self._wait_for(
                lambda: st.get("closed_gap_stats:data") is not None
                and "15" in json.loads(st.get("closed_gap_stats:data"))["data"])

        cached = json.loads(st.get("closed_gap_stats:data"))["data"]["15"]
        assert cached["gaps_processable"] == 8

    def test_scan_on_open_sector_does_not_touch_closed_store(
        self, discovery_client, tmp_path,
    ):
        """An enabled/active sector's lot must never write to the closed
        store — that store is reserved for closed sectors only."""
        client, _ = discovery_client
        reg = {"active_sector": 1, "sectors": {"1": {
            "enabled": True, "total_tics": 1000, "downloaded": 1000,
            "cursor": 500, "canonical_list": str(tmp_path / "no_canon.csv"),
        }}}

        async def _fake_create(*args, **kwargs):
            return _make_scan_popen("dry_run")

        from exonit import discovery_state
        from exonit.shared_state import get_shared_state
        st = get_shared_state()

        # sector 1 is "moving" (enabled, active) -- the LIVE gap_stats store
        # legitimately calls sector_gap_stats for it; only the CLOSED store
        # must stay untouched, which is what this test actually checks.
        with patch.object(discovery_state, "load_registry", return_value=reg), \
             patch.object(discovery_state, "get_sector_state",
                          return_value="active"), \
             patch.object(discovery_state, "sector_gap_stats",
                          return_value={"gaps_processable": 500, "no_data": 0,
                                        "completed": 500, "work_list_size": 1000}), \
             patch("asyncio.create_subprocess_exec", side_effect=_fake_create):
            resp = client.post("/api/discovery/scan",
                               json={"sector": 1, "count": 100})
            assert resp.status_code == 200, resp.text
            time.sleep(0.2)  # give any stray background thread a chance

        assert st.get("closed_gap_stats:data") is None


# ---------------------------------------------------------------------------
# Dead-run watchdog (Fase 3 Task 4)
# ---------------------------------------------------------------------------

class TestDiscoveryWatchdog:
    """exercises app.state.discovery_watchdog_check_once() directly — a
    single deterministic pass, instead of racing the real 60s-interval
    background thread."""

    def _mk_run(self, db, run_id="discovery_s0001_lot1_2026-07-05"):
        db.create_discovery_run(run_id, sector=1,
                                lot_csv="/tmp/s0001_lot1.csv",
                                dry_run=False, chained=False,
                                server_url="http://localhost:8400")
        return run_id

    def test_watchdog_marks_dead_pid_stale_heartbeat_failed(
            self, discovery_client, monkeypatch):
        client, tmp_path = discovery_client
        app = client.app

        from exonit.db import ExoNitDB
        db = ExoNitDB(str(tmp_path / "test.db"))
        try:
            run_id = self._mk_run(db)
            db.update_discovery_run(run_id, state="SCANNING", pid=424242)
            # Force heartbeat_at/updated_at into the past, well beyond the
            # default 180s staleness threshold, without waiting for real time
            # to pass.
            stale_iso = (datetime.now(timezone.utc)
                         - timedelta(seconds=999)).isoformat()
            db._exec(
                "UPDATE discovery_runs SET heartbeat_at = ?, updated_at = ? "
                "WHERE run_id = ?",
                (stale_iso, stale_iso, run_id))

            with patch("os.kill", side_effect=ProcessLookupError):
                app.state.discovery_watchdog_check_once()

            row = db.get_discovery_run(run_id)
            assert row["state"] == "FAILED"
            assert "orchestrator_died" in row["detail"]
            assert "SCANNING" in row["detail"]
        finally:
            db.close()

        state = json.loads((tmp_path / "pipeline_state.json").read_text())
        assert state["state"] == "FAILED"

    def test_watchdog_leaves_fresh_heartbeat_run_alone(
            self, discovery_client, monkeypatch):
        """Same dead pid, but a heartbeat from just now — must NOT be
        touched (the orchestrator may simply be between polls; only a dead
        pid AND a stale heartbeat together indicate a truly dead run)."""
        client, tmp_path = discovery_client
        app = client.app

        from exonit.db import ExoNitDB
        db = ExoNitDB(str(tmp_path / "test.db"))
        try:
            run_id = self._mk_run(db, "discovery_s0001_lot2_2026-07-05")
            db.update_discovery_run(run_id, state="SCANNING", pid=424242,
                                    heartbeat=True)

            with patch("os.kill", side_effect=ProcessLookupError):
                app.state.discovery_watchdog_check_once()

            row = db.get_discovery_run(run_id)
            assert row["state"] == "SCANNING", (
                "a run with a fresh heartbeat must not be marked FAILED "
                "even if its pid looks dead"
            )
        finally:
            db.close()

    def test_watchdog_leaves_alive_pid_alone(self, discovery_client, monkeypatch):
        """Dead-looking heartbeat but the pid is alive — must not be touched."""
        client, tmp_path = discovery_client
        app = client.app

        from exonit.db import ExoNitDB
        db = ExoNitDB(str(tmp_path / "test.db"))
        try:
            run_id = self._mk_run(db, "discovery_s0001_lot3_2026-07-05")
            db.update_discovery_run(run_id, state="SCANNING", pid=424242)
            stale_iso = (datetime.now(timezone.utc)
                         - timedelta(seconds=999)).isoformat()
            db._exec(
                "UPDATE discovery_runs SET heartbeat_at = ?, updated_at = ? "
                "WHERE run_id = ?",
                (stale_iso, stale_iso, run_id))

            with patch("os.kill", return_value=None):  # pid alive
                app.state.discovery_watchdog_check_once()

            row = db.get_discovery_run(run_id)
            assert row["state"] == "SCANNING"
        finally:
            db.close()

    def test_watchdog_ignores_dry_run(self, discovery_client, monkeypatch):
        client, tmp_path = discovery_client
        app = client.app

        from exonit.db import ExoNitDB
        db = ExoNitDB(str(tmp_path / "test.db"))
        try:
            run_id = "discovery_s0001_lot4_2026-07-05"
            db.create_discovery_run(run_id, sector=1,
                                    lot_csv="/tmp/s0001_lot4.csv",
                                    dry_run=True, chained=False,
                                    server_url="http://localhost:8400")
            db.update_discovery_run(run_id, state="SCANNING", pid=424242)
            stale_iso = (datetime.now(timezone.utc)
                         - timedelta(seconds=999)).isoformat()
            db._exec(
                "UPDATE discovery_runs SET heartbeat_at = ?, updated_at = ? "
                "WHERE run_id = ?",
                (stale_iso, stale_iso, run_id))

            with patch("os.kill", side_effect=ProcessLookupError):
                app.state.discovery_watchdog_check_once()

            row = db.get_discovery_run(run_id)
            assert row["state"] == "SCANNING", "dry_run rows must be excluded"
        finally:
            db.close()

    def test_watchdog_no_active_run_is_a_noop(self, discovery_client):
        """No discovery_runs row at all — must not raise."""
        client, tmp_path = discovery_client
        app = client.app
        app.state.discovery_watchdog_check_once()  # must not raise


# ---------------------------------------------------------------------------
# POST /api/discovery/resume (Fase 3 Task 4)
# ---------------------------------------------------------------------------

def _delayed_db_update(db_path, run_id, delay=0.05, **kwargs):
    """Spawn a background thread that calls update_discovery_run after
    *delay* seconds — simulates a REAL resumed orchestrator subprocess's own
    (asynchronous, independent-of-the-parent) first act, as opposed to a
    Popen mock's side_effect running synchronously inside the endpoint's own
    call stack.

    F3 (final-review fix): _confirm_orchestrator_resumed's baseline is read
    AFTER the endpoint's own "record the new pid" write, specifically so
    that write alone can never satisfy the confirmation (see its docstring).
    A resume test whose fake orchestrator writes ITS transition synchronously
    inside the Popen mock — i.e. before that baseline is even read — would
    have its write silently swallowed into the baseline and then time out.
    Using a real background thread with a short delay instead reproduces the
    actual ordering a real subprocess has: independent progress that happens
    AFTER the endpoint has finished its own bookkeeping.
    """
    from exonit.db import ExoNitDB

    def _write():
        time.sleep(delay)
        fake_db = ExoNitDB(db_path)
        try:
            fake_db.update_discovery_run(run_id, **kwargs)
        finally:
            fake_db.close()

    threading.Thread(target=_write, daemon=True).start()


class TestDiscoveryResume:

    def test_resume_returns_404_when_nothing_to_resume(self, discovery_client):
        client, _ = discovery_client
        resp = client.post("/api/discovery/resume", json={})
        assert resp.status_code == 404
        assert resp.json()["reason"] == "nothing_to_resume"

    def test_resume_returns_404_for_unknown_explicit_run_id(self, discovery_client):
        client, _ = discovery_client
        resp = client.post("/api/discovery/resume", json={"run_id": "nope"})
        assert resp.status_code == 404
        assert resp.json()["reason"] == "not_found"

    def test_resume_returns_409_when_orchestrator_alive(
            self, discovery_client, monkeypatch):
        """A FAILED run whose recorded orchestrator_pid is still alive (a
        stale/incorrect pid, or a race) must never be double-spawned."""
        client, tmp_path = discovery_client

        from exonit.db import ExoNitDB
        db = ExoNitDB(str(tmp_path / "test.db"))
        try:
            run_id = "discovery_s0001_lot6_2026-07-05"
            db.create_discovery_run(run_id, sector=1,
                                    lot_csv="/tmp/s0001_lot6.csv",
                                    dry_run=False, chained=False,
                                    server_url="http://localhost:8400")
            db.update_discovery_run(run_id, state="FAILED", pid=13579,
                                    detail="fpp_backfill poll failed")

            with patch("os.kill", return_value=None):  # pid alive
                resp = client.post("/api/discovery/resume",
                                   json={"run_id": run_id})

            assert resp.status_code == 409
            assert resp.json()["reason"] == "orchestrator_alive"
        finally:
            db.close()

    def test_resume_returns_409_for_done_run(self, discovery_client):
        client, tmp_path = discovery_client
        from exonit.db import ExoNitDB
        db = ExoNitDB(str(tmp_path / "test.db"))
        try:
            run_id = "discovery_s0001_lot7_2026-07-05"
            db.create_discovery_run(run_id, sector=1,
                                    lot_csv="/tmp/s0001_lot7.csv",
                                    dry_run=False, chained=False,
                                    server_url="http://localhost:8400")
            db.update_discovery_run(run_id, state="DONE", finished=True)

            resp = client.post("/api/discovery/resume", json={"run_id": run_id})
            assert resp.status_code == 409
            assert resp.json()["reason"] == "already_done"
        finally:
            db.close()

    def test_resume_failed_run_spawns_with_resume_and_force_argv(
            self, discovery_client, monkeypatch):
        """200 + the orchestrator argv contains --resume <run_id> and
        --force (since the row is FAILED). F3 (final-review fix): the fake
        resumed process now must make PROGRESS (a further row update) via a
        background thread AFTER the endpoint's own pid-write, mimicking F1's
        own pid/heartbeat stamp — a state write happening synchronously
        inside the Popen mock itself (the pre-F3 shape of this test) would
        be swallowed into _confirm_orchestrator_resumed's baseline and time
        out (see _delayed_db_update's docstring)."""
        client, tmp_path = discovery_client
        monkeypatch.setenv("EXONIT_ORCH_SPAWN_CONFIRM_POLL_SECONDS", "0.05")

        from exonit.db import ExoNitDB
        test_db_path = str(tmp_path / "test.db")
        db = ExoNitDB(test_db_path)
        try:
            run_id = "discovery_s0001_lot8_2026-07-05"
            db.create_discovery_run(run_id, sector=1,
                                    lot_csv="/tmp/s0001_lot8.csv",
                                    dry_run=False, chained=False,
                                    server_url="http://localhost:8400")
            db.update_discovery_run(run_id, state="FAILED", pid=24680,
                                    detail="fpp_backfill poll failed")
        finally:
            db.close()

        orchestrator_calls = []

        def _popen_dispatch(cmd, **kwargs):
            orchestrator_calls.append(cmd)
            resume_idx = cmd.index("--resume") + 1
            _delayed_db_update(test_db_path, cmd[resume_idx], delay=0.1,
                               state="SCANNING")
            return MagicMock(pid=99001)

        with patch("os.kill", side_effect=ProcessLookupError), \
                patch("subprocess.Popen", side_effect=_popen_dispatch):
            resp = client.post("/api/discovery/resume", json={"run_id": run_id})

        assert resp.status_code == 200
        assert resp.json() == {"status": "resumed", "run_id": run_id}
        assert len(orchestrator_calls) == 1
        argv = orchestrator_calls[0]
        assert "--resume" in argv
        assert argv[argv.index("--resume") + 1] == run_id
        assert "--force" in argv, "a FAILED row must be resumed with --force"

        db = ExoNitDB(test_db_path)
        try:
            row = db.get_discovery_run(run_id)
            assert row["orchestrator_pid"] == 99001, (
                "a resumed run's orchestrator_pid must point at the NEW "
                "process (99001), not the dead one it replaced (24680)"
            )
        finally:
            db.close()

    def test_resume_default_candidate_prefers_active_over_failed(
            self, discovery_client, monkeypatch):
        """No explicit run_id: an active (non-terminal) run takes priority
        over an older FAILED one, and does NOT get --force (only FAILED
        rows do). F3: fake progress happens via a background thread (see
        _delayed_db_update) so it lands AFTER the confirmation baseline."""
        client, tmp_path = discovery_client
        monkeypatch.setenv("EXONIT_ORCH_SPAWN_CONFIRM_POLL_SECONDS", "0.05")

        from exonit.db import ExoNitDB
        test_db_path = str(tmp_path / "test.db")
        db = ExoNitDB(test_db_path)
        try:
            failed_id = "discovery_s0001_lot10_2026-07-05"
            db.create_discovery_run(failed_id, sector=1,
                                    lot_csv="/tmp/s0001_lot10.csv",
                                    dry_run=False, chained=False,
                                    server_url="http://localhost:8400")
            db.update_discovery_run(failed_id, state="FAILED", pid=1,
                                    detail="boom")

            active_id = "discovery_s0001_lot11_2026-07-05"
            db.create_discovery_run(active_id, sector=1,
                                    lot_csv="/tmp/s0001_lot11.csv",
                                    dry_run=False, chained=False,
                                    server_url="http://localhost:8400")
            db.update_discovery_run(active_id, state="SCANNING", pid=2)
        finally:
            db.close()

        orchestrator_calls = []

        def _popen_dispatch(cmd, **kwargs):
            orchestrator_calls.append(cmd)
            resume_idx = cmd.index("--resume") + 1
            _delayed_db_update(test_db_path, cmd[resume_idx], delay=0.1,
                               state="FPP_BACKFILL")
            return MagicMock(pid=99002)

        with patch("os.kill", side_effect=ProcessLookupError), \
                patch("subprocess.Popen", side_effect=_popen_dispatch):
            resp = client.post("/api/discovery/resume", json={})

        assert resp.status_code == 200
        assert resp.json() == {"status": "resumed", "run_id": active_id}
        argv = orchestrator_calls[0]
        assert argv[argv.index("--resume") + 1] == active_id
        assert "--force" not in argv, "a non-FAILED row must NOT get --force"

    def test_resume_of_non_terminal_row_with_doa_process_returns_502(
            self, discovery_client, monkeypatch):
        """F3 (final-review fix): the OLD state-based confirmation
        (_confirm_orchestrator_started(run_id, {"FAILED", "ENQUEUEING"}))
        was VACUOUS here — a row stuck at SCANNING is already outside that
        stale-set, so it would have returned True on the very FIRST poll
        with zero evidence the resumed process is alive. The fake Popen here
        never makes any further progress (no thread, no write) — simulating
        a DOA process (Popen's exec succeeds, but the script crashes/exits
        before doing anything) — so the NEW progress-based confirmation
        must time out and report 502."""
        client, tmp_path = discovery_client
        monkeypatch.setenv("EXONIT_ORCH_SPAWN_CONFIRM_TIMEOUT_SECONDS", "0.3")
        monkeypatch.setenv("EXONIT_ORCH_SPAWN_CONFIRM_POLL_SECONDS", "0.05")

        from exonit.db import ExoNitDB
        test_db_path = str(tmp_path / "test.db")
        db = ExoNitDB(test_db_path)
        try:
            run_id = "discovery_s0001_lot22_2026-07-05"
            db.create_discovery_run(run_id, sector=1,
                                    lot_csv="/tmp/s0001_lot22.csv",
                                    dry_run=False, chained=False,
                                    server_url="http://localhost:8400")
            db.update_discovery_run(run_id, state="SCANNING", pid=13131)
        finally:
            db.close()

        def _popen_dispatch(cmd, **kwargs):
            # DOA: no further row update, ever.
            return MagicMock(pid=14141)

        with patch("os.kill", side_effect=ProcessLookupError), \
                patch("subprocess.Popen", side_effect=_popen_dispatch):
            resp = client.post("/api/discovery/resume", json={"run_id": run_id})

        assert resp.status_code == 502
        assert resp.json() == {"status": "error",
                               "reason": "orchestrator_spawn_failed"}

        db = ExoNitDB(test_db_path)
        try:
            row = db.get_discovery_run(run_id)
            assert row["state"] == "FAILED"
            assert row["detail"] == "orchestrator_spawn_failed"
        finally:
            db.close()

    def test_resume_of_non_terminal_row_with_heartbeat_returns_200(
            self, discovery_client, monkeypatch):
        """F3 companion: same non-terminal (SCANNING) starting row as the DOA
        test above, but the fake resumed process DOES stamp a heartbeat
        shortly after spawn (mimicking discovery_orchestrator.py's F1 own-
        pid/heartbeat write) — the progress-based confirmation must see
        this and return 200."""
        client, tmp_path = discovery_client
        monkeypatch.setenv("EXONIT_ORCH_SPAWN_CONFIRM_POLL_SECONDS", "0.05")

        from exonit.db import ExoNitDB
        test_db_path = str(tmp_path / "test.db")
        db = ExoNitDB(test_db_path)
        try:
            run_id = "discovery_s0001_lot23_2026-07-05"
            db.create_discovery_run(run_id, sector=1,
                                    lot_csv="/tmp/s0001_lot23.csv",
                                    dry_run=False, chained=False,
                                    server_url="http://localhost:8400")
            db.update_discovery_run(run_id, state="SCANNING", pid=15151)
        finally:
            db.close()

        def _popen_dispatch(cmd, **kwargs):
            resume_idx = cmd.index("--resume") + 1
            _delayed_db_update(test_db_path, cmd[resume_idx], delay=0.1,
                               heartbeat=True)
            return MagicMock(pid=16161)

        with patch("os.kill", side_effect=ProcessLookupError), \
                patch("subprocess.Popen", side_effect=_popen_dispatch):
            resp = client.post("/api/discovery/resume", json={"run_id": run_id})

        assert resp.status_code == 200
        assert resp.json() == {"status": "resumed", "run_id": run_id}

    def test_resume_popen_crash_returns_502(self, discovery_client, monkeypatch):
        client, tmp_path = discovery_client
        from exonit.db import ExoNitDB
        db = ExoNitDB(str(tmp_path / "test.db"))
        try:
            run_id = "discovery_s0001_lot12_2026-07-05"
            db.create_discovery_run(run_id, sector=1,
                                    lot_csv="/tmp/s0001_lot12.csv",
                                    dry_run=False, chained=False,
                                    server_url="http://localhost:8400")
            db.update_discovery_run(run_id, state="FAILED", pid=9,
                                    detail="boom")
        finally:
            db.close()

        with patch("os.kill", side_effect=ProcessLookupError), \
                patch("subprocess.Popen", side_effect=OSError("exec failed")):
            resp = client.post("/api/discovery/resume", json={"run_id": run_id})

        assert resp.status_code == 502
        assert resp.json() == {"status": "error",
                               "reason": "orchestrator_spawn_failed"}

        db = ExoNitDB(str(tmp_path / "test.db"))
        try:
            row = db.get_discovery_run(run_id)
            assert row["state"] == "FAILED"
            assert row["detail"] == "orchestrator_spawn_failed"
        finally:
            db.close()


# ---------------------------------------------------------------------------
# /api/discovery/download — start + status
# ---------------------------------------------------------------------------

class TestDiscoveryDownload:

    def _fake_popen(self, tmp_path):
        """Return a mock Popen handle that looks 'running'."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None  # still running
        mock_proc.returncode = None
        return mock_proc

    def test_download_start_returns_started(self, discovery_client, monkeypatch):
        """POST /api/discovery/download returns {status: started, pid: N}."""
        client, tmp_path = discovery_client
        mock_proc = self._fake_popen(tmp_path)

        with patch("subprocess.Popen", return_value=mock_proc):
            resp = client.post("/api/discovery/download",
                               json={"sector": 1, "count": 500})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        assert "pid" in data

    def test_download_already_running_returns_409(self, discovery_client, monkeypatch):
        """Second start while first is running → 409.

        The guard now reads from shared_state and checks pid liveness via
        psutil, so we patch psutil to report the pid as alive.
        """
        client, tmp_path = discovery_client
        mock_proc = self._fake_popen(tmp_path)

        mock_psutil_proc = MagicMock()
        mock_psutil_proc.cmdline.return_value = ["python", "discovery_download.py"]
        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("psutil.pid_exists", return_value=True), \
             patch("psutil.Process", return_value=mock_psutil_proc):
            r1 = client.post("/api/discovery/download", json={"sector": 1, "count": 100})
            assert r1.status_code == 200
            r2 = client.post("/api/discovery/download", json={"sector": 1, "count": 100})
        assert r2.status_code == 409
        assert r2.json()["status"] == "already_running"

    def test_download_status_never_run(self, discovery_client):
        """GET /api/discovery/download/status before any download → never_run."""
        client, _ = discovery_client
        resp = client.get("/api/discovery/download/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is False
        assert data.get("never_run") is True

    def test_download_status_while_running(self, discovery_client):
        """Status endpoint shows running=True after start."""
        client, tmp_path = discovery_client
        mock_proc = self._fake_popen(tmp_path)

        # Patch psutil so the status endpoint sees the pid as alive and
        # belonging to the download script (cross-process running check).
        mock_psutil_proc = MagicMock()
        mock_psutil_proc.cmdline.return_value = ["python", "discovery_download.py"]
        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("psutil.pid_exists", return_value=True), \
             patch("psutil.Process", return_value=mock_psutil_proc):
            client.post("/api/discovery/download", json={"sector": 1, "count": 100})
            resp = client.get("/api/discovery/download/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is True
        assert data.get("never_run") is None or not data.get("never_run")

    def test_download_status_finished(self, discovery_client):
        """Status shows running=False + returncode when process exits."""
        client, tmp_path = discovery_client
        mock_proc = MagicMock()
        mock_proc.pid = 99
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0  # exited

        # Patch psutil so the status endpoint sees the pid as dead.
        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("psutil.pid_exists", return_value=False):
            client.post("/api/discovery/download", json={"sector": 1, "count": 100})
            resp = client.get("/api/discovery/download/status")

        data = resp.json()
        assert data["running"] is False
        # returncode comes from the local proc fast-path (pid matches).
        assert data["returncode"] == 0


# ---------------------------------------------------------------------------
# /api/discovery/download/status — log parsing
# ---------------------------------------------------------------------------

class TestDownloadStatusLogParsing:
    """Unit-level tests for the log-parsing logic inside api_discovery_download_status.

    We drive the endpoint by starting a fake download that points to a log file
    we control, then assert that the returned done/total/failed/skipped values
    match the last progress line in the log.
    """

    def _start_with_log(self, client, tmp_path, log_lines):
        """Write log_lines to a temp file, start a fake-finished download that
        points at that file, and return the status response JSON."""
        log_path = tmp_path / "dl.log"
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        # Mock a process that has already exited (returncode=0, poll()=0)
        mock_proc = MagicMock()
        mock_proc.pid = 42
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0

        with patch("subprocess.Popen", return_value=mock_proc):
            r = client.post("/api/discovery/download", json={"sector": 1, "count": 500})
            assert r.status_code == 200, r.text

        # Overwrite log_path in shared_state so the status endpoint reads
        # from our controlled file.  The cross-process metadata is stored in
        # shared_state (not in the local _discovery_download dict anymore).
        from exonit.shared_state import get_shared_state
        st = get_shared_state()
        existing = st.hget_json("discovery:download:state", "current") or {}
        existing["log_path"] = str(log_path)
        st.hset_json("discovery:download:state", "current", existing)

        return client.get("/api/discovery/download/status").json()

    def _start_with_log_running(self, client, tmp_path, log_lines):
        """Like _start_with_log, but the pid is reported alive (running=True)
        for the status GET — needed to exercise the "running" branch of the
        phase derivation (fetching_script / selecting / downloading)."""
        log_path = tmp_path / "dl.log"
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        mock_proc = MagicMock()
        mock_proc.pid = 42
        mock_proc.returncode = None
        mock_proc.poll.return_value = None  # still running

        with patch("subprocess.Popen", return_value=mock_proc):
            r = client.post("/api/discovery/download", json={"sector": 1, "count": 500})
            assert r.status_code == 200, r.text

        from exonit.shared_state import get_shared_state
        st = get_shared_state()
        existing = st.hget_json("discovery:download:state", "current") or {}
        existing["log_path"] = str(log_path)
        st.hset_json("discovery:download:state", "current", existing)

        mock_psutil_proc = MagicMock()
        mock_psutil_proc.cmdline.return_value = ["python", "discovery_download.py"]
        with patch("psutil.pid_exists", return_value=True), \
             patch("psutil.Process", return_value=mock_psutil_proc):
            return client.get("/api/discovery/download/status").json()

    # ------------------------------------------------------------------
    # Phase derivation for the muted download-prep phases (2026-07-04):
    # bulk-script fetch + batch selection, previously invisible while running.
    # ------------------------------------------------------------------

    def test_phase_fetching_script_with_bytes(self, discovery_client):
        """bulk_fetch_start + bulk_fetch markers, still running →
        phase=fetching_script with script_bytes_done/total surfaced."""
        client, tmp_path = discovery_client
        lines = [
            '{"status": "bulk_fetch_start", "bytes_total": 2420000000, '
            '"url": "https://example.com/s0014.sh"}',
            '{"status": "bulk_fetch", "bytes_done": 500000000, '
            '"bytes_total": 2420000000}',
        ]
        data = self._start_with_log_running(client, tmp_path, lines)
        assert data["phase"] == "fetching_script"
        assert data["script_bytes_done"] == 500000000
        assert data["script_bytes_total"] == 2420000000

    def test_phase_fetching_script_unknown_total(self, discovery_client):
        """HEAD probe failed (bytes_total=None, fail-open) → still reports
        phase=fetching_script, with script_bytes_total null."""
        client, tmp_path = discovery_client
        lines = [
            '{"status": "bulk_fetch_start", "bytes_total": null, '
            '"url": "https://example.com/s0014.sh"}',
            '{"status": "bulk_fetch", "bytes_done": 42, "bytes_total": null}',
        ]
        data = self._start_with_log_running(client, tmp_path, lines)
        assert data["phase"] == "fetching_script"
        assert data["script_bytes_done"] == 42
        assert data["script_bytes_total"] is None

    def test_phase_fetching_script_survives_200_line_tail_window(self, discovery_client):
        """Regression: a long-running fetch (e.g. sector 14, 2.42 GB, 25-40
        min at the default 5s poll interval) emits a bulk_fetch_start line
        ONCE, followed by hundreds of periodic bulk_fetch lines. The status
        endpoint only tails the last 200 log lines — once more than ~200
        bulk_fetch lines have accumulated, the single bulk_fetch_start line
        rolls out of that window. phase must still report fetching_script
        (derived from ANY surviving bulk-fetch-family line, not specifically
        bulk_fetch_start), with byte counters reflecting the LAST bulk_fetch
        line seen — not silently revert to phase=downloading/done=0/total=0
        (the exact frozen-UI bug this feature fixes)."""
        client, tmp_path = discovery_client
        lines = [
            '{"status": "bulk_fetch_start", "bytes_total": 2420000000, '
            '"url": "https://example.com/s0014.sh"}',
        ]
        # 399 periodic bulk_fetch lines simulating an active ~33-minute fetch
        # (5s poll interval) — well past the 200-line tail window, so the
        # bulk_fetch_start line above has definitely aged out.
        n = 399
        for i in range(1, n + 1):
            bytes_done = int(2420000000 * i / n)
            lines.append(
                '{"status": "bulk_fetch", "bytes_done": %d, '
                '"bytes_total": 2420000000}' % bytes_done
            )
        data = self._start_with_log_running(client, tmp_path, lines)
        assert data["phase"] == "fetching_script", (
            f"bulk_fetch_start aged out of the 200-line tail window — "
            f"got phase={data['phase']!r}, expected fetching_script"
        )
        # Byte counters must reflect the LAST bulk_fetch line, not the first.
        assert data["script_bytes_done"] == 2420000000
        assert data["script_bytes_total"] == 2420000000

    def test_phase_selecting(self, discovery_client):
        """select_start marker seen, still running, no per-file progress
        line yet → phase=selecting."""
        client, tmp_path = discovery_client
        lines = ['{"status": "select_start"}']
        data = self._start_with_log_running(client, tmp_path, lines)
        assert data["phase"] == "selecting"

    def test_phase_selecting_then_progress_line_is_downloading(self, discovery_client):
        """select_start followed by a [N/M] progress line → phase=downloading:
        the per-file progress line takes priority over the earlier marker."""
        client, tmp_path = discovery_client
        lines = [
            '{"status": "select_start"}',
            "[100/5000] done=100 skip=0 fail=0 5.0/s eta=10min",
        ]
        data = self._start_with_log_running(client, tmp_path, lines)
        assert data["phase"] == "downloading"

    def test_old_log_without_new_markers_is_unchanged(self, discovery_client):
        """Regression: an old-format log with none of the new markers behaves
        exactly as before — phase=downloading, script_bytes_* stay null."""
        client, tmp_path = discovery_client
        lines = ["Starting download of sector 1 …"]
        data = self._start_with_log_running(client, tmp_path, lines)
        assert data["phase"] == "downloading"
        assert data["script_bytes_done"] is None
        assert data["script_bytes_total"] is None

    def test_real_format_progress_line(self, discovery_client):
        """Parses the actual download_tglc_batch.py progress line format correctly."""
        client, tmp_path = discovery_client
        lines = [
            "Starting download of sector 1 …",
            "[100/5000] done=100 skip=2 fail=1 4.5/s eta=10.2min",
            "[500/5000] done=500 skip=4 fail=2 5.1/s eta=8.7min",
            "[1000/5000] done=1000 skip=5 fail=3 5.3/s eta=7.1min",
        ]
        data = self._start_with_log(client, tmp_path, lines)
        assert data["done"]    == 1000
        assert data["total"]   == 5000
        assert data["skipped"] == 5
        assert data["failed"]  == 3

    def test_finish_line_parsed(self, discovery_client):
        """[finish] line is parsed even though it has no [N/M] prefix."""
        client, tmp_path = discovery_client
        lines = [
            "[100/5000] done=100 skip=0 fail=0 5.0/s eta=10min",
            "[finish] done=5000 skip=12 fail=7 in 4.2min",
        ]
        data = self._start_with_log(client, tmp_path, lines)
        # done/skipped/failed come from [finish]; total came from last [N/M] line
        assert data["done"]    == 5000
        assert data["skipped"] == 12
        assert data["failed"]  == 7
        assert data["total"]   == 5000  # retained from the last [N/M] line

    def test_empty_log_returns_zeros(self, discovery_client):
        """Empty log (no matching lines) returns all zeros — no crash, no false positives."""
        client, tmp_path = discovery_client
        data = self._start_with_log(client, tmp_path, [])
        assert data["done"]    == 0
        assert data["total"]   == 0
        assert data["skipped"] == 0
        assert data["failed"]  == 0

    def test_unrelated_log_lines_return_zeros(self, discovery_client):
        """Log with no progress lines returns zeros (regression: old regex never matched anyway)."""
        client, tmp_path = discovery_client
        lines = [
            "INFO  starting download",
            "DEBUG fetching TIC 12345",
            "WARNING MAST rate-limit, sleeping 5s",
        ]
        data = self._start_with_log(client, tmp_path, lines)
        assert data["done"]    == 0
        assert data["total"]   == 0
        assert data["skipped"] == 0
        assert data["failed"]  == 0

    def test_field_order_independent(self, discovery_client):
        """Fields can appear in any order — not required to be done→skip→fail."""
        client, tmp_path = discovery_client
        lines = [
            # Reversed order: fail first, then skip, then done
            "[200/1000] fail=9 skip=3 done=200 3.0/s eta=2min",
        ]
        data = self._start_with_log(client, tmp_path, lines)
        assert data["done"]    == 200
        assert data["total"]   == 1000
        assert data["skipped"] == 3
        assert data["failed"]  == 9

    # ------------------------------------------------------------------
    # rate / eta_min extraction
    # ------------------------------------------------------------------

    def test_rate_and_eta_extracted_from_progress_line(self, discovery_client):
        """rate and eta_min are extracted from real progress-line format."""
        client, tmp_path = discovery_client
        lines = [
            "[1000/5000] done=1000 skip=0 fail=0 19.9/s eta=3.3min",
        ]
        data = self._start_with_log(client, tmp_path, lines)
        assert data["rate"]    == pytest.approx(19.9)
        assert data["eta_min"] == pytest.approx(3.3)

    def test_rate_and_eta_last_progress_line_wins(self, discovery_client):
        """When multiple progress lines exist, the last one's rate/eta are returned."""
        client, tmp_path = discovery_client
        lines = [
            "[100/5000] done=100 skip=0 fail=0 4.5/s eta=10.2min",
            "[500/5000] done=500 skip=0 fail=0 5.1/s eta=8.7min",
            "[1000/5000] done=1000 skip=0 fail=0 19.9/s eta=3.3min",
        ]
        data = self._start_with_log(client, tmp_path, lines)
        assert data["rate"]    == pytest.approx(19.9)
        assert data["eta_min"] == pytest.approx(3.3)

    def test_finish_line_retains_last_seen_rate_eta(self, discovery_client):
        """[finish] line has no rate/eta — last progress-line values are preserved."""
        client, tmp_path = discovery_client
        lines = [
            "[100/5000] done=100 skip=0 fail=0 5.0/s eta=10min",
            "[finish] done=5000 skip=12 fail=7 in 4.2min",
        ]
        data = self._start_with_log(client, tmp_path, lines)
        assert data["done"]    == 5000
        assert data["rate"]    == pytest.approx(5.0)
        assert data["eta_min"] == pytest.approx(10.0)

    def test_finish_only_rate_eta_null(self, discovery_client):
        """[finish] with no preceding progress line → rate and eta_min are null."""
        client, tmp_path = discovery_client
        lines = [
            "[finish] done=5000 skip=0 fail=0 in 6.2min",
        ]
        data = self._start_with_log(client, tmp_path, lines)
        assert data["rate"]    is None
        assert data["eta_min"] is None

    def test_empty_log_rate_eta_null(self, discovery_client):
        """Empty log returns rate=null, eta_min=null."""
        client, tmp_path = discovery_client
        data = self._start_with_log(client, tmp_path, [])
        assert data["rate"]    is None
        assert data["eta_min"] is None

    def test_rate_integer_format(self, discovery_client):
        """rate regex matches integer rates like '5/s' (no decimal)."""
        client, tmp_path = discovery_client
        lines = [
            "[200/1000] done=200 skip=0 fail=0 5/s eta=2.5min",
        ]
        data = self._start_with_log(client, tmp_path, lines)
        assert data["rate"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# /api/workers/start — farmctl wrapper
# ---------------------------------------------------------------------------

class TestWorkersStart:

    def test_workers_start_all_offline_calls_farmctl(self, discovery_client, monkeypatch):
        """All campaign workers (ops/workers.json) offline → farmctl called for each."""
        client, _ = discovery_client
        call_log = []

        def _fake_farmctl_start(name):
            call_log.append(name)
            return (0, f"worker {name} started", "")

        # Patch the closure-local function via the module attribute.
        # Since _farmctl_start is defined as a closure inside create_app,
        # we patch subprocess.run so _farmctl_start returns our canned result.
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0,
                                               stdout="started ok\n",
                                               stderr="")
            resp = client.post("/api/workers/start")

        assert resp.status_code == 200
        results = resp.json()
        assert isinstance(results, list)
        from exonit.server import _campaign_worker_names
        expected_names = _campaign_worker_names()  # live from ops/workers.json
        assert expected_names  # sanity: manifest readable, non-empty
        assert len(results) == len(expected_names)
        actions = {r["worker"]: r["action"] for r in results}
        for name in expected_names:
            assert name in actions
            assert actions[name] in ("started", "already_online", "failed")

    def test_workers_start_online_worker_not_called(self, discovery_client):
        """Worker that has a recent heartbeat reports already_online.

        Uses worker-HP (instances=1) so a single heartbeat is enough to
        satisfy expected=1.  worker-<host> requires 3 instances (its
        manifest value) — that behaviour is tested in test_workers_start.py.
        """
        client, _ = discovery_client

        # Register a worker via the real heartbeat endpoint
        import time
        auth = {"Authorization": "Bearer exonit-default-token"}
        reg_resp = client.post(
            "/api/worker/register",
            json={"name": "worker-HP", "capabilities": {"gpu": "directml"}},
            headers=auth,
        )
        assert reg_resp.status_code == 200

        # Now worker-HP is in workers_state with a fresh heartbeat
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0,
                                               stdout="started\n", stderr="")
            resp = client.post("/api/workers/start")

        assert resp.status_code == 200
        results = {r["worker"]: r for r in resp.json()}
        # HP (instances=1) should be already_online
        assert results["worker-HP"]["action"] == "already_online"
        # Others may be started or failed (no SSH in test env)
        assert results["worker-default"]["action"] in ("started", "failed", "already_online")

    def test_maceli_and_hp_included(self, discovery_client):
        """<host> (M3 laptop, GPU-enabled) and HP are both in the campaign
        start list. <host> was added on request; if offline when the button
        is pressed it just reports 'failed' rather than being skipped."""
        client, _ = discovery_client
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0,
                                               stdout="ok\n", stderr="")
            resp = client.post("/api/workers/start")

        names = {r["worker"] for r in resp.json()}
        assert "worker-<host>" in names
        assert "worker-HP" in names


# ---------------------------------------------------------------------------
# /api/admin/reset_all — compose baselines + worker stats
# ---------------------------------------------------------------------------

class TestResetAll:

    def test_reset_all_no_auth_required(self, discovery_client):
        """reset_all is called from the dashboard (browser) with NO token —
        like /api/workers/start — so it must succeed unauthenticated.
        (Regression: it used to require the worker_token → UI got 401.)"""
        client, _ = discovery_client
        resp = client.post("/api/admin/reset_all")
        assert resp.status_code == 200

    def test_reset_all_returns_ok(self, discovery_client):
        """reset_all (no auth, as the UI calls it) → 200 with ok:true."""
        client, _ = discovery_client
        resp = client.post("/api/admin/reset_all")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "reset" in data
        assert "workers_reset" in data

    def test_reset_all_clears_baselines(self, discovery_client):
        """After reset_all, the health/work baselines are cleared in the shared
        store (re-captured on next call). Verified by querying the store keys."""
        client, _ = discovery_client

        # Trigger a baseline snapshot so the keys are written to the store
        client.get("/api/health/work")

        # Reset all (no auth, as the dashboard calls it)
        client.post("/api/admin/reset_all")

        # After reset, the shared-store keys should be absent / zero.
        from exonit.shared_state import get_shared_state
        from exonit import server as srv
        store = get_shared_state()
        assert store.get(srv._PRIMARY_BASELINE_PENDING_KEY) is None
        assert store.get(srv._EXOMINER_BASELINE_PENDING_KEY) is None
        assert store.get(srv._LORA_BASELINE_PENDING_KEY) is None
        assert store.get_int(srv._FPP_BACKFILL_DONE_KEY) == 0

    def test_reset_all_clears_worker_stats(self, discovery_client):
        """reset_all also resets worker-level counters."""
        client, _ = discovery_client
        auth = {"Authorization": "Bearer exonit-default-token"}

        # Register a worker and fake some stats
        reg = client.post(
            "/api/worker/register",
            json={"name": "test-reset-worker", "capabilities": {"gpu": "none"}},
            headers=auth,
        )
        worker_id = reg.json()["worker_id"]

        # Manually bump counters via app.state.workers_state
        from fastapi.testclient import TestClient
        from exonit.server import create_app
        # (workers_state is the same object referenced by app.state)
        app = client.app
        ws = app.state.workers_state
        if worker_id in ws:
            ws[worker_id]["tasks_completed"] = 99
            ws[worker_id]["total_processing_time"] = 1000.0

        # Reset all
        client.post("/api/admin/reset_all", headers=auth)

        # Worker stats should be zeroed in-memory
        if worker_id in ws:
            assert ws[worker_id]["tasks_completed"] == 0
            assert ws[worker_id]["total_processing_time"] == 0.0

    def test_reset_baselines_still_works(self, discovery_client):
        """Existing reset_baselines endpoint is not broken."""
        client, _ = discovery_client
        auth = {"Authorization": "Bearer exonit-default-token"}
        resp = client.post("/api/admin/reset_baselines", headers=auth)
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_reset_worker_stats_still_works(self, discovery_client):
        """Existing reset_worker_stats endpoint is not broken."""
        client, _ = discovery_client
        auth = {"Authorization": "Bearer exonit-default-token"}
        resp = client.post("/api/admin/reset_worker_stats", headers=auth)
        assert resp.status_code == 200
        assert resp.json()["ok"] is True


# ---------------------------------------------------------------------------
# POST /api/worker/enqueue_bulk
# ---------------------------------------------------------------------------

class TestEnqueueBulk:
    """Tests for the bulk-enqueue endpoint added in the bulk-enqueue feature."""

    AUTH = {"Authorization": "Bearer exonit-default-token"}

    def _bulk_post(self, client, targets, auth=None):
        headers = auth if auth is not None else self.AUTH
        return client.post(
            "/api/worker/enqueue_bulk",
            json={"targets": targets},
            headers=headers,
        )

    def test_bulk_inserts_all_rows(self, discovery_client):
        """POSTing 1000 synthetic TICs inserts all 1000 into the tasks table."""
        client, tmp_path = discovery_client
        targets = [{"tic_id": i, "label": "candidate"} for i in range(1, 1001)]

        t0 = __import__("time").monotonic()
        resp = self._bulk_post(client, targets)
        elapsed = __import__("time").monotonic() - t0

        assert resp.status_code == 200
        data = resp.json()
        assert data["enqueued"] == 1000
        assert data["skipped"] == 0
        assert data["failed"] == 0
        assert "elapsed_seconds" in data
        # Single-transaction executemany for 1000 rows must be well under 1 s
        # on any reasonable hardware (SQLite write batch is nearly O(1) in time).
        assert elapsed < 1.0, f"1000-row bulk insert took {elapsed:.2f}s (expected <1s)"

    def test_bulk_idempotent(self, discovery_client):
        """Re-POSTing the same 1000 TICs skips all of them (already enqueued)."""
        client, tmp_path = discovery_client
        targets = [{"tic_id": i, "label": "candidate"} for i in range(2001, 3001)]

        r1 = self._bulk_post(client, targets)
        assert r1.status_code == 200
        assert r1.json()["enqueued"] == 1000

        r2 = self._bulk_post(client, targets)
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["skipped"] == 1000
        assert d2["enqueued"] == 0

    def test_bulk_missing_token_returns_401(self, discovery_client):
        """Request with no auth header returns 401."""
        client, _ = discovery_client
        resp = self._bulk_post(client, [{"tic_id": 1}], auth={})
        assert resp.status_code == 401

    def test_bulk_wrong_token_returns_401(self, discovery_client):
        """Request with wrong token returns 401."""
        client, _ = discovery_client
        bad_auth = {"Authorization": "Bearer wrong-token"}
        resp = self._bulk_post(client, [{"tic_id": 1}], auth=bad_auth)
        assert resp.status_code == 401

    def test_bulk_malformed_body_returns_400(self, discovery_client):
        """Body without 'targets' list returns 400."""
        client, _ = discovery_client
        resp = client.post(
            "/api/worker/enqueue_bulk",
            json={"wrong_key": 42},
            headers=self.AUTH,
        )
        assert resp.status_code == 400

    def test_bulk_targets_not_list_returns_400(self, discovery_client):
        """targets as a string instead of list returns 400."""
        client, _ = discovery_client
        resp = client.post(
            "/api/worker/enqueue_bulk",
            json={"targets": "not-a-list"},
            headers=self.AUTH,
        )
        assert resp.status_code == 400

    def test_bulk_empty_targets_list_returns_200(self, discovery_client):
        """Empty targets list is a valid no-op (0 enqueued, 0 skipped)."""
        client, _ = discovery_client
        resp = self._bulk_post(client, [])
        assert resp.status_code == 200
        data = resp.json()
        assert data["enqueued"] == 0
        assert data["skipped"] == 0

    def test_bulk_partial_skip(self, discovery_client):
        """First half already enqueued → second half inserted, first half skipped."""
        client, _ = discovery_client
        first_half = [{"tic_id": i} for i in range(10001, 10051)]
        second_half = [{"tic_id": i} for i in range(10051, 10101)]

        r1 = self._bulk_post(client, first_half)
        assert r1.status_code == 200
        assert r1.json()["enqueued"] == 50

        r2 = self._bulk_post(client, first_half + second_half)
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["enqueued"] == 50   # second_half inserted
        assert d2["skipped"] == 50    # first_half skipped
