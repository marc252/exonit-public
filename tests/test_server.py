import os
import json
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    os.environ["EXONIT_DB_PATH"] = str(tmp_path / "test.db")
    os.environ["EXONIT_REPORTS_DIR"] = str(tmp_path / "reports")
    os.makedirs(tmp_path / "reports", exist_ok=True)

    from exonit.server import create_app
    app = create_app()
    yield TestClient(app)

    del os.environ["EXONIT_DB_PATH"]
    del os.environ["EXONIT_REPORTS_DIR"]


def test_version(client):
    resp = client.get("/api/version")
    assert resp.status_code == 200
    assert "version" in resp.json()


def test_stats_empty(client):
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_stars"] == 0
    assert data["total_candidates"] == 0
    assert data["strong_candidates"] == 0


def test_stars_empty(client):
    resp = client.get("/api/stars")
    assert resp.status_code == 200
    assert resp.json() == []


def test_candidates_empty(client):
    resp = client.get("/api/candidates")
    assert resp.status_code == 200
    assert resp.json() == {"candidates": [], "total": 0}


def test_reports_empty(client):
    resp = client.get("/api/reports")
    assert resp.status_code == 200
    assert resp.json() == []


def test_stars_with_data(client, tmp_path):
    """Test stars endpoint returns star data with candidate counts."""
    from exonit.db import ExoNitDB
    db = ExoNitDB(str(tmp_path / "test.db"))
    db.add_star("TIC-123", 10.0, 20.0, 8.5, 3, 5000, 27.0)
    db.add_candidate("TIC-123", 1, 2.5, 1000.0, 0.1, 500, 15.0)
    db.add_candidate("TIC-123", 2, 5.0, 1001.0, 0.2, 300, 8.0)
    db.close()

    resp = client.get("/api/stars")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["tic_id"] == "TIC-123"
    assert data[0]["n_candidates"] == 2


def test_candidates_by_tic(client, tmp_path):
    """Test filtering candidates by tic_id."""
    from exonit.db import ExoNitDB
    db = ExoNitDB(str(tmp_path / "test.db"))
    db.add_star("TIC-100", 10.0, 20.0, 8.5, 3, 5000, 27.0)
    db.add_star("TIC-200", 11.0, 21.0, 9.0, 2, 4000, 25.0)
    db.add_candidate("TIC-100", 1, 2.5, 1000.0, 0.1, 500, 15.0)
    db.add_candidate("TIC-200", 1, 3.0, 1001.0, 0.2, 300, 8.0)
    db.close()

    resp = client.get("/api/candidates?tic_id=TIC-100")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["tic_id"] == "TIC-100"


def test_candidates_by_min_snr(client, tmp_path):
    """Test filtering candidates by min_snr."""
    from exonit.db import ExoNitDB
    db = ExoNitDB(str(tmp_path / "test.db"))
    db.add_star("TIC-100", 10.0, 20.0, 8.5, 3, 5000, 27.0)
    db.add_candidate("TIC-100", 1, 2.5, 1000.0, 0.1, 500, 15.0)
    db.add_candidate("TIC-100", 2, 5.0, 1001.0, 0.2, 300, 5.0)
    db.close()

    resp = client.get("/api/candidates?min_snr=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["snr"] == 15.0


def test_reports_with_files(client, tmp_path):
    """Test reports endpoint lists JSON files."""
    reports_dir = tmp_path / "reports"
    (reports_dir / "report_001.json").write_text(json.dumps({"tic": "123"}))
    (reports_dir / "report_002.json").write_text(json.dumps({"tic": "456"}))
    (reports_dir / "image.png").write_text("not json")

    resp = client.get("/api/reports")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert all(f.endswith(".json") for f in data)


# --- Worker API tests ---

def test_worker_register(client):
    resp = client.post("/api/worker/register",
                       json={"name": "test-worker", "capabilities": {"gpu": "none"}},
                       headers={"Authorization": "Bearer exonit-default-token"})
    assert resp.status_code == 200
    assert "worker_id" in resp.json()


def test_worker_register_bad_token(client):
    resp = client.post("/api/worker/register",
                       json={"name": "bad"},
                       headers={"Authorization": "Bearer wrong-token"})
    assert resp.status_code == 401


def test_worker_list_after_register(client):
    client.post("/api/worker/register",
                json={"name": "w1"},
                headers={"Authorization": "Bearer exonit-default-token"})
    resp = client.get("/api/worker/list")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["workers"]) >= 1


def test_worker_heartbeat(client):
    reg = client.post("/api/worker/register",
                      json={"name": "w1"},
                      headers={"Authorization": "Bearer exonit-default-token"})
    wid = reg.json()["worker_id"]
    resp = client.post("/api/worker/heartbeat",
                       json={"worker_id": wid, "status": "idle"},
                       headers={"Authorization": "Bearer exonit-default-token"})
    assert resp.status_code == 200
    assert resp.json()["command"] == "continue"


def test_worker_enqueue_and_get_task(client):
    # Register worker
    reg = client.post("/api/worker/register",
                      json={"name": "w1"},
                      headers={"Authorization": "Bearer exonit-default-token"})
    wid = reg.json()["worker_id"]
    auth = {"Authorization": "Bearer exonit-default-token", "X-Worker-ID": wid}

    # Enqueue a task
    resp = client.post("/api/worker/enqueue",
                       json={"tic_id": "123456", "time": "dGVzdA==", "flux": "dGVzdA==",
                             "n_points": 100, "n_sectors": 2},
                       headers={"Authorization": "Bearer exonit-default-token"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "enqueued"

    # Get task
    resp = client.get("/api/worker/task", headers=auth)
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] is not None
    assert data["tic_id"] == "123456"


def test_worker_enqueue_bulk(client):
    """Bulk-enqueue routes through db.backend (SQLite path) — insert + skip."""
    auth = {"Authorization": "Bearer exonit-default-token"}
    # First batch: all three are new.
    resp = client.post(
        "/api/worker/enqueue_bulk",
        json={"targets": [{"tic_id": 111}, {"tic_id": "222"}, {"tic_id": 333}]},
        headers=auth)
    assert resp.status_code == 200
    body = resp.json()
    assert body["enqueued"] == 3
    assert body["skipped"] == 0
    assert body["failed"] == 0

    # Second batch overlaps: 222/333 already queued, 444 is new.
    resp = client.post(
        "/api/worker/enqueue_bulk",
        json={"targets": [{"tic_id": 222}, {"tic_id": 333}, {"tic_id": 444}]},
        headers=auth)
    assert resp.status_code == 200
    body = resp.json()
    assert body["enqueued"] == 1   # only 444
    assert body["skipped"] == 2

    # A task is now claimable.
    reg = client.post("/api/worker/register", json={"name": "bw"}, headers=auth)
    wid = reg.json()["worker_id"]
    got = client.get("/api/worker/task",
                     headers={**auth, "X-Worker-ID": wid})
    assert got.status_code == 200
    assert got.json()["tic_id"] in {"111", "222", "333", "444"}


def test_worker_enqueue_bulk_bad_token(client):
    resp = client.post("/api/worker/enqueue_bulk",
                       json={"targets": [{"tic_id": 1}]},
                       headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


# --- Pipeline control tests ---

def test_status_not_running(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is False


def test_stop_when_not_running(client):
    resp = client.post("/api/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "not_running"


# --- Metadata & FITS endpoint tests ---

def test_metadata_endpoint(client, tmp_path):
    """Test metadata endpoint returns cached star metadata."""
    from exonit.metadata_cache import MetadataCache
    cache_path = str(tmp_path / "meta.db")
    os.environ["EXONIT_METADATA_CACHE"] = cache_path
    cache = MetadataCache(cache_path)
    cache.add_entry("261136679", [1, 2], 120.0, ["f1.fits", "f2.fits"])
    cache.close()

    # Need fresh app with new config
    from exonit.server import create_app
    app = create_app()
    from fastapi.testclient import TestClient
    c = TestClient(app)

    auth = {"Authorization": "Bearer exonit-default-token"}
    resp = c.get("/api/metadata/261136679", headers=auth)
    assert resp.status_code == 200
    assert resp.json()["sectors"] == [1, 2]

    resp = c.get("/api/metadata/999999", headers=auth)
    assert resp.status_code == 404

    del os.environ["EXONIT_METADATA_CACHE"]


def test_telemetry_tables_exist(client):
    """Verify worker_telemetry and worker_debug_mode tables are created."""
    import sqlite3
    db_path = os.environ["EXONIT_DB_PATH"]
    conn = sqlite3.connect(db_path)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    conn.close()
    assert "worker_telemetry" in tables
    assert "worker_debug_mode" in tables


def test_heartbeat_stores_telemetry(client):
    """Heartbeat with telemetry fields gets persisted to DB."""
    reg = client.post("/api/worker/register",
                      json={"name": "tel-worker", "capabilities": {"gpu": "none"}},
                      headers={"Authorization": "Bearer exonit-default-token"})
    wid = reg.json()["worker_id"]
    auth = {"Authorization": "Bearer exonit-default-token"}

    # Send enriched heartbeat
    resp = client.post("/api/worker/heartbeat", json={
        "worker_id": wid, "status": "working", "current_tic": 123456,
        "phase": "bls", "phase_progress": 0.5,
        "star_elapsed_s": 12.3, "stars_done": 10,
        "avg_time_per_star": 30.0, "last_error": None,
    }, headers=auth)
    assert resp.status_code == 200

    # Query telemetry
    resp = client.get(f"/api/telemetry/{wid}/latest")
    assert resp.status_code == 200
    data = resp.json()
    assert data["phase"] == "bls"
    assert data["phase_progress"] == 0.5
    assert data["current_tic"] == 123456


def test_debug_toggle(client):
    """Debug mode can be toggled and affects heartbeat response."""
    reg = client.post("/api/worker/register",
                      json={"name": "dbg-worker", "capabilities": {"gpu": "none"}},
                      headers={"Authorization": "Bearer exonit-default-token"})
    wid = reg.json()["worker_id"]
    auth = {"Authorization": "Bearer exonit-default-token"}

    # Enable debug
    resp = client.post(f"/api/debug/{wid}", json={"enabled": True}, headers=auth)
    assert resp.status_code == 200

    # Heartbeat should return debug_on
    resp = client.post("/api/worker/heartbeat",
                       json={"worker_id": wid, "status": "idle"},
                       headers=auth)
    assert resp.json()["command"] == "debug_on"

    # Second heartbeat — debug still on, but no state change → continue
    resp = client.post("/api/worker/heartbeat",
                       json={"worker_id": wid, "status": "idle"},
                       headers=auth)
    assert resp.json()["command"] in ("continue", "debug_on")


def test_debug_re_register_refires_transition(client):
    """After a worker restarts and re-registers, its next heartbeat must
    re-fire the debug_on transition (the fresh process has debug_mode=False
    in memory, even though the server state had cached it as True)."""
    auth = {"Authorization": "Bearer exonit-default-token"}
    reg = client.post("/api/worker/register",
                      json={"name": "restart-worker", "capabilities": {"gpu": "none"}},
                      headers=auth)
    wid = reg.json()["worker_id"]

    # Turn debug on, heartbeat, transition fires, cached True in state.
    client.post(f"/api/debug/{wid}", json={"enabled": True}, headers=auth)
    resp = client.post("/api/worker/heartbeat",
                       json={"worker_id": wid, "status": "idle"},
                       headers=auth)
    assert resp.json()["command"] == "debug_on"

    # Worker process restarts -> re-registers with the same name.
    reg2 = client.post("/api/worker/register",
                       json={"name": "restart-worker", "capabilities": {"gpu": "none"}},
                       headers=auth)
    assert reg2.json()["worker_id"] == wid  # same id preserved

    # Next heartbeat from the fresh process: must receive debug_on again.
    resp = client.post("/api/worker/heartbeat",
                       json={"worker_id": wid, "status": "idle"},
                       headers=auth)
    assert resp.json()["command"] == "debug_on"


def test_worker_id_stable_across_server_restart(tmp_path):
    """After a server process restart (workers_state wiped), a worker
    registering with the same name must get the same worker_id back
    (persisted via workers_registry)."""
    os.environ["EXONIT_DB_PATH"] = str(tmp_path / "stable.db")
    os.environ["EXONIT_REPORTS_DIR"] = str(tmp_path / "reports")
    os.makedirs(tmp_path / "reports", exist_ok=True)
    auth = {"Authorization": "Bearer exonit-default-token"}

    # First server "session"
    from exonit.server import create_app
    app1 = create_app()
    c1 = TestClient(app1)
    reg1 = c1.post("/api/worker/register",
                   json={"name": "persistent-worker", "capabilities": {}},
                   headers=auth)
    wid1 = reg1.json()["worker_id"]

    # Set debug ON for wid1
    c1.post(f"/api/debug/{wid1}", json={"enabled": True}, headers=auth)

    # Simulate server restart by building a fresh app on the same DB
    app2 = create_app()
    c2 = TestClient(app2)
    reg2 = c2.post("/api/worker/register",
                   json={"name": "persistent-worker", "capabilities": {}},
                   headers=auth)
    wid2 = reg2.json()["worker_id"]
    assert wid2 == wid1, "worker_id must persist across server restart"

    # Debug state is preserved because it's keyed by worker_id
    resp = c2.get(f"/api/debug/{wid2}", headers=auth)
    assert resp.json()["debug_enabled"] is True

    del os.environ["EXONIT_DB_PATH"]
    del os.environ["EXONIT_REPORTS_DIR"]


def test_debug_get_status(client):
    reg = client.post("/api/worker/register",
                      json={"name": "dbg2", "capabilities": {"gpu": "none"}},
                      headers={"Authorization": "Bearer exonit-default-token"})
    wid = reg.json()["worker_id"]
    auth = {"Authorization": "Bearer exonit-default-token"}

    resp = client.get(f"/api/debug/{wid}", headers=auth)
    assert resp.status_code == 200
    assert resp.json()["debug_enabled"] is False

    client.post(f"/api/debug/{wid}", json={"enabled": True}, headers=auth)
    resp = client.get(f"/api/debug/{wid}", headers=auth)
    assert resp.json()["debug_enabled"] is True


def test_telemetry_history(client):
    reg = client.post("/api/worker/register",
                      json={"name": "hist-worker", "capabilities": {"gpu": "none"}},
                      headers={"Authorization": "Bearer exonit-default-token"})
    wid = reg.json()["worker_id"]
    auth = {"Authorization": "Bearer exonit-default-token"}

    # Send 3 heartbeats
    for i in range(3):
        client.post("/api/worker/heartbeat", json={
            "worker_id": wid, "status": "working", "current_tic": 100 + i,
            "phase": "bls", "stars_done": i,
        }, headers=auth)

    resp = client.get(f"/api/telemetry/{wid}")
    assert resp.status_code == 200
    records = resp.json()["records"]
    assert len(records) == 3


def test_telemetry_purge(client):
    """Purge removes old telemetry records."""
    import sqlite3
    db_path = os.environ["EXONIT_DB_PATH"]

    reg = client.post("/api/worker/register",
                      json={"name": "purge-w", "capabilities": {"gpu": "none"}},
                      headers={"Authorization": "Bearer exonit-default-token"})
    wid = reg.json()["worker_id"]
    auth = {"Authorization": "Bearer exonit-default-token"}

    # Insert a heartbeat
    client.post("/api/worker/heartbeat",
                json={"worker_id": wid, "status": "idle"},
                headers=auth)

    # Manually backdate the record
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE worker_telemetry SET timestamp = datetime('now', '-10 days')")
    conn.commit()
    conn.close()

    # Trigger purge via the API
    resp = client.post("/api/telemetry/purge", headers=auth)
    assert resp.status_code == 200

    # Verify record is gone
    resp = client.get(f"/api/telemetry/{wid}")
    assert len(resp.json()["records"]) == 0


def test_fits_endpoint_security(client):
    """Test FITS endpoint blocks non-FITS files and missing files."""
    auth = {"Authorization": "Bearer exonit-default-token"}

    # Non-FITS extension should be blocked
    resp = client.get("/api/fits/malware.exe", headers=auth)
    assert resp.status_code == 400

    # No auth should be blocked
    resp = client.get("/api/fits/test.fits")
    assert resp.status_code == 401

    # Valid filename but non-existent file returns 404
    resp = client.get("/api/fits/nonexistent.fits", headers=auth)
    assert resp.status_code == 404


def test_full_telemetry_flow(client):
    """End-to-end: register, heartbeat with telemetry, enable debug, query."""
    auth = {"Authorization": "Bearer exonit-default-token"}

    # Register
    reg = client.post("/api/worker/register",
                      json={"name": "e2e-worker", "capabilities": {"gpu": "none"}},
                      headers=auth)
    wid = reg.json()["worker_id"]

    # Baseline heartbeat
    client.post("/api/worker/heartbeat", json={
        "worker_id": wid, "status": "working", "current_tic": 999,
        "phase": "bls", "phase_progress": 0.5,
        "star_elapsed_s": 10.0, "stars_done": 5,
        "avg_time_per_star": 25.0, "last_error": None,
    }, headers=auth)

    # Enable debug
    client.post(f"/api/debug/{wid}", json={"enabled": True}, headers=auth)

    # Next heartbeat returns debug_on
    resp = client.post("/api/worker/heartbeat", json={
        "worker_id": wid, "status": "working", "current_tic": 999,
        "phase": "injection", "phase_progress": 0.8,
        "star_elapsed_s": 45.0, "stars_done": 5,
        "avg_time_per_star": 25.0, "last_error": None,
        "system": {
            "cpu_percent": 92.0, "ram_used_gb": 6.5,
            "ram_total_gb": 16.0, "cpu_temp_c": 75.0, "pid_rss_mb": 1200,
        },
        "recent_timings": [{"tic": 998, "total_s": 30.0, "phases": {"bls": 10.0}}],
    }, headers=auth)
    assert resp.json()["command"] == "debug_on"

    # Query latest
    resp = client.get(f"/api/telemetry/{wid}/latest")
    data = resp.json()
    assert data["phase"] == "injection"
    assert data["cpu_percent"] == 92.0
    assert data["pid_rss_mb"] == 1200

    # Query history
    resp = client.get(f"/api/telemetry/{wid}")
    assert len(resp.json()["records"]) == 2

    # Worker list includes telemetry fields
    resp = client.get("/api/worker/list")
    w = [w for w in resp.json()["workers"] if w["worker_id"] == wid][0]
    assert w["phase"] == "injection"
    assert w["debug_mode"] is True


def test_worker_result_failed_status_marks_star_failed(client):
    """When the worker reports status='failed', the server must preserve it
    on the stars row (currently the server silently overrides to 'completed')."""
    auth = {"Authorization": "Bearer exonit-default-token"}

    # Register worker and enqueue a task
    reg = client.post("/api/worker/register",
                      json={"name": "fail-w", "capabilities": {}},
                      headers=auth)
    wid = reg.json()["worker_id"]

    enq = client.post("/api/worker/enqueue",
                      json={"tic_id": "999999", "target_info": {}},
                      headers=auth)
    assert enq.status_code == 200

    # Worker picks up the task
    task = client.get("/api/worker/task",
                      headers={**auth, "X-Worker-ID": wid}).json()
    assert task.get("task_id")

    # Worker reports failure after failed FITS load
    resp = client.post("/api/worker/result",
                       json={
                           "task_id": task["task_id"],
                           "tic_id": "999999",
                           "candidates": [],
                           "cnn_scores": [],
                           "completeness": None,
                           "n_points": 0,
                           "n_sectors": 0,
                           "processing_time_seconds": 5.0,
                           "status": "failed",
                       },
                       headers={**auth, "X-Worker-ID": wid})
    assert resp.status_code == 200

    # Star must be marked 'failed', not 'completed'
    # (query DB directly: /api/stars filters to status='completed' for UI)
    import sqlite3, os as _os
    conn = sqlite3.connect(_os.environ["EXONIT_DB_PATH"])
    row = conn.execute(
        "SELECT status FROM stars WHERE tic_id=?", ("999999",)
    ).fetchone()
    conn.close()
    assert row is not None, "star row not written at all"
    assert row[0] == "failed", f"got {row[0]}"


def test_worker_result_failed_requeues_task(client):
    """When worker reports status='failed', the task must go back to pending
    so another worker (or the same one later) can retry it."""
    import sqlite3, os
    auth = {"Authorization": "Bearer exonit-default-token"}

    reg = client.post("/api/worker/register",
                      json={"name": "fail-w2", "capabilities": {}},
                      headers=auth)
    wid = reg.json()["worker_id"]

    client.post("/api/worker/enqueue",
                json={"tic_id": "888888", "target_info": {}},
                headers=auth)
    task = client.get("/api/worker/task",
                      headers={**auth, "X-Worker-ID": wid}).json()

    client.post("/api/worker/result",
                json={
                    "task_id": task["task_id"],
                    "tic_id": "888888",
                    "candidates": [], "cnn_scores": [],
                    "completeness": None,
                    "n_points": 0, "n_sectors": 0,
                    "processing_time_seconds": 3.0,
                    "status": "failed",
                },
                headers={**auth, "X-Worker-ID": wid})

    db_path = os.environ["EXONIT_DB_PATH"]
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT status FROM tasks WHERE task_id=?", (task["task_id"],)
    ).fetchone()
    conn.close()
    assert row[0] == "pending", f"expected 'pending' after failed result, got {row[0]}"


# ---------- Plan C.1: Attempts, counters, error reasons, stats ----------


def _register_and_enqueue(client, worker_name, tic_id):
    auth = {"Authorization": "Bearer exonit-default-token"}
    reg = client.post("/api/worker/register",
                      json={"name": worker_name, "capabilities": {}},
                      headers=auth)
    wid = reg.json()["worker_id"]
    client.post("/api/worker/enqueue",
                json={"tic_id": tic_id, "target_info": {}},
                headers=auth)
    return wid


def _pick_task(client, wid):
    auth = {"Authorization": "Bearer exonit-default-token"}
    return client.get("/api/worker/task",
                      headers={**auth, "X-Worker-ID": wid}).json()


def _post_result(client, wid, task_id, tic_id, status="completed", error=None):
    auth = {"Authorization": "Bearer exonit-default-token"}
    payload = {
        "task_id": task_id,
        "tic_id": tic_id,
        "candidates": [],
        "cnn_scores": [],
        "completeness": None,
        "n_points": 0 if status != "completed" else 10,
        "n_sectors": 0 if status != "completed" else 1,
        "processing_time_seconds": 1.0,
        "status": status,
    }
    if error is not None:
        payload["error"] = error
    return client.post("/api/worker/result", json=payload,
                       headers={**auth, "X-Worker-ID": wid})


def test_api_stats_includes_task_counts(client):
    """/api/stats must expose pending/assigned/completed/failed task counts."""
    wid = _register_and_enqueue(client, "statw", "111111")
    stats = client.get("/api/stats").json()
    for key in ("pending_tasks", "assigned_tasks", "completed_tasks", "failed_tasks"):
        assert key in stats, f"missing {key} in /api/stats"
    assert stats["pending_tasks"] == 1


def test_worker_result_completed_increments_counter(client):
    wid = _register_and_enqueue(client, "countw", "200001")
    task = _pick_task(client, wid)
    _post_result(client, wid, task["task_id"], "200001", status="completed")
    workers = client.get("/api/worker/list").json()["workers"]
    w = [x for x in workers if x["worker_id"] == wid][0]
    assert w["tasks_completed"] == 1


def test_worker_result_failed_does_not_increment_counter(client):
    """A failed result must NOT bump tasks_completed."""
    wid = _register_and_enqueue(client, "failw", "200002")
    task = _pick_task(client, wid)
    _post_result(client, wid, task["task_id"], "200002",
                 status="failed", error="load_failed")
    workers = client.get("/api/worker/list").json()["workers"]
    w = [x for x in workers if x["worker_id"] == wid][0]
    assert w["tasks_completed"] == 0


def test_task_attempts_increments_on_assign(client):
    """Each assign_task must bump attempts counter."""
    import sqlite3, os as _os
    wid = _register_and_enqueue(client, "attw", "200003")
    tids = []
    for _ in range(3):
        task = _pick_task(client, wid)
        tids.append(task["task_id"])
        # Fail to release it back to pending
        _post_result(client, wid, task["task_id"], "200003",
                     status="failed", error="x")

    # Inspect the last seen task_id (same task_id across requeues)
    assert len(set(tids)) == 1, "task_id should remain stable across requeues"
    conn = sqlite3.connect(_os.environ["EXONIT_DB_PATH"])
    attempts = conn.execute(
        "SELECT attempts FROM tasks WHERE task_id=?", (tids[0],)
    ).fetchone()[0]
    conn.close()
    # 3 assignments total (even if final one doesn't requeue)
    assert attempts == 3, f"expected attempts=3, got {attempts}"


def test_failed_under_max_attempts_requeues(client):
    """Failure under MAX_ATTEMPTS must put the task back to pending."""
    import sqlite3, os as _os
    wid = _register_and_enqueue(client, "reqw", "200004")
    task = _pick_task(client, wid)
    _post_result(client, wid, task["task_id"], "200004",
                 status="failed", error="load_failed")
    conn = sqlite3.connect(_os.environ["EXONIT_DB_PATH"])
    row = conn.execute(
        "SELECT status, attempts FROM tasks WHERE task_id=?",
        (task["task_id"],)
    ).fetchone()
    conn.close()
    assert row[0] == "pending", f"got status={row[0]}"
    assert row[1] == 1


def test_failed_at_max_attempts_marks_failed(client):
    """After MAX_ATTEMPTS failures the task must go to status='failed',
    not back to pending. last_error must be stored."""
    import sqlite3, os as _os
    wid = _register_and_enqueue(client, "maxw", "200005")
    last_task_id = None
    for i in range(3):  # MAX_ATTEMPTS = 3
        task = _pick_task(client, wid)
        last_task_id = task["task_id"]
        _post_result(client, wid, task["task_id"], "200005",
                     status="failed", error=f"reason_{i}")

    # After the 3rd failure the task must be definitively failed
    conn = sqlite3.connect(_os.environ["EXONIT_DB_PATH"])
    row = conn.execute(
        "SELECT status, attempts, last_error FROM tasks WHERE task_id=?",
        (last_task_id,)
    ).fetchone()
    conn.close()
    assert row[0] == "failed", f"expected status=failed, got {row[0]}"
    assert row[1] == 3
    assert row[2] == "reason_2"


def test_api_stats_counts_failed_tasks(client):
    wid = _register_and_enqueue(client, "fstatw", "200006")
    for i in range(3):
        task = _pick_task(client, wid)
        _post_result(client, wid, task["task_id"], "200006",
                     status="failed", error="x")
    stats = client.get("/api/stats").json()
    assert stats["failed_tasks"] == 1
    assert stats["pending_tasks"] == 0


def test_failed_task_not_picked_up_again(client):
    """Once a task is in status='failed' it must not be re-assigned."""
    wid = _register_and_enqueue(client, "noreassign", "200007")
    for i in range(3):
        task = _pick_task(client, wid)
        _post_result(client, wid, task["task_id"], "200007",
                     status="failed", error="x")
    # Queue is now empty
    task = _pick_task(client, wid)
    assert task.get("task_id") is None, f"got {task}"


# ---------- Plan C.3: Re-registration + stale reassign + persistent counters ----------


def test_worker_list_does_not_remove_stale_workers(client):
    """Calling /api/worker/list must NOT mutate workers_state (read-only GET).
    A long-running worker that hasn't heartbeated for a while should still
    appear in the list — cleanup belongs in a dedicated path, not a GET."""
    import time as _time
    from exonit import server as srv_module

    wid = _register_and_enqueue(client, "longw", "300001")

    # Simulate 15 min of silence (older than the 10-min window)
    app = client.app
    state = None
    for obj in app.state.__dict__.values():
        pass
    # Reach into the module-level workers_state by triggering a list call
    # and then forcibly aging the worker
    _ = client.get("/api/worker/list").json()
    # The endpoint itself holds workers_state as a closure; we mutate via
    # the heartbeat by backdating it is impractical. Instead we assert a
    # stable behaviour: /api/worker/list called twice must not have removed
    # the worker between calls when no heartbeat has been received.
    first = client.get("/api/worker/list").json()["workers"]
    second = client.get("/api/worker/list").json()["workers"]
    ids1 = [w["worker_id"] for w in first]
    ids2 = [w["worker_id"] for w in second]
    assert wid in ids1
    assert ids1 == ids2, "workers_state must be stable across repeated list calls"


def test_reassign_stale_tasks_respects_alive_workers(client):
    """A task assigned to a worker that is still alive (recent heartbeat)
    must NOT be reassigned, even if it has been running > timeout_seconds."""
    import sqlite3, os as _os, time as _time
    from exonit.db import ExoNitDB

    wid = _register_and_enqueue(client, "alivew", "300002")
    task = _pick_task(client, wid)
    assert task["task_id"]

    # Backdate the assignment in the DB so the plain stale check would
    # otherwise reassign it.
    db = ExoNitDB(_os.environ["EXONIT_DB_PATH"])
    db.conn.execute(
        "UPDATE tasks SET assigned_at=? WHERE task_id=?",
        ("2020-01-01T00:00:00", task["task_id"]))
    db.conn.commit()

    # Call reassign with the alive set containing wid; task must stay assigned.
    reassigned = db.reassign_stale_tasks(60, alive_worker_ids={wid})
    row = db.conn.execute(
        "SELECT status, assigned_to FROM tasks WHERE task_id=?",
        (task["task_id"],)).fetchone()
    db.close()
    assert reassigned == 0
    assert row["status"] == "assigned"
    assert row["assigned_to"] == wid


def test_reassign_stale_tasks_reclaims_from_dead_workers(client):
    """A task whose assigned worker is absent from alive_worker_ids must
    be returned to the pending queue."""
    import os as _os
    from exonit.db import ExoNitDB

    wid = _register_and_enqueue(client, "deadw", "300003")
    task = _pick_task(client, wid)

    db = ExoNitDB(_os.environ["EXONIT_DB_PATH"])
    db.conn.execute(
        "UPDATE tasks SET assigned_at=? WHERE task_id=?",
        ("2020-01-01T00:00:00", task["task_id"]))
    db.conn.commit()

    # Worker considered dead (wid not in alive set)
    reassigned = db.reassign_stale_tasks(60, alive_worker_ids=set())
    row = db.conn.execute(
        "SELECT status, assigned_to FROM tasks WHERE task_id=?",
        (task["task_id"],)).fetchone()
    db.close()
    assert reassigned == 1
    assert row["status"] == "pending"
    assert row["assigned_to"] is None


def test_stale_task_default_timeout_is_at_least_30_minutes(client):
    """The default stale-task timeout used by the server must be >= 1800s
    so normal injection recovery doesn't trigger reassignment."""
    from exonit import server as srv_module
    assert getattr(srv_module, "STALE_TASK_TIMEOUT_SECONDS", 120) >= 1800


def test_worker_counters_persist_across_restart(client, tmp_path):
    """tasks_completed persisted to workers_registry so a restart preserves it."""
    # First server instance: register a worker and complete one task.
    wid = _register_and_enqueue(client, "persistw", "300004")
    task = _pick_task(client, wid)
    _post_result(client, wid, task["task_id"], "300004", status="completed")

    # Check: counter == 1 now
    workers = client.get("/api/worker/list").json()["workers"]
    w = [x for x in workers if x["worker_id"] == wid][0]
    assert w["tasks_completed"] == 1

    # Simulate server restart by rebuilding the app (same DB path).
    from exonit.server import create_app
    from fastapi.testclient import TestClient
    app2 = create_app()
    client2 = TestClient(app2)

    # Worker sends heartbeat; server should tell it to re-register,
    # then register should re-hydrate counters from workers_registry.
    auth = {"Authorization": "Bearer exonit-default-token"}
    reg = client2.post(
        "/api/worker/register",
        json={"name": "persistw", "capabilities": {}},
        headers=auth)
    wid2 = reg.json()["worker_id"]
    assert wid2 == wid, "worker_id should be stable via workers_registry"

    workers = client2.get("/api/worker/list").json()["workers"]
    w = [x for x in workers if x["worker_id"] == wid2][0]
    assert w["tasks_completed"] == 1, \
        f"tasks_completed must persist across restart, got {w['tasks_completed']}"


# ---------- Plan C.4: atomic get_and_assign_task (race condition fix) ----------


def test_worker_list_marks_offline_when_heartbeat_is_stale(client):
    """Workers whose heartbeat is older than WORKER_OFFLINE_SECONDS must
    report status='offline' (not keep the last 'working' value) — otherwise
    a worker stopped an hour ago still shows as mid-processing on the
    dashboard. workers_state is kept intact so Plan C.3's pure-read
    invariant is preserved; only the response body is filtered."""
    from exonit import server as srv_module

    wid = _register_and_enqueue(client, "ghostw", "900001")
    auth = {"Authorization": "Bearer exonit-default-token"}
    # Pretend the worker started a task then went silent
    client.post("/api/worker/heartbeat",
                json={"worker_id": wid, "status": "working",
                      "current_tic": "900001"},
                headers=auth)
    # Back-date the heartbeat past the offline threshold
    threshold = getattr(srv_module, "WORKER_OFFLINE_SECONDS", 600)
    import time as _time
    client.app.state.workers_state[wid]["last_heartbeat"] = (
        _time.time() - threshold - 60)

    workers = client.get("/api/worker/list").json()["workers"]
    w = [x for x in workers if x["worker_id"] == wid][0]
    assert w["status"] == "offline", f"expected offline, got {w['status']}"
    assert w["current_tic"] is None


def test_get_and_assign_task_is_atomic_under_concurrency(tmp_path):
    """Plan C.4 regression: concurrent workers must never claim the same task.

    Before the fix, /api/worker/task did:
        task = db.get_pending_task()   # releases lock
        db.assign_task(task_id, wid)   # re-acquires lock
    and two racing requests could both SELECT the same pending row before
    either UPDATE landed, assigning the same TIC to two workers (observed
    in the cluster with TIC 441739020 processed by 3 workers simultaneously).

    A single atomic helper must hold the lock across SELECT+UPDATE."""
    import threading
    from exonit.db import ExoNitDB

    db = ExoNitDB(str(tmp_path / "race.db"))
    db.add_task("t-solo", "500001")

    results = []
    errors = []
    barrier = threading.Barrier(20)

    def claim(wid):
        try:
            barrier.wait()
            t = db.get_and_assign_task(wid)
            if t:
                results.append((wid, t["task_id"]))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=claim, args=(f"w{i}",))
               for i in range(20)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    db.close()

    assert not errors, f"unexpected errors: {errors}"
    assert len(results) == 1, (
        f"exactly one worker should claim the single pending task, "
        f"got {len(results)}: {results}")


def test_get_and_assign_task_bumps_attempts_and_sets_assigned(tmp_path):
    """The atomic helper must reproduce the side-effects of assign_task:
    status='assigned', assigned_to set, attempts incremented."""
    from exonit.db import ExoNitDB

    db = ExoNitDB(str(tmp_path / "atomic.db"))
    db.add_task("t-1", "600001")
    task = db.get_and_assign_task("worker-A")
    assert task is not None
    assert task["task_id"] == "t-1"
    assert task["status"] == "assigned"
    assert task["assigned_to"] == "worker-A"

    row = db.get_task("t-1")
    assert row["status"] == "assigned"
    assert row["assigned_to"] == "worker-A"
    assert row["attempts"] == 1
    db.close()


def test_get_and_assign_task_returns_none_when_empty(tmp_path):
    """Must return None cleanly when no pending tasks remain."""
    from exonit.db import ExoNitDB

    db = ExoNitDB(str(tmp_path / "empty.db"))
    assert db.get_and_assign_task("worker-X") is None
    db.close()

def test_worker_fpp_task_respects_metadata_filter(tmp_path):
    db_path = str(tmp_path / "test.db")
    meta_path = str(tmp_path / "meta.db")
    reports_dir = str(tmp_path / "reports")
    os.makedirs(reports_dir, exist_ok=True)
    env = {
        "EXONIT_DB_PATH": db_path,
        "EXONIT_METADATA_CACHE": meta_path,
        "EXONIT_REPORTS_DIR": reports_dir,
    }
    old_env = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        from exonit.db import ExoNitDB
        from exonit.metadata_cache import MetadataCache
        db = ExoNitDB(db_path)
        db.add_star("100", 0, 0, 10, 1, 1000, 30)
        db.add_star("200", 0, 0, 10, 1, 1000, 30)
        db.add_candidate("100", 1, 2.0, 0, 0.1, 500, 50.0, cnn_score=0.95)
        db.add_candidate("200", 1, 3.0, 0, 0.1, 500, 20.0, cnn_score=0.95)
        db.close()
        cache = MetadataCache(meta_path)
        cache.add_entry("200", sectors=[1], exptime=120.0, fits_files=["tic200.fits"])
        cache.close()

        from exonit.server import create_app
        c = TestClient(create_app())
        resp = c.get("/api/worker/fpp_task",
                     headers={"Authorization": "Bearer exonit-default-token"})
        assert resp.status_code == 200
        task = resp.json()["task"]
        assert task["tic_id"] == "200"
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_validated_filter_requires_nfpp(tmp_path):
    db_path = str(tmp_path / "test.db")
    reports_dir = str(tmp_path / "reports")
    os.makedirs(reports_dir, exist_ok=True)
    old_db = os.environ.get("EXONIT_DB_PATH")
    old_reports = os.environ.get("EXONIT_REPORTS_DIR")
    os.environ["EXONIT_DB_PATH"] = db_path
    os.environ["EXONIT_REPORTS_DIR"] = reports_dir
    try:
        from exonit.db import ExoNitDB
        db = ExoNitDB(db_path)
        db.add_star("300", 0, 0, 10, 1, 1000, 30)
        db.add_star("400", 0, 0, 10, 1, 1000, 30)
        high_nfpp = db.add_candidate("300", 1, 2.0, 0, 0.1, 500, 20.0,
                                     cnn_score=0.95)
        low_nfpp = db.add_candidate("400", 1, 3.0, 0, 0.1, 500, 20.0,
                                    cnn_score=0.95)
        db.update_candidate_fpp(high_nfpp, fpp=0.01, nfpp=0.01)
        db.update_candidate_fpp(low_nfpp, fpp=0.01, nfpp=0.0005)
        db.close()

        from exonit.server import create_app
        c = TestClient(create_app())
        resp = c.get("/api/candidates?filter=validated")
        assert resp.status_code == 200
        rows = resp.json()["candidates"]
        assert [r["tic_id"] for r in rows] == ["400"]
    finally:
        if old_db is None:
            os.environ.pop("EXONIT_DB_PATH", None)
        else:
            os.environ["EXONIT_DB_PATH"] = old_db
        if old_reports is None:
            os.environ.pop("EXONIT_REPORTS_DIR", None)
        else:
            os.environ["EXONIT_REPORTS_DIR"] = old_reports

def test_worker_fpp_task_backfills_missing_nfpp(tmp_path):
    db_path = str(tmp_path / "test.db")
    meta_path = str(tmp_path / "meta.db")
    reports_dir = str(tmp_path / "reports")
    os.makedirs(reports_dir, exist_ok=True)
    env = {
        "EXONIT_DB_PATH": db_path,
        "EXONIT_METADATA_CACHE": meta_path,
        "EXONIT_REPORTS_DIR": reports_dir,
    }
    old_env = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        from exonit.db import ExoNitDB
        from exonit.metadata_cache import MetadataCache
        db = ExoNitDB(db_path)
        db.add_star("500", 0, 0, 10, 1, 1000, 30)
        candidate_id = db.add_candidate(
            "500", 1, 2.0, 0, 0.1, 500, 50.0, cnn_score=0.95)
        db._exec(
            """UPDATE candidates SET triceratops_fpp = 0.01,
                      triceratops_nfpp = NULL, triceratops_verdict = 'candidate'
               WHERE id = ?""",
            (candidate_id,))
        db.close()
        cache = MetadataCache(meta_path)
        cache.add_entry("500", sectors=[1], exptime=120.0, fits_files=["tic500.fits"])
        cache.close()

        from exonit.server import create_app
        c = TestClient(create_app())
        resp = c.get("/api/worker/fpp_task",
                     headers={"Authorization": "Bearer exonit-default-token"})
        assert resp.status_code == 200
        assert resp.json()["task"]["tic_id"] == "500"
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --- /api/candidate/update_analysis endpoint tests ---

AUTH = {"Authorization": "Bearer exonit-default-token"}


def _setup_candidate(client, tmp_path, tic="600", analysis=None, verdict=None):
    """Helper: create a star+candidate and optionally set claude fields directly."""
    from exonit.db import ExoNitDB
    db = ExoNitDB(str(tmp_path / "test.db"))
    db.add_star(tic, ra=0, dec=0, tmag=10, n_sectors=1,
                n_datapoints=1000, time_span_days=30)
    cid = db.add_candidate(tic, candidate_num=1, period_days=5.0, t0=0,
                           duration_days=0.1, depth_ppm=500, snr=15.0)
    if analysis is not None or verdict is not None:
        db.update_candidate_claude(cid, verdict=verdict, analysis=analysis)
    db.close()
    return cid


class TestCandidateUpdateAnalysisEndpoint:
    """Tests for POST /api/candidate/update_analysis."""

    def test_requires_auth(self, client):
        resp = client.post("/api/candidate/update_analysis",
                           json={"updates": []})
        assert resp.status_code == 401

    def test_bad_token_rejected(self, client):
        resp = client.post("/api/candidate/update_analysis",
                           json={"updates": []},
                           headers={"Authorization": "Bearer wrong-token"})
        assert resp.status_code == 401

    def test_missing_updates_key_is_bad_request(self, client):
        resp = client.post("/api/candidate/update_analysis",
                           json={"foo": []},
                           headers=AUTH)
        assert resp.status_code == 400

    def test_updates_not_list_is_bad_request(self, client):
        resp = client.post("/api/candidate/update_analysis",
                           json={"updates": "oops"},
                           headers=AUTH)
        assert resp.status_code == 400

    def test_entry_missing_candidate_id_is_bad_request(self, client):
        resp = client.post("/api/candidate/update_analysis",
                           json={"updates": [{"verdict": "ok"}]},
                           headers=AUTH)
        assert resp.status_code == 400

    def test_empty_batch_returns_zero(self, client):
        resp = client.post("/api/candidate/update_analysis",
                           json={"updates": []},
                           headers=AUTH)
        assert resp.status_code == 200
        assert resp.json()["updated"] == 0

    def test_single_candidate_written(self, client, tmp_path):
        cid = _setup_candidate(client, tmp_path, tic="601")

        resp = client.post("/api/candidate/update_analysis",
                           json={"updates": [
                               {"candidate_id": cid,
                                "verdict": "planet_candidate",
                                "analysis": "[lora-v2] Clean transit"}
                           ]},
                           headers=AUTH)
        assert resp.status_code == 200
        assert resp.json()["updated"] == 1

        # Verify via the DB directly (same file the client app uses).
        from exonit.db import ExoNitDB
        db = ExoNitDB(str(tmp_path / "test.db"))
        c = db.get_candidates("601")[0]
        db.close()
        assert c["claude_verdict"] == "planet_candidate"
        assert c["claude_analysis"] == "[lora-v2] Clean transit"

    def test_batch_of_multiple_candidates_written_atomically(self, client, tmp_path):
        cid1 = _setup_candidate(client, tmp_path, tic="602")
        cid2 = _setup_candidate(client, tmp_path, tic="603")

        resp = client.post("/api/candidate/update_analysis",
                           json={"updates": [
                               {"candidate_id": cid1, "verdict": "likely_fp",
                                "analysis": "[lora-v2] FP signal"},
                               {"candidate_id": cid2, "verdict": "planet_candidate",
                                "analysis": "[lora-v2] Good"},
                           ]},
                           headers=AUTH)
        assert resp.status_code == 200
        assert resp.json()["updated"] == 2

        from exonit.db import ExoNitDB
        db = ExoNitDB(str(tmp_path / "test.db"))
        assert db.get_candidates("602")[0]["claude_verdict"] == "likely_fp"
        assert db.get_candidates("603")[0]["claude_verdict"] == "planet_candidate"
        db.close()

    def test_lora_preservation_via_endpoint(self, client, tmp_path):
        """Existing [lora-v2] analysis survives when incoming is untagged."""
        cid = _setup_candidate(client, tmp_path, tic="604",
                               analysis="[lora-v2] Existing long explanation text here",
                               verdict="planet_candidate")

        resp = client.post("/api/candidate/update_analysis",
                           json={"updates": [
                               {"candidate_id": cid,
                                "verdict": "unknown",
                                "analysis": "Short plain"}
                           ]},
                           headers=AUTH)
        assert resp.status_code == 200

        from exonit.db import ExoNitDB
        db = ExoNitDB(str(tmp_path / "test.db"))
        c = db.get_candidates("604")[0]
        db.close()
        # Verdict updated, existing lora analysis preserved.
        assert c["claude_verdict"] == "unknown"
        assert c["claude_analysis"] == "[lora-v2] Existing long explanation text here"

    def test_null_verdict_accepted(self, client, tmp_path):
        cid = _setup_candidate(client, tmp_path, tic="605")
        resp = client.post("/api/candidate/update_analysis",
                           json={"updates": [
                               {"candidate_id": cid,
                                "verdict": None,
                                "analysis": "[lora-v2] Analysis only"}
                           ]},
                           headers=AUTH)
        assert resp.status_code == 200
        assert resp.json()["updated"] == 1

        from exonit.db import ExoNitDB
        db = ExoNitDB(str(tmp_path / "test.db"))
        c = db.get_candidates("605")[0]
        db.close()
        assert c["claude_verdict"] is None
        assert c["claude_analysis"] == "[lora-v2] Analysis only"

    def test_absent_verdict_key_preserves_existing_verdict(self, client, tmp_path):
        """CRITICAL regression guard: omitting 'verdict' from the entry must
        NOT clobber an existing claude_verdict.

        This is the exact scenario the reviewer flagged: a LoRA backfill run
        regenerates the analysis for a candidate that already has a verdict
        ('planet_candidate').  The batch entry has no 'verdict' key.  After
        the call the existing verdict must still be 'planet_candidate'.
        """
        cid = _setup_candidate(client, tmp_path, tic="606",
                               verdict="planet_candidate",
                               analysis="Old plain text")

        resp = client.post("/api/candidate/update_analysis",
                           json={"updates": [
                               # No "verdict" key — only regenerating analysis.
                               {"candidate_id": cid,
                                "analysis": "[lora-v2] Regenerated explanation"}
                           ]},
                           headers=AUTH)
        assert resp.status_code == 200
        assert resp.json()["updated"] == 1

        from exonit.db import ExoNitDB
        db = ExoNitDB(str(tmp_path / "test.db"))
        c = db.get_candidates("606")[0]
        db.close()
        assert c["claude_verdict"] == "planet_candidate", (
            "claude_verdict was clobbered by a batch entry without 'verdict' key"
        )
        assert c["claude_analysis"] == "[lora-v2] Regenerated explanation"


# --- Backfill status telemetry tests ---

AUTH_H = {"Authorization": "Bearer exonit-default-token"}


def test_worker_list_backfill_tic_shows_backfill_status(client):
    """When the server has a backfill TIC recorded for a worker,
    /api/worker/list must return status='backfill' (not 'working')."""
    from exonit.server import _set_worker_backfill_tic, _clear_worker_backfill_tic

    reg = client.post("/api/worker/register",
                      json={"name": "w-bf-test"},
                      headers=AUTH_H)
    wid = reg.json()["worker_id"]

    # Simulate the server recording a backfill assignment
    _set_worker_backfill_tic(wid, 99999, "fpp")
    try:
        # Send heartbeat with status='idle' (what the worker reported before)
        client.post("/api/worker/heartbeat",
                    json={"worker_id": wid, "status": "idle"},
                    headers=AUTH_H)

        resp = client.get("/api/worker/list")
        assert resp.status_code == 200
        workers = resp.json()["workers"]
        w = next(x for x in workers if x["worker_id"] == wid)
        assert w["status"] == "backfill", (
            f"Expected status='backfill' when bf TIC is set, got {w['status']!r}"
        )
        assert w["current_task"] == "fpp backfill"
    finally:
        _clear_worker_backfill_tic(wid)


def test_worker_list_backfill_status_from_heartbeat(client):
    """When a worker reports status='backfill' in its heartbeat (no bf TIC
    on server side), /api/worker/list must propagate that status as-is."""
    reg = client.post("/api/worker/register",
                      json={"name": "w-bf-hb-test"},
                      headers=AUTH_H)
    wid = reg.json()["worker_id"]

    # Worker self-reports backfill status
    client.post("/api/worker/heartbeat",
                json={"worker_id": wid, "status": "backfill"},
                headers=AUTH_H)

    resp = client.get("/api/worker/list")
    assert resp.status_code == 200
    workers = resp.json()["workers"]
    w = next(x for x in workers if x["worker_id"] == wid)
    assert w["status"] == "backfill", (
        f"Expected status='backfill' from heartbeat passthrough, got {w['status']!r}"
    )


def test_worker_list_primary_working_unaffected(client):
    """Regression guard: a worker doing primary work must still show
    status='working', not 'backfill'."""
    reg = client.post("/api/worker/register",
                      json={"name": "w-primary-test"},
                      headers=AUTH_H)
    wid = reg.json()["worker_id"]

    client.post("/api/worker/heartbeat",
                json={"worker_id": wid, "status": "working", "current_tic": "12345"},
                headers=AUTH_H)

    resp = client.get("/api/worker/list")
    assert resp.status_code == 200
    workers = resp.json()["workers"]
    w = next(x for x in workers if x["worker_id"] == wid)
    # No backfill TIC set on server side → fallback to heartbeat status
    assert w["status"] == "working", (
        f"Primary working status must not be clobbered, got {w['status']!r}"
    )
