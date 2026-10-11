"""When a worker submits a result with lc_load_recovered=True, the server
must persist 1 in the tasks.lc_load_recovered column."""
from __future__ import annotations

import os
import tempfile

from fastapi.testclient import TestClient

from exonit.config import ExoNitConfig
from exonit.db import ExoNitDB
from exonit.server import create_app


def _setup_app(tmp_dir):
    cfg = ExoNitConfig()
    cfg.db_path = os.path.join(tmp_dir, "exonit.db")
    cfg.metadata_cache_path = os.path.join(tmp_dir, "metadata.db")
    cfg.berger_db_path = os.path.join(tmp_dir, "berger.db")
    return create_app(cfg), cfg


def test_worker_result_with_recovered_flag_persists_one(tmp_path):
    app, cfg = _setup_app(str(tmp_path))
    client = TestClient(app)
    db = ExoNitDB(cfg.db_path)
    db._exec(
        "INSERT INTO tasks (task_id, tic_id, status, attempts) "
        "VALUES (?, ?, 'assigned', 1)",
        ("t-1", "11111"))
    body = {
        "task_id": "t-1", "tic_id": "11111",
        "candidates": [], "cnn_scores": [],
        "processing_time_seconds": 5.0, "gpu_used": False,
        "n_sectors": 1, "n_points": 200, "time_span_days": 27.0,
        "status": "completed", "lc_load_recovered": True,
    }
    r = client.post("/api/worker/result", json=body,
                    headers={"Authorization": "Bearer exonit-default-token"})
    assert r.status_code == 200
    row = db.conn.execute(
        "SELECT status, lc_load_recovered FROM tasks WHERE task_id='t-1'"
    ).fetchone()
    assert row[0] == "completed"
    assert row[1] == 1


def test_worker_result_without_recovered_flag_persists_zero(tmp_path):
    app, cfg = _setup_app(str(tmp_path))
    client = TestClient(app)
    db = ExoNitDB(cfg.db_path)
    db._exec(
        "INSERT INTO tasks (task_id, tic_id, status, attempts) "
        "VALUES (?, ?, 'assigned', 1)",
        ("t-2", "22222"))
    body = {
        "task_id": "t-2", "tic_id": "22222",
        "candidates": [], "cnn_scores": [],
        "processing_time_seconds": 3.0, "gpu_used": False,
        "n_sectors": 1, "n_points": 200, "time_span_days": 27.0,
        "status": "completed",
        # lc_load_recovered intentionally omitted
    }
    r = client.post("/api/worker/result", json=body,
                    headers={"Authorization": "Bearer exonit-default-token"})
    assert r.status_code == 200
    row = db.conn.execute(
        "SELECT lc_load_recovered FROM tasks WHERE task_id='t-2'"
    ).fetchone()
    assert row[0] == 0
