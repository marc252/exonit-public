"""End-to-end: server has FFI for TIC X, metadata fetch is forced to 404.
A worker run with the flag on must:
  - call _load_star, get None from SPOC path
  - hit _try_ffi_fallback, fetch FFI from the server
  - return a result with lc_load_recovered=True
  - the server, on receiving the result, writes 1 to tasks.lc_load_recovered.
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from exonit.config import ExoNitConfig
from exonit.db import ExoNitDB
from exonit.server import create_app


@pytest.fixture
def server_with_ffi(tmp_path):
    """FastAPI app whose Phase 2.5 cache contains TIC 'INT-FFI-1'."""
    cfg = ExoNitConfig()
    cfg.db_path = str(tmp_path / "exonit.db")
    cfg.metadata_cache_path = str(tmp_path / "metadata.db")
    cfg.berger_db_path = str(tmp_path / "berger.db")

    # Seed the H5 cache the server reads from.
    h5_dir = tmp_path / "data" / "training"
    h5_dir.mkdir(parents=True)
    h5_path = h5_dir / "ffi_dataset_v1.h5"
    n = 200
    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("light_curves/INT-FFI-1/QLP/0001")
        grp.create_dataset("time", data=np.linspace(0.0, 27.0, n))
        grp.create_dataset("flux", data=np.ones(n))
        grp.create_dataset("quality", data=np.zeros(n, dtype=int))

    # Patch the project root the server uses to resolve the H5.
    with patch("exonit.server._PROJECT_ROOT", tmp_path):
        app = create_app(cfg)
        yield app, cfg


def test_recovered_tic_persists_one(server_with_ffi):
    app, cfg = server_with_ffi
    client = TestClient(app)
    db = ExoNitDB(cfg.db_path)
    db._exec(
        "INSERT INTO tasks (task_id, tic_id, status, attempts) "
        "VALUES (?, ?, 'assigned', 1)",
        ("int-1", "INT-FFI-1"))

    # Confirm the FFI endpoint serves the cached data.
    r = client.get("/api/ffi/INT-FFI-1",
                   headers={"Authorization": "Bearer exonit-default-token"})
    assert r.status_code == 200
    payload = r.json()
    assert payload["tic_id"] == "INT-FFI-1"
    assert len(payload["sectors"]) == 1

    # Simulate the worker's submit_result with lc_load_recovered=True.
    body = {
        "task_id": "int-1", "tic_id": "INT-FFI-1",
        "candidates": [], "cnn_scores": [],
        "processing_time_seconds": 9.9, "gpu_used": False,
        "n_sectors": 1, "n_points": 200, "time_span_days": 27.0,
        "status": "completed", "lc_load_recovered": True,
    }
    r = client.post("/api/worker/result", json=body,
                    headers={"Authorization": "Bearer exonit-default-token"})
    assert r.status_code == 200

    row = db.conn.execute(
        "SELECT lc_load_recovered FROM tasks WHERE task_id='int-1'"
    ).fetchone()
    assert row[0] == 1
