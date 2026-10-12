"""Worker-result handler: candidate sector comes from the task row (P1)."""
from unittest.mock import patch

import pytest

from exonit.db import ExoNitDB

AUTH = {"Authorization": "Bearer exonit-default-token", "x-worker-id": "w"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE", str(tmp_path / "ps.json"))
    from exonit.server import create_app
    from fastapi.testclient import TestClient
    with TestClient(create_app()) as c:
        yield c


def _post(client, task_id, tic, n_sectors, period):
    body = {"task_id": task_id, "tic_id": tic, "status": "completed",
            "n_sectors": n_sectors, "n_points": 1000, "time_span_days": 26.0,
            "ra": 1.0, "dec": 2.0, "tmag": 10.0,
            "candidates": [{"candidate_num": 1, "period_days": period, "t0": 0.0,
                            "duration_days": 0.1, "depth_ppm": 500.0, "snr": 12.0}],
            "cnn_scores": [0.8], "processing_time_seconds": 1.0}
    return client.post("/api/worker/result", json=body, headers=AUTH)


def _rows(path, tic):
    return [(r["sector"], r["period_days"]) for r in
            sorted(ExoNitDB(path).get_candidates(tic), key=lambda r: r["sector"])]


def test_straggler_result_lands_in_task_sector(client, tmp_path):
    path = str(tmp_path / "test.db")
    ExoNitDB(path).add_task("t-700", "700", sector=6)
    with patch("exonit.routes.workers._active_run_sector", return_value=None):
        assert _post(client, "t-700", "700", 1, 3.0).status_code == 200
    assert _rows(path, "700") == [(6, 3.0)]


def test_reobservation_keeps_other_sector_even_if_deeper(client, tmp_path):
    path = str(tmp_path / "test.db")
    db = ExoNitDB(path)
    db.add_task("t-701a", "701", sector=6)
    db.add_task("t-701b", "701", sector=7)
    with patch("exonit.routes.workers._active_run_sector", return_value=None):
        _post(client, "t-701a", "701", 1, 3.0)
        # n_sectors 2 > stored 1 -> replace-when-deeper, scoped to sector 7
        _post(client, "t-701b", "701", 2, 6.01)
    assert _rows(path, "701") == [(6, 3.0), (7, 6.01)]
