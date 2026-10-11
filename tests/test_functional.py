"""Functional tests for ExoNit data integrity pipeline.

Verify that golden worker results flow correctly through the FastAPI server
into SQLite (stars, candidates, completeness, tasks) and that telemetry,
update manifests, and edge cases behave as expected.
"""

import base64
import numpy as np
import pytest

from pathlib import Path
from exonit.db import ExoNitDB

# load_golden is defined in tests/conftest.py — import via direct path
# to avoid collision with root conftest.py
import json

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_golden(name: str) -> dict:
    """Load a golden fixture JSON file from tests/fixtures/."""
    path = _FIXTURES_DIR / name
    return json.loads(path.read_text())


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _submit_golden(client, headers, golden, tmp_path):
    """Enqueue a task, assign it to the worker, submit golden result.

    Returns an ExoNitDB instance for assertions (caller must close).
    """
    # 1. Enqueue
    time_arr = np.linspace(0, 100, golden.get("n_points", 1000))
    flux_arr = np.ones_like(time_arr)
    enq = client.post("/api/worker/enqueue", headers=headers, json={
        "tic_id": golden["tic_id"],
        "time": base64.b64encode(time_arr.tobytes()).decode(),
        "flux": base64.b64encode(flux_arr.tobytes()).decode(),
        "n_points": golden.get("n_points", 1000),
        "n_sectors": golden.get("n_sectors", 1),
    })
    assert enq.status_code == 200
    task_id = enq.json()["task_id"]

    # 2. Worker picks up the task
    pick = client.get("/api/worker/task", headers=headers)
    assert pick.status_code == 200
    assert pick.json()["task_id"] == task_id

    # 3. Submit result with real task_id
    result_body = {**golden, "task_id": task_id}
    resp = client.post("/api/worker/result", headers=headers, json=result_body)
    assert resp.status_code == 200

    db = ExoNitDB(str(tmp_path / "test.db"))
    return db


def _is_period_match(detected, expected, tolerance=0.10):
    """Check if detected period matches expected within tolerance,
    including 1:2 and 2:1 harmonics."""
    for ratio in [1.0, 0.5, 2.0]:
        if abs(detected - expected * ratio) / (expected * ratio) <= tolerance:
            return True
    return False


# ------------------------------------------------------------------ #
# Test 1: Pipeline E2E (class with 5 sub-tests)
# ------------------------------------------------------------------ #

class TestPipelineE2E:
    """End-to-end pipeline tests using golden_tic_150430484."""

    GOLDEN_NAME = "golden_tic_150430484.json"
    TIC_ID = "150430484"
    EXPECTED_PERIOD = 5.154

    @pytest.fixture(autouse=True)
    def setup(self, functional_client, tmp_path):
        client, worker_id, headers, tmp_path = functional_client
        golden = load_golden(self.GOLDEN_NAME)
        self.db = _submit_golden(client, headers, golden, tmp_path)
        self.client = client
        self.headers = headers
        self.tmp_path = tmp_path
        yield
        self.db.close()

    def test_star_saved_correctly(self):
        star = self.db.get_star(self.TIC_ID)
        assert star is not None, "Star should exist in DB"
        assert star["status"] == "completed"
        assert star["n_datapoints"] > 0
        assert star["n_sectors"] > 0

    def test_candidate_detected(self):
        candidates = self.db.get_candidates(self.TIC_ID)
        assert len(candidates) >= 1, "Should have at least 1 candidate"
        high_snr = [c for c in candidates if c["snr"] >= 7]
        assert len(high_snr) >= 1, "At least 1 candidate with SNR >= 7"
        matched = any(
            _is_period_match(c["period_days"], self.EXPECTED_PERIOD)
            for c in high_snr
        )
        assert matched, f"No candidate matches expected period {self.EXPECTED_PERIOD}d"

    def test_cnn_and_fpp_present(self):
        candidates = self.db.get_candidates(self.TIC_ID)
        best = max(candidates, key=lambda c: c["snr"])
        assert best["cnn_score"] is not None, "Best candidate should have cnn_score"
        assert best["triceratops_fpp"] is not None, "Best candidate should have triceratops_fpp"

    def test_completeness_saved(self):
        rows = self.db._query(
            "SELECT * FROM completeness WHERE tic_id = ?", (self.TIC_ID,)
        )
        assert len(rows) > 0, "Completeness rows should exist for this star"

    def test_task_completed(self):
        tasks = self.db._query(
            "SELECT * FROM tasks WHERE tic_id = ?", (self.TIC_ID,)
        )
        assert len(tasks) >= 1
        task = tasks[0]
        assert task["status"] == "completed"
        assert task["processing_time"] is not None
        assert task["processing_time"] > 0


# ------------------------------------------------------------------ #
# Test 2: Anti fill-gaps
# ------------------------------------------------------------------ #

def test_anti_fill_gaps(functional_client):
    """Stars with gaps should NOT inflate n_datapoints beyond sector cap."""
    client, worker_id, headers, tmp_path = functional_client
    golden = load_golden("golden_tic_254113311.json")
    db = _submit_golden(client, headers, golden, tmp_path)
    try:
        star = db.get_star("254113311")
        assert star is not None
        assert star["n_datapoints"] <= star["n_sectors"] * 30000, (
            f"n_datapoints={star['n_datapoints']} exceeds "
            f"n_sectors*30000={star['n_sectors'] * 30000}"
        )
    finally:
        db.close()


# ------------------------------------------------------------------ #
# Test 3: Completeness saved with valid probabilities
# ------------------------------------------------------------------ #

def test_completeness_saved(functional_client):
    """Completeness grid should have >= 10 rows, all probs in [0, 1]."""
    client, worker_id, headers, tmp_path = functional_client
    golden = load_golden("golden_tic_150430484.json")
    db = _submit_golden(client, headers, golden, tmp_path)
    try:
        rows = db._query(
            "SELECT * FROM completeness WHERE tic_id = ?", ("150430484",)
        )
        assert len(rows) >= 10, f"Expected >= 10 completeness rows, got {len(rows)}"
        for row in rows:
            assert 0.0 <= row["detection_prob"] <= 1.0, (
                f"detection_prob={row['detection_prob']} out of [0,1]"
            )
    finally:
        db.close()


def test_completeness_write_failure_is_non_fatal(functional_client, monkeypatch):
    """add_completeness_many raising must not fail POST /api/worker/result nor
    skip complete_task — completeness is auxiliary (Fase 4a Task 4, follow-up
    d: exonit/routes/workers.py wraps the write in the same non-fatal
    try/except pattern as the neighboring FPP/ExoMiner producer hooks)."""
    client, worker_id, headers, tmp_path = functional_client
    golden = load_golden("golden_tic_150430484.json")

    def _boom(self, tic_id, cells):
        raise RuntimeError("boom: completeness write path down")

    monkeypatch.setattr(ExoNitDB, "add_completeness_many", _boom)

    # _submit_golden already asserts the POST returns 200.
    db = _submit_golden(client, headers, golden, tmp_path)
    try:
        star = db.get_star("150430484")
        assert star is not None
        assert star["status"] == "completed"

        tasks = db._query(
            "SELECT * FROM tasks WHERE tic_id = ?", ("150430484",))
        assert len(tasks) >= 1
        assert tasks[0]["status"] == "completed"

        # The write really failed — no completeness rows persisted.
        rows = db._query(
            "SELECT * FROM completeness WHERE tic_id = ?", ("150430484",))
        assert rows == []
    finally:
        db.close()


# ------------------------------------------------------------------ #
# Test 4: Candidate consistency
# ------------------------------------------------------------------ #

def test_candidate_consistency(functional_client):
    """All candidates should have consistent physical values."""
    client, worker_id, headers, tmp_path = functional_client
    golden = load_golden("golden_tic_150430484.json")
    db = _submit_golden(client, headers, golden, tmp_path)
    try:
        candidates = db.get_candidates("150430484")
        assert len(candidates) >= 1
        for i, c in enumerate(candidates):
            assert c["period_days"] > 0, "period must be positive"
            assert 0 < c["duration_days"] < c["period_days"], (
                "duration must be between 0 and period"
            )
            assert c["depth_ppm"] > 0, "depth must be positive"
            assert c["snr"] >= 7, "SNR should be >= 7"
            if c["cnn_score"] is not None:
                assert 0.0 <= c["cnn_score"] <= 1.0, "cnn_score must be in [0,1]"
            assert c["candidate_num"] == i + 1, (
                f"candidate_num should be sequential, got {c['candidate_num']} "
                f"at position {i}"
            )
    finally:
        db.close()


# ------------------------------------------------------------------ #
# Test 5: Broken pipe — no false completion
# ------------------------------------------------------------------ #

def test_broken_pipe_no_false_completion(functional_client):
    """If worker picks a task but never submits, star must NOT appear."""
    client, worker_id, headers, tmp_path = functional_client
    tic_id = "999999999"

    # Enqueue
    time_arr = np.linspace(0, 10, 100)
    flux_arr = np.ones(100)
    enq = client.post("/api/worker/enqueue", headers=headers, json={
        "tic_id": tic_id,
        "time": base64.b64encode(time_arr.tobytes()).decode(),
        "flux": base64.b64encode(flux_arr.tobytes()).decode(),
        "n_points": 100,
        "n_sectors": 1,
    })
    assert enq.status_code == 200
    task_id = enq.json()["task_id"]

    # Worker picks it up (assigns) but NEVER submits result
    pick = client.get("/api/worker/task", headers=headers)
    assert pick.status_code == 200
    assert pick.json()["task_id"] == task_id

    db = ExoNitDB(str(tmp_path / "test.db"))
    try:
        # Star should NOT exist
        star = db.get_star(tic_id)
        assert star is None, "Star should not exist if result never submitted"

        # Task should be assigned
        task = db.get_task(task_id)
        assert task["status"] == "assigned"

        # After reassign_stale_tasks(0), task goes back to pending
        db.reassign_stale_tasks(0)
        task = db.get_task(task_id)
        assert task["status"] == "pending", (
            f"Expected 'pending' after reassign, got '{task['status']}'"
        )
    finally:
        db.close()


# ------------------------------------------------------------------ #
# Test 6: Task data returned correctly
# ------------------------------------------------------------------ #

def test_task_data_returned(functional_client):
    """GET /api/worker/task returns the enqueued task (autonomous contract).

    The legacy base64 time/flux passthrough was removed 2026-06-09: this
    test was its ONLY client — no production enqueuer ever sent arrays.
    Workers fetch data themselves from the server (autonomous mode).
    """
    client, worker_id, headers, tmp_path = functional_client

    # No tasks yet — should return task_id: None
    resp = client.get("/api/worker/task", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["task_id"] is None

    tic_id = "888888888"
    enq = client.post("/api/worker/enqueue", headers=headers, json={
        "tic_id": tic_id,
    })
    assert enq.status_code == 200
    task_id = enq.json()["task_id"]

    # Worker picks it up: only task_id + tic_id, no array payload.
    resp = client.get("/api/worker/task", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == task_id
    assert data["tic_id"] == tic_id
    assert "time" not in data and "flux" not in data


# ------------------------------------------------------------------ #
# Test 7: Recall reference stars (parametrized)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("fixture_name, expected_period", [
    ("golden_tic_150430484.json", 5.154),
    ("golden_tic_140691463.json", 2.084),
    ("golden_tic_142105158.json", 2.936),
])
def test_recall_reference_star(functional_client, fixture_name, expected_period):
    """Known transiting systems must be recovered with correct period."""
    client, worker_id, headers, tmp_path = functional_client
    golden = load_golden(fixture_name)
    db = _submit_golden(client, headers, golden, tmp_path)
    try:
        candidates = db.get_candidates(golden["tic_id"])
        assert len(candidates) >= 1, f"No candidates for {golden['tic_id']}"
        matched = any(
            _is_period_match(c["period_days"], expected_period)
            for c in candidates
        )
        assert matched, (
            f"No candidate matches expected period {expected_period}d "
            f"(got: {[c['period_days'] for c in candidates]})"
        )
    finally:
        db.close()


# ------------------------------------------------------------------ #
# Test 8: Worker update manifest
# ------------------------------------------------------------------ #

def test_worker_update_manifest(functional_client):
    """GET /api/worker/update returns valid file manifest."""
    client, worker_id, headers, tmp_path = functional_client

    # No auth needed for update manifest
    resp = client.get("/api/worker/update")
    assert resp.status_code == 200
    data = resp.json()
    files = data["files"]
    assert len(files) > 0, "Manifest should list at least one file"

    for entry in files:
        assert "filename" in entry
        assert "sha256" in entry
        assert len(entry["sha256"]) == 64, (
            f"SHA256 hash should be 64 chars, got {len(entry['sha256'])}"
        )

    filenames = [f["filename"] for f in files]
    assert "worker.py" in filenames, "worker.py must be in update manifest"


# ------------------------------------------------------------------ #
# Test 9: Worker name in telemetry
# ------------------------------------------------------------------ #

def test_worker_name_in_telemetry(functional_client):
    """After submitting a result, telemetry should contain the worker name."""
    client, worker_id, headers, tmp_path = functional_client
    golden = load_golden("golden_tic_150430484.json")
    db = _submit_golden(client, headers, golden, tmp_path)
    db.close()

    resp = client.get("/api/telemetry")
    assert resp.status_code == 200
    data = resp.json()
    records = data["records"]
    assert len(records) > 0, "Telemetry should have at least one record"

    worker_names = [r["worker"] for r in records]
    assert "test-functional-worker" in worker_names, (
        f"Expected 'test-functional-worker' in telemetry, got: {worker_names}"
    )
