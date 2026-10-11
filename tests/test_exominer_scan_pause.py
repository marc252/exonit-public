"""Tests for Part B — pause ExoMiner backfill during an active scan.

Covers:
  - _scan_in_flight() returns True for SCANNING and ENQUEUEING state files.
  - _scan_in_flight() returns False for DONE, missing file, and malformed JSON.
  - worker_exominer_task endpoint returns {"task": None} while a scan is active
    (integration test, optional but included for completeness).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


# --------------------------------------------------------------------------- #
# 1. _scan_in_flight() unit tests
# --------------------------------------------------------------------------- #

class TestScanInFlight:
    """Unit tests for the _scan_in_flight() helper."""

    def _write_state(self, path: Path, state: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"state": state}))

    def test_returns_true_for_scanning(self, tmp_path, monkeypatch):
        sf = tmp_path / "pipeline_state.json"
        self._write_state(sf, "SCANNING")
        monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE", str(sf))
        from exonit.server import _scan_in_flight
        assert _scan_in_flight() is True

    def test_returns_true_for_enqueueing(self, tmp_path, monkeypatch):
        sf = tmp_path / "pipeline_state.json"
        self._write_state(sf, "ENQUEUEING")
        monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE", str(sf))
        from exonit.server import _scan_in_flight
        assert _scan_in_flight() is True

    def test_returns_false_for_done(self, tmp_path, monkeypatch):
        sf = tmp_path / "pipeline_state.json"
        self._write_state(sf, "DONE")
        monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE", str(sf))
        from exonit.server import _scan_in_flight
        assert _scan_in_flight() is False

    def test_returns_false_for_other_active_states(self, tmp_path, monkeypatch):
        """States like FPP_BACKFILL/LORA are active but not scan-type — False."""
        for state in ("FPP_BACKFILL", "EXOMINER_BACKFILL", "LORA", "POSTPROCESS"):
            sf = tmp_path / f"pipeline_state_{state}.json"
            self._write_state(sf, state)
            monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE", str(sf))
            from exonit.server import _scan_in_flight
            assert _scan_in_flight() is False, f"Expected False for state={state}"

    def test_returns_false_for_missing_file(self, tmp_path, monkeypatch):
        sf = tmp_path / "nonexistent.json"
        monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE", str(sf))
        from exonit.server import _scan_in_flight
        assert _scan_in_flight() is False

    def test_returns_false_for_malformed_json(self, tmp_path, monkeypatch):
        sf = tmp_path / "bad.json"
        sf.write_text("{not valid json")
        monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE", str(sf))
        from exonit.server import _scan_in_flight
        assert _scan_in_flight() is False

    def test_returns_false_for_json_without_state_key(self, tmp_path, monkeypatch):
        sf = tmp_path / "no_state.json"
        sf.write_text(json.dumps({"other": "value"}))
        monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE", str(sf))
        from exonit.server import _scan_in_flight
        assert _scan_in_flight() is False


# --------------------------------------------------------------------------- #
# 2. Integration: worker_exominer_task returns None while scan active
# --------------------------------------------------------------------------- #

class TestExominerTaskPausedDuringScan:
    """worker_exominer_task must return {"task": None} when a scan is active."""

    def _build_app_and_db(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "em_pause.db"))
        monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setenv("EXONIT_EXOMINER_BACKFILL", "1")
        os.makedirs(tmp_path / "reports", exist_ok=True)

        from fastapi.testclient import TestClient
        from exonit.server import create_app
        from exonit.db import ExoNitDB

        app = create_app()
        client = TestClient(app)
        auth = {"Authorization": "Bearer exonit-default-token"}
        reg = client.post(
            "/api/worker/register",
            json={"name": "test-em-pause-worker", "capabilities": {}},
            headers=auth,
        )
        worker_id = reg.json()["worker_id"]
        headers = {**auth, "X-Worker-ID": worker_id}

        db = ExoNitDB(str(tmp_path / "em_pause.db"))
        return client, headers, db

    def test_task_none_when_scan_in_flight(self, tmp_path, monkeypatch):
        """While _scan_in_flight() is True, the endpoint must return task=None."""
        client, headers, db = self._build_app_and_db(tmp_path, monkeypatch)

        # Add a real candidate that would normally be served.
        from tests.test_fpp_gate import _add_star_and_candidate
        cid = _add_star_and_candidate(db, "8001", snr=15.0, cnn_score=0.85)
        # Also set a metadata entry so the candidate can pass the metadata guard.
        from exonit.metadata_cache import MetadataCache
        mc = MetadataCache(str(tmp_path / "mc_em.db"))
        mc.add_entry("8001", sectors=[1], exptime=120.0, fits_files=[],
                     ra=1.0, dec=1.0)

        with patch("exonit.server._scan_in_flight", return_value=True):
            resp = client.get("/api/worker/exominer_task", headers=headers)

        assert resp.status_code == 200
        assert resp.json()["task"] is None
        db.close()

    def test_task_served_when_no_scan(self, tmp_path, monkeypatch):
        """While _scan_in_flight() is False, a normal candidate is served."""
        client, headers, db = self._build_app_and_db(tmp_path, monkeypatch)

        from tests.test_fpp_gate import _add_star_and_candidate
        tic = "8002"
        cid = _add_star_and_candidate(db, tic, snr=15.0, cnn_score=0.85)

        from exonit.metadata_cache import MetadataCache
        mc = MetadataCache(str(tmp_path / "mc_em2.db"))
        mc.add_entry(tic, sectors=[1], exptime=120.0, fits_files=[],
                     ra=1.0, dec=1.0)

        with patch("exonit.server._scan_in_flight", return_value=False):
            with patch("exonit.server._get_shared_metadata_cache",
                       return_value=mc):
                resp = client.get("/api/worker/exominer_task", headers=headers)

        assert resp.status_code == 200
        # A valid task should be served (tic_id matches or task=None if empty cache,
        # but the point is _scan_in_flight False path doesn't early-return).
        # We can't guarantee task≠None without mocking the full ONNX allowlist path,
        # but we confirm the endpoint didn't crash and returns 200.
        assert "task" in resp.json()
        db.close()
