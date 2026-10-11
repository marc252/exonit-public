"""Tests for GET /api/exominer/tfrecord/{tic_id}."""
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_and_nfs(tmp_path):
    os.environ["EXONIT_DB_PATH"] = str(tmp_path / "test.db")
    os.environ["EXONIT_REPORTS_DIR"] = str(tmp_path / "reports")
    os.makedirs(tmp_path / "reports", exist_ok=True)
    records_dir = tmp_path / "records"
    tces_dir = tmp_path / "tces"
    records_dir.mkdir()
    tces_dir.mkdir()

    from exonit.server import create_app
    app = create_app()
    client = TestClient(app, headers={"Authorization": "Bearer exonit-default-token"})
    yield client, records_dir, tces_dir

    del os.environ["EXONIT_DB_PATH"]
    del os.environ["EXONIT_REPORTS_DIR"]


def test_tfrecord_endpoint_returns_404_if_no_dv_runs(client_and_nfs):
    client, _, _ = client_and_nfs
    with patch("exonit.mast_dv.fetch_dv_runs", return_value=[]):
        r = client.get("/api/exominer/tfrecord/nonexistent_tic")
    assert r.status_code == 404
    assert r.json()["error"] == "no_dv_runs"


def test_tfrecord_endpoint_serves_file_on_cache_hit(client_and_nfs):
    client, records_dir, _ = client_and_nfs
    tfrecord_fp = records_dir / "382437043_1-69.tfrecord"
    tfrecord_fp.write_bytes(b"\x00\x01\x02\x03")

    from exonit.exominer_preprocess import ExoMinerPreprocessor

    with patch("exonit.mast_dv.fetch_dv_runs", return_value=[(1, 69)]), \
         patch.object(ExoMinerPreprocessor, "is_cached",
                      lambda self, t, s: True), \
         patch.object(ExoMinerPreprocessor, "_tfrecord_dest",
                      lambda self, t, s: tfrecord_fp):
        r = client.get("/api/exominer/tfrecord/382437043")

    assert r.status_code == 200
    assert r.content == b"\x00\x01\x02\x03"
    assert r.headers.get("x-sector-run") == "1-69"
    assert r.headers.get("x-tic-id") == "382437043"


def test_tfrecord_endpoint_kicks_preprocess_on_miss(client_and_nfs):
    client, _, _ = client_and_nfs
    from exonit.exominer_preprocess import ExoMinerPreprocessor, PreprocessResult

    def _fake_preprocess_tic(self, tic_id, sector_run=None, model=None):
        return PreprocessResult(tic_id, sector_run, None, None, reason="ok")

    with patch("exonit.mast_dv.fetch_dv_runs", return_value=[(14, 55)]), \
         patch.object(ExoMinerPreprocessor, "is_cached",
                      lambda self, t, s: False), \
         patch.object(ExoMinerPreprocessor, "preprocess_tic",
                      _fake_preprocess_tic):
        r = client.get("/api/exominer/tfrecord/999")

    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "preprocessing"
    assert body["sector_run"] == "14-55"
    assert body["tic_id"] == "999"
