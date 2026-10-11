"""Tests for the new /api/mast/dv, /api/ticv8, /api/gaia proxy endpoints."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from exonit.server import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EXONIT_METADATA_CACHE", str(tmp_path / "meta.db"))
    monkeypatch.setenv("EXONIT_WORKER_TOKEN", "test-tok")
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    return TestClient(create_app())


AUTH = {"Authorization": "Bearer test-tok"}


def test_dv_xml_endpoint_returns_cached_bytes(client, tmp_path):
    from exonit.exominer_native.fetch_mast_dv import DVXmlStore
    store = DVXmlStore(base_dir=str(tmp_path / "dv_cache"))
    fake_xml = b"<fake xml/>"
    store.put("143143769", "1-36", fake_xml)
    with patch("exonit.server.get_dv_store", return_value=store):
        r = client.get("/api/mast/dv/143143769?sector_run=1-36", headers=AUTH)
    assert r.status_code == 200
    assert r.content == fake_xml


def test_dv_xml_endpoint_404_when_not_available(client, tmp_path):
    from exonit.exominer_native.fetch_mast_dv import DVXmlStore
    store = DVXmlStore(base_dir=str(tmp_path / "dv_cache"))
    store.mark_missing("999999999", "1-36")
    with patch("exonit.server.get_dv_store", return_value=store):
        r = client.get("/api/mast/dv/999999999?sector_run=1-36", headers=AUTH)
    assert r.status_code == 404


def test_dv_xml_endpoint_rejects_bad_auth(client):
    r = client.get("/api/mast/dv/143143769?sector_run=1-36",
                   headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_ticv8_returns_cached_json(client, tmp_path):
    from exonit.exominer_native.fetch_ticv8 import TICv8Store
    store = TICv8Store(db_path=str(tmp_path / "ticv8.db"))
    store.put("143143769", {"Tmag": 11.3, "Teff": 5776.0, "radius": 1.0})
    with patch("exonit.server.get_ticv8_store", return_value=store):
        r = client.get("/api/ticv8/143143769", headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    assert data["Tmag"] == 11.3
    assert data["Teff"] == 5776.0


def test_gaia_returns_cached_json(client, tmp_path):
    from exonit.exominer_native.fetch_gaia import GaiaStore
    store = GaiaStore(db_path=str(tmp_path / "gaia.db"))
    store.put("143143769", {"source_id": "123...", "ruwe": 0.98, "parallax": 12.5})
    with patch("exonit.server.get_gaia_store", return_value=store):
        r = client.get("/api/gaia/143143769", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["ruwe"] == 0.98


def test_onnx_endpoint_serves_file(client, tmp_path, monkeypatch):
    fake_onnx = tmp_path / "exominer.onnx"
    fake_onnx.write_bytes(b"fake-onnx-bytes")
    monkeypatch.setattr("exonit.server._ONNX_MODEL_PATH", fake_onnx)
    r = client.get("/api/exominer/onnx", headers=AUTH)
    assert r.status_code == 200
    assert r.content == b"fake-onnx-bytes"
