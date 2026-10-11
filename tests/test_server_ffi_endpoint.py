"""Tests for GET /api/ffi/{tic_id}."""
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    os.environ["EXONIT_DB_PATH"] = str(tmp_path / "test.db")
    os.environ["EXONIT_REPORTS_DIR"] = str(tmp_path / "reports")
    os.makedirs(tmp_path / "reports", exist_ok=True)

    from exonit.server import create_app
    app = create_app()
    c = TestClient(app, headers={"Authorization": "Bearer exonit-default-token"})
    yield c

    del os.environ["EXONIT_DB_PATH"]
    del os.environ["EXONIT_REPORTS_DIR"]


def test_ffi_endpoint_returns_404_for_unknown_tic(client, monkeypatch):
    import exonit.server as server_mod
    monkeypatch.setattr(server_mod, "_load_ffi_for_tic", lambda tic_id: None)
    r = client.get("/api/ffi/999999999999")
    assert r.status_code == 404


def test_ffi_endpoint_returns_payload_for_known_tic(client, monkeypatch):
    import exonit.server as server_mod
    captured = {}

    def fake_loader(tic_id):
        captured["tic"] = tic_id
        return [{
            "sector": 14,
            "source": "tglc",
            "time": [0.0, 0.5, 1.0],
            "flux": [1.0, 0.99, 1.0],
            "quality": [0, 0, 0],
        }]

    monkeypatch.setattr(server_mod, "_load_ffi_for_tic", fake_loader)
    r = client.get("/api/ffi/12345")
    assert r.status_code == 200
    data = r.json()
    assert data["tic_id"] == "12345"
    assert len(data["sectors"]) == 1
    assert data["sectors"][0]["sector"] == 14
    assert captured["tic"] == "12345"


def test_ffi_endpoint_returns_401_without_auth(tmp_path):
    os.environ["EXONIT_DB_PATH"] = str(tmp_path / "test.db")
    os.environ["EXONIT_REPORTS_DIR"] = str(tmp_path / "reports")
    os.makedirs(tmp_path / "reports", exist_ok=True)

    from exonit.server import create_app
    app = create_app()
    client_no_auth = TestClient(app)
    r = client_no_auth.get("/api/ffi/12345")
    assert r.status_code == 401

    del os.environ["EXONIT_DB_PATH"]
    del os.environ["EXONIT_REPORTS_DIR"]
