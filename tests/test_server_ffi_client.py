"""Tests for worker/server_ffi_client.py — Phase 3 FFI HTTP client.

Imports are done inside each test to avoid triggering worker/__init__.py
(which eagerly imports wotan via worker.worker). conftest.py appends
worker/ to sys.path so the flat `server_ffi_client` module resolves
without any sys.path manipulation here.
"""
import numpy as np
import pytest
import requests


def test_fetch_ffi_returns_concatenated_arrays():
    import requests_mock as requests_mock_lib
    from server_ffi_client import fetch_ffi

    payload = {
        "tic_id": "12345",
        "sectors": [
            {
                "sector": 14,
                "source": "tglc",
                "time": [0.0, 1.0],
                "flux": [1.0, 0.99],
                "quality": [0, 0],
            },
            {
                "sector": 15,
                "source": "tglc",
                "time": [27.0, 28.0],
                "flux": [1.0, 1.0],
                "quality": [0, 0],
            },
        ],
    }
    with requests_mock_lib.Mocker() as mock:
        mock.get("http://server:8400/api/ffi/12345", json=payload, status_code=200)
        time, flux, quality = fetch_ffi("http://server:8400", "12345", token="tok")

    assert len(time) == 4
    assert np.allclose(time, [0.0, 1.0, 27.0, 28.0])
    assert np.allclose(flux, [1.0, 0.99, 1.0, 1.0])


def test_fetch_ffi_returns_none_on_404():
    import requests_mock as requests_mock_lib
    from server_ffi_client import fetch_ffi

    with requests_mock_lib.Mocker() as mock:
        mock.get("http://server:8400/api/ffi/12345", status_code=404)
        result = fetch_ffi("http://server:8400", "12345", token="tok")

    assert result is None


def test_fetch_ffi_returns_none_on_timeout():
    import requests_mock as requests_mock_lib
    from server_ffi_client import fetch_ffi

    with requests_mock_lib.Mocker() as mock:
        mock.get(
            "http://server:8400/api/ffi/12345",
            exc=requests.exceptions.ConnectTimeout,
        )
        result = fetch_ffi(
            "http://server:8400", "12345", token="tok", timeout=0.1
        )

    assert result is None


def test_fetch_ffi_returns_none_on_malformed_sector():
    import requests_mock as requests_mock_lib
    from server_ffi_client import fetch_ffi

    payload = {
        "tic_id": "12345",
        "sectors": [
            {"sector": 14, "source": "tglc",
             "time": [0.0, 1.0]},  # missing "flux"
        ],
    }
    with requests_mock_lib.Mocker() as mock:
        mock.get("http://server:8400/api/ffi/12345", json=payload,
                 status_code=200)
        result = fetch_ffi("http://server:8400", "12345", token="tok")
    assert result is None


def test_fetch_ffi_integration_against_test_client(monkeypatch, tmp_path):
    """Integration test: fetch_ffi against the real FastAPI server using
    Authorization: Bearer auth. Closes the gap where the unit tests use
    requests-mock and never validate that the right header reaches the
    server."""
    from fastapi.testclient import TestClient
    from exonit.server import create_app
    import exonit.server as server_mod

    monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))

    # Stub the loader so we don't need a real HDF5 file.
    monkeypatch.setattr(server_mod, "_load_ffi_for_tic",
                        lambda tic_id: [{
                            "sector": 14, "source": "tglc",
                            "time": [0.0, 1.0],
                            "flux": [1.0, 0.99],
                            "quality": [0, 0],
                        }])

    app = create_app()
    client = TestClient(app)

    # Use a custom server_url that requests can use as a base via session
    # adapter — but since the easiest path is to call fetch_ffi with a
    # mocked transport that delegates to the TestClient, we monkeypatch
    # requests.get to dispatch via TestClient.
    import requests
    original_get = requests.get

    def routed_get(url, **kwargs):
        # url like 'http://server:8400/api/ffi/12345' — extract the path
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path
        headers = kwargs.get("headers", {})
        # Use the TestClient instead
        r = client.get(path, headers=headers,
                       timeout=kwargs.get("timeout", 30))

        # Wrap in a minimal Response shim so fetch_ffi's r.json() and
        # r.status_code work
        class _R:
            status_code = r.status_code
            def json(self): return r.json()
        return _R()

    monkeypatch.setattr(requests, "get", routed_get)

    from server_ffi_client import fetch_ffi
    result = fetch_ffi("http://server:8400", "12345",
                       token="exonit-default-token")
    assert result is not None, \
        "fetch_ffi must succeed end-to-end with real server auth"
    time, flux, quality = result
    assert len(time) == 2

    # Also test wrong token → fetch_ffi returns None (server returns 401)
    result_unauth = fetch_ffi("http://server:8400", "12345",
                              token="wrong-token")
    assert result_unauth is None, \
        "fetch_ffi must return None when auth rejects"
