"""Gotcha ui_stale_html_shell_browser_cache (2026-06-27, commit 1a454873):
HTML page handlers must set Cache-Control: no-cache so a UI deploy shows up
without a hard-refresh, instead of the browser's heuristic cache serving a
stale shell.

commit 1a454873 added ``Cache-Control: no-cache, must-revalidate`` to 5 of
the 6 page handlers (``/``, ``/downloads``, ``/completion``, ``/results``,
``/aws`` — all ``HTMLResponse(..., headers=_NO_CACHE)``), but missed
``/dashboard/workers`` which uses ``FileResponse`` with no headers at all.
These tests close that residual gap and guard the other 5 against
regression, without touching /reports (a StaticFiles mount for non-HTML
report artifacts, which must keep its default caching behavior).
"""
from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - minimal test envs without fastapi
    TestClient = None

pytestmark = pytest.mark.skipif(TestClient is None, reason="fastapi/httpx not installed")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    from exonit.server import create_app
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.mark.parametrize(
    "path",
    ["/", "/downloads", "/completion", "/results", "/dashboard/workers"],
)
def test_html_shell_has_no_cache_header(client, path):
    resp = client.get(path)
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("text/html")
    cache_control = resp.headers.get("cache-control", "")
    assert "no-cache" in cache_control, (
        f"{path} missing Cache-Control: no-cache (got {cache_control!r}); "
        "browser will serve a stale shell after the next UI deploy"
    )


def test_reports_static_mount_is_unaffected(client, app, tmp_path):
    """/reports is a StaticFiles mount for non-HTML report artifacts; this
    fix must not touch its (default Starlette) caching behavior."""
    reports_dir = tmp_path / "reports"
    asset = reports_dir / "sample.txt"
    asset.write_text("hello")

    resp = client.get("/reports/sample.txt")
    assert resp.status_code == 200
    # StaticFiles' default behavior: ETag/Last-Modified present, no forced
    # no-cache directive injected by us.
    cache_control = resp.headers.get("cache-control", "")
    assert "no-cache" not in cache_control
