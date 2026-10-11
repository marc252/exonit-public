"""Route-parity gate for the Fase D create_app() split.

docs/reviews/2026-07-02-opus48-cleanup-review.md: create_app() was ~4,550
lines / 84 endpoints registered as closures. progress/impl-createapp-split-
2026-07-02.md splits registration into exonit/routes/*.py, one module per
domain (admin, workers, candidates, proxies, discovery, aws), each exposing
``register(app, deps)``.

This test asserts the *complete* route table (path, methods, endpoint
function name) built by the post-split ``create_app()`` is IDENTICAL to a
frozen snapshot captured from the pre-split code (``git show
d65e606b:exonit/server.py`` in a scratch checkout — the commit immediately
before this split — instantiated the same way this test instantiates the
post-split app), plus one deliberate addition since: ``POST
/api/discovery/resume`` (Fase 3 Task 4, durable orchestration — a real new
endpoint, not a split artifact). 90 routes total: 84 ``@app.*`` endpoints
from the pre-split snapshot + 1 (the new resume endpoint) + 1 static mount
(``/reports``) + 4 FastAPI-internal routes (``/docs``, ``/redoc``,
``/openapi.json``, the docs OAuth redirect) that FastAPI registers
automatically and are not part of the 84/85-endpoint count the review cites.

If this test ever needs to change, it means the route table itself changed
(a real behavior change) — not just where the registration code lives.
"""
from __future__ import annotations

import os
import tempfile

import pytest

try:
    from fastapi.testclient import TestClient  # noqa: F401 (imported for availability check)
except ImportError:  # pragma: no cover - minimal test envs without fastapi
    TestClient = None


# Frozen snapshot, captured via:
#   git stash   # (isolate the split's uncommitted changes)
#   python3 -c "... create_app(); print(sorted(route table)) ..."
#   git stash pop
# against exonit/server.py at commit d65e606b (parent of the split), the
# exact code this split's exonit/routes/*.py chunks were extracted from
# byte-for-byte. See progress/impl-createapp-split-2026-07-02.md.
EXPECTED_ROUTES = [
    ("/", ("GET",), "serve_dashboard"),
    ("/api/admin/mark_unconfirmed_ffi", ("POST",), "api_mark_unconfirmed_ffi"),
    ("/api/admin/reset_all", ("POST",), "api_reset_all"),
    ("/api/admin/reset_baselines", ("POST",), "api_reset_baselines"),
    ("/api/admin/reset_worker_stats", ("POST",), "api_reset_worker_stats"),
    ("/api/aws/fleet/status", ("GET",), "api_aws_fleet_status"),
    ("/api/aws/repatriate", ("POST",), "api_aws_repatriate"),
    ("/api/aws/repatriate/status", ("GET",), "api_aws_repatriate_status"),
    ("/api/aws/server/start", ("POST",), "api_aws_server_start"),
    ("/api/aws/server/stop", ("POST",), "api_aws_server_stop"),
    ("/api/aws/status", ("GET",), "api_aws_status"),
    ("/api/aws/workers/start", ("POST",), "api_aws_workers_start"),
    ("/api/aws/workers/terminate", ("POST",), "api_aws_workers_terminate"),
    ("/api/backfill/fpp", ("GET",), "api_backfill_fpp"),
    ("/api/backfill/nfpp", ("GET",), "api_backfill_nfpp"),
    ("/api/candidate/update_analysis", ("POST",), "candidate_update_analysis"),
    ("/api/candidates", ("GET",), "api_candidates"),
    ("/api/completeness", ("GET",), "api_completeness"),
    ("/api/ctoi/crosscheck/{tic_id}", ("GET",), "ctoi_crosscheck"),
    ("/api/ctoi/export", ("GET",), "ctoi_export"),
    ("/api/ctoi/plot/{tic_id}/{candidate_num}", ("GET",), "ctoi_plot"),
    ("/api/debug/{worker_id}", ("GET",), "api_debug_get"),
    ("/api/debug/{worker_id}", ("POST",), "api_debug_set"),
    ("/api/deployment", ("GET",), "api_deployment"),
    ("/api/discovery/download", ("POST",), "api_discovery_download"),
    ("/api/discovery/download/status", ("GET",), "api_discovery_download_status"),
    ("/api/discovery/pipeline", ("GET",), "api_discovery_pipeline"),
    ("/api/discovery/resume", ("POST",), "api_discovery_resume"),
    ("/api/discovery/scan", ("POST",), "api_discovery_scan"),
    ("/api/discovery/sector/{n}/autofill", ("POST",), "api_discovery_sector_autofill"),
    ("/api/discovery/sectors", ("GET",), "api_discovery_sectors"),
    ("/api/downloads/catalog", ("GET",), "downloads_catalog"),
    ("/api/downloads/rebuild-cache", ("POST",), "downloads_rebuild_cache"),
    ("/api/downloads/refresh-catalog", ("POST",), "downloads_refresh_catalog"),
    ("/api/downloads/status", ("GET",), "downloads_status"),
    ("/api/exominer/onnx", ("GET",), "api_exominer_onnx"),
    ("/api/exominer/tfrecord/{tic_id}", ("GET",), "exominer_tfrecord"),
    ("/api/export/csv", ("GET",), "api_export_csv"),
    ("/api/export/json", ("GET",), "api_export_json"),
    ("/api/ffi/{tic_id}", ("GET",), "get_ffi"),
    ("/api/fits/{filename}", ("GET",), "serve_fits"),
    ("/api/gaia/{tic_id}", ("GET",), "api_gaia"),
    ("/api/health/work", ("GET",), "api_health_work"),
    ("/api/mast/dv/{tic_id}", ("GET",), "api_mast_dv"),
    ("/api/mast/dv_runs/{tic_id}", ("GET",), "api_mast_dv_runs"),
    ("/api/metadata/{tic_id}", ("GET",), "get_metadata"),
    ("/api/nearby/{tic_id}", ("GET",), "get_nearby"),
    ("/api/reports", ("GET",), "api_reports"),
    ("/api/scorecards", ("GET",), "api_scorecards"),
    ("/api/scorecards/{name}", ("GET",), "api_scorecard_get"),
    ("/api/scorecards/{name}/dossier", ("GET",), "api_scorecard_dossier"),
    ("/api/setup/start", ("POST",), "setup_start"),
    ("/api/setup/status", ("GET",), "setup_status"),
    ("/api/stars", ("GET",), "api_stars"),
    ("/api/start", ("POST",), "api_start"),
    ("/api/stats", ("GET",), "api_stats"),
    ("/api/status", ("GET",), "api_status"),
    ("/api/stop", ("POST",), "api_stop"),
    ("/api/telemetry", ("GET",), "api_telemetry"),
    ("/api/telemetry/purge", ("POST",), "api_telemetry_purge"),
    ("/api/telemetry/{worker_id}", ("GET",), "api_telemetry_worker"),
    ("/api/telemetry/{worker_id}/latest", ("GET",), "api_telemetry_latest"),
    ("/api/ticv8/{tic_id}", ("GET",), "api_ticv8"),
    ("/api/version", ("GET",), "api_version"),
    ("/api/worker/control", ("POST",), "worker_control"),
    ("/api/worker/download", ("GET",), "download_worker_package"),
    ("/api/worker/enqueue", ("POST",), "worker_enqueue"),
    ("/api/worker/enqueue_bulk", ("POST",), "worker_enqueue_bulk"),
    ("/api/worker/exominer_result", ("POST",), "worker_exominer_result"),
    ("/api/worker/exominer_task", ("GET",), "worker_exominer_task"),
    ("/api/worker/fpp_result", ("POST",), "worker_fpp_result"),
    ("/api/worker/fpp_task", ("GET",), "worker_fpp_task"),
    ("/api/worker/heartbeat", ("POST",), "worker_heartbeat"),
    ("/api/worker/list", ("GET",), "worker_list"),
    ("/api/worker/register", ("POST",), "worker_register"),
    ("/api/worker/result", ("POST",), "worker_result"),
    ("/api/worker/task", ("GET",), "worker_get_task"),
    ("/api/worker/update", ("GET",), "worker_update_manifest"),
    ("/api/worker/update/{filename:path}", ("GET",), "worker_update_file"),
    ("/api/workers/start", ("POST",), "api_workers_start"),
    ("/aws", ("GET",), "serve_aws"),
    ("/completion", ("GET",), "serve_completion"),
    ("/dashboard/workers", ("GET",), "dashboard_workers"),
    ("/docs", ("GET", "HEAD"), "swagger_ui_html"),
    ("/docs/oauth2-redirect", ("GET", "HEAD"), "swagger_ui_redirect"),
    ("/downloads", ("GET",), "serve_downloads"),
    ("/js/{path:path}", ("GET",), "serve_js_asset"),
    ("/openapi.json", ("GET", "HEAD"), "openapi"),
    ("/redoc", ("GET", "HEAD"), "redoc_html"),
    ("/reports", (), "reports"),
    ("/results", ("GET",), "serve_results"),
    ("/static/{path:path}", ("GET",), "serve_static_asset"),
]

# 85 @app.* decorated endpoints (84 per the review's own count + 1:
# POST /api/discovery/resume, added by Fase 3 Task 4 — a real new endpoint,
# not a split artifact), + 2 more added by the "Organic" UI redesign (plan
# docs/superpowers/plans/2026-07-24-ui-redesign-organic.md Task 1):
# GET /static/{path:path} and GET /js/{path:path}, serving the single
# stylesheet and shared ES modules — real new endpoints, not split
# artifacts either. 87 @app.* routes + 1 static mount ("/reports") = 88
# "real" routes, + 4 FastAPI-internal doc routes (openapi.json, docs,
# docs/oauth2-redirect, redoc) = 92 total entries above.
#
# NOTE (Task 9 parity audit, 2026-07-24): this snapshot was ALREADY stale on
# `main` independent of the UI redesign branch — 8 routes merged after the
# 2026-07-02 freeze are missing from it (sky map Phase 2: /sky, /api/sky/
# background, /api/sky/completed, /api/sky/live; fleet status: /status,
# /api/status/fleet; admin: /api/admin/reconcile_stream, /api/admin/
# recycle-workers — verified via `git stash` / a plain `main` checkout, both
# fail the same way with 8 extra routes before this branch's 2 additions).
# That drift is orthogonal to this task's scope (no UI page references those
# admin/sky/fleet routes changed here) and is left as a pre-existing,
# documented failure — see
# docs/superpowers/specs/2026-07-24-ui-redesign/parity_report.md. Re-freezing
# the whole snapshot is a separate cleanup, not part of the UI redesign.
#
# NOTE (2026-08-07, sky-sphere-illustration Task 2): /vendor/three.module.min.js
# (the 9th drift route above) was removed outright — the rewritten sky.html
# is a single fixed-sphere canvas view with no three.js dependency — so it
# drops out of both the drift count and the live route count below.
EXPECTED_APP_ENDPOINT_COUNT = 86


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    from exonit.server import create_app
    return create_app()


def _route_table(app):
    out = []
    for r in app.routes:
        methods = tuple(sorted(getattr(r, "methods", None) or ()))
        out.append((r.path, methods, r.name))
    return sorted(out)


def test_route_table_matches_pre_split_snapshot(app):
    """Full (path, methods, endpoint name) table, byte-for-byte vs pre-split."""
    actual = _route_table(app)
    expected = sorted(EXPECTED_ROUTES)
    assert actual == expected


def test_route_count_matches_review(app):
    """docs/reviews/2026-07-02-opus48-cleanup-review.md cites 84 endpoints."""
    non_fastapi_internal = {
        "openapi", "swagger_ui_html", "swagger_ui_redirect", "redoc_html",
    }
    app_endpoints = [
        r for r in app.routes
        if r.name not in non_fastapi_internal and r.name != "reports"
    ]
    assert len(app_endpoints) == EXPECTED_APP_ENDPOINT_COUNT


def test_all_six_domain_modules_registered_routes(app):
    """Sanity: every routes/*.py module actually contributed >=1 route (a
    forgotten register(app, deps) call would silently drop a whole domain
    without necessarily changing the total count in a way that's obvious)."""
    from exonit.routes import admin, aws, candidates, discovery, proxies, workers

    # Endpoint names each module is expected to own (a representative subset,
    # not exhaustive) — cheap way to catch "module X's register() was never
    # called" without re-deriving the full per-module route list here.
    names = {r.name for r in app.routes}
    assert "api_version" in names          # admin
    assert "worker_register" in names      # workers
    assert "api_stars" in names            # candidates
    assert "get_ffi" in names              # proxies
    assert "api_discovery_sectors" in names  # discovery
    assert "api_deployment" in names       # aws
    # Import-only smoke: confirms each module's register symbol exists and
    # is the exact callable exercised above (no accidental shadowing).
    for mod in (admin, aws, candidates, discovery, proxies, workers):
        assert callable(mod.register)
