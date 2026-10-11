"""Endpoints for pipeline progress.

Constraints: temp DB, temp pipeline_state.json, MemoryBackend shared state
(tests/conftest.py's _isolate_shared_state autouse fixture). No real Redis.
"""

import json

import pytest
from fastapi.testclient import TestClient

TOKEN = "test-token"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE",
                       str(tmp_path / "pipeline_state.json"))
    monkeypatch.setenv("EXONIT_METADATA_CACHE", str(tmp_path / "meta.db"))
    monkeypatch.delenv("EXONIT_METADATA_DB_URL", raising=False)
    monkeypatch.setenv("EXONIT_WORKER_TOKEN", TOKEN)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

    from exonit.server import create_app
    with TestClient(create_app()) as c:
        yield c, tmp_path


def _auth():
    return {"Authorization": f"Bearer {TOKEN}"}


def _mark(**over):
    base = {"run_id": "run-A", "phase": "FINALIZE",
            "substep": "compact_completeness", "unit": "buckets",
            "done": 3, "total": 56, "interval_s": 10}
    base.update(over)
    return base


def _write_pipeline(tmp_path, **fields):
    (tmp_path / "pipeline_state.json").write_text(json.dumps(fields))


def test_post_requires_auth(client):
    c, _ = client
    assert c.post("/api/progress", json=_mark()).status_code == 401


def test_post_rejects_malformed_payload(client):
    c, _ = client
    r = c.post("/api/progress", json=_mark(run_id=""), headers=_auth())
    assert r.status_code == 400
    assert "run_id" in r.json()["error"]


def test_post_then_pipeline_get_round_trip(client):
    c, tmp_path = client
    _write_pipeline(tmp_path, state="FINALIZE", run_id="run-A")
    assert c.post("/api/progress", json=_mark(), headers=_auth()).status_code == 200

    body = c.get("/api/discovery/pipeline").json()
    assert body["state"] == "FINALIZE"
    assert body["progress"]["done"] == 3
    assert body["progress"]["total"] == 56
    assert body["progress"]["substep"] == "compact_completeness"
    assert body["progress"]["stale"] is False


def test_pipeline_get_has_null_progress_when_nothing_reported(client):
    c, tmp_path = client
    _write_pipeline(tmp_path, state="STOP_WORKERS", run_id="run-A")
    assert c.get("/api/discovery/pipeline").json()["progress"] is None


def test_pipeline_get_ignores_progress_of_another_phase(client):
    c, tmp_path = client
    _write_pipeline(tmp_path, state="FREE_GPU", run_id="run-A")
    c.post("/api/progress", json=_mark(), headers=_auth())
    assert c.get("/api/discovery/pipeline").json()["progress"] is None


def test_out_of_order_marks_do_not_regress(client):
    c, tmp_path = client
    _write_pipeline(tmp_path, state="FINALIZE", run_id="run-A")
    c.post("/api/progress", json=_mark(done=40), headers=_auth())
    c.post("/api/progress", json=_mark(done=2), headers=_auth())
    assert c.get("/api/discovery/pipeline").json()["progress"]["done"] == 40


def test_pipeline_get_never_500s_when_shared_state_is_broken(client, monkeypatch):
    c, tmp_path = client
    _write_pipeline(tmp_path, state="FINALIZE", run_id="run-A")

    import exonit.progress as P

    def boom(*a, **kw):
        raise RuntimeError("redis is down")

    monkeypatch.setattr(P, "read_progress", boom)
    r = c.get("/api/discovery/pipeline")
    assert r.status_code == 200
    assert r.json()["progress"] is None


def test_pipeline_get_still_works_without_a_state_file(client):
    c, _ = client
    body = c.get("/api/discovery/pipeline").json()
    assert body["state"] == "IDLE"
    assert body["never_run"] is True
    assert body["progress"] is None


def test_enqueueing_progress_is_read_under_the_fixed_key(client):
    """The scan endpoint reports ENQUEUEING under ENQUEUEING_RUN_ID, and the
    pipeline endpoint must look it up there even though the state file still
    holds the previous run's id."""
    c, tmp_path = client
    from exonit.progress import ENQUEUEING_RUN_ID
    _write_pipeline(tmp_path, state="ENQUEUEING", run_id="run-anterior")
    c.post("/api/progress",
           json=_mark(run_id=ENQUEUEING_RUN_ID, phase="ENQUEUEING",
                      substep="enqueue_from_csv", unit="TICs",
                      done=1200, total=300000),
           headers=_auth())
    body = c.get("/api/discovery/pipeline").json()
    assert body["progress"]["done"] == 1200
    assert body["progress"]["unit"] == "TICs"


# --- server-side feeder for /api/health/work -----------------------------

def _materialise_run_tics(tmp_path, run_id="run-A", tics=(1, 2, 3)):
    """health/work only scopes to a run whose TIC set exists — and the feeder
    only runs on a scoped request, because the unscoped payload holds GLOBAL
    counters that would be wrong as this run's progress."""
    from exonit.db import ExoNitDB
    ExoNitDB(str(tmp_path / "test.db")).add_discovery_run_tics(run_id, list(tics))


def test_health_work_feeds_the_current_queue_phase(client, monkeypatch):
    """Polling health/work with a run_id is what keeps queue progress fresh."""
    c, tmp_path = client
    _materialise_run_tics(tmp_path)
    _write_pipeline(tmp_path, state="SCANNING", run_id="run-A")

    import exonit.progress as P
    seen = {}
    real = P.feed_queue_phase

    def spy(state, **kw):
        seen.update(kw)
        return real(state, **kw)

    monkeypatch.setattr(P, "feed_queue_phase", spy)
    assert c.get("/api/health/work?run_id=run-A").status_code == 200
    assert seen["run_id"] == "run-A"
    assert seen["phase"] == "SCANNING"
    assert "primary_tasks" in seen["health"]


def test_health_work_feeds_the_record_on_the_nocache_path_too(client, monkeypatch):
    """The orchestrator's own poll must keep the progress record alive.

    Regression, 2026-08-01: nocache=1 returned before the feeder ever ran, so
    the ONLY thing refreshing a queue phase's progress record was an open
    dashboard. With no browser watching, the record expired mid-phase
    (PROGRESS_TTL_S) and the next poll rebuilt it — re-capturing the FPP
    baseline against whatever the queue happened to be at that instant, which
    is what put a fake 30k-item step into the rate window.
    """
    c, tmp_path = client
    _materialise_run_tics(tmp_path)
    _write_pipeline(tmp_path, state="SCANNING", run_id="run-A")

    import exonit.progress as P
    seen = {}
    real = P.feed_queue_phase

    def spy(state, **kw):
        seen.update(kw)
        return real(state, **kw)

    monkeypatch.setattr(P, "feed_queue_phase", spy)
    assert c.get("/api/health/work?nocache=1&run_id=run-A").status_code == 200
    assert seen["run_id"] == "run-A"
    assert seen["phase"] == "SCANNING"


def test_health_work_payload_carries_its_observation_time(client):
    """Consumers measuring a rate need to know when the counters were read."""
    import time as _t
    c, tmp_path = client
    _materialise_run_tics(tmp_path)
    body = c.get("/api/health/work?nocache=1&run_id=run-A").json()
    assert "computed_at" in body
    assert abs(_t.time() - body["computed_at"]) < 60.0


def test_health_work_does_not_feed_another_run(client, monkeypatch):
    c, tmp_path = client
    _materialise_run_tics(tmp_path, run_id="run-B")
    _write_pipeline(tmp_path, state="SCANNING", run_id="run-A")

    import exonit.progress as P
    calls = []
    monkeypatch.setattr(P, "feed_queue_phase",
                        lambda *a, **k: calls.append(k))
    c.get("/api/health/work?run_id=run-B")
    assert calls == []


def test_health_work_never_500s_when_the_feeder_breaks(client, monkeypatch):
    c, tmp_path = client
    _materialise_run_tics(tmp_path)
    _write_pipeline(tmp_path, state="SCANNING", run_id="run-A")

    import exonit.progress as P
    monkeypatch.setattr(P, "feed_queue_phase",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert c.get("/api/health/work?run_id=run-A").status_code == 200


# --- total ETA on the pipeline endpoint -----------------------------------

def test_pipeline_get_exposes_eta_total(client):
    c, tmp_path = client
    _write_pipeline(tmp_path, state="FINALIZE", run_id="run-A")
    c.post("/api/progress", json=_mark(), headers=_auth())
    body = c.get("/api/discovery/pipeline").json()
    assert "eta_total_s" in body        # value may be null; the key must exist


def test_scoped_health_work_stays_cached(client, monkeypatch):
    """Regression: feeding progress from /api/health/work must not turn the
    scoped request into an uncached one. It is polled by the orchestrator
    every 30 s and by every open dashboard every 15 s, and each _compute()
    runs a battery of COUNT(*) over tables of millions of rows.

    Measured by counting DB queries: the second identical request inside the
    10 s TTL must issue strictly fewer than the first, because the whole
    payload comes from the cache instead of being recomputed.
    """
    c, tmp_path = client
    _materialise_run_tics(tmp_path)
    _write_pipeline(tmp_path, state="SCANNING", run_id="run-A")

    from exonit.db import ExoNitDB
    original = ExoNitDB._query
    counter = {"n": 0}

    def counting(self, *a, **kw):
        counter["n"] += 1
        return original(self, *a, **kw)

    monkeypatch.setattr(ExoNitDB, "_query", counting)

    counter["n"] = 0
    assert c.get("/api/health/work?run_id=run-A").status_code == 200
    first = counter["n"]

    counter["n"] = 0
    assert c.get("/api/health/work?run_id=run-A").status_code == 200
    second = counter["n"]

    assert first > 1, "the first scoped request should compute the payload"
    assert second < first, (
        f"second scoped request issued {second} queries vs {first} on the "
        f"first — the scoped payload is not being cached")
