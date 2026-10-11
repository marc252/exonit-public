"""Tests for /api/worker/fpp_task + /api/worker/fpp_result serving from the
"fpp" work stream behind EXONIT_WORK_STREAMS=1 (Fase 1, Task 3).

Flag ON: a MemoryWorkStream is injected by monkeypatching
``exonit.work_streams.get_stream`` — the handler resolves ``get_stream``
via module-attribute access on every call (``work_streams.get_stream(...)``,
not a bound import), so the patch takes effect regardless of when the app
was built.

Flag OFF (default, unset): the legacy scan path must be completely
untouched. We do not modify the pre-existing legacy tests
(tests/test_server.py::test_worker_fpp_task_respects_metadata_filter /
test_worker_fpp_task_backfills_missing_nfpp) — they keep passing unchanged,
which is itself the flag-off regression proof. This file adds one more
flag-off scenario for locality (case (e) of the task brief), duplicating
(not modifying) that same shape.
"""
from __future__ import annotations

import os
import time

import pytest

from exonit import work_streams

AUTH = {"Authorization": "Bearer exonit-default-token"}


def _seed_candidate(db_path, tic_id, **overrides):
    from exonit.db import ExoNitDB

    db = ExoNitDB(db_path)
    db.add_star(tic_id, 0, 0, 10, 1, 1000, 30)
    kwargs = dict(cnn_score=0.9)
    kwargs.update(overrides)
    candidate_id = db.add_candidate(tic_id, 1, 2.0, 0, 0.1, 500, 50.0, **kwargs)
    db.close()
    return candidate_id


def _seed_metadata(meta_path, *tic_ids):
    """Register FITS metadata for *tic_ids*. The stream path (like the
    legacy scan's `if not _mc.get_entry(...): continue`) only serves
    candidates whose FITS metadata is cached — a metadata-less serve makes
    the worker fail its LC load and POST the permanent -1/-1 sentinel."""
    from exonit.metadata_cache import MetadataCache

    cache = MetadataCache(meta_path)
    for tic in tic_ids:
        cache.add_entry(tic, sectors=[1], exptime=120.0,
                        fits_files=[f"tic{tic}.fits"])
    cache.close()


def _build_env(tmp_path, monkeypatch, visibility_timeout_s=1800):
    """TestClient with EXONIT_WORK_STREAMS=1 and a MemoryWorkStream injected
    in place of exonit.work_streams.get_stream."""
    db_path = str(tmp_path / "test.db")
    meta_path = str(tmp_path / "meta.db")
    reports_dir = str(tmp_path / "reports")
    os.makedirs(reports_dir, exist_ok=True)

    monkeypatch.setenv("EXONIT_DB_PATH", db_path)
    monkeypatch.setenv("EXONIT_METADATA_CACHE", meta_path)
    monkeypatch.setenv("EXONIT_REPORTS_DIR", reports_dir)
    monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE",
                       str(tmp_path / "pipeline_state.json"))
    monkeypatch.setenv("EXONIT_WORK_STREAMS", "1")

    memory_ws = work_streams.MemoryWorkStream(
        "fpp", visibility_timeout_s=visibility_timeout_s)
    monkeypatch.setattr(work_streams, "get_stream", lambda name: memory_ws)

    from exonit.server import create_app
    app = create_app()

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        yield client, db_path, meta_path, memory_ws


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    yield from _build_env(tmp_path, monkeypatch)


@pytest.fixture
def app_env_short_vt(tmp_path, monkeypatch):
    """Same as app_env but with a sub-second stream visibility timeout so a
    test can exercise PEL redelivery without waiting the production 1800s."""
    yield from _build_env(tmp_path, monkeypatch, visibility_timeout_s=0.2)


class TestFppTaskFromStream:
    """EXONIT_WORK_STREAMS=1 — serve from the stream, no PG scans."""

    def test_empty_queue_returns_none_with_zero_db_queries(self, app_env, monkeypatch):
        client, db_path, _meta_path, memory_ws = app_env

        from exonit.db import ExoNitDB
        calls = []
        original = ExoNitDB._query_one

        def spy(self, sql, params=()):
            calls.append(sql)
            return original(self, sql, params)

        monkeypatch.setattr(ExoNitDB, "_query_one", spy)

        resp = client.get("/api/worker/fpp_task",
                          headers={**AUTH, "X-Worker-ID": "w1"})
        assert resp.status_code == 200
        assert resp.json() == {"task": None}
        assert calls == [], (
            "empty queue must cost zero candidate/star DB queries — only "
            f"the Redis/Memory claim() call — got: {calls}")

    def test_eligible_item_payload_matches_legacy_shape(self, app_env):
        client, db_path, meta_path, memory_ws = app_env
        candidate_id = _seed_candidate(db_path, "700")
        _seed_metadata(meta_path, "700")
        memory_ws.enqueue(str(candidate_id),
                          {"candidate_id": candidate_id, "tic_id": "700"})

        resp = client.get("/api/worker/fpp_task",
                          headers={**AUTH, "X-Worker-ID": "w1"})
        assert resp.status_code == 200
        task = resp.json()["task"]
        assert task is not None
        # Same 9-field shape _build_backfill_payload returns on the legacy
        # path (candidate_id/tic_id/candidate_num/period_days/t0/
        # duration_days/depth_ppm/snr/n_sectors) — byte-identical keys.
        assert set(task.keys()) == {
            "candidate_id", "tic_id", "candidate_num", "period_days", "t0",
            "duration_days", "depth_ppm", "snr", "n_sectors",
        }
        assert task["candidate_id"] == candidate_id
        assert task["tic_id"] == "700"
        assert task["n_sectors"] == 1
        assert memory_ws.stats()["pending"] == 1   # claimed, not yet acked

    def test_stale_ineligible_item_autoacks_and_serves_next(self, app_env):
        client, db_path, meta_path, memory_ws = app_env
        # Candidate A already has fpp+nfpp done (neither arm applies any
        # more) — simulates a stream entry that went stale between enqueue
        # and claim (the MANDATORY re-check the brief calls out).
        from exonit.db import ExoNitDB
        cand_a = _seed_candidate(db_path, "701")
        db = ExoNitDB(db_path)
        db.update_candidate_fpp(cand_a, fpp=0.01, nfpp=0.01)
        db.close()
        cand_b = _seed_candidate(db_path, "702")
        _seed_metadata(meta_path, "702")

        memory_ws.enqueue(str(cand_a), {"candidate_id": cand_a, "tic_id": "701"})
        memory_ws.enqueue(str(cand_b), {"candidate_id": cand_b, "tic_id": "702"})

        resp = client.get("/api/worker/fpp_task",
                          headers={**AUTH, "X-Worker-ID": "w1"})
        assert resp.status_code == 200
        task = resp.json()["task"]
        assert task is not None
        assert task["tic_id"] == "702", (
            "stale candidate A must be auto-acked and skipped; B (still "
            "eligible) must be served instead")
        # A was ack'd (removed entirely); B is claimed (pending), nothing
        # left in the raw queue.
        assert memory_ws.stats() == {"queued": 0, "pending": 1, "dlq": 0}

    def test_fpp_result_acks_stream_item(self, app_env):
        client, db_path, meta_path, memory_ws = app_env
        candidate_id = _seed_candidate(db_path, "703")
        _seed_metadata(meta_path, "703")
        memory_ws.enqueue(str(candidate_id),
                          {"candidate_id": candidate_id, "tic_id": "703"})

        resp = client.get("/api/worker/fpp_task",
                          headers={**AUTH, "X-Worker-ID": "w1"})
        assert resp.json()["task"]["candidate_id"] == candidate_id
        assert memory_ws.stats()["pending"] == 1

        resp2 = client.post("/api/worker/fpp_result",
                            headers={**AUTH, "X-Worker-ID": "w1"},
                            json={"candidate_id": candidate_id, "fpp": 0.02,
                                  "nfpp": 0.001})
        assert resp2.status_code == 200
        assert memory_ws.stats() == {"queued": 0, "pending": 0, "dlq": 0}

    def test_fpp_result_ack_failure_still_200_and_triggers_verdict(
            self, app_env, monkeypatch, caplog):
        """I1 final-review fix: a Redis hiccup at ack() time must not 500 the
        worker (the FPP result is already committed to PG) and must not skip
        the verdict-classifier trigger that follows the ack in this handler."""
        client, db_path, meta_path, memory_ws = app_env
        candidate_id = _seed_candidate(db_path, "704")
        _seed_metadata(meta_path, "704")
        memory_ws.enqueue(str(candidate_id),
                          {"candidate_id": candidate_id, "tic_id": "704"})
        resp = client.get("/api/worker/fpp_task",
                          headers={**AUTH, "X-Worker-ID": "w1"})
        assert resp.json()["task"]["candidate_id"] == candidate_id

        # settle() replaced the bare ack() on the result path (2026-07-14,
        # speculative tail execution): it both releases the assignment (acking)
        # and adjudicates a speculative race. Same degradation contract as
        # before — a Redis hiccup here must not 500 the worker, must not lose
        # its result, and must not skip the verdict trigger.
        def _raise_settle(item_key):
            raise RuntimeError("redis down")
        monkeypatch.setattr(memory_ws, "settle", _raise_settle)

        import exonit.routes.workers as workers_mod

        class _SyncExecutor:
            """Records submit() calls without spawning a real background
            thread — deterministic proof the trigger fired, no timing race."""
            def __init__(self):
                self.calls = []

            def submit(self, fn, *args, **kwargs):
                self.calls.append((fn, args, kwargs))

                class _F:
                    def result(self_inner):
                        return None
                return _F()

        sync_executor = _SyncExecutor()
        monkeypatch.setattr(workers_mod, "_get_verdict_executor",
                            lambda: sync_executor)

        with caplog.at_level("WARNING"):
            resp2 = client.post(
                "/api/worker/fpp_result",
                headers={**AUTH, "X-Worker-ID": "w1"},
                json={"candidate_id": candidate_id, "fpp": 0.02, "nfpp": 0.001,
                      "tic_id": "704", "candidate": {"cnn_score": 0.9}})
        assert resp2.status_code == 200, (
            "settle() raising must be caught, not surface as a 500")
        assert resp2.json() == {"status": "ok"}
        assert len(sync_executor.calls) == 1, (
            "verdict trigger must still run even though settle() failed")
        assert sync_executor.calls[0][0] is workers_mod._run_verdict_background
        assert any("fpp stream settle failed" in r.message
                   for r in caplog.records), (
            "settle() failure must be logged as a warning")

        # ...and the result itself must still be PERSISTED: a Redis failure
        # degrades to the pre-speculation behaviour (write unconditionally),
        # never to a silently dropped FPP.
        from exonit.db import ExoNitDB
        db = ExoNitDB(db_path)
        assert db.get_candidates("704")[0]["triceratops_fpp"] == 0.02
        db.close()

    def test_max_iterations_caps_a_run_of_stale_entries(self, app_env):
        client, db_path, _meta_path, memory_ws = app_env
        from exonit.db import ExoNitDB
        db = ExoNitDB(db_path)
        # 6 stale entries (already fpp+nfpp done) — one more than the
        # handler's 5-iteration cap — must not hang or 500; must give up
        # cleanly with {"task": None} rather than loop forever.
        stale_ids = []
        for i in range(6):
            tic = f"71{i}"
            db.add_star(tic, 0, 0, 10, 1, 1000, 30)
            cid = db.add_candidate(tic, 1, 2.0, 0, 0.1, 500, 50.0,
                                   cnn_score=0.9)
            db.update_candidate_fpp(cid, fpp=0.01, nfpp=0.01)
            stale_ids.append(cid)
        db.close()
        for cid, tic in zip(stale_ids, [f"71{i}" for i in range(6)]):
            memory_ws.enqueue(str(cid), {"candidate_id": cid, "tic_id": tic})

        resp = client.get("/api/worker/fpp_task",
                          headers={**AUTH, "X-Worker-ID": "w1"})
        assert resp.status_code == 200
        assert resp.json() == {"task": None}
        # 5 iterations ack 5 stale entries; the 6th is left queued for the
        # next poll.
        assert memory_ws.stats()["queued"] == 1


class TestFppTaskMetadataFilter:
    """Metadata-availability filter on the stream path (review follow-up).

    Serving a candidate whose FITS metadata is not cached makes the worker
    fail its LC load and POST fpp=-1/nfpp=-1, which update_candidate_fpp
    writes PERMANENTLY (the terminal uncomputable sentinel, excluded from
    both eligibility arms forever). The stream path must therefore mirror
    the legacy scan's `if not _mc.get_entry(c["tic_id"]): continue` — and,
    because metadata-missing is TRANSIENT, it must NOT ack: the item stays
    in the PEL and is redelivered after the visibility timeout.
    """

    def test_metadata_less_candidate_not_served_and_not_acked(
            self, app_env, monkeypatch):
        client, db_path, _meta_path, memory_ws = app_env
        candidate_id = _seed_candidate(db_path, "720")
        # No _seed_metadata call — the cache has no entry for TIC 720.
        memory_ws.enqueue(str(candidate_id),
                          {"candidate_id": candidate_id, "tic_id": "720"})

        # Belt-and-braces: also assert via a spy that no payload could have
        # been built even if the response were tampered with.
        import exonit.server as srv
        built = []
        original_build = srv._build_backfill_payload
        monkeypatch.setattr(
            "exonit.routes.workers._build_backfill_payload",
            lambda c: built.append(c) or original_build(c))

        resp = client.get("/api/worker/fpp_task",
                          headers={**AUTH, "X-Worker-ID": "w1"})
        assert resp.status_code == 200
        assert resp.json() == {"task": None}
        assert built == [], "no payload may be built for a metadata-less item"
        # NOT acked: the item must remain pending (in the PEL) so it is
        # redelivered after the visibility timeout — an ack here would
        # silently drop the candidate until a manual stream rebuild.
        assert memory_ws.stats() == {"queued": 0, "pending": 1, "dlq": 0}

    def test_metadata_appearing_later_lets_redelivery_serve_it(
            self, app_env_short_vt):
        client, db_path, meta_path, memory_ws = app_env_short_vt
        candidate_id = _seed_candidate(db_path, "721")
        memory_ws.enqueue(str(candidate_id),
                          {"candidate_id": candidate_id, "tic_id": "721"})

        # First poll: metadata missing → not served, left in the PEL.
        resp = client.get("/api/worker/fpp_task",
                          headers={**AUTH, "X-Worker-ID": "w1"})
        assert resp.json() == {"task": None}
        assert memory_ws.stats()["pending"] == 1

        # Metadata arrives (e.g. the Phase 2.5 cache populated it) and the
        # visibility timeout (0.2s in this fixture) elapses → the stale PEL
        # entry is redelivered and served normally this time.
        _seed_metadata(meta_path, "721")
        time.sleep(0.25)
        resp2 = client.get("/api/worker/fpp_task",
                           headers={**AUTH, "X-Worker-ID": "w2"})
        task = resp2.json()["task"]
        assert task is not None
        assert task["candidate_id"] == candidate_id
        assert task["tic_id"] == "721"
        assert memory_ws.stats()["pending"] == 1   # reclaimed, not yet acked


class TestFppTaskFlagOff:
    """EXONIT_WORK_STREAMS unset (default) — legacy scan path, untouched."""

    def test_flag_off_uses_legacy_scan_path(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test.db")
        meta_path = str(tmp_path / "meta.db")
        reports_dir = str(tmp_path / "reports")
        os.makedirs(reports_dir, exist_ok=True)
        monkeypatch.delenv("EXONIT_WORK_STREAMS", raising=False)
        monkeypatch.setenv("EXONIT_DB_PATH", db_path)
        monkeypatch.setenv("EXONIT_METADATA_CACHE", meta_path)
        monkeypatch.setenv("EXONIT_REPORTS_DIR", reports_dir)

        from exonit.db import ExoNitDB
        from exonit.metadata_cache import MetadataCache
        db = ExoNitDB(db_path)
        db.add_star("800", 0, 0, 10, 1, 1000, 30)
        db.add_candidate("800", 1, 2.0, 0, 0.1, 500, 50.0, cnn_score=0.95)
        db.close()
        cache = MetadataCache(meta_path)
        cache.add_entry("800", sectors=[1], exptime=120.0,
                        fits_files=["tic800.fits"])
        cache.close()

        from exonit.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())
        resp = c.get("/api/worker/fpp_task", headers=AUTH)
        assert resp.status_code == 200
        assert resp.json()["task"]["tic_id"] == "800"
