"""Tests for /api/worker/exominer_task + /api/worker/exominer_result serving
from the "exominer" work stream behind EXONIT_WORK_STREAMS=1 (Fase 1,
Task 4 — mirror of Task 3's FPP suite).

Flag ON: a MemoryWorkStream is injected by monkeypatching
``exonit.work_streams.get_stream`` — the handler resolves ``get_stream``
via module-attribute access on every call, not a bound import, so the
patch takes effect regardless of when the app was built.

Flag OFF (default, unset): the legacy scan path must be completely
untouched. We do not modify the pre-existing legacy tests
(tests/test_exominer_backfill_spoc_gate.py, tests/test_exominer_scan_pause.py)
— they keep passing unchanged, which is itself the flag-off regression
proof. This file adds one more flag-off scenario for locality, duplicating
(not modifying) that same shape.

Metadata-availability handling (TestExominerNoMetadataRace below): the
stream path treats a missing FITS-metadata entry as TRANSIENT — hold back,
NO ack, NO DB write — exactly like the FPP stream, and deliberately UNLIKE
the legacy scan (which -1-marks at serve time; that marking was a
2026-06-10 queue-drain mechanism, commit 2ccb7888, not an eligibility
fact). Every row on this path has has_spoc=1 = MAST-confirmed SPOC data
EXISTS, so a claim-time -1 would permanently and silently lose a real
ExoMiner score whenever the metadata gap is only indexing lag — and the
producer hook enqueues within SECONDS of the has_spoc flip, removing the
informal grace period the legacy snr-ordered backlog provided. Exhaustion
(max_deliveries) sends the item to the DLQ (visible, re-enqueueable by a
post-indexing rebuild), never to -1. The -1 sentinel remains exclusively
at the SPOC-gate producer paths (has_spoc=0 per-TIC, has_spoc=NULL bulk) —
covered by tests/test_spoc_gate.py and
tests/test_exominer_backfill_spoc_gate.py, unchanged.
"""
from __future__ import annotations

import os
import time

import pytest
from unittest.mock import patch

from exonit import work_streams

AUTH = {"Authorization": "Bearer exonit-default-token"}


def _seed_candidate(db_path, tic_id, has_spoc=1, cnn_score=0.9,
                    exominer_score=None, **overrides):
    from exonit.db import ExoNitDB

    db = ExoNitDB(db_path)
    db.add_star(tic_id, 0, 0, 10, 1, 1000, 30, status="completed")
    if has_spoc is not None:
        # Runs BEFORE add_candidate below: db.set_star_has_spoc's own
        # producer hook (Task 4) queries this TIC's candidates at call
        # time, finds none yet, and is a no-op — so tests keep full manual
        # control over what lands in the (injected) stream.
        db.set_star_has_spoc(tic_id, has_spoc)
    kwargs = dict(cnn_score=cnn_score)
    kwargs.update(overrides)
    candidate_id = db.add_candidate(tic_id, 1, 2.0, 0, 0.1, 500, 50.0, **kwargs)
    if exominer_score is not None:
        db.update_candidate_exominer(candidate_id, score=exominer_score)
    db.close()
    return candidate_id


def _seed_metadata(meta_path, *tic_ids):
    """Register FITS metadata for *tic_ids* so the handler can serve them
    (a metadata-less item is held back, not served — see module docstring)."""
    from exonit.metadata_cache import MetadataCache

    cache = MetadataCache(meta_path)
    for tic in tic_ids:
        cache.add_entry(tic, sectors=[1], exptime=120.0,
                        fits_files=[f"tic{tic}.fits"])
    cache.close()


def _build_env(tmp_path, monkeypatch, visibility_timeout_s=1800):
    """TestClient with EXONIT_WORK_STREAMS=1 + EXONIT_EXOMINER_BACKFILL=1
    and a MemoryWorkStream injected in place of
    exonit.work_streams.get_stream."""
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
    monkeypatch.setenv("EXONIT_EXOMINER_BACKFILL", "1")
    monkeypatch.delenv("EXONIT_EXOMINER_WORKERS", raising=False)
    monkeypatch.delenv("EXONIT_EXOMINER_WORKERS_FILE", raising=False)

    from exonit import server as srv
    srv._exominer_allowlist_cache.update(
        {"mtime": 0.0, "names": None, "src": ""})

    memory_ws = work_streams.MemoryWorkStream(
        "exominer", visibility_timeout_s=visibility_timeout_s)
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


class TestExominerTaskFromStream:
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

        resp = client.get("/api/worker/exominer_task",
                          headers={**AUTH, "X-Worker-ID": "w1"})
        assert resp.status_code == 200
        assert resp.json() == {"task": None}
        assert calls == [], (
            "empty queue must cost zero candidate/star DB queries — only "
            f"the Redis/Memory claim() call — got: {calls}")

    def test_eligible_item_payload_matches_legacy_shape(self, app_env):
        client, db_path, meta_path, memory_ws = app_env
        candidate_id = _seed_candidate(db_path, "900")
        _seed_metadata(meta_path, "900")
        memory_ws.enqueue(str(candidate_id),
                          {"candidate_id": candidate_id, "tic_id": "900"})

        resp = client.get("/api/worker/exominer_task",
                          headers={**AUTH, "X-Worker-ID": "w1"})
        assert resp.status_code == 200
        task = resp.json()["task"]
        assert task is not None
        # Same 9-field shape _build_backfill_payload returns on the legacy
        # path — byte-identical keys.
        assert set(task.keys()) == {
            "candidate_id", "tic_id", "candidate_num", "period_days", "t0",
            "duration_days", "depth_ppm", "snr", "n_sectors",
        }
        assert task["candidate_id"] == candidate_id
        assert task["tic_id"] == "900"
        assert task["n_sectors"] == 1
        assert memory_ws.stats()["pending"] == 1   # claimed, not yet acked

    def test_stale_ineligible_item_autoacks_and_serves_next(self, app_env):
        client, db_path, meta_path, memory_ws = app_env
        # Candidate A already has an exominer_score — no longer eligible —
        # simulating a stream entry gone stale between enqueue and claim
        # (the MANDATORY re-check the brief calls out).
        cand_a = _seed_candidate(db_path, "901", exominer_score=0.5)
        cand_b = _seed_candidate(db_path, "902")
        _seed_metadata(meta_path, "902")

        memory_ws.enqueue(str(cand_a), {"candidate_id": cand_a, "tic_id": "901"})
        memory_ws.enqueue(str(cand_b), {"candidate_id": cand_b, "tic_id": "902"})

        resp = client.get("/api/worker/exominer_task",
                          headers={**AUTH, "X-Worker-ID": "w1"})
        assert resp.status_code == 200
        task = resp.json()["task"]
        assert task is not None
        assert task["tic_id"] == "902", (
            "stale candidate A must be auto-acked and skipped; B (still "
            "eligible) must be served instead")
        assert memory_ws.stats() == {"queued": 0, "pending": 1, "dlq": 0}

    def test_exominer_result_acks_stream_item(self, app_env):
        client, db_path, meta_path, memory_ws = app_env
        candidate_id = _seed_candidate(db_path, "903")
        _seed_metadata(meta_path, "903")
        memory_ws.enqueue(str(candidate_id),
                          {"candidate_id": candidate_id, "tic_id": "903"})

        resp = client.get("/api/worker/exominer_task",
                          headers={**AUTH, "X-Worker-ID": "w1"})
        assert resp.json()["task"]["candidate_id"] == candidate_id
        assert memory_ws.stats()["pending"] == 1

        resp2 = client.post("/api/worker/exominer_result",
                            headers={**AUTH, "X-Worker-ID": "w1"},
                            json={"candidate_id": candidate_id, "score": 0.82})
        assert resp2.status_code == 200
        assert memory_ws.stats() == {"queued": 0, "pending": 0, "dlq": 0}

        from exonit.db import ExoNitDB
        db = ExoNitDB(db_path)
        row = db._query_one(
            "SELECT exominer_score FROM candidates WHERE id=?", (candidate_id,))
        assert row["exominer_score"] == pytest.approx(0.82)
        db.close()

    def test_exominer_result_ack_failure_still_200_and_triggers_verdict(
            self, app_env, monkeypatch, caplog):
        """I1 final-review fix: a Redis hiccup at ack() time must not 500
        the worker — the score (and the verdict re-trigger, which in this
        handler already runs BEFORE the ack) are already committed to PG."""
        client, db_path, meta_path, memory_ws = app_env
        candidate_id = _seed_candidate(db_path, "904")
        _seed_metadata(meta_path, "904")
        memory_ws.enqueue(str(candidate_id),
                          {"candidate_id": candidate_id, "tic_id": "904"})
        resp = client.get("/api/worker/exominer_task",
                          headers={**AUTH, "X-Worker-ID": "w1"})
        assert resp.json()["task"]["candidate_id"] == candidate_id

        def _raise_ack(item_key):
            raise RuntimeError("redis down")
        monkeypatch.setattr(memory_ws, "ack", _raise_ack)

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
                "/api/worker/exominer_result",
                headers={**AUTH, "X-Worker-ID": "w1"},
                json={"candidate_id": candidate_id, "score": 0.82})
        assert resp2.status_code == 200, (
            "ack() raising must be caught, not surface as a 500")
        assert resp2.json() == {"status": "ok"}
        assert len(sync_executor.calls) == 1, (
            "verdict re-trigger must still run even though ack() failed")
        assert sync_executor.calls[0][0] is workers_mod._run_verdict_background
        assert any("exominer stream ack failed" in r.message
                   for r in caplog.records), (
            "ack() failure must be logged as a warning")

        from exonit.db import ExoNitDB
        db = ExoNitDB(db_path)
        row = db._query_one(
            "SELECT exominer_score FROM candidates WHERE id=?", (candidate_id,))
        assert row["exominer_score"] == pytest.approx(0.82)
        db.close()

    def test_max_iterations_caps_a_run_of_stale_entries(self, app_env):
        client, db_path, _meta_path, memory_ws = app_env
        # 6 stale entries (already scored) — one more than the handler's
        # 5-iteration cap — must not hang or 500; must give up cleanly with
        # {"task": None} rather than loop forever.
        stale_ids = []
        for i in range(6):
            tic = f"91{i}"
            cid = _seed_candidate(db_path, tic, exominer_score=0.5)
            stale_ids.append(cid)
        for cid, tic in zip(stale_ids, [f"91{i}" for i in range(6)]):
            memory_ws.enqueue(str(cid), {"candidate_id": cid, "tic_id": tic})

        resp = client.get("/api/worker/exominer_task",
                          headers={**AUTH, "X-Worker-ID": "w1"})
        assert resp.status_code == 200
        assert resp.json() == {"task": None}
        # 5 iterations ack 5 stale entries; the 6th is left queued for the
        # next poll.
        assert memory_ws.stats()["queued"] == 1


class TestExominerNoMetadataRace:
    """The reviewer-named race: the producer hook enqueues within seconds of
    the has_spoc=1 flip; if the TIC's FITS metadata is not yet indexed when
    a worker claims the item, the handler must NEVER write the permanent -1
    sentinel (has_spoc=1 = SPOC data EXISTS; -1 would silently lose a real
    ExoMiner score). Instead: hold back WITHOUT ack (PEL redelivery grace),
    and at max_deliveries exhaustion the stream moves it to the DLQ —
    visible and re-enqueueable by a post-indexing rebuild.
    """

    def test_has_spoc_flip_race_no_minus1_and_item_held(self, app_env):
        """End-to-end race: candidate exists, THEN has_spoc flips to 1 (the
        db.set_star_has_spoc producer hook fires and enqueues for real),
        metadata still absent, a worker claims immediately → no -1 write,
        no serve, item held in the PEL (not acked, not DLQ'd)."""
        client, db_path, _meta_path, memory_ws = app_env
        from exonit.db import ExoNitDB

        db = ExoNitDB(db_path)
        db.add_star("904", 0, 0, 10, 1, 1000, 30, status="completed")
        candidate_id = db.add_candidate("904", 1, 2.0, 0, 0.1, 500, 50.0,
                                        cnn_score=0.9)
        # The has_spoc flip AFTER the candidate exists — this drives the
        # REAL producer hook (db.set_star_has_spoc → maybe_enqueue_exominer
        # → the injected memory stream), not a manual enqueue.
        db.set_star_has_spoc("904", 1)
        db.close()
        assert memory_ws.stats()["queued"] == 1, (
            "producer hook must have enqueued the candidate on the flip")

        # Worker claims seconds after the flip; metadata NOT yet indexed.
        resp = client.get("/api/worker/exominer_task",
                          headers={**AUTH, "X-Worker-ID": "w1"})
        assert resp.status_code == 200
        assert resp.json() == {"task": None}
        # Held in the PEL: NOT acked (would silently drop it), NOT DLQ'd.
        assert memory_ws.stats() == {"queued": 0, "pending": 1, "dlq": 0}

        db = ExoNitDB(db_path)
        row = db._query_one(
            "SELECT exominer_score FROM candidates WHERE id=?", (candidate_id,))
        assert row["exominer_score"] is None, (
            "claim-time missing metadata must NEVER write the permanent -1 "
            "sentinel for a has_spoc=1 candidate — SPOC data exists; the "
            "gap is indexing lag")
        db.close()

    def test_metadata_arriving_before_exhaustion_lets_redelivery_serve_it(
            self, app_env_short_vt):
        """Metadata indexed between deliveries → the PEL redelivery serves
        the candidate normally (score preserved, nothing lost)."""
        client, db_path, meta_path, memory_ws = app_env_short_vt
        candidate_id = _seed_candidate(db_path, "906")
        memory_ws.enqueue(str(candidate_id),
                          {"candidate_id": candidate_id, "tic_id": "906"})

        # First poll: metadata missing → held, no write.
        resp = client.get("/api/worker/exominer_task",
                          headers={**AUTH, "X-Worker-ID": "w1"})
        assert resp.json() == {"task": None}
        assert memory_ws.stats()["pending"] == 1

        # Metadata arrives (indexing job ran) and the visibility timeout
        # (0.2s in this fixture) elapses → redelivered and served.
        _seed_metadata(meta_path, "906")
        time.sleep(0.25)
        resp2 = client.get("/api/worker/exominer_task",
                           headers={**AUTH, "X-Worker-ID": "w2"})
        task = resp2.json()["task"]
        assert task is not None
        assert task["candidate_id"] == candidate_id
        assert task["tic_id"] == "906"
        assert memory_ws.stats()["pending"] == 1   # reclaimed, not yet acked

        from exonit.db import ExoNitDB
        db = ExoNitDB(db_path)
        row = db._query_one(
            "SELECT exominer_score FROM candidates WHERE id=?", (candidate_id,))
        assert row["exominer_score"] is None   # still awaiting the worker
        db.close()

    def test_exhaustion_goes_to_dlq_never_minus1(self, app_env_short_vt):
        """Metadata never arrives → after max_deliveries (3) delivery
        cycles the stream moves the item to the DLQ. exominer_score stays
        NULL throughout — the candidate remains re-enqueueable by a
        post-indexing rebuild, instead of being silently -1-tombstoned."""
        client, db_path, _meta_path, memory_ws = app_env_short_vt
        assert memory_ws.max_deliveries == 3
        candidate_id = _seed_candidate(db_path, "907")
        memory_ws.enqueue(str(candidate_id),
                          {"candidate_id": candidate_id, "tic_id": "907"})

        # Deliveries 1..3: claimed, held (no metadata), redelivered after
        # the 0.2s visibility timeout. The 4th claim attempt sees
        # deliveries == max_deliveries and moves the item to the DLQ.
        for _ in range(3):
            resp = client.get("/api/worker/exominer_task",
                              headers={**AUTH, "X-Worker-ID": "w1"})
            assert resp.json() == {"task": None}
            assert memory_ws.stats()["pending"] == 1
            time.sleep(0.25)
        resp = client.get("/api/worker/exominer_task",
                          headers={**AUTH, "X-Worker-ID": "w1"})
        assert resp.json() == {"task": None}
        assert memory_ws.stats() == {"queued": 0, "pending": 0, "dlq": 1}, (
            "exhausted metadata-less item must land in the DLQ (visible, "
            "rebuildable), not vanish")

        from exonit.db import ExoNitDB
        db = ExoNitDB(db_path)
        row = db._query_one(
            "SELECT exominer_score FROM candidates WHERE id=?", (candidate_id,))
        assert row["exominer_score"] is None, (
            "exhaustion must terminate in the DLQ, NEVER in a -1 write — "
            "the DB stays rebuild-eligible")
        db.close()

    def test_rebuild_does_not_mark_metadata_less_rows(self, app_env):
        """The rebuild derivation enqueues every WHERE-eligible row and
        never -1-marks metadata-less has_spoc=1 rows (F1-faithful: the -1
        sentinel belongs only to the SPOC-gate producer paths)."""
        _client, db_path, _meta_path, _memory_ws = app_env
        candidate_id = _seed_candidate(db_path, "908")
        # No metadata for TIC 908.

        import importlib.util
        import pathlib
        script = (pathlib.Path(__file__).resolve().parents[1]
                  / "scripts" / "rebuild_work_streams.py")
        spec = importlib.util.spec_from_file_location("rebuild_ws_t4", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        from exonit.db import ExoNitDB
        db = ExoNitDB(db_path)
        rows = mod._derive_exominer(db)
        assert (str(candidate_id),
                {"candidate_id": candidate_id, "tic_id": "908"}) in rows, (
            "metadata-less has_spoc=1 row must still be enqueued by the "
            "rebuild (the handler holds it back; DLQ is the terminal state)")
        row = db._query_one(
            "SELECT exominer_score FROM candidates WHERE id=?", (candidate_id,))
        assert row["exominer_score"] is None, (
            "rebuild must not blind-mark -1")
        db.close()


class TestExominerTaskGatesBeforeStream:
    """Flag / scan-pause / allowlist gates are evaluated BEFORE the stream
    is touched at all — same order and semantics as the legacy path."""

    def test_backfill_flag_off_returns_none_stream_untouched(
            self, app_env, monkeypatch):
        client, db_path, meta_path, memory_ws = app_env
        monkeypatch.delenv("EXONIT_EXOMINER_BACKFILL", raising=False)
        candidate_id = _seed_candidate(db_path, "910")
        _seed_metadata(meta_path, "910")
        memory_ws.enqueue(str(candidate_id),
                          {"candidate_id": candidate_id, "tic_id": "910"})

        resp = client.get("/api/worker/exominer_task",
                          headers={**AUTH, "X-Worker-ID": "w1"})
        assert resp.status_code == 200
        assert resp.json() == {"task": None}
        assert memory_ws.stats() == {"queued": 1, "pending": 0, "dlq": 0}

    def test_scan_in_flight_returns_none_stream_untouched(
            self, app_env):
        client, db_path, meta_path, memory_ws = app_env
        candidate_id = _seed_candidate(db_path, "911")
        _seed_metadata(meta_path, "911")
        memory_ws.enqueue(str(candidate_id),
                          {"candidate_id": candidate_id, "tic_id": "911"})

        with patch("exonit.server._scan_in_flight", return_value=True):
            resp = client.get("/api/worker/exominer_task",
                              headers={**AUTH, "X-Worker-ID": "w1"})
        assert resp.status_code == 200
        assert resp.json() == {"task": None}
        assert memory_ws.stats() == {"queued": 1, "pending": 0, "dlq": 0}

    def test_allowlist_denies_worker_stream_untouched(
            self, app_env, monkeypatch):
        client, db_path, meta_path, memory_ws = app_env
        monkeypatch.setenv("EXONIT_EXOMINER_WORKERS", "some-other-worker")
        from exonit import server as srv
        srv._exominer_allowlist_cache.update(
            {"mtime": 0.0, "names": None, "src": ""})
        candidate_id = _seed_candidate(db_path, "912")
        _seed_metadata(meta_path, "912")
        memory_ws.enqueue(str(candidate_id),
                          {"candidate_id": candidate_id, "tic_id": "912"})

        resp = client.get("/api/worker/exominer_task",
                          headers={**AUTH, "X-Worker-ID": "w1"})
        assert resp.status_code == 200
        assert resp.json() == {"task": None}
        assert memory_ws.stats() == {"queued": 1, "pending": 0, "dlq": 0}


class TestExominerTaskFlagOff:
    """EXONIT_WORK_STREAMS unset (default) — legacy scan path, untouched."""

    def test_flag_off_uses_legacy_scan_path(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test.db")
        meta_path = str(tmp_path / "meta.db")
        reports_dir = str(tmp_path / "reports")
        os.makedirs(reports_dir, exist_ok=True)
        monkeypatch.delenv("EXONIT_WORK_STREAMS", raising=False)
        monkeypatch.setenv("EXONIT_EXOMINER_BACKFILL", "1")
        monkeypatch.setenv("EXONIT_DB_PATH", db_path)
        monkeypatch.setenv("EXONIT_METADATA_CACHE", meta_path)
        monkeypatch.setenv("EXONIT_REPORTS_DIR", reports_dir)

        from exonit.db import ExoNitDB
        from exonit.metadata_cache import MetadataCache
        db = ExoNitDB(db_path)
        db.add_star("920", 0, 0, 10, 1, 1000, 30, status="completed")
        db.set_star_has_spoc("920", 1)
        db.add_candidate("920", 1, 2.0, 0, 0.1, 500, 50.0, cnn_score=0.95)
        db.close()
        cache = MetadataCache(meta_path)
        cache.add_entry("920", sectors=[1], exptime=120.0,
                        fits_files=["tic920.fits"])
        cache.close()

        from exonit.server import create_app
        from fastapi.testclient import TestClient
        with patch("exonit.server._scan_in_flight", return_value=False):
            c = TestClient(create_app())
            resp = c.get("/api/worker/exominer_task", headers=AUTH)
        assert resp.status_code == 200
        assert resp.json()["task"]["tic_id"] == "920"
