"""The Rp-gate wired into the server: eager (at TIC completion) and lazy (at
serve time), plus the invariant that must not break — the FPP queue and the
run's own progress counter must always agree.

If the gate removed candidates from the queue but /api/health/work kept
counting them as actionable, the orchestrator would wait on work nobody will
ever serve — the exact shape of the 2026-07-14 hang. Both are driven by
`triceratops_fpp IS NULL`, and the gate WRITES the -1 sentinel, so they move
together. These tests pin that.
"""
from __future__ import annotations

import os

import pytest

from exonit import stellar_gates, work_streams

AUTH = {"Authorization": "Bearer exonit-default-token"}

# A 3 R☉ subgiant with a 1% transit: implied companion = 3·√0.01·9.95 ≈ 3.0 R_Jup
# ...so 2% is unambiguously stellar, and 0.05% is unambiguously planetary.
GIANT_R = 8.0
DEEP_PPM = 20000.0      # → ~11 R_Jup on that star: an eclipsing binary
SHALLOW_PPM = 200.0     # → ~1.1 R_Jup: a plausible planet


@pytest.fixture
def env(tmp_path, monkeypatch):
    from exonit.db import ExoNitDB
    from exonit.metadata_cache import MetadataCache

    db_path = str(tmp_path / "t.db")
    meta_path = str(tmp_path / "m.db")
    monkeypatch.setenv("EXONIT_DB_PATH", db_path)
    monkeypatch.setenv("EXONIT_METADATA_CACHE", meta_path)
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "rep"))
    monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE", str(tmp_path / "s.json"))
    monkeypatch.setenv("EXONIT_WORK_STREAMS", "1")
    monkeypatch.setenv("EXONIT_FPP_GATE", "1")
    os.makedirs(str(tmp_path / "rep"), exist_ok=True)

    db = ExoNitDB(db_path)
    db.add_star("555", 0, 0, 12, 1, 1000, 30)
    deep = db.add_candidate("555", 1, 3.0, 0, 0.1, DEEP_PPM, 50.0, cnn_score=0.9)
    shallow = db.add_candidate("555", 2, 5.0, 0, 0.1, SHALLOW_PPM, 30.0,
                               cnn_score=0.9)
    db.close()

    cache = MetadataCache(meta_path)
    cache.add_entry("555", sectors=[1], exptime=120.0, fits_files=["a.fits"])
    # The host: a giant with a real mass (so the MASS gate does NOT fire and we
    # are testing the Rp gate in isolation).
    cache.add_nearby_stars("555", [{
        "ID": "555", "ra": 0.0, "dec": 0.0, "mass": 1.1, "rad": GIANT_R,
        "Teff": 4800.0, "Tmag": 12.0, "plx": 1.0,
    }], gal_b=20.0)
    cache.close()

    ws = work_streams.MemoryWorkStream("fpp")
    monkeypatch.setattr(work_streams, "get_stream", lambda name: ws)
    yield {"db_path": db_path, "meta_path": meta_path, "deep": deep,
           "shallow": shallow, "ws": ws}


def _fpp(db_path, cid):
    from exonit.db import ExoNitDB
    db = ExoNitDB(db_path)
    row = db._query_one(
        "SELECT triceratops_fpp, fpp_skip_reason FROM candidates WHERE id = ?",
        (cid,))
    db.close()
    return row


class TestEagerGateAtTicCompletion:

    def test_deep_transit_on_a_giant_is_marked_before_it_ever_reaches_a_worker(
            self, env):
        import exonit.server as server
        from exonit.db import ExoNitDB
        from exonit.metadata_cache import MetadataCache

        db = ExoNitDB(env["db_path"])
        mc = MetadataCache(env["meta_path"])
        server._run_companion_gate_background(db, "555", mc)
        db.close()

        deep = _fpp(env["db_path"], env["deep"])
        assert deep["triceratops_fpp"] == -1
        assert deep["fpp_skip_reason"] == "stellar_companion", (
            "the reason must say it is a binary, not that the system failed")

        shallow = _fpp(env["db_path"], env["shallow"])
        assert shallow["triceratops_fpp"] is None, (
            "a plausible planet on the SAME star must be left alone")

    def test_unknown_radius_gates_nothing(self, env, monkeypatch):
        """Fail-open. A star we know nothing about is not a binary."""
        import exonit.server as server
        from exonit.db import ExoNitDB
        from exonit.metadata_cache import MetadataCache

        monkeypatch.setattr(server, "resolve_star_radius", lambda *a: None)
        db = ExoNitDB(env["db_path"])
        mc = MetadataCache(env["meta_path"])
        server._run_companion_gate_background(db, "555", mc)
        db.close()

        assert _fpp(env["db_path"], env["deep"])["triceratops_fpp"] is None

    def test_never_overwrites_a_real_fpp_result(self, env):
        import exonit.server as server
        from exonit.db import ExoNitDB
        from exonit.metadata_cache import MetadataCache

        db = ExoNitDB(env["db_path"])
        db.update_candidate_fpp(env["deep"], 0.02, nfpp=0.001)
        mc = MetadataCache(env["meta_path"])
        server._run_companion_gate_background(db, "555", mc)
        db.close()

        assert _fpp(env["db_path"], env["deep"])["triceratops_fpp"] == 0.02


class TestLazyGateAtServeTime:
    """The eager gate only covers NEW work. The already-queued backlog (410k
    items on 2026-07-14) is drained here, when a worker asks for the item."""

    def test_queued_binary_is_marked_and_acked_instead_of_served(self, env):
        from fastapi.testclient import TestClient

        from exonit.server import create_app

        ws = env["ws"]
        ws.enqueue(str(env["deep"]), {"candidate_id": env["deep"]})

        with TestClient(create_app()) as client:
            resp = client.get("/api/worker/fpp_task",
                              headers={**AUTH, "X-Worker-ID": "w1"})
            assert resp.json()["task"] is None, (
                "an eclipsing binary must never be handed to a worker")

        assert _fpp(env["db_path"], env["deep"])["fpp_skip_reason"] == \
            "stellar_companion"
        assert ws.stats() == {"queued": 0, "pending": 0, "dlq": 0}, (
            "the item must be ACKED out of the queue, not left pending — "
            "otherwise the run waits forever on work nobody will serve")

    def test_plausible_planet_on_the_same_star_is_still_served(self, env):
        from fastapi.testclient import TestClient

        from exonit.server import create_app

        env["ws"].enqueue(str(env["shallow"]), {"candidate_id": env["shallow"]})
        with TestClient(create_app()) as client:
            resp = client.get("/api/worker/fpp_task",
                              headers={**AUTH, "X-Worker-ID": "w1"})
            task = resp.json()["task"]
        assert task is not None and task["candidate_id"] == env["shallow"]


def test_queue_and_run_counter_stay_in_agreement(env):
    """The invariant that a hang is made of. /api/health/work counts
    `triceratops_fpp IS NULL`; the gate writes -1. So a gated candidate leaves
    the queue AND the counter in the same instant. If a future gate ever skips
    a candidate WITHOUT writing the sentinel, this test fails and the run would
    have hung."""
    from fastapi.testclient import TestClient

    from exonit.server import create_app

    env["ws"].enqueue(str(env["deep"]), {"candidate_id": env["deep"]})
    env["ws"].enqueue(str(env["shallow"]), {"candidate_id": env["shallow"]})

    with TestClient(create_app()) as client:
        before = client.get("/api/health/work?nocache=1").json()["fpp_backfill"]
        assert before["actionable"] == 2

        # Drain: the binary is gated away, the planet is served.
        client.get("/api/worker/fpp_task", headers={**AUTH, "X-Worker-ID": "w1"})

        after = client.get("/api/health/work?nocache=1").json()["fpp_backfill"]

    assert after["actionable"] == 1, (
        "the gated binary must disappear from the run's progress counter too, "
        f"not just from the queue (got {after})")
