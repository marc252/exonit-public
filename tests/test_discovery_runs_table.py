"""discovery_runs run-record accessors (Fase 3, Task 1).

Follows the SQLite-tmpdir Database fixture pattern used across tests/
(see tests/test_db.py)."""

import pytest
from exonit.db import ExoNitDB


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    database = ExoNitDB(db_path)
    yield database
    database.close()


def _mk(db, run_id="discovery_s0014_lot20260704T220000_2026-07-04"):
    db.create_discovery_run(run_id, sector=14,
                            lot_csv="data/discovery/s0014_lot_x.csv",
                            dry_run=False, chained=False,
                            server_url="http://localhost:8400")
    return run_id


def test_create_and_get_roundtrip(db):
    rid = _mk(db)
    row = db.get_discovery_run(rid)
    assert row["state"] == "ENQUEUEING"
    assert row["sector"] == 14
    assert row["checkpoints"] == {}
    assert row["started_at"]            # tz-aware iso, non-empty


def test_get_active_run_excludes_terminal(db):
    rid = _mk(db)
    assert db.get_active_discovery_run()["run_id"] == rid
    db.update_discovery_run(rid, state="DONE", finished=True)
    assert db.get_active_discovery_run() is None
    row = db.get_discovery_run(rid)
    assert row["finished_at"] is not None


def test_update_state_detail_pid_heartbeat(db):
    rid = _mk(db)
    db.update_discovery_run(rid, state="SCANNING", pid=4242)
    row = db.get_discovery_run(rid)
    assert row["state"] == "SCANNING" and row["orchestrator_pid"] == 4242
    old_hb = row["heartbeat_at"]
    db.update_discovery_run(rid, heartbeat=True)
    assert db.get_discovery_run(rid)["heartbeat_at"] != old_hb
    db.update_discovery_run(rid, state="FAILED", detail="boom")
    assert db.get_discovery_run(rid)["detail"] == "boom"


def test_checkpoints_merge_per_step(db):
    rid = _mk(db)
    db.set_discovery_run_checkpoint(rid, "postprocess",
                                    {"done": True, "scorecard_json": "a.json"})
    db.set_discovery_run_checkpoint(rid, "lora", {"done": True})
    cps = db.get_discovery_run(rid)["checkpoints"]
    assert cps["postprocess"]["scorecard_json"] == "a.json"
    assert cps["lora"] == {"done": True}


def test_unknown_run_returns_none(db):
    assert db.get_discovery_run("nope") is None
