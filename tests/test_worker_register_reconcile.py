"""Regression tests for the 2026-07-10 register()-500-loop and stale-shutdown
incidents (see progress/impl-register-500-and-stale-shutdown.md).

Bug 1: ``ExoNitDB.register_worker`` used to INSERT ... ON CONFLICT(worker_id),
but ``workers_registry.name`` also has a UNIQUE constraint. A worker_id that
drifted out of sync with a name already present in the DB (e.g. a stale
``workers_state`` Redis entry keyed by NAME instead of worker_id) made every
register() call raise a unique-violation -> HTTP 500 -> the worker retried
forever without ever running a task (worker-<host>-1, 2026-07-10, 14min).

Bug 2: dashboard "shutdown" commands queued in the ``worker:pending_commands``
Redis hash for an OFFLINE worker sat there indefinitely; the worker's first
heartbeat after coming back online (possibly weeks later) consumed the stale
command and killed itself 1s after registering (<host> x6, HP, same incident).

Covers:
1. ``register_worker`` idempotent by name: a second call with a fresh
   worker_id for an existing name does not raise, does not duplicate the row,
   and leaves the original worker_id authoritative.
2. ``/api/worker/register`` self-heals a workers_state entry that is keyed by
   NAME instead of worker_id: returns the DB's worker_id (200, not 500) and
   leaves workers_state clean (no name-keyed entry), preserving counters.
3. ``/api/worker/heartbeat`` discards a pending command older than
   ``_PENDING_COMMAND_MAX_AGE_SECONDS`` (or missing ``ts`` entirely) instead
   of returning it; a recent command is still delivered.
4. ``/api/worker/register`` clears any pending command queued for its
   worker_id before the worker existed in this process's lifetime.
"""
from __future__ import annotations

import os
import time

import pytest

from exonit.db import ExoNitDB
from exonit.shared_state import get_shared_state, reset_shared_state


# --------------------------------------------------------------------- #
# 1. DB-level idempotency
# --------------------------------------------------------------------- #

@pytest.fixture
def db(tmp_path):
    database = ExoNitDB(str(tmp_path / "test.db"))
    yield database
    database.close()


def test_register_worker_idempotent_by_name(db):
    """Same name, new worker_id -> no raise, no duplicate row, original wid wins."""
    db.register_worker("wid-original", "worker-dup-name")
    assert db.get_worker_id_by_name("worker-dup-name") == "wid-original"

    # Simulate a fresh worker_id showing up for the same name (e.g. Redis
    # cache drift, or workers_registry reset while the worker kept its old
    # in-memory identity momentarily).
    db.register_worker("wid-new-imposter", "worker-dup-name")

    # The DB is still authoritative on the ORIGINAL worker_id — no exception
    # was raised, and no second row was created for this name.
    assert db.get_worker_id_by_name("worker-dup-name") == "wid-original"
    rows = db._query(
        "SELECT worker_id FROM workers_registry WHERE name=?",
        ("worker-dup-name",))
    assert len(rows) == 1
    assert rows[0]["worker_id"] == "wid-original"

    # The imposter worker_id was never persisted as its own row.
    rows2 = db._query(
        "SELECT worker_id FROM workers_registry WHERE worker_id=?",
        ("wid-new-imposter",))
    assert rows2 == []


# --------------------------------------------------------------------- #
# 2-4. HTTP-level: /api/worker/register + /api/worker/heartbeat
# --------------------------------------------------------------------- #

@pytest.fixture
def app_client(tmp_path):
    """FastAPI TestClient with an isolated DB, no worker pre-registered."""
    os.environ["EXONIT_DB_PATH"] = str(tmp_path / "test.db")
    os.environ["EXONIT_REPORTS_DIR"] = str(tmp_path / "reports")
    os.makedirs(tmp_path / "reports", exist_ok=True)

    from fastapi.testclient import TestClient
    from exonit.server import create_app
    app = create_app()
    client = TestClient(app)
    auth = {"Authorization": "Bearer exonit-default-token"}

    yield client, auth, app, tmp_path

    del os.environ["EXONIT_DB_PATH"]
    del os.environ["EXONIT_REPORTS_DIR"]


def test_register_self_heals_name_keyed_workers_state(app_client):
    """workers_state has a corrupt entry keyed by NAME (not worker_id); the DB
    has the authoritative hex worker_id. register() must return the DB's
    worker_id (200) and clean up workers_state instead of 500ing forever."""
    client, auth, app, tmp_path = app_client
    name = "worker-<host>-1"

    db = ExoNitDB(str(tmp_path / "test.db"))
    db_wid = "ca7ae5fc3220f649"
    db.register_worker(db_wid, name)
    db.update_worker_stats(db_wid, tasks_completed=42)
    db.close()

    workers_state = app.state.workers_state
    # Corrupt entry: keyed by the NAME string instead of the hex worker_id,
    # as observed in the 2026-07-10 incident.
    workers_state[name] = {
        "name": name,
        "capabilities": {},
        "status": "idle",
        "last_heartbeat": time.time(),
        "current_tic": None,
        "tasks_completed": 7,  # stale in-memory count, DB has 42
    }

    resp = client.post("/api/worker/register",
                        json={"name": name, "capabilities": {"gpu": "none"}},
                        headers=auth)
    assert resp.status_code == 200, resp.text
    assert resp.json()["worker_id"] == db_wid

    # workers_state is clean: no more entry keyed by the bad (name) key.
    assert name not in workers_state
    assert db_wid in workers_state
    entry = workers_state[db_wid]
    assert dict(entry)["name"] == name

    # The stale entry's in-session counter (7) is what gets carried over by
    # the reconciliation itself — it's the worker's actual in-process
    # progress this session, which would otherwise be silently discarded by
    # dropping the bad key. (DB-persisted counters are a separate rehydration
    # path used only when the good worker_id has no prior in-process entry
    # at all.)
    assert dict(entry)["tasks_completed"] == 7


def test_register_returns_200_repeatedly_after_heal(app_client):
    """Once healed, subsequent register() calls for the same name keep
    working (no regression to the 500-loop)."""
    client, auth, app, tmp_path = app_client
    name = "worker-repeat"

    resp1 = client.post("/api/worker/register",
                         json={"name": name, "capabilities": {}},
                         headers=auth)
    assert resp1.status_code == 200
    wid = resp1.json()["worker_id"]

    for _ in range(3):
        resp = client.post("/api/worker/register",
                            json={"name": name, "capabilities": {}},
                            headers=auth)
        assert resp.status_code == 200
        assert resp.json()["worker_id"] == wid


# --------------------------------------------------------------------- #
# 3. Stale pending command discarded by heartbeat
# --------------------------------------------------------------------- #

def _register(client, auth, name="worker-hb-test"):
    resp = client.post("/api/worker/register",
                        json={"name": name, "capabilities": {}},
                        headers=auth)
    assert resp.status_code == 200
    return resp.json()["worker_id"]


def _heartbeat(client, auth, wid):
    return client.post("/api/worker/heartbeat",
                        json={"worker_id": wid, "status": "idle",
                              "current_tic": None},
                        headers=auth)


def test_heartbeat_discards_stale_shutdown_command(app_client):
    client, auth, app, tmp_path = app_client
    wid = _register(client, auth)

    st = get_shared_state()
    # A shutdown queued 10 minutes ago (older than the 300s threshold).
    st.hset_json("worker:pending_commands", str(wid),
                 {"command": "shutdown", "ts": time.time() - 600})

    resp = _heartbeat(client, auth, wid)
    assert resp.status_code == 200
    assert resp.json()["command"] != "shutdown"

    # The stale entry must be consumed (removed), not left to be re-tried.
    assert st.hget_json("worker:pending_commands", str(wid)) is None


def test_heartbeat_discards_command_without_ts(app_client):
    """A pending command with no 'ts' field (queued by pre-fix code, or a
    Redis entry surviving from before this deploy) must be treated as
    stale, not fresh."""
    client, auth, app, tmp_path = app_client
    wid = _register(client, auth)

    st = get_shared_state()
    st.hset_json("worker:pending_commands", str(wid), {"command": "shutdown"})

    resp = _heartbeat(client, auth, wid)
    assert resp.status_code == 200
    assert resp.json()["command"] != "shutdown"


def test_heartbeat_delivers_recent_command(app_client):
    client, auth, app, tmp_path = app_client
    wid = _register(client, auth)

    st = get_shared_state()
    st.hset_json("worker:pending_commands", str(wid),
                 {"command": "pause", "ts": time.time() - 5})

    resp = _heartbeat(client, auth, wid)
    assert resp.status_code == 200
    assert resp.json()["command"] == "pause"


# --------------------------------------------------------------------- #
# 4. register() clears pending commands for its worker_id
# --------------------------------------------------------------------- #

def test_register_clears_pending_command_for_wid(app_client):
    client, auth, app, tmp_path = app_client
    name = "worker-fresh-start"

    # First register to learn the worker_id this name will be assigned.
    wid = _register(client, auth, name=name)

    st = get_shared_state()
    # Simulate a command queued while the worker was offline, before this
    # (re)registration.
    st.hset_json("worker:pending_commands", str(wid),
                 {"command": "shutdown", "ts": time.time()})
    assert st.hget_json("worker:pending_commands", str(wid)) is not None

    # Worker (re)starts and registers again — any legacy pending command for
    # its worker_id must be dropped, not delivered on the next heartbeat.
    resp = client.post("/api/worker/register",
                        json={"name": name, "capabilities": {}},
                        headers=auth)
    assert resp.status_code == 200
    assert resp.json()["worker_id"] == wid
    assert st.hget_json("worker:pending_commands", str(wid)) is None
