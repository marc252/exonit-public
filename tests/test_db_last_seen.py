"""Regression test: workers_registry.last_seen must refresh on heartbeat.

Bug observed 2026-04-20 on <host>: three workers were heartbeating
every 30s (HP=18s ago, <host>=21s ago, default=9s ago per /api/worker/list)
yet ``SELECT name, last_seen FROM workers_registry`` still showed
timestamps from **hours earlier** — the value from the last *registration*,
not the last *heartbeat*. Result: dashboard / SQL-based "when was worker
X alive" queries were actively misleading.

Fix: ``ExoNitDB.insert_telemetry`` now updates ``last_seen=datetime('now')``
for the worker in the same transaction as the telemetry insert.
"""
from __future__ import annotations

import time

import pytest

from exonit.db import ExoNitDB


@pytest.fixture
def db(tmp_path):
    database = ExoNitDB(str(tmp_path / "test.db"))
    yield database
    database.close()


def _fetch_last_seen(db, worker_id):
    row = db._query_one(
        "SELECT last_seen FROM workers_registry WHERE worker_id=?",
        (worker_id,),
    )
    return row["last_seen"] if row else None


def test_insert_telemetry_refreshes_last_seen(db):
    worker_id = "w-fixture-1"
    db.register_worker(worker_id, "worker-test")
    first = _fetch_last_seen(db, worker_id)
    assert first is not None

    # Backdate the registry row so the refresh is observable even with
    # sqlite's datetime('now') second resolution.
    db._exec(
        "UPDATE workers_registry SET last_seen='2020-01-01 00:00:00' "
        "WHERE worker_id=?",
        (worker_id,),
    )
    backdated = _fetch_last_seen(db, worker_id)
    assert backdated == "2020-01-01 00:00:00"

    db.insert_telemetry(worker_id, {"status": "idle", "current_tic": None})
    refreshed = _fetch_last_seen(db, worker_id)

    assert refreshed != "2020-01-01 00:00:00", (
        "insert_telemetry should refresh workers_registry.last_seen")
    assert refreshed > "2020-01-01 00:00:00"


def test_telemetry_for_unknown_worker_does_not_crash(db):
    """Inserting telemetry for an un-registered worker_id must not raise;
    the row in worker_telemetry is still useful even if nothing in
    workers_registry matches the UPDATE."""
    # No register_worker call on purpose.
    db.insert_telemetry("w-ghost", {"status": "idle"})
    # If we got here without raising, the UPDATE with zero matching rows
    # behaves as a no-op (expected sqlite semantics) and the INSERT landed.
    latest = db.get_latest_telemetry("w-ghost")
    assert latest is not None
    assert latest["worker_id"] == "w-ghost"


def test_multiple_heartbeats_keep_updating(db):
    worker_id = "w-hb"
    db.register_worker(worker_id, "worker-hb")

    db._exec(
        "UPDATE workers_registry SET last_seen='2020-01-01 00:00:00' "
        "WHERE worker_id=?",
        (worker_id,),
    )
    db.insert_telemetry(worker_id, {"status": "working"})
    t1 = _fetch_last_seen(db, worker_id)
    assert t1 > "2020-01-01 00:00:00"

    # A second heartbeat after a brief sleep must land on or after t1.
    # (Can't assert strict monotonicity inside the same second.)
    time.sleep(1.1)
    db.insert_telemetry(worker_id, {"status": "working"})
    t2 = _fetch_last_seen(db, worker_id)
    assert t2 >= t1
    assert t2 != "2020-01-01 00:00:00"
