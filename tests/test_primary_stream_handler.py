"""Tests for /api/worker/task + /api/worker/result + /api/worker/enqueue(_bulk)
serving the PRIMARY (BLS) task queue from the "primary" work stream behind
EXONIT_WORK_STREAMS=1 (Fase 1, Task 5 — mirror of Task 3/4's FPP/ExoMiner
suites, plus the DLQ-reconciliation contract that is new here).

Flag ON: a MemoryWorkStream is injected by monkeypatching
``exonit.work_streams.get_stream`` — the handler resolves ``get_stream`` via
module-attribute access on every call, not a bound import, so the patch
takes effect regardless of when the app was built.

Flag OFF (default, unset): the legacy scan path (db.get_and_assign_task +
db.reassign_stale_tasks) must be completely untouched. We do not modify the
pre-existing legacy tests in tests/test_server.py — they keep passing
unchanged, which is itself the flag-off regression proof. This file adds
one more flag-off scenario for locality, duplicating (not modifying) that
same shape.

Unlike FPP/ExoMiner, the primary task's only eligibility fact is its own
``status`` column (no joined candidates/stars filter) — but its orphan
mechanism is the most complex of the three to retire correctly, because a
primary task has a real PG state machine (pending -> assigned ->
completed/failed) that the FPP/ExoMiner candidate streams don't. The DLQ
tests below (TestPrimaryDlqReconciliation) exercise the invariant that
replaces ``reassign_stale_tasks``: a message that exhausts max_deliveries
must not leave its PG row stuck 'assigned' forever.
"""
from __future__ import annotations

import json
import os
import time

import pytest

from exonit import work_streams
from exonit.db import MAX_TASK_ATTEMPTS, ExoNitDB

AUTH = {"Authorization": "Bearer exonit-default-token"}


def _register(client, name):
    resp = client.post("/api/worker/register",
                       json={"name": name, "capabilities": {}},
                       headers=AUTH)
    assert resp.status_code == 200
    return resp.json()["worker_id"]


def _heartbeat(client, wid, tic_id, status="working"):
    resp = client.post("/api/worker/heartbeat",
                       json={"worker_id": wid, "status": status,
                             "current_tic": tic_id},
                       headers=AUTH)
    assert resp.status_code == 200


def _enqueue_bulk(client, tic_ids):
    targets = [{"tic_id": t} for t in tic_ids]
    resp = client.post("/api/worker/enqueue_bulk",
                       json={"targets": targets}, headers=AUTH)
    assert resp.status_code == 200
    return resp.json()


def _get_task(client, wid):
    return client.get("/api/worker/task",
                      headers={**AUTH, "X-Worker-ID": wid}).json()


def _post_result(client, wid, task_id, tic_id, status="completed", **extra):
    body = {"task_id": task_id, "tic_id": tic_id, "status": status,
            "candidates": [], "cnn_scores": []}
    body.update(extra)
    resp = client.post("/api/worker/result", json=body,
                       headers={**AUTH, "X-Worker-ID": wid})
    assert resp.status_code == 200
    return resp


def _build_env(tmp_path, monkeypatch, visibility_timeout_s=1800,
               sweep_interval_s=None):
    """TestClient with EXONIT_WORK_STREAMS=1 and a MemoryWorkStream injected
    in place of exonit.work_streams.get_stream.

    ``sweep_interval_s`` (G1, review package 2026-07-05): when given, sets
    EXONIT_PRIMARY_SWEEP_INTERVAL_SECONDS so a test can exercise the
    defense-in-depth sweep's throttle-then-resweep cycle without waiting
    the production 300s default. None (default) leaves it unset — the
    production default applies, which is what every test that only cares
    about the sweep firing ONCE (the throttle gate starts absent in every
    test via conftest's per-test shared-state reset) needs.
    """
    db_path = str(tmp_path / "test.db")
    reports_dir = str(tmp_path / "reports")
    os.makedirs(reports_dir, exist_ok=True)

    monkeypatch.setenv("EXONIT_DB_PATH", db_path)
    monkeypatch.setenv("EXONIT_REPORTS_DIR", reports_dir)
    monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE",
                       str(tmp_path / "pipeline_state.json"))
    monkeypatch.setenv("EXONIT_WORK_STREAMS", "1")
    if sweep_interval_s is not None:
        monkeypatch.setenv("EXONIT_PRIMARY_SWEEP_INTERVAL_SECONDS",
                           str(sweep_interval_s))

    memory_ws = work_streams.MemoryWorkStream(
        "primary", visibility_timeout_s=visibility_timeout_s)
    monkeypatch.setattr(work_streams, "get_stream", lambda name: memory_ws)

    from exonit.server import create_app
    app = create_app()

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        yield client, db_path, memory_ws


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    yield from _build_env(tmp_path, monkeypatch)


@pytest.fixture
def app_env_short_vt(tmp_path, monkeypatch):
    """Same as app_env but with a sub-second stream visibility timeout so a
    test can exercise PEL redelivery/steal without waiting the production
    1800s."""
    yield from _build_env(tmp_path, monkeypatch, visibility_timeout_s=0.2)


@pytest.fixture
def app_env_short_sweep(tmp_path, monkeypatch):
    """Same as app_env but with a sub-second G1 sweep-throttle interval so
    a test can exercise the throttle-then-resweep cycle without waiting
    the production 300s default."""
    yield from _build_env(tmp_path, monkeypatch, sweep_interval_s=0.2)


class TestPrimaryTaskFromStream:
    """EXONIT_WORK_STREAMS=1 — serve from the "primary" stream, no PG scans."""

    def test_empty_poll_runs_g1_sweep_once_no_legacy_cascade_zero_extra_queries(
            self, app_env, monkeypatch):
        """G1 (review package, 2026-07-05) superseded the original claim
        that the flag-on empty path NEVER calls reassign_stale_tasks: it
        now does, but only as the throttled defense-in-depth sweep (at
        most once per EXONIT_PRIMARY_SWEEP_INTERVAL_SECONDS — see
        TestPrimarySweepDefenseInDepth for the dedicated throttle tests),
        never as the legacy multi-step cascade (build alive-map -> call
        get_and_assign_task again if reassigned>0) that only the
        flag-OFF path (worker_get_task) still runs. reassign_stale_tasks
        itself queries via backend.tx_query, not ExoNitDB._query_one, so
        the zero-extra-queries guarantee for _query_one-routed calls
        still holds even though the sweep now runs.
        """
        client, db_path, memory_ws = app_env
        wid = _register(client, "idle-w")   # BEFORE the spies: registration
                                            # itself does a workers_registry
                                            # lookup unrelated to /task.

        calls = []
        original = ExoNitDB.reassign_stale_tasks

        def spy(self, *a, **kw):
            calls.append((a, kw))
            return original(self, *a, **kw)

        monkeypatch.setattr(ExoNitDB, "reassign_stale_tasks", spy)

        query_calls = []
        original_query_one = ExoNitDB._query_one

        def query_spy(self, sql, params=()):
            query_calls.append(sql)
            return original_query_one(self, sql, params)

        monkeypatch.setattr(ExoNitDB, "_query_one", query_spy)

        resp = client.get("/api/worker/task",
                          headers={**AUTH, "X-Worker-ID": wid})
        assert resp.status_code == 200
        assert resp.json() == {"task_id": None, "wait_seconds": 10}
        assert len(calls) == 1, (
            "G1: the first empty poll (throttle gate absent) must run the "
            f"low-frequency sweep exactly once — got: {calls}")
        assert calls[0][1].get("collect_ids") is True, (
            "the sweep must request task_ids (not just a count) so it can "
            "re-enqueue every row it flips to pending")
        assert query_calls == [], (
            "an empty queue (no DLQ entries either) must still cost zero "
            f"ExoNitDB._query_one-routed queries — got: {query_calls}")

    def test_roundtrip_enqueue_bulk_task_result_pg_mirror(self, app_env):
        client, db_path, memory_ws = app_env
        result = _enqueue_bulk(client, ["500001"])
        assert result["enqueued"] == 1

        db = ExoNitDB(db_path)
        row = db.get_task_stats()
        assert row.get("pending") == 1
        db.close()

        wid = _register(client, "roundtrip-w")
        resp = _get_task(client, wid)
        assert set(resp.keys()) == {"task_id", "tic_id"}, (
            "legacy flat shape (NOT the FPP/ExoMiner {'task': ...} wrapper)")
        assert resp["tic_id"] == "500001"
        task_id = resp["task_id"]
        assert task_id

        db = ExoNitDB(db_path)
        row = db.get_task(task_id)
        assert row["status"] == "assigned"
        assert row["assigned_to"] == wid
        assert row["attempts"] == 1
        db.close()
        assert memory_ws.stats() == {"queued": 0, "pending": 1, "dlq": 0}

        _post_result(client, wid, task_id, "500001", status="completed")

        db = ExoNitDB(db_path)
        row = db.get_task(task_id)
        assert row["status"] == "completed"
        db.close()
        assert memory_ws.stats() == {"queued": 0, "pending": 0, "dlq": 0}

    def test_dead_worker_steal_reassigns_and_bumps_attempts(
            self, app_env_short_vt):
        client, db_path, memory_ws = app_env_short_vt
        _enqueue_bulk(client, ["500101"])

        wid_a = _register(client, "dead-a")
        _heartbeat(client, wid_a, "500101")
        resp = _get_task(client, wid_a)
        assert resp["tic_id"] == "500101"
        task_id = resp["task_id"]

        # Worker A "dies": its heartbeat falls outside WORKER_ALIVE_WINDOW.
        client.app.state.workers_state[wid_a]["last_heartbeat"] = (
            time.time() - 700)
        time.sleep(0.25)   # let the stream's visibility timeout elapse

        wid_b = _register(client, "dead-b")
        resp2 = _get_task(client, wid_b)
        assert resp2["task_id"] == task_id
        assert resp2["tic_id"] == "500101"

        db = ExoNitDB(db_path)
        row = db.get_task(task_id)
        assert row["assigned_to"] == wid_b
        assert row["attempts"] == 2
        db.close()

    def test_keep_if_protects_alive_worker_on_same_tic(
            self, app_env_short_vt):
        client, db_path, memory_ws = app_env_short_vt
        _enqueue_bulk(client, ["500201"])

        wid_a = _register(client, "alive-a")
        _heartbeat(client, wid_a, "500201")
        resp = _get_task(client, wid_a)
        assert resp["tic_id"] == "500201"
        task_id = resp["task_id"]

        # A is still alive and on the same TIC — refresh right before B polls.
        _heartbeat(client, wid_a, "500201")
        time.sleep(0.25)   # visibility timeout elapses, but keep_if vetoes

        wid_b = _register(client, "alive-b")
        resp2 = _get_task(client, wid_b)
        assert resp2 == {"task_id": None, "wait_seconds": 10}

        db = ExoNitDB(db_path)
        row = db.get_task(task_id)
        assert row["assigned_to"] == wid_a
        db.close()

    def test_abandon_ceiling_steals_from_alive_stuck_worker(
            self, app_env_short_vt):
        """Review fix: branch-1 parity (ABSOLUTE_ABANDON_TIMEOUT_SECONDS=
        3600s) must override branch-3 liveness/current_tic protection.

        Simulates the real stuck-worker failure mode: A's heartbeat THREAD
        stays alive and keeps reporting the same current_tic (so plain
        branch-3 keep_if would protect it forever) while its BLS/numba
        compute thread is actually hung. Only the assigned_at hard ceiling
        can recover this — the negative control (assigned_at recent,
        branch-3 protection holds) is
        test_keep_if_protects_alive_worker_on_same_tic above.
        """
        client, db_path, memory_ws = app_env_short_vt
        _enqueue_bulk(client, ["500701"])

        wid_a = _register(client, "stuck-a")
        _heartbeat(client, wid_a, "500701")
        resp = _get_task(client, wid_a)
        assert resp["tic_id"] == "500701"
        task_id = resp["task_id"]

        # A's heartbeat stays fresh and on-tic (do NOT backdate it) — the
        # whole point is that branch-3 alone would protect A forever.
        _heartbeat(client, wid_a, "500701")

        # Backdate the PG row itself: assigned long enough ago to trip the
        # 3600s abandon ceiling.
        db = ExoNitDB(db_path)
        db._exec("UPDATE tasks SET assigned_at=? WHERE task_id=?",
                 ("2020-01-01T00:00:00+00:00", task_id))
        db.close()

        time.sleep(0.25)   # let the stream's visibility timeout elapse

        wid_b = _register(client, "stuck-b")
        resp2 = _get_task(client, wid_b)
        assert resp2["task_id"] == task_id, (
            "the abandon ceiling must override branch-3 liveness/tic "
            "protection — the task must be stolen from the stuck worker")
        assert resp2["tic_id"] == "500701"

        db = ExoNitDB(db_path)
        row = db.get_task(task_id)
        assert row["assigned_to"] == wid_b
        db.close()

    def test_requeue_on_failure_resets_delivery_count(self, app_env):
        client, db_path, memory_ws = app_env
        _enqueue_bulk(client, ["500301"])
        wid = _register(client, "fail-once-w")
        resp = _get_task(client, wid)
        task_id = resp["task_id"]

        _post_result(client, wid, task_id, "500301", status="failed")

        db = ExoNitDB(db_path)
        row = db.get_task(task_id)
        assert row["status"] == "pending"
        assert row["attempts"] == 1, (
            "requeue_or_fail_task itself must not bump attempts — only "
            "get_task_row_and_assign/get_and_assign_task do")
        db.close()
        assert memory_ws.stats() == {"queued": 1, "pending": 0, "dlq": 0}

        item = memory_ws.claim("probe-worker")
        assert item is not None
        assert item.item_key == task_id
        assert item.deliveries == 1, (
            "the re-enqueued message must be a FRESH one — the old "
            "delivery count must not carry over")

    def test_permanent_fail_acks_without_reenqueue(self, app_env):
        client, db_path, memory_ws = app_env
        _enqueue_bulk(client, ["500401"])
        wid = _register(client, "fail-perm-w")
        resp = _get_task(client, wid)
        task_id = resp["task_id"]

        db = ExoNitDB(db_path)
        db._exec("UPDATE tasks SET attempts=? WHERE task_id=?",
                 (MAX_TASK_ATTEMPTS, task_id))
        db.close()

        _post_result(client, wid, task_id, "500401", status="failed")

        db = ExoNitDB(db_path)
        row = db.get_task(task_id)
        assert row["status"] == "failed"
        db.close()
        assert memory_ws.stats() == {"queued": 0, "pending": 0, "dlq": 0}

    def test_enqueue_bulk_preserves_service_order(self, app_env):
        client, db_path, memory_ws = app_env
        _enqueue_bulk(client, ["600001", "600002", "600003"])
        wid = _register(client, "order-w")
        served = [_get_task(client, wid)["tic_id"] for _ in range(3)]
        assert served == ["600001", "600002", "600003"]

    def test_singular_enqueue_wires_producer_hook(self, app_env):
        client, db_path, memory_ws = app_env
        resp = client.post("/api/worker/enqueue",
                           json={"tic_id": "700001"}, headers=AUTH)
        assert resp.status_code == 200
        assert memory_ws.stats()["queued"] == 1


class TestPrimaryDlqReconciliation:
    """The DLQ-reconciliation contract that replaces reassign_stale_tasks
    for messages the stream itself gave up on (max_deliveries exhausted).
    """

    def test_dlq_exhaustion_pending_row_is_reconciled_not_stuck(
            self, app_env_short_vt):
        client, db_path, memory_ws = app_env_short_vt
        _enqueue_bulk(client, ["500501"])
        wid = _register(client, "dlq-w")
        resp = _get_task(client, wid)
        task_id = resp["task_id"]

        # Drive the injected stream directly to DLQ exhaustion — this tests
        # the reconciliation CONTRACT, not the generic steal mechanic
        # (already covered generically by tests/test_work_streams.py).
        for _ in range(memory_ws.max_deliveries):
            time.sleep(0.25)
            memory_ws.claim("ghost", keep_if=lambda k, c: False)
        assert memory_ws.stats()["dlq"] == 1

        db = ExoNitDB(db_path)
        row = db.get_task(task_id)
        assert row["status"] == "assigned", (
            "sanity: stuck 'assigned' before reconciliation runs")
        db.close()

        # An empty poll triggers _reconcile_primary_dlq.
        empty_wid = _register(client, "poller-w")
        resp2 = _get_task(client, empty_wid)
        assert resp2 == {"task_id": None, "wait_seconds": 10}

        db = ExoNitDB(db_path)
        row = db.get_task(task_id)
        assert row["status"] == "pending", (
            "PG attempts (1) is well under MAX_TASK_ATTEMPTS — the row "
            "must be reconciled back to pending, not left stuck")
        db.close()
        assert memory_ws.stats()["queued"] == 1, (
            "a fresh stream message must be re-enqueued so the "
            "stream-only consumer can see this task again")
        assert memory_ws.stats()["dlq"] == 0 and memory_ws.peek_dlq() == [], (
            "G4: a successfully-reconciled entry must be pruned from the "
            "DLQ, not linger there forever")

    def test_dlq_exhaustion_exhausted_attempts_fails_without_reenqueue(
            self, app_env_short_vt):
        client, db_path, memory_ws = app_env_short_vt
        _enqueue_bulk(client, ["500601"])
        wid = _register(client, "dlq-fail-w")
        resp = _get_task(client, wid)
        task_id = resp["task_id"]

        db = ExoNitDB(db_path)
        db._exec("UPDATE tasks SET attempts=? WHERE task_id=?",
                 (MAX_TASK_ATTEMPTS, task_id))
        db.close()

        for _ in range(memory_ws.max_deliveries):
            time.sleep(0.25)
            memory_ws.claim("ghost", keep_if=lambda k, c: False)
        assert memory_ws.stats()["dlq"] == 1

        empty_wid = _register(client, "poller-fail-w")
        resp2 = _get_task(client, empty_wid)
        assert resp2 == {"task_id": None, "wait_seconds": 10}

        db = ExoNitDB(db_path)
        row = db.get_task(task_id)
        assert row["status"] == "failed"
        db.close()
        assert memory_ws.stats()["queued"] == 0, (
            "attempts already exhausted — must NOT be re-enqueued")
        assert memory_ws.stats()["dlq"] == 0 and memory_ws.peek_dlq() == [], (
            "G4: the 'failed' outcome is also a real reconciliation — "
            "prune it from the DLQ too")

    def test_dlq_unreconciled_entry_is_not_pruned(self, app_env_short_vt):
        """G4 (review package, 2026-07-05): an entry reconcile_dlq_task
        declines to touch (its own TOCTOU guard: status is no longer
        'assigned' — here simulated as a legitimate late completion race
        that lands between the DLQ-move and the reconciliation scan) must
        be LEFT IN the DLQ, not pruned — pruning a no-op outcome would
        risk silently hiding a genuinely stuck/missing row from a future
        scan (see _reconcile_primary_dlq's docstring)."""
        client, db_path, memory_ws = app_env_short_vt
        _enqueue_bulk(client, ["500801"])
        wid = _register(client, "dlq-race-w")
        resp = _get_task(client, wid)
        task_id = resp["task_id"]

        for _ in range(memory_ws.max_deliveries):
            time.sleep(0.25)
            memory_ws.claim("ghost", keep_if=lambda k, c: False)
        assert memory_ws.stats()["dlq"] == 1

        # Simulate a legitimate late completion that raced the DLQ move —
        # reconcile_dlq_task's own status-guard makes it a no-op now.
        db = ExoNitDB(db_path)
        db.complete_task(task_id, processing_time=1.0)
        db.close()

        empty_wid = _register(client, "poller-race-w")
        resp2 = _get_task(client, empty_wid)
        assert resp2 == {"task_id": None, "wait_seconds": 10}

        db = ExoNitDB(db_path)
        row = db.get_task(task_id)
        assert row["status"] == "completed", (
            "sanity: the legitimate completion must survive untouched")
        db.close()
        assert memory_ws.stats()["dlq"] == 1, (
            "an unreconciled (no-op outcome) entry must remain in the "
            "DLQ, not be pruned")
        assert len(memory_ws.peek_dlq()) == 1


class TestPrimarySweepDefenseInDepth:
    """G1 (review package, 2026-07-05): a low-frequency sweep on the
    empty-poll path that recovers 'assigned' rows with NO stream message
    at all — the case _reconcile_primary_dlq's DLQ-based reconciliation
    structurally cannot reach (no message => nothing ever hits
    max_deliveries or lands in the DLQ). Two proven real causes: (a)
    legacy-era rows already 'assigned' in PG at the very first cutover
    to EXONIT_WORK_STREAMS=1; (b) a steal whose OLD worker's late result
    POST acks the (already-stolen) message, followed by the NEW worker
    also dying — the row stays 'assigned' with its message gone, having
    never reached max_deliveries.
    """

    def test_stuck_assigned_row_with_no_message_is_recovered_by_sweep(
            self, app_env):
        client, db_path, memory_ws = app_env
        db = ExoNitDB(db_path)
        db.add_task("orphan-tk", "600901", priority=2)
        db._exec(
            "UPDATE tasks SET status='assigned', assigned_to=?, "
            "assigned_at=? WHERE task_id=?",
            ("long-gone-worker", "2020-01-01T00:00:00+00:00", "orphan-tk"))
        db.close()
        assert memory_ws.stats() == {"queued": 0, "pending": 0, "dlq": 0}, (
            "sanity: the orphaned row has NO stream message whatsoever — "
            "this is exactly the case _reconcile_primary_dlq cannot reach")

        wid = _register(client, "sweep-recovery-w")
        resp = _get_task(client, wid)
        assert resp == {"task_id": None, "wait_seconds": 10}, (
            "the sweep re-enqueues a fresh message but does not retry "
            "claim() within the SAME request")

        db = ExoNitDB(db_path)
        row = db.get_task("orphan-tk")
        assert row["status"] == "pending", (
            "G1: the sweep must have flipped the messageless orphan back "
            "to pending")
        assert row["assigned_to"] is None
        db.close()
        assert memory_ws.stats()["queued"] == 1, (
            "a fresh stream message must be enqueued for the row the "
            "sweep just recovered")

        # End-to-end: the recovered row must actually be servable again.
        resp2 = _get_task(client, wid)
        assert resp2["task_id"] == "orphan-tk"
        assert resp2["tic_id"] == "600901"
        db = ExoNitDB(db_path)
        row = db.get_task("orphan-tk")
        assert row["status"] == "assigned"
        assert row["assigned_to"] == wid
        db.close()

    def test_throttle_honored_second_poll_within_interval_no_resweep(
            self, app_env, monkeypatch):
        client, db_path, memory_ws = app_env
        wid = _register(client, "throttle-w")

        calls = []
        original = ExoNitDB.reassign_stale_tasks

        def spy(self, *a, **kw):
            calls.append(1)
            return original(self, *a, **kw)

        monkeypatch.setattr(ExoNitDB, "reassign_stale_tasks", spy)

        resp1 = _get_task(client, wid)
        assert resp1 == {"task_id": None, "wait_seconds": 10}
        assert len(calls) == 1, "first empty poll must sweep"

        resp2 = _get_task(client, wid)
        assert resp2 == {"task_id": None, "wait_seconds": 10}
        assert len(calls) == 1, (
            "a second empty poll within EXONIT_PRIMARY_SWEEP_INTERVAL_SECONDS "
            f"(default 300s) must NOT re-run the sweep — got {len(calls)} calls")

    def test_sweep_resumes_after_throttle_interval_elapses(
            self, app_env_short_sweep, monkeypatch):
        client, db_path, memory_ws = app_env_short_sweep
        wid = _register(client, "resweep-w")

        calls = []
        original = ExoNitDB.reassign_stale_tasks

        def spy(self, *a, **kw):
            calls.append(1)
            return original(self, *a, **kw)

        monkeypatch.setattr(ExoNitDB, "reassign_stale_tasks", spy)

        _get_task(client, wid)
        assert len(calls) == 1
        _get_task(client, wid)
        assert len(calls) == 1, "still within the 0.2s throttle window"

        time.sleep(0.3)   # EXONIT_PRIMARY_SWEEP_INTERVAL_SECONDS=0.2 (fixture)
        _get_task(client, wid)
        assert len(calls) == 2, (
            "once the throttle interval elapses, the next empty poll "
            "must sweep again")

    def test_flag_off_never_touches_sweep_gate(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test.db")
        reports_dir = str(tmp_path / "reports")
        os.makedirs(reports_dir, exist_ok=True)
        monkeypatch.delenv("EXONIT_WORK_STREAMS", raising=False)
        monkeypatch.setenv("EXONIT_DB_PATH", db_path)
        monkeypatch.setenv("EXONIT_REPORTS_DIR", reports_dir)

        from exonit.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())

        wid = _register(c, "legacy-sweep-w")
        resp = c.get("/api/worker/task",
                     headers={**AUTH, "X-Worker-ID": wid})
        assert resp.json() == {"task_id": None, "wait_seconds": 10}

        from exonit.shared_state import get_shared_state
        assert get_shared_state().get("primary:sweep_gate") is None, (
            "flag-off path must never touch the G1 sweep's throttle gate "
            "— that gate belongs exclusively to the streams empty-poll path")


class TestPrimaryTaskFlagOff:
    """EXONIT_WORK_STREAMS unset (default) — legacy scan path, untouched."""

    def test_flag_off_uses_legacy_scan_path(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test.db")
        reports_dir = str(tmp_path / "reports")
        os.makedirs(reports_dir, exist_ok=True)
        monkeypatch.delenv("EXONIT_WORK_STREAMS", raising=False)
        monkeypatch.setenv("EXONIT_DB_PATH", db_path)
        monkeypatch.setenv("EXONIT_REPORTS_DIR", reports_dir)

        from exonit.server import create_app
        from fastapi.testclient import TestClient
        c = TestClient(create_app())

        wid = _register(c, "legacy-w")
        resp = c.post("/api/worker/enqueue", json={"tic_id": "800001"},
                      headers=AUTH)
        assert resp.status_code == 200

        resp2 = _get_task(c, wid)
        assert resp2["tic_id"] == "800001"


class TestPrimaryRebuildScript:
    def test_derive_primary_orders_by_priority_then_created_at(
            self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = ExoNitDB(db_path)
        db.add_task("t-a", "900001", priority=2)
        db.add_task("t-b", "900002", priority=1)
        db.add_task("t-c", "900003", priority=2)

        import importlib.util
        import pathlib
        script = (pathlib.Path(__file__).resolve().parents[1]
                  / "scripts" / "rebuild_work_streams.py")
        spec = importlib.util.spec_from_file_location("rebuild_ws_t5", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        rows = mod._derive_primary(db)
        assert len(rows) == 3
        assert rows[0][0] == "t-b", (
            "priority=1 task must sort before priority=2 tasks")
        assert {rows[1][0], rows[2][0]} == {"t-a", "t-c"}
        assert rows[0][1] == {"task_id": "t-b", "tic_id": "900002"}
        db.close()
