"""Worker task/result/heartbeat/registry endpoints (Fase D create_app() split).

Extracted verbatim from exonit.server.create_app() — see
docs/reviews/2026-07-02-opus48-cleanup-review.md and
progress/impl-createapp-split-2026-07-02.md for the module map. Endpoint
bodies below are byte-identical to the pre-split code (git move-detection
should match them); only this file's import/register() preamble is new.

Covers: /api/telemetry (+ per-worker + purge), /api/worker/register,
/api/worker/heartbeat, /api/worker/list, /api/debug/{worker_id},
/api/worker/task, /api/worker/result, /api/candidate/update_analysis,
/api/worker/enqueue(_bulk), /api/worker/fpp_task(+result),
/api/worker/exominer_task(+result), /api/worker/update(+file),
/api/worker/control, /api/worker/download (deployment zip).
"""

import asyncio
import json
import os
import re
import secrets
import time as _time
from typing import Optional

from fastapi import Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from exonit import done_state, server as _server
from exonit import stellar_gates, work_producers, work_streams
from exonit.db import _to_epoch
from exonit.db_backend import column_exists
from exonit.server import (
    ABSOLUTE_ABANDON_TIMEOUT_SECONDS,
    EXOMINER_ASSIGNMENT_TIMEOUT_SECONDS,
    FPP_ASSIGNMENT_TIMEOUT_SECONDS,
    STALE_TASK_TIMEOUT_SECONDS,
    WORKER_ALIVE_WINDOW_SECONDS,
    WORKER_OFFLINE_SECONDS,
    _EXOMINER_ASSIGNMENTS_KEY,
    _EXOMINER_HAS_WORK_KEY,
    _EXOMINER_HAS_WORK_TTL,
    _FPP_ASSIGNMENTS_KEY,
    _FPP_BACKFILL_DONE_KEY,
    _FPP_BACKFILL_STARTED_AT_KEY,
    _PROJECT_ROOT,
    _active_run_sector,
    _build_backfill_payload,
    _claim_candidate,
    _clear_worker_backfill_tic,
    _get_exominer_executor,
    _get_fpp_gate_executor,
    _get_spoc_executor,
    _get_verdict_executor,
    _get_worker_backfill_tic,
    _load_exominer_allowlist,
    _reap_stale_assignments,
    _run_exominer_background,
    _run_fpp_gate_background,
    _run_verdict_background,
    _set_worker_backfill_tic,
    _verdict_enabled,
    _worker_alive_and_on,
    _worker_alive_and_on_current_tic,
    _WORKERS_MANIFEST,
    logger,
)
from exonit.routes.deps import ServerDeps
from exonit.shared_state import get_shared_state


#: Address the fleet page shows for the server itself, and the fallback IP for
#: any worker with no SSH hop (the instances running on the server box).
#: Overridable via the environment so a server migration is a config change,
#: not a code change.
_SERVER_IP = os.environ.get("EXONIT_SERVER_IP", "127.0.0.1")

# NOTE: _get_shared_metadata_cache / target_mass_is_uncomputable /
# _scan_in_flight / _run_spoc_check_background are called as `_server.<name>`
# (not plain-imported) below — tests patch them via
# patch/monkeypatch.setattr("exonit.server.<name>", ...) AFTER the app has
# already been built; see routes/proxies.py's NOTE for the full rationale.


def _insert_tasks_bulk_pg(backend, task_rows, sector, run_id=None) -> None:
    """PG-only: one multi-row INSERT via ``unnest()`` instead of
    ``executemany`` (Q4, perf review 2026-09-26).

    ``executemany`` on psycopg3 still pipelines one statement per row
    (measured 35 s / 300k rows in the review's pg_stat_statements
    window); a naive multi-VALUES INSERT would need 3-4 bound params per
    row and hit PostgreSQL's 65535-parameter-per-statement limit past
    roughly 16-20k rows. ``unnest()`` takes exactly 3-5 ARRAY params
    regardless of row count (one per column, ``sector``/``run_id``
    additive — see below) — one round trip, no chunking needed for this
    reason (the caller may still be chunking for other reasons
    upstream). Column types are cast explicitly (``bigint[]``/
    ``text[]``) to match ``tasks.priority``/``tasks.sector``/
    ``tasks.run_id`` in ``schema_pg.sql`` rather than relying on
    psycopg's inferred array element type.

    run_id (S3, docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md
    §3.1): an ADDITIVE column, same "only referenced when the caller
    passed one" contract as ``sector`` — a caller (or a pre-migration
    SQLite DB, see the sibling ``_insert_tasks_bulk_sqlite``) that never
    passes ``run_id`` gets the EXACT pre-existing SQL text back (3 or 4
    columns), so this generalisation is a byte-for-byte no-op for every
    caller that predates S3.

    Review fix (2026-09-26): ``sector``/``run_id`` are ALSO dropped from
    the INSERT — even when the caller passed one — if
    ``column_exists(backend, "tasks", <col>)`` says the column isn't there
    yet (a code-before-DDL deploy, which the spec's own §6 window table
    schedules for BOTH of these columns). Without this, the first real
    lot after a code deploy would fail EVERY row's INSERT on production
    Postgres the moment ``discovery_run.py`` starts sending ``sector``/
    ``run_id`` — silently converting the whole lot to ``failed`` (caught by
    the caller's broad ``except Exception``) rather than inserting the
    rows without the not-yet-existing column, which is what every other
    "additive column" call site in this codebase already does
    byte-for-byte until its own DDL lands.
    """
    task_ids = [t[0] for t in task_rows]
    tic_ids = [t[1] for t in task_rows]
    priorities = [t[2] for t in task_rows]
    columns = ["task_id", "tic_id", "priority"]
    array_casts = ["?::text[]", "?::text[]", "?::bigint[]"]
    params = [task_ids, tic_ids, priorities]
    if sector is not None and column_exists(backend, "tasks", "sector"):
        columns.append("sector")
        array_casts.append("?::bigint[]")
        params.append([sector] * len(task_rows))
    if run_id is not None and column_exists(backend, "tasks", "run_id"):
        columns.append("run_id")
        array_casts.append("?::text[]")
        params.append([str(run_id)] * len(task_rows))
    backend.execute(
        f"INSERT INTO tasks ({', '.join(columns)}) "
        f"SELECT * FROM unnest({', '.join(array_casts)})",
        tuple(params))


def _insert_tasks_bulk_sqlite(backend, task_rows, sector, run_id=None) -> None:
    """SQLite counterpart of ``_insert_tasks_bulk_pg`` (S3): same
    additive-column generalisation of the pre-Q4 ``executemany`` path,
    byte-for-byte unchanged SQL text when *run_id*/*sector* are absent OR
    (review fix, 2026-09-26) not yet present as columns — SQLite normally
    self-migrates on every process start (``exonit/db.py``'s
    ``ExoNitDB.__init__``), but a caller can still hand this function a
    pre-migration connection (e.g. a test simulating that exact state, or
    an operational DB opened some other way), so the same
    ``column_exists`` guard as the PG path applies here too.
    """
    columns = ["task_id", "tic_id", "priority"]
    has_sector = sector is not None and column_exists(backend, "tasks", "sector")
    has_run_id = run_id is not None and column_exists(backend, "tasks", "run_id")
    if has_sector:
        columns.append("sector")
    if has_run_id:
        columns.append("run_id")
    placeholders = ", ".join("?" * len(columns))
    rows = []
    for tid, tic, prio in task_rows:
        row = [tid, tic, prio]
        if has_sector:
            row.append(sector)
        if has_run_id:
            row.append(str(run_id))
        rows.append(tuple(row))
    backend.executemany(
        f"INSERT INTO tasks ({', '.join(columns)}) VALUES ({placeholders})",
        rows)


def _adopt_live_tasks(backend, tics: list, run_id: str) -> int:
    """S4 §2.1: re-tag LIVE (pending/assigned) tasks for *tics* to
    *run_id*, EXCEPT tasks already owned by a still-live foreign run
    (``discovery_runs.state NOT IN ('FAILED', 'DONE')``). Returns the
    total UPDATE rowcount across chunks — the authoritative ``adopted``
    count when this runs (see ``_enqueue_bulk_blocking``'s docstring).

    A task with ``run_id`` already equal to *run_id* is matched too (the
    ``run_id = ?`` arm of the WHERE) — harmless (0-row UPDATE effect) but
    still counted, which is exactly right for a Q5 chunk retry WITHIN
    the same lot: the retried chunk's tasks were already tagged by the
    first (network-blipped-response) attempt, and the caller's ``total =
    created + adopted`` must still reach the real lot size.

    Never raises: a missing ``tasks.run_id`` column (S3 DDL not deployed
    yet, or a pre-migration SQLite DB) is checked FOR up front via
    ``column_exists`` (review fix, 2026-09-26 — was try/execute/catch
    only; the proactive check avoids a full round-trip + stack-trace log
    per chunk in the common code-before-DDL window, and skips straight to
    the sentinel) and the UPDATE itself is still wrapped in try/except as
    a defence-in-depth fallback (e.g. a permissions issue, or a race where
    the column was dropped between the check and the query) — either way
    this never fails the whole enqueue_bulk request over a best-effort
    re-tag.
    """
    if not tics:
        return 0
    if not column_exists(backend, "tasks", "run_id"):
        logger.info(
            "S4 adoption UPDATE skipped for run_id=%s (%d tics) -- "
            "tasks.run_id not present yet (pre-DDL); falling back to the "
            "TIC-count estimate for 'adopted'", run_id, len(tics))
        return -1  # sentinel: caller keeps its provisional estimate
    total = 0
    _CHUNK = 10000
    try:
        for i in range(0, len(tics), _CHUNK):
            chunk = tics[i:i + _CHUNK]
            placeholders = ",".join("?" * len(chunk))
            sql = (
                "UPDATE tasks SET run_id = ? "
                f"WHERE tic_id IN ({placeholders}) "
                "AND status IN ('pending', 'assigned') "
                "AND (run_id IS NULL OR run_id = ? "
                "OR NOT EXISTS (SELECT 1 FROM discovery_runs r "
                "WHERE r.run_id = tasks.run_id "
                "AND r.state NOT IN ('FAILED', 'DONE')))"
            )
            params = (str(run_id), *chunk, str(run_id))
            total += backend.execute_rowcount(sql, params)
    except Exception:
        from exonit.db_backend import invalidate_column_exists
        invalidate_column_exists(backend, "tasks", "run_id")
        logger.warning(
            "S4 adoption UPDATE failed for run_id=%s (%d tics) — "
            "falling back to the TIC-count estimate for 'adopted' "
            "(non-fatal: tasks.run_id may predate the S3 DDL; column "
            "existence cache invalidated so the next call re-checks)",
            run_id, len(tics), exc_info=True)
        return -1  # sentinel: caller keeps its provisional estimate
    return total


def register(app, deps: ServerDeps) -> None:
    db = deps.db
    cfg = deps.cfg
    berger = deps.berger
    workers_state = deps.workers_state
    _check_auth = deps.check_auth
    _cached_json = deps.cached_json

    # --- Worker telemetry (per-star timing breakdown) shared state ---------
    # Cross-process (PATRÓ A3): stored in shared_state so all uvicorn worker
    # processes see the same records regardless of which one wrote them.
    _TELEMETRY_KEY = "telemetry:records"                # field=seq -> timing record
    _TELEMETRY_SEQ_KEY = "telemetry:seq"
    _TELEMETRY_MAX = 500
    # Pending dashboard->worker commands (pause/resume/shutdown), consumed by
    # the next heartbeat. field=worker_id -> {"command": str, "ts": epoch}.
    _PENDING_COMMANDS_KEY = "worker:pending_commands"
    # A "shutdown" (or pause/resume) queued for an offline worker sits in
    # Redis until that worker's NEXT heartbeat — which can be weeks later if
    # the machine was simply turned off. The 2026-07-10 incident: <host> (x6)
    # and HP came back online after being off for a long stretch, and their
    # very first heartbeat consumed a stale "shutdown" queued long before,
    # killing the process 1s after it registered with 0 tasks done. A command
    # this old can no longer reflect operator intent for the worker that's
    # now starting up, so anything older than this (or missing a timestamp —
    # i.e. queued before this fix shipped) is discarded unconsumed instead of
    # being honoured.
    _PENDING_COMMAND_MAX_AGE_SECONDS = 300

    def _clear_pending_command(worker_id) -> None:
        """Drop any pending dashboard command for *worker_id*.

        Called from worker_register: a worker that just (re)started cannot
        have a legitimate command queued from before it existed in this
        process's lifetime — see _PENDING_COMMAND_MAX_AGE_SECONDS above for
        the incident this guards against.
        """
        get_shared_state().hdel(_PENDING_COMMANDS_KEY, str(worker_id))

    def _telemetry_append(record: dict) -> None:
        st = get_shared_state()
        seq = st.incr(_TELEMETRY_SEQ_KEY)
        st.hset_json(_TELEMETRY_KEY, str(seq), record)
        # Periodic trim: drop fields older than the last _TELEMETRY_MAX.
        if seq % 50 == 0:
            cutoff = seq - _TELEMETRY_MAX
            for k in st.hkeys(_TELEMETRY_KEY):
                try:
                    if int(k) <= cutoff:
                        st.hdel(_TELEMETRY_KEY, k)
                except ValueError:
                    st.hdel(_TELEMETRY_KEY, k)

    def _telemetry_records() -> list:
        recs = get_shared_state().hgetall_json(_TELEMETRY_KEY)
        return [recs[k] for k in sorted(recs, key=lambda x: int(x))]

    @app.get("/api/telemetry")
    def api_telemetry(worker: Optional[str] = Query(None),
                      limit: int = Query(50)):
        """Worker performance telemetry — timing breakdown per star."""
        records = _telemetry_records()
        data = records[-limit:]
        if worker:
            data = [t for t in data if t.get("worker") == worker]

        # Compute averages per worker
        by_worker = {}
        for t in records:
            w = t.get("worker", "unknown")
            if w not in by_worker:
                by_worker[w] = {"count": 0, "load": 0, "bls": 0, "vetting": 0,
                                "cnn": 0, "injection": 0, "total": 0}
            by_worker[w]["count"] += 1
            for k in ["load", "bls", "vetting", "cnn", "injection", "total"]:
                by_worker[w][k] += t.get(k, 0)

        summary = {}
        for w, s in by_worker.items():
            n = s["count"]
            summary[w] = {
                "count": n,
                "avg_load": round(s["load"] / n, 2),
                "avg_bls": round(s["bls"] / n, 2),
                "avg_vetting": round(s["vetting"] / n, 2),
                "avg_cnn": round(s["cnn"] / n, 2),
                "avg_injection": round(s["injection"] / n, 2),
                "avg_total": round(s["total"] / n, 2),
                # Latest backend per stage for this worker. `bls_backend` and
                # `injection_backend` are emitted by workers running the post-
                # 2026-05-14 telemetry split; fall back to the legacy
                # `gpu_backend` (which only ever reflected BLS) for older
                # workers still on the cloud queue. `or "unknown"` covers the
                # case where the field is absent on legacy records (yields
                # None, which `next` would otherwise return verbatim).
                "bls_backend": next((t.get("bls_backend") or t.get("gpu_backend") or "unknown"
                                     for t in reversed(records)
                                     if t.get("worker") == w), "unknown"),
                "injection_backend": next((t.get("injection_backend") or "unknown"
                                           for t in reversed(records)
                                           if t.get("worker") == w), "unknown"),
                "gpu_backend": next((t.get("gpu_backend") for t in reversed(records)
                                     if t.get("worker") == w), "unknown"),
            }

        return {"records": data, "summary": summary}

    # --- Worker API routes ---

    def _release_restarted_worker_claims(worker_id, name) -> None:
        """Give back the work a just-restarted worker process was holding.

        /api/worker/register is called by a worker process at STARTUP, so by
        definition the process on the other end holds no in-flight work — yet
        the server may still be holding some FOR it:

          * work-stream PEL entries it claimed moments before it died. The
            consumer name is derived from the worker NAME, so the restarted
            process re-registers under the SAME id and the entries look owned
            by a live consumer. Nobody serves them; nobody acks them.
          * a stale ``worker:backfill_tic`` record, which makes
            _worker_alive_and_on() veto any steal of that TIC's item.

        On 2026-07-14 the fleet autoscaler restarted the fleet at the
        SCAN->FPP transition, ~40 s after 7 workers had claimed FPP items.
        The run then sat at FPP_BACKFILL with an empty queue and 7 unserved
        items for the FULL visibility timeout (30 min) before the steal path
        recovered them. The timeouts are the backstop; this is the fix — and
        it is what makes fleet autoscaling safe to run mid-phase at all.

        Best-effort by construction: a Redis hiccup here must degrade to the
        old timeout-based recovery, never to a worker that cannot register.
        """
        try:
            _clear_worker_backfill_tic(worker_id)
        except Exception:
            logger.warning("register: could not clear backfill_tic for %s",
                           name, exc_info=True)
        if not work_producers.streams_enabled():
            return
        for stage in ("primary", "fpp", "exominer"):
            try:
                n = work_streams.get_stream(stage).release_consumer(worker_id)
                if n:
                    logger.info(
                        "register: %s restarted holding %d %s item(s) — "
                        "requeued immediately (would otherwise wait out the "
                        "%ds visibility timeout)",
                        name, n, stage,
                        work_streams.get_stream(stage).visibility_timeout_s)
            except Exception:
                logger.warning(
                    "register: could not release %s claims for %s (falling "
                    "back to the visibility timeout)", stage, name,
                    exc_info=True)

    @app.post("/api/worker/register")
    async def worker_register(request: Request):
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        name = body.get("name", "unnamed")

        # Re-register: reuse existing worker_id and preserve stats
        existing = [(wid, w) for wid, w in workers_state.items()
                    if w["name"] == name]
        if existing:
            worker_id, old_state = existing[0]
            # Remove any duplicates (keep first)
            for wid, _ in existing[1:]:
                del workers_state[wid]

            # Self-heal: workers_state (Redis "workers:registry") is a cache
            # keyed by worker_id, but it can drift from workers_registry (PG/
            # SQLite) — e.g. a stale entry keyed by the worker's *name*
            # instead of its worker_id (the 2026-07-10 worker-<host>-1
            # incident: DB had worker_id=ca7ae5fc3220f649 for this name, but
            # workers_state's key WAS the name). The DB is the source of
            # truth for the (name -> worker_id) mapping; if it disagrees with
            # what we're about to use, fix workers_state instead of trusting
            # the stale cache — otherwise register_worker's name-uniqueness
            # (see db.py) makes every subsequent call 500 forever.
            db_worker_id = db.get_worker_id_by_name(name)
            if db_worker_id and db_worker_id != worker_id:
                logger.warning(
                    "worker_register: workers_state had stale worker_id=%r "
                    "for name=%r, DB says %r — reconciling",
                    worker_id, name, db_worker_id)
                preserved = dict(workers_state[worker_id])
                del workers_state[worker_id]
                bad_worker_id, worker_id = worker_id, db_worker_id
                if worker_id in workers_state:
                    # The good worker_id already has its own entry (e.g. a
                    # previous heartbeat cycle) — keep it and just fold in
                    # whatever counters the stale-keyed entry had accrued.
                    workers_state[worker_id].update({
                        k: v for k, v in preserved.items()
                        if k in ("tasks_completed", "tasks_failed",
                                  "total_processing_time")
                    })
                else:
                    # No entry under the correct worker_id yet — seed one
                    # from the DB-persisted counters (source of truth for a
                    # worker_id we've never held state for in this process),
                    # falling back to whatever the stale entry had.
                    stats = db.get_worker_stats(worker_id)
                    preserved.pop("name", None)
                    preserved.setdefault(
                        "tasks_completed", stats["tasks_completed"])
                    preserved.setdefault(
                        "tasks_failed", stats["tasks_failed"])
                    preserved.setdefault(
                        "total_processing_time", stats["total_processing_time"])
                    workers_state[worker_id] = preserved
                logger.info(
                    "worker_register: workers_state cleaned up, dropped "
                    "stale key %r for name=%r", bad_worker_id, name)

            workers_state[worker_id].update({
                "name": name,
                "capabilities": body.get("capabilities", {}),
                "status": "idle",
                "last_heartbeat": _time.time(),
                "current_tic": None,
                # Preserve: tasks_completed, total_processing_time
            })
            # The worker process has just restarted: drop any cached
            # debug_mode state so the next heartbeat re-fires the
            # debug_on/debug_off transition based on the DB.
            workers_state[worker_id].pop("debug_mode", None)
            db.register_worker(worker_id, name)
            _clear_pending_command(worker_id)
            _release_restarted_worker_claims(worker_id, name)
            return {"worker_id": worker_id}

        # Prefer a stable worker_id persisted across server restarts.
        worker_id = db.get_worker_id_by_name(name) or secrets.token_hex(8)
        # Rehydrate counters persisted from previous sessions so a restart
        # doesn't silently reset the dashboard numbers to zero.
        stats = db.get_worker_stats(worker_id)
        workers_state[worker_id] = {
            "name": name,
            "capabilities": body.get("capabilities", {}),
            "status": "idle",
            "last_heartbeat": _time.time(),
            "current_tic": None,
            "tasks_completed": stats["tasks_completed"],
            "tasks_failed": stats["tasks_failed"],
            "total_processing_time": stats["total_processing_time"],
        }
        db.register_worker(worker_id, name)
        _clear_pending_command(worker_id)
        _release_restarted_worker_claims(worker_id, name)
        return {"worker_id": worker_id}

    @app.post("/api/worker/heartbeat")
    async def worker_heartbeat(request: Request):
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        wid = body.get("worker_id")
        if wid not in workers_state:
            # Worker unknown (server restarted?) — tell it to re-register
            return {"command": "re-register"}

        # P6 (2026-07-02): batch every per-heartbeat registry write into a
        # single .update() round trip (one hget_json + one hset_json)
        # instead of one hget_json+hset_json pair PER FIELD (previously
        # 5-9 Redis round trips per heartbeat — see worker_registry.py's
        # _WorkerEntryView.__setitem__). One prior read (`prev`) supplies
        # both the "did current_tic change" and "did debug_mode change"
        # comparisons that used to re-read the registry mid-function.
        prev = workers_state.get(wid, {})
        new_tic = body.get("current_tic")
        prev_tic = prev.get("current_tic")
        debug_on = db.get_debug_mode(wid)
        worker_was_debug = prev.get("debug_mode", False)

        updates = {
            "status": body.get("status", "idle"),
            "last_heartbeat": _time.time(),
            "current_tic": new_tic,
            "debug_mode": debug_on,
        }
        if new_tic != prev_tic:
            updates["tic_started_at"] = _time.time() if new_tic else 0
        # Per-worker introspection fields exposed via /api/worker/list for
        # farmctl status. Optional in the heartbeat body; old workers omit
        # them and the keys remain absent from the registry — farmctl shows
        # a placeholder rather than crashing.
        for k in ("git_sha", "flag_states", "uptime_s", "pid"):
            if k in body:
                updates[k] = body[k]
        workers_state[wid].update(updates)

        # Persist telemetry
        db.insert_telemetry(wid, body)

        # Check for debug mode toggle (using the pre-update snapshot so this
        # reflects what actually changed this heartbeat, not the value we
        # just wrote above).
        if debug_on != worker_was_debug:
            cmd = "debug_on" if debug_on else "debug_off"
            return {"command": cmd}

        st = get_shared_state()
        pending = st.hget_json(_PENDING_COMMANDS_KEY, str(wid))
        if pending:
            st.hdel(_PENDING_COMMANDS_KEY, str(wid))
            ts = pending.get("ts")
            # Missing ts = queued before this fix (or by an older client) —
            # treat as stale rather than assume it's fresh. See
            # _PENDING_COMMAND_MAX_AGE_SECONDS above for why an old command
            # must not be honoured on a worker that's only just come back.
            age = (_time.time() - ts) if ts else float("inf")
            if age <= _PENDING_COMMAND_MAX_AGE_SECONDS:
                return {"command": pending.get("command", "continue")}
            logger.warning(
                "worker_heartbeat: discarding stale pending command %r for "
                "worker_id=%r (age=%.0fs > %ds)",
                pending.get("command"), wid, age, _PENDING_COMMAND_MAX_AGE_SECONDS)
        return {"command": "continue"}

    @app.get("/api/worker/list")
    def worker_list():
        def _compute():
            now = _time.time()
            # Read-only: no workers_state mutation. Stale cleanup happens in
            # the background loop below so repeated GETs can't evict a worker
            # that's mid-injection-recovery and briefly paused heartbeats.
            # Cached (2026-09-25, fix A) — shares the aggregate with
            # /api/status and /api/stats instead of each recomputing it on
            # its own TTL; see ExoNitDB.get_task_stats_cached's docstring.
            task_stats = db.get_task_stats_cached()
            workers_out = []
            for wid, w in workers_state.items():
                ago = int(now - w["last_heartbeat"])
                # Count from in-memory (resets on server restart, but accurate within session)
                db_done = w.get("tasks_completed", 0)
                db_avg = round(w.get("total_processing_time", 0) / max(1, db_done), 1) if db_done else 0
                latest = db.get_latest_telemetry(wid)
                is_offline = ago >= WORKER_OFFLINE_SECONDS
                # Backfill (FPP/ExoMiner) TIC is tracked server-side at assignment
                # because the worker's heartbeat is GIL-starved during the compute
                # and reports current_tic=None by the time it fires. Prefer it over
                # the (stale) heartbeat value so the dashboard shows who does what.
                bf = None if is_offline else _get_worker_backfill_tic(wid)
                cur_tic = bf[0] if bf else (None if is_offline else w["current_tic"])
                cur_status = "offline" if is_offline else ("backfill" if bf else w["status"])
                workers_out.append({
                    "worker_id": wid,
                    "name": w["name"],
                    "capabilities": w["capabilities"],
                    "status": cur_status,
                    "online": not is_offline,
                    "current_tic": cur_tic,
                    "current_task": (bf[1] + " backfill") if bf else None,
                    "tasks_completed": db_done or 0,
                    "stars_completed": db_done or 0,
                    "last_heartbeat_ago": ago,
                    "avg_seconds_per_star": round(db_avg or 0, 1),
                    "phase": None if is_offline else (latest["phase"] if latest else None),
                    "phase_progress": None if is_offline else (latest["phase_progress"] if latest else None),
                    "debug_mode": db.get_debug_mode(wid),
                    # farmctl introspection (set when the worker reports them)
                    "git_sha": w.get("git_sha"),
                    "flag_states": w.get("flag_states"),
                    "uptime_s": None if is_offline else w.get("uptime_s"),
                    "pid": w.get("pid"),
                })
            return {
                "workers": workers_out,
                "tasks": task_stats,
            }
        return _cached_json("worker_list", 5.0, _compute)

    # ── Status fleet map endpoint ──────────────────────────────────────

    # Names a worker reports in `bls_gpu_info` when it has NO accelerated BLS
    # backend. They are the name of a *software* fallback, not of a GPU, so
    # they must never win over the real device string from the hardware probe.
    _NOT_A_GPU_NAME = {
        "numba", "cpu", "cpu only (numba)", "none", "-", "—",
        # Backend tokens the native (Rust/Swift) worker ports report as their
        # "gpu" capability string instead of a hardware name — e.g. the Rust
        # port on <host> reports bls_gpu_info="cuda" while its Python FPP
        # sidecar (worker-<host>-rust-py) reports the real device string
        # ("NVIDIA GeForce RTX 4070 (12.0 GB)"). Without this, merging the
        # two families into one machine card (_machine_key, 2026-08-16) let
        # whichever family iterated first win, and the backend token often
        # beat the real name.
        "cuda", "opencl", "sycl", "metal", "ispc",
    }

    def _gpu_identity(insts, platform_name):
        """(gpu_name, gpu_backend) for a machine, from ALL its instances' caps.

        Why this is not just `caps["bls_gpu_info"]` (the pre-2026-07-25
        behaviour): on every Mac in the fleet BLS runs on numba (MLX is not
        the BLS backend there) while injection and FPP run on Metal. Reading
        the GPU identity off the BLS backend therefore reported "No GPU
        reported (numba)" for six Apple machines that were, at that moment,
        pushing 80-100% GPU utilisation and 85 C — the telemetry columns next
        to it proved the GPU was real and busy. Owner report 2026-07-25.

        Three sources, in this order, because each is right where the others
        are wrong:
          1. `bls_gpu_info` when it names real hardware — it carries the most
             specific string anyone reports ("Intel(R) Arc(TM) Graphics"),
             whereas `hardware["opencl"]` is the generic probe label
             ("AMD/Intel OpenCL iGPU") on the same machine.
          2. the `hardware` probe map — the only source on Apple Silicon,
             where it holds "Apple M4 (Metal/MLX)".
          3. platform + the backend actually in use — the M1/M3 Macs report
             `hardware.metal == false` (no MLX installed) yet run
             `fpp_backend == "metal-kernel"`; naming them from the platform is
             the honest answer, and beats claiming they have no GPU.

        Aggregated over every instance, not just the representative one:
        <host> deliberately runs 3 CPU instances + 1 Arc instance, so which
        instance answers must not change what the machine appears to be.
        """
        name = None
        accel = None
        # Preference order for the accelerator, strongest first: a machine
        # with any CUDA instance is a CUDA machine.
        rank = {"cuda": 3, "metal": 2, "opencl": 1}
        for _wid, w in insts:
            caps = w.get("capabilities") or {}
            hw = caps.get("hardware") or {}
            if name is None:
                cand = caps.get("bls_gpu_info") or caps.get("gpu")
                if cand and str(cand).strip().lower() not in _NOT_A_GPU_NAME:
                    name = str(cand).strip()
            if name is None:
                for key in ("cuda", "metal", "opencl"):
                    v = hw.get(key)
                    if isinstance(v, str) and v.strip():
                        name = v.strip()
                        break
            # What the machine actually computes on — BLS *or* FPP, since a
            # host can be numba for BLS and Metal/OpenCL for FPP+injection.
            backends = f"{caps.get('bls_backend') or ''} {caps.get('fpp_backend') or ''}".lower()
            for key in ("cuda", "metal", "opencl"):
                if key in backends and rank[key] > rank.get(accel, 0):
                    accel = key
        if name is None and accel == "metal" and platform_name == "macos":
            name = "Apple Silicon GPU"
        if accel is None:
            # No accelerated backend anywhere on this machine: report the BLS
            # backend verbatim so a genuinely CPU-only box still reads as one.
            for _wid, w in insts:
                bls = ((w.get("capabilities") or {}).get("bls_backend"))
                if bls:
                    accel = bls
                    break
        return name or "—", accel

    # A physical machine can have several worker FAMILIES registered under
    # different names at once: the native port (`-rust`), its FPP-only
    # Python sidecar (`-rust-py`), the Swift port (`-swift`), and — while a
    # migration is mid-flight or a retired deployment's rows haven't aged
    # out yet — the original plain-Python family with no suffix at all.
    # Longest-first so "-rust-py" isn't swallowed by "-rust".
    _FAMILY_SUFFIXES = ("-rust-py", "-rust", "-swift")

    def _machine_key(name):
        """Physical-machine grouping key for a worker instance/manifest name.

        Strips, in this order: the trailing `-N` instance suffix (as
        status_fleet always has), then at most ONE worker-family suffix.
        `worker-<host>-rust-py-3` -> `worker-<host>`;
        `worker-<host>-swift-2` -> `worker-<host>`;
        `worker-default-rust-1` -> `worker-default`. Only the listed family
        suffixes are stripped — `worker-<host>` must NOT collapse into
        `worker-<host>` (a different, real machine: the M1 mac mini), so an
        arbitrary trailing `-m4`/`-<anything>` is never touched.

        Without this, /api/status/fleet showed one card per REGISTERED
        FAMILY instead of one per box: 25 cards for ~11 real machines
        (owner report 2026-08-16), each with partial data (the rust family
        has cpu% but no real gpu_name; the Python sidecar has the GPU/RAM
        but is a "different" machine to the old grouping).
        """
        base = re.sub(r"-\d+$", "", name)
        for suf in _FAMILY_SUFFIXES:
            if base.endswith(suf):
                return base[: -len(suf)]
        return base

    # A machine not present in ops/workers.json is only shown if at least
    # one of its instances has heartbeated within this window. Retired
    # families/hosts (worker-HP, out of the fleet since 2026-08-07; test
    # artefacts like worker-sidecar-livetest) still have rows in
    # workers_state from their last session and must eventually fall off
    # the map instead of showing up forever as a stale "offline" card. Seven
    # days is deliberately generous — well past WORKER_OFFLINE_SECONDS (a
    # few minutes) and past any single overnight run — so a machine that's
    # merely off for the weekend, or mid-family-migration, still shows up;
    # only something nobody has touched for over a week is dropped.
    _FLEET_STALE_SECONDS = 7 * 24 * 3600

    @app.get("/api/status/fleet")
    def status_fleet():
        def _compute():
            now = _time.time()
            # Static per-computer info from ops/workers.json, grouped by
            # _machine_key too: today every manifest entry's `name` is
            # already a bare machine key (no family suffix), but a manifest
            # entry could in principle carry one (e.g. a future dedicated
            # "-rust-py" sizing row) and must land on the same card as its
            # siblings rather than open a second one.
            try:
                manifest = json.loads(_WORKERS_MANIFEST.read_text())
            except Exception:
                manifest = {"workers": []}
            static = {}
            for w in manifest.get("workers", []):
                name = w.get("name")
                if not name:
                    continue
                comp = _machine_key(name)
                entry = static.setdefault(
                    comp, {"ip": None, "platform": None,
                           "_instances_sum": 0, "_instances_fpp_sum": 0})
                # ssh_host is null for the local box — there is no SSH hop
                # to itself. Fall back to the server's own address rather
                # than showing a machine with no IP at all. worker-default
                # (the instances running ON the server box) used to be
                # skipped entirely here; it is a real machine doing real
                # work (and the one whose spare CPU matters most, since it
                # also serves this dashboard), so it is included on the same
                # terms as the rest.
                ip = ((w.get("ssh_host") or "").split("@")[-1] or _SERVER_IP)
                if not entry["ip"]:
                    entry["ip"] = ip
                if not entry["platform"] and w.get("platform"):
                    entry["platform"] = w.get("platform")
                # `instances` (scan-phase sizing) and `instances_fpp`
                # (FPP-phase sizing) are two DIFFERENT-PHASE pools that never
                # run concurrently — farmctl prunes down to whichever one the
                # current phase calls for (see ops/workers.json's _doc).
                # Summing them (the first cut of this fix) overcounted actual
                # concurrent capacity: <host> (8 scan / 17 fpp) came out
                # as 25 when the real ceiling is 17; <host> (8 rust scan /
                # 10 rust-py fpp) came out as 18 when it's 10. Each entry's
                # own two fields are summed first (a manifest could in
                # principle list more than one entry for the same machine
                # key), THEN we take the max across entries below —
                # matching "the biggest pool this machine is ever sized to
                # run" rather than "every pool added together".
                entry["_instances_sum"] += int(w.get("instances", 1) or 0)
                entry["_instances_fpp_sum"] += int(w.get("instances_fpp", 0) or 0)
            for entry in static.values():
                entry["instances_total"] = max(
                    entry.pop("_instances_sum"), entry.pop("_instances_fpp_sum"))
            manifest_keys = set(static.keys())

            # Group live instances by computer: strip the -N instance
            # suffix, then the worker-family suffix (rust/rust-py/swift) so
            # every family registered for the same physical box lands on
            # one card. See _machine_key's docstring for the bug this fixes.
            groups = {}
            for wid, w in workers_state.items():
                comp = _machine_key(w["name"])
                groups.setdefault(comp, []).append((wid, w))

            computers = []
            for comp, insts in groups.items():
                # Drop machines that are neither in the current manifest nor
                # have heard from in _FLEET_STALE_SECONDS: retired families
                # (worker-HP, out since 2026-08-07) and test artefacts
                # (worker-sidecar-livetest) fall off the map instead of
                # accumulating as permanent stale cards. A manifest machine
                # is always kept, even fully offline (e.g. a laptop worker, off for
                # days) — it's still configured fleet, just paused.
                in_manifest = comp in manifest_keys
                if not in_manifest:
                    newest_hb = max((w["last_heartbeat"] for _, w in insts), default=0)
                    if (now - newest_hb) >= _FLEET_STALE_SECONDS:
                        continue

                st = static.get(comp, {})
                online = [(wid, w) for wid, w in insts
                          if (now - w["last_heartbeat"]) < WORKER_OFFLINE_SECONDS]

                # Telemetry: try each instance, online (most recent
                # heartbeat first) before offline, and keep the first one
                # that actually reports numbers. A merged card can have one
                # family that never sends system/GPU telemetry (e.g. the
                # Rust worker) sitting next to one that does (its Python FPP
                # sidecar) — picking a fixed "representative" would silently
                # blank out the card's cpu/ram/gpu columns whenever that
                # family happens to answer first.
                def _hb_sort_key(item, _now=now):
                    _wid, _w = item
                    is_online = (_now - _w["last_heartbeat"]) < WORKER_OFFLINE_SECONDS
                    return (0 if is_online else 1, -_w["last_heartbeat"])

                latest = {}
                for cand_wid, _cand_w in sorted(insts, key=_hb_sort_key):
                    row = db.get_latest_telemetry(cand_wid)
                    if not row:
                        continue
                    row = dict(row) if hasattr(row, "keys") else row
                    telem_fields = (
                        "cpu_percent", "ram_used_gb", "ram_total_gb",
                        "gpu_util_pct", "gpu_mem_used_gb", "gpu_mem_total_gb",
                        "gpu_temp_c", "gpu_power_w",
                    )
                    if any(row.get(f) is not None for f in telem_fields):
                        latest = row
                        break

                any_working = any(w["status"] == "working" for _, w in online)
                gpu_name, gpu_backend = _gpu_identity(online or insts, st.get("platform"))
                # cpu_cores: max over every instance in the merged group that
                # reports it — the old "representative only" read silently
                # dropped it whenever the instance answering first happened
                # to be from a family that doesn't report cpu_cores.
                cpu_cores = None
                for _wid, w in insts:
                    c = (w.get("capabilities") or {}).get("cpu_cores")
                    if isinstance(c, (int, float)) and (cpu_cores is None or c > cpu_cores):
                        cpu_cores = c
                # instances_total: manifest capacity when configured; a
                # machine outside the manifest (still shown because it has
                # recent heartbeats — see the staleness filter above) falls
                # back to its live online count, never the raw historical
                # row count, which only grows as retired families' rows age.
                instances_total = st.get("instances_total")
                if instances_total is None:
                    instances_total = len(online)
                computers.append({
                    "id": comp,
                    "friendly_name": comp.replace("worker-", ""),
                    "platform": st.get("platform"),
                    "ip": st.get("ip"),
                    "gpu_name": gpu_name,
                    "gpu_backend": gpu_backend,
                    "cpu_cores": cpu_cores,
                    "instances_total": instances_total,
                    "instances_online": len(online),
                    "online": len(online) > 0,
                    "status": "working" if any_working else ("idle" if online else "offline"),
                    "last_heartbeat_ago": int(now - max((w["last_heartbeat"] for _, w in insts), default=now)),
                    "cpu_percent": latest.get("cpu_percent"),
                    "ram_used_gb": latest.get("ram_used_gb"),
                    "ram_total_gb": latest.get("ram_total_gb"),
                    "gpu_util_pct": latest.get("gpu_util_pct"),
                    "gpu_mem_used_gb": latest.get("gpu_mem_used_gb"),
                    "gpu_mem_total_gb": latest.get("gpu_mem_total_gb"),
                    "gpu_temp_c": latest.get("gpu_temp_c"),
                    "gpu_power_w": latest.get("gpu_power_w"),
                    # Historical, not filtered by online/family: instances
                    # from a machine's now-retired family (e.g. the old
                    # Python worker-<host> before the Rust port) still
                    # contributed real completed tasks to this physical box.
                    "tasks_completed": sum(w.get("tasks_completed", 0) for _, w in insts),
                })
            computers.sort(key=lambda c: c["id"])
            return {
                "server": {"name": "ExoNit server", "ip": _SERVER_IP,
                           "port": 8400, "healthy": True},
                "computers": computers,
            }
        return _cached_json("status_fleet", 5.0, _compute)

    # --- Telemetry endpoints ---

    @app.post("/api/telemetry/purge")
    async def api_telemetry_purge(request: Request):
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        retention = 5  # default retention days
        db.purge_telemetry(retention)
        return {"status": "ok", "retention_days": retention}

    @app.get("/api/telemetry/{worker_id}/latest")
    def api_telemetry_latest(worker_id: str):
        row = db.get_latest_telemetry(worker_id)
        if not row:
            return JSONResponse({"error": "no data"}, status_code=404)
        return dict(row)

    @app.get("/api/telemetry/{worker_id}")
    def api_telemetry_worker(worker_id: str, limit: int = Query(100)):
        rows = db.get_telemetry(worker_id, limit)
        return {"records": [dict(r) for r in rows]}

    # --- Debug mode endpoints ---

    @app.post("/api/debug/{worker_id}")
    async def api_debug_set(worker_id: str, request: Request):
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        enabled = body.get("enabled", False)
        db.set_debug_mode(worker_id, enabled)
        return {"status": "ok", "debug_enabled": enabled}

    @app.get("/api/debug/{worker_id}")
    def api_debug_get(worker_id: str):
        enabled = db.get_debug_mode(worker_id)
        return {"debug_enabled": enabled}

    # --- Primary (BLS) task queue over the "primary" work stream ----------

    _PRIMARY_STREAM_MAX_ITERATIONS = 5  # same bound as the FPP/ExoMiner
                                        # streams: cap one HTTP request's
                                        # work even if a run of not-pending
                                        # (already-handled elsewhere)
                                        # entries lands back to back.

    # G1 (review package, 2026-07-05): defense-in-depth sweep throttle.
    # _reconcile_primary_dlq (below) only recovers 'assigned' rows whose
    # stream message survived long enough to hit max_deliveries — it can
    # never touch a row that has NO message at all. Two proven ways that
    # happens: (a) legacy-era 'assigned' rows already sitting in PG at the
    # very first cutover to EXONIT_WORK_STREAMS=1 (never had a message to
    # begin with); (b) steal -> the OLD worker's late result POST acks the
    # (already-stolen) message -> the NEW worker later dies too -> the row
    # is 'assigned' but its message is gone (acked by the race) and it
    # never reached max_deliveries either, so it is not in the DLQ. Legacy
    # recovered both for free on every empty poll via its state-machine-
    # only full scan (reassign_stale_tasks); this sweep reuses that SAME
    # function so streams' recovery does not depend on message survival.
    # Same shared-state timestamp-key throttle pattern as
    # _EXOMINER_HAS_WORK_KEY / _FPP_BACKFILL_STARTED_AT_KEY (server.py) —
    # here a `set_if_absent(..., ttl=interval)` gate: cross-process safe
    # (one server process wins the sweep for the whole fleet; the rest
    # skip until the TTL expires), so the amortised cost stays a single
    # full 'assigned' scan per interval regardless of fleet size or poll
    # rate, matching legacy's per-empty-poll cost only far less often
    # (the stream's own PEL/XCLAIM cycle already covers the common orphan
    # cases — dead worker, alive-but-moved-on — far more cheaply and far
    # more frequently; this sweep is strictly the backstop for messages
    # that no longer exist for that cycle to act on).
    _PRIMARY_SWEEP_KEY = "primary:sweep_gate"
    _PRIMARY_SWEEP_INTERVAL_SECONDS_DEFAULT = 300.0

    def _primary_sweep_interval_seconds():
        """EXONIT_PRIMARY_SWEEP_INTERVAL_SECONDS, default 300s (5 min).

        Parsed fresh on every call (cheap: one os.environ.get + float()) so
        tests can monkeypatch it to a sub-second value without an app
        rebuild — same convention as the stream visibility_timeout_s knobs.
        """
        try:
            v = float(os.environ.get(
                "EXONIT_PRIMARY_SWEEP_INTERVAL_SECONDS",
                str(_PRIMARY_SWEEP_INTERVAL_SECONDS_DEFAULT)))
            if v <= 0:
                raise ValueError
            return v
        except (TypeError, ValueError):
            return _PRIMARY_SWEEP_INTERVAL_SECONDS_DEFAULT

    def _primary_sweep_stale_assignments(ws):
        """G1 (review package, 2026-07-05): low-frequency defense-in-depth
        sweep, run from the empty-poll path only (same call site legacy
        used for reassign_stale_tasks — see the module comment above for
        the two permanent-stuck cases this closes).

        Throttled to at most once per _primary_sweep_interval_seconds()
        via a shared-state set_if_absent gate: the first caller across the
        whole fleet within the window wins (returns True) and runs the
        sweep; every other caller within that window sees the key already
        set (returns False) and returns immediately — zero extra cost on
        top of the gate check itself.

        Calls the EXISTING db.reassign_stale_tasks with collect_ids=True
        (same args as the legacy call site just below in worker_get_task —
        reusing, not reimplementing, its 5-branch logic) so this function
        learns exactly WHICH (task_id, tic_id) rows it just flipped
        'assigned' -> 'pending', and enqueues a FRESH stream message for
        each one. The SET-dedup inside ws.enqueue (SADD gate) makes a race
        against a legitimate concurrent producer harmless: at most one of
        the two enqueue calls for the same task_id survives, and either
        survivor carries the same task_id/tic_id payload.
        """
        interval = _primary_sweep_interval_seconds()
        if not get_shared_state().set_if_absent(
                _PRIMARY_SWEEP_KEY, "1", ttl=interval):
            return
        now = _time.time()
        alive = {
            wid for wid, w in workers_state.items()
            if now - w.get("last_heartbeat", 0) < WORKER_ALIVE_WINDOW_SECONDS
        }
        worker_current_tic = {
            wid: workers_state[wid].get("current_tic") for wid in alive
        }
        try:
            reset_rows = db.reassign_stale_tasks(
                STALE_TASK_TIMEOUT_SECONDS, alive_worker_ids=alive,
                worker_current_tic=worker_current_tic,
                abandon_timeout_seconds=ABSOLUTE_ABANDON_TIMEOUT_SECONDS,
                collect_ids=True)
        except Exception:
            logger.warning("primary sweep: reassign_stale_tasks failed",
                          exc_info=True)
            return
        for task_id, tic_id in reset_rows:
            try:
                ws.enqueue(str(task_id), {"task_id": task_id, "tic_id": tic_id})
            except Exception:
                logger.warning(
                    "primary sweep: re-enqueue failed for task_id=%s "
                    "tic=%s", task_id, tic_id, exc_info=True)

    def _reconcile_primary_dlq(ws):
        """Fase 1 (streams) DLQ reconciliation for the "primary" stream —
        the ONE place that replaces reassign_stale_tasks's role for tasks
        the stream can no longer track (see the module-level design note
        above this function's call site / the dispatch prompt for the full
        rationale): an entry that hit max_deliveries is moved to the DLQ
        and its message removed — no future claim() will ever touch that
        task_id again, so its PG row (left 'assigned' by the LAST
        get_task_row_and_assign UPDATE) would stay assigned forever
        without this. Runs on the empty-poll path only (mirrors legacy:
        reassign_stale_tasks is likewise only invoked when
        get_and_assign_task finds nothing).

        Honest correction (G4, review package 2026-07-05) of an earlier
        overstatement: db.reconcile_dlq_task's 'pending' (attempts < max)
        branch reads as symmetric with its 'failed' branch, but under this
        stream's DEFAULT settings (STAGE_SETTINGS["primary"].max_deliveries
        == db.MAX_TASK_ATTEMPTS == 3) it is structurally unreachable in
        production: every successful claim of this task (fresh or a
        legitimate steal) runs get_task_row_and_assign, which bumps PG
        `attempts` in lockstep with the stream's own delivery count — so a
        message that has been delivered `max_deliveries` times has, by
        construction, also bumped `attempts` to `max_deliveries` (==
        MAX_TASK_ATTEMPTS by default). The 'failed' branch is what fires
        in practice; 'pending' is only reachable if an operator tunes
        STAGE_SETTINGS so max_deliveries < MAX_TASK_ATTEMPTS, or in tests
        that inflate stream deliveries directly via ws.claim() without a
        matching get_task_row_and_assign call (see
        tests/test_primary_stream_handler.py's DLQ suite).

        G4 (review package, 2026-07-05): entries this call ACTUALLY
        resolves (final is 'pending' or 'failed' — a real state
        transition just happened) are pruned from the DLQ afterwards
        (ws.remove_dlq) so the list shrinks as rows are reconciled instead
        of growing without bound. An entry whose outcome is None (no
        transition — most commonly reconcile_dlq_task's own TOCTOU guard
        catching a legitimate late completion race, or a row that no
        longer exists) is deliberately left in place: pruning on a no-op
        would risk silently hiding a genuinely stuck/missing row from a
        future scan, and the peek-based design (never popping) already
        makes indefinite re-scanning of an unresolved entry safe and
        idempotent.
        """
        def _dlq_entry_tic_id(entry, task_id):
            """Best-effort tic_id for a DLQ entry: the stream payload first
            (no DB round-trip), falling back to a task_id lookup. Shared by
            both the 'pending' re-enqueue branch and the 'failed' done-state
            dual-write branch below."""
            tic_id = None
            try:
                payload = json.loads(entry.get("p") or "{}")
                tic_id = payload.get("tic_id")
            except Exception:
                tic_id = None
            if tic_id is None:
                row = db.get_task(task_id)
                tic_id = row["tic_id"] if row else None
            return tic_id

        try:
            entries = ws.peek_dlq()
        except Exception:
            return
        for entry in entries:
            task_id = entry.get("k")
            if not task_id:
                continue
            try:
                final = db.reconcile_dlq_task(task_id)
                if final == "pending":
                    tic_id = _dlq_entry_tic_id(entry, task_id)
                    ws.enqueue(str(task_id), {"task_id": task_id, "tic_id": tic_id})
                elif final == "failed":
                    # done-state dual-write (§2.2 cami calent, review fix
                    # 2026-09-26): this is a SECOND, independent path that
                    # can transition a task to terminal 'failed' — the DLQ
                    # reconciliation for a primary-stream message that
                    # exhausted max_deliveries (see db.reconcile_dlq_task's
                    # docstring: under default tuning this 'failed' branch
                    # is the one that actually fires in practice). Without
                    # this, a TIC whose task fails via THIS path never gets
                    # tic_sector_state.attempts incremented and can never
                    # reach exhausted(t, S) — same "unbounded reintent" shape
                    # bug the 08-10 regression test guards against, via a
                    # different trigger. Mirrors the requeue_or_fail_task
                    # hook above (:~1295): only on the TERMINAL transition
                    # (final == 'failed', never 'pending' — an internal
                    # requeue is not a lot-level failure), same
                    # try/except + logger.warning — must never break DLQ
                    # reconciliation.
                    tic_id = _dlq_entry_tic_id(entry, task_id)
                    if tic_id:
                        try:
                            done_state.mark_tic_failed(
                                db.backend, tic_id, task_id,
                                error="dlq_exhausted_max_deliveries")
                        except Exception:
                            logger.warning(
                                "done_state.mark_tic_failed (DLQ) failed "
                                "for tic=%s task=%s (non-fatal)",
                                tic_id, task_id, exc_info=True)
                if final is not None:
                    try:
                        ws.remove_dlq(entry.get("msg_id"))
                    except Exception:
                        logger.warning(
                            "primary DLQ prune failed for task_id=%s",
                            task_id, exc_info=True)
            except Exception:
                logger.warning(
                    "primary DLQ reconciliation failed for task_id=%s",
                    task_id, exc_info=True)

    def _primary_task_from_stream(worker_id):
        """Fase 1 (streams): serve /api/worker/task from the "primary" work
        stream instead of get_and_assign_task's scan. Idle cost: one
        claim() call — zero SQL on nearly every poll — since neither the
        ORDER BY LIMIT 1 SELECT nor the legacy reassign_stale_tasks call
        site below is reached; the stream's own PEL/XCLAIM cycle
        (visibility_timeout_s=1800s, matching STALE_TASK_TIMEOUT_SECONDS)
        is the orphan mechanism instead, backstopped by two empty-poll-only
        mechanisms for messages the stream itself can no longer act on:
        _reconcile_primary_dlq (a message that hit max_deliveries) and
        _primary_sweep_stale_assignments (G1: a row that has NO message at
        all — pre-cutover legacy rows, or a steal/late-ack race that lost
        the message without exhausting deliveries). The sweep runs
        reassign_stale_tasks itself, but throttled to at most once per
        EXONIT_PRIMARY_SWEEP_INTERVAL_SECONDS (default 300s) fleet-wide —
        see its own docstring.

        Per claimed item this does exactly ONE UPDATE-guarded SELECT
        (db.get_task_row_and_assign) to re-check the task is still
        assignable ('pending' — a fresh dequeue — OR 'assigned' — a
        legitimate steal of a stale PEL entry, already vetted by this
        function's own keep_if) — MANDATORY correctness: a task can be
        completed/failed by a path that never touches the stream (a real
        result POST racing this exact steal). Terminal (completed/failed)
        or missing -> ack + next iteration, capped at
        _PRIMARY_STREAM_MAX_ITERATIONS.

        Unlike FPP/ExoMiner, there is no metadata-availability hold-back
        here: a primary task's only "eligibility" fact is its own status
        column, not a joined table, so every ack-vs-serve decision below is
        a clean permanent/success split — nothing transient to wait on.

        Also unlike FPP/ExoMiner: item_key IS the task_id (a TEXT PK), not
        an integer PK, so no int() cast anywhere on this path.
        """
        ws = work_streams.get_stream("primary")

        def _keep_if(item_key, consumer):
            """Veto a steal unless one of reassign_stale_tasks's own two
            reassign conditions holds: branch 1 (assigned so long ago the
            worker MUST be stuck, regardless of liveness/current_tic) or
            branch 4 (alive but not on this exact tic, or dead — handled by
            _worker_alive_and_on_current_tic below).

            Branch-1 parity (ABSOLUTE_ABANDON_TIMEOUT_SECONDS=3600s) is
            enforced HERE, not inside _worker_alive_and_on_current_tic:
            that helper only knows about workers_state (liveness/tic), not
            assigned_at, and this ceiling must OVERRIDE liveness/tic —
            without it, a worker whose heartbeat thread stays alive and
            keeps reporting the same current_tic while its BLS/numba
            compute thread is actually hung (a known real failure mode —
            heartbeat and compute run on separate threads) would veto
            every steal of its task FOREVER: the message never redelivers,
            max_deliveries never triggers, the DLQ never sees it, and the
            PG row stays 'assigned' eternally — exactly the stuck-forever
            case Task 5 exists to prevent.

            Missing/unparseable assigned_at falls through to the liveness
            check below (same tolerance _to_epoch's callers elsewhere give
            a bad timestamp) rather than skipping the row outright, since
            keep_if's only question is "may this be stolen?", not
            "should reassign_stale_tasks look at this row at all?".
            """
            owner_row = db._query_one(
                "SELECT tic_id, assigned_at FROM tasks WHERE task_id = ?",
                (item_key,))
            if owner_row is None:
                return False
            assigned_ts = _to_epoch(owner_row.get("assigned_at"))
            if (assigned_ts is not None
                    and _time.time() - assigned_ts > ABSOLUTE_ABANDON_TIMEOUT_SECONDS):
                return False   # branch 1: hard ceiling, steal unconditionally
            return _worker_alive_and_on_current_tic(
                workers_state, consumer, owner_row["tic_id"])

        def _empty():
            _reconcile_primary_dlq(ws)
            _primary_sweep_stale_assignments(ws)
            return {"task_id": None, "wait_seconds": 10}

        for _ in range(_PRIMARY_STREAM_MAX_ITERATIONS):
            item = ws.claim(worker_id, keep_if=_keep_if)
            if item is None:
                return _empty()
            task_id = item.payload.get("task_id", item.item_key)
            row = db.get_task_row_and_assign(task_id, worker_id)
            if row is None:
                ws.ack(item.item_key)      # gone or no longer pending
                continue
            return {"task_id": row["task_id"], "tic_id": row["tic_id"]}
        return _empty()

    @app.get("/api/worker/task")
    def worker_get_task(request: Request):
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        worker_id = request.headers.get("x-worker-id", "")
        if work_producers.streams_enabled():
            return _primary_task_from_stream(worker_id)
        task = db.get_and_assign_task(worker_id)
        if not task:
            # No pending tasks — only reassign tasks whose assigned worker
            # has gone silent. Alive workers are mid-processing; reassigning
            # from under them doubles the work and burns `attempts`.
            now = _time.time()
            alive = {
                wid for wid, w in workers_state.items()
                if now - w.get("last_heartbeat", 0) < WORKER_ALIVE_WINDOW_SECONDS
            }
            # Build a map of which TIC each alive worker is currently on.
            # This comes from workers_state (in-memory heartbeat data) so
            # no DB read is needed here.
            worker_current_tic = {
                wid: workers_state[wid].get("current_tic") for wid in alive
            }
            reassigned = db.reassign_stale_tasks(
                STALE_TASK_TIMEOUT_SECONDS, alive_worker_ids=alive,
                worker_current_tic=worker_current_tic,
                abandon_timeout_seconds=ABSOLUTE_ABANDON_TIMEOUT_SECONDS)
            if reassigned > 0:
                task = db.get_and_assign_task(worker_id)
        if not task:
            return {"task_id": None, "wait_seconds": 10}
        return {"task_id": task["task_id"], "tic_id": task["tic_id"]}

    @app.post("/api/worker/result")
    async def worker_result(request: Request):
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        task_id = body.get("task_id")
        worker_id = request.headers.get("x-worker-id") or body.get("worker_id")
        processing_time = body.get("processing_time_seconds", body.get("processing_time", 0))
        gpu_used = body.get("gpu_used", False)

        # Save star (always, even without candidates)
        tic_id = body.get("tic_id")
        candidates = body.get("candidates", [])
        cnn_scores = body.get("cnn_scores", [])
        reported_status = body.get("status", "completed")
        if tic_id:
            # Replace-when-deeper (deep-rescan spec §5): read the OLD star
            # BEFORE add_star refreshes it — comparing after the upsert would
            # compare the new value against itself and never fire.
            _old_star = db.get_star(tic_id)
            _new_n_sectors = body.get("n_sectors", 0) or 0
            _replace_deeper = (
                reported_status == "completed"
                and _old_star is not None
                and _new_n_sectors > (_old_star.get("n_sectors") or 0)
            )
            # §2.2 (done-state design, 2026-09-26, operator decision #2): do NOT
            # overwrite an already-completed star with a later failed/aborted
            # result. add_star is REPLACE semantics (db.py's upsert()) — a
            # failure here used to blank ra/dec/n_sectors/n_datapoints on a
            # perfectly good star (confirmed on prod: 135,172/135,172 failed
            # stars rows have n_sectors=0, including previously-completed
            # ones), which also knocked the TIC out of
            # build_deep_pass_worklist.py's n_sectors-based selection. done(t)
            # (tic_sector_state) is unaffected either way — it is monotonic by
            # construction (mark_tic_failed's CASE guard) — this only protects
            # the `stars` row's descriptive fields.
            _skip_star_overwrite = (
                reported_status in ("failed", "aborted")
                and _old_star is not None
                and _old_star.get("status") == "completed"
            )
            if not _skip_star_overwrite:
                db.add_star(
                    tic_id,
                    ra=body.get("ra"),
                    dec=body.get("dec"),
                    tmag=body.get("tmag"),
                    n_sectors=body.get("n_sectors", 0),
                    n_datapoints=body.get("n_points", body.get("n_datapoints", 0)),
                    time_span_days=body.get("time_span_days", 0),
                    status=reported_status,
                )
            if _replace_deeper:
                _n_wiped = db.delete_candidates_for_tic(tic_id)
                if _n_wiped:
                    logger.info(
                        "replace-when-deeper TIC %s: %d old candidate(s) "
                        "wiped (n_sectors %s -> %d); new result supersedes",
                        tic_id, _n_wiped,
                        _old_star.get("n_sectors"), _new_n_sectors)
            if reported_status == "completed":
                _run_sector = _active_run_sector()  # tag discovery inserts with the run's sector
                # ExoMiner producer hook (Fase 1, Task 4) needs the star's
                # CURRENT has_spoc value, which nothing else in this branch
                # queries (the SPOC-gate, when enabled, resolves it
                # asynchronously further below via the executor submit —
                # see db.set_star_has_spoc's own hook for that transition).
                # Fetched ONCE per star (not per candidate) and skipped
                # entirely when the streams flag is off.
                _em_has_spoc = None
                if work_producers.streams_enabled():
                    _star_for_em = db.get_star(tic_id)
                    _em_has_spoc = _star_for_em.get("has_spoc") if _star_for_em else None
                for i, c in enumerate(candidates):
                    cnn = cnn_scores[i] if i < len(cnn_scores) else None
                    fpp = c.get("triceratops_fpp")
                    nfpp = c.get("triceratops_nfpp", c.get("nfpp"))
                    candidate_id = db.add_candidate(
                        tic_id,
                        candidate_num=c.get("candidate_num", i + 1),
                        period_days=c.get("period_days", 0),
                        t0=c.get("t0", 0),
                        duration_days=c.get("duration_days", 0),
                        depth_ppm=c.get("depth_ppm", 0),
                        snr=c.get("snr", 0),
                        cnn_score=cnn,
                        triceratops_fpp=fpp,
                        triceratops_nfpp=nfpp,
                        triceratops_prob_planet=c.get("triceratops_prob_planet"),
                        triceratops_verdict=c.get("triceratops_verdict"),
                        exominer_ffi_score=c.get("exominer_ffi_score"),
                        exominer_ffi_model_version=c.get("exominer_ffi_model_version"),
                        # v8 residue removed (minor dedup 11(d),
                        # docs/reviews/2026-07-02-opus48-cleanup-review.md):
                        # db.add_candidate's write path for
                        # exominer_ffi_score_v8 was already dropped by the
                        # db-layer change; the accept-but-ignore kwarg is
                        # gone from this call too now.
                        cross_source_name=c.get("cross_source_name"),
                        cross_source_snr=c.get("cross_source_snr"),
                        cross_source_depth_ppm=c.get("cross_source_depth_ppm"),
                        cross_source_depth_ratio=c.get("cross_source_depth_ratio"),
                        cross_source_agreement=(
                            None if c.get("cross_source_agreement") is None
                            else int(bool(c.get("cross_source_agreement")))),
                        sector=_run_sector,
                    )
                    # FPP producer hook (Fase 1, Task 2): best-effort mirror
                    # of the legacy fpp_task eligibility (workers.py:805-824,
                    # see exonit.work_producers) for the row just written.
                    # Placed at the routes layer (not inside db.add_candidate)
                    # because star_status is free here — this whole branch
                    # only runs under `if reported_status == "completed":` —
                    # whereas db.add_candidate has no star context and would
                    # need an extra query per candidate to get it. Approximate
                    # (uses the incoming payload, not necessarily what
                    # COALESCE ultimately persisted on a reprocess) but
                    # harmless: Task 3's handler re-checks eligibility
                    # against PG before serving the candidate, and the
                    # rebuild script is the reconciliation path regardless.
                    try:
                        if work_producers.streams_enabled():
                            work_producers.maybe_enqueue_fpp({
                                "candidate_id": candidate_id,
                                "tic_id": tic_id,
                                "snr": c.get("snr", 0),
                                "cnn_score": cnn,
                                # OR-gate (verdict_eligible): em_ffi-rescued
                                # candidates (cnn<0.7) must still reach the
                                # FPP queue when em_ffi>=0.5.
                                "exominer_ffi_score": c.get("exominer_ffi_score"),
                                "triceratops_fpp": fpp,
                                "nfpp": nfpp,
                                "star_status": "completed",
                            })
                    except Exception:
                        logger.warning(
                            "FPP producer hook failed for candidate_id=%s "
                            "tic=%s", candidate_id, tic_id, exc_info=True)
                    # ExoMiner producer hook (Fase 1, Task 4): best-effort
                    # mirror of the legacy exominer_task eligibility
                    # (has_spoc=1 AND cnn_score>=0.7 AND exominer_score IS
                    # NULL) for the row just written. Almost always a
                    # no-op on a star's FIRST completion (has_spoc is still
                    # NULL until the SPOC-gate background thread resolves
                    # it — see db.set_star_has_spoc's own hook for that
                    # path); this one exists for the REPROCESSED-star case:
                    # new candidates added to a TIC whose has_spoc was
                    # already resolved to 1 by an earlier pass.
                    try:
                        if work_producers.streams_enabled():
                            work_producers.maybe_enqueue_exominer({
                                "candidate_id": candidate_id,
                                "tic_id": tic_id,
                                "cnn_score": cnn,
                                "exominer_score": c.get("exominer_score"),
                                "has_spoc": _em_has_spoc,
                                "star_status": "completed",
                            })
                    except Exception:
                        logger.warning(
                            "ExoMiner producer hook failed for "
                            "candidate_id=%s tic=%s", candidate_id, tic_id,
                            exc_info=True)
                    # Persist ExoMiner scores if the worker ran inference.
                    if c.get("exominer_score") is not None:
                        db.update_candidate_exominer(
                            candidate_id,
                            score=c.get("exominer_score"),
                            tce_period=c.get("exominer_tce_period"),
                            sector_run=c.get("exominer_sector_run"),
                        )

        # Save completeness (injection-recovery results) — the ~72-cell grid is
        # written in ONE transaction (add_completeness_many) instead of ~72
        # separate commits, which was the dominant write-amplification per
        # result (the main WAL-churn / lock-contention source during a run).
        completeness = body.get("completeness")
        if completeness and tic_id and reported_status == "completed":
            cells = [
                (cell["radius_earth"], cell["period_days"], cell["detection_prob"])
                for cell in completeness.get("details", [])
            ]
            # Fase 4a Task 4 follow-up (d): completeness is auxiliary — a
            # write-path failure here (e.g. the PG staging table/connection
            # down under EXONIT_COMPLETENESS_PG=1) must not fail the result
            # write or skip complete_task below. Same non-fatal
            # try/except-then-log.warning pattern as the neighboring
            # FPP/ExoMiner producer hooks above.
            try:
                db.add_completeness_many(tic_id, cells)
            except Exception:
                logger.warning(
                    "add_completeness_many failed for tic=%s (non-fatal)",
                    tic_id, exc_info=True)

        # Complete or requeue/fail task based on reported status
        if task_id:
            if reported_status in ("failed", "aborted"):
                error_reason = body.get("error") or reported_status
                final = db.requeue_or_fail_task(task_id, error=error_reason)
                logger.warning(
                    "worker=%s tic=%s task=%s failed (%s) -> %s",
                    worker_id, tic_id, task_id, error_reason, final)
                # done-state dual-write (§2.2 cami calent): ONLY a TERMINAL
                # failure (final == 'failed', worker-attempts exhausted)
                # counts toward exhausted(t, sector) — an internal requeue
                # ('pending') is not a lot-level failure. Best-effort: must
                # never break the result path (spec §5 risk #1's bucle 08-10
                # invers guard depends on this write, but a write failure here
                # should not fail the whole /api/worker/result call).
                if final == "failed" and tic_id:
                    try:
                        done_state.mark_tic_failed(
                            db.backend, tic_id, task_id, error=error_reason)
                    except Exception:
                        logger.warning(
                            "done_state.mark_tic_failed failed for "
                            "tic=%s task=%s (non-fatal)",
                            tic_id, task_id, exc_info=True)
                # Fase 1 (streams): release the stream's PEL entry regardless
                # of outcome (mirrors worker_fpp_result/worker_exominer_result's
                # unconditional ack); when PG says the task lives on
                # ('pending'), enqueue a FRESH message so it is servable again
                # — PG's `attempts` (already checked by requeue_or_fail_task
                # above) is the retry cap of record, not the stream's own
                # max_deliveries, so resetting the delivery count here is
                # correct, not a bypass.
                if work_producers.streams_enabled():
                    try:
                        _ws = work_streams.get_stream("primary")
                        _ws.ack(str(task_id))
                        if final == "pending":
                            _ws.enqueue(str(task_id),
                                       {"task_id": task_id, "tic_id": tic_id})
                    except Exception:
                        logger.warning(
                            "primary stream ack/requeue failed for task_id=%s "
                            "tic=%s", task_id, tic_id, exc_info=True)
            else:
                db.complete_task(
                    task_id, processing_time, gpu_used=gpu_used,
                    lc_load_recovered=bool(body.get("lc_load_recovered", False)))
                # done-state dual-write (§2.2 cami calent): see the failed
                # branch's comment above for the try/except rationale.
                if tic_id:
                    try:
                        done_state.mark_tic_done(db.backend, tic_id, task_id)
                    except Exception:
                        logger.warning(
                            "done_state.mark_tic_done failed for tic=%s "
                            "task=%s (non-fatal)",
                            tic_id, task_id, exc_info=True)
                if work_producers.streams_enabled():
                    try:
                        work_streams.get_stream("primary").ack(str(task_id))
                    except Exception:
                        logger.warning(
                            "primary stream ack failed for task_id=%s tic=%s",
                            task_id, tic_id, exc_info=True)

        # Store telemetry
        timings = body.get("timings")
        if timings and tic_id:
            worker_name = ""
            if worker_id and worker_id in workers_state:
                worker_name = workers_state[worker_id].get("name", "")
            _telemetry_append({
                "tic_id": tic_id,
                "worker": worker_name,
                **timings,
            })

        # Update worker stats — only bump counters on a real completion.
        # B6 (2026-07-02): ws.increment() is a single atomic backend RMW
        # (Lua script on Redis) instead of `ws["k"] = ws.get("k", 0) + delta`,
        # which was two SEPARATE round trips (a read, then a write) — racy
        # even though each individual read/write was itself atomic, since a
        # concurrent result for the same worker could land between them and
        # get its own increment clobbered.
        if worker_id and worker_id in workers_state:
            ws = workers_state[worker_id]
            if reported_status == "completed":
                new_completed = ws.increment("tasks_completed", 1)
                new_total_time = ws.increment("total_processing_time", processing_time)
                db.update_worker_stats(
                    worker_id,
                    tasks_completed=new_completed,
                    total_processing_time=new_total_time)
            else:
                new_failed = ws.increment("tasks_failed", 1)
                db.update_worker_stats(
                    worker_id, tasks_failed=new_failed)
            ws["status"] = "idle"

        # Run verdict classifier in background for strong candidates.
        # OR-gate (2026-07-11, docs/reviews/injection_e2e_results_2026-07-11.md
        # + docs/reviews/ffi_cnn_investigation_2026-07-11.md §5):
        # snr>=10 AND (cnn>=0.7 OR exominer_ffi_score>=0.5) — strictly more
        # permissive than the old cnn>=0.7-only gate. See
        # work_producers.verdict_eligible for the shared predicate + NULL
        # handling.
        if _verdict_enabled and tic_id:
            for i, c in enumerate(candidates):
                cnn = cnn_scores[i] if i < len(cnn_scores) else None
                snr = c.get("snr", 0)
                if work_producers.verdict_eligible(
                        snr, cnn, c.get("exominer_ffi_score")):
                    _get_verdict_executor().submit(
                        _run_verdict_background, db, tic_id, c, cnn)

        # ExoMiner: opt-in via EXONIT_ENABLE_EXOMINER_AUTO=1 (off by default
        # while Docker is only available at the central server).
        # Uses bounded executor (not a raw Thread) so thread count stays
        # capped at EXONIT_EXOMINER_MAX_CONCURRENT regardless of burst rate.
        if (tic_id and candidates and reported_status == "completed"
                and os.environ.get("EXONIT_ENABLE_EXOMINER_AUTO") == "1"):
            _get_exominer_executor().submit(
                _run_exominer_background, db, tic_id, candidates)

        # SPOC-gate: opt-in via EXONIT_SPOC_GATE=1 (off by default).
        # For discovery runs where most TICs are FFI-only (TGLC), this avoids
        # ~160k DV-fetch round-trips by pre-marking FFI-only candidates with
        # exominer_score=-1 directly, without touching the existing
        # worker_exominer_task / exominer_pending queries.
        # Uses bounded executor (not a raw Thread) — pool IS the concurrency
        # limit (no DistributedSemaphore inside the worker fn).
        if (tic_id and candidates and reported_status == "completed"
                and os.environ.get("EXONIT_SPOC_GATE") == "1"):
            _get_spoc_executor().submit(
                _server._run_spoc_check_background, db, tic_id, candidates)

        # FPP-gate (eager): opt-in via EXONIT_FPP_GATE=1 (off by default).
        # Mirror of the SPOC-gate above but 100% local (no MAST round trip):
        # pre-mark FPP-uncomputable candidates at TIC completion so the FPP
        # backfill never serves them to a worker (which would return fpp=-1 after
        # a wasted FITS download + detrend). Same target_mass_is_uncomputable()
        # the lazy gate in worker_fpp_task uses → bit-identical decision. The lazy
        # gate stays as a backstop for the pre-existing backlog and any TIC whose
        # nearby_stars cache was not yet populated at completion time.
        if (tic_id and candidates and reported_status == "completed"
                and os.environ.get("EXONIT_FPP_GATE") == "1"):
            _get_fpp_gate_executor().submit(
                _run_fpp_gate_background, db, tic_id,
                _server._get_shared_metadata_cache(cfg), berger)

        return {"status": "ok"}

    @app.post("/api/candidate/update_analysis")
    async def candidate_update_analysis(request: Request):
        """Batch-update claude_verdict + claude_analysis for a list of candidates.

        This endpoint is the authoritative write path for explanation backfill
        scripts (e.g. generate_missing_explanations.py).  All incoming rows are
        written in a single SQLite transaction via
        ``db.update_candidate_claude_batch``, which respects the [lora-v2]
        preservation logic already implemented there.

        Request body::

            {
                "updates": [
                    {"candidate_id": 42, "verdict": "planet_candidate",
                     "analysis": "[lora-v2] Nice transit"},
                    {"candidate_id": 99, "analysis": "..."},
                    ...
                ]
            }

        The ``"verdict"`` key is **optional**.  When absent, the existing
        ``claude_verdict`` in the DB is left untouched.  When present (even
        as ``null``), it is written.  This lets callers that only regenerate
        the analysis (e.g. LoRA backfill for rows that already have a verdict)
        avoid accidentally clobbering ``claude_verdict``.

        Returns ``{"updated": N}`` where *N* is the number of rows processed.
        """
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            body = await request.json()
        except Exception as exc:
            return JSONResponse(
                {"error": "bad_request", "detail": str(exc)}, status_code=400)
        updates = body.get("updates")
        if not isinstance(updates, list):
            return JSONResponse(
                {"error": "bad_request",
                 "detail": "'updates' must be a list"},
                status_code=400)
        # Validate and normalise each entry.
        clean = []
        for entry in updates:
            if not isinstance(entry, dict) or "candidate_id" not in entry:
                return JSONResponse(
                    {"error": "bad_request",
                     "detail": "each entry must have 'candidate_id'"},
                    status_code=400)
            try:
                cid = int(entry["candidate_id"])
            except (TypeError, ValueError):
                return JSONResponse(
                    {"error": "bad_request",
                     "detail": f"invalid candidate_id: {entry['candidate_id']!r}"},
                    status_code=400)
            row = {"candidate_id": cid, "analysis": entry.get("analysis")}
            # Preserve the absence/presence of "verdict" so that
            # update_candidate_claude_batch can distinguish "do not touch
            # claude_verdict" (key absent) from "write this verdict" (key
            # present, even if None).  See db.update_candidate_claude_batch
            # docstring for the full convention.
            if "verdict" in entry:
                row["verdict"] = entry["verdict"]
            clean.append(row)
        n = db.update_candidate_claude_batch(clean)
        return {"updated": n}

    @app.post("/api/worker/enqueue")
    async def worker_enqueue(request: Request):
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        tic_id = body.get("tic_id")
        sector = body.get("sector")
        task_id = secrets.token_hex(8)
        db.add_task(task_id, tic_id, sector=sector)
        # (task_data_store removed 2026-06-09: no client ever sent time/flux —
        # enqueue.py posts only tic_id — so it always stored four Nones, and
        # being per-process it was broken under --workers N anyway.)
        # Primary work-stream producer hook (Fase 1, Task 5): this endpoint
        # is a live fallback path (exonit.enqueue calls it only when the
        # server predates /enqueue_bulk) that inserts tasks OUTSIDE
        # enqueue_bulk; without this hook a task inserted here would sit
        # 'pending' forever once EXONIT_WORK_STREAMS=1, since
        # worker_get_task's flag-on path no longer scans PG at all.
        try:
            if work_producers.streams_enabled():
                work_producers.maybe_enqueue_primary(
                    {"task_id": task_id, "tic_id": tic_id, "status": "pending"})
        except Exception:
            logger.warning(
                "primary producer hook failed for task_id=%s tic=%s",
                task_id, tic_id, exc_info=True)
        return {"task_id": task_id, "status": "enqueued"}

    @app.post("/api/worker/enqueue_bulk")
    async def worker_enqueue_bulk(request: Request):
        """Bulk-enqueue a list of targets in a single SQL transaction.

        Accepts ``{"targets": [{"tic_id": int|str, "label": "candidate"}, ...]}``.
        Uses the same auth token as ``/api/worker/enqueue``.

        Skips TICs that already have a task row (any status), so repeated
        calls with the same payload are idempotent.  All inserts are wrapped
        in a single ``executemany`` transaction for O(1) round-trips regardless
        of batch size — avoids the per-TIC HTTP overhead that caused timeouts
        at ~1.5k TICs with the single-row endpoint.

        With ``"force_rescan": true`` in the body (deep-rescan spec
        2026-07-23), the dedup only considers LIVE task rows (pending or
        assigned) — a TIC whose tasks are all completed/failed is
        re-enqueued for a fresh pass, while a TIC still being worked is
        never duplicated. Absent the flag, behaviour is unchanged.

        With ``"sector": S`` in the body, every inserted row is tagged
        ``tasks.sector = S`` (done-state design, 2026-09-26 — see
        exonit/done_state.py). When ``EXONIT_DONE_STATE=1`` (default off:
        behaviour above is unchanged byte-for-byte) AND ``sector`` is given
        AND ``force_rescan`` is absent, the ``tasks``-row check switches from
        "ANY row blocks" to "only LIVE rows (pending/assigned) block" — the
        same predicate ``force_rescan`` already uses — because done(t)
        (completed in ANY sector) / exhausted(t, S) (failed twice for this
        sector — see done_state.MAX_LOTS_PER_SECTOR) now own the "already
        handled" decision instead, via ``done_state.blocked_for_enqueue``.
        With ``force_rescan``, only the active-task check applies (done/
        exhausted are not consulted at all —
        a deep rescan must be able to supersede either). If
        ``done_state.blocked_for_enqueue`` itself raises in this mode (e.g.
        ``tic_sector_state`` missing), the request FAILS CLOSED — a 503
        JSON error, no rows inserted — rather than silently falling back to
        the widened "only live rows block" check with no done(t)/
        exhausted(t,S) protection at all.

        Returns ``{"enqueued": N, "skipped": M, "failed": K, "adopted": A,
        "elapsed_seconds": F}``.

        ``"run_id": "..."`` in the body (S4, docs/superpowers/specs/2026-09-
        26-enqueue-s1-s4-design.md §2): identifies the LOT this call belongs
        to. ``adopted`` counts ``skipped`` TICs that were blocked by a LIVE
        task (pending/assigned) — only distinguishable from an "ANY row
        blocks" skip when the dedup already uses the live-rows predicate
        (``force_rescan`` or the done-state dedup below); otherwise
        ``adopted`` is always 0. With ``EXONIT_LOT_ADOPT_ACTIVE=1`` AND a
        ``run_id`` in the body AND ``tasks.run_id`` present (S3 DDL), those
        live tasks are additionally RE-TAGGED to this ``run_id`` (an UPDATE,
        see ``_adopt_live_tasks`` below) so a re-run after a dead/timed-out
        scan (S4's whole point) does not lose the tasks the dead run already
        created — ``adopted`` then reports the UPDATE's real rowcount
        instead of the TIC-count estimate. Flag OFF (default): the count is
        still computed and returned (harmless), but no UPDATE ever runs —
        see that flag's docstring at its ``os.environ.get`` call site below.

        Q4 (perf review 2026-09-26, docs/reviews/2026-09-26-enqueue-path-
        performance-review.md): the whole dedup+insert+producer-hook body
        below is synchronous DB/Redis work, run via ``asyncio.to_thread`` so
        it never blocks this (single, shared) uvicorn process's event loop —
        at campaign scale a chunk of this call measured ~0.7 s x 61 chunks
        while other workers were polling different endpoints on the SAME
        process. On PostgreSQL the insert itself is a single multi-row
        ``INSERT ... SELECT FROM unnest(...)`` (see ``_insert_tasks_bulk_pg``
        below) instead of ``executemany`` — one round trip regardless of
        batch size, and unnest's array params sidestep the 65535-parameter
        limit a plain multi-VALUES INSERT would hit past ~16k rows. SQLite
        keeps ``executemany`` unchanged (no array-bind support, and it was
        never the bottleneck the review measured).
        """
        t0_bulk = _time.monotonic()
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        try:
            body = await request.json()
        except Exception as exc:
            return JSONResponse(
                {"error": "bad_request", "detail": str(exc)}, status_code=400
            )

        targets = body.get("targets")
        if not isinstance(targets, list):
            return JSONResponse(
                {"error": "bad_request",
                 "detail": "'targets' must be a list"},
                status_code=400,
            )

        # Validate each entry has a tic_id
        rows_to_insert = []
        failed = 0
        for entry in targets:
            if not isinstance(entry, dict) or "tic_id" not in entry:
                failed += 1
                continue
            try:
                tic_id = str(int(entry["tic_id"]))
            except (TypeError, ValueError):
                tic_id = str(entry["tic_id"]).strip()
            if not tic_id:
                failed += 1
                continue
            rows_to_insert.append(tic_id)

        force_rescan = bool(body.get("force_rescan"))
        try:
            sector = int(body["sector"]) if body.get("sector") is not None else None
        except (TypeError, ValueError):
            sector = None
        # run_id (S4): a plain string identity, not parsed/validated beyond
        # "truthy" — the caller (exonit.enqueue._post_bulk) always sends the
        # same run_id discovery_run.py minted for this lot; an empty/absent
        # value degrades to "adopted = count only, no UPDATE" (see
        # _enqueue_bulk_blocking's docstring), never an error.
        run_id = body.get("run_id") or None

        result = await asyncio.to_thread(
            _enqueue_bulk_blocking, rows_to_insert, failed, force_rescan, sector,
            run_id)
        if "error" in result:
            return JSONResponse(
                {k: v for k, v in result.items() if k != "status_code"},
                status_code=result["status_code"])

        result["elapsed_seconds"] = round(_time.monotonic() - t0_bulk, 3)
        return result

    def _enqueue_bulk_blocking(rows_to_insert, failed, force_rescan, sector,
                               run_id=None) -> dict:
        """Synchronous body of ``worker_enqueue_bulk`` — see that route's
        docstring (Q4) for why this runs via ``asyncio.to_thread`` instead of
        inline on the event loop. Byte-for-byte the pre-Q4 logic otherwise;
        returns either ``{"enqueued", "skipped", "failed", "adopted"}`` or
        ``{"error", "detail", "status_code"}`` for the fail-closed
        done_state case (the caller turns the latter into a JSONResponse).

        run_id / adopted (S4): see the route's own docstring for the full
        contract — summary: ``adopted`` counts TICs skipped because a LIVE
        task already exists for them (only meaningful when the dedup below
        already uses the live-rows predicate), and — with
        ``EXONIT_LOT_ADOPT_ACTIVE=1`` and a ``run_id`` — those live tasks are
        re-tagged to *run_id* via ``_adopt_live_tasks``.
        """
        # Find which TICs already have a task row — skip them.
        # Routed through the backend API (db.backend) so it works on both SQLite
        # and PG: db.conn is the completeness SQLite connection in PG mode and
        # db._lock does not exist there, so the old direct-conn path would break.
        # The IN (...) list is chunked: a single huge IN can exceed the per-
        # statement parameter limit (SQLite ~32766, PG 65535) on big batches.
        #
        # Dedup: without force_rescan, ANY existing task row skips the TIC
        # (idempotent re-posts). With force_rescan (deep-rescan spec
        # 2026-07-23), only LIVE rows (pending/assigned) skip — a completed
        # or failed TIC gets a fresh task so the deeper scan can supersede
        # its old analysis; live work is never duplicated, which also makes
        # repeated force calls idempotent (the fresh pending row blocks the
        # second call).
        #
        # done-state design (2026-09-26, EXONIT_DONE_STATE=1, sector given,
        # not force_rescan): the ``tasks``-table "ANY row blocks" rule above
        # is REPLACED by "only LIVE rows block" — same predicate force_rescan
        # already uses — because done(t)/exhausted(t, sector) (via
        # done_state.blocked_for_enqueue below) now own the "already handled"
        # decision instead. Without this, a terminal 'failed' tasks row would
        # still block here regardless of attempts/exhausted(t,S), defeating
        # the whole point of the (b) requirement (a single failure must stay
        # selectable). Flag off (default), sector None, or force_rescan:
        # byte-for-byte the pre-existing behaviour.
        _done_state_dedup = (
            done_state.is_enabled() and not force_rescan and sector is not None
        )
        # S4 §2.1: whether the tasks-table dedup query below uses the LIVE-
        # rows predicate — only then does a match mean "blocked by a task
        # that could be ADOPTED" rather than just "blocked by SOME row"
        # (completed/failed included). Kept as its own flag rather than
        # re-deriving it from dedup_sql_tail's truthiness at the bottom of
        # this function, purely for readability at the two call sites below.
        _live_rows_dedup = bool(force_rescan or _done_state_dedup)
        dedup_sql_tail = (
            " AND status IN ('pending', 'assigned')"
            if _live_rows_dedup else ""
        )
        existing_tics: set = set()
        live_tic_matches: set = set()
        if rows_to_insert:
            _CHUNK = 10000
            for i in range(0, len(rows_to_insert), _CHUNK):
                chunk = rows_to_insert[i:i + _CHUNK]
                placeholders = ",".join("?" * len(chunk))
                rows = db.backend.query(
                    f"SELECT DISTINCT tic_id FROM tasks WHERE tic_id IN ({placeholders})"
                    f"{dedup_sql_tail}",
                    chunk,
                )
                matched = {r["tic_id"] for r in rows}
                existing_tics.update(matched)
                if _live_rows_dedup:
                    live_tic_matches.update(matched)
            if _done_state_dedup:
                try:
                    existing_tics.update(
                        done_state.blocked_for_enqueue(
                            db.backend, sector, rows_to_insert))
                except Exception:
                    # Fail CLOSED (review fix, 2026-09-26): the tasks-table
                    # check above was already widened from "any row blocks"
                    # to "only live rows block" on the assumption that
                    # blocked_for_enqueue would supply the done(t)/
                    # exhausted(t,S) half of the dedup. If it raises (e.g.
                    # tic_sector_state missing — the flag was flipped on
                    # before the DDL, an operator-error the deploy runbook
                    # forbids but env-var typos happen), silently proceeding
                    # would insert fresh tasks for already-completed/
                    # exhausted TICs with NO dedup protection at all for
                    # this batch. Reject the whole request instead — no
                    # inserts happen below this point yet, so nothing to
                    # roll back.
                    logger.error(
                        "done_state.blocked_for_enqueue failed for sector=%s "
                        "(%d tics) — rejecting this enqueue_bulk batch "
                        "(fail-closed) instead of silently bypassing the "
                        "done(t)/exhausted(t,S) dedup", sector,
                        len(rows_to_insert), exc_info=True)
                    return {
                        "error": "done_state_unavailable",
                        "detail": (
                            "done_state.blocked_for_enqueue failed for "
                            f"sector={sector}; rejecting rather than "
                            "risking duplicate/already-done TICs"),
                        "status_code": 503,
                    }

        to_insert = [t for t in rows_to_insert if t not in existing_tics]
        skipped = len(rows_to_insert) - len(to_insert)

        # Insert all new rows in one bulk write: unnest() on PG (Q4, one
        # round trip regardless of batch size — see _insert_tasks_bulk_pg),
        # executemany on SQLite (one transaction/commit, unchanged). The
        # backend translates the ? placeholders for PG.
        enqueued = 0
        insert_failed = 0
        task_rows = []
        if to_insert:
            task_rows = [(secrets.token_hex(8), tic_id, 2) for tic_id in to_insert]
            try:
                # sector/run_id are ADDITIVE columns (done-state/S3 designs):
                # only referenced in the INSERT when the caller passed one,
                # same "old callers keep the exact pre-existing SQL" contract
                # as db.add_task above. Review fix (2026-09-26): even when the
                # caller DOES pass one, _insert_tasks_bulk_pg/_sqlite silently
                # drop it (column_exists guard) if the column genuinely isn't
                # there yet — so this INSERT is safe regardless of deploy
                # order (code before DDL, e.g. discovery_run.py (S4) already
                # mints/forwards run_id unconditionally on every real lot,
                # which is exactly what the spec's own §6 window table
                # schedules to happen before S3's DDL lands).

                if db.backend.dialect.name == "postgres":
                    _insert_tasks_bulk_pg(db.backend, task_rows, sector, run_id)
                else:
                    _insert_tasks_bulk_sqlite(db.backend, task_rows, sector, run_id)
                enqueued = len(to_insert)
            except Exception as exc:
                logger.error("enqueue_bulk insert failed: %s", exc)
                insert_failed = len(to_insert)
                task_rows = []   # nothing was actually inserted — do not enqueue

            # Primary work-stream producer hook (Fase 1, Task 5): task_id is
            # generated in Python (secrets.token_hex) BEFORE the INSERT — not
            # a DB-assigned id — so task_rows already holds every inserted
            # row's real primary key in insertion order; no RETURNING /
            # post-insert SELECT is needed to recover them. Preserves the
            # legacy "priority ASC, created_at ASC" service order BY
            # CONSTRUCTION: every bulk-enqueued row here shares priority=2
            # (the only value any writer in this repo ever uses), so
            # ordering collapses to created_at ASC among this batch;
            # enqueuing onto the stream in the SAME order these rows were
            # inserted (one write call) reproduces that tie-break exactly —
            # unnest() preserves array-index order in its output rows, same
            # as executemany's row order. (A future call site that ever
            # wrote a different priority would need a priority-aware
            # rebuild — see scripts/rebuild_work_streams.py's
            # _derive_primary, the reconciliation path for exactly that kind
            # of drift.)
            #
            # G3 (review package, 2026-07-05): ONE batched
            # maybe_enqueue_primary_many() call, not one maybe_enqueue_primary()
            # round trip per row in a Python for-loop — at this project's
            # 50k-100k-task discovery-run enqueue scale the per-row loop was a
            # synchronous SADD+XADD Redis round trip per task inline in this
            # async endpoint, able to block the event loop for tens of
            # seconds. A single try/except around the whole batch (not
            # per-row) matches every other best-effort producer-hook call
            # site in this file (e.g. worker_enqueue's singular hook below).
            if work_producers.streams_enabled():
                try:
                    work_producers.maybe_enqueue_primary_many(
                        [{"task_id": _task_id, "tic_id": _tic_id,
                          "status": "pending"}
                         for _task_id, _tic_id, _prio in task_rows])
                except Exception:
                    logger.warning(
                        "primary producer hook (batch) failed for %d "
                        "task rows", len(task_rows), exc_info=True)

        # S4 §2.1/§2.2: adopted defaults to the TIC-count estimate (0 unless
        # the dedup used the live-rows predicate — see _live_rows_dedup);
        # EXONIT_LOT_ADOPT_ACTIVE=1 + a run_id upgrades it to the real
        # UPDATE rowcount via _adopt_live_tasks, which also RE-TAGS those
        # live tasks to this run_id. Flag OFF (default): the count above is
        # still returned (harmless — the caller decides whether to act on
        # it; see discovery_run.py's EXONIT_LOT_ADOPT_ACTIVE gate), but no
        # UPDATE ever runs, so tasks.run_id is never touched by this
        # request. A -1 sentinel from _adopt_live_tasks (UPDATE itself
        # failed, e.g. column missing) keeps the provisional estimate rather
        # than reporting a bogus negative count.
        adopted = len(live_tic_matches)
        if (os.environ.get("EXONIT_LOT_ADOPT_ACTIVE", "0") == "1"
                and run_id and live_tic_matches):
            updated = _adopt_live_tasks(db.backend, list(live_tic_matches), run_id)
            if updated >= 0:
                adopted = updated

        return {
            "enqueued": enqueued,
            "skipped": skipped,
            "failed": failed + insert_failed,
            "adopted": adopted,
        }

    # --- FPP backfill tasks for workers ---

    _FPP_STREAM_MAX_ITERATIONS = 5  # bound one HTTP request's work, even if
                                    # a run of stale/ineligible entries lands
                                    # back to back (each iteration is one
                                    # claim() + at most one cheap SELECT).

    def _fpp_task_from_stream(worker_id):
        """Fase 1 (streams): serve /api/worker/fpp_task from the "fpp" work
        stream instead of the PG scan below. Idle cost: one claim() call
        against Redis/Memory — zero candidate/star queries — since the
        two-arm EXISTS guard and the two LIMIT-200 scans the legacy path
        needs below are never reached.

        Per claimed item this does exactly ONE SELECT (candidate PK JOIN
        stars) to re-check REAL eligibility via work_producers.fpp_eligible.
        This re-check is MANDATORY correctness, not an optimisation: the
        producer hooks (work_producers.maybe_enqueue_fpp, called from
        db.add_candidate's caller and db.update_candidate_fpp) enqueue from
        incoming-payload / pre-COALESCE values, not the row db ultimately
        persists — so a queued item can be stale (already handled, or never
        actually eligible) by the time it is claimed. Ineligible -> ack +
        next iteration, capped at _FPP_STREAM_MAX_ITERATIONS so a run of
        stale entries can never spin one HTTP request forever (the next
        poll picks up where this one left off).

        An eligible item is additionally held back (no serve, NO ack —
        see the inline comment) while its FITS metadata is not yet in the
        metadata cache, mirroring the legacy scan's availability filter.
        """
        ws = work_streams.get_stream("fpp")

        def _keep_if(item_key, consumer):
            # Only reached for PEL entries idle past the stream's
            # visibility_timeout_s (1800s) — a rare path, not the per-poll
            # hot path — so one extra SELECT here is cheap.
            # item_key arrives as a str (work-stream keys are always
            # strings); cast to int before binding against the integer PK —
            # SQLite's type affinity would silently coerce a bare string
            # here, but PostgreSQL (production) does not, and would raise
            # "operator does not exist: integer = text" on a raw string.
            try:
                cid = int(item_key)
            except (TypeError, ValueError):
                return False
            owner_row = db._query_one(
                "SELECT tic_id FROM candidates WHERE id = ?", (cid,))
            if owner_row is None:
                return False
            return _worker_alive_and_on(workers_state, consumer,
                                        owner_row["tic_id"])

        _fpp_gate_active = os.environ.get("EXONIT_FPP_GATE") == "1"
        # Same process-wide MetadataCache the legacy path resolves before its
        # candidate loop (a dict lookup after the first call, so it costs
        # nothing on the idle {"task": None} fast path).
        _mc = _server._get_shared_metadata_cache(cfg)

        def _next_item():
            """The item to consider this iteration: a real claim if any is
            available, else — only once the queue is EMPTY — a speculative copy
            of a tail item some other worker is still grinding on.

            The speculative fallback is what stops one straggler from holding a
            whole run hostage: exactly the 16-minutes-for-31-candidates tail of
            run s0014/lot20260714T175216. It costs nothing when there is real
            work (claim() wins first) and nothing when the tail is healthy
            (claim_speculative returns None until an item has been idle past the
            stream's visibility timeout). See work_streams.claim_speculative.
            """
            item = ws.claim(worker_id, keep_if=_keep_if)
            if item is not None:
                return item
            try:
                return ws.claim_speculative(worker_id)
            except Exception:
                logger.warning("fpp speculative claim failed", exc_info=True)
                return None

        for _ in range(_FPP_STREAM_MAX_ITERATIONS):
            item = _next_item()
            if item is None:
                return {"task": None}
            try:
                candidate_id = int(item.payload.get("candidate_id",
                                                     item.item_key))
            except (TypeError, ValueError):
                ws.ack(item.item_key)          # malformed message — drop it
                continue
            row = db._query_one(
                """SELECT c.id, c.tic_id, c.candidate_num, c.period_days,
                          c.t0, c.duration_days, c.depth_ppm, c.snr,
                          c.cnn_score, c.exominer_ffi_score,
                          c.triceratops_fpp,
                          c.triceratops_nfpp AS nfpp, s.n_sectors,
                          s.status AS star_status
                   FROM candidates c JOIN stars s ON c.tic_id = s.tic_id
                   WHERE c.id = ?""",
                (candidate_id,))
            if row is None:
                ws.ack(item.item_key)          # candidate gone
                continue
            # FPP-gate (lazy, opt-in EXONIT_FPP_GATE=1): mirror the scan
            # path's serve-time pre-marking of mass-uncomputable targets
            # (below, in the legacy loop) exactly — same
            # target_mass_is_uncomputable() call, same
            # db.mark_fpp_uncomputable() write — so a candidate the legacy
            # path would silently sentinel never reaches a worker via the
            # stream path either. Arm-A-only, like the legacy gate (a
            # candidate with a real triceratops_fpp already has a
            # computable mass by definition, so only fpp-still-NULL rows
            # are gate candidates).
            if _fpp_gate_active and row.get("triceratops_fpp") is None:
                try:
                    if _server.target_mass_is_uncomputable(
                            _mc, berger, row["tic_id"]):
                        db.mark_fpp_uncomputable(candidate_id)
                        ws.ack(item.item_key)
                        continue
                    # Rp-gate (2026-07-14, stellar_gates.py): the star is
                    # computable, but depth + R* imply a companion far too big
                    # to be a planet — an eclipsing binary or a blend. Running
                    # TRICERATOPS on it buys nothing; geometry already answered.
                    # This lazy arm is what drains the ALREADY-QUEUED backlog
                    # (the eager gate at TIC completion only covers new work).
                    # ack() is right for a speculative copy too: the item is
                    # settled for everyone, not just for this worker.
                    if stellar_gates.implies_stellar_companion(
                            _server.resolve_star_radius(_mc, row["tic_id"]),
                            row.get("depth_ppm")):
                        db.mark_fpp_uncomputable(
                            candidate_id, reason="stellar_companion")
                        ws.ack(item.item_key)
                        continue
                except Exception:
                    pass  # never let the gate break the backfill
            if not work_producers.fpp_eligible(row):
                ws.ack(item.item_key)          # stale: handled elsewhere
                continue
            # Metadata availability filter — mirror of the legacy scan's
            # `if not _mc.get_entry(c["tic_id"]): continue`. Serving a
            # candidate whose FITS metadata is not cached makes the worker
            # fail its LC load and POST fpp=-1/nfpp=-1, which
            # update_candidate_fpp persists as the PERMANENT uncomputable
            # sentinel (excluded from both eligibility arms forever) — a
            # transient cache gap must never be converted into that.
            # Deliberately do NOT ack (controller-adjudicated semantics):
            # metadata-missing is TRANSIENT — the legacy scan retries such
            # candidates on every poll, and an ack here would silently drop
            # the candidate until a manual stream rebuild. Leaving the item
            # un-acked keeps it in this consumer's PEL: it is redelivered
            # after the visibility timeout, and if the gap never resolves it
            # lands in the DLQ after max_deliveries — visible and inspectable,
            # strictly better than legacy's silent forever-retry.
            if not _mc.get_entry(row["tic_id"]):
                if item.speculative:
                    # Nothing to keep: a speculative copy owns no PEL entry, so
                    # "leave it un-acked" means the real owner still has it. Just
                    # hand the replica slot back so the next idle worker may try.
                    ws.release_replica(item.item_key)
                continue
            _set_worker_backfill_tic(worker_id, row["tic_id"], "FPP")
            return _build_backfill_payload(row)
        return {"task": None}

    @app.get("/api/worker/fpp_task")
    def worker_fpp_task(request: Request):
        """Return next candidate needing TRICERATOPS FPP calculation.

        Tracks in-memory assignments to prevent multiple idle workers from
        grabbing the same top-SNR candidate (race observed 2026-05-02:
        <host> + a laptop worker both processed TIC 28873762 simultaneously when the
        regular task queue was empty).
        """
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        worker_id = request.headers.get("x-worker-id", "")
        # ── Fase 1 (streams): serve from the work stream, no PG scans ──
        if work_producers.streams_enabled():
            return _fpp_task_from_stream(worker_id)
        # ── legacy scan path (unchanged) ──
        # Cheap idle guard. The heavy SELECT below is a JOIN + filter + ORDER BY
        # over the candidates with snr>=10 & (cnn>=0.7 OR em_ffi>=0.5) — ~1.7 s
        # every poll, even when there is nothing to hand out. Idle workers poll
        # continuously, so across N server processes that scan saturated the
        # cores doing useless work. The predicate-complete partial indexes
        # idx_cand_fpp_todo / idx_cand_nfpp_todo are EMPTY when no candidate
        # needs FPP, so these two EXISTS checks return in ~0.1 ms and we bail
        # before the heavy query.
        # (Over-approximation: omits the s.status='completed' JOIN condition, so
        # a positive here just means "run the real query"; a negative is a
        # definite "no work" — the real query is strictly more restrictive.)
        # Each arm needs ORDER BY snr DESC (+ LIMIT 1) to actually USE the
        # predicate-complete partial index. On SQLite, INDEXED BY forces it.
        # On PostgreSQL, INDEXED BY is stripped (it is a SQLite-only hint) and
        # WITHOUT the ORDER BY the planner mis-estimates the predicate as ~1.5M
        # rows — it cannot see that snr>=10 & (cnn>=0.7 OR em_ffi>=0.5) are
        # precisely the rows that GOT an FPP, so their fpp-IS-NULL intersection
        # is now empty. With that bogus estimate + LIMIT 1 it picks a full Seq
        # Scan of the sector partition (~400 ms/call × ~1.5 idle polls/s = the
        # 2026-06-26 idle-CPU regression on <host>). ORDER BY snr DESC
        # matches the index sort key, so PG drives the now-empty partial index
        # instead → ~0.2 ms. Both indexes are ON candidates(snr DESC) WHERE
        # <predicate> so the sort is free on both backends; the main query
        # below already orders this way.
        # Wrapped so a missing index never breaks the endpoint — on any error we
        # just fall through to the real query (correct, only slower).
        # OR-gate (2026-07-11, docs/reviews/injection_e2e_results_2026-07-11.md
        # + docs/reviews/ffi_cnn_investigation_2026-07-11.md §5): the
        # cnn_score>=0.7-only predicate below is replaced everywhere in this
        # endpoint by "(cnn_score >= 0.7 OR exominer_ffi_score >= 0.5)" — the
        # exact literal text idx_cand_fpp_todo / idx_cand_nfpp_todo are now
        # built with (exonit/db.py), so INDEXED BY keeps matching. Strictly
        # more permissive than before (monotonic — see
        # work_producers.verdict_eligible).
        try:
            _fpp_arm1 = db._query_one(
                "SELECT 1 FROM candidates INDEXED BY idx_cand_fpp_todo "
                "WHERE snr >= 10.0 AND (cnn_score >= 0.7 OR exominer_ffi_score >= 0.5) "
                "AND triceratops_fpp IS NULL "
                "ORDER BY snr DESC LIMIT 1")
            _fpp_arm2 = None if _fpp_arm1 else db._query_one(
                "SELECT 1 FROM candidates INDEXED BY idx_cand_nfpp_todo "
                "WHERE snr >= 10.0 AND (cnn_score >= 0.7 OR exominer_ffi_score >= 0.5) "
                "AND triceratops_fpp >= 0 "
                "AND triceratops_fpp < 0.5 AND triceratops_nfpp IS NULL "
                "ORDER BY snr DESC LIMIT 1")
            if not _fpp_arm1 and not _fpp_arm2:
                return {"task": None}
        except Exception:
            pass  # guard is a fast-path optimization; never let it break the poll
        # Find candidates needing FPP that have metadata (FITS available).
        # Two-arm form: each arm's predicate matches a predicate-complete partial
        # index (idx_cand_fpp_todo / idx_cand_nfpp_todo) EXACTLY, so both SQLite
        # (via INDEXED BY) and PostgreSQL (which strips the hint but recognises the
        # predicate) drive an snr-ordered index scan that stops at LIMIT 200 — no
        # candidates-table scan.  We merge the two capped arms in Python:
        # top-200-of-union == top-200 of each arm's own top-200, so this is exact
        # (the arms are mutually exclusive: fpp IS NULL vs fpp >= 0, no overlap).
        #
        # History: the earlier `c.id IN (SELECT ... UNION ...)` form planned fine on
        # SQLite but DEGENERATED on PostgreSQL — the IN-subquery doesn't force index
        # use and, with stale post-migration stats, the planner chose a nested-loop
        # join-filter over a 6.5M-row Materialize (cost ~1.9e11): ~3 min per call ×
        # ~33 polling server connections = the FPP-backfill PG meltdown of
        # 2026-06-26.  The two index-driven arms run in ~2.5 ms regardless of how
        # fresh the planner statistics are.
        # Fallback: if either index is missing, INDEXED BY raises OperationalError;
        # we catch it and re-run the original broad OR query (correct, only slower).
        _mc = _server._get_shared_metadata_cache(cfg)
        try:
            _fpp_arm_a = db._query(
                """SELECT c.*, s.n_sectors FROM candidates c INDEXED BY idx_cand_fpp_todo
                   JOIN stars s ON c.tic_id = s.tic_id
                   WHERE c.snr >= 10.0
                     AND (c.cnn_score >= 0.7 OR c.exominer_ffi_score >= 0.5)
                     AND c.triceratops_fpp IS NULL
                     AND s.status = 'completed'
                   ORDER BY c.snr DESC
                   LIMIT 200""")
            _fpp_arm_b = db._query(
                """SELECT c.*, s.n_sectors FROM candidates c INDEXED BY idx_cand_nfpp_todo
                   JOIN stars s ON c.tic_id = s.tic_id
                   WHERE c.snr >= 10.0
                     AND (c.cnn_score >= 0.7 OR c.exominer_ffi_score >= 0.5)
                     AND c.triceratops_fpp >= 0 AND c.triceratops_fpp < 0.5
                     AND c.triceratops_nfpp IS NULL
                     AND s.status = 'completed'
                   ORDER BY c.snr DESC
                   LIMIT 200""")
            candidates = sorted(
                _fpp_arm_a + _fpp_arm_b,
                key=lambda _c: _c["snr"], reverse=True)[:200]
        except Exception:
            # Index missing or renamed — fall back to the original broad query.
            # Result is identical; only performance differs.
            candidates = db._query(
                """SELECT c.*, s.n_sectors FROM candidates c
                   JOIN stars s ON c.tic_id = s.tic_id
                   WHERE c.snr >= 10.0
                     AND (c.cnn_score >= 0.7 OR c.exominer_ffi_score >= 0.5)
                     AND (
                       c.triceratops_fpp IS NULL
                       OR (
                         c.triceratops_fpp >= 0
                         AND c.triceratops_fpp < 0.5
                         AND c.triceratops_nfpp IS NULL
                       )
                     )
                     AND s.status = 'completed'
                   ORDER BY c.snr DESC
                   LIMIT 200""")

        # Reap stale FPP assignments (worker died silently) and pick a
        # candidate that nobody else is currently working on.
        # A2a: atomic claim via HSETNX; reap via hgetall_json + hdel.
        # D12: _reap_stale_assignments/_claim_candidate/_build_backfill_payload
        # are shared with worker_exominer_task below.
        _reap_stale_assignments(_FPP_ASSIGNMENTS_KEY, FPP_ASSIGNMENT_TIMEOUT_SECONDS)
        now = _time.time()

        # FPP-gate: when EXONIT_FPP_GATE=1, pre-mark candidates whose target
        # star has no usable stellar mass so workers never waste a round-trip
        # on them (they would return fpp=-1 immediately after downloading the
        # FITS and running the detrend).  The check replicates the same
        # cache → Berger → derivation enrichment that /api/nearby performs,
        # so the decision is bit-identical to what the worker would see.
        # Guard: never touch a candidate whose fpp is already set (the UPDATE
        # in db.mark_fpp_uncomputable carries AND triceratops_fpp IS NULL).
        _fpp_gate_active = os.environ.get("EXONIT_FPP_GATE") == "1"

        task_candidate = None
        for c in candidates:
            cid = c["id"]
            if not _mc.get_entry(c["tic_id"]):
                continue
            # FPP-gate: skip (and pre-mark) uncomputable targets.
            if _fpp_gate_active and c.get("triceratops_fpp") is None:
                try:
                    if _server.target_mass_is_uncomputable(_mc, berger, c["tic_id"]):
                        db.mark_fpp_uncomputable(cid)
                        continue
                    # Rp-gate (2026-07-14): the star is fine, but THIS transit
                    # is too deep for it to be a planet — see stellar_gates.
                    if stellar_gates.implies_stellar_companion(
                            _server.resolve_star_radius(_mc, c["tic_id"]),
                            c.get("depth_ppm")):
                        db.mark_fpp_uncomputable(
                            cid, reason="stellar_companion")
                        continue
                except Exception:
                    pass  # never let the gate break the backfill
            if _claim_candidate(_FPP_ASSIGNMENTS_KEY, cid, now=now):
                task_candidate = c
                break
        if not task_candidate:
            return {"task": None}
        c = task_candidate
        _set_worker_backfill_tic(request.headers.get("x-worker-id"),
                                 c["tic_id"], "FPP")
        return _build_backfill_payload(c)

    @app.post("/api/worker/fpp_result")
    async def worker_fpp_result(request: Request):
        """Receive TRICERATOPS FPP result from worker, trigger Claude in background."""
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        _clear_worker_backfill_tic(request.headers.get("x-worker-id"))
        body = await request.json()
        candidate_id = body.get("candidate_id")
        fpp = body.get("fpp")
        nfpp = body.get("nfpp")
        prob_planet = body.get("prob_planet")
        verdict = body.get("verdict")

        # Adjudicate the result BEFORE writing it (2026-07-14, speculative tail
        # execution). settle() both releases the stream's assignment — the PEL
        # entry IS the assignment, so acking it is what the old ack() call below
        # used to do — and tells us whether this worker LOST a speculative race
        # for this candidate. It only ever reports "loser" for an item that was
        # actually raced; an ordinary item yields "winner" (in flight, now
        # acked) or "unknown" (legacy path, manual re-run, or a result landing
        # after reconcile_stream rebuilt the queue), and both of those are
        # persisted exactly as before.
        #
        # A loser's result is DISCARDED, not written. TRICERATOPS is a Monte
        # Carlo over np.random with no seed (worker/fpp_native.py), so the two
        # racers computed two different FPP values: writing both would leave the
        # row holding whichever POST won the NETWORK race, nondeterministically.
        # Speculation is allowed to cost duplicate CPU; it is not allowed to
        # cost a nondeterministic number in the database.
        outcome = "unknown"
        if candidate_id is not None:
            if work_producers.streams_enabled():
                try:
                    outcome = work_streams.get_stream("fpp").settle(
                        str(candidate_id))
                except Exception:
                    # A Redis hiccup must not 500 the worker, and must not lose
                    # its result: degrade to the pre-speculation behaviour
                    # (persist unconditionally).
                    logger.warning(
                        "fpp stream settle failed for candidate_id=%s",
                        candidate_id, exc_info=True)
            else:
                # Legacy path: the claims hash is the assignment.
                get_shared_state().hdel(_FPP_ASSIGNMENTS_KEY, str(candidate_id))

        if outcome == "loser":
            logger.info(
                "fpp candidate_id=%s: speculative race already won by another "
                "worker — discarding this (equally valid, but different) result",
                candidate_id)
            return {"status": "ok", "outcome": "speculative_loser"}

        if candidate_id and fpp is not None:
            db.update_candidate_fpp(candidate_id, fpp, nfpp=nfpp,
                                    prob_planet=prob_planet,
                                    verdict=verdict)
            # Bump cumulative progress counter so /api/backfill/fpp can
            # report real done count instead of inferring it from queue
            # size (which is unreliable while the primary pipeline keeps
            # adding new FPP-actionable candidates).
            _st = get_shared_state()
            _st.incr(_FPP_BACKFILL_DONE_KEY)
            _st.set_if_absent(_FPP_BACKFILL_STARTED_AT_KEY, str(_time.time()))

        # Trigger verdict classifier in background
        if _verdict_enabled and candidate_id:
            tic_id = body.get("tic_id")
            candidate = body.get("candidate") or {}
            if tic_id and candidate:
                _get_verdict_executor().submit(
                    _run_verdict_background, db, tic_id, candidate,
                    candidate.get("cnn_score"))

        return {"status": "ok"}

    # --- ExoMiner backfill tasks for workers ---

    _EXOMINER_STREAM_MAX_ITERATIONS = 5  # same bound as _fpp_task_from_stream:
                                        # cap one HTTP request's work even if
                                        # a run of stale/ineligible/no-
                                        # metadata entries lands back to back.

    def _exominer_task_from_stream(worker_id):
        """Fase 1 (streams): serve /api/worker/exominer_task from the
        "exominer" work stream instead of the has_work-TTL-gated PG scan
        below. Idle cost: one claim() call against Redis/Memory — the
        idle-gate cache (``_EXOMINER_HAS_WORK_KEY``) and the LIMIT-200 scan
        the legacy path needs below are never reached.

        Per claimed item this does exactly ONE SELECT (candidate PK JOIN
        stars) to re-check REAL eligibility via
        work_producers.exominer_eligible. This re-check is MANDATORY
        correctness, not an optimisation — see _fpp_task_from_stream's
        docstring for the full rationale (approximate producer-hook
        payloads vs. what PG ultimately persisted). Ineligible -> ack +
        next iteration, capped at _EXOMINER_STREAM_MAX_ITERATIONS.

        Metadata-availability handling matches the FPP stream (hold back,
        NO ack, NO write) and INTENTIONALLY DIVERGES from the legacy scan
        below, which -1-marks the TIC's candidates at serve time
        (db.mark_ffi_only_candidates). That legacy marking is a 2026-06-10
        QUEUE-DRAIN mechanism (commit 2ccb7888: stranded no-metadata TICs
        clogged the ORDER BY snr LIMIT 200 window and the backfill never
        drained), NOT an eligibility fact: every row reaching this check
        has has_spoc=1 — MAST-confirmed SPOC data EXISTS — so the -1
        "FFI-only / no DV" sentinel would be factually wrong, a permanent
        silent loss of a real ExoMiner score whenever the metadata gap is
        just indexing lag. The producer hook enqueues within seconds of
        the has_spoc flip; the legacy backlog's slow snr-ordered window
        was an informal grace period that FIFO+instant-enqueue removes,
        so claim-time marking here would fire on exactly those fresh rows.
        The stream has no clogging problem to drain: a held-back item sits
        in this consumer's PEL (not at the queue head), is retried on
        redelivery every visibility_timeout (metadata indexed in the
        meantime -> serves normally), and after max_deliveries lands in
        the DLQ — visible, inspectable, re-enqueued by a post-indexing
        rebuild. -1 is NEVER written on this path; the sentinel remains
        exclusively at the SPOC-gate producer paths (has_spoc=0 per-TIC
        via _run_spoc_check_background, has_spoc=NULL bulk via
        mark_unconfirmed_ffi_candidates) — the set legacy genuinely marks
        as policy rather than as drain plumbing.
        """
        ws = work_streams.get_stream("exominer")

        def _keep_if(item_key, consumer):
            # Only reached for PEL entries idle past visibility_timeout_s —
            # see _fpp_task_from_stream's _keep_if for the full rationale
            # (int() cast: PG raises on integer = text, SQLite would not).
            try:
                cid = int(item_key)
            except (TypeError, ValueError):
                return False
            owner_row = db._query_one(
                "SELECT tic_id FROM candidates WHERE id = ?", (cid,))
            if owner_row is None:
                return False
            return _worker_alive_and_on(workers_state, consumer,
                                        owner_row["tic_id"])

        _mc = _server._get_shared_metadata_cache(cfg)

        for _ in range(_EXOMINER_STREAM_MAX_ITERATIONS):
            item = ws.claim(worker_id, keep_if=_keep_if)
            if item is None:
                return {"task": None}
            try:
                candidate_id = int(item.payload.get("candidate_id",
                                                     item.item_key))
            except (TypeError, ValueError):
                ws.ack(item.item_key)          # malformed message — drop it
                continue
            row = db._query_one(
                """SELECT c.id, c.tic_id, c.candidate_num, c.period_days,
                          c.t0, c.duration_days, c.depth_ppm, c.snr,
                          c.cnn_score, c.exominer_score, s.n_sectors,
                          s.status AS star_status, s.has_spoc
                   FROM candidates c JOIN stars s ON c.tic_id = s.tic_id
                   WHERE c.id = ?""",
                (candidate_id,))
            if row is None:
                ws.ack(item.item_key)          # candidate gone
                continue
            if not work_producers.exominer_eligible(row):
                ws.ack(item.item_key)          # stale: handled elsewhere
                continue
            tic_id = row["tic_id"]
            # Metadata availability — hold back WITHOUT ack and WITHOUT any
            # DB write (see docstring: has_spoc=1 proves SPOC data exists,
            # so a claim-time -1 would silently lose a real score; PEL
            # redelivery is the grace period, DLQ at max_deliveries is the
            # visible terminal state).
            if not _mc.get_entry(tic_id):
                continue
            _set_worker_backfill_tic(worker_id, tic_id, "ExoMiner")
            return _build_backfill_payload(row)
        return {"task": None}

    @app.get("/api/worker/exominer_task")
    def worker_exominer_task(request: Request):
        """Return next candidate needing ExoMiner score.

        Same race-safe pattern as worker_fpp_task: the
        ``"exominer:assignments"`` hash in the shared store claims a
        candidate atomically (HSETNX) so two idle workers — even in
        separate server processes — can't race on the same candidate.

        Two layers of gating:
        - ``EXONIT_EXOMINER_BACKFILL=1`` to enable the endpoint at all
          (was hard-disabled 2026-05-02 after a period-alias bug marked
          strong candidates as -1; fix shipped 2026-05-16, re-enabled by
          env var).
        - ``EXONIT_EXOMINER_WORKERS`` (comma-separated worker names) to
          restrict which workers receive tasks. Workers without the
          exominer_plusplus_single.onnx model file installed locally
          would raise FileNotFoundError on every task and mark each
          candidate as -1, undoing real scores. Until ExoMiner native is
          deployed fleet-wide (AWS+GCP cloud workers don't ship the
          model yet) this allowlist keeps tasks flowing only to capable
          workers. Leave the env var unset/empty to allow all workers.
        """
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if os.environ.get("EXONIT_EXOMINER_BACKFILL", "0") != "1":
            return {"task": None}
        # Pause ExoMiner backfill while a scan is in flight: ExoMiner ONNX runs on
        # the GPU, same device as the BLS scan, so serving backfill during SCANNING
        # makes workers contend with the primary pipeline for the GPU. FPP backfill
        # (TRICERATOPS, CPU) is left running.
        if _server._scan_in_flight():
            return {"task": None}
        allowlist = _load_exominer_allowlist()
        if allowlist is not None:
            worker_id = request.headers.get("x-worker-id", "")
            worker_name = workers_state.get(worker_id, {}).get("name", "")
            if worker_name not in allowlist:
                return {"task": None}
        # ── Fase 1 (streams): serve from the work stream, no PG scans ──
        if work_producers.streams_enabled():
            return _exominer_task_from_stream(
                request.headers.get("x-worker-id", ""))
        # ── legacy scan path (unchanged) ──
        # Meltdown-safe idle gate. The honest "is there assignable work?" check
        # JOINs stars (has_spoc=1) and is expensive on PG (full stars scan); the
        # old guard tested only ``exominer_score IS NULL AND cnn>=0.7``, which on
        # the FFI-dominated DB matches ~1.5M candidates that are has_spoc=FALSE and
        # therefore NEVER assignable — so it never short-circuited and every idle
        # poll fell through to the heavy JOIN+sort query (the 2026-06-20 PG
        # meltdown). Cache the honest answer for a few seconds so the scan runs at
        # most ~once per window across the whole fleet. On any error fall through
        # to the real query (a missed cache is never a correctness issue).
        _st = get_shared_state()
        _has_work = _st.get(_EXOMINER_HAS_WORK_KEY)
        if _has_work is None:
            try:
                _row = db._query_one(
                    "SELECT 1 FROM candidates c JOIN stars s ON c.tic_id = s.tic_id "
                    "WHERE c.exominer_score IS NULL AND c.cnn_score >= 0.7 "
                    "AND s.status = 'completed' AND s.has_spoc = 1 LIMIT 1")
                _has_work = "1" if _row else "0"
            except Exception:
                _has_work = "1"  # fall through to the real query on any error
            _st.set(_EXOMINER_HAS_WORK_KEY, _has_work, ttl=_EXOMINER_HAS_WORK_TTL)
        if _has_work == "0":
            return {"task": None}
        _mc = _server._get_shared_metadata_cache(cfg)
        candidates = db._query(
            """SELECT c.id, c.tic_id, c.candidate_num, c.period_days, c.t0,
                      c.duration_days, c.depth_ppm, c.snr, s.n_sectors
               FROM candidates c JOIN stars s ON c.tic_id = s.tic_id
               WHERE s.status = 'completed'
                 AND s.has_spoc = 1
                 AND c.cnn_score >= 0.7
                 AND c.exominer_score IS NULL
               ORDER BY c.snr DESC
               LIMIT 200""")

        # A2a: atomic claim via HSETNX; reap stale via hgetall_json + hdel.
        # D12: _reap_stale_assignments/_claim_candidate/_build_backfill_payload
        # are shared with worker_fpp_task above.
        _reap_stale_assignments(_EXOMINER_ASSIGNMENTS_KEY, EXOMINER_ASSIGNMENT_TIMEOUT_SECONDS)
        now = _time.time()

        task_candidate = None
        _marked_no_metadata: set = set()  # avoid marking the same TIC twice per poll
        for c in candidates:
            cid = c["id"]
            tic_id = c["tic_id"]
            if not _mc.get_entry(tic_id):
                # No metadata entry → this TIC is un-serveable by the worker-native
                # ExoMiner path (inputs cannot be built without metadata).  Mark all
                # eligible candidates for this TIC with the -1 sentinel so they leave
                # the queue permanently.  Semantics are identical to the SPOC-gate
                # sentinel: verdict v2.1 treats -1 as feature -1.0 (non-regressive).
                # Guard: mark_ffi_only_candidates only touches rows with
                # exominer_score IS NULL, so it never overwrites a real score.
                if tic_id not in _marked_no_metadata:
                    n = db.mark_ffi_only_candidates(tic_id)
                    logger.info(
                        "ExoMiner-serve TIC %s: no metadata entry, "
                        "marked %d candidate(s) exominer_score=-1",
                        tic_id, n)
                    _marked_no_metadata.add(tic_id)
                continue
            if _claim_candidate(_EXOMINER_ASSIGNMENTS_KEY, cid, now=now):
                task_candidate = c
                break
        if not task_candidate:
            return {"task": None}
        c = task_candidate
        _set_worker_backfill_tic(request.headers.get("x-worker-id"),
                                 c["tic_id"], "ExoMiner")
        return _build_backfill_payload(c)

    @app.post("/api/worker/exominer_result")
    async def worker_exominer_result(request: Request):
        """Receive ExoMiner score from worker; release assignment slot."""
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        _clear_worker_backfill_tic(request.headers.get("x-worker-id"))
        body = await request.json()
        candidate_id = body.get("candidate_id")
        score = body.get("score")
        tce_period = body.get("tce_period")
        sector_run = body.get("sector_run")

        if candidate_id and score is not None:
            db.update_candidate_exominer(
                candidate_id, score=score,
                tce_period=tce_period, sector_run=sector_run)
            # Re-run verdict classifier now that ExoMiner is populated.
            # The rescue gate (apply_planet_rescue_gate) requires EM to
            # fire — without this re-trigger, hot Jupiters classified at
            # primary-result time stay tagged likely_false_positive
            # forever, even after EM is filled in.
            if _verdict_enabled:
                cand_row = db._query(
                    """SELECT c.tic_id, c.candidate_num, c.period_days,
                              c.t0, c.duration_days, c.depth_ppm, c.snr,
                              c.cnn_score, c.triceratops_fpp,
                              c.triceratops_nfpp, c.triceratops_prob_planet,
                              c.exominer_score, c.exominer_ffi_score,
                              c.exominer_tce_period, c.exominer_sector_run
                       FROM candidates c WHERE c.id = ?""",
                    (candidate_id,))
                if cand_row:
                    r = cand_row[0]
                    cand_dict = dict(r)
                    snr = cand_dict.get("snr") or 0
                    cnn = cand_dict.get("cnn_score")
                    # OR-gate — see work_producers.verdict_eligible.
                    if work_producers.verdict_eligible(
                            snr, cnn, cand_dict.get("exominer_ffi_score")):
                        _get_verdict_executor().submit(
                            _run_verdict_background, db,
                            str(cand_dict["tic_id"]), cand_dict, cnn)

        # Release the assignment slot regardless of outcome so the next
        # worker can pick up new work. Fase 1 (streams): the stream's own
        # PEL entry is the assignment — ack() releases it (removing it from
        # the queue entirely); the legacy claims-hash hdel stays
        # legacy-path-only (mirrors worker_fpp_result above). Guarded like
        # that ack: a Redis hiccup here must not 500 the worker after the
        # score (and the verdict re-trigger above) are already committed.
        if candidate_id is not None:
            if work_producers.streams_enabled():
                try:
                    work_streams.get_stream("exominer").ack(str(candidate_id))
                except Exception:
                    logger.warning(
                        "exominer stream ack failed for candidate_id=%s",
                        candidate_id, exc_info=True)
            else:
                get_shared_state().hdel(_EXOMINER_ASSIGNMENTS_KEY, str(candidate_id))
        return {"status": "ok"}

    # --- Data endpoints (metadata + FITS serving) ---

    # --- Worker auto-update ---

    @app.get("/api/worker/update")
    def worker_update_manifest():
        """Serve update manifest so workers can check for new versions.

        Enumerates EVERY syncable worker file dynamically — top-level
        ``worker/*`` files plus ``worker/models/*`` — instead of a hardcoded
        list, so any code change (incl. a brand-new module) propagates without
        editing this endpoint. The per-file endpoint only accepts a
        single-level subdir, so deeper packages (e.g. ``exominer_native/``)
        are intentionally NOT served here.
        """
        import hashlib
        worker_dir = _PROJECT_ROOT / "worker"
        # Runtime code + data only (whitelist), so logs/backups/installers
        # (*.gz, *.bak, *.log, config*.json, setup_*.sh/.bat, requirements.txt)
        # are never distributed to workers.
        code_ext = {".py", ".csv"}
        model_skip_ext = {".log", ".pyc", ".pyo"}

        def _syncable():
            for f in sorted(worker_dir.glob("*")):
                if f.is_file() and not f.name.startswith(".") \
                        and f.suffix in code_ext:
                    yield f.name
            models = worker_dir / "models"
            if models.is_dir():
                for f in sorted(models.glob("*")):
                    if f.is_file() and f.suffix not in model_skip_ext:
                        yield f"models/{f.name}"

        files = []
        for name in _syncable():
            fpath = worker_dir / name
            try:
                raw = fpath.read_bytes()
            except OSError:
                continue
            normalized = raw.replace(b"\r\n", b"\n")
            sha = hashlib.sha256(normalized).hexdigest()
            files.append({"filename": name, "sha256": sha})
        return {"files": files}

    @app.get("/api/worker/update/{filename:path}")
    def worker_update_file(filename: str):
        """Serve a worker file for auto-update.

        Accepts filenames with a single-level subdirectory (e.g.
        ``models/exominer_plusplus_single.keras``). Traversal (``..``) and
        absolute paths are rejected.
        """
        worker_dir = _PROJECT_ROOT / "worker"
        if ".." in filename or filename.startswith("/") or filename.count("/") > 1:
            return JSONResponse({"error": "Invalid"}, status_code=400)
        fpath = worker_dir / filename
        if fpath.exists() and fpath.is_file():
            media = "application/octet-stream" if fpath.suffix in (".keras", ".onnx") else "text/plain"
            return FileResponse(fpath, media_type=media)
        return JSONResponse({"error": "Not found"}, status_code=404)

    @app.post("/api/worker/control")
    async def worker_control(request: Request):
        # Intentionally unauthenticated: this is a DASHBOARD button (pause/
        # resume/stop a worker), same LAN-trust model as /api/stop,
        # /api/discovery/scan and the /api/admin/* resets. Worker-facing
        # endpoints (which carry the token) all use _check_auth.
        body = await request.json()
        wid = body.get("worker_id")
        command = body.get("command")
        if wid and command in ("pause", "resume", "shutdown"):
            get_shared_state().hset_json(
                _PENDING_COMMANDS_KEY, str(wid),
                {"command": command, "ts": _time.time()})
            return {"status": "ok", "command": command}
        return JSONResponse({"error": "invalid command"}, status_code=400)

    # --- Worker package download ---

    def _get_server_ip(request: Request) -> str:
        """Get the server IP as seen by the client."""
        host = request.headers.get("host", "").split(":")[0]
        if host and host not in ("localhost", "127.0.0.1", "0.0.0.0"):
            return host
        # Fallback: detect LAN IP
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "SERVER_IP"

    @app.get("/api/worker/download")
    def download_worker_package(request: Request):
        """Generate and serve a ZIP with all worker files for easy deployment."""
        import io
        import zipfile
        from fastapi.responses import StreamingResponse

        worker_dir = _PROJECT_ROOT / "worker"
        if not worker_dir.is_dir():
            return JSONResponse({"error": "Worker directory not found"}, status_code=404)

        server_ip = _get_server_ip(request)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Worker files (skip config.json — we generate a template below)
            skip = {"__pycache__", ".update_manifest.json", "config.json",
                    "venv", "fits_cache"}
            for f in worker_dir.iterdir():
                if f.name.startswith("__") or f.name.startswith("."):
                    continue
                if f.name in skip:
                    continue
                if f.is_file():
                    zf.write(f, f"exonit-worker/{f.name}")
                elif f.is_dir() and f.name == "models":
                    for mf in f.iterdir():
                        if mf.is_file():
                            zf.write(mf, f"exonit-worker/models/{mf.name}")

            # Generate config.json with the real server IP
            config_template = {
                "server_url": f"http://{server_ip}:{cfg.port}",
                "token": cfg.worker_token,
                "worker_name": "worker-CHANGEME",
                "cpu_threads": 4,
                "gpu_enabled": False,
                "poll_interval_seconds": 5,
                "heartbeat_interval_seconds": 30,
                "fits_cache_dir": "fits_cache",
                "gui_port": 8401,
            }
            zf.writestr("exonit-worker/config.json",
                        json.dumps(config_template, indent=2))

            # Add README with universal installer instructions
            readme = f"""# ExoNit Worker

## Quick Install (all platforms)

    python install.py

This will auto-detect your OS, install dependencies, configure GPU,
and connect to the server at http://{server_ip}:{cfg.port}

## One-liner (Linux/macOS via SSH)

    curl -sL http://{server_ip}:{cfg.port}/api/worker/download -o exonit-worker.zip && unzip -o exonit-worker.zip && cd exonit-worker && python3 install.py

## Options

    python install.py --update                          # Update deps only
    python install.py --unattended --server-url URL     # Non-interactive

## Auto-updates
The worker auto-updates from the server every 5 minutes.
"""
            zf.writestr("exonit-worker/README.md", readme)

        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=exonit-worker.zip"},
        )

