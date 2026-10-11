"""FastAPI server for ExoNit dashboard and API."""

import json
import logging
import mimetypes
import os
import re
import subprocess
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from exonit import __version__
from exonit import stellar_gates
from exonit.berger_catalog import BergerCatalog
from exonit.config import ExoNitConfig
from exonit.db import ExoNitDB, _tic_str
from exonit.dist_semaphore import DistributedSemaphore
from exonit.routes.deps import ServerDeps
from exonit.shared_state import get_shared_state
from exonit.verdict_io import build_star_info, default_vetting_result
from exonit.worker_recycle import install as _install_worker_recycle
from exonit.worker_registry import WorkerRegistry

# NOTE (Fase D create_app() split, 2026-07-02): asyncio / secrets / signal /
# sys / Query / RedirectResponse / Response / datetime / timezone /
# select_in_chunks were imported here only for endpoint bodies that have
# since moved to exonit/routes/*.py (which import them there instead) —
# removed as dead imports rather than left stale. No behavior change:
# nothing in this file (module-level helpers, create_app() wiring, main())
# used them.
#
# `subprocess` is DELIBERATELY kept even though nothing in this file calls
# it directly: tests/test_aws_endpoints.py and tests/test_aws_page.py patch
# `exonit.server.subprocess.Popen` (the real subprocess module, reached via
# this module's own attribute — a plain `import subprocess` here is what
# makes `exonit.server.subprocess` resolvable at all for that patch target).
# Removing it breaks those tests with AttributeError even though the split
# moved every actual subprocess.* call site into
# exonit/routes/{admin,aws,discovery}.py.

logger = logging.getLogger("exonit.server")

# Tasks stay assigned until the worker either reports a result or has been
# silent for this many seconds. Anything shorter (e.g. the old 120s) causes
# spurious reassignment during injection recovery (which alone can take 5-7
# min on a long-baseline target). 2026-08-07 (operator decision): 1800 -> 900. The 1800
# guard protected Python workers whose heartbeat stalls under the GIL during
# heavy BLS/injection; the primary fleet is now ~96% native (Rust/Swift,
# heartbeat on its own thread, no GIL stalls) and 900s still leaves 2x margin
# over the 5-7 min legit worst case of the remaining Python scanners
# (laptop/M1). Worst downside is duplicate compute, never a wrong result.
STALE_TASK_TIMEOUT_SECONDS = 900  # 15 min
ABSOLUTE_ABANDON_TIMEOUT_SECONDS = 3600  # hard ceiling: a task assigned longer
# than this is reassigned regardless of worker liveness / current_tic — no
# legitimate single TIC takes an hour (worst legit case ~7 min, even on the
# slowest worker under GIL pressure).
WORKER_OFFLINE_SECONDS = 300       # /api/worker/list surfaces a worker as
                                   # status='offline' once its last heartbeat
                                   # is older than this (dashboard UX — does
                                   # NOT evict from workers_state so Plan C.3's
                                   # "GETs are pure-read" invariant holds).
                                   # 300s (2026-07-14, operator request): was
                                   # 600s, which meant a worker that died could
                                   # sit "online" on the dashboard for 10 min.
                                   # Now aligned with WORKER_ALIVE_WINDOW below,
                                   # so what the dashboard calls offline is
                                   # exactly what the work-stealing logic calls
                                   # dead — one definition of "alive", not two.
WORKER_ALIVE_WINDOW_SECONDS = 300  # a worker whose last heartbeat is within
                                   # this window is considered still processing.
                                   # 300s (2026-07-14, operator request): a dead
                                   # worker's work is reclaimed twice as fast.
                                   #
                                   # ⚠️ KNOWN TENSION, read before raising or
                                   # lowering this again. It was 120s, then
                                   # RAISED to 600s because numba parallel
                                   # sections can hold the GIL for 5-7 min on
                                   # long-baseline TICs, blocking the heartbeat
                                   # THREAD while the worker is perfectly
                                   # healthy — at 120s that made the server
                                   # declare live workers dead and hand their
                                   # task to someone else (HP + <host>-M1 both
                                   # ran TIC 81178747 on 2026-04-29).
                                   # 300s sits INSIDE that documented 5-7 min
                                   # stall window, so the duplicate-assignment
                                   # failure mode is reachable again — but only
                                   # for a task that ALSO exceeds its stream's
                                   # visibility timeout / STALE_TASK_TIMEOUT
                                   # (both 1800s), which is the actual trigger.
                                   # If duplicate TICs reappear in the fleet
                                   # (the health/work "duplicate_tics" counter),
                                   # this constant is the first suspect.

# FPP backfill assignment tracking. Without this, /api/worker/fpp_task is
# stateless: every idle worker polling at the same time receives the same
# top-SNR candidate, wasting compute.
# A2a: backed by shared store hash "fpp:assignments" {candidate_id: {"ts": unix}}.
# HSETNX gives atomic cross-process claim; stale entries are reaped on read.
FPP_ASSIGNMENT_TIMEOUT_SECONDS = 1800  # 30 min
_FPP_ASSIGNMENTS_KEY = "fpp:assignments"

# Same pattern as FPP backfill, applied to ExoMiner backfill
# (/api/worker/exominer_task).
# A2a: backed by shared store hash "exominer:assignments".
EXOMINER_ASSIGNMENT_TIMEOUT_SECONDS = 1800
_EXOMINER_ASSIGNMENTS_KEY = "exominer:assignments"
# TTL-cached "is there real ExoMiner backfill work?" flag. The honest check
# (JOIN stars WHERE has_spoc=1) is expensive on PG (full stars scan ~370k
# buffers); computing it on every worker poll caused the 2026-06-20 PG meltdown
# once the real queue drained (the old cheap guard only tested
# ``exominer_score IS NULL`` which matches ~1.5M FFI candidates that are NEVER
# assignable — has_spoc=FALSE — so it never short-circuited). Caching the result
# for a few seconds runs that scan at most ~once/window regardless of fleet size.
_EXOMINER_HAS_WORK_KEY = "exominer:has_work"
_EXOMINER_HAS_WORK_TTL = 20.0

# Worker -> TIC tracking for backfill (FPP / ExoMiner). The worker sets its
# own current_tic but reports it via the heartbeat, which is GIL-starved
# during the synchronous TRICERATOPS/ExoMiner compute — by the time the
# heartbeat fires the worker has already reset current_tic to None, so the
# dashboard never shows who is processing what. The fpp_task/exominer_task
# endpoints DO know the worker_id (X-Worker-ID header) at assignment time, so
# we record it here and /api/worker/list prefers it over the stale heartbeat
# value.
# A2a: backed by shared store hash "worker:backfill_tic"
# {worker_id: {"tic_id", "task_type", "ts"}}.
_WORKER_BACKFILL_TIC_KEY = "worker:backfill_tic"

#: Hard ceiling on how long ONE backfill item may be held by a worker that
#: still looks alive (2026-07-14: the FPP tail).
#:
#: ``_worker_alive_and_on`` below is a backfill stream's ``keep_if``: it vetoes
#: stealing an item while the owning worker has a fresh heartbeat and is still
#: registered on that TIC. Without a ceiling, "alive + on-tic" is an UNBOUNDED
#: veto — and a worker can be alive and on-tic while making no progress at all
#: (<host>'s WDDM VRAM freeze: heartbeat fine, compute silently dead; a Mac that
#: went to sleep mid-item; a numba deadlock). The item then never returns to the
#: queue and the whole run waits on it, with the rest of the fleet idle. Run
#: s0014/lot20260714T175216 spent 16 of its 34 FPP minutes on its last 31
#: candidates, in flat steps of zero progress; the operator unstuck it by
#: KILLING workers by hand, which is precisely this veto being lifted manually.
#:
#: This is the exact counterpart of ABSOLUTE_ABANDON_TIMEOUT_SECONDS on the
#: primary path (branch 1 of reassign_stale_tasks): past the ceiling the item is
#: stealable no matter how healthy the owner claims to be. Set well ABOVE the
#: real p99 of a single TRICERATOPS run (minutes, not quarter-hours) so it never
#: fires on legitimately slow work — and note the speculative tail path
#: (work_streams.claim_speculative) already covers a genuinely slow-but-alive
#: worker WITHOUT taking its item away, so this ceiling only ever has to catch
#: the worker that is lying about being alive.
BACKFILL_HOLD_CEILING_SECONDS = int(
    os.environ.get("EXONIT_BACKFILL_HOLD_CEILING_S", "900"))


def _set_worker_backfill_tic(worker_id, tic_id, task_type):
    """Record that ``worker_id`` started a backfill TIC (called at assignment)."""
    if not worker_id:
        return
    get_shared_state().hset_json(
        _WORKER_BACKFILL_TIC_KEY,
        str(worker_id),
        {"tic_id": tic_id, "task_type": task_type, "ts": _time.time()},
    )


def _clear_worker_backfill_tic(worker_id):
    """Clear a worker's backfill TIC (called when its *_result arrives)."""
    if not worker_id:
        return
    get_shared_state().hdel(_WORKER_BACKFILL_TIC_KEY, str(worker_id))


def _get_worker_backfill_tic(worker_id, max_age=1800):
    """Return (tic_id, task_type) the worker is backfilling, or None if absent/stale."""
    if not worker_id:
        return None
    store = get_shared_state()
    bf = store.hget_json(_WORKER_BACKFILL_TIC_KEY, str(worker_id))
    if not bf:
        return None
    if _time.time() - bf["ts"] > max_age:
        store.hdel(_WORKER_BACKFILL_TIC_KEY, str(worker_id))
        return None
    return (bf["tic_id"], bf["task_type"])


def _backfill_hold_age(worker_id):
    """Seconds since *worker_id* was handed its current backfill item, or None
    if it holds none.

    A pure READ, unlike ``_get_worker_backfill_tic``: that helper HDELs the
    entry once it is older than ``max_age``, which is fine for its caller (the
    dashboard's "who is on what") but wrong here — the ``keep_if`` ceiling must
    be able to observe an OVERDUE hold repeatedly (every stream poll, from any
    of the A3 uvicorn processes) and still find it there. Deleting the evidence
    on first read would make the second reader see "no hold at all", which
    ``_worker_alive_and_on`` also treats as stealable — same verdict by luck,
    but it would silently wipe the dashboard's view of a worker that is, in
    fact, still grinding on that TIC.
    """
    if not worker_id:
        return None
    bf = get_shared_state().hget_json(_WORKER_BACKFILL_TIC_KEY, str(worker_id))
    if not bf:
        return None
    return _time.time() - bf["ts"]


def _worker_alive_and_on(workers_state, worker_id, tic_id) -> bool:
    """True iff *worker_id* is alive (fresh heartbeat) AND its recorded
    ``worker:backfill_tic`` entry is *tic_id* — i.e. it is legitimately
    still processing this exact item right now, so a work-stream's
    ``keep_if`` must not let it be stolen.

    Mirrors branch 3 of ``db.reassign_stale_tasks`` (the primary-task
    reassign logic): alive + on-this-tic => keep; dead, alive-but-moved-on,
    or no ``backfill_tic`` record at all => fair game to steal. Shared
    ``keep_if`` building block for every backfill work stream (Fase 1's
    "fpp" stream now; "exominer"/"primary" streams in later Fase-1 tasks
    reuse it the same way — see routes/workers.py's ``_fpp_task_from_stream``
    for the first call site).

    ``tic_id`` is compared as ``str()`` on both sides for the same reason
    ``reassign_stale_tasks`` does: a JSON round-trip through shared_state
    can turn an int tic_id into a string (or vice versa) without changing
    its meaning.

    BOUNDED (2026-07-14): the veto expires at BACKFILL_HOLD_CEILING_SECONDS.
    "Alive and on-tic" is the worker's OWN account of itself, and a frozen
    worker keeps telling that story forever (see the constant's docstring for
    the failure modes and the run it cost). Past the ceiling we stop believing
    it and let the item be stolen — exactly what branch 1 of
    reassign_stale_tasks does to a primary task that has been assigned too long.
    """
    w = workers_state.get(worker_id)
    if not w:
        return False
    if _time.time() - w.get("last_heartbeat", 0) >= WORKER_ALIVE_WINDOW_SECONDS:
        return False
    hold_age = _backfill_hold_age(worker_id)
    if hold_age is None or hold_age > BACKFILL_HOLD_CEILING_SECONDS:
        return False
    bf = _get_worker_backfill_tic(worker_id)
    if not bf:
        return False
    owner_tic, _task_type = bf
    return str(owner_tic) == str(tic_id)


def _worker_alive_and_on_current_tic(workers_state, worker_id, tic_id) -> bool:
    """True iff *worker_id* is alive (fresh heartbeat) AND its self-reported
    heartbeat field ``current_tic`` equals *tic_id* — the PRIMARY work
    stream's ``keep_if`` (Fase 1, Task 5), mirroring branch 3 of
    ``db.reassign_stale_tasks`` exactly (the legacy call site building
    ``worker_current_tic`` from ``workers_state[wid].get("current_tic")``
    is ``worker_get_task``, routes/workers.py's legacy path).

    Deliberately NOT ``_worker_alive_and_on`` above: that helper reads the
    ``worker:backfill_tic`` hash, which is populated ONLY by
    ``_set_worker_backfill_tic`` at FPP/ExoMiner task-serve time — the
    primary (BLS) pipeline never writes that hash, so reusing it here would
    always return False and let every alive-and-working primary task be
    stolen out from under its worker the moment the stream's visibility
    timeout elapses. The primary pipeline's only per-worker TIC signal has
    always been the heartbeat's own ``current_tic`` field (trusted at the
    30-min stale-task age the same way WORKER_ALIVE_WINDOW trusts it at
    10 min — see reassign_stale_tasks branch 3's docstring) — read directly
    from ``workers_state`` here instead of the backfill hash.

    This helper is branch-3-only (alive + on-this-tic => keep). Branch 1's
    ``ABSOLUTE_ABANDON_TIMEOUT_SECONDS`` hard ceiling (a task assigned so
    long ago the worker MUST be stuck, regardless of what this helper
    reports) is deliberately NOT folded in here — this function's inputs
    don't include ``assigned_at``, and the FPP/ExoMiner ``keep_if``
    callbacks that call the sibling ``_worker_alive_and_on`` have their own
    separate semantics, so the ceiling would not belong in a function they
    also call. The primary stream's own ``keep_if``
    (routes/workers.py::_primary_task_from_stream) enforces branch-1 parity
    itself, before ever calling this helper.
    """
    w = workers_state.get(worker_id)
    if not w:
        return False
    if _time.time() - w.get("last_heartbeat", 0) >= WORKER_ALIVE_WINDOW_SECONDS:
        return False
    return _tic_str(w.get("current_tic")) == _tic_str(tic_id)


# --- FPP/ExoMiner backfill task-endpoint helpers (D12) --------------------
# docs/reviews/2026-07-02-opus48-cleanup-review.md: worker_fpp_task and
# worker_exominer_task share this reap/claim/payload skeleton verbatim. The
# per-candidate gate logic and the main "which candidates need work" query
# genuinely differ between the two (different partial indexes, different
# guard mechanism, different per-candidate side effects — FPP-uncomputable
# pre-marking vs no-metadata sentinel-marking) and stay inline at each call
# site rather than being forced into one shape.


def _reap_stale_assignments(assignments_key: str, timeout_seconds: float) -> None:
    """Drop assignment entries older than *timeout_seconds* (worker died
    silently mid-backfill, so the candidate never got released normally).

    Identical logic previously duplicated in worker_fpp_task
    (``_FPP_ASSIGNMENTS_KEY``) and worker_exominer_task
    (``_EXOMINER_ASSIGNMENTS_KEY``).
    """
    st = get_shared_state()
    now = _time.time()
    for cid_str, entry in st.hgetall_json(assignments_key).items():
        if now - entry.get("ts", 0) > timeout_seconds:
            st.hdel(assignments_key, cid_str)


def _claim_candidate(assignments_key: str, candidate_id, now: float = None) -> bool:
    """Atomically claim *candidate_id* under *assignments_key* (HSETNX).

    One caller wins; every other concurrent poller — even in a different
    server process — loses. *now* defaults to ``_time.time()`` but callers
    that claim several candidates in one poll pass the same *now* snapshot
    taken at the top of the loop, exactly matching the pre-extraction
    behaviour (all claim attempts in one poll cycle stamped with one ``ts``).
    """
    if now is None:
        now = _time.time()
    return get_shared_state().hsetnx_json(
        assignments_key, str(candidate_id), {"ts": now})


def _build_backfill_payload(candidate) -> dict:
    """Build the ``{"task": {...}}`` response shared by worker_fpp_task and
    worker_exominer_task — identical 8-field shape in both endpoints."""
    return {"task": {
        "candidate_id": candidate["id"],
        "tic_id": candidate["tic_id"],
        "candidate_num": candidate["candidate_num"],
        "period_days": candidate["period_days"],
        "t0": candidate["t0"],
        "duration_days": candidate["duration_days"],
        "depth_ppm": candidate["depth_ppm"],
        "snr": candidate["snr"],
        "n_sectors": candidate.get("n_sectors", 1),
    }}


# Process-wide MetadataCache instances, keyed by db_path. Avoids re-opening
# the SQLite handle on every request (the cache is consulted by many
# endpoints — /api/fits, /api/metadata, /api/nearby, the FPP/ExoMiner
# backfill queues, etc.). MetadataCache already serialises access with an
# internal threading.Lock, so sharing across threads is safe; what we save
# by sharing is the per-request SQLite connection open + WAL/journal init
# (~2–7 ms each on <host>) and the file-descriptor churn under
# concurrent load. The dict is keyed by db_path so unit tests that build
# multiple apps against different tmp paths each get their own instance.
_shared_metadata_caches: dict = {}
_shared_metadata_caches_lock = threading.Lock()


def _object_store():
    """Return the process-wide object store (MinIO/S3), or None if disabled.

    Thin wrapper over object_store.store() (which is itself a lazy singleton) so
    serve_fits can prefer object storage with a one-liner and no per-request
    boto3 client creation.
    """
    from exonit import object_store
    return object_store.store()


def _get_shared_metadata_cache(cfg):
    """Return a process-wide MetadataCache keyed by cfg.metadata_cache_path.

    Lazy-initialised on first call per path. Caller MUST NOT call .close()
    on the returned instance — it lives until the process exits (or the
    server is reconfigured to point at a different DB).
    """
    db_path = cfg.metadata_cache_path
    mc = _shared_metadata_caches.get(db_path)
    if mc is not None:
        return mc
    with _shared_metadata_caches_lock:
        mc = _shared_metadata_caches.get(db_path)
        if mc is None:
            from exonit.metadata_cache import MetadataCache
            mc = MetadataCache(db_path)
            _shared_metadata_caches[db_path] = mc
    return mc

# FPP backfill progress: cumulative counter of successful FPP scorings since
# server start. Snapshot-baseline semantics (`done = baseline - pending`) are
# wrong here because new candidates enter the FPP-actionable pool as the
# primary pipeline runs; pending can grow above the initial snapshot and the
# bar gets stuck at 0/baseline forever. Count actual completions instead.
# A2a: "done" stored as scalar counter "fpp_backfill:done" (incr-based).
#       "started_at" stored as scalar "fpp_backfill:started_at".
_FPP_BACKFILL_DONE_KEY = "fpp_backfill:done"
_FPP_BACKFILL_STARTED_AT_KEY = "fpp_backfill:started_at"

# Same baseline pattern for ExoMiner backfill and LoRA explanation backfill.
# The progress bars at the dashboard need a stable initial count to compute
# percent done; without this the bars would be meaningless because the
# pending counts can grow when workers add new candidates.
# A2a: "pending" scalar set-once via set_if_absent; "started_at" likewise.
# NOTE: the "max monotonic" bump (elif pending > stored) uses get_int+set_int
# under a non-atomic read-modify-write.  A race here is cosmetic (a progress
# bar baseline off by a small amount for one request) and acceptable per spec.
_EXOMINER_BASELINE_PENDING_KEY = "exominer_baseline:pending"
_EXOMINER_BASELINE_STARTED_AT_KEY = "exominer_baseline:started_at"
_LORA_BASELINE_PENDING_KEY = "lora_baseline:pending"
_LORA_BASELINE_STARTED_AT_KEY = "lora_baseline:started_at"

# Primary BLS task queue baseline. Snapshots `pending + assigned` on first
# call after server start so the UI can show "X done of Y in this run".
# The operator resets via POST /api/admin/reset_baselines when launching a new big run.
# A2a: same set_if_absent + get_int pattern as exominer/lora baselines.
_PRIMARY_BASELINE_PENDING_KEY = "primary_baseline:pending"
_PRIMARY_BASELINE_STARTED_AT_KEY = "primary_baseline:started_at"


def reset_progress_baselines() -> list:
    """Forget every dashboard baseline so the next read re-seeds it.

    The baselines above are set-once by design and nothing ever expires them,
    so one minted for a run stays frozen through the next one. Since
    `done = baseline - pending`, the new run is born already half-finished: a
    100,000-TIC lot launched on 2026-07-26 opened at "200,049 of 299,944" —
    the counters of the 300,000-TIC lot before it. The header sentence, the
    bar, the ETA and the time-remaining were all wrong for its whole life.

    Callers: POST /api/admin/reset_baselines (the dashboard's "Reset stats"
    button) and, since 2026-07-26, POST /api/discovery/scan — every new
    run_id now starts from a clean slate without anyone remembering to press
    anything. Only in-memory counters are cleared, never raw data.
    """
    st = get_shared_state()
    st.delete(_PRIMARY_BASELINE_PENDING_KEY)
    st.delete(_PRIMARY_BASELINE_STARTED_AT_KEY)
    st.delete(_EXOMINER_BASELINE_PENDING_KEY)
    st.delete(_EXOMINER_BASELINE_STARTED_AT_KEY)
    st.delete(_LORA_BASELINE_PENDING_KEY)
    st.delete(_LORA_BASELINE_STARTED_AT_KEY)
    st.set_int(_FPP_BACKFILL_DONE_KEY, 0)
    st.delete(_FPP_BACKFILL_STARTED_AT_KEY)
    # Invalidate cached dashboard responses so a stale pre-reset snapshot is
    # not served for up to the TTL. Reviewer finding (2026-06-20): the
    # /api/health/work baseline-seeding side-effect runs only on a cache
    # miss, so without this the reset is effectively deferred up to 10s.
    # Only the GLOBAL health_work entry is named here; the per-run ones
    # ("cache:health_work:<run_id>", added when the key learned about scope)
    # cannot be enumerated through this interface. Since 2026-07-31 the cache
    # is stale-while-revalidate (see _cached_json below): an un-deleted
    # per-run entry is not silently evicted after its TTL, it is served stale
    # until the next request past that TTL kicks a background refresh — so a
    # pre-reset snapshot for a per-run key can outlive its nominal TTL by one
    # more poll cycle. Acceptable: these are the orchestrator's own per-run
    # counters, re-seeded by the reset that owns the run, not a
    # correctness issue for the dashboard.
    for _ck in ("cache:health_work", "cache:backfill_fpp",
                "cache:backfill_nfpp", "cache:stats", "cache:status"):
        try:
            st.delete(_ck)
        except Exception:
            pass
    return [
        "primary_baseline", "exominer_baseline",
        "lora_baseline", "fpp_backfill_progress",
    ]

# Claude analysis for strong candidates (runs in background thread)
_claude_available = False
try:
    from exonit.claude_analysis import analyze_candidate, select_model
    if os.environ.get("ANTHROPIC_API_KEY"):
        _claude_available = True
except ImportError:
    pass

# Local verdict classifier — runs by default (no API key needed). Disable with
# EXONIT_VERDICT_DISABLED=1. Errors during predict (e.g. no model trained yet)
# are swallowed by the background thread itself.
_verdict_enabled = os.environ.get("EXONIT_VERDICT_DISABLED") != "1"

# Concurrency cap for ExoMiner Docker invocations. _run_exominer_background
# fires a new thread per worker result with candidates, and each thread spawns
# its own `docker run` ExoMiner. Without a cap, N concurrent workers reporting
# nearly-synchronously fire N containers — on 2026-05-15 this OOM'd the GCP
# server (e2-standard-4, 16 GB) because each container peaks at ~2 GB and 4-6
# concurrent saturated RAM.
#
# A2b: now a DistributedSemaphore (cross-process lease) so that N uvicorn
# worker processes share a single global cap — not N independent in-memory
# semaphores.  Lazy singleton; lease_ttl=120s (generous: a docker run rarely
# takes >2 min, and the renewer extends as long as the thread lives).
_exominer_docker_semaphore: Optional[DistributedSemaphore] = None
_exominer_docker_semaphore_lock = threading.Lock()

# Concurrency cap for SPOC-check background threads.  Each thread does a
# lightweight lightkurve search_lightcurve call against MAST and is
# idempotent (result is cached on disk).  Default 4 is safe; tune via
# EXONIT_SPOC_CHECK_MAX_CONCURRENT.  lease_ttl=60s (MAST calls are fast).
_spoc_check_semaphore: Optional[DistributedSemaphore] = None
_spoc_check_semaphore_lock = threading.Lock()


def _get_spoc_check_semaphore() -> DistributedSemaphore:
    """Return the (lazily-initialised) distributed SPOC-check semaphore."""
    global _spoc_check_semaphore
    if _spoc_check_semaphore is None:
        with _spoc_check_semaphore_lock:
            if _spoc_check_semaphore is None:
                try:
                    n = int(os.environ.get(
                        "EXONIT_SPOC_CHECK_MAX_CONCURRENT", "4"))
                except ValueError:
                    n = 4
                _spoc_check_semaphore = DistributedSemaphore(
                    get_shared_state(),
                    "spoc_check",
                    max(1, n),
                    lease_ttl=60.0,
                )
    return _spoc_check_semaphore


def _get_exominer_docker_semaphore() -> DistributedSemaphore:
    """Return the (lazily-initialised) distributed ExoMiner Docker semaphore."""
    global _exominer_docker_semaphore
    if _exominer_docker_semaphore is None:
        with _exominer_docker_semaphore_lock:
            if _exominer_docker_semaphore is None:
                try:
                    n = int(os.environ.get(
                        "EXONIT_EXOMINER_MAX_CONCURRENT", "1"))
                except ValueError:
                    n = 1
                _exominer_docker_semaphore = DistributedSemaphore(
                    get_shared_state(),
                    "exominer_docker",
                    max(1, n),
                    lease_ttl=120.0,
                )
    return _exominer_docker_semaphore


# ---------------------------------------------------------------------------
# Bounded background-task executors — one per task type, per-process.
#
# RATIONALE (incident 2026-06-04):
#   The old pattern fired one threading.Thread per worker result. Each thread
#   immediately blocked on the DistributedSemaphore (Redis polling every 0.5s).
#   Under backlog (burst of worker results) hundreds of threads piled up
#   polling Redis → 650-700 threads/process, ~5 cores of CPU thrash, 1026
#   Redis connections, server unresponsive.
#
# FIX:
#   Replace thread-per-result with a fixed-size ThreadPoolExecutor. Excess work
#   items are queued as cheap Future objects, not OS threads.  Thread count is
#   bounded by max_workers regardless of incoming result rate.
#
# SPOC executor (N_spoc × nprocs MAST connections — acceptable):
#   Pool size = EXONIT_SPOC_CHECK_MAX_CONCURRENT (default 4).
#   The DistributedSemaphore is NO LONGER needed inside _run_spoc_check_background
#   because the pool itself IS the concurrency limit (per process).  The getter
#   is left intact in case other callers reference it.
#
# ExoMiner executor (N_exo × nprocs threads contend on the global Docker lease):
#   Pool size = EXONIT_EXOMINER_MAX_CONCURRENT (default 1).
#   The DistributedSemaphore is KEPT inside _run_exominer_background so that
#   Docker containers stay globally bounded across all uvicorn worker processes.
#   With N_exo workers × nprocs processes each holding at most N_exo threads,
#   total thread count stays small and the lease contention is trivial.
#
# NOTE: executor queues are unbounded (futures are cheap). Add a BoundedSemaphore
#   wrapper around submit() if you ever need to back-pressure result ingestion.
# ---------------------------------------------------------------------------

_spoc_executor: Optional[ThreadPoolExecutor] = None
_spoc_executor_lock = threading.Lock()

_exominer_executor: Optional[ThreadPoolExecutor] = None
_exominer_executor_lock = threading.Lock()

# DV-run lookup cache used by the SPOC-gate's has-SPOC branch (see
# _run_spoc_check_background / docs/superpowers/specs/exominer-dv-gate.md).
# MastDVCache wraps a single sqlite3 connection opened with
# check_same_thread=False and guards every access with its own internal
# lock, so — same rationale as _spoc_executor above — one process-wide
# instance is safe to share across the executor pool's threads instead of
# opening a fresh sqlite connection per TIC.
_dv_cache = None
_dv_cache_lock = threading.Lock()

# SPOC-gate offline-first (docs/superpowers/specs/2026-08-03-spoc-gate-
# prefetch-offline-design.md): with EXONIT_SPOC_GATE_OFFLINE=1, an empty
# bulk index (never ingested) leaves the gate's state NULL rather than
# falling back to MAST — see _run_spoc_check_background. That is a
# preflight-checklist miss (the operator forgot to run
# scripts/prefetch_spoc_bulk.py before the run), not a per-TIC event, so it
# is logged once per process — a per-TIC warning at discovery-run scale
# would flood the log for exactly the population (every TIC) that offline
# mode exists to protect from a MAST round-trip in the first place.
_spoc_gate_offline_empty_index_warned: set = set()
_spoc_gate_offline_empty_index_warned_lock = threading.Lock()


def _warn_spoc_gate_offline_empty_index_once(kind: str, tic_id) -> None:
    """Log the "offline mode + never-ingested bulk index" warning exactly
    once per *kind* (``"lc"``/``"dv"``) per process."""
    with _spoc_gate_offline_empty_index_warned_lock:
        if kind in _spoc_gate_offline_empty_index_warned:
            return
        _spoc_gate_offline_empty_index_warned.add(kind)
    logger.warning(
        "SPOC-gate: EXONIT_SPOC_GATE_OFFLINE=1 but the bulk %s index has "
        "never been ingested (TIC %s is the first miss this process saw) — "
        "leaving has_spoc/exominer_score state unresolved (NULL) and never "
        "calling MAST, as offline mode requires. Run "
        "scripts/prefetch_spoc_bulk.py before starting the run "
        "(further occurrences of this warning are suppressed for this "
        "process; kind=%s coverage stays empty until then).",
        kind, tic_id, kind)


def _get_spoc_executor() -> ThreadPoolExecutor:
    """Return the (lazily-initialised) bounded SPOC-check executor."""
    global _spoc_executor
    if _spoc_executor is None:
        with _spoc_executor_lock:
            if _spoc_executor is None:
                try:
                    n = int(os.environ.get(
                        "EXONIT_SPOC_CHECK_MAX_CONCURRENT", "4"))
                except ValueError:
                    n = 4
                _spoc_executor = ThreadPoolExecutor(
                    max_workers=max(1, n),
                    thread_name_prefix="spoc",
                )
                # Give the shared astroquery MAST session a real socket
                # timeout so a dead MAST endpoint can't hang a check forever
                # at ssl.read (2026-08-03 outage — see exonit.mast_http).
                from exonit.mast_http import install_mast_session_timeout
                install_mast_session_timeout()
    return _spoc_executor


def _shutdown_spoc_executor_fast() -> None:
    """Cancel this process's queued SPOC checks so a recycled worker exits now.

    With EXONIT_SPOC_GATE=1 a discovery scan submits one MAST search per
    completed TIC; the queue grows faster than the pool drains it, and
    concurrent.futures' atexit hook then keeps a "shut down" uvicorn worker
    alive for hours finishing the backlog — during which the --workers
    supervisor cannot respawn it, leaving the service with no listeners
    (2026-08-01: 85-min outage window, orchestrator marked the run FAILED).

    Dropping the queue is safe: checks are idempotent, disk-cached, and any
    TIC left unresolved is re-resolved by the lazy has_spoc gate in
    worker_exominer_task. The in-flight check (one per pool thread) still
    completes — only never-started ones are cancelled.
    """
    with _spoc_executor_lock:
        ex = _spoc_executor
    if ex is None:
        return
    try:
        backlog = ex._work_queue.qsize()  # best-effort, private API
    except Exception:
        backlog = -1
    ex.shutdown(wait=False, cancel_futures=True)
    logger.info(
        "SPOC-check executor: fast shutdown, ~%d queued check(s) cancelled "
        "(lazy gate will re-resolve them)", backlog)


def _get_dv_cache():
    """Return the (lazily-initialised) shared MastDVCache for the SPOC-gate.

    Tests exercising the has-SPOC branch of _run_spoc_check_background
    MUST monkeypatch this (or the fetch_dv_runs it feeds) — the default
    MastDVCache() path lives under data/dv_runs_cache.db in the repo, and
    tests must never touch that real file or make a real MAST call.
    """
    global _dv_cache
    if _dv_cache is None:
        with _dv_cache_lock:
            if _dv_cache is None:
                from exonit.mast_dv import MastDVCache
                _dv_cache = MastDVCache()
    return _dv_cache


def _get_exominer_executor() -> ThreadPoolExecutor:
    """Return the (lazily-initialised) bounded ExoMiner executor."""
    global _exominer_executor
    if _exominer_executor is None:
        with _exominer_executor_lock:
            if _exominer_executor is None:
                try:
                    n = int(os.environ.get(
                        "EXONIT_EXOMINER_MAX_CONCURRENT", "1"))
                except ValueError:
                    n = 1
                _exominer_executor = ThreadPoolExecutor(
                    max_workers=max(1, n),
                    thread_name_prefix="exominer",
                )
    return _exominer_executor


_fpp_gate_executor: Optional[ThreadPoolExecutor] = None
_fpp_gate_executor_lock = threading.Lock()


_verdict_executor: Optional[ThreadPoolExecutor] = None
_verdict_executor_lock = threading.Lock()


def _get_verdict_executor() -> ThreadPoolExecutor:
    """Bounded executor for _run_verdict_background (same pattern as the
    ExoMiner/SPOC/FPP-gate executors). The previous thread-per-strong-candidate
    launches were the last unbounded background path left after the 2026-06-04
    incident fix."""
    global _verdict_executor
    if _verdict_executor is None:
        with _verdict_executor_lock:
            if _verdict_executor is None:
                try:
                    n = int(os.environ.get(
                        "EXONIT_VERDICT_MAX_CONCURRENT", "4"))
                except ValueError:
                    n = 4
                _verdict_executor = ThreadPoolExecutor(
                    max_workers=max(1, n),
                    thread_name_prefix="verdict",
                )
    return _verdict_executor


def _get_fpp_gate_executor() -> ThreadPoolExecutor:
    """Return the (lazily-initialised) bounded FPP-gate executor."""
    global _fpp_gate_executor
    if _fpp_gate_executor is None:
        with _fpp_gate_executor_lock:
            if _fpp_gate_executor is None:
                try:
                    n = int(os.environ.get(
                        "EXONIT_FPP_GATE_MAX_CONCURRENT", "4"))
                except ValueError:
                    n = 4
                _fpp_gate_executor = ThreadPoolExecutor(
                    max_workers=max(1, n),
                    thread_name_prefix="fpp_gate",
                )
    return _fpp_gate_executor


_preprocess_executor: Optional[ThreadPoolExecutor] = None
_preprocess_executor_lock = threading.Lock()


def _get_preprocess_executor() -> ThreadPoolExecutor:
    """Return the (lazily-initialised) bounded ExoMiner-preprocess executor.

    Minor dedup 11(e) (docs/reviews/2026-07-02-opus48-cleanup-review.md):
    replaces a raw per-request ``threading.Thread(daemon=True)`` spawn in the
    ``/api/exominer/preprocess/{tic_id}`` cache-miss path — each preprocess
    run is 3-5 min of CPU/IO, so an unbounded burst of concurrent misses
    would pile up threads the same way the pre-A2 SPOC/ExoMiner backfill
    threads did (see the ThreadPoolExecutor migration note above).
    """
    global _preprocess_executor
    if _preprocess_executor is None:
        with _preprocess_executor_lock:
            if _preprocess_executor is None:
                try:
                    n = int(os.environ.get(
                        "EXONIT_PREPROCESS_MAX_CONCURRENT", "2"))
                except ValueError:
                    n = 2
                _preprocess_executor = ThreadPoolExecutor(
                    max_workers=max(1, n),
                    thread_name_prefix="preprocess",
                )
    return _preprocess_executor


def _scan_in_flight() -> bool:
    """True when a discovery scan is actively running (SCANNING / ENQUEUEING).

    Used to pause GPU-contending ExoMiner backfill during a scan so workers
    don't fight the primary BLS pipeline for the GPU. Best-effort: any error
    (missing/half-written state file) returns False so backfill is never
    wedged off by a transient read failure.
    """
    try:
        sf = Path(os.environ.get(
            "EXONIT_PIPELINE_STATE_FILE",
            str(_PROJECT_ROOT / "data" / "discovery" / "pipeline_state.json")))
        if not sf.exists():
            return False
        return json.loads(sf.read_text()).get("state") in (
            "SCANNING", "ENQUEUEING")
    except Exception:
        return False


def _active_run_sector():
    """Return the active discovery run's TESS sector (int), or None.

    Reads the pipeline state file (written by the orchestrator / scan endpoint).
    New candidates are inserted via /api/worker/result only while the run is
    SCANNING, so we tag them with the run's ``sector`` during that phase instead
    of the historical default 1 (which conflated every home insert into the
    sector-1 partition).  Best-effort: any error, no active scan, or a missing
    sector → None, and db.add_candidate keeps the default 1 (no regression).
    """
    try:
        sf = Path(os.environ.get(
            "EXONIT_PIPELINE_STATE_FILE",
            str(_PROJECT_ROOT / "data" / "discovery" / "pipeline_state.json")))
        if not sf.exists():
            return None
        st = json.loads(sf.read_text())
        if st.get("state") != "SCANNING":
            return None
        sec = st.get("sector")
        return int(sec) if sec is not None else None
    except Exception:
        return None


# Resolve project paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_WWW_DIR = _PROJECT_ROOT / "www"
_ONNX_MODEL_PATH = _PROJECT_ROOT / "worker" / "models" / "exominer_plusplus_single.onnx"
_WORKERS_MANIFEST = _PROJECT_ROOT / "ops" / "workers.json"


# Module-level singleton for the DV XML proxy cache. Defined at module scope
# (not inside create_app) so unit tests can patch("exonit.server.get_dv_store").
_dv_store = None


def get_dv_store():
    """Return the process-wide DVXmlStore singleton (lazy-initialized)."""
    global _dv_store
    if _dv_store is None:
        from exonit.exominer_native.fetch_mast_dv import DVXmlStore
        _dv_store = DVXmlStore(
            base_dir=str(_PROJECT_ROOT / "data" / "dv_xml_cache")
        )
    return _dv_store


# Module-level singleton for the TICv8 proxy cache. Mirrors the get_dv_store
# pattern so tests can patch("exonit.server.get_ticv8_store").
_ticv8_store = None


def get_ticv8_store():
    """Return the process-wide TICv8Store singleton (lazy-initialized)."""
    global _ticv8_store
    if _ticv8_store is None:
        from exonit.exominer_native.fetch_ticv8 import TICv8Store
        _ticv8_store = TICv8Store(
            db_path=str(_PROJECT_ROOT / "data" / "ticv8_cache.db")
        )
    return _ticv8_store


# Module-level singleton for the Gaia DR3 + RUWE proxy cache. Mirrors the
# get_ticv8_store pattern so tests can patch("exonit.server.get_gaia_store").
_gaia_store = None


def get_gaia_store():
    """Return the process-wide GaiaStore singleton (lazy-initialized)."""
    global _gaia_store
    if _gaia_store is None:
        from exonit.exominer_native.fetch_gaia import GaiaStore
        _gaia_store = GaiaStore(
            db_path=str(_PROJECT_ROOT / "data" / "gaia_cache.db")
        )
    return _gaia_store


def _json_safe(obj):
    """Recursively replace NaN/Inf floats with None so FastAPI can serialize.

    The TICv8 / Gaia proxy responses frequently contain missing stellar
    parameters as NaN (pandas/MAST default). Stdlib json rejects NaN per
    spec (ValueError: Out of range float values are not JSON compliant).
    Clients treat None as "missing" and fall back to training-set medians.
    """
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _get_db_path(config=None) -> str:
    """Get database DSN/path from env, config, or default.

    A PostgreSQL DSN in ``EXONIT_DATABASE_URL`` wins over any SQLite path so a
    single env var flips the server to PG. ``make_backend``/``ExoNitDB``
    discriminate a DSN from a filesystem path (``is_pg_url``), so the same return
    value works for both backends.
    """
    if os.environ.get("EXONIT_DATABASE_URL"):
        return os.environ["EXONIT_DATABASE_URL"]
    if os.environ.get("EXONIT_DB_PATH"):
        return os.environ["EXONIT_DB_PATH"]
    if config and hasattr(config, "db_path"):
        return config.db_path
    return str(_PROJECT_ROOT / "data" / "exonit.db")


def _get_reports_dir(config=None) -> str:
    """Get reports directory from env, config, or default."""
    if os.environ.get("EXONIT_REPORTS_DIR"):
        return os.environ["EXONIT_REPORTS_DIR"]
    if config and hasattr(config, "reports_dir"):
        return config.reports_dir
    return str(_PROJECT_ROOT / "reports")


def _scorecards_dir() -> Path:
    """Return the canonical scorecards directory path.

    Module-level so tests can monkeypatch it:
        patch("exonit.server._scorecards_dir", return_value=tmp_path / "sc")
    """
    return _PROJECT_ROOT / "data" / "scorecards"


def _toi_catalog_path() -> str:
    """Return the canonical TOI catalog CSV path.

    Module-level so tests can monkeypatch it:
        patch("exonit.server._toi_catalog_path", return_value=str(tmp_csv))
    """
    return str(_PROJECT_ROOT / "data" / "toi_catalog.csv")


# P7 (docs/reviews/2026-07-02-opus48-cleanup-review.md): /api/downloads/catalog
# used to re-parse the ~8k-row TOI CSV on every request. Cached by file mtime,
# mirroring exonit.ctoi._load_toi_catalog's pattern (module-level dict keyed
# by mtime, invalidated when the file changes — e.g. after
# /api/downloads/refresh-catalog re-downloads it).
_downloads_catalog_cache: dict = {"path": None, "mtime": None, "rows": None}
_downloads_catalog_cache_lock = threading.Lock()


def _load_downloads_catalog_rows(catalog_path: str) -> list:
    """Parse the TOI catalog CSV into deduped row dicts, cached by mtime.

    Keeps the fields ``/api/downloads/catalog`` needs (tmag, period) and the
    endpoint's original first-row-per-TIC dedup semantics (unlike
    ``exonit.ctoi._load_toi_catalog``, which groups ALL rows per TIC for
    TOI-crosscheck purposes — a different shape for a different caller).

    Returns a list of ``{"tic_id", "disposition", "tmag", "period"}`` dicts,
    one per unique TIC ID (first CSV row wins, matching the endpoint's
    previous per-request behaviour).
    """
    if not os.path.exists(catalog_path):
        return []
    mtime = os.path.getmtime(catalog_path)
    with _downloads_catalog_cache_lock:
        if (_downloads_catalog_cache["path"] == catalog_path
                and _downloads_catalog_cache["mtime"] == mtime
                and _downloads_catalog_cache["rows"] is not None):
            return _downloads_catalog_cache["rows"]

    import csv
    rows = []
    seen_tics = set()
    with open(catalog_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tic = row.get("TIC ID", "").strip()
            if not tic or tic in seen_tics:
                continue
            seen_tics.add(tic)
            rows.append({
                "tic_id": tic,
                "disposition": row.get("TFOPWG Disposition", "").strip(),
                "tmag": row.get("TMag Value", ""),
                "period": row.get("Period (days)", ""),
            })

    with _downloads_catalog_cache_lock:
        _downloads_catalog_cache["path"] = catalog_path
        _downloads_catalog_cache["mtime"] = mtime
        _downloads_catalog_cache["rows"] = rows
    return rows


# --- AWS control-plane endpoint helpers (minor dedup 11(b)) ----------------

def _require_deployment(target: str):
    """Return a 403 ``JSONResponse`` if the running server isn't *target*
    (``"aws"`` or ``"home"``), else ``None``.

    Shared guard for the six ``/api/aws/*`` endpoints below — each used to
    hand-roll ``from exonit.config import deployment; if deployment() !=
    target: return JSONResponse(...)``.
    """
    from exonit.config import deployment
    if deployment() != target:
        return JSONResponse({"error": f"not_{target}_deployment"}, status_code=403)
    return None


def _aws_try(fn_name: str, *args, action_label: str = None, **kwargs):
    """Call ``exonit.aws_control.<fn_name>(*args, **kwargs)`` (deferred
    import, same timing as before) and translate any exception into a 500
    ``JSONResponse``.

    Shared by the four ``/api/aws/*`` endpoints whose failure mode is a
    plain 500 ``{"error": f"{label}_failed", "detail": str(exc)}``:
    workers/start, workers/terminate, server/start, server/stop.
    fleet/status and /api/aws/status intentionally do NOT use this — their
    contract is to degrade in place (never 500) so the dashboard still shows
    partial info when AWS is flaky.

    Returns ``(result, None)`` on success or ``(None, JSONResponse)`` on
    failure — the caller returns the JSONResponse immediately if not None.
    """
    from exonit import aws_control
    label = action_label or fn_name
    try:
        return getattr(aws_control, fn_name)(*args, **kwargs), None
    except Exception as exc:
        return None, JSONResponse(
            {"error": f"{label}_failed", "detail": str(exc)}, status_code=500)


# --- Cache-first proxy-fetch helper (minor dedup 11(c)) --------------------

class _ProxyNotFound(Exception):
    """Raised by a ``_proxy_fetch`` fetch_fn to signal a specific 404 reason
    distinct from the generic ``not_found_error`` (e.g. api_gaia's RA/Dec
    precondition, checked only on a cache miss)."""

    def __init__(self, error_code: str):
        super().__init__(error_code)
        self.error_code = error_code


def _proxy_fetch(store, key: tuple, fetch_fn, *, not_found_error: str,
                 to_response=None, mark_missing_fn=None,
                 known_missing_value=None, log_label: str = "proxy"):
    """Cache-first proxy fetch: return the cached value for *key* in *store*,
    or call *fetch_fn* on a miss, persist the result, and return it.

    Shared skeleton for api_ticv8/api_gaia (JSON stores with a plain 2-state
    cache: absent vs cached) and api_mast_dv (a bytes store whose cache
    distinguishes "never fetched" (``None``) from "confirmed missing, don't
    retry" (``known_missing_value``, e.g. ``b""``) via *mark_missing_fn*).

    Parameters
    ----------
    store: object exposing ``.get(*key)`` / ``.put(*key, data)``.
    key: tuple of positional args passed to ``store.get``/``store.put``
        (e.g. ``(tic_id,)`` or ``(tic_id, sector_run)``).
    fetch_fn: zero-arg callable performing the live fetch; returns the
        fetched data, or ``None`` if nothing exists upstream. May raise
        ``_ProxyNotFound(code)`` to signal a specific 404 reason evaluated
        only on a cache miss (before the network fetch would otherwise run).
    not_found_error: error string returned (404) when fetch_fn yields None.
    to_response: callable(data) -> FastAPI-serializable return value.
        Defaults to ``_json_safe``.
    mark_missing_fn: optional callable(*key) invoked right before the 404
        when fetch_fn returns None — persists a "known missing" sentinel so
        future requests skip the network call.
    known_missing_value: if not None, a cache HIT equal to this value is
        "confirmed missing" and short-circuits straight to the 404 (no fetch
        attempted).
    log_label: prefix used in the fetch-exception warning log line.
    """
    if to_response is None:
        to_response = _json_safe
    cached = store.get(*key)
    if cached is not None:
        if known_missing_value is not None and cached == known_missing_value:
            return JSONResponse({"error": not_found_error}, status_code=404)
        return to_response(cached)
    try:
        data = fetch_fn()
    except _ProxyNotFound as nf:
        return JSONResponse({"error": nf.error_code}, status_code=404)
    except Exception as exc:
        logger.warning("%s fetch failed %s: %s", log_label,
                       " ".join(str(k) for k in key), exc)
        return JSONResponse({"error": "fetch_failed"}, status_code=502)
    if data is None:
        if mark_missing_fn is not None:
            mark_missing_fn(*key)
        return JSONResponse({"error": not_found_error}, status_code=404)
    store.put(*key, data)
    return to_response(data)


def _run_verdict_background(db, tic_id, candidate, cnn_score):
    """Run the local verdict classifier for a strong candidate in a background
    thread. Drop-in replacement for the previous _run_claude_background.

    Stores the verdict with an empty analysis string. LoRA explanation
    generation is deferred exclusively to the closeout batch step
    (scripts/lora_via_gpuq.sh → generate_missing_explanations.py
    --include_stale), which runs after the full scan completes. This
    avoids inline HTTP calls to the explainer service (unreachable during
    runs, and unavailable on AWS workers).
    """
    try:
        from exonit.verdict_classifier import predict_verdict
        cand_num = candidate.get("candidate_num", 1)
        # Pull stellar params from stars table for richer features. n_datapoints
        # is included for v2 hierarchical model (it's one of its 32 features).
        star_row = db._query(
            "SELECT tmag, ra, dec, n_sectors, time_span_days, n_datapoints "
            "FROM stars WHERE tic_id=?",
            (tic_id,))
        # D8 (docs/reviews/2026-07-02-opus48-cleanup-review.md): pure dict
        # construction extracted to exonit.verdict_io so offline recovery
        # scripts (recompute_verdict.py) can build the identical shape
        # without re-typing this by hand.
        star_info = build_star_info(tic_id, star_row[0] if star_row else None)
        vetting = default_vetting_result()
        # FFI/SPOC routing (docs/superpowers/specs/2026-09-19-verdict-v3-tfopwg
        # -design.md §5): resolved once per candidate via the same shared
        # metadata cache /api/metadata uses. resolve_data_source() itself
        # never raises (logs + defaults to 'spoc' on any lookup failure);
        # the cache-getter call is wrapped too, so even a cache-construction
        # failure (e.g. bad DB path) cannot add a new verdict-scoring
        # failure mode — it just falls back to 'spoc' (v2.1).
        data_source = "spoc"
        try:
            from exonit.config import ExoNitConfig
            from exonit.verdict_routing import resolve_data_source
            data_source = resolve_data_source(
                tic_id, _get_shared_metadata_cache(ExoNitConfig()))
        except Exception as route_exc:
            logger.warning("Verdict routing failed TIC %s, defaulting to "
                           "spoc: %s", tic_id, route_exc)
        result = predict_verdict(candidate, vetting, star_info=star_info,
                                 data_source=data_source)
        if not result:
            return

        # analysis is intentionally left empty here; the LoRA explainer
        # fills it in at closeout via generate_missing_explanations.py
        # --include_stale (which catches both NULL and non-[lora-v2] text).
        analysis = ""

        db.update_candidate_claude(
            db._query(
                "SELECT id FROM candidates WHERE tic_id = ? AND candidate_num = ?",
                (tic_id, cand_num)
            )[0]["id"],
            result["verdict"],
            analysis,
        )
        logger.info("Verdict TIC %s #%s: %s (%.2f) [lora deferred to closeout]",
                    tic_id, cand_num, result["verdict"], result["confidence"])
    except Exception as e:
        logger.warning("Verdict failed TIC %s: %s", tic_id, e)


def _run_exominer_background(db, tic_id, candidates):
    """Score ExoMiner TCEs for a TIC and update the matching BLS candidates.

    Enabled by setting the environment variable ``EXONIT_ENABLE_EXOMINER_AUTO=1``.
    The server-side service handles Docker + caching — cache hits are effectively
    free. Misses take a few minutes on CPU and happen in this background thread.

    Wrapped in a process-wide semaphore (see ``_get_exominer_docker_semaphore``)
    so concurrent worker results do not fire unbounded `docker run` invocations.
    """
    with _get_exominer_docker_semaphore():
        _run_exominer_background_inner(db, tic_id, candidates)


def _run_exominer_background_inner(db, tic_id, candidates):
    try:
        from exonit.exominer_service import ExoMinerService, match_tce_to_period
        svc = ExoMinerService()
        try:
            result = svc.predict_tic(str(tic_id))
        finally:
            svc.close()
        if not result.tces:
            return
        for c in candidates:
            cand_num = c.get("candidate_num")
            period = c.get("period_days")
            if cand_num is None or period is None:
                continue
            matched = match_tce_to_period(result.tces, float(period))
            if matched is None:
                continue
            row = db._query(
                "SELECT id FROM candidates WHERE tic_id = ? AND candidate_num = ?",
                (str(tic_id), cand_num)
            )
            if not row:
                continue
            db.update_candidate_exominer(
                row[0]["id"],
                score=matched.score,
                tce_period=matched.period,
                sector_run=result.sector_run,
            )
        logger.info("ExoMiner TIC %s: scored %d TCEs (run %s, cache=%s)",
                    tic_id, len(result.tces), result.sector_run, result.from_cache)
    except Exception as e:
        logger.warning("ExoMiner failed TIC %s: %s", tic_id, e)


def _run_spoc_check_background(db, tic_id, candidates):
    """Check MAST for 2-min SPOC + DV availability and set the ExoMiner-SPOC
    sentinel for a TIC's candidates, PER CANDIDATE SECTOR.

    Enabled by setting the environment variable ``EXONIT_SPOC_GATE=1``.
    Off by default — activate on the server before a discovery run to avoid
    ~160k DV-fetch round-trips for TICs that are FFI-only (TGLC).

    Offline-first resolution (docs/superpowers/specs/2026-08-03-spoc-gate-
    prefetch-offline-design.md): both MAST questions below (has-SPOC, DV
    runs) are first checked against the local bulk index
    (``exonit.spoc_bulk``, ingested by ``scripts/prefetch_spoc_bulk.py``
    from ``tesscurl_sector_{N}_lc.sh``/``_dv.sh``) before ever considering
    MAST. The two questions are resolved INDEPENDENTLY (a TIC's sector may
    be LC-bulk-ingested but not yet DV-bulk-ingested, or vice versa) with
    the same asymmetric contract for each:

        bulk says PRESENT      -> trust it, no MAST (always reliable: a
                                   positive hit only requires ONE sector's
                                   bulk data to be ingested, not full
                                   coverage).
        bulk says ABSENT       -> only a verdict on its own if
                                   EXONIT_SPOC_GATE_OFFLINE=1 (the operator
                                   is asserting the bulk ingest is COMPLETE
                                   for every sector this run can reach —
                                   ``exonit.spoc_bulk.has_spoc_bulk``'s
                                   docstring has the full reasoning). Without
                                   the flag, "absent" is inconclusive and
                                   falls back to the on-demand MAST path
                                   below, unchanged.
        bulk index never
        ingested (empty)       -> same fallback to on-demand MAST UNLESS
                                   EXONIT_SPOC_GATE_OFFLINE=1, in which case
                                   MAST must never be called at all: the
                                   state is left exactly as found (the
                                   existing fail-soft contract) and a
                                   once-per-process warning is logged (see
                                   _warn_spoc_gate_offline_empty_index_once)
                                   — this is a preflight miss, not a
                                   per-candidate failure.

    For ``dv_runs_bulk`` specifically (returns a list, not a bool) "PRESENT"
    means non-empty and "ABSENT" means ``[]`` (the DV bulk index IS ingested
    but has no report for this TIC) — ``None`` is the separate "index never
    ingested at all" case. All three ``dv_runs_bulk`` sources (per-sector
    ``tesscurl_sector_{N}_dv.sh`` AND multisector
    ``tesscurl_multisector_s{START}-s{END}_dv.sh``, see
    ``exonit.spoc_bulk.ingest_multisector_dv``) feed the same table, so a
    non-empty result here is reliable regardless of which kind of script
    produced it.

    Bulk verdicts are NEVER written to the per-TIC
    ``data/ffi_cache/.spoc_check/<tic>.json`` cache — the bulk index is
    itself the persistent source, and writing millions of extra cache files
    for TICs the bulk path already answered would reproduce exactly the
    "millions of small files" problem the bulk design was hoping to avoid
    on the on-demand side (gotcha_find_glob_repo_tglc_cache_millions).

    Decision logic (docs/superpowers/specs/exominer-dv-gate-sector.md —
    fixing a premise error in docs/superpowers/specs/exominer-dv-gate.md):

        no SPOC (FFI-only)          -> has_spoc=0, all candidates cnn>=0.7 -> -1
        has SPOC, NO DV report AT ALL
            (fetch_dv_runs returns [])-> has_spoc=1, all candidates cnn>=0.7 -> -1
        has SPOC, SOME DV report(s) -> has_spoc=1, resolved PER CANDIDATE:
            candidate's own sector covered by a run     -> NULL (queue-eligible)
                                                            (+ retracts a stale -1)
            candidate's own sector NOT covered by a run -> -1

    Premise error fixed here (docs/superpowers/specs/exominer-dv-gate-sector.md,
    2026-07-14): the previous version of this gate, once it confirmed the TIC
    had ANY DV report, cleared -1 / left NULL for EVERY cnn>=0.7 candidate of
    that TIC — treating "has DV somewhere" as "has DV for OUR candidate". A
    DV report only covers the sector RANGE SPOC ran it over (see
    ``exonit.mast_dv.parse_dv_filename`` — ``(start, end)`` tuples); a TIC
    can have DV for sector 12 and none for sector 14, and ExoMiner-SPOC can
    only score a TCE against a report covering ITS sector. Measured against
    real worker logs after re-queuing 10,230 candidates recovered under the
    old "any DV" premise: 5,138 came back "no DV product" (DV existed, just
    not for that candidate's sector), 1,686 "no TCE matched", 1,557
    "preprocess failed" (e.g. "No LC data read for TIC ... sector_run
    12-12") — only ~7 got a real score. The fix below resolves the has-DV
    branch per candidate against ``candidates.sector``
    (reference_candidates_sector_column_semantics — the real TESS sector,
    not the historical AWS-dedup quirk) instead of per-TIC.

    The prior premise error this whole function already closed once (still
    true, unaffected by the fix above): the old-old version stopped at "has
    SPOC?" and left every has-SPOC TIC's candidates NULL regardless of DV at
    all. SPOC only runs DV where its own pipeline found a TCE, so most
    has-SPOC TICs never get one — those candidates queued into the ExoMiner
    backfill, got a 404 from MAST one by one, and wasted the round-trip (see
    gotcha_exominer_backfill_spoc_gate_hole). Worse: the bulk sweep
    db.mark_unconfirmed_ffi_candidates() marks -1 on any TIC whose has_spoc
    was still unresolved at EXOMINER_BACKFILL entry, and if THIS gate later
    resolved that same TIC to has-SPOC-with-DV, the sweep's -1 was never
    retracted — an irreversible loss of the ExoMiner-SPOC signal for that
    candidate (measured 2026-07-13: 10,230 candidates this way). This
    function fixes both: it only ever writes a NEW -1 once it has confirmed
    there is no DV to score THAT candidate against (by sector), and when it
    confirms DV covering a candidate's sector IS present it retracts any
    stale -1 the sweep left behind for that same candidate.

    The DV lookup (exonit.mast_dv.fetch_dv_runs) only runs for TICs that
    have SPOC — FFI-only TICs never pay it. In discovery FFI runs, has-SPOC
    TICs are a small minority, so the extra MAST round-trip is marginal on
    the hot path; it is timed and logged per TIC (dv_fetch_seconds) so the
    real added cost is visible in production logs. It is ONE fetch per TIC
    regardless of how many candidates the TIC has — the per-candidate
    resolution below reuses the already-fetched ``runs`` list purely in
    Python (``exonit.mast_dv.dv_runs_cover_sector``), no extra MAST or cache
    round-trips per candidate.

    Fail-soft (spec requirement): if the DV lookup raises or times out, ALL
    candidates are left exactly as found (NULL or a pre-existing -1) and a
    warning is logged — no sentinel is guessed. A candidate whose own
    ``sector`` cannot be read (should not happen — the column defaults to 1,
    see reference_candidates_sector_column_semantics — but this stays
    fail-soft rather than assume) is likewise left untouched, per-candidate,
    with a warning. An unresolved state is safe (a later gate run,
    scripts/fix_exominer_spurious_sentinel.py, or
    scripts/check_exominer_sentinel_consistency.py can resolve it later);
    writing a sentinel on a failed lookup would risk introducing a NEW
    spurious -1, which is exactly the class of bug this function exists to
    close.

    Idempotent: re-running this on an already-resolved TIC changes nothing
    except retracting a spurious -1, which is the fix itself.

    Concurrency is bounded by the _spoc_executor pool
    (EXONIT_SPOC_CHECK_MAX_CONCURRENT, default 4) — the old
    DistributedSemaphore wrapper has been removed; the pool IS the limit.
    search_lightcurve results are cached in data/ffi_cache/.spoc_check/
    <tic>.json; DV lookups are cached in data/dv_runs_cache.db
    (see _get_dv_cache / exonit.mast_dv.MastDVCache) — subsequent calls for
    the same TIC are free either way.
    """
    try:
        from exonit.ffi_download import is_ffi_only_at_mast
        from exonit.mast_http import call_with_deadline
        from exonit.spoc_bulk import dv_runs_bulk, has_spoc_bulk
        offline = os.environ.get("EXONIT_SPOC_GATE_OFFLINE") == "1"
        cache_dir = _PROJECT_ROOT / "data" / "ffi_cache"

        bulk_has_spoc = has_spoc_bulk(int(tic_id))
        if bulk_has_spoc is True:
            ffi_only = False
        elif bulk_has_spoc is False and offline:
            ffi_only = True
        elif bulk_has_spoc is None and offline:
            _warn_spoc_gate_offline_empty_index_once("lc", tic_id)
            return  # fail-soft: state left exactly as found, no MAST call
        else:
            # bulk_has_spoc is False (inconclusive without the offline
            # assertion) or None (bulk never ingested, offline not set) —
            # unchanged on-demand MAST fallback.
            ffi_only = call_with_deadline(
                is_ffi_only_at_mast, int(tic_id), cache_dir,
                name=f"spoc-check-{tic_id}")
        has_spoc_val = 0 if ffi_only else 1
        db.set_star_has_spoc(tic_id, has_spoc_val)

        if ffi_only:
            n = db.mark_ffi_only_candidates(tic_id)
            logger.info("SPOC-gate TIC %s: FFI-only, marked %d candidate(s) "
                        "exominer_score=-1", tic_id, n)
            return

        # Has SPOC: the sentinel now depends on whether SPOC ever published
        # a DV report for this TIC, not just on 2-min availability. Same
        # bulk-first, offline-first resolution as has_spoc above — resolved
        # INDEPENDENTLY of how has_spoc itself was determined (a TIC's
        # sector may be LC-bulk-ingested but not yet DV-bulk-ingested).
        try:
            from exonit.mast_dv import dv_runs_cover_sector, fetch_dv_runs
            from exonit.mast_http import call_with_deadline
            t0 = _time.monotonic()
            bulk_runs = dv_runs_bulk(int(tic_id))
            if bulk_runs:
                # Non-empty -> reliable positive, same reasoning as
                # has_spoc_bulk's True: a real DV hit only requires ONE
                # ingested sector/multisector script, not full coverage.
                runs = bulk_runs
            elif bulk_runs is not None and offline:
                # [] -> DV bulk IS ingested and this TIC has no report in
                # it. Only trusted as a final "no DV" verdict under the
                # operator's completeness assertion (mirrors
                # has_spoc_bulk's False branch) — without the flag this
                # is inconclusive (see the final `else` below).
                runs = bulk_runs
            elif bulk_runs is None and offline:
                _warn_spoc_gate_offline_empty_index_once("dv", tic_id)
                return  # fail-soft: state left exactly as found, no MAST call
            else:
                # bulk_runs is [] (inconclusive without the offline
                # assertion) or None (bulk never ingested, offline not
                # set) — unchanged on-demand MAST fallback.
                runs = call_with_deadline(
                    fetch_dv_runs, tic_id, cache=_get_dv_cache(),
                    name=f"dv-fetch-{tic_id}")
            dv_fetch_seconds = _time.monotonic() - t0
        except Exception as dv_exc:
            logger.warning(
                "SPOC-gate TIC %s: has SPOC but DV lookup failed (%s: %s) "
                "— leaving exominer_score state unchanged (fail-soft)",
                tic_id, type(dv_exc).__name__, dv_exc)
            return

        if not runs:
            # No DV report at all -> no candidate of this TIC, in any
            # sector, can ever be scored. Correctly a TIC-level decision.
            n = db.mark_spoc_no_dv_candidates(tic_id)
            logger.info(
                "SPOC-gate TIC %s: has SPOC, no DV report, marked %d "
                "candidate(s) exominer_score=-1 [dv_fetch=%.2fs]",
                tic_id, n, dv_fetch_seconds)
            return

        # Some DV exists — but possibly not for every sector this TIC's
        # candidates were found in. Resolve per candidate against its own
        # `sector` (reference_candidates_sector_column_semantics).
        rows = db.get_candidates(tic_id)
        mark_ids, clear_ids, skipped = [], [], 0
        for row in rows:
            cnn = row.get("cnn_score")
            if cnn is None or cnn < 0.7:
                continue
            sector = row.get("sector")
            if sector is None:
                skipped += 1
                continue
            covered = dv_runs_cover_sector(runs, int(sector))
            score = row.get("exominer_score")
            if covered:
                if score == -1:
                    clear_ids.append(row["id"])
                # NULL or a real score -> already correct, no-op.
            else:
                if score is None:
                    mark_ids.append(row["id"])
                # already -1 (correct) or a real score -> no-op.

        n_cleared = db.clear_spurious_minus_one_candidates_for_ids(clear_ids)
        n_marked = db.mark_spoc_no_dv_candidates_for_ids(mark_ids)
        logger.info(
            "SPOC-gate TIC %s: has SPOC+DV (%d run(s)), resolved %d "
            "candidate(s) per-sector -> %d marked -1 (sector not covered), "
            "%d spurious -1 retracted (sector covered) [dv_fetch=%.2fs]",
            tic_id, len(runs), len(rows), n_marked, n_cleared,
            dv_fetch_seconds)
        if skipped:
            logger.warning(
                "SPOC-gate TIC %s: %d candidate(s) had no sector recorded "
                "— left unresolved (fail-soft)", tic_id, skipped)
    except Exception as e:
        logger.warning("SPOC-gate failed TIC %s: %s", tic_id, e)


def _run_fpp_gate_background(db, tic_id, mc, berger):
    """Eager FPP-gate: pre-mark FPP-uncomputable candidates at TIC completion.

    Called from worker_task_result via the bounded _fpp_gate_executor when
    EXONIT_FPP_GATE=1.  Mirrors _run_spoc_check_background in structure.

    Uses target_mass_is_uncomputable (same function as the lazy gate in
    worker_fpp_task) to decide whether the target star has usable stellar
    mass.  Decision is bit-identical to what the worker would compute.

    If uncomputable: calls db.mark_fpp_uncomputable_for_tic which batch-marks
    all queue-eligible candidates (snr>=10, cnn>=0.7 OR exominer_ffi_score
    >=0.5, fpp IS NULL — OR-gate 2026-07-11) with fpp=-1/nfpp=-1/
    verdict='failed', saving those workers a full FITS download + detrend
    round-trip.

    Errors are caught and printed; the gate must NEVER break result posting.
    """
    try:
        if target_mass_is_uncomputable(mc, berger, tic_id):
            n = db.mark_fpp_uncomputable_for_tic(tic_id)
            logger.info("FPP-gate TIC %s: uncomputable mass, pre-marked %d "
                        "candidate(s) fpp=-1", tic_id, n)
            # No mass → no FPP, whatever the depth says. The Rp-gate below would
            # only re-mark rows this call has already taken, so skip it.
            return
        logger.info("FPP-gate TIC %s: mass computable, leaving FPP queue "
                    "intact", tic_id)
    except Exception as e:
        logger.warning("FPP-gate failed TIC %s: %s", tic_id, e)
        return

    # The star is computable — but the individual candidates may still be
    # geometrically impossible as planets. Second gate, same executor.
    _run_companion_gate_background(db, tic_id, mc)


def resolve_star_radius(cache, tic_id):
    """R* (R☉) for *tic_id* from the metadata cache, or None.

    Deliberately the SAME source and the same lookup as step 1 of
    target_mass_is_uncomputable (the nearby_stars row for the target itself),
    so the mass gate and the radius gate can never disagree about which star
    they are talking about.

    Berger is NOT consulted here on purpose: the local Berger table holds ~4k
    entries and covers none of the discovery targets measured on 2026-07-14, so
    reading it would add a lookup and change nothing. If it is ever filled in,
    fold it in HERE and both gates inherit it at once.

    None (missing entry, missing/NaN radius) means "unknown" — every caller
    must fail open on it. An absent catalogue row is not evidence of anything.
    """
    import math

    cached = cache.get_nearby_stars(tic_id)
    if not cached:
        return None
    stars, _gal_b = cached
    target = next((s for s in stars
                   if str(s.get("ID", "")) == str(tic_id)), None)
    if target is None:
        return None
    rad = target.get("rad")
    try:
        rad = float(rad)
    except (TypeError, ValueError):
        return None
    if math.isnan(rad) or rad <= 0:
        return None
    return rad


def _run_companion_gate_background(db, tic_id, mc):
    """Eager Rp-gate: pre-mark candidates whose depth implies a STELLAR
    companion for this host star (2026-07-14).

    Runs at TIC completion, right after the mass gate, from the same bounded
    executor. The mass gate catches the stars TRICERATOPS CANNOT run on; this
    one catches the candidates it SHOULD NOT be run on — the ones that are
    eclipsing binaries by simple geometry, whatever TRICERATOPS would say.

    Why this matters (lot 20260714T175216): 91% of the candidates that reach
    TRICERATOPS imply a companion above 2 R_Jup, median 7.1 R_Jup. The queue
    was 11x bigger than the physics allows.

    Fail-open and fail-soft: no radius → nothing marked; any error → logged and
    swallowed. The gate must never break result posting, and must never turn
    missing data into a verdict.
    """
    try:
        r_star = resolve_star_radius(mc, tic_id)
        if r_star is None:
            logger.info("Rp-gate TIC %s: no stellar radius — nothing gated",
                        tic_id)
            return
        max_depth = stellar_gates.max_planetary_depth_ppm(r_star)
        if max_depth is None:
            return
        n = db.mark_fpp_stellar_companion_for_tic(tic_id, max_depth)
        if n:
            logger.info(
                "Rp-gate TIC %s: R*=%.2f R☉ → a planet cannot exceed %.0f ppm; "
                "pre-marked %d candidate(s) as stellar companions",
                tic_id, r_star, max_depth, n)
    except Exception as e:
        logger.warning("Rp-gate failed TIC %s: %s", tic_id, e)


def derive_mass_from_logg_rad(logg, rad):
    """Recover stellar mass (in Msun) from log g (dex, CGS) and R* (Rsun).

    Uses g = GM/R² with logg_sun = 4.438 dex (CGS). Stop-gap before the
    Berger+2023 catalog integration (#10) for TIC v8 targets whose mass
    field is null but logg and rad are present — common for evolved /
    subgiant stars. Returns None if either input is missing/invalid.
    """
    if logg is None or rad is None:
        return None
    try:
        return float(rad) ** 2 * 10 ** (float(logg) - 4.438)
    except (TypeError, ValueError):
        return None


def target_mass_is_uncomputable(cache, berger_catalog, tic_id: str) -> bool:
    """Return True when the target star has no usable stellar mass.

    Replicates EXACTLY the three-step enrichment that /api/nearby applies
    before serving params to the FPP worker, so server-side and worker-side
    decisions are bit-identical:

      1. Raw cache lookup  (nearby_stars where neighbor_id == tic_id).
      2. Berger+2023 overlay (overrides mass/rad/Teff when present).
      3. derive_mass_from_logg_rad fallback  (uses logg from metadata).

    Returns True  → mass is NaN/None after all three steps → worker would
                    return fpp=-1 immediately; safe to pre-mark.
    Returns False → mass is a finite number; FPP is calculable.
    Returns False → target not in nearby_stars cache (cannot determine;
                    safest to let the worker decide).

    Called only when EXONIT_FPP_GATE=1 so the default codepath is
    unchanged.  'cache' is a MetadataCache instance; 'berger_catalog' is
    the module-level BergerCatalog instance from create_app.
    """
    import math

    # Step 1: fetch target row from nearby_stars cache.
    cached = cache.get_nearby_stars(tic_id)
    if not cached:
        # No nearby-stars entry at all — can't determine; let worker decide.
        return False
    stars, _gal_b = cached

    # Find the target itself (neighbor_id == tic_id).
    target = None
    for s in stars:
        if str(s.get("ID", "")) == str(tic_id):
            target = s
            break
    if target is None:
        # Target row absent in nearby_stars list — can't determine.
        return False

    # Helper: treat float NaN and None as "missing".
    def _missing(v):
        if v is None:
            return True
        try:
            return math.isnan(float(v))
        except (TypeError, ValueError):
            return True

    mass = target.get("mass")
    rad  = target.get("rad")
    teff = target.get("Teff")

    # Step 2: Berger+2023 overlay — cheap dict lookup (catalog in memory).
    if berger_catalog is not None:
        try:
            bg = berger_catalog.lookup_by_tic(int(tic_id))
        except (ValueError, TypeError):
            bg = None
        if bg is not None:
            mass = bg.get("mass")
            rad  = bg.get("rad")
            teff = bg.get("teff")

    # Step 3: derive mass from logg + rad when mass is still missing.
    # logg comes from the metadata cache's main table (not nearby_stars).
    if _missing(mass) and not _missing(rad):
        entry = cache.get_entry(tic_id)
        logg = entry.get("logg") if entry else None
        derived = derive_mass_from_logg_rad(logg, rad)
        if derived is not None:
            mass = derived

    # Uncomputable = ANY of mass / rad / Teff is missing/NaN.
    # Worker condition (fpp_native.py:~1182-1190):
    #   if np.isnan(M_s) or np.isnan(R_s) or np.isnan(Teff): return -1
    # M_s+R_s are both needed together (logg computation, ~line 1189) and
    # Teff is needed for limb-darkening (~line 1190), so a single absent
    # parameter causes the worker to skip with -1.  The gate mirrors this
    # OR condition exactly: if any one of the three is absent the FPP call
    # would produce -1 anyway, so we mark it uncomputable server-side.
    return _missing(mass) or _missing(rad) or _missing(teff)


_exominer_allowlist_cache: dict = {"mtime": 0.0, "names": None, "src": ""}
_exominer_allowlist_lock = threading.Lock()


def _load_exominer_allowlist() -> Optional[set[str]]:
    """Return the current ExoMiner backfill allowlist, or None for 'allow all'.

    Resolution order (first hit wins):
    1. File at ``EXONIT_EXOMINER_WORKERS_FILE`` (one worker_name per line,
       blank lines and ``#``-comments ignored). Reloaded automatically when
       the file's mtime changes — no server restart needed to add or remove
       workers. **Fail-closed:** if the path is set but the file is missing
       or unreadable, returns an empty set (denies all) rather than falling
       back to env var, because silently allowing every worker if the file
       gets deleted would risk re-introducing the period-alias damage.
    2. Env var ``EXONIT_EXOMINER_WORKERS`` (comma-separated). Read once at
       startup, restart required to change.
    3. Neither set → returns None → endpoint allows every worker.
    """
    path = os.environ.get("EXONIT_EXOMINER_WORKERS_FILE", "").strip()
    if path:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        with _exominer_allowlist_lock:
            if (
                _exominer_allowlist_cache["src"] != path
                or _exominer_allowlist_cache["mtime"] != mtime
            ):
                names: set[str] = set()
                if mtime > 0:
                    try:
                        with open(path) as f:
                            for line in f:
                                s = line.strip()
                                if s and not s.startswith("#"):
                                    names.add(s)
                    except OSError:
                        names = set()
                _exominer_allowlist_cache.update(
                    {"src": path, "mtime": mtime, "names": names}
                )
            return _exominer_allowlist_cache["names"]
    raw = os.environ.get("EXONIT_EXOMINER_WORKERS", "").strip()
    if not raw:
        return None
    return {n.strip() for n in raw.split(",") if n.strip()}


def _load_ffi_for_tic(tic_id: str) -> Optional[list]:
    """Return FFI light curves for *tic_id* from the Phase 2.5 HDF5 cache.

    Each entry in the returned list is a dict with keys:
      sector (int), source (str), time (list[float]),
      flux (list[float]), quality (list[int]).

    Returns None if the TIC is not in the cache or the cache does not exist.
    """
    import h5py

    h5_path = _PROJECT_ROOT / "data" / "training" / "ffi_dataset_v1.h5"
    if not h5_path.exists():
        return None
    sectors: list = []
    try:
        with h5py.File(h5_path, "r") as f:
            grp = f.get(f"light_curves/{tic_id}")
            if grp is None:
                return None
            for src in grp.keys():
                for sec_str in grp[src].keys():
                    s = grp[src][sec_str]
                    sectors.append({
                        "sector": int(sec_str.lstrip("0") or "0"),
                        "source": src,
                        "time": s["time"][:].tolist(),
                        "flux": s["flux"][:].tolist(),
                        "quality": (
                            s["quality"][:].tolist()
                            if "quality" in s
                            else [0] * len(s["time"])
                        ),
                    })
    except (KeyError, OSError, ValueError):
        return None
    return sectors or None


def _expected_instances(name: str, manifest_path: Path = None) -> int:
    """Return the expected number of worker instances for *name*.

    Reads ops/workers.json (or *manifest_path* if provided) and returns the
    ``instances`` field for the matching worker entry (default 1).  Returns 1
    on any error (absent manifest, missing key, bad JSON) so callers degrade
    gracefully.
    """
    path = manifest_path if manifest_path is not None else _WORKERS_MANIFEST
    try:
        manifest = json.loads(path.read_text())
        for w in manifest.get("workers", []):
            if w.get("name") == name:
                return int(w.get("instances", 1))
    except Exception:
        pass
    return 1


def _campaign_worker_names(manifest_path: Path = None) -> list:
    """Return the ordered worker names the "Start workers" button targets.

    Read live from ops/workers.json — the same source ``bin/farmctl`` and
    :func:`_expected_instances` use — so adding a worker there takes effect
    with no code edit and no server restart.  Returns ``[]`` on any error
    (absent/malformed manifest); farmctl would fail on the same file anyway,
    so a hardcoded fallback list would buy nothing (its names could not be
    launched without their ops/workers.json entry).
    """
    path = manifest_path if manifest_path is not None else _WORKERS_MANIFEST
    try:
        manifest = json.loads(path.read_text())
        return [w["name"] for w in manifest.get("workers", []) if w.get("name")]
    except Exception:
        return []


def _count_online_instances(
    name: str,
    workers_state_values,
    now: float,
    offline_threshold: float = None,
) -> int:
    """Count how many instances of worker *name* are currently online.

    A worker entry is considered online when its ``last_heartbeat`` is within
    *offline_threshold* seconds of *now* (defaults to WORKER_OFFLINE_SECONDS).

    Matches entries whose ``name`` field equals *name* exactly **or** matches
    the suffixed pattern ``<name>-<positive-integer>`` (the multi-instance
    naming convention produced by farmctl / agent.py).
    """
    threshold = offline_threshold if offline_threshold is not None else WORKER_OFFLINE_SECONDS
    pattern = re.compile(rf"^{re.escape(name)}-\d+$")
    count = 0
    for w in workers_state_values:
        wname = w.get("name", "")
        if wname == name or pattern.fullmatch(wname):
            if now - w.get("last_heartbeat", 0) < threshold:
                count += 1
    return count


def _lora_workload_counts(db, strong_only: bool) -> tuple[int, int]:
    """Return (lora_stale, lora_missing) for the LoRA progress bar.

    When *strong_only* is True (EXONIT_LORA_STRONG_ONLY=1), both counts are
    restricted to rows with ``claude_verdict = 'planet_candidate'`` — matching
    the ``--strong-only`` policy of ``generate_missing_explanations.py``.

    When False the behaviour is identical to before this flag was introduced.
    """
    verdict_filter_bare = " AND claude_verdict = 'planet_candidate'" if strong_only else ""
    verdict_filter_aliased = " AND c.claude_verdict = 'planet_candidate'" if strong_only else ""

    lora_stale = db._query_scalar(
        f"""SELECT COUNT(*) FROM candidates
           WHERE claude_analysis IS NOT NULL
             AND length(claude_analysis) > 80
             AND claude_analysis NOT LIKE '[lora-v2]%'"""
        + verdict_filter_bare)
    # OR-gate (2026-07-11, docs/reviews/injection_e2e_results_2026-07-11.md +
    # docs/reviews/ffi_cnn_investigation_2026-07-11.md §5): candidates now
    # reach claude_verdict='planet_candidate' via cnn_score>=0.7 OR
    # exominer_ffi_score>=0.5 (work_producers.verdict_eligible). A plain
    # cnn_score>=0.7 filter here would undercount em_ffi-rescued
    # planet_candidate rows that generate_missing_explanations.py's
    # --strong-only mode (which gates only on claude_verdict, not cnn) DOES
    # pick up — keeping this in lockstep with that population is the whole
    # point of the progress-bar count.
    lora_missing = db._query_scalar(
        f"""SELECT COUNT(*) FROM candidates c
           JOIN stars s ON c.tic_id = s.tic_id
           WHERE s.status = 'completed'
             AND (c.cnn_score >= 0.7 OR c.exominer_ffi_score >= 0.5)
             AND (c.claude_analysis IS NULL
                  OR length(coalesce(c.claude_analysis,'')) <= 80)"""
        + verdict_filter_aliased)
    return lora_stale, lora_missing


# Cache for the discovery work-list size (canonical CSV + TGLC index). The list
# is ~532k+ TICs and reloading it on every /api/discovery/sectors poll cost
# ~0.6-4.6 s. The size only changes when the canonical CSV changes or new TGLC
# rows are indexed (a download), so we cache it.
#
# History / why TTL+row-count and not the file signature:
#   The original cache keyed on the TGLC DB file's (st_size, st_mtime). During a
#   download the DB file is written constantly, so its mtime/size churned every
#   poll → the key never matched → the full ~532k work list was rebuilt on EVERY
#   /api/discovery/sectors poll, hammering the UI/selector. The file signature is
#   far too sensitive: a WAL checkpoint or an unrelated sector's insert also
#   changes it even though THIS sector's size is unchanged.
#
# New invalidation = TTL + per-sector row-count:
#   * Key is stable: (canonical_csv, tglc_index, sector). The cache entry
#     survives across polls regardless of DB writes.
#   * Within _WORK_LIST_TTL_S of the last full compute we return the cached size
#     WITHOUT touching the DB at all — the hot per-poll path stays ~ms even mid
#     download.
#   * After the TTL we re-check the cheap invalidation signal: the canonical
#     CSV's (size, mtime) and the COUNT(*) of TGLC rows for THIS sector
#     (index-backed by ix_tglc_sector → ~ms, not a full table scan). If neither
#     changed, the work-list size cannot have changed, so we refresh the TTL and
#     return the cached size without the ~532k reload. Only a real change in the
#     canonical file or this sector's row count triggers the full recompute.
# Result: a running download no longer reshuffles the selector every poll, and
# correctness is preserved (the count is the exact size-determining signal).
#
# Q7 (perf review 2026-09-26, docs/reviews/2026-09-26-enqueue-path-
# performance-review.md): this used to be a plain per-process dict — with
# ``uvicorn --workers 4`` (A3), each of the 4 processes keeps its OWN copy,
# so a TTL miss (every 30s) triggered the cheap-but-not-free row-count
# re-validation independently in each process (measured 9,645 calls / 7h,
# ~112ms average, 18% of the exonit_meta load in the review's
# pg_stat_statements window). Moved into shared_state (Redis in prod,
# MemoryBackend in tests) — same pattern as Q1's gap-stats cache above —
# so all 4 processes share one cached value and the TTL/row-count
# re-validation runs once per interval, not up to 4x. Same
# stale-while-in-TTL semantics, now correct across processes.
#
# TTL 30 -> 300 s (2026-10-05): even shared, the re-validation still ran once
# per sector per 30 s — the /api/discovery/sectors loop touches all 13 sectors
# and the UI + Prometheus exporter poll it continuously — i.e. ~34k
# `SELECT COUNT(*) FROM tglc WHERE sector = ?` per day (mean 86 ms, up to ~1 s
# on sector 11's 13M rows): the #2 statement by call count. work_list_size is
# display-only (the lot selector and gap stats compute their own sizes), and
# the tglc row count only moves during a download burst (hours), so a <= 5 min
# lag on the UI number is the whole cost of a 10x cut.
_WORK_LIST_SIZE_HKEY = "work_list_size:v1"
_WORK_LIST_TTL_S = 300.0


def _canonical_sig(path: str):
    """(st_size, int(st_mtime)) of *path*, or None when absent/unreadable."""
    try:
        st = os.stat(path)
        return (st.st_size, int(st.st_mtime))
    except OSError:
        return None


def _tglc_sector_rowcount(tglc_index: str, sector: Optional[int]) -> Optional[int]:
    """COUNT(*) of TGLC rows for *sector* (or all rows when sector is None).

    Index-backed by ix_tglc_sector → cheap. Returns None on any error so the
    caller falls back to a full recompute rather than trusting a bad signal.

    Routed through MetadataCache.tglc_sector_count (exonit/metadata_backend.py's
    seam, 2026-07-20 TGLC->PG migration F1) instead of a raw sqlite3.connect —
    see docs/superpowers/specs/2026-07-20-tglc-index-to-pg.md item (e).
    """
    try:
        from exonit.metadata_cache import MetadataCache
        mc = MetadataCache(tglc_index)
        try:
            return mc.tglc_sector_count(sector)
        finally:
            mc.close()
    except Exception:
        return None


def _work_list_cache_field(canonical_csv: str, tglc_index: str,
                           sector: Optional[int]) -> str:
    """Stable shared_state hash field for one (canonical, index, sector)."""
    return f"{canonical_csv}|{tglc_index}|{sector}"


def _extended_work_list_len(canonical_csv: str, tglc_index: str,
                            sector: Optional[int] = None) -> int:
    """len(load_extended_canonical(...)) with a TTL + row-count cache.

    See the _WORK_LIST_SIZE_HKEY comment for the invalidation rationale and
    the Q7 cross-process motivation. The public behaviour (returned size) is
    identical to recomputing every call.

    The recompute counts in SQL (count_extended_canonical) instead of
    building the list and taking len(): materialising sector 14's ~8M-element
    work list here allocated GBs per cache miss, in EVERY uvicorn worker
    independently (2026-07-27 — the server processes had ratcheted to ~5 GB
    RSS each with transient +3 GB spikes). The cache in front of it only ever
    hid how expensive the miss was.

    shared_state failures (Redis down, corrupt entry) fail OPEN to a full
    recompute — never raise, never serve a wrong/stale size silently past a
    genuine change.
    """
    from exonit.discovery_state import count_extended_canonical
    from exonit.shared_state import get_shared_state
    import time as _time

    field = _work_list_cache_field(canonical_csv, tglc_index, sector)
    st = get_shared_state()
    now = _time.time()

    cached = None
    try:
        cached = st.hget_json(_WORK_LIST_SIZE_HKEY, field)
    except Exception:
        cached = None

    if cached is not None:
        size = cached.get("size")
        computed_at = cached.get("computed_at", 0.0)
        prev_rowcount = cached.get("rowcount")
        # JSON round-trips a tuple through a list — normalise both sides to
        # lists before comparing (see the equivalent gap-stats int-key note
        # in routes/discovery.py's _get_gap_stats for the same class of bug).
        prev_csv_sig = cached.get("csv_sig")
        # Fast path: within the TTL we trust the cache without any DB/FS check.
        if size is not None and now - computed_at < _WORK_LIST_TTL_S:
            return size
        # TTL expired: cheaply re-validate. If the canonical CSV and this
        # sector's TGLC row count are both unchanged, the size is unchanged —
        # refresh the TTL and skip the ~532k reload.
        csv_sig = _canonical_sig(canonical_csv) if canonical_csv else None
        csv_sig_json = list(csv_sig) if csv_sig is not None else None
        rowcount = _tglc_sector_rowcount(tglc_index, sector)
        if (size is not None and rowcount is not None
                and csv_sig_json == prev_csv_sig and rowcount == prev_rowcount):
            try:
                st.hset_json(_WORK_LIST_SIZE_HKEY, field, {
                    "size": size, "computed_at": now,
                    "rowcount": prev_rowcount, "csv_sig": prev_csv_sig,
                })
            except Exception:
                pass  # TTL refresh is an optimisation, not correctness-critical
            return size

    # Miss or genuine change: full recompute.
    csv_sig = _canonical_sig(canonical_csv) if canonical_csv else None
    csv_sig_json = list(csv_sig) if csv_sig is not None else None
    rowcount = _tglc_sector_rowcount(tglc_index, sector)
    size = count_extended_canonical(
        canonical_csv_path=canonical_csv or "", tglc_index_path=tglc_index,
        sector=sector)
    try:
        st.hset_json(_WORK_LIST_SIZE_HKEY, field, {
            "size": size, "computed_at": now,
            "rowcount": rowcount, "csv_sig": csv_sig_json,
        })
    except Exception:
        pass  # caching is an optimisation — a write failure must not 500
    return size


def _check_pid_running_for_download(pid: int) -> bool:
    """Return True iff *pid* is an alive, non-zombie discovery_download process.

    A zombie process (defunct child not yet reaped by its parent) passes the
    os.kill(pid, 0) test because the PID still exists in the process table —
    but it is actually dead.  Without an explicit zombie check, the status
    endpoint reports "running: True" forever for a zombie, producing the
    "indexing forever" symptom observed in the 2026-06-12 AWS incident.

    Two paths:
    - psutil available (normal case): use STATUS_ZOMBIE for explicit check.
    - psutil unavailable: read /proc/<pid>/stat (Linux) and inspect state char.
      On non-Linux without psutil we conservatively return True (best-effort).

    The "discovery_download" cmdline guard provides best-effort PID-recycling
    protection but is secondary to the zombie check.
    """
    try:
        import psutil as _psutil
        if not _psutil.pid_exists(pid):
            return False
        try:
            p = _psutil.Process(pid)
            if p.status() == _psutil.STATUS_ZOMBIE:
                return False
            # Best-effort PID-recycling check.
            cmdline = " ".join(p.cmdline())
            return "discovery_download" in cmdline
        except (_psutil.NoSuchProcess, _psutil.AccessDenied):
            return False
    except ImportError:
        # psutil not available — read /proc/<pid>/stat directly on Linux.
        # The comm field (process name) is wrapped in parentheses; it can
        # contain spaces/parens itself, so we find the LAST ')' and parse
        # the state char from after it.  State 'Z' = zombie.
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return False
        # Signal 0 succeeded — process exists.  Check zombie status via /proc.
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as _sf:
                _stat_text = _sf.read()
            _state_char = _stat_text[_stat_text.rfind(")") + 2]
            return _state_char != "Z"
        except (OSError, IndexError):
            # /proc not available (non-Linux), process disappeared, or stat
            # content was malformed/truncated (e.g. truncated mid-transition
            # with no state char after ')').  Conservatively report alive since
            # os.kill(0) succeeded.
            return True


# ---------------------------------------------------------------------------
# Advisory-lock helpers for the autofill cross-process guard (Task 10).
# Module-level so they can be patched in tests without affecting the
# broader is_pg_url / connect_operational code paths.
# ---------------------------------------------------------------------------

# Arbitrary int4 namespace for pg_try_advisory_lock(int4, int4) calls.
# Chosen to not clash with any other advisory-lock usage in this app.
_AUTOFILL_PG_LOCK_NS = 4558


def _try_pg_advisory_lock(db_url: str, sector: int):
    """Try to acquire a PG session-level advisory lock keyed by *sector*.

    Returns:
        psycopg connection  — lock acquired; keep it open to hold the lock,
                              close it to release.
        None                — lock already held by another session/process.
        False               — *db_url* is not a PostgreSQL DSN (SQLite path):
                              advisory locks are not applicable; caller should
                              proceed without a cross-process guard.

    Raises on unexpected PG connection or query errors so the caller can
    log and decide whether to proceed without the guard.

    Advisory locks are session-scoped in PostgreSQL: if the holding process
    crashes, the connection is dropped and PG releases all its session locks
    automatically — no stale locks possible.
    """
    from exonit.db_backend import is_pg_url

    if not is_pg_url(db_url):
        # SQLite / non-PG deployment: no advisory lock.
        return False

    import psycopg  # noqa: PLC0415 — deferred; only needed on PG paths

    conn = psycopg.connect(db_url)
    conn.autocommit = True
    try:
        cur = conn.execute(
            "SELECT pg_try_advisory_lock(%s, %s)",
            (_AUTOFILL_PG_LOCK_NS, sector),
        )
        row = cur.fetchone()
        # psycopg.connect without row_factory yields plain tuples.
        acquired = row[0] if row else False
        if acquired:
            return conn
        conn.close()
        return None
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        raise


# --- Fase D route modules (2026-07-02 create_app() split) ------------------
# Imported here (module scope, but AFTER every module-level name defined
# above) rather than at the top of the file: each routes/*.py module does
# `from exonit import server as _server` / `from exonit.server import <name>`
# to reach the helpers above, which makes this a circular import — safe only
# once everything those modules need is already bound in this module's
# namespace. Importing at the very top of this file (before e.g. `logger` or
# `WORKER_OFFLINE_SECONDS` exist) would fail with a partial-initialization
# ImportError.
from exonit.routes import admin as admin_routes
from exonit.routes import aws as aws_routes
from exonit.routes import candidates as candidates_routes
from exonit.routes import discovery as discovery_routes
from exonit.routes import progress as progress_routes
from exonit.routes import proxies as proxies_routes
from exonit.routes import sky as sky_routes
from exonit.routes import workers as workers_routes


_STARTUP_TELEMETRY_PURGE_KEY = "startup:telemetry_purge"
# TTL just needs to outlast the brief window where all N uvicorn workers boot
# nearly simultaneously (A3 multi-process mode). Short enough that a later
# restart (well past 300s) always re-purges; long enough that the 4 processes
# spawned within the same startup burst never all win the NX race.
_STARTUP_TELEMETRY_PURGE_TTL_S = 300.0


def _purge_telemetry_once(db, shared_state, retention_days: int = 5) -> bool:
    """Run ``db.purge_telemetry`` at most once per startup burst, cross-process.

    Uses the shared-state NX primitive (``set_if_absent``) so that with the
    A3 multi-process server (uvicorn --workers N), only the first process to
    reach this line performs the DELETE; the others skip it. Without this
    guard, N processes issue concurrent DELETEs against PostgreSQL and 3 of 4
    typically die with ``psycopg.errors.DeadlockDetected`` at boot (harmless
    but noisy). Works identically for MemoryBackend (single-process A0-A2
    deployments), where the first (only) caller always wins.

    Returns True if this call performed the purge, False if it was skipped.
    """
    won = shared_state.set_if_absent(
        _STARTUP_TELEMETRY_PURGE_KEY, "1", ttl=_STARTUP_TELEMETRY_PURGE_TTL_S
    )
    if won:
        db.purge_telemetry(retention_days)
        logger.info(
            "Startup telemetry purge: executed (retention=%dd)", retention_days
        )
    else:
        logger.debug(
            "Startup telemetry purge skipped — another process holds the "
            "startup lock"
        )
    return won


def create_app(config=None) -> FastAPI:
    """Factory function to create the FastAPI application."""
    db_path = _get_db_path(config)
    reports_dir = _get_reports_dir(config)

    cfg = config if config else ExoNitConfig()

    app = FastAPI(title="ExoNit", version=__version__)
    app.add_middleware(GZipMiddleware, minimum_size=10_000)
    db = ExoNitDB(db_path)
    berger = BergerCatalog(cfg.berger_db_path)

    # Purge old telemetry on startup. Guarded cross-process (see
    # _purge_telemetry_once) so only one of the N uvicorn workers in A3
    # multi-process mode actually runs the DELETE.
    _purge_telemetry_once(db, get_shared_state(), 5)

    # --- Worker state (shared, dict-compatible) ---
    # WorkerRegistry wraps the SharedStateBackend (Redis or in-memory) with the
    # same MutableMapping interface as the plain dict it replaces, so all 35+
    # call-sites across the routes/*.py modules work without modification.  In
    # single-process deployments (A0-A2) the backend is typically MemoryBackend
    # (or Redis when available).  A3 (implemented): with uvicorn --workers N all
    # processes share state via the same Redis hash (MemoryBackend is rejected
    # at startup when EXONIT_SERVER_WORKERS > 1).
    workers_state: WorkerRegistry = WorkerRegistry(get_shared_state())
    # Exposed via app.state so tests (and future introspection endpoints)
    # can inspect/mutate the registry without reaching into a routes closure.
    app.state.workers_state = workers_state

    def _check_auth(request: Request):
        auth = request.headers.get("authorization", "")
        return auth == f"Bearer {cfg.worker_token}"

    # --- Shared-state response cache (stale-while-revalidate, 2026-07-31) ---
    # Measured 2026-07-31 against PG on <host> (data/scorecards run notes,
    # see progress/plan-swr-cache-2026-07-31.md): the dashboard aggregates
    # (COUNT(*)/GROUP BY over candidates=22M rows / tasks=12M rows) cost
    # 9-14s per endpoint. The old cache was expire-then-block: once the Redis
    # key expired, the NEXT request paid the full compute inline — so every
    # arrival at the dashboard after a quiet stretch waited 9-14s for the
    # first numbers, and health_work's 10s TTL was shorter than its own
    # compute time, meaning it almost never actually hit.
    #
    # SWR fixes this by decoupling "is this fresh enough to serve" from "do we
    # need to recompute": a stale entry is still returned immediately, and a
    # background thread refreshes it. These are cumulative/near-cumulative
    # counters polled every 15-30s by the UI — serving one poll cycle's worth
    # of staleness is invisible to a human, and is a strictly better trade
    # than blocking the request on a decade-old COUNT(*).
    #
    # Wire format for the cached value: {"_swr_ts": <epoch float>,
    # "payload": <compute() result>}, stored with a native Redis TTL of
    # _SWR_PERSIST_TTL_S (24h) — that TTL is a backstop cap on how stale a
    # served payload can ever get (an endpoint nobody has polled in a day),
    # NOT the freshness window; *ttl_s* is the freshness window, checked
    # against "_swr_ts" on every read.
    #
    # Single-flight: refresh is guarded by a Redis key
    # (cache:refresh:<key>, NX, TTL 120s) so only one of the 4 uvicorn A3
    # processes actually recomputes per stale hit; the rest just re-serve the
    # stale payload. The lock is deleted in a finally so a failed compute()
    # can be retried by the next stale hit; if the thread dies outright the
    # lock's own 120s TTL self-heals.
    #
    # Legacy compatibility: an entry written by the old code (a bare payload
    # dict, no "_swr_ts" key — its own native TTL already enforced freshness)
    # is served as-is and treated as fresh; it is rewritten in the new wrapped
    # format on the next refresh.
    #
    # Failure-safe: any exception from shared_state (or a missing/corrupt
    # entry) falls through to calling compute() inline, exactly like before
    # SWR existed — caching (and its background thread) never breaks an
    # endpoint. The background thread itself is wrapped in try/except with
    # logger.warning; it must never take the process down.
    _SWR_PERSIST_TTL_S = 24 * 3600.0

    # Hard ceiling on served staleness, as a multiple of the endpoint's own
    # freshness window (2026-08-01). "Serve stale, refresh behind" was written
    # for the case where the NEXT poll is seconds away — but nothing bounded
    # how stale "stale" could get. With _SWR_PERSIST_TTL_S at 24 h, an endpoint
    # nobody has polled for hours (a dashboard that was closed, an exporter
    # that was down) answered the first poll back with a payload from hours
    # ago, marked in no way as old, and only the poll AFTER that got real
    # numbers. On 2026-08-01 that arrived as a ~50-minute-old health/work
    # payload feeding the FPP progress baseline, which is what made the rate
    # gauge read ~7,000 TIC/min (see exonit/progress.py and the _feed_progress
    # note in routes/admin.py).
    #
    # Past this multiple we pay compute() inline rather than answer with
    # something that old. Single-flighted through the same lock the background
    # refresh uses, so a burst of arrivals still costs at most one compute:
    # whoever wins the lock computes, everyone else is served the stale entry
    # they would have got anyway.
    _SWR_MAX_STALE_FACTOR = 10.0
    #: Floor under the computed ceiling, so a short-TTL endpoint (/api/status
    #: at 3 s, /api/stars at 10 s) does not fall back to inline compute after a
    #: mere half-minute of nobody polling. The ceiling exists to stop payloads
    #: that are grossly out of date, not to undo SWR for the quiet-then-busy
    #: pattern it was written for.
    _SWR_MIN_MAX_STALE_S = 120.0

    def _swr_start_refresh(key: str, cache_key: str, lock_key: str,
                            compute: Callable[[], dict]) -> None:
        """Background refresh for a stale SWR entry. Runs in its own daemon
        thread — equivalent to running compute() in FastAPI's threadpool, the
        same place it would run for an inline request today."""

        def _run() -> None:
            try:
                fresh = compute()
                try:
                    get_shared_state().set(
                        cache_key,
                        json.dumps({"_swr_ts": _time.time(), "payload": fresh}),
                        ttl=_SWR_PERSIST_TTL_S,
                    )
                except Exception:
                    logger.warning(
                        "SWR background refresh: failed to write cache for %r",
                        key, exc_info=True,
                    )
            except Exception:
                logger.warning(
                    "SWR background refresh: compute() failed for %r",
                    key, exc_info=True,
                )
            finally:
                try:
                    get_shared_state().delete(lock_key)
                except Exception:
                    pass

        threading.Thread(target=_run, daemon=True, name=f"swr-refresh-{key}").start()

    def _swr_refresh_inline(key: str, cache_key: str, lock_key: str,
                            compute: Callable[[], dict]) -> dict | None:
        """Refresh a past-max_stale entry in the request thread and return it.

        Same work as _swr_start_refresh, minus the thread: the caller wants the
        fresh answer, not the stale one it was about to be handed. Returns None
        if compute() failed, so the caller can fall back to serving stale
        rather than 500 — a cache that cannot refresh must not take the
        endpoint down.
        """
        # The lock is held across the write, not just the compute: releasing it
        # early would let the next arrival read the entry we are about to
        # replace, find it still past the ceiling, and compute all over again.
        try:
            try:
                fresh = compute()
            except Exception:
                logger.warning("SWR inline refresh: compute() failed for %r",
                               key, exc_info=True)
                return None
            try:
                get_shared_state().set(
                    cache_key,
                    json.dumps({"_swr_ts": _time.time(), "payload": fresh}),
                    ttl=_SWR_PERSIST_TTL_S,
                )
            except Exception:
                logger.warning(
                    "SWR inline refresh: failed to write cache for %r",
                    key, exc_info=True)
            return fresh
        finally:
            try:
                get_shared_state().delete(lock_key)
            except Exception:
                pass

    def _cached_json(key: str, ttl_s: float, compute: Callable[[], dict],
                     max_stale_s: float | None = None) -> dict:
        """Return cached JSON response for *key* or call *compute* and cache it.

        Stale-while-revalidate: a fresh entry (age <= ttl_s) is returned
        as-is; a stale entry is ALSO returned as-is, and a background thread
        (single-flighted via a Redis lock) refreshes it for the next caller.
        Only an absent/corrupt entry, or a shared_state failure, pays for
        compute() inline.

        Parameters
        ----------
        key:
            Short endpoint identifier, e.g. ``"stats"``.  The Redis key will be
            ``cache:<key>``.
        ttl_s:
            Freshness window in seconds. Older than this, the entry is stale:
            still served, but triggers a background refresh.
        compute:
            Zero-argument callable that returns the response dict.  Must be
            idempotent and safe to call concurrently (thundering-herd on a
            cache miss, or on a lost single-flight race, is acceptable).
        max_stale_s:
            Hard ceiling on served staleness (2026-08-01). Past it the entry is
            not served at all: the caller pays compute() inline (single-flighted
            — a loser of that race is still served the stale entry). Defaults to
            ``_SWR_MAX_STALE_FACTOR * ttl_s``; pass a number to override, or
            ``math.inf`` for the original unbounded behaviour. Floored at
            ``_SWR_MIN_MAX_STALE_S``.
        """
        if max_stale_s is None:
            max_stale_s = max(_SWR_MAX_STALE_FACTOR * ttl_s,
                              _SWR_MIN_MAX_STALE_S)
        cache_key = f"cache:{key}"
        try:
            raw = get_shared_state().get(cache_key)
        except Exception:
            # shared_state unavailable — fall through to inline compute().
            raw = None

        if raw is not None:
            try:
                entry = json.loads(raw)
            except Exception:
                entry = None

            if isinstance(entry, dict) and "_swr_ts" in entry:
                age = _time.time() - entry["_swr_ts"]
                payload = entry.get("payload")
                if age <= ttl_s:
                    return payload
                # Stale: try to win the single-flight refresh lock; whether we
                # win it or not, serve the stale payload immediately — UNLESS
                # it is past max_stale_s, in which case the winner of that same
                # lock computes inline instead of answering with it. Losers are
                # still served the stale entry: making every arrival in a burst
                # pay a 9-14 s compute would be a worse failure than one more
                # poll cycle of staleness.
                lock_key = f"cache:refresh:{key}"
                try:
                    got_lock = get_shared_state().set_if_absent(
                        lock_key, "1", ttl=120.0)
                except Exception:
                    got_lock = False
                if got_lock and age > max_stale_s:
                    fresh = _swr_refresh_inline(key, cache_key, lock_key,
                                                compute)
                    # None = compute() failed. Serving the stale payload beats
                    # turning a cache problem into a 500 on the dashboard.
                    return payload if fresh is None else fresh
                if got_lock:
                    _swr_start_refresh(key, cache_key, lock_key, compute)
                return payload
            elif entry is not None:
                # Legacy unwrapped entry (pre-SWR code) — its presence already
                # means the old native TTL considered it fresh.
                return entry
            # entry is None (corrupt JSON) — fall through to inline compute().

        # Cache miss (absent key, corrupt entry): pay for it inline, exactly
        # like the pre-SWR behaviour.
        result = compute()

        try:
            get_shared_state().set(
                cache_key,
                json.dumps({"_swr_ts": _time.time(), "payload": result}),
                ttl=_SWR_PERSIST_TTL_S,
            )
        except Exception:
            # Best-effort: if we can't write the cache, still return the result.
            pass

        return result

    # --- Fase D create_app() split (2026-07-02) ----------------------------
    # docs/reviews/2026-07-02-opus48-cleanup-review.md: create_app() used to be
    # ~4,550 lines / 84 endpoints defined inline as closures — the root cause
    # of the server-side logic duplication the review found (the same fix
    # applied to one arm and not its sibling, repeatedly). Endpoint
    # registration now lives in exonit/routes/*.py, one module per domain,
    # each exposing register(app, deps). ServerDeps bundles exactly the
    # objects referenced by more than one domain (db, cfg, berger,
    # workers_state, the auth guard, the response cache) — objects used by
    # only one domain (e.g. the pipeline Popen-handle dict, or a domain's own
    # Redis-key constants) are defined locally inside that domain's
    # register() instead, unchanged from where they lived in create_app()
    # before the split. See progress/impl-createapp-split-2026-07-02.md for
    # the full module map and the multi-process (A3) scoping rationale.
    deps = ServerDeps(
        db=db,
        db_path=db_path,
        cfg=cfg,
        reports_dir=reports_dir,
        berger=berger,
        workers_state=workers_state,
        check_auth=_check_auth,
        cached_json=_cached_json,
    )

    admin_routes.register(app, deps)
    workers_routes.register(app, deps)
    candidates_routes.register(app, deps)
    proxies_routes.register(app, deps)
    progress_routes.register(app, deps)
    discovery_routes.register(app, deps)
    aws_routes.register(app, deps)
    sky_routes.register(app, deps)

    # --- Static files ---

    # Mount reports directory for static access (create if needed)
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/reports",
        StaticFiles(directory=str(reports_path)),
        name="reports",
    )

    # The HTML pages are thin shells whose data is fetched live via /api/*;
    # the markup itself changes on every UI deploy.  Without revalidation the
    # browser's heuristic cache serves a stale shell (e.g. an old sector
    # dropdown), so force a conditional re-fetch on each load.  read_text() is
    # cheap and these pages load infrequently, so the cost is negligible.
    _NO_CACHE = {"Cache-Control": "no-cache, must-revalidate"}

    @app.get("/", response_class=HTMLResponse)
    def serve_dashboard():
        index = _WWW_DIR / "index.html"
        if index.exists():
            return HTMLResponse(content=index.read_text(), headers=_NO_CACHE)
        return HTMLResponse(content="<h1>ExoNit Dashboard</h1><p>www/index.html not found</p>")

    @app.get("/dashboard/workers")
    def dashboard_workers():
        html_path = os.path.join(os.path.dirname(__file__), '..', 'www', 'dashboard-workers.html')
        return FileResponse(html_path, media_type="text/html", headers=_NO_CACHE)

    @app.get("/downloads", response_class=HTMLResponse)
    def serve_downloads():
        page = _WWW_DIR / "downloads.html"
        if page.exists():
            return HTMLResponse(content=page.read_text(), headers=_NO_CACHE)
        return HTMLResponse(content="<h1>Downloads</h1><p>www/downloads.html not found</p>")

    @app.get("/completion", response_class=HTMLResponse)
    def serve_completion():
        page = _WWW_DIR / "completion.html"
        if page.exists():
            return HTMLResponse(content=page.read_text(), headers=_NO_CACHE)
        return HTMLResponse(content="<h1>Completion</h1><p>www/completion.html not found</p>")

    @app.get("/results", response_class=HTMLResponse)
    def serve_results():
        page = _WWW_DIR / "results.html"
        if page.exists():
            return HTMLResponse(content=page.read_text(), headers=_NO_CACHE)
        return HTMLResponse(content="<h1>Results</h1><p>www/results.html not found</p>")

    @app.get("/status", response_class=HTMLResponse)
    def serve_status():
        page = _WWW_DIR / "status.html"
        if page.exists():
            return HTMLResponse(content=page.read_text(), headers=_NO_CACHE)
        return HTMLResponse(content="<h1>Status</h1><p>www/status.html not found</p>")

    @app.get("/sky", response_class=HTMLResponse)
    def serve_sky():
        page = _WWW_DIR / "sky.html"
        if page.exists():
            return HTMLResponse(content=page.read_text(), headers=_NO_CACHE)
        return HTMLResponse(content="<h1>Sky</h1><p>www/sky.html not found</p>")

    # (The /aws page was removed 2026-08-17 with the Deep Field redesign —
    # the cloud fleet is decommissioned; the /api/aws/* endpoints remain.)

    # Static assets for the "Organic" redesign (docs/superpowers/plans/
    # 2026-07-24-ui-redesign-organic.md Task 1): the single stylesheet under
    # www/static/ and the shared ES modules under www/js/. Same _NO_CACHE
    # headers as the HTML shells above (stale-shell gotcha — these files
    # change on every UI deploy, same as the pages that import them).
    # Path-traversal safe: resolve() then relative_to() the asset root,
    # same idea as scripts/ui_preview_server.py's _serve_file().
    def _serve_www_asset(root: Path, rel_path: str):
        root = root.resolve()
        target = (root / rel_path).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return JSONResponse({"error": "not found"}, status_code=404)
        if not target.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        return FileResponse(target, media_type=media_type, headers=_NO_CACHE)

    @app.get("/static/{path:path}")
    def serve_static_asset(path: str):
        return _serve_www_asset(_WWW_DIR / "static", path)

    @app.get("/js/{path:path}")
    def serve_js_asset(path: str):
        return _serve_www_asset(_WWW_DIR / "js", path)

    # --- Quiet-poll access-log filter ---
    # Install the filter directly inside create_app() so it runs in EVERY
    # uvicorn worker process.  With --workers N, uvicorn uses factory=True and
    # calls create_app() in each forked child; any filter attached only in the
    # parent process (e.g. in serve()) would NOT survive the fork.  The
    # uvicorn.access logger is per-process, so attaching here is the right spot.
    #
    # The filter is gated by EXONIT_QUIET_POLL_LOGS (default "1" = active).
    # Set EXONIT_QUIET_POLL_LOGS=0 in the environment to restore full access
    # logs (useful for point-in-time diagnostics without a server restart —
    # although note that changes to env vars require a restart to take effect).
    #
    # Trade-off: polling access logs for the high-frequency worker endpoints are
    # dropped (approx 90 req/s in active phases).  Loss is mitigated by:
    #   • Redis `last_heartbeat` still tracks which workers are alive.
    #   • Errors (4xx/5xx) are NEVER filtered — only 2xx are suppressed.
    #   • EXONIT_QUIET_POLL_LOGS=0 re-enables full logs when needed.
    _install_quiet_poll_filter()

    # --- Post-run worker recycling (A3 multi-process only, no-op otherwise) --
    # See exonit/worker_recycle.py + docs/superpowers/specs/
    # 2026-07-17-worker-recycle-postrun.md. Wires an asyncio background task
    # into this app instance's startup/shutdown that watches
    # recycle:generation and, staggered via recycle:slot, SIGTERMs this
    # process so uvicorn's --workers supervisor respawns it fresh.
    _install_worker_recycle(app, get_shared_state())

    # --- SPOC-executor fast shutdown -------------------------------------
    # Must run at lifespan shutdown (per uvicorn worker process), BEFORE the
    # interpreter's concurrent.futures atexit join — otherwise a recycled
    # worker lingers for hours draining queued MAST checks and its
    # replacement is never spawned. See _shutdown_spoc_executor_fast.
    app.router.add_event_handler("shutdown", _shutdown_spoc_executor_fast)

    return app


# High-frequency worker endpoints whose 2xx access logs are suppressed when
# EXONIT_QUIET_POLL_LOGS=1 (the default).  Only GET/POST status 2xx hits these
# paths that generate volume; errors are always logged.
_QUIET_POLL_PATHS: frozenset[str] = frozenset({
    "/api/worker/task",
    "/api/worker/fpp_task",
    "/api/worker/exominer_task",
    "/api/worker/heartbeat",
    # /api/worker/list and /api/worker/update are lower-frequency (triggered by
    # UI refresh, not tight worker poll loops) so they are kept in the logs.
})


class _QuietPollFilter(logging.Filter):
    """Drop 2xx access-log records for high-frequency polling endpoints.

    uvicorn.access emits LogRecord objects with:
        record.args = (client_addr, method, full_path, http_version, status_code)

    This is the public contract verified by AccessFormatter in uvicorn.logging.
    If the tuple has an unexpected shape we conservatively keep the record
    (return True) so no error ever goes missing due to a parse failure.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            _client_addr, _method, full_path, _http_version, status_code = record.args  # type: ignore[misc]
            # Strip query string so "/api/worker/task?foo=bar" matches.
            path = full_path.split("?", 1)[0]
            status = int(status_code)
            if path in _QUIET_POLL_PATHS and 200 <= status < 300:
                return False  # discard — high-frequency 2xx, no diagnostic value
        except Exception:
            # Unexpected format — never swallow a potentially important record.
            pass
        return True


def _install_quiet_poll_filter() -> None:
    """Attach _QuietPollFilter to the uvicorn.access logger in this process.

    Idempotent: if a filter with the same name is already present (e.g.
    because tests call create_app() multiple times) we skip re-adding it.
    Called from create_app() so it runs once per uvicorn worker process.
    Controlled by EXONIT_QUIET_POLL_LOGS env var (default "1" = active).
    """
    if os.environ.get("EXONIT_QUIET_POLL_LOGS", "1") == "0":
        return  # diagnostic mode — full access logs requested
    access_logger = logging.getLogger("uvicorn.access")
    for f in access_logger.filters:
        if isinstance(f, _QuietPollFilter):
            return  # already installed
    access_logger.addFilter(_QuietPollFilter())


def _check_disk_at_startup() -> None:
    """Print a warning at startup if disk is dangerously full.

    Defended against by `data/exominer_work` symlinked to /mnt/disc2 (see
    project_disc2_temp_dirs memory) but residual logs and DB growth can
    still fill /. A loud warning is cheaper than the database-full crash.
    """
    import shutil
    try:
        du = shutil.disk_usage("/")
    except OSError:
        return
    free_gb = du.free / (1024 ** 3)
    if free_gb < 1:
        logger.critical("only %.1f GB free on / — writes will fail soon", free_gb)
    elif free_gb < 5:
        logger.warning("only %.1f GB free on / — consider cleanup", free_gb)


def _rotate_old_logs(keep_days: int = 7) -> int:
    """Compress log files older than ``keep_days`` so they don't pile up.

    Walks ``logs/`` and ``server.log`` / ``worker/*.log``; gz-compresses
    files older than the threshold and deletes anything older than 30 days.
    Returns count of files acted on. Best-effort, never raises.
    """
    import gzip
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[1]
    cutoff = _time.time() - keep_days * 86400
    hard_cutoff = _time.time() - 30 * 86400
    candidates = list((project_root / "logs").glob("*.log")) + list(
        (project_root / "worker").glob("worker_*.log"))
    n = 0
    for p in candidates:
        try:
            mtime = p.stat().st_mtime
            if mtime < hard_cutoff:
                p.unlink()
                n += 1
                continue
            if mtime < cutoff and not p.name.endswith(".gz"):
                gz = p.with_suffix(p.suffix + ".gz")
                if gz.exists():
                    continue
                with open(p, "rb") as fi, gzip.open(gz, "wb") as fo:
                    fo.writelines(fi)
                p.unlink()
                n += 1
        except OSError:
            continue
    return n


def _uvicorn_max_requests() -> int | None:
    """Parse ``EXONIT_UVICORN_MAX_REQUESTS`` into a ``limit_max_requests`` value.

    Returns the positive int to pass to ``uvicorn.run(limit_max_requests=...)``,
    or ``None`` when recycling is disabled. Unset defaults to 20000. ``"0"``
    is the deliberate, silent opt-out. An invalid value (non-integer or
    negative) falls back to the 20000 default WITH a ``logger.warning`` —
    this is an OOM safeguard, so a config typo must fail SAFE (recycling
    stays on) rather than silently disabling the protection it exists for.
    """
    raw = os.environ.get("EXONIT_UVICORN_MAX_REQUESTS", "20000")
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "EXONIT_UVICORN_MAX_REQUESTS=%r is not an integer — falling "
            "back to the default (20000) uvicorn worker recycling limit",
            raw)
        return 20000
    if value < 0:
        logger.warning(
            "EXONIT_UVICORN_MAX_REQUESTS=%d is negative — falling back to "
            "the default (20000) uvicorn worker recycling limit", value)
        return 20000
    if value == 0:
        return None
    return value


def main(workers: int = 1):
    """Run the server with uvicorn.

    Parameters
    ----------
    workers:
        Number of uvicorn worker processes.  Defaults to 1 (single-process,
        current production mode).  Pass N>1 to enable multi-process serving
        (A3 feature) — requires a reachable Redis instance.

    Notes
    -----
    Startup-only side effects (_check_disk_at_startup, _rotate_old_logs) run
    once in the parent process regardless of *workers*.

    When workers > 1:
    - ``EXONIT_SERVER_WORKERS`` is set in the environment before forking so
      child processes inherit the value and the shared_state hard-assert fires
      if Redis is absent.
    - A preflight Redis ping is done in the parent to fail fast with a clear
      message before N child processes are spawned.
    - uvicorn receives an import-string + factory=True so each forked process
      calls create_app() independently (required by uvicorn multiprocessing).
    - ``limit_max_requests`` recycles each worker process periodically (see
      the comment at the call site below for the "why").

    When workers <= 1:
    - Exact pre-A3 behaviour: app instance passed directly to uvicorn.run.
      No Redis requirement; backward-compatible with all existing deployments.
    """
    import uvicorn

    _check_disk_at_startup()
    rotated = _rotate_old_logs()
    if rotated:
        logger.info("[startup] log rotation: %d file(s) compressed/deleted", rotated)

    config = ExoNitConfig()
    port = config.port

    if workers <= 1:
        # Pre-A3 single-process path — unchanged behaviour.
        # NOT eligible for limit_max_requests: there is no supervisor here
        # (that only exists in uvicorn's multiprocess manager, workers>1
        # below). In this path the process IS the server — hitting the
        # request limit would make it exit 0, and systemd's
        # Restart=on-failure does not restart a clean exit, so the server
        # would just stay down. Single-process is dev-only in practice, but
        # keep this path untouched regardless.
        app = create_app(config)
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        # Multi-process path (A3).
        # 1. Propagate workers count to child processes via environment.
        os.environ["EXONIT_SERVER_WORKERS"] = str(workers)

        # 2. Preflight: verify Redis is reachable before forking N processes.
        #    get_shared_state() will raise RuntimeError with a clear message if
        #    Redis is absent (the hard-assert in shared_state.py triggers because
        #    EXONIT_SERVER_WORKERS is already set above).
        from exonit.shared_state import get_shared_state, reset_shared_state
        get_shared_state()   # raises RuntimeError if Redis unavailable
        reset_shared_state()  # clear singleton; each child will create its own

        # 3. Launch N worker processes via uvicorn's built-in multiprocessing.
        #    factory=True tells uvicorn to call create_app() in each child.
        #    limit_max_requests recycles each worker process periodically —
        #    added 2026-07-31 after the OOM incident where the 4 uvicorn
        #    processes had each grown to ~8 GB (glibc heap high-water that
        #    never shrinks back after a day of serving a run), on top of the
        #    always-on local Qwen model and a 14.5 GB postprocess spike.
        #    EXONIT_UVICORN_MAX_REQUESTS defaults to 20000; at the ~0.5 req/s
        #    a process sees in an active run (fleet heartbeats + dashboard/
        #    exporter polling) that is a recycle roughly every ~11h per
        #    process — the cost is invisible, but the heap high-water no
        #    longer accumulates for days. Set to "0" to disable.
        #    Safe because this is workers>1 ONLY: uvicorn's Multiprocess
        #    supervisor (uvicorn/supervisors/multiprocess.py) polls each
        #    child with is_alive() and respawns it the moment it exits
        #    cleanly at the limit — verified against the installed
        #    uvicorn==0.44.0 before this change (see the implementation
        #    handoff). State survives a recycle: A3 requires all cross-
        #    process state in Redis; the two known in-memory residues
        #    (_discovery_orchestrator["proc"] / _discovery_download["proc"])
        #    are deliberately process-local Popen handles with Redis/DB/pid
        #    as the cross-process source of truth, and the orchestrator
        #    subprocess itself (spawned WITHOUT start_new_session, so same
        #    process group) is not killed by its parent exiting — it is
        #    orphaned and reparented to init/systemd, and the dead-run
        #    watchdog tracks it by pid via /proc, not via the Popen handle.
        #    The one real exposure: sector auto-fill downloads run in a
        #    daemon thread inside the worker process (discovery_autofill),
        #    so a recycle mid-download would kill it the same way a full
        #    server restart already can (known, documented gotcha) — rare
        #    at N=20000 and out of scope to fix here.
        max_requests = _uvicorn_max_requests()
        uvicorn_kwargs = {}
        if max_requests is not None:
            uvicorn_kwargs["limit_max_requests"] = max_requests
        uvicorn.run(
            "exonit.server:create_app",
            factory=True,
            workers=workers,
            host="0.0.0.0",
            port=port,
            **uvicorn_kwargs,
        )


if __name__ == "__main__":
    main()
