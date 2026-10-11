"""Post-run worker recycling: staggered self-restart of A3 multi-process
uvicorn workers, to reclaim per-process memory that peaks and never comes
back down (glibc fragmentation + per-process caches — a worker respawned
mid-day sits at ~0.8 GB RSS vs ~3 GB + ~1 GB swap for one that has run all
day). See docs/superpowers/specs/2026-07-17-worker-recycle-postrun.md.

Mechanism
---------
uvicorn's ``--workers N`` supervisor already respawns any child that dies
(observed in prod: PID 943056 respawned cleanly). A worker that wants to
recycle therefore just sends itself SIGTERM — uvicorn's graceful shutdown
finishes in-flight requests, then the supervisor reforks it fresh. This
module only decides WHEN a given process should do that, staggered so at
most one worker recycles at a time.

Coordination (shared_state — Redis in prod, MemoryBackend in tests; see
exonit/shared_state.py):
  - ``RECYCLE_GENERATION_KEY``: int, the requested "generation" (starts at
    0). Each process reads it once at startup (``birth_generation``); when
    the shared value is later found to be HIGHER, this process is stale and
    should recycle.
  - ``RECYCLE_SLOT_KEY``: a ``SET NX EX=RECYCLE_SLOT_TTL_S`` lock. Only one
    process can hold it at a time; the TTL is what staggers the REST of the
    fleet roughly ``RECYCLE_SLOT_TTL_S`` apart (no explicit release — it just
    expires). ``shared_state.set_if_absent(..., ttl=...)`` already IS this
    exact primitive (used elsewhere for the startup-telemetry-purge lock),
    so no new backend method was needed.

Disarmed entirely in single-process mode (``EXONIT_SERVER_WORKERS`` absent
or <= 1, see ``server_workers_enabled``): there is no supervisor to respawn
a self-killed lone process there, so recycling would be an outage, not a
memory reclaim.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from typing import Callable

from exonit.shared_state import SharedStateBackend

logger = logging.getLogger(__name__)

# --- shared_state keys / tuning (spec: generation + SET NX EX=180, 30s tick) --
RECYCLE_GENERATION_KEY = "recycle:generation"
RECYCLE_SLOT_KEY = "recycle:slot"
RECYCLE_SLOT_TTL_S = 180.0
RECYCLE_CHECK_INTERVAL_S = 30.0

# Duplicated literal from exonit/routes/discovery.py's own
# ``_DOWNLOAD_STATE_KEY`` / ``_DL_FIELD`` (module-local there, not exported
# anywhere in this codebase — tests/test_download_shared_state.py and
# tests/test_indexing_status_fix.py already duplicate the same two literals
# for the same reason). Reusing the EXISTING download-guard state rather
# than inventing a new one, per the spec.
DOWNLOAD_STATE_KEY = "discovery:download:state"
DOWNLOAD_STATE_FIELD = "current"


def server_workers_enabled(env: dict | None = None) -> bool:
    """True only if ``EXONIT_SERVER_WORKERS`` is set and parses to > 1.

    Mirrors ``shared_state._read_server_workers()``'s "absent/parse-error ->
    1" behaviour so a missing or malformed value never arms the self-kill
    loop — a lone single process self-terminating would be an outage.
    """
    env = os.environ if env is None else env
    raw = env.get("EXONIT_SERVER_WORKERS", "1")
    try:
        return int(raw) > 1
    except (ValueError, TypeError):
        return False


def _download_pid_alive(pid: int) -> bool:
    """Zombie-aware pid check, delegated to exonit.server's existing one.

    Local (lazy) import to avoid a circular import: exonit.server imports
    this module (to wire the recycle loop into create_app()); by the time
    THIS function actually runs (well after both modules have finished
    importing, during a live request/background tick), the circularity is
    moot. See ``exonit.server._check_pid_running_for_download`` for the
    zombie-detection rationale — it is the same guard
    ``/api/discovery/download/status`` uses for the download process itself.
    """
    from exonit.server import _check_pid_running_for_download
    return _check_pid_running_for_download(pid)


def download_active(
    shared_state: SharedStateBackend,
    pid_checker: Callable[[int], bool] | None = None,
) -> bool:
    """True if a discovery download is currently running, per shared_state.

    Reads the SAME cross-process record ``/api/discovery/download`` writes
    (``DOWNLOAD_STATE_KEY`` / ``DOWNLOAD_STATE_FIELD``) rather than a new
    key — the download is a Popen child of one of the workers; recycling
    that worker out from under it is not something to risk (spec: "no
    juguem").

    Fails toward True (treat as active / defer) on any error: an uncertain
    read must never let a worker self-terminate mid-download.
    """
    pid_checker = pid_checker or _download_pid_alive
    try:
        meta = shared_state.hget_json(DOWNLOAD_STATE_KEY, DOWNLOAD_STATE_FIELD)
    except Exception:
        logger.warning(
            "worker_recycle: download-state read failed — deferring "
            "recycle out of caution", exc_info=True)
        return True
    if not meta:
        return False
    pid = meta.get("pid")
    if not pid:
        return False
    try:
        return bool(pid_checker(pid))
    except Exception:
        logger.warning(
            "worker_recycle: download pid-liveness check raised — "
            "deferring recycle out of caution", exc_info=True)
        return True


def _rss_mb(pid: int) -> float | None:
    """Best-effort RSS in MB, for the log line only. Never raises."""
    try:
        import psutil
        return psutil.Process(pid).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def maybe_recycle_once(
    shared_state: SharedStateBackend,
    birth_generation: int,
    exit_callback: Callable[[], None],
    *,
    pid_checker: Callable[[int], bool] | None = None,
    slot_ttl_s: float = RECYCLE_SLOT_TTL_S,
    worker_pid: int | None = None,
) -> bool:
    """Single check-and-maybe-recycle tick.

    Returns True iff THIS call won the recycle slot and invoked
    *exit_callback* (real code: ``os.kill(os.getpid(), SIGTERM)``; tests
    inject a spy instead). Never raises — any shared_state failure is
    logged and treated as "not yet, re-check next tick", never as licence
    to kill a healthy worker on an uncertain read.
    """
    try:
        current_generation = shared_state.get_int(RECYCLE_GENERATION_KEY, 0)
    except Exception:
        logger.warning("worker_recycle: generation read failed", exc_info=True)
        return False

    if current_generation <= birth_generation:
        return False  # nothing requested since this process was born

    if download_active(shared_state, pid_checker):
        logger.debug(
            "worker_recycle: generation %d > %d but a download is active "
            "— deferring", current_generation, birth_generation)
        return False

    try:
        won_slot = shared_state.set_if_absent(
            RECYCLE_SLOT_KEY, str(worker_pid or os.getpid()), ttl=slot_ttl_s)
    except Exception:
        logger.warning("worker_recycle: slot claim failed", exc_info=True)
        return False

    if not won_slot:
        return False  # another process already owns the recycle slot

    pid = worker_pid or os.getpid()
    rss_mb = _rss_mb(pid)
    logger.warning(
        "worker_recycle: recycling worker pid=%d gen %d->%d rss=%s MB",
        pid, birth_generation, current_generation,
        f"{rss_mb:.0f}" if rss_mb is not None else "?")
    exit_callback()
    return True


def _default_exit_callback() -> None:
    """SIGTERM to self — uvicorn's graceful shutdown drains in-flight
    requests, then its multi-process supervisor respawns a fresh child."""
    os.kill(os.getpid(), signal.SIGTERM)


async def recycle_loop(
    shared_state: SharedStateBackend,
    birth_generation: int,
    *,
    exit_callback: Callable[[], None] = _default_exit_callback,
    pid_checker: Callable[[int], bool] | None = None,
    check_interval_s: float = RECYCLE_CHECK_INTERVAL_S,
    slot_ttl_s: float = RECYCLE_SLOT_TTL_S,
) -> None:
    """Ticks ``maybe_recycle_once`` every *check_interval_s* until cancelled
    or this process wins the recycle slot (at which point it is about to be
    SIGTERM'd anyway — no point ticking further).

    Meant to run as an ``asyncio.Task`` created at app startup and cancelled
    at shutdown — see ``install()``.
    """
    while True:
        try:
            won = maybe_recycle_once(
                shared_state, birth_generation, exit_callback,
                pid_checker=pid_checker, slot_ttl_s=slot_ttl_s)
            if won:
                return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "worker_recycle: loop tick raised unexpectedly (non-fatal)",
                exc_info=True)
        await asyncio.sleep(check_interval_s)


def install(app, shared_state: SharedStateBackend, *, env: dict | None = None) -> None:
    """Wire the recycle loop into *app*'s startup/shutdown — ONLY when
    ``EXONIT_SERVER_WORKERS`` > 1 (A3 multi-process mode). No-op in
    single-process mode (see ``server_workers_enabled``).

    Uses ``app.router.add_event_handler`` — Starlette's underlying startup/
    shutdown primitive (the same one FastAPI's deprecated ``@app.on_event``
    sugar wraps) — rather than a lifespan context manager: this codebase has
    no existing lifespan precedent (every other create_app()-time background
    loop, e.g. ``routes/discovery.py``'s autofill-resume and dead-run
    watchdog, is a plain daemon ``threading.Thread`` instead), and this is
    the minimal hook that fires exactly once per uvicorn worker process at
    boot/teardown without restructuring how the app object is constructed.
    """
    if not server_workers_enabled(env):
        logger.info(
            "worker_recycle: EXONIT_SERVER_WORKERS<=1 (single-process) — "
            "recycle loop NOT armed (no supervisor to respawn a self-killed "
            "lone process)")
        return

    state: dict = {"task": None}

    async def _on_startup() -> None:
        birth_generation = shared_state.get_int(RECYCLE_GENERATION_KEY, 0)
        logger.info(
            "worker_recycle: armed pid=%d birth_generation=%d "
            "check_interval=%.0fs slot_ttl=%.0fs",
            os.getpid(), birth_generation, RECYCLE_CHECK_INTERVAL_S,
            RECYCLE_SLOT_TTL_S)
        state["task"] = asyncio.create_task(
            recycle_loop(shared_state, birth_generation))

    async def _on_shutdown() -> None:
        task = state.get("task")
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app.router.add_event_handler("startup", _on_startup)
    app.router.add_event_handler("shutdown", _on_shutdown)
