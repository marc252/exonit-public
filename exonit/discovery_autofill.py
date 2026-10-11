"""Durable auto-fill manager for ExoNit discovery campaigns.

Chains gap lots until a sector is closed.  Replaces the external
``ops/night_runner.sh`` with a server-side, resume-safe loop.

Public API
----------
``autofill_tick(sector, *, deps) -> dict``
    Execute one auto-fill iteration (scan → enqueue → launch → wait → persist).
    Returns a status dict; never raises.

``run_autofill(sector, *, deps) -> None``
    Loop ``autofill_tick`` until the sector is complete, an alert fires, or an
    external stop is detected (``status == 'stopped'`` in the DB).

Safety logic (hard-won lessons from the night_runner WAL-lock bug)
------------------------------------------------------------------
- **Empty-lot guard**: if a scan returns ``enqueued==0`` while
  ``gaps_processable>0``, set ``status='alert'`` and stop the loop immediately.
  Never chain empty lots — that is the exact bug that caused the June infinite
  loop.
- **Anomaly cap**: ``MAX_ANOMALIES=2`` consecutive FAILED orchestrator runs
  → ``status='alert'``, stop.
- **Completion**: ``gaps_processable==0`` → ``set_sector_state('complete')``,
  ``status='done'``.
- **Resume-safe**: state is persisted via ``set_autofill`` after every
  significant event.  A crash mid-tick can be resumed by calling
  ``run_autofill`` again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

#: Maximum consecutive FAILED orchestrator runs before alerting and stopping.
MAX_ANOMALIES: int = 2


def terminal_state_is_fresh(
    state_mtime: float, launched_at: Optional[float]
) -> bool:
    """Return ``True`` if a terminal pipeline state is fresh enough to trust.

    The autofill poll loop must ignore a terminal state (``DONE``/``IDLE``/
    ``FAILED``) written *before* the current lot's orchestrator was launched —
    that is the stale state left over from the previous lot, and reading it
    causes a phantom ``DONE`` (premature stop, or a second concurrent
    orchestrator on the next tick).

    Parameters
    ----------
    state_mtime:
        Last-modified time of the pipeline state file (``Path.stat().st_mtime``).
    launched_at:
        Wall-clock recorded right after spawning the orchestrator subprocess
        (``_discovery_orchestrator['started_at']``).  Same host as
        *state_mtime*, so the comparison is clock-skew-free.  ``None`` means no
        launch timestamp has ever been recorded (no successful launch yet); we
        then cannot judge freshness and fall back to accepting the state,
        preserving the pre-guard behaviour.

    Notes
    -----
    A pure function on purpose: the real ``_poll`` closure reads the file and
    sleeps, which is hard to unit-test; this isolates the one piece of logic
    worth testing directly.
    """
    return launched_at is None or state_mtime >= launched_at


def _insert_no_data(db_path: str, sector: int, tic_ids: Iterable) -> None:
    """Idempotently insert TIC IDs into ``discovery_no_data`` with ``reason='no_fits'``.

    Uses ``ON CONFLICT (tic_id, sector) DO NOTHING`` (valid in both SQLite
    3.24+ and PostgreSQL), so repeated calls for the same TIC are safe.
    The backend translates ``?`` placeholders and ``datetime('now')`` for PG.
    """
    from exonit.db_backend import connect_operational

    b = connect_operational(db_path)
    try:
        for tic_id in tic_ids:
            b.execute(
                "INSERT INTO discovery_no_data(tic_id, sector, reason, checked_at) "
                "VALUES (?, ?, 'no_fits', datetime('now')) "
                "ON CONFLICT (tic_id, sector) DO NOTHING",
                (str(tic_id), sector),
            )
    finally:
        b.close()


@dataclass
class AutofillDeps:
    """Dependency container injected into the auto-fill manager.

    All callables are kept out of the module so tests can supply lightweight
    mocks without triggering real scans, DB writes, or process launches.

    Attributes
    ----------
    db_path:
        Operational DB path (SQLite path or ``postgresql://`` DSN).
        Used to persist state via ``get/set_autofill`` and
        ``get/set_sector_state``.
    gap_stats:
        ``(sector: int) -> dict`` — returns the current gap metric.
        Must include at least ``{'gaps_processable': int}``.
    scan:
        ``(sector: int) -> dict`` — enqueue the next lot.  Must return a dict
        with at least ``{'enqueued': int, 'lot_csv': str}``.  Returns
        ``{'enqueued': 0}`` when nothing was enqueued.
    launch:
        ``(lot_csv: str, sector: int) -> None`` — start the pipeline
        orchestrator for *lot_csv*.  May be a no-op when the scan step already
        launched the orchestrator; provided separately so tests can verify it
        was (or was not) called.
    poll:
        ``() -> dict`` — blocking call that returns when the current pipeline
        run reaches a terminal state (``DONE`` or ``FAILED``).  Must include
        a ``'state'`` key, e.g. ``{'state': 'DONE'}`` or
        ``{'state': 'FAILED', 'detail': '...'}``.
    """

    db_path: str
    gap_stats: Callable
    scan: Callable
    launch: Callable
    poll: Callable
    download: Optional[Callable] = None
    """Optional download dep: ``(sector: int, tic_ids: list[int]) -> set[int]``.

    Triggers a FITS download attempt for *tic_ids* in *sector* and returns the
    subset of TIC IDs that *still* have no data after the attempt.  When
    ``None``, the download-retry step is skipped and ``gaps_processable==0``
    goes straight to completion (backward-compatible with Task-8 callers).
    Tests supply a lightweight mock; real callers wire in the
    ``/api/discovery/download`` path.
    """
    on_complete: Optional[Callable] = None
    """Optional completion hook: ``(sector: int) -> None``.

    Called right after ``set_sector_state(sector, 'complete', ...)`` (both
    completion branches below). Q1 (perf review 2026-09-26): a sector
    reaching completion is a lot boundary where the cached gap-stats numbers
    (now up to 1 h stale, EXONIT_GAP_STATS_TTL_S) should refresh immediately
    rather than wait out the TTL. ``None`` (default) skips the hook —
    backward-compatible with existing callers/tests. Exceptions raised here
    must be swallowed by the caller (real wiring logs and never lets a
    refresh failure affect the autofill state machine).
    """


def autofill_tick(sector: int, *, deps: AutofillDeps) -> dict:
    """Execute one auto-fill iteration and persist the new state.

    Flow
    ----
    1. Load current autofill state from DB (tolerates absent row / NULL fields
       on the first tick — Task 7 concern).
    2. If DB ``status == 'stopped'``: return ``{'status': 'stopped'}`` immediately.
    3. Read gap stats via ``deps.gap_stats(sector)``.
    4. ``gaps_processable == 0`` → mark sector ``complete``, set autofill
       ``done``, return ``{'status': 'done'}``.
    5. Call ``deps.scan(sector)`` → ``{enqueued, lot_csv, …}``.
    6. **Empty-lot guard**: ``enqueued == 0`` while ``gaps_processable > 0``
       → set status ``alert``, stop.  Never chain empty lots.
    7. Call ``deps.launch(lot_csv, sector)``.
    8. Persist in-progress checkpoint (resume-safe: crash here = resumable).
    9. Call ``deps.poll()`` → terminal state dict.
    10. ``DONE``: advance ``lots_done``, reset ``anomalies`` to 0.
        non-``DONE``: increment ``anomalies``; ``>= MAX_ANOMALIES`` → alert+stop.

    Returns
    -------
    dict
        Always contains a ``'status'`` key; one of
        ``'running'``, ``'done'``, ``'alert'``, ``'stopped'``.
    """
    from exonit.discovery_state import (
        get_autofill,
        set_autofill,
        set_sector_state,
    )

    db_path = deps.db_path

    def _notify_complete(sec: int) -> None:
        """Best-effort ``deps.on_complete`` call; never raises (Q1)."""
        if deps.on_complete is None:
            return
        try:
            deps.on_complete(sec)
        except Exception:
            logger.warning(
                "[autofill s%d] on_complete hook raised; ignoring", sec,
                exc_info=True)

    # --- 1. Load current state (tolerate NULL / absent row) ---
    af = get_autofill(sector, db_path=db_path)
    if af is None:
        lots_done = 0
        anomalies = 0
        current_status = "running"
    else:
        lots_done = int(af.get("lots_done") or 0)
        anomalies = int(af.get("anomalies") or 0)
        current_status = af.get("status") or "running"

    # --- 2. External stop ---
    if current_status == "stopped":
        logger.info("[autofill s%d] status is 'stopped'; skipping tick", sector)
        return {"status": "stopped"}

    # --- 3. Gap stats ---
    stats = deps.gap_stats(sector)
    gaps = int(stats.get("gaps_processable") or 0)

    # --- 4. Completion (with optional download-retry for dataless TICs) ---
    if gaps == 0:
        # Check whether there are TICs that have no FITS downloaded yet.
        # ``sector_gap_stats`` (the production gap_stats dep) populates
        # ``missing_download_tics``; mocks in tests may return it too.
        missing_dl_tics = list(stats.get("missing_download_tics") or [])

        if missing_dl_tics and deps.download is not None:
            # Attempt a re-download for TICs that have no FITS file.
            logger.info(
                "[autofill s%d] gaps_processable=0 but %d TIC(s) have no FITS; "
                "attempting re-download",
                sector,
                len(missing_dl_tics),
            )
            still_missing = set(deps.download(sector, missing_dl_tics))

            if still_missing:
                # TICs that are still without data after the download retry are
                # permanently dataless.  Insert them into discovery_no_data so
                # they are excluded from subsequent gaps_processable counts.
                logger.info(
                    "[autofill s%d] %d TIC(s) still missing after re-download; "
                    "marking as no_data (reason='no_fits')",
                    sector,
                    len(still_missing),
                )
                _insert_no_data(db_path, sector, still_missing)

            # Re-check gap stats after the download+mark step.
            stats2 = deps.gap_stats(sector)
            gaps2 = int(stats2.get("gaps_processable") or 0)
            missing_dl_tics2 = list(stats2.get("missing_download_tics") or [])

            if gaps2 == 0 and not missing_dl_tics2:
                # Fully drained: no processable gaps and no pending downloads.
                logger.info(
                    "[autofill s%d] fully drained after download+mark; "
                    "marking sector complete",
                    sector,
                )
                set_sector_state(sector, "complete", db_path=db_path)
                _notify_complete(sector)
                set_autofill(
                    sector,
                    db_path=db_path,
                    status="done",
                    lots_done=lots_done,
                    anomalies=0,
                )
                return {"status": "done", "gaps_processable": 0, "lots_done": lots_done}

            # Some newly downloaded TICs now have processable gaps, or there
            # are still missing downloads.  Signal 'running' so the next tick
            # picks up the new gaps or retries.
            logger.info(
                "[autofill s%d] re-check after download: gaps2=%d missing2=%d; "
                "continuing",
                sector,
                gaps2,
                len(missing_dl_tics2),
            )
            set_autofill(sector, db_path=db_path, status="running", lots_done=lots_done)
            return {"status": "running", "lots_done": lots_done}

        # No missing downloads (or no download dep): straight completion.
        logger.info(
            "[autofill s%d] gaps_processable=0; marking sector complete", sector
        )
        set_sector_state(sector, "complete", db_path=db_path)
        _notify_complete(sector)
        set_autofill(
            sector,
            db_path=db_path,
            status="done",
            lots_done=lots_done,
            anomalies=0,
        )
        return {"status": "done", "gaps_processable": 0, "lots_done": lots_done}

    # --- 5. Scan (enqueue next lot) ---
    scan_result = deps.scan(sector)
    enqueued = int(scan_result.get("enqueued") or 0)
    lot_csv = scan_result.get("lot_csv") or ""

    # --- 6. Empty-lot guard ---
    # This is THE fix for the June WAL-lock bug: never start an orchestrator
    # run for an empty lot, which would immediately drain, trigger the "done"
    # heuristic, loop again, and repeat indefinitely while gaps > 0.
    if enqueued == 0:
        alert_msg = (
            f"scan returned enqueued=0 while gaps_processable={gaps} "
            f"(possible WAL-lock contention, selection bug, or download gap)"
        )
        logger.error("[autofill s%d] empty-lot guard triggered: %s", sector, alert_msg)
        # Do NOT increment anomalies here: empty-lot is a different condition
        # from "consecutive FAILED orchestrator run".  The loop stops
        # immediately on empty-lot anyway, so the anomaly counter stays clean.
        set_autofill(
            sector,
            db_path=db_path,
            status="alert",
            last_alert=alert_msg,
            anomalies=anomalies,
            lots_done=lots_done,
        )
        return {"status": "alert", "reason": "empty_lot", "alert": alert_msg}

    # --- 7. Launch orchestrator ---
    deps.launch(lot_csv, sector)

    # --- 8. Persist in-progress checkpoint (resume-safe) ---
    # If the process crashes between here and the poll completing, the next
    # run_autofill call will find status='running' and resume from this lot.
    set_autofill(
        sector,
        db_path=db_path,
        status="running",
        current_lot=lot_csv,
        lots_done=lots_done,
        anomalies=anomalies,
    )

    # --- 9. Poll for terminal state ---
    pipeline = deps.poll()
    final_state = pipeline.get("state", "FAILED")
    logger.info(
        "[autofill s%d] orchestrator finished with state=%r", sector, final_state
    )

    # --- 9b. Honor an external stop that arrived during the (possibly long)
    # poll.  The stop endpoint sets status='stopped' but cannot interrupt the
    # in-flight poll; without this re-check the post-poll handler below would
    # overwrite status back to 'running' and the loop would launch another lot,
    # silently ignoring the user's stop.  _poll may also return a 'STOPPED'
    # sentinel (Change B in server.py) when it observes the stop directly.
    af_after_poll = get_autofill(sector, db_path=db_path)
    stopped_externally = (
        final_state == "STOPPED"
        or (af_after_poll is not None and af_after_poll.get("status") == "stopped")
    )
    if stopped_externally:
        logger.info(
            "[autofill s%d] external stop detected after poll; honoring "
            "(not overwriting status, not relaunching)", sector,
        )
        return {"status": "stopped"}

    # --- 10. Handle result ---
    if final_state == "DONE":
        new_lots_done = lots_done + 1
        set_autofill(
            sector,
            db_path=db_path,
            status="running",
            current_lot=lot_csv,
            lots_done=new_lots_done,
            anomalies=0,  # reset consecutive-anomaly counter on success
        )
        return {"status": "running", "lots_done": new_lots_done}

    # Non-DONE terminal state (FAILED or unexpected)
    new_anomalies = anomalies + 1
    if new_anomalies >= MAX_ANOMALIES:
        alert_msg = (
            f"orchestrator reached {final_state!r}; "
            f"{new_anomalies} consecutive anomaly(s) (cap={MAX_ANOMALIES})"
        )
        logger.error(
            "[autofill s%d] anomaly cap reached: %s", sector, alert_msg
        )
        set_autofill(
            sector,
            db_path=db_path,
            status="alert",
            anomalies=new_anomalies,
            last_alert=alert_msg,
            lots_done=lots_done,
            current_lot=lot_csv,
        )
        return {"status": "alert", "reason": "anomaly_cap", "alert": alert_msg}

    # Below the anomaly cap: log, persist, signal caller to continue
    logger.warning(
        "[autofill s%d] orchestrator %r (anomaly %d/%d); will retry next tick",
        sector, final_state, new_anomalies, MAX_ANOMALIES,
    )
    set_autofill(
        sector,
        db_path=db_path,
        status="running",
        anomalies=new_anomalies,
        lots_done=lots_done,
        current_lot=lot_csv,
    )
    return {"status": "running", "anomalies": new_anomalies}


def run_autofill(sector: int, *, deps: AutofillDeps) -> None:
    """Loop ``autofill_tick`` until the sector closes or an alert fires.

    Initialises or resumes the autofill state:

    - **No row in DB**: create with ``status='running'``, start the loop.
    - **Row with status ``'done'`` or ``'alert'``**: already terminal; log
      and return immediately.  Caller must reset the row (or status) to restart.
    - **Row with status ``'running'``, ``'stopped'``, or NULL**: resume by
      setting status back to ``'running'`` and entering the loop.

    Each ``autofill_tick`` call persists its result, so the loop is
    resume-safe: a crash exits the loop, and the next ``run_autofill`` call
    picks up from the last persisted state.

    The loop exits when ``autofill_tick`` returns a terminal status:
    ``'done'``, ``'alert'``, or ``'stopped'``.
    """
    from exonit.discovery_state import get_autofill, set_autofill

    db_path = deps.db_path

    # --- Initialise / resume ---
    af = get_autofill(sector, db_path=db_path)
    if af is None:
        # First call for this sector
        set_autofill(
            sector,
            db_path=db_path,
            status="running",
            lots_done=0,
            anomalies=0,
        )
        logger.info("[autofill s%d] starting (no prior state)", sector)
    else:
        current = af.get("status") or "running"
        if current in ("done", "alert"):
            logger.info(
                "[autofill s%d] already terminal (status=%r); "
                "reset to restart",
                sector, current,
            )
            return
        # running / stopped / NULL → resume
        set_autofill(sector, db_path=db_path, status="running")
        logger.info("[autofill s%d] resuming (was status=%r)", sector, current)

    # --- Main loop ---
    while True:
        result = autofill_tick(sector, deps=deps)
        status = result.get("status")

        if status in ("done", "alert", "stopped"):
            logger.info(
                "[autofill s%d] loop exiting (terminal status=%r)", sector, status
            )
            break

        # status == "running" → proceed to next lot
        logger.info(
            "[autofill s%d] lot complete, continuing. lots_done=%s",
            sector, result.get("lots_done"),
        )
