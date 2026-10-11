"""Discovery campaign endpoints: sectors/gap-stats/downloads/scan/autofill (Phase D split).

Extracted verbatim from exonit.server.create_app() — see
docs/reviews/2026-07-02-opus48-cleanup-review.md and
progress/impl-createapp-split-2026-07-02.md for the module map. Endpoint
bodies below are byte-identical to the pre-split code (git move-detection
should match them); only this file's import/register() preamble is new.

Covers: /api/discovery/sectors, /api/discovery/scan, /api/discovery/pipeline,
/api/discovery/download(+status), /api/discovery/sector/{n}/autofill,
/api/workers/start (campaign convenience button — targets the fleet listed in
ops/workers.json via _server._campaign_worker_names(), read live so adding a
worker there needs neither a code edit nor a server restart).
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time as _time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from exonit import server as _server
from exonit.db import _to_epoch
from exonit.procutil import reap_in_background as _reap_in_background
from exonit.server import (
    _PROJECT_ROOT,
    _check_pid_running_for_download,
    _count_online_instances,
    _extended_work_list_len,
    logger,
)
from exonit import progress as progress_model
from exonit.routes.deps import ServerDeps
from exonit.shared_state import get_shared_state

# NOTE: _try_pg_advisory_lock / _expected_instances are called as
# `_server.<name>` (not plain-imported) below — tests patch them via
# patch("exonit.server.<name>", ...) after the app has already been built
# (see proxies.py's NOTE for the full rationale).


def _lot_stamp_from_csv(lot_csv: str) -> str:
    """Lot identifier for run_id minting, from the lot CSV basename.

    Phase 3 Task 2 (durable orchestration): MUST match
    scripts/discovery_orchestrator.py's ``_lot_number_from_csv`` EXACTLY —
    naming compatibility with existing scorecards/dossiers is a hard
    requirement. Deliberately duplicated rather than imported: ``scripts/``
    is not a dependency of ``exonit/`` anywhere else in the codebase, and the
    plan itself calls for a copy of the regex, not a new cross-package import.
    See tests/test_discovery_api.py::test_lot_stamp_from_csv_matches_orchestrator_convention
    for the parity guard.
    """
    import re
    name = Path(lot_csv).name
    m = re.search(r"lot_(\d{8}T\d{6})", name)   # auto lots -> timestamp
    if m:
        return m.group(1)
    m = re.search(r"lot(\d+)", name)             # numbered lots -> number
    return m.group(1) if m else "X"


def _lot_prefix_from_csv(lot_csv: str) -> str:
    """``"lot"`` or ``"deep_lot"`` run_id segment prefix, from the lot CSV.

    FIX-1 (final-review, spec §4 deep-pass-multisector design): deep-rescan
    lots must mint a DISTINCTIVE run_id (``discovery_s0014_deep_lot<ts>_
    <date>``) so dossiers/discovery_runs rows are separable from campaign
    lots at a glance. Deep lot CSVs are named ``s<sector>_deep_lot_<ts>.csv``
    (see scripts/discovery_run.py's --force-rescan lot writer); anything
    else is a normal campaign lot.

    Deliberately kept separate from ``_lot_stamp_from_csv``: that helper's
    bare ts/number return value is relied on byte-identically elsewhere
    (tests/test_discovery_api.py::test_lot_stamp_from_csv_matches_orchestrator_convention
    and the run_id-minting site below), so the deep/normal distinction is
    layered on top here instead of changing its contract.

    MUST match scripts/discovery_orchestrator.py's twin helper of the same
    name EXACTLY -- both-sites-must-agree invariant: the server mints the
    run_id ONCE here before spawning the orchestrator, and the
    orchestrator's own CLI-fallback derivation (``_derive_run_id``) must
    never disagree with it for the same lot_csv.
    """
    return "deep_lot" if "_deep_lot_" in Path(lot_csv).name else "lot"


def _pid_is_alive(pid) -> bool:
    """True if a process with this pid appears to be alive.

    Bare existence check only — no identity/cmdline confirmation. Kept for
    call sites where that is genuinely fine (a false "alive" costs a
    delayed action, never a double-spawn). For the orchestrator liveness
    guards, use ``_pid_is_alive_matching`` instead (see its docstring for
    why: PID recycling).

    ``os.kill(pid, 0)`` sends no signal, only checks existence/permission:
      - ProcessLookupError (ESRCH): definitively dead.
      - PermissionError: the pid exists but is owned by another user — we
        can't be sure, and treating it as dead risks a double-spawn racing
        a healthy orchestrator, so it counts as ALIVE (the safe direction).
      - Anything else (bad pid, etc.): also treated as ALIVE — a guard must
        never raise, and a false "alive" only costs a delayed resume/no-op
        watchdog tick, never data loss.
    """
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return True


# cmdline substring identifying an orchestrator process, shared by the
# scan/resume Popen argv (``scripts/discovery_orchestrator.py``, see
# ``_orch_cmd`` below) and ``_pid_is_alive_matching``'s guard checks.
_ORCHESTRATOR_CMDLINE_MARKER = "discovery_orchestrator.py"


def _pid_is_alive_matching(pid, marker: str) -> bool:
    """True if *pid* is alive AND its cmdline looks like our orchestrator.

    PID-recycling hazard (2026-07-10 review F4): now that
    ``_reap_in_background`` promptly wait()s on every orchestrator child,
    a dead orchestrator's pid is returned to the kernel's free pool almost
    immediately instead of lingering as a zombie — so a bare
    ``_pid_is_alive(pid)`` can be fooled by an *unrelated* process the
    kernel later reuses that pid for, making a crashed run look like it
    still has a live orchestrator forever (resume 409s forever, the
    dead-run watchdog never fires). Confirming the cmdline at ``/proc``
    closes that window, the same way ``_check_pid_running_for_download``
    (exonit/server.py) already does for the download process.

    Two-step, both fail toward ALIVE (never toward a false "dead" that
    could double-spawn a second orchestrator against a healthy run):
      1. ``_pid_is_alive(pid)`` — cheap existence gate, same semantics/
         safe-direction as everywhere else in this file.
      2. Only if step 1 says alive: read ``/proc/<pid>/cmdline`` and check
         *marker* is a substring. A read failure here (the process exited
         between steps 1 and 2, or /proc is unavailable — non-Linux; the
         SERVER itself always runs on Linux/<host>, see CLAUDE.md) is
         inconclusive, not a proof of death, so it is treated as ALIVE —
         same fallback direction ``_check_pid_running_for_download`` uses
         for its own /proc read. Only a *successful* read whose content
         does NOT contain *marker* is a definitive "this is a different,
         recycled-pid process" and returns False.
    """
    if not _pid_is_alive(pid):
        return False
    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as _cf:
            _cmdline = _cf.read().replace(b"\x00", b" ").decode(
                "utf-8", errors="replace")
        return marker in _cmdline
    except (OSError, ValueError):
        return True


# Zombie-leak fix (2026-07-07), generalised (2026-07-10 review F3): the
# actual reaping logic now lives in exonit.procutil (shared with the
# aws.py repatriate Popen and the admin.py pipeline-run Popen, which leak
# the exact same way) — see exonit/procutil.py's docstring. Imported at
# module scope above as ``_reap_in_background`` so the 4 existing call
# sites in this file need no changes.


def register(app, deps: ServerDeps) -> None:
    db = deps.db
    db_path = deps.db_path
    cfg = deps.cfg
    workers_state = deps.workers_state
    _cached_json = deps.cached_json

    # Discovery registry path — duplicated as a plain literal (not shared via
    # deps): it is not currently patchable via exonit.server either (it was
    # a create_app()-local before this split; see
    # tests/test_discovery_api.py's own `hasattr(srv, "_DISCOVERY_REGISTRY")`
    # guard, which already anticipates this), so a local literal here is
    # behavior-identical to the pre-split code.
    _DISCOVERY_REGISTRY = str(_PROJECT_ROOT / "data" / "discovery" / "sectors.json")

    # --- Discovery campaign endpoints ---

    # Per-process local store for the Popen handle (NOT serialisable → stays
    # in-process only).  Cross-process state (pid, log_path, started_at) lives
    # in shared_state under _DOWNLOAD_STATE_KEY so all uvicorn workers can see it.
    #
    # "token" (2026-07-10 review F5): a fresh uuid4 minted at every spawn,
    # stored both here (local) and in the shared_state record (cross-process).
    # The status endpoint's local-Popen fast path used to trust
    # ``local_proc.pid == pid`` alone to mean "I own the handle for THIS
    # download" — but pids are reaped promptly now (2026-07-07 fix) and can
    # be recycled by a NEW download spawned on a sibling uvicorn worker,
    # which would make this stale local handle's poll() report the OLD
    # download's exit code as if it were the new one's. Comparing the token
    # too closes that window: it only matches when this process's own local
    # handle really is the one shared_state currently points at.
    _discovery_download: dict = {
        "proc": None,       # Popen handle or None — local to this process only
        "token": None,      # uuid4 hex identifying which spawn "proc" is from
    }
    # Exposed via app.state so tests can reach the in-process dict if needed.
    app.state.discovery_download = _discovery_download

    # Redis / shared-state key for cross-process download metadata.
    _DOWNLOAD_STATE_KEY = "discovery:download:state"
    # Field names stored in the hash.
    _DL_FIELD = "current"

    # Redis / shared-state key for repatriate process metadata (same pattern).
    _REPATRIATE_STATE_KEY = "aws:repatriate:state"
    _REPATRIATE_FIELD = "current"

    # Module-level state for the background orchestrator process.
    _discovery_orchestrator: dict = {
        "proc": None,       # Popen handle or None
        "started_at": None,
    }

    # ── Background-refreshed gap-stats store (perf) ──────────────────────────
    # sector_gap_stats is expensive: it pulls the multi-million-row completed
    # set and intersects it with each sector's work list. Computing it
    # synchronously inside /api/discovery/sectors made the UI hang ~30 s on every
    # cache miss (and it re-ran the full scan once PER sector). Instead we
    # compute it ONCE for all sectors (single completed-set scan) in a background
    # thread, refreshed every _GAP_STATS_TTL s, and the endpoint reads the last
    # result non-blocking — serving null gap fields until the first compute lands
    # (the UI then falls back to examined/total_tics, which is cheap and already
    # total-relative).
    #
    # P4 (docs/reviews/2026-07-02-opus48-cleanup-review.md): this used to be a
    # plain dict local to create_app() — with ``uvicorn --workers 4`` (A3),
    # each process has its OWN copy, so the expensive completed-set scan ran
    # up to 4x independently and different requests could see different
    # "last computed" snapshots depending which process served them. Moved
    # into shared_state (Redis in prod, MemoryBackend in tests) so all 4
    # processes share one cached value and only one of them computes it at a
    # time. The "who computes" mutex is a self-expiring set_if_absent lock
    # (TTL comfortably above how long the scan can take) instead of the old
    # in-process threading.Lock + "computing" bool — same
    # stale-while-revalidate semantics, now correct across processes.
    #
    # Perf review 2026-09-26 (docs/reviews/2026-09-26-enqueue-path-performance-
    # review.md, Q1): the 300 s TTL made `_load_tglc_set`'s
    # `SELECT tic_id FROM tglc WHERE sector=$1` (the single most expensive
    # query in a 7 h pg_stat_statements window, 40 % of exonit_meta) re-run
    # for ALL 15 registered sectors — including tancats / disabled ones with
    # no active auto-fill — every 5 minutes. Raised to 1 h (still
    # env-overridable for anyone who wants the old cadence back) and,
    # separately, disabled/complete sectors are skipped entirely below unless
    # their auto-fill loop is actively running (see the `enabled`/`state`
    # check in `_compute_all_gap_stats`).
    _GAP_STATS_TTL = float(os.environ.get("EXONIT_GAP_STATS_TTL_S", "3600"))
    # S2 (2026-09-26, docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md
    # §5.2): a sector that is merely "enabled" (open, still receiving lots
    # eventually) but not the one an operator is watching right now does not
    # need the same 1 h cadence as the active sector — its gaps barely move
    # between polls. Only the active sector(s) (state == 'active') and the
    # lot-boundary force-refresh get the short TTL; every other
    # enabled/paused sector gets this much longer one.
    _GAP_STATS_IDLE_TTL = float(
        os.environ.get("EXONIT_GAP_STATS_IDLE_TTL_S", "21600"))
    _GAP_STATS_DATA_KEY = "gap_stats:data"
    _GAP_STATS_LOCK_KEY = "gap_stats:computing"
    _GAP_STATS_LOCK_TTL = 120.0  # safety net: auto-releases if the computing
                                 # process dies mid-scan without a chance to
                                 # run its `finally`.

    def _sector_gap_stats_ttl(num: int, s_entry: dict) -> Optional[float]:
        """Per-sector refresh cadence (S2 §5.2). ``None`` means "never
        auto-refresh this sector's gap stats" (closed/complete, no autofill
        running right now) — the exact same skip condition
        ``_compute_all_gap_stats`` used to inline before this refactor (Q1,
        perf review 2026-09-26); a running auto-fill still counts as "moving"
        even for a disabled/complete sector. Errors in the cheap state
        lookups fail OPEN (idle cadence, never "skip forever") so a
        transient PG hiccup here never silently hides a sector's numbers.
        """
        from exonit.discovery_state import get_sector_state, get_autofill
        enabled = bool(s_entry.get("enabled", False))
        try:
            state = get_sector_state(num, db_path=db_path)
        except Exception:
            state = None  # fail open below
        if state == "active":
            return _GAP_STATS_TTL
        if enabled and state != "complete":
            return _GAP_STATS_IDLE_TTL
        try:
            af_row = get_autofill(num, db_path=db_path)
            autofill_running = bool(
                af_row and af_row.get("status") == "running")
        except Exception:
            autofill_running = True  # fail open: don't skip
        return _GAP_STATS_IDLE_TTL if autofill_running else None

    def _compute_all_gap_stats(
        sectors_raw: dict, tglc_index: str, only_sectors: Optional[set] = None,
    ) -> dict:
        """Compute gap stats for every "moving" sector in *sectors_raw*, or
        only the ones in *only_sectors* when given (S2: a per-sector
        force-refresh / stale-TTL pass must never re-scan sectors nobody
        asked for). A sector whose ``_sector_gap_stats_ttl`` is ``None``
        (closed, no running autofill) is skipped even when explicitly listed
        in *only_sectors* — "closed/complete without autofill: never" (spec
        §5.2) is absolute, not just a cadence default.

        Legacy path (``EXONIT_GAP_STATS_V2`` off, or SQLite): pulls the
        multi-million-row completed set ONCE for every sector computed in
        this call (unchanged shape). V2 path (``sector_gap_stats``'s own
        dispatch): no whole-DB set at all — this loop only decides WHICH
        sectors to (re)compute; ``sector_gap_stats`` decides HOW.
        """
        from exonit.discovery_state import load_completed_tic_set, sector_gap_stats
        from exonit import gap_stats_v2 as _gsv2
        # V2 never needs the whole-DB completed set (that is the entire
        # point of S2) — only load it for the legacy path, and only once for
        # every sector this call will actually touch.
        completed = None if _gsv2.is_enabled() else load_completed_tic_set(db_path)
        out: dict = {}
        for s_key, s_entry in sorted(
                sectors_raw.items(),
                key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 10**9):
            try:
                num = int(s_key)
            except (TypeError, ValueError):
                continue
            if only_sectors is not None and num not in only_sectors:
                continue

            if _sector_gap_stats_ttl(num, s_entry) is None:
                continue

            try:
                gs = sector_gap_stats(
                    num, db_path=db_path,
                    canonical_csv=s_entry.get("canonical_list", "") or "",
                    tglc_index=tglc_index, completed=completed)
                out[num] = {
                    "gaps_processable": int(gs["gaps_processable"]),
                    "no_data": int(gs["no_data"]),
                    "completed": int(gs["completed"]),
                    # Same work list `completed` was counted over — the
                    # pct_complete denominator must be at least this big
                    # (see the max(total_tics, …) note in the route).
                    "work_list_size": int(gs.get("work_list_size") or 0),
                    # done-state design (2026-09-26): only present with
                    # EXONIT_DONE_STATE=1 (see sector_gap_stats). None here
                    # keeps the /api/discovery/sectors response shape
                    # unchanged when the flag is off.
                    "failed_terminal": (
                        int(gs["failed_terminal"])
                        if "failed_terminal" in gs else None
                    ),
                }
            except Exception as exc:
                # Stale-while-error (S2 §5.2): this sector simply stays out
                # of `out`. `_write_gap_stats_cache`'s merge (below) then
                # leaves whatever was cached for it untouched — an fdw
                # outage or a transient PG hiccup NEVER drops a sector's
                # last-known-good numbers, and NEVER triggers a fallback to
                # the legacy whole-DB path from here.
                logger.warning("[gap_stats] sector %s failed: %s", s_key, exc)
        return out

    def _write_gap_stats_cache(partial: dict) -> None:
        """Merge *partial* (``{sector:int -> stats dict}``) into the shared
        cache, stamping each updated sector's own ``computed_at`` (S2 §5.2:
        per-sector cadence needs a per-sector timestamp, not one global one
        for the whole payload). Sectors absent from *partial* — outside this
        call's ``only_sectors`` scope, or failed with an exception above —
        keep whatever was already cached for them: a partial refresh (one
        sector) or a partial failure (fdw down for one sector) can never
        blank out every other sector's last-known-good numbers.
        """
        st = get_shared_state()
        raw = None
        try:
            raw = st.get(_GAP_STATS_DATA_KEY)
        except Exception as exc:
            logger.warning(
                "[gap_stats] shared-state read (for merge) failed: %s", exc)
        existing: dict = {}
        if raw:
            try:
                existing = json.loads(raw).get("data") or {}
            except (TypeError, ValueError):
                existing = {}
        now = _time.time()
        for num, stats in partial.items():
            entry = dict(stats)
            entry["computed_at"] = now
            existing[str(num)] = entry
        try:
            get_shared_state().set(
                _GAP_STATS_DATA_KEY, json.dumps({"data": existing}))
        except Exception as exc:
            logger.warning("[gap_stats] shared-state write failed: %s", exc)

    def _release_gap_stats_lock() -> None:
        try:
            get_shared_state().delete(_GAP_STATS_LOCK_KEY)
        except Exception:
            pass

    def _refresh_gap_stats_bg(
        sectors_raw: dict, tglc_index: str, only_sectors: Optional[set] = None,
    ) -> None:
        """Recompute gap stats in a daemon thread (stale-while-revalidate).

        Caller must already hold the ``_GAP_STATS_LOCK_KEY`` mutex (acquired
        via ``set_if_absent`` — the "who computes" claim across all A3
        processes); this releases it when done, success or failure.
        """
        def _bg():
            try:
                d = _compute_all_gap_stats(
                    sectors_raw, tglc_index, only_sectors=only_sectors)
                _write_gap_stats_cache(d)
            except Exception as exc:
                logger.warning("[gap_stats] bg refresh failed: %s", exc)
            finally:
                _release_gap_stats_lock()
        threading.Thread(target=_bg, daemon=True, name="gap-stats").start()

    def _force_gap_stats_refresh_now(
        sectors_raw: dict, tglc_index: str, sector: Optional[int] = None,
    ) -> None:
        """Kick an immediate gap-stats recompute, ignoring the TTL.

        Q1 (perf review 2026-09-26): the TTL was raised 300 s -> 3600 s
        (env-overridable) to stop the whole-DB completed-set + per-sector
        `tglc` scans from re-running every 5 minutes regardless of whether
        anything changed. That trades away freshness *between* lots — so at
        the two moments an operator actually cares about a fresh number (a
        new lot starting, and a sector reaching completion) we force a
        recompute instead of waiting up to an hour for it. Same
        stale-while-revalidate shape as the TTL path: a background thread
        does the work, this call never blocks the caller. A no-op (not an
        error) when a refresh is already in flight — it will land within
        seconds regardless of who asked for it.

        S2 (2026-09-26, §5.2): *sector*, when given, scopes the recompute to
        THAT sector only — a lot starting/completing on sector 6 has no
        reason to re-scan sectors 5/7/11/36/45 too (the old behaviour, when
        this recomputed every "moving" sector on every lot boundary). Omit
        it to keep the previous whole-registry behaviour (still used by a
        few call sites — e.g. a manual "recompute everything" trigger).
        """
        st = get_shared_state()
        try:
            got_lock = st.set_if_absent(
                _GAP_STATS_LOCK_KEY, "1", ttl=_GAP_STATS_LOCK_TTL)
        except Exception:
            got_lock = False
        if got_lock:
            only = {sector} if sector is not None else None
            _refresh_gap_stats_bg(sectors_raw, tglc_index, only_sectors=only)

    # Exposed on app.state (same pattern as discovery_watchdog_check_once /
    # discovery_download above) so Q1 unit tests can exercise the skip-logic
    # and force-refresh hook directly, without going through both layers of
    # HTTP response caching (_cached_json's 30 s SWR + this module's own
    # gap-stats SWR) that the /api/discovery/sectors endpoint wraps around it.
    app.state.gap_stats_compute_all = _compute_all_gap_stats
    app.state.gap_stats_force_refresh_now = _force_gap_stats_refresh_now
    app.state.gap_stats_ttl_s = _GAP_STATS_TTL
    app.state.gap_stats_idle_ttl_s = _GAP_STATS_IDLE_TTL

    def _get_gap_stats(sectors_raw: dict, tglc_index: str) -> dict:
        """Return gap stats without ever hanging on a stale cache.

        S2 (2026-09-26, §5.2): staleness — and therefore "does this sector
        need a background refresh" — is now decided PER SECTOR
        (``_sector_gap_stats_ttl``), each with its own ``computed_at``
        embedded in the cached entry, instead of one TTL/timestamp for the
        whole payload. A sector missing from the cache (never computed yet,
        or dropped by a stale-while-error partial write) is simply absent
        from the returned dict — the caller already falls back to
        examined/total_tics for that case (unchanged contract).

        - Any sector past its own TTL: return the cached value now (or
          nothing, if never cached) and try to become the one process
          (across all 4 A3 workers) that refreshes JUST the stale sectors in
          the background — no request ever waits on the multi-second/minute
          fdw or whole-DB scan.
        - Fully cold cache (nothing cached for ANY sector, e.g. right after a
          server restart): the first process to win the lock computes inline
          ONCE so the very first response carries real numbers, not nulls;
          concurrent callers (any process) during that window get {} and
          fall back to examined/total in the UI — same as before.
        """
        st = get_shared_state()
        raw = None
        try:
            raw = st.get(_GAP_STATS_DATA_KEY)
        except Exception as exc:
            logger.warning("[gap_stats] shared-state read failed: %s", exc)

        existing: dict = {}
        # Back-compat fallback: a payload written before this per-sector
        # refactor (or by anything else that still writes the old shape) has
        # ONE `computed_at` for the whole payload instead of one per sector.
        # Used only when a cached entry has no `computed_at` of its own —
        # without this, every such entry would read as "never computed"
        # (epoch 0) and get force-refreshed on the very next read, which is
        # both wasteful and, for a legacy-shaped payload that is genuinely
        # fresh, wrong.
        legacy_computed_at = None
        if raw:
            try:
                payload = json.loads(raw)
                existing = payload.get("data") or {}
                legacy_computed_at = payload.get("computed_at")
            except (TypeError, ValueError):
                existing = {}

        now = _time.time()
        result: dict = {}
        stale: set = set()
        for s_key, s_entry in sectors_raw.items():
            try:
                num = int(s_key)
            except (TypeError, ValueError):
                continue
            cached = existing.get(str(num))
            ttl = _sector_gap_stats_ttl(num, s_entry)
            if cached:
                # JSON round-trips dict keys through strings and drops
                # nothing else — strip the internal `computed_at` bookkeeping
                # key before handing the entry to callers (unchanged output
                # shape: gaps_processable/no_data/completed/work_list_size/
                # failed_terminal only, exactly like the pre-S2 payload).
                result[num] = {k: v for k, v in cached.items()
                              if k != "computed_at"}
                entry_computed_at = cached.get("computed_at")
                if entry_computed_at is None:
                    entry_computed_at = (
                        legacy_computed_at if legacy_computed_at is not None else 0.0)
                if ttl is not None and (now - entry_computed_at) > ttl:
                    stale.add(num)
            elif ttl is not None:
                stale.add(num)
            # else: never cached AND currently "never auto-refresh" (closed,
            # no autofill) — stays absent from `result`, same as before.

        if not stale:
            return result

        try:
            got_lock = st.set_if_absent(
                _GAP_STATS_LOCK_KEY, "1", ttl=_GAP_STATS_LOCK_TTL)
        except Exception:
            got_lock = False
        if not got_lock:
            # Another process already won the race — don't duplicate the
            # scan; degrade like before until it lands.
            return result

        if result:
            # Warm for at least one sector: never block, background-refresh
            # only the stale ones, return what we have now.
            _refresh_gap_stats_bg(sectors_raw, tglc_index, only_sectors=stale)
            return result

        # Fully cold cache: compute inline once (only the stale/eligible
        # sectors — never a sector this call's own TTL logic would skip).
        try:
            d = _compute_all_gap_stats(sectors_raw, tglc_index, only_sectors=stale)
            _write_gap_stats_cache(d)
            return dict(d)
        except Exception as exc:
            # Completed-set scan / fdw unavailable (e.g. PG down): never
            # 500 — the UI falls back to examined/total. Don't poison the
            # cache, so the next request retries.
            logger.warning("[gap_stats] inline compute failed: %s", exc)
            return {}
        finally:
            _release_gap_stats_lock()

    # ── Background-refreshed CLOSED-sector gap-stats store (2026-10-05) ─────
    # `_sector_gap_stats_ttl` returns None — "never auto-refresh" — for any
    # sector that is closed (disabled, or state=='complete') with no autofill
    # loop actively running right now (Q1, 2026-09-26). That is exactly right
    # for the NORMAL cadence (closed sectors barely move, and a V2 sector_gap_
    # stats call costs ~40 s of PG time each), but it also means such a
    # sector NEVER gets a first number at all: `_get_gap_stats` only adds a
    # sector to `stale` (and therefore only ever computes it) when its own
    # TTL is not None. The 2026-10-05 UI audit found exactly this — s1/2/3/
    # 14/15/17/18/23/26 (all closed) showing null gaps_processable/pct_
    # complete forever, forcing the UI's examined/total_tics fallback (cursor-
    # based, wrong for exactly the reasons documented on api_discovery_
    # sectors below).
    #
    # Fix: a SEPARATE store, SEPARATE lock, and a much longer TTL (24 h,
    # env-overridable) — same stale-while-revalidate shape as `_get_gap_stats`
    # and `_get_downloaded_counts` above (shared_state: Redis in prod,
    # MemoryBackend in tests; a self-expiring `set_if_absent` lock is the
    # cross-process "who computes" claim), but on its own cadence entirely
    # independent of the normal gap-stats TTL/idle-TTL so this NEVER makes a
    # closed sector recompute on the ~1 h / ~6 h cadence that path uses for
    # moving sectors. Computed ONE sector at a time (plain sequential loop,
    # no extra threads per sector) with per-sector failures isolated — a bad
    # sector is logged and simply missing from the result, exactly like
    # `_compute_all_gap_stats`'s own per-sector try/except.
    #
    # Never computed inline on a cold cache (mirrors `_get_downloaded_counts`,
    # not `_get_gap_stats`): there can be close to a dozen closed sectors at
    # once, each costing ~40 s on the V2/PG path, so a cold-cache inline
    # compute here could block a request for several minutes — unacceptable
    # for an endpoint documented as "never hangs". A cold cache always kicks
    # off (or joins) a background refresh and returns {} immediately; the
    # caller merges this dict UNDER the live gap_data dict (closed sectors
    # are never also "moving", so there is no real overlap — gap_data simply
    # wins if one somehow exists).
    _CLOSED_GAP_STATS_TTL = float(
        os.environ.get("EXONIT_CLOSED_GAP_STATS_TTL_S", "86400"))
    _CLOSED_GAP_STATS_DATA_KEY = "closed_gap_stats:data"
    _CLOSED_GAP_STATS_LOCK_KEY = "closed_gap_stats:computing"
    # Headroom for a full sequential pass over every closed sector (~40 s
    # each on the V2/PG path; a dozen closed sectors is still well under an
    # hour) — mirrors the sizing rationale of _GAP_STATS_LOCK_TTL/
    # _DOWNLOADED_COUNTS_LOCK_TTL above: must outlast the real computation so
    # it only ever expires early if the computing process genuinely dies.
    _CLOSED_GAP_STATS_LOCK_TTL = 3600.0

    def _is_closed_sector(num: int, s_entry: dict) -> bool:
        """True for exactly the sectors `_compute_all_gap_stats` skips:
        closed (disabled, or state=='complete') with no autofill actively
        running. Reusing `_sector_gap_stats_ttl` keeps the two stores'
        notion of "closed" identical by construction — there is only one
        place that decides it."""
        return _sector_gap_stats_ttl(num, s_entry) is None

    def _compute_all_closed_gap_stats(
        sectors_raw: dict, tglc_index: str, only_sectors: Optional[set] = None,
    ) -> dict:
        """Compute gap stats for every CLOSED sector in *sectors_raw* (or
        only the ones in *only_sectors*, when given — same per-sector scoping
        contract as `_compute_all_gap_stats`). Mirrors that function's shape
        exactly, restricted to `_is_closed_sector` instead of "has a non-None
        TTL".
        """
        from exonit.discovery_state import load_completed_tic_set, sector_gap_stats
        from exonit import gap_stats_v2 as _gsv2
        completed = None if _gsv2.is_enabled() else load_completed_tic_set(db_path)
        out: dict = {}
        for s_key, s_entry in sorted(
                sectors_raw.items(),
                key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 10**9):
            try:
                num = int(s_key)
            except (TypeError, ValueError):
                continue
            if only_sectors is not None and num not in only_sectors:
                continue
            if not _is_closed_sector(num, s_entry):
                continue
            try:
                gs = sector_gap_stats(
                    num, db_path=db_path,
                    canonical_csv=s_entry.get("canonical_list", "") or "",
                    tglc_index=tglc_index, completed=completed)
                out[num] = {
                    "gaps_processable": int(gs["gaps_processable"]),
                    "no_data": int(gs["no_data"]),
                    "completed": int(gs["completed"]),
                    "work_list_size": int(gs.get("work_list_size") or 0),
                    "failed_terminal": (
                        int(gs["failed_terminal"])
                        if "failed_terminal" in gs else None
                    ),
                }
            except Exception as exc:
                logger.warning(
                    "[closed_gap_stats] sector %s failed: %s", s_key, exc)
        return out

    def _write_closed_gap_stats_cache(partial: dict) -> None:
        """Merge *partial* into the shared closed-gap-stats cache, stamping
        each updated sector's own `computed_at`. Exactly mirrors
        `_write_gap_stats_cache` against the CLOSED store's own key, so a
        partial refresh (one sector) or a partial failure never blanks out
        another closed sector's last-known-good numbers."""
        st = get_shared_state()
        raw = None
        try:
            raw = st.get(_CLOSED_GAP_STATS_DATA_KEY)
        except Exception as exc:
            logger.warning(
                "[closed_gap_stats] shared-state read (for merge) failed: %s", exc)
        existing: dict = {}
        if raw:
            try:
                existing = json.loads(raw).get("data") or {}
            except (TypeError, ValueError):
                existing = {}
        now = _time.time()
        for num, stats in partial.items():
            entry = dict(stats)
            entry["computed_at"] = now
            existing[str(num)] = entry
        try:
            get_shared_state().set(
                _CLOSED_GAP_STATS_DATA_KEY, json.dumps({"data": existing}))
        except Exception as exc:
            logger.warning(
                "[closed_gap_stats] shared-state write failed: %s", exc)

    def _release_closed_gap_stats_lock() -> None:
        try:
            get_shared_state().delete(_CLOSED_GAP_STATS_LOCK_KEY)
        except Exception:
            pass

    def _refresh_closed_gap_stats_bg(
        sectors_raw: dict, tglc_index: str, only_sectors: Optional[set] = None,
    ) -> None:
        """Recompute closed-sector gap stats in a daemon thread. Caller must
        already hold `_CLOSED_GAP_STATS_LOCK_KEY` (set_if_absent); released
        here on both success and failure. Mirrors `_refresh_gap_stats_bg`."""
        def _bg():
            try:
                d = _compute_all_closed_gap_stats(
                    sectors_raw, tglc_index, only_sectors=only_sectors)
                _write_closed_gap_stats_cache(d)
            except Exception as exc:
                logger.warning("[closed_gap_stats] bg refresh failed: %s", exc)
            finally:
                _release_closed_gap_stats_lock()
        threading.Thread(
            target=_bg, daemon=True, name="closed-gap-stats").start()

    def _force_closed_gap_stats_refresh_now(
        sectors_raw: dict, tglc_index: str, sector: Optional[int] = None,
    ) -> None:
        """Kick an immediate closed-gap-stats recompute, ignoring the 24 h
        TTL — used when a mop-up lot is launched against a closed sector
        (POST /api/discovery/scan with allow_disabled=True) so the dashboard
        reflects it without waiting out the full day. A no-op when a refresh
        is already in flight, same semantics as `_force_gap_stats_refresh_now`.
        """
        st = get_shared_state()
        try:
            got_lock = st.set_if_absent(
                _CLOSED_GAP_STATS_LOCK_KEY, "1", ttl=_CLOSED_GAP_STATS_LOCK_TTL)
        except Exception:
            got_lock = False
        if got_lock:
            only = {sector} if sector is not None else None
            _refresh_closed_gap_stats_bg(sectors_raw, tglc_index, only_sectors=only)

    # Exposed on app.state for direct unit testing, mirroring the
    # gap_stats_compute_all/gap_stats_force_refresh_now exposure above.
    app.state.closed_gap_stats_compute_all = _compute_all_closed_gap_stats
    app.state.closed_gap_stats_force_refresh_now = _force_closed_gap_stats_refresh_now
    app.state.closed_gap_stats_ttl_s = _CLOSED_GAP_STATS_TTL

    def _pipeline_lot_end_info() -> Optional[tuple]:
        """``(sector, finish_ts)`` when ``pipeline_state.json`` shows a
        FINISHED (DONE/FAILED) run for a specific sector, else ``None``.

        Review fix (2026-10-05, item 1 of
        progress/review-discovery-closed-sector-gap-stats-2026-10-05.md):
        the scan-START trigger (``_force_closed_gap_stats_refresh_now`` call
        inside ``api_discovery_scan``) only refreshes the closed store at the
        BEGINNING of a mop-up lot, before a single TIC has been processed —
        it does nothing for the lot's RESULT. A closed sector has no idle-TTL
        self-healing cadence the way an open/active sector does (that is the
        whole point of the closed store), so without an end-of-lot trigger
        too, the dashboard would show the pre-mop-up snapshot for up to the
        full 24 h TTL after the lot actually finishes.

        Reuses the EXACT same file/shape ``api_discovery_pipeline`` already
        reads (no new infrastructure, no orchestrator changes): the
        orchestrator's ``_set_state`` merges (not replaces) the state file on
        every transition, so the ``sector`` written by this endpoint's own
        ENQUEUEING write (see ``_discovery_set_pipeline_state`` above)
        persists through to the final DONE/FAILED write for the same run.
        ``finished_at`` is only set by SOME terminal-state call sites in
        ``discovery_orchestrator.py``; ``updated_at`` is set unconditionally
        on every single state transition (including DONE/FAILED), so it is
        used as the fallback finish timestamp when ``finished_at`` is absent.

        Never raises — any missing file, malformed JSON, missing field, or
        unparseable timestamp is treated as "nothing to do", exactly like
        every other best-effort read in this module.
        """
        try:
            state_path = Path(_PIPELINE_STATE_FILE)
            if not state_path.exists():
                return None
            body = json.loads(state_path.read_text())
        except Exception:
            return None
        if body.get("state") not in ("DONE", "FAILED"):
            return None
        sector = body.get("sector")
        if sector is None:
            return None
        try:
            sector = int(sector)
        except (TypeError, ValueError):
            return None
        finish_ts_str = body.get("finished_at") or body.get("updated_at")
        if not finish_ts_str:
            return None
        try:
            finish_ts = datetime.fromisoformat(finish_ts_str).timestamp()
        except (TypeError, ValueError):
            return None
        return sector, finish_ts

    def _get_closed_gap_stats(sectors_raw: dict, tglc_index: str) -> dict:
        """Return gap stats for closed sectors, never blocking.

        Only ever returns/refreshes sectors `_is_closed_sector` currently
        agrees are closed — a sector that re-opens (re-enabled, or an
        autofill loop starts on it) simply stops being considered here on
        the very next call; the live `_get_gap_stats` store takes over for
        it (and wins the merge in the caller) without this store fighting it
        for the same sector.

        Warm (within 24 h): return it immediately. Past the TTL: try to
        become the one process that refreshes it in the background and still
        return the (possibly stale) cached value now. Cold (never cached):
        never compute inline (see the module comment above) — kick off a
        background compute if one is not already running and return {} so
        the caller falls back exactly like it does today, until the first
        background compute lands.

        Review fix (2026-10-05, item 1): ALSO treated as stale, regardless of
        the 24 h TTL, when `_pipeline_lot_end_info()` says a run against that
        exact sector finished (DONE/FAILED) more recently than this sector's
        own cached `computed_at` — the lot-END counterpart to the scan-START
        force-refresh in `api_discovery_scan`. Scoped to that one sector only
        (added to the same `stale` set the TTL path already scopes its
        refresh to); every other closed sector is untouched.
        """
        st = get_shared_state()
        raw = None
        try:
            raw = st.get(_CLOSED_GAP_STATS_DATA_KEY)
        except Exception as exc:
            logger.warning(
                "[closed_gap_stats] shared-state read failed: %s", exc)

        existing: dict = {}
        if raw:
            try:
                existing = json.loads(raw).get("data") or {}
            except (TypeError, ValueError):
                existing = {}

        lot_end = _pipeline_lot_end_info()

        now = _time.time()
        result: dict = {}
        stale: set = set()
        for s_key, s_entry in sectors_raw.items():
            try:
                num = int(s_key)
            except (TypeError, ValueError):
                continue
            if not _is_closed_sector(num, s_entry):
                continue
            cached = existing.get(str(num))
            if cached:
                result[num] = {k: v for k, v in cached.items()
                              if k != "computed_at"}
                entry_computed_at = cached.get("computed_at") or 0.0
                if (now - entry_computed_at) > _CLOSED_GAP_STATS_TTL:
                    stale.add(num)
                elif (lot_end is not None and lot_end[0] == num
                      and entry_computed_at < lot_end[1]):
                    stale.add(num)
            else:
                stale.add(num)
                # A closed sector with NO cached entry at all that also just
                # finished a lot has no `entry_computed_at` to compare
                # against lot_end — it is already in `stale` via the
                # never-cached branch above, so no extra check is needed.

        if not stale:
            return result

        try:
            got_lock = st.set_if_absent(
                _CLOSED_GAP_STATS_LOCK_KEY, "1", ttl=_CLOSED_GAP_STATS_LOCK_TTL)
        except Exception:
            got_lock = False
        if got_lock:
            _refresh_closed_gap_stats_bg(sectors_raw, tglc_index, only_sectors=stale)
        # Never compute inline, warm or cold (see module comment) — always
        # return whatever is already cached (possibly {}) right away.
        return result

    # ── Background-refreshed downloaded-counts store (2026-07-25 fix) ───────
    # The registry's per-sector 'downloaded' field (data/discovery/sectors.json)
    # is written ONCE at the very end of a whole discovery_download.py run
    # (see that script's comment at the _update_registry call site) — so it is
    # stale for the entire duration of a multi-day download. Worse, the
    # download process runs on whichever worker host is doing the fetching
    # (e.g. <host>), each with its OWN local sectors.json, while this
    # server reads only ITS OWN copy — a registry write from a different host
    # never reaches the file this endpoint serves (2026-07-25 sector 18
    # incident: registry stuck reporting 2,509,935 while the real, shared,
    # host-independent PG/SQLite metadata index — MetadataCache.
    # indexed_basenames — already held 5,249,999).
    #
    # Fix: derive 'downloaded' from the index (shared across every host, so
    # it is correct regardless of which worker performed the download)
    # instead of the registry file. Same stale-while-revalidate shape as
    # _get_gap_stats above (one shared-state store, refreshed in a background
    # daemon thread, guarded by a self-expiring cross-process lock) — but with
    # ONE deliberate difference from that pattern: this NEVER computes inline,
    # not even on a cold cache. sector_gap_stats' completed-set scan is
    # bounded to roughly 30 s; MetadataCache.indexed_basenames' fits_blobs
    # lookup (an un-indexable leading-wildcard LIKE — see its docstring and
    # scripts/migration/schema_pg_meta.sql's note on why partitioning does
    # not help it) measured ~1 minute for a SINGLE large sector, and this
    # store computes it for EVERY registered sector. An inline cold-cache
    # compute here could block a request for several minutes, which would
    # break this endpoint's documented "never raises 500 / never hangs"
    # contract. A cold cache therefore always kicks off (or joins) a
    # background refresh and returns {} immediately; the caller (below, in
    # _compute()) falls back to the registry's stored 'downloaded' value per
    # sector whenever a number is missing here — a stale number, never a
    # false zero.
    #
    # TTL: 30 min. This number moves slowly (downloads run over multiple
    # days), so 30-minute staleness is imperceptible to the dashboard, and it
    # matches the existing ops cron that already treats 30 min as an
    # acceptable refresh cadence for this exact field
    # (scripts/ops/recompute_download_registry.sh). It also bounds how often
    # the expensive multi-sector fits_blobs scan can run to twice an hour.
    #
    # Change-detection gate (2026-09-25, fix C): "twice an hour" was still
    # twice an hour of a 76.5M-row seq scan even during the long stretches
    # (multi-day between download bursts) where NOTHING changed in tglc or
    # fits_blobs — the same waste scripts/ops/recompute_download_registry.sh
    # has independently (before==after, every 30 min, round the clock; see
    # that script's own gate). MetadataCache.download_registry_fingerprint()
    # is a pg_stat_user_tables lookup (no data scan) for both tables; when it
    # matches what was stored at the last REAL recompute, the background
    # refresh below just re-stamps the cache entry's freshness instead of
    # repeating the scan. _DOWNLOAD_REGISTRY_MAX_STALE_HOURS is the safety
    # net: even with a matching fingerprint, force a real recompute past this
    # age (catches a fingerprint that is technically unchanged but wrong for
    # some reason we didn't anticipate). None (SQLite/test backend, or PG
    # unreachable) means "no signal" -> always recompute, unchanged from
    # before this fix.
    _DOWNLOAD_REGISTRY_MAX_STALE_HOURS = float(os.environ.get(
        "EXONIT_DOWNLOAD_REGISTRY_MAX_STALE_HOURS", "6"))
    _DOWNLOADED_COUNTS_TTL = 1800.0
    _DOWNLOADED_COUNTS_DATA_KEY = "downloaded_counts:data"
    _DOWNLOADED_COUNTS_LOCK_KEY = "downloaded_counts:computing"
    _DOWNLOADED_COUNTS_LOCK_TTL = 2400.0  # safety net: headroom for a
                                          # multi-sector scan (heavier than
                                          # gap_stats' single completed-set
                                          # scan) to finish before the lock
                                          # self-expires if the computing
                                          # process dies mid-scan.
                                          #
                                          # Raised 600 -> 2400 on 2026-08-02,
                                          # after the switch to the SQL COUNT
                                          # (indexed_basename_count): the full
                                          # 15-sector refresh went from under
                                          # 600 s to ~13 min measured on
                                          # <host>, so the lock was expiring
                                          # MID-scan and the next dashboard
                                          # poll (every 5 s) could claim it and
                                          # start a second concurrent refresh —
                                          # observed as two uvicorn workers at
                                          # ~0.5 GB at once, doubling the PG
                                          # load. Must stay comfortably above
                                          # the real refresh duration; it only
                                          # ever costs a delayed retry when the
                                          # computing process genuinely dies.

    def _compute_all_downloaded_counts(sectors_raw: dict, tglc_index: str) -> dict:
        """{sector_number: indexed-basename-count} for every sector in the
        registry, via ONE MetadataCache connection (mirrors
        _compute_all_gap_stats' single-scan-then-iterate shape, but each
        sector here is its own query — indexed_basename_count has no
        cross-sector aggregate to share).

        Counting happens in SQL (MetadataCache.indexed_basename_count), NOT
        by taking len() of the materialised basename set: this loop runs in
        a background thread of a live uvicorn worker, and the set version
        pulled millions of dict rows per sector into that process — measured
        2026-08-02 on <host> as a sawtooth to ~8.4 GB RSS and back, once per
        sector, at a steady ~24% of a core, for one integer per sector.

        A per-sector failure (e.g. a transient PG hiccup) is logged and that
        sector is simply omitted from the result — never raised past this
        function — so one bad sector cannot poison every other sector's
        count, and the caller's per-sector registry fallback covers the gap.
        """
        from exonit.metadata_cache import MetadataCache
        mc = MetadataCache(tglc_index)
        out: dict = {}
        try:
            for s_key in sectors_raw:
                try:
                    num = int(s_key)
                except (TypeError, ValueError):
                    continue
                try:
                    out[num] = mc.indexed_basename_count(num)
                except Exception as exc:
                    logger.warning(
                        "[downloaded_counts] sector %s failed: %s", s_key, exc)
        finally:
            mc.close()
        return out

    def _write_downloaded_counts_cache(data: dict, fingerprint=None,
                                       last_full_recompute=None) -> None:
        """*fingerprint*/*last_full_recompute* are the change-detection gate's
        bookkeeping (see the constant block above) — ``None`` for a backend
        that can't fingerprint (SQLite/tests), in which case the gate always
        reads "changed" on the next cycle, i.e. always recomputes."""
        try:
            get_shared_state().set(
                _DOWNLOADED_COUNTS_DATA_KEY,
                json.dumps({
                    "data": data,
                    "computed_at": _time.time(),
                    "fingerprint": fingerprint,
                    "last_full_recompute": last_full_recompute,
                }))
        except Exception as exc:
            logger.warning(
                "[downloaded_counts] shared-state write failed: %s", exc)

    def _release_downloaded_counts_lock() -> None:
        try:
            get_shared_state().delete(_DOWNLOADED_COUNTS_LOCK_KEY)
        except Exception:
            pass

    def _get_download_registry_fingerprint(tglc_index: str):
        """See MetadataCache.download_registry_fingerprint's docstring.
        Opens its own short-lived MetadataCache — this runs at most once per
        background refresh cycle, not per sector, so a dedicated connection
        here is negligible next to the scan it may let us skip."""
        from exonit.metadata_cache import MetadataCache
        mc = MetadataCache(tglc_index)
        try:
            return mc.download_registry_fingerprint()
        except Exception as exc:
            logger.warning("[downloaded_counts] fingerprint read failed: %s", exc)
            return None
        finally:
            mc.close()

    def _refresh_downloaded_counts_bg(sectors_raw: dict, tglc_index: str,
                                      prev_payload: dict | None = None) -> None:
        """Recompute downloaded counts in a daemon thread.

        Caller must already hold the _DOWNLOADED_COUNTS_LOCK_KEY mutex
        (acquired via set_if_absent — the "who computes" claim across all A3
        processes); this releases it when done, success or failure. Mirrors
        _refresh_gap_stats_bg exactly.

        Change-detection gate: before paying for the per-sector scan, check
        whether tglc/fits_blobs have actually changed since *prev_payload*'s
        last REAL recompute (fingerprint match + within the safety-net
        window). If so, just re-stamp the existing data's freshness — same
        numbers, no scan. See the constant block near _DOWNLOADED_COUNTS_TTL.
        """
        def _bg():
            try:
                now = _time.time()
                fingerprint = _get_download_registry_fingerprint(tglc_index)
                prev = prev_payload or {}
                prev_data = prev.get("data")
                last_full = prev.get("last_full_recompute") or 0.0
                unchanged = (
                    fingerprint is not None
                    and prev_data
                    and fingerprint == prev.get("fingerprint")
                    and (now - last_full) <
                        _DOWNLOAD_REGISTRY_MAX_STALE_HOURS * 3600.0
                )
                if unchanged:
                    logger.info(
                        "[downloaded_counts] tglc/fits_blobs fingerprint "
                        "unchanged since last recompute (%.0fs ago) — "
                        "skipping the scan, renewing cache freshness only",
                        now - last_full)
                    _write_downloaded_counts_cache(
                        prev_data, fingerprint=fingerprint,
                        last_full_recompute=last_full)
                    return
                d = _compute_all_downloaded_counts(sectors_raw, tglc_index)
                _write_downloaded_counts_cache(
                    d, fingerprint=fingerprint, last_full_recompute=now)
            except Exception as exc:
                logger.warning("[downloaded_counts] bg refresh failed: %s", exc)
            finally:
                _release_downloaded_counts_lock()
        threading.Thread(
            target=_bg, daemon=True, name="downloaded-counts").start()

    def _get_downloaded_counts(sectors_raw: dict, tglc_index: str) -> dict:
        """Return {sector_number: indexed-basename-count}, never blocking.

        Warm cache (within TTL): return it immediately. Past the TTL: try to
        become the one process that refreshes it in the background and still
        return the (possibly stale) cached value now. Cold cache (nothing
        cached yet, e.g. right after a server restart): still never compute
        inline (see the module comment above for why this deviates from
        _get_gap_stats here) — kick off the background compute if not
        already running, and return {} so every sector falls back to its
        registry value until the first background compute lands.
        """
        st = get_shared_state()
        raw = None
        try:
            raw = st.get(_DOWNLOADED_COUNTS_DATA_KEY)
        except Exception as exc:
            logger.warning(
                "[downloaded_counts] shared-state read failed: %s", exc)

        payload = None
        if raw:
            try:
                payload = json.loads(raw)
            except (TypeError, ValueError):
                payload = None

        now = _time.time()
        data: dict = {}
        stale = True
        if payload and payload.get("data"):
            # JSON round-trips dict keys through strings — restore int keys
            # so callers can look up by sector number (an int, from
            # int(s_key) in the per-sector loop below) without a silent
            # always-miss.
            data = {int(k): v for k, v in payload["data"].items()}
            stale = (now - payload.get("computed_at", 0.0)) > _DOWNLOADED_COUNTS_TTL

        if stale:
            try:
                got_lock = st.set_if_absent(
                    _DOWNLOADED_COUNTS_LOCK_KEY, "1",
                    ttl=_DOWNLOADED_COUNTS_LOCK_TTL)
            except Exception:
                got_lock = False
            if got_lock:
                _refresh_downloaded_counts_bg(sectors_raw, tglc_index, payload)
            # else: another process already owns the refresh in flight —
            # don't duplicate the expensive multi-sector scan.

        return data

    # Per-process thread tracker for autofill loops (Task 10).
    # Key = sector int; value = threading.Thread.
    # Within-process guard: this dict + _autofill_threads_lock prevent two
    # threads in the SAME process from running the same sector.
    # Cross-process guard: _try_pg_advisory_lock() (module-level) acquires
    # pg_try_advisory_lock(_AUTOFILL_PG_LOCK_NS, sector) on a dedicated
    # connection — only one uvicorn worker across all 4 A3 processes runs
    # the loop at a time.
    _autofill_threads: dict = {}
    _autofill_threads_lock = threading.Lock()

    # Campaign-level constants
    _DISCOVERY_REGISTRY = str(_PROJECT_ROOT / "data" / "discovery" / "sectors.json")
    # Orchestrator state file (must match DEFAULT_STATE_FILE in discovery_orchestrator.py).
    # Overridable via env var so tests can isolate it from the live state file.
    _PIPELINE_STATE_FILE = os.environ.get(
        "EXONIT_PIPELINE_STATE_FILE",
        str(_PROJECT_ROOT / "data" / "discovery" / "pipeline_state.json"),
    )
    # The "Start workers" button targets the fleet in ops/workers.json, read
    # live via _server._campaign_worker_names() — no hardcoded duplicate, so
    # adding a worker there needs neither a code edit nor a server restart.
    _FARMCTL_BIN = str(_PROJECT_ROOT / "bin" / "farmctl")

    def _farmctl_start(worker_name: str) -> tuple:
        """Invoke 'bin/farmctl start <name>' and return (returncode, stdout, stderr).

        Factored out so tests can monkeypatch server._farmctl_start without
        spawning real SSH sessions.
        """
        # The UI's "Start workers" button is a legitimate fleet starter, so it
        # opts in to the farmctl start guard (bin/farmctl START_GUARD_ENV)
        # explicitly. The guard exists because automated agents started the
        # real fleet three times on 2026-07-13/14 despite written instructions
        # not to; a caller that cannot name itself here cannot start workers.
        env = dict(os.environ, EXONIT_FARMCTL_ALLOW_START="1")
        result = subprocess.run(
            # --no-prune: this button means "bring this worker UP", never "size
            # the fleet down to the scan phase" (2026-07-14). Without it, the
            # implicit --phase scan would make farmctl STOP any instance running
            # under the FPP sizing — so pressing "Start workers" while an FPP
            # backfill is in flight would tear down the very instances draining
            # it (<host> runs 10 in scan, 19 in fpp). The orchestrator's
            # _scale_fleet is the ONLY caller that legitimately wants a phase
            # resize, and it passes --phase explicitly.
            [_FARMCTL_BIN, "start", worker_name, "--no-prune"],
            # 120s (was 45s): worker-<host> launches 6 native-Windows instances
            # over SSH, which exceeds 45s and made the UI "Start workers" button
            # time out mid-launch and report failure.
            capture_output=True, text=True, timeout=120, env=env,
        )
        return result.returncode, result.stdout, result.stderr

    @app.get("/api/discovery/sectors")
    def api_discovery_sectors():
        """Return sector registry with persistent sector-progress fields.

        Robust: returns sane zeros if registry or canonical list is missing;
        never raises 500.

        Each sector entry includes:
        - ``total_tics``       — sector-wide total per CAMPAIGN.md (from MAST).
                                 2026-10-05: has drifted into an unreliable
                                 TIC count for CLOSED sectors (nothing writes
                                 it any more; it tracks ``downloaded``, a
                                 FITS-file-basename count) — but it is still
                                 the only cheap signal that a sector's
                                 download may be INCOMPLETE while the sector
                                 is open/active (``discovery_run.py``'s
                                 ``need_download`` gate only requires the
                                 current lot's TICs, not the whole sector).
                                 Kept for back-compat, the ``pending``/UI
                                 fallback denominator before gap stats land,
                                 AND as part of ``pct_complete``'s
                                 denominator while a sector's download is not
                                 yet verifiably complete (see below).
        - ``work_list_size``   — number of TICs in the extended work list
                                 (canonical CSV + TGLC-indexed extras).  This
                                 is what the scan engine actually operates on.
        - ``downloaded``       — count of FITS indexed for this sector, from
                                 the shared PG/SQLite metadata index (host-
                                 independent — correct regardless of which
                                 worker host performed the download), served
                                 from a background-refreshed store (see the
                                 ``_DOWNLOADED_COUNTS_*`` block below).  Falls
                                 back to the registry's own stored value
                                 (``sectors.json``, host-local, written once
                                 at the end of a ``discovery_download.py``
                                 run) only while the index hasn't produced a
                                 number for this sector yet, or a per-sector
                                 index lookup failed — 2026-07-25 fix: the
                                 registry alone used to freeze mid-download
                                 and, worse, could point at the wrong host
                                 entirely (see ``docs/STATUS.md``).
        - ``cursor``           — persistent cursor from the registry (sectors.json).
        - ``examined``         — same as cursor: persistent count of TICs examined
                                 across all runs.  Does NOT reset when the DB is
                                 cleared (key property for the AWS use-case where
                                 the DB is wiped between runs).
        - ``pending``          — real remaining processable work: equals
                                 ``gaps_processable`` (work_list - completed -
                                 no_data, TOIs excluded) once gap stats have
                                 been computed. Falls back to
                                 ``total_tics - examined`` only until the
                                 first background gap-stats compute lands
                                 (``gaps_processable is None``) — that
                                 fallback is a sector-wide estimate against a
                                 different population (the full MAST sector,
                                 not the downloaded work list) and never
                                 reaches 0 at sector close, so it is
                                 transitional only.
        - ``state``            — UI-controlled sector state: ``coming_soon``,
                                 ``active``, ``paused``, ``complete``.
        - ``gaps_processable`` — real remaining work (work_list - completed - no_data).
        - ``no_data``          — TICs permanently marked as dataless.
        - ``pct_complete``     — (completed + no_data) / denominator, where
                                 the denominator is ``work_list_size`` alone
                                 ONLY once the sector's download is
                                 verifiably complete (``downloaded >=
                                 total_tics``, both known and positive);
                                 otherwise ``max(total_tics, work_list_size)``
                                 (the pre-2026-10-05 formula), so an
                                 ``enabled``/``active`` sector still mid-
                                 download never overstates progress against
                                 its not-yet-fully-downloaded work list (the
                                 historical "sector 23 shows 54 % instead of
                                 ~24 %" bug — see the ``download_complete``
                                 comment in ``_compute()`` for the full
                                 rationale). Also computed for CLOSED sectors
                                 now (see the ``_CLOSED_GAP_STATS_*`` block
                                 below) — previously null forever for a
                                 disabled/complete sector, which forced the
                                 UI's cursor-based ``examined/total_tics``
                                 fallback even for a sector with a handful of
                                 real gaps left (s5, s14, s15, s17 2026-10-05).

        The top-level ``active_sector`` is derived as the first sector whose
        ``state == 'active'``, falling back to the registry value for
        back-compat when no sector has been explicitly set to active.

        Caching: the per-sector enrichment (gap stats + state lookup) crosses
        SQLite and PG; the entire response is cached for 30 s via
        ``_cached_json`` to avoid hammering the DB on every UI poll.
        Underneath that 30 s cache, ``downloaded`` (like ``gaps_processable``/
        ``no_data``/``pct_complete``) is itself served from its own
        background-refreshed store with a much longer TTL (30 min) — the 30 s
        outer cache only avoids recomputing ``work_list_size``/state lookups
        on every poll; it does not control how often the expensive index
        scan behind ``downloaded`` actually runs.
        """
        _TGLC_INDEX = str(_PROJECT_ROOT / "data" / "metadata_cache.db")

        def _compute() -> dict:
            # Imports inside _compute so mocks applied before a request are
            # resolved at compute time (not at endpoint-definition time).
            from exonit.discovery_state import (
                load_registry,
                get_sector_state,
                get_autofill,
            )

            try:
                reg = load_registry(_DISCOVERY_REGISTRY)
            except Exception:
                reg = {"active_sector": 1, "sectors": {}}

            reg_active = int(reg.get("active_sector", 1))
            sectors_raw = reg.get("sectors", {})
            sectors_out = []

            # Expensive gap stats come from a background-refreshed store (never
            # blocks this request). Empty until the first background compute
            # lands; per-sector fields then fall back to null and the UI shows
            # cheap examined/total_tics instead.
            gap_data = _get_gap_stats(sectors_raw, _TGLC_INDEX)

            # Closed (disabled/complete, no running autofill) sectors are
            # deliberately skipped by the store above (Q1, 2026-09-26) — they
            # get their OWN background-refreshed store instead, on a 24 h
            # cadence (see the _CLOSED_GAP_STATS_* block above). A sector can
            # only ever be "moving" in one of the two stores at a time, so a
            # plain per-sector fallback (gap_data wins when both somehow have
            # an entry) is enough; no sector-level merge logic needed.
            closed_gap_data = _get_closed_gap_stats(sectors_raw, _TGLC_INDEX)

            # 'downloaded' likewise comes from a background-refreshed store
            # backed by the shared PG/SQLite metadata index — see the
            # _DOWNLOADED_COUNTS_* block above for why (registry-vs-index
            # cross-host staleness, 2026-07-25). Empty until the first
            # background compute lands; per-sector falls back to the
            # registry's own stored value below.
            downloaded_counts = _get_downloaded_counts(sectors_raw, _TGLC_INDEX)

            for s_key, s_entry in sorted(
                    sectors_raw.items(),
                    key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 10**9):
                try:
                    number = int(s_key)
                except ValueError:
                    number = 0

                # `or 0`: a sector entry may carry an explicit null total_tics
                # (e.g. a sector staged for download but not yet sized) — guard
                # it so the docstring's "never raises 500" actually holds (was
                # `None - int` at the `pending` computation below).
                total_tics = s_entry.get("total_tics", 0) or 0
                # Prefer the index-derived count (host-independent, current —
                # see the _DOWNLOADED_COUNTS_* store above); fall back to the
                # registry's own stored value when the index hasn't produced
                # a number for this sector yet (cold cache) or a per-sector
                # compute failed (index unreachable) — a stale number, never
                # a false zero.
                _dc = downloaded_counts.get(number)
                downloaded = _dc if _dc is not None else (
                    s_entry.get("downloaded", 0) or 0)
                cursor = s_entry.get("cursor", 0) or 0
                enabled = s_entry.get("enabled", False)

                canonical_csv = s_entry.get("canonical_list", "")

                # Work-list size (canonical + TGLC index), mtime-cached so it
                # only reloads the ~532k-TIC list when new data is downloaded —
                # not on every UI poll (was ~0.8 s/poll).
                try:
                    work_list_size = _extended_work_list_len(
                        canonical_csv or "", _TGLC_INDEX, sector=number)
                except Exception:
                    work_list_size = 0

                # 'examined' = persistent registry cursor (sectors.json).
                # It survives a DB reset and reflects the sector's progress
                # accumulated across all runs, not the current run's tasks.
                # Key advantage: on AWS the DB is wiped between runs but the
                # registry cursor persists → progress is not lost.
                # At home, cursor ≈ COUNT(completed) (difference <0.5%, verified).
                examined = cursor

                # Per-sector gap metric, read from the background store
                # (cheap) — gap_data (the "moving" sectors) first, falling
                # back to closed_gap_data (closed sectors, 24 h cadence; see
                # the _CLOSED_GAP_STATS_* block above) for a sector that is
                # never going to show up in the former.
                #
                # pct_complete denominator (2026-10-05 fix, revised after
                # review — see progress/review-discovery-closed-sector-gap-
                # stats-2026-10-05.md item 2): registry `total_tics` has
                # drifted into an unreliable TIC count for the handful of
                # CLOSED sectors this fix was diagnosed against (it tracks
                # `downloaded`, a FITS-file count ~0.7-1 % larger than the
                # real work list) — dividing by it made a near-complete
                # sector with a handful of real gaps (e.g. s5: work list
                # 2,564,123, 8 gaps) read as 99.3 % instead of ~100 %.
                # BUT an `enabled=True`/`state=='active'` sector can
                # legitimately still be receiving bulk downloads while under
                # active scan (`discovery_run.py`'s `need_download` gate only
                # requires the CURRENT LOT's TICs, not the whole sector) —
                # for that case `work_list_size` genuinely understates the
                # real sector size (the historical "sector 23 shows 54 %
                # instead of ~24 %" bug this file's `max(total_tics,
                # work_list_size)` formula was written to fix), and dividing
                # by it alone would reintroduce that exact bug.
                #
                # Rule: only drop `total_tics` from the denominator once the
                # sector's download is verifiably COMPLETE — `downloaded >=
                # total_tics` and both are known positive numbers (the one
                # condition that actually proves "nothing left to pull in").
                # Otherwise (still downloading, or either number is
                # unknown/zero) keep the pre-2026-10-05 `max(total_tics,
                # work_list_size)` safety net.
                download_complete = (
                    total_tics > 0 and downloaded > 0 and downloaded >= total_tics
                )
                g = gap_data.get(number)
                if g is None:
                    g = closed_gap_data.get(number)
                if g is not None:
                    gaps_processable = g["gaps_processable"]
                    no_data = g["no_data"]
                    gap_wl = int(g.get("work_list_size") or 0) or work_list_size
                    denom = gap_wl if download_complete else max(total_tics, gap_wl)
                    pct_complete = (
                        (g["completed"] + g["no_data"]) / denom if denom else 0.0
                    )
                    # done-state design (2026-09-26): None with the flag off
                    # (see _compute_all_gap_stats) or before the first
                    # background gap-stats compute.
                    failed_terminal = g.get("failed_terminal")
                else:
                    # Not computed yet — let the UI fall back to examined/total.
                    gaps_processable = None
                    no_data = None
                    pct_complete = None
                    failed_terminal = None

                # Per-sector UI-controlled state (PG discovery_sector_state).
                # An explicit row (active / paused / complete) always wins. When
                # there is no row, get_sector_state returns 'coming_soon'; we then
                # DERIVE from reality so a sector that is enabled or already has
                # progress is not falsely labelled "coming soon" (sector 2 bug):
                # such a sector is 'paused' (in the campaign, not the active
                # focus). Only a genuinely untouched sector stays 'coming_soon'.
                # 'active' is never derived — it comes only from an explicit row —
                # so active_sector stays correct.
                try:
                    state = get_sector_state(number, db_path=db_path)
                except Exception:
                    state = "coming_soon"
                if state == "coming_soon" and (enabled or examined > 0):
                    state = "paused"

                # Autofill status (Task 10 of discovery-ui-source-of-truth plan).
                # Guarded so SQLite deployments and transient PG errors still
                # return a sane response (never a 500).
                try:
                    af_row = get_autofill(number, db_path=db_path)
                    autofill = {
                        "status": af_row.get("status") if af_row else None,
                        "current_lot": af_row.get("current_lot") if af_row else None,
                        "lots_done": int(af_row.get("lots_done") or 0) if af_row else 0,
                        "last_alert": af_row.get("last_alert") if af_row else None,
                    }
                except Exception:
                    autofill = {
                        "status": None, "current_lot": None,
                        "lots_done": 0, "last_alert": None,
                    }

                sectors_out.append({
                    "number": number,
                    "enabled": enabled,
                    "total_tics": total_tics,
                    "work_list_size": work_list_size,
                    "downloaded": downloaded,
                    "cursor": cursor,
                    "examined": examined,
                    # gaps_processable (work_list - completed - no_data, TOIs
                    # excluded) is the honest remaining-work metric and
                    # reaches 0 at sector close. The old total_tics - examined
                    # formula compares two different populations (full MAST
                    # sector vs. the downloaded work list the cursor advances
                    # through) and never reaches 0 — phantom remaining. Only
                    # fall back to it while gap stats haven't computed yet.
                    "pending": (
                        max(0, gaps_processable) if gaps_processable is not None
                        else max(0, total_tics - examined)
                    ),
                    # New fields (Task 4 of discovery-ui-source-of-truth plan)
                    "state": state,
                    "gaps_processable": gaps_processable,
                    "no_data": no_data,
                    "pct_complete": pct_complete,
                    # done-state design (2026-09-26): exhausted(t, sector) —
                    # failed twice for this sector (EXONIT_DONE_STATE only;
                    # None otherwise). Already folded into gaps_processable/
                    # pct_complete above; exposed separately so the UI can
                    # label "given up on" TICs distinctly from "done".
                    "failed_terminal": failed_terminal,
                    # Auto-fill status (Task 10)
                    "autofill": autofill,
                })

            # Derive active_sector: first sector with state=='active',
            # otherwise fall back to the registry value (back-compat for
            # deployments that have not yet set explicit sector states).
            active = next(
                (s["number"] for s in sectors_out if s["state"] == "active"),
                reg_active,
            )
            return {"active_sector": active, "sectors": sectors_out}

        return _cached_json("discovery_sectors", 30.0, _compute)

    def _discovery_set_pipeline_state(state_name: str, *, fresh: bool = False,
                                      **extra) -> None:
        """Write pipeline_state.json with *state_name* (best-effort, atomic).

        Mirrors discovery_orchestrator._set_state so the UI pipeline bar
        reflects the scan/enqueue phase (and any failure) even before the
        orchestrator process takes over. Never raises — state reporting must
        not break the scan endpoint.

        ``fresh=True`` starts from an EMPTY state instead of merging into the
        previous file (same meaning as the orchestrator's ``fresh``). Used for
        the ENQUEUEING write of a new launch: merging kept the PREVIOUS lot's
        run_id / lot_csv / orchestrator_pid / scorecard paths / fleet warnings
        in the mirror for the whole enqueue (15+ min on a 240k-TIC lot), so
        the UI showed the closed lot as if it were enqueueing again (incident
        2026-09-24, s5 lot 5).
        """
        try:
            sf = Path(_PIPELINE_STATE_FILE)
            current = {}
            if not fresh and sf.exists():
                try:
                    current = json.loads(sf.read_text())
                except Exception:
                    current = {}
            current["state"] = state_name
            current["updated_at"] = datetime.now(timezone.utc).isoformat()
            current.update(extra)
            sf.parent.mkdir(parents=True, exist_ok=True)
            tmp = sf.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(current, indent=2))
            tmp.replace(sf)
        except Exception as exc:
            logger.warning("could not write pipeline state %s: %s", state_name, exc)

    #: States that mean a discovery run/pipeline is already in flight.
    _ACTIVE_PIPELINE_STATES = {
        "ENQUEUEING", "SCANNING", "FPP_BACKFILL", "EXOMINER_BACKFILL",
        "STOP_WORKERS", "POSTPROCESS", "AWS_FINALIZE",
        "STOP_NVIDIARACK", "LORA", "FREE_GPU",
    }

    def _mark_orchestrator_spawn_failed(
            run_id: str, detail: str = "orchestrator_spawn_failed") -> None:
        """Best-effort: flip a run to FAILED in PG and the JSON mirror.

        Phase 3 Task 4: the single choke point used whenever a Popen for
        discovery_orchestrator.py raises, or spawns but never confirms (see
        _confirm_orchestrator_started) — the 2026-06-30 orphan-run incident
        this task fixes. Previously a Popen failure only logged a warning and
        fell through to a silent HTTP 200, leaving discovery_runs (and the
        pipeline_state.json mirror the UI reads) stuck ENQUEUEING forever.
        Never raises: called from error paths that must still return a clean
        HTTP response regardless of PG's reachability.
        """
        if run_id:
            try:
                # F6 (final-review fix): finished=True stamps finished_at —
                # this IS a terminal transition (FAILED), same as every other
                # terminal write in discovery_orchestrator.py's _persist_state.
                db.update_discovery_run(run_id, state="FAILED", detail=detail,
                                        finished=True)
            except Exception as exc:
                logger.warning(
                    "could not mark discovery_runs FAILED for run_id=%s: %s",
                    run_id, exc)
        _discovery_set_pipeline_state("FAILED", detail=detail)

    def _spawn_confirm_timeout_and_poll() -> tuple:
        """Read the spawn-confirmation timeout/poll env vars, guarded.

        F3 (final-review fix, deferred Minor): a malformed
        EXONIT_ORCH_SPAWN_CONFIRM_*_SECONDS value must not crash the scan/
        resume endpoint — fall back to the documented defaults instead of
        letting float() raise ValueError.
        """
        try:
            timeout_s = float(os.environ.get(
                "EXONIT_ORCH_SPAWN_CONFIRM_TIMEOUT_SECONDS", "15"))
        except (TypeError, ValueError):
            timeout_s = 15.0
        try:
            poll_s = float(os.environ.get(
                "EXONIT_ORCH_SPAWN_CONFIRM_POLL_SECONDS", "0.5"))
        except (TypeError, ValueError):
            poll_s = 0.5
        return timeout_s, poll_s

    async def _confirm_orchestrator_started(run_id: str, stale_states: set) -> bool:
        """Poll discovery_runs.state until it leaves *stale_states*, or time out.

        Phase 3 Task 4: after a "successful" Popen, the process could still be
        DOA (broken argv, import error, PG unreachable at ITS startup) and
        look identical to a healthy launch from the caller's side — Popen
        only reports whether the OS could exec the interpreter, not whether
        the script went on to do anything. The orchestrator's first act in
        run() is always a state transition away from whatever the row held
        going in (ENQUEUEING for a fresh scan launch — see
        discovery_orchestrator.py's dispatch loop), so polling for that
        transition is a reliable liveness signal without having to parse its
        log or wait() on the Popen handle (which would block on a
        long-running process by design).

        Used ONLY for the scan endpoint's fresh-launch path, where the row is
        known to start in ENQUEUEING. For --resume, see
        _confirm_orchestrator_resumed below — a resumed row can already be in
        ANY non-terminal state, which makes this state-based check vacuous
        there (see that function's docstring).

        Polls PG rather than the Popen object itself so this works uniformly
        whether the caller owns the handle or not. Never raises: a PG read
        hiccup mid-poll is treated as "not yet confirmed" and retried until
        the deadline. Non-blocking (await asyncio.sleep) so the single
        uvicorn event loop stays free to serve other requests while waiting.
        """
        timeout_s, poll_s = _spawn_confirm_timeout_and_poll()
        deadline = _time.time() + timeout_s
        while True:
            try:
                row = db.get_discovery_run(run_id)
            except Exception as exc:
                logger.warning(
                    "spawn-confirmation poll failed for run_id=%s (non-"
                    "fatal, retrying): %s", run_id, exc)
                row = None
            if row is not None and row.get("state") not in stale_states:
                return True
            if _time.time() >= deadline:
                return False
            await asyncio.sleep(poll_s)

    async def _confirm_orchestrator_resumed(
            run_id: str, baseline_updated_at, baseline_heartbeat_at) -> bool:
        """Poll discovery_runs until updated_at/heartbeat_at advance past the
        given baseline, or time out.

        F3 (final-review fix): _confirm_orchestrator_started(run_id,
        {"FAILED", "ENQUEUEING"}) is VACUOUS for a resume whose persisted
        state is already non-terminal (SCANNING/POSTPROCESS/LORA/...) — that
        state is never in {"FAILED", "ENQUEUEING"} to begin with, so the OLD
        code returned True on the very first poll regardless of whether the
        new process ever executed a single line. Progress is the only real
        signal available for a resume: discovery_orchestrator.py's F1 fix
        makes the freshly-(re)started run() stamp its own pid + heartbeat as
        its very first act, before doing anything else — that write bumps
        BOTH updated_at (unconditionally, on every update_discovery_run call)
        and heartbeat_at.

        The baseline MUST be captured AFTER this endpoint's OWN
        "record the new pid in PG" write (not merely "before the Popen" in
        isolation) — that write itself bumps updated_at unconditionally, so
        reading the baseline any earlier would let our OWN housekeeping
        satisfy the check with zero information about whether the child
        process is alive. Captured this way, a DOA child (Popen's exec
        succeeds but the process crashes/exits before reaching run()'s F1
        stamp) never advances the row further → timeout → caller reports
        orchestrator_spawn_failed, exactly like the scan endpoint's path.
        """
        timeout_s, poll_s = _spawn_confirm_timeout_and_poll()
        deadline = _time.time() + timeout_s
        while True:
            try:
                row = db.get_discovery_run(run_id)
            except Exception as exc:
                logger.warning(
                    "resume spawn-confirmation poll failed for run_id=%s "
                    "(non-fatal, retrying): %s", run_id, exc)
                row = None
            if row is not None and (
                    row.get("updated_at") != baseline_updated_at
                    or row.get("heartbeat_at") != baseline_heartbeat_at):
                return True
            if _time.time() >= deadline:
                return False
            await asyncio.sleep(poll_s)

    @app.post("/api/discovery/scan")
    async def api_discovery_scan(request: Request):
        """Prepare and enqueue the next discovery lot for a sector.

        Calls scripts/discovery_run.py as a subprocess, captures its JSON
        output, and maps exit codes to HTTP status codes:
          200  started / dry_run
          409  need_download  (not enough TGLC data downloaded)
          400  sector_not_enabled / error / no canonical list
          500  enqueue_failed

        """
        try:
            from exonit.discovery_state import active_sector, derive_run_id
            body = await request.json()
            # Sector: explicit request value, else this deployment's active
            # sector from the registry (home=1, AWS=2). No hardcoded default.
            sector = int(body.get("sector") or active_sector(_DISCOVERY_REGISTRY))
            count = int(body.get("count", 100))
        except Exception as exc:
            return JSONResponse(
                {"status": "error", "reason": "bad_request", "detail": str(exc)},
                status_code=400,
            )

        # Task 7 (deep-pass multisector): optional mode=deep_rescan drives
        # scripts/discovery_run.py's --tics-csv/--force-rescan flags (Task 6).
        # Validated BEFORE the anti-double-launch guard below so a bad
        # request never flips pipeline state to ENQUEUEING.
        mode = body.get("mode")
        tics_csv = body.get("tics_csv")
        if mode not in (None, "deep_rescan"):
            return JSONResponse(
                {"status": "error", "reason": "unknown_mode", "mode": mode},
                status_code=400)
        if mode == "deep_rescan" and not tics_csv:
            return JSONResponse(
                {"status": "error", "reason": "deep_rescan_requires_tics_csv"},
                status_code=400)

        # D (2026-10-05): one-off mop-up lot of a CLOSED sector through the
        # normal full pipeline (e.g. sector 15's ~2.5k ready gaps), without
        # flipping it back to `enabled` in the registry. Maps 1:1 onto
        # scripts/discovery_run.py's existing --allow-disabled/--full-scan
        # flags (both pre-existing — used today by the autofill manager and
        # verify_sector_complete.py); this just lets the UI's one-shot Scan
        # button reach them too. Strictly typed to bool only (no "true"/1/
        # truthy-string coercion) so a stray non-bool value can never
        # silently enable scanning a disabled sector. Rejected outright
        # together with mode=deep_rescan: that mode already has its own
        # enable/force semantics (--force-rescan always re-enqueues
        # regardless of `enabled`), and no combination of it with these two
        # flags has ever been exercised — reject with 400 rather than guess.
        allow_disabled = body.get("allow_disabled", False)
        full_scan = body.get("full_scan", False)
        if not isinstance(allow_disabled, bool) or not isinstance(full_scan, bool):
            return JSONResponse(
                {"status": "error",
                 "reason": "allow_disabled_and_full_scan_must_be_bool"},
                status_code=400)
        if mode == "deep_rescan" and (allow_disabled or full_scan):
            return JSONResponse(
                {"status": "error",
                 "reason": "allow_disabled_full_scan_incompatible_with_deep_rescan"},
                status_code=400)

        # Anti-double-launch guard. With a single uvicorn event loop, a second
        # scan fired while one is still in flight used to just stack another
        # enqueue (clicking "Scan" repeatedly did exactly that). Refuse cleanly
        # with 409 if a run/pipeline is already active.
        _sf_guard = Path(_PIPELINE_STATE_FILE)
        if _sf_guard.exists():
            try:
                _cur_state = json.loads(_sf_guard.read_text())
            except Exception:
                _cur_state = {}
            if _cur_state.get("state") in _ACTIVE_PIPELINE_STATES:
                return JSONResponse(
                    {"status": "already_running",
                     "state": _cur_state.get("state"),
                     "detail": ("a discovery run is already in progress; wait "
                                "for it to finish or reset the pipeline")},
                    status_code=409,
                )

        # Mark ENQUEUEING immediately: the UI bar is honest from the first
        # moment, and a concurrent click now trips the guard above.
        _discovery_set_pipeline_state(
            "ENQUEUEING", fresh=True, sector=sector, requested=count,
            started_at=datetime.now(timezone.utc).isoformat())

        # Q1 (perf review 2026-09-26): a new lot starting is exactly the lot
        # boundary where a stale gap-stats number (now cached up to 1 h,
        # EXONIT_GAP_STATS_TTL_S) is most visible to an operator watching the
        # sectors page. Force a background recompute now instead of waiting
        # out the TTL; never blocks this request (see docstring). S2
        # (2026-09-26 §5.2): scoped to THIS sector only — the other
        # registered sectors are untouched by a lot starting here.
        try:
            from exonit.discovery_state import load_registry as _load_reg_q1
            _reg_q1 = _load_reg_q1(_DISCOVERY_REGISTRY)
            _force_gap_stats_refresh_now(
                _reg_q1.get("sectors", {}),
                str(_PROJECT_ROOT / "data" / "metadata_cache.db"),
                sector=sector)
        except Exception as exc:
            logger.warning("[gap_stats] scan-start force refresh failed: %s", exc)

        # B (2026-10-05): a lot starting against a CLOSED sector (the
        # allow_disabled mop-up case above) must also invalidate/recompute
        # THAT sector's snapshot in the SEPARATE closed-sector store — the
        # live gap_data store's force-refresh just above is a no-op for a
        # closed sector by design (_compute_all_gap_stats skips it), so
        # without this a mop-up lot's progress would sit invisible in the UI
        # for up to the closed store's own 24 h TTL. Scoped to this sector
        # only, same never-blocks shape as the live-store refresh above.
        try:
            _sector_entry_q1 = _reg_q1.get("sectors", {}).get(str(sector), {})
            if _is_closed_sector(sector, _sector_entry_q1):
                _force_closed_gap_stats_refresh_now(
                    _reg_q1.get("sectors", {}),
                    str(_PROJECT_ROOT / "data" / "metadata_cache.db"),
                    sector=sector)
        except Exception as exc:
            logger.warning(
                "[closed_gap_stats] scan-start force refresh failed: %s", exc)

        cmd = [
            sys.executable, str(_PROJECT_ROOT / "scripts" / "discovery_run.py"),
            "--sector", str(sector),
            "--count", str(count),
            "--db", db_path,
            "--registry", _DISCOVERY_REGISTRY,
        ]
        if mode == "deep_rescan":
            cmd += ["--tics-csv", str(tics_csv), "--force-rescan"]
        if allow_disabled:
            cmd += ["--allow-disabled"]
        if full_scan:
            cmd += ["--full-scan"]
        # Run discovery_run.py WITHOUT blocking the event loop. The old code
        # used subprocess.Popen + communicate() inside this async handler,
        # which froze the single uvicorn event loop for the entire enqueue.
        # While frozen, the server could not even answer the /enqueue_bulk POSTs
        # that enqueue_from_csv.py makes back to it, so enqueue fell back to the
        # slow per-target path and eventually hit the 600 s timeout (only
        # ~480/1000 enqueued). asyncio.create_subprocess_exec + await keeps the
        # loop free, so the bulk path works and 1000 TICs enqueue in seconds.
        # Own process group (start_new_session) so a timeout can SIGKILL the
        # whole tree incl. the enqueue_from_csv.py grandchild (2026-05-28 orphan
        # glob incident).
        import signal as _signal
        # Enqueue timeout, env-tunable (EXONIT_SCAN_TIMEOUT_S, default 600).
        # 2026-07-18: a 300k enqueue crept past 600 s (549 s at lot 1 of the
        # same size, >600 at lot 3) while sector downloads shared the disk and
        # the TGLC index grew to 22M rows -- the kill left 54k orphan tasks.
        # The value is a backstop against a HUNG enqueue, not a pace-setter,
        # so a generous ceiling is safe.
        scan_timeout_s = float(os.environ.get("EXONIT_SCAN_TIMEOUT_S", "600"))
        # ENQUEUEING is the one phase the ORCHESTRATOR does not own — this
        # endpoint spawns discovery_run.py directly — so the server is the one
        # that hands it its progress identity. It reports under the fixed
        # ENQUEUEING key rather than a run_id, because this run's id is
        # derived from the lot CSV the child is still writing (it is minted
        # further down, once lot_csv exists).
        scan_env = progress_model.child_env(
            progress_model.ENQUEUEING_RUN_ID, "ENQUEUEING")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(_PROJECT_ROOT),
                start_new_session=True,
                env=scan_env,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=scan_timeout_s)
            except asyncio.TimeoutError:
                try:
                    os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                try:
                    await proc.wait()  # reap the killed process
                except Exception:
                    pass
                _discovery_set_pipeline_state(
                    "FAILED",
                    detail=f"scan timed out after {scan_timeout_s:.0f} s")
                return JSONResponse(
                    {"status": "error",
                     "error": f"discovery_run.py timed out after {scan_timeout_s:.0f} s; "
                              "process group killed"},
                    status_code=500,
                )
            stdout = stdout_b.decode("utf-8", "replace") if stdout_b else ""
            stderr = stderr_b.decode("utf-8", "replace") if stderr_b else ""
            result = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
        except Exception as exc:
            _discovery_set_pipeline_state("FAILED", detail=f"scan exec error: {exc}")
            return JSONResponse({"error": str(exc), "status": "error"}, status_code=500)

        # Parse first JSON line from stdout
        raw_json = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    raw_json = json.loads(line)
                except ValueError:
                    pass
                break

        status = raw_json.get("status", "error")
        if status in ("started", "dry_run"):
            # chunk4: kick off post-run orchestrator on 'started'.
            # Guard against double-launch: skip if an orchestrator is already
            # running per the state file (state not in {IDLE, DONE, FAILED}).
            if status == "started":
                # Forget the previous run's progress baselines. They are
                # set-once in the shared store and `done = baseline - pending`,
                # so without this a fresh lot inherits the last one's counters
                # and is born already "done": the 100k lot of 2026-07-26 opened
                # at "200,049 of 299,944", the figures of the 300k before it,
                # and its header, bar, ETA and time-remaining stayed wrong for
                # the whole run. the operator had to press "Reset stats" by hand after
                # every launch — and this endpoint is THE way runs are started,
                # by the UI button and by the chained-lot runner alike.
                # Placed here, after the enqueue has committed: the next
                # /api/health/work re-seeds the baseline from the new lot's
                # own pending count.
                try:
                    _server.reset_progress_baselines()
                except Exception:
                    logger.warning("scan: could not reset the "
                                   "progress baselines", exc_info=True)
                # Bust the health_work cache (10 s TTL) so the orchestrator's
                # FIRST SCANNING poll sees the just-committed tasks, not a stale
                # pre-enqueue pending=0 snapshot. Without this the orchestrator
                # raced the cache, concluded "primary tasks drained" on its very
                # first poll (0 s after launch), ran `farmctl stop --all` and
                # closed out the run with 50k tasks still pending (2026-06-26 S23).
                try:
                    _st_bust = get_shared_state()
                    for _ck in ("cache:health_work", "cache:status",
                                "cache:stats", "cache:backfill_fpp",
                                "cache:backfill_nfpp"):
                        _st_bust.delete(_ck)
                except Exception:
                    pass
                lot_csv = raw_json.get("lot_csv", "")
                # Derived from the canonical set, MINUS ENQUEUEING: at this
                # point the state file says ENQUEUEING from THIS very run, so
                # including it would block our own orchestrator launch.
                _active_states = _ACTIVE_PIPELINE_STATES - {"ENQUEUEING"}
                _sf = Path(_PIPELINE_STATE_FILE)
                _already_running = False
                if _sf.exists():
                    try:
                        _prev = json.loads(_sf.read_text())
                        if _prev.get("state") in _active_states:
                            _already_running = True
                    except Exception:
                        pass
                if not _already_running and lot_csv:
                    import tempfile as _tempfile
                    _log_fd, _log_path = _tempfile.mkstemp(
                        prefix="discovery_orch_", suffix=".log")
                    os.close(_log_fd)
                    # Phase 3 Task 2 (durable orchestration): mint run_id ONCE,
                    # here, and create its discovery_runs PG row before
                    # spawning the orchestrator — which receives it via
                    # --run-id and must NEVER re-derive it (that
                    # re-derivation, hours later at POSTPROCESS, was the
                    # midnight-crossing bug this fixes). Naming MUST match
                    # the historical convention (_lot_stamp_from_csv is a
                    # deliberate copy of the orchestrator's own regex).
                    # F5 (final-review fix): mint the SAME UTC date source
                    # _derive_run_id uses (discovery_orchestrator.py), not
                    # local date.today() — a local-vs-UTC midnight straddle
                    # used to be able to mint a run_id here that the
                    # orchestrator's own (UTC) convention would never derive
                    # to independently, silently defeating the "never
                    # re-derive run_id" invariant if anything ever fell back
                    # to re-deriving.
                    # FIX-1 (final-review, spec §4): the "lot" literal is now
                    # a deep-aware prefix (_lot_prefix_from_csv) instead of
                    # hardcoded, so deep-rescan lots mint a distinctive
                    # "deep_lot<ts>" run_id segment while normal lots are
                    # byte-identical to before (still "lot<ts>"). Must derive
                    # the SAME run_id as discovery_orchestrator.py's
                    # _derive_run_id for the same lot_csv (see that
                    # function's docstring for the invariant).
                    #
                    # S3 (docs/superpowers/specs/2026-09-26-enqueue-s1-s4-
                    # design.md §3.1): "the child wins" — discovery_run.py
                    # now mints run_id itself (before writing tasks.run_id at
                    # enqueue time) and returns it as raw_json["run_id"].
                    # Preferring that value here — instead of re-deriving —
                    # means this route and the enqueued tasks always agree,
                    # even in the one case they otherwise wouldn't: a scan
                    # that straddles UTC midnight between the child minting
                    # its run_id and this parent reaching this line. A run
                    # started by an OLDER discovery_run.py (no "run_id" key
                    # in its JSON output) falls back to the same derivation
                    # as before (now the shared exonit.discovery_state.
                    # derive_run_id, S3's unification of this file's/the
                    # orchestrator's two identical copies).
                    run_id = raw_json.get("run_id") or derive_run_id(
                        sector, lot_csv, now_utc=datetime.now(timezone.utc))
                    try:
                        db.create_discovery_run(
                            run_id, sector=sector, lot_csv=lot_csv,
                            dry_run=False, chained=False,
                            server_url=f"http://localhost:{cfg.port}")
                    except Exception as _exc:
                        # F4 (final-review fix, deploy-order hazard): this
                        # used to be warning-then-continue — the orchestrator
                        # would still get spawned and run to completion, but
                        # the spawn-confirmation poll below could NEVER see a
                        # row that was never created (e.g. schema_pg.sql not
                        # yet applied to a fresh PG), so after its timeout it
                        # marks the MIRROR FAILED while a perfectly healthy
                        # orchestrator keeps running underneath — the
                        # anti-double-launch guard reopens and a second scan
                        # can double-launch against the same lot. Fail FAST,
                        # before the Popen, instead.
                        logger.warning(
                            "could not create discovery_runs row for "
                            "run_id=%s: %s", run_id, _exc)
                        _mark_orchestrator_spawn_failed(
                            run_id, detail=f"run_record_create_failed: {_exc}")
                        return JSONResponse(
                            {"status": "error",
                             "reason": "run_record_create_failed"},
                            status_code=502)
                    _orch_cmd = [
                        sys.executable,
                        str(_PROJECT_ROOT / "scripts" / "discovery_orchestrator.py"),
                        "--lot-csv", lot_csv,
                        "--sector", str(sector),
                        "--server-url", f"http://localhost:{cfg.port}",
                        "--state-file", _PIPELINE_STATE_FILE,
                        "--run-id", run_id,
                        *(["--db", db_path] if db_path else []),
                    ]
                    try:
                        with open(_log_path, "wb") as _lf:
                            _proc = subprocess.Popen(
                                _orch_cmd,
                                stdout=_lf,
                                stderr=subprocess.STDOUT,
                                cwd=str(_PROJECT_ROOT),
                            )
                    except Exception as _exc:
                        # Phase 3 Task 4 (2026-06-30 orphan-run incident): a
                        # Popen failure used to just log a warning and fall
                        # through to the 200 below, leaving discovery_runs
                        # (and the JSON mirror) stuck ENQUEUEING forever with
                        # nothing ever going to work it. Now: FAILED + 502.
                        logger.warning("could not spawn orchestrator: %s", _exc)
                        _mark_orchestrator_spawn_failed(run_id)
                        return JSONResponse(
                            {"status": "error",
                             "reason": "orchestrator_spawn_failed"},
                            status_code=502)

                    _discovery_orchestrator["proc"] = _proc
                    _reap_in_background(_proc)
                    _discovery_orchestrator["started_at"] = _time.time()
                    raw_json["orchestrator_pid"] = _proc.pid
                    raw_json["run_id"] = run_id

                    # Phase 3 Task 4: record the pid in PG (the JSON mirror
                    # only gets it from the orchestrator's OWN first
                    # _set_state call — a gap that would leave discovery_runs
                    # .orchestrator_pid NULL forever, silently defeating both
                    # the dead-run watchdog and this endpoint's own guard on
                    # /api/discovery/resume, both of which key off this
                    # column). Best-effort: a failed write here must not fail
                    # an otherwise-healthy scan.
                    try:
                        db.update_discovery_run(run_id, pid=_proc.pid)
                    except Exception as _exc:
                        logger.warning(
                            "could not record orchestrator_pid for "
                            "run_id=%s (non-fatal): %s", run_id, _exc)

                    # Phase 3 Task 4: confirm the orchestrator actually took
                    # over (moved off ENQUEUEING) before reporting success —
                    # see _confirm_orchestrator_started's docstring.
                    confirmed = await _confirm_orchestrator_started(
                        run_id, {"ENQUEUEING"})
                    if not confirmed:
                        _mark_orchestrator_spawn_failed(run_id)
                        return JSONResponse(
                            {"status": "error",
                             "reason": "orchestrator_spawn_failed"},
                            status_code=502)
            return JSONResponse(raw_json, status_code=200)
        elif status == "need_download":
            _discovery_set_pipeline_state(
                "FAILED", detail="need_download: not enough TGLC data downloaded")
            return JSONResponse(raw_json, status_code=409)
        elif status == "enqueue_failed":
            _discovery_set_pipeline_state(
                "FAILED", detail=raw_json.get("detail", "enqueue_failed"))
            return JSONResponse(raw_json, status_code=500)
        else:
            # sector_not_enabled, canonical_list_not_found, error, etc.
            if not raw_json:
                raw_json = {"status": "error", "detail": result.stderr[:500]}
            _discovery_set_pipeline_state(
                "FAILED", detail=raw_json.get("reason") or raw_json.get("detail")
                or "scan error")
            return JSONResponse(raw_json, status_code=400)

    @app.post("/api/discovery/resume")
    async def api_discovery_resume(request: Request):
        """Manually re-spawn the orchestrator for a stalled/dead run.

        Phase 3 Task 4: the operator-facing recovery action for a run the
        dead-run watchdog (or the scan endpoint's own spawn confirmation)
        marked FAILED — or one that is still non-terminal but whose
        orchestrator process is confirmed dead. Deliberately never invoked
        automatically: the operator decides when to resume (operator rule:
        the operator launches the discovery runs).

        Body: optional ``{"run_id": "..."}``. Without it, the candidate is
        the single active (non-terminal) discovery_runs row if one exists,
        else the most recently FAILED one.

        Responses:
          200  {"status": "resumed", "run_id": ...}
          404  no candidate run (explicit run_id not found, or nothing to
               resume at all)
          409  {"status": "error", "reason": "already_done" | "orchestrator_alive"}
          502  {"status": "error", "reason": "orchestrator_spawn_failed"}
               (Popen raised, or the spawn was never confirmed within the
               same window the scan endpoint uses)
        """
        try:
            body = await request.json()
            if not isinstance(body, dict):
                body = {}
        except Exception:
            body = {}
        explicit_run_id = body.get("run_id")

        if explicit_run_id:
            run = db.get_discovery_run(explicit_run_id)
            if run is None:
                return JSONResponse(
                    {"status": "error", "reason": "not_found",
                     "detail": f"no discovery_runs row for run_id={explicit_run_id}"},
                    status_code=404)
        else:
            run = db.get_active_discovery_run() or db.get_latest_failed_discovery_run()
            if run is None:
                return JSONResponse(
                    {"status": "error", "reason": "nothing_to_resume"},
                    status_code=404)

        run_id = run["run_id"]

        if run["state"] == "DONE":
            return JSONResponse(
                {"status": "error", "reason": "already_done", "run_id": run_id},
                status_code=409)

        # Guard 1: this run's OWN recorded orchestrator is still alive —
        # never double-spawn a second orchestrator against the same run_id.
        # _pid_is_alive_matching (not bare _pid_is_alive): pids get reaped
        # promptly now (2026-07-07 fix) and can be recycled by an unrelated
        # process — see its docstring (F4).
        if _pid_is_alive_matching(run.get("orchestrator_pid"),
                                   _ORCHESTRATOR_CMDLINE_MARKER):
            return JSONResponse(
                {"status": "error", "reason": "orchestrator_alive", "run_id": run_id},
                status_code=409)

        # Guard 2: defense in depth — the JSON mirror is the guard every
        # other endpoint trusts (the scan endpoint's own double-launch guard,
        # above). If it shows a pipeline actively in flight with a live
        # orchestrator, refuse regardless of what THIS row's own pid column
        # looks like (covers e.g. a stale orchestrator_pid left over on the
        # SAME run_id from a previous attempt, or the at-most-one-active-run
        # invariant being violated).
        _sf_guard = Path(_PIPELINE_STATE_FILE)
        if _sf_guard.exists():
            try:
                _cur_state = json.loads(_sf_guard.read_text())
            except Exception:
                _cur_state = {}
            if (_cur_state.get("state") in _ACTIVE_PIPELINE_STATES
                    and _pid_is_alive_matching(_cur_state.get("orchestrator_pid"),
                                                _ORCHESTRATOR_CMDLINE_MARKER)):
                return JSONResponse(
                    {"status": "error", "reason": "orchestrator_alive", "run_id": run_id},
                    status_code=409)

        force = run["state"] == "FAILED"
        _orch_cmd = [
            sys.executable,
            str(_PROJECT_ROOT / "scripts" / "discovery_orchestrator.py"),
            "--resume", run_id,
            "--state-file", _PIPELINE_STATE_FILE,
            *(["--force"] if force else []),
            *(["--db", db_path] if db_path else []),
        ]
        import tempfile as _tempfile
        _log_fd, _log_path = _tempfile.mkstemp(
            prefix="discovery_orch_resume_", suffix=".log")
        os.close(_log_fd)
        try:
            with open(_log_path, "wb") as _lf:
                _proc = subprocess.Popen(
                    _orch_cmd,
                    stdout=_lf,
                    stderr=subprocess.STDOUT,
                    cwd=str(_PROJECT_ROOT),
                )
        except Exception as _exc:
            logger.warning(
                "could not spawn orchestrator for --resume %s: %s",
                run_id, _exc)
            _mark_orchestrator_spawn_failed(run_id)
            return JSONResponse(
                {"status": "error", "reason": "orchestrator_spawn_failed"},
                status_code=502)

        _discovery_orchestrator["proc"] = _proc
        _reap_in_background(_proc)
        _discovery_orchestrator["started_at"] = _time.time()

        # Phase 3 Task 4: record the new pid in PG — same rationale as the
        # scan endpoint's identical write (see its comment): otherwise a
        # resumed run's orchestrator_pid would keep pointing at the OLD
        # (dead) process forever, breaking both the watchdog and a second
        # /api/discovery/resume call's own alive-guard.
        try:
            db.update_discovery_run(run_id, pid=_proc.pid)
        except Exception as _exc:
            logger.warning(
                "could not record orchestrator_pid for run_id=%s "
                "(non-fatal): %s", run_id, _exc)

        # F3 (final-review fix): the scan endpoint's state-based confirmation
        # (_confirm_orchestrator_started) is vacuous here — a resumed row's
        # persisted state can already be ANY non-terminal state (SCANNING,
        # POSTPROCESS, ...), so {"FAILED", "ENQUEUEING"} would never contain
        # it and this would return True instantly with zero evidence the new
        # process is even alive. Confirm by OBSERVED PROGRESS instead: read
        # the baseline AFTER our own pid write above (see
        # _confirm_orchestrator_resumed's docstring for why exactly there),
        # then wait for the row to move again — which only the new process's
        # own F1 pid/heartbeat stamp (or a further state transition) can do.
        try:
            _baseline_row = db.get_discovery_run(run_id) or {}
        except Exception as _exc:
            logger.warning(
                "could not read baseline row for run_id=%s before resume "
                "confirmation (non-fatal, baseline treated as unknown): %s",
                run_id, _exc)
            _baseline_row = {}
        confirmed = await _confirm_orchestrator_resumed(
            run_id, _baseline_row.get("updated_at"),
            _baseline_row.get("heartbeat_at"))
        if not confirmed:
            _mark_orchestrator_spawn_failed(run_id)
            return JSONResponse(
                {"status": "error", "reason": "orchestrator_spawn_failed"},
                status_code=502)

        return {"status": "resumed", "run_id": run_id}

    @app.get("/api/discovery/pipeline")
    def api_discovery_pipeline():
        """Return the current orchestrator state from pipeline_state.json.

        Returns {"state":"IDLE","never_run":true} if the file does not exist.
        Robust to missing or malformed JSON — never raises 500.

        2026-07-27: also carries ``progress`` — the live record for the current
        (run_id, state) written by whichever script or server-side feeder owns
        that phase (see exonit/progress.py), or null when nothing has reported.
        Null is the pre-existing behaviour: the UI falls back to the historic
        prior, so a phase that does not report yet looks exactly as before.
        """

        state_path = Path(_PIPELINE_STATE_FILE)
        if not state_path.exists():
            return {"state": "IDLE", "never_run": True, "progress": None}
        try:
            body = json.loads(state_path.read_text())
        except Exception:
            return {"state": "IDLE", "never_run": True, "parse_error": True,
                    "progress": None}

        body["progress"] = None
        phase = body.get("state")
        run_id = progress_model.progress_lookup_run_id(phase, body.get("run_id"))
        if run_id and phase:
            try:
                body["progress"] = progress_model.read_progress(
                    get_shared_state(), run_id, phase)
            except Exception as exc:
                # Telemetry must never take the pipeline endpoint down.
                logger.warning("could not read progress for %s/%s: %s",
                               run_id, phase, exc)

        body["eta_total_s"] = None
        try:
            # `lot` = number of TICs in this run, which is what the
            # lot-scaling priors are expressed per 1k of. It is NOT the
            # current phase's total (buckets, hosts, steps...). The SCANNING
            # record's total IS the lot size (one primary task per TIC), so
            # read that one — it survives for the whole run under its own key.
            lot = None
            scanning = progress_model.read_progress(
                get_shared_state(), run_id, "SCANNING") if run_id else None
            if scanning:
                lot = scanning.get("total")
            body["eta_total_s"] = progress_model.eta_total_s(
                state=phase or "", progress=body.get("progress"),
                priors=progress_model.load_priors(), lot=lot)
        except Exception as exc:
            logger.warning("could not compute total ETA: %s", exc)
        return body

    @app.post("/api/discovery/download")
    async def api_discovery_download(request: Request):
        """Spawn scripts/discovery_download.py in the background.

        Returns {"status":"started","pid":N} immediately.
        Returns 409 if a download is already running.

        Cross-process guard: reads {pid, log_path, started_at} from shared_state
        (Redis-backed in multi-worker mode) so any uvicorn worker can detect a
        download started by a sibling process.  The Popen object is kept only in
        the local per-process dict — it is not serialisable.
        """
        import time as _time_mod

        # --- Cross-process anti-double guard -----------------------------------
        # Check shared_state first so all workers see the same running state.
        _st = get_shared_state()
        existing = _st.hget_json(_DOWNLOAD_STATE_KEY, _DL_FIELD)
        if existing is not None:
            existing_pid = existing.get("pid")
            if existing_pid is not None:
                if _check_pid_running_for_download(existing_pid):
                    return JSONResponse({"status": "already_running"}, status_code=409)

        from exonit.discovery_state import active_sector
        body = await request.json()
        # Sector from request, else this deployment's active sector (registry).
        sector = int(body.get("sector") or active_sector(_DISCOVERY_REGISTRY))
        count = int(body.get("count", 500))

        import tempfile
        log_fd, log_path = tempfile.mkstemp(prefix="discovery_dl_", suffix=".log")
        os.close(log_fd)

        cmd = [
            sys.executable, str(_PROJECT_ROOT / "scripts" / "discovery_download.py"),
            "--sector", str(sector),
            "--count", str(count),
            "--registry", _DISCOVERY_REGISTRY,
            "--progress-log", log_path,  # route per-file progress into this file
        ]
        try:
            # Open in APPEND mode so the child's inherited fd starts at EOF.
            # mkstemp already created the file (empty); "wb" would reset the
            # file-position to 0 and any write from discovery_download.py
            # (stdout) would overwrite the beginning of the file while
            # download_tglc_batch.py (which opens --log with its own "a"
            # handle) writes at EOF.  With O_APPEND both handles are
            # position-independent and always write to EOF atomically.
            with open(log_path, "ab") as log_fh:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    cwd=str(_PROJECT_ROOT),
                )
            # log_fh is closed in the parent; child keeps its inherited fd
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

        # Keep the Popen handle locally (not serialisable), tagged with a
        # fresh token identifying THIS spawn — see the "token" comment on
        # _discovery_download's definition above (review F5, PID recycling).
        _dl_token = uuid.uuid4().hex
        _discovery_download["proc"] = proc
        _discovery_download["token"] = _dl_token
        # Zombie-leak fix (2026-07-07): reap on exit from THIS worker. The
        # status endpoint's local_proc.poll() only reaps when the request
        # happens to land on the worker that owns the handle (~1/4 of the
        # time under A3), so without this a finished download can linger as
        # a <defunct> until the server restarts — same fix as the orchestrator.
        _reap_in_background(proc)

        # Persist cross-process metadata to shared_state.  Any worker can now
        # derive running/finished state by checking whether pid is alive.
        _st.hset_json(_DOWNLOAD_STATE_KEY, _DL_FIELD, {
            "pid": proc.pid,
            "log_path": log_path,
            "started_at": _time_mod.time(),
            "token": _dl_token,
        })

        return {"status": "started", "pid": proc.pid}

    @app.get("/api/discovery/download/status")
    def api_discovery_download_status():
        """Report background download state.

        Reads cross-process metadata (pid, log_path, started_at) from
        shared_state so all uvicorn workers return the correct state
        regardless of which worker handled the POST.

        Running/finished derivation (cross-process):
        - ``running`` = psutil.pid_exists(pid) AND cmdline contains
          "discovery_download" (best-effort PID-recycling guard).  Falls
          back to os.kill(pid, 0) if psutil is unavailable.
        - ``returncode``: cannot be obtained cross-process (only the parent
          Popen can call poll()/wait()).  Derived from the log instead:
          0 if the log contains a ``[finish]`` line; None if still running.
          The local proc handle is used as a fast-path when available.
        - ``finished``: pid is dead (not running).  If the log has ``[finish]``
          → completed OK; if dead without ``[finish]`` → died mid-run.

        Parses the log file for done/fail/skip counts produced by
        download_tglc_batch.py progress lines. Returns sane zeros when
        no download has ever been started.
        """
        # Primary source of truth: shared_state (cross-process).
        _st = get_shared_state()
        dl_meta = _st.hget_json(_DOWNLOAD_STATE_KEY, _DL_FIELD)

        if dl_meta is None:
            return {"running": False, "never_run": True,
                    "done": 0, "failed": 0, "skipped": 0,
                    "total": 0, "returncode": None}

        pid = dl_meta.get("pid")
        log_path = dl_meta.get("log_path")

        # Derive running state: alive AND not a zombie AND the right process.
        # See _check_pid_running_for_download() for the full rationale.
        running = False
        if pid is not None:
            running = _check_pid_running_for_download(pid)

        # Fast-path: if this process owns the Popen handle, call poll() to reap
        # any zombie and return the real exit code.
        # NOTE: Popen.returncode is only populated after poll()/wait() is called.
        # Without poll(), a finished child stays a zombie and returncode is None
        # forever.  poll() both reaps the child and returns the exit code.
        #
        # pid alone is NOT enough to prove local_proc is the handle for THIS
        # download (2026-07-10 review F5): _reap_in_background now reaps a
        # finished download's pid promptly, so the kernel can hand that same
        # pid to a BRAND NEW download spawned on another uvicorn worker
        # before this worker's next status poll — matching on pid alone
        # would then report the OLD, already-finished download's returncode
        # as if it belonged to the new one. The token (minted fresh at every
        # spawn, see _discovery_download's definition) disambiguates.
        local_proc = _discovery_download.get("proc")
        local_token = _discovery_download.get("token")
        if (local_proc is not None and local_proc.pid == pid
                and local_token is not None and local_token == dl_meta.get("token")):
            returncode = local_proc.poll()  # reaps zombie + returns real exit code
        else:
            # Cross-process path: we don't own the Popen handle, so poll() is
            # unavailable.  Derive returncode from JSON markers in the log.
            # (Old logs without JSON markers fall back to the [finish] heuristic.)
            returncode = None

        done = failed = skipped = total = 0
        rate = None    # files/sec — last seen on a progress line
        eta_min = None  # minutes remaining — last seen on a progress line
        log_has_finish = False   # [finish] line seen (old-format compat)
        log_has_done_json = False   # {"status":"done"} JSON line seen
        log_has_index_start = False  # {"status":"index_start"} seen
        log_has_upload_start = False  # {"status":"upload_start"} seen (MinIO)
        log_has_index_failed = False  # {"status":"index_failed"} seen
        log_has_progress_line = False  # any "[N/M] ..." per-file line seen
        # Bulk-script fetch phase (silent ~25-40min curl of the MAST .sh on a
        # big sector, previously invisible to this endpoint — see
        # progress/impl-download-observability-2026-07-04.md).
        # NOTE: unlike select_start/index_start/upload_start (genuinely
        # one-shot — nothing else is emitted after them until the phase
        # changes), bulk_fetch_start is followed by a long stream of
        # same-phase bulk_fetch siblings (one every ~5s while curl runs). On
        # a long fetch (e.g. sector 14: 2.42 GB, 25-40 min) that pushes the
        # single bulk_fetch_start line out of the tail-200-line window well
        # before the fetch finishes. So this flag is set True by EITHER
        # "bulk_fetch_start" OR "bulk_fetch" — any surviving line from this
        # phase inside the window is enough to know we're still in it —
        # while log_has_bulk_fetch_done (a true one-shot terminal marker)
        # supersedes it once seen.
        log_in_bulk_fetch_phase = False   # {"status":"bulk_fetch_start"|"bulk_fetch"} seen
        log_has_bulk_fetch_done = False   # {"status":"bulk_fetch_done"} seen
        log_has_select_start = False      # {"status":"select_start"} seen
        script_bytes_done = None   # bytes of the bulk .sh fetched so far
        script_bytes_total = None  # expected size of the bulk .sh (HEAD; may be None)
        if log_path and Path(log_path).exists():
            try:
                import json as _json
                import re as _re2
                # Progress line format from download_tglc_batch.py:
                #   [1000/5000] done=1000 skip=0 fail=0 19.9/s eta=3.3min
                #   [finish] done=5000 skip=3 fail=2 in 4.2min
                # JSON phase markers from discovery_download.py:
                #   {"status": "bulk_fetch_start", "bytes_total": N|null, "url": ...}
                #   {"status": "bulk_fetch", "bytes_done": N, "bytes_total": N|null}
                #   {"status": "bulk_fetch_done", "bytes_done": N}
                #   {"status": "select_start"}
                #   {"status": "index_start"}
                #   {"status": "done", "downloaded": N, ...}
                #   {"status": "index_failed", "returncode": N, ...}
                _re_prefix  = _re2.compile(r"\[(\d+)/(\d+)\]")
                _re_done    = _re2.compile(r"\bdone=(\d+)")
                _re_skip    = _re2.compile(r"\bskip=(\d+)")
                _re_fail    = _re2.compile(r"\bfail=(\d+)")
                _re_rate    = _re2.compile(r"\b(\d+(?:\.\d+)?)/s\b")
                _re_eta     = _re2.compile(r"\beta=(\d+(?:\.\d+)?)min\b")
                with open(log_path, encoding="utf-8", errors="replace") as fh:
                    # Tail last 200 lines for efficiency on large logs
                    lines = fh.readlines()[-200:]
                for line in lines:
                    # Try JSON phase markers first (emitted by discovery_download.py).
                    line_stripped = line.strip()
                    if line_stripped.startswith("{"):
                        try:
                            obj = _json.loads(line_stripped)
                            st = obj.get("status")
                            if st == "index_start":
                                log_has_index_start = True
                            elif st == "upload_start":
                                log_has_upload_start = True
                            elif st == "done":
                                log_has_done_json = True
                            elif st == "index_failed":
                                log_has_index_failed = True
                            elif st == "bulk_fetch_start":
                                log_in_bulk_fetch_phase = True
                                bt = obj.get("bytes_total")
                                if bt is not None:
                                    script_bytes_total = bt
                            elif st == "bulk_fetch":
                                log_in_bulk_fetch_phase = True
                                bd = obj.get("bytes_done")
                                bt = obj.get("bytes_total")
                                if bd is not None:
                                    script_bytes_done = bd
                                if bt is not None:
                                    script_bytes_total = bt
                            elif st == "bulk_fetch_done":
                                log_has_bulk_fetch_done = True
                                bd = obj.get("bytes_done")
                                if bd is not None:
                                    script_bytes_done = bd
                            elif st == "select_start":
                                log_has_select_start = True
                            # Skip the rest of log parsing for JSON lines.
                            continue
                        except ValueError:
                            pass  # not valid JSON — fall through to text parsing
                    mp = _re_prefix.search(line)
                    if mp:
                        # progress line — update all values from this line
                        log_has_progress_line = True
                        total = int(mp.group(2))
                        md = _re_done.search(line)
                        ms = _re_skip.search(line)
                        mf = _re_fail.search(line)
                        mr = _re_rate.search(line)
                        me = _re_eta.search(line)
                        if md:
                            done = int(md.group(1))
                        if ms:
                            skipped = int(ms.group(1))
                        if mf:
                            failed = int(mf.group(1))
                        if mr:
                            rate = float(mr.group(1))
                        if me:
                            eta_min = float(me.group(1))
                    elif "[finish]" in line:
                        # final summary line — no N/M prefix, no rate/eta;
                        # keep rate/eta_min from last progress line seen above
                        log_has_finish = True
                        md = _re_done.search(line)
                        ms = _re_skip.search(line)
                        mf = _re_fail.search(line)
                        if md:
                            done = int(md.group(1))
                        if ms:
                            skipped = int(ms.group(1))
                        if mf:
                            failed = int(mf.group(1))
            except OSError:
                pass

        # Derive phase and returncode.
        #
        # Phase semantics (new field; existing fields unchanged for UI compat):
        #   "fetching_script" — pid alive, fetching the MAST bulk .sh (can be
        #                        multi-GB / tens of minutes on a big sector)
        #   "selecting"     — pid alive, scanning the bulk script for the next
        #                      not-yet-downloaded batch (minutes on ~8M lines)
        #   "downloading"   — pid alive, per-file download in progress
        #   "indexing"      — pid alive, index_start seen (or done==total heuristic)
        #   "done"          — pid dead, {"status":"done"} JSON seen, or [finish] compat
        #   "index_failed"  — pid dead, {"status":"index_failed"} seen
        #   "died"          — pid dead, no terminal marker (crashed mid-run)
        #
        # Backward compat: old logs without JSON markers use [finish] → "done".
        # The UI currently derives indexing via done>=total heuristic; the new
        # `phase` field makes this explicit and also surfaces "died" as an error
        # instead of silently showing "Stopped" with returncode=None.
        if running:
            # Priority order, latest phase first: uploading > indexing >
            # downloading (any per-file progress line seen) > selecting >
            # fetching_script > downloading (default — also covers old logs
            # with no JSON markers at all, preserving prior behaviour).
            if log_has_upload_start:
                phase = "uploading"
            elif log_has_index_start:
                phase = "indexing"
            elif log_has_progress_line:
                phase = "downloading"
            elif log_has_select_start:
                phase = "selecting"
            elif log_in_bulk_fetch_phase and not log_has_bulk_fetch_done:
                phase = "fetching_script"
            else:
                phase = "downloading"
        elif log_has_done_json:
            phase = "done"
        elif log_has_index_failed:
            phase = "index_failed"
        elif log_has_finish:
            # Backward compat: old log format without JSON markers.
            phase = "done"
        elif pid is not None:
            # Process is dead but no terminal marker was found.
            # NOTE: this is the "indexing forever" scenario from the incident —
            # the process emitted [finish] for the download but died before
            # emitting the final {"status":"done"} JSON (flush was missing).
            # With flush=True now in _emit(), this should not happen for new runs.
            phase = "died"
        else:
            phase = "done"  # no pid means never_run (already handled above)

        # Cross-process returncode derivation.
        # Fast-path (local proc) already has the real returncode from poll().
        if returncode is None:
            if phase == "done":
                returncode = 0
            elif phase == "index_failed":
                returncode = 1
            # "died" → returncode stays None (unknown)
            # "downloading"/"indexing" → None (still running)

        return {
            "running": running,
            "done": done,
            "failed": failed,
            "skipped": skipped,
            "total": total,
            "returncode": returncode,
            "rate": rate,
            "eta_min": eta_min,
            "phase": phase,
            "script_bytes_done": script_bytes_done,
            "script_bytes_total": script_bytes_total,
        }

    # ------------------------------------------------------------------ #
    # Auto-fill manager — deps, dispatch, endpoint, startup resume (Task 10)
    # ------------------------------------------------------------------ #

    def _make_autofill_deps(sector: int):
        """Construct a real AutofillDeps wired to local subprocess paths.

        All callables run synchronously in the background daemon thread.
        They do NOT touch the FastAPI event loop, so asyncio is not involved.

        The deps are intentionally constructed fresh per sector call so that
        the closures capture the correct ``sector``-specific values where needed
        (e.g. the scan ``--sector`` argument).
        """
        import time as _time_autofill
        from exonit.discovery_autofill import AutofillDeps
        from exonit.discovery_state import (
            load_registry,
            sector_gap_stats as _sector_gap_stats,
        )

        _tglc_idx = str(_PROJECT_ROOT / "data" / "metadata_cache.db")

        def _gap_stats(sec: int) -> dict:
            reg = load_registry(_DISCOVERY_REGISTRY)
            s_entry = reg.get("sectors", {}).get(str(sec), {})
            canonical = s_entry.get("canonical_list", "")
            return _sector_gap_stats(
                sec, db_path=db_path,
                canonical_csv=canonical, tglc_index=_tglc_idx,
            )

        def _scan(sec: int) -> dict:
            """Enqueue next lot via discovery_run.py; return {enqueued, lot_csv}."""
            # --allow-disabled: auto-fill is an explicit per-sector user action;
            # it must close gaps even in a sector whose registry entry is
            # disabled (e.g. a near-complete sector that left the sequential
            # campaign — exactly sector 1, which is enabled=false but still has
            # 4773 scattered holes). Without this the scan returns
            # sector_not_enabled -> enqueued=0 -> empty-lot guard fires.
            cmd = [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "discovery_run.py"),
                "--sector", str(sec),
                "--count", "100",
                "--db", db_path,
                "--registry", _DISCOVERY_REGISTRY,
                "--allow-disabled",
            ]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=600, cwd=str(_PROJECT_ROOT),
                )
            except subprocess.TimeoutExpired:
                logger.error("[autofill s%d] scan timed out", sec)
                return {"enqueued": 0, "lot_csv": ""}
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        data = json.loads(line)
                        if data.get("status") in ("started", "dry_run"):
                            return {
                                "enqueued": int(data.get("enqueued") or 0),
                                "lot_csv": data.get("lot_csv") or "",
                            }
                        return {"enqueued": 0, "lot_csv": ""}
                    except ValueError:
                        pass
            return {"enqueued": 0, "lot_csv": ""}

        def _launch(lot_csv: str, sec: int) -> None:
            """Spawn the discovery orchestrator for lot_csv."""
            if not lot_csv:
                return
            import tempfile as _tmpfile
            _log_fd, _log_path = _tmpfile.mkstemp(
                prefix="discovery_orch_autofill_", suffix=".log")
            os.close(_log_fd)
            # --chained: gap-fill lots run SCANNING + FPP + ExoMiner then stop at
            # DONE, skipping the end-of-campaign tail (STOP_WORKERS / LORA /
            # POSTPROCESS / FINALIZE). Chaining ~dozens of 100-TIC lots through
            # the full tail would stop the fleet and cold-start vLLM every lot.
            orch_cmd = [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "discovery_orchestrator.py"),
                "--lot-csv", lot_csv,
                "--sector", str(sec),
                "--server-url", f"http://localhost:{cfg.port}",
                "--state-file", _PIPELINE_STATE_FILE,
                "--chained",
                *(["--db", db_path] if db_path else []),
            ]
            try:
                with open(_log_path, "wb") as _lf:
                    proc = subprocess.Popen(
                        orch_cmd, stdout=_lf, stderr=subprocess.STDOUT,
                        cwd=str(_PROJECT_ROOT),
                    )
                _discovery_orchestrator["proc"] = proc
                _reap_in_background(proc)
                _discovery_orchestrator["started_at"] = _time_autofill.time()
            except Exception as exc:
                logger.error(
                    "[autofill s%d] could not spawn orchestrator: %s", sec, exc)

        def _poll() -> dict:
            """Block until pipeline_state.json reaches a *fresh* terminal state.

            Polls every 6 s; times out after 6 h (enough for any realistic
            sector run).  Terminal states are DONE, FAILED, and IDLE.

            Staleness guard (race fix).  ``_launch`` spawns the orchestrator
            subprocess and records ``started_at`` immediately afterwards.  The
            subprocess takes a few hundred ms (up to a couple of seconds under
            load) to boot and overwrite the state file with ``SCANNING``; until
            then the file still holds the *previous* lot's terminal state.  If
            we read that, the autofill loop sees a phantom ``DONE`` and either
            stops on the empty-lot guard or — worse — spawns a second
            concurrent orchestrator on the next tick.  So we ignore any
            terminal state whose file was last written *before* this lot was
            launched (``mtime < started_at``).  Both timestamps come from this
            host's clock for the local state file, so the comparison is
            skew-free (unlike the PG/<host> ``assigned_at`` skew).  A
            time-based ``sleep`` was rejected: it only works if startup is
            reliably faster than the guess, which is exactly the assumption
            that breaks under prod load.

            ``started_at`` is ``None`` only before the first successful launch
            (the dict initialises it to ``None``).  After any successful
            launch it holds the most-recent launch time, so a later Popen
            failure leaves a stale-but-non-None value — the guard then stays
            conservative (it just keeps polling) rather than accepting the old
            state.  When it is ``None`` we cannot judge freshness and fall back
            to accepting any terminal state, preserving the pre-guard
            behaviour.  Freshness logic lives in
            ``discovery_autofill.terminal_state_is_fresh`` so it can be unit
            tested.
            """
            from exonit.discovery_autofill import terminal_state_is_fresh
            from exonit.discovery_state import get_autofill as _get_af_poll

            _TERMINAL = {"DONE", "FAILED", "IDLE"}
            state_path = Path(_PIPELINE_STATE_FILE)
            launched_at = _discovery_orchestrator.get("started_at")
            for _ in range(3600):      # 3600 * 6 s = 6 h max
                # Responsive stop: bail out promptly if the user stopped the
                # auto-fill while we were polling (the orchestrator may be slow
                # or stuck, so we must not wait for its terminal state).
                try:
                    _af = _get_af_poll(sector, db_path=db_path)
                    if _af is not None and _af.get("status") == "stopped":
                        return {"state": "STOPPED"}
                except Exception:
                    pass
                if state_path.exists():
                    try:
                        st = json.loads(state_path.read_text())
                        if st.get("state") in _TERMINAL and terminal_state_is_fresh(
                            state_path.stat().st_mtime, launched_at
                        ):
                            return st
                    except Exception:
                        pass
                _time_autofill.sleep(6)
            return {"state": "FAILED", "detail": "poll timed out after 6 h"}

        def _download(sec: int, tic_ids: list) -> set:
            """Run discovery_download.py for the sector, return still-missing TICs.

            The download script downloads the next batch of FITS files from the
            MAST bulk script (not specific TIC IDs — the script doesn't support
            individual selection by TIC ID).  After it finishes, we query the
            TGLC index to find which of the requested tic_ids are now indexed.
            """
            cmd = [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "discovery_download.py"),
                "--sector", str(sec),
                "--count", "50000",
                "--registry", _DISCOVERY_REGISTRY,
            ]
            try:
                subprocess.run(
                    cmd, cwd=str(_PROJECT_ROOT),
                    timeout=7200,   # max 2 h for a large download batch
                )
            except subprocess.TimeoutExpired:
                logger.warning("[autofill s%d] download timed out", sec)
            # Check which of tic_ids are now indexed. Routed through
            # MetadataCache.tglc_tic_ids (exonit/metadata_backend.py's seam,
            # 2026-07-20 TGLC->PG migration F1) instead of a raw
            # sqlite3.connect — see the spec's item (e).
            indexed: set = set()
            try:
                from exonit.metadata_cache import MetadataCache
                mc = MetadataCache(_tglc_idx)
                try:
                    indexed = set(mc.tglc_tic_ids(sec))
                finally:
                    mc.close()
            except Exception as exc:
                logger.warning(
                    "[autofill s%d] could not query tglc index: %s", sec, exc)
            return {t for t in tic_ids if t not in indexed}

        def _on_complete(sec: int) -> None:
            """Q1 (perf review 2026-09-26): force a gap-stats refresh at the
            sector-DONE lot boundary instead of waiting out the (now 1 h)
            TTL — see ``AutofillDeps.on_complete``. S2 (2026-09-26 §5.2):
            scoped to *sec* only."""
            reg = load_registry(_DISCOVERY_REGISTRY)
            _force_gap_stats_refresh_now(reg.get("sectors", {}), _tglc_idx, sector=sec)

        return AutofillDeps(
            db_path=db_path,
            gap_stats=_gap_stats,
            scan=_scan,
            launch=_launch,
            poll=_poll,
            download=_download,
            on_complete=_on_complete,
        )

    def _dispatch_autofill_loop(sector: int) -> None:
        """Dispatch run_autofill in a daemon thread for *sector*.

        In-process guard (threading.Lock + dict): if a thread for this sector
        already exists and is alive, the call is a no-op.  Thread registration
        and advisory-lock acquisition happen atomically under
        *_autofill_threads_lock* so both guards are checked atomically within
        this process.

        Cross-process guard (PG only): calls the module-level
        ``_try_pg_advisory_lock(db_path, sector)``.  If another process
        already holds the lock, dispatch is skipped (clean return, no thread,
        no scan).  The lock is released when the loop thread exits by closing
        the dedicated connection.

        On SQLite (tests / single-process dev), ``_try_pg_advisory_lock``
        returns ``False`` immediately (no-op path) and the function proceeds
        normally.

        The daemon thread does NOT block the event loop and does NOT block
        server shutdown (daemon=True).
        """
        from exonit.discovery_autofill import run_autofill

        with _autofill_threads_lock:
            existing = _autofill_threads.get(sector)
            if existing is not None and existing.is_alive():
                logger.debug(
                    "[autofill s%d] thread already alive in this process; "
                    "skipping dispatch",
                    sector,
                )
                return

            # Cross-process guard: PG advisory lock, acquired while holding
            # _autofill_threads_lock so no two in-process calls can race past
            # this point simultaneously.
            # _try_pg_advisory_lock returns:
            #   a psycopg connection  — PG lock acquired (caller must close)
            #   None                  — PG lock NOT acquired (another process owns it)
            #   False                 — SQLite / non-PG, skip the lock entirely
            try:
                lock_result = _server._try_pg_advisory_lock(db_path, sector)
                if lock_result is None:
                    logger.info(
                        "[autofill s%d] PG advisory lock already held by "
                        "another process; skipping dispatch",
                        sector,
                    )
                    return
                lock_conn = lock_result if lock_result is not False else None
            except Exception as exc:
                # Connection/query error: log and proceed without the
                # cross-process guard so a transient PG hiccup never
                # silently kills the autofill loop.
                logger.warning(
                    "[autofill s%d] could not acquire PG advisory lock "
                    "(%s); proceeding without cross-process guard",
                    sector, exc,
                )
                lock_conn = None

            _lock_conn_final = lock_conn  # captured by _run_safe closure

            def _run_safe() -> None:
                try:
                    run_autofill(sector, deps=_make_autofill_deps(sector))
                except Exception as exc:
                    logger.error(
                        "[autofill s%d] run_autofill raised unexpectedly: %s",
                        sector, exc,
                    )
                finally:
                    # Close the dedicated connection to release the session-
                    # scoped advisory lock so the next restart can re-acquire.
                    if _lock_conn_final is not None:
                        try:
                            _lock_conn_final.close()
                        except Exception as e:
                            logger.warning(
                                "[autofill s%d] error releasing PG advisory "
                                "lock connection: %s",
                                sector, e,
                            )

            t = threading.Thread(
                target=_run_safe,
                name=f"autofill-s{sector}",
                daemon=True,
            )
            _autofill_threads[sector] = t
        t.start()
        logger.info("[autofill s%d] background loop dispatched", sector)

    @app.post("/api/discovery/sector/{n}/autofill")
    async def api_discovery_sector_autofill(n: int, request: Request):
        """Start or stop the durable auto-fill manager for sector *n*.

        Body: ``{"action": "start"|"stop"}``

        ``start`` (idempotent):
            Checks the ``discovery_autofill`` DB row.  If status is already
            ``'running'``, returns the current status without dispatching a
            second loop.  Otherwise, sets status to ``'running'`` and dispatches
            ``run_autofill`` in a background daemon thread.

            The single-instance guard is enforced via the DB row (not an
            in-memory flag) so it works correctly under A3 multi-process mode.
            An additional in-process thread-alive check prevents two threads
            from running in the SAME uvicorn worker.

        ``stop`` (cooperative):
            Writes ``status = 'stopped'`` to the DB.  The running loop reads
            the DB at each tick boundary and exits when it sees ``'stopped'``.

        Returns ``{"sector": int, "status": str, ...}``.

        Validation: returns 400 for an unrecognised *action* (never 500).
        """
        from exonit.discovery_state import get_autofill, set_autofill

        try:
            body = await request.json()
            action = body.get("action")
        except Exception as exc:
            return JSONResponse(
                {"error": "bad_request", "detail": str(exc)}, status_code=400
            )

        if action not in ("start", "stop"):
            return JSONResponse(
                {"error": "invalid_action",
                 "detail": (
                     f"action must be 'start' or 'stop', got {action!r}")},
                status_code=400,
            )

        # --- stop ---
        if action == "stop":
            try:
                set_autofill(n, db_path=db_path, status="stopped")
            except Exception as exc:
                logger.error("[autofill s%d] could not set stopped: %s", n, exc)
                return JSONResponse(
                    {"error": "db_error", "detail": str(exc)}, status_code=500
                )
            return {"sector": n, "status": "stopped"}

        # --- start ---
        # Cross-process guard: read DB row.
        try:
            af = get_autofill(n, db_path=db_path)
        except Exception:
            af = None

        # In-process guard: check if a thread is already alive in THIS process.
        with _autofill_threads_lock:
            existing_thread = _autofill_threads.get(n)
            thread_alive = existing_thread is not None and existing_thread.is_alive()

        db_running = af is not None and af.get("status") == "running"
        if db_running and thread_alive:
            # Fully idempotent: DB says running AND thread is alive here.
            return {
                "sector": n,
                "status": "running",
                "lots_done": int(af.get("lots_done") or 0),
                "current_lot": af.get("current_lot"),
                "idempotent": True,
            }
        if db_running and not thread_alive:
            # DB says running but no live thread in this process.  This can
            # happen when the previous run_autofill loop crashed before any DB
            # write: the PG advisory lock was released but the DB row kept
            # status='running', leaving the sector stuck.  Attempt to restart
            # via _dispatch_autofill_loop — which is a safe no-op if another
            # process genuinely holds the advisory lock (cross-process guard
            # prevents duplicate execution).  On SQLite / single-process the
            # lock is skipped and the loop restarts unconditionally.
            _dispatch_autofill_loop(n)
            return {
                "sector": n,
                "status": "running",
                "lots_done": int(af.get("lots_done") or 0),
                "current_lot": af.get("current_lot"),
                "idempotent": True,
            }

        # Not already running: persist 'running' in the DB first (so sibling
        # processes that poll the DB see the new status before the thread even
        # starts), then dispatch.
        try:
            set_autofill(
                n, db_path=db_path, status="running",
                lots_done=int(af.get("lots_done") or 0) if af else 0,
            )
        except Exception as exc:
            logger.error("[autofill s%d] could not set running: %s", n, exc)
            return JSONResponse(
                {"error": "db_error", "detail": str(exc)}, status_code=500
            )

        _dispatch_autofill_loop(n)

        try:
            af_after = get_autofill(n, db_path=db_path)
        except Exception:
            af_after = None

        return {
            "sector": n,
            "status": "running",
            "lots_done": int(af_after.get("lots_done") or 0) if af_after else 0,
            "current_lot": af_after.get("current_lot") if af_after else None,
            "idempotent": False,
        }

    # --- Startup resume: re-dispatch any autofill loops that were 'running'
    # before this server process started.  Runs in a daemon thread so it does
    # not delay the first request.  The loop itself reads the current DB state;
    # if the status was changed to 'stopped' or 'done' before the thread enters
    # run_autofill, the loop exits immediately.
    def _resume_autofill_on_startup() -> None:
        """Scan the sector registry and re-dispatch any 'running' autofill loops."""
        try:
            from exonit.discovery_state import get_autofill, load_registry
            reg = load_registry(_DISCOVERY_REGISTRY)
            for s_key in reg.get("sectors", {}):
                try:
                    sector_n = int(s_key)
                    af = get_autofill(sector_n, db_path=db_path)
                    if af is not None and af.get("status") == "running":
                        logger.info(
                            "[startup] resuming autofill for sector %d "
                            "(was status='running')",
                            sector_n,
                        )
                        _dispatch_autofill_loop(sector_n)
                except Exception as exc:
                    logger.warning(
                        "[startup] could not check/resume autofill for sector "
                        "%s: %s", s_key, exc,
                    )
        except Exception as exc:
            logger.warning("[startup] autofill resume scan failed: %s", exc)

    threading.Thread(
        target=_resume_autofill_on_startup,
        name="autofill-startup-resume",
        daemon=True,
    ).start()

    # --- Dead-run watchdog (Phase 3 Task 4) -------------------------------
    # Detects an orchestrator that died mid-run without ever reaching a
    # terminal state (SIGKILL/OOM/host reboot) and marks the discovery_runs
    # row FAILED so a human can `POST /api/discovery/resume` it. This is the
    # "died mid-flight" counterpart to the scan endpoint's own spawn
    # confirmation above, which only covers "never started in the first
    # place". Deliberately NEVER auto-resumes — the operator decides when (see
    # the operator rule: the operator launches the discovery runs); the resume is a
    # separate, explicit 1-click/1-command action.
    #
    # Registered the same way every other background task in this file is
    # (autofill-startup-resume above, the gap-stats cache-warm thread
    # earlier): a daemon threading.Thread with its own sleep loop, started
    # here at register()/create_app() time. This codebase has no
    # asyncio.create_task/lifespan-hook precedent to follow instead, and the
    # DB reads/writes involved (get_active_discovery_run/update_discovery_run)
    # are synchronous (sqlite3/psycopg) regardless, so a plain thread is the
    # right fit — it never blocks the single uvicorn event loop the request
    # handlers share.
    _WATCHDOG_INTERVAL_SECONDS = float(os.environ.get(
        "EXONIT_RUN_WATCHDOG_INTERVAL_SECONDS", "60"))
    _WATCHDOG_HEARTBEAT_STALE_SECONDS = float(os.environ.get(
        "EXONIT_RUN_HEARTBEAT_STALE_SECONDS", "180"))

    def _discovery_watchdog_check_once() -> None:
        """Single watchdog pass. Never raises (blanket try/except).

        Exposed as ``app.state.discovery_watchdog_check_once`` so tests can
        drive exactly one deterministic pass instead of racing a real 60 s
        sleep loop.
        """
        try:
            run = db.get_active_discovery_run()
            if run is None:
                return
            if run.get("dry_run"):
                # Dry runs are simulated pipelines (no real fleet/orchestrator
                # side effects) — never something to flag or auto-fail.
                return
            pid = run.get("orchestrator_pid")
            if not pid:
                # No pid recorded yet — either the row was just created (the
                # scan endpoint's own spawn confirmation covers that window)
                # or this is a pre-Task-2 row. Nothing to check.
                return
            # _pid_is_alive_matching (F4): pids are reaped promptly now and
            # can be recycled by an unrelated process — see its docstring.
            if _pid_is_alive_matching(pid, _ORCHESTRATOR_CMDLINE_MARKER):
                return

            # tz-aware epoch-seconds comparison — the SAME convention used
            # by the orphan-task auto-release elsewhere in this codebase
            # (exonit.db's _to_epoch: SQLite returns an ISO str,
            # PG/psycopg returns an already-aware datetime; _to_epoch
            # normalises both).
            now_epoch = datetime.now(timezone.utc).timestamp()
            hb_epoch = _to_epoch(run.get("heartbeat_at"))
            if hb_epoch is not None:
                stale = (now_epoch - hb_epoch) > _WATCHDOG_HEARTBEAT_STALE_SECONDS
            else:
                # No heartbeat ever recorded (died before the first one, or a
                # pre-Task-2 row) — fall back to updated_at so a run that just
                # started (and hasn't had a chance to heartbeat yet) isn't
                # flagged prematurely.
                updated_epoch = _to_epoch(run.get("updated_at"))
                stale = (updated_epoch is not None and
                         (now_epoch - updated_epoch) > _WATCHDOG_HEARTBEAT_STALE_SECONDS)
            if not stale:
                return

            detail = f"orchestrator_died (state={run['state']})"
            logger.warning(
                "[watchdog] run_id=%s: orchestrator pid=%s is dead and "
                "heartbeat/updated_at is stale (>%.0fs) — marking FAILED",
                run["run_id"], pid, _WATCHDOG_HEARTBEAT_STALE_SECONDS)
            # F6 (final-review fix): finished=True stamps finished_at — this
            # IS a terminal transition, same as every other terminal write.
            db.update_discovery_run(run["run_id"], state="FAILED", detail=detail,
                                    finished=True)
            _discovery_set_pipeline_state("FAILED", detail=detail)
        except Exception as exc:
            logger.warning(
                "[watchdog] discovery run check failed (non-fatal): %s", exc)

    def _discovery_watchdog_loop() -> None:
        while True:
            _time.sleep(_WATCHDOG_INTERVAL_SECONDS)
            _discovery_watchdog_check_once()

    app.state.discovery_watchdog_check_once = _discovery_watchdog_check_once
    threading.Thread(
        target=_discovery_watchdog_loop,
        name="discovery-run-watchdog",
        daemon=True,
    ).start()

    @app.post("/api/workers/start")
    async def api_workers_start():
        """Start the campaign workers that are not currently online.

        Workers targeted: every entry in ops/workers.json, read live via
        _server._campaign_worker_names() (so adding a worker there is picked up
        with no code edit and no server restart). A worker that is offline when
        the button is pressed just reports 'failed' rather than being skipped.

        Per-worker: checks workers_state for recent heartbeat; if online already,
        reports 'already_online'. Otherwise, invokes bin/farmctl start <name>.

        Returns list of per-worker action dicts.
        """
        now = _time.time()
        results = []
        for name in _server._campaign_worker_names():
            # Determine if all expected instances are currently online.
            # For multi-instance workers (instances > 1), farmctl names each
            # process <name>-1, <name>-2, … so we match both the base name
            # and the suffixed pattern.
            expected = _server._expected_instances(name)
            online_count = _count_online_instances(name, workers_state.values(), now)
            if online_count >= expected:
                results.append({
                    "worker": name,
                    "action": "already_online",
                    "detail": f"{online_count}/{expected} online",
                })
                continue
            try:
                rc, stdout, stderr = _farmctl_start(name)
                if rc == 0:
                    results.append({
                        "worker": name, "action": "started",
                        "detail": f"{online_count}/{expected} were online; {stdout.strip()}",
                    })
                else:
                    results.append({
                        "worker": name, "action": "failed",
                        "detail": (stderr.strip() or stdout.strip())[:300],
                    })
            except Exception as exc:
                results.append({"worker": name, "action": "failed", "detail": str(exc)})

        return results

