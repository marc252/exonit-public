#!/usr/bin/env python3
"""Prepare and enqueue the next discovery lot for a given sector.

This is the script the UI "Start Scan" button (discovery path) will call.
It computes the next block of TICs in canonical order, validates that the
TGLC data is downloaded, writes a lot CSV, enqueues to the server, and
advances the registry cursor.

Exit codes
----------
0  started OK (or dry-run preview)
2  sector not enabled / canonical list problem
3  need more TGLC downloads before scanning
4  enqueue_from_csv.py returned non-zero
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Guarantee this script imports ITS OWN sibling `exonit` package rather than
# whatever `exonit` happens to resolve first on sys.path (e.g. a stale
# editable install from a different checkout/worktree) — same convention as
# scripts/build_deep_pass_worklist.py and others.
sys.path.insert(0, str(ROOT))

from exonit.discovery_state import (
    load_registry,
    get_sector,
    set_cursor,
    set_deep_cursor,
    derive_done_set,
    derive_run_id,
    next_block,
    load_extended_canonical,
    active_sector,
)
from exonit.config import config as _config
from exonit.db_backend import is_pg_url
from exonit.progress_client import ProgressReporter


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _emit(obj: dict) -> None:
    """Print JSON and flush."""
    print(json.dumps(obj))
    sys.stdout.flush()


def _load_canonical(csv_path: str) -> list:
    """Read tic_id,label CSV and return ordered list of int TIC ids."""
    tics = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tic = row.get("tic_id", "").strip()
            if tic and tic != "nan":
                try:
                    tics.append(int(tic))
                except ValueError:
                    pass
    return tics


def _load_downloaded_set(tglc_index_path: str, sector: int | None = None) -> set:
    """Return the set of int TIC ids present in the TGLC index (sector-scoped).

    Phase 2 fix (2026-08-27): routed through MetadataCache — the
    metadata_backend seam — exactly like ``load_extended_canonical`` (spec
    2026-07-20-tglc-index-to-pg, D3 item e). The old raw ``sqlite3.connect``
    read the FROZEN ``data/metadata_cache.db``, which holds nothing indexed
    after the 2026-07-21 PG migration: a survey of such a sector (s18, s26,
    ...) built its work list from PG but its downloaded_set from the frozen
    file → ``ready=0`` → a spurious ``need_download`` with the sector fully
    downloaded and indexed. s15/s17 never hit this only because their rows
    predate the freeze.
    """
    try:
        from exonit.metadata_cache import MetadataCache
        mc = MetadataCache(tglc_index_path)
        try:
            indexed = mc.tglc_tic_ids(sector)
        finally:
            mc.close()
    except (sqlite3.OperationalError, OSError):
        # index missing/unreadable — same contract as before: empty set
        return set()
    result = set()
    for t in indexed:
        try:
            result.add(int(t))
        except (ValueError, TypeError):
            pass
    return result


def _auto_lot_path(sector: int, ready_tics: list) -> str:
    """Generate an automatic lot CSV filename."""
    # We can't know the lot number cheaply; use timestamp for uniqueness
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"data/discovery/s{sector:04d}_lot_{ts}.csv"


def _write_lot_csv(path: str, tic_ids: list) -> None:
    """Write a tic_id,label CSV for the given TIC ids."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tic_id", "label"])
        for t in tic_ids:
            writer.writerow([t, "candidate"])


def _load_toi_tic_set(catalog_path: str) -> set:
    """Return the set of int TIC ids present in the TOI catalog.

    Returns an empty set (with a warning) if the file is missing, so the
    caller can proceed without filtering rather than crashing.
    """
    import warnings as _warnings
    result: set = set()
    try:
        with open(catalog_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raw = row.get("TIC ID", "").strip()
                if not raw:
                    continue
                try:
                    result.add(int(float(raw)))
                except (ValueError, TypeError):
                    pass
    except FileNotFoundError:
        _warnings.warn(
            f"[discovery_run] TOI catalog not found at {catalog_path!r}; "
            "TOI filter disabled for this lot.",
            RuntimeWarning,
            stacklevel=2,
        )
    return result


def _lot_adopt_active() -> bool:
    """EXONIT_LOT_ADOPT_ACTIVE (S4, docs/superpowers/specs/2026-09-26-
    enqueue-s1-s4-design.md §2.2; default 0). With 0, this script IGNORES
    the ``adopted`` count ``/api/worker/enqueue_bulk`` always computes and
    returns (harmless): the lot's effective size stays ``tasks_created``
    only, byte-for-byte pre-S4 behaviour. With 1, ``adopted`` TICs (tasks
    already created by an earlier, dead/timed-out attempt at this same lot,
    now re-tagged to this run_id) count toward the lot's effective size too.
    """
    return os.environ.get("EXONIT_LOT_ADOPT_ACTIVE", "0") == "1"


def _parse_enqueue_summary(stdout: str) -> dict | None:
    """Extract the ``ENQUEUE_SUMMARY_JSON: {...}`` line emitted by
    scripts/enqueue_from_csv.py (last matching line wins, in case anything
    else was printed to stdout).  Returns None if the line is absent
    (backward-compat with an older enqueue_from_csv.py that doesn't emit
    it yet) or malformed — callers must fall back to the pre-existing
    len(ready)/len(selected) behaviour in that case, never raise.
    """
    marker = "ENQUEUE_SUMMARY_JSON: "
    summary = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(marker):
            try:
                summary = json.loads(line[len(marker):])
            except (json.JSONDecodeError, ValueError):
                continue
    return summary


def _append_campaign_log(
    sector: int,
    lot_csv: str,
    first_row: int,
    last_row: int,
    tag: str = "lot-auto",
) -> None:
    """Append one history line to data/discovery/CAMPAIGN.md.

    Parameters
    ----------
    first_row, last_row:
        1-based row indices of the first and last selected TIC, matching
        the CAMPAIGN.md cursor semantics (row index into the canonical CSV
        for normal lots, or into the deep-rescan worklist CSV for deep
        lots — NOT a simple count).
    tag:
        Row label in CAMPAIGN.md — ``"lot-auto"`` for a normal campaign
        lot, ``"lot-deep"`` for a deep-rescan lot (spec 2026-07-23 §4).
    """
    campaign_md = ROOT / "data" / "discovery" / "CAMPAIGN.md"
    if not campaign_md.exists():
        return
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    count = last_row - first_row + 1
    line = (
        f"\n| {tag} | {first_row}–{last_row} ({count} TICs) "
        f"| `{Path(lot_csv).name}` | ENQUEUED {ts} |"
    )
    with open(campaign_md, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _finish_survey_enqueue(
    plan: dict, ready: list, lot_csv: str, sector: int, run_id: str,
    db_path: str, args, rep, remaining_undone, advance_cursor_fn,
) -> int:
    """Shared tail of the survey (non-deep) enqueue path (steps 6-9):
    write the lot CSV, run enqueue_from_csv.py, parse its summary, compute
    the S4 effective lot size + "no TIC lost" rule, and — only on
    success — call *advance_cursor_fn* (which persists whatever cursor
    fields this lot's selector uses) before appending to CAMPAIGN.md and
    emitting the final plan.

    Shared between the legacy (canonical/next_block-driven) selector and
    the S1 paged selector (:mod:`exonit.lot_selector`) — the only
    difference between the two call sites is HOW ``ready``/``plan``'s
    cursor-related fields were computed before this point, not what
    happens after enqueuing. Returns the process exit code (0 on success).
    """
    _write_lot_csv(lot_csv, ready)
    # Reuse the same reporter created at the top of the survey path (Q9)
    # rather than a fresh one — total was unknown until now, so set it
    # directly before switching substeps (report()'s own total= kwarg only
    # overrides a single call, not rep.finish()'s later one).
    rep.total = len(ready)
    rep.report(0, substep="enqueue_from_csv",
              message=f"enqueuing {len(ready)} TICs")
    enqueue_proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "enqueue_from_csv.py"),
         "--csv", lot_csv, "--db", db_path,
         "--sector", str(sector),
         "--run-id", run_id,
         # Pass this deployment's token so enqueue auth matches the server
         # (home and AWS use different tokens — avoids 401 on AWS).
         "--token", os.environ.get("EXONIT_WORKER_TOKEN", "exonit-default-token")],
        capture_output=True, text=True,
    )
    if enqueue_proc.returncode != 0:
        _emit({
            "status": "enqueue_failed",
            "run_id": run_id,
            "detail": (
                f"enqueue_from_csv.py exited {enqueue_proc.returncode}:\n"
                f"{enqueue_proc.stderr.strip()}"
            ),
        })
        return 4
    rep.finish(message="enqueued")

    # Report the tasks ACTUALLY created, not len(ready): enqueue_targets
    # (called by enqueue_from_csv.py) additionally skips TICs already present
    # in `stars` (analyzed_set) or without metadata (no_fits) — filters that
    # derive_done_set/lot_selector above does not apply. Parse
    # enqueue_from_csv.py's own summary so `enqueued` reflects real task
    # rows, not the pre-filter selection count (root cause of the empty-lot
    # loop, 2026-08).
    #
    # S4 (docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md §2.1):
    # "enqueued" is the LOT's EFFECTIVE size — tasks_created PLUS adopted
    # (TICs whose task was already created by an earlier, dead/timed-out
    # attempt at this same lot and re-tagged to this run_id) — gated by
    # EXONIT_LOT_ADOPT_ACTIVE so a flag-off deployment stays byte-for-byte
    # pre-S4 (effective == tasks_created; "adopted" is still reported but
    # ignored — see _lot_adopt_active's docstring).
    enqueue_summary = _parse_enqueue_summary(enqueue_proc.stdout)
    created = len(ready)
    adopted = 0
    enq_failed = 0
    if enqueue_summary is not None:
        created = enqueue_summary.get("enqueued", len(ready))
        adopted = enqueue_summary.get("adopted", 0)
        enq_failed = enqueue_summary.get("failed", 0)
        plan["enqueue_skipped"] = enqueue_summary.get("skipped")
        plan["enqueue_failed"] = enq_failed
        # tasks_created/adopted are new (S4) fields — like enqueue_skipped/
        # enqueue_failed above, only added when a real summary line was
        # parsed (backward compat: an older enqueue_from_csv.py with no
        # ENQUEUE_SUMMARY_JSON line gets the pre-S4 plan shape exactly).
        plan["tasks_created"] = created
        plan["adopted"] = adopted
    effective = created + (adopted if _lot_adopt_active() else 0)
    plan["enqueued"] = effective

    if enq_failed > 0:
        # "No TIC lost" rule (S4 §2.1, operator decision §9.3): a chunk
        # lost despite Q5's retries must never silently advance the cursor
        # past it — a re-run of this SAME lot (cursor untouched) adopts the
        # tasks already created (S4) and retries the rest.
        _emit({
            "status": "enqueue_failed",
            "reason": "enqueue_chunk_failed",
            "sector": sector,
            "run_id": run_id,
            "tasks_created": created,
            "adopted": adopted,
            "enqueue_failed": enq_failed,
            "detail": (
                f"{enq_failed} TIC(s) failed to enqueue after retries — "
                "cursor left untouched so a re-run can adopt the tasks "
                "already created (S4) and retry the rest"
            ),
        })
        return 4

    real_enqueued = effective

    if real_enqueued == 0:
        # Same reason/semantics as the "not selected" no_undone_tics case
        # above: the plan is genuinely exhausted (every selected TIC was
        # already analyzed or has no FITS) — the cursor must NOT advance for
        # a lot that never actually existed as task rows.
        _emit({
            "status": "error",
            "reason": "no_undone_tics",
            "requested": args.count,
            "run_id": run_id,
            "selected": len(ready),
            "adopted": adopted,
            "remaining_undone": remaining_undone,
            "enqueue_skipped": plan.get("enqueue_skipped"),
            "enqueue_failed": plan.get("enqueue_failed"),
            "detail": (
                "all selected TICs are already analysed (stars) "
                "or have no FITS; no task created"
            ),
        })
        return 2

    # 8. Advance cursor in registry (selector-specific — legacy sets just
    # `cursor`; the S1 paged selector also persists `cursor_seq`).
    advance_cursor_fn()

    plan["status"] = "started"
    plan["enqueue_stdout"] = enqueue_proc.stdout.strip()
    _emit(plan)
    return 0


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Prepare and enqueue the next discovery lot."
    )
    p.add_argument("--sector", type=int, default=None,
                   help="Sector number (default: registry active_sector)")
    p.add_argument("--count", type=int, required=True,
                   help="Number of TICs to enqueue in this lot")
    p.add_argument("--db", default=None,
                   help="ExoNit operational DB: a SQLite path or a "
                        "postgresql:// DSN (default: config.db_path, which "
                        "honours EXONIT_DATABASE_URL)")
    p.add_argument("--registry", default="data/discovery/sectors.json",
                   help="Sector registry JSON path")
    p.add_argument("--canonical", default=None,
                   help="Canonical ordered TIC list CSV (default: from registry)")
    p.add_argument("--tglc-index", default="data/metadata_cache.db",
                   help="TGLC index SQLite DB path")
    p.add_argument("--out", default=None,
                   help="Lot CSV output path (default: auto)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan JSON without enqueuing or updating registry")
    p.add_argument("--allow-disabled", action="store_true",
                   help="Scan even if the sector's registry entry has "
                        "enabled=false. Used by the UI auto-fill to close gaps "
                        "in a sector that has left the sequential campaign "
                        "(e.g. a near-complete sector with scattered holes). "
                        "The normal CLI/server scan still respects enabled.")
    p.add_argument("--toi-catalog", default=None,
                   help="Path to TOI catalog CSV (default: data/toi_catalog.csv relative "
                        "to repo root). TICs in this catalog are skipped from the work list "
                        "(the cursor still advances past them normally).")
    p.add_argument("--tics-csv", default=None,
                   help="Deep-rescan worklist CSV (tic_id column, pre-cleaned by "
                        "scripts/build_deep_pass_worklist.py). Enables deep-rescan mode: "
                        "slices by the sector's own deep_cursor (independent of the "
                        "campaign cursor). Requires --force-rescan.")
    p.add_argument("--force-rescan", action="store_true",
                   help="Deep-rescan mode: force re-enqueue of already-completed TICs "
                        "via enqueue_from_csv.py --force-rescan. Requires --tics-csv.")
    p.add_argument("--full-scan", action="store_true",
                   help="S1 (docs/superpowers/specs/2026-09-26-enqueue-s1-s4-"
                        "design.md §4.1.8): force the legacy O(sector) "
                        "selector even when EXONIT_SELECTOR_PAGED=1 -- the "
                        "exact-recall escape hatch for a near-closed sector "
                        "(used by verify_sector_complete.py and the "
                        "autofill manager). No effect on a sector with a "
                        "canonical_list (already always legacy) or when the "
                        "paged selector was never going to run anyway.")
    p.add_argument("--paged", action="store_true",
                   help="S1: force the paged selector for THIS invocation "
                        "even if EXONIT_SELECTOR_PAGED=0 (for A/B testing "
                        "against the legacy selector — see --dump-selection). "
                        "Still requires EXONIT_DONE_STATE=1 and no "
                        "canonical_list, same as the env-var gate.")
    p.add_argument("--dump-selection", default=None,
                   help="S1: with --dry-run, write the FULL selected TIC "
                        "list (not just the 10-item preview in the JSON "
                        "plan) to this CSV path -- for comparing the paged "
                        "and legacy selectors' output on the same inputs.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # --tics-csv and --force-rescan are mutually required (deep-rescan mode
    # needs both; neither makes sense alone — spec 2026-07-23 §4).
    if bool(args.tics_csv) != bool(args.force_rescan):
        _emit({"status": "error", "reason": "deep_mode_flags"})
        return 2

    # No hardcoded default: home runs sector 1, AWS sector 2 — the registry
    # (or EXONIT_SECTOR) decides, same as the server does.
    sector = args.sector if args.sector is not None else active_sector(args.registry)

    # 1. Load registry and validate sector
    registry = load_registry(args.registry)
    sector_entry = get_sector(registry, sector)

    if sector_entry is None:
        _emit({
            "status": "error",
            "reason": "sector_not_found",
            "sector": sector,
        })
        return 2

    if not sector_entry.get("enabled", False) and not args.allow_disabled:
        _emit({
            "status": "error",
            "reason": "sector_not_enabled",
            "sector": sector,
        })
        return 2

    # Resolve the operational DB now (needed by both branches below): --db
    # wins, else config.db_path (which honours EXONIT_DATABASE_URL → a
    # postgresql:// DSN). Moved up from the old step 3 so the deep-rescan
    # branch — which returns before the canonical/TGLC/done-set steps —
    # has it available too.
    db_path = args.db or _config.db_path

    if args.tics_csv:
        # --- DEEP-RESCAN MODE (spec 2026-07-23 §4) ---
        # Its own cursor, its own lot naming, campaign cursor untouched.
        # The worklist arrives pre-cleaned by build_deep_pass_worklist.py:
        # done_set/TOI/downloaded checks are NOT applied here.
        worklist = _load_canonical(args.tics_csv)   # same csv reader: tic_id col
        if not worklist:
            _emit({"status": "error", "reason": "deep_worklist_empty"})
            return 2
        deep_cursor = sector_entry.get("deep_cursor", 0)
        selected = worklist[deep_cursor:deep_cursor + args.count]
        if not selected:
            _emit({"status": "error", "reason": "deep_worklist_exhausted",
                   "deep_cursor": deep_cursor, "worklist_size": len(worklist)})
            return 2
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        lot_csv = args.out or str(
            ROOT / "data" / "discovery" / f"s{sector:04d}_deep_lot_{ts}.csv")
        new_deep_cursor = deep_cursor + len(selected)
        # Mint run_id ONCE, here — the single source of truth (S3, see
        # exonit.discovery_state.derive_run_id's docstring) — and pass it
        # through to enqueue_from_csv.py so tasks.run_id is set at INSERT
        # time. The scan endpoint (exonit/routes/discovery.py) prefers
        # this minted value over its own (legacy) re-derivation once it
        # sees it in this plan's JSON output ("the child wins").
        run_id = derive_run_id(sector, lot_csv)
        plan = {
            "status": "dry_run" if args.dry_run else "started",
            "mode": "deep_rescan",
            "sector": sector,
            "requested": args.count,
            "enqueued": len(selected),
            "lot_csv": lot_csv,
            "run_id": run_id,
            "deep_cursor_before": deep_cursor,
            "deep_cursor_after": new_deep_cursor,
            "remaining_deep": len(worklist) - new_deep_cursor,
            "tics_selected": selected[:10],
        }
        if args.dry_run:
            _emit(plan)
            return 0
        _write_lot_csv(lot_csv, selected)
        rep = ProgressReporter(unit="TICs", total=len(selected),
                               substep="enqueue_from_csv", interval_s=15)
        rep.report(0, message=f"enqueuing {len(selected)} TICs")
        enqueue_proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "enqueue_from_csv.py"),
             "--csv", lot_csv, "--db", db_path, "--force-rescan",
             "--sector", str(sector),
             "--run-id", run_id,
             "--token", os.environ.get("EXONIT_WORKER_TOKEN",
                                       "exonit-default-token")],
            capture_output=True, text=True,
        )
        if enqueue_proc.returncode != 0:
            _emit({"status": "enqueue_failed", "run_id": run_id,
                   "detail": enqueue_proc.stderr.strip()})
            return 4
        rep.finish(message="enqueued")

        # Report the tasks ACTUALLY created, not len(selected): enqueue_targets
        # skips TICs already present in `stars` (analyzed_set, bypassed only in
        # the "already analyzed" step 1 above, not steps 2-3) or without
        # metadata. Parse enqueue_from_csv.py's own summary so `enqueued` here
        # reflects real task rows, not the pre-filter selection count.
        #
        # S4 (docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md §2.1):
        # "enqueued" here is the LOT's EFFECTIVE size — tasks_created PLUS
        # adopted (TICs whose task was already created by an earlier,
        # dead/timed-out attempt at this same lot and re-tagged to this
        # run_id) — gated by EXONIT_LOT_ADOPT_ACTIVE so a flag-off deployment
        # stays byte-for-byte pre-S4 (effective == tasks_created, "adopted"
        # is still reported but ignored).
        enqueue_summary = _parse_enqueue_summary(enqueue_proc.stdout)
        created = len(selected)
        adopted = 0
        enq_failed = 0
        if enqueue_summary is not None:
            created = enqueue_summary.get("enqueued", len(selected))
            adopted = enqueue_summary.get("adopted", 0)
            enq_failed = enqueue_summary.get("failed", 0)
            plan["enqueue_skipped"] = enqueue_summary.get("skipped")
            plan["enqueue_failed"] = enq_failed
            # tasks_created/adopted are new (S4) fields — like
            # enqueue_skipped/enqueue_failed above, only added when a real
            # summary line was parsed (backward compat: an older
            # enqueue_from_csv.py with no ENQUEUE_SUMMARY_JSON line gets the
            # pre-S4 plan shape exactly).
            plan["tasks_created"] = created
            plan["adopted"] = adopted
        effective = created + (adopted if _lot_adopt_active() else 0)
        plan["enqueued"] = effective

        if enq_failed > 0:
            # "No TIC lost" rule (S4 §2.1): a chunk lost despite Q5's
            # retries must never silently advance the cursor past it — a
            # re-run of this SAME plan (deep_cursor untouched) adopts the
            # tasks already created (S4) and retries the rest.
            _emit({
                "status": "enqueue_failed",
                "reason": "enqueue_chunk_failed",
                "mode": "deep_rescan",
                "sector": sector,
                "run_id": run_id,
                "tasks_created": created,
                "adopted": adopted,
                "enqueue_failed": enq_failed,
                "detail": (
                    f"{enq_failed} TIC(s) failed to enqueue after retries — "
                    "deep_cursor left untouched so a re-run can adopt the "
                    "tasks already created (S4) and retry the rest"
                ),
            })
            return 4

        real_enqueued = effective

        if real_enqueued == 0:
            # Different reason than the survey branch's no_undone_tics on
            # purpose: build_deep_pass_worklist.py pre-cleans the worklist,
            # so 0 real creates here is an anomaly (not a naturally exhausted
            # plan) — the supervisor must pause+alert, not silently advance
            # as if the deep-rescan plan were complete.
            _emit({
                "status": "error",
                "reason": "no_tasks_created",
                "mode": "deep_rescan",
                "sector": sector,
                "run_id": run_id,
                "requested": args.count,
                "selected": len(selected),
                "adopted": adopted,
                "enqueue_skipped": plan.get("enqueue_skipped"),
                "enqueue_failed": plan.get("enqueue_failed"),
                "detail": (
                    "all selected TICs were skipped by enqueue_from_csv.py "
                    "(already in stars or no metadata) — no task rows "
                    "created; deep_cursor left untouched"
                ),
            })
            return 2

        set_deep_cursor(args.registry, sector, new_deep_cursor)
        _append_campaign_log(sector, lot_csv, deep_cursor + 1,
                             new_deep_cursor, tag="lot-deep")
        plan["enqueue_stdout"] = enqueue_proc.stdout.strip()
        _emit(plan)
        return 0

    # Q9 (perf review 2026-09-26, docs/reviews/2026-09-26-enqueue-path-
    # performance-review.md): a single ProgressReporter for the whole survey
    # path, reused across substeps. Before this, the UI/orchestrator had no
    # progress mark at all until step 7 (enqueue_from_csv) — the ~90s spent
    # in steps 2-3 (build_work_list: load_extended_canonical over the
    # canonical CSV + TGLC index; done_set: derive_done_set's completed-set
    # scan/intersect) showed as a flat 0, indistinguishable from a hung
    # process. total is unknown until the work list is built, so it starts
    # None; set once known (step 7, unchanged from before) before switching
    # to the enqueue_from_csv substep. No-op (ProgressReporter.enabled is
    # False) unless this process was actually launched as part of a run —
    # see that class's docstring.
    rep = ProgressReporter(unit="TICs", total=None,
                           substep="build_work_list", interval_s=15)
    rep.report(0, message="building work list (canonical CSV + TGLC index)")

    canonical_csv = args.canonical or sector_entry.get("canonical_list")

    # S1 (docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md §4):
    # the O(lot) paged selector replaces steps 2-4 below (load_extended_
    # canonical + derive_done_set + next_block, an O(sector) pipeline) for
    # the case it actually helps: NO canonical_list (a frozen CSV means a
    # SMALL, already-bounded work list — the legacy path stays exact and
    # cheap there) and not --full-scan (the exact-recall escape hatch for a
    # near-closed sector, §4.1.8). Requires EXONIT_DONE_STATE=1 (this
    # selector reads tic_sector_state directly, no legacy fallback of its
    # own) — a flag mismatch degrades to the legacy path with a warning,
    # never a crash.
    _use_paged = (
        not args.full_scan
        and (args.paged or os.environ.get("EXONIT_SELECTOR_PAGED", "0") == "1")
        and not canonical_csv
    )
    if _use_paged:
        from exonit import done_state as _done_state_mod
        if not _done_state_mod.is_enabled():
            print(
                "WARNING: EXONIT_SELECTOR_PAGED requested but "
                "EXONIT_DONE_STATE=0 -- falling back to the legacy "
                "selector (paged mode has no done-state-independent path)",
                file=sys.stderr,
            )
            _use_paged = False

    if _use_paged:
        from exonit.discovery_state import get_cursor_seq_if_valid, set_cursor_seq
        from exonit.lot_selector import select_lot_paged

        toi_catalog_path = (
            args.toi_catalog if args.toi_catalog
            else str(ROOT / "data" / "toi_catalog.csv")
        )
        toi_tic_set = _load_toi_tic_set(toi_catalog_path)

        old_cursor = sector_entry.get("cursor", 0)
        cursor_seq_in = get_cursor_seq_if_valid(sector_entry)
        rep.report(0, substep="select_lot_paged",
                  message="selecting next lot (O(lot) paged selector)")
        block = select_lot_paged(
            sector, args.count,
            db_path=db_path,
            tglc_index_path=args.tglc_index,
            cursor=old_cursor,
            cursor_seq=cursor_seq_in,
            toi_tic_set=toi_tic_set,
        )
        selected = block["selected"]
        ready = block["ready"]

        if not selected:
            _emit({
                "status": "error",
                "reason": "no_undone_tics",
                "remaining_undone": block["remaining_undone"],
            })
            return 2

        lot_csv = args.out or _auto_lot_path(sector, ready)
        run_id = derive_run_id(sector, lot_csv)
        new_cursor = block["new_cursor"]
        new_cursor_seq = block["new_cursor_seq"]

        plan = {
            "status": "dry_run" if args.dry_run else "started",
            "sector": sector,
            "requested": args.count,
            "enqueued": len(ready),
            "lot_csv": lot_csv,
            "run_id": run_id,
            "cursor_before": old_cursor,
            "cursor_after": new_cursor,
            "remaining_undone": block["remaining_undone"],
            "remaining_undone_is_upper_bound": True,
            "tics_selected": selected[:10],
            "tois_skipped": block["tois_skipped"],
            "retry_selected": block["retry_selected"],
            "selector": "paged",
        }

        if args.dry_run:
            if args.dump_selection:
                _write_lot_csv(args.dump_selection, selected)
            _emit(plan)
            return 0

        def _advance_paged_cursor():
            set_cursor(args.registry, sector, new_cursor)
            set_cursor_seq(args.registry, sector, new_cursor_seq, new_cursor)
            tag = "lot-auto"
            if block["retry_selected"] > 0:
                tag += f" (+{block['retry_selected']} retry)"
            _append_campaign_log(sector, lot_csv, old_cursor + 1, new_cursor,
                                 tag=tag)

        return _finish_survey_enqueue(
            plan, ready, lot_csv, sector, run_id, db_path, args, rep,
            block["remaining_undone"], _advance_paged_cursor)

    # --- legacy selector (O(sector)): load_extended_canonical + derive_
    # done_set + next_block. Byte-for-byte unchanged from before S1. ---

    # 2. Build the extended work list: canonical CSV rows first, then any
    #    TICs that are in the TGLC index but not yet in the canonical CSV
    #    (in rowid ASC order — the order they were indexed).

    # Restrict the TGLC extension to this run's sector: the index interleaves
    # sectors by rowid, so an unfiltered list would feed undone sector-1 TICs
    # into a sector-2 run before reaching any sector-2 target.
    # Q2 (perf review 2026-09-26): return_indexed_set=True reuses the SAME
    # tglc_tic_ids() query this call already makes for the extension list as
    # the downloaded_set input below, instead of a second, separate
    # `_load_downloaded_set` dump of the whole sector's tglc rows a few
    # lines down (measured ~2.1 s PG + a second multi-million-row Python set
    # per lot). `indexed_set` is byte-for-byte what
    # `_load_downloaded_set(args.tglc_index, sector)` would have returned.
    canonical_tics, indexed_set = load_extended_canonical(
        canonical_csv_path=str(canonical_csv) if canonical_csv else "",
        tglc_index_path=args.tglc_index,
        sector=sector,
        return_indexed_set=True,
    )

    # Log the breakdown for visibility.
    try:
        import csv as _csv
        frozen_count = 0
        frozen_path = canonical_csv if canonical_csv else ""
        if frozen_path and Path(frozen_path).exists():
            with open(frozen_path, newline="", encoding="utf-8") as fh:
                frozen_count = sum(
                    1 for row in _csv.DictReader(fh)
                    if row.get("tic_id", "").strip() not in ("", "nan")
                )
        ext_count = len(canonical_tics) - frozen_count
        print(
            f"INFO: work list = {frozen_count} canonical + {ext_count} extended "
            f"(from tglc rowid ASC) = {len(canonical_tics)} total",
            file=sys.stderr,
        )
    except Exception:
        pass

    if not canonical_tics:
        _emit({"status": "error", "reason": "canonical_list_empty"})
        return 2

    # 3. Build downloaded_set (TGLC index) and done_set (tasks table).
    #    sector=sector lets derive_done_set fold in exhausted(t, sector) when
    #    EXONIT_DONE_STATE=1 (done-state design, 2026-09-26) — a TIC that has
    #    failed twice for THIS sector is excluded from selection the same way
    #    a completed TIC is. With the flag off (default), sector is unused
    #    and behaviour is byte-for-byte the pre-existing one.
    downloaded_set = indexed_set
    rep.report(0, substep="done_set",
              message="deriving done-set from tasks table")
    # db_path was resolved above (before the deep-mode branch). A DSN is
    # always queried; a SQLite path is only queried if the file exists (a
    # missing file = empty DB).
    if is_pg_url(db_path):
        done_set = derive_done_set(db_path, canonical_tics, sector=sector)
    else:
        done_set = (derive_done_set(db_path, canonical_tics, sector=sector)
                    if Path(db_path).exists() else set())

    # 3b. Build TOI skip set — TICs known to be already in the TOI catalog are
    #     treated exactly like done TICs: the cursor advances past them but they
    #     are never enqueued (they are not discovery targets).
    toi_catalog_path = (
        args.toi_catalog if args.toi_catalog
        else str(ROOT / "data" / "toi_catalog.csv")
    )
    toi_tic_set = _load_toi_tic_set(toi_catalog_path)
    tois_skipped_count = sum(1 for t in canonical_tics if t in toi_tic_set)
    if toi_tic_set:
        print(
            f"INFO: TOI catalog loaded ({len(toi_tic_set)} entries); "
            f"{tois_skipped_count} TICs in work list are TOIs and will be skipped",
            file=sys.stderr,
        )
        # Merge TOI TICs into done_set so next_block skips them transparently.
        # The cursor still advances past them (next_block uses done_set to skip,
        # and cursor = canonical row of last selected TIC, same logic).
        done_set = done_set | {t for t in canonical_tics if t in toi_tic_set}
    else:
        tois_skipped_count = 0

    # 4. Compute next block
    block = next_block(canonical_tics, args.count, downloaded_set, done_set)
    selected = block["selected"]
    ready = block["ready"]
    missing = block["missing_download"]

    if not selected:
        _emit({
            "status": "error",
            "reason": "no_undone_tics",
            "remaining_undone": block["remaining_undone"],
        })
        return 2

    if len(ready) < len(selected):
        _emit({
            "status": "need_download",
            "requested": args.count,
            "ready": len(ready),
            "missing": len(missing),
            "hint": (
                "Download more TGLC data for sector 1 from the "
                "Downloads page before scanning."
            ),
        })
        return 3

    # 5. Determine output path
    lot_csv = args.out or _auto_lot_path(sector, ready)

    # Mint run_id ONCE, here — the single source of truth (S3, see
    # exonit.discovery_state.derive_run_id's docstring) — and pass it
    # through to enqueue_from_csv.py so tasks.run_id is set at INSERT time.
    # The scan endpoint (exonit/routes/discovery.py) prefers this minted
    # value over its own (legacy) re-derivation once it sees it in this
    # plan's JSON output ("the child wins").
    run_id = derive_run_id(sector, lot_csv)

    # Cursor = 1-based canonical row index of the last selected TIC.
    # This matches the CAMPAIGN.md convention: "cursor=3500" means rows
    # 1–3500 have been processed (regardless of how many were already done).
    old_cursor = sector_entry.get("cursor", 0)
    # selected is guaranteed non-empty here (we returned early if it was)
    first_row = canonical_tics.index(selected[0]) + 1   # 1-based
    last_row = canonical_tics.index(selected[-1]) + 1   # 1-based
    new_cursor = last_row

    plan = {
        "status": "dry_run" if args.dry_run else "started",
        "sector": sector,
        "requested": args.count,
        "enqueued": len(ready),
        "lot_csv": lot_csv,
        "run_id": run_id,
        "cursor_before": old_cursor,
        "cursor_after": new_cursor,
        "remaining_undone": block["remaining_undone"],
        "tics_selected": selected[:10],  # first 10 for preview
        "tois_skipped": tois_skipped_count,
    }

    if args.dry_run:
        if args.dump_selection:
            _write_lot_csv(args.dump_selection, selected)
        _emit(plan)
        return 0

    def _advance_legacy_cursor():
        set_cursor(args.registry, sector, new_cursor)
        _append_campaign_log(sector, lot_csv, first_row, last_row)

    return _finish_survey_enqueue(
        plan, ready, lot_csv, sector, run_id, db_path, args, rep,
        block["remaining_undone"], _advance_legacy_cursor)


if __name__ == "__main__":
    sys.exit(main())
