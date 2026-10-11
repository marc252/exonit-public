#!/usr/bin/env python3
"""Auto-finalize a completed discovery lot.

Writes two factual, deterministic updates at the end of a discovery run:
  1. CAMPAIGN.md: changes the lot row from ENQUEUED → DONE with objective stats.
  2. STATUS.md: appends/updates one row in an auto-managed section.

No scientific interpretation, no invented classifications.

CLI usage:
  python3 scripts/discovery_finalize.py \\
      --lot-csv data/discovery/s0001_lot_20260530T173210.csv \\
      --run-id discovery_s0001_lotX_2026-05-30 \\
      --scorecard-json data/scorecards/discovery_s0001_lotX_2026-05-30.json \\
      --dossier data/scorecards/discovery_s0001_lotX_2026-05-30_dossier.md
"""

import argparse
import csv
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from exonit.db_backend import connect_operational, is_pg_url  # noqa: E402

logger = logging.getLogger("discovery_finalize")

# ---------------------------------------------------------------------------
# Markers that delimit the auto-managed section inside STATUS.md
# ---------------------------------------------------------------------------

_STATUS_SECTION_START = "<!-- DISCOVERY_RUN_LOG:START -->"
_STATUS_SECTION_END = "<!-- DISCOVERY_RUN_LOG:END -->"
_STATUS_TABLE_HEADER = (
    "## Discovery run log (auto)\n"
    "| Run | Date | Lot CSV | Completed | Failed | Strong | Finalists | Scorecard |\n"
    "|---|---|---|---|---|---|---|---|"
)


# ---------------------------------------------------------------------------
# 1. Stats from the DB (read-only)
# ---------------------------------------------------------------------------

def _read_lot_tics(lot_csv: str) -> list[str]:
    """Return list of tic_id as strings from the lot CSV (col 'tic_id')."""
    tics: list[str] = []
    try:
        with open(lot_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                raw = row.get("tic_id", "").strip()
                if raw and raw.lower() != "nan":
                    # Normalise to plain integer string (drop .0 if float-like)
                    try:
                        tics.append(str(int(float(raw))))
                    except ValueError:
                        tics.append(raw)
    except Exception as exc:
        logger.warning("Could not read lot CSV %s: %s", lot_csv, exc)
    return tics


def _db_stats(db_path: str, tic_ids: list[str], run_id: str | None = None) -> dict:
    """Query task counts for the given TIC ids from the operational DB.

    Returns dict with keys: n_lot, n_completed, n_failed, n_other.
    IMPORTANT: tic_id is TEXT in the DB; we compare as strings.

    Routes through :func:`connect_operational`, so a ``postgresql://`` DSN works
    just like a SQLite path.

    run_id (S3, docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md
    §3.1 item 3): when given AND ``exonit.db.ExoNitDB.run_scope_kind(run_id)
    == "tasks"`` (``tasks.run_id`` populated for this run — see
    ``EXONIT_RUN_SCOPE``), a single ``GROUP BY status`` query scoped to
    ``run_id`` replaces the two whole-``tasks``-table ``DISTINCT tic_id``
    scans below — index-only via ``idx_tasks_run_id``, and exact for THIS
    lot specifically (no risk of counting a same-TIC task from a different
    lot/sector as this lot's completion). Falls back to the pre-S3 path
    (whole-table scan + Python intersect — the trap ``derive_done_set``
    already hit for a giant ``IN (?, ?, ...)`` clause) when ``run_id`` is
    absent or not yet scoped this way (kind ``"run_tics"``/``None``).
    """
    n_lot = len(tic_ids)
    if n_lot == 0:
        return {"n_lot": 0, "n_completed": 0, "n_failed": 0, "n_other": 0}

    lot_set = set(tic_ids)
    n_completed = 0
    n_failed = 0
    try:
        backend = connect_operational(db_path)
        try:
            scope_kind = None
            if run_id:
                try:
                    from exonit.db import ExoNitDB
                    scope_kind = ExoNitDB(db_path).run_scope_kind(run_id)
                except Exception:
                    scope_kind = None
            if scope_kind == "tasks":
                rows = backend.query(
                    "SELECT status, COUNT(DISTINCT tic_id) AS n FROM tasks "
                    "WHERE run_id = ? GROUP BY status", (run_id,))
                n_completed = sum(
                    r["n"] for r in rows if r["status"] == "completed")
                n_failed = sum(
                    r["n"] for r in rows
                    if r["status"] in ("failed", "load_failed", "error"))
            else:
                # Rather than a giant `IN (?, ?, ...)` clause (which blows
                # past the SQLite/PG parameter limit on a 50k-TIC lot — the
                # same trap `derive_done_set` already hit), the completed
                # and failed TIC sets are loaded once and intersected with
                # the lot in Python.
                completed = {
                    str(r["tic_id"]) for r in backend.query(
                        "SELECT DISTINCT tic_id FROM tasks WHERE status='completed'")
                }
                failed = {
                    str(r["tic_id"]) for r in backend.query(
                        "SELECT DISTINCT tic_id FROM tasks "
                        "WHERE status IN ('failed','load_failed','error')")
                }
                # A TIC could in principle be both completed and failed
                # (re-run); count it as completed first, matching the
                # historical query order.
                completed_in_lot = lot_set & completed
                failed_in_lot = (lot_set & failed) - completed_in_lot
                n_completed = len(completed_in_lot)
                n_failed = len(failed_in_lot)
        finally:
            backend.close()
    except Exception as exc:
        logger.warning("DB query failed (%s): %s", db_path, exc)
        n_completed = 0
        n_failed = 0

    n_other = n_lot - n_completed - n_failed
    return {
        "n_lot": n_lot,
        "n_completed": n_completed,
        "n_failed": n_failed,
        "n_other": n_other,
    }


# ---------------------------------------------------------------------------
# 2. Parse dossier markdown
# ---------------------------------------------------------------------------

def _parse_dossier(dossier_path: str) -> dict:
    """Extract n_strong and n_finalists from the dossier markdown.

    Lines we look for (generated by discovery_postprocess.py):
      Strong candidates (v6>0.5, post-veto): N.
      **Finalists (pass all + FPP<0.5): M**: ...
    (older dossiers use the Catalan form "**Finalistes (passen tot + FPP<0,5): M**:";
    both are accepted)

    Returns dict with 'n_strong' and 'n_finalists' (int or None if not found).
    """
    result: dict = {"n_strong": None, "n_finalists": None}
    try:
        text = Path(dossier_path).read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Cannot read dossier %s: %s", dossier_path, exc)
        return result

    # Strong: "Strong candidates (v6>..., post-veto): N"
    m = re.search(r"Strong candidates\s*\([^)]+\)\s*:\s*(\d+)", text)
    if m:
        result["n_strong"] = int(m.group(1))
    else:
        logger.warning("n_strong not found in dossier %s", dossier_path)

    # Finalists: "**Finalists (pass all + FPP<0.5): M**:" -- also accepts the
    # legacy Catalan label "Finalistes" (historical dossiers), and "...: 0".
    m = re.search(r"\**(?:Finalists|Finalistes)\b[^:]*:\s*(\d+)", text)
    if m:
        result["n_finalists"] = int(m.group(1))
    else:
        logger.warning("n_finalists not found in dossier %s", dossier_path)

    return result


# ---------------------------------------------------------------------------
# 3. Update CAMPAIGN.md
# ---------------------------------------------------------------------------

def _update_campaign(
    campaign_path: str,
    lot_csv_basename: str,
    scorecard_stem: str,
    stats: dict,
    dossier_info: dict,
    today: str,
) -> str:
    """Update CAMPAIGN.md, changing the lot row from ENQUEUED to DONE.

    Returns one of: 'updated', 'already_done', 'no_unique_match'.
    """
    try:
        text = Path(campaign_path).read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Cannot read CAMPAIGN.md %s: %s", campaign_path, exc)
        return "no_unique_match"

    lines = text.splitlines(keepends=True)

    # Find lines that reference this exact lot CSV basename in backticks
    backtick_pattern = f"`{lot_csv_basename}`"
    matching_indices = [
        i for i, line in enumerate(lines) if backtick_pattern in line
    ]

    if len(matching_indices) == 0:
        logger.warning(
            "CAMPAIGN.md: no line found containing `%s` — cannot finalize",
            lot_csv_basename,
        )
        return "no_unique_match"

    if len(matching_indices) > 1:
        logger.warning(
            "CAMPAIGN.md: %d lines found containing `%s` (expected exactly 1) — "
            "not touching anything",
            len(matching_indices),
            lot_csv_basename,
        )
        return "no_unique_match"

    idx = matching_indices[0]
    original_line = lines[idx]

    # Already done?
    if "DONE" in original_line:
        logger.info("CAMPAIGN.md: lot already marked DONE — idempotent, skipping")
        return "already_done"

    # Not ENQUEUED? Warn but still proceed (e.g. the line may say something else).
    if "ENQUEUED" not in original_line:
        logger.warning(
            "CAMPAIGN.md: line for `%s` does not contain ENQUEUED: %s",
            lot_csv_basename,
            original_line.strip(),
        )

    # Build the replacement status cell
    n_completed = stats["n_completed"]
    n_lot = stats["n_lot"]
    n_failed = stats["n_failed"]
    n_strong = dossier_info.get("n_strong")
    n_finalists = dossier_info.get("n_finalists")

    strong_str = str(n_strong) if n_strong is not None else "?"
    finalist_str = str(n_finalists) if n_finalists is not None else "?"

    new_status = (
        f"**DONE** ({today}) — scorecard"
        f" `{scorecard_stem}.{{json,_dossier.md}}`."
        f" {n_completed}/{n_lot} completed, {n_failed} failed."
        f" {strong_str} strong (v6>0.5, post-veto), {finalist_str} finalists."
    )

    # The line is a markdown table row: | ... | ... | ... | STATUS_CELL |
    # Replace only the last field (after the last | before the trailing |).
    # We find the last '|'-delimited chunk and swap it.
    stripped = original_line.rstrip("\n\r")

    # Split on '|', replace last non-empty trailing segment
    parts = stripped.split("|")
    # parts[0] is empty (before first |), parts[-1] may be empty (after last |)
    # Walk from right to find the status cell
    last_content_idx = len(parts) - 1
    while last_content_idx > 0 and parts[last_content_idx].strip() == "":
        last_content_idx -= 1

    if last_content_idx <= 0:
        logger.warning("CAMPAIGN.md: cannot parse table row: %s", stripped)
        return "no_unique_match"

    parts[last_content_idx] = f" {new_status} "
    new_line = "|".join(parts)
    # Preserve original trailing newline
    trailing = original_line[len(original_line.rstrip("\n\r")):]
    new_line = new_line + (trailing or "\n")

    lines[idx] = new_line
    new_text = "".join(lines)

    try:
        Path(campaign_path).write_text(new_text, encoding="utf-8")
    except Exception as exc:
        logger.error("Cannot write CAMPAIGN.md: %s", exc)
        return "no_unique_match"

    logger.info("CAMPAIGN.md: lot %s → DONE", lot_csv_basename)
    return "updated"


# ---------------------------------------------------------------------------
# 4. Update STATUS.md auto section
# ---------------------------------------------------------------------------

def _update_status(
    status_path: str,
    run_id: str,
    lot_csv_basename: str,
    stats: dict,
    dossier_info: dict,
    scorecard_stem: str,
    today: str,
) -> None:
    """Add or replace a row in the auto-managed discovery run log table."""
    n_strong = dossier_info.get("n_strong")
    n_finalists = dossier_info.get("n_finalists")

    row = (
        f"| {run_id} | {today} | `{lot_csv_basename}` "
        f"| {stats['n_completed']} | {stats['n_failed']} "
        f"| {n_strong if n_strong is not None else '?'} "
        f"| {n_finalists if n_finalists is not None else '?'} "
        f"| `{scorecard_stem}` |"
    )

    try:
        text = Path(status_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        text = ""
    except Exception as exc:
        logger.warning("Cannot read STATUS.md %s: %s", status_path, exc)
        return

    has_section = _STATUS_SECTION_START in text and _STATUS_SECTION_END in text

    if not has_section:
        # Append the entire section at the end
        section = (
            "\n" + _STATUS_SECTION_START + "\n"
            + _STATUS_TABLE_HEADER + "\n"
            + row + "\n"
            + _STATUS_SECTION_END + "\n"
        )
        new_text = text.rstrip("\n") + "\n" + section
    else:
        # Extract section content between markers
        start_idx = text.index(_STATUS_SECTION_START)
        end_idx = text.index(_STATUS_SECTION_END)
        section_body = text[start_idx + len(_STATUS_SECTION_START):end_idx]

        # Check if run_id already exists; if so, replace that line.
        # Use plain substring match (run_ids contain hyphens; \b is unreliable).
        section_lines = section_body.splitlines(keepends=True)
        found = False
        for i, sline in enumerate(section_lines):
            if run_id in sline:
                section_lines[i] = row + "\n"
                found = True
                break

        if not found:
            # Append before the END marker (after last table row)
            # Find the last non-empty line in section_lines and insert after
            section_lines.append(row + "\n")

        new_section_body = "".join(section_lines)
        new_text = (
            text[:start_idx + len(_STATUS_SECTION_START)]
            + new_section_body
            + text[end_idx:]
        )

    try:
        Path(status_path).write_text(new_text, encoding="utf-8")
    except Exception as exc:
        logger.error("Cannot write STATUS.md: %s", exc)
        return

    logger.info("STATUS.md: row for run_id=%s written", run_id)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def finalize_run(
    lot_csv: str,
    run_id: str,
    scorecard_json: str,
    dossier: str,
    db_path: str,
    campaign_path: str | None = None,
    status_path: str | None = None,
) -> dict:
    """Finalize a discovery run: update CAMPAIGN.md and STATUS.md.

    Parameters
    ----------
    lot_csv:         Path to the lot CSV (tic_id,label).
    run_id:          Run identifier (e.g. 'discovery_s0001_lotX_2026-05-30').
    scorecard_json:  Path to the scorecard JSON (may be relative to ROOT).
    dossier:         Path to the dossier markdown (may be relative to ROOT).
    db_path:         Path to exonit.db.
    campaign_path:   Path to CAMPAIGN.md (default: data/discovery/CAMPAIGN.md).
    status_path:     Path to STATUS.md (default: docs/STATUS.md).

    Returns
    -------
    dict with keys:
        campaign_action: 'updated' | 'already_done' | 'no_unique_match'
        n_lot, n_completed, n_failed, n_other
        n_strong, n_finalists (int or None)
    """
    campaign_path = campaign_path or str(ROOT / "data" / "discovery" / "CAMPAIGN.md")
    status_path = status_path or str(ROOT / "docs" / "STATUS.md")

    # Resolve relative paths to absolute (relative to ROOT)
    def _abs(p: str) -> str:
        pp = Path(p)
        if not pp.is_absolute():
            pp = ROOT / p
        return str(pp)

    lot_csv = _abs(lot_csv)
    scorecard_json = _abs(scorecard_json)
    dossier = _abs(dossier)
    # A PG DSN must NOT be path-resolved (it is not a filesystem path).
    if not is_pg_url(db_path):
        db_path = _abs(db_path)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1. Stats from DB
    tic_ids = _read_lot_tics(lot_csv)
    stats = _db_stats(db_path, tic_ids, run_id=run_id)
    if stats["n_other"] > 0:
        logger.warning(
            "%d TICs in lot have no task or unexpected status in DB "
            "(n_lot=%d, n_completed=%d, n_failed=%d)",
            stats["n_other"],
            stats["n_lot"],
            stats["n_completed"],
            stats["n_failed"],
        )

    # 2. Parse dossier
    dossier_info = _parse_dossier(dossier)

    # 3. Derive scorecard stem (basename without .json extension)
    scorecard_stem = Path(scorecard_json).stem  # e.g. discovery_s0001_lotX_2026-05-30

    lot_csv_basename = Path(lot_csv).name

    # 4. Update CAMPAIGN.md
    campaign_action = _update_campaign(
        campaign_path=campaign_path,
        lot_csv_basename=lot_csv_basename,
        scorecard_stem=scorecard_stem,
        stats=stats,
        dossier_info=dossier_info,
        today=today,
    )

    # 5. Update STATUS.md
    _update_status(
        status_path=status_path,
        run_id=run_id,
        lot_csv_basename=lot_csv_basename,
        stats=stats,
        dossier_info=dossier_info,
        scorecard_stem=scorecard_stem,
        today=today,
    )

    return {
        "campaign_action": campaign_action,
        "n_lot": stats["n_lot"],
        "n_completed": stats["n_completed"],
        "n_failed": stats["n_failed"],
        "n_other": stats["n_other"],
        "n_strong": dossier_info.get("n_strong"),
        "n_finalists": dossier_info.get("n_finalists"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Finalize a discovery run: update CAMPAIGN.md + STATUS.md."
    )
    p.add_argument("--lot-csv", required=True,
                   help="Path to the lot CSV (tic_id,label)")
    p.add_argument("--run-id", required=True,
                   help="Run identifier, e.g. discovery_s0001_lotX_2026-05-30")
    p.add_argument("--scorecard-json", required=True,
                   help="Path to scorecard JSON (may be relative to repo root)")
    p.add_argument("--dossier", required=True,
                   help="Path to dossier markdown (may be relative to repo root)")
    p.add_argument("--db", default=str(ROOT / "data" / "exonit.db"),
                   help="Path to exonit.db")
    p.add_argument("--campaign", default=str(ROOT / "data" / "discovery" / "CAMPAIGN.md"),
                   help="Path to CAMPAIGN.md")
    p.add_argument("--status", default=str(ROOT / "docs" / "STATUS.md"),
                   help="Path to STATUS.md")
    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    result = finalize_run(
        lot_csv=args.lot_csv,
        run_id=args.run_id,
        scorecard_json=args.scorecard_json,
        dossier=args.dossier,
        db_path=args.db,
        campaign_path=args.campaign,
        status_path=args.status,
    )
    import json
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
