"""Automatic lot closeout (spec §4.5, phase 1.5).

For each lot the orchestrator already produces the scorecard+dossier (step_postprocess)
and the DONE row in `data/discovery/CAMPAIGN.md` + `docs/STATUS.md`
(step_finalize -> `scripts/discovery_finalize.py`). This module ONLY adds
what is missing once the pipeline reaches `DONE`:

1. Health-check gate (`scripts/run_health_check.py`) — blocks on CRITICAL.
2. Artifact verification (scorecard + dossier present and consistent).
3. Conditional ExoMiner augment (only if the lot has ground truth).
4. Lot summary (reusing `_db_stats`/`_parse_dossier` from
   `discovery_finalize.py`, same import pattern as the orchestrator).
5. (Read-only) verification of the row in CAMPAIGN.md.
6. Weekly digest (`data/autonomy/weekly_digest.md`).
7. Git commit (never push).

`dry_run=True` (default) does all the reads/checks for real but writes no
digest and makes no commit — it is the mode for the first trial with a real
lot. `run_cmd` is injectable so tests never run a real subprocess.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from autonomy.config import REPO_ROOT

PY = sys.executable

# Same import pattern as `discovery_orchestrator.py` to reuse
# `discovery_finalize.py` without duplicating its logic. Done once, with the
# subsystem's fixed REPO_ROOT (not the `repo_root` that `close_lot` receives,
# which in tests points to a tmp_path with no code).
_SCRIPTS_DIR = str(REPO_ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from discovery_finalize import _db_stats, _parse_dossier, _read_lot_tics  # noqa: E402


@dataclass
class CloseoutResult:
    run_id: str
    status: str  # "ok" | "ok_with_warnings" | "blocked"
    steps: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "steps": self.steps,
            "summary": self.summary,
        }


def _abs(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p)


def _resolve_db(repo_root: Path) -> str:
    """Resolve the connection to the operational DB.

    Same priority order as the rest of the scripts (e.g.
    `scripts/verify_sector_complete.py::_resolve_db_path`):
    `EXONIT_DATABASE_URL` (PostgreSQL DSN, production) → `EXONIT_DB_PATH`
    (explicit SQLite path) → `data/exonit.db` (archived SQLite by default).

    Production runs on PostgreSQL via `EXONIT_DATABASE_URL` (systemd
    `EnvironmentFile=<repo>/.env`); without that environment loaded
    the closeout would silently fall back to the archived SQLite and the summary would come out
    empty (n_completed=0, everything in n_other) even though the real lot is
    100% completed in PostgreSQL.
    """
    if os.environ.get("EXONIT_DATABASE_URL"):
        return os.environ["EXONIT_DATABASE_URL"]
    if os.environ.get("EXONIT_DB_PATH"):
        return os.environ["EXONIT_DB_PATH"]
    return str(repo_root / "data" / "exonit.db")


def _step(steps: list[dict], name: str, status: str, detail: str) -> None:
    steps.append({"step": name, "status": status, "detail": detail})


# ---------------------------------------------------------------------------
# Step 2: health check (gate)
# ---------------------------------------------------------------------------

def _run_health_check(repo_root: Path, run_cmd) -> tuple[str, str]:
    cmd = [PY, "scripts/run_health_check.py", "--json"]
    try:
        proc = run_cmd(cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=120)
    except Exception as exc:  # noqa: BLE001
        return "blocked", f"could not run run_health_check.py: {exc!r}"

    rc = getattr(proc, "returncode", 2)
    tail = ((getattr(proc, "stdout", "") or "") + (getattr(proc, "stderr", "") or ""))[-500:]
    if rc == 0:
        return "ok", "health check OK"
    if rc == 1:
        return "warning", f"health check with warnings (rc=1): {tail}"
    return "blocked", f"health check CRITICAL (rc={rc}): {tail}"


# ---------------------------------------------------------------------------
# Step 3: verify artifacts
# ---------------------------------------------------------------------------

def _verify_artifacts(run_id: str, scorecard_json: str, dossier: str, repo_root: Path) -> tuple[str, str]:
    scorecard_path = _abs(scorecard_json, repo_root)
    dossier_path = _abs(dossier, repo_root)

    missing = [str(p) for p in (scorecard_path, dossier_path) if not p.exists()]
    if missing:
        return "blocked", f"missing artifact(s): {missing}"

    try:
        data = json.loads(scorecard_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return "blocked", f"unreadable scorecard JSON ({scorecard_path}): {exc!r}"

    if data.get("run_id") != run_id:
        return "blocked", (
            f"run_id {run_id!r} does not match the scorecard's "
            f"({data.get('run_id')!r})"
        )

    return "ok", "scorecard and dossier present; run_id verified"


# ---------------------------------------------------------------------------
# Step 4: conditional ExoMiner augment
# ---------------------------------------------------------------------------

def _augment_exominer(
    lot_csv_path: Path, scorecard_json: str, repo_root: Path, run_cmd, dry_run: bool
) -> tuple[str, str, str]:
    """Return (step_status, step_detail, digest_label).

    `dry_run=True` ONLY reads the lot_csv header to decide whether
    the augment would apply — it never runs `augment_scorecard_exominer_from_db.py`
    (it writes 3 real files to disk, it is not a read/check).
    """
    try:
        with lot_csv_path.open(newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), [])
    except Exception as exc:  # noqa: BLE001
        detail = f"could not read the header of {lot_csv_path}: {exc!r}"
        return "warning", detail, f"skipped: {detail}"

    if "disposition" not in header:
        reason = "sample without ground truth (deep lot): augment not applicable"
        return "skipped", reason, f"skipped: {reason}"

    db_path = _resolve_db(repo_root)
    scorecard_path = _abs(scorecard_json, repo_root)
    out_stem = str(scorecard_path.with_suffix(""))

    cmd = [
        PY, "scripts/augment_scorecard_exominer_from_db.py",
        "--sample", str(lot_csv_path),
        "--db", db_path,
        "--out", out_stem + ".augmented",
    ]

    if dry_run:
        detail = f"dry_run: would run augment with {cmd}"
        return "dry_run", detail, detail

    try:
        proc = run_cmd(cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=600)
    except Exception as exc:  # noqa: BLE001
        return "warning", f"ExoMiner augment failed to run: {exc!r}", "done (error on run)"

    rc = getattr(proc, "returncode", 1)
    if rc == 0:
        return "ok", "ExoMiner augment executed", "done"
    tail = ((getattr(proc, "stderr", "") or "") + (getattr(proc, "stdout", "") or ""))[-500:]
    return "warning", f"ExoMiner augment rc={rc}: {tail}", f"done (rc={rc})"


# ---------------------------------------------------------------------------
# Step 5: lot summary
# ---------------------------------------------------------------------------

def _build_summary(
    lot_csv_raw: str,
    lot_csv_path: Path,
    dossier_path: Path,
    db_path: str,
    started_at: Optional[str],
    finished_at: Optional[str],
    sector: Any,
) -> tuple[str, str, dict]:
    warn = False
    warn_bits: list[str] = []

    tic_ids: list[str] = []
    try:
        tic_ids = _read_lot_tics(str(lot_csv_path))
    except Exception as exc:  # noqa: BLE001
        warn = True
        warn_bits.append(f"reading lot_csv: {exc!r}")

    stats = {"n_lot": None, "n_completed": None, "n_failed": None, "n_other": None}
    try:
        stats = _db_stats(db_path, tic_ids)
    except Exception as exc:  # noqa: BLE001
        warn = True
        warn_bits.append(f"_db_stats: {exc!r}")

    dossier_info = {"n_strong": None, "n_finalists": None}
    try:
        dossier_info = _parse_dossier(str(dossier_path))
    except Exception as exc:  # noqa: BLE001
        warn = True
        warn_bits.append(f"_parse_dossier: {exc!r}")
    if dossier_info.get("n_strong") is None or dossier_info.get("n_finalists") is None:
        warn = True
        warn_bits.append("n_strong/n_finalists not found in the dossier")

    duration_h = None
    rate_tics_min = None
    try:
        t0 = datetime.fromisoformat(started_at)
        t1 = datetime.fromisoformat(finished_at)
        duration_h = round((t1 - t0).total_seconds() / 3600.0, 2)
        n_completed = stats.get("n_completed")
        if duration_h and n_completed is not None:
            rate_tics_min = round(n_completed / (duration_h * 60.0), 1)
    except Exception as exc:  # noqa: BLE001
        warn = True
        warn_bits.append(f"duration: {exc!r}")

    summary = {
        "n_lot": stats.get("n_lot"),
        "n_completed": stats.get("n_completed"),
        "n_failed": stats.get("n_failed"),
        "n_other": stats.get("n_other"),
        "n_strong": dossier_info.get("n_strong"),
        "n_finalists": dossier_info.get("n_finalists"),
        "duration_h": duration_h,
        "rate_tics_min": rate_tics_min,
        "sector": sector,
        "lot_csv": lot_csv_raw,
    }

    if warn:
        return "warning", "partial summary: " + "; ".join(warn_bits), summary
    return "ok", "complete summary", summary


# ---------------------------------------------------------------------------
# Step 6: verify CAMPAIGN.md (read-only, we never modify it here)
# ---------------------------------------------------------------------------

def _verify_campaign(lot_csv_basename: str, repo_root: Path) -> tuple[str, str]:
    campaign_path = repo_root / "data" / "discovery" / "CAMPAIGN.md"
    try:
        text = campaign_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return "warning", f"could not read CAMPAIGN.md: {exc!r}"

    pattern = f"`{lot_csv_basename}`"
    matching = [line for line in text.splitlines() if pattern in line]
    if not matching:
        return "warning", f"no row in CAMPAIGN.md contains `{lot_csv_basename}`"

    line = matching[0]
    if "**DONE**" in line:
        return "ok", "DONE row in CAMPAIGN.md"
    return "warning", f"the row for {lot_csv_basename} does not say DONE yet: {line.strip()}"


# ---------------------------------------------------------------------------
# Step 7: weekly digest
# ---------------------------------------------------------------------------

_DIGEST_HEADER = "# Weekly digest — autonomous runs\n"


def _digest_block(
    run_id: str,
    summary: dict,
    health_label: str,
    augment_label: str,
    closeout_status: str,
    dry_run: bool,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    return (
        f"## {run_id} — {ts} UTC\n"
        f"- Lot: {summary.get('lot_csv')} (sector {summary.get('sector')}) — "
        f"{summary.get('n_completed')}/{summary.get('n_lot')} completed, "
        f"{summary.get('n_failed')} failed\n"
        f"- Strong: {summary.get('n_strong')} · Finalists: {summary.get('n_finalists')} · "
        f"Duration: {summary.get('duration_h')} h ({summary.get('rate_tics_min')} TICs/min)\n"
        f"- Health check: {health_label}\n"
        f"- Augment ExoMiner: {augment_label}\n"
        f"- Closeout: {closeout_status} (dry_run={dry_run})\n"
    )


def _write_digest(
    repo_root: Path,
    run_id: str,
    summary: dict,
    health_label: str,
    augment_label: str,
    closeout_status: str,
    dry_run: bool,
) -> tuple[str, str]:
    block = _digest_block(run_id, summary, health_label, augment_label, closeout_status, dry_run)
    if dry_run:
        return "skipped", f"dry_run: digest not written; block that would be written:\n{block}"

    digest_path = repo_root / "data" / "autonomy" / "weekly_digest.md"
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    if not digest_path.exists():
        digest_path.write_text(_DIGEST_HEADER, encoding="utf-8")
    with digest_path.open("a", encoding="utf-8") as fh:
        fh.write("\n" + block)
    return "ok", f"digest updated: {digest_path}"


# ---------------------------------------------------------------------------
# Step 8: git commit (never push)
# ---------------------------------------------------------------------------

def _commit(
    repo_root: Path,
    run_id: str,
    run_cmd,
    dry_run: bool,
    commit: bool,
    status_so_far: str,
) -> tuple[str, str]:
    if dry_run:
        return "skipped", "commit not made (dry_run)"
    if not commit:
        return "skipped", "commit not made (commit=False)"
    if status_so_far == "blocked":
        return "skipped", "commit not made (status=blocked)"

    candidate_paths = [
        repo_root / "data" / "scorecards" / f"{run_id}.json",
        repo_root / "data" / "scorecards" / f"{run_id}_dossier.md",
        repo_root / "data" / "discovery" / "CAMPAIGN.md",
        repo_root / "docs" / "STATUS.md",
    ]
    existing = [str(p) for p in candidate_paths if p.exists()]
    if not existing:
        return "skipped", "none of the candidate paths exists on disk"

    try:
        run_cmd(["git", "add"] + existing, cwd=str(repo_root), capture_output=True, text=True)
        # Explicit pathspec: we only care whether there ARE staged changes in
        # THESE 4 paths — not anything else the operator may have
        # staged in the live prod checkout.
        diff_proc = run_cmd(
            ["git", "diff", "--cached", "--quiet", "--"] + existing, cwd=str(repo_root),
            capture_output=True, text=True,
        )
    except Exception as exc:  # noqa: BLE001
        return "warning", f"git add/diff failed: {exc!r}"

    if getattr(diff_proc, "returncode", 1) == 0:
        return "skipped", "nothing to commit"

    message = (
        f"chore(campaign): lot closeout {run_id}\n\n"
        "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
    )
    try:
        # Explicit pathspec in the commit too: it never takes along anything else
        # the operator may have staged in the live prod checkout.
        commit_proc = run_cmd(
            ["git", "commit", "-m", message, "--"] + existing, cwd=str(repo_root),
            capture_output=True, text=True,
        )
    except Exception as exc:  # noqa: BLE001
        return "warning", f"git commit failed: {exc!r}"

    rc = getattr(commit_proc, "returncode", 1)
    if rc != 0:
        tail = ((getattr(commit_proc, "stderr", "") or "") + (getattr(commit_proc, "stdout", "") or ""))[-500:]
        return "warning", f"git commit rc={rc}: {tail}"

    return "ok", f"commit created for: {existing}"


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def _overall_status(steps: list[dict]) -> str:
    statuses = {s["status"] for s in steps}
    if "blocked" in statuses:
        return "blocked"
    if "warning" in statuses:
        return "ok_with_warnings"
    return "ok"


def close_lot(
    state_path: Path,
    *,
    dry_run: bool = True,
    commit: bool = True,
    repo_root: Path = REPO_ROOT,
    run_cmd=subprocess.run,
) -> CloseoutResult:
    """Close a lot that has already reached `DONE` (spec §4.5). See the module docstring."""
    repo_root = Path(repo_root)
    steps: list[dict] = []

    # --- Step 1: read state ------------------------------------------------
    try:
        data = json.loads(Path(state_path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _step(steps, "read_state", "blocked", f"cannot read {state_path}: {exc!r}")
        return CloseoutResult(run_id="", status="blocked", steps=steps, summary={})

    run_id = data.get("run_id", "") or ""
    state = data.get("state")

    if state != "DONE":
        _step(
            steps, "read_state", "blocked",
            f"pipeline state is {state!r}, not 'DONE' — the lot is not closed",
        )
        return CloseoutResult(run_id=run_id, status="blocked", steps=steps, summary={})

    _step(steps, "read_state", "ok", f"state=DONE (run_id={run_id})")

    lot_csv_raw = data.get("lot_csv", "")
    scorecard_json = data.get("scorecard_json", "")
    dossier = data.get("dossier", "")
    started_at = data.get("started_at")
    finished_at = data.get("finished_at")
    sector = data.get("sector")

    lot_csv_path = _abs(lot_csv_raw, repo_root)
    dossier_path = _abs(dossier, repo_root)
    db_path = _resolve_db(repo_root)

    # --- Step 2: health check (gate) -----------------------------------------
    hc_status, hc_detail = _run_health_check(repo_root, run_cmd)
    _step(steps, "health_check", hc_status, hc_detail)
    health_label = {"ok": "ok", "warning": "warning", "blocked": "critical"}[hc_status]
    if hc_status != "ok":
        health_label = f"{health_label} {hc_detail}"

    # --- Step 3: verify artifacts ----------------------------------------
    art_status, art_detail = _verify_artifacts(run_id, scorecard_json, dossier, repo_root)
    _step(steps, "verify_artifacts", art_status, art_detail)

    # --- Step 4: conditional ExoMiner augment --------------------------------
    aug_status, aug_detail, augment_label = _augment_exominer(
        lot_csv_path, scorecard_json, repo_root, run_cmd, dry_run
    )
    _step(steps, "augment_exominer", aug_status, aug_detail)

    # --- Step 5: lot summary ------------------------------------------------
    sum_status, sum_detail, summary = _build_summary(
        lot_csv_raw, lot_csv_path, dossier_path, db_path, started_at, finished_at, sector,
    )
    _step(steps, "summary", sum_status, sum_detail)

    # --- Step 6: verify CAMPAIGN.md (read-only) ---------------------------
    lot_csv_basename = Path(lot_csv_raw).name
    camp_status, camp_detail = _verify_campaign(lot_csv_basename, repo_root)
    _step(steps, "verify_campaign", camp_status, camp_detail)

    # --- Step 7: weekly digest ----------------------------------------------
    status_so_far = _overall_status(steps)
    dig_status, dig_detail = _write_digest(
        repo_root, run_id, summary, health_label, augment_label, status_so_far, dry_run,
    )
    _step(steps, "digest", dig_status, dig_detail)

    # --- Step 8: git commit (never push) --------------------------------------
    status_before_commit = _overall_status(steps)
    commit_status, commit_detail = _commit(
        repo_root, run_id, run_cmd, dry_run, commit, status_before_commit,
    )
    _step(steps, "commit", commit_status, commit_detail)

    final_status = _overall_status(steps)
    return CloseoutResult(run_id=run_id, status=final_status, steps=steps, summary=summary)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Automatic lot closeout (spec §4.5).")
    ap.add_argument(
        "--state",
        default=str(REPO_ROOT / "data" / "discovery" / "pipeline_state.json"),
        help="Path to pipeline_state.json",
    )
    ap.add_argument(
        "--execute", action="store_true",
        help="Disable dry_run (default: dry-run)",
    )
    ap.add_argument(
        "--no-commit", action="store_true",
        help="Do not commit even if status != blocked",
    )
    ap.add_argument("--json", action="store_true", help="Print the result as JSON")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    result = close_lot(
        Path(args.state),
        dry_run=not args.execute,
        commit=not args.no_commit,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"=== closeout {result.run_id} — {result.status} ===")
        for s in result.steps:
            print(f"  [{s['status']:>8}] {s['step']}: {s['detail']}")
        print("--- summary ---")
        for k, v in result.summary.items():
            print(f"  {k}: {v}")

    return 0 if result.status in ("ok", "ok_with_warnings") else 2


if __name__ == "__main__":
    sys.exit(main())
