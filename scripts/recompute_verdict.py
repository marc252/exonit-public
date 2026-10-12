#!/usr/bin/env python3
"""Recompute claude_verdict for a directed set of (tic_id, candidate_num).

Recomputes claude_verdict using the production verdict classifier
(EXONIT_VERDICT_VERSION=v2.1) from current PG data. No LC, no fleet, no GPU.
Pure function of PG-available data — exactly mirrors server._run_verdict_background,
including its FFI/SPOC routing: this script bulk-loads an exptime map per
recompute batch and passes data_source= to predict_verdict, so it honours
EXONIT_VERDICT_FFI_VERSION from the environment exactly like the server does
(unset by default here — see the env-var block below).

Usage examples
--------------
# Dry-run — shows transition matrix, no DB writes
python3 scripts/recompute_verdict.py \\
    --from-csv progress/changed_set_s1.csv --sector 1 --dry-run

# Backup old verdicts before writing
python3 scripts/recompute_verdict.py \\
    --from-csv progress/changed_set_s1.csv --sector 1 \\
    --backup progress/backup_verdict_s1_pre_recompute.csv --dry-run

# Full recompute (orchestrator applies this; do not run manually without --dry-run first)
python3 scripts/recompute_verdict.py \\
    --from-csv progress/changed_set_s1.csv --sector 1

# Parity check: verify tool path == direct predict_verdict, no stored-verdict comparison
python3 scripts/recompute_verdict.py --parity-check 1 50
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Set env vars BEFORE importing the verdict classifier.
# predict_verdict routes via EXONIT_VERDICT_VERSION read at call time, but
# _load_v2_1_model() is cached with lru_cache — setting here ensures the
# first call lands on v2.1 exactly as production.
#
# EXONIT_VERDICT_FFI_VERSION is deliberately NOT set here — this script
# honours whatever the caller's environment has (empty by default, same
# byte-for-byte v2.1-everything behaviour as before the FFI/SPOC routing
# was added). Set it in the environment before running this script to
# recompute FFI candidates through v3_tfopwg too.
# ---------------------------------------------------------------------------
os.environ.setdefault("EXONIT_VERDICT_VERSION", "v2.1")
os.environ.setdefault("EXONIT_USE_CLAUDE_FALLBACK", "0")

ROOT = Path(__file__).resolve().parent.parent

# D8 (docs/reviews/2026-07-02-opus48-cleanup-review.md): the candidate/
# star_info dict-shaping this script needs is now owned by exonit.verdict_io
# (also imported by exonit/server.py's _run_verdict_background), so this
# import cannot drift from what production actually does. verdict_io has no
# dependency on exonit.db/exonit.server/verdict_classifier, so importing it
# here does not disturb the env-var-before-import ordering above.
from exonit.verdict_io import (  # noqa: E402
    build_candidate_dict as _verdict_io_build_candidate_dict,
    build_star_info as _verdict_io_build_star_info,
    default_vetting_result,
)
# Single source of truth for the OR-gate predicate (2026-07-11) — same
# function exonit/routes/workers.py's three verdict-trigger call sites use,
# so this script cannot drift from production's pre-gate again.
from exonit.work_producers import verdict_eligible  # noqa: E402
# FFI/SPOC routing (docs/superpowers/specs/2026-09-19-verdict-v3-tfopwg-
# design.md §5) — same threshold logic server.py's _run_verdict_background
# uses via resolve_data_source, applied here to a bulk-loaded exptime map
# (one MetadataCache read per recompute batch, not one per row).
from exonit.config import ExoNitConfig  # noqa: E402
from exonit.metadata_cache import MetadataCache  # noqa: E402
from exonit.verdict_routing import data_source_from_exptime  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("recompute_verdict")

DEFAULT_DB_URL = os.environ.get(
    "EXONIT_DATABASE_URL", "postgresql://exonit@localhost:5432/exonit")

# Commit to PG every this many rows (avoids very large transactions).
BATCH_SIZE = 500

# Columns to SELECT from candidates (subset the classifier reads).
CAND_COLS = (
    "tic_id", "candidate_num", "period_days", "depth_ppm", "duration_days",
    "snr", "cnn_score", "triceratops_fpp", "triceratops_nfpp",
    "triceratops_prob_planet", "exominer_score", "exominer_ffi_score",
    "exominer_tce_period", "exominer_sector_run", "claude_verdict",
)

# Columns to SELECT from stars — mirrors server._run_verdict_background exactly.
STAR_COLS = (
    "tic_id", "tmag", "ra", "dec", "n_sectors", "time_span_days", "n_datapoints",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> list[tuple[str, int]]:
    """Read (tic_id, candidate_num) pairs from the CSV.

    Ignores extra columns (e.g. cnn_old, cnn_new, delta from changed_set_s1.csv).
    """
    pairs: list[tuple[str, int]] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pairs.append((str(row["tic_id"]), int(row["candidate_num"])))
    return pairs


def _fetch_candidates(dsn: str, sector: int,
                      pairs: list[tuple[str, int]]) -> dict[tuple[str, int], dict]:
    """Batch-fetch candidate rows via unnest JOIN.

    Returns a dict keyed by (tic_id, candidate_num). Missing rows are absent
    from the result (the caller counts them as n_no_candidate_row).
    """
    import psycopg

    tic_ids = [p[0] for p in pairs]
    cnums = [p[1] for p in pairs]
    cols_sql = ", ".join(f"c.{c}" for c in CAND_COLS)
    query = f"""
        SELECT {cols_sql}
        FROM candidates c
        JOIN unnest(%s::text[], %s::bigint[]) AS t(tic_id, candidate_num)
          ON c.tic_id = t.tic_id AND c.candidate_num = t.candidate_num
        WHERE c.sector = %s
    """
    result: dict[tuple[str, int], dict] = {}
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (tic_ids, cnums, sector))
            for row in cur.fetchall():
                d = dict(zip(CAND_COLS, row))
                key = (str(d["tic_id"]), int(d["candidate_num"]))
                result[key] = d
    return result


def _fetch_stars(dsn: str, tic_ids: list[str]) -> dict[str, dict]:
    """Batch-fetch star rows via ANY array lookup.

    Returns a dict keyed by tic_id. Missing TICs are absent.
    """
    import psycopg

    cols_sql = ", ".join(STAR_COLS)
    query = f"""
        SELECT {cols_sql}
        FROM stars
        WHERE tic_id = ANY(%s::text[])
    """
    result: dict[str, dict] = {}
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (tic_ids,))
            for row in cur.fetchall():
                d = dict(zip(STAR_COLS, row))
                result[str(d["tic_id"])] = d
    return result


# ---------------------------------------------------------------------------
# Dict construction — D8: thin wrappers around exonit.verdict_io, the shared
# module server.py's _run_verdict_background also imports (see that module's
# docstring). Kept as local names/signatures for backward compatibility with
# this script's own call sites below and tests/test_recompute_verdict_parity.py
# rather than rewriting every call site; the actual hand-rolled dict-shaping
# logic (the former "mirror") is gone.
# ---------------------------------------------------------------------------

def _build_candidate_dict(row: dict) -> dict:
    """Build the candidate dict that predict_verdict expects.

    Delegates to exonit.verdict_io.build_candidate_dict (same field set:
    tic_id, candidate_num, plus all of CANDIDATE_FIELDS — passed through
    as-is from the DB, including sentinels like exominer_score = -1.0
    meaning 'no SPOC score', which the classifier was trained on).
    """
    return _verdict_io_build_candidate_dict(row)


def _build_star_info(star_row: dict | None, tic_id: str) -> dict:
    """Build the star_info dict that predict_verdict expects.

    Delegates to exonit.verdict_io.build_star_info. NOTE the argument order
    here is (star_row, tic_id) — the reverse of verdict_io's own
    (tic_id, star_row) — kept for backward compatibility with this script's
    pre-existing call sites and the parity test.

    Includes tic_id so _augment_with_berger can look up the Berger catalog.
    n_datapoints comes from the stars table (same as server query).
    """
    return _verdict_io_build_star_info(tic_id, star_row)


# ---------------------------------------------------------------------------
# FFI/SPOC routing
# ---------------------------------------------------------------------------

def _load_exptime_map(tic_ids: list[str]) -> dict[str, float]:
    """Bulk-load {tic_id: exptime} for *tic_ids* from the shared metadata
    cache (SQLite or PG, whichever EXONIT_METADATA_DB_URL selects — same
    seam /api/metadata and server.py's _run_verdict_background use), via
    MetadataCache.get_exptime_map's chunked batch query. One read for the
    whole recompute batch, not one get_entry() call per row.
    """
    cfg = ExoNitConfig()
    mc = MetadataCache(cfg.metadata_cache_path)
    return mc.get_exptime_map(tic_ids)


# ---------------------------------------------------------------------------
# Classifier call
# ---------------------------------------------------------------------------

def _compute_verdict(candidate: dict, star_info: dict,
                     data_source: str) -> dict:
    """Call the production classifier. predict_verdict routes to v2.1 via env
    for SPOC candidates; FFI candidates additionally honour
    EXONIT_VERDICT_FFI_VERSION if the caller's environment sets it (see the
    module docstring — this script does not set it itself)."""
    from exonit.verdict_classifier import predict_verdict
    return predict_verdict(candidate, default_vetting_result(),
                           star_info=star_info, data_source=data_source)


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------

def _write_verdicts(dsn: str, sector: int,
                    updates: list[tuple[str, float | None, str, int]]) -> int:
    """UPDATE claude_verdict + verdict_proba for a batch of
    (new_verdict, proba, tic_id, cnum).

    Commits as a single transaction per batch. Returns rows actually updated.
    """
    import psycopg

    n_updated = 0
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for new_verdict, proba, tic_id, cnum in updates:
                cur.execute(
                    """
                    UPDATE candidates
                    SET claude_verdict = %s,
                        verdict_proba = %s
                    WHERE sector = %s
                      AND tic_id  = %s
                      AND candidate_num = %s
                    """,
                    (new_verdict, proba, sector, tic_id, cnum),
                )
                n_updated += cur.rowcount
        conn.commit()
    return n_updated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _transition_matrix(
        transitions: list[tuple[str | None, str]]) -> dict[tuple[str, str], int]:
    """Count (old_verdict, new_verdict) pairs. None old_verdict → 'NULL'."""
    matrix: dict[tuple[str, str], int] = defaultdict(int)
    for old, new in transitions:
        matrix[(old or "NULL", new)] += 1
    return dict(matrix)


def _print_matrix(matrix: dict[tuple[str, str], int]) -> None:
    print("\n=== Transition matrix (old -> new : count) ===")
    for (old, new), count in sorted(matrix.items()):
        print(f"  {old:30s} -> {new:25s} : {count}")


def _write_summary(path: Path, sector: int, transitions: list,
                   n_no_cand: int, n_no_star: int, dry_run: bool,
                   n_sub_threshold: int = 0) -> None:
    """Serialise summary to progress/recompute_verdict_s{sector}_summary.json."""
    matrix = _transition_matrix(transitions)
    matrix_str = {f"{k[0]}->{k[1]}": v for k, v in matrix.items()}

    n_new_pc = sum(1 for old, new in transitions
                   if old != "planet_candidate" and new == "planet_candidate")
    n_lost_pc = sum(1 for old, new in transitions
                    if old == "planet_candidate" and new != "planet_candidate")
    n_unchanged = sum(1 for old, new in transitions if old == new)

    summary = {
        "sector": sector,
        "dry_run": dry_run,
        "n_pairs_in_csv": len(transitions) + n_no_cand + n_sub_threshold,
        "n_processed": len(transitions),
        "n_sub_threshold": n_sub_threshold,
        "n_no_candidate_row": n_no_cand,
        "n_no_star_row": n_no_star,
        "n_unchanged": n_unchanged,
        "n_new_planet_candidate": n_new_pc,
        "n_lost_planet_candidate": n_lost_pc,
        "transition_matrix": matrix_str,
    }
    path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary written to %s", path)


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

def _write_backup(backup_path: str, sector: int,
                  pairs: list[tuple[str, int]],
                  cand_map: dict[tuple[str, int], dict]) -> None:
    """Dump current claude_verdict for the target set to a CSV.

    Rows absent from PG get an empty verdict cell so the backup is complete.
    """
    with open(backup_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sector", "tic_id", "candidate_num", "claude_verdict"])
        for tic_id, cnum in pairs:
            row = cand_map.get((tic_id, cnum))
            old = row["claude_verdict"] if row else ""
            writer.writerow([sector, tic_id, cnum, old])
    logger.info("Backup written to %s (%d rows)", backup_path, len(pairs))


# ---------------------------------------------------------------------------
# Main mode: --from-csv / --sector [--dry-run] [--backup]
# ---------------------------------------------------------------------------

def run_main(args) -> int:
    """Load CSV, recompute verdicts, optionally write back.

    --dry-run: compute + print matrix, no DB writes.
    --backup PATH: dump old verdicts before writing.
    """
    pairs = _load_csv(args.from_csv)
    sector = args.sector
    dry_run = args.dry_run
    logger.info("Loaded %d pairs from %s (sector %d, dry_run=%s)",
                len(pairs), args.from_csv, sector, dry_run)

    logger.info("Fetching candidate rows from PG ...")
    cand_map = _fetch_candidates(args.db, sector, pairs)
    logger.info("  Got %d candidate rows", len(cand_map))

    distinct_tic_ids = list({p[0] for p in pairs})
    logger.info("Fetching %d star rows from PG ...", len(distinct_tic_ids))
    star_map = _fetch_stars(args.db, distinct_tic_ids)
    logger.info("  Got %d star rows", len(star_map))

    logger.info("Fetching exptime map for FFI/SPOC routing ...")
    exptime_map = _load_exptime_map(distinct_tic_ids)
    logger.info("  Got %d exptime rows", len(exptime_map))

    if args.backup:
        _write_backup(args.backup, sector, pairs, cand_map)

    # Warm up the classifier (lru_cache) before the main loop.
    logger.info("Loading verdict classifier (v2.1) ...")
    from exonit.verdict_classifier import predict_verdict  # noqa: F401
    from exonit.verdict_classifier_v2_1 import _load_v2_1_model
    _load_v2_1_model()
    logger.info("  Classifier loaded.")

    transitions: list[tuple[str | None, str]] = []
    n_no_cand = 0
    n_no_star = 0
    n_sub_threshold = 0
    pending_updates: list[tuple[str, float | None, str, int]] = []
    n_written = 0
    t0 = time.time()

    for i, (tic_id, cnum) in enumerate(pairs, start=1):
        cand_row = cand_map.get((tic_id, cnum))
        if cand_row is None:
            n_no_cand += 1
            continue

        # Strong-tier gate — mirrors production server._run_verdict_background's
        # pre-gate (exonit/routes/workers.py's three verdict-trigger call sites,
        # via exonit.work_producers.verdict_eligible): snr >= 10.0 AND
        # (cnn_score >= 0.7 OR exominer_ffi_score >= 0.5) — OR-gate, 2026-07-11
        # (docs/reviews/injection_e2e_results_2026-07-11.md + docs/reviews/
        # ffi_cnn_investigation_2026-07-11.md §5). The LightGBM classifier has
        # no internal CNN/em_ffi hard-gate; skipping here prevents
        # sub-threshold candidates from receiving a claude_verdict and
        # entering the finalist pipeline. CAND_COLS already fetches
        # exominer_ffi_score (used elsewhere in candidate/star dict building),
        # so no extra SELECT is needed for this gate.
        if not verdict_eligible(cand_row.get("snr"), cand_row.get("cnn_score"),
                                 cand_row.get("exominer_ffi_score")):
            n_sub_threshold += 1
            continue

        star_row = star_map.get(tic_id)
        if star_row is None:
            n_no_star += 1

        candidate = _build_candidate_dict(cand_row)
        star_info = _build_star_info(star_row, tic_id)
        data_source = data_source_from_exptime(exptime_map.get(tic_id))
        result = _compute_verdict(candidate, star_info, data_source)

        new_verdict = result["verdict"]
        old_verdict = cand_row.get("claude_verdict")
        transitions.append((old_verdict, new_verdict))

        if not dry_run:
            pending_updates.append(
                (new_verdict, result.get("proba"), tic_id, cnum))

        # Flush batch
        if not dry_run and len(pending_updates) >= BATCH_SIZE:
            n_written += _write_verdicts(args.db, sector, pending_updates)
            pending_updates.clear()

        if i % 2000 == 0 or i == len(pairs):
            elapsed = time.time() - t0
            eta = elapsed / i * (len(pairs) - i) if i < len(pairs) else 0.0
            logger.info("[%d/%d] %.0f%% | eta %.0fs | no_cand=%d no_star=%d sub_thresh=%d",
                        i, len(pairs), 100 * i / len(pairs), eta,
                        n_no_cand, n_no_star, n_sub_threshold)

    # Flush remainder
    if not dry_run and pending_updates:
        n_written += _write_verdicts(args.db, sector, pending_updates)

    elapsed = time.time() - t0
    matrix = _transition_matrix(transitions)
    _print_matrix(matrix)

    if not dry_run:
        logger.info("Wrote %d verdict updates in %.1fs", n_written, elapsed)

    summary_path = ROOT / "progress" / f"recompute_verdict_s{sector}_summary.json"
    _write_summary(summary_path, sector, transitions, n_no_cand, n_no_star,
                   dry_run, n_sub_threshold)
    return 0


# ---------------------------------------------------------------------------
# Parity-check mode: --parity-check SECTOR N
# ---------------------------------------------------------------------------

def run_parity_check(args) -> int:
    """Pick N candidates from SECTOR. For each, call predict_verdict via the
    tool's helper path AND via a second independent direct call. Assert identical.

    This proves there is no accidental reimplementation drift in _build_candidate_dict,
    _build_star_info, or the env-var routing. Does NOT compare to stored verdicts
    (stored verdicts may be legitimately stale).
    """
    import psycopg

    sector = int(args.parity_check[0])
    n_check = int(args.parity_check[1])
    logger.info("=== Parity check: sector %d, N=%d ===", sector, n_check)

    # Fetch N candidates ordered by cnn_score desc (prefer confident rows).
    cols_sql = ", ".join(f"c.{c}" for c in CAND_COLS)
    with psycopg.connect(args.db) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT {cols_sql}
                FROM candidates c
                WHERE c.sector = %s
                  AND c.cnn_score IS NOT NULL
                ORDER BY c.cnn_score DESC
                LIMIT %s
            """, (sector, n_check))
            rows = cur.fetchall()

    if not rows:
        logger.error("No candidates found for sector %d", sector)
        return 1

    cand_rows = [dict(zip(CAND_COLS, r)) for r in rows]
    tic_ids = list({str(r["tic_id"]) for r in cand_rows})
    star_map = _fetch_stars(args.db, tic_ids)
    exptime_map = _load_exptime_map(tic_ids)
    logger.info("Checking %d candidates from %d TICs", len(cand_rows), len(tic_ids))

    # Warm up classifier once.
    from exonit.verdict_classifier import predict_verdict

    failures = []
    for row in cand_rows:
        tic_id = str(row["tic_id"])
        cnum = int(row["candidate_num"])

        candidate = _build_candidate_dict(row)
        star_row = star_map.get(tic_id)
        star_info = _build_star_info(star_row, tic_id)
        data_source = data_source_from_exptime(exptime_map.get(tic_id))

        # Path A: tool's helper path (what the script uses for writes)
        result_a = _compute_verdict(candidate, star_info, data_source)

        # Path B: independent direct call with copied dicts (no shared state).
        # default_vetting_result() returns a fresh {"passed": True} dict on
        # every call (not a cached singleton), so Path A and Path B still
        # share no state -- same guarantee the literal {"passed": True} gave.
        result_b = predict_verdict(dict(candidate), default_vetting_result(),
                                   star_info=dict(star_info),
                                   data_source=data_source)

        if result_a["verdict"] != result_b["verdict"]:
            failures.append({
                "tic_id": tic_id,
                "candidate_num": cnum,
                "path_a": result_a["verdict"],
                "path_b": result_b["verdict"],
            })

    n_checked = len(cand_rows)
    passed = len(failures) == 0
    summary_out = {
        "sector": sector,
        "n_requested": n_check,
        "n_checked": n_checked,
        "n_failures": len(failures),
        "passed": passed,
        "failures": failures[:20],
    }
    print(json.dumps(summary_out, indent=2))

    if passed:
        logger.info("PARITY CHECK PASSED: %d candidates, all identical", n_checked)
        return 0

    logger.error("PARITY CHECK FAILED: %d/%d verdicts differ between paths",
                 len(failures), n_checked)
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recompute claude_verdict for a directed set of candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", default=DEFAULT_DB_URL,
                   help="PostgreSQL DSN (or EXONIT_DATABASE_URL env var)")
    p.add_argument("--from-csv", metavar="PATH", default=None,
                   help="CSV with tic_id,candidate_num (+ any extra columns, ignored)")
    p.add_argument("--sector", type=int, default=1,
                   help="TESS sector for the candidate rows")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute verdicts and print transition matrix; write nothing")
    p.add_argument("--backup", metavar="PATH", default=None,
                   help="Dump old claude_verdict for the target set to this CSV "
                        "before writing (skipped on --dry-run if specified without a write)")
    p.add_argument("--parity-check", nargs=2, metavar=("SECTOR", "N"),
                   default=None,
                   help="Parity-check mode: pick N candidates from SECTOR and "
                        "verify tool path == direct predict_verdict call")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.parity_check is not None:
        return run_parity_check(args)

    if args.from_csv is None:
        print("ERROR: --from-csv is required (unless using --parity-check)")
        return 2

    return run_main(args)


if __name__ == "__main__":
    sys.exit(main())
