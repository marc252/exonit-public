#!/usr/bin/env python3
"""ExoNit — CLI entry point for transit detection pipeline.

Usage:
    python run_nightly.py run                     # analyze next 10 TOI targets
    python run_nightly.py run --stars 20          # analyze 20 targets
    python run_nightly.py run --target TIC123456  # analyze a specific star
    python run_nightly.py download                # download/update TOI catalog
    python run_nightly.py stats                   # show database statistics
"""

import argparse
import builtins
import os
import sys

# Patch print() globally to survive closed stdout/stderr (happens when
# the dashboard's Node.js process closes the pipe before Python exits).
_builtin_print = builtins.print

def _safe_print(*args, **kwargs):
    try:
        _builtin_print(*args, **kwargs)
    except ValueError:
        pass

builtins.print = _safe_print

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env file (ANTHROPIC_API_KEY, etc.)
from dotenv import load_dotenv
load_dotenv()

from exonit.db import ExoNitDB
from exonit.db_backend import is_pg_url
from exonit.config import config
from exonit.targets import download_toi_catalog, parse_toi_catalog, get_next_targets
from exonit.enqueue import enqueue_targets

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
# DB_PATH may be a SQLite file path OR a postgresql:// DSN, resolved by
# config.db_path (which honours EXONIT_DATABASE_URL, then EXONIT_DB_PATH, then
# the default file). ExoNitDB/make_backend discriminate the two via is_pg_url,
# so call-sites that do ExoNitDB(DB_PATH) work for either backend unchanged.
# (The historical hardcoded data/exonit.db ignored both env vars — bug fixed.)
DB_PATH = config.db_path
CATALOG_PATH = os.path.join(DATA_DIR, "toi_catalog.csv")


def cmd_run(args):
    """Run the analysis pipeline — enqueue targets for workers."""
    os.makedirs(DATA_DIR, exist_ok=True)

    db = ExoNitDB(DB_PATH)

    if args.target:
        targets = [{"tic_id": args.target}]
        if os.path.exists(CATALOG_PATH):
            all_targets = parse_toi_catalog(CATALOG_PATH)
            match = [t for t in all_targets if t["tic_id"] == args.target]
            if match:
                targets = match
        result = enqueue_targets(targets, db)
        if result["enqueued"] > 0:
            print(f"\nEnqueued TIC {args.target} — workers will process it.")
        else:
            print(f"\nCould not enqueue TIC {args.target} (no FITS in cache or already analyzed)")
    else:
        if not os.path.exists(CATALOG_PATH):
            print("TOI catalog not found. Downloading...")
            download_toi_catalog(CATALOG_PATH)

        targets = parse_toi_catalog(CATALOG_PATH)
        next_targets = get_next_targets(targets, db, max_targets=args.stars)

        if not next_targets:
            print("No pending targets. All TOIs have been analyzed!")
            db.close()
            return

        print(f"\nSelected {len(next_targets)} targets (brightest unanalyzed TOIs)")
        enqueue_targets(next_targets, db)

    db.close()


def cmd_download(args):
    """Download/update the TOI catalog."""
    os.makedirs(DATA_DIR, exist_ok=True)
    download_toi_catalog(CATALOG_PATH)


def cmd_stats(args):
    """Show database statistics."""
    # A PG DSN is not a filesystem path: skip the file-existence check for it.
    if not is_pg_url(DB_PATH) and not os.path.exists(DB_PATH):
        print("No database found. Run an analysis first.")
        return

    db = ExoNitDB(DB_PATH)
    stats = db.get_stats()
    candidates = db.get_all_candidates(min_snr=7.0)

    print(f"\nExoNit Statistics")
    print(f"{'='*40}")
    print(f"  Stars analyzed: {stats['total_stars']}")
    print(f"  Candidates found: {stats['total_candidates']}")

    if candidates:
        print(f"\n  Top candidates (SNR >= 7):")
        for c in candidates[:10]:
            print(f"    TIC {c['tic_id']} #{c['candidate_num']}: "
                  f"P={c['period_days']:.4f}d, SNR={c['snr']:.1f}, "
                  f"depth={c['depth_ppm']:.0f}ppm")
    print(f"{'='*40}")
    db.close()


def cmd_retry_failed(args):
    """Retry previously failed stars using metadata cache."""
    db = ExoNitDB(DB_PATH)

    failed = db._query("SELECT tic_id FROM stars WHERE status = 'failed'")
    if not failed:
        print("No failed stars to retry.")
        db.close()
        return

    print(f"Found {len(failed)} failed stars. Checking metadata cache...")

    from exonit.metadata_cache import MetadataCache
    cache_path = os.path.join(DATA_DIR, "metadata_cache.db")
    if not os.path.exists(cache_path):
        print("No metadata cache found. Run 'build-cache' first.")
        db.close()
        return

    cache = MetadataCache(cache_path)
    retryable = []
    for row in failed:
        if cache.get_entry(row["tic_id"]):
            retryable.append(row["tic_id"])
    cache.close()

    print(f"  {len(retryable)} have cached metadata (can retry)")
    print(f"  {len(failed) - len(retryable)} truly missing data")

    if not retryable:
        db.close()
        return

    max_retry = args.max or len(retryable)
    to_retry = retryable[:max_retry]

    # Delete failed entries so they get re-processed
    for tic_id in to_retry:
        db._exec("DELETE FROM stars WHERE tic_id = ? AND status = 'failed'", (tic_id,))
    print(f"  Reset {len(to_retry)} stars for retry")

    targets = [{"tic_id": tid, "ra": 0, "dec": 0, "tmag": 0} for tid in to_retry]

    result = enqueue_targets(targets, db)
    print(f"\nRe-enqueued {result['enqueued']} stars for workers")
    db.close()


def cmd_reprocess(args):
    """Re-process stars from old DB using new pipeline into a fresh DB.

    Reads TIC IDs from the original DB, creates a new DB, and runs
    the full pipeline with the improved preprocessing (flatten + asymmetric
    outlier removal + fill_gaps + smart normalize).
    """
    old_db_path = args.old_db or DB_PATH
    new_db_path = args.new_db or os.path.join(DATA_DIR, "exonit_v2.db")

    if not os.path.exists(old_db_path):
        print(f"Old DB not found: {old_db_path}")
        return

    # Read TIC IDs from old DB
    import sqlite3
    old_conn = sqlite3.connect(old_db_path)
    c = old_conn.cursor()

    if args.include_failed:
        c.execute("SELECT tic_id FROM stars")
    else:
        c.execute("SELECT tic_id FROM stars WHERE status = 'completed'")
    old_tics = [r[0] for r in c.fetchall()]
    old_conn.close()

    print(f"Old DB: {len(old_tics)} stars to re-process")

    # Filter to only stars with FITS in metadata cache
    from exonit.metadata_cache import MetadataCache
    cache_path = os.path.join(DATA_DIR, "metadata_cache.db")
    cache = MetadataCache(cache_path) if os.path.exists(cache_path) else None
    reprocessable = []
    no_cache = []
    for tic in old_tics:
        if cache and cache.get_entry(tic):
            reprocessable.append(tic)
        else:
            no_cache.append(tic)
    if cache:
        cache.close()

    print(f"  In cache: {len(reprocessable)}")
    print(f"  No cache (skipping): {len(no_cache)}")

    if not reprocessable:
        print("No stars to re-process!")
        return

    # Limit if requested
    if args.max:
        reprocessable = reprocessable[:args.max]
        print(f"  Limited to: {len(reprocessable)}")

    # Create fresh DB
    os.makedirs(os.path.dirname(new_db_path), exist_ok=True)
    if os.path.exists(new_db_path) and not args.resume:
        os.remove(new_db_path)
        print(f"  Created fresh DB: {new_db_path}")
    elif os.path.exists(new_db_path) and args.resume:
        print(f"  Resuming into existing DB: {new_db_path}")
    else:
        print(f"  New DB: {new_db_path}")

    db = ExoNitDB(new_db_path)

    # Look up target info from catalog
    targets = []
    catalog_targets = {}
    if os.path.exists(CATALOG_PATH):
        from exonit.targets import parse_toi_catalog
        for t in parse_toi_catalog(CATALOG_PATH):
            catalog_targets[t["tic_id"]] = t

    for tic in reprocessable:
        if tic in catalog_targets:
            targets.append(catalog_targets[tic])
        else:
            targets.append({"tic_id": tic, "ra": 0, "dec": 0, "tmag": 0})

    print(f"\nEnqueuing {len(targets)} stars for re-processing...")

    result = enqueue_targets(targets, db)

    print(f"\n{'='*60}")
    print(f"RE-PROCESS ENQUEUED ({result['enqueued']} tasks) — workers will process them")
    print(f"  New DB: {new_db_path}")
    print(f"  Run 'python run_nightly.py compare' when workers finish to see results")
    print(f"{'='*60}")

    db.close()


def cmd_compare(args):
    """Compare old vs new DB to measure pipeline improvement."""
    old_db_path = args.old_db or DB_PATH
    new_db_path = args.new_db or os.path.join(DATA_DIR, "exonit_v2.db")

    if not os.path.exists(old_db_path) or not os.path.exists(new_db_path):
        print(f"Need both DBs: {old_db_path} and {new_db_path}")
        return

    import sqlite3

    def _load_db(path):
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT tic_id, status FROM stars")
        stars = {r[0]: r[1] for r in c.fetchall()}
        c.execute("SELECT * FROM candidates")
        candidates = [dict(r) for r in c.fetchall()]
        conn.close()
        return stars, candidates

    old_stars, old_cands = _load_db(old_db_path)
    new_stars, new_cands = _load_db(new_db_path)

    # Load TOI catalog for ground truth
    known_planets = {}  # tic_id -> disposition
    if os.path.exists(CATALOG_PATH):
        import csv
        with open(CATALOG_PATH) as f:
            reader = csv.DictReader(f)
            for row in reader:
                tic = row.get("TIC ID", "").strip()
                disp = row.get("TFOPWG Disposition", "").strip()
                period = row.get("Period (days)", "")
                if tic and disp:
                    if tic not in known_planets:
                        known_planets[tic] = []
                    known_planets[tic].append({
                        "disposition": disp,
                        "period": float(period) if period else None,
                    })

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPARISON: OLD vs NEW")
    print(f"{'='*60}")

    # Basic stats
    old_completed = sum(1 for s in old_stars.values() if s == 'completed')
    new_completed = sum(1 for s in new_stars.values() if s == 'completed')
    print(f"\n  Stars completed:  OLD={old_completed}  NEW={new_completed}")
    print(f"  Candidates found: OLD={len(old_cands)}  NEW={len(new_cands)}")

    # Group candidates by TIC
    def _by_tic(cands):
        d = {}
        for c in cands:
            d.setdefault(c["tic_id"], []).append(c)
        return d

    old_by_tic = _by_tic(old_cands)
    new_by_tic = _by_tic(new_cands)

    # Recall for known planets (KP = Known Planet, CP = Confirmed Planet)
    kp_cp_tics = set()
    pc_tics = set()
    for tic, planets in known_planets.items():
        for p in planets:
            if p["disposition"] in ("KP", "CP"):
                kp_cp_tics.add(tic)
            elif p["disposition"] == "PC":
                pc_tics.add(tic)

    # Check which KP/CP were detected (have candidates with SNR >= 7)
    def _detected(by_tic, tic_set, min_snr=7.0):
        detected = set()
        for tic in tic_set:
            if tic in by_tic:
                if any(c["snr"] >= min_snr for c in by_tic[tic]):
                    detected.add(tic)
        return detected

    # Only count KP/CP that were actually processed in both DBs
    common_completed = set(t for t, s in old_stars.items() if s == 'completed') & \
                       set(t for t, s in new_stars.items() if s == 'completed')

    kp_in_both = kp_cp_tics & common_completed
    pc_in_both = pc_tics & common_completed

    old_kp_detected = _detected(old_by_tic, kp_in_both)
    new_kp_detected = _detected(new_by_tic, kp_in_both)
    old_pc_detected = _detected(old_by_tic, pc_in_both)
    new_pc_detected = _detected(new_by_tic, pc_in_both)

    print(f"\n  Known Planets (KP/CP) in both DBs: {len(kp_in_both)}")
    if kp_in_both:
        print(f"    OLD detected: {len(old_kp_detected)}/{len(kp_in_both)} "
              f"({100*len(old_kp_detected)/len(kp_in_both):.0f}%)")
        print(f"    NEW detected: {len(new_kp_detected)}/{len(kp_in_both)} "
              f"({100*len(new_kp_detected)/len(kp_in_both):.0f}%)")

    print(f"\n  Planet Candidates (PC) in both DBs: {len(pc_in_both)}")
    if pc_in_both:
        print(f"    OLD detected: {len(old_pc_detected)}/{len(pc_in_both)} "
              f"({100*len(old_pc_detected)/len(pc_in_both):.0f}%)")
        print(f"    NEW detected: {len(new_pc_detected)}/{len(pc_in_both)} "
              f"({100*len(new_pc_detected)/len(pc_in_both):.0f}%)")

    # New detections (found in new but not in old)
    new_kp_gains = new_kp_detected - old_kp_detected
    new_kp_losses = old_kp_detected - new_kp_detected

    if new_kp_gains:
        print(f"\n  NEW KP/CP detections (gained): {len(new_kp_gains)}")
        for tic in sorted(new_kp_gains):
            cands = new_by_tic.get(tic, [])
            best = max(cands, key=lambda c: c["snr"]) if cands else None
            if best:
                print(f"    TIC {tic}: SNR={best['snr']:.1f}, P={best['period_days']:.4f}d")

    if new_kp_losses:
        print(f"\n  KP/CP LOST (regression!): {len(new_kp_losses)}")
        for tic in sorted(new_kp_losses):
            print(f"    TIC {tic}")

    # Stars that now succeed (were failed, now completed)
    recovered = set(t for t, s in new_stars.items() if s == 'completed') - \
                set(t for t, s in old_stars.items() if s == 'completed')
    if recovered:
        print(f"\n  Recovered from failed: {len(recovered)} stars now complete")

    # False positive rate estimate
    old_fp_count = sum(1 for c in old_cands if c.get("claude_verdict") == "likely_false_positive")
    new_fp_count = sum(1 for c in new_cands if c.get("claude_verdict") == "likely_false_positive")
    print(f"\n  Claude FP verdicts: OLD={old_fp_count}  NEW={new_fp_count}")

    print(f"\n{'='*60}")


def cmd_backfill(args):
    """Backfill TRICERATOPS FPP for strong candidates (informational).

    El backfill via Claude API es va eliminar el 2026-07-02: el substitueix el
    verdict classifier local del servidor (server._run_verdict_background).
    """
    from exonit.db import ExoNitDB

    db = ExoNitDB(DB_PATH)

    min_snr = args.min_snr
    min_cnn = args.min_cnn

    candidates = db.get_candidates_needing_analysis(
        min_snr=min_snr, min_cnn=min_cnn)

    print(f"ExoNit Backfill — {len(candidates)} candidates to process")
    print(f"  Criteria: SNR >= {min_snr}, CNN >= {min_cnn}")
    print(f"  TRICERATOPS: {'skip' if args.no_triceratops else 'enabled'}")
    print(f"{'='*60}")

    if not candidates:
        print("  No candidates need processing.")
        return

    if args.max:
        candidates = candidates[:args.max]
        print(f"  Limited to {args.max} candidates")

    # TRICERATOPS backfill — delegated to workers via /api/worker/fpp_task
    if not args.no_triceratops:
        fpp_todo = [
            c for c in candidates
            if c["triceratops_fpp"] is None
            or (
                c["triceratops_fpp"] is not None
                and c["triceratops_fpp"] >= 0
                and c["triceratops_fpp"] < 0.5
                and c.get("triceratops_nfpp") is None
            )
        ]
        print(f"\n  TRICERATOPS: {len(fpp_todo)} candidates without FPP/NFPP")
        if fpp_todo:
            print(f"  Workers will pick up FPP tasks via /api/worker/fpp_task")
            print(f"  Monitor progress at http://localhost:8400/dashboard")

    # Summary
    stats = db.get_stats()
    fpp_count = db._query_scalar("SELECT COUNT(*) FROM candidates WHERE triceratops_fpp IS NOT NULL")
    claude_count = db._query_scalar("SELECT COUNT(*) FROM candidates WHERE claude_verdict IS NOT NULL")
    print(f"\n{'='*60}")
    print(f"  Total candidates: {stats['total_candidates']}")
    print(f"  With TRICERATOPS FPP: {fpp_count}")
    print(f"  With Claude verdict: {claude_count}")


def cmd_serve(args):
    """Run the FastAPI dashboard server."""
    from exonit.server import main as serve_main
    serve_main(workers=args.workers)


def cmd_build_cache(args):
    """Build metadata cache from NFS FITS files."""
    from exonit.metadata_cache import MetadataCache, build_from_nfs
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, "metadata_cache.db")
    nfs = args.nfs_path

    print(f"Building metadata cache from {nfs}...")
    cache = MetadataCache(cache_path)
    stats = build_from_nfs(cache, nfs)
    print(f"  Stars: {stats['stars_added']}")
    print(f"  FITS parsed: {stats['fits_parsed']}")
    total = cache.stats()
    print(f"  Cache total: {total['total_stars']} stars, {total['total_fits_files']} files")
    cache.close()
    print(f"  Saved to {cache_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build and return the full ArgumentParser for run_nightly.py.

    Extracted from main() so that tests can import and exercise the real
    parser without executing the full script (which would parse sys.argv
    and call handlers).
    """
    parser = argparse.ArgumentParser(
        description="ExoNit — Automated exoplanet transit detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run transit detection pipeline")
    run_parser.add_argument("--stars", type=int, default=10,
                            help="Number of stars to analyze (default: 10)")
    run_parser.add_argument("--target", type=str, default=None,
                            help="Analyze a specific TIC ID")
    # (--sectors/--no-injection/--no-claude/--no-hub removed 2026-06-09:
    # they were parsed but never reached the workers — enqueue_targets has no
    # config parameter. Stage toggles belong in worker config, not here.)
    run_parser.set_defaults(func=cmd_run)

    dl_parser = sub.add_parser("download", help="Download/update TOI catalog")
    dl_parser.set_defaults(func=cmd_download)

    stats_parser = sub.add_parser("stats", help="Show database statistics")
    stats_parser.set_defaults(func=cmd_stats)

    retry_parser = sub.add_parser("retry-failed", help="Retry failed stars using metadata cache")
    retry_parser.add_argument("--max", type=int, default=None, help="Max stars to retry")
    retry_parser.set_defaults(func=cmd_retry_failed)

    serve_parser = sub.add_parser("serve", help="Run dashboard server (FastAPI)")
    serve_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes (>1 requires Redis, A3 feature; default: 1)",
    )
    serve_parser.set_defaults(func=cmd_serve)

    cache_parser = sub.add_parser("build-cache", help="Build metadata cache from NFS FITS files")
    cache_parser.add_argument("--nfs-path", default=config.nfs_cache_path,
                              help="Path to NFS FITS cache (default: EXONIT_NFS_CACHE "
                                   "or <project>/data/tess_cache)")
    cache_parser.set_defaults(func=cmd_build_cache)

    reproc_parser = sub.add_parser("reprocess",
                                    help="Re-process old stars with improved pipeline into new DB")
    reproc_parser.add_argument("--old-db", default=None,
                                help=f"Old DB path (default: {DB_PATH})")
    reproc_parser.add_argument("--new-db", default=None,
                                help="New DB path (default: data/exonit_v2.db)")
    reproc_parser.add_argument("--include-failed", action="store_true",
                                help="Also re-process previously failed stars")
    reproc_parser.add_argument("--resume", action="store_true",
                                help="Resume into existing new DB (skip already analyzed)")
    reproc_parser.add_argument("--max", type=int, default=None,
                                help="Max stars to re-process")
    reproc_parser.set_defaults(func=cmd_reprocess)

    bf_parser = sub.add_parser("backfill",
                                help="Backfill TRICERATOPS FPP for strong candidates")
    bf_parser.add_argument("--min-snr", type=float, default=10.0,
                           help="Min SNR threshold (default: 10.0)")
    bf_parser.add_argument("--min-cnn", type=float, default=0.7,
                           help="Min CNN score threshold (default: 0.7)")
    bf_parser.add_argument("--max", type=int, default=None,
                           help="Max candidates to process")
    bf_parser.add_argument("--no-triceratops", action="store_true",
                           help="Skip TRICERATOPS FPP")
    bf_parser.set_defaults(func=cmd_backfill)

    cmp_parser = sub.add_parser("compare", help="Compare old vs new pipeline results")
    cmp_parser.add_argument("--old-db", default=None,
                             help=f"Old DB path (default: {DB_PATH})")
    cmp_parser.add_argument("--new-db", default=None,
                             help="New DB path (default: data/exonit_v2.db)")
    cmp_parser.set_defaults(func=cmd_compare)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        args.command = "run"
        args.stars = 10
        args.target = None
        cmd_run(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
