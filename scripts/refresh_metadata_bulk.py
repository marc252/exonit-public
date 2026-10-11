#!/usr/bin/env python3
"""CLI wrapper around ``exonit.metadata_index.ensure_metadata_bulk``, for
validation and one-off backfill OUTSIDE the enqueue hot path.

Implements §2.5 of docs/superpowers/specs/2026-09-26-metadata-refresh-and-
bulk-enqueue-design.md. operator decision (§8, 2026-09-26): NO mass backfill —
the refresh is lazy, at enqueue time, gated by ``EXONIT_METADATA_REFRESH``.
This CLI exists for:

  (i)  pre-deploy validation against a real lot's CSV or a sector's TIC list
       (``--dry-run``, the default — writes nothing);
  (ii) an OPTIONAL, manually-triggered one-off refresh/backfill, if the operator ever
       decides one is worth it later (``--apply``);
  (iii) the sample staleness report from spec §1.3 (``--report``, PostgreSQL
       metadata backend only).

Never a cron, never wired into any automated pipeline.

Usage::

    # Preview a real lot CSV (no writes):
    python scripts/refresh_metadata_bulk.py --csv data/discovery/lot.csv

    # Preview a sector's tglc TIC list, capped:
    python scripts/refresh_metadata_bulk.py --sector 26 --limit 20000

    # Actually write (create + optionally refresh):
    python scripts/refresh_metadata_bulk.py --csv data/discovery/lot.csv \\
        --refresh --apply

    # Sample staleness report (PostgreSQL metadata backend only):
    python scripts/refresh_metadata_bulk.py --report

NEVER pointed at the production ``exonit_meta`` database from an automated
context — this is a by-hand tool. Even ``--dry-run`` opens a real connection
and issues real ``SELECT``s (no DDL, no writes), so double-check
``EXONIT_METADATA_DB_URL`` / ``--metadata-cache`` before running against
anything but a throwaway/test target.
"""
import argparse
import csv as csv_mod
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exonit.config import ExoNitConfig  # noqa: E402
from exonit.metadata_cache import MetadataCache  # noqa: E402
from exonit.metadata_index import ensure_metadata_bulk  # noqa: E402


def _load_tic_ids_from_csv(csv_path):
    """Read a lot CSV's ``tic_id`` column — same contract as
    ``scripts/enqueue_from_csv.py::load_targets``, but without the
    pandas dependency (this CLI is meant to run from any venv)."""
    tics = []
    with open(csv_path, newline="") as f:
        reader = csv_mod.DictReader(f)
        if not reader.fieldnames or "tic_id" not in reader.fieldnames:
            raise SystemExit(f"{csv_path}: no 'tic_id' column")
        for row in reader:
            v = (row.get("tic_id") or "").strip()
            if v and v.lower() != "nan":
                tics.append(str(int(float(v))))
    return tics


def _load_tic_ids_from_sector(mc, sector, limit=None):
    """Every TIC in ``tglc`` for *sector*, in the table's canonical order
    (``MetadataCache.tglc_tic_ids`` — insertion order on SQLite, ``seq`` on
    PostgreSQL), optionally capped at *limit*."""
    ids = mc.tglc_tic_ids(sector=sector)
    if limit is not None:
        ids = ids[:limit]
    return [str(t) for t in ids]


def run_report(mc, sample_pct: float) -> int:
    rows = mc.sector_staleness_report(sample_pct)
    if rows is None:
        print("--report needs a PostgreSQL metadata backend "
             "(EXONIT_METADATA_DB_URL) -- TABLESAMPLE/jsonb are PG-only.")
        return 1
    print(f"Sample: TABLESAMPLE SYSTEM ({sample_pct}) of `metadata`")
    print(f"{'sector':>6}  {'tglc_rows':>10}  {'missing':>10}  {'%missing':>9}")
    for r in rows:
        tglc_rows = r["tglc_rows"] or 0
        missing = r["missing"] or 0
        pct = (100.0 * missing / tglc_rows) if tglc_rows else 0.0
        print(f"{r['sector']:>6}  {tglc_rows:>10,}  {missing:>10,}  {pct:8.1f}%")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Bulk ensure_metadata over a lot's CSV or a sector's "
                    "tglc TIC list -- validation/backfill CLI, never a cron "
                    "(refresh stays lazy, at enqueue -- spec 2026-09-26 §8).")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--csv", default=None,
                     help="Lot CSV with a tic_id column")
    src.add_argument("--sector", type=int, default=None,
                     help="Pull every TIC in `tglc` for this sector")
    src.add_argument("--report", action="store_true",
                     help="Run the §1.3 sample staleness report and exit "
                          "(no --csv/--sector/--apply needed; PostgreSQL "
                          "metadata backend only).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of TICs (only with --sector)")
    p.add_argument("--sample-pct", type=float, default=0.1,
                   help="TABLESAMPLE SYSTEM percentage for --report "
                        "(default 0.1, matching the spec's measurement)")
    p.add_argument("--refresh", action="store_true",
                   help="Refresh stale TGLC-sourced rows (mirrors "
                        "EXONIT_METADATA_REFRESH). Default off: create + "
                        "no_fits only, existing rows left untouched.")
    p.add_argument("--apply", action="store_true",
                   help="Actually write. Default is --dry-run (no writes).")
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--metadata-cache", default=None,
                   help="Override the metadata cache path/DSN (default: "
                        "ExoNitConfig().metadata_cache_path, honouring "
                        "EXONIT_METADATA_DB_URL/EXONIT_METADATA_CACHE)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    cache_path = args.metadata_cache or ExoNitConfig().metadata_cache_path
    mc = MetadataCache(cache_path)
    try:
        if args.report:
            return run_report(mc, args.sample_pct)

        if args.csv:
            tic_ids = _load_tic_ids_from_csv(args.csv)
            print(f"Loaded {len(tic_ids):,} TICs from {args.csv}")
        elif args.sector is not None:
            tic_ids = _load_tic_ids_from_sector(mc, args.sector, args.limit)
            suffix = f" (limit {args.limit:,})" if args.limit else ""
            print(f"Loaded {len(tic_ids):,} TICs from tglc sector "
                  f"{args.sector}{suffix}")
        else:
            print("Must pass --csv, --sector, or --report", file=sys.stderr)
            return 2

        dry_run = not args.apply
        print(f"refresh_metadata_bulk: {len(tic_ids):,} TICs, "
              f"refresh={args.refresh}, dry_run={dry_run}, "
              f"chunk_size={args.chunk_size}, metadata_cache={cache_path!r}")

        result = ensure_metadata_bulk(
            tic_ids, mc, refresh=args.refresh, chunk_size=args.chunk_size,
            dry_run=dry_run)

        print(f"ready={len(result.ready):,} no_fits={len(result.no_fits):,} "
              f"created={result.created:,} refreshed={result.refreshed:,} "
              f"unchanged={result.unchanged:,} "
              f"preserved_non_tglc={result.preserved_non_tglc:,}")
        if dry_run:
            print("[dry-run] no writes performed -- pass --apply to write.")
        return 0
    finally:
        mc.close()


if __name__ == "__main__":
    sys.exit(main())
