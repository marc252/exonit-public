#!/usr/bin/env python3
"""Index downloaded TGLC light curves into a TIC→file db.

TGLC files are named by Gaia DR3 id; the TIC lives in the FITS header (TICID).
This scans a TGLC cache dir, reads each header, and writes:
  * a tglc(tic_id, sector, path, exptime, tmag, ra, dec) index
    used by exonit.metadata_index.ensure_metadata to resolve TIC→file.

Incremental: re-running skips files already indexed (by absolute path) and only
processes new files found in the cache.  Files deleted from disk but still in
the DB are left alone — this script is not responsible for cleanup.

Backend: routes through exonit.metadata_cache.MetadataCache (the
docs/superpowers/specs/2026-07-20-tglc-index-to-pg.md seam) instead of a raw
sqlite3.connect — SQLite (default) unless EXONIT_METADATA_DB_URL is set to a
postgresql:// DSN, in which case every write in this run goes to PostgreSQL,
transparently (see MetadataCache.__init__).

Skip-set (D5 of the spec, "fix the hot write-path anti-pattern while
porting"): the pre-seam version of this script ran an UNSCOPED
``SELECT path FROM tglc`` (the entire table, tens of millions of rows) on
every 10k-file chunk — the single most expensive/most-repeated query
identified against the old 61 GiB SQLite file, and the dominant contributor to
its WAL bloat. This version queries only the CURRENT chunk's candidate paths
via ``MetadataCache.tglc_paths_for()`` (SQLite: chunked ``WHERE path IN
(...)``; PostgreSQL: one ``WHERE path = ANY(%s)``) — O(chunk size), not
O(table size). One consequence (flagged, per the spec): the
"index: N (tic,sector) rows" line below now reports the count of THIS
invocation's candidate files that are indexed (already-there + newly
inserted), not the whole table's row count — computing the latter cheaply is
no longer possible without the same full-scan this fix removes, and nothing
downstream depends on it being a global total (see progress/f3-indexer-pg-
handoff.md).

Usage:
  python3 scripts/index_tglc_batch.py --cache data/discovery/tglc_cache \
      --index data/metadata_cache.db
"""
import argparse
import glob
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from astropy.io import fits

from exonit.metadata_backend import is_pg_url
from exonit.metadata_cache import MetadataCache


def _exptime_for(sector, header):
    for k in ("TIMEDEL",):  # days per cadence
        if k in header and header[k]:
            return float(header[k]) * 86400.0
    if "EXPTIME" in header and header["EXPTIME"]:
        return float(header["EXPTIME"])
    s = int(sector)
    return 1800.0 if s <= 26 else (600.0 if s <= 55 else 200.0)


def read_one(path):
    """Read FITS headers for *path* and return a DB row tuple, or None on error.

    Opens the file once with fits.open() so both HDU 0 and HDU 1 are available
    without a second round-trip.  HDU 1 is only consulted for timing keywords
    when they are absent from HDU 0.
    """
    try:
        with fits.open(path, memmap=False) as hdul:
            h0 = hdul[0].header
            tic = h0.get("TICID")
            sector = h0.get("SECTOR")
            if tic is None or sector is None:
                return None
            # Build a timing dict from HDU 0; fall back to HDU 1 only if needed.
            timing = {}
            for k in ("TIMEDEL", "EXPTIME"):
                if k in h0 and h0[k]:
                    timing[k] = h0[k]
            if "TIMEDEL" not in timing and "EXPTIME" not in timing:
                # HDU 1 needed — check it for timing keywords.
                if len(hdul) > 1:
                    h1 = hdul[1].header
                    for k in ("TIMEDEL", "EXPTIME"):
                        if k in h1 and h1[k]:
                            timing[k] = h1[k]
            exptime = _exptime_for(sector, timing)
            return (
                str(int(tic)), int(sector), os.path.abspath(path), exptime,
                h0.get("TESSMAG"), h0.get("RA_OBJ"), h0.get("DEC_OBJ"),
            )
    except Exception:
        return None


def _default_workers() -> int:
    """Return the default worker count: 75% of CPU cores, at least 1.

    Per CLAUDE.md: never use 100% of a machine's CPUs; leave ~25% free so the
    machine stays responsive.  os.cpu_count() can return None on exotic
    platforms — fall back to 1 in that case.
    """
    ncpu = os.cpu_count() or 1
    return max(1, int(ncpu * 0.75))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--files-from", default=None,
                    help="newline-separated list of FITS paths to index. When "
                         "given, skips the full --cache glob (which is O(all "
                         "files on disk) and slow once the cache holds millions) "
                         "and indexes only these — used by discovery_download "
                         "for a just-downloaded batch.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Number of parallel processes for FITS header parsing "
                         "(default: 75%% of CPU cores, at least 1)")
    args = ap.parse_args()

    # Resolve worker count: explicit --workers overrides the CPU-fraction default.
    n_workers = args.workers if args.workers is not None else _default_workers()

    # --- A. Open the index via the seam and load already-indexed paths ---
    # EXONIT_METADATA_DB_URL set -> PostgreSQL; unset -> SQLite at args.index,
    # byte-for-byte the historical behaviour (see MetadataCache.__init__).
    is_pg = is_pg_url(os.environ.get("EXONIT_METADATA_DB_URL"))
    cache = MetadataCache(args.index)
    cache.ensure_tglc_table()  # no-op on PG (schema_pg_meta.sql provisions it)
    if not is_pg:
        # busy_timeout: the index lives next to a production server that reads
        # it (metadata lookups) and a discovery run can hold it for long
        # stretches. SQLite's default (MetadataCache's own 30s) can still lose
        # to a long-lived reader -> "database is locked" killed a 681k-file
        # download at chunk 11 on 2026-07-18 while a SCANNING run was live.
        # 300s means "wait for your turn" instead of dying; combined with the
        # retry below it makes the indexer robust to coexisting with runs.
        cache.set_sqlite_busy_timeout(300_000)

    # --- Enumerate candidate FITS: explicit batch list, or full cache glob ---
    if args.files_from:
        with open(args.files_from, encoding="utf-8") as fh:
            files = [ln.strip() for ln in fh if ln.strip()]
    else:
        files = glob.glob(os.path.join(args.cache, "**", "*.fits"), recursive=True)

    # Chunk-scoped skip-set (D5): query only THIS invocation's candidate
    # paths, not the whole table — see the module docstring.
    abs_paths = [os.path.abspath(f) for f in files]
    existing_paths = cache.tglc_paths_for(abs_paths)
    new_files = [f for f, ap in zip(files, abs_paths) if ap not in existing_paths]
    n_skip = len(files) - len(new_files)
    if n_skip:
        print(
            f"scanning {len(files)} TGLC fits in cache, "
            f"{len(new_files)} new (skipping {n_skip} already indexed)"
        )
    else:
        print(f"scanning {len(files)} TGLC fits in cache, {len(new_files)} new")

    # --- B. Parse new files in parallel (ProcessPool: read_one is CPU Python) ---
    # Threads were the original approach but fits.open + header parsing is pure
    # Python CPU work, so the GIL limits throughput to ~1 core.  ProcessPool
    # bypasses the GIL at the cost of pickle round-trips, which are negligible
    # for the small tuples read_one returns.
    #
    # chunksize: sending one task per worker call (chunksize=1) for 100k items
    # creates massive IPC overhead.  A larger chunk amortises the round-trip.
    # Rule of thumb: len//(workers*8) gives ~8 batches per worker, keeping
    # the pool busy while not over-buffering results.
    rows = []
    if new_files:
        chunksize = max(1, len(new_files) // (n_workers * 8))
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                parsed = 0
                for r in pool.map(read_one, new_files, chunksize=chunksize):
                    if r is not None:
                        rows.append(r)
                    parsed += 1
                    if parsed % 5000 == 0:
                        print(f"  parsed {parsed}/{len(new_files)} …", flush=True)
        except Exception as exc:
            # ProcessPoolExecutor failed to start or map failed fatally.
            # This is not a graceful degradation: surface the error clearly
            # so the operator knows something is wrong (per CLAUDE.md).
            raise RuntimeError(
                f"ProcessPoolExecutor failed with {n_workers} workers: {exc}"
            ) from exc
    print(f"parsed {len(rows)} headers")

    if rows:
        # PostgreSQL: ensure every sector in this chunk has a partition before
        # the bulk upsert (D3's ensure_tglc_partition — a brand-new sector's
        # first indexed row would otherwise hit "no partition of relation
        # tglc found for row"). No-op on SQLite (one flat table).
        if is_pg:
            for sec in sorted({r[1] for r in rows}):
                cache.ensure_tglc_partition(sec)
            cache.tglc_add_rows_bulk(rows)
        else:
            # Retry on top of busy_timeout: even a 300s wait can lose to a
            # reader that re-acquires in a loop. 3 attempts with a pause
            # covers the pathological case; parsing (the expensive part) is
            # never redone. SQLite-only: "database is locked" has no PG
            # equivalent under MVCC.
            for attempt in range(3):
                try:
                    cache.tglc_add_rows_bulk(rows)
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() or attempt == 2:
                        raise
                    print(f"index write locked (attempt {attempt + 1}/3), retrying in 60s …",
                          flush=True)
                    time.sleep(60)

    # Report this invocation's indexed-file count (already-there + newly
    # inserted among THIS run's candidates) — see the module docstring for
    # why this is no longer a whole-table total (D5's chunk-scoped fix).
    total_rows = len(existing_paths) + len(rows)
    print(f"index: {total_rows} (tic,sector) rows")

    # Emit files_total so callers can use it for registry updates without a
    # second disk walk.  This is the same set glob already enumerated above.
    print(f"files_total={len(files)}")

    cache.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
