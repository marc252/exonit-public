#!/usr/bin/env python3
"""Pre-download all TOI catalog light curves to NFS cache.

Two-phase approach:
  Phase 1: Use lightkurve to resolve TIC IDs → MAST download URLs (metadata only)
  Phase 2: Use aria2c to download all FITS files (robust, parallel, resume-capable)

This avoids lightkurve/astropy download issues entirely.
"""

import os
import json
import time
import warnings
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import subprocess
import sys

warnings.filterwarnings('ignore')

# Make `exonit` importable when run as `python scripts/predownload_toi.py` (scripts/ is sys.path[0]).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exonit.config import config as _cfg  # noqa: E402

# SPOC FITS cache root: EXONIT_NFS_CACHE (see exonit/config.py).
CACHE_DIR = _cfg.nfs_cache_path
MAST_BASE = "https://mast.stsci.edu/api/v0.1/Download/file?uri="


class RateLimiter:
    """Token bucket rate limiter."""
    def __init__(self, rate_per_second):
        self.rate = rate_per_second
        self.lock = threading.Lock()
        self.last = time.monotonic()
        self.tokens = rate_per_second

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens += (now - self.last) * self.rate
                self.last = now
                if self.tokens > self.rate:
                    self.tokens = self.rate
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
            time.sleep(0.05)


def resolve_urls(tic_id, max_sectors=12, limiter=None):
    """Resolve TIC ID to list of (url, local_path) for FITS files.

    Uses lightkurve search (metadata only, no download).
    """
    if limiter:
        limiter.acquire()
    try:
        import lightkurve as lk
        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
        if search is None or len(search) == 0:
            return tic_id, []

        search_120s = search[search.exptime.value == 120]
        if len(search_120s) == 0:
            search_120s = search

        n = min(max_sectors, len(search_120s))
        urls = []
        for row in search_120s.table[:n]:
            uri = row['dataURI']
            filename = row['productFilename']
            # Reproduce lightkurve's cache directory structure
            obs_id = row['obs_id']
            local_dir = os.path.join(CACHE_DIR, "mastDownload", "TESS", obs_id)
            local_path = os.path.join(local_dir, filename)
            url = MAST_BASE + uri
            urls.append((url, local_path))
        return tic_id, urls
    except Exception as e:
        return tic_id, []


def main():
    parser = argparse.ArgumentParser(description="Pre-download TOI light curves")
    parser.add_argument("--rate", type=int, default=10, help="MAST queries/s for URL resolution")
    parser.add_argument("--threads", type=int, default=8, help="Parallel URL resolution threads")
    parser.add_argument("--aria2-connections", type=int, default=8, help="aria2c parallel downloads")
    parser.add_argument("--limit", type=int, default=0, help="Max stars (0=all)")
    parser.add_argument("--urls-only", action="store_true", help="Only generate URL list, don't download")
    args = parser.parse_args()

    # Load targets
    from exonit.targets import parse_toi_catalog
    from exonit.db import ExoNitDB

    db = ExoNitDB('data/exonit.db')
    all_targets = parse_toi_catalog('data/toi_catalog.csv')
    all_in_db = set(r['tic_id'] for r in db._query('SELECT tic_id FROM stars'))
    to_download = [t for t in all_targets if t['tic_id'] not in all_in_db]
    to_download.sort(key=lambda t: t['tmag'])
    if args.limit > 0:
        to_download = to_download[:args.limit]
    db.close()

    print(f"Pre-download TOI light curves")
    print(f"  Targets: {len(to_download)}")
    print(f"  Cache: {CACHE_DIR}")
    print(f"{'='*60}")

    # Phase 1: Resolve URLs (parallel, rate-limited)
    print(f"\nPhase 1: Resolving URLs ({args.threads} threads, {args.rate} req/s)...")
    t_start = time.time()
    limiter = RateLimiter(args.rate)
    all_downloads = []  # (url, local_path)
    done = 0
    no_data = 0

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {pool.submit(resolve_urls, t['tic_id'], 12, limiter): t
                   for t in to_download}
        for future in as_completed(futures):
            tic_id, urls = future.result()
            done += 1
            if urls:
                # Skip files that already exist
                new_urls = [(u, p) for u, p in urls if not os.path.exists(p)]
                all_downloads.extend(new_urls)
            else:
                no_data += 1
            if done % 100 == 0:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  {done}/{len(to_download)} resolved "
                      f"({len(all_downloads)} files, {no_data} no data) "
                      f"— {rate:.1f} stars/s")

    resolve_time = time.time() - t_start
    print(f"\n  Resolved: {done} stars in {resolve_time:.0f}s")
    print(f"  Files to download: {len(all_downloads)}")
    print(f"  No data: {no_data}")
    print(f"  Already cached: skipped")

    if not all_downloads:
        print("\nNothing to download!")
        return

    # Write aria2c input file
    aria2_input = os.path.join("data", "aria2_downloads.txt")
    with open(aria2_input, 'w') as f:
        for url, local_path in all_downloads:
            local_dir = os.path.dirname(local_path)
            filename = os.path.basename(local_path)
            f.write(f"{url}\n")
            f.write(f"  dir={local_dir}\n")
            f.write(f"  out={filename}\n")

    print(f"\n  aria2c input written: {aria2_input} ({len(all_downloads)} files)")

    if args.urls_only:
        print("  --urls-only: skipping download")
        return

    # Phase 2: Download with aria2c
    est_gb = len(all_downloads) * 2 / 1024  # ~2MB per FITS
    print(f"\nPhase 2: Downloading with aria2c ({args.aria2_connections} connections)...")
    print(f"  Estimated size: ~{est_gb:.1f} GB")
    print(f"{'='*60}")

    result = subprocess.run([
        "aria2c",
        f"--input-file={aria2_input}",
        f"--max-concurrent-downloads={args.aria2_connections}",
        "--max-connection-per-server=4",
        "--min-split-size=1M",
        "--continue=true",          # resume interrupted downloads
        "--auto-file-renaming=false",
        "--allow-overwrite=false",   # skip existing files
        "--console-log-level=warn",
        "--summary-interval=30",
    ])

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"PRE-DOWNLOAD COMPLETE — {elapsed/60:.1f} min")
    fits_count = sum(1 for _ in
                     (f for d, _, fs in os.walk(CACHE_DIR)
                      for f in fs if f.endswith('.fits')))
    print(f"  FITS files in cache: {fits_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
