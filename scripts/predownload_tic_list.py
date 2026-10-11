#!/usr/bin/env python3
"""Pre-download SPOC light curves for an explicit list of TIC IDs.

Reuses predownload_toi's URL resolver + aria2c flow, but driven by a
file of TIC IDs (one per line) instead of the full TOI catalog. Used to
fill the gap of TICs not yet present in the TESS cache (EXONIT_NFS_CACHE).

  python3 scripts/predownload_tic_list.py /tmp/missing_tics.txt
"""
import os
import sys
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predownload_toi import resolve_urls, RateLimiter, CACHE_DIR  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Pre-download SPOC light curves for a file of TIC IDs "
                                             "(one per line) into the SPOC cache (EXONIT_NFS_CACHE).")
    ap.add_argument("tic_file", help="text file with one TIC ID per line")
    ap.add_argument("rate", type=int, nargs="?", default=10, help="MAST queries per second (default 10)")
    ap.add_argument("threads", type=int, nargs="?", default=8, help="URL-resolution threads (default 8)")
    a = ap.parse_args()
    tic_file, rate, threads = a.tic_file, a.rate, a.threads

    tics = [int(l.strip()) for l in open(tic_file) if l.strip()]
    print(f"TICs to resolve: {len(tics)}  rate={rate}/s threads={threads}")

    limiter = RateLimiter(rate)
    all_downloads = []
    no_data = []
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futs = {pool.submit(resolve_urls, t, 12, limiter): t for t in tics}
        for fut in as_completed(futs):
            tic_id, urls = fut.result()
            done += 1
            if urls:
                all_downloads.extend([(u, p) for u, p in urls if not os.path.exists(p)])
            else:
                no_data.append(tic_id)
            if done % 50 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(tics)} resolved "
                      f"({len(all_downloads)} files, {len(no_data)} no SPOC data) "
                      f"— {done/el:.1f} stars/s", flush=True)

    print(f"\nResolved {done} in {time.time()-t0:.0f}s")
    print(f"  Files to download: {len(all_downloads)}")
    print(f"  No SPOC data (FFI-only / unavailable): {len(no_data)}")
    with open("data/missing_no_spoc.txt", "w") as f:
        for t in sorted(no_data):
            f.write(f"{t}\n")
    print(f"  Wrote no-SPOC list -> data/missing_no_spoc.txt")

    if not all_downloads:
        print("Nothing to download.")
        return

    aria2_input = os.path.join("data", "aria2_missing.txt")
    with open(aria2_input, "w") as f:
        for url, local_path in all_downloads:
            f.write(f"{url}\n  dir={os.path.dirname(local_path)}\n  out={os.path.basename(local_path)}\n")
    print(f"aria2c input: {aria2_input} ({len(all_downloads)} files)")

    subprocess.run([
        "aria2c",
        f"--input-file={aria2_input}",
        "--max-concurrent-downloads=8",
        "--max-connection-per-server=4",
        "--min-split-size=1M",
        "--continue=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=false",
        "--console-log-level=warn",
        "--summary-interval=30",
    ])
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
