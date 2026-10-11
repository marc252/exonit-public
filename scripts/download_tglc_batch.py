#!/usr/bin/env python3
"""Download a TGLC batch from a sliced MAST download script, in parallel.

Reads a file of `curl ... --output 'PATH' 'URL'` lines, downloads each into
--dest, skips files already present (idempotent / resumable). Polite concurrency.

Failed targets are retried automatically: after the main pass, the still-failed
set is swept again (up to --retry-passes times, with exponential backoff). This
recovers transient MAST hiccups (timeouts, connection resets, occasional 5xx)
without re-touching the files that already succeeded. The reported `fail=` count
is the residual after all retries — i.e. genuinely unrecoverable targets, each
also named on a `[failed] <relpath>` log line.
"""
import argparse
import re
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import urllib3.util.retry

QUOTED = re.compile(r"'([^']*)'")


def parse_iter(script_path):
    """Yield (relpath, url) pairs from a MAST bulk curl script, line by line.

    Uses a file iterator instead of read_text().splitlines() so the full
    script (up to ~821 MB for s0001) is never materialised in memory.
    """
    with open(script_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("curl"):
                continue
            q = QUOTED.findall(line)
            if len(q) >= 2:
                yield (q[0], q[1])  # (out_relpath, url)


def parse(script_path):
    """Return all (relpath, url) pairs as a list.

    Delegates to parse_iter(); kept for backwards compatibility with any
    caller that needs a concrete list.
    """
    return list(parse_iter(script_path))


def _build_session(n_workers: int) -> requests.Session:
    """Return a requests.Session with a shared connection pool and retry policy.

    The pool size matches the worker count so each thread can hold one
    persistent TCP+TLS connection rather than opening a new one per file.
    The HTTPAdapter Retry covers transient HTTP status codes and connection
    errors so the manual per-attempt loop only needs to handle
    non-HTTP exceptions (Timeout, ConnectionError, etc.).
    """
    retry = urllib3.util.retry.Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=n_workers,
        pool_maxsize=n_workers,
        max_retries=retry,
    )
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--log", required=True)
    ap.add_argument("--retry-passes", type=int, default=3,
                    help="Extra sweeps over the failed set after the main pass "
                         "(exponential backoff between sweeps). 0 disables.")
    # --- Chunked-caller support (2026-07-14) -------------------------------
    # discovery_download.py now downloads a batch in CHUNKS (download → index →
    # pack → free, per chunk) so an interruption cannot leave a pile of
    # downloaded-but-unindexed files. Each chunk is a separate run of THIS
    # script, but the log it writes is tailed by the UI and must keep reading
    # like ONE continuous download: a progress bar that restarts at every chunk
    # (or a counter that jumps backwards) is worse than no progress bar.
    #
    # So the caller passes what happened BEFORE this chunk, and every line we
    # emit reports GLOBAL numbers. All default to the single-run values, so a
    # plain CLI invocation is byte-for-byte identical to before.
    ap.add_argument("--progress-total", type=int, default=None,
                    help="Grand total across all chunks (default: this chunk's).")
    ap.add_argument("--prior-done", type=int, default=0)
    ap.add_argument("--prior-skip", type=int, default=0)
    ap.add_argument("--prior-fail", type=int, default=0)
    ap.add_argument("--suppress-finish", action="store_true",
                    help="Do not write the [finish] line. The chunked caller "
                         "writes ONE at the true end — the download-status "
                         "endpoint reads [finish] as 'this download completed', "
                         "so a per-chunk one would report a run that died at "
                         "chunk 2 of 10 as a clean success.")
    args = ap.parse_args()

    dest = Path(args.dest)
    jobs = parse(args.script)
    chunk_total = len(jobs)
    total = args.progress_total if args.progress_total is not None else chunk_total
    prior_done, prior_skip, prior_fail = (
        args.prior_done, args.prior_skip, args.prior_fail)
    t0 = time.monotonic()

    # Sliding-window rate: track the last up-to-512 completion timestamps.
    recent: deque = deque(maxlen=512)

    session = _build_session(args.workers)

    def fetch(job):
        """Download one job. Returns ("done"|"skip"|"fail", job).

        Tallying happens in the main thread from these return values, so there
        are no shared mutable counters across worker threads.
        """
        rel, url = job
        out = dest / rel
        if out.exists() and out.stat().st_size > 1000:
            return ("skip", job)
        out.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                r = session.get(url, timeout=60)
                r.raise_for_status()
                tmp = out.with_suffix(out.suffix + ".part")
                tmp.write_bytes(r.content)
                tmp.rename(out)
                return ("done", job)
            except Exception:
                time.sleep(0.5 * (attempt + 1))
        return ("fail", job)

    def log(msg):
        with open(args.log, "a") as f:
            f.write(msg + "\n")

    # Cumulative across all passes, so the [N/total] progress bar keeps filling
    # (against the ORIGINAL total) while the retry sweeps recover stragglers.
    done_total = [0]
    skip_total = [0]

    def run_pass(pass_jobs):
        """Download every job once; return the list of still-failed jobs."""
        failed = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(fetch, j) for j in pass_jobs]
            n = 0
            for fut in as_completed(futs):
                status, job = fut.result()
                n += 1
                if status == "done":
                    done_total[0] += 1
                elif status == "skip":
                    skip_total[0] += 1
                else:
                    failed.append(job)
                recent.append(time.monotonic())
                # GLOBAL counters (prior chunks + this one) — see the
                # --prior-* args: the log must read as one continuous download.
                g_done = prior_done + done_total[0]
                g_skip = prior_skip + skip_total[0]
                g_fail = prior_fail + len(failed)
                processed = g_done + g_skip + g_fail
                local_processed = done_total[0] + skip_total[0] + len(failed)
                if n % 250 == 0 or n == len(pass_jobs):
                    el = time.monotonic() - t0
                    # Sliding-window rate: files/sec over the last ≤512 completions.
                    if len(recent) >= 2:
                        window_rate = (len(recent) - 1) / (recent[-1] - recent[0])
                    else:
                        window_rate = local_processed / el if el else 0
                    eta = (total - processed) / window_rate / 60 if window_rate else 0
                    log(f"[{processed}/{total}] done={g_done} "
                        f"skip={g_skip} fail={g_fail} "
                        f"{window_rate:.1f}/s eta={eta:.1f}min")
        return failed

    if not prior_done and not prior_skip and not prior_fail:
        log(f"[start] {total} targets → {dest} ({args.workers} workers)")
    remaining = run_pass(jobs)
    for k in range(1, max(0, args.retry_passes) + 1):
        if not remaining:
            break
        backoff = min(30, 2 ** k)
        log(f"[retry {k}/{args.retry_passes}] {len(remaining)} failed; "
            f"retrying in {backoff}s")
        time.sleep(backoff)
        remaining = run_pass(remaining)

    fail_total = len(remaining)
    if fail_total:
        # Name the unrecoverable targets (e.g. genuinely missing on MAST) so a
        # human can inspect them; these survived the main pass + all retries.
        for rel, _url in remaining:
            log(f"[failed] {rel}")
    el = time.monotonic() - t0
    if not args.suppress_finish:
        log(f"[finish] done={done_total[0] + prior_done} "
            f"skip={skip_total[0] + prior_skip} fail={fail_total + prior_fail} "
            f"in {el/60:.1f}min")
    # stdout stays CHUNK-LOCAL: the caller sums the chunks itself, and a
    # chunked caller that read global numbers here would double-count.
    print(f"done={done_total[0]} skip={skip_total[0]} fail={fail_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
