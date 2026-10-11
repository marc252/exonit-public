#!/usr/bin/env python3
"""Download a batch of TGLC FITS files for the discovery campaign.

This is the CLI backend for the Downloads-page "download a batch" button.
It:
  1. Resolves + validates the sector (from registry active_sector; per-sector bulk script).
  2. Ensures the MAST bulk curl script is present, fetching it if needed.
  3. Parses the script into ordered (relpath, url) jobs using
     download_tglc_batch.parse() — no regex duplication.
  4. Identifies the next --count not-yet-downloaded jobs.
  5. In --dry-run mode: prints a JSON summary and exits.
  6. Otherwise: downloads via download_tglc_batch.py subprocess, then
     indexes via index_tglc_batch.py subprocess, then updates the sector
     registry downloaded count.

All output is JSON on stdout (one line).  Subprocess logs go to
<dest>/download_<timestamp>.log.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

# Add scripts/ to path so we can import the sibling download_tglc_batch script
# (not a package — the exonit package itself is importable via editable install).
_SCRIPTS_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from download_tglc_batch import parse as _parse_script, parse_iter as _parse_iter_script  # noqa: E402
from exonit.discovery_state import (  # noqa: E402
    load_registry, save_registry, active_sector, get_sector,
)

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

# Per-sector TGLC bulk download scripts (MAST hlsp_tglc). The active sector is
# config (registry active_sector / EXONIT_SECTOR), not hardcoded: home = 1,
# AWS = 2. Adding a sector = one entry here + a registry entry.
SECTOR_BULK = {
    1: {
        "bulk_url": ("https://archive.stsci.edu/hlsps/tglc/download_scripts/"
                     "hlsp_tglc_tess_ffi_s0001_tess_v1_llc.sh"),
        "script": "data/discovery/s0001_full.sh",
    },
    2: {
        "bulk_url": ("https://archive.stsci.edu/hlsps/tglc/download_scripts/"
                     "hlsp_tglc_tess_ffi_s0002_tess_v1_llc.sh"),
        "script": "data/discovery/s0002_full.sh",
    },
    3: {
        "bulk_url": ("https://archive.stsci.edu/hlsps/tglc/download_scripts/"
                     "hlsp_tglc_tess_ffi_s0003_tess_v1_llc.sh"),
        "script": "data/discovery/s0003_full.sh",
    },
    14: {
        "bulk_url": ("https://archive.stsci.edu/hlsps/tglc/download_scripts/"
                     "hlsp_tglc_tess_ffi_s0014_tess_v1_llc.sh"),
        "script": "data/discovery/s0014_full.sh",
    },
    15: {
        "bulk_url": ("https://archive.stsci.edu/hlsps/tglc/download_scripts/"
                     "hlsp_tglc_tess_ffi_s0015_tess_v1_llc.sh"),
        "script": "data/discovery/s0015_full.sh",
    },
    17: {
        "bulk_url": ("https://archive.stsci.edu/hlsps/tglc/download_scripts/"
                     "hlsp_tglc_tess_ffi_s0017_tess_v1_llc.sh"),
        "script": "data/discovery/s0017_full.sh",
    },
    23: {
        "bulk_url": ("https://archive.stsci.edu/hlsps/tglc/download_scripts/"
                     "hlsp_tglc_tess_ffi_s0023_tess_v1_llc.sh"),
        "script": "data/discovery/s0023_full.sh",
    },
}

# Hold-out coverage sectors (2026-07-20): greedy set-cover of the 151
# uncovered reserved TICs — s7(20) s36(+16) s18(+14) s5(+12) s11(+11)
# s45(+10) s26(+8) s6(+7). Same STScI URL pattern as every sector above.
for _s in (5, 6, 7, 11, 18, 26, 36, 45):
    SECTOR_BULK[_s] = {
        "bulk_url": ("https://archive.stsci.edu/hlsps/tglc/download_scripts/"
                     f"hlsp_tglc_tess_ffi_s{_s:04d}_tess_v1_llc.sh"),
        "script": f"data/discovery/s{_s:04d}_full.sh",
    }

MIN_FILE_BYTES = 1001  # same threshold as download_tglc_batch.py (>1000)


# ------------------------------------------------------------------ #
# Pure logic (importable for tests)
# ------------------------------------------------------------------ #

def select_next_jobs(jobs, dest: Path, count: int,
                     _chunk: int = 10_000, _workers: int | None = None,
                     present_keys: set | None = None):
    """Return the next *count* not-yet-downloaded (relpath, url) pairs.

    Parameters
    ----------
    jobs:
        Ordered iterable of (relpath, url) — may be a generator (parse_iter)
        or a concrete list.  When a generator is passed the call is O(N_scanned)
        in memory rather than O(N_total).
    dest:
        Root directory where output files live (relpath is relative to it).
    count:
        Maximum number of jobs to return.
    _chunk:
        Internal chunk size for batching generator consumption and parallelising
        exists()+stat() calls.  Exposed for tests; production code should use
        the default (10_000).
    _workers:
        Number of threads for the ThreadPoolExecutor.  Default is
        min(32, (os.cpu_count() or 8) * 2), which is I/O-appropriate and
        well below the 25%-CPU rule (stat() threads are not CPU-bound).
        Exposed for tests.

    Returns
    -------
    tuple[int, list]
        (already_scanned_count, selected_batch)

        already_scanned_count — number of already-present files seen while
          scanning up to the early-exit point.  NOTE: when count is small
          relative to the total script size this reflects only the scanned
          prefix, not the absolute total on disk.  The registry's `downloaded`
          field (updated after each run) is the authoritative absolute count.
        selected_batch        — up to *count* absent jobs, in order.

    Early-exit: chunk-level — scanning stops after the chunk in which the
    *count*-th absent job is found.  Within a chunk, all exists()+stat() calls
    are issued in parallel (ThreadPoolExecutor, I/O-bound), preserving the
    order of results via executor.map.  already and batch are bit-identical to
    the sequential version for the same inputs.
    """
    if _workers is None:
        _workers = min(32, (os.cpu_count() or 8) * 2)

    def _is_present(relpath: str) -> bool:
        """"Present" means INDEXED, not "sitting on disk" (2026-07-14).

        This used to return True for any file on disk, and that was the hole
        that made an interrupted download unrecoverable: a server restart killed
        discovery_download.py after it had fetched 24,747 files but before its
        index+pack step, so those files were on disk and in NO index — and the
        next run SKIPPED them (they existed) and therefore never indexed them.
        They were invisible dead weight: the pipeline cannot use a FITS the
        metadata cache has never heard of, and /api/fits/<name> 404s for it.

        Presence is now the skip-set — the union of the tglc index and
        fits_blobs, i.e. the files that are genuinely usable. A downloaded-but-
        unindexed file is therefore re-selected on the next run, and costs
        almost nothing to recover: download_tglc_batch.py skips the HTTP fetch
        for any file already on disk (its skip= counter), so the file is simply
        indexed and packed this time round.

        The on-disk check survives ONLY as the no-index fallback (present_keys
        is None), which is the old behaviour for deployments without an index.
        """
        if present_keys is not None:
            return os.path.basename(relpath) in present_keys
        out = dest / relpath
        return out.exists() and out.stat().st_size > MIN_FILE_BYTES - 1

    already = 0
    batch: list = []
    it = iter(jobs)

    with ThreadPoolExecutor(max_workers=_workers) as pool:
        while True:
            # Consume up to _chunk (relpath, url) pairs from the generator.
            chunk = list(islice(it, _chunk))
            if not chunk:
                break  # generator exhausted

            relpaths = [r for r, _ in chunk]

            # Parallel exists()+stat() — executor.map preserves input order.
            for (relpath, url), present in zip(chunk, pool.map(_is_present, relpaths)):
                if present:
                    already += 1
                else:
                    batch.append((relpath, url))
                    if len(batch) == count:
                        # Early-exit: batch is full — discard the rest of this
                        # chunk's results and stop consuming the generator.
                        return already, batch

    return already, batch


# ------------------------------------------------------------------ #
# Script download helper
# ------------------------------------------------------------------ #

# Poll interval (seconds) for the bulk-script fetch progress monitor below.
# Module-level so tests can shrink it (monkeypatch) instead of sleeping for
# real seconds per iteration.
_BULK_FETCH_POLL_INTERVAL_S = 5.0


def _head_content_length(url: str, timeout: float = 20.0):
    """Best-effort HEAD request to learn the total download size in bytes.

    Fail-open: any problem (network error, timeout, missing/non-numeric
    Content-Length header) returns None rather than raising. This is only
    used to report progress — it must never block or fail the actual
    download.
    """
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except Exception:
        return None


def _fetch_bulk_script(url: str, dest_path: Path) -> None:
    """Download the bulk curl script from *url* to *dest_path* with resume.

    This can be a multi-GB transfer (e.g. sector 14's bulk script is 2.42 GB)
    that used to run silently under ``subprocess.run(capture_output=True)``
    for tens of minutes — the status endpoint had nothing to report during
    that time. This now emits JSON progress markers via _emit() so the
    "muted" phase is visible: bulk_fetch_start (with the expected total size,
    if known) → periodic bulk_fetch (bytes_done so far) → bulk_fetch_done.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part = dest_path.with_suffix(dest_path.suffix + ".part")

    bytes_total = _head_content_length(url)
    _emit({"status": "bulk_fetch_start", "bytes_total": bytes_total, "url": url})

    # Use curl with -C - (resume) and -L (follow redirects), same as before.
    # stderr goes to a tempfile rather than PIPE: a PIPE nobody reads from can
    # deadlock once curl fills its stderr buffer. stdout is discarded (curl's
    # own progress meter writes to stderr, not stdout).
    cmd = ["curl", "-f", "-L", "-C", "-", "--output", str(part), url]
    stderr_fd, stderr_path = tempfile.mkstemp(suffix=".log", prefix="bulk_fetch_stderr_")
    os.close(stderr_fd)
    try:
        with open(stderr_path, "wb") as stderr_out:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr_out)

            # Monitor loop: poll the .part file size periodically and emit a
            # progress marker only when it has changed since the last emit,
            # so a stalled curl does not spam the log every interval.
            last_emitted = None
            while proc.poll() is None:
                time.sleep(_BULK_FETCH_POLL_INTERVAL_S)
                try:
                    size = part.stat().st_size
                except OSError:
                    size = 0
                if size != last_emitted:
                    _emit({"status": "bulk_fetch", "bytes_done": size,
                           "bytes_total": bytes_total})
                    last_emitted = size

            returncode = proc.returncode
            if returncode != 0:
                with open(stderr_path, "rb") as fh:
                    stderr_bytes = fh.read()
                raise RuntimeError(
                    f"Failed to download bulk script: "
                    f"{stderr_bytes.decode(errors='replace')[:300]}"
                )
    finally:
        try:
            os.remove(stderr_path)
        except OSError:
            pass

    final_size = part.stat().st_size
    _emit({"status": "bulk_fetch_done", "bytes_done": final_size})
    part.rename(dest_path)


# ------------------------------------------------------------------ #
# Pack + upload ONE batch of FITS, with retry+backoff (shared by
# _index_and_pack_chunk below and scripts/repair_missing_blobs.py)
# ------------------------------------------------------------------ #

# Retry schedule for the pack+upload+index step (2026-07-21 MinIO multipart
# read-timeout incident: "Read timeout on endpoint URL ...
# ?uploadId=...&partNumber=N" during a live s7 fleet run hit 56 chunks —
# the code used to record ``pack_failed`` and move on, leaving 560,000
# files indexed in ``tglc`` with no blob and no tar in object storage.
# Because the skip-set that decides "already downloaded" is the tglc ∪
# fits_blobs union, those files read as already-done forever — no re-run
# self-heals the gap. Retry+backoff itself lives in
# ``exonit.object_store.with_s3_retries`` (shared with
# scripts/compact_completeness.py — see that module's 2026-07-21 fix) so
# both callers use the exact same schedule and never duplicate the loop.


def _pack_and_upload_chunk(*, items, ostore, container_key: str, metadata_cache,
                           ci: int, progress_log_dir: Path, lot: str,
                           delete_local: bool = True) -> int:
    """Pack *items* into ONE tar, upload it to *container_key*, and index
    the result via ``add_fits_blobs_bulk`` — with retry+backoff.

    items: list of ``(member_name, local_path)`` pairs. ``member_name`` is
    the flat FITS basename — both the tar member name AND the
    ``fits_blobs.filename`` key; ``local_path`` (str or ``Path``) is where
    to read the bytes from on THIS host.

    Retries the WHOLE sequence (pack -> upload -> add_fits_blobs_bulk) via
    ``exonit.object_store.with_s3_retries`` — never just the failed part,
    because a partially-uploaded tar or a partially-applied index is not a
    state worth resuming from; starting clean each attempt is simpler and
    provably correct. Local files are never touched (nothing packed away,
    nothing unlinked) until the FULL sequence for THIS call has succeeded,
    so a failed attempt leaves every input exactly as it was — safe for the
    next retry, or for a wholly separate later run of
    scripts/repair_missing_blobs.py to pick up.

    Returns the number of files successfully uploaded+indexed. On success
    that is ``len(items)`` and (when *delete_local* is True) every local
    file has been unlinked. On total failure (every attempt raised) it is
    ``0``: a ``pack_failed`` event is emitted (unchanged shape — nothing
    downstream that parses the log needs to change), a ``pack_gap``
    manifest listing every item's local path — one per line — is written to
    ``<progress_log_dir>/pack_gap_<lot>_c<ci:03d>.txt``, and its path is
    emitted in a ``pack_gap`` event. That manifest is the self-describing
    "hole": nothing else marks these files done, so a human (or
    scripts/repair_missing_blobs.py) can act on it directly instead of
    reconstructing which files were affected from a log line.
    """
    if not items:
        return 0

    from exonit.object_store import with_s3_retries

    def _once():
        import tempfile as _tempfile

        from exonit.fits_pack import pack_to_stream as _pack_to_stream

        fd, tmp_tar = _tempfile.mkstemp(
            suffix=".tar", dir=str(progress_log_dir))
        os.close(fd)
        try:
            with open(tmp_tar, "w+b") as _out:
                locs = _pack_to_stream(
                    _out, [(name, str(p)) for name, p in items])
            ostore.put_file(container_key, tmp_tar,
                            content_type="application/x-tar")
            metadata_cache.add_fits_blobs_bulk(
                [(n, container_key, o, length) for n, o, length in locs])
            return locs
        finally:
            try:
                os.remove(tmp_tar)
            except OSError:
                pass

    def _on_retry(attempt, max_attempts, delay, exc):
        _emit({"status": "pack_retry", "chunk": ci, "attempt": attempt,
               "max_attempts": max_attempts, "delay_s": delay,
               "detail": str(exc)[:300]})

    try:
        locs = with_s3_retries(_once, ostore=ostore, on_retry=_on_retry)
    except Exception as exc:
        # Every attempt failed — record the gap so it is recoverable
        # instead of silently invisible (see the module comment above).
        gap_path = Path(progress_log_dir) / f"pack_gap_{lot}_c{ci:03d}.txt"
        gap_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gap_path, "w") as fh:
            for _name, p in items:
                fh.write(f"{p}\n")
        _emit({"status": "pack_failed", "chunk": ci, "detail": str(exc)[:300]})
        _emit({"status": "pack_gap", "chunk": ci, "manifest": str(gap_path)})
        return 0

    if delete_local:
        for _name, p in items:
            try:
                Path(p).unlink()
            except OSError:
                pass
    return len(locs)


# ------------------------------------------------------------------ #
# Index + pack ONE chunk (runs on a background thread — see main())
# ------------------------------------------------------------------ #

def _index_and_pack_chunk(*, ci: int, n_chunks: int, chunk: list, cts: str,
                          ts: str, dest: Path, args, ostore,
                          downloaded_snapshot: int,
                          uploaded_before: int, result: dict,
                          temp_files: list, temp_files_lock: threading.Lock,
                          metadata_cache) -> None:
    """Index + pack chunk *ci*. Runs on a background thread while the main
    thread downloads chunk *ci*+1 (2026-07-18 overlap pipeline — see spec
    docs/superpowers/specs/2026-07-18-download-index-pipeline.md).

    *result* is a dict private to THIS thread (a fresh one per call) — the
    caller only reads it after ``Thread.join()``, so no lock is needed for it.
    On success it holds ``indexed``, ``uploaded`` and ``uploaded_total``. On
    failure it holds ``error`` and nothing else is indexed/packed/emitted for
    this chunk — the caller must stop making progress past chunk *ci*-1.

    *temp_files* is shared with the main thread (both append filenames to it
    for the final cleanup pass), hence *temp_files_lock*.

    *ts* is the whole run's stamp (shared by every chunk, unlike *cts* which
    is per-chunk) — passed through to ``_pack_and_upload_chunk`` as the
    ``lot`` a ``pack_gap`` manifest is named after, so every gap from the
    same download run groups under one recognisable prefix.

    *metadata_cache* is ONE ``MetadataCache`` instance shared across every
    chunk of the whole run (constructed once by ``main()``, closed once at
    the end) — NOT a fresh instance per chunk (2026-07-20 review fix,
    progress/f3-indexer-pg-review.md note 2: constructing one per chunk meant
    opening a fresh ``psycopg_pool.ConnectionPool`` — a TCP connection + auth
    handshake — every ~10k files once ``EXONIT_METADATA_DB_URL`` points at
    PG, wasteful over an hours-long sector download). Safe to share across
    threads: only ONE ``_index_and_pack_chunk`` call is ever in flight at a
    time (the caller always joins chunk N's thread before starting N+1's —
    see main()'s "indexing must stay strictly sequential" comment), and even
    if that changed, ``MetadataCache``'s SQLite backend serializes every call
    behind its own ``threading.Lock`` and its PG backend's
    ``psycopg_pool.ConnectionPool`` is thread-safe by design (each borrow
    gets an exclusive connection, or blocks for one — never shared/corrupted
    state) — safe either way.

    Any exception here — not just a nonzero index-subprocess returncode — is
    caught and written into *result* rather than left to Python's default
    thread excepthook (which only logs to stderr and would leave the caller's
    ``join()`` never seeing a failure): "exceptions from the background
    thread must never be lost silently".
    """
    try:
        _emit({"status": "index_start", "chunk": ci})
        batch_list = dest / f"_idxbatch_{cts}.txt"
        with temp_files_lock:
            temp_files.append(batch_list)
        with open(batch_list, "w") as bf:
            for relpath, _u in chunk:
                bf.write(str(dest / relpath) + "\n")

        idx_cmd = [
            sys.executable, str(_SCRIPTS_DIR / "index_tglc_batch.py"),
            "--cache", str(dest),
            "--index", str(args.index),
            "--files-from", str(batch_list),
        ]
        idx_result = subprocess.run(idx_cmd, capture_output=True, text=True)
        if idx_result.returncode != 0:
            _emit({"status": "index_failed", "chunk": ci,
                   "returncode": idx_result.returncode,
                   "stderr": idx_result.stderr[:500]})
            result["error"] = {"chunk": ci, "returncode": idx_result.returncode,
                               "stderr": idx_result.stderr[:500]}
            return
        result["indexed"] = _parse_indexed(idx_result.stdout)

        # --- pack this chunk into ONE container + free local copies ---
        # A chunk that is indexed but NOT packed is the second half of the
        # same trap the chunking itself fixed: the file is in tglc, so the
        # next run considers it done, yet /api/fits/ resolves through
        # fits_blobs and 404s. So packing must happen per chunk, right after
        # its index. A packing failure is non-fatal (recorded, chunk still
        # marked done) — only an index failure halts the run.
        c_uploaded = 0
        up_failed = 0
        if ostore is not None and chunk:
            _emit({"status": "upload_start", "chunk": ci, "count": len(chunk)})
            present = [(os.path.basename(r), dest / r)
                       for r, _u in chunk if (dest / r).exists()]
            container_key = f"containers/s{args.sector:04d}/dl_{cts}.tar"
            # Retry+backoff, pack_gap-on-total-failure: see
            # _pack_and_upload_chunk's docstring (2026-07-21 fix). A total
            # failure already emits pack_failed + pack_gap itself; nothing
            # left to do here except reflect it in upload_done, exactly as
            # before.
            c_uploaded = _pack_and_upload_chunk(
                items=present, ostore=ostore, container_key=container_key,
                metadata_cache=metadata_cache, ci=ci, progress_log_dir=dest,
                lot=ts, delete_local=True)
            if present and c_uploaded == 0:
                up_failed = len(chunk)
            _emit({"status": "upload_done", "chunk": ci,
                   "uploaded": c_uploaded, "failed": up_failed})

        result["uploaded"] = c_uploaded
        result["uploaded_total"] = uploaded_before + c_uploaded
        _emit({"status": "chunk_done", "chunk": ci, "of": n_chunks,
               "downloaded": downloaded_snapshot,
               "uploaded": result["uploaded_total"]})
    except Exception as exc:  # pragma: no cover - safety net, never silent
        result["error"] = {"chunk": ci, "exception": repr(exc)}
        _emit({"status": "index_failed", "chunk": ci, "returncode": -1,
               "stderr": repr(exc)[:500]})


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Download a batch of TGLC FITS for the discovery campaign."
    )
    ap.add_argument("--sector", type=int, default=None,
                    help="Sector to download; default = registry active_sector")
    ap.add_argument("--count", type=int, default=100_000)
    ap.add_argument("--script", default=None,
                    help="Local bulk curl script path (default: per-sector)")
    ap.add_argument("--bulk-url", default=None,
                    help="URL to fetch bulk script if absent (default: per-sector)")
    ap.add_argument("--dest", default="data/discovery/tglc_cache")
    ap.add_argument("--index", default="data/metadata_cache.db",
                    help="TGLC index DB (not exonit.db)")
    ap.add_argument("--registry", default=None,
                    help=(
                        "Sector registry JSON to update. Default: "
                        "<repo_root>/data/discovery/sectors.json (anchored "
                        "to this script's own checkout, never the process "
                        "cwd). Left at this default, the actual write is "
                        "guarded — see _guard_registry_write()."
                    ))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=10_000,
                    help="Files per download→index→pack cycle (default 10000). "
                         "Progress becomes durable at every chunk boundary, so "
                         "an interruption can lose at most one chunk instead of "
                         "the whole batch. Also caps local disk use at one "
                         "chunk (~0.5 GB) rather than the full batch.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan only; do not download or write anything")
    ap.add_argument("--no-fetch-script", action="store_true",
                    help="Fail if --script is absent instead of downloading it")
    ap.add_argument("--recompute-registry", action="store_true",
                    help=(
                        "Recompute the 'downloaded' counter for the specified "
                        "sector (or all sectors in SECTOR_BULK when --sector is "
                        "not given) from the tglc ∪ fits_blobs union and write "
                        "the updated value to the registry.  Emits a JSON "
                        "summary and exits without downloading anything."
                    ))
    ap.add_argument("--progress-log", default=None,
                    help=(
                        "Path to use for the download_tglc_batch.py --log file. "
                        "When supplied, both the server's captured stdout and the "
                        "per-file progress lines land in the same file so the "
                        "status endpoint can read them.  When absent, an "
                        "auto-generated download_<TS>.log is created inside --dest."
                    ))
    return ap


# Guards stdout writes: since 2026-07-18 the index+pack step of chunk N runs
# on a background thread while the main thread is already downloading chunk
# N+1, and both sides call _emit(). print() to a shared stream is not
# guaranteed atomic across threads (the write can interleave at the byte
# level), which would corrupt the line-based JSON the status endpoint parses.
# One lock around the whole print+flush makes every emitted line atomic.
_emit_lock = threading.Lock()


def _emit(obj: dict) -> None:
    # flush=True is essential: stdout is redirected to a file (block-buffered
    # by default).  Without flush the line may never reach disk if the process
    # dies before its stdio buffers are flushed.
    line = json.dumps(obj)
    with _emit_lock:
        print(line, flush=True)


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    # --registry default: anchored to THIS script's own repo root (not the
    # process cwd) so an invocation from an unrelated cwd cannot silently
    # target a different sectors.json. Distinct from _registry_is_default
    # below: this only fixes ambiguous-cwd resolution; it does NOT by
    # itself fix the cross-host case where cwd happens to already be a
    # repo root, just the WRONG HOST's clone of it (2026-07-25 sector 18
    # incident) — that is what _guard_registry_write()/_is_server_host()
    # are for.
    _registry_is_default = args.registry is None
    if _registry_is_default:
        args.registry = str(_PROJECT_ROOT / "data" / "discovery" / "sectors.json")

    # ---- 0. --recompute-registry early exit ---------------------- #
    # Recomputes the 'downloaded' counter from the tglc ∪ fits_blobs union for
    # one or all known sectors and writes the updated values to the registry.
    # No bulk-script parsing, no network access, no downloads.
    if args.recompute_registry:
        if not _guard_registry_write(args.registry, _registry_is_default):
            return 3
        sectors = ([args.sector] if args.sector is not None
                   else sorted(SECTOR_BULK))

        # Change-detection gate (2026-09-25, fix C): the cron
        # (scripts/ops/recompute_download_registry.sh) runs this every 30
        # min over 8 sectors, and _indexed_basenames() scans tglc (~76M
        # rows) + fits_blobs (~76.5M rows) for EACH one — 1-2 min/sector —
        # even though its own log shows before==after almost every time:
        # downloads run on <host> in multi-day bursts, not continuously,
        # so most cycles have nothing new to count. A cheap PG catalog
        # fingerprint (no data scan) tells us whether either table has
        # actually changed since the last real recompute; if not, skip the
        # whole per-sector loop. Persisted to a small state file (this is a
        # standalone CLI invocation, not the long-lived server process, so
        # there is no shared_state singleton to reuse here) rather than
        # Redis, per the design's "keep it simple" for the cron path.
        from exonit.metadata_cache import MetadataCache
        _fp_probe = MetadataCache(args.index)
        try:
            fingerprint = _fp_probe.download_registry_fingerprint()
        finally:
            _fp_probe.close()
        max_stale_s = float(os.environ.get(
            "EXONIT_DOWNLOAD_REGISTRY_MAX_STALE_HOURS", "6")) * 3600.0
        state = _load_fingerprint_state()
        now = time.time()
        unchanged = (
            fingerprint is not None
            and fingerprint == state.get("fingerprint")
            and (now - state.get("last_full_recompute", 0.0)) < max_stale_s
        )
        if unchanged:
            _emit({"status": "recompute_skipped",
                   "reason": "fingerprint_unchanged", "sectors": sectors})
            return 0

        results = {}
        for sec in sectors:
            reg = load_registry(args.registry)
            before = (get_sector(reg, sec) or {}).get("downloaded")
            after = len(_indexed_basenames(args.index, sec))
            _update_registry(args.registry, sec, after)
            results[sec] = {"before": before, "after": after}
        if fingerprint is not None:
            _save_fingerprint_state(fingerprint, now)
        _emit({"status": "recompute_done", "sectors": results})
        return 0

    # ---- 1. Sector resolution + gate ----------------------------- #
    # Default sector from this deployment's registry (home=1, AWS=2).
    if args.sector is None:
        args.sector = active_sector(args.registry)
    # Gate: sector must be known here and not disabled in the registry.
    reg = load_registry(args.registry)
    entry = get_sector(reg, args.sector)
    if args.sector not in SECTOR_BULK or (entry is not None and entry.get("enabled") is False):
        _emit({"status": "error", "reason": "sector_not_enabled",
               "sector": args.sector})
        return 2
    # Resolve per-sector script/bulk-url defaults when not overridden.
    if args.script is None:
        args.script = SECTOR_BULK[args.sector]["script"]
    if args.bulk_url is None:
        args.bulk_url = SECTOR_BULK[args.sector]["bulk_url"]

    script_path = Path(args.script)
    dest = Path(args.dest)

    # ---- 2. Ensure bulk script present ---------------------------- #
    if not script_path.exists():
        if args.dry_run:
            # In dry-run we don't download anything.
            _emit({
                "status": "dry_run",
                "already_downloaded": 0,
                "to_download": 0,
                "requested": args.count,
                "dest": str(dest),
                "note": "bulk script absent; would fetch before downloading",
            })
            return 0
        if args.no_fetch_script:
            _emit({"status": "error", "reason": "script_absent",
                   "path": str(script_path)})
            return 2
        # Download the script.
        try:
            _fetch_bulk_script(args.bulk_url, script_path)
        except RuntimeError as exc:
            _emit({"status": "error", "reason": "script_fetch_failed",
                   "detail": str(exc)})
            return 2

    # ---- 3. Parse script (streaming generator, no full materialisation) -- #
    try:
        jobs_iter = _parse_iter_script(script_path)
    except Exception as exc:
        _emit({"status": "error", "reason": "script_parse_failed",
               "detail": str(exc)})
        return 2

    # ---- 4. Select next batch ------------------------------------- #
    # Emit a phase marker before the (muted, can take minutes on an ~8M-line
    # bulk script) skip-set query + selection scan below, so the status
    # endpoint has something to report instead of a blank "downloading".
    _emit({"status": "select_start"})

    # Skip-set = basenames already in the tglc index (fast local query). Lets a
    # re-run skip files whose local copy was uploaded to MinIO and deleted,
    # without a full-disk glob or a whole-bucket listing.
    present_keys = _indexed_basenames(args.index, args.sector)
    already, batch = select_next_jobs(jobs_iter, dest, args.count,
                                      present_keys=present_keys)

    # ---- 5. Dry-run output ---------------------------------------- #
    # Emits a single JSON object and nothing else, so callers can json.loads it.
    if args.dry_run:
        _emit({
            "status": "dry_run",
            "already_downloaded": already,
            "to_download": len(batch),
            "requested": args.count,
            "dest": str(dest),
        })
        return 0

    _emit({"status": "skip_set", "count": len(present_keys)})

    # Object-store handle for the upload step (None when object storage is off).
    # A DEDICATED instance, not the shared object_store.store() singleton:
    # this call site does a whole-chunk retry+backoff of its own
    # (_pack_and_upload_chunk) around a large (~0.5 GB/10k-file chunk)
    # multipart PUT, so it wants a longer read_timeout (a stalled MinIO part
    # would otherwise time out at the library default well before our own
    # retry loop gets a say) and botocore's OWN per-part retries turned OFF
    # (max_attempts=0 — two independent retry mechanisms stacking would just
    # add uncontrolled extra delay on top of ours). See ObjectStore.__init__'s
    # docstring and the 2026-07-21 MinIO multipart read-timeout incident.
    # Scoped to this one instance: the server's and every worker's own
    # object_store.store() singleton (FITS serving, presigned URLs, ...) is
    # untouched.
    try:
        from exonit import object_store as _os_mod
        _ostore = (_os_mod.ObjectStore(read_timeout=300, retries_max_attempts=0)
                  if _os_mod.is_enabled() else None)
    except Exception as exc:
        _emit({"status": "objectstore_unavailable", "detail": str(exc)})
        _ostore = None

    # ---- 6. Download --------------------------------------------- #
    if not batch:
        _emit({"status": "done", "downloaded": 0, "failed": 0,
               "indexed": 0, "dest": str(dest),
               "note": "nothing to download"})
        return 0

    dest.mkdir(parents=True, exist_ok=True)

    # Run stamp: every chunk derives its slice/index-list/container names from
    # it (<ts>_c001, _c002, …), so a container key still names the run it came
    # from and two concurrent runs cannot collide.
    ts = time.strftime("%Y%m%dT%H%M%S")

    # Use --progress-log if supplied by the server (so per-file progress lines
    # land in the same file the status endpoint tails); fall back to an
    # auto-generated path inside dest for standalone CLI use.
    if args.progress_log:
        log_path = Path(args.progress_log)
    else:
        log_path = dest / f"download_{ts}.log"

    # ---- 6b. Download → index → pack, ONE CHUNK AT A TIME ----------------- #
    # This used to be: download the WHOLE batch, then index it, then pack it.
    # Everything after the download was therefore all-or-nothing, and on
    # 2026-07-14 a server restart killed the process 24,747 files into a 100k
    # batch: those files were on disk, in no index, and (because the old
    # _is_present treated on-disk as done) the next run skipped them forever.
    # They were unusable — a FITS the metadata cache has never heard of cannot
    # be served, and /api/fits/<name> 404s for it.
    #
    # Chunking makes progress DURABLE: after each chunk the files are indexed,
    # packed into object storage and freed from local disk, so an interruption
    # can lose at most the chunk in flight — and even that self-heals now that
    # presence means "indexed" (the next run re-selects it, the downloader skips
    # the fetch because the file is already on disk, and it gets indexed).
    # It also caps local disk use at one chunk (~0.5 GB) instead of the whole
    # batch (~5.5 GB at 100k).
    #
    # The log the UI tails must still read as ONE download: the chunk's
    # downloader gets --progress-total/--prior-* so every line reports global
    # numbers, and --suppress-finish so the single [finish] line (which the
    # status endpoint reads as "this download completed") is written once, here,
    # at the true end.
    #
    # 2026-07-18: index+pack of chunk N now runs on a BACKGROUND THREAD while
    # the main thread downloads chunk N+1 — network (download) and CPU/upload
    # (index+pack) are disjoint resources, and the old strict alternation
    # measured 20-25% idle time (spec
    # docs/superpowers/specs/2026-07-18-download-index-pipeline.md). Only ONE
    # stage overlaps: before starting the index of N+1, the caller joins the
    # thread for N, so indexing itself stays strictly sequential (single
    # SQLite writer). Per-chunk durability is unchanged — a chunk is only
    # "done" (chunk_done emitted, eligible for the registry count) once its
    # own download+index+pack have all completed — only the WHEN of adjacent
    # chunks' phases now overlaps. If index N fails, the join() that detects
    # it happens AFTER chunk N+1's download has already completed in the
    # foreground (that download is simply left to finish; it is never
    # cancelled), but N+1 is never indexed/packed/marked done and the run
    # ends in index_failed exactly as before chunking.
    n_chunks = (len(batch) + args.chunk - 1) // args.chunk
    grand_total = len(batch)
    downloaded = failed = uploaded_total = 0
    prior_done = prior_skip = prior_fail = 0
    temp_files: list = []
    temp_files_lock = threading.Lock()
    t_start = time.monotonic()

    prev_thread = None   # background index+pack thread for the PREVIOUS chunk
    prev_result: dict = {}
    final_indexed = 0

    # ONE MetadataCache for the WHOLE run, shared across every chunk's
    # background index+pack thread (2026-07-20 review fix — see
    # _index_and_pack_chunk's docstring for the thread-safety argument and
    # why this replaces the old per-chunk construct-and-abandon pattern).
    from exonit.metadata_cache import MetadataCache
    mc = MetadataCache(args.index)

    try:
        for ci, start in enumerate(range(0, grand_total, args.chunk), 1):
            chunk = batch[start:start + args.chunk]
            cts = f"{ts}_c{ci:03d}"
            _emit({"status": "chunk_start", "chunk": ci, "of": n_chunks,
                   "size": len(chunk)})

            # --- download this chunk (foreground) ---
            # Overlaps with the PREVIOUS chunk's background index+pack thread,
            # if one is running — that is the whole point of this pipeline.
            slice_path = dest / f"_slice_{cts}.sh"
            with temp_files_lock:
                temp_files.append(slice_path)
            with open(slice_path, "w") as fh:
                fh.write("#!/bin/sh\n")
                for relpath, url in chunk:
                    fh.write(
                        f"curl -f --create-dirs --output '{relpath}' '{url}'\n"
                    )

            dl_script = str(_SCRIPTS_DIR / "download_tglc_batch.py")
            dl_cmd = [
                sys.executable, dl_script,
                "--script", str(slice_path),
                "--dest", str(dest),
                "--workers", str(args.workers),
                "--log", str(log_path),
                "--progress-total", str(grand_total),
                "--prior-done", str(prior_done),
                "--prior-skip", str(prior_skip),
                "--prior-fail", str(prior_fail),
                "--suppress-finish",
            ]
            dl_result = subprocess.run(dl_cmd, capture_output=True, text=True)
            if dl_result.returncode != 0:
                # A previous chunk's background index may still be running —
                # join it so its outcome (and any exception) is not left
                # dangling, but report THIS failure regardless of that outcome.
                if prev_thread is not None:
                    prev_thread.join()
                _emit({
                    "status": "download_failed",
                    "chunk": ci,
                    "returncode": dl_result.returncode,
                    "stderr": dl_result.stderr[:500],
                })
                return 1

            dl_out = dl_result.stdout.strip()
            c_done = _parse_count(dl_out, "done")
            c_skip = _parse_count(dl_out, "skip")
            c_fail = _parse_count(dl_out, "fail")
            prior_done += c_done
            prior_skip += c_skip
            prior_fail += c_fail
            downloaded += c_done
            failed += c_fail
            downloaded_snapshot = downloaded

            # --- indexing must stay strictly sequential: join chunk N's
            # background index+pack before starting chunk N+1's. By this
            # point chunk N+1 (ci) has ALREADY finished downloading above —
            # its download is never gated on N's index. ---
            if prev_thread is not None:
                prev_thread.join()
                if prev_result.get("error"):
                    # Chunk ci-1's index failed. This chunk's download already
                    # ran to completion (per the design above) but it must NOT
                    # be indexed, packed, marked chunk_done, or counted toward
                    # the registry/[finish] line — final state is
                    # index_failed, identical to the pre-overlap behaviour.
                    return 1
                uploaded_total = prev_result["uploaded_total"]
                final_indexed = prev_result["indexed"]

            result: dict = {}
            index_thread = threading.Thread(
                target=_index_and_pack_chunk,
                kwargs=dict(ci=ci, n_chunks=n_chunks, chunk=chunk, cts=cts,
                           ts=ts, dest=dest, args=args, ostore=_ostore,
                           downloaded_snapshot=downloaded_snapshot,
                           uploaded_before=uploaded_total, result=result,
                           temp_files=temp_files, temp_files_lock=temp_files_lock,
                           metadata_cache=mc),
                name=f"discovery-index-chunk-{ci}",
            )
            index_thread.start()
            prev_thread, prev_result = index_thread, result

        # No chunk N+1 to overlap the LAST chunk's index+pack with — join it.
        if prev_thread is not None:
            prev_thread.join()
            if prev_result.get("error"):
                return 1
            uploaded_total = prev_result["uploaded_total"]
            final_indexed = prev_result["indexed"]

        indexed = final_indexed
        uploaded = uploaded_total

        # Registry 'downloaded' = size of the tglc ∪ fits_blobs basename union
        # for this sector.  Accurate, sector-scoped, and correct regardless of
        # where the FITS physically live (local or MinIO).  Using the union
        # rather than bare tglc count avoids a phantom "pending" residual caused
        # by Gaia-ID duplicates whose basenames tglc's INSERT OR REPLACE
        # discarded but fits_blobs retained.
        # Once, at the end: the union query is O(rows for this sector) (~3M) and
        # running it per chunk would cost more than the download itself. An
        # interrupted run just leaves the counter stale, and the index — which
        # is the actual source of truth — is already correct.
        # Guarded (2026-07-25 fix): skip the write, loudly, when --registry
        # was left at its default AND this does not look like the server
        # host — this exact combination is what silently updated the wrong
        # host's sectors.json for sector 18. Never skips the download or
        # indexing above, only this cosmetic counter (the server-side
        # /api/discovery/sectors endpoint now derives 'downloaded' from the
        # index directly, so this registry write is a fallback display value,
        # not the source of truth).
        if _guard_registry_write(args.registry, _registry_is_default):
            _update_registry(args.registry, args.sector,
                             len(_indexed_basenames(args.index, args.sector)))

        # The ONE [finish] line, written only on the real completion path. The
        # status endpoint reads it as "this download completed"; a per-chunk one
        # would report a run that died at chunk 2 of 10 as a clean success.
        with open(log_path, "a") as _fh:
            _fh.write(
                f"[finish] done={prior_done} skip={prior_skip} "
                f"fail={prior_fail} in {(time.monotonic()-t_start)/60:.1f}min\n")

    finally:
        # Clean up every chunk's slice script and index-list.
        for p in temp_files:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        mc.close()

    _emit({
        "status": "done",
        "downloaded": downloaded,
        "failed": failed,
        "indexed": indexed,
        "uploaded": uploaded,
        "dest": str(dest),
    })
    return 0


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _parse_count(text: str, key: str) -> int:
    """Parse key=N from download_tglc_batch stdout. Returns 0 if missing."""
    for part in text.split():
        if part.startswith(key + "="):
            try:
                return int(part.split("=", 1)[1])
            except ValueError:
                pass
    return 0


def _parse_indexed(text: str) -> int:
    """Parse 'index: N (tic,sector) rows' from index_tglc_batch stdout."""
    for line in text.splitlines():
        if line.startswith("index:") and "(tic,sector) rows" in line:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    pass
    return 0


def _parse_files_total(text: str) -> int:
    """Parse 'files_total=N' from index_tglc_batch stdout."""
    for part in text.split():
        if part.startswith("files_total="):
            try:
                return int(part.split("=", 1)[1])
            except ValueError:
                pass
    return 0


def _indexed_basenames(index_path: str, sector: int) -> set:
    """Basenames of FITS already downloaded+indexed for *sector*.

    The skip-set is the UNION of:
    - ``tglc`` index rows (one row per TIC via INSERT OR REPLACE; when multiple
      Gaia IDs map to the same TIC only the last survives, losing the other
      basenames from the index).
    - ``fits_blobs`` rows (one row per physical FITS basename, PRIMARY KEY on
      filename; preserves every basename that was actually packed, including
      those lost by a tglc REPLACE collision).

    Using the union ensures a re-run never re-downloads either category, and
    the ``downloaded`` registry counter reflects all FITS that have been packed
    into object storage — not only the surviving tglc rows.

    Thin wrapper over ``MetadataCache.indexed_basenames()`` (the
    docs/superpowers/specs/2026-07-20-tglc-index-to-pg.md seam) — kept as a
    module-level function (name + signature unchanged: ``index_path``,
    ``sector``, still a plain SQLite file path or DSN string, never an
    already-open connection) because the tests in
    tests/test_discovery_download.py import it directly. Backend selection
    (SQLite vs PostgreSQL) happens inside ``MetadataCache.__init__`` via
    ``EXONIT_METADATA_DB_URL`` — this function does not need to know which
    one is active. Guard semantics preserved exactly: a missing ``tglc`` or
    ``fits_blobs`` table (or a nonexistent SQLite file — the "no index yet"
    case, e.g. a fresh deployment) degrades to an empty/tglc-only set; a
    genuine backend failure (PG down, ...) still propagates (see
    ``MetadataCache._missing_table_excs``'s docstring — this is the exact
    skip-set the download hot path trusts to decide "already indexed").
    """
    from exonit.metadata_cache import MetadataCache
    cache = MetadataCache(index_path)
    try:
        return cache.indexed_basenames(sector)
    finally:
        cache.close()


def _fingerprint_state_path() -> Path:
    """Where --recompute-registry persists the last-seen tglc/fits_blobs
    fingerprint (see MetadataCache.download_registry_fingerprint). A plain
    file, not Redis: this is a standalone CLI invocation (the cron job),
    with no shared_state singleton to reuse and no other process racing it
    (the cron's own flock already serialises runs). Overridable for test
    isolation, same convention as EXONIT_PIPELINE_STATE_FILE elsewhere."""
    return Path(os.environ.get(
        "EXONIT_DOWNLOAD_REGISTRY_FINGERPRINT_FILE",
        str(_PROJECT_ROOT / "data" / "discovery"
            / "download_registry_fingerprint.json")))


def _load_fingerprint_state() -> dict:
    """{'fingerprint': {...}, 'last_full_recompute': epoch}, or {} if the
    file is absent or corrupt. An empty dict never equals a real fingerprint
    and has last_full_recompute=0, so this degrades to "always recompute" —
    the same safe direction as a fingerprint of None."""
    try:
        return json.loads(_fingerprint_state_path().read_text())
    except (OSError, ValueError):
        return {}


def _save_fingerprint_state(fingerprint: dict, last_full_recompute: float) -> None:
    p = _fingerprint_state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "fingerprint": fingerprint,
            "last_full_recompute": last_full_recompute,
        }))
    except OSError as exc:
        # Best-effort: a failed write just means the NEXT invocation also
        # recomputes (safe direction) instead of wrongly skipping.
        print(f"[recompute-registry] failed to persist fingerprint state: {exc}",
              file=sys.stderr, flush=True)


def _is_server_host() -> bool:
    """True iff the production API server (``exonit.service``) is active here.

    2026-07-25 sector 18 incident: this download ran on <host> with
    ``--registry`` left at its default, which resolved to a real, existing
    ``sectors.json`` — just the WRONG HOST's copy (the server has lived on
    <host> since 2026-07-24; <host> has its own separate clone at the
    same path). The write silently succeeded and updated a file nobody
    reads. Guarding on a hardcoded hostname/IP would only defer the same
    bug to the next migration (this is the SECOND server host — <host>
    -> <host> — and CLAUDE.md's history shows infra keeps moving). Checking
    the actual signal instead — "is the systemd unit that owns
    ``data/discovery/sectors.json`` on THIS host active right now" — is
    self-adjusting: it keeps working across any future move without a code
    change, because it asks reality instead of a memorized address.
    """
    try:
        return subprocess.run(
            ["systemctl", "is-active", "--quiet", "exonit.service"],
            timeout=5,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        # systemctl unavailable (non-Linux host, sandboxed CI/test env, a
        # container without systemd, ...) — inconclusive, not proof this IS
        # the server. Fail toward "not the server": the safe direction here
        # is a loud warning, never a silent write to a file nobody reads.
        return False


def _guard_registry_write(registry_path: str, registry_is_default: bool) -> bool:
    """Return True iff it is safe to write ``registry_path`` now.

    Only refuses when BOTH are true:
    - ``registry_is_default`` — the caller left ``--registry`` unset and is
      relying on the argparse default, i.e. did not consciously choose this
      path (an explicit ``--registry`` — including every test in
      tests/test_discovery_download.py and every server-spawned subprocess
      call, see exonit/routes/discovery.py's ``"--registry", _DISCOVERY_REGISTRY``
      call sites — always proceeds: the caller took responsibility for the
      path, which is exactly what a shared network location or a deliberate
      one-off recompute would look like).
    - ``not _is_server_host()`` — and this host does not appear to be the
      one the server actually reads that default path on.

    This is the narrow, targeted fix for the exact failure mode that caused
    the 2026-07-25 incident (a bare CLI invocation on a worker host,
    default registry path, silently updating a copy the server never
    serves) without touching any caller that already knows what it is
    doing.
    """
    if not registry_is_default or _is_server_host():
        return True
    msg = (
        f"[registry-guard] refusing to write '{registry_path}': this host "
        "does not appear to be running the production server "
        "(exonit.service is not active) and --registry was left at its "
        "default. Writing here would update a registry copy the server "
        "never reads (see the 2026-07-25 sector 18 incident this guard "
        "exists to prevent recurring) -- pass --registry explicitly if "
        "you really mean to write to this file, or run this from the "
        "server host."
    )
    print(msg, file=sys.stderr, flush=True)
    _emit({"status": "registry_write_skipped_wrong_host",
           "registry": registry_path})
    return False


def _update_registry(registry_path: str, sector: int, files_on_disk: int) -> None:
    """Set the downloaded count for *sector* in the registry."""
    data = load_registry(registry_path)
    sectors = data.setdefault("sectors", {})
    entry = sectors.setdefault(str(sector), {})
    entry["downloaded"] = files_on_disk
    save_registry(registry_path, data)


if __name__ == "__main__":
    sys.exit(main())
