"""Ensure a TIC has a MetadataCache entry, indexing cached FITS on demand.

TOI targets are pre-indexed by the historical ingest path. Discovery (non-TOI)
targets are not, so enqueue would skip them. This module creates the add_entry
(+ fits_paths) for a single TIC from cached QLP (FFI) or SPOC light curves, so
the enqueue path works for discovery searches with no manual pre-indexing.

Idempotent and cheap: globs only the target TIC's files, never scans the whole
cache. Returns True if the TIC has metadata (already, or now), False if no data.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .config import config as _cfg
from .metadata_backend import is_pg_url

logger = logging.getLogger(__name__)

_QLP_RE = re.compile(r"hlsp_qlp_tess_ffi_s(\d{4})-\d{16}_tess")
_SPOC_RE = re.compile(r"-s(\d{4})-(\d{16})-(\d+)-s_lc\.fits$")

DEFAULT_FFI_CACHE = "data/ffi_cache"
DEFAULT_SPOC_ROOT = os.path.join(_cfg.nfs_cache_path, "mastDownload", "TESS")
# Authoritative TGLC index: the `tglc` table inside metadata_cache.db. The
# legacy standalone file data/discovery/tglc_cache/tglc_index.db was stranded
# at 4,973 TICs while the indexing scripts (and load_extended_canonical) moved
# to metadata_cache.db.tglc (34,780+). Pointing the lookup at the same place
# the indexers write to removes the drift class of bug — see the 2026-05-28
# Start Scan 10k incident where the QLP recursive glob fell through for every
# extension TIC because _tglc_lookup couldn't see them.
DEFAULT_TGLC_INDEX = "data/metadata_cache.db"
_QLP_EXPTIME = 200.0  # representative FFI cadence (same as existing QLP entries)

# Process-wide MetadataCache instances for _tglc_lookup, keyed by the
# tglc_index string each caller passes. 2026-07-20 TGLC→PG migration (F1):
# _tglc_lookup used to open a fresh, unpooled sqlite3.connect() per call (the
# enqueue "need_index" tail can call this once per not-yet-seen TIC — see
# docs/superpowers/specs/2026-07-20-tglc-index-to-pg.md §5). Routing through
# MetadataCache (exonit/metadata_backend.py's seam) instead means the SQL
# lives in one place, but MetadataCache.__init__ does real work (schema
# self-provisioning); this cache avoids paying that cost on every call, same
# pattern as exonit/server.py::_get_shared_metadata_cache.
_shared_tglc_caches: dict = {}
_shared_tglc_caches_lock = threading.Lock()


def _get_tglc_cache(tglc_index: str):
    from .metadata_cache import MetadataCache
    mc = _shared_tglc_caches.get(tglc_index)
    if mc is not None:
        return mc
    with _shared_tglc_caches_lock:
        mc = _shared_tglc_caches.get(tglc_index)
        if mc is None:
            mc = MetadataCache(tglc_index)
            _shared_tglc_caches[tglc_index] = mc
    return mc


def _tglc_rows_to_fields(rows):
    """Turn raw ``tglc`` rows (dicts with sector/path/exptime/tmag/ra/dec)
    into ``(sectors, exptime, [(basename, abspath)], tmag, ra, dec)``, or
    ``None`` if no row has a non-empty path.

    Shared core of ``_tglc_lookup`` (single-TIC path, called with whatever
    row order the query planner picks) and ``ensure_metadata_bulk`` (bulk
    path, called with rows pre-sorted by
    ``MetadataCache.tglc_lookup_bulk``'s ``ORDER BY tic_id, sector, path``) —
    extracted verbatim from the pre-2026-09-26 body of ``_tglc_lookup`` so
    both call sites derive sectors/fits_files/exptime/tmag/ra/dec
    identically.
    """
    sectors, files, exptime = set(), [], None
    tmag = ra = dec = None
    for row in rows:
        sector, path, exp = row["sector"], row["path"], row["exptime"]
        tm, r, d = row["tmag"], row["ra"], row["dec"]
        # FITS live in MinIO since the 2026-06-24 migration; the local file at
        # `path` was deleted after upload to the object store. Workers fetch by
        # basename via the server (/api/fits → MinIO presigned URL), so do NOT
        # gate on local existence. The old `os.path.exists(path)` check made
        # _tglc_lookup return None for EVERY TGLC target post-migration, which
        # sent the enqueue into the catastrophic 1.3M-file ffi_cache glob
        # fallback (~minutes/TIC) and hung the whole run at ENQUEUEING.
        if not path:
            continue
        sectors.add(int(sector))
        # abspath, NOT resolve(): resolve() does a syscall per path component
        # (realpath/lstat) to follow symlinks, and with the tglc_cache
        # dentry cache cold that's a disk hit per file per TIC (15M entries
        # in data/discovery/tglc_cache — 300k-TIC enqueue went ~3min→~40min,
        # py-spy showed MainThread stuck in _joinrealpath). The file is also
        # gone from local disk post-MinIO-migration (see comment above), so
        # there is nothing on-disk to resolve anyway. abspath is pure string
        # manipulation (cwd-join + normalize), zero I/O.
        files.append((os.path.basename(path), os.path.abspath(path)))
        exptime = float(exp) if exp else exptime
        tmag = tmag if tmag is not None else tm
        ra = ra if ra is not None else r
        dec = dec if dec is not None else d
    if not files:
        return None
    return sorted(sectors), (exptime or 1800.0), files, tmag, ra, dec


def _tglc_lookup(tic, tglc_index):
    """Return (sectors, exptime, [(basename, abspath)], tmag, ra, dec) for a TIC
    from the TGLC index db, or None. TGLC files are named by Gaia DR3 id, so the
    TIC→file mapping lives in an index built by scripts/index_tglc_batch.py.

    Routed through exonit/metadata_cache.py's MetadataCache.tglc_lookup (the
    metadata_backend.py seam) rather than a raw sqlite3.connect — see
    docs/superpowers/specs/2026-07-20-tglc-index-to-pg.md D3/item (e). Behaviour
    is unchanged on SQLite (EXONIT_METADATA_DB_URL unset, the default).
    """
    if not tglc_index:
        return None
    # The file-existence guard only makes sense for a literal SQLite path.
    # When EXONIT_METADATA_DB_URL is a PG DSN, MetadataCache uses PG
    # regardless of what `tglc_index` string is passed (see its docstring),
    # so skip the local-file check in that case rather than incorrectly
    # short-circuiting on a SQLite fallback path that may not exist locally.
    _pg_override = os.environ.get("EXONIT_METADATA_DB_URL")
    if (not is_pg_url(_pg_override) and not is_pg_url(tglc_index)
            and not os.path.exists(tglc_index)):
        return None
    mc = _get_tglc_cache(tglc_index)
    rows = mc.tglc_lookup(str(tic))
    if not rows:
        return None
    return _tglc_rows_to_fields(rows)


def _is_tglc_source(fits_files) -> bool:
    """True if a ``metadata`` row's ``fits_files`` is empty or every entry is
    a TGLC (``hlsp_tglc*``) basename — i.e. this row's source is TGLC (or
    not yet populated), so it is eligible for the enqueue-time refresh (spec
    2026-09-26, section 2.1: "source rule"). A row with even one non-TGLC
    (SPOC/QLP) basename is a pre-indexed TOI row — the "TOI short-circuit
    preserved" invariant — and is NEVER refreshed, whatever ``refresh``
    says.
    """
    if not fits_files:
        return True
    return all(str(f).startswith("hlsp_tglc") for f in fits_files)


def _glob_fallback(tic: str, ffi_cache: str, spoc_root: str):
    """QLP (FFI) then SPOC glob fallback for a TIC absent from the TGLC index.

    Ported verbatim from ``ensure_metadata``'s pre-2026-09-26 tail (docs/
    superpowers/specs/2026-09-26-metadata-refresh-and-bulk-enqueue-design.md
    section 2.2, step 3): the recursive glob over ``ffi_cache`` (QLP), then
    the ``spoc_root`` directory scan, only reached when neither the TGLC
    index nor an existing ``metadata`` row has this TIC. Shared by
    ``ensure_metadata`` (via ``ensure_metadata_bulk``) and
    ``ensure_metadata_bulk``'s own "no row, no tglc" branch.

    Returns ``(sectors_sorted, exptime, fits_files, fits_rows)``, or ``None``
    if nothing is found in either cache.
    """
    pad = f"{int(tic):016d}"
    fits_rows = []
    sectors = set()
    fits_files = []
    exptime = _QLP_EXPTIME

    # QLP (FFI) — single-sector files named hlsp_qlp_..._s{NNNN}-{TIC16}_...
    for p in glob.glob(os.path.join(ffi_cache, "**", f"*{pad}*.fits"), recursive=True):
        bn = os.path.basename(p)
        m = _QLP_RE.search(bn)
        if not m or os.path.getsize(p) < 30_000:
            continue
        sectors.add(int(m.group(1)))
        fits_files.append(bn)
        fits_rows.append((bn, str(Path(p).resolve())))

    # SPOC fallback — only if no QLP found
    if not fits_files and os.path.isdir(spoc_root):
        for d in os.listdir(spoc_root):
            if pad not in d:
                continue
            dd = os.path.join(spoc_root, d)
            if not os.path.isdir(dd):
                continue
            for f in os.listdir(dd):
                m = _SPOC_RE.search(f)
                if m and str(int(m.group(2))) == tic:
                    sectors.add(int(m.group(1)))
                    fits_files.append(f)
                    fits_rows.append((f, str(Path(dd, f).resolve())))
                    exptime = float(min(exptime, int(m.group(3))))

    if not fits_files:
        return None
    return sorted(sectors), exptime, fits_files, fits_rows


def ensure_metadata(
    tic_id,
    cache_path: Optional[str] = None,
    ffi_cache: str = DEFAULT_FFI_CACHE,
    spoc_root: str = DEFAULT_SPOC_ROOT,
    tglc_index: str = DEFAULT_TGLC_INDEX,
    mc=None,
) -> bool:
    """Index `tic_id` into the MetadataCache from cached FITS if not present.

    Returns True if the TIC has a metadata entry afterwards, else False.

    Parameters
    ----------
    mc:
        An existing ``MetadataCache`` instance to reuse.  When provided,
        ``cache_path`` is ignored (the caller is responsible for opening the
        correct DB).  Passing a shared instance avoids the per-call connect /
        PRAGMA overhead that made the pre-filter O(N×0.33 ms) at 200k targets.
        If None (default), a new MetadataCache is created from ``cache_path``
        (original behaviour — fully backwards-compatible).

    Delegates to :func:`ensure_metadata_bulk` (``refresh=False``, a single-TIC
    chunk) — docs/superpowers/specs/2026-09-26-metadata-refresh-and-bulk-
    enqueue-design.md §2.2: "ensure_metadata(tic) is kept ... and now
    delegates to ensure_metadata_bulk([tic], refresh=False) to guarantee
    parity by construction". Behaviourally identical to the pre-2026-09-26
    body (same create-from-TGLC / QLP-glob / SPOC-glob precedence, same
    already-indexed short-circuit for an existing row of ANY source) except
    that a freshly-created TGLC row's ``fits_files`` order is now the
    deterministic ``ORDER BY sector, path`` from
    ``MetadataCache.tglc_lookup_bulk`` instead of whatever order the query
    planner happened to return — a deliberate, spec-approved determinism fix
    (§2.2: "Note: today the order is the planner's"), not a behaviour
    regression: no caller of ``ensure_metadata`` depended on that order.
    """
    from .config import ExoNitConfig
    from .metadata_cache import MetadataCache

    if mc is None:
        if cache_path is None:
            cache_path = ExoNitConfig().metadata_cache_path
        mc = MetadataCache(cache_path)
    tic = str(int(tic_id))
    result = ensure_metadata_bulk(
        [tic], mc, tglc_index=tglc_index, ffi_cache=ffi_cache,
        spoc_root=spoc_root, refresh=False)
    return tic in result.ready


@dataclass
class BulkMetadataResult:
    """Outcome of a :func:`ensure_metadata_bulk` call.

    ``ready``: TICs that have a ``metadata`` row afterwards (created,
    refreshed, preserved, or already up to date) — the caller's "safe to
    enqueue" set. ``no_fits``: TICs with no data in TGLC, QLP, or SPOC — the
    caller marks these ``no_fits`` in ``stars`` and skips them. The four
    counters are the enqueue summary's ``metadata_created`` /
    ``metadata_refreshed`` / ``metadata_unchanged`` /
    ``metadata_preserved_non_tglc`` fields (spec §2.3 — the refresh's
    per-lot impact metric).
    """

    ready: List[str] = field(default_factory=list)
    no_fits: List[str] = field(default_factory=list)
    created: int = 0
    refreshed: int = 0
    unchanged: int = 0
    preserved_non_tglc: int = 0


def _process_chunk(chunk, mc, *, ffi_cache, spoc_root, refresh, dry_run,
                   chunk_size) -> BulkMetadataResult:
    """Process ONE chunk of TIC ids against `mc`'s bulk read/write APIs.

    Fault isolation, two layers (2026-09-26 review fix):

    - A single TIC's :func:`_glob_fallback` call is wrapped HERE, per TIC —
      a filesystem race/permission error walking the QLP/SPOC glob tree for
      one TIC is caught, logged with the TIC id, and demoted to ``no_fits``
      for that TIC only. Mirrors the pre-bulk ``ensure_metadata`` per-TIC
      ``try/except`` in the old ``enqueue_targets`` loop (now
      ``_enqueue_targets_legacy``) — "a transient failure here cascades into
      a permanent no_fits+skipped mark for THIS TIC", not the whole lot.
    - ``mc.get_entries_bulk``/``mc.tglc_lookup_bulk`` (reads) and
      ``mc.write_metadata_bulk`` (write) are NOT caught here — they
      propagate to :func:`ensure_metadata_bulk`, whose caller falls back to
      calling this same function per TIC (see there) on any read/write
      failure for the chunk.
    """
    result = BulkMetadataResult()
    entries = mc.get_entries_bulk(chunk, chunk_size=chunk_size)
    tglc_by_tic = mc.tglc_lookup_bulk(chunk, chunk_size=chunk_size)

    metadata_rows = []
    fits_path_rows = []

    for tic in chunk:
        entry = entries.get(tic)
        raw_tglc_rows = tglc_by_tic.get(tic)
        tglc_fields = (_tglc_rows_to_fields(raw_tglc_rows)
                      if raw_tglc_rows else None)

        if entry is None:
            if tglc_fields is not None:
                t_sectors, t_exptime, t_rows, t_tmag, t_ra, t_dec = tglc_fields
                metadata_rows.append((
                    tic, json.dumps(t_sectors), float(t_exptime),
                    json.dumps([bn for bn, _ in t_rows]),
                    None, None, None, t_tmag, t_ra, t_dec,
                ))
                fits_path_rows.extend(t_rows)
                result.created += 1
                result.ready.append(tic)
                continue

            try:
                fb = _glob_fallback(tic, ffi_cache, spoc_root)
            except Exception:
                logger.warning(
                    "ensure_metadata_bulk: _glob_fallback failed for TIC "
                    "%s — marking no_fits (per-TIC fault isolation, same "
                    "as the pre-bulk ensure_metadata loop)",
                    tic, exc_info=True)
                result.no_fits.append(tic)
                continue
            if fb is None:
                result.no_fits.append(tic)
                continue
            sectors, exptime, fits_files, fits_rows = fb
            metadata_rows.append((
                tic, json.dumps(sectors), float(exptime),
                json.dumps(fits_files),
                None, None, None, None, None, None,
            ))
            fits_path_rows.extend(fits_rows)
            result.created += 1
            result.ready.append(tic)
            continue

        # Existing row — TOI short-circuit: a non-TGLC source is never
        # touched, regardless of `refresh`.
        if not _is_tglc_source(entry["fits_files"]):
            result.preserved_non_tglc += 1
            result.ready.append(tic)
            continue

        if refresh and tglc_fields is not None:
            t_sectors, t_exptime, t_rows, t_tmag, t_ra, t_dec = tglc_fields
            if set(t_sectors) - set(entry["sectors"]):
                metadata_rows.append((
                    tic, json.dumps(t_sectors), float(t_exptime),
                    json.dumps([bn for bn, _ in t_rows]),
                    entry["stellar_radius"], entry["teff"], entry["logg"],
                    t_tmag if t_tmag is not None else entry["tmag"],
                    t_ra if t_ra is not None else entry["ra"],
                    t_dec if t_dec is not None else entry["dec"],
                ))
                fits_path_rows.extend(t_rows)
                result.refreshed += 1
                result.ready.append(tic)
                continue

        result.unchanged += 1
        result.ready.append(tic)

    if not dry_run and (metadata_rows or fits_path_rows):
        mc.write_metadata_bulk(metadata_rows, fits_path_rows)  # may raise

    return result


def _merge_bulk_result(into: BulkMetadataResult, other: BulkMetadataResult) -> None:
    into.ready.extend(other.ready)
    into.no_fits.extend(other.no_fits)
    into.created += other.created
    into.refreshed += other.refreshed
    into.unchanged += other.unchanged
    into.preserved_non_tglc += other.preserved_non_tglc


def ensure_metadata_bulk(
    tic_ids,
    mc,
    *,
    tglc_index: str = DEFAULT_TGLC_INDEX,
    ffi_cache: str = DEFAULT_FFI_CACHE,
    spoc_root: str = DEFAULT_SPOC_ROOT,
    refresh: bool = True,
    chunk_size: int = 5000,
    dry_run: bool = False,
    progress=None,
) -> BulkMetadataResult:
    """Bulk ``ensure_metadata``: create/refresh/preserve ``metadata`` rows for
    *tic_ids* in chunks of *chunk_size*, with ONE ``IN (...)`` query per table
    per chunk instead of one ``get_entry`` + one ``tglc_lookup`` per TIC.

    Implements docs/superpowers/specs/2026-09-26-metadata-refresh-and-bulk-
    enqueue-design.md §2.2. Per chunk (:func:`_process_chunk`):

    1. ``mc.get_entries_bulk(chunk)`` — existing ``metadata`` rows.
    2. ``mc.tglc_lookup_bulk(chunk)`` — ``tglc`` rows, deterministically
       ordered (``ORDER BY tic_id, sector, path``).
    3. Per TIC:
       - no row, TGLC available → **create** from TGLC.
       - no row, no TGLC → QLP/SPOC glob fallback (:func:`_glob_fallback`);
         nothing found → ``no_fits``.
       - existing row, non-TGLC source (:func:`_is_tglc_source` False) →
         **preserve** untouched (the TOI short-circuit — never refreshed).
       - existing TGLC-source row, ``refresh=True``, and TGLC has a sector
         the row doesn't → **refresh**: new ``sectors``/``fits_files``/
         ``exptime``/``tmag``/``ra``/``dec`` from TGLC, but
         ``stellar_radius``/``teff``/``logg`` are CARRIED OVER from the old
         row (TGLC never populates them), and ``tmag``/``ra``/``dec`` fall
         back to the old value when TGLC's is ``None`` — same "preserve
         cross-source enrichment, refresh the rest" pattern as
         ``scripts/toi_bench_force_tglc.py``.
       - otherwise → **unchanged** (no write).
    4. One write per chunk (``mc.write_metadata_bulk`` — a single
       ``Backend.transaction()`` for both the ``metadata`` and
       ``fits_paths`` ``executemany``s), skipped entirely in ``dry_run``
       mode or when the chunk has nothing to write.
    5. ``progress.report(done)`` after each chunk, if *progress* is given
       (a no-op :class:`exonit.progress_client.ProgressReporter` when no run
       is live — see that class's docstring).

    Fault isolation (2026-09-26 review fix — "one bad TIC or a PG blip must
    never abort the whole lot", matching the pre-bulk per-TIC
    ``try/except`` around ``ensure_metadata()``):

    - A single TIC's QLP/SPOC glob fallback failing is caught inside
      :func:`_process_chunk` and demoted to ``no_fits`` for that TIC only —
      the rest of the chunk is unaffected, no chunk-level fallback needed.
    - If the chunk's BULK READS (``get_entries_bulk``/``tglc_lookup_bulk``)
      or its BULK WRITE (``write_metadata_bulk``) raise — a transient PG
      blip, a malformed row, a pool timeout — this function catches it,
      logs the chunk that failed, and falls back to calling
      :func:`_process_chunk` ONE TIC AT A TIME for that chunk only. Each of
      those per-TIC calls is ALSO wrapped individually: if a specific TIC
      still fails even alone (a persistently bad row, not a transient
      blip), it is logged with its id and marked ``no_fits`` — exactly the
      pre-bulk loop's fault-isolation contract — while every OTHER TIC in
      the chunk, and every OTHER chunk in the lot, still completes normally.

    Determinism (idempotence): with step 2's ``ORDER BY``, two calls over
    unchanged ``tglc``/``metadata`` content produce byte-identical
    ``fits_files`` and make ZERO writes on the second call (every TIC lands
    in ``unchanged``/``preserved_non_tglc``).

    *tic_ids* order is preserved in ``ready``/``no_fits`` (first-occurrence
    dedup), so callers that care about relative order (none currently do)
    get it for free.
    """
    if not os.path.isdir(spoc_root):
        logger.warning(
            "ensure_metadata_bulk: SPOC root %r is not a directory — the SPOC "
            "glob fallback finds nothing (set EXONIT_NFS_CACHE to the SPOC "
            "FITS cache)", spoc_root)
    ids: List[str] = []
    seen = set()
    for t in tic_ids:
        s = str(int(t))
        if s not in seen:
            seen.add(s)
            ids.append(s)

    result = BulkMetadataResult()
    done = 0

    for chunk_start in range(0, len(ids), chunk_size):
        chunk = ids[chunk_start:chunk_start + chunk_size]
        try:
            chunk_result = _process_chunk(
                chunk, mc, ffi_cache=ffi_cache, spoc_root=spoc_root,
                refresh=refresh, dry_run=dry_run, chunk_size=chunk_size)
        except Exception:
            logger.exception(
                "ensure_metadata_bulk: bulk chunk [%d:%d] (%d TICs) failed "
                "(read or write) — falling back to per-TIC processing for "
                "this chunk only; the rest of the lot is unaffected",
                chunk_start, chunk_start + len(chunk), len(chunk))
            chunk_result = BulkMetadataResult()
            for tic in chunk:
                try:
                    one = _process_chunk(
                        [tic], mc, ffi_cache=ffi_cache, spoc_root=spoc_root,
                        refresh=refresh, dry_run=dry_run, chunk_size=1)
                except Exception:
                    logger.warning(
                        "ensure_metadata_bulk: per-TIC fallback failed for "
                        "TIC %s — marking no_fits (fault isolation)",
                        tic, exc_info=True)
                    chunk_result.no_fits.append(tic)
                    continue
                _merge_bulk_result(chunk_result, one)

        _merge_bulk_result(result, chunk_result)

        done += len(chunk)
        if progress is not None:
            progress.report(done)

    return result
