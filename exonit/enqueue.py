"""Standalone enqueue module — posts targets to the ExoNit server queue.

Extracted from runner.py:run_batch_distributed() to allow lightweight
enqueue without importing any algorithmic modules (BLS, CNN, etc.).

Only depends on: requests, exonit.config, exonit.metadata_cache.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests

import exonit
from exonit.config import ExoNitConfig, worker_token
from exonit.metadata_cache import MetadataCache
from exonit.metadata_index import ensure_metadata, ensure_metadata_bulk
from exonit.progress_client import ProgressReporter

logger = logging.getLogger(__name__)
_DEFAULT_SERVER_URL = "http://localhost:8400"
_TARGET_INFO_KEYS = ("tmag", "ra", "dec", "stellar_radius")

# Maximum TICs per bulk POST.  Keeps individual request bodies small and
# lets the caller observe partial progress for very large catalogs.
#
# Q5 (perf review 2026-09-26, docs/reviews/2026-09-26-enqueue-path-
# performance-review.md): 5000 -> 20000 (61 chunks -> ~16 for a 300k-TIC
# lot). Verified safe against PostgreSQL's 65535-parameter-per-statement
# limit: the server's PG insert path (Q4, exonit/routes/workers.py's
# _insert_tasks_bulk_pg) writes via `INSERT ... SELECT FROM unnest(...)`,
# which takes 3-4 ARRAY params TOTAL regardless of chunk size, not 3-4
# params PER ROW — a 20000-row chunk is nowhere near any per-statement
# param limit on either backend (the dedup SELECT and SQLite's executemany
# both already chunk/loop internally at their own, smaller, safe
# boundaries — see worker_enqueue_bulk's _CHUNK=10000 IN(...) probe).
_BULK_CHUNK_SIZE = 20000

# Q5: chunk POST retries. The server's dedup (tasks-row check, plus
# done_state when EXONIT_DONE_STATE=1) makes a retried chunk idempotent —
# TICs already inserted by an earlier, dropped-response attempt are simply
# reported back as "skipped", not double-enqueued. Today a single network
# blip (timeout, connection reset) on any one of the ~16-60 chunks in a lot
# permanently lost that whole chunk (counted as "failed" with the enqueue
# cursor already advanced past it — see discovery_run.py). A couple of
# short retries turns a transient blip into a few extra seconds instead of
# a hole in the lot.
_BULK_POST_MAX_RETRIES = 2
_BULK_POST_RETRY_BACKOFF_S = 1.0

# Zero-valued shape of the metadata-bulk-refresh summary fields (docs/
# superpowers/specs/2026-09-26-metadata-refresh-and-bulk-enqueue-design.md
# §2.3). Both `_enqueue_targets_legacy` (V2 off — nothing bulk-related ever
# runs) and `_enqueue_targets_v2` (V2 on, but this particular call had no
# candidate TICs to process) return this SAME dict shape, so `summary`'s
# `metadata_created`/`metadata_refreshed`/`metadata_unchanged`/
# `metadata_preserved_non_tglc` keys are ALWAYS present regardless of flags
# or outcome — 2026-09-26 review fix (a bare `{}` on the legacy path meant
# these keys were silently absent whenever EXONIT_ENQUEUE_METADATA_V2=0, the
# default; operator decision §8.5 makes `metadata_refreshed` a mandatory
# dossier field, so a future consumer must never see it missing).
_ZERO_METADATA_COUNTS = {
    "metadata_created": 0, "metadata_refreshed": 0,
    "metadata_unchanged": 0, "metadata_preserved_non_tglc": 0,
}


def _has_metadata(tic_id: str, cache_path: Optional[str] = None) -> bool:
    """Check if *tic_id* has an entry in the metadata cache."""
    if cache_path is None:
        cfg = ExoNitConfig()
        cache_path = cfg.metadata_cache_path
    cache = MetadataCache(cache_path)
    entry = cache.get_entry(str(tic_id))
    return entry is not None


def _probe_bulk_endpoint(server_url: str, token: str) -> bool:
    """Return True if the server supports /api/worker/enqueue_bulk.

    Sends an empty-list POST and expects either 200 (success, enqueued=0) or
    400 (bad-request for empty list).  A 404 means the endpoint does not exist
    (old server) → fall back to per-target POSTs.  Any other network failure
    is treated as "not supported" to keep the fallback safe.
    """
    try:
        resp = requests.post(
            f"{server_url}/api/worker/enqueue_bulk",
            json={"targets": []},
            timeout=5,
            headers={"Authorization": f"Bearer {token}"},
        )
        return resp.status_code != 404
    except Exception:
        return False


def _post_bulk(
    tics: List[str],
    server_url: str,
    token: str,
    force_rescan: bool = False,
    max_workers: int = 8,
    sector: Optional[int] = None,
    progress: Optional[ProgressReporter] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """POST *tics* to the server — bulk endpoint preferred, per-target fallback.

    Bulk path (server >= this commit):
      Sent in chunks of _BULK_CHUNK_SIZE via ``POST /api/worker/enqueue_bulk``,
      which inserts all rows in a single SQL transaction. This reduces HTTP
      round-trips from O(N) to O(ceil(N / chunk_size)) and avoids the
      120/600 s subprocess timeout that blew up at ~1.5k TICs.

    Fallback path (old server, 404 on /enqueue_bulk):
      Falls back to the original per-target POST loop via
      ``/api/worker/enqueue`` using a ThreadPoolExecutor.

    force_rescan:
        Deep-rescan mode (spec 2026-07-23). Forwarded as ``"force_rescan":
        true`` in the bulk payload when True, so the server's own task-row
        dedup (Task 4) can be told to re-enqueue already-completed TICs.
        Not forwarded on the legacy per-target fallback path. NEVER default
        anywhere.

    sector:
        The lot's TESS sector (done-state design, 2026-09-26). Forwarded as
        ``"sector": S`` in the bulk payload so ``tasks.sector`` is populated
        at insert time and the server's done-state dedup (behind
        ``EXONIT_DONE_STATE``) can evaluate exhausted(t, S). Not forwarded on
        the legacy per-target fallback path (that endpoint reads ``sector``
        too — see routes/workers.py's ``worker_enqueue`` — but this function
        never used it there either; low-traffic fallback, out of scope).

    progress:
        Optional :class:`~exonit.progress_client.ProgressReporter`, reported
        against (with ``substep="post_bulk"``) after each bulk chunk POSTs —
        docs/superpowers/specs/2026-09-26-metadata-refresh-and-bulk-enqueue-
        design.md §2.3. A no-op when the caller passes ``None`` or when no
        run is live (see that class's docstring); never used on the
        per-target fallback path (low-traffic, no chunking to report on).

    run_id:
        This lot's run identity (S4, docs/superpowers/specs/2026-09-26-
        enqueue-s1-s4-design.md §2). Forwarded as ``"run_id": "..."`` in the
        bulk payload so the server can (a) count ``adopted`` TICs — those
        blocked by a still-LIVE task from an earlier, dead/timed-out attempt
        at this same lot — and, with ``EXONIT_LOT_ADOPT_ACTIVE=1``, re-tag
        those tasks to this run_id instead of leaving them orphaned. ``None``
        (the default — a caller that predates S3/S4, or a manual
        ``enqueue_from_csv.py`` run without ``--run-id``) omits the key
        entirely, byte-for-byte the pre-S4 payload. Not forwarded on the
        legacy per-target fallback path (same low-traffic carve-out as
        ``sector``/``force_rescan`` above).

    Returns dict with ``enqueued``, ``skipped``, ``failed``, ``adopted``
    counts.
    """
    enqueued = 0
    skipped = 0
    failed = 0
    adopted = 0

    use_bulk = _probe_bulk_endpoint(server_url, token)

    if use_bulk:
        # Send in chunks of _BULK_CHUNK_SIZE to keep request bodies reasonable
        for chunk_start in range(0, len(tics), _BULK_CHUNK_SIZE):
            chunk = tics[chunk_start: chunk_start + _BULK_CHUNK_SIZE]
            payload = {
                "targets": [{"tic_id": tic_id, "label": "candidate"}
                            for tic_id in chunk]
            }
            if force_rescan:
                payload["force_rescan"] = True
            if sector is not None:
                payload["sector"] = sector
            if run_id:
                payload["run_id"] = run_id

            # Q5 (perf review 2026-09-26): a couple of short retries before
            # giving up on this chunk. Safe to retry blindly (no dedup
            # awareness needed here) because the server's own dedup makes a
            # repeated identical payload idempotent — see _BULK_POST_MAX_
            # RETRIES' docstring above.
            data = None
            last_exc = None
            for attempt in range(_BULK_POST_MAX_RETRIES + 1):
                try:
                    resp = requests.post(
                        f"{server_url}/api/worker/enqueue_bulk",
                        json=payload,
                        # Each chunk is self-contained; 60 s is ample even
                        # for a 20000-row unnest()/executemany insert on a
                        # loaded instance.
                        timeout=60,
                        headers={"Authorization": f"Bearer {token}"},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < _BULK_POST_MAX_RETRIES:
                        logger.warning(
                            "enqueue_bulk chunk failed (start=%d, n=%d), "
                            "retrying (attempt %d/%d): %s",
                            chunk_start, len(chunk), attempt + 1,
                            _BULK_POST_MAX_RETRIES, exc,
                        )
                        time.sleep(_BULK_POST_RETRY_BACKOFF_S * (attempt + 1))

            if data is not None:
                enqueued += data.get("enqueued", 0)
                skipped += data.get("skipped", 0)
                failed += data.get("failed", 0)
                adopted += data.get("adopted", 0)
            else:
                logger.warning(
                    "enqueue_bulk chunk failed (start=%d, n=%d) after %d "
                    "retries: %s",
                    chunk_start, len(chunk), _BULK_POST_MAX_RETRIES, last_exc,
                )
                failed += len(chunk)

            if progress is not None:
                progress.report(chunk_start + len(chunk), substep="post_bulk")

    else:
        # Fallback: original per-target POST loop
        logger.info(
            "enqueue_targets: server does not support /enqueue_bulk, "
            "falling back to per-target POSTs (%d targets)",
            len(tics),
        )

        def _post_one(tic_id: str) -> str:
            payload = {"tic_id": tic_id}
            try:
                resp = requests.post(
                    f"{server_url}/api/worker/enqueue",
                    json=payload,
                    timeout=10,
                    headers={"Authorization": f"Bearer {token}"},
                )
                resp.raise_for_status()
                return "enqueued"
            except Exception as exc:
                logger.warning("Failed to enqueue TIC %s: %s", tic_id, exc)
                return "failed"

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_post_one, tic): tic for tic in tics}
            for fut in as_completed(futures):
                result = fut.result()
                if result == "enqueued":
                    enqueued += 1
                else:
                    failed += 1

    return {"enqueued": enqueued, "skipped": skipped, "failed": failed,
            "adopted": adopted}


def enqueue_targets(
    targets: List[Dict[str, Any]],
    db: Any,
    server_url: str = _DEFAULT_SERVER_URL,
    *,
    token: Optional[str] = None,
    cache_path: Optional[str] = None,
    max_workers: int = 8,
    force_rescan: bool = False,
    sector: Optional[int] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Enqueue a list of targets to the ExoNit server for worker processing.

    For each target:
      1. Skip if already analyzed in *db* (bypassed in deep-rescan mode).
      2. Auto-index non-TOI targets into the MetadataCache (idempotent).
      3. Skip (and mark ``no_fits``) if no metadata cache entry.
      4. POST to the server — bulk path preferred, per-target fallback
         (see ``_post_bulk``).

    force_rescan:
        Deep-rescan mode (spec 2026-07-23). Skips ONLY step 1
        (already-analyzed): completed TICs are re-enqueued so a deeper
        multi-sector scan can supersede their old analysis. Steps 2-3
        (metadata / no_fits) still apply. The flag is forwarded to
        /api/worker/enqueue_bulk, whose own task-row dedup would otherwise
        skip these TICs server-side. NEVER default anywhere.

    sector:
        The lot's TESS sector (done-state design, 2026-09-26). Forwarded to
        ``_post_bulk`` (so ``tasks.sector`` is populated) and, when
        ``EXONIT_DONE_STATE=1``, used to widen step 1's local fast filter
        with exhausted(t, sector) (see :mod:`exonit.done_state`) — with the
        flag off, step 1 is unchanged (``db.completed_tic_ids()`` only).

    ``EXONIT_ENQUEUE_METADATA_V2`` (default off): routes steps 1-3 through
    the chunked bulk path (docs/superpowers/specs/2026-09-26-metadata-
    refresh-and-bulk-enqueue-design.md) instead of the ``completed_tic_ids()``
    / ``all_tic_ids()`` full-table loads + per-TIC ``ensure_metadata`` loop —
    see ``_enqueue_targets_v2`` below. With the flag off (default), this
    function's behaviour is byte-for-byte the pre-2026-09-26 implementation.
    ``EXONIT_METADATA_REFRESH`` (default off) additionally enables the
    stale-row refresh, and only has any effect when V2 is also on.

    run_id:
        This lot's run identity (S4, docs/superpowers/specs/2026-09-26-
        enqueue-s1-s4-design.md §2). Forwarded to ``_post_bulk`` only — not
        used by any local step 1-3 filter. ``None`` (default) omits it from
        the bulk payload, byte-for-byte pre-S4 behaviour.

    Returns dict with ``enqueued``, ``skipped``, ``failed``, ``adopted``
    counts and ``elapsed_seconds`` — plus, on the V2 path,
    ``metadata_created``/``metadata_refreshed``/``metadata_unchanged``/
    ``metadata_preserved_non_tglc`` (the refresh's per-lot impact metric,
    spec §2.3). ``adopted`` is always present (default 0), same "uniform
    shape regardless of flags/outcome" contract as the metadata_* keys (see
    ``_ZERO_METADATA_COUNTS``'s docstring) — it is 0 whenever ``ready_tics``
    is empty (the early return below never calls ``_post_bulk``) or the
    server predates S4.
    """
    if token is None:
        # Read live from EXONIT_WORKER_TOKEN (exonit.config.worker_token) so a
        # deployment that changes the token can't 401 the local CLI.
        token = worker_token()
    t0 = time.monotonic()

    rep = ProgressReporter(unit="TICs", total=len(targets),
                           substep="ensure_metadata", interval_s=15)

    if os.environ.get("EXONIT_ENQUEUE_METADATA_V2", "0") == "1":
        (ready_tics, no_fits_targets, skipped, metadata_counts) = (
            _enqueue_targets_v2(targets, db, force_rescan=force_rescan,
                                sector=sector, cache_path=cache_path,
                                progress=rep))
    else:
        (ready_tics, no_fits_targets, skipped, metadata_counts) = (
            _enqueue_targets_legacy(targets, db, force_rescan=force_rescan,
                                    sector=sector, cache_path=cache_path))

    # Mark no-FITS stars in the DB. Q9 (perf review 2026-09-26): a single
    # add_stars_bulk() executemany instead of one add_star() (+ one SQLite
    # commit) per TIC in a Python loop -- these rows are otherwise identical
    # (only tic_id differs), the exact case add_stars_bulk exists for.
    db.add_stars_bulk(
        no_fits_targets, ra=0, dec=0, tmag=0,
        n_sectors=0, n_datapoints=0,
        time_span_days=0, status="no_fits",
    )

    if not ready_tics:
        elapsed = time.monotonic() - t0
        summary = {
            "enqueued": 0,
            "skipped": skipped,
            "failed": 0,
            "adopted": 0,
            "elapsed_seconds": round(elapsed, 2),
            **metadata_counts,
        }
        logger.info(
            "enqueue_targets: 0 enqueued, %d skipped, 0 failed (%.1fs)",
            skipped, elapsed,
        )
        return summary

    # --- POST to the server (bulk preferred, per-target fallback) ---
    bulk_result = _post_bulk(ready_tics, server_url, token,
                             force_rescan=force_rescan, max_workers=max_workers,
                             sector=sector, progress=rep, run_id=run_id)
    enqueued = bulk_result.get("enqueued", 0)
    skipped += bulk_result.get("skipped", 0)
    failed = bulk_result.get("failed", 0)
    adopted = bulk_result.get("adopted", 0)

    elapsed = time.monotonic() - t0
    summary = {
        "enqueued": enqueued,
        "skipped": skipped,
        "failed": failed,
        "adopted": adopted,
        "elapsed_seconds": round(elapsed, 2),
        **metadata_counts,
    }
    logger.info(
        "enqueue_targets: %d enqueued, %d skipped, %d failed, %d adopted "
        "(%.1fs)",
        enqueued, skipped, failed, adopted, elapsed,
    )
    return summary


def _enqueue_targets_legacy(
    targets: List[Dict[str, Any]],
    db: Any,
    *,
    force_rescan: bool,
    sector: Optional[int],
    cache_path: Optional[str],
):
    """Pre-2026-09-26 steps 1-3 of ``enqueue_targets``, byte-for-byte
    unchanged — the ``EXONIT_ENQUEUE_METADATA_V2=0`` (default) path.

    Returns ``(ready_tics, no_fits_targets, skipped, metadata_counts)``;
    ``metadata_counts`` is always ``_ZERO_METADATA_COUNTS`` here (no
    bulk-refresh metrics on this path — but the KEYS are always present, so
    ``summary`` has a uniform shape across both paths; see that constant's
    docstring).
    """
    skipped = 0

    # --- Local fast filters (steps 1-3 for every target) ---
    # Pre-load sets once so per-target checks are O(1) instead of O(query).
    # At 200k targets the old approach (1 DB query + 2 MetadataCache opens per
    # target) took >600 s and blew up the scan timeout.  Building two sets up
    # front and reusing a single MetadataCache connection for the rare
    # need_index tail brings this down to a fraction of a second.

    # Pre-load set of already-completed TICs from stars table. When
    # EXONIT_DONE_STATE=1, also fold in exhausted(t, sector) (a TIC that
    # failed twice for THIS sector — done_state.MAX_LOTS_PER_SECTOR) so the
    # client-side filter agrees with the server's dedup (routes/workers.py's
    # worker_enqueue_bulk) and step 1's log-visible "skipped" count is
    # accurate; with the flag off (default) this is unchanged.
    analyzed_set: set = db.completed_tic_ids()
    if sector is not None:
        from exonit import done_state
        if done_state.is_enabled():
            try:
                analyzed_set = analyzed_set | (
                    done_state.exhausted_tic_ids_from_backend(
                        db.backend, sector))
            except Exception:
                logger.warning(
                    "done_state.exhausted_tic_ids failed for sector=%s "
                    "(non-fatal; falling back to stars-only analyzed_set)",
                    sector, exc_info=True)

    # Pre-load set of TICs already indexed in the metadata cache.
    if cache_path is None:
        from exonit.config import ExoNitConfig
        _cfg_cache_path = ExoNitConfig().metadata_cache_path
    else:
        _cfg_cache_path = cache_path
    _mc = MetadataCache(_cfg_cache_path)
    metadata_set: set = _mc.all_tic_ids()

    no_fits_targets: List[str] = []
    ready_tics: List[str] = []
    need_index: List[str] = []

    for target in targets:
        tic_id = str(target["tic_id"])

        # 1. Already analyzed? (bypassed in deep-rescan mode)
        if not force_rescan and tic_id in analyzed_set:
            skipped += 1
            continue

        # 2. Has metadata already (O(1) set lookup)?
        if tic_id in metadata_set:
            ready_tics.append(tic_id)
        else:
            need_index.append(tic_id)

    # 3. For the small tail without metadata: try to index, reusing ONE
    #    MetadataCache connection (no per-TIC open/close).
    for tic_id in need_index:
        try:
            indexed = ensure_metadata(tic_id, cache_path=cache_path, mc=_mc)
        except Exception as e:
            # Never silent: a transient failure here (MAST/network/cache)
            # cascades into a permanent no_fits+skipped mark for this TIC.
            logger.warning(
                "ensure_metadata failed for TIC %s (%s: %s) — if it has no "
                "cached metadata it will be marked no_fits and skipped",
                tic_id, type(e).__name__, e)
            indexed = False

        if indexed:
            ready_tics.append(tic_id)
        else:
            no_fits_targets.append(tic_id)
            skipped += 1

    return ready_tics, no_fits_targets, skipped, dict(_ZERO_METADATA_COUNTS)


def _enqueue_targets_v2(
    targets: List[Dict[str, Any]],
    db: Any,
    *,
    force_rescan: bool,
    sector: Optional[int],
    cache_path: Optional[str],
    progress: Optional[ProgressReporter],
):
    """``EXONIT_ENQUEUE_METADATA_V2=1`` steps 1-3 of ``enqueue_targets``.

    Replaces the legacy path's two ~27-30M row full-table loads
    (``db.completed_tic_ids()``, ``mc.all_tic_ids()``) with chunked ``IN
    (...)`` probes scoped to THIS lot's TIC ids
    (``db.completed_tic_ids_in``), and the per-TIC ``ensure_metadata`` loop
    with :func:`exonit.metadata_index.ensure_metadata_bulk` — one ``IN
    (...)`` per table per 5000-TIC chunk, refreshing stale TGLC-sourced rows
    when ``EXONIT_METADATA_REFRESH=1`` (docs/superpowers/specs/
    2026-09-26-metadata-refresh-and-bulk-enqueue-design.md §2.3).

    Returns ``(ready_tics, no_fits_targets, skipped, metadata_counts)``.
    """
    skipped = 0
    tic_ids = [str(target["tic_id"]) for target in targets]

    # Step 1: already-analyzed filter, scoped to this lot (not the whole
    # `stars` table) — same done_state widening as the legacy path, applied
    # to the same (now chunk-probed) analyzed_set.
    analyzed_set: set = db.completed_tic_ids_in(tic_ids)
    if sector is not None:
        from exonit import done_state
        if done_state.is_enabled():
            try:
                analyzed_set = analyzed_set | (
                    done_state.exhausted_tic_ids_from_backend(
                        db.backend, sector))
            except Exception:
                logger.warning(
                    "done_state.exhausted_tic_ids failed for sector=%s "
                    "(non-fatal; falling back to stars-only analyzed_set)",
                    sector, exc_info=True)

    candidate_tics: List[str] = []
    for tic_id in tic_ids:
        if not force_rescan and tic_id in analyzed_set:
            skipped += 1
            continue
        candidate_tics.append(tic_id)

    if cache_path is None:
        from exonit.config import ExoNitConfig
        _cfg_cache_path = ExoNitConfig().metadata_cache_path
    else:
        _cfg_cache_path = cache_path
    _mc = MetadataCache(_cfg_cache_path)

    refresh = os.environ.get("EXONIT_METADATA_REFRESH", "0") == "1"
    bulk_result = ensure_metadata_bulk(
        candidate_tics, _mc, refresh=refresh,
        progress=progress) if candidate_tics else None

    if bulk_result is None:
        return [], [], skipped, dict(_ZERO_METADATA_COUNTS)

    skipped += len(bulk_result.no_fits)
    metadata_counts = {
        "metadata_created": bulk_result.created,
        "metadata_refreshed": bulk_result.refreshed,
        "metadata_unchanged": bulk_result.unchanged,
        "metadata_preserved_non_tglc": bulk_result.preserved_non_tglc,
    }
    return bulk_result.ready, bulk_result.no_fits, skipped, metadata_counts
