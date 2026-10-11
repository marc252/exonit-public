"""Server-side helper: batch-fetch stellar parameters from the TIC via MAST.

Used as a fallback in discovery_postprocess.py for FFI-only TICs whose
MetadataCache R* is empty (no local FITS header → no cached stellar params).

Design rules
------------
- **Server-side only.** Do NOT call from workers (violates
  [[feedback_workers_no_external]]).  Only ``discovery_postprocess.py`` at
  closeout time should use this, and only when ``--fetch-missing-stellar`` is
  passed.
- **Bounded.** The caller MUST pass a small, priority-ordered set (the
  finalist-candidate ``survivors`` — hundreds/lot, NOT every planet_candidate).
  This helper adds three independent safety bounds so it can never hang
  postprocess even if the caller slips: a hard ``max_ids`` cap, per-request
  ``chunk_size`` batching, and an overall wall-clock ``deadline_s``.  (Incident
  2026-06-29: an unbounded ~19k-ID single query flooded MAST and hung ~50 min.)
- **Graceful / fail-open.** Any failure (missing ``astroquery``, network error,
  timeout, one bad chunk, deadline) → that TIC's R* is simply absent, so the
  depth gate skips it (a candidate is NEVER demoted for lack of R*).  Recall is
  always preserved; only specificity is (partially) lost on failure.

Why MAST on-demand is acceptable here despite [[feedback_no_ondemand_mast]]:
it runs server-side (authorised proxy, not a worker), at closeout only, for the
few finalist-candidate survivors lacking local R*, is chunked + bounded +
graceful, and the FFI-only metadata gap is exactly the lesson from the δ-Scuti /
bright-giant false-positive analysis.  Fail-open always preserves recall.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _coerce_float(v) -> Optional[float]:
    """Safely convert *v* to float.  Returns None on failure or non-finite result.

    Broad except: TIC table cells can be numpy masked constants, which raise
    assorted errors on ``float()``; any of them mean "no value" → None.
    """
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:  # noqa: BLE001 — pure coercion helper, any failure → None
        return None


def fetch_stellar_params_mast(
    tic_ids,
    *,
    timeout: int = 30,
    chunk_size: int = 100,
    max_ids: int = 1500,
    deadline_s: float = 150.0,
) -> Dict[str, Dict]:
    """Return stellar params ``{tic_id: {rad, teff, logg}}`` from the TIC via MAST.

    Bounded + graceful: queries in chunks of ``chunk_size``, stops issuing new
    chunks once ``deadline_s`` elapses, and never queries more than ``max_ids``
    IDs.  Any failure (import, network, timeout, bad chunk) is swallowed and the
    affected TICs are simply absent from the result.

    Parameters
    ----------
    tic_ids : iterable of str or int
        TIC IDs to look up, **priority-ordered** (highest-value first): if the
        ``max_ids`` cap or the ``deadline_s`` budget cuts the run short, the
        head of the list is the part that gets fetched.  Accepts str or int; all
        are coerced to str in the result dict (the convention used in
        postprocess.py).
    timeout : int
        Per-request MAST timeout in seconds (default 30).
    chunk_size : int
        TIC IDs per ``query_criteria`` request (default 100).  Keeps each
        request small enough that MAST answers quickly.
    max_ids : int
        Hard upper bound on IDs queried (default 1500).  A backstop against an
        unbounded caller; excess (lowest-priority) IDs are dropped with a log
        line (no silent truncation).
    deadline_s : float
        Overall wall-clock budget in seconds (default 150).  Once exceeded, no
        further chunks are issued; whatever was fetched so far is returned.

    Returns
    -------
    dict[str, dict]
        Keys: TIC ID (str).  Values: ``{rad, teff, logg}`` where each field is a
        float in TIC units (R_sun, K, dex) or ``None`` if not available.
        Returns ``{}`` on total failure (no astroquery, every chunk failed,
        empty input, etc.).
    """
    ids = [str(t) for t in tic_ids]
    if not ids:
        return {}

    # Hard cap: never issue an effectively-unbounded series of MAST requests.
    # The caller should already bound the set (finalist-candidate survivors);
    # this is a backstop.  IDs are priority-ordered, so keep the head and log
    # the drop rather than truncating silently.
    if len(ids) > max_ids:
        logger.warning(
            "fetch_stellar_params_mast: %d IDs exceeds max_ids=%d; querying the "
            "first %d only (remaining R* fail-open). Caller should bound the set.",
            len(ids), max_ids, max_ids,
        )
        ids = ids[:max_ids]

    try:
        from astroquery.mast import Catalogs  # lazy import — optional dependency
    except Exception as exc:  # noqa: BLE001 — astroquery absent → skip gate, recall safe
        logger.warning(
            "fetch_stellar_params_mast: astroquery unavailable (%s: %s); "
            "depth gate will skip missing R*", type(exc).__name__, exc,
        )
        return {}

    Catalogs.TIMEOUT = timeout

    result: Dict[str, Dict] = {}
    start = time.monotonic()
    n_chunks = (len(ids) + chunk_size - 1) // chunk_size

    for i in range(0, len(ids), chunk_size):
        # Wall-clock budget: stop issuing new chunks once the deadline passes so
        # a slow/unreachable MAST can never hang postprocess for minutes.  What
        # we have so far is returned; the rest fail-open (recall preserved).
        if time.monotonic() - start > deadline_s:
            logger.warning(
                "fetch_stellar_params_mast: deadline %.0fs reached after %d/%d "
                "IDs; returning partial (remaining R* fail-open)",
                deadline_s, i, len(ids),
            )
            break

        chunk = ids[i:i + chunk_size]
        try:
            table = Catalogs.query_criteria(catalog="Tic", ID=chunk)
        except Exception as exc:  # noqa: BLE001 — one bad chunk must not abort the rest
            logger.warning(
                "fetch_stellar_params_mast: chunk [%d:%d] failed (%s: %s); skipping",
                i, i + len(chunk), type(exc).__name__, exc,
            )
            continue

        if table is None or len(table) == 0:
            continue

        cols = set(getattr(table, "colnames", ()))
        for row in table:
            try:
                tid = str(row["ID"])
            except (TypeError, ValueError, KeyError):
                continue
            result[tid] = {
                "rad": _coerce_float(row["rad"]) if "rad" in cols else None,
                "teff": _coerce_float(row["Teff"]) if "Teff" in cols else None,
                "logg": _coerce_float(row["logg"]) if "logg" in cols else None,
            }

    logger.info(
        "fetch_stellar_params_mast: %d/%d TIC(s) returned params (≤%d chunk(s) of %d)",
        len(result), len(ids), n_chunks, chunk_size,
    )
    return result
