"""S2 — sector gap-stats without whole-DB loads, via ``postgres_fdw``.

Implements docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md §5.

The legacy gap metric (``exonit.discovery_state.sector_gap_stats``) pays for
one whole-DB completed-set scan (``SELECT DISTINCT tic_id FROM tasks WHERE
status='completed'``, ~27.6M rows) PER CALL, then intersects it in Python
with the sector's work list. This module replaces the expensive term —
``completed(S) = |worklist(S) ∩ done|`` — with ONE server-side count via a
foreign-data-wrapper semi-join (§5.1): PostgreSQL pulls the SMALL side (the
sector's ``tglc`` rows, 2.5-13M) across the fdw link and probes it against
the LOCAL ``tic_sector_state`` table with an ``EXISTS`` — never materialising
the 27.6M-row completed set, and never moving it into a Python process.

Every other term in the gap metric (``no_data``, ``tois_skipped``,
``failed_terminal``, ``missing_download_tics``) is already small (tens of
thousands at most — TOI catalog, ``discovery_no_data``, ``exhausted(t, S)``,
the canonical CSV when one exists) and is computed with bounded probes
against those small sets, via :func:`exonit.metadata_cache.MetadataCache.
tglc_tic_ids_for_tics` (chunked ``IN``-list against ``tglc``) and this
module's own :func:`_probe_done_chunked` (chunked ``IN``-list against
``tic_sector_state`` — see that function's docstring for why this is a
LOCAL, always-chunked probe and deliberately not
``exonit.done_state._done_ids_scoped``, whose >100k-id fallback is exactly
the kind of whole-table scan this module exists to eliminate). Never a full
``tglc_S`` or whole-DB load, however large any of these candidate sets grow.

Requires the DDL in ``scripts/migration/2026-09-26_s2_postgres_fdw.sql``
applied on the operational DB (schema ``meta_fdw``, foreign table
``meta_fdw.tglc`` pointing at ``exonit_meta``). Gated behind
``EXONIT_GAP_STATS_V2`` (default 0) AND ``EXONIT_DONE_STATE=1`` (the
``tic_sector_state`` table this module reads is that spec's table) — see
:func:`exonit.discovery_state.sector_gap_stats`'s dispatch for the
flag-off/misconfigured-flag fallback to the legacy path.

A genuine runtime failure here (fdw server down, foreign table absent,
statement timeout) is NEVER swallowed into a silent fallback — it propagates
to the caller. The stale-while-error policy (keep the last cached value
instead of falling back to the legacy whole-DB scan) lives one layer up, in
the background gap-stats refresh loop (``exonit/routes/discovery.py``) —
that is the "uvicorn worker" the spec's §5.2 forbids from ever falling back
to legacy on an fdw error.
"""

from __future__ import annotations

import os
import re
from pathlib import Path as _Path
from typing import Optional


class FdwUnavailable(Exception):
    """Raised when the V2 path is asked to run against a non-PostgreSQL
    operational backend (SQLite has no ``postgres_fdw``) — callers translate
    this into the legacy path, never a crash."""


# work_mem / statement_timeout are env-configured constants, not user input,
# but SET/SET LOCAL do not accept bound parameters at the wire protocol level
# (psycopg has nothing to bind them to) — the value is string-interpolated
# into the SQL, so it is validated against a strict allow-pattern first
# (PostgreSQL GUC value syntax: digits + optional unit, e.g. "256MB", "300s").
_GUC_VALUE_RE = re.compile(r"^[0-9]+(?:[A-Za-z]{1,3})?$")

_DEFAULT_WORK_MEM = "256MB"
_DEFAULT_STATEMENT_TIMEOUT = "300s"


def _guc_value(env_var: str, default: str) -> str:
    val = os.environ.get(env_var, default)
    if not _GUC_VALUE_RE.match(val):
        import warnings
        warnings.warn(
            f"[gap_stats_v2] {env_var}={val!r} is not a valid PostgreSQL GUC "
            f"value (expected e.g. '256MB' / '300s'); using default {default!r}.",
            RuntimeWarning, stacklevel=3)
        return default
    return val


def is_enabled() -> bool:
    """True when ``EXONIT_GAP_STATS_V2=1``.

    Callers (``exonit.discovery_state.sector_gap_stats``) additionally
    require ``exonit.done_state.is_enabled()`` before actually dispatching to
    this module — see that function's docstring for the warn+legacy
    behaviour when V2 is requested without done-state.
    """
    return os.environ.get("EXONIT_GAP_STATS_V2", "0") == "1"


# --------------------------------------------------------------------------- #
# The one expensive term, computed server-side
# --------------------------------------------------------------------------- #

def completed_count_via_fdw(backend, sector: int) -> int:
    """``|tglc_S ∩ done|`` — the sector's completed count, computed entirely
    inside PostgreSQL via the ``meta_fdw.tglc`` foreign table.

    Runs inside one transaction with ``SET LOCAL work_mem`` /
    ``SET LOCAL statement_timeout`` (spec §5.1) so this single query never
    blows past the connection's default ``work_mem`` (needed for the hash
    semi-join) nor hangs the caller indefinitely if the foreign server is in
    trouble.

    Raises :class:`FdwUnavailable` when *backend* is not PostgreSQL. Any
    other failure (foreign server down, ``meta_fdw.tglc`` absent, statement
    timeout) propagates as whatever psycopg raises — this function never
    swallows an error into a fallback (see the module docstring).
    """
    if backend.dialect.name != "postgres":
        raise FdwUnavailable(
            "gap_stats_v2.completed_count_via_fdw requires a PostgreSQL "
            "operational backend (SQLite has no postgres_fdw); the caller "
            "must use the legacy gap-stats path instead.")

    work_mem = _guc_value("EXONIT_GAP_STATS_V2_WORK_MEM", _DEFAULT_WORK_MEM)
    timeout = _guc_value(
        "EXONIT_GAP_STATS_V2_STATEMENT_TIMEOUT", _DEFAULT_STATEMENT_TIMEOUT)

    with backend.transaction() as conn:
        conn.execute(f"SET LOCAL work_mem = '{work_mem}'")
        conn.execute(f"SET LOCAL statement_timeout = '{timeout}'")
        cur = conn.execute(
            "SELECT count(*) AS n FROM meta_fdw.tglc t "
            "WHERE t.sector = %s AND EXISTS ("
            "  SELECT 1 FROM tic_sector_state s "
            "  WHERE s.tic_id = t.tic_id AND s.status = 'completed')",
            (sector,),
        )
        row = cur.fetchone()
        return int(row["n"]) if row is not None else 0


# --------------------------------------------------------------------------- #
# Small bounded probes (never a whole-tglc_S / whole-DB load)
# --------------------------------------------------------------------------- #

#: Chunk size for :func:`_probe_done_chunked`'s ``IN`` lists — matches the
#: review's request (≤5000), well under both SQLite's and PostgreSQL's
#: per-statement parameter limits and small enough that even a
#: pathologically large candidate set (a sector whose `exhausted(t,S)` or
#: canonical-CSV count blows past the "tens of thousands" this module
#: otherwise expects) issues many small index-only queries instead of ever
#: falling through to a full-table scan.
_PROBE_DONE_CHUNK_SIZE = 5000


def _probe_done_chunked(backend, ids, *, chunk_size: int = _PROBE_DONE_CHUNK_SIZE) -> set:
    """``done(t) ∩ ids`` via ONLY chunked ``tic_id IN (...)`` probes against
    ``tic_sector_state``'s completed-only partial index (``idx_tss_done``) —
    NEVER a full-table scan, no matter how many *ids* are passed.

    Deliberately NOT :func:`exonit.done_state._done_ids_scoped`: that helper
    falls back to an UNFILTERED ``SELECT tic_id FROM tic_sector_state WHERE
    status = 'completed'`` (via ``backend.iter_id_column``) once the
    candidate list exceeds ``_CHUNK_PROBE_MAX_IDS`` (100k) — exactly the
    whole-DB-style load this module exists to eliminate, and one that would
    run silently inside a live uvicorn worker if `exhausted(t, S)` or a
    sector's canonical-minus-tglc set ever grew past that threshold (review
    2026-09-26, progress/review-enqueue-s2.md item 1). This module therefore
    keeps its own probe, always chunked, with no size-dependent branch at
    all — and never touches ``exonit/done_state.py`` (a parallel branch is
    independently adding a ``probe_done`` helper there; keeping this local
    avoids stepping on that work).

    Returns the subset of *ids* (as the STRING form ``tic_id`` is stored in)
    that are ``done(t)`` — callers compare with ``str(t) in result``.
    """
    str_ids = list({str(t) for t in ids})
    if not str_ids:
        return set()
    from exonit.db_backend import select_in_chunks
    rows = select_in_chunks(
        backend.query,
        "SELECT tic_id FROM tic_sector_state WHERE status = 'completed' AND tic_id IN",
        str_ids, chunk_size=chunk_size)
    return {r["tic_id"] for r in rows}


def _toi_in_worklist(mc, toi_set: set, canonical_set: set, sector: int) -> set:
    """``TOI ∩ (canonical ∪ tglc_S)`` — bounded by ``len(toi_set)`` (~7k
    globally), never by the sector's tglc size."""
    if not toi_set:
        return set()
    in_canonical = toi_set & canonical_set
    in_tglc = mc.tglc_tic_ids_for_tics(toi_set, sector)
    return in_canonical | (in_tglc - canonical_set)


def _exhausted_in_worklist(mc, exhausted_ints: set, canonical_set: set,
                           sector: int) -> set:
    """``exhausted(·, S) ∩ (canonical ∪ tglc_S)`` — bounded by
    ``len(exhausted_ints)`` (tens of thousands at most, per-sector)."""
    if not exhausted_ints:
        return set()
    in_canonical = exhausted_ints & canonical_set
    in_tglc = mc.tglc_tic_ids_for_tics(exhausted_ints, sector)
    return in_canonical | (in_tglc - canonical_set)


# --------------------------------------------------------------------------- #
# Public entry point — same output shape as discovery_state.sector_gap_stats
# --------------------------------------------------------------------------- #

def sector_gap_stats_v2(
    sector: int,
    *,
    db_path: str,
    canonical_csv: str,
    tglc_index: str,
    toi_catalog: Optional[str] = None,
) -> dict:
    """V2 gap metric — same keys/semantics as
    :func:`exonit.discovery_state.sector_gap_stats`, computed WITHOUT ever
    loading the whole completed-tasks set or the whole ``tglc_S`` set.

    Parity by construction: every term below is exactly the set legacy
    computes by filtering the materialised ``work_list`` — this just counts
    each one with a bounded query instead of a Python membership test over a
    list that, for a full sector, would be millions of elements.
    """
    from exonit.db_backend import connect_operational
    from exonit.metadata_cache import MetadataCache
    from exonit import done_state as _done_state
    from exonit.discovery_state import (
        _load_canonical_tics, count_extended_canonical, load_toi_tic_set,
    )

    backend = connect_operational(db_path)
    if backend.dialect.name != "postgres":
        backend.close()
        raise FdwUnavailable(
            "sector_gap_stats_v2 requires a PostgreSQL operational DB "
            "(SQLite has no postgres_fdw); use the legacy path.")

    _toi_cat = (
        toi_catalog if toi_catalog is not None
        else str(_Path(__file__).resolve().parent.parent / "data" / "toi_catalog.csv")
    )
    toi_set = load_toi_tic_set(_toi_cat)

    canonical_tics, canonical_set = _load_canonical_tics(canonical_csv)

    # work_list_size: already an all-SQL COUNT (no whole-list materialisation)
    # — reused as-is, S1/S2 share this helper.
    work_list_size = count_extended_canonical(canonical_csv, tglc_index, sector)

    mc = MetadataCache(tglc_index)
    try:
        # --- completed: fdw count over tglc_S, plus the canonical-only slice
        #     (canonical TICs never downloaded, so absent from tglc_S, but
        #     already done via some other sector) — empty for every sector
        #     without a canonical list (§5.2). ---
        completed_count = completed_count_via_fdw(backend, sector)
        if canonical_set:
            canonical_in_tglc = mc.tglc_tic_ids_for_tics(canonical_set, sector)
            canonical_minus_tglc = canonical_set - canonical_in_tglc
            done_among_canonical_minus_tglc_strs = _probe_done_chunked(
                backend, canonical_minus_tglc)
            completed_count += len(done_among_canonical_minus_tglc_strs)
        else:
            canonical_minus_tglc = set()
            done_among_canonical_minus_tglc_strs = set()

        # --- no_data: unconditional per-sector count, exactly like legacy
        #     (not filtered against the work list — see sector_gap_stats's
        #     equivalent note: a TIC can be both no_data(S) and done(t) via
        #     another sector; legacy does not dedupe that either). ---
        no_data_rows = backend.query(
            "SELECT tic_id FROM discovery_no_data WHERE sector = ?", (sector,))
        no_data_tic_strs = {r["tic_id"] for r in no_data_rows}
        nd = len(no_data_tic_strs)

        # --- tois_skipped: TOI ∩ worklist, minus done, minus no_data. ---
        toi_worklist = _toi_in_worklist(mc, toi_set, canonical_set, sector)
        toi_done_strs = _probe_done_chunked(backend, toi_worklist)
        tois_skipped = sum(
            1 for t in toi_worklist
            if str(t) not in toi_done_strs and str(t) not in no_data_tic_strs
        )

        # --- failed_terminal: exhausted(t, S) ∩ worklist, minus done, minus
        #     no_data, minus TOI (done-state design §3 — always computed here
        #     since V2 requires EXONIT_DONE_STATE=1). ---
        exhausted_strs = _done_state.exhausted_tic_ids(db_path, sector)
        exhausted_ints = {int(t) for t in exhausted_strs if str(t).lstrip("-").isdigit()}
        exhausted_worklist = _exhausted_in_worklist(
            mc, exhausted_ints, canonical_set, sector)
        exhausted_done_strs = _probe_done_chunked(backend, exhausted_worklist)
        failed_terminal = sum(
            1 for t in exhausted_worklist
            if str(t) not in exhausted_done_strs
            and str(t) not in no_data_tic_strs
            and t not in toi_set
        )

        # --- missing_download_tics: canonical-only, minus done, no_data, TOI
        #     (empty whenever there is no canonical list — §5.2). Order
        #     matches legacy: canonical CSV row order. ---
        missing_download_tics = [
            t for t in canonical_tics
            if t in canonical_minus_tglc
            and str(t) not in done_among_canonical_minus_tglc_strs
            and str(t) not in no_data_tic_strs
            and t not in toi_set
        ]
    finally:
        mc.close()
        backend.close()

    gaps = max(0, work_list_size - completed_count - nd - tois_skipped
              - failed_terminal)
    pct = ((completed_count + nd + tois_skipped + failed_terminal)
          / work_list_size if work_list_size else 0.0)

    return {
        "work_list_size": work_list_size,
        "completed": completed_count,
        "no_data": nd,
        "tois_skipped": tois_skipped,
        "gaps_processable": gaps,
        "pct_complete": pct,
        "missing_download_tics": missing_download_tics,
        "failed_terminal": failed_terminal,
    }
