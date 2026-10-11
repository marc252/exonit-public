"""S1 — O(lot) paged selector for the survey (non-canonical) discovery path.

docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md §4. Replaces
``exonit.discovery_state.load_extended_canonical`` + ``derive_done_set`` +
``next_block`` — an O(sector) pipeline that loads the WHOLE sector's TGLC
index and the WHOLE completed-TIC set into Python — with an O(lot) one: a
handful of small-set probes (retry pool, TOI) plus a bounded number of
``LIMIT``-ed pages walked forward from a persisted ``seq`` cursor.

Only used when (see ``scripts/discovery_run.py``'s call site):
  - ``EXONIT_SELECTOR_PAGED=1`` (default 0 — flag off is byte-for-byte the
    legacy path, this module is never imported/called);
  - ``EXONIT_DONE_STATE=1`` (this module reads ``tic_sector_state``
    directly — it has no legacy fallback of its own);
  - the sector has NO ``canonical_list`` in the registry (the six open
    sectors as of 2026-09-26 — a sector WITH a canonical CSV keeps the
    legacy path exactly, see the spec's §0 summary);
  - the caller did not pass ``--full-scan`` (the exact-recall safety net for
    a near-closed sector — see ``verify_sector_complete.py``/autofill).

Accepted trade-off (§4.1.8, operator decision §8.3/§9.3): a TIC BEHIND the
cursor that is neither done(t) nor exhausted(t,S) nor in the retry pool
(e.g. a lost best-effort ``mark_tic_done`` write) is never reselected in
paged mode — the legacy path would eventually re-see it (it always rescans
from the front). The retry pool (attempts < MAX_LOTS_PER_SECTOR, §1's (b)
requirement) is the safety net for the one real, common case (a task that
failed once); ``--full-scan`` is the exact-recall net for everything else,
used only on near-closed sectors.
"""

from __future__ import annotations

import logging
from typing import Optional

from exonit import done_state
from exonit.db_backend import connect_operational
from exonit.metadata_cache import MetadataCache

logger = logging.getLogger(__name__)

#: Rows fetched per ``tglc_tic_ids_page`` call. The spec's §4.1.7 cost
#: estimate (~5-10 s/lot) is based on this value; a caller with an unusually
#: sparse sector tail may want a bigger page (fewer round trips) but this is
#: not currently exposed as a CLI flag — see discovery_run.py's --full-scan
#: escape hatch for that case instead.
DEFAULT_PAGE_SIZE = 20_000


def select_lot_paged(
    sector: int,
    count: int,
    *,
    db_path: str,
    tglc_index_path: str,
    cursor: int,
    cursor_seq: Optional[int],
    toi_tic_set: set,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> dict:
    """Select up to *count* not-done TICs for *sector*, O(lot) (§4.1).

    Parameters
    ----------
    cursor:
        The sector's current (legacy, row-count) ``cursor`` — used ONLY to
        bootstrap ``cursor_seq`` when the latter is ``None`` (first paged
        call for this sector, or a cursor mismatch — see
        ``exonit.discovery_state.get_cursor_seq_if_valid``).
    cursor_seq:
        The persisted ``seq``-based cursor (``None`` triggers the bootstrap
        described above).
    toi_tic_set:
        Pre-loaded TOI catalog TIC set (same one ``discovery_run.py``'s
        legacy path already loads) — TOIs are skipped like a done TIC but
        counted separately (``tois_skipped``).

    Returns
    -------
    dict, shaped like ``exonit.discovery_state.next_block``'s PLUS:
      remaining_undone_is_upper_bound (always ``True`` — §4.1.6: this is a
          CEILING, not the exact gap count S2's gap-stats provides)
      new_cursor_seq          — persist alongside ``new_cursor``
      tois_skipped            — TOI TICs encountered while scanning this lot
      retry_selected          — how many of ``selected`` came from the retry
                                pool (placed FIRST — a deliberate difference
                                from the legacy path's pure canonical order)
      end_reached             — the page scan reached the end of this
                                sector's ``tglc`` partition

    ``ready`` is always identical to ``selected`` and ``missing_download``
    is always ``[]`` — every candidate comes from ``tglc`` itself, so it is
    downloaded by definition (§4.1.5: need_download is impossible here).
    """
    backend = connect_operational(db_path)
    mc = MetadataCache(tglc_index_path)
    try:
        # 1. Small sets (§4.1.1): exhausted(S) and the retry pool. Both are
        # O(tens of thousands) at campaign scale, never O(sector).
        exhausted = done_state.exhausted_tic_ids_from_backend(backend, sector)
        retry_pool = done_state.retryable_failed_ids(
            backend, (sector, done_state.LEGACY_SECTOR))

        # 2. Retry set (§4.1.2): retry_pool ∩ tglc_S, ordered by seq, minus
        # done/exhausted/TOI. Placed FIRST in the output — the (b)
        # requirement of the done-state design (a TIC that failed once must
        # not wait behind the whole rest of the sector to be retried).
        retry_seqs = mc.tglc_seq_for_tics(sector, retry_pool) if retry_pool else {}
        retry_candidates = sorted(retry_seqs, key=lambda t: retry_seqs[t])
        retry_done = done_state.probe_done(backend, retry_candidates)
        retry_ordered = [
            t for t in retry_candidates
            if t not in retry_done
            and str(t) not in exhausted
            and t not in toi_tic_set
        ]

        selected: list = []
        selected_set: set = set()
        retry_selected = 0
        for tic in retry_ordered:
            if len(selected) >= count:
                break
            selected.append(tic)
            selected_set.add(tic)
            retry_selected += 1

        # 3. Cursor bootstrap (§4.1.3): only when the caller has no valid
        # persisted cursor_seq (first paged lot for this sector, or a
        # cursor moved by a non-paged path since the last bootstrap).
        if cursor_seq is None:
            cursor_seq = (
                0 if cursor == 0
                else (mc.tglc_seq_at_offset(sector, cursor - 1) or 0)
            )

        # 4. Pages (§4.1.4): walk forward from cursor_seq until count is met
        # or the sector's tglc partition is exhausted.
        tois_skipped = 0
        scanned = 0
        last_seq = cursor_seq
        end_reached = False
        while len(selected) < count:
            page = mc.tglc_tic_ids_page(sector, last_seq, page_size)
            if not page:
                end_reached = True
                break
            page_tics = [tic for tic, _seq in page]
            done_page = done_state.probe_done(backend, page_tics)
            for tic, seq in page:
                scanned += 1
                last_seq = seq
                if tic in selected_set:
                    continue  # dedupe retry ∩ page (§4.4)
                if tic in toi_tic_set:
                    tois_skipped += 1
                    continue
                if str(tic) in exhausted:
                    continue
                if tic in done_page:
                    continue
                selected.append(tic)
                selected_set.add(tic)
                if len(selected) >= count:
                    break

        # new_cursor = "files consumed" (selected or not — matches the
        # legacy row-count semantics of "the row of the last selected
        # TIC"); if only retry contributed, no page was ever scanned, so
        # the cursor does not move at all (§4.1.5).
        new_cursor = cursor + scanned
        new_cursor_seq = last_seq

        # 5. remaining_undone (§4.1.6): an UPPER BOUND, not the exact gap
        # count (that is S2's job) — it does not subtract TICs completed by
        # OTHER sectors' work ahead of the cursor.
        sector_total = mc.tglc_sector_count(sector) or 0
        remaining_undone = (
            max(0, len(retry_ordered) - retry_selected)
            + max(0, sector_total - new_cursor)
        )

        return {
            "selected": selected,
            "ready": list(selected),
            "missing_download": [],
            "remaining_undone": remaining_undone,
            "remaining_undone_is_upper_bound": True,
            "new_cursor": new_cursor,
            "new_cursor_seq": new_cursor_seq,
            "tois_skipped": tois_skipped,
            "retry_selected": retry_selected,
            "end_reached": end_reached,
        }
    finally:
        backend.close()
        mc.close()
