"""Canonical (TIC, sector) done-state — read predicates + write helpers.

Implements docs/superpowers/specs/2026-09-26-tic-sector-done-state-design.md.

Model (§2.2): a new table, ``tic_sector_state(tic_id, sector, status,
attempts, last_task_id, last_error, updated_at)``, PK ``(tic_id, sector)``,
kept up to date on the worker-result path (``routes/workers.py``'s
``worker_result`` calls :func:`mark_tic_done` / :func:`mark_tic_failed`
unconditionally — dual-write, always on, best-effort — never on the critical
result path's failure mode: callers wrap these in try/except).

Predicates (§1 of the spec):
    done(t)          — a ``completed`` row exists for *t* in ANY sector.
                        Monotonic: a later failure never un-does it.
    exhausted(t, S)  — the row for (t, S) is ``failed`` with
                        ``attempts >= MAX_LOTS_PER_SECTOR`` (operator decision,
                        §7.1: 2 — one terminal worker-failure, one retry lot,
                        a second terminal failure exhausts the TIC for that
                        sector).

``no_data(t, S)`` (the pre-existing ``discovery_no_data`` table) is
DELIBERATELY left out of the selection predicates this module exposes
(:func:`done_and_exhausted`, :func:`blocked_for_enqueue`) — its interaction
with the download-retry / missing-download machinery in
``scripts/discovery_run.py`` and ``exonit/discovery_autofill.py`` is
unchanged by this spec (§3: "no_data(t,S) inalterat"); only the gap-stats
*reporting* side (``exonit.discovery_state.sector_gap_stats``'s new
``failed_terminal`` field) is touched, via this module's
:func:`exhausted_tic_ids`.

Every READER in the codebase (selector, enqueuer, server-side dedup,
gap-stats) is gated behind :func:`is_enabled` (``EXONIT_DONE_STATE``, default
OFF): with the flag off, every call site keeps its pre-existing behaviour
byte-for-byte. Only the WRITE side (mark_tic_done/mark_tic_failed) always
runs — see the deploy order in the spec's §5 for why (dual-write first,
flip readers only once the backfill + a verify pass confirm parity).
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

# Operator decision, spec §7.1: one retry lot per sector before a TIC is
# considered exhausted for that sector (a single terminal worker-failure — 3
# exhausted worker attempts, see db.MAX_TASK_ATTEMPTS — does NOT exhaust the
# TIC; a SECOND terminal failure for the same sector does).
MAX_LOTS_PER_SECTOR = 2

# Sector value written by the backfill (§2.3) for legacy rows with no
# reliable per-run sector attribution, and by any write path that receives
# tasks.sector = NULL (e.g. a hand-run enqueue_from_csv.py without --sector).
LEGACY_SECTOR = 0

# derive_done_set's chunked-probe-vs-full-scan crossover, same rationale and
# measured numbers as exonit.discovery_state.derive_done_set (module-level so
# tests can reference it instead of hard-coding the boundary).
_CHUNK_PROBE_MAX_IDS = 100_000


def is_enabled() -> bool:
    """True when EXONIT_DONE_STATE=1 — gates every READER in this module's
    callers. Writes (mark_tic_done/mark_tic_failed) are never gated."""
    return os.environ.get("EXONIT_DONE_STATE", "0") == "1"


# --------------------------------------------------------------------------- #
# Reads — backend-level primitives
# --------------------------------------------------------------------------- #

def _done_ids(backend) -> set:
    # Q3 (perf review 2026-09-26): iter_id_column streams tuple rows via a
    # server-side cursor on PG instead of query()'s eager dict_row list --
    # same fix, same rationale as discovery_state.load_completed_tic_set,
    # for this table's equivalent full scan.
    return {
        str(tic_id) for tic_id in backend.iter_id_column(
            "SELECT tic_id FROM tic_sector_state WHERE status = 'completed'")
    }


def _exhausted_ids(backend, sector: int) -> set:
    return {
        str(tic_id) for tic_id in backend.iter_id_column(
            "SELECT tic_id FROM tic_sector_state "
            "WHERE sector = ? AND status = 'failed' AND attempts >= ?",
            (sector, MAX_LOTS_PER_SECTOR))
    }


def _done_ids_scoped(backend, ids: list) -> set:
    """done(t) ∩ ids, chunk-probe (bounded ids) or full-scan (sector-scale).

    Same chunk-vs-crossover rationale as
    ``exonit.discovery_state.derive_done_set``: a scattered multi-million-id
    probe against a PK-indexed table costs far more than one full scan +
    Python intersect, so only bounded candidate lists (<= _CHUNK_PROBE_MAX_IDS)
    take the chunked path.
    """
    if len(ids) > _CHUNK_PROBE_MAX_IDS:
        done_all = _done_ids(backend)
        return {t for t in ids if str(t) in done_all}
    from exonit.db_backend import select_in_chunks
    str_ids = [str(t) for t in ids]
    rows = select_in_chunks(
        backend.query,
        "SELECT tic_id FROM tic_sector_state WHERE status = 'completed' "
        "AND tic_id IN",
        str_ids)
    done_strs = {r["tic_id"] for r in rows}
    return {t for t in ids if str(t) in done_strs}


# --------------------------------------------------------------------------- #
# Reads — path/DSN-based (scripts, discovery_state.py)
# --------------------------------------------------------------------------- #

def load_done_tic_set(db_path: str) -> set:
    """Return the set of TIC ids (str) with done(t) == True (any sector).

    Drop-in replacement shape for
    ``exonit.discovery_state.load_completed_tic_set`` when
    ``EXONIT_DONE_STATE=1``.
    """
    from exonit.db_backend import connect_operational
    backend = connect_operational(db_path)
    try:
        return _done_ids(backend)
    finally:
        backend.close()


def exhausted_tic_ids(db_path: str, sector: int) -> set:
    """Return the set of TIC ids (str) with exhausted(t, sector) == True."""
    from exonit.db_backend import connect_operational
    backend = connect_operational(db_path)
    try:
        return _exhausted_ids(backend, sector)
    finally:
        backend.close()


def exhausted_tic_ids_from_backend(backend, sector: int) -> set:
    """Same as :func:`exhausted_tic_ids`, for a caller that already holds an
    open ``Backend`` (e.g. ``exonit.enqueue.enqueue_targets`` via ``db.backend``)."""
    return _exhausted_ids(backend, sector)


def done_and_exhausted(
    db_path: str, sector: int, tic_ids: Iterable, *, done: Optional[set] = None
) -> set:
    """Subset of *tic_ids* that are done(t) (any sector) OR exhausted(t, sector).

    Extends ``exonit.discovery_state.derive_done_set``'s "subset of tic_ids to
    exclude from selection" contract with the attempts-exhausted predicate —
    the (b) requirement of the spec's §1 checklist: a TIC that has failed
    once for this sector (``attempts < MAX_LOTS_PER_SECTOR``) stays
    selectable; a second terminal failure excludes it.

    Parameters
    ----------
    done:
        Optional pre-loaded done(t) set (str), e.g. from
        :func:`load_done_tic_set` — same amortised-scan convention as
        ``derive_done_set``'s ``completed`` parameter. When ``None``, this
        loads/probes it itself (chunk-vs-scan, see :func:`_done_ids_scoped`).
    """
    ids = list(tic_ids)
    if not ids:
        return set()
    from exonit.db_backend import connect_operational
    backend = connect_operational(db_path)
    try:
        base = ({t for t in ids if str(t) in done} if done is not None
                else _done_ids_scoped(backend, ids))
        exhausted = _exhausted_ids(backend, sector)
        base |= {t for t in ids if str(t) in exhausted}
        return base
    finally:
        backend.close()


# --------------------------------------------------------------------------- #
# Reads — backend-based (server route dedup, already holds a Backend/db)
# --------------------------------------------------------------------------- #

def retryable_failed_ids(backend, sectors: Iterable[int]) -> set:
    """TIC ids (str) that are ``failed`` for one of *sectors*, with
    ``attempts < MAX_LOTS_PER_SECTOR`` — i.e. NOT yet exhausted(t, S) for
    that sector (S1, docs/superpowers/specs/2026-09-26-enqueue-s1-s4-
    design.md §4.1.2).

    *sectors* is normally ``(sector, LEGACY_SECTOR)`` — the requirement (b)
    of the done-state design's own §1 checklist ("a single failure must
    stay selectable") covers both this sector's own failures AND the
    legacy/backfilled rows written with ``sector=0`` (spec's own note: "the 0
    covers the legacy/backfill failures").

    Small result by construction (a lot's worth of failures, not the whole
    table) — a plain ``IN (...)`` + status/attempts filter, no chunking
    needed (``sectors`` is at most a handful of values).
    """
    sector_list = list(sectors)
    if not sector_list:
        return set()
    placeholders = ",".join("?" for _ in sector_list)
    rows = backend.query(
        f"SELECT tic_id FROM tic_sector_state WHERE sector IN ({placeholders}) "
        "AND status = 'failed' AND attempts < ?",
        (*sector_list, MAX_LOTS_PER_SECTOR))
    return {r["tic_id"] for r in rows}


def probe_done(backend, ids: Iterable) -> set:
    """done(t) ∩ *ids*, ALWAYS via a chunked probe — never the full-scan
    fallback :func:`_done_ids_scoped` takes past ``_CHUNK_PROBE_MAX_IDS``.

    For callers whose candidate list is bounded by construction (the S1
    paged selector's retry pool and one page at a time — never a whole
    sector), so a full table scan would never be the right trade-off
    regardless of size; unlike :func:`_done_ids_scoped`, this never
    second-guesses that.
    """
    ids_list = list(ids)
    if not ids_list:
        return set()
    from exonit.db_backend import select_in_chunks
    str_ids = [str(t) for t in ids_list]
    rows = select_in_chunks(
        backend.query,
        "SELECT tic_id FROM tic_sector_state WHERE status = 'completed' "
        "AND tic_id IN",
        str_ids)
    done_strs = {r["tic_id"] for r in rows}
    return {t for t in ids_list if str(t) in done_strs}


def blocked_for_enqueue(backend, sector: int, tic_ids: Iterable) -> set:
    """Subset of *tic_ids* that are done(t) OR exhausted(t, sector).

    Used by ``/api/worker/enqueue_bulk``'s dedup (absent ``force_rescan``):
    request-sized *tic_ids* (a lot, not a whole sector), so a chunked probe
    is always the right strategy here — never the full-scan fallback (see
    :func:`done_and_exhausted` for the sector-scale selector path instead).
    """
    ids = list(tic_ids)
    if not ids:
        return set()
    from exonit.db_backend import select_in_chunks
    str_ids = [str(t) for t in ids]
    done_rows = select_in_chunks(
        backend.query,
        "SELECT tic_id FROM tic_sector_state WHERE tic_id IN",
        str_ids, sql_after=" AND status = 'completed'")
    exhausted_rows = select_in_chunks(
        backend.query,
        "SELECT tic_id FROM tic_sector_state WHERE tic_id IN",
        str_ids,
        sql_after=" AND sector = ? AND status = 'failed' AND attempts >= ?",
        params_extra=(sector, MAX_LOTS_PER_SECTOR))
    blocked = ({r["tic_id"] for r in done_rows}
               | {r["tic_id"] for r in exhausted_rows})
    return {t for t in ids if str(t) in blocked}


# --------------------------------------------------------------------------- #
# Writes — always on, called from the worker-result path
# --------------------------------------------------------------------------- #

def _resolve_task_sector(backend, task_id) -> int:
    """The LOT's sector for *task_id* (``tasks.sector``), or LEGACY_SECTOR
    (0) when the task row is missing or its sector is NULL (unattributed —
    see the spec's §5 risk #2)."""
    if task_id is None:
        return LEGACY_SECTOR
    row = backend.query_one(
        "SELECT sector FROM tasks WHERE task_id = ?", (task_id,))
    if row is None or row.get("sector") is None:
        return LEGACY_SECTOR
    return int(row["sector"])


_MARK_DONE_COLS = ("tic_id", "sector", "status", "attempts", "last_task_id",
                   "updated_at")


def mark_tic_done(backend, tic_id, task_id=None) -> None:
    """Upsert (tic_id, sector) as ``completed`` (§2.2 hot path).

    A success always sets status='completed', whatever it was before — this
    is the write side of done(t)'s monotonicity (a later failure elsewhere
    cannot undo it, and a later success for the SAME (tic, sector) simply
    reconfirms it). ``attempts`` is only ever incremented by
    :func:`mark_tic_failed`; a success does not reset it.
    """
    sector = _resolve_task_sector(backend, task_id)
    sql = backend.dialect.upsert_on_conflict(
        "tic_sector_state", _MARK_DONE_COLS,
        conflict_cols=("tic_id", "sector"),
        update_cols=("status", "last_task_id", "updated_at"),
        raw_values={"updated_at": backend.dialect.now()})
    backend.execute(sql, (str(tic_id), sector, "completed", 0, task_id))


def mark_tic_failed(backend, tic_id, task_id=None, error=None) -> None:
    """Upsert (tic_id, sector) as a terminal failure (§2.2 hot path).

    ONLY for a task's terminal failure (``db.requeue_or_fail_task`` returning
    'failed', i.e. worker-attempts exhausted) — an internal requeue
    ('pending') does not call this. ``attempts`` counts LOT-level terminal
    failures for this (tic, sector), capped in spirit at
    ``MAX_LOTS_PER_SECTOR`` by the exhausted(t,S) predicate (the counter
    itself is not clamped — it is a monotonically increasing count of how
    many times this happened, useful for diagnostics beyond the threshold).

    A prior ``completed`` status for this exact (tic, sector) is NEVER
    downgraded by a later failure — the ``CASE`` below preserves it. This
    upsert is written as one dialect-neutral SQL string (``?`` placeholders +
    ``datetime('now')``, auto-translated for PG by
    ``PgBackend.execute``/``PgDialect.translate``) because
    ``Dialect.upsert_on_conflict`` cannot express the ``attempts = attempts +
    1`` self-referential increment or the ``status`` CASE guard.
    """
    sector = _resolve_task_sector(backend, task_id)
    sql = (
        "INSERT INTO tic_sector_state "
        "(tic_id, sector, status, attempts, last_task_id, last_error, "
        "updated_at) "
        "VALUES (?, ?, 'failed', 1, ?, ?, datetime('now')) "
        "ON CONFLICT (tic_id, sector) DO UPDATE SET "
        "attempts = tic_sector_state.attempts + 1, "
        "status = CASE WHEN tic_sector_state.status = 'completed' "
        "THEN 'completed' ELSE 'failed' END, "
        "last_task_id = excluded.last_task_id, "
        "last_error = excluded.last_error, "
        "updated_at = datetime('now')"
    )
    backend.execute(sql, (str(tic_id), sector, task_id, error))
