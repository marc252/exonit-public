"""Machine-readable sector registry for the ExoNit discovery campaign.

The registry file (data/discovery/sectors.json) is the canonical source of
truth for sector state, cursor position, and downloaded/done counts.
CAMPAIGN.md is the human log; this module is the machine interface.

All functions take explicit paths so tests can use tmp files without touching
the live DB or registry.
"""

import json
import os
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


# ------------------------------------------------------------------ #
# Default registry content (returned when file is missing)
# ------------------------------------------------------------------ #

_DEFAULT_REGISTRY: dict = {
    "active_sector": 1,
    "sectors": {
        "1": {
            "target_list": "data/discovery/target_lists/s0001.csv",
            "canonical_list": "data/discovery/s0001_batch0001_sample.csv",
            "total_tics": 2464267,
            "downloaded": 5000,
            "cursor": 3500,
            "enabled": True,
        }
    },
}


def load_registry(path: str) -> dict:
    """Load the sector registry from *path*.

    Returns a deep copy of _DEFAULT_REGISTRY when the file is absent or
    unreadable, so callers always get a usable dict.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        import copy
        return copy.deepcopy(_DEFAULT_REGISTRY)


def save_registry(path: str, data: dict) -> None:
    """Atomically write *data* as JSON to *path* (temp-file + rename)."""
    dir_ = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def get_sector(data: dict, n: int) -> Optional[dict]:
    """Return the sector entry for sector *n* from *data*, or None."""
    return data.get("sectors", {}).get(str(n))


def _set_sector_key(path: str, sector: int, key: str, value) -> None:
    """Load registry, set one key on the sector entry, save atomically."""
    data = load_registry(path)
    entry = data.setdefault("sectors", {}).setdefault(str(sector), {})
    entry[key] = value
    save_registry(path, data)


def set_cursor(path: str, sector: int, cursor: int) -> None:
    """Load registry, set cursor for *sector*, save atomically."""
    _set_sector_key(path, sector, "cursor", cursor)


def set_cursor_seq(path: str, sector: int, cursor_seq: int, cursor: int) -> None:
    """S1 (docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md §4.1.3):
    persist the paged selector's ``seq``-based cursor alongside the legacy
    row-count ``cursor``.

    Also stores ``cursor_seq_for_cursor = cursor`` (the ``cursor`` value
    ``cursor_seq`` was computed against) — the rollback-safety net the
    spec's §4.5 risk note calls for: if a legacy (non-paged) lot ever moves
    ``cursor`` without going through this function, the paged selector's
    NEXT bootstrap must re-derive ``cursor_seq`` from scratch rather than
    trust a now-stale value for a different ``cursor``. See
    :func:`get_cursor_seq_if_valid`.
    """
    data = load_registry(path)
    entry = data.setdefault("sectors", {}).setdefault(str(sector), {})
    entry["cursor_seq"] = cursor_seq
    entry["cursor_seq_for_cursor"] = cursor
    save_registry(path, data)


def get_cursor_seq_if_valid(sector_entry: dict) -> Optional[int]:
    """Return the sector entry's persisted ``cursor_seq`` IF it was computed
    against the entry's CURRENT ``cursor`` value, else ``None`` (forces the
    paged selector to re-bootstrap — see :func:`set_cursor_seq`'s
    docstring).
    """
    if sector_entry.get("cursor_seq_for_cursor") != sector_entry.get("cursor", 0):
        return None
    return sector_entry.get("cursor_seq")


def set_deep_cursor(path: str, sector: int, cursor: int) -> None:
    """Deep-rescan progress cursor — independent of the campaign cursor.

    Row index (1-based) into the deep-pass worklist CSV of the last TIC
    already enqueued by a deep lot; the campaign cursor semantics are
    untouched (spec 2026-07-23 §4).
    """
    _set_sector_key(path, sector, "deep_cursor", cursor)


def active_sector(path: str = "data/discovery/sectors.json",
                  env_override=None) -> int:
    """Return the sector this deployment should process.

    Precedence: explicit *env_override* arg → ``EXONIT_SECTOR`` env var →
    registry ``active_sector`` → 1. This replaces the old hardcoded sector
    constants: home runs sector 1 (registry default), an AWS deployment ships
    a registry with ``active_sector: 2``. The env override exists for one-off
    runs/tests without editing the registry.
    """
    ov = env_override if env_override is not None else os.environ.get("EXONIT_SECTOR")
    if ov not in (None, ""):
        return int(ov)
    data = load_registry(path)
    return int(data.get("active_sector", 1))


# ------------------------------------------------------------------ #
# Run identity (S3, docs/superpowers/specs/2026-09-26-enqueue-s1-s4-
# design.md §3.1) — single canonical derive_run_id, replacing the three
# textually-identical copies that used to live in scripts/discovery_run.py
# (new, S3), scripts/discovery_orchestrator.py's _derive_run_id, and
# exonit/routes/discovery.py's scan-endpoint minting block. Those two call
# sites keep their OWN thin wrappers (test suites patch/import them by name
# — e.g. tests/test_discovery_orchestrator.py's
# ``patch.object(orch, "_derive_run_id")`` and
# tests/test_discovery_api.py's direct ``_lot_prefix_from_csv``/
# ``_lot_stamp_from_csv`` imports), but every one of them now calls THIS
# function (or, for the two private helpers, is functionally identical to
# it) rather than re-deriving the ``lot<ts>_<date>`` string independently —
# the exact drift risk the spec's §7.3 risk item calls out.
# ------------------------------------------------------------------ #

def _lot_number_from_csv(lot_csv: str) -> str:
    """Unique lot identifier for the run_id, from the CSV filename.

    Byte-for-byte copy of scripts/discovery_orchestrator.py's
    ``_lot_number_from_csv`` / exonit/routes/discovery.py's
    ``_lot_stamp_from_csv`` (three names, one contract — see this section's
    module comment):
    - Auto lots ``s0001_lot_20260531T082820.csv`` -> ``'20260531T082820'``
      (timestamp; unique per run, so two runs the same day don't collide).
    - Numbered lots ``s0001_lot4.csv`` -> ``'4'``.
    - Fallback -> ``'X'``.
    """
    name = Path(lot_csv).name
    m = re.search(r"lot_(\d{8}T\d{6})", name)   # auto lots -> timestamp
    if m:
        return m.group(1)
    m = re.search(r"lot(\d+)", name)             # numbered lots -> number
    return m.group(1) if m else "X"


def _lot_prefix_from_csv(lot_csv: str) -> str:
    """``"lot"`` or ``"deep_lot"`` run_id segment prefix, from the lot CSV.

    Byte-for-byte copy of the two per-file helpers of the same name (see
    this section's module comment): deep-rescan lots (``s<sector>_deep_lot_
    <ts>.csv``, written by ``discovery_run.py``'s --force-rescan branch)
    mint a distinctive ``discovery_s<sector>_deep_lot<ts>_<date>`` run_id;
    every other lot CSV name gets the plain ``"lot"`` prefix.
    """
    return "deep_lot" if "_deep_lot_" in Path(lot_csv).name else "lot"


def derive_run_id(sector, lot_csv: str, now_utc: Optional[datetime] = None) -> str:
    """Canonical run_id: ``discovery_s<sector:04d>_lot<lot>_<YYYY-MM-DD>``
    (UTC "today"); deep-rescan lots mint ``discovery_s<sector:04d>_deep_lot
    <lot>_<YYYY-MM-DD>`` instead (see ``_lot_prefix_from_csv``).

    THE single source of truth for this string (S3 §3.1's "both-sites-must-
    agree" invariant) — every minting site (scripts/discovery_run.py at
    lot_csv-selection time; the scan endpoint's post-subprocess mint,
    exonit/routes/discovery.py; the orchestrator's manual-CLI fallback,
    scripts/discovery_orchestrator.py's ``_derive_run_id``) calls this
    function (directly, or a thin same-signature wrapper of it) instead of
    re-deriving the format.

    Parameters
    ----------
    now_utc:
        Injected "now" for tests (the mint uses UTC "today" — a local-vs-UTC
        midnight straddle used to be able to mint two different run_ids for
        the same lot_csv across the two independent copies this replaces).
        Defaults to ``datetime.now(timezone.utc)``.
    """
    today = (now_utc or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    return (f"discovery_s{int(sector):04d}_"
            f"{_lot_prefix_from_csv(lot_csv)}{_lot_number_from_csv(lot_csv)}_{today}")


# ------------------------------------------------------------------ #
# Extended canonical list
# ------------------------------------------------------------------ #

def _load_canonical_tics(canonical_csv_path: str) -> tuple:
    """Return ``(ordered_unique_tics, tics_set)`` parsed from the canonical CSV.

    Shared by ``load_extended_canonical`` (which needs the ORDER) and
    ``count_extended_canonical`` (which needs only the set), so the two can
    never drift on what counts as a valid row: non-numeric, empty and
    ``"nan"`` tic_ids are skipped, duplicates keep their first position.
    A missing file yields ``([], set())`` — the caller then works off the
    TGLC index alone.
    """
    import csv as _csv

    frozen: list = []
    frozen_set: set = set()
    try:
        with open(canonical_csv_path, newline="", encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                raw = row.get("tic_id", "").strip()
                if not raw or raw == "nan":
                    continue
                try:
                    t = int(raw)
                except ValueError:
                    continue
                if t not in frozen_set:
                    frozen.append(t)
                    frozen_set.add(t)
    except FileNotFoundError:
        pass  # canonical missing — extension will be the entire index
    return frozen, frozen_set


def count_extended_canonical(
    canonical_csv_path: str,
    tglc_index_path: str,
    sector: Optional[int] = None,
) -> int:
    """``len(load_extended_canonical(...))`` WITHOUT materialising the list.

    Same number, flat memory. The list version allocates four 8M-element
    Python objects for sector 14 (the DB rows, the extension list, the
    concatenated result — plus, until 2026-07-27, a dead ``set``); callers
    that only want the SIZE were paying multi-GB allocation spikes on the
    server for a number that SQL can count. Measured on <host> 2026-07-27:
    the uvicorn workers ratcheted to ~5 GB RSS each with transient +3 GB
    spikes under run load; this path was the only one in the hot set that
    builds anything of that order.

    Exactness contract — this MUST agree with ``len(load_extended_canonical
    (...))`` for every input, including the awkward ones:
      - Duplicate ``tglc`` rows for the same TIC are counted once per ROW
        there (the extension list-comprehension does not dedupe among
        indexed TICs), so a plain ``COUNT(*)`` is the faithful count, not
        ``COUNT(DISTINCT tic_id)``.
      - TICs present in BOTH inputs are counted once (they stay in the
        frozen prefix and are filtered out of the extension), so the
        overlap has to be subtracted.
      - A missing/absent TGLC table yields the canonical count alone,
        mirroring the list version's ``except`` path.
    """
    frozen, frozen_set = _load_canonical_tics(canonical_csv_path)

    try:
        from .metadata_cache import MetadataCache
        mc = MetadataCache(tglc_index_path)
        try:
            total = mc.tglc_sector_count(sector)
            if total is None:
                return len(frozen)  # table not there yet — canonical only
            if not frozen_set:
                # No canonical list (the sector-14/15/17 case): every indexed
                # row is extension, so one COUNT(*) is the whole answer.
                return int(total)
            overlap = mc.tglc_rowcount_for_tics(frozen_set, sector)
        finally:
            mc.close()
    except (sqlite3.OperationalError, OSError):
        return len(frozen)  # tglc DB missing — no extension

    return len(frozen) + int(total) - int(overlap)


def load_extended_canonical(
    canonical_csv_path: str,
    tglc_index_path: str,
    sector: Optional[int] = None,
    *,
    return_indexed_set: bool = False,
):
    """Return the ordered work list = canonical CSV rows + any extra TICs in the TGLC index.

    When ``sector`` is given, the TGLC extension is restricted to that sector's
    rows. This matters once the index holds more than one sector: the tglc rowid
    order interleaves sectors, so an unfiltered work list for a "sector 2" run
    would pick undone sector-1 TICs first. Filtering keeps a per-sector run pure.

    The canonical CSV defines the initial batch in its row order (first
    occurrence of each tic_id is kept; duplicates are dropped while
    preserving the first position).  Any TIC that is present in the TGLC
    index but not in the canonical list is appended in ``rowid ASC`` order
    (roughly the order they were indexed).

    Parameters
    ----------
    canonical_csv_path:
        Path to the canonical CSV (columns: tic_id, label).  When the file
        is absent the list is built from the TGLC index alone.
    tglc_index_path:
        Path to the metadata_cache DB (table ``tglc``, column ``tic_id``) —
        a SQLite path, or a ``postgresql://`` DSN
        (docs/superpowers/specs/2026-07-20-tglc-index-to-pg.md).  When
        absent, only the canonical rows are returned.
    return_indexed_set:
        Q2 (perf review 2026-09-26, docs/reviews/2026-09-26-enqueue-path-
        performance-review.md): when ``True``, also return the ``set`` of
        TIC ids this call already pulled from the TGLC index (the same
        rows ``exonit.discovery_state._load_tglc_set``/
        ``scripts.discovery_run._load_downloaded_set`` would fetch with a
        SECOND, separate query for the very same sector). Callers that need
        "is this TIC downloaded?" (the downloaded_set input to
        ``next_block``) get it for free from the query this function makes
        anyway, instead of dumping the whole sector's ``tglc`` rows twice
        per lot. Default ``False`` keeps the return type a plain ``list``
        for every existing caller (back-compat).

    Returns
    -------
    list[int]
        Deduplicated, ordered work list:
        ``frozen_canonical_tics + extension_from_tglc``.
        Returns ``[]`` when both inputs are missing.
    tuple[list[int], set[int]]
        ``(work_list, indexed_set)`` when *return_indexed_set* is ``True``.
        ``indexed_set`` is exactly what
        ``_load_tglc_set``/``_load_downloaded_set`` would have returned for
        the same *tglc_index_path*/*sector* (empty set when the index is
        missing/unreadable, same as those functions).
    """
    # --- load canonical CSV (frozen list) ---
    frozen, frozen_set = _load_canonical_tics(canonical_csv_path)

    # --- load TGLC index (insertion-order — SQLite rowid / PG seq) ---
    # Routed through exonit/metadata_cache.py's MetadataCache.tglc_tic_ids
    # (the metadata_backend.py seam) instead of a raw sqlite3.connect — see
    # the spec's D3/item (e). Ordering contract unchanged on SQLite.
    indexed: list = []
    try:
        from .metadata_cache import MetadataCache
        mc = MetadataCache(tglc_index_path)
        try:
            indexed = mc.tglc_tic_ids(sector)
        finally:
            mc.close()
    except (sqlite3.OperationalError, OSError):
        pass  # tglc DB missing — no extension

    # --- build extension: indexed TICs not already in canonical ---
    # (a `set(indexed)` used to be built here and never read — for sector 14
    # that was ~8M ints of pure waste on every call; removed 2026-07-27.
    # Q2 below deliberately reintroduces it, but only when a caller asks for
    # it via return_indexed_set — no cost for callers that don't.)
    extension = [t for t in indexed if t not in frozen_set]

    result = frozen + extension

    # Log breakdown when running as a script (logger may not exist; use print
    # only when actually running discovery_run.py via __main__; here we just
    # return the list and let the caller log if it wants).
    if return_indexed_set:
        return result, set(indexed)
    return result


# ------------------------------------------------------------------ #
# DB query helpers
# ------------------------------------------------------------------ #

def load_completed_tic_set(db_path: str) -> set:
    """Return the set of TIC ids (as str) with at least one completed task.

    This is the expensive query at campaign scale (millions of completed
    rows). Pulling it ONCE and reusing it across every sector's gap
    computation is the difference between a snappy ``/api/discovery/sectors``
    and a 30 s hang: the previous code re-ran this full scan once per sector.

    tic_id is stored as TEXT; the result is index-backed by
    idx_tasks_status_priority. Routed through connect_operational so a
    ``postgresql://`` DSN works just like a SQLite path.

    With ``EXONIT_DONE_STATE=1`` (done-state design, 2026-09-26 — see
    :mod:`exonit.done_state`), delegates to
    ``done_state.load_done_tic_set()`` — done(t) via ``tic_sector_state``
    (monotonic: a later failure elsewhere never removes a TIC from this set).
    Default OFF: unchanged (``tasks``-based, exactly as before this change).
    """
    from exonit import done_state
    if done_state.is_enabled():
        return done_state.load_done_tic_set(db_path)
    # Q3 (perf review 2026-09-26): streams via Backend.iter_id_column (tuple
    # rows, server-side cursor on PG) instead of query() -- this is the
    # single query behind the ~27.6M-row completed-set scan measured at a
    # ~7 GB Python peak (one dict_row per row, materialized as one list,
    # before this set comprehension could discard everything but the id).
    from exonit.db_backend import connect_operational
    backend = connect_operational(db_path)
    try:
        return {
            str(tic_id) for tic_id in backend.iter_id_column(
                "SELECT DISTINCT tic_id FROM tasks WHERE status = 'completed'")
        }
    finally:
        backend.close()


def derive_done_set(
    db_path: str, tic_ids: Iterable[int], *, completed: Optional[set] = None,
    sector: Optional[int] = None,
) -> set:
    """Return the subset of *tic_ids* that have at least one completed task.

    Read-only; never writes.

    Parameters
    ----------
    db_path:
        Path to the operational DB (SQLite path or ``postgresql://`` DSN).
    tic_ids:
        Iterable of integer TIC ids to check.
    completed:
        Optional pre-loaded set of completed TIC ids (str), e.g. from
        :func:`load_completed_tic_set`.  When provided, the DB is NOT queried —
        the caller has already paid for the scan and we just intersect.  This
        is how the sectors endpoint computes all sectors from a single scan.
        When ``None``, the completed set is loaded here (back-compat).
    sector:
        The sector this selection is for. With ``EXONIT_DONE_STATE=1`` AND a
        *sector* given, the returned set is widened with exhausted(t, sector)
        (done-state design §1(b): a TIC failed twice for THIS sector is
        excluded from selection exactly like a completed one, via
        :func:`exonit.done_state.done_and_exhausted`). Ignored (as before)
        when the flag is off or *sector* is ``None`` — the "no_data(t,S)
        unchanged" note in the spec's §3 table applies: this never touches
        ``discovery_no_data``.

    Returns
    -------
    set[int]
        Subset of *tic_ids* where a task with status='completed' exists (or,
        with the flag on and *sector* given, is otherwise exhausted for that
        sector).
    """
    ids = list(tic_ids)
    if not ids:
        return set()

    from exonit import done_state
    if done_state.is_enabled() and sector is not None:
        return done_state.done_and_exhausted(
            db_path, sector, ids, done=completed)

    # Load the completed-TIC set once and intersect in Python.
    #
    # The previous implementation built "WHERE tic_id IN (?,?,...)" with one
    # placeholder per id. At campaign scale `tic_ids` is the whole work list
    # (531k+ after the FFI index grew), which blew past SQLite's
    # SQLITE_MAX_VARIABLE_NUMBER → "too many SQL variables" and crashed the scan
    # launch (HTTP 400, no run). Querying the completed set once and intersecting
    # in Python scales to any work-list size. tic_id is stored as TEXT, so
    # compare on the string form; return the matching input ids.
    if completed is None:
        completed = load_completed_tic_set(db_path)

    return {t for t in ids if str(t) in completed}


# ------------------------------------------------------------------ #
# Gap-metric helper (Tasks 2–3 of the UI source-of-truth plan)
# ------------------------------------------------------------------ #

def _load_tglc_set(tglc_index_path: str, sector: Optional[int] = None) -> set:
    """Return the set of integer TIC IDs present in the TGLC index for *sector*.

    TICs in this set have had their FITS file downloaded and indexed.  Absent
    TICs (in the canonical CSV but not here) still need a download.

    Returns an empty set when the index file is missing or the table does not
    exist (so callers degrade gracefully).

    Routed through MetadataCache.tglc_tic_ids (exonit/metadata_backend.py's
    seam) instead of a raw sqlite3.connect — see load_extended_canonical's
    equivalent note above. Order doesn't matter here (result is a set), so
    this is safe on either backend without an ordering decision.
    """
    try:
        from .metadata_cache import MetadataCache
        mc = MetadataCache(tglc_index_path)
        try:
            return set(mc.tglc_tic_ids(sector))
        finally:
            mc.close()
    except (sqlite3.OperationalError, OSError):
        return set()


def load_toi_tic_set(catalog_path: str) -> set:
    """Return the set of int TIC ids in the TOI catalog CSV.

    Mirrors ``scripts/discovery_run._load_toi_tic_set`` exactly:
    column "TIC ID" (exact string), ``int(float(raw))``, skip blanks.
    Returns an empty set when the file is missing so callers always
    get a usable value (no crash).
    """
    import csv as _csv
    import warnings as _warnings
    result: set = set()
    try:
        with open(catalog_path, newline="", encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                raw = row.get("TIC ID", "").strip()
                if not raw:
                    continue
                try:
                    result.add(int(float(raw)))
                except (ValueError, TypeError):
                    pass
    except FileNotFoundError:
        _warnings.warn(
            f"[discovery_state] TOI catalog not found at {catalog_path!r}; "
            "TOI subtraction disabled.",
            RuntimeWarning,
            stacklevel=2,
        )
    return result


def sector_gap_stats(
    sector: int,
    *,
    db_path: str,
    canonical_csv: str,
    tglc_index: str,
    completed: Optional[set] = None,
    toi_catalog: Optional[str] = None,
) -> dict:
    """Return the real gap metric for a sector.

    Crosses three data sources:
    - The TGLC work list (canonical CSV + tglc index) for the total size.
    - The ``tasks`` table (via ``derive_done_set``) for completed count.
    - The ``discovery_no_data`` table in PG for the dataless-TIC count.

    Parameters
    ----------
    toi_catalog:
        Path to the TOI catalog CSV (column "TIC ID").  When ``None``,
        defaults to ``<project_root>/data/toi_catalog.csv``.  When the
        file is absent the loader returns an empty set and the metric
        behaves exactly as before this parameter was added.

    Returns
    -------
    dict with keys:
        work_list_size        — total TICs in the sector work list
        completed             — TICs with at least one completed task
        no_data               — TICs marked as permanently dataless
        tois_skipped          — work-list TOIs not in completed or no_data
                                (the scanner skips these; they are not
                                real discovery gaps)
        gaps_processable      — work_list_size - completed - no_data
                                - tois_skipped (≥0)
        pct_complete          — (completed + no_data + tois_skipped)
                                / work_list_size
        missing_download_tics — canonical TICs with no FITS in the index,
                                not yet completed, not marked no_data, and
                                not a TOI; these are the TICs the autofill
                                manager should attempt to re-download before
                                marking them as no_data.

    V2 dispatch (S2, 2026-09-26, docs/superpowers/specs/2026-09-26-enqueue-
    s1-s4-design.md §5): with ``EXONIT_GAP_STATS_V2=1`` AND
    ``EXONIT_DONE_STATE=1``, delegates to
    :func:`exonit.gap_stats_v2.sector_gap_stats_v2` — same output keys/
    semantics, computed via a ``postgres_fdw`` server-side count instead of
    the whole-DB completed-set scan below. Falls back to this legacy path
    (with a ``RuntimeWarning``) when V2 is requested without done-state
    (misconfiguration), and silently for a non-PostgreSQL operational DB
    (SQLite has no ``postgres_fdw`` — dev/tests always use this legacy path,
    per the spec's "SQLite keeps legacy path"). A genuine V2 runtime failure
    (fdw down, ``meta_fdw.tglc`` absent, statement timeout) is NOT caught
    here — it propagates to the caller; the stale-while-error policy lives in
    the background gap-stats cache refresh (``exonit/routes/discovery.py``),
    which must never fall back to this legacy path on an fdw error (spec
    §5.2 — that whole-DB scan is exactly what S2 exists to avoid running in
    a live uvicorn worker).
    """
    from exonit import gap_stats_v2 as _gsv2
    if _gsv2.is_enabled():
        from exonit import done_state as _done_state_check
        if not _done_state_check.is_enabled():
            import warnings as _warnings
            _warnings.warn(
                "[gap_stats] EXONIT_GAP_STATS_V2=1 requires EXONIT_DONE_STATE=1; "
                "falling back to the legacy whole-DB gap-stats path for this call.",
                RuntimeWarning, stacklevel=2)
        else:
            try:
                return _gsv2.sector_gap_stats_v2(
                    sector, db_path=db_path, canonical_csv=canonical_csv,
                    tglc_index=tglc_index, toi_catalog=toi_catalog)
            except _gsv2.FdwUnavailable:
                pass  # SQLite operational DB — legacy path below, no warning.

    import csv as _csv
    from pathlib import Path as _Path

    # --- Resolve TOI catalog path and load the skip set ---
    _toi_cat = (
        toi_catalog
        if toi_catalog is not None
        else str(_Path(__file__).resolve().parents[1] / "data" / "toi_catalog.csv")
    )
    toi_set = load_toi_tic_set(_toi_cat)

    # --- Load canonical TICs (frozen list from CSV only) ---
    canonical_tics: list = []
    canonical_set: set = set()
    try:
        with open(canonical_csv, newline="", encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                raw = row.get("tic_id", "").strip()
                if not raw or raw == "nan":
                    continue
                try:
                    t = int(raw)
                except ValueError:
                    continue
                if t not in canonical_set:
                    canonical_tics.append(t)
                    canonical_set.add(t)
    except FileNotFoundError:
        pass

    # --- Load TGLC set (downloaded TICs) ---
    tglc_set = _load_tglc_set(tglc_index, sector)

    # --- Build full work list (canonical + TGLC extension), same as
    #     load_extended_canonical but we already have both sets. ---
    # Extension: TGLC TICs not in the canonical list (rowid order is lost here,
    # but for gap counting the order does not matter).
    work_list: list = canonical_tics + [t for t in tglc_set if t not in canonical_set]
    work_list_size = len(work_list)

    # PG-aware intersection. ``completed`` (the expensive completed-TIC set) may
    # be passed in by a caller computing several sectors from a single scan; if
    # absent, derive_done_set loads it itself.
    done = derive_done_set(db_path, work_list, completed=completed)
    completed_count = len(done)

    from exonit.db_backend import connect_operational
    b = connect_operational(db_path)
    try:
        # Fetch the TIC IDs in no_data so we can both count them and filter
        # missing_download_tics in one query.
        no_data_rows = b.query(
            "SELECT tic_id FROM discovery_no_data WHERE sector = ?",
            (sector,),
        )
        no_data_tic_strs: set = {r["tic_id"] for r in no_data_rows}
        nd = len(no_data_tic_strs)
    finally:
        b.close()

    # --- TOI-skipped count ---
    # TOIs in the work list that are NOT already counted as completed or
    # no_data.  The scanner skips them (they are not discovery targets), so
    # they must not be counted as gaps.  We never double-subtract: if a TOI
    # also has a completed task it is already in `done`; if it is in no_data
    # it is already counted in `nd`.
    toi_skipped = sum(
        1 for t in work_list
        if t in toi_set and t not in done and str(t) not in no_data_tic_strs
    )

    # --- failed_terminal (done-state design, 2026-09-26 §3) ---
    # exhausted(t, sector): failed twice for THIS sector (see
    # done_state.MAX_LOTS_PER_SECTOR). Counts toward pct_complete and out of
    # gaps_processable — a lot will never make progress on these again — but
    # is reported separately so the UI/verify script can distinguish "really
    # done" from "given up on". Gated behind EXONIT_DONE_STATE: with the flag
    # off this is always 0 and the returned dict has NO "failed_terminal" key
    # at all, so callers see the exact same shape/values as before this
    # change (spec requirement: readers default to current behaviour).
    failed_terminal = 0
    from exonit import done_state as _done_state
    if _done_state.is_enabled():
        exhausted_strs = _done_state.exhausted_tic_ids(db_path, sector)
        failed_terminal = sum(
            1 for t in work_list
            if str(t) in exhausted_strs
            and t not in done
            and str(t) not in no_data_tic_strs
            and t not in toi_set
        )

    gaps = max(0, work_list_size - completed_count - nd - toi_skipped
              - failed_terminal)
    pct = ((completed_count + nd + toi_skipped + failed_terminal)
          / work_list_size if work_list_size else 0.0)

    # --- missing_download_tics ---
    # Canonical TICs that have no FITS in the TGLC index AND are not yet
    # completed AND are not already marked as no_data AND are not TOIs.
    # Only canonical TICs can be "awaiting download": TGLC extension TICs
    # are by definition already indexed (downloaded).  TOIs are excluded
    # because the scanner skips them — they will never be processed and
    # should not be handed to the autofill manager.
    done_strs: set = {str(t) for t in done}
    missing_download_tics: list = [
        t for t in canonical_tics
        if t not in tglc_set
        and str(t) not in done_strs
        and str(t) not in no_data_tic_strs
        and t not in toi_set
    ]

    result = {
        "work_list_size": work_list_size,
        "completed": completed_count,
        "no_data": nd,
        "tois_skipped": toi_skipped,
        "gaps_processable": gaps,
        "pct_complete": pct,
        "missing_download_tics": missing_download_tics,
    }
    # "failed_terminal" is ONLY present with EXONIT_DONE_STATE=1 — with the
    # flag off, `failed_terminal` is always 0 here (gaps/pct above are then
    # numerically identical to the pre-existing formula too), and omitting
    # the key keeps the returned shape byte-identical to before this change
    # for every caller that does `dict.get("failed_terminal")`-style access.
    if _done_state.is_enabled():
        result["failed_terminal"] = failed_terminal
    return result


def get_sector_state(sector: int, *, db_path: str) -> str:
    """Return the UI-controlled state string for *sector*.

    Reads from the ``discovery_sector_state`` PG table.
    Returns ``'coming_soon'`` when the sector has no row (not yet set).
    """
    from exonit.db_backend import connect_operational
    b = connect_operational(db_path)
    try:
        row = b.query_one(
            "SELECT state FROM discovery_sector_state WHERE sector = ?",
            (sector,),
        )
        return row["state"] if row else "coming_soon"
    finally:
        b.close()


def set_sector_state(
    sector: int,
    state: str,
    *,
    db_path: str,
    detail: str = "",
) -> None:
    """Upsert the UI-controlled state for *sector* in ``discovery_sector_state``.

    Allowed *state* values: ``coming_soon``, ``active``, ``paused``, ``complete``.
    ``updated_at`` is always refreshed to now().
    """
    from exonit.db_backend import connect_operational
    b = connect_operational(db_path)
    try:
        # Routed through the Dialect.upsert_on_conflict() seam (D13,
        # 2026-07-02 cleanup review) instead of a hand-built ON CONFLICT DO
        # UPDATE string — behaves identically on both SQLite (used by
        # tests/test_discovery_autofill_manager.py) and PG (production).
        # updated_at is a raw SQL expression (dialect-translated
        # datetime('now') -> NOW() on PG) so the DB clock stays authoritative
        # — not a bound Python-side value.
        sql = b.dialect.upsert_on_conflict(
            "discovery_sector_state",
            ("sector", "state", "detail", "updated_at"),
            conflict_cols=("sector",),
            raw_values={"updated_at": "datetime('now')"},
        )
        b.execute(sql, (sector, state, detail))
    finally:
        b.close()


# ------------------------------------------------------------------ #
# Auto-fill state helpers (Task 7 of the UI source-of-truth plan)
# ------------------------------------------------------------------ #

#: Columns that callers may pass to set_autofill().  Any unknown kwarg is
#: silently dropped so callers don't need to import this set.
_AUTOFILL_ALLOWED_FIELDS = frozenset(
    {"status", "current_lot", "lots_done", "anomalies", "last_alert"}
)


def get_autofill(sector: int, *, db_path: str) -> Optional[dict]:
    """Return the auto-fill row for *sector* as a dict, or None when absent.

    Reads from the ``discovery_autofill`` PG table.  Returns ``None`` when
    the sector has no row (auto-fill has never been started for it).

    Keys in the returned dict match the column names:
    ``sector``, ``status``, ``current_lot``, ``lots_done``, ``anomalies``,
    ``last_alert``, ``updated_at``.
    """
    from exonit.db_backend import connect_operational
    b = connect_operational(db_path)
    try:
        return b.query_one(
            "SELECT sector, status, current_lot, lots_done, anomalies, "
            "last_alert, updated_at "
            "FROM discovery_autofill WHERE sector = ?",
            (sector,),
        )
    finally:
        b.close()


def set_autofill(sector: int, *, db_path: str, **fields) -> None:
    """Upsert auto-fill state for *sector* in ``discovery_autofill``.

    Only the keyword arguments present in *fields* are written; unspecified
    columns retain their existing values (or table-level defaults on first
    insert).  ``updated_at`` is always refreshed to now().

    Allowed field names: ``status``, ``current_lot``, ``lots_done``,
    ``anomalies``, ``last_alert``.  Unknown keys are silently ignored so
    callers can pass a wider dict without error.

    Parameters
    ----------
    sector:
        The sector integer (primary key).
    db_path:
        Path to the SQLite DB or a ``postgresql://`` DSN.
    **fields:
        Field name → value pairs to write.
    """
    safe = {k: v for k, v in fields.items() if k in _AUTOFILL_ALLOWED_FIELDS}

    from exonit.db_backend import connect_operational
    b = connect_operational(db_path)
    try:
        field_names = list(safe.keys())
        field_vals = list(safe.values())

        # Routed through the Dialect.upsert_on_conflict() seam (D13,
        # 2026-07-02 cleanup review) instead of a hand-built ON CONFLICT DO
        # UPDATE string — behaves identically on both SQLite (used by
        # tests/test_discovery_autofill_manager.py) and PG (production).
        # Columns: sector (conflict key) + only the passed fields + updated_at
        # (raw SQL expression -> DB clock, not a Python-side value).
        # update_cols defaults to "every inserted column not in conflict_cols"
        # = field_names + updated_at, matching "only the passed fields +
        # updated_at are refreshed on conflict; unmentioned columns retain
        # their stored values".
        columns = ("sector", *field_names, "updated_at")
        sql = b.dialect.upsert_on_conflict(
            "discovery_autofill",
            columns,
            conflict_cols=("sector",),
            raw_values={"updated_at": "datetime('now')"},
        )
        b.execute(sql, (sector, *field_vals))
    finally:
        b.close()


# ------------------------------------------------------------------ #
# Block selection
# ------------------------------------------------------------------ #

def next_block(
    canonical_tics: list,
    count: int,
    downloaded_set: set,
    done_set: set,
) -> dict:
    """Pick the next *count* not-done TICs in canonical order.

    Parameters
    ----------
    canonical_tics:
        Ordered list of int TIC ids (the processing order is the order
        of this list — never random).
    count:
        Maximum number of TICs to select.
    downloaded_set:
        Set of TIC ids whose TGLC FITS file has been downloaded and indexed.
    done_set:
        Set of TIC ids that already have a completed task in the DB.

    Returns
    -------
    dict with keys:
        selected          — list of up to *count* not-done TICs
        ready             — subset of selected that are downloaded
        missing_download  — subset of selected that are not yet downloaded
        remaining_undone  — total count of not-done TICs left (including selected)
    """
    not_done = [t for t in canonical_tics if t not in done_set]
    selected = not_done[:count]
    ready = [t for t in selected if t in downloaded_set]
    missing_download = [t for t in selected if t not in downloaded_set]
    return {
        "selected": selected,
        "ready": ready,
        "missing_download": missing_download,
        "remaining_undone": len(not_done),
    }
