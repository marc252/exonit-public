"""SPOC Data Validation (DV) XML discovery for TESS targets.

ExoMiner (and any TCE-level classifier) requires knowing which multi-sector
runs SPOC has published DV products for, since TCEs are defined per run.
MAST query responses are ~2-5s per TIC and are stable over time (DV products
are static once published), so results are cached locally in SQLite.

Primary API:
    cache = MastDVCache()
    runs = fetch_dv_runs(tic_id, cache=cache)         # list[(start, end)]
    best = pick_best_run(runs)                         # "start-end" str or None
    ok = has_dv_for_sector(tic_id, sector, cache=cache) # bool, SECTOR-aware

The cache distinguishes cache-miss (None) from queried-but-empty ([]),
so TICs that SPOC has never processed are cached as [] and not re-queried.
"""
import os
import re
import sqlite3
import threading
from typing import List, Optional, Tuple

try:
    from astroquery.mast import Observations  # type: ignore
except ImportError:  # pragma: no cover
    Observations = None  # type: ignore


_DV_FILENAME_RE = re.compile(r"-s(\d{4})-s(\d{4})-\d+-\d+_dvr\.xml$")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dv_runs (
    tic_id     TEXT PRIMARY KEY,
    runs_json  TEXT NOT NULL,
    updated_at TEXT DEFAULT (datetime('now'))
)
"""


def parse_dv_filename(filename: str) -> Optional[Tuple[int, int]]:
    """Return (start_sector, end_sector) from a SPOC DV report XML filename.

    Matches only ``_dvr.xml`` (the report) and ignores ``_dvt.xml``
    (time series), light curves and other products.
    """
    m = _DV_FILENAME_RE.search(filename)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def pick_best_run(runs: List[Tuple[int, int]]) -> Optional[str]:
    """Pick the best multi-sector run for ExoMiner inference.

    Strategy: prefer the longest multi-sector run (more sectors → more
    transits stacked). Fall back to the latest single-sector run.
    Returns a ``"start-end"`` string consumable by ExoMiner's
    ``sector_run`` column, or ``None`` if no runs are available.
    """
    if not runs:
        return None
    multi = [(s, e) for (s, e) in runs if s != e]
    if multi:
        s, e = max(multi, key=lambda r: r[1] - r[0])
        return f"{s}-{e}"
    single = [(s, e) for (s, e) in runs if s == e]
    if single:
        s, e = max(single, key=lambda r: r[0])
        return f"{s}-{e}"
    return None


def dv_runs_cover_sector(runs: List[Tuple[int, int]], sector: int) -> bool:
    """Pure predicate: does any ``(start, end)`` run in ``runs`` cover ``sector``?

    A DV report is published for a specific sector RANGE — a single-sector
    run has ``start == end``, a multi-sector run stacks several sectors
    (``start < end``). ExoMiner-SPOC can only score a TCE against a report
    that actually covers the sector the TCE (and, downstream, our candidate)
    was found in — "this TIC has DV somewhere" is NOT the same question as
    "this TIC has DV for THIS sector" (see ``has_dv_for_sector``, the
    I/O-aware wrapper around this check, for the bug this closes).

    No I/O — kept separate from ``has_dv_for_sector`` so callers that
    already hold a fetched ``runs`` list (e.g. the SPOC-gate, which fetches
    once per TIC then checks coverage once per candidate) don't pay a
    redundant cache lookup per candidate.
    """
    return any(start <= sector <= end for start, end in runs)


class MastDVCache:
    """SQLite-backed cache for MAST DV-run lookups.

    DV XML availability is static: once SPOC publishes a run it doesn't
    disappear, and new runs only appear when SPOC processes new sectors.
    Cache entries therefore don't need a TTL — they can be explicitly
    invalidated with ``put_runs`` or by bumping the stored row.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(
            os.path.dirname(__file__), "..", "data", "dv_runs_cache.db"
        )
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

    def get_runs(self, tic_id: str) -> Optional[List[Tuple[int, int]]]:
        """Return cached runs or None on cache miss.

        A return of ``[]`` means the TIC was queried and has no DV runs.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT runs_json FROM dv_runs WHERE tic_id = ?", (str(tic_id),)
            ).fetchone()
        if row is None:
            return None
        import json

        return [tuple(r) for r in json.loads(row[0])]

    def put_runs(self, tic_id: str, runs: List[Tuple[int, int]]) -> None:
        """Store runs (or ``[]`` for negative cache) — upserts."""
        import json

        payload = json.dumps([list(r) for r in runs])
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO dv_runs (tic_id, runs_json) VALUES (?, ?)",
                (str(tic_id), payload),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def fetch_dv_runs(
    tic_id: str,
    cache: Optional[MastDVCache] = None,
    force_refresh: bool = False,
) -> List[Tuple[int, int]]:
    """Fetch DV runs for ``tic_id`` from MAST, using cache if present.

    Returns a (possibly empty) list of ``(start, end)`` sector tuples.
    On cache hit returns without hitting MAST. On cache miss queries
    MAST and stores the result (including empty lists).
    """
    if cache is not None and not force_refresh:
        cached = cache.get_runs(str(tic_id))
        if cached is not None:
            return cached

    if Observations is None:
        raise ImportError("astroquery not installed — required for MAST queries")

    obs = Observations.query_criteria(
        target_name=str(tic_id),
        obs_collection="TESS",
        dataproduct_type="timeseries",
    )
    if len(obs) == 0:
        runs: List[Tuple[int, int]] = []
    else:
        products = Observations.get_product_list(obs)
        filenames = products["productFilename"]
        seen = set()
        for fn in filenames:
            parsed = parse_dv_filename(str(fn))
            if parsed is not None:
                seen.add(parsed)
        runs = sorted(seen)

    if cache is not None:
        cache.put_runs(str(tic_id), runs)
    return runs


def has_dv_for_sector(
    tic_id: str,
    sector: int,
    cache: Optional[MastDVCache] = None,
    force_refresh: bool = False,
) -> bool:
    """True iff SPOC published a DV report covering ``sector`` for ``tic_id``.

    Fix (docs/superpowers/specs/exominer-dv-gate-sector.md, 2026-07-14): the
    SPOC-gate previously asked "does this TIC have DV at ALL?" — but a DV
    report only covers the sector RANGE it was run over, and ExoMiner-SPOC
    can only score a candidate against a report covering the candidate's OWN
    sector. Measured against real worker logs after re-queuing 10,230
    candidates recovered under the "any DV" premise: 5,138 came back
    "no DV product" (the TIC had DV, just not for that candidate's sector)
    and 1,557 "preprocess failed" (e.g. "No LC data read for TIC ... sector_run
    12-12") — the overwhelming majority were never scoreable. This is the
    corrected, per-sector question.

    Delegates to ``fetch_dv_runs`` (so it shares the same on-disk cache and
    fail-soft-by-raising contract — callers decide what "unknown" means, this
    function never swallows an error) and ``dv_runs_cover_sector`` for the
    actual range check. Raises whatever ``fetch_dv_runs`` raises (network /
    astroquery errors) — callers that need fail-soft behaviour (e.g. the
    SPOC-gate) must catch around the call, exactly as they already do around
    ``fetch_dv_runs``.
    """
    runs = fetch_dv_runs(tic_id, cache=cache, force_refresh=force_refresh)
    return dv_runs_cover_sector(runs, sector)
