"""Bulk-ingested local index answering the SPOC-gate's two per-TIC MAST
questions offline, from ~2 static files per TESS sector instead of a MAST
round-trip per TIC — see
``docs/superpowers/specs/2026-08-03-spoc-gate-prefetch-offline-design.md``.

The two questions (unchanged semantics, just a different data source):

1. Does this TIC have a 2-min SPOC light curve, in ANY sector? — the exact
   question ``exonit.ffi_download.is_ffi_only_at_mast`` answers via
   ``author=SPOC, exptime=120``. A TIC "has SPOC" iff its TIC id appears
   anywhere in a ``tesscurl_sector_{N}_lc.sh`` bulk-download script (these
   list every ``*_lc.fits`` SPOC product for that sector).
2. Which multi-sector DV runs (``(start, end)`` sector ranges) does SPOC
   have published for this TIC? — the exact question
   ``exonit.mast_dv.fetch_dv_runs`` answers. The same ``(start, end)`` pairs
   are embedded in the filenames listed by ``tesscurl_sector_{N}_dv.sh``
   (parsed with ``exonit.mast_dv.parse_dv_filename`` — NOT re-implemented
   here, to keep exactly one place that knows that filename grammar). Full
   DV coverage requires TWO kinds of script, both ingested into the same
   ``spoc_dv_index`` table: the per-sector script above (verified against
   the real archive, 2026-08-03: 100% single-sector ``sN-sN`` rows, zero
   multi-sector) AND a SEPARATE ``tesscurl_multisector_s{START}-s{END}_dv.sh``
   per range, at the same directory listing, for any DV report spanning
   more than one sector — see ``ingest_multisector_dv``. Omitting the
   latter silently under-reports DV for every TIC whose only report is
   multi-sector.

Ingested by ``scripts/prefetch_spoc_bulk.py`` (one HTTP GET pair per
sector, plus a directory-listing scrape for multisector DV scripts);
consumed by ``exonit.server._run_spoc_check_background`` (the SPOC-gate)
via ``has_spoc_bulk`` / ``dv_runs_bulk`` below.

Backend selection mirrors ``exonit/metadata_cache.py``'s ``MetadataCache``
seam exactly: ``EXONIT_METADATA_DB_URL`` set to a ``postgresql://`` DSN ->
PostgreSQL (the same ``exonit_meta`` database the TGLC index lives in, per
the spec's §1); unset (the default) -> SQLite at *db_path* — this is what
lets ``tests/test_spoc_bulk.py`` exercise the full ingest/query surface with
zero network and zero real PostgreSQL, and what lets a local/dev server run
the offline gate without provisioning PG at all. The spec (§1) only asked
for a PG table, reasoning "the gate lives on the server, which already has
PG" — the SQLite fallback here is a deliberate widening of that, requested
by the implementation briefing for testability parity with the TGLC index
pattern (``exonit/metadata_index.py``); see the 2026-08-03 implementation
handoff for the full rationale.

Unlike ``tglc`` (tens of millions of rows, PARTITION BY LIST per sector —
see ``scripts/migration/schema_pg_meta.sql``), the three tables here top out
around ~2M rows total for the whole mission (§ "Dimensionament" of the
spec) — three orders of magnitude smaller, so no partitioning is needed and
the schema is self-provisioned identically on both backends (unlike
``MetadataCache.__init__``, which only self-provisions on SQLite and treats
PG schema as out-of-band-provisioned — that convention exists there because
of the partitioning decision, which does not apply here).
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Dict, Iterable, List, Optional, Tuple

from .mast_dv import parse_dv_filename
from .metadata_backend import is_pg_url, make_metadata_backend

logger = logging.getLogger("exonit.spoc_bulk")

DEFAULT_DB_PATH = "data/spoc_bulk_index.db"

# --------------------------------------------------------------------------- #
# Schema — portable across SQLite and PostgreSQL (no partitioning needed,
# see module docstring). Deviates from the spec's literal §1 DDL in two
# ways, both documented in the implementation handoff:
#   - spoc_bulk_meta's primary key is (sector, kind), not sector alone — the
#     spec's single-column PK would let ingesting a sector's DV script
#     silently clobber that same sector's LC coverage row (or vice versa),
#     which breaks bulk_coverage()'s "sectors ingested per kind" contract.
#   - fetched_at is TEXT (an ISO datetime string via each dialect's now()),
#     not TIMESTAMPTZ — for portability to SQLite, which has no such type;
#     mirrors metadata_cache.py's own `updated_at TEXT DEFAULT
#     (datetime('now'))` convention.
# --------------------------------------------------------------------------- #

_CREATE_LC_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS spoc_lc_index (
    tic    BIGINT NOT NULL,
    sector INTEGER NOT NULL,
    PRIMARY KEY (tic, sector)
)
"""
_CREATE_LC_TIC_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS ix_spoc_lc_index_tic ON spoc_lc_index(tic)"
)

_CREATE_DV_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS spoc_dv_index (
    tic          BIGINT NOT NULL,
    sector_start INTEGER NOT NULL,
    sector_end   INTEGER NOT NULL,
    PRIMARY KEY (tic, sector_start, sector_end)
)
"""
_CREATE_DV_TIC_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS ix_spoc_dv_index_tic ON spoc_dv_index(tic)"
)

_CREATE_META_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS spoc_bulk_meta (
    sector     INTEGER NOT NULL,
    kind       TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    n_rows     INTEGER NOT NULL,
    PRIMARY KEY (sector, kind)
)
"""

_SCHEMA_STATEMENTS = (
    _CREATE_LC_TABLE_SQL, _CREATE_LC_TIC_INDEX_SQL,
    _CREATE_DV_TABLE_SQL, _CREATE_DV_TIC_INDEX_SQL,
    _CREATE_META_TABLE_SQL,
)

# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #

# LC bulk script lines contain the product filename either as the curl `-o`/
# `--output` argument or embedded in the download URL's query string (or
# both) — deliberately a substring search, not a full-line match, so it is
# indifferent to quoting style and to which of those forms a given sector's
# script uses. Tolerant of the timestamp prefix (`tess<14 digits>`) by not
# anchoring to it at all: only the `-s{sector}-{tic}-{scid}-{c}_lc.fits`
# suffix (present in every SPOC 2-min LC filename since sector 1) is
# matched. Mirrors exonit/metadata_index.py's `_SPOC_RE` (same suffix, same
# non-anchored `.search()` use) rather than duplicating a fresh pattern.
_LC_FILENAME_RE = re.compile(r"-s(\d{4})-(\d{16})-\d+-[a-zA-Z]_lc\.fits")

# DV bulk script lines are parsed in two steps: first pull out just the
# `..._dvr.xml` filename token (curl lines otherwise carry a full URL around
# it, and mast_dv.parse_dv_filename's regex is anchored with `$` so it needs
# the token in isolation), then hand that token to parse_dv_filename for the
# (start, end) pair — see the module docstring: that parsing logic is NOT
# duplicated here. The TIC itself is not something parse_dv_filename exposes
# (it only returns the sector range), so a second, narrower regex extracts
# it from the same token — this does not re-implement the (start, end)
# parsing, only adds the one extra capture group parse_dv_filename doesn't
# need for its own callers.
_DVR_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+_dvr\.xml")
_DV_TIC_RE = re.compile(r"-s\d{4}-s\d{4}-(\d{16})-\d+_dvr\.xml$")


def _parse_lc_lines(text: str) -> Iterable[Tuple[int, int]]:
    """Yield ``(tic, sector)`` pairs parsed from a ``tesscurl_sector_N_lc.sh``
    script's content. Blank lines and comments (``#...``) are skipped; any
    other line without a matching filename is silently ignored (tolerant of
    the script's shebang/header lines)."""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LC_FILENAME_RE.search(line)
        if not m:
            continue
        sector = int(m.group(1))
        tic = int(m.group(2))
        yield tic, sector


def _parse_dv_lines(text: str) -> Iterable[Tuple[int, int, int]]:
    """Yield ``(tic, sector_start, sector_end)`` triples parsed from a
    ``tesscurl_sector_N_dv.sh`` script's content. Only ``_dvr.xml`` (the DV
    report) lines are matched — ``_dvt.xml`` (time series) and other DV
    products carry the same sector range but would double-count it, and
    ``parse_dv_filename`` already ignores them by design."""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token_match = _DVR_TOKEN_RE.search(line)
        if not token_match:
            continue
        token = token_match.group(0)
        parsed = parse_dv_filename(token)
        tic_match = _DV_TIC_RE.search(token)
        if parsed is None or tic_match is None:
            continue
        start, end = parsed
        yield int(tic_match.group(1)), start, end


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #

class SpocBulkIndex:
    """Thread-safe local index over ``spoc_lc_index`` / ``spoc_dv_index`` /
    ``spoc_bulk_meta`` — see the module docstring for backend selection."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        _env_url = os.environ.get("EXONIT_METADATA_DB_URL")
        target = _env_url if is_pg_url(_env_url) else db_path
        self._backend = make_metadata_backend(target)
        self._is_pg = self._backend.dialect.name == "postgres"
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        if self._is_pg:
            for stmt in _SCHEMA_STATEMENTS:
                self._backend.execute(stmt)
            return
        with self._backend.raw_connection() as conn:
            for stmt in _SCHEMA_STATEMENTS:
                conn.execute(stmt)
            conn.commit()

    # ------------------------------------------------------------------
    # Write — per-sector transactional DELETE + INSERT (§2 of the spec:
    # "Re-ingesting a sector is a transactional DELETE + COPY").
    # ------------------------------------------------------------------

    def replace_lc_sector(self, sector: int, rows: Iterable[Tuple[int, int]]) -> int:
        """Atomically replace every ``spoc_lc_index`` row for *sector*.

        Deduplicated on ``(tic, sector)`` before insert — a bulk script
        listing the same product twice (observed historically for a handful
        of re-delivered files) must not raise a PK violation. Returns the
        number of distinct rows written.
        """
        sector = int(sector)
        distinct = sorted({(int(tic), int(sec)) for tic, sec in rows})
        with self._backend.transaction() as tx:
            tx.execute("DELETE FROM spoc_lc_index WHERE sector = ?", (sector,))
            if distinct:
                tx.executemany(
                    "INSERT INTO spoc_lc_index (tic, sector) VALUES (?, ?)",
                    distinct)
        return len(distinct)

    def replace_dv_multisector(self, range_start: int, range_end: int,
                                rows: Iterable[Tuple[int, int, int]]) -> int:
        """Atomically replace ``spoc_dv_index`` rows belonging to the
        ``tesscurl_multisector_s{range_start}-s{range_end}_dv.sh`` script.

        Verified against the real archive (2026-08-03): multi-sector DV runs
        are NOT interleaved into any per-sector ``tesscurl_sector_{N}_dv.sh``
        (those are 100% ``sN-sN`` rows) — MAST instead publishes a SEPARATE
        script per ``(start, end)`` range at the same directory listing
        (observed e.g. ``s0001-s0002`` through ``s0001-s0069``; ranges do
        not all start at ``s0001``). Every row in that script carries the
        SAME ``(start, end)`` pair as the script's own name (generalizing
        ``replace_dv_sector``'s "final sector" assumption, now confirmed for
        the single-sector case, to the multisector case) — a differing row
        means that assumption does not hold for this script and must be
        surfaced loudly rather than filed under the wrong range.

        Symmetric-scoping check (Round 3 review, see ``replace_dv_sector``'s
        docstring for the mirror-image bug this ingest path must NOT have):
        the DELETE below is already scoped to the exact
        ``(sector_start, sector_end)`` pair, never a bare ``sector_end``, so
        it cannot touch a per-sector row (which always has
        ``sector_start == sector_end``) unless this range were itself
        degenerate (``range_start == range_end`` — not a real multisector
        range; a genuine multisector script name always has ``START !=
        END``). No further scoping needed here.
        """
        range_start, range_end = int(range_start), int(range_end)
        distinct = sorted({(int(tic), int(s), int(e)) for tic, s, e in rows})
        for tic, s, e in distinct:
            if (s, e) != (range_start, range_end):
                raise ValueError(
                    f"DV row (tic={tic}, start={s}, end={e}) does not match "
                    f"multisector script range s{range_start:04d}-"
                    f"s{range_end:04d} — the 'a multisector script only "
                    "contains rows for its own range' assumption does not "
                    "hold for this script; investigate before trusting this "
                    "ingest.")
        with self._backend.transaction() as tx:
            tx.execute(
                "DELETE FROM spoc_dv_index "
                "WHERE sector_start = ? AND sector_end = ?",
                (range_start, range_end))
            if distinct:
                tx.executemany(
                    "INSERT INTO spoc_dv_index "
                    "(tic, sector_start, sector_end) VALUES (?, ?, ?)",
                    distinct)
        return len(distinct)

    def replace_dv_sector(self, sector: int, rows: Iterable[Tuple[int, int, int]]) -> int:
        """Atomically replace ``spoc_dv_index`` rows belonging to *sector*'s
        PER-SECTOR ``tesscurl_sector_{sector}_dv.sh`` script.

        A per-sector script's rows live under their FINAL sector (§
        "Key observation" of the spec: multi-sector DV reports are
        delivered with the last sector's bulk script) — verified against
        the real archive (2026-08-03): every row parsed from a per-sector
        script is single-sector, ``sector_start == sector_end == sector``.
        So "belonging to *sector*" for THIS ingest path means
        ``sector_end = sector AND sector_start = sector_end`` — the
        second, redundant-looking clause is load-bearing: without it the
        DELETE would also remove any MULTISECTOR row
        (``replace_dv_multisector``) that happens to share the same
        ``sector_end`` (e.g. a ``s0011-s0014`` multisector row has
        ``sector_end = 14``, same as any per-sector ``s0014-s0014`` row) —
        both ingest paths share this one ``spoc_dv_index`` table, and a
        first-time per-sector-14 ingest arriving AFTER an earlier
        multisector ingest ending at sector 14 would otherwise silently
        wipe that multisector row while ``spoc_bulk_meta``/``coverage()``
        keeps reporting it as ingested. See
        ``tests.test_spoc_bulk.TestDvRunsBulkUnion`` for the regression
        test covering this exact cross-kind ordering.
        """
        sector = int(sector)
        distinct = sorted({(int(tic), int(s), int(e)) for tic, s, e in rows})
        for tic, s, e in distinct:
            if e != sector:
                raise ValueError(
                    f"DV row (tic={tic}, start={s}, end={e}) does not belong "
                    f"to sector {sector} (expected sector_end == {sector}) — "
                    "the 'DV rows live under their final sector' assumption "
                    "(spec 'Key observation') does not hold for this script; "
                    "investigate before trusting this ingest.")
        with self._backend.transaction() as tx:
            tx.execute(
                "DELETE FROM spoc_dv_index "
                "WHERE sector_end = ? AND sector_start = sector_end",
                (sector,))
            if distinct:
                tx.executemany(
                    "INSERT INTO spoc_dv_index "
                    "(tic, sector_start, sector_end) VALUES (?, ?, ?)",
                    distinct)
        return len(distinct)

    def record_meta(self, sector: int, kind: str, n_rows: int) -> None:
        """Upsert ``spoc_bulk_meta`` for ``(sector, kind)`` — called AFTER
        the corresponding ``replace_*_sector`` succeeds, so a sector never
        appears "covered" in ``coverage()`` while its rows are only
        partially written (see ``ingest_sector``)."""
        d = self._backend.dialect
        sql = d.upsert(
            "spoc_bulk_meta", ["sector", "kind", "fetched_at", "n_rows"],
            ["sector", "kind"], raw_values={"fetched_at": d.now()})
        self._backend.execute(sql, (int(sector), kind, int(n_rows)))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def has_lc_coverage(self) -> bool:
        """True iff at least one sector's LC script has ever been ingested."""
        n = self._backend.query_scalar(
            "SELECT COUNT(*) FROM spoc_bulk_meta WHERE kind = 'lc'")
        return bool(n)

    def has_dv_coverage(self) -> bool:
        """True iff at least one DV script has ever been ingested — either a
        per-sector ``tesscurl_sector_{N}_dv.sh`` (``kind='dv'``) or a
        multisector ``tesscurl_multisector_s{START}-s{END}_dv.sh``
        (``kind`` prefixed ``dv_multi_``, see ``ingest_multisector_dv``).
        Both kinds insert into the same ``spoc_dv_index`` table, so either
        one alone is enough for ``dv_runs_bulk`` to stop returning ``None``
        (never-ingested) and start returning a real (possibly empty) list."""
        n = self._backend.query_scalar(
            "SELECT COUNT(*) FROM spoc_bulk_meta "
            "WHERE kind = 'dv' OR kind LIKE 'dv_multi_%'")
        return bool(n)

    def has_spoc(self, tic: int) -> Optional[bool]:
        """True/False if the LC bulk index has ever been ingested (at least
        one sector), else ``None`` — see ``has_spoc_bulk``'s docstring for
        the full three-way contract this feeds into the SPOC-gate."""
        if not self.has_lc_coverage():
            return None
        n = self._backend.query_scalar(
            "SELECT COUNT(*) FROM spoc_lc_index WHERE tic = ?", (int(tic),))
        return bool(n)

    def dv_runs(self, tic: int) -> Optional[List[Tuple[int, int]]]:
        """DV ``(start, end)`` runs for *tic* if the DV bulk index has ever
        been ingested (possibly ``[]`` — queried, genuinely no DV), else
        ``None`` (index never ingested — distinct from "queried, no DV")."""
        if not self.has_dv_coverage():
            return None
        rows = self._backend.query(
            "SELECT sector_start, sector_end FROM spoc_dv_index WHERE tic = ?",
            (int(tic),))
        return sorted(
            (int(r["sector_start"]), int(r["sector_end"])) for r in rows)

    def coverage(self) -> Dict[str, List]:
        """``{"lc": [sectors...], "dv": [sectors...], "dv_multi":
        ["sSTART-sEND", ...]}`` — ingested coverage per kind, ordered
        ascending. Empty list (not missing key) for a kind never ingested.
        Multisector DV ranges are reported under the separate ``"dv_multi"``
        key as human-readable ``"sSTART-sEND"`` strings (sorted
        lexicographically, which is numerically correct given the 4-digit
        zero-padding) rather than folded into ``"dv"`` — a different script,
        at a different (range, not single-sector) granularity."""
        rows = self._backend.query(
            "SELECT sector, kind FROM spoc_bulk_meta ORDER BY sector")
        out: Dict[str, List] = {"lc": [], "dv": [], "dv_multi": []}
        for r in rows:
            kind = r["kind"]
            if kind.startswith("dv_multi_"):
                out["dv_multi"].append(kind[len("dv_multi_"):])
            else:
                out.setdefault(kind, []).append(int(r["sector"]))
        out["dv_multi"].sort()
        return out

    def close(self) -> None:
        self._backend.close()


# --------------------------------------------------------------------------- #
# Process-wide singleton (mirrors exonit/metadata_index.py's
# _shared_tglc_caches — avoids paying MetadataCache-style construction cost
# (schema self-provisioning queries) on every SPOC-gate call).
# --------------------------------------------------------------------------- #

_shared_indexes: Dict[str, SpocBulkIndex] = {}
_shared_indexes_lock = threading.Lock()


def _get_index(db_path: Optional[str] = None) -> SpocBulkIndex:
    """Return the shared ``SpocBulkIndex`` for *db_path*.

    ``db_path`` defaults to ``None`` (resolved to the CURRENT value of
    module-level ``DEFAULT_DB_PATH`` here, at call time) rather than binding
    ``= DEFAULT_DB_PATH`` in the signature — a plain default argument is
    evaluated once, at function-definition time, so
    ``tests/test_spoc_bulk.py`` monkeypatching ``exonit.spoc_bulk.
    DEFAULT_DB_PATH`` to a ``tmp_path`` file (the only way to redirect
    ``exonit.server._run_spoc_check_background``'s calls, which never pass
    ``db_path`` themselves) would silently have no effect with the naive
    form. Mirrors ``exonit.mast_dv.MastDVCache.__init__``'s identical
    ``db_path or <computed default>`` convention.
    """
    resolved = db_path if db_path is not None else DEFAULT_DB_PATH
    idx = _shared_indexes.get(resolved)
    if idx is not None:
        return idx
    with _shared_indexes_lock:
        idx = _shared_indexes.get(resolved)
        if idx is None:
            idx = SpocBulkIndex(resolved)
            _shared_indexes[resolved] = idx
    return idx


def reset_shared_indexes_for_tests() -> None:
    """Drop every cached ``SpocBulkIndex`` instance. Test-only: without this
    a SQLite ``tmp_path`` fixture from one test could leak into the next via
    the process-wide singleton keyed by *db_path* string."""
    with _shared_indexes_lock:
        for idx in _shared_indexes.values():
            idx.close()
        _shared_indexes.clear()


def _read_target_never_ingested(db_path: Optional[str]) -> bool:
    """True iff a READ against *db_path* can be answered "empty index" (None)
    WITHOUT constructing a backend at all.

    ``sqlite3.connect(path, ...)`` creates an empty ``.db`` file on disk the
    instant it is called, even for a pure read — so every caller that only
    ever READS the bulk index (every existing SPOC-gate test in
    ``tests/test_spoc_gate.py``, none of which mock ``exonit.spoc_bulk`` at
    all, since it did not exist before this feature) would otherwise leave a
    stray ``data/spoc_bulk_index.db`` behind on first run. Mirrors
    ``exonit/metadata_index.py::_tglc_lookup``'s identical file-existence
    guard (and its docstring's reasoning: "only makes sense for a literal
    SQLite path... skip it when EXONIT_METADATA_DB_URL is a PG DSN").
    Never short-circuits the write path (``ingest_sector``/``SpocBulkIndex``
    construction) — only the read-only public functions below use this.
    """
    resolved = db_path if db_path is not None else DEFAULT_DB_PATH
    if is_pg_url(os.environ.get("EXONIT_METADATA_DB_URL")):
        return False  # PG is authoritative regardless of `resolved`
    if is_pg_url(resolved):
        return False
    return not os.path.exists(resolved)


# --------------------------------------------------------------------------- #
# Public API (exonit.server._run_spoc_check_background and
# scripts/prefetch_spoc_bulk.py both go through this, never SpocBulkIndex
# directly, so the singleton/caching behaviour above is not something every
# call site needs to know about).
# --------------------------------------------------------------------------- #

def ingest_sector(sector: int, kind: str, text: str,
                   db_path: Optional[str] = None) -> int:
    """Parse *text* (the raw content of an already-downloaded
    ``tesscurl_sector_{sector}_{kind}.sh``) and (re-)ingest *sector* for
    *kind* (``"lc"`` or ``"dv"``) into the local bulk index.

    Idempotent: safe to call again for the same sector — the previous
    ingest's rows are fully replaced (DELETE + INSERT), never merged.
    Returns the number of distinct rows written.
    """
    idx = _get_index(db_path)
    if kind == "lc":
        n = idx.replace_lc_sector(sector, _parse_lc_lines(text))
    elif kind == "dv":
        n = idx.replace_dv_sector(sector, _parse_dv_lines(text))
    else:
        raise ValueError(f"unknown kind {kind!r} (expected 'lc' or 'dv')")
    idx.record_meta(sector, kind, n)
    return n


def _multisector_kind(range_start: int, range_end: int) -> str:
    """``spoc_bulk_meta.kind`` for a multisector DV script — embeds the full
    range so the ``(sector, kind)`` PK cannot collide between two ranges
    that happen to share the same final sector (e.g. ``s0001-s0069`` and
    ``s0050-s0069``), which a bare ``sector``-only kind label would risk."""
    return f"dv_multi_s{int(range_start):04d}-s{int(range_end):04d}"


def ingest_multisector_dv(range_start: int, range_end: int, text: str,
                           db_path: Optional[str] = None) -> int:
    """Parse *text* (the raw content of an already-downloaded
    ``tesscurl_multisector_s{range_start}-s{range_end}_dv.sh``) and
    (re-)ingest it into the local bulk index.

    Multisector DV runs (a TIC observed across MORE than one sector) are
    NOT included in any per-sector ``tesscurl_sector_{N}_dv.sh`` — verified
    against the real archive (2026-08-03): ``tesscurl_sector_14_dv.sh`` is
    15,618 products, 100% ``s0014-s0014``, zero multi-sector rows. MAST
    instead publishes each ``(start, end)`` range as its OWN script at the
    same ``download_scripts/sector/`` directory listing (e.g.
    ``s0001-s0002`` through ``s0001-s0069`` observed for sector-1-anchored
    ranges; ranges do not all start at ``s0001``). Without ingesting these
    separately, ``dv_runs_bulk`` systematically under-reports DV coverage
    for every TIC whose only DV report is multi-sector — see
    ``scripts/prefetch_spoc_bulk.py``'s directory-listing discovery, which
    calls this for every ``tesscurl_multisector_s{A}-s{B}_dv.sh`` it finds.

    Rows land in the SAME ``spoc_dv_index`` table as ``ingest_sector``'s DV
    rows — ``dv_runs_bulk(tic)`` returns the union of both sources with no
    further caller-side logic needed. Coverage bookkeeping
    (``spoc_bulk_meta``) uses a distinct ``kind`` per range (see
    ``_multisector_kind``) so ``bulk_coverage()`` can report per-sector and
    multisector ingests separately (the ``"dv_multi"`` key).

    Idempotent: same DELETE+INSERT replace semantics as ``ingest_sector``.
    Returns the number of distinct rows written.
    """
    idx = _get_index(db_path)
    n = idx.replace_dv_multisector(range_start, range_end, _parse_dv_lines(text))
    idx.record_meta(range_end, _multisector_kind(range_start, range_end), n)
    return n


def has_spoc_bulk(tic, db_path: Optional[str] = None) -> Optional[bool]:
    """True/False if *tic* is present/absent in the ingested LC bulk index,
    ``None`` if the LC bulk index has never been ingested at all.

    Presence (``True``) is ALWAYS a reliable positive — the bulk script for
    the sector the TIC's LC actually lives in has been ingested, or the TIC
    would not be in the index. Absence (``False``) is only as reliable as
    the ingest's COMPLETENESS: it means "not found among the sectors
    ingested so far", which is only equivalent to "has no SPOC LC in any
    sector, ever" if every published sector has been ingested — a stronger
    claim this function itself cannot verify, which is why the caller
    (``exonit.server._run_spoc_check_background``) only trusts ``False`` as
    a ffi-only verdict when the operator has asserted full coverage via
    ``EXONIT_SPOC_GATE_OFFLINE=1`` (see that function's docstring for the
    full asymmetric decision table).
    """
    if _read_target_never_ingested(db_path):
        return None
    return _get_index(db_path).has_spoc(int(tic))


def dv_runs_bulk(tic, db_path: Optional[str] = None) -> Optional[List[Tuple[int, int]]]:
    """DV ``(start, end)`` runs for *tic* from the ingested DV bulk index,
    or ``None`` if the DV bulk index has never been ingested at all.

    An empty list (as opposed to ``None``) means "the DV bulk index IS
    ingested and this TIC has no DV report in it" — the same "queried,
    genuinely absent" signal ``exonit.mast_dv.fetch_dv_runs`` returns for a
    cache hit on a TIC with no DV, and is handled identically downstream
    (``db.mark_spoc_no_dv_candidates``).
    """
    if _read_target_never_ingested(db_path):
        return None
    return _get_index(db_path).dv_runs(int(tic))


def bulk_coverage(db_path: Optional[str] = None) -> Dict[str, List]:
    """``{"lc": [sectors...], "dv": [sectors...], "dv_multi":
    ["sSTART-sEND", ...]}`` currently ingested — see
    ``SpocBulkIndex.coverage``."""
    if _read_target_never_ingested(db_path):
        return {"lc": [], "dv": [], "dv_multi": []}
    return _get_index(db_path).coverage()
