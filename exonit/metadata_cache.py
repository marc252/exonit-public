"""Local cache for MAST star metadata + the TGLC file index.

Avoids repeated MAST queries (slow, ~35% failure rate) by persisting TIC
metadata locally. Also indexes the TGLC (FFI) bulk-download file listing
(table ``tglc``) and packed-FITS locations (``fits_blobs``) — see
``docs/superpowers/specs/2026-07-20-tglc-index-to-pg.md``.

SQLite (default) or PostgreSQL, selected by ``exonit/metadata_backend.py``'s
seam — see ``MetadataCache.__init__``.
"""
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set

from .metadata_backend import is_pg_url, make_metadata_backend

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS metadata (
    tic_id      TEXT PRIMARY KEY,
    sectors     TEXT NOT NULL,
    exptime     REAL NOT NULL,
    fits_files  TEXT NOT NULL,
    stellar_radius REAL,
    teff        REAL,
    logg        REAL,
    tmag        REAL,
    ra          REAL,
    dec         REAL,
    updated_at  TEXT DEFAULT (datetime('now'))
)
"""

_CREATE_FITS_PATHS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS fits_paths (
    filename TEXT PRIMARY KEY,
    full_path TEXT NOT NULL
);
"""

# Packed-FITS location index: which tar container in object storage holds a
# given FITS, and the byte range of its data (for an HTTP Range read).
_CREATE_FITS_BLOBS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS fits_blobs (
    filename      TEXT PRIMARY KEY,
    container_key TEXT NOT NULL,
    data_offset   INTEGER NOT NULL,
    data_length   INTEGER NOT NULL
);
"""

_CREATE_NEARBY_STARS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS nearby_stars (
    tic_id      TEXT NOT NULL,
    neighbor_id TEXT NOT NULL,
    ra          REAL,
    dec         REAL,
    mass        REAL,
    rad         REAL,
    teff        REAL,
    tmag        REAL,
    plx         REAL,
    gal_b       REAL,
    updated_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (tic_id, neighbor_id)
);
"""

# The TGLC bulk-index table itself is NOT created here (never was, even
# pre-seam): only scripts/index_tglc_batch.py provisions it on SQLite, and on
# PostgreSQL it is provisioned out-of-band by
# scripts/migration/schema_pg_meta.sql. See MetadataCache.tglc_* helpers below.

# SQLite-only DDL for `tglc`, ported verbatim from scripts/index_tglc_batch.py's
# historical raw-connection CREATE TABLE/INDEX (F3, D3 of the spec: "SQL lives
# in ONE file"). Deliberately does NOT add ix_tglc_sector/ix_tglc_path here —
# the pre-seam script never created those either (they exist on the live
# SQLite file from an earlier one-time migration, not from this script), and
# reproducing that exactly avoids an unplanned index build against the live
# 61 GiB file. See MetadataCache.ensure_tglc_table.
_CREATE_TGLC_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tglc (
    tic_id TEXT, sector INTEGER, path TEXT, exptime REAL,
    tmag REAL, ra REAL, dec REAL, PRIMARY KEY(tic_id, sector))
"""
_CREATE_TGLC_TIC_INDEX_SQL = "CREATE INDEX IF NOT EXISTS ix_tglc_tic ON tglc(tic_id)"


class MetadataCache:
    """Thread-safe cache for MAST star metadata + the TGLC file index.

    Backend selection (D2/D3 of the 2026-07-20 TGLC->PG spec):
    ``EXONIT_METADATA_DB_URL`` unset (the default) -> SQLite at *db_path*,
    byte-for-byte the historical behaviour. Set to a ``postgresql://`` DSN ->
    every caller of this class transparently moves to PostgreSQL, regardless
    of what *db_path* string they happen to pass — this keeps the class's
    public constructor signature unchanged (required by D3: existing callers
    across exonit/enqueue.py, exonit/ctoi.py, exonit/setup_wizard.py,
    exonit/routes/candidates.py, exonit/server.py etc. do not need to change)
    while still making the single env var the one on/off switch for the
    whole process, exactly like ``EXONIT_DATABASE_URL`` does for
    ``exonit/db_backend.py``'s operational-DB seam. A DSN can also be passed
    directly as *db_path* (mirrors ``exonit/db_backend.make_backend``).
    """

    def __init__(self, db_path: str = "data/metadata_cache.db") -> None:
        _env_url = os.environ.get("EXONIT_METADATA_DB_URL")
        target = _env_url if is_pg_url(_env_url) else db_path
        self._backend = make_metadata_backend(target)
        self._is_pg = self._backend.dialect.name == "postgres"

        if self._is_pg:
            # Schema is provisioned out-of-band (scripts/migration/
            # schema_pg_meta.sql), mirroring ExoNitDB's PG path — no
            # CREATE TABLE / ALTER TABLE here (see that file's DEPLOY NOTE
            # convention, and exonit/db.py's PG branch for the precedent).
            return

        # Schema self-provisioning (SQLite only, unchanged from pre-seam
        # behaviour). Uses the backend's raw_connection() — construction-time
        # DDL, not a query the dialect needs to translate.
        import sqlite3
        with self._backend.raw_connection() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_FITS_PATHS_TABLE_SQL)
            conn.execute(_CREATE_FITS_BLOBS_TABLE_SQL)
            conn.execute(_CREATE_NEARBY_STARS_TABLE_SQL)
            # Migrate: add stellar param columns if missing
            for col, typ in [("stellar_radius", "REAL"), ("teff", "REAL"),
                             ("logg", "REAL"), ("tmag", "REAL"),
                             ("ra", "REAL"), ("dec", "REAL")]:
                try:
                    conn.execute(f"ALTER TABLE metadata ADD COLUMN {col} {typ}")
                except sqlite3.OperationalError:
                    pass
            conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_entry(
        self,
        tic_id: str,
        sectors: List[int],
        exptime: float,
        fits_files: List[str],
        stellar_radius: Optional[float] = None,
        teff: Optional[float] = None,
        logg: Optional[float] = None,
        tmag: Optional[float] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
    ) -> None:
        """Insert or replace a metadata entry."""
        dialect = self._backend.dialect
        columns = ["tic_id", "sectors", "exptime", "fits_files",
                   "stellar_radius", "teff", "logg", "tmag", "ra", "dec",
                   "updated_at"]
        sql = dialect.upsert("metadata", columns, ["tic_id"],
                             raw_values={"updated_at": dialect.now()})
        # NOT dialect.translate()'d here: backend.execute translates the SQL
        # itself (a pre-translated '?' -> '%s' would be double-escaped to
        # '%%s' by PgBackend's own translate — see db.py's identical note on
        # add_run_tics_bulk).
        self._backend.execute(sql, (
            tic_id, json.dumps(sectors), float(exptime), json.dumps(fits_files),
            stellar_radius, teff, logg, tmag, ra, dec,
        ))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_entry(self, tic_id: str) -> Optional[Dict[str, Any]]:
        """Return metadata dict for *tic_id*, or None if not cached."""
        row = self._backend.query_one(
            "SELECT * FROM metadata WHERE tic_id = ?",
            (tic_id,),
        )
        if row is None:
            return None
        return {
            "tic_id": row["tic_id"],
            "sectors": json.loads(row["sectors"]),
            "exptime": row["exptime"],
            "fits_files": json.loads(row["fits_files"]),
            "stellar_radius": row["stellar_radius"],
            "teff": row["teff"],
            "logg": row["logg"],
            "tmag": row["tmag"],
            "ra": row["ra"],
            "dec": row["dec"],
        }

    def cached_tic_ids(self, tic_ids: Iterable[str], chunk_size: int = 5000) -> Set[str]:
        """Return the subset of *tic_ids* present in the cache.

        Batched equivalent of ``{t for t in tic_ids if get_entry(t) is not
        None}``, but does the membership check with chunked ``SELECT tic_id
        ... WHERE tic_id IN (...)`` queries instead of one locked ``SELECT *``
        + two ``json.loads`` per id. Needed by callers that check membership
        for tens/hundreds of thousands of ids in one go (e.g. /api/health/
        work's FPP-backfill count, which used to do exactly that per-id and
        took 18-30 s once the queue grew past ~300k rows — see
        progress/impl-healthwork-perf.md).

        chunk_size caps ids per statement well under SQLite's per-statement
        bound-parameter limit (999 on older builds, 32766 on builds since
        3.32.0) and PostgreSQL's 65535-parameter limit.
        """
        ids = list({str(t) for t in tic_ids})
        found: Set[str] = set()
        for i in range(0, len(ids), chunk_size):
            batch = ids[i:i + chunk_size]
            placeholders = ",".join("?" * len(batch))
            rows = self._backend.query(
                f"SELECT tic_id FROM metadata WHERE tic_id IN ({placeholders})",
                tuple(batch),
            )
            found.update(r["tic_id"] for r in rows)
        return found

    def get_exptime_map(self, tic_ids: Iterable[str],
                        chunk_size: int = 5000) -> Dict[str, float]:
        """Return {tic_id: exptime} for the subset of *tic_ids* present in the
        cache, in chunked ``SELECT tic_id, exptime ... WHERE tic_id IN (...)``
        queries — same batching rationale as ``cached_tic_ids`` (avoids one
        round trip per id for bulk callers, e.g.
        ``scripts/recompute_verdict.py``'s FFI/SPOC routing, which needs the
        exptime for every TIC in a recompute batch up front rather than one
        ``get_entry`` call per row). TICs absent from the cache are absent
        from the returned dict (caller treats "not found" as the
        ``data_source_from_exptime(None)`` conservative default, same as
        ``resolve_data_source``).
        """
        ids = list({str(t) for t in tic_ids})
        out: Dict[str, float] = {}
        for i in range(0, len(ids), chunk_size):
            batch = ids[i:i + chunk_size]
            placeholders = ",".join("?" * len(batch))
            rows = self._backend.query(
                f"SELECT tic_id, exptime FROM metadata WHERE tic_id IN ({placeholders})",
                tuple(batch),
            )
            for r in rows:
                out[r["tic_id"]] = r["exptime"]
        return out

    def get_entries_bulk(self, tic_ids: Iterable[str],
                         chunk_size: int = 5000) -> Dict[str, Dict[str, Any]]:
        """Bulk equivalent of ``get_entry``: ``{tic_id: entry_dict}`` for the
        subset of *tic_ids* that already have a ``metadata`` row.

        Same chunked ``SELECT ... WHERE tic_id IN (...)`` pattern as
        ``cached_tic_ids``/``get_exptime_map`` (5000/chunk — comfortably
        under SQLite's ~32766 and PostgreSQL's 65535 per-statement
        bound-parameter limits). Used by
        ``exonit.metadata_index.ensure_metadata_bulk`` to decide
        create/refresh/preserve/unchanged for a whole chunk in ONE round trip
        instead of one ``get_entry`` per TIC — see
        docs/superpowers/specs/2026-09-26-metadata-refresh-and-bulk-enqueue-design.md
        §2.2 step 1.
        """
        ids = list({str(t) for t in tic_ids})
        out: Dict[str, Dict[str, Any]] = {}
        for i in range(0, len(ids), chunk_size):
            batch = ids[i:i + chunk_size]
            placeholders = ",".join("?" * len(batch))
            rows = self._backend.query(
                f"SELECT tic_id, sectors, exptime, fits_files, "
                f"stellar_radius, teff, logg, tmag, ra, dec "
                f"FROM metadata WHERE tic_id IN ({placeholders})",
                tuple(batch),
            )
            for row in rows:
                out[row["tic_id"]] = {
                    "tic_id": row["tic_id"],
                    "sectors": json.loads(row["sectors"]),
                    "exptime": row["exptime"],
                    "fits_files": json.loads(row["fits_files"]),
                    "stellar_radius": row["stellar_radius"],
                    "teff": row["teff"],
                    "logg": row["logg"],
                    "tmag": row["tmag"],
                    "ra": row["ra"],
                    "dec": row["dec"],
                }
        return out

    def tglc_lookup_bulk(self, tic_ids: Iterable[str],
                         chunk_size: int = 5000) -> Dict[str, List[Dict[str, Any]]]:
        """Bulk equivalent of ``tglc_lookup``: ``{tic_id: [rows]}`` for every
        TIC in *tic_ids* that has ``tglc`` rows.

        ``ORDER BY tic_id, sector, path`` (unlike the single-TIC
        ``tglc_lookup``, which relies on whatever order the query planner
        picks — see that method's docstring and
        ``exonit.metadata_index._tglc_lookup``): two calls over the same
        ``tglc`` content must produce byte-identical ``fits_files`` ordering
        in ``ensure_metadata_bulk``, which is the basis of that function's
        idempotence guarantee (docs/superpowers/specs/
        2026-09-26-metadata-refresh-and-bulk-enqueue-design.md §2.2, "with
        ORDER BY in step 2, fits_files ends up ordered"). All rows for a given
        ``tic_id`` land in the same chunk (chunking is over ``tic_ids``, not
        rows), so this ordering is exact regardless of chunk boundaries.

        Empty (not an error) if ``tglc`` is missing entirely — mirrors
        ``tglc_lookup``'s missing-table contract.
        """
        ids = list({str(t) for t in tic_ids})
        out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for i in range(0, len(ids), chunk_size):
            batch = ids[i:i + chunk_size]
            placeholders = ",".join("?" * len(batch))
            try:
                rows = self._backend.query(
                    f"SELECT tic_id, sector, path, exptime, tmag, ra, dec "
                    f"FROM tglc WHERE tic_id IN ({placeholders}) "
                    f"ORDER BY tic_id, sector, path",
                    tuple(batch),
                )
            except self._missing_table_excs():
                continue
            for row in rows:
                out[row["tic_id"]].append(row)
        return dict(out)

    def write_metadata_bulk(self, metadata_rows, fits_path_rows) -> None:
        """Upsert *metadata_rows* + *fits_path_rows* in ONE transaction.

        metadata_rows: list of tuples matching ``add_entry``'s column order
        (``tic_id, sectors_json, exptime, fits_files_json, stellar_radius,
        teff, logg, tmag, ra, dec`` — ``sectors``/``fits_files`` already
        ``json.dumps()``'d by the caller).
        fits_path_rows: list of ``(filename, full_path)`` tuples,
        ``add_fits_paths_bulk``'s shape.

        Used by ``exonit.metadata_index.ensure_metadata_bulk`` for its
        per-chunk write: a single ``Backend.transaction()`` around both
        ``executemany`` calls, so a chunk's ``metadata`` + ``fits_paths``
        rows commit atomically (all-or-nothing per chunk) instead of
        ``add_entry``'s one-commit-per-row — see docs/superpowers/specs/
        2026-09-26-metadata-refresh-and-bulk-enqueue-design.md §2.2 step 4.
        """
        if not metadata_rows and not fits_path_rows:
            return
        d = self._backend.dialect
        meta_cols = ["tic_id", "sectors", "exptime", "fits_files",
                     "stellar_radius", "teff", "logg", "tmag", "ra", "dec",
                     "updated_at"]
        meta_sql = d.upsert("metadata", meta_cols, ["tic_id"],
                            raw_values={"updated_at": d.now()})
        fits_sql = d.upsert("fits_paths", ["filename", "full_path"],
                            ["filename"])
        with self._backend.transaction() as tx:
            if metadata_rows:
                tx.executemany(meta_sql, metadata_rows)
            if fits_path_rows:
                tx.executemany(fits_sql, fits_path_rows)

    def get_all_tic_ids(self) -> List[str]:
        """Return list of all cached TIC IDs."""
        rows = self._backend.query("SELECT tic_id FROM metadata")
        return [r["tic_id"] for r in rows]

    def all_tic_ids(self) -> set:
        """Return the set of all cached TIC IDs.

        Used by enqueue_targets to build a pre-loaded set for O(1) per-target
        lookup instead of constructing a new MetadataCache per target.

        Q3 (perf review 2026-09-26): streams via ``Backend.iter_id_column``
        (tuple rows, server-side cursor on PG) instead of ``query()`` — same
        rationale as ``tglc_tic_ids`` above, for the ``metadata`` table.
        """
        return set(self._backend.iter_id_column("SELECT tic_id FROM metadata"))

    def stats(self) -> Dict[str, int]:
        """Return aggregate counts: total_stars and total_fits_files."""
        rows = self._backend.query("SELECT fits_files FROM metadata")
        total_stars = len(rows)
        total_fits = sum(len(json.loads(r["fits_files"])) for r in rows)
        return {"total_stars": total_stars, "total_fits_files": total_fits}

    # ------------------------------------------------------------------
    # FITS path index
    # ------------------------------------------------------------------

    def add_fits_path(self, filename: str, full_path: str) -> None:
        """Index a FITS file path for fast lookup."""
        d = self._backend.dialect
        sql = d.upsert("fits_paths", ["filename", "full_path"], ["filename"])
        self._backend.execute(sql, (filename, full_path))

    def add_fits_paths_bulk(self, entries) -> None:
        """Bulk index FITS paths. entries: list of (filename, full_path) tuples."""
        d = self._backend.dialect
        sql = d.upsert("fits_paths", ["filename", "full_path"], ["filename"])
        self._backend.executemany(sql, entries)

    def get_fits_path(self, filename: str) -> Optional[str]:
        """Get full path for a FITS filename. Returns str or None."""
        row = self._backend.query_one(
            "SELECT full_path FROM fits_paths WHERE filename = ?",
            (filename,),
        )
        return row["full_path"] if row is not None else None

    # ------------------------------------------------------------------
    # Packed FITS index (object-storage containers)
    # ------------------------------------------------------------------

    def add_fits_blobs_bulk(self, entries) -> None:
        """Bulk-index packed FITS locations.

        entries: list of (filename, container_key, data_offset, data_length).
        """
        d = self._backend.dialect
        sql = d.upsert("fits_blobs",
                       ["filename", "container_key", "data_offset", "data_length"],
                       ["filename"])
        self._backend.executemany(sql, entries)

    def get_fits_blob(self, filename: str):
        """Return (container_key, data_offset, data_length) for a packed FITS,
        or None if it is not in a container."""
        row = self._backend.query_one(
            "SELECT container_key, data_offset, data_length "
            "FROM fits_blobs WHERE filename = ?",
            (filename,),
        )
        if row is None:
            return None
        return (row["container_key"], row["data_offset"], row["data_length"])

    # ------------------------------------------------------------------
    # Nearby stars cache (for FPP)
    # ------------------------------------------------------------------

    def add_nearby_stars(self, tic_id: str, stars: List[Dict], gal_b: float) -> None:
        """Cache nearby stars for a target. stars: list of dicts with
        ID, ra, dec, mass, rad, Teff, Tmag, plx."""
        d = self._backend.dialect
        insert_sql = d.upsert(
            "nearby_stars",
            ["tic_id", "neighbor_id", "ra", "dec", "mass", "rad", "teff",
             "tmag", "plx", "gal_b"],
            ["tic_id", "neighbor_id"],
        )
        delete_sql = "DELETE FROM nearby_stars WHERE tic_id = ?"
        # Neither SQL is dialect.translate()'d here: tx.execute() (both
        # _SqliteTx and _PgTx) translates internally — see add_entry's note.
        # DELETE + per-star INSERT stay one atomic unit (matches the previous
        # single-lock/single-commit behaviour on SQLite; PG gets a real
        # transaction via backend.transaction()).
        with self._backend.transaction() as tx:
            tx.execute(delete_sql, (tic_id,))
            for s in stars:
                tx.execute(insert_sql, (
                    tic_id, str(s['ID']), s.get('ra'), s.get('dec'),
                    s.get('mass'), s.get('rad'), s.get('Teff'),
                    s.get('Tmag'), s.get('plx'), gal_b))

    def get_nearby_stars(self, tic_id: str) -> Optional[tuple]:
        """Return (stars_list, gal_b) from cache, or None if not cached."""
        rows = self._backend.query(
            "SELECT * FROM nearby_stars WHERE tic_id = ?",
            (tic_id,),
        )
        if not rows:
            return None
        stars = []
        gal_b = None
        for r in rows:
            stars.append({
                'ID': r['neighbor_id'],
                'ra': r['ra'], 'dec': r['dec'],
                'mass': r['mass'] if r['mass'] else float('nan'),
                'rad': r['rad'] if r['rad'] else float('nan'),
                'Teff': r['teff'] if r['teff'] else float('nan'),
                'Tmag': r['tmag'] if r['tmag'] else float('nan'),
                'plx': r['plx'] if r['plx'] else float('nan'),
            })
            if gal_b is None:
                gal_b = r['gal_b']
        return (stars, gal_b if gal_b is not None else 0.0)

    def count_nearby_cached(self) -> int:
        """Return number of TICs with cached nearby stars."""
        val = self._backend.query_scalar(
            "SELECT COUNT(DISTINCT tic_id) FROM nearby_stars")
        return int(val) if val is not None else 0

    # ------------------------------------------------------------------
    # TGLC index (D3 of docs/superpowers/specs/2026-07-20-tglc-index-to-pg.md)
    # ------------------------------------------------------------------
    #
    # These replace the ~8-10 raw `sqlite3.connect()` call sites across
    # exonit/ that queried the `tglc` table directly (exonit/metadata_index.py,
    # exonit/discovery_state.py, exonit/server.py, exonit/routes/discovery.py)
    # with a single, backend-agnostic implementation. `tglc` itself is not
    # created by MetadataCache (see the module-level comment above) — only by
    # scripts/index_tglc_batch.py (SQLite) or schema_pg_meta.sql (PG); callers
    # must tolerate a missing table (SQLite: sqlite3.OperationalError; PG:
    # psycopg.errors.UndefinedTable — both are handled per-method below, via
    # _missing_table_excs(), NARROWLY — not a bare `except Exception`).

    def _missing_table_excs(self):
        """Exception types that legitimately mean "the table does not exist
        yet" for the active backend — NOT a bare ``except Exception``.

        2026-07-20 review fix (progress/f1-metadata-backend-review.md item 4
        / D7 of the spec: "PG-down = index-down is accepted posture ... no
        SQLite live-fallback"). The first cut of the ``tglc_*`` helpers below
        caught bare ``Exception`` and degraded to an empty result for
        anything — including a genuine PG connection failure — which would
        silently read as "0 indexed" on the live download hot path once F3/F4
        wire these in, instead of surfacing the outage. Only "table missing"
        (the SQLite-era case these helpers' `except` blocks were originally
        written for — an un-provisioned/older ``tglc``/``fits_blobs``) is
        swallowed; everything else (connection refused, pool timeout, auth
        failure, ...) propagates to the caller.
        """
        if self._is_pg:
            import psycopg
            return (psycopg.errors.UndefinedTable,)
        import sqlite3
        return (sqlite3.OperationalError,)

    def tglc_lookup(self, tic_id) -> List[Dict[str, Any]]:
        """Return raw ``tglc`` rows for *tic_id*: dicts with sector, path,
        exptime, tmag, ra, dec. Empty list (not an error) if the table is
        missing or the TIC has no rows — mirrors the previous
        ``exonit/metadata_index.py::_tglc_lookup``'s "OperationalError -> no
        result" behaviour, generalised to either backend. A genuine backend
        failure (PG connection refused/timed out, ...) is NOT one of these
        cases and propagates (see ``_missing_table_excs``'s docstring).
        """
        try:
            return self._backend.query(
                "SELECT sector, path, exptime, tmag, ra, dec "
                "FROM tglc WHERE tic_id = ?",
                (str(tic_id),),
            )
        except self._missing_table_excs():
            return []

    def tglc_tic_ids(self, sector: Optional[int] = None) -> List[int]:
        """Ordered list of int TIC IDs in ``tglc``, optionally restricted to
        *sector*.

        Ordering (D3 of the spec — decided here, since PostgreSQL has no
        ``rowid``):
        - SQLite: ``ORDER BY rowid ASC`` — unchanged, exact physical
          insertion order (the historical semantics
          ``discovery_state.load_extended_canonical``/``_load_tglc_set``
          depend on for "extension TICs appended in roughly the order they
          were indexed").
        - PostgreSQL: ``ORDER BY seq ASC``, where ``seq`` is a
          ``BIGINT`` column backed by ONE shared sequence across every LIST
          partition (``scripts/migration/schema_pg_meta.sql``, same manual
          ``CREATE SEQUENCE`` + ``ALTER TABLE ... SET DEFAULT nextval(...)``
          pattern already used for ``candidates.id`` in schema_pg.sql — a
          native ``GENERATED ALWAYS AS IDENTITY`` column does NOT reliably
          share one counter across partitions). ``seq`` is monotonically
          increasing with insertion order, so this is the closest possible
          mirror of SQLite's rowid semantics on a backend with no rowid.

        Returns an empty list when the table does not exist. A genuine
        backend failure propagates instead (see ``_missing_table_excs``).

        Q3 (perf review 2026-09-26): streams via ``Backend.iter_id_column``
        instead of ``query()`` — this is the single most expensive query in
        the whole enqueue path (40% of a measured 7h exonit_meta window,
        re-run for every registered sector), and it used to build one
        dict_row/``Row`` per matched TIC before this loop discarded
        everything but the id.
        """
        order_col = "seq" if self._is_pg else "rowid"
        if sector is not None:
            sql = (f"SELECT tic_id FROM tglc WHERE sector = ? "
                   f"ORDER BY {order_col} ASC")
            params = (int(sector),)
        else:
            sql = f"SELECT tic_id FROM tglc ORDER BY {order_col} ASC"
            params = ()
        out = []
        try:
            for tic_id in self._backend.iter_id_column(sql, params):
                try:
                    out.append(int(tic_id))
                except (ValueError, TypeError):
                    pass
        except self._missing_table_excs():
            return []
        return out

    def tglc_sector_count(self, sector: Optional[int] = None) -> Optional[int]:
        """``COUNT(*)`` of ``tglc`` rows for *sector*, or every row when
        *sector* is None. Index-backed by ``ix_tglc_sector`` when given a
        sector. Returns ``None`` when the table does not exist yet, so
        callers can distinguish "0 rows" from "could not determine" —
        mirrors ``exonit/server.py::_tglc_sector_rowcount``'s previous
        contract exactly (it feeds a cache-invalidation check that must NOT
        trust a bad signal as "unchanged"). A genuine backend failure (PG
        down, ...) is NOT folded into that ``None`` — it propagates instead
        (see ``_missing_table_excs``'s docstring: a silent ``None`` here
        would read identically to "table not there yet" to the caller, the
        exact "PG outage read as harmless" failure mode D7 rules out).
        """
        try:
            if sector is not None:
                val = self._backend.query_scalar(
                    "SELECT COUNT(*) FROM tglc WHERE sector = ?",
                    (int(sector),))
            else:
                val = self._backend.query_scalar("SELECT COUNT(*) FROM tglc")
            return int(val) if val is not None else 0
        except self._missing_table_excs():
            return None

    def tglc_seq_at_offset(self, sector: int, offset: int) -> Optional[int]:
        """The ``seq``/``rowid`` value of the row at *offset* (0-based) in
        *sector*'s ``tglc`` rows, ordered ascending.

        S1 (docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md §4.1.3):
        bootstraps the paged selector's ``cursor_seq`` from the legacy
        row-count ``cursor`` the FIRST time a sector switches to paged mode
        (``SELECT seq ... OFFSET cursor-1 LIMIT 1`` in the spec's wording —
        callers pass ``offset = cursor - 1``). Returns ``None`` when *offset*
        is out of range (fewer rows than that) or the table does not exist
        yet — callers treat that as "start from the very beginning" (0).
        """
        order_col = "seq" if self._is_pg else "rowid"
        try:
            val = self._backend.query_scalar(
                f"SELECT {order_col} FROM tglc WHERE sector = ? "
                f"ORDER BY {order_col} ASC LIMIT 1 OFFSET ?",
                (int(sector), int(offset)))
            return int(val) if val is not None else None
        except self._missing_table_excs():
            return None

    def tglc_tic_ids_page(self, sector: int, after_seq: int,
                          limit: int = 20000) -> List[tuple]:
        """One page of ``(tic_id, seq)`` pairs for *sector*, ``seq`` strictly
        greater than *after_seq*, ordered ascending, capped at *limit* rows.

        S1's O(lot) paged selector (:mod:`exonit.lot_selector`): the whole
        point of this method is that it costs O(``limit``), not O(sector) —
        unlike ``tglc_tic_ids``, which returns the WHOLE sector's rows.
        Index-backed by ``ix_tglc_seq`` (PG, per-partition CONCURRENTLY —
        scripts/migration/2026-09-26_tglc_seq_index.sql) / the SQLite
        ``rowid`` (no index needed — see ``tglc_tic_ids``'s equivalent
        note). Returns ``[]`` when the table does not exist yet or the page
        is empty (end of sector reached) — both read the same to the
        caller (nothing more to select).
        """
        order_col = "seq" if self._is_pg else "rowid"
        try:
            rows = self._backend.query(
                f"SELECT tic_id, {order_col} AS seq FROM tglc "
                f"WHERE sector = ? AND {order_col} > ? "
                f"ORDER BY {order_col} ASC LIMIT ?",
                (int(sector), int(after_seq), int(limit)))
        except self._missing_table_excs():
            return []
        out = []
        for r in rows:
            try:
                out.append((int(r["tic_id"]), int(r["seq"])))
            except (ValueError, TypeError):
                pass
        return out

    def tglc_seq_for_tics(self, sector: int, tic_ids,
                          chunk_size: int = 20000) -> dict:
        """``{tic_id (int): seq (int)}`` for the subset of *tic_ids* present
        in *sector*'s ``tglc`` rows.

        S1's retry-pool probe (:mod:`exonit.lot_selector`): "is this
        previously-failed TIC still in the sector's downloaded set, and at
        what seq (for ordering the retry set)?" — the same chunked-``IN``
        pattern as ``tglc_rowcount_for_tics`` (20 000/chunk, comfortably
        under both backends' per-statement bound-parameter limits). Returns
        ``{}`` when the table does not exist yet.
        """
        order_col = "seq" if self._is_pg else "rowid"
        tics = list(tic_ids)
        out: dict = {}
        if not tics:
            return out
        try:
            for i in range(0, len(tics), chunk_size):
                chunk = tics[i:i + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                params = [int(sector)] + [str(int(t)) for t in chunk]
                rows = self._backend.query(
                    f"SELECT tic_id, {order_col} AS seq FROM tglc "
                    f"WHERE sector = ? AND tic_id IN ({placeholders})",
                    tuple(params))
                for r in rows:
                    try:
                        out[int(r["tic_id"])] = int(r["seq"])
                    except (ValueError, TypeError):
                        pass
        except self._missing_table_excs():
            return {}
        return out

    def tglc_rowcount_for_tics(self, tic_ids, sector: Optional[int] = None) -> int:
        """``COUNT(*)`` of ``tglc`` rows whose ``tic_id`` is in *tic_ids*.

        Serves ``discovery_state.count_extended_canonical``'s overlap term:
        the TICs already in the canonical prefix must not be counted twice.
        Row count, not distinct-TIC count, so it composes with
        ``tglc_sector_count`` on the same (row) unit.

        The IN list is chunked at 20 000 parameters: psycopg3 refuses a
        statement with more than 65 535 of them
        ([[gotcha_postprocess_in_list_65535_param_limit]]), and a canonical
        list is human-sized today (~5k) but nothing in the schema caps it.
        Returns 0 when the table does not exist, matching the callers'
        "no extension" reading.

        Note the ``tic_id`` column is TEXT in the tglc schema, so the
        parameters are stringified — passing ints matches no rows on SQLite
        and errors on PG.
        """
        tics = list(tic_ids)
        if not tics:
            return 0
        total = 0
        chunk_size = 20000
        try:
            for i in range(0, len(tics), chunk_size):
                chunk = tics[i:i + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                params = [str(int(t)) for t in chunk]
                if sector is not None:
                    sql = (f"SELECT COUNT(*) FROM tglc WHERE sector = ? "
                           f"AND tic_id IN ({placeholders})")
                    params = [int(sector)] + params
                else:
                    sql = (f"SELECT COUNT(*) FROM tglc "
                           f"WHERE tic_id IN ({placeholders})")
                val = self._backend.query_scalar(sql, tuple(params))
                total += int(val) if val is not None else 0
        except self._missing_table_excs():
            return 0
        return total

    def tglc_tic_ids_for_tics(self, tic_ids, sector: Optional[int] = None) -> Set[int]:
        """Subset of *tic_ids* present in ``tglc`` for *sector*.

        Companion to ``tglc_rowcount_for_tics`` (same chunking, same IN-list
        bound) for callers that need to know WHICH of a small candidate set
        of TIC ids intersects the sector's ``tglc`` rows -- not just how many.
        S2 gap-stats v2 (``exonit/gap_stats_v2.py``) uses this for the TOI /
        exhausted / canonical small-set probes against a sector, so it never
        has to materialise the whole (potentially multi-million-row) `tglc_S`
        set just to answer "which of these ~thousands of TICs are in this
        sector's index".

        Returns an empty set when the table does not exist, matching
        ``tglc_rowcount_for_tics``'s "no extension" reading.
        """
        tics = list({int(t) for t in tic_ids})
        if not tics:
            return set()
        out: Set[int] = set()
        chunk_size = 20000
        try:
            for i in range(0, len(tics), chunk_size):
                chunk = tics[i:i + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                params = [str(t) for t in chunk]
                if sector is not None:
                    sql = (f"SELECT tic_id FROM tglc WHERE sector = ? "
                           f"AND tic_id IN ({placeholders})")
                    params = [int(sector)] + params
                else:
                    sql = f"SELECT tic_id FROM tglc WHERE tic_id IN ({placeholders})"
                rows = self._backend.query(sql, tuple(params))
                for r in rows:
                    try:
                        out.add(int(r["tic_id"]))
                    except (ValueError, TypeError):
                        pass
        except self._missing_table_excs():
            return set()
        return out

    def tglc_distinct_tic_count(self, sector: Optional[int] = None) -> int:
        """``COUNT(DISTINCT tic_id)`` in ``tglc``, optionally for *sector*.

        Distinct from ``tglc_sector_count`` (a row count — TICs observed
        across multiple Gaia IDs collapse via the PK, but a row count is
        still what the previous raw call sites needed); this is for the
        batch scripts (``discovery_run.py``, ``verify_sector_complete.py``)
        migrated in a later phase. Returns 0 when the table does not exist
        yet; a genuine backend failure propagates (see
        ``_missing_table_excs``).
        """
        try:
            if sector is not None:
                val = self._backend.query_scalar(
                    "SELECT COUNT(DISTINCT tic_id) FROM tglc WHERE sector = ?",
                    (int(sector),))
            else:
                val = self._backend.query_scalar(
                    "SELECT COUNT(DISTINCT tic_id) FROM tglc")
            return int(val) if val is not None else 0
        except self._missing_table_excs():
            return 0

    def tglc_paths_for(self, paths: Iterable[str], chunk_size: int = 900) -> Set[str]:
        """Return the subset of *paths* present in ``tglc.path``.

        Chunk-scoped skip-set lookup (D5 of the spec): the fix for
        ``scripts/index_tglc_batch.py``'s unscoped ``SELECT path FROM tglc``
        (a full-table scan run once per 10k-file download chunk — the
        dominant contributor to the historical WAL-bloat/lock-contention
        incidents). SQLite: chunked ``WHERE path IN (...)`` (chunk_size
        capped well under the ~999/32766 per-statement variable limit).
        PostgreSQL: a single ``WHERE path = ANY(?)`` per call — an array
        bind avoids both the per-statement parameter limit AND the need to
        chunk at all.

        Returns an empty set when the table does not exist yet; a genuine
        backend failure propagates (see ``_missing_table_excs``) — on the
        chunk-scoped skip-set's real caller (F3's index_tglc_batch.py), a
        silently-empty result would make a live PG outage look identical to
        "nothing indexed yet" and re-download/re-pack files that are already
        there.
        """
        paths = list(paths)
        if not paths:
            return set()
        found: Set[str] = set()
        try:
            if self._is_pg:
                rows = self._backend.query(
                    "SELECT path FROM tglc WHERE path = ANY(?)",
                    (paths,))
                found.update(r["path"] for r in rows)
            else:
                for i in range(0, len(paths), chunk_size):
                    batch = paths[i:i + chunk_size]
                    placeholders = ",".join("?" * len(batch))
                    rows = self._backend.query(
                        f"SELECT path FROM tglc WHERE path IN ({placeholders})",
                        tuple(batch))
                    found.update(r["path"] for r in rows)
        except self._missing_table_excs():
            return set()
        return found

    def tglc_add_rows_bulk(self, rows) -> None:
        """Bulk upsert ``tglc`` rows (``INSERT OR REPLACE`` semantics).

        rows: iterable of (tic_id, sector, path, exptime, tmag, ra, dec)
        tuples, matching ``scripts/index_tglc_batch.py``'s column order. A
        TIC/sector PK collision (two Gaia IDs mapping to the same TIC in the
        same sector) REPLACES the row — the historical behaviour that
        motivates the ``fits_blobs``-union invariant in ``indexed_basenames``
        below.
        """
        d = self._backend.dialect
        sql = d.upsert(
            "tglc", ["tic_id", "sector", "path", "exptime", "tmag", "ra", "dec"],
            ["tic_id", "sector"])
        self._backend.executemany(sql, list(rows))

    def indexed_basenames(self, sector: int) -> Set[str]:
        """Basenames of FITS already downloaded+indexed for *sector*.

        Exact port of ``scripts/discovery_download.py::_indexed_basenames``'s
        union semantics (see that function's docstring, and the dedicated
        test class in tests/test_discovery_download.py) — the skip-set is the
        UNION of:
        - ``tglc`` rows for *sector* (one row per TIC; a PK(tic_id,sector)
          collision from a duplicate Gaia ID loses a basename).
        - ``fits_blobs`` rows whose filename matches ``%-sNNNN-%`` (one row
          per physical FITS basename — preserves basenames a tglc REPLACE
          collision dropped).

        Guard: ``fits_blobs`` may be absent (older deployments); if that
        query fails with a missing-table error the tglc-only set is returned
        (matches the ported function's fallback). Narrowed to missing-table
        only (not "or any other error", the ported script's original,
        SQLite-era wording) — a genuine backend failure on EITHER query
        propagates instead of silently degrading (see
        ``_missing_table_excs``'s docstring / D7's PG-down-is-hard-dependency
        posture): this is the union skip-set the download hot path (F3) will
        trust to decide "already indexed" vs "needs downloading", so a PG
        outage read as an empty union here would re-download nothing-is-wrong
        AND, worse, tell the caller everything still needs downloading.
        """
        try:
            rows = self._backend.query(
                "SELECT path FROM tglc WHERE sector = ?",
                (int(sector),))
            names = {os.path.basename(r["path"]) for r in rows if r["path"]}
        except self._missing_table_excs():
            return set()
        try:
            pattern = f"%-s{int(sector):04d}-%"
            rows = self._backend.query(
                "SELECT filename FROM fits_blobs WHERE filename LIKE ?",
                (pattern,))
            names.update(r["filename"] for r in rows)
        except self._missing_table_excs():
            pass  # fits_blobs absent — tglc set is sufficient
        return names

    #: Last path segment of ``tglc.path``, computed in SQL. Portable across
    #: both backends: SQLite and PG both implement ``rtrim(string, chars)``
    #: as "strip trailing characters belonging to *chars*", so
    #: ``rtrim(path, replace(path, '/', ''))`` leaves exactly the directory
    #: prefix up to and including the last ``/`` — its length is the offset
    #: the basename starts at. Matches ``os.path.basename`` on every case
    #: that matters here, including the two edge ones: a path with NO slash
    #: (rtrim consumes everything, offset 0 -> the whole string) and a path
    #: ENDING in a slash (nothing is stripped -> empty basename).
    _BASENAME_SQL = "substr(path, length(rtrim(path, replace(path, '/', ''))) + 1)"

    #: PG-only spelling of the same basename (2026-10-05). The portable
    #: ``rtrim(path, <every non-slash char of path>)`` above is quadratic per
    #: row on PG (each trailing char is tested against a ~100-char set): on
    #: sector 6's 5.48M rows the tglc arm of ``indexed_basename_count`` took
    #: 42.6 s of a 58 s query, vs 0.5 s for a plain count of the same rows.
    #: ``reverse(split_part(reverse(path), '/', 1))`` is linear and gives the
    #: identical result on every edge case (no slash -> whole string;
    #: trailing slash -> ''): verified live — 0 mismatches over all 5.48M
    #: sector-6 paths, union count identical (5,517,576), 55.8 s -> 19.6 s.
    #: SQLite has no reverse()/split_part(), so it keeps the portable form.
    _BASENAME_SQL_PG = "reverse(split_part(reverse(path), '/', 1))"

    @property
    def _basename_sql(self) -> str:
        return self._BASENAME_SQL_PG if self._is_pg else self._BASENAME_SQL

    def indexed_basename_count(self, sector: int) -> int:
        """``len(indexed_basenames(sector))``, counted in SQL.

        Same number, without ever materialising the set. The counting is what
        the caller wanted; the set was only ever an intermediate — and an
        expensive one: ``tglc`` holds ~76 M rows (13.1 M in its largest
        sector) and ``fits_blobs`` ~76.5 M, so building the union in Python
        pulled millions of dict rows over the wire and peaked at ~8 GB RSS
        *per sector* in the server's ``downloaded-counts`` refresh thread —
        for a single integer. Measured 2026-08-02 on <host>: a sawtooth to
        ~8.4 GB and back, once per sector, at a steady ~24% of a core.

        Parity with ``indexed_basenames`` is exact, and deliberately so —
        every asymmetry below mirrors that method:

        * ``UNION`` (not ``UNION ALL``) reproduces the Python ``set``: the
          two sources overlap by construction, so the count must dedup.
        * The ``path IS NOT NULL AND path <> ''`` filter is the SQL spelling
          of ``if r["path"]``. Note it filters the PATH, not the basename —
          a path ending in ``/`` yields an empty basename that IS counted,
          exactly as the set would hold ``''``.
        * ``fits_blobs`` filenames are NOT filtered, because the set
          comprehension does not filter them either.
        * Missing-table fallbacks land on the same two answers: no ``tglc``
          -> 0 (``fits_blobs`` is never consulted, even if present); no
          ``fits_blobs`` -> the tglc-only count.

        The ``fits_blobs`` predicate keeps its leading-wildcard ``LIKE``, so
        PG still sequentially scans that table — but it scans and counts
        server-side instead of shipping every row to Python. Making that
        scan itself cheap needs an index (a generated sector column, or a
        trigram index on ``filename``); that is a schema change on a 76.5 M
        row table, deliberately left out of this fix.
        """
        pattern = f"%-s{int(sector):04d}-%"
        union_sql = (
            "SELECT COUNT(*) FROM ("
            f"SELECT {self._basename_sql} AS name FROM tglc "
            "WHERE sector = ? AND path IS NOT NULL AND path <> '' "
            "UNION "
            "SELECT filename FROM fits_blobs WHERE filename LIKE ?"
            ") AS u")
        try:
            return int(self._backend.query_scalar(
                union_sql, (int(sector), pattern)) or 0)
        except self._missing_table_excs():
            pass  # one of the two tables is absent — narrow down below
        tglc_only_sql = (
            "SELECT COUNT(*) FROM ("
            f"SELECT DISTINCT {self._basename_sql} AS name FROM tglc "
            "WHERE sector = ? AND path IS NOT NULL AND path <> ''"
            ") AS u")
        try:
            return int(self._backend.query_scalar(
                tglc_only_sql, (int(sector),)) or 0)
        except self._missing_table_excs():
            return 0  # no tglc either — same answer indexed_basenames gives

    def download_registry_fingerprint(self) -> Optional[Dict[str, Dict[str, int]]]:
        """Cheap proxy for "has tglc or fits_blobs changed since we last
        looked?" — for callers deciding whether to repeat the expensive
        ``indexed_basename_count``/``indexed_basenames`` scan above.

        Reads PostgreSQL's per-table stats-collector counters
        (``pg_stat_user_tables``) for exactly the two tables that scan
        touches. This is a catalog lookup (in-memory counters the collector
        already maintains), never a scan of the tables themselves, so it
        costs essentially nothing however large ``tglc``/``fits_blobs`` grow
        — unlike the union-count it exists to let callers skip.

        ``tglc`` is a declaratively PARTITIONED table in production
        (``scripts/migration/schema_pg_meta.sql``, ``PARTITION BY LIST
        (sector)``, one ``tglc_sNNNN`` child per sector): PostgreSQL routes
        every ``INSERT INTO tglc`` to the matching child, and ONLY the
        child's ``pg_stat_user_tables`` row moves — the parent ``tglc``
        relation's own row stays permanently zero (2026-09-25 review finding:
        a naive ``WHERE relname = 'tglc'`` read that parent and was
        therefore blind to every TGLC download, reacting only to
        ``fits_blobs`` — which lags behind, since packing to object storage
        is a separate later step). ``pg_partition_tree(<table>::regclass)``
        returns the WHOLE partition tree (parent + every descendant, or just
        the table itself for a non-partitioned one like ``fits_blobs`` — so
        the same code path handles both without a branch), and this sums the
        counters across every relation in it. The relation COUNT is included
        too: a brand-new ``tglc_sNNNN`` partition (``ensure_tglc_partition``)
        appears in ``pg_partition_tree`` — and gets its own zeroed
        ``pg_stat_user_tables`` row — the instant it's created, before a
        single row lands in it, so attaching one flips the fingerprint even
        on an otherwise exact stats tie.

        Returns ``None`` on SQLite (no such view/function), if either table
        is missing entirely (the ``::regclass`` cast raises), or on any other
        read failure (stats collector down, permissions, connection lost,
        ...): callers MUST treat ``None`` as "no signal — always recompute",
        exactly the behaviour that predates this method.

        These counters are lifetime-cumulative and reset to 0 on a
        PostgreSQL restart (documented stats-collector behaviour), so a
        later fingerprint can legitimately be SMALLER than a stored one.
        Callers must compare for exact equality, never "no larger than
        before" — any difference, larger OR smaller, means "recompute",
        which covers the post-restart case with no extra logic needed here.
        """
        if not self._is_pg:
            return None
        try:
            out: Dict[str, Dict[str, int]] = {}
            for table in ("tglc", "fits_blobs"):
                # UNION of two branches, not just pg_partition_tree() alone:
                # verified against real PostgreSQL 18 that
                # pg_partition_tree() returns ZERO rows for a plain,
                # non-partitioned table (e.g. fits_blobs) -- it is only
                # populated for an actual partitioning hierarchy (and, for a
                # partitioned one, it ALSO already includes the parent's own
                # row at level 0 -- also verified). Branch 1 (direct relid
                # match) is what makes fits_blobs (and any future
                # non-partitioned table) work at all; branch 2 (the
                # partition tree) is what makes tglc's children visible, and
                # duplicates branch 1's row for a partitioned table's parent.
                #
                # relid is SELECTed (and UNION, not UNION ALL, used) so that
                # dedup happens on relation identity, not on the four numeric
                # columns being coincidentally equal -- two DIFFERENT empty
                # partitions both showing all-zero counters must still count
                # as two relations (n_relations below), or a brand-new empty
                # partition attach would go undetected whenever another
                # all-zero partition already existed.
                rows = self._backend.query(
                    "SELECT relid, n_tup_ins, n_tup_upd, n_tup_del, n_live_tup "
                    "FROM pg_stat_user_tables WHERE relid = ?::regclass::oid "
                    "UNION "
                    "SELECT s.relid, s.n_tup_ins, s.n_tup_upd, s.n_tup_del, s.n_live_tup "
                    "FROM pg_stat_user_tables s "
                    "JOIN pg_partition_tree(?::regclass) pt ON pt.relid = s.relid",
                    (table, table))
                out[table] = {
                    "n_tup_ins": sum(int(r["n_tup_ins"] or 0) for r in rows),
                    "n_tup_upd": sum(int(r["n_tup_upd"] or 0) for r in rows),
                    "n_tup_del": sum(int(r["n_tup_del"] or 0) for r in rows),
                    "n_live_tup": sum(int(r["n_live_tup"] or 0) for r in rows),
                    "n_relations": len(rows),
                }
        except Exception:
            return None
        return out

    def tglc_basenames_missing_blobs(self, sector: int) -> Dict[str, str]:
        """Basenames present in ``tglc`` for *sector* but ABSENT from
        ``fits_blobs`` — the exact "packed-to-MinIO gap"
        ``scripts/repair_missing_blobs.py`` exists to close (2026-07-21
        MinIO multipart read-timeout incident: 56 s7 chunks got
        ``pack_failed`` and left their 10k files each indexed in ``tglc``
        with no blob and no tar in object storage — invisible to every later
        run because ``indexed_basenames``'s tglc ∪ fits_blobs union already
        counts them as "done").

        Returns ``{basename: path}`` — *path* is ``tglc.path`` exactly as
        stored (the absolute path recorded by ``index_tglc_batch.py`` at
        index time, see that script's ``read_one``), which
        ``scripts/repair_missing_blobs.py`` uses to locate the RAW FITS on
        disk without a second query.

        PostgreSQL: ONE server-side query — ``NOT EXISTS`` against
        ``fits_blobs`` (``PRIMARY KEY`` on ``filename``, so each check is an
        index lookup, not a scan) with the basename extracted via
        ``substring(path from '[^/]+$')`` (last path segment). No basenames
        cross into Python beyond the gap-sized result itself — the "avoid
        loading millions of basenames into Python" requirement this method
        exists to satisfy.

        SQLite: no portable server-side "last path segment" expression, and
        this backend is the dev/small-sector path (the current production
        metadata store is PostgreSQL — see docs/superpowers/specs/
        2026-07-20-tglc-index-to-pg.md), so it falls back to computing the
        set difference from two per-sector queries — the same shape
        ``indexed_basenames`` already issues — chunking the ``fits_blobs``
        membership check the same way ``tglc_paths_for`` does.

        Returns an empty dict when ``tglc`` is missing (nothing to repair —
        no index means no gap either). If ``fits_blobs`` itself is missing,
        every ``tglc`` basename for the sector is "missing" (nothing has
        ever been packed). A genuine backend failure (PG down, ...)
        propagates instead of silently degrading (see
        ``_missing_table_excs``'s docstring — same "PG-down must not read as
        silently OK" posture as every other ``tglc_*`` helper here).
        """
        if self._is_pg:
            try:
                rows = self._backend.query(
                    "SELECT path, substring(path from '[^/]+$') AS basename "
                    "FROM tglc WHERE sector = ? "
                    "AND NOT EXISTS (SELECT 1 FROM fits_blobs fb "
                    "WHERE fb.filename = substring(tglc.path from '[^/]+$'))",
                    (int(sector),))
            except self._missing_table_excs():
                return {}
            return {r["basename"]: r["path"] for r in rows if r["basename"]}

        try:
            rows = self._backend.query(
                "SELECT path FROM tglc WHERE sector = ?", (int(sector),))
            tglc_by_basename = {os.path.basename(r["path"]): r["path"]
                                for r in rows if r["path"]}
        except self._missing_table_excs():
            return {}
        if not tglc_by_basename:
            return {}
        names_list = list(tglc_by_basename)
        have: Set[str] = set()
        chunk_size = 900
        try:
            for i in range(0, len(names_list), chunk_size):
                batch = names_list[i:i + chunk_size]
                placeholders = ",".join("?" * len(batch))
                rows = self._backend.query(
                    f"SELECT filename FROM fits_blobs WHERE filename IN ({placeholders})",
                    tuple(batch))
                have.update(r["filename"] for r in rows)
        except self._missing_table_excs():
            return dict(tglc_by_basename)  # fits_blobs absent -> all missing
        return {bn: p for bn, p in tglc_by_basename.items() if bn not in have}

    def _partition_race_excs(self):
        """Exception types raised when THIS caller loses a race against a
        concurrent ``ensure_tglc_partition`` call for the SAME sector (PG
        only — see that method's docstring).

        2026-07-20 review fix (progress/f3-indexer-pg-review.md, point 5,
        BLOCKER): ``CREATE TABLE IF NOT EXISTS ... PARTITION OF`` is NOT
        atomic against true concurrent DDL in PostgreSQL — the "does it
        already exist" check and the CREATE are not one operation, so two
        sessions can both see "missing" and both attempt the CREATE; only one
        commits, the other(s) fail. Reproduced empirically by the reviewer: 5
        concurrent callers -> 4 crashes. Two distinct error classes can
        surface depending on exactly where PG detects the collision:
        ``DuplicateTable`` (42P07, the common case — the loser's CREATE
        itself fails against the now-existing relation) and, more rarely,
        ``UniqueViolation`` against a system catalog (pg_class/pg_type) if
        the collision is detected earlier in the parse/plan pipeline for the
        same underlying race. Both mean exactly the same thing here: the
        partition this call wanted now exists — created by the other
        caller — which is precisely the outcome ``IF NOT EXISTS`` was meant
        to guarantee, so both are swallowed as success.
        """
        import psycopg
        return (psycopg.errors.DuplicateTable, psycopg.errors.UniqueViolation)

    def ensure_tglc_partition(self, sector: int) -> None:
        """Ensure a LIST partition for *sector* exists on ``tglc`` (PG only).

        No-op on SQLite (one flat table holds every sector — see
        ``scripts/migration/schema_pg_meta.sql``'s partitioning note). On
        PostgreSQL, idempotently creates ``tglc_s<NNNN> PARTITION OF tglc FOR
        VALUES IN (<sector>)`` so ``tglc_add_rows_bulk`` never hits "no
        partition of relation tglc found for row" on a brand-new sector.
        Called by the indexer (F3) before its first bulk insert for a sector;
        harmless to call repeatedly (``IF NOT EXISTS``) — including under
        CONCURRENT callers for the same sector, which lose a benign race
        instead of crashing (see ``_partition_race_excs``'s docstring; this
        is the 2026-07-20 review's required fix, reproduced against real PG
        with 5 concurrent callers before the fix and re-verified after it in
        ``tests/test_index_tglc_batch.py::TestPgEnsureTglcPartitionConcurrency``).
        """
        if not self._is_pg:
            return
        sector = int(sector)
        table = f"tglc_s{sector:04d}"
        # DDL cannot be parameterised; `sector`/`table` are validated ints /
        # a fixed f-string pattern, not user input, so this is not an
        # injection risk (same reasoning as scripts/migration/schema_pg.sql's
        # per-sector CREATE TABLE ... PARTITION OF statements).
        try:
            self._backend.execute(
                f"CREATE TABLE IF NOT EXISTS {table} PARTITION OF tglc "
                f"FOR VALUES IN ({sector})")
        except self._partition_race_excs():
            # Each PgBackend.execute() call borrows-and-returns its own
            # connection via the pool's context manager (see PgBackend.execute),
            # which already rolls back the failed statement's transaction
            # before the connection goes back to the pool — nothing further
            # to clean up here. The partition exists (created by the winner);
            # goal achieved.
            pass

    def ensure_tglc_table(self) -> None:
        """Ensure the ``tglc`` table (+ ``ix_tglc_tic``) exists (SQLite only).

        F3 counterpart of ``ensure_tglc_partition`` (PG-only): before the seam,
        ``scripts/index_tglc_batch.py`` self-provisioned ``tglc`` via its own
        raw connection on every run (see the module comment above — this table
        was deliberately never created by ``MetadataCache.__init__``). On
        PostgreSQL the table is provisioned out-of-band by
        ``scripts/migration/schema_pg_meta.sql`` (F2) — no-op here, mirroring
        ``ensure_tglc_partition``'s SQLite/PG symmetry the other way round.
        """
        if self._is_pg:
            return
        with self._backend.raw_connection() as conn:
            conn.execute(_CREATE_TGLC_TABLE_SQL)
            conn.execute(_CREATE_TGLC_TIC_INDEX_SQL)
            conn.commit()

    def set_sqlite_busy_timeout(self, ms: int) -> None:
        """Override this instance's SQLite ``busy_timeout`` (SQLite only;
        no-op on PostgreSQL, which has no such concept — MVCC readers never
        block writers).

        ``SqliteBackend`` unifies every ``MetadataCache`` caller's
        ``busy_timeout`` at 30s (D3 of the 2026-07-20 spec) — correct for
        everything EXCEPT ``scripts/index_tglc_batch.py``, which needs its
        historical 300s wait (added 2026-07-18 after a "database is locked"
        kill on a 681k-file download — see that script's own comment): it can
        coexist with a live discovery run holding the file open for hours.
        Each ``MetadataCache`` instance owns its own dedicated SQLite
        connection (this script always constructs its own instance, as a
        subprocess — never the server's shared per-process singleton), so
        this override is scoped to the caller's own connection only and
        cannot affect any other consumer's 30s default.
        """
        if self._is_pg:
            return
        with self._backend.raw_connection() as conn:
            conn.execute(f"PRAGMA busy_timeout = {int(ms)}")

    @property
    def is_pg(self) -> bool:
        """True when this instance is backed by PostgreSQL (PUBLIC mirror of
        ``self._is_pg`` for external callers, e.g.
        ``scripts/refresh_metadata_bulk.py``'s ``--report``, that need to
        branch on backend without reaching into a private attribute)."""
        return self._is_pg

    # Sample staleness report — docs/superpowers/specs/2026-09-26-metadata-
    # refresh-and-bulk-enqueue-design.md §1.3's repeatable measurement query:
    # per sector, what fraction of `tglc` rows belong to a TIC whose
    # `metadata` row does not (yet) list that sector. PostgreSQL only
    # (TABLESAMPLE + jsonb containment); cost ~1-2 min at production scale
    # per the spec.
    _SECTOR_STALENESS_SQL = """
        WITH s AS (
            SELECT tic_id, sectors::jsonb AS secs
            FROM metadata TABLESAMPLE SYSTEM (?)
        ),
        j AS (
            SELECT s.tic_id, s.secs, t.sector
            FROM s JOIN tglc t ON t.tic_id = s.tic_id
        )
        SELECT sector, count(*) AS tglc_rows,
               count(*) FILTER (WHERE NOT secs @> to_jsonb(sector)) AS missing
        FROM j GROUP BY sector ORDER BY sector
    """

    def sector_staleness_report(self, sample_pct: float = 0.1):
        """Per-sector sample of §1.3's staleness measurement: for the
        ``sample_pct``%% of ``metadata`` rows sampled, how many ``tglc`` rows
        per sector belong to a TIC whose ``metadata.sectors`` does not (yet)
        contain that sector.

        Returns a list of ``{"sector", "tglc_rows", "missing"}`` dicts, or
        ``None`` on SQLite (TABLESAMPLE/jsonb containment are PostgreSQL-only
        — this is a diagnostic tool, not a hot-path query, so no SQLite
        fallback is provided).
        """
        if not self._is_pg:
            return None
        return self._backend.query(self._SECTOR_STALENESS_SQL, (sample_pct,))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying connection/pool."""
        self._backend.close()


# ---------------------------------------------------------------------------
# Cache population helpers
# ---------------------------------------------------------------------------

_FITS_RE = re.compile(
    r"tess\d+-s(?P<sector>\d{4})-(?P<tic_id>\d{16})-(?P<exptime>\d+)-s_lc\.fits$"
)


def build_from_fits_listing(
    cache: MetadataCache,
    fits_filenames: List[str],
) -> Dict[str, int]:
    """Populate *cache* from a list of TESS FITS filenames.

    Parses each filename to extract sector, TIC ID and exposure time, groups
    entries by TIC ID, then inserts one cache entry per star.

    Returns a stats dict with keys ``stars_added`` and ``fits_parsed``.
    """
    # tic_id (stripped) -> {"sectors": set, "fits_files": list, "exptime": float}
    stars: Dict[str, Any] = defaultdict(lambda: {"sectors": set(), "fits_files": [], "exptime": 0.0})
    fits_parsed = 0

    for fname in fits_filenames:
        m = _FITS_RE.search(fname)
        if m is None:
            continue
        tic_id = str(int(m.group("tic_id")))  # strip leading zeros
        sector = int(m.group("sector"))
        exptime = float(int(m.group("exptime")))
        stars[tic_id]["sectors"].add(sector)
        stars[tic_id]["fits_files"].append(fname)
        stars[tic_id]["exptime"] = exptime
        fits_parsed += 1

    for tic_id, data in stars.items():
        cache.add_entry(
            tic_id=tic_id,
            sectors=sorted(data["sectors"]),
            exptime=data["exptime"],
            fits_files=data["fits_files"],
        )

    return {"stars_added": len(stars), "fits_parsed": fits_parsed}


def _extract_stellar_params(fits_path: str) -> Dict[str, Optional[float]]:
    """Extract stellar parameters from a TESS FITS header."""
    try:
        from astropy.io import fits as pyfits
        hdr = pyfits.getheader(fits_path, ext=0)
        return {
            "stellar_radius": hdr.get("RADIUS"),
            "teff": hdr.get("TEFF"),
            "logg": hdr.get("LOGG"),
            "tmag": hdr.get("TESSMAG"),
            "ra": hdr.get("RA_OBJ"),
            "dec": hdr.get("DEC_OBJ"),
        }
    except Exception:
        return {}


def build_from_nfs(
    cache: MetadataCache,
    nfs_path: Optional[str] = None,
    extract_stellar: bool = True,
) -> Dict[str, int]:
    """Walk *nfs_path* for ``*_lc.fits`` files and populate *cache*.

    *nfs_path* defaults to ``config.nfs_cache_path`` (``EXONIT_NFS_CACHE``).

    If *extract_stellar* is True, reads FITS headers to get stellar params
    (RADIUS, TEFF, LOGG, TESSMAG, RA, DEC) — one read per star.

    Returns a stats dict with keys ``stars_added`` and ``fits_parsed``.
    """
    if nfs_path is None:
        from .config import config as _config
        nfs_path = _config.nfs_cache_path
    fits_filenames: List[str] = []
    path_entries: List[tuple] = []
    for root, _dirs, files in os.walk(nfs_path):
        for fname in files:
            if fname.endswith("_lc.fits"):
                fits_filenames.append(fname)
                path_entries.append((fname, os.path.join(root, fname)))

    print(f"build_from_nfs: found {len(fits_filenames)} _lc.fits files under {nfs_path}")
    stats = build_from_fits_listing(cache, fits_filenames)
    cache.add_fits_paths_bulk(path_entries)
    print(f"build_from_nfs: {stats['stars_added']} stars added, {stats['fits_parsed']} files parsed, "
          f"{len(path_entries)} paths indexed")

    # Extract stellar params from FITS headers (one per star)
    if extract_stellar:
        tic_ids = cache.get_all_tic_ids()
        stellar_count = 0
        for i, tic_id in enumerate(tic_ids):
            entry = cache.get_entry(tic_id)
            if not entry or entry.get("stellar_radius") is not None:
                continue  # already has stellar params
            # Read first FITS file for this star
            for fname in entry.get("fits_files", []):
                path = cache.get_fits_path(fname)
                if path and os.path.isfile(path):
                    params = _extract_stellar_params(path)
                    if params:
                        cache.add_entry(
                            tic_id=tic_id,
                            sectors=entry["sectors"],
                            exptime=entry["exptime"],
                            fits_files=entry["fits_files"],
                            **params,
                        )
                        stellar_count += 1
                    break
            if (i + 1) % 1000 == 0:
                print(f"  Stellar params: {i+1}/{len(tic_ids)} checked, {stellar_count} extracted")
        print(f"build_from_nfs: {stellar_count} stellar params extracted from FITS headers")
        stats["stellar_extracted"] = stellar_count

    return stats
