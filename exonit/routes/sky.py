"""Live sky map endpoints (Fase 1+2, docs/superpowers/specs/2026-07-18-
sky-map-phase1.md and -phase2.md).

Serves the precomputed static star-field background (scripts/
precompute_sky_background.py) plus a live snapshot of in-progress and
recently-completed stars for www/sky.html's fixed-sphere canvas 2D map.

Patró Fase D: ``def register(app, deps: ServerDeps)``, closures ``@app.get``
(see docs/reviews/2026-07-02-opus48-cleanup-review.md for the module split
rationale). Read-only, public endpoints (no auth), same as ``/api/stars``.

Every endpoint goes through ``deps.cached_json`` (gotcha ec2045de:
uncached aggregate polling against PG mid-run saturates I/O) — the
background file is the one exception, since it's a static file served
straight off disk, not a DB query.
"""

import sqlite3
import threading

from fastapi import Query
from fastapi.responses import FileResponse, JSONResponse

from exonit import server as _server
from exonit.server import _PROJECT_ROOT
from exonit.routes.deps import ServerDeps

# NOTE: _get_shared_metadata_cache is called as `_server.<name>` (not
# plain-imported) below — tests patch it via patch("exonit.server.<name>", ...)
# after the app has already been built, same rationale as proxies.py's NOTE.

# Sostre (ceiling) on the number of in-progress TICs resolved per /api/sky/live
# call — matches the spec's "sostre 500" and keeps the MetadataCache lookups
# (one per active TIC) bounded even if the fleet has many workers assigned.
_ACTIVE_CEILING = 500

# "recent" default window for /api/sky/live — enough to keep the map's
# just-completed layer populated between 5s polls without re-sending the
# whole history (the client accumulates across polls itself).
_RECENT_LIMIT = 200

_SKY_BG_DIR = _PROJECT_ROOT / "data" / "discovery" / "sky_bg"

# Fase 2: per-star distance (parallax-derived), written by scripts/
# crossmatch_gaia_distances.py. Read-only, lazy, fail-safe: the file may not
# exist yet for a sector (crossmatch is a one-shot operator/orchestrator
# step, never on-demand), and callers must treat that as "no distances",
# never an error — same spirit as _get_shared_metadata_cache in server.py,
# but scoped to this module per the spec.
_GAIA_DIST_PATH = _PROJECT_ROOT / "data" / "discovery" / "sky_bg" / "gaia_dist.db"

# Keyed by str(path) (not a bare singleton) so tests that monkeypatch
# _GAIA_DIST_PATH to a fresh tmp_path each time never collide with a
# connection opened for a different path in an earlier test. A missing file
# is deliberately NOT cached as None: the crossmatch can complete later
# while the server keeps running, and re-checking os.path.exists() is cheap.
_gaia_dist_conns: dict = {}
_gaia_dist_conns_lock = threading.Lock()


def _get_gaia_dist_conn():
    """Lazy shared read-only connection to gaia_dist.db, or None if absent."""
    path_str = str(_GAIA_DIST_PATH)
    con = _gaia_dist_conns.get(path_str)
    if con is not None:
        return con
    with _gaia_dist_conns_lock:
        con = _gaia_dist_conns.get(path_str)
        if con is None:
            if not _GAIA_DIST_PATH.exists():
                return None
            # check_same_thread=False: FastAPI runs sync handlers on a
            # threadpool, so this shared connection is used from many
            # threads. Cross-thread use of one sqlite3 connection is only
            # safe serialized — _dist_pc_batch holds the module lock for
            # every execute.
            con = sqlite3.connect(
                f"file:{path_str}?mode=ro", uri=True, timeout=30.0,
                check_same_thread=False)
            con.execute("PRAGMA busy_timeout = 30000")
            _gaia_dist_conns[path_str] = con
    return con


def _dist_pc_batch(tic_ids) -> dict:
    """Return {tic_id: dist_pc} for the given tic_ids in one batched read.

    Looked up by tic_id alone (not (tic_id, sector)): a star's Gaia parallax
    doesn't depend on which TESS sector observed it, so any sector's
    crossmatch row for that tic_id is a valid distance. Fail-safe: an empty
    dict (every tic_id absent -> null downstream) if gaia_dist.db doesn't
    exist yet, never an error.
    """
    con = _get_gaia_dist_conn()
    if con is None or not tic_ids:
        return {}
    ids = list({str(t) for t in tic_ids})
    lookup = {}
    chunk_size = 900  # well under SQLite's bound-parameter limit
    for i in range(0, len(ids), chunk_size):
        batch = ids[i:i + chunk_size]
        placeholders = ",".join("?" * len(batch))
        with _gaia_dist_conns_lock:
            rows = con.execute(
                f"SELECT tic_id, dist_pc FROM gaia_dist WHERE tic_id IN "
                f"({placeholders}) AND dist_pc IS NOT NULL",
                batch,
            ).fetchall()
        for tic_id, dist_pc in rows:
            lookup[str(tic_id)] = dist_pc
    return lookup

# Strong-tier predicate, textually identical to the idx_cand_strong partial
# index (scripts/migration/schema_pg.sql:246-248) — the spec points at this
# exact index rather than db._strong_where's fuller agreement-gate predicate,
# so this classification intentionally stays that simpler shape. NULLIF(...,
# -1): exominer_score carries a -1 no-DV cache-negative sentinel (see
# db._strong_where's docstring) that must read as "absent", not "score -1".
_STRONG_PREDICATE = """(c.cnn_score >= 0.7 OR c.exominer_ffi_score >= 0.5)
      AND COALESCE(NULLIF(c.exominer_score, -1), c.exominer_ffi_score) >= 0.7"""

# Single query (CASE + correlated EXISTS subqueries) classifying every
# returned star's result tier in one round trip — no N+1 per star.
_CLASSIFIED_STARS_SQL = f"""
    SELECT s.tic_id, s.ra, s.dec, s.tmag, s.analyzed_at,
           CASE
             WHEN EXISTS (
                 SELECT 1 FROM candidates c
                 WHERE c.tic_id = s.tic_id AND {_STRONG_PREDICATE}
             ) THEN 'strong'
             WHEN EXISTS (
                 SELECT 1 FROM candidates c WHERE c.tic_id = s.tic_id
             ) THEN 'candidate'
             ELSE 'none'
           END AS result
    FROM stars s
    WHERE s.status = 'completed'
    ORDER BY s.analyzed_at DESC
    LIMIT ?
"""


def register(app, deps: ServerDeps) -> None:
    db = deps.db
    cfg = deps.cfg
    _cached_json = deps.cached_json

    def _classified_stars(limit: int):
        rows = db._query(_CLASSIFIED_STARS_SQL, (limit,))
        rows = list(rows)
        dist_lookup = _dist_pc_batch([r["tic_id"] for r in rows])
        return [
            {
                "tic_id": str(r["tic_id"]),
                "ra": r["ra"],
                "dec": r["dec"],
                "tmag": r["tmag"],
                "result": r["result"],
                "dist_pc": dist_lookup.get(str(r["tic_id"])),
            }
            for r in rows
        ]

    def _active_stars():
        """TICs currently ``assigned`` to a worker, resolved to ra/dec/tmag.

        Capped at _ACTIVE_CEILING; TICs with no MetadataCache entry (not yet
        downloaded/indexed) are skipped silently — the spec's contract.
        """
        import os as _os
        if not _os.path.exists(cfg.metadata_cache_path):
            return []
        rows = db._query(
            "SELECT DISTINCT tic_id FROM tasks WHERE status = 'assigned' "
            "LIMIT ?", (_ACTIVE_CEILING,))
        cache = _server._get_shared_metadata_cache(cfg)
        resolved = []
        for row in rows:
            tic_id = str(row["tic_id"])
            entry = cache.get_entry(tic_id)
            if not entry or entry.get("ra") is None or entry.get("dec") is None:
                continue
            resolved.append((tic_id, entry))
        dist_lookup = _dist_pc_batch([tic_id for tic_id, _ in resolved])
        active = [
            {
                "tic_id": tic_id,
                "ra": entry["ra"],
                "dec": entry["dec"],
                "tmag": entry.get("tmag"),
                "dist_pc": dist_lookup.get(tic_id),
            }
            for tic_id, entry in resolved
        ]
        return active

    @app.get("/api/sky/background")
    def sky_background(sector: int = Query(..., description="TESS sector number")):
        path = _SKY_BG_DIR / f"sector_{sector}.json"
        if not path.exists():
            return JSONResponse(
                {
                    "error": f"no background for sector {sector} — run "
                             f"scripts/precompute_sky_background.py --sector {sector}",
                },
                status_code=404,
            )
        # Static file, regenerated infrequently (once per sector): short
        # Cache-Control lets the browser skip re-fetching on every poll
        # without going stale for long if it's ever regenerated.
        return FileResponse(
            path, media_type="application/json",
            headers={"Cache-Control": "public, max-age=300"})

    @app.get("/api/sky/live")
    def sky_live():
        def _compute():
            return {
                "active": _active_stars(),
                "recent": _classified_stars(_RECENT_LIMIT),
            }
        return _cached_json("sky_live", 5.0, _compute)

    @app.get("/api/sky/completed")
    def sky_completed(limit: int = Query(5000, ge=1, le=5000)):
        def _compute():
            return {"stars": _classified_stars(limit)}
        return _cached_json(f"sky_completed:{limit}", 30.0, _compute)
