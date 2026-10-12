"""Candidates/stars/exports/downloads/scorecards/cTOI endpoints (Fase D split).

Extracted verbatim from exonit.server.create_app() — see
docs/reviews/2026-07-02-opus48-cleanup-review.md and
progress/impl-createapp-split-2026-07-02.md for the module map. Endpoint
bodies below are byte-identical to the pre-split code (git move-detection
should match them); only this file's import/register() preamble is new.

Covers: /api/stars, /api/candidates, /api/export/csv(+json),
/api/completeness, /api/ctoi/*, /api/downloads/* (status/catalog/refresh/
rebuild), /api/scorecards*.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Query
from fastapi.responses import JSONResponse

from exonit import server as _server
from exonit.db_backend import select_in_chunks
from exonit.server import (
    _PROJECT_ROOT,
    _load_downloads_catalog_rows,
    logger,
)
from exonit.routes.deps import ServerDeps

# NOTE: _get_shared_metadata_cache / _scorecards_dir / _toi_catalog_path are
# called as `_server.<name>` (not plain-imported) below — tests patch them
# via patch/patch.object("exonit.server.<name>"/(srv, "<name>"), ...) AFTER
# the app has already been built; see proxies.py's NOTE for the full
# rationale.


def register(app, deps: ServerDeps) -> None:
    db = deps.db
    cfg = deps.cfg
    _cached_json = deps.cached_json

    @app.get("/api/stars")
    def api_stars(
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        # Rewritten to limit stars FIRST (using the partial index on completed
        # rows ordered by analyzed_at), then count candidates only for the
        # returned rows via a correlated subquery.  The old LEFT JOIN + GROUP BY
        # forced PG to aggregate all ~3M stars × 7.5M candidates before paging,
        # causing 5–7 s / 35 GB I/O per call during backfill dashboard polling.
        # Semantically identical: stars.tic_id is the PK so the old GROUP BY
        # yielded exactly one row per star.
        def _compute():
            return db._query(
                """SELECT s.*,
                          (SELECT COUNT(*) FROM candidates c
                           WHERE c.tic_id = s.tic_id) AS n_candidates
                   FROM stars s
                   WHERE s.status = 'completed'
                   ORDER BY s.analyzed_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            )
        # Cache key includes pagination params so different pages don't collide.
        return _cached_json(f"api_stars:{limit}:{offset}", 10.0, _compute)

    @app.get("/api/candidates")
    def api_candidates(
        tic_id: Optional[str] = Query(None),
        min_snr: Optional[float] = Query(None),
        filter: Optional[str] = Query(None),
        sort: Optional[str] = Query("snr"),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        if tic_id:
            return db.get_candidates(tic_id)

        def _compute():
            # Build query based on filter
            where_clauses = []
            params = []
            if filter == "strong":
                threshold = float(os.environ.get(
                    "EXONIT_PHASE3_AGREEMENT_THRESHOLD", "0.6"))
                # D1 (docs/reviews/2026-07-02-opus48-cleanup-review.md):
                # repointed at db._strong_where — the single source of truth
                # also used by db.strong_candidates/count_strong_candidates
                # (this was previously a third hand-built copy, the family
                # that shipped the LEAST-vs-min dialect bug). The endpoint's
                # own unconditional "snr >= 10" folds into _strong_where's
                # min_snr floor: effective floor is 10, or max(10, the
                # caller's min_snr) if higher — mathematically identical to
                # ANDing the old hardcoded "snr >= 10" with a separate
                # "snr >= ?" clause for the caller's min_snr.
                effective_min_snr = 10.0 if min_snr is None else max(10.0, min_snr)
                strong_sql, strong_params = db._strong_where(
                    threshold, effective_min_snr)
                where_clauses.append(strong_sql)
                params.extend(strong_params)
            elif filter == "validated":
                where_clauses.append(
                    """(triceratops_verdict = 'validated' OR (
                       triceratops_verdict IS NULL
                       AND triceratops_fpp IS NOT NULL
                       AND triceratops_nfpp IS NOT NULL
                       AND triceratops_fpp >= 0 AND triceratops_fpp < 0.015
                       AND triceratops_nfpp >= 0 AND triceratops_nfpp < 0.001))""".strip())
            elif filter == "planet":
                where_clauses.append("claude_verdict = 'planet_candidate'")

            # filter=="strong" already folded min_snr into the predicate above.
            if min_snr is not None and filter != "strong":
                where_clauses.append("snr >= ?")
                params.append(min_snr)

            where = " AND ".join(where_clauses) if where_clauses else "1=1"
            # B5 (docs/reviews/2026-07-02-opus48-cleanup-review.md): every
            # sort column here is nullable (snr/period_days/depth_ppm/
            # cnn_score are all REAL with no NOT NULL) — PG's default NULL
            # ordering differs by direction from SQLite's (see
            # exonit.db_backend.Dialect.nulls_last), so NULL-scored/NULL-
            # dated rows silently reordered across the PG migration. Explicit
            # NULLS LAST on every option: NULL-valued rows always sort last
            # regardless of ASC/DESC, a single consistent UI contract instead
            # of mirroring each direction's dialect-specific default.
            sort_col = {
                "snr": db.dialect.nulls_last("snr", "DESC"),
                "period": db.dialect.nulls_last("period_days", "ASC"),
                "depth": db.dialect.nulls_last("depth_ppm", "DESC"),
                "cnn": db.dialect.nulls_last("cnn_score", "DESC"),
            }.get(sort, db.dialect.nulls_last("snr", "DESC"))

            total = db._query_scalar(
                f"SELECT COUNT(*) FROM candidates WHERE {where}", tuple(params))
            rows = db._query(
                f"SELECT * FROM candidates WHERE {where} ORDER BY {sort_col} "
                f"LIMIT ? OFFSET ?",
                tuple(params) + (limit, offset))
            return {"candidates": [dict(r) for r in rows], "total": total}

        # P1 (docs/reviews/2026-07-02-opus48-cleanup-review.md): the
        # no-filter call is the UI's default 15s dashboard poll and used to
        # run an uncached COUNT(*) over the whole candidates table (plus the
        # paged SELECT) on every single poll — same class of bug as
        # /api/stars (fixed ec2045de). Cache the whole response (rows + count
        # share the same WHERE) keyed by every param that changes the query.
        # TTL 20s: short enough that admin actions (verdict recompute, FPP
        # results landing) show up promptly, long enough that a steady 15s
        # UI poll almost always hits a warm cache across all 4 A3 processes.
        cache_key = f"api_candidates:{filter}:{min_snr}:{sort}:{limit}:{offset}"
        return _cached_json(cache_key, 20.0, _compute)

    # P2 (docs/reviews/2026-07-02-opus48-cleanup-review.md): hard row cap for
    # the export endpoints below. db.get_all_candidates(min_snr) has no
    # LIMIT — on the multi-million-row candidates table this is the exact
    # query shape that saturated PG I/O in the 2026-06-20 has_spoc meltdown
    # (see docs gotchas). These endpoints have no server-side pagination
    # (the whole result is buffered into one response), so instead of adding
    # pagination (a bigger UI change) we cap at _EXPORT_ROW_CAP rows and
    # surface an explicit truncation flag — request a higher min_snr to
    # shrink the true result under the cap instead. Overridable via env var
    # so tests can exercise the truncation path without inserting 50k rows.
    try:
        _EXPORT_ROW_CAP = int(os.environ.get("EXONIT_EXPORT_ROW_CAP", "50000"))
    except ValueError:
        _EXPORT_ROW_CAP = 50000

    def _bounded_candidate_export(min_snr):
        """Fetch candidates for /api/export/* with a hard row cap.

        Same predicate/ORDER BY as db.get_all_candidates(min_snr) (snr DESC
        NULLS LAST) — just bounded. Fetches cap+1 rows to detect overflow,
        then trims to the cap. Returns (rows, truncated).
        """
        order = db.dialect.nulls_last("snr")
        if min_snr is not None:
            rows = db._query(
                f"SELECT * FROM candidates WHERE snr >= ? ORDER BY {order} LIMIT ?",
                (min_snr, _EXPORT_ROW_CAP + 1))
        else:
            rows = db._query(
                f"SELECT * FROM candidates ORDER BY {order} LIMIT ?",
                (_EXPORT_ROW_CAP + 1,))
        truncated = len(rows) > _EXPORT_ROW_CAP
        return rows[:_EXPORT_ROW_CAP], truncated

    @app.get("/api/export/csv")
    def api_export_csv(min_snr: Optional[float] = Query(None)):
        """Export candidates as CSV.

        Contract (P2, 2026-07-02): capped at _EXPORT_ROW_CAP (50,000) rows,
        ordered by snr DESC (highest-SNR rows kept on truncation). If the
        true result exceeds the cap, the response header
        ``X-Export-Truncated: true`` is set — request a higher min_snr to
        bring the true result under the cap for a complete export.
        """
        import csv
        import io
        from fastapi.responses import StreamingResponse

        candidates, truncated = _bounded_candidate_export(min_snr)
        if not candidates:
            return JSONResponse({"error": "No candidates"}, status_code=404)

        buf = io.StringIO()
        fields = ["tic_id", "candidate_num", "period_days", "t0",
                  "duration_days", "depth_ppm", "snr", "cnn_score",
                  "triceratops_fpp", "claude_verdict", "verdict_proba"]
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for c in candidates:
            writer.writerow(dict(c))

        buf.seek(0)
        headers = {"Content-Disposition": "attachment; filename=exonit_candidates.csv"}
        if truncated:
            headers["X-Export-Truncated"] = "true"
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers=headers,
        )

    @app.get("/api/export/json")
    def api_export_json(min_snr: Optional[float] = Query(None)):
        """Export candidates + stars as JSON.

        Contract (P2, 2026-07-02): both candidates and stars are capped at
        _EXPORT_ROW_CAP (50,000) rows each — previously both ran fully
        unbounded (stars alone is multi-million rows) on every call.
        ``total_stars``/``total_candidates`` report the (possibly capped)
        counts actually included in this response, not the true DB totals;
        ``truncated_stars``/``truncated_candidates`` flag whether either cap
        was hit.
        """
        candidates, cand_truncated = _bounded_candidate_export(min_snr)
        stars = db._query(
            "SELECT * FROM stars ORDER BY analyzed_at DESC LIMIT ?",
            (_EXPORT_ROW_CAP + 1,))
        stars_truncated = len(stars) > _EXPORT_ROW_CAP
        stars = stars[:_EXPORT_ROW_CAP]
        return {
            "export_date": __import__("datetime").datetime.now().isoformat(),
            "total_stars": len(stars),
            "total_candidates": len(candidates),
            "stars": [dict(s) for s in stars],
            "candidates": [dict(c) for c in candidates],
            "truncated_stars": stars_truncated,
            "truncated_candidates": cand_truncated,
        }

    @app.get("/api/completeness")
    def api_completeness():
        """Aggregated completeness heatmap: avg detection probability per (radius, period) cell.

        Reads from completeness_agg (incremental aggregate) instead of doing a
        full GROUP BY over the completeness table (25M+ rows → 34 s → < 1 ms).
        completeness_agg is kept in exact sync with completeness by triggers on
        that table plus the ON CONFLICT DO UPDATE writers in ExoNitDB.
        Response format is identical to the previous query.

        Routed through db.query_completeness_agg() rather than db._query(): the
        completeness subsystem stays on the dedicated SQLite connection in PG
        mode too (the generic backend would read the empty PG table otherwise).
        """
        return db.query_completeness_agg()

    # --- cTOI / ExoFOP endpoints ---

    @app.get("/api/ctoi/plot/{tic_id}/{candidate_num}")
    def ctoi_plot(tic_id: str, candidate_num: int):
        """Generate phase-folded transit plot for a candidate on demand."""
        row = db._query(
            """SELECT * FROM candidates
               WHERE tic_id = ? AND candidate_num = ?""",
            (tic_id, candidate_num))
        if not row:
            return JSONResponse({"error": "Candidate not found"}, status_code=404)
        candidate = dict(row[0])

        from exonit.ctoi import load_light_curve, generate_phase_fold_plot
        time, flux = load_light_curve(tic_id, cfg.metadata_cache_path)
        if time is None:
            return JSONResponse({"error": "No light curve data"}, status_code=404)

        png_bytes = generate_phase_fold_plot(tic_id, candidate, time, flux)
        from fastapi.responses import Response
        return Response(content=png_bytes, media_type="image/png")

    @app.get("/api/ctoi/crosscheck/{tic_id}")
    def ctoi_crosscheck(tic_id: str):
        """Cross-check a TIC against the TOI catalog."""
        candidates = db._query(
            "SELECT * FROM candidates WHERE tic_id = ? ORDER BY candidate_num",
            (tic_id,))
        if not candidates:
            return JSONResponse({"error": "No candidates"}, status_code=404)

        from exonit.ctoi import crosscheck_toi
        results = []
        for c in candidates:
            c = dict(c)
            matches = crosscheck_toi(tic_id, c['period_days'], cfg.catalog_path)
            results.append({
                "candidate_num": c['candidate_num'],
                "period_days": c['period_days'],
                "snr": c['snr'],
                "toi_matches": matches,
            })
        return {"tic_id": tic_id, "candidates": results}

    @app.get("/api/ctoi/export")
    def ctoi_export(min_snr: float = 10.0, min_cnn: float = 0.7,
                    max_fpp: float = 0.5, exclude_known: bool = True):
        """Export new cTOI candidates as CSV for ExoFOP submission."""
        from exonit.ctoi import export_exofop_csv
        csv_data, n_new = export_exofop_csv(
            db.backend, cfg.catalog_path,
            min_snr=min_snr, min_cnn=min_cnn,
            max_fpp=max_fpp, exclude_known=exclude_known)
        from fastapi.responses import Response
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=exonit_ctoi_{datetime.now().strftime('%Y%m%d')}.csv",
                "X-Total-Candidates": str(n_new),
            })

    # --- Download manager endpoints ---

    @app.get("/api/downloads/status")
    def downloads_status():
        """Overview of FITS cache and download status."""
        cache_stats = {"total_stars": 0, "total_fits": 0}
        stellar_count = 0
        if os.path.exists(cfg.metadata_cache_path):
            cache = _server._get_shared_metadata_cache(cfg)
            cache_stats = cache.stats()
            # Count stars with stellar params
            all_ids = cache.get_all_tic_ids()
            for tic in all_ids[:100]:  # sample
                e = cache.get_entry(tic)
                if e and e.get("stellar_radius"):
                    stellar_count += 1

        # Check NFS mount
        import shutil
        nfs_path = cfg.nfs_cache_path
        nfs_available = os.path.isdir(nfs_path)
        nfs_usage = {}
        if nfs_available:
            usage = shutil.disk_usage(nfs_path)
            nfs_usage = {
                "total_gb": round(usage.total / 1e9, 1),
                "used_gb": round(usage.used / 1e9, 1),
                "free_gb": round(usage.free / 1e9, 1),
            }

        # DB stats
        db_stars = db.get_stats()

        return {
            "cache": cache_stats,
            "stellar_params_sample": f"{stellar_count}/100",
            "nfs": {"available": nfs_available, **nfs_usage},
            "db": db_stars,
            "catalogs": _get_catalog_info(),
        }

    def _get_catalog_info():
        """Get info about available catalogs."""
        catalog_path = str(_PROJECT_ROOT / "data" / "toi_catalog.csv")
        result = {"toi": {"available": False}}
        if os.path.exists(catalog_path):
            import csv
            with open(catalog_path) as f:
                reader = csv.reader(f)
                lines = sum(1 for _ in reader) - 1
            result["toi"] = {
                "available": True,
                "path": catalog_path,
                "entries": lines,
                "size_mb": round(os.path.getsize(catalog_path) / 1e6, 1),
            }
        return result

    @app.get("/api/downloads/catalog")
    def downloads_catalog(
        limit: int = Query(default=50),
        offset: int = Query(default=0),
        filter: str = Query(default="all"),
    ):
        """List catalog entries with their download/analysis status.

        Perf (P7, docs/reviews/2026-07-02-opus48-cleanup-review.md): the CSV
        used to be re-parsed on every request; it's now cached in-process by
        file mtime (see _load_downloads_catalog_rows, same pattern as
        exonit.ctoi._load_toi_catalog). The per-row has_fits / is_analyzed
        checks — previously 2 DB round trips PER catalog row (N+1, ~8k CSV
        rows) BEFORE pagination — are now 2 bulk lookups: has_fits via
        MetadataCache.all_tic_ids() (the store's own established
        O(1)-per-target index, already used by enqueue_targets for the same
        purpose) and is_analyzed via one chunked ``tic_id IN (...)`` query
        (db_backend.select_in_chunks).
        """
        catalog_path = _server._toi_catalog_path()
        rows = _load_downloads_catalog_rows(catalog_path)
        if not rows:
            return {"entries": [], "total": 0}

        fits_tic_ids = set()
        if os.path.exists(cfg.metadata_cache_path):
            fits_tic_ids = _server._get_shared_metadata_cache(cfg).all_tic_ids()

        analyzed_tic_ids = {
            str(r["tic_id"]) for r in select_in_chunks(
                db.backend.query,
                "SELECT tic_id FROM stars WHERE status = 'completed' AND tic_id IN",
                [r["tic_id"] for r in rows])
        }

        entries = []
        for row in rows:
            tic = row["tic_id"]
            has_fits = tic in fits_tic_ids
            is_analyzed = tic in analyzed_tic_ids

            status = "pending"
            if is_analyzed:
                status = "analyzed"
            elif has_fits:
                status = "downloaded"

            if filter == "pending" and status != "pending":
                continue
            elif filter == "downloaded" and status != "downloaded":
                continue
            elif filter == "analyzed" and status != "analyzed":
                continue

            entries.append({
                "tic_id": tic,
                "disposition": row["disposition"],
                "tmag": row["tmag"],
                "period": row["period"],
                "status": status,
                "has_fits": has_fits,
            })

        total = len(entries)
        n_analyzed = sum(1 for e in entries if e["status"] == "analyzed")
        n_downloaded = sum(1 for e in entries if e["status"] == "downloaded")
        n_pending = sum(1 for e in entries if e["status"] == "pending")
        page = entries[offset:offset + limit]
        return {
            "entries": page, "total": total, "offset": offset, "limit": limit,
            "counts": {"analyzed": n_analyzed, "downloaded": n_downloaded, "pending": n_pending},
        }

    @app.post("/api/downloads/refresh-catalog")
    async def downloads_refresh_catalog():
        """Download/update the TOI catalog.

        The HTTP download is blocking, so it runs in a worker thread via
        asyncio.to_thread — otherwise it would freeze the whole event loop
        (every worker heartbeat / task assignment / poll stalls) for the
        duration of the network fetch.
        """
        from exonit.targets import download_toi_catalog
        catalog_path = str(_PROJECT_ROOT / "data" / "toi_catalog.csv")
        try:
            await asyncio.to_thread(download_toi_catalog, catalog_path)
            return {"status": "ok", "message": "Catalog updated"}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/downloads/rebuild-cache")
    async def downloads_rebuild_cache():
        """Rebuild metadata cache from NFS FITS files.

        build_from_nfs walks the NFS tree (can take a long time), so it runs in
        a worker thread via asyncio.to_thread to keep the event loop responsive.
        """
        from exonit.metadata_cache import MetadataCache, build_from_nfs

        def _rebuild():
            cache = MetadataCache(cfg.metadata_cache_path)
            try:
                return build_from_nfs(cache, cfg.nfs_cache_path)
            finally:
                cache.close()

        stats = await asyncio.to_thread(_rebuild)
        return {"status": "ok", **stats}

    def _scorecard_date(name: str, path: Path) -> str:
        """Extract YYYY-MM-DD from filename like foo-bar-2026-05-27.json, else mtime."""
        import re as _re
        m = _re.search(r"(\d{4}-\d{2}-\d{2})", name)
        if m:
            return m.group(1)
        try:
            import datetime
            mtime = path.stat().st_mtime
            return datetime.date.fromtimestamp(mtime).isoformat()
        except OSError:
            return ""

    # Stable top-level keys extracted from scorecard JSON as summary.
    # Only keys that actually exist in the real files — verified against
    # discovery_s0001_lotN and baseline-100 scorecards.
    _SCORECARD_SUMMARY_KEYS = (
        # discovery-style scorecards
        "campaign", "sector", "lot", "date", "n_stars", "stars_processed",
        "primary_done", "primary_failed", "candidate_rows",
        "distinct_tics_with_candidate", "v6_ge_0.5", "clean_planets",
        # baseline-style scorecards
        "generated_at", "sample", "n_tics",
    )

    @app.get("/api/scorecards")
    def api_scorecards():
        """List data/scorecards/*.json, newest first.

        Returns [{name, date, summary}, ...] where summary contains only
        stable top-level keys that exist in the JSON (no invented fields).
        Malformed files are skipped with a log note.
        """
        sc_dir = _server._scorecards_dir()
        if not sc_dir.is_dir():
            return []

        def _sc_sort_key(p: Path):
            # Newest first. Primary key = date parsed from the filename (stable
            # across git clones); tie-break = mtime so same-day runs order by
            # generation time. Sorting by name alone is wrong: the date is at
            # the END of the name and the prefix (lot4, lot20260529T…, lotX)
            # varies, so name order != date order.
            try:
                mtime = p.stat().st_mtime
            except OSError:
                mtime = 0.0
            return (_scorecard_date(p.name, p), mtime)

        results = []
        for p in sorted(sc_dir.glob("*.json"), key=_sc_sort_key, reverse=True):
            date = _scorecard_date(p.name, p)
            summary = {}
            try:
                with open(p, encoding="utf-8") as fh:
                    data = json.load(fh)
                for k in _SCORECARD_SUMMARY_KEYS:
                    if k in data:
                        summary[k] = data[k]
            except Exception as exc:
                logger.info("scorecards: skipping %s: %s", p.name, exc)
                continue
            results.append({"name": p.name, "date": date, "summary": summary})

        return results

    @app.get("/api/scorecards/{name}")
    def api_scorecard_get(name: str):
        """Serve a single scorecard JSON by filename.

        Validates that name contains no path separators or traversal sequences.
        Returns 404 if missing, 400 if the name is unsafe.
        """
        # Safety: reject traversal and any path separator
        if ".." in name or "/" in name or "\\" in name or not name.endswith(".json"):
            return JSONResponse({"error": "invalid filename"}, status_code=400)
        p = _server._scorecards_dir() / name
        if not p.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("scorecard read error %s: %s", name, exc)
            return JSONResponse({"error": "read error"}, status_code=500)

    @app.get("/api/scorecards/{name}/dossier")
    def api_scorecard_dossier(name: str):
        """Serve the sibling vetting dossier (``<base>_dossier.md``) for a run.

        The scorecard JSON only carries aggregate quality metrics; a run's
        *strong candidates* (potential discoveries) live in the dossier. Returns
        ``{"text": ...}`` or 404 when the run produced no dossier.
        """
        if ".." in name or "/" in name or "\\" in name or not name.endswith(".json"):
            return JSONResponse({"error": "invalid filename"}, status_code=400)
        p = _server._scorecards_dir() / (name[:-len(".json")] + "_dossier.md")
        if not p.exists():
            return JSONResponse({"error": "no dossier"}, status_code=404)
        try:
            return {"text": p.read_text(encoding="utf-8")}
        except Exception as exc:
            logger.warning("dossier read error %s: %s", name, exc)
            return JSONResponse({"error": "read error"}, status_code=500)

