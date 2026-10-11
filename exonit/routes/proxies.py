"""External-data proxy endpoints (TICv8/Gaia/MAST-DV/FFI/FITS/ExoMiner) (Fase D split).

Extracted verbatim from exonit.server.create_app() — see
docs/reviews/2026-07-02-opus48-cleanup-review.md and
progress/impl-createapp-split-2026-07-02.md for the module map. Endpoint
bodies below are byte-identical to the pre-split code (git move-detection
should match them); only this file's import/register() preamble is new.

Covers: /api/exominer/tfrecord/{tic_id}, /api/ffi/{tic_id},
/api/metadata/{tic_id}, /api/nearby/{tic_id}, /api/mast/dv_runs/{tic_id},
/api/mast/dv/{tic_id}, /api/ticv8/{tic_id}, /api/gaia/{tic_id},
/api/fits/{filename}, /api/exominer/onnx.

Workers must never hit MAST/Gaia/DV directly — the server is the sole proxy
(design decision D4, see docstrings below).
"""

import os

from fastapi import Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response

from exonit import server as _server
from exonit.server import (
    _PROJECT_ROOT,
    _ProxyNotFound,
    _get_preprocess_executor,
    _json_safe,
    _object_store,
    _proxy_fetch,
    derive_mass_from_logg_rad,
    logger,
)
from exonit.routes.deps import ServerDeps

# NOTE: get_dv_store / get_ticv8_store / get_gaia_store / _get_shared_metadata_cache
# / _ONNX_MODEL_PATH / _load_ffi_for_tic are called as `_server.<name>` (not
# plain-imported) below — several tests patch them via
# patch("exonit.server.<name>", ...) AFTER the app (and this module's
# register()) has already run; a `from exonit.server import <name>` binding
# taken at register()-time would freeze the pre-patch reference. Looking the
# name up on the `_server` module object at call time picks up the patch,
# exactly matching the pre-split behaviour (a bare name inside create_app()
# resolved via the enclosing module's __dict__ at call time too).


def register(app, deps: ServerDeps) -> None:
    db = deps.db
    cfg = deps.cfg
    berger = deps.berger
    _check_auth = deps.check_auth

    @app.get("/api/exominer/tfrecord/{tic_id}")
    def exominer_tfrecord(tic_id: str, request: Request):
        """Serve the cached ExoMiner TFRecord for ``tic_id`` so the worker
        can run inference locally with its isolated TF venv.

        Flow:
          - Cache hit → 200 FileResponse + X-Sector-Run header.
          - Cache miss → 202 + kick off preprocessor in a background thread
            so the next worker visit to this TIC will hit the cache.
          - TIC has no DV runs at MAST → 404.
        """
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        try:
            from exonit.exominer_preprocess import ExoMinerPreprocessor
            from exonit.mast_dv import MastDVCache, fetch_dv_runs, pick_best_run
        except Exception as e:
            return JSONResponse({"error": f"exominer modules unavailable: {e}"},
                                status_code=500)

        tic_id = str(tic_id)
        pre = ExoMinerPreprocessor()

        # Resolve sector_run without hitting MAST if cached
        mast_cache = MastDVCache()
        try:
            runs = fetch_dv_runs(tic_id, cache=mast_cache)
        except Exception as e:
            mast_cache.close()
            return JSONResponse({"error": f"mast query failed: {e}"}, status_code=502)
        finally:
            mast_cache.close()

        if not runs:
            return JSONResponse({"error": "no_dv_runs"}, status_code=404)

        sector_run = pick_best_run(runs)
        if pre.is_cached(tic_id, sector_run):
            fp = pre._tfrecord_dest(tic_id, sector_run)
            return FileResponse(
                str(fp),
                media_type="application/octet-stream",
                headers={"X-Sector-Run": sector_run, "X-Tic-Id": tic_id},
            )

        # Cache miss — kick off preprocessor in background so subsequent
        # polls succeed. Do NOT block this request on the 3–5 min preprocess.
        # Bounded executor (minor dedup 11(e)) instead of a raw per-request
        # threading.Thread — see _get_preprocess_executor.
        def _kick_preprocess():
            try:
                ExoMinerPreprocessor().preprocess_tic(tic_id, sector_run=sector_run)
            except Exception as exc:  # noqa: BLE001 — background thread
                logger.warning("ExoMiner preprocess failed TIC %s: %s", tic_id, exc)

        _get_preprocess_executor().submit(_kick_preprocess)
        return JSONResponse(
            {"status": "preprocessing", "tic_id": tic_id, "sector_run": sector_run},
            status_code=202,
        )

    # --- Data endpoints ---

    @app.get("/api/ffi/{tic_id}")
    def get_ffi(tic_id: str, request: Request):
        """Return Phase 2.5 FFI light curves for the given TIC.

        Workers call this endpoint to obtain the FFI flux data needed for
        Phase 3 ExoMiner-FFI inference. The server is the sole proxy to the
        local HDF5 cache; workers must NOT fetch from MAST directly.

        Returns 200 + {tic_id, sectors} on cache hit, 404 on miss.
        """
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        sectors = _server._load_ffi_for_tic(tic_id)
        if not sectors:
            return JSONResponse(
                {"error": f"no FFI cache for TIC {tic_id}"}, status_code=404
            )
        return {"tic_id": tic_id, "sectors": sectors}

    @app.get("/api/metadata/{tic_id}")
    def get_metadata(tic_id: str, request: Request):
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if not os.path.exists(cfg.metadata_cache_path):
            return JSONResponse({"error": "No metadata cache"}, status_code=404)
        cache = _server._get_shared_metadata_cache(cfg)
        entry = cache.get_entry(tic_id)
        if not entry:
            return JSONResponse({"error": "Not found"}, status_code=404)
        return entry

    @app.get("/api/nearby/{tic_id}")
    def get_nearby(tic_id: str, request: Request):
        """Return cached nearby TIC stars for FPP. If not cached,
        query MAST on-demand, cache the result, and return it."""
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if not os.path.exists(cfg.metadata_cache_path):
            return JSONResponse({"error": "No metadata cache"}, status_code=404)
        cache = _server._get_shared_metadata_cache(cfg)

        # Minor dedup 11(a): the inline NaN/Inf sanitizer here was semantically
        # identical to the module-level _json_safe (~line 585) except it did
        # not recurse into tuples — a strict subset with no observable
        # difference for this endpoint's actual data shapes (lists of star
        # dicts, no tuples). Use the shared one.

        def _overlay_berger(star_list):
            """Override mass/rad/Teff with Berger+2023 values when the TIC is
            in the catalog. Applied to both cache-hit and MAST-fetched paths
            so prior cache entries receive homogeneous parameters too.
            Returns count of stars overlaid."""
            n = 0
            for s in star_list:
                try:
                    bg = berger.lookup_by_tic(int(s['ID']))
                except (ValueError, TypeError):
                    continue
                if bg is None:
                    continue
                s['mass'] = bg['mass']
                s['rad'] = bg['rad']
                s['Teff'] = bg['teff']
                s['stellar_source'] = 'Berger+2023'
                n += 1
            return n

        # 1. Try cache
        cached = cache.get_nearby_stars(tic_id)
        if cached:
            stars, gal_b = cached
            _overlay_berger(stars)
            return _json_safe(
                {"tic_id": tic_id, "stars": stars, "gal_b": gal_b,
                 "source": "cache"})

        # 2. Get RA/Dec from metadata cache
        entry = cache.get_entry(tic_id)
        if not entry or not entry.get("ra") or not entry.get("dec"):
            return JSONResponse(
                {"error": f"No coordinates for TIC {tic_id}"},
                status_code=404)

        ra, dec = entry["ra"], entry["dec"]

        # 3. Query MAST (with timeout)
        import numpy as np
        try:
            from astroquery.mast import Catalogs
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            Catalogs.TIMEOUT = 30
            Catalogs.PAGESIZE = 50
            coord = SkyCoord(ra=ra, dec=dec, unit="deg")
            nearby = Catalogs.query_region(
                coord, radius=60 * u.arcsec, catalog="TIC")

            stars = []
            n_mass_derived = 0
            for r in nearby:
                try:
                    s = {
                        'ID': str(r['ID']),
                        'ra': float(r['ra']),
                        'dec': float(r['dec']),
                        'mass': float(r['mass']) if r['mass'] else None,
                        'rad': float(r['rad']) if r['rad'] else None,
                        'Teff': float(r['Teff']) if r['Teff'] else None,
                        'Tmag': float(r['Tmag']) if r['Tmag'] else None,
                        'plx': float(r['plx']) if r['plx'] else None,
                    }
                    if s['mass'] is None and s['rad'] is not None:
                        try:
                            logg = float(r['logg']) if r['logg'] else None
                        except (ValueError, TypeError):
                            logg = None
                        derived = derive_mass_from_logg_rad(logg, s['rad'])
                        if derived is not None:
                            s['mass'] = derived
                            n_mass_derived += 1
                    if s['Tmag'] is not None:
                        stars.append(s)
                except (ValueError, TypeError):
                    continue
            n_berger = _overlay_berger(stars)
            if n_berger:
                logger.info(
                    "TIC %s: Berger+2023 applied to %d/%d nearby stars",
                    tic_id, n_berger, len(stars))
            if n_mass_derived:
                logger.info(
                    "TIC %s: derived mass from logg+rad for %d/%d nearby stars",
                    tic_id, n_mass_derived, len(stars))

            gal_b = float(coord.galactic.b.deg)

            # Save to cache
            # Convert None to NaN for cache storage
            stars_for_cache = []
            for s in stars:
                sc = dict(s)
                for k in ('mass', 'rad', 'Teff', 'Tmag', 'plx'):
                    if sc[k] is None:
                        sc[k] = float('nan')
                stars_for_cache.append(sc)
            cache.add_nearby_stars(tic_id, stars_for_cache, gal_b)

            return _json_safe(
                {"tic_id": tic_id, "stars": stars, "gal_b": gal_b,
                 "source": "mast"})

        except Exception as e:
            return JSONResponse(
                {"error": f"MAST query failed: {e}"},
                status_code=502)

    @app.get("/api/mast/dv_runs/{tic_id}")
    def api_mast_dv_runs(tic_id: str, request: Request):
        """Proxy endpoint: list the DV sector_runs MAST hosts for a TIC.

        Returns JSON ``{"tic_id": str, "sector_runs": list[str]}`` sorted
        newest-first (by end sector DESC). Empty list when MAST has no
        products. Used by the native worker to choose a sector_run
        before calling ``/api/mast/dv/{tic_id}``.
        """
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            _ = int(tic_id)
        except ValueError:
            return JSONResponse({"error": "Invalid tic_id"}, status_code=400)
        try:
            from exonit.exominer_native.fetch_mast_dv import discover_sector_runs
            runs = discover_sector_runs(tic_id)
        except Exception as e:
            logger.warning("dv_runs discovery failed %s: %s", tic_id, e)
            return JSONResponse(
                {"error": "discovery_failed"}, status_code=502)
        return {"tic_id": str(tic_id), "sector_runs": runs}

    @app.get("/api/mast/dv/{tic_id}")
    def api_mast_dv(tic_id: str, request: Request,
                    sector_run: str = Query(...)):
        """Proxy endpoint: serve MAST DV XML bytes for (tic_id, sector_run).

        Cache-first: returns bytes from DVXmlStore if present, "known-missing"
        -> 404, else fetches from MAST, stores, and returns. Part of design
        decision D4 (workers never hit MAST/Gaia directly).
        """
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            _ = int(tic_id)
        except ValueError:
            return JSONResponse({"error": "Invalid tic_id"}, status_code=400)
        store = _server.get_dv_store()

        def _fetch():
            from exonit.exominer_native.fetch_mast_dv import fetch_from_mast
            return fetch_from_mast(tic_id, sector_run)

        return _proxy_fetch(
            store, (tic_id, sector_run), _fetch,
            not_found_error="no_dv_product",
            to_response=lambda data: Response(content=data, media_type="application/xml"),
            mark_missing_fn=store.mark_missing,
            known_missing_value=b"",
            log_label="mast_dv")

    @app.get("/api/ticv8/{tic_id}")
    def api_ticv8(tic_id: str, request: Request):
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            _ = int(tic_id)
        except ValueError:
            return JSONResponse({"error": "Invalid tic_id"}, status_code=400)
        store = _server.get_ticv8_store()

        def _fetch():
            from exonit.exominer_native.fetch_ticv8 import fetch_from_mast
            return fetch_from_mast(tic_id)

        return _proxy_fetch(
            store, (tic_id,), _fetch,
            not_found_error="no_tic_row", log_label="ticv8")

    @app.get("/api/gaia/{tic_id}")
    def api_gaia(tic_id: str, request: Request):
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            _ = int(tic_id)
        except ValueError:
            return JSONResponse({"error": "Invalid tic_id"}, status_code=400)
        store = _server.get_gaia_store()

        def _fetch():
            # RA/Dec precondition — checked lazily so it only runs on a cache
            # miss, matching the endpoint's original behaviour exactly (a
            # cached Gaia entry never re-triggers the ticv8 lookup).
            ticv8 = _server.get_ticv8_store().get(tic_id)
            ra = (ticv8 or {}).get("ra") or (ticv8 or {}).get("RA")
            dec = (ticv8 or {}).get("dec") or (ticv8 or {}).get("Dec")
            if ticv8 is None or ra is None or dec is None:
                raise _ProxyNotFound("no_ra_dec")
            from exonit.exominer_native.fetch_gaia import fetch_from_gaia
            return fetch_from_gaia(tic_id, float(ra), float(dec))

        return _proxy_fetch(
            store, (tic_id,), _fetch,
            not_found_error="no_gaia_match", log_label="gaia")

    @app.get("/api/fits/{filename}")
    def serve_fits(filename: str, request: Request):
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if ".." in filename or "/" in filename or not filename.endswith(".fits"):
            return JSONResponse({"error": "Invalid filename"}, status_code=400)
        store = _object_store()
        # 1. Packed container (the primary store): resolve the filename to its
        #    (container, byte-range) in the fits_blobs index and stream just that
        #    member's bytes from object storage via an HTTP Range read. Millions
        #    of tiny FITS live in a handful of tar containers, so this is the
        #    normal path once migrated.
        if store is not None and os.path.exists(cfg.metadata_cache_path):
            try:
                blob = _server._get_shared_metadata_cache(cfg).get_fits_blob(filename)
            except Exception:
                blob = None
            if blob is not None:
                container_key, offset, length = blob
                try:
                    data = store.get_range(container_key, offset, length)
                    return Response(content=data, media_type="application/fits")
                except Exception as e:
                    logger.warning("container range read failed for %s: %s; "
                                   "falling back", filename, e)
        # 2. Individual object (legacy/transition: files uploaded per-file before
        #    packing): 307-redirect to a presigned URL so the worker pulls it
        #    directly from MinIO.
        if store is not None:
            try:
                if store.exists(filename):
                    url = store.presign_get(filename, expires=3600)
                    return RedirectResponse(url=url, status_code=307)
            except Exception as e:
                logger.warning("object-store lookup failed for %s: %s; "
                               "falling back to local", filename, e)
        # Look up the metadata_cache's recorded path first (NFS mastDownload)
        fits_path = None
        if os.path.exists(cfg.metadata_cache_path):
            cache = _server._get_shared_metadata_cache(cfg)
            fits_path = cache.get_fits_path(filename)
        if fits_path and os.path.exists(fits_path):
            return FileResponse(fits_path, media_type="application/fits")
        # Fallback: a flat fits_cache/ under the project root (populated by
        # predownload scripts). Only accept the file if it's structurally
        # valid, so we don't serve leftover truncated downloads.
        fallback_path = _PROJECT_ROOT / "fits_cache" / filename
        if fallback_path.exists():
            from exonit.fits_validator import is_valid_fits
            ok, reason = is_valid_fits(fallback_path)
            if ok:
                return FileResponse(str(fallback_path), media_type="application/fits")
            logger.warning(
                "fits_cache fallback rejected for %s: %s", filename, reason)
        return JSONResponse({"error": "File not found"}, status_code=404)

    @app.get("/api/exominer/onnx")
    def api_exominer_onnx(request: Request):
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if not _server._ONNX_MODEL_PATH.exists():
            return JSONResponse({"error": "onnx_missing"}, status_code=404)
        return FileResponse(str(_server._ONNX_MODEL_PATH), media_type="application/octet-stream")

