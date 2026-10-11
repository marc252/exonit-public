"""First-run setup — downloads TOI catalog, metadata, and FITS from MAST.

Runs in background threads so it doesn't block the server.
Progress is tracked via a shared state dict that the API can query.
"""

import json
import os
import sqlite3
import threading
import time

setup_state = {
    "running": False,
    "phase": "idle",       # idle, catalog, metadata, fits, done, error
    "progress": 0,
    "total": 0,
    "current_tic": "",
    "message": "",
    "fits_dir": "",
    "errors": 0,
    "started_at": None,
}

_lock = threading.Lock()


def get_state():
    with _lock:
        s = dict(setup_state)
        if s["started_at"]:
            s["elapsed_seconds"] = int(time.time() - s["started_at"])
        return s


def _update(phase=None, progress=None, total=None, tic=None, msg=None):
    with _lock:
        if phase:
            setup_state["phase"] = phase
        if progress is not None:
            setup_state["progress"] = progress
        if total is not None:
            setup_state["total"] = total
        if tic is not None:
            setup_state["current_tic"] = tic
        if msg:
            setup_state["message"] = msg


def is_first_run(metadata_cache_path, catalog_path):
    """Check if this is a fresh install (no catalog or no metadata)."""
    has_catalog = os.path.exists(catalog_path)
    has_cache = os.path.exists(metadata_cache_path)
    if has_cache:
        try:
            conn = sqlite3.connect(metadata_cache_path)
            count = conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
            conn.close()
            if count > 0:
                return False
        except Exception:
            pass
    return not has_catalog or not has_cache


def start_setup(catalog_path, metadata_cache_path, fits_dir, n_stars=100):
    """Start the setup process in a background thread."""
    with _lock:
        if setup_state["running"]:
            return False
        setup_state["running"] = True
        setup_state["phase"] = "starting"
        setup_state["progress"] = 0
        setup_state["total"] = 0
        setup_state["errors"] = 0
        setup_state["fits_dir"] = fits_dir
        setup_state["started_at"] = time.time()

    t = threading.Thread(
        target=_run_setup,
        args=(catalog_path, metadata_cache_path, fits_dir, n_stars),
        daemon=True)
    t.start()
    return True


def _run_setup(catalog_path, metadata_cache_path, fits_dir, n_stars):
    """Main setup flow: catalog → metadata → FITS."""
    try:
        os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
        os.makedirs(fits_dir, exist_ok=True)

        # Phase 1: Download TOI catalog
        _update(phase="catalog", msg="Downloading TOI catalog from ExoFOP...")
        from exonit.targets import download_toi_catalog, parse_toi_catalog
        download_toi_catalog(catalog_path)
        targets = parse_toi_catalog(catalog_path)
        _update(msg=f"Catalog downloaded: {len(targets)} stars")

        # Select stars to process (brightest first, up to n_stars)
        targets.sort(key=lambda t: t.get("tmag", 99))
        selected = targets[:n_stars]
        _update(total=len(selected))

        # Phase 2: Build metadata cache + download FITS
        _update(phase="metadata", msg="Querying MAST for star data and downloading FITS...")
        from exonit.metadata_cache import MetadataCache

        cache = MetadataCache(metadata_cache_path)

        import warnings
        warnings.filterwarnings('ignore')

        for i, target in enumerate(selected):
            tic_id = target["tic_id"]
            _update(progress=i + 1, tic=tic_id,
                    msg=f"[{i+1}/{len(selected)}] TIC {tic_id}")

            # Skip if already in cache with FITS
            existing = cache.get_entry(str(tic_id))
            if existing and existing.get("fits_files"):
                # Check if FITS exist on disk
                has_local = any(
                    os.path.exists(os.path.join(fits_dir, f))
                    or (cache.get_fits_path(f) and os.path.exists(cache.get_fits_path(f)))
                    for f in existing["fits_files"]
                )
                if has_local:
                    continue

            try:
                _download_star(tic_id, cache, fits_dir)
            except Exception as e:
                print(f"  Setup: failed TIC {tic_id}: {e}")
                with _lock:
                    setup_state["errors"] += 1

            # Rate limit MAST queries
            time.sleep(0.5)

        cache.close()

        _update(phase="done", msg=f"Setup complete: {len(selected)} stars ready")

    except Exception as e:
        _update(phase="error", msg=f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        with _lock:
            setup_state["running"] = False


def _download_star(tic_id, cache, fits_dir):
    """Download metadata + FITS for one star from MAST."""
    import lightkurve as lk

    # Search MAST for this star
    search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
    if search is None or len(search) == 0:
        # Try QLP
        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="QLP")
    if search is None or len(search) == 0:
        return

    # Filter to exact TIC
    if hasattr(search, 'target_name'):
        exact = search[search.target_name == str(tic_id)]
        if len(exact) > 0:
            search = exact

    # Prefer 120s cadence
    search_120 = search[search.exptime.value == 120]
    if len(search_120) == 0:
        search_120 = search

    # Limit sectors
    n = min(20, len(search_120))
    search_120 = search_120[:n]

    # Extract sector info
    import re
    sectors = []
    for r in search_120:
        m = re.search(r'Sector (\d+)', str(r.mission))
        if m:
            sectors.append(int(m.group(1)))
    sectors = sorted(set(sectors))

    # Download FITS
    fits_files = []
    for item in search_120:
        try:
            lc = item.download(download_dir=fits_dir)
            # Find the downloaded file
            fname = None
            if hasattr(lc, 'filename') and lc.filename:
                fname = os.path.basename(str(lc.filename))
            elif hasattr(lc, 'meta') and 'FILENAME' in lc.meta:
                fname = os.path.basename(lc.meta['FILENAME'])

            if fname:
                fits_files.append(fname)
                # Index the path
                local_path = os.path.join(fits_dir, "mastDownload", "TESS")
                for root, dirs, files in os.walk(local_path):
                    for f in files:
                        if f == fname:
                            cache.add_fits_path(f, os.path.join(root, f))
        except Exception:
            continue

    if not fits_files:
        return

    # Extract stellar params from first FITS
    stellar_radius = None
    teff = None
    logg = None
    tmag = None
    ra = None
    dec = None

    try:
        from astropy.io import fits as pf
        first_fits = cache.get_fits_path(fits_files[0])
        if first_fits and os.path.exists(first_fits):
            with pf.open(first_fits) as hdul:
                hdr = hdul[0].header
                stellar_radius = hdr.get('RADIUS')
                teff = hdr.get('TEFF')
                logg = hdr.get('LOGG')
                tmag = hdr.get('TESSMAG')
                ra = hdr.get('RA_OBJ')
                dec = hdr.get('DEC_OBJ')
    except Exception:
        pass

    # Determine exptime
    exptime = 120.0
    try:
        exptime = float(search_120[0].exptime.value)
    except Exception:
        pass

    # Save to metadata cache
    cache.add_entry(
        tic_id=str(tic_id),
        sectors=sectors,
        exptime=exptime,
        fits_files=fits_files,
        stellar_radius=stellar_radius,
        teff=teff,
        logg=logg,
        tmag=tmag,
        ra=ra,
        dec=dec,
    )
