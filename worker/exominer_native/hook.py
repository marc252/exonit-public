"""Worker-side ExoMiner-native scoring hook.

Self-contained ExoMiner++ inference at the worker, mirroring the API of
``worker/exominer_client.py::score_candidates`` so the worker main loop
can switch between legacy (server-side TFRecord + subprocess) and native
(in-process preprocess + ONNX) via ``config["exominer_mode"]``.

Pipeline per TIC
----------------
1. Fetch DV XML bytes                → GET /api/mast/dv/{tic}?sector_run=...
   - 404 → return None (TIC has no DV product; skip permanently)
   - 202 / 5xx / network → WARNING + return None (retry next round)
2. Fetch TICv8 stellar params JSON   → GET /api/ticv8/{tic}
3. Fetch Gaia DR3 + RUWE JSON        → GET /api/gaia/{tic}
4. Fetch LC FITS file(s)             → GET /api/metadata/{tic} for filename
                                       list, then GET /api/fits/{fname}
                                       for each (streamed to tempdir).
5. orchestrate.run_tic(...)          → list[TCEFeatures]
6. ExoMinerOnnxInferer(onnx_path).score(tce.feeds) per TCE (session cached
   module-wide, keyed by onnx_path).
7. Match each BLS candidate period to the closest TCE period with
   PERIOD_FACTORS aliases (1, 2, ½, 3, ⅓, 4, ¼ — see exominer_period_match.py),
   tol=2% relative; mutate candidate dict with
   ``exominer_score``, ``exominer_tce_period``, ``exominer_sector_run``.
8. Clean up tempfiles; return summary dict.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .inference import ExoMinerOnnxInferer
from .preprocess import orchestrate as _orchestrate


# Period matching: shared single source of truth for the worker side
# (exominer_period_match.py — the server mirrors it in exonit/exominer_service.py).
from exominer_period_match import (  # noqa: F401
    PERIOD_TOL, PERIOD_FACTORS, match_period as _match_period,
)

# HTTP timeouts (seconds) — short enough that a dead server doesn't stall the
# worker, long enough that cold-cache MAST fetches on the server complete.
_TIMEOUT_JSON = 30
_TIMEOUT_XML = 60
_TIMEOUT_FITS = 180

_log = logging.getLogger(__name__)

# Module-level ONNX session cache. Keyed by (onnx_path, provider-tuple-ish)
# so multiple worker configs in one process would not stomp each other. The
# typical case is exactly one entry.
_ONNX_SESSIONS: Dict[str, ExoMinerOnnxInferer] = {}


# ---------------------------------------------------------------------------
# Period matching
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _discover_sector_runs(
    server_url: str,
    tic_id: str,
    token: str,
    timeout: int = 30,
) -> list[str]:
    """GET /api/mast/dv_runs/{tic_id} → newest-first list of sector_runs.

    Returns empty list on any failure — the caller falls back to the
    per-candidate sector_run or the default "1-36".
    """
    try:
        r = requests.get(
            f"{server_url}/api/mast/dv_runs/{tic_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
        if r.status_code == 200:
            data = r.json()
            return list(data.get("sector_runs", []))
    except Exception:
        pass
    return []


def _fetch_dv_xml(
    server_url: str, tic_id: str, sector_run: str, token: str,
) -> Optional[bytes]:
    """GET /api/mast/dv/{tic_id}?sector_run=... Returns bytes on 200.

    Returns the sentinel ``b""`` on 404 (caller must distinguish skip-permanent
    from transient). Returns ``None`` on 202 / transient error.
    """
    url = f"{server_url.rstrip('/')}/api/mast/dv/{tic_id}"
    try:
        r = requests.get(
            url,
            headers=_headers(token),
            params={"sector_run": sector_run},
            timeout=_TIMEOUT_XML,
        )
    except requests.RequestException as e:
        _log.warning("exominer-native dv_xml %s: request failed: %s", tic_id, e)
        return None
    if r.status_code == 200:
        return r.content
    if r.status_code == 404:
        return b""  # sentinel — caller should return None and skip the TIC
    if r.status_code == 202:
        _log.info("exominer-native dv_xml %s: 202 (server warming cache)", tic_id)
        return None
    _log.warning("exominer-native dv_xml %s: status=%d", tic_id, r.status_code)
    return None


def _fetch_ticv8(
    server_url: str, tic_id: str, token: str,
) -> Optional[Dict[str, Any]]:
    url = f"{server_url.rstrip('/')}/api/ticv8/{tic_id}"
    try:
        r = requests.get(url, headers=_headers(token), timeout=_TIMEOUT_JSON)
    except requests.RequestException as e:
        _log.warning("exominer-native ticv8 %s: request failed: %s", tic_id, e)
        return None
    if r.status_code == 200:
        try:
            return r.json()
        except ValueError:
            _log.warning("exominer-native ticv8 %s: invalid JSON", tic_id)
            return None
    if r.status_code == 404:
        _log.info("exominer-native ticv8 %s: 404 (no TIC row)", tic_id)
        return None
    _log.warning("exominer-native ticv8 %s: status=%d", tic_id, r.status_code)
    return None


def _fetch_gaia(
    server_url: str, tic_id: str, token: str,
) -> Optional[Dict[str, Any]]:
    url = f"{server_url.rstrip('/')}/api/gaia/{tic_id}"
    try:
        r = requests.get(url, headers=_headers(token), timeout=_TIMEOUT_JSON)
    except requests.RequestException as e:
        _log.warning("exominer-native gaia %s: request failed: %s", tic_id, e)
        return None
    if r.status_code == 200:
        try:
            return r.json()
        except ValueError:
            _log.warning("exominer-native gaia %s: invalid JSON", tic_id)
            return None
    if r.status_code == 404:
        _log.info("exominer-native gaia %s: 404 (no Gaia match)", tic_id)
        return None
    _log.warning("exominer-native gaia %s: status=%d", tic_id, r.status_code)
    return None


def _fetch_fits_paths(
    server_url: str,
    tic_id: str,
    token: str,
    sector_run: str,
    dest_dir: Path,
) -> List[Path]:
    """Resolve filenames via /api/metadata/{tic} and download each FITS to
    ``dest_dir``. Returns the list of local paths (may be empty).

    astropy.io.fits demands a real seekable file on disk, so streaming to a
    tempdir is mandatory; we cannot use BytesIO unless we accept a full-file
    in-memory buffer (LC FITS are ~2 MB each so that would work too, but the
    read_tess_light_curve helpers take paths).
    """
    meta_url = f"{server_url.rstrip('/')}/api/metadata/{tic_id}"
    try:
        r = requests.get(meta_url, headers=_headers(token), timeout=_TIMEOUT_JSON)
    except requests.RequestException as e:
        _log.warning("exominer-native metadata %s: request failed: %s", tic_id, e)
        return []
    if r.status_code != 200:
        _log.warning("exominer-native metadata %s: status=%d", tic_id, r.status_code)
        return []
    try:
        meta = r.json()
    except ValueError:
        _log.warning("exominer-native metadata %s: invalid JSON", tic_id)
        return []
    fits_files = meta.get("fits_files", []) or []
    if not fits_files:
        _log.warning("exominer-native metadata %s: empty fits_files list", tic_id)
        return []

    dest_dir.mkdir(parents=True, exist_ok=True)
    local_paths: List[Path] = []
    for fname in fits_files:
        # Basic safety — never trust server-provided filenames to be clean.
        safe = os.path.basename(fname)
        if not safe.endswith(".fits"):
            continue
        local = dest_dir / safe
        url = f"{server_url.rstrip('/')}/api/fits/{safe}"
        try:
            rr = requests.get(
                url,
                headers=_headers(token),
                timeout=_TIMEOUT_FITS,
                stream=True,
            )
        except requests.RequestException as e:
            _log.warning("exominer-native fits %s: %s request failed: %s",
                         tic_id, safe, e)
            continue
        if rr.status_code != 200:
            _log.warning("exominer-native fits %s: %s status=%d",
                         tic_id, safe, rr.status_code)
            continue
        try:
            with open(local, "wb") as f:
                for chunk in rr.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        except OSError as e:
            _log.warning("exominer-native fits %s: %s write failed: %s",
                         tic_id, safe, e)
            continue
        local_paths.append(local)
    return local_paths


# ---------------------------------------------------------------------------
# ONNX session cache
# ---------------------------------------------------------------------------

def _get_inferer(onnx_path: str) -> ExoMinerOnnxInferer:
    """Return a module-cached ``ExoMinerOnnxInferer`` for ``onnx_path``.

    The session is created lazily on first ``.score(...)`` call, so this just
    ensures we don't re-instantiate the wrapper per-TIC.
    """
    inferer = _ONNX_SESSIONS.get(onnx_path)
    if inferer is None:
        inferer = ExoMinerOnnxInferer(onnx_path)
        _ONNX_SESSIONS[onnx_path] = inferer
    return inferer


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def prepare_exominer_inputs(
    tic_id: str,
    candidates: List[Dict[str, Any]],
    *,
    server_url: str,
    token: str,
) -> Optional[Dict[str, Any]]:
    """Download everything ``score_prepared_exominer`` needs to run, with
    no compute on the critical path.

    Returned dict layout::

        {
            "tic_id": str,
            "sector_run": str,
            "dv_xml": bytes,
            "ticv8": dict,
            "gaia": dict,
            "lc_paths": list[Path],
            "tmp_root": Path,        # caller must rmtree when done
        }

    Returns ``None`` on a skip signal (no DV product, no LC FITS,
    transient server error) — exactly the same skip semantics as the
    old monolithic ``score_candidates_native``.

    Pulled out so the worker can run downloads in a background thread
    while it scores the previous TIC on the main thread, halving the
    per-task wallclock once the queue is warm.
    """
    if not candidates:
        return None
    sector_run = str(candidates[0].get("sector_run") or "")
    if not sector_run:
        runs = _discover_sector_runs(server_url, tic_id, token)
        sector_run = runs[0] if runs else "1-36"

    tmp_root = Path(tempfile.mkdtemp(prefix=f"exominer-{tic_id}-"))
    try:
        dv_xml = _fetch_dv_xml(server_url, tic_id, sector_run, token)
        if dv_xml is None:
            shutil.rmtree(tmp_root, ignore_errors=True)
            return None
        if dv_xml == b"":
            _log.info("exominer-native %s: no DV product (404)", tic_id)
            shutil.rmtree(tmp_root, ignore_errors=True)
            return None

        ticv8 = _fetch_ticv8(server_url, tic_id, token) or {}
        gaia = _fetch_gaia(server_url, tic_id, token) or {}

        lc_paths = _fetch_fits_paths(
            server_url, tic_id, token, sector_run, tmp_root / "fits")
        if not lc_paths:
            _log.warning(
                "exominer-native %s: no LC FITS files available", tic_id)
            shutil.rmtree(tmp_root, ignore_errors=True)
            return None
        return {
            "tic_id": tic_id,
            "sector_run": sector_run,
            "dv_xml": dv_xml,
            "ticv8": ticv8,
            "gaia": gaia,
            "lc_paths": lc_paths,
            "tmp_root": tmp_root,
        }
    except Exception:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise


def score_prepared_exominer(
    candidates: List[Dict[str, Any]],
    prepared: Dict[str, Any],
    *,
    onnx_path: str,
) -> Dict[str, Any]:
    """Preprocess + score + match using download outputs from
    ``prepare_exominer_inputs``. Pure compute, no HTTP.

    Mutates each candidate dict in place with ``exominer_score``,
    ``exominer_tce_period``, ``exominer_sector_run`` when a TCE matches.
    Cleans up the scratch dir from ``prepared``.
    """
    tic_id = prepared["tic_id"]
    sector_run = prepared["sector_run"]
    tmp_root = prepared["tmp_root"]
    try:
        try:
            tces = _orchestrate.run_tic(
                tic_id,
                sector_run,
                dv_xml_bytes=prepared["dv_xml"],
                lc_fits_paths=prepared["lc_paths"],
                ticv8_data=prepared["ticv8"],
                gaia_data=prepared["gaia"],
            )
        except Exception as e:  # pylint: disable=broad-except
            _log.error("exominer-native %s: preprocess failed: %s",
                       tic_id, e, exc_info=True)
            return {
                "tic_id": tic_id, "sector_run": sector_run,
                "n_tces": 0, "n_matched": 0,
                "n_candidates": len(candidates), "per_tce_scores": [],
            }
        if not tces:
            _log.info("exominer-native %s: 0 TCEs after preprocess", tic_id)
            return {
                "tic_id": tic_id, "sector_run": sector_run,
                "n_tces": 0, "n_matched": 0,
                "n_candidates": len(candidates), "per_tce_scores": [],
            }

        inferer = _get_inferer(onnx_path)
        per_tce_scores: List[tuple] = []
        score_by_uid: Dict[str, float] = {}
        for tce in tces:
            try:
                s = inferer.score(tce.feeds)
            except Exception as e:  # pylint: disable=broad-except
                _log.error("exominer-native %s: inference failed for %s: %s",
                           tic_id, tce.uid, e, exc_info=True)
                continue
            per_tce_scores.append((tce.uid, float(s)))
            score_by_uid[tce.uid] = float(s)

        if not score_by_uid:
            return {
                "tic_id": tic_id, "sector_run": sector_run,
                "n_tces": len(tces), "n_matched": 0,
                "n_candidates": len(candidates), "per_tce_scores": [],
            }

        scored_tces = [t for t in tces if t.uid in score_by_uid]
        n_matched = 0
        for c in candidates:
            target = c.get("period_days") or c.get("period")
            matched = _match_period(scored_tces, target)
            if matched is None:
                continue
            c["exominer_score"] = score_by_uid[matched.uid]
            c["exominer_tce_period"] = float(matched.period)
            c["exominer_sector_run"] = sector_run
            n_matched += 1

        return {
            "tic_id": tic_id, "sector_run": sector_run,
            "n_tces": len(tces), "n_matched": n_matched,
            "n_candidates": len(candidates), "per_tce_scores": per_tce_scores,
        }
    finally:
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except OSError:
            pass


def score_candidates_native(
    tic_id: str,
    candidates: List[Dict[str, Any]],
    *,
    server_url: str,
    token: str,
    onnx_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Self-contained ExoMiner scoring at the worker (legacy entrypoint).

    Synchronous wrapper that calls ``prepare_exominer_inputs`` then
    ``score_prepared_exominer``. Kept for the worker's fallback path
    (queue empty + sync fetch+score) and for tests that exercise the
    full pipeline in one call.

    Mutates each candidate dict IN PLACE adding:
      - ``exominer_score``        (float, [0, 1])
      - ``exominer_tce_period``   (float, days)
      - ``exominer_sector_run``   (str, e.g. ``"1-36"``)

    Returns a summary dict, or ``None`` if ``prepare_exominer_inputs``
    signaled a skip (no DV, no LC, transient).
    """
    if not candidates:
        return None
    prepared = prepare_exominer_inputs(
        tic_id, candidates, server_url=server_url, token=token)
    if prepared is None:
        return None
    return score_prepared_exominer(
        candidates, prepared, onnx_path=onnx_path)
