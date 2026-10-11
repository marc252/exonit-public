"""HTTP client for the server's Phase 2.5 FFI endpoint.

Returns (time, flux, quality) as concatenated numpy arrays sorted by
time, or None on any error (404, timeout, network). The caller must
treat None as "FFI unavailable; skip Phase 3".
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import requests

LOG = logging.getLogger(__name__)


def fetch_ffi(
    server_url: str, tic_id: str, *, token: str, timeout: float = 30.0
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """GET /api/ffi/{tic_id}; concatenate sectors, return (t, f, q).

    Authenticates via Authorization: Bearer <token>. Returns None on
    any failure (HTTP error, timeout, malformed payload).
    """
    url = f"{server_url.rstrip('/')}/api/ffi/{tic_id}"
    try:
        r = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
    except (requests.exceptions.RequestException, OSError) as exc:
        LOG.warning("ffi fetch failed for tic %s: %s", tic_id, exc)
        return None
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        LOG.warning("ffi fetch HTTP %d for tic %s", r.status_code, tic_id)
        return None
    try:
        payload = r.json()
        sectors = payload["sectors"]
    except (ValueError, KeyError) as exc:
        LOG.warning("ffi response malformed for tic %s: %s", tic_id, exc)
        return None
    times, fluxes, quals = [], [], []
    try:
        for s in sectors:
            times.append(np.asarray(s["time"], dtype=float))
            fluxes.append(np.asarray(s["flux"], dtype=float))
            quals.append(
                np.asarray(s.get("quality", [0] * len(s["time"])), dtype=int)
            )
    except (KeyError, TypeError, ValueError) as exc:
        LOG.warning("ffi sector data malformed for tic %s: %s", tic_id, exc)
        return None
    if not times:
        return None
    t = np.concatenate(times)
    f = np.concatenate(fluxes)
    q = np.concatenate(quals)
    order = np.argsort(t)
    return t[order], f[order], q[order]
