"""Gaia DR2/DR3 + RUWE fetch + cache for /api/gaia.

Queries via astroquery.gaia.Gaia. For each TIC we resolve the Gaia source
through RA/Dec cross-match (taken from TICv8 first to avoid a redundant
query — so GaiaStore takes an optional `ticv8_getter` helper).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Callable, Optional


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS gaia_cache (
    tic_id     TEXT PRIMARY KEY,
    data_json  TEXT NOT NULL,
    fetched_at TEXT DEFAULT (datetime('now'))
)
"""


class GaiaStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_CREATE_SQL)
        self._conn.commit()

    def get(self, tic_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT data_json FROM gaia_cache WHERE tic_id=?",
                (str(tic_id),),
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, tic_id: str, data: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO gaia_cache (tic_id, data_json) VALUES (?, ?)",
                (str(tic_id), json.dumps(data)),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def fetch_from_gaia(tic_id: str, ra: float, dec: float, radius_arcsec: float = 3.0) -> Optional[dict]:
    """Cone-search Gaia DR3 at (ra, dec), return nearest source + RUWE."""
    from astroquery.gaia import Gaia

    adql = f"""
        SELECT TOP 1 source_id, ra, dec, parallax, ruwe, phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(POINT('ICRS', ra, dec),
                         CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec / 3600.0}))
        ORDER BY phot_g_mean_mag ASC
    """
    job = Gaia.launch_job(adql)
    rows = job.get_results()
    if len(rows) == 0:
        return None
    r = rows[0]
    return {
        "source_id": str(r["source_id"]),
        "ra": float(r["ra"]),
        "dec": float(r["dec"]),
        "parallax": None if getattr(r["parallax"], "mask", False) else float(r["parallax"]),
        "ruwe": None if getattr(r["ruwe"], "mask", False) else float(r["ruwe"]),
        "phot_g_mean_mag": None if getattr(r["phot_g_mean_mag"], "mask", False) else float(r["phot_g_mean_mag"]),
    }
