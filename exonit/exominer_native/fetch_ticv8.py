"""TICv8 stellar parameters fetch + cache for /api/ticv8.

Cache entries never expire (TIC catalog is static). Fetched via
astroquery.mast.Catalogs.query_criteria(catalog='TIC').
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ticv8_cache (
    tic_id     TEXT PRIMARY KEY,
    data_json  TEXT NOT NULL,
    fetched_at TEXT DEFAULT (datetime('now'))
)
"""


class TICv8Store:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_CREATE_SQL)
        self._conn.commit()

    def get(self, tic_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT data_json FROM ticv8_cache WHERE tic_id=?",
                (str(tic_id),),
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, tic_id: str, data: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO ticv8_cache (tic_id, data_json) VALUES (?, ?)",
                (str(tic_id), json.dumps(data)),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def fetch_from_mast(tic_id: str) -> Optional[dict]:
    """Hit MAST TIC catalog. Return {} if found (even empty), None if no row."""
    from astroquery.mast import Catalogs

    rows = Catalogs.query_criteria(catalog="TIC", ID=str(tic_id))
    if len(rows) == 0:
        return None
    r = rows[0]
    keys = (
        "Tmag", "Teff", "rad", "mass", "logg", "rho", "MH",
        "ra", "dec", "contratio",
    )
    out = {}
    for k in keys:
        if k in r.colnames:
            v = r[k]
            if hasattr(v, "mask") and getattr(v, "mask", False):
                out[k] = None
            else:
                out[k] = float(v) if hasattr(v, "item") or isinstance(v, (int, float)) else str(v)
    return out
