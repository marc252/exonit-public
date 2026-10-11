"""MAST DV XML fetch + cache for the /api/mast/dv endpoint.

Cache layout:
  {base_dir}/
    index.sqlite               # tic_id, sector_run -> {status, filename, fetched_at}
    xml/{tic_id}_{sector_run}.xml

Status values: "present" (bytes available), "missing" (confirmed no DV product).
Status entries never expire -- DV products are static once SPOC publishes them.
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Optional


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS dv_xml_index (
    tic_id      TEXT NOT NULL,
    sector_run  TEXT NOT NULL,
    status      TEXT NOT NULL,   -- 'present' | 'missing'
    filename    TEXT,
    fetched_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (tic_id, sector_run)
)
"""

_SECTOR_RE = re.compile(r"^\d+-\d+$")


def _validate_sector_run(sector_run: str) -> None:
    if not _SECTOR_RE.match(sector_run):
        raise ValueError(f"invalid sector_run: {sector_run!r}")


class DVXmlStore:
    """Filesystem + SQLite index for DV XML bytes."""

    def __init__(self, base_dir: str):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        (self.base / "xml").mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self.base / "index.sqlite", check_same_thread=False
        )
        self._conn.execute(_CREATE_SQL)
        self._conn.commit()

    def _xml_path(self, tic_id: str, sector_run: str) -> Path:
        return self.base / "xml" / f"{tic_id}_{sector_run}.xml"

    def get(self, tic_id: str, sector_run: str) -> Optional[bytes]:
        """Return cached XML bytes, empty-bytes sentinel for known-missing,
        or None for a cache miss."""
        _validate_sector_run(sector_run)
        with self._lock:
            row = self._conn.execute(
                "SELECT status, filename FROM dv_xml_index "
                "WHERE tic_id=? AND sector_run=?",
                (str(tic_id), sector_run),
            ).fetchone()
        if row is None:
            return None
        status, _filename = row
        if status != "present":
            return b""  # sentinel for "known missing"
        p = self._xml_path(str(tic_id), sector_run)
        return p.read_bytes() if p.exists() else None

    def put(self, tic_id: str, sector_run: str, xml_bytes: bytes) -> None:
        _validate_sector_run(sector_run)
        p = self._xml_path(str(tic_id), sector_run)
        p.write_bytes(xml_bytes)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO dv_xml_index "
                "(tic_id, sector_run, status, filename) "
                "VALUES (?, ?, 'present', ?)",
                (str(tic_id), sector_run, p.name),
            )
            self._conn.commit()

    def mark_missing(self, tic_id: str, sector_run: str) -> None:
        _validate_sector_run(sector_run)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO dv_xml_index "
                "(tic_id, sector_run, status) VALUES (?, ?, 'missing')",
                (str(tic_id), sector_run),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def discover_sector_runs(tic_id: str) -> list[str]:
    """Return the ``start-end`` sector-run IDs for all DV XML products
    MAST hosts for this TIC, newest first.

    Empty list if MAST has no timeseries observations or no DV products.
    """
    from astroquery.mast import Observations

    obs = Observations.query_criteria(
        target_name=str(tic_id),
        obs_collection="TESS",
        dataproduct_type="timeseries",
    )
    if len(obs) == 0:
        return []
    products = Observations.get_product_list(obs)
    pat = re.compile(
        rf"tess.*-s(\d+)-s(\d+)-0*{tic_id}-.*?_dvr\.xml$"
    )
    runs: set[tuple[int, int]] = set()
    for fn in products["productFilename"]:
        m = pat.search(str(fn))
        if m:
            runs.add((int(m.group(1)), int(m.group(2))))
    # sort by end sector descending (newest first), then span
    return [f"{s}-{e}" for s, e in sorted(runs, key=lambda r: (-r[1], r[0] - r[1]))]


def fetch_from_mast(tic_id: str, sector_run: str) -> Optional[bytes]:
    """Hit MAST for this TIC + sector_run. Return XML bytes or None.

    Uses astroquery + product filename regex to pick the right DV XML.
    """
    from astroquery.mast import Observations

    obs = Observations.query_criteria(
        target_name=str(tic_id),
        obs_collection="TESS",
        dataproduct_type="timeseries",
    )
    if len(obs) == 0:
        return None
    products = Observations.get_product_list(obs)
    # Match pattern: tess...-s{start:04d}-s{end:04d}-{tic:016d}-...-dvr.xml
    start, end = sector_run.split("-")
    pat = re.compile(
        rf"tess.*-s{int(start):04d}-s{int(end):04d}-0*{tic_id}-.*?_dvr\.xml$"
    )
    filenames = products["productFilename"]
    mask = [bool(pat.search(str(fn))) for fn in filenames]
    matched = products[mask]  # type: ignore[index]
    if len(matched) == 0:
        return None
    # Download first match to a temp area and return bytes
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        result = Observations.download_products(
            matched[:1], download_dir=td
        )
        for row in result:
            lp = row.get("Local Path")
            if lp and os.path.exists(str(lp)):
                return Path(str(lp)).read_bytes()
    return None
