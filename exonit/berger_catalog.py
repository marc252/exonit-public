"""Berger+2023 TESS-host stellar properties lookup.

Reads the SQLite produced by `scripts/build_berger_catalog.py` and keeps
its contents in memory (~4180 rows). Designed to degrade gracefully:
if the DB is missing or unreadable, `lookup_by_tic` returns None and
callers fall back to the next data source.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional

logger = logging.getLogger("exonit.berger")


class BergerCatalog:
    """In-memory view of the Berger+2023 TESS-host catalog.

    Each entry exposes the keys: tic_id, mass, mass_err, rad, rad_err,
    teff, teff_err, logg, logg_err, feh, feh_err, dr3_source_id, source.
    """

    def __init__(self, db_path: str):
        self._by_tic: dict[int, dict] = {}
        self._by_gaia: dict[int, dict] = {}
        self._db_path = db_path
        self._load()

    def _load(self) -> None:
        try:
            conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
        except sqlite3.OperationalError as e:
            logger.warning(
                "Berger catalog not available at %s (%s); lookups will return None",
                self._db_path, e)
            return
        try:
            cur = conn.execute(
                "SELECT tic_id, mass, mass_err, rad, rad_err, "
                "teff, teff_err, logg, logg_err, feh, feh_err, "
                "dr3_source_id, source FROM berger_2023"
            )
            cols = [c[0] for c in cur.description]
            for row in cur:
                entry = dict(zip(cols, row))
                self._by_tic[entry["tic_id"]] = entry
                gaia = entry.get("dr3_source_id")
                if gaia is not None:
                    self._by_gaia[gaia] = entry
        finally:
            conn.close()
        logger.info(
            "Berger+2023 catalog loaded: %d TICs, %d with Gaia DR3 id",
            len(self._by_tic), len(self._by_gaia))

    @property
    def loaded(self) -> bool:
        return bool(self._by_tic)

    def lookup_by_tic(self, tic_id: int) -> Optional[dict]:
        """Return the entry for this TIC, or None if absent."""
        return self._by_tic.get(int(tic_id)) if tic_id is not None else None

    def lookup_by_gaia(self, gaia_dr3_id: int) -> Optional[dict]:
        """Return the entry for this Gaia DR3 source_id, or None."""
        return self._by_gaia.get(int(gaia_dr3_id)) if gaia_dr3_id is not None else None

    def __len__(self) -> int:
        return len(self._by_tic)
