"""Tests for the Berger+2023 TESS-host stellar properties catalog lookup."""

import os
import sqlite3

import pytest

from exonit.berger_catalog import BergerCatalog


def _build_mini_db(path: str) -> None:
    """Create a tiny SQLite with the same schema as the production build."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE berger_2023 (
            tic_id        INTEGER PRIMARY KEY,
            teff          REAL NOT NULL,
            teff_err      REAL,
            logg          REAL NOT NULL,
            logg_err      REAL,
            feh           REAL,
            feh_err       REAL,
            mass          REAL NOT NULL,
            mass_err      REAL,
            rad           REAL NOT NULL,
            rad_err       REAL,
            dr3_source_id INTEGER,
            source        TEXT NOT NULL DEFAULT 'Berger+2023'
        );
        CREATE INDEX idx_berger_gaia ON berger_2023(dr3_source_id);
    """)
    conn.executemany(
        "INSERT INTO berger_2023 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (348835438, 4839.18, 111.0, 3.31, 0.11, -0.03, 0.13,
             1.17, 0.13, 3.92, 0.18, 5735664051960550144, "Berger+2023"),
            (202428245, 4885.90, 107.9, 3.43, 0.10, -0.10, 0.15,
             1.17, 0.15, 3.43, 0.17, 1619730990225434368, "Berger+2023"),
            (303432813, 4890.00, 100.0, 4.50, 0.05, 0.0, 0.10,
             0.81, 0.05, 0.78, 0.02, None, "Berger+2023"),
        ],
    )
    conn.commit()
    conn.close()


@pytest.fixture
def berger_db(tmp_path):
    path = tmp_path / "berger_mini.db"
    _build_mini_db(str(path))
    return str(path)


class TestLoading:
    def test_loads_present_db(self, berger_db):
        bc = BergerCatalog(berger_db)
        assert bc.loaded is True
        assert len(bc) == 3

    def test_missing_db_degrades_gracefully(self, tmp_path):
        bc = BergerCatalog(str(tmp_path / "does_not_exist.db"))
        assert bc.loaded is False
        assert len(bc) == 0
        assert bc.lookup_by_tic(348835438) is None


class TestLookupByTic:
    def test_hit_returns_dict_with_stellar_params(self, berger_db):
        bc = BergerCatalog(berger_db)
        entry = bc.lookup_by_tic(348835438)
        assert entry is not None
        assert entry["mass"] == pytest.approx(1.17)
        assert entry["rad"] == pytest.approx(3.92)
        assert entry["teff"] == pytest.approx(4839.18)
        assert entry["logg"] == pytest.approx(3.31)
        assert entry["source"] == "Berger+2023"

    def test_miss_returns_none(self, berger_db):
        bc = BergerCatalog(berger_db)
        assert bc.lookup_by_tic(999999999) is None

    def test_none_input_returns_none(self, berger_db):
        bc = BergerCatalog(berger_db)
        assert bc.lookup_by_tic(None) is None

    def test_string_tic_coerced(self, berger_db):
        bc = BergerCatalog(berger_db)
        entry = bc.lookup_by_tic("348835438")
        assert entry is not None
        assert entry["mass"] == pytest.approx(1.17)


class TestLookupByGaia:
    def test_hit_returns_same_entry(self, berger_db):
        bc = BergerCatalog(berger_db)
        entry = bc.lookup_by_gaia(1619730990225434368)
        assert entry is not None
        assert entry["tic_id"] == 202428245

    def test_miss_returns_none(self, berger_db):
        bc = BergerCatalog(berger_db)
        assert bc.lookup_by_gaia(1) is None

    def test_entry_without_gaia_not_indexed(self, berger_db):
        """TIC 303432813 has dr3_source_id=NULL in the fixture."""
        bc = BergerCatalog(berger_db)
        assert bc.lookup_by_tic(303432813) is not None
        assert bc.lookup_by_gaia(None) is None


_PROD_DB = "data/catalogs/berger_2023_stellar.db"


@pytest.mark.skipif(
    not os.path.exists(_PROD_DB),
    reason="Production Berger DB not built in this environment",
)
class TestProductionCatalog:
    """Smoke checks against the committed Berger DB. Skipped if not present."""

    def test_expected_row_count_and_subgiant_hit(self):
        bc = BergerCatalog(_PROD_DB)
        assert len(bc) == 4180  # all TESS hosts from arXiv:2301.11338
        entry = bc.lookup_by_tic(348835438)
        assert entry is not None
        # Known evolved subgiant that fails TIC v8 mass lookup
        assert entry["logg"] < 4.0
        assert entry["mass"] > 0
        assert entry["rad"] > 1.0

    def test_known_miss_falls_through(self):
        bc = BergerCatalog(_PROD_DB)
        # TIC 319068154: observed as M=nan but not in Berger.
        # Caller should fall through to X-hack.
        assert bc.lookup_by_tic(319068154) is None
