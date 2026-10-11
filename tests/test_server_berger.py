"""End-to-end tests for Berger+2023 overlay at /api/nearby/{tic_id}.

These use the cache-hit path so they don't require MAST connectivity.
They verify that values cached under the TIC v8 regime are overwritten
with Berger values at response time, and that the `stellar_source` tag
is set accordingly.
"""

import os
import sqlite3

import pytest
from fastapi.testclient import TestClient


def _seed_metadata_cache(cache_path: str, target_tic: str,
                         neighbor_tic: str) -> None:
    """Pre-populate a metadata_cache.db with a cached nearby query.

    Values deliberately differ from the Berger fixture so we can detect
    the overlay. Only the target TIC is populated in Berger.
    """
    conn = sqlite3.connect(cache_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (
            tic_id TEXT PRIMARY KEY, status TEXT, attempts INTEGER,
            last_error TEXT, updated_at TEXT, stellar_radius REAL,
            teff REAL, logg REAL, tmag REAL, ra REAL, dec REAL
        );
        CREATE TABLE IF NOT EXISTS fits_paths (
            filename TEXT PRIMARY KEY, full_path TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS nearby_stars (
            tic_id TEXT NOT NULL, neighbor_id TEXT NOT NULL,
            ra REAL, dec REAL, mass REAL, rad REAL, teff REAL,
            tmag REAL, plx REAL, gal_b REAL,
            updated_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (tic_id, neighbor_id)
        );
    """)
    conn.executemany(
        "INSERT INTO nearby_stars "
        "(tic_id, neighbor_id, ra, dec, mass, rad, teff, tmag, plx, gal_b) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (target_tic, target_tic, 10.0, 20.0,
             None, None, 4500.0, 10.5, 1.0, -15.0),
            (target_tic, neighbor_tic, 10.01, 20.01,
             0.5, 0.4, 3500.0, 14.2, 0.5, -15.0),
        ],
    )
    conn.commit()
    conn.close()


def _seed_berger(berger_path: str, tic: int, mass: float, rad: float,
                 teff: float) -> None:
    conn = sqlite3.connect(berger_path)
    conn.executescript("""
        CREATE TABLE berger_2023 (
            tic_id INTEGER PRIMARY KEY,
            teff REAL NOT NULL, teff_err REAL,
            logg REAL NOT NULL, logg_err REAL,
            feh REAL, feh_err REAL,
            mass REAL NOT NULL, mass_err REAL,
            rad REAL NOT NULL, rad_err REAL,
            dr3_source_id INTEGER,
            source TEXT NOT NULL DEFAULT 'Berger+2023'
        );
    """)
    conn.execute(
        "INSERT INTO berger_2023 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (tic, teff, 100.0, 3.5, 0.1, 0.0, 0.1,
         mass, 0.1, rad, 0.1, 123456789, "Berger+2023"),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def client_with_berger(tmp_path, monkeypatch):
    target_tic = "348835438"
    neighbor_tic = "200000001"

    metadata_path = tmp_path / "metadata_cache.db"
    berger_path = tmp_path / "berger.db"
    _seed_metadata_cache(str(metadata_path), target_tic, neighbor_tic)
    _seed_berger(str(berger_path), 348835438,
                 mass=1.17, rad=3.92, teff=4839.18)

    monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EXONIT_METADATA_CACHE", str(metadata_path))
    monkeypatch.setenv("EXONIT_BERGER_DB", str(berger_path))
    os.makedirs(tmp_path / "reports", exist_ok=True)

    from exonit.server import create_app
    app = create_app()
    return TestClient(app), target_tic, neighbor_tic


class TestBergerOverlayCacheHit:
    def test_target_star_is_overwritten_with_berger_values(
            self, client_with_berger):
        client, target_tic, _ = client_with_berger
        resp = client.get(
            f"/api/nearby/{target_tic}",
            headers={"Authorization": "Bearer exonit-default-token"},
        )
        assert resp.status_code == 200
        data = resp.json()
        target = next(s for s in data["stars"] if s["ID"] == target_tic)
        assert target["mass"] == pytest.approx(1.17)
        assert target["rad"] == pytest.approx(3.92)
        assert target["Teff"] == pytest.approx(4839.18)
        assert target["stellar_source"] == "Berger+2023"

    def test_neighbor_not_in_berger_keeps_original_values(
            self, client_with_berger):
        client, target_tic, neighbor_tic = client_with_berger
        resp = client.get(
            f"/api/nearby/{target_tic}",
            headers={"Authorization": "Bearer exonit-default-token"},
        )
        assert resp.status_code == 200
        data = resp.json()
        neighbor = next(s for s in data["stars"]
                        if s["ID"] == neighbor_tic)
        assert neighbor["mass"] == pytest.approx(0.5)
        assert neighbor["rad"] == pytest.approx(0.4)
        assert neighbor["Teff"] == pytest.approx(3500.0)
        assert "stellar_source" not in neighbor


class TestBergerGracefulAbsence:
    def test_server_starts_when_berger_db_missing(self, tmp_path, monkeypatch):
        metadata_path = tmp_path / "metadata_cache.db"
        _seed_metadata_cache(str(metadata_path), "100", "101")
        monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setenv("EXONIT_METADATA_CACHE", str(metadata_path))
        monkeypatch.setenv("EXONIT_BERGER_DB",
                           str(tmp_path / "nonexistent.db"))
        os.makedirs(tmp_path / "reports", exist_ok=True)

        from exonit.server import create_app
        app = create_app()
        client = TestClient(app)
        resp = client.get(
            "/api/nearby/100",
            headers={"Authorization": "Bearer exonit-default-token"},
        )
        assert resp.status_code == 200
        data = resp.json()
        target = next(s for s in data["stars"] if s["ID"] == "100")
        assert "stellar_source" not in target
