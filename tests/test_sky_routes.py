"""Tests for the live sky map endpoints (exonit/routes/sky.py), Fase 1 + 2.

See docs/superpowers/specs/2026-07-18-sky-map-phase1.md and -phase2.md.
Covers:

- ``/api/sky/live``: active TICs resolved via MetadataCache + a single-query
  none/candidate/strong classification of recently-completed stars.
- ``/api/sky/completed``: same classification, larger ceiling, for the
  initial map seed.
- ``/api/sky/background``: 404 with a clear message when the precomputed
  file is missing, 200 serving it when present.
- The 500-row ceiling on ``active``.
- Fase 2: ``dist_pc`` fail-safe null without a gaia_dist.db fixture, present
  when one exists, for active/recent/completed.
"""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sky_app(tmp_path, monkeypatch):
    """Build a TestClient + a raw ExoNitDB handle for seeding rows.

    Same pattern as tests/test_server_dashboard_cache.py and
    tests/test_discovery_api.py: point every path-based env var at tmp_path
    so nothing touches the real repo's data/ directory, then hand back the
    ExoNitDB instance so tests can call its add_star/add_task/add_candidate
    helpers directly instead of round-tripping through HTTP.
    """
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("EXONIT_DB_PATH", db_path)
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EXONIT_METADATA_CACHE", str(tmp_path / "metadata_cache.db"))
    os.makedirs(tmp_path / "reports", exist_ok=True)

    from exonit.shared_state import reset_shared_state
    reset_shared_state()

    from exonit.db import ExoNitDB
    from exonit.server import create_app

    db = ExoNitDB(db_path)
    app = create_app()
    client = TestClient(app)
    yield client, db, tmp_path
    db.close()


def _mark_exominer(db, tic_id, candidate_num, exominer_score=None,
                    exominer_ffi_score=None):
    """Set exominer_score/exominer_ffi_score on an already-inserted candidate.

    add_candidate() has no kwarg for exominer_score (only exominer_ffi_score
    is a constructor arg — see db.py's docstring on the -1 sentinel), so
    tests set it directly via _exec, same idiom as
    tests/test_exominer_status.py and tests/test_sentinel_consistency.py.
    """
    if exominer_score is not None:
        db._exec(
            "UPDATE candidates SET exominer_score=? "
            "WHERE tic_id=? AND candidate_num=?",
            (exominer_score, tic_id, candidate_num))
    if exominer_ffi_score is not None:
        db._exec(
            "UPDATE candidates SET exominer_ffi_score=? "
            "WHERE tic_id=? AND candidate_num=?",
            (exominer_ffi_score, tic_id, candidate_num))


class TestSkyLiveClassification:
    def test_none_candidate_strong_classification(self, sky_app):
        client, db, tmp_path = sky_app

        # tic 100: completed, no candidates at all -> 'none'.
        db.add_star("100", ra=10.0, dec=20.0, tmag=11.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)

        # tic 200: completed, one weak candidate (cnn_score below the
        # idx_cand_strong threshold) -> 'candidate'.
        db.add_star("200", ra=11.0, dec=21.0, tmag=12.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)
        db.add_candidate("200", 1, period_days=3.0, t0=1000.0,
                         duration_days=0.1, depth_ppm=500.0, snr=8.0,
                         cnn_score=0.3)

        # tic 300: completed, a candidate that satisfies the idx_cand_strong
        # predicate (schema_pg.sql:246-248) -> 'strong'.
        db.add_star("300", ra=12.0, dec=22.0, tmag=9.5,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)
        db.add_candidate("300", 1, period_days=5.0, t0=1000.0,
                         duration_days=0.1, depth_ppm=1000.0, snr=15.0,
                         cnn_score=0.9)
        _mark_exominer(db, "300", 1, exominer_score=0.8)

        r = client.get("/api/sky/live")
        assert r.status_code == 200
        data = r.json()
        assert "active" in data and "recent" in data

        by_tic = {s["tic_id"]: s for s in data["recent"]}
        assert by_tic["100"]["result"] == "none"
        assert by_tic["200"]["result"] == "candidate"
        assert by_tic["300"]["result"] == "strong"
        # ra/dec/tmag come straight from the stars row.
        assert by_tic["300"]["ra"] == 12.0
        assert by_tic["300"]["dec"] == 22.0

    def test_strong_via_ffi_minus1_sentinel_fallback(self, sky_app):
        """exominer_score=-1 (no-DV sentinel) falls back to exominer_ffi_score.

        Mirrors _strong_where's NULLIF(exominer_score, -1) handling — a
        FFI-only strong candidate must still classify as 'strong', not be
        vetoed by the -1 placeholder.
        """
        client, db, tmp_path = sky_app
        db.add_star("400", ra=13.0, dec=23.0, tmag=10.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)
        db.add_candidate("400", 1, period_days=4.0, t0=1000.0,
                         duration_days=0.1, depth_ppm=800.0, snr=12.0,
                         cnn_score=0.8, exominer_ffi_score=0.75)
        _mark_exominer(db, "400", 1, exominer_score=-1)

        r = client.get("/api/sky/live")
        assert r.status_code == 200
        by_tic = {s["tic_id"]: s for s in r.json()["recent"]}
        assert by_tic["400"]["result"] == "strong"


class TestSkyLiveActive:
    def test_active_resolved_via_metadata_cache(self, sky_app):
        client, db, tmp_path = sky_app

        from exonit.metadata_cache import MetadataCache
        cache = MetadataCache(str(tmp_path / "metadata_cache.db"))
        cache.add_entry("500", sectors=[14], exptime=1800.0,
                        fits_files=["500.fits"], ra=30.0, dec=40.0, tmag=8.5)

        db.add_task("task-500", "500")
        db.assign_task("task-500", "worker-1")

        r = client.get("/api/sky/live")
        assert r.status_code == 200
        active = r.json()["active"]
        assert len(active) == 1
        assert active[0]["tic_id"] == "500"
        assert active[0]["ra"] == 30.0
        assert active[0]["dec"] == 40.0

    def test_active_tic_without_metadata_is_skipped(self, sky_app):
        """Per spec: TICs with no MetadataCache entry are omitted silently."""
        client, db, tmp_path = sky_app
        db.add_task("task-600", "600")
        db.assign_task("task-600", "worker-1")

        r = client.get("/api/sky/live")
        assert r.status_code == 200
        assert r.json()["active"] == []

    def test_active_ceiling_of_500_respected(self, sky_app):
        client, db, tmp_path = sky_app

        from exonit.metadata_cache import MetadataCache
        cache = MetadataCache(str(tmp_path / "metadata_cache.db"))
        for i in range(600):
            tic = str(1000 + i)
            cache.add_entry(tic, sectors=[14], exptime=1800.0,
                            fits_files=[f"{tic}.fits"],
                            ra=float(i), dec=float(i), tmag=10.0)
            db.add_task(f"task-{tic}", tic)
            db.assign_task(f"task-{tic}", "worker-1")

        r = client.get("/api/sky/live")
        assert r.status_code == 200
        assert len(r.json()["active"]) <= 500


class TestSkyCompleted:
    def test_completed_snapshot_shape_and_ceiling(self, sky_app):
        client, db, tmp_path = sky_app
        for i in range(5):
            tic = str(2000 + i)
            db.add_star(tic, ra=float(i), dec=float(i), tmag=10.0,
                        n_sectors=1, n_datapoints=100, time_span_days=27.0)

        r = client.get("/api/sky/completed?limit=3")
        assert r.status_code == 200
        stars = r.json()["stars"]
        assert len(stars) == 3
        assert all(s["result"] == "none" for s in stars)


class TestSkyDistPc:
    """Fase 2: dist_pc lookup — fail-safe null without gaia_dist.db, present
    with it, via the shared read-only connection in exonit/routes/sky.py.
    """

    def _make_gaia_dist(self, path, rows):
        """rows: list of (tic_id, dist_pc|None) tuples."""
        import sqlite3
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(path))
        con.execute(
            "CREATE TABLE gaia_dist (tic_id TEXT, sector INTEGER, gaia_id TEXT, "
            "parallax REAL, parallax_error REAL, dist_pc REAL, "
            "PRIMARY KEY(tic_id, sector))")
        con.executemany(
            "INSERT INTO gaia_dist (tic_id, sector, dist_pc) VALUES (?, 14, ?)",
            rows)
        con.commit()
        con.close()

    def test_dist_pc_null_without_gaia_dist_db(self, sky_app, monkeypatch):
        """No gaia_dist.db anywhere -> dist_pc: null everywhere, no error."""
        client, db, tmp_path = sky_app
        from exonit.routes import sky as sky_routes
        monkeypatch.setattr(sky_routes, "_GAIA_DIST_PATH", tmp_path / "no_such_gaia_dist.db")

        db.add_star("100", ra=10.0, dec=20.0, tmag=11.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)

        r = client.get("/api/sky/live")
        assert r.status_code == 200
        by_tic = {s["tic_id"]: s for s in r.json()["recent"]}
        assert by_tic["100"]["dist_pc"] is None

    def test_dist_pc_present_in_completed_and_live_recent(self, sky_app, monkeypatch):
        client, db, tmp_path = sky_app
        from exonit.routes import sky as sky_routes
        gaia_dist_path = tmp_path / "gaia_dist.db"
        self._make_gaia_dist(gaia_dist_path, [("700", 123.4), ("701", None)])
        monkeypatch.setattr(sky_routes, "_GAIA_DIST_PATH", gaia_dist_path)

        db.add_star("700", ra=14.0, dec=24.0, tmag=9.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)
        db.add_star("701", ra=15.0, dec=25.0, tmag=9.5,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)
        # tic 702 has a completed star but no gaia_dist row at all.
        db.add_star("702", ra=16.0, dec=26.0, tmag=10.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)

        r = client.get("/api/sky/completed?limit=10")
        assert r.status_code == 200
        by_tic = {s["tic_id"]: s for s in r.json()["stars"]}
        assert by_tic["700"]["dist_pc"] == pytest.approx(123.4)
        assert by_tic["701"]["dist_pc"] is None  # NULL in the DB, not invented
        assert by_tic["702"]["dist_pc"] is None  # absent from gaia_dist entirely

        r2 = client.get("/api/sky/live")
        assert r2.status_code == 200
        by_tic2 = {s["tic_id"]: s for s in r2.json()["recent"]}
        assert by_tic2["700"]["dist_pc"] == pytest.approx(123.4)

    def test_dist_pc_present_for_active_stars(self, sky_app, monkeypatch):
        client, db, tmp_path = sky_app
        from exonit.routes import sky as sky_routes
        gaia_dist_path = tmp_path / "gaia_dist.db"
        self._make_gaia_dist(gaia_dist_path, [("800", 42.0)])
        monkeypatch.setattr(sky_routes, "_GAIA_DIST_PATH", gaia_dist_path)

        from exonit.metadata_cache import MetadataCache
        cache = MetadataCache(str(tmp_path / "metadata_cache.db"))
        cache.add_entry("800", sectors=[14], exptime=1800.0,
                        fits_files=["800.fits"], ra=30.0, dec=40.0, tmag=8.5)
        db.add_task("task-800", "800")
        db.assign_task("task-800", "worker-1")

        r = client.get("/api/sky/live")
        assert r.status_code == 200
        active = r.json()["active"]
        assert len(active) == 1
        assert active[0]["dist_pc"] == pytest.approx(42.0)


class TestSkyBackground:
    def test_missing_background_returns_404_with_hint(self, sky_app, monkeypatch):
        client, db, tmp_path = sky_app
        from exonit.routes import sky as sky_routes
        monkeypatch.setattr(sky_routes, "_SKY_BG_DIR", tmp_path / "sky_bg")

        r = client.get("/api/sky/background?sector=14")
        assert r.status_code == 404
        assert "precompute_sky_background.py" in r.json()["error"]

    def test_existing_background_served_200(self, sky_app, monkeypatch):
        client, db, tmp_path = sky_app
        from exonit.routes import sky as sky_routes
        bg_dir = tmp_path / "sky_bg"
        bg_dir.mkdir()
        payload = {"sector": 14, "count": 2, "points": [[10.0, 20.0, 9.0], [11.0, 21.0, 10.0]]}
        (bg_dir / "sector_14.json").write_text(json.dumps(payload))
        monkeypatch.setattr(sky_routes, "_SKY_BG_DIR", bg_dir)

        r = client.get("/api/sky/background?sector=14")
        assert r.status_code == 200
        assert r.json() == payload
