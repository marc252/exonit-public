"""Tests for the FPP-gate (EXONIT_FPP_GATE=1) server-side optimisation.

Covers:
  - db.mark_fpp_uncomputable: writes fpp=-1/nfpp=-1/verdict='failed', guards
    AND triceratops_fpp IS NULL (never overwrites an existing value).
  - target_mass_is_uncomputable: the three enrichment steps
      1. raw cache lookup
      2. Berger+2023 overlay
      3. derive_mass_from_logg_rad fallback
    with the key equivalence cases from the brief:
      * target massless in cache + no Berger + no logg  → True  (pre-mark)
      * target with mass in cache                        → False (serve)
      * target massless in cache but Berger has mass     → False (serve)
      * target not in nearby_stars cache                 → False (cannot
                                                            determine, let
                                                            worker decide)
  - worker_fpp_task gate integration: gate active → uncomputable candidate
    pre-marked and skipped; computable candidate served normally.
  - gate OFF (env var unset) → no marking, behaviour unchanged.
"""

from __future__ import annotations

import math
import os
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from exonit.db import ExoNitDB, fpp_verdict
from exonit.server import target_mass_is_uncomputable, _run_fpp_gate_background


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_db(tmp_path, name="test.db"):
    return ExoNitDB(str(tmp_path / name))


def _add_star_and_candidate(db, tic_id, snr=15.0, cnn_score=0.85,
                             fpp=None):
    """Insert a completed star + one candidate. Returns candidate_id."""
    db.add_star(
        tic_id,
        ra=10.0, dec=-20.0, tmag=11.0,
        n_sectors=1, n_datapoints=1000,
        time_span_days=27.0,
        status="completed",
    )
    cid = db.add_candidate(
        tic_id,
        candidate_num=1,
        period_days=3.0,
        t0=0.0,
        duration_days=0.1,
        depth_ppm=1000,
        snr=snr,
        cnn_score=cnn_score,
        triceratops_fpp=fpp,
    )
    return cid


def _make_cache(tmp_path, tic_id, mass=None, rad=None, teff=None, logg=None,
                add_nearby=True):
    """Create a minimal MetadataCache with optional nearby_stars row."""
    from exonit.metadata_cache import MetadataCache
    mc = MetadataCache(str(tmp_path / "metadata_cache.db"))
    # Add main entry (needed for get_entry used in derive_mass fallback)
    mc.add_entry(
        tic_id,
        sectors=[1],
        exptime=120.0,
        fits_files=[],
        stellar_radius=rad,
        teff=teff,
        logg=logg,
        ra=10.0,
        dec=-20.0,
    )
    if add_nearby:
        # Build a nearby_stars row where the target is its own neighbour.
        star = {
            "ID": tic_id,
            "ra": 10.0, "dec": -20.0,
            "mass": mass if mass is not None else float("nan"),
            "rad": rad if rad is not None else float("nan"),
            "Teff": teff if teff is not None else float("nan"),
            "Tmag": 11.0,
            "plx": float("nan"),
        }
        mc.add_nearby_stars(tic_id, [star], gal_b=30.0)
    return mc


def _null_berger():
    """A BergerCatalog stub that always returns None (TIC not in catalog)."""
    b = MagicMock()
    b.lookup_by_tic.return_value = None
    return b


def _berger_with(mass=1.0, rad=1.0, teff=5778.0):
    """A BergerCatalog stub that returns fixed stellar params."""
    b = MagicMock()
    b.lookup_by_tic.return_value = {"mass": mass, "rad": rad, "teff": teff}
    return b


# --------------------------------------------------------------------------- #
# 1. db.mark_fpp_uncomputable
# --------------------------------------------------------------------------- #

class TestMarkFppUncomputable:
    """mark_fpp_uncomputable writes fpp=-1/nfpp=-1/verdict='failed'."""

    def test_marks_null_candidate(self, tmp_path):
        db = _make_db(tmp_path)
        cid = _add_star_and_candidate(db, "1001", fpp=None)
        updated = db.mark_fpp_uncomputable(cid)
        assert updated is True
        row = db._query_one("SELECT * FROM candidates WHERE id=?", (cid,))
        assert row["triceratops_fpp"] == -1
        assert row["triceratops_nfpp"] == -1
        assert row["triceratops_verdict"] == "failed"
        db.close()

    def test_does_not_overwrite_existing_positive_fpp(self, tmp_path):
        """Guard: AND triceratops_fpp IS NULL — must not touch existing ≥0."""
        db = _make_db(tmp_path)
        cid = _add_star_and_candidate(db, "1002", fpp=None)
        # Write a real FPP result first.
        db.update_candidate_fpp(cid, 0.30, nfpp=0.0002)
        # Now try gate — should be a no-op.
        updated = db.mark_fpp_uncomputable(cid)
        assert updated is False
        row = db._query_one("SELECT * FROM candidates WHERE id=?", (cid,))
        assert row["triceratops_fpp"] == pytest.approx(0.30)
        db.close()

    def test_does_not_overwrite_existing_minus_one(self, tmp_path):
        """Calling twice: second call must be a no-op (already -1)."""
        db = _make_db(tmp_path)
        cid = _add_star_and_candidate(db, "1003", fpp=None)
        db.mark_fpp_uncomputable(cid)
        updated2 = db.mark_fpp_uncomputable(cid)
        assert updated2 is False
        db.close()

    def test_verdict_is_failed(self, tmp_path):
        """fpp_verdict(-1, -1) must equal 'failed' per db.py convention."""
        assert fpp_verdict(-1, -1) == "failed"
        db = _make_db(tmp_path)
        cid = _add_star_and_candidate(db, "1004", fpp=None)
        db.mark_fpp_uncomputable(cid)
        row = db._query_one(
            "SELECT triceratops_verdict FROM candidates WHERE id=?", (cid,))
        assert row["triceratops_verdict"] == "failed"
        db.close()


# --------------------------------------------------------------------------- #
# 2. target_mass_is_uncomputable
# --------------------------------------------------------------------------- #

class TestTargetMassIsUncomputable:
    """Unit tests for the three-step enrichment helper."""

    def test_massless_no_berger_no_derivation_is_uncomputable(self, tmp_path):
        """No mass, no rad, no Teff, not in Berger → True."""
        mc = _make_cache(tmp_path, "2001",
                         mass=None, rad=None, teff=None, logg=None)
        assert target_mass_is_uncomputable(mc, _null_berger(), "2001") is True

    def test_mass_in_cache_is_computable(self, tmp_path):
        """Target has mass, rad, Teff in cache → False."""
        mc = _make_cache(tmp_path, "2002",
                         mass=1.0, rad=1.0, teff=5778.0)
        assert target_mass_is_uncomputable(mc, _null_berger(), "2002") is False

    def test_berger_provides_mass_so_computable(self, tmp_path):
        """Cache has no mass but Berger has it → False (should not be marked)."""
        # TIC 25139786 is the motivating real-world case (mass=None in cache,
        # but present in Berger → mass provided → FPP is calculable).
        mc = _make_cache(tmp_path, "2003",
                         mass=None, rad=0.885, teff=None, logg=None)
        berger = _berger_with(mass=0.89, rad=0.885, teff=5200.0)
        assert target_mass_is_uncomputable(mc, berger, "2003") is False

    def test_no_nearby_cache_returns_false(self, tmp_path):
        """If target is not in nearby_stars cache → False (safe: let worker decide)."""
        mc = _make_cache(tmp_path, "2004",
                         add_nearby=False)
        assert target_mass_is_uncomputable(mc, _null_berger(), "2004") is False

    def test_mass_derivable_from_logg_rad(self, tmp_path):
        """Cache has rad + logg but no mass; derive_mass_from_logg_rad fills it."""
        # logg=4.5, rad=0.95 → mass = 0.95² × 10^(4.5-4.438) ≈ 0.93 Msun
        mc = _make_cache(tmp_path, "2005",
                         mass=None, rad=0.95, teff=5500.0, logg=4.5)
        # Need to put the target in nearby_stars without mass too.
        # _make_cache already built the row; logg is on the metadata entry.
        assert target_mass_is_uncomputable(mc, _null_berger(), "2005") is False

    def test_missing_rad_uncomputable_even_with_logg(self, tmp_path):
        """Without rad, derivation is impossible; logg alone is not enough.
        Worker checks np.isnan(R_s) → -1 if rad also missing."""
        mc = _make_cache(tmp_path, "2006",
                         mass=None, rad=None, teff=None, logg=4.5)
        assert target_mass_is_uncomputable(mc, _null_berger(), "2006") is True

    def test_target_row_missing_from_nearby_stars_list(self, tmp_path):
        """nearby_stars present but target TIC not among neighbours → False."""
        from exonit.metadata_cache import MetadataCache
        mc = MetadataCache(str(tmp_path / "mc2.db"))
        # Add a nearby star for tic 9999 with a DIFFERENT neighbour id.
        mc.add_entry("9999", sectors=[1], exptime=120.0, fits_files=[],
                     ra=10.0, dec=-20.0)
        mc.add_nearby_stars("9999", [
            {"ID": "11111", "ra": 10.01, "dec": -20.01,
             "mass": float("nan"), "rad": float("nan"),
             "Teff": float("nan"), "Tmag": 12.0, "plx": float("nan")},
        ], gal_b=30.0)
        # Target itself (9999) is not in the list.
        assert target_mass_is_uncomputable(mc, _null_berger(), "9999") is False

    def test_nan_floats_in_cache_treated_as_missing(self, tmp_path):
        """float NaN from cache (get_nearby_stars returns NaN for absent cols)
        must be treated as missing — not as a finite value."""
        mc = _make_cache(tmp_path, "2007",
                         mass=float("nan"), rad=float("nan"), teff=float("nan"))
        assert target_mass_is_uncomputable(mc, _null_berger(), "2007") is True


# --------------------------------------------------------------------------- #
# 3. Integration: worker_fpp_task with gate active / inactive
# --------------------------------------------------------------------------- #

class TestFppGateIntegration:
    """End-to-end: gate marks uncomputable candidate, serves computable one."""

    def _build_app_and_db(self, tmp_path, monkeypatch):
        """Create a minimal app+client for fpp_task endpoint tests."""
        monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "integ.db"))
        monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
        os.makedirs(tmp_path / "reports", exist_ok=True)

        from fastapi.testclient import TestClient
        from exonit.server import create_app

        app = create_app()
        client = TestClient(app)
        auth = {"Authorization": "Bearer exonit-default-token"}
        reg = client.post(
            "/api/worker/register",
            json={"name": "test-fpp-gate-worker", "capabilities": {}},
            headers=auth,
        )
        worker_id = reg.json()["worker_id"]
        headers = {**auth, "X-Worker-ID": worker_id}

        db = ExoNitDB(str(tmp_path / "integ.db"))
        return client, headers, db

    def test_gate_off_by_default_no_marking(self, tmp_path, monkeypatch):
        """Without EXONIT_FPP_GATE=1, uncomputable candidates are served
        (or simply not marked) — exact current behaviour unchanged."""
        monkeypatch.delenv("EXONIT_FPP_GATE", raising=False)
        client, headers, db = self._build_app_and_db(tmp_path, monkeypatch)

        tic = "3001"
        cid = _add_star_and_candidate(db, tic, snr=12.0, cnn_score=0.8)

        # target_mass_is_uncomputable would return True for this TIC
        # (no nearby_stars cache), but the gate is OFF.
        with patch("exonit.server.target_mass_is_uncomputable",
                   return_value=True) as mock_fn:
            resp = client.get("/api/worker/fpp_task", headers=headers)
            # Gate is off → helper must NOT be called.
            mock_fn.assert_not_called()

        row = db._query_one("SELECT triceratops_fpp FROM candidates WHERE id=?",
                            (cid,))
        assert row["triceratops_fpp"] is None  # unchanged
        db.close()

    def test_gate_on_marks_uncomputable_and_skips(self, tmp_path, monkeypatch):
        """With EXONIT_FPP_GATE=1, an uncomputable candidate is pre-marked
        fpp=-1 and NOT served (task=None when only that candidate exists)."""
        monkeypatch.setenv("EXONIT_FPP_GATE", "1")
        client, headers, db = self._build_app_and_db(tmp_path, monkeypatch)

        tic = "3002"
        cid = _add_star_and_candidate(db, tic, snr=12.0, cnn_score=0.8)

        # Provide a metadata entry so the _mc.get_entry check passes.
        from exonit.metadata_cache import MetadataCache
        mc_path = str(tmp_path / "integ_metadata_cache.db")
        mc = MetadataCache(mc_path)
        mc.add_entry(tic, sectors=[1], exptime=120.0, fits_files=[],
                     ra=10.0, dec=-20.0)

        with patch("exonit.server.target_mass_is_uncomputable",
                   return_value=True):
            with patch("exonit.server._get_shared_metadata_cache",
                       return_value=mc):
                resp = client.get("/api/worker/fpp_task", headers=headers)

        assert resp.status_code == 200
        # No computable candidate → task=None
        assert resp.json()["task"] is None

        # The candidate must have been pre-marked.
        row = db._query_one("SELECT triceratops_fpp FROM candidates WHERE id=?",
                            (cid,))
        assert row["triceratops_fpp"] == -1
        db.close()

    def test_gate_on_computable_candidate_served(self, tmp_path, monkeypatch):
        """With EXONIT_FPP_GATE=1, a computable candidate is served normally."""
        monkeypatch.setenv("EXONIT_FPP_GATE", "1")
        client, headers, db = self._build_app_and_db(tmp_path, monkeypatch)

        tic = "3003"
        cid = _add_star_and_candidate(db, tic, snr=12.0, cnn_score=0.8)

        from exonit.metadata_cache import MetadataCache
        mc_path = str(tmp_path / "integ_mc2.db")
        mc = MetadataCache(mc_path)
        mc.add_entry(tic, sectors=[1], exptime=120.0, fits_files=[],
                     ra=10.0, dec=-20.0)

        with patch("exonit.server.target_mass_is_uncomputable",
                   return_value=False):
            with patch("exonit.server._get_shared_metadata_cache",
                       return_value=mc):
                resp = client.get("/api/worker/fpp_task", headers=headers)

        assert resp.status_code == 200
        task = resp.json().get("task")
        assert task is not None
        assert task["tic_id"] == tic

        # Candidate must NOT have been marked.
        row = db._query_one("SELECT triceratops_fpp FROM candidates WHERE id=?",
                            (cid,))
        assert row["triceratops_fpp"] is None
        db.close()

    def test_gate_never_overwrites_existing_fpp(self, tmp_path, monkeypatch):
        """A candidate with fpp already set must never be touched by the gate,
        even if target_mass_is_uncomputable would return True."""
        monkeypatch.setenv("EXONIT_FPP_GATE", "1")
        client, headers, db = self._build_app_and_db(tmp_path, monkeypatch)

        tic = "3004"
        cid = _add_star_and_candidate(db, tic, snr=12.0, cnn_score=0.8)
        # Pre-set fpp to a real value.
        db.update_candidate_fpp(cid, 0.25, nfpp=0.0001)

        from exonit.metadata_cache import MetadataCache
        mc_path = str(tmp_path / "integ_mc3.db")
        mc = MetadataCache(mc_path)
        mc.add_entry(tic, sectors=[1], exptime=120.0, fits_files=[],
                     ra=10.0, dec=-20.0)

        with patch("exonit.server.target_mass_is_uncomputable",
                   return_value=True):
            with patch("exonit.server._get_shared_metadata_cache",
                       return_value=mc):
                client.get("/api/worker/fpp_task", headers=headers)

        row = db._query_one("SELECT triceratops_fpp FROM candidates WHERE id=?",
                            (cid,))
        assert row["triceratops_fpp"] == pytest.approx(0.25)
        db.close()


# --------------------------------------------------------------------------- #
# 4. db.mark_fpp_uncomputable_for_tic (new batch marker)
# --------------------------------------------------------------------------- #

class TestMarkFppUncomputableForTic:
    """Batch FPP-gate marker: all eligible candidates of a TIC are pre-marked."""

    def _add_candidate(self, db, tic_id, snr=15.0, cnn_score=0.85, fpp=None,
                        candidate_num=None):
        """Insert a candidate for an existing star. Returns candidate_id."""
        db._exec(
            "UPDATE stars SET status='completed' WHERE tic_id=?", (tic_id,))
        cid = db.add_candidate(
            tic_id,
            candidate_num=candidate_num or 1,
            period_days=3.0,
            t0=0.0,
            duration_days=0.1,
            depth_ppm=1000,
            snr=snr,
            cnn_score=cnn_score,
            triceratops_fpp=fpp,
        )
        return cid

    def test_marks_all_eligible_candidates_of_tic(self, tmp_path):
        """All candidates of a TIC with snr>=10, cnn>=0.7, fpp IS NULL get marked."""
        db = _make_db(tmp_path)
        db.add_star("5001", ra=1.0, dec=1.0, tmag=10.0, n_sectors=1,
                    n_datapoints=1000, time_span_days=27.0, status="completed")
        cid1 = self._add_candidate(db, "5001", snr=15.0, cnn_score=0.85,
                                   candidate_num=1)
        cid2 = self._add_candidate(db, "5001", snr=12.0, cnn_score=0.90,
                                   candidate_num=2)
        n = db.mark_fpp_uncomputable_for_tic("5001")
        assert n == 2
        for cid in (cid1, cid2):
            row = db._query_one("SELECT * FROM candidates WHERE id=?", (cid,))
            assert row["triceratops_fpp"] == -1
            assert row["triceratops_nfpp"] == -1
            assert row["triceratops_verdict"] == "failed"
        db.close()

    def test_never_overwrites_existing_nonnull_fpp(self, tmp_path):
        """IS NULL guard: a candidate with fpp already set must not be touched."""
        db = _make_db(tmp_path)
        db.add_star("5002", ra=1.0, dec=1.0, tmag=10.0, n_sectors=1,
                    n_datapoints=1000, time_span_days=27.0, status="completed")
        cid_real = self._add_candidate(db, "5002", snr=15.0, cnn_score=0.85,
                                       candidate_num=1)
        cid_null = self._add_candidate(db, "5002", snr=12.0, cnn_score=0.90,
                                       candidate_num=2)
        # Pre-set a real FPP on cid_real.
        db.update_candidate_fpp(cid_real, 0.35, nfpp=0.0002)
        n = db.mark_fpp_uncomputable_for_tic("5002")
        # Only cid_null is eligible (cid_real has fpp=0.35).
        assert n == 1
        row_real = db._query_one("SELECT triceratops_fpp FROM candidates WHERE id=?",
                                  (cid_real,))
        assert row_real["triceratops_fpp"] == pytest.approx(0.35)
        row_null = db._query_one("SELECT triceratops_fpp FROM candidates WHERE id=?",
                                  (cid_null,))
        assert row_null["triceratops_fpp"] == -1
        db.close()

    def test_never_overwrites_existing_minus_one(self, tmp_path):
        """Calling twice: second call must return 0 (already -1)."""
        db = _make_db(tmp_path)
        db.add_star("5003", ra=1.0, dec=1.0, tmag=10.0, n_sectors=1,
                    n_datapoints=1000, time_span_days=27.0, status="completed")
        self._add_candidate(db, "5003", snr=15.0, cnn_score=0.85)
        db.mark_fpp_uncomputable_for_tic("5003")
        n2 = db.mark_fpp_uncomputable_for_tic("5003")
        assert n2 == 0
        db.close()

    def test_respects_snr_predicate(self, tmp_path):
        """Candidates with snr < 10.0 must not be touched."""
        db = _make_db(tmp_path)
        db.add_star("5004", ra=1.0, dec=1.0, tmag=10.0, n_sectors=1,
                    n_datapoints=1000, time_span_days=27.0, status="completed")
        cid_low = self._add_candidate(db, "5004", snr=9.9, cnn_score=0.85,
                                      candidate_num=1)
        cid_ok = self._add_candidate(db, "5004", snr=10.1, cnn_score=0.85,
                                     candidate_num=2)
        n = db.mark_fpp_uncomputable_for_tic("5004")
        assert n == 1
        row_low = db._query_one("SELECT triceratops_fpp FROM candidates WHERE id=?",
                                 (cid_low,))
        assert row_low["triceratops_fpp"] is None
        row_ok = db._query_one("SELECT triceratops_fpp FROM candidates WHERE id=?",
                                (cid_ok,))
        assert row_ok["triceratops_fpp"] == -1
        db.close()

    def test_respects_cnn_predicate(self, tmp_path):
        """Candidates with cnn_score < 0.7 must not be touched."""
        db = _make_db(tmp_path)
        db.add_star("5005", ra=1.0, dec=1.0, tmag=10.0, n_sectors=1,
                    n_datapoints=1000, time_span_days=27.0, status="completed")
        cid_low = self._add_candidate(db, "5005", snr=15.0, cnn_score=0.69,
                                      candidate_num=1)
        cid_ok = self._add_candidate(db, "5005", snr=15.0, cnn_score=0.71,
                                     candidate_num=2)
        n = db.mark_fpp_uncomputable_for_tic("5005")
        assert n == 1
        row_low = db._query_one("SELECT triceratops_fpp FROM candidates WHERE id=?",
                                 (cid_low,))
        assert row_low["triceratops_fpp"] is None
        row_ok = db._query_one("SELECT triceratops_fpp FROM candidates WHERE id=?",
                                (cid_ok,))
        assert row_ok["triceratops_fpp"] == -1
        db.close()

    def test_returns_zero_for_unknown_tic(self, tmp_path):
        """A TIC with no candidates returns 0 (no error)."""
        db = _make_db(tmp_path)
        n = db.mark_fpp_uncomputable_for_tic("99999")
        assert n == 0
        db.close()


# --------------------------------------------------------------------------- #
# 5. _run_fpp_gate_background
# --------------------------------------------------------------------------- #

class TestRunFppGateBackground:
    """Unit tests for the eager FPP-gate runner."""

    def _fake_db(self, n_marked=2):
        """Stub db with mark_fpp_uncomputable_for_tic returning n_marked."""
        db = MagicMock()
        db.mark_fpp_uncomputable_for_tic.return_value = n_marked
        return db

    def test_marks_when_uncomputable(self, tmp_path, caplog):
        # print()->logger.* (prints->logger cleanup, 2026-07-02): these
        # background gate functions now log instead of printing to stdout,
        # so assertions moved from capsys to caplog.
        db = self._fake_db(n_marked=3)
        mc = MagicMock()
        berger = MagicMock()
        with patch("exonit.server.target_mass_is_uncomputable", return_value=True), \
             caplog.at_level("INFO", logger="exonit.server"):
            _run_fpp_gate_background(db, "7001", mc, berger)
        db.mark_fpp_uncomputable_for_tic.assert_called_once_with("7001")
        assert "7001" in caplog.text
        assert "pre-marked 3" in caplog.text

    def test_noop_when_computable(self, tmp_path, caplog):
        db = self._fake_db()
        mc = MagicMock()
        berger = MagicMock()
        with patch("exonit.server.target_mass_is_uncomputable", return_value=False), \
             caplog.at_level("INFO", logger="exonit.server"):
            _run_fpp_gate_background(db, "7002", mc, berger)
        db.mark_fpp_uncomputable_for_tic.assert_not_called()
        assert "7002" in caplog.text
        assert "computable" in caplog.text

    def test_never_raises_on_db_error(self, tmp_path, caplog):
        """Even if db.mark_fpp_uncomputable_for_tic throws, no exception escapes."""
        db = MagicMock()
        db.mark_fpp_uncomputable_for_tic.side_effect = RuntimeError("DB locked")
        mc = MagicMock()
        berger = MagicMock()
        with patch("exonit.server.target_mass_is_uncomputable", return_value=True), \
             caplog.at_level("INFO", logger="exonit.server"):
            # Must not raise.
            _run_fpp_gate_background(db, "7003", mc, berger)
        assert "FPP-gate failed TIC 7003" in caplog.text

    def test_never_raises_on_check_error(self, tmp_path, caplog):
        """target_mass_is_uncomputable exception must not escape either."""
        db = self._fake_db()
        mc = MagicMock()
        berger = MagicMock()
        with patch("exonit.server.target_mass_is_uncomputable",
                   side_effect=ValueError("oops")), \
             caplog.at_level("INFO", logger="exonit.server"):
            _run_fpp_gate_background(db, "7004", mc, berger)
        assert "FPP-gate failed TIC 7004" in caplog.text
