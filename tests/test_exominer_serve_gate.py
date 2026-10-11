"""Tests for the ExoMiner-serve no-metadata gate.

Bug: when all top-200 candidates belong to TICs with no metadata cache entry,
worker_exominer_task silently returned {"task": None} every poll and the queue
never drained (EXOMINER_BACKFILL stuck, run FAILED).

Fix: instead of a silent ``continue``, mark all eligible candidates for that
TIC with exominer_score=-1 (same sentinel as the SPOC-gate) so they leave the
queue permanently.  A subsequent call will no longer include them in the
top-200 window.

Test matrix
-----------
1. TIC without metadata → candidates marked -1 in DB; second call no longer
   returns them (queue drains).
2. TIC with metadata → candidate served normally (anti-regression).
3. Guard IS NULL: candidate already scored (not NULL) is NOT overwritten.
4. Mixed batch: some TICs with metadata, some without → the ones with metadata
   are served; the ones without are marked -1.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_app_and_db(tmp_path, monkeypatch):
    """Return (client, headers, db) with ExoMiner backfill enabled."""
    monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "em_gate.db"))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EXONIT_EXOMINER_BACKFILL", "1")
    os.makedirs(tmp_path / "reports", exist_ok=True)

    from fastapi.testclient import TestClient
    from exonit.server import create_app
    from exonit.db import ExoNitDB

    app = create_app()
    client = TestClient(app)
    auth = {"Authorization": "Bearer exonit-default-token"}
    reg = client.post(
        "/api/worker/register",
        json={"name": "test-em-gate-worker", "capabilities": {}},
        headers=auth,
    )
    worker_id = reg.json()["worker_id"]
    headers = {**auth, "X-Worker-ID": worker_id}

    db = ExoNitDB(str(tmp_path / "em_gate.db"))
    return client, headers, db


def _add_star_and_candidate(db, tic_id, snr=15.0, cnn_score=0.85,
                             exominer_score=None):
    """Insert a completed star + one candidate; return candidate_id.

    Sets has_spoc=1 so the candidate passes the backfill SPOC-gate filter
    (AND s.has_spoc = 1 was added to worker_exominer_task and the pending
    count query).  These tests exercise the per-candidate no-metadata path,
    which only fires for SPOC-confirmed TICs (has_spoc=NULL rows are now
    handled earlier by mark_unconfirmed_ffi_candidates at EXOMINER_BACKFILL
    entry — see test_exominer_backfill_spoc_gate.py).
    """
    db.add_star(
        tic_id,
        ra=10.0, dec=-20.0, tmag=11.0,
        n_sectors=1, n_datapoints=1000,
        time_span_days=27.0,
        status="completed",
    )
    db.set_star_has_spoc(tic_id, 1)  # must pass has_spoc=1 filter
    cid = db.add_candidate(
        tic_id,
        candidate_num=1,
        period_days=3.0,
        t0=0.0,
        duration_days=0.1,
        depth_ppm=1000,
        snr=snr,
        cnn_score=cnn_score,
    )
    if exominer_score is not None:
        db.update_candidate_exominer(cid, score=exominer_score)
    return cid


def _make_mc(tmp_path, name="mc.db"):
    """Return an empty MetadataCache."""
    from exonit.metadata_cache import MetadataCache
    return MetadataCache(str(tmp_path / name))


def _add_mc_entry(mc, tic_id):
    """Add a minimal metadata entry for tic_id."""
    mc.add_entry(
        tic_id,
        sectors=[1],
        exptime=120.0,
        fits_files=[],
        ra=1.0,
        dec=1.0,
    )


# ---------------------------------------------------------------------------
# Test 1: TIC without metadata entry → candidates marked -1, queue drains
# ---------------------------------------------------------------------------

class TestNoMetadataMarksSentinel:
    """TIC without metadata cache entry must be marked -1, not skipped silently."""

    def test_candidates_marked_minus1_after_poll(self, tmp_path, monkeypatch):
        """After one poll, all candidates for the no-metadata TIC must have
        exominer_score=-1 in the DB."""
        client, headers, db = _build_app_and_db(tmp_path, monkeypatch)
        tic = "99001"
        cid = _add_star_and_candidate(db, tic, snr=20.0)

        # Empty metadata cache: no entry for this TIC.
        mc = _make_mc(tmp_path, "mc_noentry.db")

        with patch("exonit.server._scan_in_flight", return_value=False):
            with patch("exonit.server._get_shared_metadata_cache",
                       return_value=mc):
                resp = client.get("/api/worker/exominer_task", headers=headers)

        # The endpoint may return task=None (no serveable candidate) or the
        # next candidate if there were more TICs — either way, the candidate
        # for this TIC must be marked in the DB.
        assert resp.status_code == 200

        row = db._query_one(
            "SELECT exominer_score FROM candidates WHERE id = ?", (cid,))
        assert row is not None
        assert row["exominer_score"] == -1.0, (
            f"Expected exominer_score=-1, got {row['exominer_score']}")

        db.close()

    def test_second_poll_does_not_return_marked_candidate(self, tmp_path,
                                                           monkeypatch):
        """After the first poll marks the candidate -1, the idle guard (which
        checks exominer_score IS NULL) returns early on the second poll —
        so the same candidate is never reconsidered."""
        client, headers, db = _build_app_and_db(tmp_path, monkeypatch)
        tic = "99002"
        cid = _add_star_and_candidate(db, tic, snr=20.0)
        mc = _make_mc(tmp_path, "mc_noentry2.db")

        with patch("exonit.server._scan_in_flight", return_value=False):
            with patch("exonit.server._get_shared_metadata_cache",
                       return_value=mc):
                # First poll: marks candidate -1
                client.get("/api/worker/exominer_task", headers=headers)
                # Second poll: idle guard fires → task=None
                resp2 = client.get("/api/worker/exominer_task", headers=headers)

        assert resp2.status_code == 200
        assert resp2.json()["task"] is None

        db.close()


# ---------------------------------------------------------------------------
# Test 2: TIC with metadata → served normally (anti-regression)
# ---------------------------------------------------------------------------

class TestWithMetadataServedNormally:
    """Candidate whose TIC has a metadata entry must be served as a task."""

    def test_candidate_served_when_metadata_present(self, tmp_path, monkeypatch):
        client, headers, db = _build_app_and_db(tmp_path, monkeypatch)
        tic = "99101"
        cid = _add_star_and_candidate(db, tic, snr=18.0)

        mc = _make_mc(tmp_path, "mc_entry.db")
        _add_mc_entry(mc, tic)

        with patch("exonit.server._scan_in_flight", return_value=False):
            with patch("exonit.server._get_shared_metadata_cache",
                       return_value=mc):
                resp = client.get("/api/worker/exominer_task", headers=headers)

        assert resp.status_code == 200
        task = resp.json()["task"]
        assert task is not None, "Expected a task to be served"
        assert str(task["tic_id"]) == tic

        # Score must NOT be marked -1
        row = db._query_one(
            "SELECT exominer_score FROM candidates WHERE id = ?", (cid,))
        assert row["exominer_score"] is None, (
            "Serveable candidate must not be marked -1")

        db.close()

    def test_candidate_score_not_altered(self, tmp_path, monkeypatch):
        """Serving a candidate does not change its exominer_score (stays NULL
        until the worker posts the result)."""
        client, headers, db = _build_app_and_db(tmp_path, monkeypatch)
        tic = "99102"
        cid = _add_star_and_candidate(db, tic, snr=18.0)

        mc = _make_mc(tmp_path, "mc_entry2.db")
        _add_mc_entry(mc, tic)

        with patch("exonit.server._scan_in_flight", return_value=False):
            with patch("exonit.server._get_shared_metadata_cache",
                       return_value=mc):
                client.get("/api/worker/exominer_task", headers=headers)

        row = db._query_one(
            "SELECT exominer_score FROM candidates WHERE id = ?", (cid,))
        assert row["exominer_score"] is None

        db.close()


# ---------------------------------------------------------------------------
# Test 3: Guard IS NULL — already-scored candidate not overwritten
# ---------------------------------------------------------------------------

class TestIsNullGuard:
    """mark_ffi_only_candidates must not overwrite an existing real score."""

    def test_existing_real_score_not_overwritten(self, tmp_path, monkeypatch):
        """A candidate with exominer_score=0.85 (already scored) must not be
        touched even if its TIC has no metadata entry."""
        client, headers, db = _build_app_and_db(tmp_path, monkeypatch)
        tic = "99201"
        # Insert with a pre-existing score so it is excluded from the
        # exominer_score IS NULL window (the query filters it out).
        # We verify via mark_ffi_only_candidates directly that it won't overwrite.
        cid = _add_star_and_candidate(db, tic, snr=20.0,
                                       exominer_score=0.85)

        # Directly test the DB method's IS NULL guard.
        n = db.mark_ffi_only_candidates(tic)
        assert n == 0, (
            f"mark_ffi_only_candidates must not overwrite real score (updated {n})")

        row = db._query_one(
            "SELECT exominer_score FROM candidates WHERE id = ?", (cid,))
        assert row["exominer_score"] == pytest.approx(0.85)

        db.close()

    def test_minus1_sentinel_not_overwritten(self, tmp_path, monkeypatch):
        """A candidate already marked -1 must not be touched a second time."""
        client, headers, db = _build_app_and_db(tmp_path, monkeypatch)
        tic = "99202"
        cid = _add_star_and_candidate(db, tic, snr=20.0,
                                       exominer_score=-1.0)

        n = db.mark_ffi_only_candidates(tic)
        assert n == 0, "Already-marked sentinel must not be updated again"

        db.close()


# ---------------------------------------------------------------------------
# Test 4: Mixed batch — some with metadata (served), some without (marked -1)
# ---------------------------------------------------------------------------

class TestMixedBatch:
    """When the top-200 window contains TICs both with and without metadata,
    the ones without are marked -1 and the first one with metadata is served."""

    def test_no_metadata_marked_with_metadata_served(self, tmp_path,
                                                      monkeypatch):
        client, headers, db = _build_app_and_db(tmp_path, monkeypatch)

        # Two TICs: tic_no_meta has higher SNR (appears first in ORDER BY snr
        # DESC); tic_with_meta appears second but has metadata.
        tic_no_meta = "99301"
        tic_with_meta = "99302"
        cid_no = _add_star_and_candidate(db, tic_no_meta, snr=25.0)
        cid_yes = _add_star_and_candidate(db, tic_with_meta, snr=20.0)

        mc = _make_mc(tmp_path, "mc_mixed.db")
        _add_mc_entry(mc, tic_with_meta)  # only tic_with_meta has an entry

        with patch("exonit.server._scan_in_flight", return_value=False):
            with patch("exonit.server._get_shared_metadata_cache",
                       return_value=mc):
                resp = client.get("/api/worker/exominer_task", headers=headers)

        assert resp.status_code == 200
        task = resp.json()["task"]

        # The no-meta TIC's candidate must be marked -1.
        row_no = db._query_one(
            "SELECT exominer_score FROM candidates WHERE id = ?", (cid_no,))
        assert row_no["exominer_score"] == -1.0, (
            "No-metadata TIC candidate must be marked -1")

        # The with-meta TIC's candidate must be served (score still NULL in DB).
        row_yes = db._query_one(
            "SELECT exominer_score FROM candidates WHERE id = ?", (cid_yes,))
        assert row_yes["exominer_score"] is None, (
            "Served candidate score must stay NULL until worker posts result")

        # The task returned must be for the with-meta TIC.
        if task is not None:
            assert str(task["tic_id"]) == tic_with_meta

        db.close()
