"""Tests for the ExoMiner backfill SPOC-gate fix.

Root cause: SPOC-gate threads crash/time-out under load, leaving stars with
has_spoc=NULL.  Those candidates fall into the ExoMiner backfill selection
(exominer_score IS NULL AND cnn>=0.7), causing thousands of DV-fetch 404s.

Fix:
  (a) db.mark_unconfirmed_ffi_candidates() — bulk UPDATE exominer_score=-1
      for all has_spoc=NULL stars.  Mirrors mark_ffi_only_candidates per-TIC
      but operates across the whole DB in one query.
  (b) server.py worker_exominer_task + /api/health/work count — both backfill
      selection queries now have AND s.has_spoc = 1, so has_spoc=NULL/0 rows
      are never served.

2026-07-02 UPDATE (progress/investigation-exominer-minus1-2026-07-02.md +
progress/impl-minus1-fix-2026-07-02.md): this file's ORIGINAL claim that
"the COALESCE gate then uses exominer_ffi_score correctly" once
exominer_score=-1 was FALSE — COALESCE(-1, ffi_score) evaluates to -1, not
ffi_score, so a -1-marked FFI-only TIC with a genuinely strong FFI score was
being hard-vetoed out of the strong tier (the exact bug the 2026-07-02 fix
addresses). ``db._strong_where`` now wraps every read of exominer_score in
``NULLIF(exominer_score, -1)`` so -1 reads as absent, same as a real NULL —
this file's tests below (TestStrongSetNonRegression) were asserting the OLD
(buggy) exclusion and have been corrected to assert the fix instead. The
writers here (mark_unconfirmed_ffi_candidates et al.) are UNCHANGED — they
still store the literal -1; only the strong-tier READ path's interpretation
of that stored -1 changed.

Test matrix
-----------
1.  Non-regression: strong_candidates / count_strong_candidates give the SAME
    set regardless of whether FFI-only rows arrived at -1 via the old slow
    backfill path or the new bulk-mark.  Both paths leave exominer_score=-1;
    the COALESCE(NULLIF(...)) gate then falls back to exominer_ffi_score
    correctly (post-2026-07-02 fix — see UPDATE note above).

2.  Backfill selection query no longer selects has_spoc=NULL or has_spoc=0.
    A has_spoc=1 row IS returned.

3.  mark_unconfirmed_ffi_candidates:
    (a) marks exominer_score=NULL AND cnn>=0.7 rows for has_spoc=NULL stars
    (b) does NOT touch has_spoc=1 rows
    (c) does NOT overwrite a real score (>=0) or an existing -1
    (d) is idempotent (second call returns 0)
    (e) does NOT touch rows with cnn_score < 0.7

4.  /api/admin/mark_unconfirmed_ffi endpoint exists, returns 200 + marked count.
"""

from __future__ import annotations

import os
import pytest

from exonit.db import ExoNitDB


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path, name="test.db"):
    return ExoNitDB(str(tmp_path / name))


def _add_star(db, tic_id, has_spoc=None, status="completed"):
    """Insert a star with given has_spoc value."""
    db.add_star(
        tic_id,
        ra=10.0, dec=-20.0, tmag=11.0,
        n_sectors=1, n_datapoints=1000,
        time_span_days=27.0,
        status=status,
    )
    if has_spoc is not None:
        db.set_star_has_spoc(tic_id, has_spoc)


def _add_candidate(db, tic_id, cnn_score=0.85, exominer_score=None,
                   exominer_ffi_score=None, snr=15.0, candidate_num=1):
    """Insert a candidate; optionally set scores directly."""
    cid = db.add_candidate(
        tic_id,
        candidate_num=candidate_num,
        period_days=3.0,
        t0=0.0,
        duration_days=0.1,
        depth_ppm=1000,
        snr=snr,
        cnn_score=cnn_score,
    )
    if exominer_score is not None:
        db.update_candidate_exominer(cid, score=exominer_score)
    if exominer_ffi_score is not None:
        db._exec(
            "UPDATE candidates SET exominer_ffi_score=? WHERE id=?",
            (exominer_ffi_score, cid))
    return cid


# ---------------------------------------------------------------------------
# 1. Non-regression: strong set is identical between old (slow) and new (bulk)
# ---------------------------------------------------------------------------

class TestStrongSetNonRegression:
    """strong_candidates / count_strong_candidates must give the same set
    whether FFI rows arrive at -1 via the old slow backfill or the new bulk-mark.

    Setup: two stars.
      - tic_spoc: has_spoc=1, exominer_ffi_score=0.8, exominer_score=NULL
        → backfill should score it via DV; here we simulate a real score of 0.75.
      - tic_ffi: has_spoc=NULL, exominer_ffi_score=0.9, exominer_score=NULL
        → old path: backfill writes -1 (DV 404); new path: bulk-mark writes -1.

    In both cases:
      - tic_spoc ends with exominer_score=0.75: included in strong set
        (COALESCE(0.75, 0.8)=0.75>=0.7, min(0.75,0.8)=0.75>=0.6).
      - tic_ffi ends with exominer_score=-1: -1 is a no-DV cache-negative
        sentinel, not a real score (see progress/investigation-exominer-
        minus1-2026-07-02.md), so it reads as absent —
        COALESCE(NULLIF(-1,-1), 0.9)=0.9>=0.7 → INCLUDED in strong set via
        the FFI score (same as if exominer_score had stayed NULL).
    """

    def _setup(self, tmp_path, name):
        db = _make_db(tmp_path, name)
        # SPOC star: will get a real ExoMiner score via DV
        _add_star(db, "1001", has_spoc=1)
        _add_candidate(db, "1001", cnn_score=0.85, exominer_ffi_score=0.8)
        # FFI-only star: has_spoc=NULL (gate failed)
        _add_star(db, "1002", has_spoc=None)
        _add_candidate(db, "1002", cnn_score=0.85, exominer_ffi_score=0.9)
        return db

    def test_old_path_simulate_slow_backfill(self, tmp_path):
        """Simulate the old slow backfill: manually write -1 for FFI, +score for SPOC.
        Verify strong set contains tic_spoc AND tic_ffi (the -1-marked FFI
        TIC falls back to its passing exominer_ffi_score=0.9 — see the
        2026-07-02 UPDATE note in the module docstring)."""
        db = self._setup(tmp_path, "old_path.db")

        # Old path: SPOC DV returns real score, FFI DV returns 404 → -1
        cid_spoc = db._query_one(
            "SELECT id FROM candidates WHERE tic_id='1001'")["id"]
        db.update_candidate_exominer(cid_spoc, score=0.75)

        cid_ffi = db._query_one(
            "SELECT id FROM candidates WHERE tic_id='1002'")["id"]
        db.update_candidate_exominer(cid_ffi, score=-1.0)  # 404 result

        strong = db.strong_candidates()
        strong_tics = {c["tic_id"] for c in strong}
        assert "1001" in strong_tics, "SPOC TIC must be in strong set"
        assert "1002" in strong_tics, (
            "FFI-only TIC (score=-1, ffi=0.9) must be in strong set — -1 "
            "reads as absent and falls back to the passing FFI score")

        count = db.count_strong_candidates()
        assert count == len(strong)
        db.close()

    def test_new_path_bulk_mark_then_score_spoc(self, tmp_path):
        """New path: bulk-mark FFI candidates, then score SPOC via DV.
        Strong set must be identical to old path."""
        db = self._setup(tmp_path, "new_path.db")

        # New path step 1: bulk-mark all has_spoc=NULL before ExoMiner backfill
        n = db.mark_unconfirmed_ffi_candidates()
        assert n == 1, f"Expected 1 row marked, got {n}"

        # New path step 2: SPOC backfill now only runs on has_spoc=1 rows
        cid_spoc = db._query_one(
            "SELECT id FROM candidates WHERE tic_id='1001'")["id"]
        db.update_candidate_exominer(cid_spoc, score=0.75)

        strong = db.strong_candidates()
        strong_tics = {c["tic_id"] for c in strong}
        assert "1001" in strong_tics, "SPOC TIC must be in strong set"
        assert "1002" in strong_tics, (
            "FFI-only TIC (score=-1, ffi=0.9) must be in strong set — -1 "
            "reads as absent and falls back to the passing FFI score")

        count = db.count_strong_candidates()
        assert count == len(strong)
        db.close()

    def test_strong_sets_are_identical(self, tmp_path):
        """The strong set IDs from old path and new path must be equal."""
        db_old = self._setup(tmp_path, "old.db")
        db_new = self._setup(tmp_path, "new.db")

        # Old path
        cid_spoc_old = db_old._query_one(
            "SELECT id FROM candidates WHERE tic_id='1001'")["id"]
        db_old.update_candidate_exominer(cid_spoc_old, score=0.75)
        cid_ffi_old = db_old._query_one(
            "SELECT id FROM candidates WHERE tic_id='1002'")["id"]
        db_old.update_candidate_exominer(cid_ffi_old, score=-1.0)
        strong_old = {c["tic_id"] for c in db_old.strong_candidates()}

        # New path
        db_new.mark_unconfirmed_ffi_candidates()
        cid_spoc_new = db_new._query_one(
            "SELECT id FROM candidates WHERE tic_id='1001'")["id"]
        db_new.update_candidate_exominer(cid_spoc_new, score=0.75)
        strong_new = {c["tic_id"] for c in db_new.strong_candidates()}

        assert strong_old == strong_new, (
            f"Strong sets differ: old={strong_old}, new={strong_new}")

        db_old.close()
        db_new.close()

    def test_null_and_minus1_are_equivalent_in_strong_gate(self, tmp_path):
        """exominer_score=NULL and exominer_score=-1 must be EQUIVALENT for
        the strong-tier gate — both read as "no SPOC score" and fall back to
        exominer_ffi_score.

        Historical note (2026-07-02, progress/investigation-exominer-
        minus1-2026-07-02.md): this test used to assert the OPPOSITE — that
        COALESCE(-1, ffi_score) evaluates to -1 (not ffi_score) and so a -1
        row was excluded even with a passing FFI score. That was the actual
        bug: bulk-marking a candidate exominer_score=-1 (a no-DV cache-
        negative, not a real score) silently hard-vetoed it out of the
        strong tier. ``db._strong_where`` now wraps exominer_score in
        ``NULLIF(exominer_score, -1)`` so -1 reads as absent, same as NULL.
        """
        db = _make_db(tmp_path, "coalesce_test.db")
        _add_star(db, "2001", has_spoc=None)
        _add_candidate(db, "2001", cnn_score=0.85, exominer_ffi_score=0.9)

        # Before bulk-mark: exominer_score=NULL → falls back to ffi=0.9 → included
        strong_before = db.strong_candidates()
        assert any(c["tic_id"] == "2001" for c in strong_before), (
            "Before bulk-mark: FFI TIC with NULL exominer_score and "
            "ffi>=0.7 must be included via the FFI fallback.")

        # After bulk-mark: exominer_score=-1 → NULLIF(-1,-1)=NULL → still
        # falls back to ffi=0.9 → still included (unchanged by the mark).
        db.mark_unconfirmed_ffi_candidates()
        strong_after = db.strong_candidates()
        assert any(c["tic_id"] == "2001" for c in strong_after), (
            "After bulk-mark: FFI TIC with exominer_score=-1 must remain "
            "included — -1 must read identically to NULL in the strong gate.")

        db.close()


# ---------------------------------------------------------------------------
# 2. Backfill selection: has_spoc filter
# ---------------------------------------------------------------------------

class TestBackfillSelectionHasSpocFilter:
    """The worker_exominer_task and /api/health/work count must not return
    candidates for has_spoc=NULL or has_spoc=0 stars."""

    def _build_app_and_db(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "bf_filter.db"))
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
            json={"name": "test-spoc-filter-worker", "capabilities": {}},
            headers=auth,
        )
        worker_id = reg.json()["worker_id"]
        headers = {**auth, "X-Worker-ID": worker_id}
        db = ExoNitDB(str(tmp_path / "bf_filter.db"))
        return client, headers, db

    def test_has_spoc_null_not_served(self, tmp_path, monkeypatch):
        """A candidate for a has_spoc=NULL star must NOT be returned by
        worker_exominer_task."""
        from unittest.mock import patch
        from exonit.metadata_cache import MetadataCache

        client, headers, db = self._build_app_and_db(tmp_path, monkeypatch)
        _add_star(db, "3001", has_spoc=None)
        _add_candidate(db, "3001", cnn_score=0.85)

        mc = MetadataCache(str(tmp_path / "mc.db"))
        mc.add_entry("3001", sectors=[1], exptime=120.0, fits_files=[],
                     ra=1.0, dec=1.0)

        with patch("exonit.server._scan_in_flight", return_value=False):
            with patch("exonit.server._get_shared_metadata_cache",
                       return_value=mc):
                resp = client.get("/api/worker/exominer_task", headers=headers)

        assert resp.status_code == 200
        assert resp.json()["task"] is None, (
            "has_spoc=NULL candidate must NOT be served by worker_exominer_task")
        db.close()

    def test_has_spoc_0_not_served(self, tmp_path, monkeypatch):
        """A candidate for a has_spoc=0 (confirmed FFI-only) star must NOT be
        returned by worker_exominer_task."""
        from unittest.mock import patch
        from exonit.metadata_cache import MetadataCache

        client, headers, db = self._build_app_and_db(tmp_path, monkeypatch)
        _add_star(db, "3002", has_spoc=0)
        _add_candidate(db, "3002", cnn_score=0.85)

        mc = MetadataCache(str(tmp_path / "mc2.db"))
        mc.add_entry("3002", sectors=[1], exptime=120.0, fits_files=[],
                     ra=1.0, dec=1.0)

        with patch("exonit.server._scan_in_flight", return_value=False):
            with patch("exonit.server._get_shared_metadata_cache",
                       return_value=mc):
                resp = client.get("/api/worker/exominer_task", headers=headers)

        assert resp.status_code == 200
        assert resp.json()["task"] is None, (
            "has_spoc=0 candidate must NOT be served by worker_exominer_task")
        db.close()

    def test_has_spoc_1_is_served(self, tmp_path, monkeypatch):
        """A candidate for a has_spoc=1 star IS served by worker_exominer_task."""
        from unittest.mock import patch
        from exonit.metadata_cache import MetadataCache

        client, headers, db = self._build_app_and_db(tmp_path, monkeypatch)
        _add_star(db, "3003", has_spoc=1)
        _add_candidate(db, "3003", cnn_score=0.85)

        mc = MetadataCache(str(tmp_path / "mc3.db"))
        mc.add_entry("3003", sectors=[1], exptime=120.0, fits_files=[],
                     ra=1.0, dec=1.0)

        with patch("exonit.server._scan_in_flight", return_value=False):
            with patch("exonit.server._get_shared_metadata_cache",
                       return_value=mc):
                resp = client.get("/api/worker/exominer_task", headers=headers)

        assert resp.status_code == 200
        task = resp.json()["task"]
        assert task is not None, (
            "has_spoc=1 candidate MUST be served by worker_exominer_task")
        assert str(task["tic_id"]) == "3003"
        db.close()

    def test_health_work_count_excludes_has_spoc_null(self, tmp_path, monkeypatch):
        """The /api/health/work exominer_backfill.pending count must not include
        has_spoc=NULL candidates."""
        client, headers, db = self._build_app_and_db(tmp_path, monkeypatch)

        # has_spoc=NULL candidate — must NOT count
        _add_star(db, "4001", has_spoc=None)
        _add_candidate(db, "4001", cnn_score=0.85)

        # has_spoc=1 candidate — must count
        _add_star(db, "4002", has_spoc=1)
        _add_candidate(db, "4002", cnn_score=0.85)

        resp = client.get("/api/health/work")
        assert resp.status_code == 200
        em = resp.json()["exominer_backfill"]
        # Only 1 row (has_spoc=1) should be pending
        assert em["pending"] == 1, (
            f"Expected pending=1 (has_spoc=1 only), got {em['pending']}")
        db.close()

    def test_health_work_count_excludes_has_spoc_0(self, tmp_path, monkeypatch):
        """The /api/health/work exominer_backfill.pending count must not include
        has_spoc=0 candidates."""
        client, headers, db = self._build_app_and_db(tmp_path, monkeypatch)

        # has_spoc=0 candidate — must NOT count
        _add_star(db, "4003", has_spoc=0)
        _add_candidate(db, "4003", cnn_score=0.85)

        resp = client.get("/api/health/work")
        assert resp.status_code == 200
        em = resp.json()["exominer_backfill"]
        assert em["pending"] == 0, (
            f"Expected pending=0 (no has_spoc=1 rows), got {em['pending']}")
        db.close()


# ---------------------------------------------------------------------------
# 3. mark_unconfirmed_ffi_candidates — scope and idempotency
# ---------------------------------------------------------------------------

class TestMarkUnconfirmedFfiCandidates:
    """Unit tests for db.mark_unconfirmed_ffi_candidates()."""

    def test_marks_null_spoc_cnn_high(self, tmp_path):
        """Rows with has_spoc=NULL and cnn>=0.7 and exominer_score=NULL are marked."""
        db = _make_db(tmp_path, "a.db")
        _add_star(db, "10", has_spoc=None)
        _add_candidate(db, "10", cnn_score=0.9)
        _add_candidate(db, "10", cnn_score=0.75, candidate_num=2)

        n = db.mark_unconfirmed_ffi_candidates()
        assert n == 2

        rows = db._query("SELECT exominer_score FROM candidates WHERE tic_id='10'")
        for row in rows:
            assert row["exominer_score"] == -1.0
        db.close()

    def test_does_not_touch_has_spoc_1(self, tmp_path):
        """Rows for has_spoc=1 stars must NOT be touched."""
        db = _make_db(tmp_path, "b.db")
        _add_star(db, "20", has_spoc=1)
        cid = _add_candidate(db, "20", cnn_score=0.9)

        n = db.mark_unconfirmed_ffi_candidates()
        assert n == 0

        row = db._query_one(
            "SELECT exominer_score FROM candidates WHERE id=?", (cid,))
        assert row["exominer_score"] is None
        db.close()

    def test_does_not_touch_has_spoc_0(self, tmp_path):
        """Rows for has_spoc=0 stars (confirmed FFI-only, already handled
        per-TIC by the SPOC gate) are NOT touched by this bulk method.

        Note: in practice those rows already have exominer_score=-1 written
        by the per-TIC mark_ffi_only_candidates call; the IS NULL guard makes
        this test academic but it confirms the predicate is tight.
        """
        db = _make_db(tmp_path, "c.db")
        _add_star(db, "30", has_spoc=0)
        # Simulate state where the SPOC gate already ran (set -1) — IS NULL guard
        cid_marked = _add_candidate(db, "30", cnn_score=0.9,
                                    exominer_score=-1.0)
        # Also add a NULL one (gate missed it somehow) — should NOT be touched
        # because has_spoc=0 is not in the bulk-mark predicate.
        # Wait — actually the predicate is has_spoc IS NULL, so has_spoc=0 is
        # NOT included.  This confirms correctness.
        cid_null = _add_candidate(db, "30", cnn_score=0.9, candidate_num=2)

        n = db.mark_unconfirmed_ffi_candidates()
        assert n == 0, (
            "has_spoc=0 rows must NOT be touched by mark_unconfirmed_ffi_candidates "
            "(predicate is has_spoc IS NULL, not has_spoc != 1)")

        row = db._query_one(
            "SELECT exominer_score FROM candidates WHERE id=?", (cid_null,))
        assert row["exominer_score"] is None  # unchanged (has_spoc=0, not NULL)
        db.close()

    def test_does_not_overwrite_real_score(self, tmp_path):
        """A real score (>=0) on a has_spoc=NULL star must NOT be overwritten."""
        db = _make_db(tmp_path, "d.db")
        _add_star(db, "40", has_spoc=None)
        cid = _add_candidate(db, "40", cnn_score=0.9, exominer_score=0.82)

        n = db.mark_unconfirmed_ffi_candidates()
        assert n == 0

        row = db._query_one(
            "SELECT exominer_score FROM candidates WHERE id=?", (cid,))
        assert row["exominer_score"] == pytest.approx(0.82)
        db.close()

    def test_does_not_overwrite_existing_minus1(self, tmp_path):
        """Already-marked -1 rows (has_spoc=NULL) must not be updated again."""
        db = _make_db(tmp_path, "e.db")
        _add_star(db, "50", has_spoc=None)
        _add_candidate(db, "50", cnn_score=0.9, exominer_score=-1.0)

        n = db.mark_unconfirmed_ffi_candidates()
        assert n == 0
        db.close()

    def test_ignores_low_cnn(self, tmp_path):
        """Candidates with cnn_score < 0.7 are never in the backfill queue."""
        db = _make_db(tmp_path, "f.db")
        _add_star(db, "60", has_spoc=None)
        _add_candidate(db, "60", cnn_score=0.5)

        n = db.mark_unconfirmed_ffi_candidates()
        assert n == 0
        db.close()

    def test_idempotent(self, tmp_path):
        """Running twice must produce the same final state; second call returns 0."""
        db = _make_db(tmp_path, "g.db")
        _add_star(db, "70", has_spoc=None)
        _add_candidate(db, "70", cnn_score=0.9)

        n1 = db.mark_unconfirmed_ffi_candidates()
        assert n1 == 1

        n2 = db.mark_unconfirmed_ffi_candidates()
        assert n2 == 0, "Second call must be a no-op (IS NULL guard)"

        row = db._query_one(
            "SELECT exominer_score FROM candidates WHERE tic_id='70'")
        assert row["exominer_score"] == -1.0
        db.close()

    def test_mixed_stars(self, tmp_path):
        """Only has_spoc=NULL rows are marked; has_spoc=1 rows are untouched."""
        db = _make_db(tmp_path, "h.db")

        _add_star(db, "80", has_spoc=None)
        cid_null = _add_candidate(db, "80", cnn_score=0.85)

        _add_star(db, "81", has_spoc=1)
        cid_spoc = _add_candidate(db, "81", cnn_score=0.85)

        n = db.mark_unconfirmed_ffi_candidates()
        assert n == 1

        r_null = db._query_one(
            "SELECT exominer_score FROM candidates WHERE id=?", (cid_null,))
        assert r_null["exominer_score"] == -1.0

        r_spoc = db._query_one(
            "SELECT exominer_score FROM candidates WHERE id=?", (cid_spoc,))
        assert r_spoc["exominer_score"] is None
        db.close()


# ---------------------------------------------------------------------------
# 4. /api/admin/mark_unconfirmed_ffi endpoint
# ---------------------------------------------------------------------------

class TestMarkUnconfirmedFfiEndpoint:
    """POST /api/admin/mark_unconfirmed_ffi must return 200 + marked count."""

    def _build_app_and_db(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "ep.db"))
        monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
        os.makedirs(tmp_path / "reports", exist_ok=True)

        from fastapi.testclient import TestClient
        from exonit.server import create_app
        from exonit.db import ExoNitDB

        app = create_app()
        client = TestClient(app)
        db = ExoNitDB(str(tmp_path / "ep.db"))
        return client, db

    def test_endpoint_returns_200_and_count(self, tmp_path, monkeypatch):
        """Endpoint must return {"ok": True, "marked": N}."""
        client, db = self._build_app_and_db(tmp_path, monkeypatch)

        # Insert one has_spoc=NULL candidate that should be marked
        _add_star(db, "9001", has_spoc=None)
        _add_candidate(db, "9001", cnn_score=0.9)

        resp = client.post("/api/admin/mark_unconfirmed_ffi")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["marked"] == 1

        # Verify DB state
        row = db._query_one(
            "SELECT exominer_score FROM candidates WHERE tic_id='9001'")
        assert row["exominer_score"] == -1.0
        db.close()

    def test_endpoint_idempotent(self, tmp_path, monkeypatch):
        """Second call must return marked=0."""
        client, db = self._build_app_and_db(tmp_path, monkeypatch)
        _add_star(db, "9002", has_spoc=None)
        _add_candidate(db, "9002", cnn_score=0.9)

        client.post("/api/admin/mark_unconfirmed_ffi")
        resp2 = client.post("/api/admin/mark_unconfirmed_ffi")
        assert resp2.status_code == 200
        assert resp2.json()["marked"] == 0
        db.close()

    def test_endpoint_zero_when_no_unconfirmed(self, tmp_path, monkeypatch):
        """When all stars have has_spoc set, endpoint returns marked=0."""
        client, db = self._build_app_and_db(tmp_path, monkeypatch)
        _add_star(db, "9003", has_spoc=1)
        _add_candidate(db, "9003", cnn_score=0.9)

        resp = client.post("/api/admin/mark_unconfirmed_ffi")
        assert resp.status_code == 200
        assert resp.json()["marked"] == 0
        db.close()
