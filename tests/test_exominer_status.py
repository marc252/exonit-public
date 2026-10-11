"""Tests for candidates.exominer_status (docs/superpowers/specs/
exominer-status-column.md — "Step 2: separate status from score").

Covers:
  - derive_exominer_status(): the pure derivation rule, including the
    2026-07-14 production-audit regression (float32 rounding artifact
    -5.960464477539063e-08 must classify as 'scored', not a sentinel).
  - SQLite auto-migration: column + idx_cand_exominer_status_todo present
    and idempotent across a double ExoNitDB() init.
  - Each live write point (mark_ffi_only_candidates, mark_spoc_no_dv_
    candidates, clear_spurious_minus_one_candidates, mark_unconfirmed_ffi_
    candidates, update_candidate_exominer) leaves status and score
    COHERENT per the spec's invariant table.
  - count_status_score_violations(): the health-check invariant, including
    that it does NOT flag the -5.96e-08 edge case as a violation.
  - The PARITY test: selecting the ExoMiner backfill queue via
    exominer_status returns EXACTLY the same candidate set as the old
    exominer_score IS NULL AND has_spoc=1 predicate, across all 5 states.
"""

from exonit.db import ExoNitDB, derive_exominer_status
from exonit.sentinel_consistency import count_status_score_violations

# The exact production float32 rounding artifact the coordinator flagged
# (2026-07-13 audit: 51 candidates in production carry this exact value).
NEGATIVE_ZERO_ARTIFACT = -5.960464477539063e-08


def _make_db(tmp_path, name="test.db"):
    return ExoNitDB(str(tmp_path / name))


def _add_star_with_candidates(db, tic_id, candidates, has_spoc=None):
    db.add_star(
        tic_id, ra=10.0, dec=-20.0, tmag=11.0,
        n_sectors=1, n_datapoints=1000, time_span_days=27.0,
        status="completed",
    )
    if has_spoc is not None:
        db.set_star_has_spoc(tic_id, has_spoc)
    ids = []
    for i, c in enumerate(candidates, start=1):
        cid = db.add_candidate(
            tic_id, candidate_num=i,
            period_days=c.get("period_days", 3.0), t0=0.0,
            duration_days=0.1, depth_ppm=1000,
            snr=c.get("snr", 12.0), cnn_score=c.get("cnn_score", 0.8),
        )
        if c.get("exominer_score") is not None:
            db.update_candidate_exominer(cid, score=c["exominer_score"])
        ids.append(cid)
    return ids


# --------------------------------------------------------------------------- #
# 1. derive_exominer_status — pure function
# --------------------------------------------------------------------------- #

class TestDeriveExominerStatus:

    def test_none_score_is_pending(self):
        assert derive_exominer_status(None) == "pending"
        assert derive_exominer_status(None, has_spoc=1) == "pending"
        assert derive_exominer_status(None, has_spoc=0) == "pending"

    def test_minus_one_with_has_spoc_1_is_no_dv(self):
        assert derive_exominer_status(-1, has_spoc=1) == "no_dv"

    def test_minus_one_with_has_spoc_0_is_no_spoc(self):
        assert derive_exominer_status(-1, has_spoc=0) == "no_spoc"

    def test_minus_one_with_has_spoc_none_is_no_spoc(self):
        """has_spoc unresolved (None) buckets with has_spoc=0 under no_spoc —
        both mean "not confirmed has-SPOC"."""
        assert derive_exominer_status(-1, has_spoc=None) == "no_spoc"

    def test_real_score_is_scored(self):
        assert derive_exominer_status(0.87) == "scored"
        assert derive_exominer_status(0.0) == "scored"
        assert derive_exominer_status(1.0) == "scored"

    def test_negative_zero_float32_artifact_is_scored_not_sentinel(self):
        """Regression (2026-07-13 production audit, 51 candidates): a score
        that rounds to a tiny negative number due to float32 ONNX output is
        a REAL score (~0.0 probability), not the -1 sentinel. A naive
        `score >= 0` check would misclassify this as neither scored nor a
        sentinel — the derivation must treat -1 as the ONLY special value."""
        assert derive_exominer_status(NEGATIVE_ZERO_ARTIFACT) == "scored"
        assert derive_exominer_status(NEGATIVE_ZERO_ARTIFACT, has_spoc=1) == "scored"
        assert derive_exominer_status(NEGATIVE_ZERO_ARTIFACT, has_spoc=None) == "scored"

    def test_other_negative_values_are_scored_not_sentinel(self):
        """Only score == -1 EXACTLY is the sentinel; any other negative
        value (even close to -1) is still a real score."""
        assert derive_exominer_status(-0.5) == "scored"
        assert derive_exominer_status(-0.999999) == "scored"


# --------------------------------------------------------------------------- #
# 2. SQLite auto-migration
# --------------------------------------------------------------------------- #

class TestMigrationColumn:

    def test_column_exists_after_init(self, tmp_path):
        db = _make_db(tmp_path)
        db.add_star("999", ra=1.0, dec=1.0, tmag=10.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)
        cid = db.add_candidate("999", candidate_num=1, period_days=3.0, t0=0.0,
                                duration_days=0.1, depth_ppm=1000, snr=10.0)
        row = db._query_one("SELECT exominer_status FROM candidates WHERE id=?", (cid,))
        assert row is not None
        assert "exominer_status" in row
        # The column is additive (no SQL DEFAULT), but add_candidate() writes
        # 'pending' explicitly at INSERT — a fresh candidate is queue-eligible
        # and must say so, or the status-based selector strands it (2026-07-14
        # review finding). The "no DEFAULT" property is what makes the column
        # safe to add to a live 13M-row table; it is not a claim about inserts.
        assert row["exominer_status"] == "pending"
        db.close()

    def test_double_init_is_idempotent(self, tmp_path):
        db1 = _make_db(tmp_path)
        db1.close()
        db2 = _make_db(tmp_path)  # re-runs ALTER TABLE + CREATE INDEX
        db2.close()

    def test_index_created(self, tmp_path):
        db = _make_db(tmp_path)
        row = db._query_one(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name='idx_cand_exominer_status_todo'")
        assert row is not None
        db.close()


# --------------------------------------------------------------------------- #
# 3. Write-point coherence — each of the 5 write points keeps status+score
#    in sync per the spec's invariant table.
# --------------------------------------------------------------------------- #

class TestMarkFfiOnlyCandidatesStatus:

    def test_sets_no_spoc(self, tmp_path):
        db = _make_db(tmp_path)
        _add_star_with_candidates(db, "100", [
            {"cnn_score": 0.9, "exominer_score": None},
        ])
        db.mark_ffi_only_candidates("100")
        cands = db.get_candidates("100")
        assert cands[0]["exominer_score"] == -1
        assert cands[0]["exominer_status"] == "no_spoc"
        db.close()


class TestMarkSpocNoDvCandidatesStatus:

    def test_sets_no_dv(self, tmp_path):
        db = _make_db(tmp_path)
        _add_star_with_candidates(db, "110", [
            {"cnn_score": 0.9, "exominer_score": None},
        ])
        db.mark_spoc_no_dv_candidates("110")
        cands = db.get_candidates("110")
        assert cands[0]["exominer_score"] == -1
        assert cands[0]["exominer_status"] == "no_dv"
        db.close()


class TestClearSpuriousMinusOneCandidatesStatus:

    def test_resets_to_pending(self, tmp_path):
        db = _make_db(tmp_path)
        _add_star_with_candidates(db, "120", [
            {"cnn_score": 0.9, "exominer_score": -1},
        ])
        db.clear_spurious_minus_one_candidates("120")
        cands = db.get_candidates("120")
        assert cands[0]["exominer_score"] is None
        assert cands[0]["exominer_status"] == "pending"
        db.close()


class TestMarkUnconfirmedFfiCandidatesStatus:

    def test_sets_no_spoc(self, tmp_path):
        db = _make_db(tmp_path)
        # has_spoc left NULL (unresolved) — the bulk sweep's own predicate.
        _add_star_with_candidates(db, "130", [
            {"cnn_score": 0.9, "exominer_score": None},
        ])
        n = db.mark_unconfirmed_ffi_candidates()
        assert n == 1
        cands = db.get_candidates("130")
        assert cands[0]["exominer_score"] == -1
        assert cands[0]["exominer_status"] == "no_spoc"
        db.close()


class TestUpdateCandidateExominerStatus:

    def test_real_score_sets_scored(self, tmp_path):
        db = _make_db(tmp_path)
        db.add_star("140", ra=1.0, dec=1.0, tmag=10.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)
        cid = db.add_candidate("140", candidate_num=1, period_days=3.0, t0=0.0,
                                duration_days=0.1, depth_ppm=1000, snr=10.0)
        db.update_candidate_exominer(cid, score=0.91, tce_period=3.0,
                                     sector_run="s0014")
        cands = db.get_candidates("140")
        assert cands[0]["exominer_score"] == 0.91
        assert cands[0]["exominer_status"] == "scored"
        db.close()

    def test_negative_zero_artifact_sets_scored(self, tmp_path):
        """The exact regression case: a real ExoMiner call that returns the
        float32 rounding artifact must NOT be misread as the -1 sentinel."""
        db = _make_db(tmp_path)
        db.add_star("141", ra=1.0, dec=1.0, tmag=10.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)
        cid = db.add_candidate("141", candidate_num=1, period_days=3.0, t0=0.0,
                                duration_days=0.1, depth_ppm=1000, snr=10.0)
        db.update_candidate_exominer(cid, score=NEGATIVE_ZERO_ARTIFACT)
        cands = db.get_candidates("141")
        assert cands[0]["exominer_score"] == NEGATIVE_ZERO_ARTIFACT
        assert cands[0]["exominer_status"] == "scored"
        db.close()

    def test_none_score_sets_pending(self, tmp_path):
        db = _make_db(tmp_path)
        db.add_star("142", ra=1.0, dec=1.0, tmag=10.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)
        cid = db.add_candidate("142", candidate_num=1, period_days=3.0, t0=0.0,
                                duration_days=0.1, depth_ppm=1000, snr=10.0)
        db.update_candidate_exominer(cid, score=None)
        cands = db.get_candidates("142")
        assert cands[0]["exominer_status"] == "pending"
        db.close()


# --------------------------------------------------------------------------- #
# 4. count_status_score_violations — health check invariant
# --------------------------------------------------------------------------- #

class TestCountStatusScoreViolations:

    def test_healthy_db_has_zero_violations(self, tmp_path):
        db = _make_db(tmp_path)
        _add_star_with_candidates(db, "200", [{"cnn_score": 0.9, "exominer_score": None}])
        db.mark_ffi_only_candidates("200")  # score=-1, status=no_spoc — coherent
        counts = count_status_score_violations(db)
        assert counts["total"] == 0
        db.close()

    def test_negative_zero_artifact_not_flagged(self, tmp_path):
        """The core regression this whole fix is about: a healthy 'scored'
        row with the float32 artifact score must NOT trip the health check."""
        db = _make_db(tmp_path)
        db.add_star("201", ra=1.0, dec=1.0, tmag=10.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)
        cid = db.add_candidate("201", candidate_num=1, period_days=3.0, t0=0.0,
                                duration_days=0.1, depth_ppm=1000, snr=10.0)
        db.update_candidate_exominer(cid, score=NEGATIVE_ZERO_ARTIFACT)
        counts = count_status_score_violations(db)
        assert counts["total"] == 0
        assert counts["bad_scored"] == 0
        db.close()

    def test_manually_broken_pending_is_flagged(self, tmp_path):
        db = _make_db(tmp_path)
        db.add_star("202", ra=1.0, dec=1.0, tmag=10.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)
        cid = db.add_candidate("202", candidate_num=1, period_days=3.0, t0=0.0,
                                duration_days=0.1, depth_ppm=1000, snr=10.0)
        # Deliberately corrupt: status says pending but score is real.
        db._exec("UPDATE candidates SET exominer_score=0.5, exominer_status='pending' "
                  "WHERE id=?", (cid,))
        counts = count_status_score_violations(db)
        assert counts["total"] == 1
        assert counts["bad_pending"] == 1
        db.close()

    def test_null_status_rows_excluded(self, tmp_path):
        """Rows not yet migrated (exominer_status IS NULL) are not counted
        as violations — that is a migration-completeness concern, not a
        coherence one."""
        db = _make_db(tmp_path)
        db.add_star("203", ra=1.0, dec=1.0, tmag=10.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)
        db.add_candidate("203", candidate_num=1, period_days=3.0, t0=0.0,
                          duration_days=0.1, depth_ppm=1000, snr=10.0, cnn_score=0.5)
        counts = count_status_score_violations(db)
        assert counts["total"] == 0
        db.close()


# --------------------------------------------------------------------------- #
# 5. PARITY — the most important test in the spec: selecting the backfill
#    queue via exominer_status must return EXACTLY the same set as the old
#    exominer_score IS NULL AND has_spoc=1 predicate.
# --------------------------------------------------------------------------- #

class TestQueueSelectionParity:
    """Seeds all 5 states via the LIVE write points (not the migration
    script — see tests/scripts/test_migrate_exominer_status.py for the
    migration-derived version of this same parity check) and asserts the
    old score-based queue predicate and the new status-based one agree."""

    def _old_queue_ids(self, db):
        rows = db._query(
            """SELECT c.id FROM candidates c JOIN stars s ON c.tic_id = s.tic_id
               WHERE c.exominer_score IS NULL AND c.cnn_score >= 0.7
                 AND s.has_spoc = 1 AND s.status = 'completed'""")
        return {r["id"] for r in rows}

    def _new_queue_ids(self, db):
        rows = db._query(
            """SELECT c.id FROM candidates c JOIN stars s ON c.tic_id = s.tic_id
               WHERE c.exominer_status = 'pending' AND c.cnn_score >= 0.7
                 AND s.has_spoc = 1 AND s.status = 'completed'""")
        return {r["id"] for r in rows}

    def test_parity_across_all_states(self, tmp_path):
        db = _make_db(tmp_path)

        # A: pending, queue-eligible has_spoc=1, cnn>=0.7 — IN the queue.
        # Seeded via the REALISTIC gate-retraction path (bulk sweep marks -1
        # while has_spoc is unresolved, gate later confirms has_spoc=1+DV
        # and retracts it via clear_spurious_minus_one_candidates) rather
        # than a bare INSERT, because a bare INSERT never touches
        # exominer_status at all — see
        # TestFreshInsertStatusGap below for that documented gap.
        ids_a = _add_star_with_candidates(
            db, "300", [{"cnn_score": 0.9, "exominer_score": -1}])
        db.set_star_has_spoc("300", 1)
        db.clear_spurious_minus_one_candidates("300")

        # B: pending but has_spoc=0 (FFI-only, low-cnn candidate never
        # marked) — cnn>=0.7 so score stays NULL forever; has_spoc excludes
        # it from the queue on BOTH selectors.
        ids_b = _add_star_with_candidates(
            db, "301", [{"cnn_score": 0.9, "exominer_score": None}], has_spoc=0)

        # C: pending, has_spoc still NULL (not yet gated) — excluded from
        # BOTH selectors (has_spoc != 1).
        ids_c = _add_star_with_candidates(
            db, "302", [{"cnn_score": 0.9, "exominer_score": None}], has_spoc=None)

        # D: no_dv (has SPOC, no DV report) — excluded from both.
        db_star_d = _add_star_with_candidates(
            db, "303", [{"cnn_score": 0.9, "exominer_score": None}], has_spoc=1)
        db.mark_spoc_no_dv_candidates("303")

        # E: no_spoc (FFI-only, confirmed) — excluded from both.
        _add_star_with_candidates(
            db, "304", [{"cnn_score": 0.9, "exominer_score": None}], has_spoc=0)
        db.mark_ffi_only_candidates("304")

        # F: scored (real score) — excluded from both.
        _add_star_with_candidates(
            db, "305", [{"cnn_score": 0.9, "exominer_score": 0.72}], has_spoc=1)

        # G: scored via the float32 negative-zero artifact — excluded from
        # both (this is the exact case the corrected derivation must get
        # right: NOT accidentally left as 'pending').
        _add_star_with_candidates(
            db, "306", [{"cnn_score": 0.9, "exominer_score": NEGATIVE_ZERO_ARTIFACT}],
            has_spoc=1)

        # H: pending, has_spoc=1, but cnn < 0.7 — excluded from both (cnn gate).
        _add_star_with_candidates(
            db, "307", [{"cnn_score": 0.5, "exominer_score": None}], has_spoc=1)

        old_ids = self._old_queue_ids(db)
        new_ids = self._new_queue_ids(db)

        assert old_ids == new_ids, (
            f"parity broken: old-only={old_ids - new_ids}, "
            f"new-only={new_ids - old_ids}")
        # Sanity: exactly candidate A is queue-eligible.
        assert old_ids == set(ids_a)
        db.close()

    def test_parity_after_dv_retraction(self, tmp_path):
        """A candidate the bulk sweep marked -1 while has_spoc was NULL,
        later confirmed has_spoc=1 with real DV — clear_spurious_minus_one_
        candidates retracts it. It must re-appear in BOTH selectors."""
        db = _make_db(tmp_path)
        ids = _add_star_with_candidates(
            db, "310", [{"cnn_score": 0.9, "exominer_score": -1}])
        db.set_star_has_spoc("310", 1)

        # Before retraction: excluded from both (score=-1, whatever status
        # the seed left it at).
        assert set(ids) - self._old_queue_ids(db) == set(ids)
        assert set(ids) - self._new_queue_ids(db) == set(ids)

        db.clear_spurious_minus_one_candidates("310")

        assert self._old_queue_ids(db) == set(ids)
        assert self._new_queue_ids(db) == set(ids)
        db.close()


class TestFreshInsertIsPending:
    """A brand-new candidate must be born 'pending' — NOT NULL.

    This class used to assert the OPPOSITE (that add_candidate left status
    NULL), documenting it as a harmless gap. The 2026-07-14 review proved it
    was not harmless: the row stayed NULL *forever*, because the SPOC gate's
    success path (has_spoc=1 + DV present) only calls
    clear_spurious_minus_one_candidates, which is a no-op on a row that never
    carried a -1. So the old score-based selector kept queueing the candidate
    while a status-based selector would never see it — the parity break the
    whole migration exists to prevent, on the DOMINANT path.

    add_candidate() now writes 'pending' at INSERT, which is exactly what
    derive_exominer_status(None) yields: status and score agree by
    construction from the first write."""

    def test_fresh_insert_is_pending_and_both_selectors_agree(self, tmp_path):
        db = _make_db(tmp_path)
        db.add_star("320", ra=1, dec=1, tmag=10, n_sectors=1, n_datapoints=100,
                    time_span_days=27.0, status="completed")
        db.set_star_has_spoc("320", 1)
        cid = db.add_candidate("320", candidate_num=1, period_days=3.0, t0=0.0,
                                duration_days=0.1, depth_ppm=1000, snr=10.0,
                                cnn_score=0.9)

        row = db.get_candidates("320")[0]
        assert row["exominer_score"] is None
        assert row["exominer_status"] == "pending"

        selector = TestQueueSelectionParity()
        assert cid in selector._old_queue_ids(db)
        assert cid in selector._new_queue_ids(db), (
            "a fresh candidate must be visible to the status-based selector "
            "too — otherwise it is stranded out of the ExoMiner queue forever")
        db.close()


class TestFreshCandidateParityOnGateSuccessPath:
    """The gate's SUCCESS path must not strand a fresh candidate at status NULL.

    Regression for the review finding of 2026-07-14. A fresh candidate is
    inserted with exominer_score IS NULL (queue-eligible). When the SPOC gate
    resolves it the COMMON, successful way — has_spoc=1 and a DV report exists —
    it calls clear_spurious_minus_one_candidates(), whose `WHERE
    exominer_score = -1` is a NO-OP on a row that never carried a sentinel.

    Before the fix, add_candidate() left exominer_status NULL, so the row stayed
    NULL forever: the OLD (score-based) selector still queued it, the NEW
    (status-based) selector never saw it. Parity broken on the dominant path —
    exactly what this migration exists to prevent. The fix: add_candidate()
    writes 'pending' at INSERT.
    """

    def test_fresh_candidate_is_pending_and_stays_queue_eligible(self, tmp_path):
        db = ExoNitDB(str(tmp_path / "t.db"))
        db.add_star("111000111", ra=1.0, dec=1.0, tmag=10.0, n_sectors=1,
                    n_datapoints=600, time_span_days=26.0, status="completed")
        db._exec("UPDATE stars SET has_spoc = 1 WHERE tic_id = ?", ("111000111",))
        cid = db.add_candidate(
            tic_id="111000111", candidate_num=1, period_days=3.5, t0=1.0,
            duration_days=0.1, depth_ppm=1000, snr=15.0, cnn_score=0.9,
            sector=14)

        # 1) fresh insert: score NULL (eligible) AND status 'pending' — coherent
        row = db._query_one(
            "SELECT exominer_score, exominer_status FROM candidates WHERE id = ?",
            (cid,))
        assert row["exominer_score"] is None
        assert row["exominer_status"] == "pending"

        # 2) the gate resolves it the SUCCESSFUL way (has_spoc + DV present):
        #    the retraction call is a no-op because no -1 was ever written.
        db.clear_spurious_minus_one_candidates("111000111")

        row = db._query_one(
            "SELECT exominer_score, exominer_status FROM candidates WHERE id = ?",
            (cid,))
        assert row["exominer_score"] is None
        assert row["exominer_status"] == "pending", (
            "fresh candidate stranded at status NULL after the gate's success "
            "path — the status-based queue selector would never see it")

        # 3) PARITY: old score-based selector and new status-based selector must
        #    agree on this row.
        old = db._query_scalar(
            """SELECT COUNT(*) FROM candidates c JOIN stars s ON c.tic_id = s.tic_id
               WHERE c.exominer_score IS NULL AND c.cnn_score >= 0.7
                 AND s.has_spoc = 1""")
        new = db._query_scalar(
            """SELECT COUNT(*) FROM candidates c JOIN stars s ON c.tic_id = s.tic_id
               WHERE c.exominer_status = 'pending' AND c.cnn_score >= 0.7
                 AND s.has_spoc = 1""")
        assert old == new == 1
