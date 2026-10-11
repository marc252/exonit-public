"""Tests for scripts/rebuild_work_streams.py::_derive_fpp.

Review follow-up (2026-07-11, progress/review-orgate.md item 1): this
function's own docstring/module docstring claims it uses "the SAME WHERE
clauses" as the legacy FPP scan / exonit.work_producers.fpp_eligible — it
was left on cnn_score>=0.7 alone (missed by the original OR-gate change).
This is the reconciliation path for writers that bypass the work-stream
producer hooks entirely (see the module docstring), so an em_ffi-rescued
candidate whose enqueue was missed must still be recoverable by a rebuild —
that is exactly the scenario these tests exercise.

Mirrors the fixture/assertion pattern of tests/test_fpp_task_query_index.py
(equivalence of the INDEXED BY arms vs the exception-fallback broad query)
and tests/test_work_producers.py (OR-gate monotonicity).
"""
from __future__ import annotations

import pytest

from exonit.db import ExoNitDB
from rebuild_work_streams import _derive_exominer, _derive_fpp


def _seed_star(db, tic_id, status="completed", has_spoc=None):
    db.add_star(tic_id, ra=0.0, dec=0.0, tmag=11.0, n_sectors=1,
               n_datapoints=100, time_span_days=27.0, status=status)
    if has_spoc is not None:
        db.set_star_has_spoc(tic_id, has_spoc)


def _seed_candidate(db, tic_id, snr, cnn_score, fpp=None, nfpp=None,
                    exominer_ffi_score=None, exominer_score=None):
    cid = db.add_candidate(
        tic_id, candidate_num=1, period_days=2.0, t0=1.0, duration_days=0.1,
        depth_ppm=500.0, snr=snr, cnn_score=cnn_score,
        triceratops_fpp=fpp, triceratops_nfpp=nfpp,
        exominer_ffi_score=exominer_ffi_score)
    if exominer_score is not None:
        db.update_candidate_exominer(cid, score=exominer_score)
    return cid


@pytest.fixture()
def db(tmp_path):
    database = ExoNitDB(str(tmp_path / "test.db"))
    yield database
    database.close()


class TestDeriveFppOrGate:
    def test_cnn_only_candidate_still_derived(self, db):
        """Monotonicity: a row that already passed the OLD gate
        (cnn>=0.7) must still be derived after the OR-gate fix."""
        _seed_star(db, "T1")
        _seed_candidate(db, "T1", snr=15.0, cnn_score=0.9, fpp=None)

        rows = _derive_fpp(db)
        tics = {r[1]["tic_id"] for r in rows}
        assert "T1" in tics

    def test_em_ffi_rescues_low_cnn_arm_a(self, db):
        """cnn<0.7 alone would fail; exominer_ffi_score>=0.5 must rescue it
        into arm A (fpp IS NULL)."""
        _seed_star(db, "T2")
        _seed_candidate(db, "T2", snr=12.0, cnn_score=0.4, fpp=None,
                        exominer_ffi_score=0.6)

        rows = _derive_fpp(db)
        tics = {r[1]["tic_id"] for r in rows}
        assert "T2" in tics, (
            "em_ffi-rescued candidate (cnn<0.7, exominer_ffi_score>=0.5) "
            "must be recoverable by the rebuild"
        )

    def test_em_ffi_rescues_low_cnn_arm_b(self, db):
        """Same rescue, arm B (low fpp, nfpp pending)."""
        _seed_star(db, "T3")
        _seed_candidate(db, "T3", snr=12.0, cnn_score=0.4, fpp=0.2,
                        nfpp=None, exominer_ffi_score=0.55)

        rows = _derive_fpp(db)
        tics = {r[1]["tic_id"] for r in rows}
        assert "T3" in tics

    def test_low_cnn_low_em_ffi_excluded(self, db):
        """Neither side of the OR-gate fires -> must not be derived."""
        _seed_star(db, "T4")
        _seed_candidate(db, "T4", snr=12.0, cnn_score=0.4, fpp=None,
                        exominer_ffi_score=0.3)

        rows = _derive_fpp(db)
        tics = {r[1]["tic_id"] for r in rows}
        assert "T4" not in tics

    def test_low_cnn_null_em_ffi_excluded_and_no_crash(self, db):
        """NULL exominer_ffi_score must never crash the query and must not
        count as passing."""
        _seed_star(db, "T5")
        _seed_candidate(db, "T5", snr=12.0, cnn_score=0.4, fpp=None,
                        exominer_ffi_score=None)

        rows = _derive_fpp(db)  # must not raise
        tics = {r[1]["tic_id"] for r in rows}
        assert "T5" not in tics

    def test_snr_floor_still_enforced_regardless_of_em_ffi(self, db):
        """The OR-gate only widens the cnn/em_ffi side; snr>=10 is still a
        hard floor even with a high em_ffi score."""
        _seed_star(db, "T6")
        _seed_candidate(db, "T6", snr=5.0, cnn_score=0.9, fpp=None,
                        exominer_ffi_score=0.9)

        rows = _derive_fpp(db)
        tics = {r[1]["tic_id"] for r in rows}
        assert "T6" not in tics

    def test_fallback_broad_query_matches_indexed_arms(self, db, monkeypatch):
        """The exception-fallback broad query (no INDEXED BY) must return
        the exact same eligible set as the indexed-arm path, both under the
        OR-gate — mirrors the existing indexed-vs-fallback equivalence
        guarantee the function's own docstring claims."""
        _seed_star(db, "T7")
        _seed_candidate(db, "T7", snr=12.0, cnn_score=0.4, fpp=None,
                        exominer_ffi_score=0.6)
        _seed_star(db, "T8")
        _seed_candidate(db, "T8", snr=15.0, cnn_score=0.9, fpp=0.2,
                        nfpp=None)

        indexed_rows = _derive_fpp(db)
        indexed_tics = {r[1]["tic_id"] for r in indexed_rows}

        # Force the fallback path: drop one of the partial indexes so
        # INDEXED BY raises and _derive_fpp falls back to the broad query.
        db.conn.execute("DROP INDEX IF EXISTS idx_cand_fpp_todo")
        db.conn.commit()

        fallback_rows = _derive_fpp(db)
        fallback_tics = {r[1]["tic_id"] for r in fallback_rows}

        assert indexed_tics == fallback_tics == {"T7", "T8"}

    def test_derive_exominer_unchanged_docker_spoc_path(self, db):
        """Item 6 of the original task (still true after this fix):
        _derive_exominer (Docker-SPOC ExoMiner backfill) stays on
        cnn_score>=0.7 alone — a low-cnn/high-em_ffi candidate must NOT be
        derived by it, even though it would now be derived by _derive_fpp."""
        _seed_star(db, "T9", has_spoc=1)
        _seed_candidate(db, "T9", snr=12.0, cnn_score=0.4,
                        exominer_ffi_score=0.9, exominer_score=None)

        rows = _derive_exominer(db)
        tics = {r[1]["tic_id"] for r in rows}
        assert "T9" not in tics, (
            "Docker-SPOC exominer rebuild must stay cnn_score>=0.7-only "
            "(deliberately unchanged, out of scope for the OR-gate)"
        )
