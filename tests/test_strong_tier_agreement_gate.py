"""Test strong-tier agreement gate (Task 8, Phase 3).

Verifies that db.strong_candidates() correctly excludes rows where
exominer_score and exominer_ffi_score disagree (one >= 0.7, other < threshold).
"""
import os

import pytest

from exonit.db import ExoNitDB
from tests._pg_test_db import pg_test_available, resolve_test_dsn


def _seed(db, tic, em, em_ffi, cnn_score=0.85):
    """Seed a star + one candidate with the given ExoMiner scores."""
    db.add_star(tic, ra=0.0, dec=0.0, tmag=11.0,
                n_sectors=1, n_datapoints=100, time_span_days=27.0)
    cid = db.add_candidate(
        tic, candidate_num=1,
        period_days=2.0, t0=1.0, duration_days=0.1,
        depth_ppm=1000.0, snr=12.0,
        cnn_score=cnn_score,
        triceratops_fpp=0.10,
        exominer_ffi_score=em_ffi,
    )
    # exominer_score is set via a separate update call (mirrors production path)
    if em is not None:
        db.update_candidate_exominer(cid, score=em)


def test_agreement_gate_filters_disagreement(tmp_path, monkeypatch):
    monkeypatch.setenv("EXONIT_PHASE3_AGREEMENT_THRESHOLD", "0.6")
    db = ExoNitDB(str(tmp_path / "test.db"))

    _seed(db, "agree",        em=0.85, em_ffi=0.80)  # both pass: keep
    _seed(db, "disagree",     em=0.85, em_ffi=0.50)  # disagree: drop
    _seed(db, "spoc_only",    em=0.85, em_ffi=None)   # one source: keep
    _seed(db, "ffi_only",     em=None, em_ffi=0.85)   # one source: keep
    _seed(db, "neither_pass", em=0.50, em_ffi=0.85)   # base em gate fails: drop

    rows = db.strong_candidates()
    tics = {r["tic_id"] for r in rows}
    assert tics == {"agree", "spoc_only", "ffi_only"}, (
        f"expected agreement gate to filter disagree+neither_pass; got {tics}"
    )


def test_endpoint_filter_strong_applies_agreement_gate(tmp_path, monkeypatch):
    """The /api/candidates?filter=strong endpoint must apply the agreement gate."""
    from fastapi.testclient import TestClient
    from exonit.server import create_app

    monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EXONIT_PHASE3_AGREEMENT_THRESHOLD", "0.6")

    app = create_app()
    client = TestClient(app)

    # Seed via the DB the endpoint uses
    db = ExoNitDB(str(tmp_path / "test.db"))
    _seed(db, "agree", em=0.85, em_ffi=0.80)
    _seed(db, "disagree", em=0.85, em_ffi=0.50)
    # Both have snr=12.0 (>= 10) from _seed, satisfying the endpoint's SNR filter

    r = client.get("/api/candidates?filter=strong",
                   headers={"Authorization": "Bearer exonit-default-token"})
    assert r.status_code == 200
    payload = r.json()
    tics = {c["tic_id"] for c in payload["candidates"]}
    assert "agree" in tics, f"agree should pass; got {tics}"
    assert "disagree" not in tics, f"disagree should be filtered by agreement gate; got {tics}"


def test_count_strong_candidates_matches_strong_candidates_len(tmp_path, monkeypatch):
    """D1: count_strong_candidates() and strong_candidates() now share the
    same _strong_where() builder — their results must stay in lockstep."""
    monkeypatch.setenv("EXONIT_PHASE3_AGREEMENT_THRESHOLD", "0.6")
    db = ExoNitDB(str(tmp_path / "test.db"))

    _seed(db, "agree",        em=0.85, em_ffi=0.80)
    _seed(db, "disagree",     em=0.85, em_ffi=0.50)
    _seed(db, "spoc_only",    em=0.85, em_ffi=None)
    _seed(db, "ffi_only",     em=None, em_ffi=0.85)
    _seed(db, "neither_pass", em=0.50, em_ffi=0.85)

    assert db.count_strong_candidates() == len(db.strong_candidates())

    # Same with the min_snr filter path (a separate branch of _strong_where).
    assert db.count_strong_candidates(min_snr=10.0) == \
        len(db.strong_candidates(min_snr=10.0))
    assert db.count_strong_candidates(min_snr=1000.0) == \
        len(db.strong_candidates(min_snr=1000.0)) == 0


def test_strong_where_matches_server_endpoint_tic_set(tmp_path, monkeypatch):
    """D1 regression guard: db._strong_where()'s predicate (used by
    strong_candidates/count_strong_candidates) must keep returning the same
    TIC set as the still-independent copy in server.py's
    /api/candidates?filter=strong (server.py is out of scope for this change;
    a follow-up repoints it at _strong_where — this test would catch any
    accidental semantic drift introduced meanwhile)."""
    from fastapi.testclient import TestClient
    from exonit.server import create_app

    monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EXONIT_PHASE3_AGREEMENT_THRESHOLD", "0.6")

    app = create_app()
    client = TestClient(app)

    db = ExoNitDB(str(tmp_path / "test.db"))
    _seed(db, "agree",        em=0.85, em_ffi=0.80)
    _seed(db, "disagree",     em=0.85, em_ffi=0.50)
    _seed(db, "spoc_only",    em=0.85, em_ffi=None)
    _seed(db, "ffi_only",     em=None, em_ffi=0.85)
    _seed(db, "neither_pass", em=0.50, em_ffi=0.85)
    # All seeded candidates have snr=12.0 (>= 10), matching the endpoint's gate.

    r = client.get("/api/candidates?filter=strong",
                   headers={"Authorization": "Bearer exonit-default-token"})
    assert r.status_code == 200
    endpoint_tics = {c["tic_id"] for c in r.json()["candidates"]}

    db_tics = {r["tic_id"] for r in db.strong_candidates(min_snr=10.0)}
    assert endpoint_tics == db_tics


def test_strong_tier_keeps_fpp_null_candidates(tmp_path, monkeypatch):
    """Candidates with NULL triceratops_fpp must remain in strong tier
    (matches the project's pattern at server.py 473-475, 685-687, 1250-1253)."""
    monkeypatch.setenv("EXONIT_PHASE3_AGREEMENT_THRESHOLD", "0.6")
    db = ExoNitDB(str(tmp_path / "test.db"))

    # Seed a candidate with FPP=None (pending)
    db.add_star("fpp_pending", ra=0.0, dec=0.0, tmag=11.0,
                n_sectors=1, n_datapoints=100, time_span_days=27.0)
    cid = db.add_candidate(
        "fpp_pending", candidate_num=1,
        period_days=2.0, t0=1.0, duration_days=0.1,
        depth_ppm=1000.0, snr=12.0,
        cnn_score=0.85, triceratops_fpp=None,  # FPP not computed yet
        exominer_ffi_score=0.80,
    )
    db.update_candidate_exominer(cid, score=0.85)

    rows = db.strong_candidates()
    tics = {r["tic_id"] for r in rows}
    assert "fpp_pending" in tics, \
        f"FPP-NULL candidates should be kept; got {tics}"


# --------------------------------------------------------------------------- #
# OR-gate (2026-07-11, docs/reviews/injection_e2e_results_2026-07-11.md +
# docs/reviews/ffi_cnn_investigation_2026-07-11.md §5): _strong_where's
# clause 1 changed from cnn_score>=0.7 alone to
# (cnn_score>=0.7 OR exominer_ffi_score>=0.5), so em_ffi-rescued candidates
# that reach claude_verdict='planet_candidate' become visible in the display
# tier too — see work_producers.verdict_eligible for the shared predicate.
# --------------------------------------------------------------------------- #

def test_strong_where_em_ffi_rescues_low_cnn(tmp_path, monkeypatch):
    """cnn<0.7 alone would have failed clause 1 pre-OR-gate; a high
    exominer_ffi_score (>=0.5 for eligibility, and here also >=0.7 so it
    ALSO carries clause 3's scoring threshold) now rescues it into the
    strong tier."""
    monkeypatch.setenv("EXONIT_PHASE3_AGREEMENT_THRESHOLD", "0.6")
    db = ExoNitDB(str(tmp_path / "test.db"))

    _seed(db, "cnn_low_ffi_rescue", em=None, em_ffi=0.80, cnn_score=0.4)

    rows = db.strong_candidates()
    tics = {r["tic_id"] for r in rows}
    assert "cnn_low_ffi_rescue" in tics, (
        f"cnn<0.7 but exominer_ffi_score>=0.5 (and >=0.7 here) must be "
        f"rescued into the strong tier; got {tics}"
    )
    assert db.count_strong_candidates() == len(db.strong_candidates())


def test_strong_where_low_cnn_low_em_ffi_still_excluded(tmp_path, monkeypatch):
    """Neither side of the OR-gate fires -> still excluded (no regression:
    the OR-gate only ADDS eligible rows, it does not relax anything else)."""
    monkeypatch.setenv("EXONIT_PHASE3_AGREEMENT_THRESHOLD", "0.6")
    db = ExoNitDB(str(tmp_path / "test.db"))

    _seed(db, "cnn_low_ffi_low", em=None, em_ffi=0.3, cnn_score=0.4)
    _seed(db, "cnn_low_ffi_null", em=None, em_ffi=None, cnn_score=0.4)

    tics = {r["tic_id"] for r in db.strong_candidates()}
    assert "cnn_low_ffi_low" not in tics
    assert "cnn_low_ffi_null" not in tics


def test_strong_where_monotonic_cnn_only_candidate_still_passes(tmp_path, monkeypatch):
    """Regression guard: a candidate that already passed the OLD
    cnn_score>=0.7-only predicate must still pass after the OR-gate change
    (monotonicity, mirrors work_producers' verdict_eligible property test)."""
    monkeypatch.setenv("EXONIT_PHASE3_AGREEMENT_THRESHOLD", "0.6")
    db = ExoNitDB(str(tmp_path / "test.db"))

    _seed(db, "cnn_pass_no_ffi", em=0.85, em_ffi=None, cnn_score=0.85)

    tics = {r["tic_id"] for r in db.strong_candidates()}
    assert "cnn_pass_no_ffi" in tics


@pytest.mark.skipif(
    not pg_test_available(resolve_test_dsn()),
    reason="EXONIT_TEST_DATABASE_URL (throwaway PG) unset or not reachable",
)
def test_strong_where_em_ffi_rescue_on_postgres(tmp_path, monkeypatch):
    """Same em_ffi rescue as test_strong_where_em_ffi_rescues_low_cnn, on
    the PG dialect."""
    monkeypatch.setenv("EXONIT_PHASE3_AGREEMENT_THRESHOLD", "0.6")
    dsn = resolve_test_dsn()
    _pg_truncate(dsn)
    db = ExoNitDB(dsn, completeness_path=str(tmp_path / "cmpl.db"))
    assert db.dialect.name == "postgres"
    try:
        _seed(db, "pg_cnn_low_ffi_rescue", em=None, em_ffi=0.80, cnn_score=0.4)
        tics = {r["tic_id"] for r in db.strong_candidates()}
        assert tics == {"pg_cnn_low_ffi_rescue"}
        assert db.count_strong_candidates() == 1
    finally:
        db.close()
        _pg_truncate(dsn)


# --------------------------------------------------------------------------- #
# exominer_score=-1 read-boundary fix (2026-07-02): -1 is a no-DV cache-
# negative sentinel, not a real score, and must read as "absent" (falls back
# to exominer_ffi_score the same way a real NULL exominer_score already
# does). See progress/investigation-exominer-minus1-2026-07-02.md +
# progress/impl-minus1-fix-2026-07-02.md.
# --------------------------------------------------------------------------- #

def test_minus1_sentinel_falls_back_to_ffi_when_ffi_passes(tmp_path, monkeypatch):
    """exominer_score=-1 + exominer_ffi_score>=0.7 -> strong (FFI carries it)."""
    monkeypatch.setenv("EXONIT_PHASE3_AGREEMENT_THRESHOLD", "0.6")
    db = ExoNitDB(str(tmp_path / "test.db"))

    _seed(db, "neg1_ffi_pass", em=-1, em_ffi=0.85)

    rows = db.strong_candidates()
    tics = {r["tic_id"] for r in rows}
    assert "neg1_ffi_pass" in tics, (
        f"exominer_score=-1 with a passing FFI score should be strong "
        f"(same as exominer_score=None); got {tics}"
    )
    assert db.count_strong_candidates() == 1


def test_minus1_sentinel_not_strong_when_ffi_below_threshold(tmp_path, monkeypatch):
    """exominer_score=-1 + exominer_ffi_score<0.7 -> not strong."""
    monkeypatch.setenv("EXONIT_PHASE3_AGREEMENT_THRESHOLD", "0.6")
    db = ExoNitDB(str(tmp_path / "test.db"))

    _seed(db, "neg1_ffi_fail", em=-1, em_ffi=0.50)

    rows = db.strong_candidates()
    tics = {r["tic_id"] for r in rows}
    assert "neg1_ffi_fail" not in tics, (
        f"exominer_score=-1 with a sub-threshold FFI score must stay "
        f"excluded; got {tics}"
    )
    assert db.count_strong_candidates() == 0


def test_minus1_sentinel_not_strong_when_ffi_null(tmp_path, monkeypatch):
    """exominer_score=-1 + exominer_ffi_score=NULL -> not strong (no real
    score at all — must NOT be misread as a passing -1 "score")."""
    monkeypatch.setenv("EXONIT_PHASE3_AGREEMENT_THRESHOLD", "0.6")
    db = ExoNitDB(str(tmp_path / "test.db"))

    _seed(db, "neg1_ffi_null", em=-1, em_ffi=None)

    rows = db.strong_candidates()
    tics = {r["tic_id"] for r in rows}
    assert "neg1_ffi_null" not in tics, (
        f"exominer_score=-1 with no FFI score must stay excluded; got {tics}"
    )
    assert db.count_strong_candidates() == 0


def test_minus1_sentinel_and_none_produce_identical_strong_set(tmp_path, monkeypatch):
    """-1 and NULL exominer_score must be indistinguishable to the gate."""
    monkeypatch.setenv("EXONIT_PHASE3_AGREEMENT_THRESHOLD", "0.6")

    db_neg1 = ExoNitDB(str(tmp_path / "neg1.db"))
    _seed(db_neg1, "t", em=-1, em_ffi=0.85)
    db_none = ExoNitDB(str(tmp_path / "none.db"))
    _seed(db_none, "t", em=None, em_ffi=0.85)

    rows_neg1 = db_neg1.strong_candidates()
    rows_none = db_none.strong_candidates()
    assert len(rows_neg1) == len(rows_none) == 1
    assert db_neg1.count_strong_candidates() == db_none.count_strong_candidates() == 1


# PG dialect coverage (EXONIT_TEST_DATABASE_URL — a throwaway DB, never
# production; see tests/_pg_test_db.py). Skips cleanly when unreachable.
_PG_DSN = resolve_test_dsn()
_PG_OP_TABLES = (
    "worker_telemetry", "worker_debug_mode", "candidates",
    "stars", "tasks", "workers_registry",
)


def _pg_truncate(dsn):
    import psycopg
    with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
        c.execute("TRUNCATE " + ", ".join(_PG_OP_TABLES) + " CASCADE")


@pytest.mark.skipif(
    not pg_test_available(_PG_DSN),
    reason="EXONIT_TEST_DATABASE_URL (throwaway PG) unset or not reachable",
)
def test_minus1_sentinel_falls_back_to_ffi_on_postgres(tmp_path, monkeypatch):
    """Same -1 -> absent fallback, on the PG dialect (LEAST vs SQLite's min())."""
    monkeypatch.setenv("EXONIT_PHASE3_AGREEMENT_THRESHOLD", "0.6")
    _pg_truncate(_PG_DSN)
    db = ExoNitDB(_PG_DSN, completeness_path=str(tmp_path / "cmpl.db"))
    assert db.dialect.name == "postgres"
    try:
        _seed(db, "neg1_ffi_pass", em=-1, em_ffi=0.85)
        _seed(db, "neg1_ffi_fail", em=-1, em_ffi=0.50)
        _seed(db, "neg1_ffi_null", em=-1, em_ffi=None)

        rows = db.strong_candidates()
        tics = {r["tic_id"] for r in rows}
        assert tics == {"neg1_ffi_pass"}, (
            f"PG _strong_where must treat -1 as absent same as SQLite; got {tics}"
        )
        assert db.count_strong_candidates() == 1
    finally:
        db.close()
        _pg_truncate(_PG_DSN)
