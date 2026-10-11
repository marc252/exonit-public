"""Equivalence tests for the optimised worker_fpp_task query.

perf/fpp-task-query-index: the original OR query used idx_candidates_snr and
scanned ~355k rows on every poll.  The replacement uses a UNION of two
INDEXED BY subqueries (idx_cand_fpp_todo / idx_cand_nfpp_todo) that cover
only the rows that actually need work.

Eligibility predicate updated 2026-07-11 to the OR-gate
(docs/reviews/injection_e2e_results_2026-07-11.md + docs/reviews/
ffi_cnn_investigation_2026-07-11.md §5): ``cnn_score >= 0.7 OR
exominer_ffi_score >= 0.5`` (was ``cnn_score >= 0.7`` alone) — both queries
below and the two partial indexes they drive were updated together, so this
file still tests same-shape equivalence against the (now OR-gated) reference
query, plus one new case proving the em_ffi rescue path.

These tests verify:
1. The new query returns EXACTLY the same candidate IDs and order as the
   original query across all interesting cases:
     - arm 1 only (fpp IS NULL)
     - arm 2 only (low fpp, nfpp IS NULL)
     - both arms active simultaneously
     - em_ffi-rescued candidate (cnn<0.7, exominer_ffi_score>=0.5)
     - candidates that must NOT appear (snr<10, cnn<0.7 and em_ffi<0.5 (or
       NULL), star not completed, fpp already done, fpp>=0.5)
2. The ORDER BY c.snr DESC LIMIT 200 ordering is preserved.
3. EXPLAIN QUERY PLAN for the new query references idx_cand_fpp_todo and
   idx_cand_nfpp_todo, and contains no "SCAN candidates" or "SCAN c".
"""

import sqlite3
import os
import pytest

from exonit.db import ExoNitDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _old_query(conn):
    """Original broad OR query — the reference (OR-gate, 2026-07-11)."""
    cur = conn.execute(
        """SELECT c.id, c.snr FROM candidates c
           JOIN stars s ON c.tic_id = s.tic_id
           WHERE c.snr >= 10.0
             AND (c.cnn_score >= 0.7 OR c.exominer_ffi_score >= 0.5)
             AND (
               c.triceratops_fpp IS NULL
               OR (
                 c.triceratops_fpp >= 0
                 AND c.triceratops_fpp < 0.5
                 AND c.triceratops_nfpp IS NULL
               )
             )
             AND s.status = 'completed'
           ORDER BY c.snr DESC
           LIMIT 200"""
    )
    return [(r[0], r[1]) for r in cur.fetchall()]


def _new_query(conn):
    """Optimised UNION query using the partial indexes (OR-gate, 2026-07-11)."""
    cur = conn.execute(
        """SELECT c.id, c.snr FROM candidates c
           JOIN stars s ON c.tic_id = s.tic_id
           WHERE c.id IN (
             SELECT id FROM candidates INDEXED BY idx_cand_fpp_todo
             WHERE snr >= 10.0 AND (cnn_score >= 0.7 OR exominer_ffi_score >= 0.5)
               AND triceratops_fpp IS NULL
             UNION
             SELECT id FROM candidates INDEXED BY idx_cand_nfpp_todo
             WHERE snr >= 10.0 AND (cnn_score >= 0.7 OR exominer_ffi_score >= 0.5)
               AND triceratops_fpp >= 0 AND triceratops_fpp < 0.5
               AND triceratops_nfpp IS NULL
           )
           AND s.status = 'completed'
           ORDER BY c.snr DESC
           LIMIT 200"""
    )
    return [(r[0], r[1]) for r in cur.fetchall()]


def _insert_star(conn, tic_id, status="completed"):
    conn.execute(
        "INSERT OR IGNORE INTO stars (tic_id, status, n_sectors) VALUES (?, ?, 1)",
        (tic_id, status),
    )


def _insert_candidate(conn, tic_id, cand_num, snr, cnn_score,
                      fpp=None, nfpp=None, exominer_ffi_score=None):
    """Insert a minimal candidate row for testing."""
    conn.execute(
        """INSERT INTO candidates
               (tic_id, candidate_num, snr, cnn_score, triceratops_fpp,
                triceratops_nfpp, period_days, t0, duration_days, depth_ppm,
                exominer_ffi_score)
           VALUES (?, ?, ?, ?, ?, ?, 5.0, 1000.0, 0.1, 500.0, ?)""",
        (tic_id, cand_num, snr, cnn_score, fpp, nfpp, exominer_ffi_score),
    )


# ---------------------------------------------------------------------------
# Fixture: in-memory DB with the full schema (indexes included)
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test_fpp_idx.db")
    # Instantiate via ExoNitDB to create tables + indexes
    db = ExoNitDB(path)
    db.close()
    return path


@pytest.fixture
def conn(db_path):
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Test 1: empty table — both queries return []
# ---------------------------------------------------------------------------

def test_empty_table_returns_empty(conn):
    assert _old_query(conn) == []
    assert _new_query(conn) == []


# ---------------------------------------------------------------------------
# Test 2: arm 1 only — candidates with fpp IS NULL
# ---------------------------------------------------------------------------

def test_arm1_fpp_null(conn):
    _insert_star(conn, "TIC001")
    _insert_candidate(conn, "TIC001", 1, snr=20.0, cnn_score=0.8, fpp=None)
    conn.commit()

    old = _old_query(conn)
    new = _new_query(conn)
    assert old == new
    assert len(old) == 1


# ---------------------------------------------------------------------------
# Test 3: arm 2 only — low fpp, nfpp IS NULL
# ---------------------------------------------------------------------------

def test_arm2_low_fpp_nfpp_null(conn):
    _insert_star(conn, "TIC002")
    _insert_candidate(conn, "TIC002", 1, snr=15.0, cnn_score=0.9,
                      fpp=0.1, nfpp=None)
    conn.commit()

    old = _old_query(conn)
    new = _new_query(conn)
    assert old == new
    assert len(old) == 1


# ---------------------------------------------------------------------------
# Test 4: both arms active simultaneously
# ---------------------------------------------------------------------------

def test_both_arms(conn):
    _insert_star(conn, "TIC003")
    _insert_candidate(conn, "TIC003", 1, snr=25.0, cnn_score=0.8, fpp=None)
    _insert_candidate(conn, "TIC003", 2, snr=18.0, cnn_score=0.75,
                      fpp=0.2, nfpp=None)
    conn.commit()

    old = _old_query(conn)
    new = _new_query(conn)
    assert old == new
    assert len(old) == 2


# ---------------------------------------------------------------------------
# Test 5: exclusion cases — none of these should appear
# ---------------------------------------------------------------------------

def test_exclusions_do_not_appear(conn):
    _insert_star(conn, "TIC_EX", status="completed")
    _insert_star(conn, "TIC_PEND", status="pending")

    # snr < 10 — must be excluded (even with a passing cnn/em_ffi)
    _insert_candidate(conn, "TIC_EX", 1, snr=5.0, cnn_score=0.8, fpp=None)
    # cnn < 0.7 AND em_ffi NULL — must be excluded (neither OR-gate arm fires)
    _insert_candidate(conn, "TIC_EX", 2, snr=20.0, cnn_score=0.5, fpp=None)
    # cnn < 0.7 AND em_ffi < 0.5 — must be excluded (OR-gate, neither side fires)
    _insert_candidate(conn, "TIC_EX", 6, snr=20.0, cnn_score=0.5, fpp=None,
                      exominer_ffi_score=0.49)
    # fpp >= 0.5 — likely_fp, already decided
    _insert_candidate(conn, "TIC_EX", 3, snr=20.0, cnn_score=0.8, fpp=0.6)
    # fpp done + nfpp done — fully completed
    _insert_candidate(conn, "TIC_EX", 4, snr=20.0, cnn_score=0.8,
                      fpp=0.1, nfpp=0.001)
    # star not completed — must be excluded
    _insert_candidate(conn, "TIC_PEND", 1, snr=20.0, cnn_score=0.8, fpp=None)
    conn.commit()

    old = _old_query(conn)
    new = _new_query(conn)
    assert old == new
    assert len(old) == 0


# ---------------------------------------------------------------------------
# Test 5b: em_ffi rescue — cnn<0.7 alone would fail, exominer_ffi_score>=0.5
# rescues it into both arms (OR-gate, 2026-07-11).
# ---------------------------------------------------------------------------

def test_em_ffi_rescue_arm1(conn):
    _insert_star(conn, "TIC_RESCUE1")
    _insert_candidate(conn, "TIC_RESCUE1", 1, snr=12.0, cnn_score=0.4,
                      fpp=None, exominer_ffi_score=0.6)
    conn.commit()

    old = _old_query(conn)
    new = _new_query(conn)
    assert old == new
    assert len(old) == 1, "cnn<0.7 but exominer_ffi_score>=0.5 must pass arm 1"


def test_em_ffi_rescue_arm2(conn):
    _insert_star(conn, "TIC_RESCUE2")
    _insert_candidate(conn, "TIC_RESCUE2", 1, snr=12.0, cnn_score=0.4,
                      fpp=0.2, nfpp=None, exominer_ffi_score=0.55)
    conn.commit()

    old = _old_query(conn)
    new = _new_query(conn)
    assert old == new
    assert len(old) == 1, "cnn<0.7 but exominer_ffi_score>=0.5 must pass arm 2"


# ---------------------------------------------------------------------------
# Test 6: mixed valid + invalid, ORDER BY snr DESC preserved
# ---------------------------------------------------------------------------

def test_mixed_order_preserved(conn):
    _insert_star(conn, "TIC_MIX")
    # Valid arm-1 candidates with different SNRs
    _insert_candidate(conn, "TIC_MIX", 1, snr=30.0, cnn_score=0.9, fpp=None)
    _insert_candidate(conn, "TIC_MIX", 2, snr=10.0, cnn_score=0.7, fpp=None)
    _insert_candidate(conn, "TIC_MIX", 3, snr=50.0, cnn_score=0.8, fpp=None)
    # Valid arm-2
    _insert_candidate(conn, "TIC_MIX", 4, snr=40.0, cnn_score=0.8,
                      fpp=0.3, nfpp=None)
    # Invalid (snr < 10)
    _insert_candidate(conn, "TIC_MIX", 5, snr=9.9, cnn_score=0.8, fpp=None)
    conn.commit()

    old = _old_query(conn)
    new = _new_query(conn)
    assert old == new
    # Verify descending SNR order
    snrs = [r[1] for r in new]
    assert snrs == sorted(snrs, reverse=True)
    assert len(new) == 4


# ---------------------------------------------------------------------------
# Test 7: EXPLAIN QUERY PLAN — new query uses partial indexes, no SCAN
# ---------------------------------------------------------------------------

def test_explain_uses_partial_indexes(conn):
    plan_rows = conn.execute(
        """EXPLAIN QUERY PLAN
           SELECT c.id, c.snr FROM candidates c
           JOIN stars s ON c.tic_id = s.tic_id
           WHERE c.id IN (
             SELECT id FROM candidates INDEXED BY idx_cand_fpp_todo
             WHERE snr >= 10.0 AND (cnn_score >= 0.7 OR exominer_ffi_score >= 0.5)
               AND triceratops_fpp IS NULL
             UNION
             SELECT id FROM candidates INDEXED BY idx_cand_nfpp_todo
             WHERE snr >= 10.0 AND (cnn_score >= 0.7 OR exominer_ffi_score >= 0.5)
               AND triceratops_fpp >= 0 AND triceratops_fpp < 0.5
               AND triceratops_nfpp IS NULL
           )
           AND s.status = 'completed'
           ORDER BY c.snr DESC
           LIMIT 200"""
    ).fetchall()

    # EXPLAIN QUERY PLAN returns rows with columns (id, parent, notused, detail).
    # The index name appears in the 'detail' column (index 3).
    plan_text = " ".join(r[3] if len(r) >= 4 else str(tuple(r))
                         for r in plan_rows)

    assert "idx_cand_fpp_todo" in plan_text, (
        f"Expected idx_cand_fpp_todo in plan, got:\n{plan_text}"
    )
    assert "idx_cand_nfpp_todo" in plan_text, (
        f"Expected idx_cand_nfpp_todo in plan, got:\n{plan_text}"
    )
    # The optimised query must not fall back to a full table scan
    assert "SCAN candidates" not in plan_text, (
        f"Unexpected SCAN candidates in plan:\n{plan_text}"
    )
    assert "SCAN c" not in plan_text, (
        f"Unexpected SCAN c in plan:\n{plan_text}"
    )


# ---------------------------------------------------------------------------
# Test 8: LIMIT 200 boundary — when > 200 eligible, only top 200 returned
# ---------------------------------------------------------------------------

def test_limit_200_applied(conn):
    _insert_star(conn, "TIC_BIG")
    for i in range(210):
        _insert_candidate(conn, "TIC_BIG", i + 1,
                          snr=10.0 + i, cnn_score=0.8, fpp=None)
    conn.commit()

    old = _old_query(conn)
    new = _new_query(conn)
    assert old == new
    assert len(new) == 200
    # Verify top SNR first
    snrs = [r[1] for r in new]
    assert snrs[0] > snrs[-1]
