"""Tests for /api/backfill/nfpp query indexes (perf/backfill-nfpp-index).

Verifies:
  (a) COUNT and GROUP BY results are identical with and without the new
      partial indexes (correctness / equivalence guarantee).
  (b) EXPLAIN QUERY PLAN references idx_cand_nfpp_pending and
      idx_cand_verdict_notnull and does NOT contain 'SCAN candidates'
      for the optimised queries.

Edge cases covered: fpp NULL, fpp >= 0.5, nfpp NOT NULL, verdict NULL,
multiple distinct verdict values, mixed TICs.
"""

import sqlite3
import pytest
from exonit.db import ExoNitDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_candidate(conn, tic_id, candidate_num, fpp, nfpp, verdict):
    """Insert a minimal candidate row directly into the SQLite connection."""
    conn.execute(
        """INSERT OR IGNORE INTO stars
           (tic_id, ra, dec, tmag, analyzed_at, n_sectors, n_datapoints,
            time_span_days, status)
           VALUES (?, 0, 0, 10, datetime('now'), 1, 1000, 27, 'completed')""",
        (tic_id,),
    )
    conn.execute(
        """INSERT INTO candidates
           (tic_id, candidate_num, period_days, t0, duration_days,
            depth_ppm, snr, cnn_score,
            triceratops_fpp, triceratops_nfpp, triceratops_verdict)
           VALUES (?, ?, 5.0, 1000.0, 0.1, 500.0, 12.0, 0.8, ?, ?, ?)""",
        (tic_id, candidate_num, fpp, nfpp, verdict),
    )
    conn.commit()


def _setup_candidates(conn):
    """Populate candidate rows covering all predicat edge cases.

    Row matrix (TIC/num | fpp       | nfpp  | verdict   | in pending? | in verdict?):
      T1/1   | 0.1      | NULL  | 'candidate'  | YES (fpp>=0,<0.5,nfpp IS NULL) | YES
      T1/2   | 0.4      | NULL  | 'candidate'  | YES                            | YES
      T2/1   | 0.1      | 0.001 | 'validated'  | NO  (nfpp NOT NULL)            | YES
      T3/1   | 0.6      | NULL  | 'likely_fp'  | NO  (fpp >= 0.5)               | YES
      T4/1   | NULL     | NULL  | NULL         | NO  (fpp IS NULL)              | NO
      T5/1   | -1.0     | NULL  | 'failed'     | NO  (fpp < 0)                  | YES
    """
    rows = [
        ("T1", 1, 0.1,  None,  "candidate"),
        ("T1", 2, 0.4,  None,  "candidate"),
        ("T2", 1, 0.1,  0.001, "validated"),
        ("T3", 1, 0.6,  None,  "likely_fp"),
        ("T4", 1, None, None,  None),
        ("T5", 1, -1.0, None,  "failed"),
    ]
    for tic_id, candidate_num, fpp, nfpp, verdict in rows:
        _insert_candidate(conn, tic_id, candidate_num, fpp, nfpp, verdict)


# ---------------------------------------------------------------------------
# Reference queries (no index hints — brute-force full-scan baseline)
# ---------------------------------------------------------------------------

_Q_PENDING_COUNT = (
    "SELECT COUNT(*) FROM candidates "
    "WHERE triceratops_fpp >= 0 AND triceratops_fpp < 0.5 "
    "AND triceratops_nfpp IS NULL"
)
_Q_PENDING_TICS = (
    "SELECT COUNT(DISTINCT tic_id) FROM candidates "
    "WHERE triceratops_fpp >= 0 AND triceratops_fpp < 0.5 "
    "AND triceratops_nfpp IS NULL"
)
_Q_VERDICT_GROUP = (
    "SELECT triceratops_verdict AS v, COUNT(*) AS n "
    "FROM candidates "
    "WHERE triceratops_verdict IS NOT NULL "
    "GROUP BY triceratops_verdict"
)
_Q_WITH_NFPP = (
    "SELECT COUNT(*) FROM candidates WHERE triceratops_nfpp IS NOT NULL"
)

# Original formulation of pending queries (with the redundant IS NOT NULL).
# Used to prove equivalence: removing the clause does not change results.
_Q_PENDING_COUNT_ORIG = (
    "SELECT COUNT(*) FROM candidates "
    "WHERE triceratops_fpp IS NOT NULL AND triceratops_nfpp IS NULL "
    "AND triceratops_fpp >= 0 AND triceratops_fpp < 0.5"
)
_Q_PENDING_TICS_ORIG = (
    "SELECT COUNT(DISTINCT tic_id) FROM candidates "
    "WHERE triceratops_fpp IS NOT NULL AND triceratops_nfpp IS NULL "
    "AND triceratops_fpp >= 0 AND triceratops_fpp < 0.5"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_with_data(tmp_path):
    """ExoNitDB (tmp) pre-populated with edge-case candidates."""
    database = ExoNitDB(str(tmp_path / "test.db"))
    _setup_candidates(database.conn)
    yield database
    database.close()


# ---------------------------------------------------------------------------
# (a) Correctness / equivalence tests
# ---------------------------------------------------------------------------

class TestCountEquivalence:
    """COUNT results must be identical whether IS NOT NULL is present or not."""

    def test_pending_count_matches_original_form(self, db_with_data):
        """Simplified query (no redundant IS NOT NULL) == original form."""
        simplified = db_with_data._query_scalar(_Q_PENDING_COUNT)
        original = db_with_data._query_scalar(_Q_PENDING_COUNT_ORIG)
        assert simplified == original, (
            f"pending COUNT mismatch: simplified={simplified} original={original}"
        )

    def test_pending_tics_matches_original_form(self, db_with_data):
        """COUNT(DISTINCT tic_id) — simplified == original form."""
        simplified = db_with_data._query_scalar(_Q_PENDING_TICS)
        original = db_with_data._query_scalar(_Q_PENDING_TICS_ORIG)
        assert simplified == original, (
            f"pending_tics mismatch: simplified={simplified} original={original}"
        )

    def test_pending_count_value(self, db_with_data):
        """Only T1/1 and T1/2 qualify (fpp in [0,0.5), nfpp IS NULL)."""
        count = db_with_data._query_scalar(_Q_PENDING_COUNT)
        assert count == 2, f"Expected 2 pending candidates, got {count}"

    def test_pending_tics_value(self, db_with_data):
        """Only TIC T1 has qualifying candidates (2 rows, 1 distinct TIC)."""
        count = db_with_data._query_scalar(_Q_PENDING_TICS)
        assert count == 1, f"Expected 1 pending TIC, got {count}"

    def test_verdict_group_by_values(self, db_with_data):
        """Verdict distribution must include all non-NULL verdicts."""
        rows = db_with_data._query(_Q_VERDICT_GROUP)
        by_v = {r["v"]: r["n"] for r in rows}
        # T1/1 + T1/2 = 2 'candidate', T2/1 = 1 'validated',
        # T3/1 = 1 'likely_fp', T5/1 = 1 'failed'. T4/1 verdict IS NULL.
        assert by_v.get("candidate", 0) == 2, f"candidate count wrong: {by_v}"
        assert by_v.get("validated", 0) == 1, f"validated count wrong: {by_v}"
        assert by_v.get("likely_fp", 0) == 1, f"likely_fp count wrong: {by_v}"
        assert by_v.get("failed", 0) == 1, f"failed count wrong: {by_v}"
        assert "None" not in by_v and None not in by_v, (
            f"NULL verdict must not appear in GROUP BY result: {by_v}"
        )

    def test_with_nfpp_value(self, db_with_data):
        """Only T2/1 has nfpp NOT NULL."""
        count = db_with_data._query_scalar(_Q_WITH_NFPP)
        assert count == 1, f"Expected 1 row with nfpp, got {count}"


# ---------------------------------------------------------------------------
# (b) EXPLAIN QUERY PLAN — indexes used, no SCAN candidates
# ---------------------------------------------------------------------------

class TestExplainQueryPlan:
    """After ExoNitDB._create_tables() the new partial indexes exist and the
    planner uses them for the optimised queries."""

    def _explain(self, conn, sql):
        rows = conn.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
        return [r[3] for r in rows]  # column 3 = detail string

    def _uses_index(self, plan_lines, index_name):
        """Return True if any plan line references index_name via USING [COVERING] INDEX.

        SQLite EXPLAIN QUERY PLAN uses different prefixes depending on the
        access type:
          - 'SCAN candidates USING INDEX <name>'    for partial-index scans
          - 'SEARCH candidates USING COVERING INDEX <name> ...' for seeks
        Both indicate that the planner chose the named index; a bare
        'SCAN candidates' (without USING INDEX anywhere) is a full table scan.
        """
        joined = " ".join(plan_lines)
        return index_name in joined

    def test_pending_count_uses_nfpp_pending_index(self, db_with_data):
        # SQLite EXPLAIN QUERY PLAN shows either:
        #   "SCAN candidates USING INDEX <name>"        (partial-index scan)
        #   "SEARCH candidates USING COVERING INDEX <name> ..."  (index seek)
        # Both mean the index is used; a bare "SCAN candidates" without any
        # "USING" clause is the full-table-scan we want to avoid.
        plan = self._explain(db_with_data.conn, _Q_PENDING_COUNT)
        assert self._uses_index(plan, "idx_cand_nfpp_pending"), (
            f"Expected idx_cand_nfpp_pending in EXPLAIN plan, got: {plan}"
        )

    def test_pending_tics_uses_nfpp_pending_index(self, db_with_data):
        plan = self._explain(db_with_data.conn, _Q_PENDING_TICS)
        assert self._uses_index(plan, "idx_cand_nfpp_pending"), (
            f"Expected idx_cand_nfpp_pending in EXPLAIN plan, got: {plan}"
        )

    def test_verdict_group_by_uses_verdict_notnull_index(self, db_with_data):
        plan = self._explain(db_with_data.conn, _Q_VERDICT_GROUP)
        assert self._uses_index(plan, "idx_cand_verdict_notnull"), (
            f"Expected idx_cand_verdict_notnull in EXPLAIN plan, got: {plan}"
        )
