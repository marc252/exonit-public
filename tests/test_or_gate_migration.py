"""Tests for the OR-gate SQLite index migration (2026-07-11).

docs/reviews/injection_e2e_results_2026-07-11.md +
docs/reviews/ffi_cnn_investigation_2026-07-11.md §5: the candidate
eligibility gate ``snr>=10 AND cnn_score>=0.7`` becomes
``snr>=10 AND (cnn_score>=0.7 OR exominer_ffi_score>=0.5)`` across the
verdict/FPP/strong chain. ``idx_cand_fpp_todo``, ``idx_cand_nfpp_todo`` and
``idx_cand_strong`` are DROP+CREATEd with the new predicate on every
``ExoNitDB.__init__`` (SQLite path; PostgreSQL is schema-managed out of
band — see scripts/migration/schema_pg.sql's deploy note). This file
verifies:

  1. The three index definitions actually carry the new OR-gate predicate
     after a fresh ``ExoNitDB()`` construction.
  2. ``idx_cand_exominer_todo`` (Docker-SPOC ExoMiner backfill,
     EXONIT_ENABLE_EXOMINER_AUTO=0 in prod) is DELIBERATELY unchanged — out
     of scope for this OR-gate (item 6 of progress/impl-orgate.md).
  3. The migration is idempotent: constructing ``ExoNitDB`` twice against
     the same SQLite file (i.e. running ``_create_tables`` twice) does not
     raise and leaves the index definitions in the same (new) state.
"""
import sqlite3

import pytest

from exonit.db import ExoNitDB


def _index_sql(conn, index_name):
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
        (index_name,),
    ).fetchone()
    return row[0] if row else None


class TestOrGatePredicateInIndexes:
    def test_idx_cand_fpp_todo_carries_or_gate(self, tmp_path):
        db = ExoNitDB(str(tmp_path / "test.db"))
        sql = _index_sql(db.conn, "idx_cand_fpp_todo")
        db.close()
        assert sql is not None
        assert "exominer_ffi_score" in sql
        assert "cnn_score >= 0.7" in sql
        assert "OR" in sql

    def test_idx_cand_nfpp_todo_carries_or_gate(self, tmp_path):
        db = ExoNitDB(str(tmp_path / "test.db"))
        sql = _index_sql(db.conn, "idx_cand_nfpp_todo")
        db.close()
        assert sql is not None
        assert "exominer_ffi_score" in sql
        assert "cnn_score >= 0.7" in sql
        assert "OR" in sql

    def test_idx_cand_strong_carries_or_gate(self, tmp_path):
        db = ExoNitDB(str(tmp_path / "test.db"))
        sql = _index_sql(db.conn, "idx_cand_strong")
        db.close()
        assert sql is not None
        # The OR-gate eligibility clause (0.5) AND the pre-existing
        # NULLIF/COALESCE scoring clause (0.7) must both be present —
        # they are two different thresholds for two different purposes.
        assert "exominer_ffi_score >= 0.5" in sql
        assert "cnn_score >= 0.7" in sql
        assert "COALESCE(NULLIF(exominer_score, -1), exominer_ffi_score) >= 0.7" in sql

    def test_idx_cand_exominer_todo_unchanged_docker_spoc_path(self, tmp_path):
        """Item 6 of progress/impl-orgate.md: the Docker-SPOC ExoMiner
        backfill index is deliberately left on cnn_score>=0.7 alone — it is
        outside the discovery chain this OR-gate targets."""
        db = ExoNitDB(str(tmp_path / "test.db"))
        sql = _index_sql(db.conn, "idx_cand_exominer_todo")
        db.close()
        assert sql is not None
        assert "cnn_score >= 0.7" in sql
        assert "exominer_ffi_score" not in sql
        assert "OR" not in sql


class TestMigrationIdempotence:
    def test_reopening_db_twice_does_not_raise(self, tmp_path):
        path = str(tmp_path / "test.db")
        db1 = ExoNitDB(path)
        db1.close()
        # Re-opening (== re-running _create_tables' DROP+CREATE migration
        # block) against the SAME file must not raise, mirroring what
        # happens on every real server restart.
        db2 = ExoNitDB(path)
        db2.close()

    def test_reopening_db_twice_leaves_or_gate_predicate_stable(self, tmp_path):
        path = str(tmp_path / "test.db")
        db1 = ExoNitDB(path)
        sql_first = _index_sql(db1.conn, "idx_cand_fpp_todo")
        db1.close()

        db2 = ExoNitDB(path)
        sql_second = _index_sql(db2.conn, "idx_cand_fpp_todo")
        db2.close()

        assert sql_first == sql_second
        assert "exominer_ffi_score" in sql_second

    def test_reopening_db_three_times_is_still_stable(self, tmp_path):
        """Explicit 'run it more than twice' — idempotence must hold for
        an arbitrary number of restarts, not just the first two."""
        path = str(tmp_path / "test.db")
        seen = []
        for _ in range(3):
            db = ExoNitDB(path)
            seen.append(_index_sql(db.conn, "idx_cand_strong"))
            db.close()
        assert len(set(seen)) == 1, f"index definition drifted across restarts: {seen}"

    def test_migration_preserves_existing_rows(self, tmp_path):
        """DROP INDEX + CREATE INDEX must never touch row data — only the
        index metadata."""
        path = str(tmp_path / "test.db")
        db1 = ExoNitDB(path)
        db1.add_star("900001", ra=0, dec=0, tmag=11, n_sectors=1,
                     n_datapoints=100, time_span_days=27.0)
        db1.add_candidate("900001", candidate_num=1, period_days=2.0,
                          t0=1.0, duration_days=0.1, depth_ppm=500.0,
                          snr=12.0, cnn_score=0.4, exominer_ffi_score=0.6)
        db1.close()

        # Re-open (re-runs the DROP+CREATE index migration) and confirm the
        # row is still there and still queryable through the new predicate.
        db2 = ExoNitDB(path)
        rows = db2._query(
            "SELECT * FROM candidates c INDEXED BY idx_cand_fpp_todo "
            "WHERE c.snr >= 10.0 "
            "AND (c.cnn_score >= 0.7 OR c.exominer_ffi_score >= 0.5) "
            "AND c.triceratops_fpp IS NULL")
        db2.close()
        assert len(rows) == 1
        assert rows[0]["tic_id"] == "900001"
