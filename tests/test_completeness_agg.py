"""Tests for completeness_agg incremental aggregate (feature: completeness-agg).

Covers:
 - Table created at DB init.
 - Trigger AFTER INSERT: new (tic, r, p) → agg row created.
 - Trigger AFTER UPDATE OF detection_prob: existing row updated → agg adjusted.
 - Upsert with identical value: idempotent (agg unchanged).
 - Trigger AFTER DELETE: agg decremented; agg row removed when n=0.
 - Backfill: completeness populated, agg empty → agg == GROUP BY after re-open.
 - add_completeness_many: same correctness guarantees as add_completeness.
 - /api/completeness endpoint: same format as the previous GROUP BY query.
"""

import os
import pytest
from exonit.db import ExoNitDB


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    database = ExoNitDB(db_path)
    yield database
    database.close()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _group_by_direct(db):
    """Ground-truth: AVG(detection_prob), COUNT(*) from completeness directly."""
    return {
        (r["radius_earth"], r["period_days"]): (r["avg_prob"], r["n"])
        for r in db._query(
            """SELECT radius_earth, period_days,
                      CAST(SUM(detection_prob) AS REAL) / COUNT(*) AS avg_prob,
                      COUNT(*) AS n
               FROM completeness
               GROUP BY radius_earth, period_days""")
    }


def _agg_contents(db):
    """Contents of completeness_agg as {(r, p): (avg, n)}."""
    return {
        (r["radius_earth"], r["period_days"]): (r["sum_prob"] / r["n"], r["n"])
        for r in db._query("SELECT * FROM completeness_agg")
    }


def _assert_agg_matches_direct(db):
    """Assert completeness_agg == GROUP BY on completeness (within float epsilon)."""
    expected = _group_by_direct(db)
    actual = _agg_contents(db)
    assert set(actual.keys()) == set(expected.keys()), (
        f"cell keys differ: actual={set(actual.keys())} expected={set(expected.keys())}")
    for key in expected:
        exp_avg, exp_n = expected[key]
        act_avg, act_n = actual[key]
        assert act_n == exp_n, f"n mismatch at {key}: {act_n} != {exp_n}"
        assert abs(act_avg - exp_avg) < 1e-9, (
            f"avg mismatch at {key}: {act_avg} != {exp_avg}")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_completeness_agg_table_exists(self, db):
        # db.list_tables() was removed (2026-07-02 cleanup, minor item 8:
        # zero-caller sqlite_master-only helper) — query sqlite_master
        # directly; this fixture is always the SQLite path.
        tables = {r["name"] for r in db._query(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "completeness_agg" in tables

    def test_completeness_agg_empty_on_fresh_db(self, db):
        rows = db._query("SELECT * FROM completeness_agg")
        assert rows == []


# ---------------------------------------------------------------------------
# Trigger AFTER INSERT
# ---------------------------------------------------------------------------

class TestInsertTrigger:
    def test_single_insert_creates_agg_row(self, db):
        db.add_star("T1", ra=0, dec=0, tmag=10.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)
        db.add_completeness("T1", 1.0, 5.0, 0.8)
        _assert_agg_matches_direct(db)

    def test_two_different_tics_same_cell_accumulates(self, db):
        for tic in ("T1", "T2"):
            db.add_star(tic, ra=0, dec=0, tmag=10.0,
                        n_sectors=1, n_datapoints=100, time_span_days=27.0)
        db.add_completeness("T1", 1.0, 5.0, 0.6)
        db.add_completeness("T2", 1.0, 5.0, 0.4)
        _assert_agg_matches_direct(db)
        agg = _agg_contents(db)
        avg, n = agg[(1.0, 5.0)]
        assert n == 2
        assert abs(avg - 0.5) < 1e-9

    def test_multiple_cells_multiple_rows_in_agg(self, db):
        db.add_star("T1", ra=0, dec=0, tmag=10.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)
        cells = [(1.0, 5.0, 0.1), (2.0, 10.0, 0.5), (4.0, 20.0, 0.9)]
        for r, p, d in cells:
            db.add_completeness("T1", r, p, d)
        _assert_agg_matches_direct(db)
        assert len(db._query("SELECT * FROM completeness_agg")) == 3


# ---------------------------------------------------------------------------
# Trigger AFTER UPDATE OF detection_prob
# ---------------------------------------------------------------------------

class TestUpdateTrigger:
    def test_upsert_changes_prob(self, db):
        db.add_star("T1", ra=0, dec=0, tmag=10.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)
        db.add_completeness("T1", 1.0, 5.0, 0.3)
        db.add_completeness("T1", 1.0, 5.0, 0.9)  # update same key
        _assert_agg_matches_direct(db)
        agg = _agg_contents(db)
        avg, n = agg[(1.0, 5.0)]
        assert n == 1
        assert abs(avg - 0.9) < 1e-9

    def test_upsert_same_value_is_idempotent(self, db):
        db.add_star("T1", ra=0, dec=0, tmag=10.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)
        db.add_completeness("T1", 1.0, 5.0, 0.7)
        db.add_completeness("T1", 1.0, 5.0, 0.7)  # same value, no real change
        _assert_agg_matches_direct(db)
        agg = _agg_contents(db)
        avg, n = agg[(1.0, 5.0)]
        assert n == 1
        assert abs(avg - 0.7) < 1e-9

    def test_upsert_two_tics_one_updates(self, db):
        for tic in ("T1", "T2"):
            db.add_star(tic, ra=0, dec=0, tmag=10.0,
                        n_sectors=1, n_datapoints=100, time_span_days=27.0)
        db.add_completeness("T1", 1.0, 5.0, 0.4)
        db.add_completeness("T2", 1.0, 5.0, 0.6)
        db.add_completeness("T1", 1.0, 5.0, 0.8)  # T1 re-run: higher detection
        _assert_agg_matches_direct(db)
        agg = _agg_contents(db)
        avg, n = agg[(1.0, 5.0)]
        assert n == 2
        assert abs(avg - 0.7) < 1e-9  # (0.8 + 0.6) / 2


# ---------------------------------------------------------------------------
# Trigger AFTER DELETE
# ---------------------------------------------------------------------------

class TestDeleteTrigger:
    def test_delete_single_row_removes_agg(self, db):
        db.add_star("T1", ra=0, dec=0, tmag=10.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)
        db.add_completeness("T1", 1.0, 5.0, 0.5)
        db._exec("DELETE FROM completeness WHERE tic_id='T1'")
        _assert_agg_matches_direct(db)
        assert db._query("SELECT * FROM completeness_agg") == []

    def test_delete_one_of_two_decrements_agg(self, db):
        for tic in ("T1", "T2"):
            db.add_star(tic, ra=0, dec=0, tmag=10.0,
                        n_sectors=1, n_datapoints=100, time_span_days=27.0)
        db.add_completeness("T1", 1.0, 5.0, 0.2)
        db.add_completeness("T2", 1.0, 5.0, 0.8)
        db._exec("DELETE FROM completeness WHERE tic_id='T1'")
        _assert_agg_matches_direct(db)
        agg = _agg_contents(db)
        avg, n = agg[(1.0, 5.0)]
        assert n == 1
        assert abs(avg - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# add_completeness_many
# ---------------------------------------------------------------------------

class TestAddCompletenessManyAgg:
    def test_batch_insert_updates_agg(self, db):
        db.add_star("T1", ra=0, dec=0, tmag=10.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)
        cells = [(r, p, 0.1 * (i + 1))
                 for i, (r, p) in enumerate([(1, 5), (2, 10), (4, 20)])]
        db.add_completeness_many("T1", cells)
        _assert_agg_matches_direct(db)

    def test_batch_upsert_updates_agg(self, db):
        db.add_star("T1", ra=0, dec=0, tmag=10.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)
        db.add_completeness_many("T1", [(1.0, 5.0, 0.2), (2.0, 10.0, 0.4)])
        db.add_completeness_many("T1", [(1.0, 5.0, 0.9)])  # upsert first cell
        _assert_agg_matches_direct(db)

    def test_batch_two_tics_accumulates(self, db):
        for tic in ("T1", "T2"):
            db.add_star(tic, ra=0, dec=0, tmag=10.0,
                        n_sectors=1, n_datapoints=100, time_span_days=27.0)
        db.add_completeness_many("T1", [(1.0, 5.0, 0.3)])
        db.add_completeness_many("T2", [(1.0, 5.0, 0.7)])
        _assert_agg_matches_direct(db)
        agg = _agg_contents(db)
        assert agg[(1.0, 5.0)][1] == 2  # n=2


# ---------------------------------------------------------------------------
# query_completeness_agg() — the public method the server endpoint now uses
# ---------------------------------------------------------------------------

class TestQueryCompletenessAgg:
    def test_empty_returns_empty_list(self, db):
        assert db.query_completeness_agg() == []

    def test_shape_and_values(self, db):
        for tic in ("T1", "T2"):
            db.add_star(tic, ra=0, dec=0, tmag=10.0,
                        n_sectors=1, n_datapoints=100, time_span_days=27.0)
        db.add_completeness("T1", 1.0, 10.0, 0.5)
        db.add_completeness("T2", 1.0, 10.0, 0.3)   # same cell → accumulates
        db.add_completeness("T1", 2.0, 20.0, 0.8)
        rows = db.query_completeness_agg()
        # Keys match the dashboard contract.
        for r in rows:
            assert set(r.keys()) == {"radius_earth", "period_days",
                                     "avg_prob", "n_stars"}
        by_cell = {(r["radius_earth"], r["period_days"]):
                   (r["avg_prob"], r["n_stars"]) for r in rows}
        assert abs(by_cell[(1.0, 10.0)][0] - 0.4) < 1e-9  # (0.5+0.3)/2
        assert by_cell[(1.0, 10.0)][1] == 2
        assert abs(by_cell[(2.0, 20.0)][0] - 0.8) < 1e-9
        assert by_cell[(2.0, 20.0)][1] == 1
        # Ordered by (radius_earth, period_days).
        keys = [(r["radius_earth"], r["period_days"]) for r in rows]
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Backfill (one-time migration)
# ---------------------------------------------------------------------------

class TestBackfill:
    def test_backfill_on_reopen(self, tmp_path):
        """DB with completeness data and no agg → re-open populates agg == GROUP BY."""
        db_path = str(tmp_path / "backfill.db")
        # Step 1: populate completeness without agg (simulate pre-migration DB
        # by manually clearing agg after inserting with the triggers active,
        # then re-opening so the backfill code runs again).
        db1 = ExoNitDB(db_path)
        for tic in ("T1", "T2", "T3"):
            db1.add_star(tic, ra=0, dec=0, tmag=10.0,
                         n_sectors=1, n_datapoints=100, time_span_days=27.0)
        cells = [(r, p, 0.1 * (i + 1))
                 for i, (r, p) in enumerate([(1, 5), (2, 10), (4, 20),
                                              (8, 40), (1, 10), (2, 20)])]
        for tic in ("T1", "T2", "T3"):
            db1.add_completeness_many(tic, cells)

        # Capture the GROUP BY ground truth before clearing agg.
        expected = _group_by_direct(db1)

        # Manually clear completeness_agg to simulate the pre-migration state.
        db1._exec("DELETE FROM completeness_agg")
        assert db1._query("SELECT * FROM completeness_agg") == []
        db1.close()

        # Step 2: re-open → backfill should run.
        db2 = ExoNitDB(db_path)
        _assert_agg_matches_direct(db2)
        actual = _agg_contents(db2)
        assert set(actual.keys()) == set(expected.keys())
        db2.close()

    def test_backfill_skipped_if_agg_nonempty(self, tmp_path):
        """If agg is already populated, backfill code is a no-op on re-open."""
        db_path = str(tmp_path / "backfill2.db")
        db1 = ExoNitDB(db_path)
        db1.add_star("T1", ra=0, dec=0, tmag=10.0,
                     n_sectors=1, n_datapoints=100, time_span_days=27.0)
        db1.add_completeness("T1", 1.0, 5.0, 0.5)
        db1.close()

        db2 = ExoNitDB(db_path)
        agg = db2._query("SELECT * FROM completeness_agg")
        assert len(agg) == 1  # still exactly 1 row, not doubled
        assert agg[0]["n"] == 1
        db2.close()


# ---------------------------------------------------------------------------
# Endpoint /api/completeness
# ---------------------------------------------------------------------------

class TestCompletenessEndpoint:
    @pytest.fixture
    def client(self, tmp_path):
        os.environ["EXONIT_DB_PATH"] = str(tmp_path / "test.db")
        os.environ["EXONIT_REPORTS_DIR"] = str(tmp_path / "reports")
        os.makedirs(tmp_path / "reports", exist_ok=True)
        from exonit.server import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        yield TestClient(app)
        del os.environ["EXONIT_DB_PATH"]
        del os.environ["EXONIT_REPORTS_DIR"]

    def test_empty_db_returns_empty_list(self, client):
        resp = client.get("/api/completeness")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_response_matches_ground_truth(self, tmp_path):
        """Endpoint avg_prob/n_stars must match direct GROUP BY on completeness."""
        db_path = str(tmp_path / "ep.db")
        db = ExoNitDB(db_path)
        for tic in ("T1", "T2", "T3"):
            db.add_star(tic, ra=0, dec=0, tmag=10.0,
                        n_sectors=1, n_datapoints=100, time_span_days=27.0)
        cells = [(r, p, 0.1 * (i + 1))
                 for i, (r, p) in enumerate([(1, 5), (2, 10), (4, 20)])]
        db.add_completeness_many("T1", cells)
        db.add_completeness_many("T2", cells)
        cells_t3 = [(r, p, 0.5) for r, p, _ in cells]
        db.add_completeness_many("T3", cells_t3)

        expected_gt = _group_by_direct(db)
        db.close()

        os.environ["EXONIT_DB_PATH"] = db_path
        os.environ["EXONIT_REPORTS_DIR"] = str(tmp_path / "reports")
        os.makedirs(tmp_path / "reports", exist_ok=True)
        try:
            from exonit.server import create_app
            from fastapi.testclient import TestClient
            app = create_app()
            client = TestClient(app)
            resp = client.get("/api/completeness")
            assert resp.status_code == 200
            data = resp.json()

            # Check required keys in each row.
            for row in data:
                assert set(row.keys()) >= {"radius_earth", "period_days",
                                           "avg_prob", "n_stars"}

            # Check values match ground truth.
            actual = {(r["radius_earth"], r["period_days"]): (r["avg_prob"], r["n_stars"])
                      for r in data}
            assert set(actual.keys()) == set(expected_gt.keys())
            for key in expected_gt:
                exp_avg, exp_n = expected_gt[key]
                act_avg, act_n = actual[key]
                assert act_n == exp_n
                assert abs(act_avg - exp_avg) < 1e-9
        finally:
            del os.environ["EXONIT_DB_PATH"]
            del os.environ["EXONIT_REPORTS_DIR"]

    def test_response_format_has_required_keys(self, client, tmp_path):
        """Each row must have radius_earth, period_days, avg_prob, n_stars."""
        db_path = os.environ["EXONIT_DB_PATH"]
        db = ExoNitDB(db_path)
        db.add_star("T1", ra=0, dec=0, tmag=10.0,
                    n_sectors=1, n_datapoints=100, time_span_days=27.0)
        db.add_completeness("T1", 1.5, 7.0, 0.6)
        db.close()

        resp = client.get("/api/completeness")
        assert resp.status_code == 200
        rows = resp.json()
        assert len(rows) == 1
        row = rows[0]
        assert row["radius_earth"] == 1.5
        assert row["period_days"] == 7.0
        assert abs(row["avg_prob"] - 0.6) < 1e-9
        assert row["n_stars"] == 1
