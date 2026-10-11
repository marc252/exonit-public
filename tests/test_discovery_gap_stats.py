"""Tests for sector_gap_stats (the real gap metric).

Requires the throwaway exonit_test PostgreSQL DB. Skips when unavailable.
Anti-prod guard: EXONIT_TEST_DATABASE_URL must not point at a protected DB name.
See tests/_pg_test_db.py for the isolation protocol.
"""

import csv
import sqlite3
import types

import pytest

from tests._pg_test_db import pg_test_available, resolve_test_dsn

# Throwaway test DB only; never production.
PG_DSN = resolve_test_dsn()

pytestmark = pytest.mark.skipif(
    not pg_test_available(PG_DSN),
    reason="EXONIT_TEST_DATABASE_URL (throwaway PG) unset or not reachable",
)

# ---------------------------------------------------------------------------
# DDL for the new tables (idempotent — matches schema_pg.sql)
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS discovery_sector_state (
  sector     INTEGER PRIMARY KEY,
  state      TEXT NOT NULL DEFAULT 'coming_soon',
  detail     TEXT,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS discovery_no_data (
  tic_id     TEXT NOT NULL,
  sector     INTEGER NOT NULL,
  reason     TEXT NOT NULL DEFAULT 'no_fits',
  checked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (tic_id, sector)
);

CREATE INDEX IF NOT EXISTS idx_no_data_sector ON discovery_no_data(sector);
"""


def _apply_ddl(dsn):
    import psycopg
    with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
        c.execute(_DDL)


def _truncate_new_tables(dsn):
    import psycopg
    with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
        c.execute("TRUNCATE discovery_no_data")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pg_test_db():
    """Apply DDL and truncate discovery_no_data before/after the test."""
    _apply_ddl(PG_DSN)
    _truncate_new_tables(PG_DSN)
    yield PG_DSN
    _truncate_new_tables(PG_DSN)


@pytest.fixture
def gap_fixture(tmp_path, pg_test_db):
    """Build a minimal but realistic fixture for sector_gap_stats:

    - 5 TICs in sector 23 (in TGLC index + canonical CSV)
    - 2 completed tasks (TICs 1001, 1002) in the PG test DB
    - 1 no-data row (TIC 1003) in discovery_no_data
    - Expected: work_list_size=5, completed=2, no_data=1, gaps=2, pct=0.6
    """
    SECTOR = 23
    TICS = [1001, 1002, 1003, 1004, 1005]
    COMPLETED_TICS = [1001, 1002]
    NO_DATA_TIC = 1003

    # --- TGLC SQLite index ---
    tglc_path = str(tmp_path / "tglc.db")
    con = sqlite3.connect(tglc_path)
    con.execute("CREATE TABLE tglc (tic_id TEXT, sector INTEGER)")
    con.executemany(
        "INSERT INTO tglc VALUES (?, ?)",
        [(str(t), SECTOR) for t in TICS],
    )
    con.commit()
    con.close()

    # --- canonical CSV ---
    canonical_path = str(tmp_path / "canonical.csv")
    with open(canonical_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tic_id", "label"])
        for t in TICS:
            writer.writerow([str(t), ""])

    # --- insert completed tasks and no_data row into PG ---
    import psycopg
    with psycopg.connect(pg_test_db, connect_timeout=5, autocommit=True) as c:
        for t in COMPLETED_TICS:
            c.execute(
                "INSERT INTO tasks(task_id, tic_id, status) VALUES (%s, %s, 'completed') "
                "ON CONFLICT (task_id) DO NOTHING",
                (f"gap-fixture-task-{t}", str(t)),
            )
        c.execute(
            "INSERT INTO discovery_no_data(tic_id, sector, reason) "
            "VALUES (%s, %s, 'no_fits') ON CONFLICT DO NOTHING",
            (str(NO_DATA_TIC), SECTOR),
        )

    fixture = types.SimpleNamespace(
        db=pg_test_db,
        canonical=canonical_path,
        tglc=tglc_path,
    )
    yield fixture

    # Cleanup: remove only the tasks we inserted (leave everything else)
    with psycopg.connect(pg_test_db, connect_timeout=5, autocommit=True) as c:
        for t in COMPLETED_TICS:
            c.execute(
                "DELETE FROM tasks WHERE task_id = %s",
                (f"gap-fixture-task-{t}",),
            )


# ---------------------------------------------------------------------------
# Task 2: sector_gap_stats real gap metric
# ---------------------------------------------------------------------------

def test_gap_stats_counts_real_gaps(gap_fixture):
    from exonit.discovery_state import sector_gap_stats

    s = sector_gap_stats(
        23,
        db_path=gap_fixture.db,
        canonical_csv=gap_fixture.canonical,
        tglc_index=gap_fixture.tglc,
    )
    assert s["work_list_size"] == 5
    assert s["completed"] == 2
    assert s["no_data"] == 1
    assert s["gaps_processable"] == 2          # 5 - 2 - 1
    assert abs(s["pct_complete"] - 0.6) < 1e-6  # (2+1)/5


# ---------------------------------------------------------------------------
# Task: TOI subtraction (sector_gap_stats must not count TOIs as gaps)
# ---------------------------------------------------------------------------

def test_gap_stats_subtracts_tois(gap_fixture, tmp_path):
    """Work-list TOIs that are not completed and not no_data reduce gaps.

    Fixture state (sector 23):
      TICS=[1001,1002,1003,1004,1005], completed={1001,1002}, no_data={1003}
      → gaps_processable=2 (TICs 1004, 1005)

    TOI catalog contains TICs 1004 and 1005.
    Expected: tois_skipped=2, gaps_processable=0, pct_complete=1.0.
    """
    import warnings
    from exonit.discovery_state import sector_gap_stats

    toi_csv = tmp_path / "toi_catalog.csv"
    toi_csv.write_text("TIC ID,Period (days)\n1004,3.14\n1005,2.72\n", encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = sector_gap_stats(
            23,
            db_path=gap_fixture.db,
            canonical_csv=gap_fixture.canonical,
            tglc_index=gap_fixture.tglc,
            toi_catalog=str(toi_csv),
        )

    assert s["tois_skipped"] == 2
    assert s["gaps_processable"] == 0
    assert abs(s["pct_complete"] - 1.0) < 1e-6
    # work_list_size, completed, no_data must be unchanged
    assert s["work_list_size"] == 5
    assert s["completed"] == 2
    assert s["no_data"] == 1
    # TOIs must be excluded from missing_download_tics
    assert 1004 not in s["missing_download_tics"]
    assert 1005 not in s["missing_download_tics"]


def test_gap_stats_toi_not_double_counted(gap_fixture, tmp_path):
    """A TOI that is already completed must NOT be subtracted again.

    TIC 1001 is in both the completed set and the TOI catalog.
    It is already counted in ``completed``, so tois_skipped must be 0
    and gaps_processable must remain 2 (unchanged from baseline).
    """
    import warnings
    from exonit.discovery_state import sector_gap_stats

    toi_csv = tmp_path / "toi_catalog_dup.csv"
    toi_csv.write_text("TIC ID,Period (days)\n1001,1.5\n", encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = sector_gap_stats(
            23,
            db_path=gap_fixture.db,
            canonical_csv=gap_fixture.canonical,
            tglc_index=gap_fixture.tglc,
            toi_catalog=str(toi_csv),
        )

    assert s["tois_skipped"] == 0        # 1001 is already in 'done', not double-counted
    assert s["gaps_processable"] == 2    # unchanged
    assert abs(s["pct_complete"] - 0.6) < 1e-6


def test_gap_stats_missing_toi_catalog(gap_fixture, tmp_path):
    """A missing TOI catalog must not crash; tois_skipped=0, metric unchanged."""
    import warnings
    from exonit.discovery_state import sector_gap_stats

    missing_path = str(tmp_path / "no_such_toi.csv")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = sector_gap_stats(
            23,
            db_path=gap_fixture.db,
            canonical_csv=gap_fixture.canonical,
            tglc_index=gap_fixture.tglc,
            toi_catalog=missing_path,
        )

    # Must not raise; must emit a RuntimeWarning
    assert any(issubclass(warning.category, RuntimeWarning) for warning in w), \
        "Expected a RuntimeWarning for missing TOI catalog"
    assert s["tois_skipped"] == 0
    assert s["gaps_processable"] == 2    # baseline unchanged
    assert abs(s["pct_complete"] - 0.6) < 1e-6
