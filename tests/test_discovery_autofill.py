"""Tests for discovery_autofill table + get_autofill/set_autofill helpers.

Requires the throwaway exonit_test PostgreSQL DB. Skips when unavailable.
Anti-prod guard: EXONIT_TEST_DATABASE_URL must not point at a protected DB name.
See tests/_pg_test_db.py for the isolation protocol.
"""

import pytest

from tests._pg_test_db import pg_test_available, resolve_test_dsn

# Throwaway test DB only (EXONIT_TEST_DATABASE_URL); never production.
PG_DSN = resolve_test_dsn()

pytestmark = pytest.mark.skipif(
    not pg_test_available(PG_DSN),
    reason="EXONIT_TEST_DATABASE_URL (throwaway PG) unset or not reachable",
)

# ---------------------------------------------------------------------------
# DDL for discovery_autofill (idempotent — same SQL as added to schema_pg.sql)
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS discovery_autofill (
  sector       INTEGER PRIMARY KEY,
  status       TEXT,
  current_lot  TEXT,
  lots_done    INT DEFAULT 0,
  anomalies    INT DEFAULT 0,
  last_alert   TEXT,
  updated_at   TIMESTAMPTZ DEFAULT now()
);
"""


def _apply_ddl(dsn):
    import psycopg
    with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
        c.execute(_DDL)


def _truncate(dsn):
    import psycopg
    with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
        c.execute("TRUNCATE discovery_autofill")


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def pg_test_db():
    """Apply DDL for discovery_autofill (idempotent) and truncate before/after."""
    _apply_ddl(PG_DSN)
    _truncate(PG_DSN)
    yield PG_DSN
    _truncate(PG_DSN)


# ---------------------------------------------------------------------------
# Task 7: get_autofill / set_autofill round-trip
# ---------------------------------------------------------------------------

def test_get_autofill_returns_none_when_no_row(pg_test_db):
    """get_autofill returns None when the sector has no row."""
    from exonit.discovery_state import get_autofill
    assert get_autofill(42, db_path=pg_test_db) is None


def test_set_then_get_autofill_basic(pg_test_db):
    """set_autofill creates a row; get_autofill returns it as a dict."""
    from exonit.discovery_state import get_autofill, set_autofill

    set_autofill(23, db_path=pg_test_db, status="running", lots_done=3)
    row = get_autofill(23, db_path=pg_test_db)

    assert row is not None
    assert row["status"] == "running"
    assert row["lots_done"] == 3
    assert row["sector"] == 23


def test_set_autofill_partial_update_preserves_other_fields(pg_test_db):
    """set_autofill with a subset of fields preserves the others from the previous set."""
    from exonit.discovery_state import get_autofill, set_autofill

    # Initial insert with multiple fields
    set_autofill(
        23,
        db_path=pg_test_db,
        status="running",
        lots_done=3,
        current_lot="lot_abc",
    )

    # Partial update: only change status
    set_autofill(23, db_path=pg_test_db, status="stopped")

    row = get_autofill(23, db_path=pg_test_db)
    assert row is not None
    assert row["status"] == "stopped"           # updated
    assert row["lots_done"] == 3               # preserved
    assert row["current_lot"] == "lot_abc"     # preserved


def test_set_autofill_anomaly_and_alert(pg_test_db):
    """anomalies and last_alert fields are stored and updated correctly."""
    from exonit.discovery_state import get_autofill, set_autofill

    set_autofill(
        7,
        db_path=pg_test_db,
        status="alert",
        anomalies=2,
        last_alert="empty lot returned while gaps > 0",
    )

    row = get_autofill(7, db_path=pg_test_db)
    assert row["status"] == "alert"
    assert row["anomalies"] == 2
    assert row["last_alert"] == "empty lot returned while gaps > 0"


def test_set_autofill_updated_at_is_refreshed(pg_test_db):
    """updated_at column is set on insert and changes on update."""
    import time
    from exonit.discovery_state import get_autofill, set_autofill

    set_autofill(5, db_path=pg_test_db, status="running")
    row1 = get_autofill(5, db_path=pg_test_db)
    ts1 = row1["updated_at"]

    # Brief pause to ensure a different timestamp
    time.sleep(0.05)

    set_autofill(5, db_path=pg_test_db, status="stopped")
    row2 = get_autofill(5, db_path=pg_test_db)
    ts2 = row2["updated_at"]

    assert ts2 >= ts1, "updated_at should be refreshed on every set_autofill call"

