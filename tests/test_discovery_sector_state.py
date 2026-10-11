"""Tests for discovery_sector_state and discovery_no_data PG tables,
and for get_sector_state / set_sector_state helpers.

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
# DDL for the new tables (idempotent — same SQL as schema_pg.sql)
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
        c.execute("TRUNCATE discovery_sector_state")
        c.execute("TRUNCATE discovery_no_data")


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def pg_test_db():
    """Apply DDL for the new tables (idempotent) and truncate before/after."""
    _apply_ddl(PG_DSN)
    _truncate_new_tables(PG_DSN)
    yield PG_DSN
    _truncate_new_tables(PG_DSN)


# ---------------------------------------------------------------------------
# Task 1: tables are reachable via db_backend
# ---------------------------------------------------------------------------

def test_sector_state_tables_exist(pg_test_db):
    from exonit.db_backend import connect_operational
    b = connect_operational(pg_test_db)
    try:
        b.execute(
            "INSERT INTO discovery_sector_state(sector,state) VALUES (999,'active') "
            "ON CONFLICT (sector) DO UPDATE SET state=EXCLUDED.state"
        )
        row = b.query_one(
            "SELECT state FROM discovery_sector_state WHERE sector=999"
        )
        assert row["state"] == "active"
    finally:
        b.close()


# ---------------------------------------------------------------------------
# Task 3: get_sector_state / set_sector_state round-trip
# ---------------------------------------------------------------------------

def test_set_then_get_sector_state(pg_test_db):
    from exonit.discovery_state import get_sector_state, set_sector_state
    set_sector_state(23, "active", db_path=pg_test_db)
    assert get_sector_state(23, db_path=pg_test_db) == "active"
    assert get_sector_state(99, db_path=pg_test_db) == "coming_soon"  # default
