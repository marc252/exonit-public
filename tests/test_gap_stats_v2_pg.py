"""Tests for S2 gap-stats v2 (postgres_fdw), docs/superpowers/specs/
2026-09-26-enqueue-s1-s4-design.md §5.3.

Requires TWO throwaway real PostgreSQL databases:
  - ``exonit_test``      (operational — ``tasks``/``tic_sector_state``/
                          ``discovery_no_data`` — see tests/_pg_test_db.py)
  - ``exonit_meta_test``  (TGLC index — ``tglc`` — see
                          tests/_metadata_pg_test_db.py)
AND the DDL in ``scripts/migration/2026-09-26_s2_postgres_fdw.sql`` applied
on ``exonit_test`` (schema ``meta_fdw``, foreign table ``meta_fdw.tglc``
pointing at ``exonit_meta_test``). That DDL needs a PostgreSQL SUPERUSER
(``CREATE EXTENSION postgres_fdw`` is not grantable to the ``exonit`` role on
this cluster, verified 2026-09-26 — see that file's header comment) — this
whole module SKIPS CLEANLY when ``meta_fdw.tglc`` is not resolvable in
``exonit_test``, so a sandbox/CI environment without superuser access to the
PG cluster (or without EXONIT_TEST_META_PG_URL set at all — its default is
unset, see tests/_metadata_pg_test_db.py) never fails here.
"""

import csv
import time
import types

import pytest

from tests._pg_test_db import pg_test_available, resolve_test_dsn
from tests._metadata_pg_test_db import pg_meta_test_available, resolve_test_meta_dsn

PG_DSN = resolve_test_dsn()
META_DSN = resolve_test_meta_dsn()

SECTOR = 23


def _meta_fdw_available(dsn) -> bool:
    """True when ``meta_fdw.tglc`` resolves in *dsn* (the DDL has been
    applied). Never raises — any failure (unreachable DB, schema absent,
    missing privilege) means "skip", not "error"."""
    if not dsn:
        return False
    try:
        import psycopg
        with psycopg.connect(dsn, connect_timeout=5) as c:
            with c.cursor() as cur:
                cur.execute("SELECT to_regclass('meta_fdw.tglc')")
                row = cur.fetchone()
                return bool(row and row[0])
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (pg_test_available(PG_DSN) and pg_meta_test_available(META_DSN)
         and _meta_fdw_available(PG_DSN)),
    reason=(
        "requires exonit_test + exonit_meta_test reachable AND "
        "meta_fdw.tglc applied on exonit_test (superuser-only DDL, see "
        "scripts/migration/2026-09-26_s2_postgres_fdw.sql) — skip is expected "
        "in any environment without cluster-superuser access"
    ),
)

# ---------------------------------------------------------------------------
# DDL for the operational-side tables this fixture needs (idempotent).
# ---------------------------------------------------------------------------

_OP_DDL = """
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

CREATE TABLE IF NOT EXISTS tic_sector_state (
  tic_id       TEXT    NOT NULL,
  sector       INTEGER NOT NULL,
  status       TEXT    NOT NULL,
  attempts     INTEGER NOT NULL DEFAULT 0,
  last_task_id TEXT,
  last_error   TEXT,
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (tic_id, sector)
);
CREATE INDEX IF NOT EXISTS idx_tss_done
    ON tic_sector_state (tic_id) WHERE status = 'completed';
CREATE INDEX IF NOT EXISTS idx_tss_sector_failed
    ON tic_sector_state (sector) WHERE status = 'failed';
"""


def _apply_op_ddl(dsn):
    import psycopg
    with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
        c.execute(_OP_DDL)


def _clean_op_tables(dsn):
    import psycopg
    with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
        c.execute("DELETE FROM tic_sector_state WHERE sector IN (%s, 0)", (SECTOR,))
        c.execute("DELETE FROM discovery_no_data WHERE sector = %s", (SECTOR,))


def _clean_meta_tables(dsn):
    import psycopg
    with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
        c.execute("DELETE FROM tglc WHERE sector = %s", (SECTOR,))


# ---------------------------------------------------------------------------
# Fixture: a small but realistic sector with every gap-metric term populated.
#
# TICs 1001..1010 in tglc (sector 23):
#   1001, 1002            -> completed (tic_sector_state)
#   1003                  -> no_data
#   1004, 1005            -> TOI (catalog), not otherwise resolved -> tois_skipped
#   1006                  -> TOI AND completed -> must NOT be double-counted
#   1007                  -> exhausted (failed, attempts=2) -> failed_terminal
#   1008, 1009, 1010      -> plain gaps (processable)
# No canonical CSV (matches the 6 currently-open sectors, spec §1 point 2):
#   work_list == tglc_S == 10.
# ---------------------------------------------------------------------------

@pytest.fixture
def gap_v2_fixture(tmp_path):
    _apply_op_ddl(PG_DSN)
    _clean_op_tables(PG_DSN)
    _clean_meta_tables(META_DSN)

    tics = list(range(1001, 1011))
    completed_tics = [1001, 1002, 1006]
    no_data_tic = 1003
    toi_tics = [1004, 1005, 1006]
    exhausted_tic = 1007

    from exonit.metadata_cache import MetadataCache
    import os
    os.environ["EXONIT_METADATA_DB_URL"] = META_DSN
    try:
        mc = MetadataCache(META_DSN)
        try:
            mc.tglc_add_rows_bulk(
                (str(t), SECTOR, f"/fake/{t}.fits", 120.0, 10.0, 1.0, 1.0)
                for t in tics
            )
        finally:
            mc.close()
    finally:
        os.environ.pop("EXONIT_METADATA_DB_URL", None)

    import psycopg
    with psycopg.connect(PG_DSN, connect_timeout=5, autocommit=True) as c:
        for t in completed_tics:
            c.execute(
                "INSERT INTO tic_sector_state(tic_id, sector, status, attempts) "
                "VALUES (%s, %s, 'completed', 0) "
                "ON CONFLICT (tic_id, sector) DO UPDATE SET status = 'completed'",
                (str(t), SECTOR),
            )
        c.execute(
            "INSERT INTO discovery_no_data(tic_id, sector, reason) "
            "VALUES (%s, %s, 'no_fits') ON CONFLICT DO NOTHING",
            (str(no_data_tic), SECTOR),
        )
        c.execute(
            "INSERT INTO tic_sector_state(tic_id, sector, status, attempts) "
            "VALUES (%s, %s, 'failed', 2) "
            "ON CONFLICT (tic_id, sector) DO UPDATE SET status = 'failed', attempts = 2",
            (str(exhausted_tic), SECTOR),
        )

    canonical_path = str(tmp_path / "canonical_empty.csv")
    with open(canonical_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tic_id", "label"])  # no rows -> no canonical list

    toi_path = str(tmp_path / "toi_catalog.csv")
    with open(toi_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["TIC ID", "Period (days)"])
        for t in toi_tics:
            writer.writerow([t, 3.14])

    fixture = types.SimpleNamespace(
        db=PG_DSN, meta_db=META_DSN,
        canonical=canonical_path, toi_catalog=toi_path,
    )
    yield fixture

    _clean_op_tables(PG_DSN)
    _clean_meta_tables(META_DSN)


def _compute(fixture, *, v2: bool, monkeypatch):
    monkeypatch.setenv("EXONIT_DONE_STATE", "1")
    monkeypatch.setenv("EXONIT_GAP_STATS_V2", "1" if v2 else "0")
    from exonit.discovery_state import sector_gap_stats
    return sector_gap_stats(
        SECTOR, db_path=fixture.db, canonical_csv=fixture.canonical,
        tglc_index=fixture.meta_db, toi_catalog=fixture.toi_catalog,
    )


class TestGapStatsV2ParityWithLegacy:
    def test_v2_matches_legacy_field_by_field(self, gap_v2_fixture, monkeypatch):
        legacy = _compute(gap_v2_fixture, v2=False, monkeypatch=monkeypatch)
        v2 = _compute(gap_v2_fixture, v2=True, monkeypatch=monkeypatch)

        for key in ("work_list_size", "completed", "no_data", "tois_skipped",
                    "gaps_processable", "failed_terminal"):
            assert v2[key] == legacy[key], (
                f"{key}: v2={v2[key]!r} != legacy={legacy[key]!r}")
        assert abs(v2["pct_complete"] - legacy["pct_complete"]) < 1e-9
        assert sorted(v2["missing_download_tics"]) == sorted(
            legacy["missing_download_tics"])

    def test_v2_expected_values(self, gap_v2_fixture, monkeypatch):
        v2 = _compute(gap_v2_fixture, v2=True, monkeypatch=monkeypatch)
        assert v2["work_list_size"] == 10
        assert v2["completed"] == 3          # 1001, 1002, 1006
        assert v2["no_data"] == 1            # 1003
        assert v2["tois_skipped"] == 2        # 1004, 1005 (1006 already completed)
        assert v2["failed_terminal"] == 1     # 1007
        # gaps: 1008, 1009, 1010
        assert v2["gaps_processable"] == 3
        assert v2["missing_download_tics"] == []  # no canonical list


class TestGapStatsV2FlagGates:
    def test_v2_requires_done_state_warns_and_falls_back(
        self, gap_v2_fixture, monkeypatch,
    ):
        import warnings
        monkeypatch.delenv("EXONIT_DONE_STATE", raising=False)
        monkeypatch.setenv("EXONIT_GAP_STATS_V2", "1")
        from exonit.discovery_state import sector_gap_stats

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s = sector_gap_stats(
                SECTOR, db_path=gap_v2_fixture.db,
                canonical_csv=gap_v2_fixture.canonical,
                tglc_index=gap_v2_fixture.meta_db,
                toi_catalog=gap_v2_fixture.toi_catalog,
            )
        assert any(issubclass(x.category, RuntimeWarning) for x in w)
        # Legacy path never adds failed_terminal without EXONIT_DONE_STATE=1
        # too — but it's set via the warning path's own check, so this must
        # still reflect done-state OFF's contract (no key at all).
        assert "failed_terminal" not in s or s.get("failed_terminal") == 0
