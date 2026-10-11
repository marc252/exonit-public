"""Tests for Fase 4a (completeness -> Parquet+PG), Task 1.

Two halves:
  * Pure-Python helpers in ``exonit/completeness_store.py`` (flag, bucket_of,
    the bucket-count constant) — no DB involved.
  * ``ExoNitDB`` PG-mode behaviour with ``EXONIT_COMPLETENESS_PG=1``: the new
    write-path staging + PG-backed readers, against the *real* PostgreSQL
    test DB (``EXONIT_TEST_DATABASE_URL``, see tests/_pg_test_db.py). Skips
    cleanly when PG / psycopg / ~/.pgpass is unavailable.

Flag OFF (unset, the default) is intentionally NOT re-tested for the
SQLite-sidecar path here — that's the existing, still-green
``tests/test_db.py -k completeness`` suite and the flag-off half of
``tests/test_db_pg_parity.py`` (``test_completeness_routed_to_sqlite`` /
``test_query_completeness_agg_reads_sqlite_in_pg_mode``), which this task
must not perturb.
"""

import zlib

import pytest

from exonit.completeness_store import (
    COMPLETENESS_BUCKETS,
    bucket_of,
    completeness_pg_enabled,
)
from exonit.db import ExoNitDB
from tests._pg_test_db import pg_test_available, resolve_test_dsn


# --------------------------------------------------------------------------- #
# completeness_pg_enabled()
# --------------------------------------------------------------------------- #

def test_completeness_pg_enabled_false_by_default(monkeypatch):
    monkeypatch.delenv("EXONIT_COMPLETENESS_PG", raising=False)
    assert completeness_pg_enabled() is False


def test_completeness_pg_enabled_true_for_exact_string_1(monkeypatch):
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    assert completeness_pg_enabled() is True


@pytest.mark.parametrize("value", ["0", "true", "True", "yes", " 1", "1 ", ""])
def test_completeness_pg_enabled_false_for_anything_else(monkeypatch, value):
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", value)
    assert completeness_pg_enabled() is False


# --------------------------------------------------------------------------- #
# COMPLETENESS_BUCKETS / bucket_of()
# --------------------------------------------------------------------------- #

def test_completeness_buckets_is_256():
    assert COMPLETENESS_BUCKETS == 256


def test_bucket_of_matches_crc32_formula():
    for tic in ("100", "281541555", "1", "999999999"):
        assert bucket_of(tic) == zlib.crc32(tic.encode("utf-8")) % 256


def test_bucket_of_is_deterministic_across_calls():
    assert bucket_of("123456") == bucket_of("123456")


def test_bucket_of_in_range_for_many_tics():
    for tic in (str(i) for i in range(0, 500000, 977)):  # coprime stride
        b = bucket_of(tic)
        assert 0 <= b < COMPLETENESS_BUCKETS


def test_bucket_of_distinguishes_different_tics():
    # Not a strict requirement of the formula, but a sanity check that the
    # 256-bucket space is actually being spread across, not collapsed.
    buckets = {bucket_of(str(i)) for i in range(1000)}
    assert len(buckets) > 100


# --------------------------------------------------------------------------- #
# ExoNitDB PG-mode write-path (EXONIT_COMPLETENESS_PG=1), real PostgreSQL.
# --------------------------------------------------------------------------- #

PG_DSN = resolve_test_dsn()

_CMPL_PG_TABLES = (
    "completeness_staging",
    "completeness_tics",
    "completeness_agg_bucket",
)

pytestmark = pytest.mark.skipif(
    not pg_test_available(PG_DSN),
    reason="EXONIT_TEST_DATABASE_URL (throwaway PG) unset or not reachable",
)


def _truncate(dsn):
    import psycopg
    with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
        c.execute("TRUNCATE " + ", ".join(_CMPL_PG_TABLES) + " CASCADE")


@pytest.fixture
def cmpl_pgdb(tmp_path, monkeypatch):
    """ExoNitDB against the real test PG, completeness-PG flag ON.

    completeness_path points at a throwaway tmp_path SQLite file that must
    stay UNTOUCHED (no completeness table ever created there) — the whole
    point of the flag is that the SQLite sidecar disappears.
    """
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    db = ExoNitDB(PG_DSN, completeness_path=str(tmp_path / "cmpl_unused.db"))
    assert db.dialect.name == "postgres"
    _truncate(PG_DSN)
    yield db
    _truncate(PG_DSN)
    db.close()


def test_flag_on_pg_mode_has_no_sqlite_completeness_sidecar(cmpl_pgdb):
    """__init__ must not open self._cmpl_conn (the sidecar disappears)."""
    assert cmpl_pgdb._cmpl_conn is None


def test_ensure_completeness_pg_schema_is_idempotent(cmpl_pgdb):
    # __init__ already ran _ensure_completeness_pg_schema() once; calling it
    # again (as a second ExoNitDB() connect would) must not raise.
    cmpl_pgdb._ensure_completeness_pg_schema()
    for table in _CMPL_PG_TABLES:
        assert cmpl_pgdb._query_scalar(f"SELECT COUNT(*) FROM {table}") == 0


def test_add_completeness_many_writes_to_staging_pg(cmpl_pgdb):
    cells = [(1.0, 5.0, 0.10), (2.0, 10.0, 0.20), (4.0, 20.0, 0.30)]
    cmpl_pgdb.add_completeness_many("123456", cells)
    rows = cmpl_pgdb.backend.query(
        "SELECT tic_id, radius_earth, period_days, detection_prob "
        "FROM completeness_staging WHERE tic_id = ? ORDER BY radius_earth",
        ("123456",),
    )
    assert len(rows) == len(cells)
    got = sorted((r["radius_earth"], r["period_days"], r["detection_prob"])
                 for r in rows)
    assert got == sorted(cells)


def test_add_completeness_many_is_pure_append_no_upsert(cmpl_pgdb):
    """No ON CONFLICT: writing the same (tic, radius, period) cell twice must
    produce TWO staging rows — the dedup/last-write-wins is compaction's job
    (Task 3), not this write path's."""
    cmpl_pgdb.add_completeness_many("777", [(1.0, 5.0, 0.10)])
    cmpl_pgdb.add_completeness_many("777", [(1.0, 5.0, 0.99)])
    n = cmpl_pgdb.backend.query_scalar(
        "SELECT COUNT(*) FROM completeness_staging WHERE tic_id = ?", ("777",))
    assert n == 2


def test_add_completeness_singular_writes_to_staging_pg(cmpl_pgdb):
    """add_completeness (singular) is the one call scripts/
    injection_probe_driver.py makes in production — it must not crash under
    the flag, and must land in staging exactly like the batched form."""
    cmpl_pgdb.add_completeness("55", 1.0, 5.0, 0.42)
    rows = cmpl_pgdb.backend.query(
        "SELECT tic_id, radius_earth, period_days, detection_prob "
        "FROM completeness_staging WHERE tic_id = ?", ("55",))
    assert len(rows) == 1
    assert rows[0]["detection_prob"] == pytest.approx(0.42)


def test_add_completeness_many_empty_is_noop_pg(cmpl_pgdb):
    cmpl_pgdb.add_completeness_many("nobody", [])
    n = cmpl_pgdb.backend.query_scalar("SELECT COUNT(*) FROM completeness_staging")
    assert n == 0


def test_add_completeness_many_one_transaction(cmpl_pgdb):
    """A large batch commits atomically — spot-check via row count, not an
    internals assertion (the backend already guarantees one connection
    checkout per call; this pins the observable behaviour)."""
    cells = [(float(r), float(p), 0.5) for r in range(10) for p in range(5)]
    cmpl_pgdb.add_completeness_many("bigbatch", cells)
    n = cmpl_pgdb.backend.query_scalar(
        "SELECT COUNT(*) FROM completeness_staging WHERE tic_id = ?", ("bigbatch",))
    assert n == len(cells)


def test_query_completeness_agg_reads_from_agg_bucket_pg(cmpl_pgdb):
    """Seed completeness_agg_bucket directly (as the Task-3 compactor would)
    across two different buckets that share the same (radius, period) cell,
    and confirm the SUM/SUM grouping collapses them into one exact row."""
    cmpl_pgdb.backend.executemany(
        "INSERT INTO completeness_agg_bucket "
        "(bucket, radius_earth, period_days, sum_prob, n) VALUES (?, ?, ?, ?, ?)",
        [
            (1, 1.0, 10.0, 5.0, 10),   # bucket 1: 10 stars, sum_prob 5.0
            (2, 1.0, 10.0, 3.0, 5),    # bucket 2: same cell, 5 stars
            (1, 2.0, 20.0, 8.0, 8),    # different cell, single bucket
        ],
    )
    rows = cmpl_pgdb.query_completeness_agg()
    by_cell = {(r["radius_earth"], r["period_days"]):
               (r["avg_prob"], r["n_stars"]) for r in rows}
    assert by_cell[(1.0, 10.0)] == (pytest.approx(8.0 / 15), 15)
    assert by_cell[(2.0, 20.0)] == (1.0, 8)
    keys = [(r["radius_earth"], r["period_days"]) for r in rows]
    assert keys == sorted(keys)


def test_query_completeness_agg_empty_pg(cmpl_pgdb):
    assert cmpl_pgdb.query_completeness_agg() == []


def test_count_completeness_tics_combines_tics_and_staging_no_double_count(cmpl_pgdb):
    """Seed completeness_tics (already-compacted TICs) with an overlap against
    staging (not-yet-compacted rows for some of the SAME tics, plus brand new
    ones) — the count must be exact, no double-counting the overlap."""
    cmpl_pgdb.backend.executemany(
        "INSERT INTO completeness_tics (tic_id, bucket) VALUES (?, ?)",
        [("100", 10), ("200", 20), ("300", 30)],
    )
    cmpl_pgdb.add_completeness_many("100", [(1.0, 5.0, 0.5)])   # already compacted -> overlap
    cmpl_pgdb.add_completeness_many("400", [(1.0, 5.0, 0.5)])   # brand new, staging-only
    cmpl_pgdb.add_completeness_many("400", [(2.0, 10.0, 0.6)])  # same tic, second cell/row
    assert cmpl_pgdb.count_completeness_tics() == 4  # 100,200,300,400 (not 5)


def test_count_completeness_tics_zero_when_both_empty(cmpl_pgdb):
    assert cmpl_pgdb.count_completeness_tics() == 0


def test_wal_checkpoint_is_noop_when_completeness_pg_enabled(cmpl_pgdb):
    assert cmpl_pgdb.wal_checkpoint("TRUNCATE") is None


def test_close_does_not_error_with_no_sqlite_sidecar(cmpl_pgdb):
    # Belt-and-braces: close() must special-case _cmpl_conn is None instead
    # of unconditionally calling .close() on it.
    cmpl_pgdb.close()
