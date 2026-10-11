"""Tests for exonit-compactd Component 1 (2026-09-09 design,
``docs/superpowers/specs/2026-09-09-compactd-service-design.md``): batch-mode
draining in ``scripts/compact_completeness.py``.

Same infrastructure as ``tests/test_compact_completeness.py`` (real
throwaway PG test DB via ``tests/_pg_test_db.py`` — skips cleanly when
unreachable, NEVER production; local-disk ``FakeStore``/``FlakyStore``, never
real MinIO) — reused directly rather than reinvented, following the same
cross-file reuse pattern as ``tests/test_pg_parity_completeness.py``
(``from tests.test_compact_completeness import FakeStore``).

Single-pass (``--batch-rows 0``) behavior is already exhaustively covered by
``tests/test_compact_completeness.py`` — not duplicated here. This file only
exercises what batch mode ADDS: multi-batch draining, per-batch id-exact
deletes, the ``batches`` summary field, and the CLI/env default wiring.
"""

import sys
from pathlib import Path

import pytest

from exonit.completeness_store import bucket_of
from tests._pg_test_db import pg_test_available, resolve_test_dsn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import compact_completeness as cc  # noqa: E402

from tests.test_compact_completeness import FakeStore, FlakyStore  # noqa: E402


PG_DSN = resolve_test_dsn()

pg_only = pytest.mark.skipif(
    not pg_test_available(PG_DSN),
    reason="EXONIT_TEST_DATABASE_URL (throwaway PG) unset or not reachable",
)


@pytest.fixture(autouse=True)
def _no_real_retry_sleep(monkeypatch):
    """Same rationale as ``tests/test_compact_completeness.py``'s fixture of
    the same name: without this, a test that drives ``with_s3_retries``
    through its full backoff schedule (30/60/120/240s) would take minutes."""
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)


_CMPL_PG_TABLES = ("completeness_staging", "completeness_tics", "completeness_agg_bucket")


def _ensure_pg_schema(dsn):
    import os as _os

    from exonit.db import ExoNitDB

    old = _os.environ.get("EXONIT_COMPLETENESS_PG")
    _os.environ["EXONIT_COMPLETENESS_PG"] = "1"
    try:
        db = ExoNitDB(dsn, completeness_path=":memory:")
        db.close()
    finally:
        if old is None:
            _os.environ.pop("EXONIT_COMPLETENESS_PG", None)
        else:
            _os.environ["EXONIT_COMPLETENESS_PG"] = old


def _truncate_pg(dsn):
    import psycopg

    with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
        c.execute("TRUNCATE " + ", ".join(_CMPL_PG_TABLES))


@pytest.fixture
def pg_ready():
    _ensure_pg_schema(PG_DSN)
    _truncate_pg(PG_DSN)
    yield PG_DSN
    _truncate_pg(PG_DSN)


def _insert_staging(dsn, rows):
    """rows: list of (tic_id, radius_earth, period_days, detection_prob)."""
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO completeness_staging "
                "(tic_id, radius_earth, period_days, detection_prob) "
                "VALUES (%s, %s, %s, %s)",
                rows,
            )


def _staging_count(dsn):
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        return conn.execute("SELECT COUNT(*) FROM completeness_staging").fetchone()[0]


def _staging_tics(dsn):
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        return {r[0] for r in conn.execute(
            "SELECT DISTINCT tic_id FROM completeness_staging").fetchall()}


def _tics_table(dsn):
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        rows = conn.execute("SELECT tic_id, bucket FROM completeness_tics").fetchall()
    return {r[0]: r[1] for r in rows}


def _read_bucket_table(store: FakeStore, bucket: int):
    import io

    from exonit.completeness_store import read_parquet_file, table_rows

    key = f"buckets/bucket={bucket:03d}/data.parquet"
    data = store.get_bytes(key)
    return table_rows(read_parquet_file(io.BytesIO(data)))


# --------------------------------------------------------------------------- #
# (a) staging needing >=3 batches drains completely, counters aggregated
#     correctly across every batch.
# --------------------------------------------------------------------------- #

@pg_only
def test_batch_mode_drains_completely_across_multiple_batches_aggregated_counters(
        tmp_path, pg_ready):
    rows = [
        ("120001", 1.0, 5.0, 0.11),
        ("120002", 2.0, 10.0, 0.22),
        ("120003", 4.0, 20.0, 0.33),
        ("120004", 1.0, 5.0, 0.44),
        ("120005", 2.0, 10.0, 0.55),
        ("120006", 4.0, 20.0, 0.66),
        ("120007", 8.0, 40.0, 0.77),
    ]
    _insert_staging(pg_ready, rows)
    expected_buckets = sorted({bucket_of(r[0]) for r in rows})

    store = FakeStore(tmp_path / "minio")
    # 7 rows, 2 per batch -> 4 batches (2, 2, 2, 1): >= 3 batches required.
    result = cc.compact(pg_ready, store, batch_rows=2)

    assert result["ok"] is True
    assert result["skipped"] is False
    assert result["batches"] == 4
    assert result["buckets_touched"] == expected_buckets
    assert result["rows_merged"] == len(rows)
    assert result["tics_new"] == len(rows)
    assert result["tics_updated"] == 0
    assert result["staging_deleted"] == len(rows)

    # staging fully drained, not just the first batch(es).
    assert _staging_count(pg_ready) == 0

    tics = _tics_table(pg_ready)
    assert set(tics.keys()) == {r[0] for r in rows}

    total_rows_in_store = sum(
        len(_read_bucket_table(store, b)) for b in expected_buckets)
    assert total_rows_in_store == len(rows)

    # No leftover .tmp- objects from any batch.
    assert not any(".tmp-" in k for k in store.all_keys())


@pg_only
def test_batch_mode_batches_count_matches_ceil_division(tmp_path, pg_ready):
    """(d) explicit check that ``batches`` reflects exactly how many
    read/merge/delete cycles were needed for a given --batch-rows, not just
    'some positive number'."""
    rows = [(f"13{i:04d}", 1.0, 5.0, 0.1) for i in range(7)]
    _insert_staging(pg_ready, rows)

    store = FakeStore(tmp_path / "minio")
    result = cc.compact(pg_ready, store, batch_rows=3)  # 3+3+1 = 3 batches

    assert result["batches"] == 3
    assert result["staging_deleted"] == 7
    assert _staging_count(pg_ready) == 0


# --------------------------------------------------------------------------- #
# (b) exact per-batch id deletes: a failure partway through the drain leaves
# EARLIER, already-successful batches compacted+deleted, and ONLY the
# in-flight batch's rows intact in staging — never a superset, never a
# subset.
# --------------------------------------------------------------------------- #

@pg_only
def test_batch_failure_deletes_exactly_the_successful_batches_ids(tmp_path, pg_ready):
    rows = [
        ("140001", 1.0, 5.0, 0.11),
        ("140002", 2.0, 10.0, 0.22),
        ("140003", 4.0, 20.0, 0.33),
        ("140004", 1.0, 5.0, 0.44),
        ("140005", 2.0, 10.0, 0.55),
        ("140006", 4.0, 20.0, 0.66),
    ]
    _insert_staging(pg_ready, rows)

    # ORDER BY written_at, id matches insertion order for this single
    # transaction-free sequence of inserts -> batch_rows=2 groups them as
    # [140001,140002] | [140003,140004] | [140005,140006] (batches 1,2,3).
    poison_bucket = bucket_of("140005")
    assert poison_bucket not in {bucket_of(t) for t in
                                  ("140001", "140002", "140003", "140004")}, (
        "test fixture needs the poisoned bucket to be untouched by the "
        "first two (successful) batches"
    )

    minio_root = tmp_path / "minio"
    inner = FakeStore(minio_root)
    flaky = FlakyStore(inner, fail_bucket=poison_bucket)

    with pytest.raises(RuntimeError, match="simulated crash"):
        cc.compact(pg_ready, flaky, batch_rows=2)

    # Batches 1 and 2 (4 rows, tics 140001-140004) fully committed+deleted;
    # batch 3 (tics 140005/140006) never got past the failing upload, so
    # its 2 rows survive UNTOUCHED in staging -- exactly those 2, not more,
    # not fewer.
    assert _staging_count(pg_ready) == 2
    assert _staging_tics(pg_ready) == {"140005", "140006"}

    # completeness_tics: batches 1-2 are in. Batch 3's OTHER bucket
    # (140006, bucket 182) succeeded in the same pass and persisted its
    # Parquet+PG state before the poisoned bucket's error surfaced -- by
    # design (compact_completeness.py: successful buckets are idempotent,
    # staging untouched, retry converges). Only the poisoned bucket's tic
    # must be absent.
    committed_tics = {"140001", "140002", "140003", "140004"}
    tics = _tics_table(pg_ready)
    assert committed_tics <= set(tics.keys())
    assert "140005" not in tics
    assert set(tics.keys()) <= committed_tics | {"140006"}

    # Retry with a healthy store converges: the surviving batch drains too.
    result = cc.compact(pg_ready, FakeStore(minio_root), batch_rows=2)
    assert result["ok"] is True
    assert result["staging_deleted"] == 2
    assert _staging_count(pg_ready) == 0
    tics_after = _tics_table(pg_ready)
    assert set(tics_after.keys()) == {r[0] for r in rows}


# --------------------------------------------------------------------------- #
# (c) --batch-rows 0 = the old single unbounded-read path, unchanged.
# --------------------------------------------------------------------------- #

@pg_only
def test_batch_rows_zero_reads_once_unbounded_single_pass(tmp_path, pg_ready, monkeypatch):
    rows = [
        ("150001", 1.0, 5.0, 0.11),
        ("150002", 2.0, 10.0, 0.22),
        ("150003", 4.0, 20.0, 0.33),
        ("150004", 1.0, 5.0, 0.44),
        ("150005", 2.0, 10.0, 0.55),
    ]
    _insert_staging(pg_ready, rows)

    seen_limits = []
    original_read = cc._read_staging_rows

    def _spy(conn, limit=None):
        seen_limits.append(limit)
        return original_read(conn, limit=limit)

    monkeypatch.setattr(cc, "_read_staging_rows", _spy)

    store = FakeStore(tmp_path / "minio")
    result = cc.compact(pg_ready, store, batch_rows=0)

    # Exactly ONE read, unbounded (limit=None) -- the pre-batch-mode call
    # shape, never LIMIT'd, never looped.
    assert seen_limits == [None]
    assert result["ok"] is True
    assert result["batches"] == 1
    assert result["rows_merged"] == len(rows)
    assert result["staging_deleted"] == len(rows)
    assert _staging_count(pg_ready) == 0


@pg_only
def test_batch_rows_zero_default_when_omitted_from_compact_call(tmp_path, pg_ready):
    """``compact()``'s own default (as opposed to the CLI's) is 0 -- calling
    it without ``batch_rows`` at all must still take the single-pass path,
    so every existing caller of the Python API is unaffected by batch mode
    unless it opts in explicitly."""
    _insert_staging(pg_ready, [("160001", 1.0, 5.0, 0.5)])
    store = FakeStore(tmp_path / "minio")
    result = cc.compact(pg_ready, store)
    assert result["batches"] == 1


# --------------------------------------------------------------------------- #
# CLI / env wiring: --batch-rows, EXONIT_COMPACT_BATCH_ROWS default.
# --------------------------------------------------------------------------- #

def test_default_batch_rows_falls_back_when_env_unset(monkeypatch):
    monkeypatch.delenv("EXONIT_COMPACT_BATCH_ROWS", raising=False)
    assert cc._default_batch_rows() == 2_000_000


def test_default_batch_rows_reads_env_override(monkeypatch):
    monkeypatch.setenv("EXONIT_COMPACT_BATCH_ROWS", "500000")
    assert cc._default_batch_rows() == 500000


def test_default_batch_rows_invalid_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("EXONIT_COMPACT_BATCH_ROWS", "not-a-number")
    assert cc._default_batch_rows() == 2_000_000


def test_parse_args_batch_rows_defaults_from_env(monkeypatch):
    monkeypatch.setenv("EXONIT_COMPACT_BATCH_ROWS", "12345")
    args = cc.parse_args(["--pg", "postgresql://x/y"])
    assert args.batch_rows == 12345


def test_parse_args_batch_rows_explicit_zero_overrides_env(monkeypatch):
    monkeypatch.setenv("EXONIT_COMPACT_BATCH_ROWS", "12345")
    args = cc.parse_args(["--pg", "postgresql://x/y", "--batch-rows", "0"])
    assert args.batch_rows == 0


def test_parse_args_batch_rows_default_with_no_env_set(monkeypatch):
    monkeypatch.delenv("EXONIT_COMPACT_BATCH_ROWS", raising=False)
    args = cc.parse_args(["--pg", "postgresql://x/y"])
    assert args.batch_rows == 2_000_000


# --------------------------------------------------------------------------- #
# Empty / locked summaries also carry the new field (regression guard: any
# future refactor of _empty_summary must not silently drop it again).
# --------------------------------------------------------------------------- #

@pg_only
def test_empty_staging_summary_has_batches_zero(tmp_path, pg_ready):
    store = FakeStore(tmp_path / "minio")
    result = cc.compact(pg_ready, store, batch_rows=2)
    assert result["skipped"] is True
    assert result["batches"] == 0
