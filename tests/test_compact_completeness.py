"""Tests for Fase 4a (completeness -> Parquet+PG), Task 3: the recurring
compaction job (``scripts/compact_completeness.py``).

Against the real throwaway PG test DB (``tests/_pg_test_db.py``,
``EXONIT_TEST_DATABASE_URL``) — skips cleanly when unreachable, NEVER
production — plus a local-disk fake object store (same duck-typed
put_file/put_bytes/get_bytes/exists/delete_object surface as
``exonit.object_store.ObjectStore``, per ``tests/test_migrate_completeness.py``'s
``FakeStore`` pattern), never real MinIO.
"""

import json
import sys
from pathlib import Path

import pytest

from exonit.completeness_store import bucket_of
from tests._pg_test_db import pg_test_available, resolve_test_dsn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import compact_completeness as cc  # noqa: E402


# --------------------------------------------------------------------------- #
# PG fixtures — mirrors tests/test_migrate_completeness.py's pattern exactly:
# reuse ExoNitDB's idempotent in-code DDL to provision the 3 Fase 4a tables on
# a fresh exonit_test, then TRUNCATE before/after each test.
# --------------------------------------------------------------------------- #

PG_DSN = resolve_test_dsn()

pg_only = pytest.mark.skipif(
    not pg_test_available(PG_DSN),
    reason="EXONIT_TEST_DATABASE_URL (throwaway PG) unset or not reachable",
)


@pytest.fixture(autouse=True)
def _no_real_retry_sleep(monkeypatch):
    """Every real compaction path now runs its S3 calls through
    ``exonit.object_store.with_s3_retries`` (2026-07-21 fix), whose backoff
    schedule (30/60/120/240s) would otherwise make ANY test exercising a
    failing/flaky store take minutes. ``time`` is a single shared module
    object (``sys.modules['time']``), so patching the attribute here also
    covers ``scripts/compact_completeness.py``'s own ``import time`` (none
    currently, but future-proof) and any other caller — never just the one
    this fixture happens to reference."""
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


# --------------------------------------------------------------------------- #
# Staging seed / read helpers
# --------------------------------------------------------------------------- #

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


def _agg_by_cell(dsn):
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        rows = conn.execute(
            "SELECT radius_earth, period_days, SUM(sum_prob), SUM(n) "
            "FROM completeness_agg_bucket GROUP BY radius_earth, period_days"
        ).fetchall()
    return {(r[0], r[1]): (r[2], r[3]) for r in rows}


def _agg_by_bucket_cell(dsn):
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        rows = conn.execute(
            "SELECT bucket, radius_earth, period_days, sum_prob, n "
            "FROM completeness_agg_bucket"
        ).fetchall()
    return {(r[0], r[1], r[2]): (r[3], r[4]) for r in rows}


def _tics_table(dsn):
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        rows = conn.execute("SELECT tic_id, bucket FROM completeness_tics").fetchall()
    return {r[0]: r[1] for r in rows}


def _expected_agg(rows):
    agg = {}
    for tic_id, r, p, prob in rows:
        key = (r, p)
        s, n = agg.get(key, (0.0, 0))
        agg[key] = (s + prob, n + 1)
    return agg


# --------------------------------------------------------------------------- #
# Fake object store (local disk) — same surface as ObjectStore, never MinIO.
# --------------------------------------------------------------------------- #

class FakeStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.calls = []
        self.reset_calls = 0

    def _path(self, key):
        return self.root / key

    def reset_client(self):
        """No-op duck-type of ``ObjectStore.reset_client`` — every real
        compaction path now goes through ``with_s3_retries``, which calls
        this before every retry (2026-07-21 fix)."""
        self.reset_calls += 1

    def exists(self, key):
        return self._path(key).exists()

    def put_file(self, key, local_path, content_type="application/octet-stream"):
        import shutil

        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, dest)
        self.calls.append(("put_file", key))

    def put_bytes(self, key, data, content_type="application/octet-stream"):
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        self.calls.append(("put_bytes", key))

    def get_bytes(self, key):
        return self._path(key).read_bytes()

    def delete_object(self, key):
        p = self._path(key)
        if p.exists():
            p.unlink()
        self.calls.append(("delete_object", key))

    def all_keys(self):
        return [str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file()]


class PoisonStore:
    """Raises loudly on ANY call — proves --dry-run never touches the store."""

    def _boom(self, *a, **k):
        raise AssertionError("dry-run must never touch the object store")

    exists = _boom
    put_file = _boom
    put_bytes = _boom
    get_bytes = _boom
    delete_object = _boom


class FlakyStore:
    """Wraps a working FakeStore; raises on EVERY call when uploading the
    tmp file for *fail_bucket* — simulates a persistent (not transient)
    failure on exactly that bucket, so ``with_s3_retries`` exhausts every
    attempt (2026-07-21 fix) and the bucket still ends up marked failed,
    unchanged from the pre-retry behaviour.

    Deliberately keyed by BUCKET ID rather than "the Nth bucket processed"
    (the pre-2026-07-17 version of this fixture): since compaction now runs
    buckets across a ThreadPoolExecutor (EXONIT_COMPACT_WORKERS), the order
    in which put_file() is called for different buckets' tmp keys is no
    longer deterministic, and a shared mutable "buckets seen so far" counter
    would itself be a data race across threads. A fixed target bucket id is
    thread-safe (no shared state to race on) and order-independent by
    construction — it behaves identically whether compaction is sequential
    or parallel.
    """

    def __init__(self, inner: FakeStore, fail_bucket: int):
        self.inner = inner
        self.fail_bucket = fail_bucket
        self.reset_calls = 0

    def reset_client(self):
        self.reset_calls += 1

    def exists(self, key):
        return self.inner.exists(key)

    def get_bytes(self, key):
        return self.inner.get_bytes(key)

    def put_bytes(self, key, data, **kw):
        return self.inner.put_bytes(key, data, **kw)

    def delete_object(self, key):
        return self.inner.delete_object(key)

    def put_file(self, key, local_path, **kw):
        if ".tmp-" in key:
            bucket_str = key.split("bucket=")[1].split("/")[0]
            if int(bucket_str) == self.fail_bucket:
                raise RuntimeError(
                    f"simulated crash uploading tmp for bucket {bucket_str}")
        return self.inner.put_file(key, local_path, **kw)


class FlakyNTimesStore:
    """Wraps a working FakeStore; raises when uploading the tmp file for
    *fail_bucket* on the first *fail_times* calls, then succeeds — proves
    ``with_s3_retries`` actually recovers a transient MinIO failure instead
    of failing the bucket outright (2026-07-21 fix, mirrors
    tests/test_discovery_download.py::_FlakyOstore for the download side of
    the same shared helper)."""

    def __init__(self, inner: FakeStore, fail_bucket: int, fail_times: int):
        self.inner = inner
        self.fail_bucket = fail_bucket
        self.fail_times = fail_times
        self.attempts = 0
        self.reset_calls = 0

    def reset_client(self):
        self.reset_calls += 1

    def exists(self, key):
        return self.inner.exists(key)

    def get_bytes(self, key):
        return self.inner.get_bytes(key)

    def put_bytes(self, key, data, **kw):
        return self.inner.put_bytes(key, data, **kw)

    def delete_object(self, key):
        return self.inner.delete_object(key)

    def put_file(self, key, local_path, **kw):
        if ".tmp-" in key:
            bucket_str = key.split("bucket=")[1].split("/")[0]
            if int(bucket_str) == self.fail_bucket:
                self.attempts += 1
                if self.attempts <= self.fail_times:
                    raise RuntimeError(
                        f"simulated transient MinIO failure #{self.attempts} "
                        f"uploading tmp for bucket {bucket_str}")
        return self.inner.put_file(key, local_path, **kw)


def _read_bucket_table(store: FakeStore, bucket: int):
    import io

    from exonit.completeness_store import read_parquet_file, table_rows

    key = f"buckets/bucket={bucket:03d}/data.parquet"
    data = store.get_bytes(key)
    return table_rows(read_parquet_file(io.BytesIO(data)))


# --------------------------------------------------------------------------- #
# (a) fresh staging over an empty bucket
# --------------------------------------------------------------------------- #

@pg_only
def test_compact_fresh_staging_over_empty_bucket(tmp_path, pg_ready):
    rows = [
        ("100001", 1.0, 5.0, 0.11),
        ("100001", 2.0, 10.0, 0.22),
        ("100002", 1.0, 5.0, 0.33),
        ("100003", 4.0, 20.0, 0.44),
        ("100003", 1.0, 5.0, 0.55),
        ("100004", 2.0, 10.0, 0.66),
    ]
    _insert_staging(pg_ready, rows)
    expected_buckets = sorted({bucket_of(r[0]) for r in rows})
    assert len(expected_buckets) >= 2, "test needs rows spanning >=2 buckets"

    store = FakeStore(tmp_path / "minio")
    result = cc.compact(pg_ready, store)

    assert result["ok"] is True
    assert result["skipped"] is False
    assert result["buckets_touched"] == expected_buckets
    assert result["rows_merged"] == len(rows)
    assert result["tics_new"] == 4
    assert result["tics_updated"] == 0
    assert result["staging_deleted"] == len(rows)

    # staging drained
    assert _staging_count(pg_ready) == 0

    # completeness_tics populated correctly
    tics = _tics_table(pg_ready)
    assert set(tics.keys()) == {"100001", "100002", "100003", "100004"}
    for tic_id, bucket in tics.items():
        assert bucket == bucket_of(tic_id)

    # aggregates match hand-computed sums
    expected = _expected_agg(rows)
    got = _agg_by_cell(pg_ready)
    assert set(got.keys()) == set(expected.keys())
    for key, (exp_sum, exp_n) in expected.items():
        got_sum, got_n = got[key]
        assert got_n == exp_n
        assert got_sum == pytest.approx(exp_sum, abs=1e-9)

    # Parquet bucket files hold exactly the seeded rows, no more/less.
    total_rows_in_store = 0
    for b in expected_buckets:
        bucket_rows = _read_bucket_table(store, b)
        total_rows_in_store += len(bucket_rows)
        for (tic_id, radius, period, prob) in bucket_rows:
            assert bucket_of(tic_id) == b
    assert total_rows_in_store == len(rows)

    # No leftover .tmp- objects.
    assert not any(".tmp-" in k for k in store.all_keys())


# --------------------------------------------------------------------------- #
# (b) real upsert: same PK re-written -> no duplication, new value wins
# --------------------------------------------------------------------------- #

@pg_only
def test_compact_upsert_same_pk_no_duplicate_new_value_wins(tmp_path, pg_ready):
    initial = [
        ("200001", 1.0, 5.0, 0.10),
        ("200001", 2.0, 10.0, 0.20),
        ("200002", 1.0, 5.0, 0.30),
    ]
    _insert_staging(pg_ready, initial)
    store = FakeStore(tmp_path / "minio")
    first = cc.compact(pg_ready, store)
    assert first["ok"] is True

    target_bucket = bucket_of("200001")
    rows_before = _read_bucket_table(store, target_bucket)
    n_before = len(rows_before)

    # Re-write ONE existing cell (200001, 1.0, 5.0) with a NEW value.
    _insert_staging(pg_ready, [("200001", 1.0, 5.0, 0.99)])
    second = cc.compact(pg_ready, store)

    assert second["ok"] is True
    assert second["rows_merged"] == 1
    assert second["tics_new"] == 0
    assert second["tics_updated"] == 1
    assert second["staging_deleted"] == 1

    rows_after = _read_bucket_table(store, target_bucket)
    assert len(rows_after) == n_before, "PK update must not duplicate the row"

    by_pk = {(t, r, p): prob for (t, r, p, prob) in rows_after}
    assert by_pk[("200001", 1.0, 5.0)] == pytest.approx(0.99)

    # Aggregate cell (1.0, 5.0) must reflect the NEW value, not old+new.
    expected_sum = 0.99 + 0.30  # 200001 updated + 200002 unchanged, both cell (1.0, 5.0)
    got = _agg_by_cell(pg_ready)
    got_sum, got_n = got[(1.0, 5.0)]
    assert got_n == 2  # still 2 stars in this cell, not 3
    assert got_sum == pytest.approx(expected_sum, abs=1e-9)

    assert _staging_count(pg_ready) == 0


# --------------------------------------------------------------------------- #
# (c) crash between buckets -> staging intact -> re-run converges
# --------------------------------------------------------------------------- #

@pg_only
def test_compact_crash_between_buckets_staging_intact_and_rerun_converges(
        tmp_path, pg_ready):
    """Also exercises the (default-on) parallel bucket path: with >=2
    buckets and EXONIT_COMPACT_WORKERS unset (default 4), the buckets in
    this test run concurrently across threads — the assertions below prove
    the all-or-nothing staging semantics survived that change exactly."""
    rows = [
        ("300001", 1.0, 5.0, 0.11),
        ("300002", 2.0, 10.0, 0.22),
        ("300003", 4.0, 20.0, 0.33),
        ("300004", 1.0, 5.0, 0.44),
        ("300005", 2.0, 10.0, 0.55),
        ("300006", 4.0, 20.0, 0.66),
    ]
    _insert_staging(pg_ready, rows)
    expected_buckets = sorted({bucket_of(r[0]) for r in rows})
    assert len(expected_buckets) >= 2, "test needs rows spanning >=2 buckets"

    minio_root = tmp_path / "minio"
    inner = FakeStore(minio_root)
    flaky = FlakyStore(inner, fail_bucket=expected_buckets[1])

    with pytest.raises(RuntimeError, match="simulated crash"):
        cc.compact(pg_ready, flaky)

    # with_s3_retries (2026-07-21) exhausted every attempt before this
    # propagated -- 1 initial + 4 retries, reset_client() before each of
    # the 4 retries -- proving the bucket really was retried, not just
    # failed outright on the first try.
    assert flaky.reset_calls == 4

    # Staging must be COMPLETELY intact — no partial delete.
    assert _staging_count(pg_ready) == len(rows)
    assert _staging_tics(pg_ready) == {r[0] for r in rows}

    # Retry with a healthy store (same on-disk root — persistent "MinIO").
    result = cc.compact(pg_ready, FakeStore(minio_root))
    assert result["ok"] is True
    assert result["buckets_touched"] == expected_buckets
    assert _staging_count(pg_ready) == 0

    expected = _expected_agg(rows)
    got = _agg_by_cell(pg_ready)
    assert set(got.keys()) == set(expected.keys())
    for key, (exp_sum, exp_n) in expected.items():
        got_sum, got_n = got[key]
        assert got_n == exp_n
        assert got_sum == pytest.approx(exp_sum, abs=1e-9)

    tics = _tics_table(pg_ready)
    assert set(tics.keys()) == {r[0] for r in rows}

    final_store = FakeStore(minio_root)
    total_rows = sum(len(_read_bucket_table(final_store, b)) for b in expected_buckets)
    assert total_rows == len(rows)

    # The manifest (written only after ALL buckets succeed) covers every
    # touched bucket after the convergent re-run.
    manifest = json.loads(final_store.get_bytes(cc.MANIFEST_KEY))
    assert set(manifest["buckets"]) == {f"{b:03d}" for b in expected_buckets}


# --------------------------------------------------------------------------- #
# (c2) 2026-07-21 fix: with_s3_retries recovers a TRANSIENT bucket upload
# failure — the incident that motivated this (compact_completeness.py's
# remote fast-path dying with rc=1 after 28 min under MinIO load from a
# concurrent download) must no longer fail the whole pass on a blip that a
# retry would have absorbed.
# --------------------------------------------------------------------------- #

@pg_only
def test_compact_bucket_upload_recovers_after_two_transient_failures(
        tmp_path, pg_ready):
    """The tmp-key upload for one bucket fails twice (simulating a MinIO
    read-timeout under load), succeeds on the 3rd attempt: the whole
    compaction still ends up ok=True, that bucket fully compacted, staging
    fully drained — no bucket marked failed, no exception propagates.

    A single tic (-> a single bucket touched) so the failing/retrying store
    is exercised on exactly one, deterministic bucket key."""
    rows = [
        ("110001", 1.0, 5.0, 0.11),
        ("110001", 2.0, 10.0, 0.22),
    ]
    _insert_staging(pg_ready, rows)
    target_bucket = bucket_of("110001")

    inner = FakeStore(tmp_path / "minio")
    flaky = FlakyNTimesStore(inner, fail_bucket=target_bucket, fail_times=2)

    result = cc.compact(pg_ready, flaky)

    assert result["ok"] is True
    assert result["buckets_touched"] == [target_bucket]
    assert result["staging_deleted"] == len(rows)
    assert flaky.attempts == 3, "must succeed on the 3rd attempt"
    assert flaky.reset_calls == 2, "reset_client before each of the 2 retries, not the 1st try"

    assert _staging_count(pg_ready) == 0
    bucket_rows = _read_bucket_table(inner, target_bucket)
    assert len(bucket_rows) == len(rows)

    manifest = json.loads(inner.get_bytes(cc.MANIFEST_KEY))
    assert set(manifest["buckets"]) == {f"{target_bucket:03d}"}
    assert not any(".tmp-" in k for k in inner.all_keys())


# --------------------------------------------------------------------------- #
# (d) MVCC race (the review's CRITICAL, reproduced): a concurrent transaction
# allocates its BIGSERIAL id BEFORE the compaction's read but commits AFTER
# it. Under the old MAX(id)-watermark delete, that never-read row's id sat
# below the watermark and the final DELETE destroyed it without it ever being
# merged — silent data loss. With delete-exactly-what-was-read semantics the
# row survives BY CONSTRUCTION and the next compaction merges it.
# --------------------------------------------------------------------------- #

@pg_only
def test_row_committed_after_read_survives_delete_and_next_compaction_merges_it(
        tmp_path, pg_ready):
    import psycopg

    # Conn A: open transaction, INSERT without commit. nextval() is
    # NON-transactional, so A's row takes the LOWEST id in staging right now
    # — exactly the id ordering that defeated the MAX(id) watermark.
    conn_a = psycopg.connect(pg_ready)  # autocommit=False: tx stays open
    conn_a.execute(
        "INSERT INTO completeness_staging "
        "(tic_id, radius_earth, period_days, detection_prob) "
        "VALUES (%s, %s, %s, %s)", ("999999", 9.0, 90.0, 0.42))

    # Committed rows land AFTER A's insert -> their ids are HIGHER than A's.
    committed = [
        ("400001", 1.0, 5.0, 0.10),
        ("400002", 2.0, 10.0, 0.20),
    ]
    _insert_staging(pg_ready, committed)

    # Interleave: A commits right after the compaction's read (invisible to
    # the read; visible to the delete — the reviewer's reproduction).
    original_read = cc._read_staging_rows

    def _patched(conn, limit=None):
        rows = original_read(conn, limit)
        conn_a.commit()
        return rows

    cc._read_staging_rows = _patched
    try:
        store = FakeStore(tmp_path / "minio")
        result = cc.compact(pg_ready, store)
    finally:
        cc._read_staging_rows = original_read
        conn_a.close()

    # Only the 2 rows actually read were deleted; A's row SURVIVES in staging.
    assert result["staging_deleted"] == len(committed)
    assert _staging_tics(pg_ready) == {"999999"}

    # And it was NOT merged in this pass (it was never read).
    tics = _tics_table(pg_ready)
    assert "999999" not in tics
    assert set(tics.keys()) == {"400001", "400002"}

    # A second compaction picks it up and drains staging completely.
    result2 = cc.compact(pg_ready, FakeStore(tmp_path / "minio"))
    assert result2["ok"] is True
    assert result2["staging_deleted"] == 1
    assert _staging_count(pg_ready) == 0
    tics2 = _tics_table(pg_ready)
    assert "999999" in tics2
    got_sum, got_n = _agg_by_cell(pg_ready)[(9.0, 90.0)]
    assert got_n == 1
    assert got_sum == pytest.approx(0.42, abs=1e-9)


# --------------------------------------------------------------------------- #
# (e) advisory lock: concurrent compaction exits clean with a message
# --------------------------------------------------------------------------- #

@pg_only
def test_advisory_lock_second_concurrent_compaction_skips_cleanly(tmp_path, pg_ready):
    import psycopg

    _insert_staging(pg_ready, [("500001", 1.0, 5.0, 0.5)])

    holder = psycopg.connect(pg_ready, autocommit=True)
    try:
        acquired = holder.execute(
            "SELECT pg_try_advisory_lock(%s, 0)", (cc._COMPACTION_PG_LOCK_NS,)
        ).fetchone()[0]
        assert acquired is True

        store = FakeStore(tmp_path / "minio")
        result = cc.compact(pg_ready, store)

        assert result["ok"] is True
        assert result["skipped"] is True
        assert result["reason"] == "locked_by_other_process"
        assert result["message"]

        # Nothing touched: staging untouched, no buckets written.
        assert _staging_count(pg_ready) == 1
        assert store.all_keys() == []
    finally:
        holder.execute("SELECT pg_advisory_unlock(%s, 0)", (cc._COMPACTION_PG_LOCK_NS,))
        holder.close()

    # Lock released -> a normal compaction now proceeds.
    store2 = FakeStore(tmp_path / "minio2")
    result2 = cc.compact(pg_ready, store2)
    assert result2["ok"] is True
    assert result2["skipped"] is False
    assert _staging_count(pg_ready) == 0


# --------------------------------------------------------------------------- #
# (f) --dry-run: informs without writing anything
# --------------------------------------------------------------------------- #

@pg_only
def test_dry_run_never_writes_store_or_pg(pg_ready):
    rows = [
        ("600001", 1.0, 5.0, 0.1),
        ("600002", 2.0, 10.0, 0.2),
        ("600002", 4.0, 20.0, 0.3),
    ]
    _insert_staging(pg_ready, rows)

    result = cc.compact(pg_ready, PoisonStore(), dry_run=True)

    assert result["dry_run"] is True
    assert result["ok"] is True
    assert result["skipped"] is False
    assert set(result["buckets_touched"]) == {bucket_of(r[0]) for r in rows}
    assert result["rows_merged"] == len(rows)
    assert result["staging_deleted"] == 0
    assert result["tics_new"] == 2
    assert result["tics_updated"] == 0

    # Nothing actually written anywhere.
    assert _staging_count(pg_ready) == len(rows)
    assert _agg_by_cell(pg_ready) == {}
    assert _tics_table(pg_ready) == {}


@pg_only
def test_dry_run_reports_tics_updated_against_already_compacted_tics(tmp_path, pg_ready):
    _insert_staging(pg_ready, [("700001", 1.0, 5.0, 0.1)])
    store = FakeStore(tmp_path / "minio")
    cc.compact(pg_ready, store)  # compact once for real

    _insert_staging(pg_ready, [
        ("700001", 2.0, 10.0, 0.2),  # existing tic, new cell
        ("700002", 1.0, 5.0, 0.3),   # brand new tic
    ])
    result = cc.compact(pg_ready, PoisonStore(), dry_run=True)
    assert result["tics_new"] == 1
    assert result["tics_updated"] == 1
    assert result["staging_deleted"] == 0
    # The already-compacted staging row from the real run is gone; only the
    # two just-seeded rows remain.
    assert _staging_count(pg_ready) == 2


# --------------------------------------------------------------------------- #
# Manifest: touched buckets updated (Task 2 schema, source="compaction"),
# untouched entries preserved, created if missing.
# --------------------------------------------------------------------------- #

@pg_only
def test_manifest_created_if_missing_with_task2_schema(tmp_path, pg_ready):
    import hashlib

    rows = [
        ("100001", 1.0, 5.0, 0.11),
        ("100002", 2.0, 10.0, 0.22),
        ("100003", 4.0, 20.0, 0.33),
    ]
    _insert_staging(pg_ready, rows)
    store = FakeStore(tmp_path / "minio")
    assert not store.exists(cc.MANIFEST_KEY)  # bucket never migrated

    result = cc.compact(pg_ready, store)

    manifest = json.loads(store.get_bytes(cc.MANIFEST_KEY))
    assert set(manifest.keys()) == {"buckets"}
    assert set(manifest["buckets"]) == {
        f"{b:03d}" for b in result["buckets_touched"]}

    per_bucket_tics = {}
    for tic_id, _r, _p, _prob in rows:
        per_bucket_tics.setdefault(bucket_of(tic_id), set()).add(tic_id)

    for key, entry in manifest["buckets"].items():
        # Same per-bucket schema as the Task 2 migrator's manifest.
        assert set(entry.keys()) == {"rows", "n_tics", "sha256", "written_at",
                                      "source"}
        assert entry["source"] == "compaction"
        bucket = int(key)
        bucket_rows = _read_bucket_table(store, bucket)
        assert entry["rows"] == len(bucket_rows)
        assert entry["n_tics"] == len(per_bucket_tics[bucket])
        # sha256 matches the object actually stored.
        stored = store.get_bytes(f"buckets/bucket={key}/data.parquet")
        assert entry["sha256"] == hashlib.sha256(stored).hexdigest()

    assert sum(e["rows"] for e in manifest["buckets"].values()) == len(rows)
    json.dumps(manifest)  # JSON round-trippable


@pg_only
def test_manifest_preserves_untouched_bucket_entries(tmp_path, pg_ready):
    store = FakeStore(tmp_path / "minio")

    # Pre-existing manifest (as the Task 2 migrator would have written it)
    # with an entry for a bucket this compaction will NOT touch.
    untouched_key = "250"
    untouched_entry = {
        "rows": 5, "n_tics": 1, "sha256": "0" * 64,
        "written_at": "2026-01-01T00:00:00+00:00",
        "source": "migrate_completeness_to_parquet.py:/snap.db",
    }
    store.put_bytes(
        cc.MANIFEST_KEY,
        json.dumps({"buckets": {untouched_key: untouched_entry}}).encode("utf-8"))

    rows = [("100001", 1.0, 5.0, 0.11), ("100002", 2.0, 10.0, 0.22)]
    _insert_staging(pg_ready, rows)
    touched = {bucket_of(r[0]) for r in rows}
    assert int(untouched_key) not in touched  # precondition

    result = cc.compact(pg_ready, store)
    assert result["ok"] is True

    manifest = json.loads(store.get_bytes(cc.MANIFEST_KEY))
    # Untouched entry preserved VERBATIM; touched buckets added/updated.
    assert manifest["buckets"][untouched_key] == untouched_entry
    for b in touched:
        assert manifest["buckets"][f"{b:03d}"]["source"] == "compaction"
    assert set(manifest["buckets"]) == {untouched_key} | {
        f"{b:03d}" for b in touched}


# --------------------------------------------------------------------------- #
# Tie-break: identical written_at (same transaction) -> higher id wins
# --------------------------------------------------------------------------- #

@pg_only
def test_tiebreak_identical_written_at_higher_id_wins(tmp_path, pg_ready):
    import psycopg

    # Two writes of the SAME cell inside ONE transaction: written_at is the
    # transaction-stable now() for both, so only the id can break the tie.
    with psycopg.connect(pg_ready) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO completeness_staging "
                "(tic_id, radius_earth, period_days, detection_prob) "
                "VALUES (%s, %s, %s, %s)", ("800001", 1.0, 5.0, 0.10))
            cur.execute(
                "INSERT INTO completeness_staging "
                "(tic_id, radius_earth, period_days, detection_prob) "
                "VALUES (%s, %s, %s, %s)", ("800001", 1.0, 5.0, 0.77))
        conn.commit()

    # Precondition: both rows really share ONE written_at value.
    with psycopg.connect(pg_ready, autocommit=True) as conn:
        distinct_ts = conn.execute(
            "SELECT COUNT(DISTINCT written_at) FROM completeness_staging"
        ).fetchone()[0]
        assert distinct_ts == 1

    store = FakeStore(tmp_path / "minio")
    result = cc.compact(pg_ready, store)

    assert result["rows_merged"] == 1  # one PK after dedup
    bucket_rows = _read_bucket_table(store, bucket_of("800001"))
    assert bucket_rows == [("800001", 1.0, 5.0, 0.77)]  # higher id won

    got_sum, got_n = _agg_by_cell(pg_ready)[(1.0, 5.0)]
    assert got_n == 1
    assert got_sum == pytest.approx(0.77, abs=1e-9)


# --------------------------------------------------------------------------- #
# Empty staging -> "nothing to do", no lock contention needed
# --------------------------------------------------------------------------- #

@pg_only
def test_compact_empty_staging_is_a_noop(tmp_path, pg_ready):
    store = FakeStore(tmp_path / "minio")
    result = cc.compact(pg_ready, store)
    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["reason"] == "nothing_to_do"
    assert result["buckets_touched"] == []
    assert result["staging_deleted"] == 0
    assert store.all_keys() == []


# --------------------------------------------------------------------------- #
# Parallelization (Fase 4a Task 4 fast-path, 2026-07-17): _compact_workers()
# env-var parsing, ThreadPoolExecutor sizing, and sequential-vs-parallel
# parity (same input -> byte-identical final state).
# --------------------------------------------------------------------------- #

def test_compact_workers_defaults_to_four(monkeypatch):
    monkeypatch.delenv("EXONIT_COMPACT_WORKERS", raising=False)
    assert cc._compact_workers() == 4


def test_compact_workers_reads_env_override(monkeypatch):
    monkeypatch.setenv("EXONIT_COMPACT_WORKERS", "8")
    assert cc._compact_workers() == 8


def test_compact_workers_invalid_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("EXONIT_COMPACT_WORKERS", "not-a-number")
    assert cc._compact_workers() == 4


def test_compact_workers_non_positive_value_clamped_to_one(monkeypatch):
    monkeypatch.setenv("EXONIT_COMPACT_WORKERS", "0")
    assert cc._compact_workers() == 1
    monkeypatch.setenv("EXONIT_COMPACT_WORKERS", "-3")
    assert cc._compact_workers() == 1


@pg_only
def test_compact_uses_threadpoolexecutor_sized_by_env_var(tmp_path, pg_ready, monkeypatch):
    """Confirms the wiring, not just the helper in isolation: a real compact()
    call actually constructs its ThreadPoolExecutor with EXONIT_COMPACT_WORKERS
    workers."""
    import concurrent.futures

    monkeypatch.setenv("EXONIT_COMPACT_WORKERS", "7")
    _insert_staging(pg_ready, [("900001", 1.0, 5.0, 0.5)])

    seen_max_workers = []
    real_executor = concurrent.futures.ThreadPoolExecutor

    class _RecordingExecutor(real_executor):
        def __init__(self, max_workers=None, *a, **kw):
            seen_max_workers.append(max_workers)
            super().__init__(max_workers=max_workers, *a, **kw)

    monkeypatch.setattr(cc.concurrent.futures, "ThreadPoolExecutor", _RecordingExecutor)

    store = FakeStore(tmp_path / "minio")
    result = cc.compact(pg_ready, store)
    assert result["ok"] is True
    assert seen_max_workers == [7]


@pg_only
def test_compact_sequential_vs_parallel_same_input_identical_final_state(
        tmp_path, pg_ready, monkeypatch):
    """The core parity guarantee from the spec: the SAME staging rows,
    compacted once with EXONIT_COMPACT_WORKERS=1 (effectively sequential,
    one bucket at a time) and once with EXONIT_COMPACT_WORKERS=8 (buckets
    genuinely overlapping across threads), must produce IDENTICAL bucket
    Parquet content, PG aggregates, completeness_tics and manifest entries
    (ignoring the inherently-different 'written_at' timestamps)."""
    rows = [
        ("910001", 1.0, 5.0, 0.11),
        ("910002", 2.0, 10.0, 0.22),
        ("910003", 4.0, 20.0, 0.33),
        ("910004", 1.0, 5.0, 0.44),
        ("910005", 2.0, 10.0, 0.55),
        ("910006", 4.0, 20.0, 0.66),
        ("910007", 8.0, 40.0, 0.77),
        ("910008", 16.0, 80.0, 0.88),
    ]
    expected_buckets = sorted({bucket_of(r[0]) for r in rows})
    assert len(expected_buckets) >= 2, "test needs rows spanning >=2 buckets"

    # --- Pass 1: sequential (1 worker) ---
    monkeypatch.setenv("EXONIT_COMPACT_WORKERS", "1")
    _insert_staging(pg_ready, rows)
    store_seq = FakeStore(tmp_path / "minio_seq")
    result_seq = cc.compact(pg_ready, store_seq)
    assert result_seq["ok"] is True
    seq_agg = _agg_by_bucket_cell(pg_ready)
    seq_tics = _tics_table(pg_ready)
    seq_bucket_rows = {b: sorted(_read_bucket_table(store_seq, b)) for b in expected_buckets}
    seq_manifest = json.loads(store_seq.get_bytes(cc.MANIFEST_KEY))

    # Reset PG state, reseed the SAME rows for a clean second pass (schema
    # already provisioned by the pg_ready fixture).
    _truncate_pg(pg_ready)

    # --- Pass 2: parallel (8 workers, more than the number of buckets) ---
    monkeypatch.setenv("EXONIT_COMPACT_WORKERS", "8")
    _insert_staging(pg_ready, rows)
    store_par = FakeStore(tmp_path / "minio_par")
    result_par = cc.compact(pg_ready, store_par)
    assert result_par["ok"] is True
    par_agg = _agg_by_bucket_cell(pg_ready)
    par_tics = _tics_table(pg_ready)
    par_bucket_rows = {b: sorted(_read_bucket_table(store_par, b)) for b in expected_buckets}
    par_manifest = json.loads(store_par.get_bytes(cc.MANIFEST_KEY))

    # Summary counters identical (order-independent totals).
    for key in ("ok", "buckets_touched", "rows_merged", "tics_new",
                "tics_updated", "staging_deleted"):
        assert result_seq[key] == result_par[key], key

    # PG aggregates, completeness_tics, and Parquet bucket content identical.
    assert seq_agg == par_agg
    assert seq_tics == par_tics
    assert seq_bucket_rows == par_bucket_rows

    # Manifest identical modulo the per-entry 'written_at' timestamp.
    assert set(seq_manifest["buckets"]) == set(par_manifest["buckets"])
    for key in seq_manifest["buckets"]:
        seq_entry = dict(seq_manifest["buckets"][key])
        par_entry = dict(par_manifest["buckets"][key])
        seq_entry.pop("written_at")
        par_entry.pop("written_at")
        assert seq_entry == par_entry


# --------------------------------------------------------------------------- #
# CLI smoke test
# --------------------------------------------------------------------------- #

def test_parse_args_defaults():
    args = cc.parse_args(["--pg", "postgresql://x/y"])
    assert args.pg_dsn == "postgresql://x/y"
    assert args.s3_bucket == "exonit-completeness"
    assert args.dry_run is False


def test_parse_args_dry_run_flag():
    args = cc.parse_args(["--pg", "postgresql://x/y", "--dry-run"])
    assert args.dry_run is True


class TestCompactProgressReporting:
    """compact_completeness reports one mark per finished bucket."""

    def test_reports_a_mark_per_completed_bucket(self, monkeypatch):
        import scripts.compact_completeness as CC

        sent = []

        class FakeReporter:
            enabled = True

            def __init__(self, **kw):
                sent.append(("init", kw))

            def report(self, done, **kw):
                sent.append(("report", done, kw))

            def finish(self, **kw):
                sent.append(("finish", kw))

        monkeypatch.setattr(CC, "ProgressReporter", FakeReporter)
        rep = CC._progress_reporter(total_buckets=3)
        CC._report_bucket_done(rep, 1)
        CC._report_bucket_done(rep, 2)
        CC._report_bucket_done(rep, 3)
        rep.finish()

        assert sent[0][1]["unit"] == "buckets"
        assert sent[0][1]["total"] == 3
        assert sent[0][1]["substep"] == "compact_completeness"
        assert [s[1] for s in sent if s[0] == "report"] == [1, 2, 3]

    def test_a_broken_reporter_never_breaks_compaction(self, monkeypatch):
        import scripts.compact_completeness as CC

        class ExplodingReporter:
            enabled = True

            def __init__(self, **kw):
                pass

            def report(self, *a, **kw):
                raise RuntimeError("telemetry exploded")

        monkeypatch.setattr(CC, "ProgressReporter", ExplodingReporter)
        rep = CC._progress_reporter(total_buckets=2)
        CC._report_bucket_done(rep, 1)     # must not raise

