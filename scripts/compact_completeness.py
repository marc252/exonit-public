#!/usr/bin/env python3
"""Recurring compaction: ``completeness_staging`` -> bucketed Parquet (MinIO)
+ exact per-bucket aggregates in PostgreSQL (Phase 4a, Task 3).

Task 1 made the completeness write-path (``ExoNitDB.add_completeness_many``,
``EXONIT_COMPLETENESS_PG=1``) a pure append into ``completeness_staging`` — no
dedup, no upsert, the same ``(tic_id, radius_earth, period_days)`` cell can
appear more than once. Task 2 did the one-shot historical snapshot -> Parquet
migration. This script is the ongoing job that keeps the two in sync: it
drains ``completeness_staging`` into the bucketed Parquet files Task 2 wrote
(``buckets/bucket=BBB/data.parquet``, MinIO) and recomputes
``completeness_agg_bucket`` / ``completeness_tics`` exactly from the merged
result — invoked at the FINALIZE step of every discovery run (Task 4) and
manually as an operator tool.

Algorithm (crash-safe — no destructive step before the new state is fully
persisted; see docs/superpowers/plans/2026-07-05-phase4-completeness-parquet.md,
Task 3):
  1. Read every currently-visible ``completeness_staging`` row (``ORDER BY
     written_at, id``) and capture the exact list of ``id`` values read. The
     final delete removes EXACTLY those ids and nothing else — never an
     ``id <= watermark`` range. Why this matters (MVCC race, empirically
     reproduced in review): a concurrent writer's transaction can allocate a
     BIGSERIAL id BEFORE this read but commit AFTER it — its row is invisible
     to the read yet its id is below any ``MAX(id)`` watermark, so a
     watermark-range delete would destroy a row that was never merged.
     Deleting only the ids actually read makes the no-data-loss guarantee
     hold BY CONSTRUCTION: a row never read is never deleted; it is simply
     picked up by the next compaction.
  2. Dedup in memory, last-write-wins per (tic_id, radius_earth, period_days):
     rows are read ``ORDER BY written_at, id`` so a plain dict build (later
     entries overwrite earlier ones) implements the tiebreak exactly —
     including identical ``written_at`` values (rows written inside one
     transaction all share the same ``now()``), where the higher ``id`` wins.
  3. Group the deduped rows by bucket (``bucket_of(tic_id)``). For each
     affected bucket: download the existing ``data.parquet`` (if any), merge
     with staging (staging ALWAYS wins on PK conflict), upload to a
     ``.tmp-<runid>`` key, verify the round-tripped footer's row count,
     upload the real key (a single PutObject is atomic per-object on
     S3/MinIO), delete the tmp key (best-effort), recompute
     ``completeness_agg_bucket`` for that bucket EXACTLY from the merged
     table (DELETE + INSERT in one transaction — never deltas), and upsert
     ``completeness_tics`` (``ON CONFLICT (tic_id) DO NOTHING``). Buckets are
     independent by construction (partitioned by hash of ``tic_id``), so this
     step runs across a ``ThreadPoolExecutor`` (``EXONIT_COMPACT_WORKERS``,
     default 4) — I/O-bound (MinIO + PG round trips), each worker thread
     opens its OWN PostgreSQL connection (psycopg connections are not
     thread-safe to share). Per-bucket log lines can interleave/reorder
     across threads; the aggregated summary below is identical regardless of
     processing order.
  4. Update ``buckets/_manifest.json`` for the touched buckets — same
     per-bucket schema as the Task 2 migrator (``rows``, ``n_tics``,
     ``sha256``, ``written_at``, ``source="compaction"``); entries for
     untouched buckets are preserved verbatim; a missing manifest (bucket
     never migrated) is created with just the touched buckets. Written ONCE,
     after every bucket thread has finished (success or failure) — never
     incrementally per bucket.
  5. Only once every affected bucket AND the manifest have succeeded:
     ``DELETE FROM completeness_staging WHERE id = ANY(<ids read>)``, chunked
     at 100k ids per statement (param-limit gotcha).
  6. If ANY bucket thread (or the manifest write) raises, step 5 never runs —
     staging is left FULLY intact, exactly as in the pre-parallel sequential
     version (not just the failed bucket's rows: the single DELETE covers
     every id read in step 1, and it either runs once everything has
     succeeded or not at all). Every already-written bucket's Parquet+PG
     state IS already persisted at that point (idempotent, whole-table
     recompute — not a delta), so a retry — whether the next scheduled
     compaction or a manual re-run — redoes the exact same merge for every
     bucket and converges to the same end state a single successful run
     would have reached.
  7. A PostgreSQL session advisory lock (dedicated namespace, distinct from
     ``exonit.server._AUTOFILL_PG_LOCK_NS``), held on the MAIN connection for
     the whole run (not the per-bucket worker connections), guards against
     two compactions running concurrently; if already held, this exits
     cleanly reporting ``skipped``/``locked_by_other_process`` rather than
     racing.

``--dry-run`` still connects to PG (it needs the real staging state to report
anything meaningful) but never touches the object store and never writes PG —
report-only, and skips the advisory lock entirely since nothing exclusive
happens.

Batch mode (``--batch-rows``, exonit-compactd Component 1, 2026-09-09
design): the incident that motivated this was ``completeness_staging``
growing to 82.7M rows while the remote SSH compaction path died silently —
step 1's single unbounded read would then have pulled the whole table into
one process's memory. ``--batch-rows N`` (env ``EXONIT_COMPACT_BATCH_ROWS``,
default 2000000) bounds step 1 to at most N rows per read
(``ORDER BY written_at, id LIMIT N``) and repeats steps 1-5 in a loop until
``completeness_staging`` is drained: each iteration captures its OWN exact
list of ids (same crash-safe semantics as above, just scoped to the batch —
still never a range, still never touches an id it did not itself read),
merges + deletes that batch, then reads the next one. The advisory lock
(step 7) is acquired ONCE and held for the whole loop, not re-acquired per
batch. ``--batch-rows 0`` disables the loop entirely: a single unbounded
read and one merge pass, byte-for-byte the pre-batch-mode behavior — pass
it explicitly to opt back into the old one-shot path (the CLI default,
when ``--batch-rows`` is omitted, is 2000000, i.e. batch mode IS the
default now). The final JSON summary accumulates every batch's counters
(``rows_merged``, ``tics_new``, ``tics_updated``, ``staging_deleted`` summed;
``buckets_touched`` unioned) and adds ``batches: int``.
"""

import argparse
import concurrent.futures
import io
import json
import logging
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Optional

from exonit.completeness_store import (
    PK_COLUMNS,
    aggregate_bucket_rows,
    assert_unique_pk,
    bucket_of,
    parquet_row_count,
    read_parquet_file,
    rows_to_table,
    sha256_of_file,
)
from exonit.object_store import ObjectStore, with_s3_retries
from exonit.progress_client import ProgressReporter


log = logging.getLogger("compact_completeness")

# Arbitrary int4 namespace for pg_try_advisory_lock(int4, int4) — a NEW
# namespace distinct from exonit.server._AUTOFILL_PG_LOCK_NS (4558), so
# compaction and the discovery autofill guard can never collide.
_COMPACTION_PG_LOCK_NS = 4559

# Same object key as the Task 2 migrator's manifest — compaction updates the
# touched buckets' entries in place.
MANIFEST_KEY = "buckets/_manifest.json"

# Ids per DELETE ... WHERE id = ANY(%s) statement. psycopg3 adapts the whole
# Python list to ONE array parameter, but chunking keeps each statement's
# array (and the resulting per-statement work) bounded — the >65535-param
# class of failure (gotcha_postprocess_in_list_65535_param_limit) must never
# get a chance to resurface if this ever changes to an IN list.
_DELETE_IDS_CHUNK = 100_000

# Default number of buckets compacted concurrently (2026-07-17, "fast completeness
# compaction" spec). Buckets are independent by construction
# (partitioned by hash of tic_id) — I/O-bound work (MinIO + PG round trips
# dominate, not CPU), so a ThreadPoolExecutor genuinely overlaps them despite
# the GIL. Overridable via EXONIT_COMPACT_WORKERS for ops tuning / tests.
_DEFAULT_COMPACT_WORKERS = 4


def _log_s3_retry(op: str, key: str):
    """``on_retry`` callback factory for ``with_s3_retries`` calls in this
    module (2026-07-21 fix — see ``exonit.object_store.with_s3_retries``'s
    docstring): logs a warning per retry instead of the caller hand-rolling
    its own retry loop. *op*/*key* just identify what was being retried in
    the log line — e.g. ``_log_s3_retry("bucket-read", key)``."""
    def _cb(attempt, attempts, delay, exc):
        log.warning(
            "S3 %s retry %d/%d for %s in %ds: %s",
            op, attempt, attempts - 1, key, delay, exc)
    return _cb


def _compact_workers() -> int:
    """Worker thread count for the per-bucket ThreadPoolExecutor — at least
    1 (an unparsable or non-positive override falls back to the default
    rather than silently disabling parallelism or raising)."""
    raw = os.environ.get("EXONIT_COMPACT_WORKERS", "")
    try:
        n = int(raw) if raw.strip() else _DEFAULT_COMPACT_WORKERS
    except ValueError:
        log.warning("EXONIT_COMPACT_WORKERS=%r is not an int; using default %d",
                    raw, _DEFAULT_COMPACT_WORKERS)
        n = _DEFAULT_COMPACT_WORKERS
    return max(1, n)


# --------------------------------------------------------------------------- #
# 1) Staging read — capture exactly what is visible NOW (ids included)
# --------------------------------------------------------------------------- #

def _read_staging_rows(conn, limit: Optional[int] = None) -> list:
    """Every ``completeness_staging`` row visible to this connection right
    now, ordered ``(written_at, id)`` ascending — the order last-write-wins
    dedup relies on. The returned rows' ``id`` values (last column) are the
    ONLY ids the final delete may touch (algorithm step 1).

    *limit* (batch mode): when set, reads at most that many rows via
    ``LIMIT`` on the same ``ORDER BY`` — still exactly the rows visible NOW,
    still exact ids, just truncated to the first *limit* in
    ``(written_at, id)`` order rather than the whole table. A row past the
    limit is simply picked up by the next read (either the next batch in
    this same run, or the next compaction), same as any row written after
    this read started."""
    query = ("SELECT tic_id, radius_earth, period_days, detection_prob, written_at, id "
             "FROM completeness_staging ORDER BY written_at, id")
    if limit:
        query += " LIMIT %s"
        return conn.execute(query, (limit,)).fetchall()
    return conn.execute(query).fetchall()


def _dedup_staging(rows: list) -> dict:
    """``{(tic_id, radius_earth, period_days): detection_prob}``, last write
    (by iteration order — callers pass rows already ``ORDER BY written_at,
    id``) wins."""
    deduped = {}
    for tic_id, radius, period, prob, _written_at, _row_id in rows:
        deduped[(tic_id, radius, period)] = prob
    return deduped


def _group_by_bucket(deduped: dict) -> dict:
    grouped: dict = {}
    for (tic_id, radius, period), prob in deduped.items():
        b = bucket_of(tic_id)
        grouped.setdefault(b, {})[(tic_id, radius, period)] = prob
    return grouped


def _delete_staging_ids(conn, ids: list) -> int:
    """Delete EXACTLY *ids* from ``completeness_staging`` — never a range.
    Chunked so no single statement carries an unbounded array."""
    deleted = 0
    for i in range(0, len(ids), _DELETE_IDS_CHUNK):
        chunk = ids[i:i + _DELETE_IDS_CHUNK]
        deleted += conn.execute(
            "DELETE FROM completeness_staging WHERE id = ANY(%s)", (chunk,)
        ).rowcount
    return deleted


# --------------------------------------------------------------------------- #
# 2) Per-bucket compaction: merge (staging always wins) -> upload -> aggregate
# --------------------------------------------------------------------------- #

def _assert_pk_floats_safe_for_join(table, source: str) -> None:
    """Guard for TWO semantics gaps between the pyarrow hash join used below
    and the Python-dict merge it replaces (2026-09-24 vectorization),
    discovered by this vectorization's own parity test
    (``tests/test_compact_completeness_merge_parity.py``):

    * NaN: pyarrow's join treats NaN == NaN when matching keys (IEEE-754
      says NaN never equals anything, including itself), whereas the old
      ``{(t, r, p): prob}`` dict build relied on exactly that IEEE
      behaviour — two NaN-keyed rows, even for the same ``tic_id``/
      ``period_days``, were NEVER the same dict key, so both survived until
      ``assert_unique_pk`` caught them as duplicates.
    * signed zero: the OPPOSITE problem — Python's ``==``/``hash`` treat
      ``-0.0`` and ``0.0`` as the same dict key (so the old merge silently
      collapsed such a collision to one row before ``assert_unique_pk`` ever
      ran), but pyarrow's hash-join key hashing does NOT (confirmed
      empirically: ``pc.equal(-0.0, 0.0)`` is ``True`` yet a hash join on
      those two values does not match them) — so with THIS merge both rows
      would survive the join and then ``assert_unique_pk`` would raise on
      what the old merge silently and correctly resolved.

    Both are unreachable with every actual writer of ``radius_earth``/
    ``period_days``: they are always positive, finite bin edges of a fixed
    literal grid (``worker/injection.py``'s ``radius_grid``/``period_grid``
    arrays — e.g. ``[0.5, 1.0, 1.5, ...]``, never 0 — ``float()``-cast by
    ``scripts/injection_probe_driver.py``), and a zero or NaN radius/period
    is physically meaningless besides. Fail loud instead of silently
    changing behaviour if that assumption is ever violated."""
    import pyarrow.compute as pc

    for col in ("radius_earth", "period_days"):
        arr = table.column(col)
        bad = pc.or_(pc.is_nan(arr), pc.equal(arr, 0))
        if pc.any(bad).as_py():
            raise ValueError(
                f"completeness PK column {col!r} of the {source} table has "
                f"a NaN or zero (incl. -0.0) value -- refusing to merge: "
                f"pyarrow's join key semantics for these two values diverge "
                f"from the pre-vectorization Python-dict merge (see "
                f"_assert_pk_floats_safe_for_join's docstring), and neither "
                f"value should ever occur in a real radius_earth/"
                f"period_days grid cell")


def _merge_bucket(existing_table, incoming_by_pk: dict):
    """Last-write-wins merge of a bucket's existing Parquet table with the
    incoming deduped staging rows for that bucket. Staging ALWAYS wins on a
    PK conflict. Returns a sorted, PK-unique ``pa.Table``.

    Pyarrow-native (2026-09-24 vectorization of the original ``{(t, r, p):
    prob}`` Python-dict merge — see git history for that version): building
    a full Python dict from ``existing_table`` meant materializing every one
    of a bucket's ~5M rows into Python tuples first (``table_rows``), which
    dominated compaction wall time. Instead: build the incoming rows straight
    into a ``pa.Table`` (same ``rows_to_table`` used everywhere else), drop
    the existing rows whose PK also appears in incoming via a ``"left anti"``
    hash join (existing rows NOT matched by any incoming key survive,
    matched ones are dropped because incoming always wins), then concatenate
    survivors with incoming — never touching Python-level row objects for
    the existing side. NaN and signed zero (``-0.0`` vs ``0.0``) are the two
    places pyarrow's join key semantics diverge from the old dict's
    ``==``/``hash`` semantics — see ``_assert_pk_floats_safe_for_join``,
    called below whenever both tables are non-empty (an actual join only
    happens then; the empty-side fast paths above can't produce a spurious
    PK collision either way)."""
    import pyarrow as pa

    incoming_table = rows_to_table(
        [(t, r, p, prob) for (t, r, p), prob in incoming_by_pk.items()])

    if existing_table.num_rows == 0:
        merged_table = incoming_table
    elif incoming_table.num_rows == 0:
        merged_table = existing_table
    else:
        _assert_pk_floats_safe_for_join(existing_table, "existing")
        _assert_pk_floats_safe_for_join(incoming_table, "incoming")
        surviving_existing = existing_table.join(
            incoming_table.select(list(PK_COLUMNS)),
            keys=list(PK_COLUMNS), join_type="left anti")
        merged_table = pa.concat_tables([surviving_existing, incoming_table])

    merged_table = merged_table.sort_by([(c, "ascending") for c in PK_COLUMNS])
    assert_unique_pk(merged_table)
    return merged_table


def _upload_bucket_atomic(store, key: str, merged_table, run_id: str) -> str:
    """Write *merged_table* to a local temp file, upload it to a
    ``.tmp-<run_id>`` key, verify the round-tripped footer's row count,
    upload the REAL key from the same verified local file (itself one atomic
    PutObject), then best-effort delete the tmp key. Returns the sha256 of
    the uploaded file (for the manifest entry).

    The upload-verify-upload sequence (everything except the local Parquet
    write and the final best-effort tmp-key delete) runs under
    ``with_s3_retries`` (2026-07-21 fix — see that function's docstring):
    the same MinIO-under-load transient failure that hit
    ``scripts/discovery_download.py``'s pack/upload (56 chunks, 2026-07-21
    incident) can equally hit a compaction bucket write, and previously any
    such failure raised straight out of here — marking the WHOLE bucket
    (and the whole compaction pass, since one bucket failure keeps
    ``completeness_staging`` untouched) failed on what may be a 30-second
    blip. Retrying is safe: every call inside (``put_file`` x2, ``get_bytes``
    for verification) is independently idempotent — a PUT overwrites the
    same key with the same bytes, a GET has no side effect — so redoing the
    whole sequence from a retry never produces a different result than one
    clean pass would have. A verification-row-count mismatch is also
    retried (a footer read of a still-settling MinIO write under load can
    legitimately mismatch transiently); a genuine local bug reproduces
    identically on every attempt and surfaces after the last one, same as
    any other exception here. The behaviour on TOTAL failure (every retry
    exhausted) is unchanged: the exception propagates, this bucket is
    marked failed, and ``completeness_staging`` stays untouched — only a
    transient timeout no longer kills the whole pass."""
    import pyarrow.parquet as pq

    tmp_key = f"{key}.tmp-{run_id}"
    fd, local_path = tempfile.mkstemp(suffix=".parquet")
    os.close(fd)
    try:
        pq.write_table(merged_table, local_path, compression="zstd",
                        use_dictionary=["tic_id"])
        sha256 = sha256_of_file(local_path)

        def _once():
            store.put_file(tmp_key, local_path)
            verify_bytes = store.get_bytes(tmp_key)
            verify_rows = parquet_row_count(io.BytesIO(verify_bytes))
            if verify_rows != merged_table.num_rows:
                raise RuntimeError(
                    f"parquet footer verification failed for {tmp_key}: "
                    f"expected {merged_table.num_rows} rows, got {verify_rows}")
            store.put_file(key, local_path)

        with_s3_retries(_once, ostore=store, on_retry=_log_s3_retry("upload", key))

        try:
            store.delete_object(tmp_key)
        except Exception:
            log.warning("failed to delete tmp object %s (non-fatal)", tmp_key)
        return sha256
    finally:
        try:
            os.unlink(local_path)
        except OSError:
            pass


def _write_bucket_agg_pg(conn, bucket: int, agg_rows: list) -> None:
    """DELETE + INSERT completeness_agg_bucket for *bucket* in ONE
    transaction — the aggregate is always recomputed exact from the merged
    table, never adjusted by delta."""
    with conn.transaction():
        conn.execute("DELETE FROM completeness_agg_bucket WHERE bucket = %s", (bucket,))
        if agg_rows:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO completeness_agg_bucket "
                    "(bucket, radius_earth, period_days, sum_prob, n) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    [(r["bucket"], r["radius_earth"], r["period_days"],
                      r["sum_prob"], r["n"]) for r in agg_rows],
                )


def _upsert_tics_pg(conn, bucket: int, tic_ids) -> None:
    if not tic_ids:
        return
    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO completeness_tics (tic_id, bucket) VALUES (%s, %s) "
                "ON CONFLICT (tic_id) DO NOTHING",
                [(t, bucket) for t in tic_ids],
            )


def _compact_bucket(conn, store, bucket: int, incoming_by_pk: dict, run_id: str):
    """Compact one bucket. Returns ``(tics_new, tics_updated,
    manifest_entry)`` for this bucket. Raises on any upload/verification
    failure — the caller's loop stops there, leaving staging untouched
    (crash-safety, algorithm step 6)."""
    import pyarrow.compute as pc

    key = f"buckets/bucket={bucket:03d}/data.parquet"

    def _read_existing():
        if store.exists(key):
            return read_parquet_file(io.BytesIO(store.get_bytes(key)))
        return rows_to_table([])

    # Same retry+backoff as the upload below (2026-07-21 fix) — a transient
    # read timeout under MinIO load must not abort this bucket either.
    # exists()+get_bytes() are both idempotent, so redoing the pair on
    # retry is safe.
    existing_table = with_s3_retries(
        _read_existing, ostore=store, on_retry=_log_s3_retry("read", key))
    # Vectorized tic-id set (2026-09-24, same pass as _merge_bucket): only
    # the distinct tic_ids are needed here (for new_tics/incoming_tics
    # below, not the merge itself), so pull them via pc.unique (C++, over
    # the whole ~5M-row column, deduplicated BEFORE it ever reaches Python)
    # instead of the old table_rows() call, which materialized all 4
    # columns of every row into Python tuples just to throw 3 of them away.
    existing_tics = set(pc.unique(existing_table.column("tic_id")).to_pylist())

    merged_table = _merge_bucket(existing_table, incoming_by_pk)
    sha256 = _upload_bucket_atomic(store, key, merged_table, run_id)

    agg_rows = aggregate_bucket_rows(merged_table, bucket)
    _write_bucket_agg_pg(conn, bucket, agg_rows)

    incoming_tics = {t for (t, _r, _p) in incoming_by_pk}
    new_tics = incoming_tics - existing_tics
    _upsert_tics_pg(conn, bucket, incoming_tics)

    manifest_entry = {
        "rows": merged_table.num_rows,
        "n_tics": pc.count_distinct(merged_table.column("tic_id")).as_py(),
        "sha256": sha256,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "source": "compaction",
    }
    return len(new_tics), len(incoming_tics) - len(new_tics), manifest_entry


def _compact_bucket_worker(pg_dsn: str, store, bucket: int, incoming_by_pk: dict,
                            run_id: str):
    """``ThreadPoolExecutor`` entry point for one bucket: opens a DEDICATED
    PostgreSQL connection (psycopg connections must never be shared across
    threads) and delegates to ``_compact_bucket``. Returns ``(bucket,
    tics_new, tics_updated, manifest_entry)`` — the bucket id travels with
    the result so the caller can build ``manifest_entries`` without relying
    on completion order. Raises exactly what ``_compact_bucket`` raises;
    the caller (``_compact_locked``) is responsible for the all-or-nothing
    staging-delete semantics (algorithm step 6)."""
    import psycopg

    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        n_new, n_updated, manifest_entry = _compact_bucket(
            conn, store, bucket, incoming_by_pk, run_id)
    return bucket, n_new, n_updated, manifest_entry


# --------------------------------------------------------------------------- #
# 3) Manifest: update touched buckets, preserve the rest (Task 2's schema)
# --------------------------------------------------------------------------- #

def _load_manifest(store) -> dict:
    """The existing ``buckets/_manifest.json``, or an empty skeleton when it
    does not exist yet (staging compacted into a bucket never migrated)."""
    if store.exists(MANIFEST_KEY):
        return json.loads(store.get_bytes(MANIFEST_KEY))
    return {"buckets": {}}


def _update_manifest(store, entries: dict) -> None:
    """Merge *entries* (``{"BBB": {rows, n_tics, sha256, written_at,
    source}}``) into the manifest — touched buckets replaced, untouched
    buckets' entries preserved verbatim.

    Read-then-write runs under ``with_s3_retries`` (2026-07-21 fix): both
    calls (``get_bytes``/``exists`` inside ``_load_manifest``, ``put_bytes``)
    are idempotent, so a transient MinIO failure anywhere in the sequence
    just redoes the same read-merge-write from scratch — *entries* itself is
    a plain dict read fresh from the closure each retry, never mutated by a
    previous attempt."""
    def _once():
        manifest = _load_manifest(store)
        manifest.setdefault("buckets", {}).update(entries)
        store.put_bytes(
            MANIFEST_KEY,
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
            content_type="application/json",
        )

    with_s3_retries(_once, ostore=store,
                     on_retry=_log_s3_retry("manifest-write", MANIFEST_KEY))


# --------------------------------------------------------------------------- #
# 4) Orchestration: locked real run + read-only dry-run
# --------------------------------------------------------------------------- #

def _empty_summary(*, dry_run: bool, reason: str, message: str) -> dict:
    return {
        "ok": True, "dry_run": dry_run, "skipped": True, "reason": reason,
        "message": message, "buckets_touched": [], "rows_merged": 0,
        "tics_new": 0, "tics_updated": 0, "staging_deleted": 0, "batches": 0,
    }


def _compact_one_batch(conn, store, batch_run_id: str, pg_dsn: str, rows: list) -> dict:
    """Merge + persist exactly *rows* (one batch already read by the caller
    via ``_read_staging_rows`` — possibly the WHOLE visible staging table
    when batching is off) and delete exactly their ids from
    ``completeness_staging``. This is the entire per-run body before batch
    mode existed, unchanged except for taking *rows* instead of reading them
    itself — *conn* is still the MAIN connection (holds the advisory lock,
    does the staging read/delete); per-bucket work still runs on SEPARATE
    connections opened inside each worker thread (see
    ``_compact_bucket_worker``), *pg_dsn* is what they connect with.
    Returns ``{buckets_touched, rows_merged, tics_new, tics_updated,
    staging_deleted}`` for THIS batch only — the caller (``_compact_locked``)
    accumulates across batches. Raises on any bucket failure, exactly as
    before (algorithm step 6): staging stays untouched for this batch, and
    the caller's loop stops there."""
    # The ONLY ids this batch's delete may remove: exactly what was read.
    read_ids = [r[5] for r in rows]
    deduped = _dedup_staging(rows)
    grouped = _group_by_bucket(deduped)

    workers = _compact_workers()
    log.info("compacting %d bucket(s) with %d worker thread(s)",
              len(grouped), workers)

    tics_new = 0
    tics_updated = 0
    manifest_entries = {}
    first_exc: Optional[BaseException] = None
    reporter = _progress_reporter(len(grouped))
    _report_bucket_done(reporter, 0, message=f"{len(grouped)} buckets to compact")
    n_done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_compact_bucket_worker, pg_dsn, store, bucket,
                        incoming_by_pk, batch_run_id): bucket
            for bucket, incoming_by_pk in grouped.items()
        }
        # Drain every future (success or failure) before deciding anything —
        # a bucket already submitted keeps running to completion regardless
        # of an earlier failure elsewhere, exactly mirroring the pre-
        # parallel loop's "every already-started bucket finishes" behaviour.
        for future in concurrent.futures.as_completed(futures):
            bucket = futures[future]
            try:
                _bucket, n_new, n_upd, entry = future.result()
            except Exception as exc:
                log.error("compaction FAILED for bucket %03d: %s", bucket, exc)
                if first_exc is None:
                    first_exc = exc
                continue
            tics_new += n_new
            tics_updated += n_upd
            manifest_entries[f"{bucket:03d}"] = entry
            log.info("bucket %03d: OK (%d new tics, %d updated tics)",
                      bucket, n_new, n_upd)
            n_done += 1
            _report_bucket_done(reporter, n_done,
                                message=f"bucket {futures[future]}")
    if first_exc is not None:
        # EXACTLY the pre-parallel semantics: any bucket failure means the
        # manifest is never updated and completeness_staging is never
        # touched (algorithm step 6) — not just the failed bucket's rows,
        # ALL of read_ids survive. Every bucket that DID succeed already has
        # its Parquet+PG state persisted (idempotent, whole-table recompute)
        # so a retry converges regardless.
        raise first_exc

    _update_manifest(store, manifest_entries)

    deleted = _delete_staging_ids(conn, read_ids)

    # AFTER `raise first_exc`, deliberately: finish() means done == total, and
    # a compaction that lost a bucket is not done. Reporting it complete here
    # would put a full bar on the dashboard for a run that raised.
    if reporter is not None:
        try:
            reporter.finish(message="compaction complete")
        except Exception:
            pass

    return {
        "buckets_touched": sorted(grouped.keys()),
        "rows_merged": len(deduped), "tics_new": tics_new,
        "tics_updated": tics_updated, "staging_deleted": deleted,
    }


def _compact_locked(conn, store, run_id: str, pg_dsn: str,
                     batch_rows: int = 0) -> dict:
    """*conn* is the MAIN connection (holds the advisory lock for the WHOLE
    call, batches included — one invocation is one lock holder, exactly as
    before batch mode existed).

    ``batch_rows == 0`` (default; also ``main()``'s CLI default when
    ``--batch-rows`` is omitted maps to 2000000, so this path is reached by
    passing ``0`` explicitly): a single call to ``_compact_one_batch`` with
    every currently-visible staging row, then stop — byte-for-byte the
    pre-batch-mode behavior, including never looping back for rows written
    concurrently while this call ran (those are simply left for the next
    compaction, same as always).

    ``batch_rows > 0``: repeats ``_read_staging_rows(limit=batch_rows)`` +
    ``_compact_one_batch`` until a read comes back empty — i.e. until
    ``completeness_staging`` is fully drained of everything visible as this
    loop progresses, not just what was visible at the very start. Peak
    memory is bounded by one batch's worth of rows, independent of how large
    the accumulated staging table is (the exonit-compactd design's whole
    point — module docstring). Each batch gets its own ``batch_run_id``
    (``run_id`` suffixed with the batch index) so a bucket touched in more
    than one batch never reuses a ``.tmp-<run_id>`` upload key across
    batches."""
    batches = 0
    rows_merged_total = 0
    tics_new_total = 0
    tics_updated_total = 0
    staging_deleted_total = 0
    buckets_touched: set = set()

    while True:
        rows = _read_staging_rows(conn, limit=(batch_rows or None))
        if not rows:
            break

        batch_run_id = run_id if batch_rows == 0 else f"{run_id}-b{batches}"
        batch = _compact_one_batch(conn, store, batch_run_id, pg_dsn, rows)
        batches += 1
        rows_merged_total += batch["rows_merged"]
        tics_new_total += batch["tics_new"]
        tics_updated_total += batch["tics_updated"]
        staging_deleted_total += batch["staging_deleted"]
        buckets_touched.update(batch["buckets_touched"])

        log.info("batch %d done: %d rows merged, %d staging rows deleted",
                  batches, batch["rows_merged"], batch["staging_deleted"])

        if batch_rows == 0:
            # Single-pass mode: the read above was already unbounded (every
            # row visible at that instant), so a 2nd iteration would only
            # ever pick up rows written AFTER this call started — exactly
            # what the pre-batch-mode code never did either.
            break

    if batches == 0:
        return _empty_summary(
            dry_run=False, reason="nothing_to_do",
            message="completeness_staging empty; nothing to compact")

    return {
        "ok": True, "dry_run": False, "skipped": False, "reason": None,
        "message": None, "buckets_touched": sorted(buckets_touched),
        "rows_merged": rows_merged_total, "tics_new": tics_new_total,
        "tics_updated": tics_updated_total,
        "staging_deleted": staging_deleted_total, "batches": batches,
    }


def _compact_dry_run(conn) -> dict:
    """Read-only: reports what a real compaction would do, without touching
    the object store or writing PG."""
    rows = _read_staging_rows(conn)
    if not rows:
        return _empty_summary(
            dry_run=True, reason="nothing_to_do",
            message="completeness_staging empty; nothing to compact")

    deduped = _dedup_staging(rows)
    grouped = _group_by_bucket(deduped)

    tics_new = 0
    tics_updated = 0
    for incoming_by_pk in grouped.values():
        incoming_tics = {t for (t, _r, _p) in incoming_by_pk}
        if not incoming_tics:
            continue
        existing = conn.execute(
            "SELECT tic_id FROM completeness_tics WHERE tic_id = ANY(%s)",
            (list(incoming_tics),),
        ).fetchall()
        existing_set = {r[0] for r in existing}
        tics_new += len(incoming_tics - existing_set)
        tics_updated += len(incoming_tics & existing_set)

    return {
        "ok": True, "dry_run": True, "skipped": False, "reason": None,
        "message": None, "buckets_touched": sorted(grouped.keys()),
        "rows_merged": len(deduped), "tics_new": tics_new,
        "tics_updated": tics_updated, "staging_deleted": 0,
    }


# --- progress reporting ---------------------------------------------------
# The compaction is a bounded loop over buckets that used to print exactly one
# line — the final JSON — after up to 45 minutes of work. From the dashboard's
# point of view that was indistinguishable from a hung process (2026-07-27,
# sector 14). One mark per finished bucket makes the difference visible.
#
# Every call here is defensive: telemetry must never be able to abort a
# compaction that is holding a PG advisory lock.

def _progress_reporter(total_buckets: int):
    try:
        return ProgressReporter(unit="buckets", total=total_buckets,
                                substep="compact_completeness", interval_s=10)
    except Exception:
        return None


def _report_bucket_done(reporter, n_done: int, message: str | None = None):
    if reporter is None:
        return
    try:
        reporter.report(n_done, message=message)
    except Exception:
        pass


def compact(pg_dsn: str, store, *, dry_run: bool = False,
            run_id: Optional[str] = None, batch_rows: int = 0) -> dict:
    """Run one compaction pass: ``completeness_staging`` -> bucketed Parquet
    (*store*) + exact PG aggregates + manifest. Returns a summary dict (see
    module docstring). Raises on any bucket upload/verification failure — the
    caller (``main()``/the orchestrator's FINALIZE hook) decides whether that
    is fatal; ``completeness_staging`` is guaranteed untouched for whichever
    batch was in flight when that happened.

    *batch_rows* (real runs only; ignored when *dry_run*): ``0`` (default of
    this function — NOT of the CLI, see ``_default_batch_rows``) reads and
    merges the whole visible staging table in one pass, exactly as before
    batch mode existed. A positive value bounds each read/merge/delete cycle
    to at most that many staging rows, looping until the table is drained —
    see ``_compact_locked``'s docstring for the full loop semantics.

    Counter semantics: ``tics_new``/``tics_updated`` describe the incoming
    staging tics relative to the state each bucket had AT THE TIME of this
    call. They are informational and NOT stable across a crash + retry: a
    retry after a partial run counts tics already persisted by the failed
    attempt as "updated" rather than "new". The RESULTING STATE (Parquet
    buckets, aggregates, ``completeness_tics``, manifest) IS stable — a retry
    converges to exactly what a single successful run would have produced.

    ``dry_run=True`` never touches *store* (may be ``None``) and never
    writes PG — it still needs a working ``pg_dsn`` to read the real staging
    state to report against.

    Bucket processing (real runs only) is parallelised across
    ``EXONIT_COMPACT_WORKERS`` threads (default 4) — see the module
    docstring's step 3 and ``_compact_bucket_worker``.
    """
    import psycopg

    if dry_run:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            return _compact_dry_run(conn)

    run_id = run_id or uuid.uuid4().hex[:12]
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        acquired = conn.execute(
            "SELECT pg_try_advisory_lock(%s, 0)", (_COMPACTION_PG_LOCK_NS,)
        ).fetchone()[0]
        if not acquired:
            return _empty_summary(
                dry_run=False, reason="locked_by_other_process",
                message="compaction advisory lock held by another "
                        "process; skipping this run")
        try:
            return _compact_locked(conn, store, run_id, pg_dsn, batch_rows=batch_rows)
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s, 0)", (_COMPACTION_PG_LOCK_NS,))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

_DEFAULT_BATCH_ROWS = 2_000_000


def _default_batch_rows() -> int:
    """``--batch-rows``'s CLI default — read fresh at ``parse_args()`` time
    (not a module-level constant) so tests can ``monkeypatch.setenv`` it.
    Same fallback pattern as ``_compact_workers``: an unparsable override
    falls back to the default rather than raising or silently disabling
    batching."""
    raw = os.environ.get("EXONIT_COMPACT_BATCH_ROWS", "")
    try:
        return int(raw) if raw.strip() else _DEFAULT_BATCH_ROWS
    except ValueError:
        log.warning("EXONIT_COMPACT_BATCH_ROWS=%r is not an int; using default %d",
                    raw, _DEFAULT_BATCH_ROWS)
        return _DEFAULT_BATCH_ROWS


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pg", dest="pg_dsn", required=True, help="PostgreSQL DSN")
    p.add_argument("--s3-bucket", default="exonit-completeness",
                   help="MinIO/S3 bucket for the Parquet buckets (default: exonit-completeness)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what compaction would do without writing MinIO or PG "
                        "(still connects to --pg to read the real staging state)")
    p.add_argument("--batch-rows", type=int, default=_default_batch_rows(),
                   help="Max completeness_staging rows read/merged/deleted per batch "
                        "(env EXONIT_COMPACT_BATCH_ROWS, default 2000000); "
                        "0 = single unbounded pass, the pre-batch-mode behavior")
    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    # read_timeout=300 + botocore's own per-part retries OFF (max_attempts=0):
    # same rationale as discovery_download.py's dedicated instance (see
    # ObjectStore.__init__'s docstring) — this module does its OWN whole-
    # sequence retry+backoff (with_s3_retries, 2026-07-21), so a stalled
    # request should surface to THAT loop instead of racing a second,
    # uncoordinated retry mechanism stacked underneath it.
    store = (None if args.dry_run
             else ObjectStore(bucket=args.s3_bucket, read_timeout=300,
                               retries_max_attempts=0))

    try:
        result = compact(args.pg_dsn, store, dry_run=args.dry_run,
                          batch_rows=args.batch_rows)
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}, indent=2, default=str))
        return 1

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
