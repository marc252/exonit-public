"""Fase 4a — shared helpers for the completeness Parquet+PG migration.

``completeness`` (183.95M rows, SQLite-only) is migrating off the hot
database: raw rows -> bucketed Parquet on MinIO, aggregates + exact counts ->
PostgreSQL. This module holds the small, backend-agnostic pieces every layer
of that migration shares (write-path staging in ``exonit/db.py`` — Task 1;
the historical one-shot migrator — Task 2; the compaction job — Task 3), so
the bucketing rule and the cutover flag live in exactly one place instead of
being redefined per-script.

See docs/superpowers/plans/2026-07-05-phase4-completeness-parquet.md.

Flag: ``EXONIT_COMPLETENESS_PG=1`` turns on the PG write-path (staging table
+ PG-backed aggregate reads) in ``ExoNitDB``. Off (default/unset) keeps the
historical SQLite completeness sidecar untouched — this module and every
caller of ``completeness_pg_enabled()`` must treat any value other than the
literal string ``"1"`` as OFF, so a typo in the env (e.g. ``"true"``) fails
safe to the well-tested SQLite path rather than silently cutting over.
"""

import os
import zlib

# Fixed bucket count for the Parquet layout (buckets/bucket=BBB/data.parquet)
# and the PG completeness_tics.bucket / completeness_agg_bucket.bucket
# columns (SMALLINT, 0-255). This is a migration constant, not a tunable —
# changing it invalidates every already-written Parquet bucket and PG
# aggregate row, so it is deliberately not env-configurable.
COMPLETENESS_BUCKETS = 256

# Primary key of one completeness cell, everywhere in the Fase 4a pipeline
# (raw SQLite table, staging, bucketed Parquet). Shared so Task 2's historical
# migrator and Task 3's compaction job sort/dedupe/assert on the exact same
# column tuple.
PK_COLUMNS = ("tic_id", "radius_earth", "period_days")


def completeness_pg_enabled() -> bool:
    """True when the completeness-on-PG write path (Fase 4a) is active.

    Only the exact string ``"1"`` enables it — mirrors the existing
    ``EXONIT_WORK_STREAMS`` flag convention (exonit/work_producers.py).
    """
    return os.environ.get("EXONIT_COMPLETENESS_PG") == "1"


def bucket_of(tic_id: str) -> int:
    """Stable bucket assignment for a TIC id, in ``[0, COMPLETENESS_BUCKETS)``.

    Uses ``zlib.crc32`` rather than Python's builtin ``hash()``: str hashing
    is randomized per-process (``PYTHONHASHSEED``) unless disabled, so the
    same TIC could land in different buckets across worker/compaction/
    migration processes. crc32 is deterministic across processes, Python
    versions and platforms — required here because the bucket a TIC's rows
    are compacted into must be the same one every reader/writer computes.
    """
    return zlib.crc32(tic_id.encode("utf-8")) % COMPLETENESS_BUCKETS


# --------------------------------------------------------------------------- #
# Shared Parquet helpers (Task 2's historical migrator + Task 3's compaction
# job both bucket/dedupe/aggregate the exact same 4-column shape). Imports of
# pyarrow are deferred to call time: this module is imported unconditionally
# by ``exonit/db.py`` (including the live server, flag OFF), and pyarrow is
# only ever needed by the two offline Parquet scripts.
# --------------------------------------------------------------------------- #

def rows_to_table(rows):
    """``[(tic_id, radius_earth, period_days, detection_prob), ...]`` -> a
    ``pa.Table`` with the fixed 4-column completeness schema.

    Handles the empty-input case explicitly (``zip(*[])`` raises) so callers
    can always get a correctly-typed table back — e.g. Task 3's compaction
    needs an empty typed table to merge against when a bucket has no existing
    Parquet file yet.
    """
    import pyarrow as pa

    if not rows:
        return pa.table({
            "tic_id": pa.array([], type=pa.string()),
            "radius_earth": pa.array([], type=pa.float64()),
            "period_days": pa.array([], type=pa.float64()),
            "detection_prob": pa.array([], type=pa.float64()),
        })
    tic_ids, radii, periods, probs = zip(*rows)
    return pa.table({
        "tic_id": pa.array(tic_ids, type=pa.string()),
        "radius_earth": pa.array(radii, type=pa.float64()),
        "period_days": pa.array(periods, type=pa.float64()),
        "detection_prob": pa.array(probs, type=pa.float64()),
    })


def table_rows(table) -> list:
    """``pa.Table`` (the 4-column completeness schema) -> a list of native
    Python ``(tic_id, radius_earth, period_days, detection_prob)`` tuples.

    Uses ``Array.to_pylist()`` (not ``to_pandas()``) deliberately: it yields
    plain ``str``/``float`` objects, never numpy/pandas boxed scalars, so a
    dict keyed on these tuples hashes/compares exactly like the same values
    read straight out of psycopg (see ``scripts/compact_completeness.py``'s
    last-write-wins merge, which builds such a dict from both sources).
    """
    cols = [table.column(c).to_pylist() for c in PK_COLUMNS + ("detection_prob",)]
    return list(zip(*cols))


def read_parquet_file(source):
    """Read exactly one physical Parquet file/buffer, with NO Hive-partition
    inference from ancestor path segments.

    ``source`` may be a path (``str``/``Path``) or a file-like object (e.g.
    ``io.BytesIO`` of bytes just downloaded from object storage).
    ``pq.read_table(path)`` — even for one file — goes through pyarrow's
    dataset API, which auto-discovers ``key=value`` path segments as
    partition columns. Every file this migration writes/reads lives under a
    ``bucket=BBB`` directory (the layout the plan calls for), so a plain
    ``read_table`` call silently injects a phantom ``bucket`` column, typed
    inconsistently depending on how many sibling partition directories exist
    at call time. ``pq.ParquetFile(source).read()`` reads only the one
    physical file/buffer, with no dataset/partitioning layer involved.
    """
    import pyarrow.parquet as pq

    return pq.ParquetFile(source).read()


def parquet_row_count(source) -> int:
    """Row count from a Parquet file/buffer's footer metadata only — no data
    read. Used to verify an uploaded object round-tripped intact without
    paying for a full table materialization."""
    import pyarrow.parquet as pq

    return pq.ParquetFile(source).metadata.num_rows


def sha256_of_file(path) -> str:
    """Streaming sha256 of a local file (1 MiB chunks) — the integrity field
    both the historical migrator and the compaction job record per bucket in
    ``buckets/_manifest.json``."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_unique_pk(table, pk_columns=PK_COLUMNS) -> None:
    """Raise ``ValueError`` if *pk_columns* is not unique within *table*.

    Defensive re-check of the caller's own bucketing/merge logic (the source
    SQLite table has this as its literal PRIMARY KEY) — exactly the kind of
    silent corruption neither the one-shot migration nor the recurring
    compaction may let through unnoticed.
    """
    if table.num_rows == 0:
        return
    df = table.select(list(pk_columns)).to_pandas()
    dup_mask = df.duplicated(keep=False)
    if dup_mask.any():
        sample = df[dup_mask].head(5).to_dict("records")
        raise ValueError(
            f"duplicate completeness PK rows in bucket parquet: "
            f"{int(dup_mask.sum())} of {len(df)} rows duplicated "
            f"(sample={sample})")


def aggregate_bucket_rows(table, bucket: int) -> list:
    """Group-by (radius_earth, period_days) -> sum_prob, n, computed FROM
    *table* — the caller passes in the just-written-then-reread (or
    just-merged) table, so the aggregate reflects exactly what a future
    reader of the Parquet file would compute."""
    df = table.to_pandas()
    if df.empty:
        return []
    grouped = (
        df.groupby(["radius_earth", "period_days"])["detection_prob"]
        .agg(sum_prob="sum", n="count")
        .reset_index()
    )
    return [
        {
            "bucket": bucket,
            "radius_earth": float(r.radius_earth),
            "period_days": float(r.period_days),
            "sum_prob": float(r.sum_prob),
            "n": int(r.n),
        }
        for r in grouped.itertuples()
    ]
