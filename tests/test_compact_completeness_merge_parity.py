"""Parity test for the 2026-09-24 pyarrow vectorization of
``scripts/compact_completeness.py``'s ``_merge_bucket`` (Fase 4a Task 3
compaction job): the per-bucket last-write-wins merge moved from building a
Python ``{(tic_id, radius_earth, period_days): detection_prob}`` dict out of
``table_rows(existing_table)`` (materializing every one of a bucket's ~5M
rows into Python tuples) to a pyarrow ``"left anti"`` hash join + concat,
operating on ``pa.Table`` objects end to end.

This file keeps a byte-for-byte copy of the OLD Python-dict ``_merge_bucket``
as the reference oracle (``_old_merge_bucket`` below — the exact body
``_merge_bucket`` had before this vectorization, see git history) and
asserts the NEW ``cc._merge_bucket`` produces an IDENTICAL result across
every edge case called out by the vectorization spec: PK collisions between
existing and incoming, duplicate tic_ids, empty existing, empty incoming,
many randomized buckets. "Identical" is checked three ways: ``pa.Table.
equals()`` (schema + values + row order), re-aggregated
``aggregate_bucket_rows()`` output, and the raw bytes of the Parquet file
each table serializes to with the SAME writer options
``_upload_bucket_atomic`` uses.

No PG, no MinIO/object store — this is a pure-function test directly against
``_merge_bucket``, not the full ``compact()`` pipeline (see
``tests/test_compact_completeness.py`` for the PG+FakeStore integration
tests, which already exercise the new code path end to end).
"""

import io
import random
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import compact_completeness as cc  # noqa: E402
from exonit.completeness_store import (  # noqa: E402
    PK_COLUMNS,
    aggregate_bucket_rows,
    assert_unique_pk,
    rows_to_table,
    table_rows,
)

_RADIUS_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]  # worker/injection.py's grid
_PERIOD_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0]


def _old_merge_bucket(existing_rows: list, incoming_by_pk: dict):
    """Reference oracle — the exact ``_merge_bucket`` body before the
    2026-09-24 pyarrow vectorization (Python-dict last-write-wins merge,
    staging always wins on PK conflict). Do NOT "improve" this function; it
    exists to freeze the pre-vectorization behavior as ground truth."""
    merged_by_pk = {(t, r, p): prob for (t, r, p, prob) in existing_rows}
    merged_by_pk.update(incoming_by_pk)
    merged_rows = [(t, r, p, prob) for (t, r, p), prob in merged_by_pk.items()]
    merged_table = rows_to_table(merged_rows)
    merged_table = merged_table.sort_by([(c, "ascending") for c in PK_COLUMNS])
    assert_unique_pk(merged_table)
    return merged_table


def _write_bytes(table) -> bytes:
    """Same writer options as ``_upload_bucket_atomic`` — for a byte-for-byte
    Parquet comparison, not just an Arrow-Table-level one."""
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd", use_dictionary=["tic_id"])
    return buf.getvalue()


def _run_parity(existing_rows: list, incoming_by_pk: dict):
    """Runs both the old and new merge on the same input and asserts every
    externally-observable result matches: the Arrow table itself, its
    Parquet serialization, and the aggregate computed from it (bucket number
    is arbitrary/fixed here — aggregate_bucket_rows doesn't touch PK
    matching, only detection_prob group-by, so parity here just confirms the
    merged table feeds it identically either way)."""
    existing_table = rows_to_table(existing_rows)

    old_result = _old_merge_bucket(existing_rows, incoming_by_pk)
    new_result = cc._merge_bucket(existing_table, incoming_by_pk)

    assert new_result.equals(old_result), (
        f"old={old_result.to_pydict()!r}\nnew={new_result.to_pydict()!r}")
    assert _write_bytes(new_result) == _write_bytes(old_result)
    assert aggregate_bucket_rows(new_result, bucket=7) == \
        aggregate_bucket_rows(old_result, bucket=7)
    return old_result


# --------------------------------------------------------------------------- #
# Edge cases named in the vectorization spec
# --------------------------------------------------------------------------- #

def test_merge_empty_existing_and_empty_incoming():
    _run_parity([], {})


def test_merge_empty_existing_nonempty_incoming():
    incoming = {("100", 1.0, 5.0): 0.5, ("100", 2.0, 5.0): 0.6, ("200", 1.0, 5.0): 0.1}
    _run_parity([], incoming)


def test_merge_nonempty_existing_empty_incoming():
    existing = [("100", 1.0, 5.0, 0.5), ("200", 2.0, 5.0, 0.6), ("100", 3.0, 5.0, 0.2)]
    _run_parity(existing, {})


def test_merge_pk_collision_between_existing_and_incoming_incoming_wins():
    existing = [("100", 1.0, 5.0, 0.1), ("100", 2.0, 5.0, 0.2), ("200", 1.0, 5.0, 0.3)]
    incoming = {("100", 1.0, 5.0): 0.99}  # same PK as an existing row
    result = _run_parity(existing, incoming)
    rows = set(table_rows(result))
    assert ("100", 1.0, 5.0, 0.99) in rows  # incoming won
    assert ("100", 2.0, 5.0, 0.2) in rows   # untouched existing row survives
    assert ("200", 1.0, 5.0, 0.3) in rows


def test_merge_duplicate_tic_ids_across_many_cells():
    existing = [("100", r, 5.0, 0.1) for r in _RADIUS_GRID[:5]]
    existing += [("200", r, 5.0, 0.2) for r in _RADIUS_GRID[:3]]
    existing += [("100", 5.0, p, 0.3) for p in _PERIOD_GRID[:3]]  # more 100 cells
    incoming = {("100", 6.0, 5.0): 0.9, ("300", 1.0, 5.0): 0.3, ("200", 1.5, 5.0): 0.4}
    _run_parity(existing, incoming)


def test_merge_signed_zero_pk_collision_raises_instead_of_silently_diverging():
    """Real discovery: Python's ``==``/``hash`` treat ``-0.0`` and ``0.0`` as
    the SAME dict key, so the OLD merge silently collapsed such a PK
    collision to one row (whichever value ``dict.update`` left behind) --
    but pyarrow's hash-join key hashing does NOT match ``-0.0`` to ``0.0``
    (confirmed: ``pc.equal(-0.0, 0.0)`` is True, yet a hash join on the two
    values keeps both), so the NEW merge would let both rows through to
    ``assert_unique_pk``, which WOULD then flag them as a duplicate PK
    (pandas' ``.duplicated()`` treats -0.0/0.0 as equal, same as Python).
    ``_assert_pk_floats_safe_for_join`` catches this before ever reaching
    the join, raising a clear, on-topic error instead of leaving it to
    surface as a confusing "duplicate PK" ValueError from deep inside
    ``assert_unique_pk``. Zero radius/period is physically meaningless and
    never produced by any real writer (see that function's docstring), so
    this is a defensive guard, not a case the real pipeline can hit."""
    existing = [("100", -0.0, 5.0, 0.1)]
    incoming_by_pk = {("100", 0.0, 5.0): 0.77}
    existing_table = rows_to_table(existing)

    with pytest.raises(ValueError, match="NaN or zero"):
        cc._merge_bucket(existing_table, incoming_by_pk)

    # The OLD merge, for contrast, silently resolved this (kept -0.0's PK
    # tuple from `existing`, since dict.update() overwrites only the VALUE
    # of an already-present key -- never the key object itself -- with
    # incoming's detection_prob).
    old_result = _old_merge_bucket(existing, incoming_by_pk)
    assert old_result.num_rows == 1
    assert table_rows(old_result)[0][3] == 0.77


def test_merge_incoming_only_no_existing_bucket_file_yet():
    """The ``store.exists(key)`` False path in ``_compact_bucket`` passes an
    EMPTY typed table (``rows_to_table([])``), not an empty list -- exercise
    that exact shape here instead of a bare ``[]``."""
    empty_existing_table = rows_to_table([])
    incoming = {("500", 1.0, 5.0): 0.5}
    new_result = cc._merge_bucket(empty_existing_table, incoming)
    old_result = _old_merge_bucket([], incoming)
    assert new_result.equals(old_result)


@pytest.mark.parametrize("seed", range(20))
def test_merge_randomized_many_buckets(seed):
    """20 independent randomized (existing, incoming) pairs per run, varying
    row counts, tic overlap, and PK collision rates -- the "many buckets"
    requirement from the spec, parametrized rather than looped so a failure
    reports its exact seed."""
    rng = random.Random(seed)
    n_tics = rng.randint(1, 12)
    tics = [str(rng.randint(100000, 999999)) for _ in range(n_tics)]

    existing = []
    seen_pk = set()
    for _ in range(rng.randint(0, 80)):
        pk = (rng.choice(tics), rng.choice(_RADIUS_GRID), rng.choice(_PERIOD_GRID))
        if pk in seen_pk:
            continue  # existing bucket file is itself already PK-unique
        seen_pk.add(pk)
        existing.append((*pk, rng.random()))

    # incoming_by_pk simulates _dedup_staging's already-deduped output
    # (last write wins is resolved before _merge_bucket ever sees it) --
    # build it by inserting into a dict in "written_at, id" order, same as
    # the real caller, so later duplicates legitimately overwrite earlier
    # ones within incoming itself too.
    incoming_by_pk = {}
    extra_tics = tics + [str(rng.randint(100000, 999999)) for _ in range(3)]
    for _ in range(rng.randint(0, 80)):
        pk = (rng.choice(extra_tics), rng.choice(_RADIUS_GRID), rng.choice(_PERIOD_GRID))
        incoming_by_pk[pk] = rng.random()

    _run_parity(existing, incoming_by_pk)


# --------------------------------------------------------------------------- #
# Micro-benchmark (informational only -- NOT an assertion; wall-clock timing
# is environment-dependent, so this only prints numbers under `pytest -s`,
# it never fails the suite).
# --------------------------------------------------------------------------- #

def test_benchmark_old_vs_new_merge_one_million_row_bucket(capsys):
    rng = random.Random(0)
    n_rows = 1_000_000
    tics = [f"{100000 + i}" for i in range(n_rows // len(_RADIUS_GRID) // len(_PERIOD_GRID) + 1)]

    existing_rows = []
    seen = set()
    for t in tics:
        for r in _RADIUS_GRID:
            for p in _PERIOD_GRID:
                if len(existing_rows) >= n_rows:
                    break
                existing_rows.append((t, r, p, rng.random()))
        if len(existing_rows) >= n_rows:
            break

    # ~1% of tics get a fresh staging batch (a realistic compaction batch:
    # a small slice of tics touched relative to the whole bucket).
    incoming_tics = rng.sample(tics, max(1, len(tics) // 100))
    incoming_by_pk = {
        (t, r, p): rng.random()
        for t in incoming_tics for r in _RADIUS_GRID for p in _PERIOD_GRID
    }

    existing_table = rows_to_table(existing_rows)

    t0 = time.perf_counter()
    old_result = _old_merge_bucket(existing_rows, incoming_by_pk)
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    new_result = cc._merge_bucket(existing_table, incoming_by_pk)
    t_new = time.perf_counter() - t0

    assert new_result.equals(old_result)

    with capsys.disabled():
        print(
            f"\n[bench] _merge_bucket, {len(existing_rows):,} existing rows, "
            f"{len(incoming_by_pk):,} incoming rows: "
            f"OLD (Python dict) = {t_old:.3f}s, NEW (pyarrow join) = {t_new:.3f}s, "
            f"speedup = {t_old / t_new:.1f}x")
