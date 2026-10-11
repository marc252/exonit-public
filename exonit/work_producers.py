"""Eligibility predicates + enqueue hooks for the work streams (Fase 1).

The predicate below MUST mirror the legacy FPP-backfill scan's WHERE
clauses byte-for-byte in semantics — copied verbatim from the two arms of
``GET /api/worker/fpp_task`` in ``exonit/routes/workers.py:805-824``:

  Arm A (fpp missing):
    c.snr >= 10.0 AND (c.cnn_score >= 0.7 OR c.exominer_ffi_score >= 0.5)
    AND c.triceratops_fpp IS NULL
    AND s.status = 'completed'

  Arm B (nfpp pending):
    c.snr >= 10.0 AND (c.cnn_score >= 0.7 OR c.exominer_ffi_score >= 0.5)
    AND c.triceratops_fpp >= 0 AND c.triceratops_fpp < 0.5
    AND c.triceratops_nfpp IS NULL
    AND s.status = 'completed'

OR-gate (2026-07-11, docs/reviews/injection_e2e_results_2026-07-11.md +
docs/reviews/ffi_cnn_investigation_2026-07-11.md §5): the CNN-only
eligibility gate kills ~67% of genuine injected transits in FFI-weak
hosts (CNN-FFI AUC ~0.49, effectively a coin flip at a high threshold),
while ExoMiner-FFI v6 (AUC 0,766 on the real hold-out) rescues a
meaningful slice of those. ``(cnn_score>=0.7 OR exominer_ffi_score>=0.5)``
is STRICTLY more permissive than the old ``cnn_score>=0.7`` alone — every
row that used to pass still passes (monotonicity; see
``verdict_eligible`` below and its tests) — so this is a pure recall
gain, no regression risk for anything already reaching the verdict/FPP
chain.

Both arms are mutually exclusive (fpp IS NULL vs fpp >= 0) and both
exclude the -1/-1 "uncomputable" sentinel written by
``db.mark_fpp_uncomputable`` / ``db.mark_fpp_uncomputable_for_tic``: -1 is
neither NULL (arm A) nor >= 0 (arm B — well, -1 < 0 so it fails the
">= 0" test), so a sentinelled candidate never re-enters the queue.

The legacy scan stays the fallback path (it still runs when this
producer/consumer path is disabled or a message is lost); the rebuild
script (``scripts/rebuild_work_streams.py``) is the reconciliation
mechanism for writers that bypass these hooks entirely — e.g.
``scripts/rescore_cnn_ffi.py``, which updates ``candidates.cnn_score`` via
raw SQL and never calls ``maybe_enqueue_fpp``.
"""
from __future__ import annotations

import os


def streams_enabled() -> bool:
    return os.environ.get("EXONIT_WORK_STREAMS") == "1"


def _get_stream(name):            # separate hook point so tests can monkeypatch
    from exonit.work_streams import get_stream
    return get_stream(name)


def verdict_eligible(snr, cnn_score, exominer_ffi_score=None) -> bool:
    """OR-gate eligibility shared by the verdict pre-gate and the FPP
    backfill queue: ``snr >= 10 AND (cnn_score >= 0.7 OR
    exominer_ffi_score >= 0.5)``.

    Single source of truth for the Python-side half of the OR-gate (the
    SQL-side copies in exonit/db.py, exonit/server.py and
    exonit/routes/{workers,admin}.py use the literal text
    ``(cnn_score >= 0.7 OR exominer_ffi_score >= 0.5)`` instead — SQLite's
    partial-index matching needs the literal, same reason the pre-existing
    ``cnn_score >= 0.7`` copies were never centralised either; see
    ``tests/test_strong_gate_consistency.py``).

    NULL handling is explicit (not left to SQL-style None-comparisons):
    a None/missing ``snr`` always fails; a None ``cnn_score`` or None
    ``exominer_ffi_score`` simply does not contribute to its side of the
    OR — it never raises and never counts as "passing". This mirrors SQL
    ``NULL >= x`` evaluating to NULL/false in a WHERE clause, which is
    already what the old ``cnn is not None and cnn >= 0.7`` checks did.

    Strictly more permissive than the pre-OR-gate predicate
    (``snr >= 10 AND cnn_score >= 0.7``): every row that passed before
    still passes now (monotonicity — see
    ``tests/test_work_producers.py::test_verdict_eligible_monotonic_*``).
    """
    if snr is None or snr < 10.0:
        return False
    cnn_ok = cnn_score is not None and cnn_score >= 0.7
    em_ffi_ok = exominer_ffi_score is not None and exominer_ffi_score >= 0.5
    return cnn_ok or em_ffi_ok


def fpp_eligible(row: dict) -> bool:
    """Mirror of the two legacy FPP-scan arms (workers.py:805-824).

    ``row`` keys: candidate_id, tic_id, snr, cnn_score, exominer_ffi_score,
    triceratops_fpp, nfpp (stands for the ``triceratops_nfpp`` column),
    star_status.
    """
    if row.get("star_status") != "completed":
        return False
    if not verdict_eligible(row.get("snr"), row.get("cnn_score"),
                             row.get("exominer_ffi_score")):
        return False
    fpp = row.get("triceratops_fpp")
    nfpp = row.get("nfpp")
    fpp_missing = fpp is None
    nfpp_pending = fpp is not None and 0 <= fpp < 0.5 and nfpp is None
    return fpp_missing or nfpp_pending


def maybe_enqueue_fpp(row: dict) -> bool:
    """Best-effort enqueue of ``row`` on the "fpp" work stream.

    No-op (returns False) unless ``EXONIT_WORK_STREAMS=1`` and the row is
    currently eligible. Callers MUST wrap this in try/except (see the
    hooks in ``db.py`` / ``routes/workers.py``) — a producer failure must
    never break a result write. Enqueue is a single message per candidate;
    the consumer (Task 3) re-checks eligibility against PG before serving
    it, so a stale/approximate ``row`` here is harmless (worst case: a
    redundant re-check, never a wrong write).
    """
    if not streams_enabled() or not fpp_eligible(row):
        return False
    return _get_stream("fpp").enqueue(
        str(row["candidate_id"]),
        {"candidate_id": row["candidate_id"], "tic_id": row["tic_id"]})


# ---------------------------------------------------------------------------
# ExoMiner (Fase 1, Task 4) — mirror of the legacy worker_exominer_task scan's
# WHERE clause, copied verbatim from exonit/routes/workers.py (pre-cutover
# main query):
#
#   c.exominer_score IS NULL AND c.cnn_score >= 0.7
#   AND s.status = 'completed' AND s.has_spoc = 1
#
# has_spoc is a tri-state column (1 = has 2-min SPOC, 0 = confirmed FFI-only,
# NULL = not yet gated by the SPOC-gate background check) — only has_spoc=1
# rows are ExoMiner-queue-eligible; 0 and NULL both fail this predicate, same
# as the legacy scan.  exominer_score IS NULL also excludes the -1 sentinel
# written by db.mark_ffi_only_candidates / mark_unconfirmed_ffi_candidates
# (permanent "no DV data" marker) and any real score already computed.
# ---------------------------------------------------------------------------


def exominer_eligible(row: dict) -> bool:
    """Mirror of the legacy ExoMiner-backfill scan predicate (workers.py's
    pre-cutover ``worker_exominer_task`` main SELECT).

    ``row`` keys: candidate_id, tic_id, cnn_score, exominer_score, has_spoc,
    star_status.
    """
    if row.get("star_status") != "completed":
        return False
    if not row.get("has_spoc"):        # NULL or 0 -> not queue-eligible
        return False
    cnn = row.get("cnn_score")
    if cnn is None or cnn < 0.7:
        return False
    return row.get("exominer_score") is None


def maybe_enqueue_exominer(row: dict) -> bool:
    """Best-effort enqueue of ``row`` on the "exominer" work stream.

    Same contract as ``maybe_enqueue_fpp``: no-op unless
    ``EXONIT_WORK_STREAMS=1`` and the row is currently eligible; callers
    MUST wrap this in try/except (see the hooks in ``db.py``'s
    ``set_star_has_spoc`` and ``routes/workers.py``'s ``worker_result``) —
    a producer failure must never break a result/gate write. The consumer
    (Task 4's ``_exominer_task_from_stream``) re-checks eligibility against
    PG before serving, so a stale/approximate ``row`` here is harmless.
    """
    if not streams_enabled() or not exominer_eligible(row):
        return False
    return _get_stream("exominer").enqueue(
        str(row["candidate_id"]),
        {"candidate_id": row["candidate_id"], "tic_id": row["tic_id"]})


# ---------------------------------------------------------------------------
# Primary (BLS) task queue (Fase 1, Task 5) — mirror of the legacy
# /api/worker/task scan's only pre-serve condition (db.get_and_assign_task,
# "WHERE status='pending' ORDER BY priority ASC, created_at ASC"). Unlike
# the FPP/ExoMiner streams there is no joined-table filter here — a task's
# own status column is its only eligibility fact.
# ---------------------------------------------------------------------------


def primary_eligible(row: dict) -> bool:
    """Mirror of the legacy primary-task scan's WHERE clause (pending only).

    ``row`` keys: task_id, tic_id, status.
    """
    return row.get("status") == "pending"


def maybe_enqueue_primary(row: dict) -> bool:
    """Best-effort enqueue of ``row`` on the "primary" work stream.

    Same contract as ``maybe_enqueue_fpp``/``maybe_enqueue_exominer``:
    no-op unless ``EXONIT_WORK_STREAMS=1`` and the row is currently
    eligible; callers MUST wrap this in try/except (see the call sites in
    routes/workers.py's worker_enqueue / worker_enqueue_bulk) — a producer
    failure must never break the enqueue HTTP response. The consumer
    (_primary_task_from_stream, via db.get_task_row_and_assign) re-checks
    status against PG before serving, so a stale/approximate ``row`` here
    is harmless.
    """
    if not streams_enabled() or not primary_eligible(row):
        return False
    return _get_stream("primary").enqueue(
        str(row["task_id"]),
        {"task_id": row["task_id"], "tic_id": row["tic_id"]})


def maybe_enqueue_primary_many(rows: list) -> int:
    """Batch variant of ``maybe_enqueue_primary`` (G3, review package
    2026-07-05).

    Filters *rows* to the eligible ones (same per-row check
    ``maybe_enqueue_primary`` uses) and enqueues them in ONE
    ``enqueue_many()`` call instead of one ``enqueue()`` round trip per
    row in a Python for-loop — the fix for ``/api/worker/enqueue_bulk``'s
    producer hook, which at this project's 50k-100k-task discovery-run
    enqueue scale was making N synchronous SADD+XADD round trips inline
    in an async endpoint (tens of seconds of blocked event loop).

    No-op (returns 0) unless ``EXONIT_WORK_STREAMS=1``. Callers MUST wrap
    this in try/except, same as ``maybe_enqueue_primary`` — a producer
    failure must never break the bulk-enqueue HTTP response. Returns the
    number of rows actually enqueued (ineligible/dedup-skipped rows don't
    count), mirroring ``enqueue_many``'s own return contract.
    """
    if not streams_enabled():
        return 0
    items = [(str(r["task_id"]), {"task_id": r["task_id"], "tic_id": r["tic_id"]})
             for r in rows if primary_eligible(r)]
    if not items:
        return 0
    return _get_stream("primary").enqueue_many(items)


# ---------------------------------------------------------------------------
# Reconciliation (2026-07-14)
# ---------------------------------------------------------------------------
# The producer hooks above enqueue at WRITE time. That is not enough, and the
# gap is not theoretical — it stranded two runs on 2026-07-14:
#
#   * A row can become eligible through a path that never calls a hook (raw-SQL
#     writers like scripts/rescore_cnn_ffi.py; a late has_spoc flip). It is then
#     eligible in PG and present in NO queue: the health/work counter says
#     "pending", no worker can ever be served it, and a phase that waits for
#     that counter to reach 0 waits forever.
#   * An item whose FITS metadata is not cached is held back at claim time (no
#     serve, no ack) and, after max_deliveries, moves to the DLQ. It leaves the
#     queue but its PG row is untouched — so it also keeps counting as pending,
#     forever. 56 FPP candidates had been sitting in the DLQ since 2026-07-12
#     doing exactly this.
#
# scripts/rebuild_work_streams.py already knew this ("re-enqueueable by a
# rebuild once the metadata is indexed") — but nothing ever ran it. The
# derivation SQL below is that script's, moved here so the script and the
# automatic reconciliation cannot drift apart, and reconcile_stream() is the
# non-destructive form the orchestrator now runs at the start of every backfill
# phase.


def _derive_fpp(db):
    """Re-derive the "fpp" stream: both arms of workers.py:805-824, unioned,
    ORDER BY snr DESC, no LIMIT.

    The two arms are mutually exclusive (triceratops_fpp IS NULL vs
    triceratops_fpp >= 0), so a plain list concatenation is exact — no
    dedup needed, unlike the live handler's top-200-per-arm merge.

    OR-gate (2026-07-11, docs/reviews/injection_e2e_results_2026-07-11.md +
    docs/reviews/ffi_cnn_investigation_2026-07-11.md §5): eligibility is
    ``snr>=10 AND (cnn_score>=0.7 OR exominer_ffi_score>=0.5)``, matching
    workers.py's FPP arms / work_producers.fpp_eligible exactly — this is
    the reconciliation path for writers that bypass the producer hooks
    (see module docstring), so an em_ffi-rescued candidate whose enqueue
    was missed must still be recoverable by a rebuild.
    """
    try:
        arm_a = db._query(
            """SELECT c.id, c.tic_id, c.snr
               FROM candidates c INDEXED BY idx_cand_fpp_todo
               JOIN stars s ON c.tic_id = s.tic_id
               WHERE c.snr >= 10.0
                 AND (c.cnn_score >= 0.7 OR c.exominer_ffi_score >= 0.5)
                 AND c.triceratops_fpp IS NULL
                 AND s.status = 'completed'
               ORDER BY c.snr DESC""")
        arm_b = db._query(
            """SELECT c.id, c.tic_id, c.snr
               FROM candidates c INDEXED BY idx_cand_nfpp_todo
               JOIN stars s ON c.tic_id = s.tic_id
               WHERE c.snr >= 10.0
                 AND (c.cnn_score >= 0.7 OR c.exominer_ffi_score >= 0.5)
                 AND c.triceratops_fpp >= 0 AND c.triceratops_fpp < 0.5
                 AND c.triceratops_nfpp IS NULL
                 AND s.status = 'completed'
               ORDER BY c.snr DESC""")
        rows = arm_a + arm_b
    except Exception:
        # Index missing/renamed on this DB — fall back to the broad OR form
        # (identical result set, only slower). Mirrors workers.py's own
        # fallback for the same reason.
        rows = db._query(
            """SELECT c.id, c.tic_id, c.snr
               FROM candidates c
               JOIN stars s ON c.tic_id = s.tic_id
               WHERE c.snr >= 10.0
                 AND (c.cnn_score >= 0.7 OR c.exominer_ffi_score >= 0.5)
                 AND (
                   c.triceratops_fpp IS NULL
                   OR (
                     c.triceratops_fpp >= 0
                     AND c.triceratops_fpp < 0.5
                     AND c.triceratops_nfpp IS NULL
                   )
                 )
                 AND s.status = 'completed'""")
    rows.sort(key=lambda r: r["snr"], reverse=True)
    return [(str(r["id"]), {"candidate_id": r["id"], "tic_id": r["tic_id"]})
            for r in rows]


def _derive_exominer(db):
    """Re-derive the "exominer" stream: same WHERE as workers.py's legacy
    scan / stream handler (s.status='completed' AND s.has_spoc=1 AND
    c.cnn_score>=0.7 AND c.exominer_score IS NULL), ORDER BY snr DESC, no
    LIMIT.

    No metadata-cache filtering and no -1 pre-marking here (see the module
    docstring): a has_spoc=1 row with a missing metadata entry is still
    genuinely eligible — the consumer holds it back at claim time and the
    stream's DLQ (max_deliveries) is the visible terminal state for gaps
    that never resolve. Marking -1 here would silently and permanently
    drop a computable ExoMiner score.
    """
    rows = db._query(
        """SELECT c.id, c.tic_id, c.snr
           FROM candidates c JOIN stars s ON c.tic_id = s.tic_id
           WHERE s.status = 'completed' AND s.has_spoc = 1
             AND c.cnn_score >= 0.7 AND c.exominer_score IS NULL
           ORDER BY c.snr DESC""")
    return [(str(r["id"]), {"candidate_id": r["id"], "tic_id": r["tic_id"]})
            for r in rows]


def _derive_primary(db):
    """Re-derive the "primary" stream: SAME WHERE + ORDER as the legacy
    get_and_assign_task query (status='pending', ORDER BY priority ASC,
    created_at ASC), no LIMIT.

    Safe to run any time the primary fleet is idle for this stage (mirrors
    the fpp/exominer rebuild's own note): ws.clear() drops any in-flight
    'assigned' tasks' stream tracking too (they are not re-derived here,
    since they are not 'pending') — a later /api/worker/result for one of
    those still completes/fails it correctly in PG; it just also calls
    ack() against a message that no longer exists, a harmless no-op.
    """
    rows = db._query(
        """SELECT task_id, tic_id FROM tasks WHERE status = 'pending'
           ORDER BY priority ASC, created_at ASC""")
    return [(r["task_id"], {"task_id": r["task_id"], "tic_id": r["tic_id"]})
            for r in rows]


def derive_eligible(db, stage: str):
    if stage == "fpp":
        return _derive_fpp(db)
    if stage == "exominer":
        return _derive_exominer(db)
    if stage == "primary":
        return _derive_primary(db)
    raise SystemExit(f"--stage {stage} not recognised")


def reconcile_stream(db, stage: str) -> dict:
    """Make the *stage* stream agree with the DB, WITHOUT clearing it.

    Enqueues every PG-eligible item that is currently in neither the queue nor
    the pending list, and drops any DLQ entry that this brings back — a
    dead-lettered item that is eligible again (its metadata finally got indexed,
    say) must not linger in the DLQ as a duplicate ghost of a live queue entry.

    Non-destructive on purpose: unlike rebuild_work_streams.py's clear+rebuild,
    this never drops an in-flight claim, so it is safe to run while the fleet is
    working — which is exactly when the orchestrator calls it (at the start of
    each backfill phase, before it begins waiting on a counter).

    Returns a summary dict; callers log it. Never raises for an empty stage.
    """
    if not streams_enabled():
        return {"stage": stage, "streams_enabled": False}

    ws = _get_stream(stage)
    eligible = derive_eligible(db, stage)
    present = ws.queued_keys()
    missing = [(k, p) for (k, p) in eligible if k not in present]

    enqueued = ws.enqueue_many(missing) if missing else 0

    # Purge the DLQ of everything that is no longer an unresolved failure:
    #
    #   * items we just re-enqueued — the queue entry is the live one now, the
    #     DLQ copy is a ghost;
    #   * items that are no longer eligible at all — they were scored in the
    #     end, or their row was deleted. Keeping them makes the DLQ a growing
    #     pile of resolved noise, and the `dlq` figure health/work reports stops
    #     meaning anything an operator can act on. (Live example: 43 ExoMiner
    #     entries for candidates that had SINCE been scored, and 2 for
    #     candidates that no longer existed.)
    #
    # What survives is exactly "exhausted its deliveries and STILL owed" —
    # which, right after a reconcile, is nothing; it refills only with items
    # that keep failing. That is the signal worth having.
    eligible_keys = {k for (k, _p) in eligible}
    requeued = {k for (k, _p) in missing}
    dlq_cleared = 0
    for entry in ws.peek_dlq():
        k = entry.get("k")
        if (k in requeued or k not in eligible_keys) and \
                ws.remove_dlq(entry.get("msg_id")):
            dlq_cleared += 1

    return {
        "stage": stage,
        "streams_enabled": True,
        "eligible": len(eligible),
        "already_queued": len(eligible) - len(missing),
        "enqueued": enqueued,
        "dlq_cleared": dlq_cleared,
    }
