"""Tests for exonit.work_producers — the FPP eligibility predicate + enqueue
hooks (Fase 1, Task 2).

The predicate mirrors the two legacy FPP-scan arms copied verbatim from
exonit/routes/workers.py:805-824 (GET /api/worker/fpp_task):

  Arm A (fpp missing):  snr >= 10.0 AND (cnn_score >= 0.7 OR
                        exominer_ffi_score >= 0.5)
                        AND triceratops_fpp IS NULL AND star completed.
  Arm B (nfpp pending): snr >= 10.0 AND (cnn_score >= 0.7 OR
                        exominer_ffi_score >= 0.5)
                        AND triceratops_fpp >= 0 AND triceratops_fpp < 0.5
                        AND triceratops_nfpp IS NULL AND star completed.

OR-gate (2026-07-11, docs/reviews/injection_e2e_results_2026-07-11.md +
docs/reviews/ffi_cnn_investigation_2026-07-11.md §5) — was
``cnn_score >= 0.7`` alone; the ``exominer_ffi_score >= 0.5`` arm is a pure
recall gain (see the monotonicity tests below).

Both arms exclude the -1/-1 "uncomputable" sentinel written by
db.mark_fpp_uncomputable[_for_tic]: -1 is neither NULL nor >= 0, so a
sentinelled candidate matches neither arm.
"""
import pytest

from exonit import work_producers


def _row(**kw):
    base = dict(candidate_id=7, tic_id=123, snr=15.0, cnn_score=0.9,
                triceratops_fpp=None, nfpp=None, star_status="completed")
    base.update(kw)
    return base


def test_eligible_when_fpp_missing_and_thresholds_met():
    assert work_producers.fpp_eligible(_row()) is True


def test_not_eligible_below_snr():
    assert work_producers.fpp_eligible(_row(snr=9.9)) is False


def test_not_eligible_below_cnn():
    # No exominer_ffi_score key in _row() -> None -> em_ffi arm does not
    # fire either, so cnn=0.69 alone must still fail (pre-OR-gate behaviour
    # preserved for rows that never had an em_ffi score).
    assert work_producers.fpp_eligible(_row(cnn_score=0.69)) is False


def test_eligible_when_cnn_low_but_em_ffi_rescues():
    # cnn<0.7 alone would fail; exominer_ffi_score>=0.5 rescues it (arm A).
    assert work_producers.fpp_eligible(
        _row(cnn_score=0.4, exominer_ffi_score=0.6)) is True


def test_not_eligible_when_cnn_low_and_em_ffi_below_threshold():
    assert work_producers.fpp_eligible(
        _row(cnn_score=0.4, exominer_ffi_score=0.49)) is False


def test_not_eligible_when_cnn_low_and_em_ffi_none():
    assert work_producers.fpp_eligible(
        _row(cnn_score=0.4, exominer_ffi_score=None)) is False


# ---------------------------------------------------------------------------
# verdict_eligible — the shared OR-gate predicate (Python-side source of
# truth mirrored by the SQL copies in exonit/db.py + exonit/routes/
# {workers,admin}.py; see work_producers.verdict_eligible's own docstring).
# ---------------------------------------------------------------------------

def test_verdict_eligible_cnn_only_still_passes():
    assert work_producers.verdict_eligible(15.0, 0.9, None) is True


def test_verdict_eligible_em_ffi_rescues_low_cnn():
    assert work_producers.verdict_eligible(12.0, 0.4, 0.6) is True


def test_verdict_eligible_neither_passes():
    assert work_producers.verdict_eligible(12.0, 0.4, 0.3) is False


def test_verdict_eligible_below_snr_fails_even_with_both_scores_high():
    assert work_producers.verdict_eligible(9.9, 0.9, 0.9) is False


def test_verdict_eligible_snr_none_fails():
    assert work_producers.verdict_eligible(None, 0.9, 0.9) is False


def test_verdict_eligible_at_thresholds_pass():
    # Boundary: exactly 0.7 / exactly 0.5 / exactly 10.0 must pass (>=).
    assert work_producers.verdict_eligible(10.0, 0.7, None) is True
    assert work_producers.verdict_eligible(10.0, None, 0.5) is True


def test_verdict_eligible_null_cnn_null_em_ffi_never_crashes():
    # Neither score present at all — must return False cleanly, no
    # TypeError from comparing None >= x.
    assert work_producers.verdict_eligible(15.0, None, None) is False


def test_verdict_eligible_null_em_ffi_with_passing_cnn_never_crashes():
    assert work_producers.verdict_eligible(15.0, 0.9, None) is True


def test_verdict_eligible_null_cnn_with_passing_em_ffi_never_crashes():
    assert work_producers.verdict_eligible(15.0, None, 0.8) is True


@pytest.mark.parametrize("snr,cnn,em_ffi", [
    (15.0, 0.9, None),
    (15.0, 0.71, 0.0),
    (10.0, 0.7, None),
    (50.0, 1.0, 1.0),
    (12.0, 0.75, None),
])
def test_verdict_eligible_monotonic_old_gate_implies_new_gate(snr, cnn, em_ffi):
    """Property test: every (snr, cnn) pair that passed the OLD gate
    (snr>=10 AND cnn>=0.7) must still pass the NEW OR-gate — the whole
    point of an OR-gate is that it is strictly more permissive. Regardless
    of exominer_ffi_score (None, 0, or any value), if the old predicate
    held the new one must hold too."""
    old_gate_passes = snr is not None and snr >= 10.0 and cnn is not None and cnn >= 0.7
    assert old_gate_passes, "fixture bug: case must satisfy the OLD gate"
    assert work_producers.verdict_eligible(snr, cnn, em_ffi) is True


@pytest.mark.parametrize("snr,cnn", [
    (15.0, 0.9), (10.0, 0.7), (50.0, 1.0), (12.0, 0.75), (99.0, 0.71),
])
@pytest.mark.parametrize("em_ffi", [None, 0.0, 0.1, 0.49, 0.5, 0.99])
def test_verdict_eligible_monotonic_grid(snr, cnn, em_ffi):
    """Exhaustive-ish grid: for every (snr, cnn) satisfying the old gate,
    verdict_eligible must be True for ANY exominer_ffi_score value
    (including NULL) — the OR-gate can only ADD eligible rows, never
    remove any that were eligible before."""
    assert work_producers.verdict_eligible(snr, cnn, em_ffi) is True


def test_not_eligible_star_not_completed():
    assert work_producers.fpp_eligible(_row(star_status="pending")) is False


def test_not_eligible_when_fpp_already_done():
    # Matches the REAL arm 2 (workers.py:805-824): arm A no longer applies
    # (triceratops_fpp is not None) and arm B requires triceratops_nfpp IS
    # NULL, which is false here (nfpp=0.02) — so neither arm fires.
    assert work_producers.fpp_eligible(
        _row(triceratops_fpp=0.01, nfpp=0.02)) is False


def test_eligible_when_nfpp_still_pending_arm_b():
    # Arm B: fpp already computed and low (< 0.5), nfpp not yet computed.
    assert work_producers.fpp_eligible(
        _row(triceratops_fpp=0.3, nfpp=None)) is True


def test_not_eligible_when_fpp_at_or_above_arm_b_ceiling():
    # Arm B only covers triceratops_fpp < 0.5 (exclusive); at/above that the
    # candidate is already a likely-FP and NFPP is never scheduled for it.
    assert work_producers.fpp_eligible(
        _row(triceratops_fpp=0.5, nfpp=None)) is False


def test_not_eligible_for_uncomputable_sentinel():
    # db.mark_fpp_uncomputable / mark_fpp_uncomputable_for_tic write
    # fpp=-1, nfpp=-1. -1 is neither NULL (arm A) nor >= 0 (arm B), so a
    # sentinelled candidate must never be re-enqueued.
    assert work_producers.fpp_eligible(
        _row(triceratops_fpp=-1, nfpp=-1)) is False


@pytest.mark.parametrize("em_ffi", [None, 0.0, 0.1, 0.5, 0.99])
def test_fpp_eligible_monotonic_old_gate_implies_new_gate(em_ffi):
    """Property test at the fpp_eligible level: a row that already passed
    the OLD gate (snr>=10, cnn>=0.7, arm A/B otherwise satisfied) must keep
    passing regardless of exominer_ffi_score (including NULL) — same
    monotonicity guarantee as verdict_eligible, one level up the stack."""
    assert work_producers.fpp_eligible(
        _row(exominer_ffi_score=em_ffi)) is True
    assert work_producers.fpp_eligible(
        _row(triceratops_fpp=0.3, nfpp=None, exominer_ffi_score=em_ffi)) is True


def test_maybe_enqueue_respects_flag(monkeypatch):
    calls = []

    class FakeStream:
        def enqueue(self, k, p):
            calls.append((k, p))
            return True

    monkeypatch.setattr(work_producers, "_get_stream",
                        lambda name: FakeStream())
    monkeypatch.delenv("EXONIT_WORK_STREAMS", raising=False)
    assert work_producers.maybe_enqueue_fpp(_row()) is False
    assert calls == []
    monkeypatch.setenv("EXONIT_WORK_STREAMS", "1")
    assert work_producers.maybe_enqueue_fpp(_row()) is True
    assert calls == [("7", {"candidate_id": 7, "tic_id": 123})]


def test_maybe_enqueue_skips_ineligible(monkeypatch):
    monkeypatch.setenv("EXONIT_WORK_STREAMS", "1")
    assert work_producers.maybe_enqueue_fpp(_row(snr=1.0)) is False


def test_streams_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("EXONIT_WORK_STREAMS", raising=False)
    assert work_producers.streams_enabled() is False
    monkeypatch.setenv("EXONIT_WORK_STREAMS", "1")
    assert work_producers.streams_enabled() is True
    monkeypatch.setenv("EXONIT_WORK_STREAMS", "0")
    assert work_producers.streams_enabled() is False


# ---------------------------------------------------------------------------
# ExoMiner (Fase 1, Task 4) — mirror of the legacy worker_exominer_task scan's
# WHERE clause: s.status='completed' AND s.has_spoc=1 AND c.cnn_score>=0.7
# AND c.exominer_score IS NULL.
# ---------------------------------------------------------------------------

def _em_row(**kw):
    base = dict(candidate_id=7, tic_id=123, cnn_score=0.9,
                exominer_score=None, has_spoc=1, star_status="completed")
    base.update(kw)
    return base


def test_exominer_eligible_when_has_spoc_and_thresholds_met():
    assert work_producers.exominer_eligible(_em_row()) is True


def test_exominer_not_eligible_below_cnn():
    assert work_producers.exominer_eligible(_em_row(cnn_score=0.69)) is False


def test_exominer_not_eligible_missing_cnn():
    assert work_producers.exominer_eligible(_em_row(cnn_score=None)) is False


def test_exominer_not_eligible_star_not_completed():
    assert work_producers.exominer_eligible(
        _em_row(star_status="pending")) is False


def test_exominer_not_eligible_has_spoc_null():
    # has_spoc IS NULL -> SPOC-gate has not resolved this star yet.
    assert work_producers.exominer_eligible(_em_row(has_spoc=None)) is False


def test_exominer_not_eligible_has_spoc_zero():
    # has_spoc=0 -> confirmed FFI-only; never queue-eligible for the
    # worker-native (DV-based) ExoMiner path.
    assert work_producers.exominer_eligible(_em_row(has_spoc=0)) is False


def test_exominer_not_eligible_when_already_scored():
    assert work_producers.exominer_eligible(
        _em_row(exominer_score=0.82)) is False


def test_exominer_not_eligible_for_ffi_only_sentinel():
    # db.mark_ffi_only_candidates / mark_unconfirmed_ffi_candidates write
    # exominer_score=-1 — not None, so a sentinelled candidate must never
    # re-enter the queue.
    assert work_producers.exominer_eligible(
        _em_row(exominer_score=-1.0)) is False


def test_maybe_enqueue_exominer_respects_flag(monkeypatch):
    calls = []

    class FakeStream:
        def enqueue(self, k, p):
            calls.append((k, p))
            return True

    monkeypatch.setattr(work_producers, "_get_stream",
                        lambda name: FakeStream())
    monkeypatch.delenv("EXONIT_WORK_STREAMS", raising=False)
    assert work_producers.maybe_enqueue_exominer(_em_row()) is False
    assert calls == []
    monkeypatch.setenv("EXONIT_WORK_STREAMS", "1")
    assert work_producers.maybe_enqueue_exominer(_em_row()) is True
    assert calls == [("7", {"candidate_id": 7, "tic_id": 123})]


def test_maybe_enqueue_exominer_skips_ineligible(monkeypatch):
    monkeypatch.setenv("EXONIT_WORK_STREAMS", "1")
    assert work_producers.maybe_enqueue_exominer(
        _em_row(cnn_score=0.1)) is False


# ---------------------------------------------------------------------------
# Primary (BLS) task queue (Fase 1, Task 5) — status='pending' is the only
# eligibility fact. G3 (review package, 2026-07-05) added the batched
# maybe_enqueue_primary_many, fixing the enqueue_bulk producer hook's
# one-enqueue()-per-task Python loop at 50k-100k-task discovery-run scale.
# ---------------------------------------------------------------------------

def _primary_row(**kw):
    base = dict(task_id="tk1", tic_id="500001", status="pending")
    base.update(kw)
    return base


def test_primary_eligible_when_pending():
    assert work_producers.primary_eligible(_primary_row()) is True


def test_primary_not_eligible_when_assigned():
    assert work_producers.primary_eligible(
        _primary_row(status="assigned")) is False


def test_maybe_enqueue_primary_respects_flag(monkeypatch):
    calls = []

    class FakeStream:
        def enqueue(self, k, p):
            calls.append((k, p))
            return True

    monkeypatch.setattr(work_producers, "_get_stream",
                        lambda name: FakeStream())
    monkeypatch.delenv("EXONIT_WORK_STREAMS", raising=False)
    assert work_producers.maybe_enqueue_primary(_primary_row()) is False
    assert calls == []
    monkeypatch.setenv("EXONIT_WORK_STREAMS", "1")
    assert work_producers.maybe_enqueue_primary(_primary_row()) is True
    assert calls == [("tk1", {"task_id": "tk1", "tic_id": "500001"})]


def test_maybe_enqueue_primary_skips_ineligible(monkeypatch):
    monkeypatch.setenv("EXONIT_WORK_STREAMS", "1")
    assert work_producers.maybe_enqueue_primary(
        _primary_row(status="assigned")) is False


def test_maybe_enqueue_primary_many_batches_eligible_rows(monkeypatch):
    calls = []

    class FakeStream:
        def enqueue_many(self, items):
            calls.append(list(items))
            return len(list(items))

    monkeypatch.setattr(work_producers, "_get_stream",
                        lambda name: FakeStream())
    monkeypatch.setenv("EXONIT_WORK_STREAMS", "1")
    rows = [_primary_row(task_id="tk1", tic_id="1"),
            _primary_row(task_id="tk2", tic_id="2", status="assigned"),
            _primary_row(task_id="tk3", tic_id="3")]
    n = work_producers.maybe_enqueue_primary_many(rows)
    assert n == 2, "the ineligible ('assigned') row must be filtered out"
    assert calls == [[("tk1", {"task_id": "tk1", "tic_id": "1"}),
                      ("tk3", {"task_id": "tk3", "tic_id": "3"})]]


def test_maybe_enqueue_primary_many_respects_flag(monkeypatch):
    calls = []

    class FakeStream:
        def enqueue_many(self, items):
            calls.append(list(items))
            return len(list(items))

    monkeypatch.setattr(work_producers, "_get_stream",
                        lambda name: FakeStream())
    monkeypatch.delenv("EXONIT_WORK_STREAMS", raising=False)
    assert work_producers.maybe_enqueue_primary_many([_primary_row()]) == 0
    assert calls == []


def test_maybe_enqueue_primary_many_empty_rows_is_noop(monkeypatch):
    monkeypatch.setenv("EXONIT_WORK_STREAMS", "1")
    assert work_producers.maybe_enqueue_primary_many([]) == 0
