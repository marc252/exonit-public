"""Unit tests for merge_same_period_candidates (P2)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "worker", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from duration_refine import merge_same_period_candidates  # noqa: E402


def _c(num, period, snr, dur=0.1):
    return {"candidate_num": num, "period_days": period, "snr": snr,
            "duration_days": dur, "t0": 1.0, "depth_ppm": 1000.0}


def test_three_slots_same_period_keep_max_snr():
    cands = [_c(1, 3.1737, 9.0, 0.159), _c(2, 3.1738, 14.0, 0.063),
             _c(3, 3.1736, 8.0, 0.032), _c(4, 7.5, 10.0)]
    out = merge_same_period_candidates(cands)
    assert [c["period_days"] for c in out] == [3.1738, 7.5]
    assert [c["candidate_num"] for c in out] == [1, 2]
    assert out[0]["duration_days"] == 0.063


def test_tie_keeps_lowest_candidate_num():
    cands = [_c(2, 3.0, 10.0, 0.2), _c(1, 3.0, 10.0, 0.1)]
    out = merge_same_period_candidates(cands)
    assert len(out) == 1 and out[0]["duration_days"] == 0.1
    assert out[0]["candidate_num"] == 1


def test_harmonics_not_merged():
    out = merge_same_period_candidates([_c(1, 3.0, 10.0), _c(2, 6.0, 9.0),
                                        _c(3, 1.5, 8.0)])
    assert len(out) == 3


def test_distinct_periods_untouched_and_input_not_mutated():
    cands = [_c(1, 2.0, 10.0), _c(2, 5.0, 9.0), _c(3, 11.0, 8.0)]
    out = merge_same_period_candidates(cands)
    assert out == cands
    assert out[0] is not cands[0]


def test_empty():
    assert merge_same_period_candidates([]) == []
