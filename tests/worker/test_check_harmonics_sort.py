"""Unit tests for the check_harmonics sort-by-SNR fix.

Audit 2026-04-17 (Observations, severity Medium/Low > Vetting):
"check_harmonics sort by SNR — sort candidates by SNR before the
loop to avoid losing the flag on ties."

The original code iterates ``all_candidates`` in caller order and
skips entries where ``other.snr <= candidate.snr``. This means:

1. Two equal-SNR harmonics never mark each other (strict <= skip),
   so a duplicate-signal P=X and P=2X with the same SNR both report
   ``is_harmonic=False`` instead of flagging one against the other.
2. Among multiple stronger parents, the first one encountered wins
   (caller-order dependent) rather than the strongest one.

Fix: sort ``all_candidates`` by SNR descending first, then break
``<=`` into ``<`` so equal-SNR harmonics still flag. Returns the
strongest matching parent.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "worker") not in sys.path:
    sys.path.append(str(ROOT / "worker"))

from worker.vetting import check_harmonics  # noqa: E402


def _cand(num: int, period: float, snr: float) -> dict:
    return {"candidate_num": num, "period_days": period, "snr": snr}


# ─────────────────────────────────────────────────────────────────────
# Test 1: equal-SNR duplicate (harmonic) still gets flagged
# ─────────────────────────────────────────────────────────────────────
def test_equal_snr_harmonic_flagged():
    cand_num = 2
    all_c = [
        _cand(1, period=4.0, snr=20.0),
        _cand(2, period=2.0, snr=20.0),   # P/2 of #1, same SNR
    ]
    # Vet candidate 2 against all
    result = check_harmonics(all_c[1], all_c)
    assert result["is_harmonic"] is True, \
        f"Expected P=2.0 flagged as harmonic of P=4.0 (equal SNR), got {result}"
    assert result["harmonic_of"] == 1


# ─────────────────────────────────────────────────────────────────────
# Test 2: strongest parent wins when multiple candidates match
# ─────────────────────────────────────────────────────────────────────
def test_strongest_parent_selected():
    cand_num = 3
    all_c = [
        _cand(1, period=2.0, snr=15.0),   # weaker parent, ratio=0.5 of #3
        _cand(2, period=2.0, snr=25.0),   # stronger parent, same ratio
        _cand(3, period=4.0, snr=10.0),   # our candidate
    ]
    result = check_harmonics(all_c[2], all_c)
    assert result["is_harmonic"] is True
    assert result["harmonic_of"] == 2, \
        f"Expected harmonic_of=2 (stronger SNR=25), got {result}"


# ─────────────────────────────────────────────────────────────────────
# Test 3: unrelated candidates produce no flag
# ─────────────────────────────────────────────────────────────────────
def test_unrelated_not_flagged():
    all_c = [
        _cand(1, period=5.0, snr=20.0),
        _cand(2, period=7.3, snr=18.0),   # 7.3/5.0 not in HARMONIC_RATIOS
    ]
    result = check_harmonics(all_c[1], all_c)
    assert result["is_harmonic"] is False


# ─────────────────────────────────────────────────────────────────────
# Test 4: self-reference doesn't match itself
# ─────────────────────────────────────────────────────────────────────
def test_self_not_matched():
    cand = _cand(1, period=5.0, snr=20.0)
    result = check_harmonics(cand, [cand])
    assert result["is_harmonic"] is False


# ─────────────────────────────────────────────────────────────────────
# Test 5: weaker candidate isn't considered a parent
# ─────────────────────────────────────────────────────────────────────
def test_weaker_not_parent():
    all_c = [
        _cand(1, period=2.0, snr=10.0),   # weaker
        _cand(2, period=4.0, snr=20.0),   # our candidate (stronger)
    ]
    # Candidate 2 should not be flagged as harmonic of weaker #1
    result = check_harmonics(all_c[1], all_c)
    assert result["is_harmonic"] is False
