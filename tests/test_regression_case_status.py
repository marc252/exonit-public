"""Pure formatting/matching tests for scripts/regression_case_status.py (no DB)."""
import regression_case_status as rcs

CASE = {"tic_id": "4206066", "sector": "6", "period_days": "3.182785",
        "t0_btjd": "3998.736"}
T0_S6 = 1471.605


def test_not_promoted_with_match():
    rows = [{"period_days": 3.1737, "t0": 1471.6155, "snr": 12.5,
             "claude_verdict": "likely_false_positive", "verdict_proba": None}]
    line = rcs.summarize_case(CASE, rows)
    assert line.startswith("TIC 4206066 s6: 1 candidates, best P=3.1737")
    assert "t0 offset=16 min" in line
    assert "verdict_proba=n/a" in line
    assert line.endswith("NOT PROMOTED")


def test_promoted_best_is_highest_snr_match():
    rows = [
        {"period_days": 3.1737, "t0": 1471.6, "snr": 12.0,
         "claude_verdict": "planet_candidate", "verdict_proba": 0.91},
        {"period_days": 3.1737, "t0": 1471.5, "snr": 8.0,
         "claude_verdict": "likely_false_positive", "verdict_proba": 0.1},
        {"period_days": 9.9, "t0": 1470.0, "snr": 30.0,
         "claude_verdict": None, "verdict_proba": None},
    ]
    line = rcs.summarize_case(CASE, rows)
    assert "3 candidates" in line and "verdict_proba=0.910" in line
    assert line.endswith("→ PROMOTED")


def test_no_candidates():
    line = rcs.summarize_case(CASE, [])
    assert "0 candidates, best P=n/a" in line and line.endswith("NOT PROMOTED")


def test_t0_offset_wraps_to_nearest_transit():
    assert abs(rcs.t0_offset_min(T0_S6 + 3.182785 + 0.01, 3.18, 3998.736, 3.182785)
               - 0.01 * 1440) < 1.0
