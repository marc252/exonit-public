"""Tests for exonit.verdict_io — D8 dict-construction extraction.

docs/reviews/2026-07-02-opus48-cleanup-review.md, item D8: the candidate/
star_info dict-building that server.py's _run_verdict_background and
scripts/recompute_verdict.py used to hand-mirror now lives here. These tests
cover the pure functions in isolation (no DB, no server, no threading).
"""
from exonit.verdict_io import (
    CANDIDATE_FIELDS,
    STAR_INFO_FIELDS,
    build_candidate_dict,
    build_star_info,
    default_vetting_result,
)


class TestBuildStarInfo:
    def test_none_star_row_returns_tic_id_only(self):
        info = build_star_info("123456", None)
        assert info == {"tic_id": "123456"}

    def test_empty_star_row_returns_tic_id_only(self):
        """An empty mapping is falsy, matching the original ``if star_row:``
        guard (both the server.py inline code and the pre-extraction
        behaviour treat an empty row the same as no row)."""
        info = build_star_info("123456", {})
        assert info == {"tic_id": "123456"}

    def test_full_star_row_populates_all_fields(self):
        row = {"tmag": 10.5, "ra": 100.0, "dec": -30.0, "n_sectors": 3,
               "time_span_days": 730.0, "n_datapoints": 1200}
        info = build_star_info(123456, row)
        assert info["tic_id"] == "123456"  # cast to str
        for k in STAR_INFO_FIELDS:
            assert info[k] == row[k]

    def test_partial_star_row_omits_missing_keys(self):
        """Mirrors server.py's original star_row[0][k] indexing behaviour
        for its actual SELECT (all 6 columns always present); a caller with
        a genuinely partial row degrades gracefully instead of KeyError."""
        row = {"tmag": 10.5, "ra": 100.0}
        info = build_star_info("999", row)
        assert info == {"tic_id": "999", "tmag": 10.5, "ra": 100.0}

    def test_extra_keys_on_star_row_are_ignored(self):
        row = {"tmag": 10.5, "ra": 100.0, "dec": -30.0, "n_sectors": 3,
               "time_span_days": 730.0, "n_datapoints": 1200,
               "unrelated_column": "should not leak"}
        info = build_star_info("999", row)
        assert "unrelated_column" not in info


class TestBuildCandidateDict:
    def _row(self, **overrides):
        row = {
            "tic_id": "42", "candidate_num": 1, "period_days": 7.2,
            "depth_ppm": 800, "duration_days": 0.1, "snr": 12.5,
            "cnn_score": 0.82, "triceratops_fpp": 0.03,
            "triceratops_nfpp": 0.97, "triceratops_prob_planet": 0.9,
            "exominer_score": 0.91, "exominer_ffi_score": None,
            "exominer_tce_period": 7.2, "exominer_sector_run": "1-40",
            # Extra columns a caller might SELECT that predict_verdict does
            # not read (mirrors server.py's worker_exominer_result, which
            # additionally selects t0 for its own purposes).
            "t0": 1234.5, "claude_verdict": "planet_candidate",
        }
        row.update(overrides)
        return row

    def test_builds_expected_shape(self):
        cand = build_candidate_dict(self._row())
        assert cand["tic_id"] == "42"
        assert cand["candidate_num"] == 1
        for k in CANDIDATE_FIELDS:
            assert k in cand

    def test_ignores_extra_row_columns(self):
        """t0 / claude_verdict are present on the row but not part of the
        classifier's expected shape — must not leak into the candidate
        dict (matches scripts/recompute_verdict.py's _build_candidate_dict,
        which never SELECTs them at all)."""
        cand = build_candidate_dict(self._row())
        assert "t0" not in cand
        assert "claude_verdict" not in cand

    def test_preserves_sentinel_values(self):
        """exominer_score == -1.0 ('no SPOC score') must pass through
        unchanged — verdict_io is a pure passthrough; the -1→absent
        normalization happens at the read sites (classifier feature-build,
        db._strong_where via NULLIF; see fix 2026-07-02)."""
        cand = build_candidate_dict(self._row(exominer_score=-1.0))
        assert cand["exominer_score"] == -1.0

    def test_tic_id_and_candidate_num_are_cast(self):
        cand = build_candidate_dict(self._row(tic_id=42, candidate_num="3"))
        assert cand["tic_id"] == "42"
        assert cand["candidate_num"] == 3
        assert isinstance(cand["candidate_num"], int)


def test_default_vetting_result_is_passed():
    assert default_vetting_result() == {"passed": True}
