"""Tests that _run_verdict_background stores analysis="" and never calls the
LoRA explainer service (HTTP). The explainer is deferred to closeout.
"""
import os
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_candidate():
    return {
        "candidate_num": 1,
        "period_days": 7.2,
        "depth_ppm": 800,
        "duration_days": 0.1,
        "snr": 12.5,
        "cnn_score": 0.82,
        "triceratops_fpp": 0.03,
        "triceratops_nfpp": 0.97,
        "triceratops_verdict": "PLANET",
        "exominer_score": 0.91,
    }


def _make_db(candidate_id=42):
    """Minimal fake db that supports the two queries _run_verdict_background does."""
    db = MagicMock()
    # stars query returns one row
    star_row = [{"tmag": 10.5, "ra": 100.0, "dec": -30.0,
                 "n_sectors": 3, "time_span_days": 730.0, "n_datapoints": 1200}]
    # candidates query returns one row with id
    cand_row = [{"id": candidate_id}]

    def _query(sql, params=()):
        if "FROM stars" in sql:
            return star_row
        if "FROM candidates" in sql:
            return cand_row
        return []

    db._query.side_effect = _query
    return db


# ---------------------------------------------------------------------------
# Core test: analysis stored as empty string, no HTTP calls
# ---------------------------------------------------------------------------

def test_run_verdict_background_stores_empty_analysis_no_http(monkeypatch):
    """_run_verdict_background must:
    1. call db.update_candidate_claude with analysis=="".
    2. never make any HTTP request to the explainer service.
    """
    from exonit.server import _run_verdict_background

    db = _make_db(candidate_id=99)
    candidate = _sample_candidate()

    fake_verdict_result = {
        "verdict": "PLANET_CANDIDATE",
        "confidence": 0.87,
        "analysis": "Some classifier text that should NOT be stored inline.",
    }

    # Patch predict_verdict so no model file is needed
    monkeypatch.setenv("EXONIT_EXPLAINER_URL", "http://fake-explainer:8500/api/explain")
    with patch("exonit.verdict_classifier.predict_verdict",
               return_value=fake_verdict_result) as mock_predict, \
         patch("requests.post") as mock_http_post:

        _run_verdict_background(db, tic_id=123456, candidate=candidate, cnn_score=0.82)

    # Verdict classifier must have been called
    mock_predict.assert_called_once()

    # update_candidate_claude must have been called with empty analysis
    db.update_candidate_claude.assert_called_once()
    _args = db.update_candidate_claude.call_args
    stored_analysis = _args[0][2]  # positional: (candidate_id, verdict, analysis)
    assert stored_analysis == "", (
        f"Expected analysis='', got {stored_analysis!r}. "
        "LoRA explanation must be deferred to closeout."
    )

    # No HTTP request to the explainer must have been made
    mock_http_post.assert_not_called()


def test_run_verdict_background_forwards_verdict_proba():
    """result["proba"] is forwarded as the verdict_proba kwarg (P3)."""
    from exonit.server import _run_verdict_background

    db = _make_db(candidate_id=5)
    fake = {"verdict": "planet_candidate", "confidence": 0.9,
            "model_used": "x", "proba": 0.37}
    with patch("exonit.verdict_classifier.predict_verdict", return_value=fake):
        _run_verdict_background(db, tic_id=1, candidate=_sample_candidate(),
                                cnn_score=0.8)
    db.update_candidate_claude.assert_called_once()
    assert db.update_candidate_claude.call_args.kwargs["verdict_proba"] == 0.37
    assert db.update_candidate_claude.call_args[0][2] == ""


def test_run_verdict_background_tolerates_missing_proba():
    """Claude-fallback dicts have no "proba" key: stored as None, no crash."""
    from exonit.server import _run_verdict_background

    db = _make_db(candidate_id=6)
    fake = {"verdict": "uncertain", "confidence": 0.5, "analysis": "t"}
    with patch("exonit.verdict_classifier.predict_verdict", return_value=fake):
        _run_verdict_background(db, tic_id=2, candidate=_sample_candidate(),
                                cnn_score=0.8)
    db.update_candidate_claude.assert_called_once()
    assert db.update_candidate_claude.call_args.kwargs["verdict_proba"] is None


def test_run_verdict_background_no_explainer_url_no_http(monkeypatch):
    """Even without EXONIT_EXPLAINER_URL set, analysis must still be empty."""
    from exonit.server import _run_verdict_background

    monkeypatch.delenv("EXONIT_EXPLAINER_URL", raising=False)

    db = _make_db(candidate_id=77)
    candidate = _sample_candidate()

    fake_verdict_result = {
        "verdict": "FALSE_POSITIVE",
        "confidence": 0.95,
        "analysis": "Classifier says FP.",
    }

    with patch("exonit.verdict_classifier.predict_verdict",
               return_value=fake_verdict_result), \
         patch("requests.post") as mock_http_post:

        _run_verdict_background(db, tic_id=999, candidate=candidate, cnn_score=0.2)

    db.update_candidate_claude.assert_called_once()
    stored_analysis = db.update_candidate_claude.call_args[0][2]
    assert stored_analysis == ""
    mock_http_post.assert_not_called()


def test_explain_via_m4_removed():
    """_explain_via_m4 must no longer exist in the server module."""
    import exonit.server as srv
    assert not hasattr(srv, "_explain_via_m4"), (
        "_explain_via_m4 was not removed. It must be eliminated as per the "
        "decoupling spec."
    )


def test_clean_lora_text_removed():
    """_clean_lora_text must no longer exist in the server module."""
    import exonit.server as srv
    assert not hasattr(srv, "_clean_lora_text"), (
        "_clean_lora_text was not removed. It must be eliminated as per the "
        "decoupling spec."
    )


def test_lora_tag_removed():
    """_LORA_TAG must no longer exist in the server module."""
    import exonit.server as srv
    assert not hasattr(srv, "_LORA_TAG"), (
        "_LORA_TAG was not removed. It must be eliminated as per the "
        "decoupling spec."
    )
