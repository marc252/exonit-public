"""Tests for v2.1 ExoMiner-FFI fallback (Phase 3).

When exominer_score (SPOC) is NULL and exominer_ffi_score is present,
predict_v2_1 must use the FFI score as the exominer feature instead of
defaulting to 0. When SPOC score is present, it takes precedence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exonit.verdict_classifier_v2_1 import (
    _base_feature_names_v2_1,
    _load_v2_1_model,
    fit_full_hierarchical,
    predict_v2_1,
    preprocess_v2_1,
    save_v2_1_model,
)
from exonit.verdict_classifier import CLASS_NAMES, compute_target

# ---------------------------------------------------------------------------
# Synthetic sector list
# ---------------------------------------------------------------------------
_TOP_SECTORS = [
    "s01", "s02", "s03", "s04", "s05",
    "s06", "s07", "s08", "s09", "s10",
    "s11", "s12", "s13", "s14", "s15",
]


# ---------------------------------------------------------------------------
# Helper: build a tiny v2.1 model sensitive to exominer_score
#
# Training design: only exominer_score varies across classes. All other
# features are fixed at neutral values. This guarantees the model will
# discriminate specifically on the exominer_score feature, making it
# possible to verify the fallback logic without a very large model.
#
# Learned decision boundaries (empirically confirmed at n_estimators=200):
#   exominer ≈ 0.0–0.4   → likely_false_positive
#   exominer ≈ 0.5       → uncertain
#   exominer ≈ 0.8–0.95  → planet_candidate
# ---------------------------------------------------------------------------

def _make_exominer_sensitive_df() -> pd.DataFrame:
    """90 rows, 3 balanced classes, only exominer_score varies."""
    verdicts = ["likely_false_positive", "uncertain", "planet_candidate"]
    exom_vals = [0.05, 0.5, 0.95]
    rows = []
    for tic in range(90):
        v_idx = tic % 3
        sec = _TOP_SECTORS[tic % len(_TOP_SECTORS)]
        rows.append({
            "tic_id": str(tic), "candidate_num": 1,
            "period_days": 2.5, "duration_days": 0.1, "depth_ppm": 500.0,
            "snr": 12.0, "time_span_days": 100.0,
            "cnn_score": 0.5,                       # neutral — not discriminating
            "triceratops_fpp": 0.5,                 # neutral — not discriminating
            "exominer_score": exom_vals[v_idx],     # sole discriminating feature
            "exominer_tce_period": 2.5,
            "exominer_sector_run": sec,
            "n_sectors": 3, "n_datapoints": 5000,
            "triceratops_nfpp": 0.5,
            "triceratops_prob_planet": 0.5,
            "berger_mass": 1.0, "berger_rad": 1.0, "berger_teff": 5778.0,
            "berger_logg": 4.4, "berger_feh": 0.0,
            "claude_verdict": verdicts[v_idx],
        })
    return pd.DataFrame(rows)


def _save_tiny_v2_1_model(tmp_path):
    """Train and save a v2.1 model sensitive to exominer_score."""
    df = _make_exominer_sensitive_df()
    y = compute_target(df)
    X = preprocess_v2_1(df, top_sectors=_TOP_SECTORS)
    feature_names = list(X.columns)

    # n_estimators=200 gives clean separation on this synthetic dataset
    stage1, stage2 = fit_full_hierarchical(
        X, y, n_estimators=200, random_state=42)

    path = tmp_path / "verdict_classifier_v2_1_2026-05-08.pkl"
    save_v2_1_model(stage1, stage2, path,
                    feature_names=feature_names,
                    top_sectors=_TOP_SECTORS)
    return path


# Neutral candidate for inference (all features at training-neutral values).
# The model uses ONLY exominer_score to classify at this operating point.
_NEUTRAL_CAND = {
    "cnn_score": 0.5, "triceratops_fpp": 0.5,
    "triceratops_nfpp": 0.5, "triceratops_prob_planet": 0.5,
    "snr": 12.0, "depth_ppm": 500.0, "period_days": 2.5,
    "duration_days": 0.1, "exominer_tce_period": 2.5,
    "exominer_sector_run": "s01",
}


# ---------------------------------------------------------------------------
# Test 1: FFI fallback when SPOC is None
# ---------------------------------------------------------------------------

def test_verdict_uses_phase3_score_when_spoc_missing(tmp_path, monkeypatch):
    """With exominer_score=None + exominer_ffi_score=0.8 (high),
    predict_v2_1 must produce planet_candidate — the FFI score is used as
    the exominer feature when SPOC is absent.

    Without the fallback, exominer_score=None becomes 0 (low), which
    causes the model to predict likely_false_positive.
    """
    _save_tiny_v2_1_model(tmp_path)
    monkeypatch.setenv("EXONIT_MODELS_DIR", str(tmp_path))
    _load_v2_1_model.cache_clear()

    cand = {**_NEUTRAL_CAND, "exominer_score": None, "exominer_ffi_score": 0.8}
    result = predict_v2_1(cand, star_info={})
    verdict = result["verdict"]
    assert verdict == "planet_candidate", (
        f"expected planet_candidate (FFI fallback active), got {verdict!r}")


# ---------------------------------------------------------------------------
# Test 2: SPOC takes precedence when both scores present
# ---------------------------------------------------------------------------

def test_verdict_prefers_spoc_when_both_present(tmp_path, monkeypatch):
    """With exominer_score=0.05 (weak SPOC) and exominer_ffi_score=0.99 (strong),
    SPOC must dominate — the fallback must NOT apply when exominer_score is not None.

    The model trained on this dataset maps exominer=0.05 → likely_false_positive.
    If the fallback were incorrectly applied, the verdict would flip to
    planet_candidate due to the strong FFI score.
    """
    _save_tiny_v2_1_model(tmp_path)
    monkeypatch.setenv("EXONIT_MODELS_DIR", str(tmp_path))
    _load_v2_1_model.cache_clear()

    cand = {**_NEUTRAL_CAND, "exominer_score": 0.05, "exominer_ffi_score": 0.99}
    result = predict_v2_1(cand, star_info={})
    verdict = result["verdict"]
    assert verdict in ("uncertain", "likely_false_positive"), (
        f"expected SPOC dominance (uncertain/lfp), got {verdict!r}")


# ---------------------------------------------------------------------------
# Test 3: exominer_score=-1 (no-DV cache-negative sentinel) must be treated
# EXACTLY like exominer_score=None — i.e. the FFI fallback must trigger for
# -1 too, not just for a real NULL.
#
# See progress/investigation-exominer-minus1-2026-07-02.md: ~61% of
# candidates.rows carry exominer_score=-1 (has_spoc / no-metadata /
# bulk-FFI-only gates), and prior to this fix -1 was NOT None so it never
# fell back to exominer_ffi_score — hard-vetoing ~46k TICs out of the
# strong tier and feeding the model an out-of-distribution feature value.
# ---------------------------------------------------------------------------

def test_verdict_minus1_sentinel_matches_none_output_exactly(tmp_path, monkeypatch):
    """predict_v2_1 with exominer_score=-1 must produce THE SAME output
    (verdict + confidence, bit-identical) as exominer_score=None, for the
    same exominer_ffi_score."""
    _save_tiny_v2_1_model(tmp_path)
    monkeypatch.setenv("EXONIT_MODELS_DIR", str(tmp_path))
    _load_v2_1_model.cache_clear()

    cand_none = {**_NEUTRAL_CAND, "exominer_score": None, "exominer_ffi_score": 0.8}
    cand_neg1 = {**_NEUTRAL_CAND, "exominer_score": -1, "exominer_ffi_score": 0.8}

    result_none = predict_v2_1(cand_none, star_info={})
    result_neg1 = predict_v2_1(cand_neg1, star_info={})

    assert result_neg1["verdict"] == result_none["verdict"], (
        f"exominer_score=-1 must fall back to FFI like None does; "
        f"got {result_neg1['verdict']!r} vs None's {result_none['verdict']!r}"
    )
    assert result_neg1["confidence"] == result_none["confidence"], (
        "exominer_score=-1 and None must feed the model an IDENTICAL "
        "feature value (both -> the FFI score) -> bit-identical confidence; "
        f"got {result_neg1['confidence']!r} vs {result_none['confidence']!r}"
    )
    # Sanity: without the fallback both would default the exominer feature
    # to 0 (low) and this synthetic model maps that to likely_false_positive
    # (see the module docstring's decision boundaries) — confirm the
    # fallback actually engaged (the FFI score of 0.8 wins), not that both
    # paths coincidentally landed on the same fallback-free answer.
    assert result_none["verdict"] == "planet_candidate"


# ---------------------------------------------------------------------------
# P3: raw P(planet) is exposed as result["proba"]
# ---------------------------------------------------------------------------

def test_v2_1_proba_is_raw_p_planet(tmp_path, monkeypatch):
    import exonit.verdict_classifier_v2_1 as v21
    _save_tiny_v2_1_model(tmp_path)
    monkeypatch.setenv("EXONIT_MODELS_DIR", str(tmp_path))
    _load_v2_1_model.cache_clear()
    monkeypatch.setattr(v21, "_predict_single_v2_1",
                        lambda *a, **k: np.array([0.1, 0.2, 0.7]))
    result = v21.predict_v2_1(dict(_NEUTRAL_CAND), star_info={})
    assert result["verdict"] == "planet_candidate"
    assert result["proba"] == pytest.approx(0.7)


def test_v2_1_rescue_gate_keeps_raw_proba(tmp_path, monkeypatch):
    """LFP flipped to planet by the rescue gate keeps the raw model p."""
    import exonit.verdict_classifier_v2_1 as v21
    _save_tiny_v2_1_model(tmp_path)
    monkeypatch.setenv("EXONIT_MODELS_DIR", str(tmp_path))
    _load_v2_1_model.cache_clear()
    monkeypatch.setattr(v21, "_predict_single_v2_1",
                        lambda *a, **k: np.array([0.8, 0.15, 0.05]))
    cand = {**_NEUTRAL_CAND, "cnn_score": 0.99, "exominer_score": 0.99,
            "triceratops_fpp": 0.0, "triceratops_nfpp": 0.0}
    result = v21.predict_v2_1(cand, star_info={})
    assert result["model_used"].endswith("+rescue")
    assert result["verdict"] == "planet_candidate"
    assert result["proba"] == pytest.approx(0.05)


def test_v2_2_proba_both_branches(monkeypatch):
    import exonit.verdict_classifier_v2_2 as v22
    for thr in ({"lfp_threshold": 0.5, "pc_threshold": 0.5}, {}):
        payload = {"stage1": None, "stage2": None, "feature_names": [],
                   "top_sectors": [], "path": "m.pkl", **thr}
        monkeypatch.setattr(v22, "_load_v2_2_model", lambda p=payload: p)
        monkeypatch.setattr(v22, "_predict_single_v2_2_stages",
                            lambda *a, **k: (0.2, 0.9))
        result = v22.predict_v2_2({}, star_info={})
        assert result["proba"] == pytest.approx((1 - 0.2) * 0.9)
