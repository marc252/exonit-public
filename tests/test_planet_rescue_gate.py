"""Tests for the hot-Jupiter rescue gate in the verdict classifier.

Investigation 2026-05-09 (big run v3): 51 KP/CP candidates were tagged
``likely_false_positive`` by v2.1 stage1, where ``depth_log1p`` carries
66.8% of the gain. Every other channel agreed they were planets:
CNN ≥ 0.95, ExoMiner ≥ 0.9, FPP and NFPP both negligible. The rescue gate
overrides ``lfp → planet_candidate`` only when all four agree, leaving EBs
(NFPP > 0.5) and contamination cases untouched.
"""
from __future__ import annotations

import math

import pytest

from exonit.verdict_classifier import (
    CLASS_NAMES,
    RESCUE_CNN_MIN,
    RESCUE_EM_MIN,
    apply_planet_rescue_gate,
)


_LFP = CLASS_NAMES[0]
_PC = CLASS_NAMES[2]
_BASE_LFP = {
    "verdict": _LFP,
    "confidence": 0.85,
    "model_used": "v2_1_local",
}


def _strong_cand(**overrides):
    cand = {
        "cnn_score": 0.99,
        "exominer_score": 0.95,
        "triceratops_fpp": 1e-30,
        "triceratops_nfpp": 1e-30,
    }
    cand.update(overrides)
    return cand


def test_strong_signal_lfp_is_rescued_to_pc():
    out = apply_planet_rescue_gate(_strong_cand(), _BASE_LFP)
    assert out["verdict"] == _PC
    assert "+rescue" in out["model_used"]


def test_rescue_confidence_is_min_of_cnn_em_clipped():
    out = apply_planet_rescue_gate(
        _strong_cand(cnn_score=0.99, exominer_score=0.92), _BASE_LFP)
    assert math.isclose(out["confidence"], 0.92, abs_tol=1e-9)


def test_rescue_confidence_floor_06():
    # Both signals are above thresholds but clipping caps at 0.6 floor and
    # 0.95 ceiling. CNN=0.99, EM=0.99 → 0.95 cap.
    out = apply_planet_rescue_gate(
        _strong_cand(cnn_score=0.99, exominer_score=0.99), _BASE_LFP)
    assert out["confidence"] == 0.95


def test_low_cnn_blocks_rescue():
    out = apply_planet_rescue_gate(
        _strong_cand(cnn_score=0.80), _BASE_LFP)
    assert out["verdict"] == _LFP


def test_low_em_blocks_rescue():
    out = apply_planet_rescue_gate(
        _strong_cand(exominer_score=0.5), _BASE_LFP)
    assert out["verdict"] == _LFP


def test_high_fpp_blocks_rescue():
    out = apply_planet_rescue_gate(
        _strong_cand(triceratops_fpp=0.97), _BASE_LFP)
    assert out["verdict"] == _LFP


def test_high_nfpp_blocks_rescue():
    out = apply_planet_rescue_gate(
        _strong_cand(triceratops_nfpp=0.97), _BASE_LFP)
    assert out["verdict"] == _LFP


def test_missing_fpp_does_not_block_rescue():
    """Sentinel -1 / NaN / missing → assumed not-disputing — gate fires."""
    for v in (-1, -1.0, float("nan"), None):
        cand = _strong_cand()
        cand["triceratops_fpp"] = v
        out = apply_planet_rescue_gate(cand, _BASE_LFP)
        assert out["verdict"] == _PC, f"failed for fpp={v!r}"


def test_missing_nfpp_does_not_block_rescue():
    for v in (-1, -1.0, float("nan"), None):
        cand = _strong_cand()
        cand["triceratops_nfpp"] = v
        out = apply_planet_rescue_gate(cand, _BASE_LFP)
        assert out["verdict"] == _PC, f"failed for nfpp={v!r}"


def test_em_ffi_fallback_used_when_em_missing():
    """If exominer_score is missing, exominer_ffi_score should be picked up."""
    cand = _strong_cand()
    cand.pop("exominer_score")
    cand["exominer_ffi_score"] = 0.94
    out = apply_planet_rescue_gate(cand, _BASE_LFP)
    assert out["verdict"] == _PC


def test_em_ffi_fallback_too_low_blocks_rescue():
    cand = _strong_cand()
    cand.pop("exominer_score")
    cand["exominer_ffi_score"] = 0.4
    out = apply_planet_rescue_gate(cand, _BASE_LFP)
    assert out["verdict"] == _LFP


def test_em_minus1_sentinel_falls_back_to_ffi_same_as_missing():
    """exominer_score=-1 (no-DV cache-negative sentinel, see progress/
    investigation-exominer-minus1-2026-07-02.md) must be treated identically
    to exominer_score being absent — the FFI fallback must still engage.

    Before the 2026-07-02 fix this silently failed the rescue for every
    FFI-only hot-Jupiter candidate: -1 is not None, so the fallback never
    ran, and -1 then failed the internal ``em_raw >= 0`` guard, forcing
    em_v=None regardless of a genuinely strong exominer_ffi_score."""
    cand = _strong_cand(exominer_score=-1)
    cand["exominer_ffi_score"] = 0.94
    out = apply_planet_rescue_gate(cand, _BASE_LFP)
    assert out["verdict"] == _PC, (
        "exominer_score=-1 with a strong FFI score must still be rescued")

    # And must produce the identical result to the None-fallback case.
    cand_none = _strong_cand()
    cand_none.pop("exominer_score")
    cand_none["exominer_ffi_score"] = 0.94
    out_none = apply_planet_rescue_gate(cand_none, _BASE_LFP)
    assert out["confidence"] == out_none["confidence"]
    assert out["verdict"] == out_none["verdict"]


def test_em_minus1_sentinel_with_low_ffi_still_blocks_rescue():
    cand = _strong_cand(exominer_score=-1)
    cand["exominer_ffi_score"] = 0.4
    out = apply_planet_rescue_gate(cand, _BASE_LFP)
    assert out["verdict"] == _LFP


def test_non_lfp_verdict_passes_through_unchanged():
    base = {"verdict": _PC, "confidence": 0.92, "model_used": "v2_1"}
    out = apply_planet_rescue_gate(_strong_cand(), base)
    assert out is base or out == base


def test_uncertain_verdict_passes_through_unchanged():
    base = {"verdict": "uncertain", "confidence": 0.7, "model_used": "v2_1"}
    out = apply_planet_rescue_gate(_strong_cand(), base)
    assert out == base


def test_cnn_at_threshold_triggers_rescue():
    out = apply_planet_rescue_gate(
        _strong_cand(cnn_score=RESCUE_CNN_MIN), _BASE_LFP)
    assert out["verdict"] == _PC


def test_em_at_threshold_triggers_rescue():
    out = apply_planet_rescue_gate(
        _strong_cand(exominer_score=RESCUE_EM_MIN), _BASE_LFP)
    assert out["verdict"] == _PC


@pytest.mark.parametrize("tic_id,depth_ppm,cnn,em,fpp,nfpp", [
    # Real hot-Jupiter cases from /tmp/kpcp_classified.csv (big run v3)
    (458478250, 18081.5, 0.999991, 0.998746, 0.0, 1.096096e-70),
    (229791084, 16349.8, 0.999673, 0.995402, 0.0, 2.015160e-158),
    (468574941, 14122.9, 0.999952, 0.999413, 7.337850e-10, 1.584841e-169),
    (236387002, 39896.0, 1.0, 0.953060, 0.0, 1.239843e-143),
])
def test_real_world_hot_jupiter_cases_rescued(tic_id, depth_ppm, cnn, em,
                                              fpp, nfpp):
    """KP/CPs that v3 wrongly tagged likely_false_positive."""
    cand = {
        "cnn_score": cnn, "exominer_score": em,
        "triceratops_fpp": fpp, "triceratops_nfpp": nfpp,
        "depth_ppm": depth_ppm,
    }
    out = apply_planet_rescue_gate(cand, _BASE_LFP)
    assert out["verdict"] == _PC, (
        f"TIC {tic_id} should be rescued (depth={depth_ppm} ppm)")
    assert "+rescue" in out["model_used"]
