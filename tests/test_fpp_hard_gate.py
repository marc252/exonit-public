"""Tests for the FPP hard-gate in the verdict classifier.

Investigation 2026-05-03: 15 candidates flagged with EM≥0.9 had TRICERATOPS
FPP≥0.5 yet the classifier still labelled them planet_candidate. The gate
forces ``likely_false_positive`` whenever FPP or NFPP is ≥0.5.
"""
from __future__ import annotations

import math

import pytest

from exonit.verdict_classifier import (
    CLASS_NAMES,
    FPP_HARD_GATE_THRESHOLD,
    apply_fpp_hard_gate,
)


_BASE_PC = {
    "verdict": "planet_candidate",
    "confidence": 0.95,
    "model_used": "stub_model",
}
_LFP = "likely_false_positive"
assert _LFP == CLASS_NAMES[0]


def test_high_fpp_overrides_planet_verdict():
    out = apply_fpp_hard_gate({"triceratops_fpp": 0.97, "triceratops_nfpp": 0.0},
                              _BASE_PC)
    assert out["verdict"] == _LFP
    assert math.isclose(out["confidence"], 0.97, abs_tol=1e-9)
    assert "+fpp_gate" in out["model_used"]


def test_high_nfpp_alone_overrides_planet_verdict():
    out = apply_fpp_hard_gate({"triceratops_fpp": 0.10, "triceratops_nfpp": 0.80},
                              _BASE_PC)
    assert out["verdict"] == _LFP
    assert math.isclose(out["confidence"], 0.80, abs_tol=1e-9)


def test_below_threshold_passes_through():
    cand = {"triceratops_fpp": 0.40, "triceratops_nfpp": 0.10}
    out = apply_fpp_hard_gate(cand, _BASE_PC)
    assert out is _BASE_PC or out == _BASE_PC


def test_threshold_boundary_is_inclusive():
    cand = {"triceratops_fpp": FPP_HARD_GATE_THRESHOLD,
            "triceratops_nfpp": 0.0}
    out = apply_fpp_hard_gate(cand, _BASE_PC)
    assert out["verdict"] == _LFP


def test_invalid_fpp_does_not_trigger():
    """Sentinel -1 (TRICERATOPS skipped) and NaN must not trigger the gate."""
    for v in (-1, -1.0, float("nan")):
        cand = {"triceratops_fpp": v, "triceratops_nfpp": v}
        out = apply_fpp_hard_gate(cand, _BASE_PC)
        assert out == _BASE_PC, f"failed for sentinel {v!r}"


def test_missing_keys_does_not_trigger():
    out = apply_fpp_hard_gate({}, _BASE_PC)
    assert out == _BASE_PC


def test_already_lfp_with_high_fpp_keeps_higher_confidence():
    """If base says LFP@0.6 and FPP=0.97, the gate raises confidence."""
    base = {"verdict": _LFP, "confidence": 0.60, "model_used": "stub"}
    out = apply_fpp_hard_gate({"triceratops_fpp": 0.97}, base)
    assert out["verdict"] == _LFP
    assert out["confidence"] == pytest.approx(0.97)


def test_confidence_clipped_to_one():
    out = apply_fpp_hard_gate({"triceratops_fpp": 1.0, "triceratops_nfpp": 0.0},
                              _BASE_PC)
    assert out["confidence"] == 1.0


@pytest.mark.parametrize("fpp,nfpp", [
    (1.0, 0.0),    # TIC 104195270, 117689799, 239541449 #1, etc.
    (0.97, 0.97),  # TIC 143257766 #1
    (0.84, 0.0),   # TIC 144043410 #1
    (0.99, 0.0),   # TIC 312899686 #4
])
def test_real_world_mode_a_cases_flip_to_lfp(fpp, nfpp):
    """Replays the 15 Mode-A candidates from the FP investigation."""
    cand = {"triceratops_fpp": fpp, "triceratops_nfpp": nfpp,
            "exominer_score": 0.98}
    base = {"verdict": "planet_candidate", "confidence": 0.92,
            "model_used": "v2_1"}
    out = apply_fpp_hard_gate(cand, base)
    assert out["verdict"] == _LFP
