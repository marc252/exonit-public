"""Shared semantics for fpp_verdict across the two FPP modules.

Audit 2026-04-17 (items #1 and #7):
  - Giacalone+ 2020 requires FPP<0.015 AND NFPP<0.001 for validation.
    The old single-arg fpp_verdict(fpp) could validate based on FPP alone,
    missing the many-nearby (high NFPP) false-positive mode.
  - TIC-only background catalogs are incomplete beyond Tmag~16-17 so BEB
    contamination in crowded fields (|b|<10 or N_nearby>15) is under-
    estimated. A validated verdict must downgrade to "candidate" when
    high_crowding is set.
"""

import pytest


_VERDICT_MODULES = [
    pytest.param("fpp_native", id="fpp_native"),
    pytest.param("triceratops_fpp", id="triceratops_fpp"),
]


def _verdict_fn(module_name):
    import importlib
    return importlib.import_module(module_name).fpp_verdict


@pytest.mark.parametrize("module_name", _VERDICT_MODULES)
class TestFppVerdictJointThreshold:
    def test_both_low_validates(self, module_name):
        verdict = _verdict_fn(module_name)
        assert verdict(0.01, nfpp=0.0005) == "validated"

    def test_fpp_low_but_nfpp_high_is_candidate(self, module_name):
        """A candidate with many nearby look-alikes (high NFPP) must NOT
        validate even if target-FPP is formally tiny."""
        verdict = _verdict_fn(module_name)
        assert verdict(0.01, nfpp=0.01) == "candidate"

    def test_fpp_high_stays_likely_fp(self, module_name):
        verdict = _verdict_fn(module_name)
        assert verdict(0.7, nfpp=0.0) == "likely_fp"

    def test_fpp_between_thresholds_is_candidate(self, module_name):
        verdict = _verdict_fn(module_name)
        assert verdict(0.3, nfpp=0.0) == "candidate"

    def test_nfpp_missing_cannot_validate(self, module_name):
        """Conservative default: without NFPP info we cannot validate."""
        verdict = _verdict_fn(module_name)
        assert verdict(0.01, nfpp=None) == "candidate"


@pytest.mark.parametrize("module_name", _VERDICT_MODULES)
class TestFppVerdictHighCrowding:
    def test_high_crowding_downgrades_validated(self, module_name):
        """TIC-only background is incomplete in crowded fields — even a
        formally-validating FPP/NFPP pair must be reported as candidate."""
        verdict = _verdict_fn(module_name)
        assert verdict(0.001, nfpp=0.0, high_crowding=True) == "candidate"

    def test_non_crowded_validates_normally(self, module_name):
        verdict = _verdict_fn(module_name)
        assert verdict(0.001, nfpp=0.0, high_crowding=False) == "validated"

    def test_high_crowding_does_not_upgrade_likely_fp(self, module_name):
        """high_crowding must never promote (e.g. a strong FP stays FP)."""
        verdict = _verdict_fn(module_name)
        assert verdict(0.7, nfpp=0.0, high_crowding=True) == "likely_fp"
