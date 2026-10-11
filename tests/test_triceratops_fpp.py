"""Tests for TRICERATOPS FPP module."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestShouldRunTriceratops:
    def test_strong_candidate_accepted(self):
        from triceratops_fpp import should_run_triceratops
        candidate = {"snr": 15.0}
        vetting = {"passed": True}
        assert should_run_triceratops(candidate, vetting, cnn_score=0.9) is True

    def test_weak_snr_rejected(self):
        from triceratops_fpp import should_run_triceratops
        candidate = {"snr": 8.0}
        vetting = {"passed": True}
        assert should_run_triceratops(candidate, vetting, cnn_score=0.9) is False

    def test_failed_vetting_rejected(self):
        from triceratops_fpp import should_run_triceratops
        candidate = {"snr": 20.0}
        vetting = {"passed": False}
        assert should_run_triceratops(candidate, vetting, cnn_score=0.9) is False

    def test_low_cnn_rejected(self):
        from triceratops_fpp import should_run_triceratops
        candidate = {"snr": 20.0}
        vetting = {"passed": True}
        assert should_run_triceratops(candidate, vetting, cnn_score=0.3) is False


class TestCalcFpp:
    @patch("triceratops_fpp._create_target")
    def test_returns_fpp_dict(self, mock_create):
        from triceratops_fpp import calc_fpp
        mock_target = MagicMock()
        mock_target.probs = [0.85, 0.10, 0.05]
        mock_target.FPP = 0.15
        mock_target.NFPP = 0.05
        mock_create.return_value = mock_target

        time = np.linspace(0, 30, 5000)
        flux = np.ones(5000)
        result = calc_fpp(
            tic_id="123456", period_days=5.0, t0=0.0,
            duration_days=0.1, depth_ppm=500,
            sectors=np.array([1, 2]), time=time, flux=flux)

        assert "fpp" in result
        assert "nfpp" in result
        assert result["fpp"] == 0.15
        mock_target.calc_probs.assert_called_once()

    @patch("triceratops_fpp._create_target")
    def test_returns_none_on_failure(self, mock_create):
        from triceratops_fpp import calc_fpp
        mock_create.side_effect = Exception("MAST down")

        time = np.linspace(0, 30, 5000)
        flux = np.ones(5000)
        result = calc_fpp(
            tic_id="123456", period_days=5.0, t0=0.0,
            duration_days=0.1, depth_ppm=500,
            sectors=np.array([1]), time=time, flux=flux)

        assert result is None


class TestOutOfTransitFluxErr:
    """Audit item #5: the sigma handed to calc_probs must reflect baseline
    noise, not the whole transit-inclusive curve. std-with-transit
    overestimates sigma and dilutes FPP significance."""

    def test_mad_matches_baseline_sigma_with_transit_dip(self):
        """Synthetic lc: σ=0.001 baseline, central 10% dip of depth 0.01.
        std of full curve will be inflated by the dip; MAD must not be."""
        import numpy as np
        from triceratops_fpp import _estimate_flux_err_oot

        rng = np.random.default_rng(42)
        n = 1000
        flux = 1.0 + rng.normal(0, 0.001, size=n)
        # Inject a 10% phase-width dip of depth 0.01
        dip_lo, dip_hi = int(0.45 * n), int(0.55 * n)
        flux[dip_lo:dip_hi] -= 0.01

        std_full = float(np.std(flux))
        mad_est = _estimate_flux_err_oot(flux)

        assert std_full > 0.002, "std must be inflated by the dip"
        assert abs(mad_est - 0.001) < 0.0003, (
            f"MAD estimate should track the 0.001 baseline σ, got {mad_est}")

    def test_returns_positive_on_flat_curve(self):
        import numpy as np
        from triceratops_fpp import _estimate_flux_err_oot

        flux = 1.0 + np.random.default_rng(0).normal(0, 0.002, size=500)
        est = _estimate_flux_err_oot(flux)
        assert est > 0
        assert abs(est - 0.002) < 0.0004


class TestFppVerdict:
    def test_validated(self):
        from triceratops_fpp import fpp_verdict
        assert fpp_verdict(0.01, nfpp=0.0005) == "validated"

    def test_candidate(self):
        from triceratops_fpp import fpp_verdict
        assert fpp_verdict(0.3, nfpp=0.0) == "candidate"

    def test_likely_fp(self):
        from triceratops_fpp import fpp_verdict
        assert fpp_verdict(0.7, nfpp=0.0) == "likely_fp"
