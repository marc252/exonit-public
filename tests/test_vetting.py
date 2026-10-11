"""Tests for statistical false positive vetting."""

import numpy as np
import pytest
from vetting import check_harmonics, check_odd_even, check_v_shape, vet_candidate


class TestCheckHarmonics:
    def test_detects_half_period_harmonic(self):
        candidates = [
            {"period_days": 10.0, "snr": 15.0, "candidate_num": 1},
            {"period_days": 5.0, "snr": 8.0, "candidate_num": 2},
        ]
        flags = check_harmonics(candidates[1], candidates)
        assert flags["is_harmonic"] is True
        assert flags["harmonic_of"] == 1

    def test_no_harmonic_for_unrelated_period(self):
        candidates = [
            {"period_days": 10.0, "snr": 15.0, "candidate_num": 1},
            {"period_days": 7.3, "snr": 8.0, "candidate_num": 2},
        ]
        flags = check_harmonics(candidates[1], candidates)
        assert flags["is_harmonic"] is False


class TestCheckOddEven:
    def test_consistent_depths_pass(self):
        time = np.linspace(0, 100, 10000)
        flux = np.ones(10000)
        for epoch in np.arange(5.0, 100.0, 10.0):
            mask = np.abs(time - epoch) < 0.15
            flux[mask] -= 0.005
        result = check_odd_even(time, flux, period=10.0, t0=5.0, duration=0.3)
        assert result["odd_even_consistent"] == True

    def test_inconsistent_depths_fail(self):
        time = np.linspace(0, 100, 10000)
        flux = np.ones(10000)
        for i, epoch in enumerate(np.arange(5.0, 100.0, 10.0)):
            mask = np.abs(time - epoch) < 0.15
            depth = 0.01 if i % 2 == 0 else 0.003
            flux[mask] -= depth
        result = check_odd_even(time, flux, period=10.0, t0=5.0, duration=0.3)
        assert result["odd_even_consistent"] == False


class TestCheckVShape:
    def test_u_shape_passes(self):
        phase = np.linspace(-0.05, 0.05, 200)
        flux = np.ones(200)
        in_transit = np.abs(phase) < 0.02
        flux[in_transit] -= 0.005
        result = check_v_shape(phase, flux)
        assert result["is_v_shaped"] == False

    def test_v_shape_fails(self):
        phase = np.linspace(-0.05, 0.05, 200)
        # V-shape: edges retain >70% of center depth (grazing binary morphology)
        flux = 1.0 - 0.01 * (1 - 0.3 * np.abs(phase) / 0.05)
        result = check_v_shape(phase, flux)
        assert result["is_v_shaped"] == True


class TestVetCandidate:
    def test_returns_all_flags(self):
        time = np.linspace(0, 60, 5000)
        flux = np.ones(5000) + np.random.normal(0, 0.001, 5000)
        candidate = {"candidate_num": 1, "period_days": 10.0, "t0": 5.0,
                      "duration_days": 0.2, "depth_ppm": 500.0, "snr": 12.0}
        result = vet_candidate(candidate, [candidate], time, flux)
        assert "is_harmonic" in result
        assert "odd_even_consistent" in result
        assert "passed" in result
