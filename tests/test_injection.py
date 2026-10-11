"""Tests for injection-recovery module."""

import numpy as np
import pytest
from injection import radius_to_depth, inject_transit, run_injection_recovery


class TestRadiusToDepth:
    def test_jupiter_is_deep(self):
        """Jupiter-sized planet should produce deep transit."""
        depth = radius_to_depth(11.2)  # Jupiter = 11.2 R_earth
        assert depth > 0.005  # > 0.5%

    def test_earth_is_shallow(self):
        """Earth-sized planet should produce shallow transit."""
        depth = radius_to_depth(1.0)
        assert depth < 0.001  # < 0.1%

    def test_smaller_star_deeper_transit(self):
        """Same planet around smaller star = deeper transit."""
        depth_solar = radius_to_depth(2.0, stellar_radius_solar=1.0)
        depth_mdwarf = radius_to_depth(2.0, stellar_radius_solar=0.5)
        assert depth_mdwarf > depth_solar


class TestInjectTransit:
    def test_preserves_input(self):
        flux = np.ones(1000)
        original = flux.copy()
        inject_transit(np.linspace(0, 30, 1000), flux, 10.0, 5.0, 0.01, 0.2)
        np.testing.assert_array_equal(flux, original)

    def test_reduces_flux(self):
        time = np.linspace(0, 30, 1000)
        flux = np.ones(1000)
        result = inject_transit(time, flux, 10.0, 5.0, 0.01, 0.2)
        assert np.min(result) < 1.0



class TestTransitDurationDays:
    """Audit item #8: replace heuristic 0.1·(radius/2) with physical T14."""

    def test_earth_around_sun_matches_known_duration(self):
        """Earth around Sun at 1 yr: T14 ≈ 13 h = 0.54 d for b=0."""
        from injection import transit_duration_days
        t14 = transit_duration_days(period_days=365.25,
                                    stellar_radius_solar=1.0,
                                    stellar_mass_solar=1.0,
                                    planet_radius_earth=1.0)
        assert 0.48 < t14 < 0.60, f"expected ~0.54 d, got {t14}"

    def test_hot_jupiter_matches_hd209458b(self):
        """HD 209458 b (M*=1.15, R*=1.19, P=3.52 d): T14 ≈ 0.13 d."""
        from injection import transit_duration_days
        t14 = transit_duration_days(period_days=3.52,
                                    stellar_radius_solar=1.19,
                                    stellar_mass_solar=1.15,
                                    planet_radius_earth=15.5)
        assert 0.10 < t14 < 0.17, f"expected ~0.13 d, got {t14}"

    def test_depends_on_period_not_planet_radius(self):
        """T14 must scale with P and R*, not with Rp (small-Rp limit)."""
        from injection import transit_duration_days
        t14_earth = transit_duration_days(period_days=5.0, planet_radius_earth=1.0)
        t14_super = transit_duration_days(period_days=5.0, planet_radius_earth=3.0)
        # Difference should be small (order k), not a factor of 3
        assert abs(t14_earth - t14_super) / t14_earth < 0.15

    def test_impact_parameter_shortens_transit(self):
        from injection import transit_duration_days
        central = transit_duration_days(period_days=5.0, impact_parameter=0.0)
        grazing = transit_duration_days(period_days=5.0, impact_parameter=0.9)
        assert grazing < central

    def test_mass_defaults_to_main_sequence_from_radius(self):
        """If stellar_mass omitted, fall back to MS relation (M ≈ R^1.25)."""
        from injection import transit_duration_days
        t14_explicit = transit_duration_days(period_days=5.0,
                                             stellar_radius_solar=1.0,
                                             stellar_mass_solar=1.0)
        t14_default = transit_duration_days(period_days=5.0,
                                            stellar_radius_solar=1.0)
        assert abs(t14_explicit - t14_default) < 1e-6


class TestInjectionDefaults:
    """Audit item #3: default n_injections_per_cell must be ≥10 so a
    Bernoulli completeness per cell has error bars tighter than ~15 pp."""

    def test_default_n_injections_is_at_least_10(self):
        import inspect
        from injection import run_injection_recovery
        sig = inspect.signature(run_injection_recovery)
        default = sig.parameters["n_injections_per_cell"].default
        assert default >= 10, (
            f"default n_injections_per_cell must be >=10 "
            f"(Christiansen+ 2015), got {default}")

    def test_injection_bls_grid_defaults_are_explicit(self):
        import inspect
        from injection import run_injection_recovery
        sig = inspect.signature(run_injection_recovery)
        assert sig.parameters["bls_n_periods"].default == 5000
        assert sig.parameters["bls_n_bins"].default == 150
        assert sig.parameters["bls_n_dur_steps"].default == 10


class TestRunInjectionRecovery:
    def test_returns_completeness_map(self):
        """Should return a completeness map with correct shape."""
        np.random.seed(42)
        time = np.linspace(0, 30, 3000)
        flux = np.ones(3000) + np.random.normal(0, 0.001, 3000)

        result = run_injection_recovery(
            time, flux,
            radius_grid=np.array([2.0, 5.0]),
            period_grid=np.array([3.0, 8.0]),
            n_injections_per_cell=2,
            snr_threshold=5.0)

        assert len(result["completeness_map"]) == 2
        assert len(result["completeness_map"][0]) == 2
        assert len(result["details"]) == 4  # 2x2 grid

    def test_large_planets_more_detectable(self):
        """Larger planets should have higher detection probability."""
        np.random.seed(42)
        time = np.linspace(0, 60, 10000)
        flux = np.ones(10000) + np.random.normal(0, 0.001, 10000)

        result = run_injection_recovery(
            time, flux,
            radius_grid=np.array([1.0, 10.0]),
            period_grid=np.array([5.0]),
            n_injections_per_cell=3,
            snr_threshold=5.0)

        small_prob = result["completeness_map"][0][0]
        large_prob = result["completeness_map"][1][0]
        assert large_prob >= small_prob
