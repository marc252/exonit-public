"""Tests for the stop-gap stellar-mass derivation helper.

Stop-gap before Berger+2023 catalog integration (#10). See
docs/runbooks/berger_2023_integration.md §1 for context.
"""
import math
import pytest

from exonit.server import derive_mass_from_logg_rad


class TestDeriveMassFromLoggRad:
    def test_sun(self):
        """At R=1, logg=4.438, should recover M=1 Msun exactly."""
        m = derive_mass_from_logg_rad(logg=4.438, rad=1.0)
        assert abs(m - 1.0) < 1e-6

    def test_red_giant(self):
        """RGB: R~10, logg~2.5 → M in the 1-2 Msun range."""
        m = derive_mass_from_logg_rad(logg=2.5, rad=10.0)
        assert 0.8 < m < 2.0  # 1.15 Msun from formula

    def test_a_type_dwarf(self):
        """A-type like TIC 319068154 (R=2.36, logg≈4.2) → ~3 Msun."""
        m = derive_mass_from_logg_rad(logg=4.2, rad=2.36)
        assert 2.5 < m < 4.0

    def test_missing_logg_returns_none(self):
        assert derive_mass_from_logg_rad(logg=None, rad=1.0) is None

    def test_missing_rad_returns_none(self):
        assert derive_mass_from_logg_rad(logg=4.4, rad=None) is None

    def test_both_missing_returns_none(self):
        assert derive_mass_from_logg_rad(logg=None, rad=None) is None

    def test_string_inputs_coerced(self):
        """Table rows from MAST can come through as strings; helper should
        coerce or bail cleanly."""
        m = derive_mass_from_logg_rad(logg="4.438", rad="1.0")
        assert abs(m - 1.0) < 1e-6

    def test_garbage_inputs_return_none(self):
        assert derive_mass_from_logg_rad(logg="nan", rad="also-nan") is None

    def test_produces_finite_value_for_observed_failing_tics(self):
        """Observed 4/100 failing TICs in baseline-100 (2026-04-21/22):
        319068154 (R=2.36, Teff=9500), 348835438 (R=3.98, Teff=4859),
        202428245 (R=3.37, Teff=5019). All had logg and rad present per
        TIC v8 header. Check we'd have derived a plausible mass."""
        # Typical early-A (T~9500): logg ~ 4.1
        assert 0.3 < derive_mass_from_logg_rad(4.1, 2.36) < 10
        # Evolved (R~4, Teff~5000): logg ~ 3.5
        assert 0.3 < derive_mass_from_logg_rad(3.5, 3.98) < 10
        assert 0.3 < derive_mass_from_logg_rad(3.5, 3.37) < 10
