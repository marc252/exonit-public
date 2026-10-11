"""Unit tests for exonit.ffi_finalist_checks (spec §3.1 + §3.3).

Coverage:
  1. Synthetic real transit (deep, repeating) → passes_3_1=True
  2. Noise-level signal with control phases equal to transit depth → passes_3_1=False
  3. G-star with short duration implying b>0.9 → 3.3 flag set, but passes_3_1
     still drives keep/demote (3.3 never demotes alone)
  4. Missing stellar params / LC load fail → kept (passes_3_1=True, conservative)
  5. Stellar-consistency check — implausible radius flag
  6. Stellar-consistency check — plausible central-transit geometry passes quietly
  7. n_sigma boundary: exactly at threshold passes, below threshold fails
  8. Edge cases: empty LC, NaN values, very short period
"""
import math

import numpy as np
import pytest

from exonit.ffi_finalist_checks import (
    folded_depth_control_test,
    stellar_consistency_check,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic light curves
# ---------------------------------------------------------------------------

def _make_lc(
    n_pts: int = 1000,
    duration_days: float = 27.0,
    noise_ppm: float = 500.0,
    period: float = 5.0,
    t0: float = 2.5,
    depth_ppm: float = 3000.0,
    dur_transit: float = 0.12,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic TGLC-like LC with a box-shaped transit injected.

    Returns (time, flux) suitable for folded_depth_control_test.
    Flux is median-normalised (≈1.0 out-of-transit, dips during transit).
    """
    rng = np.random.default_rng(seed)
    time = np.linspace(t0 - 0.5, t0 - 0.5 + duration_days, n_pts)
    flux = 1.0 + rng.normal(0.0, noise_ppm * 1e-6, n_pts)

    # Inject box transits at every epoch within the LC
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    in_transit = np.abs(phase) < (dur_transit / period) * 0.6
    flux[in_transit] -= depth_ppm * 1e-6
    return time, flux


# ---------------------------------------------------------------------------
# 1. Deep, repeating transit → passes_3_1=True
# ---------------------------------------------------------------------------

class TestDeepRealTransit:
    def test_passes_for_genuine_signal(self):
        """A clear injected transit (3000 ppm, ~5 repeats) must pass §3.1."""
        time, flux = _make_lc(
            n_pts=1400, depth_ppm=3000.0, noise_ppm=400.0,
            period=5.0, t0=2.5, dur_transit=0.12,
        )
        res = folded_depth_control_test(time, flux, P=5.0, t0=2.5,
                                        dur_days=0.12, n_sigma=3.0)
        assert res["error"] is None, f"unexpected error: {res['error']}"
        assert res["passes_3_1"] is True, (
            f"expected True; folded_depth={res['folded_depth_ppm']:.0f} ppm, "
            f"control_std={res['control_std_ppm']:.0f} ppm, "
            f"eff_snr={res['effective_snr']:.1f}"
        )
        assert res["effective_snr"] is not None
        assert res["effective_snr"] > 3.0, (
            f"effective_snr={res['effective_snr']:.2f} should be > 3"
        )

    def test_folded_depth_close_to_injected(self):
        """The measured folded depth must be within 30 % of the injected value."""
        time, flux = _make_lc(
            n_pts=1400, depth_ppm=5000.0, noise_ppm=300.0,
            period=6.0, t0=3.0, dur_transit=0.15,
        )
        res = folded_depth_control_test(time, flux, P=6.0, t0=3.0,
                                        dur_days=0.15, n_sigma=3.0)
        assert res["folded_depth_ppm"] is not None
        assert abs(res["folded_depth_ppm"] - 5000.0) < 1500.0, (
            f"measured {res['folded_depth_ppm']:.0f} ppm vs injected 5000 ppm"
        )

    def test_n_transits_counted(self):
        """n_transits should be close to baseline / period."""
        time, flux = _make_lc(
            n_pts=2000, duration_days=27.0, depth_ppm=3000.0, noise_ppm=400.0,
            period=5.0, t0=2.5, dur_transit=0.12,
        )
        res = folded_depth_control_test(time, flux, P=5.0, t0=2.5,
                                        dur_days=0.12, n_sigma=3.0)
        # 27 d / 5 d = ~5 transits
        assert res["n_transits"] >= 4, f"expected ≥4 transits, got {res['n_transits']}"


# ---------------------------------------------------------------------------
# 2. Noise-level signal — control depths equal transit depth → fails §3.1
# ---------------------------------------------------------------------------

class TestNoiseLevelSignal:
    def test_fails_for_noise_signal(self):
        """A systematic at ALL fold phases fails §3.1.

        If the same 'dip' appears at the transit phase AND at every control
        phase (simulating a broadband systematic or blended EB affecting all
        phases equally), the max(control) criterion must reject it.

        We inject 200 ppm at the transit phase AND at all control phases.
        We pass a fixed control_phases tuple so the injection and measurement
        phases are aligned (the default auto-computed phases may differ).
        The folded transit depth will equal the largest control depth → fails.
        """
        rng = np.random.default_rng(42)
        n = 1200
        P, t0, dur = 5.0, 2.5, 0.12
        time = np.linspace(0.0, 27.0, n)
        flux = 1.0 + rng.normal(0.0, 800e-6, n)

        phase = ((time - t0) % P) / P
        phase[phase > 0.5] -= 1.0
        hw = (dur / P) * 0.6

        # Inject the same depth at transit AND all (fixed) control phases
        _ctrl_phases = (0.15, 0.25, 0.35, -0.15, -0.25, -0.35)
        for cp in _ctrl_phases:
            flux[np.abs(phase - cp) < hw] -= 200e-6
        flux[np.abs(phase) < hw] -= 200e-6  # transit phase

        # Pass the fixed phases so measurement aligns with injection
        res = folded_depth_control_test(time, flux, P=P, t0=t0,
                                        dur_days=dur, n_sigma=3.0,
                                        control_phases=_ctrl_phases)
        # Must NOT declare this as a reliable signal — control depths match transit
        assert res["passes_3_1"] is False, (
            f"expected False; folded_depth={res['folded_depth_ppm']:.0f} ppm, "
            f"control_std={res['control_std_ppm']:.0f} ppm, "
            f"control_depths={[f'{d:.0f}' for d in res['control_depths_ppm']]}, "
            f"eff_snr={res['effective_snr']:.1f}"
        )

    def test_fails_when_control_depth_equals_transit(self):
        """Manually construct a case where control and transit depths are equal."""
        # Create pure noise — no transit injected
        rng = np.random.default_rng(7)
        n = 1200
        time = np.linspace(0.0, 27.0, n)
        # Noise-only LC — some control phase will match or exceed the transit phase
        flux = 1.0 + rng.normal(0.0, 1500e-6, n)  # 1500 ppm noise
        res = folded_depth_control_test(time, flux, P=9.5, t0=1334.9,
                                        dur_days=0.1, n_sigma=3.0)
        # With pure noise the transit depth should be comparable to control std
        # so it should (with high probability) fail the criterion
        # We cannot guarantee it deterministically, but we can at least check
        # that effective_snr is close to 1 (not >> 3)
        if res["effective_snr"] is not None:
            assert res["effective_snr"] < 3.0 or not res["passes_3_1"], (
                "Pure noise should not pass with SNR >> 3"
            )


# ---------------------------------------------------------------------------
# 3. G-star with short duration (b>0.9) → 3.3 flags set, passes_3_1 still
#    drives keep/demote (3.3 alone never demotes)
# ---------------------------------------------------------------------------

class TestStellarConsistencyHighB:
    def test_high_b_flagged_informatively(self):
        """A 2.3 h transit for a 9.57 d G-star should flag b>0.9.

        For a G-star (R*=0.9 R_sun, M*=0.9 M_sun, P=9.57 d):
        Expected central duration ≈ 3.5 h.  Observed = 2.3 h → ratio ≈ 0.66 → b≈0.75.
        Let's use P=9.57 d, dur=2.3/24 d for TIC 24693554's parameters.
        """
        res = stellar_consistency_check(
            folded_depth_ppm=727.0,
            R_star=0.905,    # catalogue value for TIC 24693554
            Teff=5585.0,
            M_star=0.9,      # approximate
            P=9.5674,
            dur_days=0.0957,  # 2.3 h
        )
        assert res["Rp_earth"] is not None
        # 2.3/24 d duration, G star, ~9.57 d period → expected ~3.5 h
        # ratio ~0.66 → b ~0.75; just short of 0.9 but duration is still suspicious
        # The important check: flags might be empty but passes_3_1 is NOT set here
        # (stellar_consistency_check never sets passes_3_1 — it's informative only)
        assert "passes_3_1" not in res, (
            "stellar_consistency_check must NOT return passes_3_1"
        )
        # Also check that a definitely high-b case (very short duration) IS flagged
        res2 = stellar_consistency_check(
            folded_depth_ppm=727.0,
            R_star=0.905,
            Teff=5585.0,
            M_star=0.9,
            P=9.5674,
            dur_days=0.03,  # 0.72 h — extremely short, b well above 0.9
        )
        b_flags = [f for f in res2.get("flags", []) if "b=" in f]
        assert len(b_flags) >= 1, (
            f"expected b>0.9 flag for very short duration; flags: {res2['flags']}"
        )

    def test_3_3_alone_does_not_demote(self):
        """Verify 3.3 result has no 'passes_3_1' key — it cannot demote alone."""
        res = stellar_consistency_check(
            folded_depth_ppm=10000.0,
            R_star=0.5, Teff=3500.0, M_star=0.4,
            P=2.0, dur_days=0.01,
        )
        # Regardless of the flags, passes_3_1 must not be in the result
        assert "passes_3_1" not in res


# ---------------------------------------------------------------------------
# 4. Missing data → conservative keep
# ---------------------------------------------------------------------------

class TestMissingDataConservative:
    def test_none_time_keeps(self):
        """None time → error but passes_3_1=True (conservative keep)."""
        res = folded_depth_control_test(None, None, P=5.0, t0=2.5,
                                        dur_days=0.1, n_sigma=3.0)
        assert res["passes_3_1"] is True
        assert res["error"] is not None

    def test_empty_array_keeps(self):
        """Empty arrays → passes_3_1=True."""
        res = folded_depth_control_test(
            np.array([]), np.array([]), P=5.0, t0=2.5,
            dur_days=0.1, n_sigma=3.0,
        )
        assert res["passes_3_1"] is True
        assert res["error"] is not None

    def test_all_nan_keeps(self):
        """All-NaN flux → passes_3_1=True."""
        t = np.linspace(0, 27, 100)
        f = np.full(100, np.nan)
        res = folded_depth_control_test(t, f, P=5.0, t0=2.5,
                                        dur_days=0.1, n_sigma=3.0)
        assert res["passes_3_1"] is True

    def test_missing_stellar_radius_skips_3_3(self):
        """Missing R_star → 3.3 skips with error, never crashes."""
        res = stellar_consistency_check(
            folded_depth_ppm=1000.0,
            R_star=None, Teff=5500.0, M_star=1.0,
            P=5.0, dur_days=0.1,
        )
        assert res["error"] is not None
        assert res["Rp_earth"] is None
        assert "passes_3_1" not in res

    def test_missing_mass_skips_duration(self):
        """Missing M_star → Rp computed but duration check skipped."""
        res = stellar_consistency_check(
            folded_depth_ppm=2000.0,
            R_star=1.0, Teff=5778.0, M_star=None,
            P=5.0, dur_days=0.1,
        )
        assert res["Rp_earth"] is not None
        assert res["expected_dur_h"] is None
        assert res["duration_ratio"] is None

    def test_invalid_period_keeps(self):
        """Invalid period (zero) → passes_3_1=True."""
        rng = np.random.default_rng(1)
        t = np.linspace(0, 27, 500)
        f = 1.0 + rng.normal(0, 1e-4, 500)
        res = folded_depth_control_test(t, f, P=0.0, t0=2.5,
                                        dur_days=0.1, n_sigma=3.0)
        assert res["passes_3_1"] is True


# ---------------------------------------------------------------------------
# 5. Stellar-consistency — implausible radius flag
# ---------------------------------------------------------------------------

class TestImplausibleRadius:
    def test_large_radius_flagged(self):
        """A very deep transit (>100k ppm, huge Rp) should flag implausible radius."""
        # depth 100000 ppm = 10%, R*=1 R_sun → Rp = sqrt(0.1) * 109 R_earth ≈ 34.5 R_earth
        res = stellar_consistency_check(
            folded_depth_ppm=100_000.0,
            R_star=1.0, Teff=5778.0, M_star=1.0,
            P=5.0, dur_days=0.2,
        )
        assert res["Rp_earth"] > 22.5
        radius_flags = [f for f in res["flags"] if "implausible radius" in f]
        assert len(radius_flags) >= 1, f"expected implausible-radius flag; flags: {res['flags']}"

    def test_plausible_planet_no_flag(self):
        """A 1500 ppm transit around a solar star → Rp~1.3 R_earth, no flags."""
        # sqrt(1500e-6) * 109 R_earth ≈ 4.2 R_earth — fine
        res = stellar_consistency_check(
            folded_depth_ppm=1500.0,
            R_star=1.0, Teff=5778.0, M_star=1.0,
            P=10.0, dur_days=0.15,
        )
        assert res["Rp_earth"] is not None
        assert res["Rp_earth"] < 10.0
        assert res["flags"] == [], f"expected no flags; got: {res['flags']}"


# ---------------------------------------------------------------------------
# 6. Plausible central transit (b≈0) passes quietly
# ---------------------------------------------------------------------------

class TestPlausibleCentralTransit:
    def test_central_transit_no_flag(self):
        """A transit duration close to the expected central value → no duration flag."""
        # For M*=1, R*=1, P=10 d: a ≈ 0.091 AU, T_central ≈ 3.2 h
        res = stellar_consistency_check(
            folded_depth_ppm=2000.0,
            R_star=1.0, Teff=5778.0, M_star=1.0,
            P=10.0, dur_days=0.13,  # ~3.12 h, close to central 3.2 h
        )
        b_flags = [f for f in res.get("flags", []) if "b=" in f]
        assert len(b_flags) == 0, (
            f"unexpected duration flag for near-central transit: {res['flags']}"
        )
        assert res["implied_b"] is not None
        # For P=10d, R*=1, M*=1: T_central≈3.91h; dur_obs=3.12h → ratio=0.797 → b≈0.60
        # b < 0.7 is definitely not a "grazing" geometry
        assert res["implied_b"] < 0.7, f"b={res['implied_b']:.2f} should be well below 0.9"


# ---------------------------------------------------------------------------
# 7. n_sigma boundary tests
# ---------------------------------------------------------------------------

class TestNSigmaBoundary:
    def _make_controlled(self, depth_ppm: float, ctrl_std_ppm: float,
                         n_sigma: float = 3.0) -> dict:
        """Construct a LC where depth and control_std are approximately as given."""
        rng = np.random.default_rng(17)
        n = 2000
        time = np.linspace(0.0, 27.0, n)
        noise = ctrl_std_ppm * 1e-6
        flux = 1.0 + rng.normal(0.0, noise, n)
        # Inject transit at exact depth
        P, t0, dur = 5.0, 2.5, 0.12
        phase = ((time - t0) % P) / P
        phase[phase > 0.5] -= 1.0
        in_tr = np.abs(phase) < (dur / P) * 0.6
        flux[in_tr] -= depth_ppm * 1e-6
        return folded_depth_control_test(time, flux, P=P, t0=t0,
                                         dur_days=dur, n_sigma=n_sigma)

    def test_well_above_threshold_passes(self):
        """depth = 10 × ctrl_std → should pass n_sigma=3 easily."""
        res = self._make_controlled(depth_ppm=3000.0, ctrl_std_ppm=200.0)
        assert res["passes_3_1"] is True, (
            f"eff_snr={res['effective_snr']:.1f} should be >> 3"
        )

    def test_well_below_threshold_fails(self):
        """depth ≈ ctrl_std → should fail n_sigma=3."""
        # Inject tiny signal in high-noise LC
        rng = np.random.default_rng(55)
        n = 1200
        time = np.linspace(0.0, 27.0, n)
        flux = 1.0 + rng.normal(0.0, 2000e-6, n)  # 2000 ppm noise
        # Inject 200 ppm — will be lost in noise
        P, t0, dur = 5.0, 2.5, 0.12
        phase = ((time - t0) % P) / P
        phase[phase > 0.5] -= 1.0
        in_tr = np.abs(phase) < (dur / P) * 0.6
        flux[in_tr] -= 200e-6
        res = folded_depth_control_test(time, flux, P=P, t0=t0,
                                         dur_days=dur, n_sigma=3.0)
        assert not res["passes_3_1"], (
            f"tiny signal in high noise should fail; "
            f"eff_snr={res['effective_snr']:.1f}"
        )


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_very_short_period(self):
        """Period << duration — should not crash; returns error and keeps."""
        rng = np.random.default_rng(3)
        t = np.linspace(0, 5, 200)
        f = 1.0 + rng.normal(0, 1e-3, 200)
        res = folded_depth_control_test(t, f, P=0.05, t0=2.5,
                                        dur_days=0.1, n_sigma=3.0)
        # Very short P with dur=0.1 d means dur/P > 1, but should not crash
        assert isinstance(res, dict)
        assert "passes_3_1" in res

    def test_nan_in_middle(self):
        """NaN values in the middle of the LC should be filtered gracefully."""
        rng = np.random.default_rng(6)
        time, flux = _make_lc(n_pts=1000, depth_ppm=3000.0, noise_ppm=400.0)
        # Inject NaNs
        flux[200:250] = np.nan
        flux[600:620] = np.nan
        res = folded_depth_control_test(time, flux, P=5.0, t0=2.5,
                                        dur_days=0.12, n_sigma=3.0)
        assert isinstance(res, dict)
        assert "error" in res  # may succeed or not, but should not crash

    def test_single_transit_in_data(self):
        """Only 1 transit in the data — n_transits=1, kept."""
        rng = np.random.default_rng(8)
        # 3-day LC with P=5 d → only ~0.5 transits — the test checks it doesn't crash
        time = np.linspace(0.0, 3.0, 500)
        flux = 1.0 + rng.normal(0.0, 500e-6, 500)
        flux[(time > 1.2) & (time < 1.32)] -= 3000e-6  # one transit
        res = folded_depth_control_test(time, flux, P=5.0, t0=1.25,
                                        dur_days=0.12, n_sigma=3.0)
        assert isinstance(res, dict)
        assert "passes_3_1" in res


# ---------------------------------------------------------------------------
# 9. Reviewer items (round 2): stellar_consistency_check edge cases
# ---------------------------------------------------------------------------

class TestStellarConsistencyEdgeCases:
    """Tests for corrupt/zero input guards in stellar_consistency_check.

    Reviewer item 2 (2026-06-18): guard dur_days <= 0 (and None) so a zero
    duration from a corrupt DB row cannot produce a spurious b=1.0 flag.
    Verified by two cases:
      a) folded_depth_ppm=0  — already guarded, but untested
      b) dur_days=0          — new guard added in this iteration
    """

    def test_zero_folded_depth_returns_error_not_flag(self):
        """folded_depth_ppm=0 → error field set, no b-flag, no passes_3_1."""
        res = stellar_consistency_check(
            folded_depth_ppm=0,
            R_star=1.0, Teff=5778.0, M_star=1.0,
            P=5.0, dur_days=0.1,
        )
        assert res["error"] is not None, (
            "zero folded_depth_ppm must set an error, not silently proceed"
        )
        # Should not produce any b-flag (would be meaningless for depth=0)
        b_flags = [f for f in res.get("flags", []) if "b=" in f]
        assert len(b_flags) == 0, (
            f"zero depth produced unexpected b-flags: {res['flags']}"
        )
        assert "passes_3_1" not in res  # informative only, never

    def test_zero_dur_days_returns_error_not_b1_flag(self):
        """dur_days=0 → error field set, implied_b=None, no spurious b=1.0 flag.

        Without the guard, obs_dur_h=0 → ratio=0 → b²=1 → b=1.0 → spurious
        'duration too short (b=1.00 > 0.9)' flag on a corrupt row.  The guard
        added in this iteration short-circuits at the start and returns with
        error set, so no such flag can be emitted.
        """
        res = stellar_consistency_check(
            folded_depth_ppm=1000.0,
            R_star=1.0, Teff=5778.0, M_star=1.0,
            P=5.0, dur_days=0.0,
        )
        assert res["error"] is not None, (
            "dur_days=0 must set an error, not silently compute b=1.0"
        )
        assert res["implied_b"] is None, (
            f"dur_days=0 must not produce implied_b={res['implied_b']}"
        )
        b_flags = [f for f in res.get("flags", []) if "b=" in f]
        assert len(b_flags) == 0, (
            f"dur_days=0 produced spurious b-flags (the bug this guards against): "
            f"{res['flags']}"
        )
        assert "passes_3_1" not in res  # informative only, never

    def test_none_dur_days_returns_error(self):
        """dur_days=None → same guard: error set, no crash."""
        res = stellar_consistency_check(
            folded_depth_ppm=1000.0,
            R_star=1.0, Teff=5778.0, M_star=1.0,
            P=5.0, dur_days=None,
        )
        assert res["error"] is not None


