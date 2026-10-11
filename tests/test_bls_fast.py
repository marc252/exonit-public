import numpy as np
import pytest
from bls_fast import bls_fast


def _make_synthetic_transit(n_points=5000, period=10.0, depth=0.01,
                            duration=0.2, t0=2.5, noise=0.001):
    time = np.linspace(0, 60, n_points)
    flux = np.ones(n_points) + np.random.normal(0, noise, n_points)
    phase = ((time - t0) % period) / period
    half_dur = (duration / period) / 2
    in_transit = (phase < half_dur) | (phase > 1.0 - half_dur)
    flux[in_transit] -= depth
    return time, flux


class TestBLSFast:
    def test_detects_known_period(self):
        time, flux = _make_synthetic_transit(period=10.0, depth=0.005)
        result = bls_fast(time, flux, period_min=1.0, period_max=30.0,
                          n_periods=5000, n_bins=200, n_dur_steps=10)
        assert abs(result["period"] - 10.0) < 0.5

    def test_returns_required_keys(self):
        time, flux = _make_synthetic_transit()
        result = bls_fast(time, flux, n_periods=1000)
        required = {"period", "power", "duration", "t0", "depth",
                     "depth_ppm", "snr", "periods", "power_spectrum"}
        assert required <= set(result.keys())

    def test_snr_positive_for_strong_signal(self):
        time, flux = _make_synthetic_transit(depth=0.01, noise=0.001)
        result = bls_fast(time, flux, n_periods=3000)
        assert result["snr"] > 5.0

    def test_depth_ppm_consistent(self):
        time, flux = _make_synthetic_transit()
        result = bls_fast(time, flux, n_periods=1000)
        assert abs(result["depth_ppm"] - result["depth"] * 1e6) < 1.0


class TestCanonicalSDE:
    """Audit item #4: SDE must use robust statistics (Kovács 2002).
    Old (peak-mean)/std lets a tall peak inflate std itself, suppressing
    secondary planets' SDE in multi-planet systems."""

    def test_robust_sde_beats_global_on_secondary_peak(self):
        """Two-peak spectrum: dominant peak's SDE is almost unchanged by
        the presence of a secondary peak (median+MAD is insensitive);
        the old (peak-mean)/std would be suppressed."""
        import numpy as np
        from bls_fast import _compute_sde

        rng = np.random.default_rng(0)
        n = 5000
        noise = rng.normal(1.0, 0.05, size=n)

        # Single-peak reference
        single = noise.copy()
        single[100] = 5.0
        sde_single = _compute_sde(single, peak_idx=100)

        # Two-peak spectrum — a secondary peak of height 3.0 elsewhere
        two = noise.copy()
        two[100] = 5.0
        two[3000] = 3.0
        sde_two = _compute_sde(two, peak_idx=100)

        # With robust stats the dominant peak's SDE should be stable
        # within <5% despite the secondary peak.
        ratio = sde_two / sde_single
        assert 0.95 < ratio < 1.05, (
            f"secondary peak moved dominant SDE from {sde_single:.2f} "
            f"to {sde_two:.2f} (ratio={ratio:.3f}); robust stats expected")

    def test_sde_zero_on_flat_spectrum(self):
        """Constant power spectrum → SDE = 0 (no signal)."""
        import numpy as np
        from bls_fast import _compute_sde
        flat = np.ones(100)
        assert _compute_sde(flat, peak_idx=50) == 0.0

    def test_sde_matches_canonical_formula_for_gaussian_noise(self):
        """Pure Gaussian noise + a spike of height μ+kσ → SDE ≈ k."""
        import numpy as np
        from bls_fast import _compute_sde
        rng = np.random.default_rng(7)
        spectrum = rng.normal(0.0, 1.0, size=20000)
        k = 8.0
        spectrum[0] = k
        sde = _compute_sde(spectrum, peak_idx=0)
        assert abs(sde - k) < 0.8, f"SDE={sde} should be near {k}"


class TestFrequencyGrid:
    """Audit item #9: periods must be sampled on a frequency grid
    (density ∝ 1/P²) so BLS's natural resolution ΔP ≈ P²/T is matched
    at every period — the old linspace oversampled long P and
    undersampled short P."""

    def test_first_and_last_periods_match_requested_bounds(self):
        import numpy as np
        from bls_fast import _frequency_period_grid
        periods = _frequency_period_grid(0.5, 50.0, 1000)
        assert abs(periods[0] - 0.5) < 1e-6
        assert abs(periods[-1] - 50.0) < 1e-6
        assert np.all(np.diff(periods) > 0)

    def test_more_density_at_short_periods(self):
        """The grid must pack more periods into [P_min, 2·P_min] than
        into [P_max/2, P_max] — that's the whole point of 1/P² sampling."""
        import numpy as np
        from bls_fast import _frequency_period_grid
        periods = _frequency_period_grid(0.5, 50.0, 1000)
        short_count = int(np.sum((periods >= 0.5) & (periods <= 1.0)))
        long_count = int(np.sum((periods >= 25.0) & (periods <= 50.0)))
        assert short_count > 2 * long_count, (
            f"short-P count {short_count} should dominate long-P "
            f"count {long_count} on a 1/P² grid")

    def test_linear_in_inverse_period(self):
        """Uniform spacing in 1/P (frequency) to machine epsilon."""
        import numpy as np
        from bls_fast import _frequency_period_grid
        periods = _frequency_period_grid(1.0, 20.0, 500)
        freqs = 1.0 / periods
        diffs = np.diff(freqs)
        assert np.max(np.abs(diffs - diffs[0])) < 1e-10
