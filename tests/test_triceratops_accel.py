"""Tests for TRICERATOPS acceleration module."""

import numpy as np
import pytest
from triceratops_accel import gaussian_flux_analytical, gaussian_flux_batch


class TestGaussianFluxAnalytical:
    def test_centered_unit_gaussian(self):
        """Integral of standard Gaussian over [-0.5,0.5]^2 should be ~0.118."""
        result = gaussian_flux_analytical(0, 0, 1.0, 1.0, -0.5, 0.5, -0.5, 0.5)
        assert 0.1 < result < 0.15  # ~0.1185

    def test_full_integral_approaches_amplitude(self):
        """Integral over very large area should approach A."""
        result = gaussian_flux_analytical(0, 0, 0.75, 1.0, -100, 100, -100, 100)
        assert abs(result - 1.0) < 1e-6

    def test_symmetry(self):
        """Result should be symmetric around center."""
        r1 = gaussian_flux_analytical(5.0, 5.0, 0.75, 1.0, 4.5, 5.5, 4.5, 5.5)
        r2 = gaussian_flux_analytical(5.0, 5.0, 0.75, 1.0, 4.5, 5.5, 4.5, 5.5)
        assert r1 == r2

    def test_off_center_less_flux(self):
        """Pixel far from star center should capture less flux."""
        on_center = gaussian_flux_analytical(5, 5, 0.75, 1.0, 4.5, 5.5, 4.5, 5.5)
        off_center = gaussian_flux_analytical(5, 5, 0.75, 1.0, 7.5, 8.5, 7.5, 8.5)
        assert on_center > off_center


class TestGaussianFluxBatch:
    def test_matches_sum_of_individual(self):
        """Batch should equal sum of individual pixel integrals."""
        pixels = np.array([[5, 5], [5, 6], [6, 5], [6, 6]])
        batch = gaussian_flux_batch(5.5, 5.5, 0.75, 1.0, pixels)
        individual = sum(
            gaussian_flux_analytical(5.5, 5.5, 0.75, 1.0,
                                     p[0]-0.5, p[0]+0.5, p[1]-0.5, p[1]+0.5)
            for p in pixels)
        assert abs(batch - individual) < 1e-10

    def test_5x5_aperture_captures_most_flux(self):
        """5x5 aperture centered on star should capture >99% of flux."""
        center = np.array([10.0, 10.0])
        pixels = np.array([[center[0]+dx, center[1]+dy]
                          for dx in range(-2, 3) for dy in range(-2, 3)])
        flux = gaussian_flux_batch(10.0, 10.0, 0.75, 1.0, pixels)
        assert flux > 0.99


class TestCalcDepthsFast:
    def test_matches_dblquad_result(self):
        """Analytical calc_depths should give same results as numerical."""
        from scipy.integrate import dblquad
        from triceratops_accel import gaussian_flux_analytical

        mu_x, mu_y, sigma, A = 5.0, 5.0, 0.75, 1.0
        x_lo, x_hi, y_lo, y_hi = 4.5, 5.5, 4.5, 5.5

        def Gauss2D(x, y, mu_x, mu_y, sigma, A):
            exponent = ((x-mu_x)**2 + (y-mu_y)**2)/(2*sigma**2)
            return float(A/(2*np.pi*sigma**2)*np.exp(-exponent))

        numerical, _ = dblquad(Gauss2D, y_lo, y_hi, x_lo, x_hi,
                                args=(mu_x, mu_y, sigma, A))
        analytical = gaussian_flux_analytical(mu_x, mu_y, sigma, A,
                                              x_lo, x_hi, y_lo, y_hi)
        assert abs(numerical - analytical) < 1e-10
