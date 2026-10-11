"""Tests for the worker's wotan biweight detrending path.

The goal: verify that the new `detrend_with_wotan` helper preserves the
depth of injected transits (the single most important property to carry
over from lightkurve.flatten — Hippke+ 2019 warns the Savitzky-Golay
default suppresses ~15% of transit flux) while still removing slow
rotational/instrumental trends that caused `wrong_period` misses.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "worker"))

pytest.importorskip("wotan")

from wotan_detrend import detrend_with_wotan


def _make_light_curve(
    t_start=0.0,
    span_days=27.0,
    cadence_s=120.0,
    period=3.5,
    t0=1.2,
    duration=0.15,
    depth=0.003,
    slow_amp=0.0,
    slow_period=10.0,
    noise_ppm=100.0,
    seed=42,
):
    """Synthesize a 2-min cadence light curve with optional trend + transit."""
    rng = np.random.default_rng(seed)
    step = cadence_s / 86400.0
    time = np.arange(t_start, t_start + span_days, step)
    flux = np.ones_like(time)
    # Slow stellar/instrumental trend
    if slow_amp > 0:
        flux *= 1.0 + slow_amp * np.sin(2 * np.pi * time / slow_period)
    # Box-shaped transit signal (close enough for depth-preservation check)
    phase = ((time - t0) % period) / period
    half_dur = (duration / period) / 2.0
    in_transit = (phase < half_dur) | (phase > 1.0 - half_dur)
    flux[in_transit] *= 1.0 - depth
    # Gaussian noise
    flux += rng.normal(0.0, noise_ppm * 1e-6, size=flux.shape)
    return time, flux


def _measured_depth(time, flat_flux, period, t0, duration):
    phase = ((time - t0) % period) / period
    half_dur = (duration / period) / 2.0
    in_transit = (phase < half_dur) | (phase > 1.0 - half_dur)
    out = ~in_transit & np.isfinite(flat_flux)
    ins = in_transit & np.isfinite(flat_flux)
    if not out.any() or not ins.any():
        return 0.0
    return float(np.nanmedian(flat_flux[out]) - np.nanmedian(flat_flux[ins]))


def test_biweight_preserves_transit_depth_flat_lc():
    """With no trend at all, the recovered transit depth should match the
    injected depth within a few % — wotan must not cut into the transit."""
    time, flux = _make_light_curve(
        depth=0.003, duration=0.15, period=3.5, slow_amp=0.0, noise_ppm=50.0)
    flat = detrend_with_wotan(time, flux)
    measured = _measured_depth(time, flat, period=3.5, t0=1.2, duration=0.15)
    # Expect at least 90% of the injected depth (0.0027 vs 0.003)
    assert measured > 0.0027, f"depth suppressed: measured={measured}"


def test_biweight_removes_slow_rotational_trend():
    """A 1% sinusoidal trend at P_rot=10d should be largely removed,
    leaving residual RMS well below the raw amplitude."""
    time, flux = _make_light_curve(
        depth=0.0, slow_amp=0.01, slow_period=10.0, noise_ppm=50.0)
    flat = detrend_with_wotan(time, flux)
    assert np.nanstd(flat) < 0.002, f"trend not removed, std={np.nanstd(flat)}"


def test_biweight_preserves_depth_with_rotation_and_transit():
    """Combined: rotation + transit. Without a transit mask, biweight
    still suppresses ~25% of depth when a rotation sinusoid is present;
    the iterative-mask pass (covered by test_transit_mask_preserves_full_depth)
    is what restores the rest. Here we verify the *first-pass* baseline
    is good enough for BLS to find the dip."""
    time, flux = _make_light_curve(
        depth=0.003, duration=0.15, period=3.5,
        slow_amp=0.01, slow_period=10.0, noise_ppm=50.0)
    flat = detrend_with_wotan(time, flux)
    measured = _measured_depth(time, flat, period=3.5, t0=1.2, duration=0.15)
    # Expect at least 60% of the injected depth (first pass, no mask)
    assert measured > 0.0018, f"depth suppressed too much: measured={measured}"


def test_transit_mask_preserves_full_depth():
    """When we pass transit_mask=True points, detrending must not try to
    fit through the transit — full depth must be recovered even at the
    default short window_length."""
    time, flux = _make_light_curve(
        depth=0.005, duration=0.1, period=2.0, slow_amp=0.005, noise_ppm=30.0)
    # Build mask of in-transit samples
    phase = ((time - 1.2) % 2.0) / 2.0
    half_dur = (0.1 / 2.0) / 2.0
    in_transit = (phase < half_dur) | (phase > 1.0 - half_dur)
    flat = detrend_with_wotan(time, flux, transit_mask=in_transit)
    measured = _measured_depth(time, flat, period=2.0, t0=1.2, duration=0.1)
    # With an explicit mask, recovery should be within 5% of injected
    assert measured > 0.00475, f"mask didn't help: measured={measured}"


def test_window_length_tunable():
    """Callers must be able to widen the window for rotating stars."""
    time, flux = _make_light_curve(
        depth=0.0, slow_amp=0.02, slow_period=1.5, noise_ppm=50.0)
    # Default 0.5 d won't remove a 1.5 d sinusoid well; a larger window
    # (3x) is needed. Assert that tuning the kwarg changes behaviour.
    narrow = detrend_with_wotan(time, flux, window_length=0.5)
    wide = detrend_with_wotan(time, flux, window_length=2.0)
    # Both should reduce but with different residuals (they shouldn't be identical)
    assert not np.allclose(narrow, wide, equal_nan=True)
