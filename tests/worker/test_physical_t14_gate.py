"""Unit tests for the physical T14 upper bound rejector.

Audit 2026-04-17 (Transit search Baix): BLS's ``duration_max_frac=0.15``
is unphysical at long periods. Production DB 2026-04-21 captured 15+
candidates at P∈[30,50]d with duration=0.15·P, 15-20× the physical T14
for a Jupiter-sized planet around a Sun-like star. These are always
systematics/noise, never real planets.

Fix: ``physical_t14_max(P, R*, M*)`` using injection's
``transit_duration_days`` with R_Jup, b=0, circular — the maximum
physically possible T14 at this period. Worker's multi-planet loop
rejects any candidate with T14 > 3 × T14_max.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "worker") not in sys.path:
    sys.path.append(str(ROOT / "worker"))

from worker.worker import physical_t14_max  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Test 1: T14 scales as P^(1/3) (Kepler's third law → a ~ P^2/3 → T14/P ~ P^-2/3)
# ─────────────────────────────────────────────────────────────────────
def test_t14_scales_with_period_cube_root():
    t1 = physical_t14_max(1.0, stellar_radius=1.0, stellar_mass=1.0)
    t10 = physical_t14_max(10.0, stellar_radius=1.0, stellar_mass=1.0)
    t100 = physical_t14_max(100.0, stellar_radius=1.0, stellar_mass=1.0)
    # Ratio t_P / t_1 ≈ P^(1/3)
    assert 2.0 < t10 / t1 < 2.5, \
        f"t10/t1 = {t10/t1:.2f}, expected ~2.15 (10^(1/3))"
    assert 4.4 < t100 / t1 < 4.8, \
        f"t100/t1 = {t100/t1:.2f}, expected ~4.64 (100^(1/3))"


# ─────────────────────────────────────────────────────────────────────
# Test 2: long-period unphysical durations are clearly beyond T14_max
# ─────────────────────────────────────────────────────────────────────
def test_long_period_unphysical_durations_exceed_cap():
    # Production DB samples (TIC 17409453, 232540264, ...) had duration
    # ≈ 7.5d at P=50d with Sun-like host. Physical T14_max should be
    # ~0.4d, so the ratio 7.5/0.4 = 19× is the signal.
    t14 = physical_t14_max(50.0, stellar_radius=1.0, stellar_mass=1.0)
    assert t14 < 0.5, \
        f"T14_max at P=50d should be <0.5d for R*=1, got {t14:.3f}d"
    # Production duration of 7.5d is > 3×T14 (=1.5d) → rejected
    production_duration = 7.5
    assert production_duration > 3.0 * t14


# ─────────────────────────────────────────────────────────────────────
# Test 3: short-period transits are NOT falsely rejected
# ─────────────────────────────────────────────────────────────────────
def test_short_period_real_transits_pass():
    # Hot Jupiter at P=1.5d around Sun-like: expect T14 ~2-3h = 0.1d.
    # 3×T14_max is still ≥ 0.3d, so legit hot Jupiters pass.
    t14 = physical_t14_max(1.5, stellar_radius=1.0, stellar_mass=1.0)
    real_hj_duration = 0.1  # 2.4h — typical hot Jupiter
    assert real_hj_duration < 3.0 * t14, \
        f"Real hot Jupiter T14={real_hj_duration}d should be <{3*t14:.3f}d"


# ─────────────────────────────────────────────────────────────────────
# Test 4: M-dwarf hosts (small R*) — we OVERESTIMATE T14_max (safer)
# ─────────────────────────────────────────────────────────────────────
def test_missing_stellar_params_falls_back_solar():
    # When stellar_radius is None or 0, fallback to solar (1.0) to stay
    # conservative (bigger T14_max = more lenient reject threshold).
    # An M-dwarf (R*=0.3) would have T14 ≈ 1/3 of solar; using solar
    # ensures we don't accidentally reject real transits.
    t_fallback = physical_t14_max(10.0, stellar_radius=None, stellar_mass=None)
    t_solar = physical_t14_max(10.0, stellar_radius=1.0, stellar_mass=1.0)
    assert abs(t_fallback - t_solar) < 1e-6, (
        f"Missing stellar radius should fall back to solar: "
        f"got {t_fallback:.3f}, expected {t_solar:.3f}"
    )


# ─────────────────────────────────────────────────────────────────────
# Test 5: larger stellar radius increases T14_max (longer transits OK)
# ─────────────────────────────────────────────────────────────────────
def test_large_star_allows_longer_transit():
    t_sun = physical_t14_max(5.0, stellar_radius=1.0, stellar_mass=1.0)
    t_giant = physical_t14_max(5.0, stellar_radius=3.0, stellar_mass=1.0)
    # T14 scales linearly with R*, so 3× R* should give ~3× T14
    ratio = t_giant / t_sun
    assert 2.5 < ratio < 3.5, \
        f"T14 should scale with R*: ratio={ratio:.2f}, expected ~3.0"
