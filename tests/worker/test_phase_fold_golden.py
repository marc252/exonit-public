"""Golden tests for worker/phase_fold.py — item D5
(docs/reviews/2026-07-02-opus48-cleanup-review.md).

SCIENCE PATH: each site's OLD inline formula is frozen here verbatim, exactly
as it existed at that call site *before* this change swapped it for a call
into ``worker/phase_fold.phase_fold``. Every test below asserts
``np.array_equal`` (bit-identical, NOT ``np.allclose``) between the frozen
old formula and the new shared call — that equivalence is what justifies
replacing the call site. Do not "clean up" the old formulas below; they are
intentionally kept exactly as-is so the golden proof stays meaningful if
``phase_fold`` itself changes later.
"""
import numpy as np
import pytest
from pathlib import Path

from phase_fold import phase_fold

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_LC_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "exominer_native" / "tic_143143769_s0032.lc.fits"
)

SEED = 20260702


# --------------------------------------------------------------------- #
# Old formulas, frozen verbatim from each site.
# --------------------------------------------------------------------- #

def _old_cnn_vetting(time, t0, period):
    """worker/cnn.py::TransitClassifier._phase_fold_and_bin and
    worker/vetting.py::vet_candidate — identical formula at both sites."""
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    return phase


def _old_injection(time, t0, period):
    """worker/injection.py::inject_transit (phase computation only)."""
    phase = ((time - t0) % period) / period
    return phase


def _old_fpp_native(time, t0, period):
    """worker/fpp_native.py::_phase_fold_bin (phase computation only)."""
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    return phase


def _old_cross_source(time, t0, period):
    """worker/cross_source.py::fixed_period_box_fit (phase computation only)."""
    phase = ((time - t0) / period + 0.5) % 1.0 - 0.5
    return phase


# site_name -> (old formula, phase_fold kwargs that must reproduce it)
SITES = {
    "cnn_vetting": (_old_cnn_vetting, dict(fold_order="mod_div", center="mask")),
    "injection": (_old_injection, dict(fold_order="mod_div", center=None)),
    "fpp_native": (_old_fpp_native, dict(fold_order="div_mod", center="mask")),
    "cross_source": (_old_cross_source, dict(center="shift")),
}


def _random_time_array(rng, n=10_000):
    return rng.uniform(1000.0, 4000.0, n)


def _random_t0_period_pairs(rng, n=200):
    """(t0, period) draws spanning in-range and wildly out-of-range t0."""
    pairs = []
    for _ in range(n):
        period = rng.uniform(0.2, 200.0)
        kind = rng.randint(0, 4)
        if kind == 0:
            t0 = rng.uniform(1000.0, 4000.0)  # in range
        elif kind == 1:
            t0 = rng.uniform(-5000.0, 1000.0)  # before the array (negative time-t0)
        elif kind == 2:
            t0 = rng.uniform(4000.0, 50000.0)  # far after the array
        else:
            t0 = rng.uniform(1000.0, 4000.0) - rng.randint(1, 500) * period  # many periods before
        pairs.append((t0, period))
    return pairs


@pytest.mark.parametrize("site_name", sorted(SITES))
def test_golden_random_arrays(site_name):
    """10k-point random time array x 200 random (t0, period) draws, fixed seed."""
    old_fn, kwargs = SITES[site_name]
    rng = np.random.RandomState(SEED)
    time = _random_time_array(rng)
    pairs = _random_t0_period_pairs(rng)

    for t0, period in pairs:
        old = old_fn(time.copy(), t0, period)
        new = phase_fold(time.copy(), t0, period, **kwargs)
        assert np.array_equal(old, new, equal_nan=True), (
            f"{site_name}: mismatch at t0={t0!r} period={period!r}"
        )


@pytest.mark.skipif(not REAL_LC_FIXTURE.exists(), reason="real LC fixture not available")
@pytest.mark.parametrize("site_name", sorted(SITES))
def test_golden_real_lightcurve(site_name):
    """Same proof on a real cached SPOC light curve (read-only fixture,
    already committed for the ExoMiner-native test suite)."""
    from astropy.io import fits

    old_fn, kwargs = SITES[site_name]
    with fits.open(REAL_LC_FIXTURE) as hdul:
        time = np.asarray(hdul[1].data["TIME"], dtype=np.float64)

    t0_candidates = [
        float(np.nanmin(time)) + 1.0,  # in range, near start
        float(np.nanmedian(time)),  # in range, mid
        float(np.nanmax(time)) - 0.5,  # in range, near end
        float(np.nanmin(time)) - 137.0,  # before the array
        float(np.nanmax(time)) + 900.0,  # after the array
    ]
    periods = [0.53, 1.7, 3.14159, 9.9, 27.3, 42.0]

    for t0 in t0_candidates:
        for period in periods:
            old = old_fn(time.copy(), t0, period)
            new = phase_fold(time.copy(), t0, period, **kwargs)
            assert np.array_equal(old, new, equal_nan=True), (
                f"{site_name}: mismatch on real LC at t0={t0!r} period={period!r}"
            )


def test_period_le_zero_matches_unguarded_sites_nan_behaviour():
    """No mode guards against period<=0 — matches every un-guarded inline
    site (cross_source.py guards at the *call site*, not inside the
    formula, so it never reaches phase_fold with period<=0)."""
    time = np.array([1000.0, 1001.0, 1002.0])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = phase_fold(time, 500.0, 0.0, fold_order="mod_div", center=None)
    assert np.all(np.isnan(out))


def test_unknown_fold_order_raises():
    with pytest.raises(ValueError):
        phase_fold(np.array([1.0]), 0.0, 1.0, fold_order="nope")


def test_unknown_center_raises():
    with pytest.raises(ValueError):
        phase_fold(np.array([1.0]), 0.0, 1.0, center="nope")


def test_does_not_alias_input_time_array():
    """phase_fold must never mutate the caller's `time` array in place —
    matches every original inline site (all built `phase` from a fresh
    array produced by % / /, never wrote into `time` itself)."""
    time = np.array([1000.0, 1001.5, 1300.0])
    original = time.copy()
    _ = phase_fold(time, 500.0, 3.0, fold_order="mod_div", center="mask")
    assert np.array_equal(time, original)
