"""PTP/PEB companion-parity fix (DeepSeekReview_18092026.md, item C2).

Findings from reading the actual installed upstream package
(triceratops==1.0.20, triceratops/marginal_likelihoods.py:lnZ_PTP/lnZ_PEB —
not just the review's description of it), documented here because they
changed the shape of the fix from what C2's prose literally asked for:

  * Upstream's PTP/PEB scenarios keep the TARGET star's M_s/R_s/Teff (and its
    already-computed u1/u2 limb-darkening coefficients) for the transit
    geometry (a_cm, ks, Ptra, collision check, `sample_rp`) — exactly what
    worker/fpp_native.py already did before this change. C2's framing ("the
    semi-major axis, ks, Ptra... therefore use the target star instead of the
    companion host") does not match upstream; this suite does NOT test for
    companion-conditioned a_Rs/ks, because upstream does not do that either.
  * What upstream DOES condition on the sampled companion
    (`masses_comp`/`qs_comp`) is the per-Monte-Carlo-draw flux-dilution ratio
    fed into the TTP/TEB likelihood as `companion_fr` — one value PER DRAW
    (`fluxratios_comp[mask]`), not a single scalar collapsed across draws.
    `_lnZ_PTP`/`_lnZ_PEB` used to pass `float(np.median(fluxratios_comp))`
    instead — this suite pins the fix (full per-sample array).
  * Upstream calls the likelihood with `companion_is_host=False` for both
    PTP and PEB (the target is modeled as the transit host; the companion is
    third light) — the pre-fix code passed `companion_is_host=True`, which
    flips the dilution formula in `_batch_transit_lnL`/`_batch_eb_transit_lnL`
    (F_dilute = 1/F_comp instead of F_comp) and is the single largest
    contributor to the FPP delta measured in progress/measure-fpp-ptp-peb.md.
  * Upstream reuses the TARGET's own u1/u2 (computed once from the target's
    Teff/logg, passed straight through) rather than a companion-based LDC —
    the pre-fix `_lnZ_PTP` recomputed a mean-companion-Teff LDC instead; this
    suite pins the fix (u1/u2 pass through unchanged).
  * `_lnZ_TEB` had a second, independent bug: it hardcoded its dilution array
    to `np.zeros(N)`, silently discarding whatever `companion_fr` its caller
    passed — so PEB's third-light term was always zero even before this
    change reached PTP/PEB's own call sites. Pinned separately below.
  * Upstream additionally folds a `lnprior_companion` term (bound-companion
    probability from parallax + delta-mag + the Moe & Di Stefano mass-ratio
    prior, `lnprior_bound_TP`/`lnprior_bound_EB`, ~200 lines) into Z for
    PTP/PEB. That term was NOT ported by THIS change (out of scope here) —
    it was ported separately in a later commit
    (`fix(fpp): port TRICERATOPS lnprior_companion...`), see
    tests/test_fpp_lnprior_companion.py for its coverage.

Scalar-target scenarios (TP, EB, EBx2P — companion_fr=0.0 default) must stay
bit-identical to the fix-fpp-seed commit: none of the above touches their
code path (same RNG draw order, same math).
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

import fpp_native

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Pinned to the commit SHA, not the branch name: `fix-fpp-seed` was deleted
# after it merged into main, so `git show fix-fpp-seed:...` errors on any
# checkout made after the cleanup (review finding on a345343). The commit
# object itself survives as long as it is reachable from history (it is --
# main's ancestry includes it), so pinning the SHA is stable across branch
# deletions in a way the branch name never was.
_SEED_COMMIT = "5aaf49c"


@pytest.fixture(scope="module")
def seed_commit_fpp_native(tmp_path_factory):
    """Load worker/fpp_native.py AS IT WAS at the seed-fix commit (5aaf49c,
    the commit the fix-fpp-seed branch pointed at before it was deleted;
    this worktree is built on top of it), before the PTP/PEB fix, under a
    distinct module name, so scalar-target scenario outputs can be diffed
    bit-for-bit against the current code without reinstalling a worktree.

    Skips cleanly (rather than erroring) if the commit object isn't
    reachable -- e.g. a shallow clone that doesn't include this far back in
    history -- since that's an environment limitation, not a code bug.
    """
    check = subprocess.run(
        ["git", "-C", _REPO_ROOT, "cat-file", "-e", f"{_SEED_COMMIT}^{{commit}}"],
        capture_output=True, text=True)
    if check.returncode != 0:
        pytest.skip(
            f"seed commit {_SEED_COMMIT} not reachable in this checkout "
            f"(shallow clone?): {check.stderr.strip()}")

    src = subprocess.check_output(
        ["git", "-C", _REPO_ROOT, "show", f"{_SEED_COMMIT}:worker/fpp_native.py"],
        text=True)
    tmp_dir = tmp_path_factory.mktemp("fpp_native_seed")
    path = tmp_dir / "fpp_native_seed_ref.py"
    path.write_text(src)

    spec = importlib.util.spec_from_file_location(
        "fpp_native_seed_ref", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fpp_native_seed_ref"] = mod
    spec.loader.exec_module(mod)
    mod.disable_gpu()
    yield mod
    del sys.modules["fpp_native_seed_ref"]


@pytest.fixture(autouse=True)
def _cpu_only():
    """CPU-only dispatch in both modules -- GPU backends are host-dependent
    and out of scope for this parity check (task explicitly says CPU-only
    for the measurement pass too)."""
    fpp_native.disable_gpu()


def _synthetic_lc(n=300, seed=3):
    rng = np.random.RandomState(seed)
    time = np.linspace(0.0, 20.0, n)
    flux = np.ones(n) + rng.normal(0, 0.0006, n)
    phase = ((time - 1.1 + 3.3 / 2) % 3.3) - 3.3 / 2
    flux[np.abs(phase) < 0.04] -= 0.0025
    return np.ascontiguousarray(time), np.ascontiguousarray(flux)


TARGET = dict(P_orb=3.3, M_s=0.95, R_s=0.98, Teff=5600.0, N=500, u1=0.42, u2=0.24)


# ============================================================
# Scalar-target scenarios: bit-identical to the seed commit
# ============================================================

class TestScalarTargetScenariosUnchanged:

    def test_lnZ_TTP_bit_identical_to_seed_commit(self, seed_commit_fpp_native):
        time, flux = _synthetic_lc()
        sigma = 0.0006
        old = seed_commit_fpp_native._lnZ_TTP(
            time, flux, sigma, **TARGET, rng=np.random.RandomState(11))
        new = fpp_native._lnZ_TTP(
            time, flux, sigma, **TARGET, rng=np.random.RandomState(11))
        assert new == old

    def test_lnZ_TEB_bit_identical_to_seed_commit(self, seed_commit_fpp_native):
        time, flux = _synthetic_lc()
        sigma = 0.0006
        old_eb, old_twin = seed_commit_fpp_native._lnZ_TEB(
            time, flux, sigma, **TARGET, rng=np.random.RandomState(22))
        new_eb, new_twin = fpp_native._lnZ_TEB(
            time, flux, sigma, **TARGET, rng=np.random.RandomState(22))
        assert new_eb == old_eb
        assert new_twin == old_twin


# ============================================================
# PTP/PEB: companion arrays actually reach the likelihood
# ============================================================

class TestPTPCompanionWiring:

    def test_companion_fr_is_full_per_sample_array_not_a_scalar_median(
            self, monkeypatch):
        captured = {}

        def spy_lnZ_TTP(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return -1.0

        monkeypatch.setattr(fpp_native, "_lnZ_TTP", spy_lnZ_TTP)
        time, flux = _synthetic_lc()
        fpp_native._lnZ_PTP(
            time, flux, 0.0006, TARGET["P_orb"], TARGET["M_s"], TARGET["R_s"],
            TARGET["Teff"], TARGET["N"], TARGET["u1"], TARGET["u2"], plx=5.0,
            rng=np.random.RandomState(33))

        comp_fr = captured["kwargs"]["companion_fr"]
        assert isinstance(comp_fr, np.ndarray)
        assert comp_fr.shape == (TARGET["N"],)
        # A per-draw array sampled from a continuous mass-ratio prior must
        # not collapse to (near-)one repeated value -- that was exactly the
        # `float(np.median(...))` bug.
        assert len(np.unique(np.round(comp_fr, 8))) > 10

    def test_companion_is_host_is_false_matching_upstream(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            fpp_native, "_lnZ_TTP",
            lambda *a, **k: captured.update(k) or -1.0)
        time, flux = _synthetic_lc()
        fpp_native._lnZ_PTP(
            time, flux, 0.0006, TARGET["P_orb"], TARGET["M_s"], TARGET["R_s"],
            TARGET["Teff"], TARGET["N"], TARGET["u1"], TARGET["u2"], plx=5.0,
            rng=np.random.RandomState(33))
        assert captured["companion_is_host"] is False

    def test_target_ldc_passed_through_unmodified(self, monkeypatch):
        """The pre-fix code recomputed a companion-Teff LDC here; upstream
        reuses the target's own u1/u2 unchanged."""
        captured = {}

        def spy(*args, **kwargs):
            captured["args"] = args
            return -1.0

        monkeypatch.setattr(fpp_native, "_lnZ_TTP", spy)
        time, flux = _synthetic_lc()
        fpp_native._lnZ_PTP(
            time, flux, 0.0006, TARGET["P_orb"], TARGET["M_s"], TARGET["R_s"],
            TARGET["Teff"], TARGET["N"], TARGET["u1"], TARGET["u2"], plx=5.0,
            rng=np.random.RandomState(33))
        # positional args: time, flux, sigma, P_orb, M_s, R_s, Teff, N, u1, u2
        assert captured["args"][4] == TARGET["M_s"]
        assert captured["args"][5] == TARGET["R_s"]
        assert captured["args"][6] == TARGET["Teff"]
        assert captured["args"][8] == TARGET["u1"]
        assert captured["args"][9] == TARGET["u2"]


class TestPEBCompanionWiring:

    def test_companion_fr_is_full_per_sample_array_and_not_hardcoded_zero(
            self, monkeypatch):
        captured = {}

        def spy_lnZ_TEB(*args, **kwargs):
            captured["kwargs"] = kwargs
            return (-1.0, -1.0)

        monkeypatch.setattr(fpp_native, "_lnZ_TEB", spy_lnZ_TEB)
        time, flux = _synthetic_lc()
        fpp_native._lnZ_PEB(
            time, flux, 0.0006, TARGET["P_orb"], TARGET["M_s"], TARGET["R_s"],
            TARGET["Teff"], TARGET["N"], TARGET["u1"], TARGET["u2"], plx=5.0,
            rng=np.random.RandomState(44))

        comp_fr = captured["kwargs"]["companion_fr"]
        assert isinstance(comp_fr, np.ndarray)
        assert comp_fr.shape == (TARGET["N"],)
        assert np.all(comp_fr > 0.0), (
            "a real companion flux ratio must be > 0 for every draw "
            "(q_min is bounded away from 0 in _sample_q_companion)"
        )
        assert captured["kwargs"]["companion_is_host"] is False

    def test_lnZ_TEB_no_longer_hardcodes_dilution_to_zero(self):
        """The independent _lnZ_TEB bug: comp_arr = np.zeros(N) regardless of
        the companion_fr argument. With a non-trivial per-sample fluxratio
        array, the EB dilution the kernel sees must actually vary.

        N=3000 (rather than TARGET's 500) so at least one draw reliably
        survives the transit/collision mask for both the primary-EB and
        twin-EB branches -- this is a control-flow probe, not a numeric
        parity check, so the larger N costs nothing but wall-clock.
        """
        time, flux = _synthetic_lc()
        N = 3000
        rng = np.random.RandomState(5)
        fr = rng.uniform(0.01, 0.4, size=N)
        captured = {}
        real_batch = fpp_native._batch_eb_transit_lnL

        def spy_batch(*args, **kwargs):
            captured.setdefault("companion_fluxratio", []).append(
                np.array(args[-2]))
            return real_batch(*args, **kwargs)

        original = fpp_native._batch_eb_transit_lnL
        fpp_native._batch_eb_transit_lnL = spy_batch
        try:
            target = dict(TARGET)
            target["N"] = N
            fpp_native._lnZ_TEB(
                time, flux, 0.0006, **target, companion_fr=fr,
                companion_is_host=False, rng=np.random.RandomState(6))
        finally:
            fpp_native._batch_eb_transit_lnL = original

        assert "companion_fluxratio" in captured, (
            "no draw survived the transit/collision mask -- "
            "increase N or relax the synthetic light curve")
        assert any(np.any(a > 0.0) for a in captured["companion_fluxratio"])


# ============================================================
# End-to-end: calc_fpp_native still returns a sane, finite result
# ============================================================

def _mock_target_star():
    return ([{
        "ID": "888888888", "ra": 50.0, "dec": 10.0, "Tmag": 11.0,
        "mass": 0.95, "rad": 0.98, "Teff": 5600.0, "plx": 6.0,
    }], 18.0)


@pytest.fixture
def mocked_fpp_env(monkeypatch):
    monkeypatch.setattr(fpp_native, "_query_tic_stars",
                         lambda *a, **k: _mock_target_star())
    monkeypatch.setattr(fpp_native, "_load_trilegal_stars",
                         lambda *a, **k: [])


def test_calc_fpp_native_end_to_end_finite_fpp_and_nfpp(mocked_fpp_env):
    time, flux = _synthetic_lc(n=600, seed=9)
    r = fpp_native.calc_fpp_native(
        888888888, 3.3, 1.1, duration=0.1, depth_ppm=2500.0,
        sectors=np.array([1]), time=time, flux=flux, N=200, force_cpu=True)
    assert r is not None
    assert np.isfinite(r["fpp"])
    assert np.isfinite(r["nfpp"])
    assert 0.0 <= r["fpp"] <= 1.0
    assert 0.0 <= r["nfpp"] <= 1.0
