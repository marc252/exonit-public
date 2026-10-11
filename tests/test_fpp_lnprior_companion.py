"""lnprior_companion port (PTP/PEB Monte-Carlo prior weighting).

worker/fpp_native.py's PTP/PEB marginal likelihoods (`_lnZ_PTP`/`_lnZ_PEB`)
folded a physically-sampled companion into the flux-dilution term
(78fa512) but never implemented upstream's `lnprior_companion` — the
per-draw log probability, from the Moe & Di Stefano (2017) bound-companion
rate (triceratops/priors.py:lnprior_bound_TP / lnprior_bound_EB), that the
drawn companion is actually gravitationally bound to the target. Upstream
folds this term into the SAME Monte-Carlo average as the likelihood itself
(`np.mean(np.exp(lnL + lnprior_companion + 600))`,
marginal_likelihoods.py:lnZ_PTP/lnZ_PEB) — it is not a separate additive
constant on lnZ, and it is per-draw (indexed by the same companion draw
that sets the dilution ratio), not a single scalar.

Scope (see the handoff for the full derivation): this term applies ONLY to
PTP/PEB in this port. Upstream's TTP/TEB have no such term at all (they are
literally separate functions, lnZ_TTP/lnZ_TEB, with no `lnprior_companion`
in their bodies). This port's NTP/NEB reuse `_lnZ_TTP`/`_lnZ_TEB` with the
default `lnprior_companion=0.0` -- upstream's NTP/NEB use a DIFFERENT,
unrelated prior (`lnprior_background`) that neither this port nor this
change implements; that remains a documented residual difference.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

import fpp_native
from triceratops.priors import lnprior_bound_TP, lnprior_bound_EB

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def main_fpp_native(tmp_path_factory):
    """Load worker/fpp_native.py AS IT WAS on `main` -- i.e. WITHOUT the
    lnprior_companion port -- under a distinct module name, so PTP/PEB
    outputs can be diffed against the current code without reinstalling a
    worktree. `main` already includes the seed fix (5aaf49c) and the
    PTP/PEB dilution fix (78fa512); it is exactly the pre-this-change
    baseline.
    """
    src = subprocess.check_output(
        ["git", "-C", _REPO_ROOT, "show", "main:worker/fpp_native.py"],
        text=True)
    tmp_dir = tmp_path_factory.mktemp("fpp_native_main")
    path = tmp_dir / "fpp_native_main_ref.py"
    path.write_text(src)

    spec = importlib.util.spec_from_file_location(
        "fpp_native_main_ref", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fpp_native_main_ref"] = mod
    spec.loader.exec_module(mod)
    mod.disable_gpu()
    yield mod
    del sys.modules["fpp_native_main_ref"]


@pytest.fixture(autouse=True)
def _cpu_only():
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
# _lnprior_bound_TP / _lnprior_bound_EB against the actual upstream
# package (triceratops==1.0.20, installed in .venv-server) -- this is
# stronger than a hand transcription of the ~100-line piecewise formula,
# since it also catches transcription bugs the hand-derivation could share.
# ============================================================

class TestLnpriorBoundAgainstUpstream:

    @pytest.mark.parametrize("M_s,plx", [
        (1.0, 10.0),    # M_s >= 1.0 branch, well-measured parallax
        (0.6, 5.0),     # M_s < 1.0 branch (f_act = 0.65*f_comp + 0.35*f_comp*M_act)
        (1.5, 2.0),     # M_s >= 1.0 branch, distant/low-plx target
    ])
    def test_lnprior_bound_TP_matches_upstream(self, M_s, plx):
        delta_mags_abs = np.array([0.05, 0.5, 1.5, 3.0, 6.0])
        upstream = lnprior_bound_TP(
            M_s, plx, delta_mags_abs, np.array([2.2]), np.array([1.0]))
        port = fpp_native._lnprior_bound_TP(M_s, plx, delta_mags_abs)
        # The port's physical constants (Msun/Rsun/G_cgs/au) are the
        # file's existing low-precision stand-ins, not astropy's exact
        # CODATA values (a pre-existing convention throughout this module,
        # unrelated to this change) -- match to 1e-4 relative, not bit-exact.
        np.testing.assert_allclose(port, upstream, rtol=2e-4)

    @pytest.mark.parametrize("M_s,plx", [
        (1.0, 10.0),
        (0.6, 5.0),
        (1.5, 2.0),
    ])
    def test_lnprior_bound_EB_matches_upstream(self, M_s, plx):
        delta_mags_abs = np.array([0.05, 0.5, 1.5, 3.0, 6.0])
        upstream = lnprior_bound_EB(
            M_s, plx, delta_mags_abs, np.array([2.2]), np.array([1.0]))
        port = fpp_native._lnprior_bound_EB(M_s, plx, delta_mags_abs)
        np.testing.assert_allclose(port, upstream, rtol=2e-4)

    def test_lnprior_bound_never_positive_after_clip(self):
        """_lnprior_companion clips to <=0 (a proper log probability) --
        pin that the RAW _lnprior_bound_TP/EB outputs (before that clip)
        stay finite and sane for a normal target, matching upstream's own
        range (upstream does not clip inside lnprior_bound_TP/EB itself,
        only at the call site -- see _lnprior_companion)."""
        delta_mags_abs = np.array([0.1, 1.0, 2.0, 4.0, 7.0])
        tp = fpp_native._lnprior_bound_TP(1.0, 8.0, delta_mags_abs)
        eb = fpp_native._lnprior_bound_EB(1.0, 8.0, delta_mags_abs)
        assert np.all(np.isfinite(tp))
        assert np.all(np.isfinite(eb))
        # EB's bound-companion rate (Porb > 10d) upper-bounds TP's
        # (Porb > 2500d) for the same separation -- more short-period
        # companions are allowed in, so EB's prior must never be smaller.
        assert np.all(eb >= tp - 1e-9)


class TestLnpriorCompanionCallSite:

    def test_clips_positive_to_zero_and_vetoes_brighter_companion(self):
        """Mirrors upstream's post-hoc mask in lnZ_PTP/lnZ_PEB:
        lnprior_companion[lnprior_companion > 0] = 0 (proper log prob) and
        lnprior_companion[delta_mags > 0] = -inf (a companion brighter than
        the target is vetoed, using the SIGNED delta_mag)."""
        # fluxratios_comp > 0.5 <=> companion brighter than target
        # (delta_mags = 2.5*log10(fr/(1-fr)) > 0).
        fr = np.array([0.5000001, 0.5, 0.4, 0.1, 0.01])
        out = fpp_native._lnprior_companion(1.0, 8.0, fr, "TP")
        assert out[0] == -np.inf, "brighter-than-target companion must be vetoed"
        assert np.all(out <= 0.0), "a log probability must never exceed 0"
        assert np.all(np.isfinite(out[1:]))

    def test_TP_vs_EB_scenario_selection(self):
        """EB's period floor (10d) admits more companions than TP's
        (2500d) at the same separation -- the EB branch must therefore
        never be more restrictive than TP for identical inputs."""
        fr = np.full(20, 0.05)
        tp = fpp_native._lnprior_companion(1.0, 8.0, fr.copy(), "TP")
        eb = fpp_native._lnprior_companion(1.0, 8.0, fr.copy(), "EB")
        assert np.all(eb >= tp - 1e-9)


# ============================================================
# Scalar-target scenarios (TTP/TEB): must stay bit-identical to `main`.
# This change only threads a NEW keyword (lnprior_companion, default
# 0.0 -- a no-op additive term) through _lnZ_TTP/_lnZ_TEB; it draws no new
# random numbers and does not touch the TTP/TEB call sites in
# calc_fpp_native, so scalar-target output must be untouched.
# ============================================================

class TestScalarTargetScenariosUnchanged:

    def test_lnZ_TTP_bit_identical_to_main(self, main_fpp_native):
        time, flux = _synthetic_lc()
        sigma = 0.0006
        old = main_fpp_native._lnZ_TTP(
            time, flux, sigma, **TARGET, rng=np.random.RandomState(11))
        new = fpp_native._lnZ_TTP(
            time, flux, sigma, **TARGET, rng=np.random.RandomState(11))
        assert new == old

    def test_lnZ_TEB_bit_identical_to_main(self, main_fpp_native):
        time, flux = _synthetic_lc()
        sigma = 0.0006
        old_eb, old_twin = main_fpp_native._lnZ_TEB(
            time, flux, sigma, **TARGET, rng=np.random.RandomState(22))
        new_eb, new_twin = fpp_native._lnZ_TEB(
            time, flux, sigma, **TARGET, rng=np.random.RandomState(22))
        assert new_eb == old_eb
        assert new_twin == old_twin


# ============================================================
# PTP/PEB: lnZ changes ONLY by the prior term -- with the prior
# monkeypatched to a no-op (0.0 for every draw), the result must be
# bit-identical to `main` (same RNG draw order: no new draws are consumed
# to compute lnprior_companion, it is a deterministic function of the
# ALREADY-drawn qs_comp/fluxratios_comp).
# ============================================================

class TestPTPPEBPriorIsTheOnlyDelta:

    def test_lnZ_PTP_equals_main_when_prior_neutralized(
            self, main_fpp_native, monkeypatch):
        time, flux = _synthetic_lc()
        sigma = 0.0006
        monkeypatch.setattr(
            fpp_native, "_lnprior_companion",
            lambda M_s, plx, fluxratios_comp, scenario: np.zeros_like(
                fluxratios_comp))

        old = main_fpp_native._lnZ_PTP(
            time, flux, sigma, TARGET["P_orb"], TARGET["M_s"], TARGET["R_s"],
            TARGET["Teff"], TARGET["N"], TARGET["u1"], TARGET["u2"], plx=6.0,
            rng=np.random.RandomState(33))
        new = fpp_native._lnZ_PTP(
            time, flux, sigma, TARGET["P_orb"], TARGET["M_s"], TARGET["R_s"],
            TARGET["Teff"], TARGET["N"], TARGET["u1"], TARGET["u2"], plx=6.0,
            rng=np.random.RandomState(33))
        assert new == old

    def test_lnZ_PEB_equals_main_when_prior_neutralized(
            self, main_fpp_native, monkeypatch):
        time, flux = _synthetic_lc()
        sigma = 0.0006
        monkeypatch.setattr(
            fpp_native, "_lnprior_companion",
            lambda M_s, plx, fluxratios_comp, scenario: np.zeros_like(
                fluxratios_comp))

        old_eb, old_twin = main_fpp_native._lnZ_PEB(
            time, flux, sigma, TARGET["P_orb"], TARGET["M_s"], TARGET["R_s"],
            TARGET["Teff"], TARGET["N"], TARGET["u1"], TARGET["u2"], plx=6.0,
            rng=np.random.RandomState(44))
        new_eb, new_twin = fpp_native._lnZ_PEB(
            time, flux, sigma, TARGET["P_orb"], TARGET["M_s"], TARGET["R_s"],
            TARGET["Teff"], TARGET["N"], TARGET["u1"], TARGET["u2"], plx=6.0,
            rng=np.random.RandomState(44))
        assert new_eb == old_eb
        assert new_twin == old_twin

    def test_lnZ_PTP_actually_changes_with_real_prior(self):
        """Sanity: with the real (non-neutralized) prior, PTP's lnZ must
        actually differ from the dilution-only value -- otherwise this
        whole port would be a no-op wired to nothing."""
        time, flux = _synthetic_lc()
        sigma = 0.0006
        with_prior = fpp_native._lnZ_PTP(
            time, flux, sigma, TARGET["P_orb"], TARGET["M_s"], TARGET["R_s"],
            TARGET["Teff"], TARGET["N"], TARGET["u1"], TARGET["u2"], plx=6.0,
            rng=np.random.RandomState(33))

        # Reproduce _lnZ_PTP's own draw sequence manually, but WITHOUT the
        # prior term, using the SAME rng object threaded through both the
        # qs_comp draw and the _lnZ_TTP call (exactly like _lnZ_PTP does)
        # so every OTHER random draw lines up bit-for-bit -- only the
        # prior term should differ.
        import fpp_native as fn
        rng2 = np.random.RandomState(33)
        qs_comp = fn._sample_q_companion(TARGET["N"], TARGET["M_s"], rng=rng2)
        masses_comp = qs_comp * TARGET["M_s"]
        fluxratios_comp = fn.flux_relation(masses_comp) / (
            fn.flux_relation(masses_comp)
            + fn.flux_relation(np.array([TARGET["M_s"]])))
        no_prior = fn._lnZ_TTP(
            time, flux, sigma, TARGET["P_orb"], TARGET["M_s"], TARGET["R_s"],
            TARGET["Teff"], TARGET["N"], TARGET["u1"], TARGET["u2"],
            companion_fr=fluxratios_comp, companion_is_host=False,
            rng=rng2)
        assert np.isfinite(with_prior) and np.isfinite(no_prior)
        assert with_prior != no_prior
