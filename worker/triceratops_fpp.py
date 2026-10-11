"""TRICERATOPS FPP validation for transit candidates.

LEGACY EMERGENCY FALLBACK (2026-07-02 cleanup review): ``worker.py``
prefers ``fpp_native.calc_fpp_native`` (no triceratops/pytransit
dependency) and only imports this module if that import fails — see the
``try: from fpp_native import ... except ImportError: from
triceratops_fpp import ...`` fallback chain in worker.py. In production
fpp_native has been available on every worker since its introduction, so
this module is NOT exercised in production and is NOT covered by CI.
Keep it working in principle (do not delete), but treat any change here
as unverified until it is actually exercised end-to-end.

Wraps the triceratops library to compute False Positive Probability
for transit candidates. FPP thresholds (standard in literature):
  - FPP < 0.015 → "validated" (statistically confirmed planet)
  - FPP < 0.5   → "candidate" (worth further analysis)
  - FPP >= 0.5  → "likely_fp" (probable false positive)

Requires triceratops from master (bug fixes for log priors, EB mask,
NaN/underflow in marginal likelihood). Upstream calc_depths still uses
dblquad, so we keep our analytical erf monkey-patch for speed.
"""

import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

# Apply GPU/analytical acceleration patches on import
# Upstream calc_depths still uses dblquad (slow), our patch gives ~688x speedup
try:
    from triceratops_accel import apply_patches
    apply_patches()
except Exception:
    pass

# TRILEGAL (stev.oapd.inaf.it) serves a broken SSL chain; disable verification
# for that public-catalog query so calc_probs doesn't fail with a cert error.
# query_TRILEGAL is imported into triceratops.triceratops's namespace, so patch
# it there (where calc_probs resolves it).
try:
    import triceratops.triceratops as _tt

    _orig_query_trilegal = _tt.query_TRILEGAL

    def _query_trilegal_no_ssl(*args, **kwargs):
        kwargs.setdefault("verify_ssl", False)
        return _orig_query_trilegal(*args, **kwargs)

    _tt.query_TRILEGAL = _query_trilegal_no_ssl
except Exception:
    pass

# Thresholds + gate shared with fpp_native via fpp_thresholds.py (single
# source of truth — these used to be duplicated here and could drift).
from fpp_thresholds import (  # noqa: F401
    FPP_VALIDATED, FPP_CANDIDATE, NFPP_VALIDATED,
    should_run_triceratops,
)


def _create_target(tic_id, sectors, workdir="."):
    """Create a TRICERATOPS target object, reusing cached TRILEGAL if available."""
    from triceratops import triceratops
    trilegal_fname = os.path.join(workdir, f"{tic_id}_TRILEGAL.csv")
    if os.path.exists(trilegal_fname):
        return triceratops.target(ID=int(tic_id), sectors=sectors,
                                  trilegal_fname=trilegal_fname)
    return triceratops.target(ID=int(tic_id), sectors=sectors)


def _get_real_apertures(tic_id, sectors):
    """Fetch real SPOC aperture pixels for each sector.

    Uses triceratops.funcs.get_aperture to download actual pipeline
    apertures from MAST FITS files. Falls back to None (5x5 default)
    if any sector fails.

    Returns:
        List of aperture pixel arrays, or None on failure.
    """
    try:
        from triceratops.funcs import get_aperture
        apertures = []
        for sector in sectors:
            ap = get_aperture(int(tic_id), int(sector))
            apertures.append(ap)
        return apertures
    except Exception as e:
        print(f"  Could not fetch SPOC apertures: {e}; using 5x5 default")
        return None


def _phase_fold_bin(time, flux, period, t0, n_bins=1000):
    """Phase-fold and bin a light curve for TRICERATOPS.

    TRICERATOPS only needs the transit shape (depth, ingress/egress),
    not the full temporal light curve. Phase-folding stacks all transits
    into one high-SNR profile and reduces points from 60K+ to ~1000,
    making calc_probs feasible on any GPU.

    Args:
        time: Time array (BJD).
        flux: Flux array (normalized).
        period: Orbital period in days.
        t0: Transit epoch.
        n_bins: Number of phase bins (default 1000).

    Returns:
        (time_folded, flux_folded) arrays with n_bins points.
    """
    phase = ((time - t0) / period) % 1.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(phase, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    t_binned = []
    f_binned = []
    for i in range(n_bins):
        mask = bin_idx == i
        if np.sum(mask) > 0:
            t_binned.append(t0 + bin_centers[i] * period)
            f_binned.append(np.nanmedian(flux[mask]))

    return np.array(t_binned), np.array(f_binned)


def _estimate_flux_err_oot(flux_pf):
    # Giacalone+ 2020 sigma is baseline noise, not σ(flux|transit included).
    # MAD * 1.4826 matches σ of a Gaussian on the OOT majority while staying
    # robust to the transit dip (~<20% of phase bins) without needing to
    # identify its location.
    import numpy as _np
    mad = float(_np.median(_np.abs(flux_pf - _np.median(flux_pf))))
    return 1.4826 * mad


def calc_fpp(tic_id, period_days, t0, duration_days, depth_ppm,
             sectors, time, flux, N=100000, workdir="."):
    """Calculate False Positive Probability using TRICERATOPS.

    Phase-folds the light curve to ~1000 points before running
    calc_probs to avoid GPU memory issues with long light curves.

    Args:
        tic_id: TIC identifier.
        period_days: Orbital period in days.
        t0: Transit epoch (BJD - 2457000).
        duration_days: Transit duration in days.
        depth_ppm: Transit depth in ppm.
        sectors: numpy array of TESS sector numbers.
        time: Light curve time array.
        flux: Light curve flux array (normalized).
        N: Number of MC draws (default 100000, enough for FPP precision).
        workdir: Directory for TRILEGAL cache files.

    Returns:
        Dict with fpp, nfpp, verdict or None on failure.
    """
    try:
        target = _create_target(tic_id, sectors, workdir=workdir)

        # Try to get real SPOC apertures from MAST
        ap_pixels = _get_real_apertures(tic_id, sectors)

        # calc_depths must be called before calc_probs — it computes
        # the expected transit depth for each nearby star in the aperture
        target.calc_depths(tdepth=depth_ppm, all_ap_pixels=ap_pixels)

        # Phase-fold to reduce memory: 60K+ points → ~1000
        time_pf, flux_pf = _phase_fold_bin(time, flux, period_days, t0)
        # TRICERATOPS expects scalar sigma (used in marginal likelihood calc).
        # Use MAD-based OOT estimator so the transit dip doesn't inflate σ.
        flux_err = _estimate_flux_err_oot(flux_pf)

        target.calc_probs(
            time_pf, flux_pf, flux_err,
            P_orb=period_days,
            N=N,
            parallel=True,
            verbose=0)

        fpp = target.FPP
        nfpp = target.NFPP

        fpp_f = float(fpp)
        return {
            "fpp": fpp_f,
            "nfpp": float(nfpp),
            "prob_planet": max(0.0, min(1.0, 1.0 - fpp_f)),
            "verdict": fpp_verdict(fpp_f, nfpp=float(nfpp)),
        }
    except Exception as e:
        logger.warning(f"TRICERATOPS failed for TIC {tic_id}: "
                       f"{type(e).__name__}: {e}")
        return None


def fpp_verdict(fpp, nfpp=None, high_crowding=False):
    # Validation rule (Giacalone+ 2020): FPP<0.015 AND NFPP<0.001.
    # high_crowding downgrades validated to candidate because the TIC-only
    # background catalog undercounts BEB contributors past Tmag~16.
    if fpp >= FPP_CANDIDATE:
        return "likely_fp"
    if fpp < FPP_VALIDATED and nfpp is not None and nfpp < NFPP_VALIDATED:
        return "candidate" if high_crowding else "validated"
    return "candidate"
