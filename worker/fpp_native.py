"""Native FPP calculation — no triceratops/pytransit dependency.

Reimplements the False Positive Probability calculation from
Giacalone & Dressing (2020) / TRICERATOPS using only:
  numpy, scipy, numba, requests

Transit model uses numerical ring integration with quadratic
limb darkening (Mandel & Agol 2002 equivalent, accurate to <0.01%).

Background star handling:
  Uses observed TIC stars within 60 arcsec instead of TRILEGAL
  synthetic population model. TIC provides real measured stars but
  is incomplete below Tmag ~16-17. For dense fields (|b| < 10° or
  >15 TIC neighbors), results may underestimate background FP
  probability — flagged via high_crowding in output. Pre-downloaded
  TRILEGAL CSVs are used as supplementary background sources when
  available (see {TIC_ID}_TRILEGAL.csv).

FPP thresholds (standard in literature):
  - FPP < 0.015 → "validated"
  - FPP < 0.5   → "candidate"
  - FPP >= 0.5  → "likely_fp"
"""

import os
import sys
import math
import logging
import importlib
import numpy as np
from scipy.stats import beta as beta_dist, powerlaw as powerlaw_dist
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import erf

from numba_serial import numba_parallel_guard
from phase_fold import phase_fold

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Pure-python Monte Carlo is orders of magnitude slower — never silent.
    logging.getLogger("exonit").warning(
        "numba not available: FPP Monte Carlo will run in PURE PYTHON "
        "(orders of magnitude slower). Install numba in the worker venv.")
    # Create no-op decorators mimicking numba.njit
    class _FakeNumba:
        @staticmethod
        def njit(*args, **kwargs):
            if len(args) == 1 and callable(args[0]):
                return args[0]
            def decorator(func):
                return func
            return decorator
        @staticmethod
        def prange(*a, **kw):
            return range(*a)
    numba = _FakeNumba()

# ============================================================
# GPU dispatch globals and helpers
# ============================================================

_gpu_batch_transit = None
_gpu_batch_eb_transit = None
_fpp_gpu_backend = None
_fpp_gpu_info = None


def _init_fpp_gpu():
    """Detect best available FPP GPU backend. Called once.

    On Apple Silicon, prefer the fused ``fpp_metal_kernel`` (a custom
    mx.fast.metal_kernel: one GPU thread per draw, Kepler + 100-ring loop in
    registers). Benchmarked on M4 (2026-05-27, N up to 1e5, T=1000): ~4.5-5x
    FASTER than numba, lnZ parity ~4e-3 vs numba, EB secondary-eclipse reject
    masks identical. This replaces the old verdict.

    The LEGACY ``fpp_metal`` array path is still skipped when numba is present:
    it materialises ~500 (N,T) intermediates per batch and is memory-bandwidth-
    bound (~4x slower than numba) — the bench that produced the old "numba wins
    5x, skip Metal" rule measured THAT path, not the fused kernel.

    Worker architecture (worker.py): when the fused Metal kernel is active and
    gpu_enabled=true, FPP runs on the GPU and injection runs on the CPU
    concurrently (FPP is the 50-80% chunk; the GPU is best spent there).

    Override: EXONIT_FPP_FORCE_METAL=1 enables the legacy array path anyway.
    Diagnostic: EXONIT_FPP_GPU_BACKENDS=none disables every FPP GPU backend
    (useful on hosts where a second idle OpenCL context on the same iGPU
    collateral-damages the injection queue — see the "innocent" amdgpu TDR
    investigation on worker-default, 2026-04-19).
    """
    global _gpu_batch_transit, _gpu_batch_eb_transit, _fpp_gpu_backend, _fpp_gpu_info
    backends_env = os.environ.get("EXONIT_FPP_GPU_BACKENDS", "").strip().lower()
    if backends_env == "none":
        return
    force_metal = os.environ.get("EXONIT_FPP_FORCE_METAL", "").strip() in ("1", "true", "yes")
    for module_name, backend_name in [
        ("fpp_cuda", "cuda"),
        ("fpp_metal_kernel", "metal-kernel"),  # fused custom Metal kernel — ~5x FASTER
                                               # than numba on M4 (2026-05-27 bench); the
                                               # register-fused per-draw kernel avoids the
                                               # (N,T) materialisation that made the old
                                               # array path 4x slower. Preferred on Apple Si.
        ("fpp_metal", "metal"),                # legacy MLX array path — slower than numba
        ("fpp_opencl", "opencl"),
    ]:
        if backend_name == "metal" and HAS_NUMBA and not force_metal:
            # Skip the LEGACY array Metal path: numba beats it. (The fused
            # "metal-kernel" backend above is faster than numba and is NOT skipped.)
            continue
        try:
            mod = importlib.import_module(module_name)
            if mod.is_available():
                _gpu_batch_transit = mod.batch_transit_lnL
                _gpu_batch_eb_transit = mod.batch_eb_transit_lnL
                _fpp_gpu_backend = backend_name
                _fpp_gpu_info = mod.get_info()
                return
        except ImportError:
            continue


def _configure_fpp_cpu_cores(n_cores=None):
    """Set numba thread count for FPP computation.

    If n_cores is None, defaults to floor(0.75 * cpu_count) — the same
    75%-of-CPUs formula agent.py:207-223 already uses to honour CLAUDE.md
    rule 2 ("never use all of a worker's CPUs, we must leave 25% free"). The
    previous default of max(1, cpu_count - 2) only
    leaves 2 fixed cores free regardless of host size, which violates the
    25%-free rule on anything with >8 cores (87.5% used at 16 cores,
    93.75% at 32) — found in the 2026-07-02 cleanup review (B7). Explicit
    config values (``fpp_cpu_cores``) remain authoritative — this default
    only applies when the caller passes n_cores=None.

    The effective value is clamped to numba's thread pool size
    (``NUMBA_NUM_THREADS``, fixed at numba import time from the env
    var worker.py sets at boot from ``config.cpu_threads``). That cap
    is hard — ``numba.set_num_threads`` rejects anything above it with
    ``ValueError: The number of threads must be between 1 and N``.
    Before the clamp was added, a config with ``fpp_cpu_cores > cpu_threads``
    silently fell back to numba's default and logged a confusing
    "FPP CPU core config failed" warning.

    Returns the thread count actually applied, or ``None`` if numba is
    unavailable so the caller can log accurately.
    """
    if not HAS_NUMBA:
        return None
    total = os.cpu_count() or 4
    numba_cap = getattr(numba.config, "NUMBA_NUM_THREADS", total)
    if n_cores is None:
        n_cores = max(1, math.floor(0.75 * total))
    effective = max(1, min(n_cores, numba_cap, total))
    numba.set_num_threads(effective)
    return effective


def get_fpp_backend_info():
    """Return (backend_name, info_string) for UI display.

    Since 2026-05-27 the fused ``fpp_metal_kernel`` is the preferred backend
    on Apple Silicon (see ``_init_fpp_gpu`` — it replaced the old "skip Metal,
    numba wins" verdict, which only applied to the LEGACY ``fpp_metal`` array
    path). So on a Mac, reaching the ``cpu`` return here means the fused
    kernel failed to init, not that Metal was skipped by design.
    """
    if _fpp_gpu_backend:
        return _fpp_gpu_backend, _fpp_gpu_info
    cores = numba.config.NUMBA_NUM_THREADS if HAS_NUMBA else 1
    info = f"CPU ({cores} cores, numba)"
    # Annotate the reason when a GPU is present but intentionally skipped
    try:
        metal_available = importlib.import_module("fpp_metal").is_available()
    except Exception:
        metal_available = False
    if metal_available and HAS_NUMBA:
        info += (" [legacy fpp_metal array path skipped by design; fused "
                 "fpp_metal_kernel did NOT init — see startup warnings]")
    return "cpu", info


def disable_gpu():
    """Force FPP Monte Carlo evaluation onto CPU (numba).

    Intended for worker configs that set ``gpu_enabled=false`` — e.g.
    AMD Vega integrated GPUs where the OpenCL kernel can hit the
    amdgpu TDR on heavy FPP batches. ``_init_fpp_gpu`` runs at import
    time and may have set GPU globals; this clears them.
    """
    global _gpu_batch_transit, _gpu_batch_eb_transit, _fpp_gpu_backend, _fpp_gpu_info
    _gpu_batch_transit = None
    _gpu_batch_eb_transit = None
    _fpp_gpu_backend = None
    _fpp_gpu_info = None

# ============================================================
# Constants
# ============================================================
Msun = 1.989e33       # g
Rsun = 6.957e10       # cm
Rearth = 6.371e8      # cm
G_cgs = 6.674e-8      # cm³/(g·s²)
au = 1.496e13         # cm (astronomical unit) — used by _lnprior_bound_TP/EB
pi = np.pi
ln2pi = np.log(2 * pi)

# Thresholds live in fpp_thresholds.py (shared with legacy triceratops_fpp);
# re-exported here for backwards compatibility of existing imports.
from fpp_thresholds import (  # noqa: F401
    FPP_VALIDATED, FPP_CANDIDATE, NFPP_VALIDATED,
    should_run_triceratops,
)

# ============================================================
# 1. Transit model — numba-accelerated ring integration
# ============================================================

@numba.njit(cache=True)
def _kepler_eq(M, ecc, tol=1e-10):
    """Solve Kepler's equation M = E - ecc*sin(E) via Newton's method."""
    E = M
    for _ in range(30):
        dE = (E - ecc * math.sin(E) - M) / (1.0 - ecc * math.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    return E


@numba.njit(cache=True)
def _true_anomaly(M, ecc):
    """True anomaly from mean anomaly and eccentricity."""
    if ecc < 1e-10:
        return M
    E = _kepler_eq(M, ecc)
    cosE = math.cos(E)
    nu = math.atan2(math.sqrt(1 - ecc * ecc) * math.sin(E),
                     cosE - ecc)
    return nu


@numba.njit(cache=True)
def _compute_z_arr(time, t0, period, a_Rs, inc_rad, ecc, w_rad):
    """Projected separation z(t) in units of R_star."""
    n = len(time)
    z = np.empty(n)
    two_pi = 2.0 * math.pi
    for i in range(n):
        phase = (time[i] - t0) / period
        # Offset M so transit midpoint (f=ν+ω=0 → z minimum) is at t=t0
        M = two_pi * (phase - math.floor(phase)) - w_rad
        nu = _true_anomaly(M, ecc)
        r_factor = (1.0 - ecc * ecc) / (1.0 + ecc * math.cos(nu))
        # projected separation
        f = nu + w_rad
        sin_f = math.sin(f)
        cos_f = math.cos(f)
        cos_i = math.cos(inc_rad)
        z[i] = a_Rs * r_factor * math.sqrt(
            sin_f * sin_f + cos_i * cos_i * cos_f * cos_f)
    return z


@numba.njit(cache=True)
def _transit_flux_ring(z, p, u1, u2, n_rings=100):
    """Compute transit flux for a single (z, p) using ring integration.

    Integrates I(r) * blocked_fraction(r) * 2r dr over the stellar disk
    where I(r) = 1 - u1*(1-mu) - u2*(1-mu)^2, mu = sqrt(1-r^2).
    """
    if z >= 1.0 + p:
        return 1.0
    if p <= 0.0:
        return 1.0

    omega = 1.0 - u1 / 3.0 - u2 / 6.0
    if omega <= 0.0:
        omega = 1.0

    dr = 1.0 / n_rings
    blocked = 0.0

    for i in range(n_rings):
        r = (i + 0.5) * dr
        r2 = r * r
        mu = math.sqrt(1.0 - r2)
        one_minus_mu = 1.0 - mu
        I_r = 1.0 - u1 * one_minus_mu - u2 * one_minus_mu * one_minus_mu

        # Fraction of ring at radius r blocked by planet disk at distance z
        if z < 1e-12:
            # Planet centered on star
            frac = 1.0 if r <= p else 0.0
        elif r >= z + p:
            frac = 0.0
        elif r + p <= z:
            frac = 0.0
        elif r <= abs(z - p):
            frac = 1.0 if p > z else 0.0
        else:
            arg = (r2 + z * z - p * p) / (2.0 * r * z)
            if arg >= 1.0:
                frac = 0.0
            elif arg <= -1.0:
                frac = 1.0
            else:
                frac = math.acos(arg) / math.pi

        blocked += I_r * frac * 2.0 * r * dr

    return 1.0 - blocked / omega


@numba.njit(cache=True)
def _compute_transit_lc(time, k, t0, period, a_Rs, inc_rad, ecc, w_rad,
                         u1, u2):
    """Compute full transit light curve for one parameter set."""
    z_arr = _compute_z_arr(time, t0, period, a_Rs, inc_rad, ecc, w_rad)
    n = len(time)
    flux = np.empty(n)
    for i in range(n):
        flux[i] = _transit_flux_ring(z_arr[i], k, u1, u2)
    return flux


# ============================================================
# 2. Batch transit evaluation for MC (numba parallel)
# ============================================================

@numba.njit(parallel=True, cache=True)
def _batch_transit_lnL(time, flux, sigma, ks, P_orbs, incs_rad,
                        a_Rs_arr, eccs, ws_rad, u1, u2,
                        companion_fluxratio, companion_is_host):
    """Evaluate log-likelihood for N parameter sets in parallel.

    Returns array of lnL values (one per parameter set).
    """
    N = len(ks)
    T = len(time)
    lnL = np.full(N, -np.inf)
    lnsigma = math.log(sigma)

    for j in numba.prange(N):
        model = _compute_transit_lc(
            time, ks[j], 0.0, P_orbs[j], a_Rs_arr[j],
            incs_rad[j], eccs[j], ws_rad[j], u1, u2)

        # Apply flux dilution from companion/background star
        fr = companion_fluxratio[j]
        if fr > 0.0:
            if companion_is_host:
                F_dilute = 1.0 / (fr / (1.0 - fr))
            else:
                F_dilute = fr / (1.0 - fr)
            for t in range(T):
                model[t] = (model[t] + F_dilute) / (1.0 + F_dilute)

        chi2 = 0.0
        for t in range(T):
            diff = flux[t] - model[t]
            chi2 += diff * diff
        lnL[j] = -0.5 * ln2pi - lnsigma - 0.5 * chi2 / (sigma * sigma)

    return lnL


@numba.njit(parallel=True, cache=True)
def _batch_eb_transit_lnL(time, flux, sigma, ks, eb_fluxratios, P_orbs,
                           incs_rad, a_Rs_arr, eccs, ws_rad, u1, u2,
                           companion_fluxratio, companion_is_host):
    """Evaluate log-likelihood for N EB parameter sets.

    Includes secondary eclipse depth check (reject if too deep).
    Uses analytical secondary depth estimate: for total secondary,
    secdepth ≈ F_EB / (F_star + F_EB) = eb_fluxratio, adjusted for dilution.
    """
    N = len(ks)
    T = len(time)
    lnL = np.full(N, -np.inf)
    lnsigma = math.log(sigma)

    for j in numba.prange(N):
        k = ks[j]
        if abs(k - 1.0) < 1e-6:
            k *= 0.999

        # Primary eclipse model
        model = _compute_transit_lc(
            time, k, 0.0, P_orbs[j], a_Rs_arr[j],
            incs_rad[j], eccs[j], ws_rad[j], u1, u2)

        # EB flux dilution
        fr_eb = eb_fluxratios[j]
        F_EB = fr_eb / (1.0 - fr_eb) if fr_eb < 1.0 else 1e6
        fr_comp = companion_fluxratio[j]
        F_comp = fr_comp / (1.0 - fr_comp) if fr_comp > 0 else 0.0

        # Analytical secondary depth estimate (total secondary approximation)
        if companion_is_host:
            for t in range(T):
                model[t] = (model[t] + F_EB / max(F_comp, 1e-30)) / (
                    1.0 + F_EB / max(F_comp, 1e-30))
            F_dilute = 1.0 / (F_comp + F_EB)
            for t in range(T):
                model[t] = (model[t] + F_dilute) / (1.0 + F_dilute)
            # Secondary: companion occults EB
            secdepth = fr_eb / (1.0 + F_dilute)
        else:
            for t in range(T):
                model[t] = (model[t] + F_EB) / (1.0 + F_EB)
            F_dilute = F_comp / (1.0 + F_EB)
            for t in range(T):
                model[t] = (model[t] + F_dilute) / (1.0 + F_dilute)
            # Secondary: star occults EB (total if EB smaller than star)
            secdepth = fr_eb / (1.0 + F_dilute) if F_dilute > 0 else fr_eb

        # Reject if secondary eclipse is detectable
        if secdepth >= 1.5 * sigma:
            continue

        chi2 = 0.0
        for t in range(T):
            diff = flux[t] - model[t]
            chi2 += diff * diff
        lnL[j] = -0.5 * ln2pi - lnsigma - 0.5 * chi2 / (sigma * sigma)

    return lnL


# ============================================================
# 3. Stellar relations (mass → radius, Teff, flux)
# ============================================================

# Spline nodes from Torres (2010) for M > 0.63 Msun
_Mass_Torres = np.array([0.26, 0.47, 0.59, 0.69, 0.87, 0.98, 1.085,
                          1.4, 1.65, 2.0, 2.5, 3.0, 4.4, 15.0, 40.0])
_Rad_Torres = np.array([0.28, 0.47, 0.60, 0.72, 0.9, 1.05, 1.2, 1.55,
                         1.8, 2.1, 2.4, 2.6, 3.0, 6.2, 11.0])
_Teff_Torres = np.array([3170, 3520, 3840, 4410, 5150, 5560, 5940, 6650,
                          7300, 8180, 9790, 11400, 15200, 30000, 42000])

# Cool dwarf spline for M <= 0.63 Msun
_Mass_cdwrf = np.array([0.1, 0.135, 0.2, 0.35, 0.48, 0.58, 0.63])
_Rad_cdwrf = np.array([0.12, 0.165, 0.23, 0.36, 0.48, 0.585, 0.6])
_Teff_cdwrf = np.array([2800, 3000, 3200, 3400, 3600, 3800, 4000])

_spl_R_torres = InterpolatedUnivariateSpline(_Mass_Torres, _Rad_Torres)
_spl_T_torres = InterpolatedUnivariateSpline(_Mass_Torres, _Teff_Torres)
_spl_R_cdwrf = InterpolatedUnivariateSpline(_Mass_cdwrf, _Rad_cdwrf)
_spl_T_cdwrf = InterpolatedUnivariateSpline(_Mass_cdwrf, _Teff_cdwrf)

# TESS flux relation: log10(flux) vs mass
_Mass_flux = np.array([0.1, 0.15, 0.23, 0.4, 0.58, 0.7, 0.9, 1.15,
                        1.45, 2.2, 2.8])
_logflux_nodes = np.array([-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1,
                            1.5, 2])
_spl_flux = InterpolatedUnivariateSpline(_Mass_flux, _logflux_nodes)


def stellar_relations(masses):
    """Estimate radii and Teffs from masses (Solar units)."""
    masses = np.atleast_1d(masses).copy()
    R = np.empty_like(masses)
    T = np.empty_like(masses)
    hot = masses > 0.63
    cool = ~hot
    if np.any(hot):
        R[hot] = _spl_R_torres(masses[hot])
        T[hot] = _spl_T_torres(masses[hot])
    if np.any(cool):
        R[cool] = _spl_R_cdwrf(np.clip(masses[cool], 0.1, 0.63))
        T[cool] = _spl_T_cdwrf(np.clip(masses[cool], 0.1, 0.63))
    R = np.clip(R, 0.1, None)
    T = np.clip(T, 2800, None)
    return R, T


def flux_relation(masses):
    """Estimate relative TESS-band flux from mass."""
    return 10.0 ** _spl_flux(np.atleast_1d(masses))


# ============================================================
# 4. Limb darkening coefficients
# ============================================================

_ldc_data = None


def _load_ldc():
    """Load LDC table from bundled CSV or triceratops package."""
    global _ldc_data
    if _ldc_data is not None:
        return

    # Try local copy first, then triceratops package data
    candidates = [
        os.path.join(os.path.dirname(__file__), "ldc_tess.csv"),
    ]
    try:
        from pkg_resources import resource_filename
        candidates.append(resource_filename('triceratops', 'data/ldc_tess.csv'))
    except Exception:
        pass

    for path in candidates:
        if os.path.exists(path):
            from pandas import read_csv
            df = read_csv(path)
            _ldc_data = {
                'Z': np.array(df.Z, dtype=float),
                'Teff': np.array(df.Teff, dtype=int),
                'logg': np.array(df.logg, dtype=float),
                'u1': np.array(df.aLSM, dtype=float),
                'u2': np.array(df.bLSM, dtype=float),
            }
            return

    # Fallback: solar-like defaults
    _ldc_data = {'Z': np.array([0.0]), 'Teff': np.array([5777]),
                  'logg': np.array([4.44]),
                  'u1': np.array([0.40]), 'u2': np.array([0.26])}


def get_ldc(Teff, logg, Z=0.0):
    """Look up quadratic limb darkening coefficients for TESS."""
    _load_ldc()
    d = _ldc_data
    best_Z = d['Z'][np.argmin(np.abs(d['Z'] - Z))]
    best_T = d['Teff'][np.argmin(np.abs(d['Teff'] - Teff))]
    best_g = d['logg'][np.argmin(np.abs(d['logg'] - logg))]
    mask = (d['Z'] == best_Z) & (d['Teff'] == best_T) & (d['logg'] == best_g)
    if np.any(mask):
        return float(d['u1'][mask].flat[0]), float(d['u2'][mask].flat[0])
    return 0.40, 0.26  # solar default


# ============================================================
# 5. Prior sampling (Giacalone & Dressing 2020 / Moe & Di Stefano 2017)
# ============================================================

def _sample_rp(N, M_s, rng=np.random):
    """Sample planet radii from broken power law prior (Earth radii)."""
    x = rng.rand(N)
    M_arr = np.full(N, M_s)
    R_break1, R_break2, R_min, R_max = 3.0, 6.0, 0.5, 20.0

    if M_s > 0.45:
        p1, p2, p3 = 0.0, -4.0, -0.5
    else:
        p1, p2, p3 = 0.0, -7.0, -0.5

    A1 = R_break1**p1 / R_break1**p2
    A2 = R_break2**p2 / R_break2**p3
    I1 = (R_break1**(p1 + 1) - R_min**(p1 + 1)) / (p1 + 1)
    I2 = A1 * (R_break2**(p2 + 1) - R_break1**(p2 + 1)) / (p2 + 1)
    I3 = A2 * A1 * (R_max**(p3 + 1) - R_break2**(p3 + 1)) / (p3 + 1)
    Norm = 1.0 / (I1 + I2 + I3)

    m1 = x <= Norm * I1
    m2 = (x > Norm * I1) & (x <= Norm * (I1 + I2))
    m3 = (x > Norm * (I1 + I2))

    x[m1] = (x[m1] / Norm * (p1 + 1) + R_min**(p1 + 1))**(1 / (p1 + 1))
    x[m2] = ((x[m2] / Norm - I1) * (p2 + 1) / A1
             + R_break1**(p2 + 1))**(1 / (p2 + 1))
    x[m3] = ((x[m3] / Norm - I1 - I2) * (p3 + 1) / (A1 * A2)
             + R_break2**(p3 + 1))**(1 / (p3 + 1))
    return x


def _sample_inc(N, rng=np.random):
    """Sample inclinations from sin(i) prior (degrees)."""
    x = rng.rand(N)
    Norm = 1.0  # cos(0) - cos(90) = 1
    return np.arccos(1.0 - x / Norm) * 180.0 / pi


def _sample_ecc(N, planet=True, P_orb=10.0, rng=np.random):
    """Sample eccentricities from Beta (planets) or power law (binaries)."""
    if planet:
        return beta_dist.rvs(0.867, 3.030, size=N, random_state=rng)
    else:
        nu_plus_1 = 0.2 if P_orb <= 10 else 0.6
        return powerlaw_dist.rvs(nu_plus_1, size=N, random_state=rng)


def _sample_w(N, rng=np.random):
    """Sample argument of periastron uniformly (degrees)."""
    return rng.rand(N) * 360.0


def _sample_q(N, M_s, rng=np.random):
    """Sample mass ratios for short-period binaries (Moe & Di Stefano 2017)."""
    x = rng.rand(N)
    if M_s >= 1.0:
        q_min = 0.1
    elif M_s >= 0.3:
        q_min = 0.1 / M_s
    elif M_s > 0.1:
        q_min = 0.1 / M_s
    else:
        return np.ones(N)

    p1, p2 = 0.3, -0.5
    F_twin = 0.30
    A1 = (0.3**p1) / (0.3**p2) if q_min < 0.3 else 1.0

    if q_min < 0.3:
        A2 = 1 + (F_twin / (1 - F_twin)) * (
            (1.0**(p2 + 1) - 0.3**(p2 + 1)) / (p2 + 1)) / (
            (1.0**(p2 + 1) - 0.95**(p2 + 1)) / (p2 + 1))
        I1 = (0.3**(p1 + 1) - q_min**(p1 + 1)) / (p1 + 1)
        I2 = A1 * (0.95**(p2 + 1) - 0.3**(p2 + 1)) / (p2 + 1)
        I3 = A2 * A1 * (1.0**(p2 + 1) - 0.95**(p2 + 1)) / (p2 + 1)
        Norm = 1.0 / (I1 + I2 + I3)

        m1 = x <= Norm * I1
        m2 = (x > Norm * I1) & (x <= Norm * (I1 + I2))
        m3 = x > Norm * (I1 + I2)
        x[m1] = (x[m1] / Norm * (p1 + 1) + q_min**(p1 + 1))**(1 / (p1 + 1))
        x[m2] = ((x[m2] / Norm - I1) * (p2 + 1) / A1
                  + 0.3**(p2 + 1))**(1 / (p2 + 1))
        x[m3] = ((x[m3] / Norm - I1 - I2) * (p2 + 1) / (A1 * A2)
                  + 0.95**(p2 + 1))**(1 / (p2 + 1))
    else:
        A2 = 1 + (F_twin / (1 - F_twin)) * (
            (1.0**(p2 + 1) - q_min**(p2 + 1)) / (p2 + 1)) / (
            (1.0**(p2 + 1) - 0.95**(p2 + 1)) / (p2 + 1))
        I2 = (0.95**(p2 + 1) - q_min**(p2 + 1)) / (p2 + 1)
        I3 = A2 * (1.0**(p2 + 1) - 0.95**(p2 + 1)) / (p2 + 1)
        Norm = 1.0 / (I2 + I3)

        m2 = x <= Norm * I2
        m3 = x > Norm * I2
        x[m2] = (x[m2] / Norm * (p2 + 1) + q_min**(p2 + 1))**(1 / (p2 + 1))
        x[m3] = ((x[m3] / Norm - I2) * (p2 + 1) / A2
                  + 0.95**(p2 + 1))**(1 / (p2 + 1))

    return np.clip(x, q_min, 1.0)


def _sample_q_companion(N, M_s, rng=np.random):
    """Sample mass ratios for long-period companions."""
    x = rng.rand(N)
    q_min = max(0.1 / M_s, 0.1) if M_s > 0.1 else 1.0

    if M_s <= 0.1:
        return np.ones(N)

    p1, p2 = 0.3, -0.95
    F_twin = 0.05

    if q_min < 0.3:
        A1 = (0.3**p1) / (0.3**p2)
        A2 = 1 + (F_twin / (1 - F_twin)) * (
            (1.0**(p2 + 1) - 0.3**(p2 + 1)) / (p2 + 1)) / (
            (1.0**(p2 + 1) - 0.95**(p2 + 1)) / (p2 + 1))
        I1 = (0.3**(p1 + 1) - q_min**(p1 + 1)) / (p1 + 1)
        I2 = A1 * (0.95**(p2 + 1) - 0.3**(p2 + 1)) / (p2 + 1)
        I3 = A2 * A1 * (1.0**(p2 + 1) - 0.95**(p2 + 1)) / (p2 + 1)
        Norm = 1.0 / (I1 + I2 + I3)

        m1 = x <= Norm * I1
        m2 = (x > Norm * I1) & (x <= Norm * (I1 + I2))
        m3 = x > Norm * (I1 + I2)
        x[m1] = (x[m1] / Norm * (p1 + 1) + q_min**(p1 + 1))**(1 / (p1 + 1))
        x[m2] = ((x[m2] / Norm - I1) * (p2 + 1) / A1
                  + 0.3**(p2 + 1))**(1 / (p2 + 1))
        x[m3] = ((x[m3] / Norm - I1 - I2) * (p2 + 1) / (A1 * A2)
                  + 0.95**(p2 + 1))**(1 / (p2 + 1))
    else:
        A2 = 1 + (F_twin / (1 - F_twin)) * (
            (1.0**(p2 + 1) - q_min**(p2 + 1)) / (p2 + 1)) / (
            (1.0**(p2 + 1) - 0.95**(p2 + 1)) / (p2 + 1))
        I2 = (0.95**(p2 + 1) - q_min**(p2 + 1)) / (p2 + 1)
        I3 = A2 * (1.0**(p2 + 1) - 0.95**(p2 + 1)) / (p2 + 1)
        Norm = 1.0 / (I2 + I3)

        m2 = x <= Norm * I2
        m3 = x > Norm * I2
        x[m2] = (x[m2] / Norm * (p2 + 1) + q_min**(p2 + 1))**(1 / (p2 + 1))
        x[m3] = ((x[m3] / Norm - I2) * (p2 + 1) / A2
                  + 0.95**(p2 + 1))**(1 / (p2 + 1))

    return np.clip(x, q_min, 1.0)


# ============================================================
# 5b. Bound-companion prior (PTP/PEB only)
# ============================================================
#
# Upstream TRICERATOPS folds a per-draw ``lnprior_companion`` term into the
# PTP/PEB Monte-Carlo average (marginal_likelihoods.py:lnZ_PTP/lnZ_PEB) that
# this port did not implement until now (see 78fa512's docstring note). It
# is the log probability, from the Moe & Di Stefano (2017) bound-companion
# rate, that a companion drawn with the given mass ratio/parallax/contrast
# is actually bound to the target — everything else in PTP/PEB (the
# transit/eclipse geometry, the flux dilution) is unchanged by this term;
# it only reweights the Monte-Carlo average. It does NOT apply to TTP/TEB
# (no companion in that scenario, upstream's lnZ_TTP/lnZ_TEB have no such
# term at all) nor to this port's NTP/NEB (nearby-star scenarios, which
# reuse _lnZ_TTP/_lnZ_TEB with the default lnprior_companion=0.0) — upstream
# NTP/NEB use a DIFFERENT function, lnprior_background, for an unrelated
# "is this a real background star" prior that this port has never
# implemented; that remains a documented residual difference, out of scope
# here (STP/SEB/DTP/DEB/BTP/BEB are likewise not implemented at all by this
# port and are untouched by this change).
#
# Ported from triceratops/priors.py:lnprior_bound_TP / lnprior_bound_EB and
# triceratops/funcs.py:separation_at_contrast. This port never supplies a
# contrast-curve file (matches upstream's own contrast_curve_file=None /
# molusc_file=None default path — the only one this port's calc_fpp_native
# ever exercises), so separations/contrasts are hardcoded to upstream's own
# no-contrast-curve stand-in ([2.2 arcsec] -> [1.0 mag]), and molusc_file's
# lnprior_companion=np.zeros(N) branch is simply never reached, exactly as
# upstream never reaches it without a molusc_file either.

def _separation_at_contrast(delta_mags):
    """Limiting separation (arcsec) at delta_mags, from a contrast curve.

    Mirrors triceratops/funcs.py:separation_at_contrast(delta_mags,
    separations, contrasts) = np.interp(delta_mags, contrasts,
    separations), with separations/contrasts fixed at upstream's
    no-contrast-curve default [2.2] / [1.0]. With a single-point xp,
    np.interp returns that one fp value (1.0 arcsec) for every delta_mag —
    this is upstream's actual behavior, not a simplification made here.
    """
    return np.interp(delta_mags, np.array([1.0]), np.array([2.2]))


def _lnprior_bound_TP(M_s, plx, delta_mags):
    """Log prior of a bound companion, PTP convention (Porb_comp > 2500 d).

    Mirrors triceratops/priors.py:lnprior_bound_TP exactly (Moe & Di
    Stefano 2017). ``delta_mags`` must already be ``np.abs()``-ed by the
    caller, matching upstream's call site.
    """
    if np.isnan(plx):
        plx = 0.1
    d = 1000.0 / plx
    seps = d * _separation_at_contrast(delta_mags)

    if M_s >= 1.0:
        f1 = 0.020 + 0.04 * np.log10(M_s) + 0.07 * (np.log10(M_s))**2
        f2 = 0.039 + 0.07 * np.log10(M_s) + 0.01 * (np.log10(M_s))**2
        f3 = 0.078 - 0.05 * np.log10(M_s) + 0.04 * (np.log10(M_s))**2
        alpha = 0.018
        dlogP = 0.7
        max_Porbs = ((4 * pi**2) / (G_cgs * M_s * Msun)
                     * (seps * au)**3)**(1 / 2) / 86400
        log_mP = np.log10(max_Porbs)

        t4_partial = (
            alpha * dlogP * (log_mP - 3.4)
            + f2 * (log_mP - 3.4)
            + (f3 - f2 - alpha * dlogP)
            * (0.238095 * log_mP**2 - 0.952381 * log_mP + 0.485714))
        t4 = (alpha * dlogP * (5.5 - 3.4) + f2 * (5.5 - 3.4)
              + (f3 - f2 - alpha * dlogP)
              * (0.238095 * 5.5**2 - 0.952381 * 5.5 + 0.485714))
        t5_partial = f3 * (3.33333 - 17.3566 * np.exp(-0.3 * log_mP))
        t5 = f3 * (3.33333 - 17.3566 * np.exp(-0.3 * 8.0))

        f_comp = np.zeros(len(seps))
        m = (log_mP >= 3.4) & (log_mP < 5.5)
        f_comp[m] = t4_partial[m]
        m = (log_mP >= 5.5) & (log_mP < 8.0)
        f_comp[m] = t4 + t5_partial[m]
        m = (log_mP >= 8.0)
        f_comp[m] = t4 + t5
        lnprior_bound = np.log(f_comp)
    else:
        M_act = M_s
        M_s = 1.0
        f1 = 0.020 + 0.04 * np.log10(M_s) + 0.07 * (np.log10(M_s))**2
        f2 = 0.039 + 0.07 * np.log10(M_s) + 0.01 * (np.log10(M_s))**2
        f3 = 0.078 - 0.05 * np.log10(M_s) + 0.04 * (np.log10(M_s))**2
        alpha = 0.018
        dlogP = 0.7
        max_Porbs = ((4 * pi**2) / (G_cgs * M_s * Msun)
                     * (seps * au)**3)**(1 / 2) / 86400
        log_mP = np.log10(max_Porbs)

        t4_partial = (
            alpha * dlogP * (log_mP - 3.4)
            + f2 * (log_mP - 3.4)
            + (f3 - f2 - alpha * dlogP)
            * (0.238095 * log_mP**2 - 0.952381 * log_mP + 0.485714))
        t4 = (alpha * dlogP * (5.5 - 3.4) + f2 * (5.5 - 3.4)
              + (f3 - f2 - alpha * dlogP)
              * (0.238095 * 5.5**2 - 0.952381 * 5.5 + 0.485714))
        t5_partial = f3 * (3.33333 - 17.3566 * np.exp(-0.3 * log_mP))
        t5 = f3 * (3.33333 - 17.3566 * np.exp(-0.3 * 8.0))

        f_comp = np.zeros(len(seps))
        m = (log_mP >= 3.4) & (log_mP < 5.5)
        f_comp[m] = t4_partial[m]
        m = (log_mP >= 5.5) & (log_mP < 8.0)
        f_comp[m] = t4 + t5_partial[m]
        m = (log_mP >= 8.0)
        f_comp[m] = t4 + t5
        f_act = 0.65 * f_comp + 0.35 * f_comp * M_act
        f_act[f_act < 0.0] = 0.0
        lnprior_bound = np.log(f_act)

    return lnprior_bound


def _lnprior_bound_EB(M_s, plx, delta_mags):
    """Log prior of a bound companion, PEB convention (Porb_comp > 10 d).

    Mirrors triceratops/priors.py:lnprior_bound_EB exactly (same Moe & Di
    Stefano 2017 rate as _lnprior_bound_TP, but the tertiary-companion
    period floor is 10 days rather than 2500 days, so the non-zero mask
    ranges start two steps earlier). ``delta_mags`` must already be
    ``np.abs()``-ed by the caller, matching upstream's call site.
    """
    if np.isnan(plx):
        plx = 0.1
    d = 1000.0 / plx
    seps = d * _separation_at_contrast(delta_mags)

    if M_s >= 1.0:
        f1 = 0.020 + 0.04 * np.log10(M_s) + 0.07 * (np.log10(M_s))**2
        f2 = 0.039 + 0.07 * np.log10(M_s) + 0.01 * (np.log10(M_s))**2
        f3 = 0.078 - 0.05 * np.log10(M_s) + 0.04 * (np.log10(M_s))**2
        alpha = 0.018
        dlogP = 0.7
        max_Porbs = ((4 * pi**2) / (G_cgs * M_s * Msun)
                     * (seps * au)**3)**(1 / 2) / 86400
        log_mP = np.log10(max_Porbs)

        t2_partial = (
            0.5 * (log_mP - 1.0)
            * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (log_mP - 1.0)))
        t2 = (0.5 * (2.0 - 1.0)
              * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (2.0 - 1.0)))
        t3_partial = (
            0.5 * alpha * (log_mP**2 - 5.4 * log_mP + 6.8)
            + f2 * (log_mP - 2.0))
        t3 = 0.5 * alpha * (3.4**2 - 5.4 * 3.4 + 6.8) + f2 * (3.4 - 2.0)
        t4_partial = (
            alpha * dlogP * (log_mP - 3.4)
            + f2 * (log_mP - 3.4)
            + (f3 - f2 - alpha * dlogP)
            * (0.238095 * log_mP**2 - 0.952381 * log_mP + 0.485714))
        t4 = (alpha * dlogP * (5.5 - 3.4) + f2 * (5.5 - 3.4)
              + (f3 - f2 - alpha * dlogP)
              * (0.238095 * 5.5**2 - 0.952381 * 5.5 + 0.485714))
        t5_partial = f3 * (3.33333 - 17.3566 * np.exp(-0.3 * log_mP))
        t5 = f3 * (3.33333 - 17.3566 * np.exp(-0.3 * 8.0))

        f_comp = np.zeros(len(seps))
        m = (log_mP >= 1.0) & (log_mP < 2.0)
        f_comp[m] = t2_partial[m]
        m = (log_mP >= 2.0) & (log_mP < 3.4)
        f_comp[m] = t2 + t3_partial[m]
        m = (log_mP >= 3.4) & (log_mP < 5.5)
        f_comp[m] = t2 + t3 + t4_partial[m]
        m = (log_mP >= 5.5) & (log_mP < 8.0)
        f_comp[m] = t2 + t3 + t4 + t5_partial[m]
        m = (log_mP >= 8.0)
        f_comp[m] = t2 + t3 + t4 + t5
        lnprior_bound = np.log(f_comp)
    else:
        M_act = M_s
        M_s = 1.0
        f1 = 0.020 + 0.04 * np.log10(M_s) + 0.07 * (np.log10(M_s))**2
        f2 = 0.039 + 0.07 * np.log10(M_s) + 0.01 * (np.log10(M_s))**2
        f3 = 0.078 - 0.05 * np.log10(M_s) + 0.04 * (np.log10(M_s))**2
        alpha = 0.018
        dlogP = 0.7
        max_Porbs = ((4 * pi**2) / (G_cgs * M_s * Msun)
                     * (seps * au)**3)**(1 / 2) / 86400
        log_mP = np.log10(max_Porbs)

        t2_partial = (
            0.5 * (log_mP - 1.0)
            * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (log_mP - 1.0)))
        t2 = (0.5 * (2.0 - 1.0)
              * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (2.0 - 1.0)))
        t3_partial = (
            0.5 * alpha * (log_mP**2 - 5.4 * log_mP + 6.8)
            + f2 * (log_mP - 2.0))
        t3 = 0.5 * alpha * (3.4**2 - 5.4 * 3.4 + 6.8) + f2 * (3.4 - 2.0)
        t4_partial = (
            alpha * dlogP * (log_mP - 3.4)
            + f2 * (log_mP - 3.4)
            + (f3 - f2 - alpha * dlogP)
            * (0.238095 * log_mP**2 - 0.952381 * log_mP + 0.485714))
        t4 = (alpha * dlogP * (5.5 - 3.4) + f2 * (5.5 - 3.4)
              + (f3 - f2 - alpha * dlogP)
              * (0.238095 * 5.5**2 - 0.952381 * 5.5 + 0.485714))
        t5_partial = f3 * (3.33333 - 17.3566 * np.exp(-0.3 * log_mP))
        t5 = f3 * (3.33333 - 17.3566 * np.exp(-0.3 * 8.0))

        f_comp = np.zeros(len(seps))
        m = (log_mP >= 1.0) & (log_mP < 2.0)
        f_comp[m] = t2_partial[m]
        m = (log_mP >= 2.0) & (log_mP < 3.4)
        f_comp[m] = t2 + t3_partial[m]
        m = (log_mP >= 3.4) & (log_mP < 5.5)
        f_comp[m] = t2 + t3 + t4_partial[m]
        m = (log_mP >= 5.5) & (log_mP < 8.0)
        f_comp[m] = t2 + t3 + t4 + t5_partial[m]
        m = (log_mP >= 8.0)
        f_comp[m] = t2 + t3 + t4 + t5
        f_act = 0.65 * f_comp + 0.35 * f_comp * M_act
        f_act[f_act < 0.0] = 0.0
        lnprior_bound = np.log(f_act)

    return lnprior_bound


def _lnprior_companion(M_s, plx, fluxratios_comp, scenario):
    """Per-draw ``lnprior_companion`` array fed into PTP/PEB's Z average.

    Mirrors the "calculate priors for companions" block shared by upstream
    lnZ_PTP/lnZ_PEB (contrast_curve_file=None branch): computes delta_mags
    from the already-drawn ``fluxratios_comp``, looks up the bound rate
    (``scenario`` picks TP vs EB convention), then clips it to a proper log
    probability (``<= 0``) and vetoes companions brighter than the target
    (``delta_mags > 0`` -> ``-inf``, matches upstream's post-hoc mask using
    the SIGNED delta_mags even though the prior itself was computed from
    ``np.abs(delta_mags)``).
    """
    delta_mags = 2.5 * np.log10(fluxratios_comp / (1 - fluxratios_comp))
    prior_fn = _lnprior_bound_TP if scenario == "TP" else _lnprior_bound_EB
    lnprior_companion = prior_fn(M_s, plx, np.abs(delta_mags))
    lnprior_companion[lnprior_companion > 0.0] = 0.0
    lnprior_companion[delta_mags > 0.0] = -np.inf
    return lnprior_companion


# ============================================================
# 6. Marginal likelihood for each scenario
# ============================================================

#: Consecutive GPU-batch failures before the GPU backend is retired for the
#: REST OF THIS PROCESS (never persisted — a fresh worker retries the GPU).
#:
#: Why a breaker at all (2026-07-13, mac mini M1): with 5 instances submitting
#: FPP-Metal concurrently, macOS's GPU watchdog kills the command buffer to
#: keep the UI responsive ("[METAL] Command buffer execution failed: Impacting
#: Interactivity"). Retrying straight into the same saturated GPU just feeds the
#: watchdog — the failure is a property of the machine's current GPU pressure,
#: not of the draw. Three strikes and this process finishes its run on the CPU.
_GPU_CIRCUIT_BREAKER_K = 3

_GPU_FALLBACK_STATS = {
    "consecutive": 0,       # reset by any GPU success
    "total": 0,             # every GPU->CPU fallback this process has taken
    "disabled": False,      # breaker tripped?
    "last_error": None,
}


def gpu_fallback_stats() -> dict:
    """Snapshot for the worker's telemetry payload.

    A CPU fallback must never be invisible: CLAUDE.md rule 2 says a worker
    silently running on the CPU is not an acceptable state, and the only way to
    keep that true is to COUNT the fallbacks and ship the count where the
    fleet-doctor can see it.
    """
    return dict(_GPU_FALLBACK_STATS)


def _note_gpu_failure(fail_label, exc):
    """Log a GPU batch failure LOUDLY and trip the breaker if it keeps up.

    Loudly = WARNING, not DEBUG. This path used to log at DEBUG, which meant a
    worker could spend an entire run computing FPP on the CPU — 5x slower, and
    exactly the state CLAUDE.md forbids — without a single line in the log
    saying so.
    """
    global _gpu_batch_transit, _gpu_batch_eb_transit
    log = logging.getLogger("exonit")
    _GPU_FALLBACK_STATS["consecutive"] += 1
    _GPU_FALLBACK_STATS["total"] += 1
    _GPU_FALLBACK_STATS["last_error"] = f"{type(exc).__name__}: {exc}"

    log.warning(
        "GPU %s FAILED on backend=%s (%s: %s) — this batch falls back to CPU "
        "(slower). Consecutive failures: %d/%d.",
        fail_label, _fpp_gpu_backend, type(exc).__name__, exc,
        _GPU_FALLBACK_STATS["consecutive"], _GPU_CIRCUIT_BREAKER_K)

    if (_GPU_FALLBACK_STATS["consecutive"] >= _GPU_CIRCUIT_BREAKER_K
            and not _GPU_FALLBACK_STATS["disabled"]):
        _GPU_FALLBACK_STATS["disabled"] = True
        _gpu_batch_transit = None
        _gpu_batch_eb_transit = None
        log.warning(
            "GPU FPP DISABLED for the rest of this worker process after %d "
            "consecutive failures on backend=%s. Every FPP from now on runs on "
            "the CPU. Restart the worker to retry the GPU. If this is a Mac "
            "under GPU pressure, reduce instances_fpp for this host in "
            "ops/workers.json (the M1 is already excluded).",
            _GPU_CIRCUIT_BREAKER_K, _fpp_gpu_backend)


def _dispatch_lnL(gpu_fn, cpu_fn, args, fail_label):
    """Shared GPU-try/CPU-fallback dispatch for the batched lnL kernels.

    Duplicated 3x verbatim (TTP transit, TEB primary, TEB twin) before this
    extraction — same try/except/guard shape, only the GPU/CPU function
    pair, call args and failure label varied (2026-07-02 cleanup review,
    D7).

    If ``gpu_fn`` is available (a GPU backend was detected at worker init),
    try it first with ``args``. ANY exception falls back to ``cpu_fn``,
    which runs inside ``numba_parallel_guard()`` — required on Apple
    Silicon where numba's only threading layer (workqueue) is not
    thread-safe against the background injection-recovery thread also
    entering a parallel region (see 31383add). The GPU success path
    deliberately does NOT take the guard, matching the pre-extraction
    code. If ``gpu_fn`` is None (no GPU backend detected), go straight to
    the guarded CPU path — same as the pre-extraction ``else`` branch.

    2026-07-14: the fallback is now LOUD and BOUNDED (_note_gpu_failure).
    It used to log at DEBUG and retry the GPU forever, so a worker could run an
    entire job on the CPU without saying so (CLAUDE.md rule 2 forbids exactly
    that) and, on a GPU-saturated Mac, could keep feeding the macOS watchdog
    that was killing its command buffers in the first place.

    What this canNOT catch: an MLX failure that aborts the process from a C++
    completion handler (`libc++abi: terminating due to uncaught exception`)
    never becomes a Python exception, so no `except` here can see it. That case
    is handled server-side instead — a restarted worker's claimed work is
    requeued the moment it re-registers (routes/workers.py's
    _release_restarted_worker_claims), rather than stranding the run.
    """
    if gpu_fn is not None:
        try:
            out = gpu_fn(*args)
        except Exception as e:
            # Loud + counted + breaker (see _note_gpu_failure). The CPU path
            # below is the EXISTING guarded numba one, not a new one: an
            # unguarded numba fallback on a Mac is itself a known crash
            # (numba's workqueue layer is not thread-safe against the
            # background injection thread).
            _note_gpu_failure(fail_label, e)
            with numba_parallel_guard():
                return cpu_fn(*args)
        _GPU_FALLBACK_STATS["consecutive"] = 0
        return out
    else:
        with numba_parallel_guard():
            return cpu_fn(*args)


def _lnZ_TTP(time, flux, sigma, P_orb, M_s, R_s, Teff, N,
             u1, u2, companion_fr=0.0, companion_is_host=False,
             lnprior_companion=0.0, rng=np.random):
    """Marginal likelihood of Target Transiting Planet scenario.

    ``companion_fr`` may be a scalar (TTP: no companion, 0.0 for every
    draw) or an N-length array of per-draw companion flux ratios (PTP,
    via _lnZ_PTP below) — matches upstream TRICERATOPS, which indexes
    ``fluxratios_comp[mask]`` per Monte-Carlo draw rather than collapsing
    it to a single representative value (marginal_likelihoods.py:lnZ_PTP).

    ``lnprior_companion`` is likewise a scalar 0.0 for TTP/NTP (upstream's
    lnZ_TTP has no such term at all) or an N-length array (PTP, see
    _lnprior_companion) folded into the Monte-Carlo average exactly where
    upstream folds it — inside ``np.exp(lnL + lnprior_companion + 600)``,
    not as a separate additive constant on lnZ.
    """
    companion_fr_arr = np.broadcast_to(
        np.asarray(companion_fr, dtype=np.float64), (N,))
    lnprior_arr = np.broadcast_to(
        np.asarray(lnprior_companion, dtype=np.float64), (N,))
    P_arr = np.full(N, P_orb)
    a_cm = ((G_cgs * M_s * Msun) / (4 * pi**2) * (P_arr * 86400)**2)**(1 / 3)
    a_Rs = a_cm / (R_s * Rsun)

    rps = _sample_rp(N, M_s, rng=rng)
    incs = _sample_inc(N, rng=rng)
    eccs = _sample_ecc(N, planet=True, P_orb=P_orb, rng=rng)
    argps = _sample_w(N, rng=rng)

    ks = rps * Rearth / (R_s * Rsun)
    incs_rad = incs * pi / 180.0
    ws_rad = (90.0 - argps) * pi / 180.0

    # Transit probability filter
    e_corr = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    Ptra = (rps * Rearth + R_s * Rsun) / a_cm * e_corr
    inc_min = np.full(N, pi / 2)
    can_transit = Ptra <= 1.0
    inc_min[can_transit] = np.arccos(Ptra[can_transit])

    # Filter: transiting and no collision
    coll = (rps * Rearth + R_s * Rsun) > a_cm * (1 - eccs)
    mask = (incs_rad >= inc_min) & (~coll)

    if not np.any(mask):
        return -np.inf

    N_mask = np.sum(mask)
    comp_fr = companion_fr_arr[mask]

    lnL = np.full(N, -np.inf)
    lnL[mask] = _dispatch_lnL(
        _gpu_batch_transit, _batch_transit_lnL,
        (time, flux, sigma,
         ks[mask], P_arr[mask], incs_rad[mask], a_Rs[mask],
         eccs[mask], ws_rad[mask], u1, u2,
         comp_fr, companion_is_host),
        "transit")

    Z = np.mean(np.nan_to_num(np.exp(lnL + lnprior_arr + 600)))
    return np.log(Z) if Z > 0 else -np.inf


def _lnZ_TEB(time, flux, sigma, P_orb, M_s, R_s, Teff, N, u1, u2,
              companion_fr=0.0, companion_is_host=False,
              lnprior_companion=0.0, rng=np.random):
    """Marginal likelihood of Target Eclipsing Binary (and twin EB) scenarios.

    Returns (lnZ_EB, lnZ_EB_twin).

    ``companion_fr`` may be a scalar (TEB: no companion, 0.0 for every
    draw) or an N-length array of per-draw companion flux ratios (PEB,
    via _lnZ_PEB below) — see the matching note on _lnZ_TTP.

    ``lnprior_companion`` (see _lnZ_TTP's matching note) is the SAME
    N-length array for both the EB and twin-EB branches below — upstream
    lnZ_PEB computes ``lnprior_companion`` once and folds it into both
    ``Z`` and ``Z_twin``'s Monte-Carlo average (it is indexed by the
    companion draw, independent of which of the two EB masks the target's
    own eclipsing pair falls into).
    """
    companion_fr_arr = np.broadcast_to(
        np.asarray(companion_fr, dtype=np.float64), (N,))
    lnprior_arr = np.broadcast_to(
        np.asarray(lnprior_companion, dtype=np.float64), (N,))
    P_arr = np.full(N, P_orb)

    incs = _sample_inc(N, rng=rng)
    qs = _sample_q(N, M_s, rng=rng)
    eccs = _sample_ecc(N, planet=False, P_orb=P_orb, rng=rng)
    argps = _sample_w(N, rng=rng)

    masses_eb = qs * M_s
    radii_eb, _ = stellar_relations(masses_eb)
    fluxratios_eb = flux_relation(masses_eb) / (
        flux_relation(masses_eb) + flux_relation(np.array([M_s])))

    incs_rad = incs * pi / 180.0
    ws_rad = (90.0 - argps) * pi / 180.0

    # Semi-major axes
    a_cm = ((G_cgs * (M_s + masses_eb) * Msun) / (4 * pi**2)
            * (P_arr * 86400)**2)**(1 / 3)
    a_twin_cm = ((G_cgs * (M_s + masses_eb) * Msun) / (4 * pi**2)
                 * (2 * P_arr * 86400)**2)**(1 / 3)

    a_Rs = a_cm / (R_s * Rsun)
    a_twin_Rs = a_twin_cm / (R_s * Rsun)

    ks = radii_eb / R_s

    # Transit probability
    e_corr = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    Ptra = (radii_eb * Rsun + R_s * Rsun) / a_cm * e_corr
    Ptra_twin = (radii_eb * Rsun + R_s * Rsun) / a_twin_cm * e_corr

    inc_min = np.full(N, pi / 2)
    can = Ptra <= 1.0
    inc_min[can] = np.arccos(Ptra[can])
    inc_min_twin = np.full(N, pi / 2)
    can_tw = Ptra_twin <= 1.0
    inc_min_twin[can_tw] = np.arccos(Ptra_twin[can_tw])

    coll = (radii_eb * Rsun + R_s * Rsun) > a_cm * (1 - eccs)
    coll_twin = (2 * R_s * Rsun) > a_twin_cm * (1 - eccs)

    # q < 0.95 for EB
    mask_eb = (incs_rad >= inc_min) & (~coll) & (qs < 0.95)
    # q >= 0.95 for twin EB
    mask_twin = (incs_rad >= inc_min_twin) & (~coll_twin) & (qs >= 0.95)

    comp_arr = companion_fr_arr

    lnL_eb = np.full(N, -np.inf)
    if np.any(mask_eb):
        n_eb = np.sum(mask_eb)
        lnL_eb[mask_eb] = _dispatch_lnL(
            _gpu_batch_eb_transit, _batch_eb_transit_lnL,
            (time, flux, sigma,
             ks[mask_eb], fluxratios_eb[mask_eb], P_arr[mask_eb],
             incs_rad[mask_eb], a_Rs[mask_eb], eccs[mask_eb], ws_rad[mask_eb],
             u1, u2, comp_arr[mask_eb], companion_is_host),
            "EB")

    lnL_twin = np.full(N, -np.inf)
    if np.any(mask_twin):
        lnL_twin[mask_twin] = _dispatch_lnL(
            _gpu_batch_eb_transit, _batch_eb_transit_lnL,
            (time, flux, sigma,
             ks[mask_twin], fluxratios_eb[mask_twin], 2 * P_arr[mask_twin],
             incs_rad[mask_twin], a_twin_Rs[mask_twin],
             eccs[mask_twin], ws_rad[mask_twin],
             u1, u2, comp_arr[mask_twin], companion_is_host),
            "twin EB")

    Z_eb = np.mean(np.nan_to_num(np.exp(lnL_eb + lnprior_arr + 600)))
    Z_twin = np.mean(np.nan_to_num(np.exp(lnL_twin + lnprior_arr + 600)))

    lnZ_eb = np.log(Z_eb) if Z_eb > 0 else -np.inf
    lnZ_twin = np.log(Z_twin) if Z_twin > 0 else -np.inf
    return lnZ_eb, lnZ_twin


def _lnZ_PTP(time, flux, sigma, P_orb, M_s, R_s, Teff, N, u1, u2, plx, rng=np.random):
    """Marginal likelihood of Planet around physical companion (PTP).

    Mirrors triceratops/marginal_likelihoods.py:lnZ_PTP exactly: the
    physically-sampled companion (masses_comp, via the binary mass-ratio
    prior) only sets the per-draw flux-dilution ratio fed to the TTP
    likelihood as ``companion_fr`` — one value per Monte-Carlo draw, not
    a single scalar collapsed across draws. The transit geometry itself
    (a_cm, ks, Ptra, collision check, `_sample_rp`) and the limb-darkening
    coefficients (u1, u2, passed straight through) stay the TARGET star's,
    exactly as upstream does; `companion_is_host=False` matches upstream's
    call (the target, not the companion, is modeled as the transit host,
    with the companion adding only third light). Upstream additionally
    folds a ``lnprior_companion`` term (bound-companion probability from
    parallax + Δmag + the Moe & Di Stefano mass-ratio prior,
    `lnprior_bound_TP`) into Z — see `_lnprior_companion`.
    """
    qs_comp = _sample_q_companion(N, M_s, rng=rng)
    masses_comp = qs_comp * M_s
    fluxratios_comp = flux_relation(masses_comp) / (
        flux_relation(masses_comp) + flux_relation(np.array([M_s])))
    lnprior_companion = _lnprior_companion(M_s, plx, fluxratios_comp, "TP")

    return _lnZ_TTP(time, flux, sigma, P_orb, M_s, R_s, Teff, N,
                     u1, u2, companion_fr=fluxratios_comp,
                     companion_is_host=False,
                     lnprior_companion=lnprior_companion, rng=rng)


def _lnZ_PEB(time, flux, sigma, P_orb, M_s, R_s, Teff, N, u1, u2, plx, rng=np.random):
    """Marginal likelihood of EB around physical companion (PEB).

    Same convention as _lnZ_PTP above (see its docstring) — mirrors
    triceratops/marginal_likelihoods.py:lnZ_PEB. Note this also fixes a
    second, independent bug: _lnZ_TEB used to hardcode its dilution array
    to zero (``comp_arr = np.zeros(N)``), silently discarding whatever
    ``companion_fr`` was passed in — PEB's third-light dilution was a
    no-op even before this change reached the M_s/R_s question at all.
    """
    qs_comp = _sample_q_companion(N, M_s, rng=rng)
    masses_comp = qs_comp * M_s
    fluxratios_comp = flux_relation(masses_comp) / (
        flux_relation(masses_comp) + flux_relation(np.array([M_s])))
    lnprior_companion = _lnprior_companion(M_s, plx, fluxratios_comp, "EB")

    return _lnZ_TEB(time, flux, sigma, P_orb, M_s, R_s, Teff, N,
                     u1, u2, companion_fr=fluxratios_comp,
                     companion_is_host=False,
                     lnprior_companion=lnprior_companion, rng=rng)


# ============================================================
# 7. TIC query and flux ratio estimation
# ============================================================

def _query_tic_stars(tic_id, server_url=None, token=None, timeout=15):
    """Fetch nearby TIC stars from the ExoNit server.

    The server handles MAST queries and caching centrally.
    Returns (stars_list, gal_b) or None on failure.
    """
    if not server_url:
        logging.getLogger("exonit").debug(f"FPP: no server_url, cannot query nearby stars")
        return None

    import requests

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.get(f"{server_url}/api/nearby/{tic_id}",
                            headers=headers, timeout=timeout)
        if resp.status_code != 200:
            logging.getLogger("exonit").debug(f"FPP: nearby query failed for TIC {tic_id}: "
                  f"HTTP {resp.status_code}")
            return None

        data = resp.json()
        stars = data.get("stars", [])
        gal_b = data.get("gal_b", 0.0)

        # Promote the target TIC to stars[0]. Catalogs.query_region does
        # not sort by distance from the query center, so the natural
        # position 0 is often an arbitrary faint neighbour with missing
        # catalog data — see the "FPP missing stellar params" incident
        # (2026-04-20): 39/98 baseline-100 TICs lost FPP because the
        # code further down takes `stars[0]` as the host star.
        target_idx = next(
            (i for i, s in enumerate(stars)
             if str(s.get("ID")) == str(tic_id)),
            None,
        )
        if target_idx is None:
            logging.getLogger("exonit").warning(
                f"FPP: TIC {tic_id} not found in its own /api/nearby "
                f"result ({len(stars)} stars); using stars[0] as fallback")
        elif target_idx != 0:
            stars.insert(0, stars.pop(target_idx))

        # Convert null → NaN for numeric fields
        for s in stars:
            for k in ('mass', 'rad', 'Teff', 'Tmag', 'plx'):
                if s.get(k) is None:
                    s[k] = np.nan

        source = data.get("source", "unknown")
        logging.getLogger("exonit").debug(f"FPP: TIC {tic_id} nearby stars from {source} "
              f"({len(stars)} stars)")
        return (stars, gal_b)

    except Exception as e:
        logging.getLogger("exonit").debug(f"FPP: nearby query failed for TIC {tic_id}: {e}")
        return None


def _load_trilegal_stars(tic_id, workdir, target_ra, target_dec, target_Tmag):
    """Load pre-downloaded TRILEGAL CSV as supplementary background stars.

    Returns list of star dicts compatible with _compute_flux_ratios,
    or empty list if no TRILEGAL file available.
    """
    import pandas as pd

    fname = os.path.join(workdir, f"{tic_id}_TRILEGAL.csv")
    if not os.path.exists(fname):
        return []

    try:
        df = pd.read_csv(fname, sep=r'\s+', low_memory=False)
        # TRILEGAL columns: Mact, logAge, [M/H], logL, logTe, logg,
        #   Av, mbol, Kepler, TESS, ...
        if 'TESS' not in df.columns:
            return []

        stars = []
        for _, row in df.iterrows():
            tmag = float(row['TESS'])
            # Only include stars bright enough to matter (within 8 mag)
            if tmag > target_Tmag + 8:
                continue
            teff = 10**float(row['logTe']) if 'logTe' in df.columns else 5777
            mass = float(row['Mact']) if 'Mact' in df.columns else 1.0
            logg_val = float(row['logg']) if 'logg' in df.columns else 4.4
            rad = np.sqrt(G_cgs * mass * Msun / (10**logg_val)) / Rsun

            stars.append({
                'ID': f"TRILEGAL_{len(stars)}",
                'ra': target_ra,   # TRILEGAL stars don't have exact positions
                'dec': target_dec,
                'mass': mass,
                'rad': rad,
                'Teff': teff,
                'Tmag': tmag,
                'plx': np.nan,
                'is_trilegal': True,
            })

        return stars
    except Exception as e:
        logging.getLogger("exonit").debug(f"FPP: failed to load TRILEGAL for TIC {tic_id}: {e}")
        return []


def _compute_flux_ratios(stars, depth_ppm):
    """Compute flux ratios and required transit depths for each star.

    Uses TESS magnitude differences with Gaussian PSF weighting
    based on angular separation (PSF FWHM ~ 21 arcsec).
    """
    if not stars:
        return []

    target = stars[0]
    Tmag_target = target['Tmag']

    # PSF-weighted flux contributions
    sigma_psf = 21.0  # arcsec, TESS PSF
    for s in stars:
        sep = np.sqrt((s['ra'] - target['ra'])**2 +
                       (s['dec'] - target['dec'])**2) * 3600  # arcsec
        raw_flux = 10**((Tmag_target - s['Tmag']) / 2.5)
        # PSF weighting: stars far from center contribute less
        psf_weight = np.exp(-0.5 * (sep / sigma_psf)**2)
        s['weighted_flux'] = raw_flux * psf_weight
        s['sep_arcsec'] = sep

    total_flux = sum(s['weighted_flux'] for s in stars)
    for s in stars:
        s['fluxratio'] = s['weighted_flux'] / total_flux
        s['tdepth'] = depth_ppm / s['fluxratio'] if s['fluxratio'] > 0 else 0

    return stars


# ============================================================
# 8. Phase-fold helper
# ============================================================

def _phase_fold_bin(time, flux, period, t0, n_bins=1000):
    """Phase-fold and bin light curve, centered on transit."""
    # See docs/reviews/2026-07-02-opus48-cleanup-review.md item D5 and
    # tests/worker/test_phase_fold_golden.py for the bit-identical proof.
    phase = phase_fold(time, t0, period, fold_order="div_mod", center="mask")
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(phase, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    t_out, f_out = [], []
    for i in range(n_bins):
        mask = bin_idx == i
        if np.sum(mask) > 0:
            t_out.append(bin_centers[i] * period)  # time relative to t0
            f_out.append(np.nanmedian(flux[mask]))

    return np.array(t_out), np.array(f_out)


def renorm_flux(flux, flux_err, star_fluxratio):
    """Renormalize flux for a specific star's contribution."""
    renormed = (flux - (1 - star_fluxratio)) / star_fluxratio
    renormed_err = flux_err / star_fluxratio
    return renormed, renormed_err


# ============================================================
# 9. Main FPP calculation
# ============================================================

# ============================================================
# 9. Deterministic per-candidate seed
# ============================================================
#
# Normative scheme shared with the Rust (fpp/rng.rs::tic_seed) and Swift
# (FPPRNG.ticSeed) ports — see exonit-rust-worker-fpp/parity/fpp/README.md
# "Deterministic per-TIC seed" and fixtures/fpp/rng_golden.json. Any change
# here must be mirrored in both ports or FPP/NFPP will silently diverge
# between the Python and native workers for the same candidate.


def _fnv1a64(data: bytes) -> int:
    """FNV-1a 64-bit hash (offset basis 0xcbf29ce484222325, prime 0x100000001b3)."""
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def tic_seed(tic_id, period, t0) -> int:
    """Deterministic 32-bit seed for one (tic_id, period, t0) candidate.

    key = f"{tic_id}:{period:.10e}:{t0:.10e}" (ASCII), seed = fnv1a64(key) & 0xFFFFFFFF.
    Must match Rust `tic_seed()` and Swift `FPPRNG.ticSeed()` byte-for-byte.
    """
    key = f"{tic_id}:{period:.10e}:{t0:.10e}"
    return _fnv1a64(key.encode("ascii")) & 0xFFFFFFFF


class FPPAborted(Exception):
    """Raised when FPP computation is aborted by user."""
    pass


def calc_fpp_native(tic_id, period, t0, duration, depth_ppm, sectors,
                     time, flux, N=100000, workdir=".",
                     server_url=None, token=None, abort_check=None,
                     force_cpu=False):
    """Calculate False Positive Probability without triceratops/pytransit.

    Implements Giacalone & Dressing (2020) FPP framework:
    - Scenarios: TTP, TEB, TEBx2P, PTP, PEB, PEBx2P, NTP, NEB, NEBx2P
    - Monte Carlo marginal likelihoods with N draws per scenario
    - FPP = 1 - P(TP) - P(PTP)

    Args:
        tic_id: TIC identifier.
        period: Orbital period (days).
        t0: Transit epoch (BJD - 2457000).
        duration: Transit duration (days).
        depth_ppm: Transit depth (ppm).
        sectors: Array of TESS sector numbers.
        time: Light curve time array.
        flux: Light curve flux array (normalized).
        N: MC draws per scenario (default 100000).
        workdir: Directory for cache files.
        server_url: ExoNit server URL for nearby star queries.
        token: Authentication token for server API.
        abort_check: Optional callable returning True to abort computation.

    Returns:
        Dict with fpp, nfpp, verdict, or None on failure.
        Raises FPPAborted if abort_check() returns True.
    """
    def _check_abort():
        if abort_check and abort_check():
            raise FPPAborted(f"FPP aborted for TIC {tic_id}")

    # Deterministic per-candidate seed (normative scheme, see tic_seed()
    # docstring) — a private RandomState (not the process-global np.random)
    # so two candidates never perturb each other's draw stream even if
    # calc_fpp_native is ever called from more than one thread.
    seed_u32 = tic_seed(tic_id, period, t0)
    rng = np.random.RandomState(seed_u32)

    # Temporarily disable GPU for FPP when running parallel with injection
    global _gpu_batch_transit, _gpu_batch_eb_transit
    _saved_gpu = (_gpu_batch_transit, _gpu_batch_eb_transit)
    if force_cpu:
        _gpu_batch_transit = None
        _gpu_batch_eb_transit = None

    try:
        # Phase-fold and bin
        time_pf, flux_pf = _phase_fold_bin(time, flux, period, t0)
        if len(time_pf) < 10:
            logging.getLogger("exonit").warning(
                f"FPP skip TIC {tic_id}: too few phase-folded points "
                f"({len(time_pf)})")
            return None

        # Remove NaNs
        good = ~np.isnan(time_pf) & ~np.isnan(flux_pf)
        time_pf = np.ascontiguousarray(time_pf[good], dtype=np.float64)
        flux_pf = np.ascontiguousarray(flux_pf[good], dtype=np.float64)
        sigma = float(np.std(flux_pf))
        if sigma <= 0:
            sigma = 1e-6

        # Query TIC for nearby stars (via server API)
        tic_result = _query_tic_stars(tic_id, server_url=server_url,
                                       token=token)
        if not tic_result:
            logging.getLogger("exonit").warning(
                f"FPP skip TIC {tic_id}: no TIC neighbor data")
            return None

        stars, gal_b = tic_result

        # Crowding assessment: |b| < 10° or >15 TIC neighbors
        n_nearby = len(stars) - 1
        high_crowding = abs(gal_b) < 10.0 or n_nearby > 15

        # Load TRILEGAL supplementary stars if available (helps in dense fields)
        trilegal_stars = _load_trilegal_stars(
            tic_id, workdir,
            stars[0]['ra'], stars[0]['dec'], stars[0]['Tmag'])
        if trilegal_stars:
            stars.extend(trilegal_stars)
            logging.getLogger("exonit").debug(f"FPP: +{len(trilegal_stars)} TRILEGAL background stars")

        stars = _compute_flux_ratios(stars, depth_ppm)
        target = stars[0]

        # Validate target star properties
        M_s = target.get('mass', np.nan)
        R_s = target.get('rad', np.nan)
        Teff = target.get('Teff', np.nan)
        plx = target.get('plx', np.nan)

        if np.isnan(M_s) or np.isnan(R_s) or np.isnan(Teff):
            logging.getLogger("exonit").warning(
                f"FPP skip TIC {tic_id}: missing stellar params "
                f"(M={M_s} R={R_s} Teff={Teff})")
            return None

        # Limb darkening
        logg = np.log10(G_cgs * M_s * Msun / (R_s * Rsun)**2)
        u1, u2 = get_ldc(Teff, logg)

        # Renormalize flux for target star
        flux_t, sigma_t = renorm_flux(flux_pf, sigma, target['fluxratio'])
        sigma_t = float(sigma_t)

        P_orb = float(period)

        crowding_tag = " HIGH_CROWDING" if high_crowding else ""
        trilegal_tag = f" +TRILEGAL" if trilegal_stars else ""
        logging.getLogger("exonit").debug(f"FPP: TIC {tic_id} — M={M_s:.2f} R={R_s:.2f} "
              f"Teff={Teff:.0f} u1={u1:.3f} u2={u2:.3f} "
              f"N_nearby={n_nearby} b={gal_b:.1f}°"
              f"{crowding_tag}{trilegal_tag}")

        # Compute marginal likelihoods for each scenario
        # _check_abort() between each heavy computation so Q/Ctrl+C can stop us
        lnZ = {}

        # Target star scenarios
        _check_abort()
        lnZ['TP'] = _lnZ_TTP(time_pf, flux_t, sigma_t, P_orb,
                               M_s, R_s, Teff, N, u1, u2, rng=rng)
        _check_abort()
        lnZ_eb, lnZ_eb_twin = _lnZ_TEB(time_pf, flux_t, sigma_t, P_orb,
                                          M_s, R_s, Teff, N, u1, u2, rng=rng)
        lnZ['EB'] = lnZ_eb
        lnZ['EBx2P'] = lnZ_eb_twin

        # Physical companion scenarios
        _check_abort()
        if not np.isnan(plx):
            lnZ['PTP'] = _lnZ_PTP(time_pf, flux_t, sigma_t, P_orb,
                                    M_s, R_s, Teff, N, u1, u2, plx, rng=rng)
            _check_abort()
            lnZ_peb, lnZ_peb_twin = _lnZ_PEB(time_pf, flux_t, sigma_t,
                                                P_orb, M_s, R_s, Teff,
                                                N, u1, u2, plx, rng=rng)
            lnZ['PEB'] = lnZ_peb
            lnZ['PEBx2P'] = lnZ_peb_twin
        else:
            lnZ['PTP'] = -np.inf
            lnZ['PEB'] = -np.inf
            lnZ['PEBx2P'] = -np.inf

        # Nearby star scenarios (NTP, NEB for each neighbor)
        _check_abort()
        nearby_lnZ = []
        filtered_nearby = [s for s in stars[1:] if s.get('tdepth', 0) > 0]
        for ns in filtered_nearby[:5]:  # limit to 5 nearest
            ns_M = ns.get('mass', 1.0)
            ns_R = ns.get('rad', 1.0)
            ns_Teff = ns.get('Teff', 5777)
            if np.isnan(ns_M):
                ns_M = 1.0
            if np.isnan(ns_R):
                ns_R = 1.0
            if np.isnan(ns_Teff):
                ns_Teff = 5777

            ns_logg = np.log10(G_cgs * ns_M * Msun / (ns_R * Rsun)**2)
            ns_u1, ns_u2 = get_ldc(ns_Teff, ns_logg)

            flux_ns, sigma_ns = renorm_flux(flux_pf, sigma, ns['fluxratio'])

            _check_abort()
            ntp = _lnZ_TTP(time_pf, flux_ns, float(sigma_ns), P_orb,
                            ns_M, ns_R, ns_Teff, N // 2, ns_u1, ns_u2, rng=rng)
            _check_abort()
            neb, neb_twin = _lnZ_TEB(time_pf, flux_ns, float(sigma_ns),
                                       P_orb, ns_M, ns_R, ns_Teff,
                                       N // 2, ns_u1, ns_u2, rng=rng)
            nearby_lnZ.extend([ntp, neb, neb_twin])

        # Combine all scenarios
        all_lnZ = list(lnZ.values()) + nearby_lnZ
        all_lnZ = np.array([x if np.isfinite(x) else -np.inf for x in all_lnZ])

        # Relative probabilities
        max_lnZ = np.max(all_lnZ[all_lnZ > -np.inf]) if np.any(
            all_lnZ > -np.inf) else 0
        probs = np.exp(all_lnZ - max_lnZ)
        total = np.sum(probs)

        if total <= 0:
            logging.getLogger("exonit").warning(
                f"FPP skip TIC {tic_id}: all lnZ scenarios non-finite")
            return None

        probs /= total

        # FPP = 1 - P(TP) - P(PTP)
        P_TP = probs[0]  # TP
        P_PTP = probs[3] if len(probs) > 3 else 0  # PTP
        fpp = 1.0 - P_TP - P_PTP

        # NFPP = sum of nearby star probabilities
        n_target_scenarios = len(lnZ)
        nfpp = float(np.sum(probs[n_target_scenarios:]))

        fpp = float(np.clip(fpp, 0, 1))
        nfpp = float(np.clip(nfpp, 0, 1))

        return {
            "fpp": fpp,
            "nfpp": nfpp,
            "verdict": fpp_verdict(fpp, nfpp=nfpp, high_crowding=high_crowding),
            "high_crowding": high_crowding,
            "gal_b": round(gal_b, 2),
            "n_nearby": n_nearby,
            "trilegal_used": len(trilegal_stars) > 0,
            "fpp_seed": int(seed_u32),
        }

    except FPPAborted:
        logging.getLogger("exonit").info(f"FPP aborted for TIC {tic_id}")
        raise
    except Exception as e:
        logging.getLogger("exonit").warning(
            f"FPP failed for TIC {tic_id}: {type(e).__name__}: {e}")
        import traceback
        logging.getLogger("exonit").debug(traceback.format_exc())
        return None
    finally:
        if force_cpu:
            _gpu_batch_transit, _gpu_batch_eb_transit = _saved_gpu


# ============================================================
# 10. Utilities
# ============================================================

def fpp_verdict(fpp, nfpp=None, high_crowding=False):
    # Validation rule (Giacalone+ 2020): FPP<0.015 AND NFPP<0.001.
    # high_crowding downgrades validated to candidate because the TIC-only
    # background catalog undercounts BEB contributors past Tmag~16.
    if fpp >= FPP_CANDIDATE:
        return "likely_fp"
    if fpp < FPP_VALIDATED and nfpp is not None and nfpp < NFPP_VALIDATED:
        return "candidate" if high_crowding else "validated"
    return "candidate"


# ============================================================
# Numba warm-up on import (compile transit model)
# ============================================================

def _warmup():
    """Pre-compile numba functions with small arrays."""
    t = np.linspace(-0.05, 0.05, 10)
    _compute_transit_lc(t, 0.1, 0.0, 3.0, 15.0, 89.0 * pi / 180,
                         0.0, pi / 2, 0.3, 0.2)
    _batch_transit_lnL(t, np.ones(10), 0.001,
                        np.array([0.1]), np.array([3.0]),
                        np.array([89.0 * pi / 180]),
                        np.array([15.0]),
                        np.array([0.0]), np.array([pi / 2]),
                        0.3, 0.2, np.array([0.0]), False)

_init_fpp_gpu()

try:
    _warmup()
except Exception:
    pass
