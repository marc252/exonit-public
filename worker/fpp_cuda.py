"""CUDA-accelerated FPP transit model evaluation using CuPy RawKernel.

Parallelizes batch transit likelihood computation on NVIDIA GPUs.
Each thread handles one (draw, time_point) pair, computing the full
Kepler solver -> true anomaly -> projected separation -> ring integration
-> model flux -> chi2 contribution pipeline.

Performance: significant speedup over CPU numba for large N*T workloads.
Fallback: if CuPy/CUDA not available, is_available() returns False.

Usage:
    from fpp_cuda import batch_transit_lnL, is_available
    if is_available():
        lnL = batch_transit_lnL(time, flux, sigma, ks, ...)
"""

import logging
import numpy as np

try:
    import cupy as cp
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False


# ================================================================== #
# CUDA Kernel: transit chi2 (1 thread per draw*time_point)
# ================================================================== #

_TRANSIT_CHI2_KERNEL = r"""
extern "C" __global__
void transit_chi2_kernel(
    const double* __restrict__ time,       // (T,) float64
    const double* __restrict__ flux,       // (T,) float64
    const float*  __restrict__ ks,         // (N,) float32
    const float*  __restrict__ P_orbs,     // (N,) float32
    const float*  __restrict__ incs_rad,   // (N,) float32
    const float*  __restrict__ a_Rs_arr,   // (N,) float32
    const float*  __restrict__ eccs,       // (N,) float32
    const float*  __restrict__ ws_rad,     // (N,) float32
    const float*  __restrict__ comp_fr,    // (N,) float32
    int           companion_is_host,       // scalar bool as int
    float         u1,
    float         u2,
    int           N,
    int           T,
    double*       chi2                     // (N,) float64, output
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N * T) return;

    int j = tid / T;   // draw index
    int i = tid % T;   // time index

    float k       = ks[j];
    float period   = P_orbs[j];
    float inc      = incs_rad[j];
    float a_Rs     = a_Rs_arr[j];
    float ecc      = eccs[j];
    float w        = ws_rad[j];
    float fr       = comp_fr[j];

    // --- Kepler solver (float32, 10 fixed iterations) ---
    float two_pi = 6.2831853071795864f;
    float t_val = (float)time[i];
    float phase = t_val / period;
    float M = two_pi * (phase - floorf(phase)) - w;

    float E = M;
    for (int iter = 0; iter < 10; iter++) {
        float dE = (E - ecc * sinf(E) - M) / (1.0f - ecc * cosf(E));
        E -= dE;
    }

    // --- True anomaly ---
    float nu;
    if (ecc < 1e-10f) {
        nu = M;
    } else {
        float cosE = cosf(E);
        nu = atan2f(sqrtf(1.0f - ecc * ecc) * sinf(E), cosE - ecc);
    }

    // --- Projected separation z ---
    float r_factor = (1.0f - ecc * ecc) / (1.0f + ecc * cosf(nu));
    float f = nu + w;
    float sin_f = sinf(f);
    float cos_f = cosf(f);
    float cos_i = cosf(inc);
    float z = a_Rs * r_factor * sqrtf(sin_f * sin_f + cos_i * cos_i * cos_f * cos_f);

    // --- Ring integration (100 rings) ---
    float model_flux = 1.0f;

    if (z < 1.0f + k && k > 0.0f) {
        float omega = 1.0f - u1 / 3.0f - u2 / 6.0f;
        if (omega <= 0.0f) omega = 1.0f;

        float dr = 0.01f;  // 1.0 / 100
        float blocked = 0.0f;
        float p = k;
        float p2 = p * p;
        float z2 = z * z;

        for (int ri = 0; ri < 100; ri++) {
            float r = (ri + 0.5f) * dr;
            float r2 = r * r;
            float mu = sqrtf(1.0f - r2);
            float one_minus_mu = 1.0f - mu;
            float I_r = 1.0f - u1 * one_minus_mu - u2 * one_minus_mu * one_minus_mu;

            float frac;
            if (z < 1e-12f) {
                frac = (r <= p) ? 1.0f : 0.0f;
            } else if (r >= z + p) {
                frac = 0.0f;
            } else if (r + p <= z) {
                frac = 0.0f;
            } else if (r <= fabsf(z - p)) {
                frac = (p > z) ? 1.0f : 0.0f;
            } else {
                float arg = (r2 + z2 - p2) / (2.0f * r * z);
                if (arg >= 1.0f) {
                    frac = 0.0f;
                } else if (arg <= -1.0f) {
                    frac = 1.0f;
                } else {
                    frac = acosf(arg) / 3.14159265358979323846f;
                }
            }

            blocked += I_r * frac * 2.0f * r * dr;
        }

        model_flux = 1.0f - blocked / omega;
    }

    // --- Companion dilution ---
    if (fr > 0.0f) {
        float F_dilute;
        if (companion_is_host) {
            F_dilute = 1.0f / (fr / (1.0f - fr));
        } else {
            F_dilute = fr / (1.0f - fr);
        }
        model_flux = (model_flux + F_dilute) / (1.0f + F_dilute);
    }

    // --- Chi2 contribution (float64 atomicAdd) ---
    double diff = flux[i] - (double)model_flux;
    atomicAdd(&chi2[j], diff * diff);
}
"""

# ================================================================== #
# CUDA Kernel: chi2 -> lnL conversion (1 thread per draw)
# ================================================================== #

_CHI2_TO_LNL_KERNEL = r"""
extern "C" __global__
void chi2_to_lnl_kernel(
    const double* __restrict__ chi2,   // (N,) float64
    double        sigma,
    int           T,
    int           N,
    double*       lnL                  // (N,) float64, output
) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j >= N) return;

    double ln2pi = 1.8378770664093453;
    double lnsigma = log(sigma);
    double sigma2 = sigma * sigma;
    lnL[j] = -0.5 * ln2pi - lnsigma - 0.5 * chi2[j] / sigma2;
}
"""

# ================================================================== #
# CUDA Kernel: EB transit chi2 + secdepth (1 thread per draw*time)
# ================================================================== #

_EB_TRANSIT_CHI2_KERNEL = r"""
extern "C" __global__
void eb_transit_chi2_kernel(
    const double* __restrict__ time,
    const double* __restrict__ flux,
    const float*  __restrict__ ks,
    const float*  __restrict__ eb_fluxratios,
    const float*  __restrict__ P_orbs,
    const float*  __restrict__ incs_rad,
    const float*  __restrict__ a_Rs_arr,
    const float*  __restrict__ eccs,
    const float*  __restrict__ ws_rad,
    const float*  __restrict__ comp_fr,
    int           companion_is_host,
    float         u1,
    float         u2,
    int           N,
    int           T,
    double*       chi2,          // (N,) float64, output
    float*        secdepth_out   // (N,) float32, output
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N * T) return;

    int j = tid / T;
    int i = tid % T;

    float k = ks[j];
    // Clamp k near 1.0 like CPU code
    if (fabsf(k - 1.0f) < 1e-6f) k *= 0.999f;

    float period = P_orbs[j];
    float inc    = incs_rad[j];
    float a_Rs   = a_Rs_arr[j];
    float ecc    = eccs[j];
    float w      = ws_rad[j];

    // --- Kepler solver ---
    float two_pi = 6.2831853071795864f;
    float t_val = (float)time[i];
    float phase = t_val / period;
    float M = two_pi * (phase - floorf(phase)) - w;

    float E = M;
    for (int iter = 0; iter < 10; iter++) {
        float dE = (E - ecc * sinf(E) - M) / (1.0f - ecc * cosf(E));
        E -= dE;
    }

    float nu;
    if (ecc < 1e-10f) {
        nu = M;
    } else {
        float cosE = cosf(E);
        nu = atan2f(sqrtf(1.0f - ecc * ecc) * sinf(E), cosE - ecc);
    }

    float r_factor = (1.0f - ecc * ecc) / (1.0f + ecc * cosf(nu));
    float f = nu + w;
    float sin_f = sinf(f);
    float cos_f = cosf(f);
    float cos_i = cosf(inc);
    float z = a_Rs * r_factor * sqrtf(sin_f * sin_f + cos_i * cos_i * cos_f * cos_f);

    // --- Ring integration ---
    float model_flux = 1.0f;

    if (z < 1.0f + k && k > 0.0f) {
        float omega = 1.0f - u1 / 3.0f - u2 / 6.0f;
        if (omega <= 0.0f) omega = 1.0f;

        float dr = 0.01f;
        float blocked = 0.0f;
        float p = k;
        float p2 = p * p;
        float z2 = z * z;

        for (int ri = 0; ri < 100; ri++) {
            float r = (ri + 0.5f) * dr;
            float r2 = r * r;
            float mu = sqrtf(1.0f - r2);
            float one_minus_mu = 1.0f - mu;
            float I_r = 1.0f - u1 * one_minus_mu - u2 * one_minus_mu * one_minus_mu;

            float frac;
            if (z < 1e-12f) {
                frac = (r <= p) ? 1.0f : 0.0f;
            } else if (r >= z + p) {
                frac = 0.0f;
            } else if (r + p <= z) {
                frac = 0.0f;
            } else if (r <= fabsf(z - p)) {
                frac = (p > z) ? 1.0f : 0.0f;
            } else {
                float arg = (r2 + z2 - p2) / (2.0f * r * z);
                if (arg >= 1.0f) {
                    frac = 0.0f;
                } else if (arg <= -1.0f) {
                    frac = 1.0f;
                } else {
                    frac = acosf(arg) / 3.14159265358979323846f;
                }
            }

            blocked += I_r * frac * 2.0f * r * dr;
        }

        model_flux = 1.0f - blocked / omega;
    }

    // --- EB dilution ---
    float fr_eb = eb_fluxratios[j];
    float F_EB  = (fr_eb < 1.0f) ? fr_eb / (1.0f - fr_eb) : 1e6f;
    float fr_comp = comp_fr[j];
    float F_comp  = (fr_comp > 0.0f) ? fr_comp / (1.0f - fr_comp) : 0.0f;

    float secdepth = 0.0f;

    if (companion_is_host) {
        float ratio = F_EB / fmaxf(F_comp, 1e-30f);
        model_flux = (model_flux + ratio) / (1.0f + ratio);
        float F_dilute = 1.0f / (F_comp + F_EB);
        model_flux = (model_flux + F_dilute) / (1.0f + F_dilute);
        secdepth = fr_eb / (1.0f + F_dilute);
    } else {
        model_flux = (model_flux + F_EB) / (1.0f + F_EB);
        float F_dilute = F_comp / (1.0f + F_EB);
        model_flux = (model_flux + F_dilute) / (1.0f + F_dilute);
        secdepth = (F_dilute > 0.0f) ? fr_eb / (1.0f + F_dilute) : fr_eb;
    }

    // Store secdepth (only need one thread per draw to write it)
    if (i == 0) {
        secdepth_out[j] = secdepth;
    }

    // --- Chi2 contribution ---
    double diff = flux[i] - (double)model_flux;
    atomicAdd(&chi2[j], diff * diff);
}
"""


# ================================================================== #
# Compile kernels lazily
# ================================================================== #

_kernels = {}


def _get_kernel(name, source, func_name):
    """Lazily compile and cache a RawKernel."""
    if name not in _kernels:
        _kernels[name] = cp.RawKernel(source, func_name)
    return _kernels[name]


# ================================================================== #
# Public API
# ================================================================== #

def batch_transit_lnL(time, flux, sigma, ks, P_orbs, incs_rad,
                      a_Rs_arr, eccs, ws_rad, u1, u2,
                      companion_fluxratio, companion_is_host):
    """Compute transit log-likelihood for N parameter draws on GPU.

    Returns float64 numpy array of lnL, shape (N,).
    """
    N = len(ks)
    T = len(time)

    # Transfer to GPU
    d_time = cp.asarray(time, dtype=cp.float64)
    d_flux = cp.asarray(flux, dtype=cp.float64)
    d_ks = cp.asarray(ks, dtype=cp.float32)
    d_P_orbs = cp.asarray(P_orbs, dtype=cp.float32)
    d_incs = cp.asarray(incs_rad, dtype=cp.float32)
    d_a_Rs = cp.asarray(a_Rs_arr, dtype=cp.float32)
    d_eccs = cp.asarray(eccs, dtype=cp.float32)
    d_ws = cp.asarray(ws_rad, dtype=cp.float32)
    d_comp_fr = cp.asarray(companion_fluxratio, dtype=cp.float32)

    d_chi2 = cp.zeros(N, dtype=cp.float64)

    is_host = 1 if companion_is_host else 0

    # Launch transit chi2 kernel
    block = 256
    total_threads = N * T
    grid = (total_threads + block - 1) // block

    kernel = _get_kernel('transit_chi2', _TRANSIT_CHI2_KERNEL, 'transit_chi2_kernel')
    kernel((grid,), (block,), (
        d_time, d_flux,
        d_ks, d_P_orbs, d_incs, d_a_Rs, d_eccs, d_ws,
        d_comp_fr, np.int32(is_host),
        np.float32(u1), np.float32(u2),
        np.int32(N), np.int32(T),
        d_chi2
    ))

    # Launch chi2 -> lnL kernel
    d_lnL = cp.empty(N, dtype=cp.float64)
    grid2 = (N + block - 1) // block
    lnl_kernel = _get_kernel('chi2_to_lnl', _CHI2_TO_LNL_KERNEL, 'chi2_to_lnl_kernel')
    lnl_kernel((grid2,), (block,), (
        d_chi2, np.float64(sigma), np.int32(T), np.int32(N), d_lnL
    ))

    return cp.asnumpy(d_lnL)


def batch_eb_transit_lnL(time, flux, sigma, ks, eb_fluxratios, P_orbs,
                         incs_rad, a_Rs_arr, eccs, ws_rad, u1, u2,
                         companion_fluxratio, companion_is_host):
    """Compute EB transit log-likelihood for N parameter draws on GPU.

    Returns float64 numpy array of lnL, shape (N,).
    Includes secondary eclipse rejection (lnL = -inf where secdepth >= 1.5*sigma).
    """
    N = len(ks)
    T = len(time)

    d_time = cp.asarray(time, dtype=cp.float64)
    d_flux = cp.asarray(flux, dtype=cp.float64)
    d_ks = cp.asarray(ks, dtype=cp.float32)
    d_eb_fr = cp.asarray(eb_fluxratios, dtype=cp.float32)
    d_P_orbs = cp.asarray(P_orbs, dtype=cp.float32)
    d_incs = cp.asarray(incs_rad, dtype=cp.float32)
    d_a_Rs = cp.asarray(a_Rs_arr, dtype=cp.float32)
    d_eccs = cp.asarray(eccs, dtype=cp.float32)
    d_ws = cp.asarray(ws_rad, dtype=cp.float32)
    d_comp_fr = cp.asarray(companion_fluxratio, dtype=cp.float32)

    d_chi2 = cp.zeros(N, dtype=cp.float64)
    d_secdepth = cp.zeros(N, dtype=cp.float32)

    is_host = 1 if companion_is_host else 0

    block = 256
    total_threads = N * T
    grid = (total_threads + block - 1) // block

    kernel = _get_kernel('eb_transit_chi2', _EB_TRANSIT_CHI2_KERNEL, 'eb_transit_chi2_kernel')
    kernel((grid,), (block,), (
        d_time, d_flux,
        d_ks, d_eb_fr, d_P_orbs, d_incs, d_a_Rs, d_eccs, d_ws,
        d_comp_fr, np.int32(is_host),
        np.float32(u1), np.float32(u2),
        np.int32(N), np.int32(T),
        d_chi2, d_secdepth
    ))

    # chi2 -> lnL
    d_lnL = cp.empty(N, dtype=cp.float64)
    grid2 = (N + block - 1) // block
    lnl_kernel = _get_kernel('chi2_to_lnl', _CHI2_TO_LNL_KERNEL, 'chi2_to_lnl_kernel')
    lnl_kernel((grid2,), (block,), (
        d_chi2, np.float64(sigma), np.int32(T), np.int32(N), d_lnL
    ))

    # Secondary eclipse rejection on CPU
    lnL = cp.asnumpy(d_lnL)
    secdepth = cp.asnumpy(d_secdepth)
    lnL[secdepth >= 1.5 * sigma] = -np.inf

    return lnL


def is_available():
    """True if CUDA GPU is available."""
    if not _CUDA_AVAILABLE:
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception as e:
        logging.getLogger(__name__).warning("CUDA present but unusable for FPP: %s", e)
        return False


def get_info():
    """Returns string like 'NVIDIA RTX 4070 (12.0 GB)'."""
    if not is_available():
        return None
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        mem = cp.cuda.runtime.memGetInfo()[1] / (1024**3)
        return f"{name} ({mem:.1f} GB)"
    except Exception:
        return "CUDA GPU (unknown)"
