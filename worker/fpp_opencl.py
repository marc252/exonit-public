"""OpenCL GPU backend for FPP transit model evaluation.

Universal fallback for AMD, Intel, and NVIDIA GPUs via PyOpenCL.
Implements the same interface as fpp_cuda.py.

Kernel design: 1 work-item per MC draw (fused per-time-point).
Each work-item loops over T time points internally:
  Kepler (10 iter) -> z -> ring integration (100 rings) -> model -> chi2
chi2 accumulated in float32; final lnL computed on CPU in float64.

Batching: draws processed in batches of BATCH_SIZE=1000 to prevent
iGPU memory pressure on integrated GPUs (matches bls_opencl.py pattern).

Thread-safety (spec 2026-07-24, iteration 2): this module keeps its OWN
``_ctx``/``_queue``/``_program`` globals, separate from ``bls_opencl.py``'s
``_ctx``/``_queue`` -- two independent PyOpenCL contexts on the same
physical device. That does NOT make the two modules independently safe to
run concurrently: on worker.py's architecture, FPP (this module, main
thread) and injection-recovery (``bls_opencl.py``, background thread) can
both submit work to the driver at the same time, and PyOpenCL/the
underlying NEO driver give no stronger guarantee here than MLX did for the
2026-07-12 Metal incident (see ``opencl_serial.py``'s docstring for the
full incident). Every REAL OpenCL submission below (context/program build,
buffer upload used by a kernel launch, ``enqueue_nd_range_kernel``,
``enqueue_copy`` read-back) therefore goes through
``opencl_serial.opencl_guard()`` -- the SAME process-wide ``RLock`` object
``bls_opencl.py`` uses, not a second independent lock (a second lock would
not prevent the cross-module race; see ``opencl_guard()``'s own
docstring). CPU-only numpy work (dtype casts, ``lnL`` post-processing)
stays OUTSIDE every guarded region, matching the Metal/CUDA modules' rule.
"""

import logging
import math
import numpy as np

from opencl_serial import opencl_guard
# Reuse bls_opencl's device-selection helper rather than duplicating it --
# see _init_opencl()'s docstring below (2026-07-24 review it.2, Finding 1)
# for why this module needs the exact same "never silently pick a CPU
# OpenCL device" guarantee bls_opencl.py already has. Safe to import
# unconditionally: bls_opencl.py itself guards `import pyopencl` with its
# own try/except and imports cleanly with or without PyOpenCL installed
# (same as injection.py already does at module scope).
from bls_opencl import _select_gpu_device

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

# ====================================================================== #
# OpenCL kernel source
# ====================================================================== #

_KERNEL_SOURCE = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// ---------------------------------------------------------------------------
// Transit chi2 kernel: 1 work-item per MC draw
// ---------------------------------------------------------------------------
__kernel void transit_chi2_kernel(
    __global const float* time_arr,
    __global const float* flux_obs,
    const int T,
    __global const float* ks,
    __global const float* P_orbs,
    __global const float* incs,
    __global const float* a_Rs,
    __global const float* eccs,
    __global const float* ws,
    const float u1,
    const float u2,
    __global const float* comp_fr,
    const int comp_is_host,
    const float sigma,
    __global float* chi2_out
) {
    int j = get_global_id(0);

    // Load per-draw parameters
    float k    = ks[j];
    float P    = P_orbs[j];
    float inc  = incs[j];
    float aRs  = a_Rs[j];
    float ecc  = eccs[j];
    float w    = ws[j];
    float fr   = comp_fr[j];

    // Precompute dilution factor
    float F_dilute = 0.0f;
    if (fr > 0.0f) {
        if (comp_is_host) {
            float F_comp = fr / (1.0f - fr);
            F_dilute = 1.0f / F_comp;
        } else {
            F_dilute = fr / (1.0f - fr);
        }
    }

    float cos_i  = cos(inc);
    float two_pi = 2.0f * M_PI_F;
    float ecc2   = ecc * ecc;
    float sqrt1_ecc2 = sqrt(1.0f - ecc2);
    float omega  = 1.0f - u1 / 3.0f - u2 / 6.0f;
    if (omega <= 0.0f) omega = 1.0f;

    float chi2 = 0.0f;

    for (int i = 0; i < T; i++) {
        float t = time_arr[i];

        // Mean anomaly (t0=0, transit at nu+w=0 means M offset by -w)
        float phase = t / P;
        phase = phase - floor(phase);
        float M = two_pi * phase - w;

        // Solve Kepler's equation: 10 Newton iterations
        float E = M;
        for (int iter = 0; iter < 10; iter++) {
            float sinE = sin(E);
            float cosE = cos(E);
            E -= (E - ecc * sinE - M) / (1.0f - ecc * cosE);
        }

        // True anomaly
        float sinE = sin(E);
        float cosE = cos(E);
        float nu;
        if (ecc < 1e-10f) {
            nu = M;
        } else {
            nu = atan2(sqrt1_ecc2 * sinE, cosE - ecc);
        }

        // Projected separation z
        float r_factor = (1.0f - ecc2) / (1.0f + ecc * cos(nu));
        float f    = nu + w;
        float sinf = sin(f);
        float cosf = cos(f);
        float z    = aRs * r_factor * sqrt(sinf * sinf + cos_i * cos_i * cosf * cosf);

        // Ring integration: 100 rings
        float model_flux;
        if (z >= 1.0f + k) {
            model_flux = 1.0f;
        } else if (k <= 0.0f) {
            model_flux = 1.0f;
        } else {
            float dr = 1.0f / 100.0f;
            float blocked = 0.0f;
            float z2 = z * z;
            float k2 = k * k;

            for (int ri = 0; ri < 100; ri++) {
                float r   = ((float)ri + 0.5f) * dr;
                float r2  = r * r;
                float mu  = sqrt(1.0f - r2);
                float omm = 1.0f - mu;
                float I_r = 1.0f - u1 * omm - u2 * omm * omm;

                float frac;
                if (z < 1e-12f) {
                    frac = (r <= k) ? 1.0f : 0.0f;
                } else if (r >= z + k) {
                    frac = 0.0f;
                } else if (r + k <= z) {
                    frac = 0.0f;
                } else if (r <= fabs(z - k)) {
                    frac = (k > z) ? 1.0f : 0.0f;
                } else {
                    float arg = (r2 + z2 - k2) / (2.0f * r * z);
                    if (arg >= 1.0f) {
                        frac = 0.0f;
                    } else if (arg <= -1.0f) {
                        frac = 1.0f;
                    } else {
                        frac = acos(arg) / M_PI_F;
                    }
                }

                blocked += I_r * frac * 2.0f * r * dr;
            }

            model_flux = 1.0f - blocked / omega;
        }

        // Apply companion dilution: (model + F_d) / (1 + F_d)
        if (F_dilute > 0.0f) {
            model_flux = (model_flux + F_dilute) / (1.0f + F_dilute);
        }

        float diff = flux_obs[i] - model_flux;
        chi2 += diff * diff;
    }

    chi2_out[j] = chi2;
}


// ---------------------------------------------------------------------------
// EB transit chi2 kernel: 1 work-item per MC draw, includes secondary eclipse
// ---------------------------------------------------------------------------
__kernel void eb_transit_chi2_kernel(
    __global const float* time_arr,
    __global const float* flux_obs,
    const int T,
    __global const float* ks,
    __global const float* eb_fluxratios,
    __global const float* P_orbs,
    __global const float* incs,
    __global const float* a_Rs,
    __global const float* eccs,
    __global const float* ws,
    const float u1,
    const float u2,
    __global const float* comp_fr,
    const int comp_is_host,
    const float sigma,
    __global float* chi2_out,
    __global float* secdepth_out
) {
    int j = get_global_id(0);

    float k      = ks[j];
    if (fabs(k - 1.0f) < 1e-6f) k *= 0.999f;

    float P      = P_orbs[j];
    float inc    = incs[j];
    float aRs    = a_Rs[j];
    float ecc    = eccs[j];
    float w      = ws[j];
    float fr_eb  = eb_fluxratios[j];
    float fr_comp = comp_fr[j];

    // EB dilution factors
    float F_EB   = (fr_eb < 1.0f) ? (fr_eb / (1.0f - fr_eb)) : 1e6f;
    float F_comp = (fr_comp > 0.0f) ? (fr_comp / (1.0f - fr_comp)) : 0.0f;

    float F_dil1 = 0.0f;  // first dilution step
    float F_dil2 = 0.0f;  // second dilution step
    float secdepth;

    if (comp_is_host) {
        // Step 1: EB diluted by companion
        float ratio = F_EB / max(F_comp, 1e-30f);
        F_dil1  = ratio;
        // Step 2: further dilute by combined
        float F_dilute2 = 1.0f / (F_comp + F_EB);
        F_dil2 = F_dilute2;
        secdepth = fr_eb / (1.0f + F_dilute2);
    } else {
        // Step 1: EB diluted
        F_dil1  = F_EB;
        // Step 2: companion dilution
        float F_dilute2 = F_comp / (1.0f + F_EB);
        F_dil2 = F_dilute2;
        secdepth = (F_dilute2 > 0.0f) ? (fr_eb / (1.0f + F_dilute2)) : fr_eb;
    }

    secdepth_out[j] = secdepth;

    // Mark as rejected if secondary eclipse too deep (1.5 * sigma)
    if (secdepth >= 1.5f * sigma) {
        chi2_out[j] = -1.0f;   // sentinel: rejected
        return;
    }

    float cos_i  = cos(inc);
    float two_pi = 2.0f * M_PI_F;
    float ecc2   = ecc * ecc;
    float sqrt1_ecc2 = sqrt(1.0f - ecc2);
    float omega  = 1.0f - u1 / 3.0f - u2 / 6.0f;
    if (omega <= 0.0f) omega = 1.0f;

    float chi2 = 0.0f;

    for (int i = 0; i < T; i++) {
        float t = time_arr[i];

        float phase = t / P;
        phase = phase - floor(phase);
        float M = two_pi * phase - w;

        float E = M;
        for (int iter = 0; iter < 10; iter++) {
            float sinE = sin(E);
            float cosE = cos(E);
            E -= (E - ecc * sinE - M) / (1.0f - ecc * cosE);
        }

        float sinE = sin(E);
        float cosE = cos(E);
        float nu;
        if (ecc < 1e-10f) {
            nu = M;
        } else {
            nu = atan2(sqrt1_ecc2 * sinE, cosE - ecc);
        }

        float r_factor = (1.0f - ecc2) / (1.0f + ecc * cos(nu));
        float f    = nu + w;
        float sinf = sin(f);
        float cosf = cos(f);
        float z    = aRs * r_factor * sqrt(sinf * sinf + cos_i * cos_i * cosf * cosf);

        // Ring integration: 100 rings
        float model_flux;
        if (z >= 1.0f + k) {
            model_flux = 1.0f;
        } else if (k <= 0.0f) {
            model_flux = 1.0f;
        } else {
            float dr = 1.0f / 100.0f;
            float blocked = 0.0f;
            float z2 = z * z;
            float k2 = k * k;

            for (int ri = 0; ri < 100; ri++) {
                float r   = ((float)ri + 0.5f) * dr;
                float r2  = r * r;
                float mu  = sqrt(1.0f - r2);
                float omm = 1.0f - mu;
                float I_r = 1.0f - u1 * omm - u2 * omm * omm;

                float frac;
                if (z < 1e-12f) {
                    frac = (r <= k) ? 1.0f : 0.0f;
                } else if (r >= z + k) {
                    frac = 0.0f;
                } else if (r + k <= z) {
                    frac = 0.0f;
                } else if (r <= fabs(z - k)) {
                    frac = (k > z) ? 1.0f : 0.0f;
                } else {
                    float arg = (r2 + z2 - k2) / (2.0f * r * z);
                    if (arg >= 1.0f) {
                        frac = 0.0f;
                    } else if (arg <= -1.0f) {
                        frac = 1.0f;
                    } else {
                        frac = acos(arg) / M_PI_F;
                    }
                }

                blocked += I_r * frac * 2.0f * r * dr;
            }

            model_flux = 1.0f - blocked / omega;
        }

        // Two-step EB dilution
        model_flux = (model_flux + F_dil1) / (1.0f + F_dil1);
        if (F_dil2 > 0.0f) {
            model_flux = (model_flux + F_dil2) / (1.0f + F_dil2);
        }

        float diff = flux_obs[i] - model_flux;
        chi2 += diff * diff;
    }

    chi2_out[j] = chi2;
}
"""

# ====================================================================== #
# Lazy context initialization
# ====================================================================== #

_ctx = None
_queue = None
_program = None


def _init_opencl():
    """Initialize OpenCL context and compile kernels. Returns True on success.

    Context/queue creation and program build are submission-adjacent
    operations on shared driver state (same rationale as
    ``bls_opencl._init_opencl``'s docstring) -- guarded by
    ``opencl_serial.opencl_guard()``, the SAME lock ``bls_opencl.py`` uses,
    so this module's init can never race a concurrent ``bls_opencl``
    submission (or vice versa).

    Device selection (2026-07-24 review it.2, Finding 1): this used to call
    ``cl.create_some_context(interactive=False)`` directly, exactly the
    "silent CPU fallback" mode CLAUDE.md #2 forbids and that the previous
    review already made bloquejant for ``bls_opencl.py`` (see that module's
    ``_select_gpu_device``/``_init_opencl`` docstrings) -- but the fix was
    never carried over to this sibling module. This module's own docstring
    ("Universal fallback for AMD, Intel, and NVIDIA GPUs") is exactly the
    multi-vendor-platform scenario ``_select_gpu_device()`` exists to
    disambiguate: with more than one OpenCL ICD installed (a stray CPU
    runtime alongside the Intel NEO GPU platform), ``create_some_context``
    is not guaranteed to prefer the GPU. Reusing
    ``bls_opencl._select_gpu_device()`` (rather than duplicating it here)
    keeps the "pick a GPU device, log+return False if none" contract
    defined in exactly one place -- the two modules would otherwise risk
    drifting apart the same way the Metal kill-switch check once did (see
    ``opencl_serial.py``'s docstring). If no GPU device is found on any
    platform, this returns ``False`` -- there is no
    ``create_some_context()`` fallback here anymore, mirroring
    ``bls_opencl._init_opencl``'s contract exactly.
    """
    global _ctx, _queue, _program
    if _ctx is not None:
        return True
    if not OPENCL_AVAILABLE:
        return False
    try:
        with opencl_guard():
            device = _select_gpu_device()
            if device is None:
                logging.getLogger("exonit").warning(
                    "fpp_opencl: no GPU device available on any platform -- "
                    "OpenCL FPP backend disabled (never silently "
                    "substituting a CPU OpenCL device, CLAUDE.md #2). "
                    "fpp_native's backend chain falls through to the next "
                    "FPP backend (or numba-CPU) via its own explicit path.")
                return False
            _ctx = cl.Context([device])
            _queue = cl.CommandQueue(_ctx)
            _program = cl.Program(_ctx, _KERNEL_SOURCE).build()
        return True
    except Exception as e:
        print(f"  OpenCL init failed: {e}")
        _ctx = None
        _queue = None
        _program = None
        return False


# ====================================================================== #
# Public API
# ====================================================================== #

BATCH_SIZE = 1000


def batch_transit_lnL(time, flux, sigma, ks, P_orbs, incs_rad,
                      a_Rs_arr, eccs, ws_rad, u1, u2,
                      companion_fluxratio, companion_is_host):
    """Evaluate transit log-likelihood for N parameter sets on OpenCL GPU.

    Args:
        time: float64 array of shape (T,) — observation times (days)
        flux: float64 array of shape (T,) — observed normalized flux
        sigma: scalar photometric noise (float)
        ks, P_orbs, incs_rad, a_Rs_arr, eccs, ws_rad: float64 arrays (N,)
        u1, u2: quadratic limb darkening coefficients (scalar)
        companion_fluxratio: float64 array (N,) — companion flux fraction
        companion_is_host: bool — companion dilution direction

    Returns:
        float64 numpy array of shape (N,) containing lnL values.
    """
    if not OPENCL_AVAILABLE:
        raise ImportError("PyOpenCL not available")
    if not _init_opencl():
        raise RuntimeError("Failed to initialize OpenCL")

    N = len(ks)
    T = len(time)

    # Convert to float32 for GPU computation.
    # Time range note: calc_fpp_native passes phase-folded, bin-centred times
    # (output of _phase_fold_bin), spanning roughly [-P/2, P/2] in days.
    # For typical periods (1-30 d) the range is ±15 d at most, which is safe for
    # float32 (epsilon ~1.2e-7 → relative error ~1e-7 at 15 d → 1.5 µs; the
    # orbital phase computation needs ~seconds accuracy, so this is fine).
    # No t0-shift is needed.  If this function is ever called with raw BTJD times
    # (~2000+), subtract time.min() here AND add t_offset to the kernel's epoch.
    time_f32   = np.ascontiguousarray(time, dtype=np.float32)
    flux_f32   = np.ascontiguousarray(flux, dtype=np.float32)
    sigma_f32  = np.float32(sigma)
    u1_f32     = np.float32(u1)
    u2_f32     = np.float32(u2)

    # Log-likelihood normalisation constant — mirror of fpp_native._batch_transit_lnL:
    #   lnL[j] = -0.5*ln2pi - lnsigma - 0.5*chi2/(sigma*sigma)
    # i.e. ONE constant per draw, NOT multiplied by T.  This is non-standard
    # (the textbook form is T*(-0.5*ln2pi - lnsigma)) but it IS the production
    # reference.  Do NOT "correct" this to the ×T form — see STATUS 2026-06-10.
    ln2pi    = math.log(2.0 * math.pi)
    lnsigma  = math.log(float(sigma))
    lnL_const = -0.5 * ln2pi - lnsigma   # ONE constant per draw (matches fpp_native)
    sigma2   = float(sigma) * float(sigma)

    mf = cl.mem_flags

    # Upload time/flux arrays once (shared across all batches). Buffer
    # creation with COPY_HOST_PTR is a real driver submission (same rule
    # bls_opencl.py follows) -- guarded, and shared with bls_opencl.py via
    # the SAME opencl_serial.opencl_guard() lock.
    with opencl_guard():
        d_time = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=time_f32)
        d_flux = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flux_f32)
        kernel = cl.Kernel(_program, 'transit_chi2_kernel')

    lnL_all = np.full(N, -np.inf, dtype=np.float64)

    for batch_start in range(0, N, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, N)
        bn = batch_end - batch_start

        # Slice and convert batch params -- CPU-only numpy work, stays
        # OUTSIDE the guard (same rule as bls_opencl.py's per-chunk fold).
        ks_b    = np.ascontiguousarray(ks[batch_start:batch_end],    dtype=np.float32)
        P_b     = np.ascontiguousarray(P_orbs[batch_start:batch_end], dtype=np.float32)
        inc_b   = np.ascontiguousarray(incs_rad[batch_start:batch_end], dtype=np.float32)
        aRs_b   = np.ascontiguousarray(a_Rs_arr[batch_start:batch_end], dtype=np.float32)
        ecc_b   = np.ascontiguousarray(eccs[batch_start:batch_end],   dtype=np.float32)
        ws_b    = np.ascontiguousarray(ws_rad[batch_start:batch_end],  dtype=np.float32)
        cfr_b   = np.ascontiguousarray(companion_fluxratio[batch_start:batch_end],
                                       dtype=np.float32)
        chi2_buf = np.zeros(bn, dtype=np.float32)

        # Real OpenCL submission for this batch: buffer upload, kernel
        # launch, read-back, synchronous finish -- fully self-contained
        # (own inputs/outputs), so releasing the lock between batches lets
        # a concurrent bls_opencl injection call interleave (mirrors
        # bls_opencl.bls_gpu_batch's per-chunk opencl_guard() pattern).
        with opencl_guard():
            d_ks   = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ks_b)
            d_P    = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=P_b)
            d_inc  = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inc_b)
            d_aRs  = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=aRs_b)
            d_ecc  = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ecc_b)
            d_ws   = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ws_b)
            d_cfr  = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cfr_b)
            d_chi2 = cl.Buffer(_ctx, mf.WRITE_ONLY, chi2_buf.nbytes)

            kernel(
                _queue, (bn,), None,
                d_time, d_flux,
                np.int32(T),
                d_ks, d_P, d_inc, d_aRs, d_ecc, d_ws,
                u1_f32, u2_f32,
                d_cfr,
                np.int32(1 if companion_is_host else 0),
                sigma_f32,
                d_chi2,
            )

            cl.enqueue_copy(_queue, chi2_buf, d_chi2)
            _queue.finish()

        # Convert chi2 -> lnL in float64 on CPU
        # Mirror of fpp_native: lnL = lnL_const - 0.5*chi2/sigma^2  (NOT T*lnL_const)
        chi2_f64 = chi2_buf.astype(np.float64)
        lnL_batch = lnL_const - 0.5 * chi2_f64 / sigma2
        lnL_all[batch_start:batch_end] = lnL_batch

    return lnL_all


def batch_eb_transit_lnL(time, flux, sigma, ks, eb_fluxratios, P_orbs,
                         incs_rad, a_Rs_arr, eccs, ws_rad, u1, u2,
                         companion_fluxratio, companion_is_host):
    """Evaluate EB transit log-likelihood for N parameter sets on OpenCL GPU.

    Includes secondary eclipse rejection: draws where the predicted secondary
    eclipse depth >= 1.5*sigma are assigned lnL = -inf.

    Args:
        Same as batch_transit_lnL, with additional:
        eb_fluxratios: float64 array (N,) — EB secondary flux fraction

    Returns:
        float64 numpy array of shape (N,) containing lnL values.
    """
    if not OPENCL_AVAILABLE:
        raise ImportError("PyOpenCL not available")
    if not _init_opencl():
        raise RuntimeError("Failed to initialize OpenCL")

    N = len(ks)
    T = len(time)

    # Time range: phase-folded ±P/2 days from _phase_fold_bin — safe for float32.
    # See batch_transit_lnL comment for full rationale on the t0-shift question.
    time_f32  = np.ascontiguousarray(time, dtype=np.float32)
    flux_f32  = np.ascontiguousarray(flux, dtype=np.float32)
    sigma_f32 = np.float32(sigma)
    u1_f32    = np.float32(u1)
    u2_f32    = np.float32(u2)

    # Mirror of fpp_native._batch_eb_transit_lnL: ONE constant per draw, NOT ×T.
    # Do NOT "correct" to the textbook ×T form — see STATUS 2026-06-10.
    ln2pi   = math.log(2.0 * math.pi)
    lnsigma = math.log(float(sigma))
    lnL_const = -0.5 * ln2pi - lnsigma   # ONE constant per draw (matches fpp_native)
    sigma2  = float(sigma) * float(sigma)

    mf = cl.mem_flags

    with opencl_guard():
        d_time = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=time_f32)
        d_flux = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flux_f32)
        kernel = cl.Kernel(_program, 'eb_transit_chi2_kernel')

    lnL_all = np.full(N, -np.inf, dtype=np.float64)

    for batch_start in range(0, N, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, N)
        bn = batch_end - batch_start

        # CPU-only numpy work, stays OUTSIDE the guard.
        ks_b    = np.ascontiguousarray(ks[batch_start:batch_end],           dtype=np.float32)
        ebfr_b  = np.ascontiguousarray(eb_fluxratios[batch_start:batch_end], dtype=np.float32)
        P_b     = np.ascontiguousarray(P_orbs[batch_start:batch_end],        dtype=np.float32)
        inc_b   = np.ascontiguousarray(incs_rad[batch_start:batch_end],      dtype=np.float32)
        aRs_b   = np.ascontiguousarray(a_Rs_arr[batch_start:batch_end],      dtype=np.float32)
        ecc_b   = np.ascontiguousarray(eccs[batch_start:batch_end],          dtype=np.float32)
        ws_b    = np.ascontiguousarray(ws_rad[batch_start:batch_end],        dtype=np.float32)
        cfr_b   = np.ascontiguousarray(companion_fluxratio[batch_start:batch_end],
                                       dtype=np.float32)
        chi2_buf    = np.zeros(bn, dtype=np.float32)
        secdepth_buf = np.zeros(bn, dtype=np.float32)

        # Real OpenCL submission for this batch -- guarded, shared lock
        # with bls_opencl.py (see batch_transit_lnL above).
        with opencl_guard():
            d_ks   = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ks_b)
            d_ebfr = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ebfr_b)
            d_P    = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=P_b)
            d_inc  = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inc_b)
            d_aRs  = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=aRs_b)
            d_ecc  = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ecc_b)
            d_ws   = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ws_b)
            d_cfr  = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cfr_b)
            d_chi2     = cl.Buffer(_ctx, mf.WRITE_ONLY, chi2_buf.nbytes)
            d_secdepth = cl.Buffer(_ctx, mf.WRITE_ONLY, secdepth_buf.nbytes)

            kernel(
                _queue, (bn,), None,
                d_time, d_flux,
                np.int32(T),
                d_ks, d_ebfr, d_P, d_inc, d_aRs, d_ecc, d_ws,
                u1_f32, u2_f32,
                d_cfr,
                np.int32(1 if companion_is_host else 0),
                sigma_f32,
                d_chi2,
                d_secdepth,
            )

            cl.enqueue_copy(_queue, chi2_buf, d_chi2)
            cl.enqueue_copy(_queue, secdepth_buf, d_secdepth)
            _queue.finish()

        # Draws with chi2_out == -1.0 are rejected (secondary too deep).
        # Mirror of fpp_native: lnL = lnL_const - 0.5*chi2/sigma^2  (NOT T*lnL_const).
        # Rejected draws (sentinel chi2=-1) are overwritten to -inf unconditionally.
        rejected = chi2_buf < 0.0
        chi2_f64 = chi2_buf.astype(np.float64)
        lnL_batch = lnL_const - 0.5 * chi2_f64 / sigma2
        lnL_batch[rejected] = -np.inf

        lnL_all[batch_start:batch_end] = lnL_batch

    return lnL_all


def is_available():
    """Return True if OpenCL is installed and a context can be created."""
    if not OPENCL_AVAILABLE:
        return False
    return _init_opencl()


def get_info():
    """Return a human-readable string describing the OpenCL device in use."""
    if not OPENCL_AVAILABLE:
        return "OpenCL not available (PyOpenCL not installed)"
    if not _init_opencl():
        return "OpenCL not available (no compatible device found)"
    try:
        device = _ctx.devices[0]
        name    = device.name.strip()
        vendor  = device.vendor.strip()
        mem_mb  = device.global_mem_size // (1024 * 1024)
        return f"{name} ({mem_mb} MB, OpenCL)"
    except Exception as e:
        return f"OpenCL device (info unavailable: {e})"
