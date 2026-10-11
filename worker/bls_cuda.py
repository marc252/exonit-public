"""CUDA-accelerated BLS (Box Least Squares) transit search using CuPy.

Parallelizes across trial periods on the GPU. Each period's phase-fold,
binning, and sliding window search runs as a CUDA kernel.

Performance: ~15-50x faster than CPU numpy BLS on RTX 4070.

This module does NOT fall back by itself: it imports cleanly without CuPy
(CUDA_AVAILABLE=False) and the bls_gpu* entry points raise ImportError if
called then. Backend selection/fallback lives in bls.py; probe with
is_cuda_available().

``bls_gpu_batch`` (the injection-recovery batch path) defaults to a fused
f32 kernel (spec ``docs/superpowers/specs/2026-07-11-gpu-bls-f32-unified-
kernel.md``, Phase 2 — a CUDA port of the Metal design validated in
production on Apple Silicon, ``bls_metal_batch.py``). Consumer RTX GPUs run
f64 at 1/64 of f32 throughput, so the previous all-float64
``_BLS_BATCH_FUSED_KERNEL`` (kept below as ``_bls_gpu_batch_f64``, the
one-release escape hatch behind env ``EXONIT_BLS_CUDA_F64=1``) was leaving
most of the GPU's compute on the table. ``bls_gpu`` / ``bls_gpu_two_pass``
(primary BLS, single-flux) are unchanged — that port is spec Phase 3.

Nvidiarack A/B bench (2026-07-12, RTX 5060 Ti) found the f32 path regresses
past ~20-40k points: the f32 kernel folds on the CPU (single-threaded numpy,
per the Metal design), while the legacy f64 kernel folds ON the GPU. On
Metal the baseline was numba-CPU so CPU-fold always won; on CUDA the f64
incumbent already GPU-folds, so at large N the CPU fold cost dominates and
loses (catastrophically at 80k points — see ``_F32_MAX_POINTS_DEFAULT``
below for the measured crossover table). ``bls_gpu_batch`` therefore also
routes giant targets (``len(time) > _f32_max_points()``) to the f64 kernel,
env-overridable via ``EXONIT_BLS_CUDA_F32_MAX_PTS``.

Usage:
    from bls_cuda import bls_gpu_two_pass, is_cuda_available
    if is_cuda_available():
        result = bls_gpu_two_pass(time, flux, period_min=0.5, period_max=50.0)
"""

import logging
import os
import threading
import numpy as np

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


def _env_flag(name):
    """True iff env var ``name`` is set to a truthy string ("1"/"true",
    case-insensitive). Used for the ``EXONIT_BLS_CUDA_F64`` escape hatch."""
    return os.environ.get(name, "").strip().lower() in ("1", "true")


# Raw CUDA kernel for phase-fold + bin (one thread per period)
_PHASE_FOLD_BIN_KERNEL = """
extern "C" __global__
void phase_fold_bin(
    const double* time_rel,     // input time array (N points)
    const double* flux,         // input flux array (N points)
    const double* periods,      // trial periods (n_periods)
    double* bin_sums,           // output: (n_periods, n_bins)
    int* bin_counts,            // output: (n_periods, n_bins)
    int N,                      // number of data points
    int n_bins,                 // number of phase bins
    int n_periods               // number of trial periods
) {
    int pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= n_periods) return;

    double period = periods[pid];
    int offset = pid * n_bins;

    // Initialize bins
    for (int b = 0; b < n_bins; b++) {
        bin_sums[offset + b] = 0.0;
        bin_counts[offset + b] = 0;
    }

    // Phase-fold and bin all data points for this period
    for (int i = 0; i < N; i++) {
        double phase = fmod(time_rel[i] / period, 1.0);
        if (phase < 0.0) phase += 1.0;
        int bin_idx = (int)(phase * n_bins);
        if (bin_idx >= n_bins) bin_idx = n_bins - 1;
        bin_sums[offset + bin_idx] += flux[i];
        bin_counts[offset + bin_idx] += 1;
    }
}
"""

_SLIDING_WINDOW_KERNEL = """
extern "C" __global__
void sliding_window(
    const double* bin_sums,     // (n_periods, n_bins)
    const int* bin_counts,      // (n_periods, n_bins)
    const int* dur_bins,        // trial durations in bins
    double* best_powers,        // output: best SR per period
    double* best_depths,        // output: best depth per period
    int* best_dur_idx,          // output: best duration index
    int* best_phase_idx,        // output: best phase bin
    double total_sum,           // sum of all flux
    int N,                      // total data points
    int n_bins,
    int n_periods,
    int n_dur_steps
) {
    int pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= n_periods) return;

    int offset = pid * n_bins;
    double best_sr = -1e30;
    double best_dep = 0.0;
    int best_d = 0;
    int best_p = 0;

    // Build cumulative sums (wrap-around)
    // Use shared memory would be better but this is simpler
    double cum_sums[512];   // max n_bins * 2
    int cum_counts[512];

    // Extended arrays (double length for wrap-around)
    cum_sums[0] = bin_sums[offset];
    cum_counts[0] = bin_counts[offset];
    for (int b = 1; b < n_bins; b++) {
        cum_sums[b] = cum_sums[b-1] + bin_sums[offset + b];
        cum_counts[b] = cum_counts[b-1] + bin_counts[offset + b];
    }
    for (int b = 0; b < n_bins; b++) {
        cum_sums[n_bins + b] = cum_sums[n_bins - 1 + b] + bin_sums[offset + b];
        cum_counts[n_bins + b] = cum_counts[n_bins - 1 + b] + bin_counts[offset + b];
    }

    for (int di = 0; di < n_dur_steps; di++) {
        int width = dur_bins[di];

        for (int p = 0; p < n_bins; p++) {
            int end_idx = p + width;
            double win_sum, win_count_d;
            int win_count;

            if (p == 0) {
                win_sum = cum_sums[width - 1];
                win_count = cum_counts[width - 1];
            } else {
                win_sum = cum_sums[p + width - 1] - cum_sums[p - 1];
                win_count = cum_counts[p + width - 1] - cum_counts[p - 1];
            }

            int out_count = N - win_count;
            if (win_count < 3 || out_count < 3) continue;

            double mean_in = win_sum / win_count;
            double mean_out = (total_sum - win_sum) / out_count;
            double depth = mean_out - mean_in;

            if (depth <= 0) continue;

            double sr = depth * depth * win_count * out_count / (win_count + out_count);

            if (sr > best_sr) {
                best_sr = sr;
                best_dep = depth;
                best_d = di;
                best_p = p;
            }
        }
    }

    best_powers[pid] = best_sr;
    best_depths[pid] = best_dep;
    best_dur_idx[pid] = best_d;
    best_phase_idx[pid] = best_p;
}
"""


def bls_gpu(time, flux, period_min=0.5, period_max=50.0, n_periods=10000,
            n_bins=200, n_dur_steps=15, duration_min_frac=0.01,
            duration_max_frac=0.15):
    """CUDA-accelerated BLS transit search.

    Same interface as bls_fast() but runs on GPU.

    Args:
        time: Array of observation times (days).
        flux: Array of normalized flux values.
        period_min/max: Range of trial periods (days).
        n_periods: Number of trial periods.
        n_bins: Number of phase bins (max 256 for CUDA kernel).
        n_dur_steps: Number of trial transit durations.

    Returns:
        dict with keys: period, power, duration, t0, depth, depth_ppm,
        snr, periods, power_spectrum.
    """
    if not CUDA_AVAILABLE:
        raise ImportError("CuPy not available. Install with: pip install cupy-cuda12x")

    # Clamp n_bins for CUDA kernel (static array in kernel)
    n_bins = min(n_bins, 256)

    N = len(time)
    t0_ref = time[0]
    time_rel = time - t0_ref

    # Uniform-in-frequency grid (Kovács 2002 / Ofir 2014 — natural BLS
    # resolution; linear-in-P over-samples long periods and under-samples
    # short ones, which inflates the lower envelope of the power
    # spectrum and crushes the SDE peak-to-noise ratio. See bls_fast.py
    # for the same grid).
    from bls_fast import _frequency_period_grid
    periods = _frequency_period_grid(period_min, period_max, n_periods)
    total_sum = float(np.sum(flux))

    # Compute trial durations in bins
    dur_bins = np.unique(np.linspace(
        max(1, int(duration_min_frac * n_bins)),
        min(n_bins // 3, int(duration_max_frac * n_bins)),
        n_dur_steps
    ).astype(np.int32))
    n_dur_steps_actual = len(dur_bins)

    # Transfer to GPU
    d_time = cp.asarray(time_rel, dtype=cp.float64)
    d_flux = cp.asarray(flux, dtype=cp.float64)
    d_periods = cp.asarray(periods, dtype=cp.float64)
    d_dur_bins = cp.asarray(dur_bins, dtype=cp.int32)

    # Allocate output arrays on GPU
    d_bin_sums = cp.zeros((n_periods, n_bins), dtype=cp.float64)
    d_bin_counts = cp.zeros((n_periods, n_bins), dtype=cp.int32)
    d_best_powers = cp.zeros(n_periods, dtype=cp.float64)
    d_best_depths = cp.zeros(n_periods, dtype=cp.float64)
    d_best_dur_idx = cp.zeros(n_periods, dtype=cp.int32)
    d_best_phase_idx = cp.zeros(n_periods, dtype=cp.int32)

    # Compile and launch phase-fold + bin kernel
    block_size = 256
    grid_size = (n_periods + block_size - 1) // block_size

    phase_fold_kernel = cp.RawKernel(_PHASE_FOLD_BIN_KERNEL, 'phase_fold_bin')
    phase_fold_kernel((grid_size,), (block_size,),
                      (d_time, d_flux, d_periods,
                       d_bin_sums, d_bin_counts,
                       np.int32(N), np.int32(n_bins), np.int32(n_periods)))

    # Launch sliding window kernel
    sliding_kernel = cp.RawKernel(_SLIDING_WINDOW_KERNEL, 'sliding_window')
    sliding_kernel((grid_size,), (block_size,),
                   (d_bin_sums, d_bin_counts, d_dur_bins,
                    d_best_powers, d_best_depths,
                    d_best_dur_idx, d_best_phase_idx,
                    np.float64(total_sum),
                    np.int32(N), np.int32(n_bins), np.int32(n_periods),
                    np.int32(n_dur_steps_actual)))

    # Transfer results back to CPU
    best_powers = cp.asnumpy(d_best_powers)
    best_depths = cp.asnumpy(d_best_depths)
    best_dur_indices = cp.asnumpy(d_best_dur_idx)
    best_phase_indices = cp.asnumpy(d_best_phase_idx)

    # Find best overall period
    best_idx = np.argmax(best_powers)
    best_period = periods[best_idx]
    best_dur_bin = dur_bins[best_dur_indices[best_idx]]
    best_duration = best_dur_bin / n_bins * best_period
    best_phase = best_phase_indices[best_idx] / n_bins

    # SNR as robust SDE — must match bls_fast._compute_sde (median + MAD·1.4826)
    # so that injection_recovery's snr_threshold gate behaves consistently
    # across CPU and GPU paths. Naive (mean, std) is inflated by the peak
    # itself and produces SNRs ~20× lower, which would zero injection recall.
    from bls_fast import _compute_sde
    snr = _compute_sde(best_powers, best_idx)

    return {
        "period": float(best_period),
        "power": float(best_powers[best_idx]),
        "duration": float(best_duration),
        "t0": float(best_phase * best_period + t0_ref),
        "depth": float(best_depths[best_idx]),
        "depth_ppm": float(best_depths[best_idx] * 1e6),
        "snr": snr,
        "periods": periods,
        "power_spectrum": best_powers,
    }


def bls_gpu_two_pass(time, flux, period_min=0.5, period_max=50.0, n_periods=50000,
                     n_bins=200, n_dur_steps=15, duration_min_frac=0.01,
                     duration_max_frac=0.15):
    """Two-pass BLS on CUDA: coarse scan + fine refinement around the peak.

    Mirrors ``bls_fast.bls_two_pass``'s strategy but runs both passes on
    the GPU. Pass 1 scans the full ``[period_min, period_max]`` range
    with a coarse 5000-period grid (on a T4 this is the same cost as
    ~5000 single-pass; SNR/SDE is computed on this coarse spectrum so
    the noise statistics match the canonical full-range scan). Pass 2
    refines around the best period with ``n_periods // 5`` points
    (capped at 10000) on a narrow window — same widths and same logic
    as the CPU version, so the period/duration/depth resolution is
    identical.

    Returns the same dict as ``bls_gpu``. The reported ``snr`` and
    ``power_spectrum`` come from the coarse pass (full range), while
    period/duration/t0/depth come from the fine pass — bit-for-bit
    consistent with ``bls_fast.bls_two_pass``'s contract.

    Empirically ~5× faster than ``bls_fast.bls_two_pass`` on a T4 for
    typical TESS 2-min sectors (N≈18k, span≈27d), with SNR matching to
    within 1% (the residual is the period-grid discretisation, not
    floating-point: bls_fast also has this).
    """
    # Pass 1: coarse scan
    n_coarse = min(5000, n_periods)
    r1 = bls_gpu(time, flux, period_min=period_min, period_max=period_max,
                 n_periods=n_coarse, n_bins=n_bins, n_dur_steps=n_dur_steps,
                 duration_min_frac=duration_min_frac,
                 duration_max_frac=duration_max_frac)

    if r1["snr"] < 3.0:
        # No significant signal — return coarse result, just like bls_two_pass
        return r1

    # Pass 2: refine around the best period
    coarse_step = (period_max - period_min) / n_coarse
    p_best = r1["period"]
    refine_half = max(coarse_step * 10, p_best * 0.02)  # ±2% or ±10 coarse steps
    p_min2 = max(period_min, p_best - refine_half)
    p_max2 = min(period_max, p_best + refine_half)
    n_fine = min(n_periods // 5, 10000)

    r2 = bls_gpu(time, flux, period_min=p_min2, period_max=p_max2,
                 n_periods=n_fine, n_bins=n_bins, n_dur_steps=n_dur_steps,
                 duration_min_frac=duration_min_frac,
                 duration_max_frac=duration_max_frac)

    # Use refined period/t0/duration/depth but report SNR from the coarse
    # full-range scan (SNR must be computed over the full period range for
    # proper SDE — peak vs full noise floor, not vs narrow window).
    r2["snr"] = r1["snr"]
    r2["periods"] = r1["periods"]
    r2["power_spectrum"] = r1["power_spectrum"]
    return r2


# Fused batched BLS kernel (f64 escape hatch — see _bls_gpu_batch_f64 below
# and the module docstring for the f32 path this was superseded by): one
# thread per (period, flux) pair. Each thread independently computes
# phase-fold + bin_sums + sliding window for its (period, flux) — bin_idx is
# recomputed per thread instead of being shared (would require
# ~N×n_periods×4 bytes of shared/global mem). On RTX this is
# memory-bandwidth-bound on the time_rel/flux loads, but the GPU's
# parallelism over n_periods × n_flux (≈450k threads for 5000 periods × 90
# fluxes) more than amortizes the redundant arithmetic. Stack: ~9 KB/thread
# (bin_sums[256] + bin_counts[256] + cum_sums[512] + cum_counts[512]); fits
# in the 512 KB per-thread stack but limits occupancy — acceptable since the
# kernel is compute-bound. All-float64: consumer RTX runs this at 1/64 of f32
# throughput, which is why the f32 kernel below (_BLS_BATCH_FUSED_F32_KERNEL)
# is now the default path (env EXONIT_BLS_CUDA_F64=1 selects this one).
_BLS_BATCH_FUSED_KERNEL = """
extern "C" __global__
void bls_batch_fused(
    const double* time_rel,         // (N,)
    const double* flux_t,           // (N, n_flux) transposed for coalesced reads:
                                    // threads in a warp share period (same i in
                                    // the inner loop) and access consecutive fi
                                    // values, so flux_t[i*n_flux + fi] reads
                                    // 32 doubles in 2 cache-line transactions
                                    // instead of 32 strided transactions.
    const double* periods,          // (n_periods,)
    const int* dur_bins,            // (n_dur_steps,)
    const double* total_sums,       // (n_flux,)
    double* best_powers,            // (n_flux, n_periods)
    double* best_depths,            // (n_flux, n_periods)
    int* best_dur_idx,              // (n_flux, n_periods)
    int* best_phase_idx,            // (n_flux, n_periods)
    int N, int n_bins, int n_periods,
    int n_dur_steps, int n_flux
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total_tasks = n_periods * n_flux;
    if (tid >= total_tasks) return;

    // Threads grouped by period first so warps share one period — that makes
    // time_rel[i] reads broadcast and flux_t[i*n_flux + fi] coalesced.
    int pid = tid / n_flux;
    int fi  = tid % n_flux;
    double period = periods[pid];

    double bin_sums[256];
    int bin_counts[256];
    for (int b = 0; b < n_bins; b++) { bin_sums[b] = 0.0; bin_counts[b] = 0; }

    // Phase-fold + bin (coalesced flux read pattern with transposed layout)
    for (int i = 0; i < N; i++) {
        double phase = fmod(time_rel[i] / period, 1.0);
        if (phase < 0.0) phase += 1.0;
        int bin = (int)(phase * n_bins);
        if (bin >= n_bins) bin = n_bins - 1;
        bin_sums[bin]   += flux_t[(long long)i * n_flux + fi];
        bin_counts[bin] += 1;
    }

    double total_sum = total_sums[fi];

    // Build wrap-around cumulative arrays (same layout as _SLIDING_WINDOW_KERNEL).
    double cum_sums[512];
    int    cum_counts[512];
    cum_sums[0]   = bin_sums[0];
    cum_counts[0] = bin_counts[0];
    for (int b = 1; b < n_bins; b++) {
        cum_sums[b]   = cum_sums[b-1]   + bin_sums[b];
        cum_counts[b] = cum_counts[b-1] + bin_counts[b];
    }
    for (int b = 0; b < n_bins; b++) {
        cum_sums[n_bins + b]   = cum_sums[n_bins - 1 + b]   + bin_sums[b];
        cum_counts[n_bins + b] = cum_counts[n_bins - 1 + b] + bin_counts[b];
    }

    double best_sr  = -1e30;
    double best_dep = 0.0;
    int    best_d   = 0;
    int    best_p   = 0;

    for (int di = 0; di < n_dur_steps; di++) {
        int width = dur_bins[di];
        for (int p = 0; p < n_bins; p++) {
            double win_sum;
            int    win_count;
            if (p == 0) {
                win_sum   = cum_sums[width - 1];
                win_count = cum_counts[width - 1];
            } else {
                win_sum   = cum_sums[p + width - 1]   - cum_sums[p - 1];
                win_count = cum_counts[p + width - 1] - cum_counts[p - 1];
            }
            int out_count = N - win_count;
            if (win_count < 3 || out_count < 3) continue;
            double mean_in  = win_sum / win_count;
            double mean_out = (total_sum - win_sum) / out_count;
            double depth    = mean_out - mean_in;
            if (depth <= 0) continue;
            double sr = depth * depth * win_count * out_count
                       / (double)(win_count + out_count);
            if (sr > best_sr) {
                best_sr = sr; best_dep = depth;
                best_d  = di; best_p   = p;
            }
        }
    }

    long long out_idx = (long long)fi * n_periods + pid;
    best_powers[out_idx]    = best_sr;
    best_depths[out_idx]    = best_dep;
    best_dur_idx[out_idx]   = best_d;
    best_phase_idx[out_idx] = best_p;
}
"""


def _bls_gpu_batch_f64(time, flux_list, period_min=0.5, period_max=50.0,
                       n_periods=10000, n_bins=200, n_dur_steps=15,
                       duration_min_frac=0.01, duration_max_frac=0.15):
    """CUDA-batched BLS over multiple fluxes sharing one time grid (f64
    escape hatch — see the module docstring). Not called directly by
    injection.py; ``bls_gpu_batch`` below routes here when env
    ``EXONIT_BLS_CUDA_F64=1`` is set, or when the f32 kernel's shared-memory
    budget guard trips for the requested ``n_bins``.

    Output identical in shape and content to ``bls_fast_batch``: a list
    of dicts, one per input flux, each with the same keys as ``bls_fast``.

    Used by ``injection.run_injection_recovery`` so the per-period group
    of 90 injection fluxes runs as a single GPU kernel launch instead of
    90 sequential ``bls_gpu`` calls or a numba CPU fallback. The fused
    kernel parallelises over (n_periods × n_flux) — typically several
    hundred thousand threads — saturating the GPU on injection workloads.
    """
    if not CUDA_AVAILABLE:
        raise ImportError(
            "CuPy not available. Install with: pip install cupy-cuda12x")

    # ----- normalise inputs (match bls_fast_batch exactly) -----
    if isinstance(flux_list, np.ndarray) and flux_list.ndim == 2:
        flux_matrix = np.ascontiguousarray(flux_list, dtype=np.float64)
    else:
        if len(flux_list) == 0:
            return []
        flux_matrix = np.ascontiguousarray(
            np.stack([np.asarray(f, dtype=np.float64) for f in flux_list]),
            dtype=np.float64)
    n_flux = flux_matrix.shape[0]
    if n_flux == 0:
        return []

    # Clamp n_bins for the in-kernel static arrays.
    n_bins = min(n_bins, 256)

    time = np.ascontiguousarray(time, dtype=np.float64)
    N = len(time)
    if flux_matrix.shape[1] != N:
        raise ValueError(
            f"flux_matrix shape {flux_matrix.shape} incompatible with "
            f"time length {N}")

    t0_ref = float(time[0])
    time_rel = time - t0_ref

    # Use the same frequency-spaced grid as bls_fast/bls_fast_batch so
    # period sampling matches across CPU and GPU paths.
    from bls_fast import _frequency_period_grid, _compute_sde
    periods = _frequency_period_grid(period_min, period_max, n_periods)

    dur_bins = np.unique(np.linspace(
        max(1, int(duration_min_frac * n_bins)),
        min(n_bins // 3, int(duration_max_frac * n_bins)),
        n_dur_steps
    ).astype(np.int32))
    n_dur_steps_actual = len(dur_bins)

    total_sums = flux_matrix.sum(axis=1).astype(np.float64)

    # ----- transfer to GPU (transpose flux for coalesced kernel reads) -----
    d_time = cp.asarray(time_rel, dtype=cp.float64)
    d_flux_t = cp.asarray(np.ascontiguousarray(flux_matrix.T), dtype=cp.float64)
    d_periods = cp.asarray(periods, dtype=cp.float64)
    d_dur_bins = cp.asarray(dur_bins, dtype=cp.int32)
    d_total_sums = cp.asarray(total_sums, dtype=cp.float64)

    d_best_powers = cp.zeros((n_flux, n_periods), dtype=cp.float64)
    d_best_depths = cp.zeros((n_flux, n_periods), dtype=cp.float64)
    d_best_dur_idx = cp.zeros((n_flux, n_periods), dtype=cp.int32)
    d_best_phase_idx = cp.zeros((n_flux, n_periods), dtype=cp.int32)

    # ----- launch fused kernel -----
    total_tasks = n_periods * n_flux
    block_size = 128  # smaller block → more registers/local mem per thread
    grid_size = (total_tasks + block_size - 1) // block_size

    kernel = cp.RawKernel(_BLS_BATCH_FUSED_KERNEL, "bls_batch_fused")
    kernel((grid_size,), (block_size,),
           (d_time, d_flux_t, d_periods, d_dur_bins, d_total_sums,
            d_best_powers, d_best_depths, d_best_dur_idx, d_best_phase_idx,
            np.int32(N), np.int32(n_bins), np.int32(n_periods),
            np.int32(n_dur_steps_actual), np.int32(n_flux)))

    best_powers = cp.asnumpy(d_best_powers)
    best_depths = cp.asnumpy(d_best_depths)
    best_dur_indices = cp.asnumpy(d_best_dur_idx)
    best_phase_indices = cp.asnumpy(d_best_phase_idx)

    results = []
    for fi in range(n_flux):
        bp = best_powers[fi]
        bd = best_depths[fi]
        bdi = best_dur_indices[fi]
        bpi = best_phase_indices[fi]

        best_idx = int(np.argmax(bp))
        best_period = float(periods[best_idx])
        best_dur_bin = dur_bins[bdi[best_idx]]
        best_duration = float(best_dur_bin) / n_bins * best_period
        best_phase = bpi[best_idx] / n_bins
        snr = _compute_sde(bp, best_idx)

        results.append({
            "period": best_period,
            "power": float(bp[best_idx]),
            "duration": best_duration,
            "t0": float(best_phase * best_period + t0_ref),
            "depth": float(bd[best_idx]),
            "depth_ppm": float(bd[best_idx] * 1e6),
            "snr": snr,
            "periods": periods,
            "power_spectrum": bp,
        })

    return results


# ============================================================================
# f32 fused kernel — spec Phase 2 CUDA port of bls_metal_batch.bls_gpu_batch_v2
# ============================================================================
#
# Same algorithm as the Metal reference (docs/superpowers/specs/
# 2026-07-11-gpu-bls-f32-unified-kernel.md, "Design — one algorithm, two
# backends"), ported to CUDA syntax:
#   1. Fold once per (cell, period) on the CPU in float64 (bit-identical to
#      numba: ph=(t/P)%1, b=int(ph*n_bins), clipped) -> int16 bin indices,
#      shared by all fluxes of the cell (_phase_fold_cpu below).
#   2. Exact per-period bin counts on the CPU, vectorized
#      (_counts_per_period below), uploaded to the GPU.
#   3. One CUDA thread per (flux, period): private per-bin flux sums
#      (n_bins+1 floats) in shared memory, in-place prefix sum, then the
#      (width ascending, phase ascending) candidate loop with strict '>' —
#      sequential inside the thread, so numba's tie-breaking order is
#      preserved by construction (no atomics, no parallel reductions).
#   4. Flux mean-centered in float64 before the f32 cast (depth is
#      shift-invariant; centering crushes f32 accumulation error).
#   5. Host does argmax over periods + SDE in float64, returning
#      bls_fast_batch-identical dicts.
#
# TG_FLUX threads/block (same as the Metal TG_FLUX): the kernel is compiled
# per n_bins (template substitution, cached), mirroring
# bls_metal_batch._get_kernel — a CUDA analogue of Metal's threadgroup
# memory, sized identically (_shared_mem_bytes uses the same formula as
# bls_metal_batch._threadgroup_bytes; only the per-block byte *budget*
# differs: CUDA's default static-shared-memory limit is 48 KB/block vs
# Metal's 32 KB/threadgroup).
TG_FLUX = 32

# Period chunk keeps bin_idx (chunk x N int16) + counts (chunk x n_bins f32)
# bounded: 1000 x 86k x 2B ~ 173 MB worst-case giant. Normal FFI (N~600) is
# ~1.2 MB per 1000 periods. Same default as bls_metal_batch.py.
_PERIOD_CHUNK_DEFAULT = 1000

# CUDA compute capability >= 7.0 (Volta onward — every arch in the fleet,
# Turing/Ampere/Ada/Blackwell) guarantees 48 KB of shared memory per block
# without opting into the (driver-specific) larger dynamic-shared-memory
# carve-out. Staying under this default keeps the kernel launchable on every
# GPU in the fleet without per-arch cudaFuncAttribute calls (CUDA occupancy
# tuning across Ada vs Blackwell is spec OQ4, deferred to on-hardware
# measurement).
_CUDA_SHARED_MEM_BUDGET_BYTES = 48 * 1024


def _shared_mem_bytes(n_bins):
    """Shared memory (bytes) the f32 kernel needs per block for a given
    n_bins: tg_cumcounts (n_bins+1 floats, shared) + tg_sums (TG_FLUX*
    (n_bins+1) floats, one private (n_bins+1)-float row per flux slot in
    the block). Identical formula to bls_metal_batch._threadgroup_bytes —
    same kernel design, same per-thread memory layout, different backend's
    name for the same on-chip scratch memory."""
    return TG_FLUX * (n_bins + 1) * 4 + (n_bins + 1) * 4


def _phase_fold_cpu(time_rel, periods, n_bins, period_chunk=_PERIOD_CHUNK_DEFAULT):
    """Compute per-period bin indices in numpy float64.

    Identical arithmetic to ``bls_metal.py``'s ``_phase_fold_cpu`` (and to
    numba's per-sample fold in ``bls_fast.py``: ``ph=(t/P)%1``,
    ``b=int(ph*n_bins)``, clipped to ``[0, n_bins)``) — duplicated here
    (rather than imported) so this module stays free of any import-time
    dependency on the Metal backend, matching its existing "imports cleanly
    without CuPy" independence.

    Parameters
    ----------
    time_rel : (N,) float64
        Time relative to the first sample (``time - time[0]``).
    periods : (n_periods,) float64
        Trial periods in the same units as ``time_rel``.
    n_bins : int
        Number of phase bins.
    period_chunk : int
        Vectorise over this many periods at a time to bound peak RAM.

    Returns
    -------
    bin_idx : (n_periods, N) int32
        Bin in ``[0, n_bins)`` that each (period, sample) maps to.
    """
    N = time_rel.shape[0]
    n_periods = periods.shape[0]
    bin_idx = np.empty((n_periods, N), dtype=np.int32)
    for s in range(0, n_periods, period_chunk):
        e = min(s + period_chunk, n_periods)
        # phase in [0, 1) per (period, sample) — float64 throughout.
        phase = time_rel[None, :] / periods[s:e, None]
        phase -= np.floor(phase)
        bins = (phase * n_bins).astype(np.int32)
        # Floating-point edge case: phase*n_bins may round to exactly
        # n_bins for samples whose phase is one ULP below 1.0.
        np.clip(bins, 0, n_bins - 1, out=bins)
        bin_idx[s:e] = bins
    return bin_idx


def _counts_per_period(bin_idx_chunk, n_bins):
    """Exact per-period bin counts from int16 bin indices, vectorized.

    Same formula as ``bls_metal_batch._counts_per_period``."""
    P, N = bin_idx_chunk.shape
    flat = bin_idx_chunk.astype(np.int64) + \
        (np.arange(P, dtype=np.int64)[:, None] * n_bins)
    counts = np.bincount(flat.ravel(), minlength=P * n_bins)
    return counts.reshape(P, n_bins).astype(np.float32)


# One CUDA thread per (flux, period). Launch geometry: grid=(n_flux_pad //
# TG_FLUX, n_per, 1), block=(TG_FLUX, 1, 1) — blockIdx.y selects the period
# (one block covers exactly one period, shared across up to TG_FLUX flux
# slots via blockIdx.x/threadIdx.x), the CUDA analogue of the Metal kernel's
# dispatchThreads grid=(n_flux_pad, n_per, 1) / threadgroup=(TG_FLUX, 1, 1).
# n_bins is baked into the source at compile time (@NB@/@TGSUMS@ template
# substitution, see _get_f32_kernel) rather than sized via CUDA's dynamic
# shared-memory API — this mirrors bls_metal_batch._KERNEL_SRC exactly and
# keeps the arithmetic (and therefore the numba tie-breaking order) a
# byte-for-byte port rather than a reimplementation.
_BLS_BATCH_FUSED_F32_KERNEL_SRC = r"""
extern "C" __global__
void bls_inj_batch_f32(
    const short* bin_idx,        // (n_per, N) int16 phase bin per (period, sample)
    const float* counts,         // (n_per, n_bins) exact per-period bin counts
    const float* fluxT,          // (N, n_flux) mean-centered flux, transposed
                                  // for coalesced reads (threads in a warp
                                  // share the period and access consecutive
                                  // flux slots).
    const float* totals,         // (n_flux,) sum of centered flux (~0)
    const int* widths,           // (n_dur,) trial duration widths, in bins
    const int* params,           // [N, n_flux, n_bins, n_per, n_dur]
    float* out_power,            // (n_flux, n_per)
    unsigned int* out_wj,        // (n_flux, n_per) packed (best_w<<16)|best_j
    float* out_depth             // (n_flux, n_per)
) {
    const int f   = blockIdx.x * blockDim.x + threadIdx.x;  // flux slot (padded to TG_FLUX)
    const int p   = blockIdx.y;                              // period index within this chunk
    const int lid = threadIdx.x;

    const int N      = params[0];
    const int n_flux = params[1];
    const int n_bins = params[2];      // runtime, <= @NB@
    const int n_per  = params[3];
    const int n_dur  = params[4];

    __shared__ float tg_cumcounts[@NB@ + 1];
    __shared__ float tg_sums[@TGSUMS@];

    // --- shared per-period cumulative counts (exact: integers in f32) ---
    if (lid == 0) {
        float c = 0.0f;
        tg_cumcounts[0] = 0.0f;
        const float* crow = counts + (long long)p * (long long)n_bins;
        for (int k = 0; k < n_bins; k++) {
            c += crow[k];
            tg_cumcounts[k + 1] = c;
        }
    }
    __syncthreads();

    if (f >= n_flux) return;   // padding threads: after the only barrier

    // --- private per-bin flux sums (sample order == numba order) ---
    float* sums = tg_sums + lid * (@NB@ + 1);
    for (int k = 0; k <= n_bins; k++) sums[k] = 0.0f;

    const short* brow = bin_idx + (long long)p * (long long)N;
    for (int i = 0; i < N; i++) {
        sums[(int)brow[i] + 1] += fluxT[(long long)i * (long long)n_flux + f];
    }
    // in-place prefix sum: sums[k] = sum of bins [0, k)
    for (int k = 1; k <= n_bins; k++) sums[k] += sums[k - 1];

    const float total = totals[f];
    const float Nf = (float)N;

    float best_sr  = -1e30f;
    int   best_w   = 0;
    int   best_j   = 0;
    float best_dep = 0.0f;

    for (int di = 0; di < n_dur; di++) {
        const int w = widths[di];
        for (int j = 0; j < n_bins; j++) {
            const int e = j + w;             // window [j, e), wraps past n_bins
            float wc, ws;
            if (e <= n_bins) {
                wc = tg_cumcounts[e] - tg_cumcounts[j];
                ws = sums[e] - sums[j];
            } else {
                const int e2 = e - n_bins;
                wc = (tg_cumcounts[n_bins] - tg_cumcounts[j]) + tg_cumcounts[e2];
                ws = (sums[n_bins] - sums[j]) + sums[e2];
            }
            const float oc = Nf - wc;
            if (wc < 3.0f || oc < 3.0f) continue;
            const float mean_in  = ws / wc;
            const float mean_out = (total - ws) / oc;
            const float depth = mean_out - mean_in;
            if (depth > 0.0f) {
                const float sr = depth * depth * wc * oc / (wc + oc);
                if (sr > best_sr) {          // strict '>': first (w, j) wins ties
                    best_sr = sr; best_w = w; best_j = j; best_dep = depth;
                }
            }
        }
    }

    const long long o = (long long)f * (long long)n_per + (long long)p;
    out_power[o] = best_sr;
    out_wj[o]    = ((unsigned int)best_w << 16) | ((unsigned int)best_j & 0xFFFFu);
    out_depth[o] = best_dep;
}
"""

_f32_kernel_cache = {}


def _get_f32_kernel(n_bins_cap):
    """Build (or fetch) the f32 kernel compiled for a given n_bins."""
    key = int(n_bins_cap)
    k = _f32_kernel_cache.get(key)
    if k is None:
        src = (_BLS_BATCH_FUSED_F32_KERNEL_SRC
               .replace("@NB@", str(key))
               .replace("@TGSUMS@", str(TG_FLUX * (key + 1))))
        k = cp.RawKernel(src, "bls_inj_batch_f32")
        _f32_kernel_cache[key] = k
    return k


def _bls_gpu_batch_f32(time, flux_list, period_min=0.5, period_max=50.0,
                       n_periods=10000, n_bins=150, n_dur_steps=15,
                       duration_min_frac=0.01, duration_max_frac=0.15,
                       period_chunk=_PERIOD_CHUNK_DEFAULT):
    """f32 fused-kernel batched BLS — CUDA port of
    ``bls_metal_batch.bls_gpu_batch_v2``. Same signature/return contract as
    ``bls_fast_batch``. Not called directly by injection.py — ``bls_gpu_batch``
    below is the public entry point and picks this path by default."""
    # ----- normalise inputs (match bls_fast_batch exactly) -----
    if isinstance(flux_list, np.ndarray) and flux_list.ndim == 2:
        flux_matrix = np.ascontiguousarray(flux_list, dtype=np.float64)
    else:
        if len(flux_list) == 0:
            return []
        flux_matrix = np.ascontiguousarray(
            np.stack([np.asarray(f, dtype=np.float64) for f in flux_list]),
            dtype=np.float64)
    n_flux = flux_matrix.shape[0]
    if n_flux == 0:
        return []

    time = np.ascontiguousarray(time, dtype=np.float64)
    N = len(time)
    if flux_matrix.shape[1] != N:
        raise ValueError(
            f"flux_matrix shape {flux_matrix.shape} incompatible with "
            f"time length {N}")

    t0_ref = float(time[0])
    time_rel = time - t0_ref

    from bls_fast import _frequency_period_grid, _compute_sde
    periods = _frequency_period_grid(period_min, period_max, n_periods)
    dur_bins = np.unique(np.linspace(
        max(1, int(duration_min_frac * n_bins)),
        min(n_bins // 3, int(duration_max_frac * n_bins)),
        n_dur_steps
    ).astype(np.int64))
    n_dur = len(dur_bins)

    # Mean-center in f64 before the f32 cast (depth/sr shift-invariant; kills
    # f32 accumulation error — see the module docstring and bls_metal_batch.py).
    flux_means = flux_matrix.mean(axis=1)
    flux_centered = flux_matrix - flux_means[:, None]
    totals = flux_centered.sum(axis=1)  # ~0 by construction

    fluxT_gpu = cp.asarray(
        np.ascontiguousarray(flux_centered.T, dtype=np.float32))
    totals_gpu = cp.asarray(totals.astype(np.float32))
    widths_gpu = cp.asarray(dur_bins.astype(np.int32))
    kernel = _get_f32_kernel(n_bins)

    n_flux_pad = ((n_flux + TG_FLUX - 1) // TG_FLUX) * TG_FLUX
    grid_x = n_flux_pad // TG_FLUX

    power_chunks, wj_chunks, dep_chunks = [], [], []
    for s in range(0, n_periods, period_chunk):
        e = min(s + period_chunk, n_periods)
        n_per = e - s
        # CPU-only numpy work (no GPU submission here).
        bin_idx = _phase_fold_cpu(time_rel, periods[s:e], n_bins,
                                  period_chunk=n_per).astype(np.int16)
        counts = _counts_per_period(bin_idx, n_bins)

        bin_idx_gpu = cp.asarray(bin_idx)
        counts_gpu = cp.asarray(counts)
        params_gpu = cp.asarray(
            np.array([N, n_flux, n_bins, n_per, n_dur], dtype=np.int32))

        out_power = cp.empty((n_flux, n_per), dtype=cp.float32)
        out_wj = cp.empty((n_flux, n_per), dtype=cp.uint32)
        out_depth = cp.empty((n_flux, n_per), dtype=cp.float32)

        kernel((grid_x, n_per, 1), (TG_FLUX, 1, 1),
               (bin_idx_gpu, counts_gpu, fluxT_gpu, totals_gpu, widths_gpu,
                params_gpu, out_power, out_wj, out_depth))

        power_chunks.append(cp.asnumpy(out_power).astype(np.float64))
        wj_chunks.append(cp.asnumpy(out_wj))
        dep_chunks.append(cp.asnumpy(out_depth).astype(np.float64))

    best_powers = np.concatenate(power_chunks, axis=1)   # (n_flux, n_periods)
    best_wj = np.concatenate(wj_chunks, axis=1)
    best_depths = np.concatenate(dep_chunks, axis=1)
    best_w = (best_wj >> 16).astype(np.float64)
    best_j = (best_wj & 0xFFFF).astype(np.float64)

    results = []
    for fi in range(n_flux):
        spectrum = best_powers[fi]
        best_idx = int(np.argmax(spectrum))
        best_period = float(periods[best_idx])
        snr = _compute_sde(spectrum, best_idx)
        results.append({
            "period": best_period,
            "power": float(spectrum[best_idx]),
            "duration": float(best_w[fi, best_idx] / n_bins * best_period),
            "t0": float(best_j[fi, best_idx] / n_bins * best_period + t0_ref),
            "depth": float(best_depths[fi, best_idx]),
            "depth_ppm": float(best_depths[fi, best_idx] * 1e6),
            "snr": snr,
            "periods": periods,
            "power_spectrum": spectrum,
        })
    return results


# ----------------------------------------------------------------------------
# Size routing — f32 CPU-fold regresses past ~20-40k points (measured
# 2026-07-12, <host> RTX 5060 Ti, contended by 10 live workers: ratios
# reliable, absolute times are not; 90 fluxes, 2000 periods, n_bins=150):
#
#   N_pts   f64_min   f32_min  f64/f32
#     600     0.095     0.039     2.41
#    2000     0.124     0.063     1.95
#    5000     0.171     0.103     1.66
#   10000     0.254     0.210     1.21
#   20000     0.423     0.389     1.09
#   40000     0.777     0.966     0.80
#   80000     1.482     9.939     0.15
#
# Root cause: the f32 path folds on the CPU (single-threaded numpy, per the
# Metal design — bin_idx must be exact-f64-folded and shared across all
# fluxes before upload), while the legacy f64 kernel folds ON the GPU
# (_PHASE_FOLD_BIN_KERNEL). On Metal the baseline was numba-CPU, so
# CPU-fold always won; on CUDA the f64 incumbent already GPU-folds, so past
# the crossover the CPU fold cost dominates and loses -- catastrophically
# at 80k (nonlinear blowup, 40k->80k: 0.97s->9.9s, likely the per-period
# numpy loop in _phase_fold_cpu falling out of cache). 20000 sits just past
# the crossover (1.09x, still a slight f32 edge) with margin before the
# regime flips hard, so giant targets (long-baseline SPOC LCs, N up to
# ~170k) route to the f64 kernel instead. A hybrid (f64 GPU-fold + f32
# per-bin sums in one kernel) or a threaded CPU fold could likely win back
# the 40k+ regime too -- deferred micro-phase, out of scope here.
# ----------------------------------------------------------------------------
_F32_MAX_POINTS_DEFAULT = 20000


def _f32_max_points():
    """Resolve the f32-path point-count ceiling: env override
    (``EXONIT_BLS_CUDA_F32_MAX_PTS``) or ``_F32_MAX_POINTS_DEFAULT``.
    Invalid/empty values fall back to the default rather than raising --
    a config typo must not crash injection, only lose the f32 speedup."""
    raw = os.environ.get("EXONIT_BLS_CUDA_F32_MAX_PTS", "").strip()
    if not raw:
        return _F32_MAX_POINTS_DEFAULT
    try:
        return int(raw)
    except ValueError:
        logging.getLogger(__name__).warning(
            "EXONIT_BLS_CUDA_F32_MAX_PTS=%r is not a valid integer -- "
            "falling back to the default (%d).",
            raw, _F32_MAX_POINTS_DEFAULT)
        return _F32_MAX_POINTS_DEFAULT


# Size-routing is NORMAL per-call behaviour for giant targets, not a
# degradation (unlike the shared-memory budget guard above, which signals a
# misconfiguration) -- injection.py calls bls_gpu_batch ~35x per star (one
# period-cell batch per radius x period grid cell), so a WARNING per call
# would spam the worker log for every giant target's entire injection run.
# Log once per process instead, at INFO. Mirrors the closest existing
# once-only idiom in the codebase (exonit/config.py's "warn loudly, exactly
# once" implicit-SQLite-fallback notice) adapted to a module-level flag
# since this fires from a free function, not a class instantiation.
_size_route_log_lock = threading.Lock()
_size_route_logged = False


def _log_size_route_once(n_pts, threshold):
    global _size_route_logged
    with _size_route_log_lock:
        if _size_route_logged:
            return
        _size_route_logged = True
    logging.getLogger(__name__).info(
        "bls_gpu_batch: N=%d points exceeds the f32 CPU-fold threshold "
        "(%d) -- routing to the f64 GPU-fold kernel for this and every "
        "subsequent oversized call in this process (measured crossover, "
        "2026-07-12 RTX 5060 Ti bench: past this point the f64 kernel's "
        "on-GPU fold beats the f32 kernel's CPU fold). Override with env "
        "EXONIT_BLS_CUDA_F32_MAX_PTS.", n_pts, threshold)


def bls_gpu_batch(time, flux_list, period_min=0.5, period_max=50.0,
                  n_periods=10000, n_bins=200, n_dur_steps=15,
                  duration_min_frac=0.01, duration_max_frac=0.15):
    """CUDA-batched BLS over multiple fluxes sharing one time grid.

    Output identical in shape and content to ``bls_fast_batch``: a list
    of dicts, one per input flux, each with the same keys as ``bls_fast``.

    Used by ``injection.run_injection_recovery`` so the per-period group
    of 90 injection fluxes runs as a single GPU kernel launch instead of
    90 sequential ``bls_gpu`` calls or a numba CPU fallback.

    Backend selection (spec Phase 2, see module docstring): defaults to the
    f32 fused kernel (``_bls_gpu_batch_f32``) — ~64x cheaper per-FLOP than
    f64 on consumer RTX silicon. Falls back to the original all-float64
    kernel (``_bls_gpu_batch_f64``) when, in priority order:
      1. env ``EXONIT_BLS_CUDA_F64`` is ``1``/``true`` (one-release escape
         hatch) — always wins, regardless of ``n_bins``/size.
      2. ``n_bins`` would overflow the f32 kernel's per-block shared-memory
         budget (``_shared_mem_bytes(n_bins) > _CUDA_SHARED_MEM_BUDGET_BYTES``)
         — logged as a WARNING so a misconfigured ``injection_n_bins``
         doesn't silently degrade every star on this host with no signal.
      3. ``len(time)`` exceeds ``_f32_max_points()`` (default
         ``_F32_MAX_POINTS_DEFAULT`` = 20000, env-overridable via
         ``EXONIT_BLS_CUDA_F32_MAX_PTS``) — the f32 path's CPU fold
         regresses past this point (measured 2026-07-12 <host> bench,
         see the crossover table above); logged at INFO once per process
         (not per call — this is normal routing for giant targets, not a
         misconfiguration, and injection.py calls this ~35x per star).
    """
    if not CUDA_AVAILABLE:
        raise ImportError(
            "CuPy not available. Install with: pip install cupy-cuda12x")

    use_f64 = _env_flag("EXONIT_BLS_CUDA_F64")
    if not use_f64:
        budget = _shared_mem_bytes(n_bins)
        if budget > _CUDA_SHARED_MEM_BUDGET_BYTES:
            logging.getLogger(__name__).warning(
                "bls_gpu_batch: n_bins=%d needs %d B of CUDA shared memory "
                "(TG_FLUX=%d) per block, over the %d B default budget -- "
                "falling back to the f64 kernel path for this call. Lower "
                "n_bins/injection_n_bins to use the faster f32 path (the "
                "production default 150 needs %d B, well under budget).",
                n_bins, budget, TG_FLUX, _CUDA_SHARED_MEM_BUDGET_BYTES,
                _shared_mem_bytes(150))
            use_f64 = True

    if not use_f64:
        n_pts = len(time)
        threshold = _f32_max_points()
        if n_pts > threshold:
            _log_size_route_once(n_pts, threshold)
            use_f64 = True

    if use_f64:
        return _bls_gpu_batch_f64(
            time, flux_list, period_min=period_min, period_max=period_max,
            n_periods=n_periods, n_bins=n_bins, n_dur_steps=n_dur_steps,
            duration_min_frac=duration_min_frac,
            duration_max_frac=duration_max_frac)

    return _bls_gpu_batch_f32(
        time, flux_list, period_min=period_min, period_max=period_max,
        n_periods=n_periods, n_bins=n_bins, n_dur_steps=n_dur_steps,
        duration_min_frac=duration_min_frac,
        duration_max_frac=duration_max_frac)


def is_cuda_available():
    """Check if CUDA is available for BLS acceleration."""
    if not CUDA_AVAILABLE:
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception as e:
        # Loud, not silent: cupy installed but runtime broken (driver/libcuda)
        # is the classic "GPU host quietly on CPU" failure.
        logging.getLogger(__name__).warning("CUDA present but unusable for BLS: %s", e)
        return False


def get_gpu_info():
    """Get GPU information string."""
    if not is_cuda_available():
        return None
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        mem = cp.cuda.runtime.memGetInfo()[1] / (1024**3)
        return f"{name} ({mem:.1f} GB)"
    except Exception:
        return "CUDA GPU (unknown)"
