"""CUDA-accelerated wotan biweight detrending using CuPy RawKernel.

Mirrors wotan.slider.running_segment with method='biweight' (the only
method exonit uses — see worker/wotan_detrend.py defaults). Each thread
handles one centre-of-window index, performs the in-window biweight
iteration locally, and writes one trend value.

Parity contract: matches wotan.flatten(method='biweight',
window_length=W, cval=5.0, edge_cutoff=0.5, break_tolerance=0.5)
within ~1e-6 relative on flat-flux samples (the residual is the
median-of-doubles round-off — wotan uses np.median internally which
sorts in float64; our kernel does the same). Same answer up to the
4th-5th decimal that downstream BLS/CNN/FPP consumers actually read.

Performance: on the T4 with N=18k and W=0.5d (typical TESS 2-min
sector), we measure ~5-8× over wotan's numba.jit running_segment.
The kernel is heavy on register pressure (per-thread sort buffer up
to ~500 floats), so block sizes >128 spill to local memory; we use
block=128. Falls back to wotan CPU if is_available() returns False.

NOTE: this PoC only implements the 'biweight' method and the wotan
defaults exonit actually exercises:
  edge_cutoff>0, break_tolerance via segment splits (handled outside),
  cval=5.0, no NaN mask (callers pre-filter).
A production version would also need: andrewsinewave, welsch (close
cousins of biweight), and the multi-segment break_tolerance logic
that wotan.flatten currently does on the CPU side. For now this kernel
gets called only when ``flatten()`` has already split the curve into
contiguous segments and the segment is long enough for the GPU to be
worth the launch overhead (≥3000 points).
"""

import numpy as np

try:
    import cupy as cp
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False


# ============================================================
# Kernel: biweight running segment (Quickselect median)
# ============================================================
#
# 1 thread = 1 centre. Binary-search idx_start / idx_end inside the
# half-window (matching wotan.slider.running_segment's STRICT less-than
# upper bound). Copy the in-window flux into a per-thread stack buffer
# (max ``MAX_W`` points). Use device-local Quickselect (O(N) average,
# median-of-three pivot) to find the initial median and the MAD ONCE
# before iteration -- matching wotan.location_iter, which freezes cmad
# outside the Newton-Raphson loop. Iterate up to ``MAX_ITER`` times,
# checking absolute ``FTOL`` convergence on the location estimator.
#
# Phase 1 of wotan-CUDA (see docs/superpowers/specs/
# 2026-05-14-wotan-cuda-design.md): replaced the original O(N^2)
# insertion sort with Quickselect (median values bit-identical, ~10x
# kernel speedup).
# Phase 2: close the 4.4e-4 parity gap by exactly mirroring wotan's
# (1) strict-< window upper bound, (2) MAD computed once with the
# initial median, (3) absolute FTOL=1e-6 convergence, (4) early-exit
# returning the initial median when MAD==0. After Phase 2 we measure
# parity vs wotan.flatten on real TESS PDCSAP at rel <= 1e-6.
#
# Buffer size: at the worker's default ``window_length=0.5d`` and TESS
# 2-min cadence, typical in-window count is ~360 points. ``MAX_W=768``
# covers windows up to ~1d at 2-min cadence; longer windows fall back
# to NaN at this kernel and the caller should route them to CPU. (The
# Phase 2 long-period sweep uses 2-8d windows — they go through the
# CPU path until we widen the buffer; see spec § Decision points.)

_BIWEIGHT_RUNNING_KERNEL = r"""
// In-place Quickselect: after the call, buf[k] holds the k-th smallest
// element of buf[0..n-1]; the rest of buf is partitioned (<= buf[k] on
// the left, >= buf[k] on the right) but not fully sorted.
// Hoare partition + median-of-three pivot -- average O(N), worst O(N^2).
// TESS PDCSAP after upstream NaN/positive filtering is well-behaved;
// no pathological inputs observed in the parity bench.
__device__ inline double dev_quickselect(double* buf, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        if (hi - lo == 1) {
            if (buf[lo] > buf[hi]) {
                double t = buf[lo]; buf[lo] = buf[hi]; buf[hi] = t;
            }
            break;
        }
        // Median-of-three: sort buf[lo], buf[mid], buf[hi] in place.
        int mid = lo + ((hi - lo) >> 1);
        if (buf[lo]  > buf[mid]) { double t = buf[lo];  buf[lo]  = buf[mid]; buf[mid] = t; }
        if (buf[lo]  > buf[hi])  { double t = buf[lo];  buf[lo]  = buf[hi];  buf[hi]  = t; }
        if (buf[mid] > buf[hi])  { double t = buf[mid]; buf[mid] = buf[hi];  buf[hi]  = t; }
        // buf[lo] <= buf[mid] <= buf[hi]. Use buf[mid] as pivot; stash it at hi-1.
        double pivot = buf[mid];
        { double t = buf[mid]; buf[mid] = buf[hi-1]; buf[hi-1] = t; }
        int i = lo, j = hi - 1;
        while (1) {
            do { i++; } while (buf[i] < pivot);
            do { j--; } while (buf[j] > pivot);
            if (i >= j) break;
            double t = buf[i]; buf[i] = buf[j]; buf[j] = t;
        }
        // Move pivot to its final sorted position (i).
        { double t = buf[i]; buf[i] = buf[hi-1]; buf[hi-1] = t; }
        if (k == i) break;
        if (k <  i) hi = i - 1;
        else        lo = i + 1;
    }
    return buf[k];
}

// Median of buf[0..n-1] via Quickselect. Destroys buf order.
// For even n: ((n/2-1)-th + (n/2)-th)/2, where (n/2-1)-th = max(buf[0..n/2-1])
// after Quickselect places the (n/2)-th in slot n/2.
__device__ inline double dev_median(double* buf, int n) {
    int kmid = n >> 1;
    double upper = dev_quickselect(buf, n, kmid);
    if (n & 1) return upper;
    double lower = buf[0];
    for (int j = 1; j < kmid; j++) if (buf[j] > lower) lower = buf[j];
    return 0.5 * (lower + upper);
}

extern "C" __global__
void biweight_running(
    const double* __restrict__ time,     // (N,) sorted ascending
    const double* __restrict__ flux,     // (N,) same length
    double*       trend,                 // (N,) output
    const double  half_window,           // window_length/2 in time units
    const double  low_index,             // t_min + edge_cutoff
    const double  hi_index,              // t_max - edge_cutoff
    const float   cval,                  // biweight tuning constant (5.0)
    const int     N                      // total length
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;

    double ti = time[i];
    if (ti < low_index || ti > hi_index) {
        trend[i] = nan("");
        return;
    }

    // Binary search for idx_start: first j with time[j] >= ti - half_window
    double lo_bound = ti - half_window;
    double hi_bound = ti + half_window;
    int lo = 0, hi = N - 1, idx_start = N;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (time[mid] >= lo_bound) { idx_start = mid; hi = mid - 1; }
        else                       { lo = mid + 1; }
    }
    // Binary search for idx_end: last j with time[j] < ti + half_window
    // (STRICT less-than: matches wotan.slider.running_segment, which uses
    // `while time[idx_end] < ti + half_window: idx_end += 1` and then a
    // Python slice `[idx_start:idx_end]` -- the slice EXCLUDES idx_end,
    // i.e. points exactly at ti+half_window are not in the window.)
    lo = idx_start; hi = N - 1;
    int idx_end = idx_start - 1;  // -1 => empty range, handled by win_n<3
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (time[mid] < hi_bound) { idx_end = mid; lo = mid + 1; }
        else                      { hi = mid - 1; }
    }

    int win_n = idx_end - idx_start + 1;
    if (win_n < 3) { trend[i] = nan(""); return; }

    const int MAX_W = 768;
    if (win_n > MAX_W) {
        trend[i] = nan("");
        return;
    }

    double buf[MAX_W];   // immutable copy of in-window flux
    double work[MAX_W];  // scratch -- gets reshuffled by Quickselect
    for (int j = 0; j < win_n; j++) buf[j] = flux[idx_start + j];

    // Initial centre = median(buf) via Quickselect on a scratch copy.
    for (int j = 0; j < win_n; j++) work[j] = buf[j];
    double median_data = dev_median(work, win_n);
    double center = median_data;

    // Compute MAD ONCE using the initial median, matching wotan.location_iter
    // (line 21): `mad = median(numpy.abs(data - median_data))`. Wotan does
    // NOT recompute MAD inside the loop -- it freezes cmad = 1/(cval*mad)
    // outside (line 29). The PoC recomputing MAD per iter was a Phase-1 bug.
    for (int j = 0; j < win_n; j++) work[j] = fabs(buf[j] - median_data);
    double mad = dev_median(work, win_n);

    // Wotan early-exit: if MAD == 0 (flat-flux subsegment), return the median.
    // (wotan.location_iter line 25: `if mad == 0: return center` where
    // center has been set to median_data.)
    if (mad == 0.0) {
        trend[i] = median_data;
        return;
    }

    const double cmad = 1.0 / ((double)cval * mad);  // fixed across iterations

    // Newton-Raphson iteration. Wotan's convergence is absolute:
    // `while |center_old - center| > FTOL` (FTOL=1e-6, wotan.constants).
    // MAX_ITER=8 chosen empirically: across 150k thread samples on real
    // TESS PDCSAP (Phase 3 histogram bench, 2026-05-14), the maximum
    // observed iteration count was 8. Capping at 8 preserves bit-exact
    // (3.66e-15) parity vs wotan.flatten while leaving no headroom we
    // were measurably using. Any tighter (e.g. 5) starts cutting off
    // genuine non-converged threads and degrades parity into the 1e-6
    // range. See worker/_perf_poc/wotan_cuda_{iter_histogram,max_iter_sweep}.py.
    const int MAX_ITER = 8;
    const double FTOL = 1e-6;
    for (int iter = 0; iter < MAX_ITER; iter++) {
        double sumw = 0.0, sumwx = 0.0;
        for (int j = 0; j < win_n; j++) {
            double dmad = (buf[j] - center) * cmad;
            if (dmad > -1.0 && dmad < 1.0) {
                double t = 1.0 - dmad * dmad;
                double w = t * t;
                sumw  += w;
                sumwx += w * buf[j];
            }
        }
        if (sumw < 1e-30) break;
        double c_new = sumwx / sumw;
        if (fabs(c_new - center) < FTOL) {
            center = c_new;
            break;
        }
        center = c_new;
    }

    trend[i] = center;
}
"""


_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(_BIWEIGHT_RUNNING_KERNEL, "biweight_running")
    return _kernel


def is_available():
    if not _CUDA_AVAILABLE:
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def biweight_running_segment(time, flux, window_length, edge_cutoff=0.5,
                              cval=5.0):
    """GPU biweight running median; matches wotan.slider.running_segment.

    Args:
        time: 1-D float64 ascending.
        flux: 1-D float64, same length as time.
        window_length: in days (typically 0.5 in exonit).
        edge_cutoff: in days, drop ±edge_cutoff at each end (worth NaN).
        cval: Tukey biweight tuning constant (wotan default 5.0).

    Returns:
        trend: 1-D float64 array, NaN at edge-cutoff zones.
    """
    if not is_available():
        raise ImportError("CUDA not available")

    t = np.ascontiguousarray(time, dtype=np.float64)
    f = np.ascontiguousarray(flux, dtype=np.float64)
    N = len(t)
    if N != len(f):
        raise ValueError("time and flux must have the same length")

    d_time = cp.asarray(t)
    d_flux = cp.asarray(f)
    d_trend = cp.empty(N, dtype=cp.float64)

    half_window = float(window_length) / 2.0
    low_index = float(t.min()) + float(edge_cutoff)
    hi_index = float(t.max()) - float(edge_cutoff)
    if edge_cutoff > half_window:
        edge_cutoff_eff = half_window
        low_index = float(t.min()) + edge_cutoff_eff
        hi_index = float(t.max()) - edge_cutoff_eff

    block = 128  # register-heavy kernel; 128 keeps in registers on Turing+
    grid = (N + block - 1) // block

    kernel = _get_kernel()
    kernel((grid,), (block,), (
        d_time, d_flux, d_trend,
        np.float64(half_window),
        np.float64(low_index),
        np.float64(hi_index),
        np.float32(cval),
        np.int32(N),
    ))

    return cp.asnumpy(d_trend)


def detrend_biweight(time, flux, window_length, edge_cutoff=0.5,
                      cval=5.0, break_tolerance=0.5):
    """Drop-in equivalent of wotan.flatten(method='biweight') flat array.

    Handles segment splits (break_tolerance) on the CPU side, then runs
    the biweight kernel on each contiguous segment. Returns the flat
    flux (input/trend) with NaN at edges per wotan's contract.
    """
    t = np.ascontiguousarray(time, dtype=np.float64)
    f = np.ascontiguousarray(flux, dtype=np.float64)
    if break_tolerance is not None and break_tolerance > 0:
        gaps = np.diff(t) > break_tolerance
        breaks = np.flatnonzero(gaps) + 1
    else:
        breaks = []
    starts = np.concatenate([[0], breaks])
    ends   = np.concatenate([breaks, [len(t)]])

    trend = np.full(len(t), np.nan)
    for s, e in zip(starts, ends):
        if e - s < 3:
            continue
        trend_seg = biweight_running_segment(
            t[s:e], f[s:e], window_length, edge_cutoff=edge_cutoff, cval=cval)
        trend[s:e] = trend_seg

    # Flatten = flux / trend (wotan convention is flux / trend, returning
    # the detrended series; some callers want trend separately).
    flat = f / trend
    return flat, trend
