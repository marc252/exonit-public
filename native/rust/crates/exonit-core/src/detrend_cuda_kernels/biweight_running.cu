/*
 * Verbatim copy — DO NOT EDIT the kernel body below without also updating
 * the Python source it is copied from.
 *
 * Origin: ~/exonit/worker/wotan_cuda.py, lines 73-233 (the body of
 * the `_BIWEIGHT_RUNNING_KERNEL` string, i.e. everything between the
 * opening `r"""` on line 72 and the closing `"""` on line 234), as compiled
 * at runtime by `cp.RawKernel(_BIWEIGHT_RUNNING_KERNEL, "biweight_running")`
 * in `_get_kernel()`. This is the sole CUDA detrend kernel — F4b scope per
 * docs/PLAN.md. Included into `detrend_cuda.rs` via `include_str!` and
 * NVRTC-compiled at runtime by `cudarc`, the same compile-from-source-string
 * strategy CuPy's `RawKernel` uses, so this is a driver-level port of the
 * identical compilation path, not a reimplementation.
 *
 * Reference copy for future re-sync: worker/wotan_cuda.py is production
 * code at ~/exonit (read-only from this worktree) — re-diff
 * against it if that file ever changes this kernel.
 */

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
