"""Phase 3 instrumentation: histogram of biweight iterations per thread.

Re-compiles the production kernel with one extra parameter (int* iter_out)
so we record, per centre-of-window thread, the number of biweight loop
iterations actually executed before convergence. We then summarise the
distribution across many real TESS 2-min LCs.

Goal: confirm the spec's expectation that wotan-style biweight converges
in <= 5 iterations on typical TESS PDCSAP. If P99 <= 5, we can safely
drop MAX_ITER from 10 to 5 (matching wotan's own `n_iter=5` default)
without changing parity.

We reuse the production device helpers (`dev_quickselect`, `dev_median`)
by slicing them out of `worker/wotan_cuda._BIWEIGHT_RUNNING_KERNEL`. The
instrumented kernel mirrors the production logic verbatim except for the
extra output write at the end.
"""
import glob
import os
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import wotan_cuda  # noqa: E402
import cupy as cp  # noqa: E402


FITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fits_cache")
FITS_GLOB = "tess20*-s00*-*-s_lc.fits"
N_LCS = 10  # gives ~150k thread samples — plenty for the histogram
WINDOW = 0.5
EDGE_CUTOFF = 0.5
BREAK_TOL = 0.5
CVAL = 5.0
MAX_ITER_INSTR = 20   # generous cap to see the true distribution


def _make_instrumented_kernel():
    # Pull the device helpers from the production kernel source.
    prod = wotan_cuda._BIWEIGHT_RUNNING_KERNEL
    marker = 'extern "C"'
    idx = prod.find(marker)
    assert idx > 0
    device_helpers = prod[:idx]

    # Same body as production, just with an int* iter_out parameter and
    # MAX_ITER bumped so we can see the true convergence count.
    body = r"""
extern "C" __global__
void biweight_running_instr(
    const double* __restrict__ time,
    const double* __restrict__ flux,
    double*       trend,
    int*          iter_out,
    const double  half_window,
    const double  low_index,
    const double  hi_index,
    const float   cval,
    const int     N
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;

    double ti = time[i];
    if (ti < low_index || ti > hi_index) {
        trend[i] = nan("");
        iter_out[i] = -1;   // edge cutoff
        return;
    }

    double lo_bound = ti - half_window;
    double hi_bound = ti + half_window;
    int lo = 0, hi = N - 1, idx_start = N;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (time[mid] >= lo_bound) { idx_start = mid; hi = mid - 1; }
        else                       { lo = mid + 1; }
    }
    lo = idx_start; hi = N - 1;
    int idx_end = idx_start - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (time[mid] < hi_bound) { idx_end = mid; lo = mid + 1; }
        else                      { hi = mid - 1; }
    }

    int win_n = idx_end - idx_start + 1;
    if (win_n < 3) { trend[i] = nan(""); iter_out[i] = -2; return; }

    const int MAX_W = 768;
    if (win_n > MAX_W) { trend[i] = nan(""); iter_out[i] = -3; return; }

    double buf[MAX_W];
    double work[MAX_W];
    for (int j = 0; j < win_n; j++) buf[j] = flux[idx_start + j];

    for (int j = 0; j < win_n; j++) work[j] = buf[j];
    double median_data = dev_median(work, win_n);
    double center = median_data;

    for (int j = 0; j < win_n; j++) work[j] = fabs(buf[j] - median_data);
    double mad = dev_median(work, win_n);

    if (mad == 0.0) {
        trend[i] = median_data;
        iter_out[i] = 0;   // converged before entering loop
        return;
    }

    const double cmad = 1.0 / ((double)cval * mad);
    const int MAX_ITER = """ + str(MAX_ITER_INSTR) + r""";
    const double FTOL = 1e-6;

    int n_iter = MAX_ITER;
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
        if (sumw < 1e-30) { n_iter = iter + 1; break; }
        double c_new = sumwx / sumw;
        if (fabs(c_new - center) < FTOL) {
            center = c_new;
            n_iter = iter + 1;
            break;
        }
        center = c_new;
    }
    trend[i] = center;
    iter_out[i] = n_iter;
}
"""
    return cp.RawKernel(device_helpers + body, "biweight_running_instr")


def load_lc(path):
    with fits.open(path) as h:
        t = np.asarray(h[1].data["TIME"], dtype=np.float64)
        f = np.asarray(h[1].data["PDCSAP_FLUX"], dtype=np.float64)
    m = np.isfinite(t) & np.isfinite(f) & (f > 0)
    if m.sum() < 100:
        return None
    return t[m], f[m] / np.nanmedian(f[m])


def run_segment(kernel, t, f, window_length, edge_cutoff):
    t = np.ascontiguousarray(t, dtype=np.float64)
    f = np.ascontiguousarray(f, dtype=np.float64)
    N = len(t)
    d_t = cp.asarray(t)
    d_f = cp.asarray(f)
    d_trend = cp.empty(N, dtype=cp.float64)
    d_iter = cp.full(N, -99, dtype=cp.int32)
    half = window_length / 2.0
    ec = min(edge_cutoff, half)
    low_index = float(t.min()) + ec
    hi_index = float(t.max()) - ec
    block = 128
    grid = (N + block - 1) // block
    kernel((grid,), (block,), (
        d_t, d_f, d_trend, d_iter,
        np.float64(half), np.float64(low_index), np.float64(hi_index),
        np.float32(CVAL), np.int32(N),
    ))
    return cp.asnumpy(d_iter)


def main():
    paths = sorted(glob.glob(f"{FITS_DIR}/{FITS_GLOB}"))[:N_LCS]
    print(f"=== Phase 3 - biweight iteration histogram ===")
    print(f"FITS: {FITS_DIR} ({len(paths)} LCs)\n")
    kernel = _make_instrumented_kernel()

    all_iter = []
    for i, path in enumerate(paths):
        loaded = load_lc(path)
        if loaded is None:
            continue
        t, f = loaded
        if BREAK_TOL is not None and BREAK_TOL > 0:
            gaps = np.diff(t) > BREAK_TOL
            breaks = np.flatnonzero(gaps) + 1
        else:
            breaks = []
        starts = np.concatenate([[0], breaks]).astype(int)
        ends = np.concatenate([breaks, [len(t)]]).astype(int)
        for s, e in zip(starts, ends):
            if e - s < 3:
                continue
            iters = run_segment(kernel, t[s:e], f[s:e], WINDOW, EDGE_CUTOFF)
            all_iter.append(iters)
        n_threads = sum(arr.size for arr in all_iter)
        print(f"  [{i+1:2d}/{len(paths)}] {os.path.basename(path)}  cumulative threads: {n_threads}")

    if not all_iter:
        print("\nNo data.")
        return
    iters = np.concatenate(all_iter)
    # Keep only the threads that actually ran biweight iterations (>=0).
    real = iters[iters >= 0]
    edge_cut = int((iters == -1).sum())
    too_short = int((iters == -2).sum())
    too_wide = int((iters == -3).sum())
    mad_zero = int((real == 0).sum())
    iterated = real[real > 0]

    print(f"\n=== Threads ({iters.size} total) ===")
    print(f"  edge-cutoff (NaN)       : {edge_cut}")
    print(f"  win_n < 3 (NaN)         : {too_short}")
    print(f"  win_n > MAX_W (NaN)     : {too_wide}")
    print(f"  MAD==0 early exit       : {mad_zero}")
    print(f"  ran biweight iters      : {iterated.size}")

    if iterated.size == 0:
        return

    print(f"\n=== Iteration distribution (biweight-running threads) ===")
    print(f"  min : {iterated.min()}")
    print(f"  max : {iterated.max()}")
    print(f"  mean: {iterated.mean():.3f}")
    print(f"  P50 : {np.percentile(iterated, 50):.0f}")
    print(f"  P90 : {np.percentile(iterated, 90):.0f}")
    print(f"  P99 : {np.percentile(iterated, 99):.0f}")
    print(f"  P99.9:{np.percentile(iterated, 99.9):.0f}")
    print()
    print(f"=== Per-iter count histogram ===")
    counts = np.bincount(iterated, minlength=MAX_ITER_INSTR + 1)
    cum_pct = 0.0
    for k in range(1, MAX_ITER_INSTR + 1):
        if counts[k] == 0 and k > iterated.max():
            break
        pct = 100.0 * counts[k] / iterated.size
        cum_pct += pct
        bar = "#" * int(pct / 2)
        print(f"  iter={k:2d}  n={counts[k]:7d}  {pct:6.2f}%  cum={cum_pct:6.2f}%  {bar}")
    if iterated.max() >= MAX_ITER_INSTR:
        # The cap itself is a sentinel for "did not converge in MAX_ITER_INSTR"
        non_conv = int((iterated == MAX_ITER_INSTR).sum())
        print(f"\n  ! {non_conv} threads hit MAX_ITER_INSTR={MAX_ITER_INSTR} cap "
              f"(did not converge within absolute FTOL).")


if __name__ == "__main__":
    main()
