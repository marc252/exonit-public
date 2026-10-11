"""Phase 3 decision bench: how does MAX_ITER affect speed + parity?

After Phase 2 the biweight iteration loop is dirt cheap (no median calls
inside), so the spec's 1.3-1.6x speedup expectation from dropping
MAX_ITER may not survive. This bench actually measures the trade-off.

For each candidate MAX_ITER value, re-compile the production kernel with
that limit and run the same parity+speed test as the main bench. Report
median speedup and max rel_trend per setting.
"""
import glob
import os
import sys
import time

import numpy as np
import cupy as cp
from astropy.io import fits

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import wotan_cuda  # noqa: E402
from wotan import flatten as wotan_flatten  # noqa: E402


FITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fits_cache")
FITS_GLOB = "tess20*-s00*-*-s_lc.fits"
N_LCS = 10
WINDOW = 0.5
EDGE_CUTOFF = 0.5
BREAK_TOL = 0.5
CVAL = 5.0

MAX_ITER_VALUES = [10, 8, 7, 6, 5, 4, 3]


def _build_kernel(max_iter):
    src = wotan_cuda._BIWEIGHT_RUNNING_KERNEL
    # The production kernel hard-codes ``const int MAX_ITER = 10;``.
    # Patch it for this run.
    import re
    needle_re = re.compile(r"const int MAX_ITER = \d+;")
    assert needle_re.search(src), "MAX_ITER token not found in production kernel"
    patched = needle_re.sub(f"const int MAX_ITER = {max_iter};", src)
    return cp.RawKernel(patched, "biweight_running")


def load_lc(path):
    with fits.open(path) as h:
        t = np.asarray(h[1].data["TIME"], dtype=np.float64)
        f = np.asarray(h[1].data["PDCSAP_FLUX"], dtype=np.float64)
    m = np.isfinite(t) & np.isfinite(f) & (f > 0)
    if m.sum() < 100:
        return None
    return t[m], f[m] / np.nanmedian(f[m])


def _run_kernel(kernel, t, f):
    t = np.ascontiguousarray(t, dtype=np.float64)
    f = np.ascontiguousarray(f, dtype=np.float64)
    N = len(t)
    d_t = cp.asarray(t)
    d_f = cp.asarray(f)
    d_trend = cp.empty(N, dtype=cp.float64)
    half = WINDOW / 2.0
    ec = min(EDGE_CUTOFF, half)
    low_index = float(t.min()) + ec
    hi_index = float(t.max()) - ec
    block = 128
    grid = (N + block - 1) // block
    kernel((grid,), (block,), (
        d_t, d_f, d_trend,
        np.float64(half), np.float64(low_index), np.float64(hi_index),
        np.float32(CVAL), np.int32(N),
    ))
    return cp.asnumpy(d_trend)


def _detrend_segments(kernel, t, f):
    """Mirror wotan_cuda.detrend_biweight's break_tolerance segment split."""
    if BREAK_TOL > 0:
        gaps = np.diff(t) > BREAK_TOL
        breaks = np.flatnonzero(gaps) + 1
    else:
        breaks = []
    starts = np.concatenate([[0], breaks]).astype(int)
    ends = np.concatenate([breaks, [len(t)]]).astype(int)
    trend = np.full(len(t), np.nan)
    for s, e in zip(starts, ends):
        if e - s < 3:
            continue
        trend[s:e] = _run_kernel(kernel, t[s:e], f[s:e])
    return trend


def _rel(a, b):
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() == 0:
        return float("nan")
    da = np.abs(a[finite] - b[finite])
    sa = np.maximum(np.abs(a[finite]), 1e-12)
    return float(np.max(da / sa))


def main():
    paths = sorted(glob.glob(f"{FITS_DIR}/{FITS_GLOB}"))[:N_LCS]
    print("=== Phase 3 MAX_ITER sweep ===")
    print(f"FITS: {FITS_DIR} ({len(paths)} LCs)")
    try:
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        print(f"GPU: {gpu_name}")
    except Exception:
        pass
    print()

    # Pre-load LCs and compute reference (wotan numba) once per LC.
    cache = []
    for path in paths:
        loaded = load_lc(path)
        if loaded is None:
            continue
        t, f = loaded
        _, trend_cpu = wotan_flatten(
            t, f, method="biweight", window_length=WINDOW, cval=CVAL,
            edge_cutoff=EDGE_CUTOFF, break_tolerance=BREAK_TOL,
            return_trend=True,
        )
        # Also measure CPU best-of-3 timing
        cpu_ts = []
        for _ in range(3):
            ts = time.perf_counter()
            wotan_flatten(t, f, method="biweight", window_length=WINDOW,
                          cval=CVAL, edge_cutoff=EDGE_CUTOFF,
                          break_tolerance=BREAK_TOL, return_trend=True)
            cpu_ts.append(time.perf_counter() - ts)
        cache.append((t, f, trend_cpu, min(cpu_ts) * 1000))

    print(f"{'MAX_ITER':>9}  {'min spd':>8}  {'med spd':>8}  {'max spd':>8}  "
          f"{'med rel':>10}  {'max rel':>10}  {'gain vs 10':>10}")
    baseline_med = None
    for max_iter in MAX_ITER_VALUES:
        kernel = _build_kernel(max_iter)
        # Warmup
        t0, f0, _, _ = cache[0]
        _ = _detrend_segments(kernel, t0, f0)
        speeds = []
        rels = []
        for t, f, trend_cpu, cpu_ms in cache:
            gpu_ts = []
            for _ in range(3):
                ts = time.perf_counter()
                trend_gpu = _detrend_segments(kernel, t, f)
                gpu_ts.append(time.perf_counter() - ts)
            gpu_ms = min(gpu_ts) * 1000
            speeds.append(cpu_ms / gpu_ms)
            rels.append(_rel(trend_cpu, trend_gpu))
        med = float(np.median(speeds))
        if baseline_med is None:
            baseline_med = med
        gain = med / baseline_med
        print(f"{max_iter:>9}  {min(speeds):>7.2f}x  {med:>7.2f}x  {max(speeds):>7.2f}x  "
              f"{np.median(rels):>10.2e}  {max(rels):>10.2e}  {gain:>9.3f}x")


if __name__ == "__main__":
    main()
