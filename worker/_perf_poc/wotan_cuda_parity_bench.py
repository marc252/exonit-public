"""Parity + speed check: wotan_cuda.detrend_biweight vs wotan.flatten."""
import os, sys, time, glob
# Scratch dir with code/ (bls_fast, wotan_cuda) and fits/ — set BLS_AB_DIR.
BLS_AB_DIR = os.environ.get("BLS_AB_DIR", "bls_ab")
sys.path.insert(0, os.path.join(BLS_AB_DIR, "code"))

import numpy as np
from astropy.io import fits

from wotan import flatten as wotan_flatten
import wotan_cuda


def load_lc(path):
    with fits.open(path) as h:
        t = np.asarray(h[1].data["TIME"], dtype=np.float64)
        f = np.asarray(h[1].data["PDCSAP_FLUX"], dtype=np.float64)
    m = np.isfinite(t) & np.isfinite(f) & (f > 0)
    if m.sum() < 100:
        return None
    return t[m], f[m] / np.nanmedian(f[m])


fits_files = sorted(glob.glob(os.path.join(BLS_AB_DIR, "fits", "*lc.fits")))[:10]
print(f"=== wotan_cuda parity vs wotan.flatten on {len(fits_files)} LCs ===\n")
print(f"CUDA available: {wotan_cuda.is_available()}")

# Warm-up
t0, f0 = load_lc(fits_files[0])
print("warming up wotan numba...")
_ = wotan_flatten(t0, f0, method="biweight", window_length=0.5, cval=5.0,
                  edge_cutoff=0.5, break_tolerance=0.5, return_trend=True)
print("warming up CUDA kernel...")
_ = wotan_cuda.detrend_biweight(t0, f0, 0.5)
print()

def rel(a, b):
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() == 0:
        return float("nan")
    da = np.abs(a[finite] - b[finite])
    sa = np.maximum(np.abs(a[finite]), 1e-12)
    return float(np.max(da / sa))


for i, path in enumerate(fits_files):
    t, f = load_lc(path)
    N = len(t)

    # CPU wotan
    cpu_t = []
    for _ in range(3):
        t0 = time.perf_counter()
        flat_cpu, trend_cpu = wotan_flatten(
            t, f, method="biweight", window_length=0.5, cval=5.0,
            edge_cutoff=0.5, break_tolerance=0.5, return_trend=True)
        cpu_t.append(time.perf_counter() - t0)
    cpu_ms = min(cpu_t) * 1000

    # GPU
    gpu_t = []
    for _ in range(3):
        t0 = time.perf_counter()
        flat_gpu, trend_gpu = wotan_cuda.detrend_biweight(
            t, f, window_length=0.5, edge_cutoff=0.5, cval=5.0,
            break_tolerance=0.5)
        gpu_t.append(time.perf_counter() - t0)
    gpu_ms = min(gpu_t) * 1000

    # Parity on trend (the real output; flat = f/trend will inherit it)
    rel_trend = rel(trend_cpu, trend_gpu)
    rel_flat  = rel(flat_cpu, flat_gpu)
    finite_cpu = np.isfinite(trend_cpu).sum()
    finite_gpu = np.isfinite(trend_gpu).sum()

    speedup = cpu_ms / gpu_ms if gpu_ms > 0 else float("inf")
    print(f"  [{i+1:2d}/{len(fits_files)}] N={N:5d}  cpu={cpu_ms:6.0f}ms gpu={gpu_ms:6.0f}ms "
          f"({speedup:.2f}x)  rel_trend={rel_trend:.2e}  rel_flat={rel_flat:.2e}  "
          f"finite cpu/gpu={finite_cpu}/{finite_gpu}")
