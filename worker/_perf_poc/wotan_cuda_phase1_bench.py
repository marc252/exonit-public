"""Wotan-CUDA bench: kernel parity + speedup vs wotan.flatten.

Same script for Phase 1 (Quickselect) and Phase 2 (parity-gap closure);
the kernel version is reported in the per-LC output and summary header.

Runs on <host> RTX 5060 Ti. Compares the worker's wotan_cuda
detrend against wotan.flatten on 10 real TESS 2-min LCs taken from
worker/fits_cache/. Reports per-LC CPU/GPU wall ms, speedup, and the
max relative trend disagreement against the CPU baseline.

Phase 1 success criteria (per spec):
- Quickselect-vs-insertion-sort median values must be bit-identical
  (so rel_trend must match the prior PoC's ~4.4e-4 — *not* improved
  in Phase 1; that's Phase 2's job).
- Median speedup expected: 5–8× on 5060 Ti. Spec kill-switch at < 2.5×
  on T4. We bench on 5060 Ti as the accessible proxy; T4 has to be
  benched separately in GCP.
"""
import glob
import os
import sys
import time

import numpy as np
from astropy.io import fits

# Sit inside worker/ so the flat wotan_cuda import resolves.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import wotan_cuda  # noqa: E402
from wotan import flatten as wotan_flatten  # noqa: E402


FITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fits_cache")
FITS_GLOB = "tess20*-s00*-*-s_lc.fits"  # SPOC 2-min only; QLP/TGLC don't have PDCSAP_FLUX
N_LCS = 10
WINDOW = 0.5
EDGE_CUTOFF = 0.5
BREAK_TOL = 0.5
CVAL = 5.0


def load_lc(path):
    with fits.open(path) as h:
        t = np.asarray(h[1].data["TIME"], dtype=np.float64)
        f = np.asarray(h[1].data["PDCSAP_FLUX"], dtype=np.float64)
    m = np.isfinite(t) & np.isfinite(f) & (f > 0)
    if m.sum() < 100:
        return None
    return t[m], f[m] / np.nanmedian(f[m])


def rel(a, b):
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() == 0:
        return float("nan")
    da = np.abs(a[finite] - b[finite])
    sa = np.maximum(np.abs(a[finite]), 1e-12)
    return float(np.max(da / sa))


def main():
    paths = sorted(glob.glob(f"{FITS_DIR}/{FITS_GLOB}"))[:N_LCS]
    print(f"=== wotan_cuda Phase 1 (Quickselect) parity vs wotan.flatten ===")
    print(f"FITS: {FITS_DIR} ({len(paths)} files)")
    print(f"CUDA available: {wotan_cuda.is_available()}\n")

    # Warm up both
    t0, f0 = load_lc(paths[0])
    print("warming up wotan numba ...")
    _ = wotan_flatten(
        t0, f0, method="biweight", window_length=WINDOW, cval=CVAL,
        edge_cutoff=EDGE_CUTOFF, break_tolerance=BREAK_TOL, return_trend=True,
    )
    print("warming up CUDA kernel ...")
    _ = wotan_cuda.detrend_biweight(t0, f0, WINDOW)
    print()

    speedups = []
    rels_trend = []
    rels_flat = []
    for i, path in enumerate(paths):
        loaded = load_lc(path)
        if loaded is None:
            print(f"  [{i+1:2d}] SKIP {os.path.basename(path)} (<100 finite pts)")
            continue
        t, f = loaded
        N = len(t)

        # CPU (numba wotan) — best of 3
        cpu_ts = []
        for _ in range(3):
            ts = time.perf_counter()
            flat_cpu, trend_cpu = wotan_flatten(
                t, f, method="biweight", window_length=WINDOW, cval=CVAL,
                edge_cutoff=EDGE_CUTOFF, break_tolerance=BREAK_TOL,
                return_trend=True,
            )
            cpu_ts.append(time.perf_counter() - ts)
        cpu_ms = min(cpu_ts) * 1000

        # GPU — best of 3
        gpu_ts = []
        for _ in range(3):
            ts = time.perf_counter()
            flat_gpu, trend_gpu = wotan_cuda.detrend_biweight(
                t, f, window_length=WINDOW, edge_cutoff=EDGE_CUTOFF,
                cval=CVAL, break_tolerance=BREAK_TOL,
            )
            gpu_ts.append(time.perf_counter() - ts)
        gpu_ms = min(gpu_ts) * 1000

        rt = rel(trend_cpu, trend_gpu)
        rf = rel(flat_cpu, flat_gpu)
        fcpu = int(np.isfinite(trend_cpu).sum())
        fgpu = int(np.isfinite(trend_gpu).sum())
        sp = cpu_ms / gpu_ms if gpu_ms > 0 else float("inf")
        speedups.append(sp)
        rels_trend.append(rt)
        rels_flat.append(rf)

        print(
            f"  [{i+1:2d}/{len(paths)}] N={N:5d}  cpu={cpu_ms:6.0f}ms  "
            f"gpu={gpu_ms:6.0f}ms  ({sp:5.2f}x)  "
            f"rel_trend={rt:.2e}  rel_flat={rf:.2e}  "
            f"finite cpu/gpu={fcpu}/{fgpu}"
        )

    if not speedups:
        print("\nNo LCs processed.")
        return

    speedups = np.asarray(speedups)
    rels_trend = np.asarray(rels_trend)
    rels_flat = np.asarray(rels_flat)
    try:
        import cupy as _cp
        gpu_name = _cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        gpu_name = "unknown GPU"
    print(f"\n=== Summary (wotan-CUDA bench, {gpu_name}) ===")
    print(f"  speedup  : min={speedups.min():.2f}x  median={np.median(speedups):.2f}x  "
          f"max={speedups.max():.2f}x")
    print(f"  rel_trend: min={rels_trend.min():.2e}  median={np.median(rels_trend):.2e}  "
          f"max={rels_trend.max():.2e}")
    print(f"  rel_flat : min={rels_flat.min():.2e}  median={np.median(rels_flat):.2e}  "
          f"max={rels_flat.max():.2e}")
    print()
    print(f"  PoC baseline (insertion sort): median speedup 1.10x")
    print(f"  Phase 1 spec target (5060 Ti): median speedup 5-8x")
    print(f"  Phase 1 kill-switch (T4): < 2.5x median -> abandon")


if __name__ == "__main__":
    main()
