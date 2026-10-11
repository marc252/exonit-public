"""Phase 4 offline A/B: detrend (CPU vs CUDA) -> BLS -> compare candidates.

The stratified-100 scorecard A/B at spec § Phase 4 needs the full
worker pipeline (server + worker + DB) to run for ~1 hour. Before
spending that, this script does the cheaper version: load N real
TESS 2-min LCs, detrend each one twice (CPU wotan, CUDA wotan via
worker/wotan_detrend.py's selector), run BLS on both detrended flux
arrays, and diff the resulting best-candidate (period, SNR, depth,
t0). If the candidates agree row-by-row at this stage, the full
scorecard A/B is very likely to land clean.

Worker BLS is bls_fast.bls_two_pass (the numba two-pass), invoked
identically on both flats so any difference comes from the detrend.
"""
import glob
import importlib
import os
import sys
import time

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bls_fast import bls_two_pass  # noqa: E402


FITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fits_cache")
FITS_GLOB = "tess20*-s00*-*-s_lc.fits"
N_LCS = 10
WINDOW = 0.5
PERIOD_MIN = 0.5
PERIOD_MAX = 50.0
N_PERIODS = 50_000  # worker default
N_BINS = 100
N_DUR_STEPS = 5


def _load_lc(path):
    with fits.open(path) as h:
        t = np.asarray(h[1].data["TIME"], dtype=np.float64)
        f = np.asarray(h[1].data["PDCSAP_FLUX"], dtype=np.float64)
    m = np.isfinite(t) & np.isfinite(f) & (f > 0)
    if m.sum() < 100:
        return None
    return t[m], f[m] / np.nanmedian(f[m])


def _detrend(backend, t, f):
    """Re-import wotan_detrend under a forced backend and detrend."""
    os.environ["EXONIT_WOTAN_BACKEND"] = backend
    try:
        sys.modules.pop("wotan_detrend", None)
        import wotan_detrend as wd
        # Sanity: confirm the imported backend matches what we asked.
        active = wd.get_backend_name()
        if backend != active:
            raise RuntimeError(
                f"backend mismatch: requested {backend!r}, got {active!r}"
            )
        return wd.detrend_with_wotan(t, f, window_length=WINDOW)
    finally:
        os.environ.pop("EXONIT_WOTAN_BACKEND", None)
        sys.modules.pop("wotan_detrend", None)


def _run_bls(t, flat):
    finite = np.isfinite(flat) & np.isfinite(t)
    if finite.sum() < 1000:
        return None
    return bls_two_pass(
        t[finite], flat[finite],
        period_min=PERIOD_MIN, period_max=PERIOD_MAX, n_periods=N_PERIODS,
        n_bins=N_BINS, n_dur_steps=N_DUR_STEPS,
    )


def main():
    paths = sorted(glob.glob(f"{FITS_DIR}/{FITS_GLOB}"))[:N_LCS]
    print(f"=== Phase 4 offline BLS A/B  (CPU vs CUDA detrend) ===")
    print(f"FITS: {FITS_DIR} ({len(paths)} LCs)")
    print(f"BLS: period {PERIOD_MIN}-{PERIOD_MAX} d, {N_PERIODS} periods, "
          f"{N_BINS} bins, {N_DUR_STEPS} dur steps  (worker default)\n")

    print(f"{'#':>3}  {'N':>6}  {'P_cpu':>9}  {'P_cuda':>9}  {'dP':>10}  "
          f"{'SNR_cpu':>7}  {'SNR_cuda':>8}  {'depth_cpu':>10}  {'depth_cuda':>11}")
    bad = 0
    deltas = []
    for i, path in enumerate(paths):
        loaded = _load_lc(path)
        if loaded is None:
            print(f"  [{i+1:2d}] SKIP {os.path.basename(path)} (<100 finite pts)")
            continue
        t, f = loaded

        flat_cpu = _detrend("cpu", t, f)
        flat_cuda = _detrend("cuda", t, f)

        r_cpu = _run_bls(t, flat_cpu)
        r_cuda = _run_bls(t, flat_cuda)
        if r_cpu is None or r_cuda is None:
            print(f"  [{i+1:2d}] SKIP {os.path.basename(path)} (BLS no result)")
            continue

        p_cpu = r_cpu["period"]
        p_cuda = r_cuda["period"]
        s_cpu = r_cpu["snr"]
        s_cuda = r_cuda["snr"]
        d_cpu = r_cpu["depth"]
        d_cuda = r_cuda["depth"]
        dP = abs(p_cuda - p_cpu)
        deltas.append((p_cpu, p_cuda, s_cpu, s_cuda, d_cpu, d_cuda))
        # Flag if the candidate disagrees: period off by > 0.5% relative is
        # almost certainly different; SNR/depth secondary.
        if dP / max(p_cpu, 1e-6) > 5e-3:
            bad += 1
            flag = " *"
        else:
            flag = ""
        print(f"  [{i+1:2d}/{len(paths)}]  N={len(t):>5d}  "
              f"{p_cpu:>9.5f}  {p_cuda:>9.5f}  {dP:>10.2e}  "
              f"{s_cpu:>7.2f}  {s_cuda:>8.2f}  "
              f"{d_cpu:>10.5f}  {d_cuda:>10.5f}{flag}")

    if not deltas:
        print("\nno data")
        return

    p_cpu_v = np.array([d[0] for d in deltas])
    p_cuda_v = np.array([d[1] for d in deltas])
    s_cpu_v = np.array([d[2] for d in deltas])
    s_cuda_v = np.array([d[3] for d in deltas])
    d_cpu_v = np.array([d[4] for d in deltas])
    d_cuda_v = np.array([d[5] for d in deltas])
    print(f"\n=== Aggregate ({len(deltas)} LCs) ===")
    print(f"  period      |Δ| max = {np.max(np.abs(p_cpu_v - p_cuda_v)):.3e} d")
    print(f"  SNR         |Δ| max = {np.max(np.abs(s_cpu_v - s_cuda_v)):.3e}")
    print(f"  depth       |Δ| max = {np.max(np.abs(d_cpu_v - d_cuda_v)):.3e}")
    print(f"  candidate disagreements (period rel > 0.5%): {bad}/{len(deltas)}")
    if bad == 0:
        print("\n  CUDA detrend does not perturb BLS candidate on this sample.")


if __name__ == "__main__":
    main()
