#!/usr/bin/env python3
"""Python twin of `exonit-bench` (crates/exonit-worker/src/bin/exonit_bench.rs).

Local-only, no server/network. Measures the SAME six stages, on the SAME
input FITS files, at the SAME algorithm parameters, using the REAL
production worker modules imported from `~/exonit/worker`
(read-only — nothing there is modified, nothing here is a reimplementation
of the science). Run with:

    ~/exonit/.venv/bin/python3 scripts/benchmark_python_stages.py \\
        --fits-dir tests/data --iterations 3 \\
        [--models-dir ~/exonit/worker/models] [--output out.json]

`--worker-dir` (default `~/exonit/worker`, or `$EXONIT_WORKER_DIR`)
is added to `sys.path` for the real-module imports below.

Stage -> Python production function (mirrors exonit_bench.rs's own table):

  (a) FITS read           -> `_read_lightcurve` (verbatim copy of
                              `worker/worker.py:427-453`, see its own doc
                              comment for why it is copied, not imported,
                              and the QLP/SPOC/TGLC column choice it makes).
  (b) preprocess+detrend   -> `_prepare_and_detrend`, a verbatim
                              transcription of `worker/worker.py:2306-2379`'s
                              single-FITS block: `lc.remove_nans()
                              .remove_outliers(...)` -> conditional
                              `lc.fill_gaps()` -> REAL
                              `wotan_detrend.detrend_with_wotan` -> REAL
                              `bls_fast.bls_fast` quick pass (period
                              0.5-50d, 2000 periods, 100 bins, 5 duration
                              steps, SNR gate 7.0) -> conditional
                              re-detrend with a transit mask.
  (c) BLS two-pass CPU     -> REAL `bls_fast.bls_two_pass` (production
                              defaults: period 0.5-50d, 50 000 periods,
                              200 bins, 15 duration steps — same numbers
                              `exonit_bench.rs::production_bls_params()`
                              uses), called DIRECTLY (not through `bls.py`'s
                              dispatcher) so the measurement is CPU-only
                              regardless of what backend this host would
                              otherwise auto-pick.
  (d) BLS two-pass CUDA    -> REAL `bls_cuda.bls_gpu_two_pass`, same
                              parameters, ONLY when
                              `bls_cuda.is_cuda_available()` — `null` in
                              the JSON everywhere else (<host> has no
                              NVIDIA GPU).
  (e) full per-TIC pipeline -> best-effort real-code equivalent of the
                              production main BLS loop's FIRST iteration:
                              stage (b)'s block + REAL `bls.run_bls`
                              (production dispatcher; on this host resolves
                              to `bls_fast.bls_two_pass`, same call shape as
                              `worker/worker.py:2675-2678`) + REAL
                              `resolve_harmonic` (verbatim copy of
                              `worker/worker.py:197-343`, standalone
                              function). Does NOT reproduce the
                              multi-planet loop (up to `max_planets=5`
                              iterations, accumulated-mask re-detrend,
                              protected slots, long-period sweep) that
                              lives inline in `worker.py::process_task`
                              (~1000 lines, entangled with `self`
                              state/config/network) — see the "caveats"
                              field emitted below for the exact scope gap
                              vs. the Rust side's `multi_planet::search`.
  (f) ONNX CNN + ExoMiner   -> REAL `cnn_ffi.FFITransitClassifier.classify`
      FFI scores               + REAL
                              `exominer_ffi_inference.ExoMinerFFIInferer.score`
                              (CPU EP forced via `gpu_enabled=False`),
                              scored against the stage (c) BLS two-pass
                              PEAK, exactly like the Rust side (see its own
                              matching caveat in `exonit_bench.rs`).

JSON output shape matches `exonit_bench.rs`'s exactly (same key names) so
`scripts/run_ab_bench.sh` can compute ratios generically without knowing
which side produced which file.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

DEFAULT_WORKER_DIR = Path(os.environ.get("EXONIT_WORKER_DIR", "~/exonit/worker"))


def _add_worker_dir_to_path(worker_dir: Path) -> None:
    path_str = str(worker_dir)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# ---------------------------------------------------------------------------
# Stage (a): FITS read.
# ---------------------------------------------------------------------------
#
# Verbatim copy of `worker/worker.py:427-453` (read-only source, not
# modified), not an import of `worker.py` itself: importing the real module
# would run its module-level side effects (Flask app construction,
# argparse-driven config, FPP/triceratops backend probing, GPU consistency
# checks) that have nothing to do with the FITS-reading cost this stage
# measures, and would make this script fragile to unrelated worker.py
# changes. This is the ONE stage where a copy, not a call, is the honest
# choice; every other real-module dependency below is a genuine `import`.


def _read_lightcurve(path):
    """Read a TESS light curve into a lightkurve object.

    TGLC FFI light curves store the calibrated flux in ``cal_aper_flux``
    (with quality columns ``TESS_flags``/``TGLC_flags``), which lightkurve
    cannot auto-detect. Read those by hand; fall back to ``lk.read`` for
    SPOC/QLP (verbatim from worker/worker.py:427-453)."""
    import lightkurve as lk
    from astropy.io import fits as _fits

    try:
        with _fits.open(path, memmap=False) as hdul:
            cols = getattr(hdul[1], "columns", None) if len(hdul) > 1 else None
            names = {c.name.lower() for c in cols} if cols is not None else set()
            if "cal_aper_flux" in names:
                d = hdul[1].data
                t = np.asarray(d["time"], dtype=np.float64)
                f = np.asarray(d["cal_aper_flux"], dtype=np.float64)
                qa = (np.asarray(d["TESS_flags"], dtype=np.int64)
                      if "tess_flags" in names else np.zeros(len(t), np.int64))
                qb = (np.asarray(d["TGLC_flags"], dtype=np.int64)
                      if "tglc_flags" in names else np.zeros(len(t), np.int64))
                good = ((qa == 0) & (qb == 0)
                        & np.isfinite(t) & np.isfinite(f) & (f > 0))
                return lk.LightCurve(time=t[good], flux=f[good])
    except Exception:
        pass
    return lk.read(path)


# ---------------------------------------------------------------------------
# Stage (b): preprocess + full iterative detrend.
# ---------------------------------------------------------------------------
#
# Verbatim transcription of `worker/worker.py:2306-2379`'s single-FITS-file
# block (the multi-sector `LightCurveCollection.stitch()` branch is skipped
# — matches `exonit_bench.rs::load_curve`'s own choice: this benchmark is
# per-file/single-sector on both sides of the A/B, see its doc comment).
# Calls the REAL `wotan_detrend.detrend_with_wotan` and `bls_fast.bls_fast`
# — no reimplementation.


def _prepare_and_detrend(lc, bls_fast_module, wotan_module):
    warnings.filterwarnings("ignore", message=".*tpfmodel.*oktopus.*")
    warnings.filterwarnings("ignore", message=".*cadences will be ignored due to the quality mask.*")

    lc = lc.remove_nans().remove_outliers(sigma_upper=5, sigma_lower=float("inf"))
    try:
        max_gap_days = 5
        time_vals = lc.time.value
        if len(time_vals) > 1:
            gaps = np.diff(time_vals)
            max_actual_gap = np.max(gaps)
            if max_actual_gap < max_gap_days:
                lc = lc.fill_gaps()
    except (ValueError, Exception):
        pass

    t_raw = np.array(lc.time.value, dtype=np.float64)
    f_raw = np.array(lc.flux.value, dtype=np.float64)
    raw_flux_arr = f_raw.copy()
    finite_mask = np.isfinite(t_raw) & np.isfinite(f_raw)
    flat_flux = np.full_like(f_raw, np.nan)
    flat_flux[finite_mask] = wotan_module.detrend_with_wotan(t_raw[finite_mask], f_raw[finite_mask])

    try:
        m = np.isfinite(t_raw) & np.isfinite(flat_flux)
        quick_result = bls_fast_module.bls_fast(
            t_raw[m], flat_flux[m],
            period_min=0.5, period_max=50.0, n_periods=2000,
            n_bins=100, n_dur_steps=5)
        if quick_result["snr"] >= 7.0:
            phase = ((t_raw - quick_result["t0"]) % quick_result["period"]) / quick_result["period"]
            half_dur = quick_result["duration"] / quick_result["period"]
            transit_mask = (phase < half_dur) | (phase > 1.0 - half_dur)
            flat_flux[finite_mask] = wotan_module.detrend_with_wotan(
                t_raw[finite_mask], f_raw[finite_mask],
                transit_mask=transit_mask[finite_mask])
    except Exception:
        pass

    time_arr = t_raw
    flux_arr = flat_flux
    median = np.nanmedian(flux_arr)
    if median > 1.5 or median < 0.5:
        flux_arr = flux_arr / median

    sort_idx = np.argsort(time_arr)
    time_arr, flux_arr, raw_flux_arr = time_arr[sort_idx], flux_arr[sort_idx], raw_flux_arr[sort_idx]
    mask = np.isfinite(time_arr) & np.isfinite(flux_arr) & np.isfinite(raw_flux_arr)
    return time_arr[mask], flux_arr[mask], raw_flux_arr[mask]


# ---------------------------------------------------------------------------
# Stage (e) support: `resolve_harmonic`, verbatim copy of
# `worker/worker.py:197-343` (read-only source; same copy-not-import
# rationale as `_read_lightcurve` — it is a standalone function with no
# module-level side effects of its own, but pulling it in via `import
# worker` would still trigger worker.py's module-level side effects, see
# above). Used ONLY as the real-code post-processing step after the main
# BLS call — see this file's module doc comment for the exact scope gap
# vs. the Rust side's `multi_planet::search`.
# ---------------------------------------------------------------------------


def resolve_harmonic(time, flux, result, period_min, period_max, bls_fn,
                      snr_min: float = 5.0,
                      depth_ratio_lo: float = 0.5,
                      depth_ratio_hi: float = 2.0,
                      data_source: str = "spoc",
                      long_period_recovery: bool = False,
                      long_period_min_days: float = 10.0,
                      long_snr_factor: float = 0.5):
    base_period = result["period"]
    base_depth = result.get("depth", 0.0)
    is_ffi = (data_source == "ffi")
    base_snr = result.get("snr", 0.0)

    span = 0.0
    try:
        span = float(time[-1] - time[0])
    except Exception:
        span = 0.0

    probes = []
    if is_ffi:
        for n in (2, 3, 4, 5):
            cand = base_period * n
            if cand <= period_max:
                probes.append((cand, f"x{n}", 0.6, True))
    else:
        cand = base_period * 2
        if cand <= period_max:
            probes.append((cand, "x2", 1.0, False))
        if long_period_recovery:
            from fractions import Fraction as _Fraction
            for fac in (_Fraction(3, 2), _Fraction(5, 3), _Fraction(4, 3),
                        _Fraction(5, 2)):
                cand = base_period * float(fac)
                if long_period_min_days <= cand <= period_max:
                    probes.append((cand, f"x{fac}", long_snr_factor, True))
    sub_factor = 0.6 if is_ffi else 1.0
    for n in (2, 3, 4, 5):
        cand = base_period / n
        if cand >= period_min:
            probes.append((cand, f"/{n}", sub_factor, False))

    for cand_period, label, probe_snr_factor, guard in probes:
        try:
            sub = bls_fn(time, flux,
                         period_min=cand_period * 0.95,
                         period_max=cand_period * 1.05,
                         n_periods=500)
        except Exception:
            continue
        cand_snr = sub.get("snr", 0.0)
        if cand_snr < snr_min:
            continue
        if cand_snr < base_snr * probe_snr_factor:
            continue
        if base_depth > 0:
            depth_ratio = sub.get("depth", 0.0) / base_depth
            if not (depth_ratio_lo < depth_ratio < depth_ratio_hi):
                continue
        if guard and span > 0 and sub.get("period", 0.0) > 0 \
                and span / sub["period"] < 2.0:
            continue
        return sub
    return result


# ---------------------------------------------------------------------------
# Timing harness.
# ---------------------------------------------------------------------------


def time_stage(iterations: int, fn):
    """Mirrors `exonit_bench.rs::time_stage`: 1 discarded warm-up, then
    `iterations` measured with `time.perf_counter()` (OS monotonic clock,
    same semantics as Rust's `std::time::Instant`)."""
    fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return {
        "iterations": iterations,
        "mean_s": sum(samples) / len(samples),
        "min_s": min(samples),
        "samples_s": samples,
    }


def git_sha(repo_dir: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--fits-dir", help="benchmark every *.fits file directly inside DIR (non-recursive)")
    group.add_argument("--fits", nargs="+", help="benchmark exactly these FITS files")
    p.add_argument("--iterations", type=int, default=3, help="measured iterations per stage per file (default 3)")
    p.add_argument("--models-dir", default=None, help="dir with cnn_ffi_v1.onnx / exominer_ffi_v6_ensemble.onnx (default: <worker-dir>/models if present)")
    p.add_argument("--output", default=None, help="write JSON here instead of stdout")
    p.add_argument("--worker-dir", default=str(DEFAULT_WORKER_DIR), help="path to ~/exonit/worker")
    return p.parse_args()


def load_classifiers(models_dir: Path, cnn_ffi_module, exominer_module):
    cnn = None
    cnn_path = models_dir / "cnn_ffi_v1.onnx"
    if cnn_path.exists():
        try:
            cnn = cnn_ffi_module.FFITransitClassifier(model_path=str(cnn_path), gpu_enabled=False)
        except Exception as exc:
            print(f"benchmark_python_stages: CNN model load failed: {exc}", file=sys.stderr)
    exominer = None
    exominer_path = models_dir / "exominer_ffi_v6_ensemble.onnx"
    if exominer_path.exists():
        try:
            exominer = exominer_module.ExoMinerFFIInferer(model_path=str(exominer_path), gpu_enabled=False)
        except Exception as exc:
            print(f"benchmark_python_stages: ExoMiner model load failed: {exc}", file=sys.stderr)
    return cnn, exominer


# Production BLS grid — mirrors `exonit_bench.rs::production_bls_params()`
# and `worker/worker.py:2675-2678`'s main-loop call (period 0.5-50d,
# n_periods=50000; n_bins/n_dur_steps left at bls_fast's own defaults,
# 200/15, matching `BlsParams::default()`-shaped values on the Rust side).
PERIOD_MIN = 0.5
PERIOD_MAX = 50.0
N_PERIODS = 50_000
N_BINS = 200
N_DUR_STEPS = 15


def main() -> int:
    args = parse_args()
    worker_dir = Path(args.worker_dir)
    _add_worker_dir_to_path(worker_dir)

    import lightkurve as lk  # noqa: F401 -- imported for _read_lightcurve's lk.read/lk.LightCurve calls
    import bls_fast
    import bls as bls_dispatch
    import wotan_detrend
    import cnn_ffi
    import exominer_ffi_inference

    try:
        import bls_cuda
        cuda_available = bls_cuda.is_cuda_available()
    except Exception as exc:
        bls_cuda = None
        cuda_available = False
        print(f"benchmark_python_stages: bls_cuda unavailable ({exc}) -- stage (d) will be null", file=sys.stderr)

    if args.fits_dir:
        inputs = sorted(p for p in Path(args.fits_dir).iterdir() if p.suffix == ".fits")
    else:
        inputs = [Path(p) for p in args.fits]
    if not inputs:
        print("benchmark_python_stages: no input FITS files", file=sys.stderr)
        return 2

    models_dir = Path(args.models_dir) if args.models_dir else (worker_dir / "models")
    cnn, exominer = load_classifiers(models_dir, cnn_ffi, exominer_ffi_inference) if models_dir.exists() else (None, None)

    file_results = []
    for path in inputs:
        print(f"benchmark_python_stages: {path}", file=sys.stderr)
        caveats = []

        # (a) FITS read.
        stage_a = time_stage(args.iterations, lambda: _read_lightcurve(str(path)))
        lc0 = _read_lightcurve(str(path))

        # (b) preprocess + full iterative detrend.
        stage_b = time_stage(args.iterations, lambda: _prepare_and_detrend(lc0, bls_fast, wotan_detrend))
        t_arr, flat_arr, raw_arr = _prepare_and_detrend(lc0, bls_fast, wotan_detrend)

        # (c) BLS two-pass CPU, production grid, CPU-explicit.
        stage_c = time_stage(args.iterations, lambda: bls_fast.bls_two_pass(
            t_arr, flat_arr, period_min=PERIOD_MIN, period_max=PERIOD_MAX,
            n_periods=N_PERIODS, n_bins=N_BINS, n_dur_steps=N_DUR_STEPS))
        cpu_bls = bls_fast.bls_two_pass(
            t_arr, flat_arr, period_min=PERIOD_MIN, period_max=PERIOD_MAX,
            n_periods=N_PERIODS, n_bins=N_BINS, n_dur_steps=N_DUR_STEPS)

        # (d) BLS two-pass CUDA -- only when a real CUDA GPU is usable.
        if cuda_available:
            stage_d = time_stage(args.iterations, lambda: bls_cuda.bls_gpu_two_pass(
                t_arr, flat_arr, period_min=PERIOD_MIN, period_max=PERIOD_MAX,
                n_periods=N_PERIODS, n_bins=N_BINS, n_dur_steps=N_DUR_STEPS))
        else:
            stage_d = None
            caveats.append("bls_cuda.is_cuda_available() is False on this host -- stage (d) is null")

        # (e) best-effort full-pipeline equivalent: see module doc comment
        # for the exact scope gap vs. Rust's multi_planet::search.
        def _pipeline_once():
            t2, flat2, _raw2 = _prepare_and_detrend(lc0, bls_fast, wotan_detrend)
            seed = bls_dispatch.run_bls(t2, flat2, period_min=PERIOD_MIN, period_max=PERIOD_MAX, n_periods=N_PERIODS)
            resolve_harmonic(t2, flat2, seed, period_min=PERIOD_MIN, period_max=PERIOD_MAX, bls_fn=bls_dispatch.run_bls)

        stage_e = time_stage(args.iterations, _pipeline_once)
        caveats.append(
            "stage (e) is a best-effort real-code equivalent (preprocess+detrend -> "
            "bls.run_bls production dispatcher -> resolve_harmonic), covering only the "
            "FIRST iteration of worker.py::process_task's multi-planet loop. It does NOT "
            "reproduce max_planets=5 iterations, accumulated-mask re-detrend, protected "
            "slots, or the long-period sweep, because that loop lives inline in "
            "process_task (~1000 lines entangled with self state/config/network) with no "
            "standalone function to call -- unlike the Rust side, which factored this into "
            "multi_planet::search + vetting::vet_all (following the Swift reference port's "
            "clean-room module design). This makes stage (e)'s ratio the LEAST reliable "
            "number in this benchmark; treat stages (a)-(d) and (f) as the primary evidence."
        )

        # (f) ONNX CNN + ExoMiner FFI, scored against the stage (c) BLS peak
        # (same choice as the Rust side, see exonit_bench.rs's caveat).
        if cnn is not None:
            stage_f_cnn = time_stage(args.iterations, lambda: cnn.classify(
                t_arr, flat_arr, cpu_bls["period"], cpu_bls["t0"], cpu_bls["duration"]))
        else:
            stage_f_cnn = None
            caveats.append("CNN FFI model not loaded (missing --models-dir/cnn_ffi_v1.onnx) -- stage (f) cnn is null")
        if exominer is not None:
            stage_f_exominer = time_stage(args.iterations, lambda: exominer.score(
                t_arr, flat_arr, cpu_bls["period"], cpu_bls["t0"], cpu_bls["duration"],
                cpu_bls["depth_ppm"], cpu_bls["snr"], stellar=None))
        else:
            stage_f_exominer = None
            caveats.append("ExoMiner FFI model not loaded (missing --models-dir/exominer_ffi_v6_ensemble.onnx) -- stage (f) exominer is null")
        caveats.append(
            "stage (f) scores the stage (c) BLS two-pass PEAK (period/t0/duration/depth_ppm/snr), "
            "not a vetted candidate from stage (e) -- matches the Rust side's equivalent choice "
            "in exonit_bench.rs (a quiet light curve can legitimately produce zero candidates)."
        )
        caveats.append(
            "stage (f) exominer.score() is called with stellar=None (no per-TIC server metadata "
            "offline) -- matches the Rust side's FfiStellarMetadata::default(); affects only the "
            "`stellar` feature vector's z-scored values, not timing (full ONNX forward pass either way)."
        )
        caveats.append(
            "stage (a) times _read_lightcurve, which returns a full lightkurve.LightCurve object "
            "(astropy Time/Quantity wrapping) for ALL formats -- this is what worker.py's real "
            "production code does, but it is strictly more work than the Rust side's fits.rs, "
            "which returns raw Vec<f64> with no wrapper object. This asymmetry is real production "
            "behaviour on both sides, not a benchmark artifact, and is the single largest known "
            "source of stage (a)'s ratio."
        )

        file_results.append({
            "path": str(path),
            "raw_points": len(lc0.time.value),
            "processed_points": len(t_arr),
            "sector_count": 1,
            "bls_two_pass_cpu_peak": {
                "period_days": cpu_bls["period"],
                "duration_days": cpu_bls["duration"],
                "snr": cpu_bls["snr"],
                "depth_ppm": cpu_bls["depth_ppm"],
            },
            "stages": {
                "fits_read": stage_a,
                "preprocess_detrend": stage_b,
                "bls_two_pass_cpu": stage_c,
                "bls_two_pass_cuda": stage_d,
                "pipeline_full": stage_e,
                "onnx_cnn": stage_f_cnn,
                "onnx_exominer": stage_f_exominer,
            },
            "caveats": caveats,
        })

    output = {
        "tool": "benchmark_python_stages.py",
        "worker_repo": "exonit (production worker/)",
        "git_sha": git_sha(str(worker_dir)),
        "python_version": sys.version.split()[0],
        "host": socket.gethostname(),
        "cpu_model": cpu_model(),
        "iterations_per_stage": args.iterations,
        "flags": {
            "cuda_feature": bls_cuda is not None,
            "cuda_gpu_probed": cuda_available,
            "bls_dispatch_backend": bls_dispatch.get_backend_name(),
        },
        "bls_params": {
            "period_minimum": PERIOD_MIN,
            "period_maximum": PERIOD_MAX,
            "period_count": N_PERIODS,
            "bin_count": N_BINS,
            "duration_step_count": N_DUR_STEPS,
        },
        "models_dir": str(models_dir) if models_dir.exists() else None,
        "files": file_results,
    }

    text = json.dumps(output, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
