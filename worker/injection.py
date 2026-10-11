"""Injection-recovery tests for the remote worker.

Auto-detects GPU backend: CUDA (CuPy) > Metal/MLX-fused > Metal/MLX (legacy
per-flux) > OpenCL > CPU (numpy). Runs sequentially in a single process —
GPU parallelizes internally. All batched backends amortise the phase-fold
setup across a period-cell's fluxes; Metal/MLX-fused additionally runs the
whole (flux x period) cell in one kernel launch (~6x over per-flux MLX on
Apple Silicon, see bls_metal_batch.py).
"""

import logging
import threading
import numpy as np
from bls_fast import bls_fast, bls_fast_batch
# `run_injection_recovery` / `_run_batched_gpu` take a positional parameter
# literally named `time` (light-curve timestamps), which shadows the `time`
# module inside those functions. Import `monotonic` under an alias so the
# wall-clock deadline check works regardless (monotonic is also the correct
# clock for measuring elapsed time, immune to system clock adjustments).
from time import monotonic as _clock

from phase_fold import phase_fold

EARTH_RADIUS_TO_SOLAR = 0.00916  # R_earth / R_sun
AU_TO_RSUN = 214.94  # 1 AU expressed in solar radii


def transit_duration_days(period_days, stellar_radius_solar=1.0,
                          stellar_mass_solar=None, impact_parameter=0.0,
                          planet_radius_earth=1.0):
    # T14 for an edge-on circular orbit (Seager & Mallén-Ornelas 2003):
    #   T14 = (P/π) · arcsin( (R*/a) · √((1+k)² - b²) )
    # Kepler's third law gives a/R_sun = 214.94 · (M_sun·(P/yr)²)^(1/3).
    # When stellar_mass_solar is None, fall back to an MS scaling M ≈ R^1.25
    # — accurate enough for injection grids where Rp-induced variation is
    # already dominated by b and P.
    if stellar_mass_solar is None:
        stellar_mass_solar = stellar_radius_solar ** 1.25
    a_au = (stellar_mass_solar * (period_days / 365.25) ** 2) ** (1.0 / 3.0)
    a_rsun = a_au * AU_TO_RSUN
    k = (planet_radius_earth * EARTH_RADIUS_TO_SOLAR) / stellar_radius_solar
    arg_sq = (1.0 + k) ** 2 - impact_parameter ** 2
    if arg_sq <= 0:
        return 0.0
    sin_arg = stellar_radius_solar / a_rsun * (arg_sq ** 0.5)
    if sin_arg >= 1.0:
        return period_days / 2.0
    return period_days / np.pi * np.arcsin(sin_arg)

# Auto-detect best available GPU backend
_bls_gpu = None
_bls_gpu_batch = None
_gpu_backend = "CPU"

# Try CUDA first (NVIDIA GPUs)
try:
    from bls_cuda import bls_gpu, bls_gpu_batch as _cuda_batch, is_cuda_available
    if is_cuda_available():
        _bls_gpu = bls_gpu
        _bls_gpu_batch = _cuda_batch
        _gpu_backend = "CUDA"
except Exception as _e:
    # Never silent: GPU host degrading to CPU must leave a trace
    # (worker.py escalates to ERROR when gpu_enabled=true ends up on CPU).
    logging.getLogger("exonit").warning("Injection CUDA backend unavailable: %s", _e)

# Central Metal injection kill-switch check — read EXONIT_INJECTION_METAL
# exactly ONCE here, before EITHER Metal backend below is attempted. This is
# defense-in-depth *on top of* each backend module's own check
# (bls_metal_batch.is_available() / bls_metal.is_metal_available(), both via
# metal_serial.injection_metal_disabled_by_env): the 2026-07-12 re-review
# (Finding 1) found that the legacy bls_metal path used to not read this env
# var at all, so this auto-detect chain could still resolve to Metal/MLX
# with the kill-switch set — masked in production only by a worker.py check
# that fires solely when FPP's backend resolves to "metal-kernel" (a
# different, independent condition). Gating the chain itself here means the
# kill-switch's documented contract ("disables this backend unconditionally")
# holds regardless of what either per-module check or FPP's own backend
# resolution do.
from metal_serial import injection_metal_disabled_by_env
_injection_metal_disabled_by_env = injection_metal_disabled_by_env()
if _injection_metal_disabled_by_env:
    logging.getLogger("exonit").info(
        "Injection Metal disabled by EXONIT_INJECTION_METAL=0/false -- "
        "skipping both Metal/MLX backends (fused + legacy), falling "
        "through to OpenCL/CPU.")

# Try Metal/MLX (Apple Silicon) — fused single-launch kernel first (one
# GPU thread per (flux, period), no per-flux mx.eval sync; validated
# 2026-07-11 on M1/M3/M4, see
# docs/superpowers/specs/2026-07-11-gpu-bls-f32-unified-kernel.md). Falls
# back to bls_metal's per-flux array-op path (unchanged behaviour) when the
# fused module can't be used here — MLX/metal_kernel missing, SoC not
# allowlisted, or the EXONIT_INJECTION_METAL=0 kill-switch.
if _bls_gpu is None and not _injection_metal_disabled_by_env:
    try:
        from bls_metal_batch import (bls_gpu as _metal_bls_v2,
                                      bls_gpu_batch_v2 as _metal_batch_v2,
                                      is_available as _metal_v2_available)
        if _metal_v2_available():
            _bls_gpu = _metal_bls_v2
            _bls_gpu_batch = _metal_batch_v2
            _gpu_backend = "Metal/MLX-fused"
    except Exception as _e:
        logging.getLogger("exonit").warning(
            "Injection Metal fused-kernel backend unavailable: %s", _e)

# Try Metal/MLX (Apple Silicon) — scatter-based binning + batch support
if _bls_gpu is None and not _injection_metal_disabled_by_env:
    try:
        from bls_metal import bls_gpu as _metal_bls, bls_gpu_batch as _metal_batch, is_metal_available
        if is_metal_available():
            _bls_gpu = _metal_bls
            _bls_gpu_batch = _metal_batch
            _gpu_backend = "Metal/MLX"
    except Exception as _e:
        logging.getLogger("exonit").warning("Injection Metal backend unavailable: %s", _e)

# Central OpenCL injection kill-switch check — read EXONIT_INJECTION_OPENCL
# exactly ONCE here, mirroring the Metal central check above. Defense in
# depth on top of bls_opencl.is_opencl_available()'s own check (which
# already reads opencl_serial.injection_opencl_disabled_by_env() before
# doing anything else) — same single-source-of-truth contract as the
# Metal kill-switch (2026-07-12 re-review, Finding 1), and the same
# module that also gates fpp_opencl.py's guard (opencl_serial.py). Gating
# the chain itself here means an operator sees ONE clear log line even
# when OpenCL would otherwise have resolved.
from opencl_serial import injection_opencl_disabled_by_env
_injection_opencl_disabled_by_env = injection_opencl_disabled_by_env()
if _injection_opencl_disabled_by_env:
    logging.getLogger("exonit").info(
        "Injection OpenCL disabled by EXONIT_INJECTION_OPENCL=0/false -- "
        "skipping the OpenCL backend, falling through to CPU.")

# Try OpenCL if no CUDA/Metal (AMD/Intel GPUs — e.g. Intel Arc iGPU on
# <host>). Spec 2026-07-24: this now uses the fused f32 kernel
# (worker/bls_opencl.py, same design as the Metal/CUDA backends above),
# not the pre-07-11 per-work-item design that used to live there.
# is_opencl_available() performs a REAL minimal kernel submission (build +
# launch + read-back), not just an import/context-creation check — see
# disable_gpu()'s docstring below for why that distinction is exactly
# what the old AMD Vega TDR incident was missing.
if _bls_gpu is None and not _injection_opencl_disabled_by_env:
    try:
        from bls_opencl import (bls_gpu as _opencl_bls,
                                 bls_gpu_batch as _opencl_bls_batch,
                                 is_opencl_available)
        if is_opencl_available():
            _bls_gpu = _opencl_bls
            _bls_gpu_batch = _opencl_bls_batch
            _gpu_backend = "OpenCL"
    except Exception as _e:
        logging.getLogger("exonit").warning("Injection OpenCL backend unavailable: %s", _e)


def disable_gpu():
    """Force injection-recovery onto CPU (numpy BLS).

    Call this early (before the first ``run_injection_recovery``) when
    the runtime GPU is unsuitable for long injection kernels — the worker
    does this when its config has ``gpu_enabled=false``.

    Motivation: AMD Vega integrated GPUs hit the amdgpu kernel timeout
    (TDR) during injection on LCs with tens of thousands of points,
    killing the worker process (``amdgpu: The CS has cancelled because
    the context is lost``). Falling back to CPU avoids the crash.

    Historical note (spec 2026-07-24): before this date, this TDR history
    was also the reason the OpenCL backend itself was permanently retired
    from the auto-detect chain above (the pre-07-11 per-work-item OpenCL
    design, tuned by hand for Vega, still crashed under contention — see
    the historical comment in ``worker/bls_opencl.py``). That backend is
    eligible again as of the 07-24 rewrite: the fused f32 kernel design
    (same as Metal/CUDA) plus a REAL submission smoke test in
    ``bls_opencl.is_opencl_available()`` (build + launch + read-back, not
    just import/context-creation) means a host whose GPU genuinely can't
    run the kernel now fails that probe and falls through to CPU via the
    auto-detect chain, rather than "succeeding" at import time and
    crashing later mid-injection as the old design did. This function
    remains the generic, config-driven kill-switch for EVERY GPU backend
    alike (CUDA/Metal/OpenCL) when ``gpu_enabled=false`` in a host's
    config — it is not itself OpenCL-specific, and a Vega-class host is
    simply expected to either fail the new smoke test on its own or run
    with ``gpu_enabled=false``, rather than being hardcoded out of the
    chain.
    """
    global _bls_gpu, _bls_gpu_batch, _gpu_backend
    _bls_gpu = None
    _bls_gpu_batch = None
    _gpu_backend = "CPU (disabled by config)"


def get_backend_name() -> str:
    """Return the currently active injection GPU backend name."""
    return _gpu_backend


def is_fused_metal_active() -> bool:
    """True iff the fused Metal kernel (bls_metal_batch) is the currently
    active injection GPU backend.

    Single source of truth for worker.py's metal-kernel routing decision —
    callers must use this instead of comparing ``get_backend_name()`` to
    the ``"Metal/MLX-fused"`` string literal directly, so a future label
    rename can't silently decouple the two call sites.
    """
    return _gpu_backend == "Metal/MLX-fused"


# --- GPU batch-call circuit breaker -----------------------------------
# If a GPU batch backend (Metal-fused, Metal-legacy, CUDA, OpenCL) is
# structurally broken on a given host -- e.g. a Metal-compiler quirk the
# cheap is_available() smoke test (mx.eval(mx.sum(...)), never the actual
# custom kernel source) doesn't catch -- every period-cell of every star
# would otherwise retry GPU->CPU and log a WARNING for the whole run
# (bounded per-cell, but noisy and wasteful at 100k-star scale). After
# EXONIT_INJECTION_GPU_FAILURE_THRESHOLD consecutive failures, log ONE
# ERROR and permanently route this process's injection-recovery straight
# to the CPU batch path -- no further per-cell retries or warnings.
# Thread-safe: injection runs on a background thread (worker.py) while the
# main thread may be doing FPP concurrently; a single success resets the
# streak (a backend that mostly works, with the odd transient glitch,
# should NOT trip the breaker).
_GPU_FAILURE_THRESHOLD = 3
_gpu_failure_lock = threading.Lock()
_gpu_consecutive_failures = 0
_gpu_circuit_open = False


def reset_gpu_circuit_breaker():
    """Reset the GPU-failure circuit breaker (test isolation / explicit
    re-arm after a backend change). Not called anywhere in production —
    the breaker is meant to stay open for the remainder of the process
    once tripped."""
    global _gpu_consecutive_failures, _gpu_circuit_open
    with _gpu_failure_lock:
        _gpu_consecutive_failures = 0
        _gpu_circuit_open = False


def gpu_circuit_is_open() -> bool:
    """True once the circuit breaker has tripped (GPU batch path
    permanently disabled for the remainder of this process)."""
    return _gpu_circuit_open


def _record_gpu_batch_success():
    global _gpu_consecutive_failures
    with _gpu_failure_lock:
        _gpu_consecutive_failures = 0


def _record_gpu_batch_failure(backend_label):
    """Increment the consecutive-failure streak; trip the breaker (one
    ERROR, permanent CPU fallback for this process) on the Nth in a row."""
    global _gpu_consecutive_failures, _gpu_circuit_open
    with _gpu_failure_lock:
        _gpu_consecutive_failures += 1
        streak = _gpu_consecutive_failures
        just_tripped = (not _gpu_circuit_open
                       and streak >= _GPU_FAILURE_THRESHOLD)
        if just_tripped:
            _gpu_circuit_open = True
    if just_tripped:
        logging.getLogger("exonit").error(
            "Injection GPU backend (%s) failed %d consecutive times -- "
            "disabling it for the remainder of this process; every "
            "further injection-recovery call will use the CPU batch path "
            "(bls_fast_batch) directly, with no further per-cell retries "
            "or warnings for this backend.", backend_label, streak)


def radius_to_depth(radius_earth, stellar_radius_solar=1.0):
    rp_rs = (radius_earth * EARTH_RADIUS_TO_SOLAR) / stellar_radius_solar
    return rp_rs ** 2


def inject_transit(time, flux, period, t0, depth, duration):
    flux_out = flux.copy()
    # See docs/reviews/2026-07-02-opus48-cleanup-review.md item D5 and
    # tests/worker/test_phase_fold_golden.py for the bit-identical proof.
    phase = phase_fold(time, t0, period, fold_order="mod_div", center=None)
    half_dur = (duration / period) / 2
    in_transit = (phase < half_dur) | (phase > 1.0 - half_dur)
    flux_out[in_transit] -= depth
    return flux_out


def _run_batched_gpu(time, flux, stellar_radius, stellar_mass,
                       radius_grid, period_grid,
                       n_injections_per_cell, snr_threshold, time_span,
                       batch_fn=None, backend_label=None,
                       bls_n_periods=5000, bls_n_bins=150,
                       bls_n_dur_steps=10, max_seconds=None):
    """Batched injection-recovery for any backend that exposes
    `batch_fn(time, flux_list, ...)` with the shared ``bls_gpu_batch``
    signature — Metal (MLX), OpenCL, and the CPU numba fallback via
    ``bls_fast.bls_fast_batch``. Groups injections by period (same BLS
    range) so the 90 injections per period-cell reuse phase-fold setup
    instead of doing 90 independent searches.

    When ``batch_fn`` is None, falls back to the module global
    ``_bls_gpu_batch`` (backwards-compatible with direct callers).

    ``max_seconds``: optional wall-clock deadline (elapsed since this
    call started). When set and exceeded, remaining period-cells are
    skipped (their ``recovery_counts`` stay 0) and the returned dict is
    flagged ``truncated=True``. Default ``None`` = no cap, i.e. exactly
    the previous behaviour — this guard only exists for pathological
    giant multi-sector FFI targets that would otherwise monopolise a
    worker for 30-45 min on CPU."""
    if batch_fn is None:
        batch_fn = _bls_gpu_batch
    if backend_label is None:
        backend_label = _gpu_backend
    total = len(radius_grid) * len(period_grid) * n_injections_per_cell
    logging.getLogger("exonit").debug(f"Injection-recovery: {total} injections ({backend_label} batched)")

    recovery_counts = np.zeros((len(radius_grid), len(period_grid)), dtype=int)
    truncated = False
    periods_evaluated = 0
    t_start = _clock()

    for pi, period in enumerate(period_grid):
        if max_seconds is not None and (_clock() - t_start) > max_seconds:
            truncated = True
            logging.getLogger("exonit").warning(
                "Injection-recovery exceeded max_seconds=%.1f "
                "(elapsed=%.1fs, periods_done=%d/%d, n_points=%d) -> "
                "truncating with partial completeness",
                max_seconds, _clock() - t_start, pi, len(period_grid), len(time))
            break
        # All injections for this period share the same BLS range
        p_min = max(0.5, period * 0.5)
        p_max = min(period * 2.0, time_span)

        # Pre-generate all flux arrays for this period group
        flux_batch = []
        batch_info = []  # (ri, k, target_period)

        for ri, radius in enumerate(radius_grid):
            depth = radius_to_depth(radius, stellar_radius)
            duration = max(0.02, min(transit_duration_days(
                period, stellar_radius_solar=stellar_radius,
                stellar_mass_solar=stellar_mass,
                planet_radius_earth=radius), 1.0))
            for k in range(n_injections_per_cell):
                seed = ri * len(period_grid) * n_injections_per_cell + pi * n_injections_per_cell + k
                rng = np.random.RandomState(seed)
                t0 = time[0] + rng.uniform(0, period)
                flux_injected = inject_transit(time, flux, period, t0, depth, duration)
                flux_batch.append(flux_injected)
                batch_info.append((ri, k))

        # Run entire batch with shared phase-fold. GPU kernels (Metal
        # command-buffer errors, CUDA context glitches, OOM under
        # multi-instance contention) can fail transiently without the host
        # being unhealthy — retry THIS cell once on CPU rather than losing
        # the whole star's completeness map on a 100k-star run. Never
        # silent: always logged as a WARNING. bls_fast_batch already runs
        # inside numba_parallel_guard() internally (see bls_fast.py) — do
        # NOT wrap it in a second guard here (see
        # gotcha_mac_fpp_gpu_fallback_unguarded_numba_crash). If batch_fn
        # already IS bls_fast_batch (no-GPU host), there is no different
        # backend to fall back to, so the exception propagates as before.
        #
        # Circuit breaker: once EXONIT_INJECTION_GPU_FAILURE_THRESHOLD
        # consecutive GPU failures have tripped it (see
        # _record_gpu_batch_failure), skip the GPU attempt entirely for
        # every remaining cell of every remaining star in this process —
        # a structurally broken kernel would otherwise retry-and-warn on
        # every single one of a 100k-star run's period-cells.
        is_gpu_batch_fn = batch_fn is not bls_fast_batch
        if is_gpu_batch_fn and _gpu_circuit_open:
            results = bls_fast_batch(time, flux_batch,
                                     period_min=p_min, period_max=p_max,
                                     n_periods=bls_n_periods, n_bins=bls_n_bins,
                                     n_dur_steps=bls_n_dur_steps)
        else:
            try:
                results = batch_fn(time, flux_batch,
                                   period_min=p_min, period_max=p_max,
                                   n_periods=bls_n_periods, n_bins=bls_n_bins,
                                   n_dur_steps=bls_n_dur_steps)
                if is_gpu_batch_fn:
                    _record_gpu_batch_success()
            except Exception as e:
                if not is_gpu_batch_fn:
                    raise
                _record_gpu_batch_failure(backend_label)
                logging.getLogger("exonit").warning(
                    "Injection-recovery batch_fn (%s) failed on period-cell "
                    "pi=%d/%d (period=%.4f d): %s -> retrying this cell on "
                    "CPU (bls_fast_batch)",
                    backend_label, pi, len(period_grid), period, e)
                results = bls_fast_batch(time, flux_batch,
                                         period_min=p_min, period_max=p_max,
                                         n_periods=bls_n_periods, n_bins=bls_n_bins,
                                         n_dur_steps=bls_n_dur_steps)

        # Check recoveries
        for idx, (ri, k) in enumerate(batch_info):
            r = results[idx]
            period_match = abs(r["period"] - period) / period < 0.05
            if period_match and r["snr"] >= snr_threshold:
                recovery_counts[ri, pi] += 1
        periods_evaluated += 1

    completeness = recovery_counts / n_injections_per_cell
    details = []
    for ri, radius in enumerate(radius_grid):
        depth = radius_to_depth(radius, stellar_radius)
        for pi, period in enumerate(period_grid):
            details.append({
                "radius_earth": float(radius),
                "period_days": float(period),
                "depth_ppm": float(depth * 1e6),
                "detection_prob": float(completeness[ri, pi]),
                "n_recovered": int(recovery_counts[ri, pi]),
                "n_injected": n_injections_per_cell,
            })

    logging.getLogger("exonit").debug(f"Injection-recovery: {total} injections complete ({backend_label} batched)")

    return {
        "radius_grid": radius_grid.tolist(),
        "period_grid": period_grid.tolist(),
        "completeness_map": completeness.tolist(),
        "details": details,
        "stellar_radius": stellar_radius,
        "truncated": bool(truncated),
        "periods_evaluated": periods_evaluated,
    }


def run_injection_recovery(time, flux, stellar_radius=1.0,
                           stellar_mass=None,
                           radius_grid=None, period_grid=None,
                           n_injections_per_cell=10, snr_threshold=7.0,
                           bls_n_periods=5000, bls_n_bins=150,
                           bls_n_dur_steps=10, max_seconds=None):
    """Run injection-recovery test.

    Uses the best available BLS backend: GPU batched (Metal/OpenCL/CUDA)
    > CPU batched (bls_fast_batch, numba, shared phase-fold) > individual
    bls_fast calls. The CPU batched path collapses the 90 injections per
    period-cell from 90 individual kernels to 1 kernel that amortises the
    phase-fold setup — measurable ~2-3× speedup on the M1 for the default
    grid, and it's what lets the <host> and gpu-disabled workers
    contribute meaningfully to a multi-worker scorecard.

    ``n_injections_per_cell`` defaults to 10 (Christiansen+ 2015 recommends
    hundreds per cell; 10 keeps Bernoulli sigma <=0.16 per cell at a runtime
    cost the batched path absorbs comfortably). The BLS search grid defaults
    stay unchanged but are configurable for controlled worker A/B runs.

    ``max_seconds``: optional wall-clock deadline for the whole call
    (default ``None`` = no cap, byte-identical to the pre-timeout
    behaviour). Guards against giant multi-sector FFI targets whose CPU
    injection-recovery would otherwise monopolise a worker for 30-45 min
    (e.g. TIC 243278113, 173k points / 43 sectors). When exceeded, the
    remaining period-cells (or radius-cells, in the per-call fallback) are
    skipped and the returned dict has ``truncated=True`` with whatever
    partial ``completeness_map`` was computed so far.
    """
    if radius_grid is None:
        radius_grid = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0])
    if period_grid is None:
        time_span = time[-1] - time[0]
        period_grid = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0])
        period_grid = period_grid[period_grid < time_span / 2]

    time_span = time[-1] - time[0]

    # Pick the best batched BLS available: GPU batched first, else CPU batched.
    batch_fn = _bls_gpu_batch
    backend_label = _gpu_backend
    if batch_fn is None:
        try:
            from bls_fast import bls_fast_batch
            batch_fn = bls_fast_batch
            backend_label = f"{_gpu_backend} (CPU batched via numba)"
        except ImportError:  # pragma: no cover - numba always present in prod
            batch_fn = None

    if batch_fn is not None:
        return _run_batched_gpu(
            time, flux, stellar_radius, stellar_mass,
            radius_grid, period_grid,
            n_injections_per_cell, snr_threshold, time_span,
            batch_fn=batch_fn, backend_label=backend_label,
            bls_n_periods=bls_n_periods, bls_n_bins=bls_n_bins,
            bls_n_dur_steps=bls_n_dur_steps, max_seconds=max_seconds)

    # Last-resort fallback: per-call BLS loop (only when numba is missing).
    bls_fn = _bls_gpu if _bls_gpu else bls_fast
    total = len(radius_grid) * len(period_grid) * n_injections_per_cell
    logging.getLogger("exonit").debug(f"Injection-recovery: {total} injections ({_gpu_backend})")

    recovery_counts = np.zeros((len(radius_grid), len(period_grid)), dtype=int)
    seed_counter = 0
    truncated = False
    radii_evaluated = 0
    t_start = _clock()

    for ri, radius in enumerate(radius_grid):
        if max_seconds is not None and (_clock() - t_start) > max_seconds:
            truncated = True
            logging.getLogger("exonit").warning(
                "Injection-recovery (per-call fallback) exceeded "
                "max_seconds=%.1f (elapsed=%.1fs, radii_done=%d/%d, "
                "n_points=%d) -> truncating with partial completeness",
                max_seconds, _clock() - t_start, ri, len(radius_grid), len(time))
            break
        depth = radius_to_depth(radius, stellar_radius)
        for pi, period in enumerate(period_grid):
            duration = max(0.02, min(transit_duration_days(
                period, stellar_radius_solar=stellar_radius,
                stellar_mass_solar=stellar_mass,
                planet_radius_earth=radius), 1.0))
            for k in range(n_injections_per_cell):
                rng = np.random.RandomState(seed_counter)
                seed_counter += 1
                t0 = time[0] + rng.uniform(0, period)
                flux_injected = inject_transit(time, flux, period, t0, depth, duration)
                result = bls_fn(time, flux_injected,
                                period_min=max(0.5, period * 0.5),
                                period_max=min(period * 2.0, time_span),
                                n_periods=bls_n_periods,
                                n_bins=bls_n_bins,
                                n_dur_steps=bls_n_dur_steps)
                period_match = abs(result["period"] - period) / period < 0.05
                if period_match and result["snr"] >= snr_threshold:
                    recovery_counts[ri, pi] += 1
        radii_evaluated += 1

    completeness = recovery_counts / n_injections_per_cell
    details = []
    for ri, radius in enumerate(radius_grid):
        depth = radius_to_depth(radius, stellar_radius)
        for pi, period in enumerate(period_grid):
            details.append({
                "radius_earth": float(radius),
                "period_days": float(period),
                "depth_ppm": float(depth * 1e6),
                "detection_prob": float(completeness[ri, pi]),
                "n_recovered": int(recovery_counts[ri, pi]),
                "n_injected": n_injections_per_cell,
            })

    logging.getLogger("exonit").debug(f"Injection-recovery: {total} injections complete ({_gpu_backend})")

    return {
        "radius_grid": radius_grid.tolist(),
        "period_grid": period_grid.tolist(),
        "completeness_map": completeness.tolist(),
        "details": details,
        "stellar_radius": stellar_radius,
        "truncated": bool(truncated),
        "radii_evaluated": radii_evaluated,
    }
