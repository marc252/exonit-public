"""Fused OpenCL kernel for batched BLS injection-recovery (Intel Arc iGPU).

Spec ``docs/superpowers/specs/2026-07-24-opencl-injection-backend-arc.md``:
this REPLACES the pre-07-11 per-flux/legacy-batch OpenCL design that used to
live in this file (per-work-item static ``bin_sums[200]``/``cum_sums[400]``
arrays, tuned by hand for an AMD Vega iGPU under Mesa rusticl/clover, and
disabled in production after amdgpu TDR crashes -- see the historical
``PERIOD_BATCH`` comment below) with the SAME fused f32 kernel design already
validated on Metal (``worker/bls_metal_batch.py``, spec 07-11) and CUDA
(``worker/bls_cuda.py``'s ``_bls_gpu_batch_f32``, spec 07-11 Phase 2). One
algorithm, three GPU backends -- see those two modules' docstrings for the
full "why fused" motivation (the injection stage is the dominant per-star
wall-clock cost, and a per-flux Python-loop kernel launch never saturates the
GPU).

Design (mirrors ``bls_fast.bls_fast_batch`` semantics exactly, byte-for-byte
with the Metal/CUDA kernels):
- Phase-fold on CPU in float64 (exact, same as numba: ph=(t/P)%1,
  b=int(ph*n_bins), clipped) -> int16 bin indices, shared by all fluxes of a
  period chunk (``_phase_fold_cpu`` below, duplicated rather than imported --
  same independence rationale as ``bls_cuda.py``'s own copy: this module must
  import cleanly without any other GPU backend module).
- Per-period bin COUNTS precomputed on CPU (exact, vectorized bincount),
  uploaded once per period chunk.
- One OpenCL work-item per (flux, period): work-item 0 in each work-group
  builds the period's cumulative bin counts in local memory + barrier; every
  work-item then accumulates its own private per-bin flux sums (f32, sample
  order identical to numba: ``for i in range(N)``), prefix-sums them in
  place, then loops (width ascending, phase-bin ascending) with STRICT '>'
  updates -- byte-for-byte the numba tie-breaking order (sequential inside
  the work-item, so no parallel reduction to reason about, no atomics).
- Flux is mean-centered in f64 before the f32 cast (depth = mean_out -
  mean_in is shift-invariant; centering crushes f32 accumulation error below
  the ppm floor -- same proven trick as the Metal/CUDA kernels).
- f32 window sums come from prefix-sum differences (not incremental slide),
  so rounding is bounded and deterministic.

NDRange geometry: global_size=(n_flux_pad, n_per), local_size=(TG_FLUX, 1) --
the OpenCL analogue of Metal's ``dispatchThreads``/threadgroup and CUDA's
grid/block. ``n_flux_pad`` is always a multiple of ``TG_FLUX`` by
construction, so this respects OpenCL <2.0's "global must be a multiple of
local" rule without needing the non-uniform-work-group extension.

Local memory budget: unlike Metal (fixed 32 KB/threadgroup on every Apple
GPU family) and CUDA (fixed 48 KB/block default on every compute-capability
>=7.0 GPU in the fleet), OpenCL devices vary (spec Annex/§1: "mesurar en
execucio, no assumir 32KB/48KB"). This module queries
``CL_DEVICE_LOCAL_MEM_SIZE`` / ``CL_DEVICE_MAX_WORK_GROUP_SIZE`` from the
active device at runtime instead of hardcoding a platform constant (see
``_query_local_mem_size`` / ``_query_max_work_group_size`` /
``_check_kernel_limits`` below). Measured 2026-07-24 on <host>'s Arc iGPU:
``CL_DEVICE_LOCAL_MEM_SIZE=65536`` (64 KB), ``CL_DEVICE_MAX_WORK_GROUP_SIZE=
1024`` -- comfortably over both the Metal and CUDA budgets for the
production n_bins=150 default (19,932 B, see ``_local_mem_bytes``).

Kernel is 100% f32 (fp64 IS available on this Arc iGPU via ``cl_khr_fp64``,
per spec §1, but intentionally unused in the kernel -- the CPU-side fold
stays f64, matching the Metal/CUDA design exactly rather than reopening a
"maybe f64 helps here" question the other two backends already settled).

Kill-switch: env ``EXONIT_INJECTION_OPENCL`` = ``0``/``false`` disables this
backend unconditionally (escape hatch if Arc misbehaves under multi-instance
load, without a code change, same contract as ``EXONIT_INJECTION_METAL`` --
see ``opencl_serial.injection_opencl_disabled_by_env``); anything else,
including unset, leaves it enabled.

Thread-safety: every real OpenCL submission (context/program build, kernel
launch, buffer read-back) goes through ``opencl_serial.opencl_guard()`` -- a
single process-wide ``RLock`` shared with ``fpp_opencl.py`` -- because the
module-level ``_ctx``/``_queue``/kernel-cache globals below are exactly the
"shared state, no lock" pattern that caused the 2026-07-12 MLX/Metal
incident (see ``opencl_serial.py``'s docstring). CPU-only numpy work (fold,
bincount, mean-centering) stays OUTSIDE every guarded region.

Backward compatibility: ``worker/injection.py`` still imports the legacy
names ``bls_opencl``, ``bls_opencl_batch``, ``_init_opencl`` from this module
(spec iteration 2 rewires that hook to the new ``bls_gpu``/``bls_gpu_batch``
names) -- kept here as thin aliases so the existing auto-detect chain in
``injection.py`` keeps working unmodified until that follow-up lands.
"""

import logging
import os
import threading
import time as _time

import numpy as np

from opencl_serial import opencl_guard, injection_opencl_disabled_by_env

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

# Canonical grid + SDE from the CPU reference (single source of truth,
# same imports bls_metal_batch.py / bls_cuda.py use).
from bls_fast import _compute_sde, _frequency_period_grid

TG_FLUX = 32

# Period chunk keeps bin_idx (chunk x N int16) + counts (chunk x n_bins f32)
# bounded: 1000 x 86k x 2B ~ 173 MB worst-case giant. Normal FFI (N~600) is
# ~1.2 MB per 1000 periods. Same starting default as bls_metal_batch.py /
# bls_cuda.py -- Arc's own crossover (spec §5 point 4: "f32 crossover table
# per N... do not inherit the 20000 from CUDA") is a bench-gated follow-up, not
# assumed here.
_PERIOD_CHUNK_DEFAULT = 1000

# ----------------------------------------------------------------------
# f32/GPU-vs-f64/CPU crossover -- measured 2026-07-24 on <host>'s Arc iGPU
# (validation session, progress/validation/crossover_f32_precision.py),
# 90 fluxes / n_periods=5000 / n_bins=150 (injection.py's production BLS
# defaults), NUMBA_NUM_THREADS=12, server (exonit.service) idle:
#
#   N_pts   t_cpu_f64  t_gpu_f32  cpu/gpu     max_rel_power  max_rel_depth
#      100      0.261      0.257    1.019       (all N: <1e-6, tolerances
#      200      0.299      0.282    1.061        rel_power<1e-4/rel_depth
#      300      0.337      0.293    1.149        <1e-3/rel_snr<5e-2 from
#      400      0.348      0.325    1.072        spec Sec.5 never even
#      500      0.361      0.353    1.022        approached up to N=100k)
#      600      0.349      0.362    0.966
#      700      0.344      0.374    0.918
#     1000      0.357      0.435    0.821
#     5000      0.431      1.050    0.410
#    20000      1.170      3.412    0.343
#   100000      5.487     24.834    0.221
#
# Unlike bls_cuda.py's "_F32_MAX_POINTS" (a THROUGHPUT crossover between two
# GPU kernels, f32-CPU-fold vs f64-GPU-fold -- its own docstring's framing,
# despite the name, is about wall-clock, not accuracy), this module has only
# ONE GPU kernel (f32) and no on-GPU f64 path to route to past the
# threshold. Two findings from the table above, both real and unfavourable
# for Arc as an injection accelerator:
#   1. Numerical accuracy NEVER degrades with N in the tested range (up to
#      100k points) -- period selection stayed 5/5 -> 90/90 matched at
#      every N, and rel_power/rel_depth/rel_snr stayed ~1e-6/1e-7, three to
#      four orders of magnitude inside the spec's tolerances throughout.
#      f32 precision is not the limiting factor on this backend.
#   2. THROUGHPUT is: the GPU-f32 path (single-threaded numpy CPU fold,
#      same root cause bls_cuda.py's comment already documents) only wins
#      over the 12-thread numba CPU path for very small N (peak ~1.15x at
#      N~300), reaches parity around N~600-650, and loses increasingly past
#      that (4.5x SLOWER at N=100k) -- a MUCH earlier and shallower
#      crossover than CUDA's (20-40k, with a real speedup below it). Even
#      at its best (N~300), the GPU's edge (~1.1-1.15x) is far under the
#      spec's 1.5x deployment threshold (criterion 3) on its own -- see
#      progress/validation-opencl.md Fase B for the full e2e verdict.
# _F32_MAX_POINTS below documents this measured ceiling (mirrors CUDA's
# name/env-override contract for consistency) but, deliberately, is NOT
# wired into any CPU-fallback routing inside this module: injection.py's
# existing per-cell retry + circuit breaker (injection.py:355-399) already
# owns "when does a GPU batch_fn get abandoned for bls_fast_batch", and
# adding a second, independent size-based routing decision inside
# bls_opencl.py itself (calling bls_fast_batch directly from here) would be
# a second source of truth for the same policy -- exactly the kind of
# duplicated-decision drift CLAUDE.md #7 warns against. Wiring an
# N-based auto-route (or simply always preferring CPU injection on Arc,
# given how early/shallow this crossover is) is a deliberate design
# decision for a future spec, not a silent side effect of fixing this
# constant.
_F32_MAX_POINTS_DEFAULT = 600


def _f32_max_points():
    """Resolve the f32-path point-count ceiling: env override
    (``EXONIT_BLS_OPENCL_F32_MAX_PTS``) or ``_F32_MAX_POINTS_DEFAULT``.
    Invalid/empty values fall back to the default rather than raising --
    a config typo must not crash injection, only lose visibility into the
    (already narrow) regime where the GPU path is worth using. Mirrors
    ``bls_cuda._f32_max_points()``'s contract exactly (name, env-var
    naming convention, fallback-on-error behaviour) for consistency across
    the three GPU backends."""
    raw = os.environ.get("EXONIT_BLS_OPENCL_F32_MAX_PTS", "").strip()
    if not raw:
        return _F32_MAX_POINTS_DEFAULT
    try:
        return int(raw)
    except ValueError:
        logging.getLogger("exonit").warning(
            "EXONIT_BLS_OPENCL_F32_MAX_PTS=%r is not a valid integer -- "
            "falling back to the default (%d).",
            raw, _F32_MAX_POINTS_DEFAULT)
        return _F32_MAX_POINTS_DEFAULT


# Log-once idiom mirrors bls_cuda.py's _log_size_route_once -- but here it is
# purely INFORMATIONAL (no routing decision happens, see the block comment
# above): a giant target still runs on the GPU via this module, just slower
# than CPU would have been. injection.py calls bls_gpu_batch ~35x per star
# (one period-cell batch per radius x period grid cell) so, same as CUDA,
# logging every call for an oversized target would spam a 100k-star run's
# log -- log once per process instead.
_size_note_log_lock = threading.Lock()
_size_note_logged = False


def _log_oversized_once(n_pts, threshold):
    global _size_note_logged
    with _size_note_log_lock:
        if _size_note_logged:
            return
        _size_note_logged = True
    logging.getLogger("exonit").info(
        "bls_opencl.bls_gpu_batch: N=%d points exceeds the measured f32 "
        "GPU/CPU throughput crossover (%d, Intel Arc iGPU, 2026-07-24 "
        "validation bench) -- this and every subsequent oversized call in "
        "this process will run on the GPU anyway (no f64 GPU fallback "
        "exists on this backend), but numba-CPU (bls_fast_batch) is "
        "measured FASTER at this size. Override the threshold with env "
        "EXONIT_BLS_OPENCL_F32_MAX_PTS if this changes.", n_pts, threshold)

# ----------------------------------------------------------------------
# Historical AMD Vega learnings (pre-07-24 design, kept for context): the
# OLD per-work-item kernel (one thread per PERIOD, static bin_sums[200] +
# bin_counts[200] + cum_sums[400] + cum_counts[400] arrays -- ~4.8 KB of
# private memory per work-item) blew Vega's VGPR budget and spilled to
# scratch. Under real FPP-numba contention, that spilled kernel crossed the
# 10s amdgpu gfx-ring TDR at BATCH_SIZE=1000 (single-flux path) /
# PERIOD_BATCH=1000 (legacy batch path, 90000 work-items x ~90 injections).
# The legacy batch path's empirically-safe bound after tuning was
# PERIOD_BATCH=300 (down from 1000) -- see the pre-2026-07-24 git history of
# this file and docs/runbooks/IGPU_DEBUG_NEXT.md for the full Vega
# investigation. That whole design (per-work-item full bin arrays, no
# work-group cooperation) is why it never scaled: the fused kernel below
# instead shares one period's cumulative counts across a whole work-group of
# TG_FLUX=32 flux slots via local memory (the same fix that made the Metal
# and CUDA ports both a net win), rather than each work-item repeating the
# fold+bin+cumsum work independently. Arc's own contention/TDR bound under
# real FPP-OpenCL concurrency has NOT been soak-tested yet (spec §5: soak
# >=1h required before declaring the guard/period_chunk default safe) --
# treat _PERIOD_CHUNK_DEFAULT above as a starting point, not a validated
# production value, until that soak runs.
# ----------------------------------------------------------------------

_kernel_cache = {}

_ctx = None
_queue = None


# OpenCL C source for the fused kernel -- structurally identical to
# bls_metal_batch.py's Metal Shading Language kernel and bls_cuda.py's CUDA
# C kernel (same variable names/order deliberately, so a reviewer can diff
# the three side by side). @NB@/@TGSUMS@ are template-substituted per n_bins
# at compile time (see _get_kernel), mirroring both other backends' approach
# to sizing on-chip scratch memory at kernel-build time rather than via a
# dynamic local-memory kernel argument.
_BLS_BATCH_FUSED_F32_KERNEL_SRC = r"""
__kernel void bls_inj_batch_f32(
    __global const short* bin_idx,     // (n_per, N) int16 phase bin per (period, sample)
    __global const float* counts,      // (n_per, n_bins) exact per-period bin counts
    __global const float* fluxT,       // (N, n_flux) mean-centered flux, transposed
                                        // for coalesced reads (work-items in a
                                        // sub-group share the period and access
                                        // consecutive flux slots).
    __global const float* totals,      // (n_flux,) sum of centered flux (~0)
    __global const int* widths,        // (n_dur,) trial duration widths, in bins
    __global const int* params,        // [N, n_flux, n_bins, n_per, n_dur]
    __global float* out_power,         // (n_flux, n_per)
    __global uint* out_wj,             // (n_flux, n_per) packed (best_w<<16)|best_j
    __global float* out_depth          // (n_flux, n_per)
) {
    const int f   = get_global_id(0);   // flux slot (padded to TG_FLUX)
    const int p   = get_global_id(1);   // period index within this chunk
    const int lid = get_local_id(0);

    const int N      = params[0];
    const int n_flux = params[1];
    const int n_bins = params[2];      // runtime, <= @NB@
    const int n_per  = params[3];
    const int n_dur  = params[4];

    __local float tg_cumcounts[@NB@ + 1];
    __local float tg_sums[@TGSUMS@];

    // --- shared per-period cumulative counts (exact: integers in f32) ---
    if (lid == 0) {
        float c = 0.0f;
        tg_cumcounts[0] = 0.0f;
        __global const float* crow = counts + (long)p * (long)n_bins;
        for (int k = 0; k < n_bins; k++) {
            c += crow[k];
            tg_cumcounts[k + 1] = c;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (f >= n_flux) return;   // padding work-items: after the only barrier

    // --- private per-bin flux sums (sample order == numba order) ---
    __local float* sums = tg_sums + lid * (@NB@ + 1);
    for (int k = 0; k <= n_bins; k++) sums[k] = 0.0f;

    __global const short* brow = bin_idx + (long)p * (long)N;
    for (int i = 0; i < N; i++) {
        sums[(int)brow[i] + 1] += fluxT[(long)i * (long)n_flux + f];
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

    const long o = (long)f * (long)n_per + (long)p;
    out_power[o] = best_sr;
    out_wj[o]    = ((uint)best_w << 16) | ((uint)best_j & 0xFFFFu);
    out_depth[o] = best_dep;
}
"""


def _local_mem_bytes(n_bins):
    """Local memory (bytes) the f32 kernel needs per work-group for a given
    n_bins: tg_cumcounts (n_bins+1 floats, shared) + tg_sums (TG_FLUX*
    (n_bins+1) floats, one private (n_bins+1)-float row per flux slot in
    the work-group). Identical formula to
    ``bls_metal_batch._threadgroup_bytes`` / ``bls_cuda._shared_mem_bytes``
    -- same kernel design, same on-chip scratch layout, different backend's
    name for the same thing. Pure arithmetic, no device required -- callers
    needing the actual per-device BUDGET must combine this with
    ``_query_local_mem_size()``."""
    return TG_FLUX * (n_bins + 1) * 4 + (n_bins + 1) * 4


def _phase_fold_cpu(time_rel, periods, n_bins, period_chunk=_PERIOD_CHUNK_DEFAULT):
    """Compute per-period bin indices in numpy float64.

    Identical arithmetic to ``bls_metal.py``'s / ``bls_cuda.py``'s
    ``_phase_fold_cpu`` (and to numba's per-sample fold in ``bls_fast.py``:
    ``ph=(t/P)%1``, ``b=int(ph*n_bins)``, clipped to ``[0, n_bins)``) --
    duplicated here (rather than imported) so this module stays free of any
    import-time dependency on the Metal/CUDA backends, matching their own
    "imports cleanly without the other GPU library" independence.
    """
    N = time_rel.shape[0]
    n_periods = periods.shape[0]
    bin_idx = np.empty((n_periods, N), dtype=np.int32)
    for s in range(0, n_periods, period_chunk):
        e = min(s + period_chunk, n_periods)
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

    Same formula as ``bls_metal_batch._counts_per_period`` /
    ``bls_cuda._counts_per_period``."""
    P, N = bin_idx_chunk.shape
    flat = bin_idx_chunk.astype(np.int64) + \
        (np.arange(P, dtype=np.int64)[:, None] * n_bins)
    counts = np.bincount(flat.ravel(), minlength=P * n_bins)
    return counts.reshape(P, n_bins).astype(np.float32)


def _select_gpu_device():
    """Pick a GPU OpenCL device across every available platform.

    ``cl.create_some_context(interactive=False)`` (the old design's choice,
    and this function's own first draft -- see the 2026-07-24 review,
    Finding 1) picks whatever device PyOpenCL's own heuristic lands on
    first -- if a host also has a CPU OpenCL runtime installed (e.g. pocl)
    alongside the Intel NEO GPU platform, that heuristic is not guaranteed
    to prefer the GPU. Explicit device_type filtering removes that
    ambiguity: we always want the Arc iGPU, never a CPU OpenCL device
    silently substituting for it (the exact CLAUDE.md #2 failure mode --
    "GPU on workers is mandatory; CPU fallback is NOT acceptable" -- applied
    to backend *selection*, not just the worker's headline GPU/CPU switch).

    Enumeration failures are isolated PER PLATFORM: a broken/partial
    platform (e.g. a stray CPU-only ICD that raises on
    ``get_devices()``) must not abort the search before a LATER platform
    (the real GPU one) gets a chance -- only ``cl.get_platforms()`` itself
    failing aborts the whole search (there is nothing left to iterate).
    Each skip is logged at WARNING with the platform's name so a broken ICD
    is visible in worker logs, not swallowed.

    Returns the first GPU ``cl.Device`` found, or ``None`` if no platform
    exposes one. Callers MUST treat ``None`` as "no GPU backend available"
    (``_init_opencl`` returns ``False``) -- there is deliberately no
    ``create_some_context`` fallback here anymore (see ``_init_opencl``'s
    docstring): silently constructing a CPU-OpenCL context from this
    function's ``None`` would be the literal "silent CPU fallback"
    CLAUDE.md #2 forbids.
    """
    try:
        platforms = cl.get_platforms()
    except Exception as e:
        logging.getLogger("exonit").warning(
            "OpenCL: cl.get_platforms() failed (%s) -- treating as no "
            "OpenCL platforms available.", e)
        return None
    for platform in platforms:
        try:
            devices = platform.get_devices()
        except Exception as e:
            logging.getLogger("exonit").warning(
                "OpenCL: platform %r failed to enumerate its devices (%s) "
                "-- skipping this platform and continuing to the next one "
                "(a broken/partial ICD must not hide a working GPU platform "
                "listed after it).",
                getattr(platform, "name", platform), e)
            continue
        for device in devices:
            if device.type & cl.device_type.GPU:
                return device
    return None


def _init_opencl():
    """Initialize (or reuse) the process-wide OpenCL context + command
    queue. Building the context/queue is itself a submission-adjacent
    operation on shared driver state, so it runs under ``opencl_guard()``.

    Does NOT compile the fused kernel eagerly (that happens lazily, per
    n_bins, in ``_get_kernel`` -- mirrors ``bls_metal_batch``'s
    ``_get_kernel``/``bls_cuda``'s ``_get_f32_kernel`` caches). Callers that
    need a REAL submitted-and-read-back health check must use
    ``is_opencl_available()``, not just this function -- a context/queue
    can construct successfully even when the device can't actually execute
    a kernel (the exact gap the spec's contract calls out: "prova REAL
    (submissio minima), no nomes import").

    If ``_select_gpu_device()`` finds no GPU device on any platform, this
    returns ``False`` -- there is deliberately NO ``create_some_context()``
    fallback (2026-07-24 review, Finding 1: that fallback could silently
    hand back a CPU OpenCL device with no warning, which is the exact
    "silent CPU fallback" CLAUDE.md #2 forbids, and would make
    ``is_opencl_available()``'s smoke test pass -- and ``bls_gpu_batch``
    "work" -- silently on CPU instead of the Arc GPU). Returning ``False``
    here just means "no OpenCL GPU backend on this host"; the injection
    auto-detect chain in ``worker/injection.py`` already falls through to
    numba-CPU explicitly (with its own logged warning) when every GPU
    backend probe returns unavailable -- that is the correct, VISIBLE
    fallback path, not a silent one hidden inside this function.
    """
    global _ctx, _queue
    if not OPENCL_AVAILABLE:
        return False
    if _ctx is not None:
        return True
    try:
        with opencl_guard():
            device = _select_gpu_device()
            if device is None:
                logging.getLogger("exonit").warning(
                    "OpenCL: no GPU device available on any platform -- "
                    "OpenCL injection backend disabled (never silently "
                    "substituting a CPU OpenCL device, CLAUDE.md #2). "
                    "injection.py's auto-detect chain falls through to "
                    "numba-CPU via its own explicit, logged path.")
                return False
            _ctx = cl.Context([device])
            _queue = cl.CommandQueue(_ctx)
        return True
    except Exception as e:
        logging.getLogger("exonit").warning("OpenCL init failed: %s", e)
        return False


def _query_local_mem_size():
    """Query ``CL_DEVICE_LOCAL_MEM_SIZE`` (bytes) from the active context's
    device. Requires ``_init_opencl()`` to have already succeeded. Measured
    at runtime per spec §1 ("mesurar en execucio, no assumir 32KB/48KB") --
    unlike Metal (32 KB, every Apple GPU family) and CUDA (48 KB default,
    every compute-capability >=7.0 GPU), OpenCL devices vary and Arc's own
    NEO driver exposes this directly via the standard device-info query."""
    if _ctx is None:
        raise RuntimeError(
            "_query_local_mem_size: OpenCL context not initialized -- "
            "call _init_opencl() first")
    return int(_ctx.devices[0].local_mem_size)


def _query_max_work_group_size():
    """Query ``CL_DEVICE_MAX_WORK_GROUP_SIZE`` from the active context's
    device. Same runtime-measurement rationale as
    ``_query_local_mem_size``."""
    if _ctx is None:
        raise RuntimeError(
            "_query_max_work_group_size: OpenCL context not initialized -- "
            "call _init_opencl() first")
    return int(_ctx.devices[0].max_work_group_size)


def _check_kernel_limits(n_bins):
    """Raise ``ValueError`` before touching any buffer if the requested
    ``n_bins`` would overflow this device's measured local-memory budget, or
    if ``TG_FLUX`` exceeds its max work-group size. Requires
    ``_init_opencl()`` to have already run (raises via the query helpers'
    own guard otherwise). Device-property reads, not GPU submissions -- safe
    outside ``opencl_guard()``."""
    needed = _local_mem_bytes(n_bins)
    budget = _query_local_mem_size()
    if needed > budget:
        raise ValueError(
            f"bls_gpu_batch: n_bins={n_bins} needs {needed} B of OpenCL "
            f"local memory (TG_FLUX={TG_FLUX}), over this device's "
            f"measured CL_DEVICE_LOCAL_MEM_SIZE of {budget} B. Lower "
            f"n_bins/injection_n_bins for this call -- the production "
            f"default (150) needs {_local_mem_bytes(150)} B.")
    max_wg = _query_max_work_group_size()
    if TG_FLUX > max_wg:
        raise ValueError(
            f"bls_gpu_batch: TG_FLUX={TG_FLUX} exceeds this device's "
            f"measured CL_DEVICE_MAX_WORK_GROUP_SIZE of {max_wg}.")


def _get_kernel(n_bins_cap):
    """Build (or fetch) the fused kernel compiled for a given max n_bins.
    Must run inside ``opencl_guard()`` -- compiling a ``cl.Program`` submits
    work to the driver's compiler, the same submission-adjacent risk as
    building the context (see ``_init_opencl``'s docstring)."""
    key = int(n_bins_cap)
    k = _kernel_cache.get(key)
    if k is None:
        src = (_BLS_BATCH_FUSED_F32_KERNEL_SRC
               .replace("@NB@", str(key))
               .replace("@TGSUMS@", str(TG_FLUX * (key + 1))))
        program = cl.Program(_ctx, src).build()
        k = program.bls_inj_batch_f32
        _kernel_cache[key] = k
    return k


def is_opencl_available() -> bool:
    """True iff PyOpenCL is importable, ``EXONIT_INJECTION_OPENCL`` doesn't
    disable it, AND a minimal kernel actually submits and reads back the
    expected value on this host -- not just an import check. The pre-07-24
    design's failure mode (Vega TDR crashes despite a working import/context)
    is exactly why the spec requires a real submission here (spec §2: "prova
    REAL (submissio minima), no nomes import")."""
    if injection_opencl_disabled_by_env():
        return False
    if not OPENCL_AVAILABLE:
        return False
    try:
        with opencl_guard():
            if not _init_opencl():
                return False
            mf = cl.mem_flags
            buf = cl.Buffer(_ctx, mf.WRITE_ONLY, 4)
            prog = cl.Program(
                _ctx,
                "__kernel void _bls_opencl_smoke(__global float* o) "
                "{ o[0] = 1.0f; }").build()
            prog._bls_opencl_smoke(_queue, (1,), None, buf)
            out = np.zeros(1, dtype=np.float32)
            cl.enqueue_copy(_queue, out, buf)
            _queue.finish()
            return bool(out[0] == 1.0)
    except Exception as e:
        logging.getLogger("exonit").warning("OpenCL smoke test failed: %s", e)
        return False


def get_gpu_info():
    """Real OpenCL device name (e.g. "Intel(R) Arc(TM) Graphics"), or None
    if no GPU context is up yet. Cheap: just a string attribute on the
    already-open ``_ctx`` (built once by ``_init_opencl``) -- no new device
    enumeration, no subprocess. Callers should call this AFTER
    ``is_opencl_available()``/``_init_opencl()`` has run at least once
    (worker registration does, via bls.get_hardware_info()); before that
    ``_ctx`` is None and this returns None rather than triggering init
    itself (get_gpu_info() must stay a pure read, not a side-effecting
    probe)."""
    if _ctx is None:
        return None
    try:
        return _ctx.devices[0].name.strip()
    except Exception:
        return None


def bls_gpu_batch(time, flux_list, period_min=0.5, period_max=50.0,
                  n_periods=10000, n_bins=200, n_dur_steps=15,
                  duration_min_frac=0.01, duration_max_frac=0.15,
                  period_chunk=_PERIOD_CHUNK_DEFAULT):
    """Fused-kernel batched BLS for Intel Arc (OpenCL). Same signature/
    return contract as ``bls_fast_batch`` / ``bls_metal_batch.
    bls_gpu_batch_v2`` / ``bls_cuda._bls_gpu_batch_f32`` (spec §2)."""
    if injection_opencl_disabled_by_env():
        raise RuntimeError(
            "OpenCL injection disabled by EXONIT_INJECTION_OPENCL=0/false")
    if not OPENCL_AVAILABLE:
        raise ImportError(
            "PyOpenCL not available. Install with: pip install pyopencl")

    with opencl_guard():
        if not _init_opencl():
            raise RuntimeError("Failed to initialize OpenCL")

    # n_bins/TG_FLUX budget guard -- runtime-measured device limits, checked
    # before any buffer allocation (device-property reads, not a GPU
    # submission -- see _check_kernel_limits's docstring).
    _check_kernel_limits(n_bins)

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

    threshold = _f32_max_points()
    if N > threshold:
        _log_oversized_once(N, threshold)

    t0_ref = float(time[0])
    time_rel = time - t0_ref

    periods = _frequency_period_grid(period_min, period_max, n_periods)
    dur_bins = np.unique(np.linspace(
        max(1, int(duration_min_frac * n_bins)),
        min(n_bins // 3, int(duration_max_frac * n_bins)),
        n_dur_steps
    ).astype(np.int64))
    n_dur = len(dur_bins)

    # Mean-center in f64 before the f32 cast (depth/sr shift-invariant; kills
    # f32 accumulation error -- see bls_metal_batch.py / bls_cuda.py
    # rationale, identical here).
    flux_means = flux_matrix.mean(axis=1)
    flux_centered = flux_matrix - flux_means[:, None]
    totals = flux_centered.sum(axis=1)  # ~0 by construction

    mf = cl.mem_flags
    with opencl_guard():
        fluxT_buf = cl.Buffer(
            _ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=np.ascontiguousarray(flux_centered.T, dtype=np.float32))
        totals_buf = cl.Buffer(
            _ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=totals.astype(np.float32))
        widths_buf = cl.Buffer(
            _ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=dur_bins.astype(np.int32))
        kernel = _get_kernel(n_bins)

    n_flux_pad = ((n_flux + TG_FLUX - 1) // TG_FLUX) * TG_FLUX

    # Per-chunk opencl_guard(): each iteration's buffer upload + kernel
    # launch + read-back is fully self-contained (its own inputs, its own
    # output buffers, synchronously finished via _queue.finish() before the
    # lock is released) -- nothing OpenCL-side is left dangling across chunk
    # boundaries, so releasing the lock between chunks is safe. That lets a
    # concurrent FPP-OpenCL call (main thread) interleave between chunks
    # instead of blocking for the whole injection call, while still
    # serializing every actual GPU submission against the same process-wide
    # lock FPP uses (mirrors bls_metal_batch.bls_gpu_batch_v2's per-chunk
    # metal_guard() pattern exactly).
    power_chunks, wj_chunks, dep_chunks = [], [], []
    for s in range(0, n_periods, period_chunk):
        e = min(s + period_chunk, n_periods)
        n_per = e - s
        # CPU-only numpy work stays outside the guard.
        bin_idx = _phase_fold_cpu(time_rel, periods[s:e], n_bins,
                                  period_chunk=n_per).astype(np.int16)
        counts = _counts_per_period(bin_idx, n_bins)

        out_power = np.empty((n_flux, n_per), dtype=np.float32)
        out_wj = np.empty((n_flux, n_per), dtype=np.uint32)
        out_depth = np.empty((n_flux, n_per), dtype=np.float32)

        with opencl_guard():
            bin_idx_buf = cl.Buffer(
                _ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=np.ascontiguousarray(bin_idx))
            counts_buf = cl.Buffer(
                _ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=np.ascontiguousarray(counts))
            params_buf = cl.Buffer(
                _ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=np.array([N, n_flux, n_bins, n_per, n_dur],
                                 dtype=np.int32))
            out_power_buf = cl.Buffer(_ctx, mf.WRITE_ONLY, out_power.nbytes)
            out_wj_buf = cl.Buffer(_ctx, mf.WRITE_ONLY, out_wj.nbytes)
            out_depth_buf = cl.Buffer(_ctx, mf.WRITE_ONLY, out_depth.nbytes)

            kernel(_queue, (n_flux_pad, n_per), (TG_FLUX, 1),
                   bin_idx_buf, counts_buf, fluxT_buf, totals_buf,
                   widths_buf, params_buf, out_power_buf, out_wj_buf,
                   out_depth_buf)

            cl.enqueue_copy(_queue, out_power, out_power_buf)
            cl.enqueue_copy(_queue, out_wj, out_wj_buf)
            cl.enqueue_copy(_queue, out_depth, out_depth_buf)
            _queue.finish()

        power_chunks.append(out_power.astype(np.float64))
        wj_chunks.append(out_wj)
        dep_chunks.append(out_depth.astype(np.float64))

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


def bls_gpu(time, flux, period_min=0.5, period_max=50.0, n_periods=10000,
            n_bins=200, n_dur_steps=15, duration_min_frac=0.01,
            duration_max_frac=0.15):
    """Single-flux fused-kernel BLS -- thin wrapper over ``bls_gpu_batch``,
    same interface as ``bls_fast`` / ``bls_metal_batch.bls_gpu`` /
    ``bls_cuda.bls_gpu``."""
    results = bls_gpu_batch(
        time, [flux], period_min=period_min, period_max=period_max,
        n_periods=n_periods, n_bins=n_bins, n_dur_steps=n_dur_steps,
        duration_min_frac=duration_min_frac,
        duration_max_frac=duration_max_frac)
    return results[0]


# --- Backward-compat aliases -------------------------------------------
# worker/injection.py:122 still imports these legacy names
# (`from bls_opencl import bls_opencl, bls_opencl_batch, _init_opencl`) --
# rewiring that hook to the bls_gpu/bls_gpu_batch names above is spec
# iteration 2, deliberately out of scope here. Aliasing keeps the existing
# auto-detect chain working unmodified in the meantime.
bls_opencl = bls_gpu
bls_opencl_batch = bls_gpu_batch


def benchmark(n_points=50000, n_periods=10000):
    """Benchmark the fused OpenCL kernel vs CPU (numba) BLS on the
    injection-recovery batch path."""
    from bls_fast import bls_fast_batch

    np.random.seed(42)
    time = np.linspace(0, 60, n_points)
    flux = np.ones(n_points, dtype=np.float64) + np.random.normal(0, 0.001, n_points)
    # Inject transit
    phase = ((time - 5.0) % 10.0) / 10.0
    flux[(phase < 0.01) | (phase > 0.99)] -= 0.005

    print(f"Benchmark (fused kernel) -- {n_points} points, {n_periods} periods")
    print("=" * 50)

    # CPU
    print("\n[CPU numba]...")
    t0 = _time.time()
    result_cpu = bls_fast_batch(time, [flux], n_periods=n_periods)[0]
    cpu_time = _time.time() - t0
    print(f"  Time: {cpu_time:.2f}s")
    print(f"  Period: {result_cpu['period']:.4f}d, SNR: {result_cpu['snr']:.1f}")

    # OpenCL — warmup
    print("\n[OpenCL Arc iGPU] (warmup)...")
    try:
        bls_gpu(time[:1000], flux[:1000], n_periods=100)
    except Exception as e:
        print(f"  OpenCL failed: {e}")
        return

    # OpenCL — real
    print("[OpenCL Arc iGPU]...")
    t0 = _time.time()
    result_ocl = bls_gpu(time, flux, n_periods=n_periods)
    ocl_time = _time.time() - t0
    print(f"  Time: {ocl_time:.2f}s")
    print(f"  Period: {result_ocl['period']:.4f}d, SNR: {result_ocl['snr']:.1f}")

    speedup = cpu_time / ocl_time if ocl_time > 0 else 0
    print(f"\n{'='*50}")
    print(f"CPU: {cpu_time:.2f}s | OpenCL: {ocl_time:.2f}s | Speedup: {speedup:.2f}x")
    if speedup > 1:
        print(f"OpenCL is {speedup:.1f}x FASTER")
    else:
        print(f"CPU is {1/speedup:.1f}x faster")
    print(f"{'='*50}")

    # Verify results match
    period_diff = abs(result_cpu['period'] - result_ocl['period'])
    print(f"\nPeriod match: diff={period_diff:.4f}d ({'OK' if period_diff < 0.5 else 'MISMATCH'})")


if __name__ == "__main__":
    benchmark()
