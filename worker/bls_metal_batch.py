"""Fused Metal kernel for batched BLS injection-recovery (Apple Silicon).

Motivation (2026-07-11 analysis, spec
``docs/superpowers/specs/2026-07-11-gpu-bls-f32-unified-kernel.md``): on the
Macs the injection stage is ~85% of per-star wall-clock in FFI discovery runs
(inj≈4.5s of 5.2s on M4), the GPU sits at 1-10% (only FPP bursts), and the
existing ``bls_metal.bls_gpu_batch`` loops over fluxes in Python with one
``mx.eval`` per flux — the batch dimension never reaches the GPU. This module
puts the WHOLE period-cell batch (n_flux x n_periods) into ONE kernel launch,
following the recipe that made ``fpp_metal_kernel.py`` a 5x win: keep
per-thread state in registers/threadgroup memory, no global atomics (the
thing that killed the April P6 attempt).

Design (mirrors ``bls_fast.bls_fast_batch`` semantics exactly):
- Phase-fold on CPU in float64 (exact, same as numba: ph=(t/P)%1, b=int(ph*nb),
  clip) -> int16 bin indices, shared by all fluxes of the cell.
- Per-period bin COUNTS precomputed on CPU (int, exact), cumsum'd in-kernel.
- One GPU thread per (flux, period): accumulates its private per-bin flux
  sums in threadgroup memory (f32, sample order identical to numba),
  prefix-sums them, then loops (width asc, phase-bin asc) with strict '>'
  updates — byte-for-byte the numba tie-breaking order (sequential inside
  the thread, so no parallel reduction to reason about).
- Flux is mean-centered in f64 before the f32 cast (depth = mean_out - mean_in
  is shift-invariant; centering crushes f32 accumulation error below the ppm
  floor — same proven trick as bls_metal.py).
- f32 window sums come from prefix-sum differences (not incremental slide), so
  rounding is bounded and deterministic.

Threadgroup memory budget: TG_FLUX*(n_bins+1)*4 + (n_bins+1)*4 bytes.
TG_FLUX=32, n_bins=150 -> ~19.9 KB (< 32 KB). n_bins=200 -> ~26 KB, still OK.

Drop-in: ``bls_gpu_batch_v2`` has the same signature/return as
``bls_fast.bls_fast_batch`` / ``bls_metal.bls_gpu_batch``, so it can be passed
as ``batch_fn`` to ``injection._run_batched_gpu`` unchanged.

Per-SoC gate (measurement, not architecture defaults): the full bench
(2026-07-11, idle fleet) covered all three Apple Silicon SoCs in the fleet —
M1 (worker-<host>), M3 (worker-<host>), M4 (worker-<host>*) — with
injection e2e 3.2x/4.7x/6.6x vs numba-CPU, completeness map max|Delta|=0.0000
(identical) on all three, and FPP(Metal) || injection(Metal) contention 4/4
clean rounds with parity intact on all three. Result: enable on every Apple
Silicon chip (``_SUPPORTED_SOC_PREFIXES`` matches any "Apple M*" brand
string), overturning the earlier "M1 stays on CPU injection" rule.

That old rule does NOT re-apply here even though it was also about M1 +
Metal + injection: it was measured against ``bls_metal.py``'s per-flux
Python-loop design (one ``mx.eval`` sync per flux -> injection measured 2.74x
SLOWER than CPU on M1, 2026-05-27) and the 2026-04 M1 hang was the old
per-flux design contending with FPP on the command buffer. The fused
single-launch kernel here is a different design: M1 injection is 3.2x
FASTER than CPU and the contention hang does not reproduce (4/4 clean). It
also never enters numba ``parallel=True`` on the injection side at all,
which removes the workqueue-crash class (see ``numba_serial.py``) from
injection entirely on every SoC, not just M1. Per-SoC decisions must still
come from measurement, never assumed from architecture — if a future SoC
(M5+) benches differently, gate it explicitly rather than extending the
prefix match on faith.

Kill-switch: env ``EXONIT_INJECTION_METAL`` = ``0``/``false`` disables this
backend unconditionally (escape hatch if a specific host misbehaves under
multi-instance load, without a code change); ``1``/``true`` force-enables it
bypassing the SoC allowlist (used to benchmark an unmeasured chip before
deciding whether to add it to the allowlist).
"""

import os

import numpy as np

from metal_serial import metal_guard, injection_metal_disabled_by_env

try:
    import mlx.core as mx
    MLX_AVAILABLE = hasattr(mx, "fast") and hasattr(mx.fast, "metal_kernel")
except ImportError:
    MLX_AVAILABLE = False

# Canonical grid + SDE from the CPU reference (single source of truth).
from bls_fast import _compute_sde, _frequency_period_grid
from bls_metal import _phase_fold_cpu

TG_FLUX = 32

# Period chunk keeps bin_idx (chunk x N int16) + counts (chunk x n_bins f32)
# bounded: 1000 x 86k x 2B ~ 173 MB worst-case giant. Normal FFI (N~600) is
# ~1.2 MB per 1000 periods.
_PERIOD_CHUNK_DEFAULT = 1000

# Apple Metal's per-threadgroup memory limit (all GPU families since the
# A11/M1 generation). The kernel allocates tg_cumcounts[n_bins+1] +
# tg_sums[TG_FLUX*(n_bins+1)] floats (4 bytes each) -- see _threadgroup_bytes
# below and the module docstring's "Threadgroup memory budget" section.
# Production default n_bins=150 -> ~19.9 KB, comfortably under this; a host
# config raising injection_n_bins unboundedly would otherwise either fail
# the kernel compile/launch obscurely or (past a byte-exact edge the driver
# happens to tolerate) corrupt threadgroup memory -- guarded explicitly at
# bls_gpu_batch_v2 entry instead.
_METAL_THREADGROUP_LIMIT_BYTES = 32 * 1024


def _threadgroup_bytes(n_bins):
    """Threadgroup memory the kernel needs for a given n_bins: tg_cumcounts
    (n_bins+1 floats, shared) + tg_sums (TG_FLUX*(n_bins+1) floats, one
    private (n_bins+1)-float row per flux slot in the group)."""
    return TG_FLUX * (n_bins + 1) * 4 + (n_bins + 1) * 4

# Measurement-gated allowlist (see module docstring for the 2026-07-11 bench
# that covers M1/M3/M4): any Apple M-series SoC. Extend only after a passing
# bench for the new chip, never by assumption.
_SUPPORTED_SOC_PREFIXES = ("Apple M",)

_KERNEL_SRC = r"""
    uint f   = thread_position_in_grid.x;   // flux slot (padded to TG_FLUX)
    uint p   = thread_position_in_grid.y;   // period index within this chunk
    uint lid = thread_position_in_threadgroup.x;

    const int N      = (int)params[0];
    const int n_flux = (int)params[1];
    const int n_bins = (int)params[2];      // runtime, <= @NB@
    const int n_per  = (int)params[3];
    const int n_dur  = (int)params[4];

    threadgroup float tg_cumcounts[@NB@ + 1];
    threadgroup float tg_sums[@TGSUMS@];

    // --- shared per-period cumulative counts (exact: integers in f32) ---
    if (lid == 0) {
        float c = 0.0f;
        tg_cumcounts[0] = 0.0f;
        const device float* crow = counts + (ulong)p * (ulong)n_bins;
        for (int k = 0; k < n_bins; k++) {
            c += crow[k];
            tg_cumcounts[k + 1] = c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (f >= (uint)n_flux) return;   // padding threads: after the only barrier

    // --- private per-bin flux sums (sample order == numba order) ---
    threadgroup float* sums = tg_sums + lid * (@NB@ + 1);
    for (int k = 0; k <= n_bins; k++) sums[k] = 0.0f;

    const device short* brow = bin_idx + (ulong)p * (ulong)N;
    for (int i = 0; i < N; i++) {
        sums[(int)brow[i] + 1] += fluxT[(ulong)i * (ulong)n_flux + f];
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

    const ulong o = (ulong)f * (ulong)n_per + (ulong)p;
    out_power[o] = best_sr;
    out_wj[o]    = ((uint)best_w << 16) | ((uint)best_j & 0xFFFFu);
    out_depth[o] = best_dep;
"""

_kernel_cache = {}


def _get_kernel(n_bins_cap):
    """Build (or fetch) the kernel compiled for a given max n_bins."""
    key = int(n_bins_cap)
    k = _kernel_cache.get(key)
    if k is None:
        src = (_KERNEL_SRC
               .replace("@NB@", str(key))
               .replace("@TGSUMS@", str(TG_FLUX * (key + 1))))
        k = mx.fast.metal_kernel(
            name=f"bls_inj_batch_nb{key}",
            input_names=["bin_idx", "counts", "fluxT", "totals", "widths",
                         "params"],
            output_names=["out_power", "out_wj", "out_depth"],
            source=src,
        )
        _kernel_cache[key] = k
    return k


def is_supported_soc():
    """True iff this host's Apple Silicon SoC has been bench-validated for
    the fused kernel (see module docstring for the 2026-07-11 M1/M3/M4
    bench). Falls back to False on any probe failure — an unrecognised
    chip string is not assumed safe."""
    try:
        import subprocess
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                           capture_output=True, text=True, timeout=5)
        chip = r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return False
    return any(chip.startswith(p) for p in _SUPPORTED_SOC_PREFIXES)


def is_available():
    """True iff the fused kernel can run on this host.

    Requires MLX + the ``mx.fast.metal_kernel`` API present and a working
    GPU, AND (the SoC is bench-validated OR the kill-switch force-enables
    it). Env ``EXONIT_INJECTION_METAL``: ``0``/``false`` hard-disables
    (per-host escape hatch, no code change needed); ``1``/``true`` bypasses
    the SoC allowlist (used to bench an unmeasured chip before deciding
    whether to add it to ``_SUPPORTED_SOC_PREFIXES``).
    """
    # Shared with bls_metal.is_metal_available() -- see
    # metal_serial.injection_metal_disabled_by_env's docstring for why the
    # helper lives there instead of here or in bls_metal.py.
    if injection_metal_disabled_by_env():
        return False
    if not MLX_AVAILABLE:
        return False
    try:
        # metal_guard(): shared with fpp_metal_kernel/bls_metal/fpp_metal --
        # this smoke test submits real Metal work (mx.eval), so it must not
        # race a concurrent FPP-Metal submission on another thread.
        with metal_guard():
            x = mx.array([1.0, 2.0], dtype=mx.float32)
            mx.eval(mx.sum(x))
    except Exception:
        return False
    override = os.environ.get("EXONIT_INJECTION_METAL", "").strip().lower()
    if override in ("1", "true"):
        return True
    return is_supported_soc()


def _counts_per_period(bin_idx_chunk, n_bins):
    """Exact per-period bin counts from int16 bin indices, vectorized."""
    P, N = bin_idx_chunk.shape
    flat = bin_idx_chunk.astype(np.int64) + \
        (np.arange(P, dtype=np.int64)[:, None] * n_bins)
    counts = np.bincount(flat.ravel(), minlength=P * n_bins)
    return counts.reshape(P, n_bins).astype(np.float32)


def bls_gpu_batch_v2(time, flux_list, period_min=0.5, period_max=50.0,
                     n_periods=10000, n_bins=200, n_dur_steps=15,
                     duration_min_frac=0.01, duration_max_frac=0.15,
                     period_chunk=_PERIOD_CHUNK_DEFAULT):
    """Fused-kernel batched BLS. Same signature/return as bls_fast_batch."""
    # n_bins validation comes first (pure arithmetic, independent of MLX
    # being installed) so a misconfigured injection_n_bins fails the same
    # way on every host, and this guard is testable without real hardware.
    _budget = _threadgroup_bytes(n_bins)
    if _budget > _METAL_THREADGROUP_LIMIT_BYTES:
        raise ValueError(
            f"bls_gpu_batch_v2: n_bins={n_bins} needs {_budget} B of Metal "
            f"threadgroup memory (TG_FLUX={TG_FLUX}), over the "
            f"{_METAL_THREADGROUP_LIMIT_BYTES} B per-threadgroup limit. "
            f"Lower n_bins/injection_n_bins for this call -- the production "
            f"default (150) needs {_threadgroup_bytes(150)} B, well under "
            f"budget.")

    if not MLX_AVAILABLE:
        raise ImportError("MLX not available. Install with: pip install mlx")

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

    t0_ref = float(time[0])
    time_rel = time - t0_ref

    periods = _frequency_period_grid(period_min, period_max, n_periods)
    dur_bins = np.unique(np.linspace(
        max(1, int(duration_min_frac * n_bins)),
        min(n_bins // 3, int(duration_max_frac * n_bins)),
        n_dur_steps
    ).astype(np.int64))
    n_dur = len(dur_bins)

    # Mean-center in f64 (depth/sr shift-invariant; kills f32 accumulation
    # error — see bls_metal.py rationale).
    flux_means = flux_matrix.mean(axis=1)
    flux_centered = flux_matrix - flux_means[:, None]
    totals = flux_centered.sum(axis=1)  # ~0 by construction

    # metal_guard(): these three mx.array() uploads run once per call (not
    # per chunk) but still touch the shared MLX device/command-queue state
    # -- reviewer-requested closure of a gap where an "uncited assumption
    # about MLX internals" (that plain mx.array() construction never
    # submits) was the only thing standing between this and the exact bug
    # class that killed the fleet. Costs nothing measurable (once per call,
    # before the per-chunk loop the narrow guard below exists to keep
    # interleave-able) so there is no reason to rely on that assumption
    # instead of just guarding it.
    with metal_guard():
        fluxT_mx = mx.array(
            np.ascontiguousarray(flux_centered.T, dtype=np.float32))
        totals_mx = mx.array(totals.astype(np.float32))
        widths_mx = mx.array(dur_bins.astype(np.int32))
        kernel = _get_kernel(n_bins)
    n_flux_pad = ((n_flux + TG_FLUX - 1) // TG_FLUX) * TG_FLUX

    # Per-chunk metal_guard(): each iteration's mx.array()+kernel-launch+
    # mx.eval() is fully self-contained (its own inputs, its own graph node,
    # synchronously evaluated before the lock is released) -- nothing MLX-
    # side is left dangling across chunk boundaries, so releasing the lock
    # between chunks is safe. That lets a concurrent FPP-Metal call (main
    # thread) interleave between chunks instead of blocking for the whole
    # injection call, while still serializing every actual GPU submission
    # against the same process-wide lock FPP uses (fpp_metal_kernel).
    power_chunks, wj_chunks, dep_chunks = [], [], []
    for s in range(0, n_periods, period_chunk):
        e = min(s + period_chunk, n_periods)
        n_per = e - s
        # CPU-only numpy work stays outside the guard.
        bin_idx = _phase_fold_cpu(time_rel, periods[s:e], n_bins,
                                  period_chunk=n_per).astype(np.int16)
        counts = _counts_per_period(bin_idx, n_bins)

        with metal_guard():
            params = mx.array(np.array(
                [N, n_flux, n_bins, n_per, n_dur], dtype=np.float32))
            outs = kernel(
                inputs=[mx.array(bin_idx), mx.array(counts), fluxT_mx,
                        totals_mx, widths_mx, params],
                grid=(n_flux_pad, n_per, 1),
                threadgroup=(TG_FLUX, 1, 1),
                output_shapes=[(n_flux, n_per)] * 3,
                output_dtypes=[mx.float32, mx.uint32, mx.float32])
            mx.eval(*outs)
        # These np.array() readouts run outside metal_guard(), but that's
        # safe: mx.eval(*outs) above already forced full materialization
        # of every buffer *inside* the guard -- the readout below only
        # copies already-realized memory, it does not submit any new
        # command-buffer work to MLX (2026-07-12 re-review, "outside the
        # guard" note).
        power_chunks.append(np.array(outs[0], dtype=np.float64))
        wj_chunks.append(np.array(outs[1], dtype=np.uint32))
        dep_chunks.append(np.array(outs[2], dtype=np.float64))

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
    """Single-flux fused-kernel BLS — thin wrapper over ``bls_gpu_batch_v2``,
    same interface as ``bls_fast`` / ``bls_metal.bls_gpu`` / ``bls_cuda.bls_gpu``."""
    results = bls_gpu_batch_v2(
        time, [flux], period_min=period_min, period_max=period_max,
        n_periods=n_periods, n_bins=n_bins, n_dur_steps=n_dur_steps,
        duration_min_frac=duration_min_frac,
        duration_max_frac=duration_max_frac)
    return results[0]


def kernel_reference_numpy(time, flux_list, period_min=0.5, period_max=50.0,
                           n_periods=10000, n_bins=200, n_dur_steps=15,
                           duration_min_frac=0.01, duration_max_frac=0.15):
    """Pure-numpy simulator of the Metal kernel's exact f32 arithmetic.

    Runs anywhere (no MLX) — used to validate window/wrap/tie-break logic and
    f32 rounding behaviour against bls_fast_batch before touching a Mac.
    Slow: for small validation workloads only.
    """
    flux_matrix = np.ascontiguousarray(np.stack(
        [np.asarray(f, dtype=np.float64) for f in flux_list]))
    n_flux, N = flux_matrix.shape
    time = np.ascontiguousarray(time, dtype=np.float64)
    t0_ref = float(time[0])
    time_rel = time - t0_ref

    periods = _frequency_period_grid(period_min, period_max, n_periods)
    dur_bins = np.unique(np.linspace(
        max(1, int(duration_min_frac * n_bins)),
        min(n_bins // 3, int(duration_max_frac * n_bins)),
        n_dur_steps
    ).astype(np.int64))

    flux_means = flux_matrix.mean(axis=1)
    flux_centered = (flux_matrix - flux_means[:, None])
    totals = flux_centered.sum(axis=1).astype(np.float32)
    fluxT32 = flux_centered.astype(np.float32)

    bin_idx = _phase_fold_cpu(time_rel, periods, n_bins,
                              period_chunk=1000).astype(np.int16)

    f32 = np.float32
    best_powers = np.full((n_flux, n_periods), -1e30, dtype=np.float64)
    best_w = np.zeros((n_flux, n_periods))
    best_j = np.zeros((n_flux, n_periods))
    best_depths = np.zeros((n_flux, n_periods))

    for pi in range(n_periods):
        b = bin_idx[pi]
        cnt = np.bincount(b, minlength=n_bins).astype(f32)
        cumc = np.concatenate([[f32(0.0)], np.cumsum(cnt, dtype=f32)])
        for fi in range(n_flux):
            sums = np.zeros(n_bins + 1, dtype=f32)
            np.add.at(sums, b + 1, fluxT32[fi])
            sums = np.cumsum(sums, dtype=f32)
            total = totals[fi]
            Nf = f32(N)
            bs, bw, bj, bd = f32(-1e30), 0, 0, f32(0.0)
            for w in dur_bins:
                for j in range(n_bins):
                    e = j + w
                    if e <= n_bins:
                        wc = f32(cumc[e] - cumc[j])
                        ws = f32(sums[e] - sums[j])
                    else:
                        e2 = e - n_bins
                        wc = f32(f32(cumc[n_bins] - cumc[j]) + cumc[e2])
                        ws = f32(f32(sums[n_bins] - sums[j]) + sums[e2])
                    oc = f32(Nf - wc)
                    if wc < 3.0 or oc < 3.0:
                        continue
                    mean_in = f32(ws / wc)
                    mean_out = f32(f32(total - ws) / oc)
                    depth = f32(mean_out - mean_in)
                    if depth > 0:
                        sr = f32(f32(f32(f32(depth * depth) * wc) * oc)
                                 / f32(wc + oc))
                        if sr > bs:
                            bs, bw, bj, bd = sr, w, j, depth
            best_powers[fi, pi] = bs
            best_w[fi, pi] = bw
            best_j[fi, pi] = bj
            best_depths[fi, pi] = bd

    results = []
    for fi in range(n_flux):
        spectrum = best_powers[fi]
        best_idx = int(np.argmax(spectrum))
        best_period = float(periods[best_idx])
        results.append({
            "period": best_period,
            "power": float(spectrum[best_idx]),
            "duration": float(best_w[fi, best_idx] / n_bins * best_period),
            "t0": float(best_j[fi, best_idx] / n_bins * best_period + t0_ref),
            "depth": float(best_depths[fi, best_idx]),
            "depth_ppm": float(best_depths[fi, best_idx] * 1e6),
            "snr": _compute_sde(spectrum, best_idx),
            "periods": periods,
            "power_spectrum": spectrum,
        })
    return results
