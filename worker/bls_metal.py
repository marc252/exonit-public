"""Apple Silicon (Metal) accelerated BLS using MLX.

This module mirrors ``bls_fast.py`` exactly — same period grid, same SDE,
same dict shape, same tie-breaking — but offloads the per-flux scatter,
sliding-window subtract, and argmax to the integrated GPU via MLX. The
design choice is **hybrid CPU/GPU**:

- The phase-fold (``time / period`` modulo 1) is computed once on the
  CPU in numpy float64. MLX does not support float64 on the GPU, and
  float32 lacks the precision needed for ~800 d TESS baselines: empirically
  ~0.04 % of samples per period fall in the wrong bin under pure-float32
  phase folding, which is enough to flip the chosen peak between adjacent
  harmonics on weak injections. That is unacceptable for a backend that
  must reproduce the TOI catalogue.
- Everything that scales with ``n_flux × n_periods × n_bins × n_widths``
  (scatter flux into bins, cumulative sums, sliding-window subtract,
  argmax across phase positions) stays on the GPU.

The CPU phase-fold is amortised over all fluxes in a batch — it is
computed once per period grid and reused. For the 720-evaluation
injection-recovery workload, that fixed cost is small relative to the
per-flux GPU work, so the hybrid design still wins on throughput while
matching numba bit-for-bit on weak signals.

Install: pip install mlx
"""

import numpy as np

# Reuse the canonical period grid and SDE from bls_fast — single source of
# truth, so any future refinement automatically applies to both backends.
from bls_fast import _compute_sde, _frequency_period_grid
from metal_serial import metal_guard, injection_metal_disabled_by_env

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


# Maximum number of (period, sample) entries to materialise in a single
# numpy buffer during the CPU phase-fold. 1000 periods × 86 817 samples
# × 8 bytes ≈ 690 MB, comfortable on a 16 GB box. Tune for tighter RAM.
_PERIOD_CHUNK_DEFAULT = 1000


def _phase_fold_cpu(time_rel, periods, n_bins, period_chunk=_PERIOD_CHUNK_DEFAULT):
    """Compute per-period bin indices in numpy float64.

    Parameters
    ----------
    time_rel : (N,) float64
        Time relative to the first sample (``time - time[0]``).
    periods : (n_periods,) float64
        Trial periods in the same units as ``time_rel``.
    n_bins : int
        Number of phase bins.
    period_chunk : int
        Vectorise over this many periods at a time to bound peak RAM.

    Returns
    -------
    bin_idx : (n_periods, N) int32
        Bin in ``[0, n_bins)`` that each (period, sample) maps to.
    """
    N = time_rel.shape[0]
    n_periods = periods.shape[0]
    bin_idx = np.empty((n_periods, N), dtype=np.int32)
    for s in range(0, n_periods, period_chunk):
        e = min(s + period_chunk, n_periods)
        # phase ∈ [0, 1) per (period, sample) — float64 throughout.
        phase = time_rel[None, :] / periods[s:e, None]
        phase -= np.floor(phase)
        bins = (phase * n_bins).astype(np.int32)
        # Floating-point edge case: phase*n_bins may round to exactly
        # n_bins for samples whose phase is one ULP below 1.0.
        np.clip(bins, 0, n_bins - 1, out=bins)
        bin_idx[s:e] = bins
    return bin_idx


def _bin_counts_metal(bin_idx_mx, n_periods, n_bins, N):
    """Compute per-period bin counts on the GPU via scatter add.

    Returns ``cum_counts`` of shape (n_periods, 2*n_bins+1) float32:
    cumulative sum of [bin_counts, bin_counts] prefixed with 0, ready
    for the sliding-window difference ``cum[j+w] - cum[j]``.
    """
    # Flatten (period, sample) → unique scatter target by offsetting each
    # period's bin range by `period_idx * n_bins`.
    offsets = mx.arange(n_periods, dtype=mx.int32)[:, None] * n_bins
    flat_idx = (bin_idx_mx + offsets).reshape(-1)
    ones = mx.ones(n_periods * N, dtype=mx.float32)
    flat_counts = mx.zeros(n_periods * n_bins, dtype=mx.float32)
    flat_counts = flat_counts.at[flat_idx].add(ones)
    bin_counts = flat_counts.reshape(n_periods, n_bins)
    ext_counts = mx.concatenate([bin_counts, bin_counts], axis=1)
    zeros_col = mx.zeros((n_periods, 1), dtype=mx.float32)
    cum_counts = mx.concatenate([zeros_col, mx.cumsum(ext_counts, axis=1)],
                                axis=1)
    return cum_counts, flat_idx, zeros_col


def _bls_metal_per_flux(flux_mx, flat_idx, cum_counts, zeros_col,
                        n_periods, n_bins, N, total_sum_mx, dur_bins, N_mx):
    """Run the BLS inner loop for one flux on the GPU.

    Returns four numpy arrays of shape (n_periods,):
      - best_powers (float32) — best SR found across (period, width, j)
      - best_durations_bins (int32) — winning width in bins
      - best_phase_bins (int32) — winning j (phase bin index)
      - best_depths (float32) — depth at the winning (period, width, j)
    """
    # Scatter flux → bin_sums per period using the same flat_idx as bin_counts
    # so per-sample alignment is bit-identical to the count map.
    flat_flux = mx.broadcast_to(flux_mx[None, :], (n_periods, N)).reshape(-1)
    flat_sums = mx.zeros(n_periods * n_bins, dtype=mx.float32)
    flat_sums = flat_sums.at[flat_idx].add(flat_flux)
    bin_sums = flat_sums.reshape(n_periods, n_bins)
    ext_sums = mx.concatenate([bin_sums, bin_sums], axis=1)
    cum_sums = mx.concatenate([zeros_col, mx.cumsum(ext_sums, axis=1)], axis=1)

    # Tracking arrays (one slot per period) — match numba's
    # initial state (best_sr=-1e30, best_d=0, best_p=0, best_dep=0.0).
    best_sr = mx.full((n_periods,), -1e30, dtype=mx.float32)
    best_d = mx.zeros((n_periods,), dtype=mx.int32)
    best_p = mx.zeros((n_periods,), dtype=mx.int32)
    best_dep = mx.zeros((n_periods,), dtype=mx.float32)

    one_f = mx.array(1.0, dtype=mx.float32)
    neg_inf = mx.array(-1e30, dtype=mx.float32)
    zero_f = mx.array(0.0, dtype=mx.float32)

    for width_int in dur_bins:
        w = int(width_int)
        # Sliding window over n_bins phase positions, with wrap via the
        # double-extended cum buffers (length 2*n_bins+1).
        win_counts = cum_counts[:, w:n_bins + w] - cum_counts[:, :n_bins]
        win_sums = cum_sums[:, w:n_bins + w] - cum_sums[:, :n_bins]
        out_counts = N_mx - win_counts
        valid = (win_counts >= 3.0) & (out_counts >= 3.0)
        mean_in = mx.where(valid, win_sums / mx.maximum(win_counts, one_f),
                           zero_f)
        mean_out = mx.where(valid,
                            (total_sum_mx - win_sums)
                            / mx.maximum(out_counts, one_f),
                            zero_f)
        depth = mean_out - mean_in
        sr = mx.where(valid & (depth > 0),
                      depth * depth * win_counts * out_counts
                      / (win_counts + out_counts),
                      neg_inf)

        # Best (sr, j, depth) for this width — argmax over phase positions.
        max_idx = mx.argmax(sr, axis=1).astype(mx.int32)  # (n_periods,)
        max_sr = mx.take_along_axis(sr, max_idx[:, None].astype(mx.int32),
                                    axis=1).squeeze(axis=1)
        max_dep = mx.take_along_axis(depth, max_idx[:, None].astype(mx.int32),
                                     axis=1).squeeze(axis=1)

        # Strict ">" to match numba's tie-breaking: first width to reach a
        # given SR keeps it (later widths only displace strictly higher SRs).
        improved = max_sr > best_sr
        best_sr = mx.where(improved, max_sr, best_sr)
        best_d = mx.where(improved, mx.array(w, dtype=mx.int32), best_d)
        best_p = mx.where(improved, max_idx, best_p)
        best_dep = mx.where(improved, max_dep, best_dep)

    mx.eval(best_sr, best_d, best_p, best_dep)
    return (np.array(best_sr), np.array(best_d),
            np.array(best_p), np.array(best_dep))


def bls_gpu_batch(time, flux_list, period_min=0.5, period_max=50.0,
                  n_periods=10000, n_bins=200, n_dur_steps=15,
                  duration_min_frac=0.01, duration_max_frac=0.15,
                  period_chunk=_PERIOD_CHUNK_DEFAULT):
    """GPU-batched BLS, output identical in shape and content to ``bls_fast_batch``.

    Parameters mirror ``bls_fast_batch``. ``period_chunk`` controls peak
    CPU RAM during the float64 phase-fold (default 1000 periods).

    Returns
    -------
    list of dict
        One per input flux, each with the same keys as ``bls_fast``'s
        return: period, power, duration, t0, depth, depth_ppm, snr,
        periods, power_spectrum.
    """
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

    # ----- CPU phase-fold (float64) — once, shared by all fluxes -----
    bin_idx = _phase_fold_cpu(time_rel, periods, n_bins,
                              period_chunk=period_chunk)

    # ----- precompute count side on GPU -----
    # metal_guard(): shared process-wide lock (see metal_serial.py) --
    # this module is a reachable fallback (injection.py falls back to it
    # when bls_metal_batch is unavailable) and must not submit Metal work
    # concurrently with a main-thread FPP-Metal call. N_mx's mx.array()
    # upload is folded in here too (once per call, not per flux) rather
    # than left bare after the block -- same reviewer-requested closure as
    # bls_metal_batch.py's fluxT_mx/totals_mx/widths_mx: don't rely on an
    # uncited assumption about plain mx.array() never submitting when
    # guarding it costs nothing measurable.
    with metal_guard():
        bin_idx_mx = mx.array(bin_idx)  # int32, shape (n_periods, N)
        cum_counts, flat_idx, zeros_col = _bin_counts_metal(
            bin_idx_mx, n_periods, n_bins, N)
        mx.eval(cum_counts, flat_idx)
        N_mx = mx.array(float(N), dtype=mx.float32)

    # ----- per-flux GPU inner loop -----
    results = []
    for fi in range(n_flux):
        flux_f = flux_matrix[fi]
        # Centre flux on its float64 mean before pushing to GPU. ``depth`` =
        # ``mean_out - mean_in`` is invariant under a constant flux shift,
        # so this is mathematically a no-op — but in float32 it shrinks
        # every bin sum by a factor of ~mean/std (≈1e4 for normalised TESS
        # light curves) and crushes the per-sample accumulation error
        # below the 1 ppm noise floor. Without this step the float32 sum
        # of ~600 samples × ~1.0 each loses ~6 decimal digits of precision.
        flux_mean = float(flux_f.mean())
        flux_centered = flux_f - flux_mean
        total_sum_centered = float(flux_centered.sum())  # ~ 0 by construction

        # Per-flux metal_guard(): flux_mx/total_mx's mx.array() uploads are
        # folded into the same guarded region as the actual GPU work below
        # (same reviewer-requested closure as above -- don't leave bare
        # mx.array() calls outside the guard on this path). _bls_metal_per_flux
        # builds its op graph (loop over durations) and evaluates it
        # (mx.eval at the end) as one self-contained unit -- nothing dangles
        # across flux boundaries, so releasing the lock between fluxes (like
        # bls_metal_batch releases it between period chunks) lets FPP
        # interleave instead of blocking for the whole batch.
        with metal_guard():
            flux_mx = mx.array(flux_centered.astype(np.float32))
            total_mx = mx.array(total_sum_centered, dtype=mx.float32)
            bp, bd_bins, bph_bins, bdep = _bls_metal_per_flux(
                flux_mx, flat_idx, cum_counts, zeros_col,
                n_periods, n_bins, N, total_mx, dur_bins, N_mx)

        # Convert tracking arrays to physical units (match bls_fast_batch).
        best_durations = bd_bins.astype(np.float64) / n_bins * periods
        best_phases = bph_bins.astype(np.float64) / n_bins
        best_depths = bdep.astype(np.float64)
        best_powers = bp.astype(np.float64)

        best_idx = int(np.argmax(best_powers))
        best_period = float(periods[best_idx])
        snr = _compute_sde(best_powers, best_idx)

        results.append({
            "period": best_period,
            "power": float(best_powers[best_idx]),
            "duration": float(best_durations[best_idx]),
            "t0": float(best_phases[best_idx] * best_period + t0_ref),
            "depth": float(best_depths[best_idx]),
            "depth_ppm": float(best_depths[best_idx] * 1e6),
            "snr": snr,
            "periods": periods,
            "power_spectrum": best_powers,
        })

    return results


def bls_gpu(time, flux, period_min=0.5, period_max=50.0, n_periods=10000,
            n_bins=200, n_dur_steps=15, duration_min_frac=0.01,
            duration_max_frac=0.15):
    """Single-flux GPU BLS — thin wrapper over ``bls_gpu_batch``.

    Same interface as ``bls_fast`` and ``bls_cuda.bls_gpu``.
    """
    results = bls_gpu_batch(
        time, [flux], period_min=period_min, period_max=period_max,
        n_periods=n_periods, n_bins=n_bins, n_dur_steps=n_dur_steps,
        duration_min_frac=duration_min_frac,
        duration_max_frac=duration_max_frac)
    return results[0]


def is_metal_available():
    """Check if MLX/Metal is available (Apple Silicon).

    Honors the ``EXONIT_INJECTION_METAL`` kill-switch exactly like
    ``bls_metal_batch.is_available()`` (shared check, see
    ``metal_serial.injection_metal_disabled_by_env``) — this legacy
    per-flux path used to ignore the switch entirely (2026-07-12
    re-review, Finding 1), so setting it to ``0`` did not actually stop
    injection from silently falling back onto this backend (documented
    2.74x SLOWER than CPU on M1) whenever the fused-kernel path in
    ``injection.py``'s auto-detect failed for any other reason.
    """
    if injection_metal_disabled_by_env():
        return False
    if not MLX_AVAILABLE:
        return False
    try:
        with metal_guard():
            x = mx.array([1.0, 2.0, 3.0])
            r = mx.sum(x)
            mx.eval(r)
        return True
    except Exception:
        return False


def get_gpu_info():
    """Get Apple Silicon GPU info string."""
    if not is_metal_available():
        return None
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return f"{result.stdout.strip()} (Metal/MLX)"
    except Exception:
        pass
    return "Apple Silicon (Metal/MLX)"
