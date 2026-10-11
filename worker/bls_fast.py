"""
ExoNit — Fast BLS (Box Least Squares) with numba acceleration.

Inner loop (phase-fold + bin + sliding window) runs in compiled numba code
with parallel execution across periods. Falls back to numpy if numba unavailable.

Performance: ~10-50x faster than pure numpy version for large arrays.
"""

import numpy as np

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from numba_serial import numba_parallel_guard


def _frequency_period_grid(period_min, period_max, n_periods):
    # Uniform spacing in 1/P matches BLS's natural resolution
    # (ΔP ≈ P²/T from Kovács 2002 / Ofir 2014 / TLS docs). Linear-in-P grids
    # over-sample long periods and under-sample short periods.
    freq = np.linspace(1.0 / period_min, 1.0 / period_max, n_periods)
    return 1.0 / freq


def _compute_sde(power_spectrum, peak_idx):
    # Canonical SDE (Kovács 2002 §5): a tall peak inflates std(spectrum)
    # itself, which suppresses secondary peaks in multi-planet systems.
    # Median + MAD·1.4826 is robust — the peak barely moves them.
    p = np.asarray(power_spectrum, dtype=np.float64)
    med = float(np.nanmedian(p))
    mad = float(np.nanmedian(np.abs(p - med)))
    sigma = 1.4826 * mad
    if sigma <= 0:
        return 0.0
    return float((p[peak_idx] - med) / sigma)

if HAS_NUMBA:
    @numba.njit(cache=True)
    def _bls_single_period(time_rel, flux, N, period, n_bins, dur_bins,
                           n_dur, total_sum):
        """BLS for a single trial period — numba compiled."""
        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins, dtype=numba.int64)

        # Phase fold + bin
        for i in range(N):
            ph = (time_rel[i] / period) % 1.0
            b = int(ph * n_bins)
            if b >= n_bins:
                b = n_bins - 1
            bin_sums[b] += flux[i]
            bin_counts[b] += 1

        # Cumulative sums for wrap-around sliding window
        ext_sums = np.empty(2 * n_bins)
        ext_counts = np.empty(2 * n_bins, dtype=numba.int64)
        for i in range(n_bins):
            ext_sums[i] = bin_sums[i]
            ext_sums[i + n_bins] = bin_sums[i]
            ext_counts[i] = bin_counts[i]
            ext_counts[i + n_bins] = bin_counts[i]

        cum_sums = np.empty(2 * n_bins + 1)
        cum_counts = np.empty(2 * n_bins + 1, dtype=numba.int64)
        cum_sums[0] = 0.0
        cum_counts[0] = 0
        for i in range(2 * n_bins):
            cum_sums[i + 1] = cum_sums[i] + ext_sums[i]
            cum_counts[i + 1] = cum_counts[i] + ext_counts[i]

        best_sr = -1e30
        best_d = 0
        best_p = 0
        best_dep = 0.0

        for di in range(n_dur):
            width = dur_bins[di]
            for j in range(n_bins):
                wc = cum_counts[j + width] - cum_counts[j]
                oc = N - wc
                if wc < 3 or oc < 3:
                    continue
                ws = cum_sums[j + width] - cum_sums[j]
                mean_in = ws / wc
                mean_out = (total_sum - ws) / oc
                depth = mean_out - mean_in
                if depth > 0:
                    sr = depth * depth * wc * oc / (wc + oc)
                    if sr > best_sr:
                        best_sr = sr
                        best_d = width
                        best_p = j
                        best_dep = depth

        return best_sr, best_d, best_p, best_dep

    @numba.njit(parallel=True, cache=True)
    def _bls_all_periods(time_rel, flux, N, periods, n_periods, n_bins,
                         dur_bins, n_dur, total_sum):
        """BLS over all trial periods in parallel."""
        best_powers = np.empty(n_periods)
        best_durations = np.empty(n_periods)
        best_phases = np.empty(n_periods)
        best_depths = np.empty(n_periods)

        for pi in numba.prange(n_periods):
            sr, d, p, dep = _bls_single_period(
                time_rel, flux, N, periods[pi], n_bins, dur_bins, n_dur,
                total_sum)
            best_powers[pi] = sr
            best_durations[pi] = d
            best_phases[pi] = p
            best_depths[pi] = dep

        return best_powers, best_durations, best_phases, best_depths

    @numba.njit(parallel=True, cache=True)
    def _bls_batch_all_periods(time_rel, flux_matrix, N, periods, n_periods,
                               n_bins, dur_bins, n_dur, total_sums):
        """Batched BLS: multi-flux over a shared period grid.

        Shares the per-period phase-fold setup (bin index computation,
        bin_counts, cumulative counts) across all fluxes — those only
        depend on time_rel and period, not on the flux array. For n_flux
        candidates per period call, the shared work amortizes by up to
        (n_flux - 1) × N division+mod+cast operations, which dominate
        the inner BLS loop when N is large. Empirically ~2.9× total
        speedup for n_flux=90 on M1 (CPU path, 8 numba threads).

        flux_matrix: (n_flux, N) float64 contiguous.
        Returns: 4 arrays of shape (n_flux, n_periods).
        """
        n_flux = flux_matrix.shape[0]
        best_powers = np.empty((n_flux, n_periods))
        best_durations = np.empty((n_flux, n_periods))
        best_phases = np.empty((n_flux, n_periods))
        best_depths = np.empty((n_flux, n_periods))

        for pi in numba.prange(n_periods):
            period = periods[pi]

            # ----- shared across all fluxes for this period -----
            bin_idx = np.empty(N, dtype=numba.int64)
            bin_counts = np.zeros(n_bins, dtype=numba.int64)
            for i in range(N):
                ph = (time_rel[i] / period) % 1.0
                b = int(ph * n_bins)
                if b >= n_bins:
                    b = n_bins - 1
                bin_idx[i] = b
                bin_counts[b] += 1

            ext_counts = np.empty(2 * n_bins, dtype=numba.int64)
            for i in range(n_bins):
                ext_counts[i] = bin_counts[i]
                ext_counts[i + n_bins] = bin_counts[i]
            cum_counts = np.empty(2 * n_bins + 1, dtype=numba.int64)
            cum_counts[0] = 0
            for i in range(2 * n_bins):
                cum_counts[i + 1] = cum_counts[i] + ext_counts[i]

            # ----- per-flux inner loop -----
            for fi in range(n_flux):
                total_sum_f = total_sums[fi]

                bin_sums = np.zeros(n_bins)
                for i in range(N):
                    bin_sums[bin_idx[i]] += flux_matrix[fi, i]

                ext_sums = np.empty(2 * n_bins)
                for i in range(n_bins):
                    ext_sums[i] = bin_sums[i]
                    ext_sums[i + n_bins] = bin_sums[i]
                cum_sums = np.empty(2 * n_bins + 1)
                cum_sums[0] = 0.0
                for i in range(2 * n_bins):
                    cum_sums[i + 1] = cum_sums[i] + ext_sums[i]

                best_sr = -1e30
                best_d = 0
                best_p = 0
                best_dep = 0.0

                for di in range(n_dur):
                    width = dur_bins[di]
                    for j in range(n_bins):
                        wc = cum_counts[j + width] - cum_counts[j]
                        oc = N - wc
                        if wc < 3 or oc < 3:
                            continue
                        ws = cum_sums[j + width] - cum_sums[j]
                        mean_in = ws / wc
                        mean_out = (total_sum_f - ws) / oc
                        depth = mean_out - mean_in
                        if depth > 0:
                            sr = depth * depth * wc * oc / (wc + oc)
                            if sr > best_sr:
                                best_sr = sr
                                best_d = width
                                best_p = j
                                best_dep = depth

                best_powers[fi, pi] = best_sr
                best_durations[fi, pi] = best_d
                best_phases[fi, pi] = best_p
                best_depths[fi, pi] = best_dep

        return best_powers, best_durations, best_phases, best_depths


def bls_fast(time, flux, period_min=0.5, period_max=50.0, n_periods=10000,
             n_bins=200, n_dur_steps=15, duration_min_frac=0.01, duration_max_frac=0.15):
    """
    BLS transit search with numba-accelerated inner loop.

    Falls back to pure numpy if numba is not installed.

    Parameters
    ----------
    time : array-like
        Time values of the light curve.
    flux : array-like
        Flux values of the light curve.
    period_min, period_max : float
        Period search range (same units as time).
    n_periods : int
        Number of trial periods.
    n_bins : int
        Number of phase bins.
    n_dur_steps : int
        Number of transit duration steps.
    duration_min_frac, duration_max_frac : float
        Duration range as fraction of period.

    Returns
    -------
    dict with period, power, duration, t0, depth, depth_ppm, snr,
         periods (array), power_spectrum (array)
    """
    time = np.ascontiguousarray(time, dtype=np.float64)
    flux = np.ascontiguousarray(flux, dtype=np.float64)
    N = len(time)
    t0_ref = time[0]
    time_rel = time - t0_ref

    periods = _frequency_period_grid(period_min, period_max, n_periods)

    dur_bins = np.unique(np.linspace(
        max(1, int(duration_min_frac * n_bins)),
        min(n_bins // 3, int(duration_max_frac * n_bins)),
        n_dur_steps
    ).astype(np.int64))

    total_sum = float(np.sum(flux))

    if HAS_NUMBA:
        with numba_parallel_guard():
            bp, bd, bph, bdep = _bls_all_periods(
                time_rel, flux, N, periods, n_periods, n_bins,
                dur_bins, len(dur_bins), total_sum)
        best_powers = bp
        best_durations = bd / n_bins * periods
        best_phases = bph / n_bins
        best_depths = bdep
    else:
        # Pure numpy fallback (original implementation)
        best_powers = np.zeros(n_periods)
        best_durations = np.zeros(n_periods)
        best_phases = np.zeros(n_periods)
        best_depths = np.zeros(n_periods)

        _bin_sums = np.zeros(n_bins)
        _bin_counts = np.zeros(n_bins, dtype=int)

        for pi, period in enumerate(periods):
            phase = (time_rel / period) % 1.0
            bin_idx = (phase * n_bins).astype(int)
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)

            _bin_sums.fill(0)
            _bin_counts.fill(0)
            np.add.at(_bin_sums, bin_idx, flux)
            np.add.at(_bin_counts, bin_idx, 1)

            ext_sums = np.concatenate([_bin_sums, _bin_sums])
            ext_counts = np.concatenate([_bin_counts, _bin_counts])
            cum_sums = np.cumsum(ext_sums)
            cum_counts = np.cumsum(ext_counts)

            best_sr = -1e30
            best_d = 0
            best_p = 0
            best_dep = 0

            for width in dur_bins:
                win_sums = cum_sums[width:n_bins + width] - cum_sums[:n_bins]
                win_counts = cum_counts[width:n_bins + width] - cum_counts[:n_bins]
                out_counts = N - win_counts
                valid = (win_counts >= 3) & (out_counts >= 3)
                if not np.any(valid):
                    continue
                with np.errstate(divide='ignore', invalid='ignore'):
                    mean_in = np.where(valid, win_sums / win_counts, 0)
                    mean_out = np.where(valid, (total_sum - win_sums) / out_counts, 0)
                    depth = mean_out - mean_in
                    sr = np.where(valid & (depth > 0),
                                  depth * depth * win_counts * out_counts / (win_counts + out_counts),
                                  -1e30)
                max_idx = np.argmax(sr)
                if sr[max_idx] > best_sr:
                    best_sr = sr[max_idx]
                    best_d = width
                    best_p = max_idx
                    best_dep = depth[max_idx]

            best_powers[pi] = best_sr
            best_durations[pi] = best_d / n_bins * period
            best_phases[pi] = best_p / n_bins
            best_depths[pi] = best_dep

    best_idx = np.argmax(best_powers)
    best_period = periods[best_idx]

    snr = _compute_sde(best_powers, best_idx)

    return {
        "period": float(best_period),
        "power": float(best_powers[best_idx]),
        "duration": float(best_durations[best_idx]),
        "t0": float(best_phases[best_idx] * best_period + t0_ref),
        "depth": float(best_depths[best_idx]),
        "depth_ppm": float(best_depths[best_idx] * 1e6),
        "snr": snr,
        "periods": periods,
        "power_spectrum": best_powers,
    }


def bls_fast_batch(time, flux_list, period_min=0.5, period_max=50.0,
                   n_periods=10000, n_bins=200, n_dur_steps=15,
                   duration_min_frac=0.01, duration_max_frac=0.15):
    """Batched BLS across multiple flux arrays sharing one period grid.

    Equivalent to ``[bls_fast(time, f, ...) for f in flux_list]`` but
    amortizes the per-period phase-fold setup across all fluxes. Used
    by ``injection.run_injection_recovery`` to collapse the 720 single
    calls per TIC (9 radii × 8 periods × 10 injections) into one batch
    per period group — the same pattern ``bls_metal.bls_gpu_batch``
    already uses for GPU backends.

    Parameters
    ----------
    flux_list : list of 1D arrays, or 2D array of shape (n_flux, N)
        All fluxes must share the same ``time`` grid.
    remaining kwargs : see ``bls_fast``.

    Returns
    -------
    list of dict, one per input flux, each matching ``bls_fast``'s
    output (except the returned ``periods`` / ``power_spectrum`` arrays
    — kept per-flux for parity with ``bls_fast`` callers).
    """
    # Accept list or 2D array; normalize to contiguous (n_flux, N) float64.
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

    t0_ref = time[0]
    time_rel = time - t0_ref

    periods = _frequency_period_grid(period_min, period_max, n_periods)
    dur_bins = np.unique(np.linspace(
        max(1, int(duration_min_frac * n_bins)),
        min(n_bins // 3, int(duration_max_frac * n_bins)),
        n_dur_steps
    ).astype(np.int64))

    total_sums = flux_matrix.sum(axis=1).astype(np.float64)

    if HAS_NUMBA:
        with numba_parallel_guard():
            bp, bd, bph, bdep = _bls_batch_all_periods(
                time_rel, flux_matrix, N, periods, n_periods, n_bins,
                dur_bins, len(dur_bins), total_sums)
    else:
        # Fallback: just iterate bls_fast per flux. Keeps semantics but
        # loses the shared-setup speedup (unavailable without numba).
        return [bls_fast(time, flux_matrix[fi],
                         period_min=period_min, period_max=period_max,
                         n_periods=n_periods, n_bins=n_bins,
                         n_dur_steps=n_dur_steps,
                         duration_min_frac=duration_min_frac,
                         duration_max_frac=duration_max_frac)
                for fi in range(n_flux)]

    results = []
    for fi in range(n_flux):
        best_powers_f = bp[fi]
        best_durations_f = bd[fi] / n_bins * periods
        best_phases_f = bph[fi] / n_bins
        best_depths_f = bdep[fi]

        best_idx = int(np.argmax(best_powers_f))
        best_period = float(periods[best_idx])
        snr = _compute_sde(best_powers_f, best_idx)

        results.append({
            "period": best_period,
            "power": float(best_powers_f[best_idx]),
            "duration": float(best_durations_f[best_idx]),
            "t0": float(best_phases_f[best_idx] * best_period + t0_ref),
            "depth": float(best_depths_f[best_idx]),
            "depth_ppm": float(best_depths_f[best_idx] * 1e6),
            "snr": snr,
            "periods": periods,
            "power_spectrum": best_powers_f,
        })
    return results


def bls_two_pass(time, flux, period_min=0.5, period_max=50.0, n_periods=50000,
                 n_bins=200, n_dur_steps=15, duration_min_frac=0.01,
                 duration_max_frac=0.15):
    """Two-pass BLS: coarse scan + fine refinement around the best peak.

    Pass 1: 5000 periods (coarse) — finds approximate best period.
    Pass 2: n_periods/10 periods in narrow range around peak — refines.

    Typically 3-5x faster than single-pass with n_periods trials,
    with identical results for well-separated peaks.
    """
    # Pass 1: coarse scan
    n_coarse = min(5000, n_periods)
    r1 = bls_fast(time, flux, period_min=period_min, period_max=period_max,
                  n_periods=n_coarse, n_bins=n_bins, n_dur_steps=n_dur_steps,
                  duration_min_frac=duration_min_frac,
                  duration_max_frac=duration_max_frac)

    if r1["snr"] < 3.0:
        # No significant signal — return coarse result
        return r1

    # Pass 2: refine around best period
    coarse_step = (period_max - period_min) / n_coarse
    p_best = r1["period"]
    refine_half = max(coarse_step * 10, p_best * 0.02)  # ±2% or ±10 coarse steps
    p_min2 = max(period_min, p_best - refine_half)
    p_max2 = min(period_max, p_best + refine_half)
    n_fine = min(n_periods // 5, 10000)

    r2 = bls_fast(time, flux, period_min=p_min2, period_max=p_max2,
                  n_periods=n_fine, n_bins=n_bins, n_dur_steps=n_dur_steps,
                  duration_min_frac=duration_min_frac,
                  duration_max_frac=duration_max_frac)

    # Use refined period/t0/duration but report SNR from the coarse full-range scan
    # (SNR must be computed over the full period range for proper SDE)
    r2["snr"] = r1["snr"]
    r2["periods"] = r1["periods"]
    r2["power_spectrum"] = r1["power_spectrum"]
    return r2
