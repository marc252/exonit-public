"""Phase-fold + median-filter views for FFI CNN training.

Direct port of worker/cnn_astronet.py:_phase_fold_sort and _median_filter,
extracted into a module shared by scripts and tests.
"""
from __future__ import annotations

import numpy as np

GLOBAL_BINS = 201
LOCAL_BINS = 61
GLOBAL_BIN_WIDTH_FACTOR = 1.2 / 201
LOCAL_BIN_WIDTH_FACTOR = 0.16
LOCAL_NUM_DURATIONS = 2


class ViewError(ValueError):
    """Raised when light curve is too sparse for view computation."""


def phase_fold_sort(time, flux, period, t0, duration=0.0):
    """Center mid-transit at 0, phase-fold and sort by phase.

    Our BLS reports t0 as the start of the transit window, so we add
    duration/2 to align with AstroNet convention (mid-transit at 0).
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    t0_center = t0 + duration / 2.0
    t = (time - t0_center) % period
    t[t >= period / 2] -= period
    order = np.argsort(t)
    return t[order], flux[order]


def _median_filter(x, y, num_bins, bin_width, x_min, x_max):
    """Bin median filter — port of light_curve_util.median_filter (Yu+ 2019).

    Computes median of y in each bin of width bin_width evenly spaced on
    [x_min, x_max]. Empty bins are filled by linear interpolation.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    in_range = (x >= x_min) & (x <= x_max)
    if in_range.sum() < 5 or (x[-1] - x[0]) < (x_max - x_min) / 2:
        raise ViewError("sparse light curve in view window")
    bin_spacing = (x_max - x_min - bin_width) / (num_bins - 1)
    result = np.full(num_bins, np.nan)
    for i in range(num_bins):
        lo = x_min + i * bin_spacing
        hi = lo + bin_width
        mask = (x >= lo) & (x < hi)
        if mask.any():
            result[i] = np.median(y[mask])
    return _fill_empty(result)


def _fill_empty(v):
    """Fill NaN bins by linear interpolation (Yu+ 2019)."""
    v = v.copy()
    n = len(v)
    nan = np.isnan(v)
    if not nan.any():
        return v
    if nan.all():
        raise ViewError("all bins empty")
    idx = np.arange(n)
    v[nan] = np.interp(idx[nan], idx[~nan], v[~nan])
    return v


def _normalize(view):
    """Normalize: median→0, |min|→1 (Yu+ 2019 AstroNet)."""
    view = view - np.median(view)
    m = np.abs(np.min(view))
    if m < 1e-12:
        m = 1e-12
    return (view / m).astype(np.float32)


def global_view(time, flux, period, t0, duration):
    """Compute global view (phase-folded full period, 201 bins)."""
    t_sorted, f_sorted = phase_fold_sort(time, flux, period, t0, duration)
    return _normalize(_median_filter(
        t_sorted, f_sorted,
        num_bins=GLOBAL_BINS,
        bin_width=period * GLOBAL_BIN_WIDTH_FACTOR,
        x_min=-period / 2,
        x_max=period / 2,
    ))


def local_view(time, flux, period, t0, duration):
    """Compute local view (zoomed around transit, 61 bins)."""
    t_sorted, f_sorted = phase_fold_sort(time, flux, period, t0, duration)
    half_window = min(period / 2, LOCAL_NUM_DURATIONS * duration)
    return _normalize(_median_filter(
        t_sorted, f_sorted,
        num_bins=LOCAL_BINS,
        bin_width=duration * LOCAL_BIN_WIDTH_FACTOR,
        x_min=-half_window,
        x_max=half_window,
    ))


AUX_CHANNELS = 3  # odd-local, even-local, secondary-local


def _raw_local_binned(time, flux, period, t0, duration):
    """Median-binned local view WITHOUT normalization, or None if sparse.

    Same window/binning as ``local_view`` but returns raw flux levels so
    callers can normalize several views on a shared scale (needed to keep
    odd/even depth differences meaningful).
    """
    t_sorted, f_sorted = phase_fold_sort(time, flux, period, t0, duration)
    half_window = min(period / 2, LOCAL_NUM_DURATIONS * duration)
    try:
        return _median_filter(
            t_sorted, f_sorted,
            num_bins=LOCAL_BINS,
            bin_width=duration * LOCAL_BIN_WIDTH_FACTOR,
            x_min=-half_window,
            x_max=half_window,
        )
    except ViewError:
        return None


def aux_views(time, flux, period, t0, duration):
    """Return (AUX_CHANNELS, LOCAL_BINS) float32: [odd, even, secondary].

    Discriminative views that distinguish planets from eclipsing binaries /
    false positives, all derived from the same (detrended) flux:
      - odd vs even local transit views, normalized on a SHARED scale so a
        depth difference (EB signature) survives.
      - secondary-eclipse local view (fold at phase 0.5), scaled by the
        PRIMARY transit amplitude so its relative depth is meaningful.
    Sparse/uncomputable channels are left as zeros.
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    out = np.zeros((AUX_CHANNELS, LOCAL_BINS), dtype=np.float32)
    if period <= 0 or duration <= 0 or time.size < 50:
        return out

    # Transit number per point (mid-transit centered), parity → odd/even.
    t0c = t0 + duration / 2.0
    n = np.round((time - t0c) / period)
    odd_m = (np.mod(n, 2) != 0)
    even_m = ~odd_m

    odd_raw = (_raw_local_binned(time[odd_m], flux[odd_m], period, t0, duration)
               if odd_m.sum() >= 50 else None)
    even_raw = (_raw_local_binned(time[even_m], flux[even_m], period, t0, duration)
                if even_m.sum() >= 50 else None)
    if odd_raw is not None and even_raw is not None:
        both = np.concatenate([odd_raw, even_raw])
        med = float(np.median(both))
        amp = abs(float(np.min(both - med)))
        amp = amp if amp > 1e-12 else 1e-12
        out[0] = ((odd_raw - med) / amp).astype(np.float32)
        out[1] = ((even_raw - med) / amp).astype(np.float32)

    # Secondary: fold half a period away; scale by primary depth.
    base_raw = _raw_local_binned(time, flux, period, t0, duration)
    sec_raw = _raw_local_binned(time, flux, period, t0 + period / 2.0, duration)
    if base_raw is not None and sec_raw is not None:
        med = float(np.median(base_raw))
        amp = abs(float(np.min(base_raw - med)))
        amp = amp if amp > 1e-12 else 1e-12
        out[2] = ((sec_raw - med) / amp).astype(np.float32)

    return out
