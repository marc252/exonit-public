"""NASA-compatible Lomb-Scargle periodogram features for ExoMiner.

Ported from ``src_preprocessing/lc_preprocessing/lc_periodogram.py`` in the
NASA ExoMiner repo (https://github.com/nasa/ExoMiner). The parameters used
here come from ``config_preprocessing.yaml`` in the same repo:

    p_min_tce        = 0.2 days    # shortest period to measure
    k_harmonics      = 5           # f_max = k_harmonics / p_min_tce
    p_max_obs        = 54 days     # longest period to measure
    downsampling_f   = 2           # freq-grid downsampling factor
    smooth_filter    = boxkernel   # lightkurve smoothing mode
    smooth_filter_w_f = 2          # smoothing kernel width factor
    n_durations      = 2           # #transit-durations for pulse model

The output feature vectors have exactly the length the ONNX model
expects — for the default config this is 674 frequency bins.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .ephemeris import create_binary_time_series, find_first_epoch_after_this_time

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config constants (match NASA config_preprocessing.yaml)
# ---------------------------------------------------------------------------

P_MIN_TCE = 0.2
K_HARMONICS = 5
P_MAX_OBS = 54.0
DOWNSAMPLING_F = 2
SMOOTH_FILTER_TYPE = "boxkernel"
SMOOTH_FILTER_W_F = 2
N_DURATIONS = 2  # transit durations for transit pulse model


# ---------------------------------------------------------------------------
# Frequency grid
# ---------------------------------------------------------------------------

def compute_frequency_range(
    p_min_tce: float = P_MIN_TCE,
    k_harmonics: int = K_HARMONICS,
    p_max_obs: float = P_MAX_OBS,
    downsampling_f: float = DOWNSAMPLING_F,
) -> np.ndarray:
    """Port of NASA's ``compute_frequency_range`` — linear freq grid in 1/days."""
    f_max = (1.0 / p_min_tce) * k_harmonics
    f_min = 1.0 / p_max_obs
    n_freq_pts = int((f_max - f_min) / (downsampling_f * f_min))
    return np.linspace(f_min, f_max, n_freq_pts, endpoint=True)


# ---------------------------------------------------------------------------
# Transit pulse model (synthetic LC with transit dips at TCE ephemeris)
# ---------------------------------------------------------------------------

def compute_transit_pulse_model(
    time: np.ndarray,
    flux: np.ndarray,
    tce_period: float,
    tce_t0: float,
    tce_duration: float,
    tce_depth_ppm: float,
    tce_maxmesd: float = 0.0,
    wst_depth_ppm: float = 0.0,
    n_durations: float = N_DURATIONS,
) -> np.ndarray:
    """Build a synthetic flux array with the TCE (and weak-secondary) transit
    dips at the TCE ephemeris. Based on NASA's ``compute_lc_transit_pulse_model``.

    Args:
        time: 1D array of timestamps (days, BJD-adjusted).
        flux: 1D array of observed flux (used only for its length + median).
        tce_period: orbital period (days).
        tce_t0: transit epoch (days, same frame as ``time``).
        tce_duration: transit duration (days).
        tce_depth_ppm: primary transit depth (ppm).
        tce_maxmesd: MES-derived secondary phase offset (days).
        wst_depth_ppm: weak-secondary transit depth (ppm). 0 = skip secondary.
        n_durations: #transit-durations wide the synthetic dip should be.
    """
    flux_med = float(np.nanmedian(flux)) if flux.size else 1.0
    t0 = float(tce_t0)
    per = float(tce_period)
    dur = float(tce_duration)  # in days
    depth = float(tce_depth_ppm) * 1e-6
    wst_depth = float(wst_depth_ppm) * 1e-6

    if not (np.isfinite(per) and per > 0 and np.isfinite(dur) and dur > 0):
        return np.full_like(time, flux_med, dtype=float)

    # Primary transit cadences
    first_t = find_first_epoch_after_this_time(t0, per, float(time[0]))
    in_pri = create_binary_time_series(time, first_t, dur * n_durations, per)

    pulse = np.full_like(time, flux_med, dtype=float)
    pulse[in_pri > 0] = flux_med - depth * flux_med

    # Weak-secondary transit cadences (if present)
    if np.isfinite(tce_maxmesd) and wst_depth > 0:
        sec_first = find_first_epoch_after_this_time(
            t0 + float(tce_maxmesd), per, float(time[0]))
        in_sec = create_binary_time_series(time, sec_first, dur * n_durations, per)
        pulse[in_sec > 0] = flux_med - wst_depth * flux_med

    return pulse


# ---------------------------------------------------------------------------
# Lomb-Scargle periodogram + smoothing
# ---------------------------------------------------------------------------

def _compute_pgram_smoothed(
    time: np.ndarray,
    flux: np.ndarray,
    f_arr: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute the smoothed, max-normalized Lomb-Scargle amplitude periodogram.

    Returns ``(pgram_smooth_norm, max_power)`` where ``pgram_smooth_norm`` is
    in ``[0, 1]`` and ``max_power`` is the un-normalized peak (pre-norm).
    """
    import lightkurve as lk
    from lightkurve import periodogram as lk_periodogram

    mask = np.isfinite(time) & np.isfinite(flux)
    if mask.sum() < 100:
        return np.zeros(len(f_arr), dtype=np.float32), 0.0

    lc = lk.LightCurve(time=np.asarray(time[mask], dtype=float),
                       flux=np.asarray(flux[mask], dtype=float))

    # Lomb-Scargle with amplitude normalization + the fast NUFFT method.
    pgram = lk_periodogram.LombScarglePeriodogram.from_lightcurve(
        lc, frequency=f_arr, normalization="amplitude", ls_method="fast",
    )

    # boxkernel smoother: width = frequency step × smooth_filter_w_f
    smooth_w = float(np.diff(f_arr)[0]) * SMOOTH_FILTER_W_F
    pgram_smooth = pgram.smooth(
        method=SMOOTH_FILTER_TYPE, filter_width=smooth_w)

    max_power = float(pgram_smooth.max_power.value)
    if max_power <= 0 or not np.isfinite(max_power):
        return np.zeros(len(f_arr), dtype=np.float32), 0.0

    return (np.asarray(pgram_smooth.power.value / max_power, dtype=np.float32),
            max_power)


# ---------------------------------------------------------------------------
# Public entry point for orchestrate.run_tic
# ---------------------------------------------------------------------------

def compute_pgram_features(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    tce_period: float,
    tce_t0: float,
    tce_duration: float,
    tce_depth_ppm: float,
    tce_maxmesd: float = 0.0,
    wst_depth_ppm: float = 0.0,
) -> dict[str, Any]:
    """Compute the 4 periodogram features the ExoMiner ONNX model expects:

        pgram_smooth               — LS amplitude pgram of the LC, max-norm.
        pgram_smooth_max_power     — scalar peak of pgram_smooth (pre-norm).
        pgram_tpm_smooth           — LS amplitude pgram of the transit pulse model.
        pgram_tpm_smooth_max_power — scalar peak of pgram_tpm_smooth (pre-norm).

    On any failure the function returns zero-filled arrays / zero scalars so
    the caller can still produce a feed dict.
    """
    try:
        f_arr = compute_frequency_range()
    except Exception as e:
        _log.warning("compute_frequency_range failed: %s", e)
        return {
            "pgram_smooth": np.zeros(674, dtype=np.float32),
            "pgram_smooth_max_power": 0.0,
            "pgram_tpm_smooth": np.zeros(674, dtype=np.float32),
            "pgram_tpm_smooth_max_power": 0.0,
        }

    pgram_smooth, pgram_smooth_maxp = _compute_pgram_smoothed(time, flux, f_arr)

    # Transit pulse model + its own pgram
    try:
        lc_tpm = compute_transit_pulse_model(
            time, flux,
            tce_period=tce_period, tce_t0=tce_t0,
            tce_duration=tce_duration, tce_depth_ppm=tce_depth_ppm,
            tce_maxmesd=tce_maxmesd, wst_depth_ppm=wst_depth_ppm,
        )
        pgram_tpm_smooth, pgram_tpm_smooth_maxp = _compute_pgram_smoothed(
            time, lc_tpm, f_arr)
    except Exception as e:
        _log.warning("pgram TPM compute failed: %s", e)
        pgram_tpm_smooth = np.zeros(len(f_arr), dtype=np.float32)
        pgram_tpm_smooth_maxp = 0.0

    # The 674-bin pgrams are already max-normalized to [0, 1], matching the
    # oracle's ``pgram_smooth_norm`` feed directly (no further normalization
    # happens in the normalizer for these array features). The max_power
    # scalars keep the bare-name — they go through the scalar normalizer
    # (median/mad) which then emits ``<name>_norm`` feeds.
    return {
        "pgram_smooth_norm": pgram_smooth,
        "pgram_smooth_max_power": float(pgram_smooth_maxp),
        "pgram_tpm_smooth_norm": pgram_tpm_smooth,
        "pgram_tpm_smooth_max_power": float(pgram_tpm_smooth_maxp),
    }
