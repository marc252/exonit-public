"""Native port of ``src_preprocessing/lc_preprocessing/detrend_timeseries.py``.

Ports the spline-based detrend plus ``compute_bic``. The Savitzky-Golay
variant (``detrend_flux_using_sg_filter``) depends on ``lightkurve`` object
methods and is not on the native pipeline's default path — it can be ported
later if a caller needs it.

Patches vs the oracle:
  - ``from src_preprocessing.light_curve import util`` →
    ``from . import lc_util as util``
  - ``from src_preprocessing.third_party.kepler_spline import kepler_spline``
    → ``from .third_party import kepler_spline``
"""
from __future__ import annotations

import numpy as np
from astropy.stats import mad_std

from . import lc_util as util
from .third_party import kepler_spline


def detrend_flux_using_spline(flux_arrs, time_arrs, intransit_cadences, config):
    """Detrend timeseries by fitting a spline across linearly-interpolated transits.

    :param flux_arrs: list of ``np.ndarray``, timeseries.
    :param time_arrs: list of ``np.ndarray``, timestamps.
    :param intransit_cadences: list of ``np.ndarray``, in-transit flag arrays.
    :param config: dict with ``gapWidth``, ``random_seed``,
        ``spline_bkspace_{min,max,num}``, ``spline_maxiter``,
        ``spline_penalty_coeff``, ``spline_outlier_cut``.

    Returns ``(time, detrended_flux, spline_flux, flux_lininterp)``.
    """
    time_arrs, (flux_arrs, intransit_cadences), _ = util.split(
        time_arrs, [flux_arrs, intransit_cadences], gap_width=config["gapWidth"]
    )

    time, flux = np.concatenate(time_arrs), np.concatenate(flux_arrs)

    flux_arrs_lininterp = []
    for time_arr, flux_arr, intransit_cadences_arr in zip(time_arrs, flux_arrs, intransit_cadences):
        if intransit_cadences_arr.all():
            flux_lininterp_arr = np.nanmedian(flux_arr) * np.ones(len(time_arr), dtype="float")
        else:
            flux_lininterp_arr = np.interp(
                time_arr,
                time_arr[~intransit_cadences_arr],
                flux_arr[~intransit_cadences_arr],
                left=np.nan,
                right=np.nan,
            )

        flux_arrs_lininterp.append(flux_lininterp_arr)

    rng = np.random.default_rng(seed=config["random_seed"])
    for arr_i in range(len(flux_arrs_lininterp)):
        finite_idxs = np.isfinite(flux_arrs_lininterp[arr_i])
        mu, std = (
            np.nanmedian(flux_arrs_lininterp[arr_i][finite_idxs]),
            mad_std(flux_arrs_lininterp[arr_i][finite_idxs], ignore_nan=True),
        )
        flux_arrs_lininterp[arr_i][~finite_idxs] = rng.normal(mu, std, (~finite_idxs).sum())

    flux_lininterp = np.concatenate(flux_arrs_lininterp)

    spline_flux_arrs = kepler_spline.fit_kepler_spline(
        time_arrs,
        flux_arrs_lininterp,
        bkspace_min=config["spline_bkspace_min"],
        bkspace_max=config["spline_bkspace_max"],
        bkspace_num=config["spline_bkspace_num"],
        maxiter=config["spline_maxiter"],
        penalty_coeff=config["spline_penalty_coeff"],
        outlier_cut=config["spline_outlier_cut"],
        verbose=False,
    )[0]
    spline_flux = np.concatenate(spline_flux_arrs)

    finite_idxs = np.isfinite(spline_flux)

    if finite_idxs.any():
        spline_flux = np.interp(time, time[finite_idxs], spline_flux[finite_idxs], left=np.nan, right=np.nan)
    else:
        spline_flux = np.nanmedian(flux) * np.ones(len(flux))

    detrended_flux = flux / spline_flux

    dt = np.diff(time)
    cut = np.where(dt > 5 * np.nanmedian(dt))[0] + 1
    low = np.append([0], cut)
    high = np.append(cut, len(time))
    for low_i, high_i in zip(low, high):
        spline_flux[low_i:high_i] /= np.nanmedian(spline_flux[low_i:high_i])

    return time, detrended_flux, spline_flux, flux_lininterp


def detrend_flux_using_sg_filter(lc, mask_in_transit, win_len, sigma, max_poly_order, penalty_weight, break_tolerance):
    """Detrend a ``lightkurve.LightCurve`` using a Savitzky-Golay filter with BIC model selection.

    Port of NASA's ``detrend_flux_using_sg_filter`` (detrend_timeseries.py). Iterates
    polynomial orders 0..``max_poly_order``, fits each via ``lc.flatten(window_length,
    polyorder, mask=in_transit, break_tolerance, sigma)``, and picks the order with the
    lowest BIC score.

    Returns ``(time, detrended_flux, trend, models_info_df)``.
    """
    import pandas as pd

    if mask_in_transit.all():
        time = lc.time.value
        trend_arr = np.nanmedian(lc.flux.value) * np.ones_like(lc.flux.value)
        detrended_flux = lc.flux.value / trend_arr
        trend_arr = np.ones_like(lc.flux.value)
        models_info_df = pd.DataFrame({"poly_order": [0], "bic_score": [np.nan]})
        return time, detrended_flux, trend_arr, models_info_df

    n_samples = len(lc.flux.value)
    poly_order_arr = np.arange(0, max_poly_order + 1)
    best_bic, best_model = np.inf, 0
    lc_detrended = {po: {"flux": None, "trend": None, "bic": np.nan} for po in poly_order_arr}
    for poly_order in poly_order_arr:
        lc_flatten, lc_trend = lc.flatten(
            window_length=win_len, polyorder=poly_order,
            return_trend=True, sigma=sigma, mask=mask_in_transit,
            break_tolerance=break_tolerance,
        )
        lc_detrended[poly_order]["flux"] = lc_flatten
        lc_detrended[poly_order]["trend"] = lc_trend

        bic_k = compute_bic(poly_order, n_samples, lc.flux.value, lc_trend.flux.value,
                            penalty_weight=penalty_weight)
        lc_detrended[poly_order]["bic"] = bic_k
        if bic_k < best_bic:
            best_bic, best_model = bic_k, poly_order

    bic_scores = [lc_detrended[po]["bic"] for po in lc_detrended]
    models_info_df = pd.DataFrame({"poly_order": poly_order_arr, "bic_score": bic_scores})
    models_info_df.sort_values(by="bic_score", ascending=True, inplace=True)
    models_info_df.set_index("poly_order", inplace=True)

    time = lc_detrended[best_model]["flux"].time.value
    detrended_flux = lc_detrended[best_model]["flux"].flux.value
    trend = lc_detrended[best_model]["trend"].flux.value

    # Remove median from trend segment-by-segment (same logic as spline version).
    dt = np.diff(time)
    cut = np.where(dt > break_tolerance * np.nanmedian(dt))[0] + 1
    low = np.append([0], cut)
    high = np.append(cut, len(time))
    for low_i, high_i in zip(low, high):
        trend[low_i:high_i] /= np.nanmedian(trend[low_i:high_i])

    return time, detrended_flux, trend, models_info_df


def compute_bic(k, n, x, x_fit, penalty_weight=1):
    """BIC for a polynomial model of order ``k`` with ``n`` points.

    :param k: polynomial order.
    :param n: number of cadences in time series.
    :param x: raw flux time series.
    :param x_fit: flux trend.
    :param penalty_weight: penalty weight for the model's complexity.
    """
    sse = np.nansum((x - x_fit) ** 2)

    scaled_diffs = np.diff(x) / np.sqrt(2)
    sigma = np.nanmedian(np.abs(scaled_diffs)) * 1.48
    likelihood_term = k * np.log(2 * np.pi * sigma ** 2) + sse / sigma ** 2
    penalty_term = k * np.log(n)
    bic_score = likelihood_term + penalty_weight * penalty_term

    return bic_score
