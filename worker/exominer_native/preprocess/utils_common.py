"""Native port of ``src_preprocessing/lc_preprocessing/utils_preprocessing.py``.

General numpy/math helpers: view-window index helpers, transit counters,
normalisation, linear-interpolation across transits, outlier removal, and
input-sanity checks.

Patches vs the oracle:
  - ``from .utils_preprocessing_io import report_exclusion`` → replaced by
    a local ``_report_exclusion`` stub that logs a warning. Callers'
    return values don't depend on whether it writes to a file, so the
    parity gate still holds.
"""
from __future__ import annotations

import logging

import numpy as np
from astropy.stats import mad_std


_log = logging.getLogger(__name__)


def _report_exclusion(msg, output_file_path=None):
    """Worker-local stand-in for ``utils_preprocessing_io.report_exclusion``."""
    _log.warning(str(msg))


def get_out_of_transit_idxs_loc(num_bins_loc, num_transit_durations):
    """Get indices of out-of-transit cadences for the local views."""
    transit_duration_bins_loc = num_bins_loc / num_transit_durations
    nontransitcadences_loc = np.array([True] * num_bins_loc)
    nontransitcadences_loc[np.arange(
        int(np.floor((num_bins_loc - transit_duration_bins_loc) / 2)),
        int(np.ceil((num_bins_loc + transit_duration_bins_loc) / 2)),
    )] = False
    idxs_nontransitcadences_loc = np.where(nontransitcadences_loc)
    return idxs_nontransitcadences_loc


def get_out_of_transit_idxs_glob(num_bins_glob, transit_duration, orbital_period):
    """Get indices of out-of-transit cadences for the global views."""
    frac_durper = transit_duration / orbital_period
    nontransitcadences_glob = np.array([True] * num_bins_glob)
    nontransitcadences_glob[np.arange(
        int(num_bins_glob / 2 * (1 - frac_durper)),
        int(num_bins_glob / 2 * (1 + frac_durper)),
    )] = False
    idxs_nontransitcadences_glob = np.where(nontransitcadences_glob)
    return idxs_nontransitcadences_glob


def _count_num_bins(x, x_max, x_min, bin_width, num_bins):
    """Count number of points per bin."""
    pts_per_bin = np.zeros(num_bins, dtype="uint64")

    if num_bins < 2:
        raise ValueError("num_bins must be at least 2. Got: {}".format(num_bins))

    x_len = len(x)
    if x_len < 2:
        pts_per_bin[0] = 1
        return pts_per_bin

    x_min = x_min if x_min is not None else x[0]
    x_max = x_max if x_max is not None else x[-1]
    if x_min >= x_max:
        raise ValueError("x_min (got: {}) must be less than x_max (got: {})".format(x_min, x_max))
    if x_min > x[-1]:
        return pts_per_bin

    bin_width = bin_width if bin_width is not None else (x_max - x_min) / num_bins
    if bin_width <= 0:
        raise ValueError("bin_width must be positive. Got: {}".format(bin_width))
    if bin_width >= x_max - x_min:
        raise ValueError(
            "bin_width (got: {}) must be less than x_max - x_min (got: {})".format(bin_width, x_max - x_min)
        )

    bin_spacing = (x_max - x_min - bin_width) / (num_bins - 1)

    x_start = 0
    while x[x_start] < x_min:
        x_start += 1

    bin_min = x_min
    bin_max = x_min + bin_width
    j_start = x_start
    j_end = x_start

    for i in range(num_bins):
        while j_start < x_len and x[j_start] < bin_min:
            j_start += 1

        while j_end < x_len and x[j_end] < bin_max:
            j_end += 1

        pts_per_bin[i] = j_end - j_start

        bin_min += bin_spacing
        bin_max += bin_spacing

    return pts_per_bin


def count_transits(time, period, epoch, duration, num_cadences_min=1):
    """Count number of transits. Valid if at least ``num_cadences_min`` cadences fall inside the transit window."""
    if len(time) == 0:
        return 0

    mid_transit_time = epoch + period * np.ceil((time[0] - epoch) / period)
    mid_transit_times = [mid_transit_time]

    while True:
        mid_transit_time += period
        mid_transit_times.append(mid_transit_time)
        if mid_transit_time > time[-1]:
            break

    return np.sum([
        len(np.where(
            (time >= mid_transit_time - duration / 2)
            & (time <= mid_transit_time + duration / 2)
        )[0]) >= num_cadences_min
        for mid_transit_time in mid_transit_times
    ])


def min_max_normalization(arr, max_val, min_val):
    """Min-max normalization: (arr - min_val) / (max_val - min_val)."""
    return (arr - min_val) / (max_val - min_val)


def lininterp_transits(timeseries, transit_pulse_train, centroid=False):
    """Linearly interpolate the timeseries across in-transit cadences using the boundary points' values."""
    if centroid:
        num_arrs = len(timeseries["x"])
        timeseries_interp = {"x": [], "y": []}
    else:
        num_arrs = len(timeseries)
        timeseries_interp = []

    for i in range(num_arrs):

        if centroid:
            timeseries_interp["x"].append(np.array(timeseries["x"][i]))
            timeseries_interp["y"].append(np.array(timeseries["y"][i]))
        else:
            timeseries_interp.append(np.array(timeseries[i]))

        idxs_it = np.where(transit_pulse_train[i] == 1)[0]
        if len(idxs_it) == 0:
            continue

        idxs_lim = np.where(np.diff(idxs_it) > 1)[0] + 1

        start_idxs = np.insert(idxs_lim, 0, 0)
        end_idxs = np.append(idxs_lim, len(idxs_it))

        for start_idx, end_idx in zip(start_idxs, end_idxs):

            if idxs_it[start_idx] == 0 and idxs_it[end_idx - 1] == len(transit_pulse_train[i]) - 1:
                continue

            if idxs_it[start_idx] == 0:
                if centroid:
                    timeseries_interp["x"][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        timeseries["x"][i][idxs_it[end_idx - 1] + 1]
                    timeseries_interp["y"][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        timeseries["y"][i][idxs_it[end_idx - 1] + 1]
                else:
                    timeseries_interp[i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        timeseries[i][idxs_it[end_idx - 1] + 1]

            elif idxs_it[end_idx - 1] == len(transit_pulse_train[i]) - 1:
                if centroid:
                    timeseries_interp["x"][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        timeseries["x"][i][idxs_it[start_idx] - 1]
                    timeseries_interp["y"][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        timeseries["y"][i][idxs_it[start_idx] - 1]
                else:
                    timeseries_interp[i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        timeseries[i][idxs_it[start_idx] - 1]

            else:
                idxs_interp = np.array([idxs_it[start_idx] - 1, idxs_it[end_idx - 1] + 1])
                idxs_to_interp = np.arange(idxs_it[start_idx], idxs_it[end_idx - 1] + 1)

                if centroid:
                    timeseries_interp["x"][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        np.interp(idxs_to_interp, idxs_interp, timeseries["x"][i][idxs_interp])
                    timeseries_interp["y"][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        np.interp(idxs_to_interp, idxs_interp, timeseries["y"][i][idxs_interp])
                else:
                    timeseries_interp[i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        np.interp(idxs_to_interp, idxs_interp, timeseries[i][idxs_interp])

    return timeseries_interp


def remove_non_finite_values(arrs):
    """Remove cadences with non-finite values (NaN or inf) for timestamps/timeseries."""
    num_arrs = len(arrs)
    num_sub_arrs = len(arrs[0])

    for sub_arr_i in range(num_sub_arrs):
        finite_idxs = []

        for arr in arrs:
            finite_idxs.append(np.isfinite(arr[sub_arr_i]))

        finite_idxs = np.logical_and.reduce(finite_idxs, dtype="bool")

        for arr_i in range(num_arrs):
            arrs[arr_i][sub_arr_i] = arrs[arr_i][sub_arr_i][finite_idxs]

    return arrs


def remove_outliers(ts, sigma, fill=False, outlier_type="upper", seed=None):
    """Remove outliers (upper/lower/both) in a flux time series using a sigma threshold."""
    if outlier_type not in ["upper", "lower", "both"]:
        raise TypeError(f"Outlier type {outlier_type} not recognized. Choose between `upper`, `lower`, or `both`.")

    ts_arr_list = False
    if isinstance(ts, list):
        flattened_arr = [x for xs in ts for x in xs]
        rob_std = mad_std(flattened_arr, ignore_nan=True)
        mu_val = np.nanmedian(flattened_arr)
        ts_arr_list = True
    else:
        rob_std = mad_std(ts, ignore_nan=True)
        mu_val = np.nanmedian(ts)

        ts = [ts]

    idxs_out = []
    for np_ts_arr in ts:

        if outlier_type == "upper":
            idxs_out_arr = np_ts_arr > mu_val + sigma * rob_std
        elif outlier_type == "lower":
            idxs_out_arr = np_ts_arr < mu_val - sigma * rob_std
        elif outlier_type == "both":
            idxs_out_arr = np.abs(np_ts_arr) > mu_val + sigma * rob_std

        if fill:
            rng = np.random.default_rng(seed)
            np_ts_arr[idxs_out_arr] = rng.normal(mu_val, rob_std, idxs_out_arr.sum())
        else:
            np_ts_arr[idxs_out_arr] = np.nan

        idxs_out.append(idxs_out_arr)

    if not ts_arr_list:
        ts = ts[0]
        idxs_out = idxs_out[0]

    return ts, idxs_out


def check_inputs_generate_example(data, tce, config):
    """Check inputs before phase-fold + binning; fill centroid data if missing."""
    if data["centroid_dist"] is None:
        data["centroid_dist"] = []

    if len(data["centroid_dist"]) == 0 or np.isnan(data["centroid_dist"]).all():
        uid = tce["uid"] if "uid" in tce else "<unknown>"
        out_path = (
            config["exclusion_logs_dir"] / f"exclusions-{uid}.txt"
            if config is not None and "exclusion_logs_dir" in config
            else None
        )
        _report_exclusion(
            "No data points available. Setting centroid offset time series to zero (target star position).",
            out_path,
        )

        data["centroid_dist_time"] = np.array(data["flux_time"])
        data["centroid_dist"] = np.zeros(len(data["flux_time"]))

    elif "Less than 0.5% cadences are valid for centroid data." in data["errors"]:
        uid = tce["uid"] if "uid" in tce else "<unknown>"
        out_path = (
            config["exclusion_logs_dir"] / f"exclusions-{uid}.txt"
            if config is not None and "exclusion_logs_dir" in config
            else None
        )
        _report_exclusion(
            "Less than 0.5% cadences are valid for centroid data. Setting centroid "
            "offset time series to Gaussian noise using statistics from this time series.",
            out_path,
        )

        rob_std = mad_std(data["centroid_dist"], ignore_nan=True)
        med = np.nanmedian(data["centroid_dist"])
        rng = np.random.default_rng(seed=config["random_seed"])
        data["centroid_dist"] = rng.normal(med, rob_std, data["flux"].shape)
        data["centroid_dist_time"] = np.array(data["flux_time"])

    return data


def check_inputs(data):
    """Check data coming from the LC FITS files and flag low-coverage centroid issues."""
    errs = []

    n_valid_cad = np.isfinite(np.concatenate(data["all_centroids"]["x"])).sum()
    n_cad = len(np.concatenate(data["all_centroids"]["x"]))
    if n_valid_cad / n_cad < 0.005:
        errs.append("Less than 0.5% cadences are valid for centroid data.")

    return errs
