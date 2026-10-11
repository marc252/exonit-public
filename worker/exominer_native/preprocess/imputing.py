"""Native port of ``src_preprocessing/lc_preprocessing/utils_imputing.py``.

Helpers that fill in missing bins / gapped cadences so the downstream
phase-fold + model input generation can proceed.

Adapted from the oracle:
  - The oracle imports ``report_exclusion`` from
    ``utils_preprocessing_io`` and writes an exclusion record to a log
    directory whenever Gaussian statistics are NaN. The worker has no
    exclusion-log machinery, so we inline a local stub that ``logging``'s
    a warning. The callers' return values are unchanged either way, so
    the parity gate still holds.
"""
from __future__ import annotations

import logging

import numpy as np
from astropy.stats import mad_std


_log = logging.getLogger(__name__)


def _report_exclusion(msg, output_file_path=None):
    """Worker-local stand-in for the oracle's ``report_exclusion``.

    The oracle appends to a file; we emit a warning. Return-value behaviour
    of any caller is independent of which path runs, so parity is
    preserved.
    """
    _log.warning(str(msg))


def impute_binned_ts(
    binned_phase,
    binned_ts,
    period,
    duration,
    binned_ts_var=None,
    bin_fn=np.nanmedian,
    bin_var_fn=mad_std,
):
    """Impute binned time series.

    - For out-of-transit bins, missing values are filled with statistics
      computed from the oot bins (median, mad_std). If no valid oot bins
      exist, all valid bins are used.
    - For in-transit bins, linear interpolation is performed (or fall
      back to the oot strategy if no valid in-transit bin exists).

    Returns ``(binned_ts, binned_ts_var, inds_nan)``.
    """
    tmin_it, tmax_it = max(-period / 2, -1.5 * duration), min(period / 2, 1.5 * duration)
    inds_nan_bins = np.isnan(binned_ts)

    inds_it_binned_ts = (binned_phase >= tmin_it) & (binned_phase <= tmax_it)
    inds_it_nan = np.logical_and(inds_nan_bins, inds_it_binned_ts)
    inds_it_valid = np.logical_and(~inds_nan_bins, inds_it_binned_ts)

    inds_oot_binned_ts = ~inds_it_binned_ts
    inds_oot_nan = np.logical_and(inds_nan_bins, inds_oot_binned_ts)
    inds_oot_valid = np.logical_and(~inds_nan_bins, inds_oot_binned_ts)

    inds_nan = {"oot": inds_oot_nan, "it": inds_it_nan}

    if np.any(inds_oot_nan):
        if np.any(inds_oot_valid):
            mu = bin_fn(binned_ts[inds_oot_valid])
            if bin_var_fn.__name__ == "mad_std":
                sigma = bin_var_fn(binned_ts[inds_oot_valid], ignore_nan=True)
            else:
                sigma = bin_var_fn(binned_ts[inds_oot_valid])
        else:
            mu = bin_fn(binned_ts)
            if bin_var_fn.__name__ == "mad_std":
                sigma = bin_var_fn(binned_ts, ignore_nan=True)
            else:
                sigma = bin_var_fn(binned_ts)

        binned_ts = np.where(inds_oot_nan, mu, binned_ts)
        if binned_ts_var is not None:
            binned_ts_var = np.where(inds_oot_nan, sigma, binned_ts_var)

    if np.any(inds_it_nan):
        if np.any(inds_it_valid):

            if bin_var_fn.__name__ == "mad_std":
                sigma = bin_var_fn(binned_ts, ignore_nan=True)
            else:
                sigma = bin_var_fn(binned_ts)

            inds_valid_interp = ~np.isnan(binned_ts)
            binned_ts[inds_it_nan] = np.interp(
                binned_phase[inds_it_nan],
                binned_phase[inds_valid_interp],
                binned_ts[inds_valid_interp],
            )
            if binned_ts_var is not None:
                binned_ts_var[inds_it_nan] = sigma
        else:
            mu = bin_fn(binned_ts)
            if bin_var_fn.__name__ == "mad_std":
                sigma = bin_var_fn(binned_ts, ignore_nan=True)
            else:
                sigma = bin_var_fn(binned_ts)

            binned_ts = np.where(inds_it_nan, mu, binned_ts)
            if binned_ts_var is not None:
                binned_ts_var = np.where(inds_it_nan, sigma, binned_ts_var)

    return binned_ts, binned_ts_var, inds_nan


def imputing_gaps(timeseries, idxs_to_impute, seed, tce=None, config=None):
    """Impute gaps with Gaussian noise (mu = nanmedian, sigma = mad_std).

    If the time series is all-NaN (mu or sigma NaN), the timeseries is
    returned untouched and a warning is logged.
    """
    rng = np.random.default_rng(seed=seed)

    mu = np.nanmedian(timeseries)
    sigma = mad_std(timeseries, ignore_nan=True)

    if np.isnan(mu) or np.isnan(sigma):
        uid = tce["uid"] if tce is not None and "uid" in tce else "<unknown>"
        out_path = None
        if config is not None and "exclusion_logs_dir" in config:
            out_path = config["exclusion_logs_dir"] / f"exclusions_{uid}.txt"
        _report_exclusion(
            f"Gaussian statistics are NaN when imputing gapped data for TCE {uid}. Skipping imputing.",
            out_path,
        )
        return timeseries

    timeseries[idxs_to_impute] = rng.normal(mu, sigma, idxs_to_impute.sum())

    return timeseries


def impute_transits(timeseries, transit_pulse_train, centroid=False):
    """Impute transits by linearly interpolating across them.

    Boundary transits use the out-of-transit median in windows around the
    transit whose width equals the transit's own width.
    """
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

            numcadences_win = idxs_it[end_idx - 1] - idxs_it[start_idx] + 1

            if idxs_it[start_idx] == 0 and idxs_it[end_idx - 1] == len(transit_pulse_train[i]) - 1:
                continue

            if idxs_it[start_idx] == 0:

                if centroid:
                    val_interp = {
                        "x": np.median(timeseries["x"][i][
                            idxs_it[end_idx - 1] + 1:min(
                                len(timeseries["x"][i]),
                                idxs_it[end_idx - 1] + 1 + numcadences_win,
                            )
                        ]),
                        "y": np.median(timeseries["y"][i][
                            idxs_it[end_idx - 1] + 1:min(
                                len(timeseries["y"][i]),
                                idxs_it[end_idx - 1] + 1 + numcadences_win,
                            )
                        ]),
                    }
                    timeseries_interp["x"][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = val_interp["x"]
                    timeseries_interp["y"][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = val_interp["y"]
                else:
                    val_interp = np.median(timeseries[i][
                        idxs_it[end_idx - 1] + 1:min(
                            len(timeseries[i]),
                            idxs_it[end_idx - 1] + 1 + numcadences_win,
                        )
                    ])
                    timeseries_interp[i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = val_interp

            elif idxs_it[end_idx - 1] == len(transit_pulse_train[i]) - 1:

                if centroid:
                    val_interp = {
                        "x": np.median(
                            timeseries["x"][i][max(0, idxs_it[start_idx] - numcadences_win):idxs_it[start_idx] + 1]
                        ),
                        "y": np.median(
                            timeseries["y"][i][max(0, idxs_it[start_idx] - numcadences_win):idxs_it[start_idx] + 1]
                        ),
                    }
                    timeseries_interp["x"][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = val_interp["x"]
                    timeseries_interp["y"][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = val_interp["y"]
                else:
                    val_interp = np.median(
                        timeseries[i][max(0, idxs_it[start_idx] - numcadences_win):idxs_it[start_idx] + 1]
                    )
                    timeseries_interp[i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = val_interp

            else:
                idxs_interp = np.array([idxs_it[start_idx] - 1, idxs_it[end_idx - 1] + 1])
                idxs_to_interp = np.arange(idxs_it[start_idx], idxs_it[end_idx - 1] + 1)

                if centroid:
                    val_interp = {
                        "x": np.array([
                            np.median(
                                timeseries["x"][i][max(0, idxs_interp[0] - numcadences_win):idxs_interp[0] + 1]
                            ),
                            np.median(
                                timeseries["x"][i][
                                    idxs_interp[1]:min(
                                        len(timeseries["x"][i]),
                                        idxs_interp[1] + numcadences_win,
                                    )
                                    + 1
                                ]
                            ),
                        ]),
                        "y": np.array([
                            np.median(
                                timeseries["y"][i][max(0, idxs_interp[0] - numcadences_win):idxs_interp[0] + 1]
                            ),
                            np.median(
                                timeseries["y"][i][
                                    idxs_interp[1]:min(
                                        len(timeseries["y"][i]),
                                        idxs_interp[1] + numcadences_win,
                                    )
                                    + 1
                                ]
                            ),
                        ]),
                    }
                    timeseries_interp["x"][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = np.interp(
                        idxs_to_interp, idxs_interp, val_interp["x"]
                    )
                    timeseries_interp["y"][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = np.interp(
                        idxs_to_interp, idxs_interp, val_interp["y"]
                    )
                else:
                    val_interp = np.array([
                        np.median(
                            timeseries[i][max(0, idxs_interp[0] - numcadences_win):idxs_interp[0] + 1]
                        ),
                        np.median(
                            timeseries[i][
                                idxs_interp[1]:min(
                                    len(timeseries[i]),
                                    idxs_interp[1] + numcadences_win,
                                )
                                + 1
                            ]
                        ),
                    ])
                    timeseries_interp[i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = np.interp(
                        idxs_to_interp, idxs_interp, val_interp
                    )

    return timeseries_interp
