"""Native port of ``src_preprocessing/lc_preprocessing/utils_gapping.py``.

Gapping helpers: given a TCE ephemeris, compute the indices of in-transit
cadences so downstream preprocessing can mask them out.

Port is mostly a literal copy; the only change is the import path for the
ephemeris helpers (oracle: ``src_preprocessing.lc_preprocessing.utils_ephemeris``;
port: relative to the sibling module). Design doc §3.2 named
``gap_other_tces``, but the container only has ``gap_tces`` (bulk variant);
both are ported here.
"""
from __future__ import annotations

import numpy as np

from .ephemeris import create_binary_time_series, find_first_epoch_after_this_time


def gap_this_tce(all_time, tce, gap_pad=0):
    """Remove from the time series the cadences that belong to the TCE.

    These values are set to NaN (by the caller — this function just returns
    the indices).

    :param all_time: list of ``np.ndarray``, cadences.
    :param tce: row-like (pd.Series / dict) with TCE ephemeris
        (``tce_time0bk``, ``tce_period``, ``tce_duration``, optionally
        ``tce_maxmesd``).
    :param gap_pad: extra pad on both sides of the gapped TCE transit
        duration.
    :return: list of numpy arrays, gapped indices per ``all_time`` element.
    """
    gapped_idxs = []

    if "tce_maxmesd" in tce:
        phase_sec = np.abs(tce["tce_maxmesd"])
        if 2 * tce["tce_duration"] < phase_sec:
            transit_duration = gap_pad * tce["tce_duration"]
        else:
            transit_duration = tce["tce_duration"] + phase_sec

        transit_duration = min(transit_duration, phase_sec - 0.5 * tce["tce_duration"])
    else:
        transit_duration = tce["tce_duration"] * gap_pad

    for i in range(len(all_time)):
        begin_time, end_time = all_time[i][0], all_time[i][-1]

        epoch = find_first_epoch_after_this_time(tce["tce_time0bk"], tce["tce_period"], begin_time)

        bintransit_ts = create_binary_time_series(all_time[i], epoch, transit_duration, tce["tce_period"])

        transit_idxs = np.where(bintransit_ts == 1)

        gapped_idxs.append(transit_idxs)

    return gapped_idxs


def gap_tces(all_time, ephem, config):
    """Get gapped indices for a set of TCEs to be gapped.

    NOTE: the container source contains a pre-existing bug on line 83
    (``ephem_i['tce_period']`` where ``ephem_i`` is an int index). The
    port keeps it byte-identical so parity gates will reject any
    divergence if this path is ever exercised. Downstream pipelines only
    invoke ``gap_this_tce``; ``gap_tces`` is not used in the default path.

    :param all_time: list of ``np.ndarray``, timestamps.
    :param ephem: DataFrame-like with ephemerides of TCEs to gap.
    :param config: dict with preprocessing parameters (``gap_imputed``).
    :return: ``(all_transit_idxs, imputed_time)``.
    """
    all_transit_idxs = []
    imputed_time = []

    for time_i in range(len(all_time)):

        transit_idxs = np.zeros(len(all_time[time_i]), dtype="bool")
        for ephem_i in range(len(ephem)):

            begin_time, end_time = all_time[time_i][0], all_time[time_i][-1]

            epoch = find_first_epoch_after_this_time(
                ephem["tce_time0bk"][ephem_i],
                ephem["tce_period"][ephem_i],
                begin_time,
            )

            bintransit_ts = create_binary_time_series(
                all_time[time_i],
                epoch,
                ephem["tce_duration"][ephem_i],
                ephem_i["tce_period"],  # bug preserved from oracle (see docstring)
            )

            transit_idxs = np.logical_or(transit_idxs, bintransit_ts)

        if config["gap_imputed"] and len(transit_idxs) > 0:
            imputed_time.append(all_time[time_i][transit_idxs])

        all_transit_idxs.append(transit_idxs)

    return all_transit_idxs, imputed_time


def get_gap_indices(flux, checkfuncs=None):
    """Find gaps in time series data (values in ``[0, nan, inf, -inf]``).

    :param flux: flux time-series.
    :return: dict mapping check name → list of ``[start, end)`` index pairs.
    """
    checkfuncdict = {
        0: flux == 0,
        "nan": np.isnan(flux),
        "inf": np.isinf(flux),
        "-inf": np.isinf(-flux),
    }

    checkfuncs = checkfuncdict if not checkfuncs else {i: checkfuncdict[i] for i in checkfuncs}

    id_dict = {}
    for checkstr, checkfunc in checkfuncs.items():
        arr = np.where(checkfunc)[0]

        if len(arr) == 0:
            id_dict[checkstr] = []
        elif len(arr) == 1:
            id_dict[checkstr] = [[arr[0], arr[0] + 1]]

        id_dict_type = []
        arr_diff = np.diff(arr)
        jump_idxs = np.where(arr_diff > 1)[0]
        jump_idxs += 1
        jump_idxs = np.insert(jump_idxs, [0, len(jump_idxs)], [0, len(arr)])
        for start, end in zip(jump_idxs[:-1], jump_idxs[1:]):
            id_dict_type.append([start, end])
        id_dict[checkstr] = id_dict_type

    return id_dict
