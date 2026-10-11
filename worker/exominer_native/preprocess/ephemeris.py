"""Native port of ``src_preprocessing/lc_preprocessing/utils_ephemeris.py``.

Pure-math helpers around transit ephemerides. No external I/O, no TF, no
pandas — just numpy. The port is a literal copy; there are no patches to
apply because the oracle source is already ``gfile``-free and uses no
deprecated numpy aliases.

Note (2026-04-18): the design doc §3.2 named ``get_tce_duration`` as a C.4
export but that function does not exist in the container source. The four
helpers below are the actual public surface of ``utils_ephemeris.py``.
"""
from __future__ import annotations

import numpy as np


def find_first_epoch_after_this_time(epoch, period, reference_time):
    """Find the first epoch after a certain reference time.

    :param epoch: float, initial epoch value (days).
    :param period: float, period (days).
    :param reference_time: float, start reference time (days).
    :return: float, new epoch value.

    Code is ported from Jeff Smith's Matlab ephemeris matching code.
    """
    if epoch < reference_time:
        tepoch = epoch + period * np.ceil((reference_time - epoch) / period)
    else:
        tepoch = epoch - period * np.floor((epoch - reference_time) / period)

    return tepoch


def create_binary_time_series(time, epoch, duration, period):
    """Create a binary time series based on the ephemeris.

    :param time: ``np.ndarray``, time in days.
    :param epoch: float, epoch in days.
    :param duration: float, transit duration in days.
    :param period: float, orbital period in days.
    :return: ``np.ndarray`` (bool), 1's for in-transit points and 0's
        otherwise.

    Code is ported from Jeff Smith's Matlab ephemeris matching code.
    """
    binary_time_series = np.zeros(len(time), dtype="bool")

    midTransitTimes = np.arange(epoch, time[-1], period)

    if len(midTransitTimes) != 0:
        midTransitTimeBefore = midTransitTimes[0] - period
        midTransitTimeAfter = midTransitTimes[-1] + period
    else:
        midTransitTimeBefore = epoch - period
        midTransitTimeAfter = epoch

    extendedMidTransitTimes = np.concatenate(
        [[midTransitTimeBefore], midTransitTimes, [midTransitTimeAfter]]
    )

    startTransitTimes = extendedMidTransitTimes - 0.5 * duration
    endTransitTimes = extendedMidTransitTimes + 0.5 * duration

    for sTransit, eTransit in zip(startTransitTimes, endTransitTimes):
        transit_idxs = np.where(np.logical_and(time >= sTransit, time < eTransit))[0]
        binary_time_series[transit_idxs] = 1

    return binary_time_series


def lininterp_transits(timeseries, transit_pulse_train, idxs_it, start_idxs, end_idxs):
    """Linearly interpolate the timeseries across the in-transit cadences.

    :param timeseries: list of ``np.ndarray``, time-series arrays.
    :param transit_pulse_train: list of ``np.ndarray``, binary arrays that
        are 1 for in-transit cadences and 0 otherwise.
    :param idxs_it: list of ``np.ndarray``, indexes of in-transit cadences
        in each element of ``transit_pulse_train``.
    :param start_idxs: list of ``np.ndarray``, starting index for each
        transit in ``idxs_it``.
    :param end_idxs: list of ``np.ndarray``, ending index for each transit
        in ``idxs_it``.
    :return: list of ``np.ndarray``, time-series linearly interpolated
        across the transits.
    """
    timeseries_interp = []

    for i in range(len(timeseries)):

        timeseries_interp.append(np.array(timeseries[i]))

        if len(idxs_it[i]) == 0:
            continue

        for start_idx, end_idx in zip(start_idxs[i], end_idxs[i]):

            if idxs_it[i][start_idx] == 0 and idxs_it[i][end_idx - 1] == len(transit_pulse_train[i]) - 1:
                continue

            if idxs_it[i][start_idx] == 0:
                timeseries_interp[i][idxs_it[i][start_idx]:idxs_it[i][end_idx - 1] + 1] = \
                    timeseries[i][idxs_it[i][end_idx - 1] + 1]

            elif idxs_it[i][end_idx - 1] == len(transit_pulse_train[i]) - 1:
                timeseries_interp[i][idxs_it[i][start_idx]:idxs_it[i][end_idx - 1] + 1] = \
                    timeseries[i][idxs_it[i][start_idx] - 1]

            else:
                idxs_interp = np.array([idxs_it[i][start_idx] - 1, idxs_it[i][end_idx - 1] + 1])
                idxs_to_interp = np.arange(idxs_it[i][start_idx], idxs_it[i][end_idx - 1] + 1)

                timeseries_interp[i][idxs_to_interp] = np.interp(
                    idxs_to_interp,
                    idxs_interp,
                    timeseries[i][idxs_interp],
                )

    return timeseries_interp


def get_startend_idxs_inter(transit_pulse_train):
    """Get in-transit indexes and the respective start and end indexes.

    :param transit_pulse_train: list of ``np.ndarray``, binary arrays that
        are 1 for in-transit cadences and 0 otherwise.
    :return: ``(idxs_it, start_idxs, end_idxs)``, each a list of arrays.
    """
    idxs_it, start_idxs, end_idxs = [], [], []
    for i in range(len(transit_pulse_train)):

        idxs_it.append(np.where(transit_pulse_train[i] == 1)[0])

        if len(idxs_it[-1]) == 0:
            start_idxs.append([])
            end_idxs.append([])
        else:
            idxs_lim = np.where(np.diff(idxs_it[-1]) > 1)[0] + 1

            start_idxs.append(np.insert(idxs_lim, 0, 0))
            end_idxs.append(np.append(idxs_lim, len(idxs_it[-1])))

    return idxs_it, start_idxs, end_idxs
