"""Subset of ``src_preprocessing/light_curve/util.py`` needed by the port.

We carry only what the preprocessing pipeline actually calls. Current
callers: ``detrend.py`` uses ``split``; ``phase_fold.py`` uses
``phase_fold_time`` and ``phase_fold_time_aug``. Other helpers from the
oracle's ``util.py`` (e.g. ``remove_events``, ``interpolate_missing_time``)
can be ported later if a caller needs them.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def phase_fold_time(time, period, t0):
    """Create a phase-folded time vector in ``[-period / 2, period / 2)``.

    ``result[i]`` is the unique number such that
    ``result[i] = time[i] - t0 + k_i * period`` for some integer ``k_i``.
    """
    half_period = period / 2
    result = np.mod(time + (half_period - t0), period)
    result -= half_period

    return result


def phase_fold_time_aug(time, period, t0):
    """Phase-fold with transit-level bootstrap augmentation.

    Same mapping as ``phase_fold_time`` but samples complete transits with
    replacement (keeping intra-transit order) so downstream binning sees a
    re-sampled phase curve.

    Returns ``(result_sampled, sampled_idxs, num_transits)``.
    """
    half_period = period / 2
    result = np.mod(time + (half_period - t0), period)
    result -= half_period

    edge_idxs = np.where(np.diff(result) < 0)[0] + 1
    edge_idxs = np.concatenate([[0], edge_idxs, [len(result)]])

    start_end_idxs = list(zip(edge_idxs[:-1], edge_idxs[1:]))

    num_transits = len(start_end_idxs)

    transit_idxs = np.random.randint(0, num_transits, size=num_transits)
    sampled_idxs = np.concatenate([
        np.arange(start_end_idxs[transit_idx][0], start_end_idxs[transit_idx][1])
        for transit_idx in transit_idxs
    ])

    result_sampled = result[sampled_idxs]

    return result_sampled, sampled_idxs, num_transits


def split(all_time, all_time_series, gap_width=0.75, add_info=None):
    """Splits a light curve on discontinuities (gaps).

    Accepts a time-series that is either a single segment or piecewise defined
    (e.g. split by quarter breaks or gaps in the data).

    :param all_time: ``np.ndarray`` or sequence of arrays, time values.
    :param all_time_series: list of time-series to be split according to
        the same time splits; each is a list of arrays.
    :param gap_width: minimum gap size (in time units) for a split.
    :param add_info: optional dict with extra metadata per segment
        (``quarter``, ``module``, ...).
    :return: ``(out_time, out_time_series, out_add_info)``.
    """
    if isinstance(all_time, np.ndarray) and all_time.ndim == 1:
        all_time = [all_time]
    if isinstance(all_time_series, np.ndarray) and all_time_series.ndim == 1:
        all_time_series = [all_time_series]

    out_time = []
    out_time_series = [[] for i in range(len(all_time_series))]

    out_add_info = (
        {el: [] if el in ["quarter", "module"] else add_info[el] for el in add_info}
        if add_info is not None
        else None
    )

    for arr_i in range(len(all_time)):

        start = 0
        for end in range(1, len(all_time[arr_i]) + 1):

            if end == len(all_time[arr_i]) or all_time[arr_i][end] - all_time[arr_i][end - 1] > gap_width:

                out_time.append(np.array(all_time[arr_i][start:end]))

                for timeseries_i in range(len(all_time_series)):
                    out_time_series[timeseries_i].append(
                        np.array(all_time_series[timeseries_i][arr_i][start:end])
                    )

                if add_info is not None:
                    if "quarter" in out_add_info:
                        out_add_info["quarter"].append(add_info["quarter"][arr_i])
                    if "module" in out_add_info:
                        out_add_info["module"].append(add_info["module"][arr_i])

                start = end

    return out_time, out_time_series, out_add_info
