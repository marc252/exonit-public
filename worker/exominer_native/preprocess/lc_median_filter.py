"""Native port of ``src_preprocessing/light_curve/median_filter.py``.

Median-filter binning used by ``phase_fold.py`` to produce the global and
local views that feed the ONNX model. Literal copy — no patches needed.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from astropy.stats import mad_std


def median_filter(
    x,
    y,
    num_bins,
    bin_width=None,
    x_min=None,
    x_max=None,
    bin_fn=np.nanmedian,
    bin_var_fn=mad_std,
):
    """Computes the median y-value in uniform intervals (bins) along the x-axis.

    The interval ``[x_min, x_max]`` is divided into ``num_bins`` uniformly
    spaced intervals of width ``bin_width``. The value computed for each
    bin is ``bin_fn`` of all y-values whose corresponding x-value is in
    the interval.

    NOTE: ``x`` must be sorted in ascending order or the results will be
    incorrect.

    Returns ``(result, result_time, result_var, bin_counts)``.
    """
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2. Got: {}".format(num_bins))

    x_len = len(x)
    if x_len < 2:
        raise ValueError("len(x) must be at least 2. Got: {}".format(x_len))
    if x_len != len(y):
        raise ValueError("len(x) (got: {}) must equal len(y) (got: {})".format(x_len, len(y)))

    x_min = x_min if x_min is not None else x[0]
    x_max = x_max if x_max is not None else x[-1]
    if x_min >= x_max:
        raise ValueError("x_min (got: {}) must be less than x_max (got: {})".format(x_min, x_max))
    if x_min > x[-1]:
        raise ValueError(
            "x_min (got: {}) must be less than or equal to the largest value of x (got: {})".format(
                x_min, x[-1]
            )
        )

    bin_width = bin_width if bin_width is not None else (x_max - x_min) / num_bins
    if bin_width < np.median(np.diff(x)):
        bin_width = np.median(np.diff(x))
    if bin_width <= 0:
        raise ValueError("bin_width must be positive. Got: {}".format(bin_width))
    if bin_width >= x_max - x_min:
        raise ValueError(
            "bin_width (got: {}) must be less than x_max - x_min (got: {})".format(
                bin_width, x_max - x_min
            )
        )

    bin_spacing = (x_max - x_min - bin_width) / (num_bins - 1)

    result = np.nan * np.ones(num_bins, dtype="float")
    result_var = np.zeros(num_bins)
    result_time = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins, dtype="uint64")

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

        if j_end > j_start:
            result[i] = bin_fn(y[j_start:j_end])
            if bin_var_fn.__name__ == "mad_std":
                result_var[i] = bin_var_fn(y[j_start:j_end], ignore_nan=True)
            else:
                result_var[i] = bin_var_fn(y[j_start:j_end])

        bin_counts[i] = j_end - j_start
        result_time[i] = bin_min + bin_width / 2

        bin_min += bin_spacing
        bin_max += bin_spacing

    return result, result_time, result_var, bin_counts
