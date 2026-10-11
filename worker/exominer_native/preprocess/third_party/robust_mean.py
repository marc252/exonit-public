"""Function for computing a robust mean estimate in the presence of outliers.

This is a modified Python implementation of this file:
https://idlastro.gsfc.nasa.gov/ftp/pro/robust/resistant_mean.pro
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def robust_mean(y, cut):
    """Computes a robust mean estimate in the presence of outliers.

    Args:
      y: 1D numpy array. Assumed to be normally distributed with outliers.
      cut: Points more than this number of standard deviations from the median are
        ignored.

    Returns:
      mean: A robust estimate of the mean of y.
      mean_stddev: The standard deviation of the mean.
      mask: Boolean array with the same length as y. Values corresponding to
          outliers in y are False. All other values are True.
    """
    absdev = np.abs(y - np.median(y))
    sigma = 1.4826 * np.median(absdev)

    if sigma < 1.0e-24:
        sigma = 1.253 * np.mean(absdev)

    mask = absdev <= cut * sigma

    sigma = np.std(y[mask])

    sc = np.max([cut, 1.0])
    if sc <= 4.5:
        sigma /= (-0.15405 + 0.90723 * sc - 0.23584 * sc ** 2 + 0.020142 * sc ** 3)

    mask = absdev <= cut * sigma

    sigma = np.std(y[mask])

    sc = np.max([cut, 1.0])
    if sc <= 4.5:
        sigma /= (-0.15405 + 0.90723 * sc - 0.23584 * sc ** 2 + 0.020142 * sc ** 3)

    mean = np.mean(y[mask])
    mean_stddev = sigma / np.sqrt(len(y) - 1.0)

    return mean, mean_stddev, mask
