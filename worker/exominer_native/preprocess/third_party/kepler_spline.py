"""Functions for computing normalization splines for Kepler light curves.

Vendored from ``src_preprocessing/third_party/kepler_spline/kepler_spline.py``.
Only patch vs the oracle: the import path for ``robust_mean`` is relative
to the sibling module in this package (oracle: absolute
``src_preprocessing.third_party.robust_mean``).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import numpy as np
from pydl.pydlutils import bspline

from . import robust_mean


class InsufficientPointsError(Exception):
    """Indicates that insufficient points were available for spline fitting."""
    pass


class SplineError(Exception):
    """Indicates an error in the underlying spline-fitting implementation."""
    pass


def kepler_spline(time, flux, bkspace=1.5, maxiter=5, outlier_cut=3):
    """Computes a best-fit spline curve for a light curve segment."""
    if len(time) < 4:
        raise InsufficientPointsError(
            "Cannot fit a spline on less than 4 points. Got {} points.".format(len(time))
        )

    t_min = np.min(time)
    t_max = np.max(time)
    time = (time - t_min) / (t_max - t_min)
    bkspace /= (t_max - t_min)

    spline = None
    mask = None

    for _ in range(maxiter):
        if spline is None:
            mask = np.ones_like(time, dtype=np.bool_)
        else:
            residuals = flux - spline
            new_mask = robust_mean.robust_mean(residuals, cut=outlier_cut)[2]

            if np.all(new_mask == mask):
                break

            mask = new_mask

        if np.sum(mask) < 4:
            raise InsufficientPointsError(
                "Cannot fit a spline on less than 4 points. After removing "
                "outliers, got {} points.".format(np.sum(mask))
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                curve = bspline.iterfit(time[mask], flux[mask], bkspace=bkspace)[0]

            spline = curve.value(time)[0]
        except (IndexError, TypeError, ValueError) as e:
            raise SplineError(
                "Fitting spline failed with error: '{}'. This might be caused by the "
                "breakpoint spacing being too small, and/or there being insufficient "
                "points to fit the spline in one of the intervals.".format(e)
            )

    return spline, mask


class SplineMetadata(object):
    """Metadata about a spline fit."""

    def __init__(self):
        self.light_curve_mask = None
        self.bkspace = None
        self.bad_bkspaces = []
        self.likelihood_term = None
        self.penalty_term = None
        self.bic = None


def choose_kepler_spline(
    all_time,
    all_flux,
    bkspaces,
    maxiter=5,
    penalty_coeff=1.0,
    outlier_cut=3,
    verbose=True,
):
    """Computes the best-fit Kepler spline across break-point spacings."""
    best_spline = None
    metadata = SplineMetadata()

    scaled_diffs = [np.diff(f) / np.sqrt(2) for f in all_flux]
    scaled_diffs = np.concatenate(scaled_diffs) if scaled_diffs else np.array([])
    if not scaled_diffs.size:
        best_spline = [np.array([np.nan] * len(f)) for f in all_flux]
        metadata.light_curve_mask = [np.zeros_like(f, dtype=np.bool_) for f in all_flux]
        return best_spline, metadata

    sigma = np.median(np.abs(scaled_diffs)) * 1.48

    for bkspace in bkspaces:
        nparams = 0
        npoints = 0
        ssr = 0

        spline = []
        light_curve_mask = []
        bad_bkspace = False
        for time, flux in zip(all_time, all_flux):
            try:
                spline_piece, mask = kepler_spline(
                    time, flux, bkspace=bkspace, maxiter=maxiter, outlier_cut=outlier_cut
                )
            except InsufficientPointsError as e:
                if verbose:
                    warnings.warn(str(e))
                spline.append(np.array([np.nan] * len(flux)))
                light_curve_mask.append(np.zeros_like(flux, dtype=np.bool_))
                continue
            except SplineError as e:
                if verbose:
                    warnings.warn("Bad bkspace {}: {}".format(bkspace, e))
                metadata.bad_bkspaces.append(bkspace)
                bad_bkspace = True
                break

            spline.append(spline_piece)
            light_curve_mask.append(mask)

            total_time = np.max(time) - np.min(time)
            nknots = int(total_time / bkspace) + 1
            nparams += nknots + 3 - 1

            npoints += np.sum(mask)
            ssr += np.sum((flux[mask] - spline_piece[mask]) ** 2)

        if bad_bkspace or not npoints:
            continue

        likelihood_term = npoints * np.log(2 * np.pi * sigma ** 2) + ssr / sigma ** 2

        penalty_term = nparams * np.log(npoints)

        bic = likelihood_term + penalty_coeff * penalty_term

        if best_spline is None or bic < metadata.bic:
            best_spline = spline
            metadata.light_curve_mask = light_curve_mask
            metadata.bkspace = bkspace
            metadata.likelihood_term = likelihood_term
            metadata.penalty_term = penalty_term
            metadata.bic = bic

    if best_spline is None:
        best_spline = [np.array([np.nan] * len(f)) for f in all_flux]
        metadata.light_curve_mask = [np.zeros_like(f, dtype=np.bool_) for f in all_flux]

    return best_spline, metadata


def fit_kepler_spline(
    all_time,
    all_flux,
    bkspace_min=0.5,
    bkspace_max=20,
    bkspace_num=20,
    maxiter=5,
    penalty_coeff=1.0,
    outlier_cut=3,
    verbose=True,
):
    """Fits a Kepler spline with logarithmically-sampled breakpoint spacings."""
    bkspaces = np.logspace(np.log10(bkspace_min), np.log10(bkspace_max), num=bkspace_num)

    return choose_kepler_spline(
        all_time,
        all_flux,
        bkspaces,
        maxiter=maxiter,
        penalty_coeff=penalty_coeff,
        outlier_cut=outlier_cut,
        verbose=verbose,
    )
