"""TGLC centroid extraction.

QLP FFI light curves do not ship a centroid; only TGLC does. We pull
the X/Y centroid columns when present, drop NaN/quality-flagged
samples, and median-subtract so the result is zero-centred.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# Common centroid column names across TGLC versions and other FFI products.
_X_CANDIDATES = ("X_CENT", "X_PSF", "MOM_CENTR1", "PSF_CENTR1")
_Y_CANDIDATES = ("Y_CENT", "Y_PSF", "MOM_CENTR2", "PSF_CENTR2")


def find_centroid_columns(column_names) -> Optional[tuple[str, str]]:
    """Return (x_col, y_col) if both centroid columns exist, else None."""
    cols = set(c.upper() for c in column_names)
    x_col = next((c for c in _X_CANDIDATES if c in cols), None)
    y_col = next((c for c in _Y_CANDIDATES if c in cols), None)
    if x_col is None or y_col is None:
        return None
    return x_col, y_col


def extract_centroid_xy(time, x, y, quality) -> dict:
    """Return a dict of clean centroid arrays (median-subtracted).

    Drops samples with non-zero quality, NaN time, or NaN x/y.
    """
    time = np.asarray(time, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    q = np.asarray(quality, dtype=int)

    good = (
        (q == 0)
        & np.isfinite(time)
        & np.isfinite(x)
        & np.isfinite(y)
    )
    t = time[good]
    if t.size == 0:
        return {"time": t, "x": np.array([]), "y": np.array([])}
    xc = x[good] - np.median(x[good])
    yc = y[good] - np.median(y[good])
    return {"time": t, "x": xc, "y": yc}
