"""Resolve (period, t0, duration) for a dataset row according to its type.

See docs/superpowers/specs/2026-05-04-cnn-ffi-training-design.md §3.1.

Note: this module operates on an internal abstract column schema
(`tid`, `pl_orbper`, `pl_tranmid`, `pl_trandurh`). The CLI layer
(scripts/build_ffi_views.py:cmd_compute) is responsible for mapping
real TOI catalog columns ("TIC ID", "Period (days)", "Epoch (BJD)",
"Duration (hours)") to this schema and converting BJD → BTJD before
passing the catalog path here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class PeriodSource(str, Enum):
    TOI = "toi"
    SYNTHETIC = "synthetic"
    BLS = "bls"
    EXPLICIT = "explicit"  # period/t0/duration supplied directly on the row


@dataclass
class PeriodResult:
    period: float
    t0: float
    duration_days: float
    source: PeriodSource
    snr: Optional[float] = None  # populated for BLS source


_TOI_CACHE: dict = {}


def _load_toi(path: Path) -> pd.DataFrame:
    p = str(path)
    if p not in _TOI_CACHE:
        _TOI_CACHE[p] = pd.read_csv(path)
    return _TOI_CACHE[p]


class PeriodNotFound(LookupError):
    """Raised when no period can be resolved for the row."""


def _is_finite_pos(x) -> bool:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return False
    return np.isfinite(x) and x > 0


def resolve_period(row: pd.Series, time, flux, toi_catalog_path: Optional[Path]) -> PeriodResult:
    # Explicit period takes precedence over every other source. This is used for
    # ground-truth rows added with a known ephemeris (e.g. T16 sector-1 positives
    # via P_med/Tc_med, or confirmed artefact hard-negatives carrying the BLS
    # period of the family they belong to). It is orthogonal to the
    # synthetic/negative flags: a real-LC negative can still have a known period.
    if _is_finite_pos(row.get("period_explicit")):
        period = float(row["period_explicit"])
        t0 = float(row.get("t0_explicit", 0.0))
        dur_days = row.get("duration_explicit_days")
        if not _is_finite_pos(dur_days):
            # Fall back to a small fixed fraction of the period when no duration
            # is supplied; the local-view window is derived from this.
            dur_days = max(0.02, 0.05 * period)
        return PeriodResult(period, t0, float(dur_days), PeriodSource.EXPLICIT)

    if row.get("is_synthetic_row", False):
        period = float(row["inj_period_days"])
        t0 = float(row["inj_t0_btjd"])
        duration = float(row["inj_duration_hours"]) / 24.0
        return PeriodResult(period, t0, duration, PeriodSource.SYNTHETIC)

    if not row.get("is_negative_row", False):
        # Real positive: TOI catalog lookup
        if toi_catalog_path is None:
            raise PeriodNotFound("toi_catalog_path required for real positives")
        df = _load_toi(toi_catalog_path)
        match = df[df["tid"] == row["tic_id"]]
        if match.empty or match["pl_orbper"].isna().all():
            raise PeriodNotFound(f"TIC {row['tic_id']} not in TOI catalog")
        first = match.iloc[0]
        return PeriodResult(
            period=float(first["pl_orbper"]),
            t0=float(first["pl_tranmid"]),
            duration_days=float(first["pl_trandurh"]) / 24.0,
            source=PeriodSource.TOI,
        )

    # Negative: BLS detection
    return _resolve_via_bls(time, flux)


def _resolve_via_bls(time, flux):
    """BLS detect best candidate; require SNR>5 else PeriodNotFound."""
    if time is None or flux is None:
        raise PeriodNotFound("BLS needs time+flux for negative rows")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "worker"))
    import bls_fast
    time = np.ascontiguousarray(time, dtype=np.float64)
    flux = np.ascontiguousarray(flux, dtype=np.float64)
    finite = np.isfinite(time) & np.isfinite(flux)
    if finite.sum() < 1000:
        raise PeriodNotFound("too few finite points for BLS")
    result = bls_fast.bls_fast(
        time[finite], flux[finite],
        period_min=0.5, period_max=15.0,
        n_periods=10000, n_dur_steps=8,
    )
    snr = float(result["snr"])
    if snr < 5.0:
        raise PeriodNotFound(f"BLS SNR={snr:.2f} below threshold 5.0")
    return PeriodResult(
        period=float(result["period"]),
        t0=float(result["t0"]),
        duration_days=float(result["duration"]),
        source=PeriodSource.BLS,
        snr=snr,
    )
