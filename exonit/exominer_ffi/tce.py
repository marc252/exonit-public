"""TCE record dataclass + builder from a manifest row."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TCERecord:
    tic_id: str
    sector: int
    period_d: float
    t0_d: float
    duration_d: float
    depth_ppm: float
    snr: float
    source: str          # "toi" | "synthetic" | "bls"
    label: str
    has_centroid: bool


def build_tce_record(row, toi_catalog_path: Optional[Path] = None) -> Optional[TCERecord]:
    """Build a TCE record from a row of ffi_dataset_v1.parquet manifest.

    Synthetic rows: use inj_* fields directly.
    Real rows: must have a period from the TOI catalog (passed via the
    master script when a light curve is also available). When neither
    is present, return None — the dataset builder skips these rows.
    """
    is_synth = bool(row.get("is_synthetic_row"))
    if is_synth:
        period = row.get("inj_period_days")
        try:
            period = float(period) if period is not None else None
        except (TypeError, ValueError):
            period = None
        if period is None or not np.isfinite(period) or period <= 0:
            return None
        duration_h = row.get("inj_duration_hours")
        try:
            duration_h = float(duration_h) if duration_h is not None else 4.0
        except (TypeError, ValueError):
            duration_h = 4.0
        return TCERecord(
            tic_id=str(row["tic_id"]),
            sector=int(row.get("sector") or 0),
            period_d=period,
            t0_d=float(row.get("inj_t0_btjd") or 0.0),
            duration_d=duration_h / 24.0,
            depth_ppm=float(row.get("inj_depth_ppm") or 0.0),
            snr=10.0,  # Conservative default; real SNR comes at view time.
            source="synthetic",
            label=str(row.get("label") or ""),
            has_centroid=str(row.get("source", "")).lower() == "tglc",
        )

    # Real row: defer period resolution to the master script (which has
    # access to light curves and the TOI catalog).
    return None
