"""Unit tests for exonit.exominer_ffi.preprocess."""
from __future__ import annotations

import numpy as np
import pytest


# =============================================================================
# em_view_spec
# =============================================================================

def test_em_view_spec_lengths():
    from exonit.exominer_ffi.preprocess import em_view_spec
    from exonit.cnn_ffi_views import GLOBAL_BINS, LOCAL_BINS

    spec = em_view_spec()
    # Mirror Phase 2's cnn_ffi_views sizes (tuned for FFI cadence)
    assert spec["global_view_len"] == GLOBAL_BINS
    assert spec["local_view_len"] == LOCAL_BINS
    assert spec["centroid_view_len"] == LOCAL_BINS
    assert spec["stellar_n_features"] == 5  # Tmag, Teff, logg, R*, M*
    assert spec["tce_n_features"] == 4      # period, depth_norm, duration_norm, snr


def test_em_view_spec_has_mask_flag():
    from exonit.exominer_ffi.preprocess import em_view_spec

    spec = em_view_spec()
    assert "centroid_mask" in spec["mask_flags"]


# =============================================================================
# make_em_record
# =============================================================================

def test_make_em_record_qlp_no_centroid():
    """QLP rows: centroid_view is zeros and centroid_mask=0."""
    from exonit.exominer_ffi.preprocess import make_em_record

    t = np.linspace(0, 27, 5400)
    flux = np.ones_like(t)
    in_transit = ((t - 2.5) % 5.0 < 0.1) | ((5.0 - ((t - 2.5) % 5.0)) < 0.1)
    flux[in_transit] -= 0.01

    record = make_em_record(
        time=t, flux=flux,
        tce_period_d=5.0, tce_t0_d=2.5,
        tce_duration_d=0.2, tce_depth_ppm=10000.0, tce_snr=20.0,
        stellar_tmag=12.0, stellar_teff=5500.0, stellar_logg=4.4,
        stellar_rstar=1.0, stellar_mstar=1.0,
        centroid=None,
    )
    from exonit.cnn_ffi_views import GLOBAL_BINS, LOCAL_BINS
    assert record["global_view"].shape == (GLOBAL_BINS,)
    assert record["local_view"].shape == (LOCAL_BINS,)
    assert record["centroid_view"].shape == (LOCAL_BINS,)
    assert np.allclose(record["centroid_view"], 0.0)
    assert record["centroid_mask"] == 0
    assert record["stellar"].shape == (5,)
    assert record["tce"].shape == (4,)


def test_make_em_record_tglc_with_centroid():
    """TGLC rows with a transit-correlated centroid kick → mask=1."""
    from exonit.exominer_ffi.preprocess import make_em_record

    t = np.linspace(0, 27, 5400)
    flux = np.ones_like(t)

    # Centroid kicks only during transit (typical EB-like signature)
    cen_t = np.linspace(0, 27, 5400)
    cen_x = np.zeros_like(cen_t)
    cen_y = np.zeros_like(cen_t)
    in_transit = ((cen_t - 2.5) % 5.0 < 0.1) | ((5.0 - ((cen_t - 2.5) % 5.0)) < 0.1)
    cen_x[in_transit] = 0.05  # kick to the right during transit
    cen_y[in_transit] = 0.03

    record = make_em_record(
        time=t, flux=flux,
        tce_period_d=5.0, tce_t0_d=2.5,
        tce_duration_d=0.2, tce_depth_ppm=10000.0, tce_snr=20.0,
        stellar_tmag=12.0, stellar_teff=5500.0, stellar_logg=4.4,
        stellar_rstar=1.0, stellar_mstar=1.0,
        centroid={"time": cen_t, "x": cen_x, "y": cen_y},
    )
    assert record["centroid_mask"] == 1
    assert not np.allclose(record["centroid_view"], 0.0)


# =============================================================================
# tce.py builder
# =============================================================================

def test_tce_record_from_row_synthetic():
    """A synthetic-injection row builds a TCE from the inj_* fields."""
    import pandas as pd
    from exonit.exominer_ffi.tce import build_tce_record

    row = pd.Series({
        "tic_id": "100014359",
        "source": "qlp",
        "sector": 2,
        "label": "synthetic_pos",
        "is_synthetic_row": True,
        "inj_period_days": 4.5,
        "inj_t0_btjd": 2458500.5,
        "inj_duration_hours": 4.0,
        "inj_depth_ppm": 5000.0,
        "inj_impact_param": 0.3,
        "teff_k": 5500.0,
        "logg": 4.4,
        "rstar_solar": 1.0,
        "tmag": 12.0,
    })

    rec = build_tce_record(row, toi_catalog_path=None)
    assert rec is not None
    assert rec.period_d == pytest.approx(4.5)
    assert rec.duration_d == pytest.approx(4.0 / 24.0)
    assert rec.depth_ppm == 5000.0
    assert rec.source == "synthetic"
    assert rec.label == "synthetic_pos"


def test_tce_record_real_row_returns_none_when_no_period():
    """A real row without a period returns None — dataset builder skips."""
    import pandas as pd
    from exonit.exominer_ffi.tce import build_tce_record

    row = pd.Series({
        "tic_id": "999999",
        "source": "qlp",
        "sector": 2,
        "label": "PC",
        "is_synthetic_row": False,
        "inj_period_days": None,
    })
    rec = build_tce_record(row, toi_catalog_path=None)
    assert rec is None
