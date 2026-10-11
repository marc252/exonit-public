"""Fast unit tests for ``worker.exominer_native.preprocess.lc_io`` (no Docker)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from worker.exominer_native.preprocess.lc_io import (
    read_tess_light_curve,
    read_tess_light_curve_from_bytes,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE = REPO_ROOT / "tests/fixtures/exominer_native/tic_143143769_s0032.lc.fits"


EXPECTED_KEYS = {
    "all_time", "all_flux", "all_flux_err",
    "all_centroids", "all_centroids_px",
    "flag_keep", "sectors", "module", "camera", "ccd", "target_position",
}


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"fixture missing: {FIXTURE}")
def test_reads_single_sector_fits():
    data, files_not_read = read_tess_light_curve([FIXTURE])

    assert files_not_read == []
    assert set(data.keys()) == EXPECTED_KEYS
    assert len(data["all_time"]) == 1
    assert len(data["all_flux"]) == 1
    assert data["sectors"] == [32]
    assert len(data["target_position"]) == 2


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"fixture missing: {FIXTURE}")
def test_time_and_flux_aligned_after_nan_filter():
    data, _ = read_tess_light_curve([FIXTURE])
    t = data["all_time"][0]
    f = data["all_flux"][0]
    assert isinstance(t, np.ndarray) and isinstance(f, np.ndarray)
    assert t.shape == f.shape
    # PDCSAP_FLUX NaNs are meant to be stripped
    assert not np.isnan(f).any()


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"fixture missing: {FIXTURE}")
def test_read_from_bytes_matches_read_from_path():
    data_path, _ = read_tess_light_curve([FIXTURE])
    data_bytes, _ = read_tess_light_curve_from_bytes(FIXTURE.read_bytes())

    assert set(data_path) == set(data_bytes)
    for k in ("all_time", "all_flux", "all_flux_err"):
        np.testing.assert_array_equal(data_path[k][0], data_bytes[k][0])
    assert data_path["sectors"] == data_bytes["sectors"]
    assert data_path["camera"] == data_bytes["camera"]
    assert data_path["ccd"] == data_bytes["ccd"]


def test_interpolate_missing_time_raises_not_implemented(tmp_path):
    # Call on a nonexistent path to avoid needing the fixture — the
    # NotImplementedError must fire before we reach file I/O when the reader
    # encounters a real fits file that triggers the branch; using a path
    # that fails open just confirms the except-branch path, so we instead
    # exercise the feature flag via a minimal synthetic FITS we build here.
    pytest.importorskip("astropy")
    from astropy.io import fits

    # Build a 1-HDU FITS with LIGHTCURVE table carrying a NaN in TIME.
    col_time = fits.Column(name="TIME", format="E", array=np.array([0.0, np.nan, 2.0], dtype=np.float32))
    col_flux = fits.Column(name="PDCSAP_FLUX", format="E", array=np.array([1.0, 1.0, 1.0], dtype=np.float32))
    col_err = fits.Column(name="PDCSAP_FLUX_ERR", format="E", array=np.array([0.01, 0.01, 0.01], dtype=np.float32))
    col_c1 = fits.Column(name="MOM_CENTR1", format="E", array=np.zeros(3, dtype=np.float32))
    col_c2 = fits.Column(name="MOM_CENTR2", format="E", array=np.zeros(3, dtype=np.float32))
    col_p1 = fits.Column(name="POS_CORR1", format="E", array=np.zeros(3, dtype=np.float32))
    col_p2 = fits.Column(name="POS_CORR2", format="E", array=np.zeros(3, dtype=np.float32))
    col_q = fits.Column(name="QUALITY", format="J", array=np.zeros(3, dtype=np.int32))
    col_cad = fits.Column(name="CADENCENO", format="J", array=np.arange(3, dtype=np.int32))
    tbl = fits.BinTableHDU.from_columns(
        [col_time, col_flux, col_err, col_c1, col_c2, col_p1, col_p2, col_q, col_cad],
        name="LIGHTCURVE",
    )
    primary = fits.PrimaryHDU()
    primary.header["CAMERA"] = 1
    primary.header["CCD"] = 1
    primary.header["SECTOR"] = 32
    primary.header["RA_OBJ"] = 0.0
    primary.header["DEC_OBJ"] = 0.0
    fits_path = tmp_path / "synthetic.fits"
    fits.HDUList([primary, tbl]).writeto(fits_path)

    with pytest.raises(NotImplementedError):
        read_tess_light_curve([fits_path], interpolate_missing_time=True)
