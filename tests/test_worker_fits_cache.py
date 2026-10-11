"""Tests for worker.fits_cache_helper — post-download FITS validation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "worker"))

import pytest
from fits_cache_helper import (
    MIN_FITS_BYTES,
    is_valid_fits,
    ensure_fits_present,
)


def test_is_valid_fits_rejects_truncated(tmp_path):
    f = tmp_path / "trunc.fits"
    f.write_bytes(b"\x00" * 1024)  # 1 KB
    ok, reason = is_valid_fits(f)
    assert ok is False
    assert "size" in reason.lower() or "short" in reason.lower()


def test_is_valid_fits_rejects_empty(tmp_path):
    f = tmp_path / "empty.fits"
    f.write_bytes(b"")
    ok, reason = is_valid_fits(f)
    assert ok is False


def test_ensure_fits_present_purges_truncated_and_redownloads(tmp_path):
    """A pre-existing truncated file must be deleted and re-fetched."""
    target = tmp_path / "victim.fits"
    target.write_bytes(b"\x00" * 1024)   # leftover truncated cache entry
    assert target.exists()

    # Fake downloader writes a real-size payload the 1st call (successful).
    calls = []

    def fetcher(dst):
        calls.append(dst)
        # Simulate a valid-size payload. is_valid_fits still requires
        # astropy-parseable content; we use a minimal FITS primary header.
        _write_min_fits(dst)

    ok = ensure_fits_present(target, fetcher, max_retries=2)
    assert ok is True
    assert len(calls) == 1, "fetcher should be invoked once after purging truncated file"
    assert target.stat().st_size >= MIN_FITS_BYTES


def test_ensure_fits_present_reuses_valid_cached(tmp_path):
    """A cached file that is already valid must NOT be re-fetched."""
    target = tmp_path / "good.fits"
    _write_min_fits(target)

    calls = []

    def fetcher(dst):
        calls.append(dst)
        _write_min_fits(dst)

    ok = ensure_fits_present(target, fetcher, max_retries=2)
    assert ok is True
    assert calls == [], "valid cached file must not trigger a download"


def test_ensure_fits_present_retries_on_truncated_download(tmp_path):
    """If the download produces a truncated file, the helper must retry."""
    target = tmp_path / "flaky.fits"
    attempts = {"n": 0}

    def flaky_fetcher(dst):
        attempts["n"] += 1
        if attempts["n"] == 1:
            # First attempt: truncated
            dst.write_bytes(b"\x00" * 1024)
        else:
            _write_min_fits(dst)

    ok = ensure_fits_present(target, flaky_fetcher, max_retries=3)
    assert ok is True
    assert attempts["n"] == 2


def test_ensure_fits_present_gives_up_after_max_retries(tmp_path):
    """If every attempt is truncated, the helper must give up and return False."""
    target = tmp_path / "doomed.fits"
    attempts = {"n": 0}

    def bad_fetcher(dst):
        attempts["n"] += 1
        dst.write_bytes(b"\x00" * 1024)

    ok = ensure_fits_present(target, bad_fetcher, max_retries=3)
    assert ok is False
    assert attempts["n"] == 3
    assert not target.exists(), "final truncated file must be removed"


# --- helpers --------------------------------------------------------------

def _write_min_fits(path):
    """Write a minimal, valid FITS file with a LIGHTCURVE HDU so
    is_valid_fits's astropy parse accepts it."""
    from astropy.io import fits
    import numpy as np

    primary = fits.PrimaryHDU()
    col_time = fits.Column(name="TIME", format="D",
                           array=np.linspace(0.0, 1.0, 128))
    col_flux = fits.Column(name="PDCSAP_FLUX", format="E",
                           array=np.ones(128, dtype=np.float32))
    data_hdu = fits.BinTableHDU.from_columns([col_time, col_flux],
                                             name="LIGHTCURVE")
    hdul = fits.HDUList([primary, data_hdu])
    hdul.writeto(path, overwrite=True)

    # Pad the file past MIN_FITS_BYTES (it will typically already be
    # a few KB — we need it bigger than 100 KB to pass the size heuristic).
    current = path.stat().st_size
    if current < MIN_FITS_BYTES + 1024:
        with open(path, "ab") as f:
            f.write(b"\x00" * (MIN_FITS_BYTES + 2048 - current))
