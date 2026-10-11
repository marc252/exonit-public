"""Validate a TESS FITS light-curve file.

Two-stage check:
1. Size heuristic — TESS sector FITS files are ~1-2 MB. Anything under
   MIN_FITS_BYTES is almost certainly truncated.
2. Structural parse — open with astropy and read the first data HDU header.
   This catches the "advertised size != actual size" truncation that
   astropy reports with the warning we saw on <host>.
"""

from pathlib import Path


MIN_FITS_BYTES = 30_000   # 30 KB — smallest plausible TESS LC file
# NB: single-sector QLP FFI light curves are ~63-90 KB, so the old 100 KB
# threshold wrongly rejected ~33% of valid sectors. Real download failures are
# 0-byte / HTML error pages (<10 KB), comfortably below 30 KB.


def is_valid_fits(path):
    """Return (ok: bool, reason: str)."""
    p = Path(path)
    if not p.exists():
        return False, "file does not exist"
    size = p.stat().st_size
    if size == 0:
        return False, "empty file"
    if size < MIN_FITS_BYTES:
        return False, f"size too short ({size} bytes < {MIN_FITS_BYTES})"

    try:
        from astropy.io import fits
    except ImportError:
        return False, "astropy not installed"

    try:
        with fits.open(p, memmap=False) as hdul:
            _ = hdul[0].header
            if len(hdul) < 2:
                return False, "missing data HDU"
            _ = hdul[1].header
            data = hdul[1].data
            if data is None or len(data) == 0:
                return False, "empty data HDU"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    return True, "ok"
