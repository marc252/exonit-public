"""Worker-side FITS cache helper.

Validates FITS files post-download and retries on truncated / corrupt
responses, so a flaky MAST fetch can't poison the local cache with a
zero-byte or partial file that would make every subsequent attempt fail.

Shipped to every worker via the server's /api/worker/update manifest.
"""

from pathlib import Path


MIN_FITS_BYTES = 30_000   # 30 KB — smallest plausible TESS LC file
# NB: single-sector QLP FFI light curves are ~63-90 KB (≈1000-1300 pts), so the
# old 100 KB threshold wrongly rejected ~33% of valid sectors. Real download
# failures are 0-byte / HTML error pages (<10 KB), comfortably below 30 KB.


def is_valid_fits(path):
    """Return (ok: bool, reason: str).

    Mirrors exonit.fits_validator.is_valid_fits so the worker can run
    without importing the exonit package (which remote workers don't have).
    """
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


def ensure_fits_present(local_path, fetcher, max_retries=3, logger=None):
    """Ensure local_path holds a valid FITS file, downloading via fetcher if needed.

    Parameters
    ----------
    local_path : Path or str
        Target file.
    fetcher : callable(Path) -> None
        Callback that writes the FITS bytes to the supplied path. May raise;
        exceptions are caught and trigger a retry.
    max_retries : int
        Total attempts (including the first). A purge-of-cached-truncated
        followed by one fresh download counts as one attempt.
    logger : optional
        Logger-like object with .warning / .info methods. If None, no logs.

    Returns
    -------
    bool
        True if a valid FITS file is at local_path. False if all attempts
        failed (the file is removed on failure so callers don't see a
        truncated leftover).
    """
    local_path = Path(local_path)

    # Step 1: if a cached file exists, trust it only after validation.
    if local_path.exists():
        ok, reason = is_valid_fits(local_path)
        if ok:
            return True
        if logger:
            logger.warning(
                f"Purging invalid cached FITS {local_path.name}: {reason}")
        try:
            local_path.unlink()
        except OSError:
            pass

    # Step 2: download-and-validate loop.
    for attempt in range(1, max_retries + 1):
        try:
            fetcher(local_path)
        except Exception as exc:
            if logger:
                logger.warning(
                    f"Download attempt {attempt}/{max_retries} for "
                    f"{local_path.name} raised {type(exc).__name__}: {exc}")
            # Remove any partial write before retrying.
            if local_path.exists():
                try:
                    local_path.unlink()
                except OSError:
                    pass
            continue

        ok, reason = is_valid_fits(local_path)
        if ok:
            if logger and attempt > 1:
                logger.info(
                    f"Downloaded {local_path.name} after {attempt} attempts")
            return True

        if logger:
            logger.warning(
                f"Invalid FITS after download "
                f"(attempt {attempt}/{max_retries}) "
                f"{local_path.name}: {reason}")
        try:
            local_path.unlink()
        except OSError:
            pass

    # All attempts exhausted.
    if logger:
        logger.warning(
            f"Giving up on {local_path.name} after {max_retries} attempts")
    return False
