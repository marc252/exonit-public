"""Wotan biweight detrending for the ExoNit worker.

Replaces the previous `lightkurve.flatten(window_length=1001)` which is a
Savitzky-Golay that Hippke+ 2019 (AJ 158, 143) showed eats ~15% of transit
depth on typical sub-Neptune TESS candidates. Biweight with a
3 x T_14 window (default 0.5 d) preserves depth while still killing
rotational harmonics that cause `wrong_period`.

See docs/research/detrending/detrending-wotan-research-report-2026-04-17.md
for the derivation of the defaults.

Backend selection (Phase 4 of wotan-CUDA, see docs/superpowers/specs/
2026-05-14-wotan-cuda-design.md): set ``EXONIT_WOTAN_BACKEND`` to
``cpu`` to force the wotan numba path, ``cuda`` to force the GPU path
(import-time check fails if unavailable), or ``auto`` (default) which
uses CUDA when applicable (biweight, no mask, window_length <= 1.0 d)
and falls back to wotan CPU otherwise. The CUDA kernel is bit-exact
to wotan.flatten (rel <= 4e-15 across 150k thread observations) and
delivers ~19-23x median speedup on consumer GPUs (RTX 5060 Ti / 4070).
"""

import logging
import os
from typing import Optional

import numpy as np

from wotan import flatten as _wotan_flatten


logger = logging.getLogger(__name__)


DEFAULT_WINDOW_DAYS = 0.5      # ~3 x a 4-h transit duration
DEFAULT_EDGE_CUTOFF = 0.5      # drop noisy segment endpoints
DEFAULT_BREAK_TOLERANCE = 0.5  # honour SPOC momentum dumps / sector gaps
DEFAULT_CVAL = 5.0             # Tukey biweight tuning constant

# Largest window_length (days) the CUDA kernel can handle without the
# in-window stack buffer (MAX_W=768) overflowing on TESS 2-min cadence.
# At 720 cadences/day, window=2.0 d -> ~1440 points per window > MAX_W;
# window<=1.0 d gives ~720 <= MAX_W with headroom. Long-period sweep
# (PR #9: 2-8 d windows) stays on CPU per spec § Non-goals.
_CUDA_MAX_WINDOW_DAYS = 1.0


# --- Backend selection at import time ---

_forced = (os.environ.get("EXONIT_WOTAN_BACKEND") or "").strip().lower() or None

_cuda_available = False
_cuda_module = None
try:
    import wotan_cuda as _wotan_cuda
    if _wotan_cuda.is_available():
        _cuda_available = True
        _cuda_module = _wotan_cuda
except Exception as _e:  # pragma: no cover -- diagnostic only
    # warning, not debug: on a CUDA host this silently costs the ~20x
    # detrending speedup and production runs at INFO level.
    logger.warning("wotan_cuda unavailable (CPU wotan): %s", _e)

if _forced == "cuda" and not _cuda_available:
    logger.warning(
        "EXONIT_WOTAN_BACKEND=cuda requested but CUDA not available; "
        "falling back to wotan CPU.")


def _should_use_cuda(window_length: float,
                     transit_mask: Optional[np.ndarray],
                     method: str) -> bool:
    """Decide whether the CUDA kernel applies to this call.

    The kernel only handles plain biweight on contiguous segments with
    no per-point mask and a window the per-thread buffer can hold.
    Everything else falls through to the wotan numba path.
    """
    if _forced == "cpu":
        return False
    if not _cuda_available:
        return False
    if method != "biweight":
        return False
    if transit_mask is not None:
        return False
    if window_length > _CUDA_MAX_WINDOW_DAYS:
        return False
    return True


def get_backend_name() -> str:
    """Return ``cuda`` if the GPU path is active and applicable, else ``cpu``.

    Conservative: returns ``cpu`` whenever a per-call detail would force
    a fallback (mask present, long window, etc.) -- check at call-site
    via _should_use_cuda for the actual per-call decision.
    """
    if _forced == "cpu":
        return "cpu"
    return "cuda" if _cuda_available else "cpu"


def detrend_with_wotan(
    time: np.ndarray,
    flux: np.ndarray,
    window_length: float = DEFAULT_WINDOW_DAYS,
    transit_mask: Optional[np.ndarray] = None,
    method: str = "biweight",
) -> np.ndarray:
    """Return the flattened flux array (same length as input).

    Parameters
    ----------
    time, flux
        Same-length 1-D arrays, time in days.
    window_length
        Sliding-window width in days. 0.5 is the SHERLOCK-validated
        default for sub-Neptune TESS 2-min targets; widen (1.0-2.0) for
        strongly rotating or evolved hosts.
    transit_mask
        Optional boolean array, True where points are in-transit (to be
        ignored by the trend fit). Passing this in the iterative
        refinement pass preserves the full transit depth.
    method
        Wotan method. Default 'biweight'; alternatives: 'median',
        'hspline', 'pspline', 'gp' (slow). See wotan docs.
    """
    # wotan's transit_mask accepts True for in-transit samples; it
    # validates as a numpy array, so we coerce defensively.
    mask = None
    if transit_mask is not None:
        mask = np.asarray(transit_mask, dtype=bool)
        if mask.shape != time.shape:
            raise ValueError(
                f"transit_mask shape {mask.shape} != time shape {time.shape}")

    if _should_use_cuda(window_length, mask, method):
        try:
            flat, _trend = _cuda_module.detrend_biweight(
                time, flux,
                window_length=window_length,
                edge_cutoff=DEFAULT_EDGE_CUTOFF,
                cval=DEFAULT_CVAL,
                break_tolerance=DEFAULT_BREAK_TOLERANCE,
            )
            return flat
        except Exception as e:
            logger.warning(
                "CUDA wotan path failed (%s); falling back to wotan CPU.", e)
            # fall through

    flat, _trend = _wotan_flatten(
        time,
        flux,
        method=method,
        window_length=window_length,
        edge_cutoff=DEFAULT_EDGE_CUTOFF,
        break_tolerance=DEFAULT_BREAK_TOLERANCE,
        cval=DEFAULT_CVAL,
        return_trend=True,
        mask=mask,
    )
    return flat
