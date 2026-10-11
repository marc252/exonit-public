"""Physical plausibility gates on a transit signal, given the HOST STAR.

The CNN sees a light curve, not a star. An 0.5% eclipse looks the same to it
whether the host is a 1 R☉ dwarf (→ a Jupiter) or a 15 R☉ red giant (→ an
object the size of the Sun, i.e. an eclipsing binary). The depth alone is
meaningless; depth + R* is not:

    depth = (Rp / R*)²   →   Rp = R* · √depth

Measured on the sector-14 lot 20260714T175216 (2026-07-14): of the 7.348
candidates that pass every existing filter (SNR ≥ 10, CNN ≥ 0.7 or FFI ≥ 0.5)
AND have a computable FPP, **91% imply a companion larger than 2 R_Jup** — a
median implied radius of 7.1 R_Jup, which is a star, not a planet. They are
eclipsing binaries and blends, and TRICERATOPS was being spent on them.

THE THRESHOLD IS NOT ARBITRARY. Measured against the 1.301 known/confirmed
planets (KP/CP) of the TOI catalogue that carry both a stellar radius and a
depth:

    implied Rp: median 0.89 R_Jup, 99th pct 2.01, MAXIMUM 2.20 R_Jup
      cut at 2.0 R_Jup → loses 14 real planets (1.1%)   ← unacceptable
      cut at 2.5 R_Jup → loses 0
      cut at 3.0 R_Jup → loses 0, with margin           ← default

3.0 is the default rather than 2.5 because OUR depth is not the catalogue's:
BLS is known to inflate depth on our own artefacts (see the finalist-FP
analysis), so an inflated depth inflates the implied radius too. At 3.0 the
BLS may over-report a depth by up to 86% ((3.0/2.2)² = 1.86) and the most
extreme known planet would still survive the gate. Recall is not negotiable;
throughput is.

FAIL-OPEN, ALWAYS: no radius, no depth, nonsense values → NOT gated. A missing
catalogue entry must never be read as "this is a binary".
"""
from __future__ import annotations

import math
import os

#: 1 R☉ in Jupiter radii (IAU nominal: 6.957e8 m / 6.9911e7 m).
RSUN_IN_RJUP = 9.9513

#: Above this implied companion radius the signal cannot be a planet.
#: See the module docstring for the measurement behind the default.
DEFAULT_MAX_COMPANION_RJUP = 3.0


def max_companion_rjup() -> float:
    """The active threshold (env ``EXONIT_FPP_RP_MAX_RJUP`` overrides)."""
    try:
        return float(os.environ.get("EXONIT_FPP_RP_MAX_RJUP",
                                    DEFAULT_MAX_COMPANION_RJUP))
    except (TypeError, ValueError):
        return DEFAULT_MAX_COMPANION_RJUP


def _finite_positive(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f <= 0:
        return None
    return f


def implied_companion_rjup(r_star_rsun, depth_ppm) -> float | None:
    """Radius (R_Jup) of the object that would produce *depth_ppm* on a star of
    *r_star_rsun*. None when either input is unusable — the caller must treat
    that as "unknown", never as "large"."""
    r_star = _finite_positive(r_star_rsun)
    depth = _finite_positive(depth_ppm)
    if r_star is None or depth is None:
        return None
    return r_star * math.sqrt(depth / 1e6) * RSUN_IN_RJUP


def implies_stellar_companion(r_star_rsun, depth_ppm, max_rjup=None) -> bool:
    """True when this transit is too deep, for this star, to be a planet.

    Fail-open: unknown radius or unknown depth returns False (not gated).
    """
    rp = implied_companion_rjup(r_star_rsun, depth_ppm)
    if rp is None:
        return False
    return rp > (max_companion_rjup() if max_rjup is None else max_rjup)


def max_planetary_depth_ppm(r_star_rsun, max_rjup=None) -> float | None:
    """The deepest transit a star of *r_star_rsun* could show and still be
    planetary — the same rule as implies_stellar_companion, inverted so it can
    be pushed into SQL as a single per-star depth bound (that is what lets the
    eager gate mark a whole TIC's candidates in ONE statement instead of
    fetching every row into Python).

    None when the radius is unusable → the caller must gate nothing.
    """
    r_star = _finite_positive(r_star_rsun)
    if r_star is None:
        return None
    limit = max_companion_rjup() if max_rjup is None else max_rjup
    return ((limit / RSUN_IN_RJUP) / r_star) ** 2 * 1e6
