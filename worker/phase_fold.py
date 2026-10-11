"""Shared phase-fold formula for the ExoNit worker.

Consolidates independently hand-rolled phase-fold implementations that had
accumulated slightly different centering conventions across worker/cnn.py,
worker/vetting.py, worker/injection.py, worker/fpp_native.py and
worker/cross_source.py — see docs/reviews/2026-07-02-opus48-cleanup-review.md
item D5 for the catalog and progress/impl-c1-phasefold-2026-07-02.md for the
per-site proof of exact equivalence.

SCIENCE PATH: this touches scored output. Every call site that now uses this
module is proven bit-identical (``np.array_equal``, NOT ``np.allclose``) to
the inline formula it replaced by the golden tests in
tests/worker/test_phase_fold_golden.py. A site is only migrated here if that
proof holds; otherwise it keeps its own inline formula (documented in the
handoff).

Two independent axes describe every convention found in the codebase.
Floating-point division and modulo do NOT commute exactly, so these are
genuinely different bit patterns, not just readability variants — do not
"simplify" one into the other.

``fold_order`` — order of operations for the raw, wraparound-in-``[0, 1)``
fold:
    "mod_div"  ->  ``((time - t0) % period) / period``   (default)
    "div_mod"  ->  ``((time - t0) / period) % 1.0``

``center`` — how (if at all) the raw ``[0, 1)`` fold is re-centered on the
transit:
    ``None``  -> no centering; phase stays in ``[0, 1)``.
    "mask"    -> boundary-mask centering applied on top of the
                 ``fold_order`` fold: ``phase[phase > 0.5] -= 1.0``.
                 Range ``(-0.5, 0.5]`` — closed at +0.5, open at -0.5.
    "shift"   -> fused pre/post-shift centering. Always divides first
                 (mirrors the single call site that uses it,
                 ``cross_source.fixed_period_box_fit``); ``fold_order`` is
                 ignored in this mode:
                     ``((time - t0) / period + 0.5) % 1.0 - 0.5``
                 Range ``[-0.5, 0.5)`` — closed at -0.5, open at +0.5.
                 Note this is the OPPOSITE boundary parity from "mask".

No mode guards against ``period <= 0``: dividing/modulo by a non-positive
period raises a numpy ``RuntimeWarning`` and yields ``nan``, exactly as every
un-guarded inline call site behaved before this refactor. Callers that need
a guard (e.g. ``cross_source.fixed_period_box_fit``'s early return on
``period <= 0``) keep it at the call site, unchanged.
"""

import numpy as np

__all__ = ["phase_fold"]


def phase_fold(time, t0, period, *, fold_order="mod_div", center=None):
    """Phase-fold ``time`` on (``t0``, ``period``).

    Args:
        time: array-like of BTJD times.
        t0: transit epoch (same time system as ``time``).
        period: fold period. Must be > 0 for a meaningful result — not
            enforced here (see module docstring).
        fold_order: "mod_div" (default) or "div_mod" — see module docstring.
            Ignored when ``center == "shift"``.
        center: ``None`` (default, raw ``[0, 1)``), "mask" (``(-0.5, 0.5]``)
            or "shift" (``[-0.5, 0.5)``) — see module docstring.

    Returns:
        A new ndarray (never a view into ``time``) with the folded phase.
    """
    if center == "shift":
        # Fused formula — deliberately NOT built from the fold_order branch
        # below: dividing, shifting and taking the modulo in one expression
        # produces different float rounding than folding first and shifting
        # after. This is the only site (cross_source.py) that ever used this
        # convention, and it always divides first.
        return ((time - t0) / period + 0.5) % 1.0 - 0.5

    if fold_order == "mod_div":
        phase = ((time - t0) % period) / period
    elif fold_order == "div_mod":
        phase = ((time - t0) / period) % 1.0
    else:
        raise ValueError(f"phase_fold: unknown fold_order {fold_order!r}")

    if center is None:
        return phase
    if center == "mask":
        phase[phase > 0.5] -= 1.0
        return phase
    raise ValueError(f"phase_fold: unknown center {center!r}")
