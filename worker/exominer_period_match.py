"""Shared period-alias matching for ExoMiner TCE↔BLS candidate pairing.

Single source of truth for the worker side — imported by both
``exominer_native/hook.py`` (native ONNX path) and ``exominer_client.py``
(legacy TF/server-tfrecord path), which used to carry drifting copies.
The server keeps its own mirror in ``exonit/exominer_service.py`` (the
worker directory is deployed standalone, so it cannot import ``exonit``);
change BOTH if you touch the factors or tolerance.

3x and 4x aliases (plus reciprocals) were added 2026-05-16 after the
ExoMiner backfill was found marking strong candidates (SNR>30, CNN>=0.9)
as -1 because the BLS-detected period was an Nx alias of the ExoMiner
pipeline's TCE period. Observed: TIC 300871545 (BLS 1.61d vs TCE 4.82d,
ratio 3x) and TIC 388104525 (BLS 9.999d vs TCE 2.500d, ratio 4x). With
tol=2% the false-match risk from these extra factors is low: a true match
at one factor still wins on minimum relative error.
"""
from __future__ import annotations

from typing import Any, List, Optional

PERIOD_TOL = 0.02
PERIOD_FACTORS = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0, 4.0, 0.25)


def match_period(
    tces: List[Any],
    target_period: Optional[float],
    tol: float = PERIOD_TOL,
) -> Optional[Any]:
    """Return the TCE whose period best matches ``target_period`` within ``tol``.

    Accepts both the legacy dict shape (``{"period": ...}``) and the native
    ``TCEFeatures`` dataclass (``.period``). Tries every alias in
    ``PERIOD_FACTORS`` and picks the minimum-relative-error winner.
    """
    if not tces or target_period is None:
        return None
    try:
        tgt = float(target_period)
    except (TypeError, ValueError):
        return None
    if tgt <= 0:
        return None

    best = None
    best_err = float("inf")
    for tce in tces:
        if isinstance(tce, dict):
            p2 = float(tce.get("period", 0) or 0)
        else:
            p2 = float(getattr(tce, "period", 0) or 0)
        if p2 <= 0:
            continue
        for k in PERIOD_FACTORS:
            denom = max(abs(tgt), abs(k * p2))
            if denom == 0:
                continue
            err = abs(tgt - k * p2) / denom
            if err < tol and err < best_err:
                best = tce
                best_err = err
    return best
