"""Conditional gate: route a TIC to ExoMiner++ (FFI) or to the fallback.

The decision rule (CLAUDE.md rule #3 — only score where the model's inputs
genuinely exist):

  * A TIC with SPOC DV available for a TESS sector >= 36 (the SPOC-FFI DV
    domain where ExoMiner++'s difference-image / neighbor inputs exist) →
    score with ExoMiner++ 2.0 and write ``exominer_pp_score`` +
    ``exominer_pp_model_version``.
  * Otherwise (TGLC-only / s < 36 / no DV) → **skip** ExoMiner++ and fall
    back to ExoNit's own ExoMiner-FFI flux model (``exominer_ffi_score``),
    which is the current discovery behaviour. No regression for the no-DV
    path: the gate returns ``use_exominer_pp=False`` and the caller leaves
    the existing flow untouched.

This module is pure routing logic; it does NOT run inference. It depends
only on ``exonit.mast_dv`` for DV-run discovery (already cached).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# The SPOC FFI transit-search / DV pipeline started at TESS sector 36
# (Year 3). Below that, TESS-SPOC FFI HLSP delivers only _lc/_tp.fits — no
# DV report, no difference image. Verified against the released
# vetting_catalog_FFI.csv (0 sector_runs < 36). See progress/exominerpp_setup.md.
SPOC_FFI_MIN_SECTOR = 36


@dataclass(frozen=True)
class GateDecision:
    """Outcome of the ExoMiner++ routing decision for one TIC.

    Attributes:
        use_exominer_pp: True → score with ExoMiner++ 2.0 FFI.
        sector_run: the chosen ``"start-end"`` SPOC run, or None.
        reason: machine-readable reason, one of
            ``"dv_s>=36"``, ``"no_dv"``, ``"dv_below_s36"``.
        dv_runs: all DV runs discovered for the TIC (for diagnostics).
    """

    use_exominer_pp: bool
    sector_run: Optional[str]
    reason: str
    dv_runs: Tuple[Tuple[int, int], ...]


def _pick_ffi_run(runs: Sequence[Tuple[int, int]]) -> Optional[str]:
    """Pick the best DV run whose sectors are all >= ``SPOC_FFI_MIN_SECTOR``.

    Prefers the longest multi-sector run (more stacked transits), falling
    back to the latest single-sector run — mirroring
    ``mast_dv.pick_best_run`` but restricted to the FFI-DV domain.
    """
    eligible = [(s, e) for (s, e) in runs if s >= SPOC_FFI_MIN_SECTOR]
    if not eligible:
        return None
    multi = [(s, e) for (s, e) in eligible if s != e]
    if multi:
        s, e = max(multi, key=lambda r: r[1] - r[0])
        return f"{s}-{e}"
    s, e = max(eligible, key=lambda r: r[0])  # latest single-sector
    return f"{s}-{e}"


def decide_gate(
    tic_id: str,
    dv_runs: Optional[Sequence[Tuple[int, int]]] = None,
    *,
    cache=None,
) -> GateDecision:
    """Decide whether ``tic_id`` should be scored with ExoMiner++ 2.0 FFI.

    Args:
        tic_id: TIC identifier.
        dv_runs: pre-fetched list of ``(start, end)`` DV sector runs. If
            None, they are fetched via ``exonit.mast_dv.fetch_dv_runs``
            (uses ``cache`` if given). Passing them in keeps this function
            pure / offline-testable and avoids a MAST round-trip.
        cache: optional ``MastDVCache`` used only when ``dv_runs is None``.

    Returns:
        GateDecision.
    """
    if dv_runs is None:
        from exonit.mast_dv import fetch_dv_runs  # lazy: avoids astroquery import in tests

        dv_runs = fetch_dv_runs(str(tic_id), cache=cache)

    runs_t: Tuple[Tuple[int, int], ...] = tuple((int(s), int(e)) for s, e in dv_runs)

    if not runs_t:
        return GateDecision(False, None, "no_dv", runs_t)

    sector_run = _pick_ffi_run(runs_t)
    if sector_run is None:
        # DV exists but only for sectors < 36 (2-min era; no FFI DV / diff img).
        return GateDecision(False, None, "dv_below_s36", runs_t)

    return GateDecision(True, sector_run, "dv_s>=36", runs_t)


def has_spoc_dv(
    tic_id: str,
    dv_runs: Optional[Sequence[Tuple[int, int]]] = None,
    *,
    cache=None,
) -> bool:
    """Convenience: True iff ``tic_id`` has SPOC-FFI DV (s>=36) for ExoMiner++."""
    return decide_gate(tic_id, dv_runs, cache=cache).use_exominer_pp
