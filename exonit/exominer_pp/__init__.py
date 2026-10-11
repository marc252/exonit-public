"""ExoMiner++ 2.0 (NASA FFI) inference integration for ExoNit.

This package wraps NASA's published ExoMiner++ 2.0 *FFI new-architecture*
CV ensemble (arXiv 2601.14877, Zenodo 10.5281/zenodo.17707413) so it can
score a TIC that HAS SPOC-FFI DV products (TESS sector >= 36), and is
cleanly skipped on sectors that don't (TGLC-only / s < 36), where ExoNit
falls back to its own ExoMiner-FFI flux model.

Design (see ``progress/exominerpp_implementation.md``):

- Input generation is *delegated to NASA's published Docker pipeline*
  (``ghcr.io/nasa/exominer:latest`` running ``run_pipeline.py
  --data_collection_mode ffi``). Re-implementing the diff-image / neighbor /
  spline / periodogram preprocessing from scratch would diverge from the
  published 56-input contract and carry an unacceptable validation burden
  (CLAUDE.md rule #3: reproducibility innegociable). The Docker image is the
  byte-faithful reference and *also bundles the normalization statistics*
  that are absent from the Zenodo TFRecord release — resolving the
  normalization blocker documented in ``progress/exominerpp_setup.md``.

- ``contract``   — the exact 56-input model contract + a verifier.
- ``norm_stats`` — loader for the bundled per-feature normalization stats.
- ``inference``  — loads the 5-fold CV ensemble (keras 2.13) and scores.
- ``gate``       — the conditional DV-availability gate (pp vs fallback).
- ``pipeline``   — drives the NASA Docker pipeline for a TIC + sector.

Nothing here is wired into the live discovery flow or the production
server; it is an opt-in path used only on s>=36 SPOC-DV sectors.
"""

from .contract import (
    EXOMINER_PP_INPUTS,
    IMAGE_BRANCHES,
    ContractError,
    verify_feature_dict,
)
from .gate import GateDecision, decide_gate, has_spoc_dv

__all__ = [
    "EXOMINER_PP_INPUTS",
    "IMAGE_BRANCHES",
    "ContractError",
    "verify_feature_dict",
    "GateDecision",
    "decide_gate",
    "has_spoc_dv",
]
