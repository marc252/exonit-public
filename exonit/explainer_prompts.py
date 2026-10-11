"""Shared LoRA explainer system prompt (D9 dedup, 2026-07-02).

The bilingual system prompt MUST byte-match between the training-time chat
template (``scripts/build_lora_explanations_v2.py``, ``_train_system_prompt``)
and the serving-time chat template (``scripts/explainer_service_vllm_qwen3.py``,
``SYSTEM_PROMPT``) for the deployed LoRA adapter to behave as trained. Before
this consolidation the two were maintained as separate copies; they were
verified byte-identical (see
docs/reviews/2026-07-02-opus48-cleanup-review.md item D9 and
progress/impl-training-consolidation-2026-07-02.md) and are now a single
source of truth here.

NOTE: this is distinct from the unrelated "teacher" system prompt in
build_lora_explanations_v2.py (module-level ``SYSTEM_PROMPT``, used only to
ask Claude to *generate* the ground-truth dataset) — that one is not part of
the training/serving parity contract and is left untouched.

THIS FILE IS THE CANONICAL COPY. ``scripts/explainer_service_vllm_qwen3.py``
also carries a guarded ``_FALLBACK_SYSTEM_PROMPT`` (used only when this
module isn't importable, e.g. the standalone ``~/exonit-explainer`` deploy on
<host>, which has no ``exonit/`` package) that MUST stay byte-identical to
``EXPLAINER_SYSTEM_PROMPT`` below — do not edit one without the other.
``tests/test_explainer_prompts.py`` asserts the two stay in sync.
"""
from __future__ import annotations

EXPLAINER_SYSTEM_PROMPT = (
    "You are an exoplanet vetting assistant. Given the measured transit "
    "parameters and pipeline scores for a TESS candidate, write a concise "
    "(2-3 sentence) explanation of the verdict in BOTH Catalan and English, "
    "citing the specific features that drive it (period, depth, SNR, CNN, "
    "TRICERATOPS FPP/NFPP, ExoMiner-FFI, stellar context). Be technical and "
    "specific, not generic. Format: '[CA] <catalan>\\n\\n[EN] <english>'."
)
