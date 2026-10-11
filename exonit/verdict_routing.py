"""Per-candidate FFI/SPOC routing signal for the verdict classifier dispatcher.

docs/superpowers/specs/2026-09-19-verdict-v3-tfopwg-design.md §5 ("Scope of
data in production"): v3_tfopwg is trained exclusively on TGLC (FFI) rows and
must never score a SPOC candidate (train/serve distribution shift — round 2
review item 3). The dispatcher (``exonit/verdict_classifier.py``'s
``predict_verdict``) picks one model per process today via
``EXONIT_VERDICT_VERSION``; routing v3_tfopwg only to FFI candidates needs a
per-candidate signal ahead of that env-var chain.

Signal: ``metadata.exptime`` (PG ``exonit_meta.metadata``, written by
``ensure_metadata`` at enqueue time, 26.1M rows live in production). SPOC
hosts measure 120-274s; TGLC (FFI) hosts measure 623-1932s, global
distribution dominated by ~1870s — see the spec's round-2 orchestrator
response. The 400s threshold sits cleanly in the gap between the two
populations.

Conservative-by-design: any missing/unparseable/absent signal (no metadata
row, None exptime, a lookup exception) resolves to 'spoc' — the fallback that
keeps a candidate on the long-deployed, extensively-validated v2.1 path
rather than risking an FFI-trained model silently scoring an SPOC candidate.
"""
from __future__ import annotations

import logging

LOG = logging.getLogger("exonit.verdict_routing")

# Threshold that separates SPOC exptimes (120-274s, 100-star canonical
# sample) from TGLC/FFI exptimes (623-1932s, TOI-bench TGLC sample) — see the
# module docstring and the spec's §5 "Senyal de font".
FFI_EXPTIME_THRESHOLD_S = 400.0


def data_source_from_exptime(exptime) -> str:
    """Classify a single exptime value as 'ffi' or 'spoc'.

    None, missing, or unparseable (non-numeric) -> 'spoc' (conservative
    default: falls back to the long-validated v2.1 path). exptime >=
    FFI_EXPTIME_THRESHOLD_S -> 'ffi'; below it -> 'spoc'.
    """
    if exptime is None:
        return "spoc"
    try:
        value = float(exptime)
    except (TypeError, ValueError):
        return "spoc"
    if value >= FFI_EXPTIME_THRESHOLD_S:
        return "ffi"
    return "spoc"


def resolve_data_source(tic_id, metadata_cache) -> str:
    """Look up *tic_id*'s exptime in *metadata_cache* and classify it.

    Never raises: any exception from the cache lookup (missing table,
    connection error, etc.) is logged as a warning and resolved to 'spoc' —
    routing must not become a new failure mode for verdict scoring.
    """
    try:
        entry = metadata_cache.get_entry(str(tic_id))
    except Exception as exc:
        LOG.warning("resolve_data_source: metadata lookup failed for TIC %s: %s",
                    tic_id, exc)
        return "spoc"
    if entry is None:
        return "spoc"
    return data_source_from_exptime(entry.get("exptime"))
