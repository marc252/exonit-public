"""Verdict Classifier v3_tfopwg — binary LightGBM trained on TFOPWG ground
truth (KP/CP vs FP/FA), Phase 3 block 2 (Phase A).

Unlike every prior verdict model in this repo (v1/v2/v2.1/v2.2), which train
on ``claude_verdict`` (the local classifier's own past behaviour, see
``exonit/verdict_classifier.py`` module docstring), v3_tfopwg trains directly
on the TFOPWG disposition of matched TOI hosts — see
``docs/superpowers/specs/2026-09-19-verdict-v3-tfopwg-design.md``. It is NOT
a drop-in replacement for v1/v2.x in the dispatcher (``predict_verdict`` in
``exonit/verdict_classifier.py``): routing v3_tfopwg to FFI-only candidates
via a new ``EXONIT_VERDICT_FFI_VERSION`` variable is server/dispatcher work
done in a separate change (this module intentionally does not touch
``exonit/verdict_classifier.py`` / ``exonit/server.py`` /
``scripts/recompute_verdict.py``).

Do not confuse with ``scripts/exp_verdict_v3_hierarchical.py`` — an older,
superseded experiment trained on ``claude_verdict`` (the anti-pattern this
module replaces). That script is left untouched but the bare name "v3" is
reserved for it; this module is always spelled out as "v3_tfopwg".

Model shape: a single binary LightGBM P(planet) classifier (no hierarchical
two-stage split like v2.x — with TFOPWG labels the problem genuinely is
planet/not-planet, not LFP-vs-rest-then-PC-vs-UNC). The binary probability is
mapped to the repo's canonical 3-class output via two thresholds (``t_lo``,
``t_hi``) stored IN the model payload, selected by nested cross-validation at
training time (see ``scripts/train_verdict_classifier_v3_tfopwg.py`` — never
tuned to hit the acceptance gate on the same data the gate is measured on,
per spec §2 "Thresholds without circularity"):

    p >= t_hi              -> planet_candidate
    p <  t_lo               -> likely_false_positive
    t_lo <= p < t_hi        -> uncertain

``confidence`` (spec §5 "Sortida de 3 classes"):
    planet_candidate / likely_false_positive : max(p, 1-p)  (standard binary
        margin — how far p sits from the decision boundary at 0.5)
    uncertain : 1 - clamp(|p - 0.5| / ((t_hi - t_lo) / 2), 0, 1)  (1.0 exactly
        at p=0.5, the middle of the band, decaying to 0.0 at either edge)

No ``apply_planet_rescue_gate`` (spec §5 "No apply_planet_rescue_gate" —
the rescue gate exists to counter v2.x's depth-domination; v3_tfopwg reports
per-feature gain instead and, if ``depth_log1p`` exceeds 40% of gain, is
retrained with the v2.2 ``feature_fraction`` mitigation — a training-time
fix, not a post-hoc override).

``fpp_policy`` and the ``(no_fpp)`` serve path (2026-09-19, coordinator-
directed after ``progress/review-verdict-v3-tfopwg-train.md`` found the
FPP/NFPP missing-value split direction itself exploiting disposition-
correlated missingness — FP-promoted 45.9% on the FPP-present subgroup, >2x
the spec ceiling — a repeat of the Berger leakage in a different feature):
the model payload now stores an ``fpp_policy`` key
(``exonit.verdict_v3_tfopwg_features.FPP_POLICIES``: ``"all"`` — the
REJECTED baseline, kept only for the old pickle; ``"none"`` — FPP/NFPP
dropped entirely; ``"present_only"`` — FPP/NFPP kept as values, trained only
on rows where FPP was present).

For ``fpp_policy="present_only"`` specifically, spec §4's "**Forbidden**:
derive any feature from the presence/absence of FPP beyond the flag"
requirement is enforced at SERVE time, not as a trained model behaviour: a
candidate with no TRICERATOPS run has no input this variant's model was ever
trained to interpret (every training row had FPP present, by construction),
so ``predict_v3_tfopwg`` checks FPP presence BEFORE calling the model at all
and returns ``uncertain``/confidence 0.0/``model_used`` suffixed ``(no_fpp)``
directly. This is a **data-completeness policy** (an explicit "we don't have
an opinion" answer for missing input), not a verdict override in the
``apply_planet_rescue_gate``/``apply_fpp_hard_gate`` sense — those rewrite an
already-computed model verdict post-hoc; this short-circuits BEFORE any
model inference happens, precisely because there is nothing honest for the
model to say about a row unlike anything it was trained on.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np

from exonit.verdict_classifier import CLASS_NAMES, _float_or_sentinel
from exonit.verdict_v3_tfopwg_features import build_v3_tfopwg_features_single

_MODELS_DIR_ENV = "EXONIT_MODELS_DIR"
_MODEL_OVERRIDE_ENV = "EXONIT_VERDICT_V3_TFOPWG_MODEL"


def _models_dir() -> Path:
    return Path(os.environ.get(
        _MODELS_DIR_ENV,
        str(Path(__file__).resolve().parents[1] / "models")))


@functools.lru_cache(maxsize=1)
def _load_v3_tfopwg_model() -> dict:
    """Load the v3_tfopwg model payload.

    ``EXONIT_VERDICT_V3_TFOPWG_MODEL`` (if set) is an explicit path override
    that wins outright. Otherwise the lexically-latest
    ``verdict_classifier_v3_tfopwg_*.pkl`` in ``EXONIT_MODELS_DIR`` (default
    ``<repo>/models``) is loaded — same "latest by filename date" convention
    as every prior verdict_classifier_v*._load_v*_model.
    """
    import joblib

    override = os.environ.get(_MODEL_OVERRIDE_ENV)
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(
                f"{_MODEL_OVERRIDE_ENV}={override!r} does not exist")
    else:
        models_dir = _models_dir()
        candidates = sorted(models_dir.glob("verdict_classifier_v3_tfopwg_*.pkl"))
        if not candidates:
            raise FileNotFoundError(
                f"no verdict_classifier_v3_tfopwg_*.pkl in {models_dir}")
        path = candidates[-1]
    payload = joblib.load(str(path))
    payload["path"] = str(path)
    return payload


def _confidence_from_proba(p: float, t_lo: float, t_hi: float) -> tuple[str, float]:
    """Map P(planet) -> (verdict, confidence) via the stored band thresholds.

    See module docstring for the exact confidence formulas.
    """
    if p >= t_hi:
        return CLASS_NAMES[2], float(max(p, 1.0 - p))  # planet_candidate
    if p < t_lo:
        return CLASS_NAMES[0], float(max(p, 1.0 - p))  # likely_false_positive
    half_band = max((t_hi - t_lo) / 2.0, 1e-9)
    dist = abs(p - 0.5) / half_band
    confidence = float(np.clip(1.0 - dist, 0.0, 1.0))
    return CLASS_NAMES[1], confidence  # uncertain


def _fpp_present(candidate: dict) -> bool:
    """Same validity convention as ``tri_valid``
    (``exonit.verdict_classifier.preprocess_lgbm`` / v2.1's preprocessing):
    finite and >= 0. Used ONLY by the ``fpp_policy="present_only"`` serve-time
    gate below — a data-completeness check performed BEFORE inference, never
    a post-hoc verdict override.
    """
    tri = _float_or_sentinel(candidate.get("triceratops_fpp"))
    return bool(np.isfinite(tri) and tri >= 0)


def predict_v3_tfopwg(candidate: dict, star_info: dict | None = None,
                       vetting_result: dict | None = None) -> dict:
    """Binary-LightGBM inference for a single candidate.

    Same interface/return shape as ``predict_v2_1``/``predict_v2_2``:
    ``{verdict, confidence, model_used}``. Honours
    ``EXONIT_USE_CLAUDE_FALLBACK`` at ``confidence < 0.6`` (same convention as
    every other predict_v*). No rescue gate, no FPP hard gate — see module
    docstring.

    ``fpp_policy="present_only"`` payloads short-circuit to ``uncertain``/
    confidence 0.0/``model_used`` suffixed ``(no_fpp)`` for any candidate
    with no TRICERATOPS FPP reading, BEFORE calling the model or checking the
    Claude fallback — see module docstring "fpp_policy and the (no_fpp) serve
    path".
    """
    payload = _load_v3_tfopwg_model()
    model = payload["model"]
    t_lo = payload["t_lo"]
    t_hi = payload["t_hi"]
    fpp_policy = payload.get("fpp_policy", "all")
    model_name = Path(payload["path"]).name

    if fpp_policy == "present_only" and not _fpp_present(candidate):
        return {
            "verdict": CLASS_NAMES[1],  # uncertain
            "confidence": 0.0,
            "model_used": f"verdict_classifier_v3_tfopwg_local ({model_name}) (no_fpp)",
            "proba": None,  # model not evaluated
        }

    X = build_v3_tfopwg_features_single(candidate, star_info, fpp_policy=fpp_policy)
    p = float(model.predict_proba(X)[0, 1])
    verdict, confidence = _confidence_from_proba(p, t_lo, t_hi)

    use_fallback = (
        os.environ.get("EXONIT_USE_CLAUDE_FALLBACK") == "1"
        and confidence < 0.6
    )
    if use_fallback:
        try:
            from exonit.claude_analysis import analyze_candidate
            return analyze_candidate(
                candidate, vetting_result or {"passed": True},
                star_info=star_info)
        except Exception:
            pass  # fall through to local prediction

    return {
        "verdict": verdict,
        "confidence": confidence,
        "model_used": f"verdict_classifier_v3_tfopwg_local ({model_name})",
        "proba": p,
    }


__all__ = [
    "predict_v3_tfopwg",
    "_load_v3_tfopwg_model",
    "_confidence_from_proba",
    "_fpp_present",
]
