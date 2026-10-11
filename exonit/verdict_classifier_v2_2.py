"""Verdict Classifier v2.2 — v2.1 + exominer_ffi_score + depth-bias mitigation.

Builds directly on v2.1 (NFPP + Berger 2023 stellar features). Two changes
motivated by the post-sector-1 retrain (plan 2026-06-25, section A):

  1. NEW feature ``exominer_ffi_score`` (Phase-3 FFI ExoMiner). The originally
     planned co-feature ``exominer_score`` (SPOC Docker) is dead: only ~1.5k of
     the 6.5M FFI candidates ever get a non-(-1) SPOC score (FFI targets have no
     DV product). ``exominer_ffi_score`` is populated for ~3.35M labelled rows,
     so it is the usable orthogonal ExoMiner channel for the verdict model.
     We KEEP ``exominer_score`` as a feature too (it is harmless: ~-1 almost
     everywhere, LGBM treats the constant column as zero-gain). Documented so a
     future deploy on a SPOC-rich catalog still benefits from both columns.

  2. depth_log1p BIAS MITIGATION. In v2.1, ``depth_log1p`` is 66.8% of stage-1
     gain, so deep transits (real hot Jupiters) get pushed to
     ``likely_false_positive``. v2.2 supports two independent, A/B-selectable
     mitigations (see ``scripts/train_verdict_classifier_v2_2.py``):
       (a) the existing post-hoc ``apply_planet_rescue_gate`` (CNN/EM/FPP/NFPP
           all clean → override LFP→PC), and/or
       (b) training-time de-domination of depth: ``feature_fraction`` (column
           subsampling so depth is not in every tree) + an explicit
           ``depth_x_rprs`` interaction feature so depth only forces LFP when the
           implied radius ratio is also stellar. Depth is NOT removed — deep
           real HJs still need it; the goal is to stop depth ALONE forcing LFP.

The feature ordering is: v2.1 base (16 numeric + sector dummies + 8 v2.1 extras)
followed by the v2.2 extras, so a v2.1 payload column list is a strict prefix of
the v2.2 one (eases inspection / debugging).
"""
from __future__ import annotations

import functools
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from exonit.verdict_classifier import CLASS_NAMES, apply_planet_rescue_gate
from exonit.verdict_classifier_v2 import (  # noqa: F401  (re-exported for train script)
    LFP_CLASS_WEIGHT,
    fit_full_hierarchical,
    train_oof_hierarchical,
)
from exonit.verdict_classifier_v2_1 import (
    EXTRA_NUMERIC_FEATURES_V2_1,
    _base_feature_names_v2_1,
    preprocess_v2_1,
)

# v2.2 extra columns appended after the full v2.1 feature list.
#   exominer_ffi_score : Phase-3 FFI ExoMiner score (sentinel <0 -> NaN)
#   ffi_valid          : 1 when FFI score is finite and >= 0
#   depth_x_rprs       : depth_log1p * (implied rp/rs from depth), an interaction
#                        so a deep transit only pushes LFP when the radius ratio
#                        it implies is also stellar (mitigates the depth-alone bug)
EXTRA_NUMERIC_FEATURES_V2_2: list[str] = [
    "exominer_ffi_score",
    "ffi_valid",
    "depth_x_rprs",
]


def _base_feature_names_v2_2(top_sectors: Sequence[str]) -> list[str]:
    """Ordered column list: full v2.1 list + v2.2 extras."""
    return _base_feature_names_v2_1(top_sectors) + EXTRA_NUMERIC_FEATURES_V2_2


def _implied_rprs(depth_ppm: pd.Series) -> pd.Series:
    """rp/rs implied by a box-transit depth (depth = (rp/rs)^2).

    Pure geometric proxy from the BLS depth; no stellar radius needed. Used only
    to form the depth*rprs interaction so depth gets context. NaN/negative -> NaN.
    """
    d = depth_ppm.astype(float) / 1e6  # ppm -> fractional
    d = d.where(np.isfinite(d) & (d > 0))
    return np.sqrt(d)


def preprocess_v2_2(df: pd.DataFrame, top_sectors: Sequence[str]) -> pd.DataFrame:
    """Build the v2.2 feature matrix.

    The input ``df`` must contain everything ``preprocess_v2_1`` needs PLUS the
    ``exominer_ffi_score`` column (use ``None``/``NaN``/-1 when unknown).
    """
    out = preprocess_v2_1(df, top_sectors).copy()

    # exominer_ffi_score: sentinel (<0 / non-finite) -> NaN, with a validity flag.
    ffi = df["exominer_ffi_score"].astype(float) if "exominer_ffi_score" in df.columns \
        else pd.Series(np.nan, index=df.index)
    ffi_valid = (ffi.notna() & np.isfinite(ffi) & (ffi >= 0)).astype(int)
    out["exominer_ffi_score"] = ffi.where(ffi_valid == 1)
    out["ffi_valid"] = ffi_valid

    # depth * implied-rprs interaction (both already in feature space via depth_log1p,
    # but the product gives depth its geometric context so it can't force LFP alone).
    depth_log1p = out["depth_log1p"]
    rprs = _implied_rprs(df["depth_ppm"]) if "depth_ppm" in df.columns \
        else pd.Series(np.nan, index=df.index)
    out["depth_x_rprs"] = depth_log1p * rprs

    return out[_base_feature_names_v2_2(top_sectors)]


def save_v2_2_model(stage1, stage2, path: str | Path, *,
                    feature_names: list[str], top_sectors: list[str],
                    extra: dict | None = None) -> None:
    """Persist both stages + metadata to a single .pkl file."""
    payload = {
        "stage1": stage1,
        "stage2": stage2,
        "feature_names": feature_names,
        "top_sectors": top_sectors,
        "class_names": list(CLASS_NAMES),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "version": "v2.2",
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, str(path))


@functools.lru_cache(maxsize=1)
def _load_v2_2_model() -> dict:
    """Load the lexically-latest verdict_classifier_v2_2_*.pkl from EXONIT_MODELS_DIR."""
    models_dir = Path(os.environ.get(
        "EXONIT_MODELS_DIR",
        str(Path(__file__).resolve().parents[1] / "models")))
    candidates = sorted(models_dir.glob("verdict_classifier_v2_2_*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"no verdict_classifier_v2_2_*.pkl in {models_dir}")
    payload = joblib.load(str(candidates[-1]))
    payload["path"] = str(candidates[-1])
    return payload


def _predict_single_v2_2_stages(stage1, stage2, candidate: dict, star_info: dict,
                                feature_names: list[str],
                                top_sectors: list[str]) -> tuple[float, float]:
    """Build a 1-row feature DataFrame and return the two stage probabilities.

    Returns ``(p_lfp, p_pc_given_notlfp)``. Single source of truth for the v2.2
    feature build + stage inference; the proba and threshold paths both wrap it so
    they can never drift from the training-time feature engineering.
    """
    si = star_info or {}
    cand = candidate or {}

    _exom_raw = cand.get("exominer_score")
    _ffi_raw = cand.get("exominer_ffi_score")

    row = {
        "period_days": cand.get("period_days"),
        "depth_ppm": cand.get("depth_ppm"),
        "duration_days": cand.get("duration_days"),
        "snr": cand.get("snr"),
        "cnn_score": cand.get("cnn_score"),
        "triceratops_fpp": cand.get("triceratops_fpp", -1),
        "triceratops_nfpp": cand.get("triceratops_nfpp", -1),
        "triceratops_prob_planet": cand.get("triceratops_prob_planet"),
        "exominer_score": _exom_raw if _exom_raw is not None else _ffi_raw,
        "exominer_ffi_score": _ffi_raw,
        "exominer_tce_period": cand.get("exominer_tce_period"),
        "exominer_sector_run": cand.get("exominer_sector_run"),
        "n_sectors": si.get("n_sectors"),
        "time_span_days": si.get("time_span_days"),
        "n_datapoints": si.get("n_datapoints") or cand.get("n_datapoints"),
        "berger_mass": si.get("berger_mass"),
        "berger_rad": si.get("berger_rad"),
        "berger_teff": si.get("berger_teff"),
        "berger_logg": si.get("berger_logg"),
        "berger_feh": si.get("berger_feh"),
    }
    row_df = preprocess_v2_2(pd.DataFrame([row]), top_sectors)
    # Defensive: ensure exact training column order.
    row_df = row_df.reindex(columns=feature_names)

    p_lfp = float(stage1.predict_proba(row_df)[0, 1])
    p_pc_given_notlfp = float(stage2.predict_proba(row_df)[0, 1])
    return p_lfp, p_pc_given_notlfp


def _predict_single_v2_2(stage1, stage2, candidate: dict, star_info: dict,
                         feature_names: list[str],
                         top_sectors: list[str]) -> np.ndarray:
    """Build a 1-row feature DataFrame and run hierarchical inference.

    Returns length-3 probability array [p_lfp, p_unc, p_pc]. Delegates the feature
    build + stage inference to ``_predict_single_v2_2_stages``.
    """
    p_lfp, p_pc_given_notlfp = _predict_single_v2_2_stages(
        stage1, stage2, candidate, star_info, feature_names, top_sectors)
    return np.array([
        p_lfp,
        (1 - p_lfp) * (1 - p_pc_given_notlfp),
        (1 - p_lfp) * p_pc_given_notlfp,
    ])


@functools.lru_cache(maxsize=1)
def _get_berger_catalog():
    from exonit.berger_catalog import BergerCatalog
    from exonit.config import ExoNitConfig
    cfg = ExoNitConfig()
    return BergerCatalog(cfg.berger_db_path)


def _augment_with_berger(star_info: dict, candidate: dict) -> dict:
    if "berger_mass" in star_info:
        return star_info
    tic = star_info.get("tic_id") or candidate.get("tic_id")
    if tic is None:
        return star_info
    try:
        entry = _get_berger_catalog().lookup_by_tic(int(tic))
    except (TypeError, ValueError, Exception):
        entry = None
    if entry:
        star_info.setdefault("berger_mass", entry.get("mass"))
        star_info.setdefault("berger_rad", entry.get("rad"))
        star_info.setdefault("berger_teff", entry.get("teff"))
        star_info.setdefault("berger_logg", entry.get("logg"))
        star_info.setdefault("berger_feh", entry.get("feh"))
    return star_info


def predict_v2_2(candidate: dict, star_info: dict | None = None,
                 vetting_result: dict | None = None) -> dict:
    """Hierarchical v2.2 inference for a single candidate.

    Same interface as ``predict_v2_1``: returns {verdict, confidence, model_used}.
    The post-hoc ``apply_planet_rescue_gate`` is applied iff the loaded payload
    was trained with ``rescue_gate=True`` (recorded at training time); this lets
    the A/B pick the mitigation that won without changing inference code.
    """
    payload = _load_v2_2_model()
    stage1 = payload["stage1"]
    stage2 = payload["stage2"]
    feature_names = payload["feature_names"]
    top_sectors = payload["top_sectors"]

    star_info = _augment_with_berger(dict(star_info or {}), candidate)
    p_lfp, p_pc = _predict_single_v2_2_stages(
        stage1, stage2, candidate, star_info, feature_names, top_sectors)

    # Operating point. If the payload carries explicit stage thresholds
    # (re-tuned 2026-06-25 to recover KP/CP recall parity with v2.1 after the
    # depth de-domination made the default argmax too LFP-conservative), use the
    # thresholded decision rule:
    #   p_lfp >= lfp_threshold          -> likely_false_positive
    #   elif  p_pc >= pc_threshold      -> planet_candidate
    #   else                            -> uncertain
    # Backward-compatible: a payload without thresholds falls back to the original
    # hierarchical argmax (lfp_threshold implicit via argmax, pc_threshold=0.5).
    s1_thr = payload.get("lfp_threshold")
    s2_thr = payload.get("pc_threshold")
    if s1_thr is not None and s2_thr is not None:
        if p_lfp >= s1_thr:
            label_idx, confidence = 0, float(p_lfp)
        elif p_pc >= s2_thr:
            label_idx, confidence = 2, float((1 - p_lfp) * p_pc)
        else:
            label_idx, confidence = 1, float((1 - p_lfp) * (1 - p_pc))
    else:
        proba = np.array([
            p_lfp,
            (1 - p_lfp) * (1 - p_pc),
            (1 - p_lfp) * p_pc,
        ])
        label_idx = int(np.argmax(proba))
        confidence = float(proba[label_idx])

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
            pass

    model_name = Path(payload["path"]).name
    base_result = {
        "verdict": CLASS_NAMES[label_idx],
        "confidence": confidence,
        "model_used": f"verdict_classifier_v2_2_local ({model_name})",
    }
    if payload.get("rescue_gate", False):
        return apply_planet_rescue_gate(candidate, base_result)
    return base_result


__all__ = [
    "EXTRA_NUMERIC_FEATURES_V2_2",
    "preprocess_v2_2",
    "save_v2_2_model",
    "predict_v2_2",
    "_predict_single_v2_2",
    "_predict_single_v2_2_stages",
    "_load_v2_2_model",
    "_base_feature_names_v2_2",
    # re-exports for training script convenience:
    "fit_full_hierarchical",
    "train_oof_hierarchical",
    "LFP_CLASS_WEIGHT",
]
