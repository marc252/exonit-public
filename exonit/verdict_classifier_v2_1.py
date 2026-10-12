"""Verdict Classifier v2.1 — adds NFPP + Berger 2023 stellar features to v2.

Same hierarchical two-stage LightGBM as v2 (LFP-vs-rest, then PC-vs-UNC),
but with eight extra features now that the NFPP backfill is complete:

  - triceratops_nfpp: nearby-star FPP (Giacalone+ 2021 validation rule)
  - nfpp_valid: 1 when NFPP is finite/non-negative
  - triceratops_prob_planet: 1 - FPP, redundant with FPP for now but kept
    as future-proofing for sources that report prob_planet directly
  - berger_mass / berger_rad / berger_teff_log / berger_logg / berger_feh:
    isochrone-derived stellar parameters for the host (Berger 2023 GKTH catalog,
    populated for the ~4180 TESS hosts in the cross-match). Missing for
    ~70% of TICs — LightGBM handles NaN natively.

Compared to v2:
  - Adds 8 numeric features (no new sector dummies)
  - Same hyperparameters by default; train script can override via Optuna
  - Saves under verdict_classifier_v2_1_*.pkl to avoid clashing with v2 deploy
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

from exonit.verdict_classifier import CLASS_NAMES, _float_or_sentinel
from exonit.verdict_classifier_v2 import (
    LFP_CLASS_WEIGHT,
    fit_full_hierarchical,
    preprocess_v2,
    train_oof_hierarchical,
)

# v2.1 extra columns (added on top of the v2 base 16 + sector dummies)
EXTRA_NUMERIC_FEATURES_V2_1: list[str] = [
    "triceratops_nfpp",
    "nfpp_valid",
    "triceratops_prob_planet",
    "berger_mass",
    "berger_rad",
    "berger_teff_log",
    "berger_logg",
    "berger_feh",
]


def _base_feature_names_v2_1(top_sectors: Sequence[str]) -> list[str]:
    """Ordered column list: v2 base (16 numeric + sectors) + v2.1 extras."""
    base = [
        "period_log", "depth_log1p", "duration_days", "snr_log1p", "cnn_score",
        "triceratops_fpp", "tri_valid", "exominer_score",
        "exominer_tce_period_log", "n_sectors", "time_span_days_log1p",
        "duration_per_period", "expected_transits", "depth_over_snr",
        "log_depth_over_dur", "n_datapoints",
    ]
    all_cats = sorted(["__OTHER__"] + list(top_sectors))
    sector_cols = [f"sec_{c}" for c in all_cats]
    return base + sector_cols + EXTRA_NUMERIC_FEATURES_V2_1


def preprocess_v2_1(df: pd.DataFrame, top_sectors: Sequence[str]) -> pd.DataFrame:
    """Build the v2.1 feature matrix.

    The input ``df`` must already contain the same columns as for ``preprocess_v2``
    PLUS ``triceratops_nfpp``, ``triceratops_prob_planet`` and the five
    ``berger_*`` columns (use ``None``/``NaN`` when unknown).
    """
    out = preprocess_v2(df, top_sectors).copy()

    # NFPP: keep raw value; flag valid rows; clear invalid to NaN.
    nfpp = df["triceratops_nfpp"].astype(float)
    nfpp_valid = (nfpp.notna() & np.isfinite(nfpp) & (nfpp >= 0)).astype(int)
    out["triceratops_nfpp"] = nfpp.where(nfpp_valid == 1)
    out["nfpp_valid"] = nfpp_valid

    prob_p = df["triceratops_prob_planet"].astype(float)
    prob_p = prob_p.where(prob_p.notna() & np.isfinite(prob_p) & (prob_p >= 0))
    out["triceratops_prob_planet"] = prob_p

    # Berger 2023 stellar params: keep raw, only Teff log-transformed.
    out["berger_mass"] = df["berger_mass"].astype(float)
    out["berger_rad"] = df["berger_rad"].astype(float)
    teff = df["berger_teff"].astype(float)
    out["berger_teff_log"] = np.where(
        (teff > 0) & np.isfinite(teff), np.log(teff), np.nan)
    out["berger_logg"] = df["berger_logg"].astype(float)
    out["berger_feh"] = df["berger_feh"].astype(float)

    return out[_base_feature_names_v2_1(top_sectors)]


def save_v2_1_model(stage1, stage2, path: str | Path, *,
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
        "version": "v2.1",
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, str(path))


@functools.lru_cache(maxsize=1)
def _load_v2_1_model() -> dict:
    """Load the lexically-latest verdict_classifier_v2_1_*.pkl from EXONIT_MODELS_DIR."""
    models_dir = Path(os.environ.get(
        "EXONIT_MODELS_DIR",
        str(Path(__file__).resolve().parents[1] / "models")))
    candidates = sorted(models_dir.glob("verdict_classifier_v2_1_*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"no verdict_classifier_v2_1_*.pkl in {models_dir}")
    payload = joblib.load(str(candidates[-1]))
    payload["path"] = str(candidates[-1])
    return payload


def _predict_single_v2_1(stage1, stage2, candidate: dict, star_info: dict,
                         feature_names: list[str],
                         top_sectors: list[str]) -> np.ndarray:
    """Build a 1-row feature DataFrame and run hierarchical inference.

    Returns length-3 probability array [p_lfp, p_unc, p_pc].
    """
    si = star_info or {}
    cand = candidate or {}

    period = float(cand.get("period_days") or 0)
    period_log = np.log(period) if period > 0 else np.nan
    depth = float(cand.get("depth_ppm") or 0)
    depth_log1p = float(np.log1p(depth))
    duration = float(cand.get("duration_days") or 0)
    snr = float(cand.get("snr") or 0)
    snr_log1p = float(np.log1p(snr))
    cnn = float(cand.get("cnn_score") or 0)

    tri = _float_or_sentinel(cand.get("triceratops_fpp"))
    tri_valid = 1 if (np.isfinite(tri) and tri >= 0) else 0
    tri_clean = tri if tri_valid else np.nan

    nfpp = _float_or_sentinel(cand.get("triceratops_nfpp"))
    nfpp_valid = 1 if (np.isfinite(nfpp) and nfpp >= 0) else 0
    nfpp_clean = nfpp if nfpp_valid else np.nan
    prob_p = cand.get("triceratops_prob_planet")
    prob_p = float(prob_p) if (prob_p is not None and np.isfinite(prob_p)
                               and prob_p >= 0) else np.nan

    _exom_raw = cand.get("exominer_score")
    # exominer_score carries a -1 no-DV cache-negative sentinel (~61% of
    # candidates.rows — see progress/investigation-exominer-minus1-2026-07-02.md);
    # it is NOT a real score, so it must fall back to exominer_ffi_score the
    # same way a NULL exominer_score already does. Compared with `== -1`
    # (not `is -1`): the DB always hands back a float here (-1.0), and
    # Python's `-1.0 == -1` is True, so this also catches an int -1 if one
    # ever arrives.
    if _exom_raw is None or _exom_raw == -1:
        _exom_raw = cand.get("exominer_ffi_score")
    exom = float(_exom_raw or 0)
    tce = float(cand.get("exominer_tce_period") or 0)
    tce_log = np.log(tce) if (tce > 0 and np.isfinite(tce)) else np.nan

    n_sec = float(si.get("n_sectors") or 0)
    ts = float(si.get("time_span_days") or 0)
    ts_log = float(np.log1p(ts))
    n_dp = float(si.get("n_datapoints") or cand.get("n_datapoints") or 0)

    dur_per_period = duration / period if period > 0 else np.nan
    expected_transits = ts / period if period > 0 else np.nan
    depth_over_snr = depth / snr if snr > 0 else np.nan
    log_depth_over_dur = (
        float(np.log(depth / duration))
        if (duration > 0 and depth > 0) else np.nan)

    bmass = si.get("berger_mass")
    brad = si.get("berger_rad")
    bteff = si.get("berger_teff")
    blogg = si.get("berger_logg")
    bfeh = si.get("berger_feh")
    bmass_v = float(bmass) if (bmass is not None and np.isfinite(bmass)) else np.nan
    brad_v = float(brad) if (brad is not None and np.isfinite(brad)) else np.nan
    bteff_log = (float(np.log(bteff))
                 if (bteff is not None and bteff > 0 and np.isfinite(bteff))
                 else np.nan)
    blogg_v = float(blogg) if (blogg is not None and np.isfinite(blogg)) else np.nan
    bfeh_v = float(bfeh) if (bfeh is not None and np.isfinite(bfeh)) else np.nan

    sec_run = str(cand.get("exominer_sector_run") or "__NA__")
    top_set = set(top_sectors)
    sec_cat = sec_run if sec_run in top_set else "__OTHER__"
    all_cats = sorted(["__OTHER__"] + list(top_sectors))

    base_vals = [
        period_log, depth_log1p, duration, snr_log1p, cnn,
        tri_clean, tri_valid, exom, tce_log,
        n_sec, ts_log,
        dur_per_period, expected_transits, depth_over_snr, log_depth_over_dur,
        n_dp,
    ]
    sector_vals = [1 if c == sec_cat else 0 for c in all_cats]
    extra_vals = [
        nfpp_clean, nfpp_valid, prob_p,
        bmass_v, brad_v, bteff_log, blogg_v, bfeh_v,
    ]
    row_vals = base_vals + sector_vals + extra_vals
    row_df = pd.DataFrame([row_vals], columns=feature_names)

    p_lfp = float(stage1.predict_proba(row_df)[0, 1])
    p_pc_given_notlfp = float(stage2.predict_proba(row_df)[0, 1])

    return np.array([
        p_lfp,
        (1 - p_lfp) * (1 - p_pc_given_notlfp),
        (1 - p_lfp) * p_pc_given_notlfp,
    ])


@functools.lru_cache(maxsize=1)
def _get_berger_catalog():
    """Lazy-load the Berger 2023 stellar catalog once per process."""
    from exonit.berger_catalog import BergerCatalog
    from exonit.config import ExoNitConfig
    cfg = ExoNitConfig()
    return BergerCatalog(cfg.berger_db_path)


def _augment_with_berger(star_info: dict, candidate: dict) -> dict:
    """Populate berger_* keys in star_info via TIC lookup if missing."""
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


def predict_v2_1(candidate: dict, star_info: dict | None = None,
                 vetting_result: dict | None = None) -> dict:
    """Hierarchical v2.1 inference for a single candidate.

    Same interface as predict_v2: returns {verdict, confidence, model_used}.
    Honours EXONIT_USE_CLAUDE_FALLBACK at confidence < 0.6.

    Auto-populates Berger 2023 stellar features if not provided in star_info,
    using the TIC ID from star_info or candidate.
    """
    payload = _load_v2_1_model()
    stage1 = payload["stage1"]
    stage2 = payload["stage2"]
    feature_names = payload["feature_names"]
    top_sectors = payload["top_sectors"]

    star_info = _augment_with_berger(dict(star_info or {}), candidate)
    proba = _predict_single_v2_1(stage1, stage2, candidate, star_info,
                                 feature_names, top_sectors)
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
        "model_used": f"verdict_classifier_v2_1_local ({model_name})",
        "proba": float(proba[2]),  # raw P(planet_candidate)
    }
    from exonit.verdict_classifier import apply_planet_rescue_gate
    return apply_planet_rescue_gate(candidate, base_result)


__all__ = [
    "EXTRA_NUMERIC_FEATURES_V2_1",
    "preprocess_v2_1",
    "save_v2_1_model",
    "predict_v2_1",
    "_predict_single_v2_1",
    "_load_v2_1_model",
    "_base_feature_names_v2_1",
    # re-exports for training script convenience:
    "fit_full_hierarchical",
    "train_oof_hierarchical",
    "LFP_CLASS_WEIGHT",
]
