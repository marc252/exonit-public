"""Verdict Classifier v2 — hierarchical two-stage LightGBM.

Stage 1: binary LFP-vs-rest
Stage 2: among non-LFP, binary PC-vs-UNC

Winning config from v3-E hier-slow experiments (2026-04-29).
Metrics (5-fold OOF, StratifiedGroupKFold by tic_id):
  agreement=0.7682, f1_macro=0.7357
  per-class: lfp=0.665, unc=0.742, pc=0.800
"""
from __future__ import annotations

import functools
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from exonit.verdict_classifier import CLASS_NAMES, CLASS_TO_INDEX, _float_or_sentinel

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# 15 most common exominer_sector_run values from training data, stored in
# model payload at training time (pre-computed once).  At inference time
# we use the value stored in the loaded payload.  This list is the
# TRAINING-TIME default — it is overridden by the loaded model's top_sectors.
TOP_SECTOR_RUNS: list[str] = []  # populated at training time; see _load_v2_model

# Exact params used in the winning v3-E hier-slow config
LGBM_PARAMS_V2 = dict(
    objective="binary",
    metric="binary_logloss",
    max_depth=5,
    num_leaves=31,
    min_data_in_leaf=20,
    learning_rate=0.02,
    n_estimators=1500,
    reg_alpha=0.0,
    reg_lambda=0.0,
    verbose=-1,
)

# Stage 1 class weights: non-LFP=0, LFP=1
LFP_CLASS_WEIGHT = {0: 1.0, 1: 5.0}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _base_feature_names(top_sectors: Sequence[str]) -> list[str]:
    """Compute the ordered column list for a given top_sectors list."""
    # 15 numeric/derived features
    base = [
        "period_log", "depth_log1p", "duration_days", "snr_log1p", "cnn_score",
        "triceratops_fpp", "tri_valid", "exominer_score", "exominer_tce_period_log",
        "n_sectors", "time_span_days_log1p",
        "duration_per_period", "expected_transits", "depth_over_snr",
        "log_depth_over_dur", "n_datapoints",
    ]
    # sector dummies (sorted alphabetically, __OTHER__ first due to sort)
    all_cats = sorted(["__OTHER__"] + list(top_sectors))
    sector_cols = [f"sec_{c}" for c in all_cats]
    return base + sector_cols


# The tuple exported as LGBM_FEATURE_NAMES_V2 is computed from the canonical
# 15 synthetic sector names.  At inference time we always use the feature names
# stored in the model payload.
_CANONICAL_15 = [
    "s01", "s02", "s03", "s04", "s05",
    "s06", "s07", "s08", "s09", "s10",
    "s11", "s12", "s13", "s14", "s15",
]
LGBM_FEATURE_NAMES_V2: tuple[str, ...] = tuple(_base_feature_names(_CANONICAL_15))


def preprocess_v2(df: pd.DataFrame, top_sectors: Sequence[str]) -> pd.DataFrame:
    """Build the v2 feature matrix.

    Produces IDENTICAL columns and values to build_features() in
    scripts/exp_verdict_v2.py when called with with_sector_run=True
    and the same top_sectors.  Requires 'n_datapoints' and
    'exominer_sector_run' columns in df.

    Returns a DataFrame with columns matching _base_feature_names(top_sectors).
    """
    out = pd.DataFrame(index=df.index)

    period = df["period_days"].astype(float)
    duration = df["duration_days"].astype(float)
    depth = df["depth_ppm"].astype(float)
    snr = df["snr"].astype(float)
    time_span = df["time_span_days"].astype(float)

    out["period_log"] = np.where(period > 0, np.log(period), np.nan)
    out["depth_log1p"] = np.log1p(depth)
    out["duration_days"] = duration
    out["snr_log1p"] = np.log1p(snr)
    out["cnn_score"] = df["cnn_score"].astype(float)

    tri = df["triceratops_fpp"].astype(float)
    tri_valid = (tri.notna() & np.isfinite(tri) & (tri >= 0)).astype(int)
    out["triceratops_fpp"] = tri.where(tri_valid == 1)
    out["tri_valid"] = tri_valid

    out["exominer_score"] = df["exominer_score"].astype(float)
    tce = df["exominer_tce_period"].astype(float)
    out["exominer_tce_period_log"] = np.where(
        (tce > 0) & np.isfinite(tce), np.log(tce), np.nan)

    out["n_sectors"] = df["n_sectors"].astype(float)
    out["time_span_days_log1p"] = np.log1p(time_span)

    out["duration_per_period"] = np.where(period > 0, duration / period, np.nan)
    out["expected_transits"] = np.where(period > 0, time_span / period, np.nan)
    out["depth_over_snr"] = np.where(snr > 0, depth / snr, np.nan)
    out["log_depth_over_dur"] = np.where(
        (duration > 0) & (depth > 0), np.log(depth / duration), np.nan)

    if "n_datapoints" in df.columns:
        out["n_datapoints"] = df["n_datapoints"].astype(float)
    else:
        out["n_datapoints"] = np.nan

    # Sector run one-hot
    s = df["exominer_sector_run"].fillna("__NA__").astype(str) \
        if "exominer_sector_run" in df.columns else pd.Series("__NA__", index=df.index)
    top_set = set(top_sectors)
    s_cat = s.where(s.isin(top_set), "__OTHER__")
    dummies = pd.get_dummies(s_cat, prefix="sec").astype(int)
    out = pd.concat([out, dummies], axis=1)

    # Ensure ALL expected sector columns are present (fill missing with 0)
    expected_cols = _base_feature_names(top_sectors)
    for col in expected_cols:
        if col not in out.columns:
            out[col] = 0
    return out[expected_cols]


# ---------------------------------------------------------------------------
# Cross-validation split helper
# ---------------------------------------------------------------------------

def _cv_split(df: pd.DataFrame, y: np.ndarray, n_splits: int = 5,
              random_state: int = 42):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=random_state)
    yield from sgkf.split(df, y, groups=df["tic_id"].values)


# ---------------------------------------------------------------------------
# OOF training (hierarchical)
# ---------------------------------------------------------------------------

def train_oof_hierarchical(
    X: pd.DataFrame,
    y: np.ndarray,
    df: pd.DataFrame,
    *,
    lfp_weight: float = 5.0,
    random_state: int = 42,
    n_estimators: int = 1500,
    max_depth: int = 5,
    num_leaves: int = 31,
    lr: float = 0.02,
    min_data_in_leaf: int = 20,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
) -> np.ndarray:
    """5-fold hierarchical OOF. Returns (n, 3) probability matrix.

    Algorithmic parity with scripts/exp_verdict_v3_hierarchical.py.
    """
    n = len(df)
    oof = np.zeros((n, 3), dtype=float)

    base_params = dict(
        objective="binary",
        metric="binary_logloss",
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        learning_rate=lr,
        n_estimators=n_estimators,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        verbose=-1,
    )

    for _fold, (tr_idx, te_idx) in enumerate(
            _cv_split(df, y, random_state=random_state)):
        rng = np.random.RandomState(random_state)
        perm = rng.permutation(len(tr_idx))
        cut = int(0.8 * len(tr_idx))
        inner_tr = tr_idx[perm[:cut]]
        inner_va = tr_idx[perm[cut:]]

        # Stage 1: LFP (1) vs non-LFP (0)
        y1 = (y == 0).astype(int)
        m1 = lgb.LGBMClassifier(
            **base_params,
            class_weight={0: 1.0, 1: lfp_weight},
            random_state=random_state)
        m1.fit(X.iloc[inner_tr], y1[inner_tr],
               eval_set=[(X.iloc[inner_va], y1[inner_va])],
               callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
        p_lfp = m1.predict_proba(X.iloc[te_idx])[:, 1]

        # Stage 2: PC (1) vs UNC (0), trained only on non-LFP rows
        non_lfp_train = inner_tr[y[inner_tr] != 0]
        non_lfp_val = inner_va[y[inner_va] != 0]
        y2 = (y == 2).astype(int)  # 1 = PC, 0 = UNC
        m2 = lgb.LGBMClassifier(**base_params, random_state=random_state)
        m2.fit(X.iloc[non_lfp_train], y2[non_lfp_train],
               eval_set=[(X.iloc[non_lfp_val], y2[non_lfp_val])],
               callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
        p_pc_given_notlfp = m2.predict_proba(X.iloc[te_idx])[:, 1]

        oof[te_idx, 0] = p_lfp
        oof[te_idx, 1] = (1 - p_lfp) * (1 - p_pc_given_notlfp)
        oof[te_idx, 2] = (1 - p_lfp) * p_pc_given_notlfp

    return oof


# ---------------------------------------------------------------------------
# Full-data training (for final deployment model)
# ---------------------------------------------------------------------------

def fit_full_hierarchical(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    lfp_weight: float = 5.0,
    random_state: int = 42,
    n_estimators: int = 1500,
    max_depth: int = 5,
    num_leaves: int = 31,
    lr: float = 0.02,
    min_data_in_leaf: int = 20,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
) -> tuple:
    """Train stage1 and stage2 on ALL data (no CV).

    Returns (stage1_model, stage2_model).
    """
    base_params = dict(
        objective="binary",
        metric="binary_logloss",
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        learning_rate=lr,
        n_estimators=n_estimators,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        verbose=-1,
    )

    # Stage 1: LFP (1) vs non-LFP (0)
    y1 = (y == 0).astype(int)
    stage1 = lgb.LGBMClassifier(
        **base_params,
        class_weight={0: 1.0, 1: lfp_weight},
        random_state=random_state)
    stage1.fit(X, y1)

    # Stage 2: PC (1) vs UNC (0), non-LFP rows only
    non_lfp_mask = y != 0
    y2 = (y[non_lfp_mask] == 2).astype(int)
    stage2 = lgb.LGBMClassifier(**base_params, random_state=random_state)
    stage2.fit(X[non_lfp_mask], y2)

    return stage1, stage2


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_v2_model(stage1, stage2, path: str | Path, *,
                  feature_names: list[str], top_sectors: list[str]) -> None:
    """Persist both stages + metadata to a single .pkl file."""
    payload = {
        "stage1": stage1,
        "stage2": stage2,
        "feature_names": feature_names,
        "top_sectors": top_sectors,
        "class_names": list(CLASS_NAMES),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, str(path))


@functools.lru_cache(maxsize=1)
def _load_v2_model() -> dict:
    """Load the lexically-latest verdict_classifier_v2_*.pkl from EXONIT_MODELS_DIR.

    Default EXONIT_MODELS_DIR: <repo>/models.
    Stores 'path' in the returned payload.
    """
    models_dir = Path(os.environ.get(
        "EXONIT_MODELS_DIR",
        str(Path(__file__).resolve().parents[1] / "models")))
    candidates = sorted(models_dir.glob("verdict_classifier_v2_*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"no verdict_classifier_v2_*.pkl in {models_dir}")
    payload = joblib.load(str(candidates[-1]))
    payload["path"] = str(candidates[-1])
    return payload


# ---------------------------------------------------------------------------
# Fast single-row inference
# ---------------------------------------------------------------------------

def _predict_single_v2(stage1, stage2, candidate: dict,
                       star_info: dict, feature_names: list[str],
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

    # exominer_score carries a -1 no-DV cache-negative sentinel (see
    # progress/investigation-exominer-minus1-2026-07-02.md); v2 (unlike
    # v2.1) has no exominer_ffi_score fallback feature, so the minimum
    # consistent fix is to make -1 collapse to the same "missing" value
    # (0.0) that a real None already does — it must never enter the model
    # as if it were a real out-of-distribution score of -1.0.
    _exom_raw = cand.get("exominer_score")
    exom = 0.0 if (_exom_raw is None or _exom_raw == -1) else float(_exom_raw)
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
        if (duration > 0 and depth > 0)
        else np.nan
    )

    # Sector run one-hot
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
    row_vals = base_vals + sector_vals

    row_df = pd.DataFrame([row_vals], columns=feature_names)

    p_lfp = float(stage1.predict_proba(row_df)[0, 1])
    p_pc_given_notlfp = float(stage2.predict_proba(row_df)[0, 1])

    proba = np.array([
        p_lfp,
        (1 - p_lfp) * (1 - p_pc_given_notlfp),
        (1 - p_lfp) * p_pc_given_notlfp,
    ])
    return proba


def predict_v2(candidate: dict, star_info: dict | None = None,
               vetting_result: dict | None = None) -> dict:
    """Hierarchical v2 inference for a single candidate.

    Returns {verdict, confidence, model_used} — same shape as v1 predict_verdict.

    If `EXONIT_USE_CLAUDE_FALLBACK=1` and the local model's confidence is
    below 0.6, the call is routed to the Claude API instead. This keeps the
    high-quality verdict for ambiguous cases while saving ~88% of API cost
    on the easy ones.
    """
    payload = _load_v2_model()
    stage1 = payload["stage1"]
    stage2 = payload["stage2"]
    feature_names = payload["feature_names"]
    top_sectors = payload["top_sectors"]

    proba = _predict_single_v2(stage1, stage2, candidate, star_info or {},
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
            pass  # fall through to local prediction

    model_name = Path(payload["path"]).name
    return {
        "verdict": CLASS_NAMES[label_idx],
        "confidence": confidence,
        "model_used": f"verdict_classifier_v2_local ({model_name})",
        "proba": float(proba[2]),  # raw P(planet_candidate)
    }
