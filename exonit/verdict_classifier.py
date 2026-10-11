"""Verdict Classifier — local 3-class predictor replacing Claude API.

Trained on rows of exonit.db candidates table (post-Claude analysis).
Returns the same dict shape as analyze_candidate() so it's a drop-in
replacement at server.py _run_claude_background.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

CLASS_NAMES = (
    "likely_false_positive",  # 0
    "uncertain",              # 1
    "planet_candidate",       # 2
)
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def _float_or_sentinel(value, sentinel: float = -1.0) -> float:
    """Convert `value` to float, treating a missing value as `sentinel`.

    Unlike the common ``float(value or sentinel)`` idiom used across the
    verdict classifiers, this does NOT collapse a legitimate ``0.0`` into
    `sentinel` — Python truthiness makes ``0.0 or -1`` evaluate to ``-1``,
    which wrongly turned a real "no false-positive probability" FPP/NFPP
    reading into the "missing" sentinel at serving time (diverging from the
    training-side preprocessing, which already treats 0.0 as valid). NaN
    passes through unchanged so the downstream ``np.isfinite`` validity
    check still catches it.
    """
    if value is None:
        return sentinel
    return float(value)


EXPECTED_FEATURE_COLS = (
    "period_days",
    "depth_ppm",
    "duration_days",
    "snr",
    "cnn_score",
    "triceratops_fpp",
    "exominer_score",
    "exominer_tce_period",
    "tmag",
    "ra",
    "dec",
    "n_sectors",
    "time_span_days",
)


def load_dataset(db_path: str | Path) -> pd.DataFrame:
    """Load labeled candidates joined with stellar params from exonit.db.

    Returns a DataFrame with columns: tic_id, candidate_num, claude_verdict,
    plus all EXPECTED_FEATURE_COLS. Rows with NULL claude_verdict are
    excluded (only labeled training data).
    """
    con = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query("""
            SELECT
                c.tic_id, c.candidate_num, c.claude_verdict,
                c.period_days, c.depth_ppm, c.duration_days, c.snr,
                c.cnn_score, c.triceratops_fpp,
                c.exominer_score, c.exominer_tce_period,
                s.tmag, s.ra, s.dec, s.n_sectors, s.time_span_days
            FROM candidates c
            LEFT JOIN stars s ON c.tic_id = s.tic_id
            WHERE c.claude_verdict IS NOT NULL
        """, con)
    finally:
        con.close()
    df["tic_id"] = df["tic_id"].astype(str)
    return df


def compute_target(df: pd.DataFrame) -> np.ndarray:
    """Map claude_verdict → integer class label (0/1/2). Raises on unknown values."""
    unknown = set(df["claude_verdict"].unique()) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"unknown verdict values: {sorted(unknown)}")
    return df["claude_verdict"].map(CLASS_TO_INDEX).astype(int).to_numpy()


LGBM_FEATURE_NAMES = (
    "period_log",
    "depth_log1p",
    "duration_days",
    "snr_log1p",
    "cnn_score",
    "triceratops_fpp",
    "tri_valid",
    "exominer_score",
    "exominer_tce_period_log",
    "tmag",
    "ra",
    "dec",
    "n_sectors",
    "time_span_days_log1p",
)


def preprocess_lgbm(df: pd.DataFrame) -> pd.DataFrame:
    """Build LightGBM feature matrix. NaN preserved for native LGBM handling.
    TRI sentinel (<0 or non-finite) → NaN with tri_valid=0 flag."""
    out = pd.DataFrame(index=df.index)

    period = df["period_days"].astype(float)
    out["period_log"] = np.where(period > 0, np.log(period), np.nan)
    out["depth_log1p"] = np.log1p(df["depth_ppm"].astype(float))
    out["duration_days"] = df["duration_days"].astype(float)
    out["snr_log1p"] = np.log1p(df["snr"].astype(float))
    out["cnn_score"] = df["cnn_score"].astype(float)

    tri = df["triceratops_fpp"].astype(float)
    tri_valid = (tri.notna() & np.isfinite(tri) & (tri >= 0)).astype(int)
    tri_clean = tri.where(tri_valid == 1)
    out["triceratops_fpp"] = tri_clean
    out["tri_valid"] = tri_valid

    out["exominer_score"] = df["exominer_score"].astype(float)
    tce = df["exominer_tce_period"].astype(float)
    out["exominer_tce_period_log"] = np.where(
        (tce > 0) & np.isfinite(tce), np.log(tce), np.nan)

    out["tmag"] = df["tmag"].astype(float)
    out["ra"] = df["ra"].astype(float)
    out["dec"] = df["dec"].astype(float)
    out["n_sectors"] = df["n_sectors"].astype(float)
    out["time_span_days_log1p"] = np.log1p(df["time_span_days"].astype(float))

    return out[list(LGBM_FEATURE_NAMES)]


NUMERIC_IMPUTE_COLS = (
    "triceratops_fpp",
    "exominer_score",
    "exominer_tce_period_log",
    "tmag",
    "ra",
    "dec",
    "n_sectors",
    "time_span_days_log1p",
    "period_log",
    "depth_log1p",
    "snr_log1p",
    "duration_days",
)

LOGISTIC_FEATURE_NAMES = LGBM_FEATURE_NAMES + tuple(
    f"{c}_was_missing" for c in NUMERIC_IMPUTE_COLS
)


def preprocess_logistic(df: pd.DataFrame) -> pd.DataFrame:
    """Build logistic feature matrix. Median imputation per column with
    explicit *_was_missing flags so the model can learn from absence."""
    base = preprocess_lgbm(df)
    out = base.copy()
    for col in NUMERIC_IMPUTE_COLS:
        was_missing = base[col].isna().astype(int)
        median = base[col].median()
        if not np.isfinite(median):
            median = 0.0
        out[col] = base[col].fillna(median)
        out[f"{col}_was_missing"] = was_missing
    return out[list(LOGISTIC_FEATURE_NAMES)]


def cv_split(df: pd.DataFrame, y: np.ndarray, n_splits: int = 5,
             random_state: int = 42):
    """Yield (train_idx, test_idx) ensuring no tic_id appears in both train
    and test of the same fold, while keeping per-class balance."""
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state)
    yield from sgkf.split(df, y, groups=df["tic_id"].values)


import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline


LGBM_PARAMS = dict(
    objective="multiclass",
    num_class=3,
    metric="multi_logloss",
    max_depth=5,
    num_leaves=31,
    min_data_in_leaf=20,
    learning_rate=0.05,
    n_estimators=400,
    class_weight={0: 5.0, 1: 1.0, 2: 1.0},  # boost rare LFP class
    verbose=-1,
)


def make_logistic_pipeline(random_state: int) -> SkPipeline:
    return SkPipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs", max_iter=2000, C=1.0,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])


def train_oof_lgbm(X: pd.DataFrame, y: np.ndarray, df: pd.DataFrame,
                   random_state: int = 42) -> np.ndarray:
    """5-fold CV LightGBM with early stopping. Returns OOF probability matrix
    of shape (n_rows, 3)."""
    oof = np.zeros((len(df), 3), dtype=float)
    for train_idx, test_idx in cv_split(df, y, random_state=random_state):
        rng = np.random.RandomState(random_state)
        perm = rng.permutation(len(train_idx))
        cut = int(0.8 * len(train_idx))
        inner_train = train_idx[perm[:cut]]
        inner_val = train_idx[perm[cut:]]

        model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=random_state)
        model.fit(
            X.iloc[inner_train], y[inner_train],
            eval_set=[(X.iloc[inner_val], y[inner_val])],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )
        oof[test_idx] = model.predict_proba(X.iloc[test_idx])
    return oof


def train_oof_logistic(X: pd.DataFrame, y: np.ndarray, df: pd.DataFrame,
                       random_state: int = 42) -> np.ndarray:
    oof = np.zeros((len(df), 3), dtype=float)
    for train_idx, test_idx in cv_split(df, y, random_state=random_state):
        pipe = make_logistic_pipeline(random_state)
        pipe.fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = pipe.predict_proba(X.iloc[test_idx])
    return oof


from sklearn.metrics import f1_score, confusion_matrix as sk_confusion


def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Compute accuracy, agreement, F1 macro + per-class, confusion matrix.

    `agreement` is the fraction where the model's argmax matches the
    Claude label — same as accuracy, but named differently in the spec
    to emphasise we're measuring imitation of Claude rather than ground
    truth.
    """
    pred = proba.argmax(axis=1)
    accuracy = float((pred == y_true).mean())
    f1_macro = float(f1_score(y_true, pred, average="macro", zero_division=0))
    f1_per_class = f1_score(
        y_true, pred, average=None, labels=[0, 1, 2], zero_division=0).tolist()
    cm = sk_confusion(y_true, pred, labels=[0, 1, 2]).tolist()
    return {
        "accuracy": accuracy,
        "agreement": accuracy,
        "f1_macro": f1_macro,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm,
    }


import functools
import os
from datetime import datetime, timezone

import joblib


def save_model(model, path: str | Path, feature_names: list[str]) -> None:
    """Persist model + feature names + metadata to a single .pkl file."""
    payload = {
        "model": model,
        "feature_names": feature_names,
        "class_names": list(CLASS_NAMES),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, str(path))


@functools.lru_cache(maxsize=1)
def _load_model():
    """Load the latest verdict_classifier_v1*.pkl from EXONIT_MODELS_DIR
    (default: <repo>/models). Cached at process level. v2 model files
    (`verdict_classifier_v2_*.pkl`) are explicitly excluded — they have
    a different payload schema and are handled by
    exonit.verdict_classifier_v2._load_v2_model."""
    from itertools import chain
    models_dir = Path(os.environ.get(
        "EXONIT_MODELS_DIR",
        str(Path(__file__).resolve().parents[1] / "models")))
    candidates = sorted(chain(
        models_dir.glob("verdict_classifier_v1.pkl"),
        models_dir.glob("verdict_classifier_v1_*.pkl"),
    ))
    if not candidates:
        raise FileNotFoundError(
            f"no verdict_classifier_v1*.pkl in {models_dir}")
    payload = joblib.load(str(candidates[-1]))
    payload["path"] = str(candidates[-1])
    return payload


def _build_predict_dataframe(candidate: dict, star_info: dict) -> pd.DataFrame:
    """Assemble a 1-row DataFrame in the same column shape as load_dataset."""
    star_info = star_info or {}
    row = {
        "period_days": candidate.get("period_days"),
        "depth_ppm": candidate.get("depth_ppm"),
        "duration_days": candidate.get("duration_days"),
        "snr": candidate.get("snr"),
        "cnn_score": candidate.get("cnn_score"),
        "triceratops_fpp": candidate.get("triceratops_fpp", -1),
        "exominer_score": candidate.get("exominer_score"),
        "exominer_tce_period": candidate.get("exominer_tce_period"),
        "tmag": star_info.get("tmag"),
        "ra": star_info.get("ra"),
        "dec": star_info.get("dec"),
        "n_sectors": star_info.get("n_sectors"),
        "time_span_days": star_info.get("time_span_days"),
    }
    return pd.DataFrame([row])


def _predict_single_numpy(model, candidate: dict, star_info: dict):
    """Build a 1-row numpy feature array avoiding pandas overhead.
    Column order must match LGBM_FEATURE_NAMES exactly."""
    si = star_info or {}
    period = float(candidate.get("period_days") or 0)
    period_log = np.log(period) if period > 0 else np.nan

    depth = float(candidate.get("depth_ppm") or 0)
    depth_log1p = np.log1p(depth)

    duration = float(candidate.get("duration_days") or 0)

    snr = float(candidate.get("snr") or 0)
    snr_log1p = np.log1p(snr)

    cnn = float(candidate.get("cnn_score") or 0)

    tri = _float_or_sentinel(candidate.get("triceratops_fpp"))
    tri_valid = 1 if (np.isfinite(tri) and tri >= 0) else 0
    tri_clean = tri if tri_valid else np.nan

    exom = float(candidate.get("exominer_score") or 0)
    tce = float(candidate.get("exominer_tce_period") or 0)
    tce_log = np.log(tce) if (tce > 0 and np.isfinite(tce)) else np.nan

    tmag = float(si.get("tmag") or 0)
    ra = float(si.get("ra") or 0)
    dec = float(si.get("dec") or 0)
    n_sec = float(si.get("n_sectors") or 0)
    ts = float(si.get("time_span_days") or 0)
    ts_log = np.log1p(ts)

    row = np.array([[period_log, depth_log1p, duration, snr_log1p, cnn,
                     tri_clean, tri_valid, exom, tce_log,
                     tmag, ra, dec, n_sec, ts_log]], dtype=np.float64)
    return pd.DataFrame(row, columns=list(LGBM_FEATURE_NAMES))


FPP_HARD_GATE_THRESHOLD = 0.5

# Hot-Jupiter rescue gate thresholds. v2.1 stage1 classifier weights
# depth_log1p at 66.8% of total gain, so deep transits (>1% depth) bias
# toward likely_false_positive even when every other indicator agrees the
# signal is planetary. The rescue gate overrides only when ALL four
# orthogonal channels cleanly confirm planet:
#   - CNN >= 0.95             (vetting CNN strongly says planet-like)
#   - EM   >= 0.9             (ExoMiner SPOC OR FFI fallback agrees)
#   - FPP  < 0.5 (or missing) (TRICERATOPS doesn't object — missing → pass)
#   - NFPP < 0.5 (or missing) (no nearby star contamination flagged)
# An EB or background-blend FP fails at least one of these — the rescue is
# conservative enough to leave them tagged.
RESCUE_CNN_MIN = 0.95
RESCUE_EM_MIN = 0.9
RESCUE_FPP_MAX = 0.5
RESCUE_NFPP_MAX = 0.5


def apply_planet_rescue_gate(candidate: dict, base_result: dict) -> dict:
    """Override ``likely_false_positive`` → ``planet_candidate`` when CNN,
    ExoMiner, FPP and NFPP all independently agree the signal is planetary.

    Inverse of :func:`apply_fpp_hard_gate`. Designed for hot Jupiters where
    the v2.1 model's depth-dominated stage1 mis-fires. See module docstring
    constants for thresholds. Confidence becomes ``min(cnn, em)`` clipped
    to ``[0.6, 0.95]`` and ``model_used`` is suffixed with ``+rescue``.

    A no-op when ``base_result['verdict'] != 'likely_false_positive'``.
    """
    if base_result.get("verdict") != CLASS_NAMES[0]:
        return base_result

    cnn = candidate.get("cnn_score")
    cnn_v = float(cnn) if (cnn is not None and np.isfinite(cnn)) else None
    if cnn_v is None or cnn_v < RESCUE_CNN_MIN:
        return base_result

    em_raw = candidate.get("exominer_score")
    # exominer_score carries a -1 no-DV cache-negative sentinel (see
    # progress/investigation-exominer-minus1-2026-07-02.md) — treat it the
    # same as a missing (None) score so the FFI fallback below still
    # engages. Found via the same-family grep during the 2026-07-02 -1
    # read-boundary fix (progress/impl-minus1-fix-2026-07-02.md); this
    # function is unconditionally called from predict_v2_1 (the live
    # EXONIT_VERDICT_VERSION=v2.1 production path), so without this the
    # rescue gate silently skipped every FFI-only hot-Jupiter candidate
    # (exominer_score=-1 forced em_v to None below regardless of a strong
    # exominer_ffi_score, since -1 fails the "em_raw >= 0" guard).
    if em_raw is None or em_raw == -1:
        em_raw = candidate.get("exominer_ffi_score")
    em_v = float(em_raw) if (em_raw is not None and np.isfinite(em_raw)
                             and em_raw >= 0) else None
    if em_v is None or em_v < RESCUE_EM_MIN:
        return base_result

    fpp = candidate.get("triceratops_fpp")
    fpp_v = (float(fpp) if (fpp is not None and np.isfinite(fpp)
                            and fpp >= 0) else None)
    if fpp_v is not None and fpp_v >= RESCUE_FPP_MAX:
        return base_result

    nfpp = candidate.get("triceratops_nfpp")
    nfpp_v = (float(nfpp) if (nfpp is not None and np.isfinite(nfpp)
                              and nfpp >= 0) else None)
    if nfpp_v is not None and nfpp_v >= RESCUE_NFPP_MAX:
        return base_result

    confidence = max(0.6, min(0.95, min(cnn_v, em_v)))
    return {
        "verdict": CLASS_NAMES[2],  # planet_candidate
        "confidence": confidence,
        "model_used": f"{base_result.get('model_used', '?')}+rescue",
    }


def apply_fpp_hard_gate(candidate: dict, base_result: dict) -> dict:
    """Override the verdict to ``likely_false_positive`` when TRICERATOPS
    reports FPP≥0.5 or NFPP≥0.5.

    Rationale: FP investigation 2026-05-03 found 15 candidates flagged with
    EM≥0.9 whose TIC is TFOPWG=FP and which TRICERATOPS already caught
    (FPP=0.84..1.0). The classifier wasn't pulling FPP hard enough; this
    gate enforces it deterministically. NFPP≥0.5 catches the same physical
    failure mode (signal coming from a nearby star).

    Confidence becomes max(fpp, nfpp), clipped to [threshold, 1].
    The gate appends ``+fpp_gate`` to ``model_used`` for auditability.
    """
    fpp = candidate.get("triceratops_fpp")
    nfpp = candidate.get("triceratops_nfpp")
    fpp_v = float(fpp) if fpp is not None and np.isfinite(fpp) and fpp >= 0 else None
    nfpp_v = float(nfpp) if nfpp is not None and np.isfinite(nfpp) and nfpp >= 0 else None
    trip = max((v for v in (fpp_v, nfpp_v) if v is not None), default=None)
    if trip is None or trip < FPP_HARD_GATE_THRESHOLD:
        return base_result
    return {
        "verdict": CLASS_NAMES[0],  # likely_false_positive
        "confidence": min(1.0, trip),
        "model_used": f"{base_result.get('model_used', '?')}+fpp_gate",
    }


def predict_verdict(candidate: dict, vetting_result: dict,
                    star_info: dict | None = None,
                    data_source: str | None = None) -> dict:
    """Drop-in replacement for analyze_candidate(). Returns the same dict
    shape so server.py _run_claude_background callers don't change.

    Optional fallback to Claude API: if EXONIT_USE_CLAUDE_FALLBACK=1 and
    the predicted confidence is below 0.6, route the call through the
    original analyze_candidate(). Off by default.

    Set EXONIT_VERDICT_VERSION to route to a specific model:
    "v2.2" -> hierarchical v2.2 (+ exominer_ffi_score + depth-bias mitigation),
    "v2.1" -> hierarchical v2.1, "v2" -> hierarchical v2. Unset/empty defaults
    to v1 (production default, 2026-07-02). Any OTHER non-empty value raises
    ValueError instead of silently falling through to v1 — B4
    (docs/reviews/2026-07-02-opus48-cleanup-review.md): a typo'd or
    not-yet-wired version string (e.g. "v2.2" before this branch existed)
    used to route to v1 without any signal.

    FFI/SPOC routing (docs/superpowers/specs/2026-09-19-verdict-v3-tfopwg-design.md
    §5): if EXONIT_VERDICT_FFI_VERSION is non-empty AND data_source == 'ffi',
    this branch is checked BEFORE EXONIT_VERDICT_VERSION and routes to the
    FFI-only model instead — v3_tfopwg is trained exclusively on TGLC data
    and must never score an SPOC candidate. Only 'v3_tfopwg' is accepted;
    any other value raises ValueError (same B4 spirit as the guard below).
    EXONIT_VERDICT_FFI_VERSION unset/empty (default), or data_source != 'ffi',
    falls through unchanged to the EXONIT_VERDICT_VERSION logic below — so
    every SPOC candidate, and every candidate when the new env var is unset,
    keeps byte-for-byte current behaviour.
    """
    ffi_ver = os.environ.get("EXONIT_VERDICT_FFI_VERSION", "")
    if ffi_ver and data_source == "ffi":
        if ffi_ver != "v3_tfopwg":
            raise ValueError(
                f"EXONIT_VERDICT_FFI_VERSION={ffi_ver!r} is not a known FFI "
                "verdict model version (expected: v3_tfopwg). Refusing to "
                "silently fall back to the EXONIT_VERDICT_VERSION path — "
                "same rationale as B4 in "
                "docs/reviews/2026-07-02-opus48-cleanup-review.md.")
        from exonit.verdict_classifier_v3_tfopwg import predict_v3_tfopwg
        return predict_v3_tfopwg(candidate, star_info=star_info,
                                 vetting_result=vetting_result)

    ver = os.environ.get("EXONIT_VERDICT_VERSION", "v1")
    if ver == "v2.2":
        from exonit.verdict_classifier_v2_2 import predict_v2_2
        return predict_v2_2(candidate, star_info or {},
                            vetting_result=vetting_result)
    if ver == "v2.1":
        from exonit.verdict_classifier_v2_1 import predict_v2_1
        return predict_v2_1(candidate, star_info or {},
                            vetting_result=vetting_result)
    if ver == "v2":
        from exonit.verdict_classifier_v2 import predict_v2
        return predict_v2(candidate, star_info or {},
                          vetting_result=vetting_result)
    if ver != "v1":
        raise ValueError(
            f"EXONIT_VERDICT_VERSION={ver!r} is not a known verdict model "
            "version (expected one of: v1, v2, v2.1, v2.2). Refusing to "
            "silently fall back to v1 — see B4 in "
            "docs/reviews/2026-07-02-opus48-cleanup-review.md.")

    payload = _load_model()
    model = payload["model"]
    X = _predict_single_numpy(model, candidate, star_info or {})
    proba = model.predict_proba(X)[0]
    label_idx = int(np.argmax(proba))
    confidence = float(proba[label_idx])

    use_fallback = (
        os.environ.get("EXONIT_USE_CLAUDE_FALLBACK") == "1"
        and confidence < 0.6
    )
    if use_fallback:
        try:
            from exonit.claude_analysis import analyze_candidate
            return analyze_candidate(candidate, vetting_result, star_info=star_info)
        except Exception:
            pass  # fall through to local prediction

    return {
        "verdict": CLASS_NAMES[label_idx],
        "confidence": confidence,
        "model_used": f"verdict_classifier_local ({Path(payload['path']).name})",
    }
