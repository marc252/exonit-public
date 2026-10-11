"""Tests for exonit/verdict_classifier_v2.py — hierarchical verdict classifier."""
from __future__ import annotations

import math
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helper: tiny in-memory DB (copied from test_verdict_classifier.py)
# ---------------------------------------------------------------------------

def _make_db(tmp_path: Path, candidates: list[dict], stars: list[dict]) -> Path:
    """Create a tiny exonit.db-shaped database for tests."""
    p = tmp_path / "exonit.db"
    con = sqlite3.connect(p)
    con.execute("""CREATE TABLE candidates (
        id INTEGER PRIMARY KEY, tic_id TEXT, candidate_num INTEGER,
        period_days REAL, t0 REAL, duration_days REAL, depth_ppm REAL, snr REAL,
        cnn_score REAL, triceratops_fpp REAL, claude_verdict TEXT,
        claude_analysis TEXT, ctoi_submitted INTEGER,
        exominer_score REAL, exominer_tce_period REAL, exominer_sector_run TEXT
    )""")
    con.execute("""CREATE TABLE stars (
        tic_id TEXT PRIMARY KEY, ra REAL, dec REAL, tmag REAL,
        analyzed_at TEXT, n_sectors INTEGER, n_datapoints INTEGER,
        time_span_days REAL, status TEXT, sectors_observed TEXT
    )""")
    for c in candidates:
        con.execute(
            "INSERT INTO candidates(tic_id,candidate_num,period_days,t0,duration_days,"
            "depth_ppm,snr,cnn_score,triceratops_fpp,claude_verdict,exominer_score,"
            "exominer_tce_period,exominer_sector_run) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (c.get("tic_id"), c.get("candidate_num", 1), c.get("period_days", 2.5),
             c.get("t0", 1000.0), c.get("duration_days", 0.1), c.get("depth_ppm", 500.0),
             c.get("snr", 12.0), c.get("cnn_score", 0.8), c.get("triceratops_fpp", 0.1),
             c.get("claude_verdict", "planet_candidate"),
             c.get("exominer_score", 0.9), c.get("exominer_tce_period", 2.5),
             c.get("exominer_sector_run", "s01")))
    for s in stars:
        con.execute(
            "INSERT INTO stars(tic_id,ra,dec,tmag,n_sectors,n_datapoints,time_span_days,status) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (s["tic_id"], s.get("ra", 100.0), s.get("dec", 30.0), s.get("tmag", 11.0),
             s.get("n_sectors", 3), s.get("n_datapoints", 5000),
             s.get("time_span_days", 100.0), s.get("status", "completed")))
    con.commit()
    con.close()
    return p


# The 15 canonical sector run names used in synthetic tests
_SYNTHETIC_TOP_SECTORS = [
    "s01", "s02", "s03", "s04", "s05",
    "s06", "s07", "s08", "s09", "s10",
    "s11", "s12", "s13", "s14", "s15",
]


def _make_synthetic_df(n_per_sector: int = 2) -> pd.DataFrame:
    """Build a synthetic DataFrame with all 15 top sectors represented."""
    rows = []
    verdicts = ["likely_false_positive", "uncertain", "planet_candidate"]
    for i, sec in enumerate(_SYNTHETIC_TOP_SECTORS):
        for j in range(n_per_sector):
            rows.append({
                "tic_id": f"{i * 100 + j}",
                "period_days": 2.5 + i * 0.1,
                "duration_days": 0.1,
                "depth_ppm": 500.0,
                "snr": 12.0,
                "time_span_days": 100.0,
                "cnn_score": 0.5,
                "triceratops_fpp": 0.1,
                "exominer_score": 0.9,
                "exominer_tce_period": 2.5,
                "n_sectors": 3,
                "n_datapoints": 5000,
                "exominer_sector_run": sec,
                "claude_verdict": verdicts[i % 3],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from exonit.verdict_classifier_v2 import (  # noqa: E402
    LGBM_FEATURE_NAMES_V2,
    LGBM_PARAMS_V2,
    LFP_CLASS_WEIGHT,
    TOP_SECTOR_RUNS,
    fit_full_hierarchical,
    predict_v2,
    preprocess_v2,
    save_v2_model,
    train_oof_hierarchical,
    _load_v2_model,
    _predict_single_v2,
)


# ===========================================================================
# 1. preprocess_v2 column shape
# ===========================================================================

def test_preprocess_v2_returns_expected_columns():
    """Synthetic 5-row dataset with all 15 TOP_SECTOR_RUNS → 32 columns."""
    df = _make_synthetic_df(n_per_sector=1)
    # Pad to have all 15
    X = preprocess_v2(df, top_sectors=_SYNTHETIC_TOP_SECTORS)
    assert list(X.columns) == list(LGBM_FEATURE_NAMES_V2), (
        f"got {list(X.columns)}"
    )
    assert X.shape[1] == 32


# ===========================================================================
# 2. Unknown sector → __OTHER__
# ===========================================================================

def test_preprocess_v2_unknown_sector_goes_to_other():
    """sector_run='999-999' (not in top) → sec___OTHER__=1."""
    df = _make_synthetic_df(n_per_sector=1)
    # Override one row's sector_run to an unknown value
    df = df.copy()
    df.iloc[0, df.columns.get_loc("exominer_sector_run")] = "999-999"
    X = preprocess_v2(df, top_sectors=_SYNTHETIC_TOP_SECTORS)
    assert "sec___OTHER__" in X.columns
    assert X.iloc[0]["sec___OTHER__"] == 1


# ===========================================================================
# 3. period_days=0 → derived features are NaN
# ===========================================================================

def test_preprocess_v2_nan_period_to_nan_features():
    """period_days=0 → period_log=NaN, expected_transits=NaN, duration_per_period=NaN."""
    df = _make_synthetic_df(n_per_sector=1)
    df = df.copy()
    df.iloc[0, df.columns.get_loc("period_days")] = 0.0
    X = preprocess_v2(df, top_sectors=_SYNTHETIC_TOP_SECTORS)
    assert math.isnan(X.iloc[0]["period_log"])
    assert math.isnan(X.iloc[0]["expected_transits"])
    assert math.isnan(X.iloc[0]["duration_per_period"])


# ===========================================================================
# 4. train_oof_hierarchical shape + row sums
# ===========================================================================

def _make_hier_training_data():
    """60 TICs, 3 per TIC, balanced classes, with sector runs."""
    rows = []
    sectors = _SYNTHETIC_TOP_SECTORS  # 15
    verdicts = ["likely_false_positive", "uncertain", "planet_candidate"]
    for tic in range(60):
        v = verdicts[tic % 3]
        for k in range(3):
            sec = sectors[tic % 15]
            rows.append({
                "tic_id": str(tic), "candidate_num": k + 1,
                "period_days": 2.5, "duration_days": 0.1, "depth_ppm": 500.0,
                "snr": 12.0, "time_span_days": 100.0, "cnn_score": 0.5,
                "triceratops_fpp": 0.1, "exominer_score": 0.9,
                "exominer_tce_period": 2.5, "n_sectors": 3, "n_datapoints": 5000,
                "exominer_sector_run": sec,
                "claude_verdict": v,
            })
    df = pd.DataFrame(rows)
    from exonit.verdict_classifier import compute_target
    y = compute_target(df)
    X = preprocess_v2(df, top_sectors=_SYNTHETIC_TOP_SECTORS)
    return X, y, df


def test_train_oof_hierarchical_shape_and_sums():
    """OOF shape (n,3), all rows sum to 1.0 ± 1e-3, no NaN."""
    X, y, df = _make_hier_training_data()
    oof = train_oof_hierarchical(X, y, df, n_estimators=20, random_state=42)
    assert oof.shape == (len(df), 3)
    assert not np.isnan(oof).any(), "OOF contains NaN"
    row_sums = oof.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-3), f"row sums != 1: {row_sums}"


# ===========================================================================
# 5. train_oof_hierarchical high accuracy on perfect signal
# ===========================================================================

def test_train_oof_hierarchical_high_accuracy_perfect_signal():
    """With strongly separable CNN signal, accuracy > 0.85."""
    rows = []
    sectors = _SYNTHETIC_TOP_SECTORS
    verdicts = ["likely_false_positive", "uncertain", "planet_candidate"]
    cnn_vals = [0.05, 0.5, 0.95]
    for tic in range(60):
        v_idx = tic % 3
        v = verdicts[v_idx]
        cnn = cnn_vals[v_idx]
        for k in range(3):
            rows.append({
                "tic_id": str(tic), "candidate_num": k + 1,
                "period_days": 2.5, "duration_days": 0.1, "depth_ppm": 500.0,
                "snr": 12.0, "time_span_days": 100.0, "cnn_score": cnn,
                "triceratops_fpp": 0.1, "exominer_score": cnn,
                "exominer_tce_period": 2.5, "n_sectors": 3, "n_datapoints": 5000,
                "exominer_sector_run": sectors[tic % 15],
                "claude_verdict": v,
            })
    df = pd.DataFrame(rows)
    from exonit.verdict_classifier import compute_target
    y = compute_target(df)
    X = preprocess_v2(df, top_sectors=_SYNTHETIC_TOP_SECTORS)
    oof = train_oof_hierarchical(X, y, df, n_estimators=50, random_state=42)
    pred = oof.argmax(axis=1)
    acc = (pred == y).mean()
    assert acc > 0.85, f"accuracy={acc:.3f} < 0.85"


# ===========================================================================
# 6. fit_full_hierarchical returns two models
# ===========================================================================

def test_fit_full_hierarchical_returns_two_models():
    """Both returned models are fitted LightGBM classifiers."""
    import lightgbm as lgb
    X, y, df = _make_hier_training_data()
    stage1, stage2 = fit_full_hierarchical(X, y, n_estimators=20, random_state=42)
    assert isinstance(stage1, lgb.LGBMClassifier)
    assert isinstance(stage2, lgb.LGBMClassifier)
    # Both should be fitted: they have booster_
    assert stage1.booster_ is not None
    assert stage2.booster_ is not None


# ===========================================================================
# 7. save_v2_model + roundtrip
# ===========================================================================

def test_save_v2_model_roundtrip(tmp_path):
    """Saved + reloaded payload preserves stage1/stage2/feature_names/top_sectors
    and predicts identically."""
    X, y, df = _make_hier_training_data()
    stage1, stage2 = fit_full_hierarchical(X, y, n_estimators=20, random_state=42)
    path = tmp_path / "verdict_classifier_v2_test.pkl"
    save_v2_model(stage1, stage2, path,
                  feature_names=list(LGBM_FEATURE_NAMES_V2),
                  top_sectors=_SYNTHETIC_TOP_SECTORS)
    assert path.exists()

    import joblib
    payload = joblib.load(str(path))
    assert "stage1" in payload
    assert "stage2" in payload
    assert payload["feature_names"] == list(LGBM_FEATURE_NAMES_V2)
    assert payload["top_sectors"] == _SYNTHETIC_TOP_SECTORS

    # Predictions must be identical
    row = X.iloc[:1]
    p_orig_s1 = stage1.predict_proba(row)
    p_loaded_s1 = payload["stage1"].predict_proba(row)
    np.testing.assert_allclose(p_orig_s1, p_loaded_s1, atol=1e-9)


# ===========================================================================
# 8. _load_v2_model picks lexically-latest v2 file
# ===========================================================================

def test_load_picks_latest_v2_version(tmp_path, monkeypatch):
    """verdict_classifier_v2_2026-12-31.pkl chosen over ..._2026-01-01.pkl."""
    X, y, df = _make_hier_training_data()
    stage1, stage2 = fit_full_hierarchical(X, y, n_estimators=10, random_state=42)

    old_path = tmp_path / "verdict_classifier_v2_2026-01-01.pkl"
    new_path = tmp_path / "verdict_classifier_v2_2026-12-31.pkl"
    save_v2_model(stage1, stage2, old_path,
                  feature_names=list(LGBM_FEATURE_NAMES_V2),
                  top_sectors=_SYNTHETIC_TOP_SECTORS)
    save_v2_model(stage1, stage2, new_path,
                  feature_names=list(LGBM_FEATURE_NAMES_V2),
                  top_sectors=_SYNTHETIC_TOP_SECTORS)

    monkeypatch.setenv("EXONIT_MODELS_DIR", str(tmp_path))
    _load_v2_model.cache_clear()
    payload = _load_v2_model()
    assert "2026-12-31" in payload["path"], f"expected latest, got: {payload['path']}"


# ===========================================================================
# 9. predict_v2 returns drop-in dict shape
# ===========================================================================

def _save_tiny_v2_model(tmp_path):
    """Train + save a tiny v2 model for inference tests."""
    X, y, df = _make_hier_training_data()
    stage1, stage2 = fit_full_hierarchical(X, y, n_estimators=15, random_state=42)
    path = tmp_path / "verdict_classifier_v2_2026-04-29.pkl"
    save_v2_model(stage1, stage2, path,
                  feature_names=list(LGBM_FEATURE_NAMES_V2),
                  top_sectors=_SYNTHETIC_TOP_SECTORS)
    return path


def _sample_candidate():
    return {
        "period_days": 2.5, "depth_ppm": 500, "duration_days": 0.1,
        "snr": 12.0, "cnn_score": 0.8, "triceratops_fpp": 0.1,
        "exominer_score": 0.9, "exominer_tce_period": 2.5,
        "exominer_sector_run": "s01",
    }


def _sample_star():
    return {
        "tic_id": "1", "tmag": 11.0, "ra": 100.0, "dec": 30.0,
        "n_sectors": 3, "time_span_days": 100.0, "n_datapoints": 5000,
    }


def test_predict_v2_returns_drop_in_dict_shape(tmp_path, monkeypatch):
    """Returns {verdict in CLASS_NAMES, confidence in [0,1], model_used contains
    'verdict_classifier_v2'}."""
    _save_tiny_v2_model(tmp_path)
    monkeypatch.setenv("EXONIT_MODELS_DIR", str(tmp_path))
    _load_v2_model.cache_clear()

    from exonit.verdict_classifier import CLASS_NAMES
    result = predict_v2(_sample_candidate(), _sample_star())
    assert "verdict" in result
    assert "confidence" in result
    assert "model_used" in result
    assert result["verdict"] in CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0
    assert "verdict_classifier_v2" in result["model_used"]


# ===========================================================================
# 10. predict_v2 latency < 5ms
# ===========================================================================

def test_predict_v2_inference_under_5ms(tmp_path, monkeypatch):
    """Inference latency: avg < 5ms per call after warm-up."""
    _save_tiny_v2_model(tmp_path)
    monkeypatch.setenv("EXONIT_MODELS_DIR", str(tmp_path))
    _load_v2_model.cache_clear()

    cand = _sample_candidate()
    star = _sample_star()

    # Warm up (model load + JIT)
    predict_v2(cand, star)

    t0 = time.perf_counter()
    for _ in range(50):
        predict_v2(cand, star)
    elapsed = (time.perf_counter() - t0) / 50
    assert elapsed < 0.005, f"avg per-call {elapsed * 1000:.2f}ms > 5ms"


# ===========================================================================
# 11. predict_verdict routes to v2 when env var set
# ===========================================================================

def test_predict_verdict_routes_to_v2_when_env_var_set(tmp_path, monkeypatch):
    """With EXONIT_VERDICT_VERSION=v2, predict_verdict returns v2 model_used."""
    _save_tiny_v2_model(tmp_path)
    monkeypatch.setenv("EXONIT_VERDICT_VERSION", "v2")
    monkeypatch.setenv("EXONIT_MODELS_DIR", str(tmp_path))
    _load_v2_model.cache_clear()

    from exonit.verdict_classifier import predict_verdict
    result = predict_verdict(_sample_candidate(), vetting_result={}, star_info=_sample_star())
    assert "verdict_classifier_v2" in result["model_used"], (
        f"expected v2 routing, got model_used={result['model_used']!r}"
    )


# ===========================================================================
# 12. predict_verdict defaults to v1
# ===========================================================================

def test_predict_verdict_defaults_to_v1(tmp_path, monkeypatch):
    """Without EXONIT_VERDICT_VERSION, predict_verdict uses v1 behaviour."""
    import lightgbm as lgb
    from exonit.verdict_classifier import (
        LGBM_FEATURE_NAMES, LGBM_PARAMS, predict_verdict, save_model,
        _load_model,
    )
    # Save a minimal v1 model
    rng = np.random.RandomState(0)
    n = 60
    X_v1 = pd.DataFrame({c: rng.rand(n) for c in LGBM_FEATURE_NAMES})
    y_v1 = rng.randint(0, 3, size=n)
    m_v1 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=42)
    m_v1.fit(X_v1, y_v1)
    save_model(m_v1, tmp_path / "verdict_classifier_v1_2026-04-29.pkl",
               feature_names=list(LGBM_FEATURE_NAMES))

    monkeypatch.delenv("EXONIT_VERDICT_VERSION", raising=False)
    monkeypatch.setenv("EXONIT_MODELS_DIR", str(tmp_path))
    _load_model.cache_clear()

    result = predict_verdict(_sample_candidate(), vetting_result={}, star_info=_sample_star())
    assert "verdict_classifier_v2" not in result["model_used"], (
        f"expected v1 behaviour, got model_used={result['model_used']!r}"
    )
    assert "verdict_classifier" in result["model_used"]


# ===========================================================================
# 13. exominer_score=-1 (no-DV cache-negative sentinel) treated same as None
#
# v2 (unlike v2.1) has no exominer_ffi_score fallback feature — the minimum
# consistent fix is that -1 must feed the model IDENTICALLY to a real None
# (both collapse to 0.0), never as an out-of-distribution "-1.0" score. See
# progress/investigation-exominer-minus1-2026-07-02.md.
# ===========================================================================

class _RecordingStage:
    """Fake LightGBM stage: records the feature DataFrame it was called
    with instead of actually predicting — lets the test assert on the exact
    feature values built by ``_predict_single_v2`` without training a real
    model (avoids relying on a model happening to be "sensitive enough" to
    the exominer_score column to show a behavioural difference)."""

    def __init__(self):
        self.frames = []

    def predict_proba(self, X):
        self.frames.append(X.copy())
        return np.array([[0.5, 0.5]])


def test_predict_single_v2_treats_minus1_same_as_none_for_exominer_score():
    feature_names = list(LGBM_FEATURE_NAMES_V2)
    star = _sample_star()

    stage1_none, stage2_none = _RecordingStage(), _RecordingStage()
    cand_none = {**_sample_candidate(), "exominer_score": None}
    _predict_single_v2(stage1_none, stage2_none, cand_none, star,
                       feature_names, _SYNTHETIC_TOP_SECTORS)

    stage1_neg1, stage2_neg1 = _RecordingStage(), _RecordingStage()
    cand_neg1 = {**_sample_candidate(), "exominer_score": -1}
    _predict_single_v2(stage1_neg1, stage2_neg1, cand_neg1, star,
                       feature_names, _SYNTHETIC_TOP_SECTORS)

    pd.testing.assert_frame_equal(stage1_none.frames[0], stage1_neg1.frames[0])
    assert stage1_none.frames[0]["exominer_score"].iloc[0] == 0.0

    # Sanity: a real score is NOT normalized away — must differ from both.
    stage1_real, stage2_real = _RecordingStage(), _RecordingStage()
    cand_real = {**_sample_candidate(), "exominer_score": 0.42}
    _predict_single_v2(stage1_real, stage2_real, cand_real, star,
                       feature_names, _SYNTHETIC_TOP_SECTORS)
    assert stage1_real.frames[0]["exominer_score"].iloc[0] == pytest.approx(0.42)
