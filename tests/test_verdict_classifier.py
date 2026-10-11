"""Tests for exonit/verdict_classifier.py — local 3-class verdict classifier."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from exonit.verdict_classifier import (
    EXPECTED_FEATURE_COLS,
    load_dataset,
)


@pytest.fixture(autouse=True)
def _pin_verdict_v1(monkeypatch):
    """predict_verdict dispatches on EXONIT_VERDICT_VERSION. run_nightly.py
    calls load_dotenv() at import time, so any test module that imports it
    leaks the production value (v2.1) into os.environ and these v1-model
    tests fail when run after it. Pin v1 here; tests that exercise other
    versions override with their own setenv."""
    monkeypatch.setenv("EXONIT_VERDICT_VERSION", "v1")



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
            "exominer_tce_period) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (c.get("tic_id"), c.get("candidate_num", 1), c.get("period_days", 2.5),
             c.get("t0", 1000.0), c.get("duration_days", 0.1), c.get("depth_ppm", 500.0),
             c.get("snr", 12.0), c.get("cnn_score", 0.8), c.get("triceratops_fpp", 0.1),
             c.get("claude_verdict", "planet_candidate"),
             c.get("exominer_score", 0.9), c.get("exominer_tce_period", 2.5)))
    for s in stars:
        con.execute(
            "INSERT INTO stars(tic_id,ra,dec,tmag,n_sectors,time_span_days,status) "
            "VALUES (?,?,?,?,?,?,?)",
            (s["tic_id"], s.get("ra", 100.0), s.get("dec", 30.0), s.get("tmag", 11.0),
             s.get("n_sectors", 3), s.get("time_span_days", 100.0), s.get("status", "completed")))
    con.commit()
    con.close()
    return p


def test_load_dataset_returns_labeled_rows(tmp_path):
    db = _make_db(tmp_path,
                  candidates=[
                      {"tic_id": "1", "claude_verdict": "planet_candidate"},
                      {"tic_id": "2", "claude_verdict": "uncertain"},
                      {"tic_id": "3", "claude_verdict": None},  # excluded: NULL verdict
                  ],
                  stars=[{"tic_id": "1"}, {"tic_id": "2"}, {"tic_id": "3"}])
    df = load_dataset(db)
    assert len(df) == 2  # NULL verdict excluded
    assert set(df["claude_verdict"]) == {"planet_candidate", "uncertain"}


def test_load_dataset_joins_stellar_params(tmp_path):
    db = _make_db(tmp_path,
                  candidates=[{"tic_id": "1", "claude_verdict": "planet_candidate"}],
                  stars=[{"tic_id": "1", "ra": 123.45, "dec": -67.89, "tmag": 9.5}])
    df = load_dataset(db)
    assert df.loc[0, "ra"] == 123.45
    assert df.loc[0, "dec"] == -67.89
    assert df.loc[0, "tmag"] == 9.5


def test_load_dataset_has_all_expected_feature_cols(tmp_path):
    db = _make_db(tmp_path,
                  candidates=[{"tic_id": "1", "claude_verdict": "planet_candidate"}],
                  stars=[{"tic_id": "1"}])
    df = load_dataset(db)
    for col in EXPECTED_FEATURE_COLS:
        assert col in df.columns, f"missing {col}"


def test_load_dataset_excludes_unlabeled(tmp_path):
    db = _make_db(tmp_path,
                  candidates=[{"tic_id": "1", "claude_verdict": None}],
                  stars=[{"tic_id": "1"}])
    df = load_dataset(db)
    assert len(df) == 0


import numpy as np

from exonit.verdict_classifier import compute_target, CLASS_NAMES


def test_compute_target_lfp_to_zero(tmp_path):
    db = _make_db(tmp_path,
                  candidates=[{"tic_id": "1", "claude_verdict": "likely_false_positive"}],
                  stars=[{"tic_id": "1"}])
    df = load_dataset(db)
    y = compute_target(df)
    assert list(y) == [0]


def test_compute_target_uncertain_to_one(tmp_path):
    db = _make_db(tmp_path,
                  candidates=[{"tic_id": "1", "claude_verdict": "uncertain"}],
                  stars=[{"tic_id": "1"}])
    df = load_dataset(db)
    y = compute_target(df)
    assert list(y) == [1]


def test_compute_target_planet_candidate_to_two(tmp_path):
    db = _make_db(tmp_path,
                  candidates=[{"tic_id": "1", "claude_verdict": "planet_candidate"}],
                  stars=[{"tic_id": "1"}])
    df = load_dataset(db)
    y = compute_target(df)
    assert list(y) == [2]


def test_compute_target_unknown_raises(tmp_path):
    db = _make_db(tmp_path,
                  candidates=[{"tic_id": "1", "claude_verdict": "weird_value"}],
                  stars=[{"tic_id": "1"}])
    df = load_dataset(db)
    with pytest.raises(ValueError, match="unknown verdict"):
        compute_target(df)


def test_class_names_order():
    assert CLASS_NAMES == ("likely_false_positive", "uncertain", "planet_candidate")


import math

from exonit.verdict_classifier import preprocess_lgbm, LGBM_FEATURE_NAMES


def test_preprocess_lgbm_returns_expected_columns(tmp_path):
    db = _make_db(tmp_path,
                  candidates=[{"tic_id": "1", "claude_verdict": "planet_candidate"}],
                  stars=[{"tic_id": "1"}])
    df = load_dataset(db)
    X = preprocess_lgbm(df)
    assert list(X.columns) == list(LGBM_FEATURE_NAMES)


def test_preprocess_lgbm_preserves_nan_exominer(tmp_path):
    db = _make_db(tmp_path,
                  candidates=[{"tic_id": "1", "claude_verdict": "planet_candidate",
                               "exominer_score": float("nan")}],
                  stars=[{"tic_id": "1"}])
    df = load_dataset(db)
    X = preprocess_lgbm(df)
    assert math.isnan(X.loc[0, "exominer_score"])


def test_preprocess_lgbm_tri_negative_to_nan_with_flag(tmp_path):
    db = _make_db(tmp_path,
                  candidates=[
                      {"tic_id": "1", "claude_verdict": "planet_candidate", "triceratops_fpp": -1},
                      {"tic_id": "2", "claude_verdict": "uncertain", "triceratops_fpp": 0.4},
                  ],
                  stars=[{"tic_id": "1"}, {"tic_id": "2"}])
    df = load_dataset(db)
    X = preprocess_lgbm(df)
    assert math.isnan(X.loc[0, "triceratops_fpp"])
    assert X.loc[0, "tri_valid"] == 0
    assert X.loc[1, "triceratops_fpp"] == pytest.approx(0.4)
    assert X.loc[1, "tri_valid"] == 1


def test_preprocess_lgbm_log_transforms_finite(tmp_path):
    db = _make_db(tmp_path,
                  candidates=[{"tic_id": "1", "claude_verdict": "planet_candidate",
                               "period_days": 2.5, "depth_ppm": 500, "snr": 12,
                               "exominer_tce_period": 2.5}],
                  stars=[{"tic_id": "1", "time_span_days": 100}])
    df = load_dataset(db)
    X = preprocess_lgbm(df)
    assert np.isfinite(X.loc[0, "period_log"])
    assert np.isfinite(X.loc[0, "depth_log1p"])
    assert np.isfinite(X.loc[0, "snr_log1p"])
    assert np.isfinite(X.loc[0, "exominer_tce_period_log"])
    assert np.isfinite(X.loc[0, "time_span_days_log1p"])


from exonit.verdict_classifier import preprocess_logistic, LOGISTIC_FEATURE_NAMES


def test_preprocess_logistic_no_nan(tmp_path):
    db = _make_db(tmp_path,
                  candidates=[
                      {"tic_id": "1", "claude_verdict": "planet_candidate",
                       "exominer_score": float("nan")},
                      {"tic_id": "2", "claude_verdict": "uncertain", "triceratops_fpp": -1},
                  ],
                  stars=[{"tic_id": "1"}, {"tic_id": "2"}])
    df = load_dataset(db)
    X = preprocess_logistic(df)
    assert not X.isna().any().any()


def test_preprocess_logistic_adds_missing_flags(tmp_path):
    db = _make_db(tmp_path,
                  candidates=[
                      {"tic_id": "1", "claude_verdict": "planet_candidate",
                       "exominer_score": float("nan")},
                      {"tic_id": "2", "claude_verdict": "uncertain", "exominer_score": 0.5},
                  ],
                  stars=[{"tic_id": "1"}, {"tic_id": "2"}])
    df = load_dataset(db)
    X = preprocess_logistic(df)
    assert "exominer_score_was_missing" in X.columns
    assert X.loc[0, "exominer_score_was_missing"] == 1
    assert X.loc[1, "exominer_score_was_missing"] == 0


def test_preprocess_logistic_columns_match_constant(tmp_path):
    db = _make_db(tmp_path,
                  candidates=[{"tic_id": "1", "claude_verdict": "planet_candidate"}],
                  stars=[{"tic_id": "1"}])
    df = load_dataset(db)
    X = preprocess_logistic(df)
    assert list(X.columns) == list(LOGISTIC_FEATURE_NAMES)


from exonit.verdict_classifier import cv_split, train_oof_lgbm, train_oof_logistic


def _multi_tic_df_for_cv(tmp_path):
    """Build a 100-TIC × 3 candidates synthetic dataset balanced across classes."""
    cands = []
    stars = []
    for tic in range(100):
        if tic < 10:
            verdict = "likely_false_positive"
        elif tic < 50:
            verdict = "uncertain"
        else:
            verdict = "planet_candidate"
        for k in range(3):
            cands.append({"tic_id": str(tic), "candidate_num": k+1,
                          "claude_verdict": verdict})
        stars.append({"tic_id": str(tic)})
    return _make_db(tmp_path, candidates=cands, stars=stars)


def test_cv_split_no_tic_leakage(tmp_path):
    db = _multi_tic_df_for_cv(tmp_path)
    df = load_dataset(db)
    y = compute_target(df)
    folds = list(cv_split(df, y, n_splits=5, random_state=42))
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        train_tics = set(df.iloc[train_idx]["tic_id"])
        test_tics = set(df.iloc[test_idx]["tic_id"])
        assert train_tics.isdisjoint(test_tics), f"leakage: {train_tics & test_tics}"


def test_cv_split_deterministic(tmp_path):
    db = _multi_tic_df_for_cv(tmp_path)
    df = load_dataset(db)
    y = compute_target(df)
    a = [tuple(t.tolist()) for _, t in cv_split(df, y, n_splits=5, random_state=42)]
    b = [tuple(t.tolist()) for _, t in cv_split(df, y, n_splits=5, random_state=42)]
    assert a == b


def test_train_oof_lgbm_returns_proba_matrix(tmp_path):
    db = _multi_tic_df_for_cv(tmp_path)
    df = load_dataset(db)
    y = compute_target(df)
    X = preprocess_lgbm(df)
    oof = train_oof_lgbm(X, y, df, random_state=42)
    assert oof.shape == (len(df), 3)
    assert not np.isnan(oof).any()
    row_sums = oof.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-3)


def test_train_oof_logistic_returns_proba_matrix(tmp_path):
    db = _multi_tic_df_for_cv(tmp_path)
    df = load_dataset(db)
    y = compute_target(df)
    X = preprocess_logistic(df)
    oof = train_oof_logistic(X, y, df, random_state=42)
    assert oof.shape == (len(df), 3)
    assert not np.isnan(oof).any()


def test_train_oof_lgbm_high_accuracy_on_perfect_signal(tmp_path):
    """LGBM with perfect signal in cnn_score should reach high accuracy."""
    cands = []
    stars = []
    for tic in range(60):
        verdict = "likely_false_positive" if tic < 20 else (
            "uncertain" if tic < 40 else "planet_candidate")
        cnn = 0.05 if tic < 20 else (0.5 if tic < 40 else 0.95)
        for k in range(3):
            cands.append({"tic_id": str(tic), "candidate_num": k+1,
                          "claude_verdict": verdict, "cnn_score": cnn})
        stars.append({"tic_id": str(tic)})
    db = _make_db(tmp_path, candidates=cands, stars=stars)
    df = load_dataset(db)
    y = compute_target(df)
    X = preprocess_lgbm(df)
    oof = train_oof_lgbm(X, y, df, random_state=42)
    pred = oof.argmax(axis=1)
    accuracy = (pred == y).mean()
    assert accuracy > 0.85


from exonit.verdict_classifier import compute_metrics


def test_compute_metrics_perfect_prediction():
    y = np.array([0, 1, 2, 0, 1, 2])
    proba = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                      [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    m = compute_metrics(y, proba)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["agreement"] == pytest.approx(1.0)
    assert m["f1_macro"] == pytest.approx(1.0)
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_compute_metrics_partial():
    y = np.array([0, 1, 2])
    proba = np.array([[0.1, 0.8, 0.1], [0, 1, 0], [0, 0, 1]], dtype=float)
    m = compute_metrics(y, proba)
    assert m["accuracy"] == pytest.approx(2/3)
    assert m["agreement"] == pytest.approx(2/3)


def test_compute_metrics_returns_per_class_f1():
    y = np.array([0, 1, 2, 0, 1, 2])
    proba = np.eye(3)[[0, 1, 2, 0, 1, 2]].astype(float)
    m = compute_metrics(y, proba)
    assert "f1_per_class" in m
    assert len(m["f1_per_class"]) == 3


import joblib
import lightgbm as lgb

from exonit.verdict_classifier import (
    predict_verdict,
    save_model,
    _build_predict_dataframe,
    LGBM_PARAMS,
)


def _train_tiny_lgbm():
    """Helper: train a 1-shot LGBM on synthetic data for predict tests."""
    rng = np.random.RandomState(0)
    n = 60
    X = pd.DataFrame({c: rng.rand(n) for c in LGBM_FEATURE_NAMES})
    y = rng.randint(0, 3, size=n)
    model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=42)
    model.fit(X, y)
    return model


def test_predict_verdict_returns_drop_in_dict_shape(tmp_path, monkeypatch):
    model = _train_tiny_lgbm()
    save_model(model, tmp_path / "verdict_classifier_v1.pkl",
               feature_names=list(LGBM_FEATURE_NAMES))
    monkeypatch.setenv("EXONIT_MODELS_DIR", str(tmp_path))
    # clear cached model from previous tests
    from exonit import verdict_classifier as vc
    vc._load_model.cache_clear()

    candidate = {"period_days": 2.5, "depth_ppm": 500, "duration_days": 0.1,
                 "snr": 12, "cnn_score": 0.8, "triceratops_fpp": 0.1,
                 "exominer_score": 0.9, "exominer_tce_period": 2.5}
    star_info = {"tic_id": "1", "tmag": 11.0, "ra": 100.0, "dec": 30.0,
                 "n_sectors": 3, "time_span_days": 100.0}
    result = predict_verdict(candidate, vetting_result={"passed": True},
                             star_info=star_info)
    assert "verdict" in result
    assert "confidence" in result
    assert "model_used" in result
    assert result["verdict"] in CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0
    assert "verdict_classifier" in result["model_used"]


def test_predict_verdict_no_claude_api_key_required(tmp_path, monkeypatch):
    model = _train_tiny_lgbm()
    save_model(model, tmp_path / "verdict_classifier_v1.pkl",
               feature_names=list(LGBM_FEATURE_NAMES))
    monkeypatch.setenv("EXONIT_MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("EXONIT_USE_CLAUDE_FALLBACK", raising=False)
    from exonit import verdict_classifier as vc
    vc._load_model.cache_clear()

    candidate = {"period_days": 2.5, "depth_ppm": 500, "duration_days": 0.1,
                 "snr": 12, "cnn_score": 0.8, "triceratops_fpp": 0.1,
                 "exominer_score": 0.9, "exominer_tce_period": 2.5}
    result = predict_verdict(candidate, vetting_result={"passed": True},
                             star_info={"tic_id": "1", "tmag": 11.0, "ra": 100.0,
                                        "dec": 30.0, "n_sectors": 3,
                                        "time_span_days": 100.0})
    assert result is not None  # works without Claude key


import time


def test_model_persistence_roundtrip(tmp_path, monkeypatch):
    model = _train_tiny_lgbm()
    save_model(model, tmp_path / "verdict_classifier_v1.pkl",
               feature_names=list(LGBM_FEATURE_NAMES))
    monkeypatch.setenv("EXONIT_MODELS_DIR", str(tmp_path))
    from exonit import verdict_classifier as vc
    vc._load_model.cache_clear()

    payload = vc._load_model()
    assert payload["feature_names"] == list(LGBM_FEATURE_NAMES)
    assert payload["class_names"] == list(CLASS_NAMES)
    # Same model gives same predictions
    rng = np.random.RandomState(0)
    X = pd.DataFrame({c: rng.rand(5) for c in LGBM_FEATURE_NAMES})
    p_orig = model.predict_proba(X)
    p_loaded = payload["model"].predict_proba(X)
    np.testing.assert_allclose(p_orig, p_loaded, atol=1e-9)


def test_load_picks_latest_version(tmp_path, monkeypatch):
    model = _train_tiny_lgbm()
    save_model(model, tmp_path / "verdict_classifier_v1_2026-01-01.pkl",
               feature_names=list(LGBM_FEATURE_NAMES))
    save_model(model, tmp_path / "verdict_classifier_v1_2026-12-31.pkl",
               feature_names=list(LGBM_FEATURE_NAMES))
    monkeypatch.setenv("EXONIT_MODELS_DIR", str(tmp_path))
    from exonit import verdict_classifier as vc
    vc._load_model.cache_clear()

    payload = vc._load_model()
    assert "v1_2026-12-31" in payload["path"]


def test_load_v1_ignores_v2_files(tmp_path, monkeypatch):
    """v1 _load_model must not pick up v2 model files (different schema)."""
    model = _train_tiny_lgbm()
    save_model(model, tmp_path / "verdict_classifier_v1_2026-01-01.pkl",
               feature_names=list(LGBM_FEATURE_NAMES))
    # Drop a v2-shaped file (different schema) — v1 must skip it.
    import joblib
    joblib.dump({"stage1": "fake", "stage2": "fake", "top_sectors": []},
                str(tmp_path / "verdict_classifier_v2_2099-12-31.pkl"))
    monkeypatch.setenv("EXONIT_MODELS_DIR", str(tmp_path))
    from exonit import verdict_classifier as vc
    vc._load_model.cache_clear()

    payload = vc._load_model()
    assert "v1_2026-01-01" in payload["path"]


def test_predict_inference_under_5ms(tmp_path, monkeypatch):
    model = _train_tiny_lgbm()
    save_model(model, tmp_path / "verdict_classifier_v1.pkl",
               feature_names=list(LGBM_FEATURE_NAMES))
    monkeypatch.setenv("EXONIT_MODELS_DIR", str(tmp_path))
    from exonit import verdict_classifier as vc
    vc._load_model.cache_clear()

    candidate = {"period_days": 2.5, "depth_ppm": 500, "duration_days": 0.1,
                 "snr": 12, "cnn_score": 0.8, "triceratops_fpp": 0.1,
                 "exominer_score": 0.9, "exominer_tce_period": 2.5}
    star_info = {"tic_id": "1", "tmag": 11.0, "ra": 100.0, "dec": 30.0,
                 "n_sectors": 3, "time_span_days": 100.0}
    # Warm cache (model load + first JIT)
    vc.predict_verdict(candidate, {"passed": True}, star_info)
    # Time 50 calls
    t0 = time.perf_counter()
    for _ in range(50):
        vc.predict_verdict(candidate, {"passed": True}, star_info)
    elapsed = (time.perf_counter() - t0) / 50
    assert elapsed < 0.005, f"avg per-call {elapsed*1000:.2f}ms > 5ms"


# --- B4: EXONIT_VERDICT_VERSION routing (2026-07-02 cleanup review) --------
#
# predict_verdict used to silently fall through to v1 for any version string
# it did not recognise (e.g. "v2.2" before this branch existed). These tests
# lock in: (a) v2.2 dispatches to verdict_classifier_v2_2.predict_v2_2, and
# (b) an unknown version raises ValueError instead of masquerading as v1.

def test_predict_verdict_routes_v2_2(monkeypatch):
    """EXONIT_VERDICT_VERSION=v2.2 dispatches to predict_v2_2 (mocked — no
    need to load the real .pkl for a routing test)."""
    monkeypatch.setenv("EXONIT_VERDICT_VERSION", "v2.2")
    import exonit.verdict_classifier_v2_2 as vc22
    from exonit.verdict_classifier import predict_verdict

    calls = []

    def _fake_predict_v2_2(candidate, star_info=None, vetting_result=None):
        calls.append((candidate, star_info, vetting_result))
        return {"verdict": "planet_candidate", "confidence": 0.9,
                "model_used": "verdict_classifier_v2_2_local (fake.pkl)"}

    monkeypatch.setattr(vc22, "predict_v2_2", _fake_predict_v2_2)

    candidate = {"period_days": 2.5}
    star_info = {"tic_id": "1"}
    result = predict_verdict(candidate, vetting_result={"passed": True},
                             star_info=star_info)

    assert result["verdict"] == "planet_candidate"
    assert len(calls) == 1
    assert calls[0][0] is candidate
    assert calls[0][1] == star_info


def test_predict_verdict_unknown_version_raises(monkeypatch):
    """A version string that maps to no known model raises instead of
    silently defaulting to v1 (the B4 bug)."""
    from exonit.verdict_classifier import predict_verdict
    monkeypatch.setenv("EXONIT_VERDICT_VERSION", "v2.2-typo")
    with pytest.raises(ValueError, match="v2.2-typo"):
        predict_verdict({"period_days": 2.5}, vetting_result={"passed": True},
                        star_info={"tic_id": "1"})
