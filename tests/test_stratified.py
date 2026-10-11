"""Tests for exonit.stratified."""

import pytest
import pandas as pd
from pathlib import Path

from exonit.stratified import stratified_sample


def _mk_catalog(n_kp=40, n_cp=40, n_fp=30, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_kp):
        rows.append({
            "tic_id": f"K{i}", "disposition": "KP",
            "period_days": float(rng.uniform(0.5, 50)),
            "depth_ppm": float(rng.uniform(100, 20000)),
            "duration_hours": 2.0, "n_observations": int(rng.integers(1, 30)),
        })
    for i in range(n_cp):
        rows.append({
            "tic_id": f"C{i}", "disposition": "CP",
            "period_days": float(rng.uniform(0.5, 50)),
            "depth_ppm": float(rng.uniform(100, 20000)),
            "duration_hours": 2.0, "n_observations": int(rng.integers(1, 30)),
        })
    for i in range(n_fp):
        rows.append({
            "tic_id": f"F{i}", "disposition": "FP",
            "period_days": float(rng.uniform(0.5, 50)),
            "depth_ppm": float(rng.uniform(100, 20000)),
            "duration_hours": 2.0, "n_observations": int(rng.integers(1, 30)),
        })
    return pd.DataFrame(rows)


def test_stratified_sample_deterministic_with_seed():
    cat = _mk_catalog()
    a = stratified_sample(cat, n=50, seed=42)
    b = stratified_sample(cat, n=50, seed=42)
    pd.testing.assert_frame_equal(
        a.sort_values("tic_id").reset_index(drop=True),
        b.sort_values("tic_id").reset_index(drop=True),
    )


def test_stratified_sample_has_both_roles():
    cat = _mk_catalog()
    sample = stratified_sample(cat, n=50, seed=1)
    assert "role" in sample.columns
    roles = set(sample["role"].unique())
    assert roles == {"recall", "fp_rate"}
    assert (sample["role"] == "recall").sum() > 0
    assert (sample["role"] == "fp_rate").sum() > 0


def test_stratified_sample_role_dispositions():
    cat = _mk_catalog()
    sample = stratified_sample(cat, n=50, seed=2)
    recall_rows = sample[sample["role"] == "recall"]
    fp_rows = sample[sample["role"] == "fp_rate"]
    assert set(recall_rows["disposition"]).issubset({"KP", "CP"})
    assert (fp_rows["disposition"] == "FP").all()


def test_stratified_sample_approx_ratio():
    cat = _mk_catalog()
    sample = stratified_sample(cat, n=100, seed=3)
    n_recall = (sample["role"] == "recall").sum()
    n_fp = (sample["role"] == "fp_rate").sum()
    assert 70 <= n_recall <= 90
    assert 10 <= n_fp <= 30


def test_stratified_sample_on_real_toi():
    csv = Path("data/toi_catalog.csv")
    if not csv.exists():
        pytest.skip("toi_catalog.csv missing")
    from exonit.scorecard import load_toi_catalog
    cat = load_toi_catalog(csv)
    sample = stratified_sample(cat, n=100, seed=42)
    assert len(sample) <= 100
    assert "role" in sample.columns
