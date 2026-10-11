"""Tests for scripts/enqueue_from_csv.py CSV loading helper."""

import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import enqueue_from_csv  # noqa: E402


def test_load_targets_from_csv(tmp_path):
    csv = tmp_path / "sample.csv"
    pd.DataFrame([
        {"tic_id": "1", "role": "recall", "period_days": 5.0,
         "depth_ppm": 1500, "duration_hours": 2.0, "n_observations": 5,
         "disposition": "KP"},
        {"tic_id": "2", "role": "fp_rate", "period_days": 2.0,
         "depth_ppm": 500, "duration_hours": 1.0, "n_observations": 3,
         "disposition": "FP"},
    ]).to_csv(csv, index=False)

    targets = enqueue_from_csv.load_targets(csv)
    assert len(targets) == 2
    assert all(isinstance(t["tic_id"], str) for t in targets)
    assert targets[0]["tic_id"] == "1"


def test_load_targets_skip_missing_tic_id(tmp_path):
    csv = tmp_path / "sample.csv"
    pd.DataFrame([
        {"tic_id": "1", "role": "recall"},
        {"tic_id": None, "role": "recall"},
    ]).to_csv(csv, index=False)
    targets = enqueue_from_csv.load_targets(csv)
    assert len(targets) == 1
    assert targets[0]["tic_id"] == "1"
