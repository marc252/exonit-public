"""Tests TIC-blocked + stratified split."""
import numpy as np
import pandas as pd
import pytest

from exonit.cnn_ffi_splits import build_splits


@pytest.fixture
def manifest_fixture():
    rng = np.random.default_rng(0)
    rows = []
    rid = 0
    for tic in range(200):
        for src in ["qlp", "tglc"]:
            for sector_class, cad in [(2, 30), (40, 10), (60, 3)]:
                label = int(rng.uniform() < 0.7)
                rows.append({
                    "row_id": rid, "tic_id": tic, "source": src,
                    "sector": sector_class, "cadence_minutes": cad,
                    "label_binary": label, "view_status": "ok",
                    "is_synthetic": False,
                })
                rid += 1
    return pd.DataFrame(rows)


def test_split_no_tic_leakage(manifest_fixture):
    """No TIC appears in more than one split."""
    splits = build_splits(manifest_fixture, seed=42)
    train_tics = set(splits[splits["split"] == "train"]["tic_id"])
    val_tics = set(splits[splits["split"] == "val"]["tic_id"])
    test_tics = set(splits[splits["split"] == "test"]["tic_id"])
    assert not (train_tics & val_tics)
    assert not (train_tics & test_tics)
    assert not (val_tics & test_tics)


def test_split_proportions_approx_70_15_15(manifest_fixture):
    splits = build_splits(manifest_fixture, seed=42)
    pct = splits["split"].value_counts(normalize=True)
    assert 0.65 <= pct["train"] <= 0.75
    assert 0.10 <= pct["val"] <= 0.20
    assert 0.10 <= pct["test"] <= 0.20


def test_split_stratification_label_present_in_all(manifest_fixture):
    splits = build_splits(manifest_fixture, seed=42)
    for sp in ["train", "val", "test"]:
        labels = splits[splits["split"] == sp]["label_binary"].unique()
        assert set(labels) == {0, 1}, f"split {sp} missing a label"


def test_split_synthetics_no_tic_leakage_with_mixed_rows(manifest_fixture):
    """When a TIC has real + synthetic rows, all go to the same split.

    Original spec said "synthetics only in train", but dataset v1 reuses
    real-label TICs for synthetic injection, so forcing synthetics to
    train would split a TIC across two splits. We keep the harder no-leak
    constraint. G4 anti-synthetic gate is preserved by comparing
    real-only vs full at evaluation time (within the same split).
    """
    df = manifest_fixture.copy()
    # Mark every other row as synthetic — TIC has both real and synthetic rows.
    df.loc[df["row_id"] % 2 == 0, "is_synthetic"] = True
    splits = build_splits(df, seed=42)
    # No TIC across multiple splits, regardless of synthetic flag
    train_tics = set(splits[splits["split"] == "train"]["tic_id"])
    val_tics = set(splits[splits["split"] == "val"]["tic_id"])
    test_tics = set(splits[splits["split"] == "test"]["tic_id"])
    assert not (train_tics & val_tics)
    assert not (train_tics & test_tics)
    assert not (val_tics & test_tics)
    # Each TIC has all its rows in a single split
    tic_splits = splits.groupby("tic_id")["split"].nunique()
    assert (tic_splits == 1).all(), f"{(tic_splits > 1).sum()} TICs span multiple splits"


def test_split_deterministic_with_seed(manifest_fixture):
    s1 = build_splits(manifest_fixture, seed=42)
    s2 = build_splits(manifest_fixture, seed=42)
    pd.testing.assert_frame_equal(s1.sort_values("row_id"), s2.sort_values("row_id"))
