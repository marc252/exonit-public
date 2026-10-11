"""TIC-blocked + triple-stratified train/val/test splitter.

See spec §3.2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def _strat_key(row):
    return f"L{row['label_binary']}|S{row['source']}|C{row['cadence_minutes']}"


def _cadence_class(c):
    if c >= 30: return "30"
    if c >= 10: return "10"
    return "200s"


def build_splits(manifest: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Split rows into train/val/test, blocking by TIC and stratifying.

    The split is determined per TIC based on the dominant strat_key of its
    REAL rows. ALL rows of a TIC (real + synthetic) go to the same split,
    so no TIC ever appears in more than one split.

    The dataset v1 reuses real-label TICs for synthetic injection, so
    "synthetics only in train" (the original spec) would force splitting a
    TIC across two splits. We instead keep the harder constraint (no TIC
    leakage) and let synthetics flow with their TIC. The G4 anti-synthetic
    gate is still meaningful: at evaluation, the test set's real-only
    subset is compared against the test set's full subset (real+synthetic).

    TICs with synthetic-only rows (none in v1, but defensive): default to train.
    """
    rng = np.random.default_rng(seed)
    df = manifest.copy()

    if "is_synthetic" not in df.columns:
        df["is_synthetic"] = False

    real = df[~df["is_synthetic"]].copy()
    real["strat_key"] = real.apply(_strat_key, axis=1)

    # Per TIC (looking at real rows only), dominant strat_key (mode)
    tic_strat = real.groupby("tic_id")["strat_key"].agg(lambda s: s.mode().iloc[0])

    out_split_per_tic = {}
    for strat, tics in tic_strat.groupby(tic_strat).groups.items():
        tics = list(tics)
        rng.shuffle(tics)
        n = len(tics)
        n_train = max(1, int(round(n * TRAIN_FRAC)))
        n_val = max(1 if n >= 3 else 0, int(round(n * VAL_FRAC)))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = max(0, n - n_train)
        for t in tics[:n_train]:
            out_split_per_tic[t] = "train"
        for t in tics[n_train:n_train + n_val]:
            out_split_per_tic[t] = "val"
        for t in tics[n_train + n_val:]:
            out_split_per_tic[t] = "test"

    # Apply split to ALL rows (real + synthetic) by their TIC.
    # TICs with no real rows (synthetic-only — none in v1) default to "train".
    df["split"] = df["tic_id"].map(out_split_per_tic).fillna("train")
    # RESERVED TOI hold-out must never train — route those TICs to 'reserved'.
    from exonit.reserved_holdout import reassign_reserved_split
    df = reassign_reserved_split(df, "tic_id", "split", label="cnn_ffi.build_splits")
    return df[["row_id", "tic_id", "split", "label_binary", "source", "cadence_minutes", "is_synthetic"]]
