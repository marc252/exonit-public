"""ExoNit stratified sampling from TOI catalog."""

import numpy as np
import pandas as pd

from exonit.scorecard import bin_depth, bin_period

RECALL_DISPOSITIONS = ("KP", "CP")
FP_DISPOSITION = "FP"
DEFAULT_RECALL_FRACTION = 0.80


def _assign_bin_keys(df):
    """Add (depth_bin, period_bin) columns, skipping rows with NaN values."""
    df = df.copy()
    df = df.dropna(subset=["period_days", "depth_ppm"])
    df["depth_bin"] = df["depth_ppm"].apply(bin_depth)
    df["period_bin"] = df["period_days"].apply(bin_period)
    df["bin_key"] = df["depth_bin"] + "|" + df["period_bin"]
    return df


def _sample_balanced(df, n, rng):
    """Sample n rows roughly balanced across bin_key, reproducibly."""
    if len(df) == 0 or n <= 0:
        return df.iloc[0:0]
    if len(df) <= n:
        return df.sample(frac=1, random_state=rng).reset_index(drop=True)
    groups = df.groupby("bin_key")
    bins = list(groups.groups.keys())
    if not bins:
        return df.sample(n=n, random_state=rng).reset_index(drop=True)
    per_bin = max(1, n // len(bins))
    picked = []
    for _, group in groups:
        take = min(per_bin, len(group))
        picked.append(group.sample(n=take, random_state=rng))
    picked_df = pd.concat(picked)
    if len(picked_df) < n:
        leftover = df.drop(picked_df.index)
        more = leftover.sample(
            n=min(n - len(picked_df), len(leftover)),
            random_state=rng,
        )
        picked_df = pd.concat([picked_df, more])
    elif len(picked_df) > n:
        picked_df = picked_df.sample(n=n, random_state=rng)
    return picked_df.reset_index(drop=True)


def stratified_sample(catalog, n=100, seed=42,
                      recall_fraction=DEFAULT_RECALL_FRACTION):
    """Return a stratified sample DataFrame with added 'role' column.

    role ∈ {'recall', 'fp_rate'}.
    recall rows come from KP/CP dispositions, balanced across depth×period bins.
    fp_rate rows come from FP disposition.
    """
    rng = np.random.default_rng(seed)

    n_recall = int(round(n * recall_fraction))
    n_fp = n - n_recall

    recall_pool = catalog[catalog["disposition"].isin(RECALL_DISPOSITIONS)]
    recall_pool = _assign_bin_keys(recall_pool)
    recall_sample = _sample_balanced(recall_pool, n_recall, rng)
    recall_sample = recall_sample.copy()
    recall_sample["role"] = "recall"

    fp_pool = catalog[catalog["disposition"] == FP_DISPOSITION]
    fp_pool = _assign_bin_keys(fp_pool)
    fp_sample = _sample_balanced(fp_pool, n_fp, rng)
    fp_sample = fp_sample.copy()
    fp_sample["role"] = "fp_rate"

    out = pd.concat([recall_sample, fp_sample], ignore_index=True)
    out = out.drop(columns=["bin_key"], errors="ignore")
    return out
