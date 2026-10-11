"""PyTorch Dataset for FFI views v1."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FFIViewsDataset(Dataset):
    def __init__(self, h5_path, split_parquet, split: str, sources=None):
        """sources: optional iterable of source names to keep (e.g. ['qlp','synthetic'])."""
        self.h5_path = Path(h5_path)
        df = pd.read_parquet(split_parquet)
        rows = df[df["split"] == split]
        if sources:
            rows = rows[rows["source"].isin(sources)]
        self.rows = rows.reset_index(drop=True)
        self._h5 = None  # lazy open per worker

    def __len__(self):
        return len(self.rows)

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        h5 = self._open()
        rid = int(row["row_id"])
        g = np.array(h5[f"views/{rid}/global_view"], dtype=np.float32)
        l = np.array(h5[f"views/{rid}/local_view"], dtype=np.float32)
        y = float(row["label_binary"])
        return (
            torch.from_numpy(g).unsqueeze(0),
            torch.from_numpy(l).unsqueeze(0),
            torch.tensor(y, dtype=torch.float32),
        )

    def class_weights(self, clip_ratio: float = 5.0) -> np.ndarray:
        """Inverse-frequency weights, clipped to ratio."""
        counts = self.rows["label_binary"].value_counts().sort_index()
        inv = 1.0 / counts.values
        inv = inv / inv.min()
        inv = np.minimum(inv, clip_ratio)
        return inv
