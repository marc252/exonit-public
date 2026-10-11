"""PyTorch Dataset for ExoMiner-FFI training (Phase 3c)."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ExoMinerFFIDataset(Dataset):
    """Yields (global, local, centroid, mask, stellar, tce, aux, label) per row.

    Single-channel view tensors are 1×N (channel dim added); aux is 3×N
    (odd-local, even-local, secondary-local). label is float32 scalar.
    The H5 is opened lazily on first access so DataLoader workers each
    get their own handle.
    """

    def __init__(self, h5_path, splits_parquet, split: str):
        self.h5_path = Path(h5_path)
        df = pd.read_parquet(splits_parquet)
        self.rows = df[df["split"] == split].reset_index(drop=True)
        self._h5 = None  # lazy

    def __len__(self):
        return len(self.rows)

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx):
        h5 = self._open()
        row = self.rows.iloc[idx]
        rid = int(row["h5_idx"])
        g = np.asarray(h5["global_view"][rid], dtype=np.float32)
        l = np.asarray(h5["local_view"][rid], dtype=np.float32)
        c = np.asarray(h5["centroid_view"][rid], dtype=np.float32)
        mask = np.asarray([h5["centroid_mask"][rid]], dtype=np.float32)
        stellar = np.asarray(h5["stellar"][rid], dtype=np.float32)
        tce = np.asarray(h5["tce"][rid], dtype=np.float32)
        if "aux_views" in h5:
            aux = np.asarray(h5["aux_views"][rid], dtype=np.float32)
        else:  # backward-compat with pre-aux datasets
            aux = np.zeros((3, l.shape[-1]), dtype=np.float32)
        y = float(row["label_binary"])
        return (
            torch.from_numpy(g).unsqueeze(0),
            torch.from_numpy(l).unsqueeze(0),
            torch.from_numpy(c).unsqueeze(0),
            torch.from_numpy(mask),
            torch.from_numpy(stellar),
            torch.from_numpy(tce),
            torch.from_numpy(aux),
            torch.tensor(y, dtype=torch.float32),
        )

    def class_weights(self, clip_ratio: float = 5.0) -> np.ndarray:
        counts = self.rows["label_binary"].value_counts().sort_index()
        inv = 1.0 / counts.values
        inv = inv / inv.min()
        inv = np.minimum(inv, clip_ratio)
        return inv

    def tmag_bin_balanced_weights(self, h5_path=None, bins=None,
                                  clip: float = 8.0) -> np.ndarray:
        """Per-row sampling weights that decorrelate the stellar tmag from the
        label (v7b tmag-shortcut fix, stage 2).

        Within each tmag bin, up/down-weight so the bin's effective
        positive:negative ratio matches the GLOBAL ratio. A model trained on the
        reweighted distribution sees the SAME pos-fraction in every tmag bin, so
        tmag carries no label information — without dropping the feature (the
        model can still use tmag where it is genuinely informative, e.g. noise
        floor, as v6 does). Weights are clipped to ``clip`` to avoid blowing up
        on the very sparse faint-positive / bright-negative cells.

        tmag is recovered from the stellar branch (stellar[:,0] = (tmag-12)/2.5);
        rows with a missing tmag (norm==0) fall in the centred-mean bin, which is
        fine — production always supplies a real tmag, and these are the synthetic
        positives the model already handles.
        """
        import h5py
        if bins is None:
            bins = [-np.inf, 11, 12, 13, 14, 15, np.inf]
        path = Path(h5_path) if h5_path is not None else self.h5_path
        with h5py.File(path, "r") as f:
            tmag_norm = f["stellar"][:, 0]
        rids = self.rows["h5_idx"].to_numpy()
        tmag = tmag_norm[rids] * 2.5 + 12.0
        y = self.rows["label_binary"].to_numpy().astype(int)
        binidx = np.digitize(tmag, bins[1:-1])  # 0..len(bins)-2
        g_pos = float(y.mean())
        g_neg = 1.0 - g_pos
        w = np.ones(len(y), dtype=np.float64)
        for b in np.unique(binidx):
            m = binidx == b
            bp = float(y[m].mean()) if m.any() else g_pos
            bp = min(max(bp, 1e-4), 1 - 1e-4)
            # weight positives by g_pos/bp, negatives by g_neg/(1-bp)
            wp = g_pos / bp
            wn = g_neg / (1.0 - bp)
            w[m & (y == 1)] = wp
            w[m & (y == 0)] = wn
        w = np.clip(w, 1.0 / clip, clip)
        # normalise mean weight to 1 so the effective dataset size is unchanged
        w = w / w.mean()
        return w.astype(np.float64)
