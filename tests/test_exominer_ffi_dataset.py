"""Unit tests for ExoMinerFFIDataset."""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import torch


def _make_fixture(tmp_path):
    n = 6
    h5p = tmp_path / "views.h5"
    with h5py.File(h5p, "w") as f:
        f.create_dataset("rows", data=np.arange(n))
        f.create_dataset("global_view", data=np.random.randn(n, 201).astype(np.float32))
        f.create_dataset("local_view", data=np.random.randn(n, 61).astype(np.float32))
        f.create_dataset("centroid_view", data=np.random.randn(n, 61).astype(np.float32))
        f.create_dataset("centroid_mask", data=np.array([0, 1, 0, 1, 0, 1], dtype=np.int8))
        f.create_dataset("stellar", data=np.random.randn(n, 5).astype(np.float32))
        f.create_dataset("tce", data=np.random.randn(n, 4).astype(np.float32))

    splits_p = tmp_path / "splits.parquet"
    pd.DataFrame({
        "h5_idx": list(range(n)),
        "tic_id": [str(100 + i) for i in range(n)],
        "label": ["PC"] * 3 + ["FP"] * 3,
        "label_binary": [1] * 3 + [0] * 3,
        "split": ["train"] * 4 + ["val"] * 1 + ["test"] * 1,
        "centroid_mask": [0, 1, 0, 1, 0, 1],
    }).to_parquet(splits_p)

    return h5p, splits_p


def test_dataset_returns_all_inputs(tmp_path):
    from exonit.exominer_ffi.dataset import ExoMinerFFIDataset

    h5p, sp = _make_fixture(tmp_path)
    ds = ExoMinerFFIDataset(h5p, sp, split="train")
    assert len(ds) == 4
    item = ds[0]
    g, l, c, mask, stellar, tce, aux, y = item
    assert g.shape == (1, 201)
    assert l.shape == (1, 61)
    assert c.shape == (1, 61)
    assert mask.shape == (1,)
    assert stellar.shape == (5,)
    assert tce.shape == (4,)
    assert aux.shape == (3, 61)
    assert isinstance(y, torch.Tensor)


def test_dataset_class_weights(tmp_path):
    from exonit.exominer_ffi.dataset import ExoMinerFFIDataset

    h5p, sp = _make_fixture(tmp_path)
    ds = ExoMinerFFIDataset(h5p, sp, split="train")
    w = ds.class_weights()
    assert w.shape == (2,)
    assert w.min() == 1.0
