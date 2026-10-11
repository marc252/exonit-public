import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from exonit.cnn_ffi_dataset import FFIViewsDataset


@pytest.fixture
def fixture_views(tmp_path):
    h5p = tmp_path / "views.h5"
    pq = tmp_path / "split.parquet"
    rng = np.random.default_rng(0)
    with h5py.File(h5p, "w") as f:
        for rid in range(20):
            g = f.create_group(f"views/{rid}")
            g.create_dataset("global_view", data=rng.standard_normal(201).astype(np.float32))
            g.create_dataset("local_view", data=rng.standard_normal(61).astype(np.float32))
    rows = [{"row_id": rid, "tic_id": rid // 2, "split": "train" if rid < 16 else "val",
             "label_binary": rid % 2, "source": "qlp", "cadence_minutes": 30,
             "is_synthetic": False} for rid in range(20)]
    pd.DataFrame(rows).to_parquet(pq)
    return h5p, pq


def test_dataset_length(fixture_views):
    h5p, pq = fixture_views
    ds = FFIViewsDataset(h5p, pq, split="train")
    assert len(ds) == 16


def test_dataset_item_shape_and_dtype(fixture_views):
    h5p, pq = fixture_views
    ds = FFIViewsDataset(h5p, pq, split="train")
    g, l, y = ds[0]
    assert g.shape == (1, 201) and g.dtype == torch.float32
    assert l.shape == (1, 61) and l.dtype == torch.float32
    assert y.dtype == torch.float32


def test_dataset_class_weights(fixture_views):
    h5p, pq = fixture_views
    ds = FFIViewsDataset(h5p, pq, split="train")
    w = ds.class_weights(clip_ratio=5.0)
    assert len(w) == 2
    assert w[0] > 0 and w[1] > 0
    # ratio capped to 5
    assert max(w) / min(w) <= 5.01
