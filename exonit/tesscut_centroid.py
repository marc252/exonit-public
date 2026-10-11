"""Shared TESScut-centroid helpers (B8 dedup, 2026-07-02).

``scripts/tesscut_centroid_backfill.py`` (primary sweep) and
``scripts/tesscut_centroid_retry.py`` (fallback pass for the primary sweep's
failures) both download a TESScut TPF cutout and compute a per-cadence
center-of-light centroid, then write it into the shared
``data/training/ffi_dataset_v1.h5``. They used to carry independent copies
of ``_compute_com`` and, worse, diverged on the HDF5 dataset name used for
the per-cadence centroid-quality flag: backfill wrote ``quality`` (guarded —
only created if the group doesn't already have one from ingestion), retry
wrote ``quality_centroid`` — a name no reader ever consumes. That was a live
data-consistency bug, not just duplication.

Evidence for which name is correct (2026-07-02):
  * Writers at ingestion time (``scripts/build_ffi_dataset.py:263,878``) and
    the TGLC-native centroid backfill (``scripts/backfill_tglc_centroid_to_dataset.py:127-128``)
    both use ``quality``.
  * The only readers of this data
    (``scripts/build_exominer_ffi_dataset.py::_load_v1_lc`` and
    ``scripts/build_exominer_ffi_dataset_v2.py::_load_v1_lc``) read
    ``g["quality"]`` and pass it as the centroid-quality mask to
    ``extract_centroid_xy`` — they never look for ``quality_centroid``.
  * ``grep -rn "quality_centroid"`` across the repo turns up exactly one
    writer (the old ``tesscut_centroid_retry.py::_write_to_h5``) and zero
    readers.
  * Inspecting the live ``data/training/ffi_dataset_v1.h5`` (read-only,
    2026-07-02) confirms this in practice: of the first 5000 TGLC groups
    with an ``x_cent`` dataset (i.e. touched by one of the two centroid
    scripts), 5000/5000 carry ``quality`` and 1739/5000 *additionally* carry
    a dead ``quality_centroid`` dataset from a past retry run that
    ``_load_v1_lc`` silently ignores — those 1739 rows fall back to the
    stale ingestion-time flux-quality mask for their centroid instead of the
    freshly computed centroid-quality mask.

Fix: both scripts import the single implementation below, which always
writes under ``quality`` (guarded, matching the established ingestion
contract) and never under any other name.
"""
from __future__ import annotations

import threading
from pathlib import Path

import h5py
import numpy as np

_h5_lock = threading.Lock()


def compute_com(tpf):
    """Compute center-of-light centroid per cadence from a TESScut TPF.

    Returns (time, col, row, quality) numpy arrays. col/row are
    median-subtracted so they're zero-centred (matches the Phase 3 model).
    quality[i] == 1 marks a cadence where the cutout had no positive flux
    (center-of-light undefined for that cadence).
    """
    flux = np.asarray(tpf.flux, dtype=float)  # (cadences, rows, cols)
    if flux.ndim != 3:
        return None
    n_t = flux.shape[0]
    if n_t == 0:
        return None
    rows_idx = np.arange(flux.shape[1])
    cols_idx = np.arange(flux.shape[2])
    rg, cg = np.meshgrid(rows_idx, cols_idx, indexing="ij")
    out_col = np.zeros(n_t, dtype=np.float32)
    out_row = np.zeros(n_t, dtype=np.float32)
    out_q = np.zeros(n_t, dtype=np.int32)
    for i in range(n_t):
        f = flux[i]
        if not np.isfinite(f).all():
            f = np.where(np.isfinite(f), f, 0.0)
        s = f.sum()
        if s <= 0:
            out_q[i] = 1
            continue
        out_col[i] = float((f * cg).sum() / s)
        out_row[i] = float((f * rg).sum() / s)
    # Median-subtract to zero-centre (clean cadences only)
    good = out_q == 0
    if good.any():
        out_col[good] -= float(np.median(out_col[good]))
        out_row[good] -= float(np.median(out_row[good]))
    t = np.asarray(tpf.time.value, dtype=float)
    return t, out_col, out_row, out_q


def write_centroid_to_h5(h5_path: Path, hdf5_key: str, time_arr, col, row_arr, q) -> bool:
    """Atomically add x_cent/y_cent (and ``quality``, if not already present)
    to the row's group. Returns False if the group doesn't exist in the
    dataset (caller should count that as an error, not a silent skip).

    ``quality`` is written guarded (only if absent) because virtually every
    group already carries the ingestion-time flux-quality mask
    (``scripts/build_ffi_dataset.py``); overwriting it with the (different)
    centroid-quality semantics would regress the established reader
    contract. NEVER write the centroid-quality flag under any other name —
    see the module docstring (B8, 2026-07-02): a prior divergent
    ``quality_centroid`` name silently orphaned ~2000 rows' centroid-quality
    data because no reader consumes it.
    """
    key = hdf5_key.lstrip("/")
    with _h5_lock:
        with h5py.File(h5_path, "a") as h5:
            if key not in h5:
                return False
            g = h5[key]
            if "x_cent" in g:
                del g["x_cent"]
            if "y_cent" in g:
                del g["y_cent"]
            g.create_dataset("x_cent", data=col.astype(np.float32))
            g.create_dataset("y_cent", data=row_arr.astype(np.float32))
            if "quality" not in g:
                g.create_dataset("quality", data=q.astype(np.int32))
    return True


# Backwards-compatible underscore aliases: both scripts called these as
# ``_compute_com`` / ``_write_to_h5`` throughout their bodies (same
# convention as exonit/tglc_fetch.py's D11 dedup), so importers can bind the
# underscore names to these implementations without touching every call site.
_compute_com = compute_com
_write_to_h5 = write_centroid_to_h5
