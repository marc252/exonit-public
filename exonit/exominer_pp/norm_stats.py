"""Normalization statistics for ExoMiner++ 2.0 FFI.

NASA's public Zenodo TFRecord release ships *pre-normalization* records and
does NOT include the per-feature training-set normalization statistics
needed to turn them into model inputs (see ``progress/exominerpp_setup.md``
§0). Those statistics are, however, **bundled inside the published Docker
image** ``ghcr.io/nasa/exominer:latest`` at
``/app/exominer_pipeline/data/norm_stats/``. We extract them once and vendor
the three small ``.npy`` files into ``data/exominer_pp/norm_stats/`` — this
is the resolution of the "missing normalization stats" blocker.

The three files (each an ``np.load(..., allow_pickle=True).item()`` dict):

  * ``train_scalarparam_norm_stats.npy`` — 53 scalar features, each
    ``{'median', 'mad_std', 'info': {...}}`` (covers all 33 ExoMiner++
    scalar inputs).
  * ``train_centroid_norm_stats.npy`` — global/local centroid view
    ``{'median','std','clip_value',...}``.
  * ``train_diffimg_norm_stats.npy`` — per-image-type ``{'median','std',
    'min','max'}``. These are SCALAR stats applied element-wise, so they are
    image-size-independent: the same numbers normalize the 2-min 33x33 and
    the FFI 55x55 images identically.

Known gap: the bundled diff-image stats cover ``diff_imgs(_tc)`` and
``oot_imgs(_tc)`` but NOT ``snr_imgs_tc`` (the published 2-min config does
not use an SNR image branch). The FFI new-architecture model *does* have an
``snr_imgs_tc_std_trainset`` input. ``derive_snr_imgs_tc_stats`` derives the
missing median/std the same way NASA computes them — from the raw
``snr_imgs_tc`` field in the released FFI eval TFRecords — so full-fidelity
FFI scoring is reproducible once the eval shards are extracted.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_THIS = Path(__file__).resolve()
DEFAULT_NORM_STATS_DIR = _THIS.parents[2] / "data" / "exominer_pp" / "norm_stats"

SCALAR_FILE = "train_scalarparam_norm_stats.npy"
CENTROID_FILE = "train_centroid_norm_stats.npy"
DIFFIMG_FILE = "train_diffimg_norm_stats.npy"

# Image types whose diff-image stat must be present for a full-fidelity
# ExoMiner++ FFI forward pass (neighbors use a fixed min-max, no stat).
REQUIRED_DIFFIMG_KEYS = ("diff_imgs_tc", "oot_imgs_tc", "snr_imgs_tc")


class NormStatsError(RuntimeError):
    """Raised when normalization statistics are missing or incomplete."""


def norm_stats_dir() -> Path:
    """Resolve the norm-stats dir (env override ``EXONIT_EXOMINER_PP_NORM_STATS``)."""
    env = os.environ.get("EXONIT_EXOMINER_PP_NORM_STATS")
    return Path(env) if env else DEFAULT_NORM_STATS_DIR


def _load_npy_dict(path: Path) -> Dict:
    if not path.exists():
        raise NormStatsError(
            f"norm-stats file missing: {path}\n"
            "Extract from the NASA Docker image with:\n"
            "  cid=$(docker create ghcr.io/nasa/exominer:latest)\n"
            "  docker cp $cid:/app/exominer_pipeline/data/norm_stats/. "
            f"{path.parent}/\n"
            "  docker rm $cid"
        )
    return np.load(path, allow_pickle=True).item()


def load_norm_stats(stats_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """Load the three bundled normalization-stat dicts.

    Returns ``{'scalar_params': ..., 'centroid': ..., 'diff_img': ...}`` in
    the exact structure NASA's ``normalize_data_tfrecords.py`` consumes.
    """
    d = stats_dir or norm_stats_dir()
    return {
        "scalar_params": _load_npy_dict(d / SCALAR_FILE),
        "centroid": _load_npy_dict(d / CENTROID_FILE),
        "diff_img": _load_npy_dict(d / DIFFIMG_FILE),
    }


def check_completeness(stats: Dict[str, Dict]) -> Dict[str, bool]:
    """Report which required diff-image stats are present.

    Returns ``{image_type: bool}`` for REQUIRED_DIFFIMG_KEYS. ``snr_imgs_tc``
    is the one the bundled 2-min stats are missing.
    """
    diff = stats.get("diff_img", {})
    return {k: (k in diff) for k in REQUIRED_DIFFIMG_KEYS}


def derive_snr_imgs_tc_stats(
    eval_tfrecord_dir: Path,
    max_examples: Optional[int] = None,
) -> Dict[str, float]:
    """Derive ``snr_imgs_tc`` {median, std, min, max} from FFI eval TFRecords.

    Computed exactly as NASA's ``compute_normalization_stats_tfrecords.py``
    does for the other image types: median and (population) std over all
    finite pixels of the raw ``snr_imgs_tc`` field, plus min/max. The raw
    field is present in the released *pre-normalization* eval TFRecords.

    Args:
        eval_tfrecord_dir: directory of extracted ``shard-*`` eval records.
        max_examples: cap on examples scanned (None = all). A few thousand
            examples already give a stable median/MAD.

    Returns:
        ``{'median', 'std', 'min', 'max'}`` floats. Caller can splice this
        into the loaded ``diff_img`` stats dict under key ``'snr_imgs_tc'``.

    Raises:
        NormStatsError if no shards / no snr_imgs_tc field found.
    """
    try:
        import tensorflow as tf  # lazy
    except ImportError as e:  # pragma: no cover
        raise NormStatsError("tensorflow required to derive snr stats") from e

    shards = sorted(
        p for p in Path(eval_tfrecord_dir).glob("*shard-*") if p.suffix != ".csv"
    )
    if not shards:
        raise NormStatsError(f"no shard-* TFRecords under {eval_tfrecord_dir}")

    vals = []
    n = 0
    for shard in shards:
        for rec in tf.data.TFRecordDataset(str(shard)).as_numpy_iterator():
            ex = tf.train.Example()
            ex.ParseFromString(rec)
            feat = ex.features.feature
            if "snr_imgs_tc" not in feat:
                continue
            arr = np.asarray(feat["snr_imgs_tc"].float_list.value, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                vals.append(arr)
            n += 1
            if max_examples is not None and n >= max_examples:
                break
        if max_examples is not None and n >= max_examples:
            break

    if not vals:
        raise NormStatsError(
            f"no snr_imgs_tc field found in {len(shards)} shard(s) under "
            f"{eval_tfrecord_dir}"
        )

    allpix = np.concatenate(vals)
    return {
        "median": float(np.median(allpix)),
        "std": float(np.std(allpix)),
        "min": float(np.min(allpix)),
        "max": float(np.max(allpix)),
    }
