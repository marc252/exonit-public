"""Native port merging the two TFRecord-writing normalizers.

This module replaces two oracle sources:

  * ``src_preprocessing/diff_img/preprocessing/normalize_diff_img.py`` —
    per-image (per-example) min-max and mad-std normalization of the
    difference-image features.  That oracle path writes the normalized
    feature back into a TFRecord using a different suffix naming
    convention (``_minmaxnorm`` / ``_stdnorm``).  The **ExoMiner model we
    serve** does not consume those features (it consumes the
    ``_std_trainset`` outputs from ``normalize_data_tfrecords.py``), so
    this file only exposes these two helpers for completeness /
    diagnostics.  They are **not** called by :func:`normalize_example`.

  * ``src_preprocessing/normalize_tfrecord_dataset/normalize_data_tfrecords.py``
    — full training-set-statistics normalization of scalar features,
    centroid views, and difference images.  This is the stage whose
    output feeds directly into the ONNX graph, so everything the model
    needs is implemented here, pure numpy.

Trimmed from the oracle sources:
  - ``tf.train.Example`` / ``tf.io.TFRecordWriter`` / ``TFRecordDataset``
    — the native worker keeps tensors in memory.
  - ``multiprocessing.Pool`` wrappers and the ``__main__`` entry points
    (``normalize_examples_main`` / ``preprocess_tfrecs_...``).
  - YAML config loading (``normalize_examples_main``): the worker
    receives an in-memory config dict.
  - ``example_util`` helpers (``set_tensor_feature`` / ``get_feature``):
    all data is already a plain dict by the time it reaches us.
  - Kepler-only pathways: TESS only.

Public API:
  * :func:`load_norm_stats` — read the three ``.npy`` files shipped in
    ``worker/exominer_native/data/norm_stats/`` into a dict shaped like
    ``{'scalar_params': {...}, 'centroid': {...}, 'diff_img': {...}}``.
  * :func:`normalize_example` — apply the full training-set-statistics
    normalization to one example dict and return a new dict with the
    ``_norm`` / ``_std_clip`` / ``_std_noclip`` / ``_std_trainset`` /
    ``_minmaxnorm_trainset`` / ``_fixed_min_max_norm`` features keyed as
    the ONNX graph expects.

Diagnostic helpers (kept for compatibility with the per-image oracle
path, not called by the main entry point):
  * :func:`perimage_min_max_normalize_diff_img`
  * :func:`perimage_std_normalize_diff_img`

Internal helpers re-exposed for parity tests:
  * :func:`centering_and_normalization` — ``(view - centre) / (norm + NORM_SIGMA_EPS)``.
  * :func:`normalize_scalar_parameters`
  * :func:`normalize_centroid`
  * :func:`normalize_diff_img`

Parity strategy: the oracle entry points read/write TFRecords, so the
gate exercises the inner pure-compute helpers (``centering_and_normalization``,
``normalize_scalar_parameters``, ``normalize_diff_img`` on a synthetic
numpy-only dict that mimics the fields an ``Example`` would carry).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from astropy.stats import mad_std


_log = logging.getLogger(__name__)


# Kept byte-exact with the oracle (``lc_preprocessing/phase_fold_and_binning.py``).
NORM_SIGMA_EPS = 1e-12

# Module-level defaults kept byte-exact with the oracle.
DEFAULT_ZERO_DIVISION_EPS = 1e-10
DEFAULT_SCALAR_EPSILON = 1e-32

# Fixed min-max divisor for the ``neighbors_imgs`` path; see
# ``normalize_data_tfrecords.normalize_diff_img``.
NEIGHBORS_IMGS_MAX_MAG_RATIO = 5

# Default per-image tensor shape for ``diff_imgs`` / ``oot_imgs`` /
# ``snr_imgs`` / ``neighbors_imgs`` — matches the ExoMiner model input
# of ``(num_sectors, H, W) = (5, 33, 33)``.
DEFAULT_DIFF_IMG_DATA_SHAPE = (5, 33, 33)

# File names shipped under ``worker/exominer_native/data/norm_stats/``.
_NORM_STATS_FILENAMES = {
    "scalar_params": "train_scalarparam_norm_stats.npy",
    "centroid": "train_centroid_norm_stats.npy",
    "diff_img": "train_diffimg_norm_stats.npy",
}


def _report_exclusion(msg: str) -> None:
    """Worker-local stand-in for the oracle's ``report_exclusion``."""
    _log.warning(str(msg))


# ---------------------------------------------------------------------------
# Inner pure-compute helper — byte-exact re-export from the oracle.
# ---------------------------------------------------------------------------


def centering_and_normalization(view, val_centr, val_norm, **kwargs):
    """Centre and normalize a 1-D time series.

    Byte-exact port of
    ``src_preprocessing/lc_preprocessing/phase_fold_and_binning.py::
    centering_and_normalization`` (sans the exclusion-log side-effect when
    ``val_norm == 0`` — we only warn).
    """
    if val_norm == 0:
        _report_exclusion(
            "Dividing view by 0. Returning the non-normalized view."
        )
    return (np.asarray(view) - val_centr) / (val_norm + NORM_SIGMA_EPS)


# ---------------------------------------------------------------------------
# Loading norm-stats from disk.
# ---------------------------------------------------------------------------


def load_norm_stats(path: Any) -> dict:
    """Load training-set normalization statistics from a directory.

    Expects the three ``.npy`` files shipped in
    ``worker/exominer_native/data/norm_stats/`` (committed, 28 KB total).
    Each file contains a pickled dict; we load them under the keys
    the oracle ``normalize_examples_main`` uses internally.

    Args:
        path: str or Path, directory containing the three ``.npy`` files.

    Returns:
        dict with top-level keys ``scalar_params``, ``centroid``,
        ``diff_img``. Keys whose file is missing are omitted (the
        downstream :func:`normalize_example` gates on key presence).
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"norm_stats directory not found: {path}")

    norm_stats: dict = {}
    for key, fname in _NORM_STATS_FILENAMES.items():
        fp = path / fname
        if fp.exists():
            norm_stats[key] = np.load(fp, allow_pickle=True).item()
        else:
            _log.debug("norm_stats file missing: %s", fp)
    return norm_stats


# ---------------------------------------------------------------------------
# Scalar parameter normalization.
#
# Byte-exact port of ``normalize_data_tfrecords.normalize_scalar_parameters``,
# modulo the TFRecord ``get_feature`` shim: the native worker already
# passes a plain dict, so we read ``example[scalar_param]`` directly.
# ---------------------------------------------------------------------------


def normalize_scalar_parameters(example: dict,
                                normStatsScalars: dict,
                                epsilon: float = DEFAULT_SCALAR_EPSILON) -> dict:
    """Normalize the scalar features of ``example`` per ``normStatsScalars``.

    Args:
        example: dict, ``{feature_name: numpy_array_or_scalar}``.  For
            each ``scalar_param`` in ``normStatsScalars`` the value is
            read from ``example[scalar_param]``.  Missing keys yield
            ``NaN`` (so the caller can pre-detect gaps).
        normStatsScalars: dict, as produced by the oracle's scalar
            statistics dump.  Each inner dict carries ``median``,
            ``mad_std`` and ``info`` (with ``missing_value``,
            ``replace_value``, ``log_transform``, ``log_transform_eps``,
            ``clip_factor``, ``standardize``).
        epsilon: float, denominator stabiliser for the final standardize
            step.  Matches the oracle default (``1e-32``).

    Returns:
        dict, ``{f"{scalar_param}_norm": [value]}``.  Values are wrapped
        in single-element lists to mirror the oracle's
        ``set_float_feature`` 1-D convention.
    """
    norm_scalar_feat = {f"{scalar_param}_norm": np.nan for scalar_param in normStatsScalars}

    for scalarParam in normStatsScalars:
        if scalarParam not in example:
            # Oracle uses ``get_feature`` which returns [] for missing
            # features; we mirror that by leaving the NaN placeholder.
            continue

        scalarParamVal = np.array(example[scalarParam])
        # Squeeze length-1 containers into a plain scalar so the
        # downstream math + ``~np.isfinite`` check behave correctly.
        if scalarParamVal.shape == (1,):
            scalarParamVal = scalarParamVal[0]
        elif scalarParamVal.ndim == 0:
            scalarParamVal = scalarParamVal.item()

        replace_flag = False
        if normStatsScalars[scalarParam]["info"]["missing_value"] is not None:
            if scalarParam in ["wst_depth"]:
                replace_flag = scalarParamVal < \
                               normStatsScalars[scalarParam]["info"]["missing_value"]
            else:
                replace_flag = scalarParamVal == \
                               normStatsScalars[scalarParam]["info"]["missing_value"]

        if ~np.isfinite(scalarParamVal):
            replace_flag = True

        if replace_flag:
            if normStatsScalars[scalarParam]["info"]["replace_value"] is not None:
                scalarParamVal = normStatsScalars[scalarParam]["info"]["replace_value"]
            else:
                scalarParamVal = normStatsScalars[scalarParam]["median"]

        # log10 transform (skipping values that were replaced with the
        # already-log-transformed median — matches oracle).
        if normStatsScalars[scalarParam]["info"]["log_transform"] and \
                not (replace_flag and scalarParamVal == normStatsScalars[scalarParam]["median"]):
            if not np.isnan(normStatsScalars[scalarParam]["info"]["log_transform_eps"]):
                scalarParamVal += normStatsScalars[scalarParam]["info"]["log_transform_eps"]
            scalarParamVal = np.log10(scalarParamVal)

        # clipping
        if not np.isnan(normStatsScalars[scalarParam]["info"]["clip_factor"]):
            scalarParamVal = np.clip(
                [scalarParamVal],
                normStatsScalars[scalarParam]["median"] -
                normStatsScalars[scalarParam]["info"]["clip_factor"] *
                normStatsScalars[scalarParam]["mad_std"],
                normStatsScalars[scalarParam]["median"] +
                normStatsScalars[scalarParam]["info"]["clip_factor"] *
                normStatsScalars[scalarParam]["mad_std"],
            )[0]

        # standardize
        if normStatsScalars[scalarParam]["info"]["standardize"]:
            scalarParamVal = (scalarParamVal - normStatsScalars[scalarParam]["median"]) / \
                             (normStatsScalars[scalarParam]["mad_std"] + epsilon)

        norm_scalar_feat[f"{scalarParam}_norm"] = [scalarParamVal]

    return norm_scalar_feat


# ---------------------------------------------------------------------------
# Centroid view normalization.
#
# Byte-exact port of ``normalize_data_tfrecords.normalize_centroid``,
# minus the ``example.features.feature[...].float_list.value`` access.
# ---------------------------------------------------------------------------


def normalize_centroid(example: dict, normStatsCentroid: dict) -> dict:
    """Normalize centroid views for ``example``.

    Expects ``example`` to carry ``global_centr_view``, ``local_centr_view``
    and ``local_centr_view_var`` as 1-D numpy arrays.  Optionally
    ``global_centr_view_adjscl`` / ``local_centr_view_adjscl`` when
    ``normStatsCentroid`` includes ``_adjscl_median`` keys.
    """
    norm_centroid_feat: dict = {}

    glob_centr_view = np.asarray(example["global_centr_view"])
    loc_centr_view = np.asarray(example["local_centr_view"])
    loc_centr_view_var = np.asarray(example["local_centr_view_var"])

    # 1) clipping to physically meaningful distance in arcsec
    glob_centr_view_std_clip = np.clip(
        glob_centr_view,
        a_max=normStatsCentroid["global_centr_view"]["clip_value"],
        a_min=None,
    )
    assert normStatsCentroid["global_centr_view"]["std_clip"] > 0
    glob_centr_view_std_clip = centering_and_normalization(
        glob_centr_view_std_clip,
        np.median(glob_centr_view_std_clip),
        normStatsCentroid["global_centr_view"]["std_clip"],
    )

    loc_centr_view_std_clip = np.clip(
        loc_centr_view,
        a_max=normStatsCentroid["local_centr_view"]["clip_value"],
        a_min=None,
    )
    assert normStatsCentroid["local_centr_view"]["std_clip"] > 0
    loc_centr_view_std_clip = centering_and_normalization(
        loc_centr_view_std_clip,
        np.median(loc_centr_view_std_clip),
        normStatsCentroid["local_centr_view"]["std_clip"],
    )
    norm_centroid_feat.update({
        "global_centr_view_std_clip": glob_centr_view_std_clip,
        "local_centr_view_std_clip": loc_centr_view_std_clip,
    })

    # 2) no clipping
    assert normStatsCentroid["global_centr_view"]["std"] > 0
    glob_centr_view_std_noclip = centering_and_normalization(
        glob_centr_view,
        np.median(glob_centr_view),
        normStatsCentroid["global_centr_view"]["std"],
    )
    assert normStatsCentroid["local_centr_view"]["std"] > 0
    loc_centr_view_std_noclip = centering_and_normalization(
        loc_centr_view,
        np.median(loc_centr_view),
        normStatsCentroid["local_centr_view"]["std"],
    )
    loc_centr_view_std_noclip_var = loc_centr_view_var / normStatsCentroid["local_centr_view"]["std"]
    norm_centroid_feat.update({
        "global_centr_view_std_noclip": glob_centr_view_std_noclip,
        "local_centr_view_std_noclip": loc_centr_view_std_noclip,
        "local_centr_view_std_noclip_var": loc_centr_view_std_noclip_var,
    })

    # 4) adjusted-scale centroid (TESS)
    if "global_centr_view_adjscl_median" in normStatsCentroid:
        glob_centr_view_adjscl = np.asarray(example["global_centr_view_adjscl"])
        loc_centr_view_adjscl = np.asarray(example["local_centr_view_adjscl"])
        assert normStatsCentroid["global_centr_view"]["std_adjscl"] > 0
        glob_centr_view_std_noclip_adjscl = centering_and_normalization(
            glob_centr_view_adjscl,
            np.median(glob_centr_view_adjscl),
            normStatsCentroid["global_centr_view"]["std_adjscl"],
        )
        assert normStatsCentroid["local_centr_view"]["std_adjscl"] > 0
        loc_centr_view_std_noclip_adjscl = centering_and_normalization(
            loc_centr_view_adjscl,
            np.median(loc_centr_view_adjscl),
            normStatsCentroid["local_centr_view"]["std_adjscl"],
        )
        norm_centroid_feat.update({
            "global_centr_view_std_noclip_adjscl": glob_centr_view_std_noclip_adjscl,
            "local_centr_view_std_noclip_adjscl": loc_centr_view_std_noclip_adjscl,
        })

    return norm_centroid_feat


# ---------------------------------------------------------------------------
# Difference-image normalization with training-set statistics.
# ---------------------------------------------------------------------------


def normalize_diff_img(example: dict,
                       normStatsDiff_img: dict,
                       imgs_dims: tuple = DEFAULT_DIFF_IMG_DATA_SHAPE,
                       zero_division_eps: float = DEFAULT_ZERO_DIVISION_EPS) -> dict:
    """Training-set-stats normalize the difference-image features.

    Byte-exact port of ``normalize_data_tfrecords.normalize_diff_img``.
    The oracle reads tensors out of a TFRecord (``parse_tensor`` +
    ``reshape(imgs_dims)``); here we read them from an in-memory dict
    that is already a ``(N, H, W)`` float32 array (or equivalent).

    Features consumed (all optional, via membership check):
        ``diff_imgs``, ``oot_imgs``, ``snr_imgs``, ``neighbors_imgs``
    and their ``_tc`` target-centered counterparts.

    Returns:
        dict with keys
            ``<name>_minmaxnorm_trainset`` (skipping ``neighbors_imgs``),
            ``<name>_std_trainset`` (all present images),
            ``<name>_fixed_min_max_norm`` (only ``neighbors_imgs*``).
    """
    norm_diff_img_feat: dict = {}

    target_centered_prefix_feature_lst = ["", "_tc"]

    for target_centered_prefix_feature in target_centered_prefix_feature_lst:

        data_example = {}

        for base_name in ("diff_imgs", "oot_imgs", "snr_imgs", "neighbors_imgs"):
            key = f"{base_name}{target_centered_prefix_feature}"
            if key in example:
                # Accept both list-of-arrays (from diff_img.py) and a
                # stacked ``(N, H, W)`` numpy array.
                arr = np.asarray(example[key], dtype=np.float32).reshape(imgs_dims)
                data_example[key] = arr

        # min-max normalization (skip neighbors_imgs)
        for img_type, img_data in data_example.items():
            if "neighbors_imgs" in img_type:
                continue
            img_data_minmaxn = np.array(img_data)
            img_data_minmaxn = (
                (img_data_minmaxn - normStatsDiff_img[img_type]["min"]) /
                (normStatsDiff_img[img_type]["max"] -
                 normStatsDiff_img[img_type]["min"] + zero_division_eps)
            )
            norm_diff_img_feat[f"{img_type}_minmaxnorm_trainset"] = img_data_minmaxn

        # standardization (only for images whose stats dict actually
        # carries median/std — neighbors_imgs is normalized via the
        # fixed-divisor branch below).
        for img_type, img_data in data_example.items():
            if img_type not in normStatsDiff_img:
                continue
            if "median" not in normStatsDiff_img[img_type] \
                    or "std" not in normStatsDiff_img[img_type]:
                continue
            img_data_std = np.array(img_data)
            img_data_std = (
                (img_data_std - normStatsDiff_img[img_type]["median"]) /
                (normStatsDiff_img[img_type]["std"] + zero_division_eps)
            )
            norm_diff_img_feat[f"{img_type}_std_trainset"] = img_data_std

        # fixed min-max (neighbors_imgs only)
        for img_type, img_data in data_example.items():
            if "neighbors_imgs" not in img_type:
                continue
            img_data_minmaxn_fixed = np.array(img_data)
            img_data_minmaxn_fixed = img_data_minmaxn_fixed / NEIGHBORS_IMGS_MAX_MAG_RATIO
            norm_diff_img_feat[f"{img_type}_fixed_min_max_norm"] = img_data_minmaxn_fixed

    return norm_diff_img_feat


# ---------------------------------------------------------------------------
# Per-image (per-example) diff-image normalization helpers.
#
# These mirror ``normalize_diff_img/normalize_diff_img.py::
# perimage_normalization_diffimg``.  They are NOT used by the ExoMiner
# ONNX graph we serve — that graph consumes the ``_std_trainset``
# features from ``normalize_diff_img``.  We expose them anyway for
# diagnostic/plot parity with the oracle pipeline.
# ---------------------------------------------------------------------------


def perimage_min_max_normalize_diff_img(feature_data_example: np.ndarray,
                                        zero_division_eps: float = DEFAULT_ZERO_DIVISION_EPS) -> np.ndarray:
    """Per-image min-max normalization ``(img - min) / (max - min + eps)``.

    Input: ``(N, H, W)`` float array.  NaNs are ignored when computing
    per-image min/max.  Byte-exact with the oracle's
    ``perimage_normalization_diffimg`` min-max branch.
    """
    x = np.array(feature_data_example)
    xmin = np.nanmin(x, axis=(1, 2), keepdims=True)
    xmax = np.nanmax(x, axis=(1, 2), keepdims=True)
    return (x - xmin) / (xmax - xmin + zero_division_eps)


def perimage_std_normalize_diff_img(feature_data_example: np.ndarray,
                                    zero_division_eps: float = DEFAULT_ZERO_DIVISION_EPS) -> np.ndarray:
    """Per-image standardization with MAD-std denominator.

    Input: ``(N, H, W)`` float array.  Byte-exact with the oracle's
    ``perimage_normalization_diffimg`` standardize branch.
    """
    x = np.array(feature_data_example)
    median = np.nanmedian(x, axis=(1, 2), keepdims=True)
    madstd = np.expand_dims(mad_std(x, axis=(1, 2), ignore_nan=True), axis=(1, 2))
    return (x - median) / (madstd + zero_division_eps)


# ---------------------------------------------------------------------------
# Top-level public entry point.
# ---------------------------------------------------------------------------


def normalize_example(features: dict,
                      norm_stats: dict,
                      diff_img_shape: tuple = DEFAULT_DIFF_IMG_DATA_SHAPE) -> dict:
    """Apply training-set-statistics normalization to a single example.

    This is the dict-based replacement for the oracle's
    ``normalize_data_tfrecords.normalize_examples`` per-example body.
    The oracle reads the example from a TFRecord, applies three
    sub-normalizers conditionally, then writes the normalized features
    back as new TFRecord fields.  We take a dict in and return a dict of
    the new, normalized features (the caller is free to merge or keep
    them separate).

    Args:
        features: dict, the intermediate example dict produced by the
            upstream preprocessors (``lc_*``, ``diff_img``,
            ``phase_fold``, ``centroid``, etc.).  Unknown keys are
            ignored; normalizers only touch the keys they recognise.
        norm_stats: dict, pre-loaded normalization statistics as
            returned by :func:`load_norm_stats`.  Top-level keys
            ``scalar_params``, ``centroid`` and ``diff_img`` act as
            gates — absent keys skip the corresponding sub-normalizer.
        diff_img_shape: tuple, dimensions of each diff-image tensor.
            Defaults to the ExoMiner ``(5, 33, 33)``.

    Returns:
        dict of normalized features.  Keys follow the oracle naming
        convention (``<feature>_norm``, ``<view>_std_clip``,
        ``<view>_std_noclip``, ``<img>_std_trainset``,
        ``<img>_minmaxnorm_trainset``, ``<img>_fixed_min_max_norm``).
    """
    normalized: dict = {}

    if "scalar_params" in norm_stats:
        normalized.update(
            normalize_scalar_parameters(features, norm_stats["scalar_params"])
        )

    if "centroid" in norm_stats and "global_centr_view" in features:
        normalized.update(
            normalize_centroid(features, norm_stats["centroid"])
        )

    if "diff_img" in norm_stats and any(
        k in features for k in ("diff_imgs", "diff_imgs_tc",
                                "oot_imgs", "oot_imgs_tc",
                                "snr_imgs", "snr_imgs_tc",
                                "neighbors_imgs", "neighbors_imgs_tc")
    ):
        normalized.update(
            normalize_diff_img(features, norm_stats["diff_img"], diff_img_shape)
        )

    return normalized
