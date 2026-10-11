"""Native port of ``src_preprocessing/diff_img/preprocessing/preprocess_diff_img.py``.

Scope: preprocessing of extracted difference-image data into the per-TCE
tensor dict that the ExoMiner model consumes.  In the container this module
writes a ``.npy`` snapshot and subsequently feeds it to a TFRecord builder
(``add_data_to_tfrecords.py``) and then to a TFRecord-based normalizer
(``normalize_diff_img.py`` / ``normalize_tfrecord_dataset``).  The native
worker path skips TFRecords entirely:

  * ``preprocess_tce_diff_img`` takes the already-loaded difference-image
    dict for a single TCE (as produced by :mod:`.dv_xml`) and returns a
    pure ``dict[str, Any]`` with the per-image tensors keyed the way the
    downstream normalizer (C.14) expects.  No file I/O, no logger setup,
    no multiprocessing.
  * :func:`preprocess_diff_img_tces_dict` is a thin convenience wrapper
    that runs the single-TCE function over a mapping of ``tce_uid`` →
    diff-image-data and returns a flat ``dict[tce_uid, preprocessed_dict]``.

Helper functions (``preprocess_single_diff_img_data_for_example`` and the
compute primitives imported from ``utils_diff_img``) are copied verbatim
from the oracle source, adjusted only for sibling imports.

Trimmed from the container source:
  - The ``tf.io.TFRecordWriter`` / ``tf.train.Example`` path (the native
    worker feeds tensors straight to ONNX Runtime).
  - The ``multiprocessing.Pool`` + file-glob entry points
    (``preprocess_diff_img_tces_main``): the worker runs one TIC at a time.
  - ``plot_diff_img_data`` is already stubbed out in ``utils_diff_img``
    portion below; the ``plot_prob > 0`` branch is unreachable because we
    never pass a non-zero probability.
  - Kepler support: TESS only for the worker (``mission_name='tess'``).
  - ``report_exclusion`` from ``utils_preprocessing_io`` → local log-only
    stub following the convention used in :mod:`.phase_fold`.

Patches vs oracle imports:
  - ``from src_preprocessing.diff_img.preprocessing.utils_diff_img import *``
    → helpers inlined below (scipy.signal.convolve2d + PIL.Image only).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image
from scipy.signal import convolve2d


_log = logging.getLogger(__name__)


CENTER_OPTIONS = ("target_not_centered", "target_centered")


def _report_exclusion(msg: str) -> None:
    """Worker-local stand-in for the oracle's ``report_exclusion``."""
    _log.warning(str(msg))


# ---------------------------------------------------------------------------
# Pure-compute helpers — copied verbatim (modulo type hints) from
# ``src_preprocessing/diff_img/preprocessing/utils_diff_img.py``.
# ---------------------------------------------------------------------------


def check_for_missing_values_in_preproc_diff_data(data: dict) -> bool:
    """Returns True if any preprocessed array still contains NaNs."""
    for k, v in data.items():
        if k in ("quality", "images_numbers"):
            if np.isnan(v).sum() > 0:
                return True
        else:
            for k2, v2 in v.items():
                if k2 in ("neighbors_imgs", "neighbors_imgs_tc") and v2[0] is None:
                    continue
                if np.isnan(v2).sum() > 0:
                    return True
    return False


def initialize_data_example_with_missing_values(size_h, size_w, number_of_imgs_to_sample):
    """Initialize a per-TCE preprocessing dict filled with NaNs."""
    return {
        "images": {
            "diff_imgs": [np.nan * np.ones((size_h, size_w), dtype="float")
                          for _ in range(number_of_imgs_to_sample)],
            "oot_imgs": [np.nan * np.ones((size_h, size_w), dtype="float")
                         for _ in range(number_of_imgs_to_sample)],
            "snr_imgs": [np.nan * np.ones((size_h, size_w), dtype="float")
                         for _ in range(number_of_imgs_to_sample)],
            "target_imgs": [np.nan * np.ones((size_h, size_w), dtype="float")
                            for _ in range(number_of_imgs_to_sample)],
            "neighbors_imgs": [np.nan * np.ones((size_h, size_w), dtype="float")
                               for _ in range(number_of_imgs_to_sample)],
            "diff_imgs_tc": [np.nan * np.ones((size_h, size_w), dtype="float")
                             for _ in range(number_of_imgs_to_sample)],
            "oot_imgs_tc": [np.nan * np.ones((size_h, size_w), dtype="float")
                            for _ in range(number_of_imgs_to_sample)],
            "snr_imgs_tc": [np.nan * np.ones((size_h, size_w), dtype="float")
                            for _ in range(number_of_imgs_to_sample)],
            "target_imgs_tc": [np.nan * np.ones((size_h, size_w), dtype="float")
                               for _ in range(number_of_imgs_to_sample)],
            "neighbors_imgs_tc": [np.nan * np.ones((size_h, size_w), dtype="float")
                                  for _ in range(number_of_imgs_to_sample)],
        },
        "target_position": {
            "pixel_x": [np.nan] * number_of_imgs_to_sample,
            "pixel_y": [np.nan] * number_of_imgs_to_sample,
            "subpixel_x": [np.nan] * number_of_imgs_to_sample,
            "subpixel_y": [np.nan] * number_of_imgs_to_sample,
            "pixel_x_tc": [np.nan] * number_of_imgs_to_sample,
            "pixel_y_tc": [np.nan] * number_of_imgs_to_sample,
            "subpixel_x_tc": [np.nan] * number_of_imgs_to_sample,
            "subpixel_y_tc": [np.nan] * number_of_imgs_to_sample,
        },
        "quality": [np.nan] * number_of_imgs_to_sample,
        "images_numbers": [np.nan] * number_of_imgs_to_sample,
    }


def set_data_example_to_placeholder_values(size_h, size_w, number_of_imgs_to_sample,
                                           set_placeholder_neighbor_data=False):
    """Return a placeholder dict for TCEs where no valid imagery is available."""
    half_height, half_width = size_h // 2, size_w // 2
    data_placeholder = {
        "images": {
            "diff_imgs": [np.zeros((size_h, size_w), dtype="float")
                          for _ in range(number_of_imgs_to_sample)],
            "oot_imgs": [np.zeros((size_h, size_w), dtype="float")
                         for _ in range(number_of_imgs_to_sample)],
            "snr_imgs": [np.zeros((size_h, size_w), dtype="float")
                         for _ in range(number_of_imgs_to_sample)],
            "target_imgs": [np.zeros((size_h, size_w), dtype="float")
                            for _ in range(number_of_imgs_to_sample)],
            "diff_imgs_tc": [np.zeros((size_h, size_w), dtype="float")
                             for _ in range(number_of_imgs_to_sample)],
            "oot_imgs_tc": [np.zeros((size_h, size_w), dtype="float")
                            for _ in range(number_of_imgs_to_sample)],
            "snr_imgs_tc": [np.zeros((size_h, size_w), dtype="float")
                            for _ in range(number_of_imgs_to_sample)],
            "target_imgs_tc": [np.zeros((size_h, size_w), dtype="float")
                               for _ in range(number_of_imgs_to_sample)],
        },
        "target_position": {
            "pixel_x": [half_height] * number_of_imgs_to_sample,
            "pixel_y": [half_width] * number_of_imgs_to_sample,
            "subpixel_x": [0] * number_of_imgs_to_sample,
            "subpixel_y": [0] * number_of_imgs_to_sample,
            "pixel_x_tc": [half_height] * number_of_imgs_to_sample,
            "pixel_y_tc": [half_width] * number_of_imgs_to_sample,
            "subpixel_x_tc": [0] * number_of_imgs_to_sample,
            "subpixel_y_tc": [0] * number_of_imgs_to_sample,
        },
        "quality": [0] * number_of_imgs_to_sample,
        "images_numbers": [np.nan] * number_of_imgs_to_sample,
    }
    if set_placeholder_neighbor_data:
        data_placeholder["images"]["neighbors_imgs"] = [
            np.zeros((size_h, size_w), dtype="float") for _ in range(number_of_imgs_to_sample)
        ]
        data_placeholder["images"]["neighbors_imgs_tc"] = [
            np.zeros((size_h, size_w), dtype="float") for _ in range(number_of_imgs_to_sample)
        ]
    for target_img_i in range(number_of_imgs_to_sample):
        data_placeholder["images"]["target_imgs"][target_img_i][half_height, half_width] = 1
        data_placeholder["images"]["target_imgs_tc"][target_img_i][half_height, half_width] = 1
    return data_placeholder


def sample_image_data(n_valid_imgs, valid_images_idxs, number_of_imgs_to_sample):
    """Randomly sample valid quarters/sectors, tiling if fewer are available."""
    if n_valid_imgs < number_of_imgs_to_sample:
        k_n_valid_imgs = number_of_imgs_to_sample // n_valid_imgs
        random_sample_imgs_idxs = np.tile(valid_images_idxs, k_n_valid_imgs)
        random_sample_imgs_idxs = np.concatenate([
            random_sample_imgs_idxs,
            np.random.choice(valid_images_idxs,
                             number_of_imgs_to_sample % n_valid_imgs,
                             replace=False),
        ])
    else:
        random_sample_imgs_idxs = np.random.choice(
            valid_images_idxs, number_of_imgs_to_sample, replace=False,
        )
    return random_sample_imgs_idxs


def set_negative_value_oot_pixels(diff_img, oot_img, snr_img):
    """Mark pixels with negative out-of-transit flux as NaN across all three images."""
    curr_img_neg = oot_img < 0
    diff_img[curr_img_neg] = np.nan
    oot_img[curr_img_neg] = np.nan
    snr_img[curr_img_neg] = np.nan
    return diff_img, oot_img, snr_img


def crop_images_to_valid_size(diff_img, oot_img, snr_img):
    """Crop images to the bounding box of non-missing pixels in ``diff_img``."""
    idxs_not_missing = np.where(~np.isnan(diff_img))
    min_row, max_row = idxs_not_missing[0].min(), idxs_not_missing[0].max()
    min_col, max_col = idxs_not_missing[1].min(), idxs_not_missing[1].max()
    return (
        diff_img[min_row:max_row + 1, min_col:max_col + 1],
        oot_img[min_row:max_row + 1, min_col:max_col + 1],
        snr_img[min_row:max_row + 1, min_col:max_col + 1],
        min_col,
        min_row,
    )


def fill_missing_values_nearest_neighbors(img, window):
    """Fill NaN pixels via equal-weight neighbour convolution; empty windows stay NaN."""
    idxs_nan = np.isnan(img)
    valid_px_arr = np.ones(img.shape)
    valid_px_arr[idxs_nan] = 0
    valid_px_arr_conv = convolve2d(valid_px_arr, window, mode="same", boundary="fill", fillvalue=0)
    img_fill = img.copy()
    img_fill[idxs_nan] = 0
    img_conv = convolve2d(img_fill, window, mode="same", boundary="fill", fillvalue=0)
    valid_idxs = valid_px_arr_conv != 0
    img_conv[valid_idxs] = img_conv[valid_idxs] / valid_px_arr_conv[valid_idxs]
    img_conv[~valid_idxs] = np.nan
    img_fill[idxs_nan] = img_conv[idxs_nan]
    return img_fill


def pad_images_by_extending_edges(diff_img, oot_img, snr_img, add_pad_h, add_pad_w):
    """Pad images via edge extension so both dimensions match and target fits."""
    pad_len = [diff_img.shape[0] // 2 + add_pad_h + 1, diff_img.shape[1] // 2 + add_pad_w + 1]
    pad_len_diff = 2 * pad_len[0] + diff_img.shape[0] - (2 * pad_len[1] + diff_img.shape[1])
    if pad_len_diff >= 0:
        pad_len_row = [pad_len[0]] * 2
        pad_len_col = [pad_len[1] + pad_len_diff // 2] * 2
        pad_len_col[1] += pad_len_diff % 2
    else:
        pad_len_col = [pad_len[1]] * 2
        pad_len_diff *= -1
        pad_len_row = [pad_len[0] + pad_len_diff // 2] * 2
        pad_len_row[1] += pad_len_diff % 2

    diff_img_pad = np.pad(diff_img, (pad_len_row, pad_len_col), mode="edge")
    oot_img_pad = np.pad(oot_img, (pad_len_row, pad_len_col), mode="edge")
    snr_img_pad = np.pad(snr_img, (pad_len_row, pad_len_col), mode="edge")
    return diff_img_pad, oot_img_pad, snr_img_pad, pad_len_col[0], pad_len_row[0]


def center_images_to_target_pixel_location(diff_img, oot_img, snr_img, target_pos_col, target_pos_row):
    """Pad images so the target pixel lies at image centre."""
    target_pos_pixel_col = int(target_pos_col)
    target_pos_pixel_row = int(target_pos_row)
    center_row, center_col = diff_img.shape[0] // 2, diff_img.shape[1] // 2

    if target_pos_pixel_row >= center_row:
        pad_len_h = [0, 2 * target_pos_pixel_row - diff_img.shape[0] + 1]
        row_offset = 0
    else:
        pad_len_h = [diff_img.shape[0] - 2 * target_pos_pixel_row - 1, 0]
        row_offset = pad_len_h[0]

    if target_pos_pixel_col >= center_col:
        pad_len_w = [0, 2 * target_pos_pixel_col - diff_img.shape[1] + 1]
        col_offset = 0
    else:
        pad_len_w = [diff_img.shape[1] - 2 * target_pos_pixel_col - 1, 0]
        col_offset = pad_len_w[0]

    diff_img_tcenter = np.pad(diff_img, (pad_len_h, pad_len_w), mode="edge")
    oot_img_tcenter = np.pad(oot_img, (pad_len_h, pad_len_w), mode="edge")
    snr_img_tcenter = np.pad(snr_img, (pad_len_h, pad_len_w), mode="edge")
    return diff_img_tcenter, oot_img_tcenter, snr_img_tcenter, col_offset, row_offset


def crop_images_to_size(diff_img, oot_img, snr_img, size_h, size_w):
    """Center-crop images to ``(size_h, size_w)`` using PIL (byte-exact with oracle)."""
    half_height, half_width = size_h // 2, size_w // 2
    diff_img = Image.fromarray(diff_img)
    oot_img = Image.fromarray(oot_img)
    snr_img = Image.fromarray(snr_img)
    center_col, center_row = diff_img.size[0] // 2, diff_img.size[1] // 2

    if diff_img.size[1] <= size_h:
        upper, lower = 0, diff_img.size[1]
    else:
        upper, lower = center_row - half_height, center_row + half_height + 1
    if diff_img.size[0] <= size_w:
        left, right = 0, diff_img.size[0]
    else:
        left, right = center_col - half_width, center_col + half_width + 1

    diff_img_crop = np.array(diff_img.crop((left, upper, right, lower)))
    oot_img_crop = np.array(oot_img.crop((left, upper, right, lower)))
    snr_img_crop = np.array(snr_img.crop((left, upper, right, lower)))
    return diff_img_crop, oot_img_crop, snr_img_crop, left, upper


def resize_images_by_resampling(diff_img, oot_img, snr_img, size_f_h, size_f_w):
    """Resize images by nearest-neighbour resampling (PIL, byte-exact with oracle)."""
    size_h, size_w = diff_img.shape
    diff_img = Image.fromarray(diff_img)
    oot_img = Image.fromarray(oot_img)
    snr_img = Image.fromarray(snr_img)
    diff_img_resize = np.array(diff_img.resize(
        size=(size_h * size_f_h, size_w * size_f_w), resample=Image.Resampling.NEAREST,
    ))
    oot_img_resize = np.array(oot_img.resize(
        size=(size_h * size_f_h, size_w * size_f_w), resample=Image.Resampling.NEAREST,
    ))
    snr_img_resize = np.array(snr_img.resize(
        size=(size_h * size_f_h, size_w * size_f_w), resample=Image.Resampling.NEAREST,
    ))
    return diff_img_resize, oot_img_resize, snr_img_resize


def create_target_image(size_h, size_w, target_pos_pixel_col, target_pos_pixel_row):
    """Build a single-hot image marking the target pixel location."""
    target_pos_pixel_col_up = int(np.round(target_pos_pixel_col))
    target_pos_pixel_row_up = int(np.round(target_pos_pixel_row))
    target_img = np.zeros((size_h, size_w), dtype="float")
    if (0 <= target_pos_pixel_row_up <= size_h - 1) and (0 <= target_pos_pixel_col_up <= size_w - 1):
        target_img[target_pos_pixel_col_up, target_pos_pixel_row_up] = 1
    return target_img


def map_target_subpixel_location_to_discrete_grid(target_pos_col, target_pos_row):
    """Map subpixel offsets to a discrete {-1, 0, 1} grid; NaN rows fall back to 0,0."""
    if np.isnan(target_pos_row):
        return 0, 0
    target_pos_pixel_col = int(np.round(target_pos_col))
    target_pos_pixel_row = int(np.round(target_pos_row))
    target_col_diff = target_pos_pixel_col - target_pos_col
    target_row_diff = target_pos_pixel_row - target_pos_row
    return int(2 * target_col_diff), int(2 * target_row_diff)


def create_neighbors_img(neighbor_data, img_shape, target_mag, exclude_objs_outside=True):
    """Render a neighbours image with per-pixel brightest target-to-neighbour mag ratio."""
    if exclude_objs_outside:
        neighbor_data = {
            neighbor_id: neighbor_id_data
            for neighbor_id, neighbor_id_data in neighbor_data.items()
            if 0 <= neighbor_id_data["col_px"] < img_shape[1]
            and 0 <= neighbor_id_data["row_px"] < img_shape[0]
        }

    n_neighbors = len(neighbor_data)
    if n_neighbors == 0:
        return np.zeros(img_shape, dtype="float")

    neighbor_img = np.inf * np.ones(img_shape + (n_neighbors,), dtype="float")
    neighbor_data = dict(sorted(neighbor_data.items(), key=lambda item: item[1]["Tmag"]))
    for neighbor_i, neighbor_id in enumerate(neighbor_data):
        neighbor_row, neighbor_col = (
            int(neighbor_data[neighbor_id]["row_px"]),
            int(neighbor_data[neighbor_id]["col_px"]),
        )
        neighbor_img[neighbor_row, neighbor_col, neighbor_i] = (
            target_mag / neighbor_data[neighbor_id]["Tmag"]
        )

    neighbor_img = np.min(neighbor_img, axis=-1)
    neighbor_img[~np.isfinite(neighbor_img)] = 0
    return neighbor_img


# ---------------------------------------------------------------------------
# Per-example preprocessing — verbatim from oracle, minus logger wiring.
# ---------------------------------------------------------------------------


def preprocess_single_diff_img_data_for_example(diff_img, oot_img, snr_img, target_pos_col,
                                                target_pos_row, size_h, size_w, size_f_h,
                                                size_f_w, img_n, tce_uid, prefix,
                                                center_target=True, neighbor_data=None,
                                                target_mag=None,
                                                exclude_neighbor_objs_outside=True,
                                                log=None, proc_id=-1):
    """Full preprocessing chain for a single sector's difference imagery."""
    target_pos_col_input, target_pos_row_input = target_pos_col, target_pos_row

    target_col_disc, target_row_disc = map_target_subpixel_location_to_discrete_grid(
        target_pos_col, target_pos_row,
    )
    half_height, half_width = size_h // 2, size_w // 2

    diff_img, oot_img, snr_img = set_negative_value_oot_pixels(diff_img, oot_img, snr_img)

    if np.isnan(diff_img).all() or np.isnan(oot_img).all() or np.isnan(snr_img).all():
        if log:
            log.info(
                f"[{proc_id}] All pixels in the difference, out-of-transit, or SNR image were "
                f"missing after zeroing negative oot pixels for example {tce_uid} in {prefix} {img_n}."
            )

    diff_img, oot_img, snr_img, crop_min_col, crop_min_row = crop_images_to_valid_size(
        diff_img, oot_img, snr_img,
    )

    fill_missing_filter = np.ones((3, 3))
    diff_img = fill_missing_values_nearest_neighbors(diff_img, fill_missing_filter)
    oot_img = fill_missing_values_nearest_neighbors(oot_img, fill_missing_filter)
    snr_img = fill_missing_values_nearest_neighbors(snr_img, fill_missing_filter)

    idxs_nan = np.isnan(diff_img)
    if idxs_nan.sum() != 0:
        if log:
            log.info(
                f"[{proc_id}] {idxs_nan.sum()} pixels still missing after NN padding "
                f"for example {tce_uid} in {prefix} {img_n}."
            )
        diff_img[idxs_nan] = 0
        snr_img[idxs_nan] = 0
        oot_img[idxs_nan] = np.nanmin(oot_img)

    diff_img, oot_img, snr_img, pad_col, pad_row = pad_images_by_extending_edges(
        diff_img, oot_img, snr_img, half_height, half_width,
    )
    target_pos_col = target_pos_col - crop_min_col + pad_col
    target_pos_row = target_pos_row - crop_min_row + pad_row
    if neighbor_data is not None:
        for neighbor_id, neighbor_id_data in neighbor_data.items():
            neighbor_data[neighbor_id]["col_px"] = neighbor_id_data["col_px"] - crop_min_col + pad_col
            neighbor_data[neighbor_id]["row_px"] = neighbor_id_data["row_px"] - crop_min_row + pad_row

    diff_img, oot_img, snr_img = resize_images_by_resampling(
        diff_img, oot_img, snr_img, size_f_h, size_f_w,
    )
    target_pos_col = target_pos_col * size_f_w + (size_f_w - 1) * 0.5
    target_pos_row = target_pos_row * size_f_h + (size_f_h - 1) * 0.5
    if neighbor_data is not None:
        for neighbor_id, neighbor_id_data in neighbor_data.items():
            neighbor_data[neighbor_id]["col_px"] = (
                neighbor_id_data["col_px"] * size_f_w + (size_f_w - 1) * 0.5
            )
            neighbor_data[neighbor_id]["row_px"] = (
                neighbor_id_data["row_px"] * size_f_h + (size_f_h - 1) * 0.5
            )

    if center_target:
        if np.isnan(target_pos_col_input):
            if log:
                log.info(
                    f"[{proc_id}] Target location from DV not available "
                    f"({target_pos_row_input}, {target_pos_col_input}); skipping centering "
                    f"for {tce_uid} in {prefix} {img_n}."
                )
        else:
            (diff_img, oot_img, snr_img, center_col_offset, center_row_offset) = (
                center_images_to_target_pixel_location(
                    diff_img, oot_img, snr_img, target_pos_col, target_pos_row,
                )
            )
            target_pos_col = target_pos_col + center_col_offset
            target_pos_row = target_pos_row + center_row_offset
            if neighbor_data is not None:
                for neighbor_id, neighbor_id_data in neighbor_data.items():
                    neighbor_data[neighbor_id]["col_px"] = (
                        neighbor_id_data["col_px"] + center_col_offset
                    )
                    neighbor_data[neighbor_id]["row_px"] = (
                        neighbor_id_data["row_px"] + center_row_offset
                    )

    (diff_img, oot_img, snr_img, crop_size_col_offset, crop_size_row_offset) = (
        crop_images_to_size(
            diff_img, oot_img, snr_img, size_h * size_f_h, size_w * size_f_w,
        )
    )

    target_pos_col -= crop_size_col_offset
    target_pos_row -= crop_size_row_offset
    if neighbor_data is not None:
        for neighbor_id, neighbor_id_data in neighbor_data.items():
            neighbor_data[neighbor_id]["col_px"] = (
                neighbor_id_data["col_px"] - crop_size_col_offset
            )
            neighbor_data[neighbor_id]["row_px"] = (
                neighbor_id_data["row_px"] - crop_size_row_offset
            )

    if (target_pos_col < 0 or target_pos_row < 0 or
            target_pos_col >= diff_img.shape[0] or target_pos_row >= diff_img.shape[1]):
        if log:
            log.info(
                f"[{proc_id}] Target pixel outside image after cropping for "
                f"{tce_uid} in {prefix} {img_n}."
            )

    target_img = create_target_image(
        size_h * size_f_h, size_w * size_f_w, target_pos_col, target_pos_row,
    )

    if neighbor_data is not None:
        neighbors_img = create_neighbors_img(
            neighbor_data, diff_img.shape, target_mag, exclude_neighbor_objs_outside,
        )
    else:
        neighbors_img = None

    return (
        diff_img, oot_img, snr_img, target_img,
        (target_pos_col, target_pos_row),
        (target_col_disc, target_row_disc),
        neighbors_img,
    )


# ---------------------------------------------------------------------------
# Public API — per-TCE refactor of ``preprocess_diff_img_tces``.
# ---------------------------------------------------------------------------


def preprocess_tce_diff_img(tce_uid: str,
                            tce_diff_img_data: dict,
                            number_of_imgs_to_sample: int,
                            upscale_f: dict,
                            final_size: dict,
                            mission_name: str = "tess",
                            exclude_neighbor_objs_outside: bool = True,
                            rng: Any = None) -> dict:
    """Preprocess difference-image data for a single TCE.

    This is the native-worker replacement for the oracle's
    ``preprocess_diff_img_tces`` top-level function, which iterates over a
    full ``.npy`` file of TCEs and writes a ``diffimg_preprocess.npy`` blob
    alongside a CSV info table.  The worker calls this once per TCE, keeps
    the result in memory, and passes it to the normalization stage (C.14)
    which produces the final ``*_std_trainset`` feature tensors consumed
    by the ExoMiner ONNX graph.

    Args:
        tce_uid: str, unique TCE identifier (used only for log messages).
        tce_diff_img_data: dict, as emitted by
            :func:`worker.exominer_native.preprocess.dv_xml.get_data_from_tess_dv_xml`
            for a single TCE.  Must contain the keys ``target_ref_centroid``,
            ``image_data``, ``image_number``, ``quality_metric`` and ``mag``.
            ``neighbor_data`` is optional.
        number_of_imgs_to_sample: int, number of sectors to sample.
        upscale_f: dict with keys ``x``, ``y`` for the upscale factors.
        final_size: dict with keys ``x``, ``y`` for the target image size
            before upscaling.
        mission_name: str, currently only ``"tess"`` is supported.
        exclude_neighbor_objs_outside: bool, forwarded to
            :func:`create_neighbors_img`.
        rng: optional ``np.random.Generator`` for reproducible sampling.
            When ``None`` the module-level ``np.random`` state is used
            (matches oracle behaviour).

    Returns:
        dict — same schema as ``preprocessing_dict[tce_uid]`` in the
        oracle.  Top-level keys: ``images`` (dict of image-name → list of
        ``(H, W)`` numpy arrays), ``target_position``, ``quality``,
        ``images_numbers``, plus ``info`` with sampled-quarter diagnostics.
    """
    if mission_name == "kepler":
        prefix = "quarter"
    elif mission_name == "tess":
        prefix = "sector"
    else:
        raise ValueError(
            f"Mission not recognized ({mission_name!r}). Use 'kepler' or 'tess'.",
        )

    size_h = final_size["x"] * upscale_f["x"]
    size_w = final_size["y"] * upscale_f["y"]

    preprocessing = initialize_data_example_with_missing_values(
        size_h, size_w, number_of_imgs_to_sample,
    )
    info = {
        "uid": tce_uid,
        f"num_available_{prefix}s": np.nan,
        f"num_valid_{prefix}s": np.nan,
        f"num_sampled_{prefix}s": np.nan,
        f"sampled_{prefix}s": np.nan * np.ones(number_of_imgs_to_sample),
        "sampled_qmetrics": np.nan * np.ones(number_of_imgs_to_sample),
        f"sampled_{prefix}s_missingvalues": "no",
    }

    n_max_imgs_avail = len(tce_diff_img_data["image_number"])
    info[f"num_available_{prefix}s"] = n_max_imgs_avail

    available_imgs_idxs = [
        img_idx for img_idx in range(n_max_imgs_avail)
        if not np.isnan(tce_diff_img_data["quality_metric"][img_idx]["value"])
    ]

    valid_images_idxs = []
    for img_idx in available_imgs_idxs:
        if (
            tce_diff_img_data["target_ref_centroid"][img_idx]["col"]["uncertainty"] != -1
            and tce_diff_img_data["quality_metric"][img_idx]["valid"]
            and not np.isnan(tce_diff_img_data["image_data"][img_idx][:, :, 1, 0]).all()
            and not np.isnan(tce_diff_img_data["image_data"][img_idx][:, :, 2, 0]).all()
        ):
            valid_images_idxs.append(img_idx)

    n_valid_imgs = len(valid_images_idxs)
    info[f"num_valid_{prefix}s"] = n_valid_imgs

    if n_valid_imgs == 0:
        _report_exclusion(
            f"No valid images for {tce_uid}. Using placeholder values."
        )
        missing_data_placeholder = set_data_example_to_placeholder_values(
            size_h, size_w, number_of_imgs_to_sample,
            "neighbor_data" in tce_diff_img_data,
        )
        preprocessing.update(missing_data_placeholder)
        preprocessing["info"] = info
        return preprocessing

    if rng is not None:
        # Reproducible path — use the provided generator.  Same algorithm
        # as :func:`sample_image_data` but avoids the module-level RNG.
        if n_valid_imgs < number_of_imgs_to_sample:
            k = number_of_imgs_to_sample // n_valid_imgs
            random_sample_imgs_idxs = np.tile(valid_images_idxs, k)
            random_sample_imgs_idxs = np.concatenate([
                random_sample_imgs_idxs,
                rng.choice(valid_images_idxs,
                           number_of_imgs_to_sample % n_valid_imgs,
                           replace=False),
            ])
        else:
            random_sample_imgs_idxs = rng.choice(
                valid_images_idxs, number_of_imgs_to_sample, replace=False,
            )
    else:
        random_sample_imgs_idxs = sample_image_data(
            n_valid_imgs, valid_images_idxs, number_of_imgs_to_sample,
        )

    info[f"num_sampled_{prefix}s"] = len(random_sample_imgs_idxs)
    info[f"sampled_{prefix}s"] = np.array([
        tce_diff_img_data["image_number"][idx] for idx in random_sample_imgs_idxs
    ])
    info["sampled_qmetrics"] = np.array([
        tce_diff_img_data["quality_metric"][idx]["value"] for idx in random_sample_imgs_idxs
    ])

    for sampled_img_idx in np.unique(random_sample_imgs_idxs):
        img_idxs = np.where(random_sample_imgs_idxs == sampled_img_idx)[0]

        for option in CENTER_OPTIONS:
            diff_img = tce_diff_img_data["image_data"][sampled_img_idx][:, :, 2, 0].copy()
            oot_img = tce_diff_img_data["image_data"][sampled_img_idx][:, :, 1, 0].copy()
            snr_img = diff_img / tce_diff_img_data["image_data"][sampled_img_idx][:, :, 2, 1]

            if float(tce_diff_img_data["target_ref_centroid"][sampled_img_idx]["col"]["value"]) == -1:
                target_pos_col, target_pos_row = np.nan, np.nan
            else:
                target_pos_col = float(
                    tce_diff_img_data["target_ref_centroid"][sampled_img_idx]["col"]["value"]
                )
                target_pos_row = float(
                    tce_diff_img_data["target_ref_centroid"][sampled_img_idx]["row"]["value"]
                )

            image_number = tce_diff_img_data["image_number"][sampled_img_idx]
            target_mag = tce_diff_img_data.get("mag")
            if "neighbor_data" in tce_diff_img_data:
                neighbors_data = {
                    neighbor_id: dict(neighbor_data)
                    for neighbor_id, neighbor_data in
                    tce_diff_img_data["neighbor_data"][sampled_img_idx].items()
                }
            else:
                neighbors_data = None

            (diff_img_preproc, oot_img_preproc, snr_img_preproc, target_img, target_pos,
             target_pos_disc, neighbors_img) = preprocess_single_diff_img_data_for_example(
                diff_img, oot_img, snr_img,
                target_pos_col, target_pos_row,
                final_size["x"], final_size["y"],
                upscale_f["x"], upscale_f["y"],
                image_number, tce_uid, prefix,
                center_target=(option == "target_centered"),
                neighbor_data=neighbors_data,
                target_mag=target_mag,
                exclude_neighbor_objs_outside=exclude_neighbor_objs_outside,
                log=None, proc_id=tce_uid,
            )

            suffix_str = "_tc" if option == "target_centered" else ""

            for img_idx in img_idxs:
                preprocessing["images"][f"diff_imgs{suffix_str}"][img_idx] = diff_img_preproc
                preprocessing["images"][f"oot_imgs{suffix_str}"][img_idx] = oot_img_preproc
                preprocessing["images"][f"snr_imgs{suffix_str}"][img_idx] = snr_img_preproc
                preprocessing["images"][f"target_imgs{suffix_str}"][img_idx] = target_img
                preprocessing["target_position"][f"pixel_x{suffix_str}"][img_idx] = target_pos[1]
                preprocessing["target_position"][f"pixel_y{suffix_str}"][img_idx] = target_pos[0]
                preprocessing["target_position"][f"subpixel_x{suffix_str}"][img_idx] = target_pos_disc[1]
                preprocessing["target_position"][f"subpixel_y{suffix_str}"][img_idx] = target_pos_disc[0]
                if "neighbor_data" in tce_diff_img_data:
                    preprocessing["images"][f"neighbors_imgs{suffix_str}"][img_idx] = neighbors_img

        for img_idx in img_idxs:
            preprocessing["quality"][img_idx] = (
                tce_diff_img_data["quality_metric"][sampled_img_idx]["value"]
            )
            preprocessing["images_numbers"][img_idx] = (
                tce_diff_img_data["image_number"][sampled_img_idx]
            )

    missing_value_found = check_for_missing_values_in_preproc_diff_data(preprocessing)
    if missing_value_found:
        _report_exclusion(
            f"At least one array contained missing values for {tce_uid}. "
            "Using placeholder values.",
        )
        info[f"sampled_{prefix}s_missingvalues"] = "yes"
        missing_data_placeholder = set_data_example_to_placeholder_values(
            size_h, size_w, number_of_imgs_to_sample,
        )
        preprocessing.update(missing_data_placeholder)

    preprocessing["info"] = info
    return preprocessing


def preprocess_diff_img_tces_dict(diff_img_data_dict: dict,
                                  number_of_imgs_to_sample: int,
                                  upscale_f: dict,
                                  final_size: dict,
                                  mission_name: str = "tess",
                                  exclude_neighbor_objs_outside: bool = True,
                                  rng: Any = None) -> dict:
    """Batch wrapper around :func:`preprocess_tce_diff_img`.

    The oracle's ``preprocess_diff_img_tces`` loads a ``.npy`` file and
    writes another one; the native worker receives the dict in-memory
    (from :func:`worker.exominer_native.preprocess.dv_xml.get_data_from_tess_dv_xml`)
    and returns a dict keyed by TCE UID.
    """
    return {
        tce_uid: preprocess_tce_diff_img(
            tce_uid,
            tce_diff_img_data,
            number_of_imgs_to_sample=number_of_imgs_to_sample,
            upscale_f=upscale_f,
            final_size=final_size,
            mission_name=mission_name,
            exclude_neighbor_objs_outside=exclude_neighbor_objs_outside,
            rng=rng,
        )
        for tce_uid, tce_diff_img_data in diff_img_data_dict.items()
    }
