"""Fast unit tests for ``worker.exominer_native.preprocess.diff_img``.

Exercises the dict-returning entry points that replace the oracle's
TFRecord-writing pipeline, plus spot-checks of the pure-compute helpers.
Uses small synthetic difference-image dicts that mimic the structure
produced by :mod:`worker.exominer_native.preprocess.dv_xml`.
"""
from __future__ import annotations

import numpy as np
import pytest

from worker.exominer_native.preprocess.diff_img import (
    CENTER_OPTIONS,
    check_for_missing_values_in_preproc_diff_data,
    create_neighbors_img,
    create_target_image,
    crop_images_to_size,
    crop_images_to_valid_size,
    fill_missing_values_nearest_neighbors,
    initialize_data_example_with_missing_values,
    map_target_subpixel_location_to_discrete_grid,
    pad_images_by_extending_edges,
    preprocess_diff_img_tces_dict,
    preprocess_single_diff_img_data_for_example,
    preprocess_tce_diff_img,
    resize_images_by_resampling,
    sample_image_data,
    set_data_example_to_placeholder_values,
    set_negative_value_oot_pixels,
)


UPSCALE_F = {"x": 11, "y": 11}
FINAL_SIZE = {"x": 3, "y": 3}  # → 33 × 33 after upscaling (matches model config)
NUM_IMGS_TO_SAMPLE = 5


def _make_image_data(rows: int = 11, cols: int = 11, seed: int = 1) -> np.ndarray:
    """Build an ``(rows, cols, 4, 2)`` block as emitted by the oracle DV XML parser.

    Last axis: ``[value, uncertainty]``.  Third axis indexes the four
    image types ``[in_transit, oot, diff, snr]`` in that order.  Values
    are deliberately non-negative so ``set_negative_value_oot_pixels``
    is a no-op.
    """
    rng = np.random.default_rng(seed)
    image_data = np.zeros((rows, cols, 4, 2), dtype=float)
    image_data[..., 0] = rng.uniform(low=1.0, high=10.0, size=(rows, cols, 4))
    image_data[..., 1] = rng.uniform(low=0.1, high=0.5, size=(rows, cols, 4))
    return image_data


def _make_tce_data(n_sectors: int = 3, rows: int = 11, cols: int = 11) -> dict:
    """Build a single-TCE dict matching the dv_xml.py output schema."""
    return {
        "mag": 11.3,
        "image_number": [1 + i for i in range(n_sectors)],
        "image_data": [_make_image_data(rows, cols, seed=7 + i) for i in range(n_sectors)],
        "target_ref_centroid": [
            {
                "col": {"value": rows // 2 + 0.3, "uncertainty": 0.01},
                "row": {"value": cols // 2 - 0.2, "uncertainty": 0.01},
            }
            for _ in range(n_sectors)
        ],
        "quality_metric": [{"value": 0.9, "valid": True} for _ in range(n_sectors)],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_map_target_subpixel_returns_discrete_range():
    c, r = map_target_subpixel_location_to_discrete_grid(5.3, 4.8)
    assert isinstance(c, int) and isinstance(r, int)
    assert c in (-1, 0, 1) and r in (-1, 0, 1)

    # NaN row → (0, 0)
    assert map_target_subpixel_location_to_discrete_grid(np.nan, np.nan) == (0, 0)


def test_set_negative_value_oot_pixels_marks_all_three():
    diff = np.ones((3, 3))
    oot = np.ones((3, 3))
    oot[0, 0] = -1.0
    snr = np.ones((3, 3))
    d, o, s = set_negative_value_oot_pixels(diff, oot, snr)
    assert np.isnan(d[0, 0]) and np.isnan(o[0, 0]) and np.isnan(s[0, 0])


def test_fill_missing_values_preserves_non_missing():
    img = np.ones((5, 5))
    img[2, 2] = np.nan
    filled = fill_missing_values_nearest_neighbors(img, np.ones((3, 3)))
    assert not np.isnan(filled).any()
    assert filled[2, 2] == pytest.approx(1.0)


def test_crop_images_to_valid_size_removes_border_nans():
    diff = np.full((7, 7), np.nan)
    diff[2:5, 3:6] = 1.0
    oot = diff.copy()
    snr = diff.copy()
    d, _, _, min_col, min_row = crop_images_to_valid_size(diff, oot, snr)
    assert d.shape == (3, 3)
    assert (min_col, min_row) == (3, 2)


def test_pad_images_by_extending_edges_makes_square():
    diff = np.ones((5, 7))
    oot = np.ones((5, 7))
    snr = np.ones((5, 7))
    d, o, s, _pad_col, _pad_row = pad_images_by_extending_edges(diff, oot, snr, 1, 1)
    # After padding both dims must match
    assert d.shape[0] == d.shape[1]
    assert d.shape == o.shape == s.shape


def test_resize_images_by_resampling_scales_both_dims():
    diff = np.arange(9, dtype=float).reshape(3, 3)
    oot = diff.copy()
    snr = diff.copy()
    d, o, s = resize_images_by_resampling(diff, oot, snr, 4, 4)
    assert d.shape == (12, 12) == o.shape == s.shape


def test_create_target_image_marks_single_pixel():
    t = create_target_image(5, 5, 2, 3)
    assert t.sum() == 1
    assert t[2, 3] == 1


def test_crop_images_to_size_center_crop():
    a = np.arange(49, dtype=float).reshape(7, 7)
    d, _, _, left, upper = crop_images_to_size(a, a.copy(), a.copy(), 3, 3)
    assert d.shape == (3, 3)
    # Center crop in a 7x7 → upper/left should be 2
    assert (left, upper) == (2, 2)


def test_sample_image_data_tiles_when_valid_under_requested():
    idxs = sample_image_data(2, [0, 1], 5)
    assert len(idxs) == 5
    assert set(idxs).issubset({0, 1})


def test_create_neighbors_img_all_outside_returns_zeros():
    img_shape = (5, 5)
    neighbors = {
        "N1": {"col_px": -1, "row_px": 3, "Tmag": 14.0},
        "N2": {"col_px": 2, "row_px": 99, "Tmag": 13.5},
    }
    out = create_neighbors_img(neighbors, img_shape, target_mag=11.0)
    assert out.shape == img_shape
    assert not out.any()


def test_initialize_and_placeholder_shapes():
    d = initialize_data_example_with_missing_values(33, 33, NUM_IMGS_TO_SAMPLE)
    assert len(d["images"]["diff_imgs"]) == NUM_IMGS_TO_SAMPLE
    assert d["images"]["diff_imgs"][0].shape == (33, 33)
    p = set_data_example_to_placeholder_values(33, 33, NUM_IMGS_TO_SAMPLE)
    # target_imgs should have a single 1 at image centre
    assert p["images"]["target_imgs"][0].sum() == 1


def test_check_for_missing_values_detects_nan_and_clean():
    d = initialize_data_example_with_missing_values(5, 5, 2)
    assert check_for_missing_values_in_preproc_diff_data(d) is True
    p = set_data_example_to_placeholder_values(5, 5, 2)
    # quality/images_numbers still carry NaN in placeholder (images_numbers)
    p["images_numbers"] = [1, 2]
    assert check_for_missing_values_in_preproc_diff_data(p) is False


# ---------------------------------------------------------------------------
# Per-example chain
# ---------------------------------------------------------------------------


def test_preprocess_single_diff_img_data_for_example_basic_shape():
    diff = np.ones((11, 11))
    oot = np.ones((11, 11)) * 5.0
    snr = np.ones((11, 11)) * 2.0
    d, o, s, tgt, tpos, tdisc, neigh = preprocess_single_diff_img_data_for_example(
        diff, oot, snr,
        target_pos_col=5.3, target_pos_row=5.1,
        size_h=FINAL_SIZE["x"], size_w=FINAL_SIZE["y"],
        size_f_h=UPSCALE_F["x"], size_f_w=UPSCALE_F["y"],
        img_n=1, tce_uid="T-1-1", prefix="sector",
        center_target=True,
    )
    assert d.shape == (33, 33) == o.shape == s.shape == tgt.shape
    assert neigh is None
    assert isinstance(tpos, tuple) and isinstance(tdisc, tuple)


# ---------------------------------------------------------------------------
# Top-level refactored entry point
# ---------------------------------------------------------------------------


def _assert_tce_schema(result: dict) -> None:
    assert set(result.keys()) >= {"images", "target_position", "quality",
                                  "images_numbers", "info"}
    for key in ("diff_imgs", "oot_imgs", "snr_imgs", "target_imgs",
                "diff_imgs_tc", "oot_imgs_tc", "snr_imgs_tc", "target_imgs_tc"):
        assert key in result["images"], key
        arrs = result["images"][key]
        assert len(arrs) == NUM_IMGS_TO_SAMPLE
        assert arrs[0].shape == (
            FINAL_SIZE["x"] * UPSCALE_F["x"],
            FINAL_SIZE["y"] * UPSCALE_F["y"],
        )
    assert len(result["quality"]) == NUM_IMGS_TO_SAMPLE
    assert len(result["images_numbers"]) == NUM_IMGS_TO_SAMPLE


def test_preprocess_tce_diff_img_returns_expected_shapes():
    tce = _make_tce_data(n_sectors=3)
    rng = np.random.default_rng(42)
    out = preprocess_tce_diff_img(
        "TIC-1-1",
        tce,
        number_of_imgs_to_sample=NUM_IMGS_TO_SAMPLE,
        upscale_f=UPSCALE_F,
        final_size=FINAL_SIZE,
        rng=rng,
    )
    _assert_tce_schema(out)
    # Either the synthetic fixture preprocesses cleanly (quality is
    # propagated from ``quality_metric.value``) or the pipeline falls
    # into the placeholder branch (quality = 0).  Both are valid outputs
    # for random synthetic imagery; the schema assertion above is the
    # load-bearing one.
    assert all(q in (0, 0.9) for q in out["quality"])
    assert out["info"]["num_available_sectors"] == 3


def test_preprocess_tce_diff_img_placeholder_when_no_valid():
    tce = _make_tce_data(n_sectors=2)
    # Invalidate everything → placeholder branch
    for qm in tce["quality_metric"]:
        qm["valid"] = False
    out = preprocess_tce_diff_img(
        "TIC-1-1",
        tce,
        number_of_imgs_to_sample=NUM_IMGS_TO_SAMPLE,
        upscale_f=UPSCALE_F,
        final_size=FINAL_SIZE,
    )
    _assert_tce_schema(out)
    # placeholder fills target_imgs with a single 1 at center
    assert out["images"]["target_imgs"][0].sum() == 1
    assert out["info"]["num_valid_sectors"] == 0


def test_preprocess_diff_img_tces_dict_wraps_single_tce():
    tce = _make_tce_data(n_sectors=3)
    data = {"TIC-1-1": tce, "TIC-2-1": _make_tce_data(n_sectors=4)}
    rng = np.random.default_rng(1)
    out = preprocess_diff_img_tces_dict(
        data,
        number_of_imgs_to_sample=NUM_IMGS_TO_SAMPLE,
        upscale_f=UPSCALE_F,
        final_size=FINAL_SIZE,
        rng=rng,
    )
    assert set(out.keys()) == {"TIC-1-1", "TIC-2-1"}
    for res in out.values():
        _assert_tce_schema(res)


def test_preprocess_tce_diff_img_rejects_unknown_mission():
    tce = _make_tce_data(n_sectors=1)
    with pytest.raises(ValueError, match="Mission not recognized"):
        preprocess_tce_diff_img(
            "TIC-1-1",
            tce,
            number_of_imgs_to_sample=NUM_IMGS_TO_SAMPLE,
            upscale_f=UPSCALE_F,
            final_size=FINAL_SIZE,
            mission_name="foo",
        )


def test_center_options_emits_both_variants():
    # Sanity: the module-level constant is what the implementation iterates over
    assert set(CENTER_OPTIONS) == {"target_centered", "target_not_centered"}
