"""Fast unit tests for ``worker.exominer_native.preprocess.centroid``."""
from __future__ import annotations

import numpy as np
import pytest

from worker.exominer_native.preprocess.centroid import (
    compute_centroid_distance,
    convertpxtoradec_centr,
    correct_centroid_using_transit_depth,
    kepler_transform_pxcoordinates_mod13,
    local_normalization_centroid,
    patch_centroid_curve,
    synchronize_centroids_with_flux,
)


def test_kepler_transform_mod13_rotates_90_and_270_but_leaves_quarter_0_alone():
    # quarter 0 gets bumped to 1 internally; then delta_q for q=[1,2,3,4] → [0,1,2,3]
    # → indices 1 and 3 should be rotated, 0 and 2 untouched.
    xs = [np.array([100.0]), np.array([200.0]), np.array([300.0]), np.array([400.0])]
    ys = [np.array([500.0]), np.array([600.0]), np.array([700.0]), np.array([800.0])]
    centroids = {"x": [a.copy() for a in xs], "y": [a.copy() for a in ys]}
    add_info = {"quarter": [1, 2, 3, 4], "module": [13, 13, 13, 13]}

    out = kepler_transform_pxcoordinates_mod13(centroids, add_info)

    assert out["x"][0][0] == 100.0  # delta_q=0, untouched
    assert out["y"][0][0] == 500.0
    assert out["x"][2][0] == 300.0  # delta_q=2, untouched
    assert out["y"][2][0] == 700.0
    # indices 1 and 3 were rotated
    assert out["x"][1][0] != 200.0
    assert out["y"][1][0] != 600.0
    assert out["x"][3][0] != 400.0
    assert out["y"][3][0] != 800.0


def test_patch_centroid_curve_stitches_consecutive_quarters():
    # two quarters with a jump of +10 in both x and y; patch should remove the jump
    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.array([13.0, 14.0, 15.0])  # +10 jump
    y1 = np.array([10.0, 11.0, 12.0])
    y2 = np.array([22.0, 23.0, 24.0])  # +10 jump
    centroids = {"x": [x1.copy(), x2.copy()], "y": [y1.copy(), y2.copy()]}

    out = patch_centroid_curve(centroids)

    # after patch the second quarter should start at the last value of the first
    assert out["x"][1][0] == x1[-1]
    assert out["y"][1][0] == y1[-1]


def test_patch_centroid_curve_passes_through_1d_ndarray():
    arr = np.arange(5, dtype=float)
    out = patch_centroid_curve(arr)
    assert out is arr


def test_local_normalization_divides_by_window_median():
    xs = [np.array([2.0, 4.0, 6.0, 8.0])]
    ys = [np.array([10.0, 20.0, 30.0, 40.0])]
    centroids = {"x": xs, "y": ys}
    local_normalization_centroid(centroids, win_len=4)
    # median(x)=5, median(y)=25 → divide
    np.testing.assert_allclose(centroids["x"][0], np.array([2.0, 4.0, 6.0, 8.0]) / 5.0)
    np.testing.assert_allclose(centroids["y"][0], np.array([10.0, 20.0, 30.0, 40.0]) / 25.0)


def test_synchronize_centroids_keeps_matching_cadences():
    all_time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    centroid_time = np.array([0.0, 1.0, 3.0, 4.0])  # missing index 2
    centroids = {
        "x": np.array([10.0, 20.0, 30.0, 40.0]),
        "y": np.array([100.0, 200.0, 300.0, 400.0]),
    }
    out = synchronize_centroids_with_flux(all_time, centroid_time, centroids, thres=0.005)
    # should have 5 entries, with NaN where the centroid was missing (flux index 2)
    assert out["x"].shape == (5,)
    assert np.isnan(out["x"][2])
    assert np.isnan(out["y"][2])
    assert out["x"][0] == 10.0
    assert out["x"][-1] == 40.0


def test_convertpxtoradec_centr_identity_transform():
    n = 4
    centroid_x = np.array([1.0, 2.0, 3.0, 4.0])
    centroid_y = np.array([5.0, 6.0, 7.0, 8.0])
    cd = np.eye(2)
    ref_ccdf = np.array([[1.0], [1.0]])
    ref_apf = np.array([[0.0], [0.0]])
    ref_ang = np.array([[0.0], [0.0]])
    ra, dec = convertpxtoradec_centr(centroid_x, centroid_y, cd, ref_ccdf, ref_apf, ref_ang)
    # identity: ra = x - 1 - 0 + 1 = x, dec = y - 1 - 0 + 1 = y
    np.testing.assert_allclose(ra, centroid_x)
    np.testing.assert_allclose(dec, centroid_y)


def test_correct_centroid_using_transit_depth_centers_on_oot():
    centroid_x = np.array([10.0, 10.0])
    centroid_y = np.array([0.0, 0.0])  # cos(0)=1 for simple arithmetic
    avg_oot = {"x": 10.0, "y": 0.0}
    transit_depth = 1e5  # term = (1e6-1e5)/1e5 = 9
    out = correct_centroid_using_transit_depth(centroid_x, centroid_y, transit_depth, avg_oot)
    # centroid == oot → correction term is zero → output equals oot
    np.testing.assert_allclose(out["x"], np.array([10.0, 10.0]))
    np.testing.assert_allclose(out["y"], np.array([0.0, 0.0]))


def test_compute_centroid_distance_pixel_mode_is_euclidean():
    centroid = {"x": np.array([3.0, 0.0]), "y": np.array([4.0, 0.0])}
    target = [0.0, 0.0]
    dist = compute_centroid_distance(centroid, target)
    np.testing.assert_allclose(dist, np.array([5.0, 0.0]))


def test_compute_centroid_distance_sky_mode_scales_x_by_delta_dec():
    centroid = {"x": np.array([2.0]), "y": np.array([0.0])}
    target = [0.0, 0.0]
    # delta_dec=0.5 → sqrt((2*0.5)^2 + 0) = 1.0
    dist = compute_centroid_distance(centroid, target, delta_dec=0.5)
    np.testing.assert_allclose(dist, np.array([1.0]))
