"""Quickselect kernel correctness (Phase 1 of wotan-CUDA).

Validates that the device-local Quickselect implementation used by
``worker/wotan_cuda.py`` returns the same k-th order statistic that
``numpy.partition`` does on a variety of synthetic inputs (random
uniform, gaussian, duplicates-heavy, pre-sorted, reverse-sorted, all-
equal, single-outlier). Also validates that the median wrapper
returns ``numpy.median`` for odd and even ``n``.

These tests run a small standalone test kernel that calls the same
``dev_quickselect`` / ``dev_median`` device functions used by the
production biweight kernel — that way we exercise the exact code
paths the worker will hit, not a Python re-implementation.

Skipped if cupy / CUDA are unavailable.
"""

import os
import sys

import numpy as np
import pytest

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


pytestmark = pytest.mark.skipif(
    not _HAS_CUPY, reason="cupy not installed (CUDA-only Phase 1 test)"
)


# Re-extract the device functions from the production kernel source so
# we test the exact same code, not a copy. The production kernel
# defines `dev_quickselect` and `dev_median` as device inline functions
# at the top of `_BIWEIGHT_RUNNING_KERNEL`.
def _extract_device_helpers():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))
    import wotan_cuda  # noqa: E402

    src = wotan_cuda._BIWEIGHT_RUNNING_KERNEL
    # Carve out everything before `extern "C"` — the device helpers.
    marker = 'extern "C"'
    idx = src.find(marker)
    assert idx > 0, "could not find extern \"C\" marker in production kernel"
    return src[:idx]


_DEVICE_SRC = _extract_device_helpers() if _HAS_CUPY else ""


_TEST_KERNEL_SRC = _DEVICE_SRC + r"""
extern "C" __global__
void test_quickselect_one(
    double*    buf,   // (n,) input, OVERWRITTEN
    int*       k_ptr, // pointer to k
    int*       n_ptr, // pointer to n
    double*    out    // (1,) result
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int n = *n_ptr;
    int k = *k_ptr;
    *out = dev_quickselect(buf, n, k);
}

extern "C" __global__
void test_median_one(
    double*    buf,   // (n,) input, OVERWRITTEN
    int*       n_ptr,
    double*    out
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int n = *n_ptr;
    *out = dev_median(buf, n);
}
"""


@pytest.fixture(scope="module")
def quickselect_kernel():
    return cp.RawKernel(_TEST_KERNEL_SRC, "test_quickselect_one")


@pytest.fixture(scope="module")
def median_kernel():
    return cp.RawKernel(_TEST_KERNEL_SRC, "test_median_one")


def _gpu_quickselect(kernel, arr, k):
    d_buf = cp.asarray(arr, dtype=cp.float64).copy()
    d_k = cp.asarray(np.int32(k))
    d_n = cp.asarray(np.int32(len(arr)))
    d_out = cp.empty(1, dtype=cp.float64)
    kernel((1,), (1,), (d_buf, d_k, d_n, d_out))
    return float(cp.asnumpy(d_out)[0])


def _gpu_median(kernel, arr):
    d_buf = cp.asarray(arr, dtype=cp.float64).copy()
    d_n = cp.asarray(np.int32(len(arr)))
    d_out = cp.empty(1, dtype=cp.float64)
    kernel((1,), (1,), (d_buf, d_n, d_out))
    return float(cp.asnumpy(d_out)[0])


# ------------------------------------------------------------ #
# Quickselect: arbitrary k-th order statistic vs np.partition
# ------------------------------------------------------------ #

class TestQuickselectKthOrderStatistic:
    @pytest.mark.parametrize("n", [3, 5, 10, 50, 100, 360, 768])
    def test_random_uniform(self, quickselect_kernel, n):
        rng = np.random.default_rng(42 + n)
        arr = rng.uniform(-1.0, 1.0, size=n)
        for k in [0, n // 4, n // 2, (3 * n) // 4, n - 1]:
            expected = float(np.partition(arr.copy(), k)[k])
            got = _gpu_quickselect(quickselect_kernel, arr, k)
            assert got == pytest.approx(expected, abs=0, rel=0), (
                f"n={n} k={k}: got {got!r} expected {expected!r}"
            )

    @pytest.mark.parametrize("n", [10, 100, 360])
    def test_gaussian(self, quickselect_kernel, n):
        rng = np.random.default_rng(7 + n)
        arr = rng.standard_normal(size=n)
        for k in [0, n // 2, n - 1]:
            expected = float(np.partition(arr.copy(), k)[k])
            got = _gpu_quickselect(quickselect_kernel, arr, k)
            assert got == pytest.approx(expected, abs=0, rel=0)

    def test_pre_sorted(self, quickselect_kernel):
        arr = np.linspace(0.0, 1.0, 256)
        for k in [0, 50, 128, 200, 255]:
            expected = arr[k]
            got = _gpu_quickselect(quickselect_kernel, arr, k)
            assert got == pytest.approx(expected, abs=0, rel=0)

    def test_reverse_sorted(self, quickselect_kernel):
        arr = np.linspace(1.0, 0.0, 256)
        for k in [0, 50, 128, 200, 255]:
            expected = float(np.partition(arr.copy(), k)[k])
            got = _gpu_quickselect(quickselect_kernel, arr, k)
            assert got == pytest.approx(expected, abs=0, rel=0)

    def test_heavy_duplicates(self, quickselect_kernel):
        rng = np.random.default_rng(11)
        # Many ties: integers cast to float
        arr = rng.integers(0, 5, size=300).astype(np.float64)
        for k in [0, 75, 150, 225, 299]:
            expected = float(np.partition(arr.copy(), k)[k])
            got = _gpu_quickselect(quickselect_kernel, arr, k)
            assert got == pytest.approx(expected, abs=0, rel=0)

    def test_all_equal(self, quickselect_kernel):
        arr = np.full(100, 3.1415, dtype=np.float64)
        for k in [0, 50, 99]:
            got = _gpu_quickselect(quickselect_kernel, arr, k)
            assert got == pytest.approx(3.1415, abs=0, rel=0)

    def test_single_outlier(self, quickselect_kernel):
        arr = np.ones(101, dtype=np.float64)
        arr[50] = 1000.0
        # k=100 picks the outlier; k=50 picks one of the 1.0s
        assert _gpu_quickselect(quickselect_kernel, arr, 100) == 1000.0
        assert _gpu_quickselect(quickselect_kernel, arr, 50) == 1.0


# ------------------------------------------------------------ #
# Median wrapper: odd/even/edge sizes vs np.median
# ------------------------------------------------------------ #

class TestQuickselectMedian:
    @pytest.mark.parametrize("n", [3, 4, 5, 10, 11, 50, 51, 100, 101, 360, 361, 768])
    def test_random_uniform(self, median_kernel, n):
        rng = np.random.default_rng(101 + n)
        arr = rng.uniform(-1.0, 1.0, size=n)
        expected = float(np.median(arr))
        got = _gpu_median(median_kernel, arr)
        assert got == pytest.approx(expected, abs=0, rel=0), (
            f"n={n}: got {got!r} expected {expected!r}"
        )

    @pytest.mark.parametrize("n", [4, 10, 100, 360])
    def test_even_n_gaussian(self, median_kernel, n):
        rng = np.random.default_rng(2026 + n)
        arr = rng.standard_normal(size=n)
        expected = float(np.median(arr))
        got = _gpu_median(median_kernel, arr)
        assert got == pytest.approx(expected, abs=0, rel=0)

    def test_two_elements(self, median_kernel):
        # n=2: median = (a+b)/2
        for a, b in [(0.0, 1.0), (-3.5, 7.25), (4.0, 4.0)]:
            arr = np.asarray([a, b], dtype=np.float64)
            expected = float(np.median(arr))
            got = _gpu_median(median_kernel, arr)
            assert got == pytest.approx(expected, abs=0, rel=0)

    def test_duplicates_even(self, median_kernel):
        arr = np.asarray([1.0, 2.0, 2.0, 3.0], dtype=np.float64)
        assert _gpu_median(median_kernel, arr) == pytest.approx(2.0)

    def test_constant(self, median_kernel):
        arr = np.full(513, 42.0)
        assert _gpu_median(median_kernel, arr) == pytest.approx(42.0)
