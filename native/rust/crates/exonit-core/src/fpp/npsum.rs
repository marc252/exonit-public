//! numpy-compatible reductions.
//!
//! numpy's `add.reduce` (and therefore `np.sum`/`np.mean`/`np.std`) uses
//! PAIRWISE summation with 8-way unrolled 128-element blocks — a naive
//! sequential sum is NOT bit-identical. This is a faithful port of
//! `pairwise_sum_DOUBLE` (numpy `loops.c.src`).

const PW_BLOCKSIZE: usize = 128;

/// `pairwise_sum_DOUBLE` for a contiguous slice.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in a {
            res += v;
        }
        return res;
    }
    if n <= PW_BLOCKSIZE {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            r[0] += a[i];
            r[1] += a[i + 1];
            r[2] += a[i + 2];
            r[3] += a[i + 3];
            r[4] += a[i + 4];
            r[5] += a[i + 5];
            r[6] += a[i + 6];
            r[7] += a[i + 7];
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

/// `np.mean` (contiguous 1-D).
pub fn np_mean(a: &[f64]) -> f64 {
    pairwise_sum(a) / a.len() as f64
}

/// `np.std` (ddof=0) — mirrors `_methods._var`: pairwise mean, then
/// pairwise sum of real squared deviations, then sqrt.
pub fn np_std(a: &[f64]) -> f64 {
    let mean = np_mean(a);
    let sq: Vec<f64> = a.iter().map(|&v| (v - mean) * (v - mean)).collect();
    (pairwise_sum(&sq) / a.len() as f64).sqrt()
}

/// `np.median` — sort a copy (values, not stability, matter) and take the
/// middle element, or the numpy mean of the two middles for even n.
pub fn np_median(a: &[f64]) -> f64 {
    let mut v = a.to_vec();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        // np.mean of the two middle elements
        np_mean(&[v[n / 2 - 1], v[n / 2]])
    }
}

/// `np.nanmedian` — median ignoring NaNs (NaN if none remain).
pub fn np_nanmedian(a: &[f64]) -> f64 {
    let filtered: Vec<f64> = a.iter().copied().filter(|v| !v.is_nan()).collect();
    if filtered.is_empty() {
        f64::NAN
    } else {
        np_median(&filtered)
    }
}

/// Actual libm `pow` call, shielded from LLVM's `pow(x,2)→x*x` rewrite.
///
/// CPython's scalar `x ** 2.0` calls libm `pow`, which on this glibc is
/// NOT always bit-identical to `x * x` (1-ulp difference observed for
/// real fixture values, 2026-08-17). numpy ARRAY `**2` takes the
/// `npy_pow` fast path (`x * x`). The port must therefore distinguish
/// Python-scalar squares (→ this function) from numpy-array squares
/// (→ literal `x * x`).
#[inline]
pub fn libm_pow(x: f64, y: f64) -> f64 {
    extern "C" {
        fn pow(x: f64, y: f64) -> f64;
    }
    unsafe { pow(std::hint::black_box(x), std::hint::black_box(y)) }
}

/// `np.nan_to_num` defaults: NaN→0, +inf→f64::MAX, −inf→−f64::MAX.
pub fn nan_to_num(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else if v == f64::INFINITY {
        f64::MAX
    } else if v == f64::NEG_INFINITY {
        -f64::MAX
    } else {
        v
    }
}
