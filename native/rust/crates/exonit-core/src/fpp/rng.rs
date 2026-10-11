//! NumPy-legacy RNG transforms on top of `NumpyMt19937`, plus the
//! deterministic per-TIC seed.
//!
//! Every algorithm here is a transliteration of NumPy's legacy
//! `RandomState` C paths (`legacy-distributions.c`) — the exact draw
//! streams are pinned by `fixtures/fpp/rng_golden.json`, generated from
//! the reference venv (numpy 2.4.6). As with `injection.rs`, draw-for-draw
//! reproduction is the contract; none of these may be swapped for a
//! statistically-equivalent alternative.
//!
//! `power()` deliberately implements the *scipy* path used by
//! `fpp_native._sample_ecc(planet=False)` — `powerlaw.rvs` has no `_rvs`
//! override, so scipy draws `uniform()` and applies the `q**(1/a)` ppf.
//! (Numerically this coincides with `np.random.power`, verified 2026-08-17,
//! but the normative path is uniform + ppf.)

use crate::injection::NumpyMt19937;

/// numpy-legacy RandomState transforms with the `has_gauss` cache.
#[derive(Debug, Clone)]
pub struct FppRng {
    mt: NumpyMt19937,
    has_gauss: bool,
    gauss: f64,
}

impl FppRng {
    pub fn new(seed: u32) -> Self {
        FppRng {
            mt: NumpyMt19937::new(seed),
            has_gauss: false,
            gauss: 0.0,
        }
    }

    /// `rk_double` / `random_sample` — same stream as `np.random.rand`.
    pub fn rand(&mut self) -> f64 {
        self.mt.uniform01()
    }

    /// `legacy_gauss` — polar Box-Muller with the one-value cache.
    pub fn gauss(&mut self) -> f64 {
        if self.has_gauss {
            self.has_gauss = false;
            let temp = self.gauss;
            self.gauss = 0.0;
            return temp;
        }
        loop {
            let x1 = 2.0 * self.rand() - 1.0;
            let x2 = 2.0 * self.rand() - 1.0;
            let r2 = x1 * x1 + x2 * x2;
            if r2 < 1.0 && r2 != 0.0 {
                let f = (-2.0 * r2.ln() / r2).sqrt();
                self.gauss = f * x1;
                self.has_gauss = true;
                return f * x2;
            }
        }
    }

    /// `legacy_standard_exponential`.
    pub fn standard_exponential(&mut self) -> f64 {
        -(1.0 - self.rand()).ln()
    }

    /// `legacy_standard_gamma` (shape==1 → exponential; <1 → GD-transform
    /// rejection; >1 → Marsaglia–Tsang with `legacy_gauss`).
    pub fn standard_gamma(&mut self, shape: f64) -> f64 {
        if shape == 1.0 {
            return self.standard_exponential();
        }
        if shape < 1.0 {
            loop {
                let u = self.rand();
                let v = self.standard_exponential();
                if u <= 1.0 - shape {
                    let x = u.powf(1.0 / shape);
                    if x <= v {
                        return x;
                    }
                } else {
                    let y = -((1.0 - u) / shape).ln();
                    let x = (1.0 - shape + shape * y).powf(1.0 / shape);
                    if x <= v + y {
                        return x;
                    }
                }
            }
        }
        let b = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * b).sqrt();
        loop {
            let mut x;
            let mut v;
            loop {
                x = self.gauss();
                v = 1.0 + c * x;
                if v > 0.0 {
                    break;
                }
            }
            v = v * v * v;
            let u = self.rand();
            if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                return b * v;
            }
            if u.ln() < 0.5 * x * x + b * (1.0 - v + v.ln()) {
                return b * v;
            }
        }
    }

    /// `legacy_beta` (Jöhnk for a,b ≤ 1; two-gamma otherwise).
    pub fn beta(&mut self, a: f64, b: f64) -> f64 {
        if a <= 1.0 && b <= 1.0 {
            loop {
                let u = self.rand();
                let v = self.rand();
                let x = u.powf(1.0 / a);
                let y = v.powf(1.0 / b);
                if x + y <= 1.0 {
                    if x + y > 0.0 {
                        return x / (x + y);
                    }
                    let mut log_x = u.ln() / a;
                    let mut log_y = v.ln() / b;
                    let log_m = log_x.max(log_y);
                    log_x -= log_m;
                    log_y -= log_m;
                    return (log_x - (log_x.exp() + log_y.exp()).ln()).exp();
                }
            }
        }
        let ga = self.standard_gamma(a);
        let gb = self.standard_gamma(b);
        ga / (ga + gb)
    }

    /// scipy `powerlaw.rvs` path: uniform draw through the `q**(1/a)` ppf.
    pub fn power(&mut self, a: f64) -> f64 {
        self.rand().powf(1.0 / a)
    }
}

/// Python's `f"{x:.10e}"` — mantissa with 10 decimals, sign on the
/// exponent, exponent zero-padded to at least two digits.
fn py_e10(x: f64) -> String {
    let s = format!("{:.10e}", x);
    let (mant, exp) = s.split_once('e').expect("exponent");
    let e: i32 = exp.parse().expect("exponent digits");
    format!("{}e{}{:02}", mant, if e < 0 { "-" } else { "+" }, e.abs())
}

/// The normative per-TIC seed key (see `parity/fpp/README.md`).
pub fn tic_seed_key(tic_id: &str, period: f64, t0: f64) -> String {
    format!("{}:{}:{}", tic_id, py_e10(period), py_e10(t0))
}

/// FNV-1a 64 of the seed key, truncated to u32.
pub fn tic_seed(tic_id: &str, period: f64, t0: f64) -> u32 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for byte in tic_seed_key(tic_id, period, t0).bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    (h & 0xFFFF_FFFF) as u32
}
