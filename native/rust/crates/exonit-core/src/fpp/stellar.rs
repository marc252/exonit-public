//! Stellar mass→(radius, Teff, flux) relations and TESS limb-darkening
//! lookup — transliteration of `worker/fpp_native.py:437-538`.
//!
//! The FITPACK spline coefficients are embedded bit-for-bit
//! (`stellar_coeffs.rs`, generated) and only the `splev` evaluator (de
//! Boor / `fpbspl`) is ported; `fixtures/fpp/spline_golden.json` pins a
//! dense evaluation mesh per spline, including out-of-range extrapolation
//! (scipy `ext=0`) and the `10**x → inf` overflow of `flux_relation`.

use std::sync::OnceLock;

#[path = "stellar_coeffs.rs"]
mod stellar_coeffs;
use stellar_coeffs as sc;

fn bits(v: &[u64]) -> Vec<f64> {
    v.iter().copied().map(f64::from_bits).collect()
}

pub struct Spline {
    t: Vec<f64>,
    c: Vec<f64>,
    k: usize,
}

impl Spline {
    fn new(t_bits: &[u64], c_bits: &[u64], k: usize) -> Self {
        Spline {
            t: bits(t_bits),
            c: bits(c_bits),
            k,
        }
    }

    /// FITPACK `splev` with `e=0` (extrapolate from the boundary
    /// polynomial): locate `l` with `t[l] <= x < t[l+1]` clamped to the
    /// valid B-spline range, then de Boor (`fpbspl`) in the same
    /// operation order as the Fortran.
    pub fn eval(&self, x: f64) -> f64 {
        let k = self.k;
        let n = self.t.len();
        // 0-based valid interval for l: [k, n-k-2]
        let mut l = k;
        let hi = n - k - 2;
        while l < hi && x >= self.t[l + 1] {
            l += 1;
        }
        // fpbspl: non-zero B-splines h[0..=k] at x for interval l
        let mut h = [0.0f64; 8];
        let mut hh = [0.0f64; 8];
        h[0] = 1.0;
        for j in 1..=k {
            hh[..j].copy_from_slice(&h[..j]);
            h[0] = 0.0;
            for i in 0..j {
                let li = self.t[l + i + 1];
                let lj = self.t[l + i + 1 - j];
                if li == lj {
                    // fpbspl equal-knot guard (repeated boundary knots)
                    h[i + 1] = 0.0;
                    continue;
                }
                let f = hh[i] / (li - lj);
                h[i] += f * (li - x);
                h[i + 1] = f * (x - lj);
            }
        }
        let mut sp = 0.0;
        for j in 0..=k {
            sp += self.c[l - k + j] * h[j];
        }
        sp
    }
}

struct Splines {
    r_torres: Spline,
    t_torres: Spline,
    r_cdwrf: Spline,
    t_cdwrf: Spline,
    flux: Spline,
}

fn splines() -> &'static Splines {
    static S: OnceLock<Splines> = OnceLock::new();
    S.get_or_init(|| Splines {
        r_torres: Spline::new(sc::R_TORRES_T_BITS, sc::R_TORRES_C_BITS, sc::R_TORRES_K),
        t_torres: Spline::new(sc::T_TORRES_T_BITS, sc::T_TORRES_C_BITS, sc::T_TORRES_K),
        r_cdwrf: Spline::new(sc::R_CDWRF_T_BITS, sc::R_CDWRF_C_BITS, sc::R_CDWRF_K),
        t_cdwrf: Spline::new(sc::T_CDWRF_T_BITS, sc::T_CDWRF_C_BITS, sc::T_CDWRF_K),
        flux: Spline::new(sc::FLUX_T_BITS, sc::FLUX_C_BITS, sc::FLUX_K),
    })
}

/// `fpp_native.stellar_relations` — (radii, teffs) from masses in Msun.
pub fn stellar_relations(masses: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let s = splines();
    let mut r = Vec::with_capacity(masses.len());
    let mut t = Vec::with_capacity(masses.len());
    for &m in masses {
        let (ri, ti) = if m > 0.63 {
            (s.r_torres.eval(m), s.t_torres.eval(m))
        } else {
            let mc = m.clamp(0.1, 0.63);
            (s.r_cdwrf.eval(mc), s.t_cdwrf.eval(mc))
        };
        r.push(ri.max(0.1));
        t.push(ti.max(2800.0));
    }
    (r, t)
}

/// `fpp_native.flux_relation` — relative TESS-band flux (`10**spline`,
/// overflow to `inf` preserved).
pub fn flux_relation(masses: &[f64]) -> Vec<f64> {
    let s = splines();
    masses
        .iter()
        .map(|&m| 10.0f64.powf(s.flux.eval(m)))
        .collect()
}

struct LdcTable {
    z: Vec<f64>,
    teff: Vec<f64>,
    logg: Vec<f64>,
    u1: Vec<f64>,
    u2: Vec<f64>,
}

fn ldc_table() -> &'static LdcTable {
    static T: OnceLock<LdcTable> = OnceLock::new();
    T.get_or_init(|| {
        let csv = include_str!("data/ldc_tess.csv");
        let mut tab = LdcTable {
            z: vec![],
            teff: vec![],
            logg: vec![],
            u1: vec![],
            u2: vec![],
        };
        let mut lines = csv.lines();
        let header: Vec<&str> = lines.next().expect("ldc header").split(',').collect();
        let col = |name: &str| header.iter().position(|h| *h == name).expect("ldc column");
        let (iz, it, ig, ia, ib) = (col("Z"), col("Teff"), col("logg"), col("aLSM"), col("bLSM"));
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split(',').collect();
            tab.z.push(f[iz].parse().unwrap());
            tab.teff.push(f[it].parse().unwrap());
            tab.logg.push(f[ig].parse().unwrap());
            tab.u1.push(f[ia].parse().unwrap());
            tab.u2.push(f[ib].parse().unwrap());
        }
        tab
    })
}

fn argmin_abs_diff(values: &[f64], target: f64) -> usize {
    // np.argmin: first index of the minimum.
    let mut best = 0usize;
    let mut best_v = f64::INFINITY;
    for (i, &v) in values.iter().enumerate() {
        let d = (v - target).abs();
        if d < best_v {
            best_v = d;
            best = i;
        }
    }
    best
}

/// `fpp_native.get_ldc` — nearest-neighbour lookup of the quadratic
/// limb-darkening coefficients (Z, then Teff, then logg; joint mask;
/// solar fallback).
pub fn get_ldc(teff: f64, logg: f64, z: f64) -> (f64, f64) {
    let tab = ldc_table();
    let best_z = tab.z[argmin_abs_diff(&tab.z, z)];
    let best_t = tab.teff[argmin_abs_diff(&tab.teff, teff)];
    let best_g = tab.logg[argmin_abs_diff(&tab.logg, logg)];
    for i in 0..tab.z.len() {
        if tab.z[i] == best_z && tab.teff[i] == best_t && tab.logg[i] == best_g {
            return (tab.u1[i], tab.u2[i]);
        }
    }
    (0.40, 0.26)
}
