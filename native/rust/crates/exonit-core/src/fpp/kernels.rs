//! Batched Monte Carlo log-likelihood kernels — transliteration of
//! `worker/fpp_native.py:333-430` (`_batch_transit_lnL`,
//! `_batch_eb_transit_lnL`).
//!
//! Parallelism: rayon over draws, mirroring `numba.prange` — every draw is
//! independent, so multithreading cannot change any bit of any element.
//! The per-draw chi² loop stays sequential over time, like the Python.

use rayon::prelude::*;

use super::transit::compute_transit_lc;

fn ln2pi() -> f64 {
    (2.0 * std::f64::consts::PI).ln()
}

/// Parameter arrays shared by both kernels (one entry per draw).
pub struct BatchParams<'a> {
    pub ks: &'a [f64],
    pub p_orbs: &'a [f64],
    pub incs_rad: &'a [f64],
    pub a_rs: &'a [f64],
    pub eccs: &'a [f64],
    pub ws_rad: &'a [f64],
    pub companion_fluxratio: &'a [f64],
    pub companion_is_host: bool,
}

/// `_batch_transit_lnL`.
pub fn batch_transit_lnl(
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    u1: f64,
    u2: f64,
    p: &BatchParams,
) -> Vec<f64> {
    let lnsigma = sigma.ln();
    let ln2pi = ln2pi();
    (0..p.ks.len())
        .into_par_iter()
        .map(|j| {
            let mut model = compute_transit_lc(
                time,
                p.ks[j],
                0.0,
                p.p_orbs[j],
                p.a_rs[j],
                p.incs_rad[j],
                p.eccs[j],
                p.ws_rad[j],
                u1,
                u2,
            );
            let fr = p.companion_fluxratio[j];
            if fr > 0.0 {
                let f_dilute = if p.companion_is_host {
                    1.0 / (fr / (1.0 - fr))
                } else {
                    fr / (1.0 - fr)
                };
                for m in model.iter_mut() {
                    *m = (*m + f_dilute) / (1.0 + f_dilute);
                }
            }
            let mut chi2 = 0.0;
            for (t, m) in flux.iter().zip(model.iter()) {
                let diff = t - m;
                chi2 += diff * diff;
            }
            -0.5 * ln2pi - lnsigma - 0.5 * chi2 / (sigma * sigma)
        })
        .collect()
}

/// `_batch_eb_transit_lnL` (secondary-eclipse rejection included; rejected
/// draws keep the `-inf` initialisation, as in the Python).
pub fn batch_eb_transit_lnl(
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    u1: f64,
    u2: f64,
    eb_fluxratios: &[f64],
    p: &BatchParams,
) -> Vec<f64> {
    let lnsigma = sigma.ln();
    let ln2pi = ln2pi();
    (0..p.ks.len())
        .into_par_iter()
        .map(|j| {
            let mut k = p.ks[j];
            if (k - 1.0).abs() < 1e-6 {
                k *= 0.999;
            }
            let mut model = compute_transit_lc(
                time,
                k,
                0.0,
                p.p_orbs[j],
                p.a_rs[j],
                p.incs_rad[j],
                p.eccs[j],
                p.ws_rad[j],
                u1,
                u2,
            );
            let fr_eb = eb_fluxratios[j];
            let f_eb = if fr_eb < 1.0 {
                fr_eb / (1.0 - fr_eb)
            } else {
                1e6
            };
            let fr_comp = p.companion_fluxratio[j];
            let f_comp = if fr_comp > 0.0 {
                fr_comp / (1.0 - fr_comp)
            } else {
                0.0
            };

            let secdepth;
            if p.companion_is_host {
                let ratio = f_eb / f_comp.max(1e-30);
                for m in model.iter_mut() {
                    *m = (*m + ratio) / (1.0 + ratio);
                }
                let f_dilute = 1.0 / (f_comp + f_eb);
                for m in model.iter_mut() {
                    *m = (*m + f_dilute) / (1.0 + f_dilute);
                }
                secdepth = fr_eb / (1.0 + f_dilute);
            } else {
                for m in model.iter_mut() {
                    *m = (*m + f_eb) / (1.0 + f_eb);
                }
                let f_dilute = f_comp / (1.0 + f_eb);
                for m in model.iter_mut() {
                    *m = (*m + f_dilute) / (1.0 + f_dilute);
                }
                secdepth = if f_dilute > 0.0 {
                    fr_eb / (1.0 + f_dilute)
                } else {
                    fr_eb
                };
            }

            if secdepth >= 1.5 * sigma {
                return f64::NEG_INFINITY;
            }

            let mut chi2 = 0.0;
            for (t, m) in flux.iter().zip(model.iter()) {
                let diff = t - m;
                chi2 += diff * diff;
            }
            -0.5 * ln2pi - lnsigma - 0.5 * chi2 / (sigma * sigma)
        })
        .collect()
}
