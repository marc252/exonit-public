//! Scenario marginal likelihoods — transliteration of
//! `worker/fpp_native.py:810-970` (`_lnZ_TTP`, `_lnZ_TEB`, `_lnZ_PTP`,
//! `_lnZ_PEB`).
//!
//! Operation order follows the Python line by line (including the
//! recomputed sub-expressions numpy evaluates elementwise); reductions go
//! through `npsum` (numpy pairwise semantics). The `checkpoints` hook
//! records the same probes the Python golden generator captures.

use super::kernels::{batch_eb_transit_lnl, batch_transit_lnl, BatchParams};
use super::npsum::{libm_pow, nan_to_num, np_mean, np_median, pairwise_sum};
use super::priors::{sample_ecc, sample_inc, sample_q, sample_q_companion, sample_rp, sample_w};
use super::rng::FppRng;
use super::stellar::{flux_relation, get_ldc, stellar_relations};

pub const MSUN: f64 = 1.989e33;
pub const RSUN: f64 = 6.957e10;
pub const REARTH: f64 = 6.371e8;
pub const G_CGS: f64 = 6.674e-8;

const PI: f64 = std::f64::consts::PI;

/// Chronological checkpoint recorder mirroring `gen_fpp_golden.Recorder`.
#[derive(Default, Debug)]
pub struct Probes {
    pub samplers: Vec<(String, usize, Vec<f64>)>,
    pub dispatch: Vec<(String, usize, Vec<f64>, f64, usize)>,
    pub lnz: Vec<(String, Vec<f64>)>,
}

impl Probes {
    fn sampler(&mut self, name: &str, values: &[f64]) {
        self.samplers.push((
            name.to_string(),
            values.len(),
            values.iter().copied().take(8).collect(),
        ));
    }

    fn dispatched(&mut self, label: &str, lnl: &[f64]) {
        let mut sum = 0.0;
        let mut n_finite = 0usize;
        for &v in lnl {
            if v.is_finite() {
                sum += v;
                n_finite += 1;
            }
        }
        self.dispatch.push((
            label.to_string(),
            lnl.len(),
            lnl.iter().copied().take(8).collect(),
            sum,
            n_finite,
        ));
    }

    fn lnz(&mut self, name: &str, values: &[f64]) {
        self.lnz.push((name.to_string(), values.to_vec()));
    }
}

pub struct StarParams {
    pub m_s: f64,
    pub r_s: f64,
    pub teff: f64,
}

fn masked(v: &[f64], mask: &[bool]) -> Vec<f64> {
    v.iter()
        .zip(mask)
        .filter(|(_, &m)| m)
        .map(|(&x, _)| x)
        .collect()
}

/// `Z = np.mean(np.nan_to_num(np.exp(lnL + 600)))`, then log-or--inf.
fn lnz_from_lnl(lnl: &[f64]) -> f64 {
    let vals: Vec<f64> = lnl.iter().map(|&v| nan_to_num((v + 600.0).exp())).collect();
    let z = np_mean(&vals);
    if z > 0.0 {
        z.ln()
    } else {
        f64::NEG_INFINITY
    }
}

/// `_lnZ_TTP`.
#[allow(clippy::too_many_arguments)]
pub fn lnz_ttp(
    rng: &mut FppRng,
    probes: &mut Probes,
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    p_orb: f64,
    star: &StarParams,
    n: usize,
    u1: f64,
    u2: f64,
    companion_fr: f64,
    companion_is_host: bool,
    lnz_label: Option<&str>,
) -> f64 {
    let (m_s, r_s) = (star.m_s, star.r_s);
    // numpy: scalar prefactor uses Python `pi**2` (libm pow); the array
    // `(P*86400)**2` takes npy_pow's x*x fast path; `**(1/3)` is libm pow.
    let p_sec = p_orb * 86400.0;
    let a_cm_scalar = libm_pow(
        G_CGS * m_s * MSUN / (4.0 * libm_pow(PI, 2.0)) * (p_sec * p_sec),
        1.0 / 3.0,
    );
    let a_rs_scalar = a_cm_scalar / (r_s * RSUN);

    let rps = sample_rp(rng, n, m_s);
    probes.sampler("_sample_rp", &rps);
    let incs = sample_inc(rng, n);
    probes.sampler("_sample_inc", &incs);
    let eccs = sample_ecc(rng, n, true, p_orb);
    probes.sampler("_sample_ecc", &eccs);
    let argps = sample_w(rng, n);
    probes.sampler("_sample_w", &argps);

    let ks: Vec<f64> = rps.iter().map(|&rp| rp * REARTH / (r_s * RSUN)).collect();
    let incs_rad: Vec<f64> = incs.iter().map(|&i| i * PI / 180.0).collect();
    let ws_rad: Vec<f64> = argps.iter().map(|&w| (90.0 - w) * PI / 180.0).collect();

    let mut mask = vec![false; n];
    for j in 0..n {
        let e_corr = (1.0 + eccs[j] * (argps[j] * PI / 180.0).sin()) / (1.0 - eccs[j] * eccs[j]);
        let ptra = (rps[j] * REARTH + r_s * RSUN) / a_cm_scalar * e_corr;
        let inc_min = if ptra <= 1.0 { ptra.acos() } else { PI / 2.0 };
        let coll = (rps[j] * REARTH + r_s * RSUN) > a_cm_scalar * (1.0 - eccs[j]);
        mask[j] = incs_rad[j] >= inc_min && !coll;
    }
    if !mask.iter().any(|&m| m) {
        let v = f64::NEG_INFINITY;
        if let Some(label) = lnz_label {
            probes.lnz(label, &[v]);
        }
        return v;
    }

    let mks = masked(&ks, &mask);
    let n_mask = mks.len();
    let comp_fr_arr = vec![companion_fr; n_mask];
    let params = BatchParams {
        ks: &mks,
        p_orbs: &vec![p_orb; n_mask],
        incs_rad: &masked(&incs_rad, &mask),
        a_rs: &vec![a_rs_scalar; n_mask],
        eccs: &masked(&eccs, &mask),
        ws_rad: &masked(&ws_rad, &mask),
        companion_fluxratio: &comp_fr_arr,
        companion_is_host,
    };
    let lnl_masked = batch_transit_lnl(time, flux, sigma, u1, u2, &params);
    probes.dispatched("transit", &lnl_masked);

    let mut lnl = vec![f64::NEG_INFINITY; n];
    let mut it = lnl_masked.iter();
    for j in 0..n {
        if mask[j] {
            lnl[j] = *it.next().unwrap();
        }
    }
    let v = lnz_from_lnl(&lnl);
    if let Some(label) = lnz_label {
        probes.lnz(label, &[v]);
    }
    v
}

/// `_lnZ_TEB` — returns (lnZ_EB, lnZ_EB_twin). `companion_fr` is accepted
/// and ignored, faithfully to the Python (its kernel dilution array is
/// zeros; only `companion_is_host` is forwarded).
#[allow(clippy::too_many_arguments)]
pub fn lnz_teb(
    rng: &mut FppRng,
    probes: &mut Probes,
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    p_orb: f64,
    star: &StarParams,
    n: usize,
    u1: f64,
    u2: f64,
    _companion_fr: f64,
    companion_is_host: bool,
    lnz_label: Option<&str>,
) -> (f64, f64) {
    let (m_s, r_s) = (star.m_s, star.r_s);

    let incs = sample_inc(rng, n);
    probes.sampler("_sample_inc", &incs);
    let qs = sample_q(rng, n, m_s);
    probes.sampler("_sample_q", &qs);
    let eccs = sample_ecc(rng, n, false, p_orb);
    probes.sampler("_sample_ecc", &eccs);
    let argps = sample_w(rng, n);
    probes.sampler("_sample_w", &argps);

    let masses_eb: Vec<f64> = qs.iter().map(|&q| q * m_s).collect();
    let (radii_eb, _teffs) = stellar_relations(&masses_eb);
    let flux_eb = flux_relation(&masses_eb);
    let flux_ms = flux_relation(&[m_s])[0];
    let fluxratios_eb: Vec<f64> = flux_eb.iter().map(|&f| f / (f + flux_ms)).collect();

    let incs_rad: Vec<f64> = incs.iter().map(|&i| i * PI / 180.0).collect();
    let ws_rad: Vec<f64> = argps.iter().map(|&w| (90.0 - w) * PI / 180.0).collect();

    let mut a_cm = Vec::with_capacity(n);
    let mut a_twin_cm = Vec::with_capacity(n);
    for j in 0..n {
        // Same numpy op split as in lnz_ttp: pi**2 via libm pow, array
        // squares via x*x, cube root via libm pow.
        let p_sec = p_orb * 86400.0;
        let p_twin_sec = 2.0 * p_orb * 86400.0;
        a_cm.push(libm_pow(
            G_CGS * (m_s + masses_eb[j]) * MSUN / (4.0 * libm_pow(PI, 2.0)) * (p_sec * p_sec),
            1.0 / 3.0,
        ));
        a_twin_cm.push(libm_pow(
            G_CGS * (m_s + masses_eb[j]) * MSUN / (4.0 * libm_pow(PI, 2.0))
                * (p_twin_sec * p_twin_sec),
            1.0 / 3.0,
        ));
    }
    let a_rs: Vec<f64> = a_cm.iter().map(|&a| a / (r_s * RSUN)).collect();
    let a_twin_rs: Vec<f64> = a_twin_cm.iter().map(|&a| a / (r_s * RSUN)).collect();
    let ks: Vec<f64> = radii_eb.iter().map(|&r| r / r_s).collect();

    let mut mask_eb = vec![false; n];
    let mut mask_twin = vec![false; n];
    for j in 0..n {
        let e_corr = (1.0 + eccs[j] * (argps[j] * PI / 180.0).sin()) / (1.0 - eccs[j] * eccs[j]);
        let ptra = (radii_eb[j] * RSUN + r_s * RSUN) / a_cm[j] * e_corr;
        let ptra_twin = (radii_eb[j] * RSUN + r_s * RSUN) / a_twin_cm[j] * e_corr;
        let inc_min = if ptra <= 1.0 { ptra.acos() } else { PI / 2.0 };
        let inc_min_twin = if ptra_twin <= 1.0 {
            ptra_twin.acos()
        } else {
            PI / 2.0
        };
        let coll = (radii_eb[j] * RSUN + r_s * RSUN) > a_cm[j] * (1.0 - eccs[j]);
        let coll_twin = (2.0 * r_s * RSUN) > a_twin_cm[j] * (1.0 - eccs[j]);
        mask_eb[j] = incs_rad[j] >= inc_min && !coll && qs[j] < 0.95;
        mask_twin[j] = incs_rad[j] >= inc_min_twin && !coll_twin && qs[j] >= 0.95;
    }

    let mut lnl_eb = vec![f64::NEG_INFINITY; n];
    if mask_eb.iter().any(|&m| m) {
        let mks = masked(&ks, &mask_eb);
        let nm = mks.len();
        let params = BatchParams {
            ks: &mks,
            p_orbs: &vec![p_orb; nm],
            incs_rad: &masked(&incs_rad, &mask_eb),
            a_rs: &masked(&a_rs, &mask_eb),
            eccs: &masked(&eccs, &mask_eb),
            ws_rad: &masked(&ws_rad, &mask_eb),
            companion_fluxratio: &vec![0.0; nm],
            companion_is_host,
        };
        let out = batch_eb_transit_lnl(
            time,
            flux,
            sigma,
            u1,
            u2,
            &masked(&fluxratios_eb, &mask_eb),
            &params,
        );
        probes.dispatched("EB", &out);
        let mut it = out.iter();
        for j in 0..n {
            if mask_eb[j] {
                lnl_eb[j] = *it.next().unwrap();
            }
        }
    }

    let mut lnl_twin = vec![f64::NEG_INFINITY; n];
    if mask_twin.iter().any(|&m| m) {
        let mks = masked(&ks, &mask_twin);
        let nm = mks.len();
        let params = BatchParams {
            ks: &mks,
            p_orbs: &masked(&vec![2.0 * p_orb; n], &mask_twin),
            incs_rad: &masked(&incs_rad, &mask_twin),
            a_rs: &masked(&a_twin_rs, &mask_twin),
            eccs: &masked(&eccs, &mask_twin),
            ws_rad: &masked(&ws_rad, &mask_twin),
            companion_fluxratio: &vec![0.0; nm],
            companion_is_host,
        };
        let out = batch_eb_transit_lnl(
            time,
            flux,
            sigma,
            u1,
            u2,
            &masked(&fluxratios_eb, &mask_twin),
            &params,
        );
        probes.dispatched("twin EB", &out);
        let mut it = out.iter();
        for j in 0..n {
            if mask_twin[j] {
                lnl_twin[j] = *it.next().unwrap();
            }
        }
    }

    let lnz_eb = lnz_from_lnl(&lnl_eb);
    let lnz_twin = lnz_from_lnl(&lnl_twin);
    if let Some(label) = lnz_label {
        probes.lnz(label, &[lnz_eb, lnz_twin]);
    }
    (lnz_eb, lnz_twin)
}

/// `_lnZ_PTP` — planet around the physical companion.
#[allow(clippy::too_many_arguments)]
pub fn lnz_ptp(
    rng: &mut FppRng,
    probes: &mut Probes,
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    p_orb: f64,
    star: &StarParams,
    n: usize,
    _u1: f64,
    _u2: f64,
) -> f64 {
    let m_s = star.m_s;
    let qs_comp = sample_q_companion(rng, n, m_s);
    probes.sampler("_sample_q_companion", &qs_comp);
    let masses_comp: Vec<f64> = qs_comp.iter().map(|&q| q * m_s).collect();
    let (radii_comp, teffs_comp) = stellar_relations(&masses_comp);
    let flux_comp = flux_relation(&masses_comp);
    let flux_ms = flux_relation(&[m_s])[0];
    let fluxratios_comp: Vec<f64> = flux_comp.iter().map(|&f| f / (f + flux_ms)).collect();

    let logg_comp: Vec<f64> = masses_comp
        .iter()
        .zip(radii_comp.iter())
        .map(|(&m, &r)| {
            // numpy ARRAY `(radii*Rsun)**2` → npy_pow x*x fast path.
            let rr = r * RSUN;
            (G_CGS * m * MSUN / (rr * rr)).log10()
        })
        .collect();
    let teff_pos: Vec<f64> = teffs_comp.iter().copied().filter(|&t| t > 0.0).collect();
    let logg_fin: Vec<f64> = logg_comp
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let mean_teff = np_mean(&teff_pos);
    let mean_logg = np_mean(&logg_fin);
    let (u1_c, u2_c) = get_ldc(
        if mean_teff.is_finite() {
            mean_teff
        } else {
            5777.0
        },
        if mean_logg.is_finite() {
            mean_logg
        } else {
            4.44
        },
        0.0,
    );

    let comp_fr = np_median(&fluxratios_comp);
    lnz_ttp(
        rng,
        probes,
        time,
        flux,
        sigma,
        p_orb,
        star,
        n,
        u1_c,
        u2_c,
        comp_fr,
        true,
        Some("_lnZ_TTP"),
    )
}

/// `_lnZ_PEB` — EB around the physical companion; returns (PEB, PEBx2P).
#[allow(clippy::too_many_arguments)]
pub fn lnz_peb(
    rng: &mut FppRng,
    probes: &mut Probes,
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    p_orb: f64,
    star: &StarParams,
    n: usize,
    u1: f64,
    u2: f64,
) -> (f64, f64) {
    let m_s = star.m_s;
    let qs_comp = sample_q_companion(rng, n, m_s);
    probes.sampler("_sample_q_companion", &qs_comp);
    let masses_comp: Vec<f64> = qs_comp.iter().map(|&q| q * m_s).collect();
    let flux_comp = flux_relation(&masses_comp);
    let flux_ms = flux_relation(&[m_s])[0];
    let fluxratios_comp: Vec<f64> = flux_comp.iter().map(|&f| f / (f + flux_ms)).collect();

    let comp_fr = np_median(&fluxratios_comp);
    lnz_teb(
        rng,
        probes,
        time,
        flux,
        sigma,
        p_orb,
        star,
        n,
        u1,
        u2,
        comp_fr,
        true,
        Some("_lnZ_TEB"),
    )
}

/// Sequential-order finite sum used by the golden dispatch probes.
pub fn seq_sum_finite(a: &[f64]) -> (f64, usize) {
    let mut s = 0.0;
    let mut n = 0;
    for &v in a {
        if v.is_finite() {
            s += v;
            n += 1;
        }
    }
    (s, n)
}

/// numpy `np.sum` over a small slice (pairwise semantics re-exported for
/// the orchestrator).
pub fn np_sum(a: &[f64]) -> f64 {
    pairwise_sum(a)
}
