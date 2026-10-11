//! Transit model — transliteration of `worker/fpp_native.py:221-326`
//! (numba ring-integration path; njit == py_func bit-wise per the probe in
//! `fixtures/fpp/transit_golden.json`).

/// `_kepler_eq` — Newton solve of M = E - ecc·sin(E), tol 1e-10, 30 iters.
pub fn kepler_eq(m: f64, ecc: f64) -> f64 {
    let tol = 1e-10;
    let mut e = m;
    for _ in 0..30 {
        let de = (e - ecc * e.sin() - m) / (1.0 - ecc * e.cos());
        e -= de;
        if de.abs() < tol {
            break;
        }
    }
    e
}

/// `_true_anomaly`.
pub fn true_anomaly(m: f64, ecc: f64) -> f64 {
    if ecc < 1e-10 {
        return m;
    }
    let e = kepler_eq(m, ecc);
    let cos_e = e.cos();
    ((1.0 - ecc * ecc).sqrt() * e.sin()).atan2(cos_e - ecc)
}

/// `_compute_z_arr` — projected separation z(t) in units of R_star.
pub fn compute_z_arr(
    time: &[f64],
    t0: f64,
    period: f64,
    a_rs: f64,
    inc_rad: f64,
    ecc: f64,
    w_rad: f64,
) -> Vec<f64> {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut z = Vec::with_capacity(time.len());
    for &ti in time {
        let phase = (ti - t0) / period;
        let m = two_pi * (phase - phase.floor()) - w_rad;
        let nu = true_anomaly(m, ecc);
        let r_factor = (1.0 - ecc * ecc) / (1.0 + ecc * nu.cos());
        let f = nu + w_rad;
        let sin_f = f.sin();
        let cos_f = f.cos();
        let cos_i = inc_rad.cos();
        z.push(a_rs * r_factor * (sin_f * sin_f + cos_i * cos_i * cos_f * cos_f).sqrt());
    }
    z
}

/// `_transit_flux_ring` — flux for one (z, p) via 100-ring integration.
pub fn transit_flux_ring(z: f64, p: f64, u1: f64, u2: f64) -> f64 {
    const N_RINGS: usize = 100;
    if z >= 1.0 + p {
        return 1.0;
    }
    if p <= 0.0 {
        return 1.0;
    }
    let mut omega = 1.0 - u1 / 3.0 - u2 / 6.0;
    if omega <= 0.0 {
        omega = 1.0;
    }
    let dr = 1.0 / N_RINGS as f64;
    let mut blocked = 0.0;
    for i in 0..N_RINGS {
        let r = (i as f64 + 0.5) * dr;
        let r2 = r * r;
        let mu = (1.0 - r2).sqrt();
        let one_minus_mu = 1.0 - mu;
        let i_r = 1.0 - u1 * one_minus_mu - u2 * one_minus_mu * one_minus_mu;
        let frac = if z < 1e-12 {
            if r <= p {
                1.0
            } else {
                0.0
            }
        } else if r >= z + p {
            0.0
        } else if r + p <= z {
            0.0
        } else if r <= (z - p).abs() {
            if p > z {
                1.0
            } else {
                0.0
            }
        } else {
            let arg = (r2 + z * z - p * p) / (2.0 * r * z);
            if arg >= 1.0 {
                0.0
            } else if arg <= -1.0 {
                1.0
            } else {
                arg.acos() / std::f64::consts::PI
            }
        };
        blocked += i_r * frac * 2.0 * r * dr;
    }
    1.0 - blocked / omega
}

/// `_compute_transit_lc` — full light curve for one parameter set.
#[allow(clippy::too_many_arguments)]
pub fn compute_transit_lc(
    time: &[f64],
    k: f64,
    t0: f64,
    period: f64,
    a_rs: f64,
    inc_rad: f64,
    ecc: f64,
    w_rad: f64,
    u1: f64,
    u2: f64,
) -> Vec<f64> {
    let z = compute_z_arr(time, t0, period, a_rs, inc_rad, ecc, w_rad);
    z.into_iter()
        .map(|zi| transit_flux_ring(zi, k, u1, u2))
        .collect()
}
