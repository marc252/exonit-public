//! Phase-fold + binning — transliteration of `worker/phase_fold.py`
//! (`fold_order="div_mod"`, `center="mask"`) and
//! `fpp_native._phase_fold_bin` (`worker/fpp_native.py:1119-1136`).

use super::npsum::np_nanmedian;

/// `np.remainder(x, 1.0)` (numpy `%` semantics: result sign follows the
/// divisor).
fn np_mod1(x: f64) -> f64 {
    let mut r = x % 1.0;
    if r != 0.0 && r < 0.0 {
        r += 1.0;
    }
    r
}

/// `phase_fold(time, t0, period, fold_order="div_mod", center="mask")`.
pub fn phase_fold_div_mod_mask(time: &[f64], t0: f64, period: f64) -> Vec<f64> {
    time.iter()
        .map(|&t| {
            let mut p = np_mod1((t - t0) / period);
            if p > 0.5 {
                p -= 1.0;
            }
            p
        })
        .collect()
}

/// `np.searchsorted(bins, x, side='right')` for increasing bins — which is
/// exactly `np.digitize(x, bins)` for increasing bins.
fn searchsorted_right(bins: &[f64], x: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = bins.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if x < bins[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// `_phase_fold_bin` — returns (t_out, f_out).
pub fn phase_fold_bin(
    time: &[f64],
    flux: &[f64],
    period: f64,
    t0: f64,
    n_bins: usize,
) -> (Vec<f64>, Vec<f64>) {
    let phase = phase_fold_div_mod_mask(time, t0, period);
    // np.linspace(-0.5, 0.5, n_bins + 1): y_i = i*step + start; y[-1] = stop
    let step = 1.0 / n_bins as f64;
    let mut edges: Vec<f64> = (0..=n_bins).map(|i| i as f64 * step + -0.5).collect();
    *edges.last_mut().unwrap() = 0.5;
    let centers: Vec<f64> = (0..n_bins)
        .map(|i| 0.5 * (edges[i] + edges[i + 1]))
        .collect();

    let bin_idx: Vec<usize> = phase
        .iter()
        .map(|&p| {
            let i = searchsorted_right(&edges, p) as isize - 1;
            i.clamp(0, n_bins as isize - 1) as usize
        })
        .collect();

    let mut t_out = Vec::new();
    let mut f_out = Vec::new();
    for i in 0..n_bins {
        let vals: Vec<f64> = bin_idx
            .iter()
            .zip(flux.iter())
            .filter(|(&b, _)| b == i)
            .map(|(_, &f)| f)
            .collect();
        if !vals.is_empty() {
            t_out.push(centers[i] * period);
            f_out.push(np_nanmedian(&vals));
        }
    }
    (t_out, f_out)
}

/// `renorm_flux` (`fpp_native.py:1139-1143`) — returns (renormed flux,
/// renormed sigma).
pub fn renorm_flux(flux: &[f64], flux_err: f64, star_fluxratio: f64) -> (Vec<f64>, f64) {
    let renormed = flux
        .iter()
        .map(|&f| (f - (1.0 - star_fluxratio)) / star_fluxratio)
        .collect();
    (renormed, flux_err / star_fluxratio)
}
