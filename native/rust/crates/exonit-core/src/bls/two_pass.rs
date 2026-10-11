//! Transliterated from `Sources/ExoNitCore/BLSTwoPass.swift`
//! ("Exact control-flow port of `bls_fast.bls_two_pass`.").
//!
//! F7: both the coarse and fine passes are `optimized::search`, already
//! parallelized per-period with rayon (see that module's doc comment for
//! the determinism argument). This function wraps **both** calls in a
//! single `max_concurrency`-sized thread pool (`super::with_pool`) rather
//! than letting each call build its own — same result either way (each
//! pass is independently deterministic regardless of which pool runs it),
//! just one thread-pool construction per two-pass search instead of two.

use super::optimized;
use super::reference::{BlsError, BlsParams, BlsResult};
use super::with_pool;

/// Mirrors `BLSTwoPass.search(time:flux:parameters:maxConcurrency:)`.
pub fn search(time: &[f64], flux: &[f64], p: &BlsParams, max_concurrency: usize) -> Result<BlsResult, BlsError> {
    with_pool(max_concurrency, || search_body(time, flux, p))
}

fn search_body(time: &[f64], flux: &[f64], p: &BlsParams) -> Result<BlsResult, BlsError> {
    let coarse_count = 5_000usize.min(p.period_count);
    let mut coarse_params = *p;
    coarse_params.period_count = coarse_count;
    optimized::validate(time, flux, &coarse_params)?;
    let coarse = optimized::search_in_pool(time, flux, &coarse_params)?;
    if coarse.snr < 3.0 {
        return Ok(coarse);
    }

    let coarse_step = (p.period_maximum - p.period_minimum) / coarse_count as f64;
    let refine_half_width = (coarse_step * 10.0).max(coarse.period * 0.02);
    let mut fine_params = *p;
    fine_params.period_minimum = p.period_minimum.max(coarse.period - refine_half_width);
    fine_params.period_maximum = p.period_maximum.min(coarse.period + refine_half_width);
    fine_params.period_count = (p.period_count / 5).min(10_000);
    if fine_params.period_count < 2 {
        return Ok(coarse);
    }
    optimized::validate(time, flux, &fine_params)?;
    let fine = optimized::search_in_pool(time, flux, &fine_params)?;
    Ok(BlsResult {
        period: fine.period,
        power: fine.power,
        duration: fine.duration,
        t0: fine.t0,
        depth: fine.depth,
        depth_ppm: fine.depth_ppm,
        snr: coarse.snr,
        periods: coarse.periods,
        power_spectrum: coarse.power_spectrum,
    })
}
