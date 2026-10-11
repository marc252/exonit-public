//! Transliterated from `Sources/ExoNitCore/IterativeDetrender.swift`.
//!
//! **Design note (deviation from a literal port):** the Swift
//! `IterativeDetrender.flatten` calls `BLSCPUOptimized.search` directly to
//! get its "quick" coarse-BLS pass. `BLSCPUOptimized`/`BLSReference` are F2
//! scope (see docs/PLAN.md — F1 is FITS + preprocessing only). To avoid
//! either reimplementing BLS out of scope or leaving this module unbuildable,
//! the quick-search step is factored out behind the `QuickBlsSearch` trait.
//! F2 will provide the real BLS-backed implementation; the golden test here
//! injects the fixture's own `quick` field as ground truth (that field is
//! exactly this struct — see `scripts/generate_iterative_detrend_golden.py`
//! in the Swift repo), so the masking + refine-detrend logic downstream of a
//! quick-BLS result is still genuinely exercised against the golden fixture.

use crate::detrend::{self, DetrendError};

pub const QUICK_PERIOD_MINIMUM: f64 = 0.5;
pub const QUICK_PERIOD_MAXIMUM: f64 = 50.0;
pub const QUICK_PERIOD_COUNT: u32 = 2_000;
pub const QUICK_BIN_COUNT: u32 = 100;
pub const QUICK_DURATION_STEP_COUNT: u32 = 5;
pub const QUICK_DURATION_MINIMUM_FRACTION: f64 = 0.01;
pub const QUICK_DURATION_MAXIMUM_FRACTION: f64 = 0.15;

/// Mirrors `IterativeDetrender.flatten`'s default `snrThreshold: 7`.
pub const DEFAULT_SNR_THRESHOLD: f64 = 7.0;

/// Coarse BLS search result consumed by the iterative detrender. Field names
/// mirror `BLSResult` as used by the Swift `IterativeDetrendResult`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuickBlsResult {
    pub period: f64,
    pub power: f64,
    pub duration: f64,
    pub t0: f64,
    pub depth: f64,
    pub snr: f64,
}

/// Pluggable quick-BLS step (see module doc comment). Object-safe so callers
/// can hold it as `&dyn QuickBlsSearch`.
pub trait QuickBlsSearch {
    fn search(&self, time: &[f64], flux: &[f64]) -> QuickBlsResult;
}

#[derive(Debug, Clone, PartialEq)]
pub struct IterativeDetrendResult {
    pub flux: Vec<f64>,
    pub first_pass_flux: Vec<f64>,
    pub quick_bls: QuickBlsResult,
    pub transit_mask: Vec<bool>,
    pub was_refined: bool,
}

/// Mirrors `IterativeDetrender.flatten(time:flux:snrThreshold:maxConcurrency:)`.
/// `maxConcurrency` is dropped: it only ever fed the (out-of-scope-for-F1)
/// BLS search, and F1 is single-threaded per docs/PLAN.md (rayon lands
/// later, alongside the real BLS implementation in F2).
pub fn flatten(
    time: &[f64],
    flux: &[f64],
    quick_search: &dyn QuickBlsSearch,
    snr_threshold: f64,
) -> Result<IterativeDetrendResult, DetrendError> {
    flatten_with_backend(time, flux, quick_search, snr_threshold, &detrend::DetrendBackend::Cpu)
}

/// `flatten` restricted to a caller-selected [`detrend::DetrendBackend`]
/// (F4b) — same body as `flatten`, with both `detrend::flatten` calls
/// (first pass, mask=None; refined pass, mask=Some) routed through
/// `detrend::flatten_with_backend` instead. The refined pass still falls
/// through to CPU on a `Cuda` backend because it always passes a mask
/// (`flatten_with_backend`'s own gate — see that function's doc comment),
/// exactly mirroring Python's `_should_use_cuda` behaviour for this call
/// site.
pub fn flatten_with_backend(
    time: &[f64],
    flux: &[f64],
    quick_search: &dyn QuickBlsSearch,
    snr_threshold: f64,
    backend: &detrend::DetrendBackend,
) -> Result<IterativeDetrendResult, DetrendError> {
    if time.len() != flux.len() {
        return Err(DetrendError::LengthMismatch);
    }
    let first = detrend::flatten_with_backend(time, flux, None, backend)?;

    let mut finite_time: Vec<f64> = Vec::new();
    let mut finite_flux: Vec<f64> = Vec::new();
    for index in 0..time.len() {
        if time[index].is_finite() && first[index].is_finite() {
            finite_time.push(time[index]);
            finite_flux.push(first[index]);
        }
    }
    let quick = quick_search.search(&finite_time, &finite_flux);
    if quick.snr < snr_threshold {
        return Ok(IterativeDetrendResult {
            flux: first.clone(),
            first_pass_flux: first,
            quick_bls: quick,
            transit_mask: vec![false; time.len()],
            was_refined: false,
        });
    }

    let duration_fraction = quick.duration / quick.period;
    let mask: Vec<bool> = time
        .iter()
        .map(|&value| {
            // Swift: `(value - t0).truncatingRemainder(dividingBy: period) / period`,
            // then `if phase < 0 { phase += 1 }`. Rust's `%` on f64 is
            // fmod-equivalent (same sign as the dividend), matching
            // `truncatingRemainder` exactly — do NOT use `rem_euclid` here,
            // it has different sign semantics.
            let mut phase = (value - quick.t0) % quick.period / quick.period;
            if phase < 0.0 {
                phase += 1.0;
            }
            phase < duration_fraction || phase > 1.0 - duration_fraction
        })
        .collect();

    let refined = detrend::flatten_with_backend(time, flux, Some(&mask), backend)?;
    Ok(IterativeDetrendResult {
        flux: refined,
        first_pass_flux: first,
        quick_bls: quick,
        transit_mask: mask,
        was_refined: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedQuick(QuickBlsResult);
    impl QuickBlsSearch for FixedQuick {
        fn search(&self, _time: &[f64], _flux: &[f64]) -> QuickBlsResult {
            self.0
        }
    }

    #[test]
    fn below_threshold_snr_skips_refinement() {
        let time: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let flux = vec![1.0; 20];
        let quick = FixedQuick(QuickBlsResult {
            period: 2.0,
            power: 0.1,
            duration: 0.1,
            t0: 0.0,
            depth: 0.0,
            snr: 1.0,
        });
        let result = flatten(&time, &flux, &quick, DEFAULT_SNR_THRESHOLD).unwrap();
        assert!(!result.was_refined);
        assert!(result.transit_mask.iter().all(|&m| !m));
    }
}
