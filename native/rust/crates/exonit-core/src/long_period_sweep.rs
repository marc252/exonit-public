//! Transliterated from `Sources/ExoNitCore/LongPeriodSweep.swift`.

use std::fmt;

use crate::bls::{BlsError, BlsParams, BlsResult, BlsSearchFn};
use crate::detrend::DetrendError;
use crate::statistics::median_or;

#[derive(Debug, Clone, PartialEq)]
pub struct LongPeriodSweepOptions {
    pub baseline_threshold_days: f64,
    pub windows_days: Vec<f64>,
    pub period_minimum_days: f64,
    pub legacy_period_maximum_days: f64,
    pub period_maximum_cap_days: f64,
    pub minimum_transits: f64,
    pub base_period_count: usize,
    pub period_count_scale_reference_maximum_days: Option<f64>,
    pub bin_count: usize,
    pub duration_step_count: usize,
    pub duration_minimum_fraction: f64,
    pub duration_maximum_fraction: f64,
    pub minimum_snr: f64,
}

impl Default for LongPeriodSweepOptions {
    fn default() -> Self {
        LongPeriodSweepOptions {
            baseline_threshold_days: 100.0,
            windows_days: vec![2.0, 4.0],
            period_minimum_days: 30.0,
            legacy_period_maximum_days: 50.0,
            period_maximum_cap_days: 200.0,
            minimum_transits: 2.5,
            base_period_count: 50_000,
            period_count_scale_reference_maximum_days: Some(50.0),
            bin_count: 64,
            duration_step_count: 7,
            duration_minimum_fraction: 0.02,
            duration_maximum_fraction: 0.16,
            minimum_snr: 7.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LongPeriodSweepResult {
    pub result: BlsResult,
    pub window_days: f64,
}

/// Mirrors the Swift `LongPeriodDetrender` typealias
/// (`(time:flux:windowDays:mask:) throws -> [Double]`).
pub type LongPeriodDetrender<'a> = &'a dyn Fn(&[f64], &[f64], f64, Option<&[bool]>) -> Result<Vec<f64>, DetrendError>;

/// Unifies the two error sources `LongPeriodSweep.search` can (untyped, in
/// Swift) throw: its own length-mismatch guard, the injected `detrender`
/// (`DetrendError`), and the injected `search` step (`BlsError`) — see the
/// analogous unification in `preprocess::PreprocessError`.
#[derive(Debug, Clone, PartialEq)]
pub enum LongPeriodSweepError {
    LengthMismatch,
    Detrend(DetrendError),
    Bls(BlsError),
}

impl fmt::Display for LongPeriodSweepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LongPeriodSweepError::LengthMismatch => write!(f, "time/flux/mask lengths differ"),
            LongPeriodSweepError::Detrend(e) => write!(f, "detrend failed: {e}"),
            LongPeriodSweepError::Bls(e) => write!(f, "BLS search failed: {e}"),
        }
    }
}

impl std::error::Error for LongPeriodSweepError {}

impl From<DetrendError> for LongPeriodSweepError {
    fn from(value: DetrendError) -> Self {
        LongPeriodSweepError::Detrend(value)
    }
}

impl From<BlsError> for LongPeriodSweepError {
    fn from(value: BlsError) -> Self {
        LongPeriodSweepError::Bls(value)
    }
}

/// Mirrors `LongPeriodSweep.search(time:rawFlux:accumulatedMask:options:detrender:search:)`.
pub fn search(
    time: &[f64],
    raw_flux: &[f64],
    accumulated_mask: &[bool],
    options: &LongPeriodSweepOptions,
    detrender: LongPeriodDetrender,
    search: BlsSearchFn,
) -> Result<Option<LongPeriodSweepResult>, LongPeriodSweepError> {
    if time.len() != raw_flux.len() || time.len() != accumulated_mask.len() {
        return Err(LongPeriodSweepError::LengthMismatch);
    }
    // Swift: `let first = time.first, let last = time.last` are part of the
    // SAME guard as the length checks above, so an empty `time` also throws
    // `lengthMismatch` here (not a distinct "empty" error) — mirrored below.
    let first = match time.first() {
        Some(&v) => v,
        None => return Err(LongPeriodSweepError::LengthMismatch),
    };
    let last = *time.last().expect("non-empty: `first` succeeded above");

    let span = last - first;
    if !(span > options.baseline_threshold_days) {
        return Ok(None);
    }
    let maximum = if span > options.legacy_period_maximum_days * options.minimum_transits {
        (span / options.minimum_transits).min(options.period_maximum_cap_days)
    } else {
        options.legacy_period_maximum_days
    };
    if !(maximum > options.period_minimum_days) {
        return Ok(None);
    }
    let count = match options.period_count_scale_reference_maximum_days {
        Some(reference) => options
            .base_period_count
            .max((options.base_period_count as f64 * maximum / reference) as usize),
        None => options.base_period_count,
    };
    let parameters = BlsParams {
        period_minimum: options.period_minimum_days,
        period_maximum: maximum,
        period_count: count,
        bin_count: options.bin_count,
        duration_step_count: options.duration_step_count,
        duration_minimum_fraction: options.duration_minimum_fraction,
        duration_maximum_fraction: options.duration_maximum_fraction,
    };

    let mut best: Option<LongPeriodSweepResult> = None;
    for &window in &options.windows_days {
        let mask_arg: Option<&[bool]> =
            if accumulated_mask.iter().any(|&m| m) { Some(accumulated_mask) } else { None };
        let mut flattened = detrender(time, raw_flux, window, mask_arg)?;
        if flattened.len() != time.len() {
            return Err(LongPeriodSweepError::LengthMismatch);
        }
        let good_values: Vec<f64> = (0..flattened.len())
            .filter(|&i| !accumulated_mask[i] && flattened[i].is_finite())
            .map(|i| flattened[i])
            .collect();
        let fill = median_or(&good_values, 1.0);
        for i in 0..flattened.len() {
            if accumulated_mask[i] || !flattened[i].is_finite() {
                flattened[i] = fill;
            }
        }
        let result = search(time, &flattened, &parameters)?;
        let better = match &best {
            None => true,
            Some(current) => result.snr > current.result.snr,
        };
        if better {
            best = Some(LongPeriodSweepResult { result, window_days: window });
        }
    }
    match best {
        Some(b) if b.result.snr >= options.minimum_snr => Ok(Some(b)),
        _ => Ok(None),
    }
}
