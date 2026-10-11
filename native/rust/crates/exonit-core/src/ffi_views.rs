//! Transliterated from `Sources/ExoNitCore/FFIViewGenerator.swift`, cross-
//! checked against the Python reference the worker actually runs in
//! production (`~/exonit/worker/cnn_ffi_views.py` — read-only,
//! not committed here). The two agree on every step (phase-fold, median
//! filter, empty-bin interpolation, odd/even/secondary auxiliary views,
//! normalization) modulo language idiom; this file follows the Swift
//! naming/shape but the numeric behaviour is the Python worker's, which is
//! what the ONNX models were trained against.
//!
//! Determinism per CLAUDE.md / DESIGN.md: everything up to the two
//! `normalize()` calls stays `f64`; the cast to `f32` happens exactly once,
//! at the same point Swift casts to `Float` and Python does
//! `.astype(np.float32)` — right before the values leave this module bound
//! for an ONNX model input.

use std::cmp::Ordering;

use crate::statistics::median;

pub const GLOBAL_BIN_COUNT: usize = 201;
pub const LOCAL_BIN_COUNT: usize = 61;

/// Mirrors Swift's `FFIViewError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiViewError {
    MismatchedSeries,
    InvalidEphemeris,
    SparseLightCurve,
    AllBinsEmpty,
}

impl std::fmt::Display for FfiViewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FfiViewError::MismatchedSeries => write!(f, "time and flux series have different lengths"),
            FfiViewError::InvalidEphemeris => write!(f, "period/duration must be positive"),
            FfiViewError::SparseLightCurve => write!(f, "sparse light curve in view window"),
            FfiViewError::AllBinsEmpty => write!(f, "all bins empty"),
        }
    }
}

impl std::error::Error for FfiViewError {}

/// Mirrors Swift's `CNNFFIViews`.
#[derive(Debug, Clone, PartialEq)]
pub struct CnnFfiViews {
    pub global: Vec<f32>,
    pub local: Vec<f32>,
}

/// Mirrors `FFIViewGenerator.makeCNNViews(time:flux:period:t0:duration:)`.
pub fn make_cnn_views(time: &[f64], flux: &[f64], period: f64, t0: f64, duration: f64) -> Result<CnnFfiViews, FfiViewError> {
    if time.len() != flux.len() {
        return Err(FfiViewError::MismatchedSeries);
    }
    if !(period > 0.0) || !(duration > 0.0) {
        return Err(FfiViewError::InvalidEphemeris);
    }
    let (phase, flux_sorted) = phase_fold_sort(time, flux, period, t0, duration);
    let global = normalize(&median_filter(
        &phase,
        &flux_sorted,
        GLOBAL_BIN_COUNT,
        period * (1.2 / GLOBAL_BIN_COUNT as f64),
        -period / 2.0,
        period / 2.0,
    )?);
    let half_window = (period / 2.0).min(2.0 * duration);
    let local = normalize(&median_filter(
        &phase,
        &flux_sorted,
        LOCAL_BIN_COUNT,
        duration * 0.16,
        -half_window,
        half_window,
    )?);
    Ok(CnnFfiViews { global, local })
}

/// Mirrors `FFIViewGenerator.makeAuxiliaryViews(time:flux:period:t0:duration:)`.
/// Always returns 3 rows of `LOCAL_BIN_COUNT` zeros for channels that cannot
/// be computed (sparse odd/even split, too few points, etc.) — never errors,
/// matching Swift/Python's "conservative zero" behaviour.
pub fn make_auxiliary_views(time: &[f64], flux: &[f64], period: f64, t0: f64, duration: f64) -> Vec<Vec<f32>> {
    let mut output = vec![vec![0.0f32; LOCAL_BIN_COUNT]; 3];
    if time.len() != flux.len() || !(period > 0.0) || !(duration > 0.0) || time.len() < 50 {
        return output;
    }
    let center = t0 + duration / 2.0;
    let mut odd_time = Vec::new();
    let mut odd_flux = Vec::new();
    let mut even_time = Vec::new();
    let mut even_flux = Vec::new();
    for index in 0..time.len() {
        let transit = round_ties_even((time[index] - center) / period);
        let remainder = transit % 2.0;
        if remainder != 0.0 {
            odd_time.push(time[index]);
            odd_flux.push(flux[index]);
        } else {
            even_time.push(time[index]);
            even_flux.push(flux[index]);
        }
    }
    let odd = if odd_time.len() >= 50 { raw_local(&odd_time, &odd_flux, period, t0, duration) } else { None };
    let even = if even_time.len() >= 50 { raw_local(&even_time, &even_flux, period, t0, duration) } else { None };
    if let (Some(odd), Some(even)) = (&odd, &even) {
        let combined: Vec<f64> = odd.iter().chain(even.iter()).copied().collect();
        let center = median(&combined);
        let amplitude = combined.iter().map(|v| v - center).fold(f64::INFINITY, f64::min).abs().max(1e-12);
        output[0] = odd.iter().map(|v| ((v - center) / amplitude) as f32).collect();
        output[1] = even.iter().map(|v| ((v - center) / amplitude) as f32).collect();
    }
    let primary = raw_local(time, flux, period, t0, duration);
    let secondary = raw_local(time, flux, period, t0 + period / 2.0, duration);
    if let (Some(primary), Some(secondary)) = (&primary, &secondary) {
        let center = median(primary);
        let amplitude = primary.iter().map(|v| v - center).fold(f64::INFINITY, f64::min).abs().max(1e-12);
        output[2] = secondary.iter().map(|v| ((v - center) / amplitude) as f32).collect();
    }
    output
}

/// Mirrors `FFIViewGenerator.sigmoid(logit:)`.
pub fn sigmoid(logit: f64) -> f64 {
    let bounded = logit.max(-60.0).min(60.0);
    1.0 / (1.0 + (-bounded).exp())
}

/// Mirrors the private `phaseFoldSort`. Returns `(phase, flux)` sorted by
/// phase ascending; the sort is stable (matches Swift's `sorted(by:)`,
/// stable since Swift 5) so tied phases keep their original relative order.
fn phase_fold_sort(time: &[f64], flux: &[f64], period: f64, t0: f64, duration: f64) -> (Vec<f64>, Vec<f64>) {
    let center = t0 + duration / 2.0;
    let mut samples: Vec<(f64, f64)> = time
        .iter()
        .zip(flux.iter())
        .map(|(&sample_time, &sample_flux)| {
            let mut phase = (sample_time - center) % period;
            if phase < 0.0 {
                phase += period;
            }
            if phase >= period / 2.0 {
                phase -= period;
            }
            (phase, sample_flux)
        })
        .collect();
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    (samples.iter().map(|s| s.0).collect(), samples.iter().map(|s| s.1).collect())
}

/// Mirrors the private `rawLocal`: same window/binning as the local view but
/// returns `None` on `sparseLightCurve` instead of throwing, matching
/// `try?` in Swift (`AllBinsEmpty` is not expected to occur when the range
/// check already passed, but is treated the same way for safety).
fn raw_local(time: &[f64], flux: &[f64], period: f64, t0: f64, duration: f64) -> Option<Vec<f64>> {
    let (phase, flux_sorted) = phase_fold_sort(time, flux, period, t0, duration);
    let half_window = (period / 2.0).min(2.0 * duration);
    median_filter(&phase, &flux_sorted, LOCAL_BIN_COUNT, duration * 0.16, -half_window, half_window).ok()
}

/// Mirrors the private `medianFilter`.
fn median_filter(x: &[f64], y: &[f64], count: usize, width: f64, minimum: f64, maximum: f64) -> Result<Vec<f64>, FfiViewError> {
    let in_range_count = x.iter().filter(|&&value| value >= minimum && value <= maximum).count();
    let (Some(&first), Some(&last)) = (x.first(), x.last()) else {
        return Err(FfiViewError::SparseLightCurve);
    };
    if in_range_count < 5 || last - first < (maximum - minimum) / 2.0 {
        return Err(FfiViewError::SparseLightCurve);
    }
    let spacing = (maximum - minimum - width) / (count - 1) as f64;
    let mut bins: Vec<Option<f64>> = vec![None; count];
    for (index, bin) in bins.iter_mut().enumerate() {
        let lower = minimum + index as f64 * spacing;
        let upper = lower + width;
        let values: Vec<f64> = x
            .iter()
            .zip(y.iter())
            .filter(|(&phase, _)| phase >= lower && phase < upper)
            .map(|(_, &value)| value)
            .collect();
        if !values.is_empty() {
            *bin = Some(median(&values));
        }
    }
    interpolate_empty(&bins)
}

/// Mirrors the private `interpolateEmpty`.
fn interpolate_empty(values: &[Option<f64>]) -> Result<Vec<f64>, FfiViewError> {
    let known: Vec<usize> = values.iter().enumerate().filter_map(|(i, v)| v.map(|_| i)).collect();
    let (Some(&first), Some(&last)) = (known.first(), known.last()) else {
        return Err(FfiViewError::AllBinsEmpty);
    };
    let mut result = vec![0.0f64; values.len()];
    for index in 0..values.len() {
        if let Some(value) = values[index] {
            result[index] = value;
        } else if index < first {
            result[index] = values[first].unwrap();
        } else if index > last {
            result[index] = values[last].unwrap();
        } else {
            let upper = *known.iter().find(|&&k| k > index).unwrap();
            let lower = *known.iter().rev().find(|&&k| k < index).unwrap();
            let fraction = (index - lower) as f64 / (upper - lower) as f64;
            result[index] = values[lower].unwrap() + fraction * (values[upper].unwrap() - values[lower].unwrap());
        }
    }
    Ok(result)
}

/// Mirrors the private `normalize`: median -> 0, |min| -> 1, cast to `f32`
/// only here (the single f64->f32 boundary in the whole module).
fn normalize(values: &[f64]) -> Vec<f32> {
    let center = median(values);
    let shifted: Vec<f64> = values.iter().map(|v| v - center).collect();
    let amplitude = shifted.iter().copied().fold(f64::INFINITY, f64::min).abs().max(1e-12);
    shifted.iter().map(|v| (v / amplitude) as f32).collect()
}

/// Mirrors the private `median`.
/// Round-half-to-even (banker's rounding), matching Swift's
/// `.rounded(.toNearestOrEven)` and NumPy's default `np.round` — both used
/// by the reference implementations at this exact call site (transit-number
/// parity for the odd/even auxiliary split).
fn round_ties_even(value: f64) -> f64 {
    value.round_ties_even()
}
