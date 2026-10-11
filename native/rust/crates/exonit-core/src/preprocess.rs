//! Transliterated from `Sources/ExoNitCore/LightCurvePreprocessor.swift`.
//! Reproduces the Python worker's stitch -> clean -> positive sigma clip ->
//! conditional gap fill -> Wotan path.

use crate::detrend::{self, DetrendError};
use crate::detrend_iterative::{self, QuickBlsSearch};
use crate::fits::{LightCurveData, PhotometrySource};
use crate::gaps::{fill_if_allowed, GapFillError, GapFillSeries, SystemGaussianNoiseSource};
use crate::statistics::{median_or, nan_median};

#[derive(Debug, Clone, PartialEq)]
pub struct ProcessedLightCurve {
    pub time: Vec<f64>,
    pub flat_flux: Vec<f64>,
    pub raw_flux: Vec<f64>,
    pub sector_count: usize,
    pub max_gap_days: f64,
}

/// Unified error type for `process()`. Swift's `throws` is untyped and lets
/// `LightCurvePreprocessor.process` propagate either a `DetrendError` or a
/// `GapFillError` interchangeably; Rust needs one concrete type, so this
/// wraps both (language-driven plumbing difference only, no numeric change).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreprocessError {
    LengthMismatch,
    MaskLengthMismatch,
    EmptyInput,
    NonMonotonicTime,
    NonMonotonicCadence,
}

impl std::fmt::Display for PreprocessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            PreprocessError::LengthMismatch => "time and flux lengths differ",
            PreprocessError::MaskLengthMismatch => "transit mask length differs from time",
            PreprocessError::EmptyInput => "empty light curve",
            PreprocessError::NonMonotonicTime => "time is not strictly increasing",
            PreprocessError::NonMonotonicCadence => "cadence is not strictly increasing",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for PreprocessError {}

impl From<DetrendError> for PreprocessError {
    fn from(value: DetrendError) -> Self {
        match value {
            DetrendError::LengthMismatch => PreprocessError::LengthMismatch,
            DetrendError::MaskLengthMismatch => PreprocessError::MaskLengthMismatch,
            DetrendError::EmptyInput => PreprocessError::EmptyInput,
        }
    }
}

impl From<GapFillError> for PreprocessError {
    fn from(value: GapFillError) -> Self {
        match value {
            GapFillError::LengthMismatch => PreprocessError::LengthMismatch,
            GapFillError::NonMonotonicTime => PreprocessError::NonMonotonicTime,
            GapFillError::NonMonotonicCadence => PreprocessError::NonMonotonicCadence,
        }
    }
}

/// Which detrend pass `process` should run. Swift's `iterativeDetrend: Bool`
/// defaults to `true`; Rust has no default parameters, so callers choose
/// explicitly. `Iterative` needs a `QuickBlsSearch` impl (see
/// detrend_iterative.rs doc comment — real BLS is F2 scope).
pub enum DetrendMode<'a> {
    Biweight,
    Iterative(&'a dyn QuickBlsSearch),
}

/// Pre-detrend intermediate: stitched, cleaned (NaN-dropped), positive-
/// sigma-clipped and (conditionally) gap-filled `time`/`raw`, i.e. exactly
/// the arrays `LightCurvePreprocessor.process` hands to the detrend step.
/// Factored out of `process()` so a golden test can inject a real
/// externally-computed quick-BLS result (see detrend_iterative.rs) against
/// the exact same pre-detrend arrays the reference Python pipeline used,
/// without needing real BLS in F1. Also the natural seam for F2's BLS
/// integration later.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedLightCurve {
    pub time: Vec<f64>,
    pub raw: Vec<f64>,
    pub max_gap_days: f64,
}

/// Stitch -> clean -> positive sigma clip -> conditional gap fill. Mirrors
/// the first half of `LightCurvePreprocessor.process` (everything before the
/// detrend call).
pub fn prepare(curves: &[LightCurveData]) -> Result<PreparedLightCurve, PreprocessError> {
    if curves.is_empty() {
        return Err(PreprocessError::EmptyInput);
    }
    let mut time: Vec<f64> = Vec::new();
    let mut raw: Vec<f64> = Vec::new();
    let all_have_cadence = curves.iter().all(|c| c.cadence.is_some());
    let mut cadence: Option<Vec<i64>> = if all_have_cadence { Some(Vec::new()) } else { None };

    for curve in curves {
        if curve.time.len() != curve.flux.len() {
            return Err(PreprocessError::LengthMismatch);
        }
        if let Some(c) = &curve.cadence {
            if c.len() != curve.time.len() {
                return Err(PreprocessError::LengthMismatch);
            }
        }
        if curves.len() == 1 {
            time.extend_from_slice(&curve.time);
            raw.extend_from_slice(&curve.flux);
        } else {
            let normalization = nan_median(&curve.flux);
            if curve.source == PhotometrySource::Tglc {
                // worker.py's _read_lightcurve() upcasts cal_aper_flux to
                // np.float64 BEFORE constructing the LightCurve object, so
                // lightkurve's normalize()/nanmedian() run entirely in
                // float64 for TGLC/FFI -- never touching the FITS `E`
                // column's native Float32 dtype. Divide in double precision
                // to match.
                for index in 0..curve.time.len() {
                    time.push(curve.time[index]);
                    raw.push(curve.flux[index] / normalization);
                }
            } else {
                // SPOC/QLP go through plain lk.read(path), which DOES
                // preserve the FITS `E` column's native Float32 dtype
                // through normalize() -- keep reproducing that single-
                // rounding division.
                let normalization32 = normalization as f32;
                for index in 0..curve.time.len() {
                    time.push(curve.time[index]);
                    raw.push(((curve.flux[index] as f32) / normalization32) as f64);
                }
            }
        }
        if let Some(values) = &curve.cadence {
            if let Some(c) = cadence.as_mut() {
                c.extend_from_slice(values);
            }
        }
    }

    // LightCurve.remove_nans() removes invalid flux cadences before clipping.
    let mut finite_time: Vec<f64> = Vec::with_capacity(time.len());
    let mut finite_raw: Vec<f64> = Vec::with_capacity(time.len());
    let mut finite_cadence: Option<Vec<i64>> = cadence.as_ref().map(|_| Vec::new());
    for index in 0..time.len() {
        if time[index].is_finite() && raw[index].is_finite() {
            finite_time.push(time[index]);
            finite_raw.push(raw[index]);
            if let (Some(c), Some(fc)) = (&cadence, finite_cadence.as_mut()) {
                fc.push(c[index]);
            }
        }
    }
    let keep = upper_sigma_clip_mask(&finite_raw, 5.0, 5);
    let mut time: Vec<f64> = Vec::new();
    let mut raw: Vec<f64> = Vec::new();
    let mut cadence: Option<Vec<i64>> = finite_cadence.as_ref().map(|_| Vec::new());
    for index in 0..keep.len() {
        if keep[index] {
            time.push(finite_time[index]);
            raw.push(finite_raw[index]);
            if let (Some(fc), Some(c)) = (&finite_cadence, cadence.as_mut()) {
                c.push(fc[index]);
            }
        }
    }

    let mut max_gap = 0.0_f64;
    if time.len() > 1 {
        for index in 1..time.len() {
            max_gap = max_gap.max(time[index] - time[index - 1]);
        }
    }
    // Python calls fill_gaps only when the largest observed gap is strictly
    // below five days; its production RNG is deliberately unseeded.
    if time.len() > 1 && max_gap < 5.0 {
        let mut noise = SystemGaussianNoiseSource::new();
        let series = GapFillSeries {
            time: time.clone(),
            flux: raw.clone(),
            cadence: cadence.clone(),
        };
        let filled = fill_if_allowed(series, 5.0, 1.2, None, &mut noise)?;
        time = filled.time;
        raw = filled.flux;
        // ProcessedLightCurve has no cadence field; it only drives gap-grid
        // selection above.
    }

    Ok(PreparedLightCurve {
        time,
        raw,
        max_gap_days: max_gap,
    })
}

/// Mirrors `LightCurvePreprocessor.process(_:iterativeDetrend:maxConcurrency:)`.
/// `maxConcurrency` is dropped for the same reason as in detrend_iterative.rs
/// (single-threaded F1; the knob only ever reached the out-of-scope BLS
/// search).
pub fn process(curves: &[LightCurveData], mode: DetrendMode) -> Result<ProcessedLightCurve, PreprocessError> {
    process_with_backend(curves, mode, &detrend::DetrendBackend::Cpu)
}

/// `process` restricted to a caller-selected [`detrend::DetrendBackend`]
/// (F4b) — same body as `process`, with the detrend step routed through
/// `detrend::flatten_with_backend`/`detrend_iterative::flatten_with_backend`
/// instead of the CPU-only entry points. `process()` above is a thin
/// `Cpu`-backend wrapper kept for source compatibility with every existing
/// call site (tests, `task_processor.rs` pre-F4b) — behaviourally identical
/// to before this function existed.
pub fn process_with_backend(
    curves: &[LightCurveData],
    mode: DetrendMode,
    detrend_backend: &detrend::DetrendBackend,
) -> Result<ProcessedLightCurve, PreprocessError> {
    let prepared = prepare(curves)?;
    let time = prepared.time;
    let raw = prepared.raw;
    let max_gap = prepared.max_gap_days;

    let mut flat: Vec<f64> = match mode {
        DetrendMode::Iterative(search) => {
            detrend_iterative::flatten_with_backend(&time, &raw, search, detrend_iterative::DEFAULT_SNR_THRESHOLD, detrend_backend)?.flux
        }
        DetrendMode::Biweight => detrend::flatten_with_backend(&time, &raw, None, detrend_backend)?,
    };
    let flat_median = nan_median(&flat);
    if flat_median > 1.5 || flat_median < 0.5 {
        for value in flat.iter_mut() {
            *value /= flat_median;
        }
    }

    let mut order: Vec<usize> = (0..time.len()).collect();
    order.sort_by(|&a, &b| time[a].partial_cmp(&time[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut sorted_time: Vec<f64> = Vec::with_capacity(order.len());
    let mut sorted_flat: Vec<f64> = Vec::with_capacity(order.len());
    let mut sorted_raw: Vec<f64> = Vec::with_capacity(order.len());
    for index in order {
        if time[index].is_finite() && flat[index].is_finite() && raw[index].is_finite() {
            sorted_time.push(time[index]);
            sorted_flat.push(flat[index]);
            sorted_raw.push(raw[index]);
        }
    }
    Ok(ProcessedLightCurve {
        time: sorted_time,
        flat_flux: sorted_flat,
        raw_flux: sorted_raw,
        sector_count: curves.len(),
        max_gap_days: max_gap,
    })
}

fn upper_sigma_clip_mask(values: &[f64], sigma: f64, max_iterations: u32) -> Vec<bool> {
    let mut keep = vec![true; values.len()];
    for _ in 0..max_iterations {
        let current: Vec<f64> = (0..values.len()).filter(|&i| keep[i]).map(|i| values[i]).collect();
        if current.is_empty() {
            break;
        }
        let center = median_or(&current, f64::NAN);
        let mean = current.iter().sum::<f64>() / current.len() as f64;
        let variance = current.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / current.len() as f64;
        let limit = center + sigma * variance.sqrt();
        let mut changed = false;
        for index in 0..values.len() {
            if keep[index] && values[index] > limit {
                keep[index] = false;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors ExoNitContractTests.testStitchNormalizationSemantics
    /// (main.swift:1703): a single-segment curve must NOT be
    /// stitch-normalized; a two-segment stitch must normalize each segment
    /// to ~1.
    #[test]
    fn single_segment_is_not_stitch_normalized() {
        let time: Vec<f64> = (0..100).map(|i| i as f64 * 0.02).collect();
        let single = LightCurveData {
            time: time.clone(),
            flux: vec![100.0; time.len()],
            raw_row_count: time.len(),
            cadence: None,
            source: PhotometrySource::Unknown,
        };
        let result = process(std::slice::from_ref(&single), DetrendMode::Biweight).unwrap();
        assert!(!result.raw_flux.is_empty());
        assert!(result.raw_flux.iter().all(|&v| v == 100.0));
    }

    #[test]
    fn two_segment_stitch_normalizes_each_segment() {
        let time: Vec<f64> = (0..100).map(|i| i as f64 * 0.02).collect();
        let second_time: Vec<f64> = time.iter().map(|t| t + 10.0).collect();
        let first = LightCurveData {
            time: time.clone(),
            flux: vec![100.0; time.len()],
            raw_row_count: time.len(),
            cadence: None,
            source: PhotometrySource::Unknown,
        };
        let second = LightCurveData {
            time: second_time,
            flux: vec![200.0; time.len()],
            raw_row_count: time.len(),
            cadence: None,
            source: PhotometrySource::Unknown,
        };
        let result = process(&[first, second], DetrendMode::Biweight).unwrap();
        assert!(!result.raw_flux.is_empty());
        assert!(result.raw_flux.iter().all(|&v| (v - 1.0).abs() < 1e-7));
    }

    #[test]
    fn empty_curves_rejected() {
        assert_eq!(process(&[], DetrendMode::Biweight).unwrap_err(), PreprocessError::EmptyInput);
    }
}
