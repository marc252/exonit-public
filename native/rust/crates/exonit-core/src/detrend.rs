//! Float64 port of Wotan's `flatten(method="biweight")` path used by ExoNit.
//!
//! Transliterated from `Sources/ExoNitCore/BiweightDetrender.swift` in the
//! validated Swift port (`~/exonit-swift-worker-dev`, commit
//! `5e3978e`). f64 end to end, same iteration/reduction order and
//! tie-breaking as the Swift source — see module-level comments below at
//! each point where that matters.

/// Wotan's biweight defaults, as used everywhere in the codebase (Swift has
/// these as default parameter values; Rust has no default arguments, so they
/// are exposed as constants and `flatten` calls `flatten_with_params` with
/// them directly).
pub const DEFAULT_WINDOW_LENGTH: f64 = 0.5;
pub const DEFAULT_EDGE_CUTOFF: f64 = 0.5;
pub const DEFAULT_BREAK_TOLERANCE: f64 = 0.5;
pub const DEFAULT_CVAL: f64 = 5.0;

use crate::statistics::median;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetrendError {
    LengthMismatch,
    MaskLengthMismatch,
    EmptyInput,
}

impl std::fmt::Display for DetrendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            DetrendError::LengthMismatch => "time and flux lengths differ",
            DetrendError::MaskLengthMismatch => "transit mask length differs from time",
            DetrendError::EmptyInput => "empty light curve",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for DetrendError {}

/// Selects which implementation [`flatten_with_backend`] uses. `Cpu` is
/// this module's own `flatten`/`flatten_with_params` (always available);
/// `Cuda` (feature `cuda`, F4b) wraps an already-`probe()`d
/// `detrend_cuda::CudaDetrendBackend` — mirrors `wotan_detrend.py`'s
/// `EXONIT_WOTAN_BACKEND` selection, resolved ONCE by
/// `task_processor.rs`'s `resolve_detrend_backend` (see that function's doc
/// comment), not re-probed per call.
#[derive(Clone)]
pub enum DetrendBackend {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(std::sync::Arc<crate::detrend_cuda::CudaDetrendBackend>),
}

impl std::fmt::Debug for DetrendBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DetrendBackend::Cpu => f.write_str("Cpu"),
            #[cfg(feature = "cuda")]
            DetrendBackend::Cuda(_) => f.write_str("Cuda(..)"),
        }
    }
}

/// `flatten` restricted to a caller-selected backend. Mirrors
/// `wotan_detrend.detrend_with_wotan`'s CUDA branch exactly: CUDA only
/// applies when `transit_mask.is_none()` (`_should_use_cuda`'s `if
/// transit_mask is not None: return False` — the iterative detrender's
/// refined pass, which always passes a mask, therefore ALWAYS stays on CPU
/// here regardless of `backend`, same as Python); a CUDA kernel/launch
/// failure logs and falls through to the CPU path (`except Exception` ->
/// `_wotan_flatten`), never silently — not a panic, not a swallowed error.
pub fn flatten_with_backend(
    time: &[f64],
    flux: &[f64],
    transit_mask: Option<&[bool]>,
    backend: &DetrendBackend,
) -> Result<Vec<f64>, DetrendError> {
    match backend {
        DetrendBackend::Cpu => flatten(time, flux, transit_mask),
        #[cfg(feature = "cuda")]
        DetrendBackend::Cuda(cuda) => {
            if transit_mask.is_some() {
                return flatten(time, flux, transit_mask);
            }
            match cuda.detrend_biweight(time, flux, DEFAULT_WINDOW_LENGTH, DEFAULT_EDGE_CUTOFF, DEFAULT_CVAL, DEFAULT_BREAK_TOLERANCE) {
                Ok((flat, _trend)) => Ok(flat),
                Err(error) => {
                    eprintln!("detrend: CUDA wotan path failed ({error}); falling back to wotan CPU.");
                    flatten(time, flux, transit_mask)
                }
            }
        }
    }
}

/// Biweight-flatten `flux` against `time` using Wotan's defaults. Mirrors
/// `BiweightDetrender.flatten(time:flux:transitMask:)` (Swift default
/// parameters for windowLength/edgeCutoff/breakTolerance/cval).
pub fn flatten(
    time: &[f64],
    flux: &[f64],
    transit_mask: Option<&[bool]>,
) -> Result<Vec<f64>, DetrendError> {
    flatten_with_params(
        time,
        flux,
        DEFAULT_WINDOW_LENGTH,
        DEFAULT_EDGE_CUTOFF,
        DEFAULT_BREAK_TOLERANCE,
        DEFAULT_CVAL,
        transit_mask,
    )
}

/// Full-parameter form; mirrors `BiweightDetrender.flatten` exactly.
pub fn flatten_with_params(
    time: &[f64],
    flux: &[f64],
    window_length: f64,
    edge_cutoff: f64,
    break_tolerance: f64,
    cval: f64,
    transit_mask: Option<&[bool]>,
) -> Result<Vec<f64>, DetrendError> {
    if time.len() != flux.len() {
        return Err(DetrendError::LengthMismatch);
    }
    if time.is_empty() {
        return Err(DetrendError::EmptyInput);
    }
    if let Some(mask) = transit_mask {
        if mask.len() != time.len() {
            return Err(DetrendError::MaskLengthMismatch);
        }
    }

    let mut compressed_time: Vec<f64> = Vec::with_capacity(time.len());
    let mut compressed_flux: Vec<f64> = Vec::with_capacity(time.len());
    let mut compressed_mask: Vec<f64> = Vec::with_capacity(time.len());
    let mut original_indexes: Vec<usize> = Vec::with_capacity(time.len());

    for index in 0..time.len() {
        // Swift: `where !(time[index] * flux[index]).isNaN` — catches a NaN
        // in either operand (NaN propagates through multiplication).
        if !(time[index] * flux[index]).is_nan() {
            compressed_time.push(time[index]);
            compressed_flux.push(flux[index]);
            let masked = transit_mask.map(|m| m[index]).unwrap_or(false);
            compressed_mask.push(if masked { 0.0 } else { 1.0 });
            original_indexes.push(index);
        }
    }
    if compressed_time.is_empty() {
        return Ok(vec![f64::NAN; time.len()]);
    }

    let mut boundaries: Vec<usize> = vec![0];
    if compressed_time.len() > 1 {
        for index in 1..compressed_time.len() {
            if compressed_time[index] - compressed_time[index - 1] > break_tolerance {
                boundaries.push(index);
            }
        }
    }
    // Wotan appends len(time)+1; Python slicing clamps this to the array
    // end, mirrored below via `.min(compressed_time.len())`.
    boundaries.push(compressed_time.len() + 1);

    let mut compressed_trend: Vec<f64> = Vec::with_capacity(compressed_time.len());
    for segment in 0..boundaries.len() - 1 {
        let lower = boundaries[segment];
        let upper = boundaries[segment + 1].min(compressed_time.len());
        if lower >= upper {
            continue;
        }
        compressed_trend.extend(running_segment(
            &compressed_time[lower..upper],
            &compressed_flux[lower..upper],
            &compressed_mask[lower..upper],
            window_length,
            edge_cutoff,
            cval,
        ));
    }

    let mut trend = vec![f64::NAN; time.len()];
    for (index, &orig_index) in original_indexes.iter().enumerate() {
        let value = compressed_trend[index];
        trend[orig_index] = if value == 0.0 { f64::NAN } else { value };
    }
    Ok(flux.iter().zip(trend.iter()).map(|(f, t)| f / t).collect())
}

fn running_segment(
    time: &[f64],
    flux: &[f64],
    mask: &[f64],
    window_length: f64,
    requested_edge_cutoff: f64,
    cval: f64,
) -> Vec<f64> {
    let count = time.len();
    let mut output = vec![f64::NAN; count];
    let half_window = window_length / 2.0;
    let edge_cutoff = requested_edge_cutoff.min(half_window);
    let low = time.iter().copied().reduce(f64::min).unwrap_or(0.0) + edge_cutoff;
    let high = time.iter().copied().reduce(f64::max).unwrap_or(0.0) - edge_cutoff;
    let mut start = 0usize;
    let mut end = 0usize;
    let last_index = count.saturating_sub(1);

    for index in 0..count {
        if !(time[index] >= low && time[index] <= high) {
            continue;
        }
        while start < count && time[start] < time[index] - half_window {
            start += 1;
        }
        while end < last_index && time[end] < time[index] + half_window {
            end += 1;
        }
        if start >= end {
            continue;
        }
        let mut window: Vec<f64> = Vec::with_capacity(end - start);
        for sample in start..end {
            let masked_flux = if mask[sample] == 0.0 {
                -flux[sample]
            } else {
                flux[sample]
            };
            if masked_flux > -1e-12 {
                window.push(masked_flux);
            }
        }
        if !window.is_empty() {
            output[index] = biweight_location(&window, cval);
        }
    }
    output
}

fn biweight_location(data: &[f64], cval: f64) -> f64 {
    let initial = median(data);
    let mut deviations: Vec<f64> = data.iter().map(|v| (v - initial).abs()).collect();
    let mad = median_in_place(&mut deviations);
    if mad == 0.0 {
        return initial;
    }

    let inverse_cmad = 1.0 / (cval * mad);
    let mut center = initial;
    let mut old_center = initial;
    let mut delta = f64::INFINITY;
    while delta.abs() > 1e-6 {
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        // Iterate `data` in its original (unsorted) order — same reduction
        // order as the Swift `for value in data` loop.
        for &value in data {
            let distance = value - center;
            let scaled = distance * inverse_cmad;
            if scaled.abs() < 1.0 {
                let base = 1.0 - scaled * scaled;
                let weight = base * base;
                numerator += distance * weight;
                denominator += weight;
            }
        }
        center += numerator / denominator;
        delta = old_center - center;
        old_center = center;
    }
    center
}

fn median_in_place(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_mismatch_is_rejected() {
        assert_eq!(
            flatten(&[1.0, 2.0], &[1.0], None).unwrap_err(),
            DetrendError::LengthMismatch
        );
    }

    #[test]
    fn empty_input_is_rejected() {
        assert_eq!(flatten(&[], &[], None).unwrap_err(), DetrendError::EmptyInput);
    }

    #[test]
    fn mask_length_mismatch_is_rejected() {
        assert_eq!(
            flatten(&[1.0, 2.0], &[1.0, 1.0], Some(&[true])).unwrap_err(),
            DetrendError::MaskLengthMismatch
        );
    }
}
