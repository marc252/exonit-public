//! Transliterated from `Sources/ExoNitCore/BLSReference.swift`.
//!
//! f64 end to end; same iteration/reduction order and tie-breaking as the
//! Swift source (strict `power > bestPower` keeps the *first* maximum seen
//! while scanning duration widths then phase bins, matching Swift's `for
//! width in durationBins { for phaseBin in 0..<binCount { ... } }`).
//!
//! **F7: deliberately left single-threaded.** This module is the
//! ground-truth reference every golden fixture and every parallel BLS
//! implementation (`bls::optimized`, `bls::two_pass`) is checked against —
//! see `bls_golden.rs`'s `reference_search_matches_swift_golden` and the new
//! `bls_rayon_parity.rs`. Parallelizing the reference itself would remove
//! the one BLS code path in this crate that is *by construction*
//! independent of rayon/thread-pool behavior, which is exactly the
//! property the parity tests need on the "known good" side of the
//! comparison. If it is ever benchmarked and found too slow to serve that
//! role, it should be parallelized with the *same* per-period,
//! independent-scratch strategy as `bls::optimized` (see that module's doc
//! comment) — never a design that could, even in principle, make its
//! output depend on thread count.

use std::fmt;

use crate::statistics::median;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlsParams {
    pub period_minimum: f64,
    pub period_maximum: f64,
    pub period_count: usize,
    pub bin_count: usize,
    pub duration_step_count: usize,
    pub duration_minimum_fraction: f64,
    pub duration_maximum_fraction: f64,
}

/// Mirrors `BLSParameters.init`'s defaults.
impl Default for BlsParams {
    fn default() -> Self {
        BlsParams {
            period_minimum: 0.5,
            period_maximum: 50.0,
            period_count: 10_000,
            bin_count: 200,
            duration_step_count: 15,
            duration_minimum_fraction: 0.01,
            duration_maximum_fraction: 0.15,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlsResult {
    pub period: f64,
    pub power: f64,
    pub duration: f64,
    pub t0: f64,
    pub depth: f64,
    pub depth_ppm: f64,
    pub snr: f64,
    pub periods: Vec<f64>,
    pub power_spectrum: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BlsError {
    MismatchedLengths,
    InsufficientSamples,
    InvalidParameters(String),
    /// A GPU backend failed during an actual search call (F4a: CUDA). This
    /// is a real, worth-surfacing failure, not a silent CPU fallback —
    /// mirrors `bls_cuda.py`'s `bls_gpu`/`bls_gpu_two_pass`, which raise
    /// rather than falling back once CUDA has already been selected as the
    /// process's backend (see `bls::accelerated`'s module doc comment and
    /// `bls::cuda`'s probe-once semantics).
    Backend(String),
}

impl fmt::Display for BlsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlsError::MismatchedLengths => write!(f, "time and flux lengths differ"),
            BlsError::InsufficientSamples => write!(f, "BLS requires at least six samples"),
            BlsError::InvalidParameters(reason) => write!(f, "invalid BLS parameters: {reason}"),
            BlsError::Backend(reason) => write!(f, "BLS backend error: {reason}"),
        }
    }
}

impl std::error::Error for BlsError {}

/// Mirrors `BLSReference.search(time:flux:parameters:)`.
pub fn search(time: &[f64], flux: &[f64], p: &BlsParams) -> Result<BlsResult, BlsError> {
    if time.len() != flux.len() {
        return Err(BlsError::MismatchedLengths);
    }
    if time.len() < 6 {
        return Err(BlsError::InsufficientSamples);
    }
    if !(p.period_minimum > 0.0) || !(p.period_maximum > p.period_minimum) {
        return Err(BlsError::InvalidParameters("period range".to_string()));
    }
    if p.period_count < 2 || p.bin_count < 3 || p.duration_step_count < 1 {
        return Err(BlsError::InvalidParameters("grid dimensions".to_string()));
    }
    if !(p.duration_minimum_fraction > 0.0) || !(p.duration_maximum_fraction >= p.duration_minimum_fraction) {
        return Err(BlsError::InvalidParameters("duration range".to_string()));
    }

    let sample_count = time.len();
    let t0_reference = time[0];
    let time_relative: Vec<f64> = time.iter().map(|&v| v - t0_reference).collect();
    let periods = frequency_period_grid(p.period_minimum, p.period_maximum, p.period_count);
    let duration_bins = duration_bin_grid(p);
    let total_sum: f64 = flux.iter().sum();

    let mut best_powers = vec![0.0_f64; p.period_count];
    let mut best_durations = vec![0usize; p.period_count];
    let mut best_phases = vec![0usize; p.period_count];
    let mut best_depths = vec![0.0_f64; p.period_count];

    let mut bin_sums = vec![0.0_f64; p.bin_count];
    let mut bin_counts = vec![0i64; p.bin_count];
    let mut cumulative_sums = vec![0.0_f64; 2 * p.bin_count + 1];
    let mut cumulative_counts = vec![0i64; 2 * p.bin_count + 1];

    for period_index in 0..periods.len() {
        let period = periods[period_index];
        bin_sums.iter_mut().for_each(|v| *v = 0.0);
        bin_counts.iter_mut().for_each(|v| *v = 0);

        for index in 0..sample_count {
            let quotient = time_relative[index] / period;
            let phase = quotient - quotient.floor();
            let raw_bin = (phase * p.bin_count as f64) as i64;
            let bin = raw_bin.clamp(0, p.bin_count as i64 - 1) as usize;
            bin_sums[bin] += flux[index];
            bin_counts[bin] += 1;
        }

        cumulative_sums[0] = 0.0;
        cumulative_counts[0] = 0;
        for index in 0..(2 * p.bin_count) {
            let source = index % p.bin_count;
            cumulative_sums[index + 1] = cumulative_sums[index] + bin_sums[source];
            cumulative_counts[index + 1] = cumulative_counts[index] + bin_counts[source];
        }

        let mut best_power = -1e30_f64;
        let mut best_duration = 0usize;
        let mut best_phase = 0usize;
        let mut best_depth = 0.0_f64;

        for &width in &duration_bins {
            for phase_bin in 0..p.bin_count {
                let window_count = cumulative_counts[phase_bin + width] - cumulative_counts[phase_bin];
                let outside_count = sample_count as i64 - window_count;
                if window_count < 3 || outside_count < 3 {
                    continue;
                }
                let window_sum = cumulative_sums[phase_bin + width] - cumulative_sums[phase_bin];
                let mean_inside = window_sum / window_count as f64;
                let mean_outside = (total_sum - window_sum) / outside_count as f64;
                let depth = mean_outside - mean_inside;
                if depth <= 0.0 {
                    continue;
                }
                let power =
                    depth * depth * (window_count * outside_count) as f64 / (window_count + outside_count) as f64;
                if power > best_power {
                    best_power = power;
                    best_duration = width;
                    best_phase = phase_bin;
                    best_depth = depth;
                }
            }
        }
        best_powers[period_index] = best_power;
        best_durations[period_index] = best_duration;
        best_phases[period_index] = best_phase;
        best_depths[period_index] = best_depth;
    }

    let mut best_index = 0usize;
    for index in 1..best_powers.len() {
        if best_powers[index] > best_powers[best_index] {
            best_index = index;
        }
    }
    let best_period = periods[best_index];
    let depth = best_depths[best_index];
    Ok(BlsResult {
        period: best_period,
        power: best_powers[best_index],
        duration: best_durations[best_index] as f64 / p.bin_count as f64 * best_period,
        t0: best_phases[best_index] as f64 / p.bin_count as f64 * best_period + t0_reference,
        depth,
        depth_ppm: depth * 1_000_000.0,
        snr: sde(&best_powers, best_index),
        periods,
        power_spectrum: best_powers,
    })
}

/// Mirrors `BLSReference.frequencyPeriodGrid(minimum:maximum:count:)`.
pub fn frequency_period_grid(minimum: f64, maximum: f64, count: usize) -> Vec<f64> {
    let start = 1.0 / minimum;
    let end = 1.0 / maximum;
    let step = (end - start) / (count as f64 - 1.0);
    (0..count).map(|i| 1.0 / (start + i as f64 * step)).collect()
}

/// Mirrors `BLSReference.sde(spectrum:peakIndex:)`.
pub fn sde(spectrum: &[f64], peak_index: usize) -> f64 {
    let finite: Vec<f64> = spectrum.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return 0.0;
    }
    let med = median(&finite);
    let deviations: Vec<f64> = finite.iter().map(|v| (v - med).abs()).collect();
    let mad = median(&deviations);
    let sigma = 1.4826 * mad;
    if !(sigma > 0.0) {
        return 0.0;
    }
    (spectrum[peak_index] - med) / sigma
}

/// Mirrors `BLSReference.durationBinGrid(parameters:)`.
pub fn duration_bin_grid(p: &BlsParams) -> Vec<usize> {
    let start = 1usize.max((p.duration_minimum_fraction * p.bin_count as f64) as usize);
    let end = (p.bin_count / 3).min((p.duration_maximum_fraction * p.bin_count as f64) as usize);
    if p.duration_step_count == 1 {
        return vec![start];
    }
    let step = (end as f64 - start as f64) / (p.duration_step_count as f64 - 1.0);
    let mut result: Vec<usize> = Vec::with_capacity(p.duration_step_count);
    for index in 0..p.duration_step_count {
        // Swift `Int(Double(start) + Double(index) * step)` truncates toward
        // zero; `end` can be < `start` for degenerate parameters, making
        // `step` negative, so this must NOT be clamped to zero here.
        let value_f = start as f64 + index as f64 * step;
        let value = value_f as i64 as usize;
        if result.last().copied() != Some(value) {
            result.push(value);
        }
    }
    result
}
