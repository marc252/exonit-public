//! Transliterated from `Sources/ExoNitCore/GapFiller.swift`.
//!
//! Note on `SystemGaussianNoiseSource`: production gap-fill noise is
//! *intentionally* unseeded/stochastic (mirrors Lightkurve's own behaviour;
//! see `LightCurvePreprocessor.process` doc comment in the Swift source), so
//! there is no golden fixture pinning its exact draws — only
//! `testGapFillSemantics`-style tests with a deterministic replay source
//! (ported below as unit tests). Rather than pull in a `rand` crate
//! dependency (not on F1's allowed-dependency list — see docs/PLAN.md), the
//! system source here is a small self-contained SplitMix64 PRNG seeded from
//! OS time + a process-local counter. It is a legitimate uniform generator
//! (SplitMix64 is a standard, well-tested PRNG), just not the same generator
//! Swift's `Double.random` uses under the hood — which is fine, since
//! neither is deterministic/tested for exact values in production.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::statistics::median;

#[derive(Debug, Clone, PartialEq)]
pub struct GapFillSeries {
    pub time: Vec<f64>,
    pub flux: Vec<f64>,
    pub cadence: Option<Vec<i64>>,
}

pub trait GaussianNoiseSource {
    fn normal(&mut self, mean: f64, standard_deviation: f64) -> f64;
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform value in [0, 1) with 53 bits of mantissa precision.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

static SEED_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_seed() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
    nanos ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// System-entropy Gaussian source (Box-Muller with spare-value caching),
/// mirroring `SystemGaussianNoiseSource` in the Swift source.
pub struct SystemGaussianNoiseSource {
    spare: Option<f64>,
    rng: SplitMix64,
}

impl SystemGaussianNoiseSource {
    pub fn new() -> Self {
        Self {
            spare: None,
            rng: SplitMix64::new(next_seed()),
        }
    }
}

impl Default for SystemGaussianNoiseSource {
    fn default() -> Self {
        Self::new()
    }
}

impl GaussianNoiseSource for SystemGaussianNoiseSource {
    fn normal(&mut self, mean: f64, standard_deviation: f64) -> f64 {
        if let Some(spare) = self.spare.take() {
            return mean + standard_deviation * spare;
        }
        let mut u1 = self.rng.next_f64();
        while u1 == 0.0 {
            u1 = self.rng.next_f64();
        }
        let u2 = self.rng.next_f64();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(radius * angle.sin());
        mean + standard_deviation * radius * angle.cos()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapFillError {
    LengthMismatch,
    NonMonotonicTime,
    NonMonotonicCadence,
}

impl std::fmt::Display for GapFillError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            GapFillError::LengthMismatch => "time/flux/cadence lengths differ",
            GapFillError::NonMonotonicTime => "time is not strictly increasing",
            GapFillError::NonMonotonicCadence => "cadence is not strictly increasing",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for GapFillError {}

/// Mirrors `GapFiller.fillIfAllowed(_:maximumGlobalGapDays:spacingTolerance:standardDeviation:noise:)`.
/// Rust has no default parameter values, so all Swift defaults
/// (`maximumGlobalGapDays: 5`, `spacingTolerance: 1.2`, `standardDeviation: nil`)
/// must be passed explicitly by the caller.
pub fn fill_if_allowed<R: GaussianNoiseSource>(
    input: GapFillSeries,
    maximum_global_gap_days: f64,
    spacing_tolerance: f64,
    supplied_deviation: Option<f64>,
    noise: &mut R,
) -> Result<GapFillSeries, GapFillError> {
    if input.time.len() != input.flux.len() {
        return Err(GapFillError::LengthMismatch);
    }
    if let Some(cadence) = &input.cadence {
        if cadence.len() != input.time.len() {
            return Err(GapFillError::LengthMismatch);
        }
    }
    if input.time.len() < 2 {
        return Ok(input);
    }
    let differences: Vec<f64> = input.time.windows(2).map(|w| w[1] - w[0]).collect();
    if !differences.iter().all(|&d| d > 0.0) {
        return Err(GapFillError::NonMonotonicTime);
    }
    let max_diff = differences.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !(max_diff < maximum_global_gap_days) {
        return Ok(input);
    }

    let finite_flux: Vec<f64> = input.flux.iter().copied().filter(|v| v.is_finite()).collect();
    let mean = if finite_flux.is_empty() {
        f64::NAN
    } else {
        finite_flux.iter().sum::<f64>() / finite_flux.len() as f64
    };
    let deviation = if let Some(d) = supplied_deviation {
        d
    } else if finite_flux.is_empty() {
        f64::NAN
    } else {
        let variance = finite_flux.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>()
            / finite_flux.len() as f64;
        variance.sqrt()
    };

    if let Some(cadence) = input.cadence.clone() {
        fill_cadence_grid(input, &cadence, median(&differences), mean, deviation, noise)
    } else {
        Ok(fill_time_grid(
            input,
            median(&differences),
            spacing_tolerance,
            mean,
            deviation,
            noise,
        ))
    }
}

fn fill_cadence_grid<R: GaussianNoiseSource>(
    input: GapFillSeries,
    cadence: &[i64],
    spacing: f64,
    mean: f64,
    deviation: f64,
    noise: &mut R,
) -> Result<GapFillSeries, GapFillError> {
    let cadence_diff: Vec<i64> = cadence.windows(2).map(|w| w[1] - w[0]).collect();
    if !cadence_diff.iter().all(|&d| d > 0) {
        return Err(GapFillError::NonMonotonicCadence);
    }
    let offsets: Vec<f64> = input
        .time
        .iter()
        .zip(cadence.iter())
        .map(|(&t, &c)| t - spacing * c as f64)
        .collect();
    let mut row_by_cadence: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for (index, &value) in cadence.iter().enumerate() {
        row_by_cadence.insert(value, index);
    }

    let first = cadence[0];
    let last = cadence[cadence.len() - 1];
    let count = (last - first + 1) as usize;
    let mut output = GapFillSeries {
        time: Vec::with_capacity(count),
        flux: Vec::with_capacity(count),
        cadence: Some(Vec::with_capacity(count)),
    };
    let mut lower = 0usize;
    for value in first..=last {
        output.cadence.as_mut().unwrap().push(value);
        if let Some(&row) = row_by_cadence.get(&value) {
            output.time.push(input.time[row]);
            output.flux.push(input.flux[row]);
            lower = row;
        } else {
            while lower + 1 < cadence.len() && cadence[lower + 1] < value {
                lower += 1;
            }
            let upper = lower + 1;
            let fraction = (value - cadence[lower]) as f64 / (cadence[upper] - cadence[lower]) as f64;
            let offset = offsets[lower] + fraction * (offsets[upper] - offsets[lower]);
            output.time.push(offset + spacing * value as f64);
            output.flux.push(noise.normal(mean, deviation));
        }
    }
    Ok(output)
}

fn fill_time_grid<R: GaussianNoiseSource>(
    input: GapFillSeries,
    spacing: f64,
    tolerance: f64,
    mean: f64,
    deviation: f64,
    noise: &mut R,
) -> GapFillSeries {
    let mut output = GapFillSeries {
        time: vec![input.time[0]],
        flux: vec![input.flux[0]],
        cadence: None,
    };
    for index in 1..input.time.len() {
        while input.time[index] - *output.time.last().unwrap() > tolerance * spacing {
            let next_time = output.time.last().unwrap() + spacing;
            output.time.push(next_time);
            output.flux.push(noise.normal(mean, deviation));
        }
        output.time.push(input.time[index]);
        output.flux.push(input.flux[index]);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    struct ReplayNoise {
        values: VecDeque<f64>,
    }

    impl GaussianNoiseSource for ReplayNoise {
        fn normal(&mut self, _mean: f64, _standard_deviation: f64) -> f64 {
            self.values.pop_front().expect("replay noise exhausted")
        }
    }

    /// Mirrors ExoNitContractTests.testGapFillSemantics (main.swift:1724),
    /// cadence-grid branch.
    #[test]
    fn cadence_aware_gap_fill_matches_swift_contract_test() {
        let mut noise = ReplayNoise {
            values: VecDeque::from([999.0]),
        };
        let input = GapFillSeries {
            time: vec![1.0, 2.0, 4.0, 5.0],
            flux: vec![100.0, 101.0, 103.0, 104.0],
            cadence: Some(vec![10, 11, 13, 14]),
        };
        let filled = fill_if_allowed(input, 5.0, 1.2, None, &mut noise).unwrap();
        assert_eq!(filled.time, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(filled.cadence, Some(vec![10, 11, 12, 13, 14]));
        assert_eq!(filled.flux, vec![100.0, 101.0, 999.0, 103.0, 104.0]);
    }

    /// Mirrors ExoNitContractTests.testGapFillSemantics, time-grid branch.
    #[test]
    fn time_grid_gap_fill_matches_swift_contract_test() {
        let mut noise = ReplayNoise {
            values: VecDeque::from([777.0]),
        };
        let input = GapFillSeries {
            time: vec![0.0, 1.0, 2.0, 4.0],
            flux: vec![1.0, 1.0, 1.0, 1.0],
            cadence: None,
        };
        let filled = fill_if_allowed(input, 5.0, 1.2, None, &mut noise).unwrap();
        assert_eq!(filled.time, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(filled.flux, vec![1.0, 1.0, 1.0, 777.0, 1.0]);
    }

    /// Mirrors ExoNitContractTests.testGapFillSemantics, gate branch: a
    /// global gap of exactly `maximumGlobalGapDays` must NOT trigger a fill
    /// (strict `<` comparison), and must consume no randomness.
    #[test]
    fn global_gap_of_exactly_five_days_disables_fill() {
        let mut noise = ReplayNoise {
            values: VecDeque::from([123.0]),
        };
        let input = GapFillSeries {
            time: vec![0.0, 1.0, 6.0],
            flux: vec![1.0, 1.0, 1.0],
            cadence: None,
        };
        let gated = fill_if_allowed(input.clone(), 5.0, 1.2, None, &mut noise).unwrap();
        assert_eq!(gated, input);
        assert_eq!(noise.values.len(), 1, "gate must consume no randomness");
    }
}
