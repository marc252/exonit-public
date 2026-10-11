//! Transliterated from `Sources/ExoNitCore/CandidateVetter.swift`.

use std::fmt;

use crate::dto::{CandidateResult, VettingResult};
use crate::statistics::median;

const HARMONIC_RATIOS: [f64; 6] = [0.5, 1.0 / 3.0, 2.0, 3.0, 2.0 / 3.0, 3.0 / 2.0];
const HARMONIC_TOLERANCE: f64 = 0.02;

#[derive(Debug, Clone, PartialEq)]
pub enum VettingError {
    MismatchedSeries,
    EmptySeries,
    InvalidCandidate(i64),
}

impl fmt::Display for VettingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VettingError::MismatchedSeries => write!(f, "time and flux lengths differ"),
            VettingError::EmptySeries => write!(f, "empty light curve series"),
            VettingError::InvalidCandidate(n) => write!(f, "invalid candidate #{n}"),
        }
    }
}

impl std::error::Error for VettingError {}

/// Mirrors `CandidateVetter.vetAll(candidates:time:flux:)`.
pub fn vet_all(candidates: &[CandidateResult], time: &[f64], flux: &[f64]) -> Result<Vec<VettingResult>, VettingError> {
    if time.len() != flux.len() {
        return Err(VettingError::MismatchedSeries);
    }
    if time.is_empty() {
        return Err(VettingError::EmptySeries);
    }
    candidates
        .iter()
        .map(|candidate| {
            if !(candidate.period_days > 0.0) || !(candidate.duration_days > 0.0) {
                return Err(VettingError::InvalidCandidate(candidate.candidate_number));
            }
            let harmonic = check_harmonics(candidate, candidates);
            let odd_even = check_odd_even(candidate, time, flux);
            let v_shape = check_v_shape(candidate, time, flux);
            Ok(VettingResult {
                is_harmonic: Some(harmonic.is_harmonic),
                harmonic_of: harmonic.parent,
                harmonic_ratio: harmonic.ratio,
                odd_even_consistent: Some(odd_even.consistent),
                odd_depth: odd_even.odd_depth,
                even_depth: odd_even.even_depth,
                is_v_shaped: Some(v_shape.is_v_shaped),
                v_metric: Some(v_shape.metric),
                passed: Some(!harmonic.is_harmonic && odd_even.consistent && !v_shape.is_v_shaped),
            })
        })
        .collect()
}

struct HarmonicCheck {
    is_harmonic: bool,
    parent: Option<i64>,
    ratio: Option<f64>,
}

fn check_harmonics(candidate: &CandidateResult, all: &[CandidateResult]) -> HarmonicCheck {
    let mut parents: Vec<&CandidateResult> = all
        .iter()
        .filter(|c| c.candidate_number != candidate.candidate_number && c.snr >= candidate.snr)
        .collect();
    parents.sort_by(|a, b| {
        if a.snr != b.snr {
            b.snr.partial_cmp(&a.snr).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            a.candidate_number.cmp(&b.candidate_number)
        }
    });
    for parent in &parents {
        if parent.snr == candidate.snr && parent.candidate_number > candidate.candidate_number {
            continue;
        }
        for &ratio in &HARMONIC_RATIOS {
            if (candidate.period_days - parent.period_days * ratio).abs() / candidate.period_days < HARMONIC_TOLERANCE
            {
                return HarmonicCheck {
                    is_harmonic: true,
                    parent: Some(parent.candidate_number),
                    ratio: Some(ratio),
                };
            }
        }
    }
    HarmonicCheck { is_harmonic: false, parent: None, ratio: None }
}

struct OddEvenCheck {
    consistent: bool,
    odd_depth: Option<f64>,
    even_depth: Option<f64>,
}

fn check_odd_even(candidate: &CandidateResult, time: &[f64], flux: &[f64]) -> OddEvenCheck {
    let count = (((time[time.len() - 1] - time[0]) / candidate.period_days) as i64) + 1;
    let half_duration = candidate.duration_days / 2.0;
    let mut odd: Vec<f64> = Vec::new();
    let mut even: Vec<f64> = Vec::new();
    if count > 0 {
        for index in 0..count {
            let epoch = candidate.t0 + index as f64 * candidate.period_days;
            let transit: Vec<f64> = time
                .iter()
                .zip(flux.iter())
                .filter_map(|(&st, &sf)| if (st - epoch).abs() < half_duration { Some(sf) } else { None })
                .collect();
            if transit.len() < 3 {
                continue;
            }
            let depth = 1.0 - median(&transit);
            if index % 2 == 0 {
                even.push(depth);
            } else {
                odd.push(depth);
            }
        }
    }
    if odd.len() < 2 || even.len() < 2 {
        return OddEvenCheck { consistent: true, odd_depth: None, even_depth: None };
    }
    let odd_depth = median(&odd);
    let even_depth = median(&even);
    let maximum = odd_depth.max(even_depth);
    let minimum = odd_depth.min(even_depth);
    let consistent = (minimum > 0.0 && maximum / minimum < 3.0) || maximum < 1e-5;
    OddEvenCheck {
        consistent,
        odd_depth: Some(rounded(odd_depth * 1e6, 1)),
        even_depth: Some(rounded(even_depth * 1e6, 1)),
    }
}

struct VShapeCheck {
    is_v_shaped: bool,
    metric: f64,
}

fn check_v_shape(candidate: &CandidateResult, time: &[f64], flux: &[f64]) -> VShapeCheck {
    let window = candidate.duration_days / candidate.period_days * 2.0;
    let samples: Vec<(f64, f64)> = time
        .iter()
        .zip(flux.iter())
        .filter_map(|(&st, &sf)| {
            let mut remainder = (st - candidate.t0) % candidate.period_days;
            if remainder < 0.0 {
                remainder += candidate.period_days;
            }
            let mut phase = remainder / candidate.period_days;
            if phase > 0.5 {
                phase -= 1.0;
            }
            if phase.abs() < window {
                Some((phase, sf))
            } else {
                None
            }
        })
        .collect();
    if samples.len() < 10 {
        return VShapeCheck { is_v_shaped: false, metric: 0.0 };
    }
    let maximum_phase = samples.iter().map(|s| s.0.abs()).fold(0.0_f64, f64::max);
    if maximum_phase == 0.0 {
        return VShapeCheck { is_v_shaped: false, metric: 0.0 };
    }
    let center: Vec<f64> = samples.iter().filter(|s| s.0.abs() < maximum_phase * 0.3).map(|s| s.1).collect();
    let edge: Vec<f64> = samples.iter().filter(|s| s.0.abs() > maximum_phase * 0.6).map(|s| s.1).collect();
    if center.len() < 3 || edge.len() < 3 {
        return VShapeCheck { is_v_shaped: false, metric: 0.0 };
    }
    let center_depth = 1.0 - median(&center);
    if !(center_depth > 0.0) {
        return VShapeCheck { is_v_shaped: false, metric: 0.0 };
    }
    let metric = (1.0 - median(&edge)) / center_depth;
    VShapeCheck { is_v_shaped: metric > 0.7, metric: rounded(metric, 3) }
}

fn rounded(value: f64, places: i32) -> f64 {
    let scale = 10f64.powi(places);
    (value * scale).round_ties_even() / scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mismatched_series_is_rejected() {
        assert_eq!(vet_all(&[], &[1.0, 2.0], &[1.0]).unwrap_err(), VettingError::MismatchedSeries);
    }

    #[test]
    fn empty_series_is_rejected() {
        assert_eq!(vet_all(&[], &[], &[]).unwrap_err(), VettingError::EmptySeries);
    }

    #[test]
    fn invalid_candidate_period_is_rejected() {
        let candidate = CandidateResult::new(1, 0.0, 0.0, 1.0, 100.0, 10.0);
        assert_eq!(
            vet_all(&[candidate], &[1.0, 2.0], &[1.0, 1.0]).unwrap_err(),
            VettingError::InvalidCandidate(1)
        );
    }
}
