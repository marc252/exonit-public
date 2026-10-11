//! Transliterated from `Sources/ExoNitCore/HarmonicResolver.swift`.

use crate::bls::{BlsParams, BlsResult, BlsSearchFn};

/// Mirrors `LightCurveSource` (`spoc`/`ffi`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightCurveSource {
    Spoc,
    Ffi,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarmonicResolutionOptions {
    pub minimum_snr: f64,
    pub depth_ratio_lower: f64,
    pub depth_ratio_upper: f64,
    pub source: LightCurveSource,
    pub long_period_recovery: bool,
    pub long_period_minimum_days: f64,
    pub long_snr_factor: f64,
}

impl Default for HarmonicResolutionOptions {
    fn default() -> Self {
        HarmonicResolutionOptions {
            minimum_snr: 5.0,
            depth_ratio_lower: 0.5,
            depth_ratio_upper: 2.0,
            source: LightCurveSource::Spoc,
            long_period_recovery: false,
            long_period_minimum_days: 10.0,
            long_snr_factor: 0.5,
        }
    }
}

struct Probe {
    period: f64,
    snr_factor: f64,
    transit_guard: bool,
}

/// Mirrors `HarmonicResolver.resolve(time:flux:seed:periodMinimum:periodMaximum:options:search:)`.
pub fn resolve(
    time: &[f64],
    flux: &[f64],
    seed: &BlsResult,
    period_minimum: f64,
    period_maximum: f64,
    options: &HarmonicResolutionOptions,
    search: BlsSearchFn,
) -> BlsResult {
    let mut probes: Vec<Probe> = Vec::new();
    if options.source == LightCurveSource::Ffi {
        for multiplier in [2.0, 3.0, 4.0, 5.0] {
            let period = seed.period * multiplier;
            if period <= period_maximum {
                probes.push(Probe { period, snr_factor: 0.6, transit_guard: true });
            }
        }
    } else {
        let doubled = seed.period * 2.0;
        if doubled <= period_maximum {
            probes.push(Probe { period: doubled, snr_factor: 1.0, transit_guard: false });
        }
        if options.long_period_recovery {
            for multiplier in [3.0 / 2.0, 5.0 / 3.0, 4.0 / 3.0, 5.0 / 2.0] {
                let period = seed.period * multiplier;
                if period >= options.long_period_minimum_days && period <= period_maximum {
                    probes.push(Probe {
                        period,
                        snr_factor: options.long_snr_factor,
                        transit_guard: true,
                    });
                }
            }
        }
    }
    let demotion_factor = if options.source == LightCurveSource::Ffi { 0.6 } else { 1.0 };
    for divisor in [2.0, 3.0, 4.0, 5.0] {
        let period = seed.period / divisor;
        if period >= period_minimum {
            probes.push(Probe { period, snr_factor: demotion_factor, transit_guard: false });
        }
    }

    let span = time.last().copied().unwrap_or(0.0) - time.first().copied().unwrap_or(0.0);
    for probe in &probes {
        // worker.py's resolve_harmonic() calls bls_fn(..., period_min=cand*0.95,
        // period_max=cand*1.05, ...) WITHOUT clamping to the caller's global
        // period_min/period_max — the refine window is allowed to extend
        // outside [period_minimum, period_maximum] (e.g. a subharmonic probe
        // near period_minimum legitimately searches slightly below it). A
        // clamp here narrows the fine-BLS grid vs Python, landing on a
        // different local peak/SNR near marginal-significance systematics
        // (found via TIC 10757037 in the Swift port: identical seed BLS +
        // identical re-detrended flux, but the clamped probe window resolved
        // to a different period/SNR than Python's unclamped window). Do NOT
        // clamp `parameters.period_minimum`/`period_maximum` to
        // `period_minimum`/`period_maximum` here.
        let parameters = BlsParams {
            period_minimum: probe.period * 0.95,
            period_maximum: probe.period * 1.05,
            period_count: 500,
            ..BlsParams::default()
        };
        if !(parameters.period_maximum > parameters.period_minimum) {
            continue;
        }
        let candidate = match search(time, flux, &parameters) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if !(candidate.snr >= options.minimum_snr) {
            continue;
        }
        if !(candidate.snr >= seed.snr * probe.snr_factor) {
            continue;
        }
        if seed.depth > 0.0 {
            let ratio = candidate.depth / seed.depth;
            if !(ratio > options.depth_ratio_lower && ratio < options.depth_ratio_upper) {
                continue;
            }
        }
        if probe.transit_guard && span > 0.0 && candidate.period > 0.0 && span / candidate.period < 2.0 {
            continue;
        }
        return candidate;
    }
    seed.clone()
}
