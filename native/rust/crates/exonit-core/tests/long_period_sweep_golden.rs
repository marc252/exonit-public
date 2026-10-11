//! Golden parity test vs `fixtures/long_period_sweep_golden.json`, mirroring
//! `ExoNitContractTests.testLongPeriodSweepPythonGoldenParity`
//! (main.swift:982).

use std::cell::RefCell;

use exonit_core::bls::{reference, BlsParams, BlsResult};
use exonit_core::detrend;
use exonit_core::long_period_sweep::{self, LongPeriodSweepOptions};
use serde::Deserialize;

#[derive(Deserialize)]
struct Options {
    #[serde(rename = "baselineThresholdDays")]
    baseline_threshold_days: f64,
    #[serde(rename = "windowsDays")]
    windows_days: Vec<f64>,
    #[serde(rename = "periodMinimumDays")]
    period_minimum_days: f64,
    #[serde(rename = "legacyPeriodMaximumDays")]
    legacy_period_maximum_days: f64,
    #[serde(rename = "periodMaximumCapDays")]
    period_maximum_cap_days: f64,
    #[serde(rename = "minimumTransits")]
    minimum_transits: f64,
    #[serde(rename = "basePeriodCount")]
    base_period_count: usize,
    #[serde(rename = "periodCountScaleReferenceMaximumDays")]
    period_count_scale_reference_maximum_days: Option<f64>,
    #[serde(rename = "binCount")]
    bin_count: usize,
    #[serde(rename = "durationStepCount")]
    duration_step_count: usize,
    #[serde(rename = "durationMinimumFraction")]
    duration_minimum_fraction: f64,
    #[serde(rename = "durationMaximumFraction")]
    duration_maximum_fraction: f64,
    #[serde(rename = "minimumSNR")]
    minimum_snr: f64,
}

#[derive(Deserialize)]
struct Expected {
    #[serde(rename = "windowDays")]
    window_days: f64,
    period: f64,
    t0: f64,
    duration: f64,
    depth: f64,
    snr: f64,
    power: f64,
}

#[derive(Deserialize)]
struct Fixture {
    time: Vec<f64>,
    flux: Vec<Option<f64>>,
    #[serde(rename = "accumulatedMask")]
    accumulated_mask: Vec<bool>,
    options: Options,
    expected: Expected,
}

#[test]
fn long_period_sweep_matches_swift_golden() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/long_period_sweep_golden.json");
    let fixture: Fixture =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read long_period_sweep_golden.json"))
            .expect("parse long_period_sweep_golden.json");

    let options = LongPeriodSweepOptions {
        baseline_threshold_days: fixture.options.baseline_threshold_days,
        windows_days: fixture.options.windows_days.clone(),
        period_minimum_days: fixture.options.period_minimum_days,
        legacy_period_maximum_days: fixture.options.legacy_period_maximum_days,
        period_maximum_cap_days: fixture.options.period_maximum_cap_days,
        minimum_transits: fixture.options.minimum_transits,
        base_period_count: fixture.options.base_period_count,
        period_count_scale_reference_maximum_days: fixture.options.period_count_scale_reference_maximum_days,
        bin_count: fixture.options.bin_count,
        duration_step_count: fixture.options.duration_step_count,
        duration_minimum_fraction: fixture.options.duration_minimum_fraction,
        duration_maximum_fraction: fixture.options.duration_maximum_fraction,
        minimum_snr: fixture.options.minimum_snr,
    };
    let flux: Vec<f64> = fixture.flux.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    assert!(
        fixture.time.last().unwrap() - fixture.time.first().unwrap() > 100.0,
        "golden baseline must exceed 100 days"
    );

    // Default detrender: mirrors LongPeriodSweep.search's default argument
    // (BiweightDetrender.flatten with windowLength: window, Wotan defaults
    // otherwise).
    let detrender = |time: &[f64], flux: &[f64], window: f64, mask: Option<&[bool]>| {
        detrend::flatten_with_params(
            time,
            flux,
            window,
            detrend::DEFAULT_EDGE_CUTOFF,
            detrend::DEFAULT_BREAK_TOLERANCE,
            detrend::DEFAULT_CVAL,
            mask,
        )
    };

    // Mirrors the Swift test wiring `search` directly to `BLSReference.search`
    // (not the two-pass/accelerated path) so the dynamic period_maximum
    // clamp can be observed exactly.
    let observed_period_maximum: RefCell<Option<f64>> = RefCell::new(None);
    let search = |time: &[f64], flux: &[f64], parameters: &BlsParams| -> Result<BlsResult, reference::BlsError> {
        *observed_period_maximum.borrow_mut() = Some(parameters.period_maximum);
        reference::search(time, flux, parameters)
    };

    let outcome = long_period_sweep::search(&fixture.time, &flux, &fixture.accumulated_mask, &options, &detrender, &search)
        .expect("long-period sweep golden unexpectedly errored");
    let outcome = outcome.expect("long-period golden unexpectedly rejected");

    let expected_period_maximum =
        (fixture.time.last().unwrap() - fixture.time.first().unwrap()) / options.minimum_transits;
    assert!(
        (observed_period_maximum.borrow().unwrap_or(0.0) - expected_period_maximum).abs() < 1e-12,
        "dynamic period limit"
    );

    let expected = fixture.expected;
    assert!((outcome.window_days - expected.window_days).abs() < 1e-12, "window parity");
    assert!(
        (outcome.result.period - expected.period).abs() < 1e-10,
        "period parity {} != {}",
        outcome.result.period,
        expected.period
    );
    assert!((outcome.result.t0 - expected.t0).abs() < 1e-10, "t0 parity");
    assert!((outcome.result.duration - expected.duration).abs() < 1e-10, "duration parity");
    // Cross-language running-biweight accumulation differs by a few ulps;
    // the absolute bound is below 0.002% of this 0.28% transit depth
    // (mirrors main.swift:1069-1070).
    assert!(
        (outcome.result.depth - expected.depth).abs() < 3e-8,
        "depth parity {} != {}",
        outcome.result.depth,
        expected.depth
    );
    // SNR divides by a small MAD, amplifying the same accumulation error.
    assert!(
        (outcome.result.snr - expected.snr).abs() < 1e-3,
        "snr parity {} != {}",
        outcome.result.snr,
        expected.snr
    );
    // Power is depth squared, so it inherits the small cross-language drift.
    assert!(
        (outcome.result.power - expected.power).abs() < 2e-9,
        "power parity {} != {}",
        outcome.result.power,
        expected.power
    );
}
