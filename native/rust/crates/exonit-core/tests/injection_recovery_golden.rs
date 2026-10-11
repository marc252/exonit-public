//! Golden parity test vs `fixtures/injection_recovery_golden.json`,
//! mirroring `ExoNitContractTests.testInjectionRecoveryGoldenParity`
//! (main.swift:864). Draw-for-draw exact on recovered counts/grids
//! (`NumpyMt19937` reproduces NumPy's `RandomState` bit-for-bit), `< 1e-15`
//! tolerance on the completeness ratios — the same tolerance the Swift
//! contract test uses.

use exonit_core::injection_recovery::{self, InjectionRecoveryOptions};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    time: Vec<f64>,
    flux: Vec<f64>,
    stellar_radius: f64,
    radius_grid: Vec<f64>,
    period_grid: Vec<f64>,
    injections_per_cell: usize,
    snr_threshold: f64,
    bls_n_periods: usize,
    bls_n_bins: usize,
    bls_n_dur_steps: usize,
    expected: Expected,
}

#[derive(Deserialize)]
struct Expected {
    radius_grid: Vec<f64>,
    period_grid: Vec<f64>,
    recovered_counts: Vec<Vec<i64>>,
    completeness_map: Vec<Vec<f64>>,
    truncated: bool,
    periods_evaluated: i64,
}

#[test]
fn injection_recovery_matches_swift_golden() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/injection_recovery_golden.json");
    let fixture: Fixture =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read injection_recovery_golden.json"))
            .expect("parse injection_recovery_golden.json");

    let options = InjectionRecoveryOptions {
        stellar_radius_solar: fixture.stellar_radius,
        stellar_mass_solar: None,
        radius_grid_earth: fixture.radius_grid.clone(),
        period_grid_days: Some(fixture.period_grid.clone()),
        injections_per_cell: fixture.injections_per_cell,
        minimum_snr: fixture.snr_threshold,
        period_count: fixture.bls_n_periods,
        bin_count: fixture.bls_n_bins,
        duration_step_count: fixture.bls_n_dur_steps,
        maximum_seconds: None,
    };

    let result = injection_recovery::run(&fixture.time, &fixture.flux, &options, 1, || false)
        .expect("injection_recovery::run");

    assert_eq!(result.radius_grid, Some(fixture.expected.radius_grid.clone()), "injection radius grid");
    assert_eq!(result.period_grid, Some(fixture.expected.period_grid.clone()), "injection period grid");
    assert_eq!(result.truncated, Some(fixture.expected.truncated), "injection truncation");
    assert_eq!(result.periods_evaluated, Some(fixture.expected.periods_evaluated), "injection periods evaluated");

    let actual_completeness = result.completeness_map.as_ref().expect("completeness_map present");

    for radius_index in 0..fixture.radius_grid.len() {
        for period_index in 0..fixture.period_grid.len() {
            let index = radius_index * fixture.period_grid.len() + period_index;
            let recovered_count = result.details[index]
                .recovered_count
                .expect("recovered_count present");
            assert_eq!(
                recovered_count, fixture.expected.recovered_counts[radius_index][period_index],
                "injection recovered count[{radius_index},{period_index}]"
            );
            let diff = (actual_completeness[radius_index][period_index]
                - fixture.expected.completeness_map[radius_index][period_index])
                .abs();
            assert!(
                diff < 1e-15,
                "injection completeness[{radius_index},{period_index}]: diff={diff}"
            );
        }
    }
}
