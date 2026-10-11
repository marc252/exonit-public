//! Golden parity tests vs `fixtures/bls_golden.json`, mirroring
//! `ExoNitContractTests.testBLSGoldenParity` and `.testBLSTwoPassParity`
//! (Sources/ExoNitContractTests/main.swift:381, :654).
//!
//! `fixtures/bls_golden.json` did not exist anywhere in the Swift repo (the
//! Swift test hardcodes the same numbers inline instead of loading a JSON
//! fixture) — regenerated here, read-only, via:
//!
//! ```text
//! cd ~/exonit-swift-worker-dev && \
//!   ~/exonit/.venv/bin/python3 scripts/generate_bls_golden.py \
//!   > ~/exonit-rust-worker-f2/fixtures/bls_golden.json
//! ```
//!
//! `generate_bls_golden.py` imports `reference-python/bls_fast.py` (the
//! Python reference `BLSReference`/`BLSCPUOptimized` were themselves ported
//! from and validated against) and runs it against the exact same
//! deterministic transit-model light curve as `testBLSGoldenParity`. Its
//! output matched the Swift-hardcoded values bit-for-bit when regenerated
//! (both ultimately trace to the same `bls_fast.bls_fast` reference), so the
//! fixture and the Swift contract test's inline literals are the same
//! ground truth expressed two ways.

use exonit_core::bls::{accelerated, optimized, reference, two_pass};
use serde::Deserialize;

#[derive(Deserialize)]
struct BlsGolden {
    period: f64,
    power: f64,
    duration: f64,
    t0: f64,
    depth: f64,
    #[allow(dead_code)]
    depth_ppm: f64,
    snr: f64,
}

fn model_light_curve() -> (Vec<f64>, Vec<f64>) {
    let period = 10.0 / 3.0;
    let time: Vec<f64> = (0..360).map(|i| i as f64 / 12.0).collect();
    let flux: Vec<f64> = time
        .iter()
        .map(|&value| {
            let phase = (value / period) % 1.0;
            if (0.20..0.28).contains(&phase) {
                0.988
            } else {
                1.0
            }
        })
        .collect();
    (time, flux)
}

/// Mirrors `testBLSGoldenParity`'s `BLSReference.search` half, against the
/// fixture (see module doc comment) rather than the Swift-inline literals —
/// same numbers, and the tolerances mirror main.swift:397-402 exactly.
#[test]
fn reference_search_matches_swift_golden() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/bls_golden.json");
    let golden: BlsGolden =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read bls_golden.json")).expect("parse bls_golden.json");

    let (time, flux) = model_light_curve();
    let params = reference::BlsParams {
        period_minimum: 2.0,
        period_maximum: 5.0,
        period_count: 121,
        bin_count: 50,
        duration_step_count: 7,
        duration_minimum_fraction: 0.04,
        duration_maximum_fraction: 0.12,
    };
    let result = reference::search(&time, &flux, &params).expect("reference BLS search");

    assert!((result.period - golden.period).abs() < 1e-14, "period parity");
    assert!((result.power - golden.power).abs() < 1e-14, "power parity");
    assert!((result.duration - golden.duration).abs() < 1e-14, "duration parity");
    assert!((result.t0 - golden.t0).abs() < 1e-14, "t0 parity");
    assert!((result.depth - golden.depth).abs() < 1e-14, "depth parity");
    assert!((result.snr - golden.snr).abs() < 1e-10, "SDE parity");
}

/// Mirrors `testBLSGoldenParity`'s second half: `BLSCPUOptimized.search ==
/// BLSReference.search` bit-for-bit, even with `maxConcurrency: 4` in Swift
/// (see `bls::optimized`'s module doc comment for why F2's single-threaded
/// port still satisfies this).
#[test]
fn optimized_search_is_bit_identical_to_reference() {
    let (time, flux) = model_light_curve();
    let params = reference::BlsParams {
        period_minimum: 2.0,
        period_maximum: 5.0,
        period_count: 121,
        bin_count: 50,
        duration_step_count: 7,
        duration_minimum_fraction: 0.04,
        duration_maximum_fraction: 0.12,
    };
    let reference_result = reference::search(&time, &flux, &params).expect("reference BLS search");
    let optimized_result = optimized::search(&time, &flux, &params, 4).expect("optimized BLS search");
    assert_eq!(optimized_result, reference_result, "parallel CPU path must be bit-identical");
}

/// Mirrors `testBLSTwoPassParity` (main.swift:654) — hardcoded in the Swift
/// suite the same way (no JSON fixture there either).
#[test]
fn two_pass_search_matches_swift_golden() {
    let (time, flux) = model_light_curve();
    let params = reference::BlsParams {
        period_count: 10_000,
        bin_count: 120,
        duration_step_count: 10,
        ..reference::BlsParams::default()
    };
    let result = two_pass::search(&time, &flux, &params, 2).expect("two-pass BLS search");

    assert!((result.period - 3.326623878947138).abs() < 1e-14, "two-pass period");
    assert!((result.power - 0.0035533297297300143).abs() < 1e-14, "two-pass power");
    assert!((result.duration - 0.27721865657892814).abs() < 1e-14, "two-pass duration");
    assert!((result.t0 - 0.7207685071052132).abs() < 1e-14, "two-pass t0");
    assert!((result.depth - 0.011927927927928406).abs() < 1e-14, "two-pass depth");
    assert!((result.snr - 49.34005822782964).abs() < 1e-10, "two-pass coarse SNR");
    assert_eq!(result.periods.len(), 5_000, "two-pass reports coarse spectrum");
}

/// `accelerated::search(.., ComputeBackend::Cpu, ..)` must be exactly
/// `two_pass::search` (the only backend implemented in F2 — see
/// `bls::accelerated`'s module doc comment).
#[test]
fn accelerated_cpu_backend_matches_two_pass() {
    let (time, flux) = model_light_curve();
    let params = reference::BlsParams {
        period_count: 10_000,
        bin_count: 120,
        duration_step_count: 10,
        ..reference::BlsParams::default()
    };
    let direct = two_pass::search(&time, &flux, &params, 2).expect("two-pass BLS search");
    let via_backend =
        accelerated::search(&time, &flux, &params, accelerated::ComputeBackend::Cpu, 2).expect("accelerated CPU search");
    assert_eq!(direct, via_backend, "accelerated Cpu backend must match two_pass::search exactly");
}
