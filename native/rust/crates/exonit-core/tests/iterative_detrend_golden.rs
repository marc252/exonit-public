//! Golden parity test vs `fixtures/iterative_detrend_golden.json`, generated
//! by `generate_iterative_detrend_golden.py` in the Swift repo.
//!
//! Real BLS (the actual "quick" coarse search) is F2 scope — see the design
//! note in `detrend_iterative.rs`. This test does NOT validate BLS; it
//! injects the fixture's own `quick` field verbatim (that field IS a
//! `QuickBlsResult`) via a fixture-backed `QuickBlsSearch`, and then
//! genuinely validates two things against the golden:
//!   1. the first-pass biweight flatten (`golden.first`) — same tolerance as
//!      the wotan golden, since it's the same `detrend::flatten` call;
//!   2. the transit-mask computation + refined flatten built on top of the
//!      injected quick result (`golden.mask`, `golden.final`) — this is the
//!      actual logic IterativeDetrender.swift adds beyond BiweightDetrender.

use exonit_core::detrend;
use exonit_core::detrend_iterative::{self, QuickBlsResult, QuickBlsSearch};
use serde::Deserialize;

#[derive(Deserialize)]
struct Quick {
    period: f64,
    power: f64,
    duration: f64,
    t0: f64,
    depth: f64,
    snr: f64,
}

#[derive(Deserialize)]
struct Golden {
    time: Vec<Option<f64>>,
    flux: Vec<Option<f64>>,
    first: Vec<Option<f64>>,
    #[serde(rename = "final")]
    flux_final: Vec<Option<f64>>,
    quick: Quick,
    mask: Vec<bool>,
}

struct FixtureQuickSearch(QuickBlsResult);

impl QuickBlsSearch for FixtureQuickSearch {
    fn search(&self, _time: &[f64], _flux: &[f64]) -> QuickBlsResult {
        self.0
    }
}

fn to_f64(v: &Option<f64>) -> f64 {
    v.unwrap_or(f64::NAN)
}

#[test]
fn iterative_detrend_matches_swift_golden_masking_and_refine() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/iterative_detrend_golden.json"
    );
    let raw = std::fs::read_to_string(path).expect("read iterative_detrend_golden.json");
    let golden: Golden = serde_json::from_str(&raw).expect("parse iterative_detrend_golden.json");

    let time: Vec<f64> = golden.time.iter().map(to_f64).collect();
    let flux: Vec<f64> = golden.flux.iter().map(to_f64).collect();

    // (1) first-pass biweight flatten, same tolerance as
    // ExoNitContractTests.testWotanBiweightGoldenParity (5e-14).
    let first = detrend::flatten(&time, &flux, None).expect("first-pass flatten");
    for i in 0..first.len() {
        let expected = to_f64(&golden.first[i]);
        if expected.is_nan() {
            assert!(first[i].is_nan(), "first[{i}] must be NaN, got {}", first[i]);
        } else {
            assert!(
                (first[i] - expected).abs() < 5e-14,
                "first[{i}] actual={} expected={}",
                first[i],
                expected
            );
        }
    }

    let quick = QuickBlsResult {
        period: golden.quick.period,
        power: golden.quick.power,
        duration: golden.quick.duration,
        t0: golden.quick.t0,
        depth: golden.quick.depth,
        snr: golden.quick.snr,
    };
    let search = FixtureQuickSearch(quick);
    let result = detrend_iterative::flatten(&time, &flux, &search, detrend_iterative::DEFAULT_SNR_THRESHOLD)
        .expect("iterative flatten");

    // Plumbing check only (the value is literally echoed from the fixture,
    // since BLS itself is out of scope) — tolerances mirror
    // ExoNitContractTests.testIterativeDetrendGoldenParity (main.swift:1522-1523).
    assert!((result.quick_bls.period - golden.quick.period).abs() < 1e-14, "quick period plumbing");
    assert!((result.quick_bls.snr - golden.quick.snr).abs() < 1e-10, "quick SNR plumbing");

    // (2) the actual logic under test: transit mask computed from the
    // (injected) quick result, and the refined biweight flatten built on it.
    assert_eq!(
        result.transit_mask, golden.mask,
        "iterative transit mask (ExoNitContractTests main.swift:1524)"
    );
    for i in 0..result.flux.len() {
        let expected = to_f64(&golden.flux_final[i]);
        if expected.is_nan() {
            assert!(
                result.flux[i].is_nan(),
                "final[{i}] must be NaN, got {}",
                result.flux[i]
            );
        } else {
            // Tolerance mirrors ExoNitContractTests main.swift:1528: < 2e-12.
            assert!(
                (result.flux[i] - expected).abs() < 2e-12,
                "final[{i}] actual={} expected={}",
                result.flux[i],
                expected
            );
        }
    }
}
