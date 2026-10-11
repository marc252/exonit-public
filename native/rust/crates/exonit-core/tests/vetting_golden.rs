//! Golden parity test vs `fixtures/vetting_golden.json`, mirroring
//! `ExoNitContractTests.testVettingGoldenParity` (main.swift:675).

use exonit_core::dto::{CandidateResult, VettingResult};
use exonit_core::vetting;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    time: Vec<f64>,
    flux: Vec<f64>,
    candidates: Vec<CandidateResult>,
    expected: Vec<VettingResult>,
    odd_even_flux: Vec<f64>,
    odd_even_candidates: Vec<CandidateResult>,
    odd_even_expected: VettingResult,
}

/// Swift compares with `==` on `VettingResult` (exact equality on the
/// decoded doubles). The fixture values were produced by a Python reference
/// generator and round-tripped through JSON with `float_roundtrip` (see
/// Cargo.toml comment), so exact `f64` equality is the correct, not an
/// over-tight, comparison here.
#[test]
fn vetting_matches_swift_golden() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/vetting_golden.json");
    let fixture: Fixture =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read vetting_golden.json")).expect("parse vetting_golden.json");

    let result = vetting::vet_all(&fixture.candidates, &fixture.time, &fixture.flux).expect("vet_all main fixture");
    assert_eq!(result, fixture.expected, "main vetting fixture");

    let odd_even =
        vetting::vet_all(&fixture.odd_even_candidates, &fixture.time, &fixture.odd_even_flux).expect("vet_all odd/even fixture");
    assert_eq!(odd_even, vec![fixture.odd_even_expected], "odd/even vetting fixture");
}
