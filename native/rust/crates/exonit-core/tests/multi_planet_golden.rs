//! Golden parity test vs `ExoNitContractTests.testMultiPlanetFiveIterationGoldenParity`
//! (main.swift:1654) — end-to-end MultiPlanetSearcher parity for TIC
//! 10756829 using the real F2 BLS (not a fixture-injected quick result).
//!
//! The Swift test reads its two TGLC FITS files from
//! `~/exonit-swift-canary/fits_cache/...`; this repo already has
//! same-basename local copies in `tests/data/` (see the doc comment on
//! `fits_pending_golden.rs`), so this ports the full pipeline: FITS read ->
//! `preprocess::process` (real iterative-detrend quick BLS, F2's
//! `RealQuickBlsSearch`) -> `multi_planet::search` (real BLS two-pass +
//! harmonic resolution + long-period sweep, CPU backend).

use exonit_core::bls::accelerated::ComputeBackend;
use exonit_core::bls::quick::RealQuickBlsSearch;
use exonit_core::fits;
use exonit_core::harmonic_resolver::LightCurveSource;
use exonit_core::multi_planet::{self, MultiPlanetOptions};
use exonit_core::preprocess::{self, DetrendMode};
use std::path::PathBuf;

fn data_dir() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data"))
}

struct Expected {
    period: f64,
    t0: f64,
    duration: f64,
    depth: f64,
    snr: f64,
}

/// Mirrors `testMultiPlanetFiveIterationGoldenParity`'s baseline (candidates
/// 1-5 for TIC 10756829, `baseline.jsonl`, rounded the same way production
/// rounds: 4 decimals period/t0/duration, 1 decimal depth/SNR).
#[test]
fn multi_planet_five_iteration_matches_swift_golden() {
    let paths = [
        data_dir().join("hlsp_tglc_tess_ffi_gaiaid-2080664784495597312-s0014-cam1-ccd1_tess_v1_llc.fits"),
        data_dir().join("hlsp_tglc_tess_ffi_gaiaid-2080664784495597312-s0015-cam1-ccd2_tess_v1_llc.fits"),
    ];
    for path in &paths {
        assert!(path.exists(), "expected local TGLC fixture at {}", path.display());
    }
    let curves: Vec<_> = paths.iter().map(|p| fits::read_light_curve(p).expect("read TGLC FITS")).collect();

    let search = RealQuickBlsSearch::new(1);
    let processed = preprocess::process(&curves, DetrendMode::Iterative(&search)).expect("process TGLC pipeline");

    let options = MultiPlanetOptions::default();
    let result = multi_planet::search(
        &processed.time,
        &processed.raw_flux,
        &processed.flat_flux,
        None,
        None,
        LightCurveSource::Ffi,
        &options,
        4,
        ComputeBackend::Cpu,
    )
    .expect("multi-planet search");

    let expected: [Expected; 5] = [
        Expected { period: 1.9989, t0: 1684.2977, duration: 0.2399, depth: 8423.4, snr: 8.9 },
        Expected { period: 3.9856, t0: 1687.3645, duration: 0.1993, depth: 7084.3, snr: 9.1 },
        Expected { period: 0.9998, t0: 1684.4079, duration: 0.13, depth: 5925.9, snr: 14.5 },
        Expected { period: 0.9998, t0: 1684.4679, duration: 0.07, depth: 7013.5, snr: 14.8 },
        Expected { period: 0.9998, t0: 1684.5079, duration: 0.03, depth: 9045.1, snr: 9.7 },
    ];

    assert_eq!(
        result.candidates.len(),
        expected.len(),
        "multi-planet candidate count actual={} expected={}",
        result.candidates.len(),
        expected.len()
    );
    for (index, want) in expected.iter().enumerate() {
        let actual = &result.candidates[index];
        assert!(
            (actual.period_days - want.period).abs() < 1e-4,
            "candidate #{} period actual={} expected={}",
            index + 1,
            actual.period_days,
            want.period
        );
        assert!(
            (actual.t0 - want.t0).abs() < 1e-4,
            "candidate #{} t0 actual={} expected={}",
            index + 1,
            actual.t0,
            want.t0
        );
        assert!(
            (actual.duration_days - want.duration).abs() < 1e-4,
            "candidate #{} duration actual={} expected={}",
            index + 1,
            actual.duration_days,
            want.duration
        );
        assert!(
            (actual.depth_ppm - want.depth).abs() < 0.1,
            "candidate #{} depth actual={} expected={}",
            index + 1,
            actual.depth_ppm,
            want.depth
        );
        assert!(
            (actual.snr - want.snr).abs() < 0.1,
            "candidate #{} SNR actual={} expected={}",
            index + 1,
            actual.snr,
            want.snr
        );
    }
}
