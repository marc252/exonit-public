//! Golden parity test vs `fixtures/cnn_ffi_golden.json`, mirroring
//! `ExoNitContractTests.testCNNFFIViewGoldenParity` and the feature-parity
//! half of `testExoMinerFFICoreMLParity` (main.swift:708, 775) — the pure
//! math (view generation + feature engineering), no ONNX model needed.
//! Model-backed inference parity lives in `onnx_ffi_golden.rs` (`#[ignore]`,
//! needs `EXONIT_MODELS_DIR`).

use exonit_core::exominer_features::{self, FfiStellarMetadata};
use exonit_core::ffi_views;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    period: f64,
    t0: f64,
    duration: f64,
    time: Vec<f64>,
    flux: Vec<f64>,
    #[serde(rename = "global_view")]
    global_view: Vec<f32>,
    #[serde(rename = "local_view")]
    local_view: Vec<f32>,
    #[serde(rename = "aux_views")]
    aux_views: Vec<Vec<f32>>,
    #[serde(rename = "stellar_raw")]
    stellar_raw: StellarRaw,
    stellar: Vec<f32>,
    tce: Vec<f32>,
    logit: f64,
    probability: f64,
}

#[derive(Deserialize)]
struct StellarRaw {
    tmag: f64,
    teff: f64,
    logg: f64,
    rstar: f64,
}

fn load_fixture() -> Fixture {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/cnn_ffi_golden.json");
    let text = std::fs::read_to_string(path).expect("read cnn_ffi_golden.json");
    serde_json::from_str(&text).expect("parse cnn_ffi_golden.json")
}

fn max_abs_error(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (*x as f64 - *y as f64).abs()).fold(0.0, f64::max)
}

/// Tolerances mirror Swift's `testCNNFFIViewGoldenParity`: view arrays
/// `< 2e-6`, sigmoid identity `< 1e-15` (pure math, no model rounding).
#[test]
fn cnn_views_match_swift_golden() {
    let fixture = load_fixture();
    let views = ffi_views::make_cnn_views(&fixture.time, &fixture.flux, fixture.period, fixture.t0, fixture.duration)
        .expect("make_cnn_views");
    assert_eq!(views.global.len(), ffi_views::GLOBAL_BIN_COUNT, "global view shape");
    assert_eq!(views.local.len(), ffi_views::LOCAL_BIN_COUNT, "local view shape");

    let global_error = max_abs_error(&views.global, &fixture.global_view);
    let local_error = max_abs_error(&views.local, &fixture.local_view);
    assert!(global_error < 2e-6, "CNN global view parity {global_error}");
    assert!(local_error < 2e-6, "CNN local view parity {local_error}");

    let sigmoid_error = (ffi_views::sigmoid(fixture.logit) - fixture.probability).abs();
    assert!(sigmoid_error < 1e-15, "CNN sigmoid parity {sigmoid_error}");
}

/// Tolerances mirror Swift's `testExoMinerFFICoreMLParity`'s feature-parity
/// assertions: aux/stellar/tce arrays `< 2e-6`. `depth_ppm`/`snr` are the
/// same hardcoded probe values (3000, 18) the Swift test uses.
#[test]
fn exominer_features_match_swift_golden() {
    let fixture = load_fixture();
    let stellar = FfiStellarMetadata {
        tmag: Some(fixture.stellar_raw.tmag),
        teff: Some(fixture.stellar_raw.teff),
        logg: Some(fixture.stellar_raw.logg),
        radius: Some(fixture.stellar_raw.rstar),
        mass: None,
    };
    let features = exominer_features::make(&fixture.time, &fixture.flux, fixture.period, fixture.t0, fixture.duration, 3000.0, 18.0, stellar)
        .expect("exominer_features::make");

    let fixture_aux_flat: Vec<f32> = fixture.aux_views.iter().flatten().copied().collect();
    let feature_aux_flat: Vec<f32> = features.auxiliary.iter().flatten().copied().collect();
    let aux_error = max_abs_error(&feature_aux_flat, &fixture_aux_flat);
    let stellar_error = max_abs_error(&features.stellar, &fixture.stellar);
    let tce_error = max_abs_error(&features.tce, &fixture.tce);

    assert!(aux_error < 2e-6, "ExoMiner aux view parity {aux_error}");
    assert!(stellar_error < 2e-6, "ExoMiner stellar feature parity {stellar_error}");
    assert!(tce_error < 2e-6, "ExoMiner tce feature parity {tce_error}");
}
