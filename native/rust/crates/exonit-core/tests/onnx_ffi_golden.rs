//! Real-model inference parity, mirroring Swift's `testCNNCoreMLInferenceParity`
//! and `testExoMinerFFICoreMLParity` (main.swift:750, 775) but against the
//! ORIGINAL `.onnx` files the Python worker loads, not a CoreML conversion —
//! see `docs/DESIGN.md`'s F3 section.
//!
//! `#[ignore]`d because the models are large binaries deliberately NOT
//! committed to this repo (`worker/models/*.onnx[.data]`,
//! `worker/models/exominer_ffi_v6_ensemble.calib.json`; multiple MB, plus
//! `ort`'s own downloaded `libonnxruntime.a` cache). Run explicitly with:
//!
//! ```sh
//! EXONIT_MODELS_DIR=~/exonit/worker/models \
//!     cargo test --all -- --ignored onnx_ffi
//! ```
//!
//! Tolerances mirror the Swift CoreML gate exactly (`< 2e-5` on
//! logit/probability) even though the backend here is ONNX Runtime, not
//! CoreML — see `docs/PLAN.md`'s F3 gate. In practice, both classifiers
//! reproduced the fixture's stored logits to double-precision-printed
//! equality against Python's onnxruntime 1.27.0 during development (see
//! the F3 session note in `docs/PLAN.md`), so `2e-5` has enormous margin;
//! it is kept at the Swift gate's value rather than tightened, since the
//! gate's job is catching a REGRESSION (wrong model, wrong provider,
//! shape/order bug), not chasing float noise.

use exonit_core::exominer_features::{self, FfiStellarMetadata};
use exonit_core::ffi_views::CnnFfiViews;
use exonit_core::onnx_cnn::CnnFfiClassifier;
use exonit_core::onnx_exominer::{ExoMinerFfiClassifier, PiecewiseLinearCalibration};
use serde::Deserialize;
use std::path::PathBuf;

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
    #[serde(rename = "stellar_raw")]
    stellar_raw: StellarRaw,
    logit: f64,
    probability: f64,
    #[serde(rename = "exominer_logit")]
    exominer_logit: f64,
    #[serde(rename = "exominer_raw_probability")]
    exominer_raw_probability: f64,
    #[serde(rename = "exominer_probability")]
    exominer_probability: f64,
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

fn models_dir() -> Option<PathBuf> {
    std::env::var_os("EXONIT_MODELS_DIR").map(PathBuf::from)
}

#[test]
#[ignore = "needs real ONNX models — see module doc for EXONIT_MODELS_DIR"]
fn cnn_onnx_matches_fixture_logit() {
    let Some(dir) = models_dir() else {
        panic!("set EXONIT_MODELS_DIR to run this test (see module doc)");
    };
    let fixture = load_fixture();
    let classifier = CnnFfiClassifier::new(dir.join("cnn_ffi_v1.onnx")).expect("load cnn_ffi_v1.onnx");
    let score = classifier
        .classify(&CnnFfiViews { global: fixture.global_view.clone(), local: fixture.local_view.clone() })
        .expect("classify");

    let logit_error = (score.logit - fixture.logit).abs();
    let probability_error = (score.probability - fixture.probability).abs();
    assert!(logit_error < 2e-5, "CNN ONNX logit parity {logit_error}");
    assert!(probability_error < 2e-5, "CNN ONNX probability parity {probability_error}");
}

#[test]
#[ignore = "needs real ONNX models — see module doc for EXONIT_MODELS_DIR"]
fn exominer_onnx_matches_fixture_logit() {
    let Some(dir) = models_dir() else {
        panic!("set EXONIT_MODELS_DIR to run this test (see module doc)");
    };
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

    let model_path = dir.join("exominer_ffi_v6_ensemble.onnx");
    // Explicit calibration, matching the exact literals Swift's contract
    // test pins (`x: [0, 0.38930313157914914, 1], y: [0, 0.5, 1]`) —
    // identical to the values in the committed
    // `exominer_ffi_v6_ensemble.calib.json` sibling, verified against the
    // real file during development. Using `with_calibration` here (rather
    // than `new`'s auto-load) keeps this test independent of whether that
    // sibling file happens to be present at `EXONIT_MODELS_DIR`.
    let calibration = PiecewiseLinearCalibration { x: vec![0.0, 0.389_303_131_579_149_14, 1.0], y: vec![0.0, 0.5, 1.0] };
    let classifier = ExoMinerFfiClassifier::with_calibration(&model_path, Some(calibration)).expect("load exominer_ffi_v6_ensemble.onnx");
    let score = classifier.classify(&features).expect("classify");

    let logit_error = (score.logit - fixture.exominer_logit).abs();
    let raw_error = (score.raw_probability - fixture.exominer_raw_probability).abs();
    let probability_error = (score.probability - fixture.exominer_probability).abs();
    assert!(logit_error < 2e-5, "ExoMiner ONNX logit parity {logit_error}");
    assert!(raw_error < 2e-5, "ExoMiner ONNX raw probability parity {raw_error}");
    assert!(probability_error < 2e-5, "ExoMiner ONNX calibration parity {probability_error}");
}

/// Auto-load path: the models directory's real `<model>.calib.json`
/// sibling should reproduce the same calibrated probability as the
/// explicit-calibration test above, proving `ExoMinerFfiClassifier::new`'s
/// sibling-file loading (the Python-parity path production code actually
/// uses) is wired correctly, not just `with_calibration`.
#[test]
#[ignore = "needs real ONNX models — see module doc for EXONIT_MODELS_DIR"]
fn exominer_onnx_auto_calibration_matches_fixture() {
    let Some(dir) = models_dir() else {
        panic!("set EXONIT_MODELS_DIR to run this test (see module doc)");
    };
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

    let model_path = dir.join("exominer_ffi_v6_ensemble.onnx");
    let calib_path = dir.join("exominer_ffi_v6_ensemble.calib.json");
    if !calib_path.exists() {
        eprintln!("skipping: {calib_path:?} not present next to the model");
        return;
    }
    let classifier = ExoMinerFfiClassifier::new(&model_path).expect("load exominer_ffi_v6_ensemble.onnx");
    let score = classifier.classify(&features).expect("classify");

    let probability_error = (score.probability - fixture.exominer_probability).abs();
    assert!(probability_error < 2e-5, "ExoMiner ONNX auto-calibration parity {probability_error}");
}
