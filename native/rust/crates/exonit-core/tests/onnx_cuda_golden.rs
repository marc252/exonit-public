//! F4b golden test: ONNX Runtime CUDA execution provider vs CPU EP, for both
//! the CNN FFI and ExoMiner FFI classifiers, against the SAME `.onnx` models
//! the Python worker loads (`worker/models/*.onnx`) — see `onnx_cnn.rs`'s
//! `build_session` doc comment for the CUDA-preferred/CPU-fallback
//! selection this test's `CnnFfiClassifier::new`/`ExoMinerFfiClassifier::new`
//! calls exercise, vs the explicit `new_cpu_only` constructors added
//! alongside it for exactly this comparison.
//!
//! `#[ignore]`d for the SAME reason as `onnx_ffi_golden.rs` (F3): the models
//! are large binaries not committed to this repo. Run explicitly with:
//!
//! ```sh
//! EXONIT_MODELS_DIR=~/exonit/worker/models \
//!     cargo test --features cuda --test onnx_cuda_golden -- --ignored --nocapture
//! ```
//!
//! This test does NOT need `--ignored`'s usual "needs NVIDIA GPU" framing
//! quite as strictly as the BLS/detrend CUDA tests: `CnnFfiClassifier::new`
//! degrades to CPU with a logged fallback if CUDA isn't usable (see
//! `build_session`'s doc comment), so this test PASSES either way — on a
//! CUDA-capable host it proves CUDA-EP-vs-CPU-EP score parity; on any other
//! host it degenerates into CPU-vs-CPU (still a valid, if less interesting,
//! regression check). It is still gated behind `--ignored` because it needs
//! `EXONIT_MODELS_DIR` (large binaries), same as `onnx_ffi_golden.rs`.
//!
//! # Tolerance
//!
//! `2e-5` on logit/probability — copied verbatim from `onnx_ffi_golden.rs`'s
//! own gate (itself the Swift CoreML gate, `docs/PLAN.md`'s F3 section),
//! since this is the SAME kind of comparison (two execution backends against
//! the same weights) just with CUDA EP instead of CoreML/CPU-ORT on the
//! "other side" — reusing the established bar rather than inventing a new
//! one for a structurally identical check.

#![cfg(feature = "cuda")]

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
fn cnn_onnx_cuda_ep_matches_cpu_ep() {
    let Some(dir) = models_dir() else {
        panic!("set EXONIT_MODELS_DIR to run this test (see module doc)");
    };
    let fixture = load_fixture();
    let model_path = dir.join("cnn_ffi_v1.onnx");

    let cuda_preferred = CnnFfiClassifier::new(&model_path).expect("load cnn_ffi_v1.onnx (CUDA-preferred)");
    let cpu_only = CnnFfiClassifier::new_cpu_only(&model_path).expect("load cnn_ffi_v1.onnx (CPU-only)");

    let views = CnnFfiViews { global: fixture.global_view.clone(), local: fixture.local_view.clone() };
    let cuda_score = cuda_preferred.classify(&views).expect("classify (CUDA-preferred session)");
    let cpu_score = cpu_only.classify(&views).expect("classify (CPU-only session)");

    let logit_error = (cuda_score.logit - cpu_score.logit).abs();
    let probability_error = (cuda_score.probability - cpu_score.probability).abs();
    eprintln!(
        "cnn_onnx_cuda_ep_matches_cpu_ep: cuda_preferred(logit={}, prob={}) cpu_only(logit={}, prob={}) \
         logit_error={logit_error:.3e} probability_error={probability_error:.3e}",
        cuda_score.logit, cuda_score.probability, cpu_score.logit, cpu_score.probability
    );
    assert!(logit_error < 2e-5, "CNN ONNX CUDA-EP-vs-CPU-EP logit parity {logit_error}");
    assert!(probability_error < 2e-5, "CNN ONNX CUDA-EP-vs-CPU-EP probability parity {probability_error}");
}

#[test]
#[ignore = "needs real ONNX models — see module doc for EXONIT_MODELS_DIR"]
fn exominer_onnx_cuda_ep_matches_cpu_ep() {
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
    // Same fixed calibration `onnx_ffi_golden.rs::exominer_onnx_matches_fixture_logit`
    // pins, for the same reason (independence from whatever `.calib.json`
    // sibling happens to be present).
    let calibration = PiecewiseLinearCalibration { x: vec![0.0, 0.389_303_131_579_149_14, 1.0], y: vec![0.0, 0.5, 1.0] };
    let cuda_preferred =
        ExoMinerFfiClassifier::with_calibration(&model_path, Some(calibration.clone())).expect("load (CUDA-preferred)");
    // Same pinned calibration on both sides, so this test compares
    // EP-vs-EP score parity, not calibration-source parity (already
    // covered by `onnx_ffi_golden.rs::exominer_onnx_auto_calibration_matches_fixture`).
    let cpu_only = ExoMinerFfiClassifier::with_calibration_cpu_only(&model_path, Some(calibration)).expect("load (CPU-only)");

    let cuda_score = cuda_preferred.classify(&features).expect("classify (CUDA-preferred session)");
    let cpu_score = cpu_only.classify(&features).expect("classify (CPU-only session)");

    let logit_error = (cuda_score.logit - cpu_score.logit).abs();
    let raw_error = (cuda_score.raw_probability - cpu_score.raw_probability).abs();
    let probability_error = (cuda_score.probability - cpu_score.probability).abs();
    eprintln!(
        "exominer_onnx_cuda_ep_matches_cpu_ep: logit_error={logit_error:.3e} raw_error={raw_error:.3e} \
         probability_error={probability_error:.3e} (cuda_preferred logit={}, cpu_only logit={})",
        cuda_score.logit, cpu_score.logit
    );
    assert!(logit_error < 2e-5, "ExoMiner ONNX CUDA-EP-vs-CPU-EP logit parity {logit_error}");
    assert!(raw_error < 2e-5, "ExoMiner ONNX CUDA-EP-vs-CPU-EP raw probability parity {raw_error}");
    assert!(probability_error < 2e-5, "ExoMiner ONNX CUDA-EP-vs-CPU-EP calibration parity {probability_error}");
}
