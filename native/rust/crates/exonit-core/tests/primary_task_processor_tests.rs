//! F2b integration tests for `PrimaryTaskProcessor`
//! (`exonit_core::task_processor::PrimaryTaskProcessor`), the real
//! FITS -> BLS -> vetting -> CNN/ExoMiner FFI -> injection-recovery
//! pipeline wired up to replace `StubTaskProcessor` in production. Two of
//! these mirror the Swift reference worker's
//! `testPrimaryFailureAndCancellation` contract test (API-failure and
//! no-light-curve both produce `status: "failed"`, never a panic/crash);
//! the third is new coverage for the F2b task's explicit ask — that a
//! worker with no ONNX models configured degrades to CNN/ExoMiner scores
//! absent rather than failing the task, exactly like the Python worker's
//! real production `FileNotFoundError` -> `classifier_ffi = None` fallback
//! (see `task_processor.rs`'s `PrimaryTaskProcessor` doc comment for why
//! this, not Swift's `main.swift` CLI wiring, is the behaviour mirrored).

#[path = "support/mod.rs"]
mod support;

use exonit_core::api::WorkerApiClient;
use exonit_core::dto::WorkerTask;
use exonit_core::runtime::FitsCache;
use exonit_core::task_processor::{PrimaryTaskProcessor, TaskProcessor};
use serde_json::Map;
use std::path::PathBuf;
use std::sync::Arc;
use support::{MockRequest, MockResponse, MockServer};

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "exonit-rust-primary-{name}-{}-{:x}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn processor(base_url: &str, cache_name: &str) -> PrimaryTaskProcessor {
    let api = Arc::new(WorkerApiClient::new(base_url, "test-token"));
    let fits_cache = Arc::new(FitsCache::new(temp_dir(cache_name)).expect("create fits cache"));
    // `models_dir: None` — the degradation-under-test configuration; see
    // `completes_without_models_and_degrades_scores_gracefully` below.
    // `gpu_enabled: false` — these tests exercise the network/pipeline
    // control flow, not BLS backend selection (that is F4a's
    // `resolve_compute_backend`, exercised separately).
    PrimaryTaskProcessor::new(api, fits_cache, None, 2, false)
}

fn task(tic_id: &str, config: Map<String, serde_json::Value>) -> WorkerTask {
    WorkerTask { task_id: "task-1".to_string(), tic_id: tic_id.to_string(), config }
}

/// Swift: `testPrimaryFailureAndCancellation`'s 500-response half — a
/// transport/API failure anywhere in the pipeline must surface as
/// `status: "failed"` with a descriptive `error`, never a panic.
#[test]
fn metadata_fetch_failure_returns_failed_status() {
    let server = MockServer::start(|_request: &MockRequest| MockResponse::json(500, r#"{"error":"temporary failure"}"#));
    let processor = processor(&server.base_url, "metadata-failure");
    let result = processor.process(&task("123", Map::new()), "wid");
    assert_eq!(result.status, "failed", "API errors must surface as a failed result, not a crash");
    assert!(result.error.is_some(), "failed result must carry a description");
    assert_eq!(result.task_id.as_deref(), Some("task-1"));
    assert_eq!(result.tic_id.as_deref(), Some("123"));
}

/// No FITS files listed and no FFI sectors available (404) -> `NoLightCurve`,
/// surfaced the same way as any other pipeline error: `status: "failed"`.
#[test]
fn no_light_curve_returns_failed_status() {
    let server = MockServer::start(|request: &MockRequest| {
        if request.path == "/api/metadata/123" {
            return MockResponse::json(200, r#"{"tic_id":"123","fits_files":[],"sectors":[]}"#);
        }
        MockResponse::json(404, r#"{"error":"not_found"}"#)
    });
    let processor = processor(&server.base_url, "no-light-curve");
    let result = processor.process(&task("123", Map::new()), "wid");
    assert_eq!(result.status, "failed");
    assert!(
        result.error.as_deref().is_some_and(|e| e.contains("no usable light curve")),
        "expected a no-usable-light-curve error, got {:?}",
        result.error
    );
}

/// F2b's explicit degradation requirement: with no models directory
/// configured, the full BLS -> vetting pipeline still runs to completion
/// over a real FITS fixture and produces plausible candidates, but every
/// `cnn_scores` entry and every candidate's `exominer_ffi_score` stay
/// `None` — scores absent, not a crash (matches the Python worker's
/// `classifier_ffi = None` fallback; see `PrimaryTaskProcessor`'s
/// `cnn_available()`/`exominer_available()` for the machinery that also
/// feeds `WorkerRuntime`'s honest `cnn_ffi_available`/`exominer_ffi_available`
/// register capabilities).
#[test]
fn completes_without_models_and_degrades_scores_gracefully() {
    // Deliberately the SAME two-sector TGLC fixture `multi_planet_golden.rs`
    // uses (TIC 10756829, s0014+s0015), not `tests/data/qlp_sample.fits`:
    // that single-sector QLP fixture has a sub-5-day gap, so
    // `preprocess::prepare`'s conditional gap-fill runs with its
    // intentionally unseeded noise (matches real production — see
    // `preprocess.rs`'s doc comment) and was observed flaking this exact
    // test ~50% of the time when that noise pushed the lone candidate's SNR
    // just under/over the 7.0 threshold. The multi-sector gap here is well
    // over 5 days, so no RNG is on the critical path — the same reason
    // `multi_planet_golden.rs`'s tight-tolerance assertions are reproducible
    // run to run.
    let fits_paths = [
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/data/hlsp_tglc_tess_ffi_gaiaid-2080664784495597312-s0014-cam1-ccd1_tess_v1_llc.fits"
        )),
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/data/hlsp_tglc_tess_ffi_gaiaid-2080664784495597312-s0015-cam1-ccd2_tess_v1_llc.fits"
        )),
    ];
    for path in &fits_paths {
        assert!(path.exists(), "expected local TGLC fixture at {}", path.display());
    }
    let fits_bytes: Vec<Vec<u8>> = fits_paths.iter().map(|p| std::fs::read(p).expect("read TGLC fixture")).collect();
    let fits_bytes_for_server = fits_bytes.clone();

    let server = MockServer::start(move |request: &MockRequest| {
        if request.path == "/api/metadata/123" {
            return MockResponse::json(
                200,
                r#"{"tic_id":"123","fits_files":["s0014.fits","s0015.fits"],"sectors":[14,15]}"#,
            );
        }
        if request.path == "/api/fits/s0014.fits" {
            return MockResponse::bytes(200, "application/fits", fits_bytes_for_server[0].clone());
        }
        if request.path == "/api/fits/s0015.fits" {
            return MockResponse::bytes(200, "application/fits", fits_bytes_for_server[1].clone());
        }
        MockResponse::json(404, r#"{"error":"not_found"}"#)
    });

    let api = Arc::new(WorkerApiClient::new(&server.base_url, "test-token"));
    let fits_cache = Arc::new(FitsCache::new(temp_dir("degrade-scores")).expect("create fits cache"));
    let processor = PrimaryTaskProcessor::new(api, fits_cache, None, 2, false);
    assert!(!processor.cnn_available(), "no models_dir configured -> CNN classifier must be absent");
    assert!(!processor.exominer_available(), "no models_dir configured -> ExoMiner classifier must be absent");

    // Otherwise the default search (matches `multi_planet_golden.rs`, which
    // finds 5 stable candidates here) — only injection-recovery is disabled,
    // since that's a separate, already-tested RNG subsystem
    // (`injection_recovery_golden.rs`) this test isn't exercising.
    let mut config = Map::new();
    config.insert("run_injection_recovery".to_string(), serde_json::json!(false));

    let result = processor.process(&task("123", config), "wid");
    assert_eq!(result.status, "completed", "pipeline must complete with error={:?}", result.error);
    assert!(!result.candidates.is_empty(), "expected at least one BLS candidate");
    assert_eq!(result.vetting_results.len(), result.candidates.len(), "one vetting result per candidate");
    assert_eq!(result.cnn_scores.len(), result.candidates.len(), "one cnn_scores slot per candidate");
    assert!(result.cnn_scores.iter().all(Option::is_none), "no CNN classifier configured -> every score must be absent");
    assert!(
        result.candidates.iter().all(|c| c.exominer_ffi_score.is_none()),
        "no ExoMiner classifier configured -> exominer_ffi_score must stay absent on every candidate"
    );
    assert!(result.completeness.is_none(), "run_injection_recovery=false must skip injection-recovery");
    assert!(result.point_count > 0, "expected a non-empty light curve");
    assert_eq!(result.sector_count, 2);

    for candidate in &result.candidates {
        assert!(candidate.period_days > 0.0, "candidate period must be positive: {candidate:?}");
        assert!(candidate.duration_days > 0.0, "candidate duration must be positive: {candidate:?}");
        assert!(candidate.snr.is_finite() && candidate.snr > 0.0, "candidate SNR must be a finite positive number: {candidate:?}");
        assert!(candidate.depth_ppm.is_finite(), "candidate depth must be finite: {candidate:?}");
    }
}
