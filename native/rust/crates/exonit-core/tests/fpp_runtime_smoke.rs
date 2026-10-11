//! F8 smoke: the native FPP backfill path end-to-end at the processor level
//! — poll payload → metadata → FFI light curve → nearby → `calc_fpp` →
//! result contract — against a mock server serving a real committed parity
//! fixture. This is a FLOW test (contract shapes, success and failure),
//! not a bit-parity test — bit parity is `fpp_e2e_golden.rs`.

#[path = "support/mod.rs"]
mod support;

use exonit_core::api::WorkerApiClient;
use exonit_core::dto::BackfillTask;
use exonit_core::runtime::FitsCache;
use exonit_core::task_processor::{PrimaryTaskProcessor, TaskProcessor};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use support::{MockRequest, MockResponse, MockServer};

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "exonit-rust-fppsmoke-{name}-{}-{:x}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn unhex_arr(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|h| {
            let h = h.as_str().unwrap();
            let bytes: Vec<u8> = (0..h.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap())
                .collect();
            f64::from_le_bytes(bytes.try_into().unwrap())
        })
        .collect()
}

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/fpp/inputs/10757120.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).unwrap()
}

fn backfill_task(input: &Value) -> BackfillTask {
    serde_json::from_value(json!({
        "candidate_id": 4242,
        "tic_id": "10757120",
        "candidate_num": 1,
        "period_days": input["period"],
        "t0": input["t0"],
        "duration_days": input["duration"],
        "depth_ppm": input["depth_ppm"],
        "snr": input["snr"],
        "n_sectors": input["n_sectors"],
    }))
    .unwrap()
}

#[test]
fn fpp_processor_success_flow() {
    let input = fixture();
    let ffi_body = json!({
        "tic_id": "10757120",
        "sectors": [{
            "sector": 14,
            "source": "tglc",
            "time": unhex_arr(&input["time_hex"]),
            "flux": unhex_arr(&input["flux_hex"]),
        }]
    })
    .to_string();
    let nearby_body = input["nearby"].to_string();
    let server = MockServer::start(move |request: &MockRequest| {
        if request.path == "/api/metadata/10757120" {
            return MockResponse::json(
                200,
                r#"{"tic_id":"10757120","fits_files":[],"sectors":[14]}"#,
            );
        }
        if request.path == "/api/ffi/10757120" {
            return MockResponse::json(200, &ffi_body);
        }
        if request.path == "/api/nearby/10757120" {
            return MockResponse::json(200, &nearby_body);
        }
        MockResponse::json(404, r#"{"error":"not_found"}"#)
    });
    let api = Arc::new(WorkerApiClient::new(&server.base_url, "test-token"));
    let cache = Arc::new(FitsCache::new(temp_dir("success")).expect("cache"));
    let processor = PrimaryTaskProcessor::new(api, cache, None, 2, false);

    let task = backfill_task(&input);
    let raw = json!({"candidate_id": 4242, "tic_id": "10757120", "extra": "echo-me"})
        .as_object()
        .unwrap()
        .clone();
    let outcome = processor
        .process_fpp(&task, &raw, "wid")
        .expect("FPP supported");
    assert!(outcome.error.is_none(), "success flow: {:?}", outcome.error);
    let req = outcome.request;
    assert_eq!(req.candidate_id, Some(4242));
    assert_eq!(req.tic_id.as_deref(), Some("10757120"));
    let fpp = req.fpp.expect("fpp present");
    assert!((0.0..=1.0).contains(&fpp), "fpp in [0,1], got {fpp}");
    let verdict = req.verdict.expect("verdict present");
    assert!(
        ["validated", "candidate", "likely_fp"].contains(&verdict.as_str()),
        "unexpected verdict {verdict}"
    );
    assert_eq!(req.candidate.unwrap()["extra"], "echo-me");
}

#[test]
fn fpp_processor_failure_flow_reports_failed_contract() {
    // nearby endpoint down → calc cannot run → -1/-1/"failed" + error msg.
    let input = fixture();
    let ffi_body = json!({
        "tic_id": "10757120",
        "sectors": [{
            "sector": 14,
            "source": "tglc",
            "time": unhex_arr(&input["time_hex"]),
            "flux": unhex_arr(&input["flux_hex"]),
        }]
    })
    .to_string();
    let server = MockServer::start(move |request: &MockRequest| {
        if request.path == "/api/metadata/10757120" {
            return MockResponse::json(
                200,
                r#"{"tic_id":"10757120","fits_files":[],"sectors":[14]}"#,
            );
        }
        if request.path == "/api/ffi/10757120" {
            return MockResponse::json(200, &ffi_body);
        }
        MockResponse::json(500, r#"{"error":"nearby down"}"#)
    });
    let api = Arc::new(WorkerApiClient::new(&server.base_url, "test-token"));
    let cache = Arc::new(FitsCache::new(temp_dir("failure")).expect("cache"));
    let processor = PrimaryTaskProcessor::new(api, cache, None, 2, false);

    let task = backfill_task(&input);
    let raw = serde_json::Map::new();
    let outcome = processor
        .process_fpp(&task, &raw, "wid")
        .expect("FPP supported");
    assert!(outcome.error.is_some(), "failure must carry a message");
    let req = outcome.request;
    assert_eq!(req.fpp, Some(-1.0));
    assert_eq!(req.nfpp, Some(-1.0));
    assert_eq!(req.verdict.as_deref(), Some("failed"));
}
