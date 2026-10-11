//! Ported from the `WorkerAPIClient` contract tests in
//! `Sources/ExoNitContractTests/main.swift` (Swift reference worker):
//! `testRegisterHTTPContract`, `testTaskHTTPContract`,
//! `testHeartbeatContract`, `testUnknownHeartbeatCommand`,
//! `testExtendedAPIRoutes`, `testBinaryAPIRoutes`. Swift mocks the transport
//! with `MockURLProtocol`; this crate has no such hook for `ureq`, so
//! `tests/support/mod.rs` runs a tiny real HTTP server on loopback instead
//! (see its doc comment for why).

#[path = "support/mod.rs"]
mod support;

use exonit_core::api::{WorkerApiClient, WorkerApiError};
use exonit_core::dto::{
    ExoMinerResultRequest, FppResultRequest, HeartbeatCommand, HeartbeatRequest, PrimaryResultRequest,
    RegisterRequest, TfRecordFetchResult, WorkerCapabilities,
};
use support::{MockRequest, MockResponse, MockServer};

fn capabilities() -> WorkerCapabilities {
    WorkerCapabilities {
        gpu: "metal".to_string(),
        bls_backend: "metal-native".to_string(),
        bls_gpu_info: "Apple M4".to_string(),
        fpp_backend: "metal-native".to_string(),
        cpu_cores: 10,
        fpp_cpu_cores: 2,
        hardware: Default::default(),
    }
}

/// Swift: `testRegisterHTTPContract`.
#[test]
fn register_http_contract() {
    let server = MockServer::start(|request: &MockRequest| {
        assert_eq!(request.method, "POST", "register method");
        assert_eq!(request.path, "/api/worker/register", "register path");
        assert_eq!(request.header("Authorization"), Some("Bearer test-token"), "auth");
        assert_eq!(request.header("Content-Type"), Some("application/json"), "content type");
        let body = request.body_json();
        assert_eq!(body["name"], "worker-m4", "register name");
        MockResponse::json(200, r#"{"worker_id":"wid-123"}"#)
    });
    let client = WorkerApiClient::new(server.base_url, "test-token");
    let id = client
        .register(&RegisterRequest { name: "worker-m4".to_string(), capabilities: capabilities() })
        .expect("register");
    assert_eq!(id, "wid-123", "worker id");
}

/// Swift: `testTaskHTTPContract`.
#[test]
fn task_http_contract() {
    let server = MockServer::start(|request: &MockRequest| {
        assert_eq!(request.method, "GET", "task method");
        assert_eq!(request.path, "/api/worker/task", "task path");
        assert_eq!(request.header("X-Worker-ID"), Some("wid-123"), "worker header");
        MockResponse::json(200, r#"{"task_id":"t-1","tic_id":"150430484"}"#)
    });
    let client = WorkerApiClient::new(server.base_url, "test-token");
    let task = client.get_task("wid-123").expect("get_task").expect("task present");
    assert_eq!(task.task_id, "t-1", "task id");
    assert_eq!(task.tic_id, "150430484", "tic id");
}

/// Swift: `testHeartbeatContract`.
#[test]
fn heartbeat_contract() {
    let server = MockServer::start(|request: &MockRequest| {
        assert_eq!(request.path, "/api/worker/heartbeat", "heartbeat path");
        assert_eq!(request.header("X-Worker-ID"), Some("wid-123"), "worker header");
        MockResponse::json(200, r#"{"command":"pause"}"#)
    });
    let client = WorkerApiClient::new(server.base_url, "test-token");
    let command = client
        .heartbeat(&HeartbeatRequest::new("wid-123".to_string(), "idle".to_string(), None, None, 0, 1.0, 42))
        .expect("heartbeat");
    assert_eq!(command, HeartbeatCommand::Pause, "pause command");
}

/// Swift: `testUnknownHeartbeatCommand`.
#[test]
fn unknown_heartbeat_command_fails_safe() {
    let server = MockServer::start(|_: &MockRequest| MockResponse::json(200, r#"{"command":"format_disk"}"#));
    let client = WorkerApiClient::new(server.base_url, "test-token");
    let error = client
        .heartbeat(&HeartbeatRequest::new("wid-123".to_string(), "idle".to_string(), None, None, 0, 1.0, 42))
        .expect_err("unknown heartbeat command must not be accepted");
    assert!(matches!(error, WorkerApiError::MalformedResponse(_)), "expected malformed_response, got {error:?}");
}

/// Swift: `testExtendedAPIRoutes`.
#[test]
fn extended_api_routes() {
    let server = MockServer::start(|request: &MockRequest| {
        assert_eq!(request.header("X-Worker-ID"), Some("wid"), "worker header");
        match request.path.as_str() {
            "/api/worker/fpp_task" => MockResponse::json(200, r#"{"task":null}"#),
            "/api/worker/exominer_task" => MockResponse::json(
                200,
                r#"{"task":{"candidate_id":7,"tic_id":"9","candidate_num":1,"period_days":2,"t0":1,"duration_days":0.1,"depth_ppm":500,"snr":12,"n_sectors":1}}"#,
            ),
            "/api/worker/fpp_result" | "/api/worker/exominer_result" | "/api/worker/result" => {
                assert_eq!(request.method, "POST", "result POST");
                MockResponse::json(200, r#"{"status":"ok"}"#)
            }
            other => panic!("unexpected extended route {other}"),
        }
    });
    let client = WorkerApiClient::new(server.base_url, "test-token");
    let fpp_task = client.get_fpp_task("wid").expect("fpp task call");
    let exo_task = client.get_exominer_task("wid").expect("exominer task call");
    assert!(fpp_task.is_none(), "empty FPP task");
    assert_eq!(exo_task.expect("exominer task present").candidate_id, 7, "ExoMiner task");

    client
        .post_fpp_result(
            &FppResultRequest {
                candidate_id: Some(7),
                tic_id: Some("9".to_string()),
                fpp: Some(0.02),
                nfpp: Some(0.001),
                probability_planet: Some(0.97),
                verdict: Some("planet_candidate".to_string()),
                candidate: None,
            },
            "wid",
        )
        .expect("post fpp result");
    client
        .post_exominer_result(
            &ExoMinerResultRequest {
                candidate_id: Some(7),
                tic_id: Some("9".to_string()),
                score: Some(0.93),
                tce_period: Some(2.0),
                sector_run: Some("1-69".to_string()),
            },
            "wid",
        )
        .expect("post exominer result");
    let mut failed = PrimaryResultRequest::new(Some("t".to_string()), Some("9".to_string()));
    failed.status = "failed".to_string();
    failed.error = Some("load_failed".to_string());
    client.post_primary_result(&failed, "wid").expect("post primary result");
}

/// Swift: `testBinaryAPIRoutes`.
#[test]
fn binary_api_routes() {
    let server = MockServer::start(|request: &MockRequest| match request.path.as_str() {
        "/api/exominer/tfrecord/123" => MockResponse::bytes(200, "application/octet-stream", vec![1, 2, 3])
            .with_header("X-Sector-Run", "1-69")
            .with_header("X-Tic-Id", "123"),
        "/api/fits/test.fits" => MockResponse::bytes(200, "application/fits", vec![4, 5]),
        "/api/worker/update" => MockResponse::json(200, r#"{"files":[]}"#),
        "/api/worker/update/models/model.onnx" => MockResponse::bytes(200, "application/octet-stream", vec![6, 7]),
        other => panic!("unexpected binary route {other}"),
    });
    let client = WorkerApiClient::new(server.base_url, "test-token");

    let tf = client.fetch_tfrecord("123").expect("fetch tfrecord");
    match tf {
        TfRecordFetchResult::Ready { data, sector_run, tic_id } => {
            assert_eq!(data, vec![1, 2, 3], "TFRecord payload");
            assert_eq!(sector_run, "1-69");
            assert_eq!(tic_id, "123");
        }
        other => panic!("expected TFRecord ready, got {other:?}"),
    }
    let fits_data = client.download_fits("test.fits").expect("download fits");
    assert_eq!(fits_data, vec![4, 5], "FITS bytes");
    let manifest = client.fetch_update_manifest().expect("fetch update manifest");
    assert!(manifest.files.is_empty(), "update manifest");
    let update_data = client.download_update_file("models/model.onnx").expect("download update file");
    assert_eq!(update_data, vec![6, 7], "update bytes");
}
