//! Ported from the config-related tests in
//! `Sources/ExoNitContractTests/main.swift` (Swift reference worker,
//! `~/exonit-swift-worker-dev`, commit `5e3978e`). Network/runtime
//! contract tests (HTTP register/task/heartbeat mocks, `WorkerRuntime`) are
//! out of scope for F0 and are not ported here — see docs/PLAN.md (F0b:
//! HTTP client, outbox, runtime loop).

use exonit_core::config::{WorkerConfig, WorkerConfigError};

/// Swift: `testConfigWorkerNamePathSafety`.
#[test]
fn config_worker_name_path_safety() {
    for unsafe_name in ["../victim", "a/b", ".hidden", "worker..other"] {
        let config = WorkerConfig::new("http://127.0.0.1:18400", "test-token", unsafe_name);
        let result = config.validated();
        assert!(
            matches!(result, Err(WorkerConfigError::InvalidWorkerName(_))),
            "unsafe config worker name must fail: {unsafe_name}"
        );
    }
}

/// Swift: `testCurrentConfig` — decodes the shape of the config.json
/// currently checked into the fleet (`~/exonit/worker/config.json`)
/// and applies the `--worker-name`/`--server-url` CLI overrides.
#[test]
fn current_production_config_shape() {
    let json = r#"
    {"server_url":"http://localhost:8400","token":"test-token",
     "worker_name":"worker-default","cpu_threads":2,"gpu_enabled":true,
     "poll_interval_seconds":5,"heartbeat_interval_seconds":30,
     "fits_cache_dir":"fits_cache","gui_port":8402,"fpp_cpu_cores":2,
     "log_file":"worker.log","log_level":"INFO","exominer_mode":"native",
     "ffi_fallback_on_load_fail":true}
    "#;
    let decoded: WorkerConfig = serde_json::from_str(json).expect("decode");
    let effective = decoded
        .applying(Some("worker-<host>-1"), Some("http://127.0.0.1:18400"))
        .expect("applying");
    assert_eq!(effective.worker_name, "worker-<host>-1");
    assert_eq!(effective.server_url, "http://127.0.0.1:18400");
    assert_eq!(effective.token, "test-token");
    assert_eq!(effective.cpu_threads, Some(2));
    assert!(effective.gpu_enabled);
    assert_eq!(effective.exominer_mode, "native");
}

/// Swift: `testRejectNonHTTPURL`.
#[test]
fn reject_non_http_url() {
    let config = WorkerConfig::new("file:///tmp/server", "token", "worker");
    let result = config.validated();
    assert!(
        matches!(result, Err(WorkerConfigError::InvalidServerUrl(_))),
        "file URL must fail"
    );
}

/// Not a direct Swift port (Swift's `URL(string:)` accepts almost anything
/// non-empty), but pins the decode-default behaviour that
/// `testCurrentConfig`/`AgentConfigDeriver` tests rely on implicitly: a
/// config.json missing every optional key still decodes using the same
/// defaults as the Swift memberwise `init`.
#[test]
fn minimal_config_uses_swift_defaults() {
    let json = r#"{"server_url":"http://localhost:8400","token":"t"}"#;
    let decoded: WorkerConfig = serde_json::from_str(json).expect("decode");
    assert_eq!(decoded.worker_name, "worker");
    assert_eq!(decoded.cpu_threads, None);
    assert!(!decoded.gpu_enabled);
    assert_eq!(decoded.poll_interval_seconds, 5.0);
    assert_eq!(decoded.heartbeat_interval_seconds, 30.0);
    assert_eq!(decoded.http_get_timeout_seconds, 30.0);
    assert_eq!(decoded.http_post_timeout_seconds, 10.0);
    assert_eq!(decoded.fits_cache_directory, "fits_cache");
    assert_eq!(decoded.gui_port, 8402);
    assert_eq!(decoded.fpp_cpu_cores, None);
    assert_eq!(decoded.log_file, None);
    assert_eq!(decoded.log_level, "INFO");
    assert_eq!(decoded.exominer_mode, "legacy-tfrecord");
    assert!(!decoded.ffi_fallback_on_load_fail);
    assert!(!decoded.startup_code_sync);
    assert_eq!(decoded.task_pools, None);
}

/// Positive-field validation, covering the guards in `WorkerConfig::validated`
/// (`poll_interval_seconds`, `heartbeat_interval_seconds`, `gui_port`,
/// `cpu_threads`, `fpp_cpu_cores`) that the Swift suite exercises indirectly
/// via `WorkerConfigError.invalidPositiveField`.
#[test]
fn positive_field_guards() {
    let mut base = WorkerConfig::new("http://localhost:8400", "t", "worker");
    base.poll_interval_seconds = 0.0;
    assert!(matches!(
        base.clone().validated(),
        Err(WorkerConfigError::InvalidPositiveField(ref f)) if f == "poll_interval_seconds"
    ));

    let mut base = WorkerConfig::new("http://localhost:8400", "t", "worker");
    base.gui_port = 70_000;
    assert!(matches!(
        base.validated(),
        Err(WorkerConfigError::InvalidPositiveField(ref f)) if f == "gui_port"
    ));

    let mut base = WorkerConfig::new("http://localhost:8400", "t", "worker");
    base.cpu_threads = Some(0);
    assert!(matches!(
        base.validated(),
        Err(WorkerConfigError::InvalidPositiveField(ref f)) if f == "cpu_threads"
    ));

    let mut base = WorkerConfig::new("http://localhost:8400", "t", "worker");
    base.poll_interval_seconds = f64::INFINITY;
    assert!(matches!(
        base.clone().validated(),
        Err(WorkerConfigError::InvalidPositiveField(ref f)) if f == "poll_interval_seconds"
    ));
    base.poll_interval_seconds = f64::NAN;
    assert!(matches!(
        base.validated(),
        Err(WorkerConfigError::InvalidPositiveField(ref f)) if f == "poll_interval_seconds"
    ));

    let mut base = WorkerConfig::new("http://localhost:8400", "t", "worker");
    base.poll_interval_seconds = f64::MAX;
    assert!(matches!(
        base.validated(),
        Err(WorkerConfigError::InvalidPositiveField(ref f)) if f == "poll_interval_seconds"
    ));

    let mut base = WorkerConfig::new("http://localhost:8400", "t", "worker");
    base.heartbeat_interval_seconds = f64::INFINITY;
    assert!(matches!(
        base.validated(),
        Err(WorkerConfigError::InvalidPositiveField(ref f)) if f == "heartbeat_interval_seconds"
    ));

    let mut base = WorkerConfig::new("http://localhost:8400", "t", "worker");
    base.http_post_timeout_seconds = f64::NAN;
    assert!(matches!(
        base.validated(),
        Err(WorkerConfigError::InvalidPositiveField(ref f)) if f == "http_post_timeout_seconds"
    ));
}
