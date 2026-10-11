//! Transliterated from `Sources/ExoNitCore/WorkerConfig.swift` in the Swift
//! reference worker (`~/exonit-swift-worker-dev`, commit `5e3978e`).
//!
//! No `url` crate is in the F0 dependency allowlist (serde, serde_json, libc
//! only), so `server_url` is stored as a plain `String` and validated by hand
//! (`validate_server_url_shape`) instead of Swift's `Foundation.URL`. This is
//! a deliberate scope-driven substitution, not a redesign: the only thing
//! `validated()` needs from the URL is "scheme is http/https and a host is
//! present", which the hand-rolled parser checks exactly like the Swift
//! `serverURL.scheme?.lowercased()` / `serverURL.host != nil` guard.

use crate::cli::worker_name_validator;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerConfigError {
    InvalidServerUrl(String),
    EmptyToken,
    EmptyWorkerName,
    InvalidWorkerName(String),
    InvalidPositiveField(String),
}

impl fmt::Display for WorkerConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkerConfigError::InvalidServerUrl(value) => {
                write!(f, "Invalid HTTP(S) server URL: {value}")
            }
            WorkerConfigError::EmptyToken => write!(f, "Worker token must not be empty"),
            WorkerConfigError::EmptyWorkerName => write!(f, "Worker name must not be empty"),
            WorkerConfigError::InvalidWorkerName(value) => {
                write!(f, "Invalid worker name: {value}")
            }
            WorkerConfigError::InvalidPositiveField(name) => write!(f, "{name} must be positive"),
        }
    }
}
impl std::error::Error for WorkerConfigError {}

fn default_worker_name() -> String {
    "worker".to_string()
}
fn default_poll_interval() -> f64 {
    5.0
}
fn default_heartbeat_interval() -> f64 {
    30.0
}
fn default_http_get_timeout() -> f64 {
    30.0
}
fn default_http_post_timeout() -> f64 {
    10.0
}
fn default_fits_cache_dir() -> String {
    "fits_cache".to_string()
}
fn default_gui_port() -> i64 {
    8402
}
fn default_log_level() -> String {
    "INFO".to_string()
}
fn default_exominer_mode() -> String {
    "legacy-tfrecord".to_string()
}

// Keep values within a practical range and, importantly, within the range
// accepted by the runtime's `Duration::from_secs_f64` conversion. A year is
// already far beyond a useful poll/heartbeat interval, so larger values are
// configuration mistakes rather than meaningful scheduling choices.
const MAX_INTERVAL_SECONDS: f64 = 365.0 * 24.0 * 60.0 * 60.0;

/// Mirrors `WorkerConfig` (Codable struct with a custom `init(from decoder:)`
/// that applies the same defaults as the memberwise `init(...)`). serde
/// `#[serde(default = "...")]` reproduces `decodeIfPresent(...) ?? default`
/// field by field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerConfig {
    #[serde(rename = "server_url")]
    pub server_url: String,
    pub token: String,
    #[serde(rename = "worker_name", default = "default_worker_name")]
    pub worker_name: String,
    #[serde(
        rename = "cpu_threads",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub cpu_threads: Option<i64>,
    #[serde(rename = "gpu_enabled", default)]
    pub gpu_enabled: bool,
    #[serde(rename = "poll_interval_seconds", default = "default_poll_interval")]
    pub poll_interval_seconds: f64,
    #[serde(
        rename = "heartbeat_interval_seconds",
        default = "default_heartbeat_interval"
    )]
    pub heartbeat_interval_seconds: f64,
    #[serde(rename = "http_get_timeout_seconds", default = "default_http_get_timeout")]
    pub http_get_timeout_seconds: f64,
    #[serde(rename = "http_post_timeout_seconds", default = "default_http_post_timeout")]
    pub http_post_timeout_seconds: f64,
    #[serde(rename = "fits_cache_dir", default = "default_fits_cache_dir")]
    pub fits_cache_directory: String,
    #[serde(rename = "gui_port", default = "default_gui_port")]
    pub gui_port: i64,
    #[serde(
        rename = "fpp_cpu_cores",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub fpp_cpu_cores: Option<i64>,
    #[serde(rename = "log_file", default, skip_serializing_if = "Option::is_none")]
    pub log_file: Option<String>,
    #[serde(rename = "log_level", default = "default_log_level")]
    pub log_level: String,
    #[serde(rename = "exominer_mode", default = "default_exominer_mode")]
    pub exominer_mode: String,
    #[serde(rename = "ffi_fallback_on_load_fail", default)]
    pub ffi_fallback_on_load_fail: bool,
    #[serde(rename = "startup_code_sync", default)]
    pub startup_code_sync: bool,
    #[serde(
        rename = "task_pools",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub task_pools: Option<Vec<String>>,
    /// F2b addition, no direct Swift field (Swift instead takes per-file
    /// `EXONIT_CNN_COREML_MODEL`/`EXONIT_EXOMINER_COREML_MODEL` env vars at
    /// the CLI entry point — see `Sources/ExoNitSwiftWorker/main.swift`).
    /// Directory containing `cnn_ffi_v1.onnx` / `exominer_ffi_v6_ensemble.onnx`
    /// (mirrors the Python worker's `worker/models/` convention — see
    /// `worker/cnn_ffi.py`'s `MODEL_PATH`). Resolved relative to the config
    /// file's directory by the worker entry point (`exonit-worker/src/main.rs`),
    /// same as `fits_cache_dir`; `None`/empty here after resolution means "no
    /// models directory configured", not an error — `PrimaryTaskProcessor`
    /// degrades to CNN/ExoMiner scores absent rather than failing (see
    /// `task_processor.rs`).
    #[serde(
        rename = "models_dir",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub models_dir: Option<String>,
}

impl WorkerConfig {
    /// Mirrors the Swift memberwise `init(...)` defaults, used by tests and
    /// call sites building a config programmatically rather than decoding one.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        server_url: impl Into<String>,
        token: impl Into<String>,
        worker_name: impl Into<String>,
    ) -> Self {
        WorkerConfig {
            server_url: server_url.into(),
            token: token.into(),
            worker_name: worker_name.into(),
            cpu_threads: None,
            gpu_enabled: false,
            poll_interval_seconds: 5.0,
            heartbeat_interval_seconds: 30.0,
            http_get_timeout_seconds: 30.0,
            http_post_timeout_seconds: 10.0,
            fits_cache_directory: "fits_cache".to_string(),
            gui_port: 8402,
            fpp_cpu_cores: None,
            log_file: None,
            log_level: "INFO".to_string(),
            exominer_mode: "legacy-tfrecord".to_string(),
            ffi_fallback_on_load_fail: false,
            startup_code_sync: false,
            task_pools: None,
            models_dir: None,
        }
    }

    /// Mirrors `WorkerConfig.validated()`.
    pub fn validated(self) -> Result<WorkerConfig, WorkerConfigError> {
        validate_server_url_shape(&self.server_url)
            .ok_or_else(|| WorkerConfigError::InvalidServerUrl(self.server_url.clone()))?;
        if self.token.trim().is_empty() {
            return Err(WorkerConfigError::EmptyToken);
        }
        if self.worker_name.trim().is_empty() {
            return Err(WorkerConfigError::EmptyWorkerName);
        }
        if !worker_name_validator::is_valid(&self.worker_name) {
            return Err(WorkerConfigError::InvalidWorkerName(
                self.worker_name.clone(),
            ));
        }
        if !(self.poll_interval_seconds.is_finite()
            && self.poll_interval_seconds > 0.0
            && self.poll_interval_seconds <= MAX_INTERVAL_SECONDS)
        {
            return Err(WorkerConfigError::InvalidPositiveField(
                "poll_interval_seconds".to_string(),
            ));
        }
        if !(self.heartbeat_interval_seconds.is_finite()
            && self.heartbeat_interval_seconds > 0.0
            && self.heartbeat_interval_seconds <= MAX_INTERVAL_SECONDS)
        {
            return Err(WorkerConfigError::InvalidPositiveField(
                "heartbeat_interval_seconds".to_string(),
            ));
        }
        for (name, value) in [
            ("http_get_timeout_seconds", self.http_get_timeout_seconds),
            ("http_post_timeout_seconds", self.http_post_timeout_seconds),
        ] {
            if !(value.is_finite() && value > 0.0 && value <= MAX_INTERVAL_SECONDS) {
                return Err(WorkerConfigError::InvalidPositiveField(name.to_string()));
            }
        }
        if !(self.gui_port > 0 && self.gui_port <= 65_535) {
            return Err(WorkerConfigError::InvalidPositiveField(
                "gui_port".to_string(),
            ));
        }
        if let Some(cpu_threads) = self.cpu_threads {
            if cpu_threads <= 0 {
                return Err(WorkerConfigError::InvalidPositiveField(
                    "cpu_threads".to_string(),
                ));
            }
        }
        if let Some(fpp_cpu_cores) = self.fpp_cpu_cores {
            if fpp_cpu_cores <= 0 {
                return Err(WorkerConfigError::InvalidPositiveField(
                    "fpp_cpu_cores".to_string(),
                ));
            }
        }
        Ok(self)
    }

    /// Mirrors `WorkerConfig.applying(workerName:serverURL:)`: applies the
    /// `--worker-name`/`--server-url` CLI overrides on top of a decoded
    /// config, then re-validates.
    pub fn applying(
        &self,
        worker_name: Option<&str>,
        server_url: Option<&str>,
    ) -> Result<WorkerConfig, WorkerConfigError> {
        let mut copy = self.clone();
        if let Some(name) = worker_name {
            copy.worker_name = name.to_string();
        }
        if let Some(url) = server_url {
            // Swift: `URL(string: serverURL)` fails to parse only on
            // pathological input (e.g. empty string); the real scheme/host
            // gate is `validated()` below, run unconditionally afterwards.
            if url.is_empty() {
                return Err(WorkerConfigError::InvalidServerUrl(url.to_string()));
            }
            copy.server_url = url.to_string();
        }
        copy.validated()
    }
}

/// Checks "scheme is http/https (case-insensitive) and a non-empty host
/// follows `://`" — the two things `WorkerConfig.validated()` actually reads
/// off the parsed `URL` in Swift. Returns `None` when the shape is invalid.
fn validate_server_url_shape(url: &str) -> Option<()> {
    let scheme_end = url.find("://")?;
    let scheme = url[..scheme_end].to_ascii_lowercase();
    if scheme != "http" && scheme != "https" {
        return None;
    }
    let rest = &url[scheme_end + 3..];
    let host_end = rest
        .find(|c| c == '/' || c == '?' || c == '#' || c == ':')
        .unwrap_or(rest.len());
    let host = &rest[..host_end];
    if host.is_empty() {
        None
    } else {
        Some(())
    }
}
