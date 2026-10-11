//! Transliterated from `Sources/ExoNitCore/WorkerAPI.swift` in the Swift
//! reference worker (`~/exonit-swift-worker-dev`, commit `5e3978e`).
//! The wire DTOs it sends/receives (`RegisterRequest`, `HeartbeatRequest`,
//! `WorkerTask`, `TaskPollResult`, `HeartbeatCommand`, `FfiResponse`,
//! `NearbyResponse`, `StarMetadata`, `UpdateManifest`, `TfRecordFetchResult`,
//! `PrimaryResultRequest`, `FppResultRequest`, `ExoMinerResultRequest`,
//! `StatusResponse`, `BackfillTask(Envelope)`) were already ported to
//! `dto.rs` in F0 — this module is only the HTTP client itself
//! (`WorkerAPIClient` -> `WorkerApiClient`) plus its error type
//! (`WorkerAPIError` -> `WorkerApiError`, which lived in `WorkerAPI.swift`
//! too).
//!
//! HTTP client: `ureq` v3 with rustls (the only HTTP crate in the F0/F0b
//! dependency allowlist — see docs/DESIGN.md). ureq is blocking, matching
//! the "no tokio/async, threads only" rule: `WorkerAPIClient` in Swift is
//! `async`, but every call site in this codebase (runtime.rs, the
//! outbox drain loop, the E2E harness) is sequential per-thread anyway, so a
//! blocking call is behaviourally identical — Swift's `Task { }` concurrency
//! here exists to fit `async/await` syntax, not to overlap I/O.
//!
//! Swift's `URLSession` distinguishes "transport failure" (thrown, any
//! `Error`) from "got an HTTP response" (`HTTPURLResponse`, status checked by
//! hand against `200..<300`) and never raises on a 4xx/5xx status itself.
//! ureq's default behaviour is the opposite (`http_status_as_error` defaults
//! to `true`, turning 4xx/5xx into `Err(ureq::Error::StatusCode)` with no
//! access to the body) — this client disables that
//! (`http_status_as_error(false)`) on every `Agent` it builds so status
//! checking stays explicit and symmetric with Swift, and so `HttpStatus`
//! errors carry the response body prefix like `WorkerAPIError.httpStatus`
//! does.

use crate::dto::{
    BackfillTask, BackfillTaskEnvelope, ExoMinerResultRequest, FfiResponse, FppResultRequest,
    HeartbeatCommand, HeartbeatRequest, HeartbeatResponseBody, NearbyResponse,
    PrimaryResultRequest, RegisterRequest, RegisterResponseBody, StarMetadata, StatusResponse,
    TaskPollResult, TfRecordFetchResult, UpdateManifest, WorkerTask, WorkerTaskResponse,
};
use crate::outbox::OutboxEntry;
use std::fmt;
use std::time::Duration;
use ureq::typestate::{WithBody, WithoutBody};
use ureq::{Agent, RequestBuilder};

/// Mirrors `WorkerAPIError` (`WorkerAPI.swift`). One addition not present in
/// Swift: `Transport`. Swift's `WorkerAPIClient` methods are `async throws`
/// and let the underlying `URLSession` transport error (a `URLError`, e.g.
/// timeout/connection-refused) propagate untyped — callers (`WorkerRuntime`,
/// the contract tests) catch it with a generic `catch` block alongside the
/// two non-`httpStatus` `WorkerAPIError` cases and treat all of them the
/// same way (delivery-unknown in the outbox drain, network failure in the
/// canary gate test). Rust needs one concrete error type per function
/// signature, so transport failures are wrapped as `WorkerApiError::Transport`
/// instead of a second, unrelated error type — `outbox.rs`'s drain logic
/// then matches `HttpStatus` vs. everything else, exactly mirroring Swift's
/// `switch` over `WorkerAPIError` cases plus its catch-all.
#[derive(Debug)]
pub enum WorkerApiError {
    InvalidResponse(String),
    HttpStatus(u16, String),
    MalformedResponse(String),
    Transport(String),
}

impl fmt::Display for WorkerApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkerApiError::InvalidResponse(message) => {
                write!(f, "Server returned a non-HTTP response: {message}")
            }
            WorkerApiError::HttpStatus(code, body) => write!(f, "HTTP {code}: {body}"),
            WorkerApiError::MalformedResponse(message) => {
                write!(f, "Malformed server response: {message}")
            }
            WorkerApiError::Transport(message) => write!(f, "Transport error: {message}"),
        }
    }
}
impl std::error::Error for WorkerApiError {}

/// `URLRequest.timeoutInterval`: 30s for GET, 10s for everything else
/// (`WorkerAPIClient.request(method:path:workerID:)` in Swift). ureq's
/// closest per-request analogue is `timeout_global` ("end-to-end, from DNS
/// lookup to finishing reading the response body" — covers every other
/// timeout), applied per request below.
const GET_TIMEOUT: Duration = Duration::from_secs(30);
const NON_GET_TIMEOUT: Duration = Duration::from_secs(10);

/// Response body prefix kept for error messages, mirrors Swift's
/// `data.prefix(1_024)`.
const ERROR_BODY_PREFIX: usize = 1_024;

/// Endpoints `sendOutboxEntry` is willing to replay — mirrors the Swift
/// allowlist exactly (leading-slash form, as stored in `OutboxEntry.endpoint`).
const ALLOWED_OUTBOX_ENDPOINTS: [&str; 3] = [
    "/api/worker/result",
    "/api/worker/fpp_result",
    "/api/worker/exominer_result",
];

/// Mirrors `WorkerAPIClient`. Blocking (see module doc); safe to share
/// across threads (`ureq::Agent` is internally `Arc`-backed and cheap to
/// clone/share — mirrors Swift's `URLSession`, which is also a shared,
/// thread-safe client).
pub struct WorkerApiClient {
    base_url: String,
    token: String,
    agent: Agent,
    get_timeout: Duration,
    post_timeout: Duration,
}

impl WorkerApiClient {
    /// Mirrors `WorkerAPIClient.init(baseURL:token:session:)` with the
    /// default `session: .shared`: builds a fresh `Agent` with
    /// `http_status_as_error(false)` (see module doc).
    pub fn new(base_url: impl Into<String>, token: impl Into<String>) -> Self {
        Self::with_agent(base_url, token, default_agent())
    }

    /// Test/E2E injection point, mirroring Swift's `session: URLSession`
    /// initializer parameter (contract tests pass a `mockSession()` backed by
    /// `MockURLProtocol`; this crate's tests point a real `ureq::Agent` at a
    /// local test HTTP server instead — see `tests/support/mod.rs`).
    pub fn with_agent(base_url: impl Into<String>, token: impl Into<String>, agent: Agent) -> Self {
        Self::with_agent_and_timeouts(base_url, token, agent, GET_TIMEOUT, NON_GET_TIMEOUT)
    }

    /// Constructs a client with independent end-to-end GET and non-GET
    /// timeouts. The existing `new`/`with_agent` constructors retain the
    /// Swift-compatible 30s/10s defaults.
    pub fn with_timeouts(
        base_url: impl Into<String>,
        token: impl Into<String>,
        get_timeout: Duration,
        post_timeout: Duration,
    ) -> Self {
        Self::with_agent_and_timeouts(base_url, token, default_agent(), get_timeout, post_timeout)
    }

    /// Test/integration variant of [`Self::with_timeouts`] that also accepts
    /// a caller-provided ureq agent.
    pub fn with_agent_and_timeouts(
        base_url: impl Into<String>,
        token: impl Into<String>,
        agent: Agent,
        get_timeout: Duration,
        post_timeout: Duration,
    ) -> Self {
        let base_url = base_url.into();
        WorkerApiClient {
            base_url: base_url.trim_end_matches('/').to_string(),
            token: token.into(),
            agent,
            get_timeout,
            post_timeout,
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Mirrors `WorkerAPIClient.register(_:)`.
    pub fn register(&self, payload: &RegisterRequest) -> Result<String, WorkerApiError> {
        let response: RegisterResponseBody =
            self.send_json("api/worker/register", payload, None)?;
        if response.worker_id.is_empty() {
            return Err(WorkerApiError::MalformedResponse(
                "empty worker_id".to_string(),
            ));
        }
        Ok(response.worker_id)
    }

    /// Mirrors `WorkerAPIClient.pollTask(workerID:)`.
    pub fn poll_task(&self, worker_id: &str) -> Result<TaskPollResult, WorkerApiError> {
        let response: WorkerTaskResponse =
            self.send_without_body("api/worker/task", Some(worker_id))?;
        let Some(task_id) = response.task_id else {
            return Ok(TaskPollResult::Wait {
                seconds: response.wait_seconds,
            });
        };
        let Some(tic_id) = response.tic_id else {
            return Err(WorkerApiError::MalformedResponse(
                "task_id without tic_id".to_string(),
            ));
        };
        Ok(TaskPollResult::Task(WorkerTask {
            task_id,
            tic_id,
            config: response.config,
        }))
    }

    /// Mirrors `WorkerAPIClient.getTask(workerID:)`.
    pub fn get_task(&self, worker_id: &str) -> Result<Option<WorkerTask>, WorkerApiError> {
        match self.poll_task(worker_id)? {
            TaskPollResult::Task(task) => Ok(Some(task)),
            TaskPollResult::Wait { .. } => Ok(None),
        }
    }

    /// Mirrors `WorkerAPIClient.heartbeat(_:)`.
    pub fn heartbeat(
        &self,
        payload: &HeartbeatRequest,
    ) -> Result<HeartbeatCommand, WorkerApiError> {
        let response: HeartbeatResponseBody = self.send_json(
            "api/worker/heartbeat",
            payload,
            Some(payload.worker_id.as_str()),
        )?;
        let Some(raw) = response.command else {
            return Ok(HeartbeatCommand::Continue);
        };
        parse_heartbeat_command(&raw).ok_or_else(|| {
            WorkerApiError::MalformedResponse(format!("unknown heartbeat command: {raw}"))
        })
    }

    /// Mirrors `WorkerAPIClient.getFPPTask(workerID:)`.
    pub fn get_fpp_task(&self, worker_id: &str) -> Result<Option<BackfillTask>, WorkerApiError> {
        let envelope: BackfillTaskEnvelope =
            self.send_without_body("api/worker/fpp_task", Some(worker_id))?;
        Ok(envelope.task)
    }

    /// Like `get_fpp_task` but also returns the raw task object, so the
    /// result post can echo the EXACT server dict as `candidate` (the
    /// Python worker posts `"candidate": task` verbatim, including any
    /// fields the typed struct does not model).
    pub fn get_fpp_task_raw(
        &self,
        worker_id: &str,
    ) -> Result<Option<(BackfillTask, serde_json::Map<String, serde_json::Value>)>, WorkerApiError>
    {
        let envelope: serde_json::Value =
            self.send_without_body("api/worker/fpp_task", Some(worker_id))?;
        let Some(raw_task) = envelope.get("task").and_then(|t| t.as_object()).cloned() else {
            return Ok(None);
        };
        let typed: BackfillTask =
            serde_json::from_value(serde_json::Value::Object(raw_task.clone()))
                .map_err(|e| WorkerApiError::MalformedResponse(format!("fpp task: {e}")))?;
        Ok(Some((typed, raw_task)))
    }

    /// Raw `/api/nearby/{tic}` payload for the native FPP port — the FPP
    /// parity contract consumes the exact JSON the Python worker sees.
    pub fn fetch_nearby_value(&self, tic_id: &str) -> Result<serde_json::Value, WorkerApiError> {
        self.send_without_body(&format!("api/nearby/{tic_id}"), None)
    }

    /// Mirrors `WorkerAPIClient.getExoMinerTask(workerID:)`.
    pub fn get_exominer_task(
        &self,
        worker_id: &str,
    ) -> Result<Option<BackfillTask>, WorkerApiError> {
        let envelope: BackfillTaskEnvelope =
            self.send_without_body("api/worker/exominer_task", Some(worker_id))?;
        Ok(envelope.task)
    }

    /// Mirrors `WorkerAPIClient.postPrimaryResult(_:workerID:)`.
    pub fn post_primary_result(
        &self,
        payload: &PrimaryResultRequest,
        worker_id: &str,
    ) -> Result<StatusResponse, WorkerApiError> {
        self.send_json("api/worker/result", payload, Some(worker_id))
    }

    /// Mirrors `WorkerAPIClient.postFPPResult(_:workerID:)`.
    pub fn post_fpp_result(
        &self,
        payload: &FppResultRequest,
        worker_id: &str,
    ) -> Result<StatusResponse, WorkerApiError> {
        self.send_json("api/worker/fpp_result", payload, Some(worker_id))
    }

    /// Mirrors `WorkerAPIClient.postExoMinerResult(_:workerID:)`.
    pub fn post_exominer_result(
        &self,
        payload: &ExoMinerResultRequest,
        worker_id: &str,
    ) -> Result<StatusResponse, WorkerApiError> {
        self.send_json("api/worker/exominer_result", payload, Some(worker_id))
    }

    /// Mirrors `WorkerAPIClient.fetchMetadata(ticID:workerID:)`.
    pub fn fetch_metadata(
        &self,
        tic_id: &str,
        worker_id: Option<&str>,
    ) -> Result<StarMetadata, WorkerApiError> {
        self.send_without_body(&format!("api/metadata/{tic_id}"), worker_id)
    }

    /// Mirrors `WorkerAPIClient.fetchFFI(ticID:)`: 404 becomes `None`.
    pub fn fetch_ffi(&self, tic_id: &str) -> Result<Option<FfiResponse>, WorkerApiError> {
        match self.send_without_body(&format!("api/ffi/{tic_id}"), None) {
            Ok(response) => Ok(Some(response)),
            Err(WorkerApiError::HttpStatus(404, _)) => Ok(None),
            Err(other) => Err(other),
        }
    }

    /// Mirrors `WorkerAPIClient.fetchNearby(ticID:)`.
    pub fn fetch_nearby(&self, tic_id: &str) -> Result<NearbyResponse, WorkerApiError> {
        self.send_without_body(&format!("api/nearby/{tic_id}"), None)
    }

    /// Mirrors `WorkerAPIClient.downloadFITS(filename:)`, including the
    /// path-traversal guard.
    pub fn download_fits(&self, filename: &str) -> Result<Vec<u8>, WorkerApiError> {
        if filename.is_empty() || filename.contains('/') || filename == "." || filename == ".." {
            return Err(WorkerApiError::MalformedResponse(
                "invalid FITS filename".to_string(),
            ));
        }
        self.perform_bytes(self.get_request(&format!("api/fits/{filename}"), None))
    }

    /// Mirrors `WorkerAPIClient.fetchUpdateManifest()`.
    pub fn fetch_update_manifest(&self) -> Result<UpdateManifest, WorkerApiError> {
        self.send_without_body("api/worker/update", None)
    }

    /// Mirrors `WorkerAPIClient.downloadUpdateFile(filename:)`, including the
    /// `1 or 2 non-empty, non-"."/".." path components` guard (Swift:
    /// `filename.split(separator: "/", omittingEmptySubsequences: false)`).
    pub fn download_update_file(&self, filename: &str) -> Result<Vec<u8>, WorkerApiError> {
        let components: Vec<&str> = filename.split('/').collect();
        let valid = (components.len() == 1 || components.len() == 2)
            && components
                .iter()
                .all(|c| !c.is_empty() && *c != "." && *c != "..");
        if !valid {
            return Err(WorkerApiError::MalformedResponse(
                "invalid update filename".to_string(),
            ));
        }
        self.perform_bytes(self.get_request(&format!("api/worker/update/{filename}"), None))
    }

    /// Mirrors `WorkerAPIClient.fetchTFRecord(ticID:)`: inspects the raw
    /// status/headers itself rather than going through the generic JSON path
    /// (200 = binary body + `X-Sector-Run`/`X-Tic-Id` headers, 202 = a small
    /// JSON "still preprocessing" body, 404 = not available).
    pub fn fetch_tfrecord(&self, tic_id: &str) -> Result<TfRecordFetchResult, WorkerApiError> {
        let request = self.get_request(&format!("api/exominer/tfrecord/{tic_id}"), None);
        let mut response = map_transport(request.call())?;
        let status = response.status().as_u16();
        match status {
            200 => {
                let sector_run =
                    header_value(response.headers(), "X-Sector-Run").unwrap_or_default();
                let response_tic_id = header_value(response.headers(), "X-Tic-Id")
                    .unwrap_or_else(|| tic_id.to_string());
                let data = read_body_bytes(&mut response)?;
                Ok(TfRecordFetchResult::Ready {
                    data,
                    sector_run,
                    tic_id: response_tic_id,
                })
            }
            202 => {
                #[derive(serde::Deserialize)]
                struct Pending {
                    #[serde(rename = "tic_id")]
                    tic_id: String,
                    #[serde(rename = "sector_run")]
                    sector_run: String,
                }
                let bytes = read_body_bytes(&mut response)?;
                let pending: Pending = serde_json::from_slice(&bytes)
                    .map_err(|e| WorkerApiError::MalformedResponse(e.to_string()))?;
                Ok(TfRecordFetchResult::Preprocessing {
                    tic_id: pending.tic_id,
                    sector_run: pending.sector_run,
                })
            }
            404 => Ok(TfRecordFetchResult::NotAvailable),
            other => {
                let bytes = read_body_bytes(&mut response)?;
                Err(WorkerApiError::HttpStatus(other, error_body_text(&bytes)))
            }
        }
    }

    /// Mirrors `WorkerAPIClient.sendOutboxEntry(_:)`: the single replay path
    /// used by `outbox.rs`'s `WorkerResultOutbox::drain`.
    pub fn send_outbox_entry(&self, entry: &OutboxEntry) -> Result<StatusResponse, WorkerApiError> {
        if !ALLOWED_OUTBOX_ENDPOINTS.contains(&entry.endpoint.as_str()) {
            return Err(WorkerApiError::MalformedResponse(
                "outbox endpoint is not allowed".to_string(),
            ));
        }
        let path = entry.endpoint.trim_start_matches('/');
        let request = self
            .post_request(path, Some(entry.worker_id.as_str()))
            .header("Idempotency-Key", entry.id.as_str());
        let response = map_transport(request.send(entry.payload.as_slice()))?;
        self.finish_json(response)
    }

    // -- internals -----------------------------------------------------

    fn get_request(&self, path: &str, worker_id: Option<&str>) -> RequestBuilder<WithoutBody> {
        let mut builder = self
            .agent
            .get(self.url(path))
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "application/json");
        if let Some(worker_id) = worker_id {
            builder = builder.header("X-Worker-ID", worker_id);
        }
        builder
            .config()
            .timeout_global(Some(self.get_timeout))
            .build()
    }

    fn post_request(&self, path: &str, worker_id: Option<&str>) -> RequestBuilder<WithBody> {
        let mut builder = self
            .agent
            .post(self.url(path))
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "application/json");
        if let Some(worker_id) = worker_id {
            builder = builder.header("X-Worker-ID", worker_id);
        }
        builder
            .config()
            .timeout_global(Some(self.post_timeout))
            .build()
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.base_url, path.trim_start_matches('/'))
    }

    /// POST + JSON-decode the response. Every use of this client's `send`
    /// (`register`, `heartbeat`, `postPrimaryResult`, ...) is a POST in
    /// Swift too — see `WorkerAPIClient.send`'s call sites.
    fn send_json<Body: serde::Serialize, Response: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: &Body,
        worker_id: Option<&str>,
    ) -> Result<Response, WorkerApiError> {
        let bytes = serde_json::to_vec(body)
            .map_err(|e| WorkerApiError::MalformedResponse(e.to_string()))?;
        let request = self.post_request(path, worker_id);
        let response = map_transport(request.send(bytes.as_slice()))?;
        self.finish_json(response)
    }

    /// GET + JSON-decode the response. Every use of this client's
    /// `sendWithoutBody` is a GET in Swift too.
    fn send_without_body<Response: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        worker_id: Option<&str>,
    ) -> Result<Response, WorkerApiError> {
        let response = map_transport(self.get_request(path, worker_id).call())?;
        self.finish_json(response)
    }

    fn finish_json<Response: serde::de::DeserializeOwned>(
        &self,
        mut response: ureq::http::Response<ureq::Body>,
    ) -> Result<Response, WorkerApiError> {
        let status = response.status().as_u16();
        let bytes = read_body_bytes(&mut response)?;
        if !(200..300).contains(&status) {
            return Err(WorkerApiError::HttpStatus(status, error_body_text(&bytes)));
        }
        serde_json::from_slice(&bytes).map_err(|e| WorkerApiError::MalformedResponse(e.to_string()))
    }

    fn perform_bytes(
        &self,
        request: RequestBuilder<WithoutBody>,
    ) -> Result<Vec<u8>, WorkerApiError> {
        let mut response = map_transport(request.call())?;
        let status = response.status().as_u16();
        let bytes = read_body_bytes(&mut response)?;
        if !(200..300).contains(&status) {
            return Err(WorkerApiError::HttpStatus(status, error_body_text(&bytes)));
        }
        Ok(bytes)
    }
}

/// Mirrors `HeartbeatCommand(rawValue:)` — a plain string match rather than
/// deserializing through `serde` a second time, since the raw value already
/// arrived as a `String` inside `HeartbeatResponseBody.command`.
fn parse_heartbeat_command(raw: &str) -> Option<HeartbeatCommand> {
    match raw {
        "continue" => Some(HeartbeatCommand::Continue),
        "re-register" => Some(HeartbeatCommand::ReRegister),
        "pause" => Some(HeartbeatCommand::Pause),
        "resume" => Some(HeartbeatCommand::Resume),
        "shutdown" => Some(HeartbeatCommand::Shutdown),
        "debug_on" => Some(HeartbeatCommand::DebugOn),
        "debug_off" => Some(HeartbeatCommand::DebugOff),
        _ => None,
    }
}

fn default_agent() -> Agent {
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .build()
        .into()
}

fn map_transport<T>(result: Result<T, ureq::Error>) -> Result<T, WorkerApiError> {
    result.map_err(|e| WorkerApiError::Transport(e.to_string()))
}

fn read_body_bytes(
    response: &mut ureq::http::Response<ureq::Body>,
) -> Result<Vec<u8>, WorkerApiError> {
    response
        .body_mut()
        .read_to_vec()
        .map_err(|e| WorkerApiError::InvalidResponse(e.to_string()))
}

fn error_body_text(bytes: &[u8]) -> String {
    let prefix = &bytes[..bytes.len().min(ERROR_BODY_PREFIX)];
    match std::str::from_utf8(prefix) {
        Ok(text) => text.to_string(),
        Err(_) => "<binary>".to_string(),
    }
}

fn header_value(headers: &ureq::http::HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}
