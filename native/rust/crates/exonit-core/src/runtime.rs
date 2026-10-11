//! Transliterated from `Sources/ExoNitCore/WorkerRuntime.swift` in the Swift
//! reference worker (`~/exonit-swift-worker-dev`, commit `5e3978e`),
//! adapted from Swift's `actor`/`async`/`Task` concurrency to plain OS
//! threads (CLAUDE.md / DESIGN.md rule: no tokio/async in this codebase —
//! "the worker is process-per-instance", blocking I/O throughout).
//!
//! Structural translation of the concurrency model:
//!   - `WorkerRuntimeState` (Swift `actor`, serializing access to mutable
//!     worker state) -> `Mutex<StateInner>` here. An actor is exactly
//!     "one mutex, methods instead of raw field access" once you remove
//!     Swift's `async` sugar, so this is a direct, not approximate,
//!     translation.
//!   - `WorkerRuntime.run()`'s `Task { heartbeatLoop() }` background task,
//!     cancelled via `defer { heartbeat.cancel() }` -> a `std::thread::spawn`
//!     background thread, told to stop via `WorkerRuntimeState`'s
//!     `shutdown_requested` flag (checked between every blocking sleep chunk
//!     — see `interruptible_sleep` — since a plain OS thread has no
//!     built-in cancellation channel the way a Swift `Task` does).
//!   - `WorkerRuntime`'s stored `config`/`api`/`durableOutbox`/`outbox`/
//!     `processor`/`pools` fields -> `RuntimeShared`, wrapped in one `Arc` so
//!     the main loop (this thread) and the heartbeat loop (the spawned
//!     thread) can both hold a cheap, `Send + Sync` handle to the same
//!     state without Rust's borrow checker fighting the cross-thread
//!     lifetime (an `actor`'s reference-type nature gives Swift this for
//!     free via ARC).
//!
//! Two pieces below have **no Swift source** at all — DESIGN.md's F0b task
//! list calls for them by name ("single-flight prefetch, atomic LRU cache
//! for the FITS cache") but neither exists in the Swift port (its
//! `PrimaryTaskProcessor` just downloads straight into `cacheDirectory` with
//! no eviction or de-duplication) nor, for the *task*-prefetch flavour, is
//! it wired to anything F0b can exercise yet (see `FitsCache` and
//! `task_processor.rs`'s doc comment for the scope call made here).

use crate::api::{WorkerApiClient, WorkerApiError};
use crate::config::WorkerConfig;
use crate::dto::{
    HeartbeatCommand, HeartbeatRequest, RegisterRequest, TaskPollResult, WorkerCapabilities,
};
use crate::outbox::{DurableOutbox, OutboxError, WorkerResultOutbox};
use crate::task_processor::{PrimaryTaskProcessor, TaskProcessor};
use crate::telemetry::Telemetry;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Mirrors `WorkerRuntimeError`. `Api`/`Outbox` are additions: Swift lets
/// `WorkerAPIError`/`OutboxError` (and any other thrown error) propagate
/// untyped out of `run()`'s `async throws`; `run()` here returns one
/// concrete `Result` type, so those get wrapped rather than passed through.
///
/// F8 (2026-08-05, `docs/PLAN.md`) removed the `AmbiguousDelivery(i64)`
/// variant that used to exist here: `main_loop`/`process_one` killed the
/// worker whenever the outbox had any `delivery_state = 'unknown'` entry
/// (real incident 2026-08-05: 15 fleet instances killed this way, requiring
/// manual reconciliation). That is no longer needed — the server's result
/// endpoints are idempotent upserts (see `outbox.rs::WorkerResultOutbox::drain`'s
/// doc comment for the exact citations), so a transport-ambiguous delivery is
/// just retried like any other transient failure instead of being fatal. The
/// "don't poll for new work while the outbox is non-empty" guard in
/// `main_loop` still applies unchanged, so retries are always drained before
/// any new task is picked up.
#[derive(Debug)]
pub enum WorkerRuntimeError {
    UnsupportedTaskPools(Vec<String>),
    /// Not in Swift: `fpp_backfill`/`exominer_backfill` are recognised pool
    /// names (same allowed-set as Swift) but, unlike Swift (which requires a
    /// configured Python science sidecar for them), this Rust worker has no
    /// backfill processor implementation at all yet — F2/F5 scope. See
    /// `WorkerRuntime::new`.
    BackfillPoolsUnimplemented(Vec<String>),
    CanaryNotAuthorized,
    Api(String),
    Outbox(String),
}

impl fmt::Display for WorkerRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkerRuntimeError::UnsupportedTaskPools(pools) => {
                write!(f, "unknown task pools: {}", pools.join(","))
            }
            WorkerRuntimeError::BackfillPoolsUnimplemented(pools) => {
                write!(
                    f,
                    "task pools not yet implemented in the Rust worker: {}",
                    pools.join(",")
                )
            }
            WorkerRuntimeError::CanaryNotAuthorized => {
                write!(f, "real worker execution requires EXONIT_RUST_CANARY=1")
            }
            WorkerRuntimeError::Api(message) => write!(f, "{message}"),
            WorkerRuntimeError::Outbox(message) => write!(f, "{message}"),
        }
    }
}
impl std::error::Error for WorkerRuntimeError {}

impl From<OutboxError> for WorkerRuntimeError {
    fn from(error: OutboxError) -> Self {
        WorkerRuntimeError::Outbox(error.to_string())
    }
}

// ---------------------------------------------------------------------------
// WorkerRuntimeState — mirrors the Swift `actor`
// ---------------------------------------------------------------------------

#[derive(Default)]
struct StateInner {
    worker_id: String,
    paused: bool,
    shutdown_requested: bool,
    re_registering: bool,
    debug: bool,
    current_tic: Option<String>,
    phase: Option<String>,
    stars_done: i64,
    last_error: Option<String>,
}

/// Mirrors `WorkerRuntimeState`.
pub struct WorkerRuntimeState {
    inner: Mutex<StateInner>,
}

impl WorkerRuntimeState {
    pub fn new() -> Self {
        WorkerRuntimeState {
            inner: Mutex::new(StateInner::default()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, StateInner> {
        self.inner
            .lock()
            .expect("WorkerRuntimeState mutex poisoned")
    }

    pub fn worker_id(&self) -> String {
        self.lock().worker_id.clone()
    }
    pub fn registered(&self, id: String) {
        self.lock().worker_id = id;
    }
    pub fn begin(&self, tic: String, phase: String) {
        let mut state = self.lock();
        state.current_tic = Some(tic);
        state.phase = Some(phase);
    }
    pub fn completed(&self) {
        let mut state = self.lock();
        state.current_tic = None;
        state.phase = None;
        state.stars_done += 1;
        state.last_error = None;
    }
    pub fn failed(&self, message: String) {
        let mut state = self.lock();
        state.current_tic = None;
        state.phase = None;
        state.last_error = Some(message);
    }
    pub fn heartbeat_failed(&self, message: String) {
        self.lock().last_error = Some(message);
    }
    pub fn request_shutdown(&self) {
        self.lock().shutdown_requested = true;
    }
    pub fn shutdown_requested(&self) -> bool {
        self.lock().shutdown_requested
    }
    pub fn paused(&self) -> bool {
        self.lock().paused
    }
    pub fn re_registering(&self) -> bool {
        self.lock().re_registering
    }
    fn set_re_registering(&self, value: bool) {
        self.lock().re_registering = value;
    }
    pub fn debug(&self) -> bool {
        self.lock().debug
    }
    pub fn current_tic(&self) -> Option<String> {
        self.lock().current_tic.clone()
    }
    pub fn phase(&self) -> Option<String> {
        self.lock().phase.clone()
    }
    pub fn stars_done(&self) -> i64 {
        self.lock().stars_done
    }
    pub fn last_error(&self) -> Option<String> {
        self.lock().last_error.clone()
    }

    /// Mirrors `WorkerRuntimeState.apply(_:)`.
    pub fn apply(&self, command: HeartbeatCommand) {
        let mut state = self.lock();
        match command {
            HeartbeatCommand::Pause => state.paused = true,
            HeartbeatCommand::Resume => state.paused = false,
            HeartbeatCommand::Shutdown => state.shutdown_requested = true,
            HeartbeatCommand::DebugOn => state.debug = true,
            HeartbeatCommand::DebugOff => state.debug = false,
            HeartbeatCommand::Continue | HeartbeatCommand::ReRegister => {}
        }
    }
}

impl Default for WorkerRuntimeState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FitsCache — no Swift source; see module doc comment for scope rationale.
// ---------------------------------------------------------------------------

/// Default cap on the on-disk FITS cache. Swift's `PrimaryTaskProcessor`
/// never evicts at all; this is a new, forward-looking bound (DESIGN.md's
/// F0b task list calls for an "atomic LRU cache" by name) sized generously
/// for a single worker instance's working set rather than tuned against any
/// measured footprint yet — revisit once F2's real multi-sector fetch
/// pattern exists to size it against.
const DEFAULT_FITS_CACHE_MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;

struct SingleFlightEntry {
    result: Mutex<Option<Result<(), String>>>,
    condvar: Condvar,
}

/// On-disk FITS cache keyed by filename, with:
///   - **atomic writes**: a fetched file is written to a hidden temp path
///     and `rename`d into place, so a process killed mid-download never
///     leaves a partial file at the real cache path for the next reader to
///     trip over;
///   - **single-flight de-duplication**: concurrent `get_or_fetch` calls for
///     the *same* filename block on one shared in-flight fetch rather than
///     racing duplicate downloads (same spirit as the Python worker's
///     `_PrefetchSlot` single-flight lock in `worker/worker.py`, applied
///     here to concurrent *fetches of one key* rather than Python's
///     background-prefetch-of-the-next-task use case — see the module doc
///     comment for why that second flavour isn't ported in F0b);
///   - **LRU eviction**: bounded by `max_bytes`, tracked via file mtimes
///     (`platform::touch_mtime` "touches" a file on cache hit) rather than
///     an in-memory index, so the cache stays correct across worker restarts
///     without needing to persist any index of its own.
pub struct FitsCache {
    directory: PathBuf,
    max_bytes: u64,
    in_flight: Mutex<HashMap<String, Arc<SingleFlightEntry>>>,
}

impl FitsCache {
    pub fn new(directory: impl Into<PathBuf>) -> std::io::Result<Self> {
        Self::with_max_bytes(directory, DEFAULT_FITS_CACHE_MAX_BYTES)
    }

    pub fn with_max_bytes(directory: impl Into<PathBuf>, max_bytes: u64) -> std::io::Result<Self> {
        let directory = directory.into();
        std::fs::create_dir_all(&directory)?;
        Ok(FitsCache {
            directory,
            max_bytes,
            in_flight: Mutex::new(HashMap::new()),
        })
    }

    /// On-disk directory backing this cache. F2b addition: `PrimaryTaskProcessor`
    /// needs the real file path of a cached FITS file (not just its byte
    /// count, which `get_or_fetch` already returns) to hand to
    /// `fits::read_light_curve`.
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Returns the on-disk path for `filename`, downloading it via `fetch`
    /// on a cache miss (or reusing an in-flight fetch already under way for
    /// the same filename). `fetch` is only ever invoked by the single
    /// "leader" caller for a given filename at a time.
    pub fn get_or_fetch<E: fmt::Display>(
        &self,
        filename: &str,
        fetch: impl FnOnce() -> Result<Vec<u8>, E>,
    ) -> Result<u64, String> {
        let path = cache_path(&self.directory, filename)?;
        if let Ok(metadata) = std::fs::metadata(&path) {
            touch(&path);
            return Ok(metadata.len());
        }

        let (entry, is_leader) = {
            let mut in_flight = self.in_flight_lock();
            if let Some(existing) = in_flight.get(filename) {
                (existing.clone(), false)
            } else {
                let fresh = Arc::new(SingleFlightEntry {
                    result: Mutex::new(None),
                    condvar: Condvar::new(),
                });
                in_flight.insert(filename.to_string(), fresh.clone());
                (fresh, true)
            }
        };

        if is_leader {
            let outcome = fetch().map_err(|e| e.to_string()).and_then(|bytes| {
                write_atomic(&self.directory, filename, &bytes).map_err(|e| e.to_string())
            });
            {
                let mut slot = entry
                    .result
                    .lock()
                    .expect("FitsCache single-flight slot mutex poisoned");
                *slot = Some(outcome.clone());
            }
            entry.condvar.notify_all();
            self.in_flight_lock().remove(filename);
            if outcome.is_ok() {
                self.evict_if_needed();
            }
            outcome?;
        } else {
            let mut slot = entry
                .result
                .lock()
                .expect("FitsCache single-flight slot mutex poisoned");
            while slot.is_none() {
                slot = entry
                    .condvar
                    .wait(slot)
                    .expect("FitsCache single-flight condvar wait failed");
            }
            slot.clone()
                .expect("single-flight slot signalled with no result")?;
        }
        Ok(std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0))
    }

    fn in_flight_lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Arc<SingleFlightEntry>>> {
        self.in_flight
            .lock()
            .expect("FitsCache in-flight map mutex poisoned")
    }

    fn evict_if_needed(&self) {
        let read_dir = match std::fs::read_dir(&self.directory) {
            Ok(read_dir) => read_dir,
            Err(error) => {
                eprintln!(
                    "FitsCache: cannot scan {} for eviction: {error}",
                    self.directory.display()
                );
                return;
            }
        };
        let mut files: Vec<(PathBuf, std::time::SystemTime, u64)> = Vec::new();
        let mut total: u64 = 0;
        for entry in read_dir {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    eprintln!("FitsCache: cannot inspect cache entry during eviction: {error}");
                    continue;
                }
            };
            let path = entry.path();
            let is_hidden_temp = path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with('.'))
                .unwrap_or(false);
            if is_hidden_temp {
                continue;
            }
            let metadata = match entry.metadata() {
                Ok(metadata) => metadata,
                Err(error) => {
                    eprintln!(
                        "FitsCache: cannot stat {} during eviction: {error}",
                        path.display()
                    );
                    continue;
                }
            };
            if !metadata.is_file() {
                continue;
            }
            let mtime = metadata
                .modified()
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            total += metadata.len();
            files.push((path, mtime, metadata.len()));
        }
        if total <= self.max_bytes {
            return;
        }
        files.sort_by_key(|(_, mtime, _)| *mtime);
        let mut over = total - self.max_bytes;
        for (path, _, size) in files {
            if over == 0 {
                break;
            }
            match std::fs::remove_file(&path) {
                Ok(()) => over = over.saturating_sub(size),
                Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
                    eprintln!("FitsCache: cannot evict {}: {error}", path.display());
                }
                Err(_) => {}
            }
        }
    }
}

fn write_atomic(directory: &Path, filename: &str, bytes: &[u8]) -> std::io::Result<()> {
    let _ = cache_path(directory, filename).map_err(std::io::Error::other)?;
    let final_path = directory.join(filename);
    // A fixed `.{filename}.tmp` lets two worker processes overwrite each
    // other's staging file.  `create_new` plus pid/process-local sequencing
    // gives every writer an exclusive temporary path while retaining the
    // atomic final rename.
    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let mut tmp_file = None;
    let mut tmp_path = PathBuf::new();
    for _ in 0..16 {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let candidate = directory.join(format!(
            ".{filename}.tmp.{}.{}",
            std::process::id(),
            sequence
        ));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                tmp_file = Some(file);
                tmp_path = candidate;
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    let Some(mut file) = tmp_file else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "unable to allocate unique FITS cache temporary file",
        ));
    };
    use std::io::Write;
    if let Err(error) = file.write_all(bytes).and_then(|_| file.sync_all()) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(error);
    }
    drop(file);
    if let Err(error) = std::fs::rename(&tmp_path, &final_path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(error);
    }
    Ok(())
}

/// Resolve only a single ordinary filename.  FITS names come from the API;
/// allowing separators here would let a malformed response escape the cache
/// directory (and would also make the eviction/accounting boundary ambiguous).
fn cache_path(directory: &Path, filename: &str) -> Result<PathBuf, String> {
    let path = Path::new(filename);
    if filename.is_empty() || filename.contains('\0') || path.is_absolute() {
        return Err(format!("invalid FITS cache filename: {filename:?}"));
    }
    let mut components = path.components();
    match (components.next(), components.next()) {
        (Some(std::path::Component::Normal(_)), None) => Ok(directory.join(path)),
        _ => Err(format!(
            "invalid FITS cache filename (must be a basename): {filename:?}"
        )),
    }
}

/// Sets both atime and mtime to "now". Delegates to
/// `crate::platform::touch_mtime` (Task 1 of
/// `docs/plans/2026-08-06-win32-port-pcmarc.md`: `File::set_modified`, a
/// single portable implementation replacing the previous
/// `libc::utimensat(UTIME_NOW)` call — the ONE documented behavior-shape
/// change of that refactor, see `platform::touch_mtime`'s doc comment).
fn touch(path: &Path) {
    crate::platform::touch_mtime(path);
}

// ---------------------------------------------------------------------------
// WorkerRuntime — mirrors WorkerRuntime.swift
// ---------------------------------------------------------------------------

struct RuntimeShared {
    config: WorkerConfig,
    api: Arc<WorkerApiClient>,
    state: Arc<WorkerRuntimeState>,
    durable_outbox: Arc<DurableOutbox>,
    result_outbox: WorkerResultOutbox,
    processor: Arc<dyn TaskProcessor>,
    pools: HashSet<String>,
    capabilities: WorkerCapabilities,
    started: Instant,
}

/// Mirrors `WorkerRuntime`.
pub struct WorkerRuntime {
    shared: Arc<RuntimeShared>,
    canary_authorized: bool,
}

impl WorkerRuntime {
    /// Mirrors `WorkerRuntime.init(config:outboxURL:...canaryAuthorized:session:)`.
    /// F2b: builds the real `PrimaryTaskProcessor` (BLS -> vetting -> CNN/
    /// ExoMiner FFI -> injection-recovery), loading ONNX models from
    /// `config.models_dir` if set (already resolved to an absolute path by
    /// the caller — see `exonit-worker/src/main.rs`) — missing/absent models
    /// degrade to scores-absent rather than failing construction, see
    /// `task_processor.rs`. Use [`Self::with_processor`] to inject a
    /// different `TaskProcessor` (e.g. `StubTaskProcessor`, or a test
    /// double) instead.
    pub fn new(
        config: WorkerConfig,
        outbox_path: &Path,
        canary_authorized: bool,
        agent: Option<ureq::Agent>,
    ) -> Result<Self, WorkerRuntimeError> {
        Self::build(config, outbox_path, canary_authorized, agent, None)
    }

    /// Test/E2E injection point: builds the runtime around a caller-supplied
    /// `TaskProcessor` instead of the built-in `PrimaryTaskProcessor` — used
    /// by `exonit-fake-integration`'s stub path (no `EXONIT_MODELS_DIR`) and
    /// by any test that wants `StubTaskProcessor`'s network-only behaviour
    /// without touching ONNX models. Capabilities honestly report
    /// `"custom-processor"` for the BLS/injection backend fields rather than
    /// claiming the built-in science pipeline ran (see `build_capabilities`).
    pub fn with_processor(
        config: WorkerConfig,
        outbox_path: &Path,
        canary_authorized: bool,
        agent: Option<ureq::Agent>,
        processor: Arc<dyn TaskProcessor>,
    ) -> Result<Self, WorkerRuntimeError> {
        Self::build(
            config,
            outbox_path,
            canary_authorized,
            agent,
            Some(processor),
        )
    }

    fn build(
        config: WorkerConfig,
        outbox_path: &Path,
        canary_authorized: bool,
        agent: Option<ureq::Agent>,
        override_processor: Option<Arc<dyn TaskProcessor>>,
    ) -> Result<Self, WorkerRuntimeError> {
        // Swift: `Set(config.taskPools ?? ["primary"])` — an explicit empty
        // array in config (`Some(vec![])`) is intentionally left empty, not
        // defaulted, matching `?? ["primary"]` only firing on `nil`.
        let configured_pools: Vec<String> = config
            .task_pools
            .clone()
            .unwrap_or_else(|| vec!["primary".to_string()]);
        let allowed: HashSet<&str> = ["primary", "fpp_backfill", "exominer_backfill"]
            .into_iter()
            .collect();
        let mut unsupported: Vec<String> = configured_pools
            .iter()
            .filter(|p| !allowed.contains(p.as_str()))
            .cloned()
            .collect();
        unsupported.sort();
        unsupported.dedup();
        if !unsupported.is_empty() {
            return Err(WorkerRuntimeError::UnsupportedTaskPools(unsupported));
        }
        // F8: the native FPP port makes `fpp_backfill` runnable, but only
        // behind an explicit canary flag — without `EXONIT_RUST_FPP=1` the
        // historical refusal stands, so nothing changes for today's
        // sidecar-based production manifests. Parsing mirrors
        // `canary_authorized()` (trim + "1"/"true"): farmctl's Windows env
        // prefix (`set K=V && ...`) leaves a TRAILING SPACE in the value
        // (runbook <host>, gotcha de quoting cmd.exe) — a strict `== "1"`
        // made all 10 <host> FPP instances exit_immediately on 2026-08-17.
        let native_fpp_enabled = std::env::var("EXONIT_RUST_FPP")
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true"))
            .unwrap_or(false);
        let mut unimplemented_backfill: Vec<String> = configured_pools
            .iter()
            .filter(|p| {
                (p.as_str() == "fpp_backfill" && !native_fpp_enabled)
                    || p.as_str() == "exominer_backfill"
            })
            .cloned()
            .collect();
        unimplemented_backfill.sort();
        unimplemented_backfill.dedup();
        if !unimplemented_backfill.is_empty() {
            return Err(WorkerRuntimeError::BackfillPoolsUnimplemented(
                unimplemented_backfill,
            ));
        }
        let pools: HashSet<String> = configured_pools.into_iter().collect();

        let api = Arc::new(match agent {
            Some(agent) => WorkerApiClient::with_agent_and_timeouts(
                config.server_url.clone(),
                config.token.clone(),
                agent,
                Duration::from_secs_f64(config.http_get_timeout_seconds),
                Duration::from_secs_f64(config.http_post_timeout_seconds),
            ),
            None => WorkerApiClient::with_timeouts(
                config.server_url.clone(),
                config.token.clone(),
                Duration::from_secs_f64(config.http_get_timeout_seconds),
                Duration::from_secs_f64(config.http_post_timeout_seconds),
            ),
        });
        let durable_outbox = Arc::new(DurableOutbox::new(outbox_path)?);
        // F8 (2026-08-05, `docs/PLAN.md`): reconcile any legacy
        // `delivery_state = 'unknown'` entries (written by a pre-fix worker
        // binary, or an outbox `WorkerRuntime` never even started `run()`
        // on) back to 'pending' before this runtime does anything else, so
        // startup never trips over stale ambiguous entries. See
        // `DurableOutbox::reconcile_legacy_unknown`'s doc comment for why
        // this is safe.
        let reconciled_legacy_unknown =
            durable_outbox.reconcile_legacy_unknown(crate::outbox::now_unix())?;
        if reconciled_legacy_unknown > 0 {
            eprintln!(
                "WorkerRuntime: reconciled {reconciled_legacy_unknown} legacy outbox entr{} from delivery_state='unknown' to 'pending' on startup (server result ingestion is idempotent, safe to retry — F8, 2026-08-05, docs/PLAN.md)",
                if reconciled_legacy_unknown == 1 { "y" } else { "ies" }
            );
        }
        let result_outbox = WorkerResultOutbox::new(durable_outbox.clone(), api.clone());

        let (
            processor,
            bls_backend_name,
            injection_backend_name,
            cnn_available,
            exominer_available,
            bls_gpu_used,
            wotan_gpu_used,
        ): (
            Arc<dyn TaskProcessor>,
            &'static str,
            &'static str,
            bool,
            bool,
            bool,
            bool,
        ) = match override_processor {
            // No `FitsCache` is built on this path: a caller supplying its
            // own `TaskProcessor` (tests, `exonit-fake-integration`) owns
            // whatever FITS cache that processor needs itself — building a
            // second, unused one here would just be dead work.
            Some(processor) => (
                processor,
                "custom-processor",
                "custom-processor",
                false,
                false,
                false,
                false,
            ),
            None => {
                let fits_cache_dir = Path::new(&config.fits_cache_directory).join("fits");
                let fits_cache = Arc::new(FitsCache::new(fits_cache_dir).map_err(|e| {
                    WorkerRuntimeError::Outbox(format!("creating FITS cache directory: {e}"))
                })?);
                let max_concurrency: usize = match config.cpu_threads {
                    Some(threads) if threads > 0 => threads as usize,
                    _ => available_parallelism().max(1) as usize,
                };
                let models_dir = config.models_dir.as_ref().map(PathBuf::from);
                let primary = PrimaryTaskProcessor::new(
                    api.clone(),
                    fits_cache,
                    models_dir,
                    max_concurrency,
                    config.gpu_enabled,
                );
                let cnn_available = primary.cnn_available();
                let exominer_available = primary.exominer_available();
                // F4a: honest now — `gpu_used()` reflects whether
                // `resolve_compute_backend` actually resolved CUDA (probed
                // once at construction), not just `config.gpu_enabled`.
                let bls_gpu_used = primary.gpu_used();
                let bls_backend_name = if bls_gpu_used {
                    "rust-cuda"
                } else {
                    "rust-cpu"
                };
                // F4b: same honesty extended to injection-recovery —
                // previously hardcoded `"rust-cpu"` regardless of what
                // `resolve_injection_backend` actually resolved. F-opencl:
                // now a 3-way label (`"rust-cuda"`/`"rust-opencl"`/
                // `"rust-cpu"`) via `injection_backend_label()`, replacing
                // the old 2-way `injection_gpu_used()`-derived ternary
                // (which had no way to name OpenCL specifically).
                // 2026-08-06 Spec 2: now 4-way (`"rust-cuda"`/`"rust-sycl"`/
                // `"rust-opencl"`/`"rust-cpu"`) — `injection_backend_label()`
                // itself is unchanged here, this comment just tracks the
                // set of strings `resolve_injection_backend` can now return.
                let injection_backend_name = primary.injection_backend_label();
                let wotan_gpu_used = primary.detrend_gpu_used();
                (
                    Arc::new(primary),
                    bls_backend_name,
                    injection_backend_name,
                    cnn_available,
                    exominer_available,
                    bls_gpu_used,
                    wotan_gpu_used,
                )
            }
        };
        let capabilities = build_capabilities(
            &config,
            &pools,
            bls_backend_name,
            injection_backend_name,
            cnn_available,
            exominer_available,
            bls_gpu_used,
            wotan_gpu_used,
        );

        let shared = Arc::new(RuntimeShared {
            config,
            api,
            state: Arc::new(WorkerRuntimeState::new()),
            durable_outbox,
            result_outbox,
            processor,
            pools,
            capabilities,
            started: Instant::now(),
        });
        Ok(WorkerRuntime {
            shared,
            canary_authorized,
        })
    }

    /// Mirrors the Swift `public let state` property.
    pub fn state(&self) -> &Arc<WorkerRuntimeState> {
        &self.shared.state
    }

    /// Mirrors `WorkerRuntime.run()`. Blocks the calling thread until
    /// shutdown is requested (via `state().request_shutdown()`, e.g. from a
    /// signal handler) or an unrecoverable error occurs.
    pub fn run(&self) -> Result<(), WorkerRuntimeError> {
        if !self.canary_authorized {
            return Err(WorkerRuntimeError::CanaryNotAuthorized);
        }
        register(&self.shared)?;

        let heartbeat_shared = Arc::clone(&self.shared);
        let heartbeat_handle = std::thread::spawn(move || heartbeat_loop(heartbeat_shared));

        let result = main_loop(&self.shared);

        // Mirrors `defer { heartbeat.cancel() }`: whatever ended the main
        // loop (clean shutdown or an error return), stop the heartbeat
        // thread too rather than leaking it.
        self.shared.state.request_shutdown();
        let _ = heartbeat_handle.join();
        result
    }
}

/// F2b: `bls_backend`/`injection_backend` and `cnn_ffi_available`/
/// `exominer_ffi_available` now reflect what `WorkerRuntime::build` actually
/// constructed (real `PrimaryTaskProcessor` with real BLS, vs. an injected
/// override processor of unknown capability — see the two call sites in
/// `build`), rather than F0b's fixed `"none-f0b-stub"` placeholder.
///
/// F4a update: `gpu_available` now reflects `bls_gpu_used` (whether
/// `PrimaryTaskProcessor::resolve_compute_backend` actually resolved CUDA),
/// instead of the previous hardcoded `false` — that hardcoding is now
/// stale (a CUDA backend exists in this crate as of F4a). Injection/FPP/
/// ONNX GPU backends remain unimplemented (F4b+), so `gpu_available` here
/// specifically means "BLS primary search is on GPU", same granularity as
/// `bls_backend`.
fn build_capabilities(
    config: &WorkerConfig,
    pools: &HashSet<String>,
    bls_backend: &str,
    injection_backend: &str,
    cnn_ffi_available: bool,
    exominer_ffi_available: bool,
    bls_gpu_used: bool,
    wotan_gpu_used: bool,
) -> WorkerCapabilities {
    let cores = config
        .cpu_threads
        .unwrap_or_else(|| available_parallelism());
    let fpp_cores = config.fpp_cpu_cores.unwrap_or(cores);
    let mut sorted_pools: Vec<String> = pools.iter().cloned().collect();
    sorted_pools.sort();
    let mut hardware = Map::new();
    hardware.insert(
        "runtime".to_string(),
        Value::String("rust-native".to_string()),
    );
    hardware.insert(
        "task_pools".to_string(),
        Value::Array(sorted_pools.into_iter().map(Value::String).collect()),
    );
    hardware.insert(
        "gpu_configured".to_string(),
        Value::Bool(config.gpu_enabled),
    );
    let gpu = gpu_capability(bls_backend, injection_backend, wotan_gpu_used);
    // Report both aggregate acceleration and the BLS-specific fact.  The
    // injection and detrend backends can be GPU-backed even when BLS fell
    // back to CPU, so using `bls_gpu_used` as the aggregate was misleading.
    hardware.insert("gpu_available".to_string(), Value::Bool(gpu != "none"));
    hardware.insert("bls_gpu_available".to_string(), Value::Bool(bls_gpu_used));
    hardware.insert(
        "bls_backend".to_string(),
        Value::String(bls_backend.to_string()),
    );
    hardware.insert(
        "injection_backend".to_string(),
        Value::String(injection_backend.to_string()),
    );
    // `fpp_backend`/`exominer_backend` here name the FPP/ExoMiner *backfill
    // pool* sidecar (Swift's Python science sidecar), unimplemented in this
    // crate — see `WorkerRuntimeError::BackfillPoolsUnimplemented`. Distinct
    // from `cnn_ffi_available`/`exominer_ffi_available` below, which report
    // the *primary*-pool ONNX FFI classifiers `PrimaryTaskProcessor` may or
    // may not have loaded.
    hardware.insert(
        "fpp_backend".to_string(),
        Value::String("disabled".to_string()),
    );
    hardware.insert(
        "exominer_backend".to_string(),
        Value::String("disabled".to_string()),
    );
    hardware.insert(
        "cnn_ffi_available".to_string(),
        Value::Bool(cnn_ffi_available),
    );
    hardware.insert(
        "exominer_ffi_available".to_string(),
        Value::Bool(exominer_ffi_available),
    );
    // F4b: honest wotan detrend backend name — gated independently of
    // `gpu_configured`/`gpu_available` above (see
    // `task_processor.rs`'s `resolve_detrend_backend` doc comment: this
    // backend follows `EXONIT_WOTAN_BACKEND`, not `config.gpu_enabled`).
    hardware.insert(
        "wotan_backend".to_string(),
        Value::String(if wotan_gpu_used {
            "cuda".to_string()
        } else {
            "cpu".to_string()
        }),
    );
    WorkerCapabilities {
        gpu,
        bls_backend: bls_backend.to_string(),
        bls_gpu_info: String::new(),
        fpp_backend: "disabled".to_string(),
        cpu_cores: cores,
        fpp_cpu_cores: fpp_cores,
        hardware,
    }
}

fn gpu_capability(bls_backend: &str, injection_backend: &str, wotan_gpu_used: bool) -> String {
    let mut backends = Vec::new();
    if bls_backend == "rust-cuda" {
        backends.push("cuda");
    }
    if let Some(backend) = injection_backend.strip_prefix("rust-") {
        if backend != "cpu" {
            backends.push(backend);
        }
    }
    if wotan_gpu_used {
        backends.push("cuda");
    }
    backends.sort_unstable();
    backends.dedup();
    if backends.is_empty() {
        "none".to_string()
    } else {
        backends.join("+")
    }
}

fn available_parallelism() -> i64 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i64)
        .unwrap_or(1)
}

/// Mirrors `WorkerRuntime.register()`, called both from `run()` before the
/// main loop starts and from the heartbeat loop on a `re-register` command.
fn register(shared: &RuntimeShared) -> Result<(), WorkerRuntimeError> {
    let request = RegisterRequest {
        name: shared.config.worker_name.clone(),
        capabilities: shared.capabilities.clone(),
    };
    let worker_id = shared.api.register(&request).map_err(api_error)?;
    shared.state.registered(worker_id);
    Ok(())
}

/// Re-registration is a control-plane recovery operation, not a best-effort
/// hint.  Keep retrying with bounded exponential backoff until the server
/// accepts the worker or shutdown is requested, while preserving the error in
/// heartbeat state so operators can see why recovery is taking time.
fn register_with_backoff(shared: &RuntimeShared) {
    shared.state.set_re_registering(true);
    let mut delay = Duration::from_secs(1);
    loop {
        if shared.state.shutdown_requested() {
            shared.state.set_re_registering(false);
            return;
        }
        match register(shared) {
            Ok(()) => {
                shared.state.set_re_registering(false);
                return;
            }
            Err(error) => {
                let message = format!("re-register: {error}");
                shared.state.heartbeat_failed(message.clone());
                eprintln!("WorkerRuntime: {message}; retrying in {}s", delay.as_secs());
                interruptible_sleep(delay, &shared.state);
                delay = delay.saturating_mul(2).min(Duration::from_secs(60));
            }
        }
    }
}

/// Mirrors `WorkerRuntime.run()`'s `while` loop body.
///
/// F8 (2026-08-05, `docs/PLAN.md`): this used to open with a fatal check —
/// `if delivery_unknown_count() > 0 { return Err(AmbiguousDelivery(..)) }` —
/// that killed the worker outright whenever the outbox held any ambiguous
/// entry, both on the very first pass through this loop after startup and
/// on every subsequent iteration. That check (and the matching one in
/// `process_one`/`require_unambiguous_drain`, now folded into `drain`
/// itself) is gone: `WorkerResultOutbox::drain` no longer produces
/// `'unknown'` entries at all (see its doc comment — transport failures are
/// now a normal `mark_failed` retry), and any inherited legacy 'unknown'
/// entry is already reconciled to 'pending' by `WorkerRuntime::build` before
/// `run()` is ever called. The "don't poll for new work while the outbox is
/// non-empty" guard below (`durable_outbox.count()? > 0`) is unchanged and
/// still guarantees a pending retry (of any kind) is drained before any new
/// task is picked up, preserving delivery order.
fn main_loop(shared: &Arc<RuntimeShared>) -> Result<(), WorkerRuntimeError> {
    loop {
        if shared.state.shutdown_requested() {
            return Ok(());
        }
        let drain = shared.result_outbox.drain(crate::outbox::now_unix(), 32);
        if let Some(scan_error) = &drain.scan_error {
            shared.state.failed(scan_error.clone());
        }
        if shared.durable_outbox.count()? > 0 {
            interruptible_sleep(poll_interval(shared), &shared.state);
            continue;
        }
        if shared.state.paused() || shared.state.re_registering() {
            interruptible_sleep(poll_interval(shared), &shared.state);
            continue;
        }
        let worker_id = shared.state.worker_id();
        match process_one(shared, &worker_id) {
            Ok(ProcessOutcome::DidWork) => {}
            Ok(ProcessOutcome::Wait { seconds }) => {
                interruptible_sleep(wait_interval(shared, seconds), &shared.state);
            }
            Err(other) => {
                shared.state.failed(other.to_string());
                interruptible_sleep(poll_interval(shared), &shared.state);
            }
        }
    }
}

fn poll_interval(shared: &RuntimeShared) -> Duration {
    Duration::from_secs_f64(shared.config.poll_interval_seconds.max(0.0))
}

fn wait_interval(shared: &RuntimeShared, seconds: Option<f64>) -> Duration {
    let Some(seconds) = seconds.filter(|seconds| seconds.is_finite() && *seconds >= 0.0) else {
        return poll_interval(shared);
    };
    // `wait_seconds` is server input; cap it before converting to Duration so
    // an accidental/hostile huge JSON number cannot panic the worker.
    Duration::from_secs_f64(seconds.min(24.0 * 60.0 * 60.0))
}

enum ProcessOutcome {
    DidWork,
    Wait { seconds: Option<f64> },
}

/// Mirrors `WorkerRuntime.processOne(workerID:)`. Only the `primary` pool is
/// handled — `fpp_backfill`/`exominer_backfill` cannot reach this point
/// (`WorkerRuntime::new` refuses to construct with them configured; see
/// `WorkerRuntimeError::BackfillPoolsUnimplemented`). F5 adds their
/// branches here once real backfill processors exist.
fn process_one(
    shared: &Arc<RuntimeShared>,
    worker_id: &str,
) -> Result<ProcessOutcome, WorkerRuntimeError> {
    if !shared.pools.contains("primary") {
        // F8: an FPP-only worker (pools = ["fpp_backfill"], canary flag on)
        // polls the FPP queue directly.
        if shared.pools.contains("fpp_backfill") {
            return process_one_fpp(shared, worker_id);
        }
        return Ok(ProcessOutcome::Wait { seconds: None });
    }
    let poll = shared.api.poll_task(worker_id).map_err(api_error)?;
    let task = match poll {
        TaskPollResult::Task(task) => task,
        TaskPollResult::Wait { seconds } => {
            // F8: primary idle → drain the FPP-backfill pool, mirroring the
            // Python worker's primary-first ordering.
            if shared.pools.contains("fpp_backfill") {
                return process_one_fpp_or_wait(shared, worker_id, seconds);
            }
            return Ok(ProcessOutcome::Wait { seconds });
        }
    };
    shared
        .state
        .begin(task.tic_id.clone(), "primary".to_string());
    let result = shared.processor.process(&task, worker_id);
    shared
        .result_outbox
        .enqueue_primary(&result, worker_id, crate::outbox::now_unix())
        .map_err(|e| WorkerRuntimeError::Outbox(e.to_string()))?;
    drain_after_enqueue(shared)?;
    if result.status == "completed" {
        shared.state.completed();
    } else {
        shared.state.failed(
            result
                .error
                .clone()
                .unwrap_or_else(|| "primary task failed".to_string()),
        );
    }
    Ok(ProcessOutcome::DidWork)
}

/// F8 native FPP: poll the FPP-backfill queue once and process the task if
/// any; `Wait` (with the primary poll's suggested delay, when called from
/// the primary-idle path) otherwise.
fn process_one_fpp_or_wait(
    shared: &Arc<RuntimeShared>,
    worker_id: &str,
    primary_wait: Option<f64>,
) -> Result<ProcessOutcome, WorkerRuntimeError> {
    match process_one_fpp(shared, worker_id)? {
        ProcessOutcome::DidWork => Ok(ProcessOutcome::DidWork),
        ProcessOutcome::Wait { .. } => Ok(ProcessOutcome::Wait { seconds: primary_wait }),
    }
}

fn process_one_fpp(
    shared: &Arc<RuntimeShared>,
    worker_id: &str,
) -> Result<ProcessOutcome, WorkerRuntimeError> {
    let Some((task, raw_task)) = shared.api.get_fpp_task_raw(worker_id).map_err(api_error)?
    else {
        return Ok(ProcessOutcome::Wait { seconds: None });
    };
    shared
        .state
        .begin(task.tic_id.clone(), "fpp_backfill".to_string());
    let Some(outcome) = shared.processor.process_fpp(&task, &raw_task, worker_id) else {
        // Processor without FPP support but pool configured: report the
        // reservation failed so the server frees it (never strand a claim).
        let mut req = crate::dto::FppResultRequest::new(
            Some(task.candidate_id),
            Some(task.tic_id.clone()),
            Some(-1.0),
        );
        req.nfpp = Some(-1.0);
        req.verdict = Some("failed".to_string());
        shared
            .result_outbox
            .enqueue_fpp(&req, worker_id, crate::outbox::now_unix())
            .map_err(|e| WorkerRuntimeError::Outbox(e.to_string()))?;
        drain_after_enqueue(shared)?;
        shared
            .state
            .failed("fpp_backfill configured but processor has no FPP support".to_string());
        return Ok(ProcessOutcome::DidWork);
    };
    shared
        .result_outbox
        .enqueue_fpp(&outcome.request, worker_id, crate::outbox::now_unix())
        .map_err(|e| WorkerRuntimeError::Outbox(e.to_string()))?;
    drain_after_enqueue(shared)?;
    match outcome.error {
        None => shared.state.completed(),
        Some(message) => shared.state.failed(format!("FPP TIC {}: {message}", task.tic_id)),
    }
    Ok(ProcessOutcome::DidWork)
}

/// F8 (2026-08-05): formerly `require_unambiguous_drain`, which treated any
/// `delivery_unknown` count from this drain as fatal. `drain` can no longer
/// produce that outcome (see its doc comment), so this only needs to attempt
/// an immediate best-effort delivery of the result just enqueued (so a
/// healthy connection delivers it without waiting for the next poll cycle)
/// and surface genuine local `OutboxError`s (SQLite failures) — remote
/// delivery outcomes (success, HTTP error, or transport error) are already
/// handled by `drain` itself, and any entry still pending afterwards is
/// picked up by `main_loop`'s "don't poll while the outbox is non-empty"
/// guard before any new task is taken.
fn drain_after_enqueue(shared: &RuntimeShared) -> Result<(), WorkerRuntimeError> {
    let drain = shared.result_outbox.drain(crate::outbox::now_unix(), 32);
    if let Some(scan_error) = drain.scan_error {
        return Err(WorkerRuntimeError::Outbox(scan_error));
    }
    Ok(())
}

/// Mirrors `WorkerRuntime.heartbeatLoop()`.
fn heartbeat_loop(shared: Arc<RuntimeShared>) {
    let mut telemetry = Telemetry::new();
    loop {
        if shared.state.shutdown_requested() {
            return;
        }
        interruptible_sleep(
            Duration::from_secs_f64(shared.config.heartbeat_interval_seconds.max(0.0)),
            &shared.state,
        );
        if shared.state.shutdown_requested() {
            return;
        }
        let worker_id = shared.state.worker_id();
        if worker_id.is_empty() {
            continue;
        }
        let current_tic = shared.state.current_tic();
        let status = if shared.state.paused() {
            "paused"
        } else if current_tic.is_none() {
            "idle"
        } else {
            "busy"
        };
        let payload = HeartbeatRequest {
            worker_id: worker_id.clone(),
            status: status.to_string(),
            current_tic,
            phase: shared.state.phase(),
            phase_progress: None,
            star_elapsed_seconds: None,
            stars_done: shared.state.stars_done(),
            average_time_per_star: 0.0,
            last_error: shared.state.last_error(),
            uptime_seconds: shared.started.elapsed().as_secs_f64(),
            pid: std::process::id() as i32,
            git_sha: None,
            flag_states: Map::new(),
            gpu_fallback: Map::new(),
            system: telemetry.sample().to_json_map(),
            gpu_telemetry: Map::new(),
            recent_timings: Vec::new(),
        };
        match shared.api.heartbeat(&payload) {
            Ok(HeartbeatCommand::ReRegister) => register_with_backoff(&shared),
            Ok(command) => shared.state.apply(command),
            Err(error) => shared.state.heartbeat_failed(format!("heartbeat: {error}")),
        }
    }
}

/// Sleeps `total`, checking `state.shutdown_requested()` every 100ms so a
/// shutdown request lands promptly instead of after up to a full
/// poll/heartbeat interval — the closest blocking-thread analogue of Swift
/// `Task.sleep` raising `CancellationError` as soon as the task is
/// cancelled.
fn interruptible_sleep(total: Duration, state: &WorkerRuntimeState) {
    const STEP: Duration = Duration::from_millis(100);
    let mut remaining = total;
    while remaining > Duration::ZERO {
        if state.shutdown_requested() {
            return;
        }
        let chunk = remaining.min(STEP);
        std::thread::sleep(chunk);
        remaining = remaining.saturating_sub(chunk);
    }
}

fn api_error(error: WorkerApiError) -> WorkerRuntimeError {
    WorkerRuntimeError::Api(error.to_string())
}
