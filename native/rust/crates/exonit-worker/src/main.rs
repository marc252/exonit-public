//! Entry point — real CLI parsing (see `exonit_core::cli`, transliterated
//! from `Invocation.swift`/`CLIHelp.swift`), wired to the two invocation
//! shapes: farmctl-compatible `agent.py start|stop|status|logs|version`
//! (fully functional since F0 — `exonit_core::pidlock::AgentController`
//! handles spawn/PID-lock/stale-PID detection/`--allow-start`) and direct
//! worker invocation (`--config PATH [--worker-name NAME] [--server-url URL]`).
//!
//! **Direct worker execution (F2b canary gate).** The full science pipeline
//! (FITS -> BLS -> vetting -> CNN/ExoMiner FFI -> injection-recovery, see
//! `exonit_core::task_processor::PrimaryTaskProcessor`) is wired up as of
//! F2b, but F4 (CUDA) and the F6 production gate (parity over batches of >=300
//! TICs vs fresh Python, A/B benchmark, real canary with operator permission — see
//! docs/PLAN.md) are NOT done. So direct worker execution stays refused
//! (`exit 78`, "fases incompletes") UNLESS the caller explicitly opts in
//! with `EXONIT_RUST_CANARY=1` (mirrors the Swift reference worker's
//! `EXONIT_SWIFT_CANARY` gate in `Sources/ExoNitSwiftWorker/main.swift`) —
//! a deliberate manual canary switch for controlled testing, NOT a flag
//! `agent.py start`/farmctl set on its own (`pidlock.rs`'s `start_locked`
//! does not inject it into the spawned child's environment). With the
//! canary authorized, this wires: config load + `--worker-name`/`--server-url`
//! overrides -> `fits_cache_dir`/`models_dir` resolved to absolute paths ->
//! log file redirect (`exonit_core::logfile`) -> `WorkerRuntime::new` (real
//! `PrimaryTaskProcessor`) -> shutdown source installed via
//! `exonit_core::platform::install_shutdown_watch` (Unix: a raw
//! `libc::signal` handler, async-signal-safe — only flips an `AtomicBool`;
//! extracted in `docs/plans/2026-08-06-win32-port-pcmarc.md` Task 1) polled
//! by a watcher thread that calls `WorkerRuntimeState::request_shutdown()`
//! — the same "finish the in-flight task, stop polling for new work, exit
//! clean" shutdown shape `WorkerRuntime::run()`'s main loop already
//! implements (see `runtime.rs`; Swift's own `Task.isCancelled` check is
//! only evaluated between tasks too, not mid-BLS, so this is a faithful
//! mirror, not a simplification). The worker's own PID lock is already
//! handled end-to-end by `AgentController` (`pidlock.rs`) when it spawns
//! this process — nothing additional is needed here.

use exonit_core::cli::{self, Invocation};
use exonit_core::config::WorkerConfig;
use exonit_core::pidlock::AgentController;
use exonit_core::runtime::WorkerRuntime;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

fn main() {
    let raw_arguments: Vec<String> = std::env::args().skip(1).collect();
    if raw_arguments.iter().any(|a| a == "--help" || a == "-h") {
        print!("{}", cli::help_text(&raw_arguments));
        std::process::exit(0);
    }

    match cli::parse(&raw_arguments) {
        Ok(Invocation::Agent(arguments)) => run_agent(&raw_arguments, arguments),
        Ok(Invocation::Worker(arguments)) => run_worker(arguments),
        Err(error) => {
            eprintln!("exonit-rust-worker: {error}");
            eprintln!();
            eprint!("{}", cli::help_text(&raw_arguments));
            std::process::exit(1);
        }
    }
}

/// True when `EXONIT_RUST_CANARY` is `1` or `true` (case-insensitive,
/// trimmed) — mirrors Swift's `EXONIT_SWIFT_CANARY` parsing exactly
/// (`main.swift`: `canaryValue == "1" || canaryValue == "true"` after
/// `trimmingCharacters(in: .whitespacesAndNewlines).lowercased()`).
fn canary_authorized() -> bool {
    match std::env::var("EXONIT_RUST_CANARY") {
        Ok(raw) => matches!(raw.trim().to_ascii_lowercase().as_str(), "1" | "true"),
        Err(_) => false,
    }
}

fn fail_worker(message: &str) -> ! {
    eprintln!("exonit-rust-worker: {message}");
    std::process::exit(1);
}

fn run_worker(arguments: cli::WorkerArguments) -> ! {
    // F-opencl (docs/plans/2026-08-06-opencl-backend-<host>.md, Task 4
    // Step 3): mirrors worker.py:46-63's `_boot_enable_rusticl` import-time
    // placement — must run before ANY OpenCL platform/device enumeration
    // (`exonit_core::bls::opencl_batch::OpenClBlsBatchBackend::probe`,
    // reached indirectly via `WorkerRuntime::new` -> `PrimaryTaskProcessor::new`
    // below) or rusticl (Mesa's OpenCL ICD, <host>'s AMD Vega path)
    // silently exposes zero devices. `probe()` itself also calls this as
    // defense in depth (see that function's doc comment) — this call site
    // is belt-and-suspenders, not load-bearing on its own, since whichever
    // of the two runs first wins (`set_var` only if unset). Inert without
    // the `opencl` Cargo feature (function does not exist in that build) and
    // inert on <host> (Intel's ICD, unaffected by this var).
    #[cfg(feature = "opencl")]
    exonit_core::bls::opencl_batch::boot_enable_rusticl();

    if !canary_authorized() {
        // Production gate: F4 (CUDA) and the F6 deploy gate are not done —
        // see this module's doc comment. Same message/exit code as before
        // F2b wired the real pipeline up.
        eprintln!("exonit-rust-worker: fases incompletes (vegeu docs/PLAN.md)");
        std::process::exit(78);
    }

    let config_path = PathBuf::from(&arguments.config_path);
    let config_path = std::fs::canonicalize(&config_path).unwrap_or(config_path);
    let base = config_path.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));

    let data = std::fs::read(&config_path)
        .unwrap_or_else(|error| fail_worker(&format!("reading config {}: {error}", config_path.display())));
    let decoded: WorkerConfig = serde_json::from_slice(&data)
        .unwrap_or_else(|error| fail_worker(&format!("parsing config {}: {error}", config_path.display())));
    let mut config = decoded
        .applying(arguments.worker_name.as_deref(), arguments.server_url.as_deref())
        .unwrap_or_else(|error| fail_worker(&format!("config: {error}")));

    if !Path::new(&config.fits_cache_directory).is_absolute() {
        config.fits_cache_directory = base.join(&config.fits_cache_directory).to_string_lossy().into_owned();
    }
    config.models_dir = resolve_models_dir(&config, &base).map(|p| p.to_string_lossy().into_owned());

    if let Some(log_path) = exonit_core::logfile::resolve(config.log_file.as_deref(), &base) {
        let timestamp = unix_timestamp_label();
        let start_message = format!("{timestamp} starting {}", config.worker_name);
        if let Err(error) = exonit_core::logfile::append(&start_message, &log_path) {
            fail_worker(&format!("log append {}: {error}", log_path.display()));
        }
        if let Err(error) = exonit_core::logfile::redirect_standard_streams(&log_path) {
            fail_worker(&format!("log redirect {}: {error}", log_path.display()));
        }
    }

    let state_directory = base.join(".rust-worker-state");
    let outbox_path = state_directory.join(format!("outbox-{}.sqlite", config.worker_name));

    // Installed before `config` is moved into `WorkerRuntime::new` below
    // (`install_shutdown_watch` needs `config.worker_name`/`base` — the
    // Windows implementation names its stop-file from them; Unix ignores
    // both and just installs signal handlers, see `platform::unix`).
    let shutdown_flag = exonit_core::platform::install_shutdown_watch(&config.worker_name, &base);

    // Canary already authorized above; pass `true` straight through
    // (`WorkerRuntime::run()`'s own `canary_authorized` gate exists for
    // callers — tests, `exonit-fake-integration` — that construct a runtime
    // directly without going through this CLI gate).
    let runtime = WorkerRuntime::new(config, &outbox_path, true, None)
        .unwrap_or_else(|error| fail_worker(&format!("constructing runtime: {error}")));

    let watcher_state = runtime.state().clone();
    std::thread::spawn(move || {
        loop {
            // `shutdown_requested_extra_check()` is a no-op (`false`) on
            // Unix and polls the Windows stop-file on Windows — see
            // `exonit_core::platform`'s doc comments (Task 2 of
            // docs/plans/2026-08-06-win32-port-pcmarc.md). Same watcher
            // loop structure on both platforms either way.
            if shutdown_flag.load(Ordering::SeqCst) || exonit_core::platform::shutdown_requested_extra_check() {
                watcher_state.request_shutdown();
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    });

    match runtime.run() {
        Ok(()) => std::process::exit(0),
        Err(error) => {
            eprintln!("exonit-rust-worker: {error}");
            std::process::exit(1);
        }
    }
}

/// Resolution order for the ONNX models directory: `EXONIT_MODELS_DIR` (env,
/// already the convention every F3 test/E2E harness in this crate uses) >
/// `config.models_dir` (resolved relative to the config file's directory,
/// same as `fits_cache_dir`) > `<executable_dir>/models` (mirrors Swift's
/// `executableDirectory.appendingPathComponent("Models")` default, but see
/// `PrimaryTaskProcessor`'s doc comment for why an absent directory here
/// degrades gracefully instead of refusing to start like Swift's CLI does).
fn resolve_models_dir(config: &WorkerConfig, base: &Path) -> Option<PathBuf> {
    if let Ok(value) = std::env::var("EXONIT_MODELS_DIR") {
        if !value.trim().is_empty() {
            return Some(PathBuf::from(value));
        }
    }
    if let Some(configured) = config.models_dir.as_deref() {
        if !configured.trim().is_empty() {
            let path = Path::new(configured);
            return Some(if path.is_absolute() { path.to_path_buf() } else { base.join(path) });
        }
    }
    let default_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("models")))
        .unwrap_or_else(|| base.join("models"));
    Some(default_dir)
}

fn unix_timestamp_label() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("[unix:{seconds}]")
}

fn run_agent(raw_arguments: &[String], arguments: cli::AgentArguments) -> ! {
    // `InvocationParser` only recognises the agent.py shape when argv[0]'s
    // basename is literally "agent.py" (see `cli::parse`), so this is always
    // present here — mirrors the Swift entry point's
    // `guard let agentPath = rawArguments.first`.
    let agent_path = raw_arguments
        .first()
        .expect("agent invocation implies argv[0] is agent.py-shaped");
    let worker_directory = std::path::Path::new(agent_path)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let executable =
        std::env::current_exe().unwrap_or_else(|_| PathBuf::from(std::env::args().next().unwrap_or_default()));

    let controller = AgentController::new(worker_directory, executable);
    match controller.execute(&arguments) {
        Ok(result) => {
            std::io::stdout().write_all(&result.output).ok();
            std::process::exit(result.exit_code);
        }
        Err(error) => {
            eprintln!("exonit-rust-worker: {error}");
            std::process::exit(2);
        }
    }
}
