//! Ported from the `AgentController` (PID-lock / stale-PID / spawn-guard)
//! tests in `Sources/ExoNitContractTests/main.swift` (Swift reference
//! worker). `testWorkerLogFile` is not ported: it exercises
//! `WorkerLogFile.swift`, a distinct small module not in this F0 task's
//! Swift source list (`WorkerConfig.swift`, `AgentConfigDeriver.swift`,
//! `WorkerProtocolDTO.swift`, `Invocation.swift`+`CLIHelp.swift`,
//! `AgentController.swift`+`WorkerRuntime.swift`'s PID-lock portion) — it
//! belongs with the `--config log_file` sink wiring in F0b's worker
//! bootstrap, not the agent control plane.

#[path = "support/fake_python_agent.rs"]
mod fake_python_agent;

use exonit_core::cli::{AgentAction, AgentArguments};
use exonit_core::pidlock::AgentController;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// RAII temp directory, mirroring the Swift suite's
/// `withTemporaryDirectory`/manual-`defer removeItem` helpers.
struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!("exonit-rust-tests-{label}-{nanos}-{n}"));
        std::fs::create_dir_all(&path).expect("create temp dir");
        TempDir(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn agent_args(action: AgentAction, worker_name: &str) -> AgentArguments {
    let mut args = AgentArguments::new(action);
    args.worker_name = Some(worker_name.to_string());
    args
}

fn decode(output: &[u8]) -> Value {
    serde_json::from_slice(output).expect("agent output is valid JSON")
}

/// Spawns a real, portable "long-lived worker process" stand-in
/// (`src/bin/test_sleep_helper.rs`, sleeps 30s ignoring all arguments) — the
/// cross-platform replacement for the old `Command::new("/bin/sleep")
/// .arg("30")` (Task 2 of docs/plans/2026-08-06-win32-port-pcmarc.md:
/// `/bin/sleep` doesn't exist on Windows). Tests that need a genuinely alive
/// PID to feed through `AgentController`'s stale-PID/identity checks use
/// this; the *mocked* identity strings (`.with_process_executable_path`/
/// `.with_process_command_line`) they inject are separate, purely symbolic
/// path literals never touched by the OS, so those stay unchanged.
fn spawn_long_lived() -> std::process::Child {
    std::process::Command::new(env!("CARGO_BIN_EXE_test_sleep_helper"))
        .spawn()
        .expect("spawn test_sleep_helper")
}

/// Swift: `testAgentStartGuard`.
#[test]
fn agent_start_guard() {
    let dir = TempDir::new("start-guard");
    let controller = AgentController::with_environment(
        dir.path().to_path_buf(),
        PathBuf::from("/usr/bin/true"),
        HashMap::new(),
    );
    let mut args = agent_args(AgentAction::Start, "worker-test");
    args.allow_start = false;
    let result = controller.execute(&args).expect("execute");
    let object = decode(&result.output);
    assert_eq!(result.exit_code, 2, "refused start exit code");
    assert_eq!(object["status"], "refused", "refused status");
    assert!(
        !dir.path().join(".agent.worker-test.pid").exists(),
        "refused start must have no PID side effect"
    );
}

/// Swift: `testAgentStoppedStatus`.
#[test]
fn agent_stopped_status() {
    let dir = TempDir::new("stopped-status");
    let controller = AgentController::with_environment(
        dir.path().to_path_buf(),
        PathBuf::from("/usr/bin/true"),
        HashMap::new(),
    );
    let result = controller
        .execute(&agent_args(AgentAction::Status, "worker-test"))
        .expect("execute");
    let object = decode(&result.output);
    assert_eq!(result.exit_code, 0, "status exit code");
    assert_eq!(object["status"], "stopped", "stopped status");
    assert!(object["pid"].is_null(), "stopped PID is null");
}

/// Swift: `testAgentStalePIDIdentity`. Spawns a real unrelated process,
/// records its PID as if it belonged to `worker-test`, and injects a
/// process-identity lookup that always reports a *different* executable
/// path than the controller's own — forcing the "foreign" branch. Asserts
/// `stop` treats it as stale (no signal sent) and the unrelated process
/// survives.
#[test]
fn agent_stale_pid_identity() {
    let dir = TempDir::new("stale-pid");
    let mut unrelated = spawn_long_lived();
    let unrelated_pid = unrelated.id() as i32;
    std::fs::write(
        dir.path().join(".agent.worker-test.pid"),
        unrelated_pid.to_string(),
    )
    .expect("write pid file");

    let controller = AgentController::with_environment(
        dir.path().to_path_buf(),
        PathBuf::from("/usr/bin/true"),
        HashMap::new(),
    )
    .with_process_executable_path(Box::new(|_pid| Some(PathBuf::from("/bin/sleep"))));

    let result = controller
        .execute(&agent_args(AgentAction::Stop, "worker-test"))
        .expect("execute");
    let object = decode(&result.output);
    assert_eq!(
        object["status"], "not_running",
        "foreign PID treated as stale"
    );
    assert!(
        unrelated.try_wait().expect("try_wait").is_none(),
        "agent must not signal a foreign process"
    );

    let _ = unrelated.kill();
    let _ = unrelated.wait();
}

/// Swift: `testAgentCommandLineFallbackOwned`. Executable path unreadable
/// but argv identifies the instance (spawn path + exact `--worker-name`):
/// owned, and `stop` actually stops the process.
#[test]
fn agent_command_line_fallback_owned() {
    let dir = TempDir::new("cmdline-owned");
    let mut child = spawn_long_lived();
    let child_pid = child.id() as i32;
    std::fs::write(
        dir.path().join(".agent.worker-test.pid"),
        child_pid.to_string(),
    )
    .expect("write pid file");

    let controller = AgentController::with_environment(
        dir.path().to_path_buf(),
        PathBuf::from("/bin/sleep"),
        HashMap::new(),
    )
    .with_process_executable_path(Box::new(|_pid| None))
    .with_process_command_line(Box::new(|_pid| {
        Some(vec![
            "/bin/sleep".to_string(),
            "--worker-name".to_string(),
            "worker-test".to_string(),
        ])
    }));

    let result = controller
        .execute(&agent_args(AgentAction::Stop, "worker-test"))
        .expect("execute");
    let object = decode(&result.output);
    assert_eq!(
        object["status"], "stopped",
        "fallback identity must allow stop"
    );
    let _ = child.wait();
    assert!(
        !dir.path().join(".agent.worker-test.pid").exists(),
        "stop must clear PID state"
    );
}

/// Swift: `testAgentCommandLineFallbackForeignName`. Same binary name but a
/// DIFFERENT `--worker-name` is another instance: foreign, never signalled.
#[test]
fn agent_command_line_fallback_foreign_name() {
    let dir = TempDir::new("cmdline-foreign");
    let mut unrelated = spawn_long_lived();
    let unrelated_pid = unrelated.id() as i32;
    std::fs::write(
        dir.path().join(".agent.worker-test.pid"),
        unrelated_pid.to_string(),
    )
    .expect("write pid file");

    let controller = AgentController::with_environment(
        dir.path().to_path_buf(),
        PathBuf::from("/usr/local/other/sleep"),
        HashMap::new(),
    )
    .with_process_executable_path(Box::new(|_pid| None))
    .with_process_command_line(Box::new(|_pid| {
        Some(vec![
            "/bin/sleep".to_string(),
            "--worker-name".to_string(),
            "worker-other".to_string(),
        ])
    }));

    let result = controller
        .execute(&agent_args(AgentAction::Stop, "worker-test"))
        .expect("execute");
    let object = decode(&result.output);
    assert_eq!(
        object["status"], "not_running",
        "foreign name treated as stale"
    );
    assert!(
        unrelated.try_wait().expect("try_wait").is_none(),
        "agent must not signal a foreign process"
    );
    let _ = unrelated.kill();
    let _ = unrelated.wait();
}

/// Simulates a Linux-only real-world scenario (a `cargo build` overwriting
/// `target/release/<bin>` makes a live instance's `/proc/<pid>/exe` read
/// "<path> (deleted)") via a mocked `process_executable_path`, so it runs on
/// every platform even though the scenario itself can only happen on Linux
/// (see `pidlock.rs::inspect_process`'s doc comment on the " (deleted)"
/// branch: pure string matching, deliberately left unconditional rather
/// than `#[cfg(unix)]`-gated — harmless dead code on Windows in production,
/// where `platform::process_executable_path` never returns that suffix).
/// Same file name + exact `--worker-name` argument is still our instance.
#[test]
fn agent_deleted_exe_suffix_owned() {
    let dir = TempDir::new("deleted-exe");
    let mut child = spawn_long_lived();
    let child_pid = child.id() as i32;
    std::fs::write(
        dir.path().join(".agent.worker-test.pid"),
        child_pid.to_string(),
    )
    .expect("write pid file");

    let controller = AgentController::with_environment(
        dir.path().to_path_buf(),
        PathBuf::from("/bin/sleep"),
        HashMap::new(),
    )
    .with_process_executable_path(Box::new(|_pid| Some(PathBuf::from("/bin/sleep (deleted)"))))
    .with_process_command_line(Box::new(|_pid| {
        Some(vec![
            "/bin/sleep".to_string(),
            "--worker-name".to_string(),
            "worker-test".to_string(),
        ])
    }));

    let result = controller
        .execute(&agent_args(AgentAction::Stop, "worker-test"))
        .expect("execute");
    let object = decode(&result.output);
    assert_eq!(
        object["status"], "stopped",
        "deleted-exe identity must allow stop"
    );
    let _ = child.wait();
}

/// Swift: `testAgentUnverifiablePID`. A live PID (this very test process)
/// whose identity cannot be resolved must fail closed (exit 1,
/// `"unverifiable"`) on status/stop/start, and the PID state must never be
/// deleted while unverifiable — deleting it could let a second `start`
/// spawn a duplicate alongside a real, merely-unverifiable worker.
#[test]
fn agent_unverifiable_pid() {
    let dir = TempDir::new("unverifiable");
    let pid_path = dir.path().join(".agent.worker-test.pid");
    let name_path = dir.path().join(".agent.worker-test.worker_name");
    std::fs::write(&pid_path, std::process::id().to_string()).expect("write pid");
    std::fs::write(&name_path, "worker-test").expect("write name");

    let controller = AgentController::with_environment(
        dir.path().to_path_buf(),
        PathBuf::from("/usr/bin/true"),
        HashMap::new(),
    )
    .with_process_executable_path(Box::new(|_pid| None))
    .with_process_command_line(Box::new(|_pid| None));

    let status = controller
        .execute(&agent_args(AgentAction::Status, "worker-test"))
        .expect("status");
    let status_object = decode(&status.output);
    assert_eq!(status.exit_code, 1, "unverifiable status must fail closed");
    assert_eq!(status_object["status"], "unverifiable");
    assert!(
        pid_path.exists(),
        "status must preserve live unverifiable PID"
    );

    let stop = controller
        .execute(&agent_args(AgentAction::Stop, "worker-test"))
        .expect("stop");
    let stop_object = decode(&stop.output);
    assert_eq!(stop.exit_code, 1, "unverifiable stop must fail closed");
    assert_eq!(stop_object["status"], "unverifiable");
    assert!(
        pid_path.exists(),
        "stop must preserve live unverifiable PID"
    );

    let mut start_args = agent_args(AgentAction::Start, "worker-test");
    start_args.allow_start = true;
    let start = controller.execute(&start_args).expect("start");
    let start_object = decode(&start.output);
    assert_eq!(start.exit_code, 1, "unverifiable start must fail closed");
    assert_eq!(start_object["status"], "unverifiable");
    assert!(
        pid_path.exists(),
        "start must preserve live unverifiable PID"
    );
}

fn concurrent_start_args() -> AgentArguments {
    let mut args = AgentArguments::new(AgentAction::Start);
    args.worker_name = Some("worker-test".to_string());
    args.allow_start = true;
    args
}

/// Swift: `testAgentConcurrentStart`. Two `start` calls for the same
/// worker name race; the `flock`-based startup lock
/// (`AgentController::with_startup_lock`) must serialize them so exactly
/// one spawns and the PID file is written once — the loser observes
/// "already_running" (normal race) or "unverifiable" (lost the race to
/// inspect a live-but-not-yet-owned-looking PID, same accepted outcome the
/// Swift test allows). See `src/bin/test_sleep_helper.rs` for why a
/// dedicated helper binary stands in for the worker executable here.
#[test]
fn agent_concurrent_start() {
    let dir = TempDir::new("concurrent-start");
    std::fs::write(dir.path().join("config.json"), "{}").expect("write config.json");
    let executable = PathBuf::from(env!("CARGO_BIN_EXE_test_sleep_helper"));

    let dir_a = dir.path().to_path_buf();
    let exe_a = executable.clone();
    let handle_a = std::thread::spawn(move || {
        AgentController::with_environment(dir_a, exe_a, HashMap::new())
            .execute(&concurrent_start_args())
    });
    let dir_b = dir.path().to_path_buf();
    let exe_b = executable.clone();
    let handle_b = std::thread::spawn(move || {
        AgentController::with_environment(dir_b, exe_b, HashMap::new())
            .execute(&concurrent_start_args())
    });

    let result_a = handle_a
        .join()
        .expect("thread a joined")
        .expect("execute a");
    let result_b = handle_b
        .join()
        .expect("thread b joined")
        .expect("execute b");
    let object_a = decode(&result_a.output);
    let object_b = decode(&result_b.output);

    let mut statuses = vec![
        object_a["status"].as_str().unwrap().to_string(),
        object_b["status"].as_str().unwrap().to_string(),
    ];
    statuses.sort();
    let already_running_started = vec!["already_running".to_string(), "started".to_string()];
    let started_unverifiable = vec!["started".to_string(), "unverifiable".to_string()];
    assert!(
        statuses == already_running_started || statuses == started_unverifiable,
        "one concurrent start must win and the other must fail closed: {statuses:?}"
    );

    let pids: std::collections::HashSet<i64> = [object_a["pid"].as_i64(), object_b["pid"].as_i64()]
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(pids.len(), 1, "concurrent starts must report the same PID");

    for pid in pids {
        exonit_core::platform::kill_now(pid as i32);
    }
}

/// Swift: `testAgentLogTail`.
#[test]
fn agent_log_tail() {
    let dir = TempDir::new("log-tail");
    std::fs::write(
        dir.path().join("worker-worker-test.log"),
        "one\ntwo\nthree\nfour\n",
    )
    .expect("write log");
    let controller = AgentController::with_environment(
        dir.path().to_path_buf(),
        PathBuf::from("/usr/bin/true"),
        HashMap::new(),
    );
    let mut args = agent_args(AgentAction::Logs, "worker-test");
    args.log_lines = 2;
    let result = controller.execute(&args).expect("execute");
    assert_eq!(
        String::from_utf8(result.output).unwrap(),
        "three\nfour\n",
        "tail two lines"
    );
}

// ---------------------------------------------------------------------------
// Python FPP sidecar (2026-08-07) — end-to-end through
// `AgentController::execute`'s `--fpp-instances` path. See
// `exonit_core::python_sidecar`'s module doc comment for the design.
// ---------------------------------------------------------------------------

/// `agent start <name> --fpp-instances M` without a `python_sidecar` section
/// in config.json must REFUSE loudly (exit 2), not silently fall through to
/// a native start — there is no native worker to fall back to meaningfully
/// here (the flag only exists to redirect to sidecars).
#[test]
fn fpp_instances_without_sidecar_config_refuses() {
    let dir = TempDir::new("sidecar-no-config");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"server_url":"http://x","token":"t"}"#,
    )
    .expect("write config.json");
    let controller = AgentController::with_environment(
        dir.path().to_path_buf(),
        PathBuf::from("/usr/bin/true"),
        HashMap::new(),
    );
    let mut args = agent_args(AgentAction::Start, "worker-rust-test");
    args.allow_start = true;
    args.fpp_instances = 2;
    let result = controller.execute(&args).expect("execute");
    let object = decode(&result.output);
    assert_eq!(result.exit_code, 2, "refused sidecar start exit code");
    assert_eq!(object["status"], "refused");
    assert!(
        !python_sidecar_state_path(dir.path(), "worker-rust-test").exists(),
        "refused start must have no sidecar state side effect"
    );
}

fn python_sidecar_state_path(dir: &Path, worker_name: &str) -> PathBuf {
    dir.join(format!(".agent.{worker_name}.python_sidecars.json"))
}

/// Real end-to-end: `start --fpp-instances 2` spawns two REAL `python3`
/// processes (via the fake `agent.py`), `status` reports both running, and
/// `stop` kills both and cleans up all state — no flag needed on stop/status,
/// they read back what `start` persisted. Skips cleanly if `python3` isn't
/// on `PATH` (see `fake_python_agent::python3_available`).
#[test]
fn fpp_instances_start_status_stop_round_trip() {
    if !fake_python_agent::python3_available() {
        eprintln!("skipping fpp_instances_start_status_stop_round_trip: no python3 on PATH");
        return;
    }
    let rust_dir = TempDir::new("sidecar-rust-worker-dir");
    let python_dir = TempDir::new("sidecar-python-worker-dir");
    fake_python_agent::write_fake_agent(python_dir.path());
    let config = format!(
        r#"{{"server_url":"http://x","token":"t","python_sidecar":{{"python":"python3","worker_dir":{:?},"env":{{}}}}}}"#,
        python_dir.path().to_string_lossy()
    );
    std::fs::write(rust_dir.path().join("config.json"), config).expect("write config.json");

    let controller = AgentController::with_environment(
        rust_dir.path().to_path_buf(),
        PathBuf::from("/usr/bin/true"),
        HashMap::new(),
    );

    // start
    let mut start_args = agent_args(AgentAction::Start, "worker-rust-test");
    start_args.allow_start = true;
    start_args.fpp_instances = 2;
    let start_result = controller.execute(&start_args).expect("execute start");
    let start_object = decode(&start_result.output);
    assert_eq!(start_result.exit_code, 0, "sidecar start: {start_object}");
    assert_eq!(start_object["status"], "started_sidecars");
    let instances = start_object["python_sidecars"]
        .as_array()
        .expect("python_sidecars array");
    assert_eq!(instances.len(), 2);
    assert!(
        python_sidecar_state_path(rust_dir.path(), "worker-rust-test").exists(),
        "start must persist sidecar state"
    );

    // status: no native pid file was ever written (this call never took the
    // native path), so the native half stays "stopped"/null exactly as
    // before this feature, PLUS a python_sidecars array.
    let status_result = controller
        .execute(&agent_args(AgentAction::Status, "worker-rust-test"))
        .expect("execute status");
    let status_object = decode(&status_result.output);
    assert_eq!(
        status_object["status"], "stopped",
        "native half unaffected by sidecars"
    );
    let status_rows = status_object["python_sidecars"]
        .as_array()
        .expect("python_sidecars array");
    assert_eq!(status_rows.len(), 2);
    for row in status_rows {
        assert_eq!(row["running"], true, "sidecar should be alive: {row}");
    }

    // stop
    let stop_result = controller
        .execute(&agent_args(AgentAction::Stop, "worker-rust-test"))
        .expect("execute stop");
    let stop_object = decode(&stop_result.output);
    assert_eq!(
        stop_object["status"], "not_running",
        "native half unaffected by sidecars"
    );
    let stop_rows = stop_object["python_sidecars"]
        .as_array()
        .expect("python_sidecars array");
    assert_eq!(stop_rows.len(), 2);
    for row in stop_rows {
        assert_eq!(row["exit_code"], 0, "fake agent.py stop failed: {row}");
    }
    assert!(
        !python_sidecar_state_path(rust_dir.path(), "worker-rust-test").exists(),
        "stop must remove sidecar state"
    );

    // status again: sidecar state is gone, back to the pre-feature shape
    // with no python_sidecars key at all.
    let final_status = controller
        .execute(&agent_args(AgentAction::Status, "worker-rust-test"))
        .expect("execute status");
    let final_object = decode(&final_status.output);
    assert_eq!(final_object["status"], "stopped");
    assert!(
        final_object.get("python_sidecars").is_none(),
        "no sidecar state left to report"
    );
}
