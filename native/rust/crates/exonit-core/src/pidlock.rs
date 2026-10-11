//! Transliterated from the PID-lock / stale-PID / spawn / signal portions of
//! `Sources/ExoNitCore/AgentController.swift` in the Swift reference worker.
//! The runtime scheduler (`WorkerRuntime.swift`) is explicitly **not**
//! ported here — this module only implements what `agent.py start|stop|
//! status|logs|version` needs: spawning a detached child worker process,
//! recording/reading its PID, detecting whether a recorded PID is still the
//! process we started (not a stale/foreign PID reused by the OS), and the
//! `--allow-start` / `EXONIT_FARMCTL_ALLOW_START` guard.
//!
//! Process liveness/identity, spawn-detach, stop-signalling and the startup
//! lock are all delegated to `crate::platform` (extracted in
//! `docs/plans/2026-08-06-win32-port-pcmarc.md` Task 1, pure refactor, no
//! behavior change on Linux/<host>). Before that extraction this module
//! was Linux-only, using `kill(pid, 0)` for liveness and `/proc/<pid>/exe`
//! for identity directly; `platform::unix` now implements exactly that, and
//! `platform::windows` (Task 2, not yet written) will implement the same
//! signatures for worker-<host>.

use crate::cli::{worker_name_validator, AgentAction, AgentArguments};
use crate::config_derive;
use crate::platform::{self, StopKind};
use crate::python_sidecar;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentExecutionResult {
    pub output: Vec<u8>,
    pub exit_code: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentControllerError {
    MissingWorkerName,
    InvalidWorkerName(String),
    MissingBaseConfig(String),
    SpawnFailed(i32),
    StartupLockFailed(String),
    InvalidUtf8,
    /// Not present in the Swift enum (which lets underlying `Error`s
    /// propagate untyped through `throws`); added so `execute()` has a
    /// single concrete error type in Rust. Wraps filesystem I/O failures
    /// that in Swift would surface as an opaque thrown `NSError`.
    Io(String),
    /// Wraps `AgentConfigError` from `config_derive::derive` /
    /// `apply_single_instance_overrides` when it escapes `prepare_config`
    /// uncaught (the one exception: the `--config-overrides-b64` decode
    /// error IS caught inline in `start()` and turned into a "refused" JSON
    /// response, matching Swift exactly).
    ConfigDerive(String),
}

impl fmt::Display for AgentControllerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentControllerError::MissingWorkerName => write!(f, "worker name is required"),
            AgentControllerError::InvalidWorkerName(value) => {
                write!(f, "invalid worker name: {value}")
            }
            AgentControllerError::MissingBaseConfig(path) => {
                write!(f, "base config not found: {path}")
            }
            AgentControllerError::SpawnFailed(code) => {
                write!(
                    f,
                    "spawn failed: {}",
                    std::io::Error::from_raw_os_error(*code)
                )
            }
            AgentControllerError::StartupLockFailed(message) => {
                write!(f, "startup lock failed: {message}")
            }
            AgentControllerError::InvalidUtf8 => {
                write!(f, "configuration override is not valid UTF-8")
            }
            AgentControllerError::Io(message) => write!(f, "I/O error: {message}"),
            AgentControllerError::ConfigDerive(message) => {
                write!(f, "config derivation error: {message}")
            }
        }
    }
}
impl std::error::Error for AgentControllerError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProcessInspection {
    NotRunning,
    Owned,
    Foreign,
    Unverifiable,
}

/// Injectable process-identity lookup, mirroring Swift's
/// `ProcessExecutableURL` typealias/closure parameter — tests substitute a
/// fake implementation to simulate stale/foreign/unverifiable PIDs without
/// needing to actually race real processes for every case.
pub type ProcessExecutablePathFn = Box<dyn Fn(i32) -> Option<PathBuf> + Send + Sync>;

/// Injectable argv lookup (`/proc/<pid>/cmdline`), mirroring Swift's
/// `ProcessCommandLine` (KERN_PROCARGS2) fallback: when the executable path
/// is unreadable or stale, the spawn path plus the exact `--worker-name`
/// argument still identify the instance.
pub type ProcessCommandLineFn = Box<dyn Fn(i32) -> Option<Vec<String>> + Send + Sync>;

pub struct AgentController {
    worker_directory: PathBuf,
    executable_url: PathBuf,
    environment: HashMap<String, String>,
    process_executable_path: ProcessExecutablePathFn,
    process_command_line: ProcessCommandLineFn,
}

impl AgentController {
    pub fn new(worker_directory: PathBuf, executable_url: PathBuf) -> Self {
        Self::with_environment(worker_directory, executable_url, std::env::vars().collect())
    }

    pub fn with_environment(
        worker_directory: PathBuf,
        executable_url: PathBuf,
        environment: HashMap<String, String>,
    ) -> Self {
        AgentController {
            worker_directory,
            executable_url,
            environment,
            process_executable_path: Box::new(platform::process_executable_path),
            process_command_line: Box::new(platform::process_command_line),
        }
    }

    /// Test hook, mirroring the Swift initializer's `processExecutableURL`
    /// parameter.
    pub fn with_process_executable_path(mut self, lookup: ProcessExecutablePathFn) -> Self {
        self.process_executable_path = lookup;
        self
    }

    /// Test hook, mirroring the Swift initializer's `processCommandLine`
    /// parameter.
    pub fn with_process_command_line(mut self, lookup: ProcessCommandLineFn) -> Self {
        self.process_command_line = lookup;
        self
    }

    pub fn execute(
        &self,
        arguments: &AgentArguments,
    ) -> Result<AgentExecutionResult, AgentControllerError> {
        if let Some(name) = &arguments.worker_name {
            if !worker_name_validator::is_valid(name) {
                return Err(AgentControllerError::InvalidWorkerName(name.clone()));
            }
        }
        match arguments.action {
            AgentAction::Start => self.start(arguments),
            AgentAction::Stop => self.stop(arguments),
            AgentAction::Status => self.status(arguments),
            AgentAction::Logs => self.logs(arguments),
            AgentAction::Version => json_result(
                json!({
                    "git_sha": self.git_sha(),
                    "agent_version": "2.0-rust",
                    "rust": "native",
                    "platform": platform::platform_label(),
                }),
                0,
            ),
        }
    }

    fn start(
        &self,
        arguments: &AgentArguments,
    ) -> Result<AgentExecutionResult, AgentControllerError> {
        let worker_name = arguments
            .worker_name
            .clone()
            .ok_or(AgentControllerError::MissingWorkerName)?;
        let authorised = arguments.allow_start
            || self
                .environment
                .get("EXONIT_FARMCTL_ALLOW_START")
                .map(String::as_str)
                == Some("1");
        if !authorised {
            return json_result(
                json!({
                    "status": "refused",
                    "error": "REFUSED: starting a worker needs --allow-start (or EXONIT_FARMCTL_ALLOW_START=1). bin/farmctl passes it for you.",
                }),
                2,
            );
        }

        // 2026-08-07 (python-fpp-sidecar): `--fpp-instances M` is a
        // completely separate start path — see `python_sidecar`'s module
        // doc comment. Branches BEFORE the native `--config-overrides-b64`
        // decode below because the sidecar path forwards that value
        // verbatim to the Python `agent.py` (which decodes it itself) rather
        // than decoding it here. `arguments.fpp_instances` defaults to 0, so
        // every pre-existing caller (nothing sets this flag) takes the
        // native path exactly as before this feature existed.
        if arguments.fpp_instances > 0 {
            return self.with_startup_lock(&worker_name, || {
                self.start_python_sidecars(arguments, &worker_name)
            });
        }

        let overrides_data: Option<Vec<u8>> = match &arguments.config_overrides_base64 {
            Some(encoded) => match config_derive::decode_overrides(encoded) {
                Ok(data) => Some(data),
                Err(error) => {
                    return json_result(
                        json!({ "status": "refused", "error": format!("REFUSED: {error}") }),
                        2,
                    );
                }
            },
            None => None,
        };

        self.with_startup_lock(&worker_name, || {
            self.start_locked(arguments, &worker_name, overrides_data.as_deref())
        })
    }

    fn start_locked(
        &self,
        arguments: &AgentArguments,
        worker_name: &str,
        overrides_data: Option<&[u8]>,
    ) -> Result<AgentExecutionResult, AgentControllerError> {
        self.migrate_legacy_pid(worker_name)?;
        if let Some(pid) = self.read_pid(worker_name) {
            match self.inspect_process(pid, worker_name) {
                ProcessInspection::Owned => {
                    return json_result(
                        json!({
                            "status": "already_running",
                            "pid": pid,
                            "worker_name": self.read_name(worker_name).unwrap_or_else(|| worker_name.to_string()),
                        }),
                        0,
                    );
                }
                ProcessInspection::Unverifiable => return unverifiable_result(pid),
                ProcessInspection::Foreign | ProcessInspection::NotRunning => {
                    self.remove_state(worker_name);
                }
            }
        }

        let config_path = self.prepare_config(arguments, worker_name, overrides_data)?;
        let log_path = self
            .worker_directory
            .join(format!("worker-{worker_name}.log"));
        let mut child_arguments = vec![
            "--config".to_string(),
            config_path.to_string_lossy().to_string(),
            "--worker-name".to_string(),
            worker_name.to_string(),
        ];
        if let Some(server_url) = &arguments.server_url {
            child_arguments.push("--server-url".to_string());
            child_arguments.push(server_url.clone());
        }
        let mut child_environment = self.environment.clone();
        // Rust-worker analogue of Swift's `EXONIT_SWIFT_WORKER_CHILD` marker.
        child_environment.insert("EXONIT_RUST_WORKER_CHILD".to_string(), "1".to_string());

        let (pid, mut child) = self
            .spawn_detached(&child_arguments, &child_environment, &log_path)
            .map_err(|e| AgentControllerError::SpawnFailed(e.raw_os_error().unwrap_or(-1)))?;
        // Publish state only through atomic renames.  `status` can run in a
        // separate process while start is writing; it must never observe a
        // truncated PID/name and accidentally treat a live worker as stale.
        write_state_file(&self.pid_file(worker_name), &pid.to_string())
            .map_err(|e| AgentControllerError::Io(e.to_string()))?;
        write_state_file(&self.name_file(worker_name), worker_name)
            .map_err(|e| AgentControllerError::Io(e.to_string()))?;
        std::thread::sleep(std::time::Duration::from_secs(1));
        let alive = newly_spawned_child_is_alive(&mut child, pid);
        if !alive {
            self.remove_state(worker_name);
        }
        json_result(
            json!({
                "status": if alive { "started" } else { "exited_immediately" },
                "pid": pid,
                "worker_name": worker_name,
                "git_sha": self.git_sha(),
            }),
            if alive { 0 } else { 1 },
        )
    }

    /// `--fpp-instances M` path (see `python_sidecar`'s module doc comment
    /// and this struct's `start`). Reads `python_sidecar` out of THIS
    /// agent's own `config.json` (same file `prepare_config`'s `base_url`
    /// reads for the native path), spawns M `python worker.py` instances via
    /// the Python fleet's `agent.py start`, and persists the instance names
    /// so a later `stop`/`status` (a separate CLI invocation) can find them
    /// again without needing `--fpp-instances` repeated.
    fn start_python_sidecars(
        &self,
        arguments: &AgentArguments,
        worker_name: &str,
    ) -> Result<AgentExecutionResult, AgentControllerError> {
        let config_path = self.worker_directory.join("config.json");
        let sidecar_config = match python_sidecar::read_config(&config_path) {
            Ok(Some(cfg)) => cfg,
            Ok(None) => {
                return json_result(
                    json!({
                        "status": "refused",
                        "error": "REFUSED: --fpp-instances requires a `python_sidecar` \
                                  section ({\"python\": ..., \"worker_dir\": ...}) in \
                                  this host's config.json.",
                    }),
                    2,
                );
            }
            Err(error) => {
                return json_result(
                    json!({ "status": "refused", "error": format!("REFUSED: {error}") }),
                    2,
                );
            }
        };

        let outcomes = python_sidecar::start_sidecars(
            &sidecar_config,
            worker_name,
            arguments.fpp_instances,
            arguments.server_url.as_deref(),
            arguments.config_overrides_base64.as_deref(),
        )
        .map_err(|e| AgentControllerError::Io(e.to_string()))?;

        let names: Vec<String> = outcomes.iter().map(|o| o.instance_name.clone()).collect();
        python_sidecar::write_state(&self.worker_directory, worker_name, &names)
            .map_err(|e| AgentControllerError::Io(e.to_string()))?;

        let all_ok = outcomes.iter().all(|o| o.exit_code == 0);
        let instances: Vec<Value> = outcomes
            .iter()
            .map(|o| json!({ "instance": o.instance_name, "exit_code": o.exit_code, "output": o.raw_output }))
            .collect();
        json_result(
            json!({
                "status": if all_ok { "started_sidecars" } else { "sidecars_partial_failure" },
                "worker_name": worker_name,
                "python_sidecars": instances,
                "git_sha": self.git_sha(),
            }),
            if all_ok { 0 } else { 1 },
        )
    }

    /// Stops the native worker (unchanged logic) THEN, if `start`'s
    /// `--fpp-instances` ever recorded sidecars for this `worker_name`
    /// (`python_sidecar::read_state` — absent for every worker that never
    /// used the feature, which is every pre-2026-08-07 caller), also stops
    /// those. The native JSON shape/keys/exit-code are UNCHANGED when there
    /// is no sidecar state — see `merge_sidecar_stop`.
    fn stop(
        &self,
        arguments: &AgentArguments,
    ) -> Result<AgentExecutionResult, AgentControllerError> {
        let worker_name = arguments
            .worker_name
            .clone()
            .ok_or(AgentControllerError::MissingWorkerName)?;
        let (native_value, native_exit) = self.stop_native(&worker_name, arguments.force)?;
        self.merge_sidecar_stop(&worker_name, arguments.force, native_value, native_exit)
    }

    /// The exact `stop()` body from before the python-fpp-sidecar feature,
    /// returning `(Value, exit_code)` instead of a wrapped
    /// `AgentExecutionResult` so `stop()` can layer sidecar info on top
    /// without touching a single native status/key/exit-code decision here.
    fn stop_native(
        &self,
        worker_name: &str,
        force: bool,
    ) -> Result<(Value, i32), AgentControllerError> {
        self.migrate_legacy_pid(worker_name)?;
        let pid = match self.read_pid(worker_name) {
            Some(pid) => pid,
            None => {
                self.remove_state(worker_name);
                return Ok((json!({ "status": "not_running" }), 0));
            }
        };
        match self.inspect_process(pid, worker_name) {
            ProcessInspection::NotRunning | ProcessInspection::Foreign => {
                self.remove_state(worker_name);
                return Ok((json!({ "status": "not_running" }), 0));
            }
            ProcessInspection::Unverifiable => return Ok((unverifiable_json(pid), 1)),
            ProcessInspection::Owned => {}
        }
        let recorded_name = self
            .read_name(worker_name)
            .unwrap_or_else(|| worker_name.to_string());
        let stop_kind = if force {
            StopKind::Force
        } else {
            StopKind::Soft
        };
        if let Err(err) =
            platform::request_stop(pid, stop_kind, worker_name, &self.worker_directory)
        {
            return Ok((
                json!({ "status": "error", "error": err.to_string(), "pid": pid }),
                0,
            ));
        }
        if !force {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while platform::process_alive(pid) && std::time::Instant::now() < deadline {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            if platform::process_alive(pid) {
                platform::kill_now(pid);
            }
        }
        self.remove_state(worker_name);
        Ok((
            json!({ "status": "stopped", "pid": pid, "worker_name": recorded_name }),
            0,
        ))
    }

    /// Adds a `"python_sidecars"` key (array of `{instance, exit_code,
    /// output}`, or a single `{"error": ..., "instance_names": [...]}` object
    /// on failure) to `native_value` ONLY when `read_state` finds a non-empty
    /// sidecar list — every worker that never called `--fpp-instances` has
    /// no state file, so `native_value`/`native_exit` pass through
    /// UNCHANGED, byte-for-byte the same JSON `stop()` produced before this
    /// feature existed. Sidecar failures never override `native_exit`: an
    /// operator stopping a worker wants the native process dead even if
    /// sidecar cleanup needs a second look (reported in the JSON, not by a
    /// non-zero exit that could mask "the thing I actually asked to stop,
    /// stopped fine").
    fn merge_sidecar_stop(
        &self,
        worker_name: &str,
        force: bool,
        native_value: Value,
        native_exit: i32,
    ) -> Result<AgentExecutionResult, AgentControllerError> {
        let Some(names) = python_sidecar::read_state(&self.worker_directory, worker_name) else {
            return json_result(native_value, native_exit);
        };
        if names.is_empty() {
            python_sidecar::remove_state(&self.worker_directory, worker_name);
            return json_result(native_value, native_exit);
        }
        let sidecar_field = self.stop_python_sidecars(worker_name, &names, force);
        json_result(
            insert_field(native_value, "python_sidecars", sidecar_field),
            native_exit,
        )
    }

    fn stop_python_sidecars(&self, worker_name: &str, names: &[String], force: bool) -> Value {
        let sidecar_config = match python_sidecar::read_config(
            &self.worker_directory.join("config.json"),
        ) {
            Ok(Some(cfg)) => cfg,
            Ok(None) => {
                return json!({
                    "error": "state file present but config.json has no python_sidecar \
                              section anymore; instances left for manual cleanup",
                    "instance_names": names,
                });
            }
            Err(error) => {
                return json!({ "error": format!("reading python_sidecar config: {error}"), "instance_names": names });
            }
        };
        match python_sidecar::stop_sidecars(&sidecar_config, names, force) {
            Ok(outcomes) => {
                python_sidecar::remove_state(&self.worker_directory, worker_name);
                Value::Array(
                    outcomes
                        .into_iter()
                        .map(|o| json!({ "instance": o.instance_name, "exit_code": o.exit_code, "output": o.raw_output }))
                        .collect(),
                )
            }
            Err(error) => {
                json!({ "error": format!("spawning python agent.py stop: {error}"), "instance_names": names })
            }
        }
    }

    /// Same shape as `stop()`: native `status()` logic unchanged, plus a
    /// `"python_sidecars"` key (array of `{instance, pid, running}`, PID
    /// read straight off each sidecar's own `.agent.<name>.pid` file rather
    /// than spawning a `python agent.py status` subprocess per instance —
    /// cheap enough to call on every poll) when sidecar state exists.
    fn status(
        &self,
        arguments: &AgentArguments,
    ) -> Result<AgentExecutionResult, AgentControllerError> {
        let worker_name = arguments
            .worker_name
            .clone()
            .ok_or(AgentControllerError::MissingWorkerName)?;
        let (native_value, native_exit) = self.status_native(&worker_name);
        self.merge_sidecar_status(&worker_name, native_value, native_exit)
    }

    fn status_native(&self, worker_name: &str) -> (Value, i32) {
        let pid = match self.read_pid(worker_name) {
            Some(pid) => pid,
            None => {
                self.remove_state(worker_name);
                return (
                    json!({ "status": "stopped", "pid": Value::Null, "worker_name": Value::Null, "git_sha": self.git_sha() }),
                    0,
                );
            }
        };
        match self.inspect_process(pid, worker_name) {
            ProcessInspection::Unverifiable => (unverifiable_json(pid), 1),
            ProcessInspection::Owned => (
                json!({
                    "status": "running",
                    "pid": pid,
                    "worker_name": self.read_name(worker_name).unwrap_or_else(|| worker_name.to_string()),
                    "git_sha": self.git_sha(),
                }),
                0,
            ),
            ProcessInspection::Foreign | ProcessInspection::NotRunning => {
                self.remove_state(worker_name);
                (
                    json!({ "status": "stopped", "pid": Value::Null, "worker_name": Value::Null, "git_sha": self.git_sha() }),
                    0,
                )
            }
        }
    }

    fn merge_sidecar_status(
        &self,
        worker_name: &str,
        native_value: Value,
        native_exit: i32,
    ) -> Result<AgentExecutionResult, AgentControllerError> {
        let Some(names) = python_sidecar::read_state(&self.worker_directory, worker_name) else {
            return json_result(native_value, native_exit);
        };
        if names.is_empty() {
            return json_result(native_value, native_exit);
        }
        let sidecar_worker_dir =
            python_sidecar::read_config(&self.worker_directory.join("config.json"))
                .ok()
                .flatten()
                .map(|cfg| cfg.worker_dir);
        let rows: Vec<Value> = names
            .iter()
            .map(|name| {
                let pid = sidecar_worker_dir
                    .as_deref()
                    .and_then(|dir| python_sidecar::sidecar_pid(Path::new(dir), name));
                let running = pid.map(platform::process_alive).unwrap_or(false);
                json!({ "instance": name, "pid": pid, "running": running })
            })
            .collect();
        json_result(
            insert_field(native_value, "python_sidecars", Value::Array(rows)),
            native_exit,
        )
    }

    fn logs(
        &self,
        arguments: &AgentArguments,
    ) -> Result<AgentExecutionResult, AgentControllerError> {
        let preferred = arguments
            .worker_name
            .as_ref()
            .map(|name| self.worker_directory.join(format!("worker-{name}.log")));
        let legacy = self.worker_directory.join("worker.log");
        let selected: Option<PathBuf> = if preferred.as_deref().is_some_and(Path::exists) {
            preferred
        } else if legacy.exists() {
            Some(legacy)
        } else {
            None
        };
        let Some(selected) = selected else {
            return Ok(AgentExecutionResult {
                output: b"\n".to_vec(),
                exit_code: 0,
            });
        };
        let text = std::fs::read_to_string(&selected)
            .map_err(|e| AgentControllerError::Io(e.to_string()))?;
        // Swift: `text.split(whereSeparator: \.isNewline)` — omits empty
        // subsequences by default, so CRLF doesn't produce spurious blanks.
        let all_lines: Vec<&str> = text
            .split(|c: char| c == '\n' || c == '\r')
            .filter(|line| !line.is_empty())
            .collect();
        let lines: Vec<&str> = if arguments.log_lines > 0 {
            let n = arguments.log_lines as usize;
            let start = all_lines.len().saturating_sub(n);
            all_lines[start..].to_vec()
        } else {
            all_lines
        };
        let mut output = lines.join("\n").into_bytes();
        output.push(b'\n');
        Ok(AgentExecutionResult {
            output,
            exit_code: 0,
        })
    }

    fn prepare_config(
        &self,
        arguments: &AgentArguments,
        worker_name: &str,
        overrides_data: Option<&[u8]>,
    ) -> Result<PathBuf, AgentControllerError> {
        let base_url = self.worker_directory.join("config.json");
        if !base_url.exists() {
            return Err(AgentControllerError::MissingBaseConfig(
                base_url.to_string_lossy().to_string(),
            ));
        }
        if arguments.instances == 1 && arguments.instance_index == 1 && overrides_data.is_none() {
            return Ok(base_url);
        }
        let base_data =
            std::fs::read(&base_url).map_err(|e| AgentControllerError::Io(e.to_string()))?;
        let output = if arguments.instances == 1 && arguments.instance_index == 1 {
            config_derive::apply_single_instance_overrides(
                &base_data,
                overrides_data
                    .expect("overrides_data present: single-instance-no-overrides handled above"),
            )
            .map_err(|e| AgentControllerError::ConfigDerive(e.to_string()))?
        } else {
            config_derive::derive(
                &base_data,
                overrides_data,
                worker_name,
                arguments.instance_index,
                arguments.instances,
                &self.worker_directory,
                available_cpu_count(),
            )
            .map_err(|e| AgentControllerError::ConfigDerive(e.to_string()))?
        };
        let derived_url = self
            .worker_directory
            .join(format!(".config.{worker_name}.json"));
        std::fs::write(&derived_url, &output)
            .map_err(|e| AgentControllerError::Io(e.to_string()))?;
        Ok(derived_url)
    }

    /// Spawns the worker binary detached into a new session (so it survives
    /// this short-lived `agent start` CLI process exiting), stdin from
    /// `/dev/null`, stdout+stderr appended to `log_path`. Mirrors
    /// `AgentController.spawnDetached`'s `posix_spawn` with
    /// `POSIX_SPAWN_SETSID`.
    ///
    /// Unix and Windows diverge here (not just via a `platform::*` hook, as
    /// everywhere else in this module) as of 2026-08-07: Windows no longer
    /// goes through `std::process::Command` at all — see
    /// `platform::windows::spawn_worker_detached`'s doc comment for why (SSH
    /// hang caused by `Command`'s unconditional `bInheritHandles = TRUE`).
    /// Unix's path is byte-for-byte the pre-existing one.
    #[cfg(unix)]
    fn spawn_detached(
        &self,
        arguments: &[String],
        environment: &HashMap<String, String>,
        log_path: &Path,
    ) -> std::io::Result<(i32, SpawnedChild)> {
        let log_out = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?;
        let log_err = log_out.try_clone()?;
        let mut command = std::process::Command::new(&self.executable_url);
        command.args(arguments);
        command.env_clear();
        for (key, value) in environment {
            command.env(key, value);
        }
        command.stdin(std::process::Stdio::null());
        command.stdout(log_out);
        command.stderr(log_err);
        platform::configure_detached(&mut command);
        let child = command.spawn()?;
        let pid = child.id() as i32;
        Ok((pid, SpawnedChild::Process(child)))
    }

    /// Windows: the whole `CreateProcessW` + restricted-handle-inheritance
    /// dance lives in `platform::windows::spawn_worker_detached` (it needs
    /// raw WinAPI calls `std::process::Command` has no stable hook for,
    /// unlike the `CommandExt::creation_flags` hook `configure_detached`
    /// used before). No `std::process::Child` comes back — liveness is
    /// re-derived via `platform::process_alive(pid)`, same fallback path
    /// `SpawnedChild::is_alive` already used for `Child::try_wait()` errors.
    #[cfg(windows)]
    fn spawn_detached(
        &self,
        arguments: &[String],
        environment: &HashMap<String, String>,
        log_path: &Path,
    ) -> std::io::Result<(i32, SpawnedChild)> {
        let pid = platform::spawn_worker_detached(
            &self.executable_url,
            arguments,
            environment,
            log_path,
        )?;
        Ok((pid, SpawnedChild::Detached))
    }

    fn inspect_process(&self, pid: i32, worker_name: &str) -> ProcessInspection {
        if !platform::process_alive(pid) {
            return ProcessInspection::NotRunning;
        }
        let expected = resolve_symlinks_best_effort(&self.executable_url);
        if let Some(actual) = (self.process_executable_path)(pid) {
            // A rebuild overwrites target/release/<bin>, so live instances'
            // `/proc/<pid>/exe` reads "<path> (deleted)". Same file name +
            // the exact `--worker-name` argument is still our instance
            // (mirror of the Swift release-flip branch).
            //
            // Deliberately left UNCONDITIONAL (no `#[cfg(unix)]`) rather
            // than compiled out on Windows, per
            // docs/plans/2026-08-06-win32-port-pcmarc.md Task 2: this is
            // pure string matching, no OS API involved, so it is harmless
            // dead code on Windows in production
            // (`platform::process_executable_path` there is
            // `QueryFullProcessImageNameW`, which never returns a path with
            // this suffix — the exe stays locked while running, unlike
            // Linux's `/proc/<pid>/exe` symlink). Gating it out would only
            // have broken `pidlock_tests.rs::agent_deleted_exe_suffix_owned`
            // on Windows, which exercises this exact branch via a mocked
            // `process_executable_path` closure and is valuable coverage on
            // every platform, not just the one where the real-world
            // scenario can occur.
            let display = actual.to_string_lossy();
            if let Some(trimmed) = display.strip_suffix(" (deleted)") {
                let trimmed = PathBuf::from(trimmed);
                if trimmed.file_name() == expected.file_name()
                    && self.command_line_names_instance(pid, worker_name)
                {
                    return ProcessInspection::Owned;
                }
                return ProcessInspection::Foreign;
            }
            let actual = resolve_symlinks_best_effort(&actual);
            if actual == expected {
                return ProcessInspection::Owned;
            }
            return ProcessInspection::Foreign;
        }
        // Executable path unreadable: fall back to argv, mirroring the
        // Swift KERN_PROCARGS2 fallback (production 2026-08-05, macOS
        // proc_pidpath returning 0 for live detached children).
        match (self.process_command_line)(pid) {
            None => ProcessInspection::Unverifiable,
            Some(argv) if argv.is_empty() => ProcessInspection::Unverifiable,
            Some(argv) => {
                let spawn = resolve_symlinks_best_effort(Path::new(&argv[0]));
                if spawn == expected {
                    return ProcessInspection::Owned;
                }
                if spawn.file_name() == expected.file_name()
                    && argv_names_instance(&argv, worker_name)
                {
                    return ProcessInspection::Owned;
                }
                ProcessInspection::Foreign
            }
        }
    }

    fn command_line_names_instance(&self, pid: i32, worker_name: &str) -> bool {
        (self.process_command_line)(pid)
            .map(|argv| argv_names_instance(&argv, worker_name))
            .unwrap_or(false)
    }

    fn pid_file(&self, worker_name: &str) -> PathBuf {
        self.worker_directory
            .join(format!(".agent.{worker_name}.pid"))
    }

    fn name_file(&self, worker_name: &str) -> PathBuf {
        self.worker_directory
            .join(format!(".agent.{worker_name}.worker_name"))
    }

    fn read_pid(&self, worker_name: &str) -> Option<i32> {
        let data = std::fs::read_to_string(self.pid_file(worker_name)).ok()?;
        let value: i32 = data.trim().parse().ok()?;
        if value > 0 {
            Some(value)
        } else {
            None
        }
    }

    fn read_name(&self, worker_name: &str) -> Option<String> {
        let data = std::fs::read_to_string(self.name_file(worker_name)).ok()?;
        let trimmed = data.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }

    fn remove_state(&self, worker_name: &str) {
        let _ = std::fs::remove_file(self.pid_file(worker_name));
        let _ = std::fs::remove_file(self.name_file(worker_name));
    }

    fn migrate_legacy_pid(&self, worker_name: &str) -> Result<(), AgentControllerError> {
        let destination = self.pid_file(worker_name);
        if destination.exists() {
            return Ok(());
        }
        let legacy_pid = self.worker_directory.join(".agent.pid");
        if !legacy_pid.exists() {
            return Ok(());
        }
        std::fs::rename(&legacy_pid, &destination)
            .map_err(|e| AgentControllerError::Io(e.to_string()))?;
        let legacy_name = self.worker_directory.join(".agent.worker_name");
        if legacy_name.exists() {
            std::fs::rename(&legacy_name, self.name_file(worker_name))
                .map_err(|e| AgentControllerError::Io(e.to_string()))?;
        }
        Ok(())
    }

    /// Serializes `start()` (and its stale-PID cleanup + spawn) across
    /// concurrent invocations for the same `worker_name` via an exclusive
    /// `flock` on a dedicated lock file — mirrors
    /// `AgentController.withStartupLock`.
    fn with_startup_lock<T>(
        &self,
        worker_name: &str,
        body: impl FnOnce() -> Result<T, AgentControllerError>,
    ) -> Result<T, AgentControllerError> {
        let path = self
            .worker_directory
            .join(format!(".agent.{worker_name}.start.lock"));
        let _guard = platform::acquire_startup_lock(&path)
            .map_err(|e| AgentControllerError::StartupLockFailed(e.to_string()))?;
        body()
    }

    fn git_sha(&self) -> String {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(&self.worker_directory)
            .args(["rev-parse", "--short=8", "HEAD"])
            .output();
        match output {
            Ok(out) if out.status.success() => {
                String::from_utf8_lossy(&out.stdout).trim().to_string()
            }
            _ => "unknown".to_string(),
        }
    }
}

fn unverifiable_json(pid: i32) -> Value {
    json!({
        "status": "unverifiable",
        "error": "live PID identity could not be verified; state was preserved",
        "pid": pid,
    })
}

fn unverifiable_result(pid: i32) -> Result<AgentExecutionResult, AgentControllerError> {
    json_result(unverifiable_json(pid), 1)
}

/// Inserts `key`/`value` into `base` if `base` is a JSON object, mirroring
/// what `stop()`/`status()`'s pre-sidecar bodies always produced (every
/// branch there builds an object via `json!({...})`) — the non-object
/// fallback only exists so this helper has no partial function / panic path
/// if that ever stops being true.
fn insert_field(base: Value, key: &str, value: Value) -> Value {
    match base {
        Value::Object(mut map) => {
            map.insert(key.to_string(), value);
            Value::Object(map)
        }
        other => {
            let mut map = serde_json::Map::new();
            map.insert("native".to_string(), other);
            map.insert(key.to_string(), value);
            Value::Object(map)
        }
    }
}

fn json_result(value: Value, exit_code: i32) -> Result<AgentExecutionResult, AgentControllerError> {
    // `serde_json::Map` is BTreeMap-backed by default, so this always
    // serializes with sorted keys — matching Swift's `.sortedKeys` option.
    let mut bytes =
        serde_json::to_vec(&value).map_err(|e| AgentControllerError::Io(e.to_string()))?;
    bytes.push(b'\n');
    Ok(AgentExecutionResult {
        output: bytes,
        exit_code,
    })
}

fn write_state_file(path: &Path, contents: &str) -> std::io::Result<()> {
    static STATE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("state");
    let (temporary, mut file) = loop {
        let sequence = STATE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(
            ".{file_name}.tmp.{}.{}",
            std::process::id(),
            sequence
        ));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => break (temporary, file),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    };
    use std::io::Write;
    if let Err(error) = file
        .write_all(contents.as_bytes())
        .and_then(|_| file.sync_all())
    {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    drop(file);
    if let Err(error) = std::fs::rename(&temporary, path) {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    Ok(())
}

/// What `spawn_detached` hands back for the "is it still alive a moment
/// later" check in `start_locked`: on Unix a real `std::process::Child` we
/// can `try_wait()` (reaping it, avoiding a zombie); on Windows there is no
/// such handle (`platform::windows::spawn_worker_detached` closes its own
/// `hProcess`/`hThread` immediately after `CreateProcessW` returns — see its
/// doc comment for why that's fine), so liveness is re-derived by PID via
/// `platform::process_alive`, the same fallback the Unix arm already uses
/// when `try_wait()` itself errors.
enum SpawnedChild {
    #[cfg(unix)]
    Process(std::process::Child),
    #[cfg(windows)]
    Detached,
}

fn newly_spawned_child_is_alive(child: &mut SpawnedChild, pid: i32) -> bool {
    match child {
        #[cfg(unix)]
        SpawnedChild::Process(child) => match child.try_wait() {
            Ok(Some(_status)) => false,
            Ok(None) => true,
            // Mirrors the Swift fallback: if reaping the child races with
            // another waiter, liveness by PID is enough since the PID came
            // directly from our own spawn call.
            Err(_) => platform::process_alive(pid),
        },
        #[cfg(windows)]
        SpawnedChild::Detached => platform::process_alive(pid),
    }
}

fn argv_names_instance(argv: &[String], worker_name: &str) -> bool {
    argv.windows(2)
        .any(|pair| pair[0] == "--worker-name" && pair[1] == worker_name)
}

/// Best-effort symlink resolution for identity comparison, mirroring
/// `URL.resolvingSymlinksInPath()`: falls back to the original path (rather
/// than erroring) when the path can't be canonicalized, e.g. because it no
/// longer exists.
fn resolve_symlinks_best_effort(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn available_cpu_count() -> i64 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i64)
        .unwrap_or(1)
}
