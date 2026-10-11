//! Python FPP sidecar (2026-08-07, docs/PLAN.md).
//!
//! The Rust worker has no TRICERATOPS/FPP implementation and refuses to run
//! a `fpp_backfill`/`exominer_backfill` task pool at all
//! (`WorkerRuntimeError::BackfillPoolsUnimplemented`, see `runtime.rs`) — a
//! deliberate design decision, not a gap to close here. Before this module,
//! a host running the Rust worker for `scan` needed a SEPARATE manifest
//! entry running the Python worker for the `fpp` phase (see
//! `ops/workers.json`'s `worker-<host>-rust` + `worker-<host>`
//! pair). This module lets the SAME `agent.py`-compatible Rust CLI manage
//! both: `agent start <name> --fpp-instances M` spawns M `python worker.py`
//! processes by shelling out to the Python fleet's own
//! `worker/agent.py start` — reusing its config derivation (`gui_port`,
//! `fits_cache_dir`, `cpu_threads`, per-instance log file), PID-lock,
//! detached-spawn and stale-PID handling verbatim rather than
//! reimplementing any of it in Rust. `AgentController` (`pidlock.rs`) is the
//! only caller; this module has no dependency on it (plain functions over
//! `PythonSidecarConfig` + a worker/instance-name list), so it is testable
//! in isolation with a fake `python`/`agent.py` pair (see
//! `tests/python_sidecar_tests.rs`).
//!
//! Naming: sidecar instances are named `<worker_name>-py-<i>` (1-based),
//! distinct from the `<worker_name>-<i>` scheme `_iter_instances` in
//! `bin/farmctl` already uses for NATIVE multi-instance hosts — the two
//! schemes cannot collide, so native and sidecar instances can in principle
//! coexist under the same bare `worker_name` (in production they don't,
//! today: the `scan` phase prunes native instances before the `fpp` phase's
//! single `--fpp-instances` call runs, mirroring the pre-existing
//! `instances_fpp: 0` convention for Rust-native hosts).
//!
//! State: `start_sidecars` records the instance names it spawned in
//! `.agent.<worker_name>.python_sidecars.json` next to the Rust agent's own
//! `.agent.<worker_name>.pid` (see `state_path`). `stop`/`status` in
//! `pidlock.rs` read this back — no `--fpp-instances` needed on those
//! actions — so a manifest edit between `start` and `stop` can't orphan a
//! differently-sized sidecar fleet the way `farmctl`'s own
//! `_iter_instances_all_phases` can (see that function's doc comment for the
//! pre-existing limitation this improves on, not fixes elsewhere).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

/// Mirrors the optional `python_sidecar` object inside the Rust agent's own
/// `config.json` (NOT `ops/workers.json` — this is host-local, like
/// `worker_dir`/`python` are per-host in the manifest). Shape:
/// `{"python": "<venv python path>", "worker_dir": "<dir with worker.py +
/// agent.py>", "env": {"K": "V", ...}}`. `env` is optional and merged ON TOP
/// of this process's own inherited environment when spawning each sidecar
/// (`Command` inherits the parent env by default) — it exists specifically
/// so a host whose Rust worker's own env points at Rust-specific libraries
/// (e.g. a CUDA 13 `LD_LIBRARY_PATH` for `cudarc`) can override those back
/// to whatever the Python worker needs (e.g. CUDA 12), the same way
/// `ops/workers.json`'s per-host `env` field does for pure-Python hosts
/// today. A host that doesn't need an override can leave `env` absent
/// (defaults to empty).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PythonSidecarConfig {
    pub python: String,
    pub worker_dir: String,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SidecarConfigError {
    Io(String),
    Malformed(String),
}

impl fmt::Display for SidecarConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SidecarConfigError::Io(message) => write!(f, "reading config: {message}"),
            SidecarConfigError::Malformed(message) => write!(f, "python_sidecar: {message}"),
        }
    }
}
impl std::error::Error for SidecarConfigError {}

/// Reads the optional `python_sidecar` object out of `config_path` (the
/// agent's own base `config.json`). `Ok(None)` — the "zero canvi" default —
/// when the file parses fine but has no `python_sidecar` key at all,
/// matching every config.json that predates this feature. `Err` when the
/// base config itself can't be read/parsed as a JSON object (same failure
/// mode `prepare_config` already treats as fatal for the native path), OR
/// when a `python_sidecar` key IS present but missing `python`/`worker_dir`
/// or has the wrong JSON shape — refused loudly rather than silently
/// ignored, matching `--config-overrides-b64`'s "malformed input REFUSES"
/// convention (`pidlock.rs`'s `AgentController::start`).
pub fn read_config(config_path: &Path) -> Result<Option<PythonSidecarConfig>, SidecarConfigError> {
    let data = std::fs::read(config_path).map_err(|e| SidecarConfigError::Io(e.to_string()))?;
    let value: serde_json::Value =
        serde_json::from_slice(&data).map_err(|e| SidecarConfigError::Malformed(e.to_string()))?;
    let Some(raw) = value.get("python_sidecar") else {
        return Ok(None);
    };
    if raw.is_null() {
        return Ok(None);
    }
    let cfg: PythonSidecarConfig = serde_json::from_value(raw.clone())
        .map_err(|e| SidecarConfigError::Malformed(format!("invalid python_sidecar object: {e}")))?;
    if cfg.python.trim().is_empty() {
        return Err(SidecarConfigError::Malformed("`python` must not be empty".to_string()));
    }
    if cfg.worker_dir.trim().is_empty() {
        return Err(SidecarConfigError::Malformed("`worker_dir` must not be empty".to_string()));
    }
    Ok(Some(cfg))
}

/// `<worker_name>-py-<index>` (1-based) — see this module's doc comment for
/// why this scheme can't collide with farmctl's native `<worker_name>-<i>`.
pub fn instance_name(worker_name: &str, index: i64) -> String {
    format!("{worker_name}-py-{index}")
}

fn state_path(worker_directory: &Path, worker_name: &str) -> PathBuf {
    worker_directory.join(format!(".agent.{worker_name}.python_sidecars.json"))
}

/// Persists the list of sidecar instance names `start_sidecars` just
/// spawned, so a later `stop`/`status` (a SEPARATE, short-lived CLI
/// invocation — this agent keeps no in-memory state between calls) knows
/// which names to act on without being told `--fpp-instances` again.
pub fn write_state(worker_directory: &Path, worker_name: &str, names: &[String]) -> std::io::Result<()> {
    let data = serde_json::to_vec(names).expect("Vec<String> always serializes");
    std::fs::write(state_path(worker_directory, worker_name), data)
}

/// `None` when no sidecars were ever started for this `worker_name` (no
/// state file) or the file is unreadable/corrupt — treated as "nothing to
/// stop", not an error, since `stop` must remain a safe no-op on a
/// native-only worker (the common case: every host that doesn't use this
/// feature has no such file, ever).
pub fn read_state(worker_directory: &Path, worker_name: &str) -> Option<Vec<String>> {
    let data = std::fs::read(state_path(worker_directory, worker_name)).ok()?;
    serde_json::from_slice(&data).ok()
}

pub fn remove_state(worker_directory: &Path, worker_name: &str) {
    let _ = std::fs::remove_file(state_path(worker_directory, worker_name));
}

/// One sidecar `agent.py` subprocess's outcome — raw stdout captured for the
/// caller's JSON report, not parsed here (the Python `agent.py` contract is
/// "print one JSON object", but this module treats it as opaque text; the
/// Rust caller doesn't need to understand it to report it).
#[derive(Debug, Clone)]
pub struct SidecarOutcome {
    pub instance_name: String,
    pub raw_output: String,
    pub exit_code: i32,
}

fn build_python_agent_command(
    cfg: &PythonSidecarConfig,
    action: &str,
    instance_name: &str,
) -> Command {
    let agent_py = Path::new(&cfg.worker_dir).join("agent.py");
    let mut command = Command::new(&cfg.python);
    command.arg(agent_py).arg(action).arg(instance_name);
    for (key, value) in &cfg.env {
        command.env(key, value);
    }
    command.stdin(Stdio::null());
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    command
}

/// Spawns `count` `python worker.py` instances by shelling out to
/// `<worker_dir>/agent.py start <instance_name>` for each — the SAME
/// invocation shape `bin/farmctl`'s `agent_call` already builds for a
/// pure-Python manifest entry, just issued locally instead of over SSH.
/// `--instances`/`--instance-index` are only passed when `count > 1`
/// (matches `farmctl`'s own `_iter_instances`/`cmd_start`: a lone instance
/// stays on the bit-identical single-worker `config.json` path in
/// `agent.py`'s `cmd_start`, no derived config). `server_url` and
/// `config_overrides_b64` are forwarded VERBATIM — this Rust agent does not
/// need to understand either, only pass them through, since the Python
/// `agent.py` already implements both.
///
/// All `count` children are spawned (non-blocking `Command::spawn`) BEFORE
/// waiting on any of them, so the wall-clock cost is one Python startup
/// (~1s, `agent.py cmd_start`'s own sanity-wait), not `count` of them in
/// series — with `count` in the teens (a realistic `instances_fpp`), a
/// naive spawn-then-wait loop would risk tripping `bin/farmctl`'s 30s SSH
/// timeout on the ONE `agent start` call this whole method answers.
pub fn start_sidecars(
    cfg: &PythonSidecarConfig,
    worker_name: &str,
    count: i64,
    server_url: Option<&str>,
    config_overrides_b64: Option<&str>,
) -> std::io::Result<Vec<SidecarOutcome>> {
    let mut children: Vec<(String, Child)> = Vec::with_capacity(count.max(0) as usize);
    for i in 1..=count {
        let iname = instance_name(worker_name, i);
        let mut command = build_python_agent_command(cfg, "start", &iname);
        if let Some(url) = server_url {
            command.arg("--server-url").arg(url);
        }
        if count > 1 {
            command.arg("--instances").arg(count.to_string());
            command.arg("--instance-index").arg(i.to_string());
        }
        if let Some(overrides) = config_overrides_b64 {
            command.arg("--config-overrides-b64").arg(overrides);
        }
        command.arg("--allow-start");
        let child = command.spawn()?;
        children.push((iname, child));
    }
    wait_all(children)
}

/// Stops every name in `names` by shelling out to `agent.py stop <name>`,
/// same "spawn all, then wait all" concurrency as `start_sidecars` (a stop
/// also waits up to 10s per instance internally in `agent.py`'s own
/// `cmd_stop`, so series-waiting `instances_fpp` of them could be slow).
pub fn stop_sidecars(cfg: &PythonSidecarConfig, names: &[String], force: bool) -> std::io::Result<Vec<SidecarOutcome>> {
    let mut children: Vec<(String, Child)> = Vec::with_capacity(names.len());
    for name in names {
        let mut command = build_python_agent_command(cfg, "stop", name);
        if force {
            command.arg("--force");
        }
        let child = command.spawn()?;
        children.push((name.clone(), child));
    }
    wait_all(children)
}

fn wait_all(children: Vec<(String, Child)>) -> std::io::Result<Vec<SidecarOutcome>> {
    let mut outcomes = Vec::with_capacity(children.len());
    for (instance_name, child) in children {
        let output = child.wait_with_output()?;
        let mut raw_output = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if raw_output.is_empty() {
            raw_output = String::from_utf8_lossy(&output.stderr).trim().to_string();
        }
        outcomes.push(SidecarOutcome {
            instance_name,
            raw_output,
            exit_code: output.status.code().unwrap_or(-1),
        });
    }
    Ok(outcomes)
}

/// Reads a sidecar's own PID file (written by the Python `agent.py`'s
/// `cmd_start`, `.agent.<instance_name>.pid` under `worker_dir`) directly —
/// used by `AgentController::status` to report liveness without spawning
/// another `python agent.py status` subprocess per instance on every status
/// poll.
pub fn sidecar_pid(worker_dir: &Path, instance_name: &str) -> Option<i32> {
    let text = std::fs::read_to_string(Path::new(worker_dir).join(format!(".agent.{instance_name}.pid"))).ok()?;
    text.trim().parse().ok()
}
