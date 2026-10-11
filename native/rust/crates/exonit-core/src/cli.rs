//! Transliterated from `Sources/ExoNitCore/Invocation.swift` +
//! `Sources/ExoNitCore/CLIHelp.swift` in the Swift reference worker
//! (`~/exonit-swift-worker-dev`, commit `5e3978e`).
//!
//! Parses the two CLI shapes the fleet already speaks:
//!   - direct worker invocation: `exonit-rust-worker [--config PATH] [--worker-name NAME] [--server-url URL]`
//!     (mirrors `worker.py`'s `argparse` exactly — see `~/exonit/worker/worker.py`).
//!   - farmctl-compatible agent invocation, detected when argv[0]'s basename is
//!     `agent.py`: `agent.py start|stop|status|logs|version [worker-name] [options]`
//!     (mirrors `~/exonit/worker/agent.py`'s `argparse` subcommands).
//!
//! No `clap` dependency per the F0 task constraints — hand-rolled parsing,
//! same as the Swift reference (which also hand-rolls it rather than using
//! `ArgumentParser`).

use std::fmt;
use std::path::Path;

/// Validates worker names used as filesystem path components (PID files,
/// derived config files, log files) — must never allow path traversal.
/// Mirrors `WorkerNameValidator` in `Invocation.swift`, which uses the regex
/// `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$` plus a `..` substring ban. Hand-rolled
/// here (byte-by-byte) since no `regex` dependency is authorized for F0.
pub mod worker_name_validator {
    pub fn is_valid(value: &str) -> bool {
        let bytes = value.as_bytes();
        if bytes.is_empty() || bytes.len() > 128 {
            return false;
        }
        if !bytes[0].is_ascii_alphanumeric() {
            return false;
        }
        for &b in &bytes[1..] {
            let ok = b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-';
            if !ok {
                return false;
            }
        }
        !value.contains("..")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Invocation {
    Worker(WorkerArguments),
    Agent(AgentArguments),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerArguments {
    pub config_path: String,
    pub worker_name: Option<String>,
    pub server_url: Option<String>,
}

impl Default for WorkerArguments {
    fn default() -> Self {
        WorkerArguments {
            config_path: "config.json".to_string(),
            worker_name: None,
            server_url: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentAction {
    Start,
    Stop,
    Status,
    Logs,
    Version,
}

impl AgentAction {
    fn parse(raw: &str) -> Option<AgentAction> {
        match raw {
            "start" => Some(AgentAction::Start),
            "stop" => Some(AgentAction::Stop),
            "status" => Some(AgentAction::Status),
            "logs" => Some(AgentAction::Logs),
            "version" => Some(AgentAction::Version),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            AgentAction::Start => "start",
            AgentAction::Stop => "stop",
            AgentAction::Status => "status",
            AgentAction::Logs => "logs",
            AgentAction::Version => "version",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AgentArguments {
    pub action: AgentAction,
    pub worker_name: Option<String>,
    pub server_url: Option<String>,
    pub instances: i64,
    pub instance_index: i64,
    pub config_overrides_base64: Option<String>,
    pub allow_start: bool,
    pub force: bool,
    pub log_lines: i64,
    /// Python FPP sidecar count (2026-08-07, see `python_sidecar` module and
    /// docs/PLAN.md). Zero (the default, and what every pre-existing caller
    /// produces since this flag is new) means "no sidecars" — `start`/`stop`
    /// behave EXACTLY as before this field existed. When positive on
    /// `start`, `AgentController` spawns this many `python worker.py`
    /// instances (via the Python fleet's own `worker/agent.py`) instead of a
    /// native worker, for hosts whose Rust worker has no FPP/TRICERATOPS
    /// implementation (`WorkerRuntimeError::BackfillPoolsUnimplemented`).
    /// `stop`/`status` don't need this flag themselves — they read back the
    /// sidecar instance names `start` persisted — but farmctl may still pass
    /// it as a fallback hint (see `AgentController::stop`'s doc comment).
    pub fpp_instances: i64,
}

impl AgentArguments {
    pub fn new(action: AgentAction) -> Self {
        AgentArguments {
            action,
            worker_name: None,
            server_url: None,
            instances: 1,
            instance_index: 1,
            config_overrides_base64: None,
            allow_start: false,
            force: false,
            log_lines: 50,
            fpp_instances: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvocationParseError {
    MissingValue(String),
    UnknownOption(String),
    InvalidInteger { option: String, value: String },
    InvalidAgentAction(String),
    MissingWorkerName(AgentAction),
    InvalidWorkerName(String),
    UnexpectedArgument(String),
}

impl fmt::Display for InvocationParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InvocationParseError::MissingValue(option) => write!(f, "Missing value for {option}"),
            InvocationParseError::UnknownOption(option) => write!(f, "Unknown option {option}"),
            InvocationParseError::InvalidInteger { option, value } => {
                write!(f, "Invalid integer for {option}: {value}")
            }
            InvocationParseError::InvalidAgentAction(value) => {
                write!(f, "Invalid agent action: {value}")
            }
            InvocationParseError::MissingWorkerName(action) => {
                write!(f, "Agent action {} requires a worker name", action.as_str())
            }
            InvocationParseError::InvalidWorkerName(value) => {
                write!(f, "Invalid worker name: {value}")
            }
            InvocationParseError::UnexpectedArgument(value) => {
                write!(f, "Unexpected argument: {value}")
            }
        }
    }
}
impl std::error::Error for InvocationParseError {}

/// Basename of a path-like string, mirroring `URL(fileURLWithPath:).lastPathComponent`
/// closely enough for the one use here (detecting `agent.py` in argv[0]).
fn last_path_component(value: &str) -> String {
    Path::new(value)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| value.to_string())
}

fn validate_worker_name(value: &str) -> Result<(), InvocationParseError> {
    if worker_name_validator::is_valid(value) {
        Ok(())
    } else {
        Err(InvocationParseError::InvalidWorkerName(value.to_string()))
    }
}

/// Consumes the option's value from `arguments[index + 1]`, advancing `index`
/// to point at the value itself (the caller's `index += 1` after the match
/// then steps past it) — mirrors `InvocationParser.value(after:at:in:)`.
fn take_value(
    option: &str,
    index: &mut usize,
    arguments: &[String],
) -> Result<String, InvocationParseError> {
    let next = *index + 1;
    if next >= arguments.len() {
        return Err(InvocationParseError::MissingValue(option.to_string()));
    }
    *index = next;
    Ok(arguments[next].clone())
}

fn positive_integer(raw: &str, option: &str) -> Result<i64, InvocationParseError> {
    match raw.parse::<i64>() {
        Ok(value) if value > 0 => Ok(value),
        _ => Err(InvocationParseError::InvalidInteger {
            option: option.to_string(),
            value: raw.to_string(),
        }),
    }
}

pub fn parse(arguments: &[String]) -> Result<Invocation, InvocationParseError> {
    if let Some(first) = arguments.first() {
        if last_path_component(first) == "agent.py" {
            return Ok(Invocation::Agent(parse_agent(&arguments[1..])?));
        }
    }
    Ok(Invocation::Worker(parse_worker(arguments)?))
}

fn parse_worker(arguments: &[String]) -> Result<WorkerArguments, InvocationParseError> {
    let mut result = WorkerArguments::default();
    let mut index = 0usize;
    while index < arguments.len() {
        let option = arguments[index].as_str();
        match option {
            "--config" | "--worker-name" | "--server-url" => {
                let value = take_value(option, &mut index, arguments)?;
                match option {
                    "--config" => result.config_path = value,
                    "--worker-name" => result.worker_name = Some(value),
                    _ => result.server_url = Some(value),
                }
            }
            _ => {
                if let Some(v) = option.strip_prefix("--config=") {
                    result.config_path = v.to_string();
                } else if let Some(v) = option.strip_prefix("--worker-name=") {
                    result.worker_name = Some(v.to_string());
                } else if let Some(v) = option.strip_prefix("--server-url=") {
                    result.server_url = Some(v.to_string());
                } else if option.starts_with('-') {
                    return Err(InvocationParseError::UnknownOption(option.to_string()));
                } else {
                    return Err(InvocationParseError::UnexpectedArgument(option.to_string()));
                }
            }
        }
        index += 1;
    }
    if let Some(name) = &result.worker_name {
        validate_worker_name(name)?;
    }
    Ok(result)
}

fn parse_agent(arguments: &[String]) -> Result<AgentArguments, InvocationParseError> {
    let raw_action = arguments
        .first()
        .ok_or_else(|| InvocationParseError::MissingValue("agent action".to_string()))?;
    let action = AgentAction::parse(raw_action)
        .ok_or_else(|| InvocationParseError::InvalidAgentAction(raw_action.clone()))?;
    let mut result = AgentArguments::new(action);
    let mut index = 1usize;

    if index < arguments.len() && !arguments[index].starts_with('-') {
        result.worker_name = Some(arguments[index].clone());
        index += 1;
    }

    while index < arguments.len() {
        let option = arguments[index].as_str();
        match option {
            "--server-url" | "--instances" | "--instance-index" | "--config-overrides-b64"
            | "--fpp-instances" | "-n" => {
                let raw = take_value(option, &mut index, arguments)?;
                match option {
                    "--server-url" => result.server_url = Some(raw),
                    "--config-overrides-b64" => result.config_overrides_base64 = Some(raw),
                    "--instances" => result.instances = positive_integer(&raw, option)?,
                    "--instance-index" => result.instance_index = positive_integer(&raw, option)?,
                    "--fpp-instances" => {
                        let n: i64 = raw.parse().map_err(|_| InvocationParseError::InvalidInteger {
                            option: option.to_string(),
                            value: raw.clone(),
                        })?;
                        if n < 0 {
                            return Err(InvocationParseError::InvalidInteger {
                                option: option.to_string(),
                                value: raw,
                            });
                        }
                        result.fpp_instances = n;
                    }
                    _ => {
                        let n: i64 = raw.parse().map_err(|_| InvocationParseError::InvalidInteger {
                            option: option.to_string(),
                            value: raw.clone(),
                        })?;
                        if n < 0 {
                            return Err(InvocationParseError::InvalidInteger {
                                option: option.to_string(),
                                value: raw,
                            });
                        }
                        result.log_lines = n;
                    }
                }
            }
            "--allow-start" => result.allow_start = true,
            "--force" => result.force = true,
            _ => {
                if option.starts_with('-') {
                    return Err(InvocationParseError::UnknownOption(option.to_string()));
                }
                return Err(InvocationParseError::UnexpectedArgument(option.to_string()));
            }
        }
        index += 1;
    }

    if matches!(
        action,
        AgentAction::Start | AgentAction::Stop | AgentAction::Status
    ) && result.worker_name.is_none()
    {
        return Err(InvocationParseError::MissingWorkerName(action));
    }
    if let Some(name) = &result.worker_name {
        validate_worker_name(name)?;
    }
    Ok(result)
}

/// Mirrors `CLIHelp.text(for:)`.
pub fn help_text(arguments: &[String]) -> String {
    if let Some(first) = arguments.first() {
        if last_path_component(first) == "agent.py" {
            return "Usage: exonit-rust-worker agent.py start|stop|status|logs|version [worker-name] [options]\n\n  start <worker-name> [--server-url URL] [--instances N]\n        [--instance-index I] [--config-overrides-b64 DATA]\n        [--fpp-instances M] --allow-start\n  stop <worker-name> [--force]\n  status <worker-name>\n  logs [worker-name] [-n N]\n  version\n\nStarting requires --allow-start or EXONIT_FARMCTL_ALLOW_START=1.\n\n--fpp-instances M (2026-08-07): spawns M Python FPP-sidecar workers\n(via the Python fleet's own worker/agent.py, per this host's config.json\n`python_sidecar` section: {\"python\":...,\"worker_dir\":...,\"env\":{...}})\ninstead of a native worker -- for hosts whose Rust worker has no\nFPP/TRICERATOPS implementation. Omit it for the unchanged native-only\nbehaviour. stop/status pick sidecars back up automatically (state\nrecorded by start), no flag needed.\n".to_string();
        }
    }
    "Usage: exonit-rust-worker [--config PATH] [--worker-name NAME] [--server-url URL]\n\nNative ExoNit worker for <host> (Linux). The executable also accepts\nthe farmctl compatibility form where the first argument is agent.py.\n\nDirect worker execution (real FITS -> BLS -> vetting -> CNN/ExoMiner FFI\npipeline) requires EXONIT_RUST_CANARY=1 (or \"true\"); without it the process\nrefuses to run and exits 78 (\"fases incompletes\" -- CUDA and the F6\nproduction gate are not done yet, see docs/PLAN.md). ONNX models are loaded\nfrom EXONIT_MODELS_DIR, or the config's models_dir, or <executable_dir>/models\nif neither is set; a missing/absent models directory degrades to CNN/ExoMiner\nscores being absent, not a startup failure.\n".to_string()
}
