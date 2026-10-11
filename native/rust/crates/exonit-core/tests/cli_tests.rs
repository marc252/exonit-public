//! Ported from the CLI-parsing tests in
//! `Sources/ExoNitContractTests/main.swift` (Swift reference worker).

use exonit_core::cli::{self, AgentAction, Invocation, InvocationParseError};

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

/// Swift: `testDirectWorkerInvocation`.
#[test]
fn direct_worker_invocation() {
    let args = s(&[
        "--config",
        "/tmp/worker.json",
        "--worker-name",
        "worker-<host>-1",
        "--server-url",
        "http://127.0.0.1:18400",
    ]);
    let invocation = cli::parse(&args).expect("parse");
    let Invocation::Worker(worker_args) = invocation else {
        panic!("expected worker invocation");
    };
    assert_eq!(worker_args.config_path, "/tmp/worker.json");
    assert_eq!(worker_args.worker_name.as_deref(), Some("worker-<host>-1"));
    assert_eq!(worker_args.server_url.as_deref(), Some("http://127.0.0.1:18400"));
}

/// Swift: `testAgentStartInvocation`.
#[test]
fn agent_start_invocation() {
    let args = s(&[
        "~/exonit-worker/agent.py",
        "start",
        "worker-<host>-1",
        "--server-url",
        "http://127.0.0.1:18400",
        "--instances",
        "4",
        "--instance-index",
        "1",
        "--config-overrides-b64",
        "e30=",
        "--allow-start",
    ]);
    let invocation = cli::parse(&args).expect("parse");
    let Invocation::Agent(agent_args) = invocation else {
        panic!("expected agent invocation");
    };
    assert_eq!(agent_args.action, AgentAction::Start);
    assert_eq!(agent_args.worker_name.as_deref(), Some("worker-<host>-1"));
    assert_eq!(agent_args.server_url.as_deref(), Some("http://127.0.0.1:18400"));
    assert_eq!(agent_args.instances, 4);
    assert_eq!(agent_args.instance_index, 1);
    assert_eq!(agent_args.config_overrides_base64.as_deref(), Some("e30="));
    assert!(agent_args.allow_start);
}

/// Swift: `testUnknownOption`.
#[test]
fn unknown_option_fails() {
    let args = s(&["--wat"]);
    assert!(cli::parse(&args).is_err(), "unknown option must fail");
}

/// Swift: `testMissingAgentWorkerName`.
#[test]
fn agent_start_requires_name() {
    let args = s(&["agent.py", "start", "--allow-start"]);
    let result = cli::parse(&args);
    assert!(matches!(
        result,
        Err(InvocationParseError::MissingWorkerName(AgentAction::Start))
    ));
}

/// Swift: `testWorkerNamePathSafety`.
#[test]
fn worker_name_path_safety() {
    let unsafe_names = ["../victim", "a/b", ".hidden", "worker..other", "", &"x".repeat(129)];
    for unsafe_name in unsafe_names {
        let args = s(&["agent.py", "stop", unsafe_name]);
        assert!(
            cli::parse(&args).is_err(),
            "unsafe worker name must fail: {unsafe_name}"
        );
    }
    let args = s(&["agent.py", "status", "worker-m4_1.test"]);
    let invocation = cli::parse(&args).expect("safe agent parse");
    let Invocation::Agent(agent_args) = invocation else {
        panic!("expected agent invocation");
    };
    assert_eq!(agent_args.worker_name.as_deref(), Some("worker-m4_1.test"));
}

/// Swift: `testCLIHelp`.
#[test]
fn cli_help_text() {
    let worker = cli::help_text(&[]);
    assert!(worker.contains("Usage: exonit-rust-worker"));
    assert!(worker.contains("--config"));
    let agent = cli::help_text(&s(&["/tmp/agent.py"]));
    assert!(agent.contains("agent.py start|stop|status|logs|version"));
    assert!(agent.contains("--allow-start"));
}

/// Additional case (no direct Swift counterpart): `-n` accepts zero (matches
/// `agent.py`'s own `guard let n = Int(raw), n>=0` — zero lines is valid,
/// unlike `--instances`/`--instance-index` which require strictly positive).
#[test]
fn logs_line_count_accepts_zero() {
    let args = s(&["agent.py", "logs", "worker-test", "-n", "0"]);
    let invocation = cli::parse(&args).expect("parse");
    let Invocation::Agent(agent_args) = invocation else {
        panic!("expected agent invocation");
    };
    assert_eq!(agent_args.log_lines, 0);
}

/// Additional case: negative `-n` must fail (mirrors the Swift
/// `n>=0`-guarded branch, not exercised directly by the ported suite).
#[test]
fn logs_line_count_rejects_negative() {
    let args = s(&["agent.py", "logs", "worker-test", "-n", "-1"]);
    assert!(cli::parse(&args).is_err());
}

/// 2026-08-07 (python-fpp-sidecar): `--fpp-instances` parses like
/// `--instances`, defaults to zero (the "no change" default — see
/// `AgentArguments::new`), and rejects negatives the same way `-n` does.
#[test]
fn fpp_instances_parses_and_defaults_to_zero() {
    let no_flag = s(&["agent.py", "start", "worker-<host>-rust", "--allow-start"]);
    let Invocation::Agent(agent_args) = cli::parse(&no_flag).expect("parse") else {
        panic!("expected agent invocation");
    };
    assert_eq!(agent_args.fpp_instances, 0, "omitted flag must default to 0 (zero behaviour change)");

    let with_flag = s(&[
        "agent.py",
        "start",
        "worker-<host>-rust",
        "--fpp-instances",
        "17",
        "--allow-start",
    ]);
    let Invocation::Agent(agent_args) = cli::parse(&with_flag).expect("parse") else {
        panic!("expected agent invocation");
    };
    assert_eq!(agent_args.fpp_instances, 17);
}

#[test]
fn fpp_instances_rejects_negative() {
    let args = s(&["agent.py", "start", "worker-test", "--fpp-instances", "-1", "--allow-start"]);
    assert!(cli::parse(&args).is_err());
}
