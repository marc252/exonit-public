//! `python_sidecar` module tests (2026-08-07). Config parsing / state
//! round-trip are pure unit tests; `start_sidecars`/`stop_sidecars` are
//! exercised against a REAL `python3` subprocess via the fake `agent.py` in
//! `support/fake_python_agent.rs` (skips cleanly if `python3` isn't on
//! `PATH` — see that module).

#[path = "support/fake_python_agent.rs"]
mod fake_python_agent;

use exonit_core::platform;
use exonit_core::python_sidecar::{self, PythonSidecarConfig};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!("exonit-rust-tests-pysidecar-{label}-{nanos}-{n}"));
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

#[test]
fn read_config_absent_key_returns_none() {
    let dir = TempDir::new("no-key");
    let cfg_path = dir.path().join("config.json");
    std::fs::write(&cfg_path, r#"{"server_url":"http://x","token":"t"}"#).unwrap();
    assert_eq!(
        python_sidecar::read_config(&cfg_path).expect("read"),
        None,
        "config.json with no python_sidecar key must be the zero-change default"
    );
}

#[test]
fn read_config_parses_present_key() {
    let dir = TempDir::new("with-key");
    let cfg_path = dir.path().join("config.json");
    std::fs::write(
        &cfg_path,
        r#"{
            "server_url": "http://x", "token": "t",
            "python_sidecar": {
                "python": "/usr/bin/python3",
                "worker_dir": "/opt/worker",
                "env": {"A": "1"}
            }
        }"#,
    )
    .unwrap();
    let cfg = python_sidecar::read_config(&cfg_path).expect("read").expect("present");
    assert_eq!(cfg.python, "/usr/bin/python3");
    assert_eq!(cfg.worker_dir, "/opt/worker");
    assert_eq!(cfg.env.get("A").map(String::as_str), Some("1"));
}

#[test]
fn read_config_defaults_env_to_empty() {
    let dir = TempDir::new("no-env");
    let cfg_path = dir.path().join("config.json");
    std::fs::write(
        &cfg_path,
        r#"{"server_url":"http://x","token":"t","python_sidecar":{"python":"python3","worker_dir":"/opt/worker"}}"#,
    )
    .unwrap();
    let cfg = python_sidecar::read_config(&cfg_path).expect("read").expect("present");
    assert!(cfg.env.is_empty());
}

#[test]
fn read_config_rejects_malformed_python_sidecar() {
    let dir = TempDir::new("malformed");
    let cfg_path = dir.path().join("config.json");
    std::fs::write(
        &cfg_path,
        r#"{"server_url":"http://x","token":"t","python_sidecar":{"worker_dir":"/opt/worker"}}"#,
    )
    .unwrap();
    assert!(
        python_sidecar::read_config(&cfg_path).is_err(),
        "missing `python` must REFUSE, not silently ignore (matches --config-overrides-b64 convention)"
    );
}

#[test]
fn read_config_rejects_unreadable_base_config() {
    let dir = TempDir::new("missing-base");
    let cfg_path = dir.path().join("config.json"); // never written
    assert!(python_sidecar::read_config(&cfg_path).is_err());
}

#[test]
fn instance_naming_is_stable_and_distinct_from_native_scheme() {
    // farmctl's native multi-instance scheme is `<name>-<i>` (no "py"
    // infix) — see bin/farmctl's `_iter_instances`. This must never collide.
    assert_eq!(python_sidecar::instance_name("worker-<host>-rust", 1), "worker-<host>-rust-py-1");
    assert_eq!(python_sidecar::instance_name("worker-<host>-rust", 17), "worker-<host>-rust-py-17");
}

#[test]
fn state_round_trips_and_defaults_to_none() {
    let dir = TempDir::new("state");
    assert_eq!(python_sidecar::read_state(dir.path(), "worker-x"), None, "no state file yet");
    let names = vec!["worker-x-py-1".to_string(), "worker-x-py-2".to_string()];
    python_sidecar::write_state(dir.path(), "worker-x", &names).expect("write state");
    assert_eq!(python_sidecar::read_state(dir.path(), "worker-x"), Some(names));
    python_sidecar::remove_state(dir.path(), "worker-x");
    assert_eq!(python_sidecar::read_state(dir.path(), "worker-x"), None, "removed after remove_state");
}

#[test]
fn start_sidecars_with_zero_count_spawns_nothing() {
    let dir = TempDir::new("zero-count");
    let cfg = PythonSidecarConfig {
        python: "python3".to_string(),
        worker_dir: dir.path().to_string_lossy().to_string(),
        env: BTreeMap::new(),
    };
    let outcomes = python_sidecar::start_sidecars(&cfg, "worker-x", 0, None, None).expect("start_sidecars");
    assert!(outcomes.is_empty());
}

#[test]
fn stop_sidecars_with_no_names_is_a_noop() {
    let dir = TempDir::new("no-names");
    let cfg = PythonSidecarConfig {
        python: "python3".to_string(),
        worker_dir: dir.path().to_string_lossy().to_string(),
        env: BTreeMap::new(),
    };
    let outcomes = python_sidecar::stop_sidecars(&cfg, &[], false).expect("stop_sidecars");
    assert!(outcomes.is_empty());
}

/// End-to-end against a REAL `python3` process: start 2 sidecars (exercising
/// the "spawn all, then wait all" concurrency `start_sidecars` documents),
/// confirm both PIDs are alive, stop them, and confirm both PIDs are dead
/// and their PID files removed.
#[test]
fn start_then_stop_sidecars_spawns_and_kills_real_processes() {
    if !fake_python_agent::python3_available() {
        eprintln!("skipping start_then_stop_sidecars_spawns_and_kills_real_processes: no python3 on PATH");
        return;
    }
    let worker_dir = TempDir::new("sidecar-worker-dir");
    fake_python_agent::write_fake_agent(worker_dir.path());
    let cfg = PythonSidecarConfig {
        python: "python3".to_string(),
        worker_dir: worker_dir.path().to_string_lossy().to_string(),
        env: BTreeMap::new(),
    };

    let outcomes = python_sidecar::start_sidecars(&cfg, "worker-test-rust", 2, Some("http://127.0.0.1:9/"), None)
        .expect("start_sidecars");
    assert_eq!(outcomes.len(), 2);
    for outcome in &outcomes {
        assert_eq!(outcome.exit_code, 0, "fake agent.py start failed: {}", outcome.raw_output);
    }
    let names: Vec<String> = outcomes.iter().map(|o| o.instance_name.clone()).collect();
    assert_eq!(names, vec!["worker-test-rust-py-1", "worker-test-rust-py-2"]);

    let mut pids = Vec::new();
    for name in &names {
        let pid = python_sidecar::sidecar_pid(worker_dir.path(), name).expect("pid file written by fake agent.py");
        assert!(platform::process_alive(pid), "sidecar {name} (pid {pid}) should be alive right after start");
        pids.push(pid);
    }

    let stop_outcomes = python_sidecar::stop_sidecars(&cfg, &names, false).expect("stop_sidecars");
    assert_eq!(stop_outcomes.len(), 2);
    for outcome in &stop_outcomes {
        assert_eq!(outcome.exit_code, 0, "fake agent.py stop failed: {}", outcome.raw_output);
    }
    for pid in pids {
        assert!(!platform::process_alive(pid), "sidecar pid {pid} must be dead after stop");
    }
    for name in &names {
        assert!(
            python_sidecar::sidecar_pid(worker_dir.path(), name).is_none(),
            "PID file for {name} must be removed after stop"
        );
    }
}
