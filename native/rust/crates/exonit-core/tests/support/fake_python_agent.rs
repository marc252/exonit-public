//! Fake `worker/agent.py` stand-in for `python_sidecar`/`pidlock` sidecar
//! tests (2026-08-07). Implements just enough of the REAL `agent.py`'s
//! `start`/`stop` contract (see `~/exonit/worker/agent.py`) to
//! exercise `python_sidecar::start_sidecars`/`stop_sidecars` against an
//! actual `python3` subprocess rather than a mock: `start <name> [flags...]`
//! spawns a detached long-lived child and writes `.agent.<name>.pid`;
//! `stop <name> [--force]` signals it, waits briefly, and removes the PID
//! file. Both print a one-line JSON object, matching the real script's
//! stdout contract closely enough for `AgentController`/`python_sidecar` to
//! report it verbatim without needing to understand it.
//!
//! Included via `#[path = "support/fake_python_agent.rs"]` into
//! `python_sidecar_tests.rs` and `pidlock_tests.rs` — both exercise the same
//! fake, at different layers (the module directly vs. through
//! `AgentController::execute`'s `--fpp-instances` path).
#![allow(dead_code)]

use std::path::{Path, PathBuf};

const FAKE_AGENT_PY: &str = r#"
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parent


def pid_file(name):
    return WORKER_DIR / f".agent.{name}.pid"


def alive(pid):
    # POSIX: os.kill(pid, 0). Windows: os.kill(pid, 0) means CTRL_C_EVENT
    # (WinError 87 on non-console-group pids), so probe via OpenProcess +
    # GetExitCodeProcess == STILL_ACTIVE instead.
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main():
    args = sys.argv[1:]
    if not args:
        print(json.dumps({"status": "error", "error": "missing action"}))
        return 1
    action = args[0]
    name = args[1] if len(args) > 1 else None
    rest = args[2:]
    force = "--force" in rest

    if action == "start":
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        pid_file(name).write_text(str(proc.pid))
        print(json.dumps({"status": "started", "pid": proc.pid, "worker_name": name}))
        return 0

    if action == "stop":
        pf = pid_file(name)
        if not pf.exists():
            print(json.dumps({"status": "not_running"}))
            return 0
        pid = int(pf.read_text().strip())
        # Windows has no SIGKILL; os.kill(pid, SIGTERM) is TerminateProcess
        # there, which is already the "force" semantics.
        if os.name == "nt":
            sig = signal.SIGTERM
        else:
            sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        deadline = time.time() + 5
        while time.time() < deadline:
            if not alive(pid):
                break
            time.sleep(0.05)
        pf.unlink(missing_ok=True)
        print(json.dumps({"status": "stopped", "pid": pid}))
        return 0

    print(json.dumps({"status": "error", "error": f"unknown action {action}"}))
    return 1


if __name__ == "__main__":
    sys.exit(main())
"#;

/// Writes the fake `agent.py` into `dir` (the fake sidecar `worker_dir`) and
/// returns its path.
pub fn write_fake_agent(dir: &Path) -> PathBuf {
    let path = dir.join("agent.py");
    std::fs::write(&path, FAKE_AGENT_PY).expect("write fake agent.py");
    path
}

/// Tests that spawn the fake skip cleanly (rather than failing) when
/// `python3` isn't on `PATH` — real hosts in this fleet always have one
/// (every worker host runs a Python venv already), but a bare Rust-only CI
/// runner might not.
pub fn python3_available() -> bool {
    std::process::Command::new("python3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
