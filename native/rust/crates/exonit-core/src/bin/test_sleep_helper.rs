//! Test-only helper binary, not part of the ExoNit control plane: a
//! stand-in "long-lived worker process" for `pidlock_tests.rs`'s
//! `agent_concurrent_start` test (ported from the Swift suite's
//! `testAgentConcurrentStart`). That Swift test re-executes its own test
//! binary (`CommandLine.arguments[0]`) with an `EXONIT_AGENT_TEST_CHILD=1`
//! env var short-circuit so it just sleeps instead of re-running the whole
//! contract suite. Rust's `cargo test` harness doesn't offer an equivalent
//! "reinvoke myself as a plain process" hook, so this crate ships a tiny
//! dedicated binary instead: it ignores all arguments (matching what
//! `AgentController::spawn_detached` actually passes — `--config PATH
//! --worker-name NAME [--server-url URL]`, which this binary has no need to
//! parse) and just sleeps, long enough for the concurrent-start race in the
//! test to observe it as "still alive".
fn main() {
    std::thread::sleep(std::time::Duration::from_secs(30));
}
