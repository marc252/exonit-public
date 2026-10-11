//! Ported from `testWorkerLogFile` in
//! `Sources/ExoNitContractTests/main.swift` (Swift reference worker).

use exonit_core::logfile;
use std::path::Path;

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "exonit-rust-{name}-{}-{:x}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Swift: `testWorkerLogFile`.
#[test]
fn worker_log_file_sink() {
    let directory = temp_dir("logfile");
    let resolved = logfile::resolve(Some("logs/worker.log"), &directory).expect("relative log_file resolves");
    assert_eq!(resolved, directory.join("logs").join("worker.log"), "relative log_file resolution");

    logfile::append("first", &resolved).expect("append first");
    logfile::append("second", &resolved).expect("append second");
    let contents = std::fs::read_to_string(&resolved).expect("read log file");
    assert_eq!(contents, "first\nsecond\n", "log_file append sink");

    assert_eq!(logfile::resolve(None, &directory), None, "nil log_file disables sink");
    assert_eq!(logfile::resolve(Some("   "), &directory), None, "blank log_file disables sink");
}

/// Not in the Swift suite: exercises the `..`-collapsing lexical
/// standardization path (`WorkerLogFile.resolve`'s `standardizedFileURL`
/// mirror) that `worker_log_file_sink` above doesn't touch, since its
/// configured path has no `.`/`..` components.
#[test]
fn worker_log_file_standardizes_parent_references() {
    let directory = temp_dir("logfile-standardize");
    let resolved =
        logfile::resolve(Some("logs/../logs/worker.log"), &directory).expect("path with .. resolves");
    assert_eq!(resolved, directory.join("logs").join("worker.log"), "standardized path collapses ..");

    // `Path::is_absolute()` requires a drive-letter prefix on Windows — a
    // bare-rooted path like "/tmp/..." (no prefix) is NOT absolute there
    // (it means "rooted on the current drive"), so `logfile::resolve`
    // legitimately takes the join-with-relative_to branch for it instead
    // of the platform-independent-absolute one this test wants to exercise
    // (found failing on <host>: Task 2,
    // docs/plans/2026-08-06-win32-port-pcmarc.md — `logfile.rs` itself is
    // correct/portable, `Path::is_absolute()` is the idiomatic per-platform
    // check; the leading-`/`-with-no-drive-letter fixture was a Unix
    // assumption in this test, not a production bug). A drive-qualified
    // prefix makes the input genuinely absolute on both platforms; `Path`
    // equality is component-wise (separator-style-independent), so the
    // expected value can stay written with forward slashes throughout.
    #[cfg(windows)]
    const ABSOLUTE_PREFIX: &str = "C:";
    #[cfg(not(windows))]
    const ABSOLUTE_PREFIX: &str = "";

    let absolute_input = format!("{ABSOLUTE_PREFIX}/tmp/a/../b/worker.log");
    let absolute = logfile::resolve(Some(&absolute_input), &directory).expect("absolute path resolves");
    assert_eq!(
        absolute,
        Path::new(&format!("{ABSOLUTE_PREFIX}/tmp/b/worker.log")),
        "absolute path is standardized independent of relative_to"
    );
}
