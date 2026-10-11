//! Transliterated from `Sources/ExoNitCore/WorkerLogFile.swift` in the Swift
//! reference worker (`~/exonit-swift-worker-dev`, commit `5e3978e`).
//! The on-disk log format is unchanged from the Python worker's
//! `worker-<name>.log` (append-only, one line per message; stdout/stderr
//! redirected there too once a `log_file` is configured) — the farmctl CLI
//! and the dashboard UI read it, so this must stay byte-compatible (see
//! `AgentController::logs` in `pidlock.rs`, already reading
//! `worker-<name>.log`).
//!
//! `redirect_standard_streams` itself now lives in `crate::platform`
//! (extracted in `docs/plans/2026-08-06-win32-port-pcmarc.md` Task 1, pure
//! refactor) — this module re-exports it so `main.rs`'s call site is
//! unchanged. Unix's implementation (`platform::unix::redirect_standard_streams`)
//! is the exact `libc::open`+`libc::dup2` code that used to live here
//! directly.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Component, Path, PathBuf};

/// Mirrors `WorkerLogFile.resolve(_:relativeTo:)`: `None` when the
/// configured path is absent/blank ("no log_file sink configured"), an
/// absolute path resolved standalone, or a relative path resolved against
/// `relative_to` — followed in both cases by lexical standardization
/// (mirrors `URL.standardizedFileURL`: collapses `.`/`..` components
/// textually, without touching the filesystem or following symlinks —
/// distinct from `resolvingSymlinksInPath`, which this is deliberately not).
pub fn resolve(configured_path: Option<&str>, relative_to: &Path) -> Option<PathBuf> {
    let trimmed = configured_path?.trim();
    if trimmed.is_empty() {
        return None;
    }
    let candidate = Path::new(trimmed);
    let joined = if candidate.is_absolute() { candidate.to_path_buf() } else { relative_to.join(candidate) };
    Some(standardize(&joined))
}

/// Mirrors `WorkerLogFile.append(_:to:fileManager:)`: creates the parent
/// directory and the file itself if missing, then appends `message + "\n"`.
pub fn append(message: &str, path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    file.write_all(message.as_bytes())?;
    file.write_all(b"\n")?;
    Ok(())
}

/// Mirrors `WorkerLogFile.redirectStandardStreams(to:fileManager:)`: opens
/// (creating/appending) the log file and redirects both stdout and stderr
/// onto it, so everything the process subsequently prints — including
/// output from code that writes straight to fd 1/2 rather than through
/// Rust's `stdout()` — lands in the same file the CLI/UI already tail.
/// Delegates to `crate::platform::redirect_standard_streams` (see this
/// module's doc comment).
pub fn redirect_standard_streams(path: &Path) -> std::io::Result<()> {
    crate::platform::redirect_standard_streams(path)
}

/// Lexical path standardization: drops `.` components and resolves `..`
/// against the preceding component when there is one to pop (mirrors
/// `URL.standardizedFileURL`'s textual normalization — no filesystem access,
/// no symlink resolution).
fn standardize(path: &Path) -> PathBuf {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                let popped = match result.components().next_back() {
                    Some(Component::Normal(_)) => {
                        result.pop();
                        true
                    }
                    _ => false,
                };
                if !popped {
                    result.push(component.as_os_str());
                }
            }
            other => result.push(other.as_os_str()),
        }
    }
    result
}
