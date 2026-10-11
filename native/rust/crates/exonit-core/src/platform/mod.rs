//! Platform abstraction layer for everything `pidlock.rs` / `logfile.rs` /
//! `runtime.rs` / `telemetry.rs` / `exonit-worker`'s `main.rs` need from the
//! host OS: process liveness/identity, detached spawning, stop signalling,
//! a startup lock, stdout/stderr redirection, the "shutdown requested"
//! watch, and host telemetry.
//!
//! Extracted mechanically (Task 1 of
//! `docs/plans/2026-08-06-win32-port-pcmarc.md`) from what used to be
//! Linux-only code inline in those modules — see `unix.rs`'s module doc
//! comment for exactly where each piece came from. This is a **pure
//! refactor**: the Unix behavior is byte-for-byte identical to before
//! (`cargo test --all` is the gate, not a new test). `platform/windows.rs`
//! (Task 2) is not part of this change.
//!
//! Every function below has the SAME signature on every platform that
//! implements it, so `AgentController`/`WorkerRuntime`/`FitsCache`/
//! `main.rs` call `platform::*` without any `#[cfg(...)]` of their own.

use std::path::Path;

#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub use unix::*;

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::*;

/// Which signal `request_stop` should send (or its Windows-side equivalent —
/// see `docs/plans/2026-08-06-win32-port-pcmarc.md`'s Global Constraints for
/// the stop-file design). Shared across platforms: the distinction is a
/// plain enum, not OS-specific code, so it lives here rather than being
/// duplicated per `#[cfg(...)]` implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopKind {
    Soft,
    Force,
}

/// Sets `path`'s mtime (and, incidentally, atime) to "now". Portable single
/// implementation via `std::fs::File::set_modified` (stable since Rust
/// 1.75; `SetFileTime` under the hood on Windows, `futimens`-equivalent on
/// Unix) — **no `#[cfg(...)]` split**, unlike everything else in this
/// module. This is the one documented behavior change of Task 1: it
/// replaces a `libc::utimensat(UTIME_NOW)` call that only ever existed
/// because `libc::utimensat` isn't available on Windows, not because the
/// old and new implementations differ in what they do (both just bump
/// mtime to "now" for `FitsCache`'s LRU eviction). Best-effort: a failure to
/// open/touch the file is silently ignored, matching the previous
/// `libc::utimensat` call's return value being discarded too.
pub fn touch_mtime(path: &Path) {
    if let Ok(file) = std::fs::OpenOptions::new().write(true).open(path) {
        let _ = file.set_modified(std::time::SystemTime::now());
    }
}

/// `"posix"` on Unix, `"windows"` on Windows — mirrors the Global
/// Constraints decision in `docs/plans/2026-08-06-win32-port-pcmarc.md`
/// (`if cfg!(windows) { "windows" } else { "posix" }`). Used by
/// `AgentController::execute`'s `version` action, which used to hardcode
/// `"platform": "posix"`.
pub fn platform_label() -> &'static str {
    if cfg!(windows) {
        "windows"
    } else {
        "posix"
    }
}
