//! Unix implementation of the `platform` abstraction layer. Every function
//! here is code MOVED verbatim (no logic change) out of its previous home:
//!
//!   - `process_executable_path`, `process_command_line`, `process_alive`:
//!     formerly `default_process_executable_path`,
//!     `default_process_command_line`, `process_alive` in `pidlock.rs`
//!     (only renamed to drop the `default_` prefix, since here they ARE the
//!     platform's canonical implementation, not a default some other
//!     implementation could override).
//!   - `configure_detached`: formerly the inline `pre_exec(setsid)` block in
//!     `pidlock.rs`'s `spawn_detached`.
//!   - `request_stop`/`kill_now`: formerly the `libc::kill(pid, SIGTERM/
//!     SIGKILL)` calls inline in `pidlock.rs`'s `stop()` (the 10s wait loop
//!     between them stays in `pidlock.rs` — it's OS-agnostic logic, not a
//!     platform primitive).
//!   - `StartupLock`/`acquire_startup_lock`: formerly the `open`+`flock`
//!     open/lock/unlock/close sequence inline in `pidlock.rs`'s
//!     `with_startup_lock`.
//!   - `redirect_standard_streams`: formerly `logfile.rs`'s function of the
//!     same name.
//!   - `install_shutdown_watch`: formerly `main.rs`'s
//!     `install_signal_handlers` + `handle_shutdown_signal` +
//!     `SHUTDOWN_REQUESTED` static.
//!   - `HostTelemetry`: formerly `telemetry.rs`'s `Telemetry` struct's
//!     sampling logic (`/proc/stat`, `/proc/meminfo`, `/proc/self/status`
//!     parsing) — `telemetry.rs` keeps `SystemSnapshot`/`to_json_map` (pure
//!     data, no OS calls) and now delegates sampling to this struct.
//!   - `hostname`/`cpu_model`: formerly `exonit-worker/src/bin/
//!     exonit_bench.rs`'s private functions of the same name.
//!
//! See `docs/plans/2026-08-06-win32-port-pcmarc.md` Task 1 and
//! `docs/plans/2026-08-06-win32-surface-report.md` for the exact
//! file:line provenance of each piece.

use crate::platform::StopKind;
use crate::telemetry::SystemSnapshot;
use std::ffi::CString;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

// ---------------------------------------------------------------------------
// Process identity / liveness — moved from pidlock.rs
// ---------------------------------------------------------------------------

/// Linux analogue of Swift's macOS-only `proc_pidpath(3)`: resolves a live
/// PID to its executable path via `/proc/<pid>/exe`. Returns `None` on any
/// failure (process gone, permission denied, `/proc` unavailable) — the
/// caller treats that as "unverifiable", same as a `proc_pidpath` failure.
pub fn process_executable_path(pid: i32) -> Option<PathBuf> {
    std::fs::read_link(format!("/proc/{pid}/exe")).ok()
}

/// Linux analogue of Swift's KERN_PROCARGS2 lookup: argv from
/// `/proc/<pid>/cmdline` (NUL-separated). Returns `None` on any failure.
pub fn process_command_line(pid: i32) -> Option<Vec<String>> {
    let bytes = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    if bytes.is_empty() {
        return None;
    }
    Some(
        bytes
            .split(|byte| *byte == 0)
            .filter(|part| !part.is_empty())
            .map(|part| String::from_utf8_lossy(part).into_owned())
            .collect(),
    )
}

pub fn process_alive(pid: i32) -> bool {
    let result = unsafe { libc::kill(pid, 0) };
    if result == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

// ---------------------------------------------------------------------------
// Spawn / stop — moved from pidlock.rs
// ---------------------------------------------------------------------------

/// Configures `cmd` so the spawned child survives this short-lived CLI
/// process exiting: `setsid()` in a `pre_exec` hook, detaching it into a new
/// session (mirrors `AgentController.spawnDetached`'s `posix_spawn` with
/// `POSIX_SPAWN_SETSID`).
pub fn configure_detached(cmd: &mut std::process::Command) {
    unsafe {
        cmd.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

/// Unix's `configure_detached` (`setsid()`) has no job-object-style failure
/// mode to retry around — this exists only so `pidlock.rs::spawn_detached`
/// can call it unconditionally without a `#[cfg(windows)]` of its own; it
/// is never actually invoked here since `is_breakaway_access_denied` always
/// returns `false` on this platform. See `platform::windows`'s doc comment
/// on `CREATE_BREAKAWAY_FROM_JOB` for what this pair is for on Windows.
pub fn configure_detached_without_breakaway(cmd: &mut std::process::Command) {
    configure_detached(cmd);
}

/// Always `false` on Unix — see `configure_detached_without_breakaway`'s
/// doc comment.
pub fn is_breakaway_access_denied(_err: &std::io::Error) -> bool {
    false
}

/// Sends the stop signal for `pid` (`worker_name`/`worker_dir` are unused on
/// Unix — the Windows implementation needs them to name/locate its stop
/// file; see `docs/plans/2026-08-06-win32-port-pcmarc.md`).
pub fn request_stop(pid: i32, kind: StopKind, _worker_name: &str, _worker_dir: &Path) -> std::io::Result<()> {
    let signal = match kind {
        StopKind::Soft => libc::SIGTERM,
        StopKind::Force => libc::SIGKILL,
    };
    let result = unsafe { libc::kill(pid, signal) };
    if result != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// The final escalation of a soft stop that didn't exit within the timeout
/// (or the direct effect of `--force`, sent instead via `request_stop`).
pub fn kill_now(pid: i32) {
    unsafe {
        libc::kill(pid, libc::SIGKILL);
    }
}

// ---------------------------------------------------------------------------
// Startup lock — moved from pidlock.rs's `with_startup_lock`
// ---------------------------------------------------------------------------

/// Holds an exclusive `flock` on the underlying fd for as long as it lives;
/// `Drop` unlocks and closes it. The kernel also releases the lock if the
/// process dies holding it (stale-safe, same semantics `with_startup_lock`
/// relied on before this extraction).
pub struct StartupLock {
    fd: i32,
}

impl Drop for StartupLock {
    fn drop(&mut self) {
        unsafe {
            libc::flock(self.fd, libc::LOCK_UN);
            libc::close(self.fd);
        }
    }
}

/// Opens (creating if needed) and takes an exclusive `flock` on `path`,
/// retrying on `EINTR`. Mirrors `AgentController.withStartupLock`'s
/// open+flock sequence exactly.
pub fn acquire_startup_lock(path: &Path) -> std::io::Result<StartupLock> {
    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_CREAT | libc::O_RDWR | libc::O_CLOEXEC, 0o600) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    loop {
        let rc = unsafe { libc::flock(fd, libc::LOCK_EX) };
        if rc == 0 {
            break;
        }
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::EINTR) {
            continue;
        }
        unsafe {
            libc::close(fd);
        }
        return Err(err);
    }
    Ok(StartupLock { fd })
}

// ---------------------------------------------------------------------------
// Log redirection — moved from logfile.rs
// ---------------------------------------------------------------------------

/// Opens (creating/appending) the log file and `dup2`s both stdout and
/// stderr onto it, so everything the process subsequently prints —
/// including output from code that writes straight to fd 1/2 rather than
/// through Rust's `stdout()` — lands in the same file the CLI/UI already
/// tail.
pub fn redirect_standard_streams(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let c_path = CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    let descriptor = unsafe {
        libc::open(c_path.as_ptr(), libc::O_WRONLY | libc::O_CREAT | libc::O_APPEND | libc::O_CLOEXEC, 0o644)
    };
    if descriptor < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let result = unsafe {
        if libc::dup2(descriptor, libc::STDOUT_FILENO) < 0 {
            -1
        } else if libc::dup2(descriptor, libc::STDERR_FILENO) < 0 {
            -1
        } else {
            0
        }
    };
    unsafe {
        libc::close(descriptor);
    }
    if result < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shutdown watch — moved from exonit-worker/src/main.rs
// ---------------------------------------------------------------------------

/// Async-signal-safe handler: the only thing done inside the signal handler
/// itself is an atomic store (no allocation, no locking, no I/O) — the
/// actual shutdown request is delivered by a plain watcher thread polling
/// this flag, matching the async-signal-safety requirement that a
/// `SIGTERM`/`SIGINT` handler cannot safely call into arbitrary Rust code
/// (e.g. `Mutex::lock`, which `WorkerRuntimeState` uses internally, could
/// deadlock if the signal lands while the lock is held).
static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_shutdown_signal(_signal: libc::c_int) {
    SHUTDOWN_REQUESTED.store(true, Ordering::SeqCst);
}

/// Installs the source of "shutdown requested" (SIGINT/SIGTERM handlers on
/// Unix) and returns the flag a caller-owned watcher thread should poll.
/// `worker_name`/`worker_dir` are unused on Unix — Windows additionally
/// watches for a `.agent.<worker_name>.stop` file under `worker_dir` (see
/// `docs/plans/2026-08-06-win32-port-pcmarc.md`'s Global Constraints), so
/// both platforms share this signature.
pub fn install_shutdown_watch(_worker_name: &str, _worker_dir: &Path) -> &'static AtomicBool {
    unsafe {
        libc::signal(libc::SIGINT, handle_shutdown_signal as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, handle_shutdown_signal as *const () as libc::sighandler_t);
    }
    &SHUTDOWN_REQUESTED
}

/// No-op on Unix: signals are the only shutdown source (`SIGINT`/`SIGTERM`,
/// installed by `install_shutdown_watch` above). The stop-file poll this
/// mirrors is a Windows-only primitive (`platform::windows`'s equivalent —
/// see its doc comment) — kept here, always returning `false`, purely so
/// `main.rs`'s watcher thread can call it unconditionally without a
/// `#[cfg(windows)]` of its own (Task 2 of
/// `docs/plans/2026-08-06-win32-port-pcmarc.md`; this is the one place that
/// touches Linux's watcher loop, and the added call is a cheap no-op with
/// zero observable behavior change).
pub fn shutdown_requested_extra_check() -> bool {
    false
}

// ---------------------------------------------------------------------------
// Host telemetry — moved from telemetry.rs
// ---------------------------------------------------------------------------

/// Stateful sampler: CPU percent is a delta between two `/proc/stat`
/// snapshots (mirrors `psutil.cpu_percent(interval=None)`, which also
/// remembers the previous call's counters internally rather than blocking
/// for a fresh interval). One `HostTelemetry` lives for the whole worker
/// process (owned by `telemetry::Telemetry`, which `WorkerRuntime`'s
/// heartbeat loop owns), so consecutive `sample()` calls are consecutive
/// heartbeats apart.
pub struct HostTelemetry {
    previous_cpu: Option<CpuTimes>,
}

impl HostTelemetry {
    pub fn new() -> Self {
        HostTelemetry { previous_cpu: None }
    }

    pub fn sample(&mut self) -> SystemSnapshot {
        let cpu_percent = read_cpu_times().and_then(|current| {
            let percent = self.previous_cpu.map(|previous| current.percent_busy_since(&previous));
            self.previous_cpu = Some(current);
            percent
        });
        let memory = read_meminfo();
        let pid_rss_mb = read_self_rss_kb().map(|kb| kb / 1024.0);
        SystemSnapshot {
            cpu_percent,
            ram_used_gb: memory.map(|m| m.used_gb()),
            ram_total_gb: memory.map(|m| m.total_kb as f64 / 1024.0 / 1024.0),
            swap_used_gb: memory.map(|m| m.swap_used_gb()),
            pid_rss_mb,
        }
    }
}

impl Default for HostTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
struct CpuTimes {
    idle: u64,
    total: u64,
}

impl CpuTimes {
    fn percent_busy_since(&self, previous: &CpuTimes) -> f64 {
        let total_delta = self.total.saturating_sub(previous.total);
        let idle_delta = self.idle.saturating_sub(previous.idle);
        if total_delta == 0 {
            return 0.0;
        }
        ((total_delta - idle_delta) as f64 / total_delta as f64) * 100.0
    }
}

/// Parses the aggregate `cpu` line of `/proc/stat`:
/// `cpu  user nice system idle iowait irq softirq steal guest guest_nice`
/// (all values in USER_HZ jiffies since boot). `idle` for the busy/idle
/// split is `idle + iowait`, matching the common `top`/`psutil` convention
/// (iowait is CPU sitting idle waiting on I/O, not doing work).
fn read_cpu_times() -> Option<CpuTimes> {
    let contents = std::fs::read_to_string("/proc/stat").ok()?;
    let line = contents.lines().find(|l| l.starts_with("cpu "))?;
    let fields: Vec<u64> = line.split_whitespace().skip(1).filter_map(|f| f.parse().ok()).collect();
    if fields.len() < 4 {
        return None;
    }
    let idle = fields[3] + fields.get(4).copied().unwrap_or(0);
    let total: u64 = fields.iter().sum();
    Some(CpuTimes { idle, total })
}

#[derive(Debug, Clone, Copy)]
struct MemInfo {
    total_kb: u64,
    available_kb: u64,
    swap_total_kb: u64,
    swap_free_kb: u64,
}

impl MemInfo {
    fn used_gb(self) -> f64 {
        let used_kb = self.total_kb.saturating_sub(self.available_kb);
        used_kb as f64 / 1024.0 / 1024.0
    }
    fn swap_used_gb(self) -> f64 {
        let used_kb = self.swap_total_kb.saturating_sub(self.swap_free_kb);
        used_kb as f64 / 1024.0 / 1024.0
    }
}

/// Parses the fields this module needs out of `/proc/meminfo` (all values
/// there are in kB already, `KEY:   VALUE kB` per line).
fn read_meminfo() -> Option<MemInfo> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kb = None;
    let mut available_kb = None;
    let mut free_kb = None;
    let mut buffers_kb = None;
    let mut cached_kb = None;
    let mut swap_total_kb = None;
    let mut swap_free_kb = None;
    for line in contents.lines() {
        let Some((key, rest)) = line.split_once(':') else { continue };
        let value = rest.trim().split_whitespace().next().and_then(|v| v.parse::<u64>().ok());
        match key {
            "MemTotal" => total_kb = value,
            "MemAvailable" => available_kb = value,
            "MemFree" => free_kb = value,
            "Buffers" => buffers_kb = value,
            "Cached" => cached_kb = value,
            "SwapTotal" => swap_total_kb = value,
            "SwapFree" => swap_free_kb = value,
            _ => {}
        }
    }
    let total_kb = total_kb?;
    // MemAvailable exists on every kernel >= 3.14 (<host>'s Ubuntu 26.04
    // easily qualifies); the free+buffers+cached fallback only matters for
    // ancient kernels or unusual /proc implementations.
    let available_kb =
        available_kb.unwrap_or_else(|| free_kb.unwrap_or(0) + buffers_kb.unwrap_or(0) + cached_kb.unwrap_or(0));
    Some(MemInfo {
        total_kb,
        available_kb,
        swap_total_kb: swap_total_kb.unwrap_or(0),
        swap_free_kb: swap_free_kb.unwrap_or(0),
    })
}

/// Parses `VmRSS` out of `/proc/self/status` (kB), the resident set size of
/// this worker process — mirrors Python's `psutil.Process().memory_info().rss`.
fn read_self_rss_kb() -> Option<f64> {
    let contents = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = contents.lines().find(|l| l.starts_with("VmRSS:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

// ---------------------------------------------------------------------------
// Bench helpers — moved from exonit-worker/src/bin/exonit_bench.rs
// ---------------------------------------------------------------------------

pub fn hostname() -> String {
    let mut buf = [0u8; 256];
    unsafe {
        if libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) == 0 {
            let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            return String::from_utf8_lossy(&buf[..end]).to_string();
        }
    }
    "unknown".to_string()
}

pub fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}
