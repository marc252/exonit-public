//! Windows implementation of the `platform` abstraction layer. Task 2 of
//! `docs/plans/2026-08-06-win32-port-pcmarc.md`; every function here has
//! the SAME signature as its `unix.rs` counterpart (see that module's doc
//! comment) and implements exactly the closed design decisions in the
//! plan's "Global Constraints" section — **except** `spawn_worker_detached`
//! (added 2026-08-07, see its own doc comment for why) and its private
//! helpers (`create_process`, `append_quoted_arg`, `build_command_line`,
//! `build_environment_block`, `to_wide_null`, `open_inheritable`,
//! `OwnedHandle`, `AttributeListGuard`): Unix's detached spawn stays a thin
//! `platform::configure_detached` hook into `pidlock.rs`'s shared
//! `std::process::Command`-based `spawn_detached`, so there is no
//! `unix.rs` counterpart to mirror here — see `pidlock.rs`'s
//! `spawn_detached` (now `#[cfg(unix)]`/`#[cfg(windows)]`-split) for how the
//! two sides actually diverge.
//!
//!   - **Stop suau**: a `.agent.<worker_name>.stop` file under
//!     `worker_dir`. `request_stop(Soft, ...)` creates it;
//!     `shutdown_requested_extra_check()` (polled by `main.rs`'s existing
//!     watcher thread alongside the shared `AtomicBool`, exactly the same
//!     100ms loop as Unix) deletes it and flips the shared flag once seen.
//!     `--force`/the 10s-timeout escalation both go through `kill_now`
//!     (`TerminateProcess`) — `GenerateConsoleCtrlEvent` and a named
//!     `Event` were both considered and rejected by the plan (no shared
//!     console for a DETACHED process; named-object namespace fragility
//!     across SSH logon sessions).
//!   - **Identitat**: `QueryFullProcessImageNameW`. The Linux-only
//!     `" (deleted)"` exe-replaced-during-rebuild case cannot happen here
//!     (the exe stays locked while running) — `process_command_line`
//!     always returns `None`, so a live-but-unidentifiable PID reports
//!     `Unverifiable` (fail closed), same as today.
//!   - **Spawn detached**: originally `CommandExt::creation_flags
//!     (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_
//!     JOB)` via `std::process::Command`. **Superseded 2026-08-07** (SSH-hang
//!     fix, see `spawn_worker_detached`'s doc comment below): `std::process::
//!     Command` on Windows unconditionally sets `bInheritHandles = TRUE`
//!     whenever any stdio handle is redirected (which `agent start` always
//!     does, to send the child's stdout/stderr to its log file), and
//!     `CreateProcess` with `bInheritHandles = TRUE` duplicates *every*
//!     inheritable handle in the calling process into the child — not just
//!     the log file. Over SSH, that includes handles Win32-OpenSSH's `sshd`
//!     itself holds for the exec session, so `sshd` waited for the
//!     long-running detached worker to exit before returning control to the
//!     `ssh` client: `agent start` ran the worker fine but the SSH command
//!     never returned. `spawn_worker_detached` now calls `CreateProcessW`
//!     directly with a `STARTUPINFOEXW` + `PROC_THREAD_ATTRIBUTE_HANDLE_LIST`
//!     restricting inheritance to exactly the two handles it opens itself
//!     (log file, `NUL` device) — `configure_detached`/
//!     `configure_detached_without_breakaway` (which configured a
//!     `std::process::Command`) are gone; `is_breakaway_access_denied` is
//!     kept, now checked against `CreateProcessW`'s own return.
//!   - **Startup lock**: `fs4`'s `FileExt::lock_exclusive`
//!     (`LockFileEx` under the hood; released by the OS if the holding
//!     process dies, same stale-safe semantics as Unix's `flock`).
//!   - **Liveness**: `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` +
//!     `GetExitCodeProcess() == STILL_ACTIVE`; `OpenProcess` failing with
//!     `ERROR_ACCESS_DENIED` also counts as alive (mirrors Unix's `EPERM`
//!     branch — a process we can't fully query but the OS still lets us
//!     see exists is alive).
//!   - **Telemetria**: `GetSystemTimes` (delta sampler, mirrors
//!     `/proc/stat`), `GlobalMemoryStatusEx` (RAM; swap approximated as
//!     `(TotalPageFile-AvailPageFile) - (TotalPhys-AvailPhys)`, i.e. commit
//!     usage beyond physical RAM — an approximation, documented inline),
//!     `K32GetProcessMemoryInfo` -> `WorkingSetSize` (RSS).
//!   - **Redirecció de log**: `CreateFileW(FILE_APPEND_DATA|SYNCHRONIZE,
//!     FILE_SHARE_READ|FILE_SHARE_WRITE)` + `SetStdHandle` for both
//!     `STD_OUTPUT_HANDLE`/`STD_ERROR_HANDLE`. No CRT-level `_dup2`/
//!     `_open_osfhandle` redirection is added on top (see
//!     `redirect_standard_streams`'s doc comment for why that extra layer
//!     turned out to be unnecessary here, not just unverified).
//!
//! **One documented deviation from the plan's literal windows-sys feature
//! list** (`Win32_System_Threading`, `Win32_System_ProcessStatus`,
//! `Win32_System_SystemInformation`, `Win32_Storage_FileSystem`,
//! `Win32_Foundation`, `Win32_System_Console`): `CreateFileW`'s generated
//! binding additionally requires the `Win32_Security` feature (its
//! signature references `SECURITY_ATTRIBUTES`, defined in that module —
//! confirmed by inspecting the windows-sys 0.59 source directly, not
//! guessed). Added to both Cargo.tomls; still only `windows-sys`/`fs4`
//! dependencies, per the plan's "new dependencies allowed ONLY"
//! constraint.
//!
//! **`cpu_model()` is a documented "unknown" stub, not a full registry
//! read.** The Global Constraints text allows either ("registre o
//! `unknown`"); it is bench-only informational metadata
//! (`exonit-bench`'s JSON output), not exercised by any test or the
//! production control plane, so reading
//! `HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0\ProcessorNameString`
//! (which would need the `Win32_System_Registry` feature and its own
//! wide-string buffer dance) was judged not worth the extra unsafe surface
//! for a cosmetic field — project rule 7, "contained complexity".

use crate::platform::StopKind;
use crate::telemetry::SystemSnapshot;
use fs4::fs_std::FileExt;
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_ACCESS_DENIED, FALSE, FILETIME, GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE, TRUE,
};
use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, FILE_APPEND_DATA, FILE_ATTRIBUTE_NORMAL, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_ALWAYS,
    OPEN_EXISTING, SYNCHRONIZE,
};
use windows_sys::Win32::System::Console::{SetConsoleCtrlHandler, SetStdHandle, STD_ERROR_HANDLE, STD_OUTPUT_HANDLE};
use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
use windows_sys::Win32::System::Threading::{
    CreateProcessW, DeleteProcThreadAttributeList, GetCurrentProcess, GetExitCodeProcess, GetSystemTimes,
    InitializeProcThreadAttributeList, OpenProcess, QueryFullProcessImageNameW, TerminateProcess,
    UpdateProcThreadAttribute, CREATE_BREAKAWAY_FROM_JOB, CREATE_NEW_PROCESS_GROUP, CREATE_UNICODE_ENVIRONMENT,
    DETACHED_PROCESS, EXTENDED_STARTUPINFO_PRESENT, LPPROC_THREAD_ATTRIBUTE_LIST, PROCESS_INFORMATION,
    PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_TERMINATE, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, STARTF_USESTDHANDLES,
    STARTUPINFOEXW, STARTUPINFOW,
};

/// `GetExitCodeProcess`'s sentinel meaning "still running" (winbase.h).
const STILL_ACTIVE: u32 = 259;

// ---------------------------------------------------------------------------
// Process identity / liveness
// ---------------------------------------------------------------------------

/// `QueryFullProcessImageNameW` — the Windows analogue of Unix's
/// `/proc/<pid>/exe` readlink. Returns `None` on any failure (process gone,
/// access denied, buffer too small), same "unverifiable" contract as Unix.
pub fn process_executable_path(pid: i32) -> Option<PathBuf> {
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid as u32);
        if handle.is_null() {
            return None;
        }
        // 32768 (0x8000) covers the NTFS/Win32 extended-length path limit
        // (32767 chars + NUL) — generous enough that truncation in practice
        // only happens for a QueryFullProcessImageNameW failure, not a real
        // path.
        let mut buffer: Vec<u16> = vec![0u16; 32768];
        let mut size: u32 = buffer.len() as u32;
        let ok = QueryFullProcessImageNameW(handle, 0, buffer.as_mut_ptr(), &mut size);
        CloseHandle(handle);
        if ok == 0 {
            return None;
        }
        buffer.truncate(size as usize);
        Some(PathBuf::from(std::ffi::OsString::from_wide(&buffer)))
    }
}

/// No public, reliable Windows API returns a live process's argv (see
/// `docs/plans/2026-08-06-win32-port-pcmarc.md`'s surface-report table) —
/// unlike Unix's `/proc/<pid>/cmdline` fallback, this always reports
/// unavailable. `AgentController::inspect_process` already treats "exe path
/// unreadable AND no command line" as `Unverifiable` (fail closed), which is
/// the correct, honest behavior here: never silently claim ownership of a
/// PID Windows won't let us positively identify.
pub fn process_command_line(_pid: i32) -> Option<Vec<String>> {
    None
}

/// `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` + `GetExitCodeProcess`.
/// `OpenProcess` failing with `ERROR_ACCESS_DENIED` still counts as "alive"
/// (mirrors Unix's `kill(pid, 0)` treating `EPERM` as alive: the OS is
/// telling us the PID exists, just that we can't fully query it).
pub fn process_alive(pid: i32) -> bool {
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid as u32);
        if handle.is_null() {
            let err = std::io::Error::last_os_error();
            return err.raw_os_error() == Some(ERROR_ACCESS_DENIED as i32);
        }
        let mut exit_code: u32 = 0;
        let ok = GetExitCodeProcess(handle, &mut exit_code);
        CloseHandle(handle);
        ok != 0 && exit_code == STILL_ACTIVE
    }
}

// ---------------------------------------------------------------------------
// Spawn / stop
// ---------------------------------------------------------------------------

/// `DETACHED_PROCESS` (no console) `| CREATE_NEW_PROCESS_GROUP` (so the
/// child doesn't receive Ctrl+C/Ctrl+Break meant for its parent's group) —
/// the Windows analogue of Unix's `setsid()`-in-`pre_exec` detach.
///
/// `| CREATE_BREAKAWAY_FROM_JOB` — real bug found via Task 3's E2E test on
/// <host> (2026-08-06, not anticipated by the plan): Win32-OpenSSH wraps
/// each SSH exec session in a Windows Job Object with
/// `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. Neither `DETACHED_PROCESS` nor
/// `CREATE_NEW_PROCESS_GROUP` exempts a child from its parent's job — a
/// worker spawned by `agent start` over SSH was observed surviving only
/// until the SSH channel that ran `agent start` closed (roughly 1s later,
/// right after `start_locked`'s own liveness check), at which point the
/// whole job — including the "detached" worker — was killed outright, mid
/// syscall, with no chance to log anything or respond to the stop-file.
/// `CREATE_BREAKAWAY_FROM_JOB` makes the child break away from any job it
/// would otherwise inherit, so it survives independently of the spawning
/// SSH session — this is what farmctl's whole `agent start` deployment
/// model requires (it always runs over SSH). Requires the parent job to
/// permit breakaway (`JOB_OBJECT_LIMIT_BREAKAWAY_OK`) — confirmed working
/// against <host>'s actual sshd, not merely a documented theory (see
/// docs/RUNBOOK_PCMARC.md's E2E section for the exact repro/fix
/// verification). When the CALLING process's own job does NOT permit
/// breakaway (`ERROR_ACCESS_DENIED`/5 from `CreateProcess`) — observed for
/// `cargo test`'s own test-binary process tree, unlike an SSH exec session
/// — `spawn()` fails outright rather than silently ignoring the flag; see
/// `is_breakaway_access_denied`, which `spawn_worker_detached` uses to retry
/// once without the flag in that case.
///
/// True when `err` (from a `CreateProcessW` call using
/// `CREATE_BREAKAWAY_FROM_JOB`) is `CreateProcess` refusing that flag
/// specifically (`ERROR_ACCESS_DENIED`, raw OS error 5) — the signal
/// `spawn_worker_detached` uses to decide whether to retry once without the
/// flag. Not a generic "any access-denied spawn failure" catch-all —
/// narrowly scoped to this one recoverable case; any other `CreateProcessW`
/// failure still propagates as a hard error, unchanged.
pub fn is_breakaway_access_denied(err: &std::io::Error) -> bool {
    err.raw_os_error() == Some(5)
}

// ---------------------------------------------------------------------------
// Detached worker spawn with restricted handle inheritance (SSH-hang fix,
// 2026-08-07)
// ---------------------------------------------------------------------------

/// Windows quoting rule for a single `CreateProcessW` command-line argument
/// (the algorithm Microsoft's own C/C++ runtime `argv` parser expects —
/// see "Parsing C Command-Line Arguments" on MS Learn; also what Rust's own
/// `std::process::Command` implements internally for the exact same reason).
/// Backslashes are only special immediately before a `"` or at the very end
/// of the argument (where they'd otherwise escape the closing quote this
/// function adds) — elsewhere a lone backslash is just a literal backslash,
/// which is why this cannot be a simple "escape every backslash and quote"
/// scheme.
fn append_quoted_arg(out: &mut String, arg: &str) {
    let needs_quotes = arg.is_empty() || arg.contains([' ', '\t', '"']);
    if !needs_quotes {
        out.push_str(arg);
        return;
    }
    out.push('"');
    let mut chars = arg.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            let mut backslashes = 1;
            while chars.peek() == Some(&'\\') {
                backslashes += 1;
                chars.next();
            }
            let escape = matches!(chars.peek(), Some('"') | None);
            let count = if escape { backslashes * 2 } else { backslashes };
            for _ in 0..count {
                out.push('\\');
            }
        } else if c == '"' {
            out.push('\\');
            out.push('"');
        } else {
            out.push(c);
        }
    }
    out.push('"');
}

/// Builds the single command-line string `CreateProcessW` expects (argv[0]
/// is the quoted executable path, exactly like `std::process::Command`
/// produces internally — a child reading its own `argv[0]`/`GetCommandLineW`
/// still needs to see it there even though `lpApplicationName` is what
/// actually locates the executable).
fn build_command_line(executable: &Path, arguments: &[String]) -> String {
    let mut out = String::new();
    append_quoted_arg(&mut out, &executable.to_string_lossy());
    for argument in arguments {
        out.push(' ');
        append_quoted_arg(&mut out, argument);
    }
    out
}

/// `CreateProcessW`'s environment block: `"KEY=VALUE\0"` entries
/// concatenated, terminated by an extra trailing `\0` (double-NUL). Sorted
/// by key — the API doesn't require it, but it makes the block (and any test
/// asserting on it) deterministic, matching `spawn_detached`'s `env_clear()`
/// + explicit-map semantics on the Unix side.
fn build_environment_block(environment: &HashMap<String, String>) -> Vec<u16> {
    let mut keys: Vec<&String> = environment.keys().collect();
    keys.sort();
    let mut block: Vec<u16> = Vec::new();
    for key in keys {
        let entry = format!("{key}={}", environment[key]);
        block.extend(entry.encode_utf16());
        block.push(0);
    }
    block.push(0);
    block
}

fn to_wide_null(path: &Path) -> Vec<u16> {
    let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    wide.push(0);
    wide
}

/// RAII close for a handle this function opened itself (log file / `NUL`
/// device) — guarantees the parent's copy is closed on every exit path,
/// including the early-return error branches below, without hand-duplicated
/// cleanup code at each one.
struct OwnedHandle(HANDLE);

impl Drop for OwnedHandle {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.0);
        }
    }
}

/// Opens `path` as inheritable (`SECURITY_ATTRIBUTES.bInheritHandle =
/// TRUE`) — a handle in the `PROC_THREAD_ATTRIBUTE_HANDLE_LIST` whitelist
/// below MUST have been created inheritable, or `UpdateProcThreadAttribute`
/// rejects it.
fn open_inheritable(
    path_wide: &[u16],
    desired_access: u32,
    disposition: u32,
) -> std::io::Result<OwnedHandle> {
    let mut security_attributes = SECURITY_ATTRIBUTES {
        nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: std::ptr::null_mut(),
        bInheritHandle: TRUE,
    };
    let handle = unsafe {
        CreateFileW(
            path_wide.as_ptr(),
            desired_access,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            &mut security_attributes,
            disposition,
            FILE_ATTRIBUTE_NORMAL,
            std::ptr::null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(std::io::Error::last_os_error());
    }
    Ok(OwnedHandle(handle))
}

/// RAII cleanup for the `PROC_THREAD_ATTRIBUTE_LIST` — must be dropped
/// (`DeleteProcThreadAttributeList`) before the byte buffer backing it is
/// freed; declared after the buffer in `spawn_worker_detached` so Rust's
/// last-declared-first-dropped order does that automatically.
struct AttributeListGuard(LPPROC_THREAD_ATTRIBUTE_LIST);

impl Drop for AttributeListGuard {
    fn drop(&mut self) {
        unsafe {
            DeleteProcThreadAttributeList(self.0);
        }
    }
}

/// Spawns the worker binary fully detached from the calling process AND
/// from any ambient handle inheritance it might carry — the fix for a real
/// bug found running `agent start` over SSH against <host> (2026-08-07):
/// `std::process::Command` on Windows unconditionally sets
/// `bInheritHandles = TRUE` in its `CreateProcess` call whenever any stdio
/// handle is redirected (which `agent start` always does, sending the
/// child's stdout/stderr to its log file) — and `bInheritHandles = TRUE`
/// duplicates *every* inheritable handle open in the calling process into
/// the child, not just the ones the caller intended. Over SSH, the calling
/// process is `sshd`'s own exec'd `agent start`, which holds inheritable
/// handles for the SSH session itself; those got duplicated into the
/// "detached" worker, so `sshd` waited for that duplicate to close (i.e.,
/// for the long-running worker to exit) before returning control to the
/// `ssh` client — `agent start` ran the worker correctly, but the `ssh`
/// invocation itself hung indefinitely.
///
/// The fix (Microsoft's documented mechanism for exactly this problem,
/// since Vista): call `CreateProcessW` directly with a `STARTUPINFOEXW` +
/// `PROC_THREAD_ATTRIBUTE_HANDLE_LIST` naming EXACTLY the handles allowed to
/// inherit — the log file (stdout+stderr) and the `NUL` device (stdin),
/// both opened by this function. `bInheritHandles` is still passed as
/// `TRUE` (required for the whitelist to have any effect at all), but with
/// the attribute list present, Windows inherits ONLY the listed handles —
/// every other inheritable handle in the process, including whatever `sshd`
/// holds, is left alone.
///
/// Retains `CREATE_BREAKAWAY_FROM_JOB` (Task 3's fix for the separate
/// Win32-OpenSSH Job-Object-kills-on-session-close problem — see the module
/// doc comment) with the same `ERROR_ACCESS_DENIED` fallback retry without
/// it, now checked against `CreateProcessW`'s own result instead of
/// `Command::spawn()`'s.
///
/// Returns the child's PID. Unlike the Unix side (which keeps the
/// `std::process::Child` handle to `try_wait()` it moments later in
/// `pidlock.rs`), there is nothing meaningful to keep here: this function
/// closes its own copies of `hProcess`/`hThread` immediately after
/// `CreateProcessW` returns (the child has its own independent handle
/// table), and `pidlock.rs` re-derives liveness via `process_alive(pid)`
/// exactly as it already does for `Child::try_wait()`'s own error fallback.
pub fn spawn_worker_detached(
    executable: &Path,
    arguments: &[String],
    environment: &HashMap<String, String>,
    log_path: &Path,
) -> std::io::Result<i32> {
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let log_handle = open_inheritable(&to_wide_null(log_path), FILE_APPEND_DATA | SYNCHRONIZE, OPEN_ALWAYS)?;
    let nul_path = to_wide_null(Path::new("NUL"));
    let nul_handle = open_inheritable(&nul_path, GENERIC_READ | GENERIC_WRITE, OPEN_EXISTING)?;

    // Two-call idiom `InitializeProcThreadAttributeList` requires: the first
    // call (null buffer) always "fails" — it exists only to report the
    // required buffer size via `required_size`, per MSDN.
    let mut required_size: usize = 0;
    unsafe {
        InitializeProcThreadAttributeList(std::ptr::null_mut(), 1, 0, &mut required_size);
    }
    let mut attribute_buffer: Vec<u8> = vec![0u8; required_size];
    let attribute_list: LPPROC_THREAD_ATTRIBUTE_LIST = attribute_buffer.as_mut_ptr() as LPPROC_THREAD_ATTRIBUTE_LIST;
    let initialized = unsafe { InitializeProcThreadAttributeList(attribute_list, 1, 0, &mut required_size) };
    if initialized == 0 {
        return Err(std::io::Error::last_os_error());
    }
    // Declared AFTER `attribute_buffer` so `attribute_guard` drops first
    // (`DeleteProcThreadAttributeList`, which must run before the memory it
    // points into is freed) — see `AttributeListGuard`'s doc comment.
    let attribute_guard = AttributeListGuard(attribute_list);

    let inheritable_handles: [HANDLE; 2] = [log_handle.0, nul_handle.0];
    let updated = unsafe {
        UpdateProcThreadAttribute(
            attribute_list,
            0,
            PROC_THREAD_ATTRIBUTE_HANDLE_LIST as usize,
            inheritable_handles.as_ptr() as *const c_void,
            std::mem::size_of_val(&inheritable_handles),
            std::ptr::null_mut(),
            std::ptr::null(),
        )
    };
    if updated == 0 {
        return Err(std::io::Error::last_os_error());
    }

    let mut startup_info: STARTUPINFOEXW = unsafe { std::mem::zeroed() };
    startup_info.StartupInfo.cb = std::mem::size_of::<STARTUPINFOEXW>() as u32;
    startup_info.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup_info.StartupInfo.hStdInput = nul_handle.0;
    startup_info.StartupInfo.hStdOutput = log_handle.0;
    startup_info.StartupInfo.hStdError = log_handle.0;
    startup_info.lpAttributeList = attribute_list;

    let application_name = to_wide_null(executable);
    let mut command_line = build_command_line(executable, arguments)
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect::<Vec<u16>>();
    let environment_block = build_environment_block(environment);

    let base_flags = DETACHED_PROCESS
        | CREATE_NEW_PROCESS_GROUP
        | EXTENDED_STARTUPINFO_PRESENT
        | CREATE_UNICODE_ENVIRONMENT;
    let pid = create_process(
        &application_name,
        &mut command_line,
        &environment_block,
        &startup_info,
        base_flags | CREATE_BREAKAWAY_FROM_JOB,
    )
    .or_else(|err| {
        if is_breakaway_access_denied(&err) {
            create_process(&application_name, &mut command_line, &environment_block, &startup_info, base_flags)
        } else {
            Err(err)
        }
    })?;

    drop(attribute_guard);
    Ok(pid)
}

fn create_process(
    application_name: &[u16],
    command_line: &mut [u16],
    environment_block: &[u16],
    startup_info: &STARTUPINFOEXW,
    creation_flags: u32,
) -> std::io::Result<i32> {
    let mut process_information: PROCESS_INFORMATION = unsafe { std::mem::zeroed() };
    let ok = unsafe {
        CreateProcessW(
            application_name.as_ptr(),
            command_line.as_mut_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            TRUE,
            creation_flags,
            environment_block.as_ptr() as *const c_void,
            std::ptr::null(),
            &startup_info.StartupInfo as *const STARTUPINFOW,
            &mut process_information,
        )
    };
    if ok == 0 {
        return Err(std::io::Error::last_os_error());
    }
    let pid = process_information.dwProcessId as i32;
    unsafe {
        CloseHandle(process_information.hThread);
        CloseHandle(process_information.hProcess);
    }
    Ok(pid)
}

fn stop_file_path(worker_dir: &Path, worker_name: &str) -> PathBuf {
    worker_dir.join(format!(".agent.{worker_name}.stop"))
}

fn terminate(pid: i32) -> std::io::Result<()> {
    unsafe {
        let handle = OpenProcess(PROCESS_TERMINATE, FALSE, pid as u32);
        if handle.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        let ok = TerminateProcess(handle, 1);
        CloseHandle(handle);
        if ok == 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

/// `Soft` creates the stop-file the target worker's watcher thread polls
/// for (see `shutdown_requested_extra_check`) — the 10s wait-then-escalate
/// loop is shared, platform-agnostic logic in `pidlock.rs`'s `stop()`, not
/// duplicated here. `Force` terminates immediately, matching Unix's
/// `SIGKILL`-instead-of-`SIGTERM` `--force` behavior.
pub fn request_stop(pid: i32, kind: StopKind, worker_name: &str, worker_dir: &Path) -> std::io::Result<()> {
    match kind {
        StopKind::Soft => std::fs::write(stop_file_path(worker_dir, worker_name), []),
        StopKind::Force => terminate(pid),
    }
}

/// The final escalation of a soft stop that didn't exit within the timeout
/// (or `--force`'s direct effect via `request_stop`, mirrored here for the
/// same "fire and forget, best effort" semantics `kill_now` has on Unix).
pub fn kill_now(pid: i32) {
    let _ = terminate(pid);
}

// ---------------------------------------------------------------------------
// Startup lock
// ---------------------------------------------------------------------------

/// Holds an exclusive `fs4` lock (`LockFileEx` under the hood) for as long
/// as it lives; `Drop` unlocks (closing the underlying `File` releases the
/// OS handle). The OS also releases the lock if the holding process dies —
/// same stale-safe semantics `with_startup_lock` relies on via Unix's
/// `flock`.
pub struct StartupLock {
    file: std::fs::File,
}

impl Drop for StartupLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}

pub fn acquire_startup_lock(path: &Path) -> std::io::Result<StartupLock> {
    let file = std::fs::OpenOptions::new().create(true).read(true).write(true).open(path)?;
    FileExt::lock_exclusive(&file)?;
    Ok(StartupLock { file })
}

// ---------------------------------------------------------------------------
// Log redirection
// ---------------------------------------------------------------------------

/// Opens (creating/appending) the log file and points both
/// `STD_OUTPUT_HANDLE` and `STD_ERROR_HANDLE` at the SAME handle — the
/// Windows analogue of Unix's `dup2`-onto-both-fds. `FILE_SHARE_READ |
/// FILE_SHARE_WRITE` lets the CLI's `agent logs` tail the file concurrently
/// (Windows file sharing is opt-in per-handle, unlike Unix where any
/// process can always open+read a file another process has open).
///
/// **What this covers vs. doesn't**: `SetStdHandle` retargets what
/// `GetStdHandle(STD_OUTPUT_HANDLE/STD_ERROR_HANDLE)` returns — which is
/// what `std::io::stdout()`/`eprintln!`/inherited-handle child I/O actually
/// use — so this worker's own `println!`/`eprintln!` output and the
/// spawned worker CHILD's own stdout/stderr (already separately redirected
/// to the SAME log file — via `File::try_clone` on Unix's `spawn_detached`,
/// via two independent inheritable `CreateFileW` opens on
/// `spawn_worker_detached`'s side here — unaffected by this function) both
/// land correctly. The one thing NOT covered is the
/// MSVC C runtime's own buffered `stdout`/`stderr` (fd 1/2 in CRT terms):
/// unlike Unix's `libc::dup2`, `SetStdHandle` does not retarget those CRT
/// file descriptors. This turned out not to matter for this binary — `ort`
/// (the only C/C++ dependency that could write straight to a CRT fd rather
/// than through the OS handle) logs through its own configurable logger,
/// not raw stdio, and nothing else in this process's dependency graph
/// writes to CRT fd 1/2 directly — so the `libc::open_osfhandle`+`dup2`
/// best-effort layer the plan flagged as "verify if `libc` exposes it" was
/// written, tested to compile, and then removed as genuinely unneeded
/// complexity (project rule 7) rather than kept as unverified dead code.
pub fn redirect_standard_streams(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    wide.push(0);
    let handle = unsafe {
        CreateFileW(
            wide.as_ptr(),
            FILE_APPEND_DATA | SYNCHRONIZE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            std::ptr::null(),
            OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            std::ptr::null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(std::io::Error::last_os_error());
    }
    unsafe {
        if SetStdHandle(STD_OUTPUT_HANDLE, handle) == 0 || SetStdHandle(STD_ERROR_HANDLE, handle) == 0 {
            let err = std::io::Error::last_os_error();
            CloseHandle(handle);
            return Err(err);
        }
    }
    // Intentionally NOT closed: `handle` is now what
    // `GetStdHandle(STD_OUTPUT_HANDLE/STD_ERROR_HANDLE)` returns for the
    // remaining lifetime of this process — the Windows equivalent of Unix
    // leaving fd 1/2 open forever after `dup2`. The OS reclaims it at
    // process exit.
    Ok(())
}

// ---------------------------------------------------------------------------
// Shutdown watch
// ---------------------------------------------------------------------------

static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);
/// Set once by `install_shutdown_watch`; read by every subsequent
/// `shutdown_requested_extra_check()` poll from the watcher thread. A
/// `OnceLock` (not `Mutex`) because it is written exactly once, at startup,
/// before the watcher thread that reads it is spawned (`main.rs::run_worker`
/// calls `install_shutdown_watch` before spawning the watcher thread — see
/// that module).
static STOP_FILE_PATH: OnceLock<PathBuf> = OnceLock::new();

unsafe extern "system" fn console_handler(_ctrl_type: u32) -> i32 {
    SHUTDOWN_REQUESTED.store(true, Ordering::SeqCst);
    // Non-zero = "handled, do not pass to the next handler / default
    // action" (which for CTRL_CLOSE_EVENT etc. would terminate the process
    // immediately instead of letting `WorkerRuntime::run()`'s main loop
    // finish the in-flight task and exit clean).
    TRUE
}

/// Installs `SetConsoleCtrlHandler` (mirrors Unix's `SIGINT`/`SIGTERM`
/// handlers) and remembers the `.agent.<worker_name>.stop` path
/// `shutdown_requested_extra_check` polls for — the second, file-based
/// shutdown source `agent stop`'s soft-stop path uses (see
/// `request_stop`). Returns the same shared flag `install_shutdown_watch`
/// returns on Unix, so `main.rs`'s watcher thread structure is unchanged.
pub fn install_shutdown_watch(worker_name: &str, worker_dir: &Path) -> &'static AtomicBool {
    let path = stop_file_path(worker_dir, worker_name);
    // A stop-file left over from a PREVIOUS instance of this worker_name
    // (e.g. `agent stop` created it, but that instance was already hung or
    // dead and never consumed it — it escalates to `kill_now` after the 10s
    // timeout in `pidlock.rs::stop()`, but never deletes the file itself)
    // must not be allowed to kill the NEXT `agent start`: a stop request is
    // for the process that was running THEN, not for whichever new process
    // happens to reuse this worker_name later. The newly starting worker
    // discards any stale request before it starts watching for new ones —
    // real bug found in review, 2026-08-06: without this, a stale stop-file
    // made every subsequent `agent start` for that worker_name immediately
    // self-terminate until someone deleted the file by hand.
    let _ = std::fs::remove_file(&path);
    let _ = STOP_FILE_PATH.set(path);
    unsafe {
        SetConsoleCtrlHandler(Some(console_handler), TRUE);
    }
    &SHUTDOWN_REQUESTED
}

/// Windows-only extra shutdown source `main.rs`'s watcher thread ORs in
/// alongside the shared flag (`if shutdown_flag.load(..) ||
/// platform::shutdown_requested_extra_check() { ... }`) — polling for the
/// stop-file `request_stop(Soft, ...)` creates. Deletes the file and flips
/// the shared flag the moment it is seen, so a second poll never re-detects
/// it. On Unix this function doesn't exist as a Windows primitive at all;
/// `unix.rs` provides a no-op (`false`, always) so `main.rs` can call it
/// unconditionally without a `#[cfg(...)]` of its own.
pub fn shutdown_requested_extra_check() -> bool {
    let Some(path) = STOP_FILE_PATH.get() else { return false };
    if path.exists() {
        let _ = std::fs::remove_file(path);
        SHUTDOWN_REQUESTED.store(true, Ordering::SeqCst);
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// Host telemetry
// ---------------------------------------------------------------------------

/// Windows analogue of `unix::HostTelemetry`: `GetSystemTimes` gives a
/// delta-sampled CPU percent the same way `unix::HostTelemetry` derives one
/// from two `/proc/stat` snapshots.
pub struct HostTelemetry {
    /// `(idle_100ns, total_100ns)` from the previous sample.
    previous_cpu: Option<(u64, u64)>,
}

impl HostTelemetry {
    pub fn new() -> Self {
        HostTelemetry { previous_cpu: None }
    }

    pub fn sample(&mut self) -> SystemSnapshot {
        let cpu_percent = read_cpu_times().and_then(|current| {
            let percent = self.previous_cpu.map(|previous| cpu_percent_busy_since(current, previous));
            self.previous_cpu = Some(current);
            percent
        });
        let memory = read_memory_status();
        let pid_rss_mb = read_self_rss_kb().map(|kb| kb / 1024.0);
        SystemSnapshot {
            cpu_percent,
            ram_used_gb: memory.map(ram_used_gb),
            ram_total_gb: memory.map(|m| m.ullTotalPhys as f64 / 1024.0 / 1024.0 / 1024.0),
            swap_used_gb: memory.map(swap_used_gb),
            pid_rss_mb,
        }
    }
}

impl Default for HostTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

fn cpu_percent_busy_since(current: (u64, u64), previous: (u64, u64)) -> f64 {
    let (idle, total) = current;
    let (previous_idle, previous_total) = previous;
    let total_delta = total.saturating_sub(previous_total);
    let idle_delta = idle.saturating_sub(previous_idle);
    if total_delta == 0 {
        return 0.0;
    }
    ((total_delta - idle_delta) as f64 / total_delta as f64) * 100.0
}

/// `GetSystemTimes`: `lpKernelTime` already INCLUDES idle time (per MSDN),
/// so `total = kernel + user` and `idle` (a subset of `kernel`) is the busy/
/// idle split's "idle" side — mirrors `unix::read_cpu_times`'s `idle`/
/// `total` jiffie pair exactly, just in 100ns FILETIME units instead of
/// USER_HZ jiffies (the unit cancels out in the delta-ratio calculation
/// above either way).
fn read_cpu_times() -> Option<(u64, u64)> {
    unsafe {
        let mut idle = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
        let mut kernel = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
        let mut user = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
        let ok = GetSystemTimes(&mut idle, &mut kernel, &mut user);
        if ok == 0 {
            return None;
        }
        let total = filetime_to_u64(kernel) + filetime_to_u64(user);
        Some((filetime_to_u64(idle), total))
    }
}

fn filetime_to_u64(ft: FILETIME) -> u64 {
    ((ft.dwHighDateTime as u64) << 32) | (ft.dwLowDateTime as u64)
}

fn read_memory_status() -> Option<MEMORYSTATUSEX> {
    unsafe {
        // All-integer repr(C) struct: zeroed() is a valid initial value —
        // `dwLength` (the only field the API reads on input) is set right
        // after.
        let mut status: MEMORYSTATUSEX = std::mem::zeroed();
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        let ok = GlobalMemoryStatusEx(&mut status);
        if ok == 0 {
            return None;
        }
        Some(status)
    }
}

fn ram_used_gb(status: MEMORYSTATUSEX) -> f64 {
    let used_bytes = status.ullTotalPhys.saturating_sub(status.ullAvailPhys);
    used_bytes as f64 / 1024.0 / 1024.0 / 1024.0
}

/// Swap has no direct Win32 equivalent (Windows unifies RAM+swap into one
/// commit-limited "page file" concept) — approximated per the plan's Global
/// Constraints as pagefile commit usage beyond physical RAM usage:
/// `(TotalPageFile-AvailPageFile) - (TotalPhys-AvailPhys)`, saturating at 0
/// (a system with plenty of free RAM and light commit can legitimately have
/// "negative" raw arithmetic here). This is a documented approximation, not
/// attempted byte-parity with any specific tool (see `telemetry.rs`'s
/// module doc comment on the same principle for the Linux side).
fn swap_used_gb(status: MEMORYSTATUSEX) -> f64 {
    let ram_used_bytes = status.ullTotalPhys.saturating_sub(status.ullAvailPhys);
    let pagefile_used_bytes = status.ullTotalPageFile.saturating_sub(status.ullAvailPageFile);
    let swap_used_bytes = pagefile_used_bytes.saturating_sub(ram_used_bytes);
    swap_used_bytes as f64 / 1024.0 / 1024.0 / 1024.0
}

/// `K32GetProcessMemoryInfo(GetCurrentProcess())` -> `WorkingSetSize` —
/// mirrors `unix::read_self_rss_kb`'s `/proc/self/status` `VmRSS` read
/// (Windows' closest analogue to Linux RSS).
fn read_self_rss_kb() -> Option<f64> {
    unsafe {
        let mut counters: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        let ok = K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb);
        if ok == 0 {
            return None;
        }
        Some(counters.WorkingSetSize as f64 / 1024.0)
    }
}

// ---------------------------------------------------------------------------
// Bench helpers
// ---------------------------------------------------------------------------

pub fn hostname() -> String {
    std::env::var("COMPUTERNAME").unwrap_or_else(|_| "unknown".to_string())
}

/// See this module's doc comment: a full registry read was judged not
/// worth the extra unsafe surface for this bench-only cosmetic field.
pub fn cpu_model() -> String {
    "unknown".to_string()
}
