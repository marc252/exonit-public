//! Host telemetry for the heartbeat's `system` field.
//!
//! No Swift file mirrors this module: `WorkerRuntime.swift`'s
//! `heartbeatLoop()` sends `HeartbeatRequest` with `system: [:]` (an empty
//! dictionary) always — the Swift port never grew a telemetry collector.
//! The field names and semantics below instead come from the *Python*
//! worker's `_build_heartbeat_payload` (`~/exonit/worker/worker.py`,
//! `payload["system"] = sysinfo` block), which is what the server/dashboard
//! actually expect on the wire: `cpu_percent`, `ram_used_gb`, `ram_total_gb`,
//! `swap_used_gb`, `pid_rss_mb` (Python also has a debug-only `cpu_temp_c`,
//! omitted here — no portable Linux sensor path is worth adding for F0b).
//!
//! Python gets these from `psutil`; the actual sampling (originally reading
//! `/proc` directly per DESIGN.md's F0b task list) now lives in
//! `crate::platform::HostTelemetry` (extracted in
//! `docs/plans/2026-08-06-win32-port-pcmarc.md` Task 1, pure refactor) —
//! this module keeps only the platform-agnostic `SystemSnapshot` data shape
//! and its JSON encoding, and `Telemetry` is a thin wrapper delegating
//! `sample()` to `platform::HostTelemetry`. GPU telemetry stays an
//! explicitly empty map for now (task instruction: "GPU deixa'l com a camp
//! opcional buit de moment") — `<host>`'s GPU counters are F4+ scope,
//! once the CUDA feature exists to report against.
//!
//! Exact byte-for-byte parity with `psutil`'s used-memory accounting is not
//! attempted (this is an operator-facing gauge, not part of the
//! golden-fixture science determinism gate in project rule 3) — the
//! `MemAvailable`-based calculation `platform::unix::HostTelemetry` uses is
//! the same one `psutil` and `free -h` use on modern (3.14+) Linux kernels.

use serde_json::{Map, Value};

/// One heartbeat's worth of host metrics. Every field is `None` when its
/// `/proc` source could not be read (matches Python's behaviour of degrading
/// gracefully rather than dropping the whole heartbeat).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SystemSnapshot {
    pub cpu_percent: Option<f64>,
    pub ram_used_gb: Option<f64>,
    pub ram_total_gb: Option<f64>,
    pub swap_used_gb: Option<f64>,
    pub pid_rss_mb: Option<f64>,
}

impl SystemSnapshot {
    /// Builds the `system` map exactly like Python's `sysinfo` dict — only
    /// present fields are included; an entirely empty snapshot (every
    /// `/proc` read failed) reports `{"error": "proc_unavailable"}`, mirroring
    /// Python's `{"error": "psutil_missing"}` fallback shape.
    pub fn to_json_map(self) -> Map<String, Value> {
        let mut map = Map::new();
        if let Some(v) = self.cpu_percent {
            map.insert("cpu_percent".to_string(), round2(v).into());
        }
        if let Some(v) = self.ram_used_gb {
            map.insert("ram_used_gb".to_string(), round2(v).into());
        }
        if let Some(v) = self.ram_total_gb {
            map.insert("ram_total_gb".to_string(), round2(v).into());
        }
        if let Some(v) = self.swap_used_gb {
            map.insert("swap_used_gb".to_string(), round2(v).into());
        }
        if let Some(v) = self.pid_rss_mb {
            map.insert("pid_rss_mb".to_string(), round1(v).into());
        }
        if map.is_empty() {
            map.insert("error".to_string(), Value::String("proc_unavailable".to_string()));
        }
        map
    }
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}
fn round1(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

/// One `Telemetry` lives for the whole worker process (owned by
/// `WorkerRuntime`'s heartbeat loop), so consecutive `sample()` calls are
/// consecutive heartbeats apart — the statefulness (CPU percent as a delta
/// between two samples, mirroring `psutil.cpu_percent(interval=None)`) now
/// lives in `platform::HostTelemetry`; this is a thin wrapper delegating to
/// it (see module doc comment).
pub struct Telemetry(crate::platform::HostTelemetry);

impl Telemetry {
    pub fn new() -> Self {
        Telemetry(crate::platform::HostTelemetry::new())
    }

    pub fn sample(&mut self) -> SystemSnapshot {
        self.0.sample()
    }
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new()
    }
}
