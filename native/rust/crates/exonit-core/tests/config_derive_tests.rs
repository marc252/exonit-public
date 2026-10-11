//! Ported from the `AgentConfigDeriver` tests in
//! `Sources/ExoNitContractTests/main.swift` (Swift reference worker).

use exonit_core::config_derive;
use serde_json::Value;
use std::path::Path;

/// Swift: `testMultiInstanceConfigDerivation`.
#[test]
fn multi_instance_config_derivation() {
    let directory = Path::new("/tmp/exonit-agent-fixture");
    let base = br#"{"server_url":"http://localhost:8400","token":"t","worker_name":"base","gui_port":8402,"fits_cache_dir":"fits_cache","cpu_threads":9,"fpp_cpu_cores":9}"#;
    let overrides = br#"{"gpu_enabled":true,"gui_port":9999,"cpu_threads":99}"#;
    let result = config_derive::derive(base, Some(overrides), "worker-m4-3", 3, 4, directory, 10)
        .expect("derive");
    let object: Value = serde_json::from_slice(&result).expect("parse derived config");

    assert_eq!(object["worker_name"], "worker-m4-3", "derived name");
    assert_eq!(object["gui_port"], 8404, "base GUI port wins over override");
    assert_eq!(object["gui_enabled"], false, "secondary GUI disabled");
    assert_eq!(object["cpu_threads"], 2, "75% CPU share");
    assert_eq!(object["fpp_cpu_cores"], 2, "FPP CPU share");
    assert_eq!(object["gpu_enabled"], true, "host override retained");
    // `config_derive::derive` builds this via `directory.join(...)`
    // (config_derive.rs), which inserts the PLATFORM-NATIVE separator for
    // the appended segment ("\\" on Windows, "/" on Unix) while leaving
    // `directory`'s own embedded "/" separators untouched — so the derived
    // path is legitimately "/tmp/exonit-agent-fixture\\fits_cache_3" on
    // Windows, not a bug. Comparing against `directory.join(...)` directly
    // (rather than a hardcoded forward-slash string) makes this assertion
    // platform-agnostic instead of encoding a Unix assumption (Task 2,
    // docs/plans/2026-08-06-win32-port-pcmarc.md: found failing on <host>,
    // fixed here rather than in config_derive.rs, which was already
    // correct).
    let expected_cache_dir = directory.join("fits_cache_3");
    assert_eq!(
        object["fits_cache_dir"].as_str().map(Path::new),
        Some(expected_cache_dir.as_path()),
        "cache path"
    );
    assert_eq!(object["log_file"], "worker-worker-m4-3.log", "log path");
}

/// Swift: `testSingleInstanceOverrides`.
#[test]
fn single_instance_overrides() {
    let base = br#"{"server_url":"http://localhost:8400","token":"t","worker_name":"base","gui_port":8402,"cpu_threads":6}"#;
    let overrides = br#"{"gpu_enabled":true,"cpu_threads":4}"#;
    let result = config_derive::apply_single_instance_overrides(base, overrides).expect("apply overrides");
    let object: Value = serde_json::from_slice(&result).expect("parse derived config");

    assert_eq!(object["worker_name"], "base", "single instance name untouched");
    assert_eq!(object["gui_port"], 8402, "single GUI untouched");
    assert_eq!(object["cpu_threads"], 4, "single override applied");
    assert_eq!(object["gpu_enabled"], true, "single GPU override");
}

/// Additional case (no direct Swift counterpart, but pins
/// `AgentConfigDeriver.derive`'s guard clause, which the Swift suite doesn't
/// test directly either): instance_index out of [1, instances] must be
/// rejected rather than silently wrapping/clamping.
#[test]
fn derive_rejects_invalid_instance_index() {
    let base = br#"{"server_url":"http://localhost:8400","token":"t","worker_name":"base"}"#;
    let result = config_derive::derive(base, None, "worker-x", 5, 4, Path::new("/tmp"), 8);
    assert!(result.is_err(), "instance_index > instances must fail");
}

/// Additional case: base64 override decoding round-trips and rejects
/// malformed input, mirroring `AgentConfigDeriver.decodeOverrides` (used by
/// `AgentController.start` — see `pidlock_tests.rs` for the
/// refused-malformed-overrides path through the full `start()` flow).
#[test]
fn decode_overrides_roundtrip_and_rejects_garbage() {
    // "e30=" is base64("{}"), the same fixture value the Swift suite uses in
    // `testAgentStartInvocation`.
    let decoded = config_derive::decode_overrides("e30=").expect("decode e30=");
    assert_eq!(decoded, b"{}");
    assert!(config_derive::decode_overrides("not-valid-base64!!").is_err());
    assert!(config_derive::decode_overrides("bm90LWpzb24=").is_err()); // base64("not-json")
}
