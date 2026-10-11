//! Ported from `testExtendedProtocolDTOs` in
//! `Sources/ExoNitContractTests/main.swift` (Swift reference worker), plus
//! one DTO-layer-only addition (see below). Network-transport contract
//! tests that need a mocked `URLSession`/HTTP client
//! (`testRegisterHTTPContract`, `testTaskHTTPContract`,
//! `testHeartbeatContract`, `testUnknownHeartbeatCommand`,
//! `testExtendedAPIRoutes`, `testBinaryAPIRoutes`) are out of scope for F0
//! — they exercise `WorkerAPIClient`, which is F0b's `api.rs`.

use exonit_core::dto::{
    BackfillTaskEnvelope, FfiResponse, HeartbeatCommand, NearbyResponse, UpdateManifest,
};

/// Swift: `testExtendedProtocolDTOs`.
#[test]
fn extended_protocol_dtos() {
    let backfill_json = r#"{"task":{"candidate_id":42,"tic_id":150430484,"candidate_num":1,"period_days":5.154,"t0":1234.5,"duration_days":0.15,"depth_ppm":900,"snr":14.2,"n_sectors":3}}"#;
    let envelope: BackfillTaskEnvelope = serde_json::from_str(backfill_json).expect("decode backfill");
    let task = envelope.task.expect("backfill task present");
    assert_eq!(task.candidate_id, 42, "backfill candidate");
    assert_eq!(task.tic_id, "150430484", "flexible numeric TIC");

    let ffi_json = r#"{"tic_id":12345,"sectors":[{"sector":14,"source":"tglc","time":[0,1],"flux":[1,0.99]}]}"#;
    let ffi: FfiResponse = serde_json::from_str(ffi_json).expect("decode ffi");
    assert_eq!(ffi.tic_id, "12345", "FFI TIC");
    assert_eq!(ffi.sectors[0].effective_quality(), vec![0, 0], "missing quality defaults to zero");

    let nearby_json = r#"{"tic_id":"12345","stars":[{"ID":12345,"ra":10,"dec":-20,"mass":null,"rad":1.1,"Teff":5777,"Tmag":10.5,"plx":3.2}],"gal_b":30.2,"source":"cache"}"#;
    let nearby: NearbyResponse = serde_json::from_str(nearby_json).expect("decode nearby");
    assert_eq!(nearby.stars[0].id, "12345", "nearby flexible ID");
    assert_eq!(nearby.stars[0].mass, None, "nearby nullable mass");

    let manifest_json = r#"{"files":[{"filename":"models/model.onnx","sha256":"abc"}]}"#;
    let manifest: UpdateManifest = serde_json::from_str(manifest_json).expect("decode manifest");
    assert_eq!(manifest.files[0].filename, "models/model.onnx", "update nested path");
}

/// Pure-data-layer slice of Swift's `testUnknownHeartbeatCommand`: proves an
/// unrecognised heartbeat command string fails to decode into
/// `HeartbeatCommand` at all (`serde` rejects it, same as
/// `HeartbeatCommand(rawValue:)` returning `nil` in Swift). The
/// network-level behaviour this feeds — turning that failure into
/// `WorkerAPIError::malformed_response` — belongs to F0b's `api.rs`
/// (`WorkerAPIClient::heartbeat`), not this crate yet.
#[test]
fn unknown_heartbeat_command_fails_to_decode() {
    let result: Result<HeartbeatCommand, _> = serde_json::from_str("\"format_disk\"");
    assert!(result.is_err(), "unknown heartbeat command must not decode");

    let pause: HeartbeatCommand = serde_json::from_str("\"pause\"").expect("decode pause");
    assert_eq!(pause, HeartbeatCommand::Pause);
    let re_register: HeartbeatCommand = serde_json::from_str("\"re-register\"").expect("decode re-register");
    assert_eq!(re_register, HeartbeatCommand::ReRegister);
}
