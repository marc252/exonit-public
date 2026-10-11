//! F8: the native FPP result payload must match the Python worker's
//! `/api/worker/fpp_result` contract (`worker.py:3580-3590` success,
//! `worker.py:3491-3497` failure). One deliberate difference: Python sends
//! `"prob_planet": null` explicitly (the native path has no prob_planet);
//! this port omits the key — the server reads it with `.get()`, so absent
//! and null are equivalent.

use exonit_core::dto::FppResultRequest;
use serde_json::{json, Map, Value};

fn candidate_map() -> Map<String, Value> {
    json!({
        "candidate_id": 42,
        "tic_id": "10757100",
        "candidate_num": 1,
        "period_days": 3.21,
        "t0": 1690.5,
        "duration_days": 0.12,
        "depth_ppm": 850.0,
        "snr": 12.3,
        "extra_server_field": "must be echoed verbatim"
    })
    .as_object()
    .unwrap()
    .clone()
}

#[test]
fn success_payload_matches_worker_contract() {
    let mut req = FppResultRequest::new(Some(42), Some("10757100".to_string()), Some(0.0123));
    req.nfpp = Some(0.0004);
    req.verdict = Some("validated".to_string());
    req.candidate = Some(candidate_map());
    let v = serde_json::to_value(&req).unwrap();
    assert_eq!(v["candidate_id"], 42);
    assert_eq!(v["tic_id"], "10757100");
    assert_eq!(v["fpp"], 0.0123);
    assert_eq!(v["nfpp"], 0.0004);
    assert_eq!(v["verdict"], "validated");
    assert_eq!(
        v["candidate"]["extra_server_field"],
        "must be echoed verbatim"
    );
    assert!(
        v.get("prob_planet").is_none(),
        "absent (≡ null for the server)"
    );
}

#[test]
fn failure_payload_matches_report_fpp_failed() {
    let mut req = FppResultRequest::new(Some(42), Some("10757100".to_string()), Some(-1.0));
    req.nfpp = Some(-1.0);
    req.verdict = Some("failed".to_string());
    req.candidate = Some(candidate_map());
    let v = serde_json::to_value(&req).unwrap();
    assert_eq!(v["fpp"], -1.0);
    assert_eq!(v["nfpp"], -1.0);
    assert_eq!(v["verdict"], "failed");
    assert_eq!(v["candidate"]["candidate_id"], 42);
}
