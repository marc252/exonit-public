//! Bit-exact parity of `fpp::transit` and `fpp::kernels` against
//! `fixtures/fpp/transit_golden.json`.

use exonit_core::fpp::kernels::{batch_eb_transit_lnl, batch_transit_lnl, BatchParams};
use exonit_core::fpp::transit::{
    compute_transit_lc, compute_z_arr, kepler_eq, transit_flux_ring, true_anomaly,
};
use serde_json::Value;

fn load() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/fpp/transit_golden.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("read transit_golden.json"))
        .expect("parse transit_golden.json")
}

fn unhex(v: &Value) -> f64 {
    let h = v.as_str().unwrap();
    let bytes: Vec<u8> = (0..h.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap())
        .collect();
    f64::from_le_bytes(bytes.try_into().unwrap())
}

fn unhex_arr(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(unhex).collect()
}

fn assert_bits(label: &str, i: usize, got: f64, want: f64) {
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{label}[{i}]: got {got:e} want {want:e}"
    );
}

#[test]
fn kepler_and_anomaly_match() {
    let doc = load();
    for (i, c) in doc["kepler"].as_array().unwrap().iter().enumerate() {
        let m = unhex(&c["m"]);
        let ecc = unhex(&c["ecc"]);
        assert_bits("kepler_eq", i, kepler_eq(m, ecc), unhex(&c["e_out"]));
        assert_bits("true_anomaly", i, true_anomaly(m, ecc), unhex(&c["nu_out"]));
    }
}

#[test]
fn ring_integration_matches() {
    let doc = load();
    for (i, c) in doc["ring"].as_array().unwrap().iter().enumerate() {
        let got = transit_flux_ring(
            unhex(&c["z"]),
            unhex(&c["p"]),
            unhex(&c["u1"]),
            unhex(&c["u2"]),
        );
        assert_bits("ring", i, got, unhex(&c["out"]));
    }
}

#[test]
fn lightcurves_match() {
    let doc = load();
    for (i, c) in doc["lightcurves"].as_array().unwrap().iter().enumerate() {
        let time = unhex_arr(&c["time"]);
        let (k, per, a_rs) = (unhex(&c["k"]), unhex(&c["period"]), unhex(&c["a_rs"]));
        let (inc, ecc, w) = (unhex(&c["inc_rad"]), unhex(&c["ecc"]), unhex(&c["w_rad"]));
        let (u1, u2) = (unhex(&c["u1"]), unhex(&c["u2"]));
        let z = compute_z_arr(&time, 0.0, per, a_rs, inc, ecc, w);
        let want_z = unhex_arr(&c["z"]);
        for (j, (&g, &w_)) in z.iter().zip(want_z.iter()).enumerate() {
            assert_bits(&format!("lc{i}_z"), j, g, w_);
        }
        let flux = compute_transit_lc(&time, k, 0.0, per, a_rs, inc, ecc, w, u1, u2);
        let want_f = unhex_arr(&c["flux"]);
        for (j, (&g, &w_)) in flux.iter().zip(want_f.iter()).enumerate() {
            assert_bits(&format!("lc{i}_flux"), j, g, w_);
        }
    }
}

#[test]
fn lnl_kernels_match() {
    let doc = load();
    let kern = &doc["kernels"];
    let time_pf = unhex_arr(&kern["time_pf"]);
    let flux_pf = unhex_arr(&kern["flux_pf"]);
    for case in kern["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let sigma = unhex(&case["sigma"]);
        let (u1, u2) = (unhex(&case["u1"]), unhex(&case["u2"]));
        let ks = unhex_arr(&case["ks"]);
        let p = BatchParams {
            ks: &ks,
            p_orbs: &unhex_arr(&case["p_orbs"]),
            incs_rad: &unhex_arr(&case["incs_rad"]),
            a_rs: &unhex_arr(&case["a_rs"]),
            eccs: &unhex_arr(&case["eccs"]),
            ws_rad: &unhex_arr(&case["ws_rad"]),
            companion_fluxratio: &unhex_arr(&case["comp_fr"]),
            companion_is_host: case["is_host"].as_bool().unwrap(),
        };
        let got = match case["kind"].as_str().unwrap() {
            "transit" => batch_transit_lnl(&time_pf, &flux_pf, sigma, u1, u2, &p),
            "eb" => {
                let ebfr = unhex_arr(&case["eb_fluxratios"]);
                batch_eb_transit_lnl(&time_pf, &flux_pf, sigma, u1, u2, &ebfr, &p)
            }
            other => panic!("unknown kind {other}"),
        };
        let want = unhex_arr(&case["lnl"]);
        for (j, (&g, &w_)) in got.iter().zip(want.iter()).enumerate() {
            assert_bits(&format!("lnl_{label}"), j, g, w_);
        }
    }
}
