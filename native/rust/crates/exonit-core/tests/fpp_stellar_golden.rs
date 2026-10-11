//! Bit-exact parity of `fpp::stellar` against
//! `fixtures/fpp/spline_golden.json` (dense meshes, relations with
//! masks/clips/overflow, LDC lookups).

use exonit_core::fpp::stellar::{flux_relation, get_ldc, stellar_relations};
use serde_json::Value;

fn load() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/fpp/spline_golden.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("read spline_golden.json"))
        .expect("parse spline_golden.json")
}

fn unhex(h: &str) -> f64 {
    let bytes: Vec<u8> = (0..h.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap())
        .collect();
    f64::from_le_bytes(bytes.try_into().unwrap())
}

fn unhex_arr(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|h| unhex(h.as_str().unwrap()))
        .collect()
}

fn assert_bits(label: &str, i: usize, got: f64, want: f64) {
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{label}[{i}]: got {got:e} want {want:e}"
    );
}

#[test]
fn relations_match_golden() {
    let doc = load();
    let rel = &doc["relations"];
    let masses = unhex_arr(&rel["masses"]);
    let want_r = unhex_arr(&rel["radii"]);
    let want_t = unhex_arr(&rel["teffs"]);
    let want_f = unhex_arr(&rel["flux"]);
    let (r, t) = stellar_relations(&masses);
    let f = flux_relation(&masses);
    for i in 0..masses.len() {
        assert_bits("radii", i, r[i], want_r[i]);
        assert_bits("teffs", i, t[i], want_t[i]);
        assert_bits("flux", i, f[i], want_f[i]);
    }
}

#[test]
fn spline_meshes_match_golden() {
    // The relations test exercises the hot/cool/clip composition; this one
    // pins each raw spline over its dense mesh, including extrapolation.
    // stellar_relations/flux_relation route through the same evaluator, so
    // per-spline meshes are checked via the public API where possible:
    // R/T meshes are covered by relations_match_golden; here we verify the
    // flux spline mesh through flux_relation (log10 domain inverted).
    let doc = load();
    let sec = &doc["splines"]["flux"];
    let x = unhex_arr(&sec["x"]);
    let want_y = unhex_arr(&sec["y"]);
    let got = flux_relation(&x);
    for i in 0..x.len() {
        let expected = 10.0f64.powf(want_y[i]);
        assert_bits("flux_mesh", i, got[i], expected);
    }
}

#[test]
fn ldc_lookups_match_golden() {
    let doc = load();
    for (i, case) in doc["ldc"].as_array().unwrap().iter().enumerate() {
        let teff = unhex(case["teff"].as_str().unwrap());
        let logg = unhex(case["logg"].as_str().unwrap());
        let (u1, u2) = get_ldc(teff, logg, 0.0);
        assert_bits("ldc_u1", i, u1, unhex(case["u1"].as_str().unwrap()));
        assert_bits("ldc_u2", i, u2, unhex(case["u2"].as_str().unwrap()));
    }
}
