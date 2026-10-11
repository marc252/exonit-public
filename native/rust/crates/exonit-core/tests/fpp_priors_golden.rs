//! Bit-exact parity of `fpp::priors` against the sampler sections of
//! `fixtures/fpp/rng_golden.json` (real fpp_native samplers, seed 42, N=64).

use exonit_core::fpp::priors::{
    sample_ecc, sample_inc, sample_q, sample_q_companion, sample_rp, sample_w,
};
use exonit_core::fpp::rng::FppRng;
use serde_json::Value;

fn load() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/fpp/rng_golden.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("read rng_golden.json"))
        .expect("parse rng_golden.json")
}

fn unhex(h: &str) -> f64 {
    let bytes: Vec<u8> = (0..h.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap())
        .collect();
    f64::from_le_bytes(bytes.try_into().unwrap())
}

#[test]
fn samplers_match_fpp_native() {
    let doc = load();
    let samplers = doc["samplers"].as_object().unwrap();
    let cases: Vec<(&str, Box<dyn Fn(&mut FppRng) -> Vec<f64>>)> = vec![
        ("sample_rp_ms120", Box::new(|r| sample_rp(r, 64, 1.20))),
        ("sample_rp_ms040", Box::new(|r| sample_rp(r, 64, 0.40))),
        ("sample_inc", Box::new(|r| sample_inc(r, 64))),
        (
            "sample_ecc_planet",
            Box::new(|r| sample_ecc(r, 64, true, 5.0)),
        ),
        (
            "sample_ecc_bin_p5",
            Box::new(|r| sample_ecc(r, 64, false, 5.0)),
        ),
        (
            "sample_ecc_bin_p20",
            Box::new(|r| sample_ecc(r, 64, false, 20.0)),
        ),
        ("sample_w", Box::new(|r| sample_w(r, 64))),
        ("sample_q_ms120", Box::new(|r| sample_q(r, 64, 1.20))),
        ("sample_q_ms050", Box::new(|r| sample_q(r, 64, 0.50))),
        ("sample_q_ms025", Box::new(|r| sample_q(r, 64, 0.25))),
        (
            "sample_q_comp_ms120",
            Box::new(|r| sample_q_companion(r, 64, 1.20)),
        ),
        (
            "sample_q_comp_ms025",
            Box::new(|r| sample_q_companion(r, 64, 0.25)),
        ),
    ];
    for (name, f) in cases {
        let want: Vec<f64> = samplers[name]
            .as_array()
            .unwrap_or_else(|| panic!("{name} missing"))
            .iter()
            .map(|h| unhex(h.as_str().unwrap()))
            .collect();
        let mut rng = FppRng::new(42);
        let got = f(&mut rng);
        assert_eq!(got.len(), want.len(), "{name}: length");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{name}[{i}]: got {g:e} want {w:e}"
            );
        }
    }
}
