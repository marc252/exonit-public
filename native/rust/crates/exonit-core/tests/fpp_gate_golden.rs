//! Wide parity gate: every frozen gate input (all eligible parity-cohort
//! TICs × up to 3 real candidates) must be bit-exact against the light
//! Python goldens.
//!
//! The gate fixtures are regenerable and NOT committed
//! (`fixtures/fpp/.gitignore`); when absent this test passes trivially so
//! the committed 12-fixture e2e test remains the always-on guard. Run the
//! gate via `parity/fpp/run_gate.sh`.

use exonit_core::fpp::{calc_fpp, CalcProbes, FppInputs};
use serde_json::Value;

fn fixtures_dir() -> std::path::PathBuf {
    std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/fpp")).to_path_buf()
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

struct Mismatch {
    stem: String,
    what: String,
}

fn check_bits(mis: &mut Vec<Mismatch>, stem: &str, what: &str, got: f64, want: f64) {
    if got.to_bits() != want.to_bits() {
        mis.push(Mismatch {
            stem: stem.to_string(),
            what: format!("{what}: got {got:e} want {want:e}"),
        });
    }
}

#[test]
fn gate_parity_all_fixtures() {
    let dir = fixtures_dir();
    let golden_path = dir.join("fpp_gate_golden.json");
    if !golden_path.exists() {
        eprintln!("fpp_gate_golden.json absent — gate skipped (regenerate via parity/fpp)");
        return;
    }
    let goldens: Value =
        serde_json::from_str(&std::fs::read_to_string(&golden_path).expect("gate goldens"))
            .expect("parse gate goldens");

    let mut mismatches: Vec<Mismatch> = Vec::new();
    let mut n_runs = 0usize;
    for (stem, golden) in goldens.as_object().expect("gate map") {
        let input: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("gate_inputs").join(format!("{stem}.json")))
                .unwrap_or_else(|e| panic!("{stem}: input read: {e}")),
        )
        .unwrap();
        let inputs = FppInputs {
            tic_id: input["tic_id"].as_i64().unwrap().to_string(),
            period: input["period"].as_f64().unwrap(),
            t0: input["t0"].as_f64().unwrap(),
            duration: input["duration"].as_f64().unwrap(),
            depth_ppm: input["depth_ppm"].as_f64().unwrap(),
            time: unhex_arr(&input["time_hex"]),
            flux: unhex_arr(&input["flux_hex"]),
            nearby_raw: input["nearby"].clone(),
            trilegal_csv: None,
        };
        let seed = golden["seed"].as_u64().unwrap() as u32;
        let mut probes = CalcProbes::default();
        let result = calc_fpp(&inputs, 100_000, seed, &mut probes);
        n_runs += 1;

        let cp = &golden["checkpoints"];
        // lnz chronology (the most bisectable light checkpoint)
        let want_z = cp["lnz"].as_array().unwrap();
        if probes.scen.lnz.len() != want_z.len() {
            mismatches.push(Mismatch {
                stem: stem.clone(),
                what: format!("lnz count {} vs {}", probes.scen.lnz.len(), want_z.len()),
            });
        } else {
            for (i, w) in want_z.iter().enumerate() {
                let (name, values) = &probes.scen.lnz[i];
                if name != w["fn"].as_str().unwrap() {
                    mismatches.push(Mismatch {
                        stem: stem.clone(),
                        what: format!("lnz[{i}] fn {name} vs {}", w["fn"]),
                    });
                    continue;
                }
                let want_vals = unhex_arr(&w["values"]);
                for (j, (&g, &wv)) in values.iter().zip(want_vals.iter()).enumerate() {
                    check_bits(
                        &mut mismatches,
                        stem,
                        &format!("lnz[{i}]({name})[{j}]"),
                        g,
                        wv,
                    );
                }
            }
        }
        // dispatch sums
        let want_d = cp["dispatch"].as_array().unwrap();
        if probes.scen.dispatch.len() == want_d.len() {
            for (i, w) in want_d.iter().enumerate() {
                let (label, _n, _head, sum, n_finite) = &probes.scen.dispatch[i];
                if *n_finite as u64 != w["n_finite"].as_u64().unwrap() {
                    mismatches.push(Mismatch {
                        stem: stem.clone(),
                        what: format!("dispatch[{i}]({label}) n_finite"),
                    });
                }
                check_bits(
                    &mut mismatches,
                    stem,
                    &format!("dispatch[{i}]({label}).sum"),
                    *sum,
                    unhex(w["sum_finite"].as_str().unwrap()),
                );
            }
        } else {
            mismatches.push(Mismatch {
                stem: stem.clone(),
                what: format!(
                    "dispatch count {} vs {}",
                    probes.scen.dispatch.len(),
                    want_d.len()
                ),
            });
        }
        // final result
        let want_res = &golden["result"];
        match (result, want_res.is_null()) {
            (None, true) => {}
            (None, false) => mismatches.push(Mismatch {
                stem: stem.clone(),
                what: "calc_fpp None, golden has result".into(),
            }),
            (Some(_), true) => mismatches.push(Mismatch {
                stem: stem.clone(),
                what: "calc_fpp Some, golden None".into(),
            }),
            (Some(r), false) => {
                check_bits(
                    &mut mismatches,
                    stem,
                    "fpp",
                    r.fpp,
                    unhex(want_res["fpp"].as_str().unwrap()),
                );
                check_bits(
                    &mut mismatches,
                    stem,
                    "nfpp",
                    r.nfpp,
                    unhex(want_res["nfpp"].as_str().unwrap()),
                );
                if r.verdict.as_str() != want_res["verdict"].as_str().unwrap() {
                    mismatches.push(Mismatch {
                        stem: stem.clone(),
                        what: "verdict".into(),
                    });
                }
            }
        }
    }

    if !mismatches.is_empty() {
        let mut msg = format!(
            "GATE FAIL: {} mismatches over {} runs\n",
            mismatches.len(),
            n_runs
        );
        for m in mismatches.iter().take(20) {
            msg.push_str(&format!("  {}: {}\n", m.stem, m.what));
        }
        panic!("{msg}");
    }
    println!("GATE PASS: {n_runs} runs, all bit-exact");
}
