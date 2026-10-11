//! End-to-end bit-exact parity of `fpp::calc_fpp` against
//! `fixtures/fpp/fpp_native_golden.json` (12 frozen-input fixtures with
//! chronological checkpoints).

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

fn assert_bits(ctx: &str, got: f64, want: f64) {
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{ctx}: got {got:e} want {want:e}"
    );
}

fn assert_bits_arr(ctx: &str, got: &[f64], want: &[f64]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{ctx}: length {} vs {}",
        got.len(),
        want.len()
    );
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_bits(&format!("{ctx}[{i}]"), g, w);
    }
}

#[test]
fn calc_fpp_matches_python_reference() {
    let dir = fixtures_dir();
    let goldens: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("fpp_native_golden.json")).expect("goldens"),
    )
    .expect("parse goldens");

    for (stem, golden) in goldens.as_object().expect("golden map") {
        let input: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("inputs").join(format!("{stem}.json")))
                .unwrap_or_else(|e| panic!("{stem}: input read: {e}")),
        )
        .unwrap();
        let trilegal_csv = input["trilegal_csv"].as_str().map(|name| {
            std::fs::read_to_string(dir.join("inputs").join(name)).expect("trilegal csv")
        });
        let inputs = FppInputs {
            tic_id: input["tic_id"].as_i64().unwrap().to_string(),
            period: input["period"].as_f64().unwrap(),
            t0: input["t0"].as_f64().unwrap(),
            duration: input["duration"].as_f64().unwrap(),
            depth_ppm: input["depth_ppm"].as_f64().unwrap(),
            time: unhex_arr(&input["time_hex"]),
            flux: unhex_arr(&input["flux_hex"]),
            nearby_raw: input["nearby"].clone(),
            trilegal_csv,
        };
        let seed = golden["seed"].as_u64().unwrap() as u32;
        assert_eq!(seed, input["seed"].as_u64().unwrap() as u32, "{stem}: seed");

        let mut probes = CalcProbes::default();
        let result = calc_fpp(&inputs, 100_000, seed, &mut probes);

        let cp = &golden["checkpoints"];

        // phase_fold
        let pf = &cp["phase_fold"];
        let (t_pf, f_pf) = probes.phase_fold.as_ref().expect("phase_fold probe");
        assert_eq!(t_pf.len() as u64, pf["n"].as_u64().unwrap(), "{stem}: pf n");
        assert_bits_arr(&format!("{stem}:time_pf"), t_pf, &unhex_arr(&pf["time_pf"]));
        assert_bits_arr(&format!("{stem}:flux_pf"), f_pf, &unhex_arr(&pf["flux_pf"]));

        // flux_ratios
        let want_fr = cp["flux_ratios"].as_array().unwrap();
        assert_eq!(
            probes.flux_ratios.len(),
            want_fr.len(),
            "{stem}: flux_ratios len"
        );
        for (i, w) in want_fr.iter().enumerate() {
            let (id, fr, td) = &probes.flux_ratios[i];
            assert_eq!(id, w["ID"].as_str().unwrap(), "{stem}: fr[{i}] id");
            assert_bits(
                &format!("{stem}:fr[{i}]"),
                *fr,
                unhex(w["fluxratio"].as_str().unwrap()),
            );
            assert_bits(
                &format!("{stem}:td[{i}]"),
                *td,
                unhex(w["tdepth"].as_str().unwrap()),
            );
        }

        // renorm calls
        let want_rn = cp["renorm"].as_array().unwrap();
        assert_eq!(probes.renorm.len(), want_rn.len(), "{stem}: renorm count");
        for (i, w) in want_rn.iter().enumerate() {
            let (fr, sig, head) = &probes.renorm[i];
            assert_bits(
                &format!("{stem}:rn[{i}].fr"),
                *fr,
                unhex(w["fluxratio"].as_str().unwrap()),
            );
            assert_bits(
                &format!("{stem}:rn[{i}].sigma"),
                *sig,
                unhex(w["sigma_in"].as_str().unwrap()),
            );
            assert_bits_arr(
                &format!("{stem}:rn[{i}].flux"),
                head,
                &unhex_arr(&w["flux_head"]),
            );
        }

        // samplers (chronological)
        let want_s = cp["samplers"].as_array().unwrap();
        assert_eq!(
            probes.scen.samplers.len(),
            want_s.len(),
            "{stem}: sampler count"
        );
        for (i, w) in want_s.iter().enumerate() {
            let (name, n, head) = &probes.scen.samplers[i];
            assert_eq!(name, w["fn"].as_str().unwrap(), "{stem}: sampler[{i}] fn");
            assert_eq!(
                *n as u64,
                w["n"].as_u64().unwrap(),
                "{stem}: sampler[{i}] n"
            );
            assert_bits_arr(
                &format!("{stem}:sampler[{i}]({name})"),
                head,
                &unhex_arr(&w["head"]),
            );
        }

        // dispatch (chronological)
        let want_d = cp["dispatch"].as_array().unwrap();
        assert_eq!(
            probes.scen.dispatch.len(),
            want_d.len(),
            "{stem}: dispatch count"
        );
        for (i, w) in want_d.iter().enumerate() {
            let (label, n, head, sum, n_finite) = &probes.scen.dispatch[i];
            assert_eq!(
                label,
                w["label"].as_str().unwrap(),
                "{stem}: dispatch[{i}] label"
            );
            assert_eq!(
                *n as u64,
                w["n"].as_u64().unwrap(),
                "{stem}: dispatch[{i}] n"
            );
            assert_eq!(
                *n_finite as u64,
                w["n_finite"].as_u64().unwrap(),
                "{stem}: dispatch[{i}] n_finite"
            );
            assert_bits_arr(
                &format!("{stem}:dispatch[{i}]({label})"),
                head,
                &unhex_arr(&w["head"]),
            );
            assert_bits(
                &format!("{stem}:dispatch[{i}].sum"),
                *sum,
                unhex(w["sum_finite"].as_str().unwrap()),
            );
        }

        // lnZ (chronological)
        let want_z = cp["lnz"].as_array().unwrap();
        assert_eq!(probes.scen.lnz.len(), want_z.len(), "{stem}: lnz count");
        for (i, w) in want_z.iter().enumerate() {
            let (name, values) = &probes.scen.lnz[i];
            assert_eq!(name, w["fn"].as_str().unwrap(), "{stem}: lnz[{i}] fn");
            assert_bits_arr(
                &format!("{stem}:lnz[{i}]({name})"),
                values,
                &unhex_arr(&w["values"]),
            );
        }

        // final result
        let want_res = &golden["result"];
        if want_res.is_null() {
            assert!(result.is_none(), "{stem}: expected None result");
            continue;
        }
        let r = result.unwrap_or_else(|| panic!("{stem}: calc_fpp returned None"));
        assert_bits(
            &format!("{stem}:fpp"),
            r.fpp,
            unhex(want_res["fpp"].as_str().unwrap()),
        );
        assert_bits(
            &format!("{stem}:nfpp"),
            r.nfpp,
            unhex(want_res["nfpp"].as_str().unwrap()),
        );
        assert_eq!(
            r.verdict.as_str(),
            want_res["verdict"].as_str().unwrap(),
            "{stem}: verdict"
        );
        assert_eq!(
            r.high_crowding,
            want_res["high_crowding"].as_bool().unwrap(),
            "{stem}: crowding"
        );
        assert_eq!(
            r.n_nearby as u64,
            want_res["n_nearby"].as_u64().unwrap(),
            "{stem}: n_nearby"
        );
        assert_eq!(
            r.trilegal_used,
            want_res["trilegal_used"].as_bool().unwrap(),
            "{stem}: trilegal"
        );
        let want_gal_b = want_res["gal_b"].as_f64().unwrap();
        assert_bits(&format!("{stem}:gal_b"), r.gal_b, want_gal_b);
    }
}
