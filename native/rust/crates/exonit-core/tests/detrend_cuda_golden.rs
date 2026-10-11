//! F4b golden test: CUDA wotan biweight detrend (`detrend_cuda::
//! CudaDetrendBackend`) vs the CPU implementation this crate has had since
//! F1 (`detrend::flatten_with_params`, mask=None).
//!
//! The whole file is `#![cfg(feature = "cuda")]`; the test itself is
//! additionally `#[ignore = "needs NVIDIA GPU"]` — run explicitly on
//! <host> with `cargo test --features cuda --test detrend_cuda_golden
//! -- --ignored --nocapture` (see `bls::cuda`'s module doc comment for the
//! `LD_LIBRARY_PATH` invocation this also needs).
//!
//! # Fixture
//!
//! Reuses `fixtures/wotan_biweight_golden.json` (the same fixture
//! `wotan_biweight_golden.rs`'s F1 CPU-vs-Python-wotan golden test already
//! validates `detrend::flatten` against) as a realistic, densely-sampled
//! synthetic light curve — NOT to re-derive that Python parity (already
//! covered), but as a convenient, already-committed source of realistic
//! `time`/`flux` arrays with a genuine `break_tolerance`-triggering gap
//! (max gap 1.02d > 0.5d break_tolerance, so this exercises
//! `detrend_biweight`'s multi-segment splitting, not just a single
//! contiguous window run). The fixture's `time`/`flux` contain a handful of
//! `null` (NaN) entries used by the F1 test's NaN-compression case —
//! `detrend_cuda::CudaDetrendBackend` has no NaN handling BY DESIGN (mirrors
//! `wotan_cuda.py`'s documented "callers pre-filter" contract — see that
//! module's doc comment), so this test filters them out up front and feeds
//! the SAME NaN-free arrays to both backends. `detrend::flatten_with_params`
//! is also called with `transit_mask: None` on that identical NaN-free
//! input, so both sides see byte-identical input arrays.
//!
//! # Tolerance
//!
//! `wotan_cuda.py`'s module doc comment states the production parity
//! contract for this exact kernel: "matches wotan.flatten ... within ~1e-6
//! relative on flat-flux samples"; `wotan_detrend.py`'s module doc comment
//! is more specific still: "bit-exact to wotan.flatten (rel <= 4e-15 across
//! 150k thread observations)". `detrend::flatten_with_params` (the Rust CPU
//! side of this comparison) is itself already validated against real
//! `wotan.flatten` output at `1e-6`-class tolerance by
//! `wotan_biweight_golden.rs` (F1) — so CUDA-vs-Rust-CPU inherits that
//! chain. This test uses `1e-6` relative as the gate (the conservative,
//! documented headline number) and logs the actual observed deviation —
//! see this task's final report for the real number measured on <host>,
//! which Python's own numbers suggest should land far tighter (`1e-10`-`1e-15`
//! class) in practice.

#![cfg(feature = "cuda")]

use exonit_core::detrend::{self};
use exonit_core::detrend_cuda::CudaDetrendBackend;
use serde::Deserialize;

#[derive(Deserialize)]
struct WotanGolden {
    time: Vec<Option<f64>>,
    flux: Vec<Option<f64>>,
    parameters: Parameters,
}

#[derive(Deserialize)]
struct Parameters {
    window_length: f64,
    edge_cutoff: f64,
    break_tolerance: f64,
    cval: f64,
}

fn gpu() -> CudaDetrendBackend {
    CudaDetrendBackend::probe().expect(
        "CudaDetrendBackend::probe() returned None — this test needs a real NVIDIA GPU with a \
         working driver + NVRTC on LD_LIBRARY_PATH (see bls::cuda's module doc comment, which this \
         module's doc comment points to); it is meant to run on <host> only, invoked with \
         `cargo test --features cuda -- --ignored`",
    )
}

#[test]
#[ignore = "needs NVIDIA GPU"]
fn cuda_biweight_matches_cpu_biweight_on_wotan_fixture() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/wotan_biweight_golden.json");
    let golden: WotanGolden =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read wotan_biweight_golden.json")).expect("parse wotan_biweight_golden.json");

    // Filter out the fixture's NaN (null) entries — see module doc comment:
    // `detrend_cuda` has no NaN handling by design, unlike `detrend::flatten`.
    let mut time: Vec<f64> = Vec::new();
    let mut flux: Vec<f64> = Vec::new();
    for (t, f) in golden.time.iter().zip(golden.flux.iter()) {
        if let (Some(t), Some(f)) = (t, f) {
            time.push(*t);
            flux.push(*f);
        }
    }
    assert!(time.len() > 100, "expected a substantial NaN-free subset, got {}", time.len());
    // Sanity: this fixture genuinely exercises the multi-segment
    // break_tolerance path, not just one contiguous window run.
    let max_gap = time.windows(2).map(|w| w[1] - w[0]).fold(0.0_f64, f64::max);
    assert!(max_gap > golden.parameters.break_tolerance, "fixture must contain a break_tolerance-triggering gap (max_gap={max_gap})");

    let cpu = detrend::flatten_with_params(
        &time,
        &flux,
        golden.parameters.window_length,
        golden.parameters.edge_cutoff,
        golden.parameters.break_tolerance,
        golden.parameters.cval,
        None,
    )
    .expect("CPU biweight flatten");

    let backend = gpu();
    let (cuda_flat, _trend) = backend
        .detrend_biweight(&time, &flux, golden.parameters.window_length, golden.parameters.edge_cutoff, golden.parameters.cval, golden.parameters.break_tolerance)
        .expect("CUDA biweight detrend");

    assert_eq!(cpu.len(), cuda_flat.len());
    assert_eq!(cpu.len(), time.len());

    let mut max_rel: f64 = 0.0;
    let mut compared = 0usize;
    let mut both_nan = 0usize;
    for i in 0..cpu.len() {
        let (c, g) = (cpu[i], cuda_flat[i]);
        match (c.is_finite(), g.is_finite()) {
            (false, false) => both_nan += 1,
            (true, true) => {
                let rel = (c - g).abs() / c.abs().max(1e-12);
                max_rel = max_rel.max(rel);
                compared += 1;
                assert!(rel < 1e-6, "index {i}: cpu={c} cuda={g} (rel {rel:.3e}) exceeds 1e-6 tolerance");
            }
            _ => panic!("index {i}: finiteness mismatch cpu={c} cuda={g} -- CPU and CUDA disagree on which points are edge-cutoff/NaN"),
        }
    }
    eprintln!(
        "cuda_biweight_matches_cpu_biweight_on_wotan_fixture: compared={compared} both_nan(edge)={both_nan} max_rel={max_rel:.3e} \
         (n={}, max_gap={max_gap:.3}d)",
        cpu.len()
    );
    assert!(compared > 100, "expected most of the {} points to be finite on both sides, got {compared}", cpu.len());
}
