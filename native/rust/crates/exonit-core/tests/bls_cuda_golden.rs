//! F4a golden tests: CUDA (`bls::cuda::CudaBlsBackend`) vs CPU
//! (`bls::reference`/`bls::two_pass`) on the PRIMARY BLS path.
//!
//! The whole file is `#![cfg(feature = "cuda")]` — without the `cuda`
//! Cargo feature, `bls::cuda` does not exist, so this file compiles to
//! nothing (0 tests reported, not a build failure) on `cargo test --all`.
//! Every test in it is additionally `#[ignore = "needs NVIDIA GPU"]`:
//! `CudaBlsBackend::probe()` fails (with a specific `eprintln!`, not a
//! panic) on any host without a real CUDA GPU + driver + NVRTC — which is
//! every host EXCEPT <host>. Run explicitly with:
//!
//! ```text
//! cargo test --features cuda -- --ignored
//! ```
//!
//! on <host>, with `LD_LIBRARY_PATH` covering the NVRTC .so (see
//! `bls::cuda`'s module doc comment and docs/PLAN.md's F4a section for the
//! exact path/invocation used).
//!
//! # Tolerances
//!
//! `period`/`snr`/`depth_ppm` relative tolerances (`1e-4`/`5e-3`/`5e-3`)
//! are NOT invented here — they are copied from the Python worker's own
//! CUDA-vs-CPU validation, `~/exonit/tests/test_bls_cuda_parity.py`
//! `TestTwoPassParity` (`test_period_matches`/`test_snr_matches`/
//! `test_depth_matches`), which is the production gate for
//! `bls_gpu_two_pass` vs `bls_two_pass` — the exact two functions this
//! module ports. Re-using them (rather than re-deriving new numbers) keeps
//! this Rust port honest to the SAME bar production CUDA BLS already
//! clears.
//!
//! `duration`/`t0` tolerances and the single-pass `bls_golden.json`
//! fixture comparison have no Python equivalent (`test_bls_cuda_parity.py`
//! only checks period/snr/depth, and only for the two-pass path) — those
//! are PROVISIONAL, inferred by symmetry with the covered fields (same
//! order of magnitude as the period tolerance, since duration/t0 come from
//! the same period-grid/bin-index discretization) and marked as such below.
//! Tighten (or loosen, with a comment explaining why) once real numbers
//! from a <host> run are in hand — see this session's final report for
//! the actual observed deviations.

#![cfg(feature = "cuda")]

use exonit_core::bls::cuda::CudaBlsBackend;
use exonit_core::bls::quick::RealQuickBlsSearch;
use exonit_core::bls::reference::BlsParams;
use exonit_core::bls::two_pass;
use exonit_core::fits::{self, PhotometrySource};
use exonit_core::preprocess::{self, DetrendMode};
use serde::Deserialize;
use std::path::PathBuf;

fn gpu() -> CudaBlsBackend {
    CudaBlsBackend::probe().expect(
        "CudaBlsBackend::probe() returned None — this test needs a real NVIDIA GPU with a \
         working driver + NVRTC on LD_LIBRARY_PATH (see bls::cuda's module doc comment); it is \
         meant to run on <host> only, invoked with `cargo test --features cuda -- --ignored`",
    )
}

/// Same synthetic transit light curve as `bls_golden.rs`'s
/// `model_light_curve()` — duplicated rather than imported (test binaries
/// are independent crates; matches the existing convention of each golden
/// test file owning its own fixture construction, e.g.
/// `test_bls_cuda_parity.py`'s module doc comment: "each backend's test
/// module owning its own fixtures").
fn model_light_curve() -> (Vec<f64>, Vec<f64>) {
    let period = 10.0 / 3.0;
    let time: Vec<f64> = (0..360).map(|i| i as f64 / 12.0).collect();
    let flux: Vec<f64> = time
        .iter()
        .map(|&value| {
            let phase = (value / period) % 1.0;
            if (0.20..0.28).contains(&phase) { 0.988 } else { 1.0 }
        })
        .collect();
    (time, flux)
}

#[derive(Deserialize)]
struct BlsGolden {
    period: f64,
    power: f64,
    duration: f64,
    t0: f64,
    depth: f64,
    #[allow(dead_code)]
    depth_ppm: f64,
    snr: f64,
}

fn relative_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(1e-12)
}

/// Single-pass `CudaBlsBackend::search` (the direct port of
/// `bls_cuda.bls_gpu`) against `fixtures/bls_golden.json` — the same
/// fixture `bls_golden.rs`'s `reference_search_matches_swift_golden`
/// checks the CPU path against, at `1e-14` tolerance. GPU floating-point
/// arithmetic is NOT guaranteed bit-identical to CPU even for the same
/// summation order (NVIDIA's ptxas may contract `a*b+c` into a single FMA
/// instruction where CPU code does not, rounding once instead of twice) —
/// PROVISIONAL tolerance `1e-9` relative reflects that expectation (much
/// tighter than the Python two-pass numbers below, since both sides here
/// are the SAME algorithm/summation-order port, not two independently
/// implemented backends); see this test file's module doc comment.
#[test]
#[ignore = "needs NVIDIA GPU"]
fn cuda_single_pass_matches_bls_golden_fixture() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/bls_golden.json");
    let golden: BlsGolden =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read bls_golden.json")).expect("parse bls_golden.json");

    let (time, flux) = model_light_curve();
    let params = BlsParams {
        period_minimum: 2.0,
        period_maximum: 5.0,
        period_count: 121,
        bin_count: 50,
        duration_step_count: 7,
        duration_minimum_fraction: 0.04,
        duration_maximum_fraction: 0.12,
    };
    let backend = gpu();
    let result = backend.search(&time, &flux, &params).expect("CUDA BLS search");

    let period_rel = relative_error(result.period, golden.period);
    let power_rel = relative_error(result.power, golden.power);
    let duration_rel = relative_error(result.duration, golden.duration);
    let t0_rel = relative_error(result.t0, golden.t0);
    let depth_rel = relative_error(result.depth, golden.depth);
    let snr_rel = relative_error(result.snr, golden.snr);
    eprintln!(
        "cuda_single_pass_matches_bls_golden_fixture: rel errors -- period={period_rel:.3e} \
         power={power_rel:.3e} duration={duration_rel:.3e} t0={t0_rel:.3e} depth={depth_rel:.3e} \
         snr={snr_rel:.3e}"
    );

    assert!(period_rel < 1e-9, "period rel {period_rel:.3e} (cuda={}, cpu={})", result.period, golden.period);
    assert!(power_rel < 1e-9, "power rel {power_rel:.3e} (cuda={}, cpu={})", result.power, golden.power);
    // PROVISIONAL (no Python equivalent — see module doc comment).
    assert!(duration_rel < 1e-9, "duration rel {duration_rel:.3e} (cuda={}, cpu={})", result.duration, golden.duration);
    assert!(t0_rel < 1e-9, "t0 rel {t0_rel:.3e} (cuda={}, cpu={})", result.t0, golden.t0);
    assert!(depth_rel < 1e-9, "depth rel {depth_rel:.3e} (cuda={}, cpu={})", result.depth, golden.depth);
    // SDE is a median/MAD statistic over the whole spectrum, more sensitive
    // to any per-point drift than the other fields — same looser factor
    // `bls_golden.rs` uses for its own CPU-only SDE check (`1e-10` absolute
    // there; kept relative and slightly wider here for the same reason).
    assert!(snr_rel < 1e-6, "snr rel {snr_rel:.3e} (cuda={}, cpu={})", result.snr, golden.snr);
}

/// `CudaBlsBackend::two_pass_search` (the direct port of
/// `bls_cuda.bls_gpu_two_pass`) against `bls::two_pass::search` (CPU) on
/// the same synthetic light curve `bls_golden.rs`'s
/// `two_pass_search_matches_swift_golden` uses (period_count=10_000,
/// bin_count=120, duration_step_count=10). Tolerances on period/snr/depth
/// are copied verbatim from
/// `test_bls_cuda_parity.py::TestTwoPassParity` (see module doc comment);
/// duration/t0 are PROVISIONAL at the same `1e-4` order as period.
#[test]
#[ignore = "needs NVIDIA GPU"]
fn cuda_two_pass_matches_cpu_two_pass_on_synthetic_lc() {
    let (time, flux) = model_light_curve();
    let params = BlsParams { period_count: 10_000, bin_count: 120, duration_step_count: 10, ..BlsParams::default() };

    let cpu = two_pass::search(&time, &flux, &params, 2).expect("CPU two-pass BLS search");
    let backend = gpu();
    let gpu_result = backend.two_pass_search(&time, &flux, &params).expect("CUDA two-pass BLS search");

    let period_rel = relative_error(gpu_result.period, cpu.period);
    let snr_rel = relative_error(gpu_result.snr, cpu.snr);
    let depth_ppm_rel = relative_error(gpu_result.depth_ppm, cpu.depth_ppm.max(1.0));
    let duration_rel = relative_error(gpu_result.duration, cpu.duration);
    let t0_rel = relative_error(gpu_result.t0, cpu.t0);
    eprintln!(
        "cuda_two_pass_matches_cpu_two_pass_on_synthetic_lc: rel errors -- period={period_rel:.3e} \
         snr={snr_rel:.3e} depth_ppm={depth_ppm_rel:.3e} duration={duration_rel:.3e} t0={t0_rel:.3e} \
         (cpu period={}, snr={}, depth_ppm={})",
        cpu.period, cpu.snr, cpu.depth_ppm
    );

    // From test_bls_cuda_parity.py::TestTwoPassParity, verbatim tolerances.
    assert!(period_rel < 1e-4, "period mismatch: cpu={} gpu={} (rel {period_rel:.3e})", cpu.period, gpu_result.period);
    assert!(snr_rel < 5e-3, "SNR drift: cpu={} gpu={} (rel {snr_rel:.3e})", cpu.snr, gpu_result.snr);
    assert!(
        depth_ppm_rel < 5e-3,
        "depth drift: cpu={}ppm gpu={}ppm (rel {depth_ppm_rel:.3e})",
        cpu.depth_ppm,
        gpu_result.depth_ppm
    );
    // PROVISIONAL (no Python equivalent).
    assert!(duration_rel < 1e-4, "duration mismatch: cpu={} gpu={} (rel {duration_rel:.3e})", cpu.duration, gpu_result.duration);
    assert!(t0_rel < 1e-4, "t0 mismatch: cpu={} gpu={} (rel {t0_rel:.3e})", cpu.t0, gpu_result.t0);
}

/// Same CUDA-vs-CPU two-pass comparison, but on the real deterministic
/// TGLC pipeline (TIC 10756829, the fixture `fits_pending_golden.rs` uses
/// for `tglc_10756829_preprocess_matches_swift_golden`) with
/// production-shaped BLS parameters (`period_count=50_000`,
/// `bin_count=200`, `duration_step_count=15` — `bls_gpu_two_pass`'s own
/// Python defaults) instead of the small synthetic curve above. This is
/// the "real light curve, real point count/cadence" half of the golden
/// gate the task asked for, distinct from the synthetic-curve tests.
#[test]
#[ignore = "needs NVIDIA GPU"]
fn cuda_two_pass_matches_cpu_two_pass_on_tglc_pipeline() {
    let data_dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data"));
    let paths = [
        data_dir.join("hlsp_tglc_tess_ffi_gaiaid-2080664784495597312-s0014-cam1-ccd1_tess_v1_llc.fits"),
        data_dir.join("hlsp_tglc_tess_ffi_gaiaid-2080664784495597312-s0015-cam1-ccd2_tess_v1_llc.fits"),
    ];
    let curves: Vec<_> = paths.iter().map(|p| fits::read_light_curve(p).expect("read TGLC FITS")).collect();
    assert!(curves.iter().all(|c| c.source == PhotometrySource::Tglc), "fixture sectors must be TGLC");

    // Same call `task_processor.rs::process_inner` makes (`iterative_detrend:
    // true` is the config default) — real preprocessing pipeline, not a
    // hand-rolled shortcut.
    let quick_search = RealQuickBlsSearch::new(1);
    let processed = preprocess::process(&curves, DetrendMode::Iterative(&quick_search)).expect("preprocess TGLC curves");

    let params = BlsParams { period_count: 50_000, bin_count: 200, duration_step_count: 15, ..BlsParams::default() };

    let cpu = two_pass::search(&processed.time, &processed.flat_flux, &params, 2).expect("CPU two-pass BLS search");
    let backend = gpu();
    let gpu_result =
        backend.two_pass_search(&processed.time, &processed.flat_flux, &params).expect("CUDA two-pass BLS search");

    let period_rel = relative_error(gpu_result.period, cpu.period);
    let snr_rel = relative_error(gpu_result.snr, cpu.snr);
    let depth_ppm_rel = relative_error(gpu_result.depth_ppm, cpu.depth_ppm.max(1.0));
    eprintln!(
        "cuda_two_pass_matches_cpu_two_pass_on_tglc_pipeline: rel errors -- period={period_rel:.3e} \
         snr={snr_rel:.3e} depth_ppm={depth_ppm_rel:.3e} (cpu period={}, snr={}, depth_ppm={})",
        cpu.period, cpu.snr, cpu.depth_ppm
    );

    // Also from test_bls_cuda_parity.py::TestTwoPassParity, verbatim.
    assert!(period_rel < 1e-4, "period mismatch: cpu={} gpu={} (rel {period_rel:.3e})", cpu.period, gpu_result.period);
    assert!(snr_rel < 5e-3, "SNR drift: cpu={} gpu={} (rel {snr_rel:.3e})", cpu.snr, gpu_result.snr);
    assert!(
        depth_ppm_rel < 5e-3,
        "depth drift: cpu={}ppm gpu={}ppm (rel {depth_ppm_rel:.3e})",
        cpu.depth_ppm,
        gpu_result.depth_ppm
    );
}

/// `bls::accelerated::search(.., ComputeBackend::Cuda(..), ..)` must equal
/// `CudaBlsBackend::two_pass_search` directly — the same "backend
/// dispatch is transparent" contract `bls_golden.rs`'s
/// `accelerated_cpu_backend_matches_two_pass` asserts for the CPU arm.
#[test]
#[ignore = "needs NVIDIA GPU"]
fn accelerated_cuda_backend_matches_cuda_two_pass_directly() {
    use exonit_core::bls::accelerated::{self, ComputeBackend};
    use std::sync::Arc;

    let (time, flux) = model_light_curve();
    let params = BlsParams { period_count: 10_000, bin_count: 120, duration_step_count: 10, ..BlsParams::default() };

    let backend = Arc::new(gpu());
    let direct = backend.two_pass_search(&time, &flux, &params).expect("direct CUDA two-pass search");
    let via_accelerated = accelerated::search(&time, &flux, &params, ComputeBackend::Cuda(backend), 2)
        .expect("accelerated CUDA search");

    assert_eq!(direct, via_accelerated, "accelerated Cuda backend must match CudaBlsBackend::two_pass_search exactly");
}

/// Sanity gate for `resolve_compute_backend` (`task_processor.rs`,
/// exercised indirectly via the whole pipeline elsewhere) — direct probe
/// must succeed and report a device, i.e. `PrimaryTaskProcessor::new(...,
/// gpu_enabled: true)` on this host would set `gpu_used() == true`. Not a
/// numerical parity test; a fast "is the GPU actually reachable at all"
/// smoke test to run first when diagnosing a probe failure in the tests
/// above.
#[test]
#[ignore = "needs NVIDIA GPU"]
fn cuda_probe_succeeds_on_a_real_gpu_host() {
    let _backend = gpu();
}
