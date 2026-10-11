//! F4b golden tests: CUDA batch injection-recovery BLS
//! (`bls::cuda_batch::CudaBlsBatchBackend`) vs CPU
//! (`bls::optimized::search`, the per-flux loop `injection_recovery.rs` has
//! used since F5a) — both the f64 and f32 kernels directly, plus an
//! injection-recovery-level end-to-end test comparing
//! `InjectionBackend::Cuda` against `InjectionBackend::Cpu`.
//!
//! The whole file is `#![cfg(feature = "cuda")]` (0 tests reported, not a
//! build failure, without the `cuda` Cargo feature). GPU-requiring tests are
//! additionally `#[ignore = "needs NVIDIA GPU"]`, run explicitly on
//! <host> with `cargo test --features cuda --test bls_cuda_batch_golden
//! -- --ignored --nocapture` (see `bls::cuda`'s module doc comment / the
//! F4a golden test file for the exact `LD_LIBRARY_PATH` invocation this
//! also needs).
//!
//! # Tolerances
//!
//! - **f64 batch vs CPU**: period rel `<1e-4`, SNR rel `<5e-3`, depth_ppm
//!   rel `<5e-3` — copied verbatim from
//!   `~/exonit/tests/test_bls_cuda_parity.py::TestTwoPassParity`,
//!   the SAME numbers F4a's `bls_cuda_golden.rs` reuses for its own
//!   bin+cumsum-GPU-vs-different-algorithm-CPU comparison
//!   (`CudaBlsBackend::two_pass_search` vs `bls::two_pass::search`, which
//!   internally calls `bls::optimized::search` — the exact CPU function this
//!   file's f64 test also compares against). Same shape of comparison
//!   (bin+cumsum GPU kernel vs a differently-ordered CPU summation), same
//!   bar.
//! - **f32 batch vs CPU**: period rel `<0.05`, depth rel `<0.05` — copied
//!   verbatim from `test_bls_cuda_parity.py::TestF32KernelGpuParity::
//!   test_e2e_matches_bls_fast_batch` (`bls_gpu_batch` vs `bls_fast_batch`,
//!   the production f32-kernel-vs-CPU gate). The wide tolerance is
//!   documented there as expected: "near-tie grid flips ... inherent
//!   f32<->f64 rounding at near-tied SR values".
//! - **Injection-recovery completeness map, CPU vs CUDA**: bit-identical
//!   (`max|Delta| == 0.0`) — copied verbatim from
//!   `test_bls_cuda_parity.py::TestF32KernelGpuParity::
//!   test_e2e_completeness_map_bit_identical`, "the decisive gate from the
//!   spec's Gates section (non-negotiable per CLAUDE.md #3)". The
//!   period-match-within-5%-and-SNR-over-threshold recovery decision is a
//!   coarse boolean, robust to the two kernels' small floating-point
//!   differences in practice (that is the whole point of the Python gate).

#![cfg(feature = "cuda")]

use exonit_core::bls::cuda_batch::CudaBlsBatchBackend;
use exonit_core::bls::optimized;
use exonit_core::bls::reference::BlsParams;
use exonit_core::injection_recovery::{self, InjectionBackend, InjectionRecoveryOptions};
use std::sync::Arc;

fn gpu() -> CudaBlsBatchBackend {
    CudaBlsBatchBackend::probe().expect(
        "CudaBlsBatchBackend::probe() returned None — this test needs a real NVIDIA GPU with a \
         working driver + NVRTC on LD_LIBRARY_PATH (see bls::cuda_batch's module doc comment); it \
         is meant to run on <host> only, invoked with `cargo test --features cuda -- --ignored`",
    )
}

fn relative_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(1e-12)
}

/// Same injection-like workload shape as
/// `test_bls_cuda_parity.py::_injection_like_batch`: a handful of fluxes
/// over a shared time grid, each with its own injected transit — a
/// synthetic but structurally realistic stand-in for a period-cell's
/// `injected_fluxes` group.
fn injection_like_batch(n_points: usize, n_flux: usize, period_min: f64, period_max: f64, seed: u64) -> (Vec<f64>, Vec<Vec<f64>>) {
    // Small deterministic LCG (no external RNG crate dependency) — only
    // needs to produce a reproducible pseudo-noise pattern, not match
    // NumPy's PRNG bit-for-bit (unlike `injection::NumpyMt19937`, which
    // exists precisely because injection *recovery* semantics need
    // draw-for-draw NumPy parity; this helper only needs "some noise").
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / (1u64 << 53) as f64
        }
    }
    let mut rng = Lcg(seed ^ 0x9E3779B97F4A7C15);

    let time: Vec<f64> = (0..n_points).map(|i| i as f64 * 30.0 / (n_points - 1) as f64).collect();
    let mut flux_list = Vec::with_capacity(n_flux);
    for i in 0..n_flux {
        let period = period_min + 0.3 + (period_max - period_min - 0.6) * i as f64 / (n_flux.max(2) - 1) as f64;
        let t0 = 0.7 + 0.1 * i as f64;
        let depth = 4e-3;
        let duration = 0.15 * period;
        let half_dur_fraction = (duration / period) / 2.0;
        let flux: Vec<f64> = time
            .iter()
            .map(|&t| {
                // Box-Muller-ish cheap Gaussian stand-in via CLT (sum of
                // uniforms) — good enough for "noise", not draw-for-draw.
                let noise = (0..4).map(|_| rng.next_f64()).sum::<f64>() / 4.0 - 0.5;
                let mut value = 1.0 + noise * 3e-4;
                let mut phase = ((t - t0) % period) / period;
                if phase < 0.0 {
                    phase += 1.0;
                }
                if phase < half_dur_fraction || phase > 1.0 - half_dur_fraction {
                    value -= depth;
                }
                value
            })
            .collect();
        flux_list.push(flux);
    }
    (time, flux_list)
}

#[test]
#[ignore = "needs NVIDIA GPU"]
fn cuda_batch_f64_matches_cpu_per_flux_loop() {
    let (time, flux_list) = injection_like_batch(150, 8, 2.0, 10.0, 7);
    let params = BlsParams { period_minimum: 2.0, period_maximum: 10.0, period_count: 200, bin_count: 40, duration_step_count: 5, ..BlsParams::default() };

    let backend = gpu();
    let gpu_results = backend.search_batch_f64(&time, &flux_list, &params).expect("CUDA f64 batch search");
    assert_eq!(gpu_results.len(), flux_list.len());

    for (i, flux) in flux_list.iter().enumerate() {
        let cpu = optimized::search(&time, flux, &params, 1).expect("CPU BLS search");
        let gpu_result = &gpu_results[i];
        let period_rel = relative_error(gpu_result.period, cpu.period);
        let snr_rel = relative_error(gpu_result.snr, cpu.snr);
        let depth_ppm_rel = relative_error(gpu_result.depth_ppm, cpu.depth_ppm.max(1.0));
        eprintln!(
            "cuda_batch_f64_matches_cpu_per_flux_loop[{i}]: period_rel={period_rel:.3e} snr_rel={snr_rel:.3e} \
             depth_ppm_rel={depth_ppm_rel:.3e} (cpu period={}, snr={}, depth_ppm={}; gpu period={}, snr={}, depth_ppm={})",
            cpu.period, cpu.snr, cpu.depth_ppm, gpu_result.period, gpu_result.snr, gpu_result.depth_ppm
        );
        assert!(period_rel < 1e-4, "flux #{i}: period mismatch cpu={} gpu={} (rel {period_rel:.3e})", cpu.period, gpu_result.period);
        assert!(snr_rel < 5e-3, "flux #{i}: SNR drift cpu={} gpu={} (rel {snr_rel:.3e})", cpu.snr, gpu_result.snr);
        assert!(
            depth_ppm_rel < 5e-3,
            "flux #{i}: depth drift cpu={}ppm gpu={}ppm (rel {depth_ppm_rel:.3e})",
            cpu.depth_ppm,
            gpu_result.depth_ppm
        );
    }
}

#[test]
#[ignore = "needs NVIDIA GPU"]
fn cuda_batch_f32_matches_cpu_per_flux_loop() {
    // Production-shaped: N=600, n_bins=150 (the defaults injection.py's
    // bls_gpu_batch actually runs with), same as
    // TestF32KernelGpuParity's workload dimensions.
    let (time, flux_list) = injection_like_batch(600, 48, 2.0, 10.0, 41);
    let params = BlsParams { period_minimum: 2.0, period_maximum: 10.0, period_count: 2000, bin_count: 150, duration_step_count: 10, ..BlsParams::default() };

    let backend = gpu();
    let gpu_results = backend.search_batch_f32(&time, &flux_list, &params).expect("CUDA f32 batch search");
    assert_eq!(gpu_results.len(), flux_list.len());

    for (i, flux) in flux_list.iter().enumerate() {
        let cpu = optimized::search(&time, flux, &params, 1).expect("CPU BLS search");
        let gpu_result = &gpu_results[i];
        let period_rel = relative_error(gpu_result.period, cpu.period);
        let depth_rel = relative_error(gpu_result.depth, cpu.depth.abs().max(1e-9));
        eprintln!(
            "cuda_batch_f32_matches_cpu_per_flux_loop[{i}]: period_rel={period_rel:.3e} depth_rel={depth_rel:.3e} \
             (cpu period={}, depth={}; gpu period={}, depth={})",
            cpu.period, cpu.depth, gpu_result.period, gpu_result.depth
        );
        assert!(period_rel < 0.05, "flux #{i}: period mismatch cpu={} gpu={} (rel {period_rel:.3e})", cpu.period, gpu_result.period);
        assert!(depth_rel < 0.05, "flux #{i}: depth drift {depth_rel:.3e} (cpu={} gpu={})", cpu.depth, gpu_result.depth);
    }
}

/// `search_batch`'s routing must resolve to the f32 kernel for the
/// production-default `n_bins=150` (well under the shared-memory budget,
/// under the point-count ceiling) — sanity check that the public entry
/// point (not just the two backend methods tested directly above) produces
/// a result at all on real hardware.
#[test]
#[ignore = "needs NVIDIA GPU"]
fn search_batch_routes_and_returns_results_for_production_defaults() {
    let (time, flux_list) = injection_like_batch(600, 16, 2.0, 10.0, 3);
    let params = BlsParams { period_minimum: 2.0, period_maximum: 10.0, period_count: 2000, bin_count: 150, duration_step_count: 10, ..BlsParams::default() };
    let backend = gpu();
    let results = backend.search_batch(&time, &flux_list, &params).expect("CUDA routed batch search");
    assert_eq!(results.len(), flux_list.len());
    for result in &results {
        assert!(result.period.is_finite());
        assert!(result.snr.is_finite());
    }
}

/// The decisive end-to-end gate (mirrors
/// `test_bls_cuda_parity.py::TestF32KernelGpuParity::
/// test_e2e_completeness_map_bit_identical` line for line): a full
/// `injection_recovery::run_with_backend` completeness map must be
/// BIT-IDENTICAL between `InjectionBackend::Cpu` and `InjectionBackend::Cuda`
/// on the same synthetic light curve and grid.
#[test]
#[ignore = "needs NVIDIA GPU"]
fn injection_recovery_completeness_map_bit_identical_cpu_vs_cuda() {
    let n = 600usize;
    let time: Vec<f64> = (0..n).map(|i| i as f64 * 27.0 / (n - 1) as f64).collect();
    let mut rng_state: u64 = 99;
    let mut next = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng_state >> 11) as f64) / (1u64 << 53) as f64
    };
    let flux: Vec<f64> = (0..n)
        .map(|_| {
            let noise = (0..4).map(|_| next()).sum::<f64>() / 4.0 - 0.5;
            1.0 + noise * 4e-4
        })
        .collect();

    let mut options = InjectionRecoveryOptions {
        stellar_radius_solar: 1.0,
        radius_grid_earth: vec![2.0, 4.0, 8.0],
        period_grid_days: Some(vec![2.0, 5.0, 10.0]),
        injections_per_cell: 3,
        minimum_snr: 7.0,
        period_count: 2000,
        bin_count: 150,
        duration_step_count: 10,
        ..InjectionRecoveryOptions::default()
    };
    options.stellar_mass_solar = None;

    let cpu_result = injection_recovery::run_with_backend(&time, &flux, &options, 1, || false, &InjectionBackend::Cpu)
        .expect("CPU injection-recovery run");
    let backend = Arc::new(gpu());
    let cuda_result = injection_recovery::run_with_backend(&time, &flux, &options, 1, || false, &InjectionBackend::Cuda(backend))
        .expect("CUDA injection-recovery run");

    let cpu_map = cpu_result.completeness_map.expect("cpu completeness map");
    let cuda_map = cuda_result.completeness_map.expect("cuda completeness map");
    assert_eq!(cpu_map.len(), cuda_map.len());
    let mut max_delta = 0.0f64;
    for (cpu_row, cuda_row) in cpu_map.iter().zip(cuda_map.iter()) {
        assert_eq!(cpu_row.len(), cuda_row.len());
        for (&c, &g) in cpu_row.iter().zip(cuda_row.iter()) {
            max_delta = max_delta.max((c - g).abs());
        }
    }
    eprintln!("injection_recovery_completeness_map_bit_identical_cpu_vs_cuda: max|Delta|={max_delta}");
    assert_eq!(max_delta, 0.0, "completeness map max|Delta|={max_delta} -- expected 0.0 (bit-identical, per the Python spec's Gates section)");
}
