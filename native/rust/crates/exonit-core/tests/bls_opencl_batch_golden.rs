//! F-opencl golden tests: OpenCL batch injection-recovery BLS
//! (`bls::opencl_batch::OpenClBlsBatchBackend`) vs CPU (`bls::optimized::search`,
//! the per-flux loop `injection_recovery.rs` has used since F5a).
//!
//! Structure/helper calqued from `tests/bls_cuda_batch_golden.rs` (read
//! first, per the plan's Task 3 Step 2 instruction) — same synthetic
//! injection-like workload generator (a small deterministic LCG, "some
//! noise" not draw-for-draw NumPy parity), same relative-error helper.
//! Unlike the CUDA batch backend (which has a separate f64 escape-hatch
//! kernel routed by `search_batch`'s own three-way decision), this module
//! has only ONE kernel design (f32, always) — `bls_opencl.py`'s own
//! docstring documents no on-device f64 path exists for this backend (lines
//! 130-134: "this module has only ONE GPU kernel (f32) and no on-GPU f64
//! path"). So there is only one golden test here, not two.
//!
//! The whole file is `#![cfg(feature = "opencl")]` (0 tests reported, not a
//! build failure, without the `opencl` Cargo feature). GPU-requiring tests
//! are additionally `#[ignore = "needs a real OpenCL GPU device"]`, run
//! explicitly on <host> (Arc) or <host> (Vega, `RUSTICL_ENABLE=radeonsi`)
//! with `cargo test --features opencl --test bls_opencl_batch_golden --
//! --ignored --nocapture`.
//!
//! # Tolerances
//!
//! Per the plan (`docs/plans/2026-08-06-opencl-backend-<host>.md`, Task 3
//! Step 2): period EXACT (rel 0.0 — the period comes off the shared grid, so
//! the argmax must land on the same index as the CPU reference; no f32-vs-f64
//! rounding is allowed to flip which grid point wins), depth rel `< 5e-3`,
//! SNR rel `< 5e-2`. These are documented in `bls_opencl.py:117-122` as the
//! backend's OWN spec tolerances (`rel_power<1e-4`, `rel_depth<1e-3`,
//! `rel_snr<5e-2`); in practice that module's own crossover bench (docstring,
//! lines 112-149) measured `~1e-6`/`~1e-7` and period selection matched
//! EXACTLY ("5/5 -> 90/90 matched") at every tested N up to 100k points — so
//! an exact period match here is the expected outcome, not a hopeful upper
//! bound; per the plan, a mismatch means STOP and investigate the tie-break
//! (strict `>`, first max wins — same rule CPU/Metal/CUDA/OpenCL all share),
//! not widen the tolerance.

#![cfg(feature = "opencl")]

use exonit_core::bls::opencl_batch::OpenClBlsBatchBackend;
use exonit_core::bls::optimized;
use exonit_core::bls::reference::BlsParams;
use exonit_core::injection_recovery::{self, InjectionBackend, InjectionRecoveryOptions};
use std::sync::Arc;

fn gpu() -> OpenClBlsBatchBackend {
    OpenClBlsBatchBackend::probe().expect(
        "OpenClBlsBatchBackend::probe() returned None -- this test needs a real OpenCL GPU \
         device (Intel Arc on <host>, or AMD Vega with RUSTICL_ENABLE=radeonsi on <host>; \
         see bls::opencl_batch's module doc comment); invoke with `cargo test --features opencl \
         -- --ignored`",
    )
}

fn relative_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(1e-12)
}

/// Same injection-like workload shape as `tests/bls_cuda_batch_golden.rs`'s
/// `injection_like_batch` (duplicated here rather than shared, matching that
/// file's own precedent of a self-contained per-test-binary helper — Rust
/// integration test binaries do not share code across files without a
/// separate support crate, which neither golden test file uses).
fn injection_like_batch(n_points: usize, n_flux: usize, period_min: f64, period_max: f64, seed: u64) -> (Vec<f64>, Vec<Vec<f64>>) {
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

/// Production-shaped: N=600, n_bins=150 — the defaults `injection.py`'s
/// `bls_gpu_batch` actually runs with on Arc/Vega (same dimensions
/// `cuda_batch_golden.rs`'s `cuda_batch_f32_matches_cpu_per_flux_loop` uses
/// for its own f32-kernel-vs-CPU gate).
#[test]
#[ignore = "needs a real OpenCL GPU device"]
fn opencl_batch_matches_cpu_per_flux_loop() {
    let (time, flux_list) = injection_like_batch(600, 48, 2.0, 10.0, 41);
    let params = BlsParams { period_minimum: 2.0, period_maximum: 10.0, period_count: 2000, bin_count: 150, duration_step_count: 10, ..BlsParams::default() };

    let backend = gpu();
    let gpu_results = backend.search_batch(&time, &flux_list, &params).expect("OpenCL batch search");
    assert_eq!(gpu_results.len(), flux_list.len());

    for (i, flux) in flux_list.iter().enumerate() {
        let cpu = optimized::search(&time, flux, &params, 1).expect("CPU BLS search");
        let gpu_result = &gpu_results[i];
        let period_rel = relative_error(gpu_result.period, cpu.period);
        let depth_rel = relative_error(gpu_result.depth, cpu.depth.abs().max(1e-9));
        let snr_rel = relative_error(gpu_result.snr, cpu.snr);
        eprintln!(
            "opencl_batch_matches_cpu_per_flux_loop[{i}]: period_rel={period_rel:.3e} depth_rel={depth_rel:.3e} \
             snr_rel={snr_rel:.3e} (cpu period={}, depth={}, snr={}; gpu period={}, depth={}, snr={})",
            cpu.period, cpu.depth, cpu.snr, gpu_result.period, gpu_result.depth, gpu_result.snr
        );
        assert_eq!(
            gpu_result.period, cpu.period,
            "flux #{i}: period mismatch -- the argmax must land on the same grid index as CPU \
             (bls_opencl.py's own crossover bench measured an EXACT period match at every tested \
             N; per the plan, this is a STOP-and-investigate-the-tie-break signal, not a tolerance \
             to widen). cpu={} gpu={}",
            cpu.period, gpu_result.period
        );
        assert!(depth_rel < 5e-3, "flux #{i}: depth drift {depth_rel:.3e} (cpu={} gpu={})", cpu.depth, gpu_result.depth);
        assert!(snr_rel < 5e-2, "flux #{i}: SNR drift {snr_rel:.3e} (cpu={} gpu={})", cpu.snr, gpu_result.snr);
    }
}

/// Sanity check that `search_batch` (the public entry point) produces
/// finite, well-formed results for a slightly larger production-shaped
/// batch, not just the two backend methods `bls::cuda_batch` tests directly
/// (this module has no internal f32/f64 routing to bypass -- `search_batch`
/// IS the only kernel path).
#[test]
#[ignore = "needs a real OpenCL GPU device"]
fn search_batch_returns_finite_results_for_production_defaults() {
    let (time, flux_list) = injection_like_batch(600, 16, 2.0, 10.0, 3);
    let params = BlsParams { period_minimum: 2.0, period_maximum: 10.0, period_count: 2000, bin_count: 150, duration_step_count: 10, ..BlsParams::default() };
    let backend = gpu();
    let results = backend.search_batch(&time, &flux_list, &params).expect("OpenCL batch search");
    assert_eq!(results.len(), flux_list.len());
    for result in &results {
        assert!(result.period.is_finite());
        assert!(result.snr.is_finite());
    }
}

/// F-opencl Task 4 Step 4: the decisive end-to-end gate (calc of
/// `bls_cuda_batch_golden.rs`'s `injection_recovery_completeness_map_bit_identical_cpu_vs_cuda`
/// line for line, same radius/period/injection grid and RNG seed): a full
/// `injection_recovery::run_with_backend` completeness map must be
/// DECISION-IDENTICAL between `InjectionBackend::Cpu` and
/// `InjectionBackend::OpenCl` on the same synthetic light curve and grid —
/// f32-vs-f64 tolerances must never flip a recovered/not-recovered decision
/// for any (radius, period) cell. Per the plan: if this ever fails, STOP and
/// investigate; do not widen anything to make it pass.
#[test]
#[ignore = "needs a real OpenCL GPU device"]
fn injection_recovery_completeness_map_bit_identical_cpu_vs_opencl() {
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
    let opencl_result = injection_recovery::run_with_backend(&time, &flux, &options, 1, || false, &InjectionBackend::OpenCl(backend))
        .expect("OpenCL injection-recovery run");

    let cpu_map = cpu_result.completeness_map.expect("cpu completeness map");
    let opencl_map = opencl_result.completeness_map.expect("opencl completeness map");
    assert_eq!(cpu_map.len(), opencl_map.len());
    let mut max_delta = 0.0f64;
    for (cpu_row, opencl_row) in cpu_map.iter().zip(opencl_map.iter()) {
        assert_eq!(cpu_row.len(), opencl_row.len());
        for (&c, &g) in cpu_row.iter().zip(opencl_row.iter()) {
            max_delta = max_delta.max((c - g).abs());
        }
    }
    eprintln!("injection_recovery_completeness_map_bit_identical_cpu_vs_opencl: max|Delta|={max_delta}");
    assert_eq!(max_delta, 0.0, "completeness map max|Delta|={max_delta} -- expected 0.0 (decision-identical)");
}
