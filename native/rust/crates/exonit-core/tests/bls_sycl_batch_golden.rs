//! F-sycl golden tests: SYCL batch injection-recovery BLS
//! (`bls::sycl_batch::SyclBlsBatchBackend`) vs CPU (`bls::optimized::search`,
//! the per-flux loop `injection_recovery.rs` has used since F5a) — the
//! production gate for Spec 2's Spike 1, promoted 2026-08-06 after
//! `spikes/RESULTS_2026-08-06.md`'s real-measurement result (+31% to +37%
//! median vs OpenCL, gate >=15%).
//!
//! Structure/helper calqued from `tests/bls_opencl_batch_golden.rs` (read
//! first, per the same convention that file itself followed from
//! `tests/bls_cuda_batch_golden.rs`) — same synthetic injection-like
//! workload generator (a small deterministic LCG, "some noise" not
//! draw-for-draw NumPy parity), same relative-error helper, same
//! tolerances (period EXACT, depth rel `<5e-3`, snr rel `<5e-2` — SYCL uses
//! the IDENTICAL f32 kernel algorithm as OpenCL, just a different runtime,
//! so there is no reason to expect different tolerances; the 2026-08-06
//! session's real measurements on this exact host already observed
//! `~1e-7` relative depth/snr drift at production scale, i.e. two orders
//! of magnitude inside these tolerances).
//!
//! The whole file is `#![cfg(feature = "sycl")]` (0 tests reported, not a
//! build failure, without the `sycl` Cargo feature). GPU-requiring tests
//! are additionally `#[ignore = "needs a real SYCL GPU device"]`, run
//! explicitly on <host> (Arc) with `cargo test --features sycl --test
//! bls_sycl_batch_golden -- --ignored --nocapture`.

#![cfg(feature = "sycl")]

use exonit_core::bls::optimized;
use exonit_core::bls::reference::BlsParams;
use exonit_core::bls::sycl_batch::SyclBlsBatchBackend;
use exonit_core::injection_recovery::{self, InjectionBackend, InjectionRecoveryOptions};
use std::sync::Arc;

fn gpu() -> SyclBlsBatchBackend {
    SyclBlsBatchBackend::probe().expect(
        "SyclBlsBatchBackend::probe() returned None -- this test needs a real SYCL GPU device \
         (Intel Arc on <host>; see bls::sycl_batch's module doc comment); invoke with \
         `cargo test --features sycl -- --ignored`",
    )
}

fn relative_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(1e-12)
}

/// Same injection-like workload shape as `tests/bls_opencl_batch_golden.rs`'s
/// `injection_like_batch` (duplicated here rather than shared — same
/// precedent that file itself follows relative to
/// `tests/bls_cuda_batch_golden.rs`: Rust integration test binaries do not
/// share code across files without a separate support crate, which none of
/// the three golden-batch test files use).
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

/// Production-shaped: same N=600/n_bins=150 dims
/// `tests/bls_opencl_batch_golden.rs::opencl_batch_matches_cpu_per_flux_loop`
/// uses (that file's own comment: "the defaults `injection.py`'s
/// `bls_gpu_batch` actually runs with on Arc/Vega").
#[test]
#[ignore = "needs a real SYCL GPU device"]
fn sycl_batch_matches_cpu_per_flux_loop() {
    let (time, flux_list) = injection_like_batch(600, 48, 2.0, 10.0, 41);
    let params = BlsParams { period_minimum: 2.0, period_maximum: 10.0, period_count: 2000, bin_count: 150, duration_step_count: 10, ..BlsParams::default() };

    let backend = gpu();
    let gpu_results = backend.search_batch(&time, &flux_list, &params).expect("SYCL batch search");
    assert_eq!(gpu_results.len(), flux_list.len());

    let mut max_depth_rel = 0.0f64;
    let mut max_snr_rel = 0.0f64;
    for (i, flux) in flux_list.iter().enumerate() {
        let cpu = optimized::search(&time, flux, &params, 1).expect("CPU BLS search");
        let gpu_result = &gpu_results[i];
        let period_rel = relative_error(gpu_result.period, cpu.period);
        let depth_rel = relative_error(gpu_result.depth, cpu.depth.abs().max(1e-9));
        let snr_rel = relative_error(gpu_result.snr, cpu.snr);
        max_depth_rel = max_depth_rel.max(depth_rel);
        max_snr_rel = max_snr_rel.max(snr_rel);
        eprintln!(
            "sycl_batch_matches_cpu_per_flux_loop[{i}]: period_rel={period_rel:.3e} depth_rel={depth_rel:.3e} \
             snr_rel={snr_rel:.3e} (cpu period={}, depth={}, snr={}; gpu period={}, depth={}, snr={})",
            cpu.period, cpu.depth, cpu.snr, gpu_result.period, gpu_result.depth, gpu_result.snr
        );
        assert_eq!(
            gpu_result.period, cpu.period,
            "flux #{i}: period mismatch -- the argmax must land on the same grid index as CPU \
             (same fused-kernel design/tie-break as the OpenCL sibling backend, which measured an \
             EXACT period match at this same production shape on this same host); per the plan, \
             this is a STOP-and-investigate-the-tie-break signal, not a tolerance to widen. \
             cpu={} gpu={}",
            cpu.period, gpu_result.period
        );
        assert!(depth_rel < 5e-3, "flux #{i}: depth drift {depth_rel:.3e} (cpu={} gpu={})", cpu.depth, gpu_result.depth);
        assert!(snr_rel < 5e-2, "flux #{i}: SNR drift {snr_rel:.3e} (cpu={} gpu={})", cpu.snr, gpu_result.snr);
    }
    eprintln!("sycl_batch_matches_cpu_per_flux_loop: max_depth_rel={max_depth_rel:.3e} max_snr_rel={max_snr_rel:.3e} across {} flux curves", flux_list.len());
}

/// Sanity check that `search_batch` (the public entry point) produces
/// finite, well-formed results for a slightly larger production-shaped
/// batch.
#[test]
#[ignore = "needs a real SYCL GPU device"]
fn search_batch_returns_finite_results_for_production_defaults() {
    let (time, flux_list) = injection_like_batch(600, 16, 2.0, 10.0, 3);
    let params = BlsParams { period_minimum: 2.0, period_maximum: 10.0, period_count: 2000, bin_count: 150, duration_step_count: 10, ..BlsParams::default() };
    let backend = gpu();
    let results = backend.search_batch(&time, &flux_list, &params).expect("SYCL batch search");
    assert_eq!(results.len(), flux_list.len());
    for result in &results {
        assert!(result.period.is_finite());
        assert!(result.snr.is_finite());
    }
}

/// The decisive end-to-end gate (calqued from
/// `bls_opencl_batch_golden.rs`'s
/// `injection_recovery_completeness_map_bit_identical_cpu_vs_opencl` line
/// for line, same radius/period/injection grid and RNG seed): a full
/// `injection_recovery::run_with_backend` completeness map must be
/// DECISION-IDENTICAL between `InjectionBackend::Cpu` and
/// `InjectionBackend::Sycl` on the same synthetic light curve and grid —
/// f32-vs-f64 tolerances must never flip a recovered/not-recovered
/// decision for any (radius, period) cell. Per the plan: if this ever
/// fails, STOP and investigate; do not widen anything to make it pass.
#[test]
#[ignore = "needs a real SYCL GPU device"]
fn injection_recovery_completeness_map_bit_identical_cpu_vs_sycl() {
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
    let sycl_result = injection_recovery::run_with_backend(&time, &flux, &options, 1, || false, &InjectionBackend::Sycl(backend))
        .expect("SYCL injection-recovery run");

    let cpu_map = cpu_result.completeness_map.expect("cpu completeness map");
    let sycl_map = sycl_result.completeness_map.expect("sycl completeness map");
    assert_eq!(cpu_map.len(), sycl_map.len());
    let mut max_delta = 0.0f64;
    for (cpu_row, sycl_row) in cpu_map.iter().zip(sycl_map.iter()) {
        assert_eq!(cpu_row.len(), sycl_row.len());
        for (&c, &g) in cpu_row.iter().zip(sycl_row.iter()) {
            max_delta = max_delta.max((c - g).abs());
        }
    }
    eprintln!("injection_recovery_completeness_map_bit_identical_cpu_vs_sycl: max|Delta|={max_delta}");
    assert_eq!(max_delta, 0.0, "completeness map max|Delta|={max_delta} -- expected 0.0 (decision-identical)");
}
