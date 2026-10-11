//! Focused benchmark for the primary two-pass CUDA BLS workspace.
//!
//! Input construction and backend/NVRTC initialization are outside the timed
//! loop. Environment overrides make it possible to exercise both TGLC-sized
//! and multi-sector curves without changing the harness.

use std::hint::black_box;
use std::time::Instant;

use exonit_core::bls::cuda::CudaBlsBackend;
use exonit_core::bls::reference::BlsParams;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|raw| raw.parse().ok()).unwrap_or(default)
}

fn main() {
    let n_points = env_usize("EXONIT_BENCH_POINTS", 1_200).max(6);
    let n_periods = env_usize("EXONIT_BENCH_PERIODS", 50_000).max(10);
    let iterations = env_usize("EXONIT_BENCH_ITERATIONS", 5).max(1);
    let injected_period = 3.371;
    let time: Vec<f64> = (0..n_points).map(|i| i as f64 * 27.0 / (n_points - 1) as f64).collect();
    let flux: Vec<f64> = time
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let phase = (t / injected_period).fract();
            let transit = if !(0.025..=0.975).contains(&phase) { -0.006 } else { 0.0 };
            1.0 + transit + ((i * 37) % 103) as f64 * 1e-7
        })
        .collect();
    let params = BlsParams {
        period_count: n_periods,
        bin_count: 200,
        duration_step_count: 15,
        ..BlsParams::default()
    };
    let backend = CudaBlsBackend::probe().expect("usable CUDA backend");

    let warm = backend.two_pass_search(&time, &flux, &params).expect("warm-up search");
    let expected_period = warm.period;
    let mut samples_ms = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        let result = backend.two_pass_search(&time, &flux, &params).expect("timed search");
        assert_eq!(result.period, expected_period, "workspace changed the selected period");
        black_box(result);
        samples_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    samples_ms.sort_by(f64::total_cmp);
    let median_ms = samples_ms[samples_ms.len() / 2];
    println!(
        "{{\"points\":{n_points},\"periods\":{n_periods},\"iterations\":{iterations},\"selected_period\":{expected_period},\"median_ms\":{median_ms},\"samples_ms\":{samples_ms:?}}}"
    );
}
