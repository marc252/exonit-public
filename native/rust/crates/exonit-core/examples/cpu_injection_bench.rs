//! Focused benchmark for Rayon-pool reuse in CPU injection BLS.

use std::hint::black_box;
use std::time::Instant;

use exonit_core::bls::optimized;
use exonit_core::bls::reference::BlsParams;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|raw| raw.parse().ok()).unwrap_or(default)
}

fn main() {
    let n_points = env_usize("EXONIT_BENCH_POINTS", 600).max(6);
    let n_flux = env_usize("EXONIT_BENCH_FLUXES", 24).max(1);
    let n_periods = env_usize("EXONIT_BENCH_PERIODS", 2_000).max(2);
    let threads = env_usize("EXONIT_BENCH_THREADS", 3).max(1);
    let iterations = env_usize("EXONIT_BENCH_ITERATIONS", 5).max(1);
    let mode = std::env::var("EXONIT_BENCH_MODE").unwrap_or_else(|_| "batch".to_string());
    let time: Vec<f64> = (0..n_points).map(|i| i as f64 * 27.0 / (n_points - 1) as f64).collect();
    let fluxes: Vec<Vec<f64>> = (0..n_flux)
        .map(|flux_index| {
            let period = 2.3 + 7.4 * flux_index as f64 / n_flux.max(2) as f64;
            time.iter()
                .enumerate()
                .map(|(i, &t)| {
                    let phase = (t / period).fract();
                    let transit = if !(0.03..=0.97).contains(&phase) { -0.004 } else { 0.0 };
                    1.0 + transit + ((i * 17 + flux_index * 31) % 101) as f64 * 1e-7
                })
                .collect()
        })
        .collect();
    let params = BlsParams {
        period_minimum: 2.0,
        period_maximum: 10.0,
        period_count: n_periods,
        bin_count: 150,
        duration_step_count: 10,
        ..BlsParams::default()
    };
    let run = || {
        if mode == "individual" {
            fluxes
                .iter()
                .map(|flux| optimized::search(&time, flux, &params, threads))
                .collect::<Result<Vec<_>, _>>()
        } else {
            optimized::search_batch(&time, &fluxes, &params, threads)
        }
    };
    let expected: Vec<u64> = run().expect("warm-up search").iter().map(|result| result.period.to_bits()).collect();
    let mut samples_ms = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        let results = run().expect("timed search");
        assert!(results.iter().zip(&expected).all(|(result, bits)| result.period.to_bits() == *bits));
        black_box(results);
        samples_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    samples_ms.sort_by(f64::total_cmp);
    let median_ms = samples_ms[samples_ms.len() / 2];
    println!(
        "{{\"mode\":\"{mode}\",\"points\":{n_points},\"fluxes\":{n_flux},\"periods\":{n_periods},\"threads\":{threads},\"iterations\":{iterations},\"median_ms\":{median_ms},\"samples_ms\":{samples_ms:?}}}"
    );
}
