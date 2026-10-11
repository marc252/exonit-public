//! F7 determinism gate: rayon-parallelized CPU BLS (`bls::optimized`,
//! `bls::two_pass`, and the quick-BLS coarse search inside
//! `preprocess::process`'s iterative detrend, `bls::quick::RealQuickBlsSearch`)
//! must produce EXACTLY the same [`BlsResult`]/[`MultiPlanetSearchResult`]
//! for `max_concurrency = 1` as for `max_concurrency > 1`, on every field
//! (including the full `periods`/`power_spectrum` vectors, via each type's
//! derived `PartialEq`) — not "within tolerance", bit-for-bit `==`.
//!
//! This is the sibling test to `bls_golden.rs` (which pins single-threaded
//! output against the Swift/Python-derived fixture); this file pins
//! multi-threaded output against the *same process's* single-threaded
//! output, on both a synthetic light curve and a real TGLC fixture.

use exonit_core::bls::accelerated::ComputeBackend;
use exonit_core::bls::quick::RealQuickBlsSearch;
use exonit_core::bls::{optimized, reference, two_pass, BlsParams};
use exonit_core::fits;
use exonit_core::harmonic_resolver::LightCurveSource;
use exonit_core::multi_planet::{self, MultiPlanetOptions};
use exonit_core::preprocess::{self, DetrendMode};
use std::path::PathBuf;

/// Same deterministic transit-model light curve as `bls_golden.rs`'s
/// `model_light_curve` (kept as an independent copy — cross-file private
/// helpers aren't shared between integration test binaries in Rust).
fn model_light_curve() -> (Vec<f64>, Vec<f64>) {
    let period = 10.0 / 3.0;
    let time: Vec<f64> = (0..360).map(|i| i as f64 / 12.0).collect();
    let flux: Vec<f64> = time
        .iter()
        .map(|&value| {
            let phase = (value / period) % 1.0;
            if (0.20..0.28).contains(&phase) {
                0.988
            } else {
                1.0
            }
        })
        .collect();
    (time, flux)
}

fn production_like_params() -> BlsParams {
    BlsParams {
        period_minimum: 0.5,
        period_maximum: 12.0,
        period_count: 4_000,
        bin_count: 100,
        duration_step_count: 9,
        duration_minimum_fraction: 0.01,
        duration_maximum_fraction: 0.15,
    }
}

/// `bls::optimized::search`, synthetic curve: 1 vs 8 threads.
#[test]
fn optimized_search_1_thread_equals_8_threads_synthetic() {
    let (time, flux) = model_light_curve();
    let params = production_like_params();

    let single = optimized::search(&time, &flux, &params, 1).expect("1-thread optimized search");
    let multi = optimized::search(&time, &flux, &params, 8).expect("8-thread optimized search");

    assert_eq!(single, multi, "optimized::search must be bit-identical across thread counts");
}

#[test]
fn optimized_batch_reuses_pool_without_changing_results() {
    let (time, flux) = model_light_curve();
    let fluxes = vec![flux.clone(), flux.iter().map(|value| value + 1e-5).collect()];
    let params = production_like_params();
    let expected: Vec<_> = fluxes
        .iter()
        .map(|curve| optimized::search(&time, curve, &params, 3).expect("individual search"))
        .collect();
    let actual = optimized::search_batch(&time, &fluxes, &params, 3).expect("batch search");
    assert_eq!(actual, expected);
}

/// `bls::optimized::search` with `max_concurrency = 0` ("no explicit cap",
/// runs on rayon's default/global pool) must ALSO match the 1-thread and
/// N-thread results — the "0/absent -> rayon default" branch documented in
/// `bls::with_pool` is still just a different thread count, not a
/// different algorithm.
#[test]
fn optimized_search_default_pool_equals_explicit_thread_counts() {
    let (time, flux) = model_light_curve();
    let params = production_like_params();

    let single = optimized::search(&time, &flux, &params, 1).expect("1-thread optimized search");
    let default_pool = optimized::search(&time, &flux, &params, 0).expect("default-pool optimized search");

    assert_eq!(single, default_pool, "max_concurrency=0 must match max_concurrency=1 bit-for-bit");
}

/// `bls::two_pass::search` (both coarse and fine passes share one thread
/// pool per call, see that module's doc comment): 1 vs 6 threads.
#[test]
fn two_pass_search_1_thread_equals_6_threads_synthetic() {
    let (time, flux) = model_light_curve();
    let params = production_like_params();

    let single = two_pass::search(&time, &flux, &params, 1).expect("1-thread two-pass search");
    let multi = two_pass::search(&time, &flux, &params, 6).expect("6-thread two-pass search");

    assert_eq!(single, multi, "two_pass::search must be bit-identical across thread counts");
}

/// `bls::accelerated::search(.., ComputeBackend::Cpu, ..)` — the exact call
/// path `multi_planet::search` uses in production — 1 vs 4 threads.
#[test]
fn accelerated_cpu_search_1_thread_equals_4_threads_synthetic() {
    use exonit_core::bls::accelerated;

    let (time, flux) = model_light_curve();
    let params = production_like_params();

    let single =
        accelerated::search(&time, &flux, &params, ComputeBackend::Cpu, 1).expect("1-thread accelerated CPU search");
    let multi =
        accelerated::search(&time, &flux, &params, ComputeBackend::Cpu, 4).expect("4-thread accelerated CPU search");

    assert_eq!(single, multi, "accelerated::search(Cpu) must be bit-identical across thread counts");
}

/// Every parallel BLS entry point must still match the single-threaded
/// `reference::search` ground truth exactly the way `bls_golden.rs` already
/// checks for `optimized::search(..., 4)` — this test adds the same check
/// at higher thread counts to make sure the parity isn't an artifact of a
/// specific thread count.
#[test]
fn optimized_search_matches_reference_at_various_thread_counts() {
    let (time, flux) = model_light_curve();
    let params = production_like_params();

    let truth = reference::search(&time, &flux, &params).expect("reference search");
    for threads in [1usize, 2, 3, 5, 8, 16] {
        let result = optimized::search(&time, &flux, &params, threads).expect("optimized search");
        assert_eq!(
            result.period, truth.period,
            "period parity vs reference at max_concurrency={threads}"
        );
        assert_eq!(result.power, truth.power, "power parity vs reference at max_concurrency={threads}");
        assert_eq!(
            result.duration, truth.duration,
            "duration parity vs reference at max_concurrency={threads}"
        );
        assert_eq!(result.t0, truth.t0, "t0 parity vs reference at max_concurrency={threads}");
        assert_eq!(result.depth, truth.depth, "depth parity vs reference at max_concurrency={threads}");
        assert_eq!(result.snr, truth.snr, "snr parity vs reference at max_concurrency={threads}");
    }
}

fn data_dir() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data"))
}

/// Real-fixture case (TGLC TIC 10756829, the same two-sector fixture
/// `multi_planet_golden.rs` uses): quick-BLS (via `RealQuickBlsSearch`,
/// inside the iterative detrend) AND the primary multi-planet BLS search
/// must both be bit-identical between 1 and 4 threads, end to end —
/// including the transit mask/refined-detrend decisions that quick-BLS's
/// SNR feeds into, which is exactly why this test exercises the full
/// `preprocess::process` -> `multi_planet::search` pipeline rather than a
/// single BLS call in isolation.
#[test]
fn full_pipeline_1_thread_equals_4_threads_real_tglc_fixture() {
    let paths = [
        data_dir().join("hlsp_tglc_tess_ffi_gaiaid-2080664784495597312-s0014-cam1-ccd1_tess_v1_llc.fits"),
        data_dir().join("hlsp_tglc_tess_ffi_gaiaid-2080664784495597312-s0015-cam1-ccd2_tess_v1_llc.fits"),
    ];
    for path in &paths {
        assert!(path.exists(), "expected local TGLC fixture at {}", path.display());
    }
    let curves: Vec<_> = paths.iter().map(|p| fits::read_light_curve(p).expect("read TGLC FITS")).collect();
    let options = MultiPlanetOptions::default();

    let mut results = Vec::new();
    for max_concurrency in [1usize, 4] {
        let quick_search = RealQuickBlsSearch::new(max_concurrency);
        let processed =
            preprocess::process(&curves, DetrendMode::Iterative(&quick_search)).expect("process TGLC pipeline");

        let search = multi_planet::search(
            &processed.time,
            &processed.raw_flux,
            &processed.flat_flux,
            None,
            None,
            LightCurveSource::Ffi,
            &options,
            max_concurrency,
            ComputeBackend::Cpu,
        )
        .expect("multi-planet search");

        results.push((processed, search));
    }

    let (processed_1, search_1) = &results[0];
    let (processed_4, search_4) = &results[1];

    assert_eq!(
        processed_1, processed_4,
        "preprocess::process (incl. quick-BLS-driven transit mask/refine) must be bit-identical across thread counts"
    );
    assert_eq!(
        search_1, search_4,
        "multi_planet::search must be bit-identical across thread counts on the real TGLC fixture"
    );
    assert!(!search_1.candidates.is_empty(), "sanity: pipeline should find at least one candidate on this fixture");
}
