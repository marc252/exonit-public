//! F-ispc determinism gate (2026-08-06 Spec 2 integration): the ISPC-
//! accelerated CPU BLS fold/bin/window-search path
//! (`bls::optimized::search` when `bls::ispc_fold::ispc_available` is
//! `true`) must produce EXACTLY the same [`BlsResult`] as the scalar/
//! autovec path it replaces — not "within tolerance", bit-for-bit `==`,
//! same gate style as `bls_rayon_parity.rs`'s thread-count determinism
//! tests (this file's sibling; read that one first). Runs in the NORMAL
//! suite — no GPU, no `#[ignore]` — per the task instruction ("the integration gate
//! is EXACT equality... in the normal suite (no GPU needed)").
//!
//! `spikes/RESULTS_2026-08-06.md` already established bit-exactness at
//! production-shaped batches with `--reps 1`/N=30 pairs on <host>'s own
//! CPU; this file pins that result as a permanent regression gate through
//! the ACTUAL PRODUCTION CALL PATH (`bls::optimized::search`, not the
//! spike's standalone harness), toggled via the public
//! `EXONIT_BLS_ISPC` kill-switch — the only lever this crate's public API
//! exposes to force either path from an external integration-test crate
//! (`bls::ispc_fold`'s functions are `pub(crate)`, unreachable from here).
//!
//! # Env var safety
//!
//! `EXONIT_BLS_ISPC` is process-wide state; cargo runs every `#[test]` fn
//! in this binary on its own thread by default. `ENV_LOCK` serializes the
//! (currently two) tests in this file that mutate it — TODO for whoever
//! adds a third: acquire the same lock.

#![cfg(feature = "ispc")]

use exonit_core::bls::optimized;
use exonit_core::bls::reference::BlsParams;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Same deterministic transit-model light curve as `bls_golden.rs`/
/// `bls_rayon_parity.rs`'s `model_light_curve` (duplicated per those
/// files' own established precedent — no shared support crate across
/// integration test binaries).
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

/// Same as `bls_rayon_parity.rs`'s `production_like_params` — bin_count
/// 100 is comfortably under `bls::ispc_fold`'s `ISPC_MAX_BINS` (256) cap.
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

/// Runs `optimized::search` once with the ISPC path eligible (env var
/// cleared) and once forced off (`EXONIT_BLS_ISPC=0`), restoring whatever
/// the env var was set to beforehand either way. Skips (does not fail) on
/// hosts without AVX2 — the compiled ISPC kernel specifically targets
/// `avx2-i32x8`; without AVX2, `bls::ispc_fold::ispc_available` already
/// returns `false` internally, so BOTH calls below would silently take the
/// SAME scalar path and this comparison would be a no-op tautology rather
/// than an actual gate — skipping loudly is more honest than asserting
/// something that isn't really being tested. Every host `ispc` is actually
/// BUILT for in this fleet (<host>) has AVX2 (P/E/LP-E cores all do — see
/// `spikes/README.md`'s CPU topology section), so this skip is not
/// expected to fire in CI/dev on that host, only as defense-in-depth for
/// an unexpected build target.
fn compare_ispc_vs_autovec(time: &[f64], flux: &[f64], params: &BlsParams, max_concurrency: usize) -> Option<(exonit_core::bls::BlsResult, exonit_core::bls::BlsResult)> {
    if !std::is_x86_feature_detected!("avx2") {
        eprintln!("bls_ispc_fold_parity: skipped -- host has no AVX2 (bls_fold_ispc.ispc targets avx2-i32x8)");
        return None;
    }

    let _lock = ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let previous = std::env::var("EXONIT_BLS_ISPC").ok();

    // SAFETY: serialized against every other env-var-mutating test in this
    // file by `ENV_LOCK` (held for the whole function body); no other test
    // BINARY shares this process (each `tests/*.rs` file is its own
    // process), and no other code in THIS binary reads/writes
    // `EXONIT_BLS_ISPC` outside this function.
    unsafe { std::env::remove_var("EXONIT_BLS_ISPC") };
    let ispc_result = optimized::search(time, flux, params, max_concurrency).expect("ISPC-path search (EXONIT_BLS_ISPC unset)");

    // SAFETY: see above.
    unsafe { std::env::set_var("EXONIT_BLS_ISPC", "0") };
    let autovec_result = optimized::search(time, flux, params, max_concurrency).expect("autovec-path search (EXONIT_BLS_ISPC=0)");

    // SAFETY: see above -- restores whatever the env var held before this
    // function ran (or clears it, if it was unset), so this test has no
    // observable side effect on the rest of the process.
    unsafe {
        match &previous {
            Some(value) => std::env::set_var("EXONIT_BLS_ISPC", value),
            None => std::env::remove_var("EXONIT_BLS_ISPC"),
        }
    }

    Some((ispc_result, autovec_result))
}

/// `bls::optimized::search`, synthetic curve, `max_concurrency=4` (also
/// exercises the ISPC path's rayon-chunked parallelism, not just a
/// single-thread call — see `bls::ispc_fold`'s module doc comment for why
/// chunked-parallel is the actual integration shape).
#[test]
fn optimized_search_ispc_equals_autovec_synthetic() {
    let (time, flux) = model_light_curve();
    let params = production_like_params();

    let Some((ispc_result, autovec_result)) = compare_ispc_vs_autovec(&time, &flux, &params, 4) else {
        return; // no AVX2 -- see compare_ispc_vs_autovec's doc comment.
    };

    assert_eq!(ispc_result, autovec_result, "ISPC-accelerated bls::optimized::search must be bit-identical to the autovec/scalar path it replaces");
}

/// Same gate at the crate's OWN default `BlsParams` (200 bins, 15 duration
/// steps, 10 000 periods, 0.5-50d — `BlsParams::default()`) rather than
/// the smaller `production_like_params()` shape, and at `max_concurrency=1`
/// (single rayon chunk potentially covering the WHOLE period grid,
/// depending on `ISPC_PERIOD_CHUNK`'s relationship to the thread count —
/// this is a genuinely different code path through
/// `bls::ispc_fold::search_periods`'s `par_chunks` than the 4-thread case
/// above, worth its own bit-exact check).
#[test]
fn optimized_search_ispc_equals_autovec_default_params_single_thread() {
    let (time, flux) = model_light_curve();
    let params = BlsParams::default();

    let Some((ispc_result, autovec_result)) = compare_ispc_vs_autovec(&time, &flux, &params, 1) else {
        return;
    };

    assert_eq!(
        ispc_result, autovec_result,
        "ISPC-accelerated bls::optimized::search must be bit-identical to the autovec/scalar path at BlsParams::default() / max_concurrency=1"
    );
}
