//! Real (non-fixture) implementation of `detrend_iterative::QuickBlsSearch`,
//! wired up in F2. Mirrors `IterativeDetrender.flatten`'s call to
//! `BLSCPUOptimized.search(time:flux:parameters:maxConcurrency:)` with
//! `IterativeDetrender.quickParameters` — see `detrend_iterative.rs`'s
//! module doc comment for why this seam existed since F1.

use super::optimized;
use super::reference::BlsParams;
use crate::detrend_iterative::{
    QuickBlsResult, QuickBlsSearch, QUICK_BIN_COUNT, QUICK_DURATION_MAXIMUM_FRACTION,
    QUICK_DURATION_MINIMUM_FRACTION, QUICK_DURATION_STEP_COUNT, QUICK_PERIOD_COUNT, QUICK_PERIOD_MAXIMUM,
    QUICK_PERIOD_MINIMUM,
};

/// The real coarse-BLS quick search used by `preprocess::process`'s
/// `DetrendMode::Iterative` path.
///
/// F7: `max_concurrency` is now a field (the `QuickBlsSearch` trait itself
/// has no such parameter — it was frozen non-throwing/argument-free in F1,
/// see `detrend_iterative.rs`'s module doc comment — so it cannot be
/// threaded through `search`'s signature without breaking that trait's
/// contract for every implementor). Callers construct this with whatever
/// `max_concurrency` they already resolved for the main BLS search
/// (`PrimaryTaskProcessor::process_inner` passes `self.max_concurrency`,
/// i.e. the same `config.cpu_threads`-derived cap the primary multi-planet
/// search respects), so quick-BLS's own `optimized::search` call below
/// gets the identical thread-pool ceiling — the preprocess/detrend stage
/// no longer runs its coarse BLS pass single-threaded regardless of host
/// core count (BENCHMARKS.md flagged this as a real contributor to
/// `preprocess_detrend`'s 0.71× ratio vs Python).
pub struct RealQuickBlsSearch {
    pub max_concurrency: usize,
}

impl RealQuickBlsSearch {
    pub fn new(max_concurrency: usize) -> Self {
        RealQuickBlsSearch { max_concurrency }
    }
}

impl QuickBlsSearch for RealQuickBlsSearch {
    fn search(&self, time: &[f64], flux: &[f64]) -> QuickBlsResult {
        let params = BlsParams {
            period_minimum: QUICK_PERIOD_MINIMUM,
            period_maximum: QUICK_PERIOD_MAXIMUM,
            period_count: QUICK_PERIOD_COUNT as usize,
            bin_count: QUICK_BIN_COUNT as usize,
            duration_step_count: QUICK_DURATION_STEP_COUNT as usize,
            duration_minimum_fraction: QUICK_DURATION_MINIMUM_FRACTION,
            duration_maximum_fraction: QUICK_DURATION_MAXIMUM_FRACTION,
        };
        match optimized::search(time, flux, &params, self.max_concurrency) {
            Ok(result) => QuickBlsResult {
                period: result.period,
                power: result.power,
                duration: result.duration,
                t0: result.t0,
                depth: result.depth,
                snr: result.snr,
            },
            // `QuickBlsSearch::search` is infallible by signature (see
            // detrend_iterative.rs — F1 froze this trait as non-throwing,
            // unlike Swift where `IterativeDetrender.flatten` is `throws`
            // and a `BLSError` from this step propagates out of the whole
            // detrend call). The only realistic BLS error here is
            // `insufficientSamples` (<6 finite points after the first-pass
            // detrend) — QUICK_* are fixed valid constants, so
            // `invalidParameters`/`mismatchedLengths` cannot occur. Rather
            // than panic inside a worker's science path on a degenerate
            // (near-empty) light curve, report SNR 0: `flatten`'s
            // `snr_threshold` gate (default 7.0) then takes the same
            // "not refined" branch it would for any other low-significance
            // quick search, which is the closest safe analogue available
            // under this trait's signature. This is a deliberate,
              // documented deviation from Swift's hard failure on this edge
            // case, not a numeric-parity compromise.

            Err(_) => QuickBlsResult {
                period: 0.0,
                power: 0.0,
                duration: 0.0,
                t0: 0.0,
                depth: 0.0,
                snr: 0.0,
            },
        }
    }
}
