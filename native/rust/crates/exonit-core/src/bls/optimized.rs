//! Transliterated from `Sources/ExoNitCore/BLSCPUOptimized.swift`.
//!
//! Swift parallelizes across periods with `DispatchQueue.concurrentPerform`,
//! striding `periodIndex` by `workers` per task. Each period is evaluated
//! completely independently (bin accumulation + duration/phase search reset
//! every iteration, only reading `time`/`flux`/the shared grids), so the
//! *result* cannot depend on how work is partitioned across threads or on
//! execution order — Swift's own contract test (`testBLSGoldenParity`)
//! asserts `BLSCPUOptimized.search(...) == BLSReference.search(...)`
//! bit-for-bit even with `maxConcurrency: 4`.
//!
//! **F7: parallelized with rayon**, per the same independence argument
//! Swift's comment makes (and the Python `bls_fast.py` mirror: `numba.prange`
//! over the identical `pi` period index — see `_bls_all_periods`). The
//! strategy that keeps this bit-for-bit deterministic with any thread count:
//!
//! 1. `periods.par_iter()` maps each period to a [`PeriodOutcome`] computed
//!    entirely from read-only inputs (`relative_time`, `flux`, `p`,
//!    `duration_bins`, `total_sum`) plus a **per-task-local** scratch buffer
//!    (`PeriodScratch`, allocated via `map_init` — never shared or reused
//!    across periods running concurrently on different threads). No shared
//!    mutable accumulator, no cross-period read of another period's partial
//!    state — exactly what the module doc comment above already argues
//!    makes the per-period computation itself thread-count-invariant.
//! 2. `.collect::<Vec<_>>()` on a rayon parallel iterator preserves the
//!    **source order** (this is a documented rayon guarantee — the same
//!    contract `Iterator::collect` gives for the sequential version), so the
//!    resulting `Vec<PeriodOutcome>` is indexed by `period_index` exactly as
//!    the old sequential loop's `powers`/`durations`/`phases`/`depths`
//!    arrays were.
//! 3. The **final reduction is sequential**, scanning `0..outcomes.len()` in
//!    order with strict `>` (keep the first maximum seen) — byte-identical
//!    control flow to the pre-F7 loop below, and to `reference::search`'s.
//!    No floating-point sum is ever combined across threads: the only
//!    cross-thread communication is "here is period `i`'s already-finished,
//!    independently-computed outcome".
//!
//! `max_concurrency` (previously accepted but ignored) now sizes the rayon
//! thread pool this call runs inside — see [`super::with_pool`].

use rayon::prelude::*;

use super::reference::{self, BlsError, BlsParams, BlsResult};
use super::with_pool;

/// Per-thread scratch reused across the periods a single rayon task
/// processes (via `map_init`) — never shared between concurrently-running
/// tasks, so mutating it in place carries none of the determinism risk a
/// *shared* accumulator would.
struct PeriodScratch {
    bin_sums: Vec<f64>,
    bin_counts: Vec<i64>,
    cumulative_sums: Vec<f64>,
    cumulative_counts: Vec<i64>,
}

impl PeriodScratch {
    fn new(bin_count: usize) -> Self {
        PeriodScratch {
            bin_sums: vec![0.0_f64; bin_count],
            bin_counts: vec![0i64; bin_count],
            cumulative_sums: vec![0.0_f64; 2 * bin_count + 1],
            cumulative_counts: vec![0i64; 2 * bin_count + 1],
        }
    }
}

/// `pub(crate)` (was private): 2026-08-06 Spec 2 integration —
/// `bls::ispc_fold` (feature `ispc`) constructs this struct directly from
/// the ISPC kernel's per-chunk output arrays, so both the struct and its
/// fields need at least crate-wide visibility. Still not `pub` — this is
/// an internal representation of `search_in_pool`'s per-period reduction
/// step, not part of this crate's public API.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PeriodOutcome {
    pub(crate) power: f64,
    pub(crate) duration: usize,
    pub(crate) phase: usize,
    pub(crate) depth: f64,
}

/// One period's worth of the old sequential loop body — identical
/// arithmetic and iteration order, just factored out so it can run on
/// whichever thread rayon schedules it to.
fn evaluate_period(
    period: f64,
    relative_time: &[f64],
    flux: &[f64],
    p: &BlsParams,
    duration_bins: &[usize],
    total_sum: f64,
    scratch: &mut PeriodScratch,
) -> PeriodOutcome {
    scratch.bin_sums.iter_mut().for_each(|v| *v = 0.0);
    scratch.bin_counts.iter_mut().for_each(|v| *v = 0);

    for sample_index in 0..relative_time.len() {
        let quotient = relative_time[sample_index] / period;
        let phase = quotient - quotient.floor();
        let raw_bin = (phase * p.bin_count as f64) as i64;
        let bin = raw_bin.clamp(0, p.bin_count as i64 - 1) as usize;
        scratch.bin_sums[bin] += flux[sample_index];
        scratch.bin_counts[bin] += 1;
    }

    scratch.cumulative_sums[0] = 0.0;
    scratch.cumulative_counts[0] = 0;
    for index in 0..(2 * p.bin_count) {
        let source = index % p.bin_count;
        scratch.cumulative_sums[index + 1] = scratch.cumulative_sums[index] + scratch.bin_sums[source];
        scratch.cumulative_counts[index + 1] = scratch.cumulative_counts[index] + scratch.bin_counts[source];
    }

    let mut best_power = -1e30_f64;
    let mut best_duration = 0usize;
    let mut best_phase = 0usize;
    let mut best_depth = 0.0_f64;
    for &width in duration_bins {
        for phase_bin in 0..p.bin_count {
            let window_count = scratch.cumulative_counts[phase_bin + width] - scratch.cumulative_counts[phase_bin];
            let outside_count = relative_time.len() as i64 - window_count;
            if window_count < 3 || outside_count < 3 {
                continue;
            }
            let window_sum = scratch.cumulative_sums[phase_bin + width] - scratch.cumulative_sums[phase_bin];
            let depth = (total_sum - window_sum) / outside_count as f64 - window_sum / window_count as f64;
            if depth <= 0.0 {
                continue;
            }
            let power =
                depth * depth * (window_count * outside_count) as f64 / (window_count + outside_count) as f64;
            if power > best_power {
                best_power = power;
                best_duration = width;
                best_phase = phase_bin;
                best_depth = depth;
            }
        }
    }
    PeriodOutcome { power: best_power, duration: best_duration, phase: best_phase, depth: best_depth }
}

/// 2026-08-06 Spec 2 integration: dispatch point between the ISPC-
/// accelerated batch path (feature `ispc`, `bls::ispc_fold`, promoted from
/// Spike 2 — see that module's doc comment for the full "why batched, not
/// literally one call per period" rationale) and this module's own scalar/
/// rayon-parallelized `evaluate_period` loop (unchanged, still the ONLY
/// path when the `ispc` feature is off, ISPC is unavailable at runtime, or
/// `EXONIT_BLS_ISPC=0`). Both branches return the identical
/// `Vec<PeriodOutcome>` shape `search_in_pool`'s reduction below already
/// expects — the ISPC gate `bls::ispc_fold::ispc_available` checks BEFORE
/// this function ever calls into it, so a disabled/ineligible ISPC path
/// falls straight through to the `#[cfg(feature = "ispc")]` block's tail
/// (or, with the feature off entirely, that block does not exist and this
/// function is just the scalar/rayon loop, unchanged from before this
/// integration).
fn evaluate_all_periods(
    periods: &[f64],
    relative_time: &[f64],
    flux: &[f64],
    p: &BlsParams,
    duration_bins: &[usize],
    total_sum: f64,
) -> Vec<PeriodOutcome> {
    #[cfg(feature = "ispc")]
    if super::ispc_fold::ispc_available(p.bin_count) {
        return super::ispc_fold::search_periods(periods, relative_time, flux, duration_bins, p.bin_count, total_sum);
    }
    // Parallel map (order-preserving on collect, see module doc comment)
    // over independent periods; sequential reduction happens in the caller.
    periods
        .par_iter()
        .map_init(
            || PeriodScratch::new(p.bin_count),
            |scratch, &period| evaluate_period(period, relative_time, flux, p, duration_bins, total_sum, scratch),
        )
        .collect()
}

/// `pub(crate)`: `two_pass::search` calls this directly (once per pass)
/// before `search_in_pool`, since it bypasses the public `search` wrapper
/// to share one thread pool across both passes — see that module's doc
/// comment.
pub(crate) fn validate(time: &[f64], flux: &[f64], p: &BlsParams) -> Result<(), BlsError> {
    if time.len() != flux.len() {
        return Err(BlsError::MismatchedLengths);
    }
    if time.len() < 6 {
        return Err(BlsError::InsufficientSamples);
    }
    if !(p.period_minimum > 0.0)
        || !(p.period_maximum > p.period_minimum)
        || p.period_count < 2
        || p.bin_count < 3
        || p.duration_step_count < 1
        || !(p.duration_minimum_fraction > 0.0)
        || !(p.duration_maximum_fraction >= p.duration_minimum_fraction)
    {
        return Err(BlsError::InvalidParameters("grid".to_string()));
    }
    Ok(())
}

/// Mirrors `BLSCPUOptimized.search(time:flux:parameters:maxConcurrency:)`.
/// Validates, then runs inside a fresh `max_concurrency`-sized rayon pool
/// (see [`super::with_pool`]) before dispatching to [`search_in_pool`].
pub fn search(time: &[f64], flux: &[f64], p: &BlsParams, max_concurrency: usize) -> Result<BlsResult, BlsError> {
    validate(time, flux, p)?;
    with_pool(max_concurrency, || search_in_pool(time, flux, p))
}

/// Searches several flux curves on the same time/grid while constructing at
/// most one Rayon pool. Injection-recovery evaluates 90 curves per period
/// cell in production; routing each one through [`search`] used to rebuild an
/// identical pool 90 times. The curves remain sequential at this level and
/// each period grid uses the shared pool, avoiding nested parallelism and
/// preserving result order and bit-level determinism.
pub fn search_batch(
    time: &[f64],
    fluxes: &[Vec<f64>],
    p: &BlsParams,
    max_concurrency: usize,
) -> Result<Vec<BlsResult>, BlsError> {
    for flux in fluxes {
        validate(time, flux, p)?;
    }
    with_pool(max_concurrency, || fluxes.iter().map(|flux| search_in_pool(time, flux, p)).collect())
}

/// Same as [`search`] but assumes the caller has already established the
/// rayon pool context this call should run in (or is content with the
/// ambient/global pool) — used by `two_pass::search` so a single two-pass
/// call builds exactly ONE thread pool for both the coarse and fine
/// passes, instead of one per `optimized::search` call. `pub(crate)`
/// because it skips `validate`: every caller (currently only
/// `two_pass::search`) must call `validate`-equivalent checks itself, or
/// rely on already-validated parameters — `two_pass::search`'s
/// `coarse_params`/`fine_params` are derived from an already-valid `p`, so
/// this is safe there.
pub(crate) fn search_in_pool(time: &[f64], flux: &[f64], p: &BlsParams) -> Result<BlsResult, BlsError> {
    let t0_reference = time[0];
    let relative_time: Vec<f64> = time.iter().map(|&v| v - t0_reference).collect();
    let periods = reference::frequency_period_grid(p.period_minimum, p.period_maximum, p.period_count);
    let duration_bins = reference::duration_bin_grid(p);
    let total_sum: f64 = flux.iter().sum();

    let outcomes: Vec<PeriodOutcome> = evaluate_all_periods(&periods, &relative_time, flux, p, &duration_bins, total_sum);

    let mut best_index = 0usize;
    for index in 1..outcomes.len() {
        if outcomes[index].power > outcomes[best_index].power {
            best_index = index;
        }
    }
    let powers: Vec<f64> = outcomes.iter().map(|o| o.power).collect();
    let period = periods[best_index];
    let depth = outcomes[best_index].depth;
    Ok(BlsResult {
        period,
        power: outcomes[best_index].power,
        duration: outcomes[best_index].duration as f64 / p.bin_count as f64 * period,
        t0: outcomes[best_index].phase as f64 / p.bin_count as f64 * period + t0_reference,
        depth,
        depth_ppm: depth * 1_000_000.0,
        snr: reference::sde(&powers, best_index),
        periods,
        power_spectrum: powers,
    })
}
