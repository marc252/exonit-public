//! ISPC-accelerated batch implementation of
//! `bls::optimized::evaluate_period`'s per-period fold + cumulative-bin +
//! duration/phase-search body — the kernel at `../../ispc/bls_fold_ispc.ispc`
//! (compiled by `build.rs`'s `build_ispc()` into a statically-linked
//! `libispc_bls_fold.a`, feature `ispc`). Promoted 2026-08-06 from Spec 2's
//! Spike 2 after `spikes/RESULTS_2026-08-06.md`'s real-measurement gate
//! (>=10% — measured +55% to +235% across the injection/BLS-scan grid
//! shapes and P/E-core clusters, with BIT-EXACT period/power/depth parity
//! on every one of 180 timed pairs). f64 throughout — the primary
//! (non-injection) BLS path is f64-strict, unlike Spike 1/`sycl_batch`'s
//! f32 GPU kernel.
//!
//! # Deviation: batched per rayon CHUNK, not literally one call per period
//!
//! A literal "call the ISPC kernel once per period, from inside
//! `evaluate_period`" integration would call
//! `ispc_bls_fold_batch`'s `foreach (p = 0 ... period_count)` with
//! `period_count = 1` every time — meaning ISPC's `avx2-i32x8` gang runs
//! with only 1 of 8 SIMD lanes ever active (masked off for the other 7),
//! paying full-width vector-instruction cost for scalar-equivalent useful
//! work, PLUS one FFI call per period. This is not a hypothetical concern:
//! the entire measured win in `spikes/RESULTS_2026-08-06.md` came from
//! `foreach` sweeping MANY periods per call (the spike always passed the
//! whole period grid in one call) — nothing in that measurement validates
//! a period-count-1 call pattern, and the SIMD-occupancy argument above
//! predicts it would be roughly break-even with, or slightly worse than,
//! the existing scalar Rust loop it would replace.
//!
//! This module instead partitions the period grid into chunks
//! ([`ISPC_PERIOD_CHUNK`] periods each) and has each rayon task call the
//! ISPC kernel ONCE per chunk it owns — via `par_chunks(..).map(..)`
//! instead of `bls::optimized`'s own `par_iter().map_init(..)` (see that
//! module's `evaluate_all_periods`, the dispatch point this module's
//! `search_periods` is called from). This: (a) uses the EXACT SAME
//! validated kernel byte-for-byte (`../../ispc/bls_fold_ispc.ispc` is an
//! unchanged copy of the spike's file — see that file's header comment),
//! (b) keeps genuine SIMD occupancy per call (chunk size is a multiple of
//! the SIMD width), (c) keeps rayon-level thread parallelism exactly as it
//! was (each chunk is still one independent, order-preserving-on-collect
//! rayon task — same reasoning `bls::optimized`'s module doc comment
//! already gives for why per-period tasks are safe to parallelize any way
//! rayon likes), and (d) still satisfies the task's reentrancy requirement
//! ("each rayon task gets its own scratch... the FFI is reentrant") — see
//! "Reentrancy" below.
//!
//! # Reentrancy
//!
//! `ispc_bls_fold_batch` (the exported ISPC function) has NO global/static
//! mutable state anywhere in `bls_fold_ispc.ispc` — every array it touches
//! is either a caller-supplied pointer (read-only inputs, write-only
//! outputs) or a per-lane PRIVATE scratch array declared inside the
//! `foreach` body (`bin_sums`/`bin_counts`/`cumulative_sums`/
//! `cumulative_counts`, one independent copy per SIMD lane, never shared
//! across lanes OR across calls — see that file's own doc comment,
//! "Per-lane private scratch"). This means concurrent calls from DIFFERENT
//! rayon threads (exactly what `search_periods`'s `par_chunks(..).map(..)`
//! does — multiple chunks' FFI calls can be in flight on different OS
//! threads simultaneously) are safe with NO additional locking: each call
//! only ever reads its own caller-supplied inputs and writes its own
//! caller-supplied outputs, using thread-local ISPC lane scratch that
//! never escapes the single `ispc_bls_fold_batch` invocation that
//! allocated it. Unlike `bls::sycl_batch`/`bls::opencl_batch` (which MUST
//! serialize through `bls::gpu_guard` — shared mutable GPU driver state),
//! this CPU kernel needs no equivalent guard.

use rayon::prelude::*;

use super::optimized::PeriodOutcome;

/// Must match `bls_fold_ispc.ispc`'s `#define MAX_BINS 256` exactly — see
/// that file's "Fixed-size scratch cap" doc comment for why the cap
/// exists. Production `bin_count` (150 injection-recovery / 200 scan) is
/// comfortably under this; [`ispc_available`] refuses (falls back to the
/// scalar path) rather than ever calling the kernel with an oversized
/// `bin_count` — the `.ispc` source's own defense-in-depth branch for this
/// case is therefore never actually reached in practice, by design.
const ISPC_MAX_BINS: usize = 256;

/// Number of periods each rayon chunk's single ISPC FFI call covers, when
/// the ISPC-accelerated path is active — see this module's doc comment,
/// "Deviation", for why chunking (not one call per period) is what makes
/// the measured Spike 2 speedup actually apply here. 64 is 8x the
/// `avx2-i32x8` SIMD width (8 gang lanes/step), giving every chunk at
/// least 8 full-width gang iterations to amortize the FFI call + `foreach`
/// setup cost over — NOT a GPU-memory-bound constant like
/// `bls::opencl_batch::PERIOD_CHUNK_DEFAULT` (this kernel has no device
/// memory to bound), purely a CPU parallelism-granularity choice. Small
/// enough that even the smallest production period grid (injection-
/// recovery's `period_count: 5_000` default) splits into ~78 chunks —
/// plenty for rayon's work-stealing scheduler to load-balance across any
/// realistic `max_concurrency`.
const ISPC_PERIOD_CHUNK: usize = 64;

#[link(name = "ispc_bls_fold")]
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn ispc_bls_fold_batch(
        relative_time: *const f64,
        flux: *const f64,
        sample_count: i32,
        periods: *const f64,
        period_count: i32,
        duration_bins: *const i32,
        duration_count: i32,
        bin_count: i32,
        total_sum: f64,
        out_power: *mut f64,
        out_duration: *mut i32,
        out_phase: *mut i32,
        out_depth: *mut f64,
    );
}

/// Mirrors `bls::opencl_batch::env_disables_opencl`'s contract for the ISPC
/// analogue env var `EXONIT_BLS_ISPC`: `"0"`/`"false"` (case-insensitive,
/// whitespace-tolerant) disables the ISPC CPU-fold path; unset or any
/// other value, including empty string, does not.
fn env_disables_ispc(value: Option<&str>) -> bool {
    matches!(value.map(|v| v.trim().to_ascii_lowercase()).as_deref(), Some("0") | Some("false"))
}

/// Whether `bls::optimized::evaluate_all_periods` should route this call
/// through the ISPC batch path: `bin_count` within the kernel's fixed
/// scratch cap, the env kill-switch not set, and the RUNNING cpu actually
/// supporting `avx2` (checked at runtime via `std::is_x86_feature_detected!`
/// — the compiled `.a` targets `avx2-i32x8` specifically; calling it on a
/// CPU without AVX2 would be an illegal-instruction crash, not a graceful
/// degradation, so this check is a correctness requirement, not an
/// optimization heuristic). Checked once per `search`/`search_in_pool`
/// call (bin_count/env/CPU features are all constant for the DURATION of
/// one call), not once per period — cheap regardless, but this also keeps
/// the eligibility decision consistent across every period in a single
/// search (no possibility of switching kernels mid-search).
pub(crate) fn ispc_available(bin_count: usize) -> bool {
    if bin_count > ISPC_MAX_BINS {
        return false;
    }
    if env_disables_ispc(std::env::var("EXONIT_BLS_ISPC").ok().as_deref()) {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // The `ispc` feature only ever gets ENABLED in production builds
        // for <host> (x86_64) — see Cargo.toml's `ispc` feature comment —
        // but this crate must still COMPILE (not necessarily execute the
        // kernel) on any host, matching every other GPU/SIMD backend's
        // "compiles everywhere, only actually probes/executes where the
        // hardware exists" posture in this crate (`cudarc`'s
        // dynamic-loading being the canonical example).
        false
    }
}

/// Called from `bls::optimized::evaluate_all_periods` once `ispc_available`
/// has already confirmed eligibility. Splits `periods` into
/// [`ISPC_PERIOD_CHUNK`]-sized slices, runs one `ispc_bls_fold_batch` FFI
/// call per chunk in parallel via rayon (order-preserving on `collect`,
/// same guarantee `bls::optimized`'s own module doc comment already leans
/// on), and flattens the per-chunk `Vec<PeriodOutcome>` results back into
/// one `Vec<PeriodOutcome>` indexed by `period_index` exactly like the
/// scalar path's output.
pub(crate) fn search_periods(
    periods: &[f64],
    relative_time: &[f64],
    flux: &[f64],
    duration_bins: &[usize],
    bin_count: usize,
    total_sum: f64,
) -> Vec<PeriodOutcome> {
    let widths_i32: Vec<i32> = duration_bins.iter().map(|&v| v as i32).collect();
    periods
        .par_chunks(ISPC_PERIOD_CHUNK)
        .map(|chunk| fold_chunk(chunk, relative_time, flux, &widths_i32, bin_count, total_sum))
        .collect::<Vec<Vec<PeriodOutcome>>>()
        .into_iter()
        .flatten()
        .collect()
}

/// One rayon task's worth of work: ONE FFI call covering `periods_chunk`
/// (up to [`ISPC_PERIOD_CHUNK`] periods). See this module's doc comment,
/// "Reentrancy", for why this is safe to call concurrently from multiple
/// rayon threads with no additional locking.
fn fold_chunk(
    periods_chunk: &[f64],
    relative_time: &[f64],
    flux: &[f64],
    widths_i32: &[i32],
    bin_count: usize,
    total_sum: f64,
) -> Vec<PeriodOutcome> {
    let n = periods_chunk.len();
    let mut power = vec![0f64; n];
    let mut duration = vec![0i32; n];
    let mut phase = vec![0i32; n];
    let mut depth = vec![0f64; n];

    // SAFETY: `relative_time`/`flux` are sized `sample_count` (passed
    // alongside as `relative_time.len()`); `periods_chunk`/`widths_i32`
    // are sized `period_count`/`duration_count` respectively (passed
    // alongside); `power`/`duration`/`phase`/`depth` are each freshly
    // allocated to exactly `n = periods_chunk.len()` elements, and
    // `ispc_bls_fold_batch`'s `foreach (p = 0 ... period_count)` writes
    // exactly one element to each output array per period in range — no
    // out-of-bounds access is possible given these lengths. No shared
    // mutable state crosses this call boundary (see module doc comment,
    // "Reentrancy") — safe to call from multiple threads concurrently
    // with no lock.
    unsafe {
        ispc_bls_fold_batch(
            relative_time.as_ptr(),
            flux.as_ptr(),
            relative_time.len() as i32,
            periods_chunk.as_ptr(),
            n as i32,
            widths_i32.as_ptr(),
            widths_i32.len() as i32,
            bin_count as i32,
            total_sum,
            power.as_mut_ptr(),
            duration.as_mut_ptr(),
            phase.as_mut_ptr(),
            depth.as_mut_ptr(),
        );
    }

    (0..n)
        .map(|i| PeriodOutcome { power: power[i], duration: duration[i] as usize, phase: phase[i] as usize, depth: depth[i] })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bls_ispc_env_kill_switch_contract() {
        assert!(env_disables_ispc(Some("0")));
        assert!(env_disables_ispc(Some("false")));
        assert!(env_disables_ispc(Some("FALSE")));
        assert!(env_disables_ispc(Some(" False ")));
        assert!(!env_disables_ispc(Some("1")));
        assert!(!env_disables_ispc(Some("")));
        assert!(!env_disables_ispc(None));
    }

    #[test]
    fn ispc_available_rejects_oversized_bin_count() {
        assert!(!ispc_available(ISPC_MAX_BINS + 1));
    }
}
