// PRODUCTION COPY (2026-08-06 integration, feature `ispc`) of Spike 2's
// kernel, promoted UNCHANGED after `spikes/RESULTS_2026-08-06.md`'s real-
// measurement gate: +55% to +235% median vs Rust autovec across both the
// injection and full BLS-scan grids, on both P-cores and E-cores, with
// BIT-EXACT period/power/depth parity on all 180 timed pairs (gate:
// >=10%). Unlike the SYCL kernel, this file is copied byte-for-byte from
// the spike (`spikes/ispc-fold/bls_fold_ispc.ispc`) -- no variant
// exploration was needed here, the margin over the gate was already an
// order of magnitude. Rust host module:
// `crates/exonit-core/src/bls/ispc_fold.rs`. See that module's doc comment
// for how the Rust call site batches periods per rayon chunk (a deliberate
// departure from a literal "one call per period" integration -- documented
// there, not here, since it's a Rust-side decision).
//
// SPMD (ISPC) translation of the per-period BLS fold+bin+sliding-window
// search body: `evaluate_period` in
// crates/exonit-core/src/bls/optimized.rs (itself transliterated from
// `Sources/ExoNitCore/BLSCPUOptimized.swift` -- see that Rust file's module
// doc comment for the full independence argument this SPMD port relies on
// too: every period is evaluated from read-only shared inputs plus
// per-period-local scratch, so the result cannot depend on how work is
// partitioned/ordered across lanes or threads).
//
// Spike 2 of Spec 2 (<host>), docs/superpowers/specs/2026-08-05-native-
// workers-pcmarc-<host>-<host>-design.md: "the folding/binning loop
// of the BLS compiled with ISPC ... vs the auto-vectorised Rust version".
//
// f64 THROUGHOUT -- the primary (non-injection) BLS path is f64-strict per
// CLAUDE.md/DESIGN.md (unlike the fused f32 GPU injection kernel Spike 1
// targets); ISPC's `double` type maps straight to IEEE 754 binary64 on
// every SIMD target this host can run (avx2-i32x8: no f64-narrowing, AVX2
// has native packed-double arithmetic).
//
// # Gang-parallel dimension: PERIOD, not sample or bin
//
// `foreach (p = 0 ... period_count)` assigns one SIMD lane per period --
// mirrors `bls::optimized::search_in_pool`'s `periods.par_iter()` choice of
// parallel axis (see that function's doc comment) exactly, just at SIMD
// granularity instead of rayon-thread granularity. Each lane then runs the
// FULL per-period body (fold -> cumulative sums -> duration/phase search)
// sequentially, on its own private scratch arrays.
//
// This is deliberately NOT tasked/`launch`ed: `ispc_bls_fold_batch` is a
// single `export`ed function with no `launch[]` statement anywhere in its
// body, so calling it does not spawn any ISPC worker threads -- the ENTIRE
// call executes on the calling OS thread, using only this target's native
// SIMD width. The harness (`harness/src/main.rs`) calls the Rust baseline
// through `bls::optimized::search(..., max_concurrency=1)` (a single-thread
// rayon pool) for the same reason: both arms of this spike measure
// per-CORE throughput (SIMD width vs LLVM autovectorization), not
// per-machine throughput -- multi-core scaling is rayon's job on the Rust
// side today and would be ISPC `launch[]`/`tasks` on the ISPC side, neither
// of which this spike touches.
//
// # Why ISPC, not "just trust LLVM autovec": the scatter/gather problem
//
// The inner fold loop's `bin_sums[bin] += flux[i]` has a DATA-DEPENDENT
// (varying) index (`bin` is computed from `time[i]/period`, different for
// every sample) -- classic scatter-accumulate. LLVM's loop autovectorizer
// (what `rustc -O` uses for `bls::optimized::evaluate_period`'s equivalent
// Rust loop) does not vectorize scatter-with-possible-aliasing patterns
// like this without an explicit `#[target_feature]`/intrinsic escape hatch
// -- in practice `evaluate_period`'s fold loop runs SCALAR per period
// (rayon parallelizes ACROSS periods, not within one). ISPC's SPMD model
// is built exactly for this: `bin_sums` declared without `uniform` inside
// the `foreach` becomes a genuinely PER-LANE private array (see below), so
// `bin_sums[bin] += flux[i]` is a normal per-lane scalar store into that
// lane's own memory -- no cross-lane aliasing, no atomics, and the
// vectorization the spec bets on happens automatically at the `foreach`
// level (8 periods processed per gang iteration on `avx2-i32x8`) rather
// than needing to vectorize the scatter itself.
//
// # Fixed-size scratch cap (spike simplification)
//
// `bin_sums`/`bin_counts` are capped at MAX_BINS=256 elements (host arrays
// sized 2*MAX_BINS+1 for the wrap-around cumulative arrays, mirroring
// `PeriodScratch`'s `2 * bin_count + 1` sizing in optimized.rs). Production
// bin_count is 150 (injection-recovery, InjectionRecoveryOptions::default)
// or 200 (scan, BlsParams::default) -- both comfortably under the cap. A
// non-spike port would either generate per-bin_count ISPC variants (mirrors
// bls::opencl_batch's per-n_bins Program cache) or use ISPC's `new[]`
// dynamic allocation; out of scope here, documented in README.md.

#define MAX_BINS 256

export void ispc_bls_fold_batch(
    uniform const double relative_time[],
    uniform const double flux[],
    uniform int32 sample_count,
    uniform const double periods[],
    uniform int32 period_count,
    uniform const int32 duration_bins[],
    uniform int32 duration_count,
    uniform int32 bin_count,
    uniform double total_sum,
    uniform double out_power[],
    uniform int32 out_duration[],
    uniform int32 out_phase[],
    uniform double out_depth[]
) {
    if (bin_count > MAX_BINS) {
        // Mirrors the Rust harness's own pre-flight check (src/main.rs) --
        // both sides refuse rather than silently truncating/corrupting.
        // ISPC has no exceptions; the harness checks bin_count BEFORE
        // calling this function and never reaches here with an oversized
        // bin_count, this is defense in depth only.
        foreach (p = 0 ... period_count) {
            out_power[p] = -1.0;
            out_duration[p] = -1;
            out_phase[p] = -1;
            out_depth[p] = 0.0;
        }
        return;
    }

    foreach (p = 0 ... period_count) {
        const double period = periods[p];

        // Per-lane private scratch: declared WITHOUT `uniform` inside a
        // varying (foreach) context, so ISPC allocates one independent copy
        // of each array PER PROGRAM INSTANCE (SIMD lane) -- exactly
        // `PeriodScratch`'s per-rayon-task buffers in optimized.rs, just at
        // SIMD-lane granularity instead of thread granularity. Never
        // shared/reused across periods running concurrently on different
        // lanes, same "no shared mutable accumulator" determinism argument
        // that module's doc comment makes for rayon.
        double bin_sums[MAX_BINS];
        int32 bin_counts[MAX_BINS];
        double cumulative_sums[2 * MAX_BINS + 1];
        int32 cumulative_counts[2 * MAX_BINS + 1];

        for (uniform int32 k = 0; k < bin_count; k++) {
            bin_sums[k] = 0.0;
            bin_counts[k] = 0;
        }

        // --- fold: scatter-accumulate into this lane's private bins ---
        // `relative_time[i]`/`flux[i]` are UNIFORM-index loads (same `i`
        // for every lane this iteration -- broadcast from memory), `period`
        // is VARYING (one value per lane), so `bin` ends up varying and
        // `bin_sums[bin]`/`bin_counts[bin]` are the scatter this file's
        // header comment names as ISPC's home turf.
        for (uniform int32 i = 0; i < sample_count; i++) {
            double quotient = relative_time[i] / period;
            double phase = quotient - floor(quotient);
            int32 raw_bin = (int32)(phase * (double)bin_count);
            int32 bin = clamp(raw_bin, 0, bin_count - 1);
            bin_sums[bin] += flux[i];
            bin_counts[bin] += 1;
        }

        // --- wrap-around cumulative sums, EXACT same recurrence as
        // PeriodScratch's `cumulative_sums[index+1] = cumulative_sums[index]
        // + bin_sums[index % bin_count]` loop over `0..2*bin_count` ---
        cumulative_sums[0] = 0.0;
        cumulative_counts[0] = 0;
        for (uniform int32 idx = 0; idx < 2 * bin_count; idx++) {
            int32 source = idx % bin_count; // uniform: idx and bin_count are both uniform.
            cumulative_sums[idx + 1] = cumulative_sums[idx] + bin_sums[source];
            cumulative_counts[idx + 1] = cumulative_counts[idx] + bin_counts[source];
        }

        // --- duration/phase sliding-window search, strict '>' tie-break
        // (first maximum wins, matching CPU reference/optimized/Metal/CUDA/
        // OpenCL) ---
        double best_power = -1e30;
        int32 best_duration = 0;
        int32 best_phase = 0;
        double best_depth = 0.0;

        for (uniform int32 di = 0; di < duration_count; di++) {
            uniform int32 width = duration_bins[di];
            for (uniform int32 phase_bin = 0; phase_bin < bin_count; phase_bin++) {
                int32 window_count = cumulative_counts[phase_bin + width] - cumulative_counts[phase_bin];
                int32 outside_count = sample_count - window_count;
                if (window_count < 3 || outside_count < 3) {
                    continue;
                }
                double window_sum = cumulative_sums[phase_bin + width] - cumulative_sums[phase_bin];
                double mean_inside = window_sum / (double)window_count;
                double mean_outside = (total_sum - window_sum) / (double)outside_count;
                double depth = mean_outside - mean_inside;
                if (depth <= 0.0) {
                    continue;
                }
                double power = depth * depth * (double)(window_count * outside_count) / (double)(window_count + outside_count);
                if (power > best_power) {
                    best_power = power;
                    best_duration = width;
                    best_phase = phase_bin;
                    best_depth = depth;
                }
            }
        }

        out_power[p] = best_power;
        out_duration[p] = best_duration;
        out_phase[p] = best_phase;
        out_depth[p] = best_depth;
    }
}
