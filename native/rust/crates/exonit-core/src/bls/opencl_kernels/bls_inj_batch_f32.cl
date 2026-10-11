/*
 * Verbatim copy — DO NOT EDIT the kernel body below without also updating
 * the Python source it is copied from.
 *
 * Origin: ~/exonit/worker/bls_opencl.py, lines 250-343 (the body of
 * the `_BLS_BATCH_FUSED_F32_KERNEL_SRC` string, i.e. everything between the
 * opening `r"""` on line 250 and the closing `"""` on line 343), as compiled
 * at runtime by `cl.Program(_ctx, src).build()` in `_get_kernel()`, where
 * `src` is this text with `@NB@`/`@TGSUMS@` template-substituted for a given
 * `n_bins` ceiling (see `_get_kernel`'s `.replace("@NB@", ...).replace(
 * "@TGSUMS@", ...)`). This is the SAME fused f32 batched injection-recovery
 * kernel design already validated on Metal (`bls_metal_batch.py`) and CUDA
 * (`bls_cuda.py`'s `_BLS_BATCH_FUSED_F32_KERNEL_SRC`,
 * `cuda_kernels/bls_batch_fused_f32.cu` in this crate) -- see
 * `bls_opencl.py`'s module doc comment for the full "why fused" motivation.
 *
 * Included into `bls/opencl_batch.rs` via `include_str!`, template-
 * substituted the same way (`.replace`), and built at runtime by `ocl`'s
 * `Program::builder().src(..).build(&context)`, the same compile-from-
 * source-string strategy PyOpenCL's `cl.Program(...).build()` uses, so this
 * is a driver-level port of the identical compilation path, not a
 * reimplementation.
 *
 * `@NB@`/`@TGSUMS@` MUST stay exactly as these two literal tokens (not
 * renamed) -- `bls/opencl_batch.rs`'s `kernel_source()` depends on the exact
 * byte sequence, matching `_get_kernel`'s own
 * `.replace("@NB@", str(key)).replace("@TGSUMS@", str(TG_FLUX * (key + 1)))`.
 *
 * Reference copy for future re-sync: worker/bls_opencl.py is production code
 * at ~/exonit (read-only from this worktree) -- re-diff against it
 * if that file ever changes this kernel.
 */

__kernel void bls_inj_batch_f32(
    __global const short* bin_idx,     // (n_per, N) int16 phase bin per (period, sample)
    __global const float* counts,      // (n_per, n_bins) exact per-period bin counts
    __global const float* fluxT,       // (N, n_flux) mean-centered flux, transposed
                                        // for coalesced reads (work-items in a
                                        // sub-group share the period and access
                                        // consecutive flux slots).
    __global const float* totals,      // (n_flux,) sum of centered flux (~0)
    __global const int* widths,        // (n_dur,) trial duration widths, in bins
    __global const int* params,        // [N, n_flux, n_bins, n_per, n_dur]
    __global float* out_power,         // (n_flux, n_per)
    __global uint* out_wj,             // (n_flux, n_per) packed (best_w<<16)|best_j
    __global float* out_depth          // (n_flux, n_per)
) {
    const int f   = get_global_id(0);   // flux slot (padded to TG_FLUX)
    const int p   = get_global_id(1);   // period index within this chunk
    const int lid = get_local_id(0);

    const int N      = params[0];
    const int n_flux = params[1];
    const int n_bins = params[2];      // runtime, <= @NB@
    const int n_per  = params[3];
    const int n_dur  = params[4];

    __local float tg_cumcounts[@NB@ + 1];
    __local float tg_sums[@TGSUMS@];

    // --- shared per-period cumulative counts (exact: integers in f32) ---
    if (lid == 0) {
        float c = 0.0f;
        tg_cumcounts[0] = 0.0f;
        __global const float* crow = counts + (long)p * (long)n_bins;
        for (int k = 0; k < n_bins; k++) {
            c += crow[k];
            tg_cumcounts[k + 1] = c;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (f >= n_flux) return;   // padding work-items: after the only barrier

    // --- private per-bin flux sums (sample order == numba order) ---
    __local float* sums = tg_sums + lid * (@NB@ + 1);
    for (int k = 0; k <= n_bins; k++) sums[k] = 0.0f;

    __global const short* brow = bin_idx + (long)p * (long)N;
    for (int i = 0; i < N; i++) {
        sums[(int)brow[i] + 1] += fluxT[(long)i * (long)n_flux + f];
    }
    // in-place prefix sum: sums[k] = sum of bins [0, k)
    for (int k = 1; k <= n_bins; k++) sums[k] += sums[k - 1];

    const float total = totals[f];
    const float Nf = (float)N;

    float best_sr  = -1e30f;
    int   best_w   = 0;
    int   best_j   = 0;
    float best_dep = 0.0f;

    for (int di = 0; di < n_dur; di++) {
        const int w = widths[di];
        for (int j = 0; j < n_bins; j++) {
            const int e = j + w;             // window [j, e), wraps past n_bins
            float wc, ws;
            if (e <= n_bins) {
                wc = tg_cumcounts[e] - tg_cumcounts[j];
                ws = sums[e] - sums[j];
            } else {
                const int e2 = e - n_bins;
                wc = (tg_cumcounts[n_bins] - tg_cumcounts[j]) + tg_cumcounts[e2];
                ws = (sums[n_bins] - sums[j]) + sums[e2];
            }
            const float oc = Nf - wc;
            if (wc < 3.0f || oc < 3.0f) continue;
            const float mean_in  = ws / wc;
            const float mean_out = (total - ws) / oc;
            const float depth = mean_out - mean_in;
            if (depth > 0.0f) {
                const float sr = depth * depth * wc * oc / (wc + oc);
                if (sr > best_sr) {          // strict '>': first (w, j) wins ties
                    best_sr = sr; best_w = w; best_j = j; best_dep = depth;
                }
            }
        }
    }

    const long o = (long)f * (long)n_per + (long)p;
    out_power[o] = best_sr;
    out_wj[o]    = ((uint)best_w << 16) | ((uint)best_j & 0xFFFFu);
    out_depth[o] = best_dep;
}
