/*
 * Verbatim copy — DO NOT EDIT the kernel body below without also updating
 * the Python source it is copied from.
 *
 * Origin: ~/exonit/worker/bls_cuda.py, lines 706-798 (the body of
 * the `_BLS_BATCH_FUSED_F32_KERNEL_SRC` string, i.e. everything between the
 * opening `r"""` on line 705 and the closing `"""` on line 799), as compiled
 * at runtime by `cp.RawKernel(src, "bls_inj_batch_f32")` in
 * `_get_f32_kernel()`, where `src` is this text with `@NB@`/`@TGSUMS@`
 * template-substituted for a given `n_bins` ceiling (see `_get_f32_kernel`'s
 * `.replace("@NB@", ...).replace("@TGSUMS@", ...)`). This is the DEFAULT
 * batched injection-recovery kernel (spec
 * `docs/superpowers/specs/2026-07-11-gpu-bls-f32-unified-kernel.md` Phase
 * 2 — a CUDA port of the Metal design validated on Apple Silicon,
 * `bls_metal_batch.py`) — F4b scope per docs/PLAN.md. Included into
 * `bls/cuda_batch.rs` via `include_str!`, template-substituted the same way
 * (`.replace`), and NVRTC-compiled at runtime by `cudarc`, the same
 * compile-from-source-string strategy CuPy's `RawKernel` uses, so this is a
 * driver-level port of the identical compilation path, not a
 * reimplementation.
 *
 * `@NB@`/`@TGSUMS@` MUST stay exactly as these two literal tokens (not
 * renamed) — `bls/cuda_batch.rs`'s kernel cache keys and `.replace()` calls
 * depend on the exact byte sequence, matching `_get_f32_kernel`'s own
 * `.replace("@NB@", str(key)).replace("@TGSUMS@", str(TG_FLUX * (key + 1)))`.
 *
 * Reference copy for future re-sync: worker/bls_cuda.py is production code
 * at ~/exonit (read-only from this worktree) — re-diff against it
 * if that file ever changes this kernel.
 */

extern "C" __global__
void bls_inj_batch_f32(
    const short* bin_idx,        // (n_per, N) int16 phase bin per (period, sample)
    const float* counts,         // (n_per, n_bins) exact per-period bin counts
    const float* fluxT,          // (N, n_flux) mean-centered flux, transposed
                                  // for coalesced reads (threads in a warp
                                  // share the period and access consecutive
                                  // flux slots).
    const float* totals,         // (n_flux,) sum of centered flux (~0)
    const int* widths,           // (n_dur,) trial duration widths, in bins
    const int* params,           // [N, n_flux, n_bins, n_per, n_dur]
    float* out_power,            // (n_flux, n_per)
    unsigned int* out_wj,        // (n_flux, n_per) packed (best_w<<16)|best_j
    float* out_depth             // (n_flux, n_per)
) {
    const int f   = blockIdx.x * blockDim.x + threadIdx.x;  // flux slot (padded to TG_FLUX)
    const int p   = blockIdx.y;                              // period index within this chunk
    const int lid = threadIdx.x;

    const int N      = params[0];
    const int n_flux = params[1];
    const int n_bins = params[2];      // runtime, <= @NB@
    const int n_per  = params[3];
    const int n_dur  = params[4];

    __shared__ float tg_cumcounts[@NB@ + 1];
    __shared__ float tg_sums[@TGSUMS@];

    // --- shared per-period cumulative counts (exact: integers in f32) ---
    if (lid == 0) {
        float c = 0.0f;
        tg_cumcounts[0] = 0.0f;
        const float* crow = counts + (long long)p * (long long)n_bins;
        for (int k = 0; k < n_bins; k++) {
            c += crow[k];
            tg_cumcounts[k + 1] = c;
        }
    }
    __syncthreads();

    if (f >= n_flux) return;   // padding threads: after the only barrier

    // --- private per-bin flux sums (sample order == numba order) ---
    float* sums = tg_sums + lid * (@NB@ + 1);
    for (int k = 0; k <= n_bins; k++) sums[k] = 0.0f;

    const short* brow = bin_idx + (long long)p * (long long)N;
    for (int i = 0; i < N; i++) {
        sums[(int)brow[i] + 1] += fluxT[(long long)i * (long long)n_flux + f];
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

    const long long o = (long long)f * (long long)n_per + (long long)p;
    out_power[o] = best_sr;
    out_wj[o]    = ((unsigned int)best_w << 16) | ((unsigned int)best_j & 0xFFFFu);
    out_depth[o] = best_dep;
}
