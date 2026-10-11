/*
 * Verbatim copy — DO NOT EDIT the kernel body below without also updating
 * the Python source it is copied from.
 *
 * Origin: ~/exonit/worker/bls_cuda.py, lines 95-176 (the body of
 * the `_SLIDING_WINDOW_KERNEL` string, i.e. everything between the opening
 * `"""` on line 94 and the closing `"""` on line 177), as compiled at
 * runtime by `cp.RawKernel(_SLIDING_WINDOW_KERNEL, 'sliding_window')` in
 * `bls_gpu()`. This is the second of the two kernels of the PRIMARY BLS
 * path (`bls_gpu` / `bls_gpu_two_pass`) — F4a scope per docs/PLAN.md.
 * Included into `bls/cuda.rs` via `include_str!` and NVRTC-compiled at
 * runtime by `cudarc`, the same compile-from-source-string strategy CuPy's
 * `RawKernel` uses, so this is a driver-level port of the identical
 * compilation path, not a reimplementation.
 *
 * `cum_sums[512]`/`cum_counts[512]` bound `n_bins <= 256` (the wrap-around
 * doubles it) — this is why `bls_gpu()` clamps `n_bins = min(n_bins, 256)`
 * before launching; `bls/cuda.rs` mirrors that same clamp before calling
 * this kernel.
 *
 * Reference copy for future re-sync: worker/bls_cuda.py is production code
 * at ~/exonit (read-only from this worktree) — re-diff against it
 * if that file ever changes this kernel.
 */

extern "C" __global__
void sliding_window(
    const double* bin_sums,     // (n_periods, n_bins)
    const int* bin_counts,      // (n_periods, n_bins)
    const int* dur_bins,        // trial durations in bins
    double* best_powers,        // output: best SR per period
    double* best_depths,        // output: best depth per period
    int* best_dur_idx,          // output: best duration index
    int* best_phase_idx,        // output: best phase bin
    double total_sum,           // sum of all flux
    int N,                      // total data points
    int n_bins,
    int n_periods,
    int n_dur_steps
) {
    int pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= n_periods) return;

    int offset = pid * n_bins;
    double best_sr = -1e30;
    double best_dep = 0.0;
    int best_d = 0;
    int best_p = 0;

    // Build cumulative sums (wrap-around)
    // Use shared memory would be better but this is simpler
    double cum_sums[512];   // max n_bins * 2
    int cum_counts[512];

    // Extended arrays (double length for wrap-around)
    cum_sums[0] = bin_sums[offset];
    cum_counts[0] = bin_counts[offset];
    for (int b = 1; b < n_bins; b++) {
        cum_sums[b] = cum_sums[b-1] + bin_sums[offset + b];
        cum_counts[b] = cum_counts[b-1] + bin_counts[offset + b];
    }
    for (int b = 0; b < n_bins; b++) {
        cum_sums[n_bins + b] = cum_sums[n_bins - 1 + b] + bin_sums[offset + b];
        cum_counts[n_bins + b] = cum_counts[n_bins - 1 + b] + bin_counts[offset + b];
    }

    for (int di = 0; di < n_dur_steps; di++) {
        int width = dur_bins[di];

        for (int p = 0; p < n_bins; p++) {
            int end_idx = p + width;
            double win_sum, win_count_d;
            int win_count;

            if (p == 0) {
                win_sum = cum_sums[width - 1];
                win_count = cum_counts[width - 1];
            } else {
                win_sum = cum_sums[p + width - 1] - cum_sums[p - 1];
                win_count = cum_counts[p + width - 1] - cum_counts[p - 1];
            }

            int out_count = N - win_count;
            if (win_count < 3 || out_count < 3) continue;

            double mean_in = win_sum / win_count;
            double mean_out = (total_sum - win_sum) / out_count;
            double depth = mean_out - mean_in;

            if (depth <= 0) continue;

            double sr = depth * depth * win_count * out_count / (win_count + out_count);

            if (sr > best_sr) {
                best_sr = sr;
                best_dep = depth;
                best_d = di;
                best_p = p;
            }
        }
    }

    best_powers[pid] = best_sr;
    best_depths[pid] = best_dep;
    best_dur_idx[pid] = best_d;
    best_phase_idx[pid] = best_p;
}
