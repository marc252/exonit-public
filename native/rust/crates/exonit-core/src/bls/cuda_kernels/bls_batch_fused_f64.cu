/*
 * Verbatim copy — DO NOT EDIT the kernel body below without also updating
 * the Python source it is copied from.
 *
 * Origin: ~/exonit/worker/bls_cuda.py, lines 366-462 (the body of
 * the `_BLS_BATCH_FUSED_KERNEL` string, i.e. everything between the opening
 * `"""` on line 365 and the closing `"""` on line 463), as compiled at
 * runtime by `cp.RawKernel(_BLS_BATCH_FUSED_KERNEL, 'bls_batch_fused')` in
 * `_bls_gpu_batch_f64()`. This is the f64 escape-hatch batched
 * injection-recovery kernel (env `EXONIT_BLS_CUDA_F64=1`, or the automatic
 * routing fallback in `bls_gpu_batch()` for oversized `n_bins`/`N`) — F4b
 * scope per docs/PLAN.md. Included into `bls/cuda_batch.rs` via
 * `include_str!` and NVRTC-compiled at runtime by `cudarc`, the same
 * compile-from-source-string strategy CuPy's `RawKernel` uses, so this is a
 * driver-level port of the identical compilation path, not a
 * reimplementation.
 *
 * Reference copy for future re-sync: worker/bls_cuda.py is production code
 * at ~/exonit (read-only from this worktree) — re-diff against it
 * if that file ever changes this kernel.
 */

extern "C" __global__
void bls_batch_fused(
    const double* time_rel,         // (N,)
    const double* flux_t,           // (N, n_flux) transposed for coalesced reads:
                                    // threads in a warp share period (same i in
                                    // the inner loop) and access consecutive fi
                                    // values, so flux_t[i*n_flux + fi] reads
                                    // 32 doubles in 2 cache-line transactions
                                    // instead of 32 strided transactions.
    const double* periods,          // (n_periods,)
    const int* dur_bins,            // (n_dur_steps,)
    const double* total_sums,       // (n_flux,)
    double* best_powers,            // (n_flux, n_periods)
    double* best_depths,            // (n_flux, n_periods)
    int* best_dur_idx,              // (n_flux, n_periods)
    int* best_phase_idx,            // (n_flux, n_periods)
    int N, int n_bins, int n_periods,
    int n_dur_steps, int n_flux
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total_tasks = n_periods * n_flux;
    if (tid >= total_tasks) return;

    // Threads grouped by period first so warps share one period — that makes
    // time_rel[i] reads broadcast and flux_t[i*n_flux + fi] coalesced.
    int pid = tid / n_flux;
    int fi  = tid % n_flux;
    double period = periods[pid];

    double bin_sums[256];
    int bin_counts[256];
    for (int b = 0; b < n_bins; b++) { bin_sums[b] = 0.0; bin_counts[b] = 0; }

    // Phase-fold + bin (coalesced flux read pattern with transposed layout)
    for (int i = 0; i < N; i++) {
        double phase = fmod(time_rel[i] / period, 1.0);
        if (phase < 0.0) phase += 1.0;
        int bin = (int)(phase * n_bins);
        if (bin >= n_bins) bin = n_bins - 1;
        bin_sums[bin]   += flux_t[(long long)i * n_flux + fi];
        bin_counts[bin] += 1;
    }

    double total_sum = total_sums[fi];

    // Build wrap-around cumulative arrays (same layout as _SLIDING_WINDOW_KERNEL).
    double cum_sums[512];
    int    cum_counts[512];
    cum_sums[0]   = bin_sums[0];
    cum_counts[0] = bin_counts[0];
    for (int b = 1; b < n_bins; b++) {
        cum_sums[b]   = cum_sums[b-1]   + bin_sums[b];
        cum_counts[b] = cum_counts[b-1] + bin_counts[b];
    }
    for (int b = 0; b < n_bins; b++) {
        cum_sums[n_bins + b]   = cum_sums[n_bins - 1 + b]   + bin_sums[b];
        cum_counts[n_bins + b] = cum_counts[n_bins - 1 + b] + bin_counts[b];
    }

    double best_sr  = -1e30;
    double best_dep = 0.0;
    int    best_d   = 0;
    int    best_p   = 0;

    for (int di = 0; di < n_dur_steps; di++) {
        int width = dur_bins[di];
        for (int p = 0; p < n_bins; p++) {
            double win_sum;
            int    win_count;
            if (p == 0) {
                win_sum   = cum_sums[width - 1];
                win_count = cum_counts[width - 1];
            } else {
                win_sum   = cum_sums[p + width - 1]   - cum_sums[p - 1];
                win_count = cum_counts[p + width - 1] - cum_counts[p - 1];
            }
            int out_count = N - win_count;
            if (win_count < 3 || out_count < 3) continue;
            double mean_in  = win_sum / win_count;
            double mean_out = (total_sum - win_sum) / out_count;
            double depth    = mean_out - mean_in;
            if (depth <= 0) continue;
            double sr = depth * depth * win_count * out_count
                       / (double)(win_count + out_count);
            if (sr > best_sr) {
                best_sr = sr; best_dep = depth;
                best_d  = di; best_p   = p;
            }
        }
    }

    long long out_idx = (long long)fi * n_periods + pid;
    best_powers[out_idx]    = best_sr;
    best_depths[out_idx]    = best_dep;
    best_dur_idx[out_idx]   = best_d;
    best_phase_idx[out_idx] = best_p;
}
