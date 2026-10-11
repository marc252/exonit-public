/*
 * Verbatim copy — DO NOT EDIT the kernel body below without also updating
 * the Python source it is copied from.
 *
 * Origin: ~/exonit/worker/bls_cuda.py, lines 59-91 (the body of
 * the `_PHASE_FOLD_BIN_KERNEL` string, i.e. everything between the opening
 * `"""` on line 58 and the closing `"""` on line 92), as compiled at
 * runtime by `cp.RawKernel(_PHASE_FOLD_BIN_KERNEL, 'phase_fold_bin')` in
 * `bls_gpu()`. This is one of the two kernels of the PRIMARY BLS path
 * (`bls_gpu` / `bls_gpu_two_pass`) — F4a scope per docs/PLAN.md. Included
 * into `bls/cuda.rs` via `include_str!` and NVRTC-compiled at runtime by
 * `cudarc`, the same compile-from-source-string strategy CuPy's
 * `RawKernel` uses, so this is a driver-level port of the identical
 * compilation path, not a reimplementation.
 *
 * Reference copy for future re-sync: worker/bls_cuda.py is production code
 * at ~/exonit (read-only from this worktree) — re-diff against it
 * if that file ever changes this kernel.
 */

extern "C" __global__
void phase_fold_bin(
    const double* time_rel,     // input time array (N points)
    const double* flux,         // input flux array (N points)
    const double* periods,      // trial periods (n_periods)
    double* bin_sums,           // output: (n_periods, n_bins)
    int* bin_counts,            // output: (n_periods, n_bins)
    int N,                      // number of data points
    int n_bins,                 // number of phase bins
    int n_periods               // number of trial periods
) {
    int pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= n_periods) return;

    double period = periods[pid];
    int offset = pid * n_bins;

    // Initialize bins
    for (int b = 0; b < n_bins; b++) {
        bin_sums[offset + b] = 0.0;
        bin_counts[offset + b] = 0;
    }

    // Phase-fold and bin all data points for this period
    for (int i = 0; i < N; i++) {
        double phase = fmod(time_rel[i] / period, 1.0);
        if (phase < 0.0) phase += 1.0;
        int bin_idx = (int)(phase * n_bins);
        if (bin_idx >= n_bins) bin_idx = n_bins - 1;
        bin_sums[offset + bin_idx] += flux[i];
        bin_counts[offset + bin_idx] += 1;
    }
}
