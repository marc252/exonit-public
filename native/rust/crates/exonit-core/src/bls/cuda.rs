//! CUDA driver-API port of the PRIMARY BLS path in
//! `~/exonit/worker/bls_cuda.py` — `bls_gpu` (single-pass) and
//! `bls_gpu_two_pass` (the production entry point `bls.py` wires to the
//! `"cuda"` backend), using the `phase_fold_bin` + `sliding_window` kernels
//! whose exact CUDA C sources live at `cuda_kernels/phase_fold_bin.cu` /
//! `cuda_kernels/sliding_window.cu` (verbatim copies of `bls_cuda.py`
//! lines 59-91 / 95-176 — see each file's header comment for the precise
//! origin).
//!
//! **F4a scope only** (docs/PLAN.md): the batched f32 injection-recovery
//! kernel (`bls_gpu_batch`/`_bls_gpu_batch_f32`/`_bls_gpu_batch_f64`),
//! `wotan_cuda`, and the ONNX CUDA execution provider are F4b — not
//! touched here.
//!
//! # Backend: `cudarc`, dynamic-loading
//!
//! Built on `cudarc` (Cargo feature `cuda`, workspace `Cargo.toml` for the
//! exact feature rationale). The `dynamic-loading` cudarc feature means
//! this module has **no link-time dependency** on `libcuda`/`libnvrtc` —
//! both are `dlopen()`-ed lazily, on first use, via the normal dynamic
//! linker search path (including `LD_LIBRARY_PATH`). That is what makes
//! `cargo build --features cuda` succeed on <host> (no NVIDIA GPU/no CUDA
//! toolkit installed): the crate compiles, the symbols are simply never
//! resolved because nothing calls `probe()`.
//!
//! At runtime (<host>), `libcuda.so.1` is the system NVIDIA driver
//! library (already on the default linker path, no venv involved).
//! `libnvrtc.so.12` ships inside the worker's Python venv wheel
//! (`nvidia-cuda-nvrtc-cu12`), so `LD_LIBRARY_PATH` MUST include its
//! directory for `probe()` to find it — per `ops/workers.json`'s
//! `worker-<host>` entry (read-only reference), that is
//! `~/exonit/.venv/lib/python3.14/site-packages/nvidia/cuda_nvrtc/lib`
//! (the same venv/host already sets this for the Python worker's CuPy
//! stack; the full `LD_LIBRARY_PATH` value from that entry also covers
//! `cuda_runtime`/`cublas`/etc, none of which this module needs, but
//! including the whole thing is harmless and keeps one env var for both
//! workers). See docs/PLAN.md's F4 section for the exact invocation used
//! to run the golden tests on <host>.
//!
//! # Probe-once, no per-call fallback
//!
//! Mirrors `bls.py`'s semantics EXACTLY: `bls.py` probes for a working
//! CUDA backend ONCE, at process start (module import time,
//! `is_cuda_available()`), logs a single warning if the probe fails, and
//! then either always uses CUDA or always uses the CPU fallback for the
//! rest of the process's life — it does NOT re-probe or silently fall back
//! per call. `CudaBlsBackend::probe()` is this module's equivalent single
//! probe point (called once by `task_processor.rs` at
//! `PrimaryTaskProcessor` construction, matching bls.py's import-time
//! timing as closely as a library crate can); `search()`/`two_pass_search()`
//! on an already-`probe()`d backend do NOT fall back to CPU on failure —
//! same as `bls_gpu`/`bls_gpu_two_pass`, a failure during an actual search
//! is a real error (`BlsError::Backend`), not something to swallow.
//!
//! Every failure branch in `probe()` logs via `eprintln!` — this crate's
//! existing convention for "never silent" degradation (see
//! `task_processor.rs`'s `load_cnn`/`load_exominer`) — because a
//! GPU-configured host quietly falling back to CPU is project rule 2's
//! #1 fleet gotcha (mirrors `worker.py`'s `_check_gpu_consistency`).

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use super::reference::{self, BlsError, BlsParams, BlsResult};

/// Verbatim source of `phase_fold_bin` — see
/// `cuda_kernels/phase_fold_bin.cu`'s header comment for the exact origin
/// (`bls_cuda.py` lines 59-91).
const PHASE_FOLD_BIN_SRC: &str = include_str!("cuda_kernels/phase_fold_bin.cu");
/// Verbatim source of `sliding_window` — see
/// `cuda_kernels/sliding_window.cu`'s header comment for the exact origin
/// (`bls_cuda.py` lines 95-176).
const SLIDING_WINDOW_SRC: &str = include_str!("cuda_kernels/sliding_window.cu");

/// Threads per block for both kernel launches — matches `bls_gpu`'s
/// `block_size = 256` (`bls_cuda.py:241`) exactly; the grid is sized
/// `ceil(n_periods / block_size)` one-dimensionally, same as the Python
/// `(grid_size,), (block_size,)` `RawKernel` call.
const BLOCK_SIZE: u32 = 256;

/// An initialized CUDA device + the two compiled/loaded PRIMARY-BLS
/// kernels, built once by [`probe`](Self::probe) and reused for every
/// [`search`](Self::search)/[`two_pass_search`](Self::two_pass_search)
/// call — mirrors `bls_cuda.py`'s module-level `cp.RawKernel` objects
/// (NVRTC-compiled once per process, not once per call).
///
/// `cudarc`'s `CudaStream`/`CudaFunction` are already `Send + Sync`
/// (verified against the `cudarc` 0.19.8 source: `unsafe impl Send/Sync`
/// on both), so this struct needs no extra locking to satisfy
/// `TaskProcessor: Send + Sync` — and the Rust worker's primary pool is
/// single-threaded per process today (`runtime.rs`'s `main_loop`), so
/// there is no concurrent-launch concern in practice either.
pub struct CudaBlsBackend {
    stream: Arc<CudaStream>,
    phase_fold_bin: CudaFunction,
    sliding_window: CudaFunction,
}

// `CudaFunction`/`CudaStream` do not derive `Debug` in `cudarc`; this
// exists only so `ComputeBackend` (which wraps this in an `Arc` and
// derives `Debug`) compiles without pulling in a dependency on cudarc's
// own (nonexistent) `Debug` impls.
impl std::fmt::Debug for CudaBlsBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CudaBlsBackend { .. }")
    }
}

/// `pub(crate)` (not private) so `bls::cuda_batch` (F4b) can reuse the exact
/// same error-wrapping convention instead of duplicating it.
pub(crate) fn driver_err<E: std::fmt::Display>(context: &str) -> impl Fn(E) -> BlsError + '_ {
    move |error| BlsError::Backend(format!("{context}: {error}"))
}

impl CudaBlsBackend {
    /// Probe for a usable CUDA GPU and NVRTC-compile both kernels, ONCE.
    /// Returns `None` (after logging exactly one `eprintln!` naming the
    /// specific failure) if any step fails — driver/NVRTC library missing,
    /// no device, or a compile/load error — so the caller can fall back to
    /// the CPU backend. See the module doc comment's "Probe-once" section.
    pub fn probe() -> Option<Self> {
        // SAFETY: `is_culib_present` only attempts a `dlopen()` of the
        // driver library to check whether it resolves; it does not call
        // into the library.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            eprintln!(
                "bls::cuda: libcuda not found via the dynamic loader (check LD_LIBRARY_PATH) \
                 — falling back to the CPU BLS backend, matching bls.py's is_cuda_available() semantics"
            );
            return None;
        }
        // SAFETY: same as above, for libnvrtc.
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            eprintln!(
                "bls::cuda: libnvrtc not found via the dynamic loader (check LD_LIBRARY_PATH \
                 includes .../nvidia/cuda_nvrtc/lib) — falling back to the CPU BLS backend"
            );
            return None;
        }

        let ctx = match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(error) => {
                eprintln!(
                    "bls::cuda: CUDA present but unusable (CudaContext::new failed: {error}) \
                     — falling back to the CPU BLS backend, matching bls_cuda.py's \
                     is_cuda_available() except-branch (\"CUDA present but unusable for BLS\")"
                );
                return None;
            }
        };

        let phase_fold_ptx = match compile_ptx(PHASE_FOLD_BIN_SRC) {
            Ok(ptx) => ptx,
            Err(error) => {
                eprintln!("bls::cuda: NVRTC failed to compile phase_fold_bin: {error} — falling back to the CPU BLS backend");
                return None;
            }
        };
        let sliding_window_ptx = match compile_ptx(SLIDING_WINDOW_SRC) {
            Ok(ptx) => ptx,
            Err(error) => {
                eprintln!("bls::cuda: NVRTC failed to compile sliding_window: {error} — falling back to the CPU BLS backend");
                return None;
            }
        };

        let phase_fold_module = match ctx.load_module(phase_fold_ptx) {
            Ok(module) => module,
            Err(error) => {
                eprintln!("bls::cuda: failed to load the phase_fold_bin module: {error} — falling back to the CPU BLS backend");
                return None;
            }
        };
        let sliding_window_module = match ctx.load_module(sliding_window_ptx) {
            Ok(module) => module,
            Err(error) => {
                eprintln!("bls::cuda: failed to load the sliding_window module: {error} — falling back to the CPU BLS backend");
                return None;
            }
        };

        let phase_fold_bin = match phase_fold_module.load_function("phase_fold_bin") {
            Ok(function) => function,
            Err(error) => {
                eprintln!("bls::cuda: failed to resolve the phase_fold_bin kernel: {error} — falling back to the CPU BLS backend");
                return None;
            }
        };
        let sliding_window = match sliding_window_module.load_function("sliding_window") {
            Ok(function) => function,
            Err(error) => {
                eprintln!("bls::cuda: failed to resolve the sliding_window kernel: {error} — falling back to the CPU BLS backend");
                return None;
            }
        };

        let stream = ctx.default_stream();
        Some(CudaBlsBackend { stream, phase_fold_bin, sliding_window })
    }

    /// Mirrors `bls_cuda.bls_gpu(time, flux, period_min, period_max,
    /// n_periods, n_bins, n_dur_steps, duration_min_frac,
    /// duration_max_frac)` (`bls_cuda.py:180-290`) exactly: same grid
    /// construction, same kernel launch geometry, same host-side
    /// argmax/duration/phase/SNR reduction.
    ///
    /// Parameter validation mirrors `bls::reference::search`/
    /// `bls::optimized::search` (not present in the Python function itself,
    /// which relies on its caller/numpy to catch bad input) — an added
    /// safety net, not a behavior divergence for any input that was valid
    /// to begin with.
    pub fn search(&self, time: &[f64], flux: &[f64], p: &BlsParams) -> Result<BlsResult, BlsError> {
        if time.len() != flux.len() {
            return Err(BlsError::MismatchedLengths);
        }
        if time.len() < 6 {
            return Err(BlsError::InsufficientSamples);
        }
        if !(p.period_minimum > 0.0) || !(p.period_maximum > p.period_minimum) {
            return Err(BlsError::InvalidParameters("period range".to_string()));
        }
        if p.period_count < 2 || p.bin_count < 3 || p.duration_step_count < 1 {
            return Err(BlsError::InvalidParameters("grid dimensions".to_string()));
        }
        if !(p.duration_minimum_fraction > 0.0) || !(p.duration_maximum_fraction >= p.duration_minimum_fraction) {
            return Err(BlsError::InvalidParameters("duration range".to_string()));
        }

        // "Clamp n_bins for CUDA kernel (static array in kernel)" —
        // bls_cuda.py:203, `n_bins = min(n_bins, 256)`. The
        // sliding_window kernel's `cum_sums[512]`/`cum_counts[512]`
        // per-thread arrays are sized for exactly this ceiling (see that
        // kernel's header comment).
        let n_bins = p.bin_count.min(256);
        let n_periods = p.period_count;
        let sample_count = time.len();

        let t0_reference = time[0];
        let time_relative: Vec<f64> = time.iter().map(|&v| v - t0_reference).collect();
        let periods = reference::frequency_period_grid(p.period_minimum, p.period_maximum, n_periods);
        let total_sum: f64 = flux.iter().sum();

        // Trial durations in bins, built against the CLAMPED n_bins — same
        // order as bls_gpu (clamp first, THEN build dur_bins from the
        // clamped value). `reference::duration_bin_grid` is the same
        // np.unique(np.linspace(...).astype(int32)) construction already
        // validated against bls_fast/Swift in F2's golden tests.
        let clamped_params = BlsParams { bin_count: n_bins, ..*p };
        let duration_bins = reference::duration_bin_grid(&clamped_params);
        let n_dur_steps_actual = duration_bins.len();
        let duration_bins_i32: Vec<i32> = duration_bins.iter().map(|&v| v as i32).collect();

        let grid_size = (n_periods as u32).div_ceil(BLOCK_SIZE);
        let launch_cfg = LaunchConfig { grid_dim: (grid_size, 1, 1), block_dim: (BLOCK_SIZE, 1, 1), shared_mem_bytes: 0 };

        let stream = &self.stream;
        let d_time = stream.clone_htod(&time_relative).map_err(driver_err("upload time"))?;
        let d_flux = stream.clone_htod(flux).map_err(driver_err("upload flux"))?;
        let d_periods = stream.clone_htod(&periods).map_err(driver_err("upload periods"))?;
        let d_dur_bins = stream.clone_htod(&duration_bins_i32).map_err(driver_err("upload duration bins"))?;

        let mut d_bin_sums: CudaSlice<f64> =
            stream.alloc_zeros(n_periods * n_bins).map_err(driver_err("alloc bin_sums"))?;
        let mut d_bin_counts: CudaSlice<i32> =
            stream.alloc_zeros(n_periods * n_bins).map_err(driver_err("alloc bin_counts"))?;
        let mut d_best_powers: CudaSlice<f64> = stream.alloc_zeros(n_periods).map_err(driver_err("alloc best_powers"))?;
        let mut d_best_depths: CudaSlice<f64> = stream.alloc_zeros(n_periods).map_err(driver_err("alloc best_depths"))?;
        let mut d_best_dur_idx: CudaSlice<i32> = stream.alloc_zeros(n_periods).map_err(driver_err("alloc best_dur_idx"))?;
        let mut d_best_phase_idx: CudaSlice<i32> =
            stream.alloc_zeros(n_periods).map_err(driver_err("alloc best_phase_idx"))?;

        let n_i32 = sample_count as i32;
        let n_bins_i32 = n_bins as i32;
        let n_periods_i32 = n_periods as i32;
        let n_dur_i32 = n_dur_steps_actual as i32;

        {
            let mut launch = stream.launch_builder(&self.phase_fold_bin);
            launch.arg(&d_time);
            launch.arg(&d_flux);
            launch.arg(&d_periods);
            launch.arg(&mut d_bin_sums);
            launch.arg(&mut d_bin_counts);
            launch.arg(&n_i32);
            launch.arg(&n_bins_i32);
            launch.arg(&n_periods_i32);
            // SAFETY: kernel argument types/order/count match
            // `phase_fold_bin`'s C signature exactly (verbatim copy, see
            // `cuda_kernels/phase_fold_bin.cu`); all device buffers are
            // sized for `n_periods`/`n_periods * n_bins` as the kernel
            // expects.
            unsafe { launch.launch(launch_cfg) }.map_err(driver_err("launch phase_fold_bin"))?;
        }

        {
            let mut launch = stream.launch_builder(&self.sliding_window);
            launch.arg(&d_bin_sums);
            launch.arg(&d_bin_counts);
            launch.arg(&d_dur_bins);
            launch.arg(&mut d_best_powers);
            launch.arg(&mut d_best_depths);
            launch.arg(&mut d_best_dur_idx);
            launch.arg(&mut d_best_phase_idx);
            launch.arg(&total_sum);
            launch.arg(&n_i32);
            launch.arg(&n_bins_i32);
            launch.arg(&n_periods_i32);
            launch.arg(&n_dur_i32);
            // SAFETY: same as above, matching `sliding_window`'s C
            // signature (verbatim copy, `cuda_kernels/sliding_window.cu`);
            // `n_bins <= 256` is guaranteed by the clamp above, satisfying
            // the kernel's `cum_sums[512]`/`cum_counts[512]` bound.
            unsafe { launch.launch(launch_cfg) }.map_err(driver_err("launch sliding_window"))?;
        }

        let best_powers: Vec<f64> = stream.clone_dtoh(&d_best_powers).map_err(driver_err("download best_powers"))?;
        let best_depths: Vec<f64> = stream.clone_dtoh(&d_best_depths).map_err(driver_err("download best_depths"))?;
        let best_dur_indices: Vec<i32> =
            stream.clone_dtoh(&d_best_dur_idx).map_err(driver_err("download best_dur_idx"))?;
        let best_phase_indices: Vec<i32> =
            stream.clone_dtoh(&d_best_phase_idx).map_err(driver_err("download best_phase_idx"))?;

        // `np.argmax(best_powers)` — first occurrence of the maximum,
        // scanning left to right. Same idiom as `bls::reference::search`'s
        // own best-index reduction.
        let mut best_index = 0usize;
        for index in 1..best_powers.len() {
            if best_powers[index] > best_powers[best_index] {
                best_index = index;
            }
        }

        let best_period = periods[best_index];
        let best_dur_bin = duration_bins[best_dur_indices[best_index] as usize];
        let best_duration = best_dur_bin as f64 / n_bins as f64 * best_period;
        let best_phase = best_phase_indices[best_index] as f64 / n_bins as f64;
        let depth = best_depths[best_index];

        Ok(BlsResult {
            period: best_period,
            power: best_powers[best_index],
            duration: best_duration,
            t0: best_phase * best_period + t0_reference,
            depth,
            depth_ppm: depth * 1_000_000.0,
            // Same robust-SDE formula as the CPU path — bls_cuda.py:277-278
            // explicitly reuses `bls_fast._compute_sde` "so that
            // injection_recovery's snr_threshold gate behaves consistently
            // across CPU and GPU paths"; `reference::sde` is that same
            // formula (already validated against bls_fast in F2's golden
            // tests).
            snr: reference::sde(&best_powers, best_index),
            periods,
            power_spectrum: best_powers,
        })
    }

    /// Mirrors `bls_cuda.bls_gpu_two_pass` (`bls_cuda.py:293-348`) exactly:
    /// coarse full-range pass, early-return on `snr < 3.0`, fine pass
    /// refined around the coarse peak, final result takes
    /// period/duration/t0/depth from the fine pass but SNR/spectrum from
    /// the coarse pass. Structurally identical to `bls::two_pass::search`
    /// (same control flow, ported line-for-line) with `self.search` in
    /// place of `optimized::search` — this is the CUDA analogue of that
    /// CPU two-pass port, not a fresh algorithm.
    pub fn two_pass_search(&self, time: &[f64], flux: &[f64], p: &BlsParams) -> Result<BlsResult, BlsError> {
        let coarse_count = 5_000usize.min(p.period_count);
        let mut coarse_params = *p;
        coarse_params.period_count = coarse_count;
        let coarse = self.search(time, flux, &coarse_params)?;
        if coarse.snr < 3.0 {
            return Ok(coarse);
        }

        let coarse_step = (p.period_maximum - p.period_minimum) / coarse_count as f64;
        let refine_half_width = (coarse_step * 10.0).max(coarse.period * 0.02);
        let mut fine_params = *p;
        fine_params.period_minimum = p.period_minimum.max(coarse.period - refine_half_width);
        fine_params.period_maximum = p.period_maximum.min(coarse.period + refine_half_width);
        fine_params.period_count = (p.period_count / 5).min(10_000);
        if fine_params.period_count < 2 {
            return Ok(coarse);
        }
        let fine = self.search(time, flux, &fine_params)?;
        Ok(BlsResult {
            period: fine.period,
            power: fine.power,
            duration: fine.duration,
            t0: fine.t0,
            depth: fine.depth,
            depth_ppm: fine.depth_ppm,
            snr: coarse.snr,
            periods: coarse.periods,
            power_spectrum: coarse.power_spectrum,
        })
    }
}
