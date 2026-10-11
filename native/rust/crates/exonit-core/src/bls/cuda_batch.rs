//! CUDA driver-API port of the BATCHED injection-recovery BLS path in
//! `~/exonit/worker/bls_cuda.py` — `bls_gpu_batch` and its two
//! backends `_bls_gpu_batch_f32` (default) / `_bls_gpu_batch_f64` (escape
//! hatch / oversized-input fallback), using the
//! `bls_batch_fused`/`bls_inj_batch_f32` kernels whose exact CUDA C sources
//! live at `cuda_kernels/bls_batch_fused_f64.cu` /
//! `cuda_kernels/bls_batch_fused_f32.cu` (verbatim copies — see each file's
//! header comment for the precise origin).
//!
//! **F4b scope** (docs/PLAN.md): this is the injection-recovery batch path
//! only. The PRIMARY BLS path (`bls_gpu`/`bls_gpu_two_pass`, single flux per
//! call) is `super::cuda` (F4a) — a separate module, separate kernels,
//! separate `CudaContext` handle (see below for why sharing is safe).
//!
//! # Backend: `cudarc`, dynamic-loading — same posture as `super::cuda`
//!
//! Built the same way as `super::cuda`: `cudarc`'s `dynamic-loading`
//! feature means no link-time dependency on `libcuda`/`libnvrtc`, so
//! `cargo build --features cuda` succeeds on <host> (no NVIDIA GPU). See
//! `super::cuda`'s module doc comment for the full explanation (LD_LIBRARY_PATH
//! requirements for `libnvrtc.so.12` at runtime on <host>, etc.) — it
//! applies here unchanged.
//!
//! This module calls `cudarc::driver::CudaContext::new(0)` independently of
//! `super::cuda::CudaBlsBackend::probe()`. That is safe and intentional:
//! `CudaContext::new` uses the CUDA driver's *primary context* API
//! (`cuDevicePrimaryCtxRetain`), which is reference-counted — every caller
//! on the process gets the SAME underlying driver context, just wrapped in
//! its own `Arc<CudaContext>` handle. CuPy (the Python reference) makes the
//! same assumption implicitly: `bls_cuda.py` and this module's Python
//! counterpart never coordinate a context between themselves either, they
//! just use "the current device". Two independent probes here therefore
//! mirror Python's own lack of cross-module GPU-context coordination,
//! rather than introducing a new failure mode.
//!
//! # f32-default, f64-escape-hatch routing
//!
//! `search_batch` is the direct port of `bls_gpu_batch`'s three-way routing
//! decision (checked in the same priority order, see that function's doc
//! comment below): `EXONIT_BLS_CUDA_F64` env override, then the f32 kernel's
//! per-block shared-memory budget (48 KiB, Volta+), then the f32 CPU-fold
//! point-count crossover (`EXONIT_BLS_CUDA_F32_MAX_PTS`, default 20 000 —
//! see `bls_cuda.py`'s crossover-table comment for the empirical basis).
//! Every guard clause and its log wording is transliterated, not
//! reinvented, so a worker operator reading Rust logs sees the same
//! messages they already know from the Python fleet.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use super::cuda::driver_err;
use super::reference::{self, BlsError, BlsParams, BlsResult};

/// Verbatim source of `bls_batch_fused` (the f64 escape-hatch kernel) — see
/// `cuda_kernels/bls_batch_fused_f64.cu`'s header comment for the exact
/// origin (`bls_cuda.py` lines 366-462).
const FUSED_F64_SRC: &str = include_str!("cuda_kernels/bls_batch_fused_f64.cu");
/// Verbatim (template-parameterized) source of `bls_inj_batch_f32` — see
/// `cuda_kernels/bls_batch_fused_f32.cu`'s header comment for the exact
/// origin (`bls_cuda.py` lines 706-798) and the `@NB@`/`@TGSUMS@`
/// substitution contract `get_f32_kernel` relies on.
const F32_KERNEL_SRC: &str = include_str!("cuda_kernels/bls_batch_fused_f32.cu");

/// `bls_metal_batch`/`bls_cuda.py`'s `TG_FLUX` — flux-slots-per-block for
/// the f32 kernel. See `_shared_mem_bytes`'s doc comment in `bls_cuda.py`.
const TG_FLUX: u32 = 32;

/// `bls_cuda.py`'s `_PERIOD_CHUNK_DEFAULT` — periods processed per f32
/// kernel launch, bounding peak `bin_idx`/`counts` host RAM.
const PERIOD_CHUNK_DEFAULT: usize = 1000;

/// `bls_cuda.py`'s `_CUDA_SHARED_MEM_BUDGET_BYTES` (48 KiB — guaranteed on
/// every compute capability >= 7.0 without a per-arch opt-in).
const CUDA_SHARED_MEM_BUDGET_BYTES: usize = 48 * 1024;

/// `bls_cuda.py`'s `_F32_MAX_POINTS_DEFAULT`.
const F32_MAX_POINTS_DEFAULT: usize = 20_000;

/// Mirrors `bls_cuda.py`'s `_shared_mem_bytes(n_bins)` exactly: `tg_cumcounts`
/// (`n_bins+1` floats, shared) + `tg_sums` (`TG_FLUX*(n_bins+1)` floats).
fn shared_mem_bytes(n_bins: usize) -> usize {
    TG_FLUX as usize * (n_bins + 1) * 4 + (n_bins + 1) * 4
}

/// Mirrors `bls_cuda.py`'s `_env_flag(name)`: truthy iff the env var is set
/// to `"1"`/`"true"` (case-insensitive).
fn env_flag(name: &str) -> bool {
    std::env::var(name).map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true")).unwrap_or(false)
}

/// Mirrors `bls_cuda.py`'s `_f32_max_points()`: env override
/// (`EXONIT_BLS_CUDA_F32_MAX_PTS`) or `F32_MAX_POINTS_DEFAULT`. An
/// unparseable value falls back to the default with a log line rather than
/// panicking — a config typo must not crash injection, only lose the f32
/// speedup (same rationale as the Python `except ValueError` branch).
fn f32_max_points() -> usize {
    match std::env::var("EXONIT_BLS_CUDA_F32_MAX_PTS") {
        Ok(raw) if !raw.trim().is_empty() => raw.trim().parse::<usize>().unwrap_or_else(|_| {
            eprintln!(
                "bls::cuda_batch: EXONIT_BLS_CUDA_F32_MAX_PTS={raw:?} is not a valid integer -- \
                 falling back to the default ({F32_MAX_POINTS_DEFAULT})."
            );
            F32_MAX_POINTS_DEFAULT
        }),
        _ => F32_MAX_POINTS_DEFAULT,
    }
}

/// Mirrors `bls_cuda.py`'s `_log_size_route_once`/`_size_route_logged`: the
/// size-routing fallback is normal per-call behaviour for giant targets
/// (injection calls this ~35x per star), not a misconfiguration, so it is
/// logged once per process at most — not once per call, which would spam
/// the worker log on a giant multi-sector FFI target.
static SIZE_ROUTE_LOGGED: std::sync::Once = std::sync::Once::new();

fn log_size_route_once(n_pts: usize, threshold: usize) {
    SIZE_ROUTE_LOGGED.call_once(|| {
        eprintln!(
            "bls::cuda_batch: N={n_pts} points exceeds the f32 CPU-fold threshold ({threshold}) -- \
             routing to the f64 GPU-fold kernel for this and every subsequent oversized call in \
             this process (measured crossover, 2026-07-12 <host> bench: past this point the \
             f64 kernel's on-GPU fold beats the f32 kernel's CPU fold). Override with env \
             EXONIT_BLS_CUDA_F32_MAX_PTS."
        );
    });
}

/// An initialized CUDA device + the f64 batch kernel (compiled once, like
/// `super::cuda::CudaBlsBackend`) plus a lazily-populated cache of f32
/// batch kernels keyed by `n_bins` — mirrors `bls_cuda.py`'s
/// `_f32_kernel_cache` dict (`_get_f32_kernel`), since the f32 kernel's
/// source is `n_bins`-template-substituted at compile time and therefore
/// cannot be compiled once up front the way the f64 kernel can.
pub struct CudaBlsBatchBackend {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    fused_f64: CudaFunction,
    f32_kernels: Mutex<HashMap<usize, CudaFunction>>,
}

impl std::fmt::Debug for CudaBlsBatchBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CudaBlsBatchBackend { .. }")
    }
}

impl CudaBlsBatchBackend {
    /// Probe for a usable CUDA GPU and NVRTC-compile the f64 kernel, ONCE —
    /// same failure-logging posture as `CudaBlsBackend::probe`; see that
    /// function's doc comment (every failure branch here mirrors it
    /// exactly, just for this module's kernel).
    pub fn probe() -> Option<Self> {
        // SAFETY: `is_culib_present` only attempts a `dlopen()` of the
        // driver library to check whether it resolves; it does not call
        // into the library.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            eprintln!(
                "bls::cuda_batch: libcuda not found via the dynamic loader (check LD_LIBRARY_PATH) \
                 — falling back to the CPU injection-recovery path"
            );
            return None;
        }
        // SAFETY: same as above, for libnvrtc.
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            eprintln!(
                "bls::cuda_batch: libnvrtc not found via the dynamic loader (check LD_LIBRARY_PATH \
                 includes .../nvidia/cuda_nvrtc/lib) — falling back to the CPU injection-recovery path"
            );
            return None;
        }

        let ctx = match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(error) => {
                eprintln!(
                    "bls::cuda_batch: CUDA present but unusable (CudaContext::new failed: {error}) \
                     — falling back to the CPU injection-recovery path"
                );
                return None;
            }
        };

        let fused_f64_ptx = match compile_ptx(FUSED_F64_SRC) {
            Ok(ptx) => ptx,
            Err(error) => {
                eprintln!("bls::cuda_batch: NVRTC failed to compile bls_batch_fused: {error} — falling back to the CPU injection-recovery path");
                return None;
            }
        };
        let fused_f64_module = match ctx.load_module(fused_f64_ptx) {
            Ok(module) => module,
            Err(error) => {
                eprintln!("bls::cuda_batch: failed to load the bls_batch_fused module: {error} — falling back to the CPU injection-recovery path");
                return None;
            }
        };
        let fused_f64 = match fused_f64_module.load_function("bls_batch_fused") {
            Ok(function) => function,
            Err(error) => {
                eprintln!("bls::cuda_batch: failed to resolve the bls_batch_fused kernel: {error} — falling back to the CPU injection-recovery path");
                return None;
            }
        };

        let stream = ctx.default_stream();
        Some(CudaBlsBatchBackend { ctx, stream, fused_f64, f32_kernels: Mutex::new(HashMap::new()) })
    }

    /// Compiles (or fetches from cache) the f32 kernel for a given
    /// `n_bins`. Mirrors `bls_cuda.py`'s `_get_f32_kernel`. Unlike
    /// `probe()`'s one-shot kernels, a compile failure HERE is a real
    /// runtime error (`BlsError::Backend`), not a fallback point — matches
    /// `super::cuda`'s "probed once, no per-call fallback" contract:
    /// `search_batch` has already committed to the f32 path by the time
    /// this is called.
    fn get_f32_kernel(&self, n_bins: usize) -> Result<CudaFunction, BlsError> {
        // Serialize first-use compilation so concurrent tasks cannot compile
        // and load the same n_bins-specialized kernel more than once.
        let mut cache = self.f32_kernels.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(function) = cache.get(&n_bins) {
            return Ok(function.clone());
        }
        let tgsums = TG_FLUX as usize * (n_bins + 1);
        let src = F32_KERNEL_SRC.replace("@NB@", &n_bins.to_string()).replace("@TGSUMS@", &tgsums.to_string());
        let ptx = compile_ptx(&src).map_err(driver_err("NVRTC compile bls_inj_batch_f32"))?;
        let module = self.ctx.load_module(ptx).map_err(driver_err("load bls_inj_batch_f32 module"))?;
        let function = module.load_function("bls_inj_batch_f32").map_err(driver_err("resolve bls_inj_batch_f32 kernel"))?;
        cache.insert(n_bins, function.clone());
        Ok(function)
    }

    /// Mirrors `bls_cuda.bls_gpu_batch(time, flux_list, ...)` exactly: same
    /// three-way f64/f32 routing decision (env override, shared-memory
    /// budget, point-count crossover), same per-flux result contract as
    /// `bls_fast_batch`/`bls_gpu_batch`. `flux_list[i].len()` must equal
    /// `time.len()` for every `i` — validated here (not validated in the
    /// Python function itself, which relies on its caller/numpy to catch
    /// bad input; same added-safety-net posture as `CudaBlsBackend::search`).
    pub fn search_batch(&self, time: &[f64], flux_list: &[Vec<f64>], p: &BlsParams) -> Result<Vec<BlsResult>, BlsError> {
        if flux_list.is_empty() {
            return Ok(Vec::new());
        }
        for flux in flux_list {
            if flux.len() != time.len() {
                return Err(BlsError::MismatchedLengths);
            }
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

        // Priority order verbatim from `bls_gpu_batch`'s doc comment: (1)
        // env escape hatch always wins; (2) f32 shared-memory budget; (3)
        // f32 CPU-fold point-count crossover.
        let mut use_f64 = env_flag("EXONIT_BLS_CUDA_F64");
        if !use_f64 {
            let budget = shared_mem_bytes(p.bin_count);
            if budget > CUDA_SHARED_MEM_BUDGET_BYTES {
                eprintln!(
                    "bls::cuda_batch: n_bins={} needs {} B of CUDA shared memory (TG_FLUX={}) per \
                     block, over the {} B default budget -- falling back to the f64 kernel path for \
                     this call. Lower n_bins/injection_n_bins to use the faster f32 path (the \
                     production default 150 needs {} B, well under budget).",
                    p.bin_count,
                    budget,
                    TG_FLUX,
                    CUDA_SHARED_MEM_BUDGET_BYTES,
                    shared_mem_bytes(150)
                );
                use_f64 = true;
            }
        }
        if !use_f64 {
            let n_pts = time.len();
            let threshold = f32_max_points();
            if n_pts > threshold {
                log_size_route_once(n_pts, threshold);
                use_f64 = true;
            }
        }

        if use_f64 {
            self.search_batch_f64(time, flux_list, p)
        } else {
            self.search_batch_f32(time, flux_list, p)
        }
    }

    /// Mirrors `bls_cuda._bls_gpu_batch_f64` exactly: one thread per
    /// `(period, flux)` pair, all-float64, phase-fold-on-GPU. Called
    /// directly by tests wanting to exercise this specific kernel
    /// regardless of routing (`search_batch`'s own routing is tested
    /// separately).
    pub fn search_batch_f64(&self, time: &[f64], flux_list: &[Vec<f64>], p: &BlsParams) -> Result<Vec<BlsResult>, BlsError> {
        let n_flux = flux_list.len();
        // "Clamp n_bins for the in-kernel static arrays" — bls_cuda.py:502,
        // `n_bins = min(n_bins, 256)` (the kernel's `bin_sums[256]`/
        // `bin_counts[256]` per-thread local arrays are sized for exactly
        // this ceiling).
        let n_bins = p.bin_count.min(256);
        let n_periods = p.period_count;
        let sample_count = time.len();

        let t0_reference = time[0];
        let time_relative: Vec<f64> = time.iter().map(|&v| v - t0_reference).collect();
        let periods = reference::frequency_period_grid(p.period_minimum, p.period_maximum, n_periods);

        let clamped_params = BlsParams { bin_count: n_bins, ..*p };
        let duration_bins = reference::duration_bin_grid(&clamped_params);
        let n_dur_steps_actual = duration_bins.len();
        let duration_bins_i32: Vec<i32> = duration_bins.iter().map(|&v| v as i32).collect();

        let total_sums: Vec<f64> = flux_list.iter().map(|flux| flux.iter().sum()).collect();

        // Transpose for coalesced kernel reads: flux_t[i*n_flux + fi].
        let mut flux_t = vec![0.0f64; sample_count * n_flux];
        for (fi, flux) in flux_list.iter().enumerate() {
            for (i, &value) in flux.iter().enumerate() {
                flux_t[i * n_flux + fi] = value;
            }
        }

        let block_size: u32 = 128;
        let total_tasks = (n_periods * n_flux) as u32;
        let grid_size = total_tasks.div_ceil(block_size);
        let launch_cfg = LaunchConfig { grid_dim: (grid_size, 1, 1), block_dim: (block_size, 1, 1), shared_mem_bytes: 0 };

        let stream = &self.stream;
        let d_time = stream.clone_htod(&time_relative).map_err(driver_err("upload time"))?;
        let d_flux_t = stream.clone_htod(&flux_t).map_err(driver_err("upload flux_t"))?;
        let d_periods = stream.clone_htod(&periods).map_err(driver_err("upload periods"))?;
        let d_dur_bins = stream.clone_htod(&duration_bins_i32).map_err(driver_err("upload duration bins"))?;
        let d_total_sums = stream.clone_htod(&total_sums).map_err(driver_err("upload total sums"))?;

        let mut d_best_powers: CudaSlice<f64> =
            stream.alloc_zeros(n_flux * n_periods).map_err(driver_err("alloc best_powers"))?;
        let mut d_best_depths: CudaSlice<f64> =
            stream.alloc_zeros(n_flux * n_periods).map_err(driver_err("alloc best_depths"))?;
        let mut d_best_dur_idx: CudaSlice<i32> =
            stream.alloc_zeros(n_flux * n_periods).map_err(driver_err("alloc best_dur_idx"))?;
        let mut d_best_phase_idx: CudaSlice<i32> =
            stream.alloc_zeros(n_flux * n_periods).map_err(driver_err("alloc best_phase_idx"))?;

        let n_i32 = sample_count as i32;
        let n_bins_i32 = n_bins as i32;
        let n_periods_i32 = n_periods as i32;
        let n_dur_i32 = n_dur_steps_actual as i32;
        let n_flux_i32 = n_flux as i32;

        {
            let mut launch = stream.launch_builder(&self.fused_f64);
            launch.arg(&d_time);
            launch.arg(&d_flux_t);
            launch.arg(&d_periods);
            launch.arg(&d_dur_bins);
            launch.arg(&d_total_sums);
            launch.arg(&mut d_best_powers);
            launch.arg(&mut d_best_depths);
            launch.arg(&mut d_best_dur_idx);
            launch.arg(&mut d_best_phase_idx);
            launch.arg(&n_i32);
            launch.arg(&n_bins_i32);
            launch.arg(&n_periods_i32);
            launch.arg(&n_dur_i32);
            launch.arg(&n_flux_i32);
            // SAFETY: kernel argument types/order/count match
            // `bls_batch_fused`'s C signature exactly (verbatim copy, see
            // `cuda_kernels/bls_batch_fused_f64.cu`); all device buffers are
            // sized for `n_flux * n_periods` as the kernel expects.
            unsafe { launch.launch(launch_cfg) }.map_err(driver_err("launch bls_batch_fused"))?;
        }

        let best_powers: Vec<f64> = stream.clone_dtoh(&d_best_powers).map_err(driver_err("download best_powers"))?;
        let best_depths: Vec<f64> = stream.clone_dtoh(&d_best_depths).map_err(driver_err("download best_depths"))?;
        let best_dur_indices: Vec<i32> =
            stream.clone_dtoh(&d_best_dur_idx).map_err(driver_err("download best_dur_idx"))?;
        let best_phase_indices: Vec<i32> =
            stream.clone_dtoh(&d_best_phase_idx).map_err(driver_err("download best_phase_idx"))?;

        let mut results = Vec::with_capacity(n_flux);
        for fi in 0..n_flux {
            let row = &best_powers[fi * n_periods..(fi + 1) * n_periods];
            // `np.argmax` — first occurrence of the maximum.
            let mut best_index = 0usize;
            for index in 1..row.len() {
                if row[index] > row[best_index] {
                    best_index = index;
                }
            }
            let idx = fi * n_periods + best_index;
            let best_period = periods[best_index];
            let best_dur_bin = duration_bins[best_dur_indices[idx] as usize];
            let best_duration = best_dur_bin as f64 / n_bins as f64 * best_period;
            let best_phase = best_phase_indices[idx] as f64 / n_bins as f64;
            let depth = best_depths[idx];
            results.push(BlsResult {
                period: best_period,
                power: row[best_index],
                duration: best_duration,
                t0: best_phase * best_period + t0_reference,
                depth,
                depth_ppm: depth * 1_000_000.0,
                snr: reference::sde(row, best_index),
                periods: periods.clone(),
                power_spectrum: row.to_vec(),
            });
        }
        Ok(results)
    }

    /// Mirrors `bls_cuda._bls_gpu_batch_f32` exactly: CPU phase-fold
    /// (`_phase_fold_cpu`/`_counts_per_period`, chunked over periods to
    /// bound host RAM — `PERIOD_CHUNK_DEFAULT`) feeding a per-(flux,period)
    /// f32 kernel that only does the sliding-window search, mean-centering
    /// in f64 before the f32 cast (bls_cuda.py's step 4). Called directly
    /// by tests wanting to exercise this specific kernel regardless of
    /// routing.
    pub fn search_batch_f32(&self, time: &[f64], flux_list: &[Vec<f64>], p: &BlsParams) -> Result<Vec<BlsResult>, BlsError> {
        let n_flux = flux_list.len();
        // NOT clamped — `search_batch`'s shared-memory budget guard already
        // guarantees `n_bins` fits before this is ever called, matching
        // `_bls_gpu_batch_f32`'s own lack of a clamp.
        let n_bins = p.bin_count;
        let n_periods = p.period_count;
        let sample_count = time.len();

        let t0_reference = time[0];
        let time_relative: Vec<f64> = time.iter().map(|&v| v - t0_reference).collect();
        let periods = reference::frequency_period_grid(p.period_minimum, p.period_maximum, n_periods);

        let duration_bins = reference::duration_bin_grid(p);
        let n_dur = duration_bins.len();
        let widths_i32: Vec<i32> = duration_bins.iter().map(|&v| v as i32).collect();

        // Mean-center in f64 before the f32 cast (depth/SR are
        // shift-invariant; centering kills f32 accumulation error).
        let flux_means: Vec<f64> = flux_list.iter().map(|flux| flux.iter().sum::<f64>() / sample_count as f64).collect();
        let flux_centered: Vec<Vec<f64>> =
            flux_list.iter().zip(&flux_means).map(|(flux, &mean)| flux.iter().map(|&v| v - mean).collect()).collect();
        let totals_f32: Vec<f32> = flux_centered.iter().map(|flux| flux.iter().sum::<f64>() as f32).collect();

        let mut flux_t_f32 = vec![0f32; sample_count * n_flux];
        for (fi, flux) in flux_centered.iter().enumerate() {
            for (i, &value) in flux.iter().enumerate() {
                flux_t_f32[i * n_flux + fi] = value as f32;
            }
        }

        let kernel = self.get_f32_kernel(n_bins)?;

        let stream = &self.stream;
        let d_flux_t = stream.clone_htod(&flux_t_f32).map_err(driver_err("upload fluxT"))?;
        let d_totals = stream.clone_htod(&totals_f32).map_err(driver_err("upload totals"))?;
        let d_widths = stream.clone_htod(&widths_i32).map_err(driver_err("upload widths"))?;

        // Allocate the period-chunk workspace once per batch. Previously all
        // six buffers below were allocated and freed for every chunk. Besides
        // the driver overhead, dropping a CudaSlice may synchronize the stream,
        // preventing adjacent chunks from flowing through the same stream.
        // Views restrict the final short chunk to its actual length, so neither
        // transfers nor result processing include unused workspace elements.
        let max_chunk_periods = PERIOD_CHUNK_DEFAULT.min(n_periods);
        let mut d_bin_idx: CudaSlice<i16> = stream
            .alloc_zeros(max_chunk_periods * sample_count)
            .map_err(driver_err("alloc bin_idx workspace"))?;
        let mut d_counts: CudaSlice<f32> = stream
            .alloc_zeros(max_chunk_periods * n_bins)
            .map_err(driver_err("alloc counts workspace"))?;
        let mut d_params: CudaSlice<i32> = stream.alloc_zeros(5).map_err(driver_err("alloc params workspace"))?;
        let mut d_out_power: CudaSlice<f32> = stream
            .alloc_zeros(n_flux * max_chunk_periods)
            .map_err(driver_err("alloc out_power workspace"))?;
        let mut d_out_wj: CudaSlice<u32> = stream
            .alloc_zeros(n_flux * max_chunk_periods)
            .map_err(driver_err("alloc out_wj workspace"))?;
        let mut d_out_depth: CudaSlice<f32> = stream
            .alloc_zeros(n_flux * max_chunk_periods)
            .map_err(driver_err("alloc out_depth workspace"))?;

        let n_flux_pad = (n_flux + TG_FLUX as usize - 1) / TG_FLUX as usize * TG_FLUX as usize;
        let grid_x = (n_flux_pad / TG_FLUX as usize) as u32;

        // (n_flux, n_periods) row-major accumulators, filled chunk by chunk.
        let mut power_all = vec![0f64; n_flux * n_periods];
        let mut depth_all = vec![0f64; n_flux * n_periods];
        let mut best_w_all = vec![0f64; n_flux * n_periods];
        let mut best_j_all = vec![0f64; n_flux * n_periods];

        let mut period_start = 0usize;
        while period_start < n_periods {
            let period_end = (period_start + PERIOD_CHUNK_DEFAULT).min(n_periods);
            let n_per = period_end - period_start;

            // `_phase_fold_cpu`: exact-f64 phase-fold, chunked over periods.
            let mut bin_idx = vec![0i16; n_per * sample_count];
            for (local_p, &period) in periods[period_start..period_end].iter().enumerate() {
                for (i, &t_rel) in time_relative.iter().enumerate() {
                    let raw = t_rel / period;
                    let phase = raw - raw.floor();
                    let bin = ((phase * n_bins as f64) as i32).clamp(0, n_bins as i32 - 1);
                    bin_idx[local_p * sample_count + i] = bin as i16;
                }
            }
            // `_counts_per_period`: exact per-period bin counts.
            let mut counts = vec![0f32; n_per * n_bins];
            for local_p in 0..n_per {
                for i in 0..sample_count {
                    let bin = bin_idx[local_p * sample_count + i] as usize;
                    counts[local_p * n_bins + bin] += 1.0;
                }
            }

            let params_i32: [i32; 5] = [sample_count as i32, n_flux as i32, n_bins as i32, n_per as i32, n_dur as i32];
            let bin_idx_len = n_per * sample_count;
            let counts_len = n_per * n_bins;
            let output_len = n_flux * n_per;
            stream
                .memcpy_htod(&bin_idx, &mut d_bin_idx.slice_mut(..bin_idx_len))
                .map_err(driver_err("upload bin_idx"))?;
            stream
                .memcpy_htod(&counts, &mut d_counts.slice_mut(..counts_len))
                .map_err(driver_err("upload counts"))?;
            stream
                .memcpy_htod(&params_i32, &mut d_params.slice_mut(..))
                .map_err(driver_err("upload params"))?;
            let bin_idx_view = d_bin_idx.slice(..bin_idx_len);
            let counts_view = d_counts.slice(..counts_len);
            let params_view = d_params.slice(..);
            let mut out_power_view = d_out_power.slice_mut(..output_len);
            let mut out_wj_view = d_out_wj.slice_mut(..output_len);
            let mut out_depth_view = d_out_depth.slice_mut(..output_len);

            let launch_cfg =
                LaunchConfig { grid_dim: (grid_x, n_per as u32, 1), block_dim: (TG_FLUX, 1, 1), shared_mem_bytes: 0 };
            {
                let mut launch = stream.launch_builder(&kernel);
                launch.arg(&bin_idx_view);
                launch.arg(&counts_view);
                launch.arg(&d_flux_t);
                launch.arg(&d_totals);
                launch.arg(&d_widths);
                launch.arg(&params_view);
                launch.arg(&mut out_power_view);
                launch.arg(&mut out_wj_view);
                launch.arg(&mut out_depth_view);
                // SAFETY: matches `bls_inj_batch_f32`'s C signature exactly
                // (template-substituted verbatim copy, see
                // `cuda_kernels/bls_batch_fused_f32.cu`); `n_flux_pad`
                // padding threads (`f >= n_flux`) return before writing any
                // output, so the `n_flux * n_per`-sized output buffers are
                // never written out of bounds.
                unsafe { launch.launch(launch_cfg) }.map_err(driver_err("launch bls_inj_batch_f32"))?;
            }

            let mut out_power = vec![0.0f32; output_len];
            let mut out_wj = vec![0u32; output_len];
            let mut out_depth = vec![0.0f32; output_len];
            stream.memcpy_dtoh(&out_power_view, &mut out_power).map_err(driver_err("download out_power"))?;
            stream.memcpy_dtoh(&out_wj_view, &mut out_wj).map_err(driver_err("download out_wj"))?;
            stream.memcpy_dtoh(&out_depth_view, &mut out_depth).map_err(driver_err("download out_depth"))?;

            for fi in 0..n_flux {
                for local_p in 0..n_per {
                    let src = fi * n_per + local_p;
                    let dst = fi * n_periods + (period_start + local_p);
                    power_all[dst] = out_power[src] as f64;
                    depth_all[dst] = out_depth[src] as f64;
                    let wj = out_wj[src];
                    best_w_all[dst] = (wj >> 16) as f64;
                    best_j_all[dst] = (wj & 0xFFFF) as f64;
                }
            }

            period_start = period_end;
        }

        let mut results = Vec::with_capacity(n_flux);
        for fi in 0..n_flux {
            let row = &power_all[fi * n_periods..(fi + 1) * n_periods];
            let mut best_index = 0usize;
            for index in 1..row.len() {
                if row[index] > row[best_index] {
                    best_index = index;
                }
            }
            let idx = fi * n_periods + best_index;
            let best_period = periods[best_index];
            let snr = reference::sde(row, best_index);
            let duration = best_w_all[idx] / n_bins as f64 * best_period;
            let t0 = best_j_all[idx] / n_bins as f64 * best_period + t0_reference;
            let depth = depth_all[idx];
            results.push(BlsResult {
                period: best_period,
                power: row[best_index],
                duration,
                t0,
                depth,
                depth_ppm: depth * 1_000_000.0,
                snr,
                periods: periods.clone(),
                power_spectrum: row.to_vec(),
            });
        }
        Ok(results)
    }
}

/// CPU-only routing-decision unit tests — no GPU needed (same posture as
/// `test_bls_cuda_parity.py`'s `TestSharedMemBudgetGuard`/
/// `TestBlsGpuBatchRouting`, which are plain, un-skipped test classes since
/// they only exercise host-side arithmetic and env-var parsing). Run as
/// part of `cargo test --features cuda` (NOT gated behind `--ignored`) —
/// they need the `cuda` Cargo feature to compile (this whole module is
/// feature-gated) but never touch the GPU.
///
/// Env-var tests use a process-wide `Mutex` to serialize `std::env::var`
/// mutation across the crate's `cargo test` harness (which runs tests in
/// parallel threads by default) — `std::env` is process-global state,
/// so two tests setting/clearing the same var concurrently would otherwise
/// race and produce flaky failures unrelated to the code under test.
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Verbatim numbers from `test_bls_cuda_parity.py::
    /// TestSharedMemBudgetGuard::test_shared_mem_bytes_matches_spec_values`/
    /// `test_oversized_n_bins_exceeds_budget`: n_bins=150 -> 19932 B (under
    /// budget), n_bins=200 -> 26532 B (under budget), n_bins=400 -> 52932 B
    /// (over the 48 KiB budget).
    #[test]
    fn shared_mem_bytes_matches_python_spec_values() {
        assert_eq!(shared_mem_bytes(150), 19_932);
        assert_eq!(shared_mem_bytes(200), 26_532);
        assert!(shared_mem_bytes(150) < CUDA_SHARED_MEM_BUDGET_BYTES);
        assert!(shared_mem_bytes(200) < CUDA_SHARED_MEM_BUDGET_BYTES);
        assert_eq!(shared_mem_bytes(400), 52_932);
        assert!(shared_mem_bytes(400) > CUDA_SHARED_MEM_BUDGET_BYTES);
    }

    /// Mirrors `_env_flag`'s truthy variants
    /// (`test_env_flag_truthy_variants_route_to_f64`'s parametrization).
    #[test]
    fn env_flag_truthy_variants() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        for value in ["1", "true", "True", "TRUE"] {
            // SAFETY: serialized by ENV_LOCK; this crate/tests do not read
            // this var from any other thread concurrently.
            unsafe { std::env::set_var("EXONIT_BLS_CUDA_F64_TEST_TRUTHY", value) };
            assert!(env_flag("EXONIT_BLS_CUDA_F64_TEST_TRUTHY"), "expected {value:?} to be truthy");
        }
        unsafe { std::env::remove_var("EXONIT_BLS_CUDA_F64_TEST_TRUTHY") };
    }

    /// Mirrors `_env_flag`'s falsy variants
    /// (`test_env_flag_falsy_variants_route_to_f32`'s parametrization).
    #[test]
    fn env_flag_falsy_variants() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        for value in ["0", "false", "", "garbage"] {
            unsafe { std::env::set_var("EXONIT_BLS_CUDA_F64_TEST_FALSY", value) };
            assert!(!env_flag("EXONIT_BLS_CUDA_F64_TEST_FALSY"), "expected {value:?} to be falsy");
        }
        unsafe { std::env::remove_var("EXONIT_BLS_CUDA_F64_TEST_FALSY") };
        assert!(!env_flag("EXONIT_BLS_CUDA_F64_TEST_FALSY_UNSET"), "an unset var must be falsy");
    }

    /// Mirrors `test_invalid_env_max_pts_falls_back_to_default`.
    #[test]
    fn f32_max_points_falls_back_to_default_on_invalid_value() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        unsafe { std::env::set_var("EXONIT_BLS_CUDA_F32_MAX_PTS", "not-an-int") };
        assert_eq!(f32_max_points(), F32_MAX_POINTS_DEFAULT);
        unsafe { std::env::remove_var("EXONIT_BLS_CUDA_F32_MAX_PTS") };
        assert_eq!(f32_max_points(), F32_MAX_POINTS_DEFAULT, "unset must also use the default");
    }

    /// Mirrors `test_env_max_pts_override_routes_small_workload_to_f64`'s
    /// premise: the ceiling is fully env-controlled.
    #[test]
    fn f32_max_points_honours_valid_override() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        unsafe { std::env::set_var("EXONIT_BLS_CUDA_F32_MAX_PTS", "50") };
        assert_eq!(f32_max_points(), 50);
        unsafe { std::env::remove_var("EXONIT_BLS_CUDA_F32_MAX_PTS") };
    }
}
