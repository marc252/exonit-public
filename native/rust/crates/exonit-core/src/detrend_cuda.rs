//! CUDA driver-API port of `~/exonit/worker/wotan_cuda.py` — the
//! single `biweight_running` kernel, used by `detrend_biweight()` after
//! host-side segment splitting on `break_tolerance` (same as
//! `wotan_cuda.detrend_biweight`).
//!
//! **F4b scope** (docs/PLAN.md): CUDA wotan biweight detrend. Selection at
//! the `PrimaryTaskProcessor` level mirrors `EXONIT_WOTAN_BACKEND` exactly
//! (see `task_processor.rs`'s `resolve_detrend_backend` doc comment) — this
//! module itself only implements the kernel + host-side segment loop, same
//! division of responsibility as `bls::cuda`/`bls::cuda_batch`.
//!
//! # Backend: `cudarc`, dynamic-loading — same posture as `bls::cuda`
//!
//! See `bls::cuda`'s module doc comment for the full explanation (no
//! link-time libcuda/libnvrtc dependency, `LD_LIBRARY_PATH` requirements at
//! runtime on <host>). `bls::cuda_batch`'s doc comment explains why an
//! independent `CudaContext::new(0)` call here — a third one, alongside the
//! primary-BLS and batch-BLS contexts — is safe (CUDA's primary-context API
//! is reference-counted; all three Rust `Arc<CudaContext>` handles wrap the
//! SAME underlying driver context).
//!
//! # No transit-mask support (matches Python exactly)
//!
//! `wotan_cuda.py`'s module doc comment: "this PoC only implements the
//! ... wotan defaults exonit actually exercises ... no NaN mask (callers
//! pre-filter)". This module's public API takes no mask for the same
//! reason `bls_cuda.py`'s biweight kernel doesn't: `detrend::DetrendBackend`
//! (the call-site selector in `detrend.rs`) only ever routes here when
//! `transit_mask.is_none()` — exactly `wotan_detrend.py`'s
//! `_should_use_cuda`'s `if transit_mask is not None: return False` gate.
//! Callers passing a mask always take the CPU path; this module has no
//! masked variant to call by construction, not by omission.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

/// Verbatim source of `biweight_running` — see
/// `detrend_cuda_kernels/biweight_running.cu`'s header comment for the
/// exact origin (`wotan_cuda.py` lines 73-233).
const BIWEIGHT_RUNNING_SRC: &str = include_str!("detrend_cuda_kernels/biweight_running.cu");

/// Threads per block — matches `biweight_running_segment`'s `block = 128`
/// (`wotan_cuda.py:292`) exactly: "register-heavy kernel; 128 keeps in
/// registers on Turing+".
const BLOCK_SIZE: u32 = 128;

#[derive(Debug)]
pub enum DetrendCudaError {
    LengthMismatch,
    Backend(String),
}

impl std::fmt::Display for DetrendCudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DetrendCudaError::LengthMismatch => write!(f, "time and flux lengths differ"),
            DetrendCudaError::Backend(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for DetrendCudaError {}

fn driver_err<E: std::fmt::Display>(context: &str) -> impl Fn(E) -> DetrendCudaError + '_ {
    move |error| DetrendCudaError::Backend(format!("{context}: {error}"))
}

/// An initialized CUDA device + the compiled `biweight_running` kernel,
/// built once by [`probe`](Self::probe) — same "probe once, reuse forever"
/// posture as `bls::cuda::CudaBlsBackend`/`bls::cuda_batch::CudaBlsBatchBackend`.
pub struct CudaDetrendBackend {
    stream: Arc<CudaStream>,
    biweight_running: CudaFunction,
}

impl std::fmt::Debug for CudaDetrendBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CudaDetrendBackend { .. }")
    }
}

impl CudaDetrendBackend {
    /// Probe for a usable CUDA GPU and NVRTC-compile the kernel, ONCE.
    /// Returns `None` (after logging exactly one `eprintln!`) on any
    /// failure — same failure-logging convention as `bls::cuda`'s `probe()`.
    pub fn probe() -> Option<Self> {
        // SAFETY: `is_culib_present` only attempts a `dlopen()` of the
        // driver library to check whether it resolves; it does not call
        // into the library.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            eprintln!(
                "detrend_cuda: libcuda not found via the dynamic loader (check LD_LIBRARY_PATH) \
                 — falling back to the CPU wotan detrend path"
            );
            return None;
        }
        // SAFETY: same as above, for libnvrtc.
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            eprintln!(
                "detrend_cuda: libnvrtc not found via the dynamic loader (check LD_LIBRARY_PATH \
                 includes .../nvidia/cuda_nvrtc/lib) — falling back to the CPU wotan detrend path"
            );
            return None;
        }

        let ctx = match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(error) => {
                eprintln!(
                    "detrend_cuda: CUDA present but unusable (CudaContext::new failed: {error}) \
                     — falling back to the CPU wotan detrend path"
                );
                return None;
            }
        };

        let ptx = match compile_ptx(BIWEIGHT_RUNNING_SRC) {
            Ok(ptx) => ptx,
            Err(error) => {
                eprintln!("detrend_cuda: NVRTC failed to compile biweight_running: {error} — falling back to the CPU wotan detrend path");
                return None;
            }
        };
        let module = match ctx.load_module(ptx) {
            Ok(module) => module,
            Err(error) => {
                eprintln!("detrend_cuda: failed to load the biweight_running module: {error} — falling back to the CPU wotan detrend path");
                return None;
            }
        };
        let biweight_running = match module.load_function("biweight_running") {
            Ok(function) => function,
            Err(error) => {
                eprintln!("detrend_cuda: failed to resolve the biweight_running kernel: {error} — falling back to the CPU wotan detrend path");
                return None;
            }
        };

        let stream = ctx.default_stream();
        Some(CudaDetrendBackend { stream, biweight_running })
    }

    /// Mirrors `wotan_cuda.biweight_running_segment(time, flux, window_length,
    /// edge_cutoff, cval)` exactly: single contiguous segment, no masking,
    /// no NaN handling (see module doc comment). `time` MUST be sorted
    /// ascending (same precondition as the Python function, which relies on
    /// binary search inside the kernel).
    pub fn biweight_running_segment(
        &self,
        time: &[f64],
        flux: &[f64],
        window_length: f64,
        edge_cutoff: f64,
        cval: f64,
    ) -> Result<Vec<f64>, DetrendCudaError> {
        if time.len() != flux.len() {
            return Err(DetrendCudaError::LengthMismatch);
        }
        let n = time.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let half_window = window_length / 2.0;
        let t_min = time.iter().copied().fold(f64::INFINITY, f64::min);
        let t_max = time.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut low_index = t_min + edge_cutoff;
        let mut hi_index = t_max - edge_cutoff;
        if edge_cutoff > half_window {
            let edge_cutoff_eff = half_window;
            low_index = t_min + edge_cutoff_eff;
            hi_index = t_max - edge_cutoff_eff;
        }

        let grid_size = (n as u32).div_ceil(BLOCK_SIZE);
        let launch_cfg = LaunchConfig { grid_dim: (grid_size, 1, 1), block_dim: (BLOCK_SIZE, 1, 1), shared_mem_bytes: 0 };

        let stream = &self.stream;
        let d_time = stream.clone_htod(time).map_err(driver_err("upload time"))?;
        let d_flux = stream.clone_htod(flux).map_err(driver_err("upload flux"))?;
        let mut d_trend: CudaSlice<f64> = stream.alloc_zeros(n).map_err(driver_err("alloc trend"))?;

        let half_window_arg = half_window;
        let low_index_arg = low_index;
        let hi_index_arg = hi_index;
        let cval_arg = cval as f32;
        let n_arg = n as i32;

        {
            let mut launch = stream.launch_builder(&self.biweight_running);
            launch.arg(&d_time);
            launch.arg(&d_flux);
            launch.arg(&mut d_trend);
            launch.arg(&half_window_arg);
            launch.arg(&low_index_arg);
            launch.arg(&hi_index_arg);
            launch.arg(&cval_arg);
            launch.arg(&n_arg);
            // SAFETY: kernel argument types/order/count match
            // `biweight_running`'s C signature exactly (verbatim copy, see
            // `detrend_cuda_kernels/biweight_running.cu`); `d_trend` is
            // sized `n` as the kernel expects (one thread per sample).
            unsafe { launch.launch(launch_cfg) }.map_err(driver_err("launch biweight_running"))?;
        }

        stream.clone_dtoh(&d_trend).map_err(driver_err("download trend"))
    }

    /// Mirrors `wotan_cuda.detrend_biweight(time, flux, window_length,
    /// edge_cutoff, cval, break_tolerance)` exactly: splits `time` into
    /// contiguous segments wherever the gap exceeds `break_tolerance`, runs
    /// [`biweight_running_segment`](Self::biweight_running_segment) on each
    /// segment of length >= 3 (shorter segments stay `NaN` in the trend,
    /// matching Python's `if e - s < 3: continue`), then returns
    /// `(flux / trend, trend)` — wotan's convention.
    pub fn detrend_biweight(
        &self,
        time: &[f64],
        flux: &[f64],
        window_length: f64,
        edge_cutoff: f64,
        cval: f64,
        break_tolerance: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), DetrendCudaError> {
        if time.len() != flux.len() {
            return Err(DetrendCudaError::LengthMismatch);
        }
        let n = time.len();

        // `np.diff(t) > break_tolerance` -> `np.flatnonzero(...) + 1`;
        // `starts = [0] + breaks`, `ends = breaks + [len(t)]`.
        let mut starts: Vec<usize> = vec![0];
        let mut ends: Vec<usize> = Vec::new();
        if break_tolerance > 0.0 {
            for index in 1..n {
                if time[index] - time[index - 1] > break_tolerance {
                    ends.push(index);
                    starts.push(index);
                }
            }
        }
        ends.push(n);

        let mut trend = vec![f64::NAN; n];
        for (&start, &end) in starts.iter().zip(ends.iter()) {
            if end.saturating_sub(start) < 3 {
                continue;
            }
            let segment_trend =
                self.biweight_running_segment(&time[start..end], &flux[start..end], window_length, edge_cutoff, cval)?;
            trend[start..end].copy_from_slice(&segment_trend);
        }

        let flat: Vec<f64> = flux.iter().zip(trend.iter()).map(|(f, t)| f / t).collect();
        Ok((flat, trend))
    }
}
