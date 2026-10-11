//! OpenCL port of the BATCHED injection-recovery BLS path in
//! `~/exonit/worker/bls_opencl.py` — `bls_gpu_batch`, using the
//! `bls_inj_batch_f32` kernel whose exact OpenCL C source lives at
//! `opencl_kernels/bls_inj_batch_f32.cl` (verbatim copy — see that file's
//! header comment for the precise origin). Targets <host>'s AMD Vega 10
//! iGPU (Mesa rusticl) and, incidentally, <host>'s own Intel Arc iGPU (Spec
//! 2) — see `docs/plans/2026-08-06-opencl-backend-<host>.md`.
//!
//! Same fused-kernel design as `bls::cuda_batch` (F4b, CUDA sibling): CPU
//! phase-fold in f64 (`phase_fold_cpu`/`counts_per_period`, chunked over
//! periods — `PERIOD_CHUNK_DEFAULT`) feeding a per-(flux, period) f32 kernel
//! that only does the sliding-window search, mean-centering flux in f64
//! before the f32 cast. `search_batch`'s host-side is a line-for-line
//! transliteration of `bls_gpu_batch` (`bls_opencl.py:615-770`).
//!
//! # Backend: `ocl`, GPU-only device selection, no CPU fallback
//!
//! Built on the `ocl` crate (Cargo feature `opencl`, workspace `Cargo.toml`
//! for the exact dependency rationale). Unlike `bls::cuda`/`bls::cuda_batch`
//! (`cudarc`'s `dynamic-loading` mode dlopen()s `libcuda`/`libnvrtc` lazily),
//! `ocl` links against the system OpenCL ICD loader (`libOpenCL.so.1`) at
//! link time — present on <host> (`ocl-icd-libopencl1`, confirmed via
//! `clinfo` during F-opencl session) and expected on <host> (Mesa
//! `ocl-icd`/rusticl stack, per the Python worker's own `mesa-opencl-icd`
//! dependency — `docs/plans/2026-08-06-opencl-python-inventory.md`). No
//! platform/device is ever silently substituted for the GPU: `probe()`
//! returns `None` (after an `eprintln!` naming the specific failure) rather
//! than falling back to a CPU OpenCL device — project rule 2, mirrored
//! from `bls_opencl.py`'s own `_select_gpu_device`/`_init_opencl` posture
//! (see those functions' docstrings for the exact same argument).
//!
//! # Probe-once, no per-call fallback
//!
//! Same contract as `CudaBlsBatchBackend::probe`/`CudaBlsBackend::probe`:
//! `probe()` is called ONCE, at `PrimaryTaskProcessor` construction
//! (`task_processor.rs`'s `resolve_injection_backend`); a probed backend's
//! `search_batch` does NOT re-check the env kill-switch or re-probe on
//! failure — a failure during an actual search is a real `BlsError::Backend`
//! (per-cell recovery, if any, is `injection_recovery.rs`'s job, same as the
//! CUDA batch backend).
//!
//! # Plan deviations (documented, not silent)
//!
//! - **Return type**: the plan's Task 3 "Interfaces" line names a bespoke
//!   `OpenClError`, but also says "MATEIXA signatura i tipus de retorn que
//!   `CudaBlsBatchBackend::search_batch_f32`" — which returns
//!   `Result<Vec<BlsResult>, BlsError>`, not a bespoke error type. This
//!   module follows the CUDA sibling exactly (`BlsError`/`BlsParams`/
//!   `BlsResult` from `bls::reference`, already `pub use`d at `bls::mod`)
//!   rather than introducing a second, redundant error enum — the plan's own
//!   "REUTILITZA les [structs] existents" instruction and its "adapta't al
//!   codi real" escape hatch both point the same way.
//! - **`select_gpu_device` signature**: the plan specifies
//!   `Result<(Platform, Device), OpenClError>`; this module returns
//!   `Option<(Platform, Device)>` instead, matching `bls_opencl.py`'s own
//!   `_select_gpu_device` (returns `None` on exhaustion, not an exception —
//!   see that function's docstring) and this crate's own convention of
//!   "probe failures are `None` + `eprintln!`, not `Result`" used throughout
//!   `bls::cuda`/`bls::cuda_batch`'s `probe()` bodies.
//! - **Device enumeration API**: the plan assumes `ocl::Device::list(platform,
//!   Some(ocl::flags::DeviceType::GPU))` surfaces per-platform errors so a
//!   broken ICD can be logged and skipped. In `ocl` 0.19.7, `Device::list`
//!   swallows `core::get_device_ids`'s `Result` internally
//!   (`.unwrap_or(vec![])`) and therefore can never itself report a broken
//!   platform. This module calls `ocl::core::get_device_ids` directly
//!   (the lower-level function `Device::list` wraps) to get the real
//!   `Result` back, matching PyOpenCL's own `platform.get_devices()`
//!   try/except in `_select_gpu_device` faithfully. Also: `ocl::flags` is
//!   not a real module in this crate version — `DeviceType` lives at the
//!   crate root (`ocl::DeviceType`, re-exported from `ocl::core`).
//! - **`RUSTICL_ENABLE` boot guard placement**: the plan's Task 4 step 3
//!   describes this (main.rs + `probe()`) but Task 4's own file list omits
//!   `opencl_batch.rs`. Implemented here in Task 2 instead (as
//!   `boot_enable_rusticl`, called from `probe()`) since `probe()` cannot be
//!   meaningfully tested/soaked without it; Task 4 wires the second call
//!   site (`main.rs`, "as early as possible", mirroring
//!   `worker.py:46-63`'s import-time placement).

use std::collections::HashMap;
use std::sync::Mutex;

use ocl::{Buffer, Context, Device, DeviceType, Kernel, MemFlags, Platform, Program, Queue};

use super::reference::{self, BlsError, BlsParams, BlsResult};

/// `bls_opencl.py`'s `TG_FLUX` — flux-slots-per-work-group for the fused
/// kernel. See `local_mem_bytes`'s doc comment.
pub const TG_FLUX: usize = 32;

/// `bls_opencl.py`'s `_PERIOD_CHUNK_DEFAULT` — periods processed per kernel
/// launch, bounding peak `bin_idx`/`counts` host RAM.
// Calibrated on <host>'s Vega 10 (2026-08-16): one 5,000-period chunk
// reduced the production-shaped 4,000x90x5,000 workload by about 7% versus
// 1,000-period chunks. Peak host memory remains modest (~46 MiB for the two
// phase/count arrays), and callers can lower it with the env override.
const PERIOD_CHUNK_DEFAULT: usize = 5000;
const PERIOD_CHUNK_MAX: usize = 10_000;

fn period_chunk_from_env(value: Option<&str>) -> usize {
    value
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|&chunk| chunk > 0)
        .map(|chunk| chunk.min(PERIOD_CHUNK_MAX))
        .unwrap_or(PERIOD_CHUNK_DEFAULT)
}

/// OpenCL C source of `bls_inj_batch_f32`, `@NB@`/`@TGSUMS@` template
/// placeholders intact — see `opencl_kernels/bls_inj_batch_f32.cl`'s header
/// comment for the exact origin (`bls_opencl.py` lines 250-343).
const KERNEL_SRC: &str = include_str!("opencl_kernels/bls_inj_batch_f32.cl");
const KERNEL_NAME: &str = "bls_inj_batch_f32";

/// Mirrors `bls_opencl.py`'s `_get_kernel`'s template substitution: `@NB@`
/// -> `n_bins`, `@TGSUMS@` -> `TG_FLUX*(n_bins+1)`.
pub fn kernel_source(n_bins: usize) -> String {
    let tgsums = TG_FLUX * (n_bins + 1);
    KERNEL_SRC
        .replace("@NB@", &n_bins.to_string())
        .replace("@TGSUMS@", &tgsums.to_string())
}

/// Mirrors `bls_opencl.py`'s `_local_mem_bytes(n_bins)` exactly:
/// `tg_cumcounts` (`n_bins+1` floats, shared) + `tg_sums` (`TG_FLUX*
/// (n_bins+1)` floats, one private row per flux slot in the work-group).
/// Identical formula to `bls::cuda_batch::shared_mem_bytes` — same kernel
/// design, same on-chip scratch layout, different backend's name for the
/// same thing (`local` in OpenCL == `shared` in CUDA). Pure arithmetic, no
/// device required — callers needing the actual per-device BUDGET must
/// combine this with the device's measured `CL_DEVICE_LOCAL_MEM_SIZE`
/// (`OpenClBlsBatchBackend::probe`'s `local_mem_size` field).
pub fn local_mem_bytes(n_bins: usize) -> usize {
    TG_FLUX * (n_bins + 1) * 4 + (n_bins + 1) * 4
}

/// Mirrors `opencl_serial.injection_opencl_disabled_by_env`: `EXONIT_INJECTION_OPENCL`
/// `"0"`/`"false"` (case-insensitive, whitespace-tolerant) disables OpenCL
/// injection; unset or any other value, including empty string, does not.
/// Pure function (no env access) so it is directly unit-testable; `probe()`
/// calls it with `std::env::var("EXONIT_INJECTION_OPENCL").ok().as_deref()`.
fn env_disables_opencl(value: Option<&str>) -> bool {
    matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("0") | Some("false")
    )
}

/// Mirrors `opencl_serial.opencl_guard()`: a single process-wide lock shared
/// by every OpenCL submission site in this module (context/program build,
/// buffer upload, kernel launch, buffer read-back) — the same "shared
/// mutable driver state, no lock" failure class as the 2026-07-12 MLX/Metal
/// incident (`opencl_serial.py`'s docstring), applied here even though this
/// crate's primary pool is currently single-threaded per process (defense in
/// depth, matching the Python module's own reasoning: a future FPP-OpenCL
/// caller on another thread must serialize against the SAME lock, not a
/// second independent one). CPU-only work (phase-fold, mean-centering,
/// bincount) stays OUTSIDE every guarded region, same rule as Python's.
///
/// **2026-08-06**: moved to `bls::gpu_guard::gpu_submission_guard()` and
/// shared with `bls::sycl_batch` (both target the SAME physical Arc iGPU
/// through different runtimes — see that module's doc comment for the full
/// rationale). This is a thin re-export at the old call sites' original
/// name so the rest of this file's `opencl_guard()` calls below did not
/// need touching one by one.
use super::gpu_guard::gpu_submission_guard as opencl_guard;

/// Mirrors `worker.py:46-63`'s `_boot_enable_rusticl`: rusticl (Mesa's
/// OpenCL-over-Gallium ICD — <host>'s AMD Vega path) exposes ZERO
/// devices unless `RUSTICL_ENABLE` names the Gallium driver (`radeonsi` for
/// Vega) BEFORE the first platform/device enumeration call. Without this,
/// `select_gpu_device` below silently finds nothing (not even an error to
/// log — rusticl just never appears as a platform) and this backend
/// degrades to CPU with no visible cause. `set_var` only if unset (mirrors
/// Python's `os.environ.setdefault`) so an operator's explicit override
/// always wins. Inert on <host> (Intel's `intel-opencl-icd`, a completely
/// different ICD unaffected by this var). Called from `probe()` here AND,
/// per `main.rs`'s doc comment, at worker process boot — whichever runs
/// first wins; see this module's doc comment ("Plan deviations") for why
/// both call sites live where they do.
pub fn boot_enable_rusticl() {
    if std::env::var_os("RUSTICL_ENABLE").is_none() {
        // SAFETY: called once, very early (this module's `probe()` entry
        // point and/or `main.rs`, before any other thread does OpenCL work),
        // mirroring the Python worker's own single-threaded
        // `os.environ.setdefault` boot call.
        unsafe { std::env::set_var("RUSTICL_ENABLE", "radeonsi") };
    }
}

fn platform_name(platform: &Platform) -> String {
    platform
        .name()
        .unwrap_or_else(|_| "<unknown platform>".to_string())
}

/// Mirrors `bls_opencl.py`'s `_select_gpu_device` (lines 396-447): iterate
/// every platform, return the FIRST device whose type includes `GPU` — NEVER
/// a CPU OpenCL device, and never `create_some_context`'s "whatever the
/// heuristic picks first" ambiguity. Enumeration failures are isolated PER
/// PLATFORM (a broken/partial ICD raising on device enumeration must not
/// hide a working GPU platform listed after it) — only `Platform::list()`
/// itself has no failure mode to isolate (it panics internally on a real
/// `clGetPlatformIDs` failure, same as `bls_opencl.py`'s `cl.get_platforms()`
/// failing before any per-platform loop even starts).
///
/// Queries ALL device types per platform (`DeviceType::ALL`), then filters
/// by the `GPU` bit locally — matching `_select_gpu_device`'s own
/// `platform.get_devices()` (no type filter) + `device.type & cl.device_type.GPU`
/// loop exactly, rather than asking the driver to pre-filter (see this
/// module's doc comment, "Device enumeration API", for why `ocl::Device::list`
/// itself is unusable here).
fn select_gpu_device() -> Option<(Platform, Device)> {
    for platform in Platform::list() {
        let device_ids = match ocl::core::get_device_ids(&platform, Some(DeviceType::ALL), None) {
            Ok(ids) => ids,
            Err(error) => {
                eprintln!(
                    "bls::opencl_batch: platform {:?} failed to enumerate its devices ({error}) -- \
                     skipping this platform and continuing to the next one (a broken/partial ICD \
                     must not hide a working GPU platform listed after it).",
                    platform_name(&platform)
                );
                continue;
            }
        };
        for device in Device::list_from_core(device_ids) {
            let is_gpu = matches!(
                device.info(ocl::core::DeviceInfo::Type),
                Ok(ocl::core::DeviceInfoResult::Type(dtype)) if dtype.contains(DeviceType::GPU)
            );
            if is_gpu {
                return Some((platform, device));
            }
        }
    }
    None
}

fn opencl_err<E: std::fmt::Display>(context: &str) -> impl Fn(E) -> BlsError + '_ {
    move |error| BlsError::Backend(format!("bls::opencl_batch: {context}: {error}"))
}

/// An initialized OpenCL GPU context + command queue + the device's
/// runtime-measured local-memory/work-group limits, built once by
/// [`probe`](Self::probe) and reused for every
/// [`search_batch`](Self::search_batch) call — mirrors `bls_opencl.py`'s
/// module-level `_ctx`/`_queue` globals (built once via `_init_opencl`, not
/// once per call). `programs` caches the compiled `bls_inj_batch_f32`
/// program per `n_bins` (the kernel source is `n_bins`-template-substituted
/// at build time — see `kernel_source` — so, like `bls_cuda_batch`'s
/// `f32_kernels` cache and `bls_opencl.py`'s own `_kernel_cache` dict, it
/// cannot be compiled once up front the way a fixed-size kernel could).
///
/// `Debug` is derived (not hand-written like `CudaBlsBatchBackend`'s):
/// `ocl::Context`/`Queue`/`Device`/`Program` all implement `Debug` natively
/// (unlike `cudarc`'s `CudaFunction`), so there is no missing-impl gap to
/// paper over here.
#[derive(Debug)]
pub struct OpenClBlsBatchBackend {
    context: Context,
    queue: Queue,
    #[allow(dead_code)] // kept for diagnostics/future device_info() accessor; not read yet.
    device: Device,
    /// `CL_DEVICE_LOCAL_MEM_SIZE`, bytes — queried once at probe time (spec
    /// §1 / `bls_opencl.py`'s `_query_local_mem_size`: "mesurar en execucio,
    /// no assumir 32KB/48KB", OpenCL devices vary unlike Metal/CUDA).
    local_mem_size: u64,
    /// `CL_DEVICE_MAX_WORK_GROUP_SIZE` — queried once at probe time
    /// (`bls_opencl.py`'s `_query_max_work_group_size`).
    max_work_group_size: usize,
    programs: Mutex<HashMap<usize, Program>>,
}

impl OpenClBlsBatchBackend {
    /// Probe for a usable OpenCL GPU, build+run a smoke kernel to prove a
    /// REAL submission works (not just context construction — the Vega TDR
    /// lesson `bls_opencl.py:566-594`'s docstring documents), and cache the
    /// device's runtime local-memory/work-group limits. Returns `None`
    /// (after exactly one `eprintln!` naming the specific failure) on any
    /// failure — same posture as `CudaBlsBackend::probe`/
    /// `CudaBlsBatchBackend::probe`. Order mirrors the plan exactly: (1) env
    /// kill-switch, (2) GPU device selection, (3) context+queue, (4) smoke
    /// kernel, (5) local-mem/work-group limits — all under `opencl_guard()`
    /// (mirrors `bls_opencl.py`'s `_init_opencl`/`is_opencl_available`, both
    /// of which run their entire body under `opencl_guard()`).
    pub fn probe() -> Option<Self> {
        if env_disables_opencl(std::env::var("EXONIT_INJECTION_OPENCL").ok().as_deref()) {
            eprintln!(
                "bls::opencl_batch: EXONIT_INJECTION_OPENCL disables OpenCL injection -- \
                 falling back to the CPU injection-recovery path"
            );
            return None;
        }
        boot_enable_rusticl();

        let _guard = opencl_guard()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let Some((platform, device)) = select_gpu_device() else {
            eprintln!(
                "bls::opencl_batch: no GPU OpenCL device available on any platform -- OpenCL \
                 injection backend disabled (never silently substituting a CPU OpenCL device, \
                 CLAUDE.md #2). task_processor.rs's resolve_injection_backend falls through to \
                 CPU via its own explicit path."
            );
            return None;
        };

        let context = match Context::builder()
            .platform(platform)
            .devices(device.clone())
            .build()
        {
            Ok(context) => context,
            Err(error) => {
                eprintln!("bls::opencl_batch: OpenCL context creation failed: {error} -- falling back to the CPU injection-recovery path");
                return None;
            }
        };
        let queue = match Queue::new(&context, device.clone(), None) {
            Ok(queue) => queue,
            Err(error) => {
                eprintln!("bls::opencl_batch: OpenCL command queue creation failed: {error} -- falling back to the CPU injection-recovery path");
                return None;
            }
        };

        // Real submission smoke test -- verbatim of bls_opencl.py's
        // `_bls_opencl_smoke` kernel (is_opencl_available, lines 566-594):
        // compile, launch (global=(1,)), read back, verify == 1.0. A
        // context/queue can construct successfully even when the device
        // can't actually execute a kernel (the Vega TDR failure mode this
        // guards against).
        if let Err(error) = smoke_test(&context, &queue) {
            eprintln!("bls::opencl_batch: OpenCL smoke test failed: {error} -- falling back to the CPU injection-recovery path");
            return None;
        }

        let local_mem_size = match device.info(ocl::core::DeviceInfo::LocalMemSize) {
            Ok(ocl::core::DeviceInfoResult::LocalMemSize(value)) => value,
            Ok(other) => {
                eprintln!("bls::opencl_batch: unexpected CL_DEVICE_LOCAL_MEM_SIZE result shape ({other:?}) -- falling back to the CPU injection-recovery path");
                return None;
            }
            Err(error) => {
                eprintln!("bls::opencl_batch: querying CL_DEVICE_LOCAL_MEM_SIZE failed: {error} -- falling back to the CPU injection-recovery path");
                return None;
            }
        };
        let max_work_group_size = match device.max_wg_size() {
            Ok(value) => value,
            Err(error) => {
                eprintln!("bls::opencl_batch: querying CL_DEVICE_MAX_WORK_GROUP_SIZE failed: {error} -- falling back to the CPU injection-recovery path");
                return None;
            }
        };

        eprintln!(
            "bls::opencl_batch: OpenCL GPU backend ready on {} (CL_DEVICE_LOCAL_MEM_SIZE={local_mem_size} B, \
             CL_DEVICE_MAX_WORK_GROUP_SIZE={max_work_group_size})",
            device.name().unwrap_or_else(|_| "<unnamed device>".to_string())
        );

        Some(OpenClBlsBatchBackend {
            context,
            queue,
            device,
            local_mem_size,
            max_work_group_size,
            programs: Mutex::new(HashMap::new()),
        })
    }

    /// Mirrors `bls_opencl.py`'s `_check_kernel_limits(n_bins)`: an
    /// `Err(BlsError::InvalidParameters(..))` before touching any buffer if
    /// `n_bins` would overflow this device's measured local-memory budget,
    /// or if `TG_FLUX` exceeds its max work-group size. Device-property
    /// comparisons only (already queried once at `probe()` time) — no GPU
    /// submission, safe outside `opencl_guard()`.
    fn check_kernel_limits(&self, n_bins: usize) -> Result<(), BlsError> {
        let needed = local_mem_bytes(n_bins) as u64;
        if needed > self.local_mem_size {
            return Err(BlsError::InvalidParameters(format!(
                "bls::opencl_batch: n_bins={n_bins} needs {needed} B of OpenCL local memory \
                 (TG_FLUX={TG_FLUX}), over this device's measured CL_DEVICE_LOCAL_MEM_SIZE of {} B. \
                 Lower n_bins/injection_n_bins for this call -- the production default (150) needs {} B.",
                self.local_mem_size,
                local_mem_bytes(150)
            )));
        }
        if TG_FLUX > self.max_work_group_size {
            return Err(BlsError::InvalidParameters(format!(
                "bls::opencl_batch: TG_FLUX={TG_FLUX} exceeds this device's measured \
                 CL_DEVICE_MAX_WORK_GROUP_SIZE of {}.",
                self.max_work_group_size
            )));
        }
        Ok(())
    }

    /// Compiles (or fetches from cache) the `bls_inj_batch_f32` program for
    /// a given `n_bins`. Mirrors `bls_opencl.py`'s `_get_kernel`. `Program`
    /// is `Clone` (refcounted, `clRetainProgram`/`clReleaseProgram`
    /// internally) so the cache lock is held only long enough to clone the
    /// entry out, avoiding holding `programs`'s `Mutex` across the
    /// `Kernel::builder()` calls that borrow it in `search_batch`.
    fn get_program(&self, n_bins: usize) -> Result<Program, BlsError> {
        {
            let cache = self
                .programs
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(program) = cache.get(&n_bins) {
                return Ok(program.clone());
            }
        }
        let src = kernel_source(n_bins);
        let program = Program::builder()
            .src(src)
            .build(&self.context)
            .map_err(opencl_err("build bls_inj_batch_f32 program"))?;
        let mut cache = self
            .programs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        cache.insert(n_bins, program.clone());
        Ok(program)
    }

    /// Mirrors `bls_opencl.bls_gpu_batch(time, flux_list, ...)` exactly:
    /// same host-side order of operations (limits check, f64 phase-fold +
    /// mean-centering on CPU, chunked-over-periods kernel launches, f64
    /// post-process), same per-flux result contract as `bls_fast_batch`/
    /// `CudaBlsBatchBackend::search_batch_f32`. `flux_matrix[i].len()` must
    /// equal `time.len()` for every `i` — validated here (same added-safety-
    /// net posture as `CudaBlsBatchBackend::search_batch`, which the Python
    /// function itself relies on its numpy caller to enforce instead).
    pub fn search_batch(
        &self,
        time: &[f64],
        flux_matrix: &[Vec<f64>],
        p: &BlsParams,
    ) -> Result<Vec<BlsResult>, BlsError> {
        if flux_matrix.is_empty() {
            return Ok(Vec::new());
        }
        for flux in flux_matrix {
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
        if !(p.duration_minimum_fraction > 0.0)
            || !(p.duration_maximum_fraction >= p.duration_minimum_fraction)
        {
            return Err(BlsError::InvalidParameters("duration range".to_string()));
        }

        // n_bins/TG_FLUX budget guard -- runtime-measured device limits,
        // checked before any buffer allocation.
        self.check_kernel_limits(p.bin_count)?;

        let n_flux = flux_matrix.len();
        let n_bins = p.bin_count;
        let n_periods = p.period_count;
        let sample_count = time.len();

        let t0_reference = time[0];
        let time_relative: Vec<f64> = time.iter().map(|&v| v - t0_reference).collect();
        // Canonical grid + duration bins + SDE from the same reference
        // helpers `bls::cuda_batch::search_batch_f32` reuses -- single
        // source of truth, mirrors bls_opencl.py's `_frequency_period_grid`
        // import from `bls_fast`.
        let periods =
            reference::frequency_period_grid(p.period_minimum, p.period_maximum, n_periods);
        let duration_bins = reference::duration_bin_grid(p);
        let n_dur = duration_bins.len();
        let widths_i32: Vec<i32> = duration_bins.iter().map(|&v| v as i32).collect();

        // Mean-center in f64 before the f32 cast (depth/sr shift-invariant;
        // centering kills f32 accumulation error -- see bls_opencl.py's
        // rationale, identical here).
        let flux_means: Vec<f64> = flux_matrix
            .iter()
            .map(|flux| flux.iter().sum::<f64>() / sample_count as f64)
            .collect();
        let flux_centered: Vec<Vec<f64>> = flux_matrix
            .iter()
            .zip(&flux_means)
            .map(|(flux, &mean)| flux.iter().map(|&v| v - mean).collect())
            .collect();
        let totals_f32: Vec<f32> = flux_centered
            .iter()
            .map(|flux| flux.iter().sum::<f64>() as f32)
            .collect();

        // Transpose for coalesced kernel reads: fluxT[i*n_flux + fi].
        let mut flux_t_f32 = vec![0f32; sample_count * n_flux];
        for (fi, flux) in flux_centered.iter().enumerate() {
            for (i, &value) in flux.iter().enumerate() {
                flux_t_f32[i * n_flux + fi] = value as f32;
            }
        }

        let n_flux_pad = (n_flux + TG_FLUX - 1) / TG_FLUX * TG_FLUX;
        let program = self.get_program(n_bins)?;

        // Buffers persistent across the whole call (fluxT/totals/widths),
        // created once -- mirrors bls_opencl.py's single upload of these
        // three before the per-chunk loop.
        let (flux_t_buf, totals_buf, widths_buf) = {
            let _guard = opencl_guard()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let flux_t_buf = Buffer::<f32>::builder()
                .queue(self.queue.clone())
                .flags(MemFlags::READ_ONLY | MemFlags::COPY_HOST_PTR)
                .len(flux_t_f32.len())
                .copy_host_slice(&flux_t_f32)
                .build()
                .map_err(opencl_err("upload fluxT"))?;
            let totals_buf = Buffer::<f32>::builder()
                .queue(self.queue.clone())
                .flags(MemFlags::READ_ONLY | MemFlags::COPY_HOST_PTR)
                .len(totals_f32.len())
                .copy_host_slice(&totals_f32)
                .build()
                .map_err(opencl_err("upload totals"))?;
            let widths_buf = Buffer::<i32>::builder()
                .queue(self.queue.clone())
                .flags(MemFlags::READ_ONLY | MemFlags::COPY_HOST_PTR)
                .len(widths_i32.len())
                .copy_host_slice(&widths_i32)
                .build()
                .map_err(opencl_err("upload widths"))?;
            (flux_t_buf, totals_buf, widths_buf)
        };

        // (n_flux, n_periods) row-major accumulators, filled chunk by chunk.
        let mut power_all = vec![0f64; n_flux * n_periods];
        let mut depth_all = vec![0f64; n_flux * n_periods];
        let mut best_w_all = vec![0f64; n_flux * n_periods];
        let mut best_j_all = vec![0f64; n_flux * n_periods];

        let period_chunk =
            period_chunk_from_env(std::env::var("EXONIT_OPENCL_PERIOD_CHUNK").ok().as_deref());
        let mut period_start = 0usize;
        while period_start < n_periods {
            let period_end = (period_start + period_chunk).min(n_periods);
            let n_per = period_end - period_start;

            // CPU-only numpy-equivalent work stays OUTSIDE the guard:
            // `_phase_fold_cpu` (exact-f64 phase-fold) + `_counts_per_period`
            // (exact per-period bin counts).
            let mut bin_idx = vec![0i16; n_per * sample_count];
            for (local_p, &period) in periods[period_start..period_end].iter().enumerate() {
                for (i, &t_rel) in time_relative.iter().enumerate() {
                    let raw = t_rel / period;
                    let phase = raw - raw.floor();
                    let bin = ((phase * n_bins as f64) as i32).clamp(0, n_bins as i32 - 1);
                    bin_idx[local_p * sample_count + i] = bin as i16;
                }
            }
            let mut counts = vec![0f32; n_per * n_bins];
            for local_p in 0..n_per {
                for i in 0..sample_count {
                    let bin = bin_idx[local_p * sample_count + i] as usize;
                    counts[local_p * n_bins + bin] += 1.0;
                }
            }
            let params_i32: [i32; 5] = [
                sample_count as i32,
                n_flux as i32,
                n_bins as i32,
                n_per as i32,
                n_dur as i32,
            ];

            // Per-chunk opencl_guard(): each iteration's buffer upload +
            // kernel launch + read-back is fully self-contained and
            // synchronously finished (`queue.finish()`) before the lock is
            // released -- mirrors bls_opencl.py's per-chunk `with
            // opencl_guard():` block exactly (lets a concurrent FPP-OpenCL
            // call interleave between chunks instead of blocking for the
            // whole batch).
            let (out_power, out_wj, out_depth) = {
                let _guard = opencl_guard()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());

                let bin_idx_buf = Buffer::<i16>::builder()
                    .queue(self.queue.clone())
                    .flags(MemFlags::READ_ONLY | MemFlags::COPY_HOST_PTR)
                    .len(bin_idx.len())
                    .copy_host_slice(&bin_idx)
                    .build()
                    .map_err(opencl_err("upload bin_idx"))?;
                let counts_buf = Buffer::<f32>::builder()
                    .queue(self.queue.clone())
                    .flags(MemFlags::READ_ONLY | MemFlags::COPY_HOST_PTR)
                    .len(counts.len())
                    .copy_host_slice(&counts)
                    .build()
                    .map_err(opencl_err("upload counts"))?;
                let params_buf = Buffer::<i32>::builder()
                    .queue(self.queue.clone())
                    .flags(MemFlags::READ_ONLY | MemFlags::COPY_HOST_PTR)
                    .len(params_i32.len())
                    .copy_host_slice(&params_i32)
                    .build()
                    .map_err(opencl_err("upload params"))?;
                let out_power_buf = Buffer::<f32>::builder()
                    .queue(self.queue.clone())
                    .flags(MemFlags::WRITE_ONLY)
                    .len(n_flux * n_per)
                    .build()
                    .map_err(opencl_err("alloc out_power"))?;
                let out_wj_buf = Buffer::<u32>::builder()
                    .queue(self.queue.clone())
                    .flags(MemFlags::WRITE_ONLY)
                    .len(n_flux * n_per)
                    .build()
                    .map_err(opencl_err("alloc out_wj"))?;
                let out_depth_buf = Buffer::<f32>::builder()
                    .queue(self.queue.clone())
                    .flags(MemFlags::WRITE_ONLY)
                    .len(n_flux * n_per)
                    .build()
                    .map_err(opencl_err("alloc out_depth"))?;

                let kernel = Kernel::builder()
                    .program(&program)
                    .name(KERNEL_NAME)
                    .queue(self.queue.clone())
                    .global_work_size((n_flux_pad, n_per))
                    .local_work_size((TG_FLUX, 1))
                    .arg(&bin_idx_buf)
                    .arg(&counts_buf)
                    .arg(&flux_t_buf)
                    .arg(&totals_buf)
                    .arg(&widths_buf)
                    .arg(&params_buf)
                    .arg(&out_power_buf)
                    .arg(&out_wj_buf)
                    .arg(&out_depth_buf)
                    .build()
                    .map_err(opencl_err("build bls_inj_batch_f32 kernel"))?;

                // SAFETY: kernel argument types/order/count match
                // `bls_inj_batch_f32`'s OpenCL C signature exactly (verbatim
                // kernel, template-substituted the same way as the Python
                // source -- see `opencl_kernels/bls_inj_batch_f32.cl`);
                // `n_flux_pad` padding work-items (`f >= n_flux`) return
                // before writing any output, so the `n_flux*n_per`-sized
                // output buffers are never written out of bounds.
                unsafe { kernel.enq() }.map_err(opencl_err("launch bls_inj_batch_f32"))?;

                let mut out_power = vec![0f32; n_flux * n_per];
                let mut out_wj = vec![0u32; n_flux * n_per];
                let mut out_depth = vec![0f32; n_flux * n_per];
                out_power_buf
                    .read(&mut out_power)
                    .enq()
                    .map_err(opencl_err("download out_power"))?;
                out_wj_buf
                    .read(&mut out_wj)
                    .enq()
                    .map_err(opencl_err("download out_wj"))?;
                out_depth_buf
                    .read(&mut out_depth)
                    .enq()
                    .map_err(opencl_err("download out_depth"))?;
                self.queue.finish().map_err(opencl_err("queue finish"))?;

                (out_power, out_wj, out_depth)
            };

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
            // `np.argmax` -- first occurrence of the maximum.
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

/// Real submission smoke test -- verbatim of `bls_opencl.py`'s
/// `is_opencl_available` kernel body (lines 566-594): compile a minimal
/// kernel, launch it with a single work-item, read back the result, verify
/// it matches. Proves the device can actually EXECUTE OpenCL, not just that
/// a context/queue could be constructed (the exact gap that let Vega's old
/// per-work-item kernel design pass a naive "does it import" check while
/// still TDR-crashing under real load — see `opencl_kernels/
/// bls_inj_batch_f32.cl`'s module-adjacent history in `bls_opencl.py`'s
/// docstring). Must be called from inside `opencl_guard()` (a program
/// build + kernel launch + buffer read-back are all real GPU submissions).
fn smoke_test(context: &Context, queue: &Queue) -> Result<(), BlsError> {
    let program = Program::builder()
        .src("__kernel void _bls_opencl_smoke(__global float* o) { o[0] = 1.0f; }")
        .build(context)
        .map_err(opencl_err("build smoke program"))?;
    let buffer = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(MemFlags::WRITE_ONLY)
        .len(1)
        .build()
        .map_err(opencl_err("alloc smoke buffer"))?;
    let kernel = Kernel::builder()
        .program(&program)
        .name("_bls_opencl_smoke")
        .queue(queue.clone())
        .global_work_size(1usize)
        .arg(&buffer)
        .build()
        .map_err(opencl_err("build smoke kernel"))?;
    // SAFETY: the smoke kernel writes exactly one float into a
    // one-element buffer sized for it; no other memory is touched.
    unsafe { kernel.enq() }.map_err(opencl_err("launch smoke kernel"))?;
    let mut out = [0f32; 1];
    buffer
        .read(&mut out[..])
        .enq()
        .map_err(opencl_err("read back smoke buffer"))?;
    queue.finish().map_err(opencl_err("finish smoke queue"))?;
    if out[0] == 1.0 {
        Ok(())
    } else {
        Err(BlsError::Backend(format!(
            "bls::opencl_batch: smoke kernel returned {}, expected 1.0",
            out[0]
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_source_substitutes_placeholders() {
        let src = kernel_source(150);
        assert!(!src.contains("@NB@"));
        assert!(!src.contains("@TGSUMS@"));
        assert!(src.contains("__kernel void bls_inj_batch_f32"));
        // TG_FLUX * (n_bins+1) = 32 * 151 = 4832
        assert!(src.contains("4832"));
    }

    #[test]
    fn local_mem_budget_matches_python_formula() {
        // Verbatim values from bls_opencl.py's `_local_mem_bytes` / the same
        // formula bls_cuda.py's `_shared_mem_bytes` uses: TG_FLUX*(n_bins+1)*4
        // + (n_bins+1)*4, TG_FLUX=32.
        assert_eq!(local_mem_bytes(150), 32 * 151 * 4 + 151 * 4);
        assert_eq!(local_mem_bytes(200), 32 * 201 * 4 + 201 * 4);
    }

    /// Mirrors `opencl_serial.injection_opencl_disabled_by_env`'s contract.
    #[test]
    fn injection_opencl_env_kill_switch_contract() {
        assert!(env_disables_opencl(Some("0")));
        assert!(env_disables_opencl(Some("false")));
        assert!(env_disables_opencl(Some("FALSE")));
        assert!(env_disables_opencl(Some(" False ")));
        assert!(!env_disables_opencl(Some("1")));
        assert!(!env_disables_opencl(Some("")));
        assert!(!env_disables_opencl(None));
    }

    #[test]
    fn period_chunk_override_is_bounded_and_rejects_invalid_values() {
        assert_eq!(period_chunk_from_env(None), PERIOD_CHUNK_DEFAULT);
        assert_eq!(period_chunk_from_env(Some(" 2000 ")), 2000);
        assert_eq!(period_chunk_from_env(Some("0")), PERIOD_CHUNK_DEFAULT);
        assert_eq!(period_chunk_from_env(Some("invalid")), PERIOD_CHUNK_DEFAULT);
        assert_eq!(period_chunk_from_env(Some("20000")), PERIOD_CHUNK_MAX);
    }

    /// Sanity check on the real Arc GPU host (<host>): `probe()` must
    /// succeed and report plausible device limits. Run explicitly:
    /// `cargo test --features opencl --test '*' -- --ignored --nocapture`
    /// (there is no separate golden-test file for this — the probe itself
    /// lives in this module's own `#[cfg(test)]` block since it takes no
    /// synthetic workload). Negligible GPU load (a single 1-work-item smoke
    /// kernel) -- safe to run alongside the Python worker's own OpenCL
    /// instance in production.
    #[test]
    #[ignore = "needs a real OpenCL GPU device"]
    fn opencl_probe_smoke_on_gpu_host() {
        match OpenClBlsBatchBackend::probe() {
            Some(backend) => {
                eprintln!(
                    "opencl_probe_smoke_on_gpu_host: probe() succeeded -- device={:?} \
                     local_mem_size={} max_work_group_size={}",
                    backend.device.name(),
                    backend.local_mem_size,
                    backend.max_work_group_size
                );
                assert!(backend.local_mem_size > 0);
                assert!(backend.max_work_group_size >= TG_FLUX);
            }
            None => panic!("OpenClBlsBatchBackend::probe() returned None on a host expected to have a working OpenCL GPU (see clinfo)"),
        }
    }
}
