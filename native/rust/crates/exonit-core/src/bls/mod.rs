//! BLS (Box Least Squares) — mirrors `BLSReference.swift`,
//! `BLSCPUOptimized.swift`, `BLSTwoPass.swift`, and the CPU path of
//! `BLSMetal.swift` (`BLSAccelerated`/`BLSComputeBackend`). GPU backends are
//! F4 scope; see `accelerated.rs`'s module doc comment.

pub mod accelerated;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod cuda_batch;
// 2026-08-06 Spec 2 integration: shared GPU-submission lock for
// opencl_batch/sycl_batch — see gpu_guard.rs's module doc comment.
#[cfg(any(feature = "opencl", feature = "sycl"))]
pub(crate) mod gpu_guard;
// 2026-08-06 Spec 2 integration: ISPC-accelerated CPU fold/bin/window
// search, called from optimized::search_in_pool — see ispc_fold.rs's
// module doc comment.
#[cfg(feature = "ispc")]
pub(crate) mod ispc_fold;
#[cfg(feature = "opencl")]
pub mod opencl_batch;
pub mod optimized;
pub mod quick;
pub mod reference;
// 2026-08-06 Spec 2 integration: SYCL injection-recovery batch backend —
// see sycl_batch.rs's module doc comment.
#[cfg(feature = "sycl")]
pub mod sycl_batch;
pub mod two_pass;

pub use reference::{BlsError, BlsParams, BlsResult};

/// F7: runs `f` inside a rayon [`rayon::ThreadPool`] sized to
/// `max_concurrency` threads, shared by `optimized::search` and
/// `two_pass::search` (see those modules' doc comments for exactly how the
/// per-period parallel loop stays bit-for-bit deterministic).
///
/// - `max_concurrency == 0` means "no explicit cap" and runs `f` directly —
///   this uses whichever rayon context is already active (a pool a caller
///   further up the stack already installed) or, absent one, rayon's own
///   process-global default pool (`RAYON_NUM_THREADS` env var, else
///   `std::thread::available_parallelism()`). This mirrors the task's
///   instruction verbatim ("with 0 or absent, rayon's default") — in
///   practice every real caller in this crate resolves `max_concurrency`
///   from `config.cpu_threads`/`available_parallelism()` before reaching
///   here (see `runtime.rs`'s `WorkerRuntime::build`), so `0` is a
///   test/library-API edge case, not the production path.
/// - `max_concurrency > 0` builds a **fresh thread pool for this call**
///   (not `rayon::ThreadPoolBuilder::build_global()`, which is a
///   process-wide singleton that can only be configured once — that would
///   make it impossible for a single process to honor two different
///   `cpu_threads` values, e.g. one golden test exercising both a 1-thread
///   and a 4-thread run of the same search for the determinism gate) and
///   runs `f` inside it via [`rayon::ThreadPool::install`]. This is what
///   makes `max_concurrency` — which mirrors `config.cpu_threads`, itself
///   mirroring the Python worker's own CPU-thread cap — a real per-call
///   ceiling rather than advisory: project rule 2 requires the host
///   never see all its cores claimed by one process, and the *value* is the
///   caller's config, not a default this function should second-guess.
/// - If the OS refuses to hand out the requested threads (`build()` fails),
///   falls back to running `f` uncapped rather than propagating the error —
///   a worker mid-task should degrade, not crash, on a resource hiccup that
///   has nothing to do with the science being computed.
pub(crate) fn with_pool<T: Send>(max_concurrency: usize, f: impl FnOnce() -> T + Send) -> T {
    if max_concurrency == 0 {
        return f();
    }
    match rayon::ThreadPoolBuilder::new().num_threads(max_concurrency).build() {
        Ok(pool) => pool.install(f),
        Err(_) => f(),
    }
}

/// Mirrors the Swift `BLSSearchFunction` typealias
/// (`@Sendable (time:flux:parameters:) throws -> BLSResult`), used by
/// `harmonic_resolver`, `multi_planet` and `long_period_sweep` to receive an
/// injectable BLS search step (production wires this to
/// `accelerated::search` with the configured backend; golden tests wire it
/// directly to `reference::search` or `two_pass::search` to match the exact
/// Swift contract-test call site being ported).
pub type BlsSearchFn<'a> = &'a dyn Fn(&[f64], &[f64], &BlsParams) -> Result<BlsResult, BlsError>;
