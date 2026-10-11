//! Transliterated from `Sources/ExoNitCore/InjectionRecovery.swift`.
//!
//! Orchestrates inject -> BLS -> match: for each (period, radius,
//! injection-in-cell) triple it injects a synthetic transit (`injection.rs`)
//! into the (already-detrended, caller-supplied) flux, runs the *same* CPU
//! BLS search the Swift source calls (`BLSCPUOptimized.search`, i.e. this
//! crate's `bls::optimized::search` — deliberately not the two-pass/
//! accelerated backend used by the primary scan pipeline), and counts a
//! recovery when the returned period is within 5% of the injected period
//! and the SNR clears `minimum_snr`.
//!
//! Swift's `InjectionRecovery.run` also accepts a `BLSComputeBackend`
//! (`.cpu`/`.metal`/`.automatic`) and dispatches to a Metal batch kernel
//! when available. F4b adds the CUDA analogue: [`InjectionBackend::Cuda`]
//! routes each period-cell's whole `injected_fluxes` group through
//! `bls::cuda_batch::CudaBlsBatchBackend::search_batch` in ONE call instead
//! of looping `bls::optimized::search` per flux — the same "group by period,
//! share the BLS range" batching `_run_batched_gpu` in
//! `~/exonit/worker/injection.py` does. CPU (`InjectionBackend::Cpu`,
//! `run`'s default) is unchanged: the per-flux `bls::optimized::search` loop
//! this module has had since F5a, kept as-is per this task's instructions.

use crate::bls::optimized;
use crate::bls::reference::{BlsError, BlsParams, BlsResult};
use crate::dto::{CompletenessCell, CompletenessResult};
use crate::injection;
use crate::transit_physics;
use std::time::Instant;

/// Selects which implementation [`run_with_backend`] uses for each
/// period-cell's BLS searches. `Cpu` (the type `run` uses) is the per-flux
/// `bls::optimized::search` loop this module has always used; `Cuda`
/// (feature `cuda`, F4b) wraps an already-`probe()`d
/// `bls::cuda_batch::CudaBlsBatchBackend`, resolved ONCE by
/// `task_processor.rs`'s `resolve_injection_backend` — same "probed once,
/// not per call" posture as `bls::accelerated::ComputeBackend`/
/// `detrend::DetrendBackend`. Selection semantics mirror
/// `injection.py`'s auto-detect chain gated by `worker.py`'s
/// `disable_gpu()` call (`config.gpu_enabled=false` -> CPU, unconditionally):
/// this crate has no Metal/OpenCL injection backends, so `Cuda` is the only
/// non-`Cpu` case, structurally equivalent to Python's `_gpu_backend ==
/// "CUDA"` outcome on an <host>/<host>-class host.
#[derive(Clone)]
pub enum InjectionBackend {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(std::sync::Arc<crate::bls::cuda_batch::CudaBlsBatchBackend>),
    /// F-opencl (`docs/plans/2026-08-06-opencl-backend-<host>.md`, Task 4):
    /// <host> (AMD Vega, Mesa rusticl) / <host> (Intel Arc) batch
    /// backend, resolved ONCE by `task_processor.rs`'s
    /// `resolve_injection_backend` — same "probed once" posture as `Cuda`
    /// above, but additionally routed through a process-wide circuit
    /// breaker (`opencl_circuit_breaker` below) that `Cuda` deliberately
    /// does not have (F4b left that as a documented gap; here it is real
    /// because the Vega TDR risk that motivated this whole port is exactly
    /// the failure mode the breaker exists to contain).
    #[cfg(feature = "opencl")]
    OpenCl(std::sync::Arc<crate::bls::opencl_batch::OpenClBlsBatchBackend>),
    /// 2026-08-06 Spec 2 integration (<host>): SYCL batch backend
    /// (`bls::sycl_batch::SyclBlsBatchBackend`), promoted from Spike 1
    /// after `spikes/RESULTS_2026-08-06.md`'s real-measurement gate.
    /// Resolved ONCE by `task_processor.rs`'s `resolve_injection_backend`
    /// — same "probed once" posture as `OpenCl`/`Cuda` above — and routed
    /// through its OWN circuit breaker (`sycl_circuit_breaker` below),
    /// mirroring `OpenCl`'s exactly (same GPU-batch-failure-class
    /// reasoning, same threshold, independent counter — a run of SYCL
    /// failures does not trip the OpenCl breaker or vice versa, since only
    /// one of the two is ever actually resolved per process in practice —
    /// see `resolve_injection_backend`'s CUDA > SYCL > OpenCL > CPU
    /// priority chain).
    #[cfg(feature = "sycl")]
    Sycl(std::sync::Arc<crate::bls::sycl_batch::SyclBlsBatchBackend>),
}

impl std::fmt::Debug for InjectionBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InjectionBackend::Cpu => f.write_str("Cpu"),
            #[cfg(feature = "cuda")]
            InjectionBackend::Cuda(_) => f.write_str("Cuda(..)"),
            #[cfg(feature = "opencl")]
            InjectionBackend::OpenCl(_) => f.write_str("OpenCl(..)"),
            #[cfg(feature = "sycl")]
            InjectionBackend::Sycl(_) => f.write_str("Sycl(..)"),
        }
    }
}

/// Process-wide GPU-batch circuit breaker for the OpenCL injection-recovery
/// path — mirrors `injection.py`'s `_record_gpu_batch_success`/
/// `_record_gpu_batch_failure`/`_gpu_circuit_open` (lines 210-270) exactly:
/// a Metal-compiler-quirk-class failure (structurally broken kernel on a
/// given host, not a transient glitch) would otherwise retry-and-warn on
/// every single period-cell of every remaining star in a 100k-star run.
/// After `GPU_FAILURE_THRESHOLD` CONSECUTIVE failures, log ONE `eprintln!`
/// and permanently route every remaining call in this process straight to
/// the CPU batch path — no further per-cell retries or warnings. A single
/// success resets the streak (a backend that mostly works, with the odd
/// transient glitch, must not trip the breaker).
///
/// **Deviation from the plan**: the plan's Task 4 Step 2 says the threshold
/// is overridable via env `EXONIT_INJECTION_GPU_FAILURE_THRESHOLD` — but
/// `injection.py` only ever *mentions* that env var name in comments
/// (lines 217, 367); grepping the file confirms it is never actually
/// `os.environ.get`-read anywhere, so `_GPU_FAILURE_THRESHOLD = 3` is a
/// hardcoded constant in real Python behaviour. Per this crate's
/// mirror-fidelity rule ("si el Python fa una cosa rara, el Rust la fa
/// igual"), this port mirrors the REAL (hardcoded, unoverridable) behaviour,
/// not the plan's assumption of a working override — see
/// `bls/opencl_batch.rs`'s module doc comment for the sibling note on that
/// file's own plan-vs-real-code adaptations.
#[cfg(feature = "opencl")]
mod opencl_circuit_breaker {
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    const GPU_FAILURE_THRESHOLD: u32 = 3;

    static CONSECUTIVE_FAILURES: AtomicU32 = AtomicU32::new(0);
    static CIRCUIT_OPEN: AtomicBool = AtomicBool::new(false);

    /// True once the breaker has tripped — mirrors `gpu_circuit_is_open()`/
    /// `_gpu_circuit_open`: process-wide state, not per-call or
    /// per-`OpenClBlsBatchBackend`-instance (matches Python's module-level
    /// global exactly).
    pub fn is_open() -> bool {
        CIRCUIT_OPEN.load(Ordering::SeqCst)
    }

    /// Mirrors `_record_gpu_batch_success`.
    pub fn record_success() {
        CONSECUTIVE_FAILURES.store(0, Ordering::SeqCst);
    }

    /// Mirrors `_record_gpu_batch_failure`: increments the streak, trips the
    /// breaker (logs exactly once, at the trip point — not on every failure
    /// after) on the Nth consecutive failure.
    pub fn record_failure(backend_label: &str) {
        let streak = CONSECUTIVE_FAILURES.fetch_add(1, Ordering::SeqCst) + 1;
        if streak >= GPU_FAILURE_THRESHOLD && !CIRCUIT_OPEN.swap(true, Ordering::SeqCst) {
            eprintln!(
                "injection_recovery: {backend_label} injection GPU backend failed {streak} \
                 consecutive times -- disabling it for the remainder of this process; every \
                 further injection-recovery call will use the CPU batch path (bls::optimized) \
                 directly, with no further per-cell retries or warnings for this backend."
            );
        }
    }

    #[cfg(test)]
    pub fn reset_for_test() {
        CONSECUTIVE_FAILURES.store(0, Ordering::SeqCst);
        CIRCUIT_OPEN.store(false, Ordering::SeqCst);
    }
}

/// 2026-08-06 Spec 2 integration: SYCL's own circuit breaker, mirroring
/// `opencl_circuit_breaker` above field-for-field and function-for-
/// function (same GPU-batch-failure-class reasoning — see that module's
/// doc comment, which applies unchanged here) with an INDEPENDENT counter
/// — SYCL failures never trip the OpenCL breaker or vice versa. Kept as a
/// verbatim-structure sibling module rather than a single generic/
/// parameterized breaker shared by both backends: `injection.py` has no
/// SYCL concept to mirror at all (this crate's SYCL backend has no Python
/// counterpart, unlike OpenCL's `opencl_serial.py`/`bls_opencl.py`), so
/// there is no "real Python behaviour" this needs to match beyond
/// "behave exactly like the OpenCL breaker" — duplicating the already-
/// battle-tested structure is lower-risk than introducing a new generic
/// abstraction for a two-instance case.
#[cfg(feature = "sycl")]
mod sycl_circuit_breaker {
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    const GPU_FAILURE_THRESHOLD: u32 = 3;

    static CONSECUTIVE_FAILURES: AtomicU32 = AtomicU32::new(0);
    static CIRCUIT_OPEN: AtomicBool = AtomicBool::new(false);

    pub fn is_open() -> bool {
        CIRCUIT_OPEN.load(Ordering::SeqCst)
    }

    pub fn record_success() {
        CONSECUTIVE_FAILURES.store(0, Ordering::SeqCst);
    }

    pub fn record_failure(backend_label: &str) {
        let streak = CONSECUTIVE_FAILURES.fetch_add(1, Ordering::SeqCst) + 1;
        if streak >= GPU_FAILURE_THRESHOLD && !CIRCUIT_OPEN.swap(true, Ordering::SeqCst) {
            eprintln!(
                "injection_recovery: {backend_label} injection GPU backend failed {streak} \
                 consecutive times -- disabling it for the remainder of this process; every \
                 further injection-recovery call will use the CPU batch path (bls::optimized) \
                 directly, with no further per-cell retries or warnings for this backend."
            );
        }
    }

    #[cfg(test)]
    pub fn reset_for_test() {
        CONSECUTIVE_FAILURES.store(0, Ordering::SeqCst);
        CIRCUIT_OPEN.store(false, Ordering::SeqCst);
    }
}

/// Mirrors `InjectionRecoveryOptions` (`Sources/ExoNitCore/InjectionRecovery.swift`).
/// Rust has no default parameter values, so `Default::default()` reproduces
/// the Swift memberwise-init defaults; callers override only what they need,
/// as the Swift call sites do (`var options = InjectionRecoveryOptions(); options.foo = ...`).
#[derive(Debug, Clone, PartialEq)]
pub struct InjectionRecoveryOptions {
    pub stellar_radius_solar: f64,
    pub stellar_mass_solar: Option<f64>,
    pub radius_grid_earth: Vec<f64>,
    pub period_grid_days: Option<Vec<f64>>,
    pub injections_per_cell: usize,
    pub minimum_snr: f64,
    pub period_count: usize,
    pub bin_count: usize,
    pub duration_step_count: usize,
    pub maximum_seconds: Option<f64>,
}

impl Default for InjectionRecoveryOptions {
    fn default() -> Self {
        InjectionRecoveryOptions {
            stellar_radius_solar: 1.0,
            stellar_mass_solar: None,
            radius_grid_earth: vec![0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0],
            period_grid_days: None,
            injections_per_cell: 10,
            minimum_snr: 7.0,
            period_count: 5_000,
            bin_count: 150,
            duration_step_count: 10,
            maximum_seconds: None,
        }
    }
}

/// Mirrors `InjectionRecovery.run(time:flux:options:maxConcurrency:backend:isCancelled:)`,
/// restricted to the CPU backend (see module doc comment). `max_concurrency`
/// is forwarded to `bls::optimized::search`, which currently ignores it
/// (F2 is single-threaded per `docs/PLAN.md`) — kept for call-site parity
/// with the Swift signature, same as `bls::optimized::search` itself.
pub fn run(
    time: &[f64],
    flux: &[f64],
    options: &InjectionRecoveryOptions,
    max_concurrency: usize,
    is_cancelled: impl Fn() -> bool,
) -> Result<CompletenessResult, BlsError> {
    run_with_backend(time, flux, options, max_concurrency, is_cancelled, &InjectionBackend::Cpu)
}

/// `run` restricted to a caller-selected [`InjectionBackend`] (F4b) — same
/// body as `run`, with the per-period-cell BLS search routed through
/// `backend` instead of always looping `bls::optimized::search`. See
/// [`InjectionBackend`]'s doc comment for the CUDA-batch semantics.
pub fn run_with_backend(
    time: &[f64],
    flux: &[f64],
    options: &InjectionRecoveryOptions,
    max_concurrency: usize,
    is_cancelled: impl Fn() -> bool,
    backend: &InjectionBackend,
) -> Result<CompletenessResult, BlsError> {
    // Mirrors `guard time.count == flux.count, time.count >= 6 else { throw
    // BLSError.insufficientSamples }` — both failure causes map to the same
    // error in the Swift source, so this does too.
    if time.len() != flux.len() || time.len() < 6 {
        return Err(BlsError::InsufficientSamples);
    }
    let span = time[time.len() - 1] - time[0];
    const DEFAULT_PERIOD_GRID: [f64; 8] = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0];
    let periods: Vec<f64> = options
        .period_grid_days
        .clone()
        .unwrap_or_else(|| DEFAULT_PERIOD_GRID.to_vec())
        .into_iter()
        .filter(|&period| period < span / 2.0)
        .collect();
    let radii = &options.radius_grid_earth;

    let mut counts: Vec<Vec<i64>> = vec![vec![0i64; periods.len()]; radii.len()];
    let mut evaluated_periods: i64 = 0;
    let mut truncated = false;
    let started = Instant::now();

    for (period_index, &period) in periods.iter().enumerate() {
        if is_cancelled() || exceeded(started, options.maximum_seconds) {
            truncated = true;
            break;
        }
        let minimum = f64::max(0.5, period * 0.5);
        let maximum = f64::min(period * 2.0, span);
        let parameters = BlsParams {
            period_minimum: minimum,
            period_maximum: maximum,
            period_count: options.period_count,
            bin_count: options.bin_count,
            duration_step_count: options.duration_step_count,
            ..BlsParams::default()
        };

        let mut injected_fluxes: Vec<Vec<f64>> = Vec::with_capacity(radii.len() * options.injections_per_cell);
        // (radius_index, injected_period) per injected_fluxes entry, same
        // order as Swift's `locations` array.
        let mut locations: Vec<(usize, f64)> = Vec::with_capacity(radii.len() * options.injections_per_cell);

        for (radius_index, &radius) in radii.iter().enumerate() {
            let depth = injection::depth(radius, options.stellar_radius_solar);
            let raw_duration = transit_physics::duration_days(
                period,
                options.stellar_radius_solar,
                options.stellar_mass_solar,
                0.0,
                radius,
            );
            let duration = f64::max(0.02, f64::min(raw_duration, 1.0));

            for injection_index in 0..options.injections_per_cell {
                let seed = radius_index * periods.len() * options.injections_per_cell
                    + period_index * options.injections_per_cell
                    + injection_index;
                let mut random = injection::NumpyMt19937::new(seed as u32);
                let t0 = time[0] + random.uniform01() * period;
                injected_fluxes.push(injection::inject(time, flux, period, t0, depth, duration));
                locations.push((radius_index, period));
            }
        }

        let cpu_fallback = || -> Result<Vec<BlsResult>, BlsError> {
            optimized::search_batch(time, &injected_fluxes, &parameters, max_concurrency)
        };
        let results: Vec<BlsResult> = match backend {
            InjectionBackend::Cpu => cpu_fallback()?,
            #[cfg(feature = "cuda")]
            InjectionBackend::Cuda(cuda) => match cuda.search_batch(time, &injected_fluxes, &parameters) {
                Ok(results) => results,
                Err(error) => {
                    // Never silent, one retry on THIS cell only — mirrors
                    // `_run_batched_gpu`'s per-cell try/except (GPU command-
                    // buffer/context glitches can fail transiently without
                    // the host being unhealthy). Python retries on
                    // `bls_fast_batch` (a genuinely batched CPU kernel);
                    // this port retries on the per-flux `optimized::search`
                    // loop instead, since that is this crate's only CPU
                    // implementation (no batched-CPU BLS ported yet) — same
                    // recovery *outcome*, different CPU code path.
                    eprintln!(
                        "injection_recovery: CUDA batch BLS failed on period-cell pi={period_index}/{} \
                         (period={period:.4}d): {error} -> retrying this cell on CPU (bls::optimized)",
                        periods.len()
                    );
                    cpu_fallback()?
                }
            },
            // Mirrors `injection.py`'s call site (lines 367-399) exactly:
            // if the circuit breaker is already open, skip the GPU attempt
            // entirely (straight to CPU, no per-cell log — the breaker
            // already logged once at trip time); otherwise try OpenCL,
            // record success/failure, and on failure retry THIS cell on CPU
            // (logged) same as the CUDA arm above.
            #[cfg(feature = "opencl")]
            InjectionBackend::OpenCl(opencl) => {
                if opencl_circuit_breaker::is_open() {
                    cpu_fallback()?
                } else {
                    match opencl.search_batch(time, &injected_fluxes, &parameters) {
                        Ok(results) => {
                            opencl_circuit_breaker::record_success();
                            results
                        }
                        Err(error) => {
                            opencl_circuit_breaker::record_failure("OpenCL");
                            eprintln!(
                                "injection_recovery: OpenCL batch BLS failed on period-cell pi={period_index}/{} \
                                 (period={period:.4}d): {error} -> retrying this cell on CPU (bls::optimized)",
                                periods.len()
                            );
                            cpu_fallback()?
                        }
                    }
                }
            }
            // 2026-08-06 Spec 2 integration: mirrors the OpenCL arm above
            // exactly (own circuit breaker, own log label) -- see
            // `sycl_circuit_breaker`'s module doc comment for why this is
            // a duplicated-structure sibling rather than a shared generic
            // breaker.
            #[cfg(feature = "sycl")]
            InjectionBackend::Sycl(sycl) => {
                if sycl_circuit_breaker::is_open() {
                    cpu_fallback()?
                } else {
                    match sycl.search_batch(time, &injected_fluxes, &parameters) {
                        Ok(results) => {
                            sycl_circuit_breaker::record_success();
                            results
                        }
                        Err(error) => {
                            sycl_circuit_breaker::record_failure("SYCL");
                            eprintln!(
                                "injection_recovery: SYCL batch BLS failed on period-cell pi={period_index}/{} \
                                 (period={period:.4}d): {error} -> retrying this cell on CPU (bls::optimized)",
                                periods.len()
                            );
                            cpu_fallback()?
                        }
                    }
                }
            }
        };

        for (location, result) in locations.iter().zip(results.iter()) {
            let (radius_index, injected_period) = *location;
            if (result.period - injected_period).abs() / injected_period < 0.05 && result.snr >= options.minimum_snr {
                counts[radius_index][period_index] += 1;
            }
        }
        evaluated_periods += 1;
    }

    let denominator = options.injections_per_cell.max(1) as f64;
    let completeness: Vec<Vec<f64>> = counts
        .iter()
        .map(|row| row.iter().map(|&count| count as f64 / denominator).collect())
        .collect();

    let mut details: Vec<CompletenessCell> = Vec::with_capacity(radii.len() * periods.len());
    for (radius_index, &radius) in radii.iter().enumerate() {
        for (period_index, &period) in periods.iter().enumerate() {
            details.push(CompletenessCell {
                radius_earth: radius,
                period_days: period,
                depth_ppm: Some(injection::depth(radius, options.stellar_radius_solar) * 1_000_000.0),
                detection_probability: completeness[radius_index][period_index],
                recovered_count: Some(counts[radius_index][period_index]),
                injected_count: Some(options.injections_per_cell as i64),
            });
        }
    }

    Ok(CompletenessResult {
        radius_grid: Some(radii.clone()),
        period_grid: Some(periods.clone()),
        completeness_map: Some(completeness),
        details,
        stellar_radius: Some(options.stellar_radius_solar),
        truncated: Some(truncated),
        periods_evaluated: Some(evaluated_periods),
        radii_evaluated: Some(radii.len() as i64),
    })
}

/// Mirrors `InjectionRecovery.exceeded(started:seconds:)`.
fn exceeded(started: Instant, seconds: Option<f64>) -> bool {
    match seconds {
        None => false,
        Some(seconds) => started.elapsed().as_secs_f64() >= seconds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insufficient_samples_is_rejected() {
        let time = [0.0, 1.0, 2.0];
        let flux = [1.0, 1.0, 1.0];
        let result = run(&time, &flux, &InjectionRecoveryOptions::default(), 1, || false);
        assert_eq!(result, Err(BlsError::InsufficientSamples));
    }

    #[test]
    fn cancellation_before_first_period_truncates_with_zero_periods_evaluated() {
        let time: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let flux = vec![1.0; time.len()];
        let mut options = InjectionRecoveryOptions::default();
        options.period_grid_days = Some(vec![0.5]);
        options.injections_per_cell = 1;
        options.radius_grid_earth = vec![1.0];
        let result = run(&time, &flux, &options, 1, || true).expect("run");
        assert_eq!(result.truncated, Some(true));
        assert_eq!(result.periods_evaluated, Some(0));
    }

    /// F-opencl Task 4: mirrors `test_gpu_circuit_breaker_trips_after_threshold`-
    /// class coverage of `injection.py`'s `_record_gpu_batch_failure`/
    /// `_record_gpu_batch_success` — the breaker must stay closed through
    /// two consecutive failures, trip on the third, and a fresh
    /// `reset_for_test` (test-isolation helper, mirrors
    /// `reset_gpu_circuit_breaker`) returns it to closed. No GPU needed —
    /// pure state-machine logic over the process-wide atomics.
    #[cfg(feature = "opencl")]
    #[test]
    fn opencl_circuit_breaker_trips_after_threshold_and_resets_on_success() {
        opencl_circuit_breaker::reset_for_test();
        assert!(!opencl_circuit_breaker::is_open());

        opencl_circuit_breaker::record_failure("test");
        opencl_circuit_breaker::record_failure("test");
        assert!(!opencl_circuit_breaker::is_open(), "must not trip before the 3rd consecutive failure");

        opencl_circuit_breaker::record_failure("test");
        assert!(opencl_circuit_breaker::is_open(), "must trip on the 3rd consecutive failure");

        opencl_circuit_breaker::reset_for_test();
        assert!(!opencl_circuit_breaker::is_open());
        opencl_circuit_breaker::record_failure("test");
        opencl_circuit_breaker::record_success();
        opencl_circuit_breaker::record_failure("test");
        opencl_circuit_breaker::record_failure("test");
        assert!(!opencl_circuit_breaker::is_open(), "a success must reset the streak, not just decrement it");
        opencl_circuit_breaker::reset_for_test();
    }

    /// 2026-08-06 Spec 2 integration: SYCL's circuit breaker gets the exact
    /// same coverage as `opencl_circuit_breaker_trips_after_threshold_and_resets_on_success`
    /// above (independent atomics, independent module — see
    /// `sycl_circuit_breaker`'s doc comment for why it is a duplicated-
    /// structure sibling rather than a shared generic breaker). No GPU
    /// needed here either.
    #[cfg(feature = "sycl")]
    #[test]
    fn sycl_circuit_breaker_trips_after_threshold_and_resets_on_success() {
        sycl_circuit_breaker::reset_for_test();
        assert!(!sycl_circuit_breaker::is_open());

        sycl_circuit_breaker::record_failure("test");
        sycl_circuit_breaker::record_failure("test");
        assert!(!sycl_circuit_breaker::is_open(), "must not trip before the 3rd consecutive failure");

        sycl_circuit_breaker::record_failure("test");
        assert!(sycl_circuit_breaker::is_open(), "must trip on the 3rd consecutive failure");

        sycl_circuit_breaker::reset_for_test();
        assert!(!sycl_circuit_breaker::is_open());
        sycl_circuit_breaker::record_failure("test");
        sycl_circuit_breaker::record_success();
        sycl_circuit_breaker::record_failure("test");
        sycl_circuit_breaker::record_failure("test");
        assert!(!sycl_circuit_breaker::is_open(), "a success must reset the streak, not just decrement it");
        sycl_circuit_breaker::reset_for_test();
    }
}
