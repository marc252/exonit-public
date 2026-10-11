//! `PrimaryTaskProcessor` is the F2b integration point DESIGN.md/PLAN.md's
//! F0b task called out by name: the seam a `TaskProcessor` trait implementer
//! plugs into `runtime.rs`'s primary pool. F0b shipped only `StubTaskProcessor`
//! (network legs, no science); F2b wires the real pipeline transliterated
//! from `Sources/ExoNitCore/PrimaryTaskProcessor.swift` in the Swift
//! reference worker: load light curve -> preprocess (iterative detrend,
//! real BLS quick-search) -> `multi_planet::search` (BLS two-pass + harmonic
//! resolution + long-period sweep + protected slots) -> `vetting::vet_all`
//! -> CNN FFI / ExoMiner FFI scoring per candidate (ONNX, optional — see
//! below) -> injection-recovery completeness (optional, config-gated) ->
//! `PrimaryResultRequest`.
//!
//! `StubTaskProcessor` is kept (not deleted): `exonit-fake-integration`
//! still uses it when `EXONIT_MODELS_DIR` is unset (see its module doc), and
//! it remains the simplest network-only double for any future test that
//! wants one without touching ONNX models.

use crate::api::{WorkerApiClient, WorkerApiError};
use crate::bls::accelerated::ComputeBackend;
use crate::bls::reference::BlsError;
use crate::detrend::DetrendBackend;
use crate::dto::{PrimaryResultRequest, StarMetadata, WorkerTask};
use crate::exominer_features::{self, FfiStellarMetadata};
use crate::ffi_views::{self, FfiViewError};
use crate::fits::{self, FitsError, LightCurveData, PhotometrySource};
use crate::harmonic_resolver::LightCurveSource;
use crate::injection_recovery::{self, InjectionBackend, InjectionRecoveryOptions};
use crate::multi_planet::{self, MultiPlanetError, MultiPlanetOptions};
use crate::onnx_cnn::{CnnFfiClassifier, OnnxFfiError};
use crate::onnx_exominer::ExoMinerFfiClassifier;
use crate::preprocess::{self, DetrendMode, PreprocessError};
use crate::bls::quick::RealQuickBlsSearch;
use crate::runtime::FitsCache;
use crate::long_period_sweep::LongPeriodSweepOptions;
use crate::vetting::{self, VettingError};
use serde_json::{Map, Value};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Processes one primary-pool task into a `PrimaryResultRequest`. Mirrors
/// the shape of Swift's `PrimaryTaskProcessor.process(_:workerID:)`, minus
/// its `async throws` (this crate is blocking-only, and F0b's stub never
/// fails the task for reasons a retry would fix — see `StubTaskProcessor`
/// doc comment).
pub trait TaskProcessor: Send + Sync {
    fn process(&self, task: &WorkerTask, worker_id: &str) -> PrimaryResultRequest;

    /// Native FPP-backfill processing (F8). Returns `None` when this
    /// processor does not support FPP (the runtime then never polls the
    /// pool). The default keeps every existing implementer (stub, tests)
    /// FPP-less. `raw_task` is the exact server dict, echoed back as
    /// `candidate` in the result, like the Python worker does.
    fn process_fpp(
        &self,
        _task: &crate::dto::BackfillTask,
        _raw_task: &Map<String, Value>,
        _worker_id: &str,
    ) -> Option<FppProcessOutcome> {
        None
    }
}

/// Result of one native FPP-backfill task: the payload to post, plus the
/// error message when the calculation failed (payload then carries the
/// Python `-1/-1/"failed"` contract; the message is for `state.failed` so
/// the failure is never silent).
pub struct FppProcessOutcome {
    pub request: crate::dto::FppResultRequest,
    pub error: Option<String>,
}

/// F0b's placeholder primary-pool processor. It performs the *real* network
/// legs a production processor needs — `GET /api/metadata/{tic}` then
/// `GET /api/fits/{filename}` for each listed file, through the runtime's
/// `FitsCache` (proving the single-flight + LRU cache path, not just the
/// HTTP client) — but does zero science: no BLS, no vetting, no candidates.
/// This is deliberate, not an oversight: F0b's E2E harness
/// (`exonit-fake-integration`) needs a processor that exercises
/// "register -> task -> FITS -> heartbeat -> result via outbox" against the
/// real `WorkerRuntime` loop (see docs/PLAN.md's F0b gate), and F2's future
/// `BlsTaskProcessor` will need exactly this same metadata/FITS-fetch
/// scaffolding — so it is written once here and marked non-scientific in
/// the result rather than duplicated later.
///
/// A task whose metadata/FITS legs fail outright is reported `"failed"`
/// (a real, worth-surfacing problem); a task with no FITS files listed, or
/// none of the science fields filled in, is still `"completed"` — F0b has
/// no science to run, that is not a task failure.
pub struct StubTaskProcessor {
    api: Arc<WorkerApiClient>,
    fits_cache: Arc<FitsCache>,
}

impl StubTaskProcessor {
    pub fn new(api: Arc<WorkerApiClient>, fits_cache: Arc<FitsCache>) -> Self {
        StubTaskProcessor { api, fits_cache }
    }
}

impl TaskProcessor for StubTaskProcessor {
    fn process(&self, task: &WorkerTask, worker_id: &str) -> PrimaryResultRequest {
        let started = Instant::now();
        let mut result = PrimaryResultRequest::new(Some(task.task_id.clone()), Some(task.tic_id.clone()));

        let metadata = match self.api.fetch_metadata(&task.tic_id, Some(worker_id)) {
            Ok(metadata) => metadata,
            Err(error) => {
                result.status = "failed".to_string();
                result.error = Some(format!("metadata fetch failed: {error}"));
                result.processing_time_seconds = started.elapsed().as_secs_f64();
                return result;
            }
        };

        let mut bytes_downloaded: u64 = 0;
        for filename in &metadata.fits_files {
            match self.fits_cache.get_or_fetch(filename, || self.api.download_fits(filename)) {
                Ok(bytes) => bytes_downloaded += bytes,
                Err(error) => {
                    result.status = "failed".to_string();
                    result.error = Some(format!("FITS fetch failed for {filename}: {error}"));
                    result.processing_time_seconds = started.elapsed().as_secs_f64();
                    return result;
                }
            }
        }

        result.status = "completed".to_string();
        result.sector_count = metadata.sectors.len() as i64;
        result.ra = metadata.ra;
        result.dec = metadata.dec;
        result.tmag = metadata.tmag;
        result.processing_time_seconds = started.elapsed().as_secs_f64();
        let mut timings = Map::new();
        timings.insert("stub_processor".to_string(), Value::Bool(true));
        timings.insert("fits_bytes_downloaded".to_string(), Value::from(bytes_downloaded));
        timings.insert("fits_files_fetched".to_string(), Value::from(metadata.fits_files.len() as i64));
        result.timings = Some(timings);
        result
    }
}

// ---------------------------------------------------------------------------
// PrimaryTaskProcessor — F2b real science pipeline.
// ---------------------------------------------------------------------------

/// Unifies every (untyped, in Swift) error source `PrimaryTaskProcessor.process`
/// can throw into one concrete type, mirrored back out as `PrimaryResultRequest`'s
/// `status: "failed"` + `error: <description>` by `TaskProcessor::process`
/// below — exactly Swift's outer `catch { return PrimaryResultRequest(...,
/// status: "failed", error: String(describing: error), ...) }`.
///
/// Faithful transliteration note: like Swift, a single candidate's CNN/
/// ExoMiner view-generation failure (`FfiView`/`Onnx` below) aborts the
/// *whole* task as failed, rather than just skipping that one candidate's
/// score — Swift's `for` loop over candidates uses unguarded `try`, so any
/// per-candidate error propagates out of `process` entirely (see
/// `PrimaryTaskProcessor.swift`'s scoring loop). In practice this essentially
/// never fires: candidates reaching this loop already passed BLS with a
/// positive period/duration, which is exactly what `FFIViewGenerator`
/// requires.
#[derive(Debug)]
enum ProcessingError {
    Api(WorkerApiError),
    FitsFetch(String),
    Fits(FitsError),
    NoLightCurve(String),
    Preprocess(PreprocessError),
    MultiPlanet(MultiPlanetError),
    Vetting(VettingError),
    FfiView(FfiViewError),
    Onnx(OnnxFfiError),
    Injection(BlsError),
}

impl fmt::Display for ProcessingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProcessingError::Api(e) => write!(f, "{e}"),
            ProcessingError::FitsFetch(e) => write!(f, "FITS fetch failed: {e}"),
            ProcessingError::Fits(e) => write!(f, "{e}"),
            ProcessingError::NoLightCurve(tic) => write!(f, "no usable light curve for TIC {tic}"),
            ProcessingError::Preprocess(e) => write!(f, "{e}"),
            ProcessingError::MultiPlanet(e) => write!(f, "{e}"),
            ProcessingError::Vetting(e) => write!(f, "{e}"),
            ProcessingError::FfiView(e) => write!(f, "{e}"),
            ProcessingError::Onnx(e) => write!(f, "{e}"),
            ProcessingError::Injection(e) => write!(f, "{e}"),
        }
    }
}

impl From<WorkerApiError> for ProcessingError {
    fn from(value: WorkerApiError) -> Self {
        ProcessingError::Api(value)
    }
}
impl From<FitsError> for ProcessingError {
    fn from(value: FitsError) -> Self {
        ProcessingError::Fits(value)
    }
}
impl From<PreprocessError> for ProcessingError {
    fn from(value: PreprocessError) -> Self {
        ProcessingError::Preprocess(value)
    }
}
impl From<MultiPlanetError> for ProcessingError {
    fn from(value: MultiPlanetError) -> Self {
        ProcessingError::MultiPlanet(value)
    }
}
impl From<VettingError> for ProcessingError {
    fn from(value: VettingError) -> Self {
        ProcessingError::Vetting(value)
    }
}
impl From<FfiViewError> for ProcessingError {
    fn from(value: FfiViewError) -> Self {
        ProcessingError::FfiView(value)
    }
}
impl From<OnnxFfiError> for ProcessingError {
    fn from(value: OnnxFfiError) -> Self {
        ProcessingError::Onnx(value)
    }
}
impl From<BlsError> for ProcessingError {
    fn from(value: BlsError) -> Self {
        ProcessingError::Injection(value)
    }
}

/// Mirrors `PrimaryTaskProcessor` (`Sources/ExoNitCore/PrimaryTaskProcessor.swift`).
/// CNN/ExoMiner FFI classifiers are optional: constructed once at startup
/// from `models_dir` (see `new`), `None` when the models directory isn't
/// configured, a model file is missing, or it fails to load — matching the
/// Python worker's real production behaviour (`worker/cnn_ffi.py`'s
/// `FileNotFoundError` -> `classifier_ffi = None` -> "FFI rows will use SPOC
/// classifier (fallback)", `worker/exominer_ffi_inference.py` analogously)
/// rather than Swift's `main.swift` CLI wiring, which always resolves a
/// non-nil model URL and so throws `missingRequiredModel` at startup if the
/// file isn't there (dead code in practice: the Swift release process always
/// ships `Models/`, but not a safe default for this crate to copy — see
/// `docs/PLAN.md`'s F2b session note). `process()` itself already treats
/// `cnn`/`exominer` being absent as "skip that score, don't fail the task",
/// same as Swift's `if let cnn { ... }` — this constructor just extends that
/// same graceful-degradation posture to model *loading*, not only to model
/// *absence-by-design*.
pub struct PrimaryTaskProcessor {
    api: Arc<WorkerApiClient>,
    fits_cache: Arc<FitsCache>,
    cnn: Option<CnnFfiClassifier>,
    exominer: Option<ExoMinerFfiClassifier>,
    max_concurrency: usize,
    compute_backend: ComputeBackend,
    gpu_used: bool,
    /// F4b: injection-recovery's own CUDA batch backend — gated the same
    /// way as `compute_backend` (`config.gpu_enabled` + probe), but a
    /// SEPARATE probe/backend instance (`bls::cuda_batch::CudaBlsBatchBackend`
    /// vs `bls::cuda::CudaBlsBackend`): different kernels, same "probed once
    /// at construction" contract. See `resolve_injection_backend`'s doc
    /// comment for why this mirrors `injection.py`'s `disable_gpu()` gate
    /// rather than `wotan_detrend.py`'s independent env-var gate.
    injection_backend: InjectionBackend,
    injection_gpu_used: bool,
    /// F-opencl: which concrete backend `resolve_injection_backend` actually
    /// resolved to (`"rust-cuda"`/`"rust-opencl"`/`"rust-cpu"`) — read by
    /// `runtime.rs`'s `build_capabilities` for the `injection_backend`
    /// capability field, replacing the previous `injection_gpu_used`-derived
    /// two-way `"rust-cuda"`/`"rust-cpu"` ternary (which had no way to name
    /// OpenCL).
    injection_backend_label: &'static str,
    /// F4b: wotan biweight detrend's CUDA backend — gated by
    /// `EXONIT_WOTAN_BACKEND`, NOT `config.gpu_enabled` (see
    /// `resolve_detrend_backend`'s doc comment; this is a real,
    /// deliberate divergence from the BLS/injection gate, mirroring
    /// Python's own divergent gating for this one backend).
    detrend_backend: DetrendBackend,
    detrend_gpu_used: bool,
}

impl PrimaryTaskProcessor {
    pub fn new(
        api: Arc<WorkerApiClient>,
        fits_cache: Arc<FitsCache>,
        models_dir: Option<PathBuf>,
        max_concurrency: usize,
        gpu_enabled: bool,
    ) -> Self {
        let (cnn, exominer) = match &models_dir {
            Some(dir) => (load_cnn(dir), load_exominer(dir)),
            None => (None, None),
        };
        let (compute_backend, gpu_used) = resolve_compute_backend(gpu_enabled);
        let (injection_backend, injection_gpu_used, injection_backend_label) = resolve_injection_backend(gpu_enabled);
        let (detrend_backend, detrend_gpu_used) = resolve_detrend_backend();
        PrimaryTaskProcessor {
            api,
            fits_cache,
            cnn,
            exominer,
            max_concurrency: max_concurrency.max(1),
            compute_backend,
            gpu_used,
            injection_backend,
            injection_gpu_used,
            injection_backend_label,
            detrend_backend,
            detrend_gpu_used,
        }
    }

    pub fn cnn_available(&self) -> bool {
        self.cnn.is_some()
    }

    pub fn exominer_available(&self) -> bool {
        self.exominer.is_some()
    }

    /// Whether the BLS backend resolved to a real GPU (F4a: CUDA) at
    /// construction time — see `resolve_compute_backend`'s doc comment.
    /// Read by `runtime.rs`'s `build_capabilities` for the worker's
    /// self-reported `bls_backend`/`gpu_available` capabilities, honestly
    /// this time (previously hardcoded `false`, F2b-era comment, now stale).
    pub fn gpu_used(&self) -> bool {
        self.gpu_used
    }

    /// Whether injection-recovery's CUDA batch backend resolved at
    /// construction time (F4b) — see `resolve_injection_backend`'s doc
    /// comment. Read by `runtime.rs`'s `build_capabilities` for the
    /// `injection_backend` capability, replacing the previous hardcoded
    /// `"rust-cpu"`.
    pub fn injection_gpu_used(&self) -> bool {
        self.injection_gpu_used
    }

    /// Which concrete injection-recovery backend actually resolved
    /// (`"rust-cuda"`/`"rust-opencl"`/`"rust-cpu"`) — see the
    /// `injection_backend_label` field's doc comment.
    pub fn injection_backend_label(&self) -> &'static str {
        self.injection_backend_label
    }

    /// Whether the wotan biweight detrend CUDA backend resolved at
    /// construction time (F4b) — see `resolve_detrend_backend`'s doc
    /// comment.
    pub fn detrend_gpu_used(&self) -> bool {
        self.detrend_gpu_used
    }

    /// Mirrors `PrimaryTaskProcessor.loadCurves(ticID:metadata:)`: FITS files
    /// listed in metadata take priority (downloaded through the shared
    /// `FitsCache`, single-flighted/LRU'd — see `runtime.rs`); with none
    /// listed, falls back to the FFI endpoint's already-decoded sectors.
    fn load_curves(&self, tic_id: &str, metadata: &StarMetadata) -> Result<Vec<LightCurveData>, ProcessingError> {
        if !metadata.fits_files.is_empty() {
            let mut result = Vec::with_capacity(metadata.fits_files.len());
            for filename in &metadata.fits_files {
                let api = &self.api;
                self.fits_cache
                    .get_or_fetch(filename, || api.download_fits(filename))
                    .map_err(ProcessingError::FitsFetch)?;
                let path = self.fits_cache.directory().join(filename);
                result.push(fits::read_light_curve(&path)?);
            }
            return Ok(result);
        }
        match self.api.fetch_ffi(tic_id)? {
            None => Ok(Vec::new()),
            Some(response) => Ok(response
                .sectors
                .into_iter()
                .map(|sector| {
                    let source = if sector.source.to_ascii_lowercase().contains("tglc") {
                        PhotometrySource::Tglc
                    } else {
                        PhotometrySource::Qlp
                    };
                    let point_count = sector.time.len();
                    LightCurveData { time: sector.time, flux: sector.flux, raw_row_count: point_count, cadence: None, source }
                })
                .collect()),
        }
    }

    /// Mirrors `PrimaryTaskProcessor.process(_:workerID:)`'s `do` block; the
    /// `catch` half lives in `TaskProcessor::process` below.
    fn process_inner(&self, task: &WorkerTask, worker_id: &str) -> Result<PrimaryResultRequest, ProcessingError> {
        let metadata = self.api.fetch_metadata(&task.tic_id, Some(worker_id))?;
        let curves = self.load_curves(&task.tic_id, &metadata)?;
        if curves.is_empty() {
            return Err(ProcessingError::NoLightCurve(task.tic_id.clone()));
        }

        let iterative_detrend = config_bool(&task.config, "iterative_detrend", true);
        // F7: same `max_concurrency` cap (config.cpu_threads-derived) as the
        // primary multi-planet BLS search below — quick-BLS is no longer
        // forced single-threaded (see `RealQuickBlsSearch`'s doc comment).
        let quick_search = RealQuickBlsSearch::new(self.max_concurrency);
        let mode =
            if iterative_detrend { DetrendMode::Iterative(&quick_search) } else { DetrendMode::Biweight };
        // F4b: routed through the resolved wotan detrend backend
        // (`EXONIT_WOTAN_BACKEND` — see `resolve_detrend_backend`'s doc
        // comment) instead of the always-CPU `preprocess::process`.
        let processed = preprocess::process_with_backend(&curves, mode, &self.detrend_backend)?;

        let options = multi_planet_options_from_config(&task.config);
        let source = if curves.iter().any(|c| c.source == PhotometrySource::Spoc) {
            LightCurveSource::Spoc
        } else {
            LightCurveSource::Ffi
        };

        let search = multi_planet::search(
            &processed.time,
            &processed.raw_flux,
            &processed.flat_flux,
            metadata.stellar_radius,
            metadata.stellar_mass,
            source,
            &options,
            self.max_concurrency,
            // F4a: CUDA when `config.gpu_enabled` and a GPU actually
            // initialized at construction time (`resolve_compute_backend`);
            // CPU otherwise, with the fallback already logged there. This
            // is a `Clone` (cheap: `Cpu`/`Automatic` are unit variants,
            // `Cuda` clones an `Arc`), not a move, since `self` is reused
            // for every task this processor handles.
            self.compute_backend.clone(),
        )?;
        let mut candidates = search.candidates;

        let vetting = vetting::vet_all(&candidates, &processed.time, &processed.raw_flux)?;

        let mut cnn_scores: Vec<Option<f64>> = vec![None; candidates.len()];
        for index in 0..candidates.len() {
            let (period, t0, duration, depth_ppm, snr) = {
                let c = &candidates[index];
                (c.period_days, c.t0, c.duration_days, c.depth_ppm, c.snr)
            };
            if let Some(cnn) = &self.cnn {
                let views = ffi_views::make_cnn_views(&processed.time, &processed.flat_flux, period, t0, duration)?;
                cnn_scores[index] = Some(cnn.classify(&views)?.probability);
            }
            if let Some(exominer) = &self.exominer {
                let stellar = FfiStellarMetadata {
                    tmag: metadata.tmag,
                    teff: metadata.effective_temperature,
                    logg: metadata.logg,
                    radius: metadata.stellar_radius,
                    mass: metadata.stellar_mass,
                };
                let features = exominer_features::make(
                    &processed.time,
                    &processed.flat_flux,
                    period,
                    t0,
                    duration,
                    depth_ppm,
                    snr,
                    stellar,
                )?;
                let score = exominer.classify(&features)?;
                candidates[index].exominer_ffi_score = Some(score.probability);
                candidates[index].exominer_ffi_model_version = Some("phase3_v6_ensemble_cal".to_string());
            }
        }

        let run_injection =
            config_bool(&task.config, "run_injection", config_bool(&task.config, "run_injection_recovery", true));
        let completeness = if run_injection {
            let injection = InjectionRecoveryOptions {
                stellar_radius_solar: metadata.stellar_radius.unwrap_or(1.0),
                stellar_mass_solar: metadata.stellar_mass,
                injections_per_cell: config_int(&task.config, "n_injections_per_cell", 10) as usize,
                minimum_snr: options.minimum_snr,
                period_count: config_int(&task.config, "injection_n_periods", 5_000) as usize,
                bin_count: config_int(&task.config, "injection_n_bins", 150) as usize,
                duration_step_count: config_int(&task.config, "injection_n_dur_steps", 10) as usize,
                maximum_seconds: Some(config_double(&task.config, "injection_max_seconds", 300.0)),
                ..InjectionRecoveryOptions::default()
            };
            Some(injection_recovery::run_with_backend(
                &processed.time,
                // Mirrors `PrimaryTaskProcessor.injectionFlux(from:)` ==
                // `processed.flatFlux`.
                &processed.flat_flux,
                &injection,
                self.max_concurrency,
                || false,
                &self.injection_backend,
            )?)
        } else {
            None
        };

        let span = processed.time.last().copied().unwrap_or(0.0) - processed.time.first().copied().unwrap_or(0.0);
        let mut timings = Map::new();
        timings.insert("rust_native".to_string(), Value::Bool(true));
        timings.insert("redetrend_failures".to_string(), Value::from(search.redetrend_failure_count as i64));
        timings.insert("unphysical_rejections".to_string(), Value::from(search.rejected_unphysical_count as i64));

        Ok(PrimaryResultRequest {
            task_id: Some(task.task_id.clone()),
            tic_id: Some(task.tic_id.clone()),
            candidates,
            vetting_results: vetting,
            cnn_scores,
            completeness,
            processing_time_seconds: 0.0, // filled in by `TaskProcessor::process`
            // Honest per CLAUDE.md: reflects whether `resolve_compute_backend`
            // actually resolved to CUDA at construction time (F4a), not
            // just whether `config.gpu_enabled` was requested — mirrors
            // `worker.py`'s `"gpu_used": self.gpu_enabled` field EXCEPT
            // that Python conflates "configured" with "used" here (a known
            // Python imprecision this port deliberately does not copy: see
            // `resolve_compute_backend`'s doc comment).
            gpu_used: self.gpu_used,
            sector_count: processed.sector_count as i64,
            point_count: processed.time.len() as i64,
            time_span_days: Some(span),
            ra: metadata.ra,
            dec: metadata.dec,
            tmag: metadata.tmag,
            timings: Some(timings),
            light_curve_load_recovered: false,
            status: "completed".to_string(),
            error: None,
        })
    }
}

impl PrimaryTaskProcessor {
    /// F8 native FPP: same LC path as the primary pipeline (metadata →
    /// FITS/FFI → iterative-detrend preprocess — the exact chain the scan
    /// parity gate proved bit-identical to Python's `_load_star`), then
    /// `fpp::calc_fpp` with the deterministic per-TIC seed.
    fn process_fpp_inner(
        &self,
        task: &crate::dto::BackfillTask,
        worker_id: &str,
    ) -> Result<crate::fpp::FppResult, String> {
        let metadata = self
            .api
            .fetch_metadata(&task.tic_id, Some(worker_id))
            .map_err(|e| format!("metadata: {e}"))?;
        let curves =
            self.load_curves(&task.tic_id, &metadata).map_err(|e| format!("load: {e}"))?;
        if curves.is_empty() {
            return Err(format!("no light curve data for TIC {}", task.tic_id));
        }
        let quick_search = RealQuickBlsSearch::new(self.max_concurrency);
        let processed = preprocess::process_with_backend(
            &curves,
            DetrendMode::Iterative(&quick_search),
            &self.detrend_backend,
        )
        .map_err(|e| format!("preprocess: {e}"))?;
        let nearby_raw = self
            .api
            .fetch_nearby_value(&task.tic_id)
            .map_err(|e| format!("nearby: {e}"))?;
        let trilegal_csv = std::fs::read_to_string(
            self.fits_cache.directory().join(format!("{}_TRILEGAL.csv", task.tic_id)),
        )
        .ok();
        let seed = crate::fpp::rng::tic_seed(&task.tic_id, task.period_days, task.t0);
        let inputs = crate::fpp::FppInputs {
            tic_id: task.tic_id.clone(),
            period: task.period_days,
            t0: task.t0,
            duration: task.duration_days,
            depth_ppm: task.depth_ppm,
            time: processed.time.clone(),
            flux: processed.flat_flux.clone(),
            nearby_raw,
            trilegal_csv,
        };
        let mut probes = crate::fpp::CalcProbes::default();
        // Same scoped-pool CPU ceiling as the BLS paths (`config.cpu_threads`
        // → max_concurrency; project rule 2): the FPP kernels' rayon
        // par-iters run inside this pool, so instances_fpp × cpu_threads is
        // the host's real FPP CPU budget. Thread count cannot change any
        // bit of the result (per-draw independence; verified in the parity
        // gate with 1 vs 6 threads).
        crate::bls::with_pool(self.max_concurrency, || {
            crate::fpp::calc_fpp(&inputs, 100_000, seed, &mut probes)
        })
        .ok_or_else(|| format!("FPP returned None for TIC {}", task.tic_id))
    }
}

impl TaskProcessor for PrimaryTaskProcessor {
    fn process(&self, task: &WorkerTask, worker_id: &str) -> PrimaryResultRequest {
        let started = Instant::now();
        match self.process_inner(task, worker_id) {
            Ok(mut result) => {
                result.processing_time_seconds = started.elapsed().as_secs_f64();
                result
            }
            Err(error) => {
                let mut result = PrimaryResultRequest::new(Some(task.task_id.clone()), Some(task.tic_id.clone()));
                result.status = "failed".to_string();
                result.error = Some(error.to_string());
                result.processing_time_seconds = started.elapsed().as_secs_f64();
                result
            }
        }
    }

    fn process_fpp(
        &self,
        task: &crate::dto::BackfillTask,
        raw_task: &Map<String, Value>,
        worker_id: &str,
    ) -> Option<FppProcessOutcome> {
        let mut req = crate::dto::FppResultRequest::new(
            Some(task.candidate_id),
            Some(task.tic_id.clone()),
            None,
        );
        req.candidate = Some(raw_task.clone());
        let error = match self.process_fpp_inner(task, worker_id) {
            Ok(r) => {
                req.fpp = Some(r.fpp);
                req.nfpp = Some(r.nfpp);
                req.verdict = Some(r.verdict.as_str().to_string());
                None
            }
            Err(message) => {
                // Same failure contract as `worker.py`'s `_report_fpp_failed`:
                // -1/-1/"failed", candidate echoed.
                req.fpp = Some(-1.0);
                req.nfpp = Some(-1.0);
                req.verdict = Some("failed".to_string());
                Some(message)
            }
        };
        Some(FppProcessOutcome { request: req, error })
    }
}

/// Loads the CNN FFI classifier from `<models_dir>/cnn_ffi_v1.onnx` (same
/// filename the Python worker's `worker/cnn_ffi.py::MODEL_PATH` uses).
/// Missing file or load failure both degrade to `None` with a log line —
/// see `PrimaryTaskProcessor`'s doc comment for why this is the graceful,
/// production-faithful choice rather than a hard error.
fn load_cnn(models_dir: &Path) -> Option<CnnFfiClassifier> {
    let path = models_dir.join("cnn_ffi_v1.onnx");
    if !path.exists() {
        eprintln!(
            "PrimaryTaskProcessor: CNN FFI model not present at {} — candidates will have no cnn_score (matches the Python worker's FileNotFoundError fallback)",
            path.display()
        );
        return None;
    }
    match CnnFfiClassifier::new(&path) {
        Ok(classifier) => Some(classifier),
        Err(error) => {
            eprintln!("PrimaryTaskProcessor: failed to load CNN FFI model at {}: {error} — continuing without CNN scores", path.display());
            None
        }
    }
}

/// Loads the ExoMiner FFI classifier from `<models_dir>/exominer_ffi_v6_ensemble.onnx`,
/// auto-loading its sibling `<model>.calib.json` if present (see
/// `ExoMinerFfiClassifier::new`). Same missing-file/load-failure degradation
/// as `load_cnn`.
fn load_exominer(models_dir: &Path) -> Option<ExoMinerFfiClassifier> {
    let path = models_dir.join("exominer_ffi_v6_ensemble.onnx");
    if !path.exists() {
        eprintln!(
            "PrimaryTaskProcessor: ExoMiner FFI model not present at {} — candidates will have no exominer_ffi_score",
            path.display()
        );
        return None;
    }
    match ExoMinerFfiClassifier::new(&path) {
        Ok(classifier) => Some(classifier),
        Err(error) => {
            eprintln!("PrimaryTaskProcessor: failed to load ExoMiner FFI model at {}: {error} — continuing without ExoMiner scores", path.display());
            None
        }
    }
}

/// Resolves the BLS compute backend ONCE, at `PrimaryTaskProcessor`
/// construction — mirrors `bls.py`'s import-time `is_cuda_available()`
/// probe timing (see `bls::cuda`'s module doc comment: probed once per
/// process, not once per task/call).
///
/// Honours `config.gpu_enabled` the same way `bls.py`'s backend-selection
/// priority chain does (`worker.py`'s `self.gpu_enabled` gates whether GPU
/// backends are even attempted — see CLAUDE.md's `gpu_enabled` references):
/// `gpu_enabled=false` never attempts CUDA, matching Python exactly.
/// `gpu_enabled=true` attempts `CudaBlsBackend::probe()`; any failure
/// there already logs a specific `eprintln!` (driver/NVRTC missing, no
/// device, compile/load error) and this function degrades to CPU without
/// logging again — one loud line per failure, not two.
///
/// Returns `(backend, gpu_used)` rather than just `backend` because
/// `PrimaryResultRequest.gpu_used` (the per-task result field) must be
/// honest about whether GPU was ACTUALLY used, not just configured —
/// deliberately more precise than `worker.py`'s own per-task
/// `"gpu_used": self.gpu_enabled` (which reports the config flag, not the
/// resolved backend); CLAUDE.md's "never claim capabilities the worker
/// doesn't actually have" argues for the honest field here even though it
/// is not a byte-for-byte Python port.
#[cfg(feature = "cuda")]
fn resolve_compute_backend(gpu_enabled: bool) -> (ComputeBackend, bool) {
    if !gpu_enabled {
        return (ComputeBackend::Cpu, false);
    }
    match crate::bls::cuda::CudaBlsBackend::probe() {
        Some(backend) => (ComputeBackend::Cuda(std::sync::Arc::new(backend)), true),
        None => (ComputeBackend::Cpu, false),
    }
}

/// Non-`cuda`-feature build: no CUDA backend exists in this binary at all
/// (compiled out, not just unavailable at runtime) — always CPU. Logs once
/// if the config asked for GPU anyway, since a silent CPU fallback here
/// would be exactly the "GPU host quietly on CPU" fleet gotcha CLAUDE.md
/// project rule 2 warns about (mirrors `worker.py`'s `_check_gpu_consistency`).
#[cfg(not(feature = "cuda"))]
fn resolve_compute_backend(gpu_enabled: bool) -> (ComputeBackend, bool) {
    if gpu_enabled {
        eprintln!(
            "PrimaryTaskProcessor: config.gpu_enabled=true but this binary was built without \
             the `cuda` Cargo feature — running BLS on CPU. Rebuild with `--features cuda` for \
             GPU BLS on this host."
        );
    }
    (ComputeBackend::Cpu, false)
}

/// Resolves injection-recovery's CUDA batch backend ONCE, at construction —
/// same timing contract as `resolve_compute_backend`, but a SEPARATE probe
/// (`bls::cuda_batch::CudaBlsBatchBackend::probe`, not
/// `bls::cuda::CudaBlsBackend::probe`) since the batch kernels are a
/// different module (see `bls::cuda_batch`'s doc comment for why an
/// independent `CudaContext::new(0)` call is safe here).
///
/// Gating mirrors `injection.py`'s real production gate EXACTLY:
/// `worker.py` calls `injection.disable_gpu()` (forcing CPU, unconditionally,
/// for every injection GPU backend the auto-detect chain might have picked)
/// precisely when its config has `gpu_enabled=false` — see that function's
/// docstring. Injection's GPU choice is therefore gated by the SAME
/// `config.gpu_enabled` flag as BLS, unlike wotan detrend's independent
/// `EXONIT_WOTAN_BACKEND` gate (`resolve_detrend_backend`) — this is a real
/// difference between the two Python modules' selection logic, not an
/// inconsistency introduced by this port.
/// F-opencl (`docs/plans/2026-08-06-opencl-backend-<host>.md`, Task 4):
/// resolves injection-recovery's GPU batch backend ONCE, at construction,
/// in priority order CUDA > OpenCL > CPU — mirrors `injection.py`'s real
/// production auto-detect chain exactly (CUDA > Metal fused > Metal legacy >
/// OpenCL > CPU; this crate has no Metal injection backend, so CUDA and
/// OpenCL are the only two non-CPU cases here — see
/// `docs/plans/2026-08-06-opencl-python-inventory.md`'s backend-priority
/// table). Gating mirrors `injection.py`'s real production gate EXACTLY:
/// `worker.py` calls `injection.disable_gpu()` (forcing CPU, unconditionally,
/// for every injection GPU backend the auto-detect chain might have picked)
/// precisely when its config has `gpu_enabled=false` — same
/// `config.gpu_enabled` flag as `resolve_compute_backend` (BLS primary,
/// CUDA-only — unlike wotan detrend's independent `EXONIT_WOTAN_BACKEND`
/// gate, see `resolve_detrend_backend`'s doc comment for that contrast).
///
/// Returns `(backend, gpu_used, label)`: `gpu_used`/`label` are honest about
/// what was ACTUALLY resolved (probed and confirmed working), not just what
/// `gpu_enabled` requested — same precision argument as
/// `resolve_compute_backend`'s doc comment. `label` is one of
/// `"rust-cuda"`/`"rust-sycl"`/`"rust-opencl"`/`"rust-cpu"`, read by
/// `runtime.rs`'s `build_capabilities` for the `injection_backend`
/// capability field.
///
/// **2026-08-06 Spec 2 integration**: priority chain extended to CUDA >
/// SYCL > OpenCL > CPU (was CUDA > OpenCL > CPU). SYCL ranks above OpenCL
/// because it measurably beat it on the SAME Arc iGPU both target
/// (`spikes/RESULTS_2026-08-06.md`: +31-37% median) — not a guess, a
/// real-measurement result. CUDA still ranks first: it targets different,
/// NVIDIA-only hardware (<host>/<host>) this crate has separately
/// validated, so CUDA and SYCL never actually compete for the SAME
/// physical device in any real deployment — the ordering here is
/// belt-and-suspenders for a hypothetical host with both an NVIDIA and an
/// Intel GPU, not a claim that CUDA beats SYCL on Arc (untested,
/// irrelevant: no such host exists in this fleet).
fn resolve_injection_backend(gpu_enabled: bool) -> (InjectionBackend, bool, &'static str) {
    if !gpu_enabled {
        return (InjectionBackend::Cpu, false, "rust-cpu");
    }
    if let Some(backend) = try_resolve_cuda_injection_backend() {
        return (backend, true, "rust-cuda");
    }
    if let Some(backend) = try_resolve_sycl_injection_backend() {
        return (backend, true, "rust-sycl");
    }
    if let Some(backend) = try_resolve_opencl_injection_backend() {
        return (backend, true, "rust-opencl");
    }
    #[cfg(not(any(feature = "cuda", feature = "sycl", feature = "opencl")))]
    eprintln!(
        "PrimaryTaskProcessor: config.gpu_enabled=true but this binary was built without the \
         `cuda`, `sycl`, or `opencl` Cargo features — running injection-recovery on CPU."
    );
    (InjectionBackend::Cpu, false, "rust-cpu")
}

#[cfg(feature = "cuda")]
fn try_resolve_cuda_injection_backend() -> Option<InjectionBackend> {
    crate::bls::cuda_batch::CudaBlsBatchBackend::probe().map(|backend| InjectionBackend::Cuda(std::sync::Arc::new(backend)))
}

#[cfg(not(feature = "cuda"))]
fn try_resolve_cuda_injection_backend() -> Option<InjectionBackend> {
    None
}

/// 2026-08-06 Spec 2 integration — see `resolve_injection_backend`'s doc
/// comment for the priority-chain rationale.
#[cfg(feature = "sycl")]
fn try_resolve_sycl_injection_backend() -> Option<InjectionBackend> {
    crate::bls::sycl_batch::SyclBlsBatchBackend::probe().map(|backend| InjectionBackend::Sycl(std::sync::Arc::new(backend)))
}

#[cfg(not(feature = "sycl"))]
fn try_resolve_sycl_injection_backend() -> Option<InjectionBackend> {
    None
}

#[cfg(feature = "opencl")]
fn try_resolve_opencl_injection_backend() -> Option<InjectionBackend> {
    crate::bls::opencl_batch::OpenClBlsBatchBackend::probe().map(|backend| InjectionBackend::OpenCl(std::sync::Arc::new(backend)))
}

#[cfg(not(feature = "opencl"))]
fn try_resolve_opencl_injection_backend() -> Option<InjectionBackend> {
    None
}

/// Resolves the wotan biweight detrend CUDA backend ONCE, at construction.
/// Gating mirrors `wotan_detrend.py` EXACTLY — NOT gated by
/// `config.gpu_enabled` at all (see `resolve_injection_backend`'s doc
/// comment for the contrast): `_forced = EXONIT_WOTAN_BACKEND` env var,
/// `"cpu"` always wins (forces CPU regardless of what would otherwise be
/// probed), any other value (unset, `"cuda"`, `"auto"`, ...) attempts the
/// CUDA probe and falls back to CPU on failure — `"cuda"` specifically logs
/// an extra "requested but not available" line (matching Python's own
/// extra warning for that one value), every other non-`"cpu"` value that
/// fails the probe degrades silently-to-CPU-but-loudly-logged exactly like
/// `bls::cuda`'s own probe failure branches (each already `eprintln!`s).
#[cfg(feature = "cuda")]
fn resolve_detrend_backend() -> (DetrendBackend, bool) {
    let forced = std::env::var("EXONIT_WOTAN_BACKEND").ok().map(|value| value.trim().to_ascii_lowercase());
    if forced.as_deref() == Some("cpu") {
        return (DetrendBackend::Cpu, false);
    }
    match crate::detrend_cuda::CudaDetrendBackend::probe() {
        Some(backend) => (DetrendBackend::Cuda(std::sync::Arc::new(backend)), true),
        None => {
            if forced.as_deref() == Some("cuda") {
                eprintln!(
                    "PrimaryTaskProcessor: EXONIT_WOTAN_BACKEND=cuda requested but CUDA not \
                     available; falling back to wotan CPU."
                );
            }
            (DetrendBackend::Cpu, false)
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn resolve_detrend_backend() -> (DetrendBackend, bool) {
    let forced = std::env::var("EXONIT_WOTAN_BACKEND").ok().map(|value| value.trim().to_ascii_lowercase());
    if forced.as_deref() == Some("cuda") {
        eprintln!(
            "PrimaryTaskProcessor: EXONIT_WOTAN_BACKEND=cuda requested but this binary was built \
             without the `cuda` Cargo feature — running wotan detrend on CPU."
        );
    }
    (DetrendBackend::Cpu, false)
}

// ---------------------------------------------------------------------------
// task.config readers — mirrors the private `Dictionary where Key == String,
// Value == JSONValue` extension at the bottom of `PrimaryTaskProcessor.swift`.
// ---------------------------------------------------------------------------

fn config_bool(config: &Map<String, Value>, key: &str, default: bool) -> bool {
    match config.get(key) {
        Some(Value::Bool(value)) => *value,
        _ => default,
    }
}

fn config_double(config: &Map<String, Value>, key: &str, default: f64) -> f64 {
    match config.get(key).and_then(Value::as_f64) {
        Some(value) => value,
        None => default,
    }
}

fn config_int(config: &Map<String, Value>, key: &str, default: i64) -> i64 {
    match config.get(key) {
        Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)).unwrap_or(default),
        _ => default,
    }
}

fn config_doubles(config: &Map<String, Value>, key: &str, default: &[f64]) -> Vec<f64> {
    match config.get(key) {
        Some(Value::Array(values)) => {
            let parsed: Vec<f64> = values.iter().filter_map(Value::as_f64).collect();
            if parsed.is_empty() { default.to_vec() } else { parsed }
        }
        _ => default.to_vec(),
    }
}

/// Mirrors `Dictionary.doubleRanges(_:)`: an array of 2+-element numeric
/// arrays, each read as `(row[0], row[1])`.
fn config_double_ranges(config: &Map<String, Value>, key: &str) -> Option<Vec<Vec<f64>>> {
    match config.get(key) {
        Some(Value::Array(rows)) => Some(
            rows.iter()
                .map(|row| match row {
                    Value::Array(values) => values.iter().filter_map(Value::as_f64).collect(),
                    _ => Vec::new(),
                })
                .collect(),
        ),
        _ => None,
    }
}

/// Mirrors `PrimaryTaskProcessor.multiPlanetOptions(config:)` field-by-field.
fn multi_planet_options_from_config(config: &Map<String, Value>) -> MultiPlanetOptions {
    let mut options = MultiPlanetOptions::default();
    options.maximum_planets = config_int(config, "max_planets", 5);
    options.minimum_snr = config_double(config, "snr_threshold", 7.0);
    options.period_minimum = config_double(config, "period_min", 0.5);
    let legacy_period_maximum = config_double(config, "period_max_legacy", 50.0);
    options.period_maximum = config_double(config, "period_max", legacy_period_maximum);
    options.period_count = config_int(config, "n_periods", 50_000).max(1) as usize;
    options.protected_slots_enabled = config_bool(config, "medium_period_protected_slot", true);
    options.protected_slot_minimum_snr = config_double(config, "medium_slot_min_snr", 12.0);
    if let Some(ranges) = config_double_ranges(config, "medium_slot_ranges") {
        options.protected_slot_ranges = ranges
            .into_iter()
            .filter_map(|row| if row.len() >= 2 && row[1] > row[0] { Some((row[0], row[1])) } else { None })
            .collect();
    } else if config.contains_key("medium_slot_min_days") || config.contains_key("medium_slot_max_days") {
        let minimum = config_double(config, "medium_slot_min_days", 10.0);
        let maximum = config_double(config, "medium_slot_max_days", 50.0);
        options.protected_slot_ranges = if maximum > minimum { vec![(minimum, maximum)] } else { Vec::new() };
    }
    options.harmonic_long_period_recovery = config_bool(config, "harmonic_long_period_recovery", false);
    options.harmonic_long_period_minimum_days = config_double(config, "harmonic_long_period_min_days", 10.0);
    options.harmonic_long_snr_factor = config_double(config, "harmonic_long_snr_factor", 0.5);
    if config_bool(config, "wotan_long_sweep", true) {
        let mut long = LongPeriodSweepOptions {
            baseline_threshold_days: config_double(config, "wotan_long_threshold_days", 100.0),
            windows_days: config_doubles(config, "wotan_long_windows_days", &[2.0, 4.0]),
            period_minimum_days: config_double(config, "wotan_long_period_min_days", 30.0),
            legacy_period_maximum_days: options.period_maximum,
            period_maximum_cap_days: config_double(config, "period_max_cap", 200.0),
            minimum_transits: config_double(config, "period_max_min_transits", 2.5),
            base_period_count: config_int(config, "wotan_long_n_periods", 50_000).max(1) as usize,
            minimum_snr: config_double(config, "wotan_long_snr_threshold", options.minimum_snr),
            ..LongPeriodSweepOptions::default()
        };
        if config.contains_key("wotan_long_n_periods") {
            long.period_count_scale_reference_maximum_days = None;
        }
        options.long_period_sweep = Some(long);
    } else {
        options.long_period_sweep = None;
    }
    options
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Swift: `testTaskScientificOptions` (`Sources/ExoNitContractTests/main.swift:1065`),
    /// same config keys and expected outcomes.
    #[test]
    fn multi_planet_options_from_config_matches_swift_contract_test() {
        let mut config = Map::new();
        config.insert("period_max".to_string(), Value::from(50));
        config.insert("medium_period_protected_slot".to_string(), Value::from(false));
        config.insert("medium_slot_min_snr".to_string(), Value::from(15));
        config.insert(
            "medium_slot_ranges".to_string(),
            Value::from(vec![vec![4, 9], vec![12, 40]]),
        );
        config.insert("harmonic_long_period_recovery".to_string(), Value::from(true));
        config.insert("harmonic_long_period_min_days".to_string(), Value::from(14));
        config.insert("harmonic_long_snr_factor".to_string(), Value::from(0.65));
        config.insert("wotan_long_sweep".to_string(), Value::from(false));

        let options = multi_planet_options_from_config(&config);
        assert_eq!(options.period_maximum, 50.0, "main BLS maximum must not be clipped to span/2");
        assert!(!options.protected_slots_enabled, "protected-slot disable flag must propagate");
        assert_eq!(options.protected_slot_minimum_snr, 15.0, "protected-slot SNR must propagate");
        assert_eq!(options.protected_slot_ranges, vec![(4.0, 9.0), (12.0, 40.0)], "protected-slot ranges must propagate");
        assert!(options.harmonic_long_period_recovery, "harmonic long-period gate must propagate");
        assert_eq!(options.harmonic_long_period_minimum_days, 14.0, "harmonic long-period minimum must propagate");
        assert_eq!(options.harmonic_long_snr_factor, 0.65, "harmonic long-period SNR factor must propagate");
        assert!(options.long_period_sweep.is_none(), "long sweep disable flag must propagate");
    }

    #[test]
    fn multi_planet_options_from_config_defaults_match_swift_struct_defaults() {
        let options = multi_planet_options_from_config(&Map::new());
        let defaults = MultiPlanetOptions::default();
        assert_eq!(options, defaults, "an empty task config must reproduce MultiPlanetOptions::default()");
    }

    /// Mirrors `PrimaryTaskProcessor.injectionFlux(from:)`: injection-recovery
    /// runs against the detrended flat flux, never the raw flux — asserted
    /// directly at the call site in `process_inner` (`&processed.flat_flux`),
    /// this test guards the config-reading half doesn't accidentally disable
    /// that by misreading `run_injection`/`run_injection_recovery` defaults.
    #[test]
    fn run_injection_defaults_to_true_and_legacy_key_is_honoured() {
        assert!(config_bool(&Map::new(), "run_injection", config_bool(&Map::new(), "run_injection_recovery", true)));
        let mut legacy_off = Map::new();
        legacy_off.insert("run_injection_recovery".to_string(), Value::from(false));
        assert!(!config_bool(&legacy_off, "run_injection", config_bool(&legacy_off, "run_injection_recovery", true)));
        let mut explicit_on = Map::new();
        explicit_on.insert("run_injection_recovery".to_string(), Value::from(false));
        explicit_on.insert("run_injection".to_string(), Value::from(true));
        assert!(config_bool(&explicit_on, "run_injection", config_bool(&explicit_on, "run_injection_recovery", true)));
    }

    #[test]
    fn config_readers_fall_back_to_defaults_on_missing_or_wrong_type_keys() {
        let mut config = Map::new();
        config.insert("wrong_type".to_string(), Value::from("not a number"));
        assert_eq!(config_bool(&config, "missing", true), true);
        assert_eq!(config_double(&config, "wrong_type", 3.5), 3.5);
        assert_eq!(config_int(&config, "wrong_type", 7), 7);
        assert_eq!(config_doubles(&config, "missing", &[1.0, 2.0]), vec![1.0, 2.0]);
        assert_eq!(config_double_ranges(&config, "missing"), None);
    }

    /// F2b degradation requirement: a models directory that exists but has
    /// neither model file present must not fail construction — both
    /// classifiers stay absent, matching the Python worker's real
    /// `FileNotFoundError` -> `classifier_ffi = None` fallback (see this
    /// module's `PrimaryTaskProcessor` doc comment).
    #[test]
    fn missing_model_files_degrade_to_absent_classifiers_not_a_crash() {
        let dir = std::env::temp_dir().join(format!(
            "exonit-rust-empty-models-{}-{:x}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("create empty models dir");
        assert!(load_cnn(&dir).is_none(), "no cnn_ffi_v1.onnx present -> None, not a panic");
        assert!(load_exominer(&dir).is_none(), "no exominer_ffi_v6_ensemble.onnx present -> None, not a panic");
    }

    /// Same guarantee, but via the public constructor a `WorkerRuntime`
    /// actually calls: no `models_dir` configured at all.
    #[test]
    fn new_without_models_dir_reports_honest_unavailable_capabilities() {
        let api = Arc::new(WorkerApiClient::new("http://127.0.0.1:1", "token"));
        let cache_dir = std::env::temp_dir().join(format!(
            "exonit-rust-no-models-fits-cache-{}-{:x}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let fits_cache = Arc::new(FitsCache::new(cache_dir).expect("create fits cache"));
        let processor = PrimaryTaskProcessor::new(api, fits_cache, None, 1, false);
        assert!(!processor.cnn_available());
        assert!(!processor.exominer_available());
        assert!(!processor.gpu_used(), "gpu_enabled: false must never resolve to a GPU backend");
    }

    /// F4a: `gpu_enabled: true` without the `cuda` Cargo feature must
    /// degrade to CPU (logged, not silent — see
    /// `resolve_compute_backend`'s `#[cfg(not(feature = "cuda"))]` arm),
    /// never panic or claim GPU. This test runs in both build
    /// configurations.
    ///
    /// Host-agnostic by design (fixed 2026-08-07): the original version of
    /// this test hardcoded `assert!(!processor.gpu_used())` under the
    /// assumption that it only ever ran on <host> (no NVIDIA GPU). That
    /// assumption broke the FIRST time `cargo test --features cuda --all`
    /// ran on a host where the CUDA gate is actually green end-to-end
    /// (<host>'s RTX 4070, driver 610.88, `bls::cuda::CudaBlsBackend::probe()`
    /// succeeding) — the panic message ("no NVIDIA GPU on this host") was
    /// simply false there, and `gpu_used()` was correctly reporting `true`.
    /// The `#[ignore]`d golden tests in `tests/bls_cuda_golden.rs` cover
    /// numerical correctness of the CUDA path; THIS test only needs to
    /// confirm `gpu_used()` tells the truth about whatever
    /// `resolve_compute_backend` actually resolved on THIS host, so it
    /// re-derives the expected value from the same probe instead of
    /// assuming no GPU exists.
    #[test]
    fn new_with_gpu_enabled_never_panics_and_reports_honest_gpu_used() {
        let api = Arc::new(WorkerApiClient::new("http://127.0.0.1:1", "token"));
        let cache_dir = std::env::temp_dir().join(format!(
            "exonit-rust-gpu-enabled-fits-cache-{}-{:x}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let fits_cache = Arc::new(FitsCache::new(cache_dir).expect("create fits cache"));
        let processor = PrimaryTaskProcessor::new(api, fits_cache, None, 1, true);
        #[cfg(feature = "cuda")]
        let expected_gpu_used = crate::bls::cuda::CudaBlsBackend::probe().is_some();
        #[cfg(not(feature = "cuda"))]
        let expected_gpu_used = false;
        assert_eq!(
            processor.gpu_used(),
            expected_gpu_used,
            "gpu_used() must honestly reflect whether bls::cuda::CudaBlsBackend::probe() actually \
             succeeds on this host, never a hardcoded assumption"
        );
    }

    /// F-opencl Task 4 Step 1: `gpu_enabled: false` must resolve the
    /// injection-recovery backend to `Cpu` unconditionally — even on a host
    /// with a real, working OpenCL GPU (<host>'s Arc) and the `opencl`
    /// Cargo feature compiled in. Mirrors `injection.py`'s
    /// `disable_gpu()`/`worker.py`'s `gpu_enabled=false` gate: the config
    /// flag wins before any backend is even probed, same as
    /// `new_without_models_dir_reports_honest_unavailable_capabilities`'s
    /// `gpu_used()` assertion for the primary BLS path above.
    #[test]
    fn injection_backend_never_resolves_gpu_when_disabled_even_with_opencl_feature_compiled() {
        let api = Arc::new(WorkerApiClient::new("http://127.0.0.1:1", "token"));
        let cache_dir = std::env::temp_dir().join(format!(
            "exonit-rust-injection-gpu-disabled-fits-cache-{}-{:x}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let fits_cache = Arc::new(FitsCache::new(cache_dir).expect("create fits cache"));
        let processor = PrimaryTaskProcessor::new(api, fits_cache, None, 1, false);
        assert!(!processor.injection_gpu_used(), "gpu_enabled: false must never resolve an injection GPU backend");
        assert_eq!(processor.injection_backend_label(), "rust-cpu");
    }
}
