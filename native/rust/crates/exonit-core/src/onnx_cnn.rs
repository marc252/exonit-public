//! Transliterated from `Sources/ExoNitCore/CoreMLFFIClassifier.swift`, but
//! backed by ONNX Runtime (crate `ort`, EP CPU) against the SAME `.onnx`
//! file the Python worker loads (`worker/models/cnn_ffi_v1.onnx`), rather
//! than a CoreML-converted copy — see `docs/DESIGN.md`'s F3 section. The
//! decision surface (input validation, single-output extraction, sigmoid
//! application) is a faithful port of the Swift class; only the inference
//! backend differs.
//!
//! Threading: `ort::session::Session::run` takes `&mut self`, so — exactly
//! like Swift's `NSLock` around `MLModel.prediction(from:)` — concurrent
//! callers serialize through a `Mutex<Session>` here rather than each
//! holding their own session (ONNX Runtime sessions are not cheap to
//! duplicate per worker instance; one session per model per process,
//! guarded, mirrors the Swift worker's actual deployment shape).
//!
//! GPU (F4b, feature `cuda`): [`build_session`] tries the CUDA execution
//! provider first, degrading to CPU with a log line if EITHER EP
//! registration OR session construction fails — see that function's doc
//! comment for exactly why both call sites are checked (the `ort` Rust API
//! in this version has no post-hoc "which EP is actually active" query
//! unlike Python's `session.get_providers()`, so this is the most honest
//! signal available). Mirrors `onnx_providers.py`'s `select_providers`
//! preference order (CUDA, then CPU — this crate has no DirectML EP dep) and
//! its "never silent" posture: a GPU host quietly falling back to CPU is
//! CLAUDE.md's #1 fleet gotcha.
//!
//! **Real-hardware finding (<host>, RTX 5060 Ti, compute 12.0/Blackwell,
//! 2026-08-05)**: a CUDA session can register AND commit successfully and
//! STILL fail at the first actual `session.run()` with
//! `cudaErrorNoKernelImageForDevice` ("no kernel image is available for
//! execution on the device") — the prebuilt `ort`/ONNX-Runtime CUDA binary's
//! compiled kernel images (PTX/cubin) simply do not cover this GPU's SM
//! architecture. Construction success does NOT imply execution success.
//! `CnnFfiClassifier::new`/`ExoMinerFfiClassifier::new` therefore run a
//! throwaway zero-valued inference immediately after construction (see each
//! `new`'s doc comment) and fall back to a fresh CPU-only session if THAT
//! fails — the same lesson `bls_opencl.py`'s `is_opencl_available()`
//! docstring already states for a different backend: "a REAL minimal kernel
//! submission (build + launch + read-back), not just an
//! import/context-creation check".

use std::collections::HashSet;
use std::path::Path;
use std::sync::Mutex;

use ort::session::{Session, SessionOutputs};
use ort::value::Tensor;

use crate::ffi_views::{self, CnnFfiViews};

/// Mirrors Swift's `FFIScore`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FfiScore {
    pub logit: f64,
    pub probability: f64,
}

/// Mirrors Swift's `CoreMLFFIError`, renamed for the ONNX backend and
/// extended with `Onnx` for `ort`-level failures the CoreML port had no
/// equivalent for (model load, session run).
#[derive(Debug)]
pub enum OnnxFfiError {
    InvalidViewShape {
        global: usize,
        local: usize,
    },
    InvalidModelInputs(Vec<String>),
    InvalidModelOutputs(Vec<String>),
    MissingOutput(String),
    /// A calibration sidecar exists but cannot be parsed or validated.
    /// Missing sidecars are intentionally represented by `None` and do not
    /// produce this error.
    InvalidCalibration(String),
    Onnx(ort::Error),
}

impl std::fmt::Display for OnnxFfiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnnxFfiError::InvalidViewShape { global, local } => {
                write!(f, "invalid CNN FFI shapes: global={global}, local={local}")
            }
            OnnxFfiError::InvalidModelInputs(inputs) => {
                write!(f, "unexpected ONNX inputs: {inputs:?}")
            }
            OnnxFfiError::InvalidModelOutputs(outputs) => {
                write!(f, "unexpected ONNX outputs: {outputs:?}")
            }
            OnnxFfiError::MissingOutput(name) => {
                write!(f, "ONNX output {name} is not a scalar/tensor")
            }
            OnnxFfiError::InvalidCalibration(message) => {
                write!(f, "invalid ExoMiner calibration: {message}")
            }
            OnnxFfiError::Onnx(error) => write!(f, "ONNX Runtime error: {error}"),
        }
    }
}

impl std::error::Error for OnnxFfiError {}

impl From<ort::Error> for OnnxFfiError {
    fn from(error: ort::Error) -> Self {
        OnnxFfiError::Onnx(error)
    }
}

/// Builds an ONNX Runtime session for `model_path`, preferring the CUDA
/// execution provider (feature `cuda`) and degrading to CPU — with a log
/// line, never silently — if CUDA doesn't work. Shared by
/// `CnnFfiClassifier::new` and `ExoMinerFfiClassifier::build`; see this
/// module's doc comment for the CUDA/CPU selection rationale.
///
/// Checked at BOTH points a CUDA failure can surface in `ort` 2.0's Rust
/// API, since there is no post-hoc "which EP is actually active" query
/// (unlike Python's `session.get_providers()`):
///   1. EP *registration* (`with_execution_providers`, `.error_on_failure()`
///      set so a registration failure is a real `Err` returned right there —
///      the DEFAULT `ExecutionProviderDispatch` behaviour is
///      `fail_silently()`, which would only be visible via `ort`'s internal
///      `tracing` `error!` call; this binary installs no `tracing`
///      subscriber, so that internal log is silently dropped and
///      `.error_on_failure()` is the only way to actually observe the
///      failure here);
///   2. session *construction* (`commit_from_file`) — ONNX Runtime's CUDA
///      provider is a "provider bridge" shared library, loaded and
///      initialized (cuDNN/cuBLAS included) lazily at session-creation
///      time, so a missing cuDNN/cuBLAS `.so` typically fails HERE, not at
///      step 1's `SessionOptionsAppendExecutionProvider_CUDA_V2` call.
/// Either failure retries with a fresh CPU-only `SessionBuilder`.
#[cfg(feature = "cuda")]
pub(crate) fn build_session(model_path: &Path) -> Result<Session, OnnxFfiError> {
    let cuda_attempt: ort::Result<Session> = (|| {
        Session::builder()?
            .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])?
            .commit_from_file(model_path)
    })();
    match cuda_attempt {
        Ok(session) => Ok(session),
        Err(error) => {
            eprintln!(
                "onnx: CUDA execution provider failed for {} ({error}) -- falling back to CPU EP",
                model_path.display()
            );
            Ok(Session::builder()?.commit_from_file(model_path)?)
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn build_session(model_path: &Path) -> Result<Session, OnnxFfiError> {
    Ok(Session::builder()?.commit_from_file(model_path)?)
}

/// Mirrors `CoreMLFFIClassifier`. Loads `cnn_ffi_v1.onnx` (or an equivalent
/// two-input, one-output CNN FFI model) via ONNX Runtime's CPU execution
/// provider.
pub struct CnnFfiClassifier {
    session: Mutex<Session>,
    output_name: String,
}

impl CnnFfiClassifier {
    /// `model_path` is caller-supplied (config/CLI), never hardcoded here —
    /// per DESIGN.md's F3 requirement that the models directory come from
    /// worker config, not a compiled-in path.
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self, OnnxFfiError> {
        let path = model_path.as_ref();
        let classifier = Self::from_session(build_session(path)?)?;
        // F4b: construction succeeding is not enough — see this module's
        // doc comment ("Real-hardware finding"). Only relevant when the
        // `cuda` feature is compiled in (`build_session` is CPU-only
        // otherwise, so a smoke test would just waste one inference).
        #[cfg(feature = "cuda")]
        {
            let smoke = CnnFfiViews {
                global: vec![0.0; ffi_views::GLOBAL_BIN_COUNT],
                local: vec![0.0; ffi_views::LOCAL_BIN_COUNT],
            };
            if let Err(error) = classifier.classify(&smoke) {
                eprintln!(
                    "onnx: CUDA execution provider constructed a session for {} but failed a smoke \
                     inference ({error}) -- falling back to CPU EP",
                    path.display()
                );
                return Self::new_cpu_only(path);
            }
        }
        Ok(classifier)
    }

    /// CPU-only construction, bypassing `build_session`'s CUDA-preferred
    /// selection (F4b). Exists for explicit CPU-vs-GPU comparison (golden
    /// tests) and as an escape hatch matching Python's
    /// `onnx_providers.select_providers(gpu_enabled=False)`.
    pub fn new_cpu_only(model_path: impl AsRef<Path>) -> Result<Self, OnnxFfiError> {
        Self::from_session(Session::builder()?.commit_from_file(model_path.as_ref())?)
    }

    fn from_session(session: Session) -> Result<Self, OnnxFfiError> {
        let inputs: HashSet<&str> = session.inputs().iter().map(|i| i.name()).collect();
        let expected: HashSet<&str> = ["global_view", "local_view"].into_iter().collect();
        if inputs != expected {
            return Err(OnnxFfiError::InvalidModelInputs(
                inputs.into_iter().map(String::from).collect(),
            ));
        }
        let outputs = session.outputs();
        if outputs.len() != 1 {
            return Err(OnnxFfiError::InvalidModelOutputs(
                outputs.iter().map(|o| o.name().to_string()).collect(),
            ));
        }
        let output_name = outputs[0].name().to_string();

        Ok(CnnFfiClassifier {
            session: Mutex::new(session),
            output_name,
        })
    }

    pub fn classify(&self, views: &CnnFfiViews) -> Result<FfiScore, OnnxFfiError> {
        if views.global.len() != ffi_views::GLOBAL_BIN_COUNT
            || views.local.len() != ffi_views::LOCAL_BIN_COUNT
        {
            return Err(OnnxFfiError::InvalidViewShape {
                global: views.global.len(),
                local: views.local.len(),
            });
        }
        let global = Tensor::from_array((
            [1usize, 1usize, ffi_views::GLOBAL_BIN_COUNT],
            views.global.clone().into_boxed_slice(),
        ))?;
        let local = Tensor::from_array((
            [1usize, 1usize, ffi_views::LOCAL_BIN_COUNT],
            views.local.clone().into_boxed_slice(),
        ))?;

        let mut session = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let outputs: SessionOutputs =
            session.run(ort::inputs!["global_view" => global, "local_view" => local])?;
        let output = outputs
            .get(self.output_name.as_str())
            .ok_or_else(|| OnnxFfiError::MissingOutput(self.output_name.clone()))?;
        let (_shape, data) = output.try_extract_tensor::<f32>()?;
        let logit = *data
            .first()
            .ok_or_else(|| OnnxFfiError::MissingOutput(self.output_name.clone()))?
            as f64;

        Ok(FfiScore {
            logit,
            probability: ffi_views::sigmoid(logit),
        })
    }
}
