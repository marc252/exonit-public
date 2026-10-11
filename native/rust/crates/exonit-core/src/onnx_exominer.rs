//! Transliterated from `Sources/ExoNitCore/CoreMLExoMinerFFI.swift`, backed
//! by ONNX Runtime (crate `ort`, EP CPU) against the SAME `.onnx` file the
//! Python worker loads (`worker/models/exominer_ffi_v6_ensemble.onnx`).
//!
//! Calibration matches the Python worker exactly (see
//! `~/exonit/worker/exominer_ffi_inference.py::ExoMinerFFIInferer.__init__`,
//! read-only): a monotone piecewise-linear map from raw sigmoid probability
//! to a calibrated probability, loaded from an optional sibling
//! `<model>.calib.json` (`{"x": [...], "y": [...]}`) next to the model
//! file. An absent file -> raw scores unchanged, while a present but invalid
//! file fails model construction explicitly instead of silently disabling
//! calibration. `np.interp` extrapolates by
//! clamping to the first/last `y` outside `[x[0], x[-1]]`; `apply()` below
//! reproduces that (and also Swift's `PiecewiseLinearCalibration.apply`,
//! which is the same algorithm).
//!
//! Threading and GPU extension point: see `onnx_cnn.rs`'s module doc — same
//! `Mutex<Session>` pattern, same `feature = "cuda"` seam.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::session::{Session, SessionOutputs};
use ort::value::Tensor;
use serde::Deserialize;

use crate::exominer_features::ExoMinerFfiFeatures;
use crate::ffi_views::{self, sigmoid};

const EXPECTED_INPUTS: [&str; 7] = [
    "global_view",
    "local_view",
    "centroid_view",
    "centroid_mask",
    "stellar",
    "tce",
    "aux_views",
];

/// Mirrors Swift's `PiecewiseLinearCalibration`.
#[derive(Debug, Clone, PartialEq)]
pub struct PiecewiseLinearCalibration {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

/// On-disk shape of `<model>.calib.json`, matching Python's
/// `json.load(open(calib_path))` (only `x`/`y` are consumed; `kind`,
/// `model`, `anchor_recall`, `raw_t085` are provenance metadata Python also
/// ignores at load time).
#[derive(Debug, Deserialize)]
struct CalibrationFile {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl PiecewiseLinearCalibration {
    /// Mirrors `PiecewiseLinearCalibration.apply(_:)`.
    pub fn apply(&self, value: f64) -> f64 {
        if self.x.len() != self.y.len() || self.x.len() < 2 {
            return value;
        }
        let (Some(&first_x), Some(&first_y)) = (self.x.first(), self.y.first()) else {
            return value;
        };
        let (Some(&last_x), Some(&last_y)) = (self.x.last(), self.y.last()) else {
            return value;
        };
        if value <= first_x {
            return first_y;
        }
        if value >= last_x {
            return last_y;
        }
        for index in 1..self.x.len() {
            if value <= self.x[index] {
                let fraction = (value - self.x[index - 1]) / (self.x[index] - self.x[index - 1]);
                return self.y[index - 1] + fraction * (self.y[index] - self.y[index - 1]);
            }
        }
        value
    }

    /// Validates the interpolation knots before they are used. In
    /// particular, duplicate or descending `x` values would make the
    /// interpolation denominator zero/negative and can silently produce
    /// NaNs or non-deterministic scores.
    fn validate(&self) -> Result<(), String> {
        if self.x.len() != self.y.len() {
            return Err(format!(
                "x/y length mismatch ({} != {})",
                self.x.len(),
                self.y.len()
            ));
        }
        if self.x.len() < 2 {
            return Err("at least two calibration knots are required".to_string());
        }
        if self
            .x
            .iter()
            .chain(self.y.iter())
            .any(|value| !value.is_finite())
        {
            return Err("calibration knots must be finite".to_string());
        }
        if self.x.windows(2).any(|window| window[1] <= window[0]) {
            return Err("calibration x knots must be strictly increasing".to_string());
        }
        Ok(())
    }

    /// Loads a `<model>.calib.json` sibling file if it exists, matching
    /// Python's `_get_inferer` calibration-loading path for the optional
    /// sidecar. Returns
    /// `Ok(None)` when the file does not exist; `Err` is returned for I/O,
    /// parse, or validation failures on a file that DOES exist. This keeps a
    /// missing optional sidecar compatible while surfacing a broken deploy.
    pub fn load_sibling(model_path: impl AsRef<Path>) -> std::io::Result<Option<Self>> {
        let calib_path = sibling_calib_path(model_path.as_ref());
        let text = match std::fs::read_to_string(&calib_path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let file: CalibrationFile = serde_json::from_str(&text).map_err(std::io::Error::other)?;
        let calibration = PiecewiseLinearCalibration {
            x: file.x,
            y: file.y,
        };
        calibration
            .validate()
            .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidData, message))?;
        Ok(Some(calibration))
    }
}

fn sibling_calib_path(model_path: &Path) -> PathBuf {
    // Mirrors Python's `os.path.splitext(path)[0] + ".calib.json"`:
    // strip exactly the LAST extension (".onnx"), not every extension.
    let mut path = model_path.to_path_buf();
    let stem = model_path.file_stem().unwrap_or_default().to_os_string();
    path.set_file_name(stem);
    let mut name = path.into_os_string();
    name.push(".calib.json");
    PathBuf::from(name)
}

/// Mirrors Swift's `ExoMinerFFIScore`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExoMinerFfiScore {
    pub logit: f64,
    pub raw_probability: f64,
    pub probability: f64,
}

/// Reuses `OnnxFfiError` from `onnx_cnn.rs` — the failure surface (model
/// load, input/output validation, run failure) is identical.
pub use crate::onnx_cnn::OnnxFfiError;

/// Mirrors `CoreMLExoMinerFFI`.
pub struct ExoMinerFfiClassifier {
    session: Mutex<Session>,
    output_name: String,
    calibration: Option<PiecewiseLinearCalibration>,
}

impl ExoMinerFfiClassifier {
    /// `model_path` is caller-supplied (config/CLI), never hardcoded — see
    /// `onnx_cnn.rs`'s matching note. Auto-loads a sibling
    /// `<model>.calib.json` if present, matching the Python worker's
    /// default behaviour exactly (`worker/exominer_ffi_inference.py`).
    /// Use [`Self::with_calibration`] to override (e.g. tests pinning a
    /// specific calibration regardless of what sits on disk next to the
    /// model, as the Swift contract test does).
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self, OnnxFfiError> {
        let calibration = PiecewiseLinearCalibration::load_sibling(model_path.as_ref())
            .map_err(|error| OnnxFfiError::InvalidCalibration(error.to_string()))?;
        Self::build(model_path, calibration)
    }

    pub fn with_calibration(
        model_path: impl AsRef<Path>,
        calibration: Option<PiecewiseLinearCalibration>,
    ) -> Result<Self, OnnxFfiError> {
        Self::build(model_path, calibration)
    }

    /// CPU-only construction (F4b) — see `CnnFfiClassifier::new_cpu_only`'s
    /// doc comment; same rationale, same escape hatch, for the ExoMiner FFI
    /// model.
    pub fn new_cpu_only(model_path: impl AsRef<Path>) -> Result<Self, OnnxFfiError> {
        let calibration = PiecewiseLinearCalibration::load_sibling(model_path.as_ref())
            .map_err(|error| OnnxFfiError::InvalidCalibration(error.to_string()))?;
        Self::with_calibration_cpu_only(model_path, calibration)
    }

    /// CPU-only construction with an explicit calibration override — the
    /// CPU-only analogue of [`Self::with_calibration`], for tests that need
    /// to hold calibration constant while comparing execution providers
    /// (e.g. `onnx_cuda_golden.rs`).
    pub fn with_calibration_cpu_only(
        model_path: impl AsRef<Path>,
        calibration: Option<PiecewiseLinearCalibration>,
    ) -> Result<Self, OnnxFfiError> {
        validate_calibration(calibration.as_ref())?;
        let session = ort::session::Session::builder()?.commit_from_file(model_path.as_ref())?;
        Self::from_session(session, calibration)
    }

    fn build(
        model_path: impl AsRef<Path>,
        calibration: Option<PiecewiseLinearCalibration>,
    ) -> Result<Self, OnnxFfiError> {
        validate_calibration(calibration.as_ref())?;
        // F4b: shared with `CnnFfiClassifier::new` — see `onnx_cnn.rs`'s
        // module doc comment ("Real-hardware finding") for the CUDA/CPU
        // degradation contract, including the smoke-test rationale below.
        let path = model_path.as_ref();
        let session = crate::onnx_cnn::build_session(path)?;
        let classifier = Self::from_session(session, calibration.clone())?;
        #[cfg(feature = "cuda")]
        {
            let smoke = ExoMinerFfiFeatures {
                global: vec![0.0; ffi_views::GLOBAL_BIN_COUNT],
                local: vec![0.0; ffi_views::LOCAL_BIN_COUNT],
                centroid: vec![0.0; ffi_views::LOCAL_BIN_COUNT],
                centroid_mask: vec![0.0],
                stellar: vec![0.0; 5],
                tce: vec![0.0; 4],
                auxiliary: vec![vec![0.0; ffi_views::LOCAL_BIN_COUNT]; 3],
            };
            if let Err(error) = classifier.classify(&smoke) {
                eprintln!(
                    "onnx: CUDA execution provider constructed a session for {} but failed a smoke \
                     inference ({error}) -- falling back to CPU EP",
                    path.display()
                );
                return Self::with_calibration_cpu_only(path, calibration);
            }
        }
        Ok(classifier)
    }

    fn from_session(
        session: Session,
        calibration: Option<PiecewiseLinearCalibration>,
    ) -> Result<Self, OnnxFfiError> {
        let inputs: HashSet<&str> = session.inputs().iter().map(|i| i.name()).collect();
        let expected: HashSet<&str> = EXPECTED_INPUTS.into_iter().collect();
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

        Ok(ExoMinerFfiClassifier {
            session: Mutex::new(session),
            output_name,
            calibration,
        })
    }

    pub fn classify(
        &self,
        features: &ExoMinerFfiFeatures,
    ) -> Result<ExoMinerFfiScore, OnnxFfiError> {
        if features.global.len() != ffi_views::GLOBAL_BIN_COUNT
            || features.local.len() != ffi_views::LOCAL_BIN_COUNT
            || features.centroid.len() != ffi_views::LOCAL_BIN_COUNT
            || features.centroid_mask.len() != 1
            || features.stellar.len() != 5
            || features.tce.len() != 4
            || features.auxiliary.len() != 3
            || !features
                .auxiliary
                .iter()
                .all(|row| row.len() == ffi_views::LOCAL_BIN_COUNT)
        {
            return Err(OnnxFfiError::InvalidViewShape {
                global: features.global.len(),
                local: features.local.len(),
            });
        }

        let global = Tensor::from_array((
            [1usize, 1usize, ffi_views::GLOBAL_BIN_COUNT],
            features.global.clone().into_boxed_slice(),
        ))?;
        let local = Tensor::from_array((
            [1usize, 1usize, ffi_views::LOCAL_BIN_COUNT],
            features.local.clone().into_boxed_slice(),
        ))?;
        let centroid = Tensor::from_array((
            [1usize, 1usize, ffi_views::LOCAL_BIN_COUNT],
            features.centroid.clone().into_boxed_slice(),
        ))?;
        let centroid_mask = Tensor::from_array((
            [1usize, 1usize],
            features.centroid_mask.clone().into_boxed_slice(),
        ))?;
        let stellar = Tensor::from_array((
            [1usize, 5usize],
            features.stellar.clone().into_boxed_slice(),
        ))?;
        let tce = Tensor::from_array(([1usize, 4usize], features.tce.clone().into_boxed_slice()))?;
        let aux_flat: Vec<f32> = features.auxiliary.iter().flatten().copied().collect();
        let aux_views = Tensor::from_array((
            [1usize, 3usize, ffi_views::LOCAL_BIN_COUNT],
            aux_flat.into_boxed_slice(),
        ))?;

        let mut session = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let outputs: SessionOutputs = session.run(ort::inputs![
            "global_view" => global,
            "local_view" => local,
            "centroid_view" => centroid,
            "centroid_mask" => centroid_mask,
            "stellar" => stellar,
            "tce" => tce,
            "aux_views" => aux_views,
        ])?;
        let output = outputs
            .get(self.output_name.as_str())
            .ok_or_else(|| OnnxFfiError::MissingOutput(self.output_name.clone()))?;
        let (_shape, data) = output.try_extract_tensor::<f32>()?;
        let logit = *data
            .first()
            .ok_or_else(|| OnnxFfiError::MissingOutput(self.output_name.clone()))?
            as f64;

        let raw = sigmoid(logit);
        let probability = self
            .calibration
            .as_ref()
            .map(|c| c.apply(raw))
            .unwrap_or(raw);
        Ok(ExoMinerFfiScore {
            logit,
            raw_probability: raw,
            probability,
        })
    }
}

fn validate_calibration(
    calibration: Option<&PiecewiseLinearCalibration>,
) -> Result<(), OnnxFfiError> {
    calibration
        .map(PiecewiseLinearCalibration::validate)
        .transpose()
        .map_err(|message| OnnxFfiError::InvalidCalibration(message))
        .map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::PiecewiseLinearCalibration;
    use std::path::PathBuf;

    fn fixture_path(label: &str) -> (PathBuf, PathBuf) {
        let directory =
            std::env::temp_dir().join(format!("exonit-calibration-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).expect("create calibration fixture directory");
        let model = directory.join("model.onnx");
        let sidecar = directory.join("model.calib.json");
        (model, sidecar)
    }

    #[test]
    fn missing_calibration_is_optional() {
        let (model, sidecar) = fixture_path("missing");
        assert!(!sidecar.exists());
        assert_eq!(
            PiecewiseLinearCalibration::load_sibling(model).unwrap(),
            None
        );
    }

    #[test]
    fn calibration_requires_strictly_increasing_finite_x() {
        let (model, sidecar) = fixture_path("invalid");
        std::fs::write(&sidecar, r#"{"x":[0.0,0.5,0.5],"y":[0.0,0.4,1.0]}"#)
            .expect("write invalid calibration");
        let error = PiecewiseLinearCalibration::load_sibling(model).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("strictly increasing"));
    }

    #[test]
    fn valid_calibration_loads_and_interpolates() {
        let (model, sidecar) = fixture_path("valid");
        std::fs::write(&sidecar, r#"{"x":[0.0,0.5,1.0],"y":[0.0,0.4,1.0]}"#)
            .expect("write valid calibration");
        let calibration = PiecewiseLinearCalibration::load_sibling(model)
            .expect("load calibration")
            .expect("present calibration");
        assert!((calibration.apply(0.25) - 0.2).abs() < f64::EPSILON);
    }
}
