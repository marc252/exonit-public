//! Transliterated from `Sources/ExoNitCore/ExoMinerFFIFeatures.swift`, cross-
//! checked against the Python worker's `_normalise_stellar` /
//! `_normalise_tce` in `~/exonit/worker/exominer_ffi_inference.py`
//! (read-only) — same means/scales, same feature order.
//!
//! f64 throughout the feature math; the cast to `f32` happens once per
//! value, at the exact point the Swift/Python references cast (right
//! before the feature vector is handed to the ONNX model).

use crate::ffi_views::{self, FfiViewError};

/// Mirrors Swift's `FFIStellarMetadata`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FfiStellarMetadata {
    pub tmag: Option<f64>,
    pub teff: Option<f64>,
    pub logg: Option<f64>,
    pub radius: Option<f64>,
    pub mass: Option<f64>,
}

/// Mirrors Swift's `ExoMinerFFIFeatures`.
#[derive(Debug, Clone, PartialEq)]
pub struct ExoMinerFfiFeatures {
    pub global: Vec<f32>,
    pub local: Vec<f32>,
    pub centroid: Vec<f32>,
    pub centroid_mask: Vec<f32>,
    pub stellar: Vec<f32>,
    pub tce: Vec<f32>,
    pub auxiliary: Vec<Vec<f32>>,
}

/// Mirrors `ExoMinerFFIFeatureBuilder.make(time:flux:period:t0:duration:depthPPM:snr:stellar:)`.
///
/// Centroid is unavailable to the worker (matches Python/Swift: zeros +
/// mask=0; the model was trained to ignore the centroid branch under that
/// mask).
pub fn make(
    time: &[f64],
    flux: &[f64],
    period: f64,
    t0: f64,
    duration: f64,
    depth_ppm: f64,
    snr: f64,
    stellar: FfiStellarMetadata,
) -> Result<ExoMinerFfiFeatures, FfiViewError> {
    let views = ffi_views::make_cnn_views(time, flux, period, t0, duration)?;
    let auxiliary = ffi_views::make_auxiliary_views(time, flux, period, t0, duration);

    let inferred_mass = if let Some(mass) = finite(stellar.mass) {
        Some(mass)
    } else if let (Some(logg), Some(radius)) = (finite(stellar.logg), finite(stellar.radius)) {
        Some(10f64.powf(logg) * radius * radius / 10f64.powf(4.44))
    } else {
        None
    };

    let stellar_features: Vec<f32> = [
        z(stellar.tmag, 12.0, 2.5),
        z(stellar.teff, 5_500.0, 1_500.0),
        z(stellar.logg, 4.4, 0.5),
        z(stellar.radius, 1.0, 0.5),
        z(inferred_mass, 1.0, 0.5),
    ]
    .into_iter()
    .map(|v| v as f32)
    .collect();

    let tce_features: Vec<f32> = [
        period.max(0.1).log10(),
        depth_ppm.max(1.0).log10(),
        (duration * 24.0).max(0.1).log10(),
        snr.max(1.0).log10(),
    ]
    .into_iter()
    .map(|v| v as f32)
    .collect();

    Ok(ExoMinerFfiFeatures {
        global: views.global,
        local: views.local,
        centroid: vec![0.0f32; ffi_views::LOCAL_BIN_COUNT],
        centroid_mask: vec![0.0f32],
        stellar: stellar_features,
        tce: tce_features,
        auxiliary,
    })
}

fn z(value: Option<f64>, mean: f64, scale: f64) -> f64 {
    match finite(value) {
        Some(value) => (value - mean) / scale,
        None => 0.0,
    }
}

fn finite(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite())
}
