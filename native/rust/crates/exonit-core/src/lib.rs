// ExoNit core library — modules are added in phases (see docs/PLAN.md)
//
// F0 (control plane, static parts): config, config_derive, dto, cli,
// pidlock. Transliterated from the Swift reference worker
// (`~/exonit-swift-worker-dev`) — see each module's doc comment for
// its exact Swift source file(s).

pub mod cli;
pub mod config;
pub mod config_derive;
pub mod dto;
pub mod pidlock;
pub mod platform;
pub mod python_sidecar;
mod statistics;

// F1 — FITS + preprocessing (vegeu docs/PLAN.md)
pub mod detrend;
#[cfg(feature = "cuda")]
pub mod detrend_cuda;
pub mod detrend_iterative;
pub mod fits;
pub mod gaps;
pub mod preprocess;

// F2 — BLS CPU + vetting (vegeu docs/PLAN.md)
pub mod bls;
pub mod harmonic_resolver;
pub mod long_period_sweep;
pub mod multi_planet;
pub mod transit_physics;
pub mod vetting;
// F0b — HTTP client, durable outbox, log sink, task-processor seam, and the
// blocking worker runtime loop. See each module's doc comment for its exact
// Swift source file(s).
pub mod api;
pub mod logfile;
pub mod outbox;
pub mod runtime;
pub mod task_processor;
pub mod telemetry;

// F5 — injection-recovery + FPP policy (vegeu docs/PLAN.md)
// F8 — native FPP Monte Carlo port (spec exonit 2026-08-17-fpp-rust-port)
pub mod fpp;
pub mod fpp_policy;
pub mod injection;
pub mod injection_recovery;
// F3 — ONNX inference (CNN FFI + ExoMiner FFI) via `ort`, EP CPU (see
// docs/PLAN.md). Transliterated from `Sources/ExoNitCore/FFIViewGenerator.swift`,
// `ExoMinerFFIFeatures.swift`, `CoreMLFFIClassifier.swift`,
// `CoreMLExoMinerFFI.swift` — see each module's doc comment for the exact
// Swift/Python source and the CoreML->ONNX backend swap.
pub mod exominer_features;
pub mod ffi_views;
pub mod onnx_cnn;
pub mod onnx_exominer;
