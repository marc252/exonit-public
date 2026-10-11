//! Transliterated from the CPU path of `BLSAccelerated.search` in
//! `Sources/ExoNitCore/BLSMetal.swift`.
//!
//! `BLSComputeBackend` in Swift has three cases (`cpu`, `metal`,
//! `automatic`); this crate implements the CPU path plus, as of F4a, a
//! CUDA path with no Swift equivalent at all (<host>-only — Metal is
//! macOS-only and out of scope for this Linux/Windows-targeted crate per
//! docs/DESIGN.md). `Automatic` is kept exactly as before (not extended to
//! auto-pick CUDA) because Swift's own `.automatic` resolves to CPU for
//! the *primary* BLS search unconditionally today ("Metal-primary pending
//! its own science gate" — BLSMetal.swift), so this is not a
//! reduced/placeholder behavior, it is the current real Swift behavior —
//! and because CUDA selection here is a *host capability* decision
//! (`config.gpu_enabled` + whether a GPU actually initializes), made once
//! by the caller (`task_processor.rs`, mirroring `bls.py`'s import-time
//! probe), not something `search()` should re-derive per call.
//!
//! `Metal` is still not implemented (macOS, out of scope for the
//! <host>/<host> targets this crate serves — see docs/DESIGN.md).

use super::reference::{BlsError, BlsParams, BlsResult};
use super::two_pass;
#[cfg(feature = "cuda")]
use super::cuda::CudaBlsBackend;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum ComputeBackend {
    Cpu,
    Automatic,
    /// F4a: the primary-BLS CUDA driver-API port (`super::cuda`), holding
    /// an already-`probe()`d, ready-to-use backend handle. Carrying the
    /// handle here (rather than re-probing per call) is what makes this
    /// mirror `bls.py`'s "probed once" semantics — see `bls::cuda`'s
    /// module doc comment. Only constructible with the `cuda` Cargo
    /// feature enabled; without it, requesting CUDA is a compile error
    /// rather than a silent CPU fallback.
    #[cfg(feature = "cuda")]
    Cuda(Arc<CudaBlsBackend>),
}

// `Arc<CudaBlsBackend>` has no meaningful `PartialEq` (comparing two
// distinct GPU handles for value-equality isn't a sensible operation, and
// `CudaFunction`/`CudaStream` don't implement it either) — so this compares
// backend *kind*, not payload, which is exactly what every existing call
// site of `ComputeBackend` equality (golden-test assertions selecting a
// literal `Cpu`/`Automatic`) actually needs.
impl PartialEq for ComputeBackend {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}
impl Eq for ComputeBackend {}

/// Mirrors `BLSAccelerated.search(time:flux:parameters:backend:maxConcurrency:)`,
/// restricted to the backends implemented so far (see module doc comment).
pub fn search(
    time: &[f64],
    flux: &[f64],
    parameters: &BlsParams,
    backend: ComputeBackend,
    max_concurrency: usize,
) -> Result<BlsResult, BlsError> {
    match backend {
        ComputeBackend::Cpu | ComputeBackend::Automatic => two_pass::search(time, flux, parameters, max_concurrency),
        // Two-pass on CUDA is the GPU analogue of the CPU arm above (same
        // algorithm class — bls.py's own backend selection wires "cuda" to
        // `bls_gpu_two_pass`, exactly as it wires the CPU fallback to
        // `bls_two_pass`), not a different search strategy.
        #[cfg(feature = "cuda")]
        ComputeBackend::Cuda(backend) => backend.two_pass_search(time, flux, parameters),
    }
}
