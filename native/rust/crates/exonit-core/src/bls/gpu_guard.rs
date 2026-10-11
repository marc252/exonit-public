//! Shared process-wide GPU-submission lock for <host>'s Intel Arc iGPU,
//! covering BOTH the OpenCL (`opencl_batch.rs`) and SYCL (`sycl_batch.rs`)
//! injection-recovery backends — introduced 2026-08-06 when the SYCL
//! backend was integrated (previously `opencl_batch.rs` had its own
//! private `opencl_guard()`, renamed and moved here).
//!
//! Both backends submit work to the SAME physical device (Arc) through TWO
//! DIFFERENT runtimes (Level Zero via SYCL, the Intel OpenCL ICD via
//! `ocl`) — CLAUDE.md's 2026-07-12 MLX/Metal lesson ("shared mutable
//! driver state, no lock" — restated in `opencl_batch.rs`'s own module doc
//! comment) is the same failure class regardless of which API submitted
//! the work. One process-wide `Mutex<()>` here, not two independent ones
//! (one per backend), so a SYCL submission and an OpenCL submission — e.g.
//! during a canary run comparing both backends, or any future config that
//! somehow ends up constructing both `OpenClBlsBatchBackend` and
//! `SyclBlsBatchBackend` in the same process — can never race against each
//! other on the shared GPU. In production exactly one of the two is ever
//! actually probed+used per process (`resolve_injection_backend`'s
//! priority chain picks ONE backend at construction), so in practice this
//! guard behaves identically to each backend's old private lock — the
//! sharing is defense in depth for a scenario that isn't supposed to
//! happen, the same posture `opencl_batch.rs`'s original `opencl_guard()`
//! doc comment already took toward a hypothetical concurrent FPP-OpenCL
//! caller.

#![cfg(any(feature = "opencl", feature = "sycl"))]

use std::sync::{Mutex, OnceLock};

pub(crate) fn gpu_submission_guard() -> &'static Mutex<()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD.get_or_init(|| Mutex::new(()))
}
