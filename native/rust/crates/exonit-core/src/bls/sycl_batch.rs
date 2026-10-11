//! SYCL port of the BATCHED injection-recovery BLS path — Rust host side
//! for the kernel at `../../sycl/bls_inj_batch_sycl.cpp` (compiled by
//! `build.rs`'s `build_sycl()` into `libsycl_bls_inj.so`, feature `sycl`).
//! Promoted 2026-08-06 from Spec 2's Spike 1 after
//! `spikes/RESULTS_2026-08-06.md`'s real-measurement gate (>=15%
//! end-to-end vs OpenCL on <host>'s Arc — measured +31% to +37% median
//! across 6 kernel/runtime variants; the shipped `baseline` C++ variant is
//! within noise of every other one tried, see that file's iteration
//! table).
//!
//! Same fused-kernel design as `bls::opencl_batch`/`bls::cuda_batch`: CPU
//! phase-fold in f64 (chunked over periods, `PERIOD_CHUNK_DEFAULT`) feeding
//! a per-(flux, period) f32 kernel that only does the sliding-window
//! search, mean-centering flux in f64 before the f32 cast. This module's
//! `search_batch` is a direct line-for-line port of
//! `bls::opencl_batch::OpenClBlsBatchBackend::search_batch`'s host-side
//! algorithm — same buffer shapes, same chunking, same final reduction —
//! with the `ocl` crate's buffer/kernel calls replaced by a single
//! `extern "C"` FFI call into the SYCL kernel per chunk. See that module's
//! own doc comment for the design rationale this one inherits unchanged.
//!
//! # Why no per-n_bins program cache (unlike `bls::opencl_batch`)
//!
//! The OpenCL kernel bakes `n_bins` into `__local` array sizes at COMPILE
//! time via `@NB@`/`@TGSUMS@` text substitution, so
//! `OpenClBlsBatchBackend` caches one compiled `Program` per distinct
//! `n_bins`. The SYCL kernel instead sizes its equivalent local memory at
//! kernel SUBMISSION time via `sycl::local_accessor` — no per-`n_bins`
//! recompilation, so this backend needs no program cache at all (see
//! `../../sycl/bls_inj_batch_sycl.cpp`'s file header comment, "Placeholder
//! resolution", for the full comparison). `SyclBlsBatchBackend` therefore
//! has no interior-mutable cache field — it is a stateless marker struct,
//! all real state (the lazily-initialized SYCL queue) lives on the C++
//! side behind its own mutex (`bls_inj_batch_sycl.cpp`'s `g_guard`),
//! itself additionally serialized against the OpenCL backend by
//! `bls::gpu_guard` (see that module's doc comment).
//!
//! # No pre-flight local-memory budget check (unlike `bls::opencl_batch`)
//!
//! `OpenClBlsBatchBackend::check_kernel_limits` queries
//! `CL_DEVICE_LOCAL_MEM_SIZE`/`CL_DEVICE_MAX_WORK_GROUP_SIZE` at probe time
//! (cheap via the `ocl` crate) and rejects an oversized `n_bins` BEFORE
//! touching any buffer. This module has no equivalent: the SYCL C++ side
//! exposes no such device-limit accessor today, and adding one would mean
//! a second FFI surface for a query that, in practice, is never exercised
//! — production `n_bins` (150 injection-recovery / 200 scan) is nowhere
//! near the Arc's measured 64 KB local-memory budget (confirmed via the
//! OpenCL backend's own `probe()` log line on this exact host). If
//! `n_bins` is ever large enough to actually overflow the device's local
//! memory, the SYCL runtime's own allocation failure surfaces as a caught
//! C++ exception → non-zero FFI return code → `BlsError::Backend` here,
//! same end result as `check_kernel_limits`'s explicit
//! `BlsError::InvalidParameters` just discovered one step later (at the
//! first real submission instead of before any buffer). Documented gap,
//! not a silent one.

use std::os::raw::c_char;

use super::gpu_guard::gpu_submission_guard;
use super::reference::{self, BlsError, BlsParams, BlsResult};

/// Same value as `bls::opencl_batch::TG_FLUX` — this crate's shared fused-
/// kernel design's flux-slots-per-work-group constant. Duplicated rather
/// than `use`d from `opencl_batch` because that module is gated behind a
/// DIFFERENT, independent Cargo feature (`opencl`) — a `sycl`-only build
/// (no `opencl` feature) must not need that module to exist at all. Same
/// duplication precedent as `bls::cuda_batch`'s own
/// `shared_mem_bytes`/`opencl_batch::local_mem_bytes` pair (that module's
/// doc comment: "same formula, different backend's name for the same
/// thing"). Not read by any function in this module — unlike
/// `bls::opencl_batch::search_batch`, which needs `TG_FLUX` for its own
/// `n_flux_pad` computation, this module's kernel computes `n_flux_pad`
/// INTERNALLY in `bls_inj_batch_sycl.cpp` (the SYCL side owns `nd_range`
/// construction) — kept here purely as the drift tripwire
/// `tg_flux_matches_opencl_batch_constant` (below) checks against.
#[allow(dead_code)]
const TG_FLUX: usize = 32;

/// Same value as `bls::opencl_batch::PERIOD_CHUNK_DEFAULT` — periods
/// processed per kernel launch, bounding peak `bin_idx`/`counts` host RAM.
/// Duplicated for the same reason `TG_FLUX` above is.
// Calibrated on <host>'s Intel Arc (2026-08-16): 2,000 periods reduced the
// production-shaped 4,000x90x5,000 workload by a paired median ~2.9% versus
// the inherited 1,000-period default, while remaining well inside host RAM.
const PERIOD_CHUNK_DEFAULT: usize = 2000;
const PERIOD_CHUNK_MAX: usize = 10_000;

fn period_chunk_from_env(value: Option<&str>) -> usize {
    value
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|&chunk| chunk > 0)
        .map(|chunk| chunk.min(PERIOD_CHUNK_MAX))
        .unwrap_or(PERIOD_CHUNK_DEFAULT)
}

#[link(name = "sycl_bls_inj")]
extern "C" {
    fn sycl_bls_smoke_test() -> i32;
    fn sycl_bls_last_device_name() -> *const c_char;
    #[allow(clippy::too_many_arguments)]
    fn sycl_bls_inj_batch_f32(
        bin_idx: *const i16,
        counts: *const f32,
        flux_t: *const f32,
        totals: *const f32,
        widths: *const i32,
        n_samples: i32,
        n_flux: i32,
        n_bins: i32,
        n_per: i32,
        n_dur: i32,
        out_power: *mut f32,
        out_wj: *mut u32,
        out_depth: *mut f32,
    ) -> i32;
}

/// SAFETY of every call into the block above: `bls_inj_batch_sycl.cpp`'s
/// exported functions take only plain-old-data pointers/scalars, perform
/// their own bounds checking against the dims passed alongside each
/// pointer (see that file's kernel body — `f >= n_flux` early-returns
/// before writing), and every call site below sizes its buffers to exactly
/// match the dims it passes. `sycl_bls_inj_batch_f32`/`sycl_bls_smoke_test`
/// additionally serialize themselves against `bls_inj_batch_sycl.cpp`'s own
/// `g_guard` C++-side mutex — the Rust-side `gpu_submission_guard()` lock
/// held around every call below is ADDITIONAL cross-backend serialization
/// (vs. the OpenCL backend, see `bls::gpu_guard`'s doc comment), not a
/// substitute for it.
fn device_name() -> String {
    // SAFETY: `sycl_bls_last_device_name` returns either null or a pointer
    // to a `std::string`'s internal buffer that outlives this call (a
    // `static std::string` on the C++ side, per that file's `g_device_name`
    // global) — never freed, never mutated concurrently with this read
    // (both this function and every writer of `g_device_name` acquire the
    // same C++-side `g_guard` mutex first).
    unsafe {
        let ptr = sycl_bls_last_device_name();
        if ptr.is_null() {
            "<null>".to_string()
        } else {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

/// Mirrors `opencl_serial.injection_opencl_disabled_by_env`'s contract
/// (`bls::opencl_batch::env_disables_opencl`), for the SYCL analogue env
/// var `EXONIT_INJECTION_SYCL`: `"0"`/`"false"` (case-insensitive,
/// whitespace-tolerant) disables SYCL injection; unset or any other value,
/// including empty string, does not.
fn env_disables_sycl(value: Option<&str>) -> bool {
    matches!(value.map(|v| v.trim().to_ascii_lowercase()).as_deref(), Some("0") | Some("false"))
}

fn sycl_err<E: std::fmt::Display>(context: &str) -> impl Fn(E) -> BlsError + '_ {
    move |error| BlsError::Backend(format!("bls::sycl_batch: {context}: {error}"))
}

/// Stateless marker — see this module's doc comment, "Why no per-n_bins
/// program cache", for why there is no cache/context field here the way
/// `OpenClBlsBatchBackend` has one. `Clone`+`Debug` derived trivially (no
/// fields); kept as a real struct (not a ZST function pointer or similar)
/// so `InjectionBackend::Sycl(Arc<SyclBlsBatchBackend>)` in
/// `injection_recovery.rs` has the same shape as `InjectionBackend::OpenCl`
/// / `InjectionBackend::Cuda` — uniform `Arc<Backend>` wrapping across all
/// three GPU variants, even though this one's `Arc` wraps nothing but a
/// probe-succeeded marker.
#[derive(Debug, Clone, Copy, Default)]
pub struct SyclBlsBatchBackend {
    _private: (),
}

impl SyclBlsBatchBackend {
    /// Probe for a usable SYCL GPU: env kill-switch, then a real
    /// (1-work-item) kernel submission via `sycl_bls_smoke_test`
    /// (mirrors `OpenClBlsBatchBackend::probe`'s own smoke-test-not-just-
    /// context-construction posture, and that function's Vega-TDR
    /// motivation for why a smoke test is not optional here either — the
    /// SYCL runtime can construct a queue successfully on a device that
    /// then fails a real submission). Returns `None` (after exactly one
    /// `eprintln!` naming the specific failure) on any failure — same
    /// posture as every other GPU backend's `probe()` in this crate.
    pub fn probe() -> Option<Self> {
        if env_disables_sycl(std::env::var("EXONIT_INJECTION_SYCL").ok().as_deref()) {
            eprintln!(
                "bls::sycl_batch: EXONIT_INJECTION_SYCL disables SYCL injection -- falling back \
                 to the next injection backend in resolve_injection_backend's priority chain"
            );
            return None;
        }

        let _guard = gpu_submission_guard().lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        // SAFETY: see this module's "SAFETY of every call" note above.
        let smoke_ok = unsafe { sycl_bls_smoke_test() } != 0;
        if !smoke_ok {
            eprintln!(
                "bls::sycl_batch: SYCL smoke test failed (device={}) -- SYCL injection backend \
                 disabled, resolve_injection_backend falls through to the next backend in its \
                 priority chain",
                device_name()
            );
            return None;
        }

        eprintln!("bls::sycl_batch: SYCL GPU backend ready on {}", device_name());
        Some(SyclBlsBatchBackend { _private: () })
    }

    /// Mirrors `bls::opencl_batch::OpenClBlsBatchBackend::search_batch`
    /// line for line (see that function's doc comment for the full
    /// per-step rationale — phase-fold/mean-centering in f64, transpose to
    /// f32 for coalesced kernel reads, chunked-over-periods launches,
    /// f64 post-process/argmax/SDE). The only real difference is the
    /// per-chunk GPU call itself: one `sycl_bls_inj_batch_f32` FFI call
    /// instead of an `ocl::Kernel` build+enqueue+read-back sequence.
    pub fn search_batch(&self, time: &[f64], flux_matrix: &[Vec<f64>], p: &BlsParams) -> Result<Vec<BlsResult>, BlsError> {
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
        if !(p.duration_minimum_fraction > 0.0) || !(p.duration_maximum_fraction >= p.duration_minimum_fraction) {
            return Err(BlsError::InvalidParameters("duration range".to_string()));
        }

        let n_flux = flux_matrix.len();
        let n_bins = p.bin_count;
        let n_periods = p.period_count;
        let sample_count = time.len();
        let period_chunk = period_chunk_from_env(std::env::var("EXONIT_SYCL_PERIOD_CHUNK").ok().as_deref());

        let t0_reference = time[0];
        let time_relative: Vec<f64> = time.iter().map(|&v| v - t0_reference).collect();
        let periods = reference::frequency_period_grid(p.period_minimum, p.period_maximum, n_periods);
        let duration_bins = reference::duration_bin_grid(p);
        let n_dur = duration_bins.len();
        let widths_i32: Vec<i32> = duration_bins.iter().map(|&v| v as i32).collect();

        let flux_means: Vec<f64> = flux_matrix.iter().map(|flux| flux.iter().sum::<f64>() / sample_count as f64).collect();
        let flux_centered: Vec<Vec<f64>> =
            flux_matrix.iter().zip(&flux_means).map(|(flux, &mean)| flux.iter().map(|&v| v - mean).collect()).collect();
        let totals_f32: Vec<f32> = flux_centered.iter().map(|flux| flux.iter().sum::<f64>() as f32).collect();

        let mut flux_t_f32 = vec![0f32; sample_count * n_flux];
        for (fi, flux) in flux_centered.iter().enumerate() {
            for (i, &value) in flux.iter().enumerate() {
                flux_t_f32[i * n_flux + fi] = value as f32;
            }
        }

        let mut power_all = vec![0f64; n_flux * n_periods];
        let mut depth_all = vec![0f64; n_flux * n_periods];
        let mut best_w_all = vec![0f64; n_flux * n_periods];
        let mut best_j_all = vec![0f64; n_flux * n_periods];

        let mut period_start = 0usize;
        while period_start < n_periods {
            let period_end = (period_start + period_chunk).min(n_periods);
            let n_per = period_end - period_start;

            // CPU-only work stays OUTSIDE the guard, same rule as
            // opencl_batch's chunk loop.
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

            let out_len = n_flux * n_per;
            let mut out_power = vec![0f32; out_len];
            let mut out_wj = vec![0u32; out_len];
            let mut out_depth = vec![0f32; out_len];

            {
                let _guard = gpu_submission_guard().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                // SAFETY: `bin_idx`/`counts`/`flux_t_f32`/`totals_f32`/
                // `widths_i32` are each sized to exactly the dims passed
                // alongside them (`n_per*sample_count`,
                // `n_per*n_bins`, `sample_count*n_flux`, `n_flux`, `n_dur`
                // respectively); `out_power`/`out_wj`/`out_depth` are each
                // sized `n_flux*n_per` and the kernel writes exactly one
                // element per (flux, period) pair in range (padding
                // work-items `f >= n_flux` never write, per the kernel
                // body's own early return — see bls_inj_batch_sycl.cpp).
                let rc = unsafe {
                    sycl_bls_inj_batch_f32(
                        bin_idx.as_ptr(),
                        counts.as_ptr(),
                        flux_t_f32.as_ptr(),
                        totals_f32.as_ptr(),
                        widths_i32.as_ptr(),
                        sample_count as i32,
                        n_flux as i32,
                        n_bins as i32,
                        n_per as i32,
                        n_dur as i32,
                        out_power.as_mut_ptr(),
                        out_wj.as_mut_ptr(),
                        out_depth.as_mut_ptr(),
                    )
                };
                if rc != 0 {
                    return Err(sycl_err::<String>("search_batch")(format!(
                        "sycl_bls_inj_batch_f32 returned {rc} (device={})",
                        device_name()
                    )));
                }
            }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn injection_sycl_env_kill_switch_contract() {
        assert!(env_disables_sycl(Some("0")));
        assert!(env_disables_sycl(Some("false")));
        assert!(env_disables_sycl(Some("FALSE")));
        assert!(env_disables_sycl(Some(" False ")));
        assert!(!env_disables_sycl(Some("1")));
        assert!(!env_disables_sycl(Some("")));
        assert!(!env_disables_sycl(None));
    }

    #[test]
    fn sycl_period_chunk_override_is_bounded_and_falls_back_safely() {
        assert_eq!(period_chunk_from_env(None), PERIOD_CHUNK_DEFAULT);
        assert_eq!(period_chunk_from_env(Some("")), PERIOD_CHUNK_DEFAULT);
        assert_eq!(period_chunk_from_env(Some("0")), PERIOD_CHUNK_DEFAULT);
        assert_eq!(period_chunk_from_env(Some("invalid")), PERIOD_CHUNK_DEFAULT);
        assert_eq!(period_chunk_from_env(Some(" 2000 ")), 2000);
        assert_eq!(period_chunk_from_env(Some("999999")), PERIOD_CHUNK_MAX);
    }

    #[test]
    fn tg_flux_matches_opencl_batch_constant() {
        // Both modules hardcode the same fused-kernel design constant
        // independently (see this module's TG_FLUX doc comment) -- this
        // test is the tripwire that catches the two ever drifting apart.
        #[cfg(feature = "opencl")]
        assert_eq!(TG_FLUX, super::super::opencl_batch::TG_FLUX);
        #[cfg(not(feature = "opencl"))]
        assert_eq!(TG_FLUX, 32);
    }

    /// Sanity check on the real Arc GPU host (<host>): `probe()` must
    /// succeed and report a device name. Run explicitly:
    /// `cargo test --features sycl --test '*' -- --ignored --nocapture`
    /// (mirrors `bls::opencl_batch`'s own `opencl_probe_smoke_on_gpu_host`).
    #[test]
    #[ignore = "needs a real SYCL GPU device"]
    fn sycl_probe_smoke_on_gpu_host() {
        match SyclBlsBatchBackend::probe() {
            Some(_backend) => {
                eprintln!("sycl_probe_smoke_on_gpu_host: probe() succeeded -- device={}", device_name());
            }
            None => panic!("SyclBlsBatchBackend::probe() returned None on a host expected to have a working SYCL GPU"),
        }
    }
}
