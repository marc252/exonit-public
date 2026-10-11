//! `exonit-bench` — the Rust half of the Rust-vs-Python A/B benchmark
//! harness (docs/PLAN.md's F6 "Benchmark A/B al mateix host"). Its Python
//! twin is `scripts/benchmark_python_stages.py`; both measure the SAME
//! stages, on the SAME input FITS files, with the SAME algorithm parameters,
//! so that `scripts/run_ab_bench.sh`'s interleaved A/B ratios are honest.
//!
//! Local-only: no server, no network, no task queue. Reads FITS files
//! straight off disk and runs the real `exonit-core` science functions —
//! the exact same functions `PrimaryTaskProcessor` (`task_processor.rs`)
//! calls in production — directly, bypassing the HTTP/outbox/runtime
//! machinery that has no bearing on per-stage compute cost.
//!
//! Stages measured, per input FITS file, with 1 discarded warm-up iteration
//! then N measured iterations (mean + min recorded):
//!
//!   (a) FITS read           — `fits::read_light_curve`
//!   (b) preprocess+detrend  — `preprocess::process` with
//!                              `DetrendMode::Iterative` (real BLS quick
//!                              search backing it, `bls::quick::RealQuickBlsSearch`)
//!   (c) BLS two-pass CPU    — `bls::two_pass::search` at production grid
//!                              defaults (`MultiPlanetOptions::default()`'s
//!                              implied `BlsParams`: period 0.5-50d, 50 000
//!                              periods, 200 bins, 15 duration steps)
//!   (d) BLS two-pass CUDA   — `bls::cuda::CudaBlsBackend::two_pass_search`,
//!                              same `BlsParams`, ONLY when built with
//!                              `--features cuda` AND a GPU actually probes
//!                              successfully on this host (<host> only —
//!                              this stage is `null` in the JSON everywhere
//!                              else, not a zero/skip pretending to be a
//!                              measurement)
//!   (e) full per-TIC pipeline — `preprocess::process` (iterative) ->
//!                              `multi_planet::search` -> `vetting::vet_all`,
//!                              timed as ONE block (mirrors
//!                              `PrimaryTaskProcessor.process_inner`'s
//!                              science section; the network/HTTP legs
//!                              either side of it in production are not part
//!                              of this measurement on either side of the
//!                              A/B, by design)
//!   (f) ONNX CNN + ExoMiner FFI scores — `onnx_cnn::CnnFfiClassifier` +
//!                              `onnx_exominer::ExoMinerFfiClassifier`
//!                              (CPU EP; `--features cuda` swaps the EP —
//!                              see those modules' doc comments), scored
//!                              against the CPU two-pass BLS candidate from
//!                              stage (c) (period/t0/duration/depth_ppm/snr),
//!                              not a `CandidateResult` from stage (e) — see
//!                              `run_onnx_stage`'s doc comment for why.
//!
//! JSON output: one document with host/git/flags metadata plus one entry per
//! input file, each stage recording `{iterations, mean_s, min_s, samples_s}`
//! (or `null` when the stage did not run) and a `caveats` array documenting
//! any known non-equivalence with the Python side (empty when none).

use exonit_core::bls::accelerated::ComputeBackend;
use exonit_core::bls::quick::RealQuickBlsSearch;
use exonit_core::bls::reference::BlsParams;
use exonit_core::bls::two_pass;
use exonit_core::exominer_features::{self, FfiStellarMetadata};
use exonit_core::ffi_views;
use exonit_core::fits::{self, LightCurveData};
use exonit_core::multi_planet::{self, MultiPlanetOptions};
use exonit_core::onnx_cnn::CnnFfiClassifier;
use exonit_core::onnx_exominer::ExoMinerFfiClassifier;
use exonit_core::preprocess::{self, DetrendMode};
use exonit_core::vetting;

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[cfg(feature = "cuda")]
use exonit_core::bls::cuda::CudaBlsBackend;

/// Production BLS grid, mirrored from `MultiPlanetOptions::default()` (which
/// is itself `multi_planet_options_from_config(&Map::new())`'s output — see
/// `task_processor.rs`) and from `bls_fast.bls_two_pass`'s own defaults
/// (`worker/bls_fast.py:441`: `n_periods=50000, n_bins=200, n_dur_steps=15`).
/// Both sides of the A/B use exactly these numbers for stage (c)/(d).
fn production_bls_params() -> BlsParams {
    BlsParams {
        period_minimum: 0.5,
        period_maximum: 50.0,
        period_count: 50_000,
        bin_count: 200,
        duration_step_count: 15,
        duration_minimum_fraction: 0.01,
        duration_maximum_fraction: 0.15,
    }
}

struct Args {
    inputs: Vec<PathBuf>,
    iterations: u32,
    models_dir: Option<PathBuf>,
    output: Option<PathBuf>,
    max_concurrency: usize,
    force_gpu: bool,
}

fn print_usage() {
    eprintln!(
        "Usage: exonit-bench (--fits-dir DIR | --fits FILE [FILE...]) [--iterations N] \
         [--models-dir DIR] [--output PATH] [--max-concurrency N] [--gpu]\n\n\
         --fits-dir DIR      benchmark every *.fits file directly inside DIR (non-recursive)\n\
         --fits FILE...      benchmark exactly these FITS files (space-separated, must come last)\n\
         --iterations N      measured iterations per stage per file (default 3); 1 extra warm-up\n\
         --models-dir DIR    dir with cnn_ffi_v1.onnx / exominer_ffi_v6_ensemble.onnx for stage (f)\n\
                              (default: ~/exonit/worker/models if present, else stage (f) skipped)\n\
         --output PATH       write JSON here instead of stdout\n\
         --max-concurrency N passed to bls::two_pass::search (F7: sizes the rayon thread pool\n\
                              the CPU BLS search runs inside — see bls/optimized.rs; default: \n\
                              available_parallelism)\n\
         --gpu                also attempt stage (d)/CUDA EP even when --max-concurrency-only CPU \n\
                              build would otherwise skip probing (no-op without --features cuda)"
    );
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut inputs: Vec<PathBuf> = Vec::new();
    let mut iterations: u32 = 3;
    let mut models_dir: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let default_concurrency = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut max_concurrency = default_concurrency;
    let mut force_gpu = false;

    let mut i = 0usize;
    if raw.is_empty() || raw[0] == "--help" || raw[0] == "-h" {
        return Err(String::new());
    }
    while i < raw.len() {
        match raw[i].as_str() {
            "--fits-dir" => {
                let dir = raw.get(i + 1).ok_or("--fits-dir needs a path")?;
                let entries = std::fs::read_dir(dir).map_err(|e| format!("--fits-dir {dir}: {e}"))?;
                let mut found: Vec<PathBuf> = entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("fits"))
                    .collect();
                found.sort();
                inputs.extend(found);
                i += 2;
            }
            "--fits" => {
                i += 1;
                while i < raw.len() && !raw[i].starts_with("--") {
                    inputs.push(PathBuf::from(&raw[i]));
                    i += 1;
                }
            }
            "--iterations" => {
                iterations = raw.get(i + 1).ok_or("--iterations needs a number")?.parse().map_err(|_| "bad --iterations")?;
                i += 2;
            }
            "--models-dir" => {
                models_dir = Some(PathBuf::from(raw.get(i + 1).ok_or("--models-dir needs a path")?));
                i += 2;
            }
            "--output" => {
                output = Some(PathBuf::from(raw.get(i + 1).ok_or("--output needs a path")?));
                i += 2;
            }
            "--max-concurrency" => {
                max_concurrency = raw.get(i + 1).ok_or("--max-concurrency needs a number")?.parse().map_err(|_| "bad --max-concurrency")?;
                i += 2;
            }
            "--gpu" => {
                force_gpu = true;
                i += 1;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    if inputs.is_empty() {
        return Err("no input FITS files (use --fits-dir or --fits)".to_string());
    }
    if models_dir.is_none() {
        let default = PathBuf::from("~/exonit/worker/models");
        if default.join("cnn_ffi_v1.onnx").exists() {
            models_dir = Some(default);
        }
    }
    Ok(Args { inputs, iterations, models_dir, output, max_concurrency, force_gpu })
}

/// One timed stage: `warmups` extra untimed iterations run first (discarded
/// — JIT/caching/allocator warm-up parity with the Python side's own
/// warm-up loops), then `iterations` timed with `std::time::Instant`
/// (monotonic, matches `time.perf_counter()` semantics on the Python side —
/// both are the OS monotonic clock, not wall/civil time).
fn time_stage<F: FnMut()>(iterations: u32, mut f: F) -> Value {
    // One discarded warm-up.
    f();
    let mut samples: Vec<f64> = Vec::with_capacity(iterations as usize);
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        samples.push(start.elapsed().as_secs_f64());
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    json!({
        "iterations": iterations,
        "mean_s": mean,
        "min_s": min,
        "samples_s": samples,
    })
}

fn git_sha() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Loads one FITS file into a single-curve `Vec<LightCurveData>` (one
/// sector), mirroring how `PrimaryTaskProcessor::load_curves` treats a
/// single-file TIC. Multi-sector stitching is intentionally out of scope
/// for this benchmark (see docs/PLAN.md's F1/F2 gates — the stitch step
/// itself is a fixed-cost `Vec` concat, negligible vs. detrend/BLS, and
/// mixing per-file and per-TIC timings in one file loop would confuse the
/// per-stage story this tool exists to tell).
fn load_curve(path: &Path) -> Result<LightCurveData, String> {
    fits::read_light_curve(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn run_onnx_stage(
    iterations: u32,
    time: &[f64],
    flux: &[f64],
    period: f64,
    t0: f64,
    duration: f64,
    depth_ppm: f64,
    snr: f64,
    cnn: Option<&CnnFfiClassifier>,
    exominer: Option<&ExoMinerFfiClassifier>,
) -> (Value, Value) {
    // Stage (f) scores the CPU two-pass BLS result from stage (c), not a
    // `CandidateResult` out of stage (e)'s `multi_planet::search`. Reason:
    // `multi_planet::search` can legitimately return zero candidates for a
    // quiet light curve (no signal above `minimum_snr`), which would leave
    // stage (f) with nothing to score and no honest way to report a
    // wall-clock number. The BLS two-pass peak always exists (it is the
    // best period/power in the grid regardless of significance), so it is
    // the more reliable per-file scoring target for a *compute-cost*
    // benchmark — this is not what production feeds the classifiers
    // (production only scores real vetted candidates), and is called out
    // explicitly in the top-level "caveats" field for this reason.
    let cnn_stage = match cnn {
        None => Value::Null,
        Some(classifier) => time_stage(iterations, || {
            let views = ffi_views::make_cnn_views(time, flux, period, t0, duration).expect("cnn views");
            let _ = classifier.classify(&views).expect("cnn classify");
        }),
    };
    let exominer_stage = match exominer {
        None => Value::Null,
        Some(classifier) => time_stage(iterations, || {
            let stellar = FfiStellarMetadata::default();
            let features = exominer_features::make(time, flux, period, t0, duration, depth_ppm, snr, stellar)
                .expect("exominer features");
            let _ = classifier.classify(&features).expect("exominer classify");
        }),
    };
    (cnn_stage, exominer_stage)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("exonit-bench: {msg}\n");
            }
            print_usage();
            std::process::exit(if msg.is_empty() { 0 } else { 2 });
        }
    };

    let cnn = args
        .models_dir
        .as_ref()
        .map(|dir| dir.join("cnn_ffi_v1.onnx"))
        .filter(|p| p.exists())
        .and_then(|p| CnnFfiClassifier::new(&p).map_err(|e| eprintln!("exonit-bench: CNN model load failed: {e}")).ok());
    let exominer = args
        .models_dir
        .as_ref()
        .map(|dir| dir.join("exominer_ffi_v6_ensemble.onnx"))
        .filter(|p| p.exists())
        .and_then(|p| ExoMinerFfiClassifier::new(&p).map_err(|e| eprintln!("exonit-bench: ExoMiner model load failed: {e}")).ok());

    #[cfg(feature = "cuda")]
    let cuda_backend: Option<CudaBlsBackend> = {
        let probed = CudaBlsBackend::probe();
        if args.force_gpu && probed.is_none() {
            eprintln!("exonit-bench: --gpu requested but CudaBlsBackend::probe() failed (see eprintln above) — stage (d) will be null");
        }
        probed
    };
    #[cfg(not(feature = "cuda"))]
    {
        if args.force_gpu {
            eprintln!("exonit-bench: --gpu requested but this binary was NOT built with --features cuda — stage (d) will be null");
        }
    }

    let params = production_bls_params();
    let quick_search = RealQuickBlsSearch::new(args.max_concurrency);
    let mp_options = MultiPlanetOptions::default();

    let mut file_results: Vec<Value> = Vec::new();

    for path in &args.inputs {
        eprintln!("exonit-bench: {}", path.display());
        let mut caveats: Vec<String> = Vec::new();

        // (a) FITS read.
        let stage_a = time_stage(args.iterations, || {
            let _ = load_curve(path).expect("fits read");
        });

        let curve = match load_curve(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("exonit-bench: skipping {}: {e}", path.display());
                continue;
            }
        };
        let curves = vec![curve];

        // (b) preprocess + full iterative detrend.
        let stage_b = time_stage(args.iterations, || {
            let _ = preprocess::process(&curves, DetrendMode::Iterative(&quick_search)).expect("preprocess");
        });

        // Reference processed curve (single call, outside the timed loop)
        // feeds stages (c)/(e)/(f) below.
        let processed = preprocess::process(&curves, DetrendMode::Iterative(&quick_search)).expect("preprocess (reference)");

        // (c) BLS two-pass CPU, production grid.
        let stage_c = time_stage(args.iterations, || {
            let _ = two_pass::search(&processed.time, &processed.flat_flux, &params, args.max_concurrency).expect("bls two-pass cpu");
        });
        let cpu_bls = two_pass::search(&processed.time, &processed.flat_flux, &params, args.max_concurrency)
            .expect("bls two-pass cpu (reference)");

        // (d) BLS two-pass CUDA — only with --features cuda AND a probed GPU.
        #[cfg(feature = "cuda")]
        let stage_d = match &cuda_backend {
            Some(backend) => time_stage(args.iterations, || {
                let _ = backend.two_pass_search(&processed.time, &processed.flat_flux, &params).expect("bls two-pass cuda");
            }),
            None => Value::Null,
        };
        #[cfg(not(feature = "cuda"))]
        let stage_d = Value::Null;

        // (e) full per-TIC pipeline: preprocess (iterative) -> multi_planet
        // -> vetting, timed as one block (mirrors
        // `PrimaryTaskProcessor.process_inner`'s science section).
        let backend_for_pipeline = ComputeBackend::Cpu;
        let stage_e = time_stage(args.iterations, || {
            let processed = preprocess::process(&curves, DetrendMode::Iterative(&quick_search)).expect("preprocess (stage e)");
            let search = multi_planet::search(
                &processed.time,
                &processed.raw_flux,
                &processed.flat_flux,
                None,
                None,
                exonit_core::harmonic_resolver::LightCurveSource::Ffi,
                &mp_options,
                args.max_concurrency,
                backend_for_pipeline.clone(),
            )
            .expect("multi_planet search");
            let _ = vetting::vet_all(&search.candidates, &processed.time, &processed.raw_flux).expect("vetting");
        });

        // (f) ONNX CNN + ExoMiner FFI scores, against the stage (c) BLS peak
        // (see run_onnx_stage's doc comment for why).
        let (stage_f_cnn, stage_f_exominer) = run_onnx_stage(
            args.iterations,
            &processed.time,
            &processed.flat_flux,
            cpu_bls.period,
            cpu_bls.t0,
            cpu_bls.duration,
            cpu_bls.depth_ppm,
            cpu_bls.snr,
            cnn.as_ref(),
            exominer.as_ref(),
        );
        if cnn.is_none() {
            caveats.push("CNN FFI model not loaded (missing --models-dir/cnn_ffi_v1.onnx) — stage (f) cnn is null".to_string());
        }
        if exominer.is_none() {
            caveats.push(
                "ExoMiner FFI model not loaded (missing --models-dir/exominer_ffi_v6_ensemble.onnx) — stage (f) exominer is null"
                    .to_string(),
            );
        }
        caveats.push(
            "stage (f) scores the stage (c) BLS two-pass PEAK (period/t0/duration/depth_ppm/snr), \
             not a vetted CandidateResult from stage (e)'s multi_planet::search — a quiet light \
             curve can legitimately produce zero candidates in (e), which would leave (f) with \
             nothing to time; the BLS peak always exists. Matches the Python side's equivalent \
             choice in benchmark_python_stages.py."
                .to_string(),
        );
        caveats.push(
            "stage (f) exominer feeds FfiStellarMetadata::default() (all-None stellar params, \
             centroid zeros/mask=0) — this benchmark has no per-TIC server metadata offline; \
             this affects ONLY the `stellar` feature vector's z-scored values (still exercises \
             the real ONNX forward pass at full cost), not timing."
                .to_string(),
        );
        #[cfg(not(feature = "cuda"))]
        caveats.push("binary built WITHOUT --features cuda — stage (d) is always null on this build".to_string());
        #[cfg(feature = "cuda")]
        if cuda_backend.is_none() {
            caveats.push("--features cuda set but CudaBlsBackend::probe() found no usable GPU on this host — stage (d) is null".to_string());
        }

        file_results.push(json!({
            "path": path.display().to_string(),
            "raw_points": curves_point_count(&curves),
            "processed_points": processed.time.len(),
            "sector_count": processed.sector_count,
            "bls_two_pass_cpu_peak": {
                "period_days": cpu_bls.period,
                "duration_days": cpu_bls.duration,
                "snr": cpu_bls.snr,
                "depth_ppm": cpu_bls.depth_ppm,
            },
            "stages": {
                "fits_read": stage_a,
                "preprocess_detrend": stage_b,
                "bls_two_pass_cpu": stage_c,
                "bls_two_pass_cuda": stage_d,
                "pipeline_full": stage_e,
                "onnx_cnn": stage_f_cnn,
                "onnx_exominer": stage_f_exominer,
            },
            "caveats": caveats,
        }));
    }

    let output = json!({
        "tool": "exonit-bench",
        "worker_repo": "exonit-rust-worker",
        "git_sha": git_sha(),
        "cargo_pkg_version": env!("CARGO_PKG_VERSION"),
        "host": exonit_core::platform::hostname(),
        "cpu_model": exonit_core::platform::cpu_model(),
        "iterations_per_stage": args.iterations,
        "max_concurrency": args.max_concurrency,
        "flags": {
            "cuda_feature": cfg!(feature = "cuda"),
            "cuda_gpu_probed": cfg!(feature = "cuda") && cuda_backend_available(),
        },
        "bls_params": {
            "period_minimum": params.period_minimum,
            "period_maximum": params.period_maximum,
            "period_count": params.period_count,
            "bin_count": params.bin_count,
            "duration_step_count": params.duration_step_count,
        },
        "models_dir": args.models_dir.as_ref().map(|p| p.display().to_string()),
        "files": file_results,
    });

    #[cfg(feature = "cuda")]
    fn cuda_backend_available() -> bool {
        // Re-probed here (module-scope, so `main`'s local `cuda_backend` is
        // out of reach) purely for the top-level flags summary — cheap and
        // side-effect-free (probe-once semantics live in `CudaBlsBackend`
        // itself; a second probe just repeats the same log lines, which is
        // acceptable for a benchmark CLI's final summary line).
        CudaBlsBackend::probe().is_some()
    }
    #[cfg(not(feature = "cuda"))]
    fn cuda_backend_available() -> bool {
        false
    }

    let text = serde_json::to_string_pretty(&output).expect("serialize JSON");
    match &args.output {
        Some(path) => std::fs::write(path, &text).unwrap_or_else(|e| panic!("write {}: {e}", path.display())),
        None => println!("{text}"),
    }
}

fn curves_point_count(curves: &[LightCurveData]) -> usize {
    curves.iter().map(|c| c.time.len()).sum()
}
