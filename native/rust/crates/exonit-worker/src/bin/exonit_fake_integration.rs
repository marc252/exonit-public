//! E2E harness. Mirrors `Sources/ExoNitFakeIntegration/main.swift` in the
//! Swift reference worker in spirit ("register -> task -> FITS -> heartbeat
//! -> result via outbox -> next poll waits", checked against
//! `scripts/fake_exonit_server.py`), but drives it differently: Swift's
//! version calls `WorkerAPIClient`/`PrimaryTaskProcessor`/`DurableOutbox`
//! directly, one step at a time, because building a full `WorkerRuntime`
//! there always needs CoreML model files. This crate's `WorkerRuntime` does
//! not (`StubTaskProcessor` needs none, and `PrimaryTaskProcessor` degrades
//! gracefully when `models_dir` is absent — see `exonit-core/src/task_processor.rs`),
//! so this harness runs the *real* production loop (`WorkerRuntime::run()`,
//! on a background thread) against the fake server instead of
//! re-implementing its steps.
//!
//! **F2b addition**: with `EXONIT_MODELS_DIR` set, this harness switches
//! from `StubTaskProcessor` (network legs only) to the real
//! `PrimaryTaskProcessor` (`WorkerRuntime::with_processor`), running the
//! full FITS -> BLS -> vetting -> CNN/ExoMiner FFI pipeline against a real
//! light curve fixture served by `scripts/fake_exonit_server.py --fits ...`
//! and asserts the delivered result's candidate matches an exact, pinned
//! value — see `check_real_science_result`'s doc comment for the fixture
//! choice (deliberately NOT `tests/data/qlp_sample.fits` — that one is
//! genuinely non-deterministic run to run, see below) and the exact numbers.
//! Without `EXONIT_MODELS_DIR`, behaviour is unchanged from before F2b:
//! `StubTaskProcessor`, no science, only the network-leg assertions
//! (`fits_bytes_downloaded` etc.) run.
//!
//! Usage: `exonit-fake-integration BASE_URL STATE_JSON_PATH`
//!   - BASE_URL: `http://127.0.0.1:PORT` printed by
//!     `scripts/fake_exonit_server.py` on stdout at startup.
//!   - STATE_JSON_PATH: the `--state` file the same fake server process was
//!     started with; this harness polls it directly off disk (no second
//!     HTTP endpoint needed) to know when to stop `WorkerRuntime::run()`.
//!
//! See `docs/PLAN.md`'s F0b/F2b gates for how this is invoked end-to-end.
//! Launch (real-science branch, mirrors PLAN.md's documented invocation —
//! note the fixture, NOT `qlp_sample.fits`, see `check_real_science_result`):
//!
//! ```sh
//! python3 scripts/fake_exonit_server.py \
//!   --fits tests/data/hlsp_tglc_tess_ffi_gaiaid-2080664784495597312-s0014-cam1-ccd1_tess_v1_llc.fits \
//!   --state /tmp/state.json --port 18493
//! EXONIT_MODELS_DIR=~/exonit/worker/models \
//!   cargo run --bin exonit-fake-integration -- http://127.0.0.1:18493 /tmp/state.json
//! ```

use exonit_core::api::WorkerApiClient;
use exonit_core::config::WorkerConfig;
use exonit_core::runtime::{FitsCache, WorkerRuntime};
use exonit_core::task_processor::{PrimaryTaskProcessor, StubTaskProcessor, TaskProcessor};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn fail(message: &str) -> ! {
    eprintln!("exonit-fake-integration: {message}");
    std::process::exit(1);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [base_url, state_path] = args.as_slice() else {
        eprintln!("usage: exonit-fake-integration BASE_URL STATE_JSON_PATH");
        std::process::exit(2);
    };

    let models_dir = std::env::var_os("EXONIT_MODELS_DIR").map(PathBuf::from);
    let using_real_science = models_dir.is_some();

    let work_dir = std::env::temp_dir().join(format!("exonit-rust-fake-integration-{}", std::process::id()));
    std::fs::create_dir_all(&work_dir).unwrap_or_else(|e| fail(&format!("creating work dir: {e}")));
    let mut config = WorkerConfig::new(base_url.as_str(), "fake-token", "rust-integration");
    config.fits_cache_directory = work_dir.join("fits_cache").to_string_lossy().to_string();
    // Short intervals so the poll/heartbeat cycle this harness waits on
    // (see the state-file poll loop below) completes in well under the 30s
    // deadline instead of the 5s/30s field production defaults.
    config.poll_interval_seconds = 0.05;
    config.heartbeat_interval_seconds = 0.2;
    let config = config.validated().unwrap_or_else(|e| fail(&format!("config: {e}")));

    let outbox_path: PathBuf = work_dir.join("outbox.sqlite");

    let runtime = if let Some(models_dir) = models_dir {
        // Real pipeline: build the same `WorkerApiClient`/`FitsCache` pieces
        // `WorkerRuntime::new` would build internally, but hand them to an
        // explicit `PrimaryTaskProcessor` via `with_processor` so this
        // harness controls `models_dir` directly rather than threading it
        // through `WorkerConfig` (kept minimal — this binary doesn't need
        // the config-file/CLI resolution `exonit-rust-worker`'s `main.rs`
        // does for production).
        let api = Arc::new(WorkerApiClient::new(config.server_url.clone(), config.token.clone()));
        let fits_cache = Arc::new(
            FitsCache::new(PathBuf::from(&config.fits_cache_directory).join("fits"))
                .unwrap_or_else(|e| fail(&format!("creating FITS cache: {e}"))),
        );
        let max_concurrency = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        // `gpu_enabled: false` — this harness exercises the network/outbox
        // control plane against a fake server, not GPU BLS; keeps its
        // behavior unchanged from before F4a (see `resolve_compute_backend`
        // for what this flag gates).
        let processor: Arc<dyn TaskProcessor> =
            Arc::new(PrimaryTaskProcessor::new(api, fits_cache, Some(models_dir), max_concurrency, false));
        WorkerRuntime::with_processor(config, &outbox_path, true, None, processor)
            .unwrap_or_else(|e| fail(&format!("constructing runtime: {e}")))
    } else {
        let api = Arc::new(WorkerApiClient::new(config.server_url.clone(), config.token.clone()));
        let fits_cache = Arc::new(
            FitsCache::new(PathBuf::from(&config.fits_cache_directory).join("fits"))
                .unwrap_or_else(|e| fail(&format!("creating FITS cache: {e}"))),
        );
        let processor: Arc<dyn TaskProcessor> = Arc::new(StubTaskProcessor::new(api, fits_cache));
        WorkerRuntime::with_processor(config, &outbox_path, true, None, processor)
            .unwrap_or_else(|e| fail(&format!("constructing runtime: {e}")))
    };
    let state = Arc::clone(runtime.state());

    let run_handle = std::thread::spawn(move || runtime.run());

    // Poll the fake server's on-disk state (task_polls / heartbeats /
    // results, written by scripts/fake_exonit_server.py after every
    // request) until the full cycle has happened: one delivered primary
    // result, and a second `/api/worker/task` poll that got `wait` back
    // (poll 1 returns the fake task, poll 2+ returns `{"task_id":null,
    // "wait_seconds":0.1}` — see fake_exonit_server.py).
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut last_state = serde_json::Value::Null;
    loop {
        if let Ok(text) = std::fs::read_to_string(state_path) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                let results_len = value.get("results").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
                let task_polls = value.get("task_polls").and_then(|v| v.as_i64()).unwrap_or(0);
                let heartbeats = value.get("heartbeats").and_then(|v| v.as_i64()).unwrap_or(0);
                last_state = value;
                if results_len >= 1 && task_polls >= 2 && heartbeats >= 1 {
                    break;
                }
            }
        }
        if Instant::now() >= deadline {
            fail(&format!("timed out waiting for fake server state; last observed: {last_state}"));
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    state.request_shutdown();
    match run_handle.join() {
        Ok(Ok(())) => {}
        Ok(Err(error)) => fail(&format!("WorkerRuntime::run returned an error: {error}")),
        Err(_) => fail("WorkerRuntime::run panicked"),
    }

    let results = last_state.get("results").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    let Some(first_result) = results.first() else { fail("expected at least one delivered result") };
    let status = first_result.get("status").and_then(|v| v.as_str()).unwrap_or("");
    if status != "completed" {
        fail(&format!(
            "expected the primary result to be status \"completed\", got {status:?}: {first_result}"
        ));
    }
    let task_id = first_result.get("task_id").and_then(|v| v.as_str()).unwrap_or("");
    let tic_id = first_result.get("tic_id").and_then(|v| v.as_str()).unwrap_or("");
    if task_id != "fake-task" || tic_id != "123" {
        fail(&format!("unexpected task_id/tic_id in delivered result: {first_result}"));
    }

    let heartbeats = last_state.get("heartbeats").and_then(|v| v.as_i64()).unwrap_or(0);
    if heartbeats < 1 {
        fail("expected at least one heartbeat to have been recorded by the fake server");
    }

    let mut summary = serde_json::json!({
        "worker_id": state.worker_id(),
        "task_id": task_id,
        "tic_id": tic_id,
        "status": status,
        "task_polls": last_state.get("task_polls").cloned().unwrap_or(serde_json::Value::Null),
        "heartbeats": heartbeats,
        "results_delivered": results.len(),
        "used_real_science": using_real_science,
    });

    if using_real_science {
        check_real_science_result(first_result, &mut summary);
    } else {
        let fits_bytes = first_result
            .get("timings")
            .and_then(|t| t.get("fits_bytes_downloaded"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        if fits_bytes == 0 {
            fail("expected the stub processor to have downloaded at least one non-empty FITS file");
        }
        summary["fits_bytes_downloaded"] = serde_json::json!(fits_bytes);
    }

    println!("{summary}");
    let _ = std::fs::remove_dir_all(&work_dir);
}

/// Verifies the delivered `PrimaryResultRequest` carries plausible candidate
/// data — period/SNR/CNN-ExoMiner scores in physically sane ranges, not the
/// stub's placeholder-only fields — AND pins the exact candidate values this
/// fixture+config combination always produces, asserted with hard equality
/// (tiny float tolerance for the JSON round-trip only), not a loose warning.
///
/// **Fixture choice — this is deliberately NOT `tests/data/qlp_sample.fits`.**
/// That fixture's light curve has a gap under 5 days, so
/// `preprocess::prepare`'s conditional gap-fill runs — and, by design,
/// matching the real production pipeline exactly (`preprocess.rs`'s doc
/// comment: "Python calls `fill_gaps`... its production RNG is deliberately
/// unseeded"), that step is genuinely non-deterministic run to run. This was
/// caught as a real E2E flake (2026-08-05: 3 consecutive runs of this
/// harness against `qlp_sample.fits` gave period 1.633-1.678d, SNR
/// 11.8-12.6d, and — worse — a 4th run found *zero* candidates when the
/// noise pushed the lone signal's SNR just under the 7.0 threshold, failing
/// the harness outright). Production's unseeded RNG is intentionally left
/// alone (it mirrors the real Python worker, see above) — the fix is
/// choosing an E2E fixture that never exercises it: this file,
/// `hlsp_tglc_tess_ffi_gaiaid-2080664784495597312-s0014-cam1-ccd1_tess_v1_llc.fits`
/// (the same TGLC sector `crates/exonit-core/tests/primary_task_processor_tests.rs`'s
/// `completes_without_models_and_degrades_scores_gracefully` combines with
/// its s0015 sibling), has `max_gap_days ~= 5.46` **on its own** — already
/// >= the 5.0 gap-fill threshold, so `fill_if_allowed` never runs and no RNG
/// is ever on the critical path. Verified: 5 consecutive clean runs (fresh
/// fake server + harness process each time, no orphans) of this exact
/// harness against this fixture with `EXONIT_MODELS_DIR` set produced
/// **bit-identical** output every time.
///
/// **Pinned values** (task config `{"n_periods":600,"max_planets":1,
/// "run_injection_recovery":false}`, `scripts/fake_exonit_server.py`'s
/// fixed default, `EXONIT_MODELS_DIR=~/exonit/worker/models`,
/// observed 2026-08-05): period **1.9869d**, SNR **7.3**, depth
/// **17738.6ppm**, CNN probability **0.02768654839169093**, ExoMiner
/// probability **0.020120560184798283**. The exact, golden-fixture-grade
/// (tight-tolerance, multi-candidate) pipeline parity check for a
/// *different* TIC/config still lives in
/// `crates/exonit-core/tests/multi_planet_golden.rs`; this harness's job is
/// an E2E smoke test of the whole register->poll->FITS->BLS->CNN/
/// ExoMiner->outbox->heartbeat loop, which is why it pins only one
/// candidate's values rather than the full 5-candidate baseline.
fn check_real_science_result(result: &serde_json::Value, summary: &mut serde_json::Value) {
    let candidates = result.get("candidates").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    if candidates.is_empty() {
        fail(&format!("expected at least one BLS candidate from the real pipeline, got none: {result}"));
    }
    let candidate = &candidates[0];
    let period = candidate.get("period_days").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
    let snr = candidate.get("snr").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
    let depth_ppm = candidate.get("depth_ppm").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
    if !(period > 0.0 && period < 50.0) {
        fail(&format!("candidate #1 period_days not plausible: {period} (candidate: {candidate})"));
    }
    if !(snr >= 7.0 && snr < 200.0) {
        fail(&format!("candidate #1 snr not plausible: {snr} (candidate: {candidate})"));
    }
    if !(depth_ppm.is_finite() && depth_ppm > 0.0) {
        fail(&format!("candidate #1 depth_ppm not plausible: {depth_ppm} (candidate: {candidate})"));
    }
    // Exact pin against the documented deterministic fixture — see this
    // function's doc comment. No gap-fill RNG is on the critical path for
    // this fixture, so this is a real regression gate, not a loose warning.
    const EXPECTED_PERIOD: f64 = 1.9869;
    const EXPECTED_SNR: f64 = 7.3;
    const EXPECTED_DEPTH_PPM: f64 = 17738.6;
    const TOLERANCE: f64 = 1e-6;
    if (period - EXPECTED_PERIOD).abs() > TOLERANCE {
        fail(&format!(
            "candidate #1 period {period} != pinned {EXPECTED_PERIOD} for the documented deterministic fixture \
             — expected if a different --fits file was used (see check_real_science_result's doc comment), \
             otherwise this is a real pipeline regression"
        ));
    }
    if (snr - EXPECTED_SNR).abs() > TOLERANCE {
        fail(&format!(
            "candidate #1 snr {snr} != pinned {EXPECTED_SNR} for the documented deterministic fixture \
             — expected if a different --fits file was used, otherwise this is a real pipeline regression"
        ));
    }
    if (depth_ppm - EXPECTED_DEPTH_PPM).abs() > TOLERANCE {
        fail(&format!(
            "candidate #1 depth_ppm {depth_ppm} != pinned {EXPECTED_DEPTH_PPM} for the documented deterministic \
             fixture — expected if a different --fits file was used, otherwise this is a real pipeline regression"
        ));
    }

    const EXPECTED_CNN_SCORE: f64 = 0.027_686_548_391_690_93;
    const EXPECTED_EXOMINER_SCORE: f64 = 0.020_120_560_184_798_283;

    let cnn_score = result.get("cnn_scores").and_then(|v| v.as_array()).and_then(|a| a.first()).and_then(|v| v.as_f64());
    let exominer_score = candidate.get("exominer_ffi_score").and_then(|v| v.as_f64());
    match cnn_score {
        Some(score) if !(0.0..=1.0).contains(&score) => fail(&format!("CNN FFI probability out of [0,1]: {score}")),
        Some(score) if (score - EXPECTED_CNN_SCORE).abs() > TOLERANCE => fail(&format!(
            "CNN FFI probability {score} != pinned {EXPECTED_CNN_SCORE} for the documented deterministic fixture"
        )),
        Some(_) => {}
        None => fail("expected a CNN FFI score with EXONIT_MODELS_DIR set and cnn_ffi_v1.onnx present"),
    }
    match exominer_score {
        Some(score) if !(0.0..=1.0).contains(&score) => {
            fail(&format!("ExoMiner FFI probability out of [0,1]: {score}"))
        }
        Some(score) if (score - EXPECTED_EXOMINER_SCORE).abs() > TOLERANCE => fail(&format!(
            "ExoMiner FFI probability {score} != pinned {EXPECTED_EXOMINER_SCORE} for the documented deterministic fixture"
        )),
        Some(_) => {}
        None => fail("expected an ExoMiner FFI score with EXONIT_MODELS_DIR set and exominer_ffi_v6_ensemble.onnx present"),
    }

    summary["candidate_count"] = serde_json::json!(candidates.len());
    summary["candidate_1_period_days"] = serde_json::json!(period);
    summary["candidate_1_snr"] = serde_json::json!(snr);
    summary["candidate_1_depth_ppm"] = serde_json::json!(depth_ppm);
    summary["candidate_1_cnn_score"] = serde_json::json!(cnn_score);
    summary["candidate_1_exominer_ffi_score"] = serde_json::json!(exominer_score);
}
