//! Ported from `testRuntimeCanaryAuthorization` and
//! `testHeartbeatFailurePreservesActiveTask` in
//! `Sources/ExoNitContractTests/main.swift` (Swift reference worker), plus
//! new coverage for pieces with no Swift source: pool validation (F0b's
//! `BackfillPoolsUnimplemented` refusal — Swift instead requires a
//! configured science sidecar, which does not exist in this crate yet), and
//! `FitsCache`'s single-flight + LRU eviction behaviour (see `runtime.rs`'s
//! module doc comment for why this has no Swift counterpart).

#[path = "support/mod.rs"]
mod support;

use exonit_core::config::WorkerConfig;
use exonit_core::outbox::{now_unix, DurableOutbox};
use exonit_core::runtime::{FitsCache, WorkerRuntime, WorkerRuntimeError, WorkerRuntimeState};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use support::{MockRequest, MockResponse, MockServer};

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "exonit-rust-{name}-{}-{:x}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Swift: `testHeartbeatFailurePreservesActiveTask`.
#[test]
fn heartbeat_failure_preserves_active_task() {
    let state = WorkerRuntimeState::new();
    state.begin("123".to_string(), "primary".to_string());
    state.heartbeat_failed("temporary timeout".to_string());
    assert_eq!(
        state.current_tic(),
        Some("123".to_string()),
        "heartbeat failure must preserve TIC"
    );
    assert_eq!(
        state.phase(),
        Some("primary".to_string()),
        "heartbeat failure must preserve phase"
    );
    assert_eq!(
        state.last_error(),
        Some("temporary timeout".to_string()),
        "heartbeat error remains observable"
    );
}

/// Swift: `testRuntimeCanaryAuthorization`.
#[test]
fn runtime_requires_explicit_canary_authorization() {
    let directory = temp_dir("canary-gate");
    let requests = Arc::new(AtomicUsize::new(0));
    let requests_for_handler = requests.clone();
    let server = MockServer::start(move |request: &MockRequest| -> MockResponse {
        requests_for_handler.fetch_add(1, Ordering::SeqCst);
        panic!("unauthorized runtime attempted network: {}", request.path);
    });
    let config = WorkerConfig::new(server.base_url, "test-token", "rust-canary")
        .validated()
        .expect("valid config");
    let runtime = WorkerRuntime::new(config, &directory.join("outbox.sqlite"), false, None)
        .expect("construct runtime");
    let error = runtime
        .run()
        .expect_err("runtime without canary authorization must fail closed");
    assert!(
        matches!(error, WorkerRuntimeError::CanaryNotAuthorized),
        "expected CanaryNotAuthorized, got {error:?}"
    );
    assert_eq!(
        requests.load(Ordering::SeqCst),
        0,
        "canary gate must run before any network request"
    );
}

/// No Swift source (see module doc comment): Swift's allowed-pool set
/// requires only a configured Python science sidecar for
/// `fpp_backfill`/`exominer_backfill`; this Rust worker has no backfill
/// processor implementation at all yet, so `WorkerRuntime::new` refuses to
/// construct rather than silently accepting a pool it cannot serve.
#[test]
fn unsupported_pool_name_is_rejected() {
    let directory = temp_dir("pools-unsupported");
    let mut config = WorkerConfig::new("http://127.0.0.1:1", "token", "worker")
        .validated()
        .expect("valid config");
    config.task_pools = Some(vec!["primary".to_string(), "made_up_pool".to_string()]);
    let error = match WorkerRuntime::new(config, &directory.join("outbox.sqlite"), true, None) {
        Ok(_) => panic!("unknown pool must be rejected"),
        Err(error) => error,
    };
    match error {
        WorkerRuntimeError::UnsupportedTaskPools(pools) => {
            assert_eq!(pools, vec!["made_up_pool".to_string()])
        }
        other => panic!("expected UnsupportedTaskPools, got {other:?}"),
    }
}

/// New coverage (F8, 2026-08-05, `docs/PLAN.md`, outbox auto-reconciliation
/// fix): a legacy outbox carrying a pre-fix `delivery_state = 'unknown'`
/// entry (real incident 2026-08-05: this used to kill the worker at startup
/// — `WorkerRuntimeError::AmbiguousDelivery`, now removed entirely) must not
/// prevent `WorkerRuntime` from constructing. Reconciliation happens inside
/// `WorkerRuntime::build`, before `run()` even exists to call — so
/// construction succeeding here already proves the runtime doesn't die on a
/// legacy outbox, and re-opening the same database afterwards confirms the
/// entry was actually converted back to a normal retryable 'pending' entry.
#[test]
fn runtime_construction_reconciles_legacy_unknown_outbox_entries() {
    let directory = temp_dir("legacy-unknown-startup");
    let outbox_path = directory.join("outbox.sqlite");
    {
        let outbox = DurableOutbox::new(&outbox_path).expect("seed outbox");
        outbox
            .enqueue(
                "primary:legacy-task",
                "/api/worker/result",
                "wid",
                b"{}",
                10.0,
            )
            .expect("enqueue");
        let entry = outbox.pending(10.0, 10).expect("pending")[0].clone();
        outbox
            .mark_delivery_unknown(&entry.id, "legacy timeout")
            .expect("seed legacy unknown");
        assert_eq!(
            outbox.delivery_unknown_count().expect("unknown count"),
            1,
            "fixture seeded correctly"
        );
    }

    let config = WorkerConfig::new("http://127.0.0.1:1", "token", "legacy-worker")
        .validated()
        .expect("valid config");
    let _runtime = WorkerRuntime::new(config, &outbox_path, false, None)
        .expect("construction must not fail on a legacy outbox");

    let reopened = DurableOutbox::new(&outbox_path).expect("reopen outbox");
    assert_eq!(
        reopened
            .delivery_unknown_count()
            .expect("unknown count after startup"),
        0,
        "startup reconciles legacy 'unknown' entries"
    );
    // Reconciliation stamps `next_attempt_at = now()` (real wall-clock time,
    // not the `10.0` fixture timestamp above), so due-ness must be checked
    // against `now_unix()` rather than the original enqueue time.
    assert_eq!(
        reopened
            .pending(now_unix(), 10)
            .expect("pending after startup")
            .len(),
        1,
        "reconciled entry becomes a normal pending retry, immediately due"
    );
}

#[test]
fn backfill_pools_are_not_yet_implemented() {
    // F8: without EXONIT_RUST_FPP=1 the historical refusal stands; the env
    // var is process-global, so this test must not run concurrently with
    // `fpp_backfill_pool_accepted_behind_canary_flag` mutating it — both
    // serialize on ENV_LOCK.
    let _guard = env_lock().lock().unwrap();
    std::env::remove_var("EXONIT_RUST_FPP");
    let directory = temp_dir("pools-backfill");
    let mut config = WorkerConfig::new("http://127.0.0.1:1", "token", "worker")
        .validated()
        .expect("valid config");
    config.task_pools = Some(vec!["fpp_backfill".to_string()]);
    let error = match WorkerRuntime::new(config, &directory.join("outbox.sqlite"), true, None) {
        Ok(_) => panic!("backfill pool must be rejected"),
        Err(error) => error,
    };
    match error {
        WorkerRuntimeError::BackfillPoolsUnimplemented(pools) => {
            assert_eq!(pools, vec!["fpp_backfill".to_string()])
        }
        other => panic!("expected BackfillPoolsUnimplemented, got {other:?}"),
    }
}

fn env_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

/// F8: with the canary flag set, an `fpp_backfill` pool constructs (the
/// native FPP port makes the pool runnable); `exominer_backfill` is still
/// refused even with the flag.
#[test]
fn fpp_backfill_pool_accepted_behind_canary_flag() {
    let _guard = env_lock().lock().unwrap();
    std::env::set_var("EXONIT_RUST_FPP", "1");
    let directory = temp_dir("pools-fpp-canary");
    let mut config = WorkerConfig::new("http://127.0.0.1:1", "token", "worker")
        .validated()
        .expect("valid config");
    config.task_pools = Some(vec!["fpp_backfill".to_string()]);
    let runtime = WorkerRuntime::new(config, &directory.join("outbox.sqlite"), true, None);
    assert!(
        runtime.is_ok(),
        "fpp_backfill must construct with EXONIT_RUST_FPP=1"
    );

    let mut config2 = WorkerConfig::new("http://127.0.0.1:1", "token", "worker")
        .validated()
        .expect("valid config");
    config2.task_pools = Some(vec!["exominer_backfill".to_string()]);
    match WorkerRuntime::new(config2, &directory.join("outbox2.sqlite"), true, None) {
        Err(WorkerRuntimeError::BackfillPoolsUnimplemented(pools)) => {
            assert_eq!(pools, vec!["exominer_backfill".to_string()])
        }
        Err(other) => panic!("exominer_backfill: unexpected error {other:?}"),
        Ok(_) => panic!("exominer_backfill must stay refused"),
    }
    std::env::remove_var("EXONIT_RUST_FPP");
}

/// `FitsCache`: a cache hit avoids calling `fetch` again.
#[test]
fn fits_cache_hit_skips_fetch() {
    let directory = temp_dir("fits-cache-hit");
    let cache = FitsCache::new(&directory).expect("create cache");
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_fetch = calls.clone();
    let bytes = cache
        .get_or_fetch("a.fits", || -> Result<Vec<u8>, String> {
            calls_for_fetch.fetch_add(1, Ordering::SeqCst);
            Ok(vec![1, 2, 3, 4])
        })
        .expect("first fetch");
    assert_eq!(bytes, 4, "reports fetched byte count");
    assert_eq!(calls.load(Ordering::SeqCst), 1, "first call fetches");

    let calls_for_fetch2 = calls.clone();
    let bytes2 = cache
        .get_or_fetch("a.fits", || -> Result<Vec<u8>, String> {
            calls_for_fetch2.fetch_add(1, Ordering::SeqCst);
            Ok(vec![9, 9, 9, 9])
        })
        .expect("second fetch (cache hit)");
    assert_eq!(
        bytes2, 4,
        "cache hit reports the cached size, not the fetch closure's"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "cache hit must not call fetch again"
    );
}

/// `FitsCache`: concurrent callers for the *same* filename single-flight —
/// only one of them actually calls `fetch`, the rest block on its result.
#[test]
fn fits_cache_single_flights_concurrent_fetches_of_the_same_key() {
    let directory = temp_dir("fits-cache-single-flight");
    let cache = Arc::new(FitsCache::new(&directory).expect("create cache"));
    let calls = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(std::sync::Barrier::new(8));

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let cache = cache.clone();
            let calls = calls.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                cache.get_or_fetch("shared.fits", || -> Result<Vec<u8>, String> {
                    calls.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    Ok(vec![0u8; 128])
                })
            })
        })
        .collect();

    for handle in handles {
        let bytes = handle.join().expect("thread join").expect("fetch result");
        assert_eq!(bytes, 128, "every caller observes the fetched file's size");
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "only one caller actually fetches"
    );
}

/// `FitsCache`: eviction removes the least-recently-touched file once the
/// cache exceeds its byte budget.
#[test]
fn fits_cache_evicts_least_recently_used_file_over_budget() {
    let directory = temp_dir("fits-cache-lru");
    let cache = FitsCache::with_max_bytes(&directory, 150).expect("create bounded cache");

    cache
        .get_or_fetch("old.fits", || -> Result<Vec<u8>, String> {
            Ok(vec![0u8; 100])
        })
        .expect("fetch old");
    // Ensure a distinguishable mtime ordering on filesystems with coarse
    // mtime resolution.
    std::thread::sleep(std::time::Duration::from_millis(1_100));
    cache
        .get_or_fetch("new.fits", || -> Result<Vec<u8>, String> {
            Ok(vec![0u8; 100])
        })
        .expect("fetch new");

    assert!(
        !directory.join("old.fits").exists(),
        "oldest file evicted once over budget"
    );
    assert!(
        directory.join("new.fits").exists(),
        "most recently fetched file survives eviction"
    );
}
