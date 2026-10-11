//! Ported from `testDurableOutbox`, `testAtomicOutboxClaim`,
//! `testOutboxRetryDelivery`, `testOutboxUnknownDelivery` in
//! `Sources/ExoNitContractTests/main.swift` (Swift reference worker).
//! `Date(timeIntervalSince1970: N)` becomes the plain `f64` `N.0` throughout
//! — see `outbox.rs`'s module doc comment for why.
//!
//! F8 (2026-08-05, `docs/PLAN.md`, outbox auto-reconciliation fix): the
//! Swift-ported `testOutboxUnknownDelivery` case ("a transport failure parks
//! the entry in a terminal 'unknown' state") no longer matches this crate's
//! behaviour — `WorkerResultOutbox::drain` now treats a transport failure as
//! a normal retryable `mark_failed`, never `mark_delivery_unknown`. It has
//! been replaced below by `outbox_transport_error_becomes_pending_retry_not_unknown`
//! (new behaviour) and `legacy_unknown_entries_are_reconciled_to_pending`
//! (coverage for the 'unknown' state as a legacy-only, still-supported
//! artifact of pre-fix outboxes).

#[path = "support/mod.rs"]
mod support;

use exonit_core::api::WorkerApiClient;
use exonit_core::dto::PrimaryResultRequest;
use exonit_core::outbox::{DurableOutbox, RetryDisposition, WorkerResultOutbox};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use support::{MockRequest, MockResponse, MockServer};

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("exonit-rust-{name}-{}", std::process::id()))
        .join(format!(
            "{:x}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Swift: `testDurableOutbox`.
#[test]
fn durable_outbox_lifecycle() {
    let dir = temp_dir("outbox-basic");
    let database = dir.join("outbox.sqlite");
    let outbox = DurableOutbox::new(&database).expect("open outbox");
    let payload = br#"{"task_id":"task-1","status":"completed"}"#;
    outbox
        .enqueue(
            "primary:task-1",
            "/api/worker/result",
            "wid",
            payload,
            100.0,
        )
        .expect("enqueue 1");
    outbox
        .enqueue(
            "primary:task-1",
            "/api/worker/result",
            "wid",
            payload,
            100.0,
        )
        .expect("enqueue 2 (idempotent)");
    let first = outbox.pending(100.0, 10).expect("pending 1");
    assert_eq!(first.len(), 1, "outbox idempotent enqueue");
    assert_eq!(first[0].attempts, 0, "outbox idempotent enqueue");
    let first = outbox
        .claim_pending("test-owner", 100.0, 60.0, 10)
        .expect("claim first");
    outbox
        .mark_failed(
            &first[0].id,
            "test-owner",
            "timeout",
            100.0,
            RetryDisposition::Transient,
        )
        .expect("mark failed");
    let immediate = outbox.pending(100.0, 10).expect("pending immediate");
    assert!(immediate.is_empty(), "outbox backoff");

    let reopened = DurableOutbox::new(&database).expect("reopen outbox");
    let retry = reopened.pending(103.0, 10).expect("pending after backoff");
    assert_eq!(retry.len(), 1, "outbox survives reopen");
    assert_eq!(retry[0].attempts, 1, "outbox survives reopen");
    // The original claim owner is not persisted across this test's direct
    // `pending` inspection; claim the retry before completing it.
    let retry = reopened
        .claim_pending("test-owner", 103.0, 60.0, 10)
        .expect("claim retry");
    reopened
        .mark_delivered(&retry[0].id, "test-owner")
        .expect("mark delivered");
    let empty = reopened.pending(200.0, 10).expect("pending after delivery");
    assert!(empty.is_empty(), "outbox delivered deletion");
}

/// Swift: `testAtomicOutboxClaim`.
#[test]
fn atomic_outbox_claim() {
    let dir = temp_dir("outbox-claim");
    let database = dir.join("outbox.sqlite");
    let first = DurableOutbox::new(&database).expect("open outbox 1");
    let second = DurableOutbox::new(&database).expect("open outbox 2");
    first
        .enqueue("primary:claim", "/api/worker/result", "w", b"{}", 10.0)
        .expect("enqueue");

    let first = Arc::new(first);
    let second = Arc::new(second);
    let first_clone = first.clone();
    let second_clone = second.clone();
    let claim_a = std::thread::spawn(move || first_clone.claim_pending("a", 10.0, 60.0, 32));
    let claim_b = std::thread::spawn(move || second_clone.claim_pending("b", 10.0, 60.0, 32));
    let claims_a = claim_a.join().expect("join a").expect("claim a");
    let claims_b = claim_b.join().expect("join b").expect("claim b");
    assert_eq!(
        claims_a.len() + claims_b.len(),
        1,
        "entry atomically claimed once"
    );

    let during_lease = first.pending(11.0, 32).expect("pending during lease");
    assert!(during_lease.is_empty(), "leased entry hidden from pending");
    let after_lease = first.pending(71.0, 32).expect("pending after lease expiry");
    assert_eq!(after_lease.len(), 1, "expired lease becomes pending");
}

#[test]
fn completion_requires_current_lease_owner() {
    let dir = temp_dir("outbox-lease-owner");
    let outbox = DurableOutbox::new(&dir.join("outbox.sqlite")).expect("open outbox");
    outbox
        .enqueue("primary:lease", "/api/worker/result", "wid", b"{}", 10.0)
        .expect("enqueue");
    let entry = outbox
        .claim_pending("owner-a", 10.0, 60.0, 1)
        .expect("claim")
        .remove(0);

    assert!(
        outbox.mark_delivered(&entry.id, "owner-b").is_err(),
        "another owner cannot acknowledge"
    );
    assert!(
        outbox
            .mark_failed(
                &entry.id,
                "owner-b",
                "not mine",
                10.0,
                RetryDisposition::Transient
            )
            .is_err(),
        "another owner cannot release the lease"
    );
    outbox
        .mark_failed(
            &entry.id,
            "owner-a",
            "temporary",
            10.0,
            RetryDisposition::Transient,
        )
        .expect("current owner can release the lease");
    assert_eq!(outbox.pending(13.0, 1).expect("pending retry").len(), 1);
}

#[test]
fn permanent_failure_is_not_retried() {
    let dir = temp_dir("outbox-permanent-failure");
    let outbox = DurableOutbox::new(&dir.join("outbox.sqlite")).expect("open outbox");
    outbox
        .enqueue("primary:bad", "/api/worker/result", "wid", b"{}", 10.0)
        .expect("enqueue");
    let entry = outbox
        .claim_pending("owner", 10.0, 60.0, 1)
        .expect("claim")
        .remove(0);
    outbox
        .mark_failed(
            &entry.id,
            "owner",
            "HTTP 400: invalid request",
            10.0,
            RetryDisposition::Permanent,
        )
        .expect("mark permanent failure");

    assert!(
        outbox.pending(10_000.0, 1).expect("pending").is_empty(),
        "permanent failures are terminal"
    );
    assert!(outbox
        .claim_pending("other", 10_000.0, 60.0, 1)
        .expect("claim")
        .is_empty());
    assert_eq!(
        outbox.count().expect("count"),
        0,
        "terminal entry does not block delivery"
    );
    assert_eq!(
        outbox.permanent_failure_count().expect("permanent count"),
        1,
        "terminal entry remains inspectable"
    );
}

/// Swift: `testOutboxRetryDelivery`.
#[test]
fn outbox_retry_delivery() {
    let dir = temp_dir("outbox-retry");
    let outbox = DurableOutbox::new(&dir.join("outbox.sqlite")).expect("open outbox");
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_for_handler = counter.clone();
    let server = MockServer::start(move |request: &MockRequest| {
        assert_eq!(request.path, "/api/worker/result", "outbox result path");
        assert_eq!(
            request.header("X-Worker-ID"),
            Some("wid"),
            "outbox worker header"
        );
        assert_eq!(
            request.header("Idempotency-Key"),
            Some("primary:task-retry"),
            "outbox idempotency header"
        );
        let call = counter_for_handler.fetch_add(1, Ordering::SeqCst) + 1;
        if call == 1 {
            MockResponse::json(500, r#"{"error":"temporary"}"#)
        } else {
            MockResponse::json(200, r#"{"status":"ok"}"#)
        }
    });
    let api = WorkerApiClient::new(server.base_url, "token");
    let sender = WorkerResultOutbox::new(Arc::new(outbox), Arc::new(api));
    let payload =
        PrimaryResultRequest::new(Some("task-retry".to_string()), Some("123".to_string()));
    sender
        .enqueue_primary(&payload, "wid", 100.0)
        .expect("enqueue primary");

    let failed = sender.drain(100.0, 32);
    assert_eq!(
        (failed.delivered, failed.failed),
        (0, 1),
        "first outbox delivery fails"
    );

    let delivered = sender.drain(103.0, 32);
    assert_eq!(
        (delivered.delivered, delivered.failed),
        (1, 0),
        "outbox retry succeeds"
    );
}

/// New coverage (F8, 2026-08-05, `docs/PLAN.md`): a transport failure — the
/// exact real-world case that killed 15 fleet worker instances on
/// 2026-08-05 (server may have committed the result, but the response was
/// lost) — must become a normal retryable 'pending' entry with backoff, not
/// the terminal 'unknown' state `testOutboxUnknownDelivery` (Swift) used to
/// assert. Safe because the server's result endpoints are idempotent
/// upserts (see `outbox.rs::WorkerResultOutbox::drain`'s doc comment for the
/// citations): worst case, retrying re-upserts the same values.
#[test]
fn outbox_transport_error_becomes_pending_retry_not_unknown() {
    let dir = temp_dir("outbox-transport-retry");
    let outbox = Arc::new(DurableOutbox::new(&dir.join("outbox.sqlite")).expect("open outbox"));
    let attempt = Arc::new(AtomicUsize::new(0));
    let attempt_for_handler = attempt.clone();
    // First connection: accept then drop without responding, forcing a
    // `ureq` transport error (same technique the old unknown-delivery test
    // used). Second connection onward: behave like a normal single-request
    // `Connection: close` mock server responding 200, so the *same*
    // `WorkerApiClient` (bound to this one port) can observe a successful
    // retry without needing a second server.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let call = attempt_for_handler.fetch_add(1, Ordering::SeqCst) + 1;
            if call == 1 {
                drop(stream);
                continue;
            }
            use std::io::{BufRead, BufReader, Read, Write};
            let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
            let mut request_line = String::new();
            let _ = reader.read_line(&mut request_line);
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    break;
                }
                let trimmed = line.trim_end_matches(['\r', '\n']);
                if trimmed.is_empty() {
                    break;
                }
                if let Some((name, value)) = trimmed.split_once(':') {
                    if name.trim().eq_ignore_ascii_case("content-length") {
                        content_length = value.trim().parse().unwrap_or(0);
                    }
                }
            }
            let mut body = vec![0u8; content_length];
            if content_length > 0 {
                let _ = reader.read_exact(&mut body);
            }
            let response_body: &[u8] = br#"{"status":"ok"}"#;
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                response_body.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(response_body);
            let _ = stream.flush();
        }
    });

    let api = Arc::new(WorkerApiClient::new(
        format!("http://127.0.0.1:{port}"),
        "token",
    ));
    let sender = WorkerResultOutbox::new(outbox.clone(), api);
    sender
        .enqueue_primary(
            &PrimaryResultRequest::new(Some("task-transport".to_string()), Some("789".to_string())),
            "wid",
            100.0,
        )
        .expect("enqueue primary");

    let first = sender.drain(100.0, 32);
    assert_eq!(
        (first.delivered, first.failed),
        (0, 1),
        "transport error is a retryable failure, not fatal"
    );
    assert_eq!(
        outbox.delivery_unknown_count().expect("unknown count"),
        0,
        "no legacy 'unknown' state is produced"
    );
    assert!(
        outbox
            .pending(101.0, 32)
            .expect("pending before backoff")
            .is_empty(),
        "backoff delays the retry"
    );

    let retry_due = outbox.pending(103.0, 32).expect("pending after backoff");
    assert_eq!(retry_due.len(), 1, "entry becomes due again after backoff");
    assert_eq!(
        retry_due[0].attempts, 1,
        "attempts incremented on transport-ambiguous retry"
    );
    assert!(
        retry_due[0]
            .last_error
            .as_deref()
            .unwrap_or("")
            .contains("transport-ambiguous"),
        "last_error documents why the retry happened: {:?}",
        retry_due[0].last_error
    );

    let second = sender.drain(103.0, 32);
    assert_eq!(
        (second.delivered, second.failed),
        (1, 0),
        "retry succeeds once the server is reachable"
    );
    assert_eq!(
        attempt.load(Ordering::SeqCst),
        2,
        "exactly two delivery attempts were made"
    );
}

/// New coverage (F8, 2026-08-05, `docs/PLAN.md`): the `'unknown'`
/// `delivery_state` and `mark_delivery_unknown` remain legacy-supported for
/// outboxes written by pre-fix worker binaries, but the current code never
/// produces them (see the test above) — `reconcile_legacy_unknown` is what
/// upgrades any such inherited entry back onto the normal retry path.
#[test]
fn legacy_unknown_entries_are_reconciled_to_pending() {
    let dir = temp_dir("outbox-legacy-unknown");
    let outbox = DurableOutbox::new(&dir.join("outbox.sqlite")).expect("open outbox");
    outbox
        .enqueue(
            "primary:task-legacy",
            "/api/worker/result",
            "wid",
            b"{}",
            100.0,
        )
        .expect("enqueue");
    let entry = outbox.pending(100.0, 10).expect("pending")[0].clone();
    outbox
        .mark_delivery_unknown(&entry.id, "legacy timeout")
        .expect("mark delivery unknown (legacy path)");
    assert_eq!(
        outbox.delivery_unknown_count().expect("unknown count"),
        1,
        "legacy 'unknown' state still persists"
    );
    assert!(
        outbox
            .pending(200.0, 10)
            .expect("pending while unknown")
            .is_empty(),
        "unknown entries are not due for delivery"
    );

    let reconciled = outbox
        .reconcile_legacy_unknown(500.0)
        .expect("reconcile legacy unknown");
    assert_eq!(reconciled, 1, "one legacy entry reconciled");
    assert_eq!(
        outbox
            .delivery_unknown_count()
            .expect("unknown count after reconcile"),
        0,
        "no unknown entries remain"
    );
    let due = outbox.pending(500.0, 10).expect("pending after reconcile");
    assert_eq!(
        due.len(),
        1,
        "reconciled entry is immediately due for retry"
    );
    assert_eq!(due[0].id, entry.id, "same entry, now retryable");
}
