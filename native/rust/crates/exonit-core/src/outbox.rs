//! Transliterated from `Sources/ExoNitCore/DurableOutbox.swift` +
//! `Sources/ExoNitCore/WorkerResultOutbox.swift` in the Swift reference
//! worker (`~/exonit-swift-worker-dev`, commit `5e3978e`).
//!
//! `DurableOutbox` (the raw SQLite-backed queue) via `rusqlite` with feature
//! `"bundled"` — the only SQLite crate in the F0b dependency allowlist (see
//! docs/DESIGN.md). Swift's `DurableOutbox` is an `actor`, i.e. all access is
//! serialized through Swift's cooperative-concurrency runtime; this crate has
//! no async runtime (CLAUDE.md rule: threads, not tokio/async), so the same
//! serialization is done with a plain `std::sync::Mutex<rusqlite::Connection>`
//! — any thread calling a `DurableOutbox` method blocks until it can take the
//! lock, which is exactly what an actor's serial executor gives you for a
//! single-instance actor with no internal `await` points between mutations
//! (true here: every `DurableOutbox` method is one blocking SQLite statement
//! or one transaction, with no yield in between).
//!
//! Two representational choices differ from Swift purely because Rust has no
//! `Foundation.Date`:
//!   - `Date`/`TimeInterval` become a plain `f64` Unix-epoch-seconds
//!     timestamp everywhere (`Date(timeIntervalSince1970:)` in the Swift
//!     contract tests is exactly this `f64` already, so test values carry
//!     over unchanged: `Date(timeIntervalSince1970: 100)` -> `100.0`).
//!   - `markFailed`'s backoff, `next_attempt_at = now + MIN(300, POW(2.0,
//!     attempts + 1))`, used SQLite's `POW`/`MIN` scalar functions in Swift.
//!     The bundled SQLite here is compiled *without*
//!     `SQLITE_ENABLE_MATH_FUNCTIONS` (verified empirically: `SELECT
//!     POW(2.0,3)` raises "no such function: POW"), so the same formula is
//!     computed in Rust host code from the pre-update `attempts` value
//!     (fetched by the same `SELECT` a hand-written `POW` workaround would
//!     have needed anyway) rather than inside the `UPDATE` statement. The
//!     numeric result is identical for every input this code will ever see;
//!     `f64::powi` also degrades to `+inf` instead of overflowing for
//!     absurdly large `attempts`, which `.min(300.0)` still caps correctly
//!     (a robustness margin the raw SQL expression didn't have).
//!
//! `WorkerResultOutbox` (the outbox <-> `WorkerApiClient` drain loop) is the
//! second half of this file, exactly mirroring the Swift split into two
//! files — kept together here because nothing outside this module needs
//! `DurableOutbox` on its own except `WorkerResultOutbox` and this crate's
//! tests.

use crate::api::{WorkerApiClient, WorkerApiError};
use crate::dto::{ExoMinerResultRequest, FppResultRequest, PrimaryResultRequest};
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Returns the current Unix-epoch time in seconds as `f64`, the same
/// representation `Date().timeIntervalSince1970` produces in Swift.
pub fn now_unix() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// Mirrors `OutboxEntry`.
#[derive(Debug, Clone, PartialEq)]
pub struct OutboxEntry {
    pub id: String,
    pub endpoint: String,
    pub worker_id: String,
    pub payload: Vec<u8>,
    pub created_at: f64,
    pub attempts: i64,
    pub next_attempt_at: f64,
    pub last_error: Option<String>,
}

/// Mirrors `OutboxError`.
#[derive(Debug)]
pub enum OutboxError {
    Sqlite {
        code: i32,
        message: String,
    },
    /// The row was deleted or its lease was reclaimed by another worker
    /// before this owner completed delivery.  Treating this as success would
    /// let one worker acknowledge another worker's delivery.
    LeaseLost {
        id: String,
    },
}

impl std::fmt::Display for OutboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutboxError::Sqlite { code, message } => write!(f, "SQLite {code}: {message}"),
            OutboxError::LeaseLost { id } => write!(f, "outbox lease lost for entry {id}"),
        }
    }
}
impl std::error::Error for OutboxError {}

/// Whether a failed delivery may be retried.
///
/// Permanent failures are retained in the database as `delivery_state =
/// 'failed'` for inspection, but are not selected by `pending`/`claim_pending`.
/// Keeping the row avoids silently losing the payload while ensuring a bad
/// request (for example an HTTP 4xx) cannot block the worker forever.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryDisposition {
    Transient,
    Permanent,
}

/// Short compatibility alias for callers that use the more concise name.
pub type RetryClass = RetryDisposition;

/// `SQLITE_MISUSE` (21) — used for the same "empty key/endpoint/worker ID"
/// guard Swift raises with `code: SQLITE_MISUSE`.
const SQLITE_MISUSE: i32 = 21;

impl From<rusqlite::Error> for OutboxError {
    fn from(error: rusqlite::Error) -> Self {
        // rusqlite already carries the extended SQLite error code where one
        // exists; fall back to a sentinel (-1, matches nothing SQLite itself
        // returns) for errors rusqlite raises before ever reaching the C API
        // (e.g. type conversion failures), mirroring how the Swift `error(code:)`
        // helper always has a concrete `Int32` because every failure path
        // there originates from an actual `sqlite3_*` return code.
        let code = error.sqlite_error_code().map(|c| c as i32).unwrap_or(-1);
        OutboxError::Sqlite {
            code,
            message: error.to_string(),
        }
    }
}

/// Mirrors `DurableOutbox`.
pub struct DurableOutbox {
    connection: Mutex<Connection>,
}

impl DurableOutbox {
    /// Mirrors `DurableOutbox.init(databaseURL:)`.
    pub fn new(database_path: &Path) -> Result<Self, OutboxError> {
        if let Some(parent) = database_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| OutboxError::Sqlite {
                code: -1,
                message: format!("creating outbox directory: {e}"),
            })?;
        }
        let connection = Connection::open_with_flags(
            database_path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
        )?;
        connection.busy_timeout(std::time::Duration::from_millis(5_000))?;
        connection.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;")?;
        connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS outbox (
              id TEXT PRIMARY KEY NOT NULL,
              endpoint TEXT NOT NULL,
              worker_id TEXT NOT NULL,
              payload BLOB NOT NULL,
              created_at REAL NOT NULL,
              attempts INTEGER NOT NULL DEFAULT 0,
              next_attempt_at REAL NOT NULL,
              last_error TEXT,
              lease_owner TEXT,
              lease_until REAL,
              delivery_state TEXT NOT NULL DEFAULT 'pending'
            );",
        )?;
        ensure_column(&connection, "lease_owner", "TEXT")?;
        ensure_column(&connection, "lease_until", "REAL")?;
        ensure_column(
            &connection,
            "delivery_state",
            "TEXT NOT NULL DEFAULT 'pending'",
        )?;
        connection.execute_batch(
            "CREATE INDEX IF NOT EXISTS outbox_due ON outbox(next_attempt_at, created_at);",
        )?;
        Ok(DurableOutbox {
            connection: Mutex::new(connection),
        })
    }

    /// Mirrors `DurableOutbox.enqueue(idempotencyKey:endpoint:workerID:payload:now:)`.
    pub fn enqueue(
        &self,
        idempotency_key: &str,
        endpoint: &str,
        worker_id: &str,
        payload: &[u8],
        now: f64,
    ) -> Result<(), OutboxError> {
        if idempotency_key.is_empty() || endpoint.is_empty() || worker_id.is_empty() {
            return Err(OutboxError::Sqlite {
                code: SQLITE_MISUSE,
                message: "empty outbox key, endpoint or worker ID".to_string(),
            });
        }
        let connection = self.lock();
        connection.execute(
            "INSERT OR IGNORE INTO outbox
               (id, endpoint, worker_id, payload, created_at, attempts, next_attempt_at)
             VALUES (?1, ?2, ?3, ?4, ?5, 0, ?6);",
            params![idempotency_key, endpoint, worker_id, payload, now, now],
        )?;
        Ok(())
    }

    /// Mirrors `DurableOutbox.pending(now:limit:)`.
    pub fn pending(&self, now: f64, limit: i64) -> Result<Vec<OutboxEntry>, OutboxError> {
        if limit <= 0 {
            return Ok(Vec::new());
        }
        let connection = self.lock();
        query_pending(&connection, now, limit)
    }

    /// Mirrors `DurableOutbox.claimPending(owner:now:leaseDuration:limit:)`:
    /// atomically leases up to `limit` due entries to `owner` inside a single
    /// `BEGIN IMMEDIATE` transaction, so two `DurableOutbox` handles racing on
    /// the same on-disk database (as in `testAtomicOutboxClaim`) can never
    /// both claim the same row.
    pub fn claim_pending(
        &self,
        owner: &str,
        now: f64,
        lease_duration_seconds: f64,
        limit: i64,
    ) -> Result<Vec<OutboxEntry>, OutboxError> {
        if owner.is_empty() || lease_duration_seconds <= 0.0 || limit <= 0 {
            return Ok(Vec::new());
        }
        let connection = self.lock();
        connection.execute_batch("BEGIN IMMEDIATE;")?;
        match claim_pending_locked(&connection, owner, now, lease_duration_seconds, limit) {
            Ok(entries) => {
                connection.execute_batch("COMMIT;")?;
                Ok(entries)
            }
            Err(error) => {
                let _ = connection.execute_batch("ROLLBACK;");
                Err(error)
            }
        }
    }

    /// Deletes a delivered entry, but only when `owner` still owns its lease.
    /// A zero-row update is an error: it means the lease expired/reassigned
    /// (or the row was already removed), so this caller must not acknowledge
    /// another worker's delivery.
    pub fn mark_delivered(&self, id: &str, owner: &str) -> Result<(), OutboxError> {
        let connection = self.lock();
        let changed = connection.execute(
            "DELETE FROM outbox WHERE id = ?1 AND lease_owner = ?2;",
            params![id, owner],
        )?;
        if changed == 0 {
            return Err(OutboxError::LeaseLost { id: id.to_string() });
        }
        Ok(())
    }

    /// Records a delivery failure, guarded by the current lease owner.  A
    /// transient failure returns the row to `pending` with exponential
    /// backoff; a permanent failure is retained as terminal `failed` state.
    /// See the module doc comment for why backoff is computed in Rust.
    pub fn mark_failed(
        &self,
        id: &str,
        owner: &str,
        message: &str,
        now: f64,
        disposition: RetryDisposition,
    ) -> Result<(), OutboxError> {
        let connection = self.lock();
        let attempts: Option<i64> = connection
            .query_row(
                "SELECT attempts FROM outbox WHERE id = ?1 AND lease_owner = ?2;",
                params![id, owner],
                |row| row.get(0),
            )
            .optional()?;
        let Some(attempts) = attempts else {
            return Err(OutboxError::LeaseLost { id: id.to_string() });
        };
        let (delivery_state, next_attempt_at) = match disposition {
            RetryDisposition::Transient => {
                let backoff = 2f64.powi((attempts + 1) as i32).min(300.0);
                ("pending", now + backoff)
            }
            RetryDisposition::Permanent => ("failed", now),
        };
        let truncated_message: String = message.chars().take(2_048).collect();
        let changed = connection.execute(
            "UPDATE outbox SET attempts = attempts + 1, next_attempt_at = ?1,
               last_error = ?2, delivery_state = ?3, lease_owner = NULL,
               lease_until = NULL WHERE id = ?4 AND lease_owner = ?5;",
            params![
                next_attempt_at,
                truncated_message,
                delivery_state,
                id,
                owner
            ],
        )?;
        if changed == 0 {
            return Err(OutboxError::LeaseLost { id: id.to_string() });
        }
        Ok(())
    }

    /// Mirrors `DurableOutbox.markDeliveryUnknown(id:error:)`. **Legacy only**
    /// as of the F8 outbox auto-reconciliation fix (2026-08-05, see
    /// `docs/PLAN.md`): kept so outbox databases written by pre-fix worker
    /// binaries still decode (the `delivery_state` column can already
    /// contain `'unknown'` rows), and as a low-level primitive callers may
    /// still reach for directly, but nothing in this crate's live code paths
    /// calls it anymore — `WorkerResultOutbox::drain`'s transport-error
    /// branch now calls [`Self::mark_failed`] instead (see its doc comment)
    /// so a stalled/ambiguous delivery becomes a normal retryable 'pending'
    /// entry rather than the terminal 'unknown' state this method produces.
    /// [`Self::reconcile_legacy_unknown`] is what upgrades any 'unknown' row
    /// (whether produced by this method or inherited from an old database)
    /// back onto the normal retry path.
    pub fn mark_delivery_unknown(&self, id: &str, message: &str) -> Result<(), OutboxError> {
        let connection = self.lock();
        let truncated_message: String = message.chars().take(2_048).collect();
        connection.execute(
            "UPDATE outbox SET delivery_state = 'unknown', last_error = ?1,
               lease_owner = NULL, lease_until = NULL WHERE id = ?2;",
            params![truncated_message, id],
        )?;
        Ok(())
    }

    /// F8 (2026-08-05, `docs/PLAN.md`): converts every legacy
    /// `delivery_state = 'unknown'` row back to `'pending'`, due immediately
    /// (`next_attempt_at = now`), clearing any stale lease. Safe because the
    /// server's result-ingestion endpoints are idempotent upserts, so
    /// re-delivering an entry whose outcome was merely *unknown* (not
    /// known-bad) can only re-apply the same values in the worst case — see
    /// `~/exonit/exonit/routes/workers.py` `POST /api/worker/result`
    /// (~L951, unconditional `db.add_star` upsert), `exonit/db.py::add_candidate`
    /// (upsert keyed by `(tic_id, candidate_num, sector)`, refreshes measured
    /// transit parameters without erasing later analyses), and
    /// `db.py::complete_task` (idempotent `UPDATE`). Returns the number of
    /// rows reconciled. Intended to run once, at outbox open / runtime
    /// startup (see `runtime.rs::WorkerRuntime::build`) — new code never
    /// produces `'unknown'` rows for this to find in steady state (see
    /// [`Self::mark_delivery_unknown`]'s doc comment), so this is purely a
    /// one-time upgrade path for outboxes that predate the fix.
    pub fn reconcile_legacy_unknown(&self, now: f64) -> Result<i64, OutboxError> {
        let connection = self.lock();
        // Manual/operator reconciliation may leave rows in 'delivered' (the
        // code's own success path DELETEs instead). `count()` counts them and
        // the run loop refuses new work while the outbox is non-empty — 14
        // Swift production instances idled for hours on 2026-08-05 over this.
        // Purge them before reviving 'unknown' rows (mirror of the Swift fix).
        connection.execute("DELETE FROM outbox WHERE delivery_state = 'delivered';", [])?;
        let reconciled = connection.execute(
            "UPDATE outbox SET delivery_state = 'pending', next_attempt_at = ?1,
               lease_owner = NULL, lease_until = NULL WHERE delivery_state = 'unknown';",
            params![now],
        )?;
        Ok(reconciled as i64)
    }

    /// Mirrors `DurableOutbox.pendingCount()`.
    pub fn pending_count(&self) -> Result<i64, OutboxError> {
        self.count_where("delivery_state = 'pending'")
    }

    /// Mirrors `DurableOutbox.deliveryUnknownCount()`. Legacy observability
    /// only since F8 — see [`Self::mark_delivery_unknown`]'s doc comment;
    /// should read 0 in steady state after startup reconciliation.
    pub fn delivery_unknown_count(&self) -> Result<i64, OutboxError> {
        self.count_where("delivery_state = 'unknown'")
    }

    /// Counts actionable entries. Terminal `failed` rows remain available for
    /// inspection but must not keep the runtime's "drain before new work"
    /// guard spinning forever.
    pub fn count(&self) -> Result<i64, OutboxError> {
        let connection = self.lock();
        Ok(connection.query_row(
            "SELECT COUNT(*) FROM outbox WHERE delivery_state IN ('pending', 'unknown');",
            [],
            |row| row.get(0),
        )?)
    }

    /// Number of terminal permanent failures retained for inspection.
    pub fn permanent_failure_count(&self) -> Result<i64, OutboxError> {
        self.count_where("delivery_state = 'failed'")
    }

    fn count_where(&self, predicate: &str) -> Result<i64, OutboxError> {
        let connection = self.lock();
        Ok(connection.query_row(
            &format!("SELECT COUNT(*) FROM outbox WHERE {predicate};"),
            [],
            |row| row.get(0),
        )?)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Connection> {
        // A poisoned lock means some earlier caller panicked mid-statement;
        // there is no safe recovery, so propagate the panic rather than
        // silently operating on a connection that might be mid-transaction.
        self.connection
            .lock()
            .expect("DurableOutbox connection mutex poisoned")
    }
}

fn query_pending(
    connection: &Connection,
    now: f64,
    limit: i64,
) -> Result<Vec<OutboxEntry>, OutboxError> {
    let mut statement = connection.prepare(
        "SELECT id, endpoint, worker_id, payload, created_at, attempts, next_attempt_at, last_error
         FROM outbox
         WHERE delivery_state = 'pending' AND next_attempt_at <= ?1
           AND (lease_until IS NULL OR lease_until <= ?2)
         ORDER BY created_at, id LIMIT ?3;",
    )?;
    let rows = statement.query_map(params![now, now, limit], |row| {
        Ok(OutboxEntry {
            id: row.get(0)?,
            endpoint: row.get(1)?,
            worker_id: row.get(2)?,
            payload: row.get(3)?,
            created_at: row.get(4)?,
            attempts: row.get(5)?,
            next_attempt_at: row.get(6)?,
            last_error: row.get(7)?,
        })
    })?;
    let mut entries = Vec::new();
    for row in rows {
        entries.push(row?);
    }
    Ok(entries)
}

fn claim_pending_locked(
    connection: &Connection,
    owner: &str,
    now: f64,
    lease_duration_seconds: f64,
    limit: i64,
) -> Result<Vec<OutboxEntry>, OutboxError> {
    let lease_until = now + lease_duration_seconds;
    // Select and lease in one UPDATE statement.  In particular, do not first
    // return rows from SELECT and then assume each UPDATE succeeded: a lease
    // may have been reclaimed between those operations when multiple
    // processes share the database.  SQLite's RETURNING rows are exactly the
    // rows that this UPDATE changed, so the caller can never send an entry it
    // did not actually claim.
    let mut statement = connection.prepare(
        "UPDATE outbox SET lease_owner = ?1, lease_until = ?2
         WHERE id IN (
             SELECT id FROM outbox
             WHERE delivery_state = 'pending' AND next_attempt_at <= ?3
               AND (lease_until IS NULL OR lease_until <= ?3)
             ORDER BY created_at, id LIMIT ?4
         )
           AND delivery_state = 'pending'
           AND next_attempt_at <= ?3
           AND (lease_until IS NULL OR lease_until <= ?3)
         RETURNING id, endpoint, worker_id, payload, created_at, attempts,
                   next_attempt_at, last_error;",
    )?;
    let rows = statement.query_map(params![owner, lease_until, now, limit], |row| {
        Ok(OutboxEntry {
            id: row.get(0)?,
            endpoint: row.get(1)?,
            worker_id: row.get(2)?,
            payload: row.get(3)?,
            created_at: row.get(4)?,
            attempts: row.get(5)?,
            next_attempt_at: row.get(6)?,
            last_error: row.get(7)?,
        })
    })?;
    let mut entries = Vec::new();
    for row in rows {
        entries.push(row?);
    }
    Ok(entries)
}

fn ensure_column(
    connection: &Connection,
    name: &str,
    declaration: &str,
) -> Result<(), OutboxError> {
    let mut statement = connection.prepare("PRAGMA table_info(outbox);")?;
    let mut existing = statement.query([])?;
    while let Some(row) = existing.next()? {
        let column_name: String = row.get(1)?;
        if column_name == name {
            return Ok(());
        }
    }
    drop(existing);
    drop(statement);
    connection.execute_batch(&format!(
        "ALTER TABLE outbox ADD COLUMN {name} {declaration};"
    ))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// WorkerResultOutbox — mirrors Sources/ExoNitCore/WorkerResultOutbox.swift
// ---------------------------------------------------------------------------

/// Mirrors `OutboxDrainResult`. F8 (2026-08-05): dropped the `delivery_unknown`
/// field — `drain` no longer ever produces the 'unknown' delivery state (see
/// `DurableOutbox::mark_delivery_unknown`'s doc comment), so a transport
/// failure now counts as an ordinary retryable `failed` the same as an HTTP
/// error response; `last_error` on the entry still distinguishes the two
/// causes for anyone inspecting the outbox directly.
#[derive(Debug, Clone, PartialEq)]
pub struct OutboxDrainResult {
    pub delivered: i64,
    pub failed: i64,
    pub scan_error: Option<String>,
}

/// Error surface for `WorkerResultOutbox`'s enqueue methods. Swift's
/// `enqueuePrimary`/`enqueueFPP`/`enqueueExoMiner` are `throws` and let
/// either a `WorkerAPIError.malformedResponse` (missing task/candidate id),
/// a `JSONEncoder` `EncodingError`, or an `OutboxError` (SQLite failure)
/// propagate untyped; Rust needs one concrete type, so this enum keeps the
/// three failure modes distinguishable for callers instead of collapsing
/// them into a single stringly-typed error.
#[derive(Debug)]
pub enum ResultOutboxError {
    Api(WorkerApiError),
    Outbox(OutboxError),
    Encode(String),
}

impl std::fmt::Display for ResultOutboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResultOutboxError::Api(error) => write!(f, "{error}"),
            ResultOutboxError::Outbox(error) => write!(f, "{error}"),
            ResultOutboxError::Encode(message) => write!(f, "encoding outbox payload: {message}"),
        }
    }
}
impl std::error::Error for ResultOutboxError {}

impl From<WorkerApiError> for ResultOutboxError {
    fn from(error: WorkerApiError) -> Self {
        ResultOutboxError::Api(error)
    }
}
impl From<OutboxError> for ResultOutboxError {
    fn from(error: OutboxError) -> Self {
        ResultOutboxError::Outbox(error)
    }
}

/// Monotonic per-process counter backing `generate_lease_owner`. Swift uses
/// `UUID().uuidString`; `uuid` is not in the F0b dependency allowlist
/// (ureq + rusqlite only), so this crate mints a lease owner tag from
/// pid + wall-clock time + a process-local sequence number instead. A lease
/// owner only needs to be distinguishable from *other concurrently running*
/// `WorkerResultOutbox` instances (there is exactly one per worker process,
/// per `WorkerRuntime::new`) — collision-resistance at UUID's scale is not a
/// requirement this identifier has to meet.
static LEASE_OWNER_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn generate_lease_owner() -> String {
    let sequence = LEASE_OWNER_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!("pid{}-{:.6}-{}", std::process::id(), now_unix(), sequence)
}

pub fn classify_api_failure(error: &WorkerApiError) -> RetryDisposition {
    match error {
        // A request timeout, rate limit, server failure, or an intermediary
        // timeout is expected to recover on a later attempt.  Other HTTP
        // statuses are request/auth/data errors and retrying them unchanged
        // cannot make them succeed.
        WorkerApiError::HttpStatus(code, _) if matches!(*code, 408 | 425 | 429 | 500..=599) => {
            RetryDisposition::Transient
        }
        WorkerApiError::HttpStatus(_, _) => RetryDisposition::Permanent,
        WorkerApiError::InvalidResponse(_)
        | WorkerApiError::MalformedResponse(_)
        | WorkerApiError::Transport(_) => RetryDisposition::Transient,
    }
}

#[cfg(test)]
mod tests {
    use super::{classify_api_failure, RetryDisposition};
    use crate::api::WorkerApiError;

    #[test]
    fn classifies_retryable_http_statuses() {
        for status in [408, 425, 429, 500, 503, 599] {
            assert_eq!(
                classify_api_failure(&WorkerApiError::HttpStatus(status, String::new())),
                RetryDisposition::Transient
            );
        }
        for status in [400, 401, 403, 404, 409, 422] {
            assert_eq!(
                classify_api_failure(&WorkerApiError::HttpStatus(status, String::new())),
                RetryDisposition::Permanent
            );
        }
    }

    #[test]
    fn classifies_non_http_failures_as_transient() {
        assert_eq!(
            classify_api_failure(&WorkerApiError::Transport("timeout".into())),
            RetryDisposition::Transient
        );
        assert_eq!(
            classify_api_failure(&WorkerApiError::MalformedResponse("bad json".into())),
            RetryDisposition::Transient
        );
    }
}

/// Mirrors `WorkerResultOutbox`. Holds `Arc` handles rather than borrowing
/// (Swift's `outbox: DurableOutbox, api: WorkerAPIClient` fields are
/// reference-counted for free via ARC since both are reference types there;
/// `Arc` is the direct Rust analogue and lets `WorkerRuntime` — F0b's
/// `runtime.rs` — hold a `WorkerResultOutbox` as a plain sibling field next
/// to its own `Arc<DurableOutbox>`/`Arc<WorkerApiClient>`, instead of a
/// self-referential borrow).
pub struct WorkerResultOutbox {
    outbox: Arc<DurableOutbox>,
    api: Arc<WorkerApiClient>,
    lease_owner: String,
}

impl WorkerResultOutbox {
    /// Mirrors `WorkerResultOutbox.init(outbox:api:)`.
    pub fn new(outbox: Arc<DurableOutbox>, api: Arc<WorkerApiClient>) -> Self {
        WorkerResultOutbox {
            outbox,
            api,
            lease_owner: generate_lease_owner(),
        }
    }

    /// Mirrors `WorkerResultOutbox.enqueuePrimary(_:workerID:now:)`.
    pub fn enqueue_primary(
        &self,
        result: &PrimaryResultRequest,
        worker_id: &str,
        now: f64,
    ) -> Result<(), ResultOutboxError> {
        let task_id = result
            .task_id
            .as_deref()
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                WorkerApiError::MalformedResponse(
                    "primary outbox result requires task_id".to_string(),
                )
            })?;
        self.enqueue(
            result,
            &format!("primary:{task_id}"),
            "/api/worker/result",
            worker_id,
            now,
        )
    }

    /// Mirrors `WorkerResultOutbox.enqueueFPP(_:workerID:now:)`.
    pub fn enqueue_fpp(
        &self,
        result: &FppResultRequest,
        worker_id: &str,
        now: f64,
    ) -> Result<(), ResultOutboxError> {
        let candidate_id = result.candidate_id.ok_or_else(|| {
            WorkerApiError::MalformedResponse("FPP outbox result requires candidate_id".to_string())
        })?;
        self.enqueue(
            result,
            &format!("fpp:{candidate_id}"),
            "/api/worker/fpp_result",
            worker_id,
            now,
        )
    }

    /// Mirrors `WorkerResultOutbox.enqueueExoMiner(_:workerID:now:)`.
    pub fn enqueue_exominer(
        &self,
        result: &ExoMinerResultRequest,
        worker_id: &str,
        now: f64,
    ) -> Result<(), ResultOutboxError> {
        let candidate_id = result.candidate_id.ok_or_else(|| {
            WorkerApiError::MalformedResponse(
                "ExoMiner outbox result requires candidate_id".to_string(),
            )
        })?;
        self.enqueue(
            result,
            &format!("exominer:{candidate_id}"),
            "/api/worker/exominer_result",
            worker_id,
            now,
        )
    }

    /// Mirrors `WorkerResultOutbox.pendingCount()`.
    pub fn pending_count(&self) -> Result<i64, OutboxError> {
        self.outbox.pending_count()
    }

    /// Mirrors `WorkerResultOutbox.deliveryUnknownCount()`. Legacy
    /// observability only since F8 — see `DurableOutbox::delivery_unknown_count`.
    pub fn delivery_unknown_count(&self) -> Result<i64, OutboxError> {
        self.outbox.delivery_unknown_count()
    }

    /// F8 (2026-08-05, `docs/PLAN.md`): reconciles any legacy 'unknown'
    /// entries this outbox may have inherited (see
    /// `DurableOutbox::reconcile_legacy_unknown`). Exposed here so
    /// `runtime.rs` can call it through the same handle it already holds a
    /// `WorkerResultOutbox` for, without reaching past it into the inner
    /// `DurableOutbox`.
    pub fn reconcile_legacy_unknown(&self, now: f64) -> Result<i64, OutboxError> {
        self.outbox.reconcile_legacy_unknown(now)
    }

    /// Mirrors `WorkerResultOutbox.drain(now:limit:)`.
    ///
    /// F8 (2026-08-05) fix for the delivery-ambiguity worker kill (real
    /// incident, 2026-08-05: 15 fleet instances killed and needed manual
    /// reconciliation): a transport failure (connection dropped, timeout,
    /// TLS reset, ...) genuinely cannot tell whether the server committed
    /// the result before the response was lost. Previously this branch
    /// called `mark_delivery_unknown`, parking the entry in a terminal
    /// 'unknown' state that `runtime.rs` treated as fatal. That is no longer
    /// necessary: the server's result-ingestion endpoints are idempotent
    /// upserts (`~/exonit/exonit/routes/workers.py` `POST
    /// /api/worker/result` ~L951 unconditional `add_star` upsert;
    /// `exonit/db.py::add_candidate` upsert-by-`(tic, candidate_num,
    /// sector)`, documented to refresh measured transit parameters without
    /// erasing later analyses; `db.py::complete_task`'s idempotent
    /// `UPDATE`) — so simply retrying is safe: in the worst case the server
    /// re-upserts the same values it already committed. This branch now
    /// calls `mark_failed` (same backoff mechanics as an HTTP error retry,
    /// with a message that says why) instead, so the entry stays 'pending'
    /// and is retried like any other transient failure.
    pub fn drain(&self, now: f64, limit: i64) -> OutboxDrainResult {
        let entries = match self
            .outbox
            .claim_pending(&self.lease_owner, now, 120.0, limit)
        {
            Ok(entries) => entries,
            Err(error) => {
                return OutboxDrainResult {
                    delivered: 0,
                    failed: 0,
                    scan_error: Some(error.to_string()),
                };
            }
        };
        let mut delivered = 0;
        let mut failed = 0;
        for entry in &entries {
            match self.api.send_outbox_entry(entry) {
                Ok(_) => match self.outbox.mark_delivered(&entry.id, &self.lease_owner) {
                    Ok(()) => delivered += 1,
                    Err(error) => {
                        return OutboxDrainResult {
                            delivered,
                            failed,
                            scan_error: Some(error.to_string()),
                        };
                    }
                },
                Err(WorkerApiError::HttpStatus(code, body)) => {
                    let message = format!("HTTP {code}: {body}");
                    let disposition =
                        classify_api_failure(&WorkerApiError::HttpStatus(code, body.clone()));
                    match self.outbox.mark_failed(
                        &entry.id,
                        &self.lease_owner,
                        &message,
                        now,
                        disposition,
                    ) {
                        Ok(()) => failed += 1,
                        Err(error) => {
                            return OutboxDrainResult {
                                delivered,
                                failed,
                                scan_error: Some(error.to_string()),
                            };
                        }
                    }
                }
                // Mirrors Swift's `.invalidResponse, .malformedResponse` case
                // plus the outer catch-all for any other thrown error
                // (transport failures). See this method's doc comment: safe
                // to retry (idempotent server ingestion), so this is a
                // normal `mark_failed` retry with backoff, not a terminal
                // ambiguous state anymore.
                Err(other) => {
                    let message = format!(
                        "transport-ambiguous, safe retry (server ingestion is idempotent): {other}"
                    );
                    match self.outbox.mark_failed(
                        &entry.id,
                        &self.lease_owner,
                        &message,
                        now,
                        classify_api_failure(&other),
                    ) {
                        Ok(()) => failed += 1,
                        Err(error) => {
                            return OutboxDrainResult {
                                delivered,
                                failed,
                                scan_error: Some(error.to_string()),
                            };
                        }
                    }
                }
            }
        }
        OutboxDrainResult {
            delivered,
            failed,
            scan_error: None,
        }
    }

    fn enqueue<T: serde::Serialize>(
        &self,
        value: &T,
        key: &str,
        endpoint: &str,
        worker_id: &str,
        now: f64,
    ) -> Result<(), ResultOutboxError> {
        let data =
            serde_json::to_vec(value).map_err(|e| ResultOutboxError::Encode(e.to_string()))?;
        self.outbox.enqueue(key, endpoint, worker_id, &data, now)?;
        Ok(())
    }
}
