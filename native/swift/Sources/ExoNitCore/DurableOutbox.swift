import Foundation
import SQLite3

public struct OutboxEntry: Equatable, Sendable {
    public let id: String
    public let endpoint: String
    public let workerID: String
    public let payload: Data
    public let createdAt: Date
    public let attempts: Int
    public let nextAttemptAt: Date
    public let lastError: String?
}

public enum OutboxError: Error, CustomStringConvertible, Sendable {
    case sqlite(code: Int32, message: String)
    public var description: String {
        switch self { case let .sqlite(code, message): return "SQLite \(code): \(message)" }
    }
}

public actor DurableOutbox {
    nonisolated(unsafe) private var database: OpaquePointer?
    private static let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    /// Count of legacy `delivery_state = 'unknown'` rows reconciled back to
    /// `'pending'` when this database was opened (2026-08-05 fix — see
    /// `reconcileUnknown` below). `Int` is `Sendable`, so this `let` is
    /// implicitly nonisolated and callers can read it without `await`.
    public nonisolated let reconciledUnknownCount: Int

    public init(databaseURL: URL) throws {
        try FileManager.default.createDirectory(
            at: databaseURL.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        var handle: OpaquePointer?
        let flags = SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX
        let result = sqlite3_open_v2(databaseURL.path, &handle, flags, nil)
        guard result == SQLITE_OK, let handle else {
            let message = handle.map { String(cString: sqlite3_errmsg($0)) } ?? "open failed"
            if let handle { sqlite3_close(handle) }
            throw OutboxError.sqlite(code: result, message: message)
        }
        database = handle
        sqlite3_busy_timeout(handle, 5_000)
        try Self.execute(handle, "PRAGMA journal_mode=WAL;")
        try Self.execute(handle, "PRAGMA synchronous=FULL;")
        try Self.execute(handle, """
            CREATE TABLE IF NOT EXISTS outbox (
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
            );
            """)
        try Self.ensureColumn(handle, name: "lease_owner", declaration: "TEXT")
        try Self.ensureColumn(handle, name: "lease_until", declaration: "REAL")
        try Self.ensureColumn(
            handle, name: "delivery_state", declaration: "TEXT NOT NULL DEFAULT 'pending'"
        )
        try Self.execute(handle, "CREATE INDEX IF NOT EXISTS outbox_due ON outbox(next_attempt_at, created_at);")
        // 2026-08-05 incident fix: 'unknown' rows are always safe to retry —
        // whether left by a pre-fix worker (a transport error during result
        // delivery used to be marked 'unknown' and killed the worker outright,
        // requiring manual PG reconciliation) or by the still-legible
        // .invalidResponse/.malformedResponse paths in
        // WorkerResultOutbox.drain(). The server is idempotent for outbox
        // re-sends — exonit/routes/workers.py POST /api/worker/result does an
        // unconditional add_star upsert, db.add_candidate is "insert or
        // update" (refreshing measured transit parameters while preserving
        // FPP/ExoMiner/verdicts), and complete_task is a plain UPDATE. So
        // reconciling every 'unknown' row to 'pending' on open — worst case
        // re-upserting the same values — is correct, not just convenient.
        reconciledUnknownCount = try Self.reconcileUnknown(handle, now: Date())
    }

    deinit { if let database { sqlite3_close(database) } }

    public func enqueue(
        idempotencyKey: String,
        endpoint: String,
        workerID: String,
        payload: Data,
        now: Date = Date()
    ) throws {
        guard !idempotencyKey.isEmpty, !endpoint.isEmpty, !workerID.isEmpty else {
            throw OutboxError.sqlite(code: SQLITE_MISUSE, message: "empty outbox key, endpoint or worker ID")
        }
        let statement = try prepare("""
            INSERT OR IGNORE INTO outbox
              (id, endpoint, worker_id, payload, created_at, attempts, next_attempt_at)
            VALUES (?, ?, ?, ?, ?, 0, ?);
            """)
        defer { sqlite3_finalize(statement) }
        bind(idempotencyKey, index: 1, to: statement)
        bind(endpoint, index: 2, to: statement)
        bind(workerID, index: 3, to: statement)
        payload.withUnsafeBytes { buffer in
            _ = sqlite3_bind_blob(statement, 4, buffer.baseAddress, Int32(buffer.count), Self.transient)
        }
        sqlite3_bind_double(statement, 5, now.timeIntervalSince1970)
        sqlite3_bind_double(statement, 6, now.timeIntervalSince1970)
        try stepDone(statement)
    }

    public func pending(now: Date = Date(), limit: Int = 32) throws -> [OutboxEntry] {
        guard limit > 0 else { return [] }
        let statement = try prepare("""
            SELECT id, endpoint, worker_id, payload, created_at, attempts, next_attempt_at, last_error
            FROM outbox
            WHERE delivery_state = 'pending' AND next_attempt_at <= ?
              AND (lease_until IS NULL OR lease_until <= ?)
            ORDER BY created_at, id LIMIT ?;
            """)
        defer { sqlite3_finalize(statement) }
        sqlite3_bind_double(statement, 1, now.timeIntervalSince1970)
        sqlite3_bind_double(statement, 2, now.timeIntervalSince1970)
        sqlite3_bind_int(statement, 3, Int32(min(limit, Int(Int32.max))))
        var entries: [OutboxEntry] = []
        while true {
            let result = sqlite3_step(statement)
            if result == SQLITE_DONE { break }
            guard result == SQLITE_ROW else { throw error(code: result) }
            let byteCount = Int(sqlite3_column_bytes(statement, 3))
            let payload: Data
            if byteCount == 0 { payload = Data() }
            else if let bytes = sqlite3_column_blob(statement, 3) { payload = Data(bytes: bytes, count: byteCount) }
            else { throw OutboxError.sqlite(code: SQLITE_CORRUPT, message: "NULL outbox payload") }
            entries.append(OutboxEntry(
                id: columnText(statement, 0), endpoint: columnText(statement, 1),
                workerID: columnText(statement, 2), payload: payload,
                createdAt: Date(timeIntervalSince1970: sqlite3_column_double(statement, 4)),
                attempts: Int(sqlite3_column_int64(statement, 5)),
                nextAttemptAt: Date(timeIntervalSince1970: sqlite3_column_double(statement, 6)),
                lastError: sqlite3_column_type(statement, 7) == SQLITE_NULL ? nil : columnText(statement, 7)
            ))
        }
        return entries
    }

    public func claimPending(
        owner: String,
        now: Date = Date(),
        leaseDuration: TimeInterval = 120,
        limit: Int = 32
    ) throws -> [OutboxEntry] {
        guard !owner.isEmpty, leaseDuration > 0, limit > 0 else { return [] }
        guard let database else {
            throw OutboxError.sqlite(code: SQLITE_MISUSE, message: "database closed")
        }
        try Self.execute(database, "BEGIN IMMEDIATE;")
        do {
            let entries = try pending(now: now, limit: limit)
            let statement = try prepare("""
                UPDATE outbox SET lease_owner = ?, lease_until = ?
                WHERE id = ? AND (lease_until IS NULL OR lease_until <= ?);
                """)
            defer { sqlite3_finalize(statement) }
            for entry in entries {
                sqlite3_reset(statement); sqlite3_clear_bindings(statement)
                bind(owner, index: 1, to: statement)
                sqlite3_bind_double(statement, 2, now.addingTimeInterval(leaseDuration).timeIntervalSince1970)
                bind(entry.id, index: 3, to: statement)
                sqlite3_bind_double(statement, 4, now.timeIntervalSince1970)
                try stepDone(statement)
            }
            try Self.execute(database, "COMMIT;")
            return entries
        } catch {
            try? Self.execute(database, "ROLLBACK;")
            throw error
        }
    }

    public func markDelivered(id: String) throws {
        let statement = try prepare("DELETE FROM outbox WHERE id = ?;")
        defer { sqlite3_finalize(statement) }
        bind(id, index: 1, to: statement)
        try stepDone(statement)
    }

    public func markFailed(id: String, error message: String, now: Date = Date()) throws {
        let statement = try prepare("""
            UPDATE outbox SET attempts = attempts + 1,
              next_attempt_at = ? + MIN(300.0, POW(2.0, attempts + 1)),
              last_error = ?, lease_owner = NULL, lease_until = NULL WHERE id = ?;
            """)
        defer { sqlite3_finalize(statement) }
        sqlite3_bind_double(statement, 1, now.timeIntervalSince1970)
        bind(String(message.prefix(2_048)), index: 2, to: statement)
        bind(id, index: 3, to: statement)
        try stepDone(statement)
    }

    public func markDeliveryUnknown(id: String, error message: String) throws {
        let statement = try prepare("""
            UPDATE outbox SET delivery_state = 'unknown', last_error = ?,
              lease_owner = NULL, lease_until = NULL WHERE id = ?;
            """)
        defer { sqlite3_finalize(statement) }
        bind(String(message.prefix(2_048)), index: 1, to: statement)
        bind(id, index: 2, to: statement)
        try stepDone(statement)
    }

    /// Reconciles any `delivery_state = 'unknown'` rows back to `'pending'`,
    /// immediately retryable (`next_attempt_at = now`). Called once
    /// automatically on `init` (see `reconciledUnknownCount`); also exposed
    /// here so `WorkerRuntime`'s run loop can sweep up any 'unknown' rows a
    /// still-running worker produces (the .invalidResponse/.malformedResponse
    /// paths in `WorkerResultOutbox.drain()` still write that legacy state)
    /// without ever treating them as fatal.
    @discardableResult
    public func reconcileUnknown(now: Date = Date()) throws -> Int {
        guard let database else { throw OutboxError.sqlite(code: SQLITE_MISUSE, message: "database closed") }
        return try Self.reconcileUnknown(database, now: now)
    }

    public func pendingCount() throws -> Int {
        let statement = try prepare("SELECT COUNT(*) FROM outbox WHERE delivery_state = 'pending';")
        defer { sqlite3_finalize(statement) }
        let result = sqlite3_step(statement)
        guard result == SQLITE_ROW else { throw error(code: result) }
        return Int(sqlite3_column_int64(statement, 0))
    }

    public func deliveryUnknownCount() throws -> Int {
        let statement = try prepare("SELECT COUNT(*) FROM outbox WHERE delivery_state = 'unknown';")
        defer { sqlite3_finalize(statement) }
        let result = sqlite3_step(statement)
        guard result == SQLITE_ROW else { throw error(code: result) }
        return Int(sqlite3_column_int64(statement, 0))
    }

    public func count() throws -> Int {
        let statement = try prepare("SELECT COUNT(*) FROM outbox;")
        defer { sqlite3_finalize(statement) }
        let result = sqlite3_step(statement)
        guard result == SQLITE_ROW else { throw error(code: result) }
        return Int(sqlite3_column_int64(statement, 0))
    }

    private func prepare(_ sql: String) throws -> OpaquePointer {
        guard let database else { throw OutboxError.sqlite(code: SQLITE_MISUSE, message: "database closed") }
        var statement: OpaquePointer?
        let result = sqlite3_prepare_v2(database, sql, -1, &statement, nil)
        guard result == SQLITE_OK, let statement else { throw error(code: result) }
        return statement
    }

    private func bind(_ value: String, index: Int32, to statement: OpaquePointer) {
        sqlite3_bind_text(statement, index, value, -1, Self.transient)
    }

    private func columnText(_ statement: OpaquePointer, _ index: Int32) -> String {
        guard let text = sqlite3_column_text(statement, index) else { return "" }
        return String(cString: text)
    }

    private func stepDone(_ statement: OpaquePointer) throws {
        let result = sqlite3_step(statement)
        guard result == SQLITE_DONE else { throw error(code: result) }
    }

    private func error(code: Int32) -> OutboxError {
        .sqlite(code: code, message: database.map { String(cString: sqlite3_errmsg($0)) } ?? "database closed")
    }

    private static func ensureColumn(
        _ database: OpaquePointer, name: String, declaration: String
    ) throws {
        var statement: OpaquePointer?
        let prepared = sqlite3_prepare_v2(database, "PRAGMA table_info(outbox);", -1, &statement, nil)
        guard prepared == SQLITE_OK, let statement else {
            throw OutboxError.sqlite(code: prepared, message: String(cString: sqlite3_errmsg(database)))
        }
        defer { sqlite3_finalize(statement) }
        while sqlite3_step(statement) == SQLITE_ROW {
            if let raw = sqlite3_column_text(statement, 1), String(cString: raw) == name { return }
        }
        try execute(database, "ALTER TABLE outbox ADD COLUMN \(name) \(declaration);")
    }

    @discardableResult
    private static func reconcileUnknown(_ database: OpaquePointer, now: Date) throws -> Int {
        // Manual/operator reconciliation may leave rows in 'delivered' (the
        // code's own success path DELETEs instead). Such rows are done by
        // definition, but `count()` counts them and the run loop refuses new
        // work while the outbox is non-empty — 14 production instances sat
        // idle-but-heartbeating for hours on 2026-08-05 because of exactly
        // this. Purge them before reviving 'unknown' rows.
        try execute(database, "DELETE FROM outbox WHERE delivery_state = 'delivered';")
        var statement: OpaquePointer?
        let sql = """
            UPDATE outbox SET delivery_state = 'pending', next_attempt_at = ?,
              lease_owner = NULL, lease_until = NULL WHERE delivery_state = 'unknown';
            """
        let prepared = sqlite3_prepare_v2(database, sql, -1, &statement, nil)
        guard prepared == SQLITE_OK, let statement else {
            throw OutboxError.sqlite(code: prepared, message: String(cString: sqlite3_errmsg(database)))
        }
        defer { sqlite3_finalize(statement) }
        sqlite3_bind_double(statement, 1, now.timeIntervalSince1970)
        let result = sqlite3_step(statement)
        guard result == SQLITE_DONE else {
            throw OutboxError.sqlite(code: result, message: String(cString: sqlite3_errmsg(database)))
        }
        return Int(sqlite3_changes(database))
    }

    private static func execute(_ database: OpaquePointer, _ sql: String) throws {
        var message: UnsafeMutablePointer<CChar>?
        let result = sqlite3_exec(database, sql, nil, nil, &message)
        if result != SQLITE_OK {
            let text = message.map { String(cString: $0) } ?? String(cString: sqlite3_errmsg(database))
            sqlite3_free(message)
            throw OutboxError.sqlite(code: result, message: text)
        }
    }
}
