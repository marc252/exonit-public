import Foundation

public struct OutboxDrainResult: Equatable, Sendable {
    public var delivered: Int
    public var failed: Int
    public var deliveryUnknown: Int
    public var scanError: String?

    public init(
        delivered: Int, failed: Int, deliveryUnknown: Int = 0, scanError: String?
    ) {
        self.delivered = delivered
        self.failed = failed
        self.deliveryUnknown = deliveryUnknown
        self.scanError = scanError
    }
}

public actor WorkerResultOutbox {
    private let outbox: DurableOutbox
    private let api: WorkerAPIClient
    private let encoder: JSONEncoder
    private let leaseOwner: String

    public init(outbox: DurableOutbox, api: WorkerAPIClient) {
        self.outbox = outbox; self.api = api
        self.encoder = JSONEncoder()
        self.leaseOwner = UUID().uuidString
    }

    public func enqueuePrimary(
        _ result: PrimaryResultRequest, workerID: String, now: Date = Date()
    ) async throws {
        guard let taskID = result.taskID, !taskID.isEmpty else {
            throw WorkerAPIError.malformedResponse("primary outbox result requires task_id")
        }
        try await enqueue(
            result, key: "primary:\(taskID)", endpoint: "/api/worker/result",
            workerID: workerID, now: now
        )
    }

    public func enqueueFPP(
        _ result: FPPResultRequest, workerID: String, now: Date = Date()
    ) async throws {
        guard let candidateID = result.candidateID else {
            throw WorkerAPIError.malformedResponse("FPP outbox result requires candidate_id")
        }
        try await enqueue(
            result, key: "fpp:\(candidateID)", endpoint: "/api/worker/fpp_result",
            workerID: workerID, now: now
        )
    }

    public func enqueueExoMiner(
        _ result: ExoMinerResultRequest, workerID: String, now: Date = Date()
    ) async throws {
        guard let candidateID = result.candidateID else {
            throw WorkerAPIError.malformedResponse("ExoMiner outbox result requires candidate_id")
        }
        try await enqueue(
            result, key: "exominer:\(candidateID)", endpoint: "/api/worker/exominer_result",
            workerID: workerID, now: now
        )
    }

    public func pendingCount() async throws -> Int { try await outbox.pendingCount() }
    public func deliveryUnknownCount() async throws -> Int { try await outbox.deliveryUnknownCount() }

    /// Sweeps any `delivery_state = 'unknown'` rows back to `'pending'`
    /// (see `DurableOutbox.reconcileUnknown`). Only the legacy
    /// .invalidResponse/.malformedResponse paths above still produce that
    /// state; the transport-ambiguous path no longer does.
    @discardableResult
    public func reconcileLegacyUnknown(now: Date = Date()) async throws -> Int {
        try await outbox.reconcileUnknown(now: now)
    }

    public func drain(now: Date = Date(), limit: Int = 32) async -> OutboxDrainResult {
        let entries: [OutboxEntry]
        do {
            entries = try await outbox.claimPending(
                owner: leaseOwner, now: now, leaseDuration: 120, limit: limit
            )
        }
        catch { return OutboxDrainResult(delivered: 0, failed: 0, scanError: String(describing: error)) }
        var delivered = 0
        var failed = 0
        var deliveryUnknown = 0
        for entry in entries {
            do {
                _ = try await api.sendOutboxEntry(entry)
                try await outbox.markDelivered(id: entry.id)
                delivered += 1
            } catch let error as WorkerAPIError {
                switch error {
                case .httpStatus:
                    failed += 1
                    do {
                        try await outbox.markFailed(
                            id: entry.id, error: String(describing: error), now: now
                        )
                    } catch {
                        return OutboxDrainResult(
                            delivered: delivered, failed: failed,
                            deliveryUnknown: deliveryUnknown, scanError: String(describing: error)
                        )
                    }
                case .invalidResponse, .malformedResponse:
                    deliveryUnknown += 1
                    do {
                        try await outbox.markDeliveryUnknown(
                            id: entry.id, error: String(describing: error)
                        )
                    } catch {
                        return OutboxDrainResult(
                            delivered: delivered, failed: failed,
                            deliveryUnknown: deliveryUnknown, scanError: String(describing: error)
                        )
                    }
                }
            } catch {
                // A transport failure may occur after the server committed the
                // request. 2026-08-05 incident: this used to be marked
                // 'unknown' and WorkerRuntime.run() killed the worker outright
                // ("ambiguous delivery, reconciliation required") — 14
                // production instances died this way in one day, needing
                // manual reconciliation against PG. That is unnecessary: the
                // server is idempotent for outbox re-sends
                // (exonit/routes/workers.py — add_star is an unconditional
                // upsert, db.add_candidate is "insert or update ...
                // reprocessing the same TIC should refresh measured transit
                // parameters but must preserve FPP/ExoMiner/verdicts", and
                // complete_task is a plain UPDATE). Retrying an ambiguous
                // delivery is therefore safe — the worst case is re-upserting
                // the same values — so treat it exactly like an HTTP failure:
                // same exponential backoff, retried on the next drain.
                failed += 1
                do {
                    try await outbox.markFailed(
                        id: entry.id,
                        error: "transport-ambiguous, safe retry (server ingestion is idempotent): "
                            + String(describing: error),
                        now: now
                    )
                } catch {
                    return OutboxDrainResult(
                        delivered: delivered, failed: failed,
                        deliveryUnknown: deliveryUnknown, scanError: String(describing: error)
                    )
                }
            }
        }
        return OutboxDrainResult(
            delivered: delivered, failed: failed,
            deliveryUnknown: deliveryUnknown, scanError: nil
        )
    }

    private func enqueue<T: Encodable>(
        _ value: T, key: String, endpoint: String, workerID: String, now: Date
    ) async throws {
        let data = try encoder.encode(value)
        try await outbox.enqueue(
            idempotencyKey: key, endpoint: endpoint, workerID: workerID,
            payload: data, now: now
        )
    }
}
