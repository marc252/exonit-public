import Darwin
import Foundation

public enum WorkerRuntimeError: Error, CustomStringConvertible, Sendable {
    case unsupportedTaskPools([String])
    case missingScienceSidecar
    case canaryNotAuthorized

    public var description: String {
        switch self {
        case let .unsupportedTaskPools(pools):
            return "unknown task pools: \(pools.joined(separator: ","))"
        case .missingScienceSidecar:
            return "FPP/ExoMiner pools require the versioned Python science sidecar"
        case .canaryNotAuthorized:
            return "real worker execution requires EXONIT_SWIFT_CANARY=1"
        }
    }
}

public actor WorkerRuntimeState {
    public init() {}
    public private(set) var workerID = ""
    private(set) var paused = false
    private(set) var shutdownRequested = false
    private(set) var debug = false
    public private(set) var currentTIC: String?
    public private(set) var phase: String?
    public private(set) var starsDone = 0
    public private(set) var lastError: String?

    public func registered(_ id: String) { workerID = id }
    public func begin(tic: String, phase: String) { currentTIC = tic; self.phase = phase }
    public func completed() { currentTIC = nil; phase = nil; starsDone += 1; lastError = nil }
    public func failed(_ message: String) { currentTIC = nil; phase = nil; lastError = message }
    public func heartbeatFailed(_ message: String) { lastError = message }
    public func requestShutdown() { shutdownRequested = true }
    public func apply(_ command: HeartbeatCommand) {
        switch command {
        case .pause: paused = true
        case .resume: paused = false
        case .shutdown: shutdownRequested = true
        case .debugOn: debug = true
        case .debugOff: debug = false
        case .continue, .reRegister: break
        }
    }
}

public final class WorkerRuntime: @unchecked Sendable {
    public let state = WorkerRuntimeState()
    private let config: WorkerConfig
    private let api: WorkerAPIClient
    private let durableOutbox: DurableOutbox
    private let outbox: WorkerResultOutbox
    private let processor: PrimaryTaskProcessor
    private let backfill: BackfillTaskProcessor?
    private let blsBackend: BLSComputeBackend
    private let injectionBackend: BLSComputeBackend
    private let pools: Set<String>
    private let canaryAuthorized: Bool
    private let nativeFPPEnabled: Bool
    private let started = ContinuousClock.now

    public init(
        config: WorkerConfig, outboxURL: URL,
        cnnModel: URL? = nil, exominerModel: URL? = nil,
        scienceSidecar: ScienceSidecarConfiguration? = nil,
        canaryAuthorized: Bool = false,
        session: URLSession = .shared
    ) throws {
        let configuredPools = Set(config.taskPools ?? ["primary"])
        let allowed: Set<String> = ["primary", "fpp_backfill", "exominer_backfill"]
        let unsupported = configuredPools.subtracting(allowed).sorted()
        guard unsupported.isEmpty else { throw WorkerRuntimeError.unsupportedTaskPools(unsupported) }
        self.config = config; pools = configuredPools
        self.canaryAuthorized = canaryAuthorized
        self.nativeFPPEnabled = configuredPools.contains("fpp_backfill")
            && ProcessInfo.processInfo.environment["EXONIT_SWIFT_FPP"] == "1"
        api = WorkerAPIClient(baseURL: config.serverURL, token: config.token, session: session)
        durableOutbox = try DurableOutbox(databaseURL: outboxURL)
        outbox = WorkerResultOutbox(outbox: durableOutbox, api: api)
        blsBackend = BLSComputeBackend.configured(gpuEnabled: config.gpuEnabled)
        injectionBackend = BLSComputeBackend.injectionConfigured(gpuEnabled: config.gpuEnabled)
        processor = try PrimaryTaskProcessor(
            api: api, cacheDirectory: URL(fileURLWithPath: config.fitsCacheDirectory),
            cnnModel: cnnModel, exominerModel: exominerModel,
            maximumConcurrency: config.cpuThreads ?? ProcessInfo.processInfo.activeProcessorCount,
            blsBackend: blsBackend,
            injectionBackend: injectionBackend,
            gpuEnabled: config.gpuEnabled
        )
        if configuredPools.contains("fpp_backfill") || configuredPools.contains("exominer_backfill") {
            let nativeFPP = nativeFPPEnabled
            let fitsCacheDirectory = URL(fileURLWithPath: config.fitsCacheDirectory)
            guard (nativeFPP || scienceSidecar != nil)
                && (!configuredPools.contains("exominer_backfill") || scienceSidecar != nil) else {
                throw WorkerRuntimeError.missingScienceSidecar
            }
            backfill = try BackfillTaskProcessor(
                primary: processor, configuration: scienceSidecar,
                serverURL: config.serverURL, token: config.token,
                workDirectory: fitsCacheDirectory.appendingPathComponent(
                    "swift-sidecar", isDirectory: true
                ),
                nativeFPPEnabled: nativeFPP, api: api,
                nativeFPPCacheDirectory: fitsCacheDirectory
            )
        } else {
            backfill = nil
        }
    }

    public func run() async throws {
        guard canaryAuthorized else { throw WorkerRuntimeError.canaryNotAuthorized }
        // 2026-08-05 fix: DurableOutbox.init already reconciled any leftover
        // 'unknown' rows (pre-fix worker crashes) to 'pending' when it opened
        // this database above — this is no longer fatal, just worth logging.
        if durableOutbox.reconciledUnknownCount > 0 {
            print(
                "[worker] reconciled \(durableOutbox.reconciledUnknownCount) legacy ambiguous-delivery "
                    + "outbox entries to pending at startup (safe retry: server ingestion is idempotent, "
                    + "see exonit/routes/workers.py)"
            )
        }
        if let backfill {
            try await backfill.preflight(
                requireFPP: pools.contains("fpp_backfill"),
                requireExoMiner: pools.contains("exominer_backfill")
            )
        }
        try await register()
        let heartbeat = Task { [self] in await heartbeatLoop() }
        defer { heartbeat.cancel() }
        while !Task.isCancelled, !(await state.shutdownRequested) {
            // Defensive sweep: the .invalidResponse/.malformedResponse paths in
            // WorkerResultOutbox.drain() can still write 'unknown' mid-run.
            // Reconcile it to 'pending' rather than killing the worker.
            let reconciled = try await outbox.reconcileLegacyUnknown()
            if reconciled > 0 {
                print(
                    "[worker] reconciled \(reconciled) ambiguous-delivery outbox entries to pending "
                        + "(safe retry: server ingestion is idempotent)"
                )
            }
            let drain = await outbox.drain()
            if let error = drain.scanError { await state.failed(error) }
            if drain.deliveryUnknown > 0 {
                print(
                    "[worker] WARN: outbox drain reported \(drain.deliveryUnknown) ambiguous "
                        + "delivery/deliveries; reconciled to pending and retried next cycle, not fatal"
                )
                continue
            }
            if try await durableOutbox.count() > 0 {
                try await sleep(seconds: config.pollIntervalSeconds); continue
            }
            if await state.paused {
                try await sleep(seconds: config.pollIntervalSeconds); continue
            }
            let workerID = await state.workerID
            do {
                let didWork = try await processOne(workerID: workerID)
                if !didWork { try await sleep(seconds: config.pollIntervalSeconds) }
            } catch is CancellationError { throw CancellationError() }
            catch {
                await state.failed(String(describing: error))
                try await sleep(seconds: config.pollIntervalSeconds)
            }
        }
    }

    private func processOne(workerID: String) async throws -> Bool {
        if pools.contains("primary"),
           case let .task(task) = try await api.pollTask(workerID: workerID) {
            await state.begin(tic: task.ticID, phase: "primary")
            let result = try await processor.process(task, workerID: workerID)
            try await outbox.enqueuePrimary(result, workerID: workerID)
            try await drainOutboxAfterEnqueue()
            if result.status == "completed" { await state.completed() }
            else { await state.failed(result.error ?? "primary task failed") }
            return true
        }
        if pools.contains("fpp_backfill"), let backfill,
           let task = try await api.getFPPTask(workerID: workerID) {
            await state.begin(tic: task.ticID, phase: "fpp_backfill")
            let result = try await backfill.processFPP(task, workerID: workerID)
            try await outbox.enqueueFPP(result, workerID: workerID)
            try await drainOutboxAfterEnqueue()
            if result.verdict == "failed" {
                await state.failed("native FPP processing failed for TIC \(task.ticID)")
            } else {
                await state.completed()
            }
            return true
        }
        if pools.contains("exominer_backfill"), let backfill,
           let task = try await api.getExoMinerTask(workerID: workerID) {
            await state.begin(tic: task.ticID, phase: "exominer_backfill")
            let result = try await backfill.processExoMiner(task)
            try await outbox.enqueueExoMiner(result, workerID: workerID)
            try await drainOutboxAfterEnqueue()
            await state.completed()
            return true
        }
        return false
    }

    private func drainOutboxAfterEnqueue() async throws {
        let delivered = await outbox.drain()
        if let scanError = delivered.scanError {
            throw WorkerAPIError.malformedResponse("outbox drain: \(scanError)")
        }
        if delivered.deliveryUnknown > 0 {
            // 2026-08-05: no longer fatal — .invalidResponse/.malformedResponse
            // rows get reconciled to 'pending' and retried by the run loop's
            // reconcileLegacyUnknown() sweep, same as any other backoff entry.
            print(
                "[worker] WARN: outbox drain reported \(delivered.deliveryUnknown) ambiguous "
                    + "delivery/deliveries after enqueue; reconciled to pending and retried, not fatal"
            )
        }
    }

    private func register() async throws {
        let cores = config.cpuThreads ?? ProcessInfo.processInfo.activeProcessorCount
        // Primary `.automatic` always executes on CPU (see BLSAccelerated.search), so
        // only the explicit `.metal` opt-in counts as an active Metal primary backend.
        let activeMetal = blsBackend == .metal && BLSMetal.isAvailable
        let activeBLS = BLSComputeBackend.blsCapabilityName(
            backend: blsBackend, metalAvailable: BLSMetal.isAvailable
        )
        let injectionBackendName = BLSComputeBackend.injectionCapabilityName(
            backend: injectionBackend, metalAvailable: BLSMetal.isAvailable
        )
        let workerID = try await api.register(RegisterRequest(
            name: config.workerName,
            capabilities: WorkerCapabilities(
                gpu: BLSMetal.deviceName ?? "none",
                blsBackend: activeBLS,
                blsGPUInfo: activeMetal ? (BLSMetal.deviceName ?? "Metal") : "",
                fppBackend: pools.contains("fpp_backfill") ? (nativeFPPEnabled ? "swift-native" : "python-sidecar") : "disabled", cpuCores: cores,
                fppCPUCores: config.fppCPUCores ?? cores,
                hardware: [
                    "runtime": .string("swift-native"),
                    "task_pools": .array(pools.sorted().map(JSONValue.string)),
                    "gpu_configured": .bool(config.gpuEnabled),
                    "gpu_available": .bool(BLSMetal.isAvailable),
                    "bls_backend": .string(activeBLS),
                    "injection_backend": .string(injectionBackendName),
                    "coreml_compute_units": .string(config.gpuEnabled ? "all" : "cpuOnly"),
                    "fpp_backend": .string(
                        pools.contains("fpp_backfill")
                            ? (nativeFPPEnabled ? "swift-native" : "python-sidecar")
                            : "disabled"
                    ),
                    "exominer_backend": .string(pools.contains("exominer_backfill") ? "python-sidecar" : "disabled")
                ],
            )
        ))
        await state.registered(workerID)
    }

    private func heartbeatLoop() async {
        while !Task.isCancelled, !(await state.shutdownRequested) {
            do {
                try await sleep(seconds: config.heartbeatIntervalSeconds)
                let workerID = await state.workerID
                guard !workerID.isEmpty else { continue }
                let currentTIC = await state.currentTIC
                let command = try await api.heartbeat(HeartbeatRequest(
                    workerID: workerID,
                    status: await state.paused ? "paused" : (currentTIC == nil ? "idle" : "busy"),
                    currentTIC: currentTIC, phase: await state.phase,
                    starsDone: await state.starsDone,
                    uptimeSeconds: uptime(), pid: getpid(),
                    lastError: await state.lastError
                ))
                if command == .reRegister { try await register() }
                else { await state.apply(command) }
            } catch is CancellationError { return }
            catch { await state.heartbeatFailed("heartbeat: \(error)") }
        }
    }

    private func sleep(seconds: Double) async throws {
        try await Task.sleep(for: .seconds(seconds))
    }

    private func uptime() -> Double {
        let duration = started.duration(to: .now)
        return Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
    }
}
