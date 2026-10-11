import Foundation

public struct ScienceSidecarConfiguration: Sendable {
    public let pythonURL: URL
    public let scriptURL: URL
    public let workerPythonDirectory: URL
    public let exominerONNXURL: URL

    public init(
        pythonURL: URL, scriptURL: URL,
        workerPythonDirectory: URL, exominerONNXURL: URL
    ) {
        self.pythonURL = pythonURL; self.scriptURL = scriptURL
        self.workerPythonDirectory = workerPythonDirectory
        self.exominerONNXURL = exominerONNXURL
    }
}

public final class BackfillTaskProcessor: @unchecked Sendable {
    private let primary: PrimaryTaskProcessor
    private let sidecar: ScienceSidecarRunner?
    private let nativeFPP: NativeFPPTaskProcessor?
    private let configuration: ScienceSidecarConfiguration?
    private let serverURL: URL
    private let token: String
    private let workDirectory: URL
    private let nativeFPPEnabled: Bool

    public init(
        primary: PrimaryTaskProcessor, configuration: ScienceSidecarConfiguration?,
        serverURL: URL, token: String, workDirectory: URL,
        nativeFPPEnabled: Bool = false, api: WorkerAPIClient? = nil,
        nativeFPPCacheDirectory: URL? = nil
    ) throws {
        self.primary = primary; self.configuration = configuration
        guard !nativeFPPEnabled || api != nil else {
            throw NativeFPPError.missingAPI
        }
        self.serverURL = serverURL; self.token = token; self.workDirectory = workDirectory
        self.nativeFPPEnabled = nativeFPPEnabled
        sidecar = configuration.map { ScienceSidecarRunner(
            pythonURL: $0.pythonURL, scriptURL: $0.scriptURL
        ) }
        if nativeFPPEnabled, let api {
            nativeFPP = NativeFPPTaskProcessor(
                primary: primary, api: api,
                cacheDirectory: nativeFPPCacheDirectory ?? workDirectory
            )
        } else {
            nativeFPP = nil
        }
        try FileManager.default.createDirectory(
            at: workDirectory, withIntermediateDirectories: true
        )
    }

    public func preflight(requireFPP: Bool, requireExoMiner: Bool) async throws {
        guard let sidecar, let configuration else {
            if requireExoMiner || (requireFPP && !nativeFPPEnabled) {
                throw WorkerRuntimeError.missingScienceSidecar
            }
            return
        }
        try await sidecar.preflight(
            workerPythonDirectory: configuration.workerPythonDirectory,
            requireFPP: requireFPP && !nativeFPPEnabled, requireExoMiner: requireExoMiner,
            exominerONNXURL: configuration.exominerONNXURL
        )
    }

    public func processFPP(
        _ task: BackfillTask, workerID: String
    ) async throws -> FPPResultRequest {
        if let nativeFPP { return try await nativeFPP.processFPP(task, workerID: workerID) }
        guard let sidecar, let configuration else {
            throw NativeFPPError.calculationUnavailable(task.ticID)
        }
        let loaded = try await primary.loadProcessedLightCurve(
            ticID: task.ticID, workerID: workerID
        )
        let response = try await sidecar.runFPP(
            task: task, time: loaded.processed.time,
            flux: loaded.processed.flatFlux,
            sectorCount: loaded.processed.sectorCount,
            workerPythonDirectory: configuration.workerPythonDirectory,
            workDirectory: workDirectory,
            serverURL: serverURL, token: token
        )
        return FPPResultRequest(
            candidateID: task.candidateID, ticID: task.ticID,
            fpp: response.fpp ?? -1, nfpp: response.nfpp ?? -1,
            probabilityPlanet: response.probabilityPlanet,
            verdict: response.verdict ?? (response.status == "ok" ? nil : "failed"),
            candidate: candidateObject(task)
        )
    }

    public func processExoMiner(
        _ task: BackfillTask
    ) async throws -> ExoMinerResultRequest {
        guard let sidecar, let configuration else {
            throw WorkerRuntimeError.missingScienceSidecar
        }
        let response = try await sidecar.runExoMiner(
            task: task,
            workerPythonDirectory: configuration.workerPythonDirectory,
            serverURL: serverURL, token: token,
            onnxURL: configuration.exominerONNXURL
        )
        return ExoMinerResultRequest(
            candidateID: task.candidateID, ticID: task.ticID,
            score: response.score ?? -1,
            tcePeriod: response.tcePeriod, sectorRun: response.sectorRun
        )
    }

    private func candidateObject(_ task: BackfillTask) -> [String: JSONValue] {
        [
            "candidate_id": .integer(task.candidateID), "tic_id": .string(task.ticID),
            "candidate_num": .integer(Int64(task.candidateNumber)),
            "period_days": .number(task.periodDays), "t0": .number(task.t0),
            "duration_days": .number(task.durationDays),
            "depth_ppm": .number(task.depthPPM), "snr": .number(task.snr),
            "n_sectors": .integer(Int64(task.sectorCount)),
        ]
    }
}
