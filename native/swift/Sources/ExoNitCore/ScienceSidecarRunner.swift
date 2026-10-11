import Darwin
import Foundation

private final class SidecarProcessBox: @unchecked Sendable {
    private let lock = NSLock()
    private var process: Process?
    private var cancelled = false

    func install(_ process: Process) {
        lock.lock(); self.process = process; let shouldCancel = cancelled; lock.unlock()
        if shouldCancel { terminate(process) }
    }

    func cancel() {
        lock.lock(); cancelled = true; let process = self.process; lock.unlock()
        if let process { terminate(process) }
    }

    func terminate(_ process: Process) {
        guard process.isRunning else { return }
        process.terminate()
        let pid = process.processIdentifier
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.2) {
            if process.isRunning { Darwin.kill(pid, SIGKILL) }
        }
    }
}

public struct ScienceSidecarResponse: Decodable, Equatable, Sendable {
    public let protocolVersion: Int
    public let status: String
    public let fpp: Double?
    public let nfpp: Double?
    public let probabilityPlanet: Double?
    public let verdict: String?
    public let score: Double?
    public let tcePeriod: Double?
    public let sectorRun: String?
    public let error: String?

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol", status, fpp, nfpp
        case probabilityPlanet = "probability_planet", verdict, score
        case tcePeriod = "tce_period", sectorRun = "sector_run", error
    }
}

public enum ScienceSidecarError: Error, CustomStringConvertible, Sendable {
    case missingFile(String)
    case invalidPayload
    case failed(exit: Int32, message: String)
    case protocolMismatch(Int)
    case deadlineExceeded

    public var description: String {
        switch self {
        case let .missingFile(path): return "science sidecar file missing: \(path)"
        case .invalidPayload: return "science sidecar payload is not valid JSON"
        case let .failed(exit, message): return "science sidecar exited \(exit): \(message)"
        case let .protocolMismatch(value): return "science sidecar protocol mismatch: \(value)"
        case .deadlineExceeded: return "science sidecar deadline exceeded"
        }
    }
}

public actor ScienceSidecarRunner {
    private let pythonURL: URL
    private let scriptURL: URL

    public init(pythonURL: URL, scriptURL: URL) {
        self.pythonURL = pythonURL; self.scriptURL = scriptURL
    }

    public func selfTest(
        extraArguments: [String] = [], timeoutSeconds: Double = 30
    ) async throws {
        let response = try await execute(
            arguments: ["--self-test"] + extraArguments,
            stdin: nil, timeoutSeconds: timeoutSeconds
        )
        guard response.status == "ok" else {
            throw ScienceSidecarError.failed(exit: 0, message: response.error ?? response.status)
        }
    }

    public func preflight(
        workerPythonDirectory: URL, requireFPP: Bool,
        requireExoMiner: Bool, exominerONNXURL: URL
    ) async throws {
        let object: [String: Any] = [
            "protocol": 1, "mode": "preflight",
            "worker_python_dir": workerPythonDirectory.path,
            "require_fpp": requireFPP, "require_exominer": requireExoMiner,
            "onnx_path": exominerONNXURL.path,
        ]
        let response = try await execute(
            arguments: [], stdin: try JSONSerialization.data(withJSONObject: object),
            timeoutSeconds: 60
        )
        guard response.status == "ok" else {
            throw ScienceSidecarError.failed(exit: 0, message: response.error ?? response.status)
        }
    }

    public func runFPP(
        task: BackfillTask, time: [Double], flux: [Double], sectorCount: Int,
        workerPythonDirectory: URL, workDirectory: URL,
        serverURL: URL, token: String
    ) async throws -> ScienceSidecarResponse {
        let object: [String: Any] = [
            "protocol": 1, "mode": "fpp",
            "worker_python_dir": workerPythonDirectory.path,
            "workdir": workDirectory.path,
            "server_url": serverURL.absoluteString, "token": token,
            "sector_count": sectorCount, "task": taskObject(task),
            "time": time, "flux": flux,
        ]
        return try await execute(
            arguments: [], stdin: try JSONSerialization.data(withJSONObject: object),
            timeoutSeconds: 3_600
        )
    }

    public func runExoMiner(
        task: BackfillTask, workerPythonDirectory: URL,
        serverURL: URL, token: String, onnxURL: URL
    ) async throws -> ScienceSidecarResponse {
        let object: [String: Any] = [
            "protocol": 1, "mode": "exominer",
            "worker_python_dir": workerPythonDirectory.path,
            "server_url": serverURL.absoluteString, "token": token,
            "onnx_path": onnxURL.path, "task": taskObject(task),
        ]
        return try await execute(
            arguments: [], stdin: try JSONSerialization.data(withJSONObject: object),
            timeoutSeconds: 1_800
        )
    }

    private func execute(
        arguments: [String], stdin: Data?, timeoutSeconds: Double
    ) async throws -> ScienceSidecarResponse {
        let pythonURL = self.pythonURL; let scriptURL = self.scriptURL
        guard FileManager.default.isExecutableFile(atPath: pythonURL.path) else {
            throw ScienceSidecarError.missingFile(pythonURL.path)
        }
        guard FileManager.default.fileExists(atPath: scriptURL.path) else {
            throw ScienceSidecarError.missingFile(scriptURL.path)
        }
        let processBox = SidecarProcessBox()
        let operation: @Sendable () async throws -> ScienceSidecarResponse = {
            do {
                let response: ScienceSidecarResponse = try await Task.detached(priority: .utility) {
                    let process = Process(); let output = Pipe(); let input = Pipe()
                    process.executableURL = pythonURL
                    process.arguments = [scriptURL.path] + arguments
                    process.standardOutput = output
                    process.standardError = FileHandle.standardError
                    if stdin != nil { process.standardInput = input }
                    try process.run(); processBox.install(process)
                    if let stdin {
                        input.fileHandleForWriting.write(stdin)
                        try? input.fileHandleForWriting.close()
                    }
                    let deadline = ContinuousClock.now.advanced(by: .seconds(timeoutSeconds))
                    var timedOut = false
                    while process.isRunning {
                        if ContinuousClock.now >= deadline {
                            timedOut = true; processBox.terminate(process); break
                        }
                        usleep(20_000)
                    }
                    process.waitUntilExit()
                    if timedOut { throw ScienceSidecarError.deadlineExceeded }
                    let data = output.fileHandleForReading.readDataToEndOfFile()
                    let decoded: ScienceSidecarResponse
                    do { decoded = try JSONDecoder().decode(ScienceSidecarResponse.self, from: data) }
                    catch {
                        throw ScienceSidecarError.failed(
                            exit: process.terminationStatus,
                            message: "invalid JSON response (\(data.count) bytes)"
                        )
                    }
                    if process.terminationStatus != 0 {
                        throw ScienceSidecarError.failed(
                            exit: process.terminationStatus,
                            message: decoded.error ?? decoded.status
                        )
                    }
                    return decoded
                }.value
                try Task.checkCancellation()
                return response
            } catch {
                if Task.isCancelled { throw CancellationError() }
                throw error
            }
        }
        let response = try await withTaskCancellationHandler(
            operation: operation,
            onCancel: { processBox.cancel() }
        )
        guard response.protocolVersion == 1 else {
            throw ScienceSidecarError.protocolMismatch(response.protocolVersion)
        }
        return response
    }

    private func taskObject(_ task: BackfillTask) -> [String: Any] {
        [
            "candidate_id": task.candidateID, "tic_id": task.ticID,
            "candidate_num": task.candidateNumber, "period_days": task.periodDays,
            "t0": task.t0, "duration_days": task.durationDays,
            "depth_ppm": task.depthPPM, "snr": task.snr,
            "n_sectors": task.sectorCount,
        ]
    }
}
