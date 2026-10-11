import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

public struct WorkerCapabilities: Codable, Equatable, Sendable {
    public var gpu: String
    public var blsBackend: String
    public var blsGPUInfo: String
    public var fppBackend: String
    public var cpuCores: Int
    public var fppCPUCores: Int
    public var hardware: [String: JSONValue]

    enum CodingKeys: String, CodingKey {
        case gpu
        case blsBackend = "bls_backend"
        case blsGPUInfo = "bls_gpu_info"
        case fppBackend = "fpp_backend"
        case cpuCores = "cpu_cores"
        case fppCPUCores = "fpp_cpu_cores"
        case hardware
    }

    public init(
        gpu: String,
        blsBackend: String,
        blsGPUInfo: String,
        fppBackend: String,
        cpuCores: Int,
        fppCPUCores: Int,
        hardware: [String: JSONValue] = [:]
    ) {
        self.gpu = gpu
        self.blsBackend = blsBackend
        self.blsGPUInfo = blsGPUInfo
        self.fppBackend = fppBackend
        self.cpuCores = cpuCores
        self.fppCPUCores = fppCPUCores
        self.hardware = hardware
    }
}

public struct RegisterRequest: Codable, Equatable, Sendable {
    public var name: String
    public var capabilities: WorkerCapabilities

    public init(name: String, capabilities: WorkerCapabilities) {
        self.name = name
        self.capabilities = capabilities
    }
}

private struct RegisterResponse: Decodable {
    let workerID: String
    enum CodingKeys: String, CodingKey { case workerID = "worker_id" }
}

public struct WorkerTask: Equatable, Sendable {
    public let taskID: String
    public let ticID: String
    public let config: [String: JSONValue]

    public init(taskID: String, ticID: String, config: [String: JSONValue] = [:]) {
        self.taskID = taskID
        self.ticID = ticID
        self.config = config
    }
}

public enum TaskPollResult: Equatable, Sendable {
    case task(WorkerTask)
    case wait(seconds: Double?)
}

private struct WorkerTaskResponse: Decodable {
    let taskID: String?
    let ticID: String?
    let waitSeconds: Double?
    let config: [String: JSONValue]

    enum CodingKeys: String, CodingKey {
        case taskID = "task_id"
        case ticID = "tic_id"
        case waitSeconds = "wait_seconds"
        case config
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        taskID = try c.decodeIfPresent(String.self, forKey: .taskID)
        waitSeconds = try c.decodeIfPresent(Double.self, forKey: .waitSeconds)
        config = try c.decodeIfPresent([String: JSONValue].self, forKey: .config) ?? [:]
        if let string = try? c.decodeIfPresent(String.self, forKey: .ticID) {
            ticID = string
        } else if let integer = try? c.decodeIfPresent(Int64.self, forKey: .ticID) {
            ticID = String(integer)
        } else {
            ticID = nil
        }
    }
}

public struct HeartbeatRequest: Codable, Equatable, Sendable {
    public var workerID: String
    public var status: String
    public var currentTIC: String?
    public var phase: String?
    public var phaseProgress: Double?
    public var starElapsedSeconds: Double?
    public var starsDone: Int
    public var averageTimePerStar: Double
    public var lastError: String?
    public var uptimeSeconds: Double
    public var pid: Int32
    public var gitSHA: String?
    public var flagStates: [String: JSONValue]
    public var gpuFallback: [String: JSONValue]
    public var system: [String: JSONValue]
    public var gpuTelemetry: [String: JSONValue]
    public var recentTimings: [JSONValue]

    enum CodingKeys: String, CodingKey {
        case workerID = "worker_id", status, currentTIC = "current_tic", phase
        case phaseProgress = "phase_progress", starElapsedSeconds = "star_elapsed_s"
        case starsDone = "stars_done", averageTimePerStar = "avg_time_per_star", lastError = "last_error"
        case uptimeSeconds = "uptime_s", pid, gitSHA = "git_sha", flagStates = "flag_states"
        case gpuFallback = "gpu_fallback", system, gpuTelemetry = "gpu", recentTimings = "recent_timings"
    }

    public init(
        workerID: String,
        status: String,
        currentTIC: String?,
        phase: String?,
        starsDone: Int,
        uptimeSeconds: Double,
        pid: Int32,
        phaseProgress: Double? = nil,
        starElapsedSeconds: Double? = nil,
        averageTimePerStar: Double = 0,
        lastError: String? = nil,
        gitSHA: String? = nil,
        flagStates: [String: JSONValue] = [:],
        gpuFallback: [String: JSONValue] = [:],
        system: [String: JSONValue] = [:],
        gpuTelemetry: [String: JSONValue] = [:],
        recentTimings: [JSONValue] = []
    ) {
        self.workerID = workerID; self.status = status; self.currentTIC = currentTIC; self.phase = phase
        self.starsDone = starsDone; self.uptimeSeconds = uptimeSeconds; self.pid = pid
        self.phaseProgress = phaseProgress; self.starElapsedSeconds = starElapsedSeconds
        self.averageTimePerStar = averageTimePerStar; self.lastError = lastError; self.gitSHA = gitSHA
        self.flagStates = flagStates; self.gpuFallback = gpuFallback; self.system = system
        self.gpuTelemetry = gpuTelemetry; self.recentTimings = recentTimings
    }
}

public enum HeartbeatCommand: String, Codable, Equatable, Sendable {
    case `continue`
    case reRegister = "re-register"
    case pause
    case resume
    case shutdown
    case debugOn = "debug_on"
    case debugOff = "debug_off"
}

private struct HeartbeatResponse: Decodable {
    let command: String?
}

public enum WorkerAPIError: Error, CustomStringConvertible, Sendable {
    case invalidResponse
    case httpStatus(Int, String)
    case malformedResponse(String)

    public var description: String {
        switch self {
        case .invalidResponse: return "Server returned a non-HTTP response"
        case let .httpStatus(code, body): return "HTTP \(code): \(body)"
        case let .malformedResponse(message): return "Malformed server response: \(message)"
        }
    }
}

public struct WorkerAPIClient: @unchecked Sendable {
    public let baseURL: URL
    private let token: String
    private let session: URLSession
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    public init(baseURL: URL, token: String, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.token = token
        self.session = session
        self.encoder = JSONEncoder()
        self.decoder = JSONDecoder()
    }

    public func register(_ payload: RegisterRequest) async throws -> String {
        let response: RegisterResponse = try await send(
            method: "POST", path: "api/worker/register", body: payload, workerID: nil
        )
        guard !response.workerID.isEmpty else {
            throw WorkerAPIError.malformedResponse("empty worker_id")
        }
        return response.workerID
    }

    public func pollTask(workerID: String) async throws -> TaskPollResult {
        let response: WorkerTaskResponse = try await sendWithoutBody(
            method: "GET", path: "api/worker/task", workerID: workerID
        )
        guard let taskID = response.taskID else { return .wait(seconds: response.waitSeconds) }
        guard let ticID = response.ticID else {
            throw WorkerAPIError.malformedResponse("task_id without tic_id")
        }
        return .task(WorkerTask(taskID: taskID, ticID: ticID, config: response.config))
    }

    public func getTask(workerID: String) async throws -> WorkerTask? {
        switch try await pollTask(workerID: workerID) {
        case let .task(task): return task
        case .wait: return nil
        }
    }

    public func heartbeat(_ payload: HeartbeatRequest) async throws -> HeartbeatCommand {
        let response: HeartbeatResponse = try await send(
            method: "POST", path: "api/worker/heartbeat", body: payload,
            workerID: payload.workerID
        )
        guard let raw = response.command else { return .continue }
        guard let command = HeartbeatCommand(rawValue: raw) else {
            throw WorkerAPIError.malformedResponse("unknown heartbeat command: \(raw)")
        }
        return command
    }

    public func getFPPTask(workerID: String) async throws -> BackfillTask? {
        let envelope: BackfillTaskEnvelope = try await sendWithoutBody(
            method: "GET", path: "api/worker/fpp_task", workerID: workerID
        )
        return envelope.task
    }

    public func getExoMinerTask(workerID: String) async throws -> BackfillTask? {
        let envelope: BackfillTaskEnvelope = try await sendWithoutBody(
            method: "GET", path: "api/worker/exominer_task", workerID: workerID
        )
        return envelope.task
    }

    public func postPrimaryResult(_ payload: PrimaryResultRequest, workerID: String) async throws -> StatusResponse {
        try await send(method: "POST", path: "api/worker/result", body: payload, workerID: workerID)
    }

    public func postFPPResult(_ payload: FPPResultRequest, workerID: String) async throws -> StatusResponse {
        try await send(method: "POST", path: "api/worker/fpp_result", body: payload, workerID: workerID)
    }

    public func postExoMinerResult(_ payload: ExoMinerResultRequest, workerID: String) async throws -> StatusResponse {
        try await send(method: "POST", path: "api/worker/exominer_result", body: payload, workerID: workerID)
    }

    public func fetchMetadata(ticID: String, workerID: String? = nil) async throws -> StarMetadata {
        try await sendWithoutBody(method: "GET", path: "api/metadata/\(ticID)", workerID: workerID)
    }

    public func fetchFFI(ticID: String) async throws -> FFIResponse? {
        do {
            return try await sendWithoutBody(method: "GET", path: "api/ffi/\(ticID)", workerID: nil)
        } catch WorkerAPIError.httpStatus(404, _) {
            return nil
        }
    }

    public func fetchNearby(ticID: String) async throws -> NearbyResponse {
        try await sendWithoutBody(method: "GET", path: "api/nearby/\(ticID)", workerID: nil)
    }

    public func downloadFITS(filename: String) async throws -> Data {
        guard !filename.isEmpty, !filename.contains("/"), filename != ".", filename != ".." else {
            throw WorkerAPIError.malformedResponse("invalid FITS filename")
        }
        return try await performData(request(method: "GET", path: "api/fits/\(filename)", workerID: nil))
    }

    public func fetchUpdateManifest() async throws -> UpdateManifest {
        try await sendWithoutBody(method: "GET", path: "api/worker/update", workerID: nil)
    }

    public func downloadUpdateFile(filename: String) async throws -> Data {
        let components = filename.split(separator: "/", omittingEmptySubsequences: false)
        guard (components.count == 1 || components.count == 2),
              components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw WorkerAPIError.malformedResponse("invalid update filename")
        }
        return try await performData(request(method: "GET", path: "api/worker/update/\(filename)", workerID: nil))
    }

    public func fetchTFRecord(ticID: String) async throws -> TFRecordFetchResult {
        let (data, response) = try await performRaw(
            request(method: "GET", path: "api/exominer/tfrecord/\(ticID)", workerID: nil)
        )
        switch response.statusCode {
        case 200:
            return .ready(
                data,
                sectorRun: response.value(forHTTPHeaderField: "X-Sector-Run") ?? "",
                ticID: response.value(forHTTPHeaderField: "X-Tic-Id") ?? ticID
            )
        case 202:
            struct Pending: Decodable {
                let ticID: String
                let sectorRun: String
                enum CodingKeys: String, CodingKey { case ticID = "tic_id", sectorRun = "sector_run" }
            }
            do {
                let pending = try decoder.decode(Pending.self, from: data)
                return .preprocessing(ticID: pending.ticID, sectorRun: pending.sectorRun)
            } catch {
                throw WorkerAPIError.malformedResponse(String(describing: error))
            }
        case 404:
            return .notAvailable
        default:
            throw WorkerAPIError.httpStatus(
                response.statusCode, String(data: data.prefix(1_024), encoding: .utf8) ?? "<binary>"
            )
        }
    }

    public func sendOutboxEntry(_ entry: OutboxEntry) async throws -> StatusResponse {
        let allowed = ["/api/worker/result", "/api/worker/fpp_result", "/api/worker/exominer_result"]
        guard allowed.contains(entry.endpoint) else {
            throw WorkerAPIError.malformedResponse("outbox endpoint is not allowed")
        }
        var outgoing = request(
            method: "POST", path: String(entry.endpoint.drop(while: { $0 == "/" })),
            workerID: entry.workerID
        )
        outgoing.httpBody = entry.payload
        outgoing.setValue(entry.id, forHTTPHeaderField: "Idempotency-Key")
        return try await perform(outgoing)
    }

    private func send<Body: Encodable, Response: Decodable>(
        method: String,
        path: String,
        body: Body,
        workerID: String?
    ) async throws -> Response {
        var request = request(method: method, path: path, workerID: workerID)
        request.httpBody = try encoder.encode(body)
        return try await perform(request)
    }

    private func sendWithoutBody<Response: Decodable>(
        method: String,
        path: String,
        workerID: String?
    ) async throws -> Response {
        try await perform(request(method: method, path: path, workerID: workerID))
    }

    private func request(method: String, path: String, workerID: String?) -> URLRequest {
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        request.httpMethod = method
        request.timeoutInterval = method == "GET" ? 30 : 10
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let workerID {
            request.setValue(workerID, forHTTPHeaderField: "X-Worker-ID")
        }
        return request
    }

    private func performData(_ request: URLRequest) async throws -> Data {
        let (data, response) = try await performRaw(request)
        guard (200..<300).contains(response.statusCode) else {
            throw WorkerAPIError.httpStatus(
                response.statusCode,
                String(data: data.prefix(1_024), encoding: .utf8) ?? "<binary>"
            )
        }
        return data
    }

    private func performRaw(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let (data, rawResponse) = try await session.data(for: request)
        guard let response = rawResponse as? HTTPURLResponse else {
            throw WorkerAPIError.invalidResponse
        }
        return (data, response)
    }

    private func perform<Response: Decodable>(_ request: URLRequest) async throws -> Response {
        let (data, response) = try await performRaw(request)
        guard (200..<300).contains(response.statusCode) else {
            throw WorkerAPIError.httpStatus(
                response.statusCode,
                String(data: data.prefix(1_024), encoding: .utf8) ?? "<binary>"
            )
        }
        do {
            return try decoder.decode(Response.self, from: data)
        } catch {
            throw WorkerAPIError.malformedResponse(String(describing: error))
        }
    }
}
