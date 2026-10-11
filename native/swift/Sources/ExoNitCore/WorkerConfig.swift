import Foundation

public enum WorkerConfigError: Error, Equatable, CustomStringConvertible, Sendable {
    case invalidServerURL(String)
    case emptyToken
    case emptyWorkerName
    case invalidWorkerName(String)
    case invalidPositiveField(String)

    public var description: String {
        switch self {
        case let .invalidServerURL(value): return "Invalid HTTP(S) server URL: \(value)"
        case .emptyToken: return "Worker token must not be empty"
        case .emptyWorkerName: return "Worker name must not be empty"
        case let .invalidWorkerName(value): return "Invalid worker name: \(value)"
        case let .invalidPositiveField(name): return "\(name) must be positive"
        }
    }
}

public struct WorkerConfig: Codable, Equatable, Sendable {
    public var serverURL: URL
    public var token: String
    public var workerName: String
    public var cpuThreads: Int?
    public var gpuEnabled: Bool
    public var pollIntervalSeconds: Double
    public var heartbeatIntervalSeconds: Double
    public var fitsCacheDirectory: String
    public var guiPort: Int
    public var fppCPUCores: Int?
    public var logFile: String?
    public var logLevel: String
    public var exominerMode: String
    public var ffiFallbackOnLoadFail: Bool
    public var startupCodeSync: Bool
    public var taskPools: [String]?

    enum CodingKeys: String, CodingKey {
        case serverURL = "server_url"
        case token
        case workerName = "worker_name"
        case cpuThreads = "cpu_threads"
        case gpuEnabled = "gpu_enabled"
        case pollIntervalSeconds = "poll_interval_seconds"
        case heartbeatIntervalSeconds = "heartbeat_interval_seconds"
        case fitsCacheDirectory = "fits_cache_dir"
        case guiPort = "gui_port"
        case fppCPUCores = "fpp_cpu_cores"
        case logFile = "log_file"
        case logLevel = "log_level"
        case exominerMode = "exominer_mode"
        case ffiFallbackOnLoadFail = "ffi_fallback_on_load_fail"
        case startupCodeSync = "startup_code_sync"
        case taskPools = "task_pools"
    }

    public init(
        serverURL: URL,
        token: String,
        workerName: String,
        cpuThreads: Int? = nil,
        gpuEnabled: Bool = false,
        pollIntervalSeconds: Double = 5,
        heartbeatIntervalSeconds: Double = 30,
        fitsCacheDirectory: String = "fits_cache",
        guiPort: Int = 8402,
        fppCPUCores: Int? = nil,
        logFile: String? = nil,
        logLevel: String = "INFO",
        exominerMode: String = "legacy-tfrecord",
        ffiFallbackOnLoadFail: Bool = false,
        startupCodeSync: Bool = false,
        taskPools: [String]? = nil
    ) {
        self.serverURL = serverURL
        self.token = token
        self.workerName = workerName
        self.cpuThreads = cpuThreads
        self.gpuEnabled = gpuEnabled
        self.pollIntervalSeconds = pollIntervalSeconds
        self.heartbeatIntervalSeconds = heartbeatIntervalSeconds
        self.fitsCacheDirectory = fitsCacheDirectory
        self.guiPort = guiPort
        self.fppCPUCores = fppCPUCores
        self.logFile = logFile
        self.logLevel = logLevel
        self.exominerMode = exominerMode
        self.ffiFallbackOnLoadFail = ffiFallbackOnLoadFail
        self.startupCodeSync = startupCodeSync
        self.taskPools = taskPools
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        serverURL = try c.decode(URL.self, forKey: .serverURL)
        token = try c.decode(String.self, forKey: .token)
        workerName = try c.decodeIfPresent(String.self, forKey: .workerName) ?? "worker"
        cpuThreads = try c.decodeIfPresent(Int.self, forKey: .cpuThreads)
        gpuEnabled = try c.decodeIfPresent(Bool.self, forKey: .gpuEnabled) ?? false
        pollIntervalSeconds = try c.decodeIfPresent(Double.self, forKey: .pollIntervalSeconds) ?? 5
        heartbeatIntervalSeconds = try c.decodeIfPresent(Double.self, forKey: .heartbeatIntervalSeconds) ?? 30
        fitsCacheDirectory = try c.decodeIfPresent(String.self, forKey: .fitsCacheDirectory) ?? "fits_cache"
        guiPort = try c.decodeIfPresent(Int.self, forKey: .guiPort) ?? 8402
        fppCPUCores = try c.decodeIfPresent(Int.self, forKey: .fppCPUCores)
        logFile = try c.decodeIfPresent(String.self, forKey: .logFile)
        logLevel = try c.decodeIfPresent(String.self, forKey: .logLevel) ?? "INFO"
        exominerMode = try c.decodeIfPresent(String.self, forKey: .exominerMode) ?? "legacy-tfrecord"
        ffiFallbackOnLoadFail = try c.decodeIfPresent(Bool.self, forKey: .ffiFallbackOnLoadFail) ?? false
        startupCodeSync = try c.decodeIfPresent(Bool.self, forKey: .startupCodeSync) ?? false
        taskPools = try c.decodeIfPresent([String].self, forKey: .taskPools)
    }

    public func validated() throws -> WorkerConfig {
        let scheme = serverURL.scheme?.lowercased()
        guard scheme == "http" || scheme == "https", serverURL.host != nil else {
            throw WorkerConfigError.invalidServerURL(serverURL.absoluteString)
        }
        guard !token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw WorkerConfigError.emptyToken
        }
        guard !workerName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw WorkerConfigError.emptyWorkerName
        }
        guard WorkerNameValidator.isValid(workerName) else {
            throw WorkerConfigError.invalidWorkerName(workerName)
        }
        guard pollIntervalSeconds > 0 else { throw WorkerConfigError.invalidPositiveField("poll_interval_seconds") }
        guard heartbeatIntervalSeconds > 0 else { throw WorkerConfigError.invalidPositiveField("heartbeat_interval_seconds") }
        guard guiPort > 0 && guiPort <= 65_535 else { throw WorkerConfigError.invalidPositiveField("gui_port") }
        if let cpuThreads, cpuThreads <= 0 { throw WorkerConfigError.invalidPositiveField("cpu_threads") }
        if let fppCPUCores, fppCPUCores <= 0 { throw WorkerConfigError.invalidPositiveField("fpp_cpu_cores") }
        return self
    }

    public func applying(workerName: String?, serverURL: String?) throws -> WorkerConfig {
        var copy = self
        if let workerName { copy.workerName = workerName }
        if let serverURL {
            guard let parsed = URL(string: serverURL) else {
                throw WorkerConfigError.invalidServerURL(serverURL)
            }
            copy.serverURL = parsed
        }
        return try copy.validated()
    }
}
