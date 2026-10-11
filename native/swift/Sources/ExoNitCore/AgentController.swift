import Darwin
import Foundation

@_silgen_name("flock")
private func systemFlock(_ descriptor: Int32, _ operation: Int32) -> Int32

public struct AgentExecutionResult: Sendable {
    public let output: Data
    public let exitCode: Int32

    public init(output: Data, exitCode: Int32) {
        self.output = output
        self.exitCode = exitCode
    }
}

public enum AgentControllerError: Error, CustomStringConvertible, Sendable {
    case missingWorkerName
    case invalidWorkerName(String)
    case missingBaseConfig(String)
    case spawnFailed(Int32)
    case startupLockFailed(String)
    case invalidUTF8

    public var description: String {
        switch self {
        case .missingWorkerName: return "worker name is required"
        case let .invalidWorkerName(value): return "invalid worker name: \(value)"
        case let .missingBaseConfig(path): return "base config not found: \(path)"
        case let .spawnFailed(code): return "posix_spawn failed: \(String(cString: strerror(code)))"
        case let .startupLockFailed(message): return "startup lock failed: \(message)"
        case .invalidUTF8: return "configuration override is not valid UTF-8"
        }
    }
}

public struct ProcessCommandLineInfo: Sendable {
    /// Kernel-recorded exec path (KERN_PROCARGS2) — the path passed to exec,
    /// not spoofable by the target process.
    public let executablePath: String
    public let arguments: [String]

    public init(executablePath: String, arguments: [String]) {
        self.executablePath = executablePath
        self.arguments = arguments
    }
}

public final class AgentController: @unchecked Sendable {
    public typealias ProcessExecutableURL = @Sendable (pid_t) -> URL?
    public typealias ProcessCommandLine = @Sendable (pid_t) -> ProcessCommandLineInfo?

    private enum ProcessInspection: Equatable {
        case notRunning
        case owned
        case foreign
        case unverifiable
    }

    private let workerDirectory: URL
    private let executableURL: URL
    private let environment: [String: String]
    private let fileManager: FileManager
    private let processExecutableURL: ProcessExecutableURL
    private let processCommandLine: ProcessCommandLine

    public init(
        workerDirectory: URL,
        executableURL: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default,
        processExecutableURL: ProcessExecutableURL? = nil,
        processCommandLine: ProcessCommandLine? = nil
    ) {
        self.workerDirectory = workerDirectory.standardizedFileURL
        self.executableURL = executableURL.standardizedFileURL
        self.environment = environment
        self.fileManager = fileManager
        self.processExecutableURL = processExecutableURL ?? AgentController.executableURL(for:)
        self.processCommandLine = processCommandLine ?? AgentController.commandLine(for:)
    }

    public func execute(_ arguments: AgentArguments) throws -> AgentExecutionResult {
        if let workerName = arguments.workerName, !WorkerNameValidator.isValid(workerName) {
            throw AgentControllerError.invalidWorkerName(workerName)
        }
        switch arguments.action {
        case .start: return try start(arguments)
        case .stop: return try stop(arguments)
        case .status: return try status(arguments)
        case .logs: return try logs(arguments)
        case .version: return try json([
            "git_sha": gitSHA(),
            "agent_version": "2.0-swift",
            "swift": "native",
            "platform": "posix",
        ])
        }
    }

    private func start(_ arguments: AgentArguments) throws -> AgentExecutionResult {
        guard let workerName = arguments.workerName else { throw AgentControllerError.missingWorkerName }
        let authorised = arguments.allowStart || environment["EXONIT_FARMCTL_ALLOW_START"] == "1"
        guard authorised else {
            return try json([
                "status": "refused",
                "error": "REFUSED: starting a worker needs --allow-start (or EXONIT_FARMCTL_ALLOW_START=1). bin/farmctl passes it for you.",
            ], exitCode: 2)
        }

        let overridesData: Data?
        if let encoded = arguments.configOverridesBase64 {
            do {
                overridesData = try AgentConfigDeriver.decodeOverrides(base64: encoded)
            } catch {
                return try json([
                    "status": "refused",
                    "error": "REFUSED: \(error)",
                ], exitCode: 2)
            }
        } else {
            overridesData = nil
        }

        return try withStartupLock(workerName: workerName) {
            try startLocked(arguments, workerName: workerName, overridesData: overridesData)
        }
    }

    private func startLocked(
        _ arguments: AgentArguments,
        workerName: String,
        overridesData: Data?
    ) throws -> AgentExecutionResult {
        try migrateLegacyPID(workerName: workerName)
        if let pid = readPID(workerName: workerName) {
            switch inspectProcess(pid, workerName: workerName) {
            case .owned:
                return try json([
                    "status": "already_running",
                    "pid": Int(pid),
                    "worker_name": readName(workerName: workerName) ?? workerName,
                ])
            case .unverifiable:
                return try unverifiableResult(pid: pid)
            case .foreign, .notRunning:
                removeState(workerName: workerName)
            }
        }

        let configURL = try prepareConfig(arguments, workerName: workerName, overridesData: overridesData)
        let logURL = workerDirectory.appendingPathComponent("worker-\(workerName).log")
        var childArguments = [
            executableURL.path,
            "--config", configURL.path,
            "--worker-name", workerName,
        ]
        if let serverURL = arguments.serverURL {
            childArguments += ["--server-url", serverURL]
        }
        var childEnvironment = environment
        childEnvironment["EXONIT_SWIFT_WORKER_CHILD"] = "1"
        let pid = try spawnDetached(
            executable: executableURL,
            arguments: childArguments,
            environment: childEnvironment,
            logURL: logURL
        )
        try Data("\(pid)".utf8).write(to: pidFile(workerName), options: .atomic)
        try Data(workerName.utf8).write(to: nameFile(workerName), options: .atomic)
        usleep(1_000_000)
        let alive = newlySpawnedChildIsAlive(pid)
        if !alive { removeState(workerName: workerName) }
        return try json([
            "status": alive ? "started" : "exited_immediately",
            "pid": Int(pid),
            "worker_name": workerName,
            "git_sha": gitSHA(),
        ], exitCode: alive ? 0 : 1)
    }

    private func stop(_ arguments: AgentArguments) throws -> AgentExecutionResult {
        guard let workerName = arguments.workerName else { throw AgentControllerError.missingWorkerName }
        try migrateLegacyPID(workerName: workerName)
        guard let pid = readPID(workerName: workerName) else {
            removeState(workerName: workerName)
            return try json(["status": "not_running"])
        }
        switch inspectProcess(pid, workerName: workerName) {
        case .notRunning, .foreign:
            removeState(workerName: workerName)
            return try json(["status": "not_running"])
        case .unverifiable:
            return try unverifiableResult(pid: pid)
        case .owned:
            break
        }
        let recordedName = readName(workerName: workerName) ?? workerName
        let signalToSend = arguments.force ? SIGKILL : SIGTERM
        guard Darwin.kill(pid, signalToSend) == 0 else {
            return try json([
                "status": "error", "error": String(cString: strerror(errno)), "pid": Int(pid),
            ])
        }
        if !arguments.force {
            let deadline = Date().addingTimeInterval(10)
            while processAlive(pid), Date() < deadline { usleep(100_000) }
            if processAlive(pid) { _ = Darwin.kill(pid, SIGKILL) }
        }
        removeState(workerName: workerName)
        return try json([
            "status": "stopped", "pid": Int(pid),
            "worker_name": recordedName,
        ])
    }

    private func status(_ arguments: AgentArguments) throws -> AgentExecutionResult {
        guard let workerName = arguments.workerName else { throw AgentControllerError.missingWorkerName }
        guard let pid = readPID(workerName: workerName) else {
            removeState(workerName: workerName)
            return try json([
                "status": "stopped", "pid": NSNull(), "worker_name": NSNull(),
                "git_sha": gitSHA(),
            ])
        }
        switch inspectProcess(pid, workerName: workerName) {
        case .unverifiable:
            return try unverifiableResult(pid: pid)
        case .owned:
            return try json([
                "status": "running", "pid": Int(pid),
                "worker_name": readName(workerName: workerName) ?? workerName,
                "git_sha": gitSHA(),
            ])
        case .foreign, .notRunning:
            removeState(workerName: workerName)
            return try json([
                "status": "stopped", "pid": NSNull(), "worker_name": NSNull(),
                "git_sha": gitSHA(),
            ])
        }
    }

    private func logs(_ arguments: AgentArguments) throws -> AgentExecutionResult {
        let preferred = arguments.workerName.map {
            workerDirectory.appendingPathComponent("worker-\($0).log")
        }
        let legacy = workerDirectory.appendingPathComponent("worker.log")
        let selected: URL?
        if let preferred, fileManager.fileExists(atPath: preferred.path) {
            selected = preferred
        } else if fileManager.fileExists(atPath: legacy.path) {
            selected = legacy
        } else {
            selected = nil
        }
        guard let selected else { return AgentExecutionResult(output: Data("\n".utf8), exitCode: 0) }
        let text = String(decoding: try Data(contentsOf: selected), as: UTF8.self)
        let allLines = text.split(whereSeparator: \.isNewline).map(String.init)
        let lines = arguments.logLines > 0 ? allLines.suffix(arguments.logLines) : allLines[...]
        return AgentExecutionResult(output: Data((lines.joined(separator: "\n") + "\n").utf8), exitCode: 0)
    }

    private func prepareConfig(
        _ arguments: AgentArguments,
        workerName: String,
        overridesData: Data?
    ) throws -> URL {
        let baseURL = workerDirectory.appendingPathComponent("config.json")
        guard fileManager.fileExists(atPath: baseURL.path) else {
            throw AgentControllerError.missingBaseConfig(baseURL.path)
        }
        if arguments.instances == 1, arguments.instanceIndex == 1, overridesData == nil {
            return baseURL
        }
        let baseData = try Data(contentsOf: baseURL)
        let output: Data
        if arguments.instances == 1, arguments.instanceIndex == 1, let overridesData {
            output = try AgentConfigDeriver.applySingleInstanceOverrides(
                baseConfigData: baseData, overridesData: overridesData
            )
        } else {
            output = try AgentConfigDeriver.derive(
                baseConfigData: baseData,
                overridesData: overridesData,
                workerName: workerName,
                instanceIndex: arguments.instanceIndex,
                instances: arguments.instances,
                workerDirectory: workerDirectory,
                cpuCount: ProcessInfo.processInfo.activeProcessorCount
            )
        }
        let derivedURL = workerDirectory.appendingPathComponent(".config.\(workerName).json")
        try output.write(to: derivedURL, options: .atomic)
        return derivedURL
    }

    private func spawnDetached(
        executable: URL,
        arguments: [String],
        environment: [String: String],
        logURL: URL
    ) throws -> pid_t {
        var actions: posix_spawn_file_actions_t? = nil
        var attributes: posix_spawnattr_t? = nil
        posix_spawn_file_actions_init(&actions)
        posix_spawnattr_init(&attributes)
        defer {
            posix_spawn_file_actions_destroy(&actions)
            posix_spawnattr_destroy(&attributes)
        }

        "/dev/null".withCString {
            _ = posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, $0, O_RDONLY, 0)
        }
        logURL.path.withCString {
            _ = posix_spawn_file_actions_addopen(
                &actions, STDOUT_FILENO, $0, O_WRONLY | O_CREAT | O_APPEND, 0o644
            )
        }
        _ = posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO)
        let flags = Int16(POSIX_SPAWN_SETSID | POSIX_SPAWN_CLOEXEC_DEFAULT)
        _ = posix_spawnattr_setflags(&attributes, flags)

        var argv = arguments.map { strdup($0) as UnsafeMutablePointer<CChar>? }
        argv.append(nil)
        var envp = environment.sorted { $0.key < $1.key }
            .map { strdup("\($0.key)=\($0.value)") as UnsafeMutablePointer<CChar>? }
        envp.append(nil)
        defer {
            for pointer in argv where pointer != nil { free(pointer) }
            for pointer in envp where pointer != nil { free(pointer) }
        }

        var pid: pid_t = 0
        let result = executable.path.withCString { path in
            argv.withUnsafeMutableBufferPointer { argvBuffer in
                envp.withUnsafeMutableBufferPointer { envBuffer in
                    posix_spawn(
                        &pid, path, &actions, &attributes,
                        argvBuffer.baseAddress!, envBuffer.baseAddress!
                    )
                }
            }
        }
        guard result == 0 else { throw AgentControllerError.spawnFailed(result) }
        return pid
    }

    private func inspectProcess(_ pid: pid_t, workerName: String) -> ProcessInspection {
        guard processAlive(pid) else { return .notRunning }
        let expected = executableURL.resolvingSymlinksInPath().standardizedFileURL
        if let actualURL = processExecutableURL(pid) {
            let actual = actualURL.resolvingSymlinksInPath().standardizedFileURL
            return actual == expected ? .owned : .foreign
        }
        // proc_pidpath returns 0 for our live, same-user, detached
        // posix_spawn children on macOS (production, 2026-08-05: every
        // stop/prune of a running instance answered "unverifiable" while
        // lsof and KERN_PROCARGS2 still saw the binary). Fall back to the
        // kernel-recorded spawn command line: exec path plus the exact
        // `--worker-name` argument identify the instance.
        guard let info = processCommandLine(pid) else { return .unverifiable }
        let spawnPath = URL(fileURLWithPath: info.executablePath)
            .resolvingSymlinksInPath().standardizedFileURL
        if spawnPath == expected { return .owned }
        // A release flip moves the `current` symlink, so an instance from
        // the previous release no longer resolves to the expected path.
        // Same binary name + exact instance argument is still our process.
        if spawnPath.lastPathComponent == expected.lastPathComponent,
           let flag = info.arguments.firstIndex(of: "--worker-name"),
           info.arguments.indices.contains(flag + 1),
           info.arguments[flag + 1] == workerName {
            return .owned
        }
        return .foreign
    }

    private static func executableURL(for pid: pid_t) -> URL? {
        var buffer = [CChar](repeating: 0, count: 16_384)
        let length = proc_pidpath(pid, &buffer, UInt32(buffer.count))
        guard length > 0 else { return nil }
        let pathBytes = buffer.prefix(Int(length)).prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return URL(fileURLWithPath: String(decoding: pathBytes, as: UTF8.self))
    }

    private static func commandLine(for pid: pid_t) -> ProcessCommandLineInfo? {
        // sysctl KERN_PROCARGS2 layout: Int32 argc | exec_path\0 | NUL
        // padding | argv[0]\0 argv[1]\0 … | environ. Same-user access only,
        // which is exactly the boundary we want.
        var mib: [Int32] = [CTL_KERN, KERN_PROCARGS2, pid]
        var size = 0
        guard sysctl(&mib, UInt32(mib.count), nil, &size, nil, 0) == 0,
              size > MemoryLayout<Int32>.size else { return nil }
        var buffer = [UInt8](repeating: 0, count: size)
        guard sysctl(&mib, UInt32(mib.count), &buffer, &size, nil, 0) == 0,
              size > MemoryLayout<Int32>.size else { return nil }
        let argc = Int(buffer.withUnsafeBytes { $0.loadUnaligned(as: Int32.self) })
        guard argc > 0 else { return nil }
        var offset = MemoryLayout<Int32>.size
        func nextString() -> String? {
            guard offset < size else { return nil }
            let start = offset
            while offset < size, buffer[offset] != 0 { offset += 1 }
            let value = String(decoding: buffer[start..<offset], as: UTF8.self)
            offset += 1
            return value
        }
        guard let execPath = nextString(), !execPath.isEmpty else { return nil }
        while offset < size, buffer[offset] == 0 { offset += 1 }
        var arguments: [String] = []
        for _ in 0..<argc {
            guard let argument = nextString() else { break }
            arguments.append(argument)
        }
        guard !arguments.isEmpty else { return nil }
        return ProcessCommandLineInfo(executablePath: execPath, arguments: arguments)
    }

    private func processAlive(_ pid: pid_t) -> Bool {
        if Darwin.kill(pid, 0) == 0 { return true }
        return errno == EPERM
    }

    private func pidFile(_ workerName: String) -> URL {
        workerDirectory.appendingPathComponent(".agent.\(workerName).pid")
    }

    private func nameFile(_ workerName: String) -> URL {
        workerDirectory.appendingPathComponent(".agent.\(workerName).worker_name")
    }

    private func readPID(workerName: String) -> pid_t? {
        guard let data = try? Data(contentsOf: pidFile(workerName)),
              let text = String(data: data, encoding: .utf8),
              let value = Int32(text.trimmingCharacters(in: .whitespacesAndNewlines)),
              value > 0 else {
            return nil
        }
        return value
    }

    private func readName(workerName: String) -> String? {
        guard let data = try? Data(contentsOf: nameFile(workerName)),
              let text = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !text.isEmpty else { return nil }
        return text
    }

    private func removeState(workerName: String) {
        try? fileManager.removeItem(at: pidFile(workerName))
        try? fileManager.removeItem(at: nameFile(workerName))
    }

    private func newlySpawnedChildIsAlive(_ pid: pid_t) -> Bool {
        var status: Int32 = 0
        while true {
            let result = Darwin.waitpid(pid, &status, WNOHANG)
            if result == 0 { return true }
            if result == pid { return false }
            if result == -1, errno == EINTR { continue }
            // This PID came directly from posix_spawn, so if another runtime
            // already reaped it, liveness is enough; its identity is known.
            return processAlive(pid)
        }
    }

    private func withStartupLock<T>(workerName: String, _ body: () throws -> T) throws -> T {
        let path = workerDirectory.appendingPathComponent(".agent.\(workerName).start.lock").path
        let descriptor = Darwin.open(path, O_CREAT | O_RDWR | O_CLOEXEC, 0o600)
        guard descriptor >= 0 else {
            throw AgentControllerError.startupLockFailed(String(cString: strerror(errno)))
        }
        defer { _ = Darwin.close(descriptor) }
        while systemFlock(descriptor, LOCK_EX) != 0 {
            if errno == EINTR { continue }
            throw AgentControllerError.startupLockFailed(String(cString: strerror(errno)))
        }
        defer { _ = systemFlock(descriptor, LOCK_UN) }
        return try body()
    }

    private func unverifiableResult(pid: pid_t) throws -> AgentExecutionResult {
        try json([
            "status": "unverifiable",
            "error": "live PID identity could not be verified; state was preserved",
            "pid": Int(pid),
        ], exitCode: 1)
    }

    private func migrateLegacyPID(workerName: String) throws {
        let destination = pidFile(workerName)
        guard !fileManager.fileExists(atPath: destination.path) else { return }
        let legacyPID = workerDirectory.appendingPathComponent(".agent.pid")
        guard fileManager.fileExists(atPath: legacyPID.path) else { return }
        try fileManager.moveItem(at: legacyPID, to: destination)
        let legacyName = workerDirectory.appendingPathComponent(".agent.worker_name")
        if fileManager.fileExists(atPath: legacyName.path) {
            try fileManager.moveItem(at: legacyName, to: nameFile(workerName))
        }
    }

    private func gitSHA() -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["-C", workerDirectory.path, "rev-parse", "--short=8", "HEAD"]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else { return "unknown" }
            return String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            return "unknown"
        }
    }

    private func json(_ object: [String: Any], exitCode: Int32 = 0) throws -> AgentExecutionResult {
        var data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        data.append(0x0A)
        return AgentExecutionResult(output: data, exitCode: exitCode)
    }
}
