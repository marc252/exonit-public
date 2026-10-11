import ExoNitCore
import Foundation

func fail(_ message: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(code)
}

let rawArguments = Array(CommandLine.arguments.dropFirst())
if rawArguments.contains("--help") || rawArguments.contains("-h") {
    FileHandle.standardOutput.write(Data((CLIHelp.text(for: rawArguments) + "\n").utf8))
    exit(0)
}
do {
    let invocation = try InvocationParser.parse(rawArguments)
    switch invocation {
    case let .agent(arguments):
        guard let agentPath = rawArguments.first else {
            fail("missing agent.py compatibility path")
        }
        let workerDirectory = URL(fileURLWithPath: agentPath)
            .standardizedFileURL.deletingLastPathComponent()
        let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
        let result = try AgentController(
            workerDirectory: workerDirectory,
            executableURL: executable
        ).execute(arguments)
        FileHandle.standardOutput.write(result.output)
        exit(result.exitCode)

    case let .worker(arguments):
        let configURL = URL(fileURLWithPath: arguments.configPath).standardizedFileURL
        let data = try Data(contentsOf: configURL)
        let decoded = try JSONDecoder().decode(WorkerConfig.self, from: data)
        var config = try decoded.applying(workerName: arguments.workerName, serverURL: arguments.serverURL)
        let base = configURL.deletingLastPathComponent()
        let executableDirectory = URL(fileURLWithPath: CommandLine.arguments[0])
            .standardizedFileURL.deletingLastPathComponent()
        if !URL(fileURLWithPath: config.fitsCacheDirectory).path.hasPrefix("/") {
            config.fitsCacheDirectory = base.appendingPathComponent(config.fitsCacheDirectory).path
        }
        if let logURL = WorkerLogFile.resolve(config.logFile, relativeTo: base) {
            let timestamp = ISO8601DateFormatter().string(from: Date())
            try WorkerLogFile.append("\(timestamp) starting \(config.workerName)", to: logURL)
            try WorkerLogFile.redirectStandardStreams(to: logURL)
        }
        func modelURL(environment: String, filename: String) -> URL {
            if let explicit = ProcessInfo.processInfo.environment[environment], !explicit.isEmpty {
                return URL(fileURLWithPath: explicit).standardizedFileURL
            }
            return executableDirectory.appendingPathComponent("Models", isDirectory: true)
                .appendingPathComponent(filename, isDirectory: true)
        }
        let stateDirectory = base.appendingPathComponent(".swift-worker-state", isDirectory: true)
        let configuredPools = Set(config.taskPools ?? ["primary"])
        let nativeFPP = configuredPools.contains("fpp_backfill")
            && ProcessInfo.processInfo.environment["EXONIT_SWIFT_FPP"] == "1"
        let needsSidecar = configuredPools.contains("exominer_backfill")
            || (configuredPools.contains("fpp_backfill") && !nativeFPP)
        let sidecarConfiguration: ScienceSidecarConfiguration?
        if needsSidecar {
            let environment = ProcessInfo.processInfo.environment
            let pythonCandidates = [
                environment["EXONIT_SIDECAR_PYTHON"].map { URL(fileURLWithPath: $0) },
                base.appendingPathComponent("venv/bin/python"),
                base.deletingLastPathComponent().appendingPathComponent("venv/bin/python"),
            ].compactMap { $0 }
            guard let python = pythonCandidates.first(where: {
                FileManager.default.isExecutableFile(atPath: $0.path)
            }) else {
                throw ScienceSidecarError.missingFile("EXONIT_SIDECAR_PYTHON or adjacent venv/bin/python")
            }
            let workerPythonDirectory = environment["EXONIT_PYTHON_WORKER_DIR"]
                .map { URL(fileURLWithPath: $0) } ?? base
            let sidecarScript = environment["EXONIT_SCIENCE_SIDECAR"]
                .map { URL(fileURLWithPath: $0) }
                ?? executableDirectory.appendingPathComponent("Sidecars/exonit_science_sidecar.py")
            let exominerONNX = environment["EXONIT_EXOMINER_PLUSPLUS_ONNX"]
                .map { URL(fileURLWithPath: $0) }
                ?? workerPythonDirectory.appendingPathComponent("models/exominer_plusplus_single.onnx")
            sidecarConfiguration = ScienceSidecarConfiguration(
                pythonURL: python, scriptURL: sidecarScript,
                workerPythonDirectory: workerPythonDirectory,
                exominerONNXURL: exominerONNX
            )
        } else { sidecarConfiguration = nil }
        let canaryValue = ProcessInfo.processInfo.environment["EXONIT_SWIFT_CANARY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let canaryAuthorized = canaryValue == "1" || canaryValue == "true"
        let runtime = try WorkerRuntime(
            config: config,
            outboxURL: stateDirectory.appendingPathComponent("outbox-\(config.workerName).sqlite"),
            cnnModel: modelURL(environment: "EXONIT_CNN_COREML_MODEL", filename: "cnn_ffi_v1.mlpackage"),
            exominerModel: modelURL(
                environment: "EXONIT_EXOMINER_COREML_MODEL",
                filename: "exominer_ffi_v6_ensemble.mlpackage"
            ),
            scienceSidecar: sidecarConfiguration,
            canaryAuthorized: canaryAuthorized
        )
        signal(SIGINT, SIG_IGN); signal(SIGTERM, SIG_IGN)
        let workerTask = Task { try await runtime.run() }
        let interrupt = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
        let terminate = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .global())
        interrupt.setEventHandler {
            workerTask.cancel(); Task { await runtime.state.requestShutdown() }
        }
        terminate.setEventHandler {
            workerTask.cancel(); Task { await runtime.state.requestShutdown() }
        }
        interrupt.resume(); terminate.resume()
        do { try await workerTask.value }
        catch is CancellationError { }
    }
} catch {
    fail("exonit-swift-worker: \(error)")
}
