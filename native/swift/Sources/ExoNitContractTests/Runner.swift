import ExoNitCore
import Foundation
import Darwin
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

struct TestFailure: Error, CustomStringConvertible {
    let description: String
}

func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw TestFailure(description: message) }
}

func expectThrows(_ message: String, _ body: () throws -> Void) throws {
    do {
        try body()
        throw TestFailure(description: message)
    } catch is TestFailure {
        throw TestFailure(description: message)
    } catch {
        return
    }
}

final class MockURLProtocol: URLProtocol {
    nonisolated(unsafe) static var handler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let handler = Self.handler else {
            client?.urlProtocol(self, didFailWithError: TestFailure(description: "missing mock handler"))
            return
        }
        do {
            let (response, data) = try handler(request)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}

func mockSession() -> URLSession {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [MockURLProtocol.self]
    return URLSession(configuration: configuration)
}

func bodyData(from request: URLRequest) throws -> Data {
    if let body = request.httpBody { return body }
    guard let stream = request.httpBodyStream else { return Data() }
    stream.open()
    defer { stream.close() }
    var result = Data()
    var buffer = [UInt8](repeating: 0, count: 4_096)
    while stream.hasBytesAvailable {
        let count = stream.read(&buffer, maxLength: buffer.count)
        if count < 0 { throw stream.streamError ?? TestFailure(description: "body stream read") }
        if count == 0 { break }
        result.append(buffer, count: count)
    }
    return result
}

@main
struct ContractTestRunner {
    static func main() async {
        if ProcessInfo.processInfo.environment["EXONIT_AGENT_TEST_CHILD"] == "1" {
            try? await Task.sleep(for: .seconds(30))
            return
        }
        let tests: [(String, () async throws -> Void)] = [
            ("direct worker CLI", testDirectWorkerInvocation),
            ("farmctl-compatible agent CLI", testAgentStartInvocation),
            ("unknown CLI option", testUnknownOption),
            ("agent start requires name", testMissingAgentWorkerName),
            ("worker name path safety", testWorkerNamePathSafety),
            ("config worker name path safety", testConfigWorkerNamePathSafety),
            ("current Python config", testCurrentConfig),
            ("reject non-HTTP URL", testRejectNonHTTPURL),
            ("register HTTP contract", testRegisterHTTPContract),
            ("task HTTP contract", testTaskHTTPContract),
            ("heartbeat command contract", testHeartbeatContract),
            ("unknown heartbeat command fails safe", testUnknownHeartbeatCommand),
            ("heartbeat failure preserves active task", testHeartbeatFailurePreservesActiveTask),
            ("multi-instance config derivation", testMultiInstanceConfigDerivation),
            ("single-instance override semantics", testSingleInstanceOverrides),
            ("BLS Python golden parity", testBLSGoldenParity),
            ("BLS rejects empty duration grid", testBLSEmptyDurationGrid),
            ("Metal BLS single parity", testMetalBLSSingleParity),
            ("Metal BLS batch parity", testMetalBLSBatchParity),
            ("Metal backend policy", testMetalBackendPolicy),
            ("primary automatic backend always CPU", testPrimaryAutomaticAlwaysCPU),
            ("injection backend capability naming", testInjectionBackendCapabilityNaming),
            ("injection automatic backend is decision-identical to CPU/Metal", testInjectionAutomaticBackendPolicy),
            ("BLS two-pass Python parity", testBLSTwoPassParity),
            ("vetting Python golden parity", testVettingGoldenParity),
            ("CNN FFI view golden parity", testCNNFFIViewGoldenParity),
            ("FPP stellar golden parity", { try testFPPStellarGoldenParity() }),
            ("FPP transit golden parity", { try testFPPTransitGoldenParity() }),
            ("FPP prior golden parity", { try testFPPPriorGoldenParity() }),
            ("FPP kernel golden parity", { try testFPPKernelGoldenParity() }),
            ("FPP native e2e 10756814", { try testFPPE2E10756814() }),
            ("CNN Core ML inference parity", testCNNCoreMLInferenceParity),
            ("ExoMiner FFI Core ML parity", testExoMinerFFICoreMLParity),
            ("FPP nearby galactic latitude parity", testFPPNearbyGalacticLatitudeParity),
            ("FPP gate and verdict parity", testFPPPolicyParity),
            ("injection primitives Python parity", testInjectionPrimitivesParity),
            ("injection recovery Python golden parity", testInjectionRecoveryGoldenParity),
            ("long-period sweep selection", testLongPeriodSweep),
            ("long-period sweep Python golden parity", testLongPeriodSweepPythonGoldenParity),
            ("task scientific options parity", testTaskScientificOptions),
            ("primary failure and cancellation semantics", testPrimaryFailureAndCancellation),
            ("runtime requires explicit canary authorization", testRuntimeCanaryAuthorization),
            ("science sidecar JSON contract", testScienceSidecarContract),
            ("Swift science sidecar runner", testScienceSidecarRunner),
            ("science sidecar preflight", testScienceSidecarPreflight),
            ("science sidecar deadline kills child", testScienceSidecarDeadline),
            ("backfill transient failure propagates", testBackfillTransientFailurePropagates),
            ("agent start guard", testAgentStartGuard),
            ("agent stopped status", testAgentStoppedStatus),
            ("agent stale PID identity", testAgentStalePIDIdentity),
            ("agent preserves unverifiable live PID", testAgentUnverifiablePID),
            ("agent command-line fallback owns and stops", testAgentCommandLineFallbackOwned),
            ("agent command-line fallback rejects foreign name", testAgentCommandLineFallbackForeignName),
            ("agent start is serialized", testAgentConcurrentStart),
            ("agent log tail", testAgentLogTail),
            ("worker log_file sink", testWorkerLogFile),
            ("TGLC FITS Python golden parity", testTGLCFITSGoldenParity),
            ("QLP FITS Lightkurve golden parity", testQLPFITSGoldenParity),
            ("SPOC FITS Lightkurve golden parity", testSPOCFITSGoldenParity),
            ("Wotan biweight golden parity", testWotanBiweightGoldenParity),
            ("iterative detrend golden parity", testIterativeDetrendGoldenParity),
            ("QLP preprocessing pipeline parity", testQLPPipelineGoldenParity),
            ("TGLC preprocessing pipeline parity (TIC 10756829)", testTGLCPreprocessGoldenParity),
            ("multi-planet 5-iteration golden parity (TIC 10756829)", testMultiPlanetFiveIterationGoldenParity),
            ("stitch normalization semantics", testStitchNormalizationSemantics),
            ("stitch out-of-order sectors sorted before gap fill", testStitchOutOfOrderSectorsSorted),
            ("stitch out-of-order non-cadence stitch not reordered", testStitchOutOfOrderNonCadenceNotReordered),
            ("gap fill cadence and gate semantics", testGapFillSemantics),
            ("extended protocol DTOs", testExtendedProtocolDTOs),
            ("extended API routes", testExtendedAPIRoutes),
            ("binary API routes", testBinaryAPIRoutes),
            ("durable outbox", testDurableOutbox),
            ("atomic outbox claim", testAtomicOutboxClaim),
            ("outbox retry delivery", testOutboxRetryDelivery),
            ("outbox transport ambiguous retries as pending", testOutboxTransportAmbiguousRetriesAsPending),
            ("outbox legacy unknown reconciled on open", testOutboxLegacyUnknownReconciledOnOpen),
            ("runtime reconciles legacy unknown at startup", testRuntimeReconcilesLegacyUnknownAtStartup),
            ("CLI help", testCLIHelp),
        ]
        var failures = 0
        for (name, test) in tests {
            do {
                try await test()
                print("PASS \(name)")
            } catch {
                failures += 1
                print("FAIL \(name): \(error)")
            }
        }
        print("RESULT \(tests.count - failures)/\(tests.count) passed")
        if failures != 0 { exit(1) }
    }
}

func testDirectWorkerInvocation() async throws {
    let invocation = try InvocationParser.parse([
        "--config", "/tmp/worker.json", "--worker-name", "worker-<host>-1",
        "--server-url", "http://127.0.0.1:18400",
    ])
    guard case let .worker(args) = invocation else {
        throw TestFailure(description: "expected worker invocation")
    }
    try expect(args.configPath == "/tmp/worker.json", "config path")
    try expect(args.workerName == "worker-<host>-1", "worker name")
    try expect(args.serverURL == "http://127.0.0.1:18400", "server URL")
}

func testAgentStartInvocation() async throws {
    let invocation = try InvocationParser.parse([
        "~/exonit-worker/agent.py", "start", "worker-<host>-1",
        "--server-url", "http://127.0.0.1:18400", "--instances", "4",
        "--instance-index", "1", "--config-overrides-b64", "e30=", "--allow-start",
    ])
    guard case let .agent(args) = invocation else {
        throw TestFailure(description: "expected agent invocation")
    }
    try expect(args.action == .start, "action")
    try expect(args.workerName == "worker-<host>-1", "worker name")
    try expect(args.serverURL == "http://127.0.0.1:18400", "server URL")
    try expect(args.instances == 4, "instances")
    try expect(args.instanceIndex == 1, "instance index")
    try expect(args.configOverridesBase64 == "e30=", "config overrides")
    try expect(args.allowStart, "allow-start")
}

func testUnknownOption() async throws {
    try expectThrows("unknown option must fail") { _ = try InvocationParser.parse(["--wat"]) }
}

func testMissingAgentWorkerName() async throws {
    try expectThrows("agent start without worker name must fail") {
        _ = try InvocationParser.parse(["agent.py", "start", "--allow-start"])
    }
}

func testWorkerNamePathSafety() async throws {
    for unsafe in ["../victim", "a/b", ".hidden", "worker..other", "", String(repeating: "x", count: 129)] {
        try expectThrows("unsafe worker name must fail: \(unsafe)") {
            _ = try InvocationParser.parse(["agent.py", "stop", unsafe])
        }
    }
    let safe = try InvocationParser.parse(["agent.py", "status", "worker-m4_1.test"])
    guard case let .agent(arguments) = safe else { throw TestFailure(description: "safe agent parse") }
    try expect(arguments.workerName == "worker-m4_1.test", "safe worker name preserved")
}

func testConfigWorkerNamePathSafety() async throws {
    for unsafe in ["../victim", "a/b", ".hidden", "worker..other"] {
        let config = WorkerConfig(
            serverURL: URL(string: "http://127.0.0.1:18400")!,
            token: "test-token", workerName: unsafe, taskPools: ["primary"]
        )
        try expectThrows("unsafe config worker name must fail: \(unsafe)") {
            _ = try config.validated()
        }
    }
}

func testCurrentConfig() async throws {
    let json = """
    {"server_url":"http://localhost:8400","token":"test-token",
     "worker_name":"worker-default","cpu_threads":2,"gpu_enabled":true,
     "poll_interval_seconds":5,"heartbeat_interval_seconds":30,
     "fits_cache_dir":"fits_cache","gui_port":8402,"fpp_cpu_cores":2,
     "log_file":"worker.log","log_level":"INFO","exominer_mode":"native",
     "ffi_fallback_on_load_fail":true}
    """.data(using: .utf8)!
    let decoded = try JSONDecoder().decode(WorkerConfig.self, from: json)
    let effective = try decoded.applying(
        workerName: "worker-<host>-1", serverURL: "http://127.0.0.1:18400"
    )
    try expect(effective.workerName == "worker-<host>-1", "effective name")
    try expect(effective.serverURL.absoluteString == "http://127.0.0.1:18400", "effective URL")
    try expect(effective.token == "test-token", "token")
    try expect(effective.cpuThreads == 2, "cpu threads")
    try expect(effective.gpuEnabled, "GPU")
    try expect(effective.exominerMode == "native", "ExoMiner mode")
}

func testRejectNonHTTPURL() async throws {
    let config = WorkerConfig(
        serverURL: URL(string: "file:///tmp/server")!, token: "token", workerName: "worker"
    )
    try expectThrows("file URL must fail") { _ = try config.validated() }
}

func testRegisterHTTPContract() async throws {
    MockURLProtocol.handler = { request in
        try expect(request.httpMethod == "POST", "register method")
        try expect(request.url?.path == "/api/worker/register", "register path")
        try expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer test-token", "auth")
        try expect(request.value(forHTTPHeaderField: "Content-Type") == "application/json", "content type")
        let body = try JSONSerialization.jsonObject(with: bodyData(from: request)) as? [String: Any]
        try expect(body?["name"] as? String == "worker-m4", "register name")
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 200, httpVersion: nil,
            headerFields: ["Content-Type": "application/json"]
        )!
        return (response, Data(#"{"worker_id":"wid-123"}"#.utf8))
    }
    let client = WorkerAPIClient(
        baseURL: URL(string: "http://127.0.0.1:18400")!, token: "test-token", session: mockSession()
    )
    let id = try await client.register(RegisterRequest(
        name: "worker-m4",
        capabilities: WorkerCapabilities(
            gpu: "metal", blsBackend: "metal-native", blsGPUInfo: "Apple M4",
            fppBackend: "metal-native", cpuCores: 10, fppCPUCores: 2
        )
    ))
    try expect(id == "wid-123", "worker id")
}

func testTaskHTTPContract() async throws {
    MockURLProtocol.handler = { request in
        try expect(request.httpMethod == "GET", "task method")
        try expect(request.url?.path == "/api/worker/task", "task path")
        try expect(request.value(forHTTPHeaderField: "X-Worker-ID") == "wid-123", "worker header")
        let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
        return (response, Data(#"{"task_id":"t-1","tic_id":"150430484"}"#.utf8))
    }
    let client = WorkerAPIClient(
        baseURL: URL(string: "http://127.0.0.1:18400")!, token: "test-token", session: mockSession()
    )
    let task = try await client.getTask(workerID: "wid-123")
    try expect(task?.taskID == "t-1", "task id")
    try expect(task?.ticID == "150430484", "tic id")
}

func testHeartbeatContract() async throws {
    MockURLProtocol.handler = { request in
        try expect(request.url?.path == "/api/worker/heartbeat", "heartbeat path")
        try expect(request.value(forHTTPHeaderField: "X-Worker-ID") == "wid-123", "worker header")
        let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
        return (response, Data(#"{"command":"pause"}"#.utf8))
    }
    let client = WorkerAPIClient(
        baseURL: URL(string: "http://127.0.0.1:18400")!, token: "test-token", session: mockSession()
    )
    let command = try await client.heartbeat(HeartbeatRequest(
        workerID: "wid-123", status: "idle", currentTIC: nil, phase: nil,
        starsDone: 0, uptimeSeconds: 1, pid: 42
    ))
    try expect(command == .pause, "pause command")
}

func testUnknownHeartbeatCommand() async throws {
    MockURLProtocol.handler = { request in
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil
        )!
        return (response, Data(#"{"command":"format_disk"}"#.utf8))
    }
    let client = WorkerAPIClient(
        baseURL: URL(string: "http://127.0.0.1:18400")!,
        token: "test-token", session: mockSession()
    )
    do {
        _ = try await client.heartbeat(HeartbeatRequest(
            workerID: "wid-123", status: "idle", currentTIC: nil, phase: nil,
            starsDone: 0, uptimeSeconds: 1, pid: 42
        ))
        throw TestFailure(description: "unknown heartbeat command accepted")
    } catch WorkerAPIError.malformedResponse {
        // Expected: an unknown control command must be visible and fail-safe.
    }
}

func testHeartbeatFailurePreservesActiveTask() async throws {
    let state = WorkerRuntimeState()
    await state.begin(tic: "123", phase: "primary")
    await state.heartbeatFailed("temporary timeout")
    let currentTIC = await state.currentTIC
    let phase = await state.phase
    let lastError = await state.lastError
    try expect(currentTIC == "123", "heartbeat failure must preserve TIC")
    try expect(phase == "primary", "heartbeat failure must preserve phase")
    try expect(lastError == "temporary timeout", "heartbeat error remains observable")
}

func testMultiInstanceConfigDerivation() async throws {
    let directory = URL(fileURLWithPath: "/tmp/exonit-agent-fixture", isDirectory: true)
    let base = Data(#"{"server_url":"http://localhost:8400","token":"t","worker_name":"base","gui_port":8402,"fits_cache_dir":"fits_cache","cpu_threads":9,"fpp_cpu_cores":9}"#.utf8)
    let overrides = Data(#"{"gpu_enabled":true,"gui_port":9999,"cpu_threads":99}"#.utf8)
    let result = try AgentConfigDeriver.derive(
        baseConfigData: base, overridesData: overrides,
        workerName: "worker-m4-3", instanceIndex: 3, instances: 4,
        workerDirectory: directory, cpuCount: 10
    )
    let object = try JSONSerialization.jsonObject(with: result) as! [String: Any]
    try expect(object["worker_name"] as? String == "worker-m4-3", "derived name")
    try expect(object["gui_port"] as? Int == 8404, "base GUI port wins over override")
    try expect(object["gui_enabled"] as? Bool == false, "secondary GUI disabled")
    try expect(object["cpu_threads"] as? Int == 2, "75% CPU share")
    try expect(object["fpp_cpu_cores"] as? Int == 2, "FPP CPU share")
    try expect(object["gpu_enabled"] as? Bool == true, "host override retained")
    try expect(object["fits_cache_dir"] as? String == "/tmp/exonit-agent-fixture/fits_cache_3", "cache path")
    try expect(object["log_file"] as? String == "worker-worker-m4-3.log", "log path")
}

func testSingleInstanceOverrides() async throws {
    let base = Data(#"{"server_url":"http://localhost:8400","token":"t","worker_name":"base","gui_port":8402,"cpu_threads":6}"#.utf8)
    let overrides = Data(#"{"gpu_enabled":true,"cpu_threads":4}"#.utf8)
    let result = try AgentConfigDeriver.applySingleInstanceOverrides(
        baseConfigData: base, overridesData: overrides
    )
    let object = try JSONSerialization.jsonObject(with: result) as! [String: Any]
    try expect(object["worker_name"] as? String == "base", "single instance name untouched")
    try expect(object["gui_port"] as? Int == 8402, "single GUI untouched")
    try expect(object["cpu_threads"] as? Int == 4, "single override applied")
    try expect(object["gpu_enabled"] as? Bool == true, "single GPU override")
}

func testBLSGoldenParity() async throws {
    let period = 10.0 / 3.0
    let time = (0..<360).map { Double($0) / 12.0 }
    let flux = time.map { value -> Double in
        let phase = (value / period).truncatingRemainder(dividingBy: 1.0)
        return (phase >= 0.20 && phase < 0.28) ? 0.988 : 1.0
    }
    let result = try BLSReference.search(
        time: time,
        flux: flux,
        parameters: BLSParameters(
            periodMinimum: 2.0, periodMaximum: 5.0, periodCount: 121,
            binCount: 50, durationStepCount: 7,
            durationMinimumFraction: 0.04, durationMaximumFraction: 0.12
        )
    )
    try expect(abs(result.period - 3.3333333333333335) < 1e-14, "period parity")
    try expect(abs(result.power - 0.003700400000000247) < 1e-14, "power parity")
    try expect(abs(result.duration - 0.26666666666666666) < 1e-14, "duration parity")
    try expect(abs(result.t0 - 0.6666666666666667) < 1e-14, "t0 parity")
    try expect(abs(result.depth - 0.011600000000000388) < 1e-14, "depth parity")
    try expect(abs(result.snr - 84.16049628391373) < 1e-10, "SDE parity")

    let optimized = try BLSCPUOptimized.search(
        time: time,
        flux: flux,
        parameters: BLSParameters(
            periodMinimum: 2.0, periodMaximum: 5.0, periodCount: 121,
            binCount: 50, durationStepCount: 7,
            durationMinimumFraction: 0.04, durationMaximumFraction: 0.12
        ),
        maxConcurrency: 4
    )
    try expect(optimized == result, "parallel CPU path must be bit-identical")
}

func testBLSEmptyDurationGrid() async throws {
    let parameters = BLSParameters(
        periodMinimum: 1, periodMaximum: 2, periodCount: 10,
        binCount: 3, durationStepCount: 2,
        durationMinimumFraction: 0.9, durationMaximumFraction: 0.9
    )
    try expect(
        BLSReference.durationBinGrid(parameters: parameters).isEmpty,
        "an impossible duration range must not produce descending or zero-width bins"
    )
    do {
        _ = try BLSReference.search(
            time: [0, 1, 2, 3, 4, 5], flux: [1, 1, 1, 1, 1, 1],
            parameters: parameters
        )
        throw TestFailure(description: "empty duration grid must be rejected")
    } catch let error as BLSError {
        guard case .invalidParameters = error else { throw error }
    }
    do {
        _ = try BLSCPUOptimized.search(
            time: [0, 1, 2, 3, 4, 5], flux: [1, 1, 1, 1, 1, 1],
            parameters: parameters, maxConcurrency: 2
        )
        throw TestFailure(description: "optimized CPU must reject empty duration grid")
    } catch let error as BLSError {
        guard case .invalidParameters = error else { throw error }
    }
    do {
        _ = try BLSMetal.search(
            time: [0, 1, 2, 3, 4, 5], flux: [1, 1, 1, 1, 1, 1],
            parameters: parameters
        )
        throw TestFailure(description: "Metal must reject empty duration grid")
    } catch let error as BLSError {
        guard case .invalidParameters = error else { throw error }
    }
}

func testMetalBLSSingleParity() async throws {
    try expect(BLSMetal.isAvailable, "Metal BLS must be available on the Apple Silicon worker")
    let period = 10.0 / 3.0
    let time = (0..<720).map { Double($0) / 24.0 }
    let flux = time.map { value -> Double in
        let phase = (value / period).truncatingRemainder(dividingBy: 1)
        return phase >= 0.20 && phase < 0.28 ? 0.988 : 1
    }
    let parameters = BLSParameters(
        periodMinimum: 2, periodMaximum: 5, periodCount: 1_001,
        binCount: 150, durationStepCount: 10,
        durationMinimumFraction: 0.02, durationMaximumFraction: 0.12
    )
    let cpu = try BLSCPUOptimized.search(
        time: time, flux: flux, parameters: parameters, maxConcurrency: 4
    )
    let metal = try BLSMetal.search(time: time, flux: flux, parameters: parameters)
    try expect(metal.period == cpu.period, "Metal and CPU must select the same period")
    try expect(metal.t0 == cpu.t0, "Metal and CPU must select the same phase bin")
    try expect(metal.duration == cpu.duration, "Metal and CPU must select the same duration bin")
    try expect(abs(metal.depthPPM - cpu.depthPPM) < 2, "Metal depth must stay within 2 ppm")
    try expect(abs(metal.snr - cpu.snr) < 0.02, "Metal SDE parity")
}

func testMetalBLSBatchParity() async throws {
    let time = (0..<600).map { Double($0) / 24.0 }
    let periods = [2.75, 4.125, 7.5]
    let fluxes = periods.map { period in
        time.map { value -> Double in
            let phase = (value / period).truncatingRemainder(dividingBy: 1)
            return phase >= 0.15 && phase < 0.21 ? 0.992 : 1
        }
    }
    let parameters = BLSParameters(
        periodMinimum: 2, periodMaximum: 9, periodCount: 1_501,
        binCount: 150, durationStepCount: 10,
        durationMinimumFraction: 0.02, durationMaximumFraction: 0.10
    )
    let metal = try BLSMetal.searchBatch(
        time: time, fluxes: fluxes, parameters: parameters
    )
    try expect(metal.count == fluxes.count, "Metal batch result count")
    for index in fluxes.indices {
        let cpu = try BLSCPUOptimized.search(
            time: time, flux: fluxes[index], parameters: parameters, maxConcurrency: 4
        )
        try expect(metal[index].period == cpu.period, "batch period parity \(index)")
        try expect(metal[index].t0 == cpu.t0, "batch phase parity \(index)")
        try expect(metal[index].duration == cpu.duration, "batch duration parity \(index)")
        try expect(abs(metal[index].depthPPM - cpu.depthPPM) < 2, "batch depth parity \(index)")
    }
}

func testMetalBackendPolicy() async throws {
    try expect(BLSComputeBackend.configured(gpuEnabled: true, environment: [:]) == .automatic, "gpu_enabled must select Metal automatically")
    try expect(BLSComputeBackend.configured(gpuEnabled: false, environment: [:]) == .cpu, "gpu disabled must select CPU")
    try expect(BLSComputeBackend.configured(gpuEnabled: true, environment: ["EXONIT_BLS_BACKEND": "cpu"]) == .cpu, "CPU kill switch")
    try expect(BLSComputeBackend.configured(gpuEnabled: false, environment: ["EXONIT_BLS_BACKEND": "metal"]) == .metal, "explicit Metal override")
    try expect(BLSComputeBackend.injectionConfigured(gpuEnabled: true, environment: ["EXONIT_INJECTION_METAL": "0"]) == .cpu, "injection Metal kill switch")
    try expect(BLSComputeBackend.injectionConfigured(gpuEnabled: true, environment: [:]) == .automatic, "injection Metal default")
}

/// Pins the primary-search policy decided 2026-08-04 (docs/WORKLOG.md):
/// Metal-primary is pending its own science gate (Float32 flips marginal
/// decisions vs the Float64 CPU two-pass reference, measured at 64.9%
/// recovery agreement under the old gate — see `BLSAccelerated.search`).
/// `.automatic` must therefore always dispatch the primary search on CPU,
/// even at production-scale period counts (~10,000), regardless of Metal
/// availability. Only the explicit `EXONIT_BLS_BACKEND=metal` opt-in
/// (`.metal` case) may take the primary through the Metal kernel.
func testPrimaryAutomaticAlwaysCPU() async throws {
    let time = (0..<600).map { Double($0) / 24.0 }
    let period = 4.125
    let flux = time.map { value -> Double in
        let phase = (value / period).truncatingRemainder(dividingBy: 1)
        return phase >= 0.15 && phase < 0.21 ? 0.992 : 1
    }
    let parameters = BLSParameters(
        periodMinimum: 2, periodMaximum: 9, periodCount: 10_000,
        binCount: 150, durationStepCount: 10,
        durationMinimumFraction: 0.02, durationMaximumFraction: 0.10
    )
    let cpuExplicit = try BLSAccelerated.search(
        time: time, flux: flux, parameters: parameters, backend: .cpu, maxConcurrency: 4
    )
    let automatic = try BLSAccelerated.search(
        time: time, flux: flux, parameters: parameters, backend: .automatic, maxConcurrency: 4
    )
    try expect(
        automatic == cpuExplicit,
        "primary .automatic at periodCount=10,000 must be bit-identical to explicit .cpu (no Metal dispatch)"
    )
    if BLSMetal.isAvailable {
        // Not compared to cpuExplicit for period/decision parity: Metal Float32 is
        // documented to flip marginal decisions vs CPU (this is exactly why .automatic
        // no longer routes here). This branch only proves the explicit opt-in path
        // still resolves and dispatches successfully after the .automatic routing change.
        let metalExplicit = try BLSAccelerated.search(
            time: time, flux: flux, parameters: parameters, backend: .metal, maxConcurrency: 4
        )
        try expect(
            BLSComputeBackend.configured(gpuEnabled: true, environment: ["EXONIT_BLS_BACKEND": "metal"]) == .metal,
            "explicit EXONIT_BLS_BACKEND=metal must still resolve to .metal for the primary"
        )
        try expect(
            metalExplicit.period > 0,
            "explicit .metal primary opt-in path must still dispatch and return a result"
        )
    }
}

/// Pure-function coverage for the capability strings ``WorkerRuntime``
/// reports to the server at register() time (see
/// `WorkerRuntime.swift:register()`), independent of any network mock.
func testInjectionBackendCapabilityNaming() async throws {
    try expect(
        BLSComputeBackend.blsCapabilityName(backend: .automatic, metalAvailable: true)
            == "swift-float64-cpu-two-pass",
        "primary capability name for .automatic must report CPU — Metal-primary is opt-in only (EXONIT_BLS_BACKEND=metal)"
    )
    try expect(
        BLSComputeBackend.blsCapabilityName(backend: .cpu, metalAvailable: true)
            == "swift-float64-cpu-two-pass",
        "primary capability name when CPU-forced despite Metal availability"
    )
    try expect(
        BLSComputeBackend.blsCapabilityName(backend: .automatic, metalAvailable: false)
            == "swift-float64-cpu-two-pass",
        "primary capability name when Metal unavailable"
    )
    try expect(
        BLSComputeBackend.blsCapabilityName(backend: .metal, metalAvailable: true)
            == "swift-metal-f32-single-pass",
        "primary capability name when Metal is the explicit opt-in backend"
    )
    try expect(
        BLSComputeBackend.injectionCapabilityName(backend: .automatic, metalAvailable: true)
            == "swift-metal-f32-batch",
        "injection capability name when Metal active (validated default, 2026-08-04)"
    )
    try expect(
        BLSComputeBackend.injectionCapabilityName(backend: .cpu, metalAvailable: true)
            == "swift-float64-cpu",
        "injection capability name with EXONIT_INJECTION_METAL=0 kill switch"
    )
    try expect(
        BLSComputeBackend.injectionCapabilityName(backend: .automatic, metalAvailable: false)
            == "swift-float64-cpu",
        "injection capability name when Metal unavailable"
    )
}

/// Pins the validated production policy: injection-recovery's default
/// (`.automatic`, no `EXONIT_INJECTION_METAL` override, the resolution
/// every real Apple Silicon deployment with `gpu_enabled=true` gets) must
/// actually dispatch through the Metal batch kernel and stay
/// decision-identical to the CPU path — not just resolve to the `.automatic`
/// enum case in isolation (that mechanism is covered by
/// `testMetalBackendPolicy`). Reuses the same golden fixture as
/// `testInjectionRecoveryGoldenParity`, which already proves `.metal`
/// explicit == CPU/Python; this test adds `.automatic` == `.metal` == CPU,
/// closing the loop that what production actually runs by default is what
/// was validated offline at scale (13 TICs, 7,020 injections, 100%
/// decision-identical — see docs/WORKLOG.md 2026-08-04).
func testInjectionAutomaticBackendPolicy() async throws {
    struct Golden: Decodable {
        let time: [Double]
        let flux: [Double]
        let stellarRadius: Double
        let radiusGrid: [Double]
        let periodGrid: [Double]
        let injectionsPerCell: Int
        let periodCount: Int
        let binCount: Int
        let durationStepCount: Int
        let minimumSNR: Double

        enum CodingKeys: String, CodingKey {
            case time, flux, stellarRadius = "stellar_radius"
            case radiusGrid = "radius_grid", periodGrid = "period_grid"
            case injectionsPerCell = "injections_per_cell"
            case periodCount = "bls_n_periods", binCount = "bls_n_bins"
            case durationStepCount = "bls_n_dur_steps", minimumSNR = "snr_threshold"
        }
    }
    let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let golden = try JSONDecoder().decode(
        Golden.self,
        from: Data(contentsOf: root.appendingPathComponent("Fixtures/injection_recovery_golden.json"))
    )
    var options = InjectionRecoveryOptions()
    options.stellarRadiusSolar = golden.stellarRadius
    options.radiusGridEarth = golden.radiusGrid
    options.periodGridDays = golden.periodGrid
    options.injectionsPerCell = golden.injectionsPerCell
    options.periodCount = golden.periodCount
    options.binCount = golden.binCount
    options.durationStepCount = golden.durationStepCount
    options.minimumSNR = golden.minimumSNR

    try expect(BLSMetal.isAvailable, "this contract test requires Metal hardware (runs on the Mac)")
    let automaticBackend = BLSComputeBackend.injectionConfigured(gpuEnabled: true, environment: [:])
    try expect(automaticBackend == .automatic, "no-override injection backend must resolve to .automatic")

    let cpuResult = try InjectionRecovery.run(
        time: golden.time, flux: golden.flux, options: options, maxConcurrency: 1, backend: .cpu
    )
    let automaticResult = try InjectionRecovery.run(
        time: golden.time, flux: golden.flux, options: options,
        maxConcurrency: 1, backend: automaticBackend
    )
    try expect(
        automaticResult.completenessMap == cpuResult.completenessMap,
        "default (.automatic) injection backend must be decision-identical to CPU"
    )
    try expect(
        automaticResult.details.map(\.recoveredCount) == cpuResult.details.map(\.recoveredCount),
        "default (.automatic) injection recovered counts must be decision-identical to CPU"
    )
}

func withTemporaryDirectory<T>(_ body: (URL) throws -> T) throws -> T {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("exonit-swift-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: url) }
    return try body(url)
}

func decodedObject(_ output: Data) throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(with: output) as? [String: Any] else {
        throw TestFailure(description: "agent output is not a JSON object")
    }
    return object
}

func testBLSTwoPassParity() async throws {
    let period = 10.0 / 3.0
    let time = (0..<360).map { Double($0) / 12.0 }
    let flux = time.map { value -> Double in
        let phase = (value / period).truncatingRemainder(dividingBy: 1)
        return phase >= 0.20 && phase < 0.28 ? 0.988 : 1
    }
    let result = try BLSTwoPass.search(
        time: time, flux: flux,
        parameters: BLSParameters(periodCount: 10_000, binCount: 120, durationStepCount: 10),
        maxConcurrency: 2
    )
    try expect(abs(result.period - 3.326623878947138) < 1e-14, "two-pass period")
    try expect(abs(result.power - 0.0035533297297300143) < 1e-14, "two-pass power")
    try expect(abs(result.duration - 0.27721865657892814) < 1e-14, "two-pass duration")
    try expect(abs(result.t0 - 0.7207685071052132) < 1e-14, "two-pass t0")
    try expect(abs(result.depth - 0.011927927927928406) < 1e-14, "two-pass depth")
    try expect(abs(result.snr - 49.34005822782964) < 1e-10, "two-pass coarse SNR")
    try expect(result.periods.count == 5_000, "two-pass reports coarse spectrum")
}

func testVettingGoldenParity() async throws {
    struct Fixture: Decodable {
        let time: [Double]
        let flux: [Double]
        let candidates: [CandidateResult]
        let expected: [VettingResult]
        let oddEvenFlux: [Double]
        let oddEvenCandidates: [CandidateResult]
        let oddEvenExpected: VettingResult

        enum CodingKeys: String, CodingKey {
            case time, flux, candidates, expected
            case oddEvenFlux = "odd_even_flux"
            case oddEvenCandidates = "odd_even_candidates"
            case oddEvenExpected = "odd_even_expected"
        }
    }
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
    let fixture = try JSONDecoder().decode(
        Fixture.self,
        from: Data(contentsOf: root.appendingPathComponent("Fixtures/vetting_golden.json"))
    )
    let result = try CandidateVetter.vetAll(
        candidates: fixture.candidates, time: fixture.time, flux: fixture.flux
    )
    try expect(result == fixture.expected, "main vetting fixture")
    let oddEven = try CandidateVetter.vetAll(
        candidates: fixture.oddEvenCandidates, time: fixture.time, flux: fixture.oddEvenFlux
    )
    try expect(oddEven == [fixture.oddEvenExpected], "odd/even vetting fixture")
}

func testCNNFFIViewGoldenParity() async throws {
    struct Fixture: Decodable {
        let period: Double
        let t0: Double
        let duration: Double
        let time: [Double]
        let flux: [Double]
        let globalView: [Float]
        let localView: [Float]
        let logit: Double
        let probability: Double

        enum CodingKeys: String, CodingKey {
            case period, t0, duration, time, flux, logit, probability
            case globalView = "global_view"
            case localView = "local_view"
        }
    }
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
    let fixture = try JSONDecoder().decode(
        Fixture.self,
        from: Data(contentsOf: root.appendingPathComponent("Fixtures/cnn_ffi_golden.json"))
    )
    let views = try FFIViewGenerator.makeCNNViews(
        time: fixture.time, flux: fixture.flux, period: fixture.period,
        t0: fixture.t0, duration: fixture.duration
    )
    try expect(views.global.count == 201 && views.local.count == 61, "CNN view shapes")
    let globalError = zip(views.global, fixture.globalView).map { abs(Double($0) - Double($1)) }.max() ?? 0
    let localError = zip(views.local, fixture.localView).map { abs(Double($0) - Double($1)) }.max() ?? 0
    try expect(globalError < 2e-6, "CNN global view parity \(globalError)")
    try expect(localError < 2e-6, "CNN local view parity \(localError)")
    try expect(
        abs(FFIViewGenerator.sigmoid(logit: fixture.logit) - fixture.probability) < 1e-15,
        "CNN sigmoid parity"
    )
}

func testCNNCoreMLInferenceParity() async throws {
    struct Fixture: Decodable {
        let globalView: [Float]
        let localView: [Float]
        let logit: Double
        let probability: Double
        enum CodingKeys: String, CodingKey {
            case globalView = "global_view", localView = "local_view", logit, probability
        }
    }
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
    let fixture = try JSONDecoder().decode(
        Fixture.self,
        from: Data(contentsOf: root.appendingPathComponent("Fixtures/cnn_ffi_golden.json"))
    )
    let modelURL = root.appendingPathComponent(".artifacts/coreml/cnn_ffi_v1.mlpackage")
    guard FileManager.default.fileExists(atPath: modelURL.path) else {
        throw TestFailure(description: "converted CNN Core ML fixture missing")
    }
    let classifier = try CoreMLFFIClassifier(modelURL: modelURL)
    let result = try classifier.classify(
        views: CNNFFIViews(global: fixture.globalView, local: fixture.localView)
    )
    try expect(abs(result.logit - fixture.logit) < 2e-5, "Core ML CNN logit parity")
    try expect(abs(result.probability - fixture.probability) < 2e-5, "Core ML CNN score parity")
}

func testExoMinerFFICoreMLParity() async throws {
    struct StellarRaw: Decodable { let tmag: Double; let teff: Double; let logg: Double; let rstar: Double }
    struct Fixture: Decodable {
        let period: Double; let t0: Double; let duration: Double
        let time: [Double]; let flux: [Double]
        let auxViews: [[Float]]; let stellarRaw: StellarRaw
        let stellar: [Float]; let tce: [Float]
        let exominerLogit: Double; let exominerRawProbability: Double; let exominerProbability: Double
        enum CodingKeys: String, CodingKey {
            case period, t0, duration, time, flux, stellar, tce
            case auxViews = "aux_views", stellarRaw = "stellar_raw"
            case exominerLogit = "exominer_logit"
            case exominerRawProbability = "exominer_raw_probability"
            case exominerProbability = "exominer_probability"
        }
    }
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
    let fixture = try JSONDecoder().decode(
        Fixture.self,
        from: Data(contentsOf: root.appendingPathComponent("Fixtures/cnn_ffi_golden.json"))
    )
    let features = try ExoMinerFFIFeatureBuilder.make(
        time: fixture.time, flux: fixture.flux,
        period: fixture.period, t0: fixture.t0, duration: fixture.duration,
        depthPPM: 3000, snr: 18,
        stellar: FFIStellarMetadata(
            tmag: fixture.stellarRaw.tmag, teff: fixture.stellarRaw.teff,
            logg: fixture.stellarRaw.logg, radius: fixture.stellarRaw.rstar, mass: nil
        )
    )
    let auxError = zip(features.auxiliary.flatMap { $0 }, fixture.auxViews.flatMap { $0 })
        .map { abs(Double($0) - Double($1)) }.max() ?? 0
    let stellarError = zip(features.stellar, fixture.stellar).map { abs(Double($0) - Double($1)) }.max() ?? 0
    let tceError = zip(features.tce, fixture.tce).map { abs(Double($0) - Double($1)) }.max() ?? 0
    try expect(auxError < 2e-6 && stellarError < 2e-6 && tceError < 2e-6, "ExoMiner feature parity")
    let classifier = try CoreMLExoMinerFFI(
        modelURL: root.appendingPathComponent(".artifacts/coreml/exominer_ffi_v6_ensemble.mlpackage"),
        calibration: PiecewiseLinearCalibration(
            x: [0, 0.38930313157914914, 1], y: [0, 0.5, 1]
        )
    )
    let score = try classifier.classify(features: features)
    try expect(abs(score.logit - fixture.exominerLogit) < 2e-5, "ExoMiner logit parity")
    try expect(abs(score.rawProbability - fixture.exominerRawProbability) < 2e-5, "ExoMiner raw probability parity")
    try expect(abs(score.probability - fixture.exominerProbability) < 2e-5, "ExoMiner calibration parity")
}

func testFPPPolicyParity() async throws {
    try expect(FPPPolicy.shouldRun(vettingPassed: true, snr: 10, cnnScore: nil), "FPP nil CNN")
    try expect(FPPPolicy.shouldRun(vettingPassed: true, snr: 10, cnnScore: 0.7), "FPP inclusive gates")
    try expect(!FPPPolicy.shouldRun(vettingPassed: false, snr: 20, cnnScore: 0.9), "FPP vetting gate")
    try expect(!FPPPolicy.shouldRun(vettingPassed: true, snr: 9.999, cnnScore: 0.9), "FPP SNR gate")
    try expect(!FPPPolicy.shouldRun(vettingPassed: true, snr: 20, cnnScore: 0.699), "FPP CNN gate")
    try expect(FPPPolicy.verdict(fpp: 0.5, nfpp: 0, highCrowding: false) == .likelyFalsePositive, "FPP likely FP boundary")
    try expect(FPPPolicy.verdict(fpp: 0.014, nfpp: 0.0009, highCrowding: false) == .validated, "FPP validated")
    try expect(FPPPolicy.verdict(fpp: 0.014, nfpp: 0.0009, highCrowding: true) == .candidate, "FPP crowding downgrade")
    try expect(FPPPolicy.verdict(fpp: 0.014, nfpp: nil, highCrowding: false) == .candidate, "FPP needs NFPP")
}

func testInjectionPrimitivesParity() async throws {
    var random = NumPyMT19937(seed: 0)
    let expected = [
        0.5488135039273248, 0.7151893663724195, 0.6027633760716439,
        0.5448831829968969, 0.4236547993389047,
    ]
    for value in expected {
        try expect(abs(random.uniform01() - value) < 1e-16, "NumPy RandomState parity")
    }
    try expect(
        abs(TransitPhysics.durationDays(
            periodDays: 10, stellarRadiusSolar: 1,
            stellarMassSolar: nil, planetRadiusEarth: 11.2
        ) - 0.1798480076597091) < 1e-15,
        "transit duration parity"
    )
    try expect(
        abs(TransitInjector.depth(radiusEarth: 2, stellarRadiusSolar: 1)
            - 0.00033562239999999997) < 1e-18,
        "injection depth parity"
    )
    let time = [-0.1, 0, 0.05, 0.5, 0.95, 1.0]
    let injected = TransitInjector.inject(
        time: time, flux: Array(repeating: 1, count: time.count),
        period: 1, t0: 0, depth: 0.01, duration: 0.2
    )
    try expect(injected == [1, 0.99, 0.99, 1, 0.99, 0.99], "injection mask semantics")
}

func testInjectionRecoveryGoldenParity() async throws {
    struct Golden: Decodable {
        let time: [Double]
        let flux: [Double]
        let stellarRadius: Double
        let radiusGrid: [Double]
        let periodGrid: [Double]
        let injectionsPerCell: Int
        let periodCount: Int
        let binCount: Int
        let durationStepCount: Int
        let minimumSNR: Double
        let expected: Expected

        struct Expected: Decodable {
            let radiusGrid: [Double]
            let periodGrid: [Double]
            let recoveredCounts: [[Int]]
            let completenessMap: [[Double]]
            let truncated: Bool
            let periodsEvaluated: Int

            enum CodingKeys: String, CodingKey {
                case radiusGrid = "radius_grid", periodGrid = "period_grid"
                case recoveredCounts = "recovered_counts"
                case completenessMap = "completeness_map"
                case truncated, periodsEvaluated = "periods_evaluated"
            }
        }

        enum CodingKeys: String, CodingKey {
            case time, flux, stellarRadius = "stellar_radius"
            case radiusGrid = "radius_grid", periodGrid = "period_grid"
            case injectionsPerCell = "injections_per_cell"
            case periodCount = "bls_n_periods", binCount = "bls_n_bins"
            case durationStepCount = "bls_n_dur_steps", minimumSNR = "snr_threshold"
            case expected
        }
    }

    let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let golden = try JSONDecoder().decode(
        Golden.self,
        from: Data(contentsOf: root.appendingPathComponent("Fixtures/injection_recovery_golden.json"))
    )
    var options = InjectionRecoveryOptions()
    options.stellarRadiusSolar = golden.stellarRadius
    options.radiusGridEarth = golden.radiusGrid
    options.periodGridDays = golden.periodGrid
    options.injectionsPerCell = golden.injectionsPerCell
    options.periodCount = golden.periodCount
    options.binCount = golden.binCount
    options.durationStepCount = golden.durationStepCount
    options.minimumSNR = golden.minimumSNR
    let result = try InjectionRecovery.run(
        time: golden.time, flux: golden.flux, options: options, maxConcurrency: 1
    )
    let metalResult = try InjectionRecovery.run(
        time: golden.time, flux: golden.flux, options: options,
        maxConcurrency: 1, backend: .metal
    )
    guard let actualRadiusGrid = result.radiusGrid,
          let actualPeriodGrid = result.periodGrid,
          let actualCompleteness = result.completenessMap,
          let actualTruncated = result.truncated,
          let actualPeriodsEvaluated = result.periodsEvaluated else {
        throw TestFailure(description: "injection result omitted required fields")
    }
    try expect(actualRadiusGrid == golden.expected.radiusGrid, "injection radius grid")
    try expect(actualPeriodGrid == golden.expected.periodGrid, "injection period grid")
    try expect(actualTruncated == golden.expected.truncated, "injection truncation")
    try expect(actualPeriodsEvaluated == golden.expected.periodsEvaluated, "injection periods evaluated")
    try expect(metalResult.completenessMap == result.completenessMap, "Metal injection completeness must be decision-identical to CPU/Python")
    try expect(
        metalResult.details.map(\.recoveredCount) == result.details.map(\.recoveredCount),
        "Metal injection recovery counts must be decision-identical to CPU/Python"
    )
    for radiusIndex in golden.radiusGrid.indices {
        for periodIndex in golden.periodGrid.indices {
            let index = radiusIndex * golden.periodGrid.count + periodIndex
            guard let recoveredCount = result.details[index].recoveredCount else {
                throw TestFailure(description: "injection recovered count omitted")
            }
            try expect(
                recoveredCount == golden.expected.recoveredCounts[radiusIndex][periodIndex],
                "injection recovered count[\(radiusIndex),\(periodIndex)]"
            )
            try expect(
                abs(actualCompleteness[radiusIndex][periodIndex]
                    - golden.expected.completenessMap[radiusIndex][periodIndex]) < 1e-15,
                "injection completeness[\(radiusIndex),\(periodIndex)]"
            )
        }
    }
}

func testLongPeriodSweep() async throws {
    let time = (0...120).map(Double.init)
    let flux = [Double](repeating: 1, count: time.count)
    let outcome = try LongPeriodSweep.search(
        time: time, rawFlux: flux,
        accumulatedMask: [Bool](repeating: false, count: time.count),
        options: LongPeriodSweepOptions(),
        detrender: { _, _, window, _ in [Double](repeating: window, count: time.count) },
        search: { _, values, _ in
            let snr = values[0] == 4 ? 11.0 : 8.0
            return BLSResult(
                period: 40, power: snr, duration: 0.4, t0: 1,
                depth: 0.001, depthPPM: 1_000, snr: snr,
                periods: [], powerSpectrum: []
            )
        }
    )
    try expect(outcome?.windowDays == 4, "long sweep keeps best window")
    try expect(outcome?.result.snr == 11, "long sweep keeps highest SNR")
}

func testLongPeriodSweepPythonGoldenParity() async throws {
    struct Fixture: Decodable {
        struct Options: Decodable {
            let baselineThresholdDays: Double
            let windowsDays: [Double]
            let periodMinimumDays: Double
            let legacyPeriodMaximumDays: Double
            let periodMaximumCapDays: Double
            let minimumTransits: Double
            let basePeriodCount: Int
            let periodCountScaleReferenceMaximumDays: Double?
            let binCount: Int
            let durationStepCount: Int
            let durationMinimumFraction: Double
            let durationMaximumFraction: Double
            let minimumSNR: Double
        }
        struct Expected: Decodable {
            let windowDays: Double
            let period: Double
            let t0: Double
            let duration: Double
            let depth: Double
            let snr: Double
            let power: Double
        }
        let time: [Double]
        let flux: [Double?]
        let accumulatedMask: [Bool]
        let options: Options
        let expected: Expected
    }
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
    let fixture = try JSONDecoder().decode(
        Fixture.self,
        from: Data(contentsOf: root.appendingPathComponent("Fixtures/long_period_sweep_golden.json"))
    )
    var options = LongPeriodSweepOptions()
    options.baselineThresholdDays = fixture.options.baselineThresholdDays
    options.windowsDays = fixture.options.windowsDays
    options.periodMinimumDays = fixture.options.periodMinimumDays
    options.legacyPeriodMaximumDays = fixture.options.legacyPeriodMaximumDays
    options.periodMaximumCapDays = fixture.options.periodMaximumCapDays
    options.minimumTransits = fixture.options.minimumTransits
    options.basePeriodCount = fixture.options.basePeriodCount
    options.periodCountScaleReferenceMaximumDays = fixture.options.periodCountScaleReferenceMaximumDays
    options.binCount = fixture.options.binCount
    options.durationStepCount = fixture.options.durationStepCount
    options.durationMinimumFraction = fixture.options.durationMinimumFraction
    options.durationMaximumFraction = fixture.options.durationMaximumFraction
    options.minimumSNR = fixture.options.minimumSNR
    let flux = fixture.flux.map { $0 ?? .nan }
    try expect(fixture.time.last! - fixture.time.first! > 100, "golden baseline must exceed 100 days")
    final class PeriodBox: @unchecked Sendable { var value: Double? }
    let observedPeriodMaximum = PeriodBox()
    let outcome = try LongPeriodSweep.search(
        time: fixture.time,
        rawFlux: flux,
        accumulatedMask: fixture.accumulatedMask,
        options: options,
        search: { time, flux, parameters in
            observedPeriodMaximum.value = parameters.periodMaximum
            return try BLSReference.search(time: time, flux: flux, parameters: parameters)
        }
    )
    guard let outcome else { throw TestFailure(description: "long-period golden unexpectedly rejected") }
    let expectedPeriodMaximum = (fixture.time.last! - fixture.time.first!) / options.minimumTransits
    try expect(abs((observedPeriodMaximum.value ?? 0) - expectedPeriodMaximum) < 1e-12, "dynamic period limit")
    let expected = fixture.expected
    try expect(abs(outcome.windowDays - expected.windowDays) < 1e-12, "window parity")
    try expect(abs(outcome.result.period - expected.period) < 1e-10, "period parity \(outcome.result.period) != \(expected.period)")
    try expect(abs(outcome.result.t0 - expected.t0) < 1e-10, "t0 parity")
    try expect(abs(outcome.result.duration - expected.duration) < 1e-10, "duration parity")
    // Cross-language running-biweight accumulation differs by a few ulps; the
    // absolute bound is below 0.002% of this 0.28% transit depth.
    try expect(abs(outcome.result.depth - expected.depth) < 3e-8, "depth parity \(outcome.result.depth) != \(expected.depth)")
    // SNR divides by a small MAD, amplifying the same accumulation error.
    try expect(abs(outcome.result.snr - expected.snr) < 1e-3, "snr parity \(outcome.result.snr) != \(expected.snr)")
    // Power is depth squared, so it inherits the small cross-language drift.
    try expect(abs(outcome.result.power - expected.power) < 2e-9, "power parity \(outcome.result.power) != \(expected.power)")
}

func testTaskScientificOptions() async throws {
    let config: [String: JSONValue] = [
        "period_max": .number(50),
        "medium_period_protected_slot": .bool(false),
        "medium_slot_min_snr": .number(15),
        "medium_slot_ranges": .array([
            .array([.number(4), .number(9)]),
            .array([.number(12), .number(40)]),
        ]),
        "harmonic_long_period_recovery": .bool(true),
        "harmonic_long_period_min_days": .number(14),
        "harmonic_long_snr_factor": .number(0.65),
        "wotan_long_sweep": .bool(false),
    ]
    let options = PrimaryTaskProcessor.multiPlanetOptions(config: config)
    try expect(options.periodMaximum == 50, "main BLS maximum must not be clipped to span/2")
    try expect(!options.protectedSlotsEnabled, "protected-slot disable flag must propagate")
    try expect(options.protectedSlotMinimumSNR == 15, "protected-slot SNR must propagate")
    try expect(options.protectedSlotRanges == [4...9, 12...40], "protected-slot ranges must propagate")
    try expect(options.harmonicLongPeriodRecovery, "harmonic long-period gate must propagate")
    try expect(options.harmonicLongPeriodMinimumDays == 14, "harmonic long-period minimum must propagate")
    try expect(options.harmonicLongSNRFactor == 0.65, "harmonic long-period SNR factor must propagate")
    try expect(options.longPeriodSweep == nil, "long sweep disable flag must propagate")
    let processed = ProcessedLightCurve(
        time: [0, 1], flatFlux: [1, 0.99], rawFlux: [2, 3],
        sectorCount: 1, maxGapDays: 1
    )
    try expect(
        PrimaryTaskProcessor.injectionFlux(from: processed) == processed.flatFlux,
        "injection recovery must use the detrended flat flux"
    )
}

func testPrimaryFailureAndCancellation() async throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("exonit-primary-errors-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    MockURLProtocol.handler = { request in
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 500, httpVersion: nil, headerFields: nil
        )!
        return (response, Data("temporary failure".utf8))
    }
    let processor = try PrimaryTaskProcessor(
        api: WorkerAPIClient(
            baseURL: URL(string: "http://127.0.0.1:18400")!,
            token: "test-token", session: mockSession()
        ),
        cacheDirectory: directory
    )
    let task = WorkerTask(taskID: "task-failed", ticID: "123")
    let failed = try await processor.process(task, workerID: "wid")
    try expect(failed.status == "failed", "pipeline errors must use server-compatible failed status")

    let cancelled = Task { try await processor.process(task, workerID: "wid") }
    cancelled.cancel()
    do {
        _ = try await cancelled.value
        throw TestFailure(description: "cancelled primary task must not produce an outbox result")
    } catch is CancellationError {
        // Expected: the claimed task is left for server-side lease recovery.
    }
}

func testRuntimeCanaryAuthorization() async throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("exonit-canary-gate-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let requests = LockedCounter()
    MockURLProtocol.handler = { request in
        _ = requests.increment()
        throw TestFailure(description: "unauthorized runtime attempted network: \(request.url?.absoluteString ?? "?")")
    }
    let config = try WorkerConfig(
        serverURL: URL(string: "http://127.0.0.1:18400")!,
        token: "test-token", workerName: "swift-canary",
        pollIntervalSeconds: 0.01, heartbeatIntervalSeconds: 1,
        fitsCacheDirectory: directory.path, taskPools: ["primary"]
    ).validated()
    let runtime = try WorkerRuntime(
        config: config, outboxURL: directory.appendingPathComponent("outbox.sqlite"),
        canaryAuthorized: false, session: mockSession()
    )
    do {
        try await runtime.run()
        throw TestFailure(description: "runtime without canary authorization must fail closed")
    } catch let error as WorkerRuntimeError {
        guard case .canaryNotAuthorized = error else { throw error }
    }
    try expect(requests.current == 0, "canary gate must run before any network request")
}

func testScienceSidecarContract() async throws {
    let source = URL(fileURLWithPath: #filePath)
    let root = source.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
    process.arguments = [root.appendingPathComponent("Sidecars/exonit_science_sidecar.py").path, "--self-test"]
    let output = Pipe(); let errors = Pipe()
    process.standardOutput = output; process.standardError = errors
    try process.run(); process.waitUntilExit()
    let data = output.fileHandleForReading.readDataToEndOfFile()
    let stderr = errors.fileHandleForReading.readDataToEndOfFile()
    try expect(process.terminationStatus == 0, "sidecar self-test exit: \(String(decoding: stderr, as: UTF8.self))")
    let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    try expect(object?["protocol"] as? Int == 1, "sidecar protocol version")
    try expect(object?["status"] as? String == "ok", "sidecar self-test status")
}

func testScienceSidecarRunner() async throws {
    let source = URL(fileURLWithPath: #filePath)
    let root = source.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let runner = ScienceSidecarRunner(
        pythonURL: URL(fileURLWithPath: "/usr/bin/python3"),
        scriptURL: root.appendingPathComponent("Sidecars/exonit_science_sidecar.py")
    )
    try await runner.selfTest()
}

func testScienceSidecarPreflight() async throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("exonit-sidecar-preflight-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let source = URL(fileURLWithPath: #filePath)
        let root = source.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        try Data("def calc_fpp_native(): pass\n".utf8).write(
            to: directory.appendingPathComponent("fpp_native.py")
        )
        let package = directory.appendingPathComponent("exominer_native", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        try Data().write(to: package.appendingPathComponent("__init__.py"))
        try Data("def prepare_exominer_inputs(): pass\ndef score_prepared_exominer(): pass\n".utf8).write(
            to: package.appendingPathComponent("hook.py")
        )
        let model = directory.appendingPathComponent("model.onnx")
        try Data([1]).write(to: model)
        let runner = ScienceSidecarRunner(
            pythonURL: URL(fileURLWithPath: "/usr/bin/python3"),
            scriptURL: root.appendingPathComponent("Sidecars/exonit_science_sidecar.py")
        )
        try await runner.preflight(
            workerPythonDirectory: directory, requireFPP: true,
            requireExoMiner: true, exominerONNXURL: model
        )
}

func testScienceSidecarDeadline() async throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("exonit-sidecar-deadline-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let pidFile = directory.appendingPathComponent("pid")
    let script = directory.appendingPathComponent("sleep.py")
    try Data("import os,time,sys\nopen(sys.argv[-1], 'w').write(str(os.getpid()))\ntime.sleep(30)\n".utf8).write(to: script)
    let runner = ScienceSidecarRunner(
        pythonURL: URL(fileURLWithPath: "/usr/bin/python3"), scriptURL: script
    )
    do {
        try await runner.selfTest(
            extraArguments: [pidFile.path], timeoutSeconds: 0.2
        )
        throw TestFailure(description: "sidecar deadline should fail")
    } catch ScienceSidecarError.deadlineExceeded { }
    try await Task.sleep(for: .milliseconds(300))
    let pid = Int32((try String(contentsOf: pidFile)).trimmingCharacters(in: .whitespacesAndNewlines))!
    errno = 0
    try expect(kill(pid, 0) == -1 && errno == ESRCH, "timed-out sidecar child must be gone")
}

func testBackfillTransientFailurePropagates() async throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("exonit-backfill-error-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let api = WorkerAPIClient(baseURL: URL(string: "http://127.0.0.1:1")!, token: "test")
    let primary = try PrimaryTaskProcessor(api: api, cacheDirectory: directory)
    let configuration = ScienceSidecarConfiguration(
        pythonURL: URL(fileURLWithPath: "/usr/bin/python3"),
        scriptURL: directory.appendingPathComponent("missing-sidecar.py"),
        workerPythonDirectory: directory,
        exominerONNXURL: directory.appendingPathComponent("missing.onnx")
    )
    let processor = try BackfillTaskProcessor(
        primary: primary, configuration: configuration,
        serverURL: URL(string: "http://127.0.0.1:1")!, token: "test",
        workDirectory: directory.appendingPathComponent("work")
    )
    let task = try JSONDecoder().decode(BackfillTask.self, from: Data(#"{"candidate_id":1,"tic_id":"123","candidate_num":1,"period_days":2,"t0":1,"duration_days":0.1,"depth_ppm":1000,"snr":10,"n_sectors":1}"#.utf8))
    do {
        _ = try await processor.processExoMiner(task)
        throw TestFailure(description: "transient sidecar failure must propagate")
    } catch is ScienceSidecarError { }
}

func testAgentStartGuard() async throws {
    try withTemporaryDirectory { directory in
        let controller = AgentController(
            workerDirectory: directory,
            executableURL: URL(fileURLWithPath: "/usr/bin/true"),
            environment: [:]
        )
        let result = try controller.execute(AgentArguments(
            action: .start, workerName: "worker-test", allowStart: false
        ))
        let object = try decodedObject(result.output)
        try expect(result.exitCode == 2, "refused start exit code")
        try expect(object["status"] as? String == "refused", "refused status")
        try expect(
            !FileManager.default.fileExists(
                atPath: directory.appendingPathComponent(".agent.worker-test.pid").path
            ),
            "refused start must have no PID side effect"
        )
    }
}

func testAgentStoppedStatus() async throws {
    try withTemporaryDirectory { directory in
        let controller = AgentController(
            workerDirectory: directory,
            executableURL: URL(fileURLWithPath: "/usr/bin/true")
        )
        let result = try controller.execute(AgentArguments(
            action: .status, workerName: "worker-test"
        ))
        let object = try decodedObject(result.output)
        try expect(result.exitCode == 0, "status exit code")
        try expect(object["status"] as? String == "stopped", "stopped status")
        try expect(object["pid"] is NSNull, "stopped PID is null")
    }
}

func testAgentStalePIDIdentity() async throws {
    try withTemporaryDirectory { directory in
        let unrelated = Process()
        unrelated.executableURL = URL(fileURLWithPath: "/bin/sleep")
        unrelated.arguments = ["30"]
        try unrelated.run()
        defer {
            if unrelated.isRunning { unrelated.terminate(); unrelated.waitUntilExit() }
        }
        try Data("\(unrelated.processIdentifier)".utf8).write(
            to: directory.appendingPathComponent(".agent.worker-test.pid")
        )
        let controller = AgentController(
            workerDirectory: directory,
            executableURL: URL(fileURLWithPath: "/usr/bin/true"),
            processExecutableURL: { _ in URL(fileURLWithPath: "/bin/sleep") }
        )
        let result = try controller.execute(AgentArguments(action: .stop, workerName: "worker-test"))
        let object = try decodedObject(result.output)
        try expect(object["status"] as? String == "not_running", "foreign PID treated as stale")
        try expect(unrelated.isRunning, "agent must not signal a foreign process")
    }
}

func testAgentUnverifiablePID() async throws {
    try withTemporaryDirectory { directory in
        let pidURL = directory.appendingPathComponent(".agent.worker-test.pid")
        let nameURL = directory.appendingPathComponent(".agent.worker-test.worker_name")
        try Data("\(getpid())".utf8).write(to: pidURL)
        try Data("worker-test".utf8).write(to: nameURL)
        let controller = AgentController(
            workerDirectory: directory,
            executableURL: URL(fileURLWithPath: "/usr/bin/true"),
            processExecutableURL: { _ in nil },
            processCommandLine: { _ in nil }
        )
        let status = try controller.execute(AgentArguments(action: .status, workerName: "worker-test"))
        let statusObject = try decodedObject(status.output)
        try expect(status.exitCode == 1, "unverifiable status must fail closed")
        try expect(statusObject["status"] as? String == "unverifiable", "unverifiable status")
        try expect(FileManager.default.fileExists(atPath: pidURL.path), "status must preserve live unverifiable PID")

        let stop = try controller.execute(AgentArguments(action: .stop, workerName: "worker-test"))
        let stopObject = try decodedObject(stop.output)
        try expect(stop.exitCode == 1, "unverifiable stop must fail closed")
        try expect(stopObject["status"] as? String == "unverifiable", "unverifiable stop")
        try expect(FileManager.default.fileExists(atPath: pidURL.path), "stop must preserve live unverifiable PID")

        let start = try controller.execute(AgentArguments(
            action: .start, workerName: "worker-test", allowStart: true
        ))
        let startObject = try decodedObject(start.output)
        try expect(start.exitCode == 1, "unverifiable start must fail closed")
        try expect(startObject["status"] as? String == "unverifiable", "unverifiable start")
        try expect(FileManager.default.fileExists(atPath: pidURL.path), "start must preserve live unverifiable PID")
    }
}

func testAgentCommandLineFallbackOwned() async throws {
    // Production 2026-08-05: proc_pidpath returns 0 for live detached
    // children, so every stop/prune answered "unverifiable". With the
    // KERN_PROCARGS2 fallback the instance identity (spawn path +
    // --worker-name) verifies and stop() actually stops the process.
    try withTemporaryDirectory { directory in
        let child = Process()
        child.executableURL = URL(fileURLWithPath: "/bin/sleep")
        child.arguments = ["30"]
        try child.run()
        defer {
            if child.isRunning { child.terminate(); child.waitUntilExit() }
        }
        try Data("\(child.processIdentifier)".utf8).write(
            to: directory.appendingPathComponent(".agent.worker-test.pid")
        )
        let controller = AgentController(
            workerDirectory: directory,
            executableURL: URL(fileURLWithPath: "/bin/sleep"),
            processExecutableURL: { _ in nil },
            processCommandLine: { _ in ProcessCommandLineInfo(
                executablePath: "/bin/sleep",
                arguments: ["/bin/sleep", "--worker-name", "worker-test"]
            ) }
        )
        let result = try controller.execute(AgentArguments(action: .stop, workerName: "worker-test"))
        let object = try decodedObject(result.output)
        try expect(object["status"] as? String == "stopped", "fallback identity must allow stop: \(object)")
        child.waitUntilExit()
        try expect(!FileManager.default.fileExists(
            atPath: directory.appendingPathComponent(".agent.worker-test.pid").path
        ), "stop must clear PID state")
    }
}

func testAgentCommandLineFallbackForeignName() async throws {
    // Same binary but a DIFFERENT --worker-name is another instance:
    // foreign, must not be signalled.
    try withTemporaryDirectory { directory in
        let child = Process()
        child.executableURL = URL(fileURLWithPath: "/bin/sleep")
        child.arguments = ["30"]
        try child.run()
        defer {
            if child.isRunning { child.terminate(); child.waitUntilExit() }
        }
        try Data("\(child.processIdentifier)".utf8).write(
            to: directory.appendingPathComponent(".agent.worker-test.pid")
        )
        let controller = AgentController(
            workerDirectory: directory,
            // Expected binary lives elsewhere (release flip moved `current`):
            // path equality fails, so identity rests on name + argument.
            executableURL: URL(fileURLWithPath: "/usr/local/other/sleep"),
            processExecutableURL: { _ in nil },
            processCommandLine: { _ in ProcessCommandLineInfo(
                executablePath: "/bin/sleep",
                arguments: ["/bin/sleep", "--worker-name", "worker-other"]
            ) }
        )
        let result = try controller.execute(AgentArguments(action: .stop, workerName: "worker-test"))
        let object = try decodedObject(result.output)
        try expect(object["status"] as? String == "not_running", "foreign name treated as stale: \(object)")
        try expect(child.isRunning, "agent must not signal a foreign process")
    }
}

func testAgentConcurrentStart() async throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("exonit-agent-lock-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try Data("{}".utf8).write(to: directory.appendingPathComponent("config.json"))
    let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
    var environment = ProcessInfo.processInfo.environment
    environment["EXONIT_AGENT_TEST_CHILD"] = "1"
    let arguments = AgentArguments(action: .start, workerName: "worker-test", allowStart: true)
    let first = Task.detached {
        try AgentController(workerDirectory: directory, executableURL: executable, environment: environment)
            .execute(arguments)
    }
    let second = Task.detached {
        try AgentController(workerDirectory: directory, executableURL: executable, environment: environment)
            .execute(arguments)
    }
    let results = try await [first.value, second.value]
    let objects = try results.map { try decodedObject($0.output) }
    let statuses = objects.compactMap { $0["status"] as? String }.sorted()
    let pids = objects.compactMap { $0["pid"] as? Int }
    defer { for pid in Set(pids) { _ = Darwin.kill(pid_t(pid), SIGKILL) } }
    try expect(
        statuses == ["already_running", "started"] || statuses == ["started", "unverifiable"],
        "one concurrent start must win and the other must fail closed: \(statuses)"
    )
    try expect(Set(pids).count == 1, "concurrent starts must report the same PID")
}

func testAgentLogTail() async throws {
    try withTemporaryDirectory { directory in
        let log = directory.appendingPathComponent("worker-worker-test.log")
        try Data("one\ntwo\nthree\nfour\n".utf8).write(to: log)
        let controller = AgentController(
            workerDirectory: directory,
            executableURL: URL(fileURLWithPath: "/usr/bin/true")
        )
        let result = try controller.execute(AgentArguments(
            action: .logs, workerName: "worker-test", logLines: 2
        ))
        try expect(String(decoding: result.output, as: UTF8.self) == "three\nfour\n", "tail two lines")
    }
}

func testWorkerLogFile() async throws {
    try withTemporaryDirectory { directory in
        let resolved = WorkerLogFile.resolve("logs/worker.log", relativeTo: directory)
        try expect(
            resolved == directory.appendingPathComponent("logs/worker.log").standardizedFileURL,
            "relative log_file resolution"
        )
        try WorkerLogFile.append("first", to: resolved!)
        try WorkerLogFile.append("second", to: resolved!)
        let contents = String(decoding: try Data(contentsOf: resolved!), as: UTF8.self)
        try expect(contents == "first\nsecond\n", "log_file append sink")
        try expect(WorkerLogFile.resolve(nil, relativeTo: directory) == nil, "nil log_file disables sink")
    }
}

func testTGLCFITSGoldenParity() async throws {
    let path = "~/exonit-worker/fits_cache_3/hlsp_tglc_tess_ffi_gaiaid-1719071625276929792-s0026-cam4-ccd1_tess_v1_llc.fits"
    guard FileManager.default.fileExists(atPath: path) else {
        throw TestFailure(description: "local FITS fixture missing: \(path)")
    }
    let result = try FITSReader.readTGLCLightCurve(at: URL(fileURLWithPath: path))
    try expect(result.rawRowCount == 1148, "raw FITS rows")
    try expect(result.time.count == 1146, "quality-filtered rows")
    let expectedTime = [
        2010.279306796282, 2010.3001397270705, 2010.320972658557,
        2010.3418055902766, 2010.362638522229,
    ]
    let expectedFlux = [
        1.2003788948059082, 1.1788253784179688, 1.204421043395996,
        1.23312509059906, 1.1895798444747925,
    ]
    try expect(Array(result.time.prefix(5)) == expectedTime, "first TIME values")
    try expect(Array(result.flux.prefix(5)) == expectedFlux, "first calibrated flux values")
    try expect(result.time.last == 2035.112366239291, "last TIME")
    try expect(result.flux.last == 1.0206290483474731, "last flux")
    try expect(abs(result.time.reduce(0, +) - 2318011.2019264256) < 1e-8, "TIME sum")
    try expect(abs(result.flux.reduce(0, +) - 1149.5123415589333) < 1e-10, "flux sum")
}

func testQLPFITSGoldenParity() async throws {
    let path = "~/exonit-worker/fits_cache_7/hlsp_qlp_tess_ffi_s0039-0000000300868410_tess_v01_llc.fits"
    guard FileManager.default.fileExists(atPath: path) else {
        throw TestFailure(description: "local QLP fixture missing: \(path)")
    }
    let result = try FITSReader.readLightCurve(at: URL(fileURLWithPath: path))
    try expect(result.rawRowCount == 3879, "raw QLP rows")
    try expect(result.source == .qlp, "QLP source detection")
    try expect(result.cadence?.count == result.time.count, "QLP cadence preservation")
    try expect(result.time.count == 3865, "Lightkurve default quality mask rows")
    let expectedTime = [2361.7752914909674, 2361.782235867179, 2361.789180243389, 2361.7961246195996, 2361.803068995809]
    let expectedFlux = [0.9989174008369446, 1.0083009004592896, 1.0042166709899902, 1.0057843923568726, 1.002395749092102]
    try expect(Array(result.time.prefix(5)) == expectedTime, "QLP first TIME values")
    try expect(Array(result.flux.prefix(5)) == expectedFlux, "QLP first SAP_FLUX values")
    try expect(result.time.last == 2389.712385638018, "QLP last TIME")
    try expect(result.flux.last == 0.9936617612838745, "QLP last flux")
    let finite = zip(result.time, result.flux).filter { $0.0.isFinite && $0.1.isFinite }
    try expect(finite.count == 3864, "QLP finite count")
}

func testSPOCFITSGoldenParity() async throws {
    let path = "~/exonit-worker/fits_cache_6/tess2025099153000-s0091-0000000061098812-0288-s_lc.fits"
    guard FileManager.default.fileExists(atPath: path) else {
        throw TestFailure(description: "local SPOC fixture missing: \(path)")
    }
    let result = try FITSReader.readLightCurve(at: URL(fileURLWithPath: path))
    try expect(result.source == .spoc, "SPOC source detection")
    try expect(result.cadence?.count == result.time.count, "SPOC cadence preservation")
    try expect(result.time.count == 16011, "SPOC Lightkurve default quality rows")
    let expectedTime = [3775.297118871751, 3775.2985077781677, 3775.2998966845844, 3775.3012855910006, 3775.302674497417]
    let expectedFlux = [18010.396484375, 18047.078125, 18084.96484375, 18070.76953125, 18103.92578125]
    try expect(Array(result.time.prefix(5)) == expectedTime, "SPOC first TIME values")
    try expect(Array(result.flux.prefix(5)) == expectedFlux, "SPOC first PDCSAP_FLUX values: \(Array(result.flux.prefix(5)))")
    try expect(result.time.last == 3802.7508703550393, "SPOC last TIME")
    try expect(result.flux.last?.isNaN == true, "SPOC last flux NaN")
    let finite = zip(result.time, result.flux).filter { $0.0.isFinite && $0.1.isFinite }
    try expect(finite.count == 10219, "SPOC finite count")
}

private struct WotanGolden: Decodable {
    let time: [Double?]
    let flux: [Double?]
    let transitMask: [Bool]
    let plain: [Double?]
    let masked: [Double?]

    enum CodingKeys: String, CodingKey {
        case time, flux, plain, masked
        case transitMask = "transit_mask"
    }
}

func testWotanBiweightGoldenParity() async throws {
    let url = URL(fileURLWithPath: "~/exonit-swift-worker/Fixtures/wotan_biweight_golden.json")
    let golden = try JSONDecoder().decode(WotanGolden.self, from: Data(contentsOf: url))
    let time = golden.time.map { $0 ?? .nan }
    let flux = golden.flux.map { $0 ?? .nan }
    let plain = try BiweightDetrender.flatten(time: time, flux: flux)
    let masked = try BiweightDetrender.flatten(time: time, flux: flux, transitMask: golden.transitMask)
    for (label, actual, expected) in [("plain", plain, golden.plain), ("masked", masked, golden.masked)] {
        try expect(actual.count == expected.count, "\(label) length")
        for index in actual.indices {
            if let expectedValue = expected[index] {
                try expect(abs(actual[index] - expectedValue) < 5e-14, "\(label)[\(index)] parity")
            } else {
                try expect(actual[index].isNaN, "\(label)[\(index)] must be NaN")
            }
        }
    }
}

private struct IterativeGolden: Decodable {
    struct Quick: Decodable { let period: Double; let power: Double; let duration: Double; let t0: Double; let depth: Double; let snr: Double }
    let time: [Double?]; let flux: [Double?]; let final: [Double?]; let quick: Quick; let mask: [Bool]
}

func testIterativeDetrendGoldenParity() async throws {
    let url = URL(fileURLWithPath: "~/exonit-swift-worker/Fixtures/iterative_detrend_golden.json")
    let golden = try JSONDecoder().decode(IterativeGolden.self, from: Data(contentsOf: url))
    let time = golden.time.map { $0 ?? .nan }; let flux = golden.flux.map { $0 ?? .nan }
    let result = try IterativeDetrender.flatten(time: time, flux: flux)
    try expect(abs(result.quickBLS.period - golden.quick.period) < 1e-14, "iterative quick period")
    try expect(abs(result.quickBLS.snr - golden.quick.snr) < 1e-10, "iterative quick SNR")
    try expect(result.transitMask == golden.mask, "iterative transit mask")
    for index in result.flux.indices {
        let expected = golden.final[index] ?? .nan
        if expected.isNaN { try expect(result.flux[index].isNaN, "iterative NaN[\(index)]") }
        else { try expect(abs(result.flux[index] - expected) < 2e-12, "iterative flux[\(index)]") }
    }
}

private struct QLPPipelineGolden: Decodable {
    let paths: [String]
    let maxGap: Double
    let time: [Double]
    let raw: [Double]
    let flat: [Double]

    enum CodingKeys: String, CodingKey {
        case paths, time, raw, flat
        case maxGap = "max_gap"
    }
}

func testQLPPipelineGoldenParity() async throws {
    let url = URL(fileURLWithPath: "~/exonit-swift-worker/Fixtures/qlp_pipeline_golden.json")
    let golden = try JSONDecoder().decode(QLPPipelineGolden.self, from: Data(contentsOf: url))
    let curves = try golden.paths.map { try FITSReader.readLightCurve(at: URL(fileURLWithPath: $0)) }
    let result = try LightCurvePreprocessor.process(curves)
    try expect(result.time.count == golden.time.count, "pipeline count")
    try expect(abs(result.maxGapDays - golden.maxGap) < 1e-12, "pipeline max gap")
    for index in result.time.indices {
        try expect(abs(result.time[index] - golden.time[index]) < 1e-12, "pipeline time[\(index)]")
        try expect(abs(result.rawFlux[index] - golden.raw[index]) < 2e-13, "pipeline raw[\(index)] actual=\(result.rawFlux[index]) expected=\(golden.raw[index])")
        try expect(abs(result.flatFlux[index] - golden.flat[index]) < 2e-12, "pipeline flat[\(index)]")
    }
}

private struct TGLCPreprocessGolden: Decodable {
    let paths: [String]
    let count: Int
    let timeHead: [Double]
    let timeTail: [Double]
    let rawHead: [Double]
    let rawTail: [Double]
    let flatHead: [Double]
    let flatTail: [Double]
    let timeSum: Double
    let rawSum: Double
    let flatSum: Double
    let divergenceIndex: Int
    let divergenceTime: Double
    let divergenceRaw: Double
    let divergenceFlat: Double

    enum CodingKeys: String, CodingKey {
        case paths, count
        case timeHead = "time_head", timeTail = "time_tail"
        case rawHead = "raw_head", rawTail = "raw_tail"
        case flatHead = "flat_head", flatTail = "flat_tail"
        case timeSum = "time_sum", rawSum = "raw_sum", flatSum = "flat_sum"
        case divergenceIndex = "divergence_index", divergenceTime = "divergence_time"
        case divergenceRaw = "divergence_raw", divergenceFlat = "divergence_flat"
    }
}

/// Regression guard for the TGLC/FFI stitch-normalize dtype fix (divergence
/// report for TIC 10756829): worker.py's `_read_lightcurve`
/// upcasts `cal_aper_flux` to float64 BEFORE constructing the LightCurve, so
/// for TGLC sources `normalize()`'s division must run entirely in Double —
/// never truncating through Float32 like the SPOC/QLP path does. Runs the
/// real two-sector TGLC FITS files for TIC 10756829 (the TIC that exposed
/// the bug) through the actual `FITSReader` + `LightCurvePreprocessor`
/// pipeline and compares against a compacted golden derived from the
/// reference Python worker's `_read_lightcurve`/`_load_star` dump
/// (`.artifacts/parity-257/output/preprocess-python-10756829.json` on the
/// Mac; 1356 points, sector_count=2). `divergence_*` pins the exact row
/// (first array index, t=1683.6180987597115) whose old Float32-rounded
/// value (1.0102698802947998) differed from Python's Float64 value
/// (1.0102699310975947) by half a Float32 ULP — the fix must reproduce the
/// Float64 value bit-for-bit (well within the 1e-12 tolerance below).
func testTGLCPreprocessGoldenParity() async throws {
    let url = URL(fileURLWithPath: "~/exonit-swift-worker/Fixtures/tglc_10756829_preprocess_golden.json")
    let golden = try JSONDecoder().decode(TGLCPreprocessGolden.self, from: Data(contentsOf: url))
    for path in golden.paths {
        guard FileManager.default.fileExists(atPath: path) else {
            throw TestFailure(description: "local TGLC fixture missing: \(path)")
        }
    }
    let curves = try golden.paths.map { try FITSReader.readLightCurve(at: URL(fileURLWithPath: $0)) }
    try expect(curves.allSatisfy { $0.source == .tglc }, "fixture sectors must be TGLC")
    let result = try LightCurvePreprocessor.process(curves)
    try expect(result.time.count == golden.count, "TGLC preprocess count actual=\(result.time.count) expected=\(golden.count)")

    func compareEdges(_ actual: [Double], head: [Double], tail: [Double], label: String) throws {
        for index in head.indices {
            try expect(abs(actual[index] - head[index]) < 1e-12, "TGLC \(label) head[\(index)]")
        }
        for offset in tail.indices {
            let actualValue = actual[actual.count - tail.count + offset]
            try expect(abs(actualValue - tail[offset]) < 1e-12, "TGLC \(label) tail[\(offset)]")
        }
    }
    try compareEdges(result.time, head: golden.timeHead, tail: golden.timeTail, label: "time")
    try compareEdges(result.rawFlux, head: golden.rawHead, tail: golden.rawTail, label: "raw")
    try compareEdges(result.flatFlux, head: golden.flatHead, tail: golden.flatTail, label: "flat")

    try expect(abs(result.time.reduce(0, +) - golden.timeSum) < 1e-6, "TGLC time sum")
    try expect(abs(result.rawFlux.reduce(0, +) - golden.rawSum) < 1e-9, "TGLC raw sum")
    try expect(abs(result.flatFlux.reduce(0, +) - golden.flatSum) < 1e-8, "TGLC flat sum")

    try expect(abs(result.time[golden.divergenceIndex] - golden.divergenceTime) < 1e-12, "TGLC divergence row time")
    try expect(
        abs(result.rawFlux[golden.divergenceIndex] - golden.divergenceRaw) < 1e-12,
        "TGLC divergence row raw actual=\(result.rawFlux[golden.divergenceIndex]) expected=\(golden.divergenceRaw) "
        + "(pre-fix Float32 value was 1.0102698802947998 — a regression here means the TGLC dtype fix broke)"
    )
    try expect(
        abs(result.flatFlux[golden.divergenceIndex] - golden.divergenceFlat) < 1e-9,
        "TGLC divergence row flat actual=\(result.flatFlux[golden.divergenceIndex]) expected=\(golden.divergenceFlat)"
    )
}

/// Full 5-iteration multi-planet regression guard for TIC 10756829: real
/// TGLC FITS -> LightCurvePreprocessor -> MultiPlanetSearcher, matching the
/// discovery scorecard baseline (fixtures/baseline.jsonl candidates 1-5,
/// data/scorecards s0014 lot 2026-07-23). Before the TGLC dtype fix, the
/// ~6e-8 rounding noise introduced at stitch/normalize (see
/// testTGLCPreprocessGoldenParity) propagated non-linearly through 5 rounds
/// of detect->mask->re-detrend and flipped candidate #5 from period=0.9998d
/// to its 2:1 sub-harmonic alias at period=0.5d (both SNR~9.7) — this test
/// exercises exactly that 5-round iterative loop end to end and pins all 5
/// recovered periods/SNRs, not just the preprocessing arrays.
func testMultiPlanetFiveIterationGoldenParity() async throws {
    let paths = [
        "~/exonit-swift-canary/fits_cache/hlsp_tglc_tess_ffi_gaiaid-2080664784495597312-s0014-cam1-ccd1_tess_v1_llc.fits",
        "~/exonit-swift-canary/fits_cache/hlsp_tglc_tess_ffi_gaiaid-2080664784495597312-s0015-cam1-ccd2_tess_v1_llc.fits",
    ]
    for path in paths {
        guard FileManager.default.fileExists(atPath: path) else {
            throw TestFailure(description: "local TGLC fixture missing: \(path)")
        }
    }
    let curves = try paths.map { try FITSReader.readLightCurve(at: URL(fileURLWithPath: $0)) }
    let processed = try LightCurvePreprocessor.process(curves)
    let search = try MultiPlanetSearcher.search(
        time: processed.time, rawFlux: processed.rawFlux,
        initialDetrendedFlux: processed.flatFlux,
        stellarRadiusSolar: nil, stellarMassSolar: nil,
        source: .ffi, options: MultiPlanetOptions(),
        maxConcurrency: 4, blsBackend: .cpu
    )

    // Baseline candidates 1-5 for TIC 10756829 (baseline.jsonl), rounded the
    // same way production rounds (4 decimals period/t0/duration, 1 depth/SNR).
    struct Expected { let period: Double; let t0: Double; let duration: Double; let depth: Double; let snr: Double }
    let expected: [Expected] = [
        Expected(period: 1.9989, t0: 1684.2977, duration: 0.2399, depth: 8423.4, snr: 8.9),
        Expected(period: 3.9856, t0: 1687.3645, duration: 0.1993, depth: 7084.3, snr: 9.1),
        Expected(period: 0.9998, t0: 1684.4079, duration: 0.13, depth: 5925.9, snr: 14.5),
        Expected(period: 0.9998, t0: 1684.4679, duration: 0.07, depth: 7013.5, snr: 14.8),
        Expected(period: 0.9998, t0: 1684.5079, duration: 0.03, depth: 9045.1, snr: 9.7),
    ]
    try expect(
        search.candidates.count == expected.count,
        "multi-planet candidate count actual=\(search.candidates.count) expected=\(expected.count)"
    )
    for index in expected.indices {
        let actual = search.candidates[index]
        let want = expected[index]
        try expect(
            abs(actual.periodDays - want.period) < 1e-4,
            "candidate #\(index + 1) period actual=\(actual.periodDays) expected=\(want.period) "
            + "(pre-fix candidate #5 aliased to 0.5d)"
        )
        try expect(abs(actual.t0 - want.t0) < 1e-4, "candidate #\(index + 1) t0")
        try expect(abs(actual.durationDays - want.duration) < 1e-4, "candidate #\(index + 1) duration")
        try expect(abs(actual.depthPPM - want.depth) < 0.1, "candidate #\(index + 1) depth")
        try expect(abs(actual.snr - want.snr) < 0.1, "candidate #\(index + 1) SNR")
    }
}

func testStitchNormalizationSemantics() async throws {
    let time = (0..<100).map { Double($0) * 0.02 }
    let singleCurve = LightCurveData(time: time, flux: Array(repeating: 100, count: time.count), rawRowCount: time.count)
    let single = try LightCurvePreprocessor.process([singleCurve], iterativeDetrend: false)
    try expect(!single.rawFlux.isEmpty, "single segment has finite interior")
    try expect(single.rawFlux.allSatisfy { $0 == 100 }, "single segment must not be stitch-normalized")

    let secondTime = time.map { $0 + 10 }
    let secondCurve = LightCurveData(time: secondTime, flux: Array(repeating: 200, count: time.count), rawRowCount: time.count)
    let stitched = try LightCurvePreprocessor.process([singleCurve, secondCurve], iterativeDetrend: false)
    try expect(!stitched.rawFlux.isEmpty, "stitched curve has finite interior")
    try expect(stitched.rawFlux.allSatisfy { abs($0 - 1) < 1e-7 }, "multi-segment stitch normalizes each segment")
}

func testStitchOutOfOrderSectorsSorted() async throws {
    // lightkurve's LightCurveCollection.stitch() concatenates sectors in
    // whatever order the FITS files were loaded/downloaded, not necessarily
    // chronological order. When a later-loaded sector has earlier
    // timestamps than one loaded before it, naive concatenation produces a
    // large *negative* diff at the sector boundary while every within-
    // sector step stays small and positive — exactly the shape that made
    // the maxGap gate select fill_gaps (it only looks at positive diffs)
    // and then made GapFiller throw nonMonotonicTime at the boundary.
    // Regression for the multi-sector TICs that failed on the Swift worker
    // on 2026-09-18 (303432813, 10837041, 307809773, 407591297, 178155732,
    // 68577662, 361711730, 63189173) while the Python worker, which sorts
    // by time (worker.py's `np.argsort(time_arr)`), processed them fine.
    //
    // Cadence is populated here (unlike the TGLC/FFI case below) because
    // all 8 real failing TICs are SPOC 2-min, which always carries
    // cadenceno — LightCurvePreprocessor's pre-fill sort is gated on
    // `cadence != nil`, so this must exercise that branch to be a faithful
    // regression test.
    let laterSectorTime = (0..<50).map { 20.0 + Double($0) * 0.02 }
    let laterSector = LightCurveData(
        time: laterSectorTime, flux: Array(repeating: 100, count: laterSectorTime.count),
        rawRowCount: laterSectorTime.count, cadence: (0..<50).map { Int64(1000 + $0) }
    )
    let earlierSectorTime = (0..<50).map { Double($0) * 0.02 }
    let earlierSector = LightCurveData(
        time: earlierSectorTime, flux: Array(repeating: 100, count: earlierSectorTime.count),
        rawRowCount: earlierSectorTime.count, cadence: (0..<50).map { Int64($0) }
    )
    // Curves list intentionally out of chronological order, mirroring
    // stitch() receiving sectors in worker download order.
    let stitched = try LightCurvePreprocessor.process(
        [laterSector, earlierSector], iterativeDetrend: false
    )
    try expect(!stitched.time.isEmpty, "out-of-order stitch must not throw and must produce output")
    try expect(
        zip(stitched.time.dropFirst(), stitched.time).allSatisfy { $0.0 >= $0.1 },
        "output time must be non-decreasing after out-of-order stitch"
    )
    try expect(stitched.time.first! < stitched.time.last!, "sorted output spans the full time range")
}

func testStitchOutOfOrderNonCadenceNotReordered() async throws {
    // TGLC/FFI curves never carry cadenceno: FITSReader.swift's
    // readTGLCTable() builds LightCurveData without a `cadence:` argument,
    // matching worker.py's TGLC branch (worker.py:440-451), which builds
    // `lk.LightCurve(time=, flux=)` with no cadenceno set either, so
    // `hasattr(lc, "cadenceno")` is false and lightkurve's fill_gaps() takes
    // its OTHER branch (`while (t - prevtime) > 1.2 * dt`). That branch does
    // NOT pre-sort — it silently skips inserting a filler across a backward
    // jump and appends the point in its original position, relying on the
    // later post-detrend `np.argsort` to clean the order up; it never
    // raises. LightCurvePreprocessor.process deliberately does not apply
    // its pre-fill sort for this cadence-less branch (see the `cadence !=
    // nil` guard around the sort), so this same out-of-order shape reaches
    // GapFiller unsorted.
    //
    // GapFiller then throws nonMonotonicTime — Swift's GapFiller does not
    // yet mirror lightkurve's silent tolerance for an unsorted, cadence-
    // less input, only its sorted (cadence-present) reconstruction. That
    // residual gap is real but out of scope for this fix (no TGLC/FFI TIC
    // was reported broken; see progress/impl-nonmonotonic-time.md), and
    // this test exists to make the boundary explicit rather than silently
    // assume the cadence-present fix also covers this branch.
    let laterSectorTime = (0..<50).map { 20.0 + Double($0) * 0.02 }
    let laterSector = LightCurveData(
        time: laterSectorTime, flux: Array(repeating: 100, count: laterSectorTime.count),
        rawRowCount: laterSectorTime.count, source: .tglc
    )
    let earlierSectorTime = (0..<50).map { Double($0) * 0.02 }
    let earlierSector = LightCurveData(
        time: earlierSectorTime, flux: Array(repeating: 100, count: earlierSectorTime.count),
        rawRowCount: earlierSectorTime.count, source: .tglc
    )
    try expectThrows(
        "non-cadence out-of-order stitch is not pre-sorted and still throws nonMonotonicTime, matching pre-fix behavior"
    ) {
        _ = try LightCurvePreprocessor.process([laterSector, earlierSector], iterativeDetrend: false)
    }
}

private struct ReplayNoise: GaussianNoiseSource {
    var values: [Double]
    mutating func normal(mean: Double, standardDeviation: Double) -> Double {
        values.removeFirst()
    }
}

func testGapFillSemantics() async throws {
    var cadenceNoise = ReplayNoise(values: [999])
    let cadenceInput = GapFillSeries(
        time: [1, 2, 4, 5], flux: [100, 101, 103, 104], cadence: [10, 11, 13, 14]
    )
    let cadenceFilled = try GapFiller.fillIfAllowed(cadenceInput, noise: &cadenceNoise)
    try expect(cadenceFilled.time == [1, 2, 3, 4, 5], "cadence-aware times")
    try expect(cadenceFilled.cadence == [10, 11, 12, 13, 14], "cadence-aware indices")
    try expect(cadenceFilled.flux == [100, 101, 999, 103, 104], "replay filler placement")

    var noCadenceNoise = ReplayNoise(values: [777])
    let noCadence = try GapFiller.fillIfAllowed(
        GapFillSeries(time: [0, 1, 2, 4], flux: [1, 1, 1, 1], cadence: nil),
        noise: &noCadenceNoise
    )
    try expect(noCadence.time == [0, 1, 2, 3, 4], "time-grid gap fill")
    try expect(noCadence.flux == [1, 1, 1, 777, 1], "time-grid replay placement")

    var gatedNoise = ReplayNoise(values: [123])
    let gatedInput = GapFillSeries(time: [0, 1, 6], flux: [1, 1, 1], cadence: nil)
    let gated = try GapFiller.fillIfAllowed(gatedInput, noise: &gatedNoise)
    try expect(gated == gatedInput, "global gap exactly five days disables fill")
    try expect(gatedNoise.values == [123], "gate consumes no randomness")

    // lightkurve's fill_gaps() never rejects an exact-tie (zero) diff: its
    // non-cadence path (`while (t - prevtime) > 1.2 * dt`) treats a zero
    // diff as "no gap to fill" and just appends the duplicate point as-is,
    // no exception. GapFiller must mirror that tolerance for a duplicate
    // timestamp that can legitimately survive the pre-sort (e.g. two
    // sectors sharing a cadence's BJD at their boundary).
    var dupNoise = ReplayNoise(values: [])
    let dupInput = GapFillSeries(time: [0, 1, 1, 2], flux: [1, 2, 3, 4], cadence: nil)
    let dup = try GapFiller.fillIfAllowed(dupInput, noise: &dupNoise)
    try expect(dup.time == [0, 1, 1, 2], "duplicate timestamp is tolerated, matching lightkurve's fill_gaps")
    try expect(dup.flux == [1, 2, 3, 4], "duplicate timestamp inserts no filler, values pass through unchanged")
}

func testExtendedProtocolDTOs() async throws {
    let backfillJSON = Data(#"{"task":{"candidate_id":42,"tic_id":150430484,"candidate_num":1,"period_days":5.154,"t0":1234.5,"duration_days":0.15,"depth_ppm":900,"snr":14.2,"n_sectors":3}}"#.utf8)
    let envelope = try JSONDecoder().decode(BackfillTaskEnvelope.self, from: backfillJSON)
    try expect(envelope.task?.candidateID == 42, "backfill candidate")
    try expect(envelope.task?.ticID == "150430484", "flexible numeric TIC")

    let ffi = try JSONDecoder().decode(FFIResponse.self, from: Data(#"{"tic_id":12345,"sectors":[{"sector":14,"source":"tglc","time":[0,1],"flux":[1,0.99]}]}"#.utf8))
    try expect(ffi.ticID == "12345", "FFI TIC")
    try expect(ffi.sectors[0].effectiveQuality == [0, 0], "missing quality defaults to zero")

    let nearby = try JSONDecoder().decode(NearbyResponse.self, from: Data(#"{"tic_id":"12345","stars":[{"ID":12345,"ra":10,"dec":-20,"mass":null,"rad":1.1,"Teff":5777,"Tmag":10.5,"plx":3.2}],"gal_b":30.2,"source":"cache"}"#.utf8))
    try expect(nearby.stars[0].id == "12345", "nearby flexible ID")
    try expect(nearby.stars[0].mass == nil, "nearby nullable mass")

    let manifest = try JSONDecoder().decode(UpdateManifest.self, from: Data(#"{"files":[{"filename":"models/model.onnx","sha256":"abc"}]}"#.utf8))
    try expect(manifest.files[0].filename == "models/model.onnx", "update nested path")
}

func testExtendedAPIRoutes() async throws {
    let client = WorkerAPIClient(baseURL: URL(string: "http://127.0.0.1:18400")!, token: "test-token", session: mockSession())
    MockURLProtocol.handler = { request in
        try expect(request.value(forHTTPHeaderField: "X-Worker-ID") == "wid", "worker header")
        let path = request.url!.path
        let data: Data
        switch path {
        case "/api/worker/fpp_task":
            data = Data(#"{"task":null}"#.utf8)
        case "/api/worker/exominer_task":
            data = Data(#"{"task":{"candidate_id":7,"tic_id":"9","candidate_num":1,"period_days":2,"t0":1,"duration_days":0.1,"depth_ppm":500,"snr":12,"n_sectors":1}}"#.utf8)
        case "/api/worker/fpp_result", "/api/worker/exominer_result", "/api/worker/result":
            try expect(request.httpMethod == "POST", "result POST")
            data = Data(#"{"status":"ok"}"#.utf8)
        default:
            throw TestFailure(description: "unexpected extended route \(path)")
        }
        return (HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: ["Content-Type":"application/json"])!, data)
    }
    let fppTask = try await client.getFPPTask(workerID: "wid")
    let exoTask = try await client.getExoMinerTask(workerID: "wid")
    try expect(fppTask == nil, "empty FPP task")
    try expect(exoTask?.candidateID == 7, "ExoMiner task")
    _ = try await client.postFPPResult(FPPResultRequest(candidateID: 7, ticID: "9", fpp: 0.02, nfpp: 0.001, probabilityPlanet: 0.97, verdict: "planet_candidate"), workerID: "wid")
    _ = try await client.postExoMinerResult(ExoMinerResultRequest(candidateID: 7, ticID: "9", score: 0.93, tcePeriod: 2, sectorRun: "1-69"), workerID: "wid")
    _ = try await client.postPrimaryResult(PrimaryResultRequest(taskID: "t", ticID: "9", status: "failed", error: "load_failed"), workerID: "wid")
}

func testBinaryAPIRoutes() async throws {
    let client = WorkerAPIClient(baseURL: URL(string: "http://127.0.0.1:18400")!, token: "test-token", session: mockSession())
    MockURLProtocol.handler = { request in
        let path = request.url!.path
        switch path {
        case "/api/exominer/tfrecord/123":
            return (HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: ["Content-Type":"application/octet-stream", "X-Sector-Run":"1-69", "X-Tic-Id":"123"])!, Data([1,2,3]))
        case "/api/fits/test.fits":
            return (HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: ["Content-Type":"application/fits"])!, Data([4,5]))
        case "/api/worker/update":
            return (HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: ["Content-Type":"application/json"])!, Data(#"{"files":[]}"#.utf8))
        case "/api/worker/update/models/model.onnx":
            return (HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: ["Content-Type":"application/octet-stream"])!, Data([6,7]))
        default:
            throw TestFailure(description: "unexpected binary route \(path)")
        }
    }
    let tf = try await client.fetchTFRecord(ticID: "123")
    guard case let .ready(data, sectorRun, ticID) = tf else { throw TestFailure(description: "TFRecord ready") }
    try expect(data == Data([1,2,3]) && sectorRun == "1-69" && ticID == "123", "TFRecord payload")
    let fitsData = try await client.downloadFITS(filename: "test.fits")
    let manifest = try await client.fetchUpdateManifest()
    let updateData = try await client.downloadUpdateFile(filename: "models/model.onnx")
    try expect(fitsData == Data([4,5]), "FITS bytes")
    try expect(manifest.files.isEmpty, "update manifest")
    try expect(updateData == Data([6,7]), "update bytes")
}

func testDurableOutbox() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("exonit-outbox-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let database = directory.appendingPathComponent("outbox.sqlite")
    let outbox = try DurableOutbox(databaseURL: database)
    let payload = Data(#"{"task_id":"task-1","status":"completed"}"#.utf8)
    try await outbox.enqueue(idempotencyKey: "primary:task-1", endpoint: "/api/worker/result", workerID: "wid", payload: payload, now: Date(timeIntervalSince1970: 100))
    try await outbox.enqueue(idempotencyKey: "primary:task-1", endpoint: "/api/worker/result", workerID: "wid", payload: payload, now: Date(timeIntervalSince1970: 100))
    let first = try await outbox.pending(now: Date(timeIntervalSince1970: 100), limit: 10)
    try expect(first.count == 1 && first[0].attempts == 0, "outbox idempotent enqueue")
    try await outbox.markFailed(id: first[0].id, error: "timeout", now: Date(timeIntervalSince1970: 100))
    let immediate = try await outbox.pending(now: Date(timeIntervalSince1970: 100), limit: 10)
    try expect(immediate.isEmpty, "outbox backoff")
    let reopened = try DurableOutbox(databaseURL: database)
    let retry = try await reopened.pending(now: Date(timeIntervalSince1970: 103), limit: 10)
    try expect(retry.count == 1 && retry[0].attempts == 1, "outbox survives reopen")
    try await reopened.markDelivered(id: retry[0].id)
    let empty = try await reopened.pending(now: Date(timeIntervalSince1970: 200), limit: 10)
    try expect(empty.isEmpty, "outbox delivered deletion")
}

func testAtomicOutboxClaim() async throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("exonit-claim-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let database = directory.appendingPathComponent("outbox.sqlite")
    let first = try DurableOutbox(databaseURL: database)
    let second = try DurableOutbox(databaseURL: database)
    try await first.enqueue(
        idempotencyKey: "primary:claim", endpoint: "/api/worker/result",
        workerID: "w", payload: Data("{}".utf8), now: Date(timeIntervalSince1970: 10)
    )
    async let claimA = first.claimPending(owner: "a", now: Date(timeIntervalSince1970: 10), leaseDuration: 60)
    async let claimB = second.claimPending(owner: "b", now: Date(timeIntervalSince1970: 10), leaseDuration: 60)
    let claims = try await (claimA, claimB)
    try expect(claims.0.count + claims.1.count == 1, "entry atomically claimed once")
    let duringLease = try await first.pending(now: Date(timeIntervalSince1970: 11))
    try expect(duringLease.isEmpty, "leased entry hidden from pending")
    let afterLease = try await first.pending(now: Date(timeIntervalSince1970: 71))
    try expect(afterLease.count == 1, "expired lease becomes pending")
}

private final class LockedCounter: @unchecked Sendable {
    private let lock = NSLock(); private var value = 0
    func increment() -> Int { lock.lock(); defer { lock.unlock() }; value += 1; return value }
    var current: Int { lock.lock(); defer { lock.unlock() }; return value }
}

func testOutboxRetryDelivery() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("exonit-delivery-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let outbox = try DurableOutbox(databaseURL: directory.appendingPathComponent("outbox.sqlite"))
    let counter = LockedCounter()
    MockURLProtocol.handler = { request in
        try expect(request.url?.path == "/api/worker/result", "outbox result path")
        try expect(request.value(forHTTPHeaderField: "X-Worker-ID") == "wid", "outbox worker header")
        try expect(
            request.value(forHTTPHeaderField: "Idempotency-Key") == "primary:task-retry",
            "outbox idempotency header"
        )
        let call = counter.increment()
        let status = call == 1 ? 500 : 200
        let body = call == 1 ? Data(#"{"error":"temporary"}"#.utf8) : Data(#"{"status":"ok"}"#.utf8)
        return (HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: ["Content-Type":"application/json"])!, body)
    }
    let api = WorkerAPIClient(baseURL: URL(string: "http://127.0.0.1:18400")!, token: "token", session: mockSession())
    let sender = WorkerResultOutbox(outbox: outbox, api: api)
    let payload = PrimaryResultRequest(taskID: "task-retry", ticID: "123", status: "completed")
    try await sender.enqueuePrimary(payload, workerID: "wid", now: Date(timeIntervalSince1970: 100))
    let failed = await sender.drain(now: Date(timeIntervalSince1970: 100))
    try expect(failed.delivered == 0 && failed.failed == 1, "first outbox delivery fails")
    let retainedCount = try await outbox.count()
    try expect(retainedCount == 1, "failed result retained")
    let delivered = await sender.drain(now: Date(timeIntervalSince1970: 103))
    try expect(delivered.delivered == 1 && delivered.failed == 0, "outbox retry succeeds")
    let finalCount = try await outbox.count()
    try expect(finalCount == 0, "delivered result removed")
}

/// 2026-08-05 incident fix: a transport error mid-delivery (e.g. a timeout)
/// used to be marked 'unknown' and WorkerRuntime.run() killed the worker
/// outright ("ambiguous delivery, reconciliation required") — 14 production
/// instances died this way in one day, requiring manual reconciliation
/// against PG. The server is idempotent for outbox re-sends
/// (exonit/routes/workers.py: add_star upserts unconditionally,
/// db.add_candidate is "insert or update" preserving FPP/ExoMiner/verdicts,
/// complete_task is a plain UPDATE), so retrying is safe: a transport
/// failure must now behave exactly like an HTTP failure — same exponential
/// backoff, retried on the next drain, never fatal.
func testOutboxTransportAmbiguousRetriesAsPending() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
        "exonit-transport-ambiguous-\(UUID().uuidString)", isDirectory: true
    )
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let outbox = try DurableOutbox(databaseURL: directory.appendingPathComponent("outbox.sqlite"))
    let counter = LockedCounter()
    MockURLProtocol.handler = { request in
        let call = counter.increment()
        if call == 1 { throw URLError(.timedOut) }
        return (
            HTTPURLResponse(
                url: request.url!, statusCode: 200, httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!,
            Data(#"{"status":"ok"}"#.utf8)
        )
    }
    let api = WorkerAPIClient(
        baseURL: URL(string: "http://127.0.0.1:18400")!, token: "token", session: mockSession()
    )
    let sender = WorkerResultOutbox(outbox: outbox, api: api)
    try await sender.enqueuePrimary(
        PrimaryResultRequest(taskID: "task-ambiguous", ticID: "456", status: "completed"),
        workerID: "wid", now: Date(timeIntervalSince1970: 100)
    )
    let first = await sender.drain(now: Date(timeIntervalSince1970: 100))
    try expect(
        first.delivered == 0 && first.failed == 1 && first.deliveryUnknown == 0,
        "transport failure retries as pending with backoff, not delivery-unknown"
    )
    let unknownAfterTransportFailure = try await outbox.deliveryUnknownCount()
    try expect(unknownAfterTransportFailure == 0, "transport failure never persists as delivery-unknown")
    let retained = try await outbox.count()
    try expect(retained == 1, "ambiguous delivery retained for retry")
    let delivered = await sender.drain(now: Date(timeIntervalSince1970: 103))
    try expect(
        delivered.delivered == 1 && delivered.failed == 0 && delivered.deliveryUnknown == 0,
        "ambiguous delivery succeeds on the next drain after backoff"
    )
    let remaining = try await outbox.count()
    try expect(remaining == 0, "delivered result removed")
    try expect(counter.current == 2, "ambiguous delivery retried exactly once before success")
}

/// Point 2 of the 2026-08-05 fix: any `delivery_state = 'unknown'` row —
/// left behind by a pre-fix worker, or by the still-legible
/// .invalidResponse/.malformedResponse paths in
/// `WorkerResultOutbox.drain()` — must be reconciled to 'pending' the
/// moment the outbox database is (re)opened, immediately retryable, rather
/// than requiring the manual PG reconciliation the 08-05 incident needed.
func testOutboxLegacyUnknownReconciledOnOpen() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
        "exonit-legacy-unknown-\(UUID().uuidString)", isDirectory: true
    )
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let database = directory.appendingPathComponent("outbox.sqlite")
    let outbox = try DurableOutbox(databaseURL: database)
    try expect(outbox.reconciledUnknownCount == 0, "fresh outbox has nothing to reconcile")
    try await outbox.enqueue(
        idempotencyKey: "primary:legacy", endpoint: "/api/worker/result", workerID: "wid",
        payload: Data(#"{"task_id":"legacy","status":"completed"}"#.utf8), now: Date()
    )
    let entries = try await outbox.pending(now: Date())
    try await outbox.markDeliveryUnknown(id: entries[0].id, error: "pre-fix transport ambiguity (simulated)")
    let unknownBeforeReopen = try await outbox.deliveryUnknownCount()
    try expect(unknownBeforeReopen == 1, "legacy unknown entry persisted")

    // Simulate a worker restart against the same outbox file.
    let reopened = try DurableOutbox(databaseURL: database)
    try expect(reopened.reconciledUnknownCount == 1, "reopening reconciles the legacy unknown entry")
    let unknownAfterReopen = try await reopened.deliveryUnknownCount()
    try expect(unknownAfterReopen == 0, "no unknown entries remain after reconciliation")
    let pending = try await reopened.pending(now: Date())
    try expect(pending.count == 1 && pending[0].attempts == 0, "reconciled entry is immediately retryable as pending")
}

/// Runtime-level proof of the same fix: a worker whose outbox database
/// still carries a legacy 'unknown' row from before the 08-05 fix must
/// start clean — no `ambiguousDelivery` throw at startup or in the run
/// loop — and the reconciled entry gets delivered on its own on the next
/// drain, all without operator intervention.
func testRuntimeReconcilesLegacyUnknownAtStartup() async throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("exonit-runtime-reconcile-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let outboxURL = directory.appendingPathComponent("outbox.sqlite")

    do {
        let seed = try DurableOutbox(databaseURL: outboxURL)
        try await seed.enqueue(
            idempotencyKey: "primary:legacy-task", endpoint: "/api/worker/result", workerID: "wid",
            payload: Data(#"{"task_id":"legacy-task","status":"completed"}"#.utf8), now: Date()
        )
        let entries = try await seed.pending(now: Date())
        try await seed.markDeliveryUnknown(id: entries[0].id, error: "pre-fix transport ambiguity (simulated)")
        let seededUnknownCount = try await seed.deliveryUnknownCount()
        try expect(seededUnknownCount == 1, "seeded legacy unknown entry")
    }

    let requests = LockedCounter()
    let resultDelivered = LockedCounter()
    MockURLProtocol.handler = { request in
        _ = requests.increment()
        switch request.url?.path {
        case "/api/worker/register":
            return (
                HTTPURLResponse(
                    url: request.url!, statusCode: 200, httpVersion: nil,
                    headerFields: ["Content-Type": "application/json"]
                )!,
                Data(#"{"worker_id":"wid-reconcile"}"#.utf8)
            )
        case "/api/worker/task":
            return (
                HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!,
                Data(#"{"wait_seconds":1}"#.utf8)
            )
        case "/api/worker/result":
            _ = resultDelivered.increment()
            return (
                HTTPURLResponse(
                    url: request.url!, statusCode: 200, httpVersion: nil,
                    headerFields: ["Content-Type": "application/json"]
                )!,
                Data(#"{"status":"ok"}"#.utf8)
            )
        default:
            throw TestFailure(description: "unexpected request in reconciliation test: \(request.url?.absoluteString ?? "?")")
        }
    }
    let config = try WorkerConfig(
        serverURL: URL(string: "http://127.0.0.1:18400")!,
        token: "test-token", workerName: "swift-reconcile-test",
        pollIntervalSeconds: 0.01, heartbeatIntervalSeconds: 30,
        fitsCacheDirectory: directory.path, taskPools: ["primary"]
    ).validated()
    // DurableOutbox.init runs synchronously inside WorkerRuntime.init, so
    // the legacy 'unknown' row is already reconciled to 'pending' before
    // run() (or any network request) starts.
    let runtime = try WorkerRuntime(
        config: config, outboxURL: outboxURL,
        canaryAuthorized: true, session: mockSession()
    )
    let task = Task { try await runtime.run() }
    try await Task.sleep(for: .milliseconds(300))
    task.cancel()
    do {
        _ = try await task.value
    } catch is CancellationError {
        // Expected: run() only returns via cancellation/shutdown here.
    } catch let error as WorkerRuntimeError {
        throw TestFailure(description: "runtime must not fail closed on a reconciled legacy delivery: \(error)")
    }
    try expect(requests.current > 0, "runtime must have registered and started polling after reconciliation")
    try expect(resultDelivered.current == 1, "the reconciled legacy entry must be delivered on its own")
}

func testCLIHelp() async throws {
    let worker = CLIHelp.text(for: [])
    try expect(worker.contains("Usage: exonit-swift-worker"), "worker usage")
    try expect(worker.contains("--config"), "worker config help")
    let agent = CLIHelp.text(for: ["/tmp/agent.py"])
    try expect(agent.contains("agent.py start|stop|status|logs|version"), "agent actions")
    try expect(agent.contains("--allow-start"), "agent guard help")
}
