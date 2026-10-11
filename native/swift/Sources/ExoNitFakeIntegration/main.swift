import ExoNitCore
import Foundation

@main
struct FakeIntegration {
    static func main() async throws {
        guard CommandLine.arguments.count == 4, let baseURL = URL(string: CommandLine.arguments[1]) else {
            throw NSError(domain: "fake-integration", code: 2, userInfo: [NSLocalizedDescriptionKey:"usage: exonit-fake-integration URL CNN_MLPACKAGE EXOMINER_MLPACKAGE"])
        }
        let api = WorkerAPIClient(baseURL: baseURL, token: "fake-token")
        let workerID = try await api.register(RegisterRequest(
            name: "swift-integration",
            capabilities: WorkerCapabilities(gpu: "metal", blsBackend: "swift-cpu", blsGPUInfo: "Apple", fppBackend: "disabled", cpuCores: 1, fppCPUCores: 1)
        ))
        guard case let .task(task) = try await api.pollTask(workerID: workerID) else {
            throw NSError(domain: "fake-integration", code: 3, userInfo: [NSLocalizedDescriptionKey:"expected task"])
        }
        let command = try await api.heartbeat(HeartbeatRequest(workerID: workerID, status: "working", currentTIC: task.ticID, phase: "load", starsDone: 0, uptimeSeconds: 1, pid: Int32(ProcessInfo.processInfo.processIdentifier)))
        guard command == .continue else { throw NSError(domain: "fake-integration", code: 6) }

        let temp = FileManager.default.temporaryDirectory.appendingPathComponent("exonit-fake-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temp) }
        let processor = try PrimaryTaskProcessor(
            api: api, cacheDirectory: temp.appendingPathComponent("fits"),
            cnnModel: URL(fileURLWithPath: CommandLine.arguments[2]),
            exominerModel: URL(fileURLWithPath: CommandLine.arguments[3]),
            maximumConcurrency: 2
        )
        let result = try await processor.process(task, workerID: workerID)
        guard result.status == "completed" else {
            throw NSError(
                domain: "fake-integration", code: 5,
                userInfo: [NSLocalizedDescriptionKey: result.error ?? "primary processing failed"]
            )
        }
        let durable = try DurableOutbox(databaseURL: temp.appendingPathComponent("outbox.sqlite"))
        let sender = WorkerResultOutbox(outbox: durable, api: api)
        try await sender.enqueuePrimary(result, workerID: workerID)
        let drain = await sender.drain()
        guard drain.delivered == 1, try await durable.count() == 0 else {
            throw NSError(domain: "fake-integration", code: 7, userInfo: [NSLocalizedDescriptionKey:"outbox did not drain"])
        }
        guard case let .wait(seconds) = try await api.pollTask(workerID: workerID), seconds == 0.1 else {
            throw NSError(domain: "fake-integration", code: 8, userInfo: [NSLocalizedDescriptionKey:"expected wait"])
        }
        let output: [String: Any] = [
            "worker_id":workerID, "task_id":task.taskID,
            "candidates":result.candidates.count, "points":result.pointCount,
            "outbox_delivered":drain.delivered
        ]
        let data = try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys])
        print(String(decoding: data, as: UTF8.self))
    }
}
