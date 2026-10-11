import ExoNitCore
import Foundation

@main
enum PreprocessDump {
    static func main() async throws {
        guard CommandLine.arguments.count == 6 else {
            FileHandle.standardError.write(Data("usage: exonit-preprocess-dump SERVER TOKEN TIC CACHE OUTPUT\n".utf8))
            exit(2)
        }
        let args = CommandLine.arguments
        guard let server = URL(string: args[1]) else { throw URLError(.badURL) }
        let api = WorkerAPIClient(baseURL: server, token: args[2])
        let processor = try PrimaryTaskProcessor(
            api: api,
            cacheDirectory: URL(fileURLWithPath: args[4]),
            maximumConcurrency: 7,
            blsBackend: .cpu,
            gpuEnabled: false
        )
        let loaded = try await processor.loadProcessedLightCurve(
            ticID: args[3], workerID: "offline-diagnostic"
        )
        let curve = loaded.processed
        let payload: [String: Any] = [
            "time": curve.time,
            "flat_flux": curve.flatFlux,
            "raw_flux": curve.rawFlux,
            "sector_count": curve.sectorCount,
            "max_gap_days": curve.maxGapDays,
        ]
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
        try data.write(to: URL(fileURLWithPath: args[5]), options: .atomic)
    }
}
