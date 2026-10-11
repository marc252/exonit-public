import Foundation

/// Native Swift FPP backfill path.  It deliberately reuses the primary
/// processor for FITS/FFI loading and keeps the API/sidecar boundary outside
/// the scientific calculation.
public final class NativeFPPTaskProcessor: @unchecked Sendable {
    private let primary: PrimaryTaskProcessor
    private let api: WorkerAPIClient
    private let cacheDirectory: URL

    public init(primary: PrimaryTaskProcessor, api: WorkerAPIClient, cacheDirectory: URL) {
        self.primary = primary
        self.api = api
        self.cacheDirectory = cacheDirectory
    }

    public func processFPP(
        _ task: BackfillTask, workerID: String
    ) async throws -> FPPResultRequest {
        do {
            let result = try await calculateFPP(task, workerID: workerID)
            try Task.checkCancellation()
            return result
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            if Task.isCancelled { throw CancellationError() }
            print(
                "[telemetry] swift-native fpp tic=\(task.ticID) status=failed "
                    + "error=\(String(describing: error))"
            )
            return Self.failedResult(task)
        }
    }

    private func calculateFPP(
        _ task: BackfillTask, workerID: String
    ) async throws -> FPPResultRequest {
        let loaded = try await primary.loadProcessedLightCurve(
            ticID: task.ticID, workerID: workerID
        )
        let nearby = try await api.fetchNearby(ticID: task.ticID)
        let nearbyRaw = Self.rawNearby(nearby)
        let trilegalURL = cacheDirectory.appendingPathComponent(
            "\(task.ticID)_TRILEGAL.csv"
        )
        let trilegal = try? String(contentsOf: trilegalURL, encoding: .utf8)
        let inputs = FPPInputsModel(
            ticID: task.ticID, periodDays: task.periodDays, t0: task.t0,
            durationDays: task.durationDays, depthPPM: task.depthPPM,
            time: loaded.processed.time, flux: loaded.processed.flatFlux,
            nearby: nearbyRaw, trilegalCSV: trilegal
        )
        var probes = FPPCalcProbes()
        let seed = FPPRNG.ticSeed(
            ticID: task.ticID, period: task.periodDays, t0: task.t0
        )
        guard let result = FPPCalculation.calcFPP(
            inputs, draws: 100_000, seed: seed, probes: &probes
        ) else {
            print("[telemetry] swift-native fpp tic=\(task.ticID) status=failed")
            return Self.failedResult(task)
        }
        let trilegalTag = result.trilegalUsed ? " trilegal=1" : " trilegal=0"
        print(
            "[telemetry] swift-native fpp tic=\(task.ticID) "
                + "fpp=\(result.fpp) nfpp=\(result.nfpp)"
                + trilegalTag
        )
        return FPPResultRequest(
            candidateID: task.candidateID, ticID: task.ticID,
            fpp: result.fpp, nfpp: result.nfpp,
            probabilityPlanet: nil,
            verdict: result.verdict.rawValue,
            candidate: Self.candidateObject(task)
        )
    }

    private static func jsonNumber(_ value: Double?) -> JSONValue {
        value.map(JSONValue.number) ?? .null
    }

    private static func rawNearby(_ nearby: NearbyResponse) -> [String: JSONValue] {
        let stars: [JSONValue] = nearby.stars.map { star in
            .object([
                "ID": .string(star.id), "ra": .number(star.ra),
                "dec": .number(star.dec), "mass": jsonNumber(star.mass),
                "rad": jsonNumber(star.radius), "Teff": jsonNumber(star.effectiveTemperature),
                "Tmag": jsonNumber(star.tessMagnitude), "plx": jsonNumber(star.parallax),
            ])
        }
        return [
            "stars": .array(stars), "gal_b": .number(nearby.galacticLatitude),
        ]
    }

    private static func candidateObject(_ task: BackfillTask) -> [String: JSONValue] {
        [
            "candidate_id": .integer(task.candidateID), "tic_id": .string(task.ticID),
            "candidate_num": .integer(Int64(task.candidateNumber)),
            "period_days": .number(task.periodDays), "t0": .number(task.t0),
            "duration_days": .number(task.durationDays),
            "depth_ppm": .number(task.depthPPM), "snr": .number(task.snr),
            "n_sectors": .integer(Int64(task.sectorCount)),
        ]
    }

    private static func failedResult(_ task: BackfillTask) -> FPPResultRequest {
        FPPResultRequest(
            candidateID: task.candidateID, ticID: task.ticID,
            fpp: -1, nfpp: -1, probabilityPlanet: nil,
            verdict: "failed", candidate: candidateObject(task)
        )
    }
}

public enum NativeFPPError: Error, CustomStringConvertible, Sendable {
    case calculationUnavailable(String)
    case missingAPI

    public var description: String {
        switch self {
        case let .calculationUnavailable(tic): return "native FPP calculation unavailable for TIC \(tic)"
        case .missingAPI: return "native FPP processor requires WorkerAPIClient"
        }
    }
}
