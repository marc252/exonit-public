import CoreML
import Foundation

public enum PrimaryTaskProcessorError: Error, CustomStringConvertible, Sendable {
    case noLightCurve(String)
    case missingRequiredModel(String)

    public var description: String {
        switch self {
        case let .noLightCurve(tic): return "no usable light curve for TIC \(tic)"
        case let .missingRequiredModel(path): return "required Core ML model missing: \(path)"
        }
    }
}

public struct LoadedStar: Sendable {
    public let metadata: StarMetadata
    public let processed: ProcessedLightCurve
    public let source: LightCurveSource
}

public final class PrimaryTaskProcessor: @unchecked Sendable {
    private let api: WorkerAPIClient
    private let cacheDirectory: URL
    private let cnn: CoreMLFFIClassifier?
    private let exominer: CoreMLExoMinerFFI?
    private let maximumConcurrency: Int
    private let blsBackend: BLSComputeBackend
    private let injectionBackend: BLSComputeBackend

    public init(
        api: WorkerAPIClient, cacheDirectory: URL,
        cnnModel: URL? = nil, exominerModel: URL? = nil,
        maximumConcurrency: Int = ProcessInfo.processInfo.activeProcessorCount,
        blsBackend: BLSComputeBackend = .cpu,
        injectionBackend: BLSComputeBackend? = nil,
        gpuEnabled: Bool = true
    ) throws {
        self.api = api; self.cacheDirectory = cacheDirectory
        self.maximumConcurrency = max(1, maximumConcurrency)
        self.blsBackend = blsBackend
        self.injectionBackend = injectionBackend ?? blsBackend
        try FileManager.default.createDirectory(
            at: cacheDirectory, withIntermediateDirectories: true
        )
        if let cnnModel {
            guard FileManager.default.fileExists(atPath: cnnModel.path) else {
                throw PrimaryTaskProcessorError.missingRequiredModel(cnnModel.path)
            }
            cnn = try CoreMLFFIClassifier(
                modelURL: cnnModel,
                computeUnits: gpuEnabled ? .all : .cpuOnly
            )
        } else { cnn = nil }
        if let exominerModel {
            guard FileManager.default.fileExists(atPath: exominerModel.path) else {
                throw PrimaryTaskProcessorError.missingRequiredModel(exominerModel.path)
            }
            exominer = try CoreMLExoMinerFFI(
                modelURL: exominerModel,
                calibration: PiecewiseLinearCalibration(
                    x: [0, 0.38930313157914914, 1], y: [0, 0.5, 1]
                ),
                computeUnits: gpuEnabled ? .all : .cpuOnly
            )
        } else { exominer = nil }
    }

    public static func multiPlanetOptions(
        config: [String: JSONValue]
    ) -> MultiPlanetOptions {
        var options = MultiPlanetOptions()
        options.maximumPlanets = config.int("max_planets", default: 5)
        options.minimumSNR = config.double("snr_threshold", default: 7)
        options.periodMinimum = config.double("period_min", default: 0.5)
        options.periodMaximum = config.double(
            "period_max", default: config.double("period_max_legacy", default: 50)
        )
        options.periodCount = config.int("n_periods", default: 50_000)
        options.protectedSlotsEnabled = config.bool(
            "medium_period_protected_slot", default: true
        )
        options.protectedSlotMinimumSNR = config.double(
            "medium_slot_min_snr", default: 12
        )
        if let configuredRanges = config.doubleRanges("medium_slot_ranges") {
            options.protectedSlotRanges = configuredRanges.compactMap {
                $0.count >= 2 && $0[1] > $0[0] ? $0[0]...$0[1] : nil
            }
        } else if config["medium_slot_min_days"] != nil ||
                    config["medium_slot_max_days"] != nil {
            let minimum = config.double("medium_slot_min_days", default: 10)
            let maximum = config.double("medium_slot_max_days", default: 50)
            options.protectedSlotRanges = maximum > minimum ? [minimum...maximum] : []
        }
        options.harmonicLongPeriodRecovery = config.bool(
            "harmonic_long_period_recovery", default: false
        )
        options.harmonicLongPeriodMinimumDays = config.double(
            "harmonic_long_period_min_days", default: 10
        )
        options.harmonicLongSNRFactor = config.double(
            "harmonic_long_snr_factor", default: 0.5
        )
        if config.bool("wotan_long_sweep", default: true) {
            var long = LongPeriodSweepOptions()
            long.baselineThresholdDays = config.double(
                "wotan_long_threshold_days", default: 100
            )
            long.windowsDays = config.doubles(
                "wotan_long_windows_days", default: [2, 4]
            )
            long.periodMinimumDays = config.double(
                "wotan_long_period_min_days", default: 30
            )
            long.legacyPeriodMaximumDays = options.periodMaximum
            long.periodMaximumCapDays = config.double("period_max_cap", default: 200)
            long.minimumTransits = config.double("period_max_min_transits", default: 2.5)
            long.basePeriodCount = config.int("wotan_long_n_periods", default: 50_000)
            if config["wotan_long_n_periods"] != nil {
                long.periodCountScaleReferenceMaximumDays = nil
            }
            long.minimumSNR = config.double(
                "wotan_long_snr_threshold", default: options.minimumSNR
            )
            options.longPeriodSweep = long
        } else { options.longPeriodSweep = nil }
        return options
    }

    public static func injectionFlux(from processed: ProcessedLightCurve) -> [Double] {
        processed.flatFlux
    }

    public func process(
        _ task: WorkerTask, workerID: String
    ) async throws -> PrimaryResultRequest {
        let started = ContinuousClock.now
        do {
            try Task.checkCancellation()
            let loaded = try await loadProcessedLightCurve(
                ticID: task.ticID, workerID: workerID,
                iterativeDetrend: task.config.bool("iterative_detrend", default: true)
            )
            let metadata = loaded.metadata
            let processed = loaded.processed
            let options = Self.multiPlanetOptions(config: task.config)
            let source = loaded.source
            let search = try MultiPlanetSearcher.search(
                time: processed.time, rawFlux: processed.rawFlux,
                initialDetrendedFlux: processed.flatFlux,
                stellarRadiusSolar: metadata.stellarRadius,
                stellarMassSolar: metadata.stellarMass,
                source: source, options: options, maxConcurrency: maximumConcurrency,
                blsBackend: blsBackend
            )
            var candidates = search.candidates
            let vetting = try CandidateVetter.vetAll(
                candidates: candidates, time: processed.time, flux: processed.rawFlux
            )
            var cnnScores = [Double?](repeating: nil, count: candidates.count)
            for index in candidates.indices {
                let candidate = candidates[index]
                if let cnn {
                    let views = try FFIViewGenerator.makeCNNViews(
                        time: processed.time, flux: processed.flatFlux,
                        period: candidate.periodDays, t0: candidate.t0,
                        duration: candidate.durationDays
                    )
                    cnnScores[index] = try cnn.classify(views: views).probability
                }
                if let exominer {
                    let features = try ExoMinerFFIFeatureBuilder.make(
                        time: processed.time, flux: processed.flatFlux,
                        period: candidate.periodDays, t0: candidate.t0,
                        duration: candidate.durationDays, depthPPM: candidate.depthPPM,
                        snr: candidate.snr,
                        stellar: FFIStellarMetadata(
                            tmag: metadata.tmag,
                            teff: metadata.effectiveTemperature,
                            logg: metadata.logg,
                            radius: metadata.stellarRadius,
                            mass: metadata.stellarMass
                        )
                    )
                    let score = try exominer.classify(features: features)
                    candidates[index].exominerFFIScore = score.probability
                    candidates[index].exominerFFIModelVersion = "phase3_v6_ensemble_cal"
                }
            }
            var completeness: CompletenessResult?
            let runInjection = task.config.bool(
                "run_injection",
                default: task.config.bool("run_injection_recovery", default: true)
            )
            if runInjection {
                var injection = InjectionRecoveryOptions()
                injection.stellarRadiusSolar = metadata.stellarRadius ?? 1
                injection.stellarMassSolar = metadata.stellarMass
                injection.injectionsPerCell = task.config.int("n_injections_per_cell", default: 10)
                injection.minimumSNR = options.minimumSNR
                injection.periodCount = task.config.int("injection_n_periods", default: 5_000)
                injection.binCount = task.config.int("injection_n_bins", default: 150)
                injection.durationStepCount = task.config.int("injection_n_dur_steps", default: 10)
                injection.maximumSeconds = task.config.double("injection_max_seconds", default: 300)
                completeness = try InjectionRecovery.run(
                    time: processed.time, flux: Self.injectionFlux(from: processed),
                    options: injection, maxConcurrency: maximumConcurrency,
                    backend: injectionBackend
                )
            }
            return PrimaryResultRequest(
                taskID: task.taskID, ticID: task.ticID,
                candidates: candidates, vettingResults: vetting,
                cnnScores: cnnScores, completeness: completeness,
                processingTimeSeconds: elapsed(started),
                gpuUsed: BLSMetal.isAvailable && (
                    blsBackend != .cpu || (runInjection && injectionBackend != .cpu)
                ),
                sectorCount: processed.sectorCount, pointCount: processed.time.count,
                timeSpanDays: (processed.time.last ?? 0) - (processed.time.first ?? 0),
                ra: metadata.ra, dec: metadata.dec, tmag: metadata.tmag,
                timings: [
                    "swift_native": .bool(true),
                    "redetrend_failures": .integer(Int64(search.redetrendFailureCount)),
                    "unphysical_rejections": .integer(Int64(search.rejectedUnphysicalCount)),
                ], lightCurveLoadRecovered: false
            )
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            return PrimaryResultRequest(
                taskID: task.taskID, ticID: task.ticID, status: "failed",
                error: String(describing: error), processingTimeSeconds: elapsed(started)
            )
        }
    }

    public func loadProcessedLightCurve(
        ticID: String, workerID: String, iterativeDetrend: Bool = true
    ) async throws -> LoadedStar {
        let metadata = try await api.fetchMetadata(ticID: ticID, workerID: workerID)
        let curves = try await loadCurves(ticID: ticID, metadata: metadata)
        guard !curves.isEmpty else { throw PrimaryTaskProcessorError.noLightCurve(ticID) }
        let processed = try LightCurvePreprocessor.process(
            curves, iterativeDetrend: iterativeDetrend,
            maxConcurrency: maximumConcurrency
        )
        return LoadedStar(
            metadata: metadata, processed: processed,
            source: curves.contains { $0.source == .spoc } ? .spoc : .ffi
        )
    }

    private func loadCurves(ticID: String, metadata: StarMetadata) async throws -> [LightCurveData] {
        if !metadata.fitsFiles.isEmpty {
            var result: [LightCurveData] = []
            for filename in metadata.fitsFiles {
                let destination = cacheDirectory.appendingPathComponent(filename)
                if !FileManager.default.fileExists(atPath: destination.path) {
                    let data = try await api.downloadFITS(filename: filename)
                    try data.write(to: destination, options: [.atomic])
                }
                result.append(try FITSReader.readLightCurve(at: destination))
            }
            return result
        }
        guard let ffi = try await api.fetchFFI(ticID: ticID) else { return [] }
        return ffi.sectors.map { sector in
            LightCurveData(
                time: sector.time, flux: sector.flux, rawRowCount: sector.time.count,
                source: sector.source.lowercased().contains("tglc") ? .tglc : .qlp
            )
        }
    }

    private func elapsed(_ start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: .now)
        return Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
    }
}

private extension Dictionary where Key == String, Value == JSONValue {
    func bool(_ key: String, default fallback: Bool) -> Bool {
        if case let .bool(value)? = self[key] { return value }; return fallback
    }
    func double(_ key: String, default fallback: Double) -> Double {
        switch self[key] {
        case let .number(value): return value
        case let .integer(value): return Double(value)
        default: return fallback
        }
    }
    func int(_ key: String, default fallback: Int) -> Int {
        switch self[key] {
        case let .integer(value): return Int(clamping: value)
        case let .number(value): return Int(value)
        default: return fallback
        }
    }
    func doubleRanges(_ key: String) -> [[Double]]? {
        guard case let .array(rows)? = self[key] else { return nil }
        return rows.compactMap { row in
            guard case let .array(values) = row else { return nil }
            return values.compactMap {
                switch $0 {
                case let .number(value): return value
                case let .integer(value): return Double(value)
                default: return nil
                }
            }
        }
    }

    func doubles(_ key: String, default fallback: [Double]) -> [Double] {
        guard case let .array(values)? = self[key] else { return fallback }
        let result = values.compactMap { value -> Double? in
            switch value {
            case let .number(number): return number
            case let .integer(integer): return Double(integer)
            default: return nil
            }
        }
        return result.isEmpty ? fallback : result
    }
}
