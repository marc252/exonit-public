import Foundation

public struct InjectionRecoveryOptions: Equatable, Sendable {
    public var stellarRadiusSolar: Double = 1
    public var stellarMassSolar: Double? = nil
    public var radiusGridEarth: [Double] = [0.5, 1, 1.5, 2, 3, 4, 6, 8, 10]
    public var periodGridDays: [Double]? = nil
    public var injectionsPerCell: Int = 10
    public var minimumSNR: Double = 7
    public var periodCount: Int = 5_000
    public var binCount: Int = 150
    public var durationStepCount: Int = 10
    public var maximumSeconds: Double? = nil
    public init() {}
}

public enum InjectionRecovery {
    public static func run(
        time: [Double], flux: [Double],
        options: InjectionRecoveryOptions = InjectionRecoveryOptions(),
        maxConcurrency: Int = ProcessInfo.processInfo.activeProcessorCount,
        backend: BLSComputeBackend = .cpu,
        isCancelled: @Sendable () -> Bool = { false }
    ) throws -> CompletenessResult {
        guard time.count == flux.count, time.count >= 6 else { throw BLSError.insufficientSamples }
        let span = time.last! - time.first!
        let periods = (options.periodGridDays ?? [0.5, 1, 2, 5, 10, 20, 30, 40])
            .filter { $0 < span / 2 }
        let radii = options.radiusGridEarth
        var counts = Array(
            repeating: [Int](repeating: 0, count: periods.count), count: radii.count
        )
        var evaluatedPeriods = 0
        var truncated = false
        let started = ContinuousClock.now
        for (periodIndex, period) in periods.enumerated() {
            if isCancelled() || exceeded(started: started, seconds: options.maximumSeconds) {
                truncated = true; break
            }
            let minimum = max(0.5, period * 0.5)
            let maximum = min(period * 2, span)
            let parameters = BLSParameters(
                periodMinimum: minimum, periodMaximum: maximum,
                periodCount: options.periodCount, binCount: options.binCount,
                durationStepCount: options.durationStepCount
            )
            var injectedFluxes: [[Double]] = []
            var locations: [(radius: Int, period: Double)] = []
            injectedFluxes.reserveCapacity(radii.count * options.injectionsPerCell)
            locations.reserveCapacity(radii.count * options.injectionsPerCell)
            for (radiusIndex, radius) in radii.enumerated() {
                let depth = TransitInjector.depth(
                    radiusEarth: radius, stellarRadiusSolar: options.stellarRadiusSolar
                )
                let duration = max(0.02, min(TransitPhysics.durationDays(
                    periodDays: period,
                    stellarRadiusSolar: options.stellarRadiusSolar,
                    stellarMassSolar: options.stellarMassSolar,
                    planetRadiusEarth: radius
                ), 1))
                for injection in 0..<options.injectionsPerCell {
                    let seed = radiusIndex * periods.count * options.injectionsPerCell
                        + periodIndex * options.injectionsPerCell + injection
                    var random = NumPyMT19937(seed: UInt32(truncatingIfNeeded: seed))
                    let t0 = time[0] + random.uniform01() * period
                    injectedFluxes.append(TransitInjector.inject(
                        time: time, flux: flux, period: period, t0: t0,
                        depth: depth, duration: duration
                    ))
                    locations.append((radiusIndex, period))
                }
            }
            let results: [BLSResult]
            if backend == .metal, !BLSMetal.isAvailable {
                throw BLSMetalError.unavailable
            }
            if backend != .cpu, BLSMetal.isAvailable {
                results = try BLSMetal.searchBatch(
                    time: time, fluxes: injectedFluxes, parameters: parameters
                )
            } else {
                results = try injectedFluxes.map { injected in
                    try BLSCPUOptimized.search(
                        time: time, flux: injected, parameters: parameters,
                        maxConcurrency: maxConcurrency
                    )
                }
            }
            for (location, result) in zip(locations, results) {
                if abs(result.period - location.period) / location.period < 0.05,
                   result.snr >= options.minimumSNR {
                    counts[location.radius][periodIndex] += 1
                }
            }
            evaluatedPeriods += 1
        }
        let denominator = Double(max(options.injectionsPerCell, 1))
        let completeness = counts.map { $0.map { Double($0) / denominator } }
        var details: [CompletenessCell] = []
        for radiusIndex in radii.indices {
            for periodIndex in periods.indices {
                details.append(CompletenessCell(
                    radiusEarth: radii[radiusIndex], periodDays: periods[periodIndex],
                    depthPPM: TransitInjector.depth(
                        radiusEarth: radii[radiusIndex],
                        stellarRadiusSolar: options.stellarRadiusSolar
                    ) * 1e6,
                    detectionProbability: completeness[radiusIndex][periodIndex],
                    recoveredCount: counts[radiusIndex][periodIndex],
                    injectedCount: options.injectionsPerCell
                ))
            }
        }
        return CompletenessResult(
            radiusGrid: radii, periodGrid: periods, completenessMap: completeness,
            details: details, stellarRadius: options.stellarRadiusSolar,
            truncated: truncated, periodsEvaluated: evaluatedPeriods,
            radiiEvaluated: radii.count
        )
    }

    private static func exceeded(
        started: ContinuousClock.Instant, seconds: Double?
    ) -> Bool {
        guard let seconds else { return false }
        return started.duration(to: .now) >= .seconds(seconds)
    }
}
