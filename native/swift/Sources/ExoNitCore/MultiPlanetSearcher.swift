import Foundation

public struct MultiPlanetOptions: Equatable, Sendable {
    public var maximumPlanets: Int = 5
    public var minimumSNR: Double = 7
    public var periodMinimum: Double = 0.5
    public var periodMaximum: Double = 50
    public var periodCount: Int = 50_000
    public var maximumMaskCoverage: Double = 0.30
    public var protectedSlotsEnabled: Bool = true
    public var protectedSlotMinimumSNR: Double = 12
    public var protectedSlotRanges: [ClosedRange<Double>] = [3...10, 10...50]
    public var harmonicLongPeriodRecovery: Bool = false
    public var harmonicLongPeriodMinimumDays: Double = 10
    public var harmonicLongSNRFactor: Double = 0.5
    public var longPeriodSweep: LongPeriodSweepOptions? = LongPeriodSweepOptions()

    public init() {}
}

public struct MultiPlanetSearchResult: Equatable, Sendable {
    public let candidates: [CandidateResult]
    public let accumulatedMask: [Bool]
    public let rejectedUnphysicalCount: Int
    public let redetrendFailureCount: Int
}

public enum MultiPlanetSearcher {
    public static func search(
        time: [Double], rawFlux: [Double], initialDetrendedFlux: [Double],
        stellarRadiusSolar: Double?, stellarMassSolar: Double?,
        source: LightCurveSource, options: MultiPlanetOptions = MultiPlanetOptions(),
        maxConcurrency: Int = ProcessInfo.processInfo.activeProcessorCount,
        blsBackend: BLSComputeBackend = .cpu
    ) throws -> MultiPlanetSearchResult {
        guard time.count == rawFlux.count, time.count == initialDetrendedFlux.count else {
            throw DetrendError.lengthMismatch
        }
        var candidates: [CandidateResult] = []
        var mask = [Bool](repeating: false, count: time.count)
        var current = initialDetrendedFlux
        var consecutiveDetrendFailures = 0
        var totalDetrendFailures = 0
        var rejected = 0
        let search: BLSSearchFunction = { sampleTime, sampleFlux, parameters in
            try BLSAccelerated.search(
                time: sampleTime, flux: sampleFlux, parameters: parameters,
                backend: blsBackend, maxConcurrency: maxConcurrency
            )
        }
        for iteration in 0..<max(options.maximumPlanets, 0) {
            if iteration >= 1, mask.contains(true) {
                let coverage = Double(mask.lazy.filter { $0 }.count) / Double(mask.count)
                if coverage <= options.maximumMaskCoverage {
                    do {
                        var flattened = try BiweightDetrender.flatten(
                            time: time, flux: rawFlux, transitMask: mask
                        )
                        let good = flattened.indices.filter { flattened[$0].isFinite && !mask[$0] }
                        let fill = good.isEmpty ? 1 : median(good.map { flattened[$0] })
                        for index in flattened.indices where !flattened[index].isFinite || mask[index] {
                            flattened[index] = fill
                        }
                        current = flattened
                        consecutiveDetrendFailures = 0
                    } catch {
                        consecutiveDetrendFailures += 1
                        totalDetrendFailures += 1
                        if consecutiveDetrendFailures >= 2 { break }
                    }
                }
            }
            let parameters = BLSParameters(
                periodMinimum: options.periodMinimum,
                periodMaximum: options.periodMaximum,
                periodCount: options.periodCount
            )
            let seed = try search(time, current, parameters)
            let result = HarmonicResolver.resolve(
                time: time, flux: current, seed: seed,
                periodMinimum: options.periodMinimum, periodMaximum: options.periodMaximum,
                options: HarmonicResolutionOptions(
                    source: source,
                    longPeriodRecovery: options.harmonicLongPeriodRecovery,
                    longPeriodMinimumDays: options.harmonicLongPeriodMinimumDays,
                    longSNRFactor: options.harmonicLongSNRFactor
                ), search: search
            )
            if result.snr < options.minimumSNR { break }
            let transitMask = makeMask(
                time: time, period: result.period, t0: result.t0, duration: result.duration
            )
            let physical = TransitPhysics.physicalT14Maximum(
                periodDays: result.period,
                stellarRadiusSolar: stellarRadiusSolar,
                stellarMassSolar: stellarMassSolar
            )
            if physical > 0, result.duration > 3 * physical {
                rejected += 1
                apply(transitMask, to: &mask, flux: &current)
                continue
            }
            candidates.append(candidate(
                number: iteration + 1, result: result,
                span: (time.last ?? 0) - (time.first ?? 0)
            ))
            apply(transitMask, to: &mask, flux: &current)
        }

        if options.protectedSlotsEnabled {
            let span = (time.last ?? 0) - (time.first ?? 0)
            for range in options.protectedSlotRanges {
                let upper = min(range.upperBound, options.periodMaximum)
                guard upper > range.lowerBound, span >= 2 * range.lowerBound else { continue }
                let result = try search(time, current, BLSParameters(
                    periodMinimum: range.lowerBound, periodMaximum: upper,
                    periodCount: options.periodCount
                ))
                let duplicate = candidates.contains {
                    abs(result.period - $0.periodDays) / max($0.periodDays, 1e-9) < 0.02
                }
                let physical = TransitPhysics.physicalT14Maximum(
                    periodDays: result.period,
                    stellarRadiusSolar: stellarRadiusSolar,
                    stellarMassSolar: stellarMassSolar
                )
                if result.snr >= options.protectedSlotMinimumSNR, !duplicate,
                   span / result.period >= 2,
                   !(physical > 0 && result.duration > 3 * physical) {
                    candidates.append(candidate(
                        number: candidates.count + 1, result: result, span: span
                    ))
                }
            }
        }
        if let longOptions = options.longPeriodSweep,
           let long = try? LongPeriodSweep.search(
                time: time, rawFlux: rawFlux, accumulatedMask: mask,
                options: longOptions, search: search
           ) {
            let physical = TransitPhysics.physicalT14Maximum(
                periodDays: long.result.period,
                stellarRadiusSolar: stellarRadiusSolar,
                stellarMassSolar: stellarMassSolar
            )
            if !(physical > 0 && long.result.duration > 3 * physical) {
                let span = (time.last ?? 0) - (time.first ?? 0)
                if candidates.count < options.maximumPlanets {
                    candidates.append(candidate(
                        number: candidates.count + 1, result: long.result, span: span
                    ))
                } else if let weakest = candidates.indices.min(by: {
                    candidates[$0].snr < candidates[$1].snr
                }), long.result.snr > candidates[weakest].snr {
                    candidates[weakest] = candidate(
                        number: candidates[weakest].candidateNumber,
                        result: long.result, span: span
                    )
                }
            }
        }
        return MultiPlanetSearchResult(
            candidates: candidates, accumulatedMask: mask,
            rejectedUnphysicalCount: rejected,
            redetrendFailureCount: totalDetrendFailures
        )
    }

    private static func makeMask(
        time: [Double], period: Double, t0: Double, duration: Double
    ) -> [Bool] {
        let half = duration / period / 2
        return time.map { value in
            var phase = (value - t0).truncatingRemainder(dividingBy: period) / period
            if phase < 0 { phase += 1 }
            return phase < half || phase > 1 - half
        }
    }

    private static func apply(_ added: [Bool], to mask: inout [Bool], flux: inout [Double]) {
        let fill = median(flux.filter(\.isFinite))
        for index in mask.indices where added[index] {
            mask[index] = true; flux[index] = fill
        }
    }

    private static func candidate(number: Int, result: BLSResult, span: Double) -> CandidateResult {
        CandidateResult(
            candidateNumber: number,
            periodDays: rounded(result.period, places: 4),
            t0: rounded(result.t0, places: 4),
            durationDays: rounded(result.duration, places: 4),
            depthPPM: rounded(result.depthPPM, places: 1),
            snr: rounded(result.snr, places: 1),
            blsPower: rounded(result.power, places: 4),
            expectedTransits: result.period > 0 ? Int(span / result.period) : 0
        )
    }

    private static func rounded(_ value: Double, places: Int) -> Double {
        let scale = pow(10, Double(places))
        return (value * scale).rounded(.toNearestOrEven) / scale
    }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 1 }
        let sorted = values.sorted(); let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2)
            ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
    }
}
