import Foundation

public struct IterativeDetrendResult: Equatable, Sendable {
    public let flux: [Double]
    public let firstPassFlux: [Double]
    public let quickBLS: BLSResult
    public let transitMask: [Bool]
    public let wasRefined: Bool
}

public enum IterativeDetrender {
    public static let quickParameters = BLSParameters(
        periodMinimum: 0.5, periodMaximum: 50, periodCount: 2_000,
        binCount: 100, durationStepCount: 5,
        durationMinimumFraction: 0.01, durationMaximumFraction: 0.15
    )

    public static func flatten(
        time: [Double], flux: [Double],
        snrThreshold: Double = 7,
        maxConcurrency: Int = ProcessInfo.processInfo.activeProcessorCount
    ) throws -> IterativeDetrendResult {
        guard time.count == flux.count else { throw DetrendError.lengthMismatch }
        let first = try BiweightDetrender.flatten(time: time, flux: flux)
        var finiteTime: [Double] = []
        var finiteFlux: [Double] = []
        for index in time.indices where time[index].isFinite && first[index].isFinite {
            finiteTime.append(time[index]); finiteFlux.append(first[index])
        }
        let quick = try BLSCPUOptimized.search(
            time: finiteTime, flux: finiteFlux, parameters: quickParameters,
            maxConcurrency: maxConcurrency
        )
        guard quick.snr >= snrThreshold else {
            return IterativeDetrendResult(
                flux: first, firstPassFlux: first, quickBLS: quick,
                transitMask: Array(repeating: false, count: time.count), wasRefined: false
            )
        }
        let durationFraction = quick.duration / quick.period
        let mask = time.map { value -> Bool in
            var phase = (value - quick.t0).truncatingRemainder(dividingBy: quick.period) / quick.period
            if phase < 0 { phase += 1 }
            return phase < durationFraction || phase > 1 - durationFraction
        }
        let refined = try BiweightDetrender.flatten(time: time, flux: flux, transitMask: mask)
        return IterativeDetrendResult(
            flux: refined, firstPassFlux: first, quickBLS: quick,
            transitMask: mask, wasRefined: true
        )
    }
}
