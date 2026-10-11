import Foundation

public struct LongPeriodSweepOptions: Equatable, Sendable {
    public var baselineThresholdDays: Double = 100
    public var windowsDays: [Double] = [2, 4]
    public var periodMinimumDays: Double = 30
    public var legacyPeriodMaximumDays: Double = 50
    public var periodMaximumCapDays: Double = 200
    public var minimumTransits: Double = 2.5
    public var basePeriodCount: Int = 50_000
    public var periodCountScaleReferenceMaximumDays: Double? = 50
    public var binCount: Int = 64
    public var durationStepCount: Int = 7
    public var durationMinimumFraction: Double = 0.02
    public var durationMaximumFraction: Double = 0.16
    public var minimumSNR: Double = 7

    public init() {}
}

public struct LongPeriodSweepResult: Equatable, Sendable {
    public let result: BLSResult
    public let windowDays: Double
}

public typealias LongPeriodDetrender = (
    _ time: [Double], _ flux: [Double], _ windowDays: Double, _ mask: [Bool]?
) throws -> [Double]

public enum LongPeriodSweep {
    public static func search(
        time: [Double], rawFlux: [Double], accumulatedMask: [Bool],
        options: LongPeriodSweepOptions,
        detrender: LongPeriodDetrender = { time, flux, window, mask in
            try BiweightDetrender.flatten(
                time: time, flux: flux, windowLength: window, transitMask: mask
            )
        },
        search: BLSSearchFunction
    ) throws -> LongPeriodSweepResult? {
        guard time.count == rawFlux.count, time.count == accumulatedMask.count,
              let first = time.first, let last = time.last else {
            throw DetrendError.lengthMismatch
        }
        let span = last - first
        guard span > options.baselineThresholdDays else { return nil }
        let maximum = span > options.legacyPeriodMaximumDays * options.minimumTransits
            ? min(span / options.minimumTransits, options.periodMaximumCapDays)
            : options.legacyPeriodMaximumDays
        guard maximum > options.periodMinimumDays else { return nil }
        let count: Int
        if let reference = options.periodCountScaleReferenceMaximumDays {
            count = max(
                options.basePeriodCount,
                Int(Double(options.basePeriodCount) * maximum / reference)
            )
        } else { count = options.basePeriodCount }
        let parameters = BLSParameters(
            periodMinimum: options.periodMinimumDays,
            periodMaximum: maximum, periodCount: count,
            binCount: options.binCount,
            durationStepCount: options.durationStepCount,
            durationMinimumFraction: options.durationMinimumFraction,
            durationMaximumFraction: options.durationMaximumFraction
        )
        var best: LongPeriodSweepResult?
        for window in options.windowsDays {
            var flattened = try detrender(
                time, rawFlux, window,
                accumulatedMask.contains(true) ? accumulatedMask : nil
            )
            guard flattened.count == time.count else { throw DetrendError.lengthMismatch }
            let goodValues = flattened.indices.compactMap { index -> Double? in
                let value = flattened[index]
                return !accumulatedMask[index] && value.isFinite ? value : nil
            }
            let fill = median(goodValues)
            for index in flattened.indices
                where accumulatedMask[index] || !flattened[index].isFinite {
                flattened[index] = fill
            }
            let result = try search(time, flattened, parameters)
            if best == nil || result.snr > best!.result.snr {
                best = LongPeriodSweepResult(result: result, windowDays: window)
            }
        }
        guard let best, best.result.snr >= options.minimumSNR else { return nil }
        return best
    }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 1 }
        let sorted = values.sorted(); let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2)
            ? (sorted[middle - 1] + sorted[middle]) / 2
            : sorted[middle]
    }
}
