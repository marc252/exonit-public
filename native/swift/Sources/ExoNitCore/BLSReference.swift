import Foundation

public struct BLSParameters: Equatable, Sendable {
    public var periodMinimum: Double
    public var periodMaximum: Double
    public var periodCount: Int
    public var binCount: Int
    public var durationStepCount: Int
    public var durationMinimumFraction: Double
    public var durationMaximumFraction: Double

    public init(
        periodMinimum: Double = 0.5,
        periodMaximum: Double = 50,
        periodCount: Int = 10_000,
        binCount: Int = 200,
        durationStepCount: Int = 15,
        durationMinimumFraction: Double = 0.01,
        durationMaximumFraction: Double = 0.15
    ) {
        self.periodMinimum = periodMinimum
        self.periodMaximum = periodMaximum
        self.periodCount = periodCount
        self.binCount = binCount
        self.durationStepCount = durationStepCount
        self.durationMinimumFraction = durationMinimumFraction
        self.durationMaximumFraction = durationMaximumFraction
    }
}

public struct BLSResult: Equatable, Sendable {
    public let period: Double
    public let power: Double
    public let duration: Double
    public let t0: Double
    public let depth: Double
    public let depthPPM: Double
    public let snr: Double
    public let periods: [Double]
    public let powerSpectrum: [Double]

    public init(
        period: Double, power: Double, duration: Double, t0: Double,
        depth: Double, depthPPM: Double, snr: Double,
        periods: [Double], powerSpectrum: [Double]
    ) {
        self.period = period; self.power = power; self.duration = duration; self.t0 = t0
        self.depth = depth; self.depthPPM = depthPPM; self.snr = snr
        self.periods = periods; self.powerSpectrum = powerSpectrum
    }
}

public enum BLSError: Error, CustomStringConvertible, Sendable {
    case mismatchedLengths
    case insufficientSamples
    case invalidParameters(String)

    public var description: String {
        switch self {
        case .mismatchedLengths: return "time and flux lengths differ"
        case .insufficientSamples: return "BLS requires at least six samples"
        case let .invalidParameters(reason): return "invalid BLS parameters: \(reason)"
        }
    }
}

public enum BLSReference {
    public static func search(
        time: [Double], flux: [Double], parameters p: BLSParameters = BLSParameters()
    ) throws -> BLSResult {
        guard time.count == flux.count else { throw BLSError.mismatchedLengths }
        guard time.count >= 6 else { throw BLSError.insufficientSamples }
        guard p.periodMinimum > 0, p.periodMaximum > p.periodMinimum else {
            throw BLSError.invalidParameters("period range")
        }
        guard p.periodCount >= 2, p.binCount >= 3, p.durationStepCount >= 1 else {
            throw BLSError.invalidParameters("grid dimensions")
        }
        guard p.durationMinimumFraction > 0,
              p.durationMaximumFraction >= p.durationMinimumFraction else {
            throw BLSError.invalidParameters("duration range")
        }

        guard !durationBinGrid(parameters: p).isEmpty else {
            throw BLSError.invalidParameters("duration range produces no valid bins")
        }
        let sampleCount = time.count
        let t0Reference = time[0]
        let timeRelative = time.map { $0 - t0Reference }
        let periods = frequencyPeriodGrid(
            minimum: p.periodMinimum, maximum: p.periodMaximum, count: p.periodCount
        )
        let durationBins = durationBinGrid(parameters: p)
        let totalSum = flux.reduce(0, +)

        var bestPowers = [Double](repeating: 0, count: p.periodCount)
        var bestDurations = [Int](repeating: 0, count: p.periodCount)
        var bestPhases = [Int](repeating: 0, count: p.periodCount)
        var bestDepths = [Double](repeating: 0, count: p.periodCount)

        var binSums = [Double](repeating: 0, count: p.binCount)
        var binCounts = [Int](repeating: 0, count: p.binCount)
        var cumulativeSums = [Double](repeating: 0, count: 2 * p.binCount + 1)
        var cumulativeCounts = [Int](repeating: 0, count: 2 * p.binCount + 1)

        for periodIndex in periods.indices {
            let period = periods[periodIndex]
            binSums.withUnsafeMutableBufferPointer { $0.update(repeating: 0) }
            binCounts.withUnsafeMutableBufferPointer { $0.update(repeating: 0) }

            for index in 0..<sampleCount {
                let quotient = timeRelative[index] / period
                let phase = quotient - floor(quotient)
                let rawBin = Int(phase * Double(p.binCount))
                let bin = min(max(rawBin, 0), p.binCount - 1)
                binSums[bin] += flux[index]
                binCounts[bin] += 1
            }

            cumulativeSums[0] = 0
            cumulativeCounts[0] = 0
            for index in 0..<(2 * p.binCount) {
                let source = index % p.binCount
                cumulativeSums[index + 1] = cumulativeSums[index] + binSums[source]
                cumulativeCounts[index + 1] = cumulativeCounts[index] + binCounts[source]
            }

            var bestPower = -1e30
            var bestDuration = 0
            var bestPhase = 0
            var bestDepth = 0.0

            for width in durationBins {
                for phaseBin in 0..<p.binCount {
                    let windowCount = cumulativeCounts[phaseBin + width] - cumulativeCounts[phaseBin]
                    let outsideCount = sampleCount - windowCount
                    if windowCount < 3 || outsideCount < 3 { continue }
                    let windowSum = cumulativeSums[phaseBin + width] - cumulativeSums[phaseBin]
                    let meanInside = windowSum / Double(windowCount)
                    let meanOutside = (totalSum - windowSum) / Double(outsideCount)
                    let depth = meanOutside - meanInside
                    if depth <= 0 { continue }
                    let power = depth * depth * Double(windowCount * outsideCount)
                        / Double(windowCount + outsideCount)
                    if power > bestPower {
                        bestPower = power
                        bestDuration = width
                        bestPhase = phaseBin
                        bestDepth = depth
                    }
                }
            }
            bestPowers[periodIndex] = bestPower
            bestDurations[periodIndex] = bestDuration
            bestPhases[periodIndex] = bestPhase
            bestDepths[periodIndex] = bestDepth
        }

        var bestIndex = 0
        for index in 1..<bestPowers.count where bestPowers[index] > bestPowers[bestIndex] {
            bestIndex = index
        }
        let bestPeriod = periods[bestIndex]
        let depth = bestDepths[bestIndex]
        return BLSResult(
            period: bestPeriod,
            power: bestPowers[bestIndex],
            duration: Double(bestDurations[bestIndex]) / Double(p.binCount) * bestPeriod,
            t0: Double(bestPhases[bestIndex]) / Double(p.binCount) * bestPeriod + t0Reference,
            depth: depth,
            depthPPM: depth * 1_000_000,
            snr: sde(spectrum: bestPowers, peakIndex: bestIndex),
            periods: periods,
            powerSpectrum: bestPowers
        )
    }

    public static func frequencyPeriodGrid(minimum: Double, maximum: Double, count: Int) -> [Double] {
        let start = 1.0 / minimum
        let end = 1.0 / maximum
        let step = (end - start) / Double(count - 1)
        return (0..<count).map { 1.0 / (start + Double($0) * step) }
    }

    public static func sde(spectrum: [Double], peakIndex: Int) -> Double {
        let finite = spectrum.filter(\.isFinite)
        guard !finite.isEmpty else { return 0 }
        let med = median(finite)
        let mad = median(finite.map { abs($0 - med) })
        let sigma = 1.4826 * mad
        guard sigma > 0 else { return 0 }
        return (spectrum[peakIndex] - med) / sigma
    }

    public static func durationBinGrid(parameters p: BLSParameters) -> [Int] {
        let start = max(1, Int(p.durationMinimumFraction * Double(p.binCount)))
        let end = min(p.binCount / 3, Int(p.durationMaximumFraction * Double(p.binCount)))
        guard start <= end else { return [] }
        if p.durationStepCount == 1 { return [start] }
        let step = Double(end - start) / Double(p.durationStepCount - 1)
        var result: [Int] = []
        result.reserveCapacity(p.durationStepCount)
        for index in 0..<p.durationStepCount {
            let value = Int(Double(start) + Double(index) * step)
            if result.last != value { result.append(value) }
        }
        return result
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count % 2 == 1 { return sorted[middle] }
        return (sorted[middle - 1] + sorted[middle]) / 2
    }
}
