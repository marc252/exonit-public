import Dispatch
import Foundation

private struct BLSPeriodEvaluation: Sendable {
    let power: Double
    let durationBins: Int
    let phaseBin: Int
    let depth: Double
}

private final class BLSResultCollector: @unchecked Sendable {
    private var values: [BLSPeriodEvaluation?]
    private let lock = NSLock()

    init(count: Int) {
        values = [BLSPeriodEvaluation?](repeating: nil, count: count)
    }

    func merge(_ chunk: [(Int, BLSPeriodEvaluation)]) {
        lock.lock()
        defer { lock.unlock() }
        for (index, value) in chunk { values[index] = value }
    }

    func snapshot() -> [BLSPeriodEvaluation] {
        lock.lock()
        defer { lock.unlock() }
        return values.map { $0! }
    }
}

public enum BLSCPUOptimized {
    public static func search(
        time: [Double],
        flux: [Double],
        parameters p: BLSParameters = BLSParameters(),
        maxConcurrency: Int = ProcessInfo.processInfo.activeProcessorCount
    ) throws -> BLSResult {
        guard time.count == flux.count else { throw BLSError.mismatchedLengths }
        guard time.count >= 6 else { throw BLSError.insufficientSamples }
        guard p.periodMinimum > 0, p.periodMaximum > p.periodMinimum,
              p.periodCount >= 2, p.binCount >= 3, p.durationStepCount >= 1,
              p.durationMinimumFraction > 0,
              p.durationMaximumFraction >= p.durationMinimumFraction else {
            throw BLSError.invalidParameters("grid")
        }

        guard !BLSReference.durationBinGrid(parameters: p).isEmpty else {
            throw BLSError.invalidParameters("duration range produces no valid bins")
        }
        let t0Reference = time[0]
        let relativeTime = time.map { $0 - t0Reference }
        let periods = BLSReference.frequencyPeriodGrid(
            minimum: p.periodMinimum, maximum: p.periodMaximum, count: p.periodCount
        )
        let durationBins = BLSReference.durationBinGrid(parameters: p)
        let totalSum = flux.reduce(0, +)
        let workers = min(max(1, maxConcurrency), periods.count)
        let collector = BLSResultCollector(count: periods.count)

        DispatchQueue.concurrentPerform(iterations: workers) { workerIndex in
            var chunk: [(Int, BLSPeriodEvaluation)] = []
            chunk.reserveCapacity((periods.count + workers - 1) / workers)
            var binSums = [Double](repeating: 0, count: p.binCount)
            var binCounts = [Int](repeating: 0, count: p.binCount)
            var cumulativeSums = [Double](repeating: 0, count: 2 * p.binCount + 1)
            var cumulativeCounts = [Int](repeating: 0, count: 2 * p.binCount + 1)

            var periodIndex = workerIndex
            while periodIndex < periods.count {
                binSums.withUnsafeMutableBufferPointer { $0.update(repeating: 0) }
                binCounts.withUnsafeMutableBufferPointer { $0.update(repeating: 0) }
                let period = periods[periodIndex]

                for sampleIndex in relativeTime.indices {
                    let quotient = relativeTime[sampleIndex] / period
                    let phase = quotient - floor(quotient)
                    let rawBin = Int(phase * Double(p.binCount))
                    let bin = min(max(rawBin, 0), p.binCount - 1)
                    binSums[bin] += flux[sampleIndex]
                    binCounts[bin] += 1
                }

                cumulativeSums[0] = 0
                cumulativeCounts[0] = 0
                for index in 0..<(2 * p.binCount) {
                    let source = index % p.binCount
                    cumulativeSums[index + 1] = cumulativeSums[index] + binSums[source]
                    cumulativeCounts[index + 1] = cumulativeCounts[index] + binCounts[source]
                }

                var best = BLSPeriodEvaluation(
                    power: -1e30, durationBins: 0, phaseBin: 0, depth: 0
                )
                for width in durationBins {
                    for phaseBin in 0..<p.binCount {
                        let windowCount = cumulativeCounts[phaseBin + width]
                            - cumulativeCounts[phaseBin]
                        let outsideCount = relativeTime.count - windowCount
                        if windowCount < 3 || outsideCount < 3 { continue }
                        let windowSum = cumulativeSums[phaseBin + width]
                            - cumulativeSums[phaseBin]
                        let depth = (totalSum - windowSum) / Double(outsideCount)
                            - windowSum / Double(windowCount)
                        if depth <= 0 { continue }
                        let power = depth * depth * Double(windowCount * outsideCount)
                            / Double(windowCount + outsideCount)
                        if power > best.power {
                            best = BLSPeriodEvaluation(
                                power: power,
                                durationBins: width,
                                phaseBin: phaseBin,
                                depth: depth
                            )
                        }
                    }
                }
                chunk.append((periodIndex, best))
                periodIndex += workers
            }
            collector.merge(chunk)
        }

        let evaluations = collector.snapshot()
        let powers = evaluations.map(\.power)
        var bestIndex = 0
        for index in 1..<powers.count where powers[index] > powers[bestIndex] {
            bestIndex = index
        }
        let best = evaluations[bestIndex]
        let period = periods[bestIndex]
        return BLSResult(
            period: period,
            power: best.power,
            duration: Double(best.durationBins) / Double(p.binCount) * period,
            t0: Double(best.phaseBin) / Double(p.binCount) * period + t0Reference,
            depth: best.depth,
            depthPPM: best.depth * 1_000_000,
            snr: BLSReference.sde(spectrum: powers, peakIndex: bestIndex),
            periods: periods,
            powerSpectrum: powers
        )
    }
}
