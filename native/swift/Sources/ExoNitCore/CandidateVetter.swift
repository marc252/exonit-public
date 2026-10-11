import Foundation

public enum CandidateVettingError: Error, Equatable, Sendable {
    case mismatchedSeries
    case emptySeries
    case invalidCandidate(Int)
}

public enum CandidateVetter {
    private static let harmonicRatios = [0.5, 1.0 / 3.0, 2.0, 3.0, 2.0 / 3.0, 3.0 / 2.0]
    private static let harmonicTolerance = 0.02

    public static func vetAll(
        candidates: [CandidateResult], time: [Double], flux: [Double]
    ) throws -> [VettingResult] {
        guard time.count == flux.count else { throw CandidateVettingError.mismatchedSeries }
        guard !time.isEmpty else { throw CandidateVettingError.emptySeries }
        return try candidates.map { candidate in
            guard candidate.periodDays > 0, candidate.durationDays > 0 else {
                throw CandidateVettingError.invalidCandidate(candidate.candidateNumber)
            }
            let harmonic = checkHarmonics(candidate: candidate, all: candidates)
            let oddEven = checkOddEven(candidate: candidate, time: time, flux: flux)
            let vShape = checkVShape(candidate: candidate, time: time, flux: flux)
            return VettingResult(
                isHarmonic: harmonic.isHarmonic,
                harmonicOf: harmonic.parent,
                harmonicRatio: harmonic.ratio,
                oddEvenConsistent: oddEven.consistent,
                oddDepth: oddEven.oddDepth,
                evenDepth: oddEven.evenDepth,
                isVShaped: vShape.isVShaped,
                vMetric: vShape.metric,
                passed: !harmonic.isHarmonic && oddEven.consistent && !vShape.isVShaped
            )
        }
    }

    private static func checkHarmonics(
        candidate: CandidateResult, all: [CandidateResult]
    ) -> (isHarmonic: Bool, parent: Int?, ratio: Double?) {
        let parents = all.filter {
            $0.candidateNumber != candidate.candidateNumber && $0.snr >= candidate.snr
        }.sorted {
            if $0.snr != $1.snr { return $0.snr > $1.snr }
            return $0.candidateNumber < $1.candidateNumber
        }
        for parent in parents {
            if parent.snr == candidate.snr,
               parent.candidateNumber > candidate.candidateNumber { continue }
            for ratio in harmonicRatios {
                if abs(candidate.periodDays - parent.periodDays * ratio) / candidate.periodDays
                    < harmonicTolerance {
                    return (true, parent.candidateNumber, ratio)
                }
            }
        }
        return (false, nil, nil)
    }

    private static func checkOddEven(
        candidate: CandidateResult, time: [Double], flux: [Double]
    ) -> (consistent: Bool, oddDepth: Double?, evenDepth: Double?) {
        let count = Int((time[time.count - 1] - time[0]) / candidate.periodDays) + 1
        let halfDuration = candidate.durationDays / 2
        var odd: [Double] = []
        var even: [Double] = []
        if count > 0 {
            for index in 0..<count {
                let epoch = candidate.t0 + Double(index) * candidate.periodDays
                let transit = zip(time, flux).compactMap { sampleTime, sampleFlux in
                    abs(sampleTime - epoch) < halfDuration ? sampleFlux : nil
                }
                guard transit.count >= 3 else { continue }
                let depth = 1 - median(transit)
                if index.isMultiple(of: 2) { even.append(depth) }
                else { odd.append(depth) }
            }
        }
        guard odd.count >= 2, even.count >= 2 else { return (true, nil, nil) }
        let oddDepth = median(odd)
        let evenDepth = median(even)
        let maximum = max(oddDepth, evenDepth)
        let minimum = min(oddDepth, evenDepth)
        let consistent = (minimum > 0 && maximum / minimum < 3) || maximum < 1e-5
        return (consistent, rounded(oddDepth * 1e6, places: 1), rounded(evenDepth * 1e6, places: 1))
    }

    private static func checkVShape(
        candidate: CandidateResult, time: [Double], flux: [Double]
    ) -> (isVShaped: Bool, metric: Double) {
        let window = candidate.durationDays / candidate.periodDays * 2
        let samples = zip(time, flux).compactMap { sampleTime, sampleFlux -> (Double, Double)? in
            var remainder = (sampleTime - candidate.t0)
                .truncatingRemainder(dividingBy: candidate.periodDays)
            if remainder < 0 { remainder += candidate.periodDays }
            var phase = remainder / candidate.periodDays
            if phase > 0.5 { phase -= 1 }
            return abs(phase) < window ? (phase, sampleFlux) : nil
        }
        guard samples.count >= 10 else { return (false, 0) }
        let maximumPhase = samples.map { abs($0.0) }.max() ?? 0
        guard maximumPhase != 0 else { return (false, 0) }
        let center = samples.filter { abs($0.0) < maximumPhase * 0.3 }.map(\.1)
        let edge = samples.filter { abs($0.0) > maximumPhase * 0.6 }.map(\.1)
        guard center.count >= 3, edge.count >= 3 else { return (false, 0) }
        let centerDepth = 1 - median(center)
        guard centerDepth > 0 else { return (false, 0) }
        let metric = (1 - median(edge)) / centerDepth
        return (metric > 0.7, rounded(metric, places: 3))
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[middle - 1] + sorted[middle]) / 2
        }
        return sorted[middle]
    }

    private static func rounded(_ value: Double, places: Int) -> Double {
        let scale = pow(10, Double(places))
        return (value * scale).rounded(.toNearestOrEven) / scale
    }
}
