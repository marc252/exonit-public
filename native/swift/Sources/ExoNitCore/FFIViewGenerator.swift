import Foundation

public enum FFIViewError: Error, Equatable, Sendable {
    case mismatchedSeries
    case invalidEphemeris
    case sparseLightCurve
    case allBinsEmpty
}

public struct CNNFFIViews: Equatable, Sendable {
    public let global: [Float]
    public let local: [Float]

    public init(global: [Float], local: [Float]) {
        self.global = global
        self.local = local
    }
}

public enum FFIViewGenerator {
    public static let globalBinCount = 201
    public static let localBinCount = 61

    public static func makeCNNViews(
        time: [Double], flux: [Double], period: Double, t0: Double, duration: Double
    ) throws -> CNNFFIViews {
        guard time.count == flux.count else { throw FFIViewError.mismatchedSeries }
        guard period > 0, duration > 0 else { throw FFIViewError.invalidEphemeris }
        let folded = phaseFoldSort(
            time: time, flux: flux, period: period, t0: t0, duration: duration
        )
        let global = try normalize(medianFilter(
            x: folded.phase, y: folded.flux, count: globalBinCount,
            width: period * (1.2 / Double(globalBinCount)),
            minimum: -period / 2, maximum: period / 2
        ))
        let halfWindow = min(period / 2, 2 * duration)
        let local = try normalize(medianFilter(
            x: folded.phase, y: folded.flux, count: localBinCount,
            width: duration * 0.16, minimum: -halfWindow, maximum: halfWindow
        ))
        return CNNFFIViews(global: global, local: local)
    }

    public static func makeAuxiliaryViews(
        time: [Double], flux: [Double], period: Double, t0: Double, duration: Double
    ) -> [[Float]] {
        var output = Array(
            repeating: [Float](repeating: 0, count: localBinCount), count: 3
        )
        guard time.count == flux.count, period > 0, duration > 0, time.count >= 50 else {
            return output
        }
        let center = t0 + duration / 2
        var oddTime: [Double] = [], oddFlux: [Double] = []
        var evenTime: [Double] = [], evenFlux: [Double] = []
        for index in time.indices {
            let transit = ((time[index] - center) / period).rounded(.toNearestOrEven)
            let remainder = transit.truncatingRemainder(dividingBy: 2)
            if remainder != 0 {
                oddTime.append(time[index]); oddFlux.append(flux[index])
            } else {
                evenTime.append(time[index]); evenFlux.append(flux[index])
            }
        }
        let odd = oddTime.count >= 50
            ? rawLocal(time: oddTime, flux: oddFlux, period: period, t0: t0, duration: duration)
            : nil
        let even = evenTime.count >= 50
            ? rawLocal(time: evenTime, flux: evenFlux, period: period, t0: t0, duration: duration)
            : nil
        if let odd, let even {
            let center = median(odd + even)
            let amplitude = max(abs((odd + even).map { $0 - center }.min() ?? 0), 1e-12)
            output[0] = odd.map { Float(($0 - center) / amplitude) }
            output[1] = even.map { Float(($0 - center) / amplitude) }
        }
        if let primary = rawLocal(
            time: time, flux: flux, period: period, t0: t0, duration: duration
        ), let secondary = rawLocal(
            time: time, flux: flux, period: period, t0: t0 + period / 2, duration: duration
        ) {
            let center = median(primary)
            let amplitude = max(abs(primary.map { $0 - center }.min() ?? 0), 1e-12)
            output[2] = secondary.map { Float(($0 - center) / amplitude) }
        }
        return output
    }

    public static func sigmoid(logit: Double) -> Double {
        let bounded = min(max(logit, -60), 60)
        return 1 / (1 + exp(-bounded))
    }

    private static func phaseFoldSort(
        time: [Double], flux: [Double], period: Double, t0: Double, duration: Double
    ) -> (phase: [Double], flux: [Double]) {
        let center = t0 + duration / 2
        let samples = zip(time, flux).map { sampleTime, sampleFlux -> (Double, Double) in
            var phase = (sampleTime - center).truncatingRemainder(dividingBy: period)
            if phase < 0 { phase += period }
            if phase >= period / 2 { phase -= period }
            return (phase, sampleFlux)
        }.sorted { $0.0 < $1.0 }
        return (samples.map(\.0), samples.map(\.1))
    }

    private static func rawLocal(
        time: [Double], flux: [Double], period: Double, t0: Double, duration: Double
    ) -> [Double]? {
        let folded = phaseFoldSort(
            time: time, flux: flux, period: period, t0: t0, duration: duration
        )
        let halfWindow = min(period / 2, 2 * duration)
        return try? medianFilter(
            x: folded.phase, y: folded.flux, count: localBinCount,
            width: duration * 0.16, minimum: -halfWindow, maximum: halfWindow
        )
    }

    private static func medianFilter(
        x: [Double], y: [Double], count: Int,
        width: Double, minimum: Double, maximum: Double
    ) throws -> [Double] {
        let inRangeCount = x.lazy.filter { $0 >= minimum && $0 <= maximum }.count
        guard inRangeCount >= 5, let first = x.first, let last = x.last,
              last - first >= (maximum - minimum) / 2 else {
            throw FFIViewError.sparseLightCurve
        }
        let spacing = (maximum - minimum - width) / Double(count - 1)
        var bins = [Double?](repeating: nil, count: count)
        for index in 0..<count {
            let lower = minimum + Double(index) * spacing
            let upper = lower + width
            let values = zip(x, y).compactMap { phase, value in
                phase >= lower && phase < upper ? value : nil
            }
            if !values.isEmpty { bins[index] = median(values) }
        }
        return try interpolateEmpty(bins)
    }

    private static func interpolateEmpty(_ values: [Double?]) throws -> [Double] {
        let known = values.indices.filter { values[$0] != nil }
        guard let first = known.first, let last = known.last else {
            throw FFIViewError.allBinsEmpty
        }
        var result = [Double](repeating: 0, count: values.count)
        for index in values.indices {
            if let value = values[index] {
                result[index] = value
            } else if index < first {
                result[index] = values[first]!
            } else if index > last {
                result[index] = values[last]!
            } else {
                let upper = known.first { $0 > index }!
                let lower = known.last { $0 < index }!
                let fraction = Double(index - lower) / Double(upper - lower)
                result[index] = values[lower]! + fraction * (values[upper]! - values[lower]!)
            }
        }
        return result
    }

    private static func normalize(_ values: [Double]) throws -> [Float] {
        let center = median(values)
        let shifted = values.map { $0 - center }
        let amplitude = max(abs(shifted.min() ?? 0), 1e-12)
        return shifted.map { Float($0 / amplitude) }
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2)
            ? (sorted[middle - 1] + sorted[middle]) / 2
            : sorted[middle]
    }
}
