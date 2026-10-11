import Foundation

public enum DetrendError: Error, CustomStringConvertible, Sendable {
    case lengthMismatch
    case maskLengthMismatch
    case emptyInput

    public var description: String {
        switch self {
        case .lengthMismatch: return "time and flux lengths differ"
        case .maskLengthMismatch: return "transit mask length differs from time"
        case .emptyInput: return "empty light curve"
        }
    }
}

/// Float64 port of Wotan's `flatten(method="biweight")` path used by ExoNit.
public enum BiweightDetrender {
    public static func flatten(
        time: [Double],
        flux: [Double],
        windowLength: Double = 0.5,
        edgeCutoff: Double = 0.5,
        breakTolerance: Double = 0.5,
        cval: Double = 5.0,
        transitMask: [Bool]? = nil
    ) throws -> [Double] {
        guard time.count == flux.count else { throw DetrendError.lengthMismatch }
        guard !time.isEmpty else { throw DetrendError.emptyInput }
        if let transitMask, transitMask.count != time.count {
            throw DetrendError.maskLengthMismatch
        }

        var compressedTime: [Double] = []
        var compressedFlux: [Double] = []
        var compressedMask: [Double] = []
        var originalIndexes: [Int] = []
        compressedTime.reserveCapacity(time.count)
        compressedFlux.reserveCapacity(time.count)
        compressedMask.reserveCapacity(time.count)
        originalIndexes.reserveCapacity(time.count)

        for index in time.indices where !(time[index] * flux[index]).isNaN {
            compressedTime.append(time[index])
            compressedFlux.append(flux[index])
            compressedMask.append((transitMask?[index] ?? false) ? 0.0 : 1.0)
            originalIndexes.append(index)
        }
        guard !compressedTime.isEmpty else {
            return Array(repeating: .nan, count: time.count)
        }

        var boundaries = [0]
        if compressedTime.count > 1 {
            for index in 1..<compressedTime.count
                where compressedTime[index] - compressedTime[index - 1] > breakTolerance {
                boundaries.append(index)
            }
        }
        // Wotan appends len(time)+1; Python slicing clamps this to the array end.
        boundaries.append(compressedTime.count + 1)

        var compressedTrend: [Double] = []
        compressedTrend.reserveCapacity(compressedTime.count)
        for segment in 0..<(boundaries.count - 1) {
            let lower = boundaries[segment]
            let upper = min(boundaries[segment + 1], compressedTime.count)
            guard lower < upper else { continue }
            compressedTrend.append(contentsOf: runningSegment(
                time: Array(compressedTime[lower..<upper]),
                flux: Array(compressedFlux[lower..<upper]),
                mask: Array(compressedMask[lower..<upper]),
                windowLength: windowLength,
                edgeCutoff: edgeCutoff,
                cval: cval
            ))
        }

        var trend = Array(repeating: Double.nan, count: time.count)
        for index in originalIndexes.indices {
            let value = compressedTrend[index]
            trend[originalIndexes[index]] = value == 0 ? .nan : value
        }
        return zip(flux, trend).map { $0 / $1 }
    }

    private static func runningSegment(
        time: [Double],
        flux: [Double],
        mask: [Double],
        windowLength: Double,
        edgeCutoff requestedEdgeCutoff: Double,
        cval: Double
    ) -> [Double] {
        let count = time.count
        var output = Array(repeating: Double.nan, count: count)
        let halfWindow = windowLength / 2
        let edgeCutoff = min(requestedEdgeCutoff, halfWindow)
        let low = (time.min() ?? 0) + edgeCutoff
        let high = (time.max() ?? 0) - edgeCutoff
        var start = 0
        var end = 0

        for index in 0..<count where time[index] >= low && time[index] <= high {
            while start < count && time[start] < time[index] - halfWindow { start += 1 }
            while end < count - 1 && time[end] < time[index] + halfWindow { end += 1 }
            if start >= end { continue }
            var window: [Double] = []
            window.reserveCapacity(end - start)
            for sample in start..<end {
                let maskedFlux = mask[sample] == 0 ? -flux[sample] : flux[sample]
                if maskedFlux > -1e-12 { window.append(maskedFlux) }
            }
            if !window.isEmpty { output[index] = biweightLocation(window, cval: cval) }
        }
        return output
    }

    private static func biweightLocation(_ data: [Double], cval: Double) -> Double {
        let initial = median(data)
        var deviations = data.map { abs($0 - initial) }
        let mad = medianInPlace(&deviations)
        if mad == 0 { return initial }

        let inverseCMAD = 1 / (cval * mad)
        var center = initial
        var oldCenter = initial
        var delta = Double.infinity
        while abs(delta) > 1e-6 {
            var numerator = 0.0
            var denominator = 0.0
            for value in data {
                let distance = value - center
                let scaled = distance * inverseCMAD
                if abs(scaled) < 1 {
                    let base = 1 - scaled * scaled
                    let weight = base * base
                    numerator += distance * weight
                    denominator += weight
                }
            }
            center += numerator / denominator
            delta = oldCenter - center
            oldCenter = center
        }
        return center
    }

    private static func median(_ values: [Double]) -> Double {
        var copy = values
        return medianInPlace(&copy)
    }

    private static func medianInPlace(_ values: inout [Double]) -> Double {
        values.sort()
        let middle = values.count / 2
        if values.count % 2 == 0 { return (values[middle - 1] + values[middle]) / 2 }
        return values[middle]
    }
}
