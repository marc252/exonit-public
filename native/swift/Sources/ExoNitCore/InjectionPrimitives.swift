import Foundation

/// NumPy legacy `RandomState(seed)` MT19937, including its Float64 `random_sample` path.
public struct NumPyMT19937: Sendable {
    private var state = [UInt32](repeating: 0, count: 624)
    private var index = 624

    public init(seed: UInt32) {
        state[0] = seed
        for i in 1..<624 {
            state[i] = 1_812_433_253 &* (state[i - 1] ^ (state[i - 1] >> 30)) &+ UInt32(i)
        }
    }

    public mutating func uniform01() -> Double {
        let a = nextUInt32() >> 5
        let b = nextUInt32() >> 6
        return (Double(a) * 67_108_864 + Double(b)) / 9_007_199_254_740_992
    }

    private mutating func nextUInt32() -> UInt32 {
        if index >= 624 { twist() }
        var value = state[index]; index += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C_5680
        value ^= (value << 15) & 0xEFC6_0000
        value ^= value >> 18
        return value
    }

    private mutating func twist() {
        for i in 0..<624 {
            let combined = (state[i] & 0x8000_0000) | (state[(i + 1) % 624] & 0x7FFF_FFFF)
            var value = state[(i + 397) % 624] ^ (combined >> 1)
            if combined & 1 != 0 { value ^= 0x9908_B0DF }
            state[i] = value
        }
        index = 0
    }
}

public enum TransitInjector {
    public static func depth(radiusEarth: Double, stellarRadiusSolar: Double) -> Double {
        let ratio = radiusEarth * 0.00916 / stellarRadiusSolar
        return ratio * ratio
    }

    public static func inject(
        time: [Double], flux: [Double], period: Double, t0: Double,
        depth: Double, duration: Double
    ) -> [Double] {
        var result = flux
        let half = duration / period / 2
        for index in time.indices {
            var phase = (time[index] - t0).truncatingRemainder(dividingBy: period) / period
            if phase < 0 { phase += 1 }
            if phase < half || phase > 1 - half { result[index] -= depth }
        }
        return result
    }
}
