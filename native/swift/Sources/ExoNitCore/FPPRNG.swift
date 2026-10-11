import Foundation

/// NumPy legacy RandomState transforms used by the FPP port.
public struct FPPRNG: Sendable {
    private var mt: NumPyMT19937
    private var hasGauss = false
    private var cachedGauss = 0.0

    public init(seed: UInt32) { mt = NumPyMT19937(seed: seed) }
    public mutating func rand() -> Double { mt.uniform01() }
    public mutating func gauss() -> Double {
        if hasGauss { hasGauss = false; defer { cachedGauss = 0 }; return cachedGauss }
        while true {
            let x1 = 2 * rand() - 1, x2 = 2 * rand() - 1
            let r2 = x1 * x1 + x2 * x2
            if r2 < 1 && r2 != 0 {
                let f = sqrt(-2 * log(r2) / r2)
                cachedGauss = f * x1; hasGauss = true
                return f * x2
            }
        }
    }
    public mutating func standardExponential() -> Double { -log(1 - rand()) }
    public mutating func standardGamma(_ shape: Double) -> Double {
        if shape == 1 { return standardExponential() }
        if shape < 1 {
            while true {
                let u = rand(), v = standardExponential()
                if u <= 1 - shape {
                    let x = pow(u, 1 / shape); if x <= v { return x }
                } else {
                    let y = -log((1 - u) / shape)
                    let x = pow(1 - shape + shape * y, 1 / shape)
                    if x <= v + y { return x }
                }
            }
        }
        let b = shape - 1 / 3, c = 1 / sqrt(9 * b)
        while true {
            var x: Double; var v: Double
            repeat { x = gauss(); v = 1 + c * x } while v <= 0
            v *= v * v; let u = rand()
            if u < 1 - 0.0331 * x * x * x * x { return b * v }
            if log(u) < 0.5 * x * x + b * (1 - v + log(v)) { return b * v }
        }
    }
    public mutating func beta(_ a: Double, _ b: Double) -> Double {
        if a <= 1 && b <= 1 {
            while true {
                let u = rand(), v = rand(), x = pow(u, 1 / a), y = pow(v, 1 / b)
                if x + y <= 1 {
                    if x + y > 0 { return x / (x + y) }
                    var lx = log(u) / a, ly = log(v) / b
                    let lm = max(lx, ly); lx -= lm; ly -= lm
                    return exp(lx - log(exp(lx) + exp(ly)))
                }
            }
        }
        let ga = standardGamma(a), gb = standardGamma(b)
        return ga / (ga + gb)
    }
    public mutating func power(_ a: Double) -> Double { pow(rand(), 1 / a) }

    public static func ticSeedKey(ticID: String, period: Double, t0: Double) -> String {
        "\(ticID):\(scientific(period)):\(scientific(t0))"
    }
    public static func ticSeed(ticID: String, period: Double, t0: Double) -> UInt32 {
        var h: UInt64 = 0xcbf29ce484222325
        for byte in ticSeedKey(ticID: ticID, period: period, t0: t0).utf8 {
            h ^= UInt64(byte); h = h &* 0x00000100000001b3
        }
        return UInt32(truncatingIfNeeded: h)
    }
    private static func scientific(_ x: Double) -> String {
        let raw = String(
            format: "%.10e", locale: Locale(identifier: "en_US_POSIX"), x
        ).lowercased()
        let p = raw.split(separator: "e", maxSplits: 1).map(String.init)
        guard p.count == 2, let e = Int(p[1]) else { return raw }
        return "\(p[0])e\(e < 0 ? "-" : "+")\(String(format: "%02d", abs(e)))"
    }
}
