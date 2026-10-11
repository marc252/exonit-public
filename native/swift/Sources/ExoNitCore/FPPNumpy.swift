import Foundation

/// NumPy-compatible reductions used by the FPP port.
public enum FPPNumpy {
    public static func pairwiseSum(_ a: [Double]) -> Double {
        let n = a.count
        if n < 8 { return a.reduce(0, +) }
        if n <= 128 {
            var lanes = Array(a.prefix(8)); var i = 8
            while i < n - n % 8 { for j in 0..<8 { lanes[j] += a[i + j] }; i += 8 }
            var result = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
                + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
            while i < n { result += a[i]; i += 1 }
            return result
        }
        var half = n / 2; half -= half % 8
        return pairwiseSum(Array(a[..<half])) + pairwiseSum(Array(a[half...]))
    }
    public static func mean(_ a: [Double]) -> Double { pairwiseSum(a) / Double(a.count) }
    public static func std(_ a: [Double]) -> Double {
        let m = mean(a)
        return sqrt(pairwiseSum(a.map { ($0 - m) * ($0 - m) }) / Double(a.count))
    }
    public static func median(_ a: [Double]) -> Double {
        guard !a.isEmpty else { return .nan }
        let v = a.sorted(), n = v.count
        return n % 2 == 1 ? v[n / 2] : mean([v[n / 2 - 1], v[n / 2]])
    }
    public static func nanMedian(_ a: [Double]) -> Double { median(a.filter { !$0.isNaN }) }
    public static func nanToNum(_ x: Double) -> Double {
        if x.isNaN { return 0 }; if x == .infinity { return .greatestFiniteMagnitude }
        if x == -.infinity { return -.greatestFiniteMagnitude }; return x
    }
    /// Scalar pow is kept as a named seam because Python scalar `** 2` and
    /// NumPy array `** 2` take different numerical paths.
    public static func libmPow(_ x: Double, _ y: Double) -> Double { pow(x, y) }
}
