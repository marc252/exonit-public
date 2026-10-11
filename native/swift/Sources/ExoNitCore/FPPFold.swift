import Foundation

/// Phase folding and median binning matching `phase_fold.py`.
public enum FPPFold {
    private static func mod1(_ x: Double) -> Double {
        var r = x.truncatingRemainder(dividingBy: 1)
        if r != 0 && r < 0 { r += 1 }
        return r
    }
    public static func phaseFold(_ time: [Double], t0: Double, period: Double) -> [Double] {
        time.map { t in var p = mod1((t - t0) / period); if p > 0.5 { p -= 1 }; return p }
    }
    private static func upperBound(_ bins: [Double], _ x: Double) -> Int {
        var lo = 0, hi = bins.count
        while lo < hi { let mid = (lo + hi) / 2; if x < bins[mid] { hi = mid } else { lo = mid + 1 } }
        return lo
    }
    public static func bin(time: [Double], flux: [Double], period: Double, t0: Double, bins nBins: Int) -> (time: [Double], flux: [Double]) {
        guard nBins > 0 else { return ([], []) }
        let phase = phaseFold(time, t0: t0, period: period)
        let step = 1 / Double(nBins)
        var edges = (0...nBins).map { Double($0) * step - 0.5 }; edges[nBins] = 0.5
        let centers = (0..<nBins).map { 0.5 * (edges[$0] + edges[$0 + 1]) }
        var grouped = Array(repeating: [Double](), count: nBins)
        for (p, f) in zip(phase, flux) { let index = max(0, min(nBins - 1, upperBound(edges, p) - 1)); grouped[index].append(f) }
        var outTime = [Double](), outFlux = [Double]()
        for i in 0..<nBins where !grouped[i].isEmpty { outTime.append(centers[i] * period); outFlux.append(FPPNumpy.nanMedian(grouped[i])) }
        return (outTime, outFlux)
    }
    public static func renorm(flux: [Double], fluxError: Double, starFluxRatio: Double) -> (flux: [Double], error: Double) {
        (flux.map { ($0 - (1 - starFluxRatio)) / starFluxRatio }, fluxError / starFluxRatio)
    }
}
