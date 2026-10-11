import Foundation

/// Stellar relations and TESS quadratic limb-darkening lookup for FPP.
public enum FPPStellar {
    private struct Spline {
        let knots: [Double]
        let coefficients: [Double]
        let degree: Int

        init(knotBits: [UInt64], coefficientBits: [UInt64], degree: Int) {
            knots = knotBits.map(Double.init(bitPattern:))
            coefficients = coefficientBits.map(Double.init(bitPattern:))
            self.degree = degree
        }

        /// FITPACK `splev(..., ext=0)` using the same `fpbspl` recurrence as
        /// the Rust/Python reference. Repeated endpoint knots are guarded.
        func evaluate(_ x: Double) -> Double {
            let k = degree
            let n = knots.count
            var interval = k
            let upper = n - k - 2
            while interval < upper && x >= knots[interval + 1] { interval += 1 }
            var h = [Double](repeating: 0.0, count: 8)
            var hh = [Double](repeating: 0.0, count: 8)
            h[0] = 1.0
            if k > 0 {
                for j in 1...k {
                    for i in 0..<j { hh[i] = h[i] }
                    h[0] = 0.0
                    for i in 0..<j {
                        let li = knots[interval + i + 1]
                        let lj = knots[interval + i + 1 - j]
                        if li == lj {
                            h[i + 1] = 0.0
                            continue
                        }
                        let f = hh[i] / (li - lj)
                        h[i] += f * (li - x)
                        h[i + 1] = f * (x - lj)
                    }
                }
            }
            var result = 0.0
            for j in 0...k { result += coefficients[interval - k + j] * h[j] }
            return result
        }
    }

    private static let rTorres = Spline(
        knotBits: FPPStellarCoefficients.R_TORRES_T_BITS,
        coefficientBits: FPPStellarCoefficients.R_TORRES_C_BITS,
        degree: FPPStellarCoefficients.R_TORRES_K
    )
    private static let tTorres = Spline(
        knotBits: FPPStellarCoefficients.T_TORRES_T_BITS,
        coefficientBits: FPPStellarCoefficients.T_TORRES_C_BITS,
        degree: FPPStellarCoefficients.T_TORRES_K
    )
    private static let rCool = Spline(
        knotBits: FPPStellarCoefficients.R_CDWRF_T_BITS,
        coefficientBits: FPPStellarCoefficients.R_CDWRF_C_BITS,
        degree: FPPStellarCoefficients.R_CDWRF_K
    )
    private static let tCool = Spline(
        knotBits: FPPStellarCoefficients.T_CDWRF_T_BITS,
        coefficientBits: FPPStellarCoefficients.T_CDWRF_C_BITS,
        degree: FPPStellarCoefficients.T_CDWRF_K
    )
    private static let flux = Spline(
        knotBits: FPPStellarCoefficients.FLUX_T_BITS,
        coefficientBits: FPPStellarCoefficients.FLUX_C_BITS,
        degree: FPPStellarCoefficients.FLUX_K
    )

    public static func relations(masses: [Double]) -> (radii: [Double], temperatures: [Double]) {
        var radii: [Double] = []; radii.reserveCapacity(masses.count)
        var temperatures: [Double] = []; temperatures.reserveCapacity(masses.count)
        for mass in masses {
            if mass > 0.63 {
                radii.append(max(0.1, rTorres.evaluate(mass)))
                temperatures.append(max(2800.0, tTorres.evaluate(mass)))
            } else {
                let clipped = min(0.63, max(0.1, mass))
                radii.append(max(0.1, rCool.evaluate(clipped)))
                temperatures.append(max(2800.0, tCool.evaluate(clipped)))
            }
        }
        return (radii, temperatures)
    }

    public static func fluxRelation(masses: [Double]) -> [Double] {
        masses.map { pow(10.0, flux.evaluate($0)) }
    }

    public static func limbDarkening(temperature: Double, logg: Double, metallicity: Double = 0.0) -> (u1: Double, u2: Double) {
        let rows = FPPLimbDarkeningData.rows
        guard !rows.isEmpty else { return (0.40, 0.26) }
        func nearest(_ value: Double, _ key: (FPPLimbDarkeningData.Row) -> Double) -> Double {
            var best = rows[0]; var distance = abs(key(best) - value)
            for row in rows.dropFirst() {
                let candidate = abs(key(row) - value)
                if candidate < distance { best = row; distance = candidate }
            }
            return key(best)
        }
        let z = nearest(metallicity) { $0.z }
        let teff = nearest(temperature) { $0.teff }
        let gravity = nearest(logg) { $0.logg }
        for row in rows where row.z == z && row.teff == teff && row.logg == gravity {
            return (row.u1, row.u2)
        }
        return (0.40, 0.26)
    }
}
