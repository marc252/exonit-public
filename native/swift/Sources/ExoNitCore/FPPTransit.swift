import Foundation

/// CPU reference transit model used by the native FPP implementation.
///
/// This is a direct transliteration of `worker/fpp_native.py`'s numba path
/// (`_kepler_eq` through `_compute_transit_lc`).  Keep the operation order
/// explicit: the FPP goldens compare IEEE-754 `Double` bit patterns.
public enum FPPTransit {
    public static func keplerEquation(meanAnomaly m: Double, eccentricity ecc: Double) -> Double {
        let tolerance = 1e-10
        var eccentricAnomaly = m
        for _ in 0..<30 {
            let delta = (eccentricAnomaly - ecc * sin(eccentricAnomaly) - m)
                / (1.0 - ecc * cos(eccentricAnomaly))
            eccentricAnomaly -= delta
            if abs(delta) < tolerance { break }
        }
        return eccentricAnomaly
    }

    public static func trueAnomaly(meanAnomaly m: Double, eccentricity ecc: Double) -> Double {
        if ecc < 1e-10 { return m }
        let eccentricAnomaly = keplerEquation(meanAnomaly: m, eccentricity: ecc)
        let cosE = cos(eccentricAnomaly)
        return atan2(sqrt(1.0 - ecc * ecc) * sin(eccentricAnomaly), cosE - ecc)
    }

    public static func projectedSeparation(
        time: [Double], t0: Double, period: Double, aOverRStar: Double,
        inclinationRadians: Double, eccentricity: Double, argumentPeriastronRadians: Double
    ) -> [Double] {
        let twoPi = 2.0 * Double.pi
        let cosInclination = cos(inclinationRadians)
        return time.map { ti in
            let phase = (ti - t0) / period
            let meanAnomaly = twoPi * (phase - floor(phase)) - argumentPeriastronRadians
            let nu = trueAnomaly(meanAnomaly: meanAnomaly, eccentricity: eccentricity)
            let radiusFactor = (1.0 - eccentricity * eccentricity)
                / (1.0 + eccentricity * cos(nu))
            let f = nu + argumentPeriastronRadians
            let sinF = sin(f)
            let cosF = cos(f)
            return aOverRStar * radiusFactor
                * sqrt(sinF * sinF + cosInclination * cosInclination * cosF * cosF)
        }
    }

    public static func ringFlux(z: Double, planetRadiusRatio p: Double, u1: Double, u2: Double) -> Double {
        let ringCount = 100
        if z >= 1.0 + p || p <= 0.0 { return 1.0 }
        var omega = 1.0 - u1 / 3.0 - u2 / 6.0
        if omega <= 0.0 { omega = 1.0 }
        let dr = 1.0 / Double(ringCount)
        var blocked = 0.0
        for i in 0..<ringCount {
            let r = (Double(i) + 0.5) * dr
            let r2 = r * r
            let mu = sqrt(1.0 - r2)
            let oneMinusMu = 1.0 - mu
            let intensity = 1.0 - u1 * oneMinusMu - u2 * oneMinusMu * oneMinusMu
            let fraction: Double
            if z < 1e-12 {
                fraction = r <= p ? 1.0 : 0.0
            } else if r >= z + p || r + p <= z {
                fraction = 0.0
            } else if r <= abs(z - p) {
                fraction = p > z ? 1.0 : 0.0
            } else {
                let argument = (r2 + z * z - p * p) / (2.0 * r * z)
                if argument >= 1.0 {
                    fraction = 0.0
                } else if argument <= -1.0 {
                    fraction = 1.0
                } else {
                    fraction = acos(argument) / Double.pi
                }
            }
            blocked += intensity * fraction * 2.0 * r * dr
        }
        return 1.0 - blocked / omega
    }

    public static func lightCurve(
        time: [Double], planetRadiusRatio: Double, t0: Double, period: Double,
        aOverRStar: Double, inclinationRadians: Double, eccentricity: Double,
        argumentPeriastronRadians: Double, u1: Double, u2: Double
    ) -> [Double] {
        projectedSeparation(
            time: time, t0: t0, period: period, aOverRStar: aOverRStar,
            inclinationRadians: inclinationRadians, eccentricity: eccentricity,
            argumentPeriastronRadians: argumentPeriastronRadians
        ).map { ringFlux(z: $0, planetRadiusRatio: planetRadiusRatio, u1: u1, u2: u2) }
    }
}
