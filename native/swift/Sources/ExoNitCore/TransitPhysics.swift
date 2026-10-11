import Foundation

public enum TransitPhysics {
    public static func durationDays(
        periodDays: Double,
        stellarRadiusSolar: Double = 1,
        stellarMassSolar: Double? = nil,
        impactParameter: Double = 0,
        planetRadiusEarth: Double = 1
    ) -> Double {
        guard periodDays > 0, stellarRadiusSolar > 0 else { return 0 }
        let mass = stellarMassSolar ?? pow(stellarRadiusSolar, 1.25)
        guard mass > 0 else { return 0 }
        let semimajorAU = pow(mass * pow(periodDays / 365.25, 2), 1.0 / 3.0)
        let semimajorSolar = semimajorAU * 214.94
        let radiusRatio = planetRadiusEarth * 0.00916 / stellarRadiusSolar
        let argumentSquared = pow(1 + radiusRatio, 2) - impactParameter * impactParameter
        guard argumentSquared > 0 else { return 0 }
        let sineArgument = stellarRadiusSolar / semimajorSolar * sqrt(argumentSquared)
        if sineArgument >= 1 { return periodDays / 2 }
        return periodDays / Double.pi * asin(sineArgument)
    }

    public static func physicalT14Maximum(
        periodDays: Double, stellarRadiusSolar: Double?, stellarMassSolar: Double?
    ) -> Double {
        durationDays(
            periodDays: periodDays,
            stellarRadiusSolar: stellarRadiusSolar ?? 1,
            stellarMassSolar: stellarMassSolar,
            planetRadiusEarth: 11.2
        )
    }
}
