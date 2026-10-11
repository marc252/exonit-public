import Foundation

/// Per-draw parameters for the FPP transit likelihood kernels.
public struct FPPBatchParameters: Sendable {
    public let radiusRatios: [Double]
    public let periods: [Double]
    public let inclinationsRadians: [Double]
    public let semiMajorAxes: [Double]
    public let eccentricities: [Double]
    public let argumentsRadians: [Double]
    public let companionFluxRatios: [Double]
    public let companionIsHost: Bool

    public init(
        radiusRatios: [Double], periods: [Double], inclinationsRadians: [Double],
        semiMajorAxes: [Double], eccentricities: [Double], argumentsRadians: [Double],
        companionFluxRatios: [Double], companionIsHost: Bool
    ) {
        self.radiusRatios = radiusRatios; self.periods = periods
        self.inclinationsRadians = inclinationsRadians; self.semiMajorAxes = semiMajorAxes
        self.eccentricities = eccentricities; self.argumentsRadians = argumentsRadians
        self.companionFluxRatios = companionFluxRatios; self.companionIsHost = companionIsHost
    }
}

public enum FPPKernels {
    private static func logLikelihoodConstant(sigma: Double) -> Double {
        -0.5 * log(2.0 * Double.pi) - log(sigma)
    }

    public static func transitLogLikelihood(
        time: [Double], flux: [Double], sigma: Double, u1: Double, u2: Double,
        parameters: FPPBatchParameters
    ) -> [Double] {
        let constant = logLikelihoodConstant(sigma: sigma)
        return parameters.radiusRatios.indices.map { i in
            var model = FPPTransit.lightCurve(
                time: time, planetRadiusRatio: parameters.radiusRatios[i], t0: 0.0,
                period: parameters.periods[i], aOverRStar: parameters.semiMajorAxes[i],
                inclinationRadians: parameters.inclinationsRadians[i], eccentricity: parameters.eccentricities[i],
                argumentPeriastronRadians: parameters.argumentsRadians[i], u1: u1, u2: u2
            )
            let companionRatio = parameters.companionFluxRatios[i]
            if companionRatio > 0.0 {
                let dilution = parameters.companionIsHost
                    ? 1.0 / (companionRatio / (1.0 - companionRatio))
                    : companionRatio / (1.0 - companionRatio)
                for j in model.indices { model[j] = (model[j] + dilution) / (1.0 + dilution) }
            }
            var chiSquared = 0.0
            for (observed, predicted) in zip(flux, model) {
                let difference = observed - predicted
                chiSquared += difference * difference
            }
            return constant - 0.5 * chiSquared / (sigma * sigma)
        }
    }

    public static func eclipsingBinaryLogLikelihood(
        time: [Double], flux: [Double], sigma: Double, u1: Double, u2: Double,
        binaryFluxRatios: [Double], parameters: FPPBatchParameters
    ) -> [Double] {
        let constant = logLikelihoodConstant(sigma: sigma)
        return parameters.radiusRatios.indices.map { i in
            var radiusRatio = parameters.radiusRatios[i]
            if abs(radiusRatio - 1.0) < 1e-6 { radiusRatio *= 0.999 }
            var model = FPPTransit.lightCurve(
                time: time, planetRadiusRatio: radiusRatio, t0: 0.0,
                period: parameters.periods[i], aOverRStar: parameters.semiMajorAxes[i],
                inclinationRadians: parameters.inclinationsRadians[i], eccentricity: parameters.eccentricities[i],
                argumentPeriastronRadians: parameters.argumentsRadians[i], u1: u1, u2: u2
            )
            let binaryRatio = binaryFluxRatios[i]
            let binaryFlux = binaryRatio < 1.0 ? binaryRatio / (1.0 - binaryRatio) : 1e6
            let companionRatio = parameters.companionFluxRatios[i]
            let companionFlux = companionRatio > 0.0 ? companionRatio / (1.0 - companionRatio) : 0.0
            let secondaryDepth: Double
            if parameters.companionIsHost {
                let ratio = binaryFlux / max(companionFlux, 1e-30)
                for j in model.indices { model[j] = (model[j] + ratio) / (1.0 + ratio) }
                let dilution = 1.0 / (companionFlux + binaryFlux)
                for j in model.indices { model[j] = (model[j] + dilution) / (1.0 + dilution) }
                secondaryDepth = binaryRatio / (1.0 + dilution)
            } else {
                for j in model.indices { model[j] = (model[j] + binaryFlux) / (1.0 + binaryFlux) }
                let dilution = companionFlux / (1.0 + binaryFlux)
                for j in model.indices { model[j] = (model[j] + dilution) / (1.0 + dilution) }
                secondaryDepth = dilution > 0.0 ? binaryRatio / (1.0 + dilution) : binaryRatio
            }
            if secondaryDepth >= 1.5 * sigma { return -Double.infinity }
            var chiSquared = 0.0
            for (observed, predicted) in zip(flux, model) {
                let difference = observed - predicted
                chiSquared += difference * difference
            }
            return constant - 0.5 * chiSquared / (sigma * sigma)
        }
    }
}
