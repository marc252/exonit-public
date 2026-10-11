import Foundation

public struct FFIStellarMetadata: Equatable, Sendable {
    public let tmag: Double?
    public let teff: Double?
    public let logg: Double?
    public let radius: Double?
    public let mass: Double?

    public init(
        tmag: Double?, teff: Double?, logg: Double?, radius: Double?, mass: Double?
    ) {
        self.tmag = tmag; self.teff = teff; self.logg = logg
        self.radius = radius; self.mass = mass
    }
}

public struct ExoMinerFFIFeatures: Equatable, Sendable {
    public let global: [Float]
    public let local: [Float]
    public let centroid: [Float]
    public let centroidMask: [Float]
    public let stellar: [Float]
    public let tce: [Float]
    public let auxiliary: [[Float]]
}

public enum ExoMinerFFIFeatureBuilder {
    public static func make(
        time: [Double], flux: [Double], period: Double, t0: Double,
        duration: Double, depthPPM: Double, snr: Double,
        stellar metadata: FFIStellarMetadata
    ) throws -> ExoMinerFFIFeatures {
        let views = try FFIViewGenerator.makeCNNViews(
            time: time, flux: flux, period: period, t0: t0, duration: duration
        )
        let auxiliary = FFIViewGenerator.makeAuxiliaryViews(
            time: time, flux: flux, period: period, t0: t0, duration: duration
        )
        let inferredMass: Double?
        if let mass = finite(metadata.mass) {
            inferredMass = mass
        } else if let logg = finite(metadata.logg), let radius = finite(metadata.radius) {
            inferredMass = pow(10, logg) * radius * radius / pow(10, 4.44)
        } else {
            inferredMass = nil
        }
        let stellar: [Float] = [
            z(metadata.tmag, mean: 12, scale: 2.5),
            z(metadata.teff, mean: 5_500, scale: 1_500),
            z(metadata.logg, mean: 4.4, scale: 0.5),
            z(metadata.radius, mean: 1, scale: 0.5),
            z(inferredMass, mean: 1, scale: 0.5),
        ].map(Float.init)
        let tce: [Float] = [
            log10(max(period, 0.1)),
            log10(max(depthPPM, 1)),
            log10(max(duration * 24, 0.1)),
            log10(max(snr, 1)),
        ].map(Float.init)
        return ExoMinerFFIFeatures(
            global: views.global, local: views.local,
            centroid: [Float](repeating: 0, count: 61), centroidMask: [0],
            stellar: stellar, tce: tce, auxiliary: auxiliary
        )
    }

    private static func z(_ value: Double?, mean: Double, scale: Double) -> Double {
        guard let value = finite(value) else { return 0 }
        return (value - mean) / scale
    }

    private static func finite(_ value: Double?) -> Double? {
        guard let value, value.isFinite else { return nil }
        return value
    }
}
