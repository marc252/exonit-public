import Foundation

/// Frozen inputs for one native FPP evaluation.  The nearby payload remains
/// JSON so this type can be fed directly from the `/api/nearby` response.
/// `durationDays` is retained for parity with the Rust/Python call surface;
/// the native calculation itself does not consume it.
public struct FPPInputsModel: Sendable {
    public let ticID: String
    public let periodDays: Double
    public let t0: Double
    public let durationDays: Double
    public let depthPPM: Double
    public let time: [Double]
    public let flux: [Double]
    public let nearby: [String: JSONValue]
    public let trilegalCSV: String?

    public init(
        ticID: String, periodDays: Double, t0: Double, durationDays: Double,
        depthPPM: Double, time: [Double], flux: [Double],
        nearby: [String: JSONValue], trilegalCSV: String? = nil
    ) {
        self.ticID = ticID
        self.periodDays = periodDays
        self.t0 = t0
        self.durationDays = durationDays
        self.depthPPM = depthPPM
        self.time = time
        self.flux = flux
        self.nearby = nearby
        self.trilegalCSV = trilegalCSV
    }
}

/// Rust names kept as aliases so parity harnesses can use the same vocabulary
/// without exposing a second implementation of the input model.
public typealias FppInputs = FPPInputsModel

public struct FPPResult: Equatable, Sendable {
    public let fpp: Double
    public let nfpp: Double
    public let verdict: FPPVerdict
    public let highCrowding: Bool
    public let galacticLatitude: Double
    public let nearbyCount: Int
    public let trilegalUsed: Bool

    public init(
        fpp: Double, nfpp: Double, verdict: FPPVerdict,
        highCrowding: Bool, galacticLatitude: Double,
        nearbyCount: Int, trilegalUsed: Bool
    ) {
        self.fpp = fpp
        self.nfpp = nfpp
        self.verdict = verdict
        self.highCrowding = highCrowding
        self.galacticLatitude = galacticLatitude
        self.nearbyCount = nearbyCount
        self.trilegalUsed = trilegalUsed
    }
}

/// Top-level checkpoints corresponding to Rust `CalcProbes`.
public struct FPPCalcProbes: Sendable {
    public var scen: FPPProbes
    public var phaseFold: (time: [Double], flux: [Double])?
    public var fluxRatios: [(id: String, fluxRatio: Double, transitDepth: Double)]
    public var renorm: [(fluxRatio: Double, sigma: Double, fluxHead: [Double])]

    public init() {
        scen = FPPProbes()
        phaseFold = nil
        fluxRatios = []
        renorm = []
    }
}

public typealias CalcProbes = FPPCalcProbes

private func fppFiniteMaximum(_ values: [Double]) -> Double {
    values.filter { $0.isFinite }.max() ?? 0.0
}

/// Swift counterpart of Rust `fpp::calc_fpp`.
///
/// The function intentionally has no HTTP, filesystem, or task-runtime
/// dependency.  Scenario kernels own the stochastic draw order; this layer
/// only preserves the Rust orchestration and checkpoint order.
public enum FPPCalculation {
    public static func calcFPP(
        _ inputs: FPPInputsModel, draws: Int, seed: UInt32,
        probes: inout FPPCalcProbes
    ) -> FPPResult? {
        let folded = FPPFold.bin(
            time: inputs.time, flux: inputs.flux, period: inputs.periodDays,
            t0: inputs.t0, bins: 1000
        )
        probes.phaseFold = (folded.time, folded.flux)
        guard folded.time.count >= 10 else { return nil }

        var timePF = [Double](), fluxPF = [Double]()
        timePF.reserveCapacity(folded.time.count)
        fluxPF.reserveCapacity(folded.flux.count)
        for (time, flux) in zip(folded.time, folded.flux)
        where !time.isNaN && !flux.isNaN {
            timePF.append(time)
            fluxPF.append(flux)
        }
        var sigma = FPPNumpy.std(fluxPF)
        if sigma <= 0 { sigma = 1e-6 }

        guard let nearby = FPPInputs.parseNearby(inputs.nearby, ticID: inputs.ticID),
              !nearby.stars.isEmpty else { return nil }
        var stars = nearby.stars
        let nearbyCount = stars.count - 1
        let highCrowding = abs(nearby.galacticLatitude) < 10 || nearbyCount > 15

        if let csv = inputs.trilegalCSV {
            stars.append(contentsOf: FPPInputs.loadTrilegal(
                csv, targetRA: stars[0].ra, targetDec: stars[0].dec,
                targetTmag: stars[0].tmag
            ))
        }
        let trilegalUsed = stars.contains { $0.isTrilegal }
        FPPInputs.computeFluxRatios(&stars, depthPPM: inputs.depthPPM)
        probes.fluxRatios = stars.map {
            (id: $0.id, fluxRatio: $0.fluxRatio, transitDepth: $0.transitDepth)
        }

        let target = stars[0]
        guard !target.mass.isNaN, !target.radius.isNaN, !target.teff.isNaN else {
            return nil
        }
        let logg = log10(
            FPPScenarioConstants.gCGS * target.mass * FPPScenarioConstants.msun
                / FPPNumpy.libmPow(target.radius * FPPScenarioConstants.rsun, 2)
        )
        let ldc = FPPStellar.limbDarkening(
            temperature: target.teff, logg: logg, metallicity: 0
        )
        let targetRenorm = FPPFold.renorm(
            flux: fluxPF, fluxError: sigma, starFluxRatio: target.fluxRatio
        )
        probes.renorm.append((
            fluxRatio: target.fluxRatio, sigma: sigma,
            fluxHead: Array(targetRenorm.flux.prefix(8))
        ))

        var rng = FPPRNG(seed: seed)
        let star = FPPStarParameters(
            massSolar: target.mass, radiusSolar: target.radius,
            effectiveTemperature: target.teff
        )
        let n = max(0, draws)
        let lnZTP = FPPScenarios.lnzTTP(
            rng: &rng, probes: &probes.scen, time: timePF,
            flux: targetRenorm.flux, sigma: targetRenorm.error,
            period: inputs.periodDays, star: star, count: n,
            u1: ldc.u1, u2: ldc.u2, companionFluxRatio: Array(repeating: 0, count: n),
            companionIsHost: false, label: "_lnZ_TTP"
        )
        let (lnZEB, lnZEBTwin) = FPPScenarios.lnzTEB(
            rng: &rng, probes: &probes.scen, time: timePF,
            flux: targetRenorm.flux, sigma: targetRenorm.error,
            period: inputs.periodDays, star: star, count: n,
            u1: ldc.u1, u2: ldc.u2, companionFluxRatio: Array(repeating: 0, count: n),
            companionIsHost: false, label: "_lnZ_TEB"
        )

        let physicalCompanion = !target.parallax.isNaN
        let lnZPTP: Double
        let lnZPEB: Double
        let lnZPEBTwin: Double
        if physicalCompanion {
            lnZPTP = FPPScenarios.lnzPTP(
                rng: &rng, probes: &probes.scen, time: timePF,
                flux: targetRenorm.flux, sigma: targetRenorm.error,
                period: inputs.periodDays, star: star, count: n,
                u1: ldc.u1, u2: ldc.u2, parallax: target.parallax
            )
            probes.scen.recordLnZ("_lnZ_PTP", [lnZPTP])
            (lnZPEB, lnZPEBTwin) = FPPScenarios.lnzPEB(
                rng: &rng, probes: &probes.scen, time: timePF,
                flux: targetRenorm.flux, sigma: targetRenorm.error,
                period: inputs.periodDays, star: star, count: n,
                u1: ldc.u1, u2: ldc.u2, parallax: target.parallax
            )
            probes.scen.recordLnZ("_lnZ_PEB", [lnZPEB, lnZPEBTwin])
        } else {
            lnZPTP = -.infinity
            lnZPEB = -.infinity
            lnZPEBTwin = -.infinity
        }

        var nearbyLnZ = [Double]()
        for neighbour in stars.dropFirst().filter({ $0.transitDepth > 0 }).prefix(5) {
            let mass = neighbour.mass.isNaN ? 1 : neighbour.mass
            let radius = neighbour.radius.isNaN ? 1 : neighbour.radius
            let temperature = neighbour.teff.isNaN ? 5777 : neighbour.teff
            let neighbourLogG = log10(
                FPPScenarioConstants.gCGS * mass * FPPScenarioConstants.msun
                    / FPPNumpy.libmPow(radius * FPPScenarioConstants.rsun, 2)
            )
            let neighbourLDC = FPPStellar.limbDarkening(
                temperature: temperature, logg: neighbourLogG, metallicity: 0
            )
            let renorm = FPPFold.renorm(
                flux: fluxPF, fluxError: sigma, starFluxRatio: neighbour.fluxRatio
            )
            probes.renorm.append((
                fluxRatio: neighbour.fluxRatio, sigma: sigma,
                fluxHead: Array(renorm.flux.prefix(8))
            ))
            let neighbourStar = FPPStarParameters(
                massSolar: mass, radiusSolar: radius,
                effectiveTemperature: temperature
            )
            let ntp = FPPScenarios.lnzTTP(
                rng: &rng, probes: &probes.scen, time: timePF,
                flux: renorm.flux, sigma: renorm.error,
                period: inputs.periodDays, star: neighbourStar,
                count: n / 2, u1: neighbourLDC.u1, u2: neighbourLDC.u2,
                companionFluxRatio: Array(repeating: 0, count: n / 2), companionIsHost: false,
                label: "_lnZ_TTP"
            )
            let (neb, nebTwin) = FPPScenarios.lnzTEB(
                rng: &rng, probes: &probes.scen, time: timePF,
                flux: renorm.flux, sigma: renorm.error,
                period: inputs.periodDays, star: neighbourStar,
                count: n / 2, u1: neighbourLDC.u1, u2: neighbourLDC.u2,
                companionFluxRatio: Array(repeating: 0, count: n / 2),
                companionIsHost: false, label: "_lnZ_TEB"
            )
            nearbyLnZ.append(contentsOf: [ntp, neb, nebTwin])
        }

        var allLnZ = [lnZTP, lnZEB, lnZEBTwin, lnZPTP, lnZPEB, lnZPEBTwin]
        allLnZ.append(contentsOf: nearbyLnZ)
        allLnZ = allLnZ.map { $0.isFinite ? $0 : -.infinity }
        let maxLnZ = fppFiniteMaximum(allLnZ)
        var probabilities = allLnZ.map { exp($0 - maxLnZ) }
        let total = FPPNumpy.pairwiseSum(probabilities)
        guard total > 0 else { return nil }
        for index in probabilities.indices { probabilities[index] /= total }

        let fpp = min(1, max(0, 1 - probabilities[0] - probabilities[3]))
        let nfpp = min(1, max(0, FPPNumpy.pairwiseSum(Array(probabilities.dropFirst(6)))))
        let roundedLatitude = Double(
            String(format: "%.2f", locale: Locale(identifier: "en_US_POSIX"), nearby.galacticLatitude)
        ) ?? nearby.galacticLatitude
        return FPPResult(
            fpp: fpp, nfpp: nfpp,
            verdict: FPPPolicy.verdict(
                fpp: fpp, nfpp: nfpp, highCrowding: highCrowding
            ),
            highCrowding: highCrowding, galacticLatitude: roundedLatitude,
            nearbyCount: nearbyCount, trilegalUsed: trilegalUsed
        )
    }
}

public extension FPPInputsModel {
    static func calcFPP(
        _ inputs: FPPInputsModel, draws: Int, seed: UInt32,
        probes: inout FPPCalcProbes
    ) -> FPPResult? {
        FPPCalculation.calcFPP(inputs, draws: draws, seed: seed, probes: &probes)
    }
}
