import Foundation

/// Scenario marginal likelihoods (`_lnZ_TTP`, `_lnZ_TEB`, `_lnZ_PTP`,
/// `_lnZ_PEB`) translated from `fpp_native.py`.  This module deliberately
/// has no HTTP/runtime dependency; `FPPProbes` preserves the chronological
/// sampler/dispatch/lnZ checkpoints used by the Rust/Python golden gate.
public enum FPPScenarioConstants {
    public static let msun = 1.989e33, rsun = 6.957e10, rearth = 6.371e8, gCGS = 6.674e-8
    /// Astronomical unit in cm — used by `FPPPriors`' bound-companion prior
    /// (`_lnprior_bound_TP`/`_lnprior_bound_EB` in `fpp_native.py`).
    public static let au = 1.496e13
}

public struct FPPStarParameters: Sendable {
    public let massSolar: Double, radiusSolar: Double, effectiveTemperature: Double
    public init(massSolar: Double, radiusSolar: Double, effectiveTemperature: Double = 5777) {
        self.massSolar = massSolar; self.radiusSolar = radiusSolar; self.effectiveTemperature = effectiveTemperature
    }
}

public struct FPPProbes: Sendable {
    public struct Sampler: Sendable { public let name: String, count: Int, head: [Double] }
    public struct Dispatch: Sendable { public let label: String, count: Int, head: [Double], finiteSum: Double, finiteCount: Int }
    public struct LnZ: Sendable { public let name: String, values: [Double] }
    public private(set) var samplers = [Sampler](), dispatches = [Dispatch](), lnz = [LnZ]()
    public init() {}
    public mutating func sampler(_ name: String, _ values: [Double]) { samplers.append(.init(name: name, count: values.count, head: Array(values.prefix(8)))) }
    public mutating func dispatched(_ label: String, _ values: [Double]) { var sum = 0.0; var count = 0; for v in values where v.isFinite { sum += v; count += 1 }; dispatches.append(.init(label: label, count: values.count, head: Array(values.prefix(8)), finiteSum: sum, finiteCount: count)) }
    public mutating func recordLnZ(_ name: String, _ values: [Double]) { lnz.append(.init(name: name, values: values)) }
}

private func masked(_ values: [Double], _ mask: [Bool]) -> [Double] { zip(values, mask).compactMap { $0.1 ? $0.0 : nil } }
private func restored(_ values: [Double], mask: [Bool]) -> [Double] { var out = Array(repeating: -Double.infinity, count: mask.count); var index = 0; for i in mask.indices where mask[i] { out[i] = values[index]; index += 1 }; return out }
/// `priorTerm` is `_lnprior_companion`'s per-draw N-length array (PTP/PEB)
/// or an all-zero N-length array (TTP/TEB/NTP/NEB, upstream's own
/// `lnprior_companion=0.0` no-op default) — folded into the SAME
/// Monte-Carlo average as the likelihood (`exp(lnL + priorTerm + 600)`),
/// matching upstream's `lnZ_PTP`/`lnZ_PEB`, not a separate additive term
/// on lnZ. Zero addition is a bit-exact no-op for finite `values` (`x +
/// 0.0 == x`), so TTP/TEB/NTP/NEB output is unaffected.
private func lnzFromLNL(_ values: [Double], priorTerm: [Double]) -> Double {
    let z = FPPNumpy.mean(zip(values, priorTerm).map { FPPNumpy.nanToNum(exp($0.0 + $0.1 + 600)) })
    return z > 0 ? log(z) : -Double.infinity
}
private func params(ks: [Double], periods: [Double], inclinations: [Double], axes: [Double], eccentricities: [Double], arguments: [Double], companions: [Double], host: Bool) -> FPPBatchParameters {
    FPPBatchParameters(radiusRatios: ks, periods: periods, inclinationsRadians: inclinations, semiMajorAxes: axes, eccentricities: eccentricities, argumentsRadians: arguments, companionFluxRatios: companions, companionIsHost: host)
}

public enum FPPScenarios {
    private static let pi = Double.pi

    public static func lnzTTP(rng: inout FPPRNG, probes: inout FPPProbes, time: [Double], flux: [Double], sigma: Double, period: Double, star: FPPStarParameters, count n: Int, u1: Double, u2: Double, companionFluxRatio: [Double], companionIsHost: Bool, lnpriorCompanion: [Double] = [], label: String? = nil) -> Double {
        let priorArr = lnpriorCompanion.isEmpty ? Array(repeating: 0.0, count: n) : lnpriorCompanion
        let ms = star.massSolar, rs = star.radiusSolar, periodSeconds = period * 86400
        let aCM = FPPNumpy.libmPow(FPPScenarioConstants.gCGS * ms * FPPScenarioConstants.msun / (4 * FPPNumpy.libmPow(pi, 2)) * (periodSeconds * periodSeconds), 1 / 3)
        let aRS = aCM / (rs * FPPScenarioConstants.rsun)
        let radii = FPPPriors.planetRadius(&rng, count: n, stellarMass: ms); probes.sampler("_sample_rp", radii)
        let inclinations = FPPPriors.inclinationDegrees(&rng, count: n); probes.sampler("_sample_inc", inclinations)
        let eccentricities = FPPPriors.eccentricity(&rng, count: n, planet: true, period: period); probes.sampler("_sample_ecc", eccentricities)
        let arguments = FPPPriors.argumentOfPeriastronDegrees(&rng, count: n); probes.sampler("_sample_w", arguments)
        let ks = radii.map { $0 * FPPScenarioConstants.rearth / (rs * FPPScenarioConstants.rsun) }
        let incRad = inclinations.map { $0 * pi / 180 }, argRad = arguments.map { (90 - $0) * pi / 180 }
        var mask = Array(repeating: false, count: n)
        for j in 0..<n {
            let eCorr = (1 + eccentricities[j] * sin(arguments[j] * pi / 180)) / (1 - eccentricities[j] * eccentricities[j])
            let ptra = (radii[j] * FPPScenarioConstants.rearth + rs * FPPScenarioConstants.rsun) / aCM * eCorr
            let incMin = ptra <= 1 ? acos(ptra) : pi / 2
            let collision = (radii[j] * FPPScenarioConstants.rearth + rs * FPPScenarioConstants.rsun) > aCM * (1 - eccentricities[j])
            mask[j] = incRad[j] >= incMin && !collision
        }
        guard mask.contains(true) else { if let label { probes.recordLnZ(label, [-Double.infinity]) }; return -Double.infinity }
        let ksM = masked(ks, mask), incM = masked(incRad, mask), eccM = masked(eccentricities, mask), argM = masked(argRad, mask), compM = masked(companionFluxRatio, mask)
        let m = ksM.count
        let result = FPPKernels.transitLogLikelihood(time: time, flux: flux, sigma: sigma, u1: u1, u2: u2, parameters: params(ks: ksM, periods: Array(repeating: period, count: m), inclinations: incM, axes: Array(repeating: aRS, count: m), eccentricities: eccM, arguments: argM, companions: compM, host: companionIsHost))
        probes.dispatched("transit", result)
        let value = lnzFromLNL(restored(result, mask: mask), priorTerm: priorArr)
        if let label { probes.recordLnZ(label, [value]) }; return value
    }

    public static func lnzTEB(rng: inout FPPRNG, probes: inout FPPProbes, time: [Double], flux: [Double], sigma: Double, period: Double, star: FPPStarParameters, count n: Int, u1: Double, u2: Double, companionFluxRatio: [Double], companionIsHost: Bool, lnpriorCompanion: [Double] = [], label: String? = nil) -> (Double, Double) {
        let priorArr = lnpriorCompanion.isEmpty ? Array(repeating: 0.0, count: n) : lnpriorCompanion
        let ms = star.massSolar, rs = star.radiusSolar
        let inclinations = FPPPriors.inclinationDegrees(&rng, count: n); probes.sampler("_sample_inc", inclinations)
        let qs = FPPPriors.massRatio(&rng, count: n, stellarMass: ms); probes.sampler("_sample_q", qs)
        let eccentricities = FPPPriors.eccentricity(&rng, count: n, planet: false, period: period); probes.sampler("_sample_ecc", eccentricities)
        let arguments = FPPPriors.argumentOfPeriastronDegrees(&rng, count: n); probes.sampler("_sample_w", arguments)
        let masses = qs.map { $0 * ms }, stellar = FPPStellar.relations(masses: masses), binaryFlux = FPPStellar.fluxRelation(masses: masses), mainFlux = FPPStellar.fluxRelation(masses: [ms])[0]
        let ratios = binaryFlux.map { $0 / ($0 + mainFlux) }, incRad = inclinations.map { $0 * pi / 180 }, argRad = arguments.map { (90 - $0) * pi / 180 }
        var axes = [Double](), twinAxes = [Double](); axes.reserveCapacity(n); twinAxes.reserveCapacity(n)
        for j in 0..<n { let p = period * 86400, pt = 2 * period * 86400; axes.append(FPPNumpy.libmPow(FPPScenarioConstants.gCGS * (ms + masses[j]) * FPPScenarioConstants.msun / (4 * FPPNumpy.libmPow(pi, 2)) * p * p, 1 / 3)); twinAxes.append(FPPNumpy.libmPow(FPPScenarioConstants.gCGS * (ms + masses[j]) * FPPScenarioConstants.msun / (4 * FPPNumpy.libmPow(pi, 2)) * pt * pt, 1 / 3)) }
        let radiiRatio = stellar.radii.map { $0 / rs }, axesRatio = axes.map { $0 / (rs * FPPScenarioConstants.rsun) }, twinRatio = twinAxes.map { $0 / (rs * FPPScenarioConstants.rsun) }
        var ebMask = Array(repeating: false, count: n), twinMask = ebMask
        for j in 0..<n { let eCorr = (1 + eccentricities[j] * sin(arguments[j] * pi / 180)) / (1 - eccentricities[j] * eccentricities[j]); let ptra = (stellar.radii[j] * FPPScenarioConstants.rsun + rs * FPPScenarioConstants.rsun) / axes[j] * eCorr; let ptraTwin = (stellar.radii[j] * FPPScenarioConstants.rsun + rs * FPPScenarioConstants.rsun) / twinAxes[j] * eCorr; let inc = ptra <= 1 ? acos(ptra) : pi / 2; let incTwin = ptraTwin <= 1 ? acos(ptraTwin) : pi / 2; ebMask[j] = incRad[j] >= inc && (stellar.radii[j] * FPPScenarioConstants.rsun + rs * FPPScenarioConstants.rsun) <= axes[j] * (1 - eccentricities[j]) && qs[j] < 0.95; twinMask[j] = incRad[j] >= incTwin && 2 * rs * FPPScenarioConstants.rsun <= twinAxes[j] * (1 - eccentricities[j]) && qs[j] >= 0.95 }
        func evaluate(_ mask: [Bool], _ periods: [Double], _ axes: [Double], _ label: String) -> [Double] { guard mask.contains(true) else { return [] }; let k = masked(radiiRatio, mask); let out = FPPKernels.eclipsingBinaryLogLikelihood(time: time, flux: flux, sigma: sigma, u1: u1, u2: u2, binaryFluxRatios: masked(ratios, mask), parameters: params(ks: k, periods: masked(periods, mask), inclinations: masked(incRad, mask), axes: masked(axes, mask), eccentricities: masked(eccentricities, mask), arguments: masked(argRad, mask), companions: masked(companionFluxRatio, mask), host: companionIsHost)); probes.dispatched(label, out); return out }
        let eb = evaluate(ebMask, Array(repeating: period, count: n), axesRatio, "EB"), twin = evaluate(twinMask, Array(repeating: 2 * period, count: n), twinRatio, "twin EB")
        let ebFull = restored(eb, mask: ebMask), twinFull = restored(twin, mask: twinMask)
        let values = (lnzFromLNL(ebFull, priorTerm: priorArr), lnzFromLNL(twinFull, priorTerm: priorArr)); if let label { probes.recordLnZ(label, [values.0, values.1]) }; return values
    }

    /// Mirrors `triceratops/marginal_likelihoods.py:lnZ_PTP`: the physically
    /// sampled companion (`masses`, via the binary mass-ratio prior) only
    /// sets the per-draw flux-dilution ratio fed to the TTP likelihood — the
    /// transit geometry and limb-darkening coefficients stay the TARGET
    /// star's (`u1`/`u2` passed straight through, unchanged), matching
    /// upstream's `companion_is_host=False` (the target, not the companion,
    /// is modeled as the transit host; the companion is third light only).
    /// Upstream additionally folds a `lnprior_companion` term (bound-
    /// companion probability from `parallax` + Δmag + the Moe & Di Stefano
    /// mass-ratio prior, `lnprior_bound_TP`) into `Z` — see
    /// `FPPPriors.lnpriorCompanion`.
    public static func lnzPTP(rng: inout FPPRNG, probes: inout FPPProbes, time: [Double], flux: [Double], sigma: Double, period: Double, star: FPPStarParameters, count n: Int, u1: Double, u2: Double, parallax: Double) -> Double {
        let qs = FPPPriors.companionMassRatio(&rng, count: n, stellarMass: star.massSolar); probes.sampler("_sample_q_companion", qs)
        let masses = qs.map { $0 * star.massSolar }, companionFlux = FPPStellar.fluxRelation(masses: masses), mainFlux = FPPStellar.fluxRelation(masses: [star.massSolar])[0]
        let ratios = companionFlux.map { $0 / ($0 + mainFlux) }
        let lnPriorCompanion = FPPPriors.lnpriorCompanion(massSolar: star.massSolar, parallax: parallax, fluxRatiosCompanion: ratios, scenario: .transiting)
        return lnzTTP(rng: &rng, probes: &probes, time: time, flux: flux, sigma: sigma, period: period, star: star, count: n, u1: u1, u2: u2, companionFluxRatio: ratios, companionIsHost: false, lnpriorCompanion: lnPriorCompanion, label: "_lnZ_TTP")
    }

    /// Mirrors `triceratops/marginal_likelihoods.py:lnZ_PEB` — same
    /// convention as `lnzPTP` above (see its doc comment, including the
    /// `lnprior_companion` term, PEB convention via
    /// `FPPPriors.lnpriorCompanion(..., scenario: .eclipsing)`). This also
    /// carries the matching `_lnZ_TEB` fix: the per-draw companion flux
    /// ratio is now actually wired into the eclipse likelihood's dilution
    /// term instead of being discarded (`companionIsHost: false`, target
    /// `u1`/`u2`).
    public static func lnzPEB(rng: inout FPPRNG, probes: inout FPPProbes, time: [Double], flux: [Double], sigma: Double, period: Double, star: FPPStarParameters, count n: Int, u1: Double, u2: Double, parallax: Double) -> (Double, Double) {
        let qs = FPPPriors.companionMassRatio(
            &rng, count: n, stellarMass: star.massSolar
        )
        probes.sampler("_sample_q_companion", qs)
        let masses = qs.map { $0 * star.massSolar }
        let companionFlux = FPPStellar.fluxRelation(masses: masses)
        let mainFlux = FPPStellar.fluxRelation(masses: [star.massSolar])[0]
        let ratios = companionFlux.map { $0 / ($0 + mainFlux) }
        let lnPriorCompanion = FPPPriors.lnpriorCompanion(massSolar: star.massSolar, parallax: parallax, fluxRatiosCompanion: ratios, scenario: .eclipsing)
        return lnzTEB(
            rng: &rng, probes: &probes, time: time, flux: flux, sigma: sigma,
            period: period, star: star, count: n, u1: u1, u2: u2,
            companionFluxRatio: ratios, companionIsHost: false,
            lnpriorCompanion: lnPriorCompanion, label: "_lnZ_TEB"
        )
    }

    public static func sequentialFiniteSum(_ values: [Double]) -> (Double, Int) { var sum = 0.0, count = 0; for value in values where value.isFinite { sum += value; count += 1 }; return (sum, count) }
    public static func npSum(_ values: [Double]) -> Double { FPPNumpy.pairwiseSum(values) }
}
