import ExoNitCore
import Foundation

private struct FPPKernelPriorFailure: Error, CustomStringConvertible {
    let description: String
}

private func fppKPJSON(_ name: String) throws -> [String: Any] {
    let url = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures/fpp/").appendingPathComponent(name)
    return try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
}

private func fppKPBits(_ value: Any) -> Double {
    let text = value as! String
    var bits: UInt64 = 0
    for (index, offset) in stride(from: 0, to: text.count, by: 2).enumerated() {
        let start = text.index(text.startIndex, offsetBy: offset)
        let end = text.index(start, offsetBy: 2)
        bits |= UInt64(String(text[start..<end]), radix: 16)! << (8 * index)
    }
    return Double(bitPattern: bits)
}

private func fppKPArray(_ value: Any) -> [Double] { (value as! [Any]).map(fppKPBits) }

private func fppKPCheck(_ actual: [Double], _ expected: [Double], _ label: String) throws {
    guard actual.count == expected.count else {
        throw FPPKernelPriorFailure(description: label + " length")
    }
    for (index, pair) in zip(actual, expected).enumerated() {
        let distance = pair.0.bitPattern > pair.1.bitPattern
            ? pair.0.bitPattern - pair.1.bitPattern : pair.1.bitPattern - pair.0.bitPattern
        if distance > 4 {
            throw FPPKernelPriorFailure(description: label + "[" + String(index) + "]")
        }
    }
}

func testFPPPriorGoldenParity() throws {
    let samplers = try fppKPJSON("rng_golden.json")["samplers"] as! [String: Any]
    func expected(_ name: String) -> [Double] { fppKPArray(samplers[name]!) }

    var rng = FPPRNG(seed: 42); try fppKPCheck(FPPPriors.planetRadius(&rng, count: 64, stellarMass: 1.20), expected("sample_rp_ms120"), "rp_ms120")
    rng = FPPRNG(seed: 42); try fppKPCheck(FPPPriors.planetRadius(&rng, count: 64, stellarMass: 0.40), expected("sample_rp_ms040"), "rp_ms040")
    rng = FPPRNG(seed: 42); try fppKPCheck(FPPPriors.inclinationDegrees(&rng, count: 64), expected("sample_inc"), "inc")
    rng = FPPRNG(seed: 42); try fppKPCheck(FPPPriors.eccentricity(&rng, count: 64, planet: true, period: 5.0), expected("sample_ecc_planet"), "ecc_planet")
    rng = FPPRNG(seed: 42); try fppKPCheck(FPPPriors.eccentricity(&rng, count: 64, planet: false, period: 5.0), expected("sample_ecc_bin_p5"), "ecc_bin_p5")
    rng = FPPRNG(seed: 42); try fppKPCheck(FPPPriors.eccentricity(&rng, count: 64, planet: false, period: 20.0), expected("sample_ecc_bin_p20"), "ecc_bin_p20")
    rng = FPPRNG(seed: 42); try fppKPCheck(FPPPriors.argumentOfPeriastronDegrees(&rng, count: 64), expected("sample_w"), "w")
    rng = FPPRNG(seed: 42); try fppKPCheck(FPPPriors.massRatio(&rng, count: 64, stellarMass: 1.20), expected("sample_q_ms120"), "q_ms120")
    rng = FPPRNG(seed: 42); try fppKPCheck(FPPPriors.massRatio(&rng, count: 64, stellarMass: 0.50), expected("sample_q_ms050"), "q_ms050")
    rng = FPPRNG(seed: 42); try fppKPCheck(FPPPriors.massRatio(&rng, count: 64, stellarMass: 0.25), expected("sample_q_ms025"), "q_ms025")
    rng = FPPRNG(seed: 42); try fppKPCheck(FPPPriors.companionMassRatio(&rng, count: 64, stellarMass: 1.20), expected("sample_q_comp_ms120"), "q_comp_ms120")
    rng = FPPRNG(seed: 42); try fppKPCheck(FPPPriors.companionMassRatio(&rng, count: 64, stellarMass: 0.25), expected("sample_q_comp_ms025"), "q_comp_ms025")
}

func testFPPKernelGoldenParity() throws {
    let golden = try fppKPJSON("transit_golden.json")
    let kernels = golden["kernels"] as! [String: Any]
    let time = fppKPArray(kernels["time_pf"]!)
    let flux = fppKPArray(kernels["flux_pf"]!)
    for raw in kernels["cases"] as! [[String: Any]] {
        let parameters = FPPBatchParameters(
            radiusRatios: fppKPArray(raw["ks"]!), periods: fppKPArray(raw["p_orbs"]!),
            inclinationsRadians: fppKPArray(raw["incs_rad"]!), semiMajorAxes: fppKPArray(raw["a_rs"]!),
            eccentricities: fppKPArray(raw["eccs"]!), argumentsRadians: fppKPArray(raw["ws_rad"]!),
            companionFluxRatios: fppKPArray(raw["comp_fr"]!), companionIsHost: raw["is_host"] as! Bool
        )
        let sigma = fppKPBits(raw["sigma"]!), u1 = fppKPBits(raw["u1"]!), u2 = fppKPBits(raw["u2"]!)
        let actual: [Double]
        if (raw["kind"] as! String) == "transit" {
            actual = FPPKernels.transitLogLikelihood(time: time, flux: flux, sigma: sigma, u1: u1, u2: u2, parameters: parameters)
        } else {
            actual = FPPKernels.eclipsingBinaryLogLikelihood(time: time, flux: flux, sigma: sigma, u1: u1, u2: u2, binaryFluxRatios: fppKPArray(raw["eb_fluxratios"]!), parameters: parameters)
        }
        try fppKPCheck(actual, fppKPArray(raw["lnl"]!), raw["label"] as! String)
    }
}
