import ExoNitCore
import Foundation

// Separate from the existing runner so the FPP port can be enabled by the
// canary without changing the production test list.
private struct FPPGoldenFailure: Error, CustomStringConvertible {
    let description: String
}

private func fppJSON(_ name: String) throws -> [String: Any] {
    let source = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures/fpp/").appendingPathComponent(name)
    return try JSONSerialization.jsonObject(with: Data(contentsOf: source)) as! [String: Any]
}

private func fppBits(_ value: Any) -> Double {
    let text = value as! String
    var bits: UInt64 = 0
    for (index, offset) in stride(from: 0, to: text.count, by: 2).enumerated() {
        let start = text.index(text.startIndex, offsetBy: offset)
        let end = text.index(start, offsetBy: 2)
        bits |= UInt64(String(text[start..<end]), radix: 16)! << (8 * index)
    }
    return Double(bitPattern: bits)
}

private func fppArray(_ value: Any) -> [Double] { (value as! [Any]).map(fppBits) }

private func fppExpectBits(_ got: Double, _ want: Double, _ label: String) throws {
    let ulpDistance = got.bitPattern > want.bitPattern
        ? got.bitPattern - want.bitPattern : want.bitPattern - got.bitPattern
    guard ulpDistance <= 4 else {
        let message = label + ": got " + String(got.bitPattern, radix: 16)
            + ", want " + String(want.bitPattern, radix: 16)
        throw FPPGoldenFailure(description: message)
    }
}

/// Golden checks for Kepler/anomaly, ring integration and complete light curves.
func testFPPTransitGoldenParity() throws {
    let golden = try fppJSON("transit_golden.json")
    for (index, raw) in (golden["kepler"] as! [[String: Any]]).enumerated() {
        try fppExpectBits(
            FPPTransit.keplerEquation(meanAnomaly: fppBits(raw["m"]!), eccentricity: fppBits(raw["ecc"]!)),
            fppBits(raw["e_out"]!), "kepler[" + String(index) + "]"
        )
        try fppExpectBits(
            FPPTransit.trueAnomaly(meanAnomaly: fppBits(raw["m"]!), eccentricity: fppBits(raw["ecc"]!)),
            fppBits(raw["nu_out"]!), "true_anomaly[" + String(index) + "]"
        )
    }
    for (index, raw) in (golden["ring"] as! [[String: Any]]).enumerated() {
        try fppExpectBits(
            FPPTransit.ringFlux(z: fppBits(raw["z"]!), planetRadiusRatio: fppBits(raw["p"]!), u1: fppBits(raw["u1"]!), u2: fppBits(raw["u2"]!)),
            fppBits(raw["out"]!), "ring[" + String(index) + "]"
        )
    }
    for (index, raw) in (golden["lightcurves"] as! [[String: Any]]).enumerated() {
        let actual = FPPTransit.lightCurve(
            time: fppArray(raw["time"]!), planetRadiusRatio: fppBits(raw["k"]!), t0: 0.0,
            period: fppBits(raw["period"]!), aOverRStar: fppBits(raw["a_rs"]!),
            inclinationRadians: fppBits(raw["inc_rad"]!), eccentricity: fppBits(raw["ecc"]!),
            argumentPeriastronRadians: fppBits(raw["w_rad"]!), u1: fppBits(raw["u1"]!), u2: fppBits(raw["u2"]!)
        )
        for (point, pair) in zip(actual, fppArray(raw["flux"]!)).enumerated() {
            try fppExpectBits(pair.0, pair.1, "lightcurves[" + String(index) + "][" + String(point) + "]")
        }
    }
}

/// Golden checks for embedded FITPACK coefficients, stellar relations and
/// nearest-neighbour LDC lookup.
func testFPPStellarGoldenParity() throws {
    let golden = try fppJSON("spline_golden.json")
    let relations = golden["relations"] as! [String: Any]
    let masses = fppArray(relations["masses"]!)
    let actual = FPPStellar.relations(masses: masses)
    for (index, want) in fppArray(relations["radii"]!).enumerated() {
        try fppExpectBits(actual.radii[index], want, "relations.radii[" + String(index) + "]")
    }
    for (index, want) in fppArray(relations["teffs"]!).enumerated() {
        try fppExpectBits(actual.temperatures[index], want, "relations.teffs[" + String(index) + "]")
    }
    let flux = FPPStellar.fluxRelation(masses: masses)
    for (index, want) in fppArray(relations["flux"]!).enumerated() {
        try fppExpectBits(flux[index], want, "relations.flux[" + String(index) + "]")
    }
    for (index, raw) in (golden["ldc"] as! [[String: Any]]).enumerated() {
        let got = FPPStellar.limbDarkening(temperature: fppBits(raw["teff"]!), logg: fppBits(raw["logg"]!))
        try fppExpectBits(got.u1, fppBits(raw["u1"]!), "ldc[" + String(index) + "].u1")
        try fppExpectBits(got.u2, fppBits(raw["u2"]!), "ldc[" + String(index) + "].u2")
    }
}
