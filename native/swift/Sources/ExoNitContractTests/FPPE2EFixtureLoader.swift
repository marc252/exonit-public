import ExoNitCore
import Foundation

/// Loader for the Rust `fixtures/fpp/inputs/*.json` format.  Values in
/// `time_hex` and `flux_hex` are little-endian IEEE-754 bit patterns, so the
/// loader never round-trips them through decimal JSON numbers.
enum FPPE2EFixtureLoader {
    private struct RawInput: Decodable {
        let ticID: JSONValue
        let period: Double
        let t0: Double
        let duration: Double
        let depthPPM: Double
        let timeHex: [String]
        let fluxHex: [String]
        let nearby: [String: JSONValue]
        let trilegalCSV: JSONValue?

        enum CodingKeys: String, CodingKey {
            case ticID = "tic_id", period, t0, duration
            case depthPPM = "depth_ppm", timeHex = "time_hex", fluxHex = "flux_hex"
            case nearby, trilegalCSV = "trilegal_csv"
        }
    }

    private static func scalarString(_ value: JSONValue) throws -> String {
        switch value {
        case let .string(value): return value
        case let .integer(value): return String(value)
        case let .number(value): return String(value)
        default: throw NSError(domain: "FPPE2EFixtureLoader", code: 1)
        }
    }

    private static func doubleBits(_ text: String) throws -> Double {
        var bits: UInt64 = 0
        for (index, offset) in stride(from: 0, to: text.count, by: 2).enumerated() {
            let start = text.index(text.startIndex, offsetBy: offset)
            let end = text.index(start, offsetBy: 2)
            guard let byte = UInt64(String(text[start..<end]), radix: 16) else {
                throw NSError(domain: "FPPE2EFixtureLoader", code: 2)
            }
            bits |= byte << (8 * index)
        }
        return Double(bitPattern: bits)
    }

    static func loadInput(at url: URL, trilegalDirectory: URL? = nil) throws -> FppInputs {
        let raw = try JSONDecoder().decode(RawInput.self, from: Data(contentsOf: url))
        let csv: String?
        if let value = raw.trilegalCSV {
            switch value {
            case .null: csv = nil
            default:
                let name = try scalarString(value)
                let base = trilegalDirectory ?? url.deletingLastPathComponent()
                csv = try String(contentsOf: base.appendingPathComponent(name), encoding: .utf8)
            }
        } else {
            csv = nil
        }
        return FppInputs(
            ticID: try scalarString(raw.ticID), periodDays: raw.period,
            t0: raw.t0, durationDays: raw.duration, depthPPM: raw.depthPPM,
            time: try raw.timeHex.map(doubleBits), flux: try raw.fluxHex.map(doubleBits),
            nearby: raw.nearby, trilegalCSV: csv

        )
    }
}
private func runFPPE2E(_ stem: String) throws {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures/fpp")
    let input = try FPPE2EFixtureLoader.loadInput(
        at: root.appendingPathComponent("inputs/\(stem).json")
    )
    let document = try JSONSerialization.jsonObject(
        with: Data(contentsOf: root.appendingPathComponent("fpp_native_golden.json"))
    ) as! [String: Any]
    let golden = document[stem] as! [String: Any]
    let seed = UInt32(golden["seed"] as! Int)
    var probes = FPPCalcProbes()
    guard let result = FPPCalculation.calcFPP(
        input, draws: 100_000, seed: seed, probes: &probes
    ) else {
        throw NSError(domain: "FPPE2E", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "\(stem): calcFPP returned nil"
        ])
    }
    let expected = golden["result"] as! [String: Any]
    try checkpointChecks(golden["checkpoints"] as! [String: Any], probes)
    func bits(_ text: String) -> Double {
        var value: UInt64 = 0
        for (index, offset) in stride(from: 0, to: text.count, by: 2).enumerated() {
            let start = text.index(text.startIndex, offsetBy: offset)
            let end = text.index(start, offsetBy: 2)
            value |= UInt64(String(text[start..<end]), radix: 16)! << (8 * index)
        }
        return Double(bitPattern: value)
    }
    let expectedFPP = bits(expected["fpp"] as! String)
    let expectedNFPP = bits(expected["nfpp"] as! String)
    let expectedVerdict = expected["verdict"] as! String
    guard abs(result.fpp - expectedFPP) <= 1e-12,
          abs(result.nfpp - expectedNFPP) <= 1e-12,
          result.verdict.rawValue == expectedVerdict else {
        throw NSError(domain: "FPPE2E", code: 2, userInfo: [
            NSLocalizedDescriptionKey: "\(stem): got fpp=\(result.fpp) nfpp=\(result.nfpp) verdict=\(result.verdict.rawValue); expected \(expectedFPP) \(expectedNFPP) \(expectedVerdict)"
        ])
    }
}

func testFPPE2E10756814() throws {
    for stem in [
        "10756814", "10756814_trilegal", "10756829", "10756921",
        "10757100", "10757120", "10757120_noneigh", "10757308",
        "10757863", "10758443", "10758466", "10758647",
    ] {
        do {
            try runFPPE2E(stem)
        } catch {
            throw NSError(domain: "FPPE2E", code: 99, userInfo: [
                NSLocalizedDescriptionKey: "[\(stem)] \(error)"
            ])
        }
    }
}


private func checkpointBits(_ text: String) -> Double {
    var value: UInt64 = 0
    for (index, offset) in stride(from: 0, to: text.count, by: 2).enumerated() {
        let start = text.index(text.startIndex, offsetBy: offset)
        let end = text.index(start, offsetBy: 2)
        value |= UInt64(String(text[start..<end]), radix: 16)! << (8 * index)
    }
    return Double(bitPattern: value)
}
private func checkpointBits(_ value: Any) -> Double { checkpointBits(value as! String) }
private func checkpointCount(_ got: Int, _ want: Any, _ label: String) throws {
    guard let want = want as? Int, got == want else { throw NSError(domain: "FPPE2E", code: 10, userInfo: [NSLocalizedDescriptionKey: "\(label): count mismatch"]) }
}
private func checkpointString(_ got: String, _ want: Any, _ label: String) throws {
    guard let want = want as? String, got == want else { throw NSError(domain: "FPPE2E", code: 11, userInfo: [NSLocalizedDescriptionKey: "\(label): name mismatch"]) }
}
private func checkpointFloat(
    _ got: Double, _ want: Double, _ label: String, maxULP: UInt64 = 16
) throws {
    let a = got.bitPattern, b = want.bitPattern
    let distance = a > b ? a - b : b - a
    // Linux/glibc goldens can differ slightly from Darwin libm after the
    // transit kernel's chained transcendental operations.
    guard distance <= maxULP else { throw NSError(domain: "FPPE2E", code: 12, userInfo: [NSLocalizedDescriptionKey: "\(label): ULP mismatch (\(distance) ULP)"]) }
}
private func checkpointArray(_ got: [Double], _ want: [Any], _ label: String, maxULP: UInt64 = 16) throws {
    try checkpointCount(got.count, want.count, "\(label).count")
    for (i, value) in got.enumerated() { try checkpointFloat(value, checkpointBits(want[i]), "\(label)[\(i)]", maxULP: maxULP) }
}
private func checkpointChecks(_ raw: [String: Any], _ probes: FPPCalcProbes) throws {
    let phase = raw["phase_fold"] as! [String: Any], folded = probes.phaseFold!
    try checkpointCount(folded.time.count, phase["n"]!, "phase_fold.n")
    try checkpointArray(folded.time, phase["time_pf"] as! [Any], "phase_fold.time_pf")
    try checkpointArray(folded.flux, phase["flux_pf"] as! [Any], "phase_fold.flux_pf")
    let ratios = raw["flux_ratios"] as! [[String: Any]]
    try checkpointCount(probes.fluxRatios.count, ratios.count, "flux_ratios")
    for (i, expected) in ratios.enumerated() { let got = probes.fluxRatios[i]; try checkpointString(got.id, expected["ID"]!, "flux_ratios[\(i)].ID"); try checkpointFloat(got.fluxRatio, checkpointBits(expected["fluxratio"]!), "flux_ratios[\(i)].fluxratio"); try checkpointFloat(got.transitDepth, checkpointBits(expected["tdepth"]!), "flux_ratios[\(i)].tdepth") }
    let renorm = raw["renorm"] as! [[String: Any]]
    try checkpointCount(probes.renorm.count, renorm.count, "renorm")
    for (i, expected) in renorm.enumerated() { let got = probes.renorm[i]; try checkpointFloat(got.fluxRatio, checkpointBits(expected["fluxratio"]!), "renorm[\(i)].fluxratio"); try checkpointFloat(got.sigma, checkpointBits(expected["sigma_in"]!), "renorm[\(i)].sigma_in"); try checkpointArray(got.fluxHead, expected["flux_head"] as! [Any], "renorm[\(i)].flux_head") }
    let samplers = raw["samplers"] as! [[String: Any]]
    try checkpointCount(probes.scen.samplers.count, samplers.count, "samplers")
    for (i, expected) in samplers.enumerated() { let got = probes.scen.samplers[i]; try checkpointString(got.name, expected["fn"]!, "samplers[\(i)].fn"); try checkpointCount(got.count, expected["n"]!, "samplers[\(i)].n"); try checkpointArray(got.head, expected["head"] as! [Any], "samplers[\(i)].head") }
    let dispatch = raw["dispatch"] as! [[String: Any]]
    try checkpointCount(probes.scen.dispatches.count, dispatch.count, "dispatch")
    for (i, expected) in dispatch.enumerated() { let got = probes.scen.dispatches[i]; try checkpointString(got.label, expected["label"]!, "dispatch[\(i)].label"); try checkpointCount(got.count, expected["n"]!, "dispatch[\(i)].n"); try checkpointCount(got.finiteCount, expected["n_finite"]!, "dispatch[\(i)].n_finite"); try checkpointFloat(got.finiteSum, checkpointBits(expected["sum_finite"]!), "dispatch[\(i)].sum_finite", maxULP: 64); try checkpointArray(got.head, expected["head"] as! [Any], "dispatch[\(i)].head") }
    let lnz = raw["lnz"] as! [[String: Any]]
    try checkpointCount(probes.scen.lnz.count, lnz.count, "lnz")
    // PEB's "_lnZ_TEB" entry (companion_is_host=false, nonzero per-draw
    // dilution) exercises, for the first time, a mean(exp(lnL + 600))
    // reduction over ~1e5 draws whose per-element lnL already carries the
    // usual few-ULP Linux/glibc-vs-Darwin-libm drift (see checkpointFloat's
    // comment) — that drift doesn't cancel across the reduction, so the
    // final scalar can land up to ~130 ULP off (observed on fixture
    // 10758466) even though every underlying dispatch checkpoint (n,
    // n_finite, sum_finite, head) matches within their own tolerances. This
    // branch was previously dead code (pre-fix PEB always zeroed the
    // dilution, so lnZ_eb never took real values), so no prior golden ever
    // exercised it. 256 keeps a real regression (which would land many
    // orders of magnitude off, as seen comparing against the pre-fix
    // golden) easily caught while tolerating this reduction noise.
    for (i, expected) in lnz.enumerated() { let got = probes.scen.lnz[i]; try checkpointString(got.name, expected["fn"]!, "lnz[\(i)].fn"); try checkpointArray(got.values, expected["values"] as! [Any], "lnz[\(i)].values", maxULP: 256) }
}


func testFPPNearbyGalacticLatitudeParity() throws {
    let star: JSONValue = .object(["ID": .string("1")])
    let missing = FPPInputs.parseNearby(["stars": .array([star])], ticID: "1")
    try expect(missing?.galacticLatitude == 0.0, "missing gal_b defaults to zero")
    let invalid = FPPInputs.parseNearby(
        ["stars": .array([star]), "gal_b": .null], ticID: "1"
    )
    try expect(invalid?.galacticLatitude.isNaN == true, "null gal_b maps to NaN")
}
