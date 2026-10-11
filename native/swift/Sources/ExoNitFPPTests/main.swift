import ExoNitCore
import Foundation

struct Streams: Decodable {
    let rand16: [String]
    let gauss16_rand1_gauss8: [String]
    let stdexp8: [String]
    let stdgamma_0867_8: [String]
    let stdgamma_3030_8: [String]
    let beta_scipy_16: [String]
    let power_scipy_02_8: [String]
    let power_scipy_06_8: [String]
}
struct Golden: Decodable {
    let rng: [String: Streams]
    let tic_seeds: [String: Seed]
    struct Seed: Decodable { let key: String; let seed: UInt32 }
}

func bits(_ hex: String) -> Double {
    var value: UInt64 = 0
    for byte in stride(from: 0, to: hex.count, by: 2) {
        let start = hex.index(hex.startIndex, offsetBy: byte)
        let end = hex.index(start, offsetBy: 2)
        value |= UInt64(UInt8(hex[start..<end], radix: 16)!) << UInt64(byte / 2 * 8)
    }
    return Double(bitPattern: value)
}
func check(_ got: Double, _ expected: String, _ label: String) {
    precondition(got.bitPattern == bits(expected).bitPattern, "\(label): \(got) != \(bits(expected))")
}
func stream(_ expected: [String], _ label: String, _ body: (inout FPPRNG, Int) -> Double, seed: UInt32) {
    var rng = FPPRNG(seed: seed)
    for (index, hex) in expected.enumerated() { check(body(&rng, index), hex, "\(label)[\(index)]") }
}

let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Fixtures/fpp_rng_golden.json")
let golden = try JSONDecoder().decode(Golden.self, from: Data(contentsOf: fixture))
for (seedString, streams) in golden.rng {
    let seed = UInt32(seedString)!
    stream(streams.rand16, "rand16", { r, _ in r.rand() }, seed: seed)
    stream(streams.gauss16_rand1_gauss8, "gauss", { r, i in i == 16 ? r.rand() : r.gauss() }, seed: seed)
    stream(streams.stdexp8, "stdexp", { r, _ in r.standardExponential() }, seed: seed)
    stream(streams.stdgamma_0867_8, "gamma<1", { r, _ in r.standardGamma(0.867) }, seed: seed)
    stream(streams.stdgamma_3030_8, "gamma>1", { r, _ in r.standardGamma(3.030) }, seed: seed)
    stream(streams.beta_scipy_16, "beta", { r, _ in r.beta(0.867, 3.030) }, seed: seed)
    stream(streams.power_scipy_02_8, "power.2", { r, _ in r.power(0.2) }, seed: seed)
    stream(streams.power_scipy_06_8, "power.6", { r, _ in r.power(0.6) }, seed: seed)
}
for seed in golden.tic_seeds.values {
    let parts = seed.key.split(separator: ":", maxSplits: 2).map(String.init)
    precondition(FPPRNG.ticSeedKey(ticID: parts[0], period: Double(parts[1])!, t0: Double(parts[2])!) == seed.key)
    precondition(FPPRNG.ticSeed(ticID: parts[0], period: Double(parts[1])!, t0: Double(parts[2])!) == seed.seed)
}
print("PASS FPP RNG golden parity (\(golden.rng.count) seeds, \(golden.tic_seeds.count) TIC seeds)")

let foldFixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Fixtures/fpp_fold_golden.json")
let fold = try JSONDecoder().decode(FoldGolden.self, from: Data(contentsOf: foldFixture))
let folded = FPPFold.phaseFold(fold.time, t0: fold.t0, period: fold.period)
for (i, value) in folded.enumerated() { precondition(value.bitPattern == fold.phase[i].bitPattern, "phase[\(i)]") }
let binned = FPPFold.bin(time: fold.time, flux: fold.flux, period: fold.period, t0: fold.t0, bins: fold.bins)
precondition(binned.time == fold.binnedTime && binned.flux == fold.binnedFlux, "fold bins")
precondition(FPPFold.renorm(flux: [0.9, 1.0], fluxError: 0.02, starFluxRatio: 0.8).flux == [0.875, 1.0], "renorm")
precondition(FPPNumpy.pairwiseSum([1, 2, 3, 4, 5, 6, 7]) == 28, "pairwise short")
precondition(
    FPPNumpy.pairwiseSum([1e16, 1, -1e16, 1, 1e16, 1, -1e16, 1]) == 0,
    "pairwise NumPy reduction tree"
)
precondition(FPPNumpy.mean([1, 2, 3, 4]) == 2.5, "mean")
precondition(FPPNumpy.median([3, 1, 2, 4]) == 2.5, "median")
print("PASS FPP fold/numpy golden parity")

let nearby: [String: JSONValue] = [
    "gal_b": .number(12),
    "stars": .array([
        .object(["ID": .string("other"), "ra": .number(10.001), "dec": .number(20), "mass": .number(1), "rad": .number(1), "Teff": .number(5700), "Tmag": .number(12), "plx": .number(2)]),
        .object(["ID": .integer(123), "ra": .number(10), "dec": .number(20), "mass": .number(1), "rad": .number(1), "Teff": .number(5700), "Tmag": .number(10), "plx": .number(3)])
    ])
]
var parsed = FPPInputs.parseNearby(nearby, ticID: "123")!
precondition(parsed.targetPromoted && parsed.stars.first!.id == "123", "nearby target promotion")
FPPInputs.computeFluxRatios(&parsed.stars, depthPPM: 1000)
precondition(abs(parsed.stars[0].fluxRatio - 0.865) < 0.01 && parsed.stars[0].transitDepth > 1000, "flux ratios")
let csv = "TESS logTe Mact logg\n10 3.76 1.0 4.4\n20 3.70 0.8 4.5\n30 3.70 0.8 4.5"
let trilegal = FPPInputs.loadTrilegal(csv, targetRA: 10, targetDec: 20, targetTmag: 10)
precondition(trilegal.count == 1 && trilegal[0].isTrilegal && trilegal[0].id == "TRILEGAL_0", "trilegal load/filter")
print("PASS FPP input parsing/flux-ratio contract")

var scenarioRNG = FPPRNG(seed: 0)
var scenarioProbes = FPPProbes()
_ = FPPScenarios.lnzTTP(
    rng: &scenarioRNG, probes: &scenarioProbes,
    time: [0.0, 0.1, 0.2, 0.3], flux: [1.0, 1.0, 1.0, 1.0],
    sigma: 0.01, period: 2.0,
    star: FPPStarParameters(massSolar: 1.0, radiusSolar: 1.0),
    count: 8, u1: 0.4, u2: 0.2,
    companionFluxRatio: Array(repeating: 0.0, count: 8), companionIsHost: false,
    label: "_lnZ_TTP"
)
precondition(scenarioProbes.samplers.count == 4 && scenarioProbes.dispatches.count <= 1, "scenario probes")
print("PASS FPP scenario/probe smoke")

struct FoldGolden: Decodable {
    let time: [Double], flux: [Double], t0: Double, period: Double, bins: Int
    let phase: [Double], binnedTime: [Double], binnedFlux: [Double]
    enum CodingKeys: String, CodingKey { case time, flux, t0, period, bins, phase; case binnedTime = "binned_time"; case binnedFlux = "binned_flux" }
}
