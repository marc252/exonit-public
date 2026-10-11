import Dispatch
import ExoNitCore
import Foundation

let paths = Array(CommandLine.arguments.dropFirst().prefix(2))
let fixturePaths = paths.count == 2 ? paths : [
    "~/exonit-worker/fits_cache_7/hlsp_qlp_tess_ffi_s0013-0000000300868410_tess_v01_llc.fits",
    "~/exonit-worker/fits_cache_7/hlsp_qlp_tess_ffi_s0039-0000000300868410_tess_v01_llc.fits",
]
let runs = CommandLine.arguments.count > 3 ? Int(CommandLine.arguments[3]) ?? 10 : 10
for _ in 0..<2 {
    let curves = try fixturePaths.map { try FITSReader.readLightCurve(at: URL(fileURLWithPath: $0)) }
    _ = try LightCurvePreprocessor.process(curves)
}
var samples: [Double] = []
var last: ProcessedLightCurve?
for _ in 0..<runs {
    let start = DispatchTime.now().uptimeNanoseconds
    let curves = try fixturePaths.map { try FITSReader.readLightCurve(at: URL(fileURLWithPath: $0)) }
    last = try LightCurvePreprocessor.process(curves)
    let end = DispatchTime.now().uptimeNanoseconds
    samples.append(Double(end - start) / 1_000_000_000)
}
samples.sort()
let result = last!
let output: [String: Any] = [
    "implementation": "swift-native-qlp-pipeline",
    "runs": runs,
    "median_seconds": samples[samples.count / 2],
    "p95_seconds": samples[min(samples.count - 1, Int(Double(samples.count) * 0.95))],
    "points": result.time.count,
    "sum_flat": result.flatFlux.reduce(0, +),
]
let data = try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys])
print(String(decoding: data, as: UTF8.self))
