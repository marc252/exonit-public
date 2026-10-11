import Dispatch
import ExoNitCore
import Foundation

let defaultPath = "~/exonit-worker/fits_cache_3/hlsp_tglc_tess_ffi_gaiaid-1719071625276929792-s0026-cam4-ccd1_tess_v1_llc.fits"
let path = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : defaultPath
let runs = CommandLine.arguments.count > 2 ? Int(CommandLine.arguments[2]) ?? 200 : 200
let url = URL(fileURLWithPath: path)

for _ in 0..<10 { _ = try FITSReader.readTGLCLightCurve(at: url) }
var samples: [Double] = []
samples.reserveCapacity(runs)
var last: LightCurveData?
for _ in 0..<runs {
    let start = DispatchTime.now().uptimeNanoseconds
    last = try FITSReader.readTGLCLightCurve(at: url)
    let end = DispatchTime.now().uptimeNanoseconds
    samples.append(Double(end - start) / 1_000_000_000)
}
samples.sort()
let result = last!
let output: [String: Any] = [
    "implementation": "swift-native-fits",
    "runs": runs,
    "median_seconds": samples[samples.count / 2],
    "p95_seconds": samples[min(samples.count - 1, Int(Double(samples.count) * 0.95))],
    "raw_rows": result.rawRowCount,
    "good_rows": result.time.count,
    "sum_time": result.time.reduce(0, +),
    "sum_flux": result.flux.reduce(0, +),
]
let data = try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys])
print(String(decoding: data, as: UTF8.self))
