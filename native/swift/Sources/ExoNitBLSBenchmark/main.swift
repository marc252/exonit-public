import ExoNitCore
import Foundation

let arguments = CommandLine.arguments
let threads = arguments.count > 1 ? Int(arguments[1]) ?? 2 : 2
let runs = arguments.count > 2 ? Int(arguments[2]) ?? 3 : 3
let backend = arguments.count > 3 ? arguments[3].lowercased() : "cpu"
let fluxCount = arguments.count > 4 ? Int(arguments[4]) ?? 1 : 1
let sampleCount = arguments.count > 5 ? Int(arguments[5]) ?? 600 : 600
let periodCount = arguments.count > 6 ? Int(arguments[6]) ?? 10_000 : 10_000
let signalPeriod = 7.25
let time = (0..<sampleCount).map { Double($0) / 24.0 }
let flux = time.map { value -> Double in
    let phase = (value / signalPeriod).truncatingRemainder(dividingBy: 1)
    let baseline = 1.0 + 0.0008 * sin(value * 0.73) + 0.0003 * cos(value * 1.91)
    return baseline - ((phase >= 0.20 && phase < 0.26) ? 0.006 : 0)
}
var fluxes: [[Double]] = []
fluxes.reserveCapacity(fluxCount)
for index in 0..<fluxCount {
    var row: [Double] = []
    row.reserveCapacity(flux.count)
    for (sample, value) in flux.enumerated() {
        let noiseCode = (sample * 17 + index * 31) % 101 - 50
        row.append(value + Double(noiseCode) * 1e-7)
    }
    fluxes.append(row)
}
let parameters = BLSParameters(
    periodMinimum: 0.5, periodMaximum: 50, periodCount: periodCount,
    binCount: 200, durationStepCount: 15,
    durationMinimumFraction: 0.01, durationMaximumFraction: 0.15
)
let warmup = BLSParameters(
    periodMinimum: 0.5, periodMaximum: 50, periodCount: 200,
    binCount: 200, durationStepCount: 15,
    durationMinimumFraction: 0.01, durationMaximumFraction: 0.15
)
if backend == "metal" {
    guard BLSMetal.isAvailable else { fatalError("Metal unavailable") }
    _ = try BLSMetal.searchBatch(time: time, fluxes: fluxes, parameters: warmup)
} else {
    _ = try BLSCPUOptimized.search(
        time: time, flux: flux, parameters: warmup, maxConcurrency: threads
    )
}
var seconds: [Double] = []
var last: BLSResult?
for _ in 0..<runs {
    let start = DispatchTime.now().uptimeNanoseconds
    if backend == "metal" {
        last = try BLSMetal.searchBatch(
            time: time, fluxes: fluxes, parameters: parameters
        ).first
    } else {
        precondition(fluxCount == 1, "CPU benchmark only supports one flux")
        last = try BLSCPUOptimized.search(
            time: time, flux: flux, parameters: parameters, maxConcurrency: threads
        )
    }
    let end = DispatchTime.now().uptimeNanoseconds
    seconds.append(Double(end - start) / 1_000_000_000)
}
let sorted = seconds.sorted()
let median = sorted[sorted.count / 2]
let payload: [String: Any] = [
    "backend": backend == "metal" ? "swift-metal-f32" : "swift-float64-cpu",
    "device": BLSMetal.deviceName ?? "cpu",
    "threads": threads,
    "fluxes": fluxCount,
    "samples": sampleCount,
    "periods": parameters.periodCount,
    "runs_s": seconds,
    "median_s": median,
    "best_period": last?.period ?? 0,
    "best_power": last?.power ?? 0,
]
let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
