import Foundation
import Metal

public enum BLSMetalError: Error, CustomStringConvertible, Sendable {
    case unavailable
    case mismatchedFluxLength(Int)
    case invalidBinCount(Int)
    case allocationFailed(String)
    case commandFailed(String)

    public var description: String {
        switch self {
        case .unavailable: return "Metal BLS is unavailable"
        case let .mismatchedFluxLength(index): return "Metal BLS flux \(index) has the wrong length"
        case let .invalidBinCount(count): return "Metal BLS bin count \(count) exceeds the threadgroup budget"
        case let .allocationFailed(name): return "Metal BLS could not allocate \(name)"
        case let .commandFailed(reason): return "Metal BLS command failed: \(reason)"
        }
    }
}

private struct BLSMetalParameters {
    var sampleCount: UInt32
    var fluxCount: UInt32
    var binCount: UInt32
    var periodCount: UInt32
    var durationCount: UInt32
}

private struct BLSMetalPreparedChunk: Sendable {
    let bins: [Int16]
    let cumulativeCounts: [Float]
}

private final class BLSMetalPreparedChunkBox: @unchecked Sendable {
    private let lock = NSLock()
    private let group = DispatchGroup()
    private var value: BLSMetalPreparedChunk?

    init() { group.enter() }

    func store(_ value: BLSMetalPreparedChunk) {
        lock.lock()
        self.value = value
        lock.unlock()
        group.leave()
    }

    func wait() -> BLSMetalPreparedChunk {
        group.wait()
        lock.lock()
        defer { lock.unlock() }
        return value!
    }
}

private final class BLSMetalContext: @unchecked Sendable {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let batchPipeline: MTLComputePipelineState
    let singlePipeline: MTLComputePipelineState
    let lock = NSLock()

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else { return nil }
        do {
            let compileOptions = MTLCompileOptions()
            compileOptions.fastMathEnabled = false
            let library = try device.makeLibrary(source: Self.source, options: compileOptions)
            guard let batchFunction = library.makeFunction(name: "bls_batch"),
                  let singleFunction = library.makeFunction(name: "bls_single") else { return nil }
            self.device = device
            self.queue = queue
            batchPipeline = try device.makeComputePipelineState(function: batchFunction)
            singlePipeline = try device.makeComputePipelineState(function: singleFunction)
        } catch {
            return nil
        }
    }

    private static let source = #"""
#include <metal_stdlib>
using namespace metal;

struct BLSParams {
    uint sample_count;
    uint flux_count;
    uint bin_count;
    uint period_count;
    uint duration_count;
};

kernel void bls_batch(
    device const short *bin_idx [[buffer(0)]],
    device const float *cum_counts [[buffer(1)]],
    device const float *flux_t [[buffer(2)]],
    device const float *totals [[buffer(3)]],
    device const int *widths [[buffer(4)]],
    constant BLSParams &params [[buffer(5)]],
    device float *out_power [[buffer(6)]],
    device uint *out_wj [[buffer(7)]],
    device float *out_depth [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    threadgroup float *scratch [[threadgroup(0)]])
{
    const uint f = gid.x;
    const uint p = gid.y;
    if (f >= params.flux_count || p >= params.period_count) return;

    const uint n = params.sample_count;
    const uint nb = params.bin_count;
    threadgroup float *sums = scratch + lid * (nb + 1u);
    for (uint k = 0; k <= nb; ++k) sums[k] = 0.0f;

    device const short *bins = bin_idx + (ulong)p * (ulong)n;
    for (uint i = 0; i < n; ++i) {
        sums[(uint)bins[i] + 1u] += flux_t[(ulong)i * params.flux_count + f];
    }
    for (uint k = 1; k <= nb; ++k) sums[k] += sums[k - 1u];

    device const float *counts = cum_counts + (ulong)p * (nb + 1u);
    const float total = totals[f];
    const float nf = (float)n;
    float best_power = -1.0e30f;
    uint best_width = 0u;
    uint best_phase = 0u;
    float best_depth = 0.0f;

    for (uint di = 0; di < params.duration_count; ++di) {
        const uint width = (uint)widths[di];
        for (uint phase = 0; phase < nb; ++phase) {
            const uint end = phase + width;
            float window_count;
            float window_sum;
            if (end <= nb) {
                window_count = counts[end] - counts[phase];
                window_sum = sums[end] - sums[phase];
            } else {
                const uint wrapped = end - nb;
                window_count = (counts[nb] - counts[phase]) + counts[wrapped];
                window_sum = (sums[nb] - sums[phase]) + sums[wrapped];
            }
            const float outside_count = nf - window_count;
            if (window_count < 3.0f || outside_count < 3.0f) continue;
            const float depth = (total - window_sum) / outside_count
                - window_sum / window_count;
            if (depth <= 0.0f) continue;
            const float power = depth * depth * window_count * outside_count
                / (window_count + outside_count);
            if (power > best_power) {
                best_power = power;
                best_width = width;
                best_phase = phase;
                best_depth = depth;
            }
        }
    }

    const ulong output = (ulong)f * params.period_count + p;
    out_power[output] = best_power;
    out_wj[output] = (best_width << 16u) | (best_phase & 0xffffu);
    out_depth[output] = best_depth;
}

// Single-light-curve specialization: one SIMD lane per trial period.  The
// batch kernel maps lanes to fluxes, which would waste 31/32 lanes here.
kernel void bls_single(
    device const short *bin_idx [[buffer(0)]],
    device const float *cum_counts [[buffer(1)]],
    device const float *flux [[buffer(2)]],
    device const float *totals [[buffer(3)]],
    device const int *widths [[buffer(4)]],
    constant BLSParams &params [[buffer(5)]],
    device float *out_power [[buffer(6)]],
    device uint *out_wj [[buffer(7)]],
    device float *out_depth [[buffer(8)]],
    uint p [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    threadgroup float *scratch [[threadgroup(0)]])
{
    if (p >= params.period_count) return;
    const uint n = params.sample_count;
    const uint nb = params.bin_count;
    threadgroup float *sums = scratch + lid * (nb + 1u);
    for (uint k = 0; k <= nb; ++k) sums[k] = 0.0f;
    device const short *bins = bin_idx + (ulong)p * (ulong)n;
    for (uint i = 0; i < n; ++i) sums[(uint)bins[i] + 1u] += flux[i];
    for (uint k = 1; k <= nb; ++k) sums[k] += sums[k - 1u];

    device const float *counts = cum_counts + (ulong)p * (nb + 1u);
    const float total = totals[0];
    const float nf = (float)n;
    float best_power = -1.0e30f;
    uint best_width = 0u;
    uint best_phase = 0u;
    float best_depth = 0.0f;
    for (uint di = 0; di < params.duration_count; ++di) {
        const uint width = (uint)widths[di];
        for (uint phase = 0; phase < nb; ++phase) {
            const uint end = phase + width;
            float window_count;
            float window_sum;
            if (end <= nb) {
                window_count = counts[end] - counts[phase];
                window_sum = sums[end] - sums[phase];
            } else {
                const uint wrapped = end - nb;
                window_count = (counts[nb] - counts[phase]) + counts[wrapped];
                window_sum = (sums[nb] - sums[phase]) + sums[wrapped];
            }
            const float outside_count = nf - window_count;
            if (window_count < 3.0f || outside_count < 3.0f) continue;
            const float depth = (total - window_sum) / outside_count
                - window_sum / window_count;
            if (depth <= 0.0f) continue;
            const float power = depth * depth * window_count * outside_count
                / (window_count + outside_count);
            if (power > best_power) {
                best_power = power;
                best_width = width;
                best_phase = phase;
                best_depth = depth;
            }
        }
    }
    out_power[p] = best_power;
    out_wj[p] = (best_width << 16u) | (best_phase & 0xffffu);
    out_depth[p] = best_depth;
}
"""#
}

public enum BLSComputeBackend: String, Sendable {
    case cpu
    case metal
    case automatic

    public static func configured(gpuEnabled: Bool, environment: [String: String] = ProcessInfo.processInfo.environment) -> BLSComputeBackend {
        if let forced = environment["EXONIT_BLS_BACKEND"]?.lowercased() {
            if forced == "metal" { return .metal }
            if ["cpu", "numpy", "numba"].contains(forced) { return .cpu }
        }
        return gpuEnabled ? .automatic : .cpu
    }

    /// Injection-recovery's default (`.automatic`, no `EXONIT_INJECTION_METAL`
    /// override) intentionally means "use Metal whenever available" — unlike
    /// ``BLSAccelerated/search(time:flux:parameters:backend:maxConcurrency:)``,
    /// which only takes the Metal path above a period-count threshold for the
    /// *primary* search. There is no such threshold here because
    /// ``InjectionRecovery/run(time:flux:options:maxConcurrency:backend:isCancelled:)``
    /// always batches a full period-cell (every radius x every injection,
    /// 90 fluxes at the production defaults) into a single Metal dispatch —
    /// the small-batch case that was 1.8x slower than CPU never occurs on the
    /// production grid.
    ///
    /// Validated 2026-08-04 offline (`.artifacts/parity-257/injection/`, see
    /// `docs/WORKLOG.md`): 13 diverse TICs, production defaults
    /// (n_injections_per_cell=10, 9 radii, injection_n_periods=5000,
    /// injection_n_bins=150, injection_n_dur_steps=10) — Swift CPU, Swift
    /// Metal, and the Python/numba reference are 100% decision-identical
    /// (7,020/7,020 individual injections across 702 radius/period cells,
    /// 0 differing cells in every pairwise comparison), and Metal is ~3.8x
    /// faster than CPU per TIC (median per-TIC speedup 3.80x, range
    /// 3.69x-4.24x across the 13 TICs; mean injection-only time 4.69s CPU vs
    /// 1.22s Metal). This is what justifies defaulting `.automatic` to Metal
    /// here with no size gate, in contrast to the primary search.
    public static func injectionConfigured(
        gpuEnabled: Bool,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> BLSComputeBackend {
        if let value = environment["EXONIT_INJECTION_METAL"]?.lowercased(),
           ["0", "false", "no", "off"].contains(value) {
            return .cpu
        }
        return configured(gpuEnabled: gpuEnabled, environment: environment)
    }

    /// Capability string reported to the server for the primary BLS backend
    /// — pure function of the resolved backend and Metal availability so it
    /// can be unit-tested without a network mock (see
    /// `testCapabilityNaming` in the contract test suite). `.automatic`
    /// always executes on CPU for the primary search (see
    /// ``BLSAccelerated/search(time:flux:parameters:backend:maxConcurrency:)``),
    /// so only the explicit `.metal` opt-in reports the Metal capability.
    public static func blsCapabilityName(backend: BLSComputeBackend, metalAvailable: Bool) -> String {
        (backend == .metal && metalAvailable) ? "swift-metal-f32-single-pass" : "swift-float64-cpu-two-pass"
    }

    /// Capability string reported to the server for the injection-recovery
    /// backend. Same shape as ``blsCapabilityName(backend:metalAvailable:)``
    /// but a distinct label (`swift-metal-f32-batch`) because injection
    /// always uses the batched (flux x period) Metal kernel, never the
    /// single-flux specialization.
    public static func injectionCapabilityName(backend: BLSComputeBackend, metalAvailable: Bool) -> String {
        (backend != .cpu && metalAvailable) ? "swift-metal-f32-batch" : "swift-float64-cpu"
    }
}

public enum BLSAccelerated {
    public static func search(
        time: [Double], flux: [Double], parameters: BLSParameters,
        backend: BLSComputeBackend, maxConcurrency: Int
    ) throws -> BLSResult {
        switch backend {
        case .cpu:
            return try BLSTwoPass.search(
                time: time, flux: flux, parameters: parameters,
                maxConcurrency: maxConcurrency
            )
        case .metal:
            guard BLSMetal.isAvailable else { throw BLSMetalError.unavailable }
            return try BLSMetal.search(time: time, flux: flux, parameters: parameters)
        case .automatic:
            // Metal-primary pending its own science gate (Float32 flips marginal decisions): .automatic always resolves to CPU for the primary search.
            return try BLSTwoPass.search(
                time: time, flux: flux, parameters: parameters,
                maxConcurrency: maxConcurrency
            )
        }
    }
}

public enum BLSMetal {
    private static let context = BLSMetalContext()
    private static let fluxThreads = 32
    private static let periodChunk = 1_000

    public static var isAvailable: Bool { context != nil }
    public static var deviceName: String? { context?.device.name }

    public static func search(
        time: [Double], flux: [Double], parameters: BLSParameters = BLSParameters()
    ) throws -> BLSResult {
        try searchBatch(time: time, fluxes: [flux], parameters: parameters)[0]
    }

    public static func searchBatch(
        time: [Double], fluxes: [[Double]],
        parameters p: BLSParameters = BLSParameters()
    ) throws -> [BLSResult] {
        guard time.count >= 6 else { throw BLSError.insufficientSamples }
        guard !fluxes.isEmpty else { return [] }
        for (index, flux) in fluxes.enumerated() where flux.count != time.count {
            throw BLSMetalError.mismatchedFluxLength(index)
        }
        guard p.periodMinimum > 0, p.periodMaximum > p.periodMinimum,
              p.periodCount >= 2, p.binCount >= 3, p.durationStepCount >= 1,
              p.durationMinimumFraction > 0,
              p.durationMaximumFraction >= p.durationMinimumFraction else {
            throw BLSError.invalidParameters("grid")
        }
        guard !BLSReference.durationBinGrid(parameters: p).isEmpty else {
            throw BLSError.invalidParameters("duration range produces no valid bins")
        }
        guard let context else { throw BLSMetalError.unavailable }
        let scratchBytes = fluxThreads * (p.binCount + 1) * MemoryLayout<Float>.stride
        guard scratchBytes <= context.device.maxThreadgroupMemoryLength,
              p.binCount <= Int(Int16.max) else {
            throw BLSMetalError.invalidBinCount(p.binCount)
        }

        let referenceTime = time[0]
        let relativeTime = time.map { $0 - referenceTime }
        let periods = BLSReference.frequencyPeriodGrid(
            minimum: p.periodMinimum, maximum: p.periodMaximum, count: p.periodCount
        )
        let durationBins = BLSReference.durationBinGrid(parameters: p)
        let fluxCount = fluxes.count
        var centered = [Float](repeating: 0, count: time.count * fluxCount)
        var totals = [Float](repeating: 0, count: fluxCount)
        for fluxIndex in fluxes.indices {
            let mean = fluxes[fluxIndex].reduce(0, +) / Double(time.count)
            var total = 0.0
            for sample in time.indices {
                let value = fluxes[fluxIndex][sample] - mean
                centered[sample * fluxCount + fluxIndex] = Float(value)
                total += value
            }
            totals[fluxIndex] = Float(total)
        }
        let widths = durationBins.map(Int32.init)
        let fluxBuffer = try buffer(context.device, centered, name: "centered flux")
        let totalsBuffer = try buffer(context.device, totals, name: "flux totals")
        let widthsBuffer = try buffer(context.device, widths, name: "duration widths")

        var allPowers = Array(
            repeating: [Double](repeating: 0, count: periods.count), count: fluxCount
        )
        var allPacked = Array(
            repeating: [UInt32](repeating: 0, count: periods.count), count: fluxCount
        )
        var allDepths = Array(
            repeating: [Double](repeating: 0, count: periods.count), count: fluxCount
        )

        context.lock.lock()
        defer { context.lock.unlock() }
        var currentPrepared = Self.prepareChunk(
            start: 0, count: min(periodChunk, periods.count), periods: periods,
            relativeTime: relativeTime, binCount: p.binCount
        )
        for start in stride(from: 0, to: periods.count, by: periodChunk) {
            let end = min(start + periodChunk, periods.count)
            let count = end - start
            let prepared = currentPrepared
            let nextStart = start + periodChunk
            let nextChunk: BLSMetalPreparedChunkBox?
            if nextStart < periods.count {
                let box = BLSMetalPreparedChunkBox()
                nextChunk = box
                DispatchQueue.global(qos: .userInitiated).async {
                    box.store(Self.prepareChunk(
                        start: nextStart,
                        count: min(periodChunk, periods.count - nextStart),
                        periods: periods, relativeTime: relativeTime,
                        binCount: p.binCount
                    ))
                }
            } else {
                nextChunk = nil
            }
            let bins = prepared.bins
            let cumulativeCounts = prepared.cumulativeCounts
            let binsBuffer = try buffer(context.device, bins, name: "phase bins")
            let countsBuffer = try buffer(context.device, cumulativeCounts, name: "bin counts")
            var parameters = BLSMetalParameters(
                sampleCount: UInt32(time.count), fluxCount: UInt32(fluxCount),
                binCount: UInt32(p.binCount), periodCount: UInt32(count),
                durationCount: UInt32(widths.count)
            )
            let parametersBuffer = try withUnsafeBytes(of: &parameters) { bytes in
                guard let base = bytes.baseAddress,
                      let result = context.device.makeBuffer(
                        bytes: base, length: bytes.count, options: .storageModeShared
                      ) else { throw BLSMetalError.allocationFailed("parameters") }
                return result
            }
            let outputCount = fluxCount * count
            guard let powersBuffer = context.device.makeBuffer(
                    length: outputCount * MemoryLayout<Float>.stride, options: .storageModeShared),
                  let packedBuffer = context.device.makeBuffer(
                    length: outputCount * MemoryLayout<UInt32>.stride, options: .storageModeShared),
                  let depthsBuffer = context.device.makeBuffer(
                    length: outputCount * MemoryLayout<Float>.stride, options: .storageModeShared),
                  let command = context.queue.makeCommandBuffer(),
                  let encoder = command.makeComputeCommandEncoder() else {
                throw BLSMetalError.allocationFailed("command/output")
            }
            let single = fluxCount == 1
            encoder.setComputePipelineState(single ? context.singlePipeline : context.batchPipeline)
            for (index, input) in [binsBuffer, countsBuffer, fluxBuffer, totalsBuffer, widthsBuffer,
                                   parametersBuffer, powersBuffer, packedBuffer, depthsBuffer].enumerated() {
                encoder.setBuffer(input, offset: 0, index: index)
            }
            encoder.setThreadgroupMemoryLength(scratchBytes, index: 0)
            if single {
                encoder.dispatchThreads(
                    MTLSize(width: count, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: fluxThreads, height: 1, depth: 1)
                )
            } else {
                encoder.dispatchThreads(
                    MTLSize(width: fluxCount, height: count, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: fluxThreads, height: 1, depth: 1)
                )
            }
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            guard command.status == .completed else {
                throw BLSMetalError.commandFailed(command.error?.localizedDescription ?? "unknown")
            }
            let outputPowers = powersBuffer.contents().bindMemory(to: Float.self, capacity: outputCount)
            let outputPacked = packedBuffer.contents().bindMemory(to: UInt32.self, capacity: outputCount)
            let outputDepths = depthsBuffer.contents().bindMemory(to: Float.self, capacity: outputCount)
            for fluxIndex in 0..<fluxCount {
                for localPeriod in 0..<count {
                    let output = fluxIndex * count + localPeriod
                    allPowers[fluxIndex][start + localPeriod] = Double(outputPowers[output])
                    allPacked[fluxIndex][start + localPeriod] = outputPacked[output]
                    allDepths[fluxIndex][start + localPeriod] = Double(outputDepths[output])
                }
            }
            if let nextChunk {
                currentPrepared = nextChunk.wait()
            }
        }

        return fluxes.indices.map { fluxIndex in
            let powers = allPowers[fluxIndex]
            var bestIndex = 0
            for index in 1..<powers.count where powers[index] > powers[bestIndex] {
                bestIndex = index
            }
            let packed = allPacked[fluxIndex][bestIndex]
            let width = Int(packed >> 16)
            let phase = Int(packed & 0xffff)
            let period = periods[bestIndex]
            let depth = allDepths[fluxIndex][bestIndex]
            return BLSResult(
                period: period, power: powers[bestIndex],
                duration: Double(width) / Double(p.binCount) * period,
                t0: Double(phase) / Double(p.binCount) * period + referenceTime,
                depth: depth, depthPPM: depth * 1_000_000,
                snr: BLSReference.sde(spectrum: powers, peakIndex: bestIndex),
                periods: periods, powerSpectrum: powers
            )
        }
    }

    private static func prepareChunk(
        start: Int, count: Int, periods: [Double], relativeTime: [Double], binCount: Int
    ) -> BLSMetalPreparedChunk {
        var bins = [Int16](repeating: 0, count: count * relativeTime.count)
        var cumulativeCounts = [Float](repeating: 0, count: count * (binCount + 1))
        for localPeriod in 0..<count {
            let period = periods[start + localPeriod]
            var counts = [Int](repeating: 0, count: binCount)
            for sample in relativeTime.indices {
                let quotient = relativeTime[sample] / period
                let phase = quotient - floor(quotient)
                let raw = Int(phase * Double(binCount))
                let bin = min(max(raw, 0), binCount - 1)
                bins[localPeriod * relativeTime.count + sample] = Int16(bin)
                counts[bin] += 1
            }
            var running = 0
            for bin in 0..<binCount {
                running += counts[bin]
                cumulativeCounts[localPeriod * (binCount + 1) + bin + 1] = Float(running)
            }
        }
        return BLSMetalPreparedChunk(bins: bins, cumulativeCounts: cumulativeCounts)
    }

    private static func buffer<T>(
        _ device: MTLDevice, _ values: [T], name: String
    ) throws -> MTLBuffer {
        try values.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress,
                  let result = device.makeBuffer(
                    bytes: base, length: bytes.count, options: .storageModeShared
                  ) else { throw BLSMetalError.allocationFailed(name) }
            return result
        }
    }
}
