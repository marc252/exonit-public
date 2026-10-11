import Foundation

public enum BLSTwoPass {
    /// Exact control-flow port of `bls_fast.bls_two_pass`.
    public static func search(
        time: [Double], flux: [Double],
        parameters p: BLSParameters = BLSParameters(periodCount: 50_000),
        maxConcurrency: Int = ProcessInfo.processInfo.activeProcessorCount
    ) throws -> BLSResult {
        let coarseCount = min(5_000, p.periodCount)
        var coarseParameters = p
        coarseParameters.periodCount = coarseCount
        let coarse = try BLSCPUOptimized.search(
            time: time, flux: flux, parameters: coarseParameters,
            maxConcurrency: maxConcurrency
        )
        if coarse.snr < 3 { return coarse }

        let coarseStep = (p.periodMaximum - p.periodMinimum) / Double(coarseCount)
        let refineHalfWidth = max(coarseStep * 10, coarse.period * 0.02)
        var fineParameters = p
        fineParameters.periodMinimum = max(p.periodMinimum, coarse.period - refineHalfWidth)
        fineParameters.periodMaximum = min(p.periodMaximum, coarse.period + refineHalfWidth)
        fineParameters.periodCount = min(p.periodCount / 5, 10_000)
        guard fineParameters.periodCount >= 2 else { return coarse }
        let fine = try BLSCPUOptimized.search(
            time: time, flux: flux, parameters: fineParameters,
            maxConcurrency: maxConcurrency
        )
        return BLSResult(
            period: fine.period, power: fine.power, duration: fine.duration,
            t0: fine.t0, depth: fine.depth, depthPPM: fine.depthPPM,
            snr: coarse.snr, periods: coarse.periods,
            powerSpectrum: coarse.powerSpectrum
        )
    }
}
