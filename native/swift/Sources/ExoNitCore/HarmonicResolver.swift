import Foundation

public enum LightCurveSource: String, Sendable {
    case spoc, ffi
}

public struct HarmonicResolutionOptions: Equatable, Sendable {
    public var minimumSNR: Double
    public var depthRatioLower: Double
    public var depthRatioUpper: Double
    public var source: LightCurveSource
    public var longPeriodRecovery: Bool
    public var longPeriodMinimumDays: Double
    public var longSNRFactor: Double

    public init(
        minimumSNR: Double = 5, depthRatioLower: Double = 0.5,
        depthRatioUpper: Double = 2, source: LightCurveSource = .spoc,
        longPeriodRecovery: Bool = false, longPeriodMinimumDays: Double = 10,
        longSNRFactor: Double = 0.5
    ) {
        self.minimumSNR = minimumSNR; self.depthRatioLower = depthRatioLower
        self.depthRatioUpper = depthRatioUpper; self.source = source
        self.longPeriodRecovery = longPeriodRecovery
        self.longPeriodMinimumDays = longPeriodMinimumDays
        self.longSNRFactor = longSNRFactor
    }
}

public typealias BLSSearchFunction = @Sendable (
    _ time: [Double], _ flux: [Double], _ parameters: BLSParameters
) throws -> BLSResult

public enum HarmonicResolver {
    private struct Probe {
        let period: Double
        let snrFactor: Double
        let transitGuard: Bool
    }

    public static func resolve(
        time: [Double], flux: [Double], seed: BLSResult,
        periodMinimum: Double, periodMaximum: Double,
        options: HarmonicResolutionOptions = HarmonicResolutionOptions(),
        search: BLSSearchFunction
    ) -> BLSResult {
        var probes: [Probe] = []
        if options.source == .ffi {
            for multiplier in [2.0, 3.0, 4.0, 5.0] {
                let period = seed.period * multiplier
                if period <= periodMaximum {
                    probes.append(Probe(period: period, snrFactor: 0.6, transitGuard: true))
                }
            }
        } else {
            let doubled = seed.period * 2
            if doubled <= periodMaximum {
                probes.append(Probe(period: doubled, snrFactor: 1, transitGuard: false))
            }
            if options.longPeriodRecovery {
                for multiplier in [3.0 / 2, 5.0 / 3, 4.0 / 3, 5.0 / 2] {
                    let period = seed.period * multiplier
                    if period >= options.longPeriodMinimumDays, period <= periodMaximum {
                        probes.append(Probe(
                            period: period, snrFactor: options.longSNRFactor,
                            transitGuard: true
                        ))
                    }
                }
            }
        }
        let demotionFactor = options.source == .ffi ? 0.6 : 1.0
        for divisor in [2.0, 3.0, 4.0, 5.0] {
            let period = seed.period / divisor
            if period >= periodMinimum {
                probes.append(Probe(
                    period: period, snrFactor: demotionFactor, transitGuard: false
                ))
            }
        }
        let span = ((time.last ?? 0) - (time.first ?? 0))
        for probe in probes {
            // worker.py's resolve_harmonic() calls bls_fn(..., period_min=cand*0.95,
            // period_max=cand*1.05, ...) WITHOUT clamping to the caller's global
            // period_min/period_max — the refine window is allowed to extend
            // outside [periodMinimum, periodMaximum] (e.g. a subharmonic probe
            // near periodMinimum legitimately searches slightly below it). A
            // clamp here narrows the fine-BLS grid vs Python, landing on a
            // different local peak/SNR near marginal-significance systematics
            // (found via TIC 10757037: identical seed BLS + identical
            // re-detrended flux, but the clamped probe window here resolved to
            // a different period/SNR than Python's unclamped window).
            let parameters = BLSParameters(
                periodMinimum: probe.period * 0.95,
                periodMaximum: probe.period * 1.05,
                periodCount: 500
            )
            guard parameters.periodMaximum > parameters.periodMinimum,
                  let candidate = try? search(time, flux, parameters) else { continue }
            guard candidate.snr >= options.minimumSNR,
                  candidate.snr >= seed.snr * probe.snrFactor else { continue }
            if seed.depth > 0 {
                let ratio = candidate.depth / seed.depth
                guard ratio > options.depthRatioLower, ratio < options.depthRatioUpper else {
                    continue
                }
            }
            if probe.transitGuard, span > 0, candidate.period > 0,
               span / candidate.period < 2 { continue }
            return candidate
        }
        return seed
    }
}
