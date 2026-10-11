import Foundation

public struct ProcessedLightCurve: Equatable, Sendable {
    public let time: [Double]
    public let flatFlux: [Double]
    public let rawFlux: [Double]
    public let sectorCount: Int
    public let maxGapDays: Double

    public init(
        time: [Double], flatFlux: [Double], rawFlux: [Double],
        sectorCount: Int, maxGapDays: Double
    ) {
        self.time = time
        self.flatFlux = flatFlux
        self.rawFlux = rawFlux
        self.sectorCount = sectorCount
        self.maxGapDays = maxGapDays
    }
}

public enum LightCurvePreprocessor {
    /// Reproduces the Python worker's stitch → clean → positive sigma clip →
    /// conditional gap fill → Wotan path. Gap-fill values are intentionally
    /// stochastic in production, matching Lightkurve's unseeded behavior.
    public static func process(
        _ curves: [LightCurveData],
        iterativeDetrend: Bool = true,
        maxConcurrency: Int = ProcessInfo.processInfo.activeProcessorCount
    ) throws -> ProcessedLightCurve {
        guard !curves.isEmpty else { throw DetrendError.emptyInput }
        var time: [Double] = []
        var raw: [Double] = []
        var cadence: [Int64]? = curves.allSatisfy { $0.cadence != nil } ? [] : nil

        for curve in curves {
            guard curve.time.count == curve.flux.count,
                  curve.cadence == nil || curve.cadence?.count == curve.time.count else {
                throw DetrendError.lengthMismatch
            }
            if curves.count == 1 {
                time.append(contentsOf: curve.time)
                raw.append(contentsOf: curve.flux)
            } else {
                let normalization = nanMedian(curve.flux)
                if curve.source == .tglc {
                    // worker.py's _read_lightcurve() upcasts cal_aper_flux to
                    // np.float64 BEFORE constructing the LightCurve object, so
                    // lightkurve's normalize()/nanmedian() run entirely in
                    // float64 for TGLC/FFI — never touching the FITS `E`
                    // column's native Float32 dtype. Divide in double
                    // precision to match.
                    for index in curve.time.indices {
                        time.append(curve.time[index])
                        raw.append(curve.flux[index] / normalization)
                    }
                } else {
                    // SPOC/QLP go through plain lk.read(path), which DOES
                    // preserve the FITS `E` column's native Float32 dtype
                    // through normalize() — keep reproducing that single-
                    // rounding division.
                    let normalization32 = Float(normalization)
                    for index in curve.time.indices {
                        time.append(curve.time[index])
                        raw.append(Double(Float(curve.flux[index]) / normalization32))
                    }
                }
            }
            if let values = curve.cadence { cadence?.append(contentsOf: values) }
        }

        // LightCurve.remove_nans() removes invalid flux cadences before clipping.
        var finiteTime: [Double] = []
        var finiteRaw: [Double] = []
        var finiteCadence: [Int64]? = cadence == nil ? nil : []
        for index in time.indices where time[index].isFinite && raw[index].isFinite {
            finiteTime.append(time[index])
            finiteRaw.append(raw[index])
            if let cadence { finiteCadence?.append(cadence[index]) }
        }
        let keep = upperSigmaClipMask(finiteRaw, sigma: 5, maxIterations: 5)
        time = []
        raw = []
        cadence = finiteCadence == nil ? nil : []
        for index in keep.indices where keep[index] {
            time.append(finiteTime[index])
            raw.append(finiteRaw[index])
            if let finiteCadence { cadence?.append(finiteCadence[index]) }
        }

        var maxGap = 0.0
        if time.count > 1 {
            for index in 1..<time.count { maxGap = max(maxGap, time[index] - time[index - 1]) }
        }
        // Python calls fill_gaps only when the largest observed gap is strictly
        // below five days; its production RNG is deliberately unseeded. The
        // maxGap gate above intentionally stays on the raw, pre-sort stitch
        // order (np.diff on lc.time.value before fill_gaps()) to match
        // worker.py's threshold decision bit for bit.
        if time.count > 1, maxGap < 5 {
            // lightkurve's LightCurveCollection.stitch() concatenates sectors
            // in whatever order the FITS files were loaded/downloaded, not
            // necessarily chronological order — a multi-sector TIC can come
            // back with a sector-1 block appended after a later sector.
            //
            // Python's lc.fill_gaps() tolerates that, but *only* on the
            // cadenceno-present path (true for all SPOC 2-min data, which is
            // every curve here that carries a `cadence` array): its gap
            // reconstruction re-sorts everything by cadence number
            // (`ncad[np.argsort(ncad)]`) before rebuilding the time grid, so
            // the array fill_gaps() hands back is already time-ordered
            // regardless of stitch()'s concatenation order. Reproduce that
            // side effect explicitly here — sorted, index-aligned across
            // time/flux/cadence — so GapFiller sees monotonic input instead
            // of throwing nonMonotonicTime at a sector boundary.
            //
            // TGLC/FFI curves never carry cadence (FITSReader.swift; the
            // Python side never attaches cadenceno for TGLC either — see
            // worker.py:440-451), so they go through fill_gaps()'s *other*
            // branch (`while (t - prevtime) > 1.2 * dt`), which does NOT
            // pre-sort — it just silently skips filling a backward jump and
            // appends the point in its original (possibly out-of-order)
            // position, relying on the later post-detrend argsort to clean
            // up. Sorting here for that case would diverge from Python's
            // actual (unsorted-until-later) behavior, so only apply this
            // pre-sort when cadence is present.
            if cadence != nil {
                let fillOrder = time.indices.sorted { time[$0] < time[$1] }
                time = fillOrder.map { time[$0] }
                raw = fillOrder.map { raw[$0] }
                cadence = fillOrder.map { cadence![$0] }
            }
            var noise = SystemGaussianNoiseSource()
            let filled = try GapFiller.fillIfAllowed(
                GapFillSeries(time: time, flux: raw, cadence: cadence), noise: &noise
            )
            time = filled.time
            raw = filled.flux
            cadence = filled.cadence
        }
        var flat: [Double]
        if iterativeDetrend {
            flat = try IterativeDetrender.flatten(
                time: time, flux: raw, maxConcurrency: maxConcurrency
            ).flux
        } else {
            flat = try BiweightDetrender.flatten(time: time, flux: raw)
        }
        let flatMedian = nanMedian(flat)
        if flatMedian > 1.5 || flatMedian < 0.5 {
            flat = flat.map { $0 / flatMedian }
        }

        let order = time.indices.sorted { time[$0] < time[$1] }
        var sortedTime: [Double] = []
        var sortedFlat: [Double] = []
        var sortedRaw: [Double] = []
        sortedTime.reserveCapacity(order.count)
        sortedFlat.reserveCapacity(order.count)
        sortedRaw.reserveCapacity(order.count)
        for index in order where time[index].isFinite && flat[index].isFinite && raw[index].isFinite {
            sortedTime.append(time[index])
            sortedFlat.append(flat[index])
            sortedRaw.append(raw[index])
        }
        return ProcessedLightCurve(
            time: sortedTime, flatFlux: sortedFlat, rawFlux: sortedRaw,
            sectorCount: curves.count, maxGapDays: maxGap
        )
    }

    private static func upperSigmaClipMask(
        _ values: [Double], sigma: Double, maxIterations: Int
    ) -> [Bool] {
        var keep = Array(repeating: true, count: values.count)
        for _ in 0..<maxIterations {
            let current = values.indices.compactMap { keep[$0] ? values[$0] : nil }
            if current.isEmpty { break }
            let center = median(current)
            let mean = current.reduce(0, +) / Double(current.count)
            let variance = current.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(current.count)
            let limit = center + sigma * sqrt(variance)
            var changed = false
            for index in values.indices where keep[index] && values[index] > limit {
                keep[index] = false
                changed = true
            }
            if !changed { break }
        }
        return keep
    }

    private static func nanMedian(_ values: [Double]) -> Double {
        median(values.filter(\.isFinite))
    }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return .nan }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count % 2 == 0 { return (sorted[middle - 1] + sorted[middle]) / 2 }
        return sorted[middle]
    }
}
