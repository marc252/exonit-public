import ExoNitCore
import Foundation

/// Diagnostic tool: feeds a FIXED (time, raw_flux, flat_flux) triple straight
/// into `MultiPlanetSearcher.search`, bypassing FITS download/stitch/normalize
/// entirely. Paired with the Python-side `run_bls_replay.py` monkeypatch of
/// `_load_star`, this isolates whether a Python/Swift candidate mismatch
/// comes from preprocessing (this tool would then match Python's engine fed
/// the same array) or from the BLS/masking port itself (this tool would
/// differ from Python's engine even given byte-identical input).
///
/// Input JSON: {"time": [...], "raw_flux": [...], "flat_flux": [...],
///              "sector_count": N}
/// Output JSON: {"candidates": [{"candidate_num", "period_days", "t0",
///              "duration_days", "snr", "depth_ppm", "bls_power",
///              "n_transits_expected"}, ...]}
@main
enum BLSReplay {
    static func main() throws {
        guard CommandLine.arguments.count >= 3 else {
            FileHandle.standardError.write(Data(
                "usage: exonit-bls-replay INPUT OUTPUT [TIC]\n".utf8
            ))
            exit(2)
        }
        let inputURL = URL(fileURLWithPath: CommandLine.arguments[1])
        let outputURL = URL(fileURLWithPath: CommandLine.arguments[2])

        let data = try Data(contentsOf: inputURL)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let timeRaw = object["time"] as? [Any],
              let rawFluxRaw = object["raw_flux"] as? [Any],
              let flatFluxRaw = object["flat_flux"] as? [Any]
        else {
            FileHandle.standardError.write(Data("invalid input JSON\n".utf8))
            exit(2)
        }
        func toDoubles(_ values: [Any]) -> [Double] {
            values.map { value in
                if let d = value as? Double { return d }
                if let n = value as? NSNumber { return n.doubleValue }
                return .nan
            }
        }
        let time = toDoubles(timeRaw)
        let rawFlux = toDoubles(rawFluxRaw)
        let flatFlux = toDoubles(flatFluxRaw)

        // TIC 10756829 (TGLC FFI, 2 sectors): stellar_radius/mass are both
        // null in the fixture metadata, so the physical-T14 gate is a no-op
        // on both sides regardless. Defaults below mirror worker.py's
        // config.get(...) fallbacks exactly (see multiPlanetOptions() in
        // PrimaryTaskProcessor.swift, which is MultiPlanetOptions()'s
        // memberwise-default already).
        let options = MultiPlanetOptions()
        let result = try MultiPlanetSearcher.search(
            time: time, rawFlux: rawFlux, initialDetrendedFlux: flatFlux,
            stellarRadiusSolar: nil, stellarMassSolar: nil,
            source: .ffi, options: options,
            maxConcurrency: ProcessInfo.processInfo.activeProcessorCount,
            blsBackend: .cpu
        )

        let span = (time.last ?? 0) - (time.first ?? 0)
        let candidates: [[String: Any]] = result.candidates.map { c in
            let expectedTransits = c.expectedTransits ?? Int(
                c.periodDays > 0 ? span / c.periodDays : 0
            )
            return [
                "candidate_num": c.candidateNumber,
                "period_days": c.periodDays,
                "t0": c.t0,
                "duration_days": c.durationDays,
                "depth_ppm": c.depthPPM,
                "snr": c.snr,
                "bls_power": c.blsPower ?? 0,
                "n_transits_expected": expectedTransits,
            ]
        }
        let payload: [String: Any] = [
            "candidates": candidates,
            "rejected_unphysical_count": result.rejectedUnphysicalCount,
            "redetrend_failure_count": result.redetrendFailureCount,
        ]
        let out = try JSONSerialization.data(
            withJSONObject: payload, options: [.sortedKeys, .prettyPrinted]
        )
        try out.write(to: outputURL, options: .atomic)

        print("=== candidates ===")
        for c in result.candidates {
            print(
                "#\(c.candidateNumber) P=\(c.periodDays)d t0=\(c.t0) "
                + "dur=\(c.durationDays)d SNR=\(c.snr) depth=\(c.depthPPM)ppm"
            )
        }
    }
}
