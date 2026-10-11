import Foundation

public struct FPPStarRecord: Equatable, Sendable {
    public var id: String, ra: Double, dec: Double, mass: Double, radius: Double
    public var teff: Double, tmag: Double, parallax: Double, isTrilegal: Bool
    public var weightedFlux = Double.nan, fluxRatio = Double.nan, transitDepth = Double.nan
    public init(id: String, ra: Double, dec: Double, mass: Double, radius: Double, teff: Double, tmag: Double, parallax: Double, isTrilegal: Bool = false) {
        self.id = id; self.ra = ra; self.dec = dec; self.mass = mass; self.radius = radius; self.teff = teff; self.tmag = tmag; self.parallax = parallax; self.isTrilegal = isTrilegal
    }
}
public struct FPPNearbyResponse: Equatable, Sendable { public var stars: [FPPStarRecord]; public var galacticLatitude: Double; public var targetPromoted: Bool }

public enum FPPInputs {
    private static func number(_ value: JSONValue?) -> Double {
        switch value { case let .number(v): return v; case let .integer(v): return Double(v); default: return .nan }
    }
    private static func text(_ value: JSONValue?) -> String {
        switch value { case let .string(v): return v; case let .integer(v): return String(v); case let .number(v): return String(v); case .none: return ""; default: return "null" }
    }
    public static func parseNearby(_ object: [String: JSONValue], ticID: String) -> FPPNearbyResponse? {
        guard case let .array(rawStars)? = object["stars"] else { return nil }
        var stars = rawStars.compactMap { value -> FPPStarRecord? in
            guard case let .object(s) = value else { return nil }
            return FPPStarRecord(id: text(s["ID"]), ra: number(s["ra"]), dec: number(s["dec"]), mass: number(s["mass"]), radius: number(s["rad"]), teff: number(s["Teff"]), tmag: number(s["Tmag"]), parallax: number(s["plx"]))
        }
        var promoted = false
        if let index = stars.firstIndex(where: { $0.id == ticID }), index != 0 { stars.insert(stars.remove(at: index), at: 0); promoted = true }
        let galacticLatitude = object.keys.contains("gal_b") ? number(object["gal_b"]) : 0.0
        return FPPNearbyResponse(stars: stars, galacticLatitude: galacticLatitude, targetPromoted: promoted)
    }
    public static func loadTrilegal(_ csv: String, targetRA: Double, targetDec: Double, targetTmag: Double) -> [FPPStarRecord] {
        var lines = csv.split(whereSeparator: { $0.isNewline || $0 == "\r" }).map { $0.split(whereSeparator: { $0 == " " || $0 == "\t" }) }
        guard let header = lines.first, let tess = header.firstIndex(of: "TESS") else { return [] }
        let logTe = header.firstIndex(of: "logTe"), mact = header.firstIndex(of: "Mact"), logg = header.firstIndex(of: "logg"); lines.removeFirst()
        var out = [FPPStarRecord]()
        for row in lines {
            guard row.indices.contains(tess), let tm = Double(row[tess]), tm <= targetTmag + 8 else { continue }
            let teff = logTe.flatMap { row.indices.contains($0) ? Double(row[$0]).map { pow(10, $0) } : nil } ?? 5777
            let mass = mact.flatMap { row.indices.contains($0) ? Double(row[$0]) : nil } ?? 1
            let gravity = logg.flatMap { row.indices.contains($0) ? Double(row[$0]) : nil } ?? 4.4
            let radius = sqrt(6.674e-8 * mass * 1.989e33 / pow(10, gravity)) / 6.957e10
            out.append(FPPStarRecord(id: "TRILEGAL_\(out.count)", ra: targetRA, dec: targetDec, mass: mass, radius: radius, teff: teff, tmag: tm, parallax: .nan, isTrilegal: true))
        }
        return out
    }
    public static func computeFluxRatios(_ stars: inout [FPPStarRecord], depthPPM: Double) {
        guard let target = stars.first else { return }; let sigma = 21.0
        for i in stars.indices {
            let sep = (FPPNumpy.libmPow(stars[i].ra - target.ra, 2) + FPPNumpy.libmPow(stars[i].dec - target.dec, 2)).squareRoot() * 3600
            let raw = pow(10, (target.tmag - stars[i].tmag) / 2.5)
            stars[i].weightedFlux = raw * exp(-0.5 * FPPNumpy.libmPow(sep / sigma, 2))
        }
        let total = stars.reduce(0) { $0 + $1.weightedFlux }
        for i in stars.indices { stars[i].fluxRatio = stars[i].weightedFlux / total; stars[i].transitDepth = stars[i].fluxRatio > 0 ? depthPPM / stars[i].fluxRatio : 0 }
    }
}
