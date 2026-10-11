import Foundation

public enum JSONValue: Codable, Equatable, Sendable {
    case null
    case bool(Bool)
    case integer(Int64)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let value = try? container.decode(Bool.self) { self = .bool(value) }
        else if let value = try? container.decode(Int64.self) { self = .integer(value) }
        else if let value = try? container.decode(Double.self) { self = .number(value) }
        else if let value = try? container.decode(String.self) { self = .string(value) }
        else if let value = try? container.decode([JSONValue].self) { self = .array(value) }
        else if let value = try? container.decode([String: JSONValue].self) { self = .object(value) }
        else { throw DecodingError.dataCorruptedError(in: container, debugDescription: "unsupported JSON value") }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case let .bool(value): try container.encode(value)
        case let .integer(value): try container.encode(value)
        case let .number(value): try container.encode(value)
        case let .string(value): try container.encode(value)
        case let .array(value): try container.encode(value)
        case let .object(value): try container.encode(value)
        }
    }
}

private func decodeFlexibleString<K: CodingKey>(
    _ container: KeyedDecodingContainer<K>, forKey key: K
) throws -> String {
    if let value = try? container.decode(String.self, forKey: key) { return value }
    if let value = try? container.decode(Int64.self, forKey: key) { return String(value) }
    throw DecodingError.dataCorruptedError(forKey: key, in: container, debugDescription: "expected string or integer")
}

public struct CandidateResult: Codable, Equatable, Sendable {
    public var candidateNumber: Int
    public var periodDays: Double
    public var t0: Double
    public var durationDays: Double
    public var depthPPM: Double
    public var snr: Double
    public var blsPower: Double?
    public var expectedTransits: Int?
    public var triceratopsFPP: Double? = nil
    public var triceratopsNFPP: Double? = nil
    public var nfpp: Double? = nil
    public var triceratopsProbabilityPlanet: Double? = nil
    public var triceratopsVerdict: String? = nil
    public var exominerScore: Double? = nil
    public var exominerTCEPeriod: Double? = nil
    public var exominerSectorRun: String? = nil
    public var exominerFFIScore: Double? = nil
    public var exominerFFIModelVersion: String? = nil
    public var crossSourceName: String? = nil
    public var crossSourceSNR: Double? = nil
    public var crossSourceDepthPPM: Double? = nil
    public var crossSourceDepthRatio: Double? = nil
    public var crossSourceAgreement: Bool? = nil

    enum CodingKeys: String, CodingKey {
        case candidateNumber = "candidate_num", periodDays = "period_days", t0
        case durationDays = "duration_days", depthPPM = "depth_ppm", snr
        case blsPower = "bls_power", expectedTransits = "n_transits_expected"
        case triceratopsFPP = "triceratops_fpp", triceratopsNFPP = "triceratops_nfpp", nfpp
        case triceratopsProbabilityPlanet = "triceratops_prob_planet", triceratopsVerdict = "triceratops_verdict"
        case exominerScore = "exominer_score", exominerTCEPeriod = "exominer_tce_period", exominerSectorRun = "exominer_sector_run"
        case exominerFFIScore = "exominer_ffi_score", exominerFFIModelVersion = "exominer_ffi_model_version"
        case crossSourceName = "cross_source_name", crossSourceSNR = "cross_source_snr"
        case crossSourceDepthPPM = "cross_source_depth_ppm", crossSourceDepthRatio = "cross_source_depth_ratio"
        case crossSourceAgreement = "cross_source_agreement"
    }

    public init(candidateNumber: Int, periodDays: Double, t0: Double, durationDays: Double, depthPPM: Double, snr: Double, blsPower: Double? = nil, expectedTransits: Int? = nil) {
        self.candidateNumber = candidateNumber; self.periodDays = periodDays; self.t0 = t0
        self.durationDays = durationDays; self.depthPPM = depthPPM; self.snr = snr
        self.blsPower = blsPower; self.expectedTransits = expectedTransits
    }
}

public struct VettingResult: Codable, Equatable, Sendable {
    public var isHarmonic: Bool?
    public var harmonicOf: Int?
    public var harmonicRatio: Double?
    public var oddEvenConsistent: Bool?
    public var oddDepth: Double?
    public var evenDepth: Double?
    public var isVShaped: Bool?
    public var vMetric: Double?
    public var passed: Bool?

    enum CodingKeys: String, CodingKey {
        case isHarmonic = "is_harmonic", harmonicOf = "harmonic_of", harmonicRatio = "harmonic_ratio"
        case oddEvenConsistent = "odd_even_consistent", oddDepth = "odd_depth", evenDepth = "even_depth"
        case isVShaped = "is_v_shaped", vMetric = "v_metric", passed
    }
}

public struct CompletenessCell: Codable, Equatable, Sendable {
    public var radiusEarth: Double
    public var periodDays: Double
    public var depthPPM: Double?
    public var detectionProbability: Double
    public var recoveredCount: Int?
    public var injectedCount: Int?

    enum CodingKeys: String, CodingKey {
        case radiusEarth = "radius_earth", periodDays = "period_days", depthPPM = "depth_ppm"
        case detectionProbability = "detection_prob", recoveredCount = "n_recovered", injectedCount = "n_injected"
    }
}

public struct CompletenessResult: Codable, Equatable, Sendable {
    public var radiusGrid: [Double]?
    public var periodGrid: [Double]?
    public var completenessMap: [[Double]]?
    public var details: [CompletenessCell]
    public var stellarRadius: Double?
    public var truncated: Bool?
    public var periodsEvaluated: Int?
    public var radiiEvaluated: Int?

    enum CodingKeys: String, CodingKey {
        case radiusGrid = "radius_grid", periodGrid = "period_grid", completenessMap = "completeness_map"
        case details, stellarRadius = "stellar_radius", truncated
        case periodsEvaluated = "periods_evaluated", radiiEvaluated = "radii_evaluated"
    }
}

public struct PrimaryResultRequest: Codable, Equatable, Sendable {
    public var taskID: String?
    public var ticID: String?
    public var candidates: [CandidateResult]
    public var vettingResults: [VettingResult]
    public var cnnScores: [Double?]
    public var completeness: CompletenessResult?
    public var processingTimeSeconds: Double
    public var gpuUsed: Bool
    public var sectorCount: Int
    public var pointCount: Int
    public var timeSpanDays: Double?
    public var ra: Double?
    public var dec: Double?
    public var tmag: Double?
    public var timings: [String: JSONValue]?
    public var lightCurveLoadRecovered: Bool
    public var status: String
    public var error: String?

    enum CodingKeys: String, CodingKey {
        case taskID = "task_id", ticID = "tic_id", candidates
        case vettingResults = "vetting_results", cnnScores = "cnn_scores", completeness
        case processingTimeSeconds = "processing_time_seconds", gpuUsed = "gpu_used"
        case sectorCount = "n_sectors", pointCount = "n_points", timeSpanDays = "time_span_days"
        case ra, dec, tmag, timings, lightCurveLoadRecovered = "lc_load_recovered", status, error
    }

    public init(taskID: String?, ticID: String?, status: String = "completed", error: String? = nil, candidates: [CandidateResult] = [], vettingResults: [VettingResult] = [], cnnScores: [Double?] = [], completeness: CompletenessResult? = nil, processingTimeSeconds: Double = 0, gpuUsed: Bool = false, sectorCount: Int = 0, pointCount: Int = 0, timeSpanDays: Double? = nil, ra: Double? = nil, dec: Double? = nil, tmag: Double? = nil, timings: [String: JSONValue]? = nil, lightCurveLoadRecovered: Bool = false) {
        self.taskID = taskID; self.ticID = ticID; self.status = status; self.error = error
        self.candidates = candidates; self.vettingResults = vettingResults; self.cnnScores = cnnScores
        self.completeness = completeness; self.processingTimeSeconds = processingTimeSeconds
        self.gpuUsed = gpuUsed; self.sectorCount = sectorCount; self.pointCount = pointCount
        self.timeSpanDays = timeSpanDays; self.ra = ra; self.dec = dec; self.tmag = tmag
        self.timings = timings; self.lightCurveLoadRecovered = lightCurveLoadRecovered
    }
}

public struct BackfillTaskEnvelope: Decodable, Equatable, Sendable { public let task: BackfillTask? }

public struct BackfillTask: Decodable, Equatable, Sendable {
    public let candidateID: Int64
    public let ticID: String
    public let candidateNumber: Int
    public let periodDays: Double
    public let t0: Double
    public let durationDays: Double
    public let depthPPM: Double
    public let snr: Double
    public let sectorCount: Int

    enum CodingKeys: String, CodingKey {
        case candidateID = "candidate_id", ticID = "tic_id", candidateNumber = "candidate_num"
        case periodDays = "period_days", t0, durationDays = "duration_days", depthPPM = "depth_ppm", snr
        case sectorCount = "n_sectors"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        candidateID = try c.decode(Int64.self, forKey: .candidateID)
        ticID = try decodeFlexibleString(c, forKey: .ticID)
        candidateNumber = try c.decode(Int.self, forKey: .candidateNumber)
        periodDays = try c.decode(Double.self, forKey: .periodDays); t0 = try c.decode(Double.self, forKey: .t0)
        durationDays = try c.decode(Double.self, forKey: .durationDays); depthPPM = try c.decode(Double.self, forKey: .depthPPM)
        snr = try c.decode(Double.self, forKey: .snr); sectorCount = try c.decodeIfPresent(Int.self, forKey: .sectorCount) ?? 1
    }
}

public struct FPPResultRequest: Codable, Equatable, Sendable {
    public var candidateID: Int64?; public var ticID: String?; public var fpp: Double?; public var nfpp: Double?
    public var probabilityPlanet: Double?; public var verdict: String?; public var candidate: [String: JSONValue]?
    enum CodingKeys: String, CodingKey { case candidateID = "candidate_id", ticID = "tic_id", fpp, nfpp, probabilityPlanet = "prob_planet", verdict, candidate }
    public init(candidateID: Int64?, ticID: String?, fpp: Double?, nfpp: Double? = nil, probabilityPlanet: Double? = nil, verdict: String? = nil, candidate: [String: JSONValue]? = nil) {
        self.candidateID = candidateID; self.ticID = ticID; self.fpp = fpp; self.nfpp = nfpp
        self.probabilityPlanet = probabilityPlanet; self.verdict = verdict; self.candidate = candidate
    }
}

public struct ExoMinerResultRequest: Codable, Equatable, Sendable {
    public var candidateID: Int64?; public var ticID: String?; public var score: Double?
    public var tcePeriod: Double?; public var sectorRun: String?
    enum CodingKeys: String, CodingKey { case candidateID = "candidate_id", ticID = "tic_id", score, tcePeriod = "tce_period", sectorRun = "sector_run" }
    public init(candidateID: Int64?, ticID: String?, score: Double?, tcePeriod: Double? = nil, sectorRun: String? = nil) {
        self.candidateID = candidateID; self.ticID = ticID; self.score = score; self.tcePeriod = tcePeriod; self.sectorRun = sectorRun
    }
}

public struct StatusResponse: Codable, Equatable, Sendable { public let status: String; public let outcome: String? }

public struct FFISector: Decodable, Equatable, Sendable {
    public let sector: Int; public let source: String; public let time: [Double]; public let flux: [Double]; public let quality: [Int]?
    public var effectiveQuality: [Int] { quality ?? Array(repeating: 0, count: time.count) }
}
public struct FFIResponse: Decodable, Equatable, Sendable {
    public let ticID: String; public let sectors: [FFISector]
    enum CodingKeys: String, CodingKey { case ticID = "tic_id", sectors }
    public init(from decoder: Decoder) throws { let c = try decoder.container(keyedBy: CodingKeys.self); ticID = try decodeFlexibleString(c, forKey: .ticID); sectors = try c.decode([FFISector].self, forKey: .sectors) }
}

public struct NearbyStar: Decodable, Equatable, Sendable {
    public let id: String; public let ra: Double; public let dec: Double; public let mass: Double?; public let radius: Double?
    public let effectiveTemperature: Double?; public let tessMagnitude: Double?; public let parallax: Double?; public let stellarSource: String?
    enum CodingKeys: String, CodingKey { case id = "ID", ra, dec, mass, radius = "rad", effectiveTemperature = "Teff", tessMagnitude = "Tmag", parallax = "plx", stellarSource = "stellar_source" }
    public init(from decoder: Decoder) throws { let c = try decoder.container(keyedBy: CodingKeys.self); id = try decodeFlexibleString(c, forKey: .id); ra = try c.decode(Double.self, forKey: .ra); dec = try c.decode(Double.self, forKey: .dec); mass = try c.decodeIfPresent(Double.self, forKey: .mass); radius = try c.decodeIfPresent(Double.self, forKey: .radius); effectiveTemperature = try c.decodeIfPresent(Double.self, forKey: .effectiveTemperature); tessMagnitude = try c.decodeIfPresent(Double.self, forKey: .tessMagnitude); parallax = try c.decodeIfPresent(Double.self, forKey: .parallax); stellarSource = try c.decodeIfPresent(String.self, forKey: .stellarSource) }
}
public struct NearbyResponse: Decodable, Equatable, Sendable {
    public let ticID: String; public let stars: [NearbyStar]; public let galacticLatitude: Double; public let source: String?
    enum CodingKeys: String, CodingKey { case ticID = "tic_id", stars, galacticLatitude = "gal_b", source }
    public init(from decoder: Decoder) throws { let c = try decoder.container(keyedBy: CodingKeys.self); ticID = try decodeFlexibleString(c, forKey: .ticID); stars = try c.decodeIfPresent([NearbyStar].self, forKey: .stars) ?? []; galacticLatitude = try c.decodeIfPresent(Double.self, forKey: .galacticLatitude) ?? 0; source = try c.decodeIfPresent(String.self, forKey: .source) }
}

public struct UpdateManifestFile: Codable, Equatable, Sendable { public let filename: String; public let sha256: String }
public struct UpdateManifest: Codable, Equatable, Sendable { public let files: [UpdateManifestFile] }

public struct StarMetadata: Decodable, Equatable, Sendable {
    public let ticID: String?; public let fitsFiles: [String]; public let sectors: [Int]
    public let ra: Double?; public let dec: Double?; public let tmag: Double?
    public let effectiveTemperature: Double?; public let logg: Double?
    public let stellarRadius: Double?; public let stellarMass: Double?; public let dataSource: String?
    enum CodingKeys: String, CodingKey {
        case ticID = "tic_id", fitsFiles = "fits_files", sectors, ra, dec, tmag
        case effectiveTemperature = "teff", logg
        case stellarRadius = "stellar_radius", stellarMass = "stellar_mass"
        case dataSource = "data_source"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if c.contains(.ticID) { ticID = try? decodeFlexibleString(c, forKey: .ticID) } else { ticID = nil }
        fitsFiles = try c.decodeIfPresent([String].self, forKey: .fitsFiles) ?? []
        sectors = try c.decodeIfPresent([Int].self, forKey: .sectors) ?? []
        ra = try c.decodeIfPresent(Double.self, forKey: .ra)
        dec = try c.decodeIfPresent(Double.self, forKey: .dec)
        tmag = try c.decodeIfPresent(Double.self, forKey: .tmag)
        effectiveTemperature = try c.decodeIfPresent(Double.self, forKey: .effectiveTemperature)
        logg = try c.decodeIfPresent(Double.self, forKey: .logg)
        stellarRadius = try c.decodeIfPresent(Double.self, forKey: .stellarRadius)
        stellarMass = try c.decodeIfPresent(Double.self, forKey: .stellarMass)
        dataSource = try c.decodeIfPresent(String.self, forKey: .dataSource)
    }
}

public enum TFRecordFetchResult: Equatable, Sendable {
    case ready(Data, sectorRun: String, ticID: String)
    case preprocessing(ticID: String, sectorRun: String)
    case notAvailable
}
