import Foundation

public enum PhotometrySource: String, Equatable, Sendable {
    case spoc
    case qlp
    case tglc
    case unknown
}

public struct LightCurveData: Equatable, Sendable {
    public let time: [Double]
    public let flux: [Double]
    public let rawRowCount: Int
    public let cadence: [Int64]?
    public let source: PhotometrySource

    public init(
        time: [Double], flux: [Double], rawRowCount: Int,
        cadence: [Int64]? = nil, source: PhotometrySource = .unknown
    ) {
        self.time = time
        self.flux = flux
        self.rawRowCount = rawRowCount
        self.cadence = cadence
        self.source = source
    }
}

public enum FITSError: Error, CustomStringConvertible, Sendable {
    case truncated(String)
    case invalidHeader(String)
    case unsupportedColumn(String)
    case missingColumn(String)
    case noBinaryTable

    public var description: String {
        switch self {
        case let .truncated(whereAt): return "truncated FITS at \(whereAt)"
        case let .invalidHeader(reason): return "invalid FITS header: \(reason)"
        case let .unsupportedColumn(form): return "unsupported FITS column format: \(form)"
        case let .missingColumn(name): return "missing FITS column: \(name)"
        case .noBinaryTable: return "no FITS binary table found"
        }
    }
}

private struct FITSHeader {
    let values: [String: String]
    let byteCount: Int

    func string(_ key: String) -> String? {
        guard let raw = values[key] else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        if trimmed.count >= 2, trimmed.first == "'", trimmed.last == "'" {
            return String(trimmed.dropFirst().dropLast()).trimmingCharacters(in: .whitespaces)
        }
        return trimmed
    }

    func integer(_ key: String, default fallback: Int = 0) -> Int {
        Int(string(key) ?? "") ?? fallback
    }

    func double(_ key: String, default fallback: Double = 0) -> Double {
        Double((string(key) ?? "").replacingOccurrences(of: "D", with: "E")) ?? fallback
    }
}

private struct FITSColumn {
    let name: String
    let code: Character
    let repeatCount: Int
    let offset: Int
    let byteWidth: Int
    let scale: Double
    let zero: Double
}

public enum FITSReader {
    /// Reproduces the column selection used by `_read_lightcurve` in the Python
    /// worker: calibrated aperture flux for TGLC, SAP flux for QLP and PDCSAP
    /// flux for SPOC. QLP/SPOC use Lightkurve's TESS default quality bitmask.
    public static func readLightCurve(at url: URL) throws -> LightCurveData {
        let data = try Data(contentsOf: url, options: [.mappedIfSafe])
        var offset = 0
        while offset < data.count {
            let header = try readHeader(data, offset: offset)
            let dataOffset = try checkedAdd(offset, header.byteCount, context: "HDU header offset")
            let extensionType = header.string("XTENSION")?.uppercased()
            if extensionType == "BINTABLE" {
                let columns = try tableColumns(header)
                let names = Dictionary(uniqueKeysWithValues: columns.map { ($0.name.lowercased(), $0) })
                if names["time"] != nil, names["cal_aper_flux"] != nil {
                    return try readTGLCTable(data, header: header, dataOffset: dataOffset, names: names)
                }
                if names["time"] != nil, names["pdcsap_flux"] != nil, names["quality"] != nil {
                    return try readTESSQualityTable(
                        data, header: header, dataOffset: dataOffset, names: names,
                        fluxName: "pdcsap_flux", source: .spoc
                    )
                }
                if names["time"] != nil, names["sap_flux"] != nil, names["quality"] != nil {
                    return try readTESSQualityTable(
                        data, header: header, dataOffset: dataOffset, names: names,
                        fluxName: "sap_flux", source: .qlp
                    )
                }
            }
            let payloadBytes = try dataByteCount(header)
            offset = try checkedAdd(
                dataOffset, try padded(payloadBytes), context: "next HDU offset"
            )
        }
        throw FITSError.noBinaryTable
    }

    public static func readTGLCLightCurve(at url: URL) throws -> LightCurveData {
        try readLightCurve(at: url)
    }

    private static func readTGLCTable(
        _ data: Data,
        header: FITSHeader,
        dataOffset: Int,
        names: [String: FITSColumn]
    ) throws -> LightCurveData {
        guard let timeColumn = names["time"] else { throw FITSError.missingColumn("time") }
        guard let fluxColumn = names["cal_aper_flux"] else {
            throw FITSError.missingColumn("cal_aper_flux")
        }
        let tessFlags = names["tess_flags"]
        let tglcFlags = names["tglc_flags"]
        let rowWidth = header.integer("NAXIS1")
        let rowCount = header.integer("NAXIS2")
        guard rowWidth > 0, rowCount >= 0 else {
            throw FITSError.invalidHeader("invalid NAXIS1/NAXIS2")
        }
        let tableBytes = try checkedMultiply(rowWidth, rowCount, context: "TGLC table size")
        let tableEnd = try checkedAdd(dataOffset, tableBytes, context: "TGLC table end")
        guard tableEnd <= data.count else {
            throw FITSError.truncated("binary table rows")
        }

        var time: [Double] = []
        var flux: [Double] = []
        time.reserveCapacity(rowCount)
        flux.reserveCapacity(rowCount)
        for row in 0..<rowCount {
            let base = dataOffset + row * rowWidth
            let t = try floatingValue(data, at: base, column: timeColumn)
            let f = try floatingValue(data, at: base, column: fluxColumn)
            let qa = try tessFlags.map { try integerValue(data, at: base, column: $0) } ?? 0
            let qb = try tglcFlags.map { try integerValue(data, at: base, column: $0) } ?? 0
            if qa == 0, qb == 0, t.isFinite, f.isFinite, f > 0 {
                time.append(t)
                flux.append(f)
            }
        }
        return LightCurveData(
            time: time, flux: flux, rawRowCount: rowCount, source: .tglc
        )
    }

    private static func readTESSQualityTable(
        _ data: Data,
        header: FITSHeader,
        dataOffset: Int,
        names: [String: FITSColumn],
        fluxName: String,
        source: PhotometrySource
    ) throws -> LightCurveData {
        guard let timeColumn = names["time"] else { throw FITSError.missingColumn("time") }
        guard let fluxColumn = names[fluxName] else { throw FITSError.missingColumn(fluxName) }
        guard let qualityColumn = names["quality"] else { throw FITSError.missingColumn("quality") }
        let rowWidth = header.integer("NAXIS1")
        let rowCount = header.integer("NAXIS2")
        guard rowWidth > 0, rowCount >= 0 else {
            throw FITSError.invalidHeader("invalid NAXIS1/NAXIS2")
        }
        let tableBytes = try checkedMultiply(rowWidth, rowCount, context: "TESS table size")
        let tableEnd = try checkedAdd(dataOffset, tableBytes, context: "TESS table end")
        guard tableEnd <= data.count else {
            throw FITSError.truncated("binary table rows")
        }

        // lightkurve.utils.TessQualityFlags.DEFAULT_BITMASK
        let defaultQualityBitmask: Int64 = 17_087
        var time: [Double] = []
        var flux: [Double] = []
        let cadenceColumn = names["cadenceno"]
        var cadence: [Int64]? = cadenceColumn == nil ? nil : []
        time.reserveCapacity(rowCount)
        flux.reserveCapacity(rowCount)
        cadence?.reserveCapacity(rowCount)
        for row in 0..<rowCount {
            let base = dataOffset + row * rowWidth
            let quality = try integerValue(data, at: base, column: qualityColumn)
            guard (quality & defaultQualityBitmask) == 0 else { continue }
            let valueTime = try floatingValue(data, at: base, column: timeColumn)
            guard !valueTime.isNaN else { continue }
            time.append(valueTime)
            flux.append(try floatingValue(data, at: base, column: fluxColumn))
            if let cadenceColumn {
                cadence?.append(try integerValue(data, at: base, column: cadenceColumn))
            }
        }
        return LightCurveData(
            time: time, flux: flux, rawRowCount: rowCount,
            cadence: cadence, source: source
        )
    }

    private static func readHeader(_ data: Data, offset: Int) throws -> FITSHeader {
        guard offset >= 0, offset + 80 <= data.count else {
            throw FITSError.truncated("header")
        }
        var values: [String: String] = [:]
        var cardOffset = offset
        var foundEnd = false
        while cardOffset + 80 <= data.count {
            let cardData = data[cardOffset..<(cardOffset + 80)]
            guard let card = String(data: cardData, encoding: .ascii) else {
                throw FITSError.invalidHeader("non-ASCII card")
            }
            let keyword = String(card.prefix(8)).trimmingCharacters(in: .whitespaces)
            if keyword == "END" {
                foundEnd = true
                cardOffset += 80
                break
            }
            if card.count >= 10, card[card.index(card.startIndex, offsetBy: 8)] == "=" {
                let start = card.index(card.startIndex, offsetBy: 10)
                values[keyword] = valuePart(String(card[start...]))
            }
            cardOffset += 80
        }
        guard foundEnd else { throw FITSError.truncated("END card") }
        let used = cardOffset - offset
        return FITSHeader(values: values, byteCount: try padded(used))
    }

    private static func valuePart(_ raw: String) -> String {
        var inString = false
        var previous: Character?
        for index in raw.indices {
            let character = raw[index]
            if character == "'", previous != "'" { inString.toggle() }
            if character == "/", !inString {
                return String(raw[..<index]).trimmingCharacters(in: .whitespaces)
            }
            previous = character
        }
        return raw.trimmingCharacters(in: .whitespaces)
    }

    private static func tableColumns(_ header: FITSHeader) throws -> [FITSColumn] {
        let count = header.integer("TFIELDS")
        guard count >= 0 else { throw FITSError.invalidHeader("negative TFIELDS") }
        var columns: [FITSColumn] = []
        columns.reserveCapacity(count)
        var offset = 0
        if count > 0 {
            for index in 1...count {
                let name = header.string("TTYPE\(index)") ?? "COLUMN\(index)"
                guard let form = header.string("TFORM\(index)")?.uppercased() else {
                    throw FITSError.invalidHeader("missing TFORM\(index)")
                }
                let parsed = try parseForm(form)
                let width = try checkedMultiply(
                    parsed.repeatCount, parsed.elementWidth, context: "TFORM\(index) width"
                )
                columns.append(FITSColumn(
                    name: name,
                    code: parsed.code,
                    repeatCount: parsed.repeatCount,
                    offset: offset,
                    byteWidth: width,
                    scale: header.double("TSCAL\(index)", default: 1),
                    zero: header.double("TZERO\(index)", default: 0)
                ))
                offset = try checkedAdd(offset, width, context: "column offsets")
            }
        }
        let rowWidth = header.integer("NAXIS1")
        guard rowWidth >= 0, offset <= rowWidth else {
            throw FITSError.invalidHeader("columns exceed NAXIS1")
        }
        return columns
    }

    private static func parseForm(_ form: String) throws -> (repeatCount: Int, code: Character, elementWidth: Int) {
        var digits = ""
        var code: Character?
        for character in form where !character.isWhitespace {
            if character.isNumber, code == nil { digits.append(character) }
            else if code == nil { code = character; break }
        }
        guard let code else { throw FITSError.unsupportedColumn(form) }
        let repeatCount: Int
        if digits.isEmpty { repeatCount = 1 }
        else {
            guard let parsed = Int(digits), parsed > 0 else { throw FITSError.unsupportedColumn(form) }
            repeatCount = parsed
        }
        let width: Int
        switch code {
        case "L", "B", "A": width = 1
        case "I": width = 2
        case "J", "E": width = 4
        case "K", "D", "C": width = 8
        case "M", "Q": width = 16
        case "P": width = 8
        default: throw FITSError.unsupportedColumn(form)
        }
        return (repeatCount, code, width)
    }

    private static func floatingValue(_ data: Data, at rowOffset: Int, column: FITSColumn) throws -> Double {
        let offset = rowOffset + column.offset
        let raw: Double
        switch column.code {
        case "D":
            let bits = try readUInt64(data, at: offset)
            raw = Double(bitPattern: bits)
        case "E":
            let bits = try readUInt32(data, at: offset)
            raw = Double(Float(bitPattern: bits))
        case "I", "J", "K", "B":
            raw = Double(try integerValue(data, at: rowOffset, column: column))
        default:
            throw FITSError.unsupportedColumn(String(column.code))
        }
        return raw * column.scale + column.zero
    }

    private static func integerValue(_ data: Data, at rowOffset: Int, column: FITSColumn) throws -> Int64 {
        let offset = rowOffset + column.offset
        let raw: Int64
        switch column.code {
        case "B": raw = Int64(try readUInt8(data, at: offset))
        case "I": raw = Int64(Int16(bitPattern: try readUInt16(data, at: offset)))
        case "J": raw = Int64(Int32(bitPattern: try readUInt32(data, at: offset)))
        case "K": raw = Int64(bitPattern: try readUInt64(data, at: offset))
        default: throw FITSError.unsupportedColumn(String(column.code))
        }
        return Int64(Double(raw) * column.scale + column.zero)
    }

    private static func dataByteCount(_ header: FITSHeader) throws -> Int {
        let axisCount = header.integer("NAXIS")
        guard axisCount >= 0 else { throw FITSError.invalidHeader("negative NAXIS") }
        if header.string("XTENSION")?.uppercased() == "BINTABLE" {
            let rowWidth = header.integer("NAXIS1")
            let rowCount = header.integer("NAXIS2")
            let heap = header.integer("PCOUNT")
            guard rowWidth >= 0, rowCount >= 0, heap >= 0 else {
                throw FITSError.invalidHeader("negative binary-table dimensions")
            }
            return try checkedAdd(
                try checkedMultiply(rowWidth, rowCount, context: "binary-table payload"),
                heap, context: "binary-table heap"
            )
        }
        if axisCount == 0 { return 0 }
        let bitpix = header.integer("BITPIX")
        guard [8, 16, 32, 64, -32, -64].contains(bitpix) else {
            throw FITSError.invalidHeader("BITPIX")
        }
        let bytesPerElement = abs(bitpix) / 8
        var elements = 1
        for axis in 1...axisCount {
            let dimension = header.integer("NAXIS\(axis)")
            guard dimension >= 0 else { throw FITSError.invalidHeader("negative NAXIS\(axis)") }
            elements = try checkedMultiply(elements, dimension, context: "image dimensions")
        }
        let data = try checkedMultiply(elements, bytesPerElement, context: "image payload")
        let heap = header.integer("PCOUNT")
        let groups = header.integer("GCOUNT", default: 1)
        guard heap >= 0, groups >= 1 else { throw FITSError.invalidHeader("PCOUNT/GCOUNT") }
        return try checkedMultiply(
            try checkedAdd(data, heap, context: "image heap"), groups, context: "random groups"
        )
    }

    private static func checkedAdd(_ lhs: Int, _ rhs: Int, context: String) throws -> Int {
        let result = lhs.addingReportingOverflow(rhs)
        guard !result.overflow, result.partialValue >= 0 else {
            throw FITSError.invalidHeader("integer overflow in \(context)")
        }
        return result.partialValue
    }

    private static func checkedMultiply(_ lhs: Int, _ rhs: Int, context: String) throws -> Int {
        let result = lhs.multipliedReportingOverflow(by: rhs)
        guard !result.overflow, result.partialValue >= 0 else {
            throw FITSError.invalidHeader("integer overflow in \(context)")
        }
        return result.partialValue
    }

    private static func padded(_ count: Int) throws -> Int {
        let rounded = try checkedAdd(count, 2_879, context: "FITS block padding") / 2_880
        return try checkedMultiply(rounded, 2_880, context: "FITS block padding")
    }

    private static func readUInt8(_ data: Data, at offset: Int) throws -> UInt8 {
        guard offset < data.count else { throw FITSError.truncated("UInt8") }
        return data[offset]
    }

    private static func readUInt16(_ data: Data, at offset: Int) throws -> UInt16 {
        guard offset + 2 <= data.count else { throw FITSError.truncated("UInt16") }
        return data.withUnsafeBytes { UInt16(bigEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt16.self)) }
    }

    private static func readUInt32(_ data: Data, at offset: Int) throws -> UInt32 {
        guard offset + 4 <= data.count else { throw FITSError.truncated("UInt32") }
        return data.withUnsafeBytes { UInt32(bigEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) }
    }

    private static func readUInt64(_ data: Data, at offset: Int) throws -> UInt64 {
        guard offset + 8 <= data.count else { throw FITSError.truncated("UInt64") }
        return data.withUnsafeBytes { UInt64(bigEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt64.self)) }
    }
}
