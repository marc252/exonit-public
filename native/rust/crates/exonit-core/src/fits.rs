//! Strict binary FITS light-curve reader, transliterated from
//! `Sources/ExoNitCore/FITSReader.swift`. No FITS crate is used on purpose
//! (see docs/PLAN.md — F1 rules): the Swift port reads FITS by hand for full
//! control over the exact column selection the Python worker's
//! `_read_lightcurve` performs, and this is a byte-for-byte transliteration
//! of that hand-rolled parser, not a reimplementation against a library.
//!
//! All header-derived sizes/offsets are carried as `i64` (mirroring Swift's
//! 64-bit `Int`) and only converted to `usize` at the point of slicing the
//! byte buffer, with the same overflow/negativity checks as the Swift
//! `checkedAdd`/`checkedMultiply` helpers.

use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhotometrySource {
    Spoc,
    Qlp,
    Tglc,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LightCurveData {
    pub time: Vec<f64>,
    pub flux: Vec<f64>,
    pub raw_row_count: usize,
    pub cadence: Option<Vec<i64>>,
    pub source: PhotometrySource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitsError {
    Truncated(String),
    InvalidHeader(String),
    UnsupportedColumn(String),
    MissingColumn(String),
    NoBinaryTable,
}

impl std::fmt::Display for FitsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitsError::Truncated(where_at) => write!(f, "truncated FITS at {where_at}"),
            FitsError::InvalidHeader(reason) => write!(f, "invalid FITS header: {reason}"),
            FitsError::UnsupportedColumn(form) => {
                write!(f, "unsupported FITS column format: {form}")
            }
            FitsError::MissingColumn(name) => write!(f, "missing FITS column: {name}"),
            FitsError::NoBinaryTable => write!(f, "no FITS binary table found"),
        }
    }
}

impl std::error::Error for FitsError {}

struct FitsHeader {
    values: HashMap<String, String>,
    byte_count: i64,
}

impl FitsHeader {
    fn string(&self, key: &str) -> Option<String> {
        let raw = self.values.get(key)?;
        let trimmed = raw.trim();
        if trimmed.len() >= 2 && trimmed.starts_with('\'') && trimmed.ends_with('\'') {
            Some(trimmed[1..trimmed.len() - 1].trim().to_string())
        } else {
            Some(trimmed.to_string())
        }
    }

    fn integer(&self, key: &str, default: i64) -> i64 {
        self.string(key)
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(default)
    }

    fn double(&self, key: &str, default: f64) -> f64 {
        match self.string(key) {
            Some(s) => s.replace('D', "E").parse::<f64>().unwrap_or(default),
            None => default,
        }
    }
}

#[derive(Clone)]
struct FitsColumn {
    name: String,
    code: char,
    offset: i64,
    scale: f64,
    zero: f64,
}

/// Reproduces the column selection used by `_read_lightcurve` in the Python
/// worker: calibrated aperture flux for TGLC, SAP flux for QLP and PDCSAP
/// flux for SPOC. QLP/SPOC use Lightkurve's TESS default quality bitmask.
pub fn read_light_curve(path: &Path) -> Result<LightCurveData, FitsError> {
    let data = std::fs::read(path)
        .map_err(|e| FitsError::Truncated(format!("failed to read {}: {e}", path.display())))?;
    let mut offset: i64 = 0;
    while (offset as usize) < data.len() {
        let header = read_header(&data, offset)?;
        let data_offset = checked_add(offset, header.byte_count, "HDU header offset")?;
        let extension_type = header.string("XTENSION").map(|s| s.to_uppercase());
        if extension_type.as_deref() == Some("BINTABLE") {
            let columns = table_columns(&header)?;
            let mut names: HashMap<String, &FitsColumn> = HashMap::new();
            for column in &columns {
                names.insert(column.name.to_lowercase(), column);
            }
            if names.contains_key("time") && names.contains_key("cal_aper_flux") {
                return read_tglc_table(&data, &header, data_offset, &names);
            }
            if names.contains_key("time") && names.contains_key("pdcsap_flux") && names.contains_key("quality")
            {
                return read_tess_quality_table(
                    &data,
                    &header,
                    data_offset,
                    &names,
                    "pdcsap_flux",
                    PhotometrySource::Spoc,
                );
            }
            if names.contains_key("time") && names.contains_key("sap_flux") && names.contains_key("quality") {
                return read_tess_quality_table(
                    &data,
                    &header,
                    data_offset,
                    &names,
                    "sap_flux",
                    PhotometrySource::Qlp,
                );
            }
        }
        let payload_bytes = data_byte_count(&header)?;
        offset = checked_add(data_offset, padded(payload_bytes)?, "next HDU offset")?;
    }
    Err(FitsError::NoBinaryTable)
}

pub fn read_tglc_light_curve(path: &Path) -> Result<LightCurveData, FitsError> {
    read_light_curve(path)
}

fn read_tglc_table(
    data: &[u8],
    header: &FitsHeader,
    data_offset: i64,
    names: &HashMap<String, &FitsColumn>,
) -> Result<LightCurveData, FitsError> {
    let time_column = *names
        .get("time")
        .ok_or_else(|| FitsError::MissingColumn("time".into()))?;
    let flux_column = *names
        .get("cal_aper_flux")
        .ok_or_else(|| FitsError::MissingColumn("cal_aper_flux".into()))?;
    let tess_flags = names.get("tess_flags").copied();
    let tglc_flags = names.get("tglc_flags").copied();
    let row_width = header.integer("NAXIS1", 0);
    let row_count = header.integer("NAXIS2", 0);
    if row_width <= 0 || row_count < 0 {
        return Err(FitsError::InvalidHeader("invalid NAXIS1/NAXIS2".into()));
    }
    let table_bytes = checked_multiply(row_width, row_count, "TGLC table size")?;
    let table_end = checked_add(data_offset, table_bytes, "TGLC table end")?;
    if table_end > data.len() as i64 {
        return Err(FitsError::Truncated("binary table rows".into()));
    }

    let mut time: Vec<f64> = Vec::with_capacity(row_count as usize);
    let mut flux: Vec<f64> = Vec::with_capacity(row_count as usize);
    for row in 0..row_count {
        let base = data_offset + row * row_width;
        let t = floating_value(data, base, time_column)?;
        let f = floating_value(data, base, flux_column)?;
        let qa = match tess_flags {
            Some(column) => integer_value(data, base, column)?,
            None => 0,
        };
        let qb = match tglc_flags {
            Some(column) => integer_value(data, base, column)?,
            None => 0,
        };
        if qa == 0 && qb == 0 && t.is_finite() && f.is_finite() && f > 0.0 {
            time.push(t);
            flux.push(f);
        }
    }
    Ok(LightCurveData {
        time,
        flux,
        raw_row_count: row_count as usize,
        cadence: None,
        source: PhotometrySource::Tglc,
    })
}

/// lightkurve.utils.TessQualityFlags.DEFAULT_BITMASK
const DEFAULT_QUALITY_BITMASK: i64 = 17_087;

fn read_tess_quality_table(
    data: &[u8],
    header: &FitsHeader,
    data_offset: i64,
    names: &HashMap<String, &FitsColumn>,
    flux_name: &str,
    source: PhotometrySource,
) -> Result<LightCurveData, FitsError> {
    let time_column = *names
        .get("time")
        .ok_or_else(|| FitsError::MissingColumn("time".into()))?;
    let flux_column = *names
        .get(flux_name)
        .ok_or_else(|| FitsError::MissingColumn(flux_name.into()))?;
    let quality_column = *names
        .get("quality")
        .ok_or_else(|| FitsError::MissingColumn("quality".into()))?;
    let row_width = header.integer("NAXIS1", 0);
    let row_count = header.integer("NAXIS2", 0);
    if row_width <= 0 || row_count < 0 {
        return Err(FitsError::InvalidHeader("invalid NAXIS1/NAXIS2".into()));
    }
    let table_bytes = checked_multiply(row_width, row_count, "TESS table size")?;
    let table_end = checked_add(data_offset, table_bytes, "TESS table end")?;
    if table_end > data.len() as i64 {
        return Err(FitsError::Truncated("binary table rows".into()));
    }

    let mut time: Vec<f64> = Vec::with_capacity(row_count as usize);
    let mut flux: Vec<f64> = Vec::with_capacity(row_count as usize);
    let cadence_column = names.get("cadenceno").copied();
    let mut cadence: Option<Vec<i64>> = cadence_column.map(|_| Vec::with_capacity(row_count as usize));

    for row in 0..row_count {
        let base = data_offset + row * row_width;
        let quality = integer_value(data, base, quality_column)?;
        if (quality & DEFAULT_QUALITY_BITMASK) != 0 {
            continue;
        }
        let value_time = floating_value(data, base, time_column)?;
        if value_time.is_nan() {
            continue;
        }
        time.push(value_time);
        flux.push(floating_value(data, base, flux_column)?);
        if let Some(column) = cadence_column {
            cadence.as_mut().unwrap().push(integer_value(data, base, column)?);
        }
    }
    Ok(LightCurveData {
        time,
        flux,
        raw_row_count: row_count as usize,
        cadence,
        source,
    })
}

fn read_header(data: &[u8], offset: i64) -> Result<FitsHeader, FitsError> {
    if offset < 0 || offset.checked_add(80).map_or(true, |v| v as usize > data.len()) {
        return Err(FitsError::Truncated("header".into()));
    }
    let mut values: HashMap<String, String> = HashMap::new();
    let mut card_offset = offset;
    let mut found_end = false;
    while (card_offset + 80) as usize <= data.len() {
        let start = card_offset as usize;
        let card_bytes = &data[start..start + 80];
        if !card_bytes.is_ascii() {
            return Err(FitsError::InvalidHeader("non-ASCII card".into()));
        }
        // Safe: `is_ascii()` was just checked above.
        let card = std::str::from_utf8(card_bytes).unwrap();
        let keyword = card[0..8].trim();
        if keyword == "END" {
            found_end = true;
            card_offset += 80;
            break;
        }
        if card.len() >= 10 && card.as_bytes()[8] == b'=' {
            values.insert(keyword.to_string(), value_part(&card[10..]));
        }
        card_offset += 80;
    }
    if !found_end {
        return Err(FitsError::Truncated("END card".into()));
    }
    let used = card_offset - offset;
    Ok(FitsHeader {
        values,
        byte_count: padded(used)?,
    })
}

fn value_part(raw: &str) -> String {
    let mut in_string = false;
    let mut previous: Option<char> = None;
    for (index, character) in raw.char_indices() {
        if character == '\'' && previous != Some('\'') {
            in_string = !in_string;
        }
        if character == '/' && !in_string {
            return raw[..index].trim().to_string();
        }
        previous = Some(character);
    }
    raw.trim().to_string()
}

fn table_columns(header: &FitsHeader) -> Result<Vec<FitsColumn>, FitsError> {
    let count = header.integer("TFIELDS", 0);
    if count < 0 {
        return Err(FitsError::InvalidHeader("negative TFIELDS".into()));
    }
    let mut columns: Vec<FitsColumn> = Vec::with_capacity(count as usize);
    let mut offset: i64 = 0;
    for index in 1..=count {
        let name = header
            .string(&format!("TTYPE{index}"))
            .unwrap_or_else(|| format!("COLUMN{index}"));
        let form = header
            .string(&format!("TFORM{index}"))
            .ok_or_else(|| FitsError::InvalidHeader(format!("missing TFORM{index}")))?
            .to_uppercase();
        let (repeat_count, code, element_width) = parse_form(&form)?;
        let width = checked_multiply(repeat_count, element_width, &format!("TFORM{index} width"))?;
        columns.push(FitsColumn {
            name,
            code,
            offset,
            scale: header.double(&format!("TSCAL{index}"), 1.0),
            zero: header.double(&format!("TZERO{index}"), 0.0),
        });
        offset = checked_add(offset, width, "column offsets")?;
    }
    let row_width = header.integer("NAXIS1", 0);
    if row_width < 0 || offset > row_width {
        return Err(FitsError::InvalidHeader("columns exceed NAXIS1".into()));
    }
    Ok(columns)
}

fn parse_form(form: &str) -> Result<(i64, char, i64), FitsError> {
    let mut digits = String::new();
    let mut code: Option<char> = None;
    for character in form.chars() {
        if character.is_whitespace() {
            continue;
        }
        if character.is_ascii_digit() && code.is_none() {
            digits.push(character);
        } else if code.is_none() {
            code = Some(character);
            break;
        }
    }
    let code = code.ok_or_else(|| FitsError::UnsupportedColumn(form.to_string()))?;
    let repeat_count: i64 = if digits.is_empty() {
        1
    } else {
        let parsed: i64 = digits
            .parse()
            .map_err(|_| FitsError::UnsupportedColumn(form.to_string()))?;
        if parsed <= 0 {
            return Err(FitsError::UnsupportedColumn(form.to_string()));
        }
        parsed
    };
    let width: i64 = match code {
        'L' | 'B' | 'A' => 1,
        'I' => 2,
        'J' | 'E' => 4,
        'K' | 'D' | 'C' => 8,
        'M' | 'Q' => 16,
        'P' => 8,
        _ => return Err(FitsError::UnsupportedColumn(form.to_string())),
    };
    Ok((repeat_count, code, width))
}

fn floating_value(data: &[u8], row_offset: i64, column: &FitsColumn) -> Result<f64, FitsError> {
    let offset = row_offset + column.offset;
    let raw = match column.code {
        'D' => f64::from_bits(read_u64(data, offset)?),
        'E' => f32::from_bits(read_u32(data, offset)?) as f64,
        'I' | 'J' | 'K' | 'B' => integer_value(data, row_offset, column)? as f64,
        other => return Err(FitsError::UnsupportedColumn(other.to_string())),
    };
    Ok(raw * column.scale + column.zero)
}

fn integer_value(data: &[u8], row_offset: i64, column: &FitsColumn) -> Result<i64, FitsError> {
    let offset = row_offset + column.offset;
    let raw: i64 = match column.code {
        'B' => read_u8(data, offset)? as i64,
        'I' => (read_u16(data, offset)? as i16) as i64,
        'J' => (read_u32(data, offset)? as i32) as i64,
        'K' => read_u64(data, offset)? as i64,
        other => return Err(FitsError::UnsupportedColumn(other.to_string())),
    };
    // Swift: `Int64(Double(raw) * column.scale + column.zero)` — truncates
    // toward zero and traps on overflow/NaN. Rust's `as i64` on f64
    // saturates instead of trapping (stable, well-defined since Rust 1.45);
    // this only differs from Swift for pathological TSCAL/TZERO values that
    // push the result out of i64 range, which none of the real quality/
    // cadence columns in this pipeline ever do (scale=1, zero=0).
    Ok((raw as f64 * column.scale + column.zero) as i64)
}

fn data_byte_count(header: &FitsHeader) -> Result<i64, FitsError> {
    let axis_count = header.integer("NAXIS", 0);
    if axis_count < 0 {
        return Err(FitsError::InvalidHeader("negative NAXIS".into()));
    }
    if header.string("XTENSION").map(|s| s.to_uppercase()).as_deref() == Some("BINTABLE") {
        let row_width = header.integer("NAXIS1", 0);
        let row_count = header.integer("NAXIS2", 0);
        let heap = header.integer("PCOUNT", 0);
        if row_width < 0 || row_count < 0 || heap < 0 {
            return Err(FitsError::InvalidHeader("negative binary-table dimensions".into()));
        }
        return checked_add(
            checked_multiply(row_width, row_count, "binary-table payload")?,
            heap,
            "binary-table heap",
        );
    }
    if axis_count == 0 {
        return Ok(0);
    }
    let bitpix = header.integer("BITPIX", 0);
    if ![8, 16, 32, 64, -32, -64].contains(&bitpix) {
        return Err(FitsError::InvalidHeader("BITPIX".into()));
    }
    let bytes_per_element = bitpix.abs() / 8;
    let mut elements: i64 = 1;
    for axis in 1..=axis_count {
        let dimension = header.integer(&format!("NAXIS{axis}"), 0);
        if dimension < 0 {
            return Err(FitsError::InvalidHeader(format!("negative NAXIS{axis}")));
        }
        elements = checked_multiply(elements, dimension, "image dimensions")?;
    }
    let data_bytes = checked_multiply(elements, bytes_per_element, "image payload")?;
    let heap = header.integer("PCOUNT", 0);
    let groups = header.integer("GCOUNT", 1);
    if heap < 0 || groups < 1 {
        return Err(FitsError::InvalidHeader("PCOUNT/GCOUNT".into()));
    }
    checked_multiply(
        checked_add(data_bytes, heap, "image heap")?,
        groups,
        "random groups",
    )
}

fn checked_add(lhs: i64, rhs: i64, context: &str) -> Result<i64, FitsError> {
    match lhs.checked_add(rhs) {
        Some(v) if v >= 0 => Ok(v),
        _ => Err(FitsError::InvalidHeader(format!("integer overflow in {context}"))),
    }
}

fn checked_multiply(lhs: i64, rhs: i64, context: &str) -> Result<i64, FitsError> {
    match lhs.checked_mul(rhs) {
        Some(v) if v >= 0 => Ok(v),
        _ => Err(FitsError::InvalidHeader(format!("integer overflow in {context}"))),
    }
}

fn padded(count: i64) -> Result<i64, FitsError> {
    let rounded = checked_add(count, 2_879, "FITS block padding")? / 2_880;
    checked_multiply(rounded, 2_880, "FITS block padding")
}

fn read_u8(data: &[u8], offset: i64) -> Result<u8, FitsError> {
    if offset < 0 || offset as usize >= data.len() {
        return Err(FitsError::Truncated("UInt8".into()));
    }
    Ok(data[offset as usize])
}

fn read_u16(data: &[u8], offset: i64) -> Result<u16, FitsError> {
    if offset < 0 || offset as usize + 2 > data.len() {
        return Err(FitsError::Truncated("UInt16".into()));
    }
    let o = offset as usize;
    Ok(u16::from_be_bytes([data[o], data[o + 1]]))
}

fn read_u32(data: &[u8], offset: i64) -> Result<u32, FitsError> {
    if offset < 0 || offset as usize + 4 > data.len() {
        return Err(FitsError::Truncated("UInt32".into()));
    }
    let o = offset as usize;
    Ok(u32::from_be_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]))
}

fn read_u64(data: &[u8], offset: i64) -> Result<u64, FitsError> {
    if offset < 0 || offset as usize + 8 > data.len() {
        return Err(FitsError::Truncated("UInt64".into()));
    }
    let o = offset as usize;
    Ok(u64::from_be_bytes([
        data[o],
        data[o + 1],
        data[o + 2],
        data[o + 3],
        data[o + 4],
        data[o + 5],
        data[o + 6],
        data[o + 7],
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_part_strips_trailing_comment_outside_string() {
        assert_eq!(value_part("'BINTABLE'           / binary table extension"), "'BINTABLE'");
        assert_eq!(value_part("                    3 / no comment marker inside"), "3");
    }

    #[test]
    fn value_part_ignores_slash_inside_quoted_string() {
        assert_eq!(value_part("'a/b'                / a string containing a slash"), "'a/b'");
    }

    #[test]
    fn parse_form_handles_repeat_and_bare_forms() {
        assert_eq!(parse_form("1D").unwrap(), (1, 'D', 8));
        assert_eq!(parse_form("D").unwrap(), (1, 'D', 8));
        assert_eq!(parse_form("24A").unwrap(), (24, 'A', 1));
        assert_eq!(parse_form("1E").unwrap(), (1, 'E', 4));
        assert!(parse_form("").is_err());
    }

    #[test]
    fn padded_rounds_up_to_2880_multiple() {
        assert_eq!(padded(0).unwrap(), 0);
        assert_eq!(padded(1).unwrap(), 2_880);
        assert_eq!(padded(2_880).unwrap(), 2_880);
        assert_eq!(padded(2_881).unwrap(), 5_760);
    }

    #[test]
    fn missing_file_is_a_truncated_error_not_a_panic() {
        let result = read_light_curve(Path::new("/nonexistent/path/does-not-exist.fits"));
        assert!(matches!(result, Err(FitsError::Truncated(_))));
    }
}
