//! FITSReader golden parity against REAL FITS files, independent of the
//! Swift golden fixtures (which reference exact files that only exist on
//! the operator's macOS dev machine — see fits_pending_golden.rs for that gap).
//!
//! Per the task instructions: two small real files (TGLC + QLP) were copied
//! read-only from ~/exonit/worker/fits_cache*/ into tests/data/
//! (both < 75 KB), and dumped with astropy directly (tests/tools/dump_fits.py,
//! no lightkurve, no quality masking) into tests/data/*_sample_dump.json.
//! This test re-derives FITSReader's expected row-selection logic from the
//! RAW astropy dump (not from FITSReader's own output) and compares
//! bit-for-bit, so it is a genuine independent check of fits.rs's column
//! decoding + quality/flag filtering, not just a self-consistency check.

use exonit_core::fits::{self, PhotometrySource};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Dump {
    row_count: usize,
    time: Vec<f64>,
    flux: Vec<f64>,
    flux_column: String,
    #[serde(default)]
    quality: Option<Vec<i64>>,
    #[serde(default)]
    cadenceno: Option<Vec<i64>>,
    #[serde(default)]
    tess_flags: Option<Vec<i64>>,
    #[serde(default)]
    tglc_flags: Option<Vec<i64>>,
}

fn data_dir() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data").to_string()
}

fn load_dump(name: &str) -> Dump {
    let path = format!("{}/{name}", data_dir());
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

#[test]
fn tglc_sample_matches_raw_astropy_dump() {
    let dump = load_dump("tglc_sample_dump.json");
    assert_eq!(dump.flux_column, "cal_aper_flux");
    let tess_flags = dump.tess_flags.expect("dump must carry tess_flags for TGLC");
    let tglc_flags = dump.tglc_flags.expect("dump must carry tglc_flags for TGLC");

    // Independently recompute FITSReader's TGLC row filter (qa==0 && qb==0
    // && finite(t) && finite(f) && f > 0) from the raw astropy dump.
    let mut expected_time = Vec::new();
    let mut expected_flux = Vec::new();
    for i in 0..dump.row_count {
        let t = dump.time[i];
        let f = dump.flux[i];
        if tess_flags[i] == 0 && tglc_flags[i] == 0 && t.is_finite() && f.is_finite() && f > 0.0 {
            expected_time.push(t);
            expected_flux.push(f);
        }
    }

    let fits_path = format!("{}/tglc_sample.fits", data_dir());
    let result = fits::read_light_curve(Path::new(&fits_path)).expect("read TGLC sample FITS");
    assert_eq!(result.source, PhotometrySource::Tglc);
    assert_eq!(result.raw_row_count, dump.row_count);
    assert_eq!(result.time.len(), expected_time.len(), "row-selection count");
    for i in 0..expected_time.len() {
        assert_eq!(
            result.time[i].to_bits(),
            expected_time[i].to_bits(),
            "time[{i}] bit-exact vs astropy dump"
        );
        assert_eq!(
            result.flux[i].to_bits(),
            expected_flux[i].to_bits(),
            "flux[{i}] bit-exact vs astropy dump"
        );
    }
}

#[test]
fn qlp_sample_matches_raw_astropy_dump() {
    let dump = load_dump("qlp_sample_dump.json");
    assert_eq!(dump.flux_column, "sap_flux");
    let quality = dump.quality.expect("dump must carry quality for QLP");
    let cadenceno = dump.cadenceno.expect("dump must carry cadenceno for QLP");

    // lightkurve.utils.TessQualityFlags.DEFAULT_BITMASK, independently
    // hardcoded here (not imported from fits.rs) so this test doesn't just
    // check fits.rs against itself.
    const DEFAULT_QUALITY_BITMASK: i64 = 17_087;
    let mut expected_time = Vec::new();
    let mut expected_flux = Vec::new();
    let mut expected_cadence = Vec::new();
    for i in 0..dump.row_count {
        if (quality[i] & DEFAULT_QUALITY_BITMASK) != 0 {
            continue;
        }
        let t = dump.time[i];
        if t.is_nan() {
            continue;
        }
        expected_time.push(t);
        expected_flux.push(dump.flux[i]);
        expected_cadence.push(cadenceno[i]);
    }

    let fits_path = format!("{}/qlp_sample.fits", data_dir());
    let result = fits::read_light_curve(Path::new(&fits_path)).expect("read QLP sample FITS");
    assert_eq!(result.source, PhotometrySource::Qlp);
    assert_eq!(result.raw_row_count, dump.row_count);
    assert_eq!(result.time.len(), expected_time.len(), "row-selection count");
    assert_eq!(result.cadence.as_ref().map(|c| c.len()), Some(expected_time.len()));
    for i in 0..expected_time.len() {
        assert_eq!(
            result.time[i].to_bits(),
            expected_time[i].to_bits(),
            "time[{i}] bit-exact vs astropy dump"
        );
        // flux[i] can legitimately be NaN (SPOC/QLP flux is not NaN-filtered
        // by FITSReader itself, only by quality) — compare with NaN-aware
        // equality via bit pattern only when both are non-NaN, else check
        // NaN-ness matches.
        let actual_flux = result.flux[i];
        let want_flux = expected_flux[i];
        if want_flux.is_nan() {
            assert!(actual_flux.is_nan(), "flux[{i}] expected NaN, got {actual_flux}");
        } else {
            assert_eq!(actual_flux.to_bits(), want_flux.to_bits(), "flux[{i}] bit-exact vs astropy dump");
        }
        assert_eq!(result.cadence.as_ref().unwrap()[i], expected_cadence[i], "cadence[{i}]");
    }
}
