//! Golden parity tests vs `fixtures/qlp_pipeline_golden.json` and
//! `fixtures/tglc_10756829_preprocess_golden.json`, mirroring
//! `ExoNitContractTests.testQLPPipelineGoldenParity` and
//! `.testTGLCPreprocessGoldenParity` (Sources/ExoNitContractTests/main.swift).
//!
//! The coordinator copied the 4 exact source FITS files from the operator's macOS
//! dev machine into `tests/data/` on 2026-08-04 (previously only available
//! there — see the removed `#[ignore]` history in this file's git log):
//!   - hlsp_qlp_tess_ffi_s0013-0000000300868410_tess_v01_llc.fits
//!   - hlsp_qlp_tess_ffi_s0039-0000000300868410_tess_v01_llc.fits
//!   - hlsp_tglc_tess_ffi_gaiaid-2080664784495597312-s0014-cam1-ccd1_tess_v1_llc.fits
//!   - hlsp_tglc_tess_ffi_gaiaid-2080664784495597312-s0015-cam1-ccd2_tess_v1_llc.fits
//!
//! **F2 update (2026-08-05):** both tests below now run the real coarse BLS
//! (`bls::quick::RealQuickBlsSearch`, wired through `BLSCPUOptimized`'s Rust
//! port) instead of the fixture-injected `quick` value the F1 version of
//! this file used. F1's design note (preserved in git history) explained
//! that the two fixtures do NOT embed the intermediate `quick` result the
//! way `iterative_detrend_golden.json` does, so it was derived once,
//! offline, from the authoritative Python reference
//! (`reference-python/bls_fast.py` + `wotan_detrend.py` in the Swift repo,
//! read-only) and injected via a `FixtureQuickSearch` stub — see the
//! derivation transcript further down for how those offline values were
//! obtained, they are now used only as the parity check in the doc comments
//! on each test below.
//!
//! **Real-BLS-vs-offline-Python parity, observed (Linux/glibc, this repo):**
//! - TGLC (TIC 10756829): `RealQuickBlsSearch`'s output is **bit-identical**
//!   to the offline-derived value on every field (period/power/duration/t0/
//!   depth/snr) — verified via `assert_eq!`-strength comparison during
//!   development (not repeated as a standing assertion below, since the
//!   golden-array comparison after it already exercises the pipeline built
//!   on top of that quick result).
//! - QLP (TIC 300868410): period/duration/t0 are bit-identical; power/depth/
//!   snr differ from the offline value by ~1e-15 absolute (power/depth) and
//!   ~2e-12 absolute (snr) — consistent with cross-language/cross-host
//!   summation-order drift on a value that itself sums ~1300 flux samples,
//!   not a bug (see DESIGN.md "Determinism and parity": "glibc libm ≠
//!   macOS: tolerances are revalidated on Linux"). period/t0/duration are
//!   what drive the downstream transit mask (`duration_fraction =
//!   duration/period`, phase against `t0`) — since those three are exact,
//!   the mask and the refined detrend built on it are IDENTICAL to the
//!   fixture-injected-quick version, so the golden-array assertions below
//!   are exercising the real thing with no tolerance widening.
//!
//! Derivation reproduction steps for the offline `quick` values referenced
//! above (both used `exonit-core/examples/dump_prepare` — since removed —
//! to confirm `preprocess::prepare()` is bit-exact against `lightkurve`'s
//! stitch/clean for these exact files before deriving `quick`):
//!
//! ```text
//! source ~/.cargo/env && cargo run --release --example dump_prepare -- \
//!   tests/data/hlsp_qlp_tess_ffi_s0013-0000000300868410_tess_v01_llc.fits \
//!   tests/data/hlsp_qlp_tess_ffi_s0039-0000000300868410_tess_v01_llc.fits \
//!   > /tmp/qlp_prepare.json
//! # then, in Python (reference-python on sys.path, read-only import):
//! #   t_raw, f_raw = prepare()'s time/raw (confirmed bit-exact vs
//! #   lk.LightCurveCollection(...).stitch().remove_nans().remove_outliers(...))
//! #   flat = detrend_with_wotan(t_raw, f_raw)
//! #   quick = bls_fast(t_raw[finite], flat[finite], period_min=0.5,
//! #                     period_max=50.0, n_periods=2000, n_bins=100, n_dur_steps=5)
//! ```
//! Same procedure for the TGLC pair, using `worker.py`'s `_read_lightcurve`
//! (hand-rolled TGLC column selection) instead of plain `lk.read`.

use exonit_core::bls::quick::RealQuickBlsSearch;
use exonit_core::detrend_iterative::{self, QuickBlsResult};
use exonit_core::fits::{self, PhotometrySource};
use exonit_core::preprocess::{self, DetrendMode};
use serde::Deserialize;
use std::path::{Path, PathBuf};

fn data_dir() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data"))
}

/// Golden `paths` entries point at files that only exist on the operator's Mac; this
/// maps each to its same-basename copy under `tests/data/`.
fn local_path_for(golden_path: &str) -> PathBuf {
    let basename = Path::new(golden_path)
        .file_name()
        .unwrap_or_else(|| panic!("golden path has no basename: {golden_path}"));
    let local = data_dir().join(basename);
    assert!(
        local.exists(),
        "expected local copy of {golden_path} at {} (see this file's doc comment)",
        local.display()
    );
    local
}

#[derive(Deserialize)]
struct QlpPipelineGolden {
    paths: Vec<String>,
    max_gap: f64,
    time: Vec<f64>,
    raw: Vec<f64>,
    flat: Vec<f64>,
}

/// Mirrors ExoNitContractTests.testQLPPipelineGoldenParity (main.swift:1545),
/// now driven by the real F2 BLS (see module doc comment for the observed
/// real-vs-offline-Python quick-BLS deviation on this fixture, and why it
/// does not affect the assertions below).
#[test]
fn qlp_pipeline_matches_swift_golden() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/qlp_pipeline_golden.json"
    );
    let golden: QlpPipelineGolden =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read qlp_pipeline_golden.json"))
            .expect("parse qlp_pipeline_golden.json");

    let curves: Vec<_> = golden
        .paths
        .iter()
        .map(|p| fits::read_light_curve(&local_path_for(p)).expect("read QLP FITS"))
        .collect();
    assert!(curves.iter().all(|c| c.source == PhotometrySource::Qlp));

    // Real BLS quick search (F2): verify it lands at (or extremely near) the
    // offline Python-derived value before trusting it to drive the
    // downstream masking/refine — see module doc comment for the exact
    // observed deviation (period/t0/duration bit-identical; power/depth/snr
    // off by ~1e-15/~1e-15/~2e-12 respectively, cross-language summation
    // drift on ~1300 samples).
    let prepared = preprocess::prepare(&curves).expect("prepare QLP curves");
    let quick_check = detrend_iterative::flatten(
        &prepared.time,
        &prepared.raw,
        &RealQuickBlsSearch::new(1),
        detrend_iterative::DEFAULT_SNR_THRESHOLD,
    )
    .expect("iterative flatten (real BLS)");
    let expected_quick = QuickBlsResult {
        period: 3.6161360347322726,
        power: 0.04979298734936079,
        duration: 0.03616136034732273,
        t0: 1655.9193102749518,
        depth: 0.03341578858032046,
        snr: 24.35084134704743,
    };
    assert!(
        (quick_check.quick_bls.period - expected_quick.period).abs() < 1e-12,
        "real BLS quick period vs offline Python: actual={} expected={}",
        quick_check.quick_bls.period,
        expected_quick.period
    );
    assert!(
        (quick_check.quick_bls.t0 - expected_quick.t0).abs() < 1e-9,
        "real BLS quick t0 vs offline Python: actual={} expected={}",
        quick_check.quick_bls.t0,
        expected_quick.t0
    );
    assert!(
        (quick_check.quick_bls.duration - expected_quick.duration).abs() < 1e-12,
        "real BLS quick duration vs offline Python: actual={} expected={}",
        quick_check.quick_bls.duration,
        expected_quick.duration
    );
    assert!(
        (quick_check.quick_bls.snr - expected_quick.snr).abs() < 1e-9,
        "real BLS quick SNR vs offline Python: actual={} expected={}",
        quick_check.quick_bls.snr,
        expected_quick.snr
    );

    let search = RealQuickBlsSearch::new(1);
    let result = preprocess::process(&curves, DetrendMode::Iterative(&search)).expect("process QLP pipeline");

    assert_eq!(result.time.len(), golden.time.len(), "pipeline count");
    // Tolerance mirrors ExoNitContractTests main.swift:1551.
    assert!(
        (result.max_gap_days - golden.max_gap).abs() < 1e-12,
        "pipeline max gap actual={} expected={}",
        result.max_gap_days,
        golden.max_gap
    );
    for i in 0..result.time.len() {
        // Tolerances mirror main.swift:1553-1555.
        assert!(
            (result.time[i] - golden.time[i]).abs() < 1e-12,
            "pipeline time[{i}] actual={} expected={}",
            result.time[i],
            golden.time[i]
        );
        assert!(
            (result.raw_flux[i] - golden.raw[i]).abs() < 2e-13,
            "pipeline raw[{i}] actual={} expected={}",
            result.raw_flux[i],
            golden.raw[i]
        );
        assert!(
            (result.flat_flux[i] - golden.flat[i]).abs() < 2e-12,
            "pipeline flat[{i}] actual={} expected={}",
            result.flat_flux[i],
            golden.flat[i]
        );
    }
}

#[derive(Deserialize)]
struct TglcPreprocessGolden {
    paths: Vec<String>,
    count: usize,
    time_head: Vec<f64>,
    time_tail: Vec<f64>,
    raw_head: Vec<f64>,
    raw_tail: Vec<f64>,
    flat_head: Vec<f64>,
    flat_tail: Vec<f64>,
    time_sum: f64,
    raw_sum: f64,
    flat_sum: f64,
    divergence_index: usize,
    divergence_time: f64,
    divergence_raw: f64,
    divergence_flat: f64,
}

fn compare_edges(actual: &[f64], head: &[f64], tail: &[f64], label: &str) {
    for (i, &expected) in head.iter().enumerate() {
        assert!(
            (actual[i] - expected).abs() < 1e-12,
            "TGLC {label} head[{i}] actual={} expected={}",
            actual[i],
            expected
        );
    }
    for (offset, &expected) in tail.iter().enumerate() {
        let actual_value = actual[actual.len() - tail.len() + offset];
        assert!(
            (actual_value - expected).abs() < 1e-12,
            "TGLC {label} tail[{offset}] actual={} expected={}",
            actual_value,
            expected
        );
    }
}

/// Mirrors ExoNitContractTests.testTGLCPreprocessGoldenParity (main.swift:1602)
/// — the regression pin for the TGLC float32-vs-float64 stitch/normalize bug
/// (TIC 10756829: pre-fix `divergence_raw` was 1.0102698802947998, a Float32
/// ULP away from the correct 1.0102699310975947). Quick-BLS derived offline
/// (see module doc comment); snr=11.58 >= 7.0, so the reference refined once.
#[test]
fn tglc_10756829_preprocess_matches_swift_golden() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/tglc_10756829_preprocess_golden.json"
    );
    let golden: TglcPreprocessGolden = serde_json::from_str(
        &std::fs::read_to_string(path).expect("read tglc_10756829_preprocess_golden.json"),
    )
    .expect("parse tglc_10756829_preprocess_golden.json");

    let curves: Vec<_> = golden
        .paths
        .iter()
        .map(|p| fits::read_light_curve(&local_path_for(p)).expect("read TGLC FITS"))
        .collect();
    assert!(
        curves.iter().all(|c| c.source == PhotometrySource::Tglc),
        "fixture sectors must be TGLC"
    );

    // Real BLS quick search (F2): verified bit-identical to the offline
    // Python-derived value on every field for this fixture (see module doc
    // comment) — asserted here directly (unlike the QLP test's looser
    // tolerances, which reflect a real, tiny cross-language drift on that
    // fixture; this one has none).
    let prepared = preprocess::prepare(&curves).expect("prepare TGLC curves");
    let quick_check = detrend_iterative::flatten(
        &prepared.time,
        &prepared.raw,
        &RealQuickBlsSearch::new(1),
        detrend_iterative::DEFAULT_SNR_THRESHOLD,
    )
    .expect("iterative flatten (real BLS)");
    let expected_quick = QuickBlsResult {
        period: 0.9964508603672763,
        power: 0.006558179026784366,
        duration: 0.14946762905509142,
        t0: 1684.395330430798,
        depth: 0.006308039065573756,
        snr: 11.577993648043542,
    };
    assert_eq!(quick_check.quick_bls.period, expected_quick.period, "real BLS quick period vs offline Python");
    assert_eq!(quick_check.quick_bls.power, expected_quick.power, "real BLS quick power vs offline Python");
    assert_eq!(
        quick_check.quick_bls.duration, expected_quick.duration,
        "real BLS quick duration vs offline Python"
    );
    assert_eq!(quick_check.quick_bls.t0, expected_quick.t0, "real BLS quick t0 vs offline Python");
    assert_eq!(quick_check.quick_bls.depth, expected_quick.depth, "real BLS quick depth vs offline Python");
    assert_eq!(quick_check.quick_bls.snr, expected_quick.snr, "real BLS quick SNR vs offline Python");

    let search = RealQuickBlsSearch::new(1);
    let result = preprocess::process(&curves, DetrendMode::Iterative(&search)).expect("process TGLC preprocess");

    assert_eq!(
        result.time.len(),
        golden.count,
        "TGLC preprocess count actual={} expected={}",
        result.time.len(),
        golden.count
    );

    compare_edges(&result.time, &golden.time_head, &golden.time_tail, "time");
    compare_edges(&result.raw_flux, &golden.raw_head, &golden.raw_tail, "raw");
    compare_edges(&result.flat_flux, &golden.flat_head, &golden.flat_tail, "flat");

    // Tolerances mirror main.swift:1628-1630.
    let time_sum: f64 = result.time.iter().sum();
    let raw_sum: f64 = result.raw_flux.iter().sum();
    let flat_sum: f64 = result.flat_flux.iter().sum();
    assert!(
        (time_sum - golden.time_sum).abs() < 1e-6,
        "TGLC time sum actual={time_sum} expected={}",
        golden.time_sum
    );
    assert!(
        (raw_sum - golden.raw_sum).abs() < 1e-9,
        "TGLC raw sum actual={raw_sum} expected={}",
        golden.raw_sum
    );
    assert!(
        (flat_sum - golden.flat_sum).abs() < 1e-8,
        "TGLC flat sum actual={flat_sum} expected={}",
        golden.flat_sum
    );

    // Tolerances mirror main.swift:1632-1641 — the exact regression pin.
    assert!(
        (result.time[golden.divergence_index] - golden.divergence_time).abs() < 1e-12,
        "TGLC divergence row time"
    );
    assert!(
        (result.raw_flux[golden.divergence_index] - golden.divergence_raw).abs() < 1e-12,
        "TGLC divergence row raw actual={} expected={} \
         (pre-fix Float32 value was 1.0102698802947998 — a regression here means \
         the TGLC dtype fix broke)",
        result.raw_flux[golden.divergence_index],
        golden.divergence_raw
    );
    assert!(
        (result.flat_flux[golden.divergence_index] - golden.divergence_flat).abs() < 1e-9,
        "TGLC divergence row flat actual={} expected={}",
        result.flat_flux[golden.divergence_index],
        golden.divergence_flat
    );
}
