//! Frozen-input structures for the native FPP calculation.
//!
//! `NearbyResponse` mirrors the `/api/nearby/{tic}` payload; parsing
//! reproduces `_query_tic_stars` post-processing (`fpp_native.py:977-1033`):
//! target promotion to `stars[0]` and `null → NaN` for the numeric fields.
//! `TrilegalStar` mirrors `_load_trilegal_stars` (`:1036-1081`).

use serde_json::Value;

use super::npsum::libm_pow;

#[derive(Debug, Clone)]
pub struct StarRec {
    pub id: String,
    pub ra: f64,
    pub dec: f64,
    pub mass: f64,
    pub rad: f64,
    pub teff: f64,
    pub tmag: f64,
    pub plx: f64,
    pub is_trilegal: bool,
    // filled by compute_flux_ratios
    pub weighted_flux: f64,
    pub fluxratio: f64,
    pub tdepth: f64,
}

#[derive(Debug, Clone)]
pub struct NearbyResponse {
    pub stars: Vec<StarRec>,
    pub gal_b: f64,
    pub target_promoted: bool,
}

fn num_or_nan(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
        _ => f64::NAN,
    }
}

fn id_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

/// Parse the raw `/api/nearby` JSON and apply `_query_tic_stars`
/// post-processing for `tic_id`.
pub fn parse_nearby(doc: &Value, tic_id: &str) -> Option<NearbyResponse> {
    let stars_json = doc.get("stars")?.as_array()?;
    let gal_b = doc.get("gal_b").map(num_or_nan).unwrap_or(0.0);
    let mut stars: Vec<StarRec> = stars_json
        .iter()
        .map(|s| StarRec {
            id: s.get("ID").map(id_string).unwrap_or_default(),
            ra: s.get("ra").map(num_or_nan).unwrap_or(f64::NAN),
            dec: s.get("dec").map(num_or_nan).unwrap_or(f64::NAN),
            mass: s.get("mass").map(num_or_nan).unwrap_or(f64::NAN),
            rad: s.get("rad").map(num_or_nan).unwrap_or(f64::NAN),
            teff: s.get("Teff").map(num_or_nan).unwrap_or(f64::NAN),
            tmag: s.get("Tmag").map(num_or_nan).unwrap_or(f64::NAN),
            plx: s.get("plx").map(num_or_nan).unwrap_or(f64::NAN),
            is_trilegal: false,
            weighted_flux: f64::NAN,
            fluxratio: f64::NAN,
            tdepth: f64::NAN,
        })
        .collect();
    let target_idx = stars.iter().position(|s| s.id == tic_id);
    let mut promoted = false;
    if let Some(idx) = target_idx {
        if idx != 0 {
            let t = stars.remove(idx);
            stars.insert(0, t);
            promoted = true;
        }
    }
    Some(NearbyResponse {
        stars,
        gal_b,
        target_promoted: promoted,
    })
}

/// `_load_trilegal_stars` from CSV text (whitespace-separated, header row).
/// Returns star records (positions = target's, plx NaN).
pub fn load_trilegal(csv: &str, target_ra: f64, target_dec: f64, target_tmag: f64) -> Vec<StarRec> {
    let mut lines = csv.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = match lines.next() {
        Some(h) => h.split_whitespace().collect(),
        None => return Vec::new(),
    };
    let col = |name: &str| header.iter().position(|h| *h == name);
    let (Some(i_tess),) = (col("TESS"),) else {
        return Vec::new();
    };
    let i_logte = col("logTe");
    let i_mact = col("Mact");
    let i_logg = col("logg");
    let mut out = Vec::new();
    for line in lines {
        let f: Vec<&str> = line.split_whitespace().collect();
        let get = |i: Option<usize>| -> Option<f64> { i.and_then(|i| f.get(i)?.parse().ok()) };
        let tmag = match get(Some(i_tess)) {
            Some(v) => v,
            None => continue,
        };
        if tmag > target_tmag + 8.0 {
            continue;
        }
        let teff = match i_logte {
            Some(_) => 10.0f64.powf(get(i_logte).unwrap_or(f64::NAN)),
            None => 5777.0,
        };
        let mass = if i_mact.is_some() {
            get(i_mact).unwrap_or(f64::NAN)
        } else {
            1.0
        };
        let logg_val = if i_logg.is_some() {
            get(i_logg).unwrap_or(f64::NAN)
        } else {
            4.4
        };
        let rad = (super::scenarios::G_CGS * mass * super::scenarios::MSUN
            / 10.0f64.powf(logg_val))
        .sqrt()
            / super::scenarios::RSUN;
        out.push(StarRec {
            id: format!("TRILEGAL_{}", out.len()),
            ra: target_ra,
            dec: target_dec,
            mass,
            rad,
            teff,
            tmag,
            plx: f64::NAN,
            is_trilegal: true,
            weighted_flux: f64::NAN,
            fluxratio: f64::NAN,
            tdepth: f64::NAN,
        });
    }
    out
}

/// `_compute_flux_ratios` (`fpp_native.py:1084-1112`) — PSF-weighted flux
/// ratios and required transit depths, in place. `stars[0]` is the target.
pub fn compute_flux_ratios(stars: &mut [StarRec], depth_ppm: f64) {
    if stars.is_empty() {
        return;
    }
    let (t_ra, t_dec, t_tmag) = (stars[0].ra, stars[0].dec, stars[0].tmag);
    let sigma_psf = 21.0f64;
    for s in stars.iter_mut() {
        // Python-scalar `** 2` here (dict floats) — libm pow, NOT x*x.
        let sep = (libm_pow(s.ra - t_ra, 2.0) + libm_pow(s.dec - t_dec, 2.0)).sqrt() * 3600.0;
        let raw_flux = 10.0f64.powf((t_tmag - s.tmag) / 2.5);
        // Python: np.exp(-0.5 * (sep / sigma_psf)**2) — scalar square via
        // libm pow, computed first, then scaled.
        let psf_weight = (-0.5 * libm_pow(sep / sigma_psf, 2.0)).exp();
        s.weighted_flux = raw_flux * psf_weight;
    }
    // Python `sum(...)`: plain sequential sum
    let mut total_flux = 0.0;
    for s in stars.iter() {
        total_flux += s.weighted_flux;
    }
    for s in stars.iter_mut() {
        s.fluxratio = s.weighted_flux / total_flux;
        s.tdepth = if s.fluxratio > 0.0 {
            depth_ppm / s.fluxratio
        } else {
            0.0
        };
    }
}
