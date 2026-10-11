//! Transliterated from `Sources/ExoNitCore/TransitPhysics.swift`.

/// Mirrors `TransitPhysics.durationDays(periodDays:stellarRadiusSolar:stellarMassSolar:impactParameter:planetRadiusEarth:)`.
/// Rust has no default parameter values, so all Swift defaults
/// (`stellarRadiusSolar: 1, impactParameter: 0, planetRadiusEarth: 1`) must
/// be supplied explicitly by callers.
pub fn duration_days(
    period_days: f64,
    stellar_radius_solar: f64,
    stellar_mass_solar: Option<f64>,
    impact_parameter: f64,
    planet_radius_earth: f64,
) -> f64 {
    if !(period_days > 0.0) || !(stellar_radius_solar > 0.0) {
        return 0.0;
    }
    let mass = stellar_mass_solar.unwrap_or_else(|| stellar_radius_solar.powf(1.25));
    if !(mass > 0.0) {
        return 0.0;
    }
    let semimajor_au = (mass * (period_days / 365.25).powi(2)).powf(1.0 / 3.0);
    let semimajor_solar = semimajor_au * 214.94;
    let radius_ratio = planet_radius_earth * 0.00916 / stellar_radius_solar;
    let argument_squared = (1.0 + radius_ratio).powi(2) - impact_parameter * impact_parameter;
    if !(argument_squared > 0.0) {
        return 0.0;
    }
    let sine_argument = stellar_radius_solar / semimajor_solar * argument_squared.sqrt();
    if sine_argument >= 1.0 {
        return period_days / 2.0;
    }
    period_days / std::f64::consts::PI * sine_argument.asin()
}

/// Mirrors `TransitPhysics.physicalT14Maximum(periodDays:stellarRadiusSolar:stellarMassSolar:)`
/// (`stellarRadiusSolar ?? 1`, `impactParameter: 0`, `planetRadiusEarth: 11.2`).
pub fn physical_t14_maximum(period_days: f64, stellar_radius_solar: Option<f64>, stellar_mass_solar: Option<f64>) -> f64 {
    duration_days(period_days, stellar_radius_solar.unwrap_or(1.0), stellar_mass_solar, 0.0, 11.2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_positive_period_returns_zero() {
        assert_eq!(duration_days(0.0, 1.0, None, 0.0, 1.0), 0.0);
        assert_eq!(duration_days(-1.0, 1.0, None, 0.0, 1.0), 0.0);
    }

    #[test]
    fn non_positive_stellar_radius_returns_zero() {
        assert_eq!(duration_days(10.0, 0.0, None, 0.0, 1.0), 0.0);
    }

    #[test]
    fn earth_sun_like_system_is_physically_plausible() {
        // Rough sanity check: an Earth-like period (365.25d) around a
        // Sun-like star should produce an O(0.5 day) transit duration.
        let value = physical_t14_maximum(365.25, Some(1.0), Some(1.0));
        assert!(value > 0.0 && value < 2.0, "unexpected T14 max: {value}");
    }
}
