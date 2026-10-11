//! Small deterministic statistics helpers shared by the numerical pipelines.

/// Median with the ordering and empty-input behaviour used by the original
/// per-module implementations. Empty input intentionally panics: callers
/// which have a defined fallback use [`median_or`] instead.
pub(crate) fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

/// Median with an explicit fallback for an empty input.
pub(crate) fn median_or(values: &[f64], fallback: f64) -> f64 {
    if values.is_empty() {
        fallback
    } else {
        median(values)
    }
}

/// Median of finite values, preserving the preprocessor's NaN fallback.
pub(crate) fn nan_median(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|value| value.is_finite()).collect();
    median_or(&finite, f64::NAN)
}
