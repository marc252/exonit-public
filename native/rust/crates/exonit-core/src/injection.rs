//! Transliterated from `Sources/ExoNitCore/InjectionPrimitives.swift`.
//!
//! `NumpyMt19937` reproduces NumPy's legacy `RandomState(seed)` generator
//! (MT19937) bit-for-bit, including its `random_sample` (`uniform01`) path
//! (two 32-bit draws combined as `(a*2^26 + b) / 2^53`, the standard
//! `genrand_res53` construction). `injection_recovery.rs` seeds one instance
//! per (radius, period, injection) cell with a deterministic integer seed,
//! and `Fixtures/injection_recovery_golden.json` pins the exact `t0` draws
//! this produces — so this generator must NOT be swapped for a different
//! PRNG (this is the one case in the crate where draw-for-draw reproduction,
//! not just statistical equivalence, is the contract). No `rand` crate: this
//! is a small self-contained MT19937, matching the project's F1 "no new
//! dependencies" policy (see `gaps.rs`'s `SplitMix64` for the analogous,
//! non-seeded-parity case).
//!
//! `depth`/`inject` mirror the Swift `TransitInjector` enum's static
//! methods; ported here as free functions (matching the module-as-namespace
//! convention already used by `transit_physics.rs` for the Swift
//! `TransitPhysics` enum).

/// Mirrors `NumPyMT19937` (`Sources/ExoNitCore/InjectionPrimitives.swift`).
#[derive(Debug, Clone)]
pub struct NumpyMt19937 {
    state: [u32; 624],
    index: usize,
}

impl NumpyMt19937 {
    /// Mirrors `NumPyMT19937.init(seed:)`.
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1_812_433_253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        NumpyMt19937 { state, index: 624 }
    }

    /// Mirrors `NumPyMT19937.uniform01()` (NumPy's `random_sample`/
    /// `genrand_res53`: two 32-bit draws combined into a 53-bit mantissa).
    pub fn uniform01(&mut self) -> f64 {
        let a = self.next_u32() >> 5;
        let b = self.next_u32() >> 6;
        (f64::from(a) * 67_108_864.0 + f64::from(b)) / 9_007_199_254_740_992.0
    }

    /// Mirrors `NumPyMT19937.nextUInt32()` (the MT19937 tempering step).
    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9D2C_5680;
        value ^= (value << 15) & 0xEFC6_0000;
        value ^= value >> 18;
        value
    }

    /// Mirrors `NumPyMT19937.twist()`. Sequential in-place mutation of
    /// `state` is load-bearing: once `i + 397` wraps past 624 (for
    /// `i >= 227`), it reads state slots already overwritten earlier in
    /// *this same* pass — that is the standard MT19937 twist recurrence,
    /// not a bug, and this loop must stay a plain `0..624` walk mutating
    /// `state` directly (no snapshot/copy) to match it.
    fn twist(&mut self) {
        for i in 0..624 {
            let combined = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7FFF_FFFF);
            let mut value = self.state[(i + 397) % 624] ^ (combined >> 1);
            if combined & 1 != 0 {
                value ^= 0x9908_B0DF;
            }
            self.state[i] = value;
        }
        self.index = 0;
    }
}

/// Mirrors `TransitInjector.depth(radiusEarth:stellarRadiusSolar:)`.
pub fn depth(radius_earth: f64, stellar_radius_solar: f64) -> f64 {
    let ratio = radius_earth * 0.00916 / stellar_radius_solar;
    ratio * ratio
}

/// Mirrors `TransitInjector.inject(time:flux:period:t0:depth:duration:)`.
/// Box-shaped (top-hat) transit mask: subtracts `depth` from every sample
/// whose phase falls within `duration` centered on `t0`.
pub fn inject(time: &[f64], flux: &[f64], period: f64, t0: f64, depth: f64, duration: f64) -> Vec<f64> {
    let mut result = flux.to_vec();
    let half = duration / period / 2.0;
    for index in 0..time.len() {
        let mut phase = (time[index] - t0) % period / period;
        if phase < 0.0 {
            phase += 1.0;
        }
        if phase < half || phase > 1.0 - half {
            result[index] -= depth;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transit_physics;

    /// Mirrors `ExoNitContractTests.testInjectionPrimitivesParity`
    /// (`main.swift:835`), same values and same tolerances (`1e-16` RNG,
    /// `1e-15` duration, `1e-18` depth, exact `==` on the injected mask).
    #[test]
    fn injection_primitives_matches_swift_contract_test() {
        let mut random = NumpyMt19937::new(0);
        let expected = [
            0.5488135039273248,
            0.7151893663724195,
            0.6027633760716439,
            0.5448831829968969,
            0.4236547993389047,
        ];
        for value in expected {
            assert!((random.uniform01() - value).abs() < 1e-16, "NumPy RandomState parity");
        }

        let duration = transit_physics::duration_days(10.0, 1.0, None, 0.0, 11.2);
        assert!((duration - 0.1798480076597091).abs() < 1e-15, "transit duration parity: {duration}");

        let injection_depth = depth(2.0, 1.0);
        assert!((injection_depth - 0.00033562239999999997).abs() < 1e-18, "injection depth parity: {injection_depth}");

        let time = [-0.1, 0.0, 0.05, 0.5, 0.95, 1.0];
        let flux = [1.0; 6];
        let injected = inject(&time, &flux, 1.0, 0.0, 0.01, 0.2);
        assert_eq!(injected, [1.0, 0.99, 0.99, 1.0, 0.99, 0.99], "injection mask semantics");
    }
}
