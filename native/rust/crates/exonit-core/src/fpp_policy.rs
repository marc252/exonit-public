//! Transliterated from `Sources/ExoNitCore/FPPPolicy.swift`.
//!
//! Only the policy/thresholds (whether to run FPP, and how to turn a
//! `(fpp, nfpp, high_crowding)` triple into a verdict) — the TRICERATOPS
//! execution itself is a sidecar process (`Sidecars/` in the Swift repo,
//! `reference-python/triceratops_fpp.py`), out of scope here per the task
//! brief; nothing in this module runs TRICERATOPS or talks to the sidecar.

/// Mirrors `FPPVerdict` (`String`-backed enum in Swift; raw values kept
/// identical for round-tripping through the same JSON the worker/server
/// already exchange, e.g. `CandidateResult.triceratops_verdict`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FppVerdict {
    Validated,
    Candidate,
    LikelyFalsePositive,
}

impl FppVerdict {
    /// Mirrors the Swift `String` raw value (`"validated"`, `"candidate"`,
    /// `"likely_fp"`).
    pub fn as_str(self) -> &'static str {
        match self {
            FppVerdict::Validated => "validated",
            FppVerdict::Candidate => "candidate",
            FppVerdict::LikelyFalsePositive => "likely_fp",
        }
    }
}

impl std::fmt::Display for FppVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

pub const MINIMUM_SNR: f64 = 10.0;
pub const MINIMUM_CNN_SCORE: f64 = 0.7;
pub const VALIDATED_FPP: f64 = 0.015;
pub const CANDIDATE_FPP: f64 = 0.5;
pub const VALIDATED_NFPP: f64 = 0.001;

/// Mirrors `FPPPolicy.shouldRun(vettingPassed:snr:cnnScore:)`.
pub fn should_run(vetting_passed: bool, snr: f64, cnn_score: Option<f64>) -> bool {
    if !vetting_passed || snr < MINIMUM_SNR {
        return false;
    }
    if let Some(cnn_score) = cnn_score {
        if cnn_score < MINIMUM_CNN_SCORE {
            return false;
        }
    }
    true
}

/// Mirrors `FPPPolicy.verdict(fpp:nfpp:highCrowding:)`.
pub fn verdict(fpp: f64, nfpp: Option<f64>, high_crowding: bool) -> FppVerdict {
    if fpp >= CANDIDATE_FPP {
        return FppVerdict::LikelyFalsePositive;
    }
    if fpp < VALIDATED_FPP {
        if let Some(nfpp) = nfpp {
            if nfpp < VALIDATED_NFPP {
                return if high_crowding { FppVerdict::Candidate } else { FppVerdict::Validated };
            }
        }
    }
    FppVerdict::Candidate
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors `ExoNitContractTests.testFPPPolicyParity` (`main.swift:823`),
    /// same cases and expected outcomes.
    #[test]
    fn fpp_policy_matches_swift_contract_test() {
        assert!(should_run(true, 10.0, None), "FPP nil CNN");
        assert!(should_run(true, 10.0, Some(0.7)), "FPP inclusive gates");
        assert!(!should_run(false, 20.0, Some(0.9)), "FPP vetting gate");
        assert!(!should_run(true, 9.999, Some(0.9)), "FPP SNR gate");
        assert!(!should_run(true, 20.0, Some(0.699)), "FPP CNN gate");

        assert_eq!(verdict(0.5, Some(0.0), false), FppVerdict::LikelyFalsePositive, "FPP likely FP boundary");
        assert_eq!(verdict(0.014, Some(0.0009), false), FppVerdict::Validated, "FPP validated");
        assert_eq!(verdict(0.014, Some(0.0009), true), FppVerdict::Candidate, "FPP crowding downgrade");
        assert_eq!(verdict(0.014, None, false), FppVerdict::Candidate, "FPP needs NFPP");
    }

    #[test]
    fn verdict_raw_values_match_swift_string_backing() {
        assert_eq!(FppVerdict::Validated.as_str(), "validated");
        assert_eq!(FppVerdict::Candidate.as_str(), "candidate");
        assert_eq!(FppVerdict::LikelyFalsePositive.as_str(), "likely_fp");
    }
}
