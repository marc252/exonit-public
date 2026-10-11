import Foundation

/// Monte-Carlo prior samplers translated from `fpp_native.py:545-689`.
/// Each sampler consumes its complete uniform vector before transforming it,
/// matching NumPy's legacy RandomState draw order.
public enum FPPPriors {
    private static let pi = Double.pi

    private static func uniformVector(_ rng: inout FPPRNG, count: Int) -> [Double] {
        (0..<count).map { _ in rng.rand() }
    }

    public static func planetRadius(_ rng: inout FPPRNG, count: Int, stellarMass: Double) -> [Double] {
        var x = uniformVector(&rng, count: count)
        let break1 = 3.0, break2 = 6.0, minRadius = 0.5, maxRadius = 20.0
        let (p1, p2, p3) = stellarMass > 0.45 ? (0.0, -4.0, -0.5) : (0.0, -7.0, -0.5)
        let a1 = pow(break1, p1) / pow(break1, p2)
        let a2 = pow(break2, p2) / pow(break2, p3)
        let i1 = (pow(break1, p1 + 1) - pow(minRadius, p1 + 1)) / (p1 + 1)
        let i2 = a1 * (pow(break2, p2 + 1) - pow(break1, p2 + 1)) / (p2 + 1)
        let i3 = a2 * a1 * (pow(maxRadius, p3 + 1) - pow(break2, p3 + 1)) / (p3 + 1)
        let norm = 1.0 / (i1 + i2 + i3)
        for i in x.indices {
            let u = x[i]
            if u <= norm * i1 {
                x[i] = pow(u / norm * (p1 + 1) + pow(minRadius, p1 + 1), 1 / (p1 + 1))
            } else if u <= norm * (i1 + i2) {
                x[i] = pow((u / norm - i1) * (p2 + 1) / a1 + pow(break1, p2 + 1), 1 / (p2 + 1))
            } else {
                x[i] = pow((u / norm - i1 - i2) * (p3 + 1) / (a1 * a2) + pow(break2, p3 + 1), 1 / (p3 + 1))
            }
        }
        return x
    }

    public static func inclinationDegrees(_ rng: inout FPPRNG, count: Int) -> [Double] {
        uniformVector(&rng, count: count).map { acos(1.0 - $0) * 180.0 / pi }
    }

    public static func eccentricity(_ rng: inout FPPRNG, count: Int, planet: Bool, period: Double) -> [Double] {
        if planet { return (0..<count).map { _ in rng.beta(0.867, 3.030) } }
        let shape = period <= 10.0 ? 0.2 : 0.6
        return (0..<count).map { _ in rng.power(shape) }
    }

    public static func argumentOfPeriastronDegrees(_ rng: inout FPPRNG, count: Int) -> [Double] {
        uniformVector(&rng, count: count).map { $0 * 360.0 }
    }

    private static func transformMassRatios(
        _ x: inout [Double], minimum: Double, p1: Double, p2: Double, twinFraction: Double
    ) {
        if minimum < 0.3 {
            let a1 = pow(0.3, p1) / pow(0.3, p2)
            let a2 = 1.0 + (twinFraction / (1.0 - twinFraction))
                * ((pow(1.0, p2 + 1) - pow(0.3, p2 + 1)) / (p2 + 1))
                / ((pow(1.0, p2 + 1) - pow(0.95, p2 + 1)) / (p2 + 1))
            let i1 = (pow(0.3, p1 + 1) - pow(minimum, p1 + 1)) / (p1 + 1)
            let i2 = a1 * (pow(0.95, p2 + 1) - pow(0.3, p2 + 1)) / (p2 + 1)
            let i3 = a2 * a1 * (pow(1.0, p2 + 1) - pow(0.95, p2 + 1)) / (p2 + 1)
            let norm = 1.0 / (i1 + i2 + i3)
            for i in x.indices {
                let u = x[i]
                if u <= norm * i1 {
                    x[i] = pow(u / norm * (p1 + 1) + pow(minimum, p1 + 1), 1 / (p1 + 1))
                } else if u <= norm * (i1 + i2) {
                    x[i] = pow((u / norm - i1) * (p2 + 1) / a1 + pow(0.3, p2 + 1), 1 / (p2 + 1))
                } else {
                    x[i] = pow((u / norm - i1 - i2) * (p2 + 1) / (a1 * a2) + pow(0.95, p2 + 1), 1 / (p2 + 1))
                }
            }
        } else {
            let a2 = 1.0 + (twinFraction / (1.0 - twinFraction))
                * ((pow(1.0, p2 + 1) - pow(minimum, p2 + 1)) / (p2 + 1))
                / ((pow(1.0, p2 + 1) - pow(0.95, p2 + 1)) / (p2 + 1))
            let i2 = (pow(0.95, p2 + 1) - pow(minimum, p2 + 1)) / (p2 + 1)
            let i3 = a2 * (pow(1.0, p2 + 1) - pow(0.95, p2 + 1)) / (p2 + 1)
            let norm = 1.0 / (i2 + i3)
            for i in x.indices {
                let u = x[i]
                if u <= norm * i2 {
                    x[i] = pow(u / norm * (p2 + 1) + pow(minimum, p2 + 1), 1 / (p2 + 1))
                } else {
                    x[i] = pow((u / norm - i2) * (p2 + 1) / a2 + pow(0.95, p2 + 1), 1 / (p2 + 1))
                }
            }
        }
        for i in x.indices { x[i] = min(1.0, max(minimum, x[i])) }
    }

    public static func massRatio(_ rng: inout FPPRNG, count: Int, stellarMass: Double) -> [Double] {
        var x = uniformVector(&rng, count: count)
        let minimum: Double
        if stellarMass >= 1.0 { minimum = 0.1 }
        else if stellarMass > 0.1 { minimum = 0.1 / stellarMass }
        else { return Array(repeating: 1.0, count: count) }
        transformMassRatios(&x, minimum: minimum, p1: 0.3, p2: -0.5, twinFraction: 0.30)
        return x
    }

    public static func companionMassRatio(_ rng: inout FPPRNG, count: Int, stellarMass: Double) -> [Double] {
        var x = uniformVector(&rng, count: count)
        if stellarMass <= 0.1 { return Array(repeating: 1.0, count: count) }
        let minimum = max(0.1 / stellarMass, 0.1)
        transformMassRatios(&x, minimum: minimum, p1: 0.3, p2: -0.95, twinFraction: 0.05)
        return x
    }

    // ============================================================
    // Bound-companion prior (PTP/PEB only)
    // ============================================================
    //
    // Ported from `triceratops/priors.py:lnprior_bound_TP`/`lnprior_bound_EB`
    // and `triceratops/funcs.py:separation_at_contrast`, via
    // `fpp_native.py`'s `_lnprior_bound_TP`/`_lnprior_bound_EB`/
    // `_lnprior_companion` (exonit a345343). This is the per-draw log
    // probability, from the Moe & Di Stefano (2017) bound-companion rate,
    // that a companion drawn with the given mass ratio/parallax/contrast is
    // actually bound to the target — folded into PTP/PEB's Monte-Carlo
    // average by `FPPScenarios.lnzPTP`/`lnzPEB`, NOT a separate additive
    // term on lnZ. Never applies to TTP/TEB (no such term upstream) nor to
    // this port's NTP/NEB (nearby-star scenarios reuse `lnzTTP`/`lnzTEB`
    // with the default no-op zero prior) — upstream NTP/NEB use a
    // different, unimplemented prior (`lnprior_background`); documented
    // residual, out of scope here.
    //
    // This port never supplies a contrast-curve file (matches upstream's
    // own `contrast_curve_file=None`/`molusc_file=None` default path, the
    // only one `calc_fpp_native` ever exercises), so separations/contrasts
    // are hardcoded to upstream's own no-contrast-curve stand-in.

    /// Scenario convention for the bound-companion period floor:
    /// `.transiting` (PTP, `Porb_comp > 2500 d`) or `.eclipsing` (PEB,
    /// `Porb_comp > 10 d`).
    public enum BoundCompanionScenario { case transiting, eclipsing }

    /// Limiting separation (arcsec) at `deltaMags`, from a contrast curve.
    /// Mirrors `triceratops/funcs.py:separation_at_contrast(delta_mags,
    /// separations, contrasts) = np.interp(delta_mags, contrasts,
    /// separations)`, with `separations`/`contrasts` fixed at upstream's
    /// no-contrast-curve default (`[2.2]` / `[1.0]`). With a single-point
    /// `xp` (`contrasts`), `np.interp` returns that one `fp` value (the
    /// single `separations` entry, 2.2 arcsec) for every `deltaMag`,
    /// regardless of its value — verified against upstream's actual
    /// behaviour, including NaN inputs.
    static func separationAtContrast(_ deltaMags: [Double]) -> [Double] {
        Array(repeating: 2.2, count: deltaMags.count)
    }

    /// `f1`/`f2`/`f3` coefficients shared by `lnprior_bound_TP`/`_EB`,
    /// evaluated at a given primary `mass` (Msun). `M_s` here is always a
    /// Python scalar upstream, so every `**` below is scalar power (libm
    /// `pow`, matching `FPPNumpy.libmPow` — see the project's scalar-vs-
    /// array `**` seam).
    private static func boundCompanionCoefficients(mass: Double) -> (f1: Double, f2: Double, f3: Double, alpha: Double, dlogP: Double) {
        let logM = log10(mass)
        let logM2 = FPPNumpy.libmPow(logM, 2)
        let f1 = 0.020 + 0.04 * logM + 0.07 * logM2
        let f2 = 0.039 + 0.07 * logM + 0.01 * logM2
        let f3 = 0.078 - 0.05 * logM + 0.04 * logM2
        return (f1, f2, f3, 0.018, 0.7)
    }

    /// log10 of the maximum companion orbital period (days) allowed at each
    /// drawn projected separation `seps` (AU), for a primary of `mass`
    /// (Msun). Mirrors the shared `max_Porbs`/`log_mP` preamble of
    /// `lnprior_bound_TP`/`_EB`. `seps` is a NumPy array upstream: `(seps *
    /// au)**3` has NO numpy integer-power-3 fast path (falls through to
    /// libm `pow`), while the outer `**(1 / 2)` DOES hit numpy's `0.5`
    /// fast path (`sqrt`) — both confirmed empirically against installed
    /// NumPy, not assumed from the `**2` convention alone.
    private static func log10MaxCompanionPeriodDays(seps: [Double], mass: Double) -> [Double] {
        let numerator = 4 * FPPNumpy.libmPow(pi, 2)
        let denominator = FPPScenarioConstants.gCGS * mass * FPPScenarioConstants.msun
        let coeff = numerator / denominator
        return seps.map { sep in
            let sepsAu = sep * FPPScenarioConstants.au
            let cubed = FPPNumpy.libmPow(sepsAu, 3)
            let maxPorb = sqrt(coeff * cubed) / 86400.0
            return log10(maxPorb)
        }
    }

    /// `f_comp` (pre-`log`) for the PTP convention (`Porb_comp > 2500 d`),
    /// mirroring `lnprior_bound_TP`'s piecewise Moe & Di Stefano formula
    /// (the `t4`/`t5` breakpoints only — TP's period floor skips the
    /// `t2`/`t3` segments EB has). `log_mP` (`logMP` below) is a NumPy
    /// array upstream, so `log_mP**2` hits numpy's integer-2 fast path
    /// (direct multiply), NOT libm `pow`.
    private static func lnpriorBoundTPFcomp(mass: Double, seps: [Double]) -> [Double] {
        let (_, f2, f3, alpha, dlogP) = boundCompanionCoefficients(mass: mass)
        let logMP = log10MaxCompanionPeriodDays(seps: seps, mass: mass)
        let t4 = alpha * dlogP * (5.5 - 3.4) + f2 * (5.5 - 3.4)
            + (f3 - f2 - alpha * dlogP)
            * (0.238095 * FPPNumpy.libmPow(5.5, 2) - 0.952381 * 5.5 + 0.485714)
        let t5 = f3 * (3.33333 - 17.3566 * exp(-0.3 * 8.0))
        return logMP.map { lm in
            if lm >= 3.4 && lm < 5.5 {
                return alpha * dlogP * (lm - 3.4) + f2 * (lm - 3.4)
                    + (f3 - f2 - alpha * dlogP)
                    * (0.238095 * (lm * lm) - 0.952381 * lm + 0.485714)
            } else if lm >= 5.5 && lm < 8.0 {
                return t4 + f3 * (3.33333 - 17.3566 * exp(-0.3 * lm))
            } else if lm >= 8.0 {
                return t4 + t5
            } else {
                return 0.0
            }
        }
    }

    /// `f_comp` (pre-`log`) for the PEB convention (`Porb_comp > 10 d`) —
    /// same Moe & Di Stefano (2017) rate as `lnpriorBoundTPFcomp`, but with
    /// the extra `t2`/`t3` segments (period floor two breakpoints earlier).
    private static func lnpriorBoundEBFcomp(mass: Double, seps: [Double]) -> [Double] {
        let (f1, f2, f3, alpha, dlogP) = boundCompanionCoefficients(mass: mass)
        let logMP = log10MaxCompanionPeriodDays(seps: seps, mass: mass)
        let t2 = 0.5 * (2.0 - 1.0) * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (2.0 - 1.0))
        let t3 = 0.5 * alpha * (FPPNumpy.libmPow(3.4, 2) - 5.4 * 3.4 + 6.8) + f2 * (3.4 - 2.0)
        let t4 = alpha * dlogP * (5.5 - 3.4) + f2 * (5.5 - 3.4)
            + (f3 - f2 - alpha * dlogP)
            * (0.238095 * FPPNumpy.libmPow(5.5, 2) - 0.952381 * 5.5 + 0.485714)
        let t5 = f3 * (3.33333 - 17.3566 * exp(-0.3 * 8.0))
        return logMP.map { lm in
            if lm >= 1.0 && lm < 2.0 {
                return 0.5 * (lm - 1.0) * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (lm - 1.0))
            } else if lm >= 2.0 && lm < 3.4 {
                return t2 + 0.5 * alpha * (lm * lm - 5.4 * lm + 6.8) + f2 * (lm - 2.0)
            } else if lm >= 3.4 && lm < 5.5 {
                return t2 + t3 + alpha * dlogP * (lm - 3.4) + f2 * (lm - 3.4)
                    + (f3 - f2 - alpha * dlogP)
                    * (0.238095 * (lm * lm) - 0.952381 * lm + 0.485714)
            } else if lm >= 5.5 && lm < 8.0 {
                return t2 + t3 + t4 + f3 * (3.33333 - 17.3566 * exp(-0.3 * lm))
            } else if lm >= 8.0 {
                return t2 + t3 + t4 + t5
            } else {
                return 0.0
            }
        }
    }

    /// Log prior of a bound companion, PTP convention (`Porb_comp > 2500
    /// d`). Mirrors `triceratops/priors.py:lnprior_bound_TP` exactly.
    /// `deltaMagsAbs` must already be `abs()`-ed by the caller (matches
    /// upstream's call site). The `plx` NaN fallback is kept for fidelity
    /// even though `FPPCalculation`'s `physicalCompanion` gate (mirroring
    /// `fpp_native.py`'s `if not np.isnan(plx):` guard) never reaches this
    /// function with a NaN parallax.
    static func lnpriorBoundTP(massSolar M_s: Double, parallax plx: Double, deltaMagsAbs: [Double]) -> [Double] {
        let effectivePlx = plx.isNaN ? 0.1 : plx
        let d = 1000.0 / effectivePlx
        let seps = separationAtContrast(deltaMagsAbs).map { d * $0 }
        if M_s >= 1.0 {
            return lnpriorBoundTPFcomp(mass: M_s, seps: seps).map { log($0) }
        }
        let mAct = M_s
        return lnpriorBoundTPFcomp(mass: 1.0, seps: seps).map { fComp in
            log(max(0.0, 0.65 * fComp + 0.35 * fComp * mAct))
        }
    }

    /// Log prior of a bound companion, PEB convention (`Porb_comp > 10
    /// d`). Mirrors `triceratops/priors.py:lnprior_bound_EB` exactly — see
    /// `lnpriorBoundTP`'s doc comment for the shared `plx`/`deltaMagsAbs`
    /// contract.
    static func lnpriorBoundEB(massSolar M_s: Double, parallax plx: Double, deltaMagsAbs: [Double]) -> [Double] {
        let effectivePlx = plx.isNaN ? 0.1 : plx
        let d = 1000.0 / effectivePlx
        let seps = separationAtContrast(deltaMagsAbs).map { d * $0 }
        if M_s >= 1.0 {
            return lnpriorBoundEBFcomp(mass: M_s, seps: seps).map { log($0) }
        }
        let mAct = M_s
        return lnpriorBoundEBFcomp(mass: 1.0, seps: seps).map { fComp in
            log(max(0.0, 0.65 * fComp + 0.35 * fComp * mAct))
        }
    }

    /// Per-draw `lnprior_companion` array fed into PTP/PEB's Monte-Carlo
    /// `Z` average. Mirrors the "calculate priors for companions" block
    /// shared by upstream `lnZ_PTP`/`lnZ_PEB` (`contrast_curve_file=None`
    /// branch): computes `deltaMags` from the already-drawn
    /// `fluxRatiosCompanion`, looks up the bound rate (`scenario` picks TP
    /// vs EB convention), then clips it to a proper log probability (`<=
    /// 0`) and vetoes companions brighter than the target (`deltaMags > 0`
    /// -> `-inf`, using the SIGNED `deltaMags` even though the prior
    /// itself was computed from `abs(deltaMags)`).
    public static func lnpriorCompanion(massSolar M_s: Double, parallax plx: Double, fluxRatiosCompanion: [Double], scenario: BoundCompanionScenario) -> [Double] {
        let deltaMags = fluxRatiosCompanion.map { 2.5 * log10($0 / (1 - $0)) }
        let raw = scenario == .transiting
            ? lnpriorBoundTP(massSolar: M_s, parallax: plx, deltaMagsAbs: deltaMags.map { abs($0) })
            : lnpriorBoundEB(massSolar: M_s, parallax: plx, deltaMagsAbs: deltaMags.map { abs($0) })
        return zip(raw, deltaMags).map { value, deltaMag in
            if deltaMag > 0.0 { return -Double.infinity }
            return value > 0.0 ? 0.0 : value
        }
    }
}
