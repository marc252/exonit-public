import Foundation

public enum FPPVerdict: String, Codable, Equatable, Sendable {
    case validated
    case candidate
    case likelyFalsePositive = "likely_fp"
}

public enum FPPPolicy {
    public static let minimumSNR = 10.0
    public static let minimumCNNScore = 0.7
    public static let validatedFPP = 0.015
    public static let candidateFPP = 0.5
    public static let validatedNFPP = 0.001

    public static func shouldRun(
        vettingPassed: Bool, snr: Double, cnnScore: Double?
    ) -> Bool {
        guard vettingPassed, snr >= minimumSNR else { return false }
        if let cnnScore, cnnScore < minimumCNNScore { return false }
        return true
    }

    public static func verdict(
        fpp: Double, nfpp: Double?, highCrowding: Bool
    ) -> FPPVerdict {
        if fpp >= candidateFPP { return .likelyFalsePositive }
        if fpp < validatedFPP, let nfpp, nfpp < validatedNFPP {
            return highCrowding ? .candidate : .validated
        }
        return .candidate
    }
}
