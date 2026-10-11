import CoreML
import Foundation

public struct PiecewiseLinearCalibration: Equatable, Sendable {
    public let x: [Double]
    public let y: [Double]

    public init(x: [Double], y: [Double]) {
        self.x = x; self.y = y
    }

    public func apply(_ value: Double) -> Double {
        guard x.count == y.count, let firstX = x.first, let firstY = y.first,
              let lastX = x.last, let lastY = y.last, x.count >= 2 else { return value }
        if value <= firstX { return firstY }
        if value >= lastX { return lastY }
        for index in 1..<x.count where value <= x[index] {
            let fraction = (value - x[index - 1]) / (x[index] - x[index - 1])
            return y[index - 1] + fraction * (y[index] - y[index - 1])
        }
        return value
    }
}

public struct ExoMinerFFIScore: Equatable, Sendable {
    public let logit: Double
    public let rawProbability: Double
    public let probability: Double
}

public final class CoreMLExoMinerFFI: @unchecked Sendable {
    private static let expectedInputs: Set<String> = [
        "global_view", "local_view", "centroid_view", "centroid_mask",
        "stellar", "tce", "aux_views",
    ]
    private let model: MLModel
    private let outputName: String
    private let calibration: PiecewiseLinearCalibration?
    private let lock = NSLock()

    public init(
        modelURL: URL, calibration: PiecewiseLinearCalibration? = nil,
        computeUnits: MLComputeUnits = .all
    ) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let loadURL = modelURL.pathExtension == "mlmodelc"
            ? modelURL : try MLModel.compileModel(at: modelURL)
        model = try MLModel(contentsOf: loadURL, configuration: configuration)
        let inputs = Set(model.modelDescription.inputDescriptionsByName.keys)
        guard inputs == Self.expectedInputs else {
            throw CoreMLFFIError.invalidModelInputs(inputs)
        }
        let outputs = Set(model.modelDescription.outputDescriptionsByName.keys)
        guard outputs.count == 1, let output = outputs.first else {
            throw CoreMLFFIError.invalidModelOutputs(outputs)
        }
        outputName = output
        self.calibration = calibration
    }

    public func classify(features: ExoMinerFFIFeatures) throws -> ExoMinerFFIScore {
        guard features.global.count == 201, features.local.count == 61,
              features.centroid.count == 61, features.centroidMask.count == 1,
              features.stellar.count == 5, features.tce.count == 4,
              features.auxiliary.count == 3,
              features.auxiliary.allSatisfy({ $0.count == 61 }) else {
            throw CoreMLFFIError.invalidViewShape(
                global: features.global.count, local: features.local.count
            )
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "global_view": MLFeatureValue(multiArray: array(features.global, shape: [1, 1, 201])),
            "local_view": MLFeatureValue(multiArray: array(features.local, shape: [1, 1, 61])),
            "centroid_view": MLFeatureValue(multiArray: array(features.centroid, shape: [1, 1, 61])),
            "centroid_mask": MLFeatureValue(multiArray: array(features.centroidMask, shape: [1, 1])),
            "stellar": MLFeatureValue(multiArray: array(features.stellar, shape: [1, 5])),
            "tce": MLFeatureValue(multiArray: array(features.tce, shape: [1, 4])),
            "aux_views": MLFeatureValue(
                multiArray: array(features.auxiliary.flatMap { $0 }, shape: [1, 3, 61])
            ),
        ])
        lock.lock(); defer { lock.unlock() }
        let prediction = try model.prediction(from: provider)
        guard let output = prediction.featureValue(for: outputName)?.multiArrayValue,
              output.count > 0 else {
            throw CoreMLFFIError.missingOutput(outputName)
        }
        let logit = output[0].doubleValue
        let raw = FFIViewGenerator.sigmoid(logit: logit)
        return ExoMinerFFIScore(
            logit: logit, rawProbability: raw,
            probability: calibration?.apply(raw) ?? raw
        )
    }

    private func array(_ values: [Float], shape: [NSNumber]) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: shape, dataType: .float32)
        for (index, value) in values.enumerated() { result[index] = NSNumber(value: value) }
        return result
    }
}
