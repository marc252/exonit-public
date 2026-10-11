import CoreML
import Foundation

public struct FFIScore: Equatable, Sendable {
    public let logit: Double
    public let probability: Double

    public init(logit: Double, probability: Double) {
        self.logit = logit
        self.probability = probability
    }
}

public enum CoreMLFFIError: Error, CustomStringConvertible, Sendable {
    case invalidViewShape(global: Int, local: Int)
    case invalidModelInputs(Set<String>)
    case invalidModelOutputs(Set<String>)
    case missingOutput(String)

    public var description: String {
        switch self {
        case let .invalidViewShape(global, local):
            return "invalid CNN FFI shapes: global=\(global), local=\(local)"
        case let .invalidModelInputs(inputs): return "unexpected Core ML inputs: \(inputs.sorted())"
        case let .invalidModelOutputs(outputs): return "unexpected Core ML outputs: \(outputs.sorted())"
        case let .missingOutput(name): return "Core ML output \(name) is not a scalar/multi-array"
        }
    }
}

public final class CoreMLFFIClassifier: @unchecked Sendable {
    private let model: MLModel
    private let outputName: String
    private let lock = NSLock()

    public init(modelURL: URL, computeUnits: MLComputeUnits = .all) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let loadURL: URL
        if modelURL.pathExtension == "mlmodelc" {
            loadURL = modelURL
        } else {
            loadURL = try MLModel.compileModel(at: modelURL)
        }
        model = try MLModel(contentsOf: loadURL, configuration: configuration)
        let inputs = Set(model.modelDescription.inputDescriptionsByName.keys)
        guard inputs == ["global_view", "local_view"] else {
            throw CoreMLFFIError.invalidModelInputs(inputs)
        }
        let outputs = Set(model.modelDescription.outputDescriptionsByName.keys)
        guard outputs.count == 1, let output = outputs.first else {
            throw CoreMLFFIError.invalidModelOutputs(outputs)
        }
        outputName = output
    }

    public func classify(views: CNNFFIViews) throws -> FFIScore {
        guard views.global.count == FFIViewGenerator.globalBinCount,
              views.local.count == FFIViewGenerator.localBinCount else {
            throw CoreMLFFIError.invalidViewShape(
                global: views.global.count, local: views.local.count
            )
        }
        let global = try Self.multiArray(views.global, shape: [1, 1, 201])
        let local = try Self.multiArray(views.local, shape: [1, 1, 61])
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "global_view": MLFeatureValue(multiArray: global),
            "local_view": MLFeatureValue(multiArray: local),
        ])
        lock.lock()
        defer { lock.unlock() }
        let prediction = try model.prediction(from: provider)
        let feature = prediction.featureValue(for: outputName)
        let logit: Double
        if let array = feature?.multiArrayValue, array.count > 0 {
            logit = array[0].doubleValue
        } else if let feature, feature.type == .double {
            logit = feature.doubleValue
        } else if let feature, feature.type == .int64 {
            logit = Double(feature.int64Value)
        } else {
            throw CoreMLFFIError.missingOutput(outputName)
        }
        return FFIScore(logit: logit, probability: FFIViewGenerator.sigmoid(logit: logit))
    }

    private static func multiArray(_ values: [Float], shape: [NSNumber]) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: shape, dataType: .float32)
        for (index, value) in values.enumerated() {
            result[index] = NSNumber(value: value)
        }
        return result
    }
}
