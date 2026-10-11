import Foundation

public enum AgentConfigError: Error, CustomStringConvertible, Sendable {
    case invalidJSONObject(String)
    case invalidInstance(index: Int, count: Int)

    public var description: String {
        switch self {
        case let .invalidJSONObject(reason): return "Configuration must be a JSON object: \(reason)"
        case let .invalidInstance(index, count): return "Invalid instance \(index) of \(count)"
        }
    }
}

public enum AgentConfigDeriver {
    public static func applySingleInstanceOverrides(
        baseConfigData: Data,
        overridesData: Data
    ) throws -> Data {
        var base = try object(from: baseConfigData)
        let overrides = try object(from: overridesData)
        for (key, value) in overrides { base[key] = value }
        return try encoded(base)
    }

    public static func derive(
        baseConfigData: Data,
        overridesData: Data?,
        workerName: String,
        instanceIndex: Int,
        instances: Int,
        workerDirectory: URL,
        cpuCount: Int
    ) throws -> Data {
        guard instances > 0, instanceIndex > 0, instanceIndex <= instances else {
            throw AgentConfigError.invalidInstance(index: instanceIndex, count: instances)
        }
        let rawBase = try object(from: baseConfigData)
        var config = rawBase
        if let overridesData {
            for (key, value) in try object(from: overridesData) { config[key] = value }
        }

        config["worker_name"] = workerName
        let baseGUIPort = integer(rawBase["gui_port"]) ?? 8401
        config["gui_port"] = baseGUIPort + instanceIndex - 1
        if instanceIndex > 1 { config["gui_enabled"] = false }

        let baseFits = rawBase["fits_cache_dir"] as? String ?? ""
        let cacheURL: URL
        if baseFits.isEmpty {
            cacheURL = workerDirectory.appendingPathComponent("fits_cache_\(instanceIndex)")
        } else {
            let resolved = baseFits.hasPrefix("/")
                ? URL(fileURLWithPath: baseFits)
                : workerDirectory.appendingPathComponent(baseFits)
            cacheURL = resolved.deletingLastPathComponent()
                .appendingPathComponent("\(resolved.lastPathComponent)_\(instanceIndex)")
        }
        config["fits_cache_dir"] = cacheURL.standardizedFileURL.path
        config["log_file"] = "worker-\(workerName).log"

        let budget = Int(floor(0.75 * Double(max(cpuCount, 1))))
        let baseShare = budget / instances
        let remainder = budget % instances
        let threads = max(1, baseShare + (instanceIndex <= remainder ? 1 : 0))
        config["cpu_threads"] = threads
        config["fpp_cpu_cores"] = threads
        return try encoded(config)
    }

    public static func decodeOverrides(base64: String) throws -> Data {
        guard let data = Data(base64Encoded: base64) else {
            throw AgentConfigError.invalidJSONObject("invalid base64")
        }
        _ = try object(from: data)
        return data
    }

    private static func object(from data: Data) throws -> [String: Any] {
        do {
            guard let result = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw AgentConfigError.invalidJSONObject("top-level value is not an object")
            }
            return result
        } catch let error as AgentConfigError {
            throw error
        } catch {
            throw AgentConfigError.invalidJSONObject(String(describing: error))
        }
    }

    private static func encoded(_ object: [String: Any]) throws -> Data {
        do {
            return try JSONSerialization.data(
                withJSONObject: object, options: [.prettyPrinted, .sortedKeys]
            )
        } catch {
            throw AgentConfigError.invalidJSONObject(String(describing: error))
        }
    }

    private static func integer(_ value: Any?) -> Int? {
        if let value = value as? Int { return value }
        if let number = value as? NSNumber { return number.intValue }
        return nil
    }
}
