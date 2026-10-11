import Foundation

public enum Invocation: Equatable, Sendable {
    case worker(WorkerArguments)
    case agent(AgentArguments)
}

public struct WorkerArguments: Equatable, Sendable {
    public var configPath: String
    public var workerName: String?
    public var serverURL: String?

    public init(configPath: String = "config.json", workerName: String? = nil, serverURL: String? = nil) {
        self.configPath = configPath
        self.workerName = workerName
        self.serverURL = serverURL
    }
}

public enum AgentAction: String, Equatable, Sendable {
    case start, stop, status, logs, version
}

public struct AgentArguments: Equatable, Sendable {
    public var action: AgentAction
    public var workerName: String?
    public var serverURL: String?
    public var instances: Int
    public var instanceIndex: Int
    public var configOverridesBase64: String?
    public var allowStart: Bool
    public var force: Bool
    public var logLines: Int

    public init(
        action: AgentAction,
        workerName: String? = nil,
        serverURL: String? = nil,
        instances: Int = 1,
        instanceIndex: Int = 1,
        configOverridesBase64: String? = nil,
        allowStart: Bool = false,
        force: Bool = false,
        logLines: Int = 50
    ) {
        self.action = action
        self.workerName = workerName
        self.serverURL = serverURL
        self.instances = instances
        self.instanceIndex = instanceIndex
        self.configOverridesBase64 = configOverridesBase64
        self.allowStart = allowStart
        self.force = force
        self.logLines = logLines
    }
}

public enum InvocationParseError: Error, Equatable, CustomStringConvertible, Sendable {
    case missingValue(String)
    case unknownOption(String)
    case invalidInteger(option: String, value: String)
    case invalidAgentAction(String)
    case missingWorkerName(AgentAction)
    case invalidWorkerName(String)
    case unexpectedArgument(String)

    public var description: String {
        switch self {
        case let .missingValue(option): return "Missing value for \(option)"
        case let .unknownOption(option): return "Unknown option \(option)"
        case let .invalidInteger(option, value): return "Invalid integer for \(option): \(value)"
        case let .invalidAgentAction(value): return "Invalid agent action: \(value)"
        case let .missingWorkerName(action): return "Agent action \(action.rawValue) requires a worker name"
        case let .invalidWorkerName(value): return "Invalid worker name: \(value)"
        case let .unexpectedArgument(value): return "Unexpected argument: \(value)"
        }
    }
}

public enum WorkerNameValidator {
    public static func isValid(_ value: String) -> Bool {
        value.range(
            of: #"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"#,
            options: .regularExpression
        ) != nil && !value.contains("..")
    }
}

public enum InvocationParser {
    public static func parse(_ arguments: [String]) throws -> Invocation {
        if let first = arguments.first, URL(fileURLWithPath: first).lastPathComponent == "agent.py" {
            return .agent(try parseAgent(Array(arguments.dropFirst())))
        }
        return .worker(try parseWorker(arguments))
    }

    private static func parseWorker(_ arguments: [String]) throws -> WorkerArguments {
        var result = WorkerArguments()
        var index = 0
        while index < arguments.count {
            let option = arguments[index]
            switch option {
            case "--config", "--worker-name", "--server-url":
                let value = try value(after: option, at: &index, in: arguments)
                switch option {
                case "--config": result.configPath = value
                case "--worker-name": result.workerName = value
                default: result.serverURL = value
                }
            default:
                if option.hasPrefix("--config=") {
                    result.configPath = String(option.dropFirst("--config=".count))
                } else if option.hasPrefix("--worker-name=") {
                    result.workerName = String(option.dropFirst("--worker-name=".count))
                } else if option.hasPrefix("--server-url=") {
                    result.serverURL = String(option.dropFirst("--server-url=".count))
                } else if option.hasPrefix("-") {
                    throw InvocationParseError.unknownOption(option)
                } else {
                    throw InvocationParseError.unexpectedArgument(option)
                }
            }
            index += 1
        }
        if let workerName = result.workerName { try validateWorkerName(workerName) }
        return result
    }

    private static func parseAgent(_ arguments: [String]) throws -> AgentArguments {
        guard let rawAction = arguments.first else {
            throw InvocationParseError.missingValue("agent action")
        }
        guard let action = AgentAction(rawValue: rawAction) else {
            throw InvocationParseError.invalidAgentAction(rawAction)
        }
        var result = AgentArguments(action: action)
        var index = 1

        if index < arguments.count, !arguments[index].hasPrefix("-") {
            result.workerName = arguments[index]
            index += 1
        }

        while index < arguments.count {
            let option = arguments[index]
            switch option {
            case "--server-url", "--instances", "--instance-index", "--config-overrides-b64", "-n":
                let raw = try value(after: option, at: &index, in: arguments)
                switch option {
                case "--server-url": result.serverURL = raw
                case "--config-overrides-b64": result.configOverridesBase64 = raw
                case "--instances": result.instances = try positiveInteger(raw, option: option)
                case "--instance-index": result.instanceIndex = try positiveInteger(raw, option: option)
                default:
                    guard let n = Int(raw), n >= 0 else {
                        throw InvocationParseError.invalidInteger(option: option, value: raw)
                    }
                    result.logLines = n
                }
            case "--allow-start": result.allowStart = true
            case "--force": result.force = true
            default:
                if option.hasPrefix("-") {
                    throw InvocationParseError.unknownOption(option)
                }
                throw InvocationParseError.unexpectedArgument(option)
            }
            index += 1
        }

        if [.start, .stop, .status].contains(action), result.workerName == nil {
            throw InvocationParseError.missingWorkerName(action)
        }
        if let workerName = result.workerName { try validateWorkerName(workerName) }
        return result
    }

    private static func validateWorkerName(_ value: String) throws {
        guard WorkerNameValidator.isValid(value) else {
            throw InvocationParseError.invalidWorkerName(value)
        }
    }

    private static func value(after option: String, at index: inout Int, in arguments: [String]) throws -> String {
        let next = index + 1
        guard next < arguments.count else { throw InvocationParseError.missingValue(option) }
        index = next
        return arguments[next]
    }

    private static func positiveInteger(_ raw: String, option: String) throws -> Int {
        guard let value = Int(raw), value > 0 else {
            throw InvocationParseError.invalidInteger(option: option, value: raw)
        }
        return value
    }
}
