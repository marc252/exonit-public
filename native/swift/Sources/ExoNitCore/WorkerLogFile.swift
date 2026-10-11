import Darwin
import Foundation

public enum WorkerLogFile {
    public static func resolve(_ configuredPath: String?, relativeTo directory: URL) -> URL? {
        guard let configuredPath = configuredPath?.trimmingCharacters(in: .whitespacesAndNewlines),
              !configuredPath.isEmpty else { return nil }
        if configuredPath.hasPrefix("/") {
            return URL(fileURLWithPath: configuredPath).standardizedFileURL
        }
        return directory.appendingPathComponent(configuredPath).standardizedFileURL
    }

    public static func append(_ message: String, to url: URL, fileManager: FileManager = .default) throws {
        try fileManager.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if !fileManager.fileExists(atPath: url.path) {
            guard fileManager.createFile(atPath: url.path, contents: nil) else {
                throw CocoaError(.fileWriteUnknown)
            }
        }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data((message + "\n").utf8))
    }

    public static func redirectStandardStreams(to url: URL, fileManager: FileManager = .default) throws {
        try fileManager.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let descriptor = Darwin.open(url.path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0o644)
        guard descriptor >= 0 else { throw posixError() }
        defer { _ = Darwin.close(descriptor) }
        guard Darwin.dup2(descriptor, STDOUT_FILENO) >= 0 else { throw posixError() }
        guard Darwin.dup2(descriptor, STDERR_FILENO) >= 0 else { throw posixError() }
    }

    private static func posixError() -> NSError {
        NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
    }
}
