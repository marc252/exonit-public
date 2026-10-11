import Foundation

public enum CLIHelp {
    public static func text(for arguments: [String]) -> String {
        if let first = arguments.first,
           URL(fileURLWithPath: first).lastPathComponent == "agent.py" {
            return """
            Usage: exonit-swift-worker agent.py start|stop|status|logs|version [worker-name] [options]

              start <worker-name> [--server-url URL] [--instances N]
                    [--instance-index I] [--config-overrides-b64 DATA]
                    --allow-start
              stop <worker-name> [--force]
              status <worker-name>
              logs [worker-name] [-n N]
              version

            Starting requires --allow-start or EXONIT_FARMCTL_ALLOW_START=1.
            """
        }
        return """
        Usage: exonit-swift-worker [--config PATH] [--worker-name NAME] [--server-url URL]

        Native ExoNit worker for Apple Silicon M4. The executable also accepts
        the farmctl compatibility form where the first argument is agent.py.
        """
    }
}
