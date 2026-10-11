// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ExoNitSwiftWorker",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "ExoNitCore", targets: ["ExoNitCore"]),
        .executable(name: "exonit-swift-worker", targets: ["ExoNitSwiftWorker"]),
        .executable(name: "exonit-contract-tests", targets: ["ExoNitContractTests"]),
        .executable(name: "exonit-bls-benchmark", targets: ["ExoNitBLSBenchmark"]),
        .executable(name: "exonit-fits-benchmark", targets: ["ExoNitFITSBenchmark"]),
        .executable(name: "exonit-detrend-benchmark", targets: ["ExoNitDetrendBenchmark"]),
        .executable(name: "exonit-fake-integration", targets: ["ExoNitFakeIntegration"]),
        .executable(name: "exonit-preprocess-dump", targets: ["ExoNitPreprocessDump"]),
        .executable(name: "exonit-bls-replay", targets: ["ExoNitBLSReplay"]),
        .executable(name: "exonit-fpp-tests", targets: ["ExoNitFPPTests"]),
    ],
    targets: [
        .target(
            name: "ExoNitCore",
            linkerSettings: [
                .linkedLibrary("sqlite3"),
                .linkedFramework("CoreML"),
                .linkedFramework("Metal"),
            ]
        ),
        .executableTarget(name: "ExoNitSwiftWorker", dependencies: ["ExoNitCore"]),
        .executableTarget(name: "ExoNitContractTests", dependencies: ["ExoNitCore"]),
        .executableTarget(name: "ExoNitBLSBenchmark", dependencies: ["ExoNitCore"]),
        .executableTarget(name: "ExoNitFITSBenchmark", dependencies: ["ExoNitCore"]),
        .executableTarget(name: "ExoNitDetrendBenchmark", dependencies: ["ExoNitCore"]),
        .executableTarget(name: "ExoNitFakeIntegration", dependencies: ["ExoNitCore"]),
        .executableTarget(name: "ExoNitPreprocessDump", dependencies: ["ExoNitCore"]),
        .executableTarget(name: "ExoNitBLSReplay", dependencies: ["ExoNitCore"]),
        .executableTarget(name: "ExoNitFPPTests", dependencies: ["ExoNitCore"]),
    ]
)
