// swift-tools-version: 5.7
import PackageDescription

let package = Package(
    name: "BodyMeasurementSDK",
    platforms: [
        .iOS(.v13)
    ],
    products: [
        .library(
            name: "BodyMeasurementSDK",
            targets: ["BodyMeasurementSDK"]),
    ],
    targets: [
        .target(
            name: "BodyMeasurementSDK",
            dependencies: []),
        .testTarget(
            name: "BodyMeasurementSDKTests",
            dependencies: ["BodyMeasurementSDK"]),
    ]
)
