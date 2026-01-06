# Body Measurement iOS SDK

AI-powered body measurement and size recommendation SDK for native iOS applications.

[![Swift Version](https://img.shields.io/badge/Swift-5.7+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/platform-iOS%2013.0+-lightgrey.svg)](https://developer.apple.com/ios/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🎯 **AI-Powered Measurements**: Extract precise body measurements from photos
- 👕 **Size Recommendations**: Get accurate size recommendations (85-92% accuracy)
- 🏢 **Product-Specific Sizing**: Support for brand-specific size charts
- 👔 **Fit Preferences**: Tight, regular, or loose fit options
- 👥 **Multi-Person Support**: Detect and measure multiple people in one image
- 📱 **Modern Swift**: Built with latest Swift features and async/await
- ⚡ **Fast & Reliable**: Optimized for iOS performance

## Installation

### Swift Package Manager

Add the following to your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/your-org/body-measurement-ios-sdk", from: "1.0.0")
]
```

Or in Xcode:
1. File → Add Packages...
2. Enter package URL: `https://github.com/your-org/body-measurement-ios-sdk`
3. Select version and add to your target

### CocoaPods

Add to your `Podfile`:

```ruby
pod 'BodyMeasurementSDK', '~> 1.0'
```

Then run:

```bash
pod install
```

## Quick Start

### 1. Initialize the SDK

```swift
import BodyMeasurementSDK

let client = BodyMeasurementClient(
    apiKey: "YOUR_API_KEY",
    baseURL: "https://api.yourdomain.com",
    debug: true
)
```

### 2. Capture and Process Image

```swift
import UIKit
import BodyMeasurementSDK

class ViewController: UIViewController {
    let client = BodyMeasurementClient(apiKey: "YOUR_API_KEY")

    func captureAndProcess() async {
        // Present image picker
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        present(picker, animated: true)

        // After getting image...
        do {
            let result = try await client.processMeasurement(
                image: capturedImage,
                fitPreference: .regular
            )

            print("Recommended size: \(result.recommendedSize)")
            print("Confidence: \(result.sizeProbabilities[result.recommendedSize] ?? 0)")
        } catch {
            print("Error: \(error)")
        }
    }
}
```

## API Reference

### BodyMeasurementClient

Main client for interacting with the API.

#### Initializer

```swift
init(
    apiKey: String,
    baseURL: String? = nil,
    timeout: TimeInterval = 30,
    debug: Bool = false
)
```

**Parameters:**
- `apiKey`: Your API key (required)
- `baseURL`: API base URL (optional)
- `timeout`: Request timeout in seconds (default: 30)
- `debug`: Enable debug logging (default: false)

#### Methods

##### `processMeasurement(image:productId:fitPreference:progress:)`

Process a single-person body image and get measurements.

```swift
func processMeasurement(
    image: UIImage,
    productId: String? = nil,
    fitPreference: FitPreference = .regular,
    progress: ((Double) -> Void)? = nil
) async throws -> MeasurementResult
```

**Returns:** `MeasurementResult`

```swift
struct MeasurementResult {
    let shoulderWidth: Double
    let chestWidth: Double
    let waistWidth: Double
    let hipWidth: Double
    let inseam: Double
    let armLength: Double
    let confidenceScores: [String: Double]
    let recommendedSize: String
    let sizeProbabilities: [String: Double]
    let processingTimeMs: Int
}
```

##### `processMultiPersonMeasurement(image:progress:)`

Process an image with multiple people.

```swift
func processMultiPersonMeasurement(
    image: UIImage,
    progress: ((Double) -> Void)? = nil
) async throws -> MultiPersonResult
```

##### `getProducts(skip:limit:category:)`

Get list of products with size charts.

```swift
func getProducts(
    skip: Int = 0,
    limit: Int = 10,
    category: String? = nil
) async throws -> ProductList
```

##### `getProduct(id:)`

Get a specific product by ID.

```swift
func getProduct(id: String) async throws -> Product
```

## Models

### FitPreference

```swift
enum FitPreference: String {
    case tight    // Sized down
    case regular  // Exact fit
    case loose    // Sized up
}
```

### Product

```swift
struct Product: Codable {
    let id: String
    let name: String
    let brand: String
    let category: String
    let sizeCharts: [SizeChart]
    let imageUrl: String
}
```

### SizeChart

```swift
struct SizeChart: Codable {
    let size: String
    let shoulderWidth: Double
    let chestWidth: Double
    let waistWidth: Double
    let hipWidth: Double
    let inseam: Double
    let armLength: Double
}
```

## Example App

See the [`Example`](Example/) directory for a complete iOS app demonstrating all SDK features.

To run the example:

1. Open `Example.xcodeproj` in Xcode
2. Add your API key in `Config.swift`
3. Build and run

## Error Handling

```swift
do {
    let result = try await client.processMeasurement(image: image)
} catch let error as APIError {
    switch error {
    case .unauthorized:
        print("Invalid API key")
    case .invalidImage:
        print("Please ensure full body is visible")
    case .networkError(let message):
        print("Network error: \(message)")
    default:
        print("Error: \(error)")
    }
} catch {
    print("Unknown error: \(error)")
}
```

## Best Practices

### 1. Image Quality

For best results:
- Use full-body photos (head to feet visible)
- Ensure good lighting
- Avoid baggy clothing
- Stand against plain background
- Face camera directly

### 2. Progress Tracking

```swift
let result = try await client.processMeasurement(
    image: image,
    progress: { progress in
        DispatchQueue.main.async {
            self.progressView.progress = Float(progress / 100)
        }
    }
)
```

### 3. Image Compression

```swift
// Compress image before upload
guard let imageData = image.jpegData(compressionQuality: 0.8),
      let compressedImage = UIImage(data: imageData) else {
    return
}
```

## Permissions

Add to your `Info.plist`:

```xml
<key>NSCameraUsageDescription</key>
<string>We need camera access to capture body measurements</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>We need photo library access to select images</string>
```

## Requirements

- iOS 13.0+
- Xcode 14.0+
- Swift 5.7+

## Status

⚠️ **Note**: This is currently a template SDK. Full implementation is in progress.

## License

MIT © Body Measurement Platform

## Support

- **Documentation**: https://docs.yourdomain.com
- **Issues**: https://github.com/your-org/body-measurement-ios-sdk/issues
- **Email**: support@yourdomain.com
