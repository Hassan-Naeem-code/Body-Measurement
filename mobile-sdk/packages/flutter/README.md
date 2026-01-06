# Body Measurement Flutter SDK

AI-powered body measurement and size recommendation SDK for Flutter applications.

[![pub package](https://img.shields.io/pub/v/body_measurement_sdk.svg)](https://pub.dev/packages/body_measurement_sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🎯 **AI-Powered Measurements**: Extract precise body measurements from photos
- 👕 **Size Recommendations**: Get accurate size recommendations (85-92% accuracy)
- 🏢 **Product-Specific Sizing**: Support for brand-specific size charts
- 👔 **Fit Preferences**: Tight, regular, or loose fit options
- 👥 **Multi-Person Support**: Detect and measure multiple people in one image
- 📱 **Cross-Platform**: Works on both iOS and Android
- ⚡ **Fast & Reliable**: Optimized for mobile performance

## Installation

Add this to your package's `pubspec.yaml` file:

```yaml
dependencies:
  body_measurement_sdk: ^1.0.0
```

Then run:

```bash
flutter pub get
```

## Quick Start

### 1. Initialize the SDK

```dart
import 'package:body_measurement_sdk/body_measurement_sdk.dart';

final client = BodyMeasurementClient(
  apiKey: 'YOUR_API_KEY',
  baseUrl: 'https://api.yourdomain.com',
  debug: true,
);
```

### 2. Capture and Process Image

```dart
import 'package:image_picker/image_picker.dart';

// Capture image using camera
final picker = ImagePicker();
final XFile? image = await picker.pickImage(
  source: ImageSource.camera,
  imageQuality: 80,
);

if (image != null) {
  // Process measurement
  final result = await client.processMeasurement(
    imagePath: image.path,
    fitPreference: FitPreference.regular,
  );

  print('Recommended size: ${result.recommendedSize}');
  print('Confidence: ${result.sizeProbabilities[result.recommendedSize]}');
}
```

## API Reference

### BodyMeasurementClient

Main client for interacting with the API.

#### Constructor

```dart
BodyMeasurementClient({
  required String apiKey,
  String? baseUrl,
  int timeout = 30000,
  bool debug = false,
})
```

**Parameters:**
- `apiKey` (required): Your API key
- `baseUrl` (optional): API base URL
- `timeout` (optional): Request timeout in milliseconds (default: 30000)
- `debug` (optional): Enable debug logging (default: false)

#### Methods

##### `processMeasurement()`

Process a single-person body image and get measurements.

```dart
Future<MeasurementResult> processMeasurement({
  required String imagePath,
  String? productId,
  FitPreference fitPreference = FitPreference.regular,
  Function(double)? onProgress,
})
```

**Returns:** `MeasurementResult`

```dart
class MeasurementResult {
  final double shoulderWidth;
  final double chestWidth;
  final double waistWidth;
  final double hipWidth;
  final double inseam;
  final double armLength;
  final Map<String, double> confidenceScores;
  final String recommendedSize;
  final Map<String, double> sizeProbabilities;
  final int processingTimeMs;
}
```

##### `processMultiPersonMeasurement()`

Process an image with multiple people.

```dart
Future<MultiPersonResult> processMultiPersonMeasurement({
  required String imagePath,
  Function(double)? onProgress,
})
```

##### `getProducts()`

Get list of products with size charts.

```dart
Future<ProductList> getProducts({
  int skip = 0,
  int limit = 10,
  String? category,
})
```

##### `getProduct()`

Get a specific product by ID.

```dart
Future<Product> getProduct(String productId)
```

## Models

### FitPreference

```dart
enum FitPreference {
  tight,    // Sized down
  regular,  // Exact fit
  loose,    // Sized up
}
```

### Product

```dart
class Product {
  final String id;
  final String name;
  final String brand;
  final String category;
  final List<SizeChart> sizeCharts;
  final String imageUrl;
}
```

### SizeChart

```dart
class SizeChart {
  final String size;
  final double shoulderWidth;
  final double chestWidth;
  final double waistWidth;
  final double hipWidth;
  final double inseam;
  final double armLength;
}
```

## Example App

See the [`example`](example/) directory for a complete Flutter app demonstrating all SDK features.

To run the example:

```bash
cd example
flutter run
```

## Error Handling

```dart
try {
  final result = await client.processMeasurement(
    imagePath: image.path,
  );
} on ApiException catch (e) {
  print('API Error: ${e.message}');
  print('Status Code: ${e.statusCode}');
} catch (e) {
  print('Unknown error: $e');
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

### 2. Performance

```dart
// Compress images before upload
final image = await picker.pickImage(
  source: ImageSource.camera,
  imageQuality: 80,
  maxWidth: 1920,
  maxHeight: 1920,
);
```

### 3. Progress Tracking

```dart
double progress = 0;

final result = await client.processMeasurement(
  imagePath: image.path,
  onProgress: (p) {
    setState(() {
      progress = p;
    });
  },
);
```

## Platform Setup

### Android

Add to `android/app/src/main/AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
```

### iOS

Add to `ios/Runner/Info.plist`:

```xml
<key>NSCameraUsageDescription</key>
<string>We need camera access to capture body measurements</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>We need photo library access to select images</string>
```

## Requirements

- Flutter SDK: >=3.0.0
- Dart: >=3.0.0 <4.0.0
- iOS: >=13.0
- Android: API level 21+

## License

MIT © Body Measurement Platform

## Support

- **Documentation**: https://docs.yourdomain.com
- **Issues**: https://github.com/your-org/body-measurement-sdk/issues
- **Email**: support@yourdomain.com
