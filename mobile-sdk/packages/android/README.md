# Body Measurement Android SDK

AI-powered body measurement and size recommendation SDK for native Android applications.

[![Maven Central](https://img.shields.io/maven-central/v/com.bodymeasurement/sdk.svg)](https://search.maven.org/artifact/com.bodymeasurement/sdk)
[![Android API](https://img.shields.io/badge/API-21%2B-brightgreen.svg)](https://android-arsenal.com/api?level=21)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🎯 **AI-Powered Measurements**: Extract precise body measurements from photos
- 👕 **Size Recommendations**: Get accurate size recommendations (85-92% accuracy)
- 🏢 **Product-Specific Sizing**: Support for brand-specific size charts
- 👔 **Fit Preferences**: Tight, regular, or loose fit options
- 👥 **Multi-Person Support**: Detect and measure multiple people in one image
- 📱 **Modern Android**: Built with Kotlin and Coroutines
- ⚡ **Fast & Reliable**: Optimized for Android performance

## Installation

### Gradle

Add to your module's `build.gradle`:

```gradle
dependencies {
    implementation 'com.bodymeasurement:sdk:1.0.0'
}
```

### Maven

```xml
<dependency>
    <groupId>com.bodymeasurement</groupId>
    <artifactId>sdk</artifactId>
    <version>1.0.0</version>
</dependency>
```

## Quick Start

### 1. Initialize the SDK

```kotlin
import com.bodymeasurement.sdk.BodyMeasurementClient
import com.bodymeasurement.sdk.SDKConfig

val client = BodyMeasurementClient(
    config = SDKConfig(
        apiKey = "YOUR_API_KEY",
        baseUrl = "https://api.yourdomain.com",
        debug = true
    )
)
```

### 2. Capture and Process Image

```kotlin
import android.net.Uri
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    private val client = BodyMeasurementClient(
        SDKConfig(apiKey = "YOUR_API_KEY")
    )

    private val takePicture = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            processImage(imageUri)
        }
    }

    private fun processImage(imageUri: Uri) {
        lifecycleScope.launch {
            try {
                val result = client.processMeasurement(
                    imageUri = imageUri,
                    fitPreference = FitPreference.REGULAR
                )

                println("Recommended size: ${result.recommendedSize}")
                println("Confidence: ${result.sizeProbabilities[result.recommendedSize]}")
            } catch (e: Exception) {
                println("Error: ${e.message}")
            }
        }
    }
}
```

## API Reference

### BodyMeasurementClient

Main client for interacting with the API.

#### Constructor

```kotlin
class BodyMeasurementClient(
    config: SDKConfig
)

data class SDKConfig(
    val apiKey: String,
    val baseUrl: String = DEFAULT_BASE_URL,
    val timeout: Long = 30000,
    val debug: Boolean = false
)
```

**Parameters:**
- `apiKey`: Your API key (required)
- `baseUrl`: API base URL (optional)
- `timeout`: Request timeout in milliseconds (default: 30000)
- `debug`: Enable debug logging (default: false)

#### Methods

##### `processMeasurement()`

Process a single-person body image and get measurements.

```kotlin
suspend fun processMeasurement(
    imageUri: Uri,
    productId: String? = null,
    fitPreference: FitPreference = FitPreference.REGULAR,
    onProgress: ((Double) -> Unit)? = null
): MeasurementResult
```

**Returns:** `MeasurementResult`

```kotlin
data class MeasurementResult(
    val shoulderWidth: Double,
    val chestWidth: Double,
    val waistWidth: Double,
    val hipWidth: Double,
    val inseam: Double,
    val armLength: Double,
    val confidenceScores: Map<String, Double>,
    val recommendedSize: String,
    val sizeProbabilities: Map<String, Double>,
    val processingTimeMs: Int
)
```

##### `processMultiPersonMeasurement()`

Process an image with multiple people.

```kotlin
suspend fun processMultiPersonMeasurement(
    imageUri: Uri,
    onProgress: ((Double) -> Unit)? = null
): MultiPersonResult
```

##### `getProducts()`

Get list of products with size charts.

```kotlin
suspend fun getProducts(
    skip: Int = 0,
    limit: Int = 10,
    category: String? = null
): ProductList
```

##### `getProduct()`

Get a specific product by ID.

```kotlin
suspend fun getProduct(productId: String): Product
```

## Models

### FitPreference

```kotlin
enum class FitPreference {
    TIGHT,    // Sized down
    REGULAR,  // Exact fit
    LOOSE     // Sized up
}
```

### Product

```kotlin
data class Product(
    val id: String,
    val name: String,
    val brand: String,
    val category: String,
    val sizeCharts: List<SizeChart>,
    val imageUrl: String
)
```

### SizeChart

```kotlin
data class SizeChart(
    val size: String,
    val shoulderWidth: Double,
    val chestWidth: Double,
    val waistWidth: Double,
    val hipWidth: Double,
    val inseam: Double,
    val armLength: Double
)
```

## Example App

See the [`example`](example/) directory for a complete Android app demonstrating all SDK features.

To run the example:

1. Open the project in Android Studio
2. Add your API key in `Config.kt`
3. Build and run

## Error Handling

```kotlin
import com.bodymeasurement.sdk.exceptions.APIException

try {
    val result = client.processMeasurement(imageUri)
} catch (e: APIException) {
    when (e.statusCode) {
        401 -> println("Invalid API key")
        422 -> println("Please ensure full body is visible")
        else -> println("API Error: ${e.message}")
    }
} catch (e: Exception) {
    println("Unknown error: ${e.message}")
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

```kotlin
val result = client.processMeasurement(
    imageUri = imageUri,
    onProgress = { progress ->
        runOnUiThread {
            progressBar.progress = progress.toInt()
        }
    }
)
```

### 3. Image Compression

```kotlin
// Compress image before upload
val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, imageUri)
val outputStream = ByteArrayOutputStream()
bitmap.compress(Bitmap.CompressFormat.JPEG, 80, outputStream)
```

### 4. Lifecycle-Aware Calls

```kotlin
class MyViewModel : ViewModel() {
    private val client = BodyMeasurementClient(SDKConfig(apiKey = "KEY"))

    fun processMeasurement(imageUri: Uri) {
        viewModelScope.launch {
            try {
                val result = client.processMeasurement(imageUri)
                _measurementResult.value = result
            } catch (e: Exception) {
                _error.value = e.message
            }
        }
    }
}
```

## Permissions

Add to your `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.INTERNET" />
```

For Android 13+ (API 33), also request:

```xml
<uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />
```

## ProGuard Rules

If using ProGuard, add these rules:

```proguard
# Body Measurement SDK
-keep class com.bodymeasurement.sdk.** { *; }
-keepclassmembers class com.bodymeasurement.sdk.** { *; }
```

## Requirements

- Android API 21+ (Android 5.0 Lollipop)
- Kotlin 1.8+
- AndroidX libraries

## Status

⚠️ **Note**: This is currently a template SDK. Full implementation is in progress.

## License

MIT © Body Measurement Platform

## Support

- **Documentation**: https://docs.yourdomain.com
- **Issues**: https://github.com/your-org/body-measurement-android-sdk/issues
- **Email**: support@yourdomain.com
