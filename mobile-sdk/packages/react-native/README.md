# Body Measurement React Native SDK

AI-powered body measurement and size recommendation SDK for React Native applications.

[![npm version](https://badge.fury.io/js/%40body-measurement%2Freact-native-sdk.svg)](https://badge.fury.io/js/%40body-measurement%2Freact-native-sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🎯 **AI-Powered Measurements**: Extract precise body measurements from photos
- 👕 **Size Recommendations**: Get accurate size recommendations (85-92% accuracy)
- 🏢 **Product-Specific Sizing**: Support for brand-specific size charts
- 👔 **Fit Preferences**: Tight, regular, or loose fit options
- 👥 **Multi-Person Support**: Detect and measure multiple people in one image
- 📱 **Easy Integration**: Simple React hooks and TypeScript support
- 📸 **Camera Ready**: Built-in camera and gallery integration
- ⚡ **Fast & Reliable**: Optimized for mobile performance

---

## Installation

```bash
npm install @body-measurement/react-native-sdk
```

### Additional Dependencies

For camera functionality, install:

```bash
npm install react-native-image-picker
```

For iOS:

```bash
cd ios && pod install
```

---

## Quick Start

### 1. Initialize the SDK

```typescript
import { BodyMeasurementClient } from '@body-measurement/react-native-sdk';

const client = new BodyMeasurementClient({
  apiKey: 'YOUR_API_KEY',
  baseURL: 'https://api.yourdomain.com',
  debug: __DEV__, // Enable debug logging in development
});
```

### 2. Process a Measurement

```typescript
import { useMeasurement, useCamera } from '@body-measurement/react-native-sdk';

function MyComponent() {
  const { loading, data, processMeasurement } = useMeasurement(client);
  const { captureImage } = useCamera();

  const handleCapture = async () => {
    // Capture image
    const image = await captureImage({ quality: 80 });
    if (!image) return;

    // Process measurement
    const result = await processMeasurement(image, {
      fitPreference: 'regular',
    });

    console.log('Recommended size:', result.recommended_size);
  };

  return (
    <View>
      <Button title="Take Photo" onPress={handleCapture} />
      {loading && <ActivityIndicator />}
      {data && <Text>Your size: {data.recommended_size}</Text>}
    </View>
  );
}
```

---

## API Reference

### BodyMeasurementClient

Main client for interacting with the API.

#### Constructor

```typescript
const client = new BodyMeasurementClient(config: SDKConfig)
```

**Config Options:**

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `apiKey` | string | ✅ | Your API key |
| `baseURL` | string | ❌ | API base URL (default: production URL) |
| `timeout` | number | ❌ | Request timeout in ms (default: 30000) |
| `debug` | boolean | ❌ | Enable debug logging (default: false) |

#### Methods

##### `processMeasurement(image, options, onProgress)`

Process a single-person body image and get measurements.

```typescript
const result = await client.processMeasurement(
  { uri: 'file:///path/to/image.jpg' },
  {
    productId: 'uuid-of-product', // Optional
    fitPreference: 'regular', // 'tight' | 'regular' | 'loose'
  },
  (progress) => console.log(`${progress}%`)
);
```

**Returns:** `MeasurementResponse`

```typescript
{
  shoulder_width: number;
  chest_width: number;
  waist_width: number;
  hip_width: number;
  inseam: number;
  arm_length: number;
  confidence_scores: Record<string, number>;
  recommended_size: string;
  size_probabilities: Record<string, number>;
  processing_time_ms: number;
}
```

##### `processMultiPersonMeasurement(image, options, onProgress)`

Process an image with multiple people.

```typescript
const result = await client.processMultiPersonMeasurement(
  { uri: 'file:///path/to/image.jpg' }
);

console.log(`Found ${result.valid_people_count} people`);
result.measurements.forEach((person) => {
  console.log(`Person ${person.person_id}: ${person.recommended_size}`);
});
```

**Returns:** `MultiPersonMeasurementResponse`

##### `getProducts(skip, limit, category)`

Get list of products with size charts.

```typescript
const { products, total } = await client.getProducts(0, 10, 'tops');
```

##### `getProduct(productId)`

Get a specific product by ID.

```typescript
const product = await client.getProduct('product-uuid');
console.log(product.size_charts);
```

---

## React Hooks

### `useMeasurement(client)`

Hook for processing measurements with loading/error states.

```typescript
const {
  loading,        // boolean
  error,          // Error | null
  data,           // MeasurementResponse | null
  progress,       // number (0-100)
  processMeasurement,  // (image, options) => Promise<MeasurementResponse>
  reset,          // () => void
} = useMeasurement(client);
```

**Example:**

```typescript
function MeasurementScreen() {
  const { loading, data, error, processMeasurement } = useMeasurement(client);

  const handleProcess = async (imageUri: string) => {
    try {
      await processMeasurement(
        { uri: imageUri },
        { fitPreference: 'regular' }
      );
    } catch (err) {
      Alert.alert('Error', err.message);
    }
  };

  return (
    <View>
      {loading && <ActivityIndicator />}
      {error && <Text>Error: {error.message}</Text>}
      {data && (
        <View>
          <Text>Size: {data.recommended_size}</Text>
          <Text>Confidence: {Math.round(data.size_probabilities[data.recommended_size] * 100)}%</Text>
        </View>
      )}
    </View>
  );
}
```

### `useMultiPersonMeasurement(client)`

Hook for processing multi-person measurements.

```typescript
const {
  loading,
  data,   // MultiPersonMeasurementResponse | null
  processMultiPersonMeasurement,
} = useMultiPersonMeasurement(client);
```

### `useCamera()`

Hook for camera and gallery integration.

```typescript
const {
  captureImage,        // (options?) => Promise<ImageSource | null>
  selectFromGallery,   // (options?) => Promise<ImageSource | null>
} = useCamera();
```

**Example:**

```typescript
const { captureImage, selectFromGallery } = useCamera();

const handleCamera = async () => {
  const image = await captureImage({
    cameraType: 'back',
    quality: 80,
    maxWidth: 1920,
    maxHeight: 1920,
  });

  if (image) {
    // Process the image
    await processMeasurement(image);
  }
};
```

### `useProducts(client, skip, limit, category)`

Hook for fetching products.

```typescript
const {
  loading,
  products,  // Product[]
  total,     // number
  refresh,   // () => void
} = useProducts(client, 0, 10, 'tops');
```

### `useProduct(client, productId)`

Hook for fetching a single product.

```typescript
const {
  loading,
  product,  // Product | null
  refresh,
} = useProduct(client, 'product-uuid');
```

---

## Advanced Usage

### Product-Specific Sizing

```typescript
// Get products
const { products } = useProducts(client);

// Select a product
const selectedProduct = products[0];

// Process measurement for specific product
const result = await processMeasurement(
  image,
  { productId: selectedProduct.id }
);

console.log(`Size for ${selectedProduct.name}: ${result.recommended_size}`);
```

### Fit Preferences

```typescript
const [fitPreference, setFitPreference] = useState<FitPreference>('regular');

const result = await processMeasurement(
  image,
  { fitPreference } // 'tight' | 'regular' | 'loose'
);

// If AI recommends M:
// - tight → Returns S (sized down)
// - regular → Returns M (exact fit)
// - loose → Returns L (sized up)
```

### Multi-Person Measurement

```typescript
const { processMultiPersonMeasurement } = useMultiPersonMeasurement(client);

const result = await processMultiPersonMeasurement(image);

console.log(`Found ${result.valid_people_count} valid people`);

result.measurements.forEach((person) => {
  if (person.is_valid) {
    console.log(`Person ${person.person_id}:`);
    console.log(`  Size: ${person.recommended_size}`);
    console.log(`  Confidence: ${person.validation_confidence}`);
  } else {
    console.log(`Person ${person.person_id} is invalid:`);
    console.log(`  Missing: ${person.missing_parts.join(', ')}`);
  }
});
```

### Error Handling

```typescript
import { SDKError } from '@body-measurement/react-native-sdk';

try {
  const result = await processMeasurement(image);
} catch (error) {
  if (error instanceof SDKError) {
    console.error('API Error:', error.message);
    console.error('Status Code:', error.statusCode);
    console.error('Response:', error.response);
  } else {
    console.error('Unknown error:', error);
  }
}
```

---

## TypeScript Support

The SDK is fully typed with TypeScript.

```typescript
import type {
  MeasurementResponse,
  Product,
  SizeChart,
  FitPreference,
  ImageSource,
} from '@body-measurement/react-native-sdk';
```

---

## Example App

See the `/example` directory for a complete React Native app demonstrating all SDK features.

To run the example:

```bash
cd example
npm install
npm run ios   # or npm run android
```

---

## Permissions

### iOS (Info.plist)

```xml
<key>NSCameraUsageDescription</key>
<string>We need camera access to capture body measurements</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>We need photo library access to select images</string>
```

### Android (AndroidManifest.xml)

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
```

---

## Best Practices

### 1. Image Quality

For best results:
- Use full-body photos (head to feet visible)
- Ensure good lighting
- Avoid baggy clothing
- Stand against plain background
- Face camera directly

### 2. Error Handling

Always handle errors gracefully:

```typescript
const { processMeasurement } = useMeasurement(client);

const handleProcess = async () => {
  try {
    await processMeasurement(image);
  } catch (error) {
    if (error.statusCode === 422) {
      Alert.alert('Invalid Image', 'Please ensure full body is visible');
    } else {
      Alert.alert('Error', 'Failed to process measurement');
    }
  }
};
```

### 3. Performance

- Compress images before upload (quality: 70-80)
- Limit max dimensions (1920x1920 recommended)
- Show progress indicator during processing
- Cache product data when possible

### 4. User Experience

```typescript
function MeasurementFlow() {
  const { loading, progress } = useMeasurement(client);

  return (
    <View>
      {loading && (
        <View>
          <ActivityIndicator size="large" />
          <Text>Processing... {progress}%</Text>
        </View>
      )}
    </View>
  );
}
```

---

## Troubleshooting

### Camera not working

**Error:** `Failed to capture image`

**Solution:** Make sure `react-native-image-picker` is installed and permissions are granted.

### API timeout

**Error:** `Request timeout`

**Solution:** Increase timeout in config:

```typescript
const client = new BodyMeasurementClient({
  apiKey: 'YOUR_KEY',
  timeout: 60000, // 60 seconds
});
```

### Invalid API key

**Error:** `401 Unauthorized`

**Solution:** Verify your API key is correct and active.

---

## Support

- **Documentation:** [Full API Docs](https://docs.yourdomain.com)
- **Issues:** [GitHub Issues](https://github.com/your-org/sdk/issues)
- **Email:** support@yourdomain.com

---

## License

MIT © Body Measurement Platform

---

## Changelog

### v1.0.0 (2025-01-05)

- ✅ Initial release
- ✅ Single & multi-person measurement support
- ✅ Product-specific sizing
- ✅ Fit preferences (tight/regular/loose)
- ✅ React hooks for easy integration
- ✅ Camera & gallery support
- ✅ Full TypeScript support
