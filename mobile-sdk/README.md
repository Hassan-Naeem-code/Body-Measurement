# Body Measurement Mobile SDKs

Multi-platform SDKs for integrating AI-powered body measurement and size recommendation into your mobile applications.

## 📦 Available SDKs

| Platform | Status | Language | Directory | Documentation |
|----------|--------|----------|-----------|---------------|
| **React Native** | ✅ Production | TypeScript | [`packages/react-native`](packages/react-native) | [README](packages/react-native/README.md) |
| **Flutter** | ✅ Production | Dart | [`packages/flutter`](packages/flutter) | [README](packages/flutter/README.md) |
| **iOS** | 📝 Template | Swift | [`packages/ios`](packages/ios) | [README](packages/ios/README.md) |
| **Android** | 📝 Template | Kotlin/Java | [`packages/android`](packages/android) | [README](packages/android/README.md) |

## 🚀 Quick Start

Choose your platform and follow the respective SDK documentation:

### React Native
```bash
npm install @body-measurement/react-native-sdk
```
[View React Native Documentation →](packages/react-native/README.md)

### Flutter
```yaml
dependencies:
  body_measurement_sdk: ^1.0.0
```
[View Flutter Documentation →](packages/flutter/README.md)

### iOS (Swift Package Manager)
```swift
dependencies: [
    .package(url: "https://github.com/your-org/body-measurement-ios-sdk", from: "1.0.0")
]
```
[View iOS Documentation →](packages/ios/README.md)

### Android (Gradle)
```gradle
dependencies {
    implementation 'com.bodymeasurement:sdk:1.0.0'
}
```
[View Android Documentation →](packages/android/README.md)

## ✨ Features

All SDKs provide:
- 🎯 **AI-Powered Measurements**: Extract precise body measurements from photos
- 👕 **Size Recommendations**: Get accurate size recommendations (85-92% accuracy)
- 🏢 **Product-Specific Sizing**: Support for brand-specific size charts
- 👔 **Fit Preferences**: Tight, regular, or loose fit options
- 👥 **Multi-Person Support**: Detect and measure multiple people in one image
- ⚡ **Fast & Reliable**: Optimized for mobile performance

## 📖 Documentation

- [Multi-Platform SDK Guide](docs/MULTI_PLATFORM_SDK_GUIDE.md) - Comprehensive guide covering all platforms
- [API Reference](https://docs.yourdomain.com/api) - Full API documentation
- [Best Practices](https://docs.yourdomain.com/best-practices) - Tips for optimal results

## 🏗️ Repository Structure

```
mobile-sdk/
├── packages/
│   ├── react-native/     # React Native SDK (TypeScript)
│   │   ├── src/          # Source code
│   │   ├── example/      # Example app
│   │   └── README.md     # React Native docs
│   ├── flutter/          # Flutter SDK (Dart)
│   │   ├── lib/          # Source code
│   │   ├── example/      # Example app
│   │   └── README.md     # Flutter docs
│   ├── ios/              # Native iOS SDK (Swift)
│   │   ├── Sources/      # Source code
│   │   ├── Example/      # Example app
│   │   └── README.md     # iOS docs
│   └── android/          # Native Android SDK (Kotlin)
│       ├── bodymeasurementsdk/  # Source code
│       └── README.md     # Android docs
├── docs/                 # Shared documentation
└── README.md            # This file
```

## 🔧 Development

### Prerequisites

- **React Native**: Node.js 16+, React Native CLI
- **Flutter**: Flutter SDK 3.0+, Dart 2.17+
- **iOS**: Xcode 14+, Swift 5.7+
- **Android**: Android Studio, Gradle 7+

### Building from Source

Each SDK can be built independently. Navigate to the respective package directory and follow the build instructions in its README.

## 📝 Examples

Each SDK includes a complete example application demonstrating all features:

- **React Native**: [`packages/react-native/example`](packages/react-native/example)
- **Flutter**: [`packages/flutter/example`](packages/flutter/example)
- **iOS**: [`packages/ios/Example`](packages/ios/Example)
- **Android**: [`packages/android/example`](packages/android/example)

## 🔑 API Key

All SDKs require an API key. Get yours at:
https://dashboard.yourdomain.com/api-keys

## 💡 Use Cases

- E-commerce size recommendations
- Virtual fitting rooms
- Fashion retail applications
- Custom tailoring apps
- Fitness and health tracking

## 📄 License

MIT © Body Measurement Platform

See individual package licenses for more details.

## 🤝 Support

- **Documentation**: https://docs.yourdomain.com
- **Issues**: https://github.com/your-org/mobile-sdks/issues
- **Email**: support@yourdomain.com
- **Discord**: https://discord.gg/your-community

## 🔄 Updates

Subscribe to our changelog to stay updated:
https://docs.yourdomain.com/changelog

---

Made with ❤️ by the Body Measurement Platform team
