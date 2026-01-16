'use client';

import { useState } from 'react';
import { authHelpers } from '@/lib/auth';
import {
  Code2,
  Copy,
  Check,
  ExternalLink,
  Smartphone,
  Globe,
  Terminal,
  BookOpen,
  Zap,
  Shield,
  Users,
  Gauge,
  ChevronRight,
  Package,
} from 'lucide-react';

type Platform = 'react-native' | 'flutter' | 'ios' | 'android' | 'web' | 'api';

interface SDKPlatform {
  id: Platform;
  name: string;
  icon: string;
  language: string;
  status: 'production' | 'beta';
  installCommand: string;
  packageName: string;
  docsUrl: string;
  description: string;
}

const sdkPlatforms: SDKPlatform[] = [
  {
    id: 'react-native',
    name: 'React Native',
    icon: '⚛️',
    language: 'TypeScript',
    status: 'production',
    installCommand: 'npm install @body-measurement/react-native-sdk',
    packageName: '@body-measurement/react-native-sdk',
    docsUrl: '#react-native',
    description: 'Full-featured SDK with React hooks for seamless integration',
  },
  {
    id: 'flutter',
    name: 'Flutter',
    icon: '🐦',
    language: 'Dart',
    status: 'production',
    installCommand: 'flutter pub add body_measurement_sdk',
    packageName: 'body_measurement_sdk',
    docsUrl: '#flutter',
    description: 'Cross-platform SDK for iOS and Android Flutter apps',
  },
  {
    id: 'ios',
    name: 'iOS',
    icon: '🍎',
    language: 'Swift',
    status: 'production',
    installCommand: '.package(url: "https://github.com/body-measurement/ios-sdk.git", from: "1.0.0")',
    packageName: 'BodyMeasurementSDK',
    docsUrl: '#ios',
    description: 'Native Swift SDK with async/await support',
  },
  {
    id: 'android',
    name: 'Android',
    icon: '🤖',
    language: 'Kotlin',
    status: 'production',
    installCommand: 'implementation "com.bodymeasurement:sdk:1.0.0"',
    packageName: 'com.bodymeasurement:sdk',
    docsUrl: '#android',
    description: 'Native Kotlin SDK with Coroutines support',
  },
  {
    id: 'web',
    name: 'Web SDK',
    icon: '🌐',
    language: 'JavaScript',
    status: 'production',
    installCommand: 'npm install @body-measurement/web-sdk',
    packageName: '@body-measurement/web-sdk',
    docsUrl: '#web',
    description: 'Browser SDK for web applications',
  },
  {
    id: 'api',
    name: 'REST API',
    icon: '🔌',
    language: 'Any',
    status: 'production',
    installCommand: 'curl -X POST https://api.bodymeasurement.io/v1/measurements',
    packageName: 'REST API',
    docsUrl: '#api',
    description: 'Direct API access for any programming language',
  },
];

const codeExamples: Record<Platform, string> = {
  'react-native': `import { useMeasurement } from '@body-measurement/react-native-sdk';

function MeasureScreen() {
  const { measure, loading, result } = useMeasurement({
    apiKey: 'YOUR_API_KEY',
  });

  const handleCapture = async (imageUri: string) => {
    const measurements = await measure(imageUri, {
      fitPreference: 'regular',
    });

    console.log('Size:', measurements.recommendedSize);
    console.log('Chest:', measurements.chestCircumference);
  };

  return (
    <Button onPress={() => handleCapture(photo.uri)}>
      {loading ? 'Processing...' : 'Get Measurements'}
    </Button>
  );
}`,
  flutter: `import 'package:body_measurement_sdk/body_measurement_sdk.dart';

class MeasureScreen extends StatefulWidget {
  @override
  _MeasureScreenState createState() => _MeasureScreenState();
}

class _MeasureScreenState extends State<MeasureScreen> {
  final client = BodyMeasurementClient(apiKey: 'YOUR_API_KEY');

  Future<void> measureBody(File imageFile) async {
    final result = await client.processMeasurement(
      imageFile,
      fitPreference: FitPreference.regular,
    );

    print('Size: \${result.recommendedSize}');
    print('Chest: \${result.chestCircumference} cm');
  }
}`,
  ios: `import BodyMeasurementSDK

class MeasureViewController: UIViewController {
    let client = BodyMeasurementClient(apiKey: "YOUR_API_KEY")

    func measureBody(image: UIImage) async throws {
        let result = try await client.processMeasurement(
            image: image,
            fitPreference: .regular
        )

        print("Size: \\(result.recommendedSize)")
        print("Chest: \\(result.chestCircumference) cm")
    }
}`,
  android: `import com.bodymeasurement.sdk.BodyMeasurementClient
import com.bodymeasurement.sdk.FitPreference

class MeasureActivity : AppCompatActivity() {
    private val client = BodyMeasurementClient("YOUR_API_KEY")

    suspend fun measureBody(imageFile: File) {
        val result = client.processMeasurement(
            imageFile = imageFile,
            fitPreference = FitPreference.REGULAR
        )

        Log.d("Measurement", "Size: \${result.recommendedSize}")
        Log.d("Measurement", "Chest: \${result.chestCircumference} cm")
    }
}`,
  web: `import { BodyMeasurementClient } from '@body-measurement/web-sdk';

const client = new BodyMeasurementClient({
  apiKey: 'YOUR_API_KEY',
});

async function measureFromFile(file: File) {
  const result = await client.processMeasurement(file, {
    fitPreference: 'regular',
  });

  console.log('Size:', result.recommendedSize);
  console.log('Chest:', result.chestCircumference, 'cm');
}`,
  api: `# Process body measurement with cURL
curl -X POST "https://api.bodymeasurement.io/api/v1/measurements/process-multi" \\
  -H "X-API-Key: YOUR_API_KEY" \\
  -H "Content-Type: multipart/form-data" \\
  -F "file=@body_photo.jpg" \\
  -F "fit_preference=regular"

# Response
{
  "total_people_detected": 1,
  "valid_people_count": 1,
  "measurements": [{
    "recommended_size": "M",
    "chest_circumference": 96.5,
    "waist_circumference": 82.3,
    "hip_circumference": 98.1
  }]
}`,
};

export default function SDKPage() {
  const [selectedPlatform, setSelectedPlatform] = useState<Platform>('react-native');
  const [copied, setCopied] = useState<string | null>(null);

  const brand = authHelpers.getBrand();
  const apiKey = authHelpers.getApiKey();

  const copyToClipboard = async (text: string, id: string) => {
    await navigator.clipboard.writeText(text);
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  };

  const currentPlatform = sdkPlatforms.find(p => p.id === selectedPlatform)!;

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-3xl font-bold text-gray-900">SDK Documentation</h1>
        <p className="text-gray-600 text-lg">
          Integrate body measurements into your mobile and web applications
        </p>
      </div>

      {/* Features Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { icon: Zap, label: '95%+ Accuracy', desc: 'AI-powered measurements' },
          { icon: Smartphone, label: '6 Platforms', desc: 'Native & cross-platform' },
          { icon: Shield, label: 'Production Ready', desc: 'Battle-tested SDKs' },
          { icon: Gauge, label: '<3s Processing', desc: 'Fast results' },
        ].map((feature) => (
          <div key={feature.label} className="bg-white rounded-xl border border-gray-200 p-4">
            <feature.icon className="w-8 h-8 text-indigo-600 mb-2" />
            <h3 className="font-semibold text-gray-900">{feature.label}</h3>
            <p className="text-sm text-gray-500">{feature.desc}</p>
          </div>
        ))}
      </div>

      {/* Platform Selector */}
      <div className="bg-white rounded-2xl border border-gray-200 overflow-hidden">
        <div className="border-b border-gray-200 p-4 bg-gray-50">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Choose Your Platform</h2>
          <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
            {sdkPlatforms.map((platform) => (
              <button
                key={platform.id}
                onClick={() => setSelectedPlatform(platform.id)}
                className={`p-3 rounded-xl text-center transition-all ${
                  selectedPlatform === platform.id
                    ? 'bg-indigo-600 text-white shadow-lg scale-105'
                    : 'bg-white border border-gray-200 hover:border-indigo-300 hover:bg-indigo-50'
                }`}
              >
                <div className="text-2xl mb-1">{platform.icon}</div>
                <div className={`text-xs font-medium ${
                  selectedPlatform === platform.id ? 'text-white' : 'text-gray-700'
                }`}>
                  {platform.name}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Platform Details */}
        <div className="p-6 space-y-6">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
            <div>
              <div className="flex items-center gap-3">
                <span className="text-4xl">{currentPlatform.icon}</span>
                <div>
                  <h3 className="text-2xl font-bold text-gray-900">{currentPlatform.name} SDK</h3>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="px-2 py-0.5 bg-gray-100 text-gray-700 rounded text-sm">
                      {currentPlatform.language}
                    </span>
                    <span className="px-2 py-0.5 bg-green-100 text-green-700 rounded text-sm font-medium">
                      {currentPlatform.status === 'production' ? 'Production Ready' : 'Beta'}
                    </span>
                  </div>
                </div>
              </div>
              <p className="text-gray-600 mt-2">{currentPlatform.description}</p>
            </div>
          </div>

          {/* Installation */}
          <div>
            <h4 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-2">
              Installation
            </h4>
            <div className="bg-gray-900 rounded-xl p-4 flex items-center justify-between gap-4">
              <code className="text-green-400 text-sm font-mono overflow-x-auto">
                {currentPlatform.installCommand}
              </code>
              <button
                onClick={() => copyToClipboard(currentPlatform.installCommand, 'install')}
                className="p-2 hover:bg-gray-800 rounded-lg transition-colors flex-shrink-0"
              >
                {copied === 'install' ? (
                  <Check className="w-5 h-5 text-green-400" />
                ) : (
                  <Copy className="w-5 h-5 text-gray-400" />
                )}
              </button>
            </div>
          </div>

          {/* Your API Key */}
          <div>
            <h4 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-2">
              Your API Key
            </h4>
            <div className="bg-indigo-50 border border-indigo-200 rounded-xl p-4 flex items-center justify-between gap-4">
              <code className="text-indigo-700 text-sm font-mono">
                {apiKey ? `${apiKey.slice(0, 8)}...${apiKey.slice(-8)}` : 'Not available'}
              </code>
              {apiKey && (
                <button
                  onClick={() => copyToClipboard(apiKey, 'apikey')}
                  className="p-2 hover:bg-indigo-100 rounded-lg transition-colors flex-shrink-0"
                >
                  {copied === 'apikey' ? (
                    <Check className="w-5 h-5 text-green-600" />
                  ) : (
                    <Copy className="w-5 h-5 text-indigo-600" />
                  )}
                </button>
              )}
            </div>
          </div>

          {/* Code Example */}
          <div>
            <h4 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-2">
              Quick Start Example
            </h4>
            <div className="relative">
              <div className="bg-gray-900 rounded-xl p-4 overflow-x-auto">
                <pre className="text-sm text-gray-300 font-mono whitespace-pre">
                  {codeExamples[selectedPlatform]}
                </pre>
              </div>
              <button
                onClick={() => copyToClipboard(codeExamples[selectedPlatform], 'code')}
                className="absolute top-3 right-3 p-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
              >
                {copied === 'code' ? (
                  <Check className="w-4 h-4 text-green-400" />
                ) : (
                  <Copy className="w-4 h-4 text-gray-400" />
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* API Endpoints Reference */}
      <div className="bg-white rounded-2xl border border-gray-200 p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">API Endpoints</h2>
        <div className="space-y-3">
          {[
            { method: 'POST', path: '/api/v1/measurements/process', desc: 'Single person measurement' },
            { method: 'POST', path: '/api/v1/measurements/process-multi', desc: 'Multi-person detection' },
            { method: 'POST', path: '/api/v1/products/{id}/recommend-size', desc: 'Product-specific sizing' },
            { method: 'POST', path: '/api/v1/recommend-size-bulk', desc: 'Bulk size recommendations' },
            { method: 'GET', path: '/api/v1/products', desc: 'List products with size charts' },
          ].map((endpoint) => (
            <div
              key={endpoint.path}
              className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <span className={`px-2 py-1 text-xs font-bold rounded ${
                endpoint.method === 'POST' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'
              }`}>
                {endpoint.method}
              </span>
              <code className="text-sm font-mono text-gray-700 flex-1">{endpoint.path}</code>
              <span className="text-sm text-gray-500">{endpoint.desc}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Response Schema */}
      <div className="bg-white rounded-2xl border border-gray-200 p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Measurement Response</h2>
        <p className="text-gray-600 mb-4">All SDKs return measurements in centimeters with the following structure:</p>
        <div className="bg-gray-900 rounded-xl p-4 overflow-x-auto">
          <pre className="text-sm text-gray-300 font-mono">{`{
  "person_id": 0,
  "is_valid": true,
  "gender": "female",
  "age_group": "adult",
  "demographic_label": "Adult Female",

  // Circumference measurements (cm)
  "chest_circumference": 88.5,
  "waist_circumference": 70.2,
  "hip_circumference": 95.8,
  "arm_circumference": 28.3,
  "thigh_circumference": 54.2,

  // Linear measurements (cm)
  "shoulder_width": 40.1,
  "inseam": 78.5,
  "arm_length": 56.2,
  "estimated_height_cm": 165.5,

  // Size recommendation
  "recommended_size": "S",
  "size_probabilities": {
    "XS": 0.15,
    "S": 0.65,
    "M": 0.18,
    "L": 0.02
  },

  // Confidence scores
  "detection_confidence": 0.92,
  "validation_confidence": 0.98
}`}</pre>
        </div>
      </div>

      {/* Help Section */}
      <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-2xl border border-indigo-100 p-6">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-xl bg-indigo-600 flex items-center justify-center flex-shrink-0">
            <BookOpen className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-gray-900">Need Help?</h3>
            <p className="text-gray-600 mt-1">
              Check out our full documentation or contact support for integration assistance.
            </p>
            <div className="flex gap-3 mt-4">
              <a
                href="https://docs.bodymeasurement.io"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors"
              >
                <BookOpen className="w-4 h-4" />
                Full Documentation
                <ExternalLink className="w-4 h-4" />
              </a>
              <a
                href="https://github.com/body-measurement"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 bg-gray-900 text-white rounded-lg font-medium hover:bg-gray-800 transition-colors"
              >
                GitHub Examples
                <ExternalLink className="w-4 h-4" />
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
